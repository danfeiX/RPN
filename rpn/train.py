#!/usr/bin/env python
from __future__ import print_function
import init_path

import argparse
import sys
import shutil

from torch.optim import Adam
from rpn.utils.log_utils import StatsSummarizer, PrintLogger
from rpn.data_utils import *
from torch.utils.data.dataloader import DataLoader
from rpn.utils.config import dict_to_namedtuple
from rpn.rpn_pb import *
from rpn.rpn_grid import *
torch.manual_seed(0)
np.random.seed(0)


def get_ckpt_path(exp_dir, ep, global_step):
    if not osp.exists(exp_dir):
        os.mkdir(exp_dir)
    return osp.join(exp_dir, 'ckpt_ep%i.pt' % ep)


def get_exp_dir(args):
    exp_dir = osp.join(args.ckpt_root, args.exp)
    if args.run is None:
        for ri in range(1000):
            sdir = osp.join(exp_dir, 'run%i/' % ri)
            if not osp.exists(sdir):
                os.makedirs(sdir)
                break
    else:
        sdir = osp.join(exp_dir, args.run)
        if osp.exists(sdir):
            c = input('Directory %s exists, delete?(y/n)' % sdir)
            if c == 'y':
                print('delete %s' % sdir)
                shutil.rmtree(sdir)
            os.makedirs(sdir, exist_ok=True)
    ldir = osp.join(sdir, 'log/')
    os.makedirs(ldir, exist_ok=True)
    return sdir, ldir


def train_epoch(net, optimizer, scheduler, loader, summ, global_step):
    net.train()
    summ.timer_tic('data')
    summ.timer_tic('iter')
    for batch_idx, batch in enumerate(loader):
        summ.timer_toc('data')
        summ.timer_tic('train')
        batch = tu.batch_to_cuda(batch)
        outputs = net.forward_batch(batch)
        losses = net.compute_losses(outputs, batch)
        total_loss = torch.sum(torch.stack(tuple(losses.values())))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        global_step += 1
        summ.timer_toc('train')

        summ.timer_toc('iter')
        summ.timer_tic('iter')

        summ.timer_tic('log')
        if global_step % 10 == 0:
            net.log_losses(losses, summ, global_step, prefix='train/loss/')
            net.log_outputs(outputs, batch, summ, global_step, prefix='train/')
        summ.timer_toc('log')
        if global_step % 100 == 0:
            print(summ.summarize_all_stats(prefix='train/', last_n=20))
            print(summ.summarize_timers(100))
            print(loader.dataset.timers)
        summ.timer_tic('data')
    summ.timer_toc('data')
    return global_step


def eval_epoch(net, loader, global_step, summarizer):
    net.eval()
    num_batches = 0
    for batch_idx, batch in enumerate(loader):
        batch = tu.batch_to_cuda(batch)
        outputs = net.forward_batch(batch)
        losses = net.compute_losses(outputs, batch)
        net.log_losses(losses, summarizer, global_step, prefix='eval/loss/')
        net.log_outputs(outputs, batch, summarizer, global_step + batch_idx, prefix='eval/')
        num_batches += 1
    print(summarizer.summarize_all_stats(prefix='eval/', last_n=num_batches))


def config_model(net_cf, meta):
    net_cf = net_cf.copy()
    model_class = eval(net_cf.pop('model'))
    cf = {}
    for k, v in net_cf.items():
        if isinstance(v, str) and v.startswith('eval:'):
            cf[k] = eval(v.split(':')[1])
        else:
            cf[k] = v
    print(model_class)
    print(cf)
    return model_class(**cf)


def train_net(args):
    exp_dir, log_dir = get_exp_dir(args)
    logger = PrintLogger(osp.join(log_dir, 'log.txt'))
    sys.stdout = logger
    sys.stderr = logger

    print(args)
    meta = dict_to_namedtuple(json.load(open(args.dataset + '.meta', 'r')))
    config = json.load(open(args.config, 'r'))
    # dsc = DemoDataset
    dsc = eval(config['data']['loader'])
    collate_fn = lambda c: collate_samples(c, concatenate=config['data']['collate_cat'],
                                           exclude_keys=config['data'].get('exclude_keys', ()))

    db = dsc.load(args.dataset, **config['data']['dataset'])
    dl = DataLoader(
        db, batch_size=args.batch_size, collate_fn=collate_fn,
        num_workers=args.num_workers,
        sampler=get_sampler_without_eos(dataset=db, random=True),
        # shuffle=True
    )

    # evaluation set
    edl = None
    if args.testset is not None:
        edb = dsc.load(args.testset, **config['data']['dataset'])
        edl = DataLoader(
            edb, batch_size=args.eval_batch_size, collate_fn=collate_fn,
            num_workers=args.num_workers,
            sampler=get_sampler_without_eos(dataset=edb, random=False),
        )

    net = tu.safe_cuda(config_model(config['net'], meta))

    if args.mpc_path is not None:
        ckpt = tu.load_checkpoint(args.mpc_path)
        vmpc = eval(ckpt['model_class'])(**ckpt['config'])
        vmpc.load_state_dict(ckpt['model'])
        net.vmpc = tu.safe_cuda(vmpc)
    print(net)
    print(db)

    global_step = 0
    lr_scheduler = None  # TODO LOW
    summarizer = StatsSummarizer(log_dir=log_dir)

    optimizer = Adam(net.parameters(), lr=1e-3)
    for ep in range(args.num_epoch):
        global_step = train_epoch(net, optimizer, lr_scheduler, dl, summarizer, global_step)
        if ep % args.save_freq == 0 and ep > 0:
            ckpt_path = get_ckpt_path(exp_dir, ep, global_step)
            tu.save_checkpoint(ckpt_path, net, optimizer, config=net.config)

        if ep % args.eval_freq == 0 and ep > 0 and edl is not None:
            eval_epoch(net, edl, global_step, summarizer)


def parse_args():
    parser = argparse.ArgumentParser(description='Data generation in gym')
    parser.add_argument('--num_epoch', default=100000, type=int)
    parser.add_argument('--dataset', default='data.npy', type=str)
    parser.add_argument('--save_freq', default=10, type=int, help='save frequency in epoch')
    parser.add_argument('--ckpt_root', default='checkpoints/')
    parser.add_argument('--exp', default='exp')
    parser.add_argument('--restore_path', default='checkpoints/net.pt')
    parser.add_argument('--mpc_path', type=str, default=None)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--testset', type=str, default=None)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--run', default=None, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
