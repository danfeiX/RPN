import init_path

import argparse
from rpn.data_utils import *
from torch.utils.data.dataloader import DataLoader
from rpn.tracer import parse_plan
from copy import deepcopy
from rpn.rpn_grid import *
from rpn.utils.log_utils import StatsSummarizer
import rpn.problems_grid as problems
from rpn.grid_envs import *
from collections import defaultdict
import json

np.random.seed(0)


def same_sg_list(sgl1, sgl2):
    if len(sgl1) != len(sgl2):
        return False
    for s1, s2 in zip(sgl1, sgl2):
        if s1 != s2:
            return False
    return True


def rollout_policy(env, goal, net, info, macro_action=False, replan=False, max_steps=100, display=False):
    env.reset()
    success = False
    step_i = 0
    prev_subgoals = []
    verbose = False
    if verbose:
        print('goal: ', goal)
    goal_ss = env.serialize_symbolic_state(env.goal_to_symbolic_state(goal))
    goal_ss_mask = env.goal_mask(goal)
    batch = {
        'goal': np.expand_dims(goal_ss, 0),
        'goal_mask': np.expand_dims(goal_ss_mask, 0)
    }
    while step_i < max_steps:
        if display:
            env.render('human')
        batch['states'] = env.serialize_object_state()[None, ...].astype(np.float32)
        outputs = net.forward_batch(tu.batch_to_cuda(tu.batch_to_tensor(batch)))
        if outputs is None:
            info['error']['NETWORK'] += 1
            return success, step_i

        if 'subgoal' in outputs:
            if outputs['subgoal'] is None:
                info['error'][outputs['ret']] += 1
                return success, step_i
            sg = tu.to_numpy(outputs['subgoal'][0])
            subgoals = env.deserialize_goals(sg[:, :, 1], 1 - sg[:, :, 2])

            if env.is_success(prev_subgoals) and same_sg_list(prev_subgoals, subgoals) and verbose:
                print('==================achieved subgoal but did not switch to new goal')

            if not env.is_success(prev_subgoals) and not same_sg_list(prev_subgoals, subgoals) and verbose:
                print('==================switched to a new goal without finishing the last')

            if not same_sg_list(prev_subgoals, subgoals) and verbose:
                print('prev: ', prev_subgoals, 'new: ', subgoals)
                prev_subgoals = subgoals

            if macro_action:
                for action_name, ret in goal_to_macro(env, subgoals, verbose=True):
                    if action_name is None:
                        info['error'][ret] += 1
                        return success, step_i
                    _, _, _, stuck, _ = env.step(env.actions[action_name])
                    if replan:
                        break
                    # if stuck:
                    #     break

        if not macro_action:
            action_idx = tu.to_numpy(outputs['action_preds']).argmax(axis=1)[0]
            action_name = env.action_list[action_idx]
            env.step(env.actions[action_name])

        if env.is_success(goal):
            success = True
            break
        step_i += 1
    if display:
        env.render('human')

    if not success:
        info['error']['FAILURE'] += 1
    return success, step_i


def eval_net(args):
    ckpt = tu.load_checkpoint(args.restore_path)
    net = eval(ckpt['model_class'])(**ckpt['config'])
    net.load_state_dict(ckpt['model'])
    net = tu.safe_cuda(net)
    net.eval()
    # if args.dataset is not None:
    # eval_db(net, args)
    # else:
    eval_policy(net, args)


def eval_db(net, args):
    config = json.load(open(args.config, 'r'))
    # dsc = DemoDataset
    dsc = eval(config['data']['loader'])
    collate_fn = lambda c: collate_samples(c, concatenate=config['data']['collate_cat'])

    db = dsc.load(args.dataset, **config['data']['dataset'])
    dl = DataLoader(
        db, batch_size=128, collate_fn=collate_fn,
        sampler=get_sampler_without_eos(dataset=db, random=False),
    )

    pgen = getattr(problems, args.problem)(args.num_tasks, args.num_episodes, args.seed)
    problem, num_episodes = next(pgen)
    env = eval(problem['env_name'])(**problem['env'])

    summ = StatsSummarizer()
    for _ in range(1):
        for batch_idx, batch in enumerate(dl):
            batch = tu.batch_to_cuda(batch)
            outputs = net.forward_batch(batch)
            net.log_outputs(outputs, batch, summ, batch_idx, prefix='eval/')
            net.debug(outputs, batch, env)
    print(summ.summarize_all_stats(prefix='eval/', last_n=0))


def get_goal_state(env, problem):
    env.reset()
    goal = problem['goal']
    demo_seq = []
    symbol_seq = []
    traces, extended_traces = parse_plan(deepcopy(problem['subgoals']))
    success = False
    for trace_index, trace in enumerate(traces):
        if env.is_success(trace[-1]):
            continue

        for action_name, ret in goal_to_macro(env, trace[-1]):
            if success:
                raise ValueError('suboptimal solution')
            if action_name is None:
                assert(ret == 'MACRO_FAILURE')
                print('failed')
                return None
            demo_seq.append(env.serialize_object_state())
            symbol_seq.append(env.serialize_symbolic_state())
            env.step(env.actions[action_name])
            success = env.is_success(goal)
        # print(trace[-1])
        assert(env.is_success(trace[-1]))

    assert success
    assert(env.is_success(goal))
    assert(env.is_success(traces[-1][-1]))
    return success

    # demo_seq.append(env.serialize_object_state())
    # symbol_seq.append(env.serialize_symbolic_state())
    # demo_seq = np.stack(demo_seq).astype(np.float32)
    # symbol_seq = np.stack(symbol_seq).astype(np.float32)
    # return {
    #     'goal_states': demo_seq[[-1]],
    #     'goal_mask': env.goal_mask(goal)[None, ...].astype(np.float32),
    #     'goal_symbol': symbol_seq[[-1]],
    #     'goal': symbol_seq[[-1]],
    #     'demo_sequence': demo_seq,
    #     'symbol_sequence': symbol_seq,
    # }


def inspect(env, instructions, batch, goal, net):
    env.reset()
    # env.render('human')
    net.inspect(tu.batch_to_cuda(tu.batch_to_tensor(batch)), env)
    return True, 0


def eval_policy(net, args):
    net.policy()
    pgen = getattr(problems, args.problem)(args.num_tasks, args.num_episodes, args.seed)

    n_success = 0
    num_ep = 0
    seed_counter = args.seed
    info = dict()
    info['error'] = defaultdict(int)
    info['success'] = []
    info['completion'] = []
    info['num_steps'] = []
    info['args'] = vars(args)
    for problem, num_episodes in pgen:
        env = eval(problem['env_name'])(**problem['env'])
        for ep_i in range(num_episodes):
            goal_state = None
            # get demonstration
            while goal_state is None:
                seed_counter += 1
                env.seed(seed_counter)
                goal_state = get_goal_state(env, problem)
            env.seed(seed_counter)

            if args.inspect:
                success, num_step = inspect(env, problem['instructions'], goal_state, problem['goal'], net)
            else:
                success, num_step = rollout_policy(
                    env, problem['goal'], net, info=info, macro_action=args.macro_action,
                    max_steps=100, display=args.display, replan=args.replan
                )
            info['success'].append(success)
            info['num_steps'].append(num_step)
            num_ep += 1
            n_success += int(success)
            print('%s/%s, %s' % (n_success, num_ep, num_step))
    log_file = args.restore_path + '-' + args.problem + '-s' + str(args.seed) + '-' + str(args.replan) + '.json'
    json.dump(info, open(log_file, 'w+'), indent=4)
    print('log written to %s' % log_file)


def parse_args():
    parser = argparse.ArgumentParser(description='Data generation in gym')
    parser.add_argument('--restore_path', default='checkpoints/net.pt')
    parser.add_argument('--display', default=False, action='store_true')
    parser.add_argument('--teleport', default=False, action='store_true')
    parser.add_argument('--task_file', default=None, type=str)
    parser.add_argument('--num_tasks', default=-1, type=int)
    parser.add_argument('--num_episodes', default=10, type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--swap_goal', action='store_true', default=False)
    parser.add_argument('--problem', type=str, default=None, required=True)
    parser.add_argument('--macro_action', default=False, action='store_true')
    parser.add_argument('--inspect', default=False, action='store_true')
    parser.add_argument('--replan', default=False, action='store_true')
    parser.add_argument('--config', type=str)
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    eval_net(args)


if __name__ == '__main__':
    main()
