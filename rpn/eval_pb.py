import init_path

import numpy as np
import argparse
import json
from rpn.utils.log_utils import StatsSummarizer
from rpn.env_utils import World
from rpn.bullet_envs import *
from rpn.problems_pb import factory
from rpn.env_utils import pb_session, load_world, set_rendering_pose
from rpn.plan_utils import goal_to_motion_plan
from rpn.rpn_pb import *
from rpn.data_utils import collate_samples, get_sampler_without_eos
from torch.utils.data import DataLoader
from third_party.pybullet.utils.pybullet_tools.perception import Camera
from collections import defaultdict
# np.random.seed(0)


def rollout_policy(env, goal, net, info, max_steps=100, verbose=False):
    env.reset()

    camera = Camera(240, 320)
    set_rendering_pose(camera)

    success = False
    step_i = 0
    if verbose:
        print('goal: ', goal)
    goal_ss = env.serialize_symbolic_state(env.goal_to_symbolic_state(goal))
    goal_ss_mask = env.goal_mask(goal)
    batch = {
        'goal': np.expand_dims(goal_ss, 0),
        'goal_mask': np.expand_dims(goal_ss_mask, 0)
    }
    # num_entities = goal_ss.shape[0]
    # batch['graphs'] = np.expand_dims(1 - np.eye(num_entities, dtype=np.float32), 0)
    if hasattr(net, 'env'):
        net.env = env
        net.verbose = verbose
    n_completed = 0
    for step_i in range(max_steps):
        success = env.is_success(goal)
        n_completed = env.num_success_goal(goal)
        if success or step_i >= max_steps:
            break

        # construct low-dimensional inputs
        # object_states = env.serialized_gt_state
        # num_objects = object_states.shape[0]
        # full_edges = construct_full_graph(num_objects, self_connection=True)[1]
        # object_type_indices = np.array(env.objects.object_type_indices)
        # num_types = len(env.objects.all_types)
        # states, entity_states = make_bullet_gt_input(
        #     object_states, env.serialized_symbolic_state, full_edges, object_type_indices, num_types)
        #
        # batch['states'] = states[None, ...]
        # batch['entity_states'] = entity_states[None, ...]

        # construct visual inputs
        crops = camera.get_crops(env.objects.ids, expand_ratio=1.1)
        crops = crops[None, ...].astype(np.float32) / 255.
        batch['states'] = crops.transpose((0, 1, 4, 2, 3))
        outputs = net.forward_batch(tu.batch_to_cuda(tu.batch_to_tensor(batch)))

        if 'subgoal' in outputs:
            if outputs['subgoal'] is None:
                info['error'][outputs['ret']] += 1
                info['completion'].append([n_completed, len(goal)])
                return success, step_i
            sg = tu.to_numpy(outputs['subgoal'][0])
            subgoals = env.deserialize_goals(sg[:, :, 1], 1 - sg[:, :, 2])
            # print('sg: ', subgoals)

            for action_name, action_args, action_plan, ret in goal_to_motion_plan(subgoals, env, False):
                if action_name is None:
                    info['error'][ret] += 1
                    info['completion'].append([n_completed, len(goal)])
                    return success, step_i
                if verbose:
                    if action_name == 'pick':
                        print(action_name, env.objects.name(action_args[0]))
                    elif action_name == 'place':
                        print(action_name, env.objects.name(action_args[1]))
                    else:
                        print(action_name, action_args)
                env.execute_command((action_name, action_args, action_plan), time_step=0)
        else:
            raise NotImplementedError
    if not success:
        info['error']['FAILED'] += 1
    info['completion'].append([n_completed, len(goal)])
    return success, step_i


def rollout_plan(env, commands):
    env.reset()
    for name, args, command in commands:
        env.execute_command((name, args, command), time_step=0.001)


def eval_net(args):
    ckpt = tu.load_checkpoint(args.restore_path)
    net = eval(ckpt['model_class'])(**ckpt['config'])
    if args.mpc_path is not None:
        mpc_ckpt = tu.load_checkpoint(args.mpc_path)
        mpc = eval(mpc_ckpt['model_class'])(**ckpt['config'])
        mpc.policy()
        net.vmpc = mpc

    net.load_state_dict(ckpt['model'])
    net = tu.safe_cuda(net)
    net.eval()
    if args.db:
        eval_db(net, args)
    else:
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

    pgen = factory(
        args.problem,
        task_file=args.task_file,
        start_task=args.start_task,
        num_tasks=args.num_tasks,
        num_episodes=args.num_episodes,
    )
    problem, num_episodes = next(pgen)
    with pb_session(use_gui=False):
        world = load_world(problem, World(problem['object_types']))
        env = eval(problem['env_name'])(world)

        summ = StatsSummarizer()
        for _ in range(1):
            for batch_idx, batch in enumerate(dl):
                batch = tu.batch_to_cuda(batch)
                outputs = net.forward_batch(batch)
                net.log_outputs(outputs, batch, summ, batch_idx, prefix='eval/')
                net.debug(outputs, batch, env)
        print(summ.summarize_all_stats(prefix='eval/', last_n=0))


def eval_policy(net, args):
    net.policy()
    if args.num_eval > 0:
        all_problems = []
        count = args.seed * 10000
        while len(all_problems) < args.num_eval:
            pgen = factory(
                args.problem,
                task_file=args.task_file,
                start_task=0,
                num_tasks=-1,
                num_episodes=1,
                seed=count
            )
            print('seed: ', count)
            count += 1
            for p in pgen:
                all_problems.append(p)
                if len(all_problems) >= args.num_eval:
                    break
    else:
        pgen = factory(
            args.problem,
            task_file=args.task_file,
            start_task=args.start_task,
            num_tasks=args.num_tasks,
            num_episodes=args.num_episodes,
        )
        all_problems = list(pgen)
    n_success = 0
    num_ep = 0
    info = dict()
    info['error'] = defaultdict(int)
    info['success'] = []
    info['completion'] = []
    info['num_steps'] = []
    info['args'] = vars(args)
    print('num_problems: ', len(all_problems))

    for problem, ti in all_problems:
        with pb_session(use_gui=args.display):
            world = load_world(problem, World(problem['object_types']))
            env = eval(problem['env_name'])(world)
            success, num_step = rollout_policy(env, problem['goal'], net, info, max_steps=problem['max_steps'],
                                               verbose=args.verbose)
            num_ep += 1
        n_success += int(success)
        info['success'].append(success)
        info['num_steps'].append(num_step)
        print('%s/%s, %s' % (n_success, num_ep, num_step))
    print(info['error'])

    log_file = args.log_file
    if log_file is None:
        log_file = args.restore_path + '-' + args.problem + '-s' + str(args.seed) + '.json'
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
    parser.add_argument('--start_task', default=0, type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--problem', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_eval', type=int, default=-1)
    parser.add_argument('--mpc_path', default=None, type=str)
    parser.add_argument('--db', default=False, action='store_true')
    parser.add_argument('--config', type=str)
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--verbose', default=False, action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    eval_net(args)


if __name__ == '__main__':
    main()
