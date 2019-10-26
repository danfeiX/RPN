import init_path
from rpn.grid_envs import follow_instructions, EmptyEnv, RoomGoalEnv, DeliveryEnv, SixDoorsEnv, goal_to_macro
import rpn.problems_grid as problems

import numpy as np
import time
import argparse
import json
from rpn.data_utils import BasicDataset
from rpn.db import ReplayBuffer
from rpn.tracer import parse_plan, serialize_trace
from copy import deepcopy
np.random.seed(0)
SEED_INCREMENT = 10000
DEFAULT_SEED = 0


def add_rb(env, action_name, state, task_id, goal, rb):
    rb.append(
        grid_state=env.serialize_grid_state(),
        task_id=np.array([task_id]),
        actions=np.array([env.actions[action_name].value]),
        agent_view=state['image'],
        object_state=env.serialize_object_state(flat=False),
        object_state_flat=env.serialize_object_state(flat=True),
        symbolic_state=env.serialize_symbolic_state(),
        goal_mask=env.goal_mask(goal)
    )


def add_trace(env, extended_trace, external_deps, rb):
    rb.append(**serialize_trace(env, extended_trace, external_deps))


def rollout_plan(env, problem, state, task_id, rb, display=False):
    goal = problem['goal']
    sym_state = env.symbolic_state()
    success = False

    traces, extended_traces = parse_plan(deepcopy(problem['subgoals']))
    # for e in extended_traces:
    #     for s in e:
    #         print(s)
    # for s in extended_traces[trace_index]:
    #     print(s)
    # print('-------')
    for trace_index, trace in enumerate(traces):
        if env.is_success(trace[-1]):
            continue

        for action_name, ret in goal_to_macro(env, trace[-1]):
            if success:
                raise ValueError('suboptimal solution')
            if action_name is None:
                assert(ret == 'MACRO_FAILURE')
                print('failed')
                return False
            add_rb(env, action_name, state, task_id, goal, rb)
            add_trace(env, extended_traces[trace_index], problem['dependencies'], rb)
            state, _, _, _, _ = env.step(env.actions[action_name])
            success = env.is_success(goal)

            if display:
                for _ in range(3):
                    env.render('human')
                for i, (otype, ocolor, ss) in enumerate(env.symbolic_state()):
                    for k, v in ss.items():
                        if sym_state[i][2][k] != v:
                            print(otype, ocolor, k, sym_state[i][2][k], v)
                sym_state = env.symbolic_state()
                time.sleep(1)
        assert(env.is_success(trace[-1]))

    assert success
    assert(env.is_success(goal))
    assert(env.is_success(traces[-1][-1]))

    add_rb(env, 'done', state, task_id, goal, rb)
    add_trace(env, extended_traces[-1], problem['dependencies'], rb)

    if display:
        env.render('human')
        time.sleep(1)
    return True


def create_dataset(problem_gen, filename, seed, display=False):
    rb = ReplayBuffer([
        'grid_state',
        'agent_view',
        'task_id',
        'actions',
        'object_state',
        'object_state_flat',
        'symbolic_state',
        'goal_mask',
        'goal_trace',
        'satisfied_trace',
        'reachable_trace',
        'goal_mask_trace',
        'focus_trace',
        'dependency_trace',
        'num_goal_entities'
    ])
    env = None
    for pi, (problem, num_episode) in enumerate(problem_gen):
        seed += SEED_INCREMENT
        num_success = 0
        # env = EmptyEnv(**problem['env'])
        env = eval(problem['env_name'])(**problem['env'])
        env.seed(seed)
        while num_success < num_episode:
            state = env.reset()
            if display:
                env.render()
            srb = rb.get_empty_buffer()
            if not rollout_plan(env, problem, state, pi, srb, display=display):
                continue
            assert(env.is_success(problem['goal']))
            rb.append(**srb.data)
            num_success += 1
            print('%i/%i, %i, %i===============================================' % (num_success, num_episode, pi, seed))

    BasicDataset(rb.contiguous(n_level=2)).dump(filename)
    json.dump(env.info, open(filename + '.meta', 'w+'))


def parse_args():
    parser = argparse.ArgumentParser(description='Data generation in gym')
    parser.add_argument('--num_episodes', default=2, type=int)
    parser.add_argument('--dataset', default='data.npy', type=str)
    parser.add_argument('--display', default=False, action='store_true')
    parser.add_argument('--task_file', default=None, type=str)
    parser.add_argument('--num_tasks', default=-1, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--problem', default=None, type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    pgen = getattr(problems, args.problem)(args.num_tasks, args.num_episodes, args.seed)

    create_dataset(pgen, args.dataset, seed=args.seed if args.seed is not None else DEFAULT_SEED, display=args.display)


if __name__ == '__main__':
    main()
