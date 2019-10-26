import init_path

import numpy as np
from multiprocessing import Pool
import os
import os.path as osp

import argparse
import json
from rpn.data_utils import BasicDataset
from rpn.db import ReplayBuffer
from third_party.pybullet.utils.pybullet_tools.perception import Camera
from third_party.pybullet.utils.pybullet_tools.utils import connect, disconnect
from rpn.bullet_envs import *
from rpn.plan_utils import goal_to_motion_plan
from rpn.env_utils import load_world, world_saved, pb_session, set_rendering_pose
from rpn.env_utils import World
from rpn.problems_pb import factory
from rpn.tracer import parse_plan, serialize_trace
from copy import deepcopy
from rpn.utils.timer import Timer
from math import ceil

SEED_INCREMENT = 10000


def record_state(env, rb, task_id, camera):
    data = dict(
        symbolic_state=env.serialized_symbolic_state,
        gt_state=env.serialized_gt_state,
        object_ids=np.array(env.objects.ids),
        object_type_indices=np.array(env.objects.object_type_indices),
        num_object_types=np.array([len(env.objects.all_types)]),
        task_id=np.array([task_id]),
        image_crops=camera.get_crops(env.objects.ids, expand_ratio=1.1)
    )
    # rgb, depth, obj_seg, link_seg = camera.capture_frame()
    # data['image'] = rgb
    # data['segmentation'] = obj_seg.astype(np.int8)
    # data['bbox'] = bbox
    # bbox = get_bbox2d_from_segmentation(obj_seg, env.objects.ids)
    # data['image_crops'] = crop_pad_resize(rgb, bbox[:, 1:], 24, expand_ratio=1.1)
    # box_im = draw_boxes(rgb, bbox[:, 1:])
    # import matplotlib.pyplot as plt
    # for crop in data['image_crops']:
    #     plt.imshow(crop)
    #     plt.show()

    rb.append(**data)


def record_action(env, rb, action_name, action_args, control):
    serialized_action, cont_args = env.serialize_action(action_name, action_args)
    rb.append(
        actions=serialized_action,
        # action_cont_args=cont_args,
        # controls=control
    )


def rollout_plan(env, problem, camera, task_id, replay_buffer,
                 step_command=False, time_step=0.0, teleport=False, motion_resolution=0.05):
    env.reset()
    n_step = 0
    success = False
    goal = problem['goal']
    traces, extended_traces = parse_plan(deepcopy(problem['subgoals']))

    verbose = False
    if verbose:
        for sg in problem['subgoals']:
            print(sg)
        print('#########################')
        for trace in extended_traces:
            for ts in trace:
                print(ts)
            print(' =================== ')
        print('#########################')
        for trace in traces:
            for ts in trace:
                print(ts)
            print('====================')
    # print(task_id, goal)
    dummy_args = np.zeros(env.objects.robot_dof)
    for trace_index, trace in enumerate(traces):
        # print(trace[-1])
        if env.is_success(trace[-1]):
            continue
        if success:
            raise ValueError('suboptimal solution')

        name = None
        args = None
        # print(trace[-1])
        for name, args, command, ret in goal_to_motion_plan(
                trace[-1], env, teleport=teleport, motion_resolution=motion_resolution):
            # print(name, args, ret)
            # input('step?')
            if step_command:
                pass
                # record_state(env, replay_buffer, task_id, camera)
                # replay_buffer.append(**serialize_trace(env, extended_traces[trace_index], problem['dependencies']))
                # for control, is_last in env.step_command((name, args, command), time_step=time_step):
                #     record_action(env, replay_buffer, name, args, control)
                #     if not is_last:
                #         record_state(env, replay_buffer, task_id, camera)
                #         replay_buffer.append(
                #             **serialize_trace(env, extended_traces[trace_index], problem['dependencies']))
                #     n_step += 1
                    # print(conf)
            else:
                # record state and action to be applied
                record_state(env, replay_buffer, task_id, camera)
                replay_buffer.append(**serialize_trace(env, extended_traces[trace_index], problem['dependencies']))
                integrity = env.execute_command((name, args, command), time_step=time_step)
                if not integrity:
                    return False

                record_action(env, replay_buffer, name, args, dummy_args)
                n_step += 1

                if verbose:
                    if name == 'pick':
                        print(name, env.objects.name(args[0]))
                    elif name == 'place':
                        print(name, env.objects.name(args[1]))
                    else:
                        print(name, args)

        success = env.is_success(goal)
        # print(trace[-1])
        assert(env.is_success(trace[-1]))
            # print(conf)
    # record the last state with dummy action
    # input('step?')
    assert success
    # input()
    assert(env.is_success(goal))
    assert(env.is_success(traces[-1][-1]))

    record_state(env, replay_buffer, task_id, camera)
    replay_buffer.append(**serialize_trace(env, extended_traces[-1], problem['dependencies']))
    record_action(env, replay_buffer, 'noop', [env.objects.ids[0], [(0,), (0,)]], dummy_args)
    print(n_step)
    return True


def create_dataset_problem(rb, problem, task_id, display=False, teleport=False, motion_resolution=0.05):
    # disconnect()
    camera = Camera(240, 320)
    info = None
    success = False
    while not success:
        with pb_session(use_gui=False):
            world = load_world(problem, World(problem['object_types']))
            env = eval(problem['env_name'])(world)
            info = env.info
            if display:
                with world_saved():
                    disconnect()
                    connect(use_gui=True)
                    world = load_world(problem, World(problem['object_types']))
                    env = eval(problem['env_name'])(world)

            set_rendering_pose(camera)
            srb = rb.get_empty_buffer()
            if not rollout_plan(env, problem, camera, task_id, srb,
                                step_command=False, teleport=teleport,
                                motion_resolution=motion_resolution):
                print('failed')
                continue
            rb.append(**srb.data)
        success = True

    return info


def new_rb():
    keys = [
        'gt_state',
        'task_id',
        'object_ids',
        'object_type_indices',
        'num_object_types',
        'symbolic_state',
        'actions',
        # 'action_cont_args',
        # 'controls',
        'goal_trace',
        'goal_mask_trace',
        'satisfied_trace',
        'reachable_trace',
        'focus_trace',
        'dependency_trace',
        'num_goal_entities',
        'image_crops'
    ]
    rb = ReplayBuffer(keys)
    return rb


def create_dataset_chunk(problems, filename, display, teleport, seed):
    np.random.seed(seed)
    print('seed: %i,' % seed)
    rb = new_rb()
    info = None
    timer = Timer()
    for pi, (problem, task_id) in enumerate(problems):
        timer.tic()
        info = create_dataset_problem(rb, problem, task_id, display=display, teleport=teleport)
        timer.toc()
        print('%i/%i, %i===============================================' % (pi, len(problems), task_id))
        if pi % 10 == 0:
            print('Going to take %f minutes to finish' % (timer.average_time * (len(problems) - pi) / 60))
    BasicDataset(rb.contiguous(n_level=2)).dump(filename)
    return info


def create_dataset_chunk_mp(args):
    return create_dataset_chunk(*args)


def create_dataset(problems, filename, display, teleport, seed):
    info = create_dataset_chunk(problems, filename, display, teleport, seed)
    json.dump(info, open(filename + '.meta', 'w+'))


def create_dataset_mp(problems, args):
    assert(args.dataset.endswith('.group'))
    if not osp.exists(args.dataset):
        os.mkdir(args.dataset)

    chunk_size = ceil(len(problems) / args.num_chunks)
    assert(chunk_size * args.num_chunks >= len(problems))

    print('splitting %i tasks into %i chunks of %i episodes ' % (len(problems), args.num_chunks, chunk_size))
    chunks = []
    for i in range(0, len(problems), chunk_size):
        chunks.append(problems[i:i + chunk_size])

    pool = Pool(args.num_workers)
    jobs = []
    ext = 'h5p'

    for i, chunk in enumerate(chunks):
        fn = osp.join(args.dataset, 'chunk_%04d.%s' % (i, ext))
        jobs.append((chunk, fn, args.display, args.teleport, args.seed + i * SEED_INCREMENT))

    infos = pool.map(create_dataset_chunk_mp, jobs)
    json.dump(infos[0], open(args.dataset + '.meta', 'w+'))


def parse_args():
    parser = argparse.ArgumentParser(description='Data generation for Kitchen3D')
    parser.add_argument('--num_episodes', default=2, type=int)
    parser.add_argument('--dataset', default='data.npy', type=str)
    parser.add_argument('--display', default=False, action='store_true')
    parser.add_argument('--teleport', default=False, action='store_true')
    parser.add_argument('--problem', default=None, type=str, required=True)
    parser.add_argument('--task_file', default=None, type=str)
    parser.add_argument('--num_tasks', default=-1, type=int)
    parser.add_argument('--start_task', default=0, type=int)
    parser.add_argument('--sample_tasks', default=-1, type=int)
    parser.add_argument('--num_chunks', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    pgen = factory(
        args.problem,
        task_file=args.task_file,
        start_task=args.start_task,
        num_tasks=args.num_tasks,
        num_episodes=args.num_episodes,
    )
    problems = list(pgen)
    if args.sample_tasks >= 0:
        # randomly sample a subset of tasks
        np.random.seed(0)
        inds = np.arange(len(problems))
        np.random.shuffle(inds)
        problems = [problems[i] for i in inds[:args.sample_tasks]]
    print('#problems=%i' % len(problems))

    if args.num_workers > 0:
        create_dataset_mp(problems, args)
    else:
        create_dataset(problems, args.dataset, display=args.display, teleport=args.teleport, seed=args.seed)


if __name__ == '__main__':
    main()
