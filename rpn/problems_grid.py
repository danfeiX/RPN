import init_path
from rpn.grid_envs import GridGoal
import numpy as np


def grid_ballgoal1(num_tasks, num_episodes, seed=0):
    colors = ['red', 'green', 'blue', 'yellow']

    objects = []
    for c in colors:
        objects.append(('ball', {'color': c}))
        objects.append(('floor', {'color': c}))

    for c in colors:
        goal = [GridGoal('on', True, 'floor', c), GridGoal('holding', True, 'ball', c)]
        subgoals = [
            [GridGoal('holding', True, 'ball', c)],
            [GridGoal('on', True, 'floor', c)],
            [GridGoal('on', True, 'floor', c), GridGoal('holding', True, 'ball', c)],
        ]
        yield {
                  'goal': goal,
                  'subgoals': subgoals,
                  'dependencies': [],
                  'env': {'objects': objects},
                  'env_name': 'EmptyEnv'
              }, num_episodes


def grid_door(num_tasks, num_episodes, num_door, seed=0):
    npr = np.random.RandomState(seed)
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'grey']
    assert(num_door <= len(colors))

    for _ in range(num_tasks):
        task_colors = npr.choice(colors, num_door, replace=False).tolist()

        goal = []
        for tc in task_colors:
            goal.append(GridGoal('is_open', True, 'door', tc))

        subgoals = []
        for tc in task_colors:
            subgoals.append([GridGoal('is_open', True, 'door', tc)])
        subgoals.append(goal)

        dependencies = []
        objects = []
        for c in colors:
            is_open = 'random'
            if c in task_colors:
                is_open = False
            objects.append(('door', {'color': c, 'is_locked': False, 'is_open': is_open}))
            objects.append(('key', {'color': c}))

        yield {
                  'goal': goal,
                  'subgoals': subgoals,
                  'dependencies': dependencies,
                  'env': {'objects': objects},
                  'env_name': 'EmptyEnv'
              }, num_episodes


def grid_door2(num_tasks, num_episodes, seed=0):
    return grid_door(num_tasks, num_episodes, 2, seed)


def grid_door3(num_tasks, num_episodes, seed=0):
    return grid_door(num_tasks, num_episodes, 3, seed)


def grid_door4(num_tasks, num_episodes, seed=0):
    return grid_door(num_tasks, num_episodes, 4, seed)


def grid_door6(num_tasks, num_episodes, seed=0):
    return grid_door(num_tasks, num_episodes, 6, seed)


def grid_doorkey(num_tasks, num_episodes, num_door, seed=0):
    npr = np.random.RandomState(seed)
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'grey']
    assert(num_door <= len(colors))

    for _ in range(num_tasks):
        task_colors = npr.choice(colors, num_door, replace=False).tolist()
        task_locked = npr.choice([True, False], num_door).tolist()

        goal = []
        for tc in task_colors:
            goal.append(GridGoal('is_open', True, 'door', tc))

        dependencies = []
        subgoals = []
        for tc, tl in zip(task_colors, task_locked):
            if tl:
                subgoals.append([GridGoal('holding', True, 'key', tc)])
                subgoals.append([GridGoal('is_open', True, 'door', tc), GridGoal('is_locked', False, 'door', tc)])
                dependencies.append([GridGoal('is_open', True, 'door', tc), GridGoal('is_locked', False, 'door', tc)])
                dependencies.append([GridGoal('is_locked', False, 'door', tc), GridGoal('is_open', True, 'door', tc)])
            else:
                subgoals.append([GridGoal('is_open', True, 'door', tc)])
        subgoals.append(goal)

        objects = []
        for c in colors:
            is_open = 'random'
            is_locked = 'random'
            if c in task_colors:
                is_open = False
                is_locked = task_locked[task_colors.index(c)]
            objects.append(('door', {'color': c, 'is_locked': is_locked, 'is_open': is_open}))
            objects.append(('key', {'color': c}))

        yield {
                  'goal': goal,
                  'subgoals': subgoals,
                  'dependencies': dependencies,
                  'env': {'objects': objects},
                  'env_name': 'SixDoorsEnv'
              }, num_episodes


def grid_doorkey1(num_tasks, num_episodes, seed=0):
    return grid_doorkey(num_tasks, num_episodes, 1, seed)


def grid_doorkey2(num_tasks, num_episodes, seed=0):
    return grid_doorkey(num_tasks, num_episodes, 2, seed)


def grid_doorkey3(num_tasks, num_episodes, seed=0):
    return grid_doorkey(num_tasks, num_episodes, 3, seed)


def grid_doorkey4(num_tasks, num_episodes, seed=0):
    return grid_doorkey(num_tasks, num_episodes, 4, seed)


def grid_doorkey6(num_tasks, num_episodes, seed=0):
    return grid_doorkey(num_tasks, num_episodes, 6, seed)


def grid_keydoorgoal(num_tasks, num_episodes, goal_door_locked, do_goal, seed=0):
    npr = np.random.RandomState(seed=seed)
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'grey']

    for _ in range(num_tasks):
        door_colors = npr.choice(colors, size=len(colors), replace=False).tolist()
        goal_colors = door_colors
        # goal_colors = npr.choice(colors, size=len(colors), replace=False).tolist()
        room_map = {}
        for i, (dc, gc) in enumerate(zip(door_colors, goal_colors)):
            room_map[(dc, 'door')] = i
            room_map[(gc, 'floor')] = i
        idx = int(npr.randint(0, len(colors)))
        dc = door_colors[idx]
        gc = goal_colors[idx]

        objects = []
        for oc in colors:
            locked = 'random' if oc != dc else goal_door_locked
            objects.append(('door', {'color': oc, 'is_locked': locked}))
            objects.append(('key', {'color': oc}))
            objects.append(('floor', {'color': oc}))

        if do_goal:
            goal = [GridGoal('on', True, 'floor', gc)]
        else:
            goal = [GridGoal('is_open', True, 'door', dc)]

        subgoals = []
        dependencies = []
        if goal_door_locked:
            subgoals.append([GridGoal('holding', True, 'key', dc)])
            subgoals.append([GridGoal('is_open', True, 'door', dc), GridGoal('is_locked', False, 'door', dc)])
            dependencies.extend([
                (GridGoal('is_open', True, 'door', dc), GridGoal('is_locked', False, 'door', dc)),
                (GridGoal('is_locked', False, 'door', dc), GridGoal('is_open', True, 'door', dc)),
            ])
        else:
            subgoals.append([GridGoal('is_open', True, 'door', dc)])
        if do_goal:
            subgoals.append([GridGoal('on', True, 'floor', gc)])

        subgoals.append(goal)

        yield {
                  'goal': goal,
                  'subgoals': subgoals,
                  'dependencies': dependencies,
                  'env': {'objects': objects, 'room_map': room_map},
                  'env_name': 'RoomGoalEnv'
              }, num_episodes


def grid_kdg_partial(num_task, num_episodes, seed=0):
    for p in grid_keydoorgoal(num_task, num_episodes, True, False, seed):
        yield p
    for p in grid_keydoorgoal(num_task, num_episodes, False, True, seed):
        yield p


def grid_kdg_goal(num_task, num_episodes, seed=0):
    return grid_keydoorgoal(num_task, num_episodes, False, True, seed)


def grid_kdg_door(num_task, num_episodes, seed=0):
    return grid_keydoorgoal(num_task, num_episodes, True, False, seed)


def grid_kdg_full(num_task, num_episodes, seed=0):
    return grid_keydoorgoal(num_task, num_episodes, True, True, seed)


def grid_delivery1(num_tasks, num_episodes, seed=0):
    colors = ['red', 'green', 'blue', 'yellow']
    objects = []
    for c in colors:
        objects.append(('door', {'color': c, 'is_locked': False}))
        objects.append(('key', {'color': c}))
        objects.append(('floor', {'color': c}))
        objects.append(('ball', {'color': c}))

    nt = 0
    for ball_c in colors:
        for target_c in colors:
            goal = [['holding', True, 'ball', ball_c], ['on', True, 'floor', target_c]]
            yield {
                      'instructions': (ball_c, target_c),
                      'goal': goal,
                      'env': {'objects': objects},
                      'env_name': 'DeliveryEnv'
                  }, num_episodes
            nt += 1
            if nt == num_tasks:
                break


def main():
    # a = pb_block_ntp('ntp_specs/stack/stack_2000.json', 10, 10)
    # print
    pgen = list(pb_cook_meal_3i_2d_2m_shuffle(None, 0, -1, 10, 0))
    print(len(pgen))
    for p, num_episode in pgen:
        print(p['goal'])
        # for sg in p['subgoals']:
        #     print(sg)
        # print('===============')
    # for p in grid_doorkey(10, 1, 2):
    #     print(p)

if __name__ == '__main__':
    main()

