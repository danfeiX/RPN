import init_path
from rpn.env_utils import URDFS
from rpn.bullet_task_utils import PBGoal
import json
import numpy as np
from itertools import permutations


def factory(name, **kwargs):
    if name.startswith('pb_block_ntp'):
        return pb_block_ntp(**kwargs)
    if name.startswith('pb_cook_meal_'):
        spec = name.split('pb_cook_meal_')[1]
        n_ing, n_dish, n_max, shuffle = spec.split('_')
        assert(shuffle in ['iter', 'shuffle'])
        kwargs['shuffle'] = shuffle == 'shuffle'
        kwargs['num_ingredients'] = int(n_ing.split('i')[0])
        kwargs['num_dishes'] = int(n_dish.split('d')[0])
        kwargs['max_ingredient_per_dish'] = int(n_max.split('m')[0])
        print('pb_cook_meal', kwargs)
        return pb_cook_meal(**kwargs)


def pb_block_ntp(task_json, start_task, num_tasks, num_episodes, seed):
    task_specs = json.load(open(task_json, 'r'))
    objects = [dict(path=URDFS['short_floor'], type_name='floor', fixed=True, globalScaling=0.5)]
    object_types = ['floor']
    placements = []
    for o in task_specs['scene']['objects']:
        objects.append(dict(path=o['filename'], type_name=str(o['name']), fixed=o['fixed']))
        placements.append(('on', str(o['name']) + '/0', 'floor'))
        object_types.append(str(o['name']))

    plist = []
    for t in task_specs['tasks'][start_task:start_task+num_tasks]:
        goal = []
        subgoals = []
        dependencies = []
        prev_goal = None
        prev_top = None
        for g in t['goals']:
            type_top = str(g['src'])
            type_bot = str(g['target'])
            sg = PBGoal('on', True, type_top, type_top + '/0', type_bot, type_bot + '/0')
            subgoals.append([sg])
            goal.append(sg)
            if prev_top == type_bot:
                #  if top block of the previous goal is the bottom of the current goal, set dependencies
                dependencies.append((sg, prev_goal))
            prev_top = type_top
            prev_goal = sg
        subgoals.append(goal)

        plist.append(
            {
                'robot_urdf': URDFS['ph_gripper'],
                'objects': objects,
                'object_types': object_types,
                'shapes': [],
                'placements': placements,
                'goal': goal,
                'subgoals': subgoals,
                'dependencies': dependencies,
                'env_name': 'TaskEnvBlock',
                'max_steps': 10,
            }
        )
    for p in plist:
        yield p, num_episodes


def kitchen_scene():
    objects = [
        dict(path=URDFS['short_floor'], type_name='table', fixed=True, globalScaling=0.8),
    ]
    object_types = ['table']

    equipments = ['stove', 'sink']
    containers = ['plate']
    cookwares = ['pot', 'pan']
    # ingredients = ['pear', 'apple', 'lemon', 'banana', 'plum', 'peach']
    veggies = ['cabbage', 'tomato', 'pumpkin']
    fruits = ['pear', 'orange', 'banana']
    ingredients = veggies + fruits
    # ingredients = ['pear', 'apple', 'lemon', 'banana']
    scaling = {'stove': 1.0, 'pan': 0.6, 'cabbage': 0.8, 'pumpkin': 0.8}
    copies = {'plate': 3}
    colors = {'pan': [0.67, 0.7, 0.74, 1], 'pot': [0.3, 0.3, 0.3, 1]}
    plate_colors = [[1, 1, 225/500, 1], [204/255, 255/255, 209/255, 1], [1, 1, 1, 1]]
    num_plates = copies['plate']

    placements = []
    for o_type in ingredients + containers + equipments + cookwares:
        scale = scaling.get(o_type, 1.0)
        n_copy = copies.get(o_type, 1)
        color = colors.get(o_type, None)
        for ci in range(n_copy):
            if o_type == 'plate':
                color = plate_colors[ci]
            objects.append(dict(path=URDFS[o_type], type_name=o_type, fixed=False, globalScaling=scale, color=color))
        object_types.append(o_type)

    for o_type in ingredients + containers:
        n_copy = copies.get(o_type, 1)
        for i in range(n_copy):
            placements.append(('on', o_type + '/%i' % i, 'table/0', np.array([[-0.4, -0.45], [0.4, 0.45]])))

    for o_type in cookwares:
        placements.append(('on', o_type + '/0', 'tray/0'))

    shapes = [
        # dict(geom='shape_box', type_name='workspace', w=0.8, l=0.9, h=0.015, color=(0.9297, 0.7930, 0.6758, 1)),
        dict(geom='shape_box', type_name='serving1', w=0.3, l=0.3, h=0.01, color=(215/255, 137/255, 136/255, 1)),
        dict(geom='shape_box', type_name='serving2', w=0.3, l=0.3, h=0.01, color=(165/255, 195/255, 148/255, 1)),
        dict(geom='shape_box', type_name='serving3', w=0.3, l=0.3, h=0.01, color=(111/255, 168/255, 220/255, 1)),
        dict(geom='shape_box', type_name='tray', w=0.3, l=0.6, h=0.01, color=(0.8, 0.8, 0.8, 1))
    ]
    serving_areas = ['serving1/0', 'serving2/0', 'serving3/0']
    object_types.extend(['serving1', 'serving2', 'serving3', 'tray'])

    # placements.append(('set', 'workspace/0', 'table/0', [[0, 0, 0], [0, 0, 0, 1]]))
    placements.append(('set', 'stove/0', 'table/0', [[0.7, 0, 0], [0, 0, 0, 1]]))
    placements.append(('set', 'sink/0', 'table/0', [[0, -0.7, 0], [0, 0, 0, 1]]))
    placements.append(('set', 'serving1/0', 'table/0', [[-0.45, 0.6, 0], [0, 0, 0, 1]]))
    placements.append(('set', 'serving2/0', 'table/0', [[0, 0.6, 0], [0, 0, 0, 1]]))
    placements.append(('set', 'serving3/0', 'table/0', [[0.45, 0.6, 0], [0, 0, 0, 1]]))
    placements.append(('set', 'tray/0', 'table/0', [[-0.6, -0.2, 0], [0, 0, 0, 1]]))

    return {
        'objects': objects,
        'shapes': shapes,
        'object_types': object_types,
        'placements': placements,
        'ingredients': ingredients,
        'serving_areas': serving_areas,
        'num_plates': num_plates,
        'veggies': veggies,
        'fruits': fruits
    }


def subset_sum(numbers, target, partial=[], partial_sum=0):
    if partial_sum == target:
        yield partial
    if partial_sum >= target:
        return
    for i, n in enumerate(numbers):
        remaining = numbers[i + 1:]
        yield from subset_sum(remaining, target, partial + [n], partial_sum + n)


def pb_cook_meal(task_file, start_task, num_tasks, num_episodes, num_ingredients, num_dishes, max_ingredient_per_dish,
                 shuffle, seed=0):

    npr = np.random.RandomState(seed)
    scene = kitchen_scene()
    assert(num_dishes <= num_ingredients <= len(scene['ingredients']))
    assert(max_ingredient_per_dish * num_dishes >= num_ingredients)

    # all possible splits
    splits = []
    split_nums = list(range(max_ingredient_per_dish + 1)) * num_ingredients
    for split in subset_sum(split_nums, num_ingredients):
        if len(split) == num_dishes:
            split = list(split)
            split.extend([0 for _ in range(num_dishes, len(scene['serving_areas']))])
            for s in permutations(split):
                if s not in splits:
                    splits.append(s)

    print(splits)
    # all permutations
    meals = []
    for ing_perms in permutations(scene['ingredients'], num_ingredients):
        # break perm into dishes
        for split in splits:
            dish = []
            assert(sum(split) == num_ingredients)
            top = 0
            for s in split:
                dish.append(ing_perms[top:top+s])
                top += s
            assert(top == num_ingredients)
            meals.append(dish)

    if shuffle:
        num_meals = len(meals)
        meal_idx = npr.choice(np.arange(num_meals), num_meals, replace=False)
        meals = [meals[i] for i in meal_idx]

    def binary(p, o1, o2):
        o1t = o1 if '/' in o1 else o1 + '/0'
        o2t = o2 if '/' in o2 else o2 + '/0'
        return PBGoal(p, True, o1, o1t, o2, o2t)

    def unitary(p, o1):
        o1t = o1 if '/' in o1 else o1 + '/0'
        return PBGoal(p, True, o1, o1t)

    num_tasks = len(meals) if num_tasks < 0 else num_tasks
    end_task = min(len(meals), start_task + num_tasks)

    for dishes, ti in zip(meals[start_task:end_task], range(start_task, end_task)):
        for _ in range(num_episodes):
            goal = []
            subgoals = []
            dependencies = []
            plate_order = npr.choice(np.arange(scene['num_plates']), scene['num_plates'], replace=False)

            first_dish = True
            current_cookwares = []
            for dish_i, dish_ings in enumerate(dishes):
                if len(dish_ings) == 0:
                    continue
                serving = 'serving%i' % (dish_i + 1)
                plate = 'plate/%i' % (plate_order[dish_i])

                goal.append(binary('on', plate, serving))
                for ing in dish_ings:
                    goal.append(binary('on', ing, plate))
                    goal.append(unitary('cooked', ing))

                for ing_i, ing in enumerate(dish_ings):
                    cookware = 'pan' if ing in scene['fruits'] else 'pot'
                    if first_dish:
                        first_dish = False
                        subgoals.extend([
                            [binary('on', ing, 'sink')],
                            [unitary('cleaned', ing), unitary('activated', 'sink')],
                            [binary('on', cookware, 'stove')],
                            [binary('on', ing, cookware)],
                            [unitary('cleaned', ing), unitary('cooked', ing), unitary('activated', 'stove')],
                        ])
                        dependencies.extend([
                            [unitary('cleaned', ing), unitary('activated', 'sink')],
                            [unitary('activated', 'sink'), unitary('cleaned', ing)],
                            [unitary('cooked', ing), unitary('activated', 'stove')],
                            [unitary('activated', 'stove'), unitary('cooked', ing)],
                        ])
                    else:
                        if cookware not in current_cookwares:
                            # swap cookware
                            subgoals.extend([
                                [unitary('cleaned', ing), binary('on', ing, 'sink')],
                                [binary('on', cookware, 'stove')],
                                [unitary('cleaned', ing), unitary('cooked', ing), binary('on', ing, cookware)],
                            ])
                        else:
                            subgoals.extend([
                                [unitary('cleaned', ing), binary('on', ing, 'sink')],
                                [unitary('cleaned', ing), unitary('cooked', ing), binary('on', ing, cookware)],
                            ])
                        dependencies.extend([
                            [unitary('cleaned', ing), binary('on', ing, 'sink')],
                            [binary('on', ing, 'sink'), unitary('cleaned', ing)],
                            [unitary('cooked', ing), binary('on', ing, cookware)],
                            [binary('on', ing, cookware), unitary('cooked', ing)],
                        ])
                    current_cookwares.append(cookware)

                    if ing_i == 0:
                        subgoals.extend([
                            [binary('on', plate, serving)]
                        ])

                    subgoals.extend([
                        [binary('on', ing, plate)],
                    ])
                    dependencies.extend([
                        [unitary('cooked', ing), unitary('cleaned', ing)],
                        [binary('on', ing, plate), unitary('cooked', ing)],
                        [binary('on', ing, plate), binary('on', plate, serving)]
                    ])

            subgoals.append(goal)
            # for s in subgoals:
            #     print(s)
            # for dep in dependencies:
            #     print(str(dep[0]) + '->' + str(dep[1]))
            # print('===============================')
            # print(ti, goal)

            yield {
                    'robot_urdf': URDFS['ph_gripper'],
                    'objects': scene['objects'],
                    'object_types': scene['object_types'],
                    'shapes': scene['shapes'],
                    'placements': scene['placements'],
                    'goal': goal,
                    'subgoals': subgoals,
                    'dependencies': dependencies,
                    'env_name': 'TaskEnvCook',
                    'max_steps': num_ingredients * 10,
                  }, ti



def main():
    # a = pb_block_ntp('ntp_specs/stack/stack_2000.json', 10, 10)
    # print
    pgen = list(factory('pb_cook_meal_3i_1d_3m_iter',
                        task_file=None,
                        start_task=0,
                        num_tasks=-1,
                        num_episodes=1,
                        seed=0)
                )
    print(len(pgen))
    # for p, num_episode in pgen:
    #     print(p['goal'])
        # for sg in p['subgoals']:
        #     print(sg)
        # print('===============')
    # for p in grid_doorkey(10, 1, 2):
    #     print(p)

if __name__ == '__main__':
    main()
