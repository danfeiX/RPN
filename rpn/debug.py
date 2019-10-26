import rpn.utils.torch_utils as tu
from rpn.problems_grid import *
from rpn.data_utils import BulletPreimageVisualDataset
from rpn.env_utils import World
from rpn.env_utils import load_world, pb_session
from rpn.problems_pb import factory


def examine_preimage_data(s, env, env_name):
    print(env_name)
    s = tu.batch_to_numpy(s)

    print('seq_i: ', s['seq_idx'])
    for maskee_k, mask_k in [
        ('goal', 'goal_mask'),
        ('preimage', 'preimage_mask'),
        ('subgoal', 'subgoal_mask'),
        ('goal', 'focus_mask')
    ]:
        if mask_k == 'focus_mask':
            print('focus goal, reachable=%r' % (s['reachable'] > 0.5))
        else:
            print(maskee_k)
        if maskee_k == 'goal':
            for val, mask in zip(s[maskee_k], s[mask_k]):
                print(env.deserialize_goals(val, mask))
        else:
            print(env.deserialize_goals(s[maskee_k], s[mask_k]))

    print('satisfied')
    for sat in s['satisfied']:
        print(env.deserialize_satisfied_entry(sat))

    print('dependency')
    if np.any(s['dependency'] != 0):
        for dep in s['dependency']:
            print(env.deserialize_dependency_entry(dep))


def main():
    # env_name = 'grid_kdg_partial'
    # problem_gen = eval(env_name)(10, 1)
    # problem, _ = next((iter(problem_gen)))
    # env = eval(problem['env_name'])(**problem['env'])
    # db = GridPreimageDataset.load('data/grid/kdg/kdg_partial_500_5_s0.h5', debug=True)
    # examine_preimage_data(db[0], env, env_name)
    with pb_session(use_gui=False):
        env_name = 'pb_cook_meal_3i_1d_3m_iter'
        # problem_gen = eval(env_name)('ntp_specs/stack/stack_2000.json', 0, 10, 1)
        pgen = factory(
            env_name,
            task_file=None,
            start_task=0,
            num_tasks=-1,
            num_episodes=1,
            seed=0
        )
        problem, _ = next((iter(pgen)))
        world = load_world(problem, World(problem['object_types']))
        env = eval(problem['env_name'])(world)
        env.reset()
        db = BulletPreimageVisualDataset.load('data/cook/meal_visual/3i_1d_3m_20_s0.group', debug=True)
        # db2 = BulletPreimageDataset.load('data/cook/cook_combo/rand_dep_combo3_0-12_500_s0.group', debug=True)

        examine_preimage_data(db[0], env, env_name)


if __name__ == '__main__':
    main()
