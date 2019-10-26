"""This file implements functions that create supervision for training a regression planner."""

import init_path
from copy import deepcopy
import numpy as np

"""
subgoal, goal = plan[:-1], plan[-1]

Aliasing:
For each goal in the current goal, find alias (identical goal) in the current subgoals.

In general, the aliased goal should depend on the subgoal before the alias.
If there are concurrent goals other than the alias, 
    - If the concurrent goal is also aliased (not alias), then the concurrent goal should
    take precedence since its alias is before the subgoal of the alias in question.
    - If the concurrent goal is a leaf goal, then the alias should be grouped with the leaf
    goal and become a leaf group, which jointly depends on the subgoal before the group.

"""


def lookup_subgoal_index(subgoals_trace, goal):
    preimages = []
    for i, sg in enumerate(subgoals_trace):
        for j, g in enumerate(sg):
            if g == goal:
                preimages.append(i)
    if len(preimages) > 0:
        assert len(preimages) == 1, preimages
        return preimages[0]
    else:
        return len(subgoals_trace)


def same_trace_goal_list(g1, g2):
    if len(g1) != len(g2):
        return False
    for a, b in zip(g1, g2):
        if a != b:
            return False
    return True


def is_satisfied(goals):
    s = goals[0].satisfied
    for g in goals:
        assert(s == g.satisfied)
    return s


def make_satisfied(goals):
    for g in goals:
        g.make_satisfied()


def parse_plan_helper(plan, all_traces, all_extended_trace, trace=[], extended_trace=[]):
    """
    Parse an augmented plan (goal appended with satisfied state) into a list of reachable subgoal trajectories
    :param plan: [[[g0^0, g1^0, ...], bool], [[g0^1, g1^1 , ...], bool]]
    :param all_traces: all accumulated execution trace
    :param all_extended_trace:  all accumulated traces with all subgoals and dependency information
    :param trace: placeholder for recursion
    :param extended_trace: placeholder for recursion
    :return: None
    """
    if len(plan) == 0 or is_satisfied(plan[-1]):
        # all_traces.append(trace)
        # all_extended_trace.append(extended_trace)
        return
    goals = plan[-1]
    subgoals = plan[:-1]
    if len(trace) > 0:
        for g in goals:
            if g.is_alias and g not in trace[-1]:
                return

    # group goals by dependencies
    goal_group_dict = {}
    for g in goals:
        # Look for preimages: look up each goal and see if it has appeared in the subgoals
        # default preimage index is the length of the subgoal (1 more than the last subgoal index)
        # which means that goals that don't have specified pre-image are grouped and executed last
        preimage_idx = lookup_subgoal_index(subgoals, g)
        if preimage_idx not in goal_group_dict:
            goal_group_dict[preimage_idx] = []
        goal_group_dict[preimage_idx].append(g)

    # sort goal by dependency
    sorted_goal_groups = sorted(goal_group_dict.items(), key=lambda x: x[0])
    for gi, (sg_idx, gg) in enumerate(sorted_goal_groups):
        assert(not is_satisfied(gg))
        new_extended_trace = deepcopy(extended_trace)
        new_trace = deepcopy(trace)
        # check if alias: current goal group is same as the end of the trace
        is_alias = False
        if len(new_trace) != 0:
            is_alias = same_trace_goal_list(gg, new_trace[-1])

        # extend the trace with new goals
        if not is_alias:  # don't copy alias
            new_trace.append(deepcopy(gg))
            new_extended_trace.append(([x for _, x in sorted_goal_groups], gi))

        # recursion
        sg_idx = min(sg_idx, len(subgoals) - 1)
        parse_plan_helper(subgoals[:sg_idx + 1], all_traces, all_extended_trace, new_trace, new_extended_trace)

        # accumulate trace snapshots
        # don't accumulate for alias since we don't actually achieve the alias goal
        if not is_alias:
            new_trace = deepcopy(trace)
            new_extended_trace = deepcopy(extended_trace)
            new_trace.append(deepcopy(gg))
            new_extended_trace.append(([x for _, x in sorted_goal_groups], gi))
            all_traces.append(deepcopy(new_trace))
            all_extended_trace.append(deepcopy(new_extended_trace))

            make_satisfied(gg)


def parse_plan(plan):
    traces = []
    extended_traces = []
    for i in range(len(plan) - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            for sg1 in plan[i]:
                for sg2 in plan[j]:
                    if sg1 == sg2:
                        if sg2.is_alias:
                            print('%r is twice-aliased' % sg2)
                        sg2.is_alias = True

    parse_plan_helper(deepcopy(plan), traces, extended_traces)
    return traces, extended_traces


def serialize_trace(env, extended_trace, external_deps=[]):
    """
    # TODO: CAUTION! ENV NEEDS TO BE RUNNING IN ORDER TO GET CORRECT OUTPUT
    Parse extended traces to: [current_goal, satisfied, reachable, goal_mask, dependencies, dep_mask]
    :param env: environment
    :param extended_trace: a single extended trace from parse_plan
    :param external_deps: externally specified dependencies
    :return: 
    """
    num_predicate = len(env.predicates())
    # [trace_step, e1_idx, e2_idx, p1, m1, pi2, m2, depends_on]
    dep_template = np.zeros(4 + num_predicate * 4, dtype=np.int64)
    # [trace_step, entity_idx, predicate, mask, satisfied]
    sat_template = np.zeros(3 + num_predicate * 2, dtype=np.int64)

    curr_goal_trace = []
    satisfied_trace = []
    goal_mask_trace = []
    focus_trace = []
    deps_trace = []

    for ti, (trace_step, focus_goal_group_idx) in enumerate(extended_trace):
        goal_group_exp = []
        goal_group_mask = []
        goal_group_focus = []

        # TODO: item for each goal values (True False)
        msg = []
        for ggi, goal_group in enumerate(trace_step):
            for goal in goal_group:
                goal_group_exp.append(goal.clone())
                # general prediction mask
                mask = goal.clone()
                mask.value = True
                goal_group_mask.append(mask.clone())

                rec_success = focus_goal_group_idx == ggi and (ti + 1) == len(extended_trace) and env.is_success([goal])
                oi, pi = env.goal_symbolic_state_index(goal.clone())
                sat = sat_template.copy()
                sat[0] = ti
                sat[1] = oi
                sat[2 + pi] = goal.value  # value
                sat[2 + num_predicate + pi] = True  # mask
                sat[-1] = goal.satisfied or rec_success
                satisfied_trace.append(sat)

                # focus mask
                focus = goal.clone()
                focus.value = focus_goal_group_idx == ggi
                goal_group_focus.append(focus)
                msg.append('(%r, %r)' % (goal, focus.value))
        # print('|'.join(msg))
        curr_goal = env.serialize_symbolic_state(env.goal_to_symbolic_state(goal_group_exp))
        focus = env.serialize_symbolic_state(env.goal_to_symbolic_state(goal_group_focus))
        goal_mask = env.goal_mask(goal_group_mask)
        curr_goal_trace.append(curr_goal)
        goal_mask_trace.append(goal_mask)
        focus_trace.append(focus)

        # TODO: dependencies for individual goal values
        deps_pair = []
        for ggi in range(len(trace_step)):
            for ggj in range(len(trace_step)):
                for g1 in trace_step[ggi]:
                    for g2 in trace_step[ggj]:
                        # fully connected within a group (excluding self-connection)
                        # default independence among everything else
                        if g1 != g2:
                            deps_pair.append((g1, g2, ggi == ggj))

        for g1, g2, depends_on in deps_pair:
            for eg1, eg2 in external_deps:
                if g1 == eg1 and g2 == eg2:
                    depends_on = True
                    break
            d = dep_template.copy()
            oi1, pi1 = env.goal_symbolic_state_index(g1)
            oi2, pi2 = env.goal_symbolic_state_index(g2)
            d[0] = ti
            d[1:3] = [oi1, oi2]
            d[3 + pi1] = g1.value
            d[3 + num_predicate + pi1] = True
            d[3 + num_predicate * 2 + pi2] = g2.value
            d[3 + num_predicate * 3 + pi2] = True
            d[-1] = depends_on
            deps_trace.append(d)

    if len(deps_trace) == 0:
        deps_trace.append(dep_template.copy())

    reachable_trace = np.zeros(len(extended_trace), dtype=np.bool)
    reachable_trace[-1] = True  # the last step is automatically reachable

    outputs = {
        'goal_trace': np.concatenate(curr_goal_trace, axis=0).astype(np.bool),
        'goal_mask_trace': np.concatenate(goal_mask_trace, axis=0).astype(np.bool),
        'focus_trace': np.concatenate(focus_trace, axis=0).astype(np.bool),
        'satisfied_trace': np.stack(satisfied_trace),
        'dependency_trace': np.stack(deps_trace),
        'reachable_trace': reachable_trace,
        'num_goal_entities': np.array([g.shape[0] for g in curr_goal_trace])
    }
    return outputs


def test():
    from rpn.grid_envs import GridGoal
    plan = [
        [['A'], ['B'], ['C']],
        [['A'], ['B']],
        [['D'], ['E']],
        [['C']],
        [['H'], ['G']],
        [['H']],
        [['Z']],
        [['X']]
    ]

    plan1 = [
        [['A'], ['B'], ['C']],
        [['A']],
        [['B']],
        [['C']],
    ]

    plan2 = [
        [GridGoal('holding', True, 'key', 'blue')],
        [GridGoal('open', True, 'door', 'blue'), GridGoal('is_locked', False, 'door', 'blue')],
        [GridGoal('holding', True, 'ball', 'red')],
        [GridGoal('holding', True, 'key', 'yellow')],
        [GridGoal('open', True, 'door', 'yellow'), GridGoal('is_locked', False, 'door', 'yellow')],
        [GridGoal('on', True, 'floor', 'red')],
        [GridGoal('holding', True, 'ball', 'red'), GridGoal('on', True, 'floor', 'red')],
    ]
    test_plan(plan2)


def test_plan(plan):
    all_traces, all_extended_traces = parse_plan(deepcopy(plan))
    for p in all_traces:
        for x in p:
            print(x)
        print('=================')

    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    for p in all_extended_traces:
        for x in p:
            print(x)
        print('=================')

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')


def test_problem(problems):
    from rpn.grid_envs import EmptyEnv, RoomGoalEnv

    for problem, _ in problems:
        env = eval(problem['env_name'])(**problem['env'])
        # plan = augment_plan(problem['subgoals'])
        # assert(verify_plan(plan))
        all_traces, all_extended_traces = parse_plan(deepcopy(problem['subgoals']))
        for p in all_traces:
            for x in p:
                print(x)
            print('=================')

        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        for p in all_extended_traces:
            for x in p:
                print(x)
            print('=================')

        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        for trace in all_extended_traces:
            # TODO: CAUTION! INCORRECT SATISFIED CONDITION DUE TO ENV DEPENDENCY
            parsed_trace = serialize_trace(env, trace)
            print('======================')


if __name__=='__main__':
    # test()
    from rpn.problems_grid import grid_roomgoal1
    test_problem(grid_roomgoal1(-1, 1))
