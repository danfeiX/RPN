"""This file implements RPN and baselines for the minigrid environment."""

import torch
import torch.nn as nn

from rpn.utils.torch_wrapper import MLP, time_distributed
from rpn.utils.torch_graph_utils import construct_full_graph
import rpn.utils.torch_utils as tu
import rpn.utils.np_utils as npu
from rpn.utils.eval_utils import classification_accuracy, masked_binary_accuracy
import numpy as np
import networkx as nx
from rpn.net import Net, masked_symbolic_state_index


class BCSubGoal(Net):
    """Directly predicts subgoal (E2E baseline)."""
    def __init__(self, **kwargs):
        Net.__init__(self, **kwargs)
        c = self.c

        sg_n_out = c.n_object * c.symbol_size * 3
        n_in = c.n_in * c.n_object
        self._sg_net = MLP(n_in + sg_n_out, sg_n_out, c.hidden_dims)

    def forward(self, states, goal, subgoal=None):
        states = tu.flatten(states)
        goal = tu.flatten(goal)

        # get sub-goal prediction
        sg_out = self._sg_net(torch.cat((states, goal), dim=-1))

        sg_preds = sg_out.view(states.shape[0], -1, self.c.symbol_size, 3)
        if self.policy_mode:
            sg_cls = sg_preds.argmax(dim=-1)
            subgoal = tu.to_onehot(sg_cls, 3)  # [false, true, masked]

        # get action prediction
        return {
            'subgoal_preds': sg_preds,
            'subgoal': subgoal
        }

    def forward_batch(self, batch):
        goal = masked_symbolic_state_index(batch['goal'], batch['goal_mask'])
        goal = tu.to_onehot(goal, 3)
        subgoal = None
        if not self.policy_mode:
            subgoal = masked_symbolic_state_index(batch['subgoal'], batch['subgoal_mask'])
            subgoal = tu.to_onehot(subgoal, 3)
        return self(
            states=batch['states'],
            goal=goal,
            subgoal=subgoal,
        )

    @staticmethod
    def compute_losses(outputs, batch):
        subgoal = masked_symbolic_state_index(batch['subgoal'], batch['subgoal_mask'])
        subgoal_loss = nn.CrossEntropyLoss()(
            outputs['subgoal_preds'].reshape(-1, 3),
            subgoal.reshape(-1)
        )
        return {
            'subgoal': subgoal_loss,
        }

    @staticmethod
    def log_outputs(outputs, batch, summarizer, global_step, prefix):
        subgoal = masked_symbolic_state_index(batch['subgoal'], batch['subgoal_mask'])
        subgoal_acc, subgoal_mask_acc = masked_binary_accuracy(outputs['subgoal_preds'], subgoal)
        summarizer.add_scalar(prefix + 'acc/subgoal', subgoal_acc, global_step=global_step)
        summarizer.add_scalar(prefix + 'acc/subgoal_mask', subgoal_mask_acc, global_step=global_step)


def sort_goal_graph(edges, node_index):
    # create undirected graph given edges and node index.
    g = nx.Graph()
    for n in node_index:
        g.add_node(n)
    edges = edges.tolist()
    for s, t in edges:
        if [t, s] in edges:
            g.add_edge(s, t)
    cliques = []
    # Bron-Kerbosch
    for c in nx.find_cliques(g):
        cliques.append(tuple(c))

    dag = nx.DiGraph()
    for c in cliques:
        dag.add_node(c)

    for i in range(len(cliques)):
        for j in range(i + 1, len(cliques)):
            c1 = cliques[i]
            c2 = cliques[j]
            for n1 in c1:
                for n2 in c2:
                    if [n1, n2] in edges:
                        dag.add_edge(c1, c2)
                    if [n2, n1] in edges:
                        dag.add_edge(c2, c1)

    if not nx.is_directed_acyclic_graph(dag):
        return []

    return [list(s) for s in nx.topological_sort(dag)]


class BCBP(Net):
    """Regression Planning Networks (RPN) for the Grid environment."""
    def __init__(self, **kwargs):
        Net.__init__(self, **kwargs)
        c = self.c
        input_size = c.n_in * c.n_object
        symbol_size = c.symbol_size * c.n_object * 3

        # self._policy = MLP(input_size + symbol_size, c.n_action, c.policy_dims)
        self._preimage = MLP(input_size + symbol_size, symbol_size, c.hidden_dims)
        self._reachable = MLP(input_size + symbol_size, 2, c.hidden_dims)
        self._satisfied = MLP(c.n_in + c.symbol_size * 2, 2, c.hidden_dims)
        self._dependency = MLP(c.n_in * 2 + c.symbol_size * 4, 2, c.hidden_dims)

    def _serialize_subgoals(self, entity_state, object_state, curr_goal):
        assert(len(curr_goal.shape) == 3)
        assert(len(entity_state.shape) == 3)
        assert(entity_state.shape[1] == curr_goal.shape[1])
        state_np = tu.to_numpy(entity_state)[0]
        num_predicate = curr_goal.shape[-1]
        curr_goal_np = tu.to_numpy(curr_goal[0])
        goal_object_index, goal_predicates_index = np.where(curr_goal_np != 2)

        goal_index = np.stack([goal_object_index, goal_predicates_index]).transpose()
        goal_predicates_value = curr_goal_np[(goal_object_index, goal_predicates_index)]
        num_goal = goal_index.shape[0]
        # predict satisfaction
        sat_state_inputs = state_np[goal_object_index]
        sat_predicate_mask = npu.to_onehot(goal_predicates_index, num_predicate).astype(np.float32)
        sat_predicate = sat_predicate_mask.copy()
        sat_predicate[sat_predicate_mask.astype(np.bool)] = goal_predicates_value

        sat_state_inputs = tu.to_tensor(sat_state_inputs[None, ...], device=entity_state.device)
        sat_sym_inputs_np = np.concatenate((sat_predicate, sat_predicate_mask), axis=-1)[None, ...]
        sat_sym_inputs = tu.to_tensor(sat_sym_inputs_np, device=entity_state.device)
        sat_preds = tu.to_numpy(self.forward_sat(sat_state_inputs, sat_sym_inputs).argmax(-1))[0]  # [ng]
        assert (sat_preds.shape[0] == num_goal)
        if self.verbose and self.env is not None:
            for sat_p, sat_m, sp, oi in zip(sat_predicate, sat_predicate_mask, sat_preds, goal_object_index):
                sat_pad = np.hstack([[oi], sat_p, sat_m, [sp]])
                print('[bp] sat: ', self.env.deserialize_satisfied_entry(sat_pad))

        # Construct dependency graphs
        nodes, edges = construct_full_graph(num_goal)
        src_object_index = goal_object_index[edges[:, 0]]  # list of [object_idx, predicate_idx] for each edge source
        tgt_object_index = goal_object_index[edges[:, 1]]
        src_inputs = state_np[src_object_index]  # list of object states
        tgt_inputs = state_np[tgt_object_index]

        src_predicate_value = goal_predicates_value[edges[:, 0]]  # list of predicate values for each edge source
        src_predicate_index = goal_predicates_index[edges[:, 0]]
        tgt_predicate_value = goal_predicates_value[edges[:, 1]]
        tgt_predicate_index = goal_predicates_index[edges[:, 1]]
        src_predicate_mask = npu.to_onehot(src_predicate_index, num_predicate).astype(np.float32)
        src_predicate = np.zeros_like(src_predicate_mask)
        src_predicate[src_predicate_mask.astype(np.bool)] = src_predicate_value
        tgt_predicate_mask = npu.to_onehot(tgt_predicate_index, num_predicate).astype(np.float32)
        tgt_predicate = np.zeros_like(tgt_predicate_mask)
        tgt_predicate[tgt_predicate_mask.astype(np.bool)] = tgt_predicate_value
        # dependency_inputs_np = np.concatenate(
        #     (src_inputs, tgt_inputs, src_predicate, src_predicate_mask, tgt_predicate, tgt_predicate_mask), axis=-1)
        dependency_state_inputs_np = np.concatenate((src_inputs, tgt_inputs), axis=-1)
        dependency_sym_inputs_np = np.concatenate(
            (src_predicate, src_predicate_mask, tgt_predicate, tgt_predicate_mask), axis=-1)

        if dependency_state_inputs_np.shape[0] > 0:
            dependency_state_inputs = tu.to_tensor(dependency_state_inputs_np, device=entity_state.device).unsqueeze(0)
            dependency_sym_inputs = tu.to_tensor(dependency_sym_inputs_np, device=entity_state.device).unsqueeze(0)
            deps_preds = tu.to_numpy(self.forward_dep(dependency_state_inputs, dependency_sym_inputs).argmax(-1))[0]
            dep_graph_edges = edges[deps_preds > 0]
        else:
            dep_graph_edges = np.array([])
        sorted_goal_groups = sort_goal_graph(dep_graph_edges, nodes)
        focus_group_idx = None
        for gg in reversed(sorted_goal_groups):
            if not np.any(sat_preds[gg]):  # if unsatisfied
                focus_group_idx = gg
                break

        if focus_group_idx is None:
            return None, 'NETWORK_ALL_SATISFIED'

        # focus_group_idx is a list of goal index
        focus_group_np = np.ones_like(curr_goal_np) * 2
        for fg_idx in focus_group_idx:
            fg_obj_i, fg_pred_i = goal_index[fg_idx]
            focus_group_np[fg_obj_i, fg_pred_i] = curr_goal_np[fg_obj_i, fg_pred_i]
        focus_group = tu.to_tensor(focus_group_np, device=entity_state.device).unsqueeze(0)
        focus_group = tu.to_onehot(focus_group, 3)
        return focus_group, -1

    def find_subgoal(self, object_state, entity_state, goal, graphs, max_depth=10):
        """
        Resolve the next subgoal recursively

        Planner logic:
        1. Use Bron–Kerbosch to find all maximal cliques in the (disconnected) dependency graph
        2. Form a DAG by using the cliques as nodes
        3. Sort the DAG topologically. Find the first group that is not satisfied and name it root.
        4. Use root as the mask for the current goal to form the focused group
        5. Predict the preimage and reachability of the focused group
        6. If the focus group is reachable, stop and feed the focused group to the policy.
        7. Otherwise, treat the focus goal group as the new goal and go back to 1.

        :param object_state: current state of objects
        :param goal: global goal
        :return: the next subgoal
        """

        curr_goal = goal.argmax(dim=-1)  # [1, num_object, num_predicate]
        subgoal = None
        depth = 0
        ret = -1
        while depth < max_depth:
            if self.verbose:
                print('[bp] Depth: %i ==== ' % depth)
            if (curr_goal == 2).all():
                ret = 'NETWORK_EMPTY_GOAL'
                break
            focus_group, ret = self._serialize_subgoals(entity_state, object_state, curr_goal)
            if focus_group is None:
                break
            # preimage
            bp_out = self.backward_plan(entity_state, object_state, focus_group, graphs)
            preimage_preds = bp_out['preimage_preds'].argmax(dim=-1)
            reachable_preds = bp_out['reachable_preds'].argmax(dim=-1)

            if self.verbose and self.env is not None:
                curr_goal_np = tu.to_numpy(curr_goal[0])
                print('[bp] current goals: ', self.env.deserialize_goals(curr_goal_np, curr_goal_np != 2))
                focus_group_np = tu.to_numpy(focus_group[0])
                print('[bp] focus group: ', self.env.deserialize_goals(focus_group_np[..., 1], (1 - focus_group_np[..., 2])))
                print('[bp] reachable: ', reachable_preds)

            if (reachable_preds == 1).any():
                subgoal = focus_group
                break
            curr_goal = preimage_preds
            depth += 1
        else:
            ret = 'NETWORK_MAX_DEPTH'
        if self.verbose:
            print('[bp] EOP###########')
        return {'subgoal': subgoal, 'ret': ret}

    def forward(self,
                object_states,
                entity_states,
                goal,
                focus_goal,
                satisfied_info,
                dependency_info,
                graph,
                num_entities,
                ):
        planner_out = self.dep_sat(entity_states, satisfied_info, dependency_info, num_entities)
        planner_out.update(self.backward_plan(entity_states, object_states, focus_goal, graph))
        return planner_out

    def forward_policy(self, object_states, entity_states, goal, graphs=None):
        planner_out = self.find_subgoal(object_states, entity_states, goal, graphs)
        if self.verbose and planner_out['ret'] != -1:
            print(planner_out['ret'])

        subgoal = planner_out['subgoal']
        if subgoal is None:
            return planner_out
        # TODO: policy
        return planner_out

    def forward_sat(self, sat_states, sat_info):
        return self._satisfied(torch.cat((sat_states, sat_info), dim=-1))

    def forward_dep(self, dep_states, dep_info):
        return self._dependency(torch.cat((dep_states, dep_info), dim=-1))

    def dep_sat(self, entity_states, satisfied_info, dependency_info, num_entities=None):
        sat_states = tu.gather_sequence(entity_states, satisfied_info[:, 0].long())
        satisfied_preds = self.forward_sat(sat_states, satisfied_info[:, 1:])

        dep_states_src = tu.gather_sequence(entity_states, dependency_info[:, 0].long())
        dep_states_tgt = tu.gather_sequence(entity_states, dependency_info[:, 1].long())
        dependency_preds = self.forward_dep(
            torch.cat((dep_states_src, dep_states_tgt), dim=-1), dependency_info[:, 2:]
        )
        return {
            'satisfied_preds': satisfied_preds,
            'dependency_preds': dependency_preds
        }

    def backward_plan(self, entity_states, object_states, focus_goal, graph=None):
        inputs_focus = torch.cat((tu.flatten(object_states), tu.flatten(focus_goal)), dim=-1)
        preimage_preds = self._preimage(inputs_focus)
        reachable_preds = self._reachable(inputs_focus)
        return {
            'preimage_preds': preimage_preds.reshape(focus_goal.shape[0], -1, self.c.symbol_size, 3),
            'reachable_preds': reachable_preds,
        }

    def forward_batch(self, batch):
        if not self.policy_mode:
            goal = masked_symbolic_state_index(batch['goal'], batch['goal_mask'])
            goal = tu.to_onehot(goal, 3)
            focus_goal = masked_symbolic_state_index(batch['goal'], batch['focus_mask'])
            focus_goal = tu.to_onehot(focus_goal, 3)
            satisfied_info = batch['satisfied'][:, :-1]
            dependency_info = batch['dependency'][:, :-1]

            return self.forward(
                object_states=batch['states'],
                entity_states=batch.get('entity_states', batch['states']),
                goal=goal,
                focus_goal=focus_goal,
                satisfied_info=satisfied_info,
                dependency_info=dependency_info,
                graph=batch.get('graph', None),
                num_entities=batch.get('num_entities', None)
            )
        else:
            goal = masked_symbolic_state_index(batch['goal'], batch['goal_mask'])
            goal = tu.to_onehot(goal, 3)
            return self.forward_policy(
                object_states=batch['states'],
                entity_states=batch.get('entity_states', batch['states']),
                goal=goal,
                graphs=batch.get('graphs', None),
            )

    @staticmethod
    def compute_losses(outputs, batch):
        preimage = masked_symbolic_state_index(batch['preimage'], batch['preimage_mask'])
        preimage_loss = nn.CrossEntropyLoss()(
            outputs['preimage_preds'].reshape(-1, 3) * batch['preimage_loss_mask'].reshape(-1).unsqueeze(-1),
            preimage.reshape(-1) * batch['preimage_loss_mask'].reshape(-1).long()
        )
        reachable_loss = nn.CrossEntropyLoss()(outputs['reachable_preds'], batch['reachable'].long())
        satisfied_loss = nn.CrossEntropyLoss()(outputs['satisfied_preds'], batch['satisfied'][:, -1].long())
        dependency_loss = nn.CrossEntropyLoss()(outputs['dependency_preds'], batch['dependency'][:, -1].long())

        # action_loss = nn.CrossEntropyLoss()(outputs['action_preds'], batch['action_labels'])
        return {
            'preimage': preimage_loss,
            'reachable': reachable_loss,
            'satisfied': satisfied_loss,
            'dependency': dependency_loss,
            # 'action': action_loss
        }

    @staticmethod
    def log_outputs(outputs, batch, summarizer, global_step, prefix):
        preimage = masked_symbolic_state_index(batch['preimage'], batch['preimage_mask'])
        preimage_preds = outputs['preimage_preds'].argmax(-1)
        preimage_preds.masked_fill_(batch['preimage_loss_mask'] == 0, 2)
        preimage.masked_fill_(batch['preimage_loss_mask'] == 0, 2)

        preimage_acc, preimage_mask_acc = masked_binary_accuracy(
            tu.to_onehot(preimage_preds, 3),
            preimage
        )
        reachable_acc = classification_accuracy(outputs['reachable_preds'], batch['reachable'])
        satisfied_acc = classification_accuracy(outputs['satisfied_preds'], batch['satisfied'][:, -1].long())
        dependency_acc = classification_accuracy(outputs['dependency_preds'], batch['dependency'][:, -1].long())

        summarizer.add_scalar(prefix + 'acc/preimage', preimage_acc, global_step=global_step)
        summarizer.add_scalar(prefix + 'acc/preimage_mask', preimage_mask_acc, global_step=global_step)
        summarizer.add_scalar(prefix + 'acc/reachable', reachable_acc, global_step=global_step)
        summarizer.add_scalar(prefix + 'acc/satisfied', satisfied_acc, global_step=global_step)
        summarizer.add_scalar(prefix + 'acc/dependency', dependency_acc, global_step=global_step)

    @staticmethod
    def debug(outputs, batch, env=None):
        preimage = tu.to_numpy(masked_symbolic_state_index(batch['preimage'], batch['preimage_mask']))
        preimage_preds = tu.to_numpy(outputs['preimage_preds'].argmax(-1))
        print('preimage')
        for pi, pip, pm in zip(preimage, preimage_preds, tu.to_numpy(batch['preimage_loss_mask'])):
            if np.all(pi == pip) or np.all(pm == 0):
                continue
            print('preds: ', env.masked_symbolic_state(env.deserialize_symbolic_state(pip)))
            print('label: ', env.masked_symbolic_state(env.deserialize_symbolic_state(pi)))

        focus_goal = tu.to_numpy(masked_symbolic_state_index(batch['goal'], batch['focus_mask']))

        print('reachable')
        reachable_preds = tu.to_numpy(outputs['reachable_preds'].argmax(-1))
        reachable_label = tu.to_numpy(batch['reachable'])
        for i, (rp, rl) in enumerate(zip(reachable_preds, reachable_label)):
            if int(rp) == int(rl):
                continue
            msg = 'fp' if int(rl) == 0 else 'fn'
            print(msg, env.masked_symbolic_state(env.deserialize_symbolic_state(focus_goal[i])))

        print('dependency')
        dep_preds = tu.to_numpy(outputs['dependency_preds'].argmax(-1))
        dep_label = tu.to_numpy(batch['dependency'])
        for i, (dp, dl) in enumerate((zip(dep_preds, dep_label))):
            if int(dp) == int(dl[-1]):
                continue
            msg = 'fp' if int(dl[-1]) == 0 else 'fn'
            print(msg, env.deserialize_dependency_entry(dl))


class GreedyBP(BCBP):
    """
    SS-only: nstead of performing regression planning, this model directly
    plans the next intermediate goal based on the highest-priority subgoal produced by subgoal serialization.
    """
    def __init__(self, **kwargs):
        Net.__init__(self, **kwargs)
        c = self.c
        input_size = c.n_in * c.n_object
        symbol_size = c.symbol_size * c.n_object * 3
        self._subgoal = MLP(input_size + symbol_size, symbol_size, c.hidden_dims)
        self._satisfied = MLP(c.n_in + c.symbol_size * 2, 2, c.hidden_dims)
        self._dependency = MLP(c.n_in * 2 + c.symbol_size * 4, 2, c.hidden_dims)

    def find_subgoal(self, object_state, entity_state, goal, graphs, max_depth=10):
        """
        Resolve the next subgoal directly
        """
        curr_goal = goal.argmax(dim=-1)  # [1, num_object, num_predicate]
        focus_group, ret = self._serialize_subgoals(entity_state, object_state, curr_goal)
        subgoal = None
        if focus_group is not None:
            bp_out = self.backward_plan(entity_state, object_state, focus_group, graphs)
            subgoal_preds = bp_out['subgoal_preds'].argmax(dim=-1)
            subgoal = tu.to_onehot(subgoal_preds, 3)
            if self.verbose and self.env is not None:
                curr_goal_np = tu.to_numpy(curr_goal[0])
                print('[bp] current goals: ', self.env.deserialize_goals(curr_goal_np, curr_goal_np != 2))
                focus_group_np = tu.to_numpy(focus_group[0])
                print('[bp] focus group: ', self.env.deserialize_goals(focus_group_np[..., 1], (1 - focus_group_np[..., 2])))
                subgoal_np = tu.to_numpy(subgoal[0])
                print('[bp] subgoal: ', self.env.deserialize_goals(subgoal_np[..., 1], (1 - subgoal_np[..., 2])))

        return {'subgoal': subgoal, 'ret': ret}

    def backward_plan(self, entity_states, object_states, focus_goal, graph=None):
        inputs_focus = torch.cat((tu.flatten(object_states), tu.flatten(focus_goal)), dim=-1)
        subgoal_preds = self._subgoal(inputs_focus)
        return {
            'subgoal_preds': subgoal_preds.reshape(focus_goal.shape[0], -1, self.c.symbol_size, 3),
        }

    @staticmethod
    def compute_losses(outputs, batch):
        subgoal = masked_symbolic_state_index(batch['subgoal'], batch['subgoal_mask'])
        subgoal_loss = nn.CrossEntropyLoss()(outputs['subgoal_preds'].reshape(-1, 3), subgoal.reshape(-1))
        satisfied_loss = nn.CrossEntropyLoss()(outputs['satisfied_preds'], batch['satisfied'][:, -1].long())
        dependency_loss = nn.CrossEntropyLoss()(outputs['dependency_preds'], batch['dependency'][:, -1].long())
        return {
            'subgoal': subgoal_loss,
            'satisfied': satisfied_loss,
            'dependency': dependency_loss,
        }

    @staticmethod
    def log_outputs(outputs, batch, summarizer, global_step, prefix):
        subgoal = masked_symbolic_state_index(batch['subgoal'], batch['subgoal_mask'])
        subgoal_preds = outputs['subgoal_preds'].argmax(-1)
        subgoal_acc, subgoal_mask_acc = masked_binary_accuracy(tu.to_onehot(subgoal_preds, 3), subgoal)
        satisfied_acc = classification_accuracy(outputs['satisfied_preds'], batch['satisfied'][:, -1].long())
        dependency_acc = classification_accuracy(outputs['dependency_preds'], batch['dependency'][:, -1].long())
        summarizer.add_scalar(prefix + 'acc/subgoal', subgoal_acc, global_step=global_step)
        summarizer.add_scalar(prefix + 'acc/subgoal_mask', subgoal_mask_acc, global_step=global_step)
        summarizer.add_scalar(prefix + 'acc/satisfied', satisfied_acc, global_step=global_step)
        summarizer.add_scalar(prefix + 'acc/dependency', dependency_acc, global_step=global_step)

    @staticmethod
    def debug(outputs, batch, env=None):
        subgoal = tu.to_numpy(masked_symbolic_state_index(batch['subgoal'], batch['subgoal_mask']))
        subgoal_preds = tu.to_numpy(outputs['subgoal_preds'].argmax(-1))
        focus_goal = tu.to_numpy(masked_symbolic_state_index(batch['goal'], batch['focus_mask']))
        print('subgoal')
        for pi, pip, fg in zip(subgoal, subgoal_preds, focus_goal):
            if np.all(pi == pip):
                continue
            print('- preds: ', env.masked_symbolic_state(env.deserialize_symbolic_state(pip)))
            print('label: ', env.masked_symbolic_state(env.deserialize_symbolic_state(pi)))
            print('focused: ', env.masked_symbolic_state(env.deserialize_symbolic_state(fg)))


class MonoBP(BCBP):
    """
    RP-only: Replace the subgoal serialization procedure with an end-to-end network.
    """
    def __init__(self, **kwargs):
        Net.__init__(self, **kwargs)
        c = self.c
        input_size = c.n_in * c.n_object
        symbol_size = c.symbol_size * c.n_object * 3
        self._focus = MLP(input_size + symbol_size, symbol_size, c.hidden_dims)
        self._preimage = MLP(input_size + symbol_size, symbol_size, c.hidden_dims)
        self._reachable_encoder = MLP(c.n_in + c.symbol_size * 3, 64, layer_dims=(128,), output_activation=nn.ReLU)
        self._reachable = MLP(64, 2, [64])

    def focus(self, entity_states, object_states, goal):
        inputs = torch.cat((tu.flatten(object_states), tu.flatten(goal)), dim=-1)
        focus_preds = self._focus(inputs)
        return {
            'focus_preds': focus_preds.reshape(goal.shape[0], -1, self.c.symbol_size, 3),
        }

    def backward_plan(self, entity_states, object_states, focus_goal, graphs=None):
        inputs_focus = torch.cat((tu.flatten(object_states), tu.flatten(focus_goal)), dim=-1)
        preimage_preds = self._preimage(inputs_focus)

        reachable_inputs = torch.cat(
            (tu.flatten(entity_states, begin_axis=2), tu.flatten(focus_goal, begin_axis=2)), dim=-1

        )
        focus_enc = time_distributed(reachable_inputs, self._reachable_encoder)
        # reachable_enc = self._reachable_gn(focus_enc, graphs)
        reachable_enc_red, _ = torch.max(focus_enc, dim=-2)
        reachable_preds = self._reachable(reachable_enc_red)

        return {
            'preimage_preds': preimage_preds.reshape(focus_goal.shape[0], -1, self.c.symbol_size, 3),
            'reachable_preds': reachable_preds,
        }

    def forward(self,
                object_states,
                entity_states,
                goal,
                focus_goal,
                satisfied_info,
                dependency_info,
                graph,
                num_entities,
                ):
        planner_out = self.focus(entity_states, object_states, goal)
        planner_out.update(self.backward_plan(entity_states, object_states, focus_goal, graph))
        return planner_out

    def _serialize_subgoals(self, entity_state, object_state, curr_goal):
        curr_goal = tu.to_onehot(curr_goal, 3)
        focus_group = self.focus(entity_state, object_state, curr_goal)['focus_preds']
        focus_group = tu.to_onehot(focus_group.argmax(-1), 3)
        return focus_group, -1

    @staticmethod
    def compute_losses(outputs, batch):
        preimage = masked_symbolic_state_index(batch['preimage'], batch['preimage_mask'])
        preimage_loss = nn.CrossEntropyLoss()(
            outputs['preimage_preds'].reshape(-1, 3) * batch['preimage_loss_mask'].reshape(-1).unsqueeze(-1),
            preimage.reshape(-1) * batch['preimage_loss_mask'].reshape(-1).long()
        )
        focus = masked_symbolic_state_index(batch['goal'], batch['focus_mask'])
        focus_loss = nn.CrossEntropyLoss()(outputs['focus_preds'].reshape(-1, 3), focus.reshape(-1))
        reachable_loss = nn.CrossEntropyLoss()(outputs['reachable_preds'], batch['reachable'].long())
        return {
            'subgoal': focus_loss,
            'preimage': preimage_loss,
            'reachable': reachable_loss
        }

    @staticmethod
    def log_outputs(outputs, batch, summarizer, global_step, prefix):
        preimage = masked_symbolic_state_index(batch['preimage'], batch['preimage_mask'])
        preimage_preds = outputs['preimage_preds'].argmax(-1)
        preimage_preds.masked_fill_(batch['preimage_loss_mask'] == 0, 2)
        preimage.masked_fill_(batch['preimage_loss_mask'] == 0, 2)

        preimage_acc, preimage_mask_acc = masked_binary_accuracy(
            tu.to_onehot(preimage_preds, 3),
            preimage
        )
        focus = masked_symbolic_state_index(batch['goal'], batch['focus_mask'])
        focus_preds = outputs['focus_preds'].argmax(-1)
        focus_acc, focus_mask_acc = masked_binary_accuracy(tu.to_onehot(focus_preds, 3), focus)
        reachable_acc = classification_accuracy(outputs['reachable_preds'], batch['reachable'])
        summarizer.add_scalar(prefix + 'acc/focus', focus_acc, global_step=global_step)
        summarizer.add_scalar(prefix + 'acc/focus_mask', focus_mask_acc, global_step=global_step)
        summarizer.add_scalar(prefix + 'acc/preimage', preimage_acc, global_step=global_step)
        summarizer.add_scalar(prefix + 'acc/preimage_mask', preimage_mask_acc, global_step=global_step)
        summarizer.add_scalar(prefix + 'acc/reachable', reachable_acc, global_step=global_step)
