"""This file implements RPN and baselines for the Kitchen3D environment."""

import torch
import torch.nn as nn

from rpn.utils.torch_wrapper import MLP, time_distributed
import rpn.utils.torch_utils as tu
from rpn.utils.eval_utils import classification_accuracy, masked_binary_accuracy
from rpn.net import Net, masked_symbolic_state_index
from rpn.rpn_grid import BCBP

IMAGE_SIZE = 24  # size of the renormalized object boxes.


class ImageEncoder(nn.Module):
    def __init__(self, im_size, out_size, out_activation=nn.ReLU()):
        super(ImageEncoder, self).__init__()
        fc_size = (im_size // (2 ** 3)) ** 2
        self._net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        self._fc1 = nn.Linear(fc_size * 32, out_size)
        self._out_act = out_activation

    def forward(self, input_im):
        im_enc = self._net(input_im)
        im_enc = self._fc1(tu.flatten(im_enc, begin_axis=1))
        if self._out_act is not None:
            im_enc = self._out_act(im_enc)
        return im_enc


class ReachableNet(nn.Module):
    def __init__(self, feat_size, symbol_size, symbol_enc_size=32):
        super(ReachableNet, self).__init__()
        self._sym_encoder = MLP(symbol_size * 3, symbol_enc_size, output_activation=nn.ReLU)
        self._reachable_encoder = MLP(feat_size * 2 + symbol_enc_size, 64, layer_dims=(128,), output_activation=nn.ReLU)
        self._reachable = MLP(64, 2, [64])

    def forward(self, entity_enc, focus_goal):
        sym_enc = time_distributed(tu.flatten(focus_goal, begin_axis=2), self._sym_encoder)
        focus_enc = self._reachable_encoder(torch.cat([entity_enc, sym_enc], dim=-1))
        reachable_enc_red, _ = torch.max(focus_enc, dim=-2)
        return self._reachable(reachable_enc_red)


class SatisfiedNet(nn.Module):
    def __init__(self, feat_size, symbol_size, hidden_dims, symbol_enc_size=32):
        super(SatisfiedNet, self).__init__()
        self._sym_encoder = MLP(symbol_size * 2, symbol_enc_size, output_activation=nn.ReLU)
        self._satisfied = MLP(feat_size * 2 + symbol_enc_size, 2, hidden_dims)

    def forward(self, sat_enc, sat_info):
        sym_enc = self._sym_encoder(sat_info)
        return self._satisfied(torch.cat((sat_enc, sym_enc), dim=-1))


class DependencyNet(nn.Module):
    def __init__(self, feat_size, symbol_size, hidden_dims, symbol_enc_size=32):
        super(DependencyNet, self).__init__()
        self._sym_encoder = MLP(symbol_size * 4, symbol_enc_size, output_activation=nn.ReLU)
        self._dependency = MLP(feat_size * 4 + symbol_enc_size, 2, hidden_dims)

    def forward(self, dep_enc, dep_info):
        sym_enc = self._sym_encoder(dep_info)
        return self._dependency(torch.cat((dep_enc, sym_enc), dim=-1))


class VBP(BCBP):
    """Regression Planning Networks."""
    def __init__(self, **kwargs):
        Net.__init__(self, **kwargs)
        c = self.c
        n_entities = c.n_object * (c.n_object + 1)
        input_size = c.im_enc_size * c.n_object
        symbol_size = c.symbol_size * n_entities * 3

        self._state_encoder = ImageEncoder(IMAGE_SIZE, c.im_enc_size)

        self._preimage = MLP(input_size + symbol_size, symbol_size, c.hidden_dims)
        self._reachable = ReachableNet(c.im_enc_size, c.symbol_size)

        self._satisfied = SatisfiedNet(c.im_enc_size, c.symbol_size, c.hidden_dims)
        self._dependency = DependencyNet(c.im_enc_size, c.symbol_size, c.hidden_dims)

        src, tgt = torch.meshgrid(torch.arange(c.n_object), torch.arange(c.n_object))
        self.register_buffer('edge_src', src.contiguous().view(-1))
        self.register_buffer('edge_tgt', tgt.contiguous().view(-1))
        self._object_feat_pad = nn.ConstantPad1d((0, c.im_enc_size), 0)

    def batch_entity_features(self, object_feat):
        """
        :param object_feat: [B, N, D]
        :return: [B, N * (N + 1), D * 2]
        """
        edge_src = self.edge_src.view(1, -1).expand(object_feat.shape[0], -1)
        edge_tgt = self.edge_tgt.view(1, -1).expand(object_feat.shape[0], -1)
        src_feat = tu.gather_sequence_n(object_feat, edge_src)
        tgt_feat = tu.gather_sequence_n(object_feat, edge_tgt)
        edge_feat = torch.cat([src_feat, tgt_feat], dim=-1)
        object_feat_padded = self._object_feat_pad(object_feat)
        return torch.cat([object_feat_padded, edge_feat], dim=1)

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

        object_feat = time_distributed(object_states, self._state_encoder)
        entity_feat = self.batch_entity_features(object_feat)
        planner_out = self.dep_sat(entity_feat, satisfied_info, dependency_info, num_entities)
        planner_out.update(self.backward_plan(entity_feat, object_feat, focus_goal, graph))
        return planner_out

    def forward_sat(self, sat_states, sat_info):
        return self._satisfied(sat_states, sat_info)

    def forward_dep(self, dep_states, dep_info):
        return self._dependency(dep_states, dep_info)

    def forward_policy(self, object_states, entity_states, goal, graphs=None):
        object_feat = time_distributed(object_states, self._state_encoder)
        entity_feat = self.batch_entity_features(object_feat)
        planner_out = self.find_subgoal(object_feat, entity_feat, goal, graphs)
        if self.verbose and planner_out['ret'] != -1:
            print(planner_out['ret'])

        subgoal = planner_out['subgoal']
        if subgoal is None:
            return planner_out
        # TODO: policy
        return planner_out

    def backward_plan(self, entity_states, object_states, focus_goal, graph=None):
        inputs_focus = torch.cat((tu.flatten(object_states), tu.flatten(focus_goal)), dim=-1)
        preimage_preds = self._preimage(inputs_focus)
        reachable_preds = self._reachable(entity_states, focus_goal)
        return {
            'preimage_preds': preimage_preds.reshape(focus_goal.shape[0], -1, self.c.symbol_size, 3),
            'reachable_preds': reachable_preds,
        }


class VGreedyBP(VBP):
    """SS-only."""
    def __init__(self, **kwargs):
        Net.__init__(self, **kwargs)
        c = self.c
        n_entities = c.n_object * (c.n_object + 1)
        input_size = c.im_enc_size * c.n_object
        symbol_size = c.symbol_size * n_entities * 3

        self._state_encoder = ImageEncoder(IMAGE_SIZE, c.im_enc_size)

        self._subgoal = MLP(input_size + symbol_size, symbol_size, c.hidden_dims)

        self._satisfied = SatisfiedNet(c.im_enc_size, c.symbol_size, c.hidden_dims)
        self._dependency = DependencyNet(c.im_enc_size, c.symbol_size, c.hidden_dims)

        src, tgt = torch.meshgrid(torch.arange(c.n_object), torch.arange(c.n_object))
        self.register_buffer('edge_src', src.contiguous().view(-1))
        self.register_buffer('edge_tgt', tgt.contiguous().view(-1))
        self._object_feat_pad = nn.ConstantPad1d((0, c.im_enc_size), 0)

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


class VMonoBP(VBP):
    """RP-only."""
    def __init__(self, **kwargs):
        Net.__init__(self, **kwargs)
        c = self.c
        n_entities = c.n_object * (c.n_object + 1)
        input_size = c.im_enc_size * c.n_object
        symbol_size = c.symbol_size * n_entities * 3

        self._state_encoder = ImageEncoder(IMAGE_SIZE, c.im_enc_size)

        self._focus = MLP(input_size + symbol_size, symbol_size, c.hidden_dims)
        self._preimage = MLP(input_size + symbol_size, symbol_size, c.hidden_dims)
        self._reachable = ReachableNet(c.im_enc_size, c.symbol_size)

        src, tgt = torch.meshgrid(torch.arange(c.n_object), torch.arange(c.n_object))
        self.register_buffer('edge_src', src.contiguous().view(-1))
        self.register_buffer('edge_tgt', tgt.contiguous().view(-1))
        self._object_feat_pad = nn.ConstantPad1d((0, c.im_enc_size), 0)

    def focus(self, entity_states, object_states, goal):
        inputs = torch.cat((tu.flatten(object_states), tu.flatten(goal)), dim=-1)
        focus_preds = self._focus(inputs)
        return {
            'focus_preds': focus_preds.reshape(goal.shape[0], -1, self.c.symbol_size, 3),
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
        object_feat = time_distributed(object_states, self._state_encoder)
        entity_feat = self.batch_entity_features(object_feat)
        planner_out = self.focus(entity_feat, object_feat, goal)
        planner_out.update(self.backward_plan(entity_feat, object_feat, focus_goal, graph))
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


class VSG(VBP):
    """E2E baseline."""
    def __init__(self, **kwargs):
        Net.__init__(self, **kwargs)
        c = self.c
        n_entities = c.n_object * (c.n_object + 1)
        input_size = c.im_enc_size * c.n_object
        symbol_size = c.symbol_size * n_entities * 3

        self._state_encoder = ImageEncoder(IMAGE_SIZE, c.im_enc_size)
        self._subgoal = MLP(input_size + symbol_size, symbol_size, c.hidden_dims)

    def find_subgoal(self, object_state, entity_state, goal, graphs, max_depth=10):
        return {
            'subgoal': tu.to_onehot(self.plan(object_state, goal)['subgoal_preds'].argmax(-1), 3),
            'ret': -1
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
        object_feat = time_distributed(object_states, self._state_encoder)
        planner_out = self.plan(object_feat, goal)
        return planner_out

    def forward_policy(self, object_states, entity_states, goal, graphs=None):
        object_feat = time_distributed(object_states, self._state_encoder)
        planner_out = self.find_subgoal(object_feat, None, goal, graphs)
        if self.verbose and planner_out['ret'] != -1:
            print(planner_out['ret'])

        subgoal = planner_out['subgoal']
        if subgoal is None:
            return planner_out
        # TODO: policy
        return planner_out

    def plan(self, object_states, goal):
        inputs = torch.cat((tu.flatten(object_states), tu.flatten(goal)), dim=-1)
        sg_preds = self._subgoal(inputs)
        return {
            'subgoal_preds': sg_preds.reshape(goal.shape[0], -1, self.c.symbol_size, 3),
        }

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