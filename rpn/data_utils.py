import numpy as np
import os
import os.path as osp
import json
import torch
from rpn.db import SequenceDB, SAMPLE, SEQUENCE
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import default_collate
from rpn.utils.torch_graph_utils import construct_full_graph, get_edge_features
import rpn.utils.torch_utils as tu
import rpn.utils.np_utils as npu
import deepdish
import h5py
from rpn.utils.timer import Timers
from rpn.utils.config import dict_to_namedtuple


def sample_actions(num_samples, num_objects, num_actions):
    # TODO: merge this to make_action_graph_inputs
    samples = np.zeros((num_samples, num_objects), dtype=np.int64)
    rand_obj = np.random.randint(0, num_objects, size=num_samples)
    # action 0 is noop
    rand_act = np.random.randint(1, num_actions, size=num_samples)
    samples[np.arange(num_samples), rand_obj] = rand_act
    return samples


def make_action_graph_inputs(serialized_action, object_ids, num_nodes):
    object_arg_index = np.array([object_ids.index(a) for a in serialized_action[3:]])
    action_index = serialized_action[1]
    action_inputs = np.zeros(num_nodes, dtype=np.int64)
    # action is indexed by both action_index and arg index
    if len(object_arg_index) > 0:
        action_inputs[object_arg_index] = action_index + np.arange(len(object_arg_index))
    return action_inputs, object_arg_index


def to_graph_dense(inputs, input_ids, input_len, id_to_index_map):
    s = (input_len,) + inputs.shape[1:]
    dense_inputs = np.zeros(s, dtype=inputs.dtype)
    map_index = np.array([id_to_index_map[iid] for iid in input_ids])
    dense_inputs[map_index, ...] = inputs
    return dense_inputs


def collate_samples(samples, concatenate=False, exclude_keys=()):
    batch = {}
    for k in samples[0].keys():
        if k in exclude_keys:
            batch[k] = [s[k] for s in samples]
        # elif isinstance(samples[0][k], Data):
        #     batch[k] = Batch.from_data_list([s[k] for s in samples])
        elif concatenate and len(samples[0][k].shape) > 1:
            batch[k] = torch.cat([s[k] for s in samples], dim=0)
        else:
            batch[k] = default_collate([s[k] for s in samples])
    return batch


def get_keyframes(symbol_sequence):
    """
    Find frames where symbol state changes
    :param symbol_sequence:
    :return:
    """
    symbol_sequence = symbol_sequence.reshape((symbol_sequence.shape[0], -1))
    keyframes = np.any(symbol_sequence[:-1] != symbol_sequence[1:], axis=1)
    keyframes = np.where(keyframes)[0] + 1
    assert(np.all(keyframes > 0) and np.all(keyframes < symbol_sequence.shape[0]))
    return keyframes


def make_gt_inputs(state, current_unitary, node_index, edge_index, id_to_index_map):
    # construct inputs
    # append action inputs to nodes
    # append the current unitary state to input since they are only visible
    current_unitary_dense = to_graph_dense(
        current_unitary[:, 1:], current_unitary[:, 0], node_index.shape[0], id_to_index_map)
    node_inputs = np.concatenate([state, current_unitary_dense], axis=1)

    edge_inputs = get_edge_features(node_inputs, edge_index, lambda a, b: np.concatenate([a, b], axis=1))
    # edge_inputs = get_edge_poses(state, edge_index)
    return node_inputs, edge_inputs


def make_graph_labels(sym_unitary, sym_binary, node_index, edge_index, id_to_index_map):
    # dense labels
    unitary_dense = to_graph_dense(
        sym_unitary[:, 1:], sym_unitary[:, 0], node_index.shape[0], id_to_index_map)

    binary_index, binary_val = sym_binary[:, :2], sym_binary[:, 2:]
    binary_index = [(s, t) for s, t in binary_index.astype(np.int64)]
    binary_dense = to_graph_dense(
        binary_val, binary_index, edge_index.shape[0], id_to_index_map)

    assert (sym_unitary.shape[0] == node_index.shape[0])
    assert (sym_binary.shape[0] == edge_index.shape[0])
    return unitary_dense, binary_dense


def get_id_to_graph_index_map(object_ids, node_index, edge_index):
    """
    Get object id to graph index map
    :param object_ids: a list object ids
    :param node_index: index of each node
    :param edge_index: index of edges
    :return: a dictionary mapping object id (or pairs) to graph index
    """
    id_to_index_map = {}
    for i, ni in enumerate(node_index):
        id_to_index_map[object_ids[ni]] = i

    for i, (s, t) in enumerate(edge_index):
        id_to_index_map[(object_ids[s], object_ids[t])] = i
    return id_to_index_map


def get_edge_poses(object_poses, edge_index):
    edge_poses = np.zeros((edge_index.shape[0], 7), dtype=object_poses.dtype)
    s_idx = edge_index[:, 0]
    t_idx = edge_index[:, 1]
    edge_poses[:, :3] = object_poses[s_idx, :3] - object_poses[t_idx, :3]
    edge_poses[:, 3:7] = object_poses[s_idx, 3:]
    return edge_poses


def get_sampler_without_eos(dataset, random=True):
    index = dataset.db.index_without_eos()
    return SubsetSampler(index, random=random)


class SubsetSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    :param: indices (sequence): a sequence of indices
    """

    def __init__(self, indices, random=True):
        self.indices = indices
        self.random = random

    def __iter__(self):
        if self.random:
            return (self.indices[i] for i in torch.randperm(len(self.indices)))
        else:
            return (self.indices[i] for i in torch.arange(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class DDWrapper(object):
    """
    A dict-like DeepDish wrapper for partially loading an hdf5 file
    """
    def __init__(self, filename, perm_load_keys=(), prefix='/'):
        self._fn = filename
        self._perm_dict = {}
        self._prefix = prefix
        self._perm_load_keys = perm_load_keys
        for k in self._perm_load_keys:
            if '/' in k:
                continue
            self._perm_dict[prefix + k] = deepdish.io.load(self._fn, prefix + k)

    def __getitem__(self, key):
        if not isinstance(key, str):
            return deepdish.io.load(self._fn, self._prefix, sel=deepdish.aslice[key])
        elif self._prefix + key in self._perm_dict:
            return self._perm_dict[self._prefix + key]
        else:
            new_keys = []
            for k in self._perm_load_keys:
                if '/' in k:
                    ks = k.split('/')
                    assert(ks[0] == key)
                    new_keys.append('/'.join(ks[1:]))
            return self.__class__(self._fn, perm_load_keys=new_keys, prefix=self._prefix + key + '/')


class BasicDataset(Dataset):
    """Basic dataset for iterating task demonstration data."""
    def __init__(self, db, **kwargs):
        super(BasicDataset, self).__init__()
        self.db = db
        self.config = dict_to_namedtuple(kwargs)
        self.timers = Timers()
        seed = kwargs.get('seed', 0)
        self._npr = np.random.RandomState(seed)
        self._first = True

    def __len__(self):
        return self.db.num_samples(SAMPLE)  # number of samples

    def __str__(self):
        return 'num_sample=%i, avg_sequence_length=%f' % \
               (len(self), float(self.db.average_sequence_len()))

    def reload_data(self):
        self.db = self.load(self.config.data_file).db

    @classmethod
    def load(cls, data_file, **kwargs):
        print('loading data from %s' % data_file)
        if data_file.endswith('.h5'):
            rc = deepdish.io.load(data_file)
            db = SequenceDB.deserialize(rc)
        elif data_file.endswith('.h5p'):
            h5f = h5py.File(data_file, 'r')
            sample_len = json.loads(str(np.array(h5f['sample_len'])))
            rc = {
                'db': h5f['db'],
                'sample_len': sample_len
            }
            db = SequenceDB.deserialize(rc)
        elif data_file.endswith('.group'):
            db_list = []
            for f in sorted(os.listdir(data_file)):
                db_list.append(BasicDataset.load(osp.join(data_file, f)).db)
            db = GroupDB(db_list)
        else:
            raise NotImplementedError(data_file)
        kwargs['data_file'] = data_file
        return cls(db, **kwargs)

    def dump(self, data_file):
        db = self.db.serialize()
        if data_file.endswith('.h5'):
            deepdish.io.save(data_file, db)
        elif data_file.endswith('.h5p'):
            h5f = h5py.File(data_file, 'w')
            dbg = h5f.create_group('db')
            for k, v in db['db'].items():
                dbg.create_dataset(k, data=v)
            h5f.create_dataset('sample_len', data=json.dumps(db['sample_len']))
        else:
            raise NotImplementedError
        print('data saved to %s' % data_file)


class GroupDB(object):
    """A wrapper for iterating a group of datasets. Takes a list of BasicDataset as input."""
    def __init__(self, db_list):
        self.db_list = db_list
        self.index_range_to_db = []
        top = 0
        for db in db_list:
            dblen = db.num_samples(SAMPLE)
            self.index_range_to_db.append((top, top + dblen))
            top += dblen
        print(self.index_range_to_db)

    def num_samples(self, level):
        return self.index_range_to_db[-1][1]

    def average_sequence_len(self):
        accum = []
        for db in self.db_list:
            accum.append(db.sample_len[SEQUENCE]['actions'])
        return np.mean(np.hstack(accum))

    def translate_index(self, index):
        for i, (begin, end) in enumerate(self.index_range_to_db):
            if begin <= index < end:
                return i, begin
        raise IndexError

    def get_db_item(self, key, index):
        db_idx, begin_idx = self.translate_index(index)
        return self.db_list[db_idx].get_db_item(key, index - begin_idx)

    def get_db_item_list(self, key, index, end_index):
        db_idx, begin_idx = self.translate_index(index)
        return self.db_list[db_idx].get_db_item_list(key, index - begin_idx, end_index - begin_idx)

    def sequence_index_from_sample_index(self, index):
        db_idx, begin_idx = self.translate_index(index)
        return self.db_list[db_idx].sequence_index_from_sample_index(index - begin_idx)

    def eos_index_from_sample_index(self, index):
        db_idx, begin_idx = self.translate_index(index)
        seq_i = self.db_list[db_idx].sequence_index_from_sample_index(index - begin_idx)
        bi = self.db_list[db_idx].sequence_begin_index(seq_i)
        gi = bi + self.db_list[db_idx].sequence_length(seq_i)
        return gi + begin_idx

    def index_without_eos(self):
        """
        Compute index of database such that End of Sequence (EoS) is not included
        :param db: database from db.py
        :return: indices in 1-D array
        """
        all_index = np.arange(self.num_samples(SAMPLE))
        k = list(self.db_list[0].data_keys)[0]
        sample_end_index = []
        for db, (begin_idx, end_idx) in zip(self.db_list, self.index_range_to_db):
            sample_end_index.append(
                begin_idx + db.sample_begin_index[SEQUENCE][k] + db.sample_len[SEQUENCE][k] - 1
            )
        return np.setdiff1d(all_index, np.hstack(sample_end_index))


class PreimageDataset(BasicDataset):
    """Dataset with precondition and dependency for training RPN."""
    def __init__(self, db, **kwargs):
        super(PreimageDataset, self).__init__(db, **kwargs)
        self._graph_edges = {}
        self._gnn_edges = {}

    def graph_edges(self, num_nodes):
        if num_nodes not in self._graph_edges:
            self._graph_edges[num_nodes] = construct_full_graph(num_nodes, self_connection=True)[1]
        return self._graph_edges[num_nodes]

    def gnn_edges(self, num_nodes):
        if num_nodes not in self._gnn_edges:
            self._gnn_edges[num_nodes] = construct_full_graph(num_nodes, self_connection=False)[1]
        return self._gnn_edges[num_nodes]

    def get_plan_sample(self, index):
        debug = hasattr(self.config, 'debug') and self.config.debug
        if debug:
            print('WARNING!!!!!!!!!!!! DEBUG MODE!')

        with self.timers.timed('goal_trace'):
            num_goal_entities = self.db.get_db_item('num_goal_entities', index)
            goal_split = np.cumsum(num_goal_entities)[:-1]
            goal_trace = self.db.get_db_item('goal_trace', index).astype(np.float32)
            goal_mask_trace = self.db.get_db_item('goal_mask_trace', index).astype(np.float32)
            goal_trace = np.split(goal_trace, goal_split, axis=0)
            goal_mask_trace = np.split(goal_mask_trace, goal_split, axis=0)
            assert(len(goal_trace) == num_goal_entities.shape[0])

            # pick a random step in the trace
            trace_len = len(goal_trace)
            trace_idx = int(self._npr.randint(0, trace_len))

            goal = goal_trace[trace_idx]
            goal_mask = goal_mask_trace[trace_idx]

            # find preimage, dummy if trace_idx is the last step
            if trace_idx < trace_len - 1:
                preimage = goal_trace[trace_idx + 1]
                preimage_mask = goal_mask_trace[trace_idx + 1]
                preimage_loss_mask = np.ones_like(preimage)
            else:
                # end of the trace
                preimage = np.zeros_like(goal_trace[trace_idx])
                preimage_mask = np.zeros_like(goal_mask_trace[trace_idx])
                preimage_loss_mask = np.zeros_like(preimage_mask)

            reachable = self.db.get_db_item('reachable_trace', index)[trace_idx].astype(np.float32)
            focus_trace = self.db.get_db_item('focus_trace', index).astype(np.float32)
            focus_trace = np.split(focus_trace, goal_split, axis=0)
            focus_mask = focus_trace[trace_idx]

            if debug:
                reachable = self.db.get_db_item('reachable_trace', index).astype(np.float32)
                goal = goal_trace
                goal_mask = goal_mask_trace
                focus_mask = focus_trace

            # ground truth subgoal for the policy, use the last reachable goals as the goal mask
            subgoal = goal_trace[-1]
            subgoal_mask = focus_trace[-1]

        with self.timers.timed('sat_deps'):
            # satisfied and dependency have their own inputs
            sat_trace = self.db.get_db_item('satisfied_trace', index)
            ri = int(self._npr.randint(0, sat_trace.shape[0]))
            last_step = np.array(sat_trace[ri, 0] == sat_trace[-1, 0]).astype(np.float32)
            satisfied = sat_trace[ri, 1:].astype(np.float32)
            if debug:
                satisfied = sat_trace[:, 1:].astype(np.float32)
            # dependencies
            deps_trace = self.db.get_db_item('dependency_trace', index)
            ri = int(self._npr.randint(0, deps_trace.shape[0]))
            deps = deps_trace[ri, 1:].astype(np.float32)
            deps_loss_mask = np.array(np.any(deps > 0)).astype(np.float32)
            if debug:
                deps = deps_trace[:, 1:].astype(np.float32)

        sample = {
            'goal': goal,
            'goal_mask': goal_mask,
            'focus_mask': focus_mask,
            'satisfied': satisfied,
            'reachable': np.array(reachable),
            'preimage': preimage,
            'preimage_mask': preimage_mask,
            'preimage_loss_mask': preimage_loss_mask,
            'subgoal': subgoal,
            'subgoal_mask': subgoal_mask,
            'dependency': deps,
            'dependency_loss_mask': deps_loss_mask,
            'last_step': last_step,
        }
        return tu.batch_to_tensor(sample)


class GridPreimageDataset(PreimageDataset):
    def __getitem__(self, index):
        seq_i = self.db.sequence_index_from_sample_index(index)
        with self.timers.timed('state'):
            action = self.db.get_db_item('actions', index)
            current_state = self.db.get_db_item('object_state_flat', index).astype(np.float32)

        plan_sample = self.get_plan_sample(index)
        with self.timers.timed('sample'):
            sample = {
                'states': tu.to_tensor(current_state),
                'action_labels': tu.to_tensor(np.array(action[0])),
                'num_entities': tu.to_tensor(np.array(current_state.shape[0])),
                'seq_idx': seq_i
            }
        sample.update(plan_sample)
        return sample


def make_bullet_gt_input(current_state, sym_state, full_edges, object_types, num_types):
    num_objects = current_state.shape[0]
    edge_state = get_edge_poses(current_state, full_edges)
    assert (edge_state.shape[0] == num_objects ** 2)
    assert (sym_state.shape[0] == (num_objects * (num_objects + 1)))
    type_onehot = npu.to_onehot(object_types, num_types)
    current_state = np.concatenate(
        [type_onehot, np.zeros_like(type_onehot), sym_state[:num_objects], current_state], axis=-1
    )
    edge_state = np.concatenate(
        [type_onehot[full_edges[:, 0]], type_onehot[full_edges[:, 1]], sym_state[num_objects:], edge_state], axis=-1
    )
    assert(edge_state.shape[0] == num_objects ** 2)
    return current_state, np.concatenate([current_state, edge_state], axis=0)


class BulletPreimageDataset(PreimageDataset):
    """Dataset with ground truth state inputs. Not used in the experiments."""
    def __getitem__(self, index):
        if self._first:
            self._first = False
        seq_i = self.db.sequence_index_from_sample_index(index)
        with self.timers.timed('state'):
            current_state = self.db.get_db_item('gt_state', index)
            num_objects = current_state.shape[0]
            full_edges = self.graph_edges(num_objects)
            types = self.db.get_db_item('object_type_indices', index)
            num_types = self.db.get_db_item('num_object_types', index)[0]
            sym_state = self.db.get_db_item('symbolic_state', index)
            current_state, entity_states = make_bullet_gt_input(
                current_state, sym_state, full_edges, types, num_types
            )

        plan_sample = self.get_plan_sample(index)
        with self.timers.timed('sample'):
            sample = {
                'states': tu.to_tensor(current_state),
                'entity_states': tu.to_tensor(entity_states),
                'num_entities': tu.to_tensor(np.array(entity_states.shape[0])),
                'seq_idx': seq_i
            }
        sample.update(plan_sample)
        return sample


class BulletPreimageVisualDataset(PreimageDataset):
    """Preimage data with visual inputs."""
    def __getitem__(self, index):
        if self._first:
            self._first = False
        seq_i = self.db.sequence_index_from_sample_index(index)
        plan_sample = self.get_plan_sample(index)
        with self.timers.timed('state'):
            current_state = self.db.get_db_item('image_crops', index)
            current_state = current_state.transpose((0, 3, 1, 2)).astype(np.float32) / 255.
        with self.timers.timed('sample'):
            sample = {
                'states': tu.to_tensor(current_state),
                'seq_idx': seq_i
            }
        sample.update(plan_sample)
        return sample


class BulletVisualDataset(BasicDataset):
    def __getitem__(self, index):
        if self._first:
            self._first = False
        seq_i = self.db.sequence_index_from_sample_index(index)
        with self.timers.timed('state'):
            current_state = self.db.get_db_item('image_crops', index)
            current_state = current_state.transpose((0, 3, 1, 2)).astype(np.float32) / 255.
            next_state = self.db.get_db_item('image_crops', index + 1)
            next_state = next_state.transpose((0, 3, 1, 2)).astype(np.float32) / 255.

        num_actions = 4
        num_objects = current_state.shape[0]
        action = self.db.get_db_item('actions', index)
        action_label = np.zeros(num_objects * num_actions, dtype=np.float32)
        action_label[action[2]] = 1
        action_label = action_label.reshape((num_objects, num_actions))
        with self.timers.timed('sample'):
            sample = {
                'states': tu.to_tensor(current_state),
                'next_states': tu.to_tensor(next_state),
                'action_labels': tu.to_tensor(action_label),
                'seq_idx': seq_i
            }
        return sample


class BulletPlanVisualDataset(BasicDataset):
    """Dataset for visual MPC"""
    def __getitem__(self, index):
        if self._first:
            self._first = False
        seq_i = self.db.sequence_index_from_sample_index(index)
        with self.timers.timed('state'):
            current_state = self.db.get_db_item('image_crops', index)
            current_state = current_state.transpose((0, 3, 1, 2)).astype(np.float32) / 255.

        num_actions = 4
        num_objects = current_state.shape[0]
        gi = self.db.eos_index_from_sample_index(index)
        assert(index + 1 < gi and gi - index < 100)
        gi = int(np.random.randint(index + 2, gi + 1))
        action_sequence = self.db.get_db_item_list('actions', index, gi)
        seq_len = gi - index
        action_sequence_label = np.zeros((seq_len, num_objects * num_actions), dtype=np.float32)
        for i, a in enumerate(action_sequence):
            action_sequence_label[i][a[2]] = 1
        action_sequence_label = action_sequence_label.reshape((seq_len, num_objects, num_actions))

        next_state = self.db.get_db_item('image_crops', gi - 1)
        next_state = next_state.transpose((0, 3, 1, 2)).astype(np.float32) / 255.

        with self.timers.timed('sample'):
            sample = {
                'states': tu.to_tensor(current_state),
                'next_states': tu.to_tensor(next_state),
                'action_sequence': tu.to_tensor(action_sequence_label),
                'seq_idx': seq_i
            }
        return sample


def main():
    pass


if __name__ == '__main__':
    main()
