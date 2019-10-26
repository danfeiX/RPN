from collections import OrderedDict
import numpy as np
from third_party.pybullet.utils.pybullet_tools.utils import wait_for_duration
from third_party.pybullet.utils.pybullet_tools.kuka_primitives import set_pose, BodyConf
from builtins import int


class PBGoal(object):
    """Goal definition for pybullet environment."""
    def __init__(self, predicate, value, type1, name1, type2=None, name2=None, satisfied=False, is_alias=False):
        """
        Goal definition for pybullet environment. Can represent both objects and relationships
        :param predicate: predicate name
        :param value: boolean value
        :param type1: type of the entity 1
        :param name1: name of the entity 1
        :param type2: type of the entity 2 (if relationship)
        :param name2: name of the entity 2 (if relationshi)
        :param satisfied: if the goal is satisfied (default False)
        :param is_alias: if is used as an alias in parsing dependencies.
        """
        assert(value in [True, False])
        assert(name1 is not None and type1 is not None)
        assert((name2 is not None and type2 is not None) or (name2 is None and type2 is None))
        self.type1 = type1
        self.name1 = name1
        self.type2 = type2
        self.name2 = name2
        self.value = value
        self.satisfied = satisfied
        self.predicate = predicate
        self.is_alias = is_alias

    @property
    def is_relationship(self):
        return self.name2 is not None

    @property
    def keys(self):
        return ['predicate', 'value', 'type1', 'name1', 'type2', 'name2', 'satisfied', 'is_alias']

    @property
    def comp_keys(self):
        return ['predicate', 'value', 'name1', 'name2']

    def make_satisfied(self):
        self.satisfied = True

    def __eq__(self, other):
        for k in self.comp_keys:
            if getattr(self, k) != getattr(other, k):
                return False
        return True

    def __repr__(self):
        msg = []
        for k in ['predicate', 'value', 'name1', 'name2', 'satisfied']:
            msg.append('%r' % getattr(self, k))
        # return '(' + ', '.join(msg) + ')' + ' is_alias=%r' % self.is_alias
        return '(' + ', '.join(msg) + ')'
        # msg = self.predicate + '(' + self.type1
        # if self.is_relationship:
        #     msg += ', ' + self.type2
        #
        # msg += ')'
        # return msg

    def __copy__(self):
        return self.clone()

    def __deepcopy__(self, memodict={}):
        return self.clone()

    def clone(self):
        cd = {}
        for k in self.keys:
            cd[k] = getattr(self, k)
        return self.__class__(**cd)


class TaskEnv(object):
    def __init__(self, objects):
        self._objects = objects
        self._symbolic_state = OrderedDict()
        self.prev_state = None
        self._predicate_funcs = None
        self._action_funcs = None
        assert(len(self.action_names()) == len(self.num_action_args()))
        self._symbolic_state = self.symbolic_state_template()
        assert(self.action_names()[0] == 'noop')

    @property
    def objects(self):
        return self._objects

    @property
    def entities(self):
        ents = [(o,) for o in self._objects]
        for o1 in self._objects:
            for o2 in self._objects:
                ents.append((o1, o2))
        return ents

    @property
    def default_fluents(self):
        return {}

    @staticmethod
    def action_names():
        raise NotImplementedError

    @staticmethod
    def num_action_args():
        raise NotImplementedError

    @staticmethod
    def unitary_predicates():
        raise NotImplementedError

    @staticmethod
    def binary_predicates():
        raise NotImplementedError

    def predicates(self):
        return self.unitary_predicates() + self.binary_predicates()

    @classmethod
    def get_action_index(cls, action_name):
        return cls.action_names().index(action_name)

    def get_arg_serialized_action_index(self, action_name):
        """
        Get serialized index of the action spaced by the number of arguments
        :param action_name: name of the action
        :return: serialized index of the action
        """
        begin_idx = np.cumsum([0] + self.num_action_args()[:-1])
        action_index = self.get_action_index(action_name)
        return begin_idx[action_index]

    def get_obj_serialized_action_index(self, action_name, object_args):
        """
        Get serialized index of the action spaced by the number of objects
        :param action_name: name of the actions
        :param object_args: list of object ids that are action arguments
        :return: serialized index of the action
        """
        begin_idx = np.cumsum([0] + [len(self.objects)] * len(self.action_names()))[:-1]
        action_index = self.get_action_index(action_name)
        # handle more than one object
        object_index = self.objects.ids.index(object_args[0])
        return begin_idx[action_index] + object_index

    @classmethod
    def get_action_name(cls, action_index):
        return cls.action_names()[action_index]

    def reset(self):
        self._symbolic_state = self.symbolic_state_template()
        self.update_predicates()
        assert(self.check_integrity())
        return self.symbolic_state

    def symbolic_state_template(self):
        ss = OrderedDict()
        for ent in self.entities:
            idk = (ent[0].uid,) if len(ent) == 1 else (ent[0].uid, ent[1].uid)
            ss[idk] = OrderedDict([(pn, False) for pn in self.predicates()])
        # assign default predicate values
        for d, v in self.default_fluents.items():
            assert(d in ss)
            for p, pv in v.items():
                assert(p in ss[d])
                ss[d][p] = pv
        return ss

    def _initialize_state(self):
        self._symbolic_state = self.symbolic_state_template()

    @property
    def symbolic_state(self):
        return self._symbolic_state.copy()

    @property
    def serialized_gt_state(self):
        return self.serialize_gt_state(self.objects)

    @property
    def serialized_visual_state(self):
        raise NotImplementedError()  # TODO: LOW

    def is_success(self, goals):
        for goal in goals:
            assert(isinstance(goal, PBGoal))
            if goal.is_relationship:
                sk = (self.objects[goal.name1].uid, self.objects[goal.name2].uid)
            else:
                sk = (self.objects[goal.name1].uid,)
            if self.symbolic_state[sk][goal.predicate] != goal.value:
                return False
        return True

    def num_success_goal(self, goals):
        n_success = 0
        for goal in goals:
            assert(isinstance(goal, PBGoal))
            if goal.is_relationship:
                sk = (self.objects[goal.name1].uid, self.objects[goal.name2].uid)
            else:
                sk = (self.objects[goal.name1].uid,)
            if self.symbolic_state[sk][goal.predicate] == goal.value:
                n_success += 1
        return n_success

    def step_command(self, action, time_step=0.0):
        # TODO: HIGH add action preconditions
        self.prev_state = self.symbolic_state.copy()
        name, args, command = action

        discrete_args, _ = self.parse_action(name, args)

        command = command.refine()
        is_last = False
        last_conf = BodyConf(self.objects.robot).configuration
        for i, body_path in enumerate(command.body_paths):
            for step_i, conf in body_path.iterator():
                wait_for_duration(time_step)
                assert(not is_last)
                is_last = step_i == (len(body_path.path) - 1) and i == (len(command.body_paths) - 1)
                if is_last:
                    # apply symbolic action to state if the last pose
                    # TODO: this might cause some trouble...
                    self.update_predicates()
                    af = self._action_funcs[name]
                    assert(af(*discrete_args))
                conf_diff = np.array(conf) - np.array(last_conf)
                last_conf = conf
                yield conf_diff, is_last

    def execute_command(self, action, time_step=0.0):
        self.prev_state = self.symbolic_state.copy()
        name, args, command = action
        discrete_args, _ = self.parse_action(name, args)
        # apply actual command
        conf = command.refine().execute(time_step=time_step)

        # apply symbolic action to state
        af = self._action_funcs[name]
        assert(af(*discrete_args))
        self.update_predicates()
        integrity = self.check_integrity()
        if not integrity:
            print('integrity check failed')
        return integrity

    @property
    def serialized_symbolic_state(self):
        return self.serialize_symbolic_state(self.symbolic_state)

    def serialize_symbolic_state(self, sym_state):
        sss = np.zeros((len(sym_state), len(self.predicates())), dtype=np.float32)
        for i, id_key in enumerate(sym_state):
            for j, (p_name, pv) in enumerate(sym_state[id_key].items()):
                sss[i, j] = float(pv)
        return sss

    def deserialize_symbolic_state(self, serialized_sym_state):
        ss_dict = self.symbolic_state_template()
        assert(len(serialized_sym_state) == len(ss_dict))
        for ss, (id_key, predicates) in zip(serialized_sym_state, ss_dict.items()):
            for s_val, p_name in zip(ss, predicates.keys()):
                predicates[p_name] = s_val
        return ss_dict

    def deserialize_satisfied_entry(self, satisfied):
        assert(satisfied.shape == (len(self.predicates()) * 2 + 2,))
        predicates = self.predicates()
        num_p = len(predicates)
        oi = int(satisfied[0])
        dp = satisfied[1:-1].reshape((2, num_p))
        pi = int(dp[1].argmax())
        pv = bool(dp[0][pi])
        o = self.entities[oi]
        sat = bool(satisfied[-1])
        names = [o[0].name] if len(o) == 1 else [o[0].name, o[1].name]
        return names + [predicates[pi], pv, sat]

    def deserialize_dependency_entry(self, dependency):
        predicates = self.predicates()
        num_p = len(predicates)
        oi1, oi2 = dependency[0:2]
        dp = dependency[2:-1].reshape((4, num_p))
        pi1 = int(dp[1].argmax())
        pv1 = bool(dp[0][pi1])
        pi2 = int(dp[3].argmax())
        pv2 = bool(dp[2][pi2])
        o1 = self.entities[int(oi1)]
        o2 = self.entities[int(oi2)]
        dep = bool(dependency[-1])
        names1 = [o1[0].name] if len(o1) == 1 else [o1[0].name, o1[1].name]
        names2 = [o2[0].name] if len(o2) == 1 else [o2[0].name, o2[1].name]
        return names1 + [predicates[pi1], pv1] + names2 + [predicates[pi2], pv2, dep]

    def deserialize_goals(self, goal_value, goal_mask):
        val_ss = self.deserialize_symbolic_state(goal_value)
        mask_ss = self.deserialize_symbolic_state(goal_mask)
        goals = []
        for i, (id_key, predicates) in enumerate(mask_ss.items()):
            for p_name, p_val in predicates.items():
                if p_val > 0.5:
                    name1 = self.entities[i][0].name
                    name2 = self.entities[i][1].name if len(self.entities[i]) == 2 else None
                    g = PBGoal(p_name, val_ss[id_key][p_name] > 0.5, name1, name1, name2, name2)
                    goals.append(g)
        return goals

    @staticmethod
    def serialize_gt_state(objects):
        poses = np.zeros((len(objects), 7), dtype=np.float32)
        for oi, obj in enumerate(objects):
            pos, quat = obj.pose
            poses[oi, :3] = pos
            poses[oi, 3:] = quat
        return poses

    def set_pose_with_serialized_state(self, states):
        assert(states.shape[0] == len(self.objects))
        for o, pose in zip(self.objects, states):
            set_pose(o.uid, (pose[:3], pose[3:]))

    def parse_action(self, action_name, action_args):
        # TODO: implement continuous part for other functions
        discrete_args, cont_args = action_args[:-1], action_args[-1]
        for a in discrete_args:
            assert(a in self.objects.ids)
        assert(len(discrete_args) == self.num_action_args()[self.action_names().index(action_name)])
        return discrete_args, cont_args

    def serialize_action(self, action_name, action_args):
        action_args, cont_args = self.parse_action(action_name, action_args)
        action_index = self.get_action_index(action_name)
        action_index_with_args = self.get_arg_serialized_action_index(action_name)
        if action_name == 'place':
            action_index_with_objs = self.get_obj_serialized_action_index(action_name, action_args[1:])
        else:
            action_index_with_objs = self.get_obj_serialized_action_index(action_name, action_args)
        serialized_action = [action_index, action_index_with_args, action_index_with_objs] + action_args
        serialized_action = np.array(serialized_action, dtype=np.int64)
        cont_args = np.array(cont_args[0] + cont_args[1])
        return serialized_action, cont_args

    @staticmethod
    def find_goal_entity(goal, entities):
        for i, ent in enumerate(entities):
            if not goal.is_relationship and len(ent) == 1:
                if ent[0].name == goal.name1:
                    return i, ent
            if goal.is_relationship and len(ent) == 2:
                if ent[0].name == goal.name1 and ent[1].name == goal.name2:
                    return i, ent
        return -1, None

    def goal_mask(self, goals):
        entities = self.entities
        mask = np.zeros((len(entities), len(self.predicates())), dtype=np.float32)
        for goal in goals:
            ent_i, ent = self.find_goal_entity(goal, entities)
            assert(ent is not None)
            g_index = self.predicates().index(goal.predicate)
            mask[ent_i, g_index] = 1
        assert(int(mask.sum()) == len(goals))
        return mask

    def goal_to_symbolic_state(self, goals):
        ss = self.symbolic_state_template()
        entities = self.entities
        visited = []
        for goal in goals:
            ent_i, ent = self.find_goal_entity(goal, entities)
            assert(ent is not None)
            ks = (ent[0].uid,) if len(ent) == 1 else (ent[0].uid, ent[1].uid)
            assert(ks in ss)
            ss[ks][goal.predicate] = goal.value
            assert((ks + (goal.predicate,)) not in visited)
            visited.append(ks + (goal.predicate,))
        return ss

    def goal_symbolic_state_index(self, goal):
        ent_i, ent = self.find_goal_entity(goal, self.entities)
        assert(ent is not None)
        p_idx = self.predicates().index(goal.predicate)
        return ent_i, p_idx

    def masked_symbolic_state(self, masked_ss):
        out = []
        for i, (oid, d) in enumerate(masked_ss.items()):
            for k in d:
                if d[k] != 2:
                    out.append(([self.objects.name(o) for o in oid], k, d[k] == 1))
        return out

    @property
    def info(self):
        return {
            'action_names': self.action_names(),
            'object_ids': self.objects.ids,
            'type_indices': self.objects.object_type_indices,
            'types': self.objects.all_types,
            'state_size': 7 + len(self.predicates()) + len(self.objects.all_types) * 2,
            'num_action_args': self.num_action_args(),
            'unitary_predicates': self.unitary_predicates(),
            'binary_predicates': self.binary_predicates(),
            'predicates': self.predicates()
        }

    def update_predicates(self):
        for pf in self._predicate_funcs:
            pf()

    def _safe_apply(self, key, predicate, negated):
        """
        Safely change a predicate
        :param key: object associated with the predicate
        :param predicate: predicate name
        :param negated: if negating the predicate
        :return: True if successfully applied
        """
        if self._symbolic_state[key][predicate] == negated:
            self._symbolic_state[key][predicate] = not negated
            # print(key, predicate, not negated)
            return True
        else:
            return False

    def _apply(self, key, predicate, negated):
        # if self._sym_state[key][predicate] == negated:
        #     print(key, predicate, not negated)
        self._symbolic_state[key][predicate] = not negated

    def applicable(self, action_name, *args):
        raise NotImplementedError

    def check_integrity(self):
        return True
