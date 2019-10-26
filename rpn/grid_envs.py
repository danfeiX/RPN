from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.roomgrid import RoomGrid
from collections import OrderedDict
import time


OBJECT_CLASS = {
    'door': Door,
    'key': Key,
    'ball': Ball,
    'floor': Floor
}


class GridGoal(object):
    def __init__(self, predicate, value, type, color, satisfied=False, is_alias=False):
        assert(predicate in ['is_open', 'is_locked', 'holding', 'on'])
        assert(value in [True, False])
        self.type = type
        self.color = color
        self.value = value
        self.predicate = predicate
        self.satisfied = satisfied
        self.is_alias = is_alias

    @property
    def keys(self):
        return ['predicate', 'value', 'type', 'color', 'satisfied', 'is_alias']

    @property
    def comp_keys(self):
        return ['predicate', 'value', 'type', 'color']

    def make_satisfied(self):
        self.satisfied = True

    def __eq__(self, other):
        for k in self.comp_keys:
            if getattr(self, k) != getattr(other, k):
                return False
        return True

    def __repr__(self):
        msg = []
        for k in self.keys:
            msg.append('%r' % getattr(self, k))
        return '(' + ', '.join(msg) + ')'

    def __copy__(self):
        return self.clone()

    def __deepcopy__(self, memodict={}):
        return self.clone()

    def clone(self):
        cd = {}
        for k in self.keys:
            cd[k] = getattr(self, k)
        return self.__class__(**cd)


def open_door_inst(color):
    return [
        ['goto', 'facing', 'key', color],
        ['pickup'],
        ['goto', 'facing', 'door', color],
        ['toggle']
    ]


class InfoEnv(object):
    def check_grid(self):
        for j in range(self.height):
            for i in range(self.width):
                obj = self.grid.get(i, j)
                if obj is not None and obj.cur_pos is not None:
                    assert(same([i, j], obj.cur_pos))
        for oname, oinfo in self.object_info:
            if len(self.find(oname, oinfo['color'])) != 1:
                raise KeyError(print(self.find(oname, oinfo['color'])))

    def find(self, o_type, o_color):
        objs = []
        for o in self.grid.grid + [self.carrying]:
            if o is not None and o.type == o_type and o.color == o_color:
                objs.append(o)
        return objs

    @property
    def object_info(self):
        return self._object_info

    @property
    def action_list(self):
        return list(self.actions.__dict__['_member_map_'].keys())

    @property
    def objects(self):
        objects = []
        for oname, oinfo in self.object_info:
            obj = self.find(oname, oinfo['color'])
            if len(obj) == 0:
                objects.append(None)
            else:
                objects.append(obj[0])
        return objects

    @staticmethod
    def symbolic_state_template():
        return OrderedDict({
            'is_locked': False,
            'is_open': False,
            'on': False,
            'holding': False,
            'missing': False
        })

    @staticmethod
    def predicates():
        return ['is_locked', 'is_open', 'on', 'holding', 'missing']

    def symbolic_state(self, ignore_missing=False):
        state = []
        for o in self.objects:
            enc = self.symbolic_state_template()
            if o is None:
                if not ignore_missing:
                    enc['missing'] = True
                    state.append((None, None, enc))
                continue
            else:
                enc['is_locked'] = hasattr(o, 'is_locked') and o.is_locked
                enc['is_open'] = hasattr(o, 'is_open') and o.is_open
                enc['holding'] = self.carrying is not None and self.carrying == o
                enc['on'] = same(o.cur_pos, self.agent_pos)
                state.append((o.type, o.color, enc))
        return state

    def serialize_symbolic_state(self, sym_state=None, ignore_missing=False):
        if sym_state is None:
            sym_state = self.symbolic_state(ignore_missing)
        assert(len(sym_state) == len(self.objects))
        enc = np.zeros((len(sym_state), len(sym_state[0][2])), dtype=np.float32)
        for i, (_, _, ss) in enumerate(sym_state):
            for j, v in enumerate(ss.values()):
                enc[i, j] = float(v)
        return enc

    def deserialize_symbolic_state(self, sym_state, ignore_missing=False):
        assert(sym_state.shape[0] == len(self.objects))
        ss_list = []
        for ss, o in zip(sym_state, self.objects):
            if ignore_missing and o is None:
                continue
            o_info = [o.type, o.color] if o is not None else [None, None]
            state = self.symbolic_state_template()
            for k, v in zip(state.keys(), ss):
                state[k] = v
            ss_list.append(o_info + [state])
        return ss_list

    def deserialize_dependency(self, dependency):
        dd_list = []
        predicates = self.predicates()
        assert(dependency.shape[0] == dependency.shape[1] == len(self.objects))
        assert(dependency.shape[2] == dependency.shape[3] == len(self.predicates()))
        for i in range(dependency.shape[0]):
            o1 = self.objects[i]
            o1_info = [o1.type, o1.color] if o1 is not None else [None, None]
            for j in range(dependency.shape[1]):
                o2 = self.objects[j]
                o2_info = [o2.type, o2.color] if o2 is not None else [None, None]
                dd = {}
                for ii in range(dependency.shape[2]):
                    for jj in range(dependency.shape[3]):
                        dd[(predicates[ii], predicates[jj])] = dependency[i, j, ii, jj]
                dd_list.append([o1_info, o2_info, dd])
        return dd_list

    def deserialize_satisfied_entry(self, satisfied):
        predicates = self.predicates()
        num_p = len(predicates)
        oi = int(satisfied[0])
        dp = satisfied[1:-1].reshape((2, num_p))
        pi = int(dp[1].argmax())
        pv = bool(dp[0][pi])
        o = self.objects[oi]
        sat = bool(satisfied[-1])
        return [o.type, o.color, predicates[pi], pv, sat]

    def deserialize_dependency_entry(self, dependency):
        predicates = self.predicates()
        num_p = len(predicates)
        oi1, oi2 = dependency[0:2]
        dp = dependency[2:-1].reshape((4, num_p))
        pi1 = int(dp[1].argmax())
        pv1 = bool(dp[0][pi1])
        pi2 = int(dp[3].argmax())
        pv2 = bool(dp[2][pi2])
        o1 = self.objects[int(oi1)]
        o2 = self.objects[int(oi2)]
        dep = bool(dependency[-1])
        return [o1.type, o1.color, predicates[pi1], pv1, o2.type, o2.color, predicates[pi2], pv2, dep]

    def deserialize_goals(self, goal_value, goal_mask):
        val_ss = self.deserialize_symbolic_state(goal_value)
        mask_ss = self.deserialize_symbolic_state(goal_mask)
        goals = []
        for i, (o_type, o_color, predicates) in enumerate(mask_ss):
            for p_name, p_val in predicates.items():
                if p_val > 0.5:
                    g = GridGoal(p_name, val_ss[i][2][p_name] > 0.5, o_type, o_color)
                    goals.append(g)
        return goals

    def is_success(self, goals):
        success = True
        for g in goals:
            if g.predicate == 'on':
                objects = self.find(g.type, g.color)
                satisfied = g.value == same(self.agent_pos, objects[0].cur_pos)
            elif g.predicate == 'holding':
                satisfied = g.value == (
                        self.carrying is not None and
                        g.type == self.carrying.type and
                        g.color == self.carrying.color
                )
            elif g.predicate == 'is_open':
                objects = self.find(g.type, g.color)
                satisfied = False
                if hasattr(objects[0], 'is_open'):
                    satisfied = g.value == objects[0].is_open
                # print('is_open', objects[0].is_open, 'satisfied', satisfied)
            elif g.predicate == 'is_locked':
                objects = self.find(g.type, g.color)
                satisfied = False
                if hasattr(objects[0], 'is_locked'):
                    satisfied = g.value == objects[0].is_locked
                # print('is_locked', objects[0].is_locked, 'satisfied', satisfied)
            else:
                raise NotImplementedError
            success = success and satisfied
        return success

    def goal_mask(self, goal):
        objects = self.objects
        predicates = self.predicates()
        mask = np.zeros((len(self.objects), len(predicates)), dtype=np.float32)
        for i, o in enumerate(objects):
            if o is None:
                continue
            for g in goal:
                if o.type == g.type and o.color == g.color:
                    g_index = predicates.index(g.predicate)
                    mask[i, g_index] = 1
        assert(int(mask.sum()) == len(goal))  # every goal is used
        return mask

    def goal_to_symbolic_state(self, goal_list):
        """
        Parse a list of goals into symbolic state. Fill with empty dict if an object does not have a
        pairing goal
        :param goal_list: list of symbolic state with [[predicate, value, type, color]]
        :return: list of symbolic state with [[type, color, enc_dict]]
        """
        dss_list = []  # dense ss list
        count = 0
        for o in self.objects:
            if o is None:
                dss_list.append([None, None, self.symbolic_state_template()])
                continue
            new_ss = [o.type, o.color, self.symbolic_state_template()]
            for g in goal_list:
                if g.color == o.color and g.type == o.type:
                    new_ss[2][g.predicate] = g.value
                    count += 1
            dss_list.append(new_ss)
        assert(len(goal_list) == count)
        return dss_list

    def goal_symbolic_state_index(self, goal):
        for i, o in enumerate(self.objects):
            if o is None:
                continue
            if goal.type == o.type and goal.color == o.color:
                p_idx = self.predicates().index(goal.predicate)
                return i, p_idx
        raise IndexError('cannot find the symbolic state')

    @staticmethod
    def masked_symbolic_state(masked_ss):
        out = []
        for i, (t, c, d) in enumerate(masked_ss):
            for k in d:
                if d[k] != 2:
                    out.append((t, c, k, d[k] == 1))
        return out

    @property
    def info(self):
        return {
            'actions': self.action_list,
            'grid_size': (self.grid.height, self.grid.width),
            'agent_view_size': self.agent_view_size,
            'object_to_idx': OBJECT_TO_IDX,
            'color_to_idx': COLOR_TO_IDX,
            # [type, color, is_open, is_locked, holding, position]
            'object_encode_flat_size': len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + self.NUM_OBJECT_ATTR + self.MAX_FOV * 4 + 2,
            'object_encode_size': len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + self.NUM_OBJECT_ATTR + 2,
            'symbolic_state_size': len(self.symbolic_state_template()),
            'object_types': [otype for otype, _ in self.object_info],
            'object_colors': [oinfo['color'] for _, oinfo in self.object_info]
        }

    def make_instruction(self, inst):
        return inst

    def serialize_object_state(self, flat=True, ignore_missing=False):
        obj_enc = []
        n_obj = len(OBJECT_TO_IDX)
        n_color = len(COLOR_TO_IDX)
        pos_enc_size = self.MAX_FOV * 4 + 2 if flat else 2
        for obj in self.objects:
            enc = np.zeros(n_obj + n_color + self.NUM_OBJECT_ATTR + pos_enc_size, dtype=np.int64)
            if obj is None:  # destroyed or missing
                if not ignore_missing:
                    obj_enc.append(enc)
                continue
            assert (obj.cur_pos is not None)
            enc[OBJECT_TO_IDX[obj.type]] = 1
            enc[n_obj + COLOR_TO_IDX[obj.color]] = 1
            if hasattr(obj, 'is_open'):
                enc[n_obj + n_color] = int(obj.is_open)
            if hasattr(obj, 'is_locked'):
                enc[n_obj + n_color + 1] = int(obj.is_locked)
            if obj.cur_pos[0] < 0:  # being held
                enc[n_obj + n_color + 2] = 1
            else:
                rpos = relative_position(self.agent_pos, self.agent_dir, obj.cur_pos)
                if flat:
                    enc[n_obj + n_color + self.NUM_OBJECT_ATTR + self.MAX_FOV + rpos[0]] = 1
                    enc[n_obj + n_color + self.NUM_OBJECT_ATTR + self.MAX_FOV * 3 + 1 + rpos[1]] = 1
                else:
                    enc[-pos_enc_size:] = rpos
            obj_enc.append(enc)

        return np.stack(obj_enc)

    def serialize_grid_state(self):
        full_grid = self.grid.encode()
        full_grid[self.agent_pos[0]][self.agent_pos[1]] = np.array([255, self.agent_dir, 0])
        return full_grid


class EmptyEnv(MiniGridEnv, InfoEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """
    MAX_FOV = 10
    NUM_OBJECT_ATTR = 3

    def __init__(
        self,
        size=8,
        objects=None,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        assert(size < self.MAX_FOV)
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self._object_info = objects

        super(EmptyEnv, self).__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        # self.grid.set(width - 2, height - 2, Goal())
        # self.grid.set(rdx, rdy, Door('yellow', is_locked=True))

        # Place a yellow key on the left side
        for o in self._object_info:
            if o[0] == 'door':
                if 'is_locked' in o[1] and o[1]['is_locked'] == 'random':
                    o[1]['is_locked'] = self._rand_bool()
                if 'is_open' in o[1] and o[1]['is_open'] == 'random':
                    if 'is_locked' in o[1] and o[1]['is_locked']:
                        o[1]['is_open'] = False
                    else:
                        o[1]['is_open'] = self._rand_bool()

            self.place_obj(
                obj=OBJECT_CLASS[o[0]](**o[1]),
                top=(0, 0),
                size=(width, height)
            )
        # Place the agent
        if self.agent_start_pos is not None:
            self.start_pos = self.agent_start_pos
            self.start_dir = self.agent_start_dir
        else:
            self.place_agent()
        self.mission = 'None'

        self.check_grid()


class SixDoorsEnv(MiniGridEnv, InfoEnv):
    MAX_FOV = 12
    NUM_OBJECT_ATTR = 3

    def __init__(
        self,
        size=11,
        objects=None,
        colors=None,
    ):
        assert(size < self.MAX_FOV)
        self._object_info = objects
        self._colors = colors

        super(SixDoorsEnv, self).__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        door_locs = [
            (1, 5),
            (width - 2, 5),
            (3, 1),
            (width - 4, 1),
            (3, height - 2),
            (width - 4, height - 2)
        ]
        rdindex = self.np_random.choice(np.arange(len(door_locs)), len(door_locs), replace=False)
        door_locs = [door_locs[i] for i in rdindex]

        door_count = 0

        for o in self._object_info:
            if o[0] == 'door':
                if 'is_locked' in o[1] and o[1]['is_locked'] == 'random':
                    o[1]['is_locked'] = self._rand_bool()
                if 'is_open' in o[1] and o[1]['is_open'] == 'random':
                    if 'is_locked' in o[1] and o[1]['is_locked']:
                        o[1]['is_open'] = False
                    else:
                        o[1]['is_open'] = self._rand_bool()

                self.place_obj(
                    obj=OBJECT_CLASS[o[0]](**o[1]),
                    top=door_locs[door_count],
                    size=(1, 1)
                )
                door_count += 1
        for o in self._object_info:
            if o[0] != 'door':
                self.place_obj(
                    obj=OBJECT_CLASS[o[0]](**o[1]),
                    top=(2, 2),
                    size=(width - 2, height - 2)
                )

        # Place the agent
        self.place_agent()
        self.mission = 'None'

        self.check_grid()


class RoomGoalEnv(RoomGrid, InfoEnv):
    MAX_FOV = 20
    NUM_OBJECT_ATTR = 3

    def __init__(
        self,
        num_rows=3,
        room_size=5,
        seed=None,
        max_steps=1000,
        objects=None,
        room_map=None,
    ):
        assert num_rows * room_size < self.MAX_FOV
        self._object_info = objects
        self._room_map = room_map
        super(RoomGoalEnv, self).__init__(
            room_size=room_size,
            num_rows=num_rows,
            max_steps=max_steps,
            seed=seed
        )

    def _gen_grid(self, width, height):
        super(RoomGoalEnv, self)._gen_grid(width, height, (1, 1))

        # Connect the middle column rooms into a hallway
        for j in range(1, self.num_rows):
            self.remove_wall(1, j, 3)

        room_idx = [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)]
        door_idx = [0, 0, 0, 2, 2, 2]

        for o in self._object_info:
            if o[0] == 'door':
                if 'is_locked' in o[1] and o[1]['is_locked'] == 'random':
                    o[1]['is_locked'] = self._rand_bool()
                r_idx = self._room_map[(o[1]['color'], 'door')]
                self.add_door(room_idx[r_idx][0], room_idx[r_idx][1], door_idx[r_idx],
                              color=o[1]['color'], locked=o[1]['is_locked'])
            elif o[0] == 'key':
                self.add_object(1, self._rand_int(0, self.num_rows), 'key', color=o[1]['color'])

            elif o[0] == 'floor':
                r_idx = self._room_map[(o[1]['color'], 'floor')]
                self.add_object(room_idx[r_idx][0], room_idx[r_idx][1], kind='floor', color=o[1]['color'])
            else:
                raise NotImplementedError

        self.place_agent(1, self._rand_int(0, self.num_rows))

        self.mission = 'None'
        self.check_grid()


class DeliveryEnv(RoomGrid, InfoEnv):
    MAX_FOV = 20
    NUM_OBJECT_ATTR = 3

    def __init__(
        self,
        num_rows=2,
        room_size=6,
        seed=None,
        max_steps=1000,
        objects=None,
        colors=None,
    ):
        assert num_rows * room_size < self.MAX_FOV
        self.env_info = {}
        self._object_info = objects
        self._colors = colors
        super(DeliveryEnv, self).__init__(
            room_size=room_size,
            num_rows=num_rows,
            max_steps=max_steps,
            seed=seed
        )

    def _gen_grid(self, width, height):
        super(DeliveryEnv, self)._gen_grid(width, height, (1, 1))

        # Connect the middle column rooms into a hallway
        for j in range(1, self.num_rows):
            self.remove_wall(1, j, 3)

        room_idx = [(0, 0), (0, 1), (2, 0), (2, 1)]
        colors = ['red', 'green', 'blue', 'yellow']
        n_color = len(colors)
        door_idx = [0, 0, 2, 2]

        door_choice = self.np_random.choice(np.arange(n_color), size=n_color, replace=False)
        obj_choice = self.np_random.choice(np.arange(n_color), size=n_color, replace=False)

        door_colors = [colors[i] for i in door_choice]
        obj_colors = [colors[i] for i in obj_choice]
        goal_colors = door_colors

        self.env_info = {}
        self.env_info['ball'] = {}
        self.env_info['goal'] = {}

        for ri, dc, di, oc, gc in zip(room_idx, door_colors, door_idx, obj_colors, goal_colors):
            door, _ = self.add_door(ri[0], ri[1], di, color=dc, locked=self._rand_bool())
            self.add_object(ri[0], ri[1], kind='floor', color=dc)
            self.add_object(ri[0], ri[1], kind='ball', color=oc)
            self.add_object(1, self._rand_int(0, self.num_rows), 'key', color=dc)
            self.env_info['ball'][oc] = {'door_color': dc, 'is_locked': door.is_locked}
            self.env_info['goal'][gc] = {'door_color': dc, 'is_locked': door.is_locked}
            # self.env_info['target_color'][]

        self.place_agent(1, self._rand_int(0, self.num_rows))
        # Make sure all rooms are accessible
        # self.connect_all()

        # self.obj = obj
        # self.mission = "pick up the %s %s" % (obj.color, obj.type)
        self.check_grid()

    def make_instruction(self, instructions):
        inst = []
        ball_color, goal_color = instructions
        goal_door = self.env_info['goal'][goal_color]
        ball_door = self.env_info['ball'][ball_color]
        same_room = goal_door['door_color'] == ball_door['door_color']

        if goal_door['is_locked']:
            inst.extend(open_door_inst(goal_door['door_color']))
        if ball_door['is_locked'] and not same_room:
            inst.extend(open_door_inst(ball_door['door_color']))

        if not ball_door['is_locked']:
            inst.extend([
                ['goto', 'facing', 'door', ball_door['door_color']],
                ['toggle']
            ])
        inst.extend([
            ['goto', 'facing', 'ball', ball_color],
            ['pickup']
        ])

        if not goal_door['is_locked'] and not same_room:
            inst.extend([
                ['goto', 'facing', 'door', goal_door['door_color']],
                ['toggle']
            ])

        inst.extend([
            ['goto', 'on', 'floor', goal_color],
        ])
        return inst


class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        assert(len(self.position) == 3)

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return same(self.position, other.position)


def get_action(pos1, pos2):
    if pos1[2] != pos2[2]:
        if (pos2[2], pos1[2]) in [(1, 0), (2, 1), (3, 2), (0, 3)]:
            return 'right'
        else:
            return 'left'
    else:
        return 'forward'


def return_path(current_node):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    path = path[::-1]  # Return reversed path
    actions = []
    for p1, p2 in zip(path[:-1], path[1:]):
        actions.append(get_action(p1, p2))
    return path, actions


def next_positions(curr_pos):
    pos = []
    dm = {
        0: (1, 0),
        1: (0, 1),
        2: (-1, 0),
        3: (0, -1)
    }
    cd = curr_pos[2]
    pos.append((dm[cd][0] + curr_pos[0], dm[cd][1] + curr_pos[1], cd))
    new_dir = cd + 1
    if new_dir > 3:
        new_dir = 0
    pos.append((curr_pos[0], curr_pos[1], new_dir))
    new_dir = cd - 1
    if new_dir < 0:
        new_dir = 3
    pos.append((curr_pos[0], curr_pos[1], new_dir))
    return pos


def positions_around(env, target_pos):
    valid_pos = []
    for dp in ((1, 0, 2), (0, 1, 3), (-1, 0, 0), (0, -1, 1)):
        new_p = (dp[0] + target_pos[0], dp[1] + target_pos[1], dp[2])
        cell = env.grid.get(new_p[0], new_p[1])
        if cell is None or cell.can_overlap():
            valid_pos.append(new_p)
    return valid_pos


def position_on(env, target_pos):
    poss = []
    for i in range(4):
        poss.append([target_pos[0], target_pos[1], i])
    return poss


def astar(maze, start, end):
    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Adding a stop condition
    outer_iterations = 0
    max_iterations = 10000

    # Loop until you find the end
    while len(open_list) > 0:
        outer_iterations += 1

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        if outer_iterations > max_iterations:
            # if we hit this point return the path such as it is
            # it will not contain the destination
            print("giving up on pathfinding too many iterations")
            return return_path(current_node)

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            return return_path(current_node)

        # Generate children
        children = []

        for node_pos in next_positions(current_node.position):  # Adjacent squares
            # Make sure walkable terrain
            node = maze.grid.get(node_pos[0], node_pos[1])
            # if node is not None and node.type == 'door':
            #     print(node.can_overlap())
            if node is not None and not node.can_overlap():
                continue

            # Create new node
            new_node = Node(current_node, node_pos)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            a = child.position[2]
            b = end_node.position[2]
            dir_dist = min([abs(a - b), abs(a + 4 - b), abs(a - 4 - b)])
            child.h = math.sqrt(((child.position[0] - end_node.position[0]) ** 2) + (
                        (child.position[1] - end_node.position[1]) ** 2)) + dir_dist
            child.f = child.g + child.h

            # Child is already in the open list
            if len([open_node for open_node in open_list if child == open_node and child.g > open_node.g]) > 0:
                continue

            # Add the child to the open list
            open_list.append(child)


def same(a, b):
    for x, y in zip(a, b):
        if x != y:
            return False
    return True


def plan_goto(env, agent_pos, object_pos, mode='facing'):
    if mode == 'facing':
        target_poss = positions_around(env, object_pos)
    elif mode == 'on':
        target_poss = position_on(env, object_pos)
    else:
        raise NotImplementedError

    min_path = None
    for target_pos in target_poss:
        if same(agent_pos, target_pos):
            return [], []
        # print(agent_pos)
        path = astar(env, agent_pos, target_pos)
        # print(len(path[1]))
        if path is None:
            # print('no solution: ', target_pos, agent_pos)
            continue
        assert(None not in path)
        if min_path is None or len(path[1]) < len(min_path[1]):
            min_path = path
    return min_path


def follow_instructions(env, instructions):
    instructions = env.make_instruction(instructions)
    for inst in instructions:
        if inst[0] == 'goto':
            if isinstance(inst[2], str):
                targets = env.find(*inst[2:])
                assert len(targets) == 1, targets
                target_pos = targets[0].cur_pos
            else:
                target_pos = inst[2:]
            solution = plan_goto(env, list(env.agent_pos) + [env.agent_dir], target_pos, mode=inst[1])
            if solution is None:
                yield None
            path, actions = solution
            # print(path)
            # print(actions)
            for a in actions:
                yield a
        else:
            yield inst[0]


def goal_to_macro(env, goals, verbose=False):
    def goto(o_type, o_color, mode):
        targets = env.find(o_type, o_color)
        target_pos = targets[0].cur_pos
        return plan_goto(env, list(env.agent_pos) + [env.agent_dir], target_pos, mode=mode)

    # check if all objects referred by the goals are valid
    for g in goals:
        o = env.find(g.type, g.color)
        if len(o) != 1:
            yield None, 'MACRO_BAD_GOAL'
        o = o[0]
        if (g.predicate == 'is_open' and g.value) or (g.predicate == 'is_locked' and not g.value):
            if o.type != 'door' or o.is_open:
                yield None, 'MACRO_BAD_GOAL'
        if g.predicate == 'holding' and not o.can_pickup():
            yield None, 'MACRO_BAD_GOAL'
        if g.predicate == 'holding' and env.carrying is not None:
            yield None, 'MACRO_BAD_GOAL'
        if g.predicate == 'on' and not o.can_overlap():
            yield None, 'MACRO_BAD_GOAL'

    is_open_door = False
    for g in goals:
        if g.predicate == 'is_open':
            is_open_door = True
            for g1 in goals:
                if g1 != g:
                    is_open_door = is_open_door and g1.color == g.color and g1.type == g.type
                    is_open_door = is_open_door and g1.predicate == 'is_locked' and not g1.value
            if is_open_door:
                goals = [g]
                break

    actions = []
    if len(goals) == 1 and goals[0].predicate == 'on' and goals[0].value:
        solution = goto(goals[0].type, goals[0].color, mode='on')
        if solution is None:
            yield None, 'MACRO_FAILURE'
        _, actions = solution
    elif len(goals) == 1 and goals[0].predicate == 'holding' and goals[0].value:
        solution = goto(goals[0].type, goals[0].color, mode='facing')
        if solution is None:
            yield None, 'MACRO_FAILURE'
        _, actions = solution
        actions.append('pickup')
    elif is_open_door:
        solution = goto(goals[0].type, goals[0].color, mode='facing')
        if solution is None:
            yield None, 'MACRO_FAILURE'
        _, actions = solution
        actions.append('toggle')
    else:
        if verbose:
            print('WARNING!!! cannot parse goals %r' % goals)
        yield None, 'MACRO_BAD_GOAL'
    for a in actions:
        yield a, -1


def relative_position(agent_pos, agent_dir, pos):
    rpos = [pos[0] - agent_pos[0], pos[1] - agent_pos[1]]
    if agent_dir == 0:
        rpos = [-rpos[1], rpos[0]]
    elif agent_dir == 2:
        rpos = [rpos[1], -rpos[0]]
    elif agent_dir == 3:
        rpos = [-rpos[0], -rpos[1]]
    return rpos


register(
    id='MiniGrid-room-v0',
    entry_point='gym_minigrid.envs:EmptyEnv'
)


def main():
    e = DeliveryEnv()

    # while True:
    #     e.reset()
    #     for _ in range(3):
    #         e.render('human')
    #     time.sleep(10)

    # inst = [['goto', 'facing', 'key', 'yellow'],
    #         ['pickup'], ['goto', 'facing', 'door', 'yellow'],
    #         ['toggle'], ['goto', 'on', 'floor', 'yellow']]

    inst = [['goto', 'facing', 'door', 'yellow'],
            ['pickup'], ['goto', 'facing', 'door', 'yellow'],
            ['toggle'], ['goto', 'on', 'floor', 'yellow']]
    while True:
        print('-------------')
        e.reset()
        plan = follow_instructions(e, inst)
        e.render('human')
        for a in plan:
            if a is None:
                print('failed')
                time.sleep(3)
                break
            e.step(e.actions[a])

            # time.sleep(0.1)
            # print(e.agent_pos)
            e.render('human')
            # print(door.is_open)
            # print(door.can_overlap())

        e.render('human')
        # time.sleep(2)
    print()


if __name__ == '__main__':
    main()
