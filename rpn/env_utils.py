import init_path
from collections import OrderedDict
from builtins import int
from future.utils import listitems


from third_party.pybullet.utils.pybullet_tools.utils import WorldSaver, connect, dump_world, get_pose, set_pose, Pose, \
    Point, set_camera, stable_z, create_box, create_cylinder, create_plane, HideOutput, load_model, \
    BLOCK_URDF, BLOCK1_URDF, get_configuration, SINK_URDF, STOVE_URDF, load_model, get_body_name, \
    disconnect, DRAKE_IIWA_URDF, PLATE_URDF, get_bodies, user_input, HideOutput, SHROOM_URDF, sample_placement, \
    get_movable_joints, pairwise_collision, stable_z, sample_placement_region, step_simulation
from contextlib import contextmanager
import pybullet as p
import numpy as np

from third_party.pybullet.utils.pybullet_tools.utils_oo import Body

URDFS = {
    'short_floor': 'models/short_floor.urdf',
    'plate': 'models/ycb/029_plate/textured.urdf',
    # 'plate': 'models/dinnerware/plate.urdf',
    'pot': 'models/dinnerware/pan_tefal.urdf',
    'pan': 'models/skillet/pan_tefal.urdf',
    'cracker_box': 'models/ycb/003_cracker_box/textured.urdf',
    'master_chef_can': 'models/ycb/002_master_chef_can/textured.urdf',
    'banana': 'models/ycb/011_banana/textured.urdf',
    'strawberry': 'models/ycb/012_strawberry/textured.urdf',
    'apple': 'models/ycb/013_apple/textured.urdf',
    'lemon': 'models/ycb/014_lemon/textured.urdf',
    'peach': 'models/ycb/015_peach/textured.urdf',
    'pear': 'models/ycb/016_pear/textured.urdf',
    'orange': 'models/ycb/017_orange/textured.urdf',
    'plum': 'models/ycb/018_plum/textured.urdf',
    'cabbage': 'models/ingredients/cabbage/textured.urdf',
    'tomato': 'models/ingredients/tomato/textured.urdf',
    'pumpkin': 'models/ingredients/pumpkin/textured.urdf',
    'mug': 'models/ycb/025_mug/textured.urdf',
    'stove': 'models/cooktop/textured.urdf',
    # 'sink': 'models/sink.urdf',
    'sink': 'models/sink/tray.urdf',
    'ph_gripper': 'models/drake/objects/gripper_invisible.urdf',
}

SHAPES = {
    'shape_box': create_box,
    'shape_cylinder': create_cylinder,
    'shape_plane': create_plane
}


def load_world(world_spec, world):
    with HideOutput():
        # robot = load_model(DRAKE_IIWA_URDF)
        world.load_robot(world_spec['robot_urdf'])

        for obj_spec in world_spec['objects']:
            world.load_object(**obj_spec)

        for shape_spec in world_spec['shapes']:
            world.create_shape(**shape_spec)

        # TODO: HIGH resolve placement chain
        for placement in world_spec['placements']:
            if placement[0] == 'set':
                world.place(world.id(placement[1]), world.id(placement[2]), placement[3])

        for placement in world_spec['placements']:
            if placement[0] == 'on':
                region = None
                if len(placement) == 4:
                    region = placement[3]
                world.random_place(world.id(placement[1]), world.id(placement[2]), region=region)

    set_camera(180, -70, 1.2, Point())

    return world


def sample_scene():

    with HideOutput():
        # robot = load_model(DRAKE_IIWA_URDF)
        world = World(['table', 'stove', 'sink', 'plate', 'pot', 'cabbage'])
        world.load_robot(URDFS['ph_gripper'])
        world.load_object(URDFS['short_floor'], 'table', fixed=True, globalScaling=0.6)
        world.load_object(URDFS['stove'], 'stove', fixed=True, globalScaling=0.8)
        world.load_object(URDFS['sink'], 'sink', fixed=True)
        world.load_object(URDFS['plate'], 'plate', fixed=False)
        world.load_object(URDFS['pot'], 'pot', fixed=True)
        world.load_object(URDFS['cabbage'], 'cabbage', fixed=False)
        world.place(world.id('stove/0'), world.id('table/0'), [[0.4, 0, 0], [0, 0, 0, 1]])
        world.place(world.id('sink/0'), world.id('table/0'), [[-0.2, -0.4, 0], [0, 0, 0, 1]])
        # world.place(world.id('plate/0'), world.id('table/0'), [[-0.3, -0.3, 0], [0, 0, 0, 1]])
        world.place(world.id('plate/0'), world.id('table/0'),   [[-0.4, 0, 0], [0, 0, 0, 1]])
        # world.place(world.id('pot/0'), world.id('table/0'), [[0.3, -0.3, 0], [0, 0, 0, 1]])
        world.random_place(world.id('pot/0'), world.id('table/0'))
        # world.place(world.id('cabbage/0'), world.id('table/0'), [[0, 0, 0], [0, 0, 0, 1]])
        world.random_place(world.id('cabbage/0'), world.id('table/0'), np.array([[-0.3, -0.2], [0.3, 0.2]]))

    set_camera(180, -60, 1, Point())
    return world


@contextmanager
def pb_session(use_gui):
    options = '--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0'
    connect(use_gui=use_gui)
    yield
    disconnect()


@contextmanager
def world_saved():
    saved_world = WorldSaver()
    yield
    saved_world.restore()


def set_rendering_pose(camera):
    camera.set_pose_ypr([0, 0, 0], 1.5, 180, -75)


class World(object):
    def __init__(self, object_types, objects=(), robot=None):
        self._odb_type = OrderedDict()
        self._odb_id = OrderedDict()
        self._robot = robot
        self._calls = []
        self._all_types = object_types
        for o in objects:
            self.add_object(o)

    def reset(self):
        self._odb_type = OrderedDict()
        self._odb_id = OrderedDict()
        self._robot = None
        self._calls = []

    def reload(self):
        """
        Reload all objects (including robot)
        """
        calls = self._calls
        self.reset()

        for c in calls:
            f = c.pop('func')
            c.pop('self')
            if 'kwargs' in c:
                c.update(c.pop('kwargs'))
            with HideOutput():
                f(**c)

    @property
    def robot(self):
        return self._robot

    @property
    def robot_dof(self):
        return len(get_movable_joints(self._robot))

    @robot.setter
    def robot(self, robot):
        self._robot = robot

    def load_robot(self, urdf_path):
        self._calls.append(dict(listitems({'func': self.load_robot}) + listitems(locals())))
        self._robot = load_model(urdf_path)

    def add_object(self, task_object, init_pose=None, randomly_place_on=None):
        assert(task_object.type in self.all_types)
        assert('/' not in task_object.type)
        assert(isinstance(task_object.type, str))
        assert(isinstance(task_object.uid, int))
        assert(task_object.uid not in self._odb_id)
        tname = task_object.type
        if tname not in self._odb_type:
            self._odb_type[tname] = []

        # rename the object
        task_object.name = '%s/%i' % (tname, len(self._odb_type[tname]))
        self._odb_type[tname].append(task_object)
        self._odb_id[task_object.uid] = task_object

        # place the object
        if init_pose is not None:
            assert (randomly_place_on is None)
            set_pose(task_object.uid, init_pose)
        elif randomly_place_on is not None:
            self.random_place(task_object.uid, randomly_place_on)

    def load_object(self, path, type_name, fixed=False, n_copy=1, init_pose=None, randomly_place_on=None,
                    color=None, **kwargs):
        self._calls.append(dict(listitems({'func': self.load_object}) + listitems(locals())))
        for _ in range(n_copy):
            o = load_model(path, fixed_base=fixed, **kwargs)
            if color is not None:
                p.changeVisualShape(o, -1, rgbaColor=color)
            self.add_object(WorldObject(o, type_name, fixed=fixed), init_pose, randomly_place_on)

    def create_shape(self, geom, type_name, fixed=False, n_copy=1, init_pose=None, randomly_place_on=None, **kwargs):
        self._calls.append(dict(listitems({'func': self.create_shape}) + listitems(locals())))
        for _ in range(n_copy):
            o = SHAPES[geom](**kwargs)
            p.changeVisualShape(o, -1, rgbaColor=kwargs['color'])
            self.add_object(WorldObject(o, type_name, fixed=fixed), init_pose, randomly_place_on)

    def random_place(self, body, surface, region=None, max_attempt=10):
        robot = [self._robot] if self._robot is not None else []
        body_id = self[body].uid
        surface_id = self[surface].uid
        ids = [oid for oid in self.ids if oid not in [body_id, surface_id]]
        return random_place(
            body_id, surface_id,
            fixed=ids + robot,
            max_attempt=max_attempt,
            region=region
        )

    def place(self, body, surface, pose):
        body_id = self[body].uid
        surface_id = self[surface].uid
        z = stable_z(body_id, surface_id)
        pose[0][-1] = z
        set_pose(body_id, pose)
        # ids = [oid for oid in self.ids if oid not in [body_id, surface_id]]
        # robot = [self._robot] if self._robot is not None else []
        # fixed = ids + robot
        # for b in fixed:
        #     if pairwise_collision(body_id, b):
        #         raise ValueError('Initial placement of %s has collision with %s' % (self[body].name, self[b].name))

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        for o in self.tolist():
            yield o

    @property
    def ids(self):
        return sorted([oid for oid in self._odb_id.keys()])

    def type_ids(self, type_name):
        return [o.uid for o in self.tolist() if o.type == type_name]

    @property
    def names(self):
        return [o.name for o in self.tolist()]

    @property
    def all_types(self):
        return self._all_types

    @property
    def object_type_indices(self):
        return [self.all_types.index(o.type) for o in self.tolist()]

    def tolist(self):
        return [self._odb_id[oid] for oid in self.ids]

    def id(self, name):
        return self[name].uid

    def name(self, oid):
        return self[oid].name

    def type(self, oid):
        return self[oid].type

    def __getitem__(self, key):
        if isinstance(key, int) and key in self._odb_id:
            return self._odb_id[key]
        elif isinstance(key, str) and key.split('/')[0] in self._odb_type:
            kp = key.split('/')
            if len(kp) == 1:
                t = kp[0]
                n = '0'
            elif len(kp) == 2:
                t, n = kp
            else:
                raise KeyError('%s is not a valid object key' % key)
            o = self._odb_type[t][int(n)]
            assert(o.name == ('%s/%s' % (t, n)))
            return o
        else:
            raise KeyError('%s is not a valid object key' % key)

    @property
    def movable(self):
        return self.__class__([o for o in self.tolist() if not o.fixed])

    @property
    def fixed(self):
        return self.__class__([o for o in self.tolist() if o.fixed])


class WorldObject(Body):
    def __init__(self, uid, type_name, fixed, boundary=None, scale=None):
        super(WorldObject, self).__init__(uid, None, boundary, scale)
        self._fixed = fixed
        self._type = type_name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def type(self):
        return self._type

    @property
    def fixed(self):
        return self._fixed

    def __repr__(self):
        return self._name


def random_place(body, surface, fixed=(), region=None, max_attempt=10):
    for _ in range(max_attempt):
        if region is None:
            pose = sample_placement(body, surface, percent=0.6)
        else:
            pose = sample_placement_region(body, surface, region=region, percent=0.3)
        set_pose(body, pose)
        if (pose is None) or any(pairwise_collision(body, b) for b in fixed):
            continue
        return pose
    return False


def main():
    with pb_session(use_gui=True):
        sample_scene()

        input()
        # # while True:
        for _ in range(1):
            step_simulation()
        input()


if __name__ == '__main__':
    main()
