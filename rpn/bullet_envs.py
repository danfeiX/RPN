from rpn.bullet_task_utils import TaskEnv
from collections import OrderedDict
from third_party.pybullet.utils.pybullet_tools.utils import is_center_stable, wait_for_duration
import pybullet as p

FRUITS = ['apple', 'pear', 'lemon', 'banana', 'peach', 'orange', 'plum', 'strawberry']
VEGGIES = ['cabbage', 'tomato', 'pumpkin']
INGREDIENTS = FRUITS + VEGGIES

COOKWARES = ['pot', 'pan']
CONTAINERS = ['plate']
ACTIVATABLE = ['stove', 'sink']
GRASPABLE = INGREDIENTS + COOKWARES + CONTAINERS


class TaskEnvBlock(TaskEnv):
    """
    BlockStacking environment
    """
    def __init__(self, objects):
        super(TaskEnvBlock, self).__init__(objects)
        self._predicate_funcs = [self.pf_on]
        self._action_funcs = OrderedDict({
            'pick': self.pick,
            'place': self.place,
        })

    @staticmethod
    def action_names():
        return ['noop', 'pick', 'place']

    @staticmethod
    def num_action_args():
        return [1, 1, 2]

    @staticmethod
    def unitary_predicates():
        return ['grasped']

    @staticmethod
    def binary_predicates():
        return ['on']

    def pf_on(self):
        for o1 in self.objects.ids:
            for o2 in self.objects.ids:
                if o1 == o2:
                    continue
                self._apply((o1, o2), 'on', negated=not is_center_stable(o1, o2))

    def pick(self, o):
        if self.applicable('pick', o):
            return self._safe_apply((o,), 'grasped', negated=False)
        return False

    def place(self, o, r):
        if self.applicable('place', o, r):
            return self._safe_apply((o,), 'grasped', negated=True)
        return False

    def _find_atop(self, o):
        atop = []
        for o1 in self.objects.ids:
            if o1 != o and self.symbolic_state[(o1, o)]['on']:
                atop.append(o1)
        return atop

    def _find_beneath(self, o):
        beneath = []
        for o1 in self.objects.ids:
            if o1 != o and self.symbolic_state[(o, o1)]['on']:
                beneath.append(o1)
        return beneath

    def _is_top_clear(self, o):
        return len(self._find_atop(o)) == 0

    def check_integrity(self):
        good = True
        for o in self.objects.ids:
            for o1 in self._find_atop(o):
                o2 = self._find_beneath(o1)
                good = good and (len(o2) == 1) and (o2[0] == o)
            o1 = self._find_beneath(o)
            good = good and (len(o1) <= 1)
            if len(o1) == 1:
                o2 = self._find_atop(o1[0])
                good = good and (o in o2)
        return good

    def applicable(self, action_name, *args):
        assert(len(args) == self.num_action_args()[self.action_names().index(action_name)])
        if action_name == 'pick':
            return self._is_top_clear(args[0])
        elif action_name == 'place':
            return self._is_top_clear(args[1])
        else:
            raise NotImplementedError


class TaskEnvCook(TaskEnvBlock):
    """
    Kitchen3D environment
    """
    COOK_COLOR = [0.396, 0.263, 0.129, 1]
    CLEAN_COLOR = [1, 1, 1, 0.5]
    STOVE_ACTIVE_COLOR = [0.8, 0, 0, 1]
    SINK_ACTIVE_COLOR = [0, 0, 0.8, 1]

    def __init__(self, objects):
        TaskEnv.__init__(self, objects)
        # TODO: be sure to order the function correctly!
        self._predicate_funcs = [
            self.pf_on,
            self.pf_cooked,
            self.pf_cleaned,
            self.pf_activated
        ]
        self._action_funcs = OrderedDict({
            'pick': self.pick,
            'place': self.place,
            'activate': self.activate,
        })

    @staticmethod
    def action_names():
        return ['noop', 'pick', 'place', 'activate']

    @staticmethod
    def num_action_args():
        return [1, 1, 2, 1]

    @staticmethod
    def unitary_predicates():
        return ['grasped', 'cooked', 'cleaned', 'activated']

    @staticmethod
    def binary_predicates():
        return ['on']

    def pf_cooked(self):
        stove = self.objects.id('stove/0')
        if not self.symbolic_state[(stove,)]['activated']:
            return

        for c in self._find_atop(stove):
            if self.objects.type(c) in COOKWARES:
                cookware = c
                for ing in self._find_atop(cookware):
                    cook_fruit = (self.objects.type(c) == 'pan' and self.objects.type(ing) in FRUITS)
                    cook_veggie = (self.objects.type(c) == 'pot' and self.objects.type(ing) in VEGGIES)
                    if (cook_fruit or cook_veggie) and self.symbolic_state[(ing,)]['cleaned']:
                        self._apply((ing,), 'cooked', negated=False)
                        p.changeVisualShape(ing, -1, rgbaColor=self.COOK_COLOR)

    def pf_cleaned(self):
        sink = self.objects.id('sink/0')
        if not self.symbolic_state[(sink,)]['activated']:
            return
        for ing in self._find_atop(sink):
            if self.objects.type(ing) in INGREDIENTS:
                self._apply((ing,), 'cleaned', negated=False)
                p.changeVisualShape(ing, -1, rgbaColor=self.CLEAN_COLOR)

    def pf_activated(self):
        sink = self.objects.id('sink/0')
        if self.symbolic_state[(sink,)]['activated']:
            p.changeVisualShape(sink, -1, rgbaColor=self.SINK_ACTIVE_COLOR)
        stove = self.objects.id('stove/0')
        if self.symbolic_state[(stove,)]['activated']:
            p.changeVisualShape(stove, -1, rgbaColor=self.STOVE_ACTIVE_COLOR)

    def activate(self, o):
        if self.applicable('activate', o):
            return self._safe_apply((o,), 'activated', negated=False)
        return False

    def deactivate(self, o):
        return self._safe_apply((o,), 'activated', negated=True)

    def applicable(self, action_name, *args):
        assert(len(args) == self.num_action_args()[self.action_names().index(action_name)])
        if action_name == 'pick':
            return self._is_top_clear(args[0]) and self.objects.type(args[0]) in GRASPABLE
        elif action_name == 'place':
            # return self._is_top_clear(args[1])
            return True
        elif action_name == 'activate':
            o = args[0]
            return self.objects.type(o) in ACTIVATABLE and not self.symbolic_state[(o,)]['activated']
        else:
            raise NotImplementedError
