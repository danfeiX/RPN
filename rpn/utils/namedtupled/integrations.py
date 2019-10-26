from .namedtupled import mapper, namedtuple
import yaml
import json
import os
import getpass


def load_lists(keys=[], values=[], name='NT'):
    """ Map namedtuples given a pair of key, value lists. """
    mapping = dict(zip(keys, values))
    return mapper(mapping, _nt_name=name)


def load_json(data=None, path=None, name='NT'):
    """ Map namedtuples with json data. """
    if data and not path:
        return mapper(json.loads(data), _nt_name=name)
    if path and not data:
        return mapper(json.load(path), _nt_name=name)
    if data and path:
        raise ValueError('expected one source and received two')


def load_yaml(data=None, path=None, name='NT'):
    """ Map namedtuples with yaml data. """
    if data and not path:
        return mapper(yaml.load(data), _nt_name=name)
    if path and not data:
        with open(path, 'r') as f:
            data = yaml.load(f)
        return mapper(data, _nt_name=name)
    if data and path:
        raise ValueError('expected one source and received two')


def load_env(keys=[], name='NT', use_getpass=False):
    """ Returns a namedtuple from a list of environment variables.
    If not found in shell, gets input with *input* or *getpass*. """
    NT = namedtuple(name, keys)
    if use_getpass:
        values = [os.getenv(x) or getpass.getpass(x) for x in keys]
    else:
        values = [os.getenv(x) or input(x) for x in keys]
    return NT(*values)