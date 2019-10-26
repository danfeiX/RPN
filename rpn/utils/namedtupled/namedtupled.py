from future import standard_library
standard_library.install_aliases()
from collections import Mapping, namedtuple, UserDict


def mapper(mapping, _nt_name='NT'):
    """ Convert mappings to namedtuples recursively. """
    if isinstance(mapping, Mapping) and not isinstance(mapping, AsDict):
        if 'type' in mapping and _nt_name == 'NT':
            _nt_name = mapping['type']
        for key, value in list(mapping.items()):
            mapping[key] = mapper(value)
        return namedtuple_wrapper(_nt_name, **mapping)
    elif isinstance(mapping, list):
        return [mapper(item) for item in mapping]
    return mapping


def namedtuple_wrapper(_nt_name, **kwargs):
    wrap = namedtuple(_nt_name, kwargs)
    return wrap(**kwargs)


class AsDict(UserDict):
    """ A class that exists just to tell `mapper` not to eat it. """


def ignore(mapping):
    """ Use ignore to prevent a mapping from being mapped to a namedtuple. """
    if isinstance(mapping, Mapping):
        return AsDict(mapping)
    elif isinstance(mapping, list):
        return [ignore(item) for item in mapping]
    return mapping


def isnamedtupleinstance(x):  # thank you http://stackoverflow.com/a/2166841/6085135
    _type = type(x)
    bases = _type.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(_type, '_fields', None)
    if not isinstance(fields, tuple):
        return False
    return all(type(i)==str for i in fields)


def reducer(obj):
    if isinstance(obj, dict):
        return {key: reducer(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [reducer(value) for value in obj]
    elif isnamedtupleinstance(obj):
        return {key: reducer(value) for key, value in obj._asdict().items()}
    elif isinstance(obj, tuple):
        return tuple(reducer(value) for value in obj)
    else:
        return obj