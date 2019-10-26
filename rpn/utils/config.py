from collections import namedtuple


def dict_to_namedtuple(d, name='config'):
    return namedtuple(name, d.keys())(**d)


def namedtuple_to_dict(d):
    return d._asdict()
