import numpy as np


def np_to_batch(x):
    return np.expand_dims(x, axis=0)


def to_onehot(cls_ind, num_cls):
    """ Convert a integer or a integer array to one-hot vector

    Args:
        cls_inds: class indices, can be either Integer or a Integer array
        num_cls: total number of classes
    Returns:
        vec: a numpy array of shape [?, num_cls]
    """
    if isinstance(cls_ind, int):
        vec = np.zeros(num_cls, dtype=np.bool)
        vec[cls_ind] = 1
    elif isinstance(cls_ind, np.ndarray):
        assert(cls_ind.dtype == np.int64 or cls_ind.dtype == np.int32)
        vec = np.zeros(list(cls_ind.shape) + [num_cls], dtype=np.bool)
        vec_shape = vec.shape
        vec = np.reshape(vec, [-1, num_cls])
        vec[np.arange(vec.shape[0]), cls_ind.reshape(-1)] = 1
        vec = np.reshape(vec, vec_shape)
    else:
        raise NotImplementedError
    return vec


def pad_sequence(sequence, target_length):
    """
    Pad a sequence to target length
    :param sequence:
    :param target_length:
    :return: padded sequence
    """

    padded = np.zeros((target_length,) + sequence.shape[1:], dtype=sequence.dtype)
    padded[:sequence.shape[0]] = sequence
    mask = np.zeros(target_length, dtype=np.float32)
    mask[:sequence.shape[0]] = 1
    return padded, mask


def pad_sequence_list(sequence, target_length):
    padded = np.zeros((len(sequence), target_length) + sequence[0].shape[1:], dtype=sequence[0].dtype)
    mask = np.zeros((len(sequence), target_length), dtype=np.float32)
    for i, s in enumerate(sequence):
        padded[i, :s.shape[0], ...] = s
        mask[i, :s.shape[0]] = 1
    return padded, mask
