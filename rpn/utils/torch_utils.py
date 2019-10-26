"""
Common PyTorch Utilities

author: Danfei Xu
"""

import torch
from torch.nn import init
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np

USE_GPU = torch.cuda.is_available()


def safe_cuda(x):
    if USE_GPU:
        return x.cuda()
    return x


def batch_to_cuda(batch, exclude_key=()):
    cuda_batch = {}
    for k, v in batch.items():
        if k not in exclude_key:
            if isinstance(v, torch.Tensor):
                v = safe_cuda(v)
            # if isinstance(v, Data) and USE_GPU:
            #     v = v.to(torch.device('cuda'))
            elif isinstance(v, (list, tuple)) and isinstance(v[0], torch.Tensor):
                v = [safe_cuda(e) for e in v]
        cuda_batch[k] = v
    return cuda_batch


def batch_to_tensor(batch, exclude_key=()):
    tensor_batch = {}
    for k, v in batch.items():
        if k not in exclude_key and isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        elif isinstance(v, (list, tuple)) and isinstance(v[0], np.ndarray):
            v = [torch.from_numpy(e) for e in v]
        tensor_batch[k] = v
    return tensor_batch


def batch_to_numpy(batch, exclude_key=()):
    np_batch = {}
    for k, v in batch.items():
        if k not in exclude_key and isinstance(v, torch.Tensor):
            v = to_numpy(v)
        elif isinstance(v, (list, tuple)) and isinstance(v[0], torch.Tensor):
            v = [to_numpy(e) for e in v]
        np_batch[k] = v
    return np_batch


def save_checkpoint(checkpoint_path, model, optimizer, verbose=True, **kwargs):
    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'model_class': model.__class__.__name__}
    state.update(kwargs)
    torch.save(state, open(checkpoint_path, 'wb+'))
    if verbose:
        print('model saved to %s' % checkpoint_path)


def load_checkpoint(checkpoint_path, verbose=True):
    if USE_GPU:
        state = torch.load(checkpoint_path)
    else:
        state = torch.load(checkpoint_path, map_location='cpu')
    if verbose:
        print('model loaded from %s' % checkpoint_path)
    return state


def init_weight(weight, method='xavier_normal', gain=np.sqrt(2), mean=0):
    """Initialize weights with methods provided by nn.init (in place)

    Args:
        weight: a Variable
        method: string key for init method
        gain: init gain
    """
    if method == 'xavier_normal':
        init.xavier_normal_(weight, gain=gain)
    elif method == 'xavier_uniform':
        init.xavier_uniform_(weight, gain=gain)
    elif method == 'orthogonal':
        init.orthogonal_(weight, gain=gain)
    elif method == 'uniform':
        init.uniform_(weight)
    elif method == 'normal':
        init.normal_(weight, mean=mean)
    else:
        raise NotImplementedError('init method %s is not implemented' % method)


def init_fc(fc, method='xavier_normal', gain=np.sqrt(2), has_bias=True):
    """ Initialize a fully connected layer

    Args:
        weight: a Variable
        method: string key for init method
        gain: init gain
    """
    init_weight(fc.weight, method, gain)
    if has_bias:
        init.constant(fc.bias, 0)


def init_rnn(rnn, method='xavier_normal', gain=np.sqrt(2)):
    """ Initialize a multi-layer RNN

    Args:
        weight: a Variable
        method: string key for init method
        gain: init gain
    """
    for layer in range(rnn.num_layers):
        init_rnn_cell(rnn, method, gain, layerfix='_l%i' % layer)


def init_rnn_cell(rnn, method='xavier_normal', gain=np.sqrt(2), layerfix=''):
    """ Initialize an RNN cell (layer)

    Args:
        weight: a Variable
        method: string key for init method
        gain: init gain
        layerfix: postfix of the layer name
    """
    init_weight(getattr(rnn, 'weight_ih' + layerfix), method, gain)
    init_weight(getattr(rnn, 'weight_ih' + layerfix), method, gain)
    init.constant_(getattr(rnn, 'bias_ih' + layerfix), 0)
    init.constant_(getattr(rnn, 'bias_hh' + layerfix), 0)


def to_tensor(np_array, cuda=False, device=None):
    """ Convert a numpy array to a tensor
    """
    if device is not None:
        return torch.from_numpy(np_array).to(device)
    if USE_GPU and cuda:
        return torch.from_numpy(np_array).cuda()
    else:
        return torch.from_numpy(np_array)


def to_numpy(tensor):
    """ Convert a tensor back to numpy array
    """
    if tensor.is_cuda:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().numpy()


def to_onehot(tensor, num_class):
    x = torch.zeros(tensor.size() + (num_class,)).to(tensor.device)
    x.scatter_(-1, tensor.unsqueeze(-1), 1)
    return x


def to_batch_first(tensor):
    """ Convert a tensor from time first to batch first

    Args:
        tensor: [T, B, ...]
    Returns:
        tensor: [B, T, ...]
    """
    return tensor.transpose(0, 1)


def repackage_state(h):
    """Wraps hidden states in new Variables, to detach them
    from their history.

    args:
        h: a Variable
    """
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_state(v) for v in h)


def gather_dim(input_tensor, inds, dim=0):
    """ Gather subset of a tensor on a given dimension with input indices

    Args:
        input_tensor: n-dimensional tensor to gather from
        inds: a numpy array of indices [N]
    Returns:
        gathered dims
    """
    if isinstance(inds, np.ndarray):
        inds = to_tensor(inds, device=input_tensor.device)
    return torch.index_select(input_tensor, dim, inds)


def gather_sequence(input_sequence, indices):
    """
    Given a batch of sequence, gather an element from each sequence
    :param input_sequence: [B, T, ...]
    :param indices: [T, ...]
    :return: [B, ...]
    """
    # make inds shape [1, T, ...]
    indices = indices.view(-1, *([1]*(input_sequence.ndimension() - 1)))
    out = input_sequence.gather(1, indices.expand(-1, 1, *input_sequence.shape[2:]))
    return out.squeeze(1)


def gather_sequence_n(input_sequence, indices):
    ind_dim = indices.ndimension()
    assert(ind_dim == 2)
    seq_dim = input_sequence.ndimension()
    inds_expand = indices.view(*(indices.shape + (1,) * (seq_dim - ind_dim)))
    out = input_sequence.gather(1, inds_expand.expand(-1, -1, *input_sequence.shape[2:]))
    return out


def unsort_dim(seq, sort_inds, dim=0):
    """ Given a sorted sequence tensor, "unsort" the sequence.
    This function is exclusively used in the dynamic_rnn function
    but it must be useful for other functions...right?

    Args:
        seq: sorted sequence (n-dimensional) to unsort
        sort_inds: the order that sequence is sorted
        dim: on which dimension to unsort
    Returns:
        an unsorted sequence of the origina shape
    """
    inv_inds = safe_cuda(torch.zeros(*sort_inds.shape)).long()
    for i, ind in enumerate(sort_inds):
        inv_inds[ind] = i
    seq = torch.index_select(seq, dim, inv_inds)
    return seq


def to_batch_seq(x, dtype=np.float32):
    """
    convert a single frame to a batched (size 1)
    sequence and to torch variable
    """
    if isinstance(x, int):
        xdim = []
    else:
        xdim = list(x.shape)
    out = np.zeros([1, 1] + xdim, dtype=dtype)
    out[0, 0, ...] = x
    return to_tensor(out)


# def pad_seq_list_to_max_len(seqs, value=0):
#     """
#     pad a list of sequence to their max length
#     inputs:
#         seqs: list([t, ...])
#     outputs:
#         padded_seq: [B, max(t), ...]
#         seq_len: numpy [B]
#     """
#     batch_size = len(seqs)
#     seq_len = np.array([s.size(0) for s in seqs])
#     max_seq_len = seq_len.max()
#
#     pad_len = np.zeros((batch_size, 2), dtype=np.int64)
#     pad_len[:, 1] = max_seq_len - seq_len
#
#     return pad_seq_list(seqs, pad_len, value)
#
#
# def pad_seq_list(seqs, pad_len, value=0):
#     """
#     pad a list of sequence to begin and end
#     inputs:
#         seqs: list([t, ...])
#         pad_len: numpy [B, 2]
#     outputs:
#         padded_seq: [B, T, ...]
#         seq_len: numpy [B]
#     """
#     seq_dim = list(seqs[0].size()[1:])
#     seq_len = np.array([s.size(0) for s in seqs])
#
#     padded_seqs = []
#     for i, seq in enumerate(seqs):
#         bp, ep = pad_len[i]
#         padded_seq = []
#         if bp == 0 and ep == 0:
#             padded_seqs.append(seq.unsqueeze(0))
#             continue
#         if bp > 0:
#             s = [bp] + seq_dim
#             pad = to_tensor(np.ones(s, dtype=np.float32) * value)
#             padded_seq.append(pad)
#         padded_seq.append(seq)
#         if ep > 0:
#             s = [ep] + seq_dim
#             pad = to_tensor(np.ones(s, dtype=np.float32) * value)
#             padded_seq.append(pad)
#
#         padded_seq = torch.cat(padded_seq, 0)
#         padded_seqs.append(padded_seq.unsqueeze(0))
#
#     padded_seqs = torch.cat(padded_seqs, 0)
#     return padded_seqs, seq_len


def pad_sequence_list(seqs, value=0):
    padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=value)
    mask = np.zeros((len(seqs), padded.shape[1]), dtype=np.float32)
    for i, s in enumerate(seqs):
        mask[i, :len(s)] = 1
    mask = to_tensor(mask, device=padded.device)
    seq_len = to_tensor(np.array([len(s) for s in seqs]), device=padded.device)
    return padded, seq_len, mask


def truncate_seq(seqs, trunc_inds):
    """
    truncate a sequence tensor to different length
    inputs:
        seqs: [B, T, ...]
        trunc_inds: numpy [B, 2]
    outputs:
        trunc_seqs: list([t, ...])
    """
    trunc_seqs = []
    split_seqs = seqs.split(1, dim=0)
    for i, seq in enumerate(split_seqs):
        b, e = trunc_inds[i]
        tseq = seq.squeeze(0)[b: e, ...]
        trunc_seqs.append(tseq)
    return trunc_seqs


def flatten(x, begin_axis=1):
    """
    flatten a tensor beginning at an axis
    :param x: tensor to flatten
    :param begin_axis: which axis to begin at
    :return: flattened tensor
    """
    fixed_size = x.size()[:begin_axis]
    _s = list(fixed_size) + [-1]
    return x.view(*_s)


def num_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def module_device(module):
    return next(module.parameters()).device


def compute_jacobian(y, x):
    """
    A hacky way to compute vector-vector jacobian: dy/dx
    :param y vector [M]
    :param x vector [N]
    :return: j [M, N]
    """
    j = []
    for l in y:
        j.append(torch.unsqueeze(compute_gradient(l, x), 0))
    return torch.cat(j, dim=0)


def compute_gradient(y, x):
    return autograd.grad(y, x, create_graph=True)[0]


def batch_matrix(w, batch_size):
    """
    batch a weight matrix
    :param w: [M, N]
    :param batch_size: B
    :return: [B, M, N]
    """
    return w.unsqueeze(0).expand(batch_size, -1, -1)

