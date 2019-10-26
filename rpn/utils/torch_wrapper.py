"""
PyTorch wrapper functions

author: Danfei Xu
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from rpn.utils.torch_utils import init_rnn, init_rnn_cell, init_fc,\
    gather_dim, unsort_dim, flatten, to_tensor
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import math

USE_GPU = torch.cuda.is_available()


def _get_init_args(kwargs, init_method='xavier_normal', init_gain=np.sqrt(2)):
    if 'init_method' in kwargs:
        init_method = kwargs.pop('init_method')
        init_gain = kwargs.pop('init_gain')

    has_bias = True
    if 'bias' in kwargs:
        has_bias = kwargs['bias']
    return init_method, init_gain, has_bias


class FC(nn.Linear):
    """An FC layer wrapper with init option"""
    def __init__(self, *args, **kwargs):
        init_method, init_gain, has_bias = _get_init_args(kwargs)
        nn.Linear.__init__(self, *args, **kwargs)
        init_fc(self, init_method, init_gain, has_bias)


class Conv2d(nn.Conv2d):
    """A Conv2d layer wrapper with init option"""
    def __init__(self, *args, **kwargs):
        init_method, init_gain, has_bias = _get_init_args(kwargs)
        nn.Conv2d.__init__(self, *args, **kwargs)
        init_fc(self, init_method, init_gain, has_bias)


class GRU(nn.GRU):

    def __init__(self, *args, **kwargs):
        init_method, init_gain, has_bias = _get_init_args(kwargs)
        nn.GRU.__init__(self, *args, **kwargs)
        init_rnn(self, init_method, init_gain)

    def zero_state(self, batch_size, cuda=True):
        hidden_layer = self.num_layers
        if self.bidirectional:
            hidden_layer *= 2
        h = Variable(torch.zeros(hidden_layer, batch_size, self.hidden_size))
        if USE_GPU and cuda:
            return h.cuda()
        else:
            return h


class LSTM(nn.LSTM):

    def __init__(self, *args, **kwargs):
        init_method, init_gain, has_bias = _get_init_args(kwargs)
        nn.LSTM.__init__(self, *args, **kwargs)
        init_rnn(self, init_method, init_gain)

    def zero_state(self, batch_size, cuda=True):
        h = Variable(torch.zeros(self.num_layers,
                                 batch_size, self.hidden_size))
        c = Variable(torch.zeros(self.num_layers,
                                 batch_size, self.hidden_size))
        if USE_GPU and cuda:
            return (h.cuda(), c.cuda())
        else:
            return (h, c)


class GRUCell(nn.GRUCell):

    def __init__(self, *args, **kwargs):
        init_method, init_gain, has_bias = _get_init_args(kwargs)
        nn.GRUCell.__init__(self, *args, **kwargs)
        init_rnn_cell(self, init_method, init_gain)

    def zero_state(self, batch_size, cuda=True):
        h = Variable(torch.zeros(batch_size, self.hidden_size))
        if USE_GPU and cuda:
            return h.cuda()
        else:
            return h


class LSTMCell(nn.LSTMCell):

    def __init__(self, *args, **kwargs):
        init_method, init_gain, has_bias = _get_init_args(kwargs)
        nn.LSTMCell.__init__(self, *args, **kwargs)
        init_rnn_cell(self, init_method, init_gain)

    def zero_state(self, batch_size, cuda=True):
        h = Variable(torch.zeros(batch_size, self.hidden_size))
        c = Variable(torch.zeros(batch_size, self.hidden_size))
        if USE_GPU and cuda:
            return (h.cuda(), c.cuda())
        else:
            return (h, c)


class DynamicGRU(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 nlayers=1,
                 bidirectional=False):
        """ A dynamic-length GRU implementation
        wraps the dynamic_rnn function

        args:
            input_size: input dimension
            hidden_size: hidden unit size
            nlayers: number of hidden layers
            bidirectional: if use bidirectional RNN
        """
        super(DynamicGRU, self).__init__()
        if bidirectional:
            hidden_size = hidden_size / 2
        self.encoder = GRU(input_size, hidden_size, nlayers,
                           batch_first=True, bidirectional=bidirectional)

    def forward(self, input, seq_len, init_state=None):
        """
        args:
            input: [B, T, ...]
            seq_len: numpy [B]
            init_state: [num_layer * num_direction, B, H]
        outputs:
            output: [B, T, H * num_direction]
            state: [num_layer * num_direction, B, H]
        """
        return dynamic_rnn(self.encoder, input, seq_len, init_state)


class TimeConv(nn.Module):
    """
    This module implements a 1-D feature Time Convolution network.
    The network assumes the input to be of shape (B, T, D) where
    B is the batch size, T is the max time step, D is the feature
    dimension. The network performs a temporal convolution on the
    T-D 2D tensor with a fixed window size

    Args:
        input_size: input feature dimension
        output_size: output feature dimension
        window_size: temporal convolution window
        out_channel: number of output channels
    """

    def __init__(self, input_size, output_size, window_size, out_channel):
        super(TimeConv, self).__init__()
        assert(window_size % 2 == 1 and window_size > 0)
        padding = (window_size - 1) / 2
        self.conv = Conv2d(1, out_channel,
                           kernel_size=(window_size, 1),
                           padding=(padding, 0))
        conv_out = out_channel * input_size
        self.fc = FC(conv_out, output_size)

    def forward(self, input):
        """
        args:
            input: [B, T, D]
        """
        batch_size, T = input.size()[:2]
        input_c = input.unsqueeze(1)  # [B, 1, T ,D]
        conv_out = F.relu(self.conv(input_c))  # [B, C, T, D]
        conv_out_t = conv_out.transpose(1, 2).contiguous()  # [B, T, C, D]
        conv_out_flat = conv_out_t.view(batch_size, T, -1)  # [B, T, C*D]
        fcout = F.relu(time_distributed(conv_out_flat, self.fc))
        return fcout


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, layer_dims=(),
                 layer_func=nn.Linear, activation=nn.ReLU, dropouts=None,
                 output_activation=None):
        super(MLP, self).__init__()
        layers = []
        dim = input_dim
        if dropouts is not None:
            assert(len(dropouts) == len(layer_dims))
        for i, l in enumerate(layer_dims):
            layers.append(layer_func(dim, l))
            layers.append(activation())
            if dropouts is not None and dropouts[i]:
                layers.append(nn.Dropout(0.5))
            dim = l
        layers.append(layer_func(dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        self._layer_func = layer_func
        self._layers = layers
        self._model = nn.Sequential(*layers)

    @property
    def layers(self):
        return self._layers

    def forward(self, input):
        return self._model(input)


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self._shape = shape
        self._input_shape = None

    def forward(self, inputs):
        self._input_shape = inputs.size()
        return inputs.view(*self._shape)


class Flatten(nn.Module):
    def __init__(self, begin_index):
        super(Flatten, self).__init__()
        self._begin_index = begin_index

    def forward(self, inputs):
        return flatten(inputs, self._begin_index)


def attention_dot(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query.unsqueeze(-2), key.transpose(-2, -1)).squeeze(-2) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn.unsqueeze(-2), value).squeeze(-2), p_attn


def positional_encoding(batch_size, seq_len, enc_dim, device=None):
    """
    Positional encoding with wave function
    :param batch_size: batch size
    :param seq_len: sequence length to encode
    :param enc_dim: dimension of the encoding
    :return: [B, T, D]
    """
    pos_i = np.tile(np.arange(seq_len)[:, None], (1, enc_dim)).astype(np.float32)
    enc_i = np.tile(np.arange(enc_dim)[None, :], (seq_len, 1)).astype(np.float32)
    pos_enc = np.zeros((seq_len, enc_dim), dtype=np.float32)
    pos_enc[:, ::2] = np.sin(pos_i[:, ::2] / (np.power(10000, 2 * enc_i[:, ::2] / enc_dim)))
    pos_enc[:, 1::2] = np.cos(pos_i[:, 1::2] / (np.power(10000, 2 * (enc_i[:, 1::2] + 1) / enc_dim)))
    pos_enc = np.tile(pos_enc[None, ...], (batch_size, 1, 1))
    return to_tensor(pos_enc, device=device)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return x


class Attention(nn.Module):
    """
    A generic sequence attention network
    """

    def __init__(self, mode='dot', dim=None, log_softmax=False):
        super(Attention, self).__init__()
        self.mode = mode
        self.log_softmax = log_softmax
        if self.mode == 'general':
            self.W = FC(*dim, bias=False)
        if self.mode == 'concat':
            in_dim = np.sum(dim)
            self.fc1 = FC(in_dim, dim[1])
            self.fc2 = FC(dim[1], 1)

    def forward(self, src_states, state, ext_weights=None):
        if self.mode == 'dot':
            return self._attend_dot(src_states, state,
                                    ext_weights=ext_weights)
        elif self.mode == 'general':
            return self._attend_general(src_states, state,
                                        ext_weights=ext_weights)
        elif self.mode == 'concat':
            return self._attend_concat(src_states, state,
                                       ext_weights=ext_weights)
        else:
            raise NotImplementedError('%s is not implemented' % self.mode)

    def _attend_general(self, src_states, state, ext_weights=None):
        """
        Perform a general attention by state * W * src_states[i]^T
        where W is a learnable tensor

        Args:
            src_states: [B, T, H]
            state: [B, H]
            ext_weights: [B, T] extra attention weights
        Return:
            context: [B, H]
            w: original weights
        """
        state_proj = self.W(state)
        return self._attend_dot(src_states, state_proj,
                                ext_weights=ext_weights)

    def _attend_dot(self, src_states, state, ext_weights=None):
        """
        dot(src_states, state)
        Args:
            src_states: [B, T, H]
            state: [B, H]
            ext_weights: [B, T] extra attention weights
        Return:
            context: [B, H]
            w_norm: attention weights
        """
        state = state.unsqueeze(2)  # [B, H, 1]
        # [B, T, H] * [B, H, 1] = [B, T, 1]
        w = torch.bmm(src_states, state).squeeze(2)  # [B, T]

        if ext_weights is not None:
            w = w * ext_weights

        w = w / (1e-4 + w.sum(1, keepdim=True).expand_as(w))

        # expand feature to [B, T, H]
        w_norm = w.unsqueeze(2).expand_as(src_states)
        weighted_src_states = src_states * w_norm
        context = torch.sum(weighted_src_states, 1).squeeze(1)
        return context, w_norm

    def _attend_concat(self, src_states, state, ext_weights=None):
        """ A concatentation-based attention implementation

        w = fc2(relu(fc1(concat[src_states, state])))
        Args:
            w: original weights
            w_softmax: weights after a softmax function
            w_sigmoid: weights after a sigmoid function
        Return:
            context: [B, H]
            w_norm: attention weights
        """
        T = src_states.size(1)
        batch_size, state_dim = state.size()
        state_t = state.unsqueeze(1).expand(batch_size, T, state_dim)
        state_cat = torch.cat([src_states, state_t], 2)
        vec = F.relu(time_distributed(state_cat, self.fc1))
        w = time_distributed(vec, self.fc2).squeeze(2)

        if ext_weights is not None:
            w = w * ext_weights

        w_softmax = nn.Softmax(dim=-1)(w)

        w_norm = w_softmax.unsqueeze(2).expand_as(src_states)
        weighted_src_states = src_states * w_norm
        context = torch.sum(weighted_src_states, 1).squeeze(1)
        return context, w_norm


def time_distributed(inputs, op, activation=None, **kwargs):
    """
    apply op on both the batch (B) and time (T) dimension
    args:
        inputs: [B, T, ...]
        op: a layer op that accepts [B, ...]
        reshape: if reshape to [B, T, newshape]
    return:
        outputs: [B, T, ...]
    """

    batch_size = inputs.size(0)
    seq_len = inputs.size(1)
    p_dim = inputs.size()[2:]
    outputs = op(inputs.view(-1, *p_dim), **kwargs)
    out_dim = outputs.size()[1:]
    if activation is not None:
        outputs = activation(outputs)
    outputs = outputs.view(batch_size, seq_len, *out_dim)
    return outputs


def apply_once(x, func):
    return func(x)


def dynamic_rnn(rnn_module, input_seq, seq_len, init_state=None):
    """
    Unrolls a variable length RNN, assumes batch-first input
    args:
        rnn_module: an RNN module
        input_seq: tensor [B, T, *]
        seq_len: numpy array [B]
        init_state: tensor [nlayer, B, *] or None
    return:
        output: tensor [B, max(seq_len), *]
        state: tensor [nlayer*ndirection, B, *]
    """

    if init_state is None:
        batch_size = seq_len.shape[0]
        init_state = rnn_module.zero_state(batch_size)

    # sort sequences by length
    sort_inds = np.array(np.argsort(seq_len)[::-1])
    input_seq = gather_dim(input_seq, sort_inds)
    init_state = gather_dim(init_state, sort_inds, dim=1)
    seq_len = seq_len[sort_inds]

    input_seq_packed = rnn_utils.pack_padded_sequence(input_seq,
                                                      seq_len,
                                                      batch_first=True)

    # shovel'em through the rnn
    output_packed, state = rnn_module(input_seq_packed, init_state)

    # reverse the sorting
    output, _ = rnn_utils.pad_packed_sequence(output_packed,
                                              batch_first=True)
    output = unsort_dim(output, sort_inds, dim=0)
    state = unsort_dim(state, sort_inds, dim=1)
    return output, state


def time_masked_loss(loss_op, preds, labels, time_mask, use_mask=False):
    """
    compute loss between preds and labels using loss_op
    element-wise loss is masked by a time_mask, which is a tensor that
    specifies the length of the sequence if use_mask is False and
    a element-wise binary mask if use_mask is True
    args:
        preds: tensor [B, T, N]
        labels: tensor [B, T]
        time_mask: numpy array [B, (T)]
    output:
        loss: float
    """
    max_seq_len = preds.size(1)
    pred_dim = preds.size()[2:] if len(preds.size()) > 2 else []
    label_dim = labels.size()[2:] if len(labels.size()) > 2 else []
    labels = labels[:, :max_seq_len].contiguous()  # trim extra paddings

    gather_inds = []
    for b, m in enumerate(time_mask):
        new_inds = np.where(m)[0] if use_mask else np.arange(m)
        gather_inds += (new_inds + max_seq_len * b).tolist()

    gather_inds = np.array(gather_inds)
    if gather_inds.size == 0:
        return None

    labels = gather_dim(labels.view(-1, *label_dim), gather_inds)
    preds = gather_dim(preds.view(-1, *pred_dim), gather_inds)

    return loss_op(preds, labels)


def embed_time_masked_loss(loss_op, preds, labels, time_mask):
    """
    compute loss between preds and labels using loss_op
    element-wise loss is masked by a time_mask, which is a tensor that
    specifies the length of the sequence if use_mask is False and
    a element-wise binary mask if use_mask is True
    args:
        preds: tensor [B, T1, T2, N]
        labels: tensor [B, T1, T2]
        time_mask: numpy array [B, T1, T2]
    output:
        loss: float
    """
    max_seq_len1 = preds.size(1)
    max_seq_len2 = preds.size(2)
    pred_dim = preds.size()[3:] if len(preds.size()) > 3 else []
    label_dim = labels.size()[3:] if len(labels.size()) > 3 else []
    # trim extra paddings
    labels = labels[:, :max_seq_len1, :max_seq_len2].contiguous()

    gather_inds = []
    for b, m1 in enumerate(time_mask):
        for t1, m2 in enumerate(m1):
            new_inds = np.where(m2)[0]
            gather_inds += (new_inds + max_seq_len2 *
                            (t1 + max_seq_len1 * b)).tolist()

    gather_inds = np.array(gather_inds)
    assert(np.all(time_mask.ravel()[gather_inds]))
    assert(time_mask.sum() == time_mask.ravel()[gather_inds].sum())
    if gather_inds.size == 0:
        return None

    labels = gather_dim(labels.view(-1, *label_dim), gather_inds)
    preds = gather_dim(preds.view(-1, *pred_dim), gather_inds)

    return loss_op(preds, labels)