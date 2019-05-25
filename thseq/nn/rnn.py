import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn
from torch.nn import Parameter
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence

import thseq.utils.tensor as utils
from thseq.utils.types import to_bool


def cell_type_factory(mode='gru'):
    mode = mode.lower()
    mappnig = {
        'gru': GRUCell,
    }
    return mappnig[mode]


def layer_type_factory(mode='gru'):
    mode = mode.lower()
    mappnig = {
        'gru': GRU,
    }
    return mappnig[mode]


class RNNCellBase(nn.modules.rnn.RNNCellBase):
    """
    A single RNN cell
    """

    def __init__(self, input_size, hidden_size, bias=True, batch_first=False, bias_forget=False, layer_norm=False,
                 learn_init=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.bias_forget = bias_forget
        self.layer_norm = layer_norm
        self.learn_init = learn_init
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.init_state = None  # need to be initialized in subclasses
        self.reset_parameters()

    def nonlinear(self, x):
        f = {'tanh': torch.tanh, 'relu': F.relu}
        return f[self.non_linearity](x)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        hx = self.get_init(input.size(0), hx)

        self.check_forward_input(input)
        self.check_forward_hidden(input, hx)

        hidden = self._forward(input, hx)

        return hidden

    def _forward(self, input, hx=None):
        r"""Defines the computation performed by a recurrent cell.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def get_init(self, batch_size, h0=None):
        if h0 is None:
            hidden_size = self.hidden_size
            assert self.init_state is not None
            h0 = self.init_state
            if h0.dim() > 1:
                h0 = tuple([h.expand(batch_size, hidden_size) for h in h0.split(1, 0)])
            else:
                h0 = h0.expand(batch_size, hidden_size)
        return h0


class GRUCell(RNNCellBase):
    r"""A gated recurrent unit (GRU) cell

    .. math::

        \begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \end{array}

    where :math:`\sigma` is the sigmoid function.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: `True`

    Inputs: input, hidden
        - **input** of shape `(batch, input_size)`: tensor containing input features
        - **hidden** of shape `(batch, hidden_size)`: tensor containing the initial hidden
          state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: h'
        - **h'** of shape `(batch, hidden_size)`: tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(3*hidden_size x input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(3*hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(3*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(3*hidden_size)`

    Examples::

        >>> rnn = nn.GRUCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
                hx = rnn(input[i], hx)
                output.append(hx)
    """

    def __init__(self, input_size, hidden_size, bias=True, batch_first=False, bias_forget=False,
                 layer_norm=False, learn_init=False):
        super().__init__(input_size, hidden_size, bias, batch_first, bias_forget, layer_norm, learn_init)
        learn_init = to_bool(learn_init)
        self.init_state = Parameter(torch.zeros(hidden_size), requires_grad=learn_init)
        if layer_norm:
            self.ln_izr = nn.LayerNorm(2 * hidden_size)
            self.ln_hzr = nn.LayerNorm(2 * hidden_size)
            self.ln_in = nn.LayerNorm(hidden_size)
            self.ln_hn = nn.LayerNorm(hidden_size)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias and self.bias_forget:
            # bias z towards 0 to keep long term dependency
            self.bias_ih.data[:self.hidden_size] = -0.5
            self.bias_hh.data[:self.hidden_size] = -0.5

    def _forward(self, input, hx=None):
        w_ih, b_ih = self.weight_ih, self.bias_ih
        w_hh, b_hh = self.weight_hh, self.bias_hh
        gi = F.linear(input, w_ih, b_ih)
        gh = F.linear(hx, w_hh, b_hh)
        i_zr, i_n = gi.split([self.hidden_size * 2, self.hidden_size], -1)
        h_zr, h_n = gh.split([self.hidden_size * 2, self.hidden_size], -1)

        if self.layer_norm:
            i_zr = self.ln_izr(i_zr)
            h_zr = self.ln_hzr(h_zr)
            i_n = self.ln_in(i_n)
            h_n = self.ln_hn(h_n)
        i_r, i_i = i_zr.chunk(2, 1)
        h_r, h_i = h_zr.chunk(2, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        h = newgate + inputgate * (hx - newgate)

        return h


# Adapted from https://github.com/eladhoffer/seq2seq.pytorch/blob/master/seq2seq/models/recurrent.py
class StackedCell(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, bias=True, batch_first=False, bias_forget=False,
                 layer_norm=False, learn_init=False, cell=GRUCell, dropout_h=None, residual=False):
        super(StackedCell, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.bias_forget = bias_forget
        self.layer_norm = layer_norm
        self.dropout_h = dropout_h
        self.residual = residual

        self.dropout = nn.Dropout(dropout_h)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            rnn = cell(input_size, hidden_size, bias, batch_first, bias_forget, layer_norm, learn_init)
            self.layers.append(rnn)
            input_size = hidden_size

    def forward(self, inputs, hidden):
        def select_layer(h_state, i):  # To work on both LSTM / GRU, RNN
            if isinstance(h_state, tuple):
                return tuple([select_layer(s, i) for s in h_state])
            else:
                return h_state[i]

        next_hidden = []
        for i, layer in enumerate(self.layers):
            next_hidden_i = layer(inputs, select_layer(hidden, i))
            output = next_hidden_i[0] if isinstance(next_hidden_i, tuple) \
                else next_hidden_i
            if i + 1 < self.num_layers:  # apply dropout in between
                output = self.dropout(output)
            if self.residual and inputs.size(-1) == output.size(-1):
                inputs = output + inputs
            else:
                inputs = output
            next_hidden.append(next_hidden_i)
        if isinstance(hidden, tuple):  # for compatibility with LSTM
            next_hidden = tuple([torch.stack(h) for h in zip(*next_hidden)])
        else:
            next_hidden = torch.stack(next_hidden)
        return inputs, next_hidden


class RNNBase(nn.Module):
    """
    A single RNN layer
    """

    def __init__(self, input_size, hidden_size, bias=True, batch_first=False, bias_forget=False, layer_norm=False,
                 learn_init=False, cell=GRUCell, dropout_r=None, drop_weight=None,
                 reverse=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.bias_forget = bias_forget
        self.dropout_r = dropout_r
        self.drop_weight = drop_weight
        self.layer_norm = layer_norm
        self.reverse = reverse

        self.cell = cell(input_size, hidden_size, bias, batch_first, bias_forget, layer_norm, learn_init)

    def recurrent(self, input, h0):
        input, batch_sizes = input
        hidden = h0
        inner = self.cell

        output = []
        input_offset = 0
        last_batch_size = batch_sizes[0]
        hiddens = []
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
        for batch_size in batch_sizes:
            step_input = input[input_offset:input_offset + batch_size]
            input_offset += batch_size

            dec = last_batch_size - batch_size
            if dec > 0:
                hiddens.append(tuple(h[-dec:] for h in hidden))
                hidden = tuple(h[:-dec] for h in hidden)
            last_batch_size = batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0]),)
            else:
                hidden = inner(step_input, hidden)

            output.append(hidden[0])
        hiddens.append(hidden)
        hiddens.reverse()

        hidden = tuple(torch.cat(h, 0) for h in zip(*hiddens))
        assert hidden[0].size(0) == batch_sizes[0]
        if flat_hidden:
            hidden = hidden[0]
        output = torch.cat(output, 0)

        return output, hidden

    def recurrent_reverse(self, input, h0):
        input, batch_sizes = input
        hidden = h0
        inner = self.cell

        output = []
        input_offset = input.size(0)
        last_batch_size = batch_sizes[-1]
        initial_hidden = hidden
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
            initial_hidden = (initial_hidden,)
        hidden = tuple(h[:batch_sizes[-1]] for h in hidden)
        for i in reversed(range(len(batch_sizes))):
            batch_size = batch_sizes[i]
            inc = batch_size - last_batch_size
            if inc > 0:
                hidden = tuple(torch.cat((h, ih[last_batch_size:batch_size]), 0)
                               for h, ih in zip(hidden, initial_hidden))
            last_batch_size = batch_size
            step_input = input[input_offset - batch_size:input_offset]
            input_offset -= batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0]),)
            else:
                hidden = inner(step_input, hidden)
            output.append(hidden[0])

        output.reverse()
        output = torch.cat(output, 0)
        if flat_hidden:
            hidden = hidden[0]

        return output, hidden

    def forward(self, input, h0=None):
        is_packed = isinstance(input, PackedSequence)
        if not is_packed:
            lengths = [input.size(0)] * input.size(1)
            input = pack_padded_sequence(input, lengths, self.batch_first)

        if h0 is None:
            h0 = self.cell.get_init(input.batch_sizes[0].item())

        func = self.recurrent_reverse if self.reverse else self.recurrent
        output, hidden = func(input, h0)
        output = PackedSequence(output, input.batch_sizes)
        if not is_packed:
            output = pad_packed_sequence(output)[0]
        # hidden=hidden.unsqueeze(0)
        return output, hidden


class GRU(RNNBase):
    def __init__(self, input_size, hidden_size, bias=True, batch_first=False, bias_forget=False, layer_norm=False,
                 learn_init=False, dropout_r=None, drop_weight=None,
                 reverse=False):
        super().__init__(input_size, hidden_size, bias, batch_first, bias_forget, layer_norm, learn_init, GRUCell,
                         dropout_r, drop_weight, reverse)


# Adapted from https://github.com/eladhoffer/seq2seq.pytorch/blob/master/seq2seq/models/recurrent.py
class Concat(nn.Sequential):
    r"""A concatenation layer, accepts RNN only

    Inputs: input, hidden
        - **input** of shape `(batch, input_size)` or `(length, batch, input_size)`: tensor containing input features
        - **hidden** of shape `(num_layer, batch, hidden_size)`: tensor containing the initial hidden
          state for each layer each element in the batch.

    Outputs: output, hidden
        - **output** of shape `(batch, hidden_size)` or `(length, batch, hidden_size)`: tensor containing the
        concatenated output for each layer
        - **hidden** of a list of [tensor of shape `()`]

    Examples::

        >>> rnn = nn.GRU(10, 20)
        >>> rnn_reverse = GRU(10, 20, reversed=True)
        >>> birnn = Concat((rnn, rnn_reverse))
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> ox, hx = birnn(input, hx)
    """

    def add_module(self, name, module):
        if not isinstance(module, (RNNBase, nn.RNNBase)):
            raise ValueError(
                'module must be instance of `RNNBase` or `torch.nn.RNNBase`. Got {}'.format(
                    type(module)))
        return super().add_module(name, module)

    def forward(self, inputs, hx=None):
        if hx is None:
            hx = tuple([None] * len(self))
        hidden = []
        outputs = []
        for i, module in enumerate(self._modules.values()):
            curr_output, h = module(inputs, hx[i])
            outputs.append(curr_output)
            hidden.append(h)
        if isinstance(outputs[0], PackedSequence):
            data = torch.cat([pack.data for pack in outputs], -1)
            output = PackedSequence(data, outputs[0].batch_sizes)
        else:
            output = torch.cat(outputs, -1)
        hidden = torch.cat(hidden, -1)
        return output, hidden


class StackedRNN(nn.Module):
    """
    Stacked RNN layer, accepts RNN layers
    """

    def __init__(self, dropout_h=0, residual=False):
        super(StackedRNN, self).__init__()
        self.residual = residual
        self.dropout_h = nn.Dropout(dropout_h)
        self.rnns = nn.ModuleDict()

    def add_module(self, name, module):
        self.rnns[name] = module

    def forward(self, inputs, hidden=None):
        hidden = hidden or tuple([None] * len(self.rnns))
        next_hidden = []
        for i, module in enumerate(self.rnns.values()):
            output, h = module(inputs, hidden[i])
            next_hidden.append(h)

            input_data = inputs.data if isinstance(inputs, PackedSequence) else inputs
            output_data = output.data if isinstance(output, PackedSequence) else output

            if self.residual and input_data.size(-1) == output_data.size(-1):
                input_data = output_data + input_data
            else:
                input_data = output_data

            if isinstance(inputs, PackedSequence):
                inputs = PackedSequence(input_data, inputs.batch_sizes)
            inputs = utils.dropout(inputs, self.training, self.dropout_h)
        return output, next_hidden
