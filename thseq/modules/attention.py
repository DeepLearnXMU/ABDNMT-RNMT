import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, query_size, attention_size):
        super(Attention, self).__init__()
        self.query_size = query_size
        self.key_size = attention_size
        self.map_query = nn.Linear(query_size, attention_size)
        self.v = nn.Linear(attention_size, 1)

    def forward(self, query, keys, values, mask):
        """
        alpha = v^T * (tanh(W * K + U * Q))
        Args:
            query: B x D
            keys: B x T x D
            values: B x T x D
            mask: B x T

        Returns:

        """
        # B x T x D
        x = keys + self.map_query(query).unsqueeze(1)
        # B x T
        x = self.v(torch.tanh(x)).squeeze(-1)
        x.data.masked_fill_(mask, -float('inf'))
        x = F.softmax(x, -1)
        output = torch.bmm(x.unsqueeze(1), values).squeeze(1)
        return output, x


def scaled_dot_attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # d_k = query.size(-1) mistake!
    d_k = key.size(-1)
    scaling = d_k ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scaling
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def additive_attention(query, key, value, V, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    b, h, t, d = key.size()
    # V.size() : h,d,1
    scores = torch.tanh(key + query)  # b,h,t,d
    scores = scores.transpose(1, 0).contiguous().view(h, b * t, d)  # h,b*t,d
    scores = torch.bmm(scores, V)  # h,b*t,1
    scores = scores.view(h, b, t, 1).transpose(1, 0)  # b,h,t

    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    r"""Multi-head Attention

    Supports reusing and dynamically extending projected keys and values to reduce computation overheads.
    We can determine the behaviors by feeding different combinations of inputs to forward function.
    1. kv=None and kv_memory=None:
        This can be used in trivial self-attention, where the whole input sequence is known
        and each input tensor serves as query, key and value.
    2. kv=(tensor, tensor) and kv_memory=None:
        Assume the keys and values are not projected and do projections.
    3. kv=None and kv_memory=(tensor,tensor):
        Reuse the projected keys and values.
        For example, this is used in decoder-encoder cross-attention.
    4. kv=(tensor,tensor) and kv_memory=(tensor,tensor) and expand_memory:
        Reuse the projected keys and values and extend them by new keys and values.
        This is used in masked self-attention, where only the partial input sequence is exposed to current query.

    Args:
        num_head: number of heads
        input_sizes: a tuple representing (query_size, key_size, value_size).
            In different cases, keys are sometimes used as values and queries may be used as both keys and values.
            We can use this known condition to simplify the calculation.
            1) When input_sizes=(query_size, None, None), query will be used as key and value;
            2) When input_sizes=(query_size, key_size, None), key will be used as value.
            The calculation is reduced to 1 linear projection in case 1) and 2 projections in case 2).
            Otherwise, there will be 3 linear projections.
        hidden_size: projection size for query and key.
        output_value_size: projection size for value
        dropout:
        attention_type: 'scaled_dot' or 'additive'.

    Inputs:
        query: a 3-d tensor of shape (batch, length_q, hidden_size) to be transformed
        kv: a tuple of key and value tensor to be transformed, (key, value)
        kv_memory: a tuple of key and value tensor already transformed, (key, value)
        mask: a 3-d tensor of shape (batch, length_q, length_k)
        expand_memory: when kv_memory are presented, expand newly transformed kv to kv_memory

    outputs:
        out: a tensor of weighted average of values.
        kv_memory: current keys and values.


    """

    def __init__(self, num_head, input_sizes, hidden_size, output_value_size, dropout=None, mode='scaled_dot'):

        """


        """
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert hidden_size % num_head == 0
        assert isinstance(input_sizes, (tuple, list)) and len(input_sizes) == 3
        self.num_head = num_head

        query_size, key_size, value_size = input_sizes

        def add_nonlinear(module):
            if mode == 'additive':
                module.add_module('1', nn.Tanh())

        if key_size is None:
            self.map_qkv = nn.Sequential(nn.Linear(query_size, 2 * hidden_size + output_value_size))
            add_nonlinear(self.map_qkv)
        elif value_size is None:
            self.map_q = nn.Linear(query_size, hidden_size)
            self.map_kv = nn.Linear(key_size, hidden_size + output_value_size)
            add_nonlinear(self.map_q)
            add_nonlinear(self.map_kv)
        else:
            self.map_q = nn.Linear(query_size, hidden_size)
            self.map_k = nn.Linear(key_size, hidden_size)
            self.map_v = nn.Linear(value_size, hidden_size)
            add_nonlinear(self.map_q)
            add_nonlinear(self.map_k)
            add_nonlinear(self.map_v)

        self.linear_out = nn.Linear(output_value_size, output_value_size)

        self.scores = None

        self.dropout = nn.Dropout(dropout) if dropout else None

        self.V = None
        if mode == 'additive':
            self.V = nn.Parameter(torch.rand(num_head, hidden_size // num_head, 1))

        self.attention_type = mode

        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.hidden_size = hidden_size
        self.output_value_size = output_value_size

    def forward(self, query, kv, kv_memory, mask=None, expand_memory=False):

        if kv is None:
            kv = (None, None)
        if kv_memory is None:
            kv_memory = (None, None)

        if not isinstance(kv, (tuple, list)):
            kv = (kv,)
        if not isinstance(kv_memory, (tuple, list)):
            kv_memory = (kv_memory,)

        squeeze = query.dim() == 2
        if squeeze:
            query = query.unsqueeze(1)

        if mask is not None:
            mask = mask.unsqueeze(1)  # broadcast to num_head dim

        k, v = kv if kv is not None else (None, None)

        batch_size = query.size(0)
        # 1) Do all the linear projections
        if self.key_size is None:
            qkv = self.map_qkv(query)  # (batch_size, length, dim)
            q, k, v = qkv.split([self.hidden_size, self.hidden_size, self.output_value_size], -1)
        elif self.value_size is None:
            q = self.map_q(query)  # (batch_size, length, dim)
            if k is not None:
                kv = self.map_kv(kv[0])  # (batch_size, length, dim)
                k, v = kv.split([self.hidden_size, self.output_value_size], -1)
            else:
                k, v = kv_memory

        else:
            q = self.map_q(query)  # (batch_size, length, dim)
            if k is not None:
                k = self.map_k(kv[0])  # (batch_size, length, dim)
                v = self.map_v(kv[1])  # (batch_size, length, dim)
            else:
                k, v = kv_memory

        if expand_memory:
            k_memory, v_memory = kv_memory
            if k_memory is not None:
                k = torch.cat([k_memory, k], 1)  # concatenate on length dimension
                v = torch.cat([v_memory, v], 1)

        kv_memory = k, v

        # split into heads
        # (batch, num_head, length, dim)
        q, k, v = [x.view(x.size(0), x.size(1), self.num_head, x.size(-1) // self.num_head).transpose(1, 2)
                   for x in (q, k, v)]

        # 2) Apply attention on all the projected vectors in batch.
        if self.attention_type == 'scaled_dot':
            x, self.scores = scaled_dot_attention(q, k, v, mask=mask,
                                                  dropout=self.dropout)
        elif self.attention_type == 'additive':
            x, self.scores = additive_attention(q, k, v, self.V, mask=mask,
                                                dropout=self.dropout, )
        else:
            raise NotImplementedError
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.output_value_size)
        out = self.linear_out(x)
        if squeeze:
            out = out.squeeze(1)
        return out, kv_memory
