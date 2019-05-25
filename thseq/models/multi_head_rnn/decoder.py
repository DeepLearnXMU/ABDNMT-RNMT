import numpy
import torch.nn as nn
import torch.nn.functional as F

from thseq.modules.multihead_attention import MultiheadAttention
from ..abs import DecoderBase


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


class DecoderLayer(nn.Module):
    def __init__(self, args, hidden_size, dropout=0.1, relu_dropout=0.):
        super().__init__()
        self.ln0 = nn.LayerNorm(hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)  # [ctx_s, ctx_t, s_prev]

        self.ln1 = nn.LayerNorm(hidden_size)
        self.encoder_attn = MultiheadAttention(hidden_size, args.num_heads, args.attention_dropout)

        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)

        self.relu_dropout = relu_dropout
        self.dropout = dropout

    def forward(self, inputs, encoder_out, incremental_state):
        # inputs is outputs from previous layer, can be either a single state or a complete sequence.
        # inputs is of shape T x B x D
        self.rnn.flatten_parameters()
        T, B, D = inputs.size()
        encoder_key = encoder_out['ctx'].transpose(0, 1)
        assert list(encoder_key.size())[1:] == [B, D], '%r != %r' % (list(encoder_key.size()), [T, B, D])
        # B x 1 x D -> 1 x B x D
        prev_h = incremental_state['prev_state'].transpose(0, 1)

        if not self.training:
            assert list(inputs.size()) == [1, B, D], list(inputs.size())

        x = inputs
        # 1. res_nd(rnn)
        residual = x
        x = self.ln0(x)
        # T x B x D, 1 x B x D
        x, prev_h = self.rnn(x, prev_h)
        # 1 x B x D -> B x 1 x D
        incremental_state['prev_state'] = prev_h.transpose(0, 1)
        # assert list(x.size()) == list(inputs.size()), '%r != %r' % (
        #     list(x.size()), list(inputs.size()))
        x = F.dropout(x, self.dropout, self.training)
        x = x + residual

        # 2. res_nd(src_attn)
        residual = x
        x = self.ln1(x)
        if 'encoder' not in incremental_state:
            incremental_state['encoder'] = {}
        x = self.encoder_attn(
            query=x,
            key=encoder_key,  # T x B x D
            value=encoder_key,
            key_padding_mask=encoder_out['mask'],  # B x T
            incremental_state=incremental_state['encoder'],
            attn_mask=None,
            static_kv=True
        )
        x = F.dropout(x, self.dropout, self.training)
        x = x + residual

        # 3. res_nd(ffl)
        residual = x
        x = self.ln2(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.relu_dropout, self.training)
        x = self.fc2(x)
        x = F.dropout(x, self.dropout, self.training)
        x = x + residual

        x = x.contiguous()
        return x


class Decoder(DecoderBase):
    def __init__(self, args, vocabulary):
        super().__init__(args, vocabulary)
        hidden_size = args.hidden_size
        dropout = args.hidden_dropout
        num_layers = args.num_layers

        self.embedding = nn.Embedding(len(vocabulary), hidden_size, vocabulary.pad_id)
        self.scaling = numpy.sqrt(args.hidden_size)
        self.layers = nn.ModuleList()
        self.layers.extend([DecoderLayer(args, hidden_size, dropout, args.relu_dropout)
                            for _ in range(num_layers)])
        self.ln = nn.LayerNorm(hidden_size)

        self.logit = nn.Linear(hidden_size, len(vocabulary))

        if args.weight_tying:
            self.logit.weight = self.embedding.weight

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, inputs, state, is_inference=False):

        if is_inference:
            inputs = inputs[:, -1:]
        # B x T -> T x B
        x = inputs.t()
        # inputs: T x B x D
        x = F.dropout(self.embedding(x), self.dropout, self.training)
        x = x * self.scaling

        for i, layer in enumerate(self.layers):
            x = layer(x, state['encoder'], state['l%d' % i])
        x = self.ln(x)

        # T x B x V -> B x T x V
        logits = self.logit(x).transpose(0, 1).contiguous()
        return logits, state
