import torch
import torch.nn as nn

from thseq.models import DecoderBase
from thseq.modules.attention import Attention

get_output = lambda hidden: hidden[0] if isinstance(hidden, tuple) else hidden

from thseq.nn.ff import Maxout


class Decoder(DecoderBase):
    def __init__(self, args, vocabulary):
        super().__init__(args, vocabulary)

        input_size = args.input_size
        hidden_size = args.hidden_size
        ctx_size = 2 * hidden_size

        # self.embedding = Embedding(len(vocabulary), input_size,True, vocabulary.pad_id)
        self.embedding = nn.Embedding(len(vocabulary), input_size, vocabulary.pad_id)
        self.dropout_input = nn.Dropout(args.dropout_input)
        self.dropout_hidden = nn.Dropout(args.dropout_hidden)

        self.cell0 = nn.GRUCell(input_size, hidden_size)
        self.attention = Attention(hidden_size, args.attention_size)
        self.cell1 = nn.GRUCell(ctx_size, hidden_size)

        readout = nn.Sequential(
            nn.Linear(input_size + hidden_size + ctx_size, input_size),
            nn.Tanh()
        )
        # maxout = Maxout(input_size + hidden_size + ctx_size, hidden_size // 2, 2)
        # linear=nn.Linear(hidden_size//2,input_size)

        dropout_output = nn.Dropout(args.dropout_hidden)
        logit = nn.Linear(input_size, len(vocabulary))

        if args.weight_tying:
            logit.weight = self.embedding.weight

        self.generator = nn.Sequential(
            # maxout,
            # linear,
            readout,
            dropout_output,
            logit
        )

    def forward(self, inputs, state,is_inference=False):
        if is_inference:
            inputs=inputs[:,-1:]
        x = self.embedding(inputs)
        x = self.dropout_input(x)
        K, V, mask, h = state['key'], state['value'], state['mask'], state['hidden']
        hs = []
        ctxs = []
        for x_i in x.split(1, 1):
            x_i = x_i.squeeze(1)  # B x D
            h = self.cell0(x_i, h)
            ctx, _ = self.attention(h, K, V, mask)
            h = self.cell1(ctx, h)
            hs.append(h)
            ctxs.append(ctx)
        hs = torch.stack(hs, 1)
        ctxs = torch.stack(ctxs, 1)
        logit = self.generator(torch.cat([x, hs, ctxs], -1))
        state['hidden'] = h
        return logit, state
