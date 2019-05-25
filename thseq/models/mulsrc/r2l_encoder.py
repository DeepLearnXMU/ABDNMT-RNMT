import torch
import torch.nn as nn
from logbook import Logger
from torch.nn.utils.rnn import pad_packed_sequence as unpack, pack_padded_sequence as pack, PackedSequence

from ..abs import EncoderBase
from thseq.nn.embedding import Embedding

class R2LEncoder(EncoderBase):
    def __init__(self, args, vocabulary):
        super().__init__(args, vocabulary)

        self.rnn = nn.GRU(args.hidden_size, args.hidden_size, bidirectional=True, batch_first=True)
        self.embedding = Embedding(len(vocabulary), args.hidden_size, True, padding_idx=vocabulary.pad_id)
        self.dropout_input = nn.Dropout(args.hidden_dropout)
        self.dropout_hidden = nn.Dropout(args.hidden_dropout)
        self.ff = nn.Sequential(nn.Linear(2 * args.hidden_size, args.hidden_size), nn.Tanh())
        # @staticmethod
        # def args_default(args):
        # args.encoder_cudnn = get_value_or_default(args, 'encoder_cudnn', False)
        # args.encoder_input_size = get_value_or_default(args, 'encoder_input_size', 1000)
        # args.encoder_hidden_size = get_value_or_default(args, 'encoder_hidden_size', 1000)
        # args.encoder_directions = get_value_or_default(args, 'encoder_directions', (2, 1))
        # args.encoder_mode = get_value_or_default(args, 'encoder_mode', 'gru')
        # args.encoder_bias = get_value_or_default(args, 'encoder_bias', 1)
        # args.encoder_bias_forget = get_value_or_default(args, 'encoder_bias_forget', 0)
        # args.encoder_layer_norm = get_value_or_default(args, 'encoder_layer_norm', 1)
        # args.encoder_learn_init = get_value_or_default(args, 'encoder_learn_init', 0)
        # args.encoder_dropout_i = get_value_or_default(args, 'encoder_dropout_i', 0.2)
        # args.encoder_dropout_r = get_value_or_default(args, 'encoder_dropout_r', None)
        # args.encoder_dropout_h = get_value_or_default(args, 'encoder_dropout_h', 0.2)
        # args.encoder_dropout_o = get_value_or_default(args, 'encoder_dropout_o', 0.2)
        # args.encoder_drop_weight = get_value_or_default(args, 'encoder_drop_weight', None)
        # args.encoder_residual = get_value_or_default(args, 'encoder_residual', 1)
        # return args

    def forward(self, input):
        self.rnn.flatten_parameters()
        mask = input.eq(self.vocabulary.pad_id)
        # Trim mask when trained with data-parallel
        lens = input.size(1) - mask.sum(1)
        if torch.cuda.device_count() > 1:
            raise RuntimeError()
        sort_i=torch.argsort(lens,descending=True)
        rev_i=torch.argsort(sort_i)
        input=input.index_select(dim=0,index=sort_i)
        lens=lens.index_select(dim=0,index=sort_i)

        x = self.embedding(input)
        x = self.dropout_input(x)
        x = pack(x, lens.tolist(), batch_first=True)
        x, hx = self.rnn(x)
        x, batch_sizes = x.data, x.batch_sizes
        x = self.dropout_hidden(x)  # N x H
        x = self.ff(x)  # N x H/2
        x = PackedSequence(x, batch_sizes)
        x, _ = unpack(x, batch_first=True)

        x=x.index_select(dim=0,index=rev_i)

        summary = x.sum(1) / lens.view(-1, 1).float()

        return {
            'ctx': x,  # B x T x D
            'mask': mask,  # B x T
            'h': summary  # B x D
        }

