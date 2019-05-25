import torch
import torch.nn as nn
from logbook import Logger
from torch.nn.utils.rnn import pad_packed_sequence as unpack, pack_padded_sequence as pack

from thseq.models import EncoderBase
from thseq.nn.embedding import Embedding

logger = Logger()


class Encoder(EncoderBase):
    def __init__(self, args, vocabulary):
        super().__init__(args, vocabulary)

        self.rnn = nn.GRU(args.input_size, args.hidden_size, bidirectional=True, batch_first=True)
        # self.embedding = Embedding(len(vocabulary), args.input_size, True, padding_idx=vocabulary.pad_id)
        self.embedding = nn.Embedding(len(vocabulary), args.input_size, padding_idx=vocabulary.pad_id)
        self.dropout_input = nn.Dropout(args.dropout_input)
        self.dropout_hidden = nn.Dropout(args.dropout_hidden)
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

    def forward(self, input, h0=None):
        mask = input.eq(self.vocabulary.pad_id)
        # Trim mask when trained with data-parallel
        lens = input.size(1) - mask.sum(1)
        if torch.cuda.device_count() > 1:
            max_len = lens.max().item()
            mask = mask[:, :max_len]
        x = self.embedding(input)
        x = self.dropout_input(x)
        x = pack(x, lens.tolist(), batch_first=True)
        x, hx = self.rnn(x, h0)
        x, _ = unpack(x, batch_first=True)
        x = self.dropout_hidden(x)

        # summary = x.sum(1) / lens.view(-1, 1).float()
        # summary=hx[1]
        summary = x.sum(1) / lens.view(-1,1).float()

        return {
            'feature': x,
            'hidden': summary,
            'mask': mask
        }
