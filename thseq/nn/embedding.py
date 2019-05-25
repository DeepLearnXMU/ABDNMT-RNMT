import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim, bias=False, padding_idx=None, max_norm=None, norm_type=2,
                 scale_grad_by_freq=False, sparse=False, pretrain_weight=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse,
                         pretrain_weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(embedding_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        input = super().forward(input)
        if self.bias is not None:
            input = input + self.bias
        return input


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, input_size, length=5000, min_timescale=1.0, max_timescale=1.0e4):
        super().__init__()

        signal = get_timing_signal(input_size, length, min_timescale, max_timescale)

        self.register_buffer('pe', signal)

    def forward(self, input):
        pe = input.new(self.pe[:, :input.size(1)])
        return pe


def get_timing_signal(input_size, length, min_timescale=1.0, max_timescale=1.0e4):
    channels = input_size

    position = torch.arange(length).float()
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales).float() * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
    if input_size % 2 > 0:
        signal = F.pad(signal, (0, 1))
    signal = signal.view([1, length, channels])
    return signal


class LearnedPositionEmbedding(nn.Module):
    def __init__(self, input_size, max_len):
        super().__init__()

        self.pe = nn.Embedding(max_len, input_size)

        self.max_len = max_len

    def forward(self, input):
        length = input.size(1)
        positions = input.new_tensor(torch.arange(length)).long()
        if length > self.max_len:
            positions[self.max_len:] = self.max_len - 1

        pe = self.pe(positions).unsqueeze(0)  # expand batch dimension

        return pe
