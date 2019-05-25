import torch.nn as nn

from ..abs import BridgeBase


class Bridge(BridgeBase):
    r"""
    Bridge the connection between encoder and decoder.
    """

    def __init__(self, args):
        super().__init__(args)
        self.map_k = nn.Sequential(nn.Linear(args.hidden_size * 2, args.attention_size),
                                   nn.Tanh())
        self.dec_init = nn.Sequential(nn.Linear(args.hidden_size * 2, args.hidden_size),
                                      nn.Tanh())

    def forward(self, encoder_state):
        K = self.map_k(encoder_state['feature'])
        h0 = self.dec_init(encoder_state['hidden'])
        return {
            'key': K,
            'value': encoder_state['feature'],
            'mask': encoder_state['mask'],
            'hidden': h0,
        }
