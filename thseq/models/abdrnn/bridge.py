import torch.nn as nn

from ..abs import BridgeBase


class Bridge(BridgeBase):
    r"""
    Bridge the connection between encoder and decoder.
    """

    def __init__(self, args):
        super().__init__(args)
        self.num_layers = args.num_layers
        self.dec_init = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size *2* args.num_layers),
                                      nn.Tanh())

    def forward(self, encoder_state):
        ctx=encoder_state['ctx']
        mask = encoder_state['mask']
        h = self.dec_init(encoder_state['h'])

        if h.ndimension() == 2:
            # B x 1 x D
            h = h.unsqueeze(1)
        hs=h.chunk(2* self.num_layers,-1)
        hs0 = hs[:self.num_layers]
        hs1=hs[self.num_layers:]
        state_r2l = {
            'encoder': {
                'ctx': ctx,
                'mask': mask
            },
        }
        for i in range(self.num_layers):
            state_r2l['l%d' % i] = {
                'prev_state': hs0[i].contiguous(),
            }
        state_l2r={
            'encoder': {
                'ctx': ctx,
                'mask': mask
            },
        }
        for i in range(self.num_layers):
            state_l2r['l%d' % i] = {
                'prev_state': hs1[i].contiguous(),
            }
        return {
            'l2r':state_l2r,
            'r2l':state_r2l
        }
