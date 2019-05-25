import torch.nn as nn

from ..abs import BridgeBase


class Bridge(BridgeBase):
    r"""
    Bridge the connection between encoder and decoder.
    """

    def __init__(self, args):
        super().__init__(args)
        self.num_layers = args.num_layers
        self.dec_init = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size * args.num_layers),
                                      nn.Tanh())


    def forward(self, encoder_states):
        encoder_state,r2l_state=encoder_states
        ctx=encoder_state['ctx']
        mask = encoder_state['mask']
        h = self.dec_init(encoder_state['h'])

        if h.ndimension() == 2:
            # B x 1 x D
            h = h.unsqueeze(1)
        hs=h.chunk(self.num_layers,-1)
        state = {
            'encoder': {
                'ctx': ctx,
                'mask': mask
            },
            'r2l':{
                'ctx':r2l_state['ctx'],
                'mask':r2l_state['mask']
            }
        }
        for i in range(self.num_layers):
            state['l%d' % i] = {
                'prev_state': hs[i].contiguous(),
            }
        return state
