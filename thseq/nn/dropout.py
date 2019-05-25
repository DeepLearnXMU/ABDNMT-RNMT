import torch.nn as nn


class RecurrentDropout(nn.Module):
    def __init__(self, p, locked=True):
        super().__init__()
        self.p = p
        self.locked = True

        self.mask = None

    def clear(self):
        self.mask = None

    def forward(self, input):
        # if self.locked and self.mask is None:
        #     self.mask = input.new_tensor()
        #     nn.Dropout
        raise NotImplementedError
