import math
from typing import List

import torch
import torch.nn as nn

from .abs import Seq2SeqBase, EncoderBase, DecoderBase, BridgeBase


def index(args: tuple, kwargs: dict, i: int):
    return [arg[i] if isinstance(arg,list) else arg for arg in args], \
           dict([(k, v[i] if isinstance(v,list) else v) for k, v in kwargs.items()])


class Encoder(EncoderBase):
    def __init__(self, models: List[Seq2SeqBase]):
        super().__init__(models[0].encoder.args, models[0].encoder.vocabulary)
        self.models = nn.ModuleList(models)

    def forward(self, *args, **kwargs):
        states = []
        for i, model in enumerate(self.models):
            # args_i,kwargs_i=index(args,kwargs,i)
            args_i,kwargs_i=args,kwargs
            states.append(model.encode(*args_i,**kwargs_i))
        return states


class Bridge(BridgeBase):

    def __init__(self, models: List[Seq2SeqBase]):
        super().__init__(models[0].bridge.args)
        self.models = nn.ModuleList(models)

    def forward(self, *args, **kwargs):
        states = []
        for i,model in enumerate(self.models):
            args_i,kwargs_i=index(args,kwargs,i)
            states.append(model.bridge(*args_i, **kwargs_i))
        return states


class Decoder(DecoderBase):

    def __init__(self, models: List[Seq2SeqBase]):
        super().__init__(models[0].decoder.args, models[0].decoder.vocabulary)
        self.models = nn.ModuleList(models)

    def forward(self, *args, **kwargs):
        log_probs = []
        states = []
        for i,model in enumerate(self.models):
            args_i,kwargs_i=index(args,kwargs,i)
            logit, state = model.decode(*args_i, **kwargs_i)
            log_prob = model.get_normalized_probs(logit, True)
            log_probs.append(log_prob)
            states.append(state)

        return log_probs, states


class AvgPrediction(Seq2SeqBase):

    def __init__(self, models: List[Seq2SeqBase], weights=None):
        super().__init__(models[0].args, Encoder(models), Bridge(models), Decoder(models))
        self.models = models
        self.weights = weights
        self.num_model = len(models)

    def decode(self, y, state):
        log_probs, states = super().decode(y, state)
        log_probs = torch.logsumexp(torch.stack(log_probs, 0), 0) - math.log(self.num_model)
        return log_probs, states

    def forward(self, *args, **kwargs):
        raise NotImplementedError()
