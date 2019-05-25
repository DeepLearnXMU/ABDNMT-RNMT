"""
NOTE: ONNX doesn't support model exporting with non-tensor
args and return values in `forward()` function.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from thseq.modules.search import beamsearch

from thseq.data.vocabulary import Vocabulary


class EncoderBase(nn.Module):
    def __init__(self, args, vocabulary: Vocabulary):
        super().__init__()
        self.args = args
        self.vocabulary = vocabulary

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return super().state_dict(destination, prefix, keep_vars)


class DecoderBase(nn.Module):
    def __init__(self, args, vocabulary: Vocabulary):
        super().__init__()
        self.args = args
        self.vocabulary = vocabulary

    def forward(self, y, state, is_inference=False):
        raise NotImplementedError()

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return super().state_dict(destination, prefix, keep_vars)


class BridgeBase(nn.Module):
    r"""
    Bridge the connection between encoder and decoder.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, encoder_state):
        raise NotImplementedError


class Seq2SeqBase(nn.Module):
    def __init__(self, args, encoder: EncoderBase, bridge: BridgeBase, decoder: DecoderBase):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.bridge = bridge
        self.decoder = decoder

    def encode(self, x):
        state = self.encoder(x)
        return state

    def decode(self, y, state, is_inference=False):
        logit, state = self.decoder(y, state, is_inference)
        return logit, state

    def forward(self, x, y):
        state = self.encode(x)
        state = self.bridge(state)
        # Reuse eos as bos to left-pad the target.
        y = F.pad(y, [1, 0], mode='constant', value=self.decoder.vocabulary.eos_id)
        # Exclude eos from y
        y = y[:, :-1]
        logit, _ = self.decode(y, state)
        return logit

    def translate(self, x, beam_width,
                  max_sequence_length=None, length_normalization_factor=0, length_normalization_const=5.0,
                  retain_attn=False):

        def fn(y, state):
            logit, state = self.decode(y, state, True)
            return self.get_normalized_probs(logit, True), state

        self.eval()
        with torch.no_grad():
            state=self.encode(x)
            state = self.bridge(state)
            lens = x.ne(self.encoder.vocabulary.pad_id).sum(1)
            hyps = beamsearch(fn, state, lens, lens.size(0),
                              beam_width=beam_width,
                              eos=self.decoder.vocabulary.eos_id,
                              expand_args=False)
        hyps = [self.decoder.vocabulary.to_words(hyp.nodes)
                for hyp in hyps]
        return hyps

    def get_normalized_probs(self, logits, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def get_logits(self, net_output):
        logits = net_output
        return logits

    def get_targets(self, sample):
        """Get targets from either the sample or the net's output."""
        # exclude bos from target
        return sample['target']

    @staticmethod
    def add_args(parser):
        raise NotImplementedError

    @classmethod
    def build(cls, args, vocabularies):
        raise NotImplementedError

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return super().state_dict(destination, prefix, keep_vars)
