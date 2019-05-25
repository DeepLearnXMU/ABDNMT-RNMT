import torch
import torch.nn.functional as F

from thseq.criterions.label_smoothed_cross_entropy import smoothed_cross_entropy
from thseq.modules.search import beamsearch
from thseq.utils.nested import clone,shape
from .bridge import Bridge
from .decoder import Decoder
from .encoder import Encoder
from ..abs import Seq2SeqBase
from ...models import register_model
from .r2l_encoder import R2LEncoder

@register_model('mulsrc')
class Seq2Seq(Seq2SeqBase):
    def __init__(self, args, vocabularies):
        self.args = args
        encoder = Encoder(args, vocabularies[0])
        decoder = Decoder(args, vocabularies[1])

        bridge = Bridge(args)

        super().__init__(args, encoder, bridge, decoder)
        self.r2l_encoder=R2LEncoder(args,vocabularies[1])
        self.r2l_encoder.embedding=self.decoder.dec.embedding

    @staticmethod
    def add_args(parser):
        parser.add_argument('--hidden-size', type=int, default=512)
        parser.add_argument('--hidden-dropout', type=float, default=0.3)
        parser.add_argument('--relu-dropout', type=float, default=0.0)
        parser.add_argument('--attention-dropout', type=float, default=0.0)
        parser.add_argument('--weight-tying', type=int, default=1)
        parser.add_argument('--num-layers', type=int, default=4)
        parser.add_argument('--num-heads', type=int, default=8)
        parser.add_argument('--ver', type=int, default=2,choices=[1,2,3])
        parser.add_argument('--method', type=str, default='residual',choices=['residual','tanh'])
        parser.add_argument('--tanh-coef',type=float,default=1.0)


    @classmethod
    def build(cls, args, vocabularies):
        return cls(args, vocabularies)

    def encode(self, x,r2l_y):
        state = self.encoder(x)
        r2l_state=self.r2l_encoder(r2l_y)
        return state,r2l_state

    def decode(self, y, state, is_inference=False):
        logit, state = self.decoder(y, state, is_inference)
        return logit, state

    def shift_input(self, y):
        # Reuse eos as bos to left-pad the target.
        y = F.pad(y, [1, 0], mode='constant', value=self.decoder.vocabulary.eos_id)
        # Exclude eos from y
        y = y[:, :-1]
        return y

    def forward(self, x, y_r2l, y_l2r):
        state = self.encode(x,y_r2l)
        state = self.bridge(state)

        l2r_input = self.shift_input(y_l2r)

        logit, _ = self.decoder(l2r_input, state)
        loss, nll_loss = smoothed_cross_entropy(logit, y_l2r, self.decoder.vocabulary.pad_id)

        return loss, nll_loss

    def infer(self, x, beam_width,
              max_sequence_length=None, length_normalization_factor=0, length_normalization_const=5.0,
              retain_attn=False):
        x,y_r2l=x # unpack
        def fn(y, state):
            logit, state = self.decoder(y, state, True)
            return self.get_normalized_probs(logit, True), state

        self.eval()
        with torch.no_grad():
            state = self.encode(x,y_r2l)
            state = self.bridge(state)

            lens = x.ne(self.encoder.vocabulary.pad_id).sum(1)
            hyps = beamsearch(fn, state, lens, lens.size(0),
                              beam_width=beam_width,
                              eos=self.decoder.vocabulary.eos_id,
                              expand_args=False)
        hyps = [self.decoder.vocabulary.to_words(hyp.nodes)
                for hyp in hyps]
        return hyps

    def translate(self, x, beam_width,
                  max_sequence_length=None, length_normalization_factor=0, length_normalization_const=5.0,
                  retain_attn=False):
        return self.infer(x, beam_width,
                          max_sequence_length=None, length_normalization_factor=0, length_normalization_const=5.0,
                          retain_attn=False)

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

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return super().state_dict(destination, prefix, keep_vars)
