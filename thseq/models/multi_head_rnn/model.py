from ..abs import Seq2SeqBase
from ...models import register_model
from .bridge import Bridge
from .decoder import Decoder
from .encoder import Encoder


@register_model('multi-head-rnn')
class Seq2Seq(Seq2SeqBase):
    def __init__(self, args, vocabularies):
        encoder = Encoder(args, vocabularies[0])
        decoder = Decoder(args, vocabularies[1])
        bridge = Bridge(args)
        super().__init__(args, encoder, bridge, decoder)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--hidden-size', type=int, default=512)
        parser.add_argument('--hidden-dropout', type=float, default=0.3)
        parser.add_argument('--relu-dropout', type=float, default=0.0)
        parser.add_argument('--attention-dropout', type=float, default=0.0)
        parser.add_argument('--weight-tying', type=int, default=1)
        parser.add_argument('--num-layers', type=int, default=4)
        parser.add_argument('--num-heads', type=int, default=8)

    @classmethod
    def build(cls, args, vocabularies):
        return cls(args, vocabularies)


    def get_logits(self, net_output):
        logits = net_output
        return logits

    def get_targets(self, sample):
        """Get targets from either the sample or the net's output."""
        # exclude bos from target
        return sample['target']


    def get_logits_r2l(self, net_output):
        logits = net_output
        return logits

    def get_targets_r2l(self, sample):
        """Get targets from either the sample or the net's output."""
        # exclude bos from target
        return sample['target']