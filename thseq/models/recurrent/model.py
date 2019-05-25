from thseq.models import Seq2SeqBase
from thseq.models import register_model
from thseq.models.recurrent.bridge import Bridge
from thseq.models.recurrent.decoder import Decoder
from thseq.models.recurrent.encoder import Encoder


@register_model('rnn')
class Seq2Seq(Seq2SeqBase):
    def __init__(self, args, vocabularies):
        encoder = Encoder(args, vocabularies[0])
        decoder = Decoder(args, vocabularies[1])
        bridge = Bridge(args)
        super().__init__(args, encoder, bridge, decoder)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--input-size', type=int, default=500)
        parser.add_argument('--hidden-size', type=int, default=1000)
        parser.add_argument('--attention-size', type=int, default=1000)
        parser.add_argument('--dropout-input', type=float, default=0.3)
        parser.add_argument('--dropout-hidden', type=float, default=0.3)
        parser.add_argument('--weight-tying', type=int, default=1)

    @classmethod
    def build(cls, args, vocabularies):
        return cls(args, vocabularies)
