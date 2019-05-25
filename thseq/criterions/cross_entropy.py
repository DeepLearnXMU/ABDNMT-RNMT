# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch.nn.functional as F

from thseq.models.abs import Seq2SeqBase
from . import FairseqCriterion, register_criterion


@register_criterion('ce')
class CrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, target_vocabulary):
        super().__init__(args, target_vocabulary)

    def forward(self, model: Seq2SeqBase, sample, reduction='sum'):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        x = model(*sample['net_input'])
        x = model.get_logits(x)
        x = x.view(-1, x.size(-1))
        target = model.get_targets(sample).view(-1)
        loss = F.cross_entropy(x, target, ignore_index=self.padding_idx, reduction='sum')
        sample_size = sample['target'].size(0) if self.args.batch_by_sentence else sample['ntokens']
        # Use tensor.data to detach variable from computation graph.
        # Otherwise, the computation graph history cannot be released
        # thus increases the memory usage.
        logging_output = {
            'loss': loss.data.item() if reduction != 'none' else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            # 'loss': loss_sum / sample_size / math.log(2),
            'loss': 0 if sample_size==0 else loss_sum / sample_size,
            'per_word_loss':0 if ntokens == 0 else loss_sum / ntokens,
            'sample_size': sample_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
        }
        return agg_output
