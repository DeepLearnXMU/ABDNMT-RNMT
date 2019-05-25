# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from . import FairseqCriterion, register_criterion


import torch.nn as nn

class SmoothedCE(nn.Module):

    def __init__(self,model,padding,eps):
        super().__init__()
        self.model=model
        self.padding_idx=padding
        self.eps=eps

    def forward(self, sample):
        reduction = 'sum'
        net_output = self.model(*sample['net_input'])
        lprobs = self.model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = self.model.get_targets(sample).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduction == 'sum':
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss,nll_loss


@register_criterion('abd-smoothed-ce')
class ABDLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(self, args, target_vocabulary):
        super().__init__(args, target_vocabulary)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0.1, type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--coeff',nargs=2,default=[1,1])

    def forward(self, model, sample, reduction='sum'):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # net_output = model(*sample['net_input'])
        if not hasattr(self,'_ce'):
            self._ce=SmoothedCE(model,self.padding_idx,self.eps).cuda()
        loss,nll_loss=nn.parallel.data_parallel(self._ce,sample)
        loss=loss.sum()
        nll_loss=nll_loss.sum()
        sample_size = sample['target'].size(0) if self.args.batch_by_sentence else sample['ntokens']
        logging_output = {
            'loss': loss.data.item() if reduction != 'none' else loss.data,
            'nll_loss': nll_loss.data.item() if reduction != 'none' else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        nll_loss = sum(log.get('nll_loss', 0) for log in logging_outputs)
        return {
            # 'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            # 'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'loss': 0 if sample_size == 0 else sum(log.get('loss', 0) for log in logging_outputs) / sample_size,
            'nll_loss': nll_loss,
            'per_word_loss': 0 if ntokens == 0 else nll_loss / ntokens,
            'sample_size': sample_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
        }
