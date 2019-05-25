# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import data_parallel



class FairseqCriterion(_Loss):
    def __init__(self, args, target_vocabulary):
        super().__init__()
        self.args = args
        self.padding_idx = target_vocabulary.pad_id


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        pass

    def forward(self, model, sample, reduction='sum'):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        raise NotImplementedError

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError

    @staticmethod
    def grad_denom(sample_sizes):
        """Compute the gradient denominator for a set of sample sizes."""
        return sum(sample_sizes)
