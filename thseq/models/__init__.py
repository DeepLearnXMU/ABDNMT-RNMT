# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os

from thseq.models.abs import BridgeBase
from thseq.models.abs import DecoderBase
from thseq.models.abs import EncoderBase
from thseq.models.abs import Seq2SeqBase
from thseq.utils.misc import recursive_import

MODEL_REGISTRY = {}
ARCH_MODEL_REGISTRY = {}
ARCH_CONFIG_REGISTRY = {}


def build_model(args, vocabularies):
    return MODEL_REGISTRY[args.arch].build(args, vocabularies)


def register_model(name):
    """Decorator to register a new model (e.g., LSTM)."""

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        if not issubclass(cls, Seq2SeqBase):
            raise ValueError('Model ({}: {}) must extend BaseFairseqModel'.format(name, cls.__name__))
        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls


def register_model_architecture(model_name, arch_name):
    """Decorator to register a new model architecture (e.g., lstm_luong_wmt_en_de)."""

    def register_model_arch_fn(fn):
        if model_name not in MODEL_REGISTRY:
            raise ValueError('Cannot register model architecture for unknown model type ({})'.format(model_name))
        if arch_name in ARCH_MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model architecture ({})'.format(arch_name))
        if not callable(fn):
            raise ValueError('Model architecture must be callable ({})'.format(arch_name))
        ARCH_MODEL_REGISTRY[arch_name] = MODEL_REGISTRY[model_name]
        ARCH_CONFIG_REGISTRY[arch_name] = fn
        return fn

    return register_model_arch_fn


recursive_import(os.path.dirname(__file__), 'thseq.models')
