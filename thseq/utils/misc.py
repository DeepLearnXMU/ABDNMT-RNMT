import importlib
import os
import random

import numpy
import torch
import torch.nn as nn


def recursive_import(dir, module_prefix=''):
    # automatically import any Python files in the models/ directory

    for file in os.listdir(dir):
        fullname = os.path.join(dir, file)
        if file.endswith('.py') and not file.startswith('_'):
            name = fullname[:fullname.find('.py')]
            relative_dir = module_prefix.replace('.', os.sep)
            module = name[name.rfind(relative_dir):].replace(os.sep, '.')
            module = '{}'.format(module)
            importlib.import_module(module)
        if os.path.isdir(fullname) and not file.startswith('_'):
            recursive_import(fullname, "{}.{}".format(module_prefix, file))


def try_get_attr(instance, attr_name):
    if hasattr(instance, attr_name):
        return getattr(instance, attr_name)
    else:
        return None


def set_seed(seed):
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def get_value_or_default(args, attr, value=None):
    val = getattr(args, attr, value)
    if val is None:
        val = value
    return val


def get_state_dict(obj, exclusions=None, recursive=True):
    keys = vars(obj).keys()
    state_dict = {}

    for key in keys:
        if not exclusions or key not in exclusions:
            prop = getattr(obj, key)
            if recursive:
                to_state_dict = try_get_attr(prop, 'state_dict')
                if callable(to_state_dict):
                    prop = to_state_dict()
            state_dict[key] = prop

    return state_dict


def load_state_dict(obj, state_dict):
    for key, val in state_dict.items():
        prop = getattr(obj, key)
        load_state_dict = try_get_attr(prop, 'load_state_dict')
        if callable(load_state_dict):
            load_state_dict(val)
        else:
            setattr(obj, key, val)


def load_pretrain(module: nn.Module, state_dict, add_prefix=''):
    loaded = []
    not_loaded = []
    for n, p in module.named_parameters(prefix=add_prefix):
        p_ = state_dict.get(n, None)
        if add_prefix:
            n = n[len(add_prefix) + 1:]
        if p_ is not None:
            p.data = p_
            loaded.append(n)
        else:
            not_loaded.append(n)
    return loaded, not_loaded


def stat_cuda(msg):
    print('-------', msg)
    print(torch.cuda.memory_allocated() / 1024 / 1024, torch.cuda.max_memory_allocated() / 1024 / 1024)
    print(torch.cuda.memory_cached() / 1024 / 1024, torch.cuda.max_memory_cached() / 1024 / 1024)


def aggregate_value_by_key(dicts, key, reduction=None):
    values = [d[key] for d in dicts]
    if reduction:
        return reduction(values)
    return values
