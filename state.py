import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import List

import torch
from logbook import Logger
import itertools
import thseq.utils as utils
from thseq.utils.meters import ElapsedTimeMeter
from thseq.utils.misc import load_state_dict, get_state_dict

logger = Logger()

META_CHECKPOINT_PATH = 'ckps.json'


@dataclass
class Checkpoint(object):
    filename: str
    num_step: int
    epoch: int
    num_step_in_epoch: int
    score: float = None


class Loader(object):
    @staticmethod
    def get_ckps(dirname):
        with open(os.path.join(dirname, META_CHECKPOINT_PATH)) as r:
            ckps = [Checkpoint(**attrs) for attrs in json.loads(r.read())]
        return ckps

    @staticmethod
    def get_path(dirname, ckp):
        return os.path.join(dirname, ckp.filename)

    @staticmethod
    def get_paths(dirname, ckps):
        paths = [os.path.join(dirname, c.filename) for c in ckps]
        return paths

    @staticmethod
    def load_state(path, map_location=None):
        return torch.load(path, map_location)

    @staticmethod
    def load_states(paths, map_location=None):
        return [Loader.load_state(path, map_location) for path in paths]

    @staticmethod
    def get_best(dirname):
        return Loader.get_bests(dirname, 1)[0]

    @staticmethod
    def get_bests(dirname, n=None):
        ckps = Loader.filter_best(Loader.get_ckps(dirname))
        ckps.sort(key=lambda c: -c.score)
        return list(zip(ckps[:n], Loader.get_paths(dirname, ckps[:n])))

    @staticmethod
    def get_latest(dirname):
        return Loader.get_latests(dirname, 1)[0]

    @staticmethod
    def get_latests(dirname, n=None):
        ckps = Loader.get_ckps(dirname)
        ckps.sort(key=lambda c: c.num_step)
        return list(zip(ckps[-n:], Loader.get_paths(dirname, ckps[-n:])))

    @staticmethod
    def get_best_window(dirname, width=1):
        """Get surrounding evaluation checkpoints of the best checkpoint.
        """
        ckps = Loader.filter_best(Loader.get_ckps(dirname))
        best = max(ckps, key=lambda c: c.score)
        i = ckps.index(best)
        ckps = ckps[:i][-width:] + ckps[i:][:width + 1]
        return list(zip(ckps, Loader.get_paths(dirname, ckps)))

    @staticmethod
    def get_latest_window_around_best(dirname, width=1):
        """Get surrounding  checkpoints of the best checkpoint.
        """

        ckps = Loader.get_ckps(dirname)
        best = max(ckps, key=lambda c: c.score or 0)
        i = ckps.index(best)
        ckps = ckps[:i] + ckps[i + 1:]
        ckps = Loader.filter_latest(ckps)
        ckps.append(best)
        ckps.sort(key=lambda c: c.num_step)
        i = ckps.index(best)

        ckps = ckps[:i][-width:] + ckps[i:][:width + 1]
        return list(zip(ckps, Loader.get_paths(dirname, ckps)))

    @staticmethod
    def filter_best(ckps):
        return list(filter(lambda c: c.score is not None, ckps))

    @staticmethod
    def filter_latest(ckps):
        return list(filter(lambda c: c.score is None, ckps))


class State(object):
    def __init__(self, save_checkpoints_secs, save_checkpoints_steps, keep_checkpoint_max, keep_best_checkpoint_max,
                 args, model, criterion, optimizer, lr_scheduler, iterator, **kwargs):
        if keep_checkpoint_max is None or keep_checkpoint_max <= 0:
            keep_checkpoint_max = sys.maxsize
        if keep_best_checkpoint_max is None or keep_best_checkpoint_max <= 0:
            keep_best_checkpoint_max = sys.maxsize
        self.keep_best_checkpoint_max = keep_best_checkpoint_max

        self.save_checkpoints_secs = save_checkpoints_secs
        self.save_checkpoints_steps = save_checkpoints_steps
        self.keep_checkpoint_max = keep_checkpoint_max
        self._best_time = None
        self._best_name = None  # best model path
        self._epoch = 0
        self._step = 0
        self._step_in_epoch = 0

        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.iterator = iterator
        for k, v in kwargs.items():
            if hasattr(self, k):
                raise ValueError('Found conflicted keys {}.'.format(k))
            setattr(self, k, v)

        self.timer = ElapsedTimeMeter()

        self.ckps: List[Checkpoint] = []
        self.best_ckps: List[Checkpoint] = []
        self.eval_scores = []
        self._meta_ckp_path = os.path.join(args.model, META_CHECKPOINT_PATH)

    @property
    def epoch(self):
        return self._epoch

    @property
    def step_in_epoch(self):
        return self._step_in_epoch

    @property
    def step(self):
        return self._step

    @property
    def elapsed_time(self):
        return self.timer.total

    def get_savename(self):
        timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
        return f'ckpt.{self.step:08d}.{timestamp}.pt'

    def increase_epoch(self):
        self._epoch += 1
        self._step_in_epoch = 0

    def increase_num_steps(self):
        self._step += 1
        self._step_in_epoch += 1

    def get_best_time(self):
        return self._best_time

    def state_dict(self):
        state = get_state_dict(self)

        if getattr(self.args, 'distributed',None):
            i = len('module.')
            model_state = state['model'].__class__()
            for k, v in state['model'].items():
                model_state[k[i:]] = v
            state['model'] = model_state

        return state

    def load_state_dict(self, state_dict):
        load_state_dict(self, state_dict)
        # restore lr
        self.lr_scheduler.step(self.epoch)
        self.lr_scheduler.step_update(self.step)

        self.timer.restart()

    def remove_ckps(self, ckps:List[Checkpoint]):
        for ckp in ckps:
            path = os.path.join(self.args.model, ckp.filename)
            if os.path.exists(path):
                os.remove(path)

    def try_save(self):
        # if not utils.distributed.is_master(self.args):
        #     return
        # Do not save a checkpoint twice.

        if any(self.step == c.num_step for c in self.ckps + self.best_ckps):
            return

        tic = time.time()
        savename = self.get_savename()
        keep_checkpoint_max = self.keep_checkpoint_max
        ckps = self.ckps
        save_checkpoints_secs = self.save_checkpoints_secs
        save_checkpoints_steps = self.save_checkpoints_steps
        num_steps = self.step

        elapse = self.timer.latest
        save = False
        if save_checkpoints_steps is not None and save_checkpoints_steps > 0:
            if num_steps % save_checkpoints_steps == 0:
                save = True
        elif save_checkpoints_secs is not None and save_checkpoints_secs > 0:
            if elapse >= save_checkpoints_secs:
                save = True
        else:
            raise RuntimeError()

        if save:
            ckp = Checkpoint(savename,
                             self.step,
                             self.epoch,
                             self.step_in_epoch)

            ckps.append(ckp)
            self.timer.accumulate()
            discard = ckps[:-keep_checkpoint_max]
            self.ckps = ckps[-keep_checkpoint_max:]

            self.remove_ckps(discard)

            self._persist(savename)
            toc = time.time() - tic

            logger.info(f'Save checkpoint: {savename}, took {timedelta(seconds=toc // 1)}')

    def add_valid_score(self, val_score, savename=None):
        # if not utils.distributed.is_master(self.args):
        #     return
        tic = time.time()
        savename = savename or self.get_savename()
        ckp = Checkpoint(savename,
                         self.step,
                         self.epoch,
                         self.step_in_epoch,
                         val_score)
        self.eval_scores.append(val_score)
        best_ckps = self.best_ckps
        if not best_ckps or val_score > best_ckps[0].score:
            self._best_time = (self.epoch + 1, self.step_in_epoch, self.step)
            self._best_name = savename

        if len(best_ckps) < self.keep_best_checkpoint_max or val_score > best_ckps[0].score:
            best_ckps.append(ckp)

            best_ckps.sort(key=lambda r: r.score)

            discard = best_ckps[:-self.keep_best_checkpoint_max]
            self.best_ckps = best_ckps[-self.keep_best_checkpoint_max:]

            self.remove_ckps(discard)

            self._persist(savename)
            toc = time.time() - tic

            logger.info(f'Save evaluation checkpoint: {savename}, '
                        f'took {timedelta(seconds=toc // 1)}')

    def _persist(self, filename):
        # 1. save checkpoint
        target = os.path.join(self.args.model, filename)
        fd, name = tempfile.mkstemp(dir=self.args.model)
        torch.save(self.state_dict(), name)
        os.close(fd)
        shutil.move(name, target)

        # 2. save meta ckp statistics
        discard=[]
        for step,group in itertools.groupby(self.ckps+self.best_ckps,lambda c:c.num_step):
            group=list(group)
            if len(group)>1:
                for item in group:
                    if item.score is None:
                        discard.append(item)
        for item in discard:
            self.ckps.remove(item)

        self.remove_ckps(discard)

        ckps = self.ckps + self.best_ckps
        ckps.sort(key=lambda c: c.num_step)

        with open(self._meta_ckp_path, 'w') as w:
            w.write(json.dumps([c.__dict__ for c in ckps],
                               indent=4,
                               sort_keys=True))
