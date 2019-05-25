# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter(object):
    """Computes the average occurrence of some event per second"""

    def __init__(self, init=0):
        self.reset(init)

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val
        self.val = val

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)


class SpeedMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.duration = 0
        self.start_time = time.time()

    def start(self):
        self.start_time = time.time()

    def stop(self, n):
        self.sum += n
        self.duration += time.time() - self.start_time

    @property
    def avg(self):
        return self.sum / self.duration


class TimeWindowMeter(object):
    def __init__(self, window=1):
        self.window = window
        self.buffer = []
        self.start_times = []

    def reset(self):
        self.buffer.clear()
        self.start_times.clear()
        self.start_times.append(time.time())

    def update(self, val):
        self.buffer.append(val)
        self.buffer = self.buffer[-self.window:]
        self.start_times = self.start_times[-self.window:]
        self.start_times.append(time.time())

    @property
    def avg(self):
        if not self.start_times:
            return 0
        return sum(self.buffer) / (time.time() - self.start_times[0])


class StopwatchMeter(object):
    """Computes the sum/avg duration of some event in seconds"""

    def __init__(self, state_less=False):
        self.state_less = state_less
        self.reset()

    def start(self):
        self.start_time = time.time()

    def stop(self, n=1):
        if self.start_time is not None:
            delta = time.time() - self.start_time
            if self.state_less:
                self.sum = delta
            else:
                self.sum += delta
                self.n += n

            self.start_time = None

    def reset(self):
        self.sum = 0
        self.n = 0
        self.start_time = None

    @property
    def avg(self):
        return self.sum / self.n


class ElapsedTimeMeter(object):
    """Records elapsed time in seconds"""

    def __init__(self, init=0):
        super().__init__()
        self.sum = init
        self.start_time = time.time()

    def restart(self):
        self.start_time = time.time()

    @property
    def total(self):
        return self.sum + (time.time() - self.start_time)

    @property
    def latest(self):
        return time.time() - self.start_time

    def accumulate(self):
        delta = time.time() - self.start_time
        self.start_time = time.time()
        self.sum += delta
