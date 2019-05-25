import argparse
import io
import itertools
import multiprocessing
import os
import subprocess
import sys
import threading
import time
import traceback
from collections import OrderedDict, deque

CMD = 'python translate.py --model {model} {extra}'


def count_lines(f):
    i = 0
    if not os.path.exists(f):
        return i
    with open(f) as r:
        for _ in r:
            i += 1
    return i


def get_proc(model, device, extra_args, silent):
    env = dict(os.environ)  # Make a copy of the current environment

    if isinstance(extra_args, (list, tuple)):
        extra_args = " ".join(extra_args)

    env['CUDA_VISIBLE_DEVICES'] = f'{device}'

    cmd = CMD.format(
        model=model,
        extra=extra_args)
    if not silent:
        sys.stderr.write("translate cmd: {}\n".format(cmd))

    p = subprocess.Popen(cmd.split(),
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=open(os.devnull, 'w'),
                         env=env)
    return p


def worker(p, queue, rqueue):
    stdin = io.TextIOWrapper(p.stdin, encoding='utf-8')
    stdout = io.TextIOWrapper(p.stdout, encoding='utf-8')
    indices = []
    while True:
        i, line = queue.get()

        if i < 0:  # end of queue
            stdin.close()
            break
        stdin.write(f'{line.strip()}\n')
        indices.append(i)
    for i in indices:
        out = stdout.readline()
        rqueue.put((i, out))


def translate(model, pending, done, src2out, devices, entry, extra, silent, write_stdout):
    n_pending = list(map(count_lines, pending))
    n_done = sum(map(count_lines, done)) if done else 0
    n_total = sum(n_pending) + n_done

    if sum(n_pending) == 0:
        return
    out_dir = os.path.dirname(next(iter(src2out.values())))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if pending:
        tic = time.time()

        fds = [open(f, encoding='utf-8') for f in pending]
        reader = itertools.chain(*fds)

        queue = multiprocessing.Queue()
        rqueue = multiprocessing.Queue()
        # feed
        for i, line in enumerate(reader):
            queue.put((i, line))

        for _ in devices:
            queue.put((-1, None))  # end of queue

        processes = [get_proc(model, device, extra, silent) for device in devices]
        threads = [threading.Thread(target=worker, args=(p, queue, rqueue)) for p in processes]

        try:
            for t in threads:
                t.daemon = True
                t.start()

            hist = deque(maxlen=5)  # average over 5 past records
            # consume to prevent holding all translations in memory
            buffer = []
            i = 0
            j = 0
            writer = None
            tic_speed = time.time()
            while i < len(pending):
                time.sleep(0.5)
                for p in processes:
                    p.poll()
                    if p.returncode is not None and p.returncode > 0:
                        sys.stderr.write("Error occurs in worker")
                        raise RuntimeError()

                added = False
                while not rqueue.empty():
                    buffer.append(rqueue.get())
                    added = True

                if added:
                    buffer = sorted(buffer, key=lambda x: x[0])

                while buffer and buffer[0][0] == sum(n_pending[:i]) + j:
                    idx, trans = buffer[0]
                    if writer is None:
                        if write_stdout:
                            writer = sys.stdout
                        else:
                            writer = open(src2out[pending[i]], 'w')
                    writer.write(trans)
                    j += 1
                    if not j < n_pending[i]:
                        if not write_stdout and writer is not None:
                            writer.close()
                            writer = None
                        i += 1
                        j = 0
                    buffer = buffer[1:]  # remove processed output

                if not silent:
                    n1 = n_done + sum(n_pending[:i]) + j + rqueue.qsize() + len(buffer)
                    hist.append(n1)
                    rate = 0.0
                    if len(hist) > 1:
                        rate = (hist[-1] - hist[0] + 0.0) / len(hist) / (time.time() - tic_speed)
                    tic_speed = time.time()
                    duration = time.strftime('%H:%M:%S', time.gmtime(time.time() - tic))
                    sys.stderr.write(f'\r{n1}/{n_total}, {rate:.2f} s/sec, {duration}')

            if not silent:
                sys.stderr.write('\n')
        except Exception as e:
            traceback.print_exc()
            for p in processes:
                p.terminate()
            sys.exit(1)


def main(args):
    model = args.model
    entry = args.entry
    inputs = args.inputs
    tag = args.tag
    cuda = args.cuda
    silent = args.silent
    remains = args.remains
    write_stdout = args.stdout
    force = args.force

    extra = " ".join(remains)

    if cuda:
        devices = cuda
    else:
        devices = ['cpu']

    model_name = model.strip(os.path.sep).replace(os.path.sep, '.')

    # skip translated
    src2out = OrderedDict()
    for name in inputs:
        output = os.path.join('translations',
                              f'{tag + "-" if tag else ""}{model_name}-{os.path.basename(name)}')
        src2out[name] = output

    if args.list_outputs:
        for output in src2out.values():
            print(output)
        sys.exit(0)

    pending = []
    done = []
    if force:
        pending = inputs
    else:
        for s, o in src2out.items():
            if os.path.exists(o) and count_lines(s) == count_lines(o):
                # skip translated
                done.append(s)
            else:
                pending.append(s)

    for f in done:
        sys.stderr.write('skip {}\n'.format(f))

    translate(model, pending, done, src2out, devices, entry, extra, silent, write_stdout)


def valid_file(parser, arg):
    if arg and not os.path.exists(arg):
        parser.error('The file doesn\'t exist: {}'.format(arg))
    else:
        return arg


def parse_args():
    parser = argparse.ArgumentParser()
    file_type = lambda arg: valid_file(parser, arg)

    parser.add_argument('model')
    parser.add_argument('--entry', '-e', default='translate.py')
    parser.add_argument('--inputs', '-i', type=file_type, nargs='+')
    parser.add_argument('--list-outputs', action='store_true',
                        help='list output names in correspondence with given model and input files, then exit')
    parser.add_argument('--tag', '-t', type=str)
    parser.add_argument('--cuda', nargs='+', type=str,
                        help='e.g. --cuda 0 0 2 or --cuda cpu')
    parser.add_argument('--silent', '-s', action="store_true", help='suppress procedural printings')
    parser.add_argument('--stdout', action="store_true",
                        help='write to stdout instead of files, if True, suppress all irrelevant stdout')
    parser.add_argument('--force', action='store_true', help='force overwrite existing translations')

    args, remains = parser.parse_known_args()
    args.remains = remains

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
