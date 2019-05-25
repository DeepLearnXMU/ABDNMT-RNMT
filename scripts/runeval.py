import argparse
import fractions
import math
import os
import subprocess
import sys
from collections import Counter
from collections import OrderedDict

import pandas
from nltk.translate.bleu_score import SmoothingFunction, modified_precision, closest_ref_length, brevity_penalty


import pandas
from nltk.translate.bleu_score import SmoothingFunction, modified_precision, closest_ref_length, brevity_penalty
import fractions

try:
    fractions.Fraction(0, 1000, _normalize=False)
    from fractions import Fraction
except TypeError:
    from nltk.compat import Fraction


def modified_corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=None, auto_reweigh=False):
    """
    modified from nltk.translate.bleu_score.corpus_bleu,
    returns 'multi-bleu.perl'-like intermediate results.
    Args:
        list_of_references:
        hypotheses:
        weights:
        smoothing_function:
        auto_reweigh:

    Returns:

    """
    # Before proceeding to compute BLEU, perform sanity checks.

    p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.
    hyp_lengths, ref_lengths = 0, 0

    assert len(list_of_references) == len(hypotheses), f"The number of hypotheses and their reference(s) should be " \
                                                       f"the same: {len(list_of_references)} != {len(hypotheses)}"

    # Iterate through each hypothesis and their corresponding references.
    for references, hypothesis in zip(list_of_references, hypotheses):
        # For each order of ngram, calculate the numerator and
        # denominator for the corpus-level modified precision.
        for i, _ in enumerate(weights, start=1):
            p_i = modified_precision(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator

        # Calculate the hypothesis length and the closest reference length.
        # Adds them to the corpus-level hypothesis and reference counts.
        hyp_len = len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)

    # Calculate corpus-level brevity penalty.
    bp = brevity_penalty(ref_lengths, hyp_lengths)

    # Uniformly re-weighting based on maximum hypothesis lengths if largest
    # order of n-grams < 4 and weights is set at default.
    if auto_reweigh:
        if hyp_lengths < 4 and weights == (0.25, 0.25, 0.25, 0.25):
            weights = (1 / hyp_lengths,) * hyp_lengths

    # Collects the various precision values for the different ngram orders.
    p_n = [Fraction(p_numerators[i], p_denominators[i], _normalize=False)
           for i, _ in enumerate(weights, start=1)]

    # Returns 0 if there's no matching n-grams
    # We only need to check for p_numerators[1] == 0, since if there's
    # no unigrams, there won't be any higher order ngrams.
    if p_numerators[1] == 0:
        return 0

    # If there's no smoothing, set use method0 from SmoothinFunction class.
    if not smoothing_function:
        smoothing_function = SmoothingFunction().method0
    # Smoothen the modified precision.
    # Note: smoothing_function() may convert values into floats;
    #       it tries to retain the Fraction object as much as the
    #       smoothing method allows.
    p_n = smoothing_function(p_n, references=references, hypothesis=hypothesis,
                             hyp_len=hyp_len)
    s = (w_i * math.log(p_i) for w_i, p_i in zip(weights, p_n))
    s = bp * math.exp(math.fsum(s))
    return s, p_n, bp, hyp_lengths / ref_lengths, hyp_lengths, ref_lengths


def corpus_bleu(references, hypothesis):
    hyp = []
    with open(hypothesis) as r:
        for l in r:
            hyp.append(l.split())

    ref = []
    for file in references:
        ref.append([])
        with open(file) as r:
            for l in r:
                ref[-1].append(l.split())

    lens = list(map(len, ref))
    if len(set(lens)) > 1:
        raise RuntimeError(f'The sizes multi-reference are inconsistent: {lens}')

    ref = list(zip(*ref))
    if len(hyp) != lens[0]:
        raise RuntimeError(f'The sizes of reference and hypothesis are inconsistent: {len(hyp)}!={lens[0]}')
    return modified_corpus_bleu(ref, hyp)


def error(p, msg, err=None):
    if p.returncode > 0:
        if err is not None:
            sys.stderr.write(err)
        raise RuntimeError(msg)


def main(args):
    models = args.models
    entry = args.entry
    inputs = args.input
    refs = args.ref
    cuda = args.cuda

    translate_script = args.translate_script

    extra = ''
    if remains:
        extra = ' '.join(remains)

    results = []
    for model in models:
        print(f'------------------- {model} -------------------')
        # translate
        cmd = f'python {translate_script} {model} --entry {entry} --inputs {" ".join(inputs)} ' \
            f'--cuda {" ".join(cuda)} {extra}'
        print(f'translate script: {cmd}')
        p = subprocess.Popen(cmd.split())
        p.wait()
        error(p, cmd)
        # get output names
        cmd = f'python {translate_script} {model} --entry {entry} --inputs {" ".join(inputs)} --list-outputs'
        p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        out = out.decode('utf-8')
        error(p, cmd, err)
        hyps = [hyp for hyp in out.split(os.linesep) if hyp]
        print('hypotheses:')
        for f in hyps:
            print(f'\t{f}')
        # post-process
        if args.post_process_scripts:
            processed = [hyp + '.post-process' for hyp in hyps]
            for f1, f2 in zip(hyps, processed):
                temp_out = f2 + '.tmp'
                with open(f1) as r, open(temp_out, 'w') as w:
                    if len(args.post_process_scripts) > 1:
                        p = subprocess.Popen([args.post_process_scripts[0]], stdin=r, stdout=subprocess.PIPE)
                        ps = [p]
                        for script in args.post_process_scripts[1:-1]:
                            print(script)
                            p_continue = subprocess.Popen([script], stdin=ps[-1].stdout, stdout=subprocess.PIPE)
                            ps.append(p_continue)
                        p_final = subprocess.Popen([args.post_process_scripts[-1]], stdin=ps[-1].stdout, stdout=w)
                        ps.append(p_final)
                        p = ps[-1]
                    else:
                        p = subprocess.Popen([args.post_process_scripts[0]], stdin=r, stdout=w)
                    p.wait()
                    error(p, 'error occurred during post processing')
                os.rename(temp_out, f2)

            hyps = processed
        if refs:
            d = OrderedDict()
            n_ref, remain = divmod(len(refs), len(inputs))
            if remain > 0:
                raise ValueError('inputs and refs does not match')

            for i, (input, hyp) in enumerate(zip(inputs, hyps)):
                i_ref = refs[i * n_ref:(i + 1) * n_ref]

                bleu, p_n, bp, ratio, hyp_len, ref_len = corpus_bleu(i_ref,hyp)
                bleu *= 100
                name = os.path.basename(input)
                sys.stdout.write(f'-- {name}\n')
                precision = "/".join(f"{float(p) * 100:.2f}" for p in p_n)
                sys.stdout.write(f'\tBLEU = {bleu:.2f},'
                                 f' {precision},'
                                 f' (BP={bp:.3f}, ratio={ratio:.3f},'
                                 f' hyp_len={hyp_len}, ref_len={ref_len})\n')
                d[name] = bleu
            sys.stdout.write(f'Avg = {sum(d.values()) / len(d)}')
            results.append(d)

    if args.csv and results:
        name = args.csv
        if not name.endswith('.csv'):
            name = '%s.csv' % name
        df = pandas.DataFrame(results, index=models)
        df.to_csv(name)
        sys.stderr.write('dumped to %s' % name)


def valid_file(parser, arg):
    arg = os.path.expanduser(arg)
    if arg and not os.path.exists(os.path.expanduser(arg)):
        parser.error('The file doesn\'t exist: {}'.format(arg))
    else:
        return arg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    file_type = lambda arg: valid_file(parser, arg)

    parser.add_argument('models', nargs='+', type=str)
    parser.add_argument('--entry', type=file_type, default='translate.py',
                        help='path to model entry, default to translate.py')
    parser.add_argument('--input', '-i', nargs='+', required=True, type=str)
    parser.add_argument('--ref', '-r', nargs='+', type=str)
    parser.add_argument('--cuda', nargs='+', default='0', type=str)

    parser.add_argument('--post-process-scripts', type=str, nargs='+',
                        help='executable scripts that take as input the stdin')
    parser.add_argument('--csv', type=str)
    parser.add_argument('--translate-script', type=file_type, default='scripts/translate.py')

    args, remains = parser.parse_known_args()
    args.remains = remains

    if args.post_process_scripts:
        for path in args.post_process_scripts:
            if not os.path.exists(path):
                raise ValueError(f'File doesn\'t exist: {path}')
            elif not os.access(path, os.X_OK):
                raise ValueError(f'File is not executable: {path}')
    if not os.path.exists(args.translate_script):
        raise FileNotFoundError(f'File doesn\'t exist: {args.translate_script}')

    main(args)
