# characterize.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import argparse
import codecs
import os


def process(fin, eow=None):
    fd = codecs.open(fin, "r", "utf-8")
    fw = codecs.open('char.%s' % os.path.basename(fin), "w", "utf-8")

    for line in fd:
        line = line.strip()
        wlist = line.split()
        clist = []

        for word in wlist:
            for char in word:
                clist.append(char)
            if eow != None:
                clist.append(eow)

        fw.write(" ".join(clist) + "\n")

    fd.close()
    fw.close()


def parseargs():
    msg = "characterize corpus"
    parser = argparse.ArgumentParser(description=msg)

    msg = "corpus"
    parser.add_argument("corpus", type=str, help=msg)
    msg = "add word seperation token"
    parser.add_argument("--eow", type=str, help=msg)

    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    process(args.corpus, args.eow)
