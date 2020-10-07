#!/usr/bin/env python3

import sys

import numpy
import twobitreader


def main():
    fa_fname, int_fname, out_fname = sys.argv[1:4]
    reader = twobitreader.TwoBitFile(fa_fname)
    cres = load_cres(int_fname)
    output = open(out_fname, 'w')
    for chrom in reader.keys():
        seq = reader[chrom]
        if chrom not in cres:
            continue
        for i in range(cres[chrom].shape[0]):
            print(">{}:{}-{}".format(chrom, cres[chrom][i, 0], cres[chrom][i, 1]),
                  file=output)
            temp = seq[cres[chrom][i, 0]:cres[chrom][i, 1]]
            pos = 0
            while pos < len(temp):
                print(temp[pos:min(len(temp), pos + 80)], file=output)
                pos += 80
    output.close()

def load_cres(fname):
    cres = {}
    for line in open(fname):
        if line.startswith('Chr'):
            continue
        chrom, tss, strand, start, end = line.split()[:5]
        cres.setdefault(chrom, [])
        cres[chrom].append((int(start), int(end)))
    for chrom in cres:
        cres[chrom].sort()
        cre = [cres[chrom][0]]
        for i in range(1, len(cres[chrom])):
            if cres[chrom][i][0] != cre[-1][0]:
                cre.append(cres[chrom][i])
        cres[chrom] = numpy.array(cre, numpy.int32)
    return cres


if __name__ == "__main__":
    main()
