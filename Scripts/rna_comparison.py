#!/usr/bin/env python3

import sys

import numpy
import matplotlib.pyplot


def main():
    in_fname, out_fname = sys.argv[1:3]
    data = numpy.load(in_fname)
    CTs = data.dtype.names[3:]
    n = len(CTs)
    rna = numpy.zeros((data.shape[0], n), dtype=numpy.float32)
    for i, ct in enumerate(CTs):
        rna[:, i] = data[ct]
    means = rna.mean(axis=1)
    for i, ct in enumerate(CTs):
        print(ct, numpy.corrcoef(rna[:, i], means)[0, 1])


if __name__ == "__main__":
    main()
