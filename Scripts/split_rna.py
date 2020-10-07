#!/usr/bin/env python

import sys
import argparse

import matplotlib.pyplot as plt
import numpy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def main():
    parser = generate_parser()
    args = parser.parse_args()
    f = numpy.load(args.RNA)
    CTs = []
    CT_indices = {}
    names = f.dtype.names
    for i, name in enumerate(names[4:]):
        ct = name.split('_')[0]
        if ct not in CT_indices:
            CTs.append(ct)
            CT_indices[ct] = [i]
        else:
            CT_indices[ct].append(i)
    rna = numpy.zeros((f.shape[0], len(CTs)), dtype=numpy.float32)
    for i, name in enumerate(CTs):
        for j in CT_indices[name]:
            rna[:, i] += f[names[j + 4]]
        rna[:, i] /= len(CT_indices[name])
    std = numpy.std(rna, axis=1)
    std[numpy.where(std == 0)] = numpy.amin(std[numpy.where(std > 0)]) * 0.5
    Y, X = numpy.histogram(std, bins=40)
    X = (X[1:] + X[:-1]) / 2
    Y = Y.astype(numpy.float32)
    troughs = numpy.where((Y[1:-1] < Y[2:]) & (Y[1:-1] < Y[:-2]))[0] + 1
    depths = Y[troughs - 1] + Y[troughs + 1] - 2 * Y[troughs]
    minX = troughs[numpy.where(depths == numpy.amax(depths))[0][0]]
    poly_reg = PolynomialFeatures(degree=2)
    X_poly = poly_reg.fit_transform(X[(minX - 1):(minX + 2)].reshape(-1, 1))
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, Y[(minX - 1):(minX + 2)])
    a, b, c = pol_reg.coef_[:3]
    split = -b / (2 * c)
    diff = numpy.where(std >= split)[0]
    nodiff = numpy.where(std < split)[0]
    numpy.save("%s_diff.npy" % args.OUTPUT, f[diff])
    numpy.save("%s_nodiff.npy" % args.OUTPUT, f[nodiff])
    print('Diff:%i  NonDiff:%i' % (diff.shape[0], nodiff.shape[0]))
    if args.PLOT is not None:
        plot_dist(X, Y, split, args.PLOT)

def plot_dist(X, Y, split, fname):
    fig, ax = plt.subplots(1,1, figsize=(8, 4))
    ax.plot(X, Y)
    ax.axvline(split)
    ax.set_xlabel('RNA standard deviation')
    ax.set_ylabel('Number of genes')
    plt.savefig(fname)

def gamma_pdf(X, alpha, beta):
    return (beta ** alpha) * (X ** (alpha - 1)) * numpy.exp(-X * beta) / scipy.special.gamma(alpha)

def generate_parser():
    """Generate an argument parser."""
    description = "%(prog)s -- Convert an RNA TPM text file to a numpy NPY file"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-r', '--rna', dest="RNA", type=str, action='store', required=True,
                        help="RNA expression file")
    parser.add_argument('-p', '--plot', dest="PLOT", type=str, action='store',
                        help="File to plot distribution and split to")
    parser.add_argument('-o', '--output', dest="OUTPUT", type=str, action='store', required=True,
                        help="Numpy NPY RNA expression file prefix to write to")
    return parser

if __name__ == "__main__":
    main()
