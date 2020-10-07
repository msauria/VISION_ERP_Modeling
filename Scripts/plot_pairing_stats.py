#!/usr/bin/env python3

import sys

import numpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def main():
    in_fname, out_fname = sys.argv[1:3]
    data = load_pairs(in_fname)
    with PdfPages(out_fname) as pdf:
        plot_cre_count_per_tss(pdf, data)
        plot_tss_count_per_cre(pdf, data)
        plot_distances(pdf, data)

def load_pairs(fname):
    data = []
    fs = open(fname)
    fs.readline()
    strand_dict = {'-': True, '+': False}
    chr_dict = {}
    for line in fs:
        line = line.rstrip('\n\r').split()
        chrom, cre_start, cre_stop, tss, strand = line[0], int(line[10]), int(line[11]), int(line[15]), strand_dict[line[18]]
        chr_dict.setdefault(chrom, len(chr_dict))
        data.append((chr_dict[chrom], (cre_start + cre_stop) // 2, tss, strand, 0, 0))
    data = numpy.array(data, dtype=numpy.dtype([('chr', numpy.int32), ('cre', numpy.int32),
                                                ('tss', numpy.int32), ('strand', numpy.bool),
                                                ('cindex', numpy.int32), ('tindex', numpy.int32)]))
    fs.close()
    cindex = data['cre'].astype(numpy.int64) * len(chr_dict) + data['chr']
    uc = numpy.unique(cindex)
    data['cindex'] = numpy.searchsorted(uc, cindex, side='left')
    tindex = data['tss'].astype(numpy.int64) * len(chr_dict) + data['chr']
    ut = numpy.unique(tindex)
    data['tindex'] = numpy.searchsorted(ut, tindex, side='left')
    return data

def plot_cre_count_per_tss(pdf, data):
    tss = numpy.unique(data['tindex'])
    counts = []
    for t in tss:
        where = numpy.where(data['tindex'] == t)[0]
        counts.append(where.shape[0])
    n, bins, patches = plt.hist(counts, 11)
    plt.title('cCREs per TSS')
    plt.xlabel('# cCREs')
    plt.ylabel('TSS count')
    pdf.savefig()
    plt.close()

def plot_tss_count_per_cre(pdf, data):
    cre = numpy.unique(data['cindex'])
    counts = []
    for c in cre:
        where = numpy.where(data['cindex'] == c)[0]
        counts.append(where.shape[0])
    n, bins, patches = plt.hist(counts, 30)
    plt.title('TSSs per cCRE')
    plt.xlabel('# TSSs')
    plt.ylabel('cCRE count')
    pdf.savefig()
    plt.close()

def plot_distances(pdf, data):
    dists = []
    for i in range(data.shape[0]):
        d = data['tss'][i] - data['cre'][i]
        if data['strand'][i]:
            d = -d
        dists.append(d)
    n, bins, patches = plt.hist(dists, 400)
    plt.title('CRE positions relative to TSSs')
    plt.xlabel('Distance (bp)')
    plt.ylabel('CRE-TSS pair count')
    pdf.savefig()
    plt.close()


if __name__ == "__main__":
    main()
