#!/usr/bin/env python

import sys
import os

import numpy
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main():
    species, out_fname = sys.argv[1:3]
    with PdfPages(out_fname) as pdf:
        plot(species, pdf)

def plot(species, pdf):
    if species == 'hg38':
        CTs = ["B", "CD4", "CD8", "CLP", "CMP", "EOS", "ERY",
               "GMP", "MK", "MONc", "MONp", "MPP", "NEU", "NK"]
    else:
        CTs = ["CFU", "CFUMK", "CMP", "ER4", "ERY-fl", "G1E", "GMP",
               "LSK", "MEP", "MK-imm-ad", "MONO", "NEU"]
    conds = ['', '_control']
    features = ['all', 'nodiff', 'diff']
    for prefix, plabel in [['both', 'CRE + Promoter'], ['cre', 'CRE only'], ['promoter', 'Promoter only']]:
        results = numpy.zeros((len(conds), len(features), len(CTs)), dtype=numpy.float32)
        for j, f in enumerate(features):
            for i, c in enumerate(conds):
                for k, ct in enumerate(CTs):
                    #fname = 'Results/%s_%s_%snopromoter%s_statistics.txt' % (species, ct, f, c)
                    fname = 'Results_%s/%s_%s_%s_%s_0%s_statistics.txt' % (
                        species, species, f, ct, prefix, c)
                    if not os.path.exists(fname):
                        results[i, j, k] = numpy.nan
                        print("Missing %s" % fname)
                    else:
                        infile = open(fname, 'r')
                        infile.readline()
                        results[i, j, k] = float(infile.readline().rstrip().split()[2])
                results[i, j, numpy.where(numpy.isnan(results[i, j, :]))[0]] = numpy.mean(
                    results[i, j, numpy.where(numpy.logical_not(numpy.isnan(results[i, j, :])))[0]])
        results[numpy.where(numpy.isnan(results))] = numpy.mean(results[numpy.where(numpy.logical_not(numpy.isnan(results)))])
        cond_labels = ['Normal', 'Shuffled CRE States']
        feature_labels = ['All Genes', 'Non-differentially Expressed Genes', 'Differentially Expressed Genes']
        colors = ['#E5007A', '#00E5FF', '#13FF00']
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        N = len(conds) * len(features)
        plt.grid(axis='y', color='gray')
        if species == 'hg38':
            ax.set_title("Human Gene Expression Prediction - %s" % plabel)
        else:
            ax.set_title("Mouse Gene Expression Prediction - %s" % plabel)
        ax.set_ylabel('Correlation of gene expression:prediction')
        data = tuple([results.reshape(N, -1)[x, :] for x in range(N)])
        parts = ax.violinplot(data,
                              showmeans=False, showmedians=False, showextrema=False)

        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(numpy.arange(2, len(features) * len(conds), 3))
        ax.set_xticklabels(cond_labels)
        ax.set_xlim(0.5, len(conds) * len(features) + 0.5)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
        offset = box.width * fig.get_size_inches()[0] * fig.dpi / (1.4 * len(conds) * len(features)) / 4
        for i, pc in enumerate(parts['bodies']):
            pc.set_edgecolor('black')
            pc.set_facecolor(colors[i % 3])
            pc.set_alpha(1)
            pc.set_offset_position('data')
            pc.set_offsets((offset*(1 - (i % 3)),0))
        quartile1, medians, quartile3 = numpy.percentile(data, [25, 50, 75], axis=1)
        whiskers = numpy.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

        inds = numpy.arange(1, len(medians) + 1).astype(numpy.float32)
        inds += .25 * (1 - numpy.mod(inds - 1, 3))
        ax.scatter(inds, medians, marker='o', color='white', s=10, zorder=3)
        ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

        patches = []
        for i, f in enumerate(feature_labels):
            patches.append(mpatches.Patch(color=colors[i], label=f))
        plt.legend(handles=patches, bbox_to_anchor=(0.0, -0.1),
                   mode='expand', edgecolor='white', loc='upper left')
        pdf.savefig()
        plt.close()

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = numpy.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = numpy.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

if __name__ == "__main__":
    main()