#!/usr/bin/env python

import sys

import numpy
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def main():
    species, out_fname = sys.argv[1:3]
    with PdfPages(out_fname) as pdf:
        beta_means, state_order = plot(species, pdf)
    output = open(out_fname.replace('.pdf', '_state_means.txt'), 'w')
    print("state\tpromoter\tcre", file=output)
    for i in range(beta_means.shape[0]):
        print("%i\t%f\t%f" % (state_order[i], beta_means[i, 0], beta_means[i, 1]), file=output)
    output.close()

def plot(species, pdf):
    if species == 'hg38': 
        CTs = ["B", "CD4", "CD8", "CLP", "CMP", "EOS", "ERY",
               "GMP", "MK", "MONc", "MONp", "MPP", "NEU", "NK"]
        state_order = [25, 13, 21, 12, 7, 26, 23, 19, 24, 16, 11, 14, 3, 9,
                       15, 2, 0, 5, 1, 4, 20, 6, 22, 10, 18, 8, 17]
        state_colors = [[253, 152, 9], [254, 215, 10], [240, 127, 71], [250, 253, 10],
                        [249, 253, 10], [251, 0, 7], [240, 128, 8], [228, 0, 55],
                        [251, 0, 7], [244, 0, 7], [185, 0, 5], [241, 249, 146],
                        [254, 255, 154], [237, 142, 190], [252, 91, 101], [236, 236, 236],
                        [254, 254, 254], [213, 211, 254], [205, 239, 208], [248, 206, 232],
                        [189, 218, 187], [46, 185, 65], [148, 153, 149], [139, 139, 139],
                        [6, 0, 227], [28, 0, 233], [46, 0, 176]]
    else:
        CTs = ["CFU", "CFUMK", "CMP", "ER4", "ERY-fl", "G1E", "GMP",
               "LSK", "MEP", "MK-imm-ad", "MONO", "NEU"]
        state_order = [5, 4, 11, 9, 12, 6, 0, 2, 7, 20, 3, 16, 22, 26,
                       13, 17, 1, 25, 14, 8, 24, 15, 10, 23, 19, 21, 18]
        state_colors = [[247, 227, 11], [249, 254, 60], [243, 33, 25],
                        [193, 51, 136], [223, 107, 18], [251, 213, 158],
                        [255, 255, 255], [124, 124, 124], [199, 6, 249],
                        [0, 0, 209], [36, 13, 224], [69, 61, 200],
                        [0, 0, 185], [193, 0, 167], [141, 0, 240], 
                        [55, 132, 52], [17, 129, 3], [11, 85, 1], 
                        [92, 177, 5], [17, 132, 3], [202, 0, 60],
                        [179, 0, 30], [238, 0, 6], [229, 123, 7],
                        [247, 101, 7], [239, 0, 7], [246, 0, 6]]
    for i in range(len(state_colors)):
        state_colors[i] = RGB2Hex(state_colors[i])
    cmap = ListedColormap(state_colors)
    conds = ['', '_control']
    features = ['all', 'diff', 'nodiff']
    cond_labels = ['Normal', 'Shuffled CRE States']
    feature_labels = ['All Genes', 'Differential Expressed Genes',
                      'Non-differentially Expressed Genes']
    beta_means = numpy.zeros((len(state_order), 2), dtype=numpy.float32)
    for prefix, plabel in [['both', 'CRE + Promoter'], ['cre', 'CRE only'], ['promoter', 'Promoter only']]:
        fig = plt.figure(constrained_layout=True, figsize=(14.5, 0.65 * len(CTs) + 1.4))
        heights = [1]
        for i in range(len(conds)):
            heights.append(2 * len(CTs))
        heights.append(1)
        gs = fig.add_gridspec(len(heights), len(features), height_ratios=heights)
        results = []
        maxval = 0
        betaN = 1
        for i, f in enumerate(features):
            for j, c in enumerate(conds):
                results.append(numpy.zeros((2 * len(CTs), len(state_order)),
                                           dtype=numpy.float32))
                for k, ct in enumerate(CTs):
                    #infile = open('Results_%s/%s_%s_%s_%s_0%s_betas.txt' % (
                    infile = open('Results_%s_PCA/%s_%s_%s_%s_0%s_betas.txt' % (
                        species, species, f, ct, prefix, c), 'r')
                    line = infile.readline().split()
                    if len(line) == 3:
                        betaN = 2
                    for l, line in enumerate(infile):
                        line = line.rstrip().split()
                        if betaN == 1:
                            results[-1][k, l] = float(line[1])
                        else:
                            results[-1][k, l] = float(line[2])
                            results[-1][k + len(CTs), l] = float(line[1])
                            if prefix == 'both':
                                beta_means[l, 0] += float(line[1])
                                beta_means[l, 1] += float(line[2])
                maxval = max(maxval, numpy.amax(numpy.abs(results[-1])))
                results[-1] = results[-1][:, state_order]
                #if betaN == 1:
                #    results[-1] = results[-1][:len(CTs), :]
        maxval = numpy.ceil(maxval * 100) / 100
        ylabels = [r'$\beta_{C,j}$', r'$\beta_{P,j}$']
        for i in range(len(features)):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(numpy.arange(len(state_order)).reshape(1, -1), cmap=cmap)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            for j in range(len(state_order)):
                ax.annotate("%i" % state_order[j], ((j + 0.5) / len(state_order), -1.5),
                            xycoords='axes fraction', ha='center', va='bottom', rotation=90)
            if i == 1:
                if species == 'hg38':
                    ax.set_title("Human Beta Values - %s" % plabel)
                else:
                    ax.set_title("Mouse Beta Values - %s" % plabel)
        for i, f in enumerate(features):
            for j, c in enumerate(conds):
                ax = fig.add_subplot(gs[j + 1, i])
                data = numpy.copy(results[i * len(conds) + j])
                data /= maxval * 2
                data += 0.5
                ax.imshow(data, cmap='seismic', norm=Normalize(0, 1))
                ax.get_yaxis().set_visible(False)
                ax.get_xaxis().set_visible(False)
                if i == 0:
                    for k in range(2):
                        ax.annotate(ylabels[k], (-0.5 / len(state_order), 0.75 - 0.5 * k),
                                    xycoords='axes fraction', ha='right', va='center')
                    ax.annotate(cond_labels[j], (-2.5 / len(state_order), 0.5),
                                xycoords='axes fraction', ha='right', va='center', rotation=90)
                ax.axhline(len(CTs) - 0.55, lw=1, color='black')
                if j == len(conds) - 1:
                    ax.annotate(feature_labels[i], (0.5, -0.5 / len(CTs)),
                                xycoords='axes fraction', ha='center', va='top')
        ax = fig.add_subplot(gs[len(heights) - 1, 1])
        ax.imshow(numpy.tile(numpy.linspace(-maxval, maxval, 513), (25, 1)), cmap='seismic')
        ax.annotate("%0.2f" % (-maxval), (0, -0.5), 
                    xycoords='axes fraction', ha='center', va='top')
        ax.annotate("%0.2f" % (maxval), (1, -0.5), 
                    xycoords='axes fraction', ha='center', va='top')
        ax.annotate(r"$\beta$ Score", (0.5, -0.5), 
                    xycoords='axes fraction', ha='center', va='top')
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        pdf.savefig()
        plt.close()
    beta_means /= len(CTs)
    beta_means = beta_means[state_order, :]
    return beta_means, state_order

def RGB2Hex(rgb):
    r, g, b = rgb
    hexval = "#%s%s%s" % (hex(r)[2:].upper().rjust(2, '0'),
                          hex(g)[2:].upper().rjust(2, '0'),
                          hex(b)[2:].upper().rjust(2, '0'))
    return hexval

if __name__ == "__main__":
    main()
