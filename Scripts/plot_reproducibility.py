#!/usr/bin/env python

import argparse
import glob
import gzip
import logging
import multiprocessing
import sys

import numpy
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def main():
    species, prefix, state_fname, rna_fname, cre_fname, out_fname = sys.argv[1:7]

    if species == 'hg38':
        state_order = numpy.array([25, 13, 21, 12, 7, 26, 23, 19, 24, 16, 11, 14, 3, 9,
                                   15, 2, 0, 5, 1, 4, 20, 6, 22, 10, 18, 8, 17])
        state_colors = [[253, 152, 9], [254, 215, 10], [240, 127, 71], [250, 253, 10],
                        [249, 253, 10], [251, 0, 7], [240, 128, 8], [228, 0, 55],
                        [251, 0, 7], [244, 0, 7], [185, 0, 5], [241, 249, 146],
                        [254, 255, 154], [237, 142, 190], [252, 91, 101], [236, 236, 236],
                        [254, 254, 254], [213, 211, 254], [205, 239, 208], [248, 206, 232],
                        [189, 218, 187], [46, 185, 65], [148, 153, 149], [139, 139, 139],
                        [6, 0, 227], [28, 0, 233], [46, 0, 176]]
        species_name = 'Human'
    else:
        state_order = numpy.array([5, 4, 11, 9, 12, 6, 0, 2, 7, 20, 3, 16, 22, 26,
                       13, 17, 1, 25, 14, 8, 24, 15, 10, 23, 19, 21, 18])
        state_colors = [[247, 227, 11], [249, 254, 60], [243, 33, 25],
                        [193, 51, 136], [223, 107, 18], [251, 213, 158],
                        [255, 255, 255], [124, 124, 124], [199, 6, 249],
                        [0, 0, 209], [36, 13, 224], [69, 61, 200],
                        [0, 0, 185], [193, 0, 167], [141, 0, 240], 
                        [55, 132, 52], [17, 129, 3], [11, 85, 1], 
                        [92, 177, 5], [17, 132, 3], [202, 0, 60],
                        [179, 0, 30], [238, 0, 6], [229, 123, 7],
                        [247, 101, 7], [239, 0, 7], [246, 0, 6]]
        species_name = 'Mouse'
    stateN = state_order.shape[0]
    for i in range(len(state_colors)):
        state_colors[i] = RGB2Hex(state_colors[i])
    colors = ['#E5007A', '#00E5FF', '#13FF00']
    N = 50
    model = LinReg(rna_fname, state_fname, cre_fname, 0)
    CTs = model.celltypes
    cre_indices = {}
    for i in range(model.cre.shape[0]):
        cre_indices[(model.cre['chr'][i].decode('utf8'), model.cre['start'][i])] = numpy.int64(i)
    creN = numpy.int64(len(cre_indices))
    tss_indices = {}
    for i in range(model.rna.shape[0]):
        tss_indices[(model.rna['chr'][i].decode('utf8'), model.rna['TSS'][i])] = numpy.int64(i) * creN
    with PdfPages(out_fname) as pdf:
        for feat, flabel in [['both', 'CRE + Promoter']]:
            cre_sets = []
            for ct in CTs:
                jobs_q = multiprocessing.JoinableQueue()
                data = []
                for i in range(2):
                    data.append([])
                    fnames = glob.glob("%s_%s_%s_*_eRP.txt.gz" % (prefix, ct, feat))
                    if i == 0:
                        for j in range(len(fnames))[::-1]:
                            if fnames[j].count('control') > 0:
                                fnames.pop(j)
                    else:
                        for j in range(len(fnames))[::-1]:
                            if fnames[j].count('control') == 0:
                                fnames.pop(j)
                    print(fnames)
                    for j in range(len(fnames)):
                        jobs_q.put((i, j, fnames[j]))
                        data[i].append(None)
                if len(data[0]) == 0 and len(data[1]) == 0:
                    continue
                print(ct)
                for i in range(N):
                    jobs_q.put(None)
                results_q = multiprocessing.JoinableQueue()
                workers = []
                for i in range(N):
                    workers.append(multiprocessing.Process(
                        target=find_pair_indices, args=(jobs_q, results_q, tss_indices, cre_indices)))
                    workers[-1].daemon = True
                    workers[-1].start()
                finished = 0
                while finished < N:
                    temp = results_q.get(True)
                    if temp == None:
                        finished += 1
                    else:
                        data[temp[0]][temp[1]] = temp[2]
                cre_sets.append(data)

                # Find Jaccard graph
                jobs_q = multiprocessing.JoinableQueue()
                results = []
                for i in range(len(data)):
                    results.append(numpy.zeros(len(data[i]) * (len(data[i]) - 1) // 2, dtype=numpy.float32))
                    x = 0
                    for j in range(len(data[i]) - 1):
                        for k in range(j + 1, len(data[i])):
                            jobs_q.put((i, x, data[i][j], data[i][k]))
                            x += 1
                for i in range(N):
                    jobs_q.put(None)
                results_q = multiprocessing.JoinableQueue()
                workers = []
                for i in range(N):
                    workers.append(multiprocessing.Process(
                        target=find_jaccard, args=(jobs_q, results_q)))
                    workers[-1].daemon = True
                    workers[-1].start()
                finished = 0
                while finished < N:
                    temp = results_q.get(True)
                    if temp == None:
                        finished += 1
                    else:
                        results[temp[0]][temp[1]] = temp[2]
                jaccards = tuple([results[x] for x in range(len(results))])
                fig = plt.figure(constrained_layout=True, figsize=(0.3 * stateN + 1.4, 11))
                fig.suptitle("%s %s Reproducibility" % (species_name, ct))
                gs = fig.add_gridspec(4, 2, height_ratios=[1, 10, 10, 10])
                if len(jaccards) > 1 and len(jaccards[1]) > 2:
                    ax = fig.add_subplot(gs[1, 0])
                    ax.grid(axis='y', color='gray')
                    ax.set_ylabel('Jaccard Index of Selected CREs')
                    parts = ax.violinplot(jaccards,
                                          showmeans=False, showmedians=False, showextrema=False)
                    for i, pc in enumerate(parts['bodies']):
                        pc.set_edgecolor('black')
                        pc.set_facecolor(colors[i % 3])
                        pc.set_alpha(1)
                    #quartile1, medians, quartile3 = numpy.percentile(jaccards, [25, 50, 75], axis=1)
                    #whiskers = numpy.array([
                    #    adjacent_values(sorted_array, q1, q3)
                    #    for sorted_array, q1, q3 in zip(jaccards, quartile1, quartile3)])
                    #whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

                    #inds = numpy.arange(1, len(medians) + 1).astype(numpy.float32)
                    #ax.scatter(inds, medians, marker='o', color='white', s=10, zorder=3)
                    #ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
                    #ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
                    features = ['Normal', 'Shuffled CRE States']
                    ax.get_xaxis().set_tick_params(direction='out')
                    ax.xaxis.set_ticks_position('bottom')
                    ax.set_xticks(numpy.arange(1, len(features) + 1))
                    ax.set_xticklabels(features)
                    ax.set_xlim(0.5, len(features) + 0.5)
                    box = ax.get_position()
                    #ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])

                    ax.set_title("%s TSS-CRE Pair Selection Reproducibility - %s" % (species_name, flabel))

                    # CRE selection graph
                    results_q = multiprocessing.JoinableQueue()
                    workers = []
                    results = []
                    for i in range(len(data)):
                        results.append(None)
                        workers.append(multiprocessing.Process(
                            target=find_usage, args=(i, data[i], results_q)))
                        workers[-1].daemon = True
                        workers[-1].start()
                    for i in range(len(data)):
                        temp = results_q.get(True)
                        results[temp[0]] = temp[1]
                    ax1 = fig.add_subplot(gs[1, -1])
                    lines = []
                    for i in range(len(results)):
                        ax1.plot(numpy.arange(1, results[i].shape[0] + 1), results[i],
                                             color=colors[i])
                        lines.append(mlines.Line2D([], [], color=colors[i], label=features[i]))
                    ax1.set_xlim(1, len(results[0]))
                    ax1.set_ylabel('Percentage of Pairs')
                    ax1.set_xlabel('# Datasets Containing TSS-CRE Pair')
                    ax1.set_title("%s TSS-CRE Pair Selection Frequency" % species_name)
                    ax1.legend(handles=lines, loc='upper right')
                    
                    # Plot betas reproducibility
                    ax2 = [fig.add_subplot(gs[2, :]), fig.add_subplot(gs[3, :])]
                    data = []
                    minval = numpy.inf
                    maxval = -numpy.inf
                    for h, cflag in enumerate(['', '_control']):
                        fnames = glob.glob("%s_%s_%s_*_betas.txt" % (prefix, ct, feat))
                        if h == 0:
                            for j in range(len(fnames))[::-1]:
                                if fnames[j].count('control') > 0:
                                    fnames.pop(j) 
                        else:
                            for j in range(len(fnames))[::-1]:
                                if fnames[j].count('control') == 0:
                                    fnames.pop(j)
                        if feat == 'both':
                            data.append(numpy.zeros((len(fnames), stateN * 2), dtype=numpy.float32))
                        else:
                            data.append(numpy.zeros((len(fnames), stateN), dtype=numpy.float32))
                        for i, fname in enumerate(fnames):
                            f = open(fname)
                            f.readline()
                            for j in range(stateN):
                                line = f.readline().rstrip().split()
                                if len(line) == 0:
                                    continue
                                if feat == 'both':
                                    data[h][i, j] = float(line[2])
                                    data[h][i, j + stateN] = float(line[1])
                                else:
                                    data[h][i, j] = float(line[1])
                        if feat == 'both':
                            data[h] = data[h][:, numpy.r_[state_order, state_order + stateN]]
                        else:
                            data[h] = data[h][:, state_order]
                        minval = min(minval, numpy.amin(data[h]))
                        maxval = max(maxval, numpy.amax(data[h]))
                        span = maxval - minval
                    for h, cflag in enumerate(['Normal', 'Shuffled CRE States']):
                        data[h] = tuple([data[h][:, x] for x in range(data[h].shape[1])])
                        a = ax2[h]
                        a.set_ylim(minval - span * 0.05, maxval + span * 0.05)
                        if feat == 'both':
                            a.set_xlim(0, stateN * 2 + 2)
                        else:
                            a.set_xlim(0, stateN + 1)
                        a.get_xaxis().set_visible(False)
                        parts = a.violinplot(data[h],
                                             showmeans=False, showmedians=False, showextrema=False)
                        a.annotate(cflag, (1.025, 0.5), rotation=-90,
                                   xycoords='axes fraction', va='center', ha='right')
                        for i, pc in enumerate(parts['bodies']):
                            pc.set_edgecolor('black')
                            pc.set_facecolor(state_colors[i % stateN])
                            pc.set_alpha(1)
                            pc.set_offset_position('data')
                            pc.set_offsets((11.5 * (i // stateN), 0))
                            a.axvline(i + 1 + (i // stateN), lw=0.25, color='gray')
                        if feat == 'both':
                            a.axvline(stateN + 1, lw=2, color='black')
                        a.axhline(0, lw=1, color='black')
                        a.set_ylabel('Beta values')
                    if feat == 'both':
                        for j in range(state_order.shape[0]):
                            ax2[1].annotate("%i" % state_order[j], ((j + 1) / (stateN * 2 + 2), -0.08),
                                            xycoords='axes fraction', ha='center', va='bottom', rotation=90)
                            ax2[1].annotate("%i" % state_order[j], ((j + 2 + stateN) / (stateN * 2 + 2), -0.08),
                                            xycoords='axes fraction', ha='center', va='bottom', rotation=90)
                        ax2[1].annotate(r"$\beta_{C}$", (0.25, -0.12), xycoords='axes fraction',
                                        ha='center', va='top')
                        ax2[1].annotate(r"$\beta_{P}$", (0.75, -0.12), xycoords='axes fraction',
                                        ha='center', va='top')
                    else:
                        for j in range(state_order.shape[0]):
                            ax2[1].annotate("%i" % state_order[j], ((j + 1) / (stateN + 1), -0.08),
                                            xycoords='axes fraction', ha='center', va='bottom', rotation=90)
                            ax2[1].annotate(r"$\beta_{C}$", (0.5, -0.12), xycoords='axes fraction',
                                            ha='center', va='top')
                    ax2[0].title.set_text("%s Beta Value Reproducibility - %s" % (species_name, flabel))
                pdf.savefig()
                plt.close()

            if len(cre_sets) < 2:
                continue
            data0, cre_set0 = find_cre_freq(cre_sets, 0)
            data1, cre_set1 = find_cre_freq(cre_sets, 1)
            hm0 = numpy.histogram2d(data0[:, 0], data0[:, 1],
                                    [numpy.linspace(0.5, len(cre_sets[0][0]) + 0.5, len(cre_sets[0][0]) + 1),
                                     numpy.linspace(0.5, len(cre_sets) + 0.5, len(cre_sets) + 1)],
                                    density=False)[0]
            hm1 = numpy.histogram2d(data1[:, 0], data1[:, 1],
                                    [numpy.linspace(0.5, len(cre_sets[0][0]) + 0.5, len(cre_sets[0][0]) + 1),
                                     numpy.linspace(0.5, len(cre_sets) + 0.5, len(cre_sets) + 1)],
                                    density=False)[0]
            hm0 /= numpy.amax(hm0)
            hm1 /= numpy.amax(hm1)
            hm0 = numpy.log2(hm0 + 1)
            hm1 = numpy.log2(hm1 + 1)
            fig = plt.figure(constrained_layout=True)
            fig.suptitle("%s CRE-TSS Pair Selection" % species_name)
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 10])
            for i, hm in enumerate([hm0, hm1]):
                ax = fig.add_subplot(gs[1, i])
                ax.imshow(hm, cmap='magma', norm=Normalize(0, 1))
            pdf.savefig()
            plt.close()
            #where = numpy.where(numpy.logical_and(data0[:, 0] >= 0.8, data0[:, 1] >= 0.8))[0]
            where = numpy.arange(data0.shape[0])
            scores = data0[where, 0] * data0[where, 1] / float(len(cre_sets[0][0]) * len(cre_sets))
            scores /= numpy.amax(scores)
            scores = numpy.round(scores * 1000.99 - 0.5).astype(numpy.int32)
            value1 = data0[where, 0]
            value2 = data0[where, 1]
            tss_i = cre_set0[where] // creN
            cre_i = cre_set0[where] - tss_i * creN
            output = open(out_fname.replace('.pdf', '.int'), 'w')
            print('track type=interact name="%s TSS-CRE" description="%s TSS-CRE" interactDirectional=true maxHeightPixels=200:100:50 useScore=on visibility=full' % (species_name, species_name), file=output)
            print('#chrom  chromStart  chromEnd  name  score  numReps numCTs  exp  color  sourceChrom  sourceStart  sourceEnd  sourceName  sourceStrand  targetChrom  targetStart  targetEnd  targetName  targetStrand', file=output)
            bool2strand = {False: "+", True: "-"}
            for i in range(tss_i.shape[0]):
                ti = tss_i[i]
                ci = cre_i[i]
                tss = model.rna['TSS'][ti]
                chrom = model.rna['chr'][ti].decode('utf8')
                strand = bool2strand[model.rna['strand'][ti]]
                start = model.cre['start'][ci]
                end = model.cre['end'][ci]
                print("%s\t%i\t%i\t.\t%i\t%i\t%i\t.\t0\t%s\t%i\t%i\t.\t.\t%s\t%i\t%i\t.\t%s" %
                      (chrom, min(tss, start), max(tss + 1, end), scores[i], value1[i], value2[i], chrom,
                       start, end, chrom, tss, tss + 1, strand), file=output)
            output.close()

def RGB2Hex(rgb):
    r, g, b = rgb
    hexval = "#%s%s%s" % (hex(r)[2:].upper().rjust(2, '0'),
                          hex(g)[2:].upper().rjust(2, '0'),
                          hex(b)[2:].upper().rjust(2, '0'))
    return hexval

def find_pair_indices(jobs_q, results_q, tss_indices, cre_indices):
    data = jobs_q.get(True)
    while data != None:
        i, j, fname = data
        temp = []
        f = gzip.open(fname, 'rb')
        f.readline()
        for line in f:
            line = line.rstrip().split()
            chrom = line[0].decode('utf8').strip("b'")
            temp.append(tss_indices[(chrom, int(line[1]))]
                        + cre_indices[(chrom, int(line[3]))])
        f.close()
        temp = numpy.array(temp)
        temp = numpy.unique(temp)
        results_q.put((i, j, temp))
        data = jobs_q.get(True)
    results_q.put(None)

def find_jaccard(jobs_q, results_q):
    data = jobs_q.get(True)
    while data != None:
        i, j, set1, set2 = data
        joint_set = numpy.unique(numpy.r_[set1, set2])
        present = numpy.zeros((joint_set.shape[0], 2), dtype=numpy.bool)
        present[numpy.searchsorted(joint_set, set1, side='left'), 0] = True
        present[numpy.searchsorted(joint_set, set2, side='left'), 1] = True
        jaccard = (numpy.sum(numpy.logical_and(present[:, 0], present[:, 1]))
                   / float(present.shape[0]))
        results_q.put((i, j, jaccard))
        data = jobs_q.get(True)
    results_q.put(None)

def find_usage(i, data, results_q):
    all_pairs = numpy.copy(data[0])
    for j in range(1, len(data)):
        all_pairs = numpy.r_[all_pairs, data[j]]
    upairs = numpy.unique(all_pairs)
    indices = numpy.searchsorted(upairs, all_pairs, side='left')
    pair_counts = numpy.bincount(indices)
    result = numpy.bincount(pair_counts, minlength=(len(data) + 1))[1:]
    result = result.astype(numpy.float32) / numpy.sum(result)
    results_q.put((i, result))

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = numpy.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = numpy.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def find_cre_freq(data, index):
    all_cres = numpy.zeros(0, dtype=numpy.int64)
    for i in range(len(data)):
        for j in range(len(data[i][index])):
            all_cres = numpy.unique(numpy.r_[all_cres, data[i][index][j]])
    counts = numpy.zeros((all_cres.shape[0], len(data)), dtype=numpy.int32)
    for i in range(len(data)):
        for j in range(len(data[i][index])):
            indices = numpy.searchsorted(all_cres, data[i][index][j])
            counts[indices, i] += 1
    freq = numpy.zeros((counts.shape[0], 2), dtype=numpy.float32)
    for i in range(counts.shape[0]):
        nonzero = numpy.where(counts[i, :] > 0)[0]
        freq[i, 0] = numpy.mean(counts[i, nonzero])
        freq[i, 1] = nonzero.shape[0]
    return freq, all_cres
    

class LinReg(object):
    log_levels = {
        -1: logging.NOTSET,
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }

    def __init__(self, rna, state, cre, verbose=2):
        self.verbose = verbose
        self.logger = logging.getLogger()
        self.logger.setLevel(self.log_levels[verbose])
        ch = logging.StreamHandler()
        ch.setLevel(self.log_levels[verbose])
        formatter = logging.Formatter(
            fmt='%(asctime)s %(levelname)s %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.propagate = False
        self.rna_fname = rna
        self.state_fname = state
        self.cre_fname = cre
        self.load_CREs()
        self.get_valid_celltypes()
        self.load_rna()
        #self.load_state()

    def __getitem__(self, key):
        """Dictionary-like lookup."""
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            return None

    def __setitem__(self, key, value):
        """Dictionary-like value setting."""
        self.__dict__[key] = value
        return None

    def get_valid_celltypes(self):
        """Find a common set of celltypes between RNA and state files and determine # of reps for each"""
        if self.rna_fname.split('.')[-1] == 'npy':
            rna_header = numpy.load(self.rna_fname).dtype.names[4:]
            rna_celltypes = [x.split('_')[0] for x in rna_header]
        else:
            fs = open(self.rna_fname)
            rna_header = fs.readline().rstrip('\r\n').split()[4:]
            fs.close()
            rna_celltypes = [x.split('_')[0] for x in rna_header]
        if self.state_fname.split('.')[-1] == 'npy':
            state_header = numpy.load(self.state_fname).dtype.names[3:-1]
            state_celltypes = [x.split('_')[0] for x in state_header]
        else:
            fs = open(self.state_fname)
            state_header = fs.readline().rstrip('\r\n').split()[4:-1]
            fs.close()
            state_celltypes = [x.split('_')[0] for x in state_header]
        celltypes = list(set(rna_celltypes).intersection(set(state_celltypes)))
        celltypes.sort()
        celltypes = numpy.array(celltypes)
        self.celltypes = celltypes
        self.cellN = self.celltypes.shape[0]
        self.cellN1 = self.cellN - 1
        self.rna_cindices = numpy.zeros(self.cellN + 1, dtype=numpy.int32)
        self.state_cindices = numpy.zeros(self.cellN + 1, dtype=numpy.int32)
        rna_valid = numpy.zeros(len(rna_header), dtype=numpy.bool)
        for i, name in enumerate(rna_celltypes):
            where = numpy.where(celltypes == name)[0]
            if where.shape[0] > 0:
                self.rna_cindices[where[0] + 1] += 1
        self.rna_cindices = numpy.cumsum(self.rna_cindices)
        pos = numpy.copy(self.rna_cindices)
        self.valid_rna_celltypes = numpy.zeros(self.rna_cindices[-1], dtype=numpy.int32)
        for i, name in enumerate(rna_celltypes):
            where = numpy.where(celltypes == name)[0]
            if where.shape[0] > 0:
                self.valid_rna_celltypes[pos[where[0]]] = i
                pos[where[0]] += 1
        self.rna_reps = self.rna_cindices[1:] - self.rna_cindices[:-1]
        self.rnaRepN = self.rna_cindices[-1]
        for i, name in enumerate(state_celltypes):
            where = numpy.where(celltypes == name)[0]
            if where.shape[0] > 0:
                self.state_cindices[where[0] + 1] += 1
        self.state_cindices = numpy.cumsum(self.state_cindices)
        pos = numpy.copy(self.state_cindices)
        self.valid_state_celltypes = numpy.zeros(self.state_cindices[-1], dtype=numpy.int32)
        for i, name in enumerate(state_celltypes):
            where = numpy.where(celltypes == name)[0]
            if where.shape[0] > 0:
                self.valid_state_celltypes[pos[where[0]]] = i
                pos[where[0]] += 1
        self.state_reps = self.state_cindices[1:] - self.state_cindices[:-1]
        self.stateRepN = self.state_cindices[-1]
        self.logger.info("Found RNA-state pairings for celltypes %s"
                         % ', '.join(list(self.celltypes)))

    def load_CREs(self):
        if self.verbose >= 2:
            print("\r%s\rLoading cCRE data" % (' ' * 80), end='', file=sys.stderr)
        fs = gzip.open(self.cre_fname, 'rb')
        header = fs.readline().rstrip(b'\r\n').split()
        data = []
        chromlen = 0
        for line in fs:
            line = line.rstrip(b'\r\n').split()
            chrom, start, end = line[:3]
            data.append((chrom.decode('utf8'), int(start), int(end)))
            chromlen = max(chromlen, len(chrom))
        fs.close()
        data = numpy.array(data, dtype=numpy.dtype([('chr', 'S%i' % chromlen),
                                                    ('start', numpy.int32),
                                                    ('end', numpy.int32)]))
        data = data[numpy.lexsort((data['start'], data['chr']))]
        self.cre = data
        self.cre_indices = numpy.r_[0, numpy.where(self.cre['chr'][1:] != self.cre['chr'][:-1])[0] + 1,
                                    self.cre.shape[0]]
        self.chroms = numpy.unique(self.cre['chr'])
        self.chr2int = {}
        for i, chrom in enumerate(self.chroms):
            self.chr2int[chrom] = i
        if self.verbose >= 2:
            print("\r%s\r" % (' ' * 80), end='', file=sys.stderr)
        self.logger.info("Chromosomes to be analyzed: %s"
                         % ', '.join([x.decode('utf-8') for x in self.chroms]))
        self.logger.info('Loaded %i CREs' % self.cre_indices[-1])

    def load_rna(self):
        if self.verbose >= 2:
            print("\r%s\rLoading RNA data" % (' ' * 80), end='', file=sys.stderr)
        if self.rna_fname.split('.')[-1] == 'npy':
            temp = numpy.load(self.rna_fname)
            valid = numpy.zeros(temp.shape[0], dtype=numpy.bool)
            for chrom in self.chroms:
                valid[numpy.where(temp['chr'] == chrom)] = True
            valid = numpy.where(valid)[0]
            data = numpy.empty(valid.shape[0], dtype=numpy.dtype([
                ('chr', temp['chr'].dtype), ('TSS', numpy.int32), ('strand', numpy.bool),
                ('rna', numpy.float32, (self.rnaRepN,))]))
            names = temp.dtype.names
            for name in names[:3]:
                data[name] = temp[name][valid]
            for i, j in enumerate(self.valid_rna_celltypes):
                data['rna'][:, i] = temp[names[j + 4]][valid]
        else:
            fs = open(self.rna_fname)
            header = fs.readline().rstrip('\r\n').split()
            genelen = 0
            strand2bool = {'+': False, '-': True}
            data = []
            for line in fs:
                line = line.rstrip('\r\n').split()
                chrom, tss, gene, strand = line[:4]
                if chrom not in self.chr2int:
                    continue
                TPM = numpy.array(line[3:], dtype=numpy.float32)[self.valid_rna_celltypes]
                data.append((chrom, int(tss), gene, strand2bool[strand], TPM))
                genelen = max(genelen, len(gene))
            fs.close()
            data = numpy.array(data, dtype=numpy.dtype([
                ('chr', self.chroms.dtype), ('TSS', numpy.int32), ('strand', numpy.bool),
                ('rna', numpy.float32, (self.rnaRepN,))]))
            data = data[numpy.lexsort((data['TSS'], data['chr']))]
        rna = numpy.zeros((data['rna'].shape[0], self.cellN), dtype=numpy.float32)
        for i in range(self.cellN):
            rna[:, i] = numpy.mean(data['rna'][:, self.rna_cindices[i]:self.rna_cindices[i + 1]], axis=1)
        self.rna = numpy.empty(data.shape[0], dtype=numpy.dtype([
                ('chr', data['chr'].dtype), ('TSS', numpy.int32), ('strand', numpy.bool),
                ('rna', numpy.float32, (self.cellN,))]))
        self.rna['chr'] = data['chr']
        self.rna['TSS'] = data['TSS']
        self.rna['strand'] = data['strand']
        self.rna['rna'] = numpy.copy(rna)
        self.rna_indices = numpy.zeros(self.chroms.shape[0] + 1, dtype=numpy.int32)
        for i, chrom in enumerate(self.chroms):
            self.rna_indices[i + 1] = (self.rna_indices[i]
                                       + numpy.where(self.rna['chr'] == chrom)[0].shape[0])
        self.tssN = self.rna.shape[0]
        if self.verbose >= 2:
            print("\r%s\r" % (' ' * 80), end='', file=sys.stderr)
        self.logger.info('Loaded %i expression profiles across %i celltypes (%i replicates)'
                         % (self.tssN, self.cellN, self.rnaRepN))

    def load_state(self):
        if self.verbose >= 2:
            print("\r%s\rLoading state data" % (' ' * 80), end='', file=sys.stderr)
        if self.state_fname.split('.')[-1] == 'npy':
            temp = numpy.load(self.state_fname)
            valid = numpy.zeros(temp.shape[0], dtype=numpy.bool)
            for chrom in self.chroms:
                valid[numpy.where(temp['chr'] == chrom)] = True
            valid = numpy.where(valid)[0]
            data = numpy.empty(valid.shape[0], dtype=numpy.dtype([
                ('chr', temp['chr'].dtype), ('start', numpy.int32), ('end', numpy.int32),
                ('state', numpy.int32, (self.stateRepN,))]))
            names = temp.dtype.names
            for name in names[:3]:
                data[name] = temp[name][valid]
            for i, j in enumerate(self.valid_state_celltypes):
                data['state'][:, i] = temp[names[j + 3]][valid]
        else:
            fs = open(self.state_fname)
            header = fs.readline().rstrip('\n\r').split()
            data = []
            for line in fs:
                line = line.rstrip('\r\n').split()
                binID, chrom, start, end = line[:4]
                if chrom not in self.chr2int:
                    continue
                state = numpy.array(line[4:-1], dtype=numpy.int32)[self.valid_state_celltypes]
                data.append((chrom, int(start), int(end), state))
            data = numpy.array(data, dtype=numpy.dtype([
                ('chr', self.chroms.dtype), ('start', numpy.int32), ('end', numpy.int32),
                ('state', numpy.int32, (self.stateRepN,))]))
            data = data[numpy.lexsort((data['start'], data['chr']))]
            fs.close()
        self.stateN = numpy.amax(data['state']) + 1
        self.state_indices = numpy.zeros(self.chroms.shape[0] + 1, dtype=numpy.int32)
        for i, chrom in enumerate(self.chroms):
            where = numpy.where(data['chr'] == chrom)[0]
            if where.shape[0] > 0:
                self.state_indices[i + 1] = self.state_indices[i] + where.shape[0]
            else:
                self.state_indices[i + 1] = self.state_indices[i]
        self.state = data
        if self.verbose >= 2:
            print("\r%s\r" % (' ' * 80), end='', file=sys.stderr)
        self.logger.info('Loaded %i non-zero state profiles (%i states) across %i celltypes (%i replicates)'
                         % (self.state.shape[0], self.stateN, self.cellN, self.stateRepN))


if __name__ == "__main__":
    main()
