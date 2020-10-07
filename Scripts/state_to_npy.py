#!/usr/bin/env python

import sys
import argparse
import gzip

import numpy

def main():
    parser = generate_parser()
    args = parser.parse_args()
    print("\r%s\rLoading data" % (' ' * 80), end='', file=sys.stderr)
    fs = gzip.open(args.STATE, 'rb')
    header = fs.readline().rstrip(b'\n\r').split()[:-1]
    data = []
    chromlen = 0
    for line in fs:
        line = line.rstrip(b'\r\n').split()
        binID, chrom, start, end = line[:4]
        state = numpy.array(line[4:-1], dtype=numpy.int8)
        data.append(tuple([chrom.decode(), int(start), int(end)] + list(state)))
        chromlen = max(chromlen, len(chrom))
    dtype = [('chr', 'S%i' % chromlen), ('start', numpy.int32), ('end', numpy.int32)]
    for name in header[4:]:
        dtype.append((name.decode(), numpy.int8))
    data = numpy.array(data, dtype=numpy.dtype(dtype))
    # Remove bins with all zero states
    print("\r%s\rRemoving zero bins" % (' ' * 80), end='', file=sys.stderr)
    maxState = numpy.zeros(data.shape[0])
    names = data.dtype.names[3:]
    cellN = len(names)
    for i in range(cellN):
        maxState = numpy.maximum(maxState, data[names[i]])
        stateN = numpy.amax(data[names[i]]) + 1
    data = data[numpy.where(maxState > 0)]
    # Sort data
    print("\r%s\rSorting data" % (' ' * 80), end='', file=sys.stderr)
    data = data[numpy.lexsort((data['start'], data['chr']))]
    chroms = numpy.unique(data['chr'])
    chr_indices = numpy.r_[0, numpy.where(data['chr'][1:] != data['chr'][:-1])[0] + 1, data.shape[0]]
    # Resolve overlapping bins
    new_bins = []
    valid = numpy.ones(data.shape[0], dtype=numpy.bool)
    r = numpy.arange(cellN)
    for i in range(chr_indices.shape[0] - 1):
        s = chr_indices[i]
        e = chr_indices[i + 1]
        where = numpy.where(data['end'][s:(e - 1)] > data['start'][(s + 1):e])[0] + s
        pos = 0
        while pos < where.shape[0]:
            print("\r%s\rResolving overlaps, %s %i of %i" % (' ' * 80, chroms[i], pos, where.shape[0]), end='', file=sys.stderr)
            j = where[pos]
            valid[j] = False
            temp = [data['start'][j], data['end'][j]]
            while j < e and data['start'][j] < temp[-1]:
                temp.insert(1, data['start'][j])
                if data['end'][j] > temp[-1]:
                    temp.append(data['end'][j])
                else:
                    temp.insert(1, data['end'][j])
                valid[j] = False
                j += 1
            temp = numpy.unique(temp)
            new_entries = numpy.zeros(temp.shape[0] - 1, dtype=numpy.dtype([
                ('start', numpy.int32), ('end', numpy.int32), ('state', numpy.int8, (cellN, stateN))]))
            new_entries['start'] = temp[:-1]
            new_entries['end'] = temp[1:]
            starts = numpy.searchsorted(new_entries['end'], data['start'][where[pos]:j], side='left')
            stops = numpy.searchsorted(new_entries['start'], data['end'][where[pos]:j], side='left')
            states = numpy.zeros((starts.shape[0], cellN), dtype=numpy.int8)
            for k in range(cellN):
                states[:, k] = data[names[k]][where[pos]:j]
            for k in range(starts.shape[0]):
                new_entries['state'][starts[k]:(stops[k] + 1), r, states[k, :]] += 1
            for k in range(new_entries.shape[0]):
                states = numpy.zeros(cellN, dtype=numpy.int8)
                for l in range(cellN):
                    best = numpy.where(new_entries['state'][k, l, :] == numpy.amax(new_entries['state'][k, l, :]))[0]
                    if best.shape[0] == 1:
                        states[l] = best[0]
                    elif best.shape[0] > 1:
                        states[l] = numpy.random.choice(best)
                if numpy.amax(states) > 0:
                    new_bins.append(tuple([chroms[i], new_entries['start'][k],
                                          new_entries['end'][k]] + list(states)))
            while pos < where.shape[0] and data['start'][where[pos]] < new_entries['end'][-1]:
                pos += 1
    new_bins = numpy.array(new_bins, dtype=data.dtype)
    print("\r%s\rParsed %i bins into %i new bins" % (' ' * 80, numpy.sum(numpy.logical_not(valid)),
                                                     new_bins.shape[0]), end='', file=sys.stderr)
    data = numpy.hstack((data[numpy.where(valid)], new_bins))
    data = data[numpy.lexsort((data['start'], data['chr']))]
    # Join unchnaged adjacent bins
    chr_indices = numpy.r_[0, numpy.where(data['chr'][1:] != data['chr'][:-1])[0] + 1, data.shape[0]]
    valid = numpy.ones(data.shape[0], dtype=numpy.bool)
    for i in range(chr_indices.shape[0] - 1):
        print("\r%s\rCollapsing data, %s" % (' ' * 80, chroms[i]), end='', file=sys.stderr)
        s = chr_indices[i]
        e = chr_indices[i + 1]
        current = s
        current_state = list(data[s])[3:]
        for j in range(s + 1, e):
            if (numpy.array_equal(list(data[j])[3:], current_state)
                and data['end'][current] == data['start'][j]):
                data['end'][current] = data['end'][j]
                valid[j] = False
            else:
                current = j
                current_state = list(data[j])[3:]
    data = data[numpy.where(valid)]
    print("\r%s\rSaving data" % (' ' * 80), end='', file=sys.stderr)
    numpy.save(args.OUTPUT, data)
    print("\r%s\r" % (' ' * 80), end='', file=sys.stderr)

def generate_parser():
    """Generate an argument parser."""
    description = "%(prog)s -- Convert an IDEAS state text file to a numpy NPY file"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-s', '--state', dest="STATE", type=str, action='store', required=True,
                        help="RNA expression file")
    parser.add_argument('-o', '--output', dest="OUTPUT", type=str, action='store', required=True,
                        help="State file")
    return parser

if __name__ == "__main__":
    main()
