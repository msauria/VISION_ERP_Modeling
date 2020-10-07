#!/usr/bin/env python

import sys
import argparse
import gzip

import numpy

def main():
    parser = generate_parser()
    args = parser.parse_args()
    fs = gzip.open(args.RNA, 'rb')
    header = fs.readline().rstrip(b'\r\n').split()
    chromlen = 0
    strand2bool = {b'+': False, b'-': True}
    data = []
    for line in fs:
        line = line.rstrip(b'\r\n').split()
        chrom, tss, gene, strand = line[:4]
        if gene != b'protein_coding':
            continue
        TPM = numpy.array(line[4:], dtype=numpy.float32)
        data.append(tuple([chrom.decode('utf8'), int(float(tss)), strand2bool[strand]] + list(TPM)))
        chromlen = max(chromlen, len(chrom))
    fs.close()
    dtype = [('chr', 'S%i' % chromlen), ('TSS', numpy.int32), ('strand', numpy.bool)]
    for gene in header[4:]:
        dtype.append((gene.decode('utf8'), numpy.float32))
    data = numpy.array(data, dtype=numpy.dtype(dtype))
    data = data[numpy.lexsort((data['TSS'], data['chr']))]
    unique = numpy.r_[0, numpy.where(numpy.logical_or(data['chr'][1:] != data['chr'][:-1],
                                                      data['TSS'][1:] != data['TSS'][:-1]))[0] + 1,
                      data.shape[0]]
    for i in range(unique.shape[0] - 1):
        if unique[i + 1] - unique[i] == 1:
            continue
        for j in range(3, len(dtype)):
            data[dtype[j][0]][unique[i]] = numpy.sum(data[dtype[j][0]][unique[i]:unique[i + 1]])
    data = data[unique[:-1]]
    numpy.save(args.OUTPUT, data)


def generate_parser():
    """Generate an argument parser."""
    description = "%(prog)s -- Convert an RNA TPM text file to a numpy NPY file"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-r', '--rna', dest="RNA", type=str, action='store', required=True,
                        help="RNA expression file")
    parser.add_argument('-o', '--output', dest="OUTPUT", type=str, action='store', required=True,
                        help="Numpy NPY RNA expression file to write to")
    return parser


if __name__ == "__main__":
    main()
