#!/usr/bin/env python

import sys
import argparse
import gzip

import numpy


def main():
    parser = generate_parser()
    args = parser.parse_args()
    rna = []
    for line in gzip.open(args.RNA, 'rb'):
        rna.append(line.rstrip().split())
    ids = {}
    for line in gzip.open(args.IDS, 'rb'):
        line = line.rstrip().split()
        if line[2] == b'protein_coding':
            ids[line[0]] = line[1]
    new_rna = []
    new_rna.append([b'chr', b'tss', b'gene_category', b'strand'] + rna[0][4:])
    for i in range(1, len(rna)):
        key = rna[i][3].split(b'.')[0]
        if key not in ids:
            continue
        if ids[key] == b'1':
            strand = b'+'
            tss = rna[i][1]
        else:
            strand = b'-'
            tss = rna[i][2]
        new_rna.append([rna[i][0], tss, b'protein_coding', strand]
                       + ["{}".format(numpy.log2(float(x) + 1)).encode() for x in rna[i][4:]])
    output = gzip.open(args.OUTPUT, 'wb')
    for line in new_rna:
        output.write(b'\t'.join(line + [b'\n']))
    output.close()

def generate_parser():
    """Generate an argument parser."""
    description = "%(prog)s -- Convert an RNA TPM text file to a numpy NPY file"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-r', '--rna', dest="RNA", type=str, action='store', required=True,
                        help="RNA expression file")
    parser.add_argument('-i', '--ids', dest="IDS", type=str, action='store', required=True,
                        help="Ensembl gene ID file")
    parser.add_argument('-o', '--output', dest="OUTPUT", type=str, action='store', required=True,
                        help="Numpy NPY RNA expression file to write to")
    return parser

if __name__ == "__main__":
    main()
