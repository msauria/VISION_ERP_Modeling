#!/usr/bin/env python

import argparse
import gzip
import logging
import multiprocessing
import sys

import numpy
import sklearn.decomposition


def main():
    parser = generate_parser()
    args = parser.parse_args()
    model = LinReg(args.rna, args.state, args.cre, args.verbose)
    model.run(args.output, args.initdist, args.promoter, args.proximal, args.distal, args.crepromdist,
              args.correlation, args.iterations, args.lessone, args.threads, args.trainstats,
              args.eRP, args.pca, args.log, args.tssdist, args.multirefine, args.singlestate,
              args.intercept, args.nozero, args.maxcres, args.skiptraining, args.shufflestates,
              args.shufflerna, args.shuffleallrna, args.outputfeatures, args.seed)

def generate_parser():
    """Generate an argument parser."""
    description = "%(prog)s -- Predict RNA expression from cCREs and Ideas states"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-r', '--rna', dest="rna", type=str, action='store', required=True,
                        help="RNA expression file")
    parser.add_argument('-s', '--state', dest="state", type=str, action='store', required=True,
                        help="State file")
    parser.add_argument('-c', '--cre', dest="cre", type=str, action='store', required=True,
                        help="CRE file")
    parser.add_argument('-l', '--lessone', dest="lessone", type=int, action='store', default=0,
                        help="Cell type to leave out")
    parser.add_argument('-o', '--output', dest="output", type=str, action='store', default='./out',
                        help="Output prefix")
    parser.add_argument('-i', '--iterations', dest="iterations", type=int, action='store', default=100,
                        help="Refinement iterations")
    parser.add_argument('-t', '--threads', dest="threads", type=int, action='store', default=1,
                        help="Number of threads to use")
    parser.add_argument('--initialization-dist', dest="initdist", type=int, action='store', default=1000,
                        help="Beta initialization distance cutoff")
    parser.add_argument('--promoter-dist', dest="promoter", type=int, action='store',
                        help="If specified, learn betas for promoters up to promoter distance cutoff")
    parser.add_argument('--proximal-dist', dest='proximal', type=int, action='store',
                        help="If specified, learn separate betas for proximal vs. distal CREs")
    parser.add_argument('--cre-dist', dest="distal", type=int, action='store',
                        help="CRE distance cutoff")
    prom = parser.add_mutually_exclusive_group()
    prom.add_argument('--cre-exclude-promoter', dest="crepromdist", type=int, action='store',
                      help="Distance to exclude of promoter for CREs")
    prom.add_argument('--tss-dist', dest="tssdist", type=int, action='store',
                      help="If specified, window around TSS to include as cCRE")
    parser.add_argument('--correlation', dest="correlation", type=float, action='store', default=0.0,
                        help="Initial correlation cutoff")
    parser.add_argument('--trainstats', dest="trainstats", action='store_true',
                        help="Output training statistics")
    parser.add_argument('--features', dest="outputfeatures", action='store_true',
                        help="Output feature vectors")
    parser.add_argument('--log', dest="log", action='store_true',
                        help="Log2-transform predictors")
    parser.add_argument('--pca', dest="pca", type=float, action='store',
                        help="Convert state ratios into PCAs")
    parser.add_argument('--multi-refinement', dest="multirefine", action='store_true',
                        help="Change multiple cCREs' inclusion status per gene per round")
    parser.add_argument('--single-state', dest="singlestate", action='store_true',
                        help="Use a single state for each cCRE")
    parser.add_argument('--intercept', dest="intercept", action='store_true',
                        help="Include an intercept term in the regression")
    parser.add_argument('--remove-zero-state', dest="nozero", action='store_true',
                        help="Do not use the zero state for prediction calculations")
    parser.add_argument('--max-CREs', dest="maxcres", action='store', type=int, default=0,
                        help="Maximum number of CREs allowed to be selected per TSS at a time (0 is no max)")
    parser.add_argument('--skip-training', dest="skiptraining", action='store_true',
                        help="Skip CRE-TSS pairining refinement")
    parser.add_argument('--shuffle-states', dest="shufflestates", action='store_true',
                        help="Shuffle the state proportions of each CRE as a negative control")
    parser.add_argument('--shuffle-rna', dest="shufflerna", action='store_true',
                        help="Shuffle the rna of the non-left out celltypes for a negative control")
    parser.add_argument('--shuffle-all-rna', dest="shuffleallrna", action='store_true',
                        help="Shuffle the rna of the non-left out celltypes at each position for a negative control")
    parser.add_argument('-e', '--eRP', dest="eRP", action='store', type=str,
                        help="A previously generated eRP TSS-cCRE pair file. Passing this will ignore initial TSS-CRE pair selection")
    parser.add_argument('--seed', dest="seed", action='store', type=int,
                        help="Random number generator state seed")
    parser.add_argument('-v', '--verbose', dest="verbose", action='store', type=int, default=2,
                        help="Verbosity level")
    return parser


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
        self.load_state()

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

    def run(self, out_prefix, initialization_dist=1000, promoter_dist=None, proximal_dist=None, distal_dist=None, crepromdist=None, corr_cutoff=0.2, refining_iter=100, lessone=0, threads=1, trainstats=False, eRP=None, pca=None, log=False, tss_dist=None, multirefine=False, singlestate=False, intercept=False, nozero=True, maxcres=None, skip_training=False, shufflestates=False, shufflerna=False, shuffleallrna=False, outputfeatures=False, seed=None):
        self.lessone = lessone # Sample to hold out and predict at end
        self.lmask = numpy.r_[numpy.arange(lessone), numpy.arange((lessone + 1), self.cellN)] # Indices excluding lessone celltype
        self.shufflestates = bool(shufflestates)
        self.shufflerna = bool(shufflerna)
        self.shuffleallrna = bool(shuffleallrna)
        self.initialization_dist = int(initialization_dist) # Distance from TSSs to initialize beta values
        if distal_dist is not None:
            self.distal_dist = int(distal_dist) # Max distance from TSSs for distal cCRE state assessment, exluding proximal, if used
            self.skip_cres = False
            if crepromdist is not None:
                self.skip_promoter_dist = int(crepromdist)
            else:
                self.skip_promoter_dist = None
        else:
            self.skip_promoter_dist = None
            self.skip_cres = True
            if promoter_dist is None:
                self.logger.error("You must either speficy a distance for CREs or promoters")
                sys.exit(0)
        if promoter_dist is not None:
            self.promoter_dist = int(promoter_dist) # Max distance from TSSs for promoter state assessment, if used
            self.skip_promoter = False
        else:
            self.skip_promoter = True
        if not self.skip_cres and proximal_dist is not None:
            self.proximal_dist = int(proximal_dist) # Max distance from TSSs for proximal cCRE state assessment, if used
            self.skip_proximal = False
            if self.proximal_dist >= self.distal_dist:
                self.logger.error("Proximal distance must be smaller than distal distance")
                sys.exit(0)
        else:
            self.skip_proximal = True
        self.corr_cutoff = float(corr_cutoff) # Correlation cutoff for initial filtering of cCREs
        self.refining_iter = int(refining_iter) # Number of refining iterations to perform
        self.threads = max(1, int(threads)) # Number of threads to use
        self.trainstats = bool(trainstats) # Indicator whether to retain training statistics
        self.log = bool(log) # Indicator whether to use log-transformed state proportions
        if self.skip_promoter_dist is not None and tss_dist is not None:
            self.TSSdist = int(tss_dist) # Window around TSS to include as a cCRE, if inludeTSS is True
            self.includeTSS = True # Indicator whether to include TSS as a cCRE if not otherwise covered by one
        else:
            self.includeTSS = False
        self.multirefine = bool(multirefine) # Indicator whether to remove multiple cCREs with each refinement
        self.singlestate = bool(singlestate) # Indicator whether to use a single state instead of proportion of states for each cCRE
        self.intercept = bool(intercept)
        self.nozero = bool(nozero)
        self.maxcres = int(maxcres)
        self.skip_training = bool(skip_training)
        self.output_features = bool(outputfeatures)
        self.out_prefix = str(out_prefix)
        self.seed = seed
        self.rng = numpy.random.RandomState(seed=seed)
        self.pca = None

        if self.shufflerna or self.shuffleallrna:
            self.shuffle_rna()
        if self.nozero:
            self.stateN1 = self.stateN - 1
        else:
            self.stateN1 = self.stateN
        if self.skip_promoter:
            self.promoter_beta_indices = []
            self.XPrs = numpy.zeros((self.tssN, self.cellN - 1, 0), dtype=numpy.float32)
            self.XPrs_l = numpy.zeros((self.tssN, 0), dtype=numpy.float32)
        else:
            self.promoter_beta_indices = numpy.arange(self.stateN1)
        if self.skip_proximal:
            self.proximal_beta_indices = []
        else:
            self.proximal_beta_indices = (numpy.arange(self.stateN1)
                                          + len(self.promoter_beta_indices))
        if self.skip_cres:
            self.distal_beta_indices = []
        else:
            self.distal_beta_indices = (numpy.arange(self.stateN1)
                                        + len(self.promoter_beta_indices)
                                        + len(self.proximal_beta_indices))

        self.logger.info("Left out celltype: %s" % (self.celltypes[self.lessone]))
        if not self.skip_promoter:
            self.logger.info("Promoter winow: %i" % self.promoter_dist)
        if not self.skip_proximal:
            self.logger.info("CRE windows: %i proximal, %i distal" % (self.proximal_dist, self.distal_dist))
        elif not self.skip_cres:
            self.logger.info("CRE window: %i" % self.distal_dist)
            if self.skip_promoter_dist is not None:
                self.logger.info("Exluding CREs in promoter distance: %i" % self.skip_promoter_dist)
            elif self.includeTSS:
                self.logger.info("TSS window to add to CREs: %i" % self.TSSdist)
        if self.log:
            self.logger.info("Log-transforming state proportions")
        if not self.skip_cres:
            self.logger.info("Beta initialization distance: %i" % self.initialization_dist)
            self.logger.info("Initial CRE correlation cutoff: %f" % self.corr_cutoff)
            self.logger.info("Number of refinement iterations: %i" % self.refining_iter)
            if self.multirefine:
                self.logger.info("Adding/removing multiple CREs per gene per refinement round")
            else:
                self.logger.info("Adding/removing one CRE per gene per refinement round")
            if self.singlestate:
                self.logger.info("Using best-perfoming state position for each CRE")
            else:
                self.logger.info("Using state proportions for each CRE")
        if self.intercept:
            self.logger.info("Including intercept term in prediction regression")
        if self.nozero:
            self.logger.info("Ignoring state zero in analysis")
        if not self.skip_cres and self.maxcres is not None:
            self.logger.info("Allowing up to %i CREs per TSS to be selected" % self.maxcres)

        self.norm_rna = numpy.copy(self.rna['rna'][:, self.lmask])
        self.norm_rna -= numpy.mean(self.norm_rna, axis=1).reshape(-1, 1)
        self.norm_rna /= ((numpy.std(self.norm_rna, axis=1, ddof=1) + 1e-5)
                          * (self.cellN - 1) ** 0.5).reshape(-1, 1)
        if not self.skip_cres:
            if self.includeTSS:
                self.append_TSSs_to_CREs()
        if not self.skip_promoter:
            self.assign_promoter_states()
        if not self.skip_cres:
            self.assign_CRE_states()

        if eRP is None:
            if not self.skip_cres:
                self.find_initial_betas()
                self.find_TSS_CRE_pairs()
        else:
            self.reconstitute_pairs(eRP)
        if not self.skip_cres and not self.skip_training:
            self.refine_pairs()
            if pca is not None:
                if pca.is_integer():
                    self.pca = int(pca)
                else:
                    self.pca = float(pca)
                self.singlestate = False
                self.refine_pairs()
        self.predict_lessone()

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

    def append_TSSs_to_CREs(self):
        """Find where the region around the TSS is not covered by a cCRE and add it to the cCRE list"""
        if self.verbose >= 2:
            print("\r%s\rAdding TSSs to cCRE data" % (' ' * 80), end='', file=sys.stderr)
        data = []
        for i, chrom in enumerate(self.chroms):
            s = self.rna_indices[i]
            e = self.rna_indices[i + 1]
            s1 = self.cre_indices[i]
            e1 = self.cre_indices[i + 1]
            TSS_st = self.rna['TSS'][s:e] - self.TSSdist * numpy.logical_not(self.rna['strand'][s:e])
            TSS_ed = self.rna['TSS'][s:e] + self.TSSdist * self.rna['strand'][s:e]
            starts = numpy.searchsorted(self.cre['end'][s:e], TSS_st, side='right')
            stops = numpy.searchsorted(self.cre['start'][s:e], TSS_ed, side='left')
            for j in range(starts.shape[0]):
                overlap = stops[j] - starts[j]
                if overlap == 0:
                    data.append((chrom, TSS_st[j], TSS_ed[j]))
        cre = numpy.hstack((self.cre, numpy.array(data, dtype=self.cre.dtype)))
        cre = cre[numpy.lexsort((cre['start'], cre['chr']))]
        self.cre = cre
        self.cre_indices = numpy.r_[0, numpy.where(self.cre['chr'][1:] != self.cre['chr'][:-1])[0] + 1,
                                    self.cre.shape[0]]
        if self.verbose >= 2:
            print("\r%s\r" % (' ' * 80), end='', file=sys.stderr)

    def shuffle_rna(self):
        lmask = numpy.copy(self.lmask)
        self.rng.shuffle(lmask)
        if not self.shuffleallrna:
            self.rna['rna'][:, self.lmask] = self.rna['rna'][:, lmask]
        else:
            for i in range(self.rna.shape[0]):
                self.rna['rna'][i, self.lmask] = self.rna['rna'][i, lmask]
                self.rng.shuffle(lmask)

    def assign_promoter_states(self):
        """Find the proportion of states in each promoter window"""
        if self.verbose >= 2:
            print("\r%s\rAssign states to promoters" % (' ' * 80), end='', file=sys.stderr)
        # Find ranges of states for each CRE
        Pranges = numpy.zeros((self.rna.shape[0], 2), dtype=numpy.int32)
        for i in range(self.rna_indices.shape[0] - 1):
            s = self.rna_indices[i]
            e = self.rna_indices[i + 1]
            if e - s == 0:
                continue
            s1 = self.state_indices[i]
            e1 = self.state_indices[i + 1]
            if e1 - s1 == 0:
                continue
            starts = numpy.searchsorted(self.state['end'][s1:e1],
                                        self.rna['TSS'][s:e] - self.promoter_dist
                                        ,#* numpy.logical_not(self.rna['strand'][s:e]),
                                        side='right') + s1
            stops = numpy.searchsorted(self.state['start'][s1:e1],
                                       self.rna['TSS'][s:e] + self.promoter_dist
                                       ,#* self.rna['strand'][s:e],
                                       side='left') + s1
            Pranges[s:e, 0] = starts
            Pranges[s:e, 1] = stops
        self.Pranges = Pranges
        # Divide list across multiple processes
        tss_queue = multiprocessing.JoinableQueue()
        results_queue = multiprocessing.JoinableQueue()
        processes = []
        for i in range(self.threads):
            processes.append(multiprocessing.Process(
                target=self._assign_promoter_state, args=(tss_queue, results_queue, self.Pranges,
                                                          self.promoter_dist,
                                                          self.rng.randint(99999), True)))
            processes[-1].daemon = True
            processes[-1].start()
        step = int(self.rna_indices[-1] / max(self.threads, 1) / 4.)
        for i in range(self.rna_indices.shape[0] - 1):
            for j in range(self.rna_indices[i], self.rna_indices[i + 1], step):
                stop = min(self.rna_indices[i + 1], j + step)
                tss_queue.put((j, stop))
        for i in range(self.threads):
            tss_queue.put(None)
        # Even though there may be multiple reps for a celltype, we only find the average state proportion across reps
        Pstates = numpy.zeros((self.rna.shape[0], self.cellN, self.stateN), dtype=numpy.float32)
        finished = 0
        while finished < self.threads:
            results = results_queue.get(True)
            if results is None:
                finished += 1
                continue
            start, stop = results[:2]
            Pstates[start:stop, :, :] = results[2]
        if self.nozero:
            Pstates = Pstates[:, :, 1:]
        if self.log:
            Pstates = numpy.log2(Pstates + 1e-3)
        self.Pstates = Pstates
        print("Promoters", numpy.amin(numpy.sum(self.Pstates, axis=2)), numpy.amax(numpy.sum(self.Pstates, axis=2)))
        self.XPrs = self.Pstates[:, self.lmask, :]
        self.XPrs_l = self.Pstates[:, self.lessone, :]
        if self.verbose >= 2:
            print("\r%s\r" % (' ' * 80), end='', file=sys.stderr)

    def assign_CRE_states(self, singlestate=False):
        """Find the proportion of states in each cCRE"""
        if self.verbose >= 2:
            print("\r%s\rAssign states to CREs" % (' ' * 80), end='', file=sys.stderr)
        # Find ranges of states for each CRE
        Cranges = numpy.zeros((self.cre.shape[0], 2), dtype=numpy.int32)
        for i in range(self.cre_indices.shape[0] - 1):
            s = self.cre_indices[i]
            e = self.cre_indices[i + 1]
            if e - s == 0:
                continue
            s1 = self.state_indices[i]
            e1 = self.state_indices[i + 1]
            if e1 - s1 == 0:
                continue
            starts = numpy.searchsorted(self.state['end'][s1:e1],
                                        self.cre['start'][s:e], side='right') + s1
            stops = numpy.searchsorted(self.state['start'][s1:e1],
                                       self.cre['end'][s:e], side='left') + s1
            Cranges[s:e, 0] = starts
            Cranges[s:e, 1] = stops
        self.Cranges = Cranges
        if not singlestate:
            # Divide list across multiple processes
            cre_queue = multiprocessing.JoinableQueue()
            results_queue = multiprocessing.JoinableQueue()
            processes = []
            for i in range(self.threads):
                processes.append(multiprocessing.Process(
                    target=self._assign_CRE_state, args=(cre_queue, results_queue,
                                                         self.rng.randint(99999))))
                processes[-1].daemon = True
                processes[-1].start()
            step = int(self.cre_indices[-1] / max(self.threads, 1) / 4.)
            for i in range(self.cre_indices.shape[0] - 1):
                for j in range(self.cre_indices[i], self.cre_indices[i + 1], step):
                    stop = min(self.cre_indices[i + 1], j + step)
                    cre_queue.put((j, stop))
            for i in range(self.threads):
                cre_queue.put(None)
            # Even though there may be multiple reps for a celltype, we only find the average state proportion across reps
            Cstates = numpy.zeros((self.cre.shape[0], self.cellN, self.stateN), dtype=numpy.float32)
            finished = 0
            while finished < self.threads:
                results = results_queue.get(True)
                if results is None:
                    finished += 1
                    continue
                start, stop = results[:2]
                Cstates[start:stop, :, :] = results[2]
            self.Cstates = Cstates
            print("CREs", numpy.amin(numpy.sum(self.Cstates, axis=2)), numpy.amax(numpy.sum(self.Cstates, axis=2)))
        if self.verbose >= 2:
            print("\r%s\r" % (' ' * 80), end='', file=sys.stderr)

    def find_initial_betas(self):
        """Using the window specified by 'initialization_dist' around TSSs, find initial beta values"""
        if self.verbose >= 2:
            print("\r%s\rFinding initial betas" % (' ' * 80), end='', file=sys.stderr)
        # Find ranges of states for each CRE
        Tranges = numpy.zeros((self.rna.shape[0], 2), dtype=numpy.int32)
        for i in range(self.rna_indices.shape[0] - 1):
            s = self.rna_indices[i]
            e = self.rna_indices[i + 1]
            if e - s == 0:
                continue
            s1 = self.state_indices[i]
            e1 = self.state_indices[i + 1]
            if e1 - s1 == 0:
                continue
            starts = numpy.searchsorted(self.state['end'][s1:e1],
                                        self.rna['TSS'][s:e] - self.initialization_dist,
                                        #* numpy.logical_not(self.rna['strand'][s:e]),
                                        side='right') + s1
            stops = numpy.searchsorted(self.state['start'][s1:e1],
                                       self.rna['TSS'][s:e] + self.initialization_dist,
                                       #* self.rna['strand'][s:e],
                                       side='left') + s1
            Tranges[s:e, 0] = starts
            Tranges[s:e, 1] = stops
        # Divide list across multiple processes
        tss_queue = multiprocessing.JoinableQueue()
        results_queue = multiprocessing.JoinableQueue()
        processes = []
        for i in range(self.threads):
            processes.append(multiprocessing.Process(
                target=self._assign_promoter_state, args=(tss_queue, results_queue, Tranges,
                                                          self.initialization_dist,
                                                          self.rng.randint(99999), True)))
            processes[-1].daemon = True
            processes[-1].start()
        step = int(self.rna_indices[-1] / max(self.threads, 1) / 4.)
        for i in range(self.rna_indices.shape[0] - 1):
            for j in range(self.rna_indices[i], self.rna_indices[i + 1], step):
                stop = min(self.rna_indices[i + 1], j + step)
                tss_queue.put((j, stop))
        for i in range(self.threads):
            tss_queue.put(None)
        # Even though there may be multiple reps for a celltype, we only find the average state proportion across reps
        Tstates = numpy.zeros((self.rna.shape[0], self.cellN, self.stateN), dtype=numpy.float32)
        finished = 0
        while finished < self.threads:
            results = results_queue.get(True)
            if results is None:
                finished += 1
                continue
            start, stop = results[:2]
            Tstates[start:stop, :, :] = results[2]
        Tstates2 = numpy.copy(Tstates)
        Tstates = Tstates[:, self.lmask, :]
        Tstates /= numpy.sum(Tstates, axis=2, keepdims=True)
        if self.nozero:
            Tstates = Tstates[:, :, 1:]
        if self.log:
            Tstates = numpy.log2(Tstates + 1e-3)
        if self.intercept:
            Tstates = numpy.dstack((Tstates, numpy.ones((Tstates.shape[0], Tstates.shape[1], 1),
                                                        dtype=numpy.float32)))
        betas = numpy.linalg.lstsq(Tstates.reshape(-1, Tstates.shape[2], order='C'),
                                   self.rna['rna'][:, self.lmask].reshape(-1, order='C'),
                                   rcond=None)[0]
        self.initial_betas = betas
        if self.verbose >= 2:
            print("\r%s\r" % (' ' * 80), end='', file=sys.stderr)

    def find_TSS_CRE_pairs(self):
        """Find all cCREs in distal_dist window and find correlation with gene expression 
           for each one, filtering out cCREs for each gene bleow threshold"""
        if self.verbose >= 2:
            print("\r%s\rFinding TSS-cCRE pairs" % (' ' * 80), end='', file=sys.stderr)
        TSS_ranges = self.find_TSS_ranges()
        if self.skip_promoter_dist is not None:
            pair_indices = numpy.r_[0, numpy.cumsum(TSS_ranges[:, 1] - TSS_ranges[:, 0]
                                                    + TSS_ranges[:, 3] - TSS_ranges[:, 2])]
        else:
            pair_indices = numpy.r_[0, numpy.cumsum(TSS_ranges[:, 1] - TSS_ranges[:, 0])]
        # If using proportion of all states covered by a cCRE, normalize predicted values for easy correlation
        if not self.singlestate:
            func = self._find_correlations
        else:
            # If using single state for cCRE, we will need to determine the best correlation for each cCRE
            func = self._find_ss_correlations
        pair_queue = multiprocessing.JoinableQueue()
        results_queue = multiprocessing.JoinableQueue()
        processes = []
        for i in range(self.threads):
            processes.append(multiprocessing.Process(
                target=func, args=(pair_queue, results_queue)))
            processes[-1].daemon = True
            processes[-1].start()
        step = 50
        if self.singlestate:
            for i in range(self.chroms.shape[0]):
                for j in range(self.rna_indices[i], self.rna_indices[i + 1], step):
                    end = min(j + step, self.rna_indices[i + 1])
                    cres = []
                    for k in range(TSS_ranges[j, 0], TSS_ranges[end - 1, -1]):
                        cres.append(self.state['state'][self.Cranges[k, 0]:self.Cranges[k, 1], :])
                    pair_queue.put((j, end, TSS_ranges[j:end, :], cres))
        else:
            for i in range(self.chroms.shape[0]):
                for j in range(self.rna_indices[i], self.rna_indices[i + 1], step):
                    end = min(j + step, self.rna_indices[i + 1])
                    pair_queue.put((j, end, TSS_ranges[j:end, :]))
        for i in range(self.threads):
            pair_queue.put(None)
        pairs = numpy.zeros((pair_indices[-1], 3), dtype=numpy.int32)
        valid = numpy.zeros(pair_indices[-1], dtype=numpy.bool)
        if self.singlestate:
            pair_states = numpy.zeros((pair_indices[-1], self.cellN, self.stateN),
                                      dtype=numpy.float32)
        finished = 0
        while finished < self.threads:
            results = results_queue.get(True)
            if results is None:
                finished += 1
                continue
            for i in range(len(results)):
                index, corrs = results[i][:2]
                s = pair_indices[index]
                e = pair_indices[index + 1]
                pairs[s:e, 0] = index
                if self.skip_promoter_dist is not None:
                    pairs[s:e, 1] = numpy.r_[numpy.arange(TSS_ranges[index, 0], TSS_ranges[index, 1]),
                                             numpy.arange(TSS_ranges[index, 2], TSS_ranges[index, 3])]
                else:
                    pairs[s:e, 1] = numpy.arange(TSS_ranges[index, 0], TSS_ranges[index, 1])
                valid[s:e] = corrs >= self.corr_cutoff
                if self.singlestate:
                    pair_states[s:e, :, :] = results[i][2]
        self.pairs = pairs[numpy.where(valid)[0], :]
        #self.pairs = pairs
        if not self.skip_proximal:
            self.pairs[:, 2] = (numpy.minimum(numpy.abs(self.rna['TSS'][self.pairs[:, 0]]
                                                        - self.cre['start'][self.pairs[:, 1]]),
                                              numpy.abs(self.rna['TSS'][self.pairs[:, 0]]
                                                        - self.cre['end'][self.pairs[:, 1]]))
                                <= self.proximal_dist)
        if self.singlestate:
            self.pair_states = pair_states
        self.TSS_indices = numpy.r_[0, numpy.cumsum(numpy.bincount(self.pairs[:, 0],
                                                                   minlength=self.tssN))]
        self.selected = numpy.ones(self.pairs.shape[0], dtype=numpy.bool)
        if self.maxcres > 0:
            where = numpy.where(self.TSS_indices[1:] - self.TSS_indices[:-1] > self.maxcres)[0]
            for i in where:
                s, e = self.TSS_indices[i:(i + 2)]
                self.selected[self.rng.choice(numpy.arange(s, e), e - s - self.maxcres,
                    replace=False)] = False
        self.proximal = numpy.where(self.pairs[:, 2])[0]
        self.distal = numpy.where(numpy.logical_not(self.pairs[:, 2]))[0]
        if self.verbose >= 2:
            print("\r%s\r" % (' ' * 80), end='', file=sys.stderr)
        kept = numpy.sum(valid)
        temp = numpy.bincount(self.pairs[:, 0], weights=self.selected, minlength=self.tssN)
        self.logger.info("Retained %i of %i TSS-CRE pairs (%0.1f%%), %i - %i CREs/TSS (median %0.1f)"
                         % (self.selected.shape[0], valid.shape[0],
                            100. * self.selected.shape[0] / valid.shape[0], numpy.amin(temp),
                            numpy.amax(temp), numpy.median(temp)))
        self.logger.info("Unique CREs in pairings: {}".format(numpy.unique(self.pairs[:, 1]).shape[0]))

    def find_PCAs(self):
        XPs, XDs = self.find_Xs()
        if XPs.shape[2] > 0:
            if self.XPrs.shape[2] > 0:
                print(XPs.shape, XDs.shape, self.XPrs.shape)
                X = numpy.vstack((XPs, XDs, self.XPrs))
            else:
                X = numpy.vstack((XPs, XDs))
        else:
            if self.XPrs.shape[2] > 0:
                print(numpy.amin(numpy.sum(XDs, axis=2)), numpy.amax(numpy.sum(XDs, axis=2)),
                      numpy.amin(numpy.sum(self.XPrs, axis=2)),
                      numpy.amax(numpy.sum(self.XPrs, axis=2)))
                print(XDs.shape, self.XPrs.shape)
                X = numpy.vstack((XDs, self.XPrs))
            else:
                X = numpy.vstack((XDs))
        """
        XPs, XDs = self.find_Xs()
        if XPs.shape[2] > 0:
            X = numpy.vstack((XPs, XDs))
        else:
            X = numpy.vstack((XDs))
        #X = numpy.copy(self.Cstates)
        starts = (self.state['start'][self.state_indices[:-1]] // 1000) * 1000
        stops = ((self.state['end'][self.state_indices[1:] - 1] - 1) // 1000 + 1) * 1000
        v_indices = numpy.r_[0, numpy.cumsum((stops - starts) // 1000)]
        valid = numpy.zeros(v_indices[-1], dtype=numpy.bool)
        for i in range(v_indices.shape[0] - 1):
            s = v_indices[i]
            e = v_indices[i + 1]
            counts = numpy.bincount((self.state['start'][self.state_indices[i]:
                                     self.state_indices[i + 1]] - starts[i]) // 1000,
                                    minlength=(e - s))
            valid[s:e] = counts > 0
        bins = numpy.zeros(numpy.sum(valid), dtype=numpy.dtype([('start', numpy.int32),
                                                                ('end', numpy.int32)]))
        X_indices = numpy.cumsum(numpy.array([0] + [numpy.sum(valid[v_indices[i]:
                                                                    v_indices[i + 1]])
                                                    for i in range(v_indices.shape[0] - 1)]))
        for i in range(v_indices.shape[0] - 1):
            s = X_indices[i]
            e = X_indices[i + 1]
            bins['start'][s:e] = starts[i] + numpy.where(valid[v_indices[i]:
                                                               v_indices[i + 1]])[0] * 1000
        bins['end'] = bins['start'] + 1000
        cre = self.cre
        cre_indices = self.cre_indices
        Cstates = self.Cstates
        Cranges = self.Cranges
        self.cre = bins
        self.cre_indices = X_indices
        self.assign_CRE_states()
        X = self.Cstates
        self.cre = cre
        self.cre_indices = cre_indices
        self.Cstates = Cstates
        self.Cranges = Cranges
        X = X[:, self.lmask, :]
        """
        X /= numpy.sum(X, axis=2, keepdims=True)
        X = X.reshape(-1, self.stateN)
        if isinstance(self.pca, float):
            PCA = sklearn.decomposition.PCA(whiten=True, svd_solver='full')
            PCA.fit(X)
            n = max(1, numpy.searchsorted(numpy.cumsum(PCA.explained_variance_ratio_), self.pca))
            PCA = sklearn.decomposition.PCA(n, whiten=True, svd_solver='full')
        else:
            PCA = sklearn.decomposition.PCA(self.pca, whiten=True, svd_solver='full')
        PCA.fit(X)
        Cstates = numpy.ascontiguousarray(PCA.transform(self.Cstates.reshape(-1, self.stateN)))
        self.pca_Cstates = Cstates.reshape(self.cre.shape[0], self.cellN, -1)
        Pstates = numpy.ascontiguousarray(PCA.transform(self.Pstates.reshape(-1, self.stateN)))
        self.pca_Pstates = Pstates.reshape(self.rna.shape[0], self.cellN, -1)
        self.pca_stateN = PCA.n_components_
        self.pca_stateN1 = self.pca_stateN
        if self.skip_promoter:
            self.promoter_beta_indices = []
            self.pca_XPrs = numpy.zeros((self.tssN, self.cellN - 1, 0), dtype=numpy.float32)
            self.pca_XPrs_l = numpy.zeros((self.tssN, 0), dtype=numpy.float32)
        else:
            self.pca_XPrs = self.pca_Pstates[:, self.lmask, :]
            self.pca_XPrs_l = self.pca_Pstates[:, self.lessone, :]
            self.promoter_beta_indices = numpy.arange(self.pca_stateN1)
        if self.skip_proximal:
            self.proximal_beta_indices = []
        else:
            self.proximal_beta_indices = (numpy.arange(self.pca_stateN1)
                                          + len(self.promoter_beta_indices))
        if self.skip_cres:
            self.distal_beta_indices = []
        else:
            self.distal_beta_indices = (numpy.arange(self.pca_stateN1)
                                        + len(self.promoter_beta_indices)
                                        + len(self.proximal_beta_indices))

    def refine_pairs(self):
        if self.verbose >= 2:
            print("\r%s\rFinding TSS-cCRE pairs" % (' ' * 80), end='', file=sys.stderr)
        if self.trainstats:
            output = open('%s_training_statistics.txt' % self.out_prefix, 'w')
            print("Iteration\t#Pairs\t%Kept\tR2adj\tMSE\tOutR2adj\tOutMSE", file=output)
            output.close()
        if self.pca is not None:
            pca = True
            self.find_PCAs()
            XPrs = self.pca_XPrs
            XPrs_l = self.pca_XPrs_l
        else:
            pca = False
            XPrs = self.XPrs
            XPrs_l = self.XPrs_l
        XPs, XDs = self.find_Xs(pca=pca)
        XPs_l, XDs_l = self.find_Xs(lessone=True, pca=pca)
        if self.intercept:
            X = numpy.dstack((self.XPrs, XPs, XDs,
                              numpy.ones((self.tssN, self.cellN - 1, 1),
                                         dtype=numpy.float32)))
            X_l = numpy.hstack((self.XPrs_l, XPs_l, XDs_l,
                                numpy.ones((self.tssN, 1), dtype=numpy.float32)))
        else:
            X = numpy.dstack((XPrs, XPs, XDs))
            X_l = numpy.hstack((XPrs_l, XPs_l, XDs_l))
        Y = self.rna['rna'][:, self.lmask]
        Y_l = self.rna['rna'][:, self.lessone]

        norm_Y = Y - numpy.mean(Y)
        var_Y = numpy.mean(norm_Y ** 2)
        norm_Y_l = Y_l - numpy.mean(Y_l)
        var_Y_l = numpy.mean(norm_Y_l ** 2)

        n = Y.shape[0] * Y.shape[1]
        n_l = Y.shape[0]
        p = X.shape[2]
        best_betas = numpy.linalg.lstsq(X.reshape(n, -1, order='C'),
                                        Y.reshape(-1, order='C'), rcond=None)[0]
        y = numpy.sum(X * best_betas.reshape(1, 1, -1), axis=2)
        y -= numpy.mean(y)
        var_y = numpy.mean(y ** 2)

        R2 = numpy.mean(norm_Y * y) ** 2 / (var_Y * var_y)
        best_R2adj = 1 - (1 - R2) * (n - 1) / (n - p - 1)
        best_selected = numpy.copy(self.selected)
        changed = numpy.ones(self.tssN, dtype=numpy.bool)
        changed_sum = self.tssN
        training_statistics = []

        y_l = numpy.sum(X_l * best_betas.reshape(1, -1), axis=1)
        SSres_l = numpy.sum((Y_l - y_l) ** 2)
        y_l -= numpy.mean(y_l)
        var_y_l = numpy.mean(y_l ** 2)
        R2_l = numpy.mean(norm_Y_l * y_l) ** 2.0 / (var_Y_l * var_y_l)
        R2adj_l = 1 - (1 - R2_l) * (n_l - 1) / (n_l - p - 1)
        best_R2adj_l = R2adj_l

        for iteration in range(self.refining_iter):
            if self.verbose >= 2:
                print("\r%s\rFinding TSS-cCRE pairs - Iteration %i of %i"
                      % (' ' * 80, iteration, self.refining_iter), end='', file=sys.stderr)
            beta_queue = multiprocessing.JoinableQueue()
            results_queue = multiprocessing.JoinableQueue()
            processes = []
            for i in range(self.threads):
                processes.append(multiprocessing.Process(
                    target=self._find_betas, args=(beta_queue, results_queue)))
                processes[-1].daemon = True
                processes[-1].start()
            for i, j in enumerate(self.lmask):
                mask = numpy.r_[numpy.arange(i), numpy.arange(i + 1, self.lmask.shape[0])]
                beta_queue.put((i, j, mask, X, Y))
            for i in range(self.threads):
                beta_queue.put(None)
            betas = numpy.zeros((self.cellN - 1, p), dtype=numpy.float32)
            pred_exp = numpy.zeros((self.tssN, self.cellN - 1), dtype=numpy.float32)
            finished = 0
            while finished < self.threads:
                results = results_queue.get(True)
                if results is None:
                    finished += 1
                    continue
                betas[results[0], :] = results[1]
                pred_exp[:, results[0]] = results[2]
            SSres = numpy.sum((Y - pred_exp) ** 2, axis=1)
            if iteration == 0:
                best_betas = numpy.copy(betas)
                training_statistics.append((0, self.selected.shape[0], 1.0, best_R2adj,
                                            numpy.sum(SSres), R2adj_l, SSres_l))
                if self.trainstats:
                    output = open('%s_training_statistics.txt' % self.out_prefix, 'a')
                    print('\t'.join([str(x) for x in training_statistics[-1]]), file=output)
                    output.close()
                if self.verbose >= 2:
                    print("\r%s\rIteration: %i, selected pairs: %0.2f%% (%i of %i), # Changed: %i, R2adj: %0.2f, MSE: %0.4e, OutR2adj: %0.2f, OutMSE: %0.4e"
                          % (' ' * 80, iteration, 100. * numpy.sum(self.selected) / self.selected.shape[0],
                             numpy.sum(self.selected), self.selected.shape[0], 0, 100 * best_R2adj,
                             numpy.sum(SSres)/n, 100*R2adj_l, SSres_l/n_l),
                          file=sys.stderr)

            pair_queue = multiprocessing.JoinableQueue()
            results_queue = multiprocessing.JoinableQueue()
            processes = []
            if self.intercept:
                intercept = betas[:, -1]
            else:
                intercept = numpy.zeros(self.cellN - 1, dtype=numpy.float32)
            for i in range(self.threads):
                processes.append(multiprocessing.Process(
                    target=self._refine_pair_CREs, args=(pair_queue, results_queue,
                                                         betas[:, self.promoter_beta_indices],
                                                         betas[:, self.proximal_beta_indices],
                                                         betas[:, self.distal_beta_indices],
                                                         intercept, iteration,
                                                         self.rng.randint(99999)
                                                         )))
                processes[-1].daemon = True
                processes[-1].start()
            for i in range(self.tssN):
                if self.TSS_indices[i + 1] - self.TSS_indices[i] > 0:
                    pair_queue.put((i, SSres[i], Y[i, :]))
            for i in range(self.threads):
                pair_queue.put(None)
            new_selected = numpy.copy(self.selected)
            changed.fill(False)
            finished = 0
            count = 0
            disp_count = 0
            if self.verbose >= 2:
                print("\r%s\rFinding TSS-cCRE pairs - Iteration %i of %i, TSS %i of %i"
                      % (' ' * 80, iteration, self.refining_iter, count, self.tssN),
                      end='', file=sys.stderr)
            while finished < self.threads:
                results = results_queue.get(True)
                if results is None:
                    finished += 1
                    continue
                count += 1
                disp_count += 1
                if results[1]:
                    s = self.TSS_indices[results[0]]
                    e = self.TSS_indices[results[0] + 1]
                    new_selected[s:e] = results[2]
                    changed[results[0]] = True
                if disp_count >= 100 and self.verbose >= 2:
                    print("\r%s\rFinding TSS-cCRE pairs - Iteration %i of %i, TSS %i of %i"
                          % (' ' * 80, iteration, self.refining_iter, count, self.tssN),
                          end='', file=sys.stderr)
                    disp_count -= 100

            if self.verbose >= 2:
                print("\r%s\rFinding TSS-cCRE pairs - Iteration %i of %i"
                      % (' ' * 80, iteration, self.refining_iter), end='', file=sys.stderr)
            changed_sum = self.selected.shape[0] - numpy.sum(numpy.equal(new_selected,
                                                                         self.selected))
            new_sum = numpy.sum(new_selected)
            old_sum = numpy.sum(self.selected)
            if changed_sum == 0:
                break
            self.selected = new_selected
            XPs, XDs = self.find_Xs(pca=pca)
            XPs_l, XDs_l = self.find_Xs(lessone=True, pca=pca)
            if self.intercept:
                X = numpy.dstack((XPrs, XPs, XDs,
                                  numpy.ones((self.tssN, self.cellN - 1, 1),
                                             dtype=numpy.float32)))
                X_l = numpy.hstack((XPrs_l, XPs_l, XDs_l,
                                    numpy.ones((self.tssN, 1), dtype=numpy.float32)))
            else:
                X = numpy.dstack((XPrs, XPs, XDs))
                X_l = numpy.hstack((XPrs_l, XPs_l, XDs_l))
            betas = numpy.linalg.lstsq(X.reshape(n, -1, order='C'),
                                       Y.reshape(-1, order='C'), rcond=None)[0]
            y = numpy.sum(X * betas.reshape(1, 1, -1), axis=2)
            SSres = numpy.sum((y - Y) ** 2)
            y -= numpy.mean(y)
            var_y = numpy.mean(y ** 2)
            R2 = numpy.mean(y * norm_Y) ** 2 / (var_y * var_Y)
            R2adj = 1 - (1 - R2) * (n - 1) / (n - p - 1)
            percent_kept = new_sum / float(new_selected.shape[0])

            y_l = numpy.sum(X_l * betas.reshape(1, -1), axis=1)
            SSres_l = numpy.sum((Y_l - y_l) ** 2.)
            y_l -= numpy.mean(y_l)
            var_y_l = numpy.mean(y_l ** 2)
            R2_l = numpy.mean(norm_Y_l * y_l) ** 2.0 / (var_Y_l * var_y_l)
            R2adj_l = 1 - (1 - R2_l) * (n_l - 1) / (n_l - p - 1)
            
            training_statistics.append((iteration + 1, new_sum, percent_kept, R2adj, SSres, R2adj_l, SSres_l))
            if self.verbose >= 2:
                print("\r%s\rIteration: %i, retained pairs: %0.2f%% (%i of %i), # Changed: %i, R2adj: %0.2f, MSE: %0.4e, OutR2adj:%0.2f, OutMSE:%0.4e"
                      % (' ' * 80, iteration + 1, 100 * percent_kept,
                         new_sum, new_selected.shape[0], changed_sum,
                         100 * R2adj, SSres/n, R2adj_l*100, SSres_l/n_l),
                      file=sys.stderr)
            if self.trainstats:
                output = open('%s_training_statistics.txt' % self.out_prefix, 'a')
                print('\t'.join([str(x) for x in training_statistics[-1]]), file=output)
                output.close()

            if R2adj > best_R2adj:
                best_R2adj = R2adj
                best_selected = numpy.copy(self.selected)
                best_betas = betas
                best_R2adj_l = R2adj_l
            if self.pca is not None:
                self.find_PCAs()
                XPrs = self.pca_XPrs
                XPrs_l = self.pca_XPrs_l

        self.betas = best_betas
        self.R2adj = best_R2adj
        self.selected = best_selected
        if self.verbose >= 2:
            print("\r%s\r" % (' ' * 80), end='', file=sys.stderr)
        temp = numpy.bincount(self.pairs[:, 0], weights=self.selected, minlength=self.tssN)
        kept = numpy.sum(self.selected)
        self.logger.info("Selected %i of %i TSS-CRE pairs (%0.1f%%), %i - %i CREs/TSS (median %0.1f)"
                         % (kept, self.selected.shape[0],
                            100. * kept / self.selected.shape[0], numpy.amin(temp),
                            numpy.amax(temp), numpy.median(temp)))
        self.logger.info("Adjusted-R2: %0.2f  Outgroup Adjusted-R2:%0.2f"
                  % (100 * self.R2adj, best_R2adj_l*100))
        self.logger.info("Unique CREs in pairings: {}".format(numpy.unique(self.pairs[numpy.where(self.selected)[0], 1]).shape[0]))

    def find_Xs(self, lessone=False, pca=False):
        if pca:
            stateN = self.pca_stateN
            Cstates = self.pca_Cstates
        else:
            stateN = self.stateN
            Cstates = self.Cstates
        if  lessone:
            if self.skip_proximal:
                XPs = numpy.zeros((self.tssN, 0), dtype=numpy.float32)
            else:
                XPs = numpy.zeros((self.tssN, stateN),
                                  dtype=numpy.float32)
                proximal = (self.pairs[:, 2] == 1) & self.selected
            if self.skip_cres:
                XDs = numpy.zeros((self.tssN, 0), dtype=numpy.float32)
                return XPs, XDs
            else:
                XDs = numpy.zeros((self.tssN, stateN),
                                  dtype=numpy.float32)
                distal = (self.pairs[:, 2] == 0) & self.selected
            if self.singlestate:
                for i in range(self.tssN):
                    s = self.TSS_indices[i]
                    e = self.TSS_indices[i + 1]
                    if not self.skip_proximal and numpy.sum(proximal[s:e]) > 0:
                        XPs[i, :] = numpy.sum(self.pair_states[numpy.where(proximal[s:e])[0] + s,
                                                               self.lessone, :], axis=0)
                    if numpy.sum(distal[s:e]) > 0:
                        XDs[i, :] = numpy.sum(self.pair_states[numpy.where(distal[s:e])[0] + s,
                                                               self.lessone, :], axis=0)
            else:
                for i in range(self.tssN):
                    s = self.TSS_indices[i]
                    e = self.TSS_indices[i + 1]
                    if not self.skip_proximal and numpy.sum(proximal[s:e]) > 0:
                        XPs[i, :] = numpy.sum(Cstates[self.pairs[s:e, 1][proximal[s:e]],
                                                      self.lessone, :], axis=0)
                    if numpy.sum(distal[s:e]) > 0:
                        XDs[i, :] = numpy.sum(Cstates[self.pairs[s:e, 1][distal[s:e]],
                                                      self.lessone, :], axis=0)
            if not self.skip_proximal:
                XPs[:, 0] = 1 - numpy.sum(XPs[:, 1:], axis=1)
                #XPs = XPs / numpy.sum(XPs, axis=1, keepdims=True)
                if self.nozero:
                    XPs = XPs[:, 1:]
                if self.log:
                    XPs = numpy.log2(XPs + 1e-3)
            XDs[:, 0] = 1 - numpy.sum(XDs[:, 1:], axis=1)
            #XDs = XDs / numpy.sum(XDs, axis=1, keepdims=True)
            if self.nozero:
                XDs = XDs[:, 1:]
            if self.log:
                XDs = numpy.log2(XDs + 1e-3)
        else:
            if self.skip_proximal:
                XPs = numpy.zeros((self.tssN, self.cellN - 1, 0), dtype=numpy.float32)
            else:
                XPs = numpy.zeros((self.tssN, self.cellN - 1, stateN),
                                  dtype=numpy.float32)
                proximal = (self.pairs[:, 2] == 1) & self.selected
            if self.skip_cres:
                XDs = numpy.zeros((self.tssN, self.cellN - 1, 0), dtype=numpy.float32)
                return XPs, XDs
            else:
                XDs = numpy.zeros((self.tssN, self.cellN - 1, stateN),
                                  dtype=numpy.float32)
                distal = (self.pairs[:, 2] == 0) & self.selected
            if self.singlestate:
                for i in range(self.tssN):
                    s = self.TSS_indices[i]
                    e = self.TSS_indices[i + 1]
                    if not self.skip_proximal and numpy.sum(proximal[s:e]) > 0:
                        XPs[i, :, :] = numpy.sum(self.pair_states[numpy.where(proximal[s:e])[0] + s,
                                                                  :, :][:, self.lmask, :], axis=0)
                    if numpy.sum(distal[s:e]) > 0:
                        XDs[i, :, :] = numpy.sum(self.pair_states[numpy.where(distal[s:e])[0] + s,
                                                                  :, :][:, self.lmask, :], axis=0)
            else:
                for i in range(self.tssN):
                    s = self.TSS_indices[i]
                    e = self.TSS_indices[i + 1]
                    if not self.skip_proximal and numpy.sum(proximal[s:e]) > 0:
                        XPs[i, :, :] = numpy.sum(Cstates[self.pairs[s:e, 1][proximal[s:e]],
                                                         :, :][:, self.lmask, :], axis=0)
                    if numpy.sum(distal[s:e]) > 0:
                        XDs[i, :, :] = numpy.sum(Cstates[self.pairs[s:e, 1][distal[s:e]],
                                                         :, :][:, self.lmask, :], axis=0)
            if not self.skip_proximal:
                XPs[:, :, 0] = 1 - numpy.sum(XPs[:, :, 1:], axis=2)
                #XPs = XPs / numpy.sum(XPs, axis=2, keepdims=True)
                if self.nozero:
                    XPs = XPs[:, :, 1:]
                if self.log:
                    XPs = numpy.log2(XPs + 1e-3)
            XDs[:, :, 0] = 1 - numpy.sum(XDs[:, :, 1:], axis=2)
            #XDs = XDs / numpy.sum(XDs, axis=2, keepdims=True)
            if self.nozero:
                XDs = XDs[:, :, 1:]
            if self.log:
                XDs = numpy.log2(XDs + 1e-3)
        return XPs, XDs

    def reconstitute_pairs(self, fname):
        if self.verbose >= 2:
            print("\r%s\rReconstituting TSS-cCRE pairs" % (' ' * 80), end='', file=sys.stderr)
        data = []
        f = open(fname)
        f.readline()
        for line in f:
            line = line.rstrip('\n\r').split()
            data.append((self.chr2int[line[0].encode()], int(line[1]), int(line[3]), int(line[6])))
        data = numpy.array(data, dtype=numpy.int32)
        data_indices = numpy.r_[0, numpy.cumsum(numpy.bincount(data[:, 0],
                                                               minlength=self.chroms.shape[0]))]
        pairs = numpy.zeros((data.shape[0], 3), dtype=numpy.int32)
        for i in range(self.chroms.shape[0]):
            s = data_indices[i]
            e = data_indices[i + 1]
            if e == s:
                continue
            s1 = self.rna_indices[i]
            e1 = self.rna_indices[i + 1]
            pairs[s:e, 0] = numpy.searchsorted(self.rna['TSS'][s1:e1], data[s:e, 1]) + s1
            s1 = self.cre_indices[i]
            e1 = self.cre_indices[i + 1]
            pairs[s:e, 1] = numpy.searchsorted(self.cre['start'][s1:e1], data[s:e, 2]) + s1
            pairs[s:e, 2] = data[s:e, 3]
        self.pairs = pairs
        self.TSS_indices = numpy.r_[0, numpy.cumsum(numpy.bincount(self.pairs[:, 0],
                                                                   minlength=self.tssN))]
        self.selected = numpy.ones(self.pairs.shape[0], dtype=numpy.bool)
        if self.maxcres > 0:
            where = numpy.where(self.TSS_indices[1:] - self.TSS_indices[:-1] > self.maxcres)[0]
            for i in where:
                s, e = self.TSS_indices[i:(i + 2)]
                self.selected[self.rng.choice(numpy.arange(s, e), e - s - self.maxcres,
                    replace=False)] = False
        self.proximal = numpy.where(self.pairs[:, 2])[0]
        self.distal = numpy.where(numpy.logical_not(self.pairs[:, 2]))[0]
        if self.singlestate:
            self.find_initial_betas()
            TSS_ranges = self.find_TSS_ranges()
            if self.skip_promoter_dist is not None:
                pair_indices = numpy.r_[0, numpy.cumsum(TSS_ranges[:, 1] - TSS_ranges[:, 0]
                                                        + TSS_ranges[:, 3] - TSS_ranges[:, 2])]
            else:
                pair_indices = numpy.r_[0, numpy.cumsum(TSS_ranges[:, 1] - TSS_ranges[:, 0])]
            func = self._find_ss_correlations
            pair_queue = multiprocessing.JoinableQueue()
            results_queue = multiprocessing.JoinableQueue()
            processes = []
            for i in range(self.threads):
                processes.append(multiprocessing.Process(
                    target=func, args=(pair_queue, results_queue)))
                processes[-1].daemon = True
                processes[-1].start()
            step = 50
            for i in range(self.chroms.shape[0]):
                for j in range(self.rna_indices[i], self.rna_indices[i + 1], step):
                    end = min(j + step, self.rna_indices[i + 1])
                    cres = []
                    for k in range(TSS_ranges[j, 0], TSS_ranges[end - 1, -1]):
                        cres.append(self.state['state'][self.Cranges[k, 0]:self.Cranges[k, 1], :])
                    pair_queue.put((j, end, TSS_ranges[j:end, :], cres))
            for i in range(self.threads):
                pair_queue.put(None)
            pair_states = numpy.zeros((pair_indices[-1], self.cellN, self.stateN),
                                      dtype=numpy.float32)
            finished = 0
            while finished < self.threads:
                results = results_queue.get(True)
                if results is None:
                    finished += 1
                    continue
                for i in range(len(results)):
                    index, corrs, states = results[i]
                    s = pair_indices[index]
                    e = pair_indices[index + 1]
                    pair_states[s:e, :, :] = states
        if self.verbose >= 2:
            print("\r%s\r" % (' ' * 80), end='', file=sys.stderr)

    def predict_lessone(self):
        if self.pca is not None:
            self.find_PCAs()
            pca = True
            XPrs = self.pca_XPrs 
            XPrs_l = self.pca_XPrs_l
            stateN = self.pca_stateN
            Cstates = self.pca_Cstates
        else:
            pca = False
            XPrs = self.XPrs 
            XPrs_l = self.XPrs_l
            stateN = self.stateN
            Cstates = self.Cstates
        XPs, XDs = self.find_Xs(pca=pca)
        if self.output_features:
            temp = numpy.vstack((XPrs, XDs))
            numpy.save('%s_features.npy' % self.out_prefix, temp)
        XPs_l, XDs_l = self.find_Xs(lessone=True, pca=pca)
        if self.intercept:
            X = numpy.dstack((XPrs, XPs, XDs,
                              numpy.ones((self.tssN, self.cellN - 1, 1),
                                         dtype=numpy.float32)))
            X_l = numpy.hstack((XPrs_l, XPs_l, XDs_l,
                              numpy.ones((self.tssN, 1),
                                         dtype=numpy.float32)))
        else:
            X = numpy.dstack((XPrs, XPs, XDs))
            X_l = numpy.hstack((XPrs_l, XPs_l, XDs_l))
        Y = self.rna['rna'][:, self.lmask]
        Y_l = self.rna['rna'][:, self.lessone]
        betas = numpy.linalg.lstsq(X.reshape(Y.size, -1, order='C'),
                                   Y.reshape(-1, order='C'), rcond=None)[0]
        Prbetas = betas[self.promoter_beta_indices]
        Pbetas = betas[self.proximal_beta_indices]
        Dbetas = betas[self.distal_beta_indices]
        y = numpy.sum(X * betas.reshape(1, 1, -1), axis=2)
        y_l = numpy.sum(X_l * betas.reshape(1, -1), axis=1)
        MSE = numpy.mean((y - Y) ** 2)
        MSE_l = numpy.mean((y_l - Y_l) ** 2)
        Y -= numpy.mean(Y)
        Y_l -= numpy.mean(Y_l)
        var_Y = numpy.mean(Y ** 2)
        var_Y_l = numpy.mean(Y_l ** 2)
        y -= numpy.mean(y)
        norm_y_l = y_l - numpy.mean(y_l)
        var_y = numpy.mean(y ** 2)
        var_y_l = numpy.mean(norm_y_l ** 2)
        n = Y.shape[0]
        n_l = Y_l.shape[0]
        p = betas.shape[0]
        R2 = numpy.mean(y * Y) ** 2 / (var_y * var_Y)
        R2adj = 1 - (1 - R2) * (n - 1) / (n - p - 1)
        R2_l = numpy.mean(y_l * Y_l) ** 2 / (var_y_l * var_Y_l)
        R2adj_l = 1 - (1 - R2_l) * (n_l - 1) / (n_l - p - 1)
        output = open('%s_statistics.txt' % self.out_prefix, 'w')
        if self.skip_cres:
            print("R2adj\tMSE\tOut_R2adj\tOut_MSE", file=output)
            print("%f\t%f\t%f\t%f" % (R2adj*100., MSE, R2adj_l*100., MSE_l),
                  file=output)
        else:
            temp = numpy.bincount(self.pairs[:, 0], weights=self.selected, minlength=self.tssN)
            print("R2adj\tMSE\tOut_R2adj\tOut_MSE\t#Retained\t#Possible\t%Retained\tMedian CREs/TSS",
                  file=output)
            print("%f\t%f\t%f\t%f\t%i\t%i\t%f\t%i" % (R2adj*100., MSE, R2adj_l*100., MSE_l,
                                                      numpy.sum(self.selected), self.selected.shape[0],
                                                      numpy.sum(self.selected)
                                                      / float(self.selected.shape[0]),
                                                      numpy.median(temp)),
                  file=output)
        output.close()
        output = open('%s_betas.txt' %  self.out_prefix, 'w')
        header = ['state']
        if not self.skip_promoter:
            header.append('promoter_betas')
        if not self.skip_proximal:
            header.append('proximal_betas')
            header.append('distal_betas')
        elif not self.skip_cres:
            header.append('cre_betas')
        if self.intercept:
            header.append('intercept')
        print('\t'.join(header), file=output)
        offset = int(self.nozero)
        for i in range(stateN - offset):
            temp = [str(i + offset)]
            if not self.skip_promoter:
                temp.append(str(Prbetas[i]))
            if not self.skip_proximal:
                temp.append(str(Pbetas[i]))
            if not self.skip_cres:
                temp.append(str(Dbetas[i]))
            if i == 0 and self.intercept:
                temp.append(str(betas[-1]))
            print('\t'.join(temp), file=output)
        output.close()
        output = gzip.open('%s_predicted_expression.txt.gz' % self.out_prefix, 'wb')
        strand_d = {True: b'-', False: b'+'}
        for i in range(self.tssN):
            output.write(b"%s\t%i\t%i\t.\t%f\t%s\n"
                         % (self.rna['chr'][i], self.rna['TSS'][i], self.rna['TSS'][i] + 1,
                            y_l[i], strand_d[self.rna['strand'][i]]))
        output.close()
        if not self.skip_cres:
            offset = int(self.nozero)
            output = gzip.open('%s_eRP.txt.gz' % self.out_prefix, 'wb')
            output.write(b"Chr\tTSS\tStrand\tcCREstart\tcCREstop\teRP\tProximal\n")
            for i in range(self.tssN):
                s = self.TSS_indices[i]
                e = self.TSS_indices[i + 1]
                if e - s == 0:
                    continue
                where = numpy.where(self.selected[s:e])[0] + s
                if where.shape[0] == 0:
                    continue
                if self.singlestate:
                    Cstates2 = self.pair_states[where, self.lessone, offset:]
                else:
                    Cstates2 = Cstates[self.pairs[where, 1], self.lessone, offset:]
                eRPs = numpy.zeros(where.shape[0], dtype=numpy.float32)
                if not self.skip_proximal:
                    pmask = numpy.where(self.pairs[where, 2] == 1)[0]
                    if self.log:
                        eRPs[pmask] = numpy.sum(numpy.log2(Cstates2[pmask, :] + 1e-3) * pbetas, axis=1)
                    else:
                        eRPs[pmask] = numpy.sum(Cstates2[pmask, :] * Pbetas, axis=1)
                dmask = numpy.where(self.pairs[where, 2] == 0)[0]
                if self.log:
                    eRPs[dmask] = numpy.sum(numpy.log2(Cstates2[dmask, :] + 1e-3) * dbetas, axis=1)
                else:
                    eRPs[dmask] = numpy.sum(Cstates2[dmask, :] * Dbetas, axis=1)
                chrom = self.rna['chr'][i]
                tss = self.rna['TSS'][i]
                strand = strand_d[self.rna['strand'][i]]
                for j, k in enumerate(where):
                    output.write(b"%s\t%i\t%s\t%i\t%i\t%f\t%i\n"
                                 % (chrom, tss, strand, self.cre['start'][self.pairs[k, 1]],
                                    self.cre['end'][self.pairs[k, 1]], eRPs[j], self.pairs[k, 2]))
            output.close()

        if not self.skip_cres:
            dtype = [('chr', self.chroms.dtype), ('TSS', numpy.int32), ('strand', numpy.bool),
                     ('cCRE', numpy.int32, (2,)), ('eRP', numpy.float32), ('proximal', numpy.bool)]
            eRP = numpy.zeros(self.selected.shape[0], dtype=numpy.dtype(dtype))
            offset = int(self.nozero)
            for i in range(self.tssN):
                s = self.TSS_indices[i]
                e = self.TSS_indices[i + 1]
                if e - s == 0:
                    continue
                where = numpy.where(self.selected[s:e])[0] + s
                if where.shape[0] == 0:
                    continue
                if self.singlestate:
                    Cstates2 = self.pair_states[where, self.lessone, offset:]
                else:
                    Cstates2 = Cstates[self.pairs[where, 1], self.lessone, offset:]
                if not self.skip_proximal:
                    pmask = numpy.where(self.pairs[where, 2] == 1)[0]
                    if self.log:
                        eRP['eRP'][where[pmask]] = numpy.sum(numpy.log2(Cstates2[pmask, :] + 1e-3)
                                                             * pbetas, axis=1)
                    else:
                        eRP['eRP'][where[pmask]] = numpy.sum(Cstates2[pmask, :] * Pbetas, axis=1)
                    eRP['proximal'][where[pmask]] = True
                dmask = numpy.where(self.pairs[where, 2] == 0)[0]
                if self.log:
                    eRP['eRP'][where[dmask]] = numpy.sum(numpy.log2(Cstates2[dmask, :] + 1e-3)
                                                         * dbetas, axis=1)
                else:
                    eRP['eRP'][where[dmask]] = numpy.sum(Cstates2[dmask, :] * Dbetas, axis=1)
                eRP['proximal'][where[dmask]] = False
                eRP['chr'][where] = self.rna['chr'][i]
                eRP['TSS'][where] = self.rna['TSS'][i]
                eRP['strand'][where] = self.rna['strand'][i]
                eRP['cCRE'][where, 0] = self.cre['start'][self.pairs[where, 1]]
                eRP['cCRE'][where, 1] = self.cre['end'][self.pairs[where, 1]]
            eRP = eRP[numpy.where(self.selected)]
            numpy.save('%s_eRP.npy' % self.out_prefix, eRP)
        output = open('%s_settings.txt' % self.out_prefix, 'w')
        print("lessone = %i" % self.lessone, file=output)
        print("states shuffled: %s" % self.shufflestates, file=output)
        print("rna celltypes shuffled: %s" % self.shufflerna, file=output)
        print("rna celltypes shuffled per gene: %s" % self.shuffleallrna, file=output)
        if not self.skip_cres:
            print("initialization_dist = %i" % self.initialization_dist, file=output)
            if not self.skip_promoter:
                print("promoter_dist = %i" % self.promoter_dist, file=output)
            if not self.skip_proximal:
                print("proximal_dist = %i" % self.proximal_dist, file=output)
                print("distal_dist = %i" % self.distal_dist, file=output)
            else:
                print("cre_dist = %i" % self.distal_dist, file=output)
            print("corr_cutoff = %f" % self.corr_cutoff, file=output)
            print("refining_iter = %i" % self.refining_iter, file=output)
            if self.skip_promoter_dist is not None:
                print("skip_cre_promoter_dist = %i" % self.skip_promoter_dist, file=output)
            elif self.includeTSS:
                print("tss_dist = %i" % self.TSSdist, file=output)
            print("multirefine = %s" % self.multirefine, file=output)
            print("singlestate = %s" % self.singlestate, file=output)
            print("maxcres = %s" % self.maxcres , file=output)
        if self.pca is not None:
            print("pca = {}".format(self.pca), file=output)
        print("log = %s" % self.log, file=output)
        print("intercept = %s" % self.intercept , file=output)
        print("nozero = %s" % self.nozero , file=output)
        print("seed = %s" % self.seed , file=output)
        output.close()

    def find_TSS_ranges(self):
        if self.skip_promoter_dist is not None:
            TSS_ranges = numpy.zeros((self.tssN, 4), dtype=numpy.int32)
        else:
            TSS_ranges = numpy.zeros((self.tssN, 2), dtype=numpy.int32)
        for i in range(self.chroms.shape[0]):
            s = self.rna_indices[i]
            e = self.rna_indices[i + 1]
            if e - s == 0:
                continue
            s1 = self.cre_indices[i]
            e1 = self.cre_indices[i + 1]
            if e1 - s1 == 0:
                continue
            TSS_ranges[s:e, 0] = numpy.searchsorted(self.cre['end'][s1:e1],
                                                    self.rna['TSS'][s:e] - self.distal_dist) + s1
            TSS_ranges[s:e, -1] = numpy.searchsorted(self.cre['start'][s1:e1],
                                                    self.rna['TSS'][s:e] + self.distal_dist) + s1
            if self.skip_promoter_dist is not None:
                TSS_ranges[s:e, 1] = numpy.searchsorted(
                    self.cre['end'][s1:e1], self.rna['TSS'][s:e] - self.skip_promoter_dist
                    * numpy.logical_not(self.rna['strand'][s:e])) + s1
                TSS_ranges[s:e, 2] = numpy.searchsorted(
                    self.cre['start'][s1:e1], self.rna['TSS'][s:e] + self.skip_promoter_dist
                    * self.rna['strand'][s:e]) + s1
        return TSS_ranges

    def _assign_promoter_state(self, in_q, out_q, Pranges, dist, seed, symmetric):
        rng = numpy.random.RandomState(seed=seed)
        order = numpy.arange(self.stateN)
        rng.shuffle(order)
        r = numpy.arange(self.stateRepN)
        udist = dist
        if symmetric:
            dist2 = dist * 2
            ddist = dist
        else:
            dist2 = dist
            ddist = 0
        while True:
            data = in_q.get(True)
            if data is None:
                break
            start, stop = data
            Pstates = numpy.zeros((stop - start, self.stateRepN, self.stateN), dtype=numpy.int32)
            for i in range(start, stop):
                Pstart, Pstop = Pranges[i, 0], Pranges[i, 1] - 1
                if Pstop < Pstart:
                    Pstates[i - start, :, 0] = dist2
                    continue
                overlap = Pstop - Pstart
                if self.rna['strand'][i]:
                    TSS_st = self.rna['TSS'][i] - ddist
                    TSS_ed = self.rna['TSS'][i] + udist
                else:
                    TSS_st = self.rna['TSS'][i] - udist
                    TSS_ed = self.rna['TSS'][i] + ddist
                #TSS_st = self.rna['TSS'][i] - dist# * numpy.logical_not(self.rna['strand'][i])
                #TSS_ed = self.rna['TSS'][i] + dist# * self.rna['strand'][i]
                if overlap >= 0:
                    size = max(0, (min(TSS_ed, self.state['end'][Pstart])
                            - max(TSS_st, self.state['start'][Pstart])))
                    Pstates[i - start, r, self.state['state'][Pstart, :]] = size
                if overlap >= 1:
                    size = max(0, (min(TSS_ed, self.state['end'][Pstop])
                            - max(TSS_st, self.state['start'][Pstop])))
                    Pstates[i - start, r, self.state['state'][Pstop, :]] += size
                if overlap >= 2:
                    for j in range(Pstart + 1, Pstop):
                        size = self.state['end'][j] - self.state['start'][j]
                        Pstates[i - start, r, self.state['state'][j, :]] += size
                Pstates[i - start, :, 0] = dist2 - numpy.sum(Pstates[i - start, :, 1:], axis=1)
            Pstates2 = numpy.zeros((stop - start, self.cellN, self.stateN), dtype=numpy.float32)
            for i in range(self.cellN):
                s = self.state_cindices[i]
                e = self.state_cindices[i + 1]
                if e - s > 1:
                    Pstates2[:, i, :] = numpy.sum(Pstates[:, s:e, :], axis=1) / (e - s)
                else:
                    Pstates2[:, i, :] = Pstates[:, s, :]
            Pstates2 /= dist2
            if self.shufflestates:
                for i in range(stop - start):
                    for j in range(self.cellN):
                        Pstates2[i, j, :] = Pstates2[i, j, order]
                        rng.shuffle(order)
            out_q.put((start, stop, Pstates2))
        out_q.put(None)

    def _assign_CRE_state(self, in_q, out_q, seed):
        rng = numpy.random.RandomState(seed=seed)
        order = numpy.arange(self.stateN)
        rng.shuffle(order)
        r = numpy.arange(self.stateRepN)
        while True:
            data = in_q.get(True)
            if data is None:
                break
            start, stop = data
            Cstates = numpy.zeros((stop - start, self.stateRepN, self.stateN), dtype=numpy.int32)
            for i in range(start, stop):
                Cstart, Cstop = self.Cranges[i, 0], self.Cranges[i, 1] - 1
                if Cstop < Cstart:
                    Cstates[i - start, :, 0] = self.cre['end'][i] - self.cre['start'][i]
                    continue
                overlap = Cstop - Cstart
                if overlap >= 0:
                    size = (min(self.cre['end'][i], self.state['end'][Cstart])
                            - max(self.cre['start'][i], self.state['start'][Cstart]))
                    Cstates[i - start, r, self.state['state'][Cstart, :]] = size
                if overlap >= 1:
                    size = (min(self.cre['end'][i], self.state['end'][Cstop])
                            - max(self.cre['start'][i], self.state['start'][Cstop]))
                    Cstates[i - start, r, self.state['state'][Cstop, :]] += size
                if overlap >= 2:
                    for j in range(Cstart + 1, Cstop):
                        size = self.state['end'][j] - self.state['start'][j]
                        Cstates[i - start, r, self.state['state'][j, :]] += size
                Cstates[i - start, :, 0] += (self.cre['end'][i] - self.cre['start'][i]
                                             - numpy.sum(Cstates[i - start, 0, :]))
            Cstates2 = numpy.zeros((stop - start, self.cellN, self.stateN), dtype=numpy.float32)
            for i in range(self.cellN):
                s = self.state_cindices[i]
                e = self.state_cindices[i + 1]
                if e - s > 1:
                    Cstates2[:, i, :] = numpy.mean(Cstates[:, s:e, :], axis=1)
                else:
                    Cstates2[:, i, :] = Cstates[:, s, :]
            Cstates2 /= numpy.sum(Cstates2, axis=2, keepdims=True)
            if self.shufflestates:
                for i in range(stop - start):
                    for j in range(self.cellN):
                        Cstates2[i, j, :] = Cstates2[i, j, order]
                        rng.shuffle(order)
            out_q.put((start, stop, Cstates2))
        out_q.put(None)

    def _find_correlations(self, in_q, out_q):
        betas = self.initial_betas[:self.Cstates.shape[2]].reshape(1, 1, -1)
        offset = int(self.nozero)
        while True:
            data = in_q.get(True)
            if data is None:
                break
            start, stop, ranges = data
            results = []
            for i in range(start, stop):
                i0 = i - start
                if ranges[i0, -1] - ranges[i0, 0] == 0:
                    continue
                if self.skip_promoter_dist is not None:
                    Cindices = numpy.r_[numpy.arange(ranges[i0, 0], ranges[i0, 1]),
                                        numpy.arange(ranges[i0, 2], ranges[i0, 3])]
                else:
                    Cindices = numpy.arange(ranges[i0, 0], ranges[i0, 1])
                if self.log:
                    Cstates = numpy.log2(self.Cstates[Cindices, :, :][:, self.lmask, offset:] + 1e-3)
                else:
                    Cstates = self.Cstates[Cindices, :, :][:, self.lmask, offset:]
                cbetas = numpy.sum(betas * Cstates, axis=2)
                cbetas -= numpy.mean(cbetas, axis=1, keepdims=True)
                cbetas /= ((numpy.std(cbetas, axis=1, keepdims=True, ddof=1) + 1e-5)
                           * (self.cellN - 1) ** 0.5)
                corrs = numpy.dot(self.norm_rna[i:(i + 1), :], cbetas.T)[0, :]
                results.append((i, corrs))
            out_q.put(results)
        out_q.put(None)

    def _find_ss_correlations(self, in_q, out_q, seed):
        rng = numpy.random.RandomState(seed=seed)
        order = numpy.arange(self.stateN)
        rng.shuffle(order)
        betas = self.initial_betas.reshape(1, 1, -1)
        while True:
            data = in_q.get(True)
            if data is None:
                break
            start, stop, ranges, cre_states = data
            ranges -= ranges[0, 0]
            cstates = []
            for i in range(len(cre_states)):
                temp = numpy.zeros((cre_states[i].shape[0], self.stateRepN, self.stateN),
                                   dtype=numpy.int32)
                temp[numpy.repeat(numpy.arange(temp.shape[0]), self.stateRepN),
                     numpy.tile(numpy.arange(self.stateRepN), temp.shape[0]),
                     cre_states[i].ravel(order='C')] = 1
                cstates.append(numpy.zeros((temp.shape[0], self.cellN, self.stateN),
                                           dtype=numpy.float32))
                for j in range(self.cellN):
                    s, e = self.state_cindices[j:(j + 2)]
                    if e - s == 1:
                        cstates[-1][:, j, :] = temp[:, s, :]
                    else:
                        cstates[-1][:, j, :] = numpy.mean(temp[:, s:e, :], axis=1)
                cstates[-1][:, :, 0] += 1.0 - numpy.sum(cstates[-1], axis=2)
                if self.shufflestates:
                    for j in range(cstates[-1].shape[0]):
                        for k in range(self.cellN):
                            cstates[-1][j, k, :] = cstates[-1][j, k, order]
                            rng.shuffle(order)
            results = []
            for i in range(start, stop):
                i0 = i - start
                n = ranges[i0, -1] - ranges[i0, 0]
                if n == 0:
                    continue
                if self.skip_promoter_dist is not None:
                    Cindices = numpy.r_[numpy.arange(ranges[i0, 0], ranges[i0, 1]),
                                        numpy.arange(ranges[i0, 2], ranges[i0, 3])]
                else:
                    Cindices = numpy.arange(ranges[i0, 0], ranges[i0, 1])
                n = Cindices.shape[0]
                states = numpy.zeros((n, self.cellN, self.stateN), dtype=numpy.float32)
                corrs = numpy.zeros(n, dtype=numpy.float32)
                for j, index in enumerate(Cindices):
                    if cstates[index].shape[0] == 0:
                        continue
                    cbetas = numpy.sum(betas * cstates[index][:, self.lmask, :], axis=2)
                    cbetas -= numpy.mean(cbetas, axis=1, keepdims=True)
                    cbetas /= numpy.std(cbetas, axis=1, keepdims=True, ddof=1) + 1e-5
                    cre_corrs = numpy.dot(self.norm_rna[i:(i + 1), :], cbetas.T)[0, :]
                    best_corr = numpy.amax(cre_corrs)
                    where = numpy.where(cre_corrs == best_corr)[0][0]
                    states[j, :, :] = cstates[index][where, :, :]
                    corrs[j] = best_corr
                results.append((i, corrs, states))
            out_q.put(results)
        out_q.put(None)

    def _find_betas(self, in_q, out_q):
        while True:
            data = in_q.get(True)
            if data is None:
                break
            index, lessone, mask, X, Y = data
            betas = numpy.linalg.lstsq(X[:, mask, :].reshape(-1, X.shape[2], order='C'),
                                       Y[:, mask].reshape(-1, order='C'), rcond=None)[0]
            pred_exp = numpy.dot(X[:, index, :], betas.reshape(-1, 1))[:, 0]
            out_q.put((data[0], betas, pred_exp))
        out_q.put(None)

    def _refine_pair_CREs(self, in_q, out_q, Prbetas, Pbetas, Dbetas, intercept, iteration, seed):
        rng = numpy.random.RandomState(seed=seed)
        if self.pca is not None:
            XPrs = self.pca_XPrs
            Cstates = self.pca_Cstates
        else:
            XPrs = self.XPrs
            Cstates = self.Cstates
        #change_cutoff = numpy.round(self.refining_iter / float(iteration + 1)) + 1
        if iteration == 0:
            change_cutoff = numpy.inf
        else:
            change_cutoff = numpy.floor(1 - numpy.log(iteration / float(self.refining_iter - 1)) / numpy.log(1.5))
        offset = int(self.nozero)
        while True:
            data = in_q.get(True)
            if data is None:
                break
            index, total_mse, Y = data
            s = self.TSS_indices[index]
            e = self.TSS_indices[index + 1]
            proximal = self.pairs[s:e, 2]
            XPrs2 = XPrs[index, :, :]
            if self.singlestate:
                Cstates2 = self.pair_states[s:e, self.lmask, :]
            else:
                Cstates2 = Cstates[self.pairs[s:e, 1], :, :][:, self.lmask, :]
            selected = numpy.copy(self.selected[s:e])
            if len(Prbetas) > 0:
                XPr = numpy.sum(Prbetas * XPrs2, axis=1)
                base_exp = numpy.copy(XPr)
            else:
                base_exp = numpy.zeros(self.cellN - 1, dtype=numpy.float32)
            base_exp += intercept
            mse = numpy.zeros(Cstates2.shape[0], dtype=numpy.float32)
            if not self.skip_proximal:
                pmask = numpy.where(proximal)[0]
                dmask = numpy.where(proximal == 0)[0]
                XD = numpy.sum(Cstates2[dmask[selected[dmask]], :, :],
                                axis=0).astype(numpy.float32)
                XD_total = numpy.sum(XDs, axis=1, keepdims=True)
                if self.nozero:
                    XD = XD[:, offset:]
                if self.log:
                    XDs = numpy.log2((XD / XD_total) + 1e-3)
                XDs = numpy.sum(XDs * Dbetas, axis=1)
                base_exp += XDs
                mask = numpy.copy(selected[pmask])
                XP = numpy.sum(Cstates2[pmask[selected[pmask]], :, :],
                               axis=0)
                XP_total = numpy.sum(XP, axis=1)
                if self.nozero:
                    XP = XP[:, offset:]
                if not self.log:
                    XP_sum = numpy.sum(XP * Pbetas, axis=1)
                for i, j in enumerate(pmask):
                    if not mask[i]:
                        if self.log:
                            #XPs = numpy.sum(
                            #    Pbetas * numpy.log2((XP + Cstates2[j, :, offset:])
                            #                        / (XP_total + numpy.sum(Cstates2[j, :, :], axis=1)
                            #                          ).reshape(-1, 1) + 1e-3), axis=1)
                            XPs = numpy.sum(Pbetas * numpy.log2(XP + Cstates2[j, :, offset:] + 1e-3), axis=1)

                        else:
                            #XPs = ((XP_sum + numpy.sum(Cstates2[j, :, offset:] * Pbetas, axis=1))
                            #       / (XP_total + numpy.sum(Cstates2[j, :, :], axis=1)))
                            XPs = XP_sum + numpy.sum(Cstates2[j, :, offset:] * Pbetas, axis=1)
                    else:
                        if self.log:
                            #XPs = numpy.sum(
                            #    Pbetas * numpy.log2((XP - Cstates2[j, :, offset:])
                            #                        / numpy.maximum(1e-5, (XP_total - numpy.sum(Cstates2[j, :, :], axis=1))
                            #                                       ).reshape(-1, 1) + 1e-3), axis=1)
                            XPs = numpy.sum(Pbetas * numpy.log2(XP - Cstates2[j, :, offset:] + 1e-3), axis=1)
                        else:
                            #XPs = ((XP_sum - numpy.sum(Cstates2[j, :, offset:] * pbetas, axis=1))
                            #       / numpy.maximum(1e-5, XP_total - numpy.sum(Cstates2[j, :, :], axis=1)))
                            XPs = XP_sum - numpy.sum(Cstates2[j, :, offset:] * pbetas, axis=1)
                    pred_exp = base_exp + XPs
                    mse[j] = numpy.sum((Y - pred_exp) ** 2)
                if self.log:
                    XPs = numpy.sum(numpy.log2(XP / XP_total.reshape(-1, 1) + 1e-3)
                                    * Pbetas, axis=1)
                else:
                    XPs = (XP_sum / XP_total)
            else:
                XPs = numpy.zeros(Y.shape[0], dtype=numpy.float32)
                XD = None
            if len(Pbetas) > 0:
                base_exp = numpy.copy(XPr)
            else:
                base_exp = numpy.zeros(self.cellN - 1, dtype=numpy.float32)
            if not self.skip_proximal:
                base_exp += XPs
            base_exp += intercept
            if XD is None:
                dmask = numpy.where(proximal == 0)[0]
                XD = numpy.sum(Cstates2[dmask[selected[dmask]], :, :],
                                axis=0).astype(numpy.float32)
                XD_total = numpy.maximum(1e-5, numpy.sum(XD, axis=1).astype(numpy.float32))
                if self.nozero:
                    XD = XD[:, offset:]
                if not self.log:
                    XD_sum = numpy.sum(XD * Dbetas, axis=1)
            else:
                XD_total = XD_total[:, 0]
            mask = numpy.copy(selected[dmask])
            for i, j in enumerate(dmask):
                if not mask[i]:
                    if self.log:
                        #XDs = numpy.sum(
                        #    Dbetas * numpy.log2((XD + Cstates2[j, :, offset:])
                        #                        / (XD_total + numpy.sum(Cstates2[j, :, :], axis=1)
                        #                           ).reshape(-1, 1) + 1e-3), axis=1)
                        XDs = numpy.sum(Dbetas * numpy.log2(XD + Cstates2[j, :, offset:]
                                                            + 1e-3), axis=1)
                    else:
                        #XDs = ((XD_sum + numpy.sum(Cstates2[j, :, offset:] * Dbetas, axis=1))
                        #      / (XD_total + numpy.sum(Cstates2[j, :, :], axis=1)))
                        XDs = XD_sum + numpy.sum(Cstates2[j, :, offset:] * Dbetas, axis=1)
                else:
                    if self.log:
                        #XDs = numpy.sum(
                        #    Dbetas * numpy.log2((XD - Cstates2[j, :, offset:])
                        #                        / numpy.maximum(1e-5, (XD_total - numpy.sum(Cstates2[j, :, :], axis=1))
                        #                           ).reshape(-1, 1) + 1e-3), axis=1)
                        XDs = numpy.sum(Dbetas * numpy.log2(XD - Cstates2[j, :, offset:]
                                                            + 1e-3), axis=1)
                    else:
                        #XDs = ((XD_sum - numpy.sum(Cstates2[j, :, offset:] * Dbetas, axis=1))
                        #      / numpy.maximum(1e-5, XD_total - numpy.sum(Cstates2[j, :, :], axis=1)))
                        XDs = XD_sum - numpy.sum(Cstates2[j, :, offset:] * Dbetas, axis=1)
                pred_exp = base_exp + XDs
                mse[j] = numpy.sum((Y - pred_exp) ** 2)
            change = numpy.where(mse < total_mse)[0]
            if change.shape[0] > 0:
                changed = True
                if self.multirefine and change.shape[0] <= change_cutoff:
                    selected[change] = numpy.logical_not(selected[change])
                else:
                    try:
                        weights = numpy.maximum(1e-5, (total_mse - mse[change]) / total_mse)
                        weights /= numpy.sum(weights)
                        weights = weights ** (1 + 9 * iteration / max(1, self.refining_iter - 1))
                        weights = (numpy.full(weights.shape[0], 1e-5, dtype=numpy.float32) +
                                   (1 - 1e-5 * weights.shape[0]) * weights / numpy.sum(weights))
                        if self.multirefine:
                            change = rng.choice(change, int(change_cutoff), p=weights,
                                                         replace=False)
                        else:
                            change = rng.choice(change, 1, p=weights,
                                                         replace=False)
                        selected[change] = numpy.logical_not(selected[change])
                    except:
                        print('a', total_mse, list((total_mse - mse[change]) / total_mse), list(weights))
                selectedN = numpy.sum(selected)
                if self.maxcres > 0 and selectedN > self.maxcres:
                    where = numpy.where(selected & (mse > total_mse))[0]
                    if where.shape[0] < self.maxcres - selectedN:
                        try:
                            weights = numpy.maximum(1e-5, (mse[where] - total_mse) / total_mse)
                            weights /= numpy.sum(weights)
                            weights = weights ** (1 + 9 * iteration / max(1, self.refining_iter - 1))
                            weights = (numpy.full(weights.shape[0], 1e-5, dtype=numpy.float32) +
                                       (1 - 1e-5 * weights.shape[0]) * weights
                                       / numpy.sum(weights))
                            change = rng.choice(where, selectedN - self.maxcres,
                                                         p=weights, replace=False)
                            selected[change] = False
                        except:
                            print('b', total_mse, list((mse[where] - total_mse) / total_mse),
                                  list(weights))
                    else:
                        selected[where] = False
                        selectedN = numpy.sum(selected)
                        if selectedN > self.maxcres:
                            where = numpy.where(selected)[0]
                            try:
                                weights = numpy.maximum(1e-5, (total_mse - mse[where]) / total_mse)
                                weights /= numpy.sum(weights)
                                weights = weights ** (1 + 9 * iteration / max(1, self.refining_iter - 1))
                                weights = (numpy.full(weights.shape[0], 1e-5, dtype=numpy.float32) +
                                           (1 - 1e-5 * weights.shape[0]) * weights
                                           / numpy.sum(weights))
                                change = rng.choice(where, selectedN - self.maxcres,
                                                             p=weights, replace=False)
                                selected[change] = False
                            except:
                                print('c', total_mse, list((total_mse - mse[where]) / total_mse),
                                      list(weights))
            else:
                changed = False
            out_q.put((index, changed, selected))
        out_q.put(None)


if __name__ == "__main__":
    main()
