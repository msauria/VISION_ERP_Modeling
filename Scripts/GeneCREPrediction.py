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
    model.run(args.output, args.init_dist, args.promoter_dist, args.cre_dist,
              args.cre_noprom, args.correlation, args.iterations, args.lessone,
              args.threads, args.train_stats, args.eRP, args.max_cres,
              args.skip_training, args.shuffle_states, args.seed)

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
    parser.add_argument('--initialization-dist', dest="init_dist", type=int, action='store', default=1000,
                        help="Beta initialization distance cutoff")
    parser.add_argument('--promoter-dist', dest="promoter_dist", type=int, action='store',
                        help="If specified, learn betas for promoters up to promoter distance cutoff")
    parser.add_argument('--cre-dist', dest="cre_dist", type=int, action='store',
                        help="CRE distance cutoff")
    parser.add_argument('--cre-exclude-promoter', dest="cre_noprom", action='store_true',
                      help="Exclude promoter from CREs")
    parser.add_argument('--correlation', dest="correlation", type=float, action='store', default=0.0,
                        help="Initial correlation cutoff")
    parser.add_argument('--trainstats', dest="train_stats", action='store_true',
                        help="Output training statistics")
    parser.add_argument('--max-CREs', dest="max_cres", action='store', type=int, default=0,
                        help="Maximum number of CREs allowed to be selected per TSS at a time (0 is no max)")
    parser.add_argument('--skip-training', dest="skip_training", action='store_true',
                        help="Skip CRE-TSS pairining refinement")
    parser.add_argument('--shuffle-states', dest="shuffle_states", action='store_true',
                        help="Shuffle the state proportions of each CRE as a negative control")
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

    def run(self, out_prefix, initialization_dist=1000, promoter_dist=None,
        cre_dist=None, cre_noprom=False, corr_cutoff=0.2, refining_iter=100,
        lessone=0, threads=1, train_stats=False, eRP=None, max_cres=None,
        skip_training=False, shuffle_states=False, seed=None):
        self.lessone = lessone # Sample to hold out and predict at end
        self.lmask = numpy.r_[numpy.arange(lessone), numpy.arange((lessone + 1), self.cellN)] # Indices excluding lessone celltype
        self.shufflestates = bool(shuffle_states)
        self.initialization_dist = int(initialization_dist) # Distance from TSSs to initialize beta values
        if cre_dist is not None:
            self.cre_dist = int(cre_dist) # Max distance from TSSs for distal cCRE state assessment, exluding proximal, if used
            self.skip_cres = False
            self.skip_cre_promoter = bool(cre_noprom)
        else:
            self.skip_cres = True
            if promoter_dist is None:
                self.logger.error("You must either speficy a distance for CREs or promoters")
                sys.exit(0)
        if promoter_dist is not None:
            self.promoter_dist = int(promoter_dist) # Max distance from TSSs for promoter state assessment, if used
            self.skip_promoter = False
        else:
            self.skip_promoter = True
        self.corr_cutoff = float(corr_cutoff) # Correlation cutoff for initial filtering of cCREs
        self.refining_iter = int(refining_iter) # Number of refining iterations to perform
        self.threads = max(1, int(threads)) # Number of threads to use
        self.trainstats = bool(train_stats) # Indicator whether to retain training statistics
        self.maxcres = int(max_cres)
        self.skip_training = bool(skip_training)
        self.out_prefix = str(out_prefix)
        self.seed = seed
        self.rng = numpy.random.RandomState(seed=seed)

        if self.skip_promoter:
            self.promoter_beta_indices = []
            self.XPrs = numpy.zeros((self.tssN, self.cellN - 1, 0), dtype=numpy.float32)
            self.XPrs_l = numpy.zeros((self.tssN, 0), dtype=numpy.float32)
        else:
            self.promoter_beta_indices = numpy.arange(self.stateN)
        if self.skip_cres:
            self.cre_beta_indices = []
        else:
            self.cre_beta_indices = numpy.arange(self.stateN) + len(self.promoter_beta_indices)

        self.logger.info("Left out celltype: {}".format(self.celltypes[self.lessone]))
        if not self.skip_promoter:
            self.logger.info("Promoter winow: {}".format(self.promoter_dist))
        if not self.skip_cres:
            self.logger.info("CRE window: {}".format(self.cre_dist))
            if self.skip_cre_promoter:
                self.logger.info("Exluding CREs in promoter")
        if not self.skip_cres:
            self.logger.info("Beta initialization distance: {}".format(self.initialization_dist))
            self.logger.info("Initial CRE correlation cutoff: {}".format(self.corr_cutoff))
            self.logger.info("Number of refinement iterations: {}".format(self.refining_iter))
        if not self.skip_cres and self.maxcres is not None:
            self.logger.info("Allowing up to {} CREs per TSS to be selected".format(self.maxcres))

        self.norm_rna = numpy.copy(self.rna['rna'][:, self.lmask])
        self.norm_rna -= numpy.mean(self.norm_rna, axis=1).reshape(-1, 1)
        self.norm_rna /= ((numpy.std(self.norm_rna, axis=1, ddof=1) + 1e-5)
                          * (self.cellN - 1) ** 0.5).reshape(-1, 1)
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
        self.predict_lessone()
        self.write_settings()

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
        self.logger.info("Found RNA-state pairings for celltypes {}".format(
                         ', '.join(list(self.celltypes))))

    def load_CREs(self):
        if self.verbose >= 2:
            print("\r{}\rLoading cCRE data".format(' ' * 80), end='', file=sys.stderr)
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
            print("\r{}\r".format(' ' * 80), end='', file=sys.stderr)
        self.logger.info("Chromosomes to be analyzed: {}".format(
                         ', '.join([x.decode('utf-8') for x in self.chroms])))
        self.logger.info('Loaded {} CREs'.format(self.cre_indices[-1]))

    def load_rna(self):
        if self.verbose >= 2:
            print("\r{}\rLoading RNA data".format(' ' * 80), end='', file=sys.stderr)
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
        self.logger.info('Loaded {} expression profiles across {} celltypes ({} replicates)'.format(
                         self.tssN, self.cellN, self.rnaRepN))

    def load_state(self):
        if self.verbose >= 2:
            print("\r{}\rLoading state data".format(' ' * 80), end='', file=sys.stderr)
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
            self.state_indices[i + 1] = self.state_indices[i] + where.shape[0]
        self.state = data
        if self.verbose >= 2:
            print("\r{}\r".format(' ' * 80), end='', file=sys.stderr)
        self.logger.info('Loaded {} non-zero state profiles ({} states) across {} celltypes ({} replicates)'.format(
                         self.state.shape[0], self.stateN, self.cellN, self.stateRepN))

    def assign_promoter_states(self):
        """Find the proportion of states in each promoter window"""
        if self.verbose >= 2:
            print("\r{}\rAssign states to promoters".format(' ' * 80), end='', file=sys.stderr)
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
                                        self.rna['TSS'][s:e] - self.promoter_dist,
                                        #* numpy.logical_not(self.rna['strand'][s:e]),
                                        side='right') + s1
            stops = numpy.searchsorted(self.state['start'][s1:e1],
                                       self.rna['TSS'][s:e] + self.promoter_dist,
                                       #* self.rna['strand'][s:e],
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
        self.Pstates = Pstates
        self.XPrs = self.Pstates[:, self.lmask, :]
        self.XPrs_l = self.Pstates[:, self.lessone, :]
        if self.verbose >= 2:
            print("\r{}\r".format(' ' * 80), end='', file=sys.stderr)

    def assign_CRE_states(self):
        """Find the proportion of states in each cCRE"""
        if self.verbose >= 2:
            print("\r{}\rAssign states to CREs".format(' ' * 80), end='', file=sys.stderr)
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
        Cstates = numpy.zeros((self.cre.shape[0], self.cellN, self.stateN), dtype=numpy.int32)
        finished = 0
        while finished < self.threads:
            results = results_queue.get(True)
            if results is None:
                finished += 1
                continue
            start, stop = results[:2]
            Cstates[start:stop, :, :] = results[2]
        self.Cstates = Cstates
        if self.verbose >= 2:
            print("\r{}\r".format(' ' * 80), end='', file=sys.stderr)

    def find_initial_betas(self):
        """Using the window specified by 'initialization_dist' around TSSs, find initial beta values"""
        if self.verbose >= 2:
            print("\r{}\rFinding initial betas".format(' ' * 80), end='', file=sys.stderr)
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
        betas = numpy.linalg.lstsq(Tstates.reshape(-1, Tstates.shape[2], order='C'),
                                   self.rna['rna'][:, self.lmask].reshape(-1, order='C'),
                                   rcond=None)[0]
        self.initial_betas = betas
        if self.verbose >= 2:
            print("\r{}\r".format(' ' * 80), end='', file=sys.stderr)

    def find_TSS_CRE_pairs(self):
        """Find all cCREs in distal_dist window and find correlation with gene expression 
           for each one, filtering out cCREs for each gene bleow threshold"""
        if self.verbose >= 2:
            print("\r{}\rFinding TSS-cCRE pairs".format(' ' * 80), end='', file=sys.stderr)
        TSS_ranges = self.find_TSS_ranges()
        if self.skip_cre_promoter:
            pair_indices = numpy.r_[0, numpy.cumsum(TSS_ranges[:, 1] - TSS_ranges[:, 0]
                                                    + TSS_ranges[:, 3] - TSS_ranges[:, 2])]
        else:
            pair_indices = numpy.r_[0, numpy.cumsum(TSS_ranges[:, 1] - TSS_ranges[:, 0])]
        # Normalize predicted values for easy correlation
        pair_queue = multiprocessing.JoinableQueue()
        results_queue = multiprocessing.JoinableQueue()
        processes = []
        for i in range(self.threads):
            processes.append(multiprocessing.Process(
                target=self._find_correlations, args=(pair_queue, results_queue)))
            processes[-1].daemon = True
            processes[-1].start()
        step = 50
        for i in range(self.chroms.shape[0]):
            for j in range(self.rna_indices[i], self.rna_indices[i + 1], step):
                end = min(j + step, self.rna_indices[i + 1])
                pair_queue.put((j, end, TSS_ranges[j:end, :]))
        for i in range(self.threads):
            pair_queue.put(None)
        pairs = numpy.zeros((pair_indices[-1], 3), dtype=numpy.int32)
        valid = numpy.zeros(pair_indices[-1], dtype=numpy.bool)
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
                if self.skip_cre_promoter:
                    pairs[s:e, 1] = numpy.r_[numpy.arange(TSS_ranges[index, 0], TSS_ranges[index, 1]),
                                             numpy.arange(TSS_ranges[index, 2], TSS_ranges[index, 3])]
                else:
                    pairs[s:e, 1] = numpy.arange(TSS_ranges[index, 0], TSS_ranges[index, 1])
                valid[s:e] = corrs >= self.corr_cutoff
        self.pairs = pairs[numpy.where(valid)[0], :]
        self.TSS_indices = numpy.r_[0, numpy.cumsum(numpy.bincount(self.pairs[:, 0],
                                                                   minlength=self.tssN))]
        self.selected = numpy.ones(self.pairs.shape[0], dtype=numpy.bool)
        if self.maxcres > 0:
            where = numpy.where(self.TSS_indices[1:] - self.TSS_indices[:-1] > self.maxcres)[0]
            for i in where:
                s, e = self.TSS_indices[i:(i + 2)]
                self.selected[self.rng.choice(numpy.arange(s, e), e - s - self.maxcres,
                    replace=False)] = False
        if self.verbose >= 2:
            print("\r{}\r".format(' ' * 80), end='', file=sys.stderr)
        kept = numpy.sum(valid)
        temp = numpy.bincount(self.pairs[:, 0], weights=self.selected, minlength=self.tssN)
        self.logger.info("Retained {} of {} TSS-CRE pairs ({:02.2f}%), {} - {} CREs/TSS (median {})".format(
                         self.selected.shape[0], valid.shape[0],
                         100. * self.selected.shape[0] / valid.shape[0],
                         numpy.amin(temp), numpy.amax(temp), numpy.median(temp)))
        self.logger.info("Unique CREs in pairings: {}".format(numpy.unique(self.pairs[:, 1]).shape[0]))

    def refine_pairs(self):
        if self.verbose >= 2:
            print("\r{}\rFinding TSS-cCRE pairs".format(' ' * 80), end='', file=sys.stderr)
        if self.trainstats:
            output = open('{}_training_statistics.txt'.format(self.out_prefix), 'w')
            print("Iteration\t#Pairs\t%Kept\tR2adj\tMSE\tOutR2adj\tOutMSE", file=output)
            output.close()
        XCs = self.find_Xs()
        X = numpy.dstack((self.XPrs, XCs[:, self.lmask, :]))
        X_l = numpy.hstack((self.XPrs_l, XCs[:, self.lessone, :]))
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
                print("\r{}\rFinding TSS-cCRE pairs - Iteration {} of {}".format(
                      ' ' * 80, iteration, self.refining_iter), end='', file=sys.stderr)
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
            betas = numpy.zeros((self.cellN1, p), dtype=numpy.float32)
            pred_exp = numpy.zeros((self.tssN, self.cellN1), dtype=numpy.float32)
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
                    output = open('{}_training_statistics.txt'.format(self.out_prefix), 'a')
                    print('\t'.join([str(x) for x in training_statistics[-1]]), file=output)
                    output.close()
                if self.verbose >= 2:
                    print("\r{}\rIteration: {}, selected pairs: {:02.2f}% ({} of {}), # Changed: {}, R2adj: {:02.2f}, MSE: {:0.4e}, OutR2adj: {:02.2f}, OutMSE: {:0.4e}".format(
                          ' ' * 80, iteration, 100. * numpy.sum(self.selected) / self.selected.shape[0],
                          numpy.sum(self.selected), self.selected.shape[0], 0, 100 * best_R2adj,
                          numpy.sum(SSres)/n, 100*R2adj_l, SSres_l/n_l),
                          file=sys.stderr)

            pair_queue = multiprocessing.JoinableQueue()
            results_queue = multiprocessing.JoinableQueue()
            processes = []
            print("\n{} {}".format(betas[:, self.promoter_beta_indices[0]],
                                   betas[:, self.cre_beta_indices[0]]))
            for i in range(self.threads):
                processes.append(multiprocessing.Process(
                    target=self._refine_pair_CREs, args=(pair_queue, results_queue,
                                                         betas[:, self.promoter_beta_indices],
                                                         betas[:, self.cre_beta_indices],
                                                         iteration, self.rng.randint(99999))))
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
                print("\r{}\rFinding TSS-cCRE pairs - Iteration {} of {}, TSS {} of {}".format(
                      ' ' * 80, iteration, self.refining_iter, count, self.tssN),
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
                    print("\r{}\rFinding TSS-cCRE pairs - Iteration {} of {}, TSS {} of {}".format(
                          ' ' * 80, iteration, self.refining_iter, count, self.tssN),
                          end='', file=sys.stderr)
                    disp_count -= 100

            if self.verbose >= 2:
                print("\r{}\rFinding TSS-cCRE pairs - Iteration {} of {}".format(
                      ' ' * 80, iteration, self.refining_iter), end='', file=sys.stderr)
            changed_sum = self.selected.shape[0] - numpy.sum(numpy.equal(new_selected,
                                                                         self.selected))
            new_sum = numpy.sum(new_selected)
            old_sum = numpy.sum(self.selected)
            if changed_sum == 0:
                break
            self.selected = new_selected
            XCs = self.find_Xs()
            X = numpy.dstack((self.XPrs, XCs[:, self.lmask, :]))
            X_l = numpy.hstack((self.XPrs_l, XCs[:, self.lessone, :]))
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
                print("\r{}\rIteration: {}, selected pairs: {:02.2f}% ({} of {}), # Changed: {}, R2adj: {:02.2f}, MSE: {:0.4e}, OutR2adj: {:02.2f}, OutMSE: {:0.4e}".format(
                      ' ' * 80, iteration + 1, 100 * percent_kept,
                      new_sum, new_selected.shape[0], changed_sum,
                      100 * R2adj, SSres/n, R2adj_l*100, SSres_l/n_l),
                      file=sys.stderr)
            if self.trainstats:
                output = open('{}_training_statistics.txt'.format(self.out_prefix), 'a')
                print('\t'.join([str(x) for x in training_statistics[-1]]), file=output)
                output.close()

            if R2adj > best_R2adj:
                best_R2adj = R2adj
                best_selected = numpy.copy(self.selected)
                best_betas = betas
                best_R2adj_l = R2adj_l

        self.betas = best_betas
        self.R2adj = best_R2adj
        self.selected = best_selected
        if self.verbose >= 2:
            print("\r{}\r".format(' ' * 80), end='', file=sys.stderr)
        temp = numpy.bincount(self.pairs[:, 0], weights=self.selected, minlength=self.tssN)
        kept = numpy.sum(self.selected)
        self.logger.info("Selected {} of {} TSS-CRE pairs ({:02.2f}%), {} - {} CREs/TSS (median {})".format(
                         kept, self.selected.shape[0],
                         100. * kept / self.selected.shape[0], numpy.amin(temp),
                         numpy.amax(temp), numpy.median(temp)))
        self.logger.info("Adjusted-R2: {:02.2f}  Outgroup Adjusted-R2:{:02.2f}".format(
                         100 * self.R2adj, best_R2adj_l*100))
        self.logger.info("Unique CREs in pairings: {}".format(
                         numpy.unique(self.pairs[numpy.where(self.selected)[0], 1]).shape[0]))

    def find_Xs(self):
        if self.skip_cres:
            XCs = numpy.zeros((self.tssN, self.cellN, 0), dtype=numpy.float32)
            return XCs
        else:
            XCs = numpy.zeros((self.tssN, self.cellN, self.stateN),
                              dtype=numpy.float32)
        for i in range(self.tssN):
            s = self.TSS_indices[i]
            e = self.TSS_indices[i + 1]
            valid = numpy.where(self.selected[s:e])[0] + s
            if valid.shape[0] > 0:
                XCs[i, :, :] = numpy.sum(self.Cstates[self.pairs[valid, 1], :, :], axis=0)
        XCs = XCs / numpy.maximum(1e-5, numpy.sum(XCs, axis=2, keepdims=True))
        return XCs

    def reconstitute_pairs(self, fname):
        if self.verbose >= 2:
            print("\r{}\rReconstituting TSS-cCRE pairs".format(' ' * 80), end='', file=sys.stderr)
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
        if self.verbose >= 2:
            print("\r{}\r".format(' ' * 80), end='', file=sys.stderr)

    def predict_lessone(self):
        XCs = self.find_Xs()
        X = numpy.dstack((self.XPrs, XCs[:, self.lmask, :]))
        X_l = numpy.hstack((self.XPrs_l, XCs[:, self.lessone, :]))
        Y = self.rna['rna'][:, self.lmask]
        Y_l = self.rna['rna'][:, self.lessone]
        betas = numpy.linalg.lstsq(X.reshape(Y.size, -1, order='C'),
                                   Y.reshape(-1, order='C'), rcond=None)[0]
        Prbetas = betas[self.promoter_beta_indices]
        Cbetas = betas[self.cre_beta_indices]
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
        output = open('{}_statistics.txt'.format(self.out_prefix), 'w')
        if self.skip_cres:
            print("R2adj\tMSE\tOut_R2adj\tOut_MSE", file=output)
            print("\t".join(R2adj*100., MSE, R2adj_l*100., MSE_l), file=output)
        else:
            temp = numpy.bincount(self.pairs[:, 0], weights=self.selected, minlength=self.tssN)
            print("R2adj\tMSE\tOut_R2adj\tOut_MSE\t#Retained\t#Possible\t%Retained\tMedian CREs/TSS",
                  file=output)
            print('\t'.join([str(x) for x in [R2adj*100., MSE, R2adj_l*100., MSE_l,
                                              numpy.sum(self.selected),
                                              self.selected.shape[0],
                                              numpy.sum(self.selected)
                                              / float(self.selected.shape[0]),
                                              numpy.median(temp)]]), file=output)
        output.close()
        output = open('{}_betas.txt'.format(self.out_prefix), 'w')
        header = ['state']
        if not self.skip_promoter:
            header.append('promoter_betas')
        if not self.skip_cres:
            header.append('cre_betas')
        print('\t'.join(header), file=output)
        for i in range(self.stateN):
            temp = [str(i)]
            if not self.skip_promoter:
                temp.append(str(Prbetas[i]))
            if not self.skip_cres:
                temp.append(str(Cbetas[i]))
            print('\t'.join(temp), file=output)
        output.close()
        output = gzip.open('{}_predicted_expression.txt.gz'.format(self.out_prefix), 'wb')
        strand_d = {True: '-', False: '+'}
        for i in range(self.tssN):
            output.write(bytes("{}\n".format('\t'.join([str(self.rna['chr'][i]), str(self.rna['TSS'][i]),
                                                        '.', str(self.rna['TSS'][i] + 1), str(y_l[i]),
                                                        strand_d[self.rna['strand'][i]]])), 'utf-8'))
        output.close()

        if not self.skip_cres:
            dtype = [('chr', self.chroms.dtype), ('TSS', numpy.int32), ('strand', numpy.bool),
                     ('cCRE', numpy.int32, (2,)), ('eRP', numpy.float32)]
            eRP = numpy.zeros(numpy.sum(self.selected), dtype=numpy.dtype(dtype))
            pos = 0
            for i in range(self.tssN):
                s = self.TSS_indices[i]
                e = self.TSS_indices[i + 1]
                if e - s == 0:
                    continue
                where = numpy.where(self.selected[s:e])[0] + s
                where2 = numpy.arange(pos, pos + where.shape[0])
                if where.shape[0] == 0:
                    continue
                Cstates = self.Cstates[self.pairs[where, 1], self.lessone, :]
                eRP['eRP'][where2] = numpy.sum(Cstates * Cbetas, axis=1)
                eRP['chr'][where2] = self.rna['chr'][i]
                eRP['TSS'][where2] = self.rna['TSS'][i]
                eRP['strand'][where2] = self.rna['strand'][i]
                eRP['cCRE'][where2, 0] = self.cre['start'][self.pairs[where, 1]]
                eRP['cCRE'][where2, 1] = self.cre['end'][self.pairs[where, 1]]
                pos += where.shape[0]
            numpy.save('{}_eRP.npy'.format(self.out_prefix), eRP)

            output = gzip.open('{}_eRP.txt.gz'.format(self.out_prefix), 'wb')
            output.write(b"Chr\tTSS\tStrand\tcCREstart\tcCREstop\teRP\n")
            for i in range(eRP.shape[0]):
                output.write(bytes("{}\n".format('\t'.join([str(eRP['chr'][i]),
                                                            str(eRP['TSS'][i]),
                                                            strand_d[eRP['strand'][i]],
                                                            str(eRP['cCRE'][i, 0]),
                                                            str(eRP['cCRE'][i, 1]),
                                                            str(eRP['eRP'][i])])), 'utf-8'))
            output.close()

    def write_settings(self):
        output = open('{}_settings.txt'.format(self.out_prefix), 'w')
        print("lessone = {}".format(self.lessone), file=output)
        print("states shuffled: {}".format(self.shufflestates), file=output)
        if not self.skip_cres:
            print("initialization_dist = {}".format(self.initialization_dist), file=output)
            print("cre_dist = {}".format(self.cre_dist), file=output)
            print("corr_cutoff = {}".format(self.corr_cutoff), file=output)
            print("refining_iter = {}".format(self.refining_iter), file=output)
            if self.skip_cre_promoter:
                print("skip_cre_promoter = True", file=output)
            print("maxcres = {}".format(self.maxcres), file=output)
        print("seed = {}".format(self.seed), file=output)
        output.close()

    def find_TSS_ranges(self):
        if self.skip_cre_promoter:
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
                                                    self.rna['TSS'][s:e] - self.cre_dist) + s1
            TSS_ranges[s:e, -1] = numpy.searchsorted(self.cre['start'][s1:e1],
                                                    self.rna['TSS'][s:e] + self.cre_dist) + s1
            if self.skip_cre_promoter:
                TSS_ranges[s:e, 1] = numpy.searchsorted(
                    self.cre['end'][s1:e1], self.rna['TSS'][s:e] - self.promoter_dist) + s1
                    #* numpy.logical_not(self.rna['strand'][s:e])) + s1
                TSS_ranges[s:e, 2] = numpy.searchsorted(
                    self.cre['start'][s1:e1], self.rna['TSS'][s:e] + self.promoter_dist) + s1
                    #* self.rna['strand'][s:e]) + s1
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
                Pstates2[:, i, :] = numpy.sum(Pstates[:, s:e, :], axis=1) / (e - s)
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
                Cstates[i - start, :, 0] = (self.cre['end'][i] - self.cre['start'][i]
                                             - numpy.sum(Cstates[i - start, :, 1:], axis=1))
            Cstates2 = numpy.zeros((stop - start, self.cellN, self.stateN), dtype=numpy.int32)
            for i in range(self.cellN):
                s = self.state_cindices[i]
                e = self.state_cindices[i + 1]
                Cstates2[:, i, :] = numpy.sum(Cstates[:, s:e, :], axis=1) // (e - s)
            #Cstates2 /= numpy.sum(Cstates2, axis=2, keepdims=True)
            if self.shufflestates:
                for i in range(stop - start):
                    for j in range(self.cellN):
                        Cstates2[i, j, :] = Cstates2[i, j, order]
                        rng.shuffle(order)
            out_q.put((start, stop, Cstates2))
        out_q.put(None)

    def _find_correlations(self, in_q, out_q):
        betas = self.initial_betas[:self.Cstates.shape[2]].reshape(1, 1, -1)
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
                if self.skip_cre_promoter:
                    Cindices = numpy.r_[numpy.arange(ranges[i0, 0], ranges[i0, 1]),
                                        numpy.arange(ranges[i0, 2], ranges[i0, 3])]
                else:
                    Cindices = numpy.arange(ranges[i0, 0], ranges[i0, 1])
                Cstates = self.Cstates[Cindices, :, :][:, self.lmask, :]
                cbetas = numpy.sum(betas * Cstates, axis=2)
                cbetas -= numpy.mean(cbetas, axis=1, keepdims=True)
                cbetas /= ((numpy.std(cbetas, axis=1, keepdims=True, ddof=1) + 1e-5)
                           * (self.cellN - 1) ** 0.5)
                corrs = numpy.dot(self.norm_rna[i:(i + 1), :], cbetas.T)[0, :]
                results.append((i, corrs))
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

    def _refine_pair_CREs(self, in_q, out_q, Prbetas, Cbetas, iteration, seed):
        rng = numpy.random.RandomState(seed=seed)
        #change_cutoff = numpy.round(self.refining_iter / float(iteration + 1)) + 1
        if iteration == 0:
            change_cutoff = numpy.inf
        else:
            change_cutoff = numpy.floor(1 - numpy.log(iteration / float(self.refining_iter - 1)) / numpy.log(1.5))
        while True:
            data = in_q.get(True)
            if data is None:
                break
            index, total_mse, Y = data
            s = self.TSS_indices[index]
            e = self.TSS_indices[index + 1]
            XPrs = self.XPrs[index, :, :]
            Cstates = self.Cstates[self.pairs[s:e, 1], :, :][:, self.lmask, :]
            selected = numpy.copy(self.selected[s:e])
            if len(Prbetas) > 0:
                base_exp = numpy.sum(Prbetas * XPrs, axis=1)
            else:
                base_exp = numpy.zeros(self.cellN1, dtype=numpy.float32)
            mse = numpy.zeros(Cstates.shape[0], dtype=numpy.float32)
            XC = numpy.sum(Cstates[selected, :, :], axis=0).astype(numpy.float32)
            XC_total = numpy.maximum(1e-5, numpy.sum(XC, axis=1).astype(numpy.float32))
            XC_sum = numpy.sum(XC * Cbetas, axis=1)
            mask = numpy.copy(selected)
            for i in range(mask.shape[0]):
                if not mask[i]:
                        XCs = ((XC_sum + numpy.sum(Cstates[i, :, :] * Cbetas, axis=1))
                              / (XC_total + numpy.sum(Cstates[i, :, :], axis=1)))
                else:
                        XCs = ((XC_sum - numpy.sum(Cstates[i, :, :] * Cbetas, axis=1))
                              / numpy.maximum(1e-5, XC_total - numpy.sum(Cstates[i, :, :], axis=1)))
                pred_exp = base_exp + XCs
                mse[i] = numpy.sum((Y - pred_exp) ** 2)
            change = numpy.where(mse < total_mse)[0]
            if change.shape[0] > 0:
                changed = True
                if change.shape[0] <= change_cutoff:
                    selected[change] = numpy.logical_not(selected[change])
                else:
                    weights = numpy.maximum(1e-5, (total_mse - mse[change]) / total_mse)
                    weights /= numpy.sum(weights)
                    weights = weights ** (1 + 9 * iteration / max(1, self.refining_iter - 1))
                    weights = (numpy.full(weights.shape[0], 1e-5, dtype=numpy.float32) +
                               (1 - 1e-5 * weights.shape[0]) * weights / numpy.sum(weights))
                    change = rng.choice(change, int(change_cutoff), p=weights,
                                        replace=False)
                    selected[change] = numpy.logical_not(selected[change])
                selectedN = numpy.sum(selected)
                if self.maxcres > 0 and selectedN > self.maxcres:
                    where = numpy.where(selected & (mse > total_mse))[0]
                    if where.shape[0] < self.maxcres - selectedN:
                        weights = numpy.maximum(1e-5, (mse[where] - total_mse) / total_mse)
                        weights /= numpy.sum(weights)
                        weights = weights ** (1 + 9 * iteration / max(1, self.refining_iter - 1))
                        weights = (numpy.full(weights.shape[0], 1e-5, dtype=numpy.float32) +
                                   (1 - 1e-5 * weights.shape[0]) * weights
                                   / numpy.sum(weights))
                        change = rng.choice(where, selectedN - self.maxcres,
                                            p=weights, replace=False)
                        selected[change] = False
                    else:
                        selected[where] = False
                        selectedN = numpy.sum(selected)
                        if selectedN > self.maxcres:
                            where = numpy.where(selected)[0]
                            weights = numpy.maximum(1e-5, (total_mse - mse[where]) / total_mse)
                            weights /= numpy.sum(weights)
                            weights = weights ** (1 + 9 * iteration / max(1, self.refining_iter - 1))
                            weights = (numpy.full(weights.shape[0], 1e-5, dtype=numpy.float32) +
                                       (1 - 1e-5 * weights.shape[0]) * weights
                                       / numpy.sum(weights))
                            change = rng.choice(where, selectedN - self.maxcres,
                                                p=weights, replace=False)
                            selected[change] = False
            else:
                changed = False
            out_q.put((index, changed, selected))
        out_q.put(None)


if __name__ == "__main__":
    main()
