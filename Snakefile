import gzip

SPECIES = ['hg38']#, 'mm10']
WC_SPECIES = "|".join(SPECIES)
STATE_FILES = {'hg38': "Data/hg38bp24bV2_2.state.gz",
               'mm10': "Data/pknorm_2_16lim_ref1mo_0424_lesshet.state.gz"}
RNA_FILES = {'hg38': "Data/HumanVISION_RNAseq_hg38_genes_tpm.txt.gz",
             'mm10': "Data/mm10_rna_1pc.txt.gz"}
CRE_FILES = {'hg38': "Data/hg38bp24b_V2_2.nonALL0.cCRE.bed.gz",
             'mm10': "Data/mm10_vision_CREs.txt.gz"}
HG38_GENE_IDS = "Data/hg38_ensembl_ids.txt.gz"
GENE_SETS = ['all', 'diff', 'nodiff']
WC_GENE_SETS = "|".join(GENE_SETS)
FEATURE_SETS = ['both', 'cre', 'promoter']
WC_FEATURE_SETS = "|".join(FEATURE_SETS)
NREPS = 50
REP_LIST = range(NREPS)
CELLTYPES = {}
LESSONE_DICT = {}
all_ct = []
for species in SPECIES:
  state_reps = gzip.open(STATE_FILES[species], 'rb').readline().split()[4:-1]
  rna_reps = gzip.open(RNA_FILES[species], 'rb').readline().split()[4:]
  state_celltypes = set([x.decode('utf8').split('_')[0] for x in state_reps])
  rna_celltypes = set([x.decode('utf8').split('_')[0] for x in rna_reps])

  CELLTYPES[species] = list(state_celltypes.intersection(rna_celltypes))
  CELLTYPES[species].sort()
  all_ct += CELLTYPES[species]
  LESSONE_DICT[species] = {}
  for i, ct in enumerate(CELLTYPES[species]):
    LESSONE_DICT[species][ct] = i
WC_CELLTYPES = "|".join(all_ct)

THREADS = 1
ITERATIONS = 30
INIT_DIST = 2000
PROMOTER_DIST = 1000
CRE_DIST = 1500000
CORR_CUTOFF = 0.5
MAX_CRES = 10


rule all:
  input:
    expand("Plots/{species}_{plot}.pdf", species=SPECIES, plot=['corrs', 'betas', 'reproducibility'])

######## Preprocess text files into numpy array files ########

rule preprocess_mm10_rna:
  input:
    RNA_FILES['mm10']
  output:
    "Data/mm10_rna_all.npy"
  conda:
    "envs/general.yaml"
  shell:
    "python3 Scripts/rna_to_npy.py -r {input} -o {output}"

rule add_ids_to_rna:
  input:
    rna=RNA_FILES['hg38'],
    ids=HG38_GENE_IDS
  output:
    "Data/hg38_annotated_rna.txt.gz"
  conda:
    "envs/general.yaml"
  shell:
    "python3 Scripts/rna_ids2type.py -r {input.rna} -i {input.ids} -o {output}"  

rule preprocess_hg38_rna:
  input:
    "Data/hg38_annotated_rna.txt.gz"
  output:
    "Data/hg38_rna_all.npy"
  conda:
    "envs/general.yaml"
  shell:
    "python3 Scripts/rna_to_npy.py -r {input} -o {output}"

rule split_rna:
  input:
    "Data/{species}_rna_all.npy"
  output:
    diff="Data/{species}_rna_diff.npy",
    nodiff="Data/{species}_rna_nodiff.npy",
    plot="Plot/{species}_rna_split.pdf"
  params:
    prefix=lambda wildcards: expand("Data/{species}_rna", species=wildcards.species)
  conda:
    "envs/general.yaml"
  shell:
    "python3 Scripts/split_rna.py -r {input} -o {params.prefix} -p {output.plot}"

rule preprocess_states:
  input:
    lambda wildcards: STATE_FILES[wildcards.species]
  output:
    "Data/{species}_states.npy"
  conda:
    "envs/general.yaml"
  shell:
    "python3 Scripts/state_to_npy.py -s {input} -o {output}"


######## Run training of model ########

def get_params(wc):
  params = {}
  params['--output'] = "Results_{}/{}_{}_{}_{}_{}".format(wc.species, wc.species, wc.genes,
                                                          wc.ct, wc.feat, wc.label)
  label = wc.label.split('_')
  if len(label) == 1:
    rep = label[0]
  else:
    rep, control = label
    params['--shuffle-states'] = ''
  params['--lessone'] = LESSONE_DICT[wc.species][wc.ct]
  params['--seed'] = rep
  if wc.feat == 'both':
    params['--promoter-dist'] = PROMOTER_DIST
    params['--cre-dist'] = CRE_DIST
  elif wc.feat == 'cre':
    params['--cre-dist'] = CRE_DIST
  elif wc.feat == 'promoter':
    params['--promoter-dist'] = PROMOTER_DIST
  params['--initialization-dist'] = INIT_DIST
  params['--iterations'] = ITERATIONS
  params['--threads'] = THREADS
  params['--correlation'] = CORR_CUTOFF
  params['--max-CREs'] = MAX_CRES
  params['--multi-refinement'] = ''
  params['--verbose'] = 1
  return ' '.join(["{} {}".format(k, v) for k, v in params.items()])

rule train_model:
  input:
    rna="Data/{species}_rna_{genes}.npy",
    states="Data/{species}_states.npy",
    cre=lambda wildcards: CRE_FILES[wildcards.species]
  output:
    betas="Results_{species}/{species}_{genes}_{ct}_{feat}_{label}_betas.txt",
    erp="Results_{species}/{species}_{genes}_{ct}_{feat}_{label}_eRP.txt.gz",
    expression="Results_{species}/{species}_{genes}_{ct}_{feat}_{label}_predicted_expression.txt.gz",
    statistics="Results_{species}/{species}_{genes}_{ct}_{feat}_{label}_statistics.txt",
    settings="Results_{species}/{species}_{genes}_{ct}_{feat}_{label}_settings.txt"
  wildcard_constraints:
    species=WC_SPECIES,
    genes=WC_GENE_SETS,
    ct=WC_CELLTYPES,
    feat='cre|both',
    label="\d+|\d+_control"
  params:
    get_params
  conda:
    "envs/general.yaml"
  shell:
    "python Scripts/GeneCREPrediction.py --rna {input.rna} --state {input.states} --cre {input.cre} {params}"


rule train_promoter_model:
  input:
    rna="Data/{species}_rna_{genes}.npy",
    states="Data/{species}_states.npy",
    cre=lambda wildcards: CRE_FILES[wildcards.species]
  output:
    betas="Results_{species}/{species}_{genes}_{ct}_{feat}_{label}_betas.txt",
    expression="Results_{species}/{species}_{genes}_{ct}_{feat}_{label}_predicted_expression.txt.gz",
    statistics="Results_{species}/{species}_{genes}_{ct}_{feat}_{label}_statistics.txt",
    settings="Results_{species}/{species}_{genes}_{ct}_{feat}_{label}_settings.txt"
  wildcard_constraints:
    species=WC_SPECIES,
    genes=WC_GENE_SETS,
    ct=WC_CELLTYPES,
    feat='promoter',
    label="\d+|\d+_control"
  params:
    get_params
  conda:
    "envs/general.yaml"
  shell:
    "python Scripts/GeneCREPrediction.py --rna {input.rna} --state {input.states} --cre {input.cre} {params}"


######## Plot figures ########

rule plot_correlations:
  input:
    treat=lambda wildcards: expand("Results_{species}/{species}_{genes}_{ct}_{feat}_0_statistics.txt", \
                                   species=wildcards.species, genes=GENE_SETS, \
                                   ct=CELLTYPES[wildcards.species], feat=FEATURE_SETS),
    control=lambda wildcards: expand("Results_{species}/{species}_{genes}_{ct}_{feat}_0_control_statistics.txt", \
                                     species=wildcards.species, genes=GENE_SETS, \
                                     ct=CELLTYPES[wildcards.species], feat=FEATURE_SETS)
  output:
    "Plots/{species}_corrs.pdf"
  params:
    prefix = lambda wildcards: expand("Results_{species}/{species}", species=wildcards.species),
    species = lambda wildcards: wildcards.species
  conda:
    "envs/general.yaml"
  shell:
    "python Scripts/plot_correlations.py {params.species} {params.prefix} {output}"

rule plot_betas:
  input:
    treat=lambda wildcards: expand("Results_{species}/{species}_{genes}_{ct}_{feat}_0_betas.txt", \
                                   species=wildcards.species, genes=GENE_SETS, \
                                   ct=CELLTYPES[wildcards.species], feat=FEATURE_SETS),
    control=lambda wildcards: expand("Results_{species}/{species}_{genes}_{ct}_{feat}_0_control_betas.txt", \
                                     species=wildcards.species, genes=GENE_SETS, \
                                     ct=CELLTYPES[wildcards.species], feat=FEATURE_SETS)
  output:
    "Plots/{species}_betas.pdf"
  params:
    prefix = lambda wildcards: expand("Results_{species}/{species}", species=wildcards.species),
    species = lambda wildcards: wildcards.species
  conda:
    "envs/general.yaml"
  shell:
    "python Scripts/plot_betas.py {params.species} {params.prefix} {output}"

rule plot_reproducibility:
  input:
    eRP=lambda wildcards: expand("Results_{species}/{species}_all_{ct}_both_{rep}_eRP.txt.gz", \
                                 species=wildcards.species, ct=CELLTYPES[wildcards.species], rep=REP_LIST),
    eRP_c=lambda wildcards: expand("Results_{species}/{species}_all_{ct}_both_{rep}_control_eRP.txt.gz", \
                                   species=wildcards.species, ct=CELLTYPES[wildcards.species], rep=REP_LIST),
    beta=lambda wildcards: expand("Results_{species}/{species}_all_{ct}_both_{rep}_betas.txt", \
                                  species=wildcards.species, ct=CELLTYPES[wildcards.species], rep=REP_LIST),
    beta_c=lambda wildcards: expand("Results_{species}/{species}_all_{ct}_both_{rep}_control_betas.txt", \
                                    species=wildcards.species, ct=CELLTYPES[wildcards.species], rep=REP_LIST),
    rna="Data/{species}_rna_all.npy",
    state="Data/{species}_states.npy",
    cre=lambda wildcards: CRE_FILES[wildcards.species]
  output:
    "Plots/{species}_reproducibility.pdf"
  params:
    prefix = lambda wildcards: expand("Results_{species}/{species}_all", species=wildcards.species),
    species = lambda wildcards: wildcards.species
  wildcard_constraints:
    rep="\d+"
  conda:
    "envs/general.yaml"
  shell:
    "python Scripts/plot_reproducibility.py {params.species} {params.prefix} {input.state} {input.rna} {input.cre} {output}"

