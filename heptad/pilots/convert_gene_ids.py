# vim: fdm=indent
'''
author:     Fabio Zanini
date:       27/04/20
content:    Evaluate Palantir data given the files on their repo.
'''
import os
import sys
import numpy as np
from scipy.io import mmread
import pandas as pd
import loompy

import matplotlib.pyplot as plt
import seaborn as sns


sys.path.append('/home/fabio/university/postdoc/singlet')
os.environ['SINGLET_CONFIG_FILENAME'] = 'singlet.yml'
import singlet





if __name__ == '__main__':

    fdn = '../../data/sequencing/me1/'
    fn_dataset = fdn+'raw.loom'
    ds = singlet.Dataset(
        dataset={
            'path': fn_dataset,
            'index_samples': 'CellID',
            'index_features': 'EnsemblID',
            },
        )

    conv = pd.read_csv(
            '../../data/gene_ensemblId_name.tsv',
            sep='\t',
            index_col=0,
            squeeze=True,
            )

    print('Restrict to features with a gene name')
    gids = ds.featurenames
    gids = gids[gids.isin(conv.index)]
    ds.query_features_by_name(gids, inplace=True)

    gene_names = np.unique(conv.loc[gids])
    from collections import defaultdict
    gmap = defaultdict(list)
    for gid, gn in conv.items():
        if gid in gids:
            gmap[gn].append(gid)

    counts = np.zeros((len(gene_names), ds.shape[1]), np.float32)
    for i, gn in enumerate(gene_names):
        counts[i] += ds.counts.loc[gmap[gn]].sum(axis=0)
    counts = pd.DataFrame(
            counts,
            index=gene_names,
            columns=ds.samplenames,
        )

    dsg = singlet.Dataset(
        samplesheet=ds.samplesheet,
        counts_table=singlet.CountsTable(counts),
        )
