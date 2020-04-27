# vim: fdm=indent
'''
author:     Fabio Zanini
date:       10/12/19
content:    Check expression of 7 genes.
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


    #fdn = '../../data/sequencing/me1/'
    #fn_cells = fdn+'barcodes.tsv'
    #fn_genes = fdn+'genes.tsv'
    #fn_counts = fdn+'matrix.mtx'

    #print('Reading raw count files')
    #df_genes = pd.read_csv(fn_genes, sep='\t', index_col=0, header=None).iloc[:, 0]
    #df_cells = pd.read_csv(fn_cells, sep='\t', header=None).values[:, 0]
    #matrix = mmread(fn_counts).todense()

    #print('Store as loom for speed')
    #fn_dataset = fdn+'raw.loom'
    #loompy.create(
    #    fn_dataset,
    #    layers={'': matrix},
    #    col_attrs={'CellID': df_cells},
    #    row_attrs={'GeneName': df_genes.values, 'EnsemblID': df_genes.index.values},
    #    )


    print('Read loom raw file')
    fdn = '../../data/sequencing/me1/'
    fn_dataset = fdn+'raw.loom'
    ds = singlet.Dataset(
        dataset={
            'path': fn_dataset,
            'index_samples': 'CellID',
            'index_features': 'EnsemblID',
            },
        )

    ds.samplesheet['coverage'] = ds.counts.sum(axis=0)
    ds.samplesheet['n_genes'] = (ds.counts >= 1).sum(axis=0)
    ds.featuresheet['exp_avg'] = ds.counts.mean(axis=1)

    cov = ds.samplesheet['coverage']
    ngenes = ds.samplesheet['n_genes']
    avg = ds.featuresheet['exp_avg']

    print('Plot QC')
    fig, axs = plt.subplots(1, 2, figsize=(5, 2.5))
    ax = axs[0]
    x = np.sort(cov.values)
    y = 1.0 - np.linspace(0, 1, len(x))
    ax.plot(x, y, lw=2)
    ax.set_xlabel('N molecules')
    ax.set_ylabel('Fraction of cells with\nmore than x molecules')

    ax = axs[1]
    x = np.sort(ngenes.values)
    y = 1.0 - np.linspace(0, 1, len(x))
    ax.plot(x, y, lw=2)
    ax.set_xlabel('N genes [1+ molecules]')
    ax.set_ylabel('Fraction of cells with\nmore than x genes')

    fig.tight_layout()


    print('Look at a few interesting genes')
    ds.counts.log(inplace=True)
    std = ds.featuresheet['exp_std'] = ds.counts.std(axis=1)

    print('Feature selection')
    features = ds.feature_selection.overdispersed()
    dsf = ds.query_features_by_name(features)

    print('PCA')
    dsc = dsf.dimensionality.pca(n_dims=30, transform=None, return_dataset='samples')

    print('t-SNE')
    vs = dsc.dimensionality.tsne(perplexity=30)
    ds.counts.unlog(inplace=True)

    print('Plot')
    fig, axs = plt.subplots(3, 4, figsize=(9, 9))
    axs = axs.ravel()
    genes = ['coverage', 'n_genes', 'MKI67', 'MCM7', 'CKS2', 'ERG', 'FLI1', 'LMO2', 'GATA2', 'RUNX1', 'LYL1', 'TAL1']
    for ig, gene in enumerate(genes):
        ax = axs[ig]
        if gene in ds.samplesheet.columns:
            gid = 'n_genes'
            clog = False
        else:
            gid = ds.featurenames[ds.featuresheet['GeneName'] == gene][0]
            clog = True

        ds.plot.scatter_reduced(
            vs,
            color_by=gid,
            cmap='viridis',
            color_log=clog,
            ax=ax,
            alpha=0.8,
            s=20,
            )
        ax.set_axis_off()
        ax.set_title(gene)
    fig.tight_layout()

    plt.ion()
    plt.show()
