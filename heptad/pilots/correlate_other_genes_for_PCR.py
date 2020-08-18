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


def load_our_data():
    ds = singlet.Dataset(dataset={
        'path': '../../data/sequencing/me1/with_gene_names.loom',
        'index_samples': 'CellID',
        'index_features': 'GeneName'},
        )

    ds.samplesheet['coverage'] = ds.counts.sum(axis=0)
    ds.samplesheet['n_genes'] = (ds.counts >= 1).sum(axis=0)
    ds.featuresheet['exp_avg'] = ds.counts.mean(axis=1)

    return ds


if __name__ == '__main__':

    print('Read loom raw file')
    ds = load_our_data()
    ds.counts.normalize('counts_per_ten_thousand', inplace=True)

    cache = pd.read_csv(
        '../../data/sequencing/me1/northstar_with_Palantir.tsv',
        sep='\t',
        index_col=0)
    vs = cache.iloc[:, :2]
    northstar_assignment = cache['northstar_assignment'].loc[ds.samplenames]

    ct1, ct2 = ['Ery-precursor', 'HSC']
    comp = pd.read_csv(
        '../../data/gene_lists/DEG_allgenes_{:}_{:}.tsv'.format(ct1, ct2),
        sep='\t',
        index_col=0,
        )

    ds.samplesheet['cellSubtype'] = northstar_assignment
    dsa = ds.average('samples', 'cellSubtype')
    tfs = ['ERG', 'FLI1', 'LMO2', 'GATA2', 'RUNX1', 'LYL1', 'TAL1']
    data = dsa.counts.loc[tfs, [ct2, ct1]]

    print('Heatmap')
    fig, ax = plt.subplots(figsize=(2, 3))
    sns.heatmap(data, ax=ax)
    fig.tight_layout()

    dsp = ds.split('cellSubtype')
    st1 = dsp[ct1].counts.get_statistics(metrics=['mean', 'std'])
    st2 = dsp[ct2].counts.get_statistics(metrics=['mean', 'std'])
    st1['std'] /= np.sqrt(dsp[ct1].n_samples - 1)
    st2['std'] /= np.sqrt(dsp[ct2].n_samples - 1)

    print('Line plot')
    tfsp = ['GATA2', 'TAL1', 'ERG', 'FLI1', 'LMO2', 'RUNX1', 'LYL1']
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.3))
    ax = axs[0]
    cmap = dict(zip(tfs, sns.color_palette('Dark2', n_colors=len(tfs))))
    for itf, tf in enumerate(tfsp):
        x = [0, 1]
        y = np.array([st2.loc[tf, 'mean'], st1.loc[tf, 'mean']])
        dy = np.array([st2.loc[tf, 'std'], st1.loc[tf, 'std']])

        yp = np.log10(y)
        dyp = dy / y / np.log(10)

        ax.errorbar(x, yp, yerr=dyp, color=cmap[tf], lw=2, alpha=0.8, zorder=10 - itf)
        ax.scatter(x, yp, color=cmap[tf], lw=2, label=tf, alpha=0.8, zorder=10 - itf)
    ax.grid(False)
    ax.set_ylim(bottom=-1.3)
    ax.set_xlim(-0.1, 1.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([ct2, ct1])
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['$10^{-1}$', '$1$', '$10$'])
    ax.set_ylabel('Counts per ten\nthousand molecules')
    ax.legend(title='Heptad:', bbox_to_anchor=(1.01, 1.01), bbox_transform=ax.transAxes)

    ax = axs[1]
    genes = ['LMO4', 'GATA1', 'KLF1', 'MYBL2', 'ST18', 'ZEB1']
    cmap = dict(zip(genes, sns.color_palette('Dark2', n_colors=len(tfs))))
    for itf, tf in enumerate(genes):
        x = [0, 1]
        y = np.array([st2.loc[tf, 'mean'], st1.loc[tf, 'mean']])
        dy = np.array([st2.loc[tf, 'std'], st1.loc[tf, 'std']])

        yp = np.log10(y)
        dyp = dy / y / np.log(10)

        ax.errorbar(x, yp, yerr=dyp, color=cmap[tf], lw=2, alpha=0.8, zorder=10 - itf)
        ax.scatter(x, yp, color=cmap[tf], lw=2, label=tf, alpha=0.8, zorder=10 - itf)
    ax.grid(False)
    ax.set_ylim(bottom=-2.8)
    ax.set_xlim(-0.1, 1.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([ct2, ct1])
    ax.set_yticks([-2, -1, 0, 1])
    ax.set_yticklabels(['$10^{-2}$', '$10^{-1}$', '$1$', '$10$'])
    ax.set_ylabel('Counts per ten\nthousand molecules')
    ax.legend(title='DEGs:', bbox_to_anchor=(1.01, 1.01), bbox_transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig('../../figures/ME1_subpops_lineplots.png')

