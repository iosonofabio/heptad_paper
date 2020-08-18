# vim: fdm=indent
'''
author:     Fabio Zanini
date:       27/04/20
content:    Evaluate Palantir data given the files on their repo.
'''
import os
import sys
import numpy as np
import scipy as sp
from collections import Counter
from scipy.io import mmread
import pandas as pd
import loompy

import matplotlib.pyplot as plt
import seaborn as sns


sys.path.append('/home/fabio/university/postdoc/singlet')
os.environ['SINGLET_CONFIG_FILENAME'] = 'singlet.yml'
import singlet
import anndata

# Add northstar
sys.path.insert(0, '/home/fabio/university/postdoc/northstar/build/lib')
import northstar


def load_palantir_data(smoothed=False):
    fn = '../../data/external/Palantir/human_cd34_bm_rep1.h5ad'
    an = anndata.read_h5ad(fn)

    genes = an.var_names
    cells = an.obs_names

    if not smoothed:
        counts = singlet.CountsTable(
            data=an.raw.X.todense().T,
            index=genes,
            columns=cells,
        )
    else:
        counts = singlet.CountsTable(
            data=an.obsm['MAGIC_imputed_data'].T,
            index=genes,
            columns=cells,
        )

    ss = singlet.SampleSheet(an.obs)
    ss['tsne_1'] = an.obsm['tsne'][:, 0]
    ss['tsne_2'] = an.obsm['tsne'][:, 1]
    ss['clusters'] = ss['clusters'].astype(str)

    ds = singlet.Dataset(
            counts_table=counts,
            samplesheet=ss,
        )
    return ds



if __name__ == '__main__':

    dsP = load_palantir_data(smoothed=False)
    dsP.counts.index.name = 'GeneName'
    dsP.counts.normalize('counts_per_ten_thousand', inplace=True)
    dsP.samplesheet['CellSubtype'] = dsP.samplesheet['clusters'].replace({
        '0': 'HSC',
        '1': 'HSC',
        '2': 'Ery-precursor',
        '3': 'Mono',
        '4': 'Mono-precursor',
        '5': 'CLP',
        '6': 'Mono',
        '7': 'pDC',
        '8': 'Ery',
        '9': 'Mega',
    })

    dsp = dsP.split('CellSubtype')
    ct1, ct2 = ['Ery-precursor', 'HSC']
    st1 = dsp[ct1].counts.get_statistics(metrics=['mean', 'std'])
    st2 = dsp[ct2].counts.get_statistics(metrics=['mean', 'std'])
    st1['std'] /= np.sqrt(dsp[ct1].n_samples - 1)
    st2['std'] /= np.sqrt(dsp[ct2].n_samples - 1)

    tfs = ['ERG', 'FLI1', 'LMO2', 'GATA2', 'RUNX1', 'LYL1', 'TAL1']
    dsa = dsP.average('samples', 'CellSubtype')
    tfsp = ['GATA2', 'TAL1', 'ERG', 'FLI1', 'LMO2', 'RUNX1', 'LYL1']
    csts = [
        'HSC',
        'Ery-precursor',
        'Ery',
        'Mono-precursor',
        'Mono',
        'pDC',
        'CLP',
        'Mega',
       ]
    data = dsa.counts.loc[tfsp, csts]
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(data, ax=ax)
    fig.tight_layout()
    fig.savefig('../../figures/Palantir_subpops_heatmap.png')

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
    fig.savefig('../../figures/Palantir_subpops_lineplots.png')


