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
import anndata



if __name__ == '__main__':

    fn = '../../data/external/Palantir/human_cd34_bm_rep1.h5ad'
    an = anndata.read_h5ad(fn)

    genes = an.var_names
    cells = an.obs_names

    counts = singlet.CountsTable(
        data=an.X.T,
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

    print('Get MAGIC smoothed data')
    counts = singlet.CountsTable(
        data=an.obsm['MAGIC_imputed_data'].T,
        index=genes,
        columns=cells,
    )
    dsM = singlet.Dataset(
            counts_table=counts,
            samplesheet=ss,
        )

    print('Plot t-SNEs from their metadata')
    genes = [
            'CD34', 'GATA1', 'GATA3',
            'ERG', 'FLI1', 'LMO2',
            'GATA2', 'RUNX1', 'LYL1', 'TAL1']
    markers = ['clusters', 'palantir_pseudotime', 'palantir_diff_potential']
    cmaps = {
        #'clusters': an.uns['cluster_colors'],
        'clusters': {
            '0': 'deeppink',
            '1': 'pink',
            '2': 'lawngreen',
            '3': 'red',
            '4': 'purple',
            '5': 'tan',
            '6': 'tomato',
            '7': 'lightseagreen',
            '8': 'forestgreen',
            '9': 'orange',
            },
        'palantir_pseudotime': 'plasma',
        'palantir_diff_potential': 'plasma',
        }

    dsp = {'original': ds, 'smoothed': dsM}
    for title, dsi in dsp.items():
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(11, 15)
        axs = []
        axs.append(fig.add_subplot(gs[0:5, 0:5]))
        axs.append(fig.add_subplot(gs[0:5, 5:10]))
        axs.append(fig.add_subplot(gs[0:5, 10:15]))
        axs.append(fig.add_subplot(gs[5:8, 0:3]))
        axs.append(fig.add_subplot(gs[5:8, 3:6]))
        axs.append(fig.add_subplot(gs[5:8, 6:9]))
        axs.append(fig.add_subplot(gs[5:8, 9:12]))
        axs.append(fig.add_subplot(gs[5:8, 12:15]))
        axs.append(fig.add_subplot(gs[8:, 0:3]))
        axs.append(fig.add_subplot(gs[8:, 3:6]))
        axs.append(fig.add_subplot(gs[8:, 6:9]))
        axs.append(fig.add_subplot(gs[8:, 9:12]))
        axs.append(fig.add_subplot(gs[8:, 12:15]))

        mpg = markers + genes
        for i in range(len(axs)):
            ax = axs[i]
            gene = mpg[i]
            dsi.plot.scatter_reduced(
                ('tsne_1', 'tsne_2'),
                color_by=gene,
                # It's already logged
                color_log=False,
                cmap=cmaps.get(gene, 'viridis'),
                ax=ax,
                alpha=0.2,
                s=15,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(gene)
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        #fig.savefig('../../figures/Palantir_tsne_{:}.png'.format(title))

    plt.ion()
    plt.show()
