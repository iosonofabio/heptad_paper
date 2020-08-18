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


tfs = ['ERG', 'FLI1', 'LMO2', 'GATA2', 'RUNX1', 'LYL1', 'TAL1']


def load_kd():
    kd = pd.read_excel(
            '../../data/knockdown/ME1 shRNA Knockdown data for Fabio.xlsx',
            sheet_name='Flipped, ordered for Fabio')
    kd.set_index(kd.columns[0], inplace=True)
    kd.index.name = 'Expression'
    return kd


def load_enhancer_counts(rename=True):
    tfs = ['ERG', 'FLI1', 'LMO2', 'GATA2', 'RUNX1', 'LYL1', 'TAL1']
    enh = pd.read_excel('../../data/enhancer_counts/motif_count_heptad_elements.xlsx')
    enh.set_index('Element', inplace=True)
    if not rename:
        return enh
    enh['GATA2'] = enh['GATA_motifs']
    enh['RUNX1'] = enh['RUNX_motifs']
    enh['ERG'] = enh['ETS_motifs']
    enh['FLI1'] = enh['ETS_motifs']
    enh['TAL1'] = enh['Ebox_motifs']
    enh['LYL1'] = enh['Ebox_motifs']
    enh['LMO2'] = enh['Ebox_motifs']
    enh = enh[tfs]
    enh.columns.name = 'Motif'
    return enh


if __name__ == '__main__':

    fn_ds = '../../data/sequencing/me1/normalized_7tfs.loom'
    if not os.path.isfile(fn_ds):
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

        ds.counts.normalize('counts_per_ten_thousand', inplace=True)

        tmp = ds.featuresheet.loc[ds.featuresheet['GeneName'].isin(tfs), 'GeneName']
        dic = {val: key for key, val in tmp.items()}
        idx = [dic[val] for val in tfs]
        dst = ds.query_features_by_name(idx)
        dst.reindex('features', 'GeneName', inplace=True)
        dst.to_dataset_file(fn_ds)
    else:
        dst = singlet.Dataset(dataset={
            'path': fn_ds,
            'index_samples': 'CellID',
            'index_features': 'GeneName',
            })

    medians = dst.counts.median(axis=1)
    counts_binary = (dst.counts.T > medians)
    p1 = counts_binary.mean(axis=0)

    gene1 = 'FLI1'
    pr2 = counts_binary.groupby(gene1).mean()

    gene2 = pr2.diff().iloc[1].idxmax()
    pr3 = counts_binary.groupby([gene1, gene2]).mean()

    gene3 = pr3.diff().loc[(True, True)].idxmax()
    pr4 = counts_binary.groupby([gene1, gene2, gene3]).mean()

    def pr2fun(gene1):
        return counts_binary.groupby(gene1).mean()

    def plot_violin(counts, reg, tgt, quantiles=2):
        from scipy.stats import gaussian_kde
        import matplotlib.transforms as transforms

        if isinstance(quantiles, int):
            ncats = quantiles
            quantiles = np.linspace(0, 1, ncats+1)
            bins = [(quantiles[i], quantiles[i+1]) for i in range(ncats)]
        elif isinstance(quantiles[0], float):
            quantiles = np.array([0.0] + list(quantiles) + [1.0])
            ncats = len(quantiles) - 1
            bins = [(quantiles[i], quantiles[i+1]) for i in range(ncats)]
        else:
            ncats = len(quantiles)
            bins = quantiles

        x = np.linspace(-1, 4, 100)
        y0 = np.zeros_like(x)

        d1 = np.log10(counts.loc[reg] + 0.1)
        k1 = gaussian_kde(d1.values)
        y1 = k1(x)

        cuts = []
        rects = []
        for le, re in bins:
            ql, qr = d1.quantile((le, re))
            rects.append((ql, qr))
            if len(cuts) == 0:
                ql -= 0.01
            cut = d1.index[(d1 > ql) & (d1 <= qr)]
            cuts.append(cut)

        y2s = []
        d2 = np.log10(counts.loc[tgt] + 0.1)
        bins_plot = []
        rects_plot = []
        d2s = []
        for i in range(ncats):
            d2i = d2[cuts[i]].values
            if len(d2i) < 5:
                continue
            k2i = gaussian_kde(d2i)
            y2i = k2i(x)
            d2s.append(d2i)
            y2s.append(y2i)
            bins_plot.append(bins[i])
            rects_plot.append(rects[i])

        if (rects_plot[0][0] == -1) and (rects_plot[0][1] == -1):
            rects_plot[0] = (-1.1, rects_plot[0][1])

        # Calculate KS test between first and last distro
        from scipy.stats import ks_2samp
        x1 = np.sort(d2s[0])
        x2 = np.sort(d2s[-1])
        pval = ks_2samp(x1, x2)[1]

        if pval > 0.05:
            return None

        ncats_plot = len(bins_plot)

        if ncats_plot < 2:
            return None

        cmap = plt.cm.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, ncats_plot))

        fig = plt.figure(figsize=(4, 1.2 + 1 * ncats_plot))
        gs = fig.add_gridspec(
                1 + ncats_plot, 1,
                height_ratios=[1.5] + ([1] * ncats_plot),
                )
        axs = []
        axs.append(fig.add_subplot(gs[0]))
        axs.append(fig.add_subplot(gs[1], sharex=axs[0]))
        for i in range(ncats_plot - 1):
            axs.append(fig.add_subplot(gs[i+2], sharex=axs[0], sharey=axs[1]))

        ax = axs[0]
        ax.fill_between(x, y0, y1, color='steelblue', alpha=0.5)
        for i in range(len(bins_plot)):
            r = plt.Rectangle(
                (rects_plot[i][0], 0), rects_plot[i][1] - rects_plot[i][0], 1,
                color=colors[i],
                alpha=0.3,
                transform=transforms.blended_transform_factory(
                    ax.transData, ax.transAxes),
                )
            ax.add_artist(r)
        ax.grid(True)
        ax.set_title(reg, fontsize=10)
        ax.set_yticklabels([])
        ax.text(0.985, 0.89, 'P = {:.1e}'.format(pval),
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.8, lw=1.5),
                )
        axs[1].set_title(tgt, fontsize=10)
        for i in range(ncats_plot):
            y2 = y2s[i]
            color = colors[i]
            ax = axs[i+1]
            ax.fill_between(x, y0, y2, color=color, alpha=0.4)
            ax.grid(True)
            ax.set_yticklabels([])

        axs[-1].set_xticks([-1, 0, 1, 2, 3, 4])
        axs[-1].set_xticks([-0.5, 0.5, 1.5, 2.5, 3.5], minor=True)
        axs[-1].set_xticklabels(['$0$', '$1$', '$10$', '$10^2$', '$10^3$', '$10^4$'])
        axs[-1].set_xlabel('Gene expression [cpm]')

        fig.tight_layout(h_pad=0.1)
        return {'fig': fig, 'pval': pval}

    from itertools import product
    for gene1, gene2 in product(tfs, tfs):
        if gene1 == gene2:
            continue
        #plot_violin(dst.counts, gene1, gene2, quantiles=3)
        res = plot_violin(dst.counts, gene1, gene2, quantiles=[[0.0, 0.1], [0.95, 1.0]])
        if res is None:
            continue

        if res['pval'] < 0.01:
            sfdn = 'great'
        else:
            sfdn = 'meh'

        fig = res['fig']
        #fig.savefig('../../figures/shared/distros/{:}/{:}_affecting_{:}.png'.format(
        #    sfdn, gene1, gene2),
        #    )

    plt.ion()
    plt.show()
