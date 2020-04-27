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

    def get_table(counts, quantiles, remove_diag=True):
        from itertools import product
        mis = {}
        for gene1, gene2 in product(tfs, tfs):
            if remove_diag and (gene1 == gene2):
                mis[(gene1, gene2)] = 0.0
                continue

            bins = [(0.0, quantiles[0])]
            for i in range(len(quantiles) - 1):
                bins.append((quantiles[i], quantiles[i+1]))
            bins.append((quantiles[-1], 1.0))

            d1 = np.log10(counts.loc[gene1] + 0.1)
            d2 = np.log10(counts.loc[gene2] + 0.1)

            labels1 = pd.Series(
                np.zeros(len(d1), np.int64),
                index=d1.index,
                )
            labels2 = pd.Series(
                np.zeros(len(d1), np.int64),
                index=d1.index,
                )
            labels = [labels1, labels2]
            for ig, (gene, d) in enumerate(zip([gene1, gene2], [d1, d2])):
                for ib, bini in enumerate(bins):
                    ql, qr = d.quantile(bini)
                    if ib == 0:
                        ql -= 0.01
                    idx = d1.index[(d > ql) & (d <= qr)]
                    labels[ig][idx] = ib

            from sklearn.metrics import mutual_info_score
            mi = mutual_info_score(
                    labels[0].values,
                    labels[1].values,
                    )
            mis[(gene1, gene2)] = mi
        mis = pd.Series(mis).unstack()
        return mis

    mis = get_table(dst.counts, [0.5])
    mis2 = get_table(dst.counts, [0.1, 0.9])
    mis3 = get_table(dst.counts, [0.1, 0.5, 0.9])

    def plot_mi(bins, ):
        from scipy.spatial.distance import pdist, squareform
        from scipy.cluster.hierarchy import linkage, leaves_list

        mis = get_table(dst.counts, bins)

        cdis = mis.values.max() - mis
        for i in range(cdis.shape[0]):
            cdis.iloc[i, i] = 0
        pdis = squareform(0.5 * (cdis + cdis.T).values)
        z = linkage(pdis, optimal_ordering=True)
        ind = leaves_list(z)

        mis_plot = mis.iloc[ind].T.iloc[ind].T
        fig, ax = plt.subplots(figsize=(4.85, 4.25))
        sns.heatmap(mis_plot, ax=ax)
        for tk in ax.get_yticklabels():
            tk.set_rotation(0)
        for tk in ax.get_xticklabels():
            tk.set_rotation(90)
        ax.set_title('Mutual information\nbins: {:}'.format(
            ', '.join(['{:.2f}'.format(x) for x in bins]),
            ))
        fig.tight_layout()

    def plot_avg_mis():
        from scipy.spatial.distance import pdist, squareform
        from scipy.cluster.hierarchy import linkage, leaves_list

        mis = None
        nbins = [2, 3, 4, 5]
        for nb in nbins:
            mi = get_table(dst.counts, np.linspace(0.1, 0.9, nb))
            if mis is None:
                mis = mi
            else:
                mis += mi / mi.values.max()
        mis /= len(nbins)

        cdis = mis.values.max() - mis
        for i in range(cdis.shape[0]):
            cdis.iloc[i, i] = 0
        pdis = squareform(0.5 * (cdis + cdis.T).values)
        z = linkage(pdis, optimal_ordering=True)
        ind = leaves_list(z)

        mis_plot = mis.iloc[ind].T.iloc[ind].T
        fig, ax = plt.subplots(figsize=(4.85, 4.25))
        sns.heatmap(mis_plot, ax=ax)
        for tk in ax.get_yticklabels():
            tk.set_rotation(0)
        for tk in ax.get_xticklabels():
            tk.set_rotation(90)
        ax.set_title('Mutual information\naverage over binnings')
        fig.tight_layout()

    plot_mi([0.1, 0.5, 0.9])

    plt.ion()
    plt.show()
