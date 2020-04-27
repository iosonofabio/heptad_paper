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

    ds.counts.normalize('counts_per_ten_thousand', inplace=True)

    print('Look at a few interesting genes')
    ds.counts.log(inplace=True)
    std = ds.featuresheet['exp_std'] = ds.counts.std(axis=1)

    print('Feature selection')
    features = ds.feature_selection.overdispersed()
    dsf = ds.query_features_by_name(features)

    print('PCA')
    dsc = dsf.dimensionality.pca(n_dims=30, transform=None, return_dataset='samples')
    ds.counts.unlog(inplace=True)

    print('Similarity graph')
    edges = dsc.graph.knn(axis='samples', n_neighbors=2, return_kind='edges')

    print('Average with neighbors')
    ds2 = ds.copy()
    ds2.counts.smoothen_neighbors(edges, 'samples', inplace=True)

    tfs = ['ERG', 'FLI1', 'LMO2', 'GATA2', 'RUNX1', 'LYL1', 'TAL1']

    #TODO: look at GATA1 fighting GATA2 and maybe resurrecting ERG in the process

    ind = ds.featurenames[ds.featuresheet['GeneName'].isin(tfs)]
    dst = ds2.query_features_by_name(ind)
    dst.reindex('features', 'GeneName', inplace=True)

    print('Just try hierarchical clustering first')
    mat = dst.counts.T
    g = sns.clustermap(
        mat,
        method='average',
        #z_score=1,
        xticklabels=True,
        yticklabels=False,
        )
    fig = g.fig

    print('Correlations')
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, leaves_list
    corr = dst.correlation.correlate_features_features().fillna(0)
    cdis = (1.0 - corr).values
    # Numerical error
    cdis[np.arange(len(cdis)), np.arange(len(cdis))] = 0
    pdis = squareform(cdis)
    z = linkage(pdis, 'average', optimal_ordering=True)
    ll = leaves_list(z)
    mat = corr.iloc[ll].T.iloc[ll].T
    fig, ax = plt.subplots()
    sns.heatmap(mat, ax=ax)
    fig.tight_layout()


    corrh = corr.copy()
    for i in range(len(corr)):
        corrh.iloc[i, i] = 0
    tmp = np.vstack(np.unravel_index(corrh.values.ravel().argsort()[::-1], corrh.shape)).T
    args = []
    for pair in tmp:
        if list(pair[::-1]) not in args:
            args.append(list(pair))

    npairs = 21
    fig, axs = plt.subplots(3, npairs // 3, figsize=(3 * npairs // 3, 8))
    axs = axs.ravel()
    args_noself = [a for a in args if a[0] != a[1]]
    for i in range(npairs):
        ax = axs[i]
        i1, i2 = args_noself[i]
        x = dst.counts.iloc[i1].values
        y = dst.counts.iloc[i2].values
        ax.scatter(x, y, s=30, zorder=10, alpha=0.1)
        #sns.kdeplot(x, y, ax=ax)

        # Conditional curve
        xmax = np.sort(x)[-2]
        xbins_left = np.array([0] + list(np.linspace(0.1, xmax, 11)[:-2]))
        xbins_right = np.array([0.01] + list(np.linspace(0.1, xmax, 11)[2:]))
        xbins_center = 0.5 * (xbins_left + xbins_right)
        #xbins = np.vstack([xbins_left, xbins_right]).T
        yc = np.zeros(len(xbins_left))
        dyc = np.zeros(len(xbins_left))
        for i in range(len(xbins_left)):
            tmp = y[(x >= xbins_left[i]) & (x < xbins_right[i])]
            if len(tmp) == 0:
                yc[i] = np.nan
                dyc[i] = np.nan
            else:
                yc[i] = tmp.mean()
                dyc[i] = tmp.std()

        g1 = dst.featurenames[i1]
        g2 = dst.featurenames[i2]
        label = '$\\rho = {:.2f}$'.format(corr.at[g1, g2])
        ax.plot(xbins_center, yc, lw=2, color='darkred', label=label)
        ax.fill_between(xbins_center, yc - dyc, yc + dyc, color='darkred', alpha=0.1)
        ax.set_xlabel(g1)
        ax.set_ylabel(g2)
        ax.legend(fontsize=8)
    fig.tight_layout()


    plt.ion()
    plt.show()
