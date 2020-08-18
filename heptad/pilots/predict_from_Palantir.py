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

    dsP = load_palantir_data(smoothed=False)
    dsP.counts.normalize('counts_per_ten_thousand', inplace=True)
    dsP.samplesheet['cellSubtype'] = dsP.samplesheet['clusters'].replace({
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

    #print('Subsample their data')
    #dsPsub = dsP.subsample(40, within_metadata='clusters')

    print('Load our data')
    ds = load_our_data()
    ds.counts.normalize('counts_per_ten_thousand', inplace=True)
    cache = pd.read_csv(
        '../../data/sequencing/me1/northstar_with_Palantir.tsv',
        sep='\t',
        index_col=0)
    northstar_assignment = cache['northstar_assignment']
    ds.samplesheet['cellSubtype'] = northstar_assignment

    print('Reduce to just HSC versus Ery-precursors')
    dsPi = dsP.query_samples_by_metadata('cellSubtype in ("HSC", "Ery-precursor")')
    dsi = ds.query_samples_by_metadata('cellSubtype in ("HSC", "Ery-precursor")')
    y_train = (dsPi.samplesheet['cellSubtype'] == 'Ery-precursor').values
    y_test = (dsi.samplesheet['cellSubtype'] == 'Ery-precursor').values

    print('Reduce to the heptad')
    tfs = ['ERG', 'FLI1', 'LMO2', 'GATA2', 'RUNX1', 'LYL1', 'TAL1']


    # Test each TF on its own
    aucs = []
    for gene in tfs:
        print('Study supervised learning with gene: {:}'.format(gene))

        dsPtf = dsPi.query_features_by_name([gene])
        dstf = dsi.query_features_by_name([gene])

        def rebalance(X, y):
            keep = y.copy()
            ind = np.random.choice((~y).nonzero()[0], size=y.sum(), replace=False)
            keep[ind] = True
            return X[keep], y[keep]

        X_train = dsPtf.counts.values.T
        X_test = dstf.counts.values.T

        X_train_reb, y_train_reb = rebalance(X_train, y_train)
        X_test_reb, y_test_reb = rebalance(X_test, y_test)

        from sklearn.metrics import f1_score, roc_auc_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        #model = LogisticRegression()
        model = RandomForestClassifier()

        tmp = []
        for i in range(10):
            model.fit(X_train, y_train)
            #model.fit(X_test_reb, y_test_reb)
            y_prob = model.predict_proba(X_test_reb)
            auc = roc_auc_score(y_test_reb, y_prob[:, 1])
            tmp.append(auc)
        auc = np.mean(auc)

        aucs.append({'gene': gene, 'auc': auc})

    hm = pd.DataFrame(aucs).set_index('gene')

    # All combinaitons of 3 TFs
    gene_lists = []
    for i1, g1 in enumerate(tfs):
        for i2, g2 in enumerate(tfs[:i1]):
            for i3, g3 in enumerate(tfs[:i2]):
                gene_lists.append([g1, g2, g3])


    aucs = []
    for genes in gene_lists:
        print('Study supervised learning with genes: {:}'.format(', '.join(genes)))

        dsPtf = dsPi.query_features_by_name(genes)
        dstf = dsi.query_features_by_name(genes)

        def rebalance(X, y):
            keep = y.copy()
            ind = np.random.choice((~y).nonzero()[0], size=y.sum(), replace=False)
            keep[ind] = True
            return X[keep], y[keep]

        X_train = dsPtf.counts.values.T
        X_test = dstf.counts.values.T

        X_train_reb, y_train_reb = rebalance(X_train, y_train)
        X_test_reb, y_test_reb = rebalance(X_test, y_test)

        from sklearn.metrics import f1_score, roc_auc_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        #model = LogisticRegression()
        model = RandomForestClassifier()

        tmp = []
        for i in range(10):
            model.fit(X_train, y_train)
            #model.fit(X_test_reb, y_test_reb)
            y_prob = model.predict_proba(X_test_reb)
            auc = roc_auc_score(y_test_reb, y_prob[:, 1])
            tmp.append(auc)
        auc = np.mean(auc)

        aucs.append({'genes': genes, 'auc': auc})

    hm = pd.DataFrame(np.zeros((len(aucs), len(tfs)), bool), columns=tfs)
    for i, auc in enumerate(aucs):
        for gene in auc['genes']:
            hm.iloc[i][gene] = True
    hm['auc'] = [x['auc'] for x in aucs]

    # Sort models
    hm.sort_values('auc', inplace=True)

    # Sort TFs by presence correlating with AUC
    from scipy.stats import spearmanr
    corrs = []
    for tf in tfs:
        corrs.append(spearmanr(hm[tf], hm['auc'])[0])
    ll = np.argsort(corrs)

    print('Plot models sorted')
    fig, axs = plt.subplots(2, 1, figsize=(8, 3.5), sharex=True)
    axs[0].plot(np.arange(len(aucs)), hm['auc'], lw=2, color='grey')
    axs[0].set_ylabel('Area under ROC')
    sns.heatmap(hm[tfs].T.iloc[ll], ax=axs[1], cbar=False, cmap='Reds')
    axs[1].set_xticklabels([])
    axs[1].set_xlabel('Random forest classifiers with 3 TFs')
    fig.tight_layout()
    fig.savefig('../../figures/random_forest_from_Palantir_3genes.png')

    print('Plot with/without GATA2')
    for gene in ['GATA2', 'TAL1']:
        x1 = hm.loc[hm[gene], 'auc'].values
        x2 = hm.loc[~hm[gene], 'auc'].values
        fig, ax = plt.subplots(figsize=(3.2, 3))
        sns.kdeplot(x2, bw=0.03, lw=2, color='grey', label='w/o {:}'.format(gene), legend=False)
        sns.kdeplot(x1, bw=0.03, lw=2, color='tomato', label='w/ {:}'.format(gene), legend=False)
        ax.set_xlabel('Area under ROC')
        ax.set_ylim(top=ax.get_ylim()[1] * 1.2)
        ax.set_ylabel('Density of models')
        ax.legend(loc='upper center', frameon=False, ncol=2, fontsize=9)
        fig.tight_layout()
        fig.savefig('../../figures/random_forest_from_Palantir_3genes_with_without_{:}.png'.format(gene))

    reba = False
    gene_lists_plot = [
            tfs,
            ['LYL1', 'GATA2', 'TAL1'],
            ['GATA2'],
            ['TAL1'],
            ]
    for genes in gene_lists_plot:
        print('Study supervised learning with genes: {:}'.format(', '.join(genes)))

        dsPtf = dsPi.query_features_by_name(genes)
        dstf = dsi.query_features_by_name(genes)

        def rebalance(X, y):
            keep = y.copy()
            ind = np.random.choice((~y).nonzero()[0], size=y.sum(), replace=False)
            keep[ind] = True
            return X[keep], y[keep]

        X_train = dsPtf.counts.values.T
        X_test = dstf.counts.values.T

        X_train_reb, y_train_reb = rebalance(X_train, y_train)
        X_test_reb, y_test_reb = rebalance(X_test, y_test)

        from sklearn.metrics import f1_score, roc_auc_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        #model = LogisticRegression()
        model = RandomForestClassifier()

        model.fit(X_train, y_train)
        #model.fit(X_test_reb, y_test_reb)
        if reba:
            y_prob = model.predict_proba(X_test_reb)
            auc = roc_auc_score(y_test_reb, y_prob[:, 1])
        else:
            y_prob = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_prob[:, 1])

        print('Plot performance on training and test data')
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        ax = axs[0]
        y_prob = model.predict_proba(X_train)
        auc = roc_auc_score(y_train, y_prob[:, 1])
        sns.kdeplot(y_prob[y_train, 1], bw=0.03, ax=ax, color='lawngreen')
        sns.kdeplot(y_prob[~y_train, 1], bw=0.03, ax=ax, color='deeppink')
        ax.plot([0.5] * 2, [0, ax.get_ylim()[1]], lw=2, color='grey', ls='--')
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
        ax.text(0.5, 0.95, 'AUC = {:.2f}'.format(auc), ha='center', va='top',
                transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='grey'))
        ax.set_xlabel('Score of Ery-precursor')
        ax.set_title('Training data')

        ax = axs[1]
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob[:, 1])
        sns.kdeplot(y_prob[y_test, 1], bw=0.03, ax=ax, color='lawngreen')
        sns.kdeplot(y_prob[~y_test, 1], bw=0.03, ax=ax, color='deeppink')
        ax.plot([0.5] * 2, [0, ax.get_ylim()[1]], lw=2, color='grey', ls='--')
        ax.set_xlabel('Score of Ery-precursor')
        ax.set_title('Test data (ME1 cell line)')
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
        ax.text(0.5, 0.95, 'AUC = {:.2f}'.format(auc), ha='center', va='top',
                transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='grey'))

        fig.suptitle('Genes: {:}'.format(', '.join(genes)))
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig('../../figures/random_forest_from_Palantir_{:}.png'.format('-'.join(genes)))

