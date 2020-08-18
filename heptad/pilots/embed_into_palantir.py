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

    ds = load_our_data()
    ds.counts.normalize('counts_per_ten_thousand', inplace=True)

    dsP = load_palantir_data(smoothed=False)
    dsP.counts.normalize('counts_per_ten_thousand', inplace=True)
    dsP.samplesheet['Cell Subtype'] = dsP.samplesheet['clusters'].replace({
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

    if True:
        print('Differential expression between the clusters in Palantir data')
        tfs = [
                'CD34', 'GATA1', 'GATA3',
                'ERG', 'FLI1', 'LMO2',
                'GATA2', 'RUNX1', 'LYL1', 'TAL1']
        # TODO
        dsPp = dsP.split('Cell Subtype')

        human_tfs = pd.read_excel('../../data/gene_lists/human_tfs.xlsx', index_col=0).index.str.strip()
        dsP.featuresheet['is_TF'] = dsP.featurenames.isin(human_tfs)
        tfsa = dsP.featurenames[dsP.featuresheet['is_TF']]

        pairs = [['Ery-precursor', 'HSC']]
        for ct1, ct2 in pairs:
            print('Comparing {:} to {:}'.format(ct1, ct2))
            comp = dsPp[ct1].compare(dsPp[ct2])
            comp.rename(
                columns={
                    'avg_self': 'avg_{:}'.format(ct1),
                    'avg_other': 'avg_{:}'.format(ct2),
                    },
                inplace=True,
                )
            comp['is_TF'] = dsP.featuresheet['is_TF']
            comp.sort_values('statistic', ascending=False, inplace=True)
            comp['rank_statistic'] = np.arange(len(comp)) + 1
            comp.loc[comp['rank_statistic'] > len(comp) / 2, 'rank_statistic'] -= len(comp)
            comp.sort_values('log2_fold_change', ascending=False, inplace=True)
            comp['rank_fold_change'] = np.arange(len(comp)) + 1
            comp.loc[comp['rank_fold_change'] > len(comp) / 2, 'rank_fold_change'] -= len(comp)
            comp = comp.loc[dsP.featurenames]

            print('Save to file')
            comp.sort_values('log2_fold_change', ascending=False).to_csv(
                '../../data/gene_lists/DEG_Palantir_allgenes_{:}_{:}.tsv'.format(ct1, ct2),
                sep='\t',
                index=True,
                )

            comp.loc[tfsa].sort_values('log2_fold_change', ascending=False).to_csv(
                '../../data/gene_lists/DEG_Palantir_allTFs_{:}_{:}.tsv'.format(ct1, ct2),
                sep='\t',
                index=True,
                )

            comp.loc[tfs].sort_values('log2_fold_change', ascending=False).to_csv(
                '../../data/gene_lists/DEG_Palantir_heptad_{:}_{:}.tsv'.format(ct1, ct2),
                sep='\t',
                index=True,
                )

    sys.exit()

    print('Subsample their data')
    dsPsub = dsP.subsample(40, within_metadata='clusters')

    if False:
        print('Merge etc based on northstar')
        ns = northstar.Subsample(
                atlas={
                    'cell_types': dsPsub.samplesheet['Cell Subtype'],
                    'counts': dsPsub.counts,
                    },
                join='intersection',
                n_pcs=35,
                resolution_parameter=0.001,
                n_features_per_cell_type=80,
                n_features_overdispersed=0,
                n_neighbors=20,
                n_neighbors_external=10,
                external_neighbors_mutual=False,
            )

        ns.new_data = ds.counts
        ns._check_init_arguments()
        ns.fetch_atlas_if_needed()
        ns.compute_feature_intersection()
        ns._check_feature_intersection()
        ns.prepare_feature_selection()
        ns.select_features()
        ns._check_feature_selection()
        ns.merge_atlas_newdata()

        print('Make PCA and graph')
        ns.compute_pca()
        ns.compute_similarity_graph()

        print('Cluster graph')
        ns.cluster_graph()
        northstar_assignment = np.concatenate(
                [dsPsub.samplesheet['Cell Subtype'].values, ns.membership],
                )

        print('Compute embedding')
        vs = ns.embed('umap')

        print('Save northstar results to file')
        out = vs.copy()
        out['northstar_assignment'] = northstar_assignment
        out.to_csv(
                '../../data/sequencing/me1/northstar_with_Palantir.tsv',
                sep='\t',
                index=True)
    else:
        cache = pd.read_csv(
            '../../data/sequencing/me1/northstar_with_Palantir.tsv',
            sep='\t',
            index_col=0)
        vs = cache.iloc[:, :2]
        northstar_assignment = cache['northstar_assignment']

    print('Make dataset with merged')
    genes = np.intersect1d(ds.featurenames, dsP.featurenames)
    ds.query_features_by_name(genes, inplace=True)
    ds.samplesheet['Data source'] = 'new_data'
    ds.samplesheet['Cell Subtype'] = 'ME1'
    dsPsub.query_features_by_name(genes, inplace=True)
    dsPsub.samplesheet['Data source'] = 'Palantir'
    dsme = singlet.concatenate([dsPsub, ds])
    dsme.samplesheet['northstar_assignment'] = northstar_assignment
    new_clusters = [x for x in np.unique(ns.membership) if x.isdigit()]

    print('Plot embedding')
    genes = ['Data source', 'Cell Subtype', 'northstar_assignment']
    cmaps = {
        #'clusters': an.uns['cluster_colors'],
        'Cell Subtype': {
            'HSC': 'deeppink',
            'Ery-precursor': 'lawngreen',
            'Mono': 'red',
            'Mono-precursor': 'purple',
            'CLP': 'tan',
            'pDC': 'lightseagreen',
            'Ery': 'forestgreen',
            'Mega': 'orange',
            'ME1': 'darkgrey',
            },
        'palantir_pseudotime': 'plasma',
        'palantir_diff_potential': 'plasma',
        }
    cmaps['northstar_assignment'] = dict(cmaps['Cell Subtype'])
    cluster_additional_colors = [
            'darkgrey',
            'darkolivegreen',
            'lime',
            'greenyellow',
            'turquoise',
            'darkviolet',
            'fuchsia',
            'violet',
            ]
    for i, newclu in enumerate(new_clusters):
        cmaps['northstar_assignment'][newclu] = cluster_additional_colors[i]

    # Put remote clusters closer
    vs_adj = vs.copy()
    ind = vs['Dimension 1'] < -5
    vs_adj.loc[ind, 'Dimension 1'] += 13
    vs_adj.loc[ind, 'Dimension 2'] -= 10

    fig, axs = plt.subplots(1, 3, figsize=(8, 7), sharex=True, sharey=True)
    for i in range(len(axs)):
        gene = genes[i]
        ax = axs[i]
        cmap = cmaps.get(gene, 'viridis')
        dsme.plot.scatter_reduced(
                vs_adj,
                color_by=gene,
                color_log=False,
                cmap=cmap,
                ax=ax,
                alpha=0.2 - (0.1 * (gene == 'tissue')),
                s=15,
                )
        #ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(gene)
        handles, labels = [], []
        for key, color in ax._singlet_cmap.items():
            handles.append(ax.scatter([], [], color=color))
            labels.append(key)
        ax.legend(
                handles, labels, loc='upper center',
                fontsize=10, ncol=1,
                bbox_to_anchor=(0.5, -0.07), bbox_transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig('../../figures/ME1_embedded_into_Palantir.png', dpi=600)
    fig.savefig('../../figures/ME1_embedded_into_Palantir.svg')
    fig.savefig('../../figures/ME1_embedded_into_Palantir.pdf')

    if False:
        print('Differential expression between the clusters in ME1')
        tfs = [
                'CD34', 'GATA1', 'GATA3',
                'ERG', 'FLI1', 'LMO2',
                'GATA2', 'RUNX1', 'LYL1', 'TAL1']
        ds.samplesheet['northstar_assignment'] = ns.membership
        dsp = ds.split('northstar_assignment')

        human_tfs = pd.read_excel('../../data/gene_lists/human_tfs.xlsx', index_col=0).index.str.strip()
        ds.featuresheet['is_TF'] = ds.featurenames.isin(human_tfs)
        tfsa = ds.featurenames[ds.featuresheet['is_TF']]

        pairs = [['Ery-precursor', 'HSC'], ['Mono', 'HSC'], ['Mono-precursor', 'HSC']]
        for ct1, ct2 in pairs:
            print('Comparing {:} to {:}'.format(ct1, ct2))
            comp = dsp[ct1].compare(dsp[ct2])
            comp.rename(
                columns={
                    'avg_self': 'avg_{:}'.format(ct1),
                    'avg_other': 'avg_{:}'.format(ct2),
                    },
                inplace=True,
                )
            comp['is_TF'] = ds.featuresheet['is_TF']
            comp.sort_values('statistic', ascending=False, inplace=True)
            comp['rank_statistic'] = np.arange(len(comp)) + 1
            comp.loc[comp['rank_statistic'] > len(comp) / 2, 'rank_statistic'] -= len(comp)
            comp.sort_values('log2_fold_change', ascending=False, inplace=True)
            comp['rank_fold_change'] = np.arange(len(comp)) + 1
            comp.loc[comp['rank_fold_change'] > len(comp) / 2, 'rank_fold_change'] -= len(comp)
            comp = comp.loc[ds.featurenames]

            print('Save to file')
            comp.sort_values('log2_fold_change', ascending=False).to_csv(
                '../../data/gene_lists/DEG_allgenes_{:}_{:}.tsv'.format(ct1, ct2),
                sep='\t',
                index=True,
                )

            comp.loc[tfsa].sort_values('log2_fold_change', ascending=False).to_csv(
                '../../data/gene_lists/DEG_allTFs_{:}_{:}.tsv'.format(ct1, ct2),
                sep='\t',
                index=True,
                )

            comp.loc[tfs].sort_values('log2_fold_change', ascending=False).to_csv(
                '../../data/gene_lists/DEG_heptad_{:}_{:}.tsv'.format(ct1, ct2),
                sep='\t',
                index=True,
                )

    if False:
        print('Compute RNA velocity with scvelo')
        ds = load_our_data()
        ds.counts.normalize('counts_per_ten_thousand', inplace=True)
        ds.samplesheet['umap1'] = vs.loc[ds.samplenames, 'Dimension 1']
        ds.samplesheet['umap2'] = vs.loc[ds.samplenames, 'Dimension 2']
        n1 = ns.n_atlas
        n = ds.n_samples
        knn = {'row': [], 'col': [], 'val': []}
        for e in ns.graph.es:
            v1, v2 = e.source - n1, e.target - n1
            if (v1 > 0) and (v2 > 0):
                knn['row'].append(v1)
                knn['col'].append(v2)
                knn['val'].append(1)
        knn = sp.sparse.coo_matrix(
                (knn['val'], (knn['row'], knn['col'])),
                shape=(n, n),
                dtype=int)

        import scvelo as scv
        fn_velocity = '../../data/sequencing/me1/velocity_me1.loom'
        adata = scv.read(fn_velocity, cache=True)
        adata.obs.index = adata.obs.index.str.slice(4, -1) + '-1'
        adata.var_names_make_unique()

        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(adata, n_pcs=25, n_neighbors=10)

        scv.tl.velocity(adata)
        scv.tl.velocity_graph(adata)

        ds.query_samples_by_name(adata.obs_names, inplace=True)
        ds.samplesheet['northstar_assignment'] = dsme.samplesheet.loc[ds.samplenames, 'northstar_assignment']
        adata.obsm['X_umap'] = ds.samplesheet.loc[adata.obs_names, ['umap1', 'umap2']].values

        n_na = ds.samplesheet['northstar_assignment'].value_counts()
        fig, ax = plt.subplots(figsize=(4.2, 4))
        ds.plot.scatter_reduced(
            ('umap1', 'umap2'),
            color_by='northstar_assignment',
            cmap=cmaps['northstar_assignment'],
            color_log=False,
            ax=ax,
            s=80,
            alpha=0.3,
            )

        scv.pl.velocity_embedding_stream(
            adata,
            basis='umap',
            size=0,
            ax=ax,
            alpha=0.7,
            )
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        handles, labels = [], []
        csts = ['HSC', 'Ery-precursor', 'Mono-precursor', 'Mono', 'Mega']
        for cst in csts:
            handles.append(ax.scatter([], [], color=ax._singlet_cmap[cst]))
            if n_na[cst] > 1:
                suf = 's'
            else:
                suf = ''
            labels.append('{:}: {:} cell{:}'.format(cst, n_na[cst], suf))
        ax.legend(
                handles, labels, loc='upper center',
                fontsize=10, ncol=2,
                bbox_to_anchor=(0.5, -0.07), bbox_transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig('../../figures/ME1_velocity.png', dpi=600)
        fig.savefig('../../figures/ME1_velocity.svg')
        fig.savefig('../../figures/ME1_velocity.pdf')

        #scv.pl.scatter(adata, basis=top_genes[:10], legend_loc='none',
        #           size=80, frameon=False, ncols=5, fontsize=20)

    print('Try and classify ME1 cells into the same bins using only heptad')
    tfs = ['ERG', 'FLI1', 'LMO2',
           'GATA2', 'RUNX1', 'LYL1', 'TAL1']
    dstf = ds.query_features_by_name(tfs)
    dstf.samplesheet['Cell Subtype'] = dstf.samplesheet['northstar_assignment']
    dsPtf = dsP.query_features_by_name(tfs)
    csts = ['HSC', 'Ery-precursor', 'Mono-precursor']
    fig, axs = plt.subplots(2, 7, figsize=(11, 4), sharex=True, sharey=True)
    for k, dsi in enumerate([dsPtf, dstf]):
        for i, tf in enumerate(tfs):
            ax = axs[k, i]
            for j, cst in enumerate(csts):
                ind = dsi.samplesheet['Cell Subtype'] == cst
                datum = np.log10(dsi.counts.loc[tf, ind].values + 0.1)
                sns.kdeplot(
                    datum,
                    bw=0.2,
                    ax=ax,
                    color=cmaps['northstar_assignment'][cst],
                    alpha=0.6 + 0.3 * (cst == 'Ery-precursor'),
                    lw=3,
                    )
            if k == 0:
                ax.set_title(tf)
            else:
                ax.set_xticks([-1, 0, 1, 2])
                ax.set_xticklabels(['$0$', '$1$', '$10$', '$10^2$'])
        axs[k, 0].set_yticks([])
    axs[0, 0].set_ylabel('Palantir data', rotation=0, ha='right')
    axs[1, 0].set_ylabel('ME1', rotation=0, ha='right')
    fig.text(0.54, 0.02, 'Gene expression [counts per ten thousand]', ha='center')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig('../../figures/heptad_distributions_compared_with_Palantir.png', dpi=600)
    fig.savefig('../../figures/heptad_distributions_compared_with_Palantir.svg')
    fig.savefig('../../figures/heptad_distributions_compared_with_Palantir.pdf')


    plt.ion()
    plt.show()
