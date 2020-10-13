# vim: fdm=indent
'''
author:     Fabio Zanini
date:       27/04/20
content:    Evaluate Palantir data given the files on their repo.
'''
import os
import sys
import pickle
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


fig_fdn = '../../figures/paper/fig4/'


# Color palettes for heptad genes
heptad_pal = {
        "LYL1": "#F46D43", "TAL1": "#FF8000", "LMO2": "#fdb863",
        "GATA2": "#4575B4", "FLI1": "#7FBC41", "RUNX1": "#5e3684",
        "ERG": "#276419",
        }
light_pal = {
        "LYL1": "#ff8350", "TAL1": "#ff9a00", "LMO2": "#ffdd77",
        "GATA2": "#538cd8", "FLI1": "#98e24e", "RUNX1": "#71419e",
        "ERG": "#2f781e",
        }
dark_pal = {
        "LYL1": "#c35736", "TAL1": "#cc6600", "LMO2": "#ca934f",
        "GATA2": "#375e90", "FLI1": "#669634", "RUNX1": "#4b2b6a",
        "ERG": "#1f5014",
        }
heptad = list(heptad_pal.keys())


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

    os.makedirs(fig_fdn, exist_ok=True)

    ds = load_our_data()
    ds.counts.normalize('counts_per_ten_thousand', inplace=True)

    if False:
        print('Plot heterogeneity of each heptad gene')
        fig, ax = plt.subplots(figsize=(3.9, 2.45))
        heptad1 = [
                'LYL1',
                'TAL1',
                'LMO2',
                'FLI1',
                'ERG',
                'GATA2',
                'RUNX1',
                ]
        for gene in heptad1:
            x = ds.counts.loc[gene].values.copy()
            x.sort()
            x = np.log10(x + 0.1)
            y = 1.0 - np.linspace(0, 1, len(x))
            ax.plot(x, y, lw=2, color=heptad_pal[gene], alpha=0.9, label=gene)
        ax.grid(True)
        ax.legend(
            title='Gene:',
            loc='upper left',
            bbox_to_anchor=(1, 1.04), bbox_transform=ax.transAxes,
            )
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(['$0$', '$1$', '$10$'])
        ax.set_xlabel('Gene expression [cptt]')
        ax.set_ylabel('Fraction of cells\nexpressing > x')
        ax.set_ylim(top=1.19)
        ax.set_yticks([0, 0.5, 1])
        ax.annotate(
                s='', xy=(1.0, 1.04), xytext=(-0.2, 1.04),
                arrowprops=dict(
                    arrowstyle='<->',
                    color=heptad_pal['LYL1'],
                    linewidth=2,
                    ),
                )
        ax.text(0.4, 1.06, '10X',
                va='bottom', ha='center',
                color=heptad_pal['LYL1'],
                )
        fig.tight_layout()

        if True:
            fxf = 'heterogeneity_heptad'
            for ext in ['png', 'pdf', 'svg']:
                if ext == 'png':
                    fig.savefig(f'{fig_fdn}{fxf}.{ext}', dpi=600)
                else:
                    fig.savefig(f'{fig_fdn}{fxf}.{ext}')

    if False:
        print('Plot correlations heatmap')
        dst = ds.query_features_by_name(heptad)
        corr = dst.correlation.correlate_features_features().fillna(0)
        corrh = corr.copy()
        for i in range(len(corr)):
            corrh.iloc[i, i] = 0

        from scipy.spatial.distance import squareform
        from scipy.cluster.hierarchy import linkage, leaves_list
        cdis = (1.0 - corr).values
        # Numerical error
        cdis[np.arange(len(cdis)), np.arange(len(cdis))] = 0
        pdis = squareform(cdis)
        z = linkage(pdis, 'average', optimal_ordering=True)
        ll = leaves_list(z)
        mat = corr.iloc[ll].T.iloc[ll].T

        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        sns.heatmap(
                corrh.iloc[ll].T.iloc[ll].T,
                ax=ax,
                cmap=sns.diverging_palette(250, 10, as_cmap=True),
                vmin=-0.07, vmax=0.13, center=0)

        highlights = [('GATA2', 'TAL1', '+'), ('GATA2', 'ERG', '-'), ('GATA2', 'LMO2', '-')]
        for (g1, g2, sig) in highlights:
            ic = list(corrh.index[ll]).index(g1)
            ir = list(corrh.index[ll]).index(g2)
            color = 'darkred' if sig == '+' else 'darkblue'
            r = plt.Rectangle(
                    (ic, ir), 1, 1, lw=2, edgecolor=color, facecolor='none',
                    clip_on=False)
            ax.add_artist(r)
            r = plt.Rectangle(
                    (ir, ic), 1, 1, lw=2, edgecolor=color, facecolor='none',
                    clip_on=False)
            ax.add_artist(r)
        ax.set_xlabel('')
        ax.set_ylabel('')
        cax = fig.get_axes()[-1]
        cax.set_ylabel('Spearman correlation')
        fig.tight_layout()

        if True:
            fxf = 'correlation_heatmap_heptad'
            for ext in ['png', 'pdf', 'svg']:
                fig.savefig(f'{fig_fdn}{fxf}.{ext}')

    if False:
        print('Plot conditional distributions')
        dst = ds.query_features_by_name(heptad)

        pairs = [('GATA2', 'TAL1'), ('TAL1', 'GATA2'), ('GATA2', 'ERG'), ('GATA2', 'LMO2')]

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

            x = np.linspace(-1, 1.1, 100)
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
                d2u = np.unique(d2i)
                if len(d2u) == 1:
                    d2u = d2u[0]
                    d2ui = (x < d2u).sum()
                    y2i = np.repeat(0, len(x))
                    y2i[d2ui] = 10
                else:
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

            ncats_plot = len(bins_plot)

            cmap = plt.cm.get_cmap('viridis')
            colors = cmap(np.linspace(0, 1, ncats_plot))

            fig, axs = plt.subplots(
                    1 + ncats_plot, 1,
                    figsize=(2, 1 + 0.5 * ncats_plot),
                    sharex=True,
                    gridspec_kw={'height_ratios': [1.8] + [1] * ncats_plot},
                    )

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
            ax.set_ylabel(reg, fontsize=10, ha='center', va='top', labelpad=15)
            ax.set_yticklabels([])
            ax.text(0.985, 0.89, 'P = {:.1e}'.format(pval),
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, lw=1.5),
                    )
            fig.text(
                    0.1, 0.6,
                    tgt+'   ',
                    fontsize=10, ha='center', va='top', rotation=90)
            for i in range(ncats_plot):
                y2 = y2s[i]
                color = colors[i]
                ax = axs[i+1]
                ax.fill_between(x, y0, y2, color=color, alpha=0.8)
                ax.grid(True)
                ax.set_yticklabels([])

            axs[-1].set_xticks([-1, 0, 1])
            axs[-1].set_xticks([-0.5, 0.5], minor=True)
            axs[-1].set_xticklabels(['$0$', '$1$', '$10$'])
            axs[-1].set_xlabel('Gene expression [cptt]')

            fig.tight_layout(h_pad=0.1)
            return {'fig': fig, 'pval': pval}

        for gene1, gene2 in pairs:
            res = plot_violin(
                    dst.counts.copy(), gene1, gene2,
                    quantiles=[[0.0, 0.05], [0.95, 1.0]],
                    )
            fig = res['fig']
            if True:
                fxf = f'conditional_distributions_{gene1}_{gene2}'
                for ext in ['png', 'pdf', 'svg']:
                    fig.savefig(f'{fig_fdn}{fxf}.{ext}')

    if True:
        print('Load palantir data')
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
        print('Assign subtypes based on Palantir + Northstar')

        fn_cache = '../../data/sequencing/me1/northstar_with_Palantir.pkl'
        if not os.path.isfile(fn_cache):
            print('Subsample palantir data')
            dsPsub = dsP.subsample(40, within_metadata='clusters')

            print('Merge etc based on northstar')
            atlas = dsPsub.to_AnnData()
            atlas.obs['CellType'] = atlas.obs['Cell Subtype']
            ns = northstar.Subsample(
                    atlas=atlas,
                    join='intersection',
                    n_pcs=35,
                    resolution_parameter=0.001,
                    n_features_per_cell_type=80,
                    n_features_overdispersed=0,
                    n_neighbors=20,
                    n_neighbors_external=10,
                    external_neighbors_mutual=False,
                )

            ns.new_data = ds.to_AnnData()
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
            edges = [[e.source, e.target] for e in ns.graph.es]
            cache = {
                'out': out,
                'edges': edges,
                }
            with open(fn_cache, 'wb') as f:
                pickle.dump(cache, f)

        else:
            print('Load northstar results from cache')
            with open(fn_cache, 'rb') as f:
                cache = pickle.load(f)
            out = cache['out']
            edges = cache['edges']
            vs = out.iloc[:, :2]
            northstar_assignment = out['northstar_assignment']

            print('Subsample palantir data like in the cache')
            snames_cache = np.intersect1d(northstar_assignment.index, dsP.samplenames)
            dsPsub = dsP.query_samples_by_name(snames_cache)

        print('Make dataset with merged')
        genes = np.intersect1d(ds.featurenames, dsP.featurenames)
        ds.query_features_by_name(genes, inplace=True)
        ds.samplesheet['Data source'] = 'new_data'
        ds.samplesheet['Cell Subtype'] = 'ME1'
        dsPsub.query_features_by_name(genes, inplace=True)
        dsPsub.samplesheet['Data source'] = 'Palantir'
        dsme = singlet.concatenate([dsPsub, ds])
        dsme.samplesheet['northstar_assignment'] = northstar_assignment
        new_clusters = [x for x in np.unique(northstar_assignment) if x.isdigit()]

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

    if False:
        print('Plot embedding (reduntant given the RNA velocity plot')
        genes = ['Data source', 'Cell Subtype', 'northstar_assignment']

        dsme.samplesheet['embed1'] = vs.iloc[:, 0]
        dsme.samplesheet['embed2'] = vs.iloc[:, 1]

        fig, axs = plt.subplots(1, 3, figsize=(8, 5), sharex=True, sharey=True)
        for i in range(len(axs)):
            gene = genes[i]
            ax = axs[i]
            cmap = cmaps.get(gene, 'viridis')
            dsme.plot.scatter_reduced(
                    ('embed1', 'embed2'),
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

    if False:
        print('Plot embedding of genes')
        genes = ['GATA2', 'TAL1', 'ERG', 'LMO2']
        dse = load_our_data()
        dse.counts.normalize('counts_per_ten_thousand', inplace=True)
        dse.samplesheet['umap1'] = vs.loc[dse.samplenames, 'Dimension 1']
        dse.samplesheet['umap2'] = vs.loc[dse.samplenames, 'Dimension 2']
        dse.query_features_by_name(heptad, inplace=True)
        # Smoothen on graph obtained only via heptad
        edges_new = dse.graph.knn(n_neighbors=10, return_kind='edges')
        dse.counts.smoothen_neighbors(edges_new, inplace=True, n_iterations=10)

        fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.5), sharex=True, sharey=True)
        axs = axs.ravel()
        for i in range(len(axs)):
            gene = genes[i]
            ax = axs[i]
            cmap = cmaps.get(gene, 'viridis')
            dse.plot.scatter_reduced(
                    ('umap1', 'umap2'),
                    color_by=gene,
                    color_log=True,
                    cmap=cmap,
                    ax=ax,
                    alpha=0.6,
                    s=15,
                    )
            ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(gene)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()

        if True:
            fxf = 'genes_embedding'
            for ext in ['png', 'svg']:
                fig.savefig(f'{fig_fdn}{fxf}.{ext}')

    if False:
        print('Compute RNA velocity with scvelo')
        ds = load_our_data()
        ds.counts.normalize('counts_per_ten_thousand', inplace=True)
        ds.samplesheet['umap1'] = vs.loc[ds.samplenames, 'Dimension 1']
        ds.samplesheet['umap2'] = vs.loc[ds.samplenames, 'Dimension 2']
        n1 = dsPsub.n_samples
        n = ds.n_samples
        knn = {'row': [], 'col': [], 'val': []}
        for e in edges:
            v1, v2 = e[0] - n1, e[1] - n1
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
        fig, ax = plt.subplots(figsize=(4, 3.7))
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

        if True:
            fxf = 'velocity_embedding'
            for ext in ['png', 'svg']:
                fig.savefig(f'{fig_fdn}{fxf}.{ext}')

    if False:
        print('Try and classify ME1 cells into the same bins using only heptad')

        print('Load northstar results from cache')
        fn_cache = '../../data/sequencing/me1/northstar_with_Palantir.pkl'
        with open(fn_cache, 'rb') as f:
            cache = pickle.load(f)
        out = cache['out']
        northstar_assignment = out['northstar_assignment']
        ds.samplesheet['cellSubtype'] = northstar_assignment.loc[ds.samplenames]

        print('Reduce to just HSC versus Ery-precursors')
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
        dsPi = dsP.query_samples_by_metadata('cellSubtype in ("HSC", "Ery-precursor")')
        dsi = ds.query_samples_by_metadata('cellSubtype in ("HSC", "Ery-precursor")')
        y_train = (dsPi.samplesheet['cellSubtype'] == 'Ery-precursor').values
        y_test = (dsi.samplesheet['cellSubtype'] == 'Ery-precursor').values

        tfs = ['ERG', 'FLI1', 'LMO2', 'GATA2', 'RUNX1', 'LYL1', 'TAL1']

        print('Study supervised learning')
        from sklearn.metrics import f1_score, roc_auc_score
        from sklearn.ensemble import RandomForestClassifier
        gene_lists = [[tf] for tf in tfs]
        gene_lists += [['GATA2', 'TAL1'], ['GATA2', 'TAL1', 'ERG'], tfs]

        fn_aucs = f'{fig_fdn}prediction_cell_state_AUCs.tsv'
        if not os.path.isfile(fn_aucs):
            nreps = 10
            hms = []
            for nrep in range(nreps):
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

                    #FIXME: something is fishy, it works better with 1 gene than multiple?
                    X_train = dsPtf.counts.values.T
                    X_test = dstf.counts.values.T

                    X_train_reb, y_train_reb = rebalance(X_train, y_train)
                    X_test_reb, y_test_reb = rebalance(X_test, y_test)

                    model = RandomForestClassifier(max_depth=10, min_samples_leaf=4)

                    tmp = []
                    for i in range(3):
                        #model.fit(X_train, y_train)
                        model.fit(X_train_reb, y_train_reb)
                        #model.fit(X_test_reb, y_test_reb)
                        #y_prob = model.predict_proba(X_test)
                        #auc = roc_auc_score(y_test, y_prob[:, 1])
                        y_prob = model.predict_proba(X_test_reb)
                        auc = roc_auc_score(y_test_reb, y_prob[:, 1])
                        tmp.append(auc)
                    auc = np.mean(auc)
                    auc_std = np.std(auc)

                    aucs.append({'gene': ', '.join(genes), 'auc': auc, 'auc_std': auc_std})
                hm = pd.DataFrame(aucs).set_index('gene')
                hms.append(hm['auc'].values)
            hms = pd.DataFrame(hms, columns=[', '.join(x) for x in gene_lists])

            hm = hms.mean(axis=0).to_frame()
            hm.columns = ['auc']
            hm['auc_std'] = hms.std(axis=0)
            hm.sort_values('auc', inplace=True)
            hm.to_csv(fn_aucs, sep='\t', index=True)
        else:
            hm = pd.read_csv(fn_aucs, sep='\t', index_col=0)

        if True:
            fig, ax = plt.subplots(figsize=(3, 3))
            hms = hm.sort_values(by='auc', ascending=True)
            for ig, (gene, row) in enumerate(hms.iterrows()):
                y = row['auc']
                dy = row['auc_std']
                color = heptad_pal.get(gene, 'grey')
                ax.bar([ig], [y], width=0.8, color=color, alpha=0.8, zorder=9)
                ax.errorbar([ig], [y], yerr=[dy], color=color, lw=2)
            ax.set_xticks(np.arange(len(hms)))
            xtl = [x if x != ', '.join(tfs) else 'All 7' for x in hms.index]
            ax.set_xticklabels(xtl, rotation=90)
            for tk in ax.get_xticklabels():
                if ',' in tk.get_text():
                    tk.set_fontsize(9)
            ax.set_ylabel('Area under ROC')
            ax.grid(True)
            ax.set_ylim(bottom=0.3)
            fig.tight_layout()

        if True:
            fxf = 'prediction_bars'
            for ext in ['png', 'svg']:
                fig.savefig(f'{fig_fdn}{fxf}.{ext}')

    if False:
        print('Plot average shifts across cell state boundary')
        ct1, ct2 = ['Ery-precursor', 'HSC']
        tfsp = ['GATA2', 'TAL1', 'ERG', 'FLI1', 'LMO2', 'RUNX1', 'LYL1']

        dsp = dsP.split('Cell Subtype')
        stP1 = dsp[ct1].counts.get_statistics(metrics=['mean', 'std'])
        stP2 = dsp[ct2].counts.get_statistics(metrics=['mean', 'std'])
        stP1['std'] /= np.sqrt(dsp[ct1].n_samples - 1)
        stP2['std'] /= np.sqrt(dsp[ct2].n_samples - 1)

        ds.samplesheet['cellSubtype'] = northstar_assignment.loc[ds.samplenames]
        dsp = ds.split('cellSubtype')
        st1 = dsp[ct1].counts.get_statistics(metrics=['mean', 'std'])
        st2 = dsp[ct2].counts.get_statistics(metrics=['mean', 'std'])
        st1['std'] /= np.sqrt(dsp[ct1].n_samples - 1)
        st2['std'] /= np.sqrt(dsp[ct2].n_samples - 1)

        if False:
            print('Line plot')
            fig, axs = plt.subplots(1, 2, figsize=(5, 2.3), sharex=True, sharey=True)
            ax = axs[0]
            cmap = heptad_pal
            for itf, tf in enumerate(tfsp):
                x = [0, 1]
                y = np.array([st2.loc[tf, 'mean'], st1.loc[tf, 'mean']])
                dy = np.array([st2.loc[tf, 'std'], st1.loc[tf, 'std']])

                yp = np.log10(y)
                dyp = dy / y / np.log(10)
                lw = 2 + (tf in ['ERG', 'GATA2', 'TAL1'])
                alpha = 0.6 + 0.3 * (tf in ['ERG', 'GATA2', 'TAL1'])

                ax.errorbar(x, yp, yerr=dyp, color=cmap[tf], lw=lw, alpha=alpha, zorder=10 - itf)
                ax.scatter(x, yp, color=cmap[tf], lw=2, label=tf, alpha=0.8, zorder=10 - itf)
            ax.grid(False)
            ax.set_xlim(-0.1, 1.1)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([ct2, ct1])
            ax.set_yticks([-1, 0, 1])
            ax.set_yticklabels(['$10^{-1}$', '$1$', '$10$'])
            ax.set_ylabel('Counts per ten\nthousand molecules')
            ax.set_title('ME1 data')

            ax = axs[1]
            for itf, tf in enumerate(tfsp):
                x = [0, 1]
                y = np.array([stP2.loc[tf, 'mean'], stP1.loc[tf, 'mean']])
                dy = np.array([stP2.loc[tf, 'std'], stP1.loc[tf, 'std']])

                yp = np.log10(y)
                dyp = dy / y / np.log(10)
                lw = 2 + (tf in ['ERG', 'GATA2', 'TAL1'])
                alpha = 0.6 + 0.3 * (tf in ['ERG', 'GATA2', 'TAL1'])

                ax.errorbar(x, yp, yerr=dyp, color=cmap[tf], lw=lw, alpha=alpha, zorder=10 - itf)
                ax.scatter(x, yp, color=cmap[tf], lw=2, label=tf, alpha=0.8, zorder=10 - itf)
            ax.grid(False)
            ax.set_ylim(-1, 1.1)
            ax.set_xlim(-0.1, 1.1)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([ct2, ct1])
            ax.legend(
                    title='Gene:',
                    bbox_to_anchor=(1.01, 1.1), bbox_transform=ax.transAxes,
                    fontsize=9,
                    )
            ax.set_title('Palantir data')
            fig.tight_layout()

            if True:
                fxf = 'average_across_boundary'
                for ext in ['png', 'svg']:
                    fig.savefig(f'{fig_fdn}{fxf}.{ext}')

        if False:
            print('Bar plot')
            fig, axs = plt.subplots(1, 2, figsize=(5, 2.3), sharex=True, sharey=True)
            ax = axs[0]
            cmap = heptad_pal
            tfso = list((st1 / st2).loc[tfsp].sort_values('mean').index)
            bar_plot_data = {}
            for itf, tf in enumerate(tfso):
                y = np.array([st2.loc[tf, 'mean'], st1.loc[tf, 'mean']])
                yp = np.log10(y)
                alpha = 0.5 + 0.5 * (tf in ['ERG', 'GATA2', 'TAL1'])
                ax.bar([itf], [yp[1] - yp[0]], color=cmap[tf], alpha=alpha, zorder=10)
                bar_plot_data[('ME1', tf)] = {
                    'y': yp[1] - yp[0],
                    }

            ax.grid(True)
            ax.set_xticks(np.arange(len(tfso)))
            ax.set_xticklabels(tfso, rotation=90)
            ax.set_yticks([-1, 0, 1])
            ax.set_yticklabels(['$10^{-1}$', '$1$', '$10$'])
            ax.set_ylabel('Fold increase\nHSC → Ery-precursor')
            ax.set_title('ME1 data')

            ax = axs[1]
            for itf, tf in enumerate(tfso):
                x = [0, 1]
                y = np.array([stP2.loc[tf, 'mean'], stP1.loc[tf, 'mean']])
                yp = np.log10(y)
                alpha = 0.5 + 0.5 * (tf in ['ERG', 'GATA2', 'TAL1'])
                ax.bar([itf], [yp[1] - yp[0]], color=cmap[tf], alpha=alpha, zorder=10)
                bar_plot_data[('Palantir', tf)] = {
                    'y': yp[1] - yp[0],
                    }


            ax.grid(True)
            ax.set_xticks(np.arange(len(tfso)))
            ax.set_xticklabels(tfso, rotation=90)
            ax.set_ylim(-1, 1.3)
            ax.set_title('Palantir data')
            fig.tight_layout()

            if True:
                fxf = 'average_across_boundary_bars'
                for ext in ['png', 'svg']:
                    fig.savefig(f'{fig_fdn}{fxf}.{ext}')

    if False:
        print('Plot average shifts across cell state boundary, randomized')
        ct1, ct2 = ['Ery-precursor', 'HSC']
        tfsp = ['GATA2', 'TAL1', 'ERG', 'FLI1', 'LMO2', 'RUNX1', 'LYL1']
        ds.samplesheet['cellSubtype'] = northstar_assignment.loc[ds.samplenames]

        dfP = dsP.counts.loc[tfsp].T.copy()
        dfP['cellSubtype'] = dsP.samplesheet['Cell Subtype']

        df = ds.counts.loc[tfsp].T.copy()
        df['cellSubtype'] = ds.samplesheet['cellSubtype']

        def get_ratios(df):
            st = df.groupby('cellSubtype').mean().loc[[ct1, ct2]]
            return st.iloc[0] / st.iloc[1]

        def randomize(df):
            ind = np.arange(len(df))
            np.random.shuffle(ind)
            df = df.copy()
            df['cellSubtype'] = df['cellSubtype'].values[ind]
            return df

        ratios_ME1 = get_ratios(df)
        ratios_rand_ME1 = pd.concat([get_ratios(randomize(df)) for i in range(10000)], axis=1).T

        ratiosP = get_ratios(dfP)
        ratios_randP = pd.concat([get_ratios(randomize(dfP)) for i in range(10000)], axis=1).T

        from scipy.stats import gaussian_kde
        for [ratios, ratios_rand, fxf_suf] in zip([ratios_ME1, ratiosP], [ratios_rand_ME1, ratios_randP], ['ME-1', 'Healthy marrow']):
            fig, axs = plt.subplots(2, 4, figsize=(8, 4), sharey=True)
            axs = axs.ravel()
            txt = {'GATA2': [0.95, 0.95, 'right'], 'TAL1': [0.9, 0.95, 'right'], 'ERG': [0.05, 0.95, 'left']}
            for gene, ax in zip(tfsp, axs):
                x = ratios_rand[gene].values
                xint = np.linspace(-4, 4, 200)
                yint = gaussian_kde(np.log2(x), bw_method=0.4)(xint)
                yint /= yint.max()
                ax.fill_between(
                        xint, 0, yint, color=heptad_pal[gene],
                        )

                r = ratios[gene]
                if r > 1:
                    pval = (x > r).mean()
                else:
                    pval = (x < r).mean()

                if pval < 1e-4:
                    pvalt = '<1e-4'
                else:
                    pvalt = '={:.1e}'.format(pval)

                ax.arrow(
                    np.log2(r), 0.7, 0, -0.6,
                    head_length=0.1,
                    head_width=0.5, length_includes_head=True,
                    overhang=0.2,
                    color='k',
                    )
                tx, ty, ta = txt.get(gene, [0.05 + 0.9 * (r > 1), 0.95, 'left' if r < 1 else 'right'])
                ax.text(tx, ty, f'P{pvalt}'.format(pval), ha=ta, va='top',
                        transform=ax.transAxes, fontsize=10)

                ax.set_title(gene)
                ax.set_xticks([-4, -2, 0, 2, 4])
                ax.set_xticklabels(['1/16', '1/4', '1', '4', '16'])
                ax.set_ylabel('Density [A.U.]')
            axs[-1].set_axis_off()
            fig.suptitle(fxf_suf)
            fig.tight_layout(rect=(0, 0, 1, 0.99))

            if True:
                fxf = f'average_across_boundary_bars_random_{fxf_suf}'
                for ext in ['png', 'svg']:
                    fig.savefig(f'{fig_fdn}{fxf}.{ext}')



    plt.ion()
    plt.show()
