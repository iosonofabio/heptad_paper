# vim: fdm=indent
'''
author:     Fabio Zanini
date:       30/06/20
content:    Plot ATAC Seq peaks
'''
import os
import sys
import pickle
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist, squareform

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/home/fabio/university/postdoc/singlet')
os.environ['SINGLET_CONFIG_FILENAME'] = 'singlet.yml'
import singlet
import anndata


fig_fdn = '../../figures/paper/fig1/'


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

    ds.samplesheet['Cell Subtype'] = ds.samplesheet['clusters'].replace({
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




def classify(vectors, profiles):
    d = cdist(vectors, profiles, metric='cityblock')
    idmin = np.argmin(d, axis=1)
    return idmin


if __name__ == '__main__':

    os.makedirs(fig_fdn, exist_ok=True)

    if False:
        print('scRNA-Seq analysis on Palantir data')
        dsi = load_palantir_data(smoothed=True)

        cmaps = {
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
        fig, ax = plt.subplots(figsize=(7, 3.5))
        dsi.plot.scatter_reduced(
                ('tsne_1', 'tsne_2'),
                color_by='clusters',
                # It's already logged
                color_log=False,
                cmap=cmaps.get('clusters', 'viridis'),
                ax=ax,
                alpha=0.2,
                s=15,
                )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()

        genes = [
                'CD34',
                'LYL1',
                'TAL1',
                'LMO2',
                'FLI1',
                'ERG',
                'GATA2',
                'RUNX1',
                ]

        fig = plt.figure(figsize=(7, 3.5))
        gs = fig.add_gridspec(2, 5, width_ratios=[10] * 4 + [1])
        axs = []
        for i in range(2):
            for j in range(4):
                axs.append(fig.add_subplot(gs[i, j]))
        axs.append(fig.add_subplot(gs[:, -1]))
        for i, (ax, gene) in enumerate(zip(axs, genes)):
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
            ax.set_axis_off()
            ax.set_title(gene)
        # Colorbar
        import matplotlib as mpl
        ax = axs[-1]
        norm = mpl.colors.Normalize(vmin=0., vmax=1.)
        cb1 = mpl.colorbar.ColorbarBase(
                ax, cmap=mpl.cm.get_cmap('viridis'),
                norm=norm,
                orientation='vertical')
        cb1.set_ticks([0, 0.33, 0.67, 1])
        cb1.set_ticklabels(['None', 'Low', 'Mid', 'High'])
        fig.tight_layout()
        if True:
            fxf = 'Palantir_embedding_genes'
            for ext in ['png', 'svg']:
                if ext == 'png':
                    fig.savefig(f'{fig_fdn}{fxf}.{ext}', dpi=600)
                else:
                    fig.savefig(f'{fig_fdn}{fxf}.{ext}')

    if True:
        print('ATAC-Seq analysis')

        print('Load ATAC-Seq data from pickle of BigWig files')
        fdn_data = '../../data/Corces_ATAC/'
        fn_res = f'{fdn_data}results_heptad.pkl'
        with open(fn_res, 'rb') as f:
            data = pickle.load(f)
        res = data['result']
        st = data['sampletable']
        pt = data['peaktable']

        # Exclude duplicate LYL peak
        pt = pt.loc[pt.index != 'LYL_P']
        pt = pt.rename(index={'LYL_P_alt': 'LYL1_P'})
        for re in res:
            if re['element'] == 'LYL_P_alt':
                re['element'] = 'LYL1_P'

        elements = 'LYL1_P, TAL1_p40, LMO2_m25, FLI1_m15, ERG_p85, GATA2_m117, GATA2_p4, RUNX1_p23, RUNX1_p141'.split(', ')

        # Exclude bulk (marrow and cord blood)
        st = st.loc[~st['CellType'].str.contains('CD34')]
        res = [x for x in res if 'CD34' not in x['celltype']]

        n_hea = (st['Condition'] == 'healthy').sum()
        n_dis = (st['Condition'] != 'healthy').sum()

        samples = list(st.index)
        ctypes = [
                #'CD34 Bone Marrow', 'CD34 Cord Blood',
                'HSC', 'MPP', 'CMP', 'MEP', 'GMP', 'LMPP',
                'Mono', 'Ery',
                'Blast', 'LSC', 'pHSC',
                ]
        ctypes_normal = [x for x in ctypes if x not in ['Blast', 'LSC', 'pHSC']]
        colors = sns.color_palette('husl', n_colors=len(ctypes_normal))
        cmap_normal = dict(zip(ctypes_normal, colors))

        if False:
            print('Plot all normals')
            for elem in elements:
                rese = [x for x in res if (x['element'] == elem) and (x['condition'] == 'healthy')]
                ctypes = list(np.unique([x['celltype'] for x in rese]))
                cou_ctypes = Counter([x['celltype'] for x in rese])

                nrows = len(ctypes)
                ncols = max(cou_ctypes.values())
                npl = [0 for x in ctypes]
                fig, axs = plt.subplots(nrows, ncols, figsize=(15, 10))
                for re in rese:
                    ir = ctypes.index(re['celltype'])
                    ic = npl[ir]
                    ax = axs[ir, ic]
                    if ic == 0:
                        ax.set_ylabel(ctypes[ir], ha='right', rotation=0, labelpad=20)
                    y = re['array']
                    x = np.arange(len(y))
                    ax.fill_between(x, 0, y, alpha=0.7, color=colors[ir])
                    npl[ir] += 1
                fig.suptitle(elem)
                fig.tight_layout(rect=(0, 0, 1, 0.95))

        if False:
            print('Plot average normal profiles')
            colors = sns.color_palette('husl', len(ctypes_normal))
            fig, axs = plt.subplots(len(ctypes_normal), len(elements), figsize=(5.5, 4.2))
            for i in range(len(ctypes_normal)):
                ctype = ctypes_normal[i]
                ymax = 0
                for j in range(len(elements)):
                    elem = elements[j]
                    ax = axs[i, j]
                    arrs = []
                    for re in res:
                        if re['celltype'] != ctype:
                            continue
                        if re['element'] != elem:
                            continue
                        arrs.append(re['array'])
                    y = np.array(arrs).sum(axis=0)
                    x = np.arange(len(y)) + pt.at[elem, 'start']
                    ax.fill_between(x, 0, y, alpha=0.7, color=colors[i])
                    if i == 0:
                        ax.set_title(elem.replace('_', '\n', 1), fontsize=9)
                    if j == 0:
                        ax.set_ylabel(ctype, va='center', ha='right', rotation=0, labelpad=15)
                    ax.set_yticks([])
                    ax.set_xticks([])
                    ymax = max(ymax, y.max())
                for j in range(len(elements)):
                    ax = axs[i, j]
                    ax.set_ylim(top=ymax*1.1)
            fig.tight_layout(w_pad=0, h_pad=0)
            if False:
                fxf = 'distributions_profiles'
                for ext in ['png', 'svg']:
                    if ext == 'png':
                        fig.savefig(f'{fig_fdn}{fxf}.{ext}', dpi=600)
                    else:
                        fig.savefig(f'{fig_fdn}{fxf}.{ext}')

        print('Transform each sample into a vector')
        vectors = np.zeros((len(samples), len(elements)))
        labels = -np.ones(len(samples), np.int32)
        for re in res:
            ir = samples.index(re['SRR'])
            if re['celltype'] in ctypes:
                labels[ir] = ctypes.index(re['celltype'])

            if re['element'] not in elements:
                continue
            ic = elements.index(re['element'])
            vectors[ir, ic] = np.sum(re['array'])

        # Normalize by total counts within heptad (for coverage differences)
        vectors = (vectors.T / vectors.sum(axis=1)).T

        print('Calculate average profiles')
        profiles = np.zeros((len(ctypes_normal), len(elements)))
        for i, ct in enumerate(ctypes_normal):
            idxi = np.array(ctypes)[labels] == ct
            tmp = vectors[idxi]
            profiles[i] = tmp.mean(axis=0)

        print('Sort samples according to cell type')
        idx = np.argsort(labels)[::-1]
        labels_idx = labels[idx]

        if False:
            print('Print block matrix of cityblock distance')
            pdis = pdist(vectors, metric='cityblock')
            mat_dis = squareform(pdis)
            mat_plot = pd.DataFrame(
                    mat_dis,
                    index=samples, columns=samples).iloc[idx].iloc[:, idx]
            lab_plot = pd.DataFrame(
                    [[x] for x in labels_idx],
                    index=mat_plot.index, columns=['Cell Type'])
            ass_plot = pd.DataFrame(
                    [[int(x != 'healthy')] for x in st['Condition'].values],
                    index=mat_plot.index, columns=['Condition']).iloc[idx]

            fig, axs = plt.subplots(1, 4, figsize=(10, 10), gridspec_kw={'width_ratios': [1, 1, 20, 0.5]})
            sns.heatmap(lab_plot, cmap=sns.color_palette('husl', len(ctypes) + 1), ax=axs[0], cbar=False, yticklabels=False)
            sns.heatmap(ass_plot, cmap=['steelblue', 'tomato'], ax=axs[1], cbar=False, yticklabels=False)
            sns.heatmap(mat_plot, ax=axs[-2], cmap='plasma', xticklabels=False, yticklabels=False, cbar=True, cbar_ax=axs[-1], cbar_kws={'label': 'Cityblock distance on fractional coverage'})
            axs[0].set_xticklabels(['CellType'], rotation=90)
            axs[1].set_xticklabels(['Condition'], rotation=90)
            axs[0].set_yticklabels([])
            axs[1].set_yticklabels([])
            axs[-2].set_yticklabels([])
            for i in range(len(ctypes)):
                ym = (labels_idx == i).nonzero()[0].mean() + 0.5
                xm = 0.5
                axs[0].text(xm, ym, ctypes[i], ha='center', va='center', fontsize=8)
            axs[1].text(xm, ass_plot['Condition'].sum() / 2, 'Neoplastic', ha='center', va='center', fontsize=8, rotation=90)
            axs[1].text(xm, n_dis + 0.5 * n_hea, 'Healthy', ha='center', va='center', fontsize=8, rotation=90)
            axs[-1].set_ylabel('Cityblock distance on fractional coverage', labelpad=15)
            fig.tight_layout(w_pad=0.3)
            if False:
                fxf = 'heatmap_distance_blocks'
                for ext in ['png', 'svg']:
                    if ext == 'png':
                        fig.savefig(f'{fig_fdn}{fxf}.{ext}', dpi=600)
                    else:
                        fig.savefig(f'{fig_fdn}{fxf}.{ext}')

        print('Prepare for some machine learning')
        idx_train = [i for i, l in enumerate(labels) if ctypes[l] not in ['Blast', 'LSC', 'pHSC']]
        idx_test = [i for i, l in enumerate(labels) if ctypes[l] in ['Blast', 'LSC', 'pHSC']]
        samples_train = np.array(samples)[idx_train]
        samples_test = np.array(samples)[idx_test]
        vectors_train = vectors[idx_train]
        vectors_test = vectors[idx_test]
        labels_train = labels[idx_train]
        labels_test = labels[idx_test]

        print('Compute the closest average profile')
        ypred = classify(vectors_train, profiles)
        ytest = classify(vectors_test, profiles)
        types_pred = np.array(ctypes_normal)[ytest]
        types_orig = np.array(ctypes)[labels_test]
        ide = pd.DataFrame(np.vstack([types_orig, types_pred]).T, index=samples_test, columns=['orig', 'pred'])
        ide['Individual'] = st.loc[ide.index, 'Individual']

        if False:
            print('Plot distances from average profiles, only healthy')
            mat_plot = pd.DataFrame(
                    cdist(vectors, profiles, metric='cityblock'),
                    index=samples,
                    columns=ctypes_normal,
                    ).iloc[idx]
            lab_plot = pd.DataFrame([[x] for x in labels_idx], index=mat_plot.index, columns=['Cell Type'])

            mat_plot = mat_plot.iloc[21:].iloc[::-1]
            lab_plot = lab_plot.iloc[21:].iloc[::-1]

            fig, axs = plt.subplots(1, 3, figsize=(4.2, 4.5), gridspec_kw={'width_ratios': [1.7, 12, 1]})
            sns.heatmap(lab_plot, cmap=[cmap_normal[x] for x in ctypes_normal], ax=axs[0], cbar=False, yticklabels=False)
            sns.heatmap(mat_plot, ax=axs[-2], cmap='plasma', xticklabels=True, yticklabels=False, cbar=True, cbar_ax=axs[-1], cbar_kws={'label': 'Cityblock distance on fractional coverage'})
            axs[0].set_xticks([])
            axs[0].set_yticklabels([])
            axs[-2].set_yticklabels([])
            for i in range(len(ctypes_normal)):
                ym = len(labels_idx) - ((labels_idx == i).nonzero()[0].mean() + 0.5)
                xm = 0.5
                axs[0].text(xm, ym, ctypes_normal[i], ha='center', va='center', fontsize=8)

            for ir, (srr, row) in enumerate(mat_plot.iterrows()):
                ic = np.argmin(row.values)
                if st.at[srr, 'CellType'] != row.index[ic]:
                    r = plt.Rectangle((ic, ir), 1, 1, lw=2.5, facecolor='none', edgecolor='red')
                    axs[-2].add_artist(r)

            axs[-1].set_ylabel('Cityblock distance on fractional coverage', labelpad=15)
            fig.tight_layout(w_pad=0.7)
            if False:
                fxf = 'heatmap_distance_profiles_healthy'
                for ext in ['png', 'svg']:
                    if ext == 'png':
                        fig.savefig(f'{fig_fdn}{fxf}.{ext}', dpi=600)
                    else:
                        fig.savefig(f'{fig_fdn}{fxf}.{ext}')

        if False:
            print('Plot line plots of distances from average profiles, AML')
            mat_plot = pd.DataFrame(
                    cdist(vectors, profiles, metric='cityblock'),
                    index=samples,
                    columns=ctypes_normal,
                    ).iloc[idx]
            lab_plot = pd.DataFrame([[x] for x in labels_idx], index=mat_plot.index, columns=['Cell Type'])

            mat_plot = mat_plot.iloc[:21]
            lab_plot = lab_plot.iloc[:21]

            fig, axs = plt.subplots(3, 1, figsize=(4.2, 2.2), sharex=True, sharey=True)
            ctypes_plot = ['pHSC', 'LSC', 'Blast']
            x = np.arange(len(ctypes_normal))
            for i in range(len(ctypes_plot)):
                ax = axs[i]
                ctype = ctypes_plot[i]
                mati = mat_plot.loc[st.loc[mat_plot.index, 'CellType'] == ctype]
                for srr, row in mati.iterrows():
                    color = cmap_normal[row.idxmin()]
                    ax.plot(x, row, lw=1, color=color, alpha=0.5)
                    xmin = ctypes_normal.index(row.idxmin())
                    ax.scatter([xmin], row.min(), s=50, marker='*', color=color)
                ax.set_xticks(x)
                ax.set_xticklabels(ctypes_normal)
                ax.set_ylabel(ctype, rotation=0, ha='right', va='center')
                ax.set_ylim(0, 0.7)
            fig.text(
                    0.037, 0.54,
                    'Cityblock distance',
                    rotation=90, ha='center', va='center',
                    )
            fig.tight_layout(rect=(0.033, 0, 1, 1))
            if False:
                fxf = 'lineplots_distance_profiles_AML'
                for ext in ['png', 'svg']:
                    if ext == 'png':
                        fig.savefig(f'{fig_fdn}{fxf}.{ext}', dpi=600)
                    else:
                        fig.savefig(f'{fig_fdn}{fxf}.{ext}')

        if False:
            print('Plot distances from average profiles')
            mat_plot = pd.DataFrame(
                    cdist(vectors, profiles, metric='cityblock'),
                    index=samples,
                    columns=ctypes_normal,
                    ).iloc[idx]
            lab_plot = pd.DataFrame([[x] for x in labels_idx], index=mat_plot.index, columns=['Cell Type'])
            ass_plot = pd.DataFrame([[int(x != 'healthy')] for x in st['Condition'].values], index=mat_plot.index, columns=['Condition']).iloc[idx]

            fig, axs = plt.subplots(1, 4, figsize=(6, 10), gridspec_kw={'width_ratios': [1, 1, 15, 0.7]})
            sns.heatmap(lab_plot, cmap=sns.color_palette('husl', len(ctypes) + 1), ax=axs[0], cbar=False, yticklabels=False)
            sns.heatmap(ass_plot, cmap=['steelblue', 'tomato'], ax=axs[1], cbar=False, yticklabels=False)
            sns.heatmap(mat_plot, ax=axs[-2], cmap='plasma', xticklabels=True, yticklabels=False, cbar=True, cbar_ax=axs[-1], cbar_kws={'label': 'Cityblock distance on fractional coverage'})
            axs[0].set_xticklabels(['CellType'], rotation=90)
            axs[1].set_xticklabels(['Condition'], rotation=90)
            axs[0].set_yticklabels([])
            axs[1].set_yticklabels([])
            axs[-2].set_yticklabels([])
            for i in range(len(ctypes)):
                ym = (labels_idx == i).nonzero()[0].mean() + 0.5
                xm = 0.5
                axs[0].text(xm, ym, ctypes[i], ha='center', va='center', fontsize=8)
            axs[1].text(xm, ass_plot['Condition'].sum() / 2, 'Neoplastic', ha='center', va='center', fontsize=8, rotation=90)
            axs[1].text(xm, len(samples_test) + 0.5 * len(samples_train), 'Healthy', ha='center', va='center', fontsize=8, rotation=90)

            axs[-2].plot(list(axs[-2].get_xlim()), [ass_plot['Condition'].sum()] * 2, lw=2, color='w', alpha=0.7)
            for ir, (srr, row) in enumerate(mat_plot.iterrows()):
                ic = np.argmin(row.values)
                if st.at[srr, 'Condition'] != 'healthy':
                    r = plt.Rectangle((ic, ir), 1, 1, lw=2, facecolor='none', edgecolor='k', zorder=10)
                    axs[-2].add_artist(r)
                elif st.at[srr, 'CellType'] != row.index[ic]:
                    r = plt.Rectangle((ic, ir), 1, 1, lw=4, facecolor='none', edgecolor='red')
                    axs[-2].add_artist(r)

            axs[-1].set_ylabel('Cityblock distance on fractional coverage', labelpad=15)
            fig.tight_layout(w_pad=0.3)
            if False:
                fxf = 'heatmap_distance_profiles'
                for ext in ['png', 'svg']:
                    if ext == 'png':
                        fig.savefig(f'{fig_fdn}{fxf}.{ext}', dpi=600)
                    else:
                        fig.savefig(f'{fig_fdn}{fxf}.{ext}')

        if True:
            print('Output predictions')
            mat_plot = pd.DataFrame(
                    cdist(vectors, profiles, metric='cityblock'),
                    index=samples,
                    columns=ctypes_normal,
                    ).iloc[idx]
            std = st.loc[st['Condition'] == 'diseased']
            pred = pd.DataFrame([], index=std.index)
            pred['Prediction'] = ''
            for srr in pred.index:
                pred.at[srr, 'Prediction'] = mat_plot.loc[srr].idxmin()
            pred['Individual'] = std['Individual']
            pred['CellType'] = std['CellType']
            pred.sort_values(by=['CellType', 'Individual'], inplace=True)

        if False:
            print('Plot typical AML patients')
            # SU444 is a good case
            srr_plot = ['SRR2920574', 'SRR2920573', 'SRR2920572']
            colors = [cmap_normal[pred.at[x, 'Prediction']] for x in srr_plot]
            fig, axs = plt.subplots(3, len(elements), figsize=(5.5, 2.2))
            for i in range(len(srr_plot)):
                srr = srr_plot[i]
                ctype = st.at[srr, 'CellType']
                pname = st.at[srr, 'Individual']
                ymax = 0
                for j in range(len(elements)):
                    elem = elements[j]
                    ax = axs[i, j]
                    arrs = []
                    for re in res:
                        if re['celltype'] != ctype:
                            continue
                        if re['individual'] != pname:
                            continue
                        if re['element'] != elem:
                            continue
                        arrs.append(re['array'])
                    y = np.array(arrs).sum(axis=0)
                    x = np.arange(len(y)) + pt.at[elem, 'start']
                    ax.fill_between(x, 0, y, alpha=0.7, color=colors[i])
                    if i == 0:
                        ax.set_title(elem.replace('_', '\n', 1), fontsize=9)
                    if j == 0:
                        ax.set_ylabel(ctype+'\n'+pname, va='center', ha='right', rotation=0, labelpad=15)
                    ax.set_yticks([])
                    ax.set_xticks([])
                    ymax = max(ymax, y.max())
                for j in range(len(elements)):
                    ax = axs[i, j]
                    ax.set_ylim(top=ymax*1.1)
            fig.tight_layout(w_pad=0, h_pad=0)
            if True:
                fxf = 'distributions_AML_examples'
                for ext in ['png', 'svg']:
                    if ext == 'png':
                        fig.savefig(f'{fig_fdn}{fxf}.{ext}', dpi=600)
                    else:
                        fig.savefig(f'{fig_fdn}{fxf}.{ext}')

        if True:
            print('Plot confusion matrix between heptad and full ATAC')
            fn_full = '../../data/Corces_ATAC/Heptad and Enh_cyto predictions.xlsx'
            dff = pd.read_excel(fn_full)
            # Remove old predictions from us
            dff = dff.loc[dff['Classifier'] == 'Enh_cyto']
            dff.set_index('SRR', inplace=True)
            del dff['Classifier']
            dff.rename(columns={'Prediction': 'Prediction_Enh_cyto'}, inplace=True)
            dff['Prediction_heptad'] = pred.loc[dff.index, 'Prediction']

            def plot_confusion(dff, title=None, ax=None, use_legend=True):
                nct = len(ctypes_normal)
                conf = pd.DataFrame(
                        np.zeros((nct, nct)),
                        index=pd.Index(ctypes_normal, name='Enh_cyto'),
                        columns=pd.Index(ctypes_normal, name='heptad'),
                        )
                for _, row in dff.iterrows():
                    conf.loc[row['Prediction_Enh_cyto'], row['Prediction_heptad']] += 1

                idx_conf = [x for x in ctypes_normal if conf[x].sum() + conf.loc[x].sum()]
                confi = conf.loc[idx_conf].T.loc[idx_conf].T
                ncti = len(confi)
                similar = [frozenset(['GMP', 'LMPP']), frozenset(['HSC', 'MPP'])]
                def sfun(x):
                    return 5 + 200 * x / confi.values.max()

                if ax is None:
                    new_fig = True
                    fig, ax = plt.subplots(figsize=(3.5, 2.5))
                else:
                    new_fig = False

                for i in range(ncti):
                    for j in range(ncti):
                        c = confi.iloc[i, j]
                        if c == 0:
                            continue
                        color = 'mediumseagreen' if i == j else 'tomato'
                        if frozenset([confi.index[i], confi.index[j]]) in similar:
                            color = 'orange'
                        s = sfun(c)
                        ax.scatter([j], [i], s=s, color=color, zorder=9)
                if use_legend:
                    lvals = [1, 2, 4]
                    llabs = [str(x) for x in lvals]
                    lhans = [ax.scatter([], [], s=sfun(x), c='grey') for x in lvals]
                    ax.legend(
                            lhans, llabs,
                            title='N. samples:', loc='upper left',
                            bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
                ax.set_xticks(np.arange(ncti))
                ax.set_yticks(np.arange(ncti))
                ax.set_xticklabels(confi.index, rotation=90)
                ax.set_yticklabels(confi.index)
                ax.set_xlabel('heptad')
                #ax.set_ylabel('Enh_cyto')
                ax.set_xlim(-0.5, ncti - 0.5)
                ax.set_ylim(ncti - 0.5, -0.5)
                ax.grid(True)
                if title is not None:
                    ax.set_title(title)

                if new_fig:
                    fig.tight_layout()
                    return fig

            fig = plot_confusion(dff)
            if False:
                fxf = 'confusion_matrix_all'
                for ext in ['png', 'svg']:
                    if ext == 'png':
                        fig.savefig(f'{fig_fdn}{fxf}.{ext}', dpi=600)
                    else:
                        fig.savefig(f'{fig_fdn}{fxf}.{ext}')

            fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.25))
            for ax, ct in zip(axs, ['pHSC', 'LSC', 'Blast']):
                plot_confusion(
                    dff.loc[dff['CellType'] == ct], title=ct, ax=ax,
                    use_legend=ax==axs[-1],
                    )
            fig.tight_layout()
            if False:
                fxf = f'confusion_matrix_subplots'
                for ext in ['png', 'svg']:
                    if ext == 'png':
                        fig.savefig(f'{fig_fdn}{fxf}.{ext}', dpi=600)
                    else:
                        fig.savefig(f'{fig_fdn}{fxf}.{ext}')

        #TODO: add bootstrapping on patients on "equal or close"

    plt.ion()
    plt.show()
