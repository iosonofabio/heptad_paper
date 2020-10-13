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


fig_fdn = '../../figures/paper/supplementary/'


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

        print('Predict cell types')
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

        if True:
            print('Plot all AML patients')

            st = st.loc[st['Condition'] == 'diseased']
            st.sort_values(by=['Individual', 'CellType'], ascending=False, inplace=True)
            srr_plot = st.index
            #cmap = {'pHSC': 'tomato', 'LSC': 'steelblue', 'Blast': 'darkred'}
            #colors = [cmap[st.at[x, 'CellType']] for x in srr_plot]
            colors = [cmap_normal[pred.at[x, 'Prediction']] for x in srr_plot]
            fig, axs = plt.subplots(len(srr_plot), len(elements), figsize=(6, 10))
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
                        ax.set_ylabel(ctype, va='center', ha='right', rotation=0, labelpad=15)
                    elif (j == len(elements) - 1) and (i % 3 == 1):
                        ax2 = ax.twinx()
                        ax2.set_yticks([])
                        ax2.set_ylabel(pname, rotation=0, va='center', ha='left', labelpad=15)
                        ax2.plot([1.2] * 2, [-0.9, 1.9], lw=2, color='k',
                                 transform=ax2.transAxes, clip_on=False)
                    ax.set_yticks([])
                    ax.set_xticks([])
                    ymax = max(ymax, y.max())
                for j in range(len(elements)):
                    ax = axs[i, j]
                    ax.set_ylim(top=ymax*1.1)
            fig.tight_layout(w_pad=0, h_pad=0)
            if True:
                fxf = 'distributions_AML_all_prediction_colors'
                for ext in ['png', 'svg']:
                    if ext == 'png':
                        fig.savefig(f'{fig_fdn}{fxf}.{ext}', dpi=600)
                    else:
                        fig.savefig(f'{fig_fdn}{fxf}.{ext}')

    plt.ion()
    plt.show()
