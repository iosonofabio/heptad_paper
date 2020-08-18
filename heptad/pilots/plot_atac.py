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

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    fdn_data = '../../data/Corces_ATAC/'
    fn_res = f'{fdn_data}results_heptad.pkl'
    with open(fn_res, 'rb') as f:
        data = pickle.load(f)
    res = data['result']
    st = data['sampletable']
    pt = data['peaktable']

    # Exclude bulk (marrow and cord blood)
    st = st.loc[~st['CellType'].str.contains('CD34')]
    res = [x for x in res if 'CD34' not in x['celltype']]

    elements = list(np.unique([x['element'] for x in res]))
    if False:
        print('Plot all normals')
        for elem in elements:
            rese = [x for x in res if (x['element'] == elem) and (x['condition'] == 'healthy')]
            ctypes = list(np.unique([x['celltype'] for x in rese]))
            cou_ctypes = Counter([x['celltype'] for x in rese])

            nrows = len(ctypes)
            ncols = max(cou_ctypes.values())
            npl = [0 for x in ctypes]
            colors = sns.color_palette('husl', n_colors=nrows)
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

    print('Transform each sample into a vector')
    samples = list(np.unique([x['SRR'] for x in res]))
    vectors = np.zeros((len(samples), len(elements)))
    ctypes = list(np.unique([x['celltype'] for x in res if x['condition'] == 'healthy']))
    ctypes += ['pHSC', 'LSC', 'Blast']
    ctypes = [
            'Ery', 'Mono',
            #'CD34 Bone Marrow', 'CD34 Cord Blood',
            'HSC', 'MPP', 'CMP', 'MEP', 'GMP',
            'Blast', 'LSC', 'pHSC',
            ]
    labels = -np.ones(len(samples), np.int32)
    for re in res:
        ir = samples.index(re['SRR'])
        if re['celltype'] in ctypes:
            labels[ir] = ctypes.index(re['celltype'])

        ic = elements.index(re['element'])
        vectors[ir, ic] = np.sum(re['array'])

    # TODO: we might prepend this by a filter that takes out peaks smaller than 10
    # Normalize by total counts within all observed elements (for coverage differences)
    vectors = (vectors.T / vectors.sum(axis=1)).T

    idx = np.argsort(labels)[::-1]
    labels_idx = labels[idx]
    mat_plot = pd.DataFrame(vectors, index=samples, columns=elements).iloc[idx]

    fig, axs = plt.subplots(1, 2, figsize=(10, 10), gridspec_kw={'width_ratios': [20, 1]})
    sns.heatmap(mat_plot, ax=axs[0], cmap='plasma', xticklabels=True, yticklabels=True, cbar=False)
    sns.heatmap([[x] for x in labels_idx], cmap=sns.color_palette('husl', len(ctypes) + 1), ax=axs[1], cbar=False)
    axs[1].set_yticklabels([])
    axs[1].set_xticklabels(['Cell Type'], rotation=90)
    for i in range(len(ctypes)):
        ym = (labels_idx == i).nonzero()[0].mean() + 0.5
        xm = 0.5
        axs[1].text(xm, ym, ctypes[i], ha='center', va='center', fontsize=8)
    fig.tight_layout()

    from scipy.spatial.distance import pdist, squareform
    pdis = pdist(vectors)
    mat_dis = squareform(pdis)
    mat_plot = pd.DataFrame(mat_dis, index=samples, columns=samples).iloc[idx].iloc[:, idx]
    fig, axs = plt.subplots(1, 2, figsize=(10, 10), gridspec_kw={'width_ratios': [20, 1]})
    sns.heatmap(mat_plot, ax=axs[0], cmap='plasma', xticklabels=True, yticklabels=True, cbar=False)
    sns.heatmap([[x] for x in labels_idx], cmap=sns.color_palette('husl', len(ctypes) + 1), ax=axs[1], cbar=False)
    axs[1].set_yticklabels([])
    axs[1].set_xticklabels(['Cell Type'], rotation=90)
    for i in range(len(ctypes)):
        ym = (labels_idx == i).nonzero()[0].mean() + 0.5
        xm = 0.5
        axs[1].text(xm, ym, ctypes[i], ha='center', va='center', fontsize=8)
    fig.tight_layout()

    print('Prepare for some machine learning')
    idx_train = [i for i, l in enumerate(labels) if ctypes[l] not in ['Blast', 'LSC', 'pHSC']]
    idx_test = [i for i, l in enumerate(labels) if ctypes[l] in ['Blast', 'LSC', 'pHSC']]
    samples_train = np.array(samples)[idx_train]
    samples_test = np.array(samples)[idx_test]
    vectors_train = vectors[idx_train]
    vectors_test = vectors[idx_test]
    labels_train = labels[idx_train]
    labels_test = labels[idx_test]

    print('Train a logistic regression')
    from sklearn.linear_model import LogisticRegression
    X = vectors_train
    y = labels_train
    model = LogisticRegression(random_state=0).fit(X, y)
    ypred = model.predict(X)

    print('Just calculate the closest average profile')
    ctypes_normal = [x for x in ctypes if x not in ['Blast', 'LSC', 'pHSC']]
    profiles = np.zeros((len(ctypes_normal), len(elements)))
    for i, ct in enumerate(ctypes_normal):
        idxi = np.array(ctypes)[labels] == ct
        tmp = vectors[idxi]
        profiles[i] = tmp.mean(axis=0)

    from scipy.spatial.distance import cdist
    def classify(vectors):
        d = cdist(vectors, profiles, metric='cityblock')
        idmin = np.argmin(d, axis=1)
        return idmin

    ypred = classify(vectors_train)
    ytest = classify(vectors_test)
    types_pred = np.array(ctypes_normal)[ytest]
    types_orig = np.array(ctypes)[labels_test]
    ide = pd.DataFrame(np.vstack([types_orig, types_pred]).T, index=samples_test, columns=['orig', 'pred'])
    ide['Individual'] = st.loc[ide.index, 'Individual']

    print('Plot distances from agerave profiles')
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
            r = plt.Rectangle((ic, ir), 1, 1, lw=2, facecolor='none', edgecolor='k')
            axs[-2].add_artist(r)
        elif st.at[srr, 'CellType'] != row.index[ic]:
            r = plt.Rectangle((ic, ir), 1, 1, lw=4, facecolor='none', edgecolor='red')
            axs[-2].add_artist(r)

    axs[-1].set_ylabel('Cityblock distance on fractional coverage', labelpad=15)

    fig.tight_layout(w_pad=0.3)




    plt.ion()
    plt.show()
