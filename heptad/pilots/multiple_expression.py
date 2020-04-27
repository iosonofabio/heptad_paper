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


if __name__ == '__main__':

    fn_ds = '../../data/sequencing/me1/normalized_7tfs.loom'
    dst = singlet.Dataset(dataset={
        'path': fn_ds,
        'index_samples': 'CellID',
        'index_features': 'GeneName',
        })

    medians = dst.counts.median(axis=1)
    counts_binary = (dst.counts.T > medians)
    p1 = counts_binary.mean(axis=0)

    from itertools import combinations
    data = []
    for i in range(len(tfs)):
        for tfis in combinations(tfs, i+1):
            p_act = counts_binary[list(tfis)].all(axis=1).mean(axis=0)
            p_exp = p1[list(tfis)].prod()

            datum = {
                'Pact': p_act,
                'Pexp': p_exp,
                'l': i+1,
                'tfs': tfis,
                }
            data.append(datum)
    data = pd.DataFrame(data).set_index('tfs')
    data['log2_fc'] = np.log2(data['Pact']) - np.log2(data['Pexp'])
