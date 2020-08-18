# vim: fdm=indent
'''
author:     Fabio Zanini
date:       09/06/20
content:    Analyze GATA2 overexpression
'''
import os
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    fdn = '../../data/external/GATA2_overexp/'
    fn_genes = fdn+'counts_gene_names.tsv'

    if False:
        fn = fdn+'counts_raw_ensemblID.tsv'
        df = pd.read_csv(fn, sep='\t', index_col=0)

        conv = pd.read_csv('../../data/gene_ensemblId_name.tsv', sep='\t', index_col=0)

        dfu = df.loc[df.index.str.startswith('ENSG')]

        ids_int = np.intersect1d(dfu.index, conv.index)
        convu = conv.loc[ids_int]
        dfu = dfu.loc[ids_int]
        dfu['GeneName'] = convu.loc[dfu.index]
        dfug = dfu.groupby('GeneName').sum()

        dfug.to_csv(fn_genes, sep='\t', index=True)


    dfug = pd.read_csv(fn_genes, sep='\t', index_col=0)
    dfug = 1e6 * dfug / dfug.sum(axis=0)
    dfug['log2_fc'] = np.log2(dfug['GATA2_OE'] + 0.1) - np.log2(dfug['Control'] + 0.1)
    dfug.to_csv(fdn+'counts_gene_names_normalized_cpm.tsv', sep='\t', index=True)

    heptad = ['GATA2', 'ERG', 'RUNX1', 'TAL1', 'LYL1', 'LMO2', 'FLI1']

    other = ['LMO4', 'GATA1', 'KLF1', 'MYBL2', 'ST18', 'ZEB1']
    dfh = dfug.loc[heptad + other]
