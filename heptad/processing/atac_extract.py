import os
import sys
import pickle
import numpy as np
import pandas as pd
import pyBigWig


if __name__ == '__main__':

    print('Load sample table')
    fdn_data = '../data/Corces_ATAC/'
    st = pd.read_excel(f'{fdn_data}ATAC_SRR_key_all_samples.xlsx').iloc[:, -2:]
    st.columns = ['SRR', 'CellType']
    st['SRR'] = st['SRR'].astype(str)
    st = st.loc[st['SRR'].str.startswith('SRR')]
    st = st.set_index('SRR', drop=False)
    st['Condition'] = 'healthy'
    st.loc[st['CellType'].str.endswith('_pHSC'), 'Condition'] = 'diseased'
    st.loc[st['CellType'].str.endswith('_LSC'), 'Condition'] = 'diseased'
    st.loc[st['CellType'].str.endswith('_Blast'), 'Condition'] = 'diseased'
    st['Individual'] = 'unknown'
    ind = st['Condition'] == 'diseased'
    st.loc[ind, 'Individual'] = [x.split('_')[0] for x in st.loc[ind]['CellType']]
    st.loc[ind, 'CellType'] = [x.split('_')[1] for x in st.loc[ind]['CellType']]

    print('Load peaks table')
    pt = pd.read_csv(fdn_data+'Heptad_peaks.csv', index_col='element')

    print('Scan BigWig files')
    res = []
    n = len(st)
    for i, (srr, trow) in enumerate(st.iterrows(), 1):
        print(f'{srr} ({i}/{n})')
        fn = f'{fdn_data}BigWig/{srr}.bw'
        cond = trow['Condition']
        ind = trow['Individual']
        ctype = trow['CellType']
        with pyBigWig.open(fn) as bw:
            for elem, row in pt.iterrows():
                chrom = row['chr']
                start = row['start']
                end = row['end']
                arr = bw.values(chrom, start, end)
                res.append({
                    'SRR': srr,
                    'element': elem,
                    'condition': cond,
                    'individual': ind,
                    'celltype': ctype,
                    'array': arr,
                    })

    with open(f'{fdn_data}results_heptad.pkl', 'wb') as f:
        pickle.dump({
            'sampletable': st,
            'peaktable': pt,
            'result': res,
            }, f)
