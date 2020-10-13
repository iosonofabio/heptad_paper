import os
import sys
import pickle
import numpy as np
import pandas as pd
import pyBigWig


if __name__ == '__main__':

    print('Load sample table')
    fdn_data = '/home/julie/DataSets/BWgraph/'
    st = pd.read_csv(f'{fdn_data}chipseq_bw_graph_key.csv')
    st['filename'] = st['filename'].astype(str)
    st = st.set_index('filename', drop=False)
    

    print('Load peaks table')
    pt = pd.read_csv(fdn_data+'Heptad_peaks.csv', index_col='element')

    print('Scan BigWig files')
    res = []
    n = len(st)
    for i, (srr, trow) in enumerate(st.iterrows(), 1):
        print(f'{srr} ({i}/{n})')
        fn = f'{fdn_data}{srr}.bw'
        cond = trow['Antibody']
        ctype = trow['Cell']
        with pyBigWig.open(fn) as bw:
            for elem, row in pt.iterrows():
                chrom = row['chr']
                start = row['start']
                end = row['end']
                arr = bw.values(chrom, start, end)
                res.append({
                    'filename': srr,
                    'element': elem,
                    'Antibody': cond,
                    'Cell': ctype,
                    'array': arr,
                    })

    with open(f'{fdn_data}results_heptad.pkl', 'wb') as f:
        pickle.dump({
            'sampletable': st,
            'peaktable': pt,
            'result': res,
            }, f)
