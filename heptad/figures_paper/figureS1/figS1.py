# This script makes plots of data extracted using figS1_preprocess.py for figure S1
import pickle
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    # Loading data
    # change the following line to the figureS1 data folder 
    fdn_data = '/home/julie/DataSets/AssiDHS/'
    with open(f'{fdn_data}results_heptad.pkl', 'rb') as f:
        data = pickle.load(f)

    # Setting variables
    res = data['result']
    st = data['sampletable']
    pt = data['peaktable']
    elements = 'LYL_P_alt, TAL1_p40, LMO2_m25, FLI1_m15, ERG_p85, GATA2_m117, GATA2_p4, RUNX1_p23, RUNX1_p141'.split(', ')\

    kind = st['type'].tolist()
    kinds = [] 
    for i in kind: 
        if i not in kinds: 
            kinds.append(i)        

            samples = st['filename'].tolist()

    #Define colors
    colors = sns.color_palette('Paired', n_colors=len(kinds))
    cmap_DHS = dict(zip(kinds, colors))
    cmap_DHS['RUNX1'] =(0.996078431372549, 0.8509803921568627,0.4627450980392157) #change shade of yellow

    #Make height variable for manual graph labelling in Illustrator
    height = []

    #Plotting
    fig, axs = plt.subplots(len(samples), len(elements), figsize=(10, 20))
    for i in range(len(samples)):
        sample = samples[i]
        ymax = 0
        
        for j in range(len(elements)):
            elem = elements[j]
            ax = axs[i, j]
            arrs = []
            for re in res:
                if re['filename'] != sample:
                    continue
                if re['element'] != elem:
                    continue
                arrs.append(re['array'])
                co = cmap_DHS.get(re['type'])
                        
            y = np.array(arrs).sum(axis=0)
            x = np.arange(len(y)) + pt.at[elem, 'start']       
            ax.fill_between(x, 0, y, alpha=1, color = co)
            if i == 0:
                ax.set_title(elem.replace('_', '\n', 1), fontsize=9)
            if j == 0:
                ax.set_ylabel(sample, va='center', ha='right', rotation=0, labelpad=15)
            ax.set_yticks([])
            ax.set_xticks([])
            ymax = max(ymax, y.max())
            for j in range(len(elements)):
                ax = axs[i, j]
                ax.set_ylim(top=ymax*1.1)
        height.append(ymax)
        fig.tight_layout(w_pad=0, h_pad=0)
    if True:
        fxf = 'distribution_profiles'
        for ext in ['png', 'svg']:
             fig.savefig(f'{fxf}.{ext}')  
    np.savetxt('height.txt', height, delimiter='\t')