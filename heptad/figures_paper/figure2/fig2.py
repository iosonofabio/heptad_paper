# This script makes plots of data extracted using fig2_preprocess.py for figure 2
import os
import sys
import pickle
from collections import Counter, defaultdict 
import numpy as np 
import pandas as pd 
from scipy.spatial.distance import pdist, cdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
	#Specify which cell type here
	cell_type = input('Enter cell type to plot: ')
	scale_factor = int(input('Enter scaling factor for minimum peak: '))

	# load peak info
	idx = ['TAL1_p40', 'LMO2_m25', 'FLI1_m15', 'LYL_P_alt', 'RUNX1_p141', 'RUNX1_p23', 'ERG_p85', 'GATA2_p4', 'GATA2_m117']
	cols = ['LYL1', 'TAL1', 'LMO2', 'FLI1', 'ERG', 'GATA2', 'RUNX1']
	peak = pd.read_excel('/home/julie/DataSets/BWgraph/JT_peakcall_check_minimal_overlap.xlsx', sheet_name = "All rel", usecols = 'A,F:M')
	peak_CD34 = peak.iloc[0:9,2:].astype(int)
	peak_CD34.index = idx
	peak_CD34.columns = cols
	peak_ME1 = peak.iloc[10:19,2:].astype(int)
	peak_ME1.index = idx
	peak_ME1.columns = cols
	peak_KG1 = peak.iloc[21:30,2:].astype(int)
	peak_KG1.index = idx
	peak_KG1.columns = cols
	peak = {'CD34':peak_CD34, 'ME1':peak_ME1, 'KG1':peak_KG1}
	peak_CD34.to_csv('peak_cd34.csv', sep = '\t')
	peak_ME1.to_csv('peak_ME1.csv', sep = '\t')
	peak_KG1.to_csv('peak_KG1.csv', sep = '\t')




	# Loading data
	# change the following line to the figure2 data folder 
	fn_res = '/home/julie/DataSets/BWgraph/results_heptad.pkl'
	with open(fn_res, 'rb') as f:
	    data = pickle.load(f)

	res = data['result']
	st = data['sampletable']
	pt = data['peaktable']   
	elements = 'LYL_P_alt, TAL1_p40, LMO2_m25, FLI1_m15, ERG_p85, GATA2_m117, GATA2_p4, RUNX1_p23, RUNX1_p141'.split(', ')
	abtypes = ['LYL1', 'TAL1', 'LMO2', 'FLI1', 'ERG', 'GATA2', 'RUNX1' ]
	heptad_colors = [(0.956862745098039,0.427450980392157,0.262745098039216), (1,0.501960784313725,0), (0.992156862745098,0.72156862745098,0.388235294117647), (0.498039215686275,0.737254901960784,0.254901960784314), (0.152941176470588,0.392156862745098,0.0980392156862745), (0.270588235294118,0.458823529411765,0.705882352941177), (0.368627450980392,0.211764705882353,0.517647058823529)]
	height = []
    

	# Plotting
	fig, axs = plt.subplots(len(abtypes), len(elements), figsize=(5.5, 4.2))
	for i in range(len(abtypes)):
	    abtype = abtypes[i]
	    ymax = 0
	    ymin = 1e10
	    for j in range(len(elements)):
	        elem = elements[j]
	        ax = axs[i, j]
	        arrs = []
	        for re in res:
	            if re['Cell'] != cell_type:
	                continue
	            if re['Antibody'] != abtype:
	                continue
	            if re['element'] != elem:
	                continue
	            arrs.append(re['array'])      
	        y = np.array(arrs).sum(axis=0) #Not sure why this is causing grief (and suddenly it's not - was an issue with mislabelled elements)
	        x = np.arange(len(y)) + pt.at[elem, 'start']
	        ax.fill_between(x, 0, y, alpha=0.7, color=heptad_colors[i])
	        if i == 0:
	            ax.set_title(elem.replace('_', '\n', 1), fontsize=9)
	        if j == 0:
	            ax.set_ylabel(abtype, va='center', ha='right', rotation=0, labelpad=15)
	        ax.set_yticks([])
	        ax.set_xticks([])
	        ymax = max(ymax, y.max())
	        if peak[cell_type].at[elem, abtype] == 1:
	        	ymin = min(ymin, y.max())
	        else:
	        	r = plt.Rectangle([0,0], 1,1, transform = ax.transAxes, lw = 2, facecolor = 'black', edgecolor = 'none', alpha = 0.15 , clip_on = False)
	        	ax.add_artist(r)
	    for j in range(len(elements)):
	        ax = axs[i, j]
	        #ax.set_ylim(top=ymax*1.1)
	        ax.set_ylim(0, ymin*scale_factor)
	    height.append(ymin*scale_factor)
	    fig.tight_layout(w_pad=0, h_pad=0)
	    if True:
	        fxf = 'distribution_profiles'
	        for ext in ['png', 'svg']:
	            fig.savefig(f'{cell_type}{fxf}.{ext}')  
	    np.savetxt(f'{cell_type}_height.txt', height, delimiter='\t')