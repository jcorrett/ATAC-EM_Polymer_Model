r"""Gene conformation simulations based on TWLC mechanics

This module imports data produced by NucleoATACR, Seurat, and Biomart to compute
conformations of regions of the genome based on TWLC mechanics and nucleosome
positioning.

Code written by John Corrette using code previously developed by Bruno Beltran, 
Deepti Kannan, and Andy Spakowitz

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import shutil
import time
import pickle
import pandas as pd
from mpl_toolkits import mplot3d
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import weibull_min
from sklearn.isotonic import IsotonicRegression

default_half_wrap = 73 # bp half nucleosome wrapping
default_packed_link = 60 # bp mean spacing
default_packed_sig = 10 # bp uniform position dist width
default_packed_lambda = 20 # bp weibull scale
default_packed_k = 2 # weibull scale

def import_gene_expression(seurat_path,exp_name,i_gene):
    df_seurat = pd.read_excel(seurat_path)
    
    gene_expression = df_seurat[exp_name][i_gene]
    
    return gene_expression

def import_gene_body(seurat_path,nucleoatacr_path,biomart_path,exp_name,i_gene,
                half_wrap=default_half_wrap):
    ## READ DATA ##
    df_seurat = pd.read_excel(seurat_path)
    df_biomart = pd.read_excel(biomart_path)

    Gene_name = df_seurat.Gene_Name[i_gene]
    tss_vec = []
    for i in range(len(df_biomart)):
        if (df_biomart.Name[i] == Gene_name):
            tss_vec.append(df_biomart.TSS[i].astype(int))
            Gene_start = df_biomart.Start[i].astype(int)
            Gene_end = df_biomart.End[i].astype(int)
    tss_vec = np.array(tss_vec)-Gene_start
    
    df_gene = pd.read_csv(os.path.join(nucleoatacr_path,Gene_name,exp_name+".csv"))
    
    location = np.array(df_gene.location)
    location = location[0:(Gene_end-Gene_start)]
    
    signal = np.array(df_gene.signal)
    signal = signal[0:(Gene_end-Gene_start)]
    
    accessible_starts = []
    accessible_ends = []
    
    if exp_name == "NAKED":
        signal_temp = np.copy(signal)

        for i in range(len(location)):
            signal_temp[i] = 0
        
    elif exp_name == "NULL":
        
        signal_temp = np.copy(signal)

        for i in range(len(location)):
            signal_temp[i] = 0
            
        accessible_starts.append(0)
        accessible_ends.append(len(location)-1)
        
    else:

        signal_temp = np.copy(signal)

        for i in range(len(location)):
            if i == 0:
                flag = np.isnan(signal[i])
                if flag == True:
                    signal_temp[i] = 0
            else:
                flag_prev = np.isnan(signal[i-1])
                flag = np.isnan(signal[i])
                if flag == True:
                    signal_temp[i] = 0
                    if flag_prev == False:
                        accessible_starts.append(i)
                else:
                    if flag_prev == True:
                        accessible_ends.append(i-1)

        if np.size(accessible_starts)>1 and np.size(accessible_ends)>1:
            if accessible_starts[0] > accessible_ends[0]:
                accessible_starts = np.insert(accessible_starts,0,1)
            if accessible_starts[-1] >= accessible_ends[-1]:
                accessible_ends.append(len(location)-1)

    signal = np.copy(signal_temp)

    return location, signal, Gene_name, Gene_start, Gene_end, tss_vec, accessible_starts, accessible_ends
    
def create_cdf(location,signal):
    cdf = np.zeros(len(location))     

    for var in range(len(location)):
        if signal[var]!=0:
            cdf[var] = np.trapz(signal[0:var],x=location[0:var])/np.trapz(signal,x=location)
        
    return cdf

def sample_cdf_uniform(location,signal,cdf,accessible_starts,accessible_ends,half_wrap=default_half_wrap,
                       packed_link=default_packed_link,packed_sig=default_packed_sig):
    
    nucs = np.array([])
    
    for i in range(len(accessible_starts)):
        idx = accessible_starts[i]
    
        while idx < accessible_ends[i]:
            idx = idx + packed_link + np.round(np.random.uniform(-0.5*packed_sig-0.5,0.5*packed_sig+0.5))+2*half_wrap+1
            if idx < accessible_ends[i]:
                nucs = np.append(nucs,location[int(idx)])
    
    
    peaks = find_peaks(signal,distance = 2*half_wrap+1,height = 1e-10)[0]

    break_con = False

    for i in range(len(peaks)):
        rand_val = np.random.uniform()
        if i != 0:
            j = 0
            while (np.abs(location[(np.abs(cdf-rand_val)).argmin()]-nucs).min() <= 2*half_wrap+1) and (not break_con):
                rand_val = np.random.uniform()
                j += 1
                if j > 1000:
                    break_con = True
        if (not break_con):
            nucs = np.append(nucs,location[(np.abs(cdf-rand_val)).argmin()])
        else:
            break
            
    nucs = np.sort(nucs)
    if np.abs(nucs[0]-location[0]) < 2*half_wrap+1:
        nucs[0] = location[0]
    else:
        nucs = np.sort(np.append(nucs,location[0]))
        
    if np.abs(nucs[-1]-location[-1]) < 2*half_wrap+1:
        nucs[-1] = location[-1]
    else:
        nucs = np.sort(np.append(nucs,location[-1]))
    
    return nucs

def sample_cdf_poisson(location,signal,cdf,accessible_starts,accessible_ends,half_wrap=default_half_wrap,
                       packed_link=default_packed_link):
    
    nucs = np.array([])
    
    for i in range(len(accessible_starts)):
        idx = accessible_starts[i]
    
        while idx < accessible_ends[i]:
            idx = idx + np.random.geometric(1/packed_link)+2*half_wrap+1
            if idx < accessible_ends[i]:
                nucs = np.append(nucs,location[int(idx)])
    
    
    peaks = find_peaks(signal,distance = 2*half_wrap+1,height = 1e-10)[0]

    break_con = False

    for i in range(len(peaks)):
        rand_val = np.random.uniform()
        if i != 0:
            j = 0
            while (np.abs(location[(np.abs(cdf-rand_val)).argmin()]-nucs).min() <= 2*half_wrap+1) and (not break_con):
                rand_val = np.random.uniform()
                j += 1
                if j > 1000:
                    break_con = True
        if (not break_con):
            nucs = np.append(nucs,location[(np.abs(cdf-rand_val)).argmin()])
        else:
            break
            
    nucs = np.sort(nucs)
    if np.abs(nucs[0]-location[0]) < 2*half_wrap+1:
        nucs[0] = location[0]
    else:
        nucs = np.sort(np.append(nucs,location[0]))
        
    if np.abs(nucs[-1]-location[-1]) < 2*half_wrap+1:
        nucs[-1] = location[-1]
    else:
        nucs = np.sort(np.append(nucs,location[-1]))
    
    return nucs

def sample_cdf_poisson_empty(location,signal,cdf,accessible_starts,accessible_ends,half_wrap=default_half_wrap,
                       packed_link=default_packed_link):
    
    nucs = np.array([])
    
    for i in range(len(accessible_starts)):
        idx = accessible_starts[i]
    
        while idx < accessible_ends[i]:
            idx = idx + np.random.geometric(1/packed_link)+2*half_wrap+1
            if idx < accessible_ends[i]:
                nucs = np.append(nucs,location[int(idx)])
            
    nucs = np.sort(nucs)
    if np.abs(nucs[0]-location[0]) < 2*half_wrap+1:
        nucs[0] = location[0]
    else:
        nucs = np.sort(np.append(nucs,location[0]))
        
    if np.abs(nucs[-1]-location[-1]) < 2*half_wrap+1:
        nucs[-1] = location[-1]
    else:
        nucs = np.sort(np.append(nucs,location[-1]))
    
    return nucs

def sample_cdf_weibull(location,signal,cdf,accessible_starts,accessible_ends,half_wrap=default_half_wrap,
                       packed_lambda=default_packed_lambda,packed_k=default_packed_k):
    
    nucs = np.array([])
    
    for i in range(len(accessible_starts)):
        idx = accessible_starts[i]
    
        while idx < accessible_ends[i]:
            idx = idx + np.round(weibull_min.rvs(packed_k,loc=0,scale=packed_lambda,size=1))+2*half_wrap+1
            if idx < accessible_ends[i]:
                nucs = np.append(nucs,location[int(idx)])
    
    
    peaks = find_peaks(signal,distance = 2*half_wrap+1,height = 1e-10)[0]

    break_con = False

    for i in range(len(peaks)):
        rand_val = np.random.uniform()
        if i != 0:
            j = 0
            while (np.abs(location[(np.abs(cdf-rand_val)).argmin()]-nucs).min() <= 2*half_wrap+1) and (not break_con):
                rand_val = np.random.uniform()
                j += 1
                if j > 1000:
                    break_con = True
        if (not break_con):
            nucs = np.append(nucs,location[(np.abs(cdf-rand_val)).argmin()])
        else:
            break
            
    nucs = np.sort(nucs)
    if np.abs(nucs[0]-location[0]) < 2*half_wrap+1:
        nucs[0] = location[0]
    else:
        nucs = np.sort(np.append(nucs,location[0]))
        
    if np.abs(nucs[-1]-location[-1]) < 2*half_wrap+1:
        nucs[-1] = location[-1]
    else:
        nucs = np.sort(np.append(nucs,location[-1]))
    
    return nucs
    
def create_links_peaks(location,signal,nucs,half_wrap=default_half_wrap):
    peaks = find_peaks(signal,distance = 2*half_wrap+1,height = 1e-10)[0]
    links = np.diff(np.append(location[0],np.append(np.sort(location[peaks]),location[-1])))
    
    links[0] = links[0]+2*half_wrap

    links[-1] = links[-1]+2*half_wrap
    
    links = links-2*half_wrap
    
    return links
    
def create_links_sample(location,signal,nucs,half_wrap=default_half_wrap):
    links = np.diff(np.sort(nucs))
    
    links[0] = links[0]+2*half_wrap

    links[-1] = links[-1]+2*half_wrap
    
    links = links-2*half_wrap
    
    return links

def create_links_count(location,signal,nucs,half_wrap=default_half_wrap):
    d = (location[-1]-location[0])/np.size(nucs)
    links = np.diff(np.append(location[0],np.append(np.round(np.linspace(location[0]+np.round(d),location[-1]-np.round(d),np.size(nucs))),location[-1])))
    links[0] = links[0]+2*half_wrap

    links[-1] = links[-1]+2*half_wrap
    
    links = links-2*half_wrap
    
    return links

def get_half_wrap():
    
    return default_half_wrap
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    