r"""Gene conformation simulations based on TWLC mechanics

This module imports simulation data produced by the gene_conf module, and performs
smoothing and continuous binning comparisons through wavelet peak finding, as well
as p-value computation using various Poisson processes

Code written by John Corrette using code previously developed by Bruno Beltran, 
Deepti Kannan, and Andy Spakowitz

"""

from stochastic_gene_conf import gene_conf

import numpy as np
import numpy.matlib
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os 
import shutil
import time
import pickle
import pandas as pd
import warnings
import sys
from mpl_toolkits import mplot3d
from scipy import sparse
from scipy.signal import find_peaks
from scipy.signal import find_peaks_cwt
from scipy.signal import peak_widths
from scipy.signal import cwt
from scipy.signal import ricker
from scipy.signal import _peak_finding
from scipy.signal import savgol_filter
from scipy.signal import peak_prominences
from scipy.signal import general_gaussian
from scipy.signal import fftconvolve
from scipy.signal import lfilter
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import wasserstein_distance
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
from pathlib import Path
from importlib import reload
from os import listdir
from os.path import isfile, join
from random import randrange
from copy import copy
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score
from sklearn.isotonic import IsotonicRegression

def load_conf_data(gene_name,exp_name,datadir,files,N_files):
    
    with open(os.path.join(datadir,gene_name,exp_name,'variables.pkl'), 'rb') as file:
        temp = pickle.load(file)
        location = temp[0]
        signal = temp[1]
        Gene_start = temp[2]
        Gene_end = temp[3]
        TSS_vec = temp[4]
        accessible_starts = temp[5]
        accessible_ends = temp[6]
        cdf = temp[7]
        half_wrap = temp[8]
        
    flag = True
    idx = 0 
    tss_idx = TSS_vec[0]
    
    if N_files >= len(files):
        N_files = int(len(files))
        
    for i in range(N_files):
        strtemp = files[i]
        if strtemp[0] != 'v':
            with open(os.path.join(datadir,gene_name,exp_name,strtemp), 'rb') as file:

                    tss_contact_temp = pickle.load(file)
                    tss_contact_temp = np.array(sparse.csc_matrix.todense(tss_contact_temp[0]))[0]
            if flag:
                N = 0
                tss_contact = np.zeros(len(tss_contact_temp))
                flag = False
            tss_contact = tss_contact+tss_contact_temp
            N += 1
            idx += 1

    x = location[:-1]

    xplot = x/1e3-x[tss_idx]/1e3

    data_mean = np.copy(tss_contact/N)
    
    return location, signal, Gene_start, Gene_end, TSS_vec, cdf, half_wrap, x, xplot, data_mean


def load_contact_mat_data(gene_name,exp_name,datadir,files,N_files):
    
    with open(os.path.join(datadir,gene_name,exp_name,'variables.pkl'), 'rb') as file:
        temp = pickle.load(file)
        location = temp[0]
        signal = temp[1]
        Gene_start = temp[2]
        Gene_end = temp[3]
        TSS_vec = temp[4]
        accessible_starts = temp[5]
        accessible_ends = temp[6]
        cdf = temp[7]
        half_wrap = temp[8]
        
    flag = True
    idx = 0 
    tss_idx = TSS_vec[0]
    
    if N_files >= len(files):
        N_files = int(len(files))
        
    for i in range(N_files):
        strtemp = files[i]
        if strtemp[0] != 'v':
            with open(os.path.join(datadir,gene_name,exp_name,strtemp), 'rb') as file:
                    temp = pickle.load(file)
                    tss_contact_temp_poisson = temp[0]
                    tss_contact_temp_poisson = np.array(sparse.csc_matrix.todense(tss_contact_temp_poisson[0]))[0]
                    
#                     tss_contact_temp_regular = temp[4]
#                     tss_contact_temp_regular = np.array(sparse.csc_matrix.todense(tss_contact_temp_regular[0]))[0]
                    
#                     tss_contact_temp_uniform = temp[8]
#                     tss_contact_temp_uniform = np.array(sparse.csc_matrix.todense(tss_contact_temp_uniform[0]))[0]
                    
                    contact_mat_temp_poisson = temp[3]
                    # contact_mat_temp_poisson = np.array(sparse.csc_matrix.todense(contact_mat_temp_poisson))
                    
#                     contact_mat_temp_regular = temp[7]
#                     contact_mat_temp_regular = np.array(sparse.csc_matrix.todense(contact_mat_temp_regular))
                    
#                     contact_mat_temp_uniform = temp[11]
#                     contact_mat_temp_uniform = np.array(sparse.csc_matrix.todense(contact_mat_temp_uniform))
            if flag:
                N = 0
                # rog_0h = np.zeros(len(files)-1)
                tss_contact_poisson = np.zeros(len(tss_contact_temp_poisson))
                # tss_contact_regular = np.zeros(len(tss_contact_temp_regular))
                # tss_contact_uniform = np.zeros(len(tss_contact_temp_uniform))
                
                contact_mat_poisson = np.zeros(np.shape(contact_mat_temp_poisson))
                # contact_mat_regular = np.zeros(np.shape(contact_mat_temp_regular))
                # contact_mat_uniform = np.zeros(np.shape(contact_mat_temp_uniform))
                flag = False
            #r[idx,:,0] = rtemp[:,0]
            #r[idx,:,1] = rtemp[:,1]
            #r[idx,:,2] = rtemp[:,2]
            # rog_0h[idx] = rogtemp
            tss_contact_poisson = tss_contact_poisson+tss_contact_temp_poisson
            # tss_contact_regular = tss_contact_regular+tss_contact_temp_regular
            # tss_contact_uniform = tss_contact_uniform+tss_contact_temp_uniform
            
            contact_mat_poisson = contact_mat_poisson+contact_mat_temp_poisson
            # contact_mat_regular = contact_mat_regular+contact_mat_temp_regular
            # contact_mat_uniform = contact_mat_uniform+contact_mat_temp_uniform
            N += 1
            # sys.stdout.flush()
            idx += 1
    # print(str(N_files)+": Completed "+exp_name+"                                ", end='\r')

    x = location[:-1]

    xplot = x/1e3-x[tss_idx]/1e3

    # N_0h = np.size(tss_contact_0h,axis=0)

    data_mean_poisson = np.copy(tss_contact_poisson/N)
    data_mean_regular = 0
    data_mean_uniform = 0
    
    contact_mat_mean_poisson = np.copy(contact_mat_poisson/N)
    # contact_mat_mean_regular = np.copy(contact_mat_regular/N)
    # contact_mat_mean_uniform = np.copy(contact_mat_uniform/N)
    
    # contact_mat_mean_poisson = 0
    contact_mat_mean_regular = 0
    contact_mat_mean_uniform = 0
    
    return location, signal, Gene_start, Gene_end, TSS_vec, cdf, half_wrap, x, xplot, data_mean_poisson, data_mean_regular, data_mean_uniform, contact_mat_mean_poisson, contact_mat_mean_regular, contact_mat_mean_uniform

def remove_background(x,y,tss_idx,i_left,i_right,window,filter_width):
    
    background_left = IsotonicRegression(increasing=True).fit(x[i_left:(tss_idx-window)],np.log(y[i_left:(tss_idx-window)]))
    background_right = IsotonicRegression(increasing=False).fit(x[(tss_idx+window):i_right],np.log(y[(tss_idx+window):i_right]))
    y_back_left = background_left.predict(x[i_left:(tss_idx-window)])
    y_back_middle = np.log(y[(tss_idx-window):(tss_idx+window)])
    y_back_right = background_right.predict(x[(tss_idx+window):i_right])
    y_temp_back = np.concatenate((y_back_left,y_back_middle,y_back_right))
    y_filtered = np.log(y[i_left:i_right])-y_temp_back
    y_filtered[y_filtered<=0] = np.log(1e-6)


    y_filtered = savgol_filter(y_filtered,filter_width+1,3)
    
    

    y_filtered[(tss_idx-int(2*window)):(tss_idx+int(2*window))] = np.mean(np.concatenate((y_filtered[0:(tss_idx-int(1.5*window))],y_filtered[(tss_idx+int(1.5*window)):])))

    
    dftemp = pd.Series(y_filtered)
    r = dftemp.rolling(window=filter_width)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    m[np.isnan(m)] = -5
    s[np.isnan(s)] = 1
    z = (dftemp-m)/s
    
    y_filtered = z.to_numpy()

    y_filtered[(tss_idx-int(2*window)):(tss_idx+int(2*window))] = 0.0
    
    return y_filtered

def define_peaks(y,wavelet_low,wavelet_high,tolerance,L):

    wavelet_array = np.arange(wavelet_low,wavelet_high)
    try:
        valleys = find_peaks_cwt(1-y,wavelet_array,min_length = int(tolerance*len(wavelet_array)))
        valleys = np.sort(np.append(valleys,-0))
        valleys = np.sort(np.append(valleys,L-1))
    except:
        valleys = []
    
    try:
        peaks = find_peaks_cwt(y,wavelet_array,min_length = int(tolerance*len(wavelet_array)))
    except:
        peaks = []
        
    if len(peaks)>0:
        peaks = peaks[y[peaks]>0]
    
    return valleys,peaks

def define_bins(x,y,valleys,peaks,peak_height):
    
    edges_left = np.zeros(len(peaks)).astype(int)
    edges_right = np.zeros(len(peaks)).astype(int)
    
    y_norm = np.zeros(len(y))
    
    for i in range(len(peaks)):
        
        minV_left = -1
        minV_right = -1
        min_val_left = np.inf
        min_val_right = np.inf
        for j in range(len(valleys)):
            if valleys[j] < peaks[i] and np.abs(valleys[j]-peaks[i])<min_val_left:
                minV_left = valleys[j]
                min_val_left = np.abs(valleys[j]-peaks[i])
            elif valleys[j] > peaks[i] and np.abs(valleys[j]-peaks[i])<min_val_right:
                minV_right = valleys[j]
                min_val_right = np.abs(valleys[j]-peaks[i])
        
        temp = np.arange(minV_left,peaks[i]).astype(int)
        y_norm[temp] = (y[temp]-np.min(y[temp]))/(np.max(y[temp])-np.min(y[temp]))
        y_norm[np.isnan(y_norm)] = 0.0

        temp = np.flip(np.arange(minV_left,peaks[i])).astype(int)
        edges_left[i] = x[minV_left]

        tol = 1e-3

        jx = temp[np.where(np.abs(y_norm[temp]-peak_height)<=tol)]
        while(np.isnan(np.mean(jx))):
            tol += 1e-3
            jx = temp[np.where(np.abs(y_norm[temp]-peak_height)<=tol)]

        edges_left[i] = np.mean(x[jx])
        
        temp = np.arange(peaks[i],minV_right).astype(int)
        y_norm[temp] = (y[temp]-np.min(y[temp]))/(np.max(y[temp])-np.min(y[temp]))
        y_norm[np.isnan(y_norm)] = 0.0

        temp = np.arange(peaks[i],minV_right)
        edges_right[i] = x[minV_right]

        tol = 1e-3

        jx = temp[np.where(np.abs(y_norm[temp]-peak_height)<=tol)]
        while(np.isnan(np.mean(jx))):
            tol += 1e-3
            jx = temp[np.where(np.abs(y_norm[temp]-peak_height)<=tol)]

        edges_right[i] = np.mean(x[jx])
        
    temp = edges_left!=edges_right
    
    edges_left = edges_left[temp]
    edges_right = edges_right[temp]
        
    return edges_left,edges_right
    
def comp_prominence(x,y,edges_left,edges_right):
    
    prom = np.zeros(len(edges_left))
    bins  = np.zeros(len(edges_left))
    
    for j in range(len(edges_left)):

        idx_left_temp = np.argmin(np.abs(edges_left[j]-x))
        idx_right_temp = np.argmin(np.abs(edges_right[j]-x))

        d_bin_temp = np.diff(np.array([x[idx_left_temp],x[idx_right_temp]]))[0]

        prom_temp = np.trapz(y[idx_left_temp:idx_right_temp],x=(x[idx_left_temp:idx_right_temp]))/d_bin_temp

        prom[j] = prom_temp
        bins[j] = d_bin_temp
        
    return prom,bins

def bin_finding_algorithm(x,y,wavelet_low,wavelet_high,tolerance,peak_height,L):
    
    valleys,peaks = define_peaks(y,wavelet_low,wavelet_high,tolerance,L)
    
    I = np.unique(np.sort(np.concatenate((valleys,peaks))))

    cs = PchipInterpolator(x[I],y[I])

    xspline = np.linspace(x[0],x[-1],L)
    yspline = cs(xspline)
    
    edges_left,edges_right = define_bins(xspline,yspline,valleys,peaks,peak_height)
    prom,bins = comp_prominence(xspline,yspline,edges_left,edges_right)
    
    return valleys,peaks,edges_left,edges_right,prom,bins

def bin_signal(interp_array,edges_left,edges_right,prom,bins,prom_val,bin_val):
    
    signal = np.zeros(len(interp_array))
    
    for i in range(len(interp_array)):
        for j in range(len(edges_left)):
            if prom[j] > prom_val and bins[j] > bin_val:
                if (interp_array[i] > edges_left[j]) and (interp_array[i] <= edges_right[j]):
                    signal[i] = 1
                    break
    return signal

def compare_signals(interp_array,signal_input,signal_gt):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i in range(len(interp_array)):

        if signal_input[i] == 1 and signal_gt[i] == 1:
            TP += 1
        elif signal_input[i] == 0 and signal_gt[i] == 0:
            TN += 1
        elif signal_input[i] == 1 and signal_gt[i] == 0:
            FP += 1
        else:
            FN += 1

    if TP*FP*TN*FN > 0:
        TPR = TP/(TP+FN)
        FPR = FP/(TN+FP)
        PPV = TP/(TP+FP)
    else:
        TPR = np.nan
        FPR = np.nan
        PPV = np.nan

    prev = np.sum(signal_input)/len(interp_array)
    
    return TPR,FPR,PPV,prev

def shrink(data, cols): #Coarse graining function
    shrunk = np.zeros(cols)
    for j in range(0,cols):
        col_sp = data.shape[0]/cols
        zz = data[int(j*col_sp) : int(j*col_sp + col_sp)]
        shrunk[j] = np.mean(zz)
    return shrunk

def process_signal(location_raw,signal_raw,xplot,alpha,tss_idx,window,filter_width,val):
    i_left=np.argmin(np.abs(xplot+alpha))
    i_right=np.argmin(np.abs(xplot-alpha))
    
    signal_smooth = remove_background(location_raw,signal_raw,tss_idx,i_left,i_right,window,filter_width)
    signal_coarse = shrink(signal_smooth[i_left:i_right],val)
    signal_coarse = (signal_coarse-np.min(signal_coarse))/(np.max(signal_coarse)-np.min(signal_coarse))
    
    return signal_coarse

def fit_hyperparams(location,signal_A,signal_B,wavelet_vec_low,wavelet_vec_high,tolerance_vec,delta):
    maxval = -np.inf

    idx = 0
    lim = 1/(len(wavelet_vec_low)*len(wavelet_vec_high)*len(tolerance_vec))

    TPR = np.zeros([len(wavelet_vec_low),len(wavelet_vec_high),len(tolerance_vec)])
    FPR = np.zeros([len(wavelet_vec_low),len(wavelet_vec_high),len(tolerance_vec)])
    
    for ix in range(len(wavelet_vec_low)):
        for jx in range(len(wavelet_vec_high)):
            for kx in range(len(tolerance_vec)):
                wavelet_low = wavelet_vec_low[ix]
                wavelet_high = wavelet_vec_high[jx]
                tolerance = tolerance_vec[kx]

                valleys_A,peaks_A = define_peaks(signal_A,wavelet_low,wavelet_high,tolerance,len(signal_A))
                valleys_B,peaks_B = define_peaks(signal_B,wavelet_low,wavelet_high,tolerance,len(signal_B))
                
                if np.min(np.array([len(peaks_A),len(peaks_B)])) >0:

                    distance_AtoB = np.zeros(len(peaks_A))
                    distance_BtoA = np.zeros(len(peaks_B))

                    for j in range(len(peaks_A)):
                        jxx = np.argmin(np.abs(location[peaks_B]-location[peaks_A[j]]))
                        distance_AtoB[j] = np.abs(location[peaks_A[j]]-location[peaks_B[jxx]])

                    for j in range(len(peaks_B)):
                        jxx = np.argmin(np.abs(location[peaks_A]-location[peaks_B[j]]))
                        distance_BtoA[j] = np.abs(location[peaks_B[j]]-location[peaks_A[jxx]])

                    distance_AtoB = np.sort(distance_AtoB)
                    distance_BtoA = np.sort(distance_BtoA)

                    val1 = sum(distance_AtoB<=delta)/len(distance_AtoB)
                    val2 = sum(distance_BtoA<=delta)/len(distance_BtoA)

                    TPR[ix,jx,kx] = val2
                    FPR[ix,jx,kx] = 1-val1

                    metric = np.min([val1,val2])

                    if metric > maxval and metric < 1.0 and metric > 0.0:
                        maxval = metric
                        maxindex = (ix,jx,kx)

                idx += lim
                sys.stdout.flush()
                print("Hyperparameter Fit Progress: "+str(idx), end='\r')

    sys.stdout.flush()
    print("Complete!                                                          ", end='\r')
    
    return maxindex

def fit_hyperparams_with_gt(location,signal,peaks_GT,wavelet_vec_low,wavelet_vec_high,tolerance_vec,delta):
    maxval = -np.inf
    # prev_val = 1.0

    idx = 0
    lim = 1/(len(wavelet_vec_low)*len(wavelet_vec_high)*len(tolerance_vec))
    
    TPR = np.zeros([len(wavelet_vec_low),len(wavelet_vec_high),len(tolerance_vec)])
    FPR = np.zeros([len(wavelet_vec_low),len(wavelet_vec_high),len(tolerance_vec)])

    for ix in range(len(wavelet_vec_low)):
        for jx in range(len(wavelet_vec_high)):
            for kx in range(len(tolerance_vec)):
                wavelet_low = wavelet_vec_low[ix]
                wavelet_high = wavelet_vec_high[jx]
                tolerance = tolerance_vec[kx]
                
                valleys,peaks = define_peaks(signal,wavelet_low,wavelet_high,tolerance,len(signal))
                
                # if len(peaks) >0:

                distance_peakstoGT = np.zeros(len(peaks))
                distance_GTtopeaks = np.zeros(len(peaks_GT))

                for j in range(len(peaks)):
                    jxx = np.argmin(np.abs(location[peaks_GT]-location[peaks[j]]))
                    distance_peakstoGT[j] = np.abs(location[peaks[j]]-location[peaks_GT[jxx]])

                for j in range(len(peaks_GT)):
                    jxx = np.argmin(np.abs(location[peaks]-location[peaks_GT[j]]))
                    distance_GTtopeaks[j] = np.abs(location[peaks_GT[j]]-location[peaks[jxx]])

                distance_peakstoGT = np.sort(distance_peakstoGT)
                distance_GTtopeaks = np.sort(distance_GTtopeaks)

                val1 = sum(distance_peakstoGT<=delta)/len(distance_peakstoGT)
                val2 = sum(distance_GTtopeaks<=delta)/len(distance_GTtopeaks)

                TPR[ix,jx,kx] = val2
                FPR[ix,jx,kx] = 1-val1

                metric = np.min([val1,val2])

                if metric > maxval and metric < 1.0 and metric > 0.0:
                    maxval = metric
                    maxindex = (ix,jx,kx)

                idx += lim
                sys.stdout.flush()
                print("Hyperparameter Fit Progress: "+str(idx), end='\r')

    sys.stdout.flush()
    print("Complete!                                                          ", end='\r')
    
    return maxindex

def compute_metric(P,args):
    wavelet_low = P[0]
    wavelet_high = P[1]
    tolerance = P[2]
    
    location = args[0]
    signal_A = args[1]
    signal_B = args[2]
    delta = args[3]
    window = args[4]
    
    valleys_A,peaks_A = define_peaks(signal_A,wavelet_low,wavelet_high,tolerance,len(signal_A))
    peaks_A = filter_peaks(location,peaks_A,window)
    valleys_B,peaks_B = define_peaks(signal_B,wavelet_low,wavelet_high,tolerance,len(signal_B))
    peaks_B = filter_peaks(location,peaks_B,window)

    if np.min(np.array([len(peaks_A),len(peaks_B)])) >0:

        distance_AtoB = np.zeros(len(peaks_A))
        distance_BtoA = np.zeros(len(peaks_B))

        for j in range(len(peaks_A)):
            jxx = np.argmin(np.abs(location[peaks_B]-location[peaks_A[j]]))
            distance_AtoB[j] = np.abs(location[peaks_A[j]]-location[peaks_B[jxx]])

        for j in range(len(peaks_B)):
            jxx = np.argmin(np.abs(location[peaks_A]-location[peaks_B[j]]))
            distance_BtoA[j] = np.abs(location[peaks_B[j]]-location[peaks_A[jxx]])

        distance_AtoB = np.sort(distance_AtoB)
        distance_BtoA = np.sort(distance_BtoA)

        val1 = sum(distance_AtoB<=delta)/len(distance_AtoB)
        val2 = sum(distance_BtoA<=delta)/len(distance_BtoA)

        metric = 1-np.min([val1,val2])
    
    else:
        metric = 1

    return metric

def compute_metric_with_gt(P,args):
    wavelet_low = P[0]
    wavelet_high = P[1]
    tolerance = P[2]
    
    location = args[0]
    signal = args[1]
    peaks_GT = args[2]
    delta = args[3]
    window = args[4]
    
    valleys,peaks = define_peaks(signal,wavelet_low,wavelet_high,tolerance,len(signal))
    peaks = filter_peaks(location,peaks,window)
                
    if len(peaks) >0:

        distance_peakstoGT = np.zeros(len(peaks))
        distance_GTtopeaks = np.zeros(len(peaks_GT))

        for j in range(len(peaks)):
            jxx = np.argmin(np.abs(location[peaks_GT]-location[peaks[j]]))
            distance_peakstoGT[j] = np.abs(location[peaks[j]]-location[peaks_GT[jxx]])

        for j in range(len(peaks_GT)):
            jxx = np.argmin(np.abs(location[peaks]-location[peaks_GT[j]]))
            distance_GTtopeaks[j] = np.abs(location[peaks_GT[j]]-location[peaks[jxx]])

        distance_peakstoGT = np.sort(distance_peakstoGT)
        distance_GTtopeaks = np.sort(distance_GTtopeaks)

        val1 = sum(distance_peakstoGT<=delta)/len(distance_peakstoGT)
        val2 = sum(distance_GTtopeaks<=delta)/len(distance_GTtopeaks)

        metric = 1-np.min([val1,val2])
        
    else:
        metric = 1
        
    return metric

def filter_peaks(location,peaks,window):
    
    peaks_filtered = []
    
    for i in range(len(peaks)):
        if (np.abs(location[peaks[i]]) > 1.05*window and np.abs(location[peaks[i]] - location[0]) > 1 and np.abs(location[peaks[i]] - location[-1]) > 1):
            peaks_filtered.append(peaks[i])
            
    peaks_filtered = np.array(peaks_filtered)
    
    return peaks_filtered

def compute_occlusion(r,R):
    
    occ_prob = np.zeros(len(r))
    
    for j in range(len(r)):
        
        if j != 0 and j != len(r)-1:
    
            t1 = r[j+1,:]-r[j,:]
            t2 = r[j-1,:]-r[j,:]

            r_lig = np.cross(t1,t2)/np.linalg.norm(np.cross(t1,t2))

            if np.min(np.linalg.norm(r-(R*r_lig+r[j,:]),axis=1)) < R:
                occ_prob[j] = 1
            else:
                occ_prob[j] = 0
    occ_prob[0] = occ_prob[1]
    occ_prob[-1] = occ_prob[-2]
    return occ_prob
        
def compute_promoter_occlusion(r,R,tss_idx,strand):
    
    if strand == '+':
        promoter = np.arange(tss_idx-2000,tss_idx+500)
    else:
        promoter = np.arange(tss_idx-500,tss_idx+2000)
        
    occ_prob = np.zeros(len(promoter))
    
    for j in range(len(promoter)):
    
        t1 = r[j+1,:]-r[j,:]
        t2 = r[j-1,:]-r[j,:]

        r_lig = np.cross(t1,t2)/np.linalg.norm(np.cross(t1,t2))

        if np.min(np.linalg.norm(r-(R*r_lig+r[j,:]),axis=1)) < R:
            occ_prob[j] = 1
        else:
            occ_prob[j] = 0

    return occ_prob


def compute_occlusion_radius(r,R_max):
    
    occ_rad = np.zeros(len(r))
    
    for j in range(len(r)):
        
        if j != 0 and j != len(r)-1:
    
            t1 = r[j+1,:]-r[j,:]
            t2 = r[j-1,:]-r[j,:]

            r_lig = np.cross(t1,t2)/np.linalg.norm(np.cross(t1,t2))
            
            R2 = R_max
            R1 = 0
            err = 1
            while err > 1e-4:
                R = np.mean([R1,R2])
                if np.min(np.linalg.norm(r-(R*r_lig+r[j,:]),axis=1)) < R:
                    R2 = R
                else:
                    R1 = R
                err = np.abs(R-np.mean([R1,R2]))
                
            occ_rad[j] = R
         
    occ_rad[0] = occ_rad[1]
    occ_rad[-1] = occ_rad[-2]        
    return occ_rad

    