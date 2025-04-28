## Simulates a "null" model that does not include any ATAC-seq information

## Import wlcstat and gene-conf packages ##
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', 'wlcstat-master')))

import wlcstat.chromo

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', 'stochastic-gene-conf-master')))
from stochastic_gene_conf import gene_conf

## Import other dependencies ##
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import shutil
import time
import pickle
import pandas as pd
from mpl_toolkits import mplot3d
from scipy import sparse
from scipy.signal import find_peaks
from scipy.stats import norm
from scipy.stats import uniform
from pathlib import Path

## Collect input arguments ##
parser = argparse.ArgumentParser()

parser.add_argument("--mode", required = True, choices=["ATAC", "Multi"], help="option to load in just ATAC or to include expression data")

parser.add_argument("--output_dir", required=True, help="where to put output files")

parser.add_argument("--exp_name", type=str, required=True, help="name of experiment to perform analyis on")
parser.add_argument("--gene_name",type=str, help="name of gene to perform analysis on")

parser.add_argument("--half_wrap", type=int, default=73, help="amount of DNA (bp) wrapped around half of a nucleosome bead")
parser.add_argument("--link_model",required=True, choices=["peaks", "sample", "count"],help="choice of linker length model")

parser.add_argument("--contact_distance", type=float, default=5.0, help="distance in nm for two parts of DNA to be considered in contact")
parser.add_argument("--num_sims", type=int,default=1, help="number of simulations to perform and average over")

parser.add_argument("--job_id", type=str, default="NULL", help="slurm job id")
parser.add_argument("--task_id", type=str, default="NULL", help="slurm task id")

a = parser.parse_args()

## Define strings for output file name ##
filename= a.output_dir + '/' + a.gene_name + '/' + a.exp_name

## Check if variables have been defined ##
my_file = Path(filename+'/variables.pkl')
if my_file.is_file():
    
    ## Load variables ##
    if a.mode == "Multi":
        with open(filename+'/variables.pkl', 'rb') as file:
            temp = pickle.load(file)
            expression = temp[0]
            location = temp[1]
            signal = temp[2]
            Gene_start = temp[3]
            Gene_end = temp[4]
            TSS_vec = temp[5]
            accessible_starts = temp[6]
            accessible_ends = temp[7]
            cdf = temp[8]
            half_wrap = temp[9]
    else:
        with open(filename+'/variables.pkl', 'rb') as file:
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
                
    if a.job_id == "NULL":
        str_job = ""
    else:
        str_job = a.job_id

    if a.task_id == "NULL":
        str_task = ""
    else:
        str_task = "_"+a.task_id

    ## Simulate 3d Conformation ##
    
    links = np.array([0])
    
    packed_link = np.random.randint(30,91) #Randomly samples a mean linker dna length
    
    while np.min(links)<1:
        ## Regular Model
        # nucs = gene_conf.sample_cdf_uniform(location,signal,cdf,accessible_starts,
        #                                     accessible_ends,half_wrap,a.packed_link,0)
        
        ## Uniform Model
        # nucs = gene_conf.sample_cdf_uniform(location,signal,cdf,accessible_starts,
        #                                     accessible_ends,half_wrap,a.packed_link,a.packed_sig)
        
        ## Poisson Model
        nucs = gene_conf.sample_cdf_poisson(location,signal,cdf,accessible_starts,
                                            accessible_ends,half_wrap,packed_link)

        if a.link_model == "peaks":
            links = gene_conf.create_links_peaks(location,signal,nucs,half_wrap)
        elif a.link_model == "sample":
            links = gene_conf.create_links_sample(location,signal,nucs,half_wrap)
        else:
            links = gene_conf.create_links_count(location,signal,nucs,half_wrap)

    tss_idx = TSS_vec[0]
    rog_mean = 0
    r2_mean = 0
    
    for i in range(a.num_sims):
        r, rdna1, rdna2, rn, un = wlcstat.chromo.gen_chromo_conf(links.astype(int),w_ins=half_wrap-1,w_outs=half_wrap)
        r = r[3*half_wrap:-3*half_wrap,0:]

        rmean = np.mean(r,axis=0)
        rog_mean = rog_mean + np.mean(np.linalg.norm(r-rmean,axis=1)**2)/a.num_sims
        r2_mean = r2_mean + (np.linalg.norm(r[0,:]-r[-1,:])**2)/a.num_sims
    
        
        tss_distance = np.linalg.norm(r-r[tss_idx,:],axis=1)
    
        if i==0:
            tss_contact = np.zeros(np.size(tss_distance))
        tss_contact_temp = np.zeros(np.size(tss_distance))
        tss_contact_temp[tss_distance<(a.contact_distance/0.34)] = 1/a.num_sims
        tss_contact = tss_contact + tss_contact_temp
        
    tss_contact = sparse.csr_matrix(tss_contact)


    ## Output data ##
    with open(filename+'/data'+str_job+str_task+'.pkl', 'wb') as file:

        # A new file will be created
        pickle.dump([tss_contact,rog_mean,r2_mean,packed_link], file)
