# ATAC-EM Polymer Model
 A twlc model informed by nucleosome positioning in ATAC-seq and EM to predict long range genomic contacts as described in [Corrette et al. "Nucleosome placement and polymer mechanics explain genomic contacts on 100kbp scales"](https://doi.org/10.1101/2024.09.24.614727).

## Anaconda Environment
 Conda environment dependencies can be found in the environment.yml file. To create a new conda environment, in a conda terminal run:
     
 ```
 conda env create -f environment.yml -n atacem
 ```
     
## Notebooks
 Notebooks are provided with sample code for 4C, EM, and ATAC/EM analysis. Metadata required is found in the data folder.

 * [4C Bedgraph Analysis](notebooks/4C_Bedgraph_Analysis.ipynb)
 * [EM Analysis](notebooks/EM_Analysis.ipynb)
 * [Nucleosome Spacing Analysis](notebooks/Nucleosome_Spacing_Analysis.ipynb)
 * [NucleoATAC Analysis](notebooks/NucleoATAC_Analysis.ipynb)
 * [ATACEM Contact Analysis](notebooks/ATACEM_Contact_Analysis.ipynb)

 
## Slurm/.py scripts
 Python scripts to produce a polymer conformation sample are provided with the associated slurm scripts needed to run them. The slurm scripts with need the account name and partition fields updated based on the HPC platform that you are using. NOTE: The computation and data storage requirements is expensive when running these scripts.
