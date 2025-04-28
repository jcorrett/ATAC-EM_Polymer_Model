# ATAC-EM_Polymer_Model
 A twlc model informed by nucleosome positioning in ATAC-seq and EM to predict long range genomic contacts

# Anaconda Environment
 Conda environment dependencies can be found in the environment.yml file. To create a new conda environment, in a conda terminal run:
     conda env create -f environment.yml -n atacem
     
# Notebooks
 Notebooks are provided with sample code for 4C, EM, and ATAC/EM analysis. Metadata required is found in the data folder
 
# Slurm/.py scripts
 Python scripts to produce a polymer conformation sample are provided with the associated slurm scripts needed to run them. The slurm scripts with need the account name and partition fields updated based on the HPC platform that you are using. NOTE: The computation and data storage requirements is expensive when running these scripts.