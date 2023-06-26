# Exploring variability patterns across fMRI statistic maps

## Table of contents
   * [How to cite?](#how-to-cite)
   * [Contents overview](#contents-overview)
   * [Installing environment](#installing-environment)

## How to cite?


## Contents overview

### `src`

This directory contains scripts and notebooks used to launch analyses.

- `louvain_matrix.py` is used to compute correlation matrix between maps of different pipelines for each group and perform community detection. It also plots the graphs and community matrices. 
- `mean_maps.py` will compute the mean statistic maps per Louvain communities based on the matrix computed using previous script. 

To launch these, just change the parameters in the script files and launch in terminal: `python3 {script_name}.py`.

### `figures`

This directory contains figures and csv files obtained when running the notebooks in the `results` directory.

## Installing environment 

To reproduce the figures, you will need to create a conda environment with the necessary packages. 

- First, download and install [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [miniforce](https://github.com/conda-forge/miniforge) if you work on a Mac Apple Silicon. 
- After this, you will need to create you own environment by running : 
```
conda create -n workEnv
conda activate workEnv
```
The name workEnv can be changed. 
- When you environment is activated, just run the `install_environment.sh` script available at the root of this directory. 
```
bash install_environment.sh
```
This script will install all the necessary packages to reproduce this analysis. 
At each step, you might have to answer y/n, answer yes any time to properly do the install. 

