# spaDWGCN

This repository introduces a biologically meaningful algorithm spaDWGCN, for self-supervised spatial domain identification. The method first creates two biologically meaningful view features, gene expression and microenvironment. By introducing distance, a crucial element in spatial transcriptomics, this method achieves state-of-the-art performance in various datasets from different ST platforms.

## Installations

First, we recommend creating a Python 3.9 virtual environment to perform our algorithm.
```bash
conda init
conda create -n spaDWGCN_env python=3.9
conda activate spaDWGCN_env
```

Installing Pytorch , and corresponding packages like torch-geometric. Please note that torch, CUDA version and python version should be compatible.

```bash
pip install torch
pip install torch-geometric
pip install numpy==1.22.4
```

We need scanpy for spatial transcriptomics research, squidpy is optional.

```bash
pip install scanpy
pip install squidpy
```

For mclust algorithm, we need to install R basic environment(optional).

```
conda install r-base r-essentials
conda install rpy2
install.packages("mclust")
```

For further study, we install Jupyter notebook.

```
pip install jupyter
```

## Usage

For further usage guide, please see STARMAP.ipynb.
