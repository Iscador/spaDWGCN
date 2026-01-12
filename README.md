# spaDWGCN

This repository introduces a biologically meaningful algorithm spaDWGCN, for self-supervised spatial domain identification. The method first creates two biologically meaningful view features, gene expression and microenvironment. By introducing distance, a crucial element in spatial transcriptomics, this method achieves state-of-the-art performance in various datasets from different ST platforms.

## Installations

First, we recommend creating a Python 3.9 virtual environment to perform our algorithm.
```bash
conda init
conda create -n spaDWGCN_env python=3.9
conda activate spaDWGCN_env
```

Installing Pytorch and the corresponding packages like torch-geometric. Please note that torch, CUDA version, and Python version should be compatible.

```bash
pip install torch
pip install torch-geometric
pip install numpy==1.22.4
```

We need scanpy for spatial transcriptomics research; squidpy is optional.

```bash
pip install scanpy==1.9.1
pip install squidpy==1.2.3
pip install matplotlib==3.6.2
```

For mclust algorithm, we need to install the R basic environment(optional).

```
conda install r-base r-essentials
conda install rpy2
R
install.packages("mclust")
```

For further study, we install Jupyter Notebook.

```
pip install jupyter
```

## Usage
```bash
git clone https://github.com/Iscador/spaDWGCN.git
cd spaDWGCN
#for STARMAP dataset (as a guidance)
python "./spaDWGCN.py" --source_data "./STARmap_20180505_BY3_1k_20251015172221.h5ad" --fig_save_path "./test/"
```

The cluster results can be found in the corresponding folder:

![mclust7](https://github.com/user-attachments/assets/ab549895-79f6-4fff-bf99-6c6fcd6fae8a)

The time cost for STARMAP dataset should be less than 1 minute.
