# spaDWGCN

This repository introduces spaDWGCN for spatial domain identification. The method first creates two biologically meaningful view features, gene expression and microenvironment. By introducing distance, a crucial element in spatial transcriptomics, this method achieves state-of-the-art performance in various datasets from different ST platforms.

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

For further study, we recommend installing Jupyter Notebook.

```
pip install jupyter
```

In addition, we provide a environment.yaml file to ensure that all dependencies could be installed correctly and smoothly.

```
conda env create -f environment.yaml
conda activate spaDWGCN_env
R
install.packages("mclust")
```

## Usage

You can run the main script with the following command-line arguments. Below are the configurable parameters, their default values, and descriptions:

### Command-line Arguments

- **`--source_data`** (str, default: `"./STARmap_20180505_BY3_1k_20251015172221.h5ad"`)
  Path or name to load the AnnData object.

- **`--fig_save_path`** (str, default: `"./temp_test/"`)
  Directory path to save the generated figures and analysis results.

- **`--epochs`** (int, default: `200`)
  Total number of training epochs for the model.

- **`--spatial_reg_parameter`** (int, default: `10`)
  Hyperparameter for spatial regularization. Controls the strength of spatial constraints.

- **`--layer_encoder`** (list of int, default: `[256, 128]`)
  Number of neurons in each layer of the GCN Encoder. Provide two integers separated by space.

- **`--layer_decoder`** (list of int, default: `[128, 256]`)
  Number of neurons in each layer of the GCN Decoder. Provide two integers separated by space.

- **`--estimate_clust_num`** (int, default: `7`)
  Estimated number of clusters for Mclust. Adjust based on biological prior knowledge.

- **`--cuda`** (int, default: `1`)
  CUDA device ID to use for GPU acceleration. Set to `-1` if running on CPU.

- **`--exp_neighbor`** (int, default: `6`)
  Number of neighbors used to construct the gene expression graph.

- **`--env_neighbor`** (int, default: `18`)
  Number of neighbors used to construct the spatial microenvironment graph.

- **`--ini_res`** (float, default: `1.0`)
  Initial resolution parameter for the preliminary Leiden clustering.

- **`--dist_mode`** (str, default: `"lnv1"`)
  Distance kernel mode used for spatial calculations.

- **`--res_set`** (float, default: `0.5`)
  Final chosen Leiden resolution. Controls the granularity of the final spatial domains.

- **`--cluster_mode`** (str, default: `leiden`)
  Cluster mode for final embedding clustering. 


### Example Usage

```bash
git clone https://github.com/Iscador/spaDWGCN.git
cd spaDWGCN
#for STARMAP dataset (as a guidance)
python "./spaDWGCN.py" --source_data "./STARmap_20180505_BY3_1k_20251015172221.h5ad" --fig_save_path "./test/" --cluster_mode mclust
```

The cluster results can be found in the corresponding folder:
<img width="651" height="318" alt="mclust7" src="https://github.com/user-attachments/assets/36f6179e-83be-4954-95cc-6736c9d95ce3" />

