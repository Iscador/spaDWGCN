import warnings
warnings.filterwarnings("ignore")
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.metrics import *
import time
import matplotlib.pyplot as plt
import squidpy as sq
import anndata as ad

#graph part need
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors


import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
import torch
import pysodb

from tqdm import tqdm
from torch_geometric.data import Data

def preprocessing(adata, pca_num = 30, 
                  nonspatial_key = 'nonspatial', 
                  initial_res = 1, initial_type_store = 'celltype', 
                  verbose = False):
    if verbose:
        print("Raw data:")
        print(adata)
    
    #check adata, if having necessary part
    if 'spatial' not in adata.obsm:
        raise ValueError("No spatial coordinates exist. Please check adata.obsm.")
    
    #preprocessing adata
    #filter low quality genes and cells
    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_cells=10)

    #select HVGs
    if adata.n_vars > 4000:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=4000)
        adata = adata[:,adata.var['highly_variable']]
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=pca_num)

    #processing nonspatial leiden cluster
    sc.pp.neighbors(adata, key_added=nonspatial_key)
    sc.tl.umap(adata, neighbors_key=nonspatial_key)
    sc.tl.leiden(adata, neighbors_key=nonspatial_key, resolution=initial_res,key_added = initial_type_store)
    
    if verbose:
        print("preprocessed complete:")
        print(adata)
    preprocessed_adata = adata.copy()
    
    return preprocessed_adata

def build_expression_graph(adata, expression_neighbor = 6, dist_mode = 'lnv1', alpha = 1, diag = 1, verbose = False):
    #check adata, if having necessary part
    if 'spatial' not in adata.obsm:
        raise ValueError("No spatial coordinates exist. Please check adata.obsm.")
    
    spatial_coords = adata.obsm['spatial']

    #create expression KNN graph, in distance mode
    nbr = NearestNeighbors(algorithm='ball_tree').fit(spatial_coords)
    distance_matrix = nbr.kneighbors_graph(n_neighbors = expression_neighbor ,mode="distance")
    if verbose:
        print("original distance is: ", distance_matrix.data)
    distance_matrix = calculate_distance_kernel(distance_matrix= distance_matrix, nbr= nbr, distance_mode = dist_mode)
    #default to_array = True, distance_matrix is an array object

    #add self loop
    mat_st = np.eye(adata.n_obs)# same scale diag matrix
    distance_matrix = alpha * distance_matrix + diag * mat_st #for array data only
    
    #save expression graph in adata.obsm
    adata.obsm['dis_adj'] = distance_matrix
    return adata

def build_microenv_graph(adata, initial_type = 'celltype', microenv_neighbor = 18, dist_mode = 'lnv1', alpha = 1, diag = 1, calculate_compo_data = True,verbose = False):
    #check adata, if having necessary part
    if 'spatial' not in adata.obsm:
        raise ValueError("No spatial coordinates exist. Please check adata.obsm.")
    if initial_type not in adata.obs:
        raise ValueError("No initial type exist. Please check adata.obs.")
    
    spatial_coords = adata.obsm['spatial']

    #create microenvironment matrix
    composition_matrix, clust_num = calculate_clust_array(adata = adata, initial_type= initial_type)

    #create expression KNN graph, in distance mode
    nbr = NearestNeighbors(algorithm='ball_tree').fit(spatial_coords)
    distance_matrix = nbr.kneighbors_graph(n_neighbors = microenv_neighbor ,mode="distance")
    if verbose:
        print("original distance is: ", distance_matrix.data)
    distance_matrix = calculate_distance_kernel(distance_matrix= distance_matrix, nbr= nbr, distance_mode = dist_mode)
    #default to_array = True, distance_matrix is an array object

    #add self loop
    mat_st = np.eye(adata.n_obs)# same scale diag matrix
    distance_matrix = alpha * distance_matrix + diag * mat_st #for array data only
    
    if calculate_compo_data:
        #add compo and compo_adj
        compo_data = distance_matrix @ composition_matrix
        adata.obsm['compo'] = compo_data
 

    #save expression graph in adata.obsm
    adata.obsm['compo_adj'] = distance_matrix
    adata.obsm['compo_matrix'] = composition_matrix

    return adata, clust_num
    


def calculate_clust_array(adata, initial_type, verbose = False):
    #check adata, if having necessary part
    if initial_type not in adata.obs:
        raise ValueError("No initial type exist. Please check adata.obs.")
    clusts = adata.obs[initial_type].value_counts()#collect cell type
    clust_num = len(clusts)
    composition_matrix= np.zeros((adata.n_obs, clust_num))
    for cell in range(0, adata.n_obs):#transfer vector to array
        for type in range(0, clust_num):
            if adata.obs['celltype'][cell] == str(type):
                composition_matrix[cell,type] = 1
    
    return composition_matrix, clust_num



def calculate_distance_kernel(distance_matrix,
                              nbr,
                              if_mean = True, 
                              distance_mode = 'lnv1', 
                              if_scale = True,
                              if_to_array = True, 
                              verbose = False):
    #if using median nearest to normalize distance
    if if_mean == True:
        distances, indices = nbr.kneighbors(n_neighbors=1)
        mean_cell_distance = np.mean(distances)
        #find the average distance of the nearest neighbor
        distance_matrix.data = distance_matrix.data/mean_cell_distance
        if verbose:
            print("after nearest distance is: ", distance_matrix.data)
    
    #choose distance methods
    if distance_mode == "lnv1":
        distance_matrix.data = 1 / (1 + distance_matrix.data)#lnv1,use the 1/(1+x) cal the distance
    if distance_mode == "lnv2":
        distance_matrix.data = 1 / (1 + distance_matrix.data ** 2)#lnv2,use the 1/(1+x**2) cal the distance
    if distance_mode == "exp":
        distance_matrix.data = np.exp(-distance_matrix.data)#exp,use the exp*(-x) cal the distance
    if distance_mode == "gaussian":
        distance_matrix.data = distance_matrix.data ** 2 
        distance_matrix.data = np.exp(-distance_matrix.data)#exp,use the exp*(-x) cal the distance
    if verbose: 
        print("after calculated distance is: ", distance_matrix.data)
    
    #if using softmax strategy to normalize distance
    if if_scale == True:
        distance_matrix_sum = distance_matrix.sum(axis=1)
        distance_matrix = distance_matrix.multiply(1 / distance_matrix_sum)
        if verbose:
            print("after scaled distance is: ", distance_matrix.data)
    
    if if_to_array == True:
        distance_matrix = distance_matrix.toarray()
    
    return distance_matrix

def cluster_method(adata, fig_save_path, data_place = 'embeddings', mode = 'leiden', res_lei = 0.5, n_cluster = 7):
    #adata = adata.copy()
    embeddings = adata.obsm[data_place]
    adata_new = sc.AnnData(embeddings)
    adata_new.obs.index = adata.obs.index
    adata_new.obsm['spatial']=adata.obsm['spatial']
    adata_new.obs=adata.obs
    print(adata_new)
    obsm_data= data_place

    if mode == 'leiden':
        sc.pp.neighbors(adata_new, use_rep='X')
        sc.tl.leiden(adata_new,resolution=res_lei,key_added="res_"+str(res_lei))
        #sq.pl.spatial_scatter(adata_new,spatial_key="spatial",shape=None,color="res_"+str(res_lei),save="/data2/zkf/MENDER/first_try/compo/"+"res_"+str(res_lei)+".jpg")
        NMI = format(normalized_mutual_info_score(adata_new.obs['ground_truth'],adata_new.obs["res_"+str(res_lei)]), '.3f')
        ARI = format(adjusted_rand_score(adata_new.obs['ground_truth'],adata_new.obs["res_"+str(res_lei)]), '.3f')
        print("res:",res_lei,", NMI=",NMI, ", ARI=",ARI)
        sq.pl.spatial_scatter(adata_new,spatial_key="spatial",shape=None,color="res_"+str(res_lei),title="DWGCN "+mode+" NMI="+ str(NMI)+" ARI="+ str(ARI),
        save=fig_save_path+"res_"+str(res_lei)+".jpg", dpi = 100)

    if mode =='mclust':
        np.random.seed(12)
        import rpy2.robjects as robjects
        robjects.r.library("mclust")
        import rpy2.robjects.numpy2ri
        rpy2.robjects.numpy2ri.activate()
        r_random_seed = robjects.r['set.seed']
        r_random_seed(12)
        rmclust = robjects.r['Mclust']
        res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata_new.X), n_cluster, 'EEE')
        #print(res)
        mclust_res = np.array(res[-2])
        adata_new.obs[f"{obsm_data}_mclust"] = mclust_res
        adata_new.obs[f"{obsm_data}_mclust"] = adata_new.obs[f"{obsm_data}_mclust"].astype('int')
        adata_new.obs[f"{obsm_data}_mclust"] = adata_new.obs[f"{obsm_data}_mclust"].astype('category')
        NMI = format(normalized_mutual_info_score(adata_new.obs['ground_truth'],adata_new.obs[f"{obsm_data}_mclust"]), '.3f')
        ARI = format(adjusted_rand_score(adata_new.obs['ground_truth'],adata_new.obs[f"{obsm_data}_mclust"]), '.3f')
        print("res:",res,", NMI=",NMI, ", ARI=",ARI)
        sq.pl.spatial_scatter(adata_new,spatial_key="spatial",shape=None,color=f"{obsm_data}_mclust",title="DWGCN "+mode+" NMI="+ str(NMI)+" ARI="+ str(ARI),
        save=fig_save_path+"mclust" +str(n_cluster)+".jpg", dpi = 100)





    
