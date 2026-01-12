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
import random
import os

#graph part need
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors

#model nedd
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
import torch
#import pysodb

from tqdm import tqdm
from torch_geometric.data import Data

from model import DWGCN, spatial_reg_loss, zinb_loss
from utlis import build_expression_graph, build_microenv_graph, preprocessing, cluster_method
from argparse import *

#train
def train(adatast,
        clust_num,
        embedding_dim=32,
        reg_theta=1e-2,
        spatial_reg_parameter = 1,
        device='cuda', lamda=0.5,
        lr_st=1e-3, weight_decay=1e-4,
        dropout=0, heads=1,
        layer_encoder=[256,128],
        layer_decoder=[128,256],
        max_epochs_st=1500,
        verbose=False, save_history=False
        ):
    X_st = adatast.X
    if isinstance(X_st, np.ndarray):
        pass
    else:
        X_st = X_st.toarray()
 
    dis = adatast.obsm['dis_adj']
    dis = torch.as_tensor(dis).to(device)
    compo_adj = adatast.obsm['compo_adj']
    compo_adj = torch.as_tensor(compo_adj).to(device)
    compo = adatast.obsm['compo']
    compo = torch.as_tensor(compo).to(torch.float32)
    dis_data = Data(torch.FloatTensor(X_st), dis.nonzero().t().contiguous().long()).to(device)
    compo_data = Data(torch.FloatTensor(compo), compo_adj.nonzero().t().contiguous().long()).to(device)
    X_st = torch.as_tensor(X_st).to(device)
    dis = torch.as_tensor(dis).to(device)
    compo = torch.as_tensor(compo).to(device)
    compo_adj = torch.as_tensor(compo_adj).to(device)
    spatial_coords = adatast.obsm['spatial']

    model = DWGCN(
        gene_dim=adata.n_vars,
        clust_dim = clust_num, 
        embedding_dim=embedding_dim,
        layer_encoder=layer_encoder,
        layer_decoder=layer_decoder,
        dropout=dropout,
        heads=heads,
    )
    model = model.to(device)    
    model.train()

    optimizer2 = torch.optim.Adam(model.parameters(), lr=lr_st, weight_decay=weight_decay)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, mode='min', factor=0.9, patience=50, min_lr=1e-5
    )
    training_history_df = pd.DataFrame(columns=['epoch', 'loss_ZINB', 'loss_total'])
    if verbose:
        print("Training...")
    pbar = tqdm(range(max_epochs_st), total=max_epochs_st, desc='Training', unit='epoch')
        
    for epoch in pbar:
        optimizer2.zero_grad()
        embeddings_st, pi_st, mu_st, theta_st, lamda = model(dis_data.x, dis_data.edge_index, compo_data.x, compo_data.edge_index)

        loss_ZINB = zinb_loss(X_st, mu_st, pi_st, theta_st, reg_theta)
        loss_spatial_reg = spatial_reg_loss(embeddings_st, spatial_coords, device = device)
        #loss_reg = regularization_loss(embeddings_st, spatial_coords)
        
        loss_total = loss_ZINB + spatial_reg_parameter * loss_spatial_reg
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss_total.backward()
        optimizer2.step()
        pbar.set_postfix(epoch=f"{epoch}", loss=f"{loss_total.detach():.3f}",
                         lr=f"{optimizer2.param_groups[0]['lr']:.4f}")

        scheduler2.step(loss_total)
        if optimizer2.param_groups[0]['lr'] < 1e-5:
            break

        if save_history:
            training_history_df = pd.concat([training_history_df,
                                             pd.DataFrame({
                                                 'epoch': [epoch],
                                                 'loss_ZINB': [float(loss_ZINB.detach().cpu().numpy())],
                                                 'loss_total': [float(loss_total.detach().cpu().numpy())],
                                             })],
                                            ignore_index=True)

    if save_history:
        adatast.uns['training_history_df_st'] = training_history_df

    model.eval()
    embeddings_st, pi_st, mu_st, theta_st, lamda = model(dis_data.x, dis_data.edge_index, compo_data.x, compo_data.edge_index)

    adatast.model = model
    adatast.obsm['embeddings'] = embeddings_st.detach().cpu().numpy()
    adatast.obsm['pi'] = pi_st.detach().cpu().numpy()
    adatast.obsm['mu'] = mu_st.detach().cpu().numpy()
    adatast.obsm['theta'] = theta_st.detach().cpu().numpy()
    adatast.uns['lamda'] = lamda.detach().cpu().numpy()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="batch parameters")
    
    parser.add_argument('--source_data', type=str, default="example.h5ad", help='Path to load anndata')
    parser.add_argument('--fig_save_path', type=str, default="./test/", help='Path to save figures and results')
    parser.add_argument('--epochs', type=int, default=200, help='training epochs')
    parser.add_argument('--spatial_reg_parameter', type=int, default=10, help='spatial reg hyparameter')
    parser.add_argument('--layer_encoder', type=int, nargs=2, default=[256,128], help='GCNEncoder neuron number')
    parser.add_argument('--layer_decoder', type=int, nargs=2, default=[128,256], help='GCNDecoder neuron number')
    parser.add_argument('--estimate_clust_num', type=int, default=7, help='estimating cluster number for mclust')
    parser.add_argument('--cuda', type=int, default=7, help='cuda kernel')
    parser.add_argument('--exp_neighbor', type=int, default=6, help='gene_expression_neighbor')
    parser.add_argument('--env_neighbor', type=int, default=18, help='microenvironment_neighbor')
    parser.add_argument('--ini_res', type=float, default=1, help='initial resolution for clust')
    parser.add_argument('--dist_mode', type=str, default="lnv1", help='distance kernel mode')
    parser.add_argument('--res_set', type=float, default=0.5, help='chosen leiden resolution') 
 
    args = parser.parse_args()
    source_data = args.source_data
    fig_save_path = args.fig_save_path
    epochs = args.epochs
    estimate_clust_num = args.estimate_clust_num
    spatial_reg_parameter = args.spatial_reg_parameter
    layer_encoder = args.layer_encoder
    layer_decoder = args.layer_decoder
    cuda = args.cuda
    exp_neighbor = args.exp_neighbor
    env_neighbor = args.env_neighbor
    ini_res = args.ini_res
    dist_mode = args.dist_mode
    res_set = args.res_set
    set_seed(1234)


    #load data
    #adata = ad.read("./STARmap_20180505_BY3_1k_20250310202358.h5ad")
    #import pysodb
    #sodb = pysodb.SODB()
    #adata_raw = sodb.load_experiment('Allen2022Molecular_aging',source_data)
    #adata = adata_raw.copy()

    adata = ad.read(source_data)
    print(adata)
    '''
    #10x data NaN process
    #adata.obs['Region'] == 'NA'
    print(adata)
    k = 0
    list=[]
    for i in range(adata.n_obs):
        if adata.obs['Region'][i] != 'Layer1' and adata.obs['Region'][i] != 'Layer2' and adata.obs['Region'][i] != 'Layer3' and adata.obs['Region'][i] != 'Layer4' and    
        adata.obs['Region'][i] != 'Layer5' and adata.obs['Region'][i] != 'Layer6' and adata.obs['Region'][i] != 'WM':  
            k = k + 1
            list.append(i)
    print(k)
    print(list)
    adata.obs['ifNA'] = True
    for i in list:
        adata.obs['ifNA'][i] = False
    adata = adata[adata.obs['ifNA'],:]
    print(adata)
    '''
    adata = preprocessing(adata)
    #initial_res = ini_res
    #initial_type_store = "celltype"
    #adata.obs['ground_truth'] = adata.obs['tissue']
    #sc.tl.leiden(adata, resolution=initial_res,key_added = initial_type_store)
    #print(adata)
    preprocessed_adata = adata.copy()
    adata = build_expression_graph(adata, expression_neighbor = exp_neighbor, dist_mode = dist_mode)
    adata, clust_num = build_microenv_graph(adata, microenv_neighbor = env_neighbor, dist_mode = dist_mode)
    device = 'cuda:'+str(cuda)
    train(adata, clust_num, spatial_reg_parameter = spatial_reg_parameter, verbose=True, device=device, max_epochs_st=epochs)
    adata_after_train = adata.copy()
    adata_after_train.model = adata.model
    cluster_method(adata, fig_save_path, mode = 'leiden', res_lei = res_set)
    cluster_method(adata, fig_save_path, mode = 'mclust', n_cluster = estimate_clust_num)
    torch.save(adata_after_train.model, fig_save_path+'model.pth')
    adata_after_train.write(fig_save_path+'adata_reg_'+str(spatial_reg_parameter)+'_epoch'+str(epochs)+'.h5ad')




