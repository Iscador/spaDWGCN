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

#model nedd
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
import torch
#import pysodb

from tqdm import tqdm
from torch_geometric.data import Data

class LinearDecoder(nn.Module):
    def __init__(self, layer_dims=None):
        super().__init__()
        if layer_dims is None:
            layer_dims = [32, 256, 512, 5000]
        assert len(layer_dims) >= 2, "#layers must >=2, including input layer and output layer."
        if isinstance(layer_dims, int):
            layer_dims = [layer_dims]
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.elu(layer(x))
        embeddings = self.layers[-1](x)
        return embeddings
#GNN model for train
class GCNEncoder(nn.Module):
    def __init__(self, layer_dims=None, dropout=0.0, heads=1):
        super().__init__()
        if layer_dims is None:
            layer_dims = [2000, 32]
        assert len(layer_dims) >= 2, "#layers must >=2, including input layer and output layer."
        self.layers = nn.ModuleList()
        self.layer_dims = layer_dims
        self.heads = heads
        self.dropout = dropout
        #self.layers.append(GATv2Conv(layer_dims[0], layer_dims[1], heads=heads, concat=False, dropout=dropout,add_self_loops=True, bias=False))
        self.layers.append(GCNConv(layer_dims[0], layer_dims[1], add_self_loops=False, bias=False))
        self.layers.append(GCNConv(layer_dims[1], layer_dims[2], add_self_loops=False, bias=False))
        self.layers.append(GCNConv(layer_dims[2], layer_dims[3], add_self_loops=False, bias=False))
        #self.layers.append(GATv2Conv(layer_dims[2], layer_dims[3], heads=heads, concat=False, dropout=dropout,add_self_loops=True, bias=False))
    def forward(self, x, edge_index):
        # Process input through each GAT layer using elu activation function
        for layer in self.layers[:-1]:
            x = F.elu(layer(x, edge_index))
        x = self.layers[-1](x, edge_index)
        return x

class GCNDecoder(nn.Module):
    def __init__(self, layer_dims=None, dropout=0.0, heads=1):
        super().__init__()
        if layer_dims is None:
            layer_dims = [2000, 32]
        assert len(layer_dims) >= 2, "#layers must >=2, including input layer and output layer."
        self.layers = nn.ModuleList()
        self.layer_dims = layer_dims
        self.heads = heads
        self.dropout = dropout
        #self.layers.append(GATv2Conv(layer_dims[0], layer_dims[1], heads=heads, concat=False, dropout=dropout,add_self_loops=True, bias=False))
        self.layers.append(GCNConv(layer_dims[0], layer_dims[1], add_self_loops=False, bias=False))
        self.layers.append(GCNConv(layer_dims[1], layer_dims[2], add_self_loops=False, bias=False))
        self.layers.append(GCNConv(layer_dims[2], layer_dims[3], add_self_loops=False, bias=False))
        #self.layers.append(GATv2Conv(layer_dims[2], layer_dims[3], heads=heads, concat=False, dropout=dropout,add_self_loops=True, bias=False))
    def forward(self, x, edge_index):
        # Process input through each GAT layer using elu activation function
        for layer in self.layers[:-1]:
            x = F.elu(layer(x, edge_index))
        x = self.layers[-1](x, edge_index)
        return x

class DWGCN(nn.Module):
    def __init__(self,
                 gene_dim,
                 clust_dim,
                 embedding_dim=32,
                 lamda=0.5,
                 dropout=0.0,
                 layer_encoder=[256,128],
                 layer_decoder=[128,256],
                 clust_layer = [128,64],
                 heads=1):
        super().__init__()
        self.gene_dim = gene_dim
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.dropout = dropout
        self.layer_encoder = layer_encoder
        self.layer_decoder = layer_decoder
        self.clust_dim = clust_dim
        self.clust_layer = clust_layer
        #self.lamda = lamda
        self.lamda = nn.Parameter(torch.tensor(0.5,dtype=torch.float32),requires_grad=True)
        self.ExpressionEncoder = GCNEncoder(
            layer_dims=[self.gene_dim] + self.layer_encoder + [self.embedding_dim],
            dropout=self.dropout, heads=self.heads
        )

        self.MicroenvEncoder = GCNEncoder(
            layer_dims=[self.clust_dim] + self.clust_layer + [self.embedding_dim],
            dropout=self.dropout, heads=self.heads
        )
        
        
        #self.Z2pi = LinearDecoder(layer_dims=[self.embedding_dim] + self.layer_decoder + [self.gene_dim])
        #self.Z2mu = LinearDecoder(layer_dims=[self.embedding_dim] + self.layer_decoder + [self.gene_dim])
        #self.Z2theta = LinearDecoder(layer_dims=[self.embedding_dim] + self.layer_decoder + [self.gene_dim])
        
        self.Z2pi = GCNDecoder(
            layer_dims=[self.embedding_dim] + self.layer_decoder + [self.gene_dim],
            dropout=self.dropout, heads=self.heads
        )
        self.Z2mu = GCNDecoder(
            layer_dims=[self.embedding_dim] + self.layer_decoder + [self.gene_dim],
            dropout=self.dropout, heads=self.heads
        )
        self.Z2theta = GCNDecoder(
            layer_dims=[self.embedding_dim] + self.layer_decoder + [self.gene_dim],
            dropout=self.dropout, heads=self.heads
        )
        
    def forward(self, x, edge_index, compo, compo_index):
        embeddings = self.ExpressionEncoder(x, edge_index)
        compo = self.MicroenvEncoder(compo, compo_index)
        #self.lamda = torch.clamp(self.lamda, min=0, max=1)
        embeddings = torch.tanh(self.lamda) * embeddings + (1 - torch.tanh(self.lamda)) * compo
        
        pi = self.Z2pi(embeddings, edge_index)
        #pi = self.Z2pi(embeddings)
        pi = F.sigmoid(pi)
        mu = self.Z2mu(embeddings, edge_index)
        #mu = self.Z2mu(embeddings)
        mu = F.softplus(mu)
        mu = torch.clamp(mu, min=1e-8, max=1e6)
        theta = self.Z2theta(embeddings, edge_index)
        #theta = self.Z2theta(embeddings)
        theta = torch.exp(theta)
        theta = torch.clamp(theta, min=1e-8, max=1e6)

        return embeddings, pi, mu, theta, self.lamda

#loss function 
def zinb_loss(true, pred, pi, theta, eps=1e-6, reg_theta=1e-4):

    """
    Compute Zero-Inflated Negative Binomial loss.
    
    Parameters:
    - true: The true values.
    - pred: The predicted mean values.
    - pi: The predicted zero-inflation probabilities.
    - theta: The dispersion parameter of the negative binomial distribution.
    - eps: A small value used for numerical stability.
    
    Returns:
    - loss: The computed loss value.
    """

    # Negative Binomial
    pred = torch.clamp(pred, min=eps, max=1e6)
    theta = torch.clamp(theta, min=eps, max=1e6)
    pi = torch.clamp(pi, min=eps, max=1-eps)
    
    t1 = torch.lgamma(theta + eps) + torch.lgamma(true + 1.0) - torch.lgamma(theta + true + eps)
    t2 = (theta + true) * torch.log(1.0 + (pred / (theta + eps))+eps) + (true * (torch.log(theta + eps) - torch.log(pred + eps)))

    nb_loss = t1 + t2

    # Zero-Inflated
    zero_inflation = -torch.log(torch.where(true > 0, 1.0 - pi+eps, pi+eps))

    # add two part
    loss = zero_inflation + nb_loss * (1 - pi) + reg_theta * theta

    return loss.mean()

def regularization_loss(emb, spatial_coords, spatial_neighbor, device='cuda:0'):

    """
    Compute the regularization loss based on the embedding matrix and spatial coordinates.

    Parameters:
    - emb: The embedding matrix. It represents the embeddings of nodes.
    - spatial_coords: The spatial coordinates of nodes. Used to construct the spatial adjacency graph.
    - spatial_neighbor: The number of neighbor used in K nearest neighbor graph for regularization loss.

    Returns:
    - pair_loss: The computed pair - wise regularization loss value.
    """
    spatial_adjancy = kneighbors_graph(spatial_coords, spatial_neighbor, mode='connectivity', include_self=False)
    spatial_adj = spatial_adjancy.toarray()
    graph_nei = torch.as_tensor(spatial_adj).to(device)
    graph_one = torch.ones(spatial_coords.shape[0],spatial_coords.shape[0]).to(device)
    graph_neg = graph_one - graph_nei
    
    mat = torch.matmul(emb, emb.T).to(device)
    norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
    mat = torch.div(mat, torch.matmul(norm, norm.T))
    if torch.any(torch.isnan(mat)):
        mat = _nan2zero(mat)
    mat = mat - torch.diag_embed(torch.diag(mat))    
    mat = torch.sigmoid(mat)  # .cpu()
    # mat = pd.DataFrame(mat.cpu().detach().numpy()).values

    # graph_neg = torch.ones(graph_nei.shape) - graph_nei

    neigh_loss = torch.mul(graph_nei, torch.log(mat)).mean()
    neg_loss = torch.mul(graph_neg, torch.log(1 - mat)).mean()
    pair_loss = -(neigh_loss + neg_loss) / 2
    return pair_loss

def spatial_reg_loss(z, spatial_coords, device= 'cuda'):

    """
    Compute the spatial regularization loss. (inspired from SpaceFlow algorithm)

    Parameters:
    - z: The embedding matrix representing the data in a latent space. Each row corresponds to an instance.
    - spatial_coords: The spatial coordinates of the data points. It is used to calculate the spatial distances.

    Returns:
    - penalty: The computed spatial regularization loss value.
    """

    coords = torch.tensor(spatial_coords).float().to(device)
    '''
    if regularization_acceleration or adata_preprocessed.shape[0] > 5000:
        cell_random_subset_1, cell_random_subset_2 = torch.randint(0, z.shape[0], (edge_subset_sz,)).to(device), torch.randint(0, z.shape[0], (edge_subset_sz,)).to(device)
        z1, z2 = torch.index_select(z, 0, cell_random_subset_1), torch.index_select(z, 0, cell_random_subset_2)
        c1, c2 = torch.index_select(coords, 0, cell_random_subset_1), torch.index_select(coords, 0, cell_random_subset_1)
        pdist = torch.nn.PairwiseDistance(p=2)

        z_dists = pdist(z1, z2)
        z_dists = z_dists / torch.max(z_dists)

        sp_dists = pdist(c1, c2)
        sp_dists = sp_dists / torch.max(sp_dists)
        n_items = z_dists.size(dim=0)
    else:
    '''
    z_dists = torch.cdist(z, z, p=2)
    z_dists = torch.div(z_dists, torch.max(z_dists)).to(device)
    sp_dists = torch.cdist(coords, coords, p=2)
    sp_dists = torch.div(sp_dists, torch.max(sp_dists)).to(device)
    n_items = z.size(dim=0) * z.size(dim=0)

    penalty = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items).to(device)
    return penalty

