import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import RelGraphConv

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os

script_dir = os.path.dirname(os.path.realpath(__file__))

"""
https://docs.dgl.ai/tutorials/models/1_gnn/4_rgcn.html#sphx-glr-tutorials-models-1-gnn-4-rgcn-py
"""


def model_from_hparams(hparams, verbose=True):
    """
    Just interfacing to create a model directly from an hparam file
    :param hparams:
    :return:
    """
    num_rels = hparams.get('argparse', 'num_edge_types')
    model = Model(dims=hparams.get('argparse', 'embedding_dims'),
                  self_loop=hparams.get('argparse', 'self_loop'),
                  conv_output=hparams.get('argparse', 'conv_output'),
                  num_rels=num_rels,
                  num_bases=-1,
                  similarity=hparams.get('argparse', 'similarity'),
                  verbose=verbose)
    return model


class Embedder(nn.Module):

    def __init__(self,
                 dims,
                 num_rels=19,
                 num_bases=-1,
                 conv_output=False,
                 self_loop=False,
                 verbose=True):
        super(Embedder, self).__init__()
        self.dims = dims
        self.conv_output = conv_output
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.self_loop = self_loop
        self.verbose = verbose

        self.layers = self.build_model()
        if self.verbose:
            print(self.layers)
            print("Num rels: ", self.num_rels)

    def build_model(self):
        layers = nn.ModuleList()

        short = self.dims[:-1]
        last_hidden, last = self.dims[-2:]
        if self.verbose:
            print("short, ", short)
            print("last_hidden, last ", last_hidden, last)

        # input feature is just node degree
        i2h = self.build_hidden_layer(1, self.dims[0])
        layers.append(i2h)

        for dim_in, dim_out in zip(short, short[1:]):
            # print('in',dim_in, dim_out)
            h2h = self.build_hidden_layer(dim_in, dim_out)
            layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer(last_hidden, last)
        # print('last',last_hidden,last)
        layers.append(h2o)
        return layers

    @property
    def current_device(self):
        """
        :return: current device this model is on
        """
        return next(self.parameters()).device

    def build_hidden_layer(self, in_dim, out_dim):
        return RelGraphConv(in_dim, out_dim, self.num_rels,
                            num_bases=self.num_bases,
                            activation=F.relu,
                            self_loop=self.self_loop)

    # No activation for the last layer
    def build_output_layer(self, in_dim, out_dim, conv=False):
        if self.conv_output:
            return RelGraphConv(in_dim, out_dim,
                                self.num_rels,
                                num_bases=self.num_bases,
                                self_loop=self.self_loop,
                                activation=None)
        else:
            return nn.Linear(in_dim, out_dim)

    def forward(self, g):
        # h = g.in_degrees().view(-1, 1).float().to(self.current_device)
        h = torch.ones(len(g.nodes())).view(-1, 1).to(self.current_device)
        for i, layer in enumerate(self.layers):
            # layer(g)
            if not self.conv_output and (i == len(self.layers) - 1):
                h = layer(h)
            else:
                h = layer(g, h, g.edata['one_hot'])
        g.ndata['h'] = h
        return g.ndata['h']


###############################################################################
# Define full R-GCN model
# ~~~~~~~~~~~~~~~~~~~~~~~

class Model(nn.Module):
    def __init__(self,
                 dims,
                 num_rels,
                 num_bases=-1,
                 conv_output=True,
                 self_loop=False,
                 hard_embed=False,
                 similarity=True,
                 normalize=False,
                 weighted=False,
                 verbose=True):
        """

        :param dims: the embeddings dimensions, a list of type [128,128,32]
        :param attributor_dims: the number of motifs to look for
        :param num_rels: the number of possible edge types
        :param num_bases: technical rGCN option

        :param rec: the constant in front of reconstruction loss
        :param mot: the constant in front of motif detection loss
        :param orth: the constant in front of dictionary orthogonality loss
        :param scaled: if we want to scale the loss by attribution norm
        :param similarity: if we want to use cosine similarities instead of distances everywhere

        :param attribute: Whether we want the network to use the attribution module
        :param convolute: If we want to use a rgcn also for the attributions

        """
        super(Model, self).__init__()
        self.verbose = verbose
        self.dims = dims
        self.dimension_embedding = dims[-1]

        self.num_rels = num_rels
        self.num_bases = num_bases

        self.similarity = similarity
        self.normalize = normalize
        self.weighted = weighted
        self.self_loop = self_loop

        # create rgcn layers for the embedder
        self.embedder = Embedder(dims=dims,
                                 num_rels=num_rels,
                                 num_bases=num_bases,
                                 self_loop=self_loop,
                                 conv_output=conv_output,
                                 verbose=verbose)

    def forward(self, g):
        # If hard embed, the embeddings are directy g.ndata['h'], otherwise we compute them and put them here
        self.embedder(g)

        # If using similarity as a supervision, we should normalize the embeddings, as their norm got unconstrained
        if self.similarity and self.normalize:
            g.ndata['h'] = F.normalize(g.ndata['h'], p=2, dim=1)
        return g.ndata['h']

    @property
    def current_device(self):
        """
        :return: current device this model is on
        """
        return next(self.parameters()).device

    # Below are loss computation function related to this model
    @staticmethod
    def matrix_cosine(a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    @staticmethod
    def matrix_dist(a, plus_one=False):
        """
        Pairwise dist of a set of a vector of size b
        returns a matrix of size (a,a)
        :param a : a torch Tensor of size a,b
        :param plus_one: if we want to get positive values
        """
        if plus_one:
            return torch.norm(a[:, None] - a, dim=2, p=2) + 1
        return torch.norm(a[:, None] - a, dim=2, p=2)

    @staticmethod
    def weighted_MSE(output, target, weight):
        if weight is None:
            return torch.nn.MSELoss()(output, target)
        return torch.mean(weight * (output - target) ** 2)

    def rec_loss(self, embeddings, target_K, graph=None):
        """
        :param embeddings: The node embeddings
        :param target_K: The similarity matrix
        :return:
        """
        if self.similarity:
            if self.normalize:
                K_predict = self.matrix_cosine(embeddings, embeddings)
            else:
                K_predict = torch.mm(embeddings, embeddings.t())

        else:
            K_predict = self.matrix_dist(embeddings)
            target_K = torch.ones(target_K.shape, device=target_K.device) - target_K

        if self.weighted:
            assert graph is not None
            import networkx as nx
            nx_graph = graph.to_networkx(edge_attrs=['one_hot'])
            nx_graph = nx.to_undirected(nx_graph)
            ordered = sorted(nx_graph.nodes())
            adj_matrix_full = nx.to_scipy_sparse_matrix(nx_graph, nodelist=ordered)

            # copy the matrix with only the non canonical
            extracted_edges = [(u, v) for u, v, e in nx_graph.edges.data('one_hot', default='0')
                               if e not in [0, 6]]
            extracted_graph = nx.Graph()
            extracted_graph.add_nodes_from(ordered)
            extracted_graph.add_edges_from(extracted_edges)
            extracted_graph = nx.to_undirected(extracted_graph)
            adj_matrix_small = nx.to_scipy_sparse_matrix(extracted_graph, nodelist=ordered)

            # This is a matrix with non zero entries for non canonical relationships
            # One must then expand it based on the number of hops
            adj_matrix_full = np.array(adj_matrix_full.todense())
            adj_matrix_small = np.array(adj_matrix_small.todense())

            expanded_connectivity = [np.eye(len(adj_matrix_full))]
            for _ in self.dims[:-1]:
                expanded_connectivity.append(expanded_connectivity[-1] @ adj_matrix_full)
            expanded_connectivity = np.sum(expanded_connectivity, axis=0)

            # What we are after is a matrix for which you start with a walk of len < max_len
            # that starts with node i and that ends with a non canonical with j
            # ie : all neighborhoods that include a non canonical.
            # multiplying on the left yields walks that start with a non canonical on the rows
            # expanded_connectivity_left = np.array(adj_matrix_small @ expanded_connectivity)
            expanded_connectivity_right = np.array(expanded_connectivity @ adj_matrix_small)
            enhanced = np.sum(expanded_connectivity_right, axis=0)
            enhanced = np.clip(enhanced, a_min=0, a_max=1)
            fraction = np.sum(enhanced) / len(enhanced)
            enhanced = ((1 / (fraction + 0.005)) * enhanced) + 1
            weight = np.outer(enhanced, enhanced)
            weight /= np.mean(weight)
            weight = torch.from_numpy(weight)
            return self.weighted_MSE(K_predict, target_K, weight)

        reconstruction_loss = torch.nn.MSELoss()(K_predict, target_K)
        return reconstruction_loss
