"""
Random util functions.
"""

import os, sys
import numpy as np
import configparser
import pickle
import gc
import datetime
import itertools
from collections import OrderedDict, Counter

from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    sys.path.append("..")
from tools.graph_utils import nx_to_dgl


class OrderedCounter(Counter, OrderedDict):
    pass


def make_histogram(obs, classes, class_transformation=None, transform_classes=None):
    """
        Create histogram vector from list of observations for given classes.
    """
    c = OrderedCounter({e: 0 for e in classes})
    c.update([o for o in obs if o in classes])

    # emphasize nc interactions
    if not class_transformation is None:
        for k, v in c.items():
            if k in transform_classes:
                c[k] = class_transformation(c[k])
    return c.values()


def histogram_embed(g, rings, edge_types, max_deg=5, hops=1):
    """
        Take graph and produce embedding of histograms of edge types and degrees for each hop.
        emb = [edge_hist_hops, deg_dist_hops]

        :param g: graph
        :param edge_types: ordered list of edge types to count
        :param max_deg: number of degrees to count
        :param hops: number of hops from each node to use for embedding

        :returns: matrix node embeddings for whole graph (num_nodes x embedding_dim)
    """
    edge_emb_len = len(edge_types) * hops
    deg_emb_len = max_deg * hops + 1
    emb_size = edge_emb_len + deg_emb_len

    faces = ['W', 'S', 'H']
    orientations = ['C', 'T']
    nc_bps = {orient + e1 + e2 for e1, e2 in itertools.product(faces, faces) for orient in orientations}
    nc_bps.remove('CWW')
    transf = lambda x: x ** 2 + 5

    emb_matrix = np.zeros((len(g.nodes), emb_size))
    i = 0
    for node, (node_rings, edge_rings) in sorted(rings.items(), key=lambda x: x[0]):
        e_embedding = [] * edge_emb_len
        d_embedding = [g.degree(node)] + [] * deg_emb_len
        for h in range(1, hops + 1):
            e_hist = make_histogram(edge_rings[h], edge_types)
            degrees = [g.degree(n) for n in node_rings[h]]
            deg_hist = make_histogram(degrees, list(range(max_deg)))
            e_embedding.extend(e_hist)
            d_embedding.extend(deg_hist)
        emb_matrix[i] = np.array(e_embedding + d_embedding)
        i += 1

    return emb_matrix

    # e_types = list(get_edge_map(graph_dir).keys())
    # pass


def get_hard_embeddings(graph_dir, e_types=None, max_graphs=-1, **kwargs):
    """
        Get hard-coded node embeddings.

        :params
        :max_graphs max number of graphs to get embeddings for
    """

    import networkx as nx
    from no_learning.hand_embed import histogram_embed
    from tools.graph_utils import get_edge_map

    Z = []
    nx_graphs = []

    # maps full nodeset index to graph and node index inside graph

    if e_types is None:
        e_types = list(get_edge_map(graph_dir).keys())
    if max_graphs > 0:
        graphlist = sorted(os.listdir(graph_dir))[:max_graphs]
    else:
        graphlist = sorted(os.listdir(graph_dir))
    ind = 0
    print(f">>> computing embeddings for {len(graphlist)} graphs")
    for i, g in tqdm(enumerate(graphlist), total=len(graphlist)):
        try:
            G, trees, rings = nx.read_gpickle(os.path.join(graph_dir, g))
        except:
            print(f"failed to load graph {g}")
            continue
        z = histogram_embed(G, rings, e_types, **kwargs)
        for j, emb in enumerate(z):
            Z.append(np.array(emb))
        nx_graphs.append(G)
        pass

    Z = np.array(Z)
    return nx_graphs, Z


def unpickle(f):
    return pickle.load(open(f, 'rb'))


def pdump(o, f):
    pickle.dump(o, open(f, 'wb'))


def makedir(path, permissive=True):
    """
    Try to make a folder, if it already exists, one can choose
    :param name:
    :param permissive: If True will overwrite existing files (good for debugging)
    :return:
    """
    try:
        os.mkdir(path)
    except FileExistsError:
        if not permissive:
            raise ValueError('This name is already taken !')
    return True

if __name__ == '__main__':
    pass
