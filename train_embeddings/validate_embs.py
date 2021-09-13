"""
Compute GED values and compare them to kernel computations as well as embeddings dot product.
"""

import cProfile
import multiprocessing
import os
import sys
import time

import pickle
import itertools
from collections import defaultdict

import numpy as np
import random
from random import shuffle
from tqdm import tqdm
from scipy.stats import pearsonr
import networkx as nx
from multiprocessing import Pool
import torch
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations_with_replacement
import seaborn as sns
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from tools.node_sim import SimFunctionNode, k_block_list
from tools.graph_utils import has_NC_bfs, bfs_expand
from tools.rna_ged_nx import ged


def get_nodelist(graph_dir='../data/annotated/NR_chops_annot'):
    """
    Get nodelist at random: sample 200 chopped graphs and then pick a random graph around a NC using rejection sampling

    Dump a list containing dicts : {'graph': nx graphlet, 'graphlet': graphlet rings, 'edge': edge rings}
    """
    graphlist = os.listdir(graph_dir)
    sampled_graphs = np.random.choice(graphlist, replace=False, size=200)
    list_return = list()
    for graph_name in sampled_graphs:
        graph_path = os.path.join(graph_dir, graph_name)
        whole_graph_dict = pickle.load(open(graph_path, 'rb'))
        graph = whole_graph_dict['graph']
        rings = whole_graph_dict['rings']
        inner_graph_nodelist = list(graph.nodes())
        random.shuffle(inner_graph_nodelist)
        found_nc = False
        for node in inner_graph_nodelist:
            if has_NC_bfs(graph=graph, node_id=node):
                found_nc = True
                break
        if not found_nc:
            continue
        subg = list(bfs_expand(graph, [node], depth=2)) + [node]
        subgraph = graph.subgraph(subg).copy()
        rings_edge = rings['edge'][node]
        rings_graphlet = rings['graphlet'][node]
        list_return.append({"node": node, "graph": subgraph, "graphlet": rings_graphlet, "edge": rings_edge})
    return list_return


def get_one_ged(args):
    graph_1, graph_2 = args
    return ged(graph_1, graph_2, timeout=40)


def get_geds(node_list):
    """
    Annotate a list with a matrix of GEDs
    """
    graphlets_list = [node['graph'] for node in node_list]
    todo = list(combinations_with_replacement(graphlets_list, 2))
    pool = multiprocessing.Pool()
    ged_matrix = list(tqdm(pool.imap(get_one_ged, todo), total=len(todo)))
    return ged_matrix


def build_experiments_list():
    def build_experiments(method, param_dict):
        combinations = itertools.product(*(param_dict[name] for name in param_dict))
        experiments = list()
        for comb in combinations:
            new_dict = {key: value for key, value in zip(param_dict.keys(), comb)}
            new_dict['method'] = method
            experiments.append(new_dict)
        return experiments

    R_1_dict = {'normalization': [None, 'sqrt'], 'idf': [True, False], 'depth': [2, 3, 4], 'decay': [0.3, 0.5, 0.8]}
    R_iso_dict = {'normalization': [None, 'sqrt'], 'idf': [True, False], 'depth': [2, 3, 4], 'decay': [0.3, 0.5, 0.8]}
    hungarian_dict = {'normalization': [None, 'sqrt'], 'idf': [True, False], 'depth': [2, 3, 4]}
    all_methods = [('R_1', R_1_dict),
                   ('R_iso', R_iso_dict),
                   ('hungarian', hungarian_dict)]

    graphlet_dict = {'depth': [2, 3, 4], 'normalization': [None, 'sqrt']}
    R_graphlets_dict = {'depth': [2, 3, 4], 'decay': [0.3, 0.5, 0.8], 'normalization': [None, 'sqrt']}
    all_methods.extend([('R_graphlets', R_graphlets_dict),
                        ('graphlet', graphlet_dict)])

    all_experiments = list()
    for method in all_methods:
        experiments = build_experiments(*method)
        all_experiments.extend(experiments)
    return all_experiments


def kernel_vs_ged(simfunc,
                  node_list,
                  ged_matrix,
                  max_nodes=None,
                  print_tqdm=True,
                  plot=True
                  ):
    """
        Compute correlation between GED and simfunc outputs.
    node_list is a list of nodes in the form return by get_nodelist :
        a dict with attributes node, graph, edge_ring, graphlet_ring
    """

    ged_flat = np.exp(-1 * np.array(ged_matrix))[:max_nodes]

    level = 'graphlet' if simfunc.method in ['graphlet', 'R_graphlets'] else 'edge'
    annots_list = [node[level] if level == 'graphlet' else node[level][1:] for node in node_list]
    todo = list(combinations_with_replacement(annots_list, 2))[:max_nodes]
    ks = list()
    if print_tqdm:
        for i, (ring_1, ring_2) in tqdm(enumerate(todo), total=len(todo)):
            k = simfunc.compare(ring_1, ring_2)
            ks.append(k)
    else:
        for i, (ring_1, ring_2) in enumerate(todo):
            k = simfunc.compare(ring_1, ring_2)
            ks.append(k)

    ks = np.array(ks)
    if plot:
        sns.regplot(ged_flat, ks, scatter_kws={'s': 1})
        plt.title(f"{simfunc.method}, pearson: {pearsonr(ged_flat, ks)}")
        plt.show()
    return pearsonr(ged_flat, ks)


def all_kernels_ged(node_list, ged_matrix):
    all_experiments = build_experiments_list()
    all_simfunc = [SimFunctionNode(**simf, hash_init='NR_chops_annot_hash') for simf in all_experiments]
    pool = multiprocessing.Pool()
    default_kernel_vs_ged = partial(kernel_vs_ged, node_list=node_list, ged_matrix=ged_matrix,
                                    plot=False, print_tqdm=False, max_nodes=None)
    correlation_values = list(tqdm(pool.imap(default_kernel_vs_ged, all_simfunc), total=len(all_simfunc)))
    return correlation_values


def embs_vs_ged(run,
                graph_path="../data/annotated/whole_v3",
                ged_pickle='../data/geds_rooted_depth2_n1000_fix.p',
                max_nodes=10000,
                plot=True):
    """
        Compute correlation between GED and simfunc outputs.
    """
    from itertools import combinations
    import seaborn as sns
    import matplotlib.pyplot as plt
    from tools.learning_utils import inference_on_list

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # geds is a distance matrix based on GED
    # nodes is a list of nodes
    geds, nodes = pickle.load(open(ged_pickle, 'rb'))
    nodes = [node[0] for node in nodes]

    ged = geds[:max_nodes, :max_nodes]
    nodes = nodes[:max_nodes]

    # Get unique graph list to get embeddings for these graphs.
    graph_list = [node_id[0].replace('.nx', '_annot.p') for node_id in nodes]
    graph_list = list(set(graph_list))

    Z, Sigma, g_inds, node_map = inference_on_list(run=run,
                                                   graphs_path=graph_path,
                                                   graph_list=graph_list,
                                                   max_graphs=None,
                                                   get_sim_mat=False,
                                                   attributions=False,
                                                   device=device)

    def embedding_cosine(embeddings, eps=1e-8):
        """
        added eps for numerical stability
        """
        embeddings = torch.from_numpy(embeddings)
        e_n = embeddings.norm(dim=1)[:, None]
        e_norm = embeddings / torch.max(e_n, eps * torch.ones_like(e_n))
        sim_mt = torch.mm(e_norm, e_norm.transpose(0, 1)).numpy()
        return sim_mt

    node_embeddings = np.array([Z[node_map[node_id]] for node_id in nodes])

    # ks = embedding_cosine(node_embeddings)
    ks = node_embeddings @ np.transpose(node_embeddings)
    return ks


def merge_results():
    dirname = 'ged_results'
    result = {}
    ged_pickle = '../data/geds_rooted_depth2_n1000_fix.p'
    geds, nodes = pickle.load(open(ged_pickle, 'rb'))
    GED = geds[np.triu_indices(len(nodes), 1)]

    # sns.distplot(GED)
    # plt.show()
    # sns.kdeplot(GED, cumulative=True)
    # plt.show()
    # return
    # 0.1 is already quite some people !

    GED = np.exp(-np.array(GED) / 5)
    ged_argsort = np.argsort(GED)
    GED = [GED[i] for i in ged_argsort]
    last = GED.index(1)
    # print(last)

    all_experiments = build_experiments_list()
    for res in os.listdir(dirname):
        number = int(res[:-2])
        if all_experiments[number]['depth'] > 2:
            continue
        experiment = str(all_experiments[number])
        ks = pickle.load(open(os.path.join(dirname, res), 'rb'))
        # ks = ks[argsort0]
        ks = [ks[i] for i in ged_argsort]

        crop = 2000
        cropped_ged, cropped_ks = GED[-crop:last], ks[-crop:last]
        score, _ = pearsonr(cropped_ged, cropped_ks)
        print(experiment, score)
        result[experiment] = score
    return result


if __name__ == "__main__":
    pass
    random.seed(0)
    np.random.seed(0)

    # node_list = get_nodelist()
    # pickle.dump(node_list, open('../results/correlations/nodelist.p', 'wb'))
    node_list = pickle.load(open('../results/correlations/nodelist.p', 'rb'))
    # node_list_2 = pickle.load(open('../results/correlations/nodelist_local.p', 'rb'))

    # ged_matrix = get_geds(node_list=node_list)
    # pickle.dump(ged_matrix, open('../results/correlations/ged_matrix.p', 'wb'))
    ged_matrix = pickle.load(open('../results/correlations/ged_matrix.p', 'rb'))

    correlation_values = all_kernels_ged(node_list=node_list, ged_matrix=ged_matrix)
    pickle.dump(correlation_values, open('../results/correlations/corralation_values.p', 'wb'))
    # correlation_values = pickle.load(open('../results/correlations/corralation_values.p', 'rb'))

    # for simf in all_experiments:
    # simfunc = SimFunctionNode(**simf, hash_init='NR_chops_annot_hash')
    # kernel_vs_ged(simfunc=simfunc, node_list=node_list, ged_matrix=ged_matrix)
    #
    # run = 'dot_rg'
    # result = {}
    # ged_pickle = '../data/geds_rooted_depth2_n1000_fix.p'
    # geds, nodes = pickle.load(open(ged_pickle, 'rb'))
    # GED = geds[np.triu_indices(len(nodes), 1)]
    # GED = np.exp(-np.array(GED) / 5)
    #
    # ged_argsort = np.argsort(GED)
    # GED = [GED[i] for i in ged_argsort]
    # last = GED.index(1)
    #
    # ks = embs_vs_ged(run=run,
    #                  graph_path="../data/annotated/whole_v3",
    #                  ged_pickle='../data/geds_rooted_depth2_n1000_fix.p',
    #                  max_nodes=10000,
    #                  plot=False)
    # # ks = ks[argsort0]
    # ks = ks[np.triu_indices(len(nodes), 1)]
    # ks = [ks[i] for i in ged_argsort]
    #
    # score, _ = pearsonr(GED, ks)
    # print(f'based on the whole list, score : {score}')
    #
    # crop = 2000
    # cropped_ged, cropped_ks = GED[-crop:last], ks[-crop:last]
    # score, _ = pearsonr(cropped_ged, cropped_ks)
    # print(f'based on {len(cropped_ged)} elements crop-last, score : {score}')
