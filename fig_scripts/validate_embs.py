"""
Compute GED values and compare them to kernel computations as well as embeddings dot product.
"""

import os
import sys

import dgl
from functools import partial
import itertools
import multiprocessing
import pickle
import numpy as np
import networkx as nx
import pandas as pd
import random
from scipy.stats import pearsonr
import time
from tqdm import tqdm
import torch

import seaborn as sns
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from config.graph_keys import GRAPH_KEYS, TOOL
from tools.graph_utils import has_NC_bfs, bfs_expand
from tools.node_sim import SimFunctionNode
from tools.rna_ged_nx import ged
from tools.learning_utils import load_model


def get_nodelist(graph_dir='data/annotated/NR_chops_annot', depth=2):
    """
    Get nodelist at random: sample 200 chopped graphs and then pick a random graph around a NC using rejection sampling

    Dump a list containing dicts : {'graph': nx graphlet, 'graphlet': graphlet rings, 'edge': edge rings}
    """
    graphlist = sorted(os.listdir(graph_dir))
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
            if has_NC_bfs(graph=graph, node_id=node, depth=depth):
                found_nc = True
                break
        if not found_nc:
            continue
        subg = list(bfs_expand(graph, [node], depth=depth)) + [node]
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
    todo = list(itertools.combinations_with_replacement(graphlets_list, 2))
    pool = multiprocessing.Pool()
    ged_matrix = list(tqdm(pool.imap(get_one_ged, todo), total=len(todo)))
    return ged_matrix


def build_experiments_list(depth=2):
    def build_experiments(method, param_dict):
        combinations = itertools.product(*(param_dict[name] for name in param_dict))
        experiments = list()
        for comb in combinations:
            new_dict = {key: value for key, value in zip(param_dict.keys(), comb)}
            new_dict['method'] = method
            experiments.append(new_dict)
        return experiments

    R_1_dict = {'normalization': [None, 'sqrt'], 'idf': [True, False], 'depth': [depth], 'decay': [0.3, 0.5, 0.8]}
    R_iso_dict = {'normalization': [None, 'sqrt'], 'idf': [True, False], 'depth': [depth], 'decay': [0.3, 0.5, 0.8]}
    hungarian_dict = {'normalization': [None, 'sqrt'], 'idf': [True, False], 'depth': [depth]}
    all_methods = [('R_1', R_1_dict),
                   ('R_iso', R_iso_dict),
                   ('hungarian', hungarian_dict)]

    graphlet_dict = {'depth': [depth], 'normalization': [None, 'sqrt']}
    R_graphlets_dict = {'depth': [depth], 'decay': [0.3, 0.5, 0.8], 'normalization': [None, 'sqrt']}
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
                  plot=True,
                  ged_thresh=None
                  ):
    """
        Compute correlation between GED and simfunc outputs.
    node_list is a list of nodes in the form return by get_nodelist :
        a dict with attributes node, graph, edge_ring, graphlet_ring
    """

    ged_matrix = np.array(ged_matrix)[:max_nodes]
    if ged_thresh is not None:
        ged_sel = ged_matrix < ged_thresh
    ged_flat = np.exp(- np.array(ged_matrix) / 5)

    level = 'graphlet' if simfunc.method in ['graphlet', 'R_graphlets'] else 'edge'
    annots_list = [node[level] for node in node_list]
    todo = list(itertools.combinations_with_replacement(annots_list, 2))[:max_nodes]
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
    ged_flat = ged_flat[:len(ks)]

    if ged_thresh is not None:
        ged_flat = ged_flat[ged_sel]
        ks = ks[ged_sel]
    if plot:
        sns.regplot(ged_flat, ks, scatter_kws={'s': 1})
        plt.title(f"{simfunc.method}, pearson: {pearsonr(ged_flat, ks)}")
        plt.show()
    return pearsonr(ged_flat, ks)


def all_kernels_ged(node_list, ged_matrix, depth=2, max_nodes=None, ged_thresh=None):
    all_experiments = build_experiments_list(depth=depth)
    all_simfunc = [SimFunctionNode(**simf, hash_init='NR_chops_annot_hash') for simf in all_experiments]
    pool = multiprocessing.Pool()
    default_kernel_vs_ged = partial(kernel_vs_ged, node_list=node_list, ged_matrix=ged_matrix,
                                    plot=False, print_tqdm=False, max_nodes=max_nodes, ged_thresh=ged_thresh)
    correlation_values = list(tqdm(pool.imap(default_kernel_vs_ged, all_simfunc), total=len(all_simfunc)))
    return correlation_values


def embs_vs_ged(run,
                node_list,
                ged_matrix,
                ged_thresh=None,
                plot=False):
    """
        Compute correlation between GED and simfunc outputs.
    """

    model = load_model(run)

    graphs_list = [node['graph'] for node in node_list]
    main_node_list = [node['node'] for node in node_list]

    edge_map = GRAPH_KEYS['edge_map'][TOOL]
    lw_labels = GRAPH_KEYS['bp_type'][TOOL]

    embeddings_list = list()
    for i, graph in enumerate(graphs_list):
        # Build dgl graph and make the inference
        one_hot = {edge: torch.tensor(edge_map[label]) for edge, label in
                   (nx.get_edge_attributes(graph, lw_labels)).items()}
        nx.set_edge_attributes(graph, name='one_hot', values=one_hot)
        g_dgl = dgl.from_networkx(nx_graph=graph, edge_attrs=['one_hot'])
        outputs = model(g_dgl).detach().cpu()

        # Find the correct index for the main node
        main_node = main_node_list[i]
        correct_index = list(sorted(graph.nodes())).index(main_node)
        embeddings_list.append(outputs[correct_index])

    # Node embeddings : array of shape (n_graphlets, embedding dim)
    node_embeddings = torch.stack(embeddings_list)

    # Now compute a distance matrix/similarity matrix
    # This follows the computations that happen internally in the model
    if model.similarity:
        if model.normalize:
            K_predict = model.matrix_cosine(node_embeddings, node_embeddings)
        else:
            K_predict = torch.mm(node_embeddings, node_embeddings.t())

    else:
        K_predict = model.matrix_dist(node_embeddings)
    K_predict = K_predict.numpy()

    # DEBUG : just check the correlation with the kernel
    # n_hops = int(run.split('_')[-1].split('.')[0])
    # filename = f'temp_ks_graphlets_{n_hops}.p'
    # if not os.path.exists(filename):
    #     simf = {'depth': n_hops, 'normalization': 'sqrt', 'method': 'graphlet'}
    #     simfunc = SimFunctionNode(**simf, hash_init='NR_chops_annot_hash')
    #     level = 'graphlet' if simfunc.method in ['graphlet', 'R_graphlets'] else 'edge'
    #     annots_list = [node[level] if level == 'graphlet' else node[level][1:] for node in node_list]
    #     compute_annots = list(itertools.combinations_with_replacement(annots_list, 2))
    #     kernel_values = list()
    #     for i, (ring_1, ring_2) in tqdm(enumerate(compute_annots), total=len(compute_annots)):
    #         k = simfunc.compare(ring_1, ring_2)
    #         kernel_values.append(k)
    #     kernel_values = np.array(kernel_values)
    #     pickle.dump(kernel_values, open(filename, 'wb'))
    # kernel_values = pickle.load(open(filename, 'rb'))

    # Put it in flat form, and compute the potentially thresholded correlation value
    upper_triangle_indices = np.triu_indices(len(node_embeddings))
    embeddings_similarities = K_predict[upper_triangle_indices]
    ged_matrix = np.array(ged_matrix)
    ged_flat = np.exp(- np.array(ged_matrix) / 5)
    if ged_thresh is not None:
        ged_sel = ged_matrix < ged_thresh
        ged_flat = ged_flat[ged_sel]
        embeddings_similarities = embeddings_similarities[ged_sel]
        # kernel_values = kernel_values[ged_sel]

    if plot:
        sns.regplot(embeddings_similarities, ged_flat, scatter_kws={'s': 1})
        plt.title(f" pearson: {pearsonr(ged_flat, embeddings_similarities)}")
        plt.show()

    return pearsonr(ged_flat, embeddings_similarities)


if __name__ == "__main__":
    pass
    random.seed(0)
    np.random.seed(0)
    graph_dir = 'data/annotated/NR_chops_annot'
    dump_dir = 'results/correlations'
    n_hops = 2
    model_name = f'new_kernel_{n_hops}'

    # Get the nodelist
    # node_list = get_nodelist(depth=n_hops, graph_dir=graph_dir)
    # pickle.dump(node_list, open(f'{dump_dir}/nodelist_{n_hops}hop.p', 'wb'))
    node_list = pickle.load(open(f'{dump_dir}/nodelist_{n_hops}hop.p', 'rb'))

    # Compute GED between graphlets
    # ged_matrix = get_geds(node_list=node_list)
    # pickle.dump(ged_matrix, open(f'{dump_dir}/ged_matrix_{n_hops}hop.p', 'wb'))
    ged_matrix = pickle.load(open(f'{dump_dir}/ged_matrix_{n_hops}hop.p', 'rb'))

    # Compute kernel values for a list of experiments and correlate them with the GED
    get_kernel_correlations = True
    recompute = False
    if get_kernel_correlations:
        all_experiments = build_experiments_list(depth=n_hops)
        if recompute:
            correlation_values = all_kernels_ged(node_list=node_list, ged_matrix=ged_matrix, depth=n_hops)
            pickle.dump(correlation_values, open(f'{dump_dir}/correlation_values_{n_hops}hop.p', 'wb'))
        correlation_values = pickle.load(open(f'{dump_dir}/correlation_values_{n_hops}hop.p', 'rb'))
        correlation_values = np.array([x[0] for x in correlation_values])

        # Same thing with a threshold on the GED value
        if recompute:
            correlation_values_thresh = all_kernels_ged(node_list=node_list, ged_matrix=ged_matrix, depth=n_hops,
                                                        ged_thresh=6)
            pickle.dump(correlation_values_thresh, open(f'{dump_dir}/correlation_values_thresh_{n_hops}hop.p', 'wb'))
        correlation_values_thresh = pickle.load(open(f'{dump_dir}/correlation_values_thresh_{n_hops}hop.p', 'rb'))
        correlation_values_thresh = np.array([x[0] for x in correlation_values_thresh])

        # Add all experiments in a pandas dataframe and sort/export to latex.
        print_df = True
        if print_df:
            for i, expe in enumerate(all_experiments):
                expe[f'{n_hops}_hop_correlation'] = correlation_values[i]
                expe[f'{n_hops}_hop_correlation_thresh'] = correlation_values_thresh[i]
            df = pd.DataFrame()
            for expe_dict in all_experiments:
                df = df.append(expe_dict, ignore_index=True)
            df = df.sort_values(by=f'{n_hops}_hop_correlation', ascending=False)
            df = df[["method", "decay", "idf", "normalization",
                     f"{n_hops}_hop_correlation", f"{n_hops}_hop_correlation_thresh"]]
            df = df.rename(columns={"method": "Method",
                                    "decay": "Decay",
                                    "idf": "IDF",
                                    "normalization": "Normalization",
                                    f"{n_hops}_hop_correlation": "Correlation",
                                    f"{n_hops}_hop_correlation_thresh": "Thresholded Correlation"
                                    })
            print(df.to_latex(index=False))

    # Compute the correlation, now using the embeddings obtained with our model, and do the same.
    get_model_correlation = True
    if get_model_correlation:
        correlation, _ = embs_vs_ged(run=model_name,
                                     ged_matrix=ged_matrix,
                                     node_list=node_list,
                                     ged_thresh=None)
        thresh_correlation, _ = embs_vs_ged(run=model_name,
                                            ged_matrix=ged_matrix,
                                            node_list=node_list,
                                            ged_thresh=6)
        print(f'Correlation with the model is {correlation:.3f}, thresholded is :{thresh_correlation:.3f}')
