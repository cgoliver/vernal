"""
    Intra vs inter GED hisgograms for MAGA.
"""
import os
import sys
import pickle
import multiprocessing as mlp

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__": sys.path.append(os.path.join(script_dir, '..'))

from build_motifs.meta_graph import *
from tools.graph_utils import whole_graph_from_node
from tools.graph_utils import induced_edge_filter
from tools.new_drawing import rna_draw_pair
from tools.rna_ged_nx import ged

np.random.seed(0)

TIMEOUT = 120
GRAPH_PATH = "../../RNAGlib/data/iguana/NR"

def do_geds(args):
    g, h = args
    return ged(g, h, timeout=TIMEOUT)

def non_redundants(mnodes):
    """
        Select non-redundant mnodes
    """

    non_redundants = set()
    for node in mnodes:
        for rep in non_redundants:
            pass

    pass

def histogram(maga_path,
              inter_size=10,
              intra_size=10):
    """
        Histogram of intra and inter motif GED.
        (If too slow, use embeddings as proxy.)
    """
    print("Loading maga")
    mgg = pickle.load(open(maga_path, 'rb'))
    print("maga loaded")
    singletons = [node for node in mgg.maga_graph.nodes() if len(node) == 4]
    print(f"got {len(singletons)} singletons")
    mnode_sample_inds = np.random.choice(list(range(len(singletons))),
                                     size=inter_size,
                                     replace=False)
    mnode_sample = [singletons[i] for i in mnode_sample_inds]
    # do inter motif GED
    motif_instances_graphs = []
    graphs = {}

    print("building dataset")
    for node in tqdm(mnode_sample):
        instances = list(mgg.maga_graph.nodes[node]['node_set'])
        if len(instances) < 1:
            print("WTH ", node)
        instance_graphs = []
        for instance in instances:
            node_inds = list(instance)
            node_ids = [mgg.reversed_node_map[n] for n in node_inds]
            g_id = node_ids[0].split(".")[0]
            if g_id not in graphs:
                G = whole_graph_from_node(node_ids[0], graph_dir=GRAPH_PATH)
                graphs[g_id] = G
            else:
                G = graphs[g_id]
            context = bfs_expand(G, node_ids, depth=1)
            G_instance = G.subgraph(context).copy()
            G_instance = induced_edge_filter(G, node_ids, depth=1)
            instance_graphs.append(G_instance)
        motif_instances_graphs.append(instance_graphs)

    motif_graphs = [m[0] for m in motif_instances_graphs]

    print(">>> computing inter GED")
    pairs = itertools.combinations(motif_graphs, 2)
    pool = mlp.Pool(20)
    inter_geds = list(pool.imap_unordered(do_geds, pairs))
    pool.close()

    # do intra motif ged
    intra_geds = []
    print(">>> computing intra GED")
    for motif in tqdm(motif_instances_graphs):
        test = motif[:intra_size]
        pairs = itertools.combinations(test, 2)
        pool = mlp.Pool(20)
        geds = list(pool.imap_unordered(do_geds, pairs))
        pool.close()
        intra_geds.extend(geds)

    # p = pickle.load(open('inter_intra.p', 'rb'))
    # inter_geds = p['inter']
    # intra_geds = p['intra']
    print("inter: ", np.mean(inter_geds), np.std(inter_geds))
    print("intra: ", np.mean(intra_geds), np.std(intra_geds))

    # sys.exit()

    pickle.dump({'inter': inter_geds, 'intra': intra_geds},
                open("inter_intra.p", "wb"))

    sns.distplot(inter_geds,
                 label='inter',
                 hist=True,
                 kde_kws={'shade': True, 'color': 'red'})
    sns.distplot(intra_geds,
                 label='intra',
                 hist=True,
                 kde_kws={'shade': True, 'color': 'blue'})

    sns.despine()
    plt.ylabel('Frequency')
    plt.xlabel('Graph Edit Distance')

    plt.legend()
    # plt.savefig("../figs/intra_inter.pdf", format="pdf")

    plt.show()


if __name__ == "__main__":
    histogram("../results/magas/default_name.p", intra_size=4, inter_size=4)
    pass
