"""
    Compare two 2.5D graphs using VERNAL similarity functions.
    Graphs must have backbone + 12 base pair geometries as edge labels.
"""


import sys
import os
from functools import partial

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

import numpy as np
from scipy.optimize import linear_sum_assignment

from prepare_data.annotator import build_ring_tree_from_graph
from tools.graphlet_hash import Hasher
from tools.node_sim import SimFunctionNode

def compare_graphs(g1, g2, depth=2):
    """
    Takes two 2.5D RNA graphs with edges having the 'label' attribute to be
    one of 13 possible labels (1 backbone + 12 geometries)
    Returns the SIMILARITY (0 dissimilar, 1 identical) between them.
    Defaults to the graphlet similarity function. I'll add more customization
    later.

    Args:
    ---
    g1 (networkx graph): networkx graph with proper LW edges
    g2 (networkx graph): graph to compare against
    depth (int): how deep to do each node's comparison.

    Returns:
    ---
    sim (float): score between 0 and 1
    """

    rings_1 = build_ring_tree_from_graph(g1, depth=5, hasher=None)
    rings_2 = build_ring_tree_from_graph(g2, depth=5, hasher=None)

    simfunc = SimFunctionNode('R_graphlets', 2, cache=False)

    cost = - np.array(
        [[simfunc.compare(ring_i, ring_j) for _,ring_i in rings_1['graphlet'].items()]\
                                          for _,ring_j in rings_2['graphlet'].items()]
                       )
    row_ind, col_ind = linear_sum_assignment(cost)
    c = - np.array(cost[row_ind, col_ind]).sum()
    return simfunc.normalize(c, max(len(g1.nodes()), len(g2.nodes())))

def k_most_similar(g, motif_db, k=5):
    """ Return the `k` most similar graphs to `g` from `motif_db.


    Args:
    ---
    g (networkx graph): input graph (output from bayespairing)
    motif_db (list): list of netwrorkx graphs (i.e. bayespairing motif db)
    k (int): number of matches to return.
    """

    return sorted(map(partial(compare_graphs, g), motif_db))[-k:]

if __name__ == "__main__":

    import networkx as nx

    G1 = nx.Graph()
    G1.add_edges_from(
        [
            (1, 2, {"label": "CWW"}),
            (2, 3, {"label": "B53"}),
            (3, 1, {"label": "B53"}),
            (1, 4, {"label": "CHS"}),
            (4, 2, {"label": "THS"}),
        ]
    )
    G2 = nx.Graph()
    G2.add_edges_from(
        [
            (6, 7, {"label": "B53"}),
            (7, 5, {"label": "B53"}),
            (7, 8, {"label": "CWW"}),
            (8, 9, {"label": "CWW"}),
        ]
    )

    s =compare_graphs(G1, G2)
    print(s)
