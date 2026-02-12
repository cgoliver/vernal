#!/usr/bin/env python3
"""Create minimal test data for vernal training."""
import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import networkx as nx
from prepare_data.annotator import build_ring_tree_from_graph

# Create small RNA-like graphs with valid edge labels
EDGE_LABELS = ['CWW', 'B53', 'CSS', 'TSS']  # Valid labels from EDGE_MAP


def create_test_graph(n_nodes=8):
    """Create a small graph with valid RNA edge labels."""
    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(i)
    # Add edges in a simple structure
    for i in range(n_nodes - 1):
        label = EDGE_LABELS[i % len(EDGE_LABELS)]
        G.add_edge(i, i + 1, label=label)
    # Add one more edge to make it less trivial
    if n_nodes >= 4:
        G.add_edge(0, n_nodes - 1, label='B53')
    return G


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data', 'annotated', 'test_data')
    os.makedirs(data_dir, exist_ok=True)

    print(f"Creating test data in {data_dir}")

    for i in range(10):  # 10 small graphs
        G = create_test_graph(n_nodes=6 + (i % 4))
        rings = build_ring_tree_from_graph(G, depth=3, hasher=None)
        data = {'graph': G, 'rings': rings}
        path = os.path.join(data_dir, f'test_{i:03d}_annot.p')
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  Created {path}")

    print("Done. Run: python train_embeddings/main.py train -n test_run -da test_data -ep 2 -bs 2")


if __name__ == '__main__':
    main()
