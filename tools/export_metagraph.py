#!/usr/bin/env python3
"""
Export a Vernal meta-graph pickle to JSON for the motif viewer.

Uses the MAGA graph when available, which has connected instance nodesets.
Falls back to building MAGA from the meta-graph if the MAGA file is missing.

Usage:
    python tools/export_metagraph.py results/mggs/my_metagraph.p -o motifs.json
    python tools/export_metagraph.py results/mggs/my_metagraph.p -o motifs.json --max-instances 5
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle

from tools.graph_utils import whole_graph_from_node


def get_pdbid(node_name):
    """Extract PDB ID from node name (e.g. '1d0t.A.1' -> '1d0t')."""
    return str(node_name).split('.')[0].lower()[:4]


def _to_int(x):
    """Convert numpy scalar or int to Python int."""
    return int(x.item()) if hasattr(x, 'item') else int(x)


def nodeset_to_instance(mgraph, nodeset, graph_dir, reversed_node_map, graph_provider=None):
    """
    Convert a nodeset (frozenset of node indices) from MAGA to a viewer instance.
    Each nodeset is guaranteed to be connected by the MAGA construction.
    """
    if len(nodeset) < 2:
        return None

    node_names = []
    for idx in nodeset:
        if idx in reversed_node_map:
            node_names.append(reversed_node_map[idx])
        else:
            return None

    pdbid = get_pdbid(node_names[0])
    annot_dir = os.path.abspath(graph_dir) if graph_dir else None
    if not annot_dir and graph_provider is None:
        return None

    try:
        G = whole_graph_from_node(
            node_names[0], annot_dir=annot_dir, graph_provider=graph_provider
        )
        sub = G.subgraph(node_names).copy()
        if sub.number_of_edges() == 0:
            return None

        name_to_idx = {reversed_node_map[idx]: idx for idx in nodeset}
        nodes = []
        for n in sub.nodes():
            idx = name_to_idx.get(n)
            cluster_id = _to_int(mgraph.labels[idx]) if idx is not None else None
            # Label as <chain>.<residue number> for graph view
            parts = str(n).split('.')
            label = f"{parts[-2]}.{parts[-1]}" if len(parts) >= 3 else (parts[-1] if parts else str(n))
            nodes.append({
                'id': n,
                'label': label,
                'cluster_id': cluster_id,
            })
        links = [{'source': u, 'target': v, 'label': d.get('label', '')} for u, v, d in sub.edges(data=True)]
        return {
            'graph_id': pdbid,
            'nodes': nodes,
            'links': links,
        }
    except Exception:
        return None


def export_from_maga(mgraph, maga_graph, graph_dir, max_instances=20, graph_provider=None):
    """Export motifs from MAGA graph - each node_set frozenset is a connected instance."""
    reversed_node_map = mgraph.reversed_node_map
    motifs = []

    for maga_node in maga_graph.nodes():
        node_set = maga_graph.nodes[maga_node].get('node_set', set())
        if not node_set:
            continue

        instances = []
        for nodeset in node_set:
            if len(instances) >= max_instances:
                break
            inst = nodeset_to_instance(
                mgraph, nodeset, graph_dir, reversed_node_map,
                graph_provider=graph_provider,
            )
            if inst:
                instances.append(inst)

        if not instances:
            continue

        # maga_node is FrozenMultiset of cluster IDs (may be numpy scalars) - convert for clean display
        def _to_int(x):
            return int(x.item()) if hasattr(x, 'item') else int(x)

        cluster_ids = tuple(_to_int(x) for x in sorted(maga_node))
        cluster_display = str(cluster_ids) if len(cluster_ids) > 1 else str(cluster_ids[0])

        motifs.append({
            'id': len(motifs),
            'cluster_id': cluster_display,
            'sigma': 0.0,
            'count': sum(len(ns) for ns in node_set),
            'num_instances': len(instances),
            'num_nodes': len(instances[0]['nodes']),
            'num_edges': len(instances[0]['links']),
            'graphs': instances,
        })

    return motifs


def export_metagraph(metagraph_path, output_path, max_instances=20, maga_graph=None):
    """Export meta-graph to JSON for the motif viewer. Uses MAGA graph when available."""
    with open(metagraph_path, 'rb') as f:
        mgraph = pickle.load(f)

    graph_provider = getattr(mgraph, 'graph_provider', None)
    graph_dir = getattr(mgraph, 'graph_dir', None)
    if graph_provider is None and not graph_dir:
        base = os.path.dirname(os.path.abspath(metagraph_path))
        graph_dir = os.path.join(base, '..', 'data', 'graphs', 'rnaglib_nr_whole')
    if graph_dir:
        graph_dir = os.path.abspath(graph_dir)
        if not os.path.isdir(graph_dir):
            proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(metagraph_path))))
            graph_dir = os.path.abspath(os.path.join(proj_root, graph_dir))
        if not os.path.isdir(graph_dir):
            raise FileNotFoundError(f"Graph directory not found: {graph_dir}")
    elif graph_provider is None:
        raise FileNotFoundError("Meta-graph has no graph_dir or graph_provider")

    # Load or build MAGA graph
    if maga_graph is None:
        maga_path = metagraph_path.replace('.p', '_maga.p')
        if os.path.isfile(maga_path):
            with open(maga_path, 'rb') as f:
                maga_graph = pickle.load(f)
            print(f">>> Loaded MAGA graph from {maga_path}")
        else:
            print(">>> MAGA graph not found, building from meta-graph...")
            from build_motifs.motifs import maga
            maga_graph = maga(mgraph, levels=6)

    motifs = export_from_maga(
        mgraph, maga_graph, graph_dir or '',
        max_instances=max_instances,
        graph_provider=graph_provider,
    )

    out = {
        'meta_graph': os.path.splitext(os.path.basename(metagraph_path))[0],
        'graph_dir': graph_dir,
        'motifs': motifs,
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"Exported {len(motifs)} motifs to {output_path}")
    return out


def main():
    parser = argparse.ArgumentParser(description='Export Vernal meta-graph to JSON for motif viewer')
    parser.add_argument('metagraph', help='Path to meta-graph pickle (e.g. results/mggs/my_metagraph.p)')
    parser.add_argument('-o', '--output', default='motifs.json', help='Output JSON path')
    parser.add_argument('--max-instances', type=int, default=20,
                        help='Max instances per motif to include (default: 20)')
    args = parser.parse_args()

    export_metagraph(args.metagraph, args.output, max_instances=args.max_instances)


if __name__ == '__main__':
    main()
