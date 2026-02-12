"""
Load RNA data from rnaglib RNADataset and convert to vernal format.

Uses rnaglib.dataset.RNADataset to fetch 2.5D graphs and structures.
Converts to vernal format (label edge attr, chain/pdb_pos node attr).
"""

import os
import sys

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

import networkx as nx
from tqdm import tqdm

from tools.graph_utils import write_nx_graph

# rnaglib LW -> vernal label (uppercase C/T for cis/trans)
LW_TO_LABEL = {
    'B53': 'B53',
    'B35': 'B35',
    'cWW': 'CWW', 'cWH': 'CWH', 'cWS': 'CWS', 'cHW': 'CHW', 'cHH': 'CHH', 'cHS': 'CHS',
    'cSW': 'CSW', 'cSH': 'CSH', 'cSS': 'CSS',
    'tWW': 'TWW', 'tWH': 'TWH', 'tWS': 'TWS', 'tHW': 'THW', 'tHH': 'THH', 'tHS': 'THS',
    'tSW': 'TSW', 'tSH': 'TSH', 'tSS': 'TSS',
}


def rnaglib_to_vernal_graph(g_rnaglib):
    """
    Convert rnaglib graph to vernal format.

    - LW edge attr -> label (uppercase)
    - Add chain, pdb_pos to node data for blob_to_graph compatibility
    - Convert to undirected for vernal (keeps one edge per pair)
    """
    G = nx.Graph()

    for n, data in g_rnaglib.nodes(data=True):
        # rnaglib node id: pdbid.chain.pos
        parts = n.split('.')
        if len(parts) >= 3:
            chain = parts[1]
            pdb_pos = parts[2]
        else:
            chain = data.get('chain_name', data.get('chain_id', ''))
            pdb_pos = str(data.get('nt_resnum', data.get('index_chain', '')))
        node_data = dict(data)
        node_data['chain'] = chain
        node_data['pdb_pos'] = pdb_pos
        G.add_node(n, **node_data)

    seen_edges = set()
    for u, v, data in g_rnaglib.edges(data=True):
        lw = data.get('LW', '')
        if lw not in LW_TO_LABEL:
            continue
        label = LW_TO_LABEL[lw]
        edge_key = frozenset([u, v])
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            G.add_edge(u, v, label=label)

    return G


def convert_dataset_to_vernal(dataset, vernal_graph_dir):
    """
    Convert RNADataset to vernal format and save as .nx

    :param dataset: rnaglib.dataset.RNADataset instance
    :param vernal_graph_dir: output path for vernal .nx graphs
    """
    os.makedirs(vernal_graph_dir, exist_ok=True)

    for i in tqdm(range(len(dataset)), desc="Converting rnaglib -> vernal"):
        try:
            item = dataset[i]
            g_rglib = item['rna']
            rna_name = dataset.all_rnas.inv[i]
            pdbid = rna_name.lower()
            g_vernal = rnaglib_to_vernal_graph(g_rglib)
            dst = os.path.join(vernal_graph_dir, f"{pdbid}.nx")
            write_nx_graph(g_vernal, dst)
        except Exception as e:
            print(f"Failed {rna_name}: {e}")

    return vernal_graph_dir


def setup_vernal_from_rnaglib(
    output_name='rnaglib_nr',
    redundancy='nr',
    version='2.0.2',
    debug=False,
    script_dir=None,
):
    """
    Main entry: create RNADataset, convert to vernal format, set up directories.

    Returns (graph_path, structure_path) for use with preprocess_data.
    """
    try:
        from rnaglib.dataset import RNADataset
    except ImportError:
        raise ImportError("rnaglib is required. Install with: pip install rnaglib")

    if script_dir is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, '..')
    data_dir = os.path.join(base_dir, 'data')

    # Create RNADataset (downloads data if needed)
    print(">>> Loading RNADataset from rnaglib...")
    dataset = RNADataset(
        redundancy=redundancy,
        version=version,
        get_pdbs=True,
        debug=debug,
        in_memory=False,
    )

    # Convert graphs to vernal format (whole graphs - read by chopper)
    vernal_graph_dir = os.path.join(data_dir, 'graphs', output_name + '_whole')
    print(f">>> Converting {len(dataset)} graphs to vernal format in {vernal_graph_dir}")
    convert_dataset_to_vernal(dataset, vernal_graph_dir)

    # Structure path from RNADataset
    structure_path = getattr(dataset, 'structures_path', None)
    if structure_path is None:
        print(">>> Warning: No structures found. Run: rnaglib_download -r nr -pdb")
        from rnaglib.utils import get_default_download_dir
        structure_path = os.path.join(get_default_download_dir(), 'structures')

    return vernal_graph_dir, structure_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='rnaglib_nr')
    parser.add_argument('-r', '--redundancy', default='nr', choices=['nr', 'all'])
    parser.add_argument('-v', '--version', default='2.0.2')
    parser.add_argument('--debug', action='store_true', help='Use small debug subset')
    args = parser.parse_args()

    graph_path, struct_path = setup_vernal_from_rnaglib(
        output_name=args.name,
        redundancy=args.redundancy,
        version=args.version,
        debug=args.debug,
    )
    print(f"\nDone. Graph path: {graph_path}")
    print(f"Structure path: {struct_path}")
    print(f"\nRun: python prepare_data/main.py -n {args.name} --source rnaglib")
