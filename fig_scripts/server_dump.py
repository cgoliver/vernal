"""
Dump graph drawings for server
"""

import sys
import os
import json
import pickle

import multiset as ms
import seaborn as sns

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from build_motifs.meta_graph import *
from tools.new_drawing import rna_draw
from tools.graph_utils import induced_edge_filter

static_path = "../../vernal_server/static"

def draw_motif(mgg,
               mnode,
               dump_path,
               graph_dir="../data/graphs",
               n_instances=30,
               ):
    try:
        os.mkdir(dump_path)
        os.mkdir(os.path.join(dump_path, "graphs"))
        os.mkdir(os.path.join(dump_path, "metadata"))
    except:
        pass

    clusts = set(list(mnode))
    pal = sns.color_palette("muted", n_colors=len(clusts))
    clust_to_ind = {c:i for i,c in enumerate(clusts)}

    instance_nodesets = list(mgg.maga_graph.nodes[mnode]['node_set'])
    motif_info = {'n_instances': len(instance_nodesets),
                  'mnode': mnode_string(mnode)
                 }

    with open(os.path.join(dump_path, f"motif_info.json"), "w") as j:
        json.dump(motif_info, j)

    for i, instance_nodes in enumerate(instance_nodesets[:n_instances]):
        instance_nodes = list(instance_nodes)
        instance_nodes = [mgg.reversed_node_map[n] for n in instance_nodes]

        # pull graph
        G = whole_graph_from_node(
                                  instance_nodes[0],
                                  graph_dir=graph_dir
                                  )
        context = bfs_expand(G, instance_nodes, depth=1)
        g = G.subgraph(context)
        g = induced_edge_filter(g, instance_nodes, depth=1)

        metadata = {}
        metadata['pdbid'] = instance_nodes[0].split(".")[0]

        with open(os.path.join(dump_path, "metadata", f"{i}.json"), "w") as j:
            json.dump(metadata, j)

        # get node colors
        node_colors = []
        node_labels = {}
        for n in g.nodes():
            int_id = mgg.node_map[n]
            if n in instance_nodes:
                clust = mgg.labels[int_id]
                node_colors.append(pal[clust_to_ind[clust]])
                node_labels[n] = str(clust)
            else:
                node_colors.append("white")
                node_labels[n] = ""

        rna_draw(g,
                 node_colors=node_colors,
                 show=False,
                 save=os.path.join(dump_path, "graphs", f"{i}.png"),
                 format="png"
                 )

    pass

def mnode_string(mnode):
    return '-'.join(map(str, sorted(mnode)))

def dump_all(mgg_name, dump_id):
    mgg = pickle.load(open(f'../results/magas/{mgg_name}.p', 'rb'))
    metadata = dict()
    try:
        os.mkdir(os.path.join(static_path, dump_id))
    except:
        pass

    for mnode in mgg.maga_graph.nodes():
        name = mnode_string(mnode)
        draw_motif(mgg, 
                   mnode, 
                   os.path.join(static_path, dump_id, name),
                   graph_dir="../../RNAGlib/data/iguana/NR"
                   )
    pass

if __name__ == "__main__":
    mgg = "default_name"
    dump_all(mgg, "bioinformatics_1")
