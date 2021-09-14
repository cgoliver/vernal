"""
    Drawing examlples of novel MAGA motifs.
"""

import os
import sys
import pickle
import json
import itertools

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from build_motifs.meta_graph import *
from tools.new_drawing import rna_draw
from tools.graph_utils import whole_graph_from_node
from tools.graph_utils import bfs_expand
from tools.graph_utils import induced_edge_filter
from tools.rna_ged_nx import ged_approx
from tools.rna_ged_nx import ged

TIMEOUT = None

def novelty(graph, motif_pickle, approxes=5):
    """
        Compute distances from graph to known motifs.
    """
    external_motifs = pickle.load(open(motif_pickle, 'rb'))

    distances = []
    for motif, instances in tqdm(external_motifs.items()):
        # patch for avoiding redundants
        found = False
        for instance_nodes in instances:
            try:
                G_motif = whole_graph_from_node(instance_nodes[0])
                G_motif = G_motif.subgraph(instance_nodes)
                G_motif = induced_edge_filter(G_motif, instance_nodes)
                my_size = len(graph.nodes())
                their_size = len(G_motif.nodes())
                print(my_size, their_size)
                if abs(my_size - their_size) > 1:
                    continue
            except FileNotFoundError:
                continue
            else:
                found = True
                break

            if found:
                print("DOING")
                d = ged(graph, G_motif, timeout=TIMEOUT)
                print(d)
                distances.append(d)
                if d < 2:
                    print(motif)
                # if len(distances) > 4:
                    # return np.array(distances)

    return np.array(distances)

def quality(graphs, approxes=100):
    intra_geds = []
    for g, h in itertools.combinations(graphs, 2):
        intra_geds.append(ged(g,h, timeout=TIMEOUT))
        print(intra_geds)

    intra_geds.append(0.1)
    return np.array(intra_geds)

def motifs_supp_fig(mgg,
                    n_instances=4,
                    n_motifs=4,
                    name='motifs',
                    graph_dir="../data/graphs"):
    from tools.motif_hash import motif_hash

    fig, axes = plt.subplots(nrows=n_motifs, ncols=n_instances+1)
    fig.set_size_inches(8.5, 11)

    mnodes_sample = [m for m in mgg.maga_graph.nodes() if len(m) <= 5]
    mnodes = np.random.choice(mnodes_sample,
                              size=n_motifs,
                              replace=False)

    for row, mnode in enumerate(mnodes):
        instance_nodesets = list(mgg.maga_graph.nodes[mnode]['node_set'])[:200]
        motif_topologies = defaultdict(list)
        mnode_instances = len(mgg.maga_graph.nodes[mnode]['node_set'])
        print("hashing")
        for instance_nodes in instance_nodesets:
            instance_nodes = list(instance_nodes)
            instance_nodes = [mgg.reversed_node_map[n] for n in instance_nodes]
            G = whole_graph_from_node(instance_nodes[0], graph_dir=graph_dir)
            context = bfs_expand(G, instance_nodes, depth=1)
            g = G.subgraph(context)
            g = induced_edge_filter(g, instance_nodes, depth=1)

            clusts = {n:mgg.labels[mgg.node_map[n]] for n in instance_nodes}

            h = motif_hash(g, instance_nodes, clusts)
            motif_topologies[h].append((g, instance_nodes))

        print('hashed')

        clusts = set(list(mnode))
        pal = sns.color_palette("muted", n_colors=len(clusts))
        clust_to_ind = {c:i for i,c in enumerate(clusts)}

        ind = 0
        # pick most populated topology to draw
        to_draw = sorted(motif_topologies.values(), key=lambda x:len(x),
                         reverse=True)[0]
        for col in range(n_instances+1):
            # if row == 1 and col == 2:
                # break
            if col == 0:
                label = f"{'-'.join(map(str, mnode))}, $n={mnode_instances}$"
                axes[row][col].text(.5, .5, label)
                axes[row][col].set_axis_off()
                continue
            try:
                g, instance_nodes = to_draw.pop()
            except:
                continue
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
                     ax=axes[row][col],
                     node_colors=node_colors,
                     show=False)

            ind += 1

    plt.tight_layout()
    plt.savefig(f"../figs/test_{name}.pdf", format="pdf")
    # plt.show()
    return
    pass

def instances_draw(mgg, mnode, n_instances=5, graph_dir="../data/graphs"):
    """
        Draw instances of a metanode.
    """
    fig, axes = plt.subplots(nrows=3, ncols=3)

    try:
        instance_nodesets = list(mgg.maga_graph.nodes[mnode]['node_set'])
    except KeyError:
        print(f"Missing nodeset: {mnode}")
        return

    pdb_set = set()
    for n in instance_nodesets:
        for node in n:
            nid = mgg.reversed_node_map[node]
            pdb_set.add(nid[0])
    titles = json.load(open("../data/pdb_titles.json", "r"))
    for pdb in pdb_set:
        tit = titles[pdb.replace(".nx", ".cif")][0].upper()
        if 'RIBOSOM' not in tit:
            print(tit, pdb)

    print("N instances ", len(instance_nodesets))
    # sample_nodesets = np.random.choice(instance_nodesets,
                                       # size=n_instances,
                                       # replace=False)

    clusts = set(list(mnode))
    print("N clusters ", len(set(mgg.labels)))
    pal = sns.color_palette("muted", n_colors=len(clusts))
    print(clusts)
    clust_to_ind = {c:i for i,c in enumerate(clusts)}

    print(">>> drawing instances")
    ind = 0
    for row in range(3):
        for col in range(3):
            # if row == 1 and col == 2:
                # break
            instance_nodes = list(instance_nodesets[ind])
            instance_nodes = [mgg.reversed_node_map[n] for n in instance_nodes]
            G = whole_graph_from_node(instance_nodes[0], graph_dir=graph_dir)
            context = bfs_expand(G, instance_nodes, depth=1)
            g = G.subgraph(context)
            g = induced_edge_filter(g, instance_nodes, depth=1)
            print(g.nodes(data=False))
            # axes[row][col].set_axis_off()


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
                     ax=axes[row][col],
                     node_colors=node_colors,
                     show=False)

            ind += 1

    plt.tight_layout()
    # plt.savefig("../figs/sample.pdf", format="pdf")
    plt.show()
    return
    print(">>> computing quality")
    # compute quality distribution
    sample_nodesets = np.random.choice(instance_nodesets,
                                       size=min(len(instance_nodesets), 10),
                                       replace=False)
    quality_graphs = []
    for nodeset in sample_nodesets:
        instance_nodes = list(nodeset)
        instance_nodes = [mgg.reversed_node_map[n] for n in nodeset]
        G = whole_graph_from_node(instance_nodes[0])
        # instance_nodes = bfs_expand(G, instance_nodes, depth=1)
        context = bfs_expand(G, instance_nodes, depth=1)
        g = G.subgraph(context).copy()
        g = induced_edge_filter(g, instance_nodes, depth=1)
        quality_graphs.append(g)

    # # print(len(quality_graphs))
    intra_geds = quality(quality_graphs)
    # # intra_geds = np.exp(-1 * intra_geds)
    g = sns.distplot(intra_geds,
                 hist=False,
                 kde_kws={'shade': True, 'color': 'blue'},
                 ax=axes[0][2])

    g.set_xlabel("Internal GED")
    # g.set_xlim([0, 1])
    # # compute distance to known motifs
    sns.despine(ax=g)

    print(">>> computing novelty")

    to_known = novelty(quality_graphs[0], '../data/pruned_motifs.p')
    # to_known = np.exp(-1 * to_known)
    g = sns.distplot(to_known,
                     hist=False,
                     kde_kws={'shade': True, 'color': 'red'},
                     ax=axes[1][2])

    # g.set_xlim([0, 1])
    axes[0][2].set_xlim(0, max(to_known))
    axes[1][2].set_xlim(0, max(to_known))
    g.set_xlabel(r"GED to Known Motifs")

    sns.despine(ax=g)

    plt.tight_layout()
    # plt.savefig("../figs/motif_display.pdf")
    plt.show()
    pass

def select_novel(pivot_path):
    """
        From overlap pivot table,
        select motifs which don't overlap with known ones.
        Then draw their instances.
    """
    import multiset as ms

    def to_multiset(motif_string):
        id_list = list(map(int, motif_string.split("-")))
        return ms.FrozenMultiset(id_list)

    df = pd.read_csv(pivot_path)
    novels = []
    not_novels = []
    df = df.set_index('theirs')
    want = ms.FrozenMultiset([189, 0])
    wants = []
    for col in df.columns:
        if len(col.split("-")) != 4:
            continue
        motif_ms = to_multiset(col)
        if df[col].max() == 0:
            novels.append(motif_ms)
            # if want.issubset(motif_ms):
                # print("HIII ", motif_ms)
                # wants.append(motif_ms)
        if df[col].max() > .8:
            not_novels.append((motif_ms, df.index[df[col] > .8].to_list()))
    # return wants
    return novels, not_novels

if __name__ == "__main__":
    print("Getting potential novel motifs")
    # novels, not_novels = select_novel("pivot_general_fuzzy_annot.csv")
    # want = select_novel("pivot_general_fuzzy_annot.csv")
    # print(f"Num novels: ", len(novels))
    # print(f"Num olds: ", len(not_novels))
    print("Loading MAGA")
    mgg = pickle.load(open('../results/magas/bioinformatics_1.p', 'rb'))
    print("Loaded MAGA")
    for i in range(100):
        print(i)
        motifs_supp_fig(mgg, 
                        name=f"grid_motifs_{i}",
                        graph_dir="../../RNAGlib/data/iguana/NR")
    sys.exit()
    # for w in want:
        # instances_draw(mgg, w, n_instances=6)

    instances_draw(mgg, ms.FrozenMultiset({192, 89, 112, 189}))
    print("NOVEL")
    # for n in novels[:-20:-1]:
    for n in novels[::-1]:
        print("DRAWING ", n)
        instances_draw(mgg, n)
    print("OLD")
    for n in not_novels[:-10:-1]:
        instances_draw(mgg, n)
    pass
