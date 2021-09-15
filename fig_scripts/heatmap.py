"""
    Code for heatmap figure (our motifs vs externals overlap)
"""

import os
import sys
import pickle
import json
from collections import defaultdict
from collections import Counter

import numpy as np
import networkx as nx
from tqdm import tqdm

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from build_motifs.meta_graph import *
# from tools.drawing import rna_draw_pair
from tools.learning_utils import inference_on_list

def node_to_motifs_build(motifs_pickle):
    """
        Dict which gives motifs for each node.
    """

    motifs_to_nodes = pickle.load(open(motifs_pickle, 'rb'))

    print("NUM MOTIFS: ", len(motifs_to_nodes))

    nodes_to_motifs = defaultdict(list)

    for motif, instances in motifs_to_nodes.items():
        for instance in instances:
            for node_id in instance:
                nodes_to_motifs[node_id].append(motif)

    return nodes_to_motifs


def motif_eval(mgg,
               node_to_motifs,
               external_motifs,
               motif_set='bgsu',
               graph_dir="../data/graphs"
               ):
    """
        Compute recall for 'bgsu' and 'carnaval' motifs.

        :param motifs: dictionary of motifs
        :param graphs: list of networkx graphs
        :param node_map: map from node_id to graph index
    """

    all_motifs_scores = {}

    # node_to_motifs = node_to_motifs_nr
    whole_graphs = {}

    for c, (maga_node, d) in tqdm(enumerate(mgg.maga_graph.nodes(data=True)), total=len(mgg.maga_graph.nodes())):
        motif_scores = defaultdict(int)
        hit_counts = defaultdict(int)
        if len(maga_node) > 6:
            continue
        # if len(maga_node) < 4:
            # continue
        maga_id = "-".join(map(str, list(maga_node)))
        try:
            nodeset = d['node_set']
        except:
            continue
        for i, instance in enumerate(nodeset):
            instance_nodes = [mgg.reversed_node_map[n] for n in list(instance)]
            g_id = instance_nodes[0].split(".")[0]
            if g_id not in whole_graphs:
                G_motif = whole_graph_from_node(instance_nodes[0],
                                                graph_dir=graph_dir
                                                )
                whole_graphs[g_id] = G_motif
            else:
                G_motif = whole_graphs[g_id]
            instance_nodes = bfs_expand(G_motif, instance_nodes, depth=1)

            # count motif instances in graph, per node
            motif_counter = Counter()
            for node in instance_nodes:
                motif_counter.update(node_to_motifs[node])
            for m in motif_counter:
                # compute instance coverage, percentage of motif instance recovered
                motif_size = len(external_motifs[m][0])
                n_instances_motif = len(external_motifs[m])
                # if n_instances_motif < 20:
                    # continue
                maga_size = len(instance_nodes)

                num, den = sorted((motif_size, maga_size))


                # fraction of nodes of motif m in our current instance
                # normalized by ratio of the sizes between our motif and theirs
                # instance_coverage = (motif_counter[m] / maga_size) * (num/ den)
                if motif_size != maga_size:
                    instance_coverage = 0
                else:
                    instance_coverage = motif_counter[m] / motif_size

                # motif_scores[m] = max(instance_coverage, motif_scores[m])
                motif_scores[m] = instance_coverage

        all_motifs_scores[maga_id] = motif_scores

    return all_motifs_scores

def motif_scores_to_df(scores):
    """
        Build pandas dataframe from scores dictionary.
        Columns 'ours', 'theirs','score'.
        Coverage score for our motifs vs their motifs.
    """

    import pandas as pd
    frozen_str = lambda s, sep: sep.join(map(str, s))
    counter = 0
    rows = []
    ours_tot = set()
    for ours, theirs in scores.items():
        ours_tot.add(ours)
        for motif_id, score in theirs.items():
            motif_id = "-".join(motif_id).replace("_", "-")
            if score < .5:
                score = 0
            rows.append({'ours': ours, 'theirs': motif_id, 'score': score})

    df = pd.DataFrame(rows)
    pivot = df.pivot(index='theirs', columns='ours', values='score')
    pivot = pivot.fillna(0)
    pivot = pivot.sort_index()
    pivot.to_csv("pivot_general_fuzzy_annot.csv")

    dataset = ['bgsu', 'rna3dmotif', 'carnaval']

    table_rows = []

    for d in dataset:
        covered = 0
        missed = 0
        # find which ones we recover
        current_motifs = [m for m in pivot.index if d in m]
        for their_motif in current_motifs:
            if pivot.loc[their_motif].max() > .6:
                covered += 1
            else:
                missed += 1

        # find which ones of ours arent in theirs.
        new_ones = 0
        old_ones = 0
        for our_motif in pivot.columns:
            for theirs in current_motifs:
                if pivot[our_motif][their_motif] > .6:
                    old_ones += 1
                    break
            else:
                new_ones += 1

        table_rows.append({'dataset': d,
                           'covered': covered,
                           'missed': missed,
                           'novel': new_ones})

    table = pd.DataFrame(table_rows)
    print(table.to_latex(index=False))
    known = [
             'IL-73276.1',
             'IL-24982.1',
             'HL-97270.1',
             'HL-72498.5',
             'HL-67042.4',
             'IL-49493.2',
             'IL-85647.2']

    # find ours that dont have match with theirs
    # NOTE: all of these get killed by pruning.
    for k in known:
        try:
            sns.distplot(pivot['bgsu-' + k])
            plt.show()
        except KeyError:
            continue

    return pivot

def motif_heatmap(df):
    from numpy.random import rand
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.image import AxesImage

    # kill columns with no hits to theirs
    empties = []
    for col in df.columns:
        if df[col].max() == 0:
            empties.append(col)
    print(f"NOVEL : ", len(empties))
    df = df.drop(columns=empties)
    df = df.replace(0, np.nan)
    print(df)


    sns.heatmap(df,
                cmap="Reds",
                cbar_kws={'label': 'Overlap'},
                vmin=0.0,
                vmax=1)
                # yticklabels=2)
    plt.tight_layout()
    plt.savefig("../figs/heatmap_general_fuzzy_annot.pdf")

    plt.show()

def make(maga_path,
         motif_pickle="../results/motifs_files/pruned_motifs_NR.p",
         graph_dir="../data/graphs"
         ):
    node_to_motifs = node_to_motifs_build(motif_pickle)

    external_motifs = pickle.load(open(motif_pickle, 'rb'))
    print("Loading maga.")
    mgg = pickle.load(open(maga_path, 'rb'))
    print("Loaded maga.")

    scores = motif_eval(mgg,
               node_to_motifs,
               external_motifs,
               graph_dir=graph_dir
               )

    df = motif_scores_to_df(scores)
    motif_heatmap(df)
    pass

if __name__ == "__main__":
    make('../results/magas/default_name.p',
         graph_dir="../../RNAGlib/data/iguana/NR"
        )
    pass
