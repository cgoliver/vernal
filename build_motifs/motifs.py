"""
    Functions to merge meta-graph nodes and aggregate motifs.
    Takes a meta-graph as input.
"""
from collections import defaultdict

import multiset as ms
from tqdm import tqdm
import networkx as nx

from tools.graph_utils import whole_graph_from_node
from tools.graph_utils import has_NC_bfs

def def_set():
    return defaultdict(set)

def merge_nodesets(maga_graph,
                   maga_adj,
                   singleton,
                    motif
                    ):
    """
        For each instance in motif, try to extend it with an edge
        that goes into the singleton cluster.
    """
    # set of instances of new motif
    merged_nodeset = set()

    # set of nodes of current motif
    nodesets = maga_graph.nodes[motif]['node_set']

    # cluster ID to merge with
    singleton = list(singleton)[0]
    # iterate through instances of current motif
    for nodeset in nodesets:
        # for each node in instances, look for a connection to the
        # singleton cluster
        for node in nodeset:
            try:
                # get nodes in form (node, *) where
                # * contains all nodes connected to `node` that
                # belong to motif `singleton`
                singleton_nodes = maga_adj[node][singleton]
            except KeyError:
                pass
            else:
                # we have a new instance for each connection from
                    # current motif to singleton
                    for s in singleton_nodes:
                        merged = nodeset.union({s})
                        if len(merged) == len(nodeset):
                            continue
                        else:
                            merged_nodeset.add(nodeset.union({s}))

    return merged_nodeset

def maga_next(maga_graph,
              maga_tree,
              maga_adj,
              mgraph,
              boring_clusters,
              min_edge=100,
              levels=6,
              max_boring=3):
    """
        Merge all connections between motifs and singletons.
        This means iterating through meta edges,
        combining the instances between the motif and the singleton,
        and adding connections from the new motif and adjacent
        singletons.

        Each edge in current state is of the form (motif, singleton).
        Once merged, the new motif is size |motif| + 1,
    """

    for l in range(levels):
        todo_edges = list(maga_graph.edges())
        done_edges = list()

        to_kill = list()

        for (singleton, motif) in tqdm(todo_edges):

            # merge phase
            new_node = motif.combine(singleton)
            new_nodeset = merge_nodesets(maga_graph,
                                         maga_adj,
                                         singleton,
                                         motif)

            # don't repeat the same edge
            edgeset = set([singleton, motif])
            if edgeset in done_edges:
                continue
            done_edges.append(edgeset)

            # remove merged edge from maga graph
            try:
                maga_graph.remove_edge(singleton, motif)
            except nx.exception.NetworkXError:
                pass
            try:
                maga_graph.remove_edge(motif, singleton)
            except nx.exception.NetworkXError:
                pass

            # get number of boring clusters in motif
            boring_count = 0
            for clust in new_node:
                if clust in boring_clusters:
                    boring_count += 1
            if boring_count > max_boring:
                continue

            # if new motif is too sparse, move to next one
            # if not, add it to the meta graph
            if len(new_nodeset) < min_edge:
                continue
            else:
                maga_graph.add_node(new_node, node_set=new_nodeset)
                maga_tree.add_edge(motif, new_node)

            # connect phase
            # for each cluster in the motif, add connection to other clutters
            # e.g. for ABC, look at all singletons adjacent to A, B, and C
            for clust in new_node:
                for nei in mgraph.graph.neighbors(clust):
                    maga_graph.add_edge(ms.FrozenMultiset([nei]), new_node)
            pass

            # if new motif is similar in size to the old one,
            # kill the old motif
            pre_merge_size = len(maga_graph.nodes[motif]['node_set'])
            if len(new_nodeset) >= .8 * pre_merge_size:
                to_kill.append(motif)
                continue

        maga_graph.remove_nodes_from(to_kill)

        yield maga_graph


def maga(mgraph, levels=10):
    print(f">>> Meta-graph has {len(mgraph.graph.nodes())} nodes",
                f"and {len(mgraph.graph.edges())} edges.")
    maga_graph = nx.relabel_nodes(mgraph.graph,
                                  {n: ms.FrozenMultiset([n]) for n in mgraph.graph})

    maga_graph = maga_graph.to_directed()

    # how many times to sample a cluster for boring ones
    n_boring_samples = 100
    # keep track of how many instances of each cluster are boring.
    boring_clusters = {c: {'samples': 0.01, 'boring': 0} for c in set(mgraph.labels)}

    maga_tree = nx.DiGraph()
    maga_tree.add_nodes_from((ms.FrozenMultiset([n]) for n in mgraph.graph))

    # this dictionary is of the following form
    # {u: {c1: {v, w}, c2: {x}}}
    # maps each node to the clusters to which it is connected
    # for each connected cluster, we store the endpoint of the edge.
    # in this case, node `u` is connected to clusters `c1` via `v`, `w`,
    # and connected to `c2` via `x`

    maga_adj = defaultdict(def_set)

    print(">>> Building MAGA graph.")
    for n in maga_graph.nodes():
        maga_graph.nodes[n]['node_set'] = set()
    for c1, c2, d in tqdm(maga_graph.edges(data=True)):
        maga_graph[c1][c2]['edge_set'] = {frozenset([u, v]) for u, v, _ in d['edge_set']}
        for u, v in d['edge_set']:

            maga_adj[u][mgraph.labels[v]].add(v)
            maga_adj[v][mgraph.labels[u]].add(u)

            for node in (u,v):
                clust = mgraph.labels[node]
                if boring_clusters[clust]['samples'] < n_boring_samples:
                    boring_clusters[clust]['samples'] += 1
                    node_id = mgraph.reversed_node_map[node]
                    G = whole_graph_from_node(node_id)
                    if not has_NC_bfs(G, node_id, depth=1):
                        boring_clusters[clust]['boring'] += 1

            u_node = ms.FrozenMultiset([mgraph.labels[u]])
            v_node = ms.FrozenMultiset([mgraph.labels[v]])
            maga_graph.nodes[u_node]['node_set'].add(frozenset([u]))
            maga_graph.nodes[v_node]['node_set'].add(frozenset([v]))

    # consider a cluster boring if at at least 80% of instances are boring
    boring_clusters = {clust for clust, counts in boring_clusters.items()\
                            if counts['boring'] / counts['samples'] > .8}

    print(">>> Doing MAGA.")
    maga_build = maga_next(maga_graph,
                           maga_tree,
                           maga_adj,
                           mgraph,
                           boring_clusters,
                           levels=levels)
    for l, maga_graph in enumerate(maga_build):
        print("maga level ", l)
        print("maga nodes ", len(maga_graph.nodes()),
                "maga edges ", len(maga_graph.edges())
                )

    return maga_graph

if __name__ == "__main__":
    pass
