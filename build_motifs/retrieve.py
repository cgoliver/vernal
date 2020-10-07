"""
File to use and expand the meta graph
"""

import sys
import os
import pickle
import json
import matplotlib.pyplot as plt
import time
import numpy as np
import random

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from tools.graph_utils import graph_from_node, whole_graph_from_node, has_NC, induced_edge_filter
from tools.learning_utils import inference_on_graph_run
from tools.drawing import rna_draw, rna_draw_pair, rna_draw_grid
from motif_build.meta_graph import MGraph, MGraphAll


def parse_json(json_file):
    """
    Parse the json motifs to only get the ones with examples in our data
    and return a dict {motif_id : list of list of nodes (instances)}
    """

    def parse_dict(dict_to_parse, motifs_prefix):
        """
        inner function to apply to bgsu and carnaval dicts
        """
        res_dict = dict()
        for motif_id, motif_instances in dict_to_parse.items():
            filtered_instances = list()
            for instance in motif_instances:
                filtered_instance = []
                for node in instance:
                    node = node['node']
                    if node is not None:
                        (a, (b, c)) = node
                        node = (a, (b, c))
                        filtered_instance.append(node)
                if filtered_instance:
                    filtered_instances.append(filtered_instance)
            if filtered_instances:
                motif_id = (motifs_prefix, motif_id)
                res_dict[motif_id] = filtered_instances
        return res_dict

    whole_dict = json.load(open(json_file, 'r'))

    motifs = dict()

    rna3dmotif = parse_dict(whole_dict['rna3dmotif'], 'rna3dmotif')
    motifs.update(rna3dmotif)

    # bgsu = parse_dict(whole_dict['bgsu'], 'bgsu')
    # motifs.update(bgsu)
    #
    # carnaval = parse_dict(whole_dict['carnaval'], 'carnaval')
    # motifs.update(carnaval)
    return motifs


def prune_motifs(motifs_dict, shortest=4, sparsest=3, non_canonical=True, non_redundant=True):
    """
    Clean the dict by removing sparse or small motifs
    :param motifs_dict:
    :return:
    """
    res_dict = {}
    sparse, short, nc = 0, 0, 0
    tot_inst, nr_inst = 0, 0
    mean_instance, mean_nodes = list(), list()
    non_redundant_list = set(os.listdir(os.path.join(script_dir, '../data/unchopped_v4_nr')))
    for mid, instances in motifs_dict.items():
        instance = instances[0]
        if non_redundant:
            tot_inst += len(instances)
            instances = [instance for instance in instances if instance[0][0] in non_redundant_list]
            nr_inst += len(instances)
        if len(instances) < sparsest:
            sparse += 1
            continue
        if len(instance) < shortest:
            short += 1
            continue
        if non_canonical:
            instance = instances[0]
            graph = whole_graph_from_node(instance[0])
            motif_graph = graph.subgraph(instance)
            if not has_NC(motif_graph):
                nc += 1
                continue
        mean_instance.append(len(instances))
        mean_nodes.append(len(instances[0]))
        res_dict[mid] = instances

    print(f'filtered {sparse} on sparsity, {short} on length, {nc} on non canonicals')
    print(f'non redundancy removed {tot_inst - nr_inst} /{tot_inst} instances')
    print(f'On average, {np.mean(mean_instance)} instances of motifs with {np.mean(mean_nodes)} nodes')
    return res_dict


def compute_embs(instance, run):
    """
    :param instance: a list of nodes that form a motif
    Parse the json motifs to only get the ones with examples in our data
    and return a dict {motif_id : list of list of nodes (instances)}
    """
    source_graph = whole_graph_from_node(instance[0])
    embs, node_map = inference_on_graph_run(run, source_graph)
    return embs, node_map


def get_outer_border(nodes, graph=None):
    if graph is None:
        graph = whole_graph_from_node(nodes[0])
    # expand the trimmed retrieval
    out_border = set()
    for node in nodes:
        for nei in graph.neighbors(node):
            if nei not in nodes:
                out_border.add(nei)
    return out_border


def trim(instance, depth=1, whole_graph=None):
    """
    Remove nodes around the border of a motif
    """
    if whole_graph is None:
        whole_graph = whole_graph_from_node(instance[0])
    out_border = get_outer_border(instance, whole_graph)

    # get the last depth ones as well as the cumulative set
    cummulative, last = out_border, out_border
    for d in range(depth):
        depth_ring = set()
        for node in last:
            for nei in whole_graph.neighbors(node):
                if nei not in cummulative:
                    depth_ring.add(nei)
        last = depth_ring
        cummulative = cummulative.union(depth_ring)
    trimmed_instance = [node for node in instance if node not in cummulative]
    return trimmed_instance


def trim_try(whole_graph, instance, depth=1):
    """
    To keep some graph, we cannot always perform trimming. Try with decreasing values of depth
    :param graph:
    :param instance:
    :param max_depth:
    :return:
    """
    trimmed = []
    trimmed_graph = whole_graph.subgraph(trimmed)
    while not trimmed_graph.edges():
        trimmed = trim(instance, depth=depth, whole_graph=whole_graph)
        trimmed_graph = whole_graph.subgraph(trimmed)
        if depth < 1:
            trimmed = instance
            trimmed_graph = whole_graph.subgraph(instance)
            break
        depth -= 1
    return trimmed, trimmed_graph, depth


def plot_instance(instance, source_graph=None):
    """
    Plot an extended, native, and trimmed motif.
    :param instance:
    :return:
    """
    if source_graph is None:
        source_graph = whole_graph_from_node(instance[0])
    trimmed = trim(instance)
    out_border = get_outer_border(instance, source_graph)
    extended = instance + list(out_border)
    g_motif = source_graph.subgraph(extended)
    rna_draw(g_motif, node_colors=['red' if n in trimmed else 'blue' if n in instance else 'grey' for n in
                                   g_motif.nodes()])
    plt.show()


def draw_hit(hit, mg, instance=None):
    """
    Plot the hit. If an instance is given, then compares this hit with the original instance
    :param hit:
    :param mg:
    :param instance:
    :param compare:
    :return:
    """
    try:
        hit_graph = whole_graph_from_node(hit[0])
    except:
        hit = [mg.reversed_node_map[i] for i in hit]
        hit_graph = whole_graph_from_node(hit[0])

    out_border = get_outer_border(hit, hit_graph)
    full_hit = hit + list(out_border)
    out_border = get_outer_border(full_hit, hit_graph)
    extended_hit = full_hit + list(out_border)
    g_hit = hit_graph.subgraph(extended_hit)
    if instance is not None:
        source_graph = whole_graph_from_node(instance[0])
        trimmed = trim(instance)
        out_border = get_outer_border(instance, source_graph)
        extended = instance + list(out_border)
        g_motif = source_graph.subgraph(extended)
        rna_draw_pair((g_motif, g_hit)
                      , node_colors=(
                ['red' if n in trimmed else 'blue' if n in instance else 'grey' for n in g_motif.nodes()],
                ['red' if n in hit else 'blue' if n in full_hit else 'grey' for n in g_hit.nodes()]))
        plt.show()
    else:
        rna_draw(g_hit, node_colors=['red' if n in hit else 'blue' if n in full_hit else 'grey' for n in g_hit.nodes()])
        plt.show()


def retrieve_instances(query_instance, mg, depth=1):
    # DEBUG
    # print(query_instance)
    # query_g = whole_graph_from_node(motif[0][0]).subgraph(motif[0])
    # failure_g = whole_graph_from_node(motif[1][0]).subgraph(motif[1])
    # failure_g2 = whole_graph_from_node(motif[2][0]).subgraph(motif[2])
    # rna_draw_pair((query_g, failure_g))
    # plt.show()
    # rna_draw_pair((query_g, failure_g2))
    # plt.show()
    # rna_draw_pair((failure_g2, failure_g))
    # plt.show()

    query_whole_graph = whole_graph_from_node(query_instance[0])

    # Sometimes one can not trim the motif as much as we could have like, so we need to trim less
    trimmed, trimmed_graph, actual_depth = trim_try(query_whole_graph, query_instance, depth=depth)
    # print('starting the retrieval')
    start = time.perf_counter()
    retrieved_instances = mg.retrieve_2(trimmed)
    print(f">>> Retrieved {len(retrieved_instances)} instances in {time.perf_counter() - start}")

    # retrieved_instances_2 = mg.retrieve_2(trimmed)
    # print(retrieved_instances == retrieved_instances_2)
    # set1 = set(retrieved_instances.items())
    # set2 = set(retrieved_instances_2.items())
    # print(set1 ^ set2)
    # print(f">>> Retrieved {len(retrieved_instances_2)} instances in {time.perf_counter() - start}")

    return retrieved_instances


def find_hits(motif, mg, depth=1, query_instance=None):
    if query_instance is None:
        query_instance = motif[0]

    retrieved_instances = retrieve_instances(mg=mg, depth=depth, query_instance=query_instance)

    sorted_scores = sorted(list(retrieved_instances.values()), key=lambda x: -x)
    # start = time.perf_counter()
    res = list()
    failed = 0
    # convert motif into set of ids
    for other_instance in motif[1:]:
        instance_res = 0
        set_form = set([mg.node_map[node] for node in other_instance])

        best = -1
        for hit, score in retrieved_instances.items():
            if len(hit.intersection(set_form)) > 0:
                if score > best:
                    best = score
                    rank = sorted_scores.index(score)
                    instance_res = (hit, score, rank)

                # DEBUG
                # try:
                #     pass
                #     # draw_hit(hit, mg, instance)
                # except:
                #     continue

            # DEBUG PLOTS
            # query_g = whole_graph_from_node(motif[0][0]).subgraph(motif[0])
            # failure_g = whole_graph_from_node(other_instance[0]).subgraph(other_instance)
            # rna_draw_pair((query_g, failure_g))
            # plt.show()
            # raise ValueError()
        if instance_res == 0:
            failed += 1
        else:
            res.append(instance_res)

    instance_ranks = [item[2] for item in res]
    if not instance_ranks:
        mean_best = len(retrieved_instances)
    else:
        mean_best = np.mean(instance_ranks)
    best_ratio = mean_best / len(retrieved_instances)
    fail_ratio = failed / len(motif[1:])
    # print(res)
    # print(f">>> Hits parsed in {time.perf_counter() - start}")

    return mean_best, best_ratio, failed, fail_ratio


def hit_ratio_all(motifs, mg, depth=1):
    # Motif 1 and 2 are isomorphic...
    # motifs = list(motifs.values())[:4]
    # query_g = whole_graph_from_node(motifs[0][0][0]).subgraph(motifs[0][0])
    # failure_g = whole_graph_from_node(motifs[1][0][0]).subgraph(motifs[1][0])
    # rna_draw_pair((query_g, failure_g))
    # plt.show()
    all_best = list()
    all_best_ratio = list()
    all_fails = list()
    all_fails_ratio = list()
    for i, (motif_id, motif) in enumerate(motifs.items()):
        if int(i) > 5:
            break
        print('attempting id : ', motif_id)
        mean_best, best_ratio, failed, fail_ratio = find_hits(motif, mg, depth=1)
        all_best.append(mean_best)
        all_fails.append(failed)
        all_best_ratio.append(best_ratio)
        all_fails_ratio.append(fail_ratio)
    print(f'on average, {np.mean(all_fails):.4f} fails for a {np.sum(all_fails_ratio):.4f} ratio')
    print(f'And {np.mean(all_best):.4f} rank for a {np.mean(all_best_ratio):.4f} ratio')
    return all_fails, all_best


def ab_testing(motifs, mg, depth=1):
    # Motif 1 and 2 are isomorphic...
    # motifs = list(motifs.values())[:4]
    # query_g = whole_graph_from_node(motifs[0][0][0]).subgraph(motifs[0][0])
    # failure_g = whole_graph_from_node(motifs[1][0][0]).subgraph(motifs[1][0])
    # rna_draw_pair((query_g, failure_g))
    # plt.show()
    all_best = list()
    all_best_ratio = list()
    all_fails = list()
    all_fails_ratio = list()
    other_all_best = list()
    other_all_best_ratio = list()
    other_all_fails = list()
    other_all_fails_ratio = list()

    all_motifs = [(motif_id, motif) for motif_id, motif in motifs.items()]
    for i, (motif_id, motif) in enumerate(all_motifs):

        # if int(motif_id) != 5:
        #     continue
        print('attempting id : ', motif_id)
        mean_best, best_ratio, failed, fail_ratio = find_hits(motif, mg, depth=1)
        all_best.append(mean_best)
        all_fails.append(failed)
        all_best_ratio.append(best_ratio)
        all_fails_ratio.append(fail_ratio)

        # Pick another random that is not the current graph
        other_random = random.randint(0, len(all_motifs) - 2)
        if other_random >= i:
            other_random += 1
        random_query_instance = all_motifs[other_random][1][0]
        mean_best, best_ratio, failed, fail_ratio = find_hits(motif, mg, depth=1, query_instance=random_query_instance)
        other_all_best.append(mean_best)
        other_all_fails.append(failed)
        other_all_best_ratio.append(best_ratio)
        other_all_fails_ratio.append(fail_ratio)
    print(f'on average, {np.mean(all_fails):.4f} fails for a {np.mean(all_fails_ratio):.4f} ratio')
    print(f'And {np.mean(all_best):.4f} rank for a {np.mean(all_best_ratio):.4f} ratio')
    print(f'For random query, {np.mean(other_all_fails):.4f} fails for a {np.mean(other_all_fails_ratio):.4f} ratio')
    print(f'And {np.mean(other_all_best):.4f} rank for a {np.mean(other_all_best_ratio):.4f} ratio')
    return all_fails, all_best


def ged_computing(motifs, mg, depth=1):
    from tools.rna_ged_nx import ged
    res_dict = dict()
    all_motifs = [(motif_id, motif) for motif_id, motif in motifs.items()]
    for i, (motif_id, motif) in enumerate(all_motifs):
        inner_dict = {}

        # if int(motif_id) != 4:
        #     continue

        # Get the hits
        print('attempting id : ', motif_id)
        query_instance = motif[0]
        query_whole_graph = whole_graph_from_node(query_instance[0])
        retrieved_instances = retrieve_instances(query_instance=query_instance, mg=mg, depth=depth)
        sorted_hits = sorted(list(retrieved_instances.items()), key=lambda x: -x[1])

        # Get the actual query that was used (because of trimming) and expand it with the depth
        # This is V2, the original one was just computing between the trimmed and the reduced hit
        # trimmed, trimmed_graph = trim_try(whole_graph=query_whole_graph, instance=query_instance)
        # query_instance_graph = query_whole_graph.subgraph(query_instance)
        trimmed, trimmed_graph, actual_depth = trim_try(whole_graph=query_whole_graph, instance=query_instance)
        query_instance_graph = induced_edge_filter(query_whole_graph, trimmed, depth=actual_depth)

        plot_index = [0, 10, 100, 1000]
        for j in plot_index:
            # In case we have less than 1000 hits
            try:
                hit = sorted_hits[j][0]
            except IndexError:
                continue
            hit = [mg.reversed_node_map[node] for node in hit]
            # print(hit)
            hit_whole_graph = whole_graph_from_node(hit[0])

            # If one changes this, one should also remove the query expansion
            hit_graph = induced_edge_filter(hit_whole_graph, hit, depth=actual_depth)
            # hit_graph = whole_graph_from_node(hit[0]).subgraph(hit)
            start = time.perf_counter()
            ged_value = ged(query_instance_graph, hit_graph, timeout=2)
            print(j, len(query_instance_graph), len(hit_graph), ged_value, time.perf_counter() - start)
            # res_dict[j].append(ged_value)
            inner_dict[j] = ged_value

            # TO PLOT THE HITS
            # expanded = hit
            # if depth > 0:
            #     out_border = get_outer_border(hit, hit_whole_graph)
            #     expanded = hit + list(out_border)
            # expanded_graph = hit_whole_graph.subgraph(expanded)

            colors = [['red' if n in trimmed else 'grey' for n in query_instance_graph.nodes()],
                      ['red' if n in hit else 'grey' for n in hit_graph.nodes()]]
            subtitles = ('', ged_value)
            rna_draw_pair((query_instance_graph, hit_graph), node_colors=colors, subtitles=subtitles)
            plt.show()

        # Pick another random that is not the current graph
        other_random = random.randint(0, len(all_motifs) - 2)
        if other_random >= i:
            other_random += 1
        random_query_instance = all_motifs[other_random][1][0]
        # random_graph = whole_graph_from_node(random_query_instance[0]).subgraph(random_query_instance)
        random_graph = induced_edge_filter(whole_graph_from_node(random_query_instance[0]),
                                           random_query_instance, depth=actual_depth)


        # TO PLOT THE RANDOM
        # colors = [['red' if n in trimmed else 'grey' for n in query_instance_graph.nodes()],
        #           ['grey' for n in random_graph.nodes()]]
        # rna_draw_pair((query_instance_graph, random_graph), node_colors=colors)
        # plt.show()

        # res_dict['random_other'].append(ged(query_instance_graph, random_graph, timeout=5))
        start = time.perf_counter()
        inner_dict['random_other'] = (ged(query_instance_graph, random_graph, timeout=2))
        print('random', len(trimmed_graph), len(hit_graph), inner_dict['random_other'], time.perf_counter() - start)
        print(inner_dict)
        res_dict[motif_id] = inner_dict

    return res_dict


def draw_smooth(motif, mg, depth=1, save=None):
    """
    Draws graphs from the retrieve further and further away
    :param motif:
    :param mg:
    :param depth:
    :return:
    """

    query_instance = motif[0]
    query_whole_g = whole_graph_from_node(query_instance[0])

    # Sometimes one can not trim the motif as much as we could have like, so we need to trim less
    # trimmed, trimmed_graph = trim_try(whole_graph=query_whole_g, instance=query_instance, max_depth=0)
    # query_instance_graph = query_whole_g.subgraph(query_instance)
    trimmed, trimmed_graph, actual_depth = trim_try(whole_graph=query_whole_g, instance=query_instance, depth=depth)
    query_instance_graph = induced_edge_filter(query_whole_g, trimmed, depth=actual_depth)

    retrieved_instances = retrieve_instances(query_instance=query_instance, mg=mg, depth=depth)
    sorted_hits = sorted(list(retrieved_instances.items()), key=lambda x: -x[1])

    # TO GET CONTEXT NODES
    # out_border = get_outer_border(motif[0], query_whole_g)
    # expanded = motif[0] + list(out_border)
    # expanded_graph = query_whole_g.subgraph(expanded)

    graphs = [query_instance_graph]
    colors = [['red' if n in trimmed else 'white' for n in query_instance_graph.nodes()]]
    # colors = [['red' if n in trimmed else 'grey' if n in query_instance else 'blue' for n in expanded_graph.nodes()]]
    subtitles = ['Query']
    plot_index = [10, 100, 1000]
    for i in plot_index:
        hit = sorted_hits[i][0]
        hit = [mg.reversed_node_map[i] for i in hit]
        hit_whole_graph = whole_graph_from_node(hit[0])
        # expand if trimmed
        full_hit = hit

        hit_graph = induced_edge_filter(hit_whole_graph, hit, depth=actual_depth)
        # if depth > 0:
        #     for d in range(depth):
        #         out_border = get_outer_border(full_hit, hit_whole_graph)
        #         full_hit = full_hit + list(out_border)
        # hit_graph = hit_whole_graph.subgraph(full_hit)

        graphs.append(hit_graph)
        colors.append(['red' if n in hit else 'white' for n in hit_graph.nodes()])
        subtitles.append(f'{i}-th hit with score : {sorted_hits[i][1]:2.2f}')
    # rna_draw_pair(graphs=graphs, node_colors=colors, subtitles=subtitles, save=save)

    rna_draw_grid(graphs=graphs, node_colors=colors, subtitles=subtitles, save=save, grid_shape=(2, 2))

    plt.show()


if __name__ == '__main__':
    pass
    random.seed(0)

    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--run', type=str, default="1hopmg")
    parser.add_argument('--run', type=str, default="2hop_unchopped")
    args, _ = parser.parse_known_args()

    # Get pruned data
    # all_motifs = parse_json('../data/all_motifs_unchopped.json')
    # pruned_motifs = prune_motifs(all_motifs)
    # print(f'{len(pruned_motifs)}/{len(all_motifs)} motifs kept')
    # pickle.dump(pruned_motifs, open('../data/pruned_motifs_chill.p', 'wb'))
    pruned_motifs = pickle.load(open('../results/motifs_files/pruned_motifs.p', 'rb'))

    # Load meta-graph model
    mgg = pickle.load(open('../results/motifs_files/' + args.run + '.p', 'rb'))

    # Use the retrieve to get hit ratio
    # all_failed, all_res = hit_ratio_all(pruned_motifs, mgg)
    # all_failed, all_res = ab_testing(pruned_motifs, mgg)
    # print(f"this is the result for {args.run}")

    sample_motif = pruned_motifs['63']
    # sample_id, sample_motif = pruned_motifs.popitem()
    # sample_id, sample_motif = pruned_motifs.popitem()
    draw_smooth(sample_motif, mgg)
    #
    # res_dict_ged = ged_computing(motifs=pruned_motifs, mg=mgg)
    # print(res_dict_ged)
    # pickle.dump(res_dict_ged, open('res_dict_ged.p', 'wb'))
