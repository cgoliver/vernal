"""
Turn the original data into the corresponding graphs and the corresponding annotations
"""

import sys
import os
import argparse

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from prepare_data.chopper import all_rna_process
from prepare_data.annotator import annotate_all
from tools.utils import makedir


def setup_dirs():
    """
    Adds all ignored features
    :return:
    """
    script_dir = os.path.dirname(__file__)
    makedir(os.path.join(script_dir, '../data/annotated'), permissive=True)
    makedir(os.path.join(script_dir, '../data/graphs'), permissive=True)
    print('Done creating files')


def preprocess_data(name, in_graph='samples_graphs', in_pdb='samples_pdb',
                    graph_path_abs=None, pdb_path_abs=None):
    """

    :param name: name of the 'experiment' : where the data is going to be produced
    :param in_graph: name of the folder of the data to read from (relative to data/)
    :param in_pdb: name of the folder of the pdb to read from (relative to data/)
    :param graph_path_abs: if set, absolute path to graph directory (overrides in_graph)
    :param pdb_path_abs: if set, absolute path to structure directory (overrides in_pdb)
    :return:
    """
    script_dir = os.path.dirname(__file__)
    base_dir = os.path.join(script_dir, '..')
    dest_graph_path = os.path.join(base_dir, "data/graphs/", name)
    annotated_path = os.path.join(base_dir, "data/annotated/", name)
    makedir(dest_graph_path, permissive=False)
    makedir(annotated_path, permissive=False)

    if graph_path_abs is not None and pdb_path_abs is not None:
        graph_path = graph_path_abs
        pdb_path = pdb_path_abs
    else:
        graph_path = os.path.join(base_dir, 'data/graphs', in_graph)
        pdb_path = os.path.join(base_dir, 'data', in_pdb)

    # do the chopping
    all_rna_process(graph_path=graph_path,
                    pdb_path=pdb_path,
                    dest=dest_graph_path)
    print('Done producing graphs')

    annotate_all(graph_path=dest_graph_path, dump_path=annotated_path)
    print('Done annotating graphs')

    print('Done producing data')


if __name__ == '__main__':
    setup_dirs()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", "-da", type=str, default="all_rna_pdb_nr")
    parser.add_argument("--graph_dir", "-g", type=str, default="rna_graphs_nr")
    parser.add_argument("--name", "-n", type=str, default="default_name")
    parser.add_argument("--source", "-s", type=str, default=None,
                        choices=['rnaglib'],
                        help="Use rnaglib to fetch data (replaces manual download)")
    parser.add_argument("--debug", action='store_true',
                        help="Use small dataset subset (with --source rnaglib)")

    args, _ = parser.parse_known_args()
    pdb_dir = args.pdb_dir
    graph_dir = args.graph_dir
    name = args.name

    if args.source == 'rnaglib':
        from prepare_data.rnaglib_loader import setup_vernal_from_rnaglib
        script_dir = os.path.dirname(os.path.realpath(__file__))
        graph_path, pdb_path = setup_vernal_from_rnaglib(
            output_name=name,
            script_dir=script_dir,
            debug=args.debug,
        )
        preprocess_data(name, graph_path_abs=graph_path, pdb_path_abs=pdb_path)
    else:
        preprocess_data(in_graph=graph_dir, in_pdb=pdb_dir, name=name)
