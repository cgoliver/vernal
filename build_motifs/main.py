"""
    Main module for motif building, given embeddings.
"""

import os
import sys
import time
import argparse
import pickle

if __name__ == "__main__":
    pass

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

def get_args():
    parser = argparse.ArgumentParser()
    # Meta graph Args
    parser.add_argument("--meta_graph", "-mg", type=str,
                        default="",
                        help="Meta graph iD to load. If none,\
                            build a new one")
    parser.add_argument("--rgcn", "-r", type=str,
                                        default="default_name",
                                        help="ID of trained embedding model")
    parser.add_argument("--graphs", "-g", type=str,
                                          default="data/graphs/rna_graphs_nr",
                                          help="Path to full graphs.")
    parser.add_argument('--mgg_name', "-mn", type=str,
                                  default="default_name",
                                  help="The name of the pickled meta graph.")
    parser.add_argument('--clust_algo', type=str,
                                        default="k_means",
                                        help="The clustering algo to use to \
                                              build the meta_graph : possible \
                                              choices are:\
                                              Gaussian Mixture Model,\
                                              K-Means or Self-Organizing Maps")
    parser.add_argument("-N", "--n_components", type=int,
                                                default=200,
                                                help="components in the\
                                                clustering")
    parser.add_argument('--prune', default=False,
                                   action='store_true',
                                   help="To make the meta graph sparser,\
                                        remove infrequent edges")
    parser.add_argument("--nc", default=False, action='store_true',
                                help="To use only nc"),

    # Motif build args

    parser.add_argument("--do_build", "-b",
                                      default=False,
                                      action="store_true",
                                      help="If True, build motifs from\
                                                  input meta-graph.")
    parser.add_argument("--levels", "-l", type=int,
                                          default=6,
                                          help="Maximum motif size (number of \
                                                merge operations)")
    parser.add_argument('--backbone', "-bb", default=False,
                                             action='store_true',
                                      help="If True, only connect via backbone.")

    parser.add_argument("--min_motif", "-m", type=int,
                                             default=100,
                                             help="Minimum number of instnaaces\
                                                   to count as motif.")
    parser.add_argument("--max_var", "-mv", type=float,
                                             default=0.01,
                                             help="Maximum cluster variance.")
    # Retrieve args

    parser.add_argument("--do_retrieve", "-rt",
                                            default=False,
                                            action="store_true",
                                            help="If True, build motifs from\
                                                  input meta-graph.")

    return parser.parse_known_args()

def build_mgraph(args):
    from build_motifs.meta_graph import MGraphAll
    start = time.perf_counter()
    mgg = MGraphAll(
                    run = args.rgcn,
                    clust_algo=args.clust_algo,
                    n_components=args.n_components,
                    optimize=False,
                    min_edge=args.min_motif,
                    max_var=args.max_var,
                    max_graphs=None,
                    graph_dir=args.graphs,
                    nc_only=args.nc
                    )
    print(f"Built Meta Graph in {time.perf_counter() - start} s")

    if args.prune:
        print("pruning")
        mgg.prune()

    print(f"Dumping meta graph in results/mggs/{args.mgg_name}")
    pickle.dump(mgg, open(os.path.join("results", "mggs", args.mgg_name + '.p'),
                          'wb'))
    return mgg

def build_motifs(mgraph, args):
    from build_motifs.motifs import maga
    maga_graph = maga(mgraph, levels=args.levels)
    pass

def retrieve():
    pass

def main():
    args,_ = get_args()
    pass
    if args.meta_graph:
        print(">>> Loading existing meta-graph.")
        mgraph = pickle.load(open(
                             os.path.join("results", "mggs", args.meta_graph + ".p"),
                             "rb"
                            ))
    else:
        print(">>> Building new meta graph.")
        mgraph = build_mgraph(args)

    if args.do_build:
        build_motifs(mgraph, args)

    if args.do_retrieve:
        retrieve(args)
    pass

if __name__ == "__main__":
    main()
    pass

