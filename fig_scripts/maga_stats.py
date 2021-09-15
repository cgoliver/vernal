import os
import sys
import pickle
from collections import defaultdict

import pandas as pd

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from build_motifs.meta_graph import *

def get_stats(mgg):
    num_nodes = len(mgg.maga_graph.nodes())

    n_instances = defaultdict(list)

    for node, data in mgg.maga_graph.nodes(data=True):
        try:
            n_instances[len(node)].append(len(data['node_set']))
        except KeyError:
            print(f">>> MISSING NODESET {node}")
            continue
    n_motifs = {k:len(v) for k,v in n_instances.items()}
    print(n_motifs)
    n_instances = {k:(np.mean(v), np.std(v)) for k,v in n_instances.items()}
    tot = sum(n_motifs.values())
    data = []
    for level in sorted(n_motifs.keys()):
        data.append({
                     'motif_size': level,
                     'n_motifs': n_motifs[level],
                     'n_instances': n_instances[level][0]}
                    )
    df = pd.DataFrame(data)
    print(df.to_latex(index=False, float_format="%.2f"))
    pass

def make(maga_path):
    mgg = pickle.load(open(maga_path, 'rb'))
    get_stats(mgg)


if __name__ == "__main__":
    make("../results/magas/default_name.p")
    pass
