import os
import sys
import pickle

from tqdm import tqdm
import networkx as nx
import dgl
import numpy as np
import torch


script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from torch.utils.data import Dataset, DataLoader, Subset
from tools.node_sim import k_block_list, simfunc_from_hparams, EDGE_MAP
from tools.graph_utils import fetch_graph


class V1(Dataset):
    def __init__(self,
                 edge_map,
                 node_simfunc,
                 annotated_path='../data/annotated/samples',
                 depth=3,
                 debug=False,
                 shuffled=False,
                 ):

        self.path = annotated_path
        # self.all_graphs = np.array(sorted(os.listdir(annotated_path)), dtype=np.string_)
        self.all_graphs = sorted(os.listdir(annotated_path))

        self.node_simfunc = node_simfunc

        if not node_simfunc is None:
            if self.node_simfunc.method in ['R_graphlets', 'graphlet']:
                self.level = 'graphlet'
            else:
                self.level = 'edge'
            self.depth = self.node_simfunc.depth
        else:
            self.level = None
            self.depth = None

        self.edge_map = edge_map
        # This is len() so we have to add the +1
        self.num_edge_types = max(self.edge_map.values()) + 1
        print(f"Found {self.num_edge_types} relations")

    def __len__(self):
        return len(self.all_graphs)

    def __getitem__(self, idx):
        g_path = os.path.join(self.path, self.all_graphs[idx])
        if g_path.endswith('.p'):
            data = pickle.load(open(g_path, 'rb'))
            graph = data['graph']
        else:
            graph = nx.read_gpickle(g_path)
        graph = nx.to_undirected(graph)
        one_hot = {edge: torch.tensor(self.edge_map[label]) for edge, label in
                   (nx.get_edge_attributes(graph, 'label')).items()}
        nx.set_edge_attributes(graph, name='one_hot', values=one_hot)

        g_dgl = dgl.DGLGraph()
        g_dgl.from_networkx(nx_graph=graph, edge_attrs=['one_hot'])

        if self.node_simfunc is not None:
            ring = data['rings'][self.level]
            return g_dgl, ring, [idx]
        else:
            return g_dgl, 0, [idx]


def collate_wrapper(node_simfunc):
    """
        Wrapper for collate function so we can use different node similarities.
    """
    if node_simfunc is not None:
        def collate_block(samples):
            # The input `samples` is a list of tuples
            #  (graph, ring, label).
            graphs, rings, idx = map(list, zip(*samples))
            batched_graph = dgl.batch(graphs)
            K = k_block_list(rings, node_simfunc)
            idx = np.array(idx)
            len_graphs = [len(graph) for graph in graphs]
            return batched_graph, torch.from_numpy(K).detach().float(), torch.from_numpy(idx), len_graphs
    else:
        def collate_block(samples):
            # The input `samples` is a list of pairs
            #  (graph, label).
            graphs, _, idx = map(list, zip(*samples))
            batched_graph = dgl.batch(graphs)
            idx = np.array(idx)
            len_graphs = [len(graph) for graph in graphs]
            return batched_graph, [1 for _ in samples], torch.from_numpy(idx), len_graphs
    return collate_block


class Loader():
    def __init__(self,
                 annotated_path='data/annotated/samples/',
                 batch_size=5,
                 num_workers=20,
                 debug=False,
                 shuffled=False,
                 edge_map=EDGE_MAP,
                 node_simfunc=None):
        """

        :param annotated_path:
        :param batch_size:
        :param num_workers:
        :param debug:
        :param shuffled:
        :param node_simfunc: The node comparison object to use for the embeddings. If None is selected,
        will just return graphs
        :param hparams:
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = V1(annotated_path=annotated_path,
                          debug=debug,
                          shuffled=shuffled,
                          node_simfunc=node_simfunc,
                          edge_map=edge_map)

        self.node_simfunc = node_simfunc
        self.num_edge_types = self.dataset.num_edge_types

    def get_data(self):
        n = len(self.dataset)
        indices = list(range(n))
        # np.random.shuffle(indices)

        np.random.seed(0)
        split_train, split_valid = 0.7, 0.7
        train_index, valid_index = int(split_train * n), int(split_valid * n)

        train_indices = indices[:train_index]
        valid_indices = indices[train_index:valid_index]
        test_indices = indices[valid_index:]

        train_set = Subset(self.dataset, train_indices)
        valid_set = Subset(self.dataset, valid_indices)
        test_set = Subset(self.dataset, test_indices)
        all_set = Subset(self.dataset, indices)

        print(f"training items: ", len(train_set))

        collate_block = collate_wrapper(self.node_simfunc)

        train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                  num_workers=self.num_workers, collate_fn=collate_block)
        # valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=self.batch_size,
        #                           num_workers=self.num_workers, collate_fn=collate_block)
        test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=self.batch_size,
                                 num_workers=self.num_workers, collate_fn=collate_block)
        all_loader = DataLoader(dataset=all_set, shuffle=True, batch_size=self.batch_size,
                                num_workers=self.num_workers, collate_fn=collate_block)

        return train_loader, test_loader, all_loader


class InferenceLoader(Loader):
    def __init__(self,
                 list_to_predict,
                 annotated_path,
                 batch_size=5,
                 num_workers=20,
                 edge_map=EDGE_MAP):
        super().__init__(
            annotated_path=annotated_path,
            batch_size=batch_size,
            num_workers=num_workers,
            edge_map=edge_map
        )
        self.dataset.all_graphs = list_to_predict
        self.dataset.path = annotated_path
        print(len(list_to_predict))

    def get_data(self):
        collate_block = collate_wrapper(None)
        train_loader = DataLoader(dataset=self.dataset,
                                  shuffle=False,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  collate_fn=collate_block)
        return train_loader


def loader_from_hparams(annotated_path, hparams, list_inference=None):
    """
        :params
        :get_sim_mat: switches off computation of rings and K matrix for faster loading.
    """
    if list_inference is None:
        node_simfunc = simfunc_from_hparams(hparams)
        loader = Loader(annotated_path=annotated_path,
                        batch_size=hparams.get('argparse', 'batch_size'),
                        num_workers=hparams.get('argparse', 'workers'),
                        edge_map=hparams.get('edges', 'edge_map'),
                        node_simfunc=node_simfunc)
        return loader

    loader = InferenceLoader(list_to_predict=list_inference,
                             annotated_path=annotated_path,
                             batch_size=hparams.get('argparse', 'batch_size'),
                             num_workers=hparams.get('argparse', 'workers'),
                             edge_map=hparams.get('edges', 'edge_map'))
    return loader
