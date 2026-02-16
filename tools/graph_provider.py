"""
Graph provider abstraction for loading graphs from different sources.

Supports:
- DirectoryGraphProvider: load from .nx files in a directory (legacy)
- RNADatasetGraphProvider: load from rnaglib RNADataset, converting to vernal format on-the-fly
"""

import os
from abc import ABC, abstractmethod

import networkx as nx

from tools.graph_utils import read_nx_graph, fetch_graph


class GraphProvider(ABC):
    """Abstract interface for graph access by index or PDB ID."""

    @abstractmethod
    def get_graph_by_index(self, index: int) -> nx.Graph:
        """Return vernal-format graph at given index."""
        pass

    @abstractmethod
    def get_graph_by_name(self, name: str) -> nx.Graph:
        """Return vernal-format graph by name (e.g. pdbid or pdbid.nx)."""
        pass

    @abstractmethod
    def list_names(self):
        """Return list of graph names in order (matches index)."""
        pass

    def __len__(self):
        return len(self.list_names())


class DirectoryGraphProvider(GraphProvider):
    """Load graphs from .nx files in a directory."""

    def __init__(self, graph_dir: str):
        self.graph_dir = os.path.abspath(graph_dir)
        self._names = sorted(
            f for f in os.listdir(self.graph_dir)
            if f.endswith('.nx') or f.endswith('.p')
        )

    def get_graph_by_index(self, index: int) -> nx.Graph:
        name = self._names[index]
        path = os.path.join(self.graph_dir, name)
        return fetch_graph(path)

    def get_graph_by_name(self, name: str) -> nx.Graph:
        # Normalize: "1d0t" or "1d0t.A.1" -> "1d0t.nx"
        pdbid = str(name).split('.')[0].lower()[:4]
        for ext in ('.nx', '.p'):
            candidate = pdbid + ext
            if candidate in self._names:
                return self.get_graph_by_index(self._names.index(candidate))
        path = os.path.join(self.graph_dir, pdbid + '.nx')
        return fetch_graph(path)

    def list_names(self):
        return self._names


class RNADatasetGraphProvider(GraphProvider):
    """Load graphs from rnaglib RNADataset, converting to vernal format on-the-fly."""

    def __init__(self, dataset):
        """
        :param dataset: rnaglib.dataset.RNADataset instance
        """
        self.dataset = dataset
        self._rnaglib_to_vernal = None  # lazy import to avoid circular deps

    def _get_converter(self):
        if self._rnaglib_to_vernal is None:
            from prepare_data.rnaglib_loader import rnaglib_to_vernal_graph
            self._rnaglib_to_vernal = rnaglib_to_vernal_graph
        return self._rnaglib_to_vernal

    def _get_item(self, index: int):
        item = self.dataset[index]
        rna_name = self.dataset.all_rnas.inv[index]
        pdbid = rna_name.lower()
        return item['rna'], pdbid

    def get_graph_by_index(self, index: int) -> nx.Graph:
        g_rnaglib, _ = self._get_item(index)
        return self._get_converter()(g_rnaglib)

    def get_graph_by_name(self, name: str) -> nx.Graph:
        pdbid = str(name).split('.')[0].lower()[:4]
        # all_rnas maps name -> index; inv maps index -> name
        try:
            idx = self.dataset.all_rnas[pdbid]
        except (KeyError, TypeError):
            # Fallback: linear search
            for i in range(len(self.dataset)):
                rna_name = self.dataset.all_rnas.inv[i]
                if rna_name.lower() == pdbid:
                    return self.get_graph_by_index(i)
            raise KeyError(f"Graph not found: {name}")
        return self.get_graph_by_index(idx)

    def list_names(self):
        return [
            self.dataset.all_rnas.inv[i].lower()
            for i in range(len(self.dataset))
        ]
