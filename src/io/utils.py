"""Dataset processing utilites"""
from typing import Set, Tuple

import jax.numpy as jnp
import jraph
import numpy as np
from scipy.sparse import csr_matrix


def onehot(labels: np.ndarray) -> jnp.ndarray:
    """
    One Hot Encode the Graph Labels

    :param labels: Numpy Array of Graph Labels
    :type labels: np.ndarray
    :return: One-Hot Encoded Graph Labels
    :rtype: jnp.ndarray
    """
    classes = set(labels)
    return jnp.identity(len(classes))[jnp.array(labels)]


def load_from_npz(
    path: str = "data/", dataset: str = "cora"
) -> Tuple[np.matrix, np.matrix, jnp.ndarray]:
    """
    Load a Dataset from .npz files

    :param path: Path to the data dir, defaults to "data/"
    :type path: str, optional
    :param dataset: Name of the dataset, defaults to "cora"
    :type dataset: str, optional
    :raises Exception: If No attributes are found in the data file
    :raises Exception: If No labels are found in the data file
    :return: Adjacency Matrix, Attribute Matrix and One-Hot Encoded Labels
    :rtype: Tuple[np.matrix, np.matrix, jnp.ndarray]
    """

    file_name: str = path + dataset + ".npz"
    with np.load(open(file_name, "rb"), allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix: np.matrix = csr_matrix(
            (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
            shape=loader["adj_shape"],
        )

        if "attr_data" in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix: np.matrix = csr_matrix(
                (loader["attr_data"], loader["attr_indices"], loader["attr_indptr"]),
                shape=loader["attr_shape"],
            ).todense()
        elif "attr_matrix" in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader["attr_matrix"]
        else:
            raise Exception("No attributes in the data file", file_name)

        if "labels_data" in loader:
            # Labels are stored as a CSR matrix
            labels = csr_matrix(
                (
                    loader["labels_data"],
                    loader["labels_indices"],
                    loader["labels_indptr"],
                ),
                shape=loader["labels_shape"],
            )
            labels = labels.nonzero()[1]
        elif "labels" in loader:
            # Labels are stored as a numpy array
            labels = loader["labels"]
        else:
            raise Exception("No labels in the data file", file_name)

    return adj_matrix, attr_matrix, onehot(labels)


def symmetrize(edges: Set) -> Set:
    """Symmetrizes the adjacency matrix"""
    inv_edges = {(d, s) for s, d in edges}
    return edges.union(inv_edges)


def add_self_loop(edges: Set, n_node: int) -> Set:
    """Adds a self loop to the edge matrix"""
    self_loop_edges = {(s, s) for s in range(n_node)}
    return edges.union(self_loop_edges)


def get_graph_edges(adj: np.matrix, features: np.ndarray) -> Tuple[Set, int]:
    """
    Symmetrizes and adds Self Loop to the Edge Matrix

    :param adj: Adjaceny Matrix
    :type adj: np.matrix
    :param features: Attribute Matrix
    :type features: np.ndarray
    :return: Attribute Matrix and number of attributes
    :rtype: Tuple[Set, int]
    """
    rows = adj.tocoo().row  # type: ignore
    cols = adj.tocoo().col  # type: ignore
    edges = {(row, col) for row, col in zip(rows, cols)}
    edges = symmetrize(edges)
    edges = add_self_loop(edges, features.shape[0])
    return edges, len(edges)


def create_jraph(
    data_path: str = "data/", dataset: str = "cora"
) -> Tuple[jraph.GraphsTuple, np.ndarray, int]:
    """
    Create GraphTuple for datasets created from .npz files

    :param data_path: Path to the data dir, defaults to "data/"
    :type data_path: str, optional
    :param dataset: Name of the dataset, defaults to "cora"
    :type dataset: str, optional
    :return: GraphTuple, Labels and Number of Classes
    :rtype: Tuple[jraph.GraphsTuple, np.ndarray, int]
    """
    adj, features, labels = load_from_npz(data_path, dataset)
    edges, n_edge = get_graph_edges(adj, np.array(features))
    n_node = len(features)
    features = jnp.asarray(features)
    graph = jraph.GraphsTuple(
        n_node=jnp.asarray([n_node]),
        n_edge=jnp.asarray([n_edge]),
        nodes=features,
        edges=None,
        globals=None,
        senders=jnp.asarray([edge[0] for edge in edges]),
        receivers=jnp.asarray([edge[1] for edge in edges]),
    )

    return graph, np.asarray(labels), labels.shape[1]
