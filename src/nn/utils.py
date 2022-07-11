"""Training Related Utilities"""
import types

import jax.numpy as jnp
import ml_collections
import numpy as np


def create_splits(
    train_nodes: np.ndarray, num_nodes: int, config: ml_collections.ConfigDict
) -> types.SimpleNamespace:
    """
    Split the output from the RSGNN Model

    :param train_nodes: Embeddings to split
    :type train_nodes: np.ndarray
    :param num_nodes: Total number of nodes
    :type num_nodes: int
    :param config: Configuration Dictionary
    :type config: ml_collections.ConfigDict
    :return: A Namespace with train, valid and test splits
    :rtype: types.SimpleNamespace
    """
    train_idx = np.array([False] * num_nodes)
    train_idx[train_nodes] = True
    valid_nodes = np.random.choice(
        np.where(np.logical_not(train_idx))[0], config.num_valid_nodes, replace=False
    )
    valid_idx = np.array([False] * num_nodes)
    valid_idx[valid_nodes] = True
    test_idx = np.logical_not(np.logical_or(train_idx, valid_idx))
    return types.SimpleNamespace(
        train=jnp.array(train_idx), valid=jnp.array(valid_idx), test=jnp.array(test_idx)
    )
