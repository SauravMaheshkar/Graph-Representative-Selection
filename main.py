"""Demo Training Script"""
import jax
import jax.numpy as jnp
import numpy as np

from src.configs.rsgnn import get_config, get_rsgnn_flags
from src.io import utils
from src.nn.engine import train_gcn, train_rsgnn


def create_splits(train_nodes, num_nodes, config):
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


def get_gcn_c_flags(num_classes, config):
    config.hid_dim = config.gcn_c_hidden_dim
    config.epochs = config.gcn_c_epochs
    config.num_classes = num_classes
    return config


if __name__ == "__main__":
    config = get_config()
    np.random.seed(config.seed)
    key = jax.random.PRNGKey(config.seed)
    graph, labels, num_classes = utils.create_jraph(config.data_path, config.dataset)
    rsgnn_config = get_rsgnn_flags(num_classes, config)
    selected = train_rsgnn(rsgnn_config, graph, key)
    key, gcn_key = jax.random.split(key)
    splits = create_splits(selected, graph.n_node[0], config)
    gcn_c_config = get_gcn_c_flags(num_classes, config)
    gcn_accu = train_gcn(config, graph, labels, gcn_key, splits)
    print(f"GCN Test Accuracy: {gcn_accu}")
