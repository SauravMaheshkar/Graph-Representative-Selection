"""Data preprocessing related tests"""
import jraph
import numpy as np

from src.io.utils import create_jraph


def test_cora_properties() -> None:
    """Test properties of cora"""
    graphs, labels, num_classes = create_jraph("data/", "cora")
    num_data_points = labels.shape[0]
    num_edges = np.asarray(graphs.n_edge)[0]
    num_nodes = np.asarray(graphs.n_node)[0]

    assert num_classes == 7
    assert num_nodes == 2708
    assert num_edges == 13264
    assert num_data_points == 2708
    assert labels.shape == (2708, 7)
    assert isinstance(graphs, jraph.GraphsTuple)
