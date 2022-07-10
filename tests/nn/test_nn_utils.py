"""Utility related tests"""
import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn
from flax.core import FrozenDict

from src.configs.rsgnn import get_config, get_rsgnn_flags
from src.io import utils
from src.nn.utils import create_model


@pytest.mark.parametrize(
    ("model_name"),
    (
        ("gcn"),
        ("rsgnn"),
    ),
)
def test_model_creation(model_name: str) -> None:
    """
    Test Model Creation Function

    :param model_name: Model Type
    :type model_name: str
    """
    config = get_config()
    key = jax.random.PRNGKey(config.seed)
    graph, _, num_classes = utils.create_jraph(config.data_path, config.dataset)
    rsgnn_config = get_rsgnn_flags(num_classes, config)
    model, init, new_rng = create_model(
        config=rsgnn_config, model_name=model_name, input=graph, rng=key
    )

    assert isinstance(model, nn.Module)
    assert isinstance(init, FrozenDict)
    assert isinstance(new_rng, jnp.ndarray)
