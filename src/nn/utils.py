"""Training Related Utilities"""
import typing
from typing import Tuple

import jax
import jax.numpy as jnp
import jraph
import ml_collections
import numpy as np
from flax import linen as nn
from flax import optim
from flax.core import FrozenDict

from src.nn import graph_models


class BestKeeper:
    """Keeps best performance and model params during training."""

    def __init__(self, min_or_max: str) -> None:
        self.min_or_max = min_or_max
        self.best_result = np.inf if min_or_max == "min" else 0.0
        self.best_params = None

    def print_to_io(self, epoch: int, result: float) -> None:
        """
        Print results to IO

        :param epoch: Epoch number
        :type epoch: int
        :param result: Metric
        :type result: float
        """
        if self.min_or_max == "min":
            print("Epoch:", epoch, "Loss:", result)
        elif self.min_or_max == "max":
            print("Epoch:", epoch, "Accuracy:", result)

    def update(self, epoch: int, result: float, params, print_to_io=True):
        """Updates the best performance and model params if necessary."""
        if print_to_io:
            self.print_to_io(epoch, result)

        if self.min_or_max == "min" and result < self.best_result:
            self.best_result = result
            self.best_params = params
        elif self.min_or_max == "max" and result > self.best_result:
            self.best_result = result
            self.best_params = params

    def get(self):
        """Return best parameters"""
        return self.best_params


@typing.no_type_check
def create_model(
    config: ml_collections.ConfigDict,
    model_name: str,
    input: jraph.GraphsTuple,
    rng: jnp.ndarray,
) -> Tuple[nn.Module, FrozenDict, jnp.ndarray]:
    """create_model Create a Model Instance

    :param config: Configuration
    :type config: ml_collections.ConfigDict
    :param model_name: Which Model to use
    :type model_name: str
    :param input: Input
    :type input: jraph.GraphsTuple
    :param rng: Random Number Generator
    :return: Model, parameters and new rng
    """
    # Split Random Number Generator
    new_rng, init_rng, dropout_rng = jax.random.split(rng, num=3)
    if model_name == "gcn":
        # For Graph Convolutional Networks
        features = [config.hid_dim, config.num_classes]
        model = graph_models.GCN(features, config.dropout_rate, config.activation_fn)
        init = model.init({"params": init_rng, "dropout": dropout_rng}, input)
    elif model_name == "rsgnn":
        # For RS-GNN Networks
        model = graph_models.RSGNN(
            config.hid_dim, config.num_reps, config.dropout_rate, config.activation_fn
        )
        init = model.init({"params": init_rng, "dropout": dropout_rng}, input, input)
    return model, init, new_rng


def create_optimizer(
    config: ml_collections.ConfigDict, init_params, w_decay: float = 0.0
):
    """
    Create Optimizer

    :param config: Configuration
    :type config: ml_collections.ConfigDict
    :param init_params: Initialized parameters
    :param w_decay: weight_decay, defaults to 0.0
    :type w_decay: float, optional
    :return: Optimizer
    """
    optimizer = optim.Adam(learning_rate=config.learning_rate, weight_decay=w_decay)
    optimizer = optimizer.create(init_params)
    return jax.device_put(optimizer)
