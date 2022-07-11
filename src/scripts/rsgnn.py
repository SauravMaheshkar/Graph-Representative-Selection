"""Training Script for RSGNN"""
import types
from typing import Tuple

import jax
import jax.numpy as jnp
import jraph
import ml_collections
import numpy as np
import optax
from flax.training import train_state

from src.configs.rsgnn import get_config, get_gcn_c_flags, get_rsgnn_flags
from src.io.utils import corrupt_graph, create_jraph
from src.nn.graph_models import GCN, RSGNN
from src.nn.utils import create_splits


def init_rsgnn_train_state(
    config: ml_collections.ConfigDict,
    random_key,
    graph_array: jraph.GraphsTuple,
) -> train_state.TrainState:
    """
    Initialize TrainState for the RSGNN Model

    :param config: Configuration Dictionary
    :type config: ml_collections.ConfigDict
    :param random_key: Random Key
    :param graph_array: Graph Data
    :type graph_array: jraph.GraphsTuple
    :return: Flax Training State
    :rtype: train_state.TrainState
    """
    # Split Random Number Generator
    initialization_rng, dropout_rng = jax.random.split(random_key, num=2)

    # Create a Model Instance
    model = RSGNN(
        hid_dim=config.hid_dim,
        num_reps=config.num_reps,
        dropout_rate=config.dropout_rate,
        activation=config.activation_fn,
    )

    # Initialize the Model
    variables = model.init(
        {"params": initialization_rng, "dropout": dropout_rng}, graph_array, graph_array
    )

    # Create a optimizer
    optimizer = optax.adam(learning_rate=config.learning_rate)

    # Create a TrainState
    return train_state.TrainState.create(
        apply_fn=model.apply, tx=optimizer, params=variables["params"]
    )


def init_gcn_train_state(
    config: ml_collections.ConfigDict,
    random_key,
    graph_array: jraph.GraphsTuple,
) -> train_state.TrainState:
    """
    Initialize TrainState for the GCN Model

    :param config: Configuration Dictionary
    :type config: ml_collections.ConfigDict
    :param random_key: Random Key
    :param graph_array: Graph Data
    :type graph_array: jraph.GraphsTuple
    :return: Flax Training State
    :rtype: train_state.TrainState
    """
    # Split Random Number Generator
    initialization_rng, dropout_rng = jax.random.split(random_key, num=2)

    features = [config.hid_dim, config.num_classes]

    # Create a Model Instance
    model = GCN(
        features=features,
        dropout_rate=config.dropout_rate,
        activation=config.activation_fn,
    )

    # Initialize the Model
    variables = model.init(
        {"params": initialization_rng, "dropout": dropout_rng}, graph_array
    )

    # Create a optimizer
    optimizer = optax.adam(learning_rate=config.learning_rate)

    # Create a TrainState
    return train_state.TrainState.create(
        apply_fn=model.apply, tx=optimizer, params=variables["params"]
    )


def rsgnn_train_step(
    state: train_state.TrainState,
    config: ml_collections.ConfigDict,
    graph: jraph.GraphsTuple,
    corrupted_graph: jraph.GraphsTuple,
    labels: jnp.ndarray,
    dropout_rng,
) -> Tuple[train_state.TrainState, jnp.ndarray]:
    """
    Training Step for the RSGNN Model

    :param state: Flax Training State of the Model
    :type state: train_state.TrainState
    :param config: Configuration Dictionary
    :type config: ml_collections.ConfigDict
    :param graph: Training Graph Data
    :type graph: jraph.GraphsTuple
    :param corrupted_graph: Corrupted Graph Data
    :type corrupted_graph: jraph.GraphsTuple
    :param labels: labels
    :type labels: jnp.ndarray
    :param dropout_rng: Dropout Random Key
    :return: Updated TrainState and loss after a step
    :rtype: Tuple[train_state.TrainState, jnp.ndarray]
    """

    # Loss Function for the RSGNN Model
    def loss_fn(params):
        _, _, _, cluster_loss, logits = state.apply_fn(
            {"params": params}, graph, corrupted_graph, rngs={"dropout": dropout_rng}
        )
        dgi_loss = -jnp.sum(jax.nn.log_sigmoid(labels * logits))
        return dgi_loss + config.lambda_value * cluster_loss

    # Create and Apply Gradient Function
    gradient_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, gradient = gradient_fn(state.params)

    # Update parameters
    state = state.apply_gradients(grads=gradient)
    return state, loss


def gcn_train_step(
    state: train_state.TrainState,
    graph: jraph.GraphsTuple,
    labels: np.ndarray,
    dropout_rng,
    splits: types.SimpleNamespace,
) -> Tuple[train_state.TrainState, jnp.ndarray]:
    """
    Training Step for the GCN Model

    :param state: Flax Training State for the Model
    :type state: train_state.TrainState
    :param config: Configuration Dictionary
    :type config: ml_collections.ConfigDict
    :param graph: Training Graph Data
    :type graph: jraph.GraphsTuple
    :param labels: labels
    :type labels: jnp.ndarray
    :param dropout_rng: Dropout Random Key
    :param splits: Embeddings split into train and validation
    :type splits: types.SimpleNamespace
    :return: Updated TrainState and loss after a step
    :rtype: Tuple[train_state.TrainState, jnp.ndarray]
    """

    # Loss Function for the GCN Model
    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params}, graph, rngs={"dropout": dropout_rng}
        )
        log_probabilites = jax.nn.log_softmax(logits)
        return -jnp.sum(log_probabilites[splits.train] * labels[splits.train])

    # Create and Apply Gradient Function
    gradient_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, gradient = gradient_fn(state.params)

    # Update parameters
    state = state.apply_gradients(grads=gradient)
    return state, loss


def rsgnn_train(
    state: train_state.TrainState,
    config: ml_collections.ConfigDict,
    graph: jraph.GraphsTuple,
    rng,
) -> np.ndarray:
    """
    Training Code for the RSGNN Model

    :param state: Flax Training State for the Model
    :type state: train_state.TrainState
    :param config: Configuration Dictionary
    :type config: ml_collections.ConfigDict
    :param graph: Training Graph Data
    :type graph: jraph.GraphsTuple
    :param rng: Random Key
    :return: Resulting Embeddings from the Trained Model
    :rtype: np.ndarray
    """

    # Extract labels from the data
    num_nodes = graph.n_node[0]
    labels = jnp.concatenate([jnp.ones(num_nodes), -jnp.ones(num_nodes)])

    for epoch in range(1, config.rsgnn_epochs + 1):
        # Split Random Number Generator
        rng, dropout_rng, corrupt_rng = jax.random.split(rng, num=3)

        # Corrupt the Training Graph
        corrupted_graph = corrupt_graph(graph, corrupt_rng)

        # Perform a Train Step
        state, loss = rsgnn_train_step(
            state, config, graph, corrupted_graph, labels, dropout_rng
        )

        # Log Metrics
        if epoch % config.log_freq == 0:
            print(f"Training Epoch {epoch} | Loss {loss}")

    # Extract Embeddings
    _, _, rep_ids, _, _ = state.apply_fn(
        {"params": state.params}, graph, corrupted_graph, train=False
    )

    return np.array(rep_ids)


def gcn_train(
    state: train_state.TrainState,
    config: ml_collections.ConfigDict,
    graph: jraph.GraphsTuple,
    labels: np.ndarray,
    splits: types.SimpleNamespace,
    rng,
) -> float:
    """
    Training Code for the GCN Model

    :param state: Flax Training State for the Model
    :type state: train_state.TrainState
    :param config: Configuration Dictionary
    :type config: ml_collections.ConfigDict
    :param graph: Training Graph Data
    :type graph: jraph.GraphsTuple
    :param labels: labels
    :type labels: jnp.ndarray
    :param splits: Embeddings split into train and validation
    :type splits: types.SimpleNamespace
    :param rng: Random Key
    :return: Accuracy
    :rtype: np.ndarray
    """

    @jax.jit
    def accuracy(params, mask):
        logits = state.apply_fn({"params": params}, graph, train=False)
        correct = jnp.argmax(logits, -1) == jnp.argmax(labels, -1)
        return jnp.sum(correct * mask) / jnp.sum(mask)

    for epoch in range(1, config.epochs + 1):
        # Split Random Number Generator
        rng, dropout_rng = jax.random.split(rng, num=2)

        # Perform a Train Step
        state, loss = gcn_train_step(state, graph, labels, dropout_rng, splits)

        # Log Metrics
        if epoch % config.log_freq == 0:
            print(f"Training Epoch {epoch} | Loss {loss}")

    # Calculate Accuracy
    acc = accuracy(state.params, splits.test)
    return float(acc)


if __name__ == "__main__":
    config = get_config()
    np.random.seed(config.seed)
    key = jax.random.PRNGKey(config.seed)
    graph, labels, num_classes = create_jraph(config.data_path, config.dataset)

    # RSGNN Training
    rsgnn_config = get_rsgnn_flags(num_classes, config)
    state = init_rsgnn_train_state(rsgnn_config, key, graph)
    selected = rsgnn_train(state, config, graph, key)

    # GCN Training
    key, gcn_key = jax.random.split(key)
    splits = create_splits(selected, graph.n_node[0], config)
    gcn_config = get_gcn_c_flags(num_classes, config)
    gcn_state = init_gcn_train_state(gcn_config, gcn_key, graph)
    print("GCN Training !!")
    gcn_accuracy = gcn_train(gcn_state, config, graph, labels, splits, gcn_key)
    print(f"GCN Test Accuracy: {gcn_accuracy}")
