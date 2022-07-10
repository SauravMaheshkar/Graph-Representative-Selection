"""Training Scripts"""
import jax
import jax.numpy as jnp
import jraph
import ml_collections
import numpy as np

from src.nn.utils import BestKeeper, create_model, create_optimizer


def train_rsgnn(config: ml_collections.ConfigDict, graph: jraph.GraphsTuple, rng):
    """Trainer function for RS-GNN."""
    n_nodes = graph.n_node[0]
    labels = jnp.concatenate([jnp.ones(n_nodes), -jnp.ones(n_nodes)])
    model, init_params, rng = create_model(config, "rsgnn", graph, rng)
    optimizer = create_optimizer(config, init_params)

    @jax.jit
    def corrupt_graph(rng):
        return graph._replace(nodes=jax.random.permutation(rng, graph.nodes))

    @jax.jit
    def train_step(optimizer, graph, c_graph, drop_rng):
        def loss_fn(params):
            _, _, _, cluster_loss, logits = model.apply(
                params, graph, c_graph, rngs={"dropout": drop_rng}
            )
            dgi_loss = -jnp.sum(jax.nn.log_sigmoid(labels * logits))
            return dgi_loss + config.lambda_value * cluster_loss

        loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        return optimizer.apply_gradient(grad), loss

    best_keeper = BestKeeper("min")
    for epoch in range(1, config.epochs + 1):
        rng, drop_rng, corrupt_rng = jax.random.split(rng, num=3)
        c_graph = corrupt_graph(corrupt_rng)
        optimizer, loss = train_step(optimizer, graph, c_graph, drop_rng)
        if epoch % config.valid_epochs == 0:
            best_keeper.update(epoch, loss, optimizer.target)

    _, _, rep_ids, _, _ = model.apply(best_keeper.get(), graph, c_graph, train=False)

    return np.array(rep_ids)


def train_gcn(config: ml_collections.ConfigDict, graph, labels, rng, splits):
    """Trainer function for a classification GCN."""
    model, init_params, rng = create_model(config, "gcn", graph, rng)
    optimizer = create_optimizer(config, init_params, config.w_decay)

    @jax.jit
    def train_step(optimizer, drop_rng):
        def loss_fn(params):
            logits = model.apply(params, graph, rngs={"dropout": drop_rng})
            log_prob = jax.nn.log_softmax(logits)
            return -jnp.sum(log_prob[splits.train] * labels[splits.train])

        loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
        return optimizer.apply_gradient(grad), loss

    @jax.jit
    def accuracy(params, mask):
        logits = model.apply(params, graph, train=False)
        correct = jnp.argmax(logits, -1) == jnp.argmax(labels, -1)
        return jnp.sum(correct * mask) / jnp.sum(mask)

    best_keeper = BestKeeper("max")
    for epoch in range(1, config.epochs + 1):
        rng, drop_rng = jax.random.split(rng)
        optimizer, _ = train_step(optimizer, drop_rng)
        if epoch % config.valid_epochs == 0:
            accu = accuracy(optimizer.target, splits.valid)
            best_keeper.update(epoch, accu, optimizer.target)

    accu = accuracy(best_keeper.get(), splits.test)
    return float(accu)
