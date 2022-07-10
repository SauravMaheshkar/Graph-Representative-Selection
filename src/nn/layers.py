"""Utility Layers for the Models"""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn


class PReLU(nn.Module):
    """A PReLU Flax Module"""

    init_fn: Callable[[Any], Any] = nn.initializers.uniform()

    @nn.compact
    def __call__(self, x):
        leakage = self.param("leakage", self.init_fn, [1])
        return jnp.maximum(0, x) + leakage * jnp.minimum(0, x)


class Activation(nn.Module):
    """Generic Flax Module for various activation functions"""

    activation: str

    def setup(self):
        if self.activation == "ReLU":
            self.act_fn = nn.relu
        elif self.activation == "SeLU":
            self.act_fn = jax.nn.selu
        elif self.activation == "PReLU":
            self.act_fn = PReLU()
        else:
            raise NameError(f"{self.activation} not recognized")

    def __call__(self, x):
        return self.act_fn(x)


class Bilinear(nn.Module):
    """A Bilinear Flax Module"""

    init_fn: Callable[[Any], Any] = nn.initializers.normal()

    @nn.compact
    def __call__(self, x_l, x_r):
        kernel = self.param("kernel", self.init_fn, [x_l.shape[-1], x_r.shape[-1]])
        return x_l @ kernel @ jnp.transpose(x_r)


class EucCluster(nn.Module):
    """Learnable KMeans Clustering Flax Module"""

    num_reps: int
    init_fn: Callable[[Any], Any] = nn.initializers.normal()

    @nn.compact
    def __call__(self, x):
        centers = self.param("centers", self.init_fn, [self.num_reps, x.shape[-1]])
        dists = jnp.sqrt(pairwise_sqeuc_dists(x, centers))
        return jnp.argmin(dists, axis=0), jnp.min(dists, axis=1), centers


@jax.jit
def dgi_readout(node_embs: jnp.ndarray) -> jnp.ndarray:
    """
    Readout function for a Deep Graph Infomax(DGI)

    :param node_embs: Node Embeddings
    :type node_embs: jnp.ndarray
    :return: 'Readout' Embeddings
    :rtype: jnp.ndarray
    """
    return jax.nn.sigmoid(jnp.mean(node_embs, axis=0))


def subtract_mean(embs: jnp.ndarray) -> jnp.ndarray:
    """
    Utility function to subtract the mean from the embedding array

    :param embs: Embeddings Array
    :type embs: jnp.ndarray
    :return: Mean Subtracted Embeddings Array
    :rtype: jnp.ndarray
    """
    return embs - jnp.mean(embs, axis=0)


def divide_by_l2_norm(embs: jnp.ndarray) -> jnp.ndarray:
    """
    Utility Function to divide the embedding array by the L2 norm

    :param embs: Embeddings Array
    :type embs: jnp.ndarray
    :return: Normalized Embedding Array
    :rtype: jnp.ndarray
    """
    norm = jnp.linalg.norm(embs, axis=1, keepdims=True)
    return embs / norm


@jax.jit
def normalize(node_embs: jnp.ndarray) -> jnp.ndarray:
    """
    Utility Function to normalize the Node Embedding Array

    :param node_embs: Node Embeddings Array
    :type node_embs: jnp.ndarray
    :return: Normalized Array
    :rtype: jnp.ndarray
    """
    return divide_by_l2_norm(subtract_mean(node_embs))


@jax.jit
def pairwise_sqeuc_dists(x, y) -> jnp.ndarray:  # pylint: disable=C0103
    """Calculate the pairwise sqeuc distance between the given arrays"""
    n = x.shape[0]  # pylint: disable=C0103
    m = y.shape[0]  # pylint: disable=C0103
    x_exp = jnp.expand_dims(x, axis=1).repeat(m, axis=1).reshape(n * m, -1)
    y_exp = jnp.expand_dims(y, axis=0).repeat(n, axis=0).reshape(n * m, -1)
    return jnp.sum(jnp.power(x_exp - y_exp, 2), axis=1).reshape(n, m)
