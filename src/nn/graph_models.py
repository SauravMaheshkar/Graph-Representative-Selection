"""Graphical Models"""

from typing import Sequence

import jax.numpy as jnp
import jraph
from flax import linen as nn

from src.nn import layers


class GCN(nn.Module):
    """A Flax Module for a Graph Convolutional Network"""

    features: Sequence[int]
    drop_rate: float
    activation: str

    @nn.compact
    def __call__(self, graph, train=True):
        for idx, latent_size in enumerate(self.features):
            # Create a Graph Convolution Method with no self edges,
            # as we took care of that while dataset preprocessing
            graph_conv = jraph.GraphConvolution(
                nn.Dense(latent_size), add_self_edges=False
            )
            graph = graph_conv(graph)
            # Pass intermediate features through the activation function
            act_fn = layers.Activation(self.activation)
            graph = graph._replace(nodes=act_fn(graph.nodes))
            if idx == len(self.features) - 1:
                # Return the features
                return graph.nodes
            # Perform dropout
            dout = nn.Dropout(rate=self.drop_rate)
            graph = graph._replace(nodes=dout(graph.nodes, deterministic=not train))


class DGI(nn.Module):
    """A Flax Module for a Deep Graph Infomax Module"""

    hid_dim: int
    dropout_rate: float = 0.5
    activation: str = "SeLU"

    @nn.compact
    def __call__(self, graph, c_graph, train=True):
        # Create a GCN Model and Bilinear Layer
        gcn = GCN([self.hid_dim], self.dropout_rate, self.activation)
        bilinear = layers.Bilinear()
        nodes1 = gcn(graph)
        nodes2 = gcn(c_graph)
        summary = layers.dgi_readout(nodes1)
        nodes = jnp.concatenate([nodes1, nodes2], axis=0)
        logits = bilinear(nodes, summary)
        return (nodes1, nodes2, summary), logits


class Cluster(nn.Module):
    """A Flax Module to find cluster centers given embeddings"""

    num_reps: int

    @nn.compact
    def __call__(self, embs):
        cluster = layers.EucCluster(self.num_reps)
        rep_ids, cluster_dists, centers = cluster(embs)
        loss = jnp.sum(cluster_dists)
        return centers, rep_ids, loss


class RSGNN(nn.Module):
    """Flax Module for a RS-GNN Model"""

    hid_dim: int
    num_reps: int
    dropout_rate: float = 0.5
    activation: str = "SeLU"

    def setup(self):
        self.dgi = DGI(self.hid_dim, self.dropout_rate, self.activation)
        self.cluster = Cluster(self.num_reps)

    def __call__(self, graph, c_graph, train=True):
        (embeddings, _, _), logits = self.dgi(graph, c_graph, train)
        embeddings = layers.normalize(embeddings)
        centers, rep_ids, cluster_loss = self.cluster(embeddings)
        return embeddings, centers, rep_ids, cluster_loss, logits
