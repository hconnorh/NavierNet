import os
import json
import torch
import numpy as np

from typing import Dict, Tuple, List, Optional, Callable
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent


class GraphSAGELayer(torch.nn.Module):
    """
    Minimal GraphSAGE-like layer implemented with vanilla PyTorch.

    This layer performs graph-based message passing, enabling each node to 
    aggregate information from its local neighborhood. For each node, the layer 
    computes the mean of its neighbors' features (mean aggregation), 
    concatenates this mean vector with the node's own feature vector 
    (self features), and then applies a linear transformation to this 
    concatenated vector. 

    The transformed features are passed through a ReLU activation. Dropout is 
    then applied during training to randomly zero out certain elements which
    helps prevent overfitting and improves model generalisation.

    The core message passing leverages `edge_index`, a tensor encoding the 
    directed graph's connectivity (source and target node pairs). A bidirectional 
    or undirected graph is recommended for effective aggregation of neighborhood 
    information. The output dimension is defined by the `out_features` argument. 
    This GraphSAGE layer forms the fundamental building block for deeper models 
    that use stacked message passing layers.
    """

    def __init__(self, in_features: int, out_features: int, dropout_p: float = 0.0) -> None:
        """
        Initialise a GraphSAGE layer with mean aggregation and linear 
        transformation.

        Args:
            in_features (int): Number of input features per node.
            out_features (int): Number of output features per node.
            dropout_p (float): Dropout probability after activation.
        """
        super().__init__()
        self.linear = torch.nn.Linear(in_features * 2, out_features)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Perform a single message-passing step of the GraphSAGE layers

        Args:
            node_features (torch.Tensor): Node features of shape [num_nodes, 
                                          in_features].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges], 
                                       where the first row contains source node 
                                       indices and the second row contains 
                                       destination node indices for each 
                                       directed edge.

        Returns:
            torch.Tensor: Updated node features of shape [num_nodes, 
                          out_features] for each node in the graph.

        Notes:
            - number of nodees is set by the number of nodes in the input 
              layer, i.e. the number of nodes in the mesh.
            - the "edge_index" tensor essentially dictates how the nodes are
              connected within the graph, and thus how information "flows"
              through the network.
        
        TODO: should play around with edge_indexing to determine impact on 
              model performance.
        """
        num_nodes = node_features.shape[0]
        in_features = node_features.shape[1]

        # edge_index: shape [2, E] with (src, dst)
        src = edge_index[0]
        dst = edge_index[1]

        # Sum neighbor features into destination nodes
        neighbor_sum = torch.zeros((num_nodes, in_features), device=node_features.device, dtype=node_features.dtype)
        neighbor_sum.index_add_(0, dst, node_features[src])

        # Degree for mean aggregation
        ones = torch.ones((dst.shape[0],), device=node_features.device, dtype=node_features.dtype)
        degree = torch.zeros((num_nodes,), device=node_features.device, dtype=node_features.dtype)
        degree.index_add_(0, dst, ones)
        degree = degree.clamp(min=1.0).unsqueeze(-1)
        neighbor_mean = neighbor_sum / degree

        # Combine self and neighbor representations
        combined = torch.cat([node_features, neighbor_mean], dim=1)
        h = self.linear(combined)
        h = self.activation(h)
        h = self.dropout(h)

        return h
