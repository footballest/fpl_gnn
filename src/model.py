"""
model.py
--------

Definition of the Graph Neural Network used to predict FPL player points.

The model is built upon the **Graph Attention Network (GAT)** architecture
introduced by Veličković et al. (2018).  PyTorch Geometric provides
the `GATConv` layer which implements this attention mechanism【127110137942256†L474-L490】.  We
stack two GAT layers: the first layer projects the input node
features to a hidden dimension using multiple attention heads, and
the second layer outputs a single scalar per node representing the
predicted FPL points.

Optionally, the class exposes a method to retrieve attention
coefficients for interpretability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class FPLGNN(nn.Module):
    """Graph Attention Network for FPL score prediction.

    Parameters
    ----------
    in_features : int
        Dimension of input node feature vectors.
    hidden_features : int
        Hidden feature size for the first GAT layer.
    heads : int, default 4
        Number of attention heads in the first GAT layer.  The final
        layer uses a single head.

    Notes
    -----
    The first GAT layer concatenates the outputs of multiple heads,
    resulting in a hidden dimension of ``hidden_features * heads`` per
    node.  The second GAT layer then reduces this to a single scalar.
    """

    def __init__(self, in_features: int, hidden_features: int, heads: int = 4) -> None:
        super().__init__()
        self.gat1 = GATConv(
            in_channels=in_features,
            out_channels=hidden_features,
            heads=heads,
            concat=True,
        )
        # Final layer: outputs scalar per node; concat=False means outputs are averaged across heads
        self.gat2 = GATConv(
            in_channels=hidden_features * heads,
            out_channels=1,
            heads=1,
            concat=False,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, in_features].
        edge_index : torch.Tensor
            Edge index tensor of shape [2, num_edges].

        Returns
        -------
        torch.Tensor
            Predictions of shape [num_nodes, 1].  Flatten if needed.
        """
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        return x


    def reset_parameters(self) -> None:
        """Reset the parameters of the GAT layers."""
        self.gat1.reset_parameters()
        self.gat2.reset_parameters()
