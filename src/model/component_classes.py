"""
GAT-LSTM Component Subclasses:
    - GAT (spatial) Encoder: Returns h_gat of shape (N_nodes, d_g)
    - Temporal Encoder (LSTM): Returns h_temp of shape (N_nodes, d_h) (last step), and updated (h_new, c_new).
    - Node Conditioner (FiLM): Returns node-specific temporal embeddings (gamma, beta) of shape (N_nodes, d_h) each.
    - Temporal Projection: Returns h_temp_proj (N_nodes, d_g), temporal encodings mapped to GAT output dim.
    - Fusion Gate (Gated Residual): Returns alpha (N_nodes, 1) in (0,1), choosing GAT and temporal weightings.
    - Regression Head: Returns ŷ (N_nodes, 1) by mapping fused embeddings to groundwater level.
"""

import sys
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)

class GATEncoder(nn.Module):
    """
    GAT encoder including exdge index and attributes. Returns final spatial embedding
    per node.
    """
    def __init__(self, in_dim: int, hid: int, out: int, heads: int, dropout: float,
                 num_layers: int, edge_dim: int):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.edge_dim = edge_dim

        # If one GAT then simple run with no attention based concat or multi head mechanisms
        if num_layers <= 1:
            self.layers.append(GATConv(in_dim, out, heads=1, dropout=dropout,
                                       add_self_loops=True, concat=False, edge_dim=edge_dim))
        
        # If multiple layers then run full architecturec with concat and heads
        else:
            # First layer: in_dim -> hid * heads (concat'd)
            self.layers.append(GATConv(in_dim, hid, heads=heads, dropout=dropout,
                                       add_self_loops=True, concat=True, edge_dim=edge_dim))
            
            # Intermediate layer(s) (hid * heads -> hid * heads)
            for _ in range(num_layers - 2):
                self.layers.append(GATConv(hid * heads, hid, heads=heads, dropout=dropout,
                                           add_self_loops=True, concat=True, edge_dim=edge_dim))
            
            # Final layer: hid * heads -> out (no concat, single head, to final dim)
            self.layers.append(GATConv(hid * heads, out, heads=1, dropout=dropout,
                                       add_self_loops=True, concat=False, edge_dim=edge_dim))

    def forward(self, x, edge_index, edge_attr=None):
        h = x
        for i, layer in enumerate(self.layers):
            h = F.dropout(h, p=self.dropout, training=self.training)  # All apply dropout
            h = layer(h, edge_index, edge_attr)
            if i < len(self.layers) - 1:
                h = F.elu(h)  # ELU in all intermediate layer transitions
        return h

class TemporalEncoder(nn.Module):
    """
    Shared LSTM temporal encoder. Returns h_temp, the last temporal embedding in
    the sequence, of shape (N_nodes, d_h), and the updated hidden and cell states
    (h_new, c_new).
    """
    def __init__(self, d_t: int, d_h: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_t, hidden_size=d_h, num_layers=num_layers,
            dropout=dropout, batch_first=True
        )

    def forward(self, x_seq, h_c=None):
        out, (h_new, c_new) = self.lstm(x_seq, h_c)
        h_last = out[:, -1, :]  # get last step along seq_len
        return h_last, (h_new, c_new)

class NodeConditioner(nn.Module):
    """
    FiLM conditioning: maps node static features to per-feature scale and shift.
    """
    def __init__(self, d_s: int, d_h: int):
        super().__init__()
        self.fc_gamma = nn.Linear(d_s, d_h)
        self.fc_beta = nn.Linear(d_s, d_h)

    def forward(self, x_static):
        gamma = self.fc_gamma(x_static)  # multiplicative (scaler)
        beta = self.fc_beta(x_static)  # Additive (bias)
        return gamma, beta  # Applied as h_temp <- gamma ⊙ h_temp + beta

class TemporalProjection(nn.Module):
    """
    Temporal head linearly projects temporal embedding to the GAT embedding
    size for fusion (defensive, should match)

    Weights and bias are zero-initialised so that at init the temporal path
    contributes zero (safe).
    """
    def __init__(self, d_h: int, d_g: int):
        super().__init__()
        self.proj = nn.Linear(d_h, d_g)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h_temp):
        return self.proj(h_temp)  # (N, d_g)

class FusionGate(nn.Module):
    """
    Produces a scalar gate alpha in (0,1) deciding the mix of GAT vs temporal.

    Args:
        in_dim (int):   Input dimension to the gate MLP; we use [h_gat || h_temp_proj],
                        so in_dim = 2 * d_g.
        bias_init (float): Bias initialisation for the last linear layer. Setting
                        bias_init=+5 makes \sigma(bias)≈0.993, so \alpha≈0.993 initially,
                        i.e. ~99.3% GAT and ~0.7% temporal (pro-GAT prior).

    Input:
        z: (N, in_dim) = concatenated embeddings

    Output:
        alpha: (N, 1) with values in (0, 1)
    """
    def __init__(self, in_dim: int, bias_init: float = 0.0, alpha_floor: float = 0.15):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
        nn.init.zeros_(self.fc.weight)  # start with zero slope
        nn.init.constant_(self.fc.bias, bias_init)  # prior set in config
        self.alpha_floor = alpha_floor  # guarantees GAT contribution to stop initial collapse

    def forward(self, z):
        a = torch.sigmoid(self.fc(z))
        if self.alpha_floor > 0:
            a = self.alpha_floor + (1.0 - self.alpha_floor) * a
        return a

class RegressorHead(nn.Module):
    """
    Final regression head.

    Args:
        in_dim (int):  Input embedding size to the head (typically d_g when GAT
                       is enabled; otherwise d_s + d_t or d_s + d_h).
        out_dim (int): Output dimension. For groundwater level, this is 1.

    Input:
        h: (N, in_dim)

    Output:
        y_hat: (N, out_dim)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, h):
        return self.linear(h)