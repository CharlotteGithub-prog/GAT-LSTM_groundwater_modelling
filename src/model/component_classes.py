"""
GAT-LSTM Component Subclasses:
    - Temporal Encoder (LSTM): Returns h_temp of shape (N_nodes, d_h) (last step), and updated (h_new, c_new).
    - Node Conditioner (FiLM): Returns node-specific temporal embeddings (gamma, beta) of shape (N_nodes, d_h) each.
    - GAT (spatial) Encoder: Returns h_gat of shape (N_nodes, d_g)
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

class TemporalEncoder(nn.Module):
    """
    Shared LSTM temporal encoder. Returns h_temp of shape (N_nodes, d_h) (last step),
    and updated (h_new, c_new).

    Args:
        d_t (int):  Number of temporal features per day. This is the width of the
                    daily catchment-aggregated vector (e.g., rainfall, AET, temp,
                    pressure, seasonal encodings, etc.).
        d_h (int):  LSTM hidden size (per layer). Also the **temporal embedding**
                    dimension output by this encoder before projection to d_g.
        num_layers (int): Number of stacked LSTM layers.

    Input:
        x_seq: Tensor of shape (N, seq_len, d_t)
            N = number of nodes for the current timestep
            seq_len = length of temporal window fed to the LSTM (you currently
                      use seq_len=1 with state carry; can be >1 with TBPTT)
        h_c (tuple or None): Optional initial LSTM states:
            h0: (num_layers, N, d_h)
            c0: (num_layers, N, d_h)

    Output:
        h_last: (N, d_h)  Last output along seq_len axis (temporal embedding)
        (h_new, c_new): Updated LSTM states with shapes as above.
    """
    def __init__(self, d_t: int, d_h: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=d_t, hidden_size=d_h, num_layers=num_layers, batch_first=True)

    def forward(self, x_seq, h_c=None):
        out, (h_new, c_new) = self.lstm(x_seq, h_c)  # out: (N, seq_len, d_h)
        h_last = out[:, -1, :]                       # take last step along seq_len
        return h_last, (h_new, c_new)

class NodeConditioner(nn.Module):
    """
    FiLM conditioning: maps node static features to per-feature scale and shift.

    Args:
        d_s (int):  Number of **static** features per node (e.g., geology, slope).
        d_h (int):  Dimension to condition (must match TemporalEncoder d_h).

    Input:
        x_static: (N, d_s)

    Output:
        gamma: (N, d_h)  Feature-wise multiplicative factor
        beta:  (N, d_h)  Feature-wise additive offset

    Applied as:  h_temp <- gamma ⊙ h_temp + beta
    """
    def __init__(self, d_s: int, d_h: int):
        super().__init__()
        self.fc_gamma = nn.Linear(d_s, d_h)
        self.fc_beta  = nn.Linear(d_s, d_h)

    def forward(self, x_static):
        gamma = self.fc_gamma(x_static)
        beta  = self.fc_beta(x_static)
        return gamma, beta

class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder (your current stack).

    Args:
        in_dim (int):    Input feature dimension to the GAT (per node). In this
                         baseline we pass [x_static || x_temporal_today], so
                         in_dim = d_s + d_t.
        hid (int):       Hidden channel size for intermediate GAT layers.
        out (int):       Output channel size of the final GAT layer; this defines
                         d_g, the **spatial embedding** dimension.
        heads (int):     Number of attention heads in multi-head GAT layers.
        dropout (float): Feature dropout rate applied before each GATConv.
        num_layers (int):Number of GATConv layers in the stack.

    Input:
        x:          (N, in_dim)
        edge_index: (2, E) graph connectivity (PyG format)
        edge_attr:  (E, d_e) optional edge features (can be None)

    Output:
        h: (N, out) = (N, d_g)   Spatial embedding per node.
    """
    def __init__(self, in_dim: int, hid: int, out: int, heads: int, dropout: float, num_layers: int):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()

        if num_layers <= 1:
            # Single-layer GAT (no concatenation, single head)
            self.layers.append(GATConv(in_dim, out, heads=1, dropout=dropout, add_self_loops=True, concat=False))
        else:
            # First layer: in_dim -> hid * heads (concatenated)
            self.layers.append(GATConv(in_dim, hid, heads=heads, dropout=dropout, add_self_loops=True, concat=True))
            # Intermediate layers (hid*heads -> hid*heads)
            for _ in range(num_layers - 2):
                self.layers.append(GATConv(hid * heads, hid, heads=heads, dropout=dropout, add_self_loops=True, concat=True))
            # Final layer: hid*heads -> out (no concat, single head)
            self.layers.append(GATConv(hid * heads, out, heads=1, dropout=dropout, add_self_loops=True, concat=False))

    def forward(self, x, edge_index, edge_attr=None):
        h = x
        for i, layer in enumerate(self.layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = layer(h, edge_index, edge_attr)
            if i < len(self.layers) - 1:
                h = F.elu(h)
        return h


class TemporalProjection(nn.Module):
    """
    Projects temporal embedding to the GAT embedding size for fusion.

    Args:
        d_h (int): Input dimension (TemporalEncoder output size).
        d_g (int): Target dimension (GATEncoder output size).

    Init:
        Weights and bias are zero-initialised so that at initialisation the
        temporal path contributes **zero** (safe, preserves baseline).
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
    Produces a scalar gate α in (0,1) deciding the mix of GAT vs temporal.

    Args:
        in_dim (int):   Input dimension to the gate MLP; we use [h_gat || h_temp_proj],
                        so in_dim = 2 * d_g.
        bias_init (float): Bias initialisation for the last linear layer. Setting
                        bias_init=+5 makes σ(bias)≈0.993, so α≈0.993 initially,
                        i.e. ~99.3% GAT and ~0.7% temporal (pro-GAT prior).

    Input:
        z: (N, in_dim) = concatenated embeddings

    Output:
        alpha: (N, 1) with values in (0, 1)
    """
    def __init__(self, in_dim: int, bias_init: float = 5.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
        nn.init.zeros_(self.fc.weight)            # start with zero slope
        nn.init.constant_(self.fc.bias, bias_init)  # strong prior to GAT

    def forward(self, z):
        return torch.sigmoid(self.fc(z))  # (N, 1)


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