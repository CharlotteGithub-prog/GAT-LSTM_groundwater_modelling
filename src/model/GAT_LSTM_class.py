import sys
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from src.model import component_classes

# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)

class GAT_LSTM_Model(nn.Module):
    # Config imported directly to get hyperparams and random seed
    def __init__(self, in_channels, temporal_features_dim, static_features_dim, hidden_channels_gat, out_channels_gat,
                 heads_gat, dropout_gat, hidden_channels_lstm, num_layers_lstm, dropout_lstm, tbptt_window, num_layers_gat,
                 num_nodes, output_dim, fusion_gate_bias_init, run_GAT, run_LSTM, edge_dim, random_seed, catchment):

        super(GAT_LSTM_Model, self).__init__()
        logger.info(f"Instantiating GAT-LSTM model for {catchment} catchment...")
        logger.info(f"Model initialised under global random seed: {random_seed}.\n")

        # Store shapes and flags
        self.run_GAT = run_GAT
        self.run_LSTM = run_LSTM
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.temporal_features_dim = temporal_features_dim  # d_t
        self.static_features_dim = static_features_dim    # d_s
        self.in_channels = in_channels
        self.edge_dim = edge_dim

        # GAT hyper-params
        self.heads_gat = heads_gat
        self.dropout_gat = dropout_gat
        self.hidden_channels_gat = hidden_channels_gat
        self.out_channels_gat = out_channels_gat            # d_g
        self.num_layers_gat = num_layers_gat

        # LSTM hyper-params
        self.hidden_channels_lstm = hidden_channels_lstm
        self.num_layers_lstm = num_layers_lstm
        self.dropout_lstm = dropout_lstm
        self.tbptt_window = tbptt_window
        
        self.fusion_gate_bias_init = fusion_gate_bias_init

        # ----- Temporal branch (shared LSTM + FiLM + projection) -----
        
        if self.run_LSTM:
            self.temporal_encoder = component_classes.TemporalEncoder(
                d_t=temporal_features_dim,      # input width per day (d_t)
                d_h=hidden_channels_lstm,       # LSTM hidden (d_h)
                num_layers=num_layers_lstm
            )
            
            # Adding dropout to lstm:
            self.temporal_encoder.lstm.dropout = dropout_lstm
            
            self.node_conditioner = component_classes.NodeConditioner(
                d_s=static_features_dim,        # static width (d_s)
                d_h=hidden_channels_lstm        # match d_h
            )
            self.temp_proj = component_classes.TemporalProjection(
                d_h=hidden_channels_lstm,       # project from d_h
                d_g=out_channels_gat            # to d_g for fusion
            )
            logger.info(f"  LSTM Enabled: input={temporal_features_dim}, hidden={hidden_channels_lstm}, layers={num_layers_lstm}")
        else:
            logger.info("  LSTM Disabled.")

        # ----- Spatial branch (GAT) -----
        
        if self.run_GAT:
            # Preserve your successful baseline: GAT sees statics + current-day temporals
            gat_input_dim = static_features_dim + temporal_features_dim  # d_s + d_t
            self.gat_encoder = component_classes.GATEncoder(
                in_dim=gat_input_dim,
                hid=hidden_channels_gat,
                out=out_channels_gat,     # defines d_g
                heads=heads_gat,
                dropout=dropout_gat,
                num_layers=num_layers_gat,
                edge_dim=edge_dim
            )
            logger.info(f"  GAT Enabled with {num_layers_gat} layers. First layer: {gat_input_dim} -> "
                        f"{hidden_channels_gat} ({heads_gat} heads); final out={out_channels_gat}")
            final_output_dim = out_channels_gat  # head input when GAT is used (d_g)
        else:
            logger.info("  GAT Disabled: linear head on concatenated features.")
            # When GAT is off, head consumes [statics || something temporal]
            final_output_dim = (hidden_channels_lstm if self.run_LSTM else temporal_features_dim) + static_features_dim

        # ----- Fusion gate (only runs when both branches are active) -----
        
        if self.run_GAT and self.run_LSTM:
            # gate input is [h_gat || h_temp_proj] with size 2 * d_g
            self.fusion_gate = component_classes.FusionGate(
                in_dim=2 * out_channels_gat,
                bias_init=fusion_gate_bias_init  # α≈0.993 to GAT
            )
        else:
            self.fusion_gate = None  # not used

        # ----- Output head -----
        
        self.output_layer = nn.Linear(final_output_dim, output_dim, bias=True)
        logger.info(f"  Output Layer: {final_output_dim} -> {output_dim}\n")

    # Forward pass of model
    def forward(self, x, edge_index, edge_attr, current_timestep_node_ids, lstm_state_store=None):
        """
        Returns:
            predictions (Tensor): (N, output_dim) groundwater level predictions.
            (h_new, c_new) (tuple): Updated LSTM states **only for the provided node ids**.
            current_timestep_node_ids (LongTensor): Passthrough for the caller to update the store.
        """
        # Split input features into temporal and static blocks
        x_temporal = x[:, :self.temporal_features_dim]   # (N, d_t)
        x_static   = x[:, self.temporal_features_dim:]   # (N, d_s)

        # ---------------- Temporal branch ----------------
        h_temp_proj = None
        h_new, c_new = None, None

        if self.run_LSTM:
            x_seq = x_temporal.unsqueeze(1)  # (N, 1, d_t)

            # Slice global state store down to the nodes present in batch
            h_c_state = None
            if lstm_state_store is not None:
                h_full, c_full = lstm_state_store['h'], lstm_state_store['c']
                h_c_state = (
                    h_full[:, current_timestep_node_ids, :].contiguous(),  # (L, N, d_h)
                    c_full[:, current_timestep_node_ids, :].contiguous()   # (L, N, d_h)
                )

            # Shared LSTM forward
            h_temp, (h_new, c_new) = self.temporal_encoder(x_seq, h_c_state)  # h_temp: (N, d_h)

            # FiLM: make temporal embedding node-specific using statics
            gamma, beta = self.node_conditioner(x_static)                     # (N, d_h) each
            h_temp = gamma * h_temp + beta                                    # (N, d_h)

            # Project temporal embedding into GAT space (d_g) for fusion
            h_temp_proj = self.temp_proj(h_temp)                               # (N, d_g)

        # ---------------- Spatial branch -----------------
        
        if self.run_GAT:
            gat_input = torch.cat([x_static, x_temporal], dim=1)               # (N, d_s + d_t)
            h_gat = self.gat_encoder(gat_input, edge_index, edge_attr)         # (N, d_g)

            if self.run_LSTM:
                # Gated residual fusion: h_fused = α * h_gat + (1-α) * h_temp_proj
                z = torch.cat([h_gat, h_temp_proj], dim=1)                     # (N, 2*d_g)
                alpha = self.fusion_gate(z)                                     # (N, 1), α≈0.993 at init
                self.last_alpha = alpha.detach().cpu()      # store for attention head (componenet contribution) inspection later
                h_fused = alpha * h_gat + (1.0 - alpha) * h_temp_proj          # (N, d_g)
            else:
                h_fused = h_gat                                                # (N, d_g)

            predictions = self.output_layer(h_fused)                           # (N, output_dim)
            return predictions, (h_new, c_new), current_timestep_node_ids

        # ---------------- No-GAT fallback ----------------
        
        # If GAT is disabled, regress directly from concatenated features
        if self.run_LSTM:
            head_in = torch.cat([x_static, h_temp], dim=1)                     # (N, d_s + d_h)
        else:
            head_in = torch.cat([x_static, x_temporal], dim=1)                 # (N, d_s + d_t)

        predictions = self.output_layer(head_in)                               # (N, output_dim)
        return predictions, (h_new, c_new), current_timestep_node_ids
    