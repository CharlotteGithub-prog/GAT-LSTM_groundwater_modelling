import sys
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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
                 num_nodes, output_dim, run_GAT, run_LSTM, edge_dim, random_seed, catchment, run_node_conditioner: bool = True,
                 fusion_mode: str = "gate", spatial_mem_decay: float = 0.9, use_h_temp_in_gat: bool = False):

        super(GAT_LSTM_Model, self).__init__()
        logger.info(f"Instantiating GAT-LSTM model for {catchment} catchment...")
        logger.info(f"Model initialised under global random seed: {random_seed}.\n")

        # Store shapes and flags
        self.run_GAT = bool(run_GAT)
        self.run_LSTM = bool(run_LSTM)
        self.run_node_conditioner = bool(run_node_conditioner)
        self.fusion_mode = fusion_mode
        
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.temporal_features_dim = temporal_features_dim  # d_t
        self.static_features_dim = static_features_dim    # d_s
        self.in_channels = in_channels
        self.edge_dim = edge_dim
        
        # Residual memory channel (EMA)
        self.spatial_mem_decay = float(spatial_mem_decay)  # 0..1, higher = slower memory
        self.use_h_temp_in_gat = bool(use_h_temp_in_gat)   # False = use raw temporals in GAT
        self.include_residual_memory = True
        self.mem_dim = 1

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

        # ----- Temporal branch (shared LSTM + FiLM + projection) -----
        
        if self.run_LSTM:
            self.temporal_encoder = component_classes.TemporalEncoder(
                d_t=temporal_features_dim,      # input width per day (d_t)
                d_h=hidden_channels_lstm,       # LSTM hidden (d_h)
                num_layers=num_layers_lstm,
                dropout=dropout_lstm
            )
            
            self.temporal_dropout = nn.Dropout(self.dropout_lstm)
            
            # Adding dropout to lstm: (INCORRECT)
            # self.temporal_encoder.lstm.dropout = dropout_lstm
            
            if self.run_node_conditioner:
                self.node_conditioner = component_classes.NodeConditioner(
                    d_s=static_features_dim,        # static width (d_s)
                    d_h=hidden_channels_lstm        # match d_h
                )
                logger.info("  NodeConditioner: ENABLED (FiLM over statics).")
            else:
                self.node_conditioner = None
                logger.info("  NodeConditioner: DISABLED (identity FiLM).")
                
            logger.info(f"  LSTM Enabled: input={temporal_features_dim}, hidden={hidden_channels_lstm}, layers={num_layers_lstm}")
        else:
            logger.info("  LSTM Disabled.")

        # ----- Spatial branch (GAT) -----
        
        if self.run_GAT:
            # Not passing h_temp when use temp True, False for now to avoid shortcutting (tempo only in base)
            base_temporal_dim = (hidden_channels_lstm if (self.run_LSTM and self.use_h_temp_in_gat)
                                 else temporal_features_dim)
            gat_input_dim = static_features_dim + base_temporal_dim + (self.mem_dim if self.include_residual_memory else 0)
            
            self.gat_encoder = component_classes.GATEncoder(
                in_dim=gat_input_dim, hid=hidden_channels_gat, out=out_channels_gat,
                heads=heads_gat, dropout=dropout_gat, num_layers=num_layers_gat, edge_dim=edge_dim
            )
            logger.info(f"  GAT Enabled with {num_layers_gat} layers. in_dim={gat_input_dim} "
                        f"(statics={static_features_dim}, temporal={base_temporal_dim}, mem={self.mem_dim}) "
                        f"→ hid={hidden_channels_gat} ({heads_gat} heads), out={out_channels_gat}")
        else:
            logger.info("  GAT Disabled: linear head on concatenated features.")

        # --- Heads ---
        
        # LSTM head (if LSTM is on)
        if self.run_LSTM:
            self.head_lstm = nn.Linear(self.hidden_channels_lstm, self.output_dim)
            # temporal projection exists ONLY to align embeddings for the gate
            self.temp_proj = component_classes.TemporalProjection(
                d_h=self.hidden_channels_lstm, d_g=self.out_channels_gat
            )

        # GAT head (if GAT is on)
        if self.run_GAT:
            self.head_gat = nn.Linear(self.out_channels_gat, self.output_dim)
            
        # --- Output calibration parameters (learned) ---

        # start just as an idewntity
        self.tau_lstm  = nn.Parameter(torch.tensor(1.0))
        self.bias_lstm = nn.Parameter(torch.tensor(0.0))
        self.tau_gat   = nn.Parameter(torch.tensor(1.0))
        self.bias_gat  = nn.Parameter(torch.tensor(0.0))

        # Fusion gate only when both branches are enabled + alternatives for ablations
        if self.run_GAT and self.run_LSTM:
            if self.fusion_mode == "gate":
                self.fusion_gate = component_classes.FusionGate(
                    in_dim=2 * self.out_channels_gat, bias_init=1.0, alpha_floor=0.15
                )
            elif self.fusion_mode == "scalar":
                self.alpha_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid->0.5
            elif self.fusion_mode == "fixed":
                self.alpha_fixed = 0.5
            elif self.fusion_mode == "add":
                pass
            else:
                raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

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
        x_static = x[:, self.temporal_features_dim:]   # (N, d_s)

        # ---------------- Temporal branch ----------------
        h_temp = None
        h_new, c_new = None, None
        y_lstm = 0.0

        if self.run_LSTM:
            x_seq = x_temporal.unsqueeze(1)  # (N, 1, d_t)

            # Slice global state store down to the nodes present in batch
            h_c_state = None
            if lstm_state_store is not None:
                h_full, c_full = lstm_state_store['h'], lstm_state_store['c']
                h_c_state = (
                    h_full[:, current_timestep_node_ids, :].contiguous(),  # (L, N, d_h)
                    c_full[:, current_timestep_node_ids, :].contiguous()
                )

            # Shared LSTM forward
            h_temp, (h_new, c_new) = self.temporal_encoder(x_seq, h_c_state)  # h_temp: (N, d_h)
            h_temp = self.temporal_dropout(h_temp)  # <-- actually regularises now

            # # FiLM: make temporal embedding node-specific using statics
            # gamma, beta = self.node_conditioner(x_static)  # (N, d_h) each
            # h_temp = gamma * h_temp + beta  # (N, d_h)

            if self.run_node_conditioner:
                gamma, beta = self.node_conditioner(x_static)   # (N, d_h)
                h_temp = gamma * h_temp + beta
            else:
                # Identity FiLM so downstream code & debug don’t break
                gamma = torch.ones_like(h_temp)
                beta  = torch.zeros_like(h_temp)

            # Define temporal embedding
            # y_lstm = self.head_lstm(h_temp)  # (N, output_dim)
            y_lstm = self.bias_lstm + self.tau_lstm * self.head_lstm(h_temp)

        # ---------------- Spatial branch -----------------
        
        g_i = None
        # Pull residual memory for the nodes in this timestep (vector length N)
        if lstm_state_store is not None and 'r_mem' in lstm_state_store:
            r_mem_curr = lstm_state_store['r_mem'][current_timestep_node_ids].unsqueeze(1)  # (N,1)
        else:
            # First call / inference without store — use zeros on the right device
            r_mem_curr = torch.zeros(x.size(0), 1, device=x.device)

        if self.run_GAT:
            # Choose temporal signal for GAT: raw temporals (recommended) or h_temp
            if self.run_LSTM and self.use_h_temp_in_gat:
                temp_for_gat = h_temp.detach()  # (N, d_h)
            else:
                temp_for_gat = x_temporal  # (N, d_t)

            gat_input = torch.cat([x_static, temp_for_gat, r_mem_curr], dim=1)  # (N, d_s + d_t_or_d_h + 1)
            g_i = self.gat_encoder(gat_input, edge_index, edge_attr)  # (N, d_g)

        # ------------- Gated / fallback fusion -------------
        
        alpha = None
        # if self.run_LSTM and self.run_GAT:
        #     # map to predictions
        #     # y_gat = self.head_gat(g_i)                 # (N, output_dim)
        #     y_gat = self.bias_gat  + self.tau_gat  * self.head_gat(g_i)
        #     # gate operates on embeddings, not outputs
        #     h_temp_proj = self.temp_proj(h_temp)       # (N, d_g)
        #     fusion_input = torch.cat([g_i, h_temp_proj], dim=1)  # (N, 2*d_g)
        #     alpha = self.fusion_gate(fusion_input)     # (N, 1)
        #     predictions = alpha * y_gat + (1 - alpha) * y_lstm
            
        #     # keep a grad-carrying handle for the aux loss
        #     self.aux_y_gat = y_gat
        
        if self.run_LSTM and self.run_GAT:
            y_gat = self.bias_gat + self.tau_gat * self.head_gat(g_i)

            if self.fusion_mode == "gate":
                h_temp_proj = self.temp_proj(h_temp)
                fusion_input = torch.cat([g_i, h_temp_proj], dim=1)
                alpha = self.fusion_gate(fusion_input)
                predictions = alpha * y_gat + (1 - alpha) * y_lstm

            elif self.fusion_mode == "scalar":
                a = torch.sigmoid(self.alpha_logit)
                alpha = a.expand(y_gat.size(0), 1)
                predictions = a * y_gat + (1 - a) * y_lstm

            elif self.fusion_mode == "fixed":
                alpha = y_gat.new_full((y_gat.size(0), 1), self.alpha_fixed)
                predictions = 0.5 * y_gat + 0.5 * y_lstm

            elif self.fusion_mode == "add":
                alpha = None
                predictions = y_lstm + y_gat

            self.aux_y_gat = y_gat

        elif self.run_GAT:
            predictions = self.head_gat(g_i)
            y_gat = self.bias_gat + self.tau_gat * self.head_gat(g_i)

        elif self.run_LSTM:
            predictions = y_lstm
        else:
            raise RuntimeError("Both run_GAT and run_LSTM are False.")
        
        # --- Compute residual vs baseline (if LSTM exists) ---
        
        if self.run_LSTM:
            residual_hat = (predictions - y_lstm).detach()  # (N,1), stop grad
        else:
            residual_hat = predictions.detach()              # degenerate case

        # Update residual memory for these nodes (EMA)
        # r_new = decay * r_prev + (1 - decay) * residual_hat
        r_mem_new = self.spatial_mem_decay * r_mem_curr + (1.0 - self.spatial_mem_decay) * residual_hat
        
        # --- Debug payload (no-grad) ---
        
        with torch.no_grad():
            self.last_debug = {
                "y_pred": predictions.detach(),
                "y_lstm_only": (y_lstm.detach() if self.run_LSTM else None),
                "baseline": (y_lstm.detach() if self.run_LSTM else None),
                "residual": (predictions - y_lstm).detach() if self.run_LSTM else predictions.detach(),
                "gamma": (gamma.detach() if self.run_LSTM else None),
                "beta": (beta.detach() if self.run_LSTM else None),
                "alpha": (alpha.detach() if alpha is not None else None), 
                "y_gat_log": (y_gat.detach() if self.run_GAT and y_gat is not None else None),
                
                # + residual memory snapshots for the nodes in this batch
                "r_mem_prev": r_mem_curr.detach(),  # (N,1)
                "r_mem_new": r_mem_new.detach(),  # (N,1)
            }

        return predictions, (h_new, c_new), current_timestep_node_ids
    