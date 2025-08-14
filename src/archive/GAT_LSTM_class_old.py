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

class GAT_LSTM_Model(nn.Module):
    # Config imported directly to get hyperparams and random seed
    def __init__(self, in_channels, temporal_features_dim, static_features_dim, hidden_channels_gat, out_channels_gat,
                 heads_gat, dropout_gat, hidden_channels_lstm, num_layers_lstm, num_layers_gat, num_nodes,
                 output_dim, run_GAT, run_LSTM, random_seed, catchment):
        """
        Initializes the GAT-LSTM Model with a flexible architecture.

        This model can be configured using [run_GAT, run_LSTM] flags to run various benchmarking runs with:
            1. GAT layers only (spatial feature learning). [True, False]
            2. LSTM layers only (temporal feature learning directly on input features). [False, True]
            3. Both GAT and LSTM layers (spatial-temporal feature learning). [True, True]
            4. Neither (a simple linear layer from input features to output). [False, False]

        The input and output dimensions of the GAT, LSTM, and the final output layer
        are dynamically adjusted based on these 'run_GAT' and 'run_LSTM' flags.
        """
        
        
        # Super init model
        super(GAT_LSTM_Model, self).__init__()
        logger.info(f"Instantiating GAT-LSTM model for {catchment} catchment...")
        logger.info(f"Model initialised under global random seed: {random_seed}.\n")
        
        # Store key attributes
        self.run_GAT = run_GAT
        self.run_LSTM = run_LSTM
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.temporal_features_dim = temporal_features_dim
        self.static_features_dim = static_features_dim
        self.in_channels = in_channels
        
        # GAT params
        self.heads_gat = heads_gat
        self.dropout_gat = dropout_gat
        self.hidden_channels_gat = hidden_channels_gat
        self.out_channels_gat = out_channels_gat
        self.num_layers_gat = num_layers_gat
        
        # LSTM params
        self.hidden_channels_lstm = hidden_channels_lstm
        self.num_layers_lstm = num_layers_lstm
        
        # --- LSTM Layer (Initially One Layer) ---
        
        if self.run_LSTM:
            self.lstm = nn.LSTM(input_size=temporal_features_dim, hidden_size=self.hidden_channels_lstm,
                                num_layers=self.num_layers_lstm, batch_first=True)
            self.lstm_output_dim = hidden_channels_lstm
            logger.info(f"  LSTM Enabled: input={temporal_features_dim}, hidden={self.hidden_channels_lstm}, layers={self.num_layers_lstm}")
        
        else:
            self.lstm_output_dim = temporal_features_dim
            logger.info("  LSTM Disabled.")

        # --- GAT Layers (Initially Two Layers) ---
        
        if self.run_GAT:
            
            # Use ModuleList to stack GAT layers
            self.gat_layers = nn.ModuleList()
            gat_input_dim = self.lstm_output_dim + static_features_dim
            
            # First GAT layer
            self.gat_layers.append(GATConv(gat_input_dim, self.hidden_channels_gat, heads=self.heads_gat,
                                           dropout=self.dropout_gat, add_self_loops=True, concat=True))
            
            # Intermediate GAT layers (if num_layers_gat > 2)
            for i in range(self.num_layers_gat - 2): # Loop for 0 to num_layers_gat - 3
                self.gat_layers.append(GATConv(self.hidden_channels_gat * self.heads_gat, self.hidden_channels_gat,
                                               heads=self.heads_gat, dropout=self.dropout_gat, add_self_loops=True, concat=True))
            
            # Last GAT layer
            if self.num_layers_gat > 1:
                self.gat_layers.append(GATConv(self.hidden_channels_gat * self.heads_gat, self.out_channels_gat,
                                               heads=1, dropout=self.dropout_gat, add_self_loops=True, concat=False))
            # if only 1 GAT layer
            else:
                self.gat_layers = nn.ModuleList([GATConv(gat_input_dim, self.out_channels_gat, heads=1,
                                                         dropout=self.dropout_gat, add_self_loops=True, concat=False)])


            logger.info(f"  GAT Enabled with {self.num_layers_gat} layers. First layer: {gat_input_dim} -> "
                        f"{self.hidden_channels_gat} ({self.heads_gat} heads)")
            
            if self.num_layers_gat > 1:
                logger.info(f"  Intermediate layers: {self.hidden_channels_gat * self.heads_gat} -> "
                            f"{self.hidden_channels_gat} ({self.heads_gat} heads)")
                logger.info(f"  Last layer: {self.hidden_channels_gat * self.heads_gat} -> {self.out_channels_gat}"
                            f" (1 head, concat=False)")
            
            # If GAT is run, LSTM's input comes from GAT's final output
            final_output_dim = self.out_channels_gat

        else:
            logger.info("  GAT Disabled: LSTM running with temporal feature inputs only.")
            final_output_dim = self.lstm_output_dim + static_features_dim 

        # --- Output Layer (Simple Linear) ---
        
        self.output_layer = nn.Linear(final_output_dim, self.output_dim, bias=True)
        logger.info(f"  Output Layer: {final_output_dim} -> {self.output_dim}\n")

        # --- Log final model architecture for clarity ---

        logger.info(f"Model Architecture:")
        
        if self.run_LSTM:
            logger.info(f"  LSTM [{self.run_LSTM}]: input={temporal_features_dim}, hidden={self.hidden_channels_lstm}, layers={self.num_layers_lstm}")
        else:
            logger.info(f"  LSTM [{self.run_LSTM}]: LSTM Layers Disabled")
        
        if self.run_GAT:
            logger.info(f"  GAT [{self.run_GAT}]: {gat_input_dim} -> {self.hidden_channels_gat} ({self.heads_gat} heads) -> {final_output_dim} (1 head, concat=False)")
        else:
            logger.info(f"  GAT [{self.run_GAT}]: GAT Layers Disabled")
        
        logger.info(f"  Output [True]: {final_output_dim} -> {self.output_dim}\n")
        
    # Define forward pass of model architecture
    def forward(self, x, edge_index, edge_attr, current_timestep_node_ids, lstm_state_store=None):
        """
        Performs a single forward pass through the GAT-LSTM model for one timestep. Processes spatial features
        using GAT layers and then temporal dependencies using an LSTM layer, producing groundwater level
        predictions for all nodes.

        Args:
            x (torch.Tensor): Node features (num_nodes_per_timestep, in_channels)
            edge_index (torch.Tensor): Graph connectivity (2, num_edges)
            edge_attr (torch.Tensor): Edge features (num_edges, num_edge_features)
            masked_node_ids (torch.Tensor): Indices of the nodes to which LSTM memory is applied and updated (e.g., training/test nodes).
            lstm_state_store (dict, optional): Global hidden and cell state store with keys 'h' and 'c'. Defaults to None.

        Returns:
            predictions (torch.Tensor): GWL predictions (num_nodes_per_timestep, output_dim)
            new_h_c_state (tuple): Updated (h, c) for the current node_ids only
            node_ids (torch.Tensor): Returned as-is for external state update
        """
        
        # Split node features into temporal and static
        x_temporal = x[:, :self.temporal_features_dim]
        x_static = x[:, self.temporal_features_dim:]
        
        # --- Prepare LSTM hidden state for current subset of nodes ---
        
        x_temporal_seq = x_temporal.unsqueeze(1)  # LSTM expects (batch, seq_len, features); seq_len=1

        # --- LSTM Forward Pass ---
        
        h_c_state_for_current_nodes = None
        
        if self.run_LSTM:
            if lstm_state_store and lstm_state_store is not None:
                h_full, c_full = lstm_state_store['h'], lstm_state_store['c']
                # Select states only for the nodes present in the current 'x' tensor
                h_c_state_for_current_nodes = (
                    h_full[:, current_timestep_node_ids, :].contiguous(),
                    c_full[:, current_timestep_node_ids, :].contiguous()
                )
                
        if self.run_LSTM:
            lstm_out, (h_new, c_new) = self.lstm(x_temporal_seq, h_c_state_for_current_nodes)
            lstm_embedding = lstm_out[:, -1, :] # Take the last timestep's output for seq_len=1
        else:
            h_new, c_new = None, None
            # lstm_embedding = torch.zeros(x.size(0), 0).to(x.device)    
            lstm_embedding = x_temporal  # When LSTM is not running, directly pass temporal features to GAT
        
        # --- GAT Forward Pass ---
        
        # Combine static features with LSTM embedding
        gat_input = torch.cat([x_static, lstm_embedding], dim=1)
        
        if self.run_GAT:
            for i, gat_layer in enumerate(self.gat_layers):
                gat_input = F.dropout(gat_input, p=self.dropout_gat, training=self.training)
                gat_input = gat_layer(gat_input, edge_index, edge_attr)
                if i < len(self.gat_layers) - 1:
                    gat_input = F.elu(gat_input)
            
        # --- Output Layer ---
        
        # This is always defined -> regardless of architecture configuration this layer always runs.
        predictions = self.output_layer(gat_input)
        
        return predictions, (h_new, c_new), current_timestep_node_ids
        