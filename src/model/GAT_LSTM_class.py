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
    def __init__(self, in_channels, hidden_channels_gat, out_channels_gat, heads_gat, dropout_gat,
                 hidden_channels_lstm, num_layers_lstm, num_nodes, output_dim, catchment):
        
        logger.info(f"Instantiating GAT-LSTM model for {catchment} catchment...")

        # Verify random seed is set for reproducibility
        logger.info(f"Model initialised with global random seed.\n")
        
        self.in_channels = in_channels
        
        # GAT params
        self.heads_gat = heads_gat
        self.dropout_gat = dropout_gat
        self.hidden_channels_gat = hidden_channels_gat
        self.out_channels_gat = out_channels_gat
        
        # LSTM params
        self.hidden_channels_lstm = hidden_channels_lstm
        self.num_layers_lstm = num_layers_lstm
        
        # Output layer params
        self.output_dim = output_dim  # one output prediction (one test station) per timestep
        self.num_nodes = num_nodes
        
        # Init model
        super(GAT_LSTM_Model, self).__init__()

        # --- GAT Layers (Initially Two Layers) ---
        
        # First GAT layer - Takes input features, outputs to hidden_channels_gat * heads_gat
        self.conv1 = GATConv(self.in_channels, self.hidden_channels_gat, heads=self.heads_gat,
                             dropout=self.dropout_gat, add_self_loops=True, concat=True)
        
        # Second GAT Layer - Takes concatenated output from conv1, outputs to out_channels_gat
        self.conv2 = GATConv(self.hidden_channels_gat * self.heads_gat, self.out_channels_gat,
                             heads=1, dropout=self.dropout_gat, add_self_loops=True, concat=False)  # 1 head for final layer after concat
    
        # --- LSTM Layer (Initially One Layer) ---
        
        self.lstm = nn.LSTM(input_size=self.out_channels_gat, hidden_size=self.hidden_channels_lstm,
                            num_layers=self.num_layers_lstm, batch_first=True)

        # --- Output Layer (Simple Linear) ---
        
        self.output_layer = nn.Linear(self.hidden_channels_lstm, self.output_dim)

        logger.info(f"Model Architecture:")
        logger.info(f"  GAT: {self.in_channels} -> {self.hidden_channels_gat} ({self.heads_gat} heads) -> {self.out_channels_gat} (1 head, concat=False)")
        logger.info(f"  LSTM: input={self.out_channels_gat}, hidden={self.hidden_channels_lstm}, layers={self.num_layers_lstm}")
        logger.info(f"  Output: {self.hidden_channels_lstm} -> {self.output_dim}\n")
        
    # Define forward pass of model architecture
    def forward(self, x, edge_index, edge_attr, h_c_state=None):
        """
        Performs a single forward pass through the GAT-LSTM model for one timestep. Processes spatial features
        using GAT layers and then temporal dependencies using an LSTM layer, producing groundwater level
        predictions for all nodes.

        Args:
            x (torch.Tensor): node features (num_nodes, in_channels)
            edge_index (torch.Tensor): graph connectivity (2, num_edges)
            edge_attr (torch.Tensor):edge features (num_edges, num_edge_features)
            h_c_state (tuple, optional): (h_n, c_n) tuple from previous LSTM step. Defaults to None.
            
        Returns:
            tuple (predictions, new_h_c_state):
                - predictions (torch.Tensor): Predicted groundwater levels for all nodes at the current timestep.
                - new_h_c_state (tuple): The updated hidden state (h_n) and cell state (c_n) of the LSTM after
                                        processing the current timestep. This is then passed to the next forward call.
        """
        # --- GAT Forward Pass ---
        
        x = F.dropout(x, p=self.conv1.dropout, training=self.training)  # Apply dropout to input features (training is an attr of the torch.nn.Module)
        x = self.conv1(x, edge_index, edge_attr)  # First GAT layer
        x = F.elu(x)  # Could also try ReLU, eLU currently used to try to avoid dead neurons (dead ReLU)
        x = F.dropout(x, p=self.conv2.dropout, training=self.training)  # Apply dropout after activation, before next layer
        x = self.conv2(x, edge_index, edge_attr)  # Second GAT layer
        
        x_lstm_input = x.view(self.num_nodes, 1, -1)  # Reshape for LSTM (no final layer before as fed straight in)
        
        # --- LSTM Forward Pass ---
        
        # If h_c_state = None, hidden and cell states automatically initialised to zeros.
        lstm_out, new_h_c_state = self.lstm(x_lstm_input, h_c_state)
        
        # --- Output Layer ---
        
        # Apply linear layer to each node's output
        predictions = self.output_layer(lstm_out.squeeze(1))
        
        return predictions, new_h_c_state
        