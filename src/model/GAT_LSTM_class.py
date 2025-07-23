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
    def __init__(self, in_channels, hidden_channels_gat, out_channels_gat, heads_gat, dropout_gat, hidden_channels_lstm,
                 num_layers_lstm, num_layers_gat, num_nodes, output_dim, run_GAT, run_LSTM, random_seed, catchment):
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
        
        logger.info(f"Instantiating GAT-LSTM model for {catchment} catchment...")
        
        # Super init model
        super(GAT_LSTM_Model, self).__init__()

        # Verify random seed is set for reproducibility
        logger.info(f"Model initialised under global random seed: {random_seed}.\n")
        
        # Model Architecture
        self.run_GAT = run_GAT
        self.run_LSTM = run_LSTM
        self.in_channels = in_channels
        self.output_dim = output_dim  # one output prediction (one test station) per timestep
        self.num_nodes = num_nodes
        
        # GAT params
        self.heads_gat = heads_gat
        self.dropout_gat = dropout_gat
        self.hidden_channels_gat = hidden_channels_gat
        self.out_channels_gat = out_channels_gat
        self.num_layers_gat = num_layers_gat
        
        # LSTM params
        self.hidden_channels_lstm = hidden_channels_lstm
        self.num_layers_lstm = num_layers_lstm

        # --- GAT Layers (Initially Two Layers) ---
        # First GAT layer - Takes input features, outputs to hidden_channels_gat * heads_gat
        # Second GAT Layer - Takes concatenated output from conv1, outputs to out_channels_gat
        
        if self.run_GAT:
            
            # Use ModuleList to stack GAT layers
            self.gat_layers = nn.ModuleList()
            
            # First GAT layer
            self.gat_layers.append(GATConv(self.in_channels, self.hidden_channels_gat, heads=self.heads_gat,
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
                self.gat_layers = nn.ModuleList([GATConv(self.in_channels, self.out_channels_gat, heads=1,
                                                         dropout=self.dropout_gat, add_self_loops=True, concat=False)])


            logger.info(f"  GAT Enabled with {self.num_layers_gat} layers. First layer: {self.in_channels} -> "
                        f"{self.hidden_channels_gat} ({self.heads_gat} heads)")
            
            if self.num_layers_gat > 1:
                logger.info(f"  Intermediate layers: {self.hidden_channels_gat * self.heads_gat} -> "
                            f"{self.hidden_channels_gat} ({self.heads_gat} heads)")
                logger.info(f"  Last layer: {self.hidden_channels_gat * self.heads_gat} -> {self.out_channels_gat}"
                            f" (1 head, concat=False)")
            
            # If GAT is run, LSTM's input comes from GAT's final output
            lstm_input_dim = self.out_channels_gat 
            
            # self.conv1 = GATConv(self.in_channels, self.hidden_channels_gat, heads=self.heads_gat,
            #                     dropout=self.dropout_gat, add_self_loops=True, concat=True)
            # self.conv2 = GATConv(self.hidden_channels_gat * self.heads_gat, self.out_channels_gat,
            #                     heads=1, dropout=self.dropout_gat, add_self_loops=True, concat=False)  # 1 head for final layer after concat
            # logger.info(f"  GAT Enabled: {self.in_channels} -> {self.hidden_channels_gat} ({self.heads_gat} heads) -> {self.out_channels_gat} (1 head)")
            
            # # If GAT is run, LSTM's input comes from GAT's output
            # lstm_input_dim = self.out_channels_gat 

        else:
            logger.info("  GAT Disabled: LSTM running with in_channels orignal node feature inputs.")
            lstm_input_dim = self.in_channels   # If GAT is disabled, LSTM's input comes from the original node features
            
        # --- LSTM Layer (Initially One Layer) ---
        
        if self.run_LSTM:
            self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=self.hidden_channels_lstm,
                                num_layers=self.num_layers_lstm, batch_first=True)
            logger.info(f"  LSTM Enabled: input={lstm_input_dim}, hidden={self.hidden_channels_lstm}, layers={self.num_layers_lstm}")
            
            # If LSTM is run, the output layer's input comes from LSTM's output
            output_layer_input_dim = self.hidden_channels_lstm 
        
        else:
            logger.info("  LSTM Disabled: Output layer's input comes from GAT Output if enabled or original features if disabled.")
            output_layer_input_dim = lstm_input_dim 

        # --- Output Layer (Simple Linear) ---
        
        self.output_layer = nn.Linear(output_layer_input_dim, self.output_dim)
        logger.info(f"  Output Layer: {output_layer_input_dim} -> {self.output_dim}\n")

        # --- Log final model architecture for clarity ---

        logger.info(f"Model Architecture:")
        if self.run_GAT:
            logger.info(f"  GAT [{self.run_GAT}]: {self.in_channels} -> {self.hidden_channels_gat} ({self.heads_gat} heads) -> {self.out_channels_gat} (1 head, concat=False)")
        else:
            logger.info(f"  GAT [{self.run_GAT}]: GAT Layers Disabled")
        if self.run_LSTM:
            logger.info(f"  LSTM [{self.run_LSTM}]: input={lstm_input_dim}, hidden={self.hidden_channels_lstm}, layers={self.num_layers_lstm}")
        else:
            logger.info(f"  LSTM [{self.run_LSTM}]: LSTM Layers Disabled")
        logger.info(f"  Output [True]: {output_layer_input_dim} -> {self.output_dim}\n")
        
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
        
        if self.run_GAT:
            for i, gat_layer in enumerate(self.gat_layers):
                x = F.dropout(x, p=self.dropout_gat, training=self.training)
                x = gat_layer(x, edge_index, edge_attr)
                if i < len(self.gat_layers) - 1: # Apply activation for all but the last GAT layer
                    x = F.elu(x)
                    
            # x = F.dropout(x, p=self.conv1.dropout, training=self.training)  # Apply dropout to input features (training is an attr of the torch.nn.Module)
            # x = self.conv1(x, edge_index, edge_attr)  # First GAT layer
            # x = F.elu(x)  # Could also try ReLU, eLU currently used to try to avoid dead neurons (dead ReLU)
            # x = F.dropout(x, p=self.conv2.dropout, training=self.training)  # Apply dropout after activation, before next layer
            # x = self.conv2(x, edge_index, edge_attr)  # Second GAT layer
    
        # Reshape for LSTM (no final layer before as fed straight in)
        x_subsequent_input = x.view(self.num_nodes, 1, -1)  # NB: If self.run_GAT is False, x remains the original input x.  
        
        predictions = None
        new_h_c_state = None # Initialise new_h_c_state
        
        # --- LSTM Forward Pass ---
        
        if self.run_LSTM:
            # If h_c_state = None, hidden and cell states automatically initialised to zeros.
            lstm_out, new_h_c_state = self.lstm(x_subsequent_input, h_c_state)
            output_layer_features = lstm_out.squeeze(1) 
        else:
            output_layer_features = x_subsequent_input.squeeze(1)
            
        # --- Output Layer ---
        
        # This is always defined -> regardless of architecture configuration this layer always runs.
        predictions = self.output_layer(output_layer_features)
        
        return predictions, new_h_c_state
        