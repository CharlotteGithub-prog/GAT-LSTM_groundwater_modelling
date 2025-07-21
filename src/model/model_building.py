# Import Libraries
import os
import sys
import torch
import logging
import torch.nn as nn
from torch_geometric.loader import DataLoader

    
# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)

from src.model.GAT_LSTM_class import GAT_LSTM_Model

# Create data loaders for each timestep (7a)
def _check_data_loader(full_dataset_loader, all_timesteps_list):

    # Assert the number of batches matches the number of timesteps
    assert len(full_dataset_loader) == len(all_timesteps_list), "DataLoader batch count mismatch!"

    # Iterate through the first three batches and check shape and consistency
    num_batches_to_check = min(3, len(full_dataset_loader))

    for i, data_batch in enumerate(full_dataset_loader):
        if i >= num_batches_to_check:
            break
        
        logger.info(f"--- Batch {i+1} from DataLoader: ---")
        logger.info(f"    x shape: {data_batch.x.shape}")
        logger.info(f"    y shape: {data_batch.y.shape}")
        logger.info(f"    edge_index shape: {data_batch.edge_index.shape}")
        logger.info(f"    edge_attr shape: {data_batch.edge_attr.shape}\n")
        
        # Verify edge_index and edge_attr are consistent across batches (static graph as expected)
        if i == 0:
            first_edge_index = data_batch.edge_index
            first_edge_attr = data_batch.edge_attr
            
        else:
            assert torch.equal(data_batch.edge_index, first_edge_index), "edge_index changed across batches!"
            assert torch.equal(data_batch.edge_attr, first_edge_attr), "edge_attr changed across batches!"

    logger.info("DataLoader functionality check complete. All assertions passed.")

def build_data_loader(all_timesteps_list, batch_size, shuffle, catchment):
    
    logger.info(f"Building DataLoader for {catchment} catchment...\n")
    # Build dataset loader using global config vals
    full_dataset_loader = DataLoader(all_timesteps_list, batch_size=batch_size, shuffle=shuffle)
    logger.info(f"Created DataLoader with {len(full_dataset_loader)} batches (timesteps).")
    
    logger.info(f"Checking DataLoader has been built as expected...\n")
    _check_data_loader(full_dataset_loader, all_timesteps_list)

    
    return full_dataset_loader

# Instantiate model using class and associated information
def _assert_instantiation_vals(model, all_timesteps_list, device, output_dim, hidden_channels_lstm,
                               num_layers_lstm, optimizer, adam_learning_rate, adam_weight_decay, criterion):
    """
    Thorough assertion testing upon instantiation to catch errors earlier in pipeline to avoid wasting time.
    """
    
    logger.info("--- Running Model Setup Test ---\n")

    # --- Testing: Model ---

    # Model type assertion
    logger.info("--- Testing Model Object ---")
    logger.info(f"Model Type: {type(model)}")
    assert isinstance(model, nn.Module), "Model is not a torch.nn.Module"
    logger.info(f"Model successfully instantiated: {type(model).__name__}")

    # Model params device assertion
    logger.info(f"Model Device: {next(model.parameters()).device}")
    assert next(model.parameters()).device == device, "Model not on the correct device!"
    logger.info("Model parameters successfully moved to the correct device.")

    # Missing params assertion
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable model parameters: {param_count}")
    assert param_count > 0, "Model has no trainable parameters!"
    logger.info("Model has trainable parameters.\n")

    # --- Testing: dummy forward pass ---

    logger.info("--- Testing Model Forward Pass with Dummy Data ---")
    try:
        # Get actual shapes from first PyG Data object
        dummy_x = all_timesteps_list[0].x.to(device)
        dummy_edge_index = all_timesteps_list[0].edge_index.to(device)
        
        # Check if edge_attr exists and if GATConv is configured to use it (via edge_dim)
        dummy_edge_attr = all_timesteps_list[0].edge_attr.to(device) if hasattr(all_timesteps_list[0], 'edge_attr') \
            and all_timesteps_list[0].edge_attr is not None else None

        # Initial hidden/cell states for LSTM (will be zeros if None)
        dummy_h_c_state = None

        # Run forward pass
        output_predictions, new_h_c_state = model(dummy_x, dummy_edge_index, dummy_edge_attr, dummy_h_c_state)
        logger.info(f"Output predictions shape: {output_predictions.shape}")
        logger.info(f"New LSTM hidden/cell state shapes: h_n={new_h_c_state[0].shape}, c_n={new_h_c_state[1].shape}")

        # Expected output shape: (num_nodes, output_dim)
        expected_output_shape = (len(all_timesteps_list[0].x), output_dim)
        assert output_predictions.shape == expected_output_shape, \
            f"Output shape mismatch! Expected {expected_output_shape}, got {output_predictions.shape}"
        logger.info("Model forward pass successful with expected output shape.")

        # Expected LSTM hidden/cell state shape: (num_layers_lstm, num_nodes, hidden_channels_lstm)
        expected_lstm_state_shape = (num_layers_lstm, len(all_timesteps_list[0].x), hidden_channels_lstm)
        assert new_h_c_state[0].shape == expected_lstm_state_shape, \
            f"LSTM hidden state shape mismatch! Expected {expected_lstm_state_shape}, got {new_h_c_state[0].shape}"
        assert new_h_c_state[1].shape == expected_lstm_state_shape, \
            f"LSTM cell state shape mismatch! Expected {expected_lstm_state_shape}, got {new_h_c_state[1].shape}"
        logger.info("LSTM hidden/cell states have expected shapes.\n")

    except Exception as e:
        logger.error(f"Error during model forward pass: {e}")

    # --- Testing: device ---
    
    # Based on what is available on device running model
    logger.info("--- Testing Device Object ---")
    logger.info(f"Device Type: {type(device)}")
    assert isinstance(device, torch.device), "Device is not a torch.device object"
    logger.info(f"Selected Device: {device}")
    if torch.cuda.is_available():
        assert str(device) == 'cuda', "CUDA available but device not set to 'cuda'!"
    else:
        assert str(device) == 'cpu', "CUDA not available but device not set to 'cpu'!"
    logger.info("Device setup is correct.\n")

    # --- Testing: optimizer ---
    
    logger.info("--- Testing 'Optimizer' Object ---")
    logger.info(f"Optimizer Type: {type(optimizer)}")
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer is not torch.optim.Adam"
    logger.info("Optimizer successfully instantiated as Adam.")

    # Learnign rate assertion
    assert optimizer.param_groups[0]['lr'] == adam_learning_rate, \
        f"Optimizer LR mismatch! Expected {adam_learning_rate}, got {optimizer.param_groups[0]['lr']}"
    logger.info(f"Optimizer LR is correct: {optimizer.param_groups[0]['lr']}")

    # Weght decay assertion
    assert optimizer.param_groups[0]['weight_decay'] == adam_weight_decay, \
        f"Optimizer Weight Decay mismatch! Expected {adam_weight_decay}, got {optimizer.param_groups[0]['weight_decay']}"
    logger.info(f"Optimizer Weight Decay is correct: {optimizer.param_groups[0]['weight_decay']}")

    # Missing params assertion
    assert len(optimizer.param_groups[0]['params']) > 0, "Optimizer has no parameters to optimise!"
    logger.info("Optimizer is managing model parameters.\n")

    # -- Testing: criterion ---
    
    logger.info("--- Testing Criterion (Loss Function) Object ---")
    logger.info(f"Criterion Type: {type(criterion)}")
    assert isinstance(criterion, nn.L1Loss), "Criterion is not torch.nn.L1Loss"
    logger.info("Loss function successfully instantiated as L1Loss (MAE).\n")

    logger.info("--- All initial setup tests PASSED ---\n")

def instantiate_model_and_associated(all_timesteps_list, config, catchment):

    # Instantiate the model
    model = GAT_LSTM_Model(
        in_channels=all_timesteps_list[0].x.shape[1],
        hidden_channels_gat=config[catchment]["model"]["params"]["hidden_channels_gat"],
        out_channels_gat=config[catchment]["model"]["params"]["out_channels_gat"],
        heads_gat=config[catchment]["model"]["params"]["heads_gat"],
        dropout_gat=config[catchment]["model"]["params"]["dropout_gat"],
        hidden_channels_lstm=config[catchment]["model"]["params"]["hidden_channels_lstm"],
        num_layers_lstm=config[catchment]["model"]["params"]["num_layers_lstm"],
        num_nodes=len(all_timesteps_list[0].x),
        output_dim=config[catchment]["model"]["params"]["output_dim"],
        catchment=catchment
    )

    # Set device if available (Ham8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    logger.info(f"Model instantiated and moved to device: {device}")
    logger.info(f"Total model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

    # Set Optimiser - weight decay implements L2 (ridge) regularisation
    learning_rate=config[catchment]["model"]["params"]["adam_learning_rate"]
    weight_decay=config[catchment]["model"]["params"]["adam_weight_decay"]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    logger.info(f"Optimiser: Adam (lr={optimizer.param_groups[0]['lr']})")

    # Set loss function (Currently: MAE)
    criterion = nn.L1Loss()
    logger.info(f"Loss Function: {type(criterion).__name__} (Mean Absolute Error)\n")
    
    # --- Run assertion testing to validation model instantiation ---
    
    _assert_instantiation_vals(
        model=model,
        all_timesteps_list=all_timesteps_list,
        device=device,
        output_dim=config[catchment]["model"]["params"]["output_dim"],
        hidden_channels_lstm=config[catchment]["model"]["params"]["hidden_channels_lstm"],
        num_layers_lstm=config[catchment]["model"]["params"]["num_layers_lstm"],
        optimizer=optimizer,
        adam_learning_rate=config[catchment]["model"]["params"]["adam_learning_rate"],
        adam_weight_decay=config[catchment]["model"]["params"]["adam_weight_decay"],
        criterion=criterion
    )
    
    return model, device, optimizer, criterion
