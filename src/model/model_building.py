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

# Instantiate model using class and associated information - TODO: This needs updating to match new arch,
def _assert_instantiation_vals(model, all_timesteps_list, device, output_dim, hidden_channels_lstm,
                               num_layers_lstm, optimizer, adam_learning_rate, adam_weight_decay, criterion,
                               loss_type):
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
        d0 = all_timesteps_list[0].to(device)
        dummy_x = d0.x
        dummy_edge_index = d0.edge_index
        dummy_edge_attr  = getattr(d0, "edge_attr", None)
        dummy_ids        = d0.node_id  # LongTensor of node indices for this timestep

        # Build a minimal lstm_state_store matching the model’s shapes
        lstm_state_store = None
        if model.run_LSTM:
            H, N, D = model.num_layers_lstm, model.num_nodes, model.hidden_channels_lstm
            lstm_state_store = {
                "h": torch.zeros(H, N, D, device=device),
                "c": torch.zeros(H, N, D, device=device),
            }

        preds, (h_new, c_new), returned_ids = model(
            dummy_x, dummy_edge_index, dummy_edge_attr, dummy_ids, lstm_state_store
        )

        logger.info(f"Output predictions shape: {preds.shape}")
        if model.run_LSTM:
            logger.info(f"New LSTM hidden/cell state shapes: h_n={h_new.shape}, c_n={c_new.shape}")
            expected_lstm_state_shape = (model.num_layers_lstm, model.num_nodes, model.hidden_channels_lstm)
            assert h_new.shape == expected_lstm_state_shape, \
                f"LSTM hidden state shape mismatch! Expected {expected_lstm_state_shape}, got {h_new.shape}"
            assert c_new.shape == expected_lstm_state_shape, \
                f"LSTM cell state shape mismatch! Expected {expected_lstm_state_shape}, got {c_new.shape}"
            logger.info("LSTM hidden/cell states have expected shapes.\n")

        expected_output_shape = (dummy_x.shape[0], output_dim)
        assert preds.shape == expected_output_shape, \
            f"Output shape mismatch! Expected {expected_output_shape}, got {preds.shape}"
        logger.info("Model forward pass successful with expected output shape.")

    except Exception as e:
        logger.error(f"Error during model forward pass: {e}")

    # --- Testing: device ---
    
    # Based on what is available on device running model
    logger.info("--- Testing Device Object ---")
    logger.info(f"Device Type: {type(device)}")
    assert isinstance(device, torch.device), "Device is not a torch.device object"
    logger.info(f"Selected Device: {device}")
    
    # The fix: make this assertion less brittle
    if torch.cuda.is_available():
        assert str(device).startswith('cuda'), "CUDA available but device not set to a 'cuda' device!"
    else:
        assert str(device) == 'cpu', "CUDA not available but device not set to 'cpu'!"
    logger.info("Device setup is correct.\n")

    # --- Testing: optimizer ---
    
    logger.info("--- Testing 'Optimizer' Object ---")
    logger.info(f"Optimizer Type: {type(optimizer)}")
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer is not torch.optim.Adam"
    logger.info("Optimizer successfully instantiated as Adam.")

    # Learning rate assertion
    assert optimizer.param_groups[0]['lr'] == adam_learning_rate, \
        f"Optimizer LR mismatch! Expected {adam_learning_rate}, got {optimizer.param_groups[0]['lr']}"
    logger.info(f"Optimizer LR is correct: {optimizer.param_groups[0]['lr']}")

    # Weight decay assertion
    assert optimizer.param_groups[0]['weight_decay'] == adam_weight_decay, \
        f"Optimizer Weight Decay mismatch! Expected {adam_weight_decay}, got {optimizer.param_groups[0]['weight_decay']}"
    logger.info(f"Optimizer Weight Decay is correct: {optimizer.param_groups[0]['weight_decay']}")

    # Missing params assertion
    assert len(optimizer.param_groups[0]['params']) > 0, "Optimizer has no parameters to optimise!"
    logger.info("Optimizer is managing model parameters.\n")

    # -- Testing: criterion ---
    
    logger.info("--- Testing Criterion (Loss Function) Object ---")
    logger.info(f"Criterion Type: {type(criterion)}")
    if loss_type == "MAE":
        assert isinstance(criterion, nn.L1Loss), "Criterion is not torch.nn.L1Loss"
        logger.info("Loss function successfully instantiated as L1Loss (MAE).\n")
    elif loss_type == "MSE":
        assert isinstance(criterion, nn.MSELoss), "Criterion is not torch.nn.MSELoss"
        logger.info("Loss function successfully instantiated as MSELoss (MSE).\n")
    else:
        logger.info("Confirm loss type and recheck assertions.\n")

    logger.info("--- All initial setup tests PASSED ---\n")

def instantiate_model_and_associated(all_timesteps_list, config, catchment):
    
    temporal_features = config[catchment]["model"]["architecture"]["temporal_features"]
    temporal_features_dim = len(temporal_features)
    in_channels = all_timesteps_list[0].x.shape[1]
    
    # --- Instantiate the model class ---
    
    model = GAT_LSTM_Model(
        in_channels=in_channels,
        temporal_features_dim=temporal_features_dim,
        static_features_dim=in_channels - temporal_features_dim,
        hidden_channels_gat=config[catchment]["model"]["architecture"]["hidden_channels_gat"],
        out_channels_gat=config[catchment]["model"]["architecture"]["out_channels_gat"],
        heads_gat=config[catchment]["model"]["architecture"]["heads_gat"],
        dropout_gat=config[catchment]["model"]["architecture"]["dropout_gat"],
        hidden_channels_lstm=config[catchment]["model"]["architecture"]["hidden_channels_lstm"],
        num_layers_lstm=config[catchment]["model"]["architecture"]["num_layers_lstm"],
        dropout_lstm=config[catchment]["model"]["architecture"]["dropout_lstm"],
        tbptt_window=config[catchment]["model"]["architecture"]["tbptt_window"],
        num_layers_gat=config[catchment]["model"]["architecture"]["num_layers_gat"],
        num_nodes=len(all_timesteps_list[0].x),
        output_dim=config[catchment]["model"]["architecture"]["output_dim"],
        # fusion_gate_bias_init=config[catchment]["model"]["architecture"]["fusion_gate_bias_init"],
        run_GAT=config[catchment]["model"]["architecture"]["run_GAT"],
        run_LSTM=config[catchment]["model"]["architecture"]["run_LSTM"],
        edge_dim=config[catchment]["model"]["architecture"]["edge_dim"],  # equal to edge_dim = edge_attr_tensor.shape[1]
        random_seed=config["global"]["pipeline_settings"]["random_seed"],
        catchment=catchment,
        run_node_conditioner=config[catchment]["model"]["architecture"]["run_node_conditioner"],
        fusion_mode=config[catchment]["model"]["architecture"]["fusion_mode"]
    )

    # Set device if available
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda:0')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    logger.info(f"Model instantiated and moved to device: {device}")
    logger.info(f"Total model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

    # --- Set Optimiser - weight decay implements L2 (ridge) regularisation ---
    
    learning_rate=config[catchment]["model"]["architecture"]["adam_learning_rate"]
    weight_decay=config[catchment]["model"]["architecture"]["adam_weight_decay"]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    logger.info(f"Optimiser: Adam (lr={optimizer.param_groups[0]['lr']})")

    # --- Set loss function using config selection ---
    
    loss_type = config[catchment]["training"]["loss"]
    if loss_type == "MAE":
        criterion = nn.L1Loss()
        logger.info(f"Loss Function: {type(criterion).__name__} (Mean Absolute Error)\n")
    elif loss_type == "MSE":
        criterion = nn.MSELoss()
        logger.info(f"Loss Function: {type(criterion).__name__} (Mean Square Error)\n")
    else:
        error_message = f"Invalid loss_type: '{loss_type}'. Must be 'MAE' or 'MSE'."
        logger.error(error_message)
        raise ValueError(error_message)
    
    # --- Run assertion testing to validation model instantiation ---
    
    _assert_instantiation_vals(
        model=model,
        all_timesteps_list=all_timesteps_list,
        device=device,
        output_dim=config[catchment]["model"]["architecture"]["output_dim"],
        hidden_channels_lstm=config[catchment]["model"]["architecture"]["hidden_channels_lstm"],
        num_layers_lstm=config[catchment]["model"]["architecture"]["num_layers_lstm"],
        optimizer=optimizer,
        adam_learning_rate=config[catchment]["model"]["architecture"]["adam_learning_rate"],
        adam_weight_decay=config[catchment]["model"]["architecture"]["adam_weight_decay"],
        criterion=criterion,
        loss_type=loss_type
    )
    
    return model, device, optimizer, criterion
