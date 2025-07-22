import os
import sys
import math
import torch
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm # For progress bars

# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)

from src.training.early_stopping_class import EarlyStopping

# Implement Model Training and Validation Loops (8a)
def _run_epoch_phase(epoch, num_epochs, all_timesteps_list, gradient_clip_max_norm, model, device, criterion,
                     optimizer, is_training=True):
    
    # Initialise model to correct mode
    if is_training:
        model.train()
        description = "Training"
        mask_attr = 'train_mask'
    else:
        model.eval()
        description = "Validation"
        mask_attr = 'val_mask'
    
    total_loss = 0.0
    h_c_state = None  # Reset LSTM hidden/cell states for new seq
    
    # Iterate over all timesteps for training, applying spatial train_mask (+ tqdm for progress bars)
    loop = tqdm(all_timesteps_list, desc=f"Epoch {epoch+1}/{num_epochs} [{description}]", leave=False)
    
    #  Loop through and train/validate model
    with torch.set_grad_enabled(is_training): # Gradients enabled for training, disabled for validation
        for i, data in enumerate(loop):
            
            # Move data to device and run forward pass
            data=data.to(device)
            predictions, h_c_state = model(data.x, data.edge_index, data.edge_attr, h_c_state)
            
            # Detach hidden states for Truncated Backpropagation Through Time (BPTT)
            if model.run_LSTM and h_c_state is not None:  # update only required if LSTM is running
                h_c_state = (h_c_state[0].detach(), h_c_state[1].detach())
                
            # KEY: Get current mask and calculate loss only on relevant nodes (spatial split)
            current_mask = getattr(data, mask_attr)
            if not current_mask.any():
                # DEFENSIVE: If a timestep for some reason has no observed training nodes (e.g. all assigned val/test/NaNs)
                logger.debug(f"Epoch {epoch+1}, Timestep {data.timestep.date()}: No {description} nodes "
                                f"({current_mask} all False). Skipping loss for this timestep.")
                continue

            loss = criterion(predictions[current_mask], data.y[current_mask])
            total_loss += loss.item()
            
            if is_training:
                optimizer.zero_grad()  # Clear gradients from prev step
                loss.backward()  # Compute gradients
                
                # Clip gradient to help prevent exploding
                if gradient_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_max_norm)
                
                optimizer.step()  # Update model parameters
                
                 # Update tqdm postfix with current batch loss and learning rate -> provides real-time value feedback in terminal
                loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
        # Calculate average loss over all timesteps processed in training (where train_mask = True)
        avg_loss = total_loss / len(all_timesteps_list)
        return avg_loss

def run_training_and_validation(num_epochs: int, early_stopping_patience: int, lr_scheduler_factor: float,
                                lr_scheduler_patience: int, min_lr: float, gradient_clip_max_norm: float,
                                model_save_path: str, loss_delta: float, verbose: bool, catchment: str,
                                model: torch.nn.Module, device: torch.device, optimizer: torch.optim.Optimizer,
                                criterion: torch.nn.modules.loss._Loss, all_timesteps_list: list):
    """
    Executes the main training and validation loop for the GAT-LSTM model.

    This function orchestrates the epoch-wise training and validation loops, incorporating:
        - Spatial masking for loss calculation (training on 'gauged' nodes, validating on 'ungauged' val nodes).
        - LSTM state management for learning temporal dependencies across the full time series.
        - Learning rate scheduling (ReduceLROnPlateau) to adjust the learning rate dynamically.
        - Early stopping to prevent overfitting and save computational resources.
        - Gradient clipping to prevent exploding gradients.
    """
    logger.info(f"Setting up Training Loop for {catchment} catchment...")
    
    # Initialise Early Stopping class (all set to False upon __init__)
    early_stopper = EarlyStopping(
        patience=early_stopping_patience,
        delta=loss_delta,
        path=model_save_path,
        verbose=verbose
    )

    # Initialise Learning Rate Scheduler (now using built in torch ReduceLROnPlateau for efficiency)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',  # Monitor validation loss being minimised
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        min_lr=min_lr,
        verbose=verbose
    ) # NB: threshold and threshold_mode kept as default, add them in config if they need tuning.
    
    # Initialise training and validation lists to store metrics for plotting and analysis
    train_losses = []
    val_losses = []

    # --- Run full training and validation loop ---

    for epoch in range(num_epochs):
        
        # --- TRAINING PHASE ---
        
        avg_train_loss = _run_epoch_phase(epoch, num_epochs, all_timesteps_list, gradient_clip_max_norm,
                                          model, device, criterion, optimizer, is_training=True)
        
        train_losses.append(avg_train_loss)
        
        # --- VALIDATION PHASE ---
        
        avg_val_loss = _run_epoch_phase(epoch, num_epochs, all_timesteps_list, gradient_clip_max_norm,
                                          model, device, criterion, optimizer, is_training=False)
        val_losses.append(avg_val_loss)
        
        # Log Epoch Results
        logger.info(f"Epoch {epoch+1}/{num_epochs} Complete | Train Loss: {avg_train_loss:.4f}"
                    f" | Val Loss: {avg_val_loss:.4f}")
        
        # Learning rate scheduler step -> adjust lr based on val loss
        lr_scheduler.step(avg_val_loss)

        # --- Save Model if best so far / Early stop if not improving ---
        
        # Run Early Stopping __call__ -> Stop if val loss not improved for {patience} epochs
        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:  # Break if early stop call found to return True
            logger.info(f"Early stopping triggered at epoch {epoch+1}.")
            break
    
    # Load the best model weights found during training (saved in EarlyStopping)
    model.load_state_dict(torch.load(early_stopper.path))
    logger.info(f"Loaded best model from {early_stopper.path}")
    
    logger.info(f"--- Training Loop Finished ---\n")
    
    return train_losses, val_losses

def save_train_val_losses(output_analysis_dir, train_losses, val_losses):
    # Save outputs to model dir (confirming it exists)
    os.makedirs(output_analysis_dir, exist_ok=True)

    # Save train_losses and val_losses as .pt files
    torch.save(train_losses, os.path.join(output_analysis_dir, "train_losses.pt"))
    torch.save(val_losses, os.path.join(output_analysis_dir, "val_losses.pt"))
    logger.info(f"Training and validation losses saved to {output_analysis_dir} as .pt files.")

    # Save train_losses and val_losses as .npy files
    np.save(os.path.join(output_analysis_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(output_analysis_dir, "val_losses.npy"), np.array(val_losses))
    logger.info(f"Training and validation losses saved to {output_analysis_dir} as .npy files.")

    # Save train_losses and val_losses as CSV files
    pd.DataFrame(train_losses).to_csv(os.path.join(output_analysis_dir, "train_losses.csv"), index=False)
    pd.DataFrame(val_losses).to_csv(os.path.join(output_analysis_dir, "val_losses.csv"), index=False)
    logger.info(f"Training and validation losses saved to {output_analysis_dir} as .csv files.\n")
