import sys
import torch
import logging
import numpy as np

# Set up logger config
logging.basicConfig(
    level=logging.INFO,
   format='%(levelname)s - %(message)s',
#    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logger for file and load config file for paths and params
logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience: int, delta: float, path: str, verbose: bool = False):
        """
        Args:
            patience (str): Number of epochs to wait for improvement in validation loss before stopping training.
            delta (float): Minimum change in loss required to count as imporovement.
            path (str): Path that improved model is saved to
            verbose (bool): If True then full logging runs for debugging, use False for reduced computation.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None  # Tracks best score (negative validation loss)
        self.early_stop = False
        self.val_loss_min = np.inf  # Tracks actual min validation loss
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss: float, model: torch.nn.Module):
        """
        Call method to update early stopping state based on current validation loss. Aim is to
        minimise val_loss, so tracking score = -val_loss (aim: maximise negative score).
        """
        score = -val_loss

        # On first run initialise best score tracker and save first iteration
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        
        # If no improvement then iterate tracker and check if early stopping is triggered
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        
        # If there is improvement then reset epoch counter and save latest iteration to path
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module):
        """Saves model when validation loss decreases (improves)."""
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss