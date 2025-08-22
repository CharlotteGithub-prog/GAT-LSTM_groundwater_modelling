# This file is part of the Dissertation project and is licensed under the MIT License.

### TRAINING OPTIMISATION PIPELINE ###

# ==============================================================================
# SECTION 1: IMPORTS & CONFIGURATION
# ==============================================================================

# --- 1a. Library Imports ---
import os
import ray
import ray.tune as tune
from ray.tune import CLIReporter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter

# Import ray training function
from src.optimisation.tune_trainable import train_model

# ==============================================================================
# SECTION 3: MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    os.environ.pop("RAY_ADDRESS", None) 
    
    # Initialise Ray to tune
    ray.init(ignore_reinit_error=True)

    # Create the search algorithm and wrap it in ConcurrencyLimiter.
    search_alg = HyperOptSearch(metric="val_loss", mode="min")
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=1)
    
    # --- Define hyperparam search space ---
    """
        - tune.choice(): for discrete values
        - tune.uniform(): for uniform distribution between a min and max
        - tune.loguniform(): for logarithmic uniform distribution
        - tune.grid_search(): for exhaustive grid search
    """
    search_space = {
        # --- GAT Architecture Params ---
        'heads_gat': tune.choice([8, 10, 12, 14, 16]),
        'dropout_gat': tune.uniform(0.3, 0.5),
        'hidden_channels_gat': tune.choice([48, 64, 80, 96]),
        'out_channels_gat': 64, # tune.choice([16, 32, 64, 96]),
        'num_layers_gat': 2, # tune.choice([1, 2, 3]),
        
        # --- LSTM Architecture Params ---
        

        # --- Optimiser & Training Params ---
        'adam_learning_rate': tune.loguniform(5e-4, 5e-3),
        'adam_weight_decay': tune.loguniform(1e-4, 1e-3),
        'early_stopping_patience': 25, # tune.choice([15, 25, 35]),
        'lr_scheduler_factor': tune.choice([0.2, 0.5, 0.8]),
        'lr_scheduler_patience': 10, # tune.choice([5, 6, 7, 8, 9, 10, 11, 12]),
        'min_lr': tune.loguniform(5e-7, 5e-6),
        'loss_delta': tune.loguniform(5e-5, 5e-4),
        'gradient_clip_max_norm': tune.choice([0.5, 1.0, 2.0]),
        'num_epochs': 200, # fixed
        "catchment": "eden"
    }
    
    # --- Run the hyperparameter search ---
    reporter = CLIReporter(metric_columns=["epoch","train_loss","val_loss","lr"])
    
    analysis = tune.run(
        train_model,
        config=search_space,
        resources_per_trial={"cpu": 4, "gpu": 1}, # 4 CPU and 1 GPU per trial
        search_alg=search_alg, # search algorithm defined at top
        num_samples=50, # n/o hyperparam combos to try
        metric="val_loss", # optimise loss
        mode="min", # minimise val loss
        name="ray_tune_gwl",
        progress_reporter=reporter,
        storage_path="/home3/swlc12/msc-groundwater-gwl-data/04_model/eden/model/ray_results" # for Ray logs and checkpoints
    )

    # --- Print the best trial's results ---
    print("Best hyperparameters found were: ", analysis.best_config)
    