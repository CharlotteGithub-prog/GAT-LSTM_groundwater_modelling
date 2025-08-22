
# --- 1a. Library Imports ---
import os
import sys
import torch
import random
import logging
from pathlib import Path
from ray.air import session
import random, numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parents[2] 
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", PROJECT_ROOT))
config_path = PROJECT_ROOT / "config" / "project_config.yaml"
if not config_path.exists():
    raise FileNotFoundError(f"Config not found at {config_path}. "
                            f"Computed PROJECT_ROOT={PROJECT_ROOT}. "
                            "If this is wrong, set env var PROJECT_ROOT=/home3/swlc12/msc-groundwater-gwl")

# ==============================================================================
# SECTION 2: RAY TUNE TRAINABLE FUNCTION
# ==============================================================================

def train_model(tune_config):
    from src.utils.config_loader import load_project_config, deep_format, expanduser_tree
    from src.model import model_building
    from src.training import model_training
    import logging, sys, os
    from pathlib import Path

    # ---- logging
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    config = load_project_config(config_path=str(config_path))

    # resolve roots to absolute paths
    project_root = PROJECT_ROOT
    raw_root_cfg = config["global"]["paths"]["raw_data_root"]
    res_root_cfg = config["global"]["paths"]["results_root"]

    raw_data_root = Path(os.environ.get("DATA_ROOT", raw_root_cfg))
    results_root  = Path(os.environ.get("RESULTS_ROOT", res_root_cfg))

    if not raw_data_root.is_absolute():
        raw_data_root = (project_root / raw_data_root).resolve()
    if not results_root.is_absolute():
        results_root = (project_root / results_root).resolve()

    # apply templating then expand ~, env, etc.
    config = deep_format(config, raw_data_root=str(raw_data_root), results_root=str(results_root))
    config = expanduser_tree(config)
    
    catchment = tune_config["catchment"]

    paths = config[catchment]["paths"]
    for k, v in list(paths.items()):
        if isinstance(v, str):
            q = Path(v)
            paths[k] = str(q if q.is_absolute() else (PROJECT_ROOT / q).resolve())

    # also make global roots absolute
    config["global"]["paths"]["raw_data_root"] = str(raw_data_root)
    config["global"]["paths"]["results_root"]  = str(results_root)

    # quick sanity prints (remove later)
    print("CWD:", Path.cwd())
    print("Resolved PyG path:", config[catchment]["paths"]["pyg_object_path"])
    print("Exists?", Path(config[catchment]["paths"]["pyg_object_path"]).exists())

    # set up seeding
    seed = config["global"]["pipeline_settings"]["random_seed"]
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    
    # ==============================================================================
    # OVERRIDE THE CONFIG WITH HYPERPARAMS FROM RAY TUNE
    # ==============================================================================
    
    # GAT Architecture Params
    config[catchment]["model"]["architecture"]["heads_gat"] = tune_config["heads_gat"]
    config[catchment]["model"]["architecture"]["dropout_gat"] = tune_config["dropout_gat"]
    config[catchment]["model"]["architecture"]["hidden_channels_gat"] = tune_config["hidden_channels_gat"]
    config[catchment]["model"]["architecture"]["out_channels_gat"] = tune_config["out_channels_gat"]
    config[catchment]["model"]["architecture"]["num_layers_gat"] = tune_config["num_layers_gat"]

    # Optimiser & Training Params
    config[catchment]["model"]["architecture"]["adam_learning_rate"] = tune_config["adam_learning_rate"]
    config[catchment]["model"]["architecture"]["adam_weight_decay"] = tune_config["adam_weight_decay"]
    config[catchment]["training"]["num_epochs"] = tune_config["num_epochs"]
    config[catchment]["training"]["early_stopping_patience"] = tune_config["early_stopping_patience"]
    config[catchment]["training"]["lr_scheduler_factor"] = tune_config["lr_scheduler_factor"]
    config[catchment]["training"]["lr_scheduler_patience"] = tune_config["lr_scheduler_patience"]
    config[catchment]["training"]["min_lr"] = tune_config["min_lr"]
    config[catchment]["training"]["loss_delta"] = tune_config["loss_delta"]
    config[catchment]["training"]["gradient_clip_max_norm"] = tune_config["gradient_clip_max_norm"]
    
    # ==============================================================================
    # MODEL INSTANTIATION
    # ==============================================================================

    # --- 7a. Build Data Loaders by Timestep ---
    all_timesteps_path = config[catchment]["paths"]["pyg_object_path"]
    if not Path(all_timesteps_path).exists():
        raise FileNotFoundError(f"PyG file not found at {all_timesteps_path}")

    all_timesteps_list = torch.load(all_timesteps_path)
    # full_dataset_loader = model_building.build_data_loader(
    #     all_timesteps_list=all_timesteps_list,
    #     batch_size=config["global"]["model"]["data_loader_batch_size"],
    #     shuffle=config["global"]["model"]["data_loader_shuffle"],
    #     catchment=catchment,
    #     seed=config["global"]["pipeline_settings"]["random_seed"]
    # )

    logger.info(f"Pipeline Step 'Create PyG DataLoaders' complete for {catchment} catchment.\n")
    
    # --- 7b. Define Graph Neural Network Architecture including loss and optimiser definition ---

    # Adjust model architecture and params in catchment-specific config. TODO: Further optimise hyperparams.
    model, device, optimizer, criterion = model_building.instantiate_model_and_associated(
        all_timesteps_list=all_timesteps_list,
        config=config,
        catchment=catchment
    )

    logger.info(f"Pipeline Step 'Instantiate GAT-LSTM Model' complete for {catchment} catchment.\n")

    # ==============================================================================
    # MODEL TRAINING
    # ==============================================================================
    
    # --- 8a. Implement Training Loop ---
    
    # Explicit path for NCC save
    pt_model_root = "/home3/swlc12/msc-groundwater-gwl-data/04_model/eden/model/pt_model"
    
    # put each trial in subfolder for easy testing
    try:
        trial_id = session.get_trial_id()
    except Exception:
        trial_id = None
    
    trial_id = trial_id or "local"
    trial_name = session.get_trial_name() or "local"
    logger.info(f"Starting trial {trial_name} (id={trial_id})")
    
    pt_model_dir  = os.path.join(pt_model_root, f"trial_{trial_id}")
    os.makedirs(pt_model_dir, exist_ok=True)
    
    train_losses, val_losses = model_training.run_training_and_validation(
        num_epochs=config[catchment]["training"]["num_epochs"],
        early_stopping_patience=config[catchment]["training"]["early_stopping_patience"],
        lr_scheduler_factor=config[catchment]["training"]["lr_scheduler_factor"],
        lr_scheduler_patience=config[catchment]["training"]["lr_scheduler_patience"],
        min_lr=config[catchment]["training"]["min_lr"],
        gradient_clip_max_norm=config[catchment]["training"]["gradient_clip_max_norm"],
        model_save_dir=pt_model_dir,  # config[catchment]["paths"]["model_dir"],
        loss_delta=config[catchment]["training"]["loss_delta"],
        verbose=config[catchment]["training"]["verbose"],
        catchment=catchment,
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        all_timesteps_list=all_timesteps_list,  # all_timesteps_list,
        scalers_dir=config[catchment]["paths"]["scalers_dir"],
        config=config
    )

    logger.info(f"Pipeline Step 'Train and Validate Model' complete for {catchment} catchment.")
    
    # Report final val loss to Ray Tune
    best = float(min(val_losses))
    session.report({"val_loss": best, "loss": best})

    # relative_train_path, relative_val_path = model_training.save_train_val_losses(
    #     output_analysis_dir=config[catchment]["paths"]["model_dir"],
    #     train_losses=train_losses,
    #     val_losses=val_losses,
    #     config=config,
    #     catchment=catchment
    # )

    # logger.info(f"Pipeline Step 'Save Training and Validation Losses' complete for {catchment} catchment.")