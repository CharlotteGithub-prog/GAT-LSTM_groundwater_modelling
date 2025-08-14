import os
import json
import torch
import random
import hashlib
import platform
import numpy as np
from datetime import datetime

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def save_run_manifest(run_dir, config, git_commit, all_timesteps_list, temporal_features, scalers_dir,
                      train_station_ids, val_station_ids, test_station_ids, processed_df_path=None,
                      edge_index_path="data/03_graph/eden/edge_index_tensor.pt",
                      edge_attr_path="data/03_graph/eden/edge_attr_tensor.pt", sentinel_value=None,
                      epsilon=None, catchment="eden"):
    
    # Make run dir
    os.makedirs(run_dir, exist_ok=True)

    # node order (constant over timesteps)
    node_id_order = all_timesteps_list[0].node_id.cpu()
    torch.save(node_id_order, os.path.join(run_dir, "node_id_order.pt"))
    
    # Get timestamp (as time index not set for some in PyG obj)
    timesteps = []
    for i, d in enumerate(all_timesteps_list):
        entry = {"i": i}
        if hasattr(d, "timestep"):
            # stringify to make JSON-safe regardless of pandas/py datetime type
            try:
                entry["timestep"] = d.timestep.isoformat()
            except Exception:
                entry["timestep"] = str(d.timestep)
        if hasattr(d, "timestep_index"):
            try:
                entry["timestep_index"] = int(d.timestep_index.item() if hasattr(d.timestep_index, "item") else d.timestep_index)
            except Exception:
                entry["timestep_index"] = None
        timesteps.append(entry)
    
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "seeds": {
            "python": random.getstate()[1][0],
            "numpy": int(np.random.get_state()[1][0]),
            "torch": int(torch.initial_seed()),
            "cuda_deterministic": bool(torch.backends.cudnn.deterministic) if torch.backends.cudnn.is_available() else None,
            "cuda_benchmark": bool(torch.backends.cudnn.benchmark) if torch.backends.cudnn.is_available() else None,
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pyg": __import__("torch_geometric").__version__,
        },
        "temporal_features": temporal_features,
        "builder_flags": {
            "sentinel_value": sentinel_value,
            "epsilon": epsilon,
            "tbptt_window": config[catchment]["model"]["architecture"].get("tbptt_window", 1),
        },
        "paths_and_hashes": {
            "edge_index_tensor.pt": {"path": edge_index_path, "sha256": sha256(edge_index_path)},
            "edge_attr_tensor.pt":  {"path": edge_attr_path,  "sha256": sha256(edge_attr_path)},
        },
        "scalers": {
            "target_scaler.pkl": sha256(os.path.join(scalers_dir, "target_scaler.pkl"))
        },
        "config": config,
        "station_splits": {
            "train": train_station_ids,
            "val": val_station_ids,
            "test": test_station_ids
        },
        "processed_df": {"path": processed_df_path, "sha256": sha256(processed_df_path)} if processed_df_path else None,
        "timesteps": timesteps
    }

    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
