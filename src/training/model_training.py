import os
import sys
import math
import time
import torch
# import pickle
import joblib
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm # For progress bars
from datetime import datetime
import torch.nn.functional as F

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
def _calc_smooth_curvature_loss(loss, data, previous_targets, sequential_mask, loss_type, pred_t,
                              lambda_curve, prev_pred, lambda_smooth):
    # First-difference (rate matching) penalty: discourages day-to-day jitter
    true_t = data.y - previous_targets
    
    # Scale to number of valid values (not masked gwl) in sequential mask - apply to next four calcs
    n_seq = sequential_mask.sum().clamp(min=1)
    
    # Encourage a match to the rate of change in the training data
    if loss_type == 'MAE':
        rate_loss = torch.sum(torch.abs(pred_t[sequential_mask] - true_t[sequential_mask])) / n_seq
    elif loss_type == 'MSE':
        rate_loss = torch.sum((pred_t[sequential_mask] - true_t[sequential_mask]) ** 2) / n_seq
    else:
        error_message = f"Invalid loss_type: '{loss_type}. Must be 'MAE' or 'MSE'."
        logger.error(error_message)
        raise ValueError(error_message)
    
    # --- Apply secondary curvature if given ---
    
    if lambda_curve > 0.0 and prev_pred is not None:
        pred_t_2 = pred_t - prev_pred
        if loss_type == 'MAE':
            curve_loss = torch.sum(torch.abs(pred_t_2[sequential_mask])) / n_seq
        else:  # for MSE
            curve_loss = torch.sum(pred_t_2[sequential_mask] ** 2) / n_seq
        loss = loss + (lambda_smooth * rate_loss) + (lambda_curve * curve_loss)
    else:
        loss = loss + (lambda_smooth * rate_loss)
    
    return loss

def _gate_regulariser_schedule(epoch, num_epochs, m0=0.25, m_final=0.05,  # mean-alpha target decays from 0.25 → 0.05
                              T_anneal=40,  lam_mean0=0.10, lam_ent0=0.01):         
    """
    Returns (m_target, lambda_mean, lambda_ent) for this epoch.
    After T_anneal epochs, both lambdas ~ 0 (no regularisation).
    """
    # linear anneal 0->1 over first T_anneal epochs (fade penalties over t anneal epochs)
    a = min(1.0, epoch / max(1, T_anneal))
    m_t = m0 * (1 - a) + m_final * a
    lambda_mean = lam_mean0 * (1 - a)  # lam_mean0 is starting weight for mean-alpha penalty
    lambda_ent  = lam_ent0  * (1 - a)  # lam_ent0 is starting weight for entropy penalty
    return m_t, lambda_mean, lambda_ent

def _peak_weights_from_targets_unscaled(targets_unscaled, c=0.8, cap=3.0, eps=1e-6):
    """
    targets_unscaled: (M, 1) or (M,)
    Returns weights >= 1 that grow with |y - median| / IQR, capped to avoid instability.
    c controls how strongly to upweight tails.
    """
    y = targets_unscaled.view(-1)
    q25 = torch.quantile(y, 0.25)
    q75 = torch.quantile(y, 0.75)
    iqr = (q75 - q25).clamp_min(eps)
    med = torch.median(y)
    w = 1.0 + c * torch.abs(y - med) / iqr
    return torch.clamp(w, max=cap)

# NEW with tbptt wider windows
def _run_epoch_phase_NEW(epoch, num_epochs, all_timesteps_list, gradient_clip_max_norm, model, device, criterion,
                     optimizer, target_scaler, lambda_smooth, lambda_curve, loss_type, gwl_node_mean,
                     lambda_res_smooth, burn_in_steps: int = 0, is_training: bool = True):
    
    # Initialise model to correct mode
    model.train() if is_training else model.eval()
    description = 'Training' if is_training else 'Validation'
    
    # Fetch scheduler
    m_target, lambda_mean, lambda_ent = _gate_regulariser_schedule(epoch, num_epochs)
    
    # Setup tbptt
    tbptt_window = getattr(model, "tbptt_window", 1)
    accum_loss = None
    steps_in_window = 0

    total_loss = 0.0
    
    # --- Residual/FiLM epoch accumulators (scalars) ---
    res_abs_sum = 0.0
    res_count = 0

    gamma_dev_sum = 0.0
    beta_abs_sum = 0.0
    film_count_nodes = 0  # counts nodes (after mask) used for FiLM stats
    
    # init for relative contribution accumulators
    res_rel_sum = 0.0
    base_abs_sum = 0.0

    total_mae_unscaled = 0.0
    num_nodes_processed = 0
    num_predictions_processed = 0
    previous_residual = None
    
    # Initialise global LSTM state store at start of epoch (for Truncated Backpropagation Through Time (TBPTT))
    if model.run_LSTM:
        lstm_state_store = {
            'h': torch.zeros(model.num_layers_lstm, model.num_nodes, model.hidden_channels_lstm).to(device),
            'c': torch.zeros(model.num_layers_lstm, model.num_nodes, model.hidden_channels_lstm).to(device)
        }
    else:
        lstm_state_store = None

    # Iterate over all timesteps for training, applying spatial train_mask (+ tqdm for progress bars)
    # loop = tqdm(all_timesteps_list, desc=f"Epoch {epoch+1}/{num_epochs} [{description}]", leave=False)
    loop = tqdm(
        all_timesteps_list,
        desc=f"Epoch {epoch+1}/{num_epochs} [{description}]",
        leave=False,
        dynamic_ncols=True,
        mininterval=float(os.getenv("TQDM_MININTERVAL", "1.0")),  # seconds
        miniters=int(os.getenv("LOG_EVERY", "200")),  # batches
        disable=not sys.stdout.isatty(),  # disable under slurm logs
    )
    
    if target_scaler is not None:
        scale = torch.tensor(target_scaler.scale_, device=device)
        mean = torch.tensor(target_scaler.mean_, device=device)
    
    # Initialise previous prediction, mask, targets trackers
    previous_predictions = None
    previous_targets = None
    prev_pred = None
    prev_mask_for_loss_and_metrics = None
    
    #  Loop through and train/validate model
    with torch.set_grad_enabled(is_training): # Gradients enabled for training, disabled for validation
        # for data in loop:  # OLD: This now replaced with following two lines
        for t_idx, data in enumerate(loop):
            burnin = (t_idx < burn_in_steps)
            pred_t = None
            
            # Move data to device and run forward pass
            data = data.to(device)
            
            mask_attr = 'train_effective_mask' if is_training else 'val_effective_mask'
            if not hasattr(data, mask_attr):
                mask_attr = 'train_mask' if is_training else 'val_mask'  # Fallback shouldn't ever execute
            
            # Select the correct nodes to process for this phase (training or validation)
            mask_for_loss_and_metrics = getattr(data, mask_attr)
            if not mask_for_loss_and_metrics.any():
                logger.debug(f"Epoch {epoch+1}, Timestep {data.timestep.date()}: No "
                             f"{('Training' if is_training else 'Validation')} nodes. Skipping.")
                continue
            
            # Calculate the mask count and continue if not valid predictions in mask
            mask_count = mask_for_loss_and_metrics.sum().item()
            if mask_count == 0:
                continue
        
            x_full = data.x
            node_ids_in_current_timestep = data.node_id  # Global IDs for all nodes in x_full
            
            # --- Run Model Pass ---
            
            predictions_all_nodes, (h_new_for_current_nodes, c_new_for_current_nodes), returned_node_ids = model(
                x_full, data.edge_index, data.edge_attr, node_ids_in_current_timestep,lstm_state_store)
            
            # bind dbg ({} prevents UnboundLocalError)
            dbg = getattr(model, "last_debug", {})
            
            # --- Residual smoothing term (helps remove jitter from GAT component) ---
            
            res_smooth_term = None
            if "residual" in dbg and lambda_res_smooth > 0.0:
                r_t = dbg["residual"]
                if isinstance(r_t, torch.Tensor):
                    
                    # squeeze to [N]
                    r_curr = r_t.squeeze(-1) if r_t.dim() > 1 else r_t
                    if previous_residual is not None and isinstance(previous_residual, torch.Tensor):
                        r_prev = previous_residual.squeeze(-1) if previous_residual.dim() > 1 else previous_residual
                        
                        # align masks across timesteps
                        seq_mask = (mask_for_loss_and_metrics & prev_mask_for_loss_and_metrics) \
                                if prev_mask_for_loss_and_metrics is not None else mask_for_loss_and_metrics
                        if seq_mask.any():
                            r_diff = r_curr - r_prev
                            res_smooth_term = (torch.abs(r_diff[seq_mask]).mean()
                                            if loss_type == "MAE"
                                            else (r_diff[seq_mask] ** 2).mean())
                   
                    # track prev for next step (detached)
                    previous_residual = r_curr.detach()
            
            # --- Aggregate residual & FiLM diagnostics for this batch ---
            
            with torch.no_grad():
                residual = dbg.get("residual", None)
                if isinstance(residual, torch.Tensor):
                    r = residual
                    if r.dim() > 1: r = r.squeeze(-1)
                    r = r[mask_for_loss_and_metrics]
                    res_abs_sum += torch.abs(r).sum().item()
                    res_count   += r.numel()

                gamma = dbg.get("gamma", None)
                beta  = dbg.get("beta",  None)
                if isinstance(gamma, torch.Tensor) and isinstance(beta, torch.Tensor):
                    g_dev = torch.abs(gamma - 1.0).mean(dim=1)
                    b_abs = torch.abs(beta).mean(dim=1)
                    g_dev_m = g_dev[mask_for_loss_and_metrics]
                    b_abs_m = b_abs[mask_for_loss_and_metrics]
                    gamma_dev_sum   += g_dev_m.sum().item()
                    beta_abs_sum    += b_abs_m.sum().item()
                    film_count_nodes += g_dev_m.numel()

                baseline = dbg.get("baseline", None)
                if isinstance(residual, torch.Tensor) and isinstance(baseline, torch.Tensor):
                    r = residual
                    if r.dim() > 1: r = r.squeeze(-1)
                    yb = baseline
                    if yb.dim() > 1: yb = yb.squeeze(-1)
                    res_rel_sum  += torch.abs(r[mask_for_loss_and_metrics]).sum().item()
                    base_abs_sum += torch.abs(yb[mask_for_loss_and_metrics]).sum().item()

            # --- Update persistent LSTM state ---
            
            if model.run_LSTM:
                lstm_state_store['h'][:, returned_node_ids, :] = (
                    h_new_for_current_nodes if (is_training and tbptt_window > 1) else h_new_for_current_nodes.detach()
                )
                lstm_state_store['c'][:, returned_node_ids, :] = (
                    c_new_for_current_nodes if (is_training and tbptt_window > 1) else c_new_for_current_nodes.detach()
                )
            
            # -------- Loss & regularisers --------

            loss = None
            if not burnin:
                
                # Compute loss (on relevant nodes only)
                # loss = criterion(predictions_all_nodes[mask_for_loss_and_metrics], data.y[mask_for_loss_and_metrics])
                
                y_pred_m = predictions_all_nodes[mask_for_loss_and_metrics]
                y_true_m = data.y[mask_for_loss_and_metrics]

                # compute unscaled targets for weighting
                if target_scaler is not None:
                    y_true_unscaled = y_true_m * scale + mean
                else:
                    y_true_unscaled = y_true_m

                weights = _peak_weights_from_targets_unscaled(y_true_unscaled, c=0.8, cap=3.0).detach()

                # base error
                if loss_type == "MAE":
                    err = torch.abs(y_pred_m - y_true_m).view(-1)
                else:
                    err = (y_pred_m - y_true_m).pow(2).view(-1)

                loss = (weights * err).mean()
                
                # --- If Smoothness Loss given then apply it ---
                
                # Compute first diff with respect to previous timestep (where available)
                pred_t = (predictions_all_nodes - previous_predictions) if previous_predictions is not None else None  # (1)
                
                if lambda_smooth > 0.0 and previous_predictions is not None: 
                    # Align with previous node
                    sequential_mask = mask_for_loss_and_metrics & prev_mask_for_loss_and_metrics  # (2)
                    if sequential_mask.any():
                        loss = _calc_smooth_curvature_loss(
                            loss, data, previous_targets, sequential_mask, loss_type, pred_t,
                            lambda_curve, prev_pred, lambda_smooth
                        )
                        
                # ---- Gate regularisers (mean-alpha target + entropy) ----
                
                alpha = dbg.get("alpha", None)
                if (alpha is not None) and (lambda_mean > 0 or lambda_ent > 0):
                    # restrict to nodes in the current loss mask
                    alpha_m = alpha[mask_for_loss_and_metrics]
                    if alpha_m.numel() > 0:
                        # mean-alpha penalty towards m_target
                        L_mean = (alpha_m.mean() - m_target).pow(2)

                        # entropy penalty (push away from 0/1 early): mean[ alpha log alpha + (1-alpha) log(1-alpha) ]
                        eps = 1e-6
                        a_clamped = alpha_m.clamp(eps, 1 - eps)
                        L_ent = (a_clamped * (a_clamped + eps).log() + (1 - a_clamped) * (1 - a_clamped + eps).log()).mean()
                        loop.set_postfix({**loop.postfix, 'alpha_mean': f"{alpha_m.mean().item():.2f}"})

                        loss = loss + lambda_mean * L_mean + lambda_ent * L_ent
                        
                # ---- Head calibration regulariser (very small) ----
                if is_training:
                    calib_reg = 1e-4 * (
                        (model.tau_lstm - 1.0).pow(2) + model.bias_lstm.pow(2)
                        + (model.tau_gat - 1.0).pow(2) + model.bias_gat.pow(2)
                    )
                    loss = loss + calib_reg

                total_loss += loss.item() * mask_count  # Scale by number of predictions in this batch so epoch avg is per-prediction
                num_predictions_processed += mask_count  # num_predictions is num node,timestep pairs

                # --- Compute unscaled MAE (mAOD) for interpretability ---
                
                if target_scaler is not None:
                    preds_orig = predictions_all_nodes[mask_for_loss_and_metrics] * scale + mean
                    targets_orig = data.y[mask_for_loss_and_metrics] * scale + mean
                    batch_mae = torch.mean(torch.abs(preds_orig - targets_orig)).item()
                    total_mae_unscaled += batch_mae * mask_count
                    num_nodes_processed += mask_count
                
            # -- Perform optimisation when in training phase ---
            
            # if is_training:
            #     optimizer.zero_grad()
            #     loss.backward()
                
            #     if gradient_clip_max_norm is not None:
            #         torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_max_norm)
            #     optimizer.step()

            #     # Update training specific progress bar
            #     postfix = {'loss': f"{loss.item():.4f}", 'lr':   f"{optimizer.param_groups[0]['lr']:.6f}"}
            #     if target_scaler is not None:
            #         postfix['mae'] = f"{batch_mae:.2f}"
            #     loop.set_postfix(postfix)
            
            # postfix update handler
            n_batches = len(all_timesteps_list)
            # should_update = (t_idx % LOG_EVERY == 0) or (t_idx + 1 == n_batches)
            should_update = (t_idx % int(os.getenv("LOG_EVERY", "200")) == 0) or (t_idx + 1 == n_batches)
            
            if is_training and not burnin:
                # accumulate loss within the TBPTT window
                accum_loss = loss if accum_loss is None else (accum_loss + loss)
                steps_in_window += 1

                is_boundary = (steps_in_window % tbptt_window == 0)  # also handle end-of-epoch below

                if is_boundary:
                    optimizer.zero_grad()
                    denom = steps_in_window if tbptt_window > 1 else 1
                    (accum_loss / denom).backward()
                    if gradient_clip_max_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_max_norm)
                    optimizer.step()

                    # detach LSTM states across ALL nodes at the boundary to truncate history
                    if model.run_LSTM and tbptt_window > 1:
                        lstm_state_store['h'] = lstm_state_store['h'].detach()
                        lstm_state_store['c'] = lstm_state_store['c'].detach()

                    # reset window accumulators
                    accum_loss = None
                    steps_in_window = 0

                # progress bar (use the current step loss for readability)
                if should_update:
                    postfix = {'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"}
                    if target_scaler is not None:
                        postfix['mae'] = f"{batch_mae:.2f}"
                    loop.set_postfix(postfix)
            
            elif is_training and burnin:
                if should_update:
                    loop.set_postfix({'burn-in': f"{t_idx+1}/{burn_in_steps}"})
            
            # store trackers for next timestep (detach to avoid exploding computation)
            previous_predictions = predictions_all_nodes.detach()
            previous_targets = data.y.detach()
            prev_mask_for_loss_and_metrics = mask_for_loss_and_metrics.clone()
            prev_pred = pred_t.detach() if pred_t is not None else None
        
        # after the for-loop, still inside with torch.set_grad_enabled(is_training):
        if is_training and accum_loss is not None:
            optimizer.zero_grad()
            denom = steps_in_window if tbptt_window > 1 else 1
            (accum_loss / denom).backward()
            if gradient_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_max_norm)
            optimizer.step()
            if model.run_LSTM and tbptt_window > 1:
                lstm_state_store['h'] = lstm_state_store['h'].detach()
                lstm_state_store['c'] = lstm_state_store['c'].detach()

    # Calculate average loss over all timesteps processed in training (where train_mask = True)
    avg_loss = total_loss / num_predictions_processed if num_predictions_processed > 0 else float('nan')
    avg_mae_unscaled = total_mae_unscaled / num_nodes_processed if num_nodes_processed > 0 else float('nan')

    # --- Summarise residual/FiLM stats for the epoch ---
    
    residual_abs_mean = (res_abs_sum / res_count) if res_count > 0 else float('nan')
    gamma_dev_mean = (gamma_dev_sum / film_count_nodes) if film_count_nodes > 0 else float('nan')
    beta_abs_mean = (beta_abs_sum  / film_count_nodes) if film_count_nodes  > 0 else float('nan')

    logger.info(
        f"[{description}] Epoch {epoch+1}/{num_epochs} | "
        f"residual_abs_mean: {residual_abs_mean:.6f} | "
        f"gamma_dev_mean: {gamma_dev_mean:.6f} | "
        f"beta_abs_mean: {beta_abs_mean:.6f}"
    )
    
    gat_rel_contrib = (res_rel_sum / base_abs_sum) if base_abs_sum > 0 else float('nan')
    logger.info(f"[{description}] Epoch {epoch+1}/{num_epochs} | "
                f"GAT rel contrib: {gat_rel_contrib:.2%}")

    return avg_loss, avg_mae_unscaled

# Stable version tested on multiple
def _run_epoch_phase(epoch, num_epochs, all_timesteps_list, gradient_clip_max_norm, model, device, criterion,
                     optimizer, target_scaler, lambda_smooth, lambda_curve, loss_type, gwl_node_mean,
                     lambda_res_smooth, burn_in_steps: int = 0, is_training: bool = True):
    
    # Initialise model to correct mode
    model.train() if is_training else model.eval()
    description = 'Training' if is_training else 'Validation'
    
    # Fetch scheduler
    m_target, lambda_mean, lambda_ent = _gate_regulariser_schedule(epoch, num_epochs)
    
    total_loss = 0.0
    
    # --- Residual/FiLM epoch accumulators (scalars) ---
    res_abs_sum = 0.0
    res_count = 0

    gamma_dev_sum = 0.0
    beta_abs_sum = 0.0
    film_count_nodes = 0  # counts nodes (after mask) used for FiLM stats
    
    # init for relative contribution accumulators
    res_rel_sum = 0.0
    base_abs_sum = 0.0

    total_mae_unscaled = 0.0
    num_nodes_processed = 0
    num_predictions_processed = 0
    previous_residual = None
    
    # Initialise global LSTM state store at start of epoch (for Truncated Backpropagation Through Time (TBPTT))
    if model.run_LSTM:
        lstm_state_store = {
            'h': torch.zeros(model.num_layers_lstm, model.num_nodes, model.hidden_channels_lstm).to(device),
            'c': torch.zeros(model.num_layers_lstm, model.num_nodes, model.hidden_channels_lstm).to(device)
        }
    else:
        lstm_state_store = None

    # Iterate over all timesteps for training, applying spatial train_mask (+ tqdm for progress bars)
    # loop = tqdm(all_timesteps_list, desc=f"Epoch {epoch+1}/{num_epochs} [{description}]", leave=False)
    loop = tqdm(
        all_timesteps_list,
        desc=f"Epoch {epoch+1}/{num_epochs} [{description}]",
        leave=False,
        dynamic_ncols=True,
        mininterval=float(os.getenv("TQDM_MININTERVAL", "1.0")),  # seconds
        miniters=int(os.getenv("LOG_EVERY", "200")),  # batches
        disable=not sys.stdout.isatty(),  # disable under slurm logs
    )
    
    if target_scaler is not None:
        scale = torch.tensor(target_scaler.scale_, device=device)
        mean = torch.tensor(target_scaler.mean_, device=device)
    
    # Initialise previous prediction, mask, targets trackers
    previous_predictions = None
    previous_targets = None
    prev_pred = None
    prev_mask_for_loss_and_metrics = None
    
    #  Loop through and train/validate model
    with torch.set_grad_enabled(is_training): # Gradients enabled for training, disabled for validation
        # for data in loop:  # OLD: This now replaced with following two lines
        for t_idx, data in enumerate(loop):
            burnin = (t_idx < burn_in_steps)
            pred_t = None 
            
            # Move data to device and run forward pass
            data = data.to(device)
            
            mask_attr = 'train_effective_mask' if is_training else 'val_effective_mask'
            if not hasattr(data, mask_attr):
                mask_attr = 'train_mask' if is_training else 'val_mask'  # Fallback shouldn't ever execute
            
            # Select the correct nodes to process for this phase (training or validation)
            mask_for_loss_and_metrics = getattr(data, mask_attr)
            if not mask_for_loss_and_metrics.any():
                logger.debug(f"Epoch {epoch+1}, Timestep {data.timestep.date()}: No "
                             f"{('Training' if is_training else 'Validation')} nodes. Skipping.")
                continue
            
            # Calculate the mask count and continue if not valid predictions in mask
            mask_count = mask_for_loss_and_metrics.sum().item()
            if mask_count == 0:
                continue
        
            x_full = data.x
            node_ids_in_current_timestep = data.node_id  # Global IDs for all nodes in x_full
            
            # --- Run Model Pass ---
            
            predictions_all_nodes, (h_new_for_current_nodes, c_new_for_current_nodes), returned_node_ids = model(
                x_full, data.edge_index, data.edge_attr, node_ids_in_current_timestep,lstm_state_store)
            
            # bind dbg ({} prevents UnboundLocalError)
            dbg = getattr(model, "last_debug", {})
            
            # --- Residual smoothing term (helps remove jitter from GAT component) ---
            
            if "residual" in dbg and lambda_res_smooth > 0.0:
                r_t = dbg["residual"]
                if isinstance(r_t, torch.Tensor) and previous_residual is not None:
                    r = r_t
                    if r.dim() > 1:
                        r = r.squeeze(-1)
                    pr = previous_residual
                    if pr.dim() > 1:
                        pr = pr.squeeze(-1)
                    r_diff = r - pr
                    res_smooth = (torch.abs(r_diff[mask_for_loss_and_metrics]).mean()
                                  if loss_type == "MAE"
                                  else (r_diff[mask_for_loss_and_metrics] ** 2).mean())
                    res_smooth_term = res_smooth
                else:
                    res_smooth_term = None
                previous_residual = r_t.detach() if isinstance(r_t, torch.Tensor) else previous_residual
            else:
                res_smooth_term = None
            
            # --- Aggregate residual & FiLM diagnostics for this batch ---
            
            with torch.no_grad():
                residual = dbg.get("residual", None)
                if isinstance(residual, torch.Tensor):
                    r = residual
                    if r.dim() > 1: r = r.squeeze(-1)
                    r = r[mask_for_loss_and_metrics]
                    res_abs_sum += torch.abs(r).sum().item()
                    res_count   += r.numel()

                gamma = dbg.get("gamma", None)
                beta  = dbg.get("beta",  None)
                if isinstance(gamma, torch.Tensor) and isinstance(beta, torch.Tensor):
                    g_dev = torch.abs(gamma - 1.0).mean(dim=1)
                    b_abs = torch.abs(beta).mean(dim=1)
                    g_dev_m = g_dev[mask_for_loss_and_metrics]
                    b_abs_m = b_abs[mask_for_loss_and_metrics]
                    gamma_dev_sum   += g_dev_m.sum().item()
                    beta_abs_sum    += b_abs_m.sum().item()
                    film_count_nodes += g_dev_m.numel()

                baseline = dbg.get("baseline", None)
                if isinstance(residual, torch.Tensor) and isinstance(baseline, torch.Tensor):
                    r = residual
                    if r.dim() > 1: r = r.squeeze(-1)
                    yb = baseline
                    if yb.dim() > 1: yb = yb.squeeze(-1)
                    res_rel_sum  += torch.abs(r[mask_for_loss_and_metrics]).sum().item()
                    base_abs_sum += torch.abs(yb[mask_for_loss_and_metrics]).sum().item()

            # --- Update persistent LSTM state ---
            
            # Detach hidden states for Truncated Backpropagation Through Time (BPTT)
            if model.run_LSTM:
                # Update the global lstm_state_store using the specific node IDs that were processed in this timestep
                # And detach the new states to prevent backprop through previous timesteps' computations.
                lstm_state_store['h'][:, returned_node_ids, :] = h_new_for_current_nodes.detach()
                lstm_state_store['c'][:, returned_node_ids, :] = c_new_for_current_nodes.detach()
            
            # -------- Loss & regularisers --------

            loss = None
            if not burnin:
            
                # Compute loss (on relevant nodes only)
                # loss = criterion(predictions_all_nodes[mask_for_loss_and_metrics], data.y[mask_for_loss_and_metrics])
                
                y_pred_m = predictions_all_nodes[mask_for_loss_and_metrics]
                y_true_m = data.y[mask_for_loss_and_metrics]

                # compute unscaled targets for weighting
                if target_scaler is not None:
                    y_true_unscaled = y_true_m * scale + mean
                else:
                    y_true_unscaled = y_true_m

                weights = _peak_weights_from_targets_unscaled(y_true_unscaled, c=0.8, cap=3.0).detach()

                # base error
                if loss_type == "MAE":
                    err = torch.abs(y_pred_m - y_true_m).view(-1)
                else:
                    err = (y_pred_m - y_true_m).pow(2).view(-1)

                loss = (weights * err).mean()
                
                # --- If Smoothness Loss given then apply it ---
                
                # Compute first diff with respect to previous timestep (where available)
                pred_t = (predictions_all_nodes - previous_predictions) if previous_predictions is not None else None  # (1)
                
                if lambda_smooth > 0.0 and previous_predictions is not None: 
                    # Align with previous node
                    sequential_mask = mask_for_loss_and_metrics & prev_mask_for_loss_and_metrics  # (2)
                    if sequential_mask.any():
                        loss = _calc_smooth_curvature_loss(
                            loss, data, previous_targets, sequential_mask, loss_type, pred_t,
                            lambda_curve, prev_pred, lambda_smooth
                        )
                        
                # ---- Gate regularisers (mean-alpha target + entropy) ----
                
                alpha = dbg.get("alpha", None)
                if (alpha is not None) and (lambda_mean > 0 or lambda_ent > 0):
                    # restrict to nodes in the current loss mask
                    alpha_m = alpha[mask_for_loss_and_metrics]
                    if alpha_m.numel() > 0:
                        # mean-alpha penalty towards m_target
                        L_mean = (alpha_m.mean() - m_target).pow(2)

                        # entropy penalty (push away from 0/1 early): mean[ alpha log alpha + (1-alpha) log(1-alpha) ]
                        eps = 1e-6
                        a_clamped = alpha_m.clamp(eps, 1 - eps)
                        L_ent = (a_clamped * (a_clamped + eps).log() + (1 - a_clamped) * (1 - a_clamped + eps).log()).mean()
                        loop.set_postfix({**loop.postfix, 'alpha_mean': f"{alpha_m.mean().item():.2f}"})

                        loss = loss + lambda_mean * L_mean + lambda_ent * L_ent
                        
                # ---- Head calibration regulariser (very small) ----
                if is_training:
                    calib_reg = 1e-4 * (
                        (model.tau_lstm - 1.0).pow(2) + model.bias_lstm.pow(2)
                        + (model.tau_gat - 1.0).pow(2) + model.bias_gat.pow(2)
                    )
                    loss = loss + calib_reg

                total_loss += loss.item() * mask_count  # Scale by number of predictions in this batch so epoch avg is per-prediction
                num_predictions_processed += mask_count  # num_predictions is num node,timestep pairs

                # --- Compute unscaled MAE (mAOD) for interpretability ---
                
                if target_scaler is not None:
                    preds_orig = predictions_all_nodes[mask_for_loss_and_metrics] * scale + mean
                    targets_orig = data.y[mask_for_loss_and_metrics] * scale + mean
                    batch_mae = torch.mean(torch.abs(preds_orig - targets_orig)).item()
                    total_mae_unscaled += batch_mae * mask_count
                    num_nodes_processed += mask_count
            
            # -- Perform optimisation when in training phase ---
            
            # postfix update handler
            n_batches = len(all_timesteps_list)
            # should_update = (t_idx % LOG_EVERY == 0) or (t_idx + 1 == n_batches)
            should_update = (t_idx % int(os.getenv("LOG_EVERY", "200")) == 0) or (t_idx + 1 == n_batches)
            
            if is_training and not burnin:
                optimizer.zero_grad()
                loss.backward()
                
                if gradient_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_max_norm)
                optimizer.step()

                # progress bar (use the current step loss for readability)
                if should_update:
                    postfix = {'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"}
                    if target_scaler is not None:
                        postfix['mae'] = f"{batch_mae:.2f}"
                    loop.set_postfix(postfix)
            
            elif is_training and burnin:
                if should_update:
                    loop.set_postfix({'burn-in': f"{t_idx+1}/{burn_in_steps}"})
            
            # store trackers for next timestep (detach to avoid exploding computation)
            previous_predictions = predictions_all_nodes.detach()
            previous_targets = data.y.detach()
            prev_mask_for_loss_and_metrics = mask_for_loss_and_metrics.clone()
            prev_pred = pred_t.detach() if pred_t is not None else None
        
    # Calculate average loss over all timesteps processed in training (where train_mask = True)
    avg_loss = total_loss / num_predictions_processed if num_predictions_processed > 0 else float('nan')
    avg_mae_unscaled = total_mae_unscaled / num_nodes_processed if num_nodes_processed > 0 else float('nan')

    # --- Summarise residual/FiLM stats for the epoch ---
    
    residual_abs_mean = (res_abs_sum / res_count) if res_count > 0 else float('nan')
    gamma_dev_mean = (gamma_dev_sum / film_count_nodes) if film_count_nodes > 0 else float('nan')
    beta_abs_mean = (beta_abs_sum  / film_count_nodes) if film_count_nodes  > 0 else float('nan')

    logger.info(
        f"[{description}] Epoch {epoch+1}/{num_epochs} | "
        f"residual_abs_mean: {residual_abs_mean:.6f} | "
        f"gamma_dev_mean: {gamma_dev_mean:.6f} | "
        f"beta_abs_mean: {beta_abs_mean:.6f}"
    )
    
    gat_rel_contrib = (res_rel_sum / base_abs_sum) if base_abs_sum > 0 else float('nan')
    logger.info(f"[{description}] Epoch {epoch+1}/{num_epochs} | "
                f"GAT rel contrib: {gat_rel_contrib:.2%}")

    return avg_loss, avg_mae_unscaled

def _generate_model_filename(config, catchment):
    """
    Generates a unique filename for the trained model based on its hyperparameters
    and a timestamp.

    Args:
        config_for_catchment (dict): The configuration dictionary for the specific catchment
                                     (e.g., config['eden']).

    Returns:
        str: A unique filename (e.g., "model_20250723-083000_GATH8_GATD0-5_..._LR0-0005.pt")
    """
    model_params = config[catchment]["model"]["architecture"]
    training_settings = config[catchment]["training"]

    # Helper to format float values for filenames (replaces '.' with '-' to avoid file path issues)
    format_float = lambda x: str(x).replace('.', '-')

    # Extracting and formatting parameters for the filename TODO: ADD RUN GAT / LSTM AS CLEAR MARKERS!!!
    # Using abbreviations for conciseness
    params_parts = [
        f"GAT{model_params['run_GAT']}",
        f"LSTM{model_params['run_LSTM']}",
        f"GATH{model_params['heads_gat']}",
        f"GATD{format_float(model_params['dropout_gat'])}",
        f"GATHC{model_params['hidden_channels_gat']}",
        f"GATOC{model_params['out_channels_gat']}",
        f"GATNL{model_params['num_layers_gat']}",
        f"LSTHC{model_params['hidden_channels_lstm']}",
        f"LSTNL{model_params['num_layers_lstm']}",
        f"OUTD{model_params['output_dim']}",
        f"LR{format_float(model_params['adam_learning_rate'])}",
        f"WD{format_float(model_params['adam_weight_decay'])}",
        f"SM{format_float(model_params['lambda_smooth'])}"
    ]

    training_parts = [
        f"E{training_settings['num_epochs']}",
        f"ESP{training_settings['early_stopping_patience']}",
        f"LRSF{format_float(training_settings['lr_scheduler_factor'])}",
        f"LRSP{training_settings['lr_scheduler_patience']}",
        f"MINLR{format_float(training_settings['min_lr'])}",
        f"LD{format_float(training_settings['loss_delta'])}",
        f"GCMN{format_float(training_settings['gradient_clip_max_norm'])}"
    ]

    # Combine all parameter parts into a single string, separated by underscores
    param_string = "_".join(params_parts + training_parts)

    # Add a timestamp for absolute uniqueness
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Final filename structure
    filename = f"model_{timestamp}_{param_string}"  # No specific extension -> added after for all cases
    return filename

def _load_node_means(scalers_dir, target_scaler, device):
    # Load node ids and means and make vector
    all_node_ids = joblib.load(os.path.join(scalers_dir, "all_node_ids.joblib"))
    gwl_mean_by_node = joblib.load(os.path.join(scalers_dir, "gwl_mean_by_node.joblib"))
    gwl_mean_vec = np.array([gwl_mean_by_node[nid] for nid in all_node_ids], dtype=np.float32)
    
    # gwl_mean to target space
    scale = target_scaler.scale_[0]
    mean = target_scaler.mean_[0]
    gwl_mean_targetspace = (gwl_mean_vec - mean) / scale
    
    # To torch tensor (on same device)
    gwl_node_mean = torch.tensor(gwl_mean_targetspace, dtype=torch.float32, device=device)
    
    return gwl_node_mean

def run_training_and_validation(num_epochs: int, early_stopping_patience: int, lr_scheduler_factor: float,
                                lr_scheduler_patience: int, min_lr: float, gradient_clip_max_norm: float,
                                model_save_dir: str, loss_delta: float, verbose: bool, catchment: str,
                                model: torch.nn.Module, device: torch.device, optimizer: torch.optim.Optimizer,
                                criterion: torch.nn.modules.loss._Loss, all_timesteps_list: list, scalers_dir: str,
                                config):
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
    
    # --- Get dynamic model save path --- 

    # Generate the unique filename for this training run
    pt_model_dir = os.path.join(model_save_dir, "pt_model")
    os.makedirs(pt_model_dir, exist_ok=True)
    
    dynamic_base_filename = _generate_model_filename(config, catchment)
    dynamic_model_filename = f"{dynamic_base_filename}.pt"
    model_save_path = os.path.join(pt_model_dir, dynamic_model_filename)

    logger.info(f"Model for {catchment} will be saved to: {model_save_path}")
    
    # --- Get Target Scaler ---
    
    target_scaler_path = os.path.join(scalers_dir, "target_scaler.pkl")
    try:
        target_scaler = joblib.load(target_scaler_path) 
        logger.info(f"Successfully loaded target scaler from: {target_scaler_path}")
    except Exception as e:
        logger.error(f"Error loading target scaler from {target_scaler_path}: {e}")
        target_scaler = None
        
    start_time = time.time()
    best_val = float("inf")
    best_epoch = -1
    
    # Get node means
    gwl_node_mean = _load_node_means(scalers_dir, target_scaler, device)
    
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
    
    # Also initialise lists to store unscaled MAE for plotting and analysis
    train_maes_unscaled = []
    val_maes_unscaled = []
    
    # Retrieve smoothing and curvature params alongside loss type
    lambda_smooth = config[catchment]["model"]["architecture"]["lambda_smooth"]
    lambda_curve = config[catchment]["model"]["architecture"]["lambda_curve"]
    lambda_res_smooth = config[catchment]["model"]["architecture"]["lambda_res_smooth"]
    loss_type = config[catchment]["training"]["loss"]
    burn_in_steps = int(config[catchment]["training"].get("burn_in_steps", 0))  # Incase not given assume none
    tbptt_window = config[catchment]["model"]["architecture"]["tbptt_window"]
    epoch_runner = _run_epoch_phase_NEW if tbptt_window > 1 else _run_epoch_phase
    
    # --- Run full training and validation loop ---

    for epoch in range(num_epochs):
        
        # --- TRAINING PHASE ---
        
        avg_train_loss, avg_train_mae_unscaled = epoch_runner(epoch, num_epochs, all_timesteps_list,
                                                                  gradient_clip_max_norm, model, device,
                                                                  criterion, optimizer, target_scaler,
                                                                  lambda_smooth, lambda_curve, loss_type,
                                                                  gwl_node_mean, lambda_res_smooth,
                                                                  burn_in_steps, is_training=True)
        train_losses.append(avg_train_loss)
        train_maes_unscaled.append(avg_train_mae_unscaled) 
        
        # --- VALIDATION PHASE ---
        
        avg_val_loss, avg_val_mae_unscaled = epoch_runner(epoch, num_epochs, all_timesteps_list,
                                                              gradient_clip_max_norm, model, device,
                                                              criterion, optimizer, target_scaler,
                                                              lambda_smooth, lambda_curve, loss_type,
                                                              gwl_node_mean, lambda_res_smooth,
                                                              burn_in_steps, is_training=False)
        val_losses.append(avg_val_loss)
        val_maes_unscaled.append(avg_val_mae_unscaled)
        
        # --- Light-weight per-epoch stats (best tracking, ETA, LR) ---
        if math.isfinite(avg_val_loss) and avg_val_loss < best_val:
            best_val = float(avg_val_loss)
            best_epoch = epoch

        elapsed = time.time() - start_time
        progress = (epoch + 1) / num_epochs
        eta_min = ((elapsed / progress) - elapsed) / 60.0 if progress > 0 else 0.0
        curr_lr = optimizer.param_groups[0]["lr"]
        since_best = (epoch - best_epoch) if best_epoch >= 0 else -1

        # Compact per-epoch summary line (losses + ES + LR + ETA)
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
            f"best={best_val:.4f} Δbest={since_best if since_best>=0 else '—'} "
            f"ES {early_stopper.counter}/{early_stopper.patience} "
            f"lr={curr_lr:.2e} ETA~{max(0.0, eta_min):.1f}m"
        )
        
        # Log Epoch Results (Standardise and Invert to Raw)
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Train MAE (GWL): {avg_train_mae_unscaled:.2f}"
                    f" | Val MAE (GWL): {avg_val_mae_unscaled:.2f}")
                    
        # Learning rate scheduler step -> adjust lr based on val loss
        # lr_scheduler.step(avg_val_loss)

        # --- Save Model if best so far / Early stop if not improving ---
        
        # Run Early Stopping __call__ -> Stop if val loss not improved for {patience} epochs
        #early_stopper(avg_val_loss, model)
        
        warmup_epochs = config[catchment]["training"].get("early_stop_warmup", 0)
        
        # Ignore and just log if infinite -> ensure that a weird NaN can't poison EarlyStopping internal state
        if math.isfinite(avg_val_loss):
            lr_scheduler.step(avg_val_loss)
            if epoch + 1 >= warmup_epochs:
                early_stopper(avg_val_loss, model)
        else:
            logger.warning(f"Epoch {epoch+1}: validation loss is not finite; "
                        "skipping LR scheduler and early stopping update for this epoch.")
        
        # Report per-epoch metrics to Ray Tune
        try:
            session.report({
                "epoch": epoch + 1,
                "train_loss": float(avg_train_loss),
                "val_loss": float(avg_val_loss),
                "best_val": float(best_val),
                "since_best": int(since_best),
                "es_cnt": int(early_stopper.counter),
                "lr": float(curr_lr),
                "time_s": int(elapsed),
                "eta_min": float(max(0.0, eta_min)),
            })
        
        # Allowt o pass reporting when not in ray tune 
        except Exception:
            pass

        if early_stopper.early_stop:  # Break if early stop call found to return True
            logger.info(f"Early stopping triggered at epoch {epoch+1}.")
            break
    
    # Load the best model weights found during training (saved in EarlyStopping + made more defensive due to failures)
    if os.path.exists(early_stopper.path):
        try:
            model.load_state_dict(torch.load(early_stopper.path, map_location=device))
            logger.info(f"[Summary] Best val={best_val:.6f} at epoch {best_epoch+1}; "
                        f"checkpoint={early_stopper.path}")
        except Exception as e:
            logger.warning(f"[Summary] Checkpoint exists but failed to load ({e}); using final weights.")
    else:
        logger.warning("[Summary] No checkpoint written; using final weights.")

    logger.info(f"--- Training Loop Finished ---\n")
    
    return train_losses, val_losses

def save_train_val_losses(output_analysis_dir, train_losses, val_losses, config, catchment):
    
    # Confirm all necessary dirs exist
    pt_losses_dir = os.path.join(output_analysis_dir, "pt_losses")
    os.makedirs(pt_losses_dir, exist_ok=True)
    npy_losses_dir = os.path.join(output_analysis_dir, "npy_losses")
    os.makedirs(npy_losses_dir, exist_ok=True)
    csv_losses_dir = os.path.join(output_analysis_dir, "csv_losses")
    os.makedirs(csv_losses_dir, exist_ok=True)
    
    # Generate the unique filename for this training run
    dynamic_base_filename = _generate_model_filename(config, catchment)

    logger.info(f"Model Losses for {catchment} will be saved to base: {dynamic_base_filename}")

    # Save train_losses and val_losses as .pt files
    torch.save(train_losses, os.path.join(pt_losses_dir, f"{dynamic_base_filename}_train_loss.pt"))
    torch.save(val_losses, os.path.join(pt_losses_dir, f"{dynamic_base_filename}_val_loss.pt"))
    logger.info(f"Training and validation losses saved to {pt_losses_dir} as .pt files.")

    # Save train_losses and val_losses as .npy files
    np.save(os.path.join(npy_losses_dir,  f"{dynamic_base_filename}_train_loss.npy"), np.array(train_losses))
    np.save(os.path.join(npy_losses_dir, f"{dynamic_base_filename}_val_loss.npy"), np.array(val_losses))
    logger.info(f"Training and validation losses saved to {output_analysis_dir} as .npy files.")

    # Save train_losses and val_losses as CSV files
    pd.DataFrame(train_losses).to_csv(os.path.join(csv_losses_dir,f"{dynamic_base_filename}_train_loss.csv"), index=False)
    pd.DataFrame(val_losses).to_csv(os.path.join(csv_losses_dir, f"{dynamic_base_filename}_val_loss.csv"), index=False)
    logger.info(f"Training and validation losses saved to {output_analysis_dir} as .csv files.\n")
    
    # Return npy filepaths
    relative_train_path = os.path.join(npy_losses_dir,  f"{dynamic_base_filename}_train_loss.npy")
    relative_val_path = os.path.join(npy_losses_dir,  f"{dynamic_base_filename}_val_loss.npy")
    
    # Save npy filepaths
    loss_paths = (relative_train_path, relative_val_path)
    loss_paths_filename = os.path.join(npy_losses_dir, f"{dynamic_base_filename}_loss_paths.joblib")
    joblib.dump(loss_paths, loss_paths_filename)
    logger.info(f"Loss file paths saved to: {loss_paths_filename}")
    
    return relative_train_path, relative_val_path
