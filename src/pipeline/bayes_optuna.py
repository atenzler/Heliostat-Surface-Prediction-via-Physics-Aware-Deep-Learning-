"""
bayes_optuna.py

Bayesian hyperparameter optimization (HPO) for DeepLARTS using Optuna.

- Defines an Optuna objective that trains DeepLARTS with sampled penalty weights
- Supports penalties for z-constraint, curvature, edge dip, tilt, and z-range
- Logs results to TensorBoard and SQLite database
- Saves best trial results as JSON

Usage:
- Run locally on 1 GPU
- Each trial trains a full model with different sampled hyperparameters
- Typical usage: 25–50 trials per GPU

References:
- Optuna: https://optuna.org/
"""


import optuna
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import torch
import json
import os
import argparse
import shutil
from datetime import datetime
from PhysConUL_DownCont.updated_code.impl import train_dnn
from PhysConUL_DownCont.updated_code.func import create_folder_struct_cluster
from PhysConUL_DownCont.updated_code.run_in_functions_cluster import build_model, setup_data_and_scenario

def objective(trial, log_dir, final_target):
    optimizer = "Adam"
    criterion = "MAE"
    scheduler_type = None
    learning_rate = 0.001
    weight_decay = 0
    batch_size = 8
    max_epochs = 100
    patience_epochs = 20
    patience_loss = 1e-4
    num_inputs = 192
    keys = [f"AUG{i}" for i in range(1, num_inputs + 1)]

    # Sample hyperparameters
    z_constraint_weight = trial.suggest_float("z_constraint_weight", 1e2, 1e6, log=True)
    output_weight_decay = trial.suggest_float("output_weight_decay", 1e-2, 100, log=True)
    edge_dip_reward_weight = trial.suggest_float("edge_dip_reward_weight", 1e2, 1e6, log=True)
    dip_margin = trial.suggest_float("dip_margin", 1e-5, 1e-3, log=True)
    curvature_penalty_weight = trial.suggest_float("curvature_penalty_weight", 1e3, 1e7, log=True)
    ref_curvature = trial.suggest_float("ref_curvature", 1e-5, 1e-3, log=True)
    tilt_penalty_weight = trial.suggest_float("tilt_penalty_weight", 1e2, 1e5)
    zrange_penalty_weight = trial.suggest_float("zrange_penalty_weight", 1.0, 100.0)
    max_zrange = trial.suggest_float("max_zrange", 1e-5, 5e-4, log=True)

    decay_output_weight_decay = True
    linear_decay = True
    cos_weight_decay = False

    print("Hyperparameters being tested:")
    print({
        "z_constraint_weight": z_constraint_weight,
        "output_weight_decay": output_weight_decay,
    })

    dirs = create_folder_struct_cluster(suffix=f"trial_{trial.number}", base_logdir=log_dir)

    print("Setting up scenario and data...")
    model = build_model()
    dataset, valid_set, _, new_scenario, prototype_surface, get_ideal_surface, \
        translation_vector, ideal_grid_xy, aim_point_area, aim_point_receiver, get_aug_surface = \
        setup_data_and_scenario(num_inputs, keys)

    print("Starting training loop...")
    train_losses, valid_losses, trained_model, surface_mae_list = train_dnn(
        dataset, valid_set, device, model, criterion, batch_size,
        optimizer, learning_rate, weight_decay, dirs,
        max_epochs, patience_epochs, patience_loss,
        aim_point_receiver, aim_point_area, new_scenario, ideal_grid_xy,
        translation_vector, prototype_surface, output_weight_decay, get_ideal_surface,
        get_aug_surface, z_constraint_weight=z_constraint_weight, decay_output_weight_decay=decay_output_weight_decay,
        use_curvature_penalty=True, curvature_penalty_weight=curvature_penalty_weight, curvature_warmup_epochs=5,
        use_edge_dip_reward=True, edge_dip_reward_weight=edge_dip_reward_weight,
        dip_margin=dip_margin, use_ddp=False, local_rank=0, ref_curvature=ref_curvature,
        tilt_penalty_weight=tilt_penalty_weight, zrange_penalty_weight=zrange_penalty_weight,
        max_zrange=max_zrange, return_surface_loss=True, linear_decay=linear_decay, cos_weight_decay=cos_weight_decay
    )

    trial_name = os.path.basename(dirs.rstrip("/"))
    target_path = os.path.join(final_target, trial_name)
    shutil.copytree(dirs, target_path, dirs_exist_ok=True)
    print(f"rial #{trial.number} finished with min surface MAE loss: {min(surface_mae_list):.6f}")

    return min(surface_mae_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    storage_path = f"sqlite:////mnt/optuna_schedulers_gpu{args.gpu}_{run_id}.db"

    log_dir = f"/mnt/optuna_tensorboard_logs/gpu_{args.gpu}_schedulers_{run_id}"
    final_target = f"/mnt/bayes_opt_{run_id}"

    study = optuna.create_study(
        direction="minimize",
        study_name=f"optuna_gpu{args.gpu}_{run_id}",
        storage=storage_path,
        load_if_exists=True
    )

    study.optimize(lambda trial: objective(trial, log_dir, final_target), n_trials=25)

    output_dir = "/mnt/final_results/best_trials"
    os.makedirs(output_dir, exist_ok=True)
    if study.best_trial:
        with open(os.path.join(output_dir, f"best_trial_gpu_{args.gpu}_{run_id}.json"), "w") as f:
            json.dump({
                "minimum validation loss": study.best_trial.value,
                "hyperparameters": study.best_trial.params,
                "trial index": study.best_trial.number
            }, f, indent=4)






