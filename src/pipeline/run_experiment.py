
"""
run_experiment.py

Entry point for training DeepLARTS with distributed data parallelism (DDP).
It is to be run on a cluster with 4 GPUs.

Features:
- Multi-GPU setup with torch.distributed (NCCL backend)
- Hyperparameter definition (optimizer, criterion, LR, weight decay, penalties)
- Model and dataset initialization
- Training loop with early stopping
- Logging and checkpoint handling

This script was originally developed for the experiment:
'standard_model_4_GPUs_5000_inputs_close_5-15m'.
It has been generalized to serve as the main training entry point
for the thesis pipeline.

References:
- Lewen et al. (2024), https://doi.org/10.48550/arXiv.2408.10802
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import torch
import os
import warnings
import shutil
from datetime import datetime
import torch.distributed as dist

from impl import train_dnn
from func import create_folder_struct_cluster
from data_setup import build_model, setup_data_and_scenario


if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=UserWarning)
    local_rank = int(os.environ["LOCAL_RANK"])

    #EARLY init of process group
    dist.init_process_group(backend="nccl", init_method="env://")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(f"[DEBUG] Rank {local_rank}: Visible GPUs = {torch.cuda.device_count()}, Current = {torch.cuda.current_device()}")

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    exp_name = "experiment"

    log_dir = ".../single_experiment_results"
    final_target = f"/code/single_experiment_results/{run_id}_{exp_name}"
    output_dir = f"/code/Results/Exp_{run_id}_{exp_name}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Starting Experiment #{run_id}")
    print("=" * 60)

    # Define the hyperparameter search space
    optimizer = "Adam"
    criterion = "MAE"
    scheduler_type = None
    learning_rate = 0.001
    weight_decay = 0

    # Shared config
    batch_size = 8  #
    max_epochs = 100  #
    patience_epochs = 100
    patience_loss = 1
    num_inputs = 4992
    keys = [f"AUG{i}" for i in range(1, num_inputs+1)]


    """ 
    
    The following booleans define the regularizations to be implemented.
    
    """

    close_to_receiver = False

    output_weight_decay_bool = False
    decay_output_weight_decay = False
    linear_decay = False
    cos_weight_decay = False
    if output_weight_decay_bool:
        output_weight_decay = 1
    else:
        output_weight_decay = 0

    use_z_constraint_weight = True
    if use_z_constraint_weight:
        z_constraint_weight = 1e5
    else:
        z_constraint_weight = 0

    use_curvature_penalty = False
    curvature_penalty_weight = 100.0
    ref_curvature = 0.02

    use_edge_dip_reward = False    #actually a penalty, not a reward!
    edge_dip_reward_weight = 1e5
    dip_margin = 0.00022

    filter_by_distance = False

    print(f"Hyperparameters used: ")
    print({
        "optimizer": optimizer,
        "criterion": criterion,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "output_weight_decay": output_weight_decay,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience_epochs": patience_epochs,
        "patience_loss": patience_loss,

    })

    # Model and data setup
    is_main_process = (local_rank == 0)

    if is_main_process:
        dirs = create_folder_struct_cluster(suffix=f"{run_id}_{exp_name}", base_logdir=log_dir)
    else:
        dirs = os.path.join(log_dir, datetime.now().strftime('%Y-%m-%d'), f"run_{run_id}_{exp_name}")
        os.makedirs(os.path.join(dirs, "tensorboard_logs"), exist_ok=True)  # ensure it exists to prevent crashes

    print("Setting up scenario and data...")
    model = build_model()
    dataset, valid_set, _, new_scenario, prototype_surface, get_ideal_surface, \
        translation_vector, ideal_grid_xy, aim_point_area, aim_point_receiver, get_aug_surface  = setup_data_and_scenario(num_inputs, keys, close_to_receiver=close_to_receiver)


    print("Starting training loop...")
    train_losses, valid_losses, trained_model = train_dnn(
        dataset, valid_set, device, model, criterion, batch_size,
        optimizer, learning_rate, weight_decay, dirs,
        max_epochs, patience_epochs, patience_loss,
        aim_point_receiver, aim_point_area, new_scenario, ideal_grid_xy, translation_vector,
        prototype_surface, output_weight_decay, get_ideal_surface, get_aug_surface, scheduler_type=scheduler_type,
        z_constraint_weight=z_constraint_weight, decay_output_weight_decay=decay_output_weight_decay,
        use_curvature_penalty=use_curvature_penalty, curvature_penalty_weight=curvature_penalty_weight, ref_curvature=ref_curvature,
        use_edge_dip_reward=use_edge_dip_reward, edge_dip_reward_weight=edge_dip_reward_weight,
        dip_margin=dip_margin, linear_decay=linear_decay, cos_weight_decay=cos_weight_decay, local_rank=local_rank, use_ddp=True,
    )

    # After training finishes, copy the log directory to the persistent output_dir
    if is_main_process:
        try:
            trial_name = os.path.basename(dirs.rstrip("/"))
            target_path = os.path.join(final_target, trial_name)
            shutil.copytree(dirs, target_path, dirs_exist_ok=True)
            print(f"✅ Copied trial logs to {target_path}")
        except Exception as e:
            print(f"⚠ Failed to copy trial logs: {e}")

