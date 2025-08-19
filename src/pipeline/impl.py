"""
Core training and evaluation implementation for heliostat surface prediction.

This module contains:
- Custom loss functions (RMSE, SSIM, curvature, edge dip, z-range, etc.)
- Training loop train_dnn is the most important function
- Evaluation on validation and test sets
- TensorBoard logging utilities for 2D heatmaps and 3D surface comparisons
- Utility classes for sampling, early stopping, and loss monitoring

Author (thesis adaptations): Anton Tenzler
General Framework was provided by: Jing Sun
"""


import copy
import time
import math
import traceback
import psutil
import sys
import os
import re
import torch
import torch.nn as nn
import logging
import random
import gc
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Sampler
from pytorch_msssim import ssim
import torch.distributed as dist
import pickle
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torchvision.transforms import ToTensor
from PIL import Image

from integrate_raytracer import  overwrite_scenario, raytracing
from artist.util.utils import convert_3d_point_to_4d_format
from artist.field.surface import Surface


class FluxLossMonitor:
    def __init__(self, min_epochs=5, loss_eps=1e-4, low_threshold=1e-5):
        self.history = []
        self.min_epochs = min_epochs
        self.loss_eps = loss_eps
        self.low_threshold = low_threshold

    def update(self, loss_val):
        self.history.append(loss_val)
        if len(self.history) > self.min_epochs:
            self.history.pop(0)

    def is_stuck(self):
        if len(self.history) < self.min_epochs:
            return False
        return (max(self.history) - min(self.history)) < self.loss_eps

    def is_very_low(self):
        return self.history[-1] < self.low_threshold


class RepulsionBuffer:
    def __init__(self, max_len=5):
        self.buffer = []
        self.max_len = max_len

    def add(self, z_pred):
        self.buffer.append(z_pred.detach().clone())
        if len(self.buffer) > self.max_len:
            self.buffer.pop(0)

    def compute(self, current_z):
        losses = []
        for past in self.buffer:
            sim = F.cosine_similarity(current_z.view(-1), past.view(-1), dim=0)
            losses.append(sim**2)  # square for sharp penalty
        if not losses:
            return torch.tensor(0.0, device=current_z.device)
        return torch.stack(losses).mean()


def plane_normal_loss(xy_grid, z_facet, target_normal=torch.tensor([0, 0, 1], dtype=torch.float32)):
    """
    Penalizes deviation of surface normal from vertical (z-axis).
    xy_grid: [8, 8, 2] — fixed X,Y positions
    z_facet: [8, 8]    — predicted Z values
    """
    device = z_facet.device
    X = xy_grid[..., 0]
    Y = xy_grid[..., 1]
    Z = z_facet

    # Flatten and stack for plane fitting: [N, 3]
    pts = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # [64, 3]

    # Center the points
    pts_centered = pts - pts.mean(dim=0, keepdim=True)

    # Safety check: skip flat surfaces
    if pts_centered.std() < 1e-8:
        return torch.tensor(0.0, device=device)

    # Try SVD
    try:
        _, _, V = torch.svd(pts_centered)
        normal = V[:, -1]  # last singular vector
    except RuntimeError:
        return torch.tensor(0.0, device=device)  # fallback value on failure

    # Normalize
    normal = normal / (normal.norm() + 1e-8)
    target_normal = target_normal.to(device)

    # Penalize deviation from vertical
    deviation = 1 - torch.abs(torch.dot(normal, target_normal))
    return deviation


def count_large_objects(min_bytes=1e6):
    #print("[DEBUG] Large object summary (in MB):")
    for obj in gc.get_objects():
        try:
            size = sys.getsizeof(obj)
            #if size > min_bytes:
                #print(f"  - {type(obj)}: {size / 1e6:.2f} MB")
        except:
            pass

def count_active_tensors():
    count = sum(1 for obj in gc.get_objects()
                if torch.is_tensor(obj) and obj.requires_grad)
    #(f"[DEBUG] Live tensors requiring grad: {count}")

def get_grad_tensor_ids():
    return {
        id(obj)
        for obj in gc.get_objects()
        if torch.is_tensor(obj) and obj.requires_grad
    }

def log_new_grad_tensors(prev_ids):
    new_ids = get_grad_tensor_ids() - prev_ids
    if new_ids:
        #print(f"[DEBUG] New grad tensors since last check: {len(new_ids)}")
        for obj in gc.get_objects():
            if id(obj) in new_ids:
                try:
                    print(f"  - {type(obj)} {obj.shape} {obj.device} grad_fn={obj.grad_fn}")
                except Exception:
                    pass
    return get_grad_tensor_ids()

def normalize_per_image(t):
    return torch.stack([(img / (img.max().clamp(min=1e-3))) for img in t])

class ChunkSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, chunk_size=8, shuffle=False):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.epoch = 0
        self._update_chunks()

    def _update_chunks(self):
        self.chunks = list(range(0, len(self.dataset), self.chunk_size))
        if self.shuffle:
            rng = random.Random(self.epoch)
            rng.shuffle(self.chunks)
        self.chunks = self.chunks[self.rank::self.num_replicas]
        #print(f"[Rank {self.rank}] ChunkSampler: assigned {len(self.chunks)} chunks")
        #print(f"[Rank {self.rank}] Chunk indices: {self.chunks[:5]} ... {self.chunks[-5:]}")

        for chunk_start in self.chunks[:5]:
            names = []
            sun_positions = []
            for i in range(chunk_start, chunk_start + self.chunk_size):
                try:
                    sample_input, _ = self.dataset[i]
                    sun_pos = sample_input[0]  # [8, 3]
                    name = sample_input[3]  # heliostat_name
                    names.append(name)
                    sun_positions.append(sun_pos[0].tolist())  # first sun_pos
                except Exception as e:
                    print(f"[Rank {self.rank}] Error reading index {i}: {e}")
            #print(f"[Rank {self.rank}] Chunk {chunk_start}-{chunk_start + self.chunk_size}: {names}")
            #print(f"[Rank {self.rank}] Sun positions in chunk: {sun_positions}")

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._update_chunks()

    def __iter__(self):
        for chunk_start in self.chunks:
            yield from range(chunk_start, chunk_start + self.chunk_size)

    def __len__(self):
        return len(self.chunks) * self.chunk_size


def z_range_penalty(z_pred, max_range=0.0005):
    """
    Penalizes Z-ranges exceeding max_range using a smooth penalty.
    z_pred: [B, 4, 8, 8]
    Returns: scalar penalty
    """
    B, F, H, W = z_pred.shape
    penalty_vals = []

    for b in range(B):
        for f in range(F):
            z = z_pred[b, f]
            z_max = z.max()
            z_min = z.min()
            range_ = z_max - z_min

            # Smooth transition instead of hard clamp
            excess = torch.tanh((range_ - max_range) * 100.0)  # Scale controls sharpness
            excess = torch.clamp(excess, min=0.0)  # Ensure no negative contribution

            penalty_vals.append(excess)

    if penalty_vals:
        penalty = torch.stack(penalty_vals).mean()
        if not torch.isfinite(penalty):
            return torch.tensor(0.0, device=z_pred.device)
        return penalty
    else:
        return torch.tensor(0.0, device=z_pred.device)



def incentivize_edge_dip(z_tensor, margin=0.0005):
    """
    Encourages edge Z values to be lower than adjacent inner values by a small margin.
    z_tensor: [B, 4, 8, 8]
    Returns: Tensor [B] — average loss per sample
    """
    B = z_tensor.shape[0]

    # Top vs. 2nd row
    top = z_tensor[:, :, 0, :]
    inner_top = z_tensor[:, :, 1, :]
    diff_top = top - inner_top + margin  # want: top < inner → diff_top ≈ 0

    # Bottom vs. 6th row
    bottom = z_tensor[:, :, 7, :]
    inner_bottom = z_tensor[:, :, 6, :]
    diff_bottom = bottom - inner_bottom + margin

    # Left vs. 2nd col
    left = z_tensor[:, :, :, 0]
    inner_left = z_tensor[:, :, :, 1]
    diff_left = left - inner_left + margin

    # Right vs. 6th col
    right = z_tensor[:, :, :, 7]
    inner_right = z_tensor[:, :, :, 6]
    diff_right = right - inner_right + margin

    total_penalty = (
        diff_top.pow(2).mean(dim=[1, 2]) +
        diff_bottom.pow(2).mean(dim=[1, 2]) +
        diff_left.pow(2).mean(dim=[1, 2]) +
        diff_right.pow(2).mean(dim=[1, 2])
    ) / 4.0

    return total_penalty  # shape: [B]

def mean_facet_curvature(z_pred):
    """
    Computes mean absolute Laplacian curvature on inner 6x6 region only.
    z_pred: [B, 4, 8, 8]
    Returns: [B] curvature per sample
    """

    # Define Laplacian kernel globally
    laplace_kernel = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]], dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    B = z_pred.shape[0]
    result = []
    for b in range(B):
        curvatures = []
        for f in range(4):
            z = z_pred[b, f:f+1].unsqueeze(0)  # shape [1, 1, 8, 8]
            lap = F.conv2d(z, laplace_kernel.to(z.device), padding=1)  # [1, 1, 8, 8]
            inner = lap[..., 1:-1, 1:-1]  # crop to [1, 1, 6, 6]
            curvatures.append(inner.abs().mean())
        result.append(torch.stack(curvatures).mean())
    return torch.stack(result)  # shape: [B]

def mse_above_threshold(pred_curvatures, reference_curvature, threshold=0.002):
    """
    Applies MSE only when curvature deviates from reference by more than a threshold.
    """
    diff = pred_curvatures - reference_curvature
    mask = diff.abs() > threshold
    loss = (diff[mask] ** 2).mean() if mask.any() else torch.tensor(0.0, device=pred_curvatures.device)
    return loss

def soft_curvature_loss(pred_curvatures, reference_curvature, delta=0.01):
    """
    Applies a soft quadratic penalty to curvature above the reference.
    Below the reference, no penalty.
    """
    diff = pred_curvatures - reference_curvature
    excess = torch.clamp(diff, min=0.0)
    # Smooth Huber-like: quadratic below delta, linear above
    loss = torch.where(
        excess < delta,
        0.5 * (excess ** 2) / delta,
        excess - 0.5 * delta
    )
    return loss.mean()

def tolerance_curvature_loss(pred_curvatures, reference_curvature, epsilon=0.01):
    """
    Only penalize curvature deviations beyond a tolerance band [ref - ε, ref + ε].
    Uses a quadratic penalty outside that range.
    """
    diff = torch.abs(pred_curvatures - reference_curvature)
    excess = torch.clamp(diff - epsilon, min=0.0)
    return (excess ** 2).mean()

def log_3d_surface_comparison(predicted_surface, surface_ref, writer, is_main_process, epoch=None,
                              aug_key="SurfaceComparison"):
    """
    Logs a 3D plot comparing predicted vs. ground-truth surfaces to TensorBoard,
    with fixed axes and custom colors.
    """

    predicted_surface = predicted_surface.detach().cpu()
    augmented_surface = surface_ref.detach().cpu()


    fig = plt.figure(figsize=(10, 8), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Surface Comparison - {aug_key}", fontsize=10)

    # Choose one facet to compare (facet 0)
    facet_gt = augmented_surface[0]
    # Center GT surface by subtracting mean Z (for comparison only)
    Z_gt = facet_gt[..., 2]
    Z_gt_centered = Z_gt - Z_gt.mean()
    X_gt, Y_gt = facet_gt[..., 0], facet_gt[..., 1]
    facet_pred = predicted_surface[0]
    facet_gt_centered = facet_gt.clone()
    facet_gt_centered[..., 2] = Z_gt_centered
    #X_gt, Y_gt, Z_gt = facet_gt[..., 0], facet_gt[..., 1], facet_gt[..., 2]
    X_pred, Y_pred, Z_pred = facet_pred[..., 0], facet_pred[..., 1], facet_pred[..., 2]


    # Plot ground truth in green
    ax.plot_surface(X_gt, Y_gt, Z_gt_centered, color='green', alpha=0.7, edgecolor='k')
    # Plot predicted in red
    ax.plot_surface(X_pred, Y_pred, Z_pred, color='red', alpha=0.6, edgecolor='k')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='k', label='Ground Truth'),
        Patch(facecolor='red', edgecolor='k', label='Predicted')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Set fixed axis limits (update as appropriate for your surfaces)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-0.002, 0.002)

    #zmin = min(Z_gt.min().item(), Z_pred.min().item())
    #zmax = max(Z_gt.max().item(), Z_pred.max().item())
    #margin = 0.01 * (zmax - zmin)
    #ax.set_zlim(zmin - margin, zmax + margin)

    ax.view_init(elev=30, azim=60)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200)
    buf.seek(0)
    image = Image.open(buf)
    image = transforms.ToTensor()(image)

    if is_main_process:
        writer.add_image(f"Surfaces/{aug_key}", image.detach().cpu(), epoch)
    plt.close(fig)

    # Compute per-point L2 distance (Euclidean)
    diff = torch.norm(facet_pred - facet_gt_centered, dim=-1)  # shape: [8, 8]

    # Create new figure for 2D difference heatmap
    fig2, ax2 = plt.subplots(figsize=(6, 5), dpi=150)
    cmap = plt.get_cmap("hot")
    im = ax2.imshow(diff.numpy(), cmap=cmap, origin="lower", vmin=0.0, vmax=0.002)

    ax2.set_title(f"Surface Error Map - {aug_key} (Facet 0)")
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label='L2 Distance')

    # Save and log the heatmap to TensorBoard
    buf2 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf2, format='png', dpi=150)
    buf2.seek(0)
    image2 = Image.open(buf2)
    image2 = transforms.ToTensor()(image2)

    if is_main_process:
        writer.add_image(f"Surfaces/ErrorMap_{aug_key}", image2.detach().cpu(), epoch)
    plt.close(fig2)

    # Explicit memory cleanup
    del fig, fig2, ax, ax2, buf, buf2, image, image2
    del predicted_surface, augmented_surface, facet_gt, facet_pred, facet_gt_centered
    del X_gt, Y_gt, Z_gt, Z_gt_centered, X_pred, Y_pred, Z_pred, diff
    del cmap, im, legend_elements
    gc.collect()
    torch.cuda.empty_cache()

def normalize_for_training(tensor):
    """Normalize tensor to 0-1 range per sample for training."""
    B = tensor.shape[0]
    normed = []
    for i in range(B):
        img = tensor[i]
        min_val = img.min()
        max_val = img.max()
        if max_val - min_val > 0:
            img = (img - min_val) / (max_val - min_val + 1e-8)
        else:
            img = torch.zeros_like(img)
        normed.append(img)
    return torch.stack(normed)

def normalize_flux_image(tensor):
    """Normalize flux images assuming they are [0, 255] or [0,1] floats."""
    if tensor.max() > 1.5:  # Heuristic: If max > 1.5, assume in [0,255]
        tensor = tensor / 255.0
    return tensor

def log_augmented_surface_errors_single(predicted_surface, surface_ref, writer, is_main_process,
                                        aug_key,  epoch=None):

    """Compare a single predicted surface to a reference augmented surface and log error metrics.

    Args:
        predicted_surface (Tensor): Control points in [4, 8, 8, 3] format (XYZ).
        augmented_surface_dict (dict): Dictionary of reference surfaces keyed by e.g., 'AUG1'.
        writer (SummaryWriter): TensorBoard writer.
        epoch (int): Current epoch number.
        aug_key (str): Key for the augmented surface to compare against."""

    predicted_surface_cpu = predicted_surface.detach().cpu()
    surface_ref_cpu = surface_ref.detach().cpu()

    def center_surface_z(surface):
        z = surface[..., 2]
        z_centered = z - z.mean()
        surface_centered = surface.clone()
        surface_centered[..., 2] = z_centered
        del z_centered
        return surface_centered

    pred_centered = center_surface_z(predicted_surface_cpu[0])
    gt_centered = center_surface_z(surface_ref_cpu[0])

    mae = F.l1_loss(pred_centered[..., 2], gt_centered[..., 2]).item() #reduction sum.
    #mse = F.mse_loss(pred_centered[..., 2], gt_centered[..., 2]).item()
    #rmse = torch.sqrt(F.mse_loss(pred_centered[..., 2], gt_centered[..., 2]) + 1e-8).item()

    # Log to TensorBoard
    if is_main_process and epoch and writer:
        writer.add_scalar("SurfaceError/MAE", mae, epoch)
        #writer.add_scalar("SurfaceError/MSE", mse, epoch)
        #writer.add_scalar("SurfaceError/RMSE", rmse, epoch)

    del pred_centered, gt_centered, predicted_surface_cpu, surface_ref_cpu  # or mse, rmse if used
    gc.collect()

    return mae


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true) + 1e-8)  # epsilon to avoid sqrt(0)

class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return 1 - ssim(y_pred, y_true, data_range=1.0, size_average=True)

class CombinedMSESSIMLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Combines MSE and SSIM losses.
        alpha: weight for MSE loss (between 0 and 1). SSIM gets (1 - alpha).
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        ssim_loss = 1 - ssim(y_pred, y_true, data_range=1.0, size_average=True)
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss


# A simple early stopping function, you may want to change it.
def early_stopping(valid_losses, patience_epochs, patience_loss):
    if len(valid_losses) < patience_epochs:
        return False
    recent_losses = valid_losses[-patience_epochs:]

    if all(x >= recent_losses[0] for x in recent_losses):
        return True

    if max(recent_losses) - min(recent_losses) < patience_loss:
        return True
    return False

import torch as th

def give_accuracy(target, prediction, batch_mode=False, normalize=False):
    assert target.size() == prediction.size(), "Target and Prediction must have same size"

    # Handle target dimensions
    if target.ndim == 2:  # (H, W)
        target = target.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    elif target.ndim == 3:  # (C, H, W)
        target = target.unsqueeze(0)  # Add batch dimension
    elif target.ndim == 4:  # (B, C, H, W)
        pass  # No change needed
    else:
        raise ValueError(f"Unexpected tensor dimensions: {target.shape}. "
                         f"Expected: (H, W), (C, H, W), or (B, C, H, W).")

    # Handle prediction dimensions
    if prediction.ndim == 2:  # (H, W)
        prediction = prediction.unsqueeze(0).unsqueeze(0)
    elif prediction.ndim == 3:  # (C, H, W)
        prediction = prediction.unsqueeze(0)
    elif prediction.ndim == 4:  # (B, C, H, W)
        pass
    else:
        raise ValueError(f"Unexpected tensor dimensions: {prediction.shape}. "
                         f"Expected: (H, W), (C, H, W), or (B, C, H, W).")

    # Normalize if requested
    if normalize is True:
        target = target / th.sum(target, dim=(2, 3), keepdim=True)
        prediction = prediction / th.sum(prediction, dim=(2, 3), keepdim=True)
    elif normalize is False:
        pass
    else:
        raise Exception("You entered a wrong value for normalize!")

    # Compute accuracy
    if batch_mode:
        diff = target - prediction
        difference_sum = th.sum(th.abs(diff), dim=(1, 2, 3))
        target_sum = th.sum(th.abs(target), dim=(1, 2, 3))

        # Avoid division by zero
        target_sum[target_sum == 0] = 1

        acc = 1 - difference_sum / target_sum
    else:
        diff = target - prediction
        difference_sum = th.sum(th.abs(diff))
        target_sum = th.sum(th.abs(target))

        if target_sum.item() == 0:
            return 0.0

        acc = 1 - difference_sum / target_sum

    return acc


def manage_saved_models(directory):
    pattern = re.compile(r'epoch_(\d+)\.pth')
    epoch_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            match = pattern.match(file)
            if match:
                epoch_num = int(match.group(1))
                file_path = os.path.join(root, file)
                epoch_files.append((file_path, epoch_num))

    # Check if there are more than 5 files
    if len(epoch_files) > 5:
        epoch_files.sort(key=lambda x: x[1])
        files_to_delete = len(epoch_files) - 5

        for i in range(files_to_delete):
            os.remove(epoch_files[i][0])

def clear_logging_handlers():
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

def create_loss(criterion_type):
    if criterion_type == 'MAE':
        return nn.L1Loss()
    elif criterion_type == 'MSE':
        return nn.MSELoss()
    elif criterion_type == 'RMSE':
        return RMSELoss()
    elif criterion_type == 'SSIM':
        return SSIMLoss()
    elif criterion_type == 'MSE+SSIM_0.5':
        return CombinedMSESSIMLoss(alpha=0.5)  # or tweak alpha
    elif criterion_type == 'MSE+SSIM_0.25_0.75':
        return CombinedMSESSIMLoss(alpha=0.25)  # or tweak alpha
    elif criterion_type == 'MSE+SSIM_0.75_0.25':
        return CombinedMSESSIMLoss(alpha=0.75)  # or tweak alpha
    else:
        raise ValueError(f"Undefined criterion type '{criterion_type}'. Please update create_loss().")

def train_dnn(train_set, valid_set,
                device, model, criterion, batch_size,
                optimizer_type, learning_rate, weight_decay, dirs,
                max_epochs, patience_epochs, patience_loss,
                heliostat_aim_point, aim_point_area, old_scenario, xy_grid, translation_vector,
                prototype_surface, output_weight_decay, get_ideal_surface,
                get_aug_surface, valid_surface_dict=None, overfit_one_helio=False,
              use_normalization=True, scheduler_bool=False,  scheduler_type="None", scheduler_params=None,
              z_constraint_weight = 0, decay_output_weight_decay=False, use_curvature_penalty = False,
              curvature_penalty_weight = 0, ref_curvature = 0.01, curvature_warmup_epochs=0,
              use_edge_dip_reward = False, edge_dip_reward_weight = 0.0, dip_margin = 0.0005, linear_decay = True,
              cos_weight_decay = False, return_surface_loss = False, local_rank=None, use_ddp=False,
              tilt_penalty_weight=0.0, zrange_penalty_weight=0.0,
              max_zrange=0.0005, use_repulsion = False, min_epochs=4, loss_eps=0.001, low_threshold=0.004, max_len=5):


    if use_ddp:
        if not dist.is_initialized():
            if torch.cuda.device_count() > 1:
                dist.init_process_group(backend="nccl", init_method='env://')
            else:
                dist.init_process_group(backend="gloo", init_method="file://tmp_shared_init", rank=0, world_size=1)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        #model_args = model
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        model_args = model.module
    else:
        model_args = model
        model.to(device)



    is_main_process = not use_ddp or (dist.get_rank() == 0)

    # Load these from saved file if needed --> for heliostat normalization
    min_vals = torch.tensor([-58.0915, 29.2537, 1.5110], device = device)
    max_vals = torch.tensor([157.8804, 243.5274, 2.0130], device = device)

    # Create logger object
    logging.basicConfig(level=logging.INFO, filename=dirs + '/loss_record.log', filemode='a',
                        format='%(asctime)s   %(levelname)s   %(message)s')


    log_dir = os.path.join(dirs, "tensorboard_logs")
    if is_main_process:
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    criterion = create_loss(criterion)


    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError("Undefined optimizer type. Update your code.")

    scheduler = None

    if scheduler_params is None:
        scheduler_params = {}

    if scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_params.get("factor", 0.5),
            patience=scheduler_params.get("patience", 6),
            threshold=scheduler_params.get("threshold", 1e-4),
            min_lr=1e-7,
            verbose=False
        )

    elif scheduler_type == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_params.get("gamma", 0.95)
        )

    elif scheduler_type == "CyclicLR":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=scheduler_params.get("base_lr", 1e-5),
            max_lr=scheduler_params.get("max_lr", 1e-2),
            step_size_up=scheduler_params.get("step_size_up", 10),
            mode='triangular',
            cycle_momentum=False
        )

    elif scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get("T_max", max_epochs),
            eta_min=scheduler_params.get("eta_min", 1e-6)
        )
    train_losses = []
    valid_losses = []
    surface_MAE_list = []
    best_surface_mae = float("inf")
    best_surface_model_state = None
    #best_valid_loss = np.inf
    #best_model_state = None
    best_train_model_state=None
    best_train_loss = np.inf


    if use_ddp:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        train_sampler = ChunkSampler(train_set, num_replicas=world_size, rank=rank, chunk_size=train_set.chunk_size,
                                     shuffle=False)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_set,
        sampler=train_sampler,
        shuffle=False,  # False if using sampler
        batch_size=batch_size,
        num_workers=0,
        drop_last=True
    )
    if use_ddp and len(train_loader) == 0:
        raise RuntimeError(f"[Rank {dist.get_rank()}] No data assigned to this rank. Training cannot continue.")


    repulsion_active = False

    first_overwrite = True

    if use_repulsion:
        flux_monitor = FluxLossMonitor(min_epochs=min_epochs, loss_eps=loss_eps, low_threshold=low_threshold)
        repulsion_buffer = RepulsionBuffer(max_len=max_len)
        lambda_repulsion = 1e3

    try:
        for epoch in range(max_epochs):
            start_time = time.time()

            if use_ddp and train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            if is_main_process:
                print(f" Starting Epoch {epoch}")
            # Train the model
            model.train()
            running_train_loss = 0.0
            for i, (inputs, z_points) in enumerate(train_loader, 0):
                grad_ids = get_grad_tensor_ids()
                if model_args.architecture_args.get("use_canting_vecs", False):
                    sun_pos, flux_img, heliostat_pos_norm, canting_vecs, heliostat_name, aug_surface_key = inputs
                    canting_vecs = canting_vecs.to(device)
                else:
                    sun_pos, flux_img, heliostat_pos_norm, heliostat_name, aug_surface_key = inputs
                    canting_vecs = None
                model.zero_grad()
                optimizer.zero_grad()
                total_loss = 0



                targetID = None

                surface_pool = [Surface(copy.deepcopy(prototype_surface)) for _ in range(batch_size)]

                for surface in surface_pool:
                    for facet in surface.facets:
                        del facet.control_points
                        facet.control_points = torch.zeros(8, 8, 3, device=device)

                sun_pos = sun_pos.to(device)  # [B, 8, 3]
                flux_img = flux_img.to(device)  # [B, 8, 1, 64, 64]
                heliostat_pos_norm = heliostat_pos_norm.to(device)  # [B, 3]
                z_points = z_points.to(device)
                heliostat_names = list(heliostat_name)

                if model_args.architecture_args.get("use_canting_vecs", False):
                    z_cntrl_points = model(flux_img, sun_pos, heliostat_pos_norm, targetID, canting_vecs=canting_vecs)
                else:
                    z_cntrl_points = model(flux_img, sun_pos, heliostat_pos_norm, targetID)

                z_pred = z_cntrl_points[0]  # [B, 4, 8, 8]
                z_mean = z_pred.view(z_pred.shape[0], -1).mean(dim=1, keepdim=True).view(-1, 1, 1, 1)  # [B,1,1,1]
                z_pred_centered = z_pred - z_mean  # broadcast subtraction
                z_cntrl_points = (z_pred_centered,)  # Replace in tuple

                # Optional exploration noise if stuck
                if repulsion_active:
                    noise = 0.00015 * torch.randn_like(z_cntrl_points[0])
                    print("Noise in repulsion:", noise)
                    z_cntrl_points = (z_cntrl_points[0] + noise,)

                zrange_penalty = z_range_penalty(z_pred, max_range=max_zrange)
                zrange_penalty_weighted = zrange_penalty_weight * zrange_penalty

                tilt_penalties = []
                for b in range(z_cntrl_points[0].shape[0]):  # loop over batch
                    for f in range(4):  # loop over facets
                        z_facet = z_cntrl_points[0][b, f]  # [8, 8]
                        tilt_loss = plane_normal_loss(xy_grid[f], z_facet)  # xy_grid[f] shape: [8, 8, 2]
                        tilt_penalties.append(tilt_loss)
                tilt_penalty = torch.stack(tilt_penalties).mean()
                tilt_penalty_weighted = tilt_penalty_weight * tilt_penalty

                if i == 0:
                    # Only plot for the first sample in the first batch every 10 epochs
                    z = z_cntrl_points[0][0].detach().unsqueeze(-1)
                    pred_surface = torch.cat([xy_grid, z], dim=3)
                    key = aug_surface_key[0]
                    del z
                    try:
                        surface_ref_cpu = get_aug_surface(key)
                        surface_ref = surface_ref_cpu.to(device)
                        del surface_ref_cpu
                    except FileNotFoundError:
                        print(f"Skipping missing surface {key}")
                        continue

                    MAE = log_augmented_surface_errors_single(pred_surface, surface_ref, writer,  is_main_process,
                                                              epoch=epoch, aug_key=key)
                    surface_MAE_list.append(MAE)


                    if MAE < best_surface_mae:
                        best_surface_mae = MAE
                        best_surface_model_state = model.state_dict()
                        if is_main_process:
                            torch.save(best_surface_model_state, os.path.join(dirs, f"best_surface_model.pth"))

                    if epoch % 10 == 0:
                        log_3d_surface_comparison(pred_surface, surface_ref, writer, is_main_process, epoch=epoch,
                                                  aug_key=key)

                    del surface_ref
                    del pred_surface

                if use_curvature_penalty and epoch >= curvature_warmup_epochs:
                    z_pred = z_cntrl_points[0]  # shape: [B, 4, 8, 8]
                    pred_curvatures = mean_facet_curvature(z_pred)  # [B]
                    reference_curvature = ref_curvature
                    curvature_penalty = mse_above_threshold(pred_curvatures, reference_curvature,
                                                            threshold=reference_curvature)
                    curvature_regularization = curvature_penalty_weight * curvature_penalty

                    if is_main_process:
                        writer.add_scalar("Penalty/curvature_penalty", curvature_penalty.detach().cpu().item(),
                                          epoch)

                    del pred_curvatures, reference_curvature, curvature_penalty
                else:
                    curvature_regularization = 0

                if decay_output_weight_decay:
                    decay_end = max_epochs #  // 2
                    if epoch <= decay_end:
                        decay_progress = epoch / decay_end

                        if linear_decay:
                            current_weight_decay = output_weight_decay * (1.0 - decay_progress)

                        elif cos_weight_decay:
                            current_weight_decay = output_weight_decay * 0.5 * (1 + math.cos(math.pi * decay_progress))

                        else:
                            current_weight_decay = output_weight_decay  # fallback in case neither decay is selected

                    else:
                        current_weight_decay = 0.0  # decay finished after epoch 20

                else:
                    current_weight_decay = output_weight_decay  # no decay at all

                l2_reg = current_weight_decay * torch.norm(z_cntrl_points[0]) ** 2

                del current_weight_decay

                # Z constraint penalty (optional 2mm check)
                z_vals = z_cntrl_points[0]  # shape: [B, 4, 8, 8]
                abs_excess = torch.clamp(torch.abs(z_vals) - 0.0005, min=0.0)  # in meters
                penalty_term = (abs_excess ** 2).mean()  # or .sum(), depending on your scale preference
                z_penalty = z_constraint_weight * penalty_term

                del penalty_term

                if use_edge_dip_reward:
                    dip_penalty = incentivize_edge_dip(z_cntrl_points[0], margin=dip_margin).mean()
                    dip_penalty_weighted = edge_dip_reward_weight * dip_penalty
                    if is_main_process:
                        writer.add_scalar("Penalty/encouraged_edge_dip", dip_penalty.detach().cpu().item(), epoch)
                    del dip_penalty
                else:
                    dip_penalty_weighted = 0.0

                heliostat_pos = heliostat_pos_norm * (max_vals - min_vals) + min_vals

                 # Iterate over each sample in the batch
                new_scenario, first_overwrite = overwrite_scenario(
                    heliostat_aim_point, heliostat_pos, z_cntrl_points[0], old_scenario,
                    prototype_surface, xy_grid, heliostat_pos.shape[0], first_overwrite, device,
                    heliostat_names=heliostat_names, get_ideal_surface=get_ideal_surface,
                    translation_vector=translation_vector, surface_pool=surface_pool
                )

                #raytracer can only take one sun position at a time, so take first sun position in every datapoint. I
                # ts going to be the same in this batch.
                sun_pos= sun_pos[:, 0, :]
                # Check if all are the same as the first entry
                if torch.allclose(sun_pos, sun_pos[0].expand_as(sun_pos)):
                    # Convert the single unique sun position to 4D format
                    sun_pos_for_raytracing = convert_3d_point_to_4d_format(sun_pos[0], device=device)  # shape [4]
                else:
                    raise ValueError("Careful, sun positions will vary within chosen batch size/chunk!")

                images = raytracing(new_scenario, sun_pos_for_raytracing, aim_point_area, batch_size,
                                    show_image=False, device=device)

                # Add a channel dimension (assuming it's grayscale, so 1 channel)
                images = images.unsqueeze(1)  # Shape: [1, 1, 64, 64]

                if dist.is_initialized():
                    dist.barrier()

                flux_img = flux_img.to(images.device)
                flux_img = flux_img[:, 0:1, :, :]

                flux_img_norm = normalize_per_image(flux_img)
                images_norm = normalize_per_image(images)

                image_loss = criterion(flux_img_norm, images_norm)

                if use_repulsion:
                    flux_monitor.update(image_loss.item())

                    repulsion_active = flux_monitor.is_stuck() and not flux_monitor.is_very_low()
                    if repulsion_active:
                        repulsion_loss = repulsion_buffer.compute(z_cntrl_points[0])
                        if is_main_process:
                            writer.add_scalar("Penalty/repulsion_loss", repulsion_loss.item(), epoch)
                    else:
                        repulsion_loss = torch.tensor(0.0, device=z_cntrl_points[0].device)

                # Always update the buffer (only when active to avoid over-regularization)
                if repulsion_active:
                    repulsion_buffer.add(z_cntrl_points[0].detach())

                loss = (image_loss + l2_reg + z_penalty + curvature_regularization + dip_penalty_weighted
                        + tilt_penalty_weighted + zrange_penalty_weighted +
                        (lambda_repulsion * repulsion_loss if repulsion_active else 0.0))


                if i == 0 and epoch % 10 == 0:
                    # Log first image side-by-side
                    comparison = torch.cat([images_norm[0], flux_img_norm[0]], dim=2)
                    if is_main_process:
                        writer.add_image("Images_training/Pred_vs_GT", comparison.detach().cpu(),
                                         global_step=epoch)
                    del comparison

                # Accumulate the loss for this data point
                total_loss = total_loss + loss
                if is_main_process:
                    logging.info(f"Epoch {epoch + 1}, Batch {i + 1}, Loss = {loss.item():.6f}")

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                running_train_loss += total_loss.detach().item()

                for surface in surface_pool:
                    for facet in surface.facets:
                        facet.control_points = None

                # Explicitly clear scenario references
                hf = new_scenario.heliostat_field
                hf.all_surface_points = None
                hf.all_surface_normals = None
                hf.all_current_aligned_surface_points = None
                hf.all_current_aligned_surface_normals = None
                hf.all_preferred_reflection_directions = None
                hf.all_aligned_heliostats = None

                del surface_pool
                del z_cntrl_points, z_pred, z_mean, z_pred_centered, z_vals, #z
                del heliostat_pos, new_scenario, sun_pos_for_raytracing
                del abs_excess, z_penalty, l2_reg, curvature_regularization
                del dip_penalty_weighted
                del images, images_norm, flux_img, flux_img_norm, image_loss
                del loss, total_loss
                torch.cuda.empty_cache()
                gc.collect()


                count_active_tensors()
                count_large_objects()

                for param_group in optimizer.param_groups:
                    if is_main_process:
                        writer.add_scalar("LearningRate", param_group['lr'], epoch)

            epoch_train_loss = running_train_loss/len(train_loader)
            train_losses.append(epoch_train_loss)

            del running_train_loss

            if epoch_train_loss < best_train_loss:
                best_train_loss = epoch_train_loss
                best_train_model_state = model.state_dict()
                if is_main_process:
                    torch.save(best_train_model_state, os.path.join(dirs, "best_train_model.pth"))

            if is_main_process:
                logging.info(f"Epoch {epoch + 1} completed. Average Training Loss: {epoch_train_loss:.6f}")
            if is_main_process:
                writer.add_scalar("Loss/train", epoch_train_loss, epoch)

            if is_main_process:
                writer.add_scalar("Memory/CPU_RSS_MB_per_epoch",
                                  psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, epoch)

            try:
                if early_stopping(train_losses, patience_epochs, patience_loss):
                    if is_main_process:
                        logging.info(f"Early stopping at epoch {epoch + 1}")
                    break
            except Exception as e:
                print(f"[Rank {rank}] Error in early stopping check: {e}")
                traceback.print_exc()

            # After each epoch
            if scheduler:
                if scheduler_type == "ReduceLROnPlateau":
                    scheduler.step(epoch_train_loss)
                else:
                    scheduler.step()


            if epoch % 10 == 0:
                if writer is not None and is_main_process:
                    writer.flush()
                    writer.close()
                    del writer
                    gc.collect()
                    writer = SummaryWriter(log_dir=log_dir)

            end_time = time.time()
            if is_main_process:
                print(f"Time for epoch {epoch}:", end_time - start_time)

        if dist.is_initialized():
            dist.barrier()

    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[Rank {rank}] Unhandled Exception: {str(e)}")
        traceback.print_exc()
        sys.exit(1)  # Fail gracefully so other ranks stop too

    if is_main_process:
        logging.info("Training is done!")


    # Clear logging handlers to close the log file properly
    clear_logging_handlers()

    if is_main_process:
        writer.close()

    if use_ddp:
        dist.destroy_process_group()

    del best_train_model_state, best_surface_model_state
    gc.collect()

    if return_surface_loss:
        return train_losses, valid_losses, model, surface_MAE_list
    else:
        return train_losses, valid_losses, model


def normalize_tensor_for_tb(tensor):
    """Normalize tensor to 0-1 range per image (for TensorBoard)."""
    # [B, C, H, W] → normalize each image independently
    B = tensor.shape[0]
    normed = []
    for i in range(B):
        img = tensor[i]
        min_val = img.min()
        max_val = img.max()
        if max_val - min_val > 0:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = torch.zeros_like(img)  # avoid division by zero
        normed.append(img)
    return torch.stack(normed)

def print_surface_heatmaps(pred_surfaces, gt_surfaces, epoch, tag="SurfaceComparisonGrid"):
    """
    Logs a 6x2 grid of predicted and ground-truth surface heatmaps to TensorBoard.

    pred_surfaces and gt_surfaces: list of tensors with shape [4, 8, 8] (Z values for 4 facets).
    writer: TensorBoard writer
    """
    import matplotlib.pyplot as plt
    import io
    import random
    from torchvision.transforms import ToTensor
    from PIL import Image

    assert len(pred_surfaces) == len(gt_surfaces)
    n = len(pred_surfaces)  # typically 6 heliostats

    fig, axs = plt.subplots(2, n, figsize=(2 * n, 4), dpi=150, constrained_layout=True)

    if n == 1:
        axs = np.array([[axs[0]], [axs[1]]])

    for col in range(n):
        pred = pred_surfaces[col].cpu().detach().numpy()  # [4, 8, 8]
        gt = gt_surfaces[col].cpu().detach().numpy()  # [4, 8, 8]

        # Combine the 4 facets into a 2x2 grid (top-left = facet 0, top-right = facet 1, etc.)
        pred_top = np.hstack((pred[0], pred[1]))
        pred_bottom = np.hstack((pred[2], pred[3]))
        pred_combined = np.vstack((pred_top, pred_bottom))

        gt_top = np.hstack((gt[0], gt[1]))
        gt_bottom = np.hstack((gt[2], gt[3]))
        gt_combined = np.vstack((gt_top, gt_bottom))

        im = axs[0, col].imshow(pred_combined, cmap="seismic", origin="lower", vmin=-0.002, vmax=0.002)
        axs[1, col].imshow(gt_combined, cmap="seismic", origin="lower", vmin=-0.002, vmax=0.002)

        axs[0, col].set_title(f"Pred {col + 1}")
        axs[1, col].set_title(f"GT {col + 1}")
        axs[0, col].axis('off')
        axs[1, col].axis('off')

    axs[0, 0].set_ylabel("Predicted")
    axs[1, 0].set_ylabel("Ground Truth")

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
    fig.colorbar(im, ax=axs, label='Z deviation [m]', shrink=0.6, location='right')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = ToTensor()(image)

    plt.show()
    plt.close(fig)


def log_surface_heatmaps(pred_surfaces, gt_surfaces, surface_names, writer, epoch, tag="SurfaceComparisonGrid"):
    """
    Logs a 6x2 grid of predicted and ground-truth surface heatmaps to TensorBoard.

    pred_surfaces and gt_surfaces: list of tensors with shape [4, 8, 8] (Z values for 4 facets).
    writer: TensorBoard writer
    """
    import matplotlib.pyplot as plt
    import io
    import random
    from torchvision.transforms import ToTensor
    from PIL import Image

    assert len(pred_surfaces) == len(gt_surfaces)
    n = len(pred_surfaces)  # typically 6 heliostats

    fig, axs = plt.subplots(2, n, figsize=(2 * n, 4), dpi=150)

    for col in range(n):
        pred = pred_surfaces[col].cpu().detach().numpy()  # [4, 8, 8]
        gt = gt_surfaces[col].cpu().detach().numpy()      # [4, 8, 8]

        # Combine 4 facets into a 2x2 grid
        pred_top = np.hstack((pred[0], pred[1]))
        pred_bottom = np.hstack((pred[2], pred[3]))
        pred_combined = np.vstack((pred_top, pred_bottom))

        gt_top = np.hstack((gt[0], gt[1]))
        gt_bottom = np.hstack((gt[2], gt[3]))
        gt_combined = np.vstack((gt_top, gt_bottom))

        im = axs[0, col].imshow(pred_combined, cmap="seismic", origin="lower", vmin=-0.002, vmax=0.002)
        axs[1, col].imshow(gt_combined, cmap="seismic", origin="lower", vmin=-0.002, vmax=0.002)

        axs[0, col].set_title(f"Pred: {surface_names[col]}", fontsize=6)
        axs[1, col].set_title(f"GT:   {surface_names[col]}", fontsize=6)
        axs[0, col].axis('off')
        axs[1, col].axis('off')

    axs[0, 0].set_ylabel("Predicted")
    axs[1, 0].set_ylabel("Ground Truth")

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label='Z deviation [m]')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = ToTensor()(image)
    if writer:
        writer.add_image(tag, image, global_step=epoch)
    plt.close(fig)

def log_surface_heatmaps_2(
    pred_surfaces,
    gt_surfaces,
    surface_names,
    writer=None,
    epoch=0,
    tag="SurfaceComparisonGrid",
    *,
    vmin=-0.002,
    vmax=0.002,
    cmap="seismic",
    title_fontsize=16,      # column titles
    rowlabel_fontsize=16,   # rotated row labels
    cbar_fontsize=12,
    tick_fontsize=10,
    rowlabel_x=0.28,        # move left/right inside the label column (0..1). Smaller => more left.
    save_path=None,
    show=True,
    close=True
):
    """
    Clean 2 x N grid with:
      - Rotated row labels on the far left ("GT", "Prediction")
      - Column titles on the top row
      - Colorbar in a fixed right column (wide enough so label never gets cut)
    Logs to TensorBoard if writer is provided.
    """

    assert len(pred_surfaces) == len(gt_surfaces) == len(surface_names)
    n = len(pred_surfaces)

    # Figure size scales with N; extra width for left labels and right colorbar
    fig_w = max(7.5, 1.5 * n + 2.6)   # tune per taste
    fig_h = 4.6
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=180)

    # Grid: 2 rows; n + 2 columns  -> [label | images x n | colorbar]
    gs = GridSpec(
        nrows=2,
        ncols=n + 2,
        figure=fig,
        width_ratios=[0.10] + [1.00] * n + [0.12],  # wider cbar so label fits
        height_ratios=[1.0, 1.0],
        wspace=0.04,
        hspace=0.02,  # tighter row spacing
    )

    # Left label axes (top = GT, bottom = Prediction)
    ax_lbl_gt   = fig.add_subplot(gs[0, 0]); ax_lbl_gt.axis("off")
    ax_lbl_pred = fig.add_subplot(gs[1, 0]); ax_lbl_pred.axis("off")

    # Move labels a bit left inside their skinny column (no overlap with images)
    ax_lbl_gt.text(rowlabel_x, 0.5, "GT", rotation=90,
                   ha="center", va="center", fontsize=rowlabel_fontsize)
    ax_lbl_pred.text(rowlabel_x, 0.5, "Prediction", rotation=90,
                     ha="center", va="center", fontsize=rowlabel_fontsize)

    last_im = None
    for col in range(n):
        ax_top = fig.add_subplot(gs[0, col + 1])
        ax_bot = fig.add_subplot(gs[1, col + 1])

        pred = pred_surfaces[col].detach().cpu().numpy()  # [4,8,8]
        gt   = gt_surfaces[col].detach().cpu().numpy()

        pred_combined = np.vstack([np.hstack([pred[0], pred[1]]),
                                   np.hstack([pred[2], pred[3]])])
        gt_combined   = np.vstack([np.hstack([gt[0], gt[1]]),
                                   np.hstack([gt[2], gt[3]])])

        # TOP: GT, BOTTOM: Prediction
        last_im = ax_top.imshow(gt_combined,   cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax_bot.imshow(pred_combined,           cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)

        ax_top.set_title(str(surface_names[col]), fontsize=title_fontsize, pad=2)

        for ax in (ax_top, ax_bot):
            ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")

    # Colorbar on the right, spanning both rows
    cax = fig.add_subplot(gs[:, n + 1])
    cbar = fig.colorbar(last_im, cax=cax)
    # 270° keeps the label on the right and “leaning inward”; increase labelpad to avoid clipping
    cbar.set_label("Z deviation [m]", fontsize=cbar_fontsize, rotation=270, labelpad=14)
    cbar.ax.tick_params(labelsize=cbar_fontsize)

    # Optional: TensorBoard logging
    if writer is not None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        image = Image.open(buf)
        image = ToTensor()(image)
        writer.add_image(tag, image, global_step=epoch)

    # Save / show
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=180)
    if show:
        plt.show()
    if close:
        plt.close(fig)


def log_3d_surface_gallery(predicted_surfaces, writer, epoch, tag="Surface3DGallery"):
    """
    Plots 5 randomly selected predicted surfaces as 3D plots in a row and logs to TensorBoard.

    predicted_surfaces: list of tensors, each of shape [4, 8, 8, 3] (XYZ surface for all 4 facets)
    """
    import matplotlib.pyplot as plt
    import io
    import random
    from torchvision.transforms import ToTensor
    from PIL import Image

    # Randomly select 5 heliostats
    if len(predicted_surfaces) < 5:
        raise ValueError("Need at least 5 surfaces to plot gallery.")
    selected = random.sample(predicted_surfaces, 5)

    fig, axs = plt.subplots(1, 5, figsize=(20, 4), subplot_kw={'projection': '3d'}, dpi=150)

    for i, surface in enumerate(selected):
        surface = surface.detach().cpu()
        ax = axs[i]

        for facet in surface:
            X, Y, Z = facet[..., 0], facet[..., 1], facet[..., 2]
            ax.plot_surface(X, Y, Z, alpha=0.7, edgecolor='k', color='orange')

        ax.set_title(f"H{i + 1}")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-0.002, 0.002)
        ax.axis('off')
        ax.view_init(elev=30, azim=60)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = ToTensor()(image)
    if writer:
        writer.add_image(tag, image, global_step=epoch)
    plt.close(fig)

def log_3d_surface_overlay_gallery(predicted_surfaces, gt_surfaces, translation_vector, writer, epoch,
                                   tag="Surface3DOverlayGallery"):
    """
    Plots 3 predicted + ground truth heliostat surfaces overlaid (after facet alignment) and logs to TensorBoard.
    """
    import matplotlib.pyplot as plt
    import io
    import random
    from torchvision.transforms import ToTensor
    from PIL import Image

    assert len(predicted_surfaces) == len(gt_surfaces)

    n = 3
    indices = random.sample(range(len(predicted_surfaces)), n)

    fig, axs = plt.subplots(1, n, figsize=(5*n, 4), subplot_kw={'projection': '3d'}, dpi=150)

    if n == 1:
        axs = [axs]

    for i, idx in enumerate(indices):
        pred = predicted_surfaces[idx].detach().cpu()  # [4, 8, 8, 3]
        gt = gt_surfaces[idx].detach().cpu()

        ax = axs[i]
        pred_combined = combine_facets(pred, translation_vector)
        gt_combined = combine_facets(gt, translation_vector)

        # Plot GT in green
        ax.plot_trisurf(gt_combined[:, 0], gt_combined[:, 1], gt_combined[:, 2], color='green',
                        alpha=0.6, edgecolor='none')
        # Plot prediction in red
        ax.plot_trisurf(pred_combined[:, 0], pred_combined[:, 1], pred_combined[:, 2], color='red',
                        alpha=0.6, edgecolor='none')

        ax.set_title(f"H{idx + 1}", fontsize=10)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-0.002, 0.002)
        ax.view_init(elev=30, azim=60)
        ax.axis('off')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    image = Image.open(buf)
    image = ToTensor()(image)
    if writer:
        writer.add_image(tag, image, global_step=epoch)
    plt.close(fig)



def combine_facets(facets, translation_vector):
    """
    Applies translation vector to each facet as done in overwrite_scenario
    and returns a combined [N, 3] surface.

    facets: Tensor of shape [4, 8, 8, 3]
    translation_vector: Tensor of shape [4, 3]
    """
    surfaces = []
    for i in range(4):
        facet = facets[i]  # shape [8, 8, 3]
        translation = translation_vector[i].to(facet).view(1, 1, 3)
        translated_facet = facet + translation
        surfaces.append(translated_facet.reshape(-1, 3))  # flatten to [64, 3]

    combined = torch.cat(surfaces, dim=0)  # shape: [256, 3]
    return combined


def evaluate_model_on_test_set(test_set, model, device, criterion, heliostat_aim_point, aim_point_area,
                               old_scenario, xy_grid,
                               translation_vector, prototype_surface, batch_size, get_ideal_surface=None,
                               get_aug_surface=None, log_dir=None,
                               use_canting_inputs=False):

    # Load these from saved file if needed --> for helisotat norm
    min_vals = torch.tensor([-58.0915, 29.2537, 1.5110], device = device)
    max_vals = torch.tensor([157.8804, 243.5274, 2.0130], device = device)

    first_overwrite = True

    mae_list = []
    flux_mae_list = []
    pred_surfaces_list = []
    gt_surfaces_list = []
    pred_surface_list_3dplot = []
    gt_surfaces_list_3d = []
    aug_surface_keys_all = []
    accuracies=[]
    heliostat_positions_all = []
    heliostat_names_all = []

    model.eval()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,
                                              drop_last=False)
    total_loss = 0

    if log_dir:
        writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard_logs"))
    else:
        writer = None

    model_args = model

    with torch.no_grad():
        for i, (inputs, z_points) in enumerate(test_loader, 0):
            grad_ids = get_grad_tensor_ids()
            if use_canting_inputs:
                sun_pos, flux_img, heliostat_pos_norm, canting_vecs, heliostat_name, aug_surface_key = inputs
                canting_vecs = canting_vecs.to(device)
            else:
                sun_pos, flux_img, heliostat_pos_norm, heliostat_name, aug_surface_key = inputs
                canting_vecs = None
            sun_pos = sun_pos.to(device)
            flux_img = flux_img.to(device)
            heliostat_pos_norm = heliostat_pos_norm.to(device)
            aug_surface_keys_all.extend(aug_surface_key)


            surface_pool = [Surface(copy.deepcopy(prototype_surface)) for _ in range(batch_size)]
            for surface in surface_pool:
                for facet in surface.facets:
                    facet.control_points = torch.zeros(8, 8, 3, device=device)

            z_cntrl_points = model(flux_img, sun_pos, heliostat_pos_norm, targetID=None, canting_vecs=canting_vecs)

            z_pred = z_cntrl_points[0]
            z_mean = z_pred.view(z_pred.shape[0], -1).mean(dim=1, keepdim=True).view(-1, 1, 1, 1)
            z_pred_centered = z_pred - z_mean
            z_cntrl_points = (z_pred_centered,)

            z = z_cntrl_points[0][0].detach().unsqueeze(-1)
            pred_surface = torch.cat([xy_grid, z], dim=3)
            pred_surface_list_3dplot.append(pred_surface)
            key = aug_surface_key[0]

            try:
                surface_ref_cpu = get_aug_surface(key)
                surface_ref = surface_ref_cpu.to(device)
                del surface_ref_cpu
            except FileNotFoundError:
                print(f"Skipping missing surface {key}")
                continue

            z_pred = pred_surface[..., 2]
            z_gt = surface_ref[..., 2] - surface_ref[..., 2].mean()

            pred_surfaces_list.append(z_pred)
            gt_surfaces_list.append(z_gt)

            gt_surface_3d = surface_ref.detach().unsqueeze(0) if surface_ref.ndim == 3 else surface_ref
            gt_surfaces_list_3d.append(gt_surface_3d)

            is_main_process = True

            mae = log_augmented_surface_errors_single(pred_surface, surface_ref, writer, is_main_process,
                                                aug_key=key, epoch=i)
            mae_list.append(mae)

            if writer:
                log_3d_surface_comparison(pred_surface, surface_ref, writer, is_main_process, epoch=i, aug_key=key)

            heliostat_pos = heliostat_pos_norm * (max_vals - min_vals) + min_vals


            # Store each heliostat's position and name
            for j in range(heliostat_pos.shape[0]):
                pos = heliostat_pos[j]  # shape: [3]
                heliostat_positions_all.append(pos.detach().cpu().numpy())
                heliostat_names_all.append(aug_surface_key[j])

            print("Length of list:", len(heliostat_positions_all))

            new_scenario, first_overwrite = overwrite_scenario(
                heliostat_aim_point, heliostat_pos, z_cntrl_points[0], old_scenario,
                prototype_surface, xy_grid, batch_size=heliostat_pos.shape[0], first_overwrite = first_overwrite,
                device=device,
                heliostat_names=heliostat_name, get_ideal_surface=get_ideal_surface,
                translation_vector=translation_vector, surface_pool=surface_pool
            )

            sun_pos_single = sun_pos[:, 0, :]

            torch.cuda.empty_cache()

            sun_pos_for_raytracing = convert_3d_point_to_4d_format(sun_pos_single[0], device=device)
            pred_image = raytracing(new_scenario, sun_pos_for_raytracing, aim_point_area, batch_size=batch_size,
                                    show_image=False, device=device,
                                    evalaute_test_data=True)
            pred_image = pred_image.unsqueeze(1).to(device)

            flux_img = flux_img[:, 0:1, :, :]

            flux_img_norm = normalize_per_image(flux_img)
            pred_image_norm = normalize_per_image(pred_image)

            # --- Sum Normalization ---
            flux_img_sum_norm = flux_img / (flux_img.sum(dim=[1, 2, 3], keepdim=True) + 1e-8) * 100
            pred_image_sum_norm = pred_image / (pred_image.sum(dim=[1, 2, 3], keepdim=True) + 1e-8) * 100

            loss = criterion(flux_img_norm, pred_image_norm)   #chose appropriate normalization here!
            flux_mae = loss.item()
            flux_mae_list.append(flux_mae)

            #accuracies.append((1.0 - flux_mae))

            total_loss += loss.item()

            flux_accuracy = give_accuracy(flux_img_sum_norm, pred_image_sum_norm, batch_mode=True, normalize=False)
            accuracies.extend(flux_accuracy.detach().cpu().tolist())

            pred_image_norm = normalize_tensor_for_tb(pred_image)
            flux_img_norm = normalize_tensor_for_tb(flux_img)

            if writer:
                # Log image pair and loss to TensorBoard
                writer.add_scalar("Loss/test_step", loss.detach().cpu().item(), i)

                # Ensure shapes match [B, C, H, W]
                assert pred_image_norm.shape == flux_img_norm.shape, (f"Shape mismatch: pred={pred_image_norm.shape}, "
                                                                      f"gt={flux_img_norm.shape}")
                # Concatenate side-by-side: [B, C, H, W*2]
                comparison_image = torch.cat([pred_image_norm, flux_img_norm], dim=3)  # concatenate along width

                writer.add_image("Test/Pred_vs_GT", comparison_image[0], global_step=i)

                log_3d_surface_comparison(pred_surface, surface_ref, writer, is_main_process, epoch=i, aug_key=key)

    if is_main_process and len(pred_surfaces_list) >= 6:
        selected_indices = random.sample(range(len(pred_surfaces_list)), 6)
        selected_preds = [pred_surfaces_list[i] for i in selected_indices]
        selected_gts = [gt_surfaces_list[i] for i in selected_indices]
        selected_names = [aug_surface_keys_all[i] for i in selected_indices]

    if writer:
        log_surface_heatmaps_2(selected_preds, selected_gts, selected_names, writer, epoch=0)

    # Save lists to file
    if log_dir:  # use the experiment log dir if available
        save_path = os.path.join(log_dir, "mae_results.pkl")
    else:
        save_path = "results.pkl"  # default fallback

    with open(save_path, "wb") as f:
        pickle.dump({
            "mae_list": mae_list,
            "flux_mae_list": flux_mae_list
        }, f)

    print(f"✅ Saved results to {save_path}")

    avg_flux_mae = sum(flux_mae_list) / len(flux_mae_list)
    print(f"✅ Average FluxImage MAE over {len(flux_mae_list)} test samples: {avg_flux_mae:.6f}")
    print("")

    avg_mae = sum(mae_list) / len(mae_list)
    print(f"\n✅ Average SurfaceError over {len(mae_list)} test samples: {avg_mae:.6f}")

    # Convert to tensor for statistics
    accuracies_tensor = torch.tensor(accuracies)

    median_acc = torch.median(accuracies_tensor).item()
    mean_acc = torch.mean(accuracies_tensor).item()
    min_acc = torch.min(accuracies_tensor).item()
    max_acc = torch.max(accuracies_tensor).item()
    q1 = torch.quantile(accuracies_tensor, 0.25).item()
    q3 = torch.quantile(accuracies_tensor, 0.75).item()
    iqr = q3 - q1

    print("\n📊 Flux Image Accuracy (1 - rel MAE):")
    print(f"→ Mean:    {mean_acc:.7f}")
    print(f"→ Median:  {median_acc:.7f}")
    print(f"→ Min:     {min_acc:.7f}")
    print(f"→ Max:     {max_acc:.7f}")
    print(f"→ Q1:      {q1:.7f}")
    print(f"→ Q3:      {q3:.7f}")
    print(f"→ IQR:     {iqr:.7f}")

    # Convert to tensor for statistics
    mae_tensor = torch.tensor(mae_list)

    median_mae = torch.median(mae_tensor).item()
    mean_mae = torch.mean(mae_tensor).item()
    min_mae = torch.min(mae_tensor).item()
    max_mae = torch.max(mae_tensor).item()
    q1_mae = torch.quantile(mae_tensor, 0.25).item()
    q3_mae = torch.quantile(mae_tensor, 0.75).item()
    iqr_mae = q3_mae - q1_mae

    print("\n📊 Flux Image MAE (relative to sum=100):")
    print(f"→ Mean:    {mean_mae:.7f}")
    print(f"→ Median:  {median_mae:.7f}")
    print(f"→ Min:     {min_mae:.7f}")
    print(f"→ Max:     {max_mae:.7f}")
    print(f"→ Q1:      {q1_mae:.7f}")
    print(f"→ Q3:      {q3_mae:.7f}")
    print(f"→ IQR:     {iqr_mae:.7f}")

    # Convert list to arrays
    positions_np = np.stack(heliostat_positions_all)  # shape: (N, 3)
    accuracies_np = np.array(accuracies)
    names = heliostat_names_all

    """plt.figure(figsize=(12, 8))
    plt.scatter(positions_np[:, 0], positions_np[:, 1], c=accuracies_np, cmap='viridis', s=100, edgecolors='k')
    plt.colorbar(label="Flux Accuracy")

    # Add labels with TEST name and accuracy
    for i, name in enumerate(names):
        label = f"{name}: {accuracies_np[i]:.2f}"
        plt.text(positions_np[i, 0] + 1, positions_np[i, 1] + 1, label, fontsize=8)

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("📍 Test Heliostat Positions with Accuracy")
    plt.grid(True)
    #plt.tight_layout()
    plt.show()"""



