"""
utils.py

Utility functions used throughout the DeepLARTS models.

Includes:
- Plotting functions for surface tensors
- Input adjustment helpers
- Random seed setting for reproducibility
- Cluster-safe logging

These functions are imported by `my_deepLarts.py` and
`styleGAN2_surfaces.py`. Also taken from previous work of @parg_ma
and based on Lewen et al. (2024)
(https://doi.org/10.48550/arXiv.2408.10802)
"""

import functools
import math
import os
from typing import (
    Callable,
    cast,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from PIL import Image
import torch
from matplotlib import cm
import torch as th
from yacs.config import CfgNode
#import data
#from defaults import get_cfg_defaults, load_config_file
#if TYPE_CHECKING:
 #   from heliostat_models import AlignedHeliostat, Heliostat
#import nurbs
# import kornia
import numpy as np
from torchvision.transforms.functional import resized_crop, crop
from torchvision.transforms import Resize
import random
#from idealSpans import give_spans
import matplotlib.pyplot as plt
T = TypeVar('T')

import torchvision.utils as vutils
#from unet.UNet_3Plus_ import UNet_3Plus
from torch.utils.data import DataLoader
#import deepLarts_utils as utilsDL
import torch.nn.functional as F
import pandas as pd

try:
    from Unsupervised_learning.Dataloader_Jan import transform_targetID, process_data_for_model, transform_helPos
except:
    pass
try:
    from skimage.metrics import structural_similarity as ssim
except:
    pass

class listSet():
    def __init__(self, liste):
        
        self.deflID = th.tensor(range(len(liste)))
    
    def __len__(self):
        return len(self.deflID)
    
    def __getitem__(self, idx):
        deflID = self.deflID[idx]
        return deflID
    

def split_IDs_across_GPUs(liste, rank, world_size, CLUSTER):
    data_folder = listSet(liste)
    if CLUSTER:
        datasampler = th.utils.data.distributed.DistributedSampler(data_folder, 
                                                                   num_replicas=world_size,
                                                                   rank=rank)
        
        folder_loader = th.utils.data.DataLoader(data_folder, 
                                                 batch_size=1, 
                                                 shuffle=False, 
                                                 sampler=datasampler)
    
    else:
        folder_loader = th.utils.data.DataLoader(data_folder, 
                                                 batch_size=1, 
                                                 shuffle=False)
    
    return folder_loader

def adjust_inputs(flux=None, sunPos=None, targetID=None, num_init_filters=8):

    if not flux == None:
        nsunPos = flux.size(1)
        multiplyier = 1 + int(num_init_filters / nsunPos + 0.5)
        flux = flux.repeat((1, multiplyier, 1, 1))[:,:num_init_filters, :, :]
    if not sunPos == None:
        nsunPos = sunPos.size(1)
        multiplyier = 1 + int(num_init_filters / nsunPos + 0.5)
        sunPos = sunPos.repeat((1, multiplyier, 1))[:,:num_init_filters, :]
    if not targetID == None:
        nsunPos = targetID.size(1)
        multiplyier = 1 + int(num_init_filters / nsunPos + 0.5)
        targetID = targetID.repeat((1, multiplyier))[:,:num_init_filters]
    
    return flux, sunPos, targetID


def compute_distribution_stats(distribution):
    """
    Compute the min, first quartile (Q1), median, third quartile (Q3), and max of a distribution.
    
    Args:
    - data (torch.Tensor): Tensor containing the distribution data.
    
    Returns:
    - dict: A dictionary containing the min, Q1, median, Q3, and max values.
    """
    
    # Ensure the input is a 1D tensor
    assert distribution.dim() == 1, "Input data must be a 1D tensor"
    
    # Compute the statistics
    min_val = torch.min(distribution)
    q1 = torch.quantile(distribution, 0.25)
    median = torch.median(distribution)
    mean = torch.mean(distribution)
    q3 = torch.quantile(distribution, 0.75)
    max_val = torch.max(distribution)
    
    return {
        "min": min_val.item(),
        "Q1": q1.item(),
        "median": median.item(),
        "mean": mean.item(),
        "Q3": q3.item(),
        "max": max_val.item()
    }


def fraction_smaller_than(tensor, value):
    """
    Calculate the fraction of values in the tensor that are smaller than the given value.
    
    Args:
    - tensor (torch.Tensor): Input tensor of dimension [N].
    - value (float): The threshold value.
    
    Returns:
    - float: The fraction of values in the tensor that are smaller than the given value.
    """
    
    # Ensure the input is a 1D tensor
    assert tensor.dim() == 1, "Input tensor must be a 1D tensor"
    
    # Count the number of values smaller than the given value
    count_smaller = torch.sum(tensor < value).item()
    
    # Calculate the fraction
    fraction = count_smaller / tensor.size(0)
    
    return fraction



def give_angle_between_normals(ideal_normal_vecs, real_normal_vecs):
    tomrad = 1000
    
    norm_ideal = th.norm(ideal_normal_vecs, dim=1, keepdim=True)
    norm_real = th.norm(real_normal_vecs, dim=1, keepdim=True)
    
    ideal_normal_vecs = ideal_normal_vecs / (norm_ideal)  
    real_normal_vecs = real_normal_vecs / (norm_real)
    
    angles = th.acos(
        th.clip(th.sum(ideal_normal_vecs * real_normal_vecs, dim=-1), -1, 1),
    ).detach().cpu()
    
    angles = tomrad*angles

    return angles


def give_test_bool(sessionstring, cfg):
    
    skiplist = cfg.DEEPLARTS.TRAIN.SKIPLIST + cfg.DEEPLARTS.TRAIN.PICEOFFACETMISSINGLIST
    hellist = cfg.DEEPLARTS.VALID.TESTSET
    validsession = cfg.DEEPLARTS.VALID.SESSIONS 
    
    validsession_bool = sessionstring.split('_')[1] in validsession
    
    validheliostat_bool = sessionstring.split('_')[0] in hellist
    
    continue_bool = not sessionstring in skiplist
    
    return (validsession_bool and validheliostat_bool and continue_bool)
    

# def give_accuracy(target, prediction):

#     diff = target - prediction 
#     difference_sum = th.sum(th.abs(diff))
#     target_sum = th.sum(th.abs(target))
#     acc = 1 - difference_sum/target_sum
    
#     return acc

def set_normals_and_surface_dir(defldir, helname, cfg):
    filedir = os.path.join(defldir)
    # defldir = os.Path(defldir)
    filelist = os.listdir(defldir)
    
    surfacedir = ''
    normalsdir = ''
    
    validsessions = cfg.DEEPLARTS.VALID.SESSIONS 
    
    session = 0 
    for file in filelist:

        file_tags = file.split('_')
        if len(file_tags) == 0:
            print('file tag = 0, continue')
            continue
        
        session_ = file_tags[-1].split('.')[0]
        if not session_ in validsessions:
            continue
        
        # if file_tags[0] == 'Helio' and file_tags[2] == 'Rim0' and file_tags[3] == 'LocalResults':
        #     surfacedir = filedir + f'\{file}'
        
        if file_tags[0] == 'Helio' and file_tags[1]==helname and file_tags[2] == 'Rim0' and file_tags[3] == 'STRAL-Input':
            normalsdir = filedir + f'\{file}'
            session = session_
    
    if normalsdir=='':
        print(f'Defl to {helname} does not exist!')
        raise Exception
        
    cfg.defrost()
    cfg.H.DEFLECT_DATA.FILENAME = normalsdir
    cfg.H.DEFLECT_DATA.ZS_PATH  = surfacedir
    cfg.freeze()
    
    return session
    
def interpolate_splines(zcntrl, target_size):

    if zcntrl.size(1) == 1:
        zcntrl = surface_to_facets(zcntrl)
    
    target_size = int(target_size/2)
    zcntrl = th.nn.functional.interpolate(zcntrl, size=(target_size), mode='bilinear')
    
    zcntrl = facets_to_surface(zcntrl)
    return zcntrl
    
    
def print_cluster(string, cluster, rank):
    if cluster:
        if rank == 0:
            print(string)
    else:
        print(string)
    
def give_accuracy(target, prediction, batch_mode=False, normalize=False):
    assert target.size() == prediction.size(), "Target and Prediction must have same size"
    
    if target.ndim == 2:  # (H, W)
        target = target.unsqueeze(0).unsqueeze(0)  # Batch- und Channel-Dimension hinzufügen
    elif target.ndim == 3:  # (C, H, W)
        target = target.unsqueeze(0)  # Nur Batch-Dimension hinzufügen
    elif target.ndim == 4:  # (B, C, H, W)
        pass  # Keine Änderung notwendig
    else:
        raise ValueError(f"Unerwartete Tensor-Dimensionen: {target.shape}. Erwartet: (H, W), (C, H, W), oder (B, C, H, W).")
    
    if prediction.ndim == 2:  # (H, W)
        prediction = prediction.unsqueeze(0).unsqueeze(0)  # Batch- und Channel-Dimension hinzufügen
    elif prediction.ndim == 3:  # (C, H, W)
        prediction = prediction.unsqueeze(0)  # Nur Batch-Dimension hinzufügen
    elif prediction.ndim == 4:  # (B, C, H, W)
        pass  # Keine Änderung notwendig
    else:
        raise ValueError(f"Unerwartete Tensor-Dimensionen: {prediction.shape}. Erwartet: (H, W), (C, H, W), oder (B, C, H, W).")
        
    if normalize==True:
        target = target/th.sum(target, dim=(2,3), keepdim=True)
        prediction = prediction/th.sum(prediction, dim=(2,3), keepdim=True)
    elif normalize==False:
        pass
    else:
        raise Exception("You entered a wrong value for normalize!")

    if batch_mode:
        # Calculate batch-wise accuracy
        diff = target - prediction
        difference_sum = th.sum(th.abs(diff), dim=(1, 2, 3))
        target_sum = th.sum(th.abs(target), dim=(1, 2, 3))
        
        # Avoid division by zero
        target_sum[target_sum == 0] = 1  # Set zeros to 1 to avoid division by zero
        
        acc = 1 - difference_sum / target_sum
    else:
        # Calculate accuracy for a single tensor
        diff = target - prediction 
        difference_sum = th.sum(th.abs(diff))
        target_sum = th.sum(th.abs(target))
        
        # Avoid division by zero
        if target_sum.item() == 0:
            return 0.0
        
        acc = 1 - difference_sum / target_sum
    
    return acc


def give_mean_pixel_error(target, prediction, batch_mode=False):
    if target.ndim == 2:  # (H, W)
        target = target.unsqueeze(0).unsqueeze(0)  # Batch- und Channel-Dimension hinzufügen
    elif target.ndim == 3:  # (C, H, W)
        target = target.unsqueeze(0)  # Nur Batch-Dimension hinzufügen
    elif target.ndim == 4:  # (B, C, H, W)
        pass  # Keine Änderung notwendig
    else:
        raise ValueError(f"Unerwartete Tensor-Dimensionen: {target.shape}. Erwartet: (H, W), (C, H, W), oder (B, C, H, W).")
    
    if prediction.ndim == 2:  # (H, W)
        prediction = prediction.unsqueeze(0).unsqueeze(0)  # Batch- und Channel-Dimension hinzufügen
    elif prediction.ndim == 3:  # (C, H, W)
        prediction = prediction.unsqueeze(0)  # Nur Batch-Dimension hinzufügen
    elif prediction.ndim == 4:  # (B, C, H, W)
        pass  # Keine Änderung notwendig
    else:
        raise ValueError(f"Unerwartete Tensor-Dimensionen: {prediction.shape}. Erwartet: (H, W), (C, H, W), oder (B, C, H, W).")
        
    if batch_mode:
        b, c, h, w = target.size()
        
        diff = target - prediction 
        difference_sum = th.sum(th.abs(diff), dim=(1, 2, 3))
        mean_pixel_error = difference_sum/ (c*h*w) 
    else:
        b, c, h, w = target.size()
        
        diff = target - prediction 
        difference_sum = th.sum(th.abs(diff))
        mean_pixel_error = difference_sum/ (b*c*h*w) 
    
    return mean_pixel_error


def cartesian_to_spherical(cartesian_tensor):
    # azi = 0 im süden
    # azi = -180 im norden    
    # azi = -90 im osten
    # azi = 90 im westen
    # Die Darstellung der Sonnenkoordinaten entspricht dem Blick vom Heliostatenfeld
    # auf den Turm 
    
    # Calculate the azimuth (azimut)
    cartesian_tensor[:,0] = cartesian_tensor[:,0]#/th.linalg.norm(cartesian_tensor[:,0])
    cartesian_tensor[:,1] = cartesian_tensor[:,1]#/th.linalg.norm(cartesian_tensor[:,1])
    
    azimuth = th.rad2deg(torch.atan2(-1*cartesian_tensor[:, 1], cartesian_tensor[:, 0]) - th.pi/2)
    mask = (azimuth < -180)
    azimuth[mask] = azimuth[mask] + 360
    
    # Calculate the elevation
    xy_projection = torch.sqrt(cartesian_tensor[:, 0] ** 2 + cartesian_tensor[:, 1] ** 2)
    elevation = th.rad2deg(torch.atan2(cartesian_tensor[:, 2], xy_projection))

    # Combine the azimuth and elevation into a single tensor
    spherical_tensor = torch.stack((azimuth, elevation), dim=1)
    
    return spherical_tensor


def spherical_to_cartesian(spherical_tensor):
    # azimuth and elevation in degrees
    azimuth = spherical_tensor[:, 0]
    elevation = spherical_tensor[:, 1]
    
    # Convert degrees to radians
    azimuth_rad = torch.deg2rad(azimuth)
    elevation_rad = torch.deg2rad(elevation)
    
    # Calculate Cartesian coordinates
    x = torch.cos(elevation_rad) * torch.cos(azimuth_rad + torch.pi/2)
    y = -torch.cos(elevation_rad) * torch.sin(azimuth_rad + torch.pi/2)
    z = torch.sin(elevation_rad)
    
    # Combine x, y, z into a single tensor
    cartesian_tensor = torch.stack((x, y, z), dim=1)
    
    return cartesian_tensor

def create_square_grid_tensor(x1, x2, y1, y2, N):
    """
    Erstellt ein quadratisches Gitter in der Form eines Tensors (2, N^2).
    """
    x = torch.linspace(x1, x2, N)
    y = torch.linspace(y1, y2, N)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    grid_x_flat = grid_x.flatten()
    grid_y_flat = grid_y.flatten()
    grid_tensor = torch.stack((grid_x_flat, grid_y_flat), dim=0)
    return grid_tensor


def give_helPos_larger_grid(nhels, device):
    
    helpos = []
    min_distance = 10
    
    # halbkreis vom Turm weg
    for distance in range(5, 400, 5):
        zs = (th.rand(nhels, device=device)/1.8+1.5).unsqueeze(0)   

        phi_max = th.pi/6
        phis = th.linspace(-phi_max, phi_max, nhels, device=device)
        rs = distance
        xs = rs*th.cos(phis).unsqueeze(0)    
        ys = rs*th.sin(phis).unsqueeze(0)   

        helPositions = th.cat((ys, xs-100, zs), dim=0)
        helpos.append(helPositions)

    helpos = th.cat(helpos, dim=1)
    # we add additional close heliostat position because of the more difficult pattern
    N_close_heliostats=30
    close_positions = create_square_grid_tensor(x1=-60, 
                                         x2=60, 
                                         y1=10, 
                                         y2=60, 
                                         N=N_close_heliostats)
    
    # close_positions = close_positions.view(2,-1)
    zs_close = (th.rand(N_close_heliostats**2, device=device)/1.8+1.5).unsqueeze(0)   
    helpos_close = th.cat((close_positions, zs_close), dim=0)
    
    helpos = th.cat((helpos, helpos_close), dim=1)    
    
    mask = (helpos[1,:] > 10)
    helpos = helpos[:,mask]
    
    mask = (helpos[0,:] > -150)
    helpos = helpos[:,mask]

    mask = (helpos[0,:] < 150)
    helpos = helpos[:,mask]
    
    # die valid Heliostatpositionen
    nhels_valid = 5
    helPos_valid = []
    for distance in range(5, 375, 20):
        zs = (th.rand(nhels_valid, device=device)/1.8+1.5).unsqueeze(0)   

        phi_max = th.pi/7
        phis = th.linspace(-phi_max, phi_max, nhels_valid, device=device)
        rs = distance
        xs = rs*th.cos(phis).unsqueeze(0)    
        ys = rs*th.sin(phis).unsqueeze(0)   

        helPositions = th.cat((ys, xs - 75, zs), dim=0)
        helPos_valid.append(helPositions)
    
    helPos_valid = th.cat(helPos_valid, dim=1)

    mask = (helPos_valid[1,:] > 10)
    helPos_valid = helPos_valid[:,mask]
    
    mask = (helPos_valid[0,:] > -150)
    helPos_valid = helPos_valid[:,mask]

    mask = (helPos_valid[0,:] < 150)
    helPos_valid = helPos_valid[:,mask]
    
    return helpos.swapaxes(0,1), helPos_valid.swapaxes(0,1)


def give_helPos_on_field_list(cfg, cluster, real=True, nhelpos=40):
    helPos_list = []
    
    if cluster:
        posdir = cfg.DIRECTORIES.JUWELS.POSDIR
    else:
        posdir = 'heldata\heliostat_position_dictionary.npy'
        
    helPos_dic = np.load(posdir, allow_pickle=True).item()
    teslist = cfg.DEEPLARTS.VALID.TESTSET
    validlist = [s.split('_')[0] for s in cfg.DEEPLARTS.VALID.VALIDSET]
    
    if not cluster: defllist = give_defl_list(cfg, cluster)
    
    skiplist = cfg.DEEPLARTS.TRAIN.SKIPLIST + cfg.DEEPLARTS.TRAIN.PICEOFFACETMISSINGLIST
    testpos = []
    validpos = []
    deflpos = []
    for key in helPos_dic:
        if key in teslist:
            testpos.append(helPos_dic[key])
        
        if key in validlist:
            validpos.append(helPos_dic[key])
            
        if not cluster:
            if key in defllist and not key in teslist and not key in validlist and not key in skiplist:
                deflpos.append(helPos_dic[key])
                
        helPos_list.append(helPos_dic[key])
    
    
    helpos_art, helPos_valid = give_helPos_larger_grid(nhelpos, device='cpu')
    
    plotting=True
    if plotting and not cluster:
        fig, ax = plt.subplots(1,1, figsize=(8, 6))
        ax.scatter(np.array(helPos_list)[:,0], np.array(helPos_list)[:,1], alpha=0.5, label='field')

        # ax.scatter(
        #     helpos_art[:, 0], 
        #     helpos_art[:, 1], 
        #     c='gray',  # Marker color
        #     s=15,      # Marker size
        #     alpha=0.5, # Transparency for better clarity with overlapping points
        #     edgecolors='k',
        #     label="train"# Black edge around markers for better contrast
        # )
        
        # ax.scatter(
        #     np.array(helPos_valid)[:,0], 
        #     np.array(helPos_valid)[:,1], 
        #     c='yellow',  # Marker color
        #     s=30,      # Marker size
        #     alpha=0.5, # Transparency for better clarity with overlapping points
        #     edgecolors='k',
        #     label="valid/test simulative"# Black edge around markers for better contrast
        # )
        
        ax.scatter(np.array(deflpos)[:,0], np.array(deflpos)[:,1], color='black', marker='.', label='train')
        ax.scatter(np.array(testpos)[:,0], np.array(testpos)[:,1], color='red', marker='.', label='test')
        ax.scatter(np.array(validpos)[:,0], np.array(validpos)[:,1], color='orange', marker='.', label='valid')

        ax.legend(loc="best")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_title("Position of Train and Test Heliostats", fontsize=16, fontweight='bold')
        ax.set_xlabel("Position East / m", fontsize=14)
        ax.set_ylabel("Position North / m", fontsize=14)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', which='major', labelsize=12)
        fig.tight_layout()
        
        fig, ax = plt.subplots(1,1)
        ax.scatter((helpos_art)[:,1], (helpos_art)[:,2])
        ax.scatter(np.array(helPos_list)[:,1], np.array(helPos_list)[:,2])
        ax.scatter(np.array(deflpos)[:,1], np.array(deflpos)[:,2], color='red', marker='x')
        ax.scatter(np.array(testpos)[:,1], np.array(testpos)[:,2], color='black', marker='x')

        ax.grid()
        
        fig, ax = plt.subplots(1,1)
        ax.hist(np.array(helPos_list)[:,2])
        plt.show()
        
    if real == False:
        helPos_list = helpos_art.tolist()
    
    helPos_valid = helPos_valid.tolist()
    
    return helPos_list, helPos_valid


def give_defl_list(cfg, cluster):
    
    if cluster:
        defldir = cfg.DIRECTORIES.JUWELS.DEFLFILLED
    else:
        defldir = cfg.DIRECTORIES.LOCAL.DEFLFILLED
        
    defllist = os.listdir(defldir)
    
    skiplist = cfg.DEEPLARTS.TRAIN.SKIPLIST
    bad_defl_list = cfg.DEEPLARTS.TRAIN.PICEOFFACETMISSINGLIST
        
    hellist = []
    for defl in defllist:
        
        tags = defl.split("_")
        
        sessionstring = tags[1] + "_" + tags[-1].split(".")[0]
        if sessionstring in skiplist:
            continue
        
        if sessionstring in bad_defl_list:
            continue
            
        helname = tags[1]

        hellist.append(helname)
    
    return hellist


def cropp_img_around_mass_center(cfg, targetID, flux, centerOfMass=None):
    # centerOfMass = center_of_mass(flux.numpy()) 
    
    if centerOfMass==None:
        flux_numpy = flux[0,0,:,:].clone().detach().cpu().numpy()
        flux_numpy[flux_numpy < 0.5*np.max(flux_numpy)] = 0
        centerOfMass = center_of_mass(flux_numpy)
        
    n_grid = flux.size(-1)
    # the final dimension in m
    targetEvaluationPlaneSize = cfg.AC.TARGET.EVALUATION_SIZE
    
    # give the dimensions of the target in m
    set_target_pos(targetID, cfg)
    
    planex = cfg.AC.RECEIVER.PLANE_Y  
    planey = cfg.AC.RECEIVER.PLANE_X 

    npixelx = targetEvaluationPlaneSize/planex * n_grid
    npixely = targetEvaluationPlaneSize/planey * n_grid
    
    center_pixelX = centerOfMass[0]
    center_pixelY = centerOfMass[1]
    
    start_crop_x = int(center_pixelX - npixelx/2 + 0.5)
    start_crop_y = int(center_pixelY - npixely/2 + 0.5)

    flux = crop(flux, start_crop_x, start_crop_y, int(npixelx + 0.5), int(npixely + 0.5))
    # flux = flux.unsqueeze(0).unsqueeze(0)
    flux = Resize(size= (cfg.DEEPLARTS.TRAIN.TARGET_IMAGE_SIZE, cfg.DEEPLARTS.TRAIN.TARGET_IMAGE_SIZE))(flux)
    
    return flux.squeeze(0).squeeze(0)


def crop_resize_image(image_tensor, center_x, center_y):
    image_size = 64
    crop_size = 64
    image_physical_size = torch.tensor([7.0, 8.0])  # Dimensions in meters
    target_physical_size = torch.tensor([5.0, 5.0])  # Desired dimensions in meters

    # Convert physical dimensions to pixel dimensions
    pixel_size = image_physical_size / image_size

    # Calculate crop region coordinates
    crop_x1 = int(center_x - (crop_size // 2) * pixel_size[0])
    crop_y1 = int(center_y - (crop_size // 2) * pixel_size[1])
    crop_x2 = int(center_x + ((crop_size // 2) - 1) * pixel_size[0])
    crop_y2 = int(center_y + ((crop_size // 2) - 1) * pixel_size[1])

    # Crop the image tensor
    cropped_image = image_tensor[crop_y1:crop_y2, crop_x1:crop_x2]

    # Resize the cropped image
    resized_image = F.interpolate(cropped_image.unsqueeze(0).unsqueeze(0), size=(target_physical_size[1], target_physical_size[0]), mode='bilinear', align_corners=False)
    resized_image = resized_image.squeeze(0)

    return resized_image


def facets_to_surface(facets):
    
    facet1 = facets[:,0,:,:]
    facet2 = facets[:,1,:,:]
    facet3 = facets[:,2,:,:]
    facet4 = facets[:,3,:,:]
    
    upper = th.cat([facet1, facet3], dim=1).unsqueeze(1)
    lower = th.cat([facet2, facet4], dim=1).unsqueeze(1)
    surface = th.cat([upper, lower], dim=3)
    
    return surface
    

def interpolate_cntrl_points_to_surface(cntrl_points, pixel_size=1024):
    if cntrl_points.ndim == 2:  # (C, H, W)
        cntrl_points = cntrl_points.unsqueeze(0)
    elif cntrl_points.ndim == 3:  # (C, H, W)
        cntrl_points = cntrl_points.unsqueeze(0)  # Nur Batch-Dimension hinzufügen
    elif cntrl_points.ndim == 4:  # (B, C, H, W)
        pass  # Keine Änderung notwendig
    else:
        raise ValueError(f"Unerwartete Tensor-Dimensionen: {cntrl_points.shape}. Erwartet: (C, H, W), oder (B, C, H, W).")
    if not cntrl_points.size(1) == 4:    
        cntrl_points = surface_to_facets(cntrl_points)
        

    surface = th.nn.functional.interpolate(cntrl_points, size=pixel_size, mode='bilinear', antialias=True)
    
    if surface.size(1) == 4:
        surface = facets_to_surface(surface)
        
    return surface.squeeze(0).squeeze(0)
    
    
    
def surface_to_facets(surface):
    batch_size, _, w, h = surface.size()

    # Split the surface tensor into upper and lower halves
    upper = surface[:, :, :int(w/2), :]
    lower = surface[:, :, int(w/2):, :]

    # Split the upper half tensor into facet1 and facet3
    facet1 = upper[:, :, :, :int(h/2)]
    facet2 = upper[:, :, :, int(h/2):]

    # Split the lower half tensor into facet2 and facet4
    facet3 = lower[:, :, :, :int(h/2)]
    facet4 = lower[:, :, :, int(h/2):]

    # Stack the facets along the second dimension
    facets = th.stack([facet1, facet2, facet3, facet4], dim=1).squeeze(2)

    return facets


def hel_para_from_df(df, index, device):
    
    def make_number(number):
        number = float(number.replace(',','.'))
        return number
    
    sunPosE = make_number(df.at[index, 'SunPosE'])
    sunPosN = make_number(df.at[index, 'SunPosN'])
    sunPosU = make_number(df.at[index, 'SunPosU'])
    sunPos = th.tensor([[(sunPosE), (sunPosN), (sunPosU)]], device=device).to(th.get_default_dtype())
    
    targetID = th.tensor([df.at[index, 'CalibrationTargetId']], device=device).to(th.get_default_dtype())
    
    created_at = df.at[index, 'CreatedAt']

    CenterOfMass = (df.at[index, 'CenterOfMassX'], df.at[index, 'CenterOfMassY'])
    
    aimPoint = th.tensor((df.at[index, 'AimPointX'], df.at[index, 'AimPointY'], df.at[index, 'AimPointZ']), device=device).unsqueeze(0).to(th.get_default_dtype())
    
    return sunPos, targetID, created_at, CenterOfMass, aimPoint


def load_defaults(configdir):
    cfg_default = get_cfg_defaults()
    if configdir:
        print(f"load: {configdir}")
        # config_file = os.path.join("configs", config_file_name)
        cfg = load_config_file(cfg_default, configdir)
    else:
        
        print('----------- The DEFAULT configuration is used! ----------')
        cfg = cfg_default

    cfg.freeze()

    # if cfg.USE_FLOAT64:
    #     th.set_default_dtype(th.float64)
    # else:
    #     th.set_default_dtype(th.float32)
    return cfg
    



def distance_to_tower(helPos):
    
    distance = th.sqrt(th.dot(helPos, helPos))   
    
    return distance

def give_helPos_in_distance(distance, nhels, device):
    
    zs = (th.rand(nhels, device=device)/2+1.5).unsqueeze(0)   
    
    phi_max = 4/5*np.pi/2
    phis = th.linspace(-phi_max, phi_max, nhels, device=device)
    rs = distance + (th.rand(1, device=device) - 0.5)
    
    xs = rs*th.sin(phis).unsqueeze(0)    
    ys = rs*th.cos(phis).unsqueeze(0)   
    
    helPositions = th.cat((xs, ys, zs), dim=0)
    
    return helPositions

def give_helPos_in_distance(distance, device):
    
    zs = (th.rand(1, device=device)/2+1.5)   
    
    phi_max = 4/5*np.pi/2
    phis = 2*th.rand(1, device=device)*phi_max -phi_max
    rs = distance + (th.rand(1, device=device) - 0.5)
    
    xs = rs*th.sin(phis).unsqueeze(0)    
    ys = rs*th.cos(phis).unsqueeze(0)   
    
    helPositions = th.cat((xs, ys, zs.unsqueeze(0)), dim=0)
    
    return helPositions



@th.no_grad()
def save_data_CPU_efficient(fluxes, zcntrls, sunPositions, targetIDs,
                            helPositions, xy_aligns, helIDs, 
                            sessions, directory, subscript, helIDs_augment=None):

    th.save(fluxes.detach().cpu(), f'{directory}/fluxes_{subscript}.pt')
    th.save(zcntrls.detach().cpu(), f'{directory}/zcntrl_{subscript}.pt')
    th.save(sessions.detach().cpu(), f'{directory}/sessions_{subscript}.pt')
    th.save(sunPositions.detach().cpu(), f'{directory}/sunPos_{subscript}.pt')
    th.save(targetIDs.detach().cpu(), f'{directory}/targetIDs_{subscript}.pt')
    th.save(helPositions.detach().cpu(), f'{directory}/helPos_{subscript}.pt')
    th.save(helIDs.detach().cpu(), f'{directory}/helIDs_{subscript}.pt')
    th.save(xy_aligns.detach().cpu(), f'{directory}/xy_aligns_{subscript}.pt')
    
    if th.is_tensor(helIDs_augment):
        th.save(helIDs_augment.detach().cpu(), f'{directory}/helIDs_augment_{subscript}.pt')
    
    print('saved')
    
    
    
def give_helID(helname):
    
    alphabet = np.array(['a','b','c','d','e','f','g','h','i','j','k','l',
               'm','n','o','p','q','r','s','t','u','v','w','x','y','z'])
    
    ids_1 = np.array(['1','2','3','4','5','6','7','8','9','10',
                   '11','12','13','14','15','16','17','18','19','20',
                   '21','22','23','24','25','26'])

    ids_2 = np.array(['01','02','03','04','05','06','07','08','09','10',
                   '11','12','13','14','15','16','17','18','19','20',
                   '21','22','23','24','25','26'])
    
    letters = helname[0:2].lower()
    id_final = helname[2:4]
    
    letter1 = letters[0]
    mask = (letter1 == alphabet) 
    id1 = ids_1[mask]

    letter2 = letters[1]
    mask = (letter2 == alphabet) 
    id2 = ids_2[mask]
    
    helID = int(id1[0] + id2[0] + str(id_final))
    return helID


def give_helname(helID):
    
    alphabet = np.array(['a','b','c','d','e','f','g','h','i','j','k','l',
               'm','n','o','p','q','r','s','t','u','v','w','x','y','z'])
    
    ids_1 = np.array(['1','2','3','4','5','6','7','8','9','10',
                   '11','12','13','14','15','16','17','18','19','20',
                   '21','22','23','24','25','26'])

    ids_2 = np.array(['01','02','03','04','05','06','07','08','09','10',
                   '11','12','13','14','15','16','17','18','19','20',
                   '21','22','23','24','25','26'])
    
    letter1 = helID[0]
    id_final = helID[3:5]
    letter2 = helID[1:3]
    
    mask = (letter1 == ids_1) 
    id1 = alphabet[mask]

    mask = (letter2 == ids_2) 
    id2 = alphabet[mask]
    
    helname = id1[0].upper() + id2[0].upper() + str(id_final)
    return helname


def give_flux(H, ENV, sunPos, cfg, targetIDs, aimPoints=False, set_alignment=None, 
              randomAimPoints=False, centralize=False, randomize_CSR=False, geometry_model='mean',
              scale_flux_to_energy_bool=False):
    
    # geometry_model can be: 'mean', 'randomized'
    if geometry_model == 'mean':
        pass
    
    elif geometry_model == 'randomized' and not th.is_tensor(set_alignment):
        cfg.defrost()
        cfg.H.GEOMETRY.ALPHA = np.random.uniform(*cfg.H.GEOMETRY.RANDOMIZE.ALPHA_PARA)
        cfg.H.GEOMETRY.BETA = np.random.uniform(*cfg.H.GEOMETRY.RANDOMIZE.BETA_PARA)
        cfg.H.GEOMETRY.AXIS1K = np.random.uniform(*cfg.H.GEOMETRY.RANDOMIZE.AXIS1K_PARA)
        cfg.H.GEOMETRY.AXIS2K = np.random.uniform(*cfg.H.GEOMETRY.RANDOMIZE.AXIS2K_PARA)
        cfg.H.GEOMETRY.GAMMA = np.random.uniform(*cfg.H.GEOMETRY.RANDOMIZE.GAMMA_PARA)
        cfg.H.GEOMETRY.DELTA = np.random.uniform(*cfg.H.GEOMETRY.RANDOMIZE.DELTA_PARA)
        cfg.H.GEOMETRY.RANDOMIZE_GEOMETRY = True
        cfg.freeze()
        
    elif geometry_model == 'randomized' and th.istensor(set_alignment):
        raise Exception('It doesnt make sense to give the exact alignment of the heliostat and still randomzing the geometry model!')

    else:
        raise Exception('geometry_model must be either mean or randomized!')
    
    # if not aimPoints==None and not targetIDs==None:
    #     raise Exception('You must give either targetIDs OR the precise Aimpoint')

    fluxes, xy_align, rays_missed, aimPoints, nrays = data.generate_training_dataset( H=H,
                                                                                ENV=ENV,
                                                                                sun_directions=sunPos,
                                                                                cfg=cfg,
                                                                                targetIDs=targetIDs,
                                                                                set_alignment=set_alignment,
                                                                                aimPoints=aimPoints,
                                                                                randomAimPoints=randomAimPoints,
                                                                                centralize=centralize,
                                                                                geometry_model=geometry_model,
                                                                                randomize_CSR=randomize_CSR)   
    
    helPos = H.position_on_field
    
    if scale_flux_to_energy_bool:

        fluxes, spillage = scale_flux_to_energy(cfg, 
                                                fluxes.squeeze(0), 
                                                helPos, 
                                                sunPos, 
                                                aimPoints, 
                                                nrays,
                                                rays_missed)
        
        fluxes = fluxes.unsqueeze(0)
        
        return fluxes, xy_align, rays_missed, spillage
    
    else:
        return fluxes, xy_align, rays_missed
    

def set_target_pos(targetID, cfg, randomAimPoint=False):
    
    randomAimPointX = np.random.uniform(-0.5, 0.5)
    randomAimPointY = np.random.uniform(-0.5, 0.5)
    
    cfg_target = cfg.AC.TARGET
    
    if targetID == 7:
        cfg.defrost()
        target_center = cfg_target.TARGET7.CENTER
        
        if randomAimPoint: 
            target_center[0] += randomAimPointX
            target_center[2] += randomAimPointY

        cfg.AC.RECEIVER.CENTER = target_center
        
        cfg.AC.RECEIVER.PLANE_X = cfg_target.TARGET7.PLANE_X
        cfg.AC.RECEIVER.PLANE_Y = cfg_target.TARGET7.PLANE_Y
        
        # consider smaller piece of the target
        # cfg.AC.RECEIVER.PLANE_X = targetPlaneSize
        # cfg.AC.RECEIVER.PLANE_Y = targetPlaneSize
        
        cfg.freeze()
    
    elif targetID == 6:
        target_center = cfg_target.TARGET6.CENTER
        
        if randomAimPoint: 
            target_center[0] += randomAimPointX
            target_center[2] += randomAimPointY
            
        cfg.defrost()
        cfg.AC.RECEIVER.CENTER = target_center
        cfg.AC.RECEIVER.PLANE_X = cfg_target.TARGET6.PLANE_X
        cfg.AC.RECEIVER.PLANE_Y = cfg_target.TARGET6.PLANE_Y

        # consider smaller piece of the target
        # cfg.AC.RECEIVER.PLANE_X = targetPlaneSize
        # cfg.AC.RECEIVER.PLANE_Y = targetPlaneSize
        
        cfg.freeze()
        
    elif targetID == 3:
        target_center = cfg_target.TARGET3.CENTER
        
        if randomAimPoint: 
            target_center[0] += randomAimPointX
            target_center[2] += randomAimPointY
            
        cfg.defrost()
        cfg.AC.RECEIVER.CENTER = target_center
        cfg.AC.RECEIVER.PLANE_X = cfg_target.TARGET3.PLANE_X
        cfg.AC.RECEIVER.PLANE_Y = cfg_target.TARGET3.PLANE_Y
        
        # consider smaller piece of the target
        # cfg.AC.RECEIVER.PLANE_X = targetPlaneSize
        # cfg.AC.RECEIVER.PLANE_Y = targetPlaneSize
        cfg.freeze()
    
    elif targetID == 'rec':
        target_center = [0, 0, 55] 
        cfg.defrost()
        
        cfg.AC.RECEIVER.CENTER = target_center
        # cfg.AC.RECEIVER.PLANE_X = 4.35  
        # cfg.AC.RECEIVER.PLANE_Y = 5.22
        targetPlaneSize = 5
        # consider smaller piece of the target
        cfg.AC.RECEIVER.PLANE_X = targetPlaneSize
        cfg.AC.RECEIVER.PLANE_Y = targetPlaneSize
        cfg.freeze()

        cfg.freeze()
        
    else:
        raise Exception('You must chose a valid targetID!!')
    
    return target_center


def to_tensorboard(writer, prefix, epoch, lr=None, loss=None, raw_loss=None, image=None, plot_interval=None, index=None):
    with th.no_grad():
        # Plot loss to Tensorboard
        if not index == None:
            iteration_str = f"_{index}"
        else:
            iteration_str = ""
        if writer:
            assert prefix, "prefix string cannot be empty"
            if lr:
                writer.add_scalar(
                    f"{prefix}/lr"+iteration_str, lr, epoch)
            if loss:
                writer.add_scalar(
                    f"{prefix}/loss"+iteration_str, loss.item(), epoch)
            if raw_loss:
                writer.add_scalar(
                    f"{prefix}/raw_loss"+iteration_str, raw_loss.item(), epoch)
                # Plot target images to TensorBoard
            if not image== None:
                assert plot_interval, "If image is given, plot interval must be defined"
                writer.add_image(
                    f"{prefix}/prediction"+iteration_str,
                    colorize(image),
                    epoch,
                )


def calculateSunAngles(
        hour: int,
        minute: int,
        sec: int,
        day: int,
        month: int,
        year: int,
        observerLatitude: float,
        observerLongitude: float,
) -> Tuple[float, float]:
    # in- and outputs are in degree
    if (
            hour < 0 or hour > 23
            or minute < 0 or minute > 59
            or sec < 0 or sec > 59
            or day < 1 or day > 31
            or month < 1 or month > 12
    ):
        raise ValueError(
            "at least one value exeeded time range in calculateSunAngles")

    else:
        observerLatitudeInt = observerLatitude / 180.0 * math.pi
        observerLongitudeInt = observerLongitude / 180.0 * math.pi

        pressureInput = 1.01325  # Pressure in bar
        temperature = 20  # Temperature in °C

        UT = hour + minute / 60.0 + sec / 3600.0
        pressure = pressureInput / 1.01325
        delta_t = 0.0

        if month <= 2:
            dyear = year - 1.0
            dmonth = month + 12.0
        else:
            dyear = year
            dmonth = month

        trunc1 = math.floor(365.25 * (dyear - 2000))
        trunc2 = math.floor(30.6001 * (dmonth + 1))
        JD_t = trunc1 + trunc2 + day + UT / 24.0 - 1158.5
        t = JD_t + delta_t / 86400.0

        # standard JD and JDE
        # (useless for the computation, they are computed for completeness)
        # JDE = t + 2452640
        # JD = JD_t + 2452640

        # HELIOCENTRIC LONGITUDE
        # linear increase + annual harmonic
        ang = 0.0172019 * t - 0.0563
        heliocLongitude = (
            1.740940
            + 0.017202768683 * t
            + 0.0334118 * math.sin(ang)
            + 0.0003488 * math.sin(2.0 * ang)
        )

        # Moon perturbation
        heliocLongitude = \
            heliocLongitude + 0.0000313 * math.sin(0.2127730 * t - 0.585)
        # Harmonic correction
        heliocLongitude = (
            heliocLongitude
            + 0.0000126 * math.sin(0.004243 * t + 1.46)
            + 0.0000235 * math.sin(0.010727 * t + 0.72)
            + 0.0000276 * math.sin(0.015799 * t + 2.35)
            + 0.0000275 * math.sin(0.021551 * t - 1.98)
            + 0.0000126 * math.sin(0.031490 * t - 0.80)
        )

        # END HELIOCENTRIC LONGITUDE CALCULATION
        # Correction to longitude due to notation
        t2 = t / 1000.0
        heliocLongitude = (
            heliocLongitude
            + (
                (
                    (-0.000000230796 * t2 + 0.0000037976) * t2
                    - 0.000020458
                ) * t2
                + 0.00003976
            ) * t2 * t2
        )

        delta_psi = 0.0000833 * math.sin(0.0009252 * t - 1.173)

        # Earth axis inclination
        epsilon = (
            -0.00000000621 * t
            + 0.409086
            + 0.0000446 * math.sin(0.0009252 * t + 0.397)
        )
        # Geocentric global solar coordinates
        geocSolarLongitude = heliocLongitude + math.pi + delta_psi - 0.00009932

        s_lambda = math.sin(geocSolarLongitude)
        rightAscension = math.atan2(
            s_lambda * math.cos(epsilon),
            math.cos(geocSolarLongitude),
        )

        declination = math.asin(math.sin(epsilon) * s_lambda)

        # local hour angle of the sun
        hourAngle = (
            6.30038809903 * JD_t
            + 4.8824623
            + delta_psi * 0.9174
            + observerLongitudeInt
            - rightAscension
        )

        c_lat = math.cos(observerLatitudeInt)
        s_lat = math.sin(observerLatitudeInt)
        c_H = math.cos(hourAngle)
        s_H = math.sin(hourAngle)

        # Parallax correction to Right Ascension
        d_alpha = -0.0000426 * c_lat * s_H
        # topOCRightAscension = rightAscension + d_alpha
        # topOCHourAngle = hourAngle - d_alpha

        # Parallax correction to Declination
        topOCDeclination = \
            declination - 0.0000426 * (s_lat - declination * c_lat)

        s_delta_corr = math.sin(topOCDeclination)
        c_delta_corr = math.cos(topOCDeclination)
        c_H_corr = c_H + d_alpha * s_H
        s_H_corr = s_H - d_alpha * c_H

        # Solar elevation angle, without refraction correction
        elevation_no_refrac = math.asin(
            s_lat * s_delta_corr
            + c_lat * c_delta_corr * c_H_corr
        )

        # Refraction correction:
        # it is calculated only if elevation_no_refrac > elev_min
        elev_min = -0.01

        if elevation_no_refrac > elev_min:
            refractionCorrection = (
                0.084217 * pressure
                / (273.0 + temperature)
                / math.tan(
                    elevation_no_refrac
                    + 0.0031376 / (elevation_no_refrac + 0.089186)
                )
            )
        else:
            refractionCorrection = 0

        # elevationAngle = \
        #     np.pi / 2 - elevation_no_refrac - refractionCorrection
        elevationAngle = elevation_no_refrac + refractionCorrection
        elevationAngle = elevationAngle * 180 / math.pi

        # azimuthAngle = math.atan2(
        #     s_H_corr,
        #     c_H_corr * s_lat - s_delta_corr/c_delta_corr * c_lat,
        # )
        azimuthAngle = -math.atan2(
            s_H_corr,
            c_H_corr * s_lat - s_delta_corr/c_delta_corr * c_lat,
        )
        azimuthAngle = azimuthAngle * 180 / math.pi

    return azimuthAngle, elevationAngle


def get_sun_array(
        *datetime: List[int],
        **observer: float,
) -> Tuple[torch.Tensor, List[List[Union[int, float]]]]:
    """Arguments must be in descending order (years, months, days, ...)."""
    years = [2021]
    months = [6]
    days = [21]
    hours = list(range(6, 19))
    minutes = [0, 30]
    secs = [0]

    num_args = len(datetime)
    if num_args == 0:
        print("generate values for 21.06.2021")
    if num_args >= 1:
        years = datetime[0]
        if num_args >= 2:
            months = datetime[1]
            if num_args >= 3:
                days = datetime[2]
                if num_args >= 4:
                    hours = datetime[3]
                    if num_args >= 5:
                        minutes = datetime[4]
                        if num_args >= 6:
                            secs = datetime[5]

    observerLatitude = observer.get('latitude', 50.92)
    observerLongitude = observer.get('longitude', 6.36)

    # sunAngles = np.empty((3,1440,2))
    extras = []
    ae = []
    for year in years:
        for month in months:
            for day in days:
                for hour in hours:
                    for minute in minutes:
                        for sec in secs:
                            azi, ele = calculateSunAngles(
                                hour,
                                minute,
                                sec,
                                day,
                                month,
                                year,
                                observerLatitude,
                                observerLongitude,
                            )
                            extras.append([
                                year,
                                month,
                                day,
                                hour,
                                minute,
                                sec,
                                azi,
                                ele,
                            ])
                            ae.append([azi, ele])
    ae = th.tensor(ae)
    sun_vecs = ae_to_vec(ae[:, 0], ae[:, 1])
    return sun_vecs, extras


def angle_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    angles = th.acos(th.clamp(
        (
            batch_dot(a, b).squeeze(-1)
            / (
                th.linalg.norm(a, dim=-1)
                * th.linalg.norm(b, dim=-1)
            )
        ),
        -1.0,
        1.0,
    )).squeeze(-1)
    return angles


def axis_angle_rotation(
        axis: torch.Tensor,
        angle_rad: torch.Tensor,
) -> torch.Tensor:
    cos = th.cos(angle_rad)
    icos = 1 - cos
    sin = th.sin(angle_rad)
    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]
    axis_sq = axis**2

    rows = [
        th.stack(row, dim=-1)
        for row in [
                [
                    cos + axis_sq[..., 0] * icos,
                    x * y * icos - z * sin,
                    x * z * icos + y * sin,
                ],
                [
                    y * x * icos + z * sin,
                    cos + axis_sq[..., 1] * icos,
                    y * z * icos - x * sin,
                ],
                [
                    z * x * icos - y * sin,
                    z * y * icos + x * sin,
                    cos + axis_sq[..., 2] * icos,
                ],
        ]
    ]
    return th.stack(rows, dim=1)


# changed by Jan
# def axis_angle_rotation(
#         axis: torch.Tensor,
#         angle_rad: torch.Tensor,
# ) -> torch.Tensor:
#     cos = th.cos(angle_rad)
#     icos = 1 - cos
#     sin = th.sin(angle_rad)
#     x = axis[..., 0]
#     y = axis[..., 1]
#     z = axis[..., 2]
#     axis_sq = axis**2

#     rows = [
#         th.stack(row, dim=-1)
#         for row in [
#                 [
#                     cos + axis_sq[..., 0] * icos,
#                     x * y * icos - z * sin,
#                     x * z * icos + y * sin,
#                 ],
#                 [
#                     y * x * icos + z * sin,
#                     cos + axis_sq[..., 1] * icos,
#                     y * z * icos - x * sin,
#                 ],
#                 [
#                     z * x * icos - y * sin,
#                     z * y * icos + x * sin,
#                     cos + axis_sq[..., 2] * icos,
#                 ],
#         ]
#     ]
#     return th.stack(rows, dim=1)


def get_rot_matrix(
        start: torch.Tensor,
        target: torch.Tensor,
) -> torch.Tensor:
    rot_angle = angle_between(start, target)
    # Handle parallel start/target normals.
    if rot_angle == 0:
        return th.eye(3)
    elif rot_angle == math.pi:
        return -th.eye(3)
    rot_axis = th.cross(target, start)
    rot_axis /= th.linalg.norm(rot_axis)
    full_rot = axis_angle_rotation(rot_axis, rot_angle)
    return full_rot


def rot_x_mat(
        angle_rad: torch.Tensor,
        dtype: th.dtype,
        device: th.device,
) -> torch.Tensor:
    cos_angle = th.cos(angle_rad)
    sin_angle = th.sin(angle_rad)
    zero = th.tensor(0, dtype=dtype, device=device)
    one = th.tensor(1, dtype=dtype, device=device)

    return th.stack([
        th.stack([one, zero, zero]),
        th.stack([zero, cos_angle, -sin_angle]),
        th.stack([zero, sin_angle, cos_angle]),
    ])


def rot_y_mat(
        angle_rad: torch.Tensor,
        dtype: th.dtype,
        device: th.device,
) -> torch.Tensor:
    cos_angle = th.cos(angle_rad)
    sin_angle = th.sin(angle_rad)
    zero = th.tensor(0, dtype=dtype, device=device)
    one = th.tensor(1, dtype=dtype, device=device)

    return th.stack([
        th.stack([cos_angle, zero, sin_angle]),
        th.stack([zero, one, zero]),
        th.stack([-sin_angle, zero, cos_angle]),
    ])


def rot_z_mat(
        angle_rad: torch.Tensor,
        dtype: th.dtype,
        device: th.device,
) -> torch.Tensor:
    cos_angle = th.cos(angle_rad)
    sin_angle = th.sin(angle_rad)
    zero = th.tensor(0, dtype=dtype, device=device)
    one = th.tensor(1, dtype=dtype, device=device)

    return th.stack([
        th.stack([cos_angle, -sin_angle, zero]),
        th.stack([sin_angle, cos_angle, zero]),
        th.stack([zero, zero, one]),
    ])


def get_z_alignments(
        heliostat: 'Heliostat',
        sun_directions: torch.Tensor,
) -> torch.Tensor:
    return th.stack([
        cast(
            'AlignedHeliostat',
            heliostat.align(sun_dir),
        ).alignment[..., -1, :]
        for sun_dir in sun_directions
    ])


def deflec_facet_zs(
        points: torch.Tensor,
        normals: torch.Tensor,
) -> torch.Tensor:
    """Calculate z values for a surface given by normals at x-y-planar
    positions.

    We are left with a different unknown offset for each z value; we
    assume this to be constant.
    """
    distances = horizontal_distance(
        points.unsqueeze(0),
        points.unsqueeze(1),
    )
    distances, closest_indices = distances.sort(dim=-1)
    del distances
    # Take closest point that isn't the point itself.
    closest_indices = closest_indices[..., 1]

    midway_normal = normals + normals[closest_indices]
    midway_normal /= th.linalg.norm(midway_normal, dim=-1, keepdims=True)

    rot_z_90deg = th.tensor(
        [
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ],
        dtype=points.dtype,
        device=points.device,
    )

    connector = points[closest_indices] - points
    connector_norm = th.linalg.norm(connector, dim=-1)
    orthogonal = th.matmul(rot_z_90deg, connector.T).T
    orthogonal /= th.linalg.norm(orthogonal, dim=-1, keepdims=True)
    tilted_connector = th.cross(orthogonal, midway_normal)
    tilted_connector /= th.linalg.norm(tilted_connector, dim=-1, keepdims=True)

    angle = th.acos(th.clamp(
        (
            batch_dot(tilted_connector, connector).squeeze(-1)
            / connector_norm
        ),
        -1,
        1,
    ))
    zs = connector_norm * th.tan(angle)

    return zs


def _all_angles(
        points: torch.Tensor,
        normals: torch.Tensor,
        closest_indices: torch.Tensor,
        remaining_indices: torch.Tensor,
) -> torch.Tensor:
    connector = (points[closest_indices] - points).unsqueeze(1)
    other_connectors = (
        points[remaining_indices]
        - points.unsqueeze(1)
    )
    angles = th.acos(th.clamp(
        (
            batch_dot(connector, other_connectors).squeeze(-1)
            / (
                th.linalg.norm(connector, dim=-1)
                * th.linalg.norm(other_connectors, dim=-1)
            )
        ),
        -1,
        1,
    )).squeeze(-1)

    # Give the angles a rotation direction.
    angles *= (
        1
        - 2 * (
            batch_dot(
                normals.unsqueeze(1),
                # Cross product does not support broadcasting, so do it
                # manually.
                th.cross(
                    th.tile(connector, (1, other_connectors.shape[1], 1)),
                    other_connectors,
                    dim=-1,
                ),
            ).squeeze(-1)
            < 0
        )
    )

    # And convert to 360° rotations.
    tau = 2 * th.tensor(math.pi, dtype=angles.dtype, device=angles.device)
    angles = th.where(angles < 0, tau + angles, angles)
    return angles


def _find_angles_in_other_slices(
        angles: torch.Tensor,
        num_slices: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = angles.dtype
    device = angles.device
    # Set up uniformly sized cake/pizza slices for which to find angles.
    tau = 2 * th.tensor(math.pi, dtype=dtype, device=device)
    angle_slice = tau / num_slices

    angle_slices = (
        th.arange(
            num_slices,
            dtype=dtype,
            device=device,
        )
        * angle_slice
    ).unsqueeze(-1).unsqueeze(-1)
    # We didn't calculate angles in the "zeroth" slice so we disregard them.
    angle_start = angle_slices[1:] - angle_slice / 2
    angle_end = angle_slices[1:] + angle_slice / 2

    # Find all angles lying in each slice.
    angles_in_slice = ((angle_start <= angles) & (angles < angle_end))
    return angles_in_slice, angle_slices


def deflec_facet_zs_many(
        points: torch.Tensor,
        normals: torch.Tensor,
        normals_ideal: torch.Tensor,
        num_samples: int = 4,
        use_weighted_average: bool = False,
        eps: float = 1e-6,
) -> torch.Tensor:
    """Calculate z values for a surface given by normals at x-y-planar
    positions.

    We are left with a different unknown offset for each z value; we
    assume this to be constant.
    """
    # TODO When `num_samples == 1`, we can just use the old method.
    device = points.device
    dtype = points.dtype

    distances = horizontal_distance(
        points.unsqueeze(0),
        points.unsqueeze(1),
    )
    distances, distance_sorted_indices = distances.sort(dim=-1)
    del distances
    # Take closest point that isn't the point itself.
    closest_indices = distance_sorted_indices[..., 1]

    # Take closest point in different directions from the given point.

    # For that, first calculate angles between direction to closest
    # point and all others, sorted by distance.
    angles = _all_angles(
        points,
        normals,
        closest_indices,
        distance_sorted_indices[..., 2:],
    ).unsqueeze(0)

    # Find positions of all angles in each slice except the zeroth one.
    angles_in_slice, angle_slices = _find_angles_in_other_slices(
        angles, num_samples)

    # And take the first one.angle we found in each slice. Remember
    # these are still sorted by distance, so we obtain the first
    # matching angle that is also closest to the desired point.
    #
    # We need to handle not having any slices except the zeroth one
    # extra.
    if len(angles_in_slice) > 1:
        angle_indices = th.argmax(angles_in_slice.long(), dim=-1)
    else:
        angle_indices = th.empty(
            (0, len(points)), dtype=th.long, device=device)

    # Select the angles we found for each slice.
    angles = th.gather(angles.squeeze(0), -1, angle_indices.T)

    # Handle _not_ having found an angle. We here create an array of
    # booleans, indicating whether we found an angle, for each slice.
    found_angles = th.gather(
        angles_in_slice,
        -1,
        angle_indices.unsqueeze(-1),
    ).squeeze(-1)
    # We always found something in the zeroth slice, so add those here.
    found_angles = th.cat([
        th.ones((1,) + found_angles.shape[1:], dtype=th.bool, device=device),
        found_angles,
    ], dim=0)
    del angles_in_slice

    # Set up some numbers for averaging.
    if use_weighted_average:
        angle_diffs = (
            th.cat([
                th.zeros((len(angles), 1), dtype=dtype, device=device),
                angles,
            ], dim=-1)
            - angle_slices.squeeze(-1).T
        )
        # Inverse difference in angle.
        weights = 1 / (angle_diffs + eps).T
        del angle_diffs
    else:
        # Number of samples we found angles for.
        num_available_samples = th.count_nonzero(found_angles, dim=0)

    # Finally, combine the indices of the closest points (zeroth slice)
    # with the indices of all closest points in the other slices.
    closest_indices = th.cat((
        closest_indices.unsqueeze(0),
        angle_indices,
    ), dim=0)
    del angle_indices, angle_slices

    midway_normal = normals + normals[closest_indices]
    midway_normal /= th.linalg.norm(midway_normal, dim=-1, keepdims=True)

    rot_90deg = axis_angle_rotation(
        normals_ideal, th.tensor(math.pi / 2, dtype=dtype, device=device))

    connector = points[closest_indices] - points
    connector_norm = th.linalg.norm(connector, dim=-1)
    orthogonal = th.matmul(
        rot_90deg.unsqueeze(0),
        connector.unsqueeze(-1),
    ).squeeze(-1)
    orthogonal /= th.linalg.norm(orthogonal, dim=-1, keepdims=True)
    tilted_connector = th.cross(orthogonal, midway_normal, dim=-1)
    tilted_connector /= th.linalg.norm(tilted_connector, dim=-1, keepdims=True)
    tilted_connector *= th.sign(connector[..., -1]).unsqueeze(-1)

    angle = th.acos(th.clamp(
        (
            batch_dot(tilted_connector, connector).squeeze(-1)
            / connector_norm
        ),
        -1,
        1,
    ))
    # Here, we handle values for which we did not find an angle. For
    # some reason, the NaNs those create propagate even to supposedly
    # unaffected values, so we handle them explicitly.
    angle = th.where(
        found_angles & ~th.isnan(angle),
        angle,
        th.tensor(0.0, dtype=dtype, device=device),
    )

    # Average over each slice.
    if use_weighted_average:
        zs = (
            (weights * connector_norm * th.tan(angle)).sum(dim=0)
            / (weights * found_angles.to(dtype)).sum(dim=0)
        )
    else:
        zs = (
            (connector_norm * th.tan(angle)).sum(dim=0)
            / num_available_samples
        )

    return zs


def with_outer_list(values: Union[List[T], List[List[T]]]) -> List[List[T]]:
    # Type errors come from T being able to be a list. So we ignore them
    # as "type negation" ("T can be everything except a list") is not
    # currently supported.

    if isinstance(values[0], list):
        return cast(List[List[T]], values)
    return cast(List[List[T]], [values])


def vec_to_ae(vec: torch.Tensor) -> torch.Tensor:
    
    # azimut = 90 grad im osten
    # azimut = 0 grad im norden
    # azimut = -90 grad im westen
    # azimut = 180 grad im süden

    """
    converts ENU vector to azimuth, elevation

    Parameters
    ----------
    vec : tensor (N,3)
        Batch of N spherical vectors

    Returns
    -------
    tensor
        returns Azi, Ele in ENU coordsystem

    """
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)
    device = vec.device

    north = th.tensor([0, 1, 0], dtype=vec.dtype, device=device)
    up = th.tensor([0, 0, 1], dtype=vec.dtype, device=device)

    xy_plane = vec.clone()
    xy_plane[:, 2] = 0
    xy_plane = xy_plane / th.linalg.norm(xy_plane, dim=1).unsqueeze(1)

    a = -th.rad2deg(th.arccos(th.matmul(xy_plane, north)))
    a = th.where(vec[:, 0] < 0, a, -a)

    e = -(th.rad2deg(th.arccos(th.matmul(vec, up))) - 90)
    return th.stack([a, e], dim=1)


def ae_to_vec(
        az: torch.Tensor,
        el: torch.Tensor,
        srange: float = 1.0,
        deg: bool = True,
) -> torch.Tensor:
    """
    Azimuth, Elevation, Slant range to target to East, North, Up

    Parameters
    ----------
    azimuth : float
            azimuth clockwise from north (degrees)
    elevation : float
        elevation angle above horizon, neglecting aberrations (degrees)
    srange : float
        slant range [meters]
    deg : bool, optional
        degrees input/output  (False: radians in/out)

    Returns
    --------
    e : float
        East ENU coordinate (meters)
    n : float
        North ENU coordinate (meters)
    u : float
        Up ENU coordinate (meters)
    """
    if deg:
        el = th.deg2rad(el)
        az = th.deg2rad(az)

    r = srange * th.cos(el)

    rot_vec = th.stack(
        [r * th.sin(az), r * th.cos(az), srange * th.sin(el)],
        dim=1,
    )
    
    # rot_vec = th.stack(
    #     [th.sin(el)*th.cos(az), th.sin(az)*th.sin(el), srange * th.cos(el)],
    #     dim=1,
    # )
    
    return rot_vec


def colorize(
        image_tensor: torch.Tensor,
        colormap: str = 'jet',
) -> torch.Tensor:
    """

    Parameters
    ----------
    image_tensor : tensor
        expects tensor of shape [H,W]
    colormap : string, optional
        choose_colormap. The default is 'jet'.

    Returns
    -------
    colored image tensor of CHW

    """
    image_tensor = image_tensor.clone() / image_tensor.max()
    prediction_image = image_tensor.squeeze().detach().cpu().numpy()

    color_map = cm.get_cmap('jet')
    mapped_image = th.tensor(color_map(prediction_image)).permute(2, 0, 1)
    # mapped_image8 = (255*mapped_image).astype('uint8')
    # print(colored_prediction_image.shape)

    return mapped_image


def load_config_file(
        cfg: CfgNode,
        config_file_loc: str,
        experiment_name: Optional[str] = None,
) -> CfgNode:
    if len(os.path.splitext(config_file_loc)[1]) == 0:
        config_file_loc += '.yaml'
    cfg.merge_from_file(config_file_loc)
    if experiment_name:
        cfg.merge_from_list(["NAME", experiment_name])
    cfg.freeze()
    return cfg


def flatten_aimpoints(aimpoints: torch.Tensor) -> torch.Tensor:
    X = th.flatten(aimpoints[0])
    Y = th.flatten(aimpoints[1])
    Z = th.flatten(aimpoints[2])
    aimpoints = th.stack((X, Y, Z), dim=1)
    return aimpoints


def curl(
        f: Callable[[torch.Tensor], torch.Tensor],
        arg: torch.Tensor,
) -> torch.Tensor:
    jac = th.autograd.functional.jacobian(f, arg, create_graph=True)

    rot_x = jac[2][1] - jac[1][2]
    rot_y = jac[0][2] - jac[2][0]
    rot_z = jac[1][0] - jac[0][1]

    return th.tensor([rot_x, rot_y, rot_z])


def find_larger_divisor(num: int) -> int:
    divisor = int(th.sqrt(th.tensor(num)))
    while num % divisor != 0:
        divisor += 1
    return divisor


def find_perpendicular_pair(
        base_vec: torch.Tensor,
        vecs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    half_pi = th.tensor(math.pi, device=vecs.device) / 2
    for vec_x in vecs[1:]:
        surface_direction_x = vec_x - base_vec
        surface_direction_x /= th.linalg.norm(surface_direction_x)
        for vec_y in vecs[2:]:
            surface_direction_y = vec_y - base_vec
            surface_direction_y /= th.linalg.norm(surface_direction_y)
            if th.isclose(
                    th.acos(th.dot(
                        surface_direction_x,
                        surface_direction_y,
                    )),
                    half_pi,
            ):
                return surface_direction_x, surface_direction_y
    raise ValueError('could not calculate surface normal')


def _cartesian_linspace_around(
        minval_x: Union[float, torch.Tensor],
        maxval_x: Union[float, torch.Tensor],
        num_x: int,
        minval_y: Union[float, torch.Tensor],
        maxval_y: Union[float, torch.Tensor],
        num_y: int,
        device: th.device,
        dtype: Optional[th.dtype] = None,
) -> torch.Tensor:
    if dtype is None:
        dtype = th.get_default_dtype()
    if not isinstance(minval_x, th.Tensor):
        minval_x = th.tensor(minval_x, dtype=dtype, device=device)
    if not isinstance(maxval_x, th.Tensor):
        maxval_x = th.tensor(maxval_x, dtype=dtype, device=device)
    if not isinstance(minval_y, th.Tensor):
        minval_y = th.tensor(minval_y, dtype=dtype, device=device)
    if not isinstance(maxval_y, th.Tensor):
        maxval_y = th.tensor(maxval_y, dtype=dtype, device=device)
    spline_max = 1

    minval_x = minval_x.clamp(0, spline_max)
    maxval_x = maxval_x.clamp(0, spline_max)
    minval_y = minval_y.clamp(0, spline_max)
    maxval_y = maxval_y.clamp(0, spline_max)

    points_x = th.linspace(
        minval_x, maxval_x, num_x, device=device)  # type: ignore[arg-type]
    points_y = th.linspace(
        minval_y, maxval_y, num_y, device=device)  # type: ignore[arg-type]
    points = th.cartesian_prod(points_x, points_y)
    return points


# TODO choose uniformly between spans (not super important
#      as our knots are uniform as well)
def initialize_spline_eval_points(
        rows: int,
        cols: int,
        device: th.device,
) -> torch.Tensor:
    return _cartesian_linspace_around(0, 1, rows, 0, 1, cols, device)


def initialize_spline_eval_points_perfectly(
        points: torch.Tensor,
        degree_x: int,
        degree_y: int,
        ctrl_points: torch.Tensor,
        ctrl_weights: torch.Tensor,
        knots_x: torch.Tensor,
        knots_y: torch.Tensor,
) -> torch.Tensor:
    eval_points, distances = nurbs.invert_points_slow(
            points,
            degree_x,
            degree_y,
            ctrl_points,
            ctrl_weights,
            knots_x,
            knots_y,
    )
    return eval_points


def round_positionally(x: torch.Tensor) -> torch.Tensor:
    """Round usually but round .5 decimal point depending on position.

    If the decimal point is .5, values in the lower half of `x` are
    rounded down while values in the upper half of `x` are rounded up.

    The halfway point is obtained by rounding up.
    """
    x_middle = int(th.tensor(len(x) / 2).round())

    # Round lower values down, upper values up.
    # This makes the indices become mirrored around the middle
    # index.
    lower_half = x[:x_middle]
    upper_half = x[x_middle:]
    point_five = th.tensor(0.5, device=x.device)

    lower_half = th.where(
        th.isclose(lower_half % 1, point_five),
        lower_half.floor(),
        lower_half,
    ).long()
    upper_half = th.where(
        th.isclose(upper_half % 1, point_five),
        upper_half.ceil(),
        upper_half,
    ).long()

    x = th.cat([lower_half, upper_half])
    return x


def horizontal_distance(
        a: torch.Tensor,
        b: torch.Tensor,
        ord: Union[int, float, str] = 2,
) -> torch.Tensor:
    return th.linalg.norm(b[..., :-1] - a[..., :-1], dim=-1, ord=ord)


def distance_weighted_avg(
        distances: torch.Tensor,
        points: torch.Tensor,
) -> torch.Tensor:
    # Handle distances of 0 just in case with a very small value.
    distances = th.where(
        distances == 0,
        th.tensor(
            th.finfo(distances.dtype).tiny,
            device=distances.device,
            dtype=distances.dtype,
        ),
        distances,
    )
    inv_distances = 1 / distances.unsqueeze(-1)
    weighted = inv_distances * points
    total = weighted.sum(dim=-2)
    total = total / inv_distances.sum(dim=-2)
    return total


def calc_knn_averages(
        points: torch.Tensor,
        neighbours: torch.Tensor,
        k: int,
) -> torch.Tensor:
    distances = horizontal_distance(
        points.unsqueeze(1),
        neighbours.unsqueeze(0),
    )
    distances, closest_indices = distances.sort(dim=-1)
    distances = distances[..., :k]
    closest_indices = closest_indices[..., :k]

    averaged = distance_weighted_avg(distances, neighbours[closest_indices])
    return averaged


def initialize_spline_ctrl_points(
        control_points: torch.Tensor,
        origin: torch.Tensor,
        rows: int,
        cols: int,
        h_width: float,
        h_height: float,
) -> None:
    device = control_points.device
    origin_offsets_x = th.linspace(
        -h_width / 2, h_width / 2, rows, device=device)
    origin_offsets_y = th.linspace(
        -h_height / 2, h_height / 2, cols, device=device)
    origin_offsets = th.cartesian_prod(origin_offsets_x, origin_offsets_y)
    origin_offsets = th.hstack((
        origin_offsets,
        th.zeros((len(origin_offsets), 1), device=device),
    ))
    control_points[:] = (origin + origin_offsets).reshape(control_points.shape)


def calc_closest_ctrl_points(
        control_points: torch.Tensor,
        world_points: torch.Tensor,
        k: int = 4,
) -> torch.Tensor:
    new_control_points = calc_knn_averages(
        control_points.reshape(-1, control_points.shape[-1]),
        world_points,
        k,
    )
    return new_control_points.reshape(control_points.shape)


def _make_structured_points_from_corners(
        points: torch.Tensor,
        rows: int,
        cols: int,
) -> Tuple[torch.Tensor, int, int]:
    x_vals = points[:, 0]
    y_vals = points[:, 1]

    x_min = x_vals.min()
    x_max = x_vals.max()
    y_min = y_vals.min()
    y_max = y_vals.max()

    x_vals = th.linspace(
        x_min, x_max, rows, device=x_vals.device)  # type: ignore[arg-type]
    y_vals = th.linspace(
        y_min, y_max, cols, device=y_vals.device)  # type: ignore[arg-type]

    structured_points = th.cartesian_prod(x_vals, y_vals)

    distances = th.linalg.norm(
        (
            structured_points.unsqueeze(1)
            - points[:, :-1].unsqueeze(0)
        ),
        dim=-1,
    )
    argmin_distances = distances.argmin(1)
    z_vals = points[argmin_distances, -1]
    structured_points = th.cat(
        [structured_points, z_vals.unsqueeze(-1)], dim=-1)

    return structured_points, rows, cols


def _make_structured_points_from_unique(
        points: torch.Tensor,
        tolerance: float,
) -> Tuple[torch.Tensor, int, int]:
    x_vals = points[:, 0]
    x_vals = th.unique(x_vals, sorted=True)

    prev_i = 0
    keep_indices = [prev_i]
    for (i, x) in enumerate(x_vals[1:]):
        if not th.isclose(x, x_vals[prev_i], atol=tolerance):
            prev_i = i
            keep_indices.append(prev_i)
    x_vals = x_vals[keep_indices]

    y_vals = points[:, 0]
    y_vals = th.unique(y_vals, sorted=True)

    prev_i = 0
    keep_indices = [prev_i]
    for (i, y) in enumerate(y_vals[1:]):
        if not th.isclose(y, y_vals[prev_i], atol=tolerance):
            prev_i = i
            keep_indices.append(prev_i)
    y_vals = y_vals[keep_indices]

    structured_points = th.cartesian_prod(x_vals, y_vals)

    distances = th.linalg.norm(
        (
            structured_points.unsqueeze(1)
            - points[:, :-1].unsqueeze(0)
        ),
        dim=-1,
    )
    argmin_distances = distances.argmin(1)
    z_vals = points[argmin_distances, -1]
    structured_points = th.cat(
        [structured_points, z_vals.unsqueeze(-1)], dim=-1)

    rows = len(x_vals)
    cols = len(y_vals)
    return structured_points, rows, cols


def make_structured_points(
        points: torch.Tensor,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        tolerance: float = 0.0075,
) -> Tuple[torch.Tensor, int, int]:
    if rows is None or cols is None:
        return _make_structured_points_from_unique(points, tolerance)
    else:
        return _make_structured_points_from_corners(points, rows, cols)


def initialize_spline_ctrl_points_perfectly(
        control_points: torch.Tensor,
        world_points: torch.Tensor,
        num_points_x: int,
        num_points_y: int,
        degree_x: int,
        degree_y: int,
        knots_x: torch.Tensor,
        knots_y: torch.Tensor,
        change_z_only: bool,
        change_knots: bool,
) -> None:
    new_control_points, new_knots_x, new_knots_y = nurbs.approximate_surface(
        world_points,
        num_points_x,
        num_points_y,
        degree_x,
        degree_y,
        control_points.shape[0],
        control_points.shape[1],
        knots_x if change_knots else None,
        knots_y if change_knots else None,
    )

    if not change_z_only:
        control_points[:, :, :-1] = new_control_points[:, :, :-1]
    control_points[:, :, -1:] = new_control_points[:, :, -1:]
    if change_knots:
        knots_x[:] = new_knots_x
        knots_y[:] = new_knots_y


def initialize_spline_knots_(knots: torch.Tensor, spline_degree: int) -> None:
    num_knot_vals = len(knots[spline_degree:-spline_degree])
    knot_vals = th.linspace(0, 1, num_knot_vals)
    knots[:spline_degree] = 0
    knots[spline_degree:-spline_degree] = knot_vals
    knots[-spline_degree:] = 1


def initialize_spline_knots(
        knots_x: torch.Tensor,
        knots_y: torch.Tensor,
        spline_degree_x: int,
        spline_degree_y: int,
) -> None:
    initialize_spline_knots_(knots_x, spline_degree_x)
    initialize_spline_knots_(knots_y, spline_degree_y)


def calc_ray_diffs(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # We could broadcast here but to avoid a warning, we tile manually.
    # TODO stimmt das so noch?
    return th.nn.functional.l1_loss(pred, target)


def calc_reflection_normals_(
        in_reflections: torch.Tensor,
        out_reflections: torch.Tensor,
) -> torch.Tensor:
    normals = ((in_reflections + out_reflections) / 2 - in_reflections)
    # Handle pass-through "reflection"
    normals = th.where(
        th.isclose(normals, th.zeros_like(normals[0])),
        out_reflections,
        normals / th.linalg.norm(normals, dim=-1).unsqueeze(-1),
    )
    return normals


def calc_reflection_normals(
        in_reflections: torch.Tensor,
        out_reflections: torch.Tensor,
) -> torch.Tensor:
    in_reflections = \
        in_reflections / th.linalg.norm(in_reflections, dim=-1).unsqueeze(-1)
    out_reflections = \
        out_reflections / th.linalg.norm(out_reflections, dim=-1).unsqueeze(-1)
    return calc_reflection_normals_(in_reflections, out_reflections)


def batch_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x * y).sum(-1).unsqueeze(-1)


def save_target(
        heliostat_origin_center: torch.Tensor,
        heliostat_face_normal: torch.Tensor,
        heliostat_points: torch.Tensor,
        heliostat_normals: torch.Tensor,
        heliostat_up_dir: Optional[torch.Tensor],

        receiver_origin_center: torch.Tensor,
        receiver_width: float,
        receiver_height: float,
        receiver_normal: torch.Tensor,
        receiver_up_dir: Optional[torch.Tensor],

        sun: torch.Tensor,
        num_rays: int,
        mean: torch.Tensor,
        cov: torch.Tensor,
        xi: Optional[torch.Tensor],
        yi: Optional[torch.Tensor],

        target_ray_directions: torch.Tensor,
        target_ray_points: torch.Tensor,
        path: str,
) -> None:
    th.save({
        'heliostat_origin_center': heliostat_origin_center,
        'heliostat_face_normal': heliostat_face_normal,
        'heliostat_points': heliostat_points,
        'heliostat_normals': heliostat_normals,
        'heliostat_up_dir': heliostat_up_dir,

        'receiver_origin_center': receiver_origin_center,
        'receiver_width': receiver_width,
        'receiver_height': receiver_height,
        'receiver_normal': receiver_normal,
        'receiver_up_dir': receiver_up_dir,

        'sun': sun,
        'num_rays': num_rays,
        'mean': mean,
        'cov': cov,
        'xi': xi,
        'yi': yi,

        'ray_directions': target_ray_directions,
        'ray_points': target_ray_points,
    }, path)


def normalize_path(path: str) -> str:
    # Normalize OS-specific paths in a non-sophisticated way.
    if '\\' in path:
        path = functools.reduce(os.path.join, path.split('\\'))
    elif '/' in path:
        path = functools.reduce(os.path.join, path.split('/'))
    return path


# def raytrace_NURBs_surface():
#     return pass
    

def give_random_targetIDs(cfg, device):
    
    targetIDs = cfg.DEEPLARTS.TRAIN.TARGET_IDs
    nsunPos = cfg.DEEPLARTS.TRAIN.NSUNPOS
    
    random_targetID_list = []
    for i in range(nsunPos):
        random_targetID_list.append(random.choice(targetIDs))
        
    return th.tensor(random_targetID_list, dtype=th.get_default_dtype(), device=device)
    

def get_heliostat_position(helName):
    
    dic = np.load('heldata\heliostat_position_dictionary.npy', allow_pickle=True).item()
    helPos = th.tensor(dic[helName])
    return helPos
    

def set_heliostat_parameter(H, helPos, zcntrl_points):
    device = zcntrl_points.device
    dtype = th.get_default_dtype()
    
    # set position of the heliostat
    H.position_on_field = helPos
    
    # calculate th cantRots as the rotation matrix between [0,0,1] and the ideally
    # canted facet spans
    decanted_normal = th.tensor([0,0,1], device=device, dtype=dtype)
    spansN, spansE = give_spans(helPos.cpu())
    
    mulN = [th.tensor([1,1,1]), th.tensor([1,1,-1]), th.tensor([1,1,1]), th.tensor([1,1,-1])]
    mulE = [th.tensor([1,1,1]), th.tensor([1,-1,1]), th.tensor([1,-1,-1]), th.tensor([1,1,-1])]
        
    cantRots = []
    for f, facet in enumerate(H.facets):
        facet_vec_x = th.mul(mulN[f], th.tensor(spansN)).to(dtype)
        facet_vec_y = th.mul(mulE[f], th.tensor(spansE)).to(dtype)
        facet_vec_z = th.cross(facet_vec_x, facet_vec_y).to(device)
        
        cantRot = get_rot_matrix(decanted_normal, facet_vec_z)
        cantRots.append(cantRot.unsqueeze(0))
    
    cantRots = th.cat(cantRots, dim=0)
    H.facets.cant_rots = cantRots
    
    for i, facet in enumerate(H.facets):
        xy_grid = facet.ctrl_points[:,:,:2]
        zcntrlFacet = zcntrl_points[i,:,:]
        zcntrlFacet = zcntrlFacet.unsqueeze(-1)
        ctrlPoints_fit = th.cat((xy_grid, zcntrlFacet), dim=-1)
        facet.set_ctrl_points(ctrlPoints_fit)
        
    return None

# def fix_pytorch3d() -> None:
#     # Monkey patch missing dtype propagation using default dtype.
#     # Not a good solution but it handles the bug.
#     tfs.Transform3d.__init__.__defaults__ = (
#         (th.get_default_dtype(),)
#         + tfs.Transform3d.__init__.__defaults__[1:]
#     )
#     tfs.Translate.__init__.__defaults__ = (
#         tfs.Translate.__init__.__defaults__[:2]
#         + (th.get_default_dtype(),)
#         + tfs.Translate.__init__.__defaults__[3:]
#     )
#     tfs.Scale.__init__.__defaults__ = (
#         tfs.Scale.__init__.__defaults__[:2]
#         + (th.get_default_dtype(),)
#         + tfs.Scale.__init__.__defaults__[3:]
#     )
#     tfs.Rotate.__init__.__defaults__ = (
#         (th.get_default_dtype(),)
#         + tfs.Rotate.__init__.__defaults__[1:]
#     )
#     tfs.RotateAxisAngle.__init__.__defaults__ = (
#         tfs.Scale.__init__.__defaults__[:2]
#         + (th.get_default_dtype(),)
#         + tfs.Rotate.__init__.__defaults__[3:]
#     )

def print_model_summary(deepLarts):
    name_deepLarts = deepLarts.name_deepLarts
    print(f'---------- You have loaded deepLarts {name_deepLarts} ------------------')
    print()
                                   
    print(f"{deepLarts.architecture_args=}")
    if deepLarts.architecture_args["use_transformer_encoder"] == False:
        print(f"{deepLarts.conv_enc_args=}")
        
        print("Number of encoder channels: ", deepLarts.enc_chs)

    else:
        print(f"{deepLarts.trans_fuse_enc_args=}")
        print(f"{deepLarts.trans_flux_enc_args=}")
    print(f"{deepLarts.styleGAN_args=}")
    print(f"{deepLarts.training_args=}")
    print(f"{deepLarts.data_args=}")
    
    
    num_params = sum(p.numel() for p in deepLarts.parameters() if p.requires_grad)
    print('Number of trainable parameters: %d' % num_params)
    
    num_params = sum(p.numel() for p in deepLarts.parameters())
    print('Number of parameters: %d' % num_params)


def give_trg_src_split(helname):
    
    # alt wurde genutzt bis zum Anfang von DR
    if helname=='AA35':
        idx_trg = [2, 1, 3] 
        idx_src = [0, 4, 3] 
        idx_mft = [0, 1, 3]

    elif helname=='AA44':
        idx_trg = [2, 3, 11]
        idx_src = [0, 1, 4, 7, 10, 12, -2]
        idx_mft = [0, 1, 2] 
        
    elif helname=='AB38':
        idx_src = [-1, -2, -4, -5]
        idx_trg = [0, 2, -3] 
        idx_mft = [9, 10, 11]
            
    elif helname=='AB44':
        idx_src = [3, 4, 5, -1]
        idx_trg = [2, -2, -3] 
        idx_mft = [5, 6, 10]
            
    elif helname=='AC30':
        idx_src = [0, -4]
        idx_trg = [1, 2, -3]
        idx_mft = [1, 4, 5, -1]
        
        
    elif helname=='AC35':
        idx_trg = [0, 1]
        idx_src = [2, 3, 4, 5, 6, 7]
        idx_mft = [1, 7, -1]

    elif helname=='AC38':
        idx_trg = [5, 6]
        idx_src = [0, 1, 2, 3, 4]
        idx_mft = [6, 7, 8]

    elif helname=='AC42':
        idx_trg = [16, 17, 19, 20]
        idx_src = [5, 7, 15, -1, -2, -3, -4, -5]
        idx_mft = [6, 15, 21]
    
    elif helname=='AD29':
        idx_src = [0, 4]
        idx_trg = [3, 5, 6, 8]
        idx_mft = [7, -2, -3]

    elif helname=='AD33':
        idx_src = [0, 1, 2, 3]
        idx_trg = [4, 5, 6]
        idx_mft = [2, 4, -1]
        
    elif helname=='AD36':
        idx_trg = [0, 1, 2, 3]
        idx_src = [5, 6, 7, 8]
        idx_mft = [2, 3, 4]

    elif helname=='AD42':
        idx_trg = [6, 7, 1]
        idx_src = [-2, -3, 2, 3]
        idx_mft = [0, 1, -1]
    
    elif helname=='AD44':
        idx_trg = [1, 2, 3, 4]
        idx_src = [-1, -2, -3, 4]
        idx_mft = [2, -2, -3]
        
    elif helname=='AE26':
        idx_trg = [0, 1, 2]
        idx_src = [4, 7, -1]
        idx_mft = [0, 2, 3]

    elif helname=='AE29':
        idx_trg = [0, 1, 2]
        idx_src = [-1, -2]
        idx_mft = [0, 1, 2]
    
    elif helname=='AE34':
        idx_trg = [1, 3, 4]
        idx_src = [0, 2, 5]
        idx_mft = [0, 1, 2]
    
    elif helname=='AF31':
        idx_trg = [2, 3, 4]
        idx_src = [0, -2, -1]
        idx_mft = [0, 1, 2]
    
    elif helname=='AF37':
        idx_trg = [1, 2, 3, 4, -6]
        idx_src = [7, -1, -2, 5]
        idx_mft = [0, 1, 2]
    
    elif helname=='AF42':
        idx_trg = [1, 3, 6]
        idx_src = [-2, -1, -3, 7, 15]
        idx_mft = [0, 1, 2]
    
    elif helname=='AF46':
        idx_trg = [5, 7, 8]
        idx_src = [3, 4, 6, 9, 10]
        idx_src = [2]
        idx_mft = [1, 2, 4]
    
    elif helname=='AX56':
        idx_trg = [1, 5, 7, 8]
        idx_src = [3, 4, 6, 9, 10]
        # idx_src = [12]

        idx_mft = [1, 2, 4]

    elif helname=='AY26':
        idx_trg = [5, 6, 7]
        idx_src = [1, 2, 3, -1, -2] 
        idx_mft = [1, 2, 3]

    elif helname=='AY32':
        idx_trg = [6, 10, 11]
        idx_src = [0, 1, 5, 7, 15, 16, 17] 
        idx_mft = [1, 2, 3]
    
    elif helname=='AY37':
        idx_trg = [4, 6, 8, 9]
        idx_src = [0, 1, 5, 7, 15, 16, 17] 
        idx_mft = [1, 2, 3]

    elif helname=='AY60':
        idx_trg = [5, 6, 7]
        idx_src = [0, 1, 2, 3, 4] 
        # idx_src = [0,1,2,3,4,5,6,7] 
        idx_mft = [1, 2, 3]

    elif helname=='BA27':
        idx_trg = [0, 2, -7, -8]
        idx_src = [-1, -3, -4, -5, 5, 6, 7] 
        idx_mft = [1, 2, 3, 4]
            
    elif helname=='BA29':
        idx_trg = [3, 4, -7, -8]
        idx_src = [-1, -3, -4, -5, 5, 6, 7] 
        idx_mft = [2, 3, 4]
    
    elif helname=='BB39':
        idx_trg = [2, 3, 4, 5]
        idx_src = [0, 1, -1] 
        idx_mft = [0, 1, 2, 3]
        
    elif helname=='BB41':
        idx_trg = [0, 1, 4, 5]
        idx_src = [2, 3, -2] 
        idx_mft = [0, 1, 2, 3]
        
    elif helname=='BC60':
        idx_trg = [2, 3, 4, 5]
        idx_src = [2, 3, -2, 6, 7] 
        idx_mft = [0, 1, 2, 3]
    else:
        idx
        
    return idx_mft, idx_trg, idx_src


def split_trg_src_extra_dataset(helname, test_args, datadic, device, show_plots=True, targetID_to_real=False):
        
    targetIDs = datadic['targetIDs'].to(device).clone()

    if targetID_to_real:
        targetIDs_real = transform_targetID(targetIDs, cfg=None, direction='toreal')
    else:
        targetIDs_real = targetIDs
        
    
    test_on_1_trg_flux = test_args["test_on_1_trg_flux"]
        
    mask_mft = (targetIDs_real == 3)
    
    targetImages_mft = datadic['targetImages'].to(device)[mask_mft].unsqueeze(0)
    unetImages_mft = datadic['unetImages'].to(device)[mask_mft].unsqueeze(0)
    fluxes_mft = datadic['fluxes'].to(device)[mask_mft].unsqueeze(0).squeeze(2)
    sunPos_mft = datadic['sunPos'].to(device)[mask_mft].unsqueeze(0)
    targetIDs_mft = datadic['targetIDs'].to(device)[mask_mft].unsqueeze(0)
    ideals_mft = datadic['fluxes_ideal'].to(device)[mask_mft].unsqueeze(0)
    
    unetImages = datadic['unetImages'].to(device)[~mask_mft].unsqueeze(0)
    fluxes = datadic['fluxes'].to(device)[~mask_mft].unsqueeze(0).squeeze(2)
    sunPos = datadic['sunPos'].to(device)[~mask_mft].unsqueeze(0)
    targetIDs = datadic['targetIDs'].to(device)[~mask_mft].unsqueeze(0)
    targetImages = datadic['targetImages'].to(device)[~mask_mft].unsqueeze(0)
    ideals = datadic['fluxes_ideal'].to(device)[~mask_mft].unsqueeze(0)
    
    # plot all Images on STJ
    if show_plots:

        plot_grid(targetImages[0,:,:,:].unsqueeze(1).permute(dims=(0,4,2,3,1)).squeeze(-1), batch_size=10, normalize=True, gray=False)
        plot_grid(unetImages[0,:,:,:].unsqueeze(1), batch_size=10, normalize=True, cmap="hot", title="Unet ST")
        plot_grid(fluxes[0,:,:,:].unsqueeze(1), batch_size=10, normalize=True, cmap="hot", title="Raytracing ST")
        plot_grid(ideals[0,:,:,:].unsqueeze(1), batch_size=10, normalize=True, cmap="hot", title="Ideals ST")
        
        n_mft_images = targetImages_mft.size(1)
        
        if n_mft_images > 0:
            plot_grid(targetImages_mft[0,:,:,:].unsqueeze(1).permute(dims=(0,4,2,3,1)).squeeze(-1), batch_size=10, normalize=True, title='Target MFT', gray=False)
            plot_grid(unetImages_mft[0,:,:,:].unsqueeze(1), batch_size=100, normalize=True, cmap="hot", title='Unet MFT')
            plot_grid(fluxes_mft[0,:,:,:].unsqueeze(1), batch_size=100, normalize=True, cmap="hot", title='Raytracing MFT')
        
    idx_mft, idx_trg, idx_src = give_trg_src_split(helname)
    
    if test_on_1_trg_flux:
        idx_trg = idx_trg[0]
        
    targetImages_target = targetImages[:, idx_trg, :, :, :]
    unetImages_target = unetImages[:, idx_trg, :, :]
    flux_target = fluxes[:, idx_trg, :, :]
    sunPos_target = sunPos[:, idx_trg, :]
    targetID_target = targetIDs[:, idx_trg]
    ideals_target = ideals[:, idx_trg, :, :]
    
    if test_on_1_trg_flux:
        targetImages_target = targetImages_target.unsqueeze(1)
        unetImages_target = unetImages_target.unsqueeze(1)
        flux_target = flux_target.unsqueeze(1)
        sunPos_target = sunPos_target.unsqueeze(1)
        targetID_target = targetID_target.unsqueeze(1)
        ideals_target = ideals_target.unsqueeze(1)
        
    targetImages_source = targetImages[:, idx_src, :, :]
    unetImages_source = unetImages[:, idx_src, :, :]
    flux_source = fluxes[:, idx_src, :, :]
    sunPos_source = sunPos[:, idx_src, :]
    targetID_source = targetIDs[:, idx_src]
    ideals_source = ideals[:, idx_src, :, :]

    data_mft = {}
    if targetImages_mft.size(1) > 0:
        
        targetImages_mft = targetImages_mft[:, idx_mft, :, :]
        unetImages_mft = unetImages_mft[:, idx_mft, :, :]
        fluxes_mft = fluxes_mft[:, idx_mft, :, :]
        sunPos_mft = sunPos_mft[:, idx_mft, :]
        targetIDs_mft = targetIDs_mft[:, idx_mft]
    
    data_mft = {"targetImages":targetImages_mft, 
               "unetImages":unetImages_mft,
               "fluxes":fluxes_mft,
               "sunPos":sunPos_mft,
               "targetIDs":targetIDs_mft,
               "ideals":ideals_mft}
    
    data_trg = {"targetImages":targetImages_target, 
               "unetImages":unetImages_target,
               "fluxes":flux_target,
               "sunPos":sunPos_target,
               "targetIDs":targetID_target,
               "ideals":ideals_target}
    
    data_src = {"targetImages":targetImages_source, 
               "unetImages":unetImages_source,
               "fluxes":flux_source,
               "sunPos":sunPos_source,
               "targetIDs":targetID_source,
               "ideals":ideals_source}

    
    return data_mft, data_trg, data_src


def split_in_n_random_images(helname, 
                             test_args, 
                             datadic, 
                             device,
                             cfg,
                             show_plots=False, 
                             targetID_to_real=False):
        
    targetIDs = datadic['targetIDs'].clone()

    if targetID_to_real:
        targetIDs_real = transform_targetID(targetIDs, cfg=None, direction='toreal')
    else:
        targetIDs_real = targetIDs
    
    # helID = give_helID(helname)
    
    # sort out target 3/4 images
    mask_mft = (targetIDs_real == 3)
    
    unetImages = datadic['unetImages'][~mask_mft].unsqueeze(0)
    fluxes = datadic['fluxes'][~mask_mft].unsqueeze(0).squeeze(2)
    sunPos = datadic['sunPos'][~mask_mft].unsqueeze(0)
    targetIDs = datadic['targetIDs'][~mask_mft].unsqueeze(0)
    targetImages = datadic['targetImages'][~mask_mft].unsqueeze(0)
    fluxesIdeal = datadic['fluxes_ideal'][~mask_mft].unsqueeze(0)

    # helID = th.tensor([datadic['helID']]).unsqueeze(0)

    
    # fluxes, sunPos, _, _,targetIDs =  process_data_for_model(cfg, 
    #                                                     direction='totrain', 
    #                                                     flux=fluxes, 
    #                                                     sunPos=sunPos, 
    #                                                     targetID=targetIDs,
    #                                                     apply_low_pass_filter=False)

    # unetImages, _, _, _, _= process_data_for_model(cfg, 
    #                                             direction='totrain', 
    #                                             flux=unetImages, 
    #                                             apply_low_pass_filter=False)
    
    # fluxesIdeal, _, _, _, _= process_data_for_model(cfg, 
    #                                                 direction='totrain', 
    #                                                 flux=fluxesIdeal, 
    #                                                 apply_low_pass_filter=False)
    
    
    accuracy = give_accuracy(unetImages.swapaxes(0,1).cpu(), 
                             fluxes.swapaxes(0,1).cpu(), 
                             batch_mode=True)
    
    idx_sort = th.randperm(accuracy.size(0))
    targetImages = targetImages[:,idx_sort,:,:,:]
    unetImages = unetImages[:,idx_sort,:,:]
    fluxes = fluxes[:,idx_sort,:,:]
    fluxesIdeal = fluxesIdeal[:,idx_sort,:,:]
    sunPos = sunPos[:,idx_sort,:]
    targetIDs = targetIDs[:,idx_sort]
    accuracy = accuracy[idx_sort]
    
    mask_accuracies = accuracy > cfg.DEEPLARTS.VALID.ACCURACY_LIMIT
    targetImages = targetImages[:,mask_accuracies,:,:,:]
    unetImages = unetImages[:,mask_accuracies,:,:]
    fluxes = fluxes[:,mask_accuracies,:,:]
    fluxesIdeal = fluxesIdeal[:,mask_accuracies,:,:]
    sunPos = sunPos[:,mask_accuracies,:]
    targetIDs = targetIDs[:,mask_accuracies]
    
    n_accuracies = unetImages.size(1)
    if n_accuracies > cfg.DEEPLARTS.VALID.N_INPUT_TARGETIMAGES:
        unetImages_source = unetImages[:,:cfg.DEEPLARTS.VALID.N_INPUT_TARGETIMAGES,:,:].clone().to(device)
        targetImages_source = targetImages[:,:cfg.DEEPLARTS.VALID.N_INPUT_TARGETIMAGES,:,:,:].clone().to(device)
        fluxes_source = fluxes[:,:cfg.DEEPLARTS.VALID.N_INPUT_TARGETIMAGES,:,:].clone().to(device)
        fluxesIdeal_source = fluxesIdeal[:,:cfg.DEEPLARTS.VALID.N_INPUT_TARGETIMAGES,:,:].clone().to(device)
        sunPos_source = sunPos[:,:cfg.DEEPLARTS.VALID.N_INPUT_TARGETIMAGES,:].clone().to(device)
        targetIDs_source = targetIDs[:,:cfg.DEEPLARTS.VALID.N_INPUT_TARGETIMAGES].clone().to(device)
    else:
        targetImages_source = targetImages.clone().to(device)
        unetImages_source = unetImages.clone().to(device)
        fluxes_source = fluxes.clone().to(device)
        fluxesIdeal_source = fluxesIdeal.clone().to(device)
        sunPos_source = sunPos.clone().to(device)
        targetIDs_source = targetIDs.clone().to(device)
    
    
    if n_accuracies > cfg.DEEPLARTS.VALID.N_TARGET_TARGETIMAGES:
        
        n_target_images = -1*cfg.DEEPLARTS.VALID.N_TARGET_TARGETIMAGES
        
        targetImages_target = targetImages[:,n_target_images:,:,:,:].clone().to(device)
        unetImages_target = unetImages[:,n_target_images:,:,:].clone().to(device)
        fluxes_target = fluxes[:,n_target_images:,:,:].clone().to(device)
        fluxesIdeal_target = fluxesIdeal[:,n_target_images:,:,:].clone().to(device)
        sunPos_target = sunPos[:,n_target_images:,:].clone().to(device)
        targetIDs_target = targetIDs[:,n_target_images:].clone().to(device)
    else:
        targetImages_target = targetImages.clone().to(device)
        unetImages_target = unetImages.clone().to(device)
        fluxes_target = fluxes.clone().to(device)
        fluxesIdeal_target = fluxesIdeal.clone().to(device)
        sunPos_target = sunPos.clone().to(device)
        targetIDs_target = targetIDs.clone()   .to(device)     
        

    targetImages_mft = datadic['targetImages'].to(device)[mask_mft].unsqueeze(0)
    unetImages_mft = datadic['unetImages'].to(device)[mask_mft].unsqueeze(0)
    fluxes_mft = datadic['fluxes'].to(device)[mask_mft].unsqueeze(0).squeeze(2)
    sunPos_mft = datadic['sunPos'].to(device)[mask_mft].unsqueeze(0)
    targetIDs_mft = datadic['targetIDs'].to(device)[mask_mft].unsqueeze(0)
    ideals_mft = datadic['fluxes_ideal'].to(device)[mask_mft].unsqueeze(0)
    data_mft = None
    
    if fluxes_mft.size(1) > 0:
        
        idx_sort = th.randperm(fluxes_mft.size(1))
    
        unetImages_mft = unetImages_mft[:,idx_sort,:,:]
        fluxes_mft = fluxes_mft[:,idx_sort,:,:]
        ideals_mft = ideals_mft[:,idx_sort,:,:]
        sunPos_mft = sunPos_mft[:,idx_sort,:]
        targetIDs_mft = targetIDs_mft[:,idx_sort]
        targetImages_mft = targetImages_mft[:,idx_sort,:,:,:]
        
        targetImages_mft = targetImages_mft[:,(-1*cfg.DEEPLARTS.VALID.N_TARGET_TARGETIMAGES):,:,:].clone()
        unetImages_mft = unetImages_mft[:,(-1*cfg.DEEPLARTS.VALID.N_TARGET_TARGETIMAGES):,:,:].clone()
        fluxes_mft = fluxes_mft[:,-1*cfg.DEEPLARTS.VALID.N_TARGET_TARGETIMAGES:,:,:].clone()
        fluxesIdeal_mft = ideals_mft[:,-1*cfg.DEEPLARTS.VALID.N_TARGET_TARGETIMAGES:,:,:].clone()
        sunPos_mft = sunPos_mft[:,-1*cfg.DEEPLARTS.VALID.N_TARGET_TARGETIMAGES:,:].clone()
        targetIDs_mft = targetIDs_mft[:,-1*cfg.DEEPLARTS.VALID.N_TARGET_TARGETIMAGES:].clone()
        
        data_mft = {"targetImages":targetImages_mft, 
                   "unetImages":unetImages_mft,
                   "fluxes":fluxes_mft,
                   "sunPos":sunPos_mft,
                   "targetIDs":targetIDs_mft,
                   "ideals":fluxesIdeal_mft}


    
    data_trg = {"targetImages":targetImages_target, 
               "unetImages":unetImages_target,
               "fluxes":fluxes_target,
               "sunPos":sunPos_target,
               "targetIDs":targetIDs_target,
               "ideals":fluxesIdeal_target}
    
    data_src = {"targetImages":targetImages_source, 
               "unetImages":unetImages_source,
               "fluxes":fluxes_source,
               "sunPos":sunPos_source,
               "targetIDs":targetIDs_source,
               "ideals":fluxesIdeal_source}

    
    return data_mft, data_trg, data_src

def give_aimPoints_extrapolation(cfg, stepsize, device):
    target_7_center = cfg.AC.TARGET.TARGET7.CENTER 
    rec_center = cfg.AC.RECEIVER.CENTER 
    
    aimPoints = []
    
    for i in np.arange(target_7_center[2] - 5, rec_center[2], stepsize):
        aimPoint = [target_7_center[0], target_7_center[1], i]
        aimPoints.append(aimPoint)
    
    return th.tensor(aimPoints, device=device, dtype=th.float32)
    
    
def is_valid_img(helname, targetImage_index):
    
    validdir = r"C:\Users\lewe_jn\Desktop\gancstr\rawdata\targetImages\validation_images"
    validfolder = os.listdir(validdir)
    
    filename = f"{helname}_{targetImage_index}.png"
    
    is_valid_img = filename in validfolder
    
    return is_valid_img
    
    
    
    
def plot_grid(batch, batch_size, title='', savename='', normalize=False, 
              save=False, cmap='jet', gray=True):
    
    # img_size = arg_parse().image_size
    
    nrow = batch_size
    fig, ax = plt.subplots(1,1)
    ax.axis("off")
    ax.set_title(title)
    grid = vutils.make_grid(batch[:batch_size**2], nrow=nrow, padding=1, normalize=normalize)
    grid = np.transpose(grid.cpu(),(1,2,0))
    
    if gray == True:
        grid = grid[:,:,0]

    ax.imshow(grid, cmap=cmap)
        
    plt.show()
    if save:
        fig.savefig(savename)
    plt.close(fig)


def give_targetPos_dic(cfg):

    cfg_target = cfg.AC.TARGET

    target_center_7 = cfg_target.TARGET7.CENTER
    plane_x_7 = cfg_target.TARGET7.PLANE_X
    plane_y_7 = cfg_target.TARGET7.PLANE_Y
        
    target_center_6 = cfg_target.TARGET6.CENTER
    plane_x_6 = cfg_target.TARGET6.PLANE_X
    plane_y_6 = cfg_target.TARGET6.PLANE_Y
    
    target_center_3 = cfg_target.TARGET3.CENTER
    plane_x_3 = cfg_target.TARGET3.PLANE_X
    plane_y_3 = cfg_target.TARGET3.PLANE_Y
    
    dic_6 = {"target_center":target_center_6, "plane_x":plane_x_6, "plane_y":plane_y_6}
    dic_7 = {"target_center":target_center_7, "plane_x":plane_x_7, "plane_y":plane_y_7}
    dic_3 = {"target_center":target_center_3, "plane_x":plane_x_3, "plane_y":plane_y_3}
    targetPos_Dic = {"6":dic_6, "7":dic_7, "3":dic_3}
    
    return targetPos_Dic


def center_of_mass(intensity_tensors, threshold_ratio=0.2):
    """
    Calculate the center of mass of a batch of intensity tensors.

    Args:
        intensity_tensors (torch.Tensor): Batch of intensity tensors with dimensions [batch_size, H, W].
        threshold_ratio (float): Percentage of the maximum intensity value to use as a threshold.
                                 Only values above this threshold will be included in the calculation.

    Returns:
        torch.Tensor: Center of mass coordinates for each intensity tensor, with shape [batch_size, 2].
                      The first dimension contains the y-coordinate (row index) and the second dimension contains
                      the x-coordinate (column index).
    """
    if intensity_tensors.dim() == 4:
        if intensity_tensors.size(1) == 1:
            intensity_tensors = intensity_tensors.squeeze(1)
        else:
            raise Exception("This function cannot handle multi-channel tensors.")

    device = intensity_tensors.device
    # Create grids for the row and column indices
    grid_y, grid_x = torch.meshgrid(
        torch.arange(intensity_tensors.shape[1], device=device),
        torch.arange(intensity_tensors.shape[2], device=device)
    )

    # Calculate the maximum intensity for each tensor
    max_intensity = th.max(intensity_tensors, dim=0, keepdim=True).values

    # Apply the threshold
    threshold = max_intensity * threshold_ratio
    mask = intensity_tensors >= threshold

    # Apply the mask to the intensity tensors
    masked_intensities = intensity_tensors * mask

    # Calculate the total intensity (sum of all intensities in each tensor)
    total_intensity = masked_intensities.sum(dim=(1, 2))

    # Prevent division by zero by setting total_intensity to 1 where it is 0
    total_intensity[total_intensity == 0] = 1

    # Calculate the center of mass for each tensor
    center_y = (masked_intensities * grid_y).sum(dim=(1, 2)) / total_intensity + 0.5
    center_x = (masked_intensities * grid_x).sum(dim=(1, 2)) / total_intensity + 0.5

    # Stack the center coordinates along the last dimension
    center_coordinates = torch.stack((center_y, center_x), dim=1).to(dtype=torch.int)

    return center_coordinates


# def center_of_mass(intensity_tensors):
#     """
#     Calculate the center of mass of a batch of intensity tensors.

#     Args:
#         intensity_tensors (torch.Tensor): Batch of intensity tensors with dimensions [batch_size, H, W].

#     Returns:
#         torch.Tensor: Center of mass coordinates for each intensity tensor, with shape [batch_size, 2].
#                       The first dimension contains the y-coordinate (row index) and the second dimension contains
#                       the x-coordinate (column index).
#     """
    
#     device = intensity_tensors.device
#     # Create grids for the row and column indices
#     grid_y, grid_x = th.meshgrid(th.arange(intensity_tensors.shape[1], device=device), th.arange(intensity_tensors.shape[2], device=device))
    
#     # Calculate the total intensity (sum of all intensities in each tensor)
#     total_intensity = intensity_tensors.sum(dim=(1, 2))
    
#     # Calculate the center of mass for each tensor
#     center_y = (intensity_tensors * grid_y).sum(dim=(1, 2)) / total_intensity + 0.5
#     center_x = (intensity_tensors * grid_x).sum(dim=(1, 2)) / total_intensity + 0.5
    
#     # Stack the center coordinates along the last dimension
#     center_coordinates = th.stack((center_y, center_x), dim=1).to(dtype=th.int)
    
#     return center_coordinates


# def translate_mass_center(x, mass_centers):
#     fluxsize = x.size(-1)
    
#     translation_x = mass_centers[:,0] - fluxsize/2
#     translation_y = mass_centers[:,1] - fluxsize/2
#     translation_x = translation_x.unsqueeze(-1).unsqueeze(-1).long()
#     translation_y = translation_y.unsqueeze(-1).unsqueeze(-1).long()

#     grid_batch, grid_x, grid_y = torch.meshgrid(
#         torch.arange(x.size(0), dtype=torch.long, device=x.device),
#         torch.arange(x.size(2), dtype=torch.long, device=x.device),
#         torch.arange(x.size(3), dtype=torch.long, device=x.device),
#     )
#     grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
#     grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
#     x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
#     x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    
#     return x

    
# def cropp_img_batch_around_mass_centers(cfg, targetID, flux, targetPos_Dic, targetEvaluationPlaneSize, planex, planey):
    
#     device = flux.device
#     print(flux.size())
#     B, C, H, W = flux.size()
#     # make flux channels to batch
#     if not C==1:
#         flux = flux.view(B*C,H,W)
        
#     # calculates center of mass
#     mass_centers = center_of_mass(flux)
#     flux = flux.unsqueeze(1)
    
#     flux_size = cfg.AC.RECEIVER.RESOLUTION_X
#     assert cfg.AC.RECEIVER.RESOLUTION_X == cfg.AC.RECEIVER.RESOLUTION_Y, "Only same resolution in x and y!"
    
#     crop_width = int(targetEvaluationPlaneSize/planex * flux_size + 0.5)
#     # crop_width = crop_width.to(th.int)
#     crop_height = int(targetEvaluationPlaneSize/planey * flux_size + 0.5)
#     # crop_height = crop_height.to(th.int)

#     # fluxsize = 128
#     # top_left_coordinates = th.tensor(mass_centers, dtype=th.int).unsqueeze(0) - th.tensor( [crop_height/2, crop_width/2], dtype=th.int)
#     # top_left_coordinates = mass_centers - th.tensor( (crop_height, crop_width), device=device)/2
#     # top_left_coordinates = top_left_coordinates.to(th.int)
#     # print(top_left_coordinates.size())
#     # flux = crop_images(flux, crop_height=crop_height, crop_width=crop_width, top_left_coordinates=top_left_coordinates)
    
#     flux = translate_mass_center(flux, mass_centers)
    
#     flux = crop_images(images_tensor=flux, 
#                        crop_height=crop_height, 
#                        crop_width=crop_width)
    
#     # flux = flux.unsqueeze(0).unsqueeze(0)
#     target_image_size = cfg.DEEPLARTS.TRAIN.TARGET_IMAGE_SIZE
#     flux = Resize(size= (target_image_size, target_image_size))(flux)
#     print(flux.size())
#     flux = flux.view(B,C,target_image_size,target_image_size).squeeze(0)
#     print(flux.size())

#     return flux


# def crop_images(images_tensor, crop_height, crop_width):
#     """
#     Crop a tensor of images at different pixels.

#     Args:
#         images_tensor (torch.Tensor): Tensor of images with dimensions [batch_size, channels, height, width].
#         crop_height (int): Height of the cropped region.
#         crop_width (int): Width of the cropped region.
#         top_left_coordinates (torch.Tensor): Tensor containing top-left coordinates of the cropped regions
#                                               for each image. Shape should be [batch_size, 2], where the first
#                                               dimension contains the y-coordinate (row index) and the second dimension
#                                               contains the x-coordinate (column index).

#     Returns:
#         torch.Tensor: Cropped images tensor with dimensions [batch_size, channels, crop_height, crop_width].
#     """
    
#     # Calculate the bottom-right coordinates for cropping
#     left_bound = int(images_tensor.size(-1)/2 - crop_width/2)
#     right_bound = int(images_tensor.size(-1)/2 + crop_width/2)
#     lower_bound = int(images_tensor.size(-1)/2 - crop_height/2)
#     upper_bound = int(images_tensor.size(-1)/2 + crop_height/2)

#     # Use advanced indexing to crop the images
#     cropped_images = images_tensor[:,:,left_bound:right_bound,lower_bound:upper_bound]
    
#     return cropped_images
    
    
    

# def crop_images(image_tensor, pixel_tensor, crop_height, crop_width):
#     """
#     Crop images in the image tensor at specified pixels in the pixel tensor.

#     Args:
#     - image_tensor (torch.Tensor): Input tensor of images with dimensions (c, w, h).
#     - pixel_tensor (torch.Tensor): Tensor of pixel coordinates with dimensions (c, 2).
#     - crop_height (int): Height of the crop around each pixel.
#     - crop_width (int): Width of the crop around each pixel.

#     Returns:
#     - cropped_images (torch.Tensor): Tensor of cropped images with dimensions (c, crop_height, crop_width).
#     """

#     c, w, h = image_tensor.size()
    
#     print(image_tensor.size())
#     print(pixel_tensor)
#     print(crop_height)
#     print(crop_width)
#     # Calculate the half crop size for height and width
#     half_crop_height = crop_height // 2
#     half_crop_width = crop_width // 2

#     # Calculate the left, right, top, and bottom bounds for cropping
#     left_bounds = torch.clamp(pixel_tensor[:, 0] - half_crop_width, min=0)
#     top_bounds = torch.clamp(pixel_tensor[:, 1] - half_crop_height, min=0)

#     # left_bounds = pixel_tensor[:, 0] - half_crop_width
#     # top_bounds = pixel_tensor[:, 1] - half_crop_height
    
#     # Create masks for indexing
#     mask_channels = torch.arange(c).unsqueeze(1)
#     mask_rows = torch.arange(-half_crop_height, half_crop_height + 1).unsqueeze(1)
#     mask_cols = torch.arange(-half_crop_width, half_crop_width + 1).unsqueeze(0)

#     # Apply masks for indexing
#     cropped_images = image_tensor[mask_channels[:, None, None], 
#                                   left_bounds[:, None] + mask_rows[None, :, :],
#                                   top_bounds[:, None] + mask_cols[None, :, :]]

#     # # Pad cropped images if necessary
#     # pad_left = crop_width - cropped_images.size(2)
#     # pad_top = crop_height - cropped_images.size(3)
#     # cropped_images = torch.nn.functional.pad(cropped_images, (0, pad_top, 0, pad_left))

#     return cropped_images
    



# def center_of_mass(intensity_tensors):
#     """
#     Calculate the center of mass of a batch of intensity tensors.

#     Args:
#         intensity_tensors (torch.Tensor): Batch of intensity tensors with dimensions [batch_size, H, W].

#     Returns:
#         torch.Tensor: Center of mass coordinates for each intensity tensor, with shape [batch_size, 2].
#                       The first dimension contains the y-coordinate (row index) and the second dimension contains
#                       the x-coordinate (column index).
#     """

#     if intensity_tensors.dim() == 4:
#         if intensity_tensors.size(1) == 1:
#             intensity_tensors = intensity_tensors.squeeze(1)
#         else:
#             raise Exception("This function can not get a channel tensor.")

#     device = intensity_tensors.device
#     # Create grids for the row and column indices
#     grid_y, grid_x = th.meshgrid(th.arange(intensity_tensors.shape[1], device=device), th.arange(intensity_tensors.shape[2], device=device))
    
#     # Calculate the total intensity (sum of all intensities in each tensor)
#     total_intensity = intensity_tensors.sum(dim=(1, 2))
    
#     # Calculate the center of mass for each tensor
#     center_y = (intensity_tensors * grid_y).sum(dim=(1, 2)) / total_intensity + 0.5
#     center_x = (intensity_tensors * grid_x).sum(dim=(1, 2)) / total_intensity + 0.5
    
#     # Stack the center coordinates along the last dimension
#     center_coordinates = th.stack((center_y, center_x), dim=1).to(dtype=th.int)
    
#     return center_coordinates


def translate_mass_center(x, mass_centers):
    fluxsize = x.size(-1)
    
    translation_x = mass_centers[:,0] - fluxsize/2
    translation_y = mass_centers[:,1] - fluxsize/2
    translation_x = translation_x.unsqueeze(-1).unsqueeze(-1).long()
    translation_y = translation_y.unsqueeze(-1).unsqueeze(-1).long()

    grid_batch, grid_x, grid_y = th.meshgrid(
        th.arange(x.size(0), dtype=th.long, device=x.device),
        th.arange(x.size(2), dtype=th.long, device=x.device),
        th.arange(x.size(3), dtype=th.long, device=x.device),
    )
    grid_x = th.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = th.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    # x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    x = x_pad.permute(0, 2, 3, 1)[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    
    return x

    
def cropp_img_batch_around_mass_centers(cfg, flux, targetEvaluationPlaneSize, planex, planey):
    
    B, C, H, W = flux.size()
    # make flux channels to batch
    if not C==1:
        flux = flux.view(B*C,1,H,W)
        
    # calculates center of mass
    mass_centers = center_of_mass(flux)
    # flux = flux.unsqueeze(1)

    flux_size = flux.size(-1)
    assert flux.size(-1) == flux.size(-2), "Only same resolution in x and y!"
    
    crop_width = int(targetEvaluationPlaneSize/planey * flux_size + 0.5)
    crop_height = int(targetEvaluationPlaneSize/planex * flux_size + 0.5)
    
    # crop_height = int(targetEvaluationPlaneSize/planey * flux_size + 0.5)
    # crop_width = int(targetEvaluationPlaneSize/planex * flux_size + 0.5)
    
    flux = translate_mass_center(flux, mass_centers)
    
    flux = crop_images(images_tensor=flux, 
                       crop_height=crop_height, 
                       crop_width=crop_width)
    
    target_image_size = cfg.DEEPLARTS.TRAIN.TARGET_IMAGE_SIZE
    flux = Resize(size= (target_image_size, target_image_size))(flux)
    flux = flux.view(B,C,target_image_size,target_image_size).squeeze(0)

    return flux


def crop_images(images_tensor, crop_height, crop_width):
    """
    Crop a tensor of images at different pixels.

    Args:
        images_tensor (torch.Tensor): Tensor of images with dimensions [batch_size, channels, height, width].
        crop_height (int): Height of the cropped region.
        crop_width (int): Width of the cropped region.
        top_left_coordinates (torch.Tensor): Tensor containing top-left coordinates of the cropped regions
                                              for each image. Shape should be [batch_size, 2], where the first
                                              dimension contains the y-coordinate (row index) and the second dimension
                                              contains the x-coordinate (column index).

    Returns:
        torch.Tensor: Cropped images tensor with dimensions [batch_size, channels, crop_height, crop_width].
    """
    
    # Calculate the bottom-right coordinates for cropping
    left_bound = int(images_tensor.size(-1)/2 - crop_width/2)
    right_bound = int(images_tensor.size(-1)/2 + crop_width/2)
    lower_bound = int(images_tensor.size(-1)/2 - crop_height/2)
    upper_bound = int(images_tensor.size(-1)/2 + crop_height/2)

    # Use advanced indexing to crop the images
    cropped_images = images_tensor[:,:,left_bound:right_bound,lower_bound:upper_bound]
    
    return cropped_images
    
    
    
def initialize_unet(cfg, device):
    
    statedicdir=r'unet/unet_raytracing.pt'
    statedic = th.load(statedicdir)
    Unet = UNet_3Plus(in_channels=1, n_classes=2, feature_scale=4, start_ch=32)
    Unet.load_state_dict(statedic)
    Unet.eval()
    
    for p in Unet.parameters():
        p.requires_grad = False
    
    Unet = Unet.to(device)
    
    return Unet
    
    
def add_random_stripes(x, amp_start=0.5, amp_end=1.3, cutout_size=(8,200)):
    
    if x.dim()==4:
        b,c,h,w = x.size()
        x = x.view(b*c,1,h,w)
        
    # cutout_size = int(x.size(2) * torch.rand(1).item()*ratio_height + 0.5), int(x.size(3) *  torch.rand(1).item()*ratio_width + 0.5)
    # cutout_size = 8, 200
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    amplitudes = torch.rand(x.size(0), device=x.device)*(amp_end-amp_start) + amp_start
    amplitudes = amplitudes.unsqueeze(-1).unsqueeze(-1)
    mask[grid_batch, grid_x, grid_y] = amplitudes*mask[grid_batch, grid_x, grid_y]
    x = x * mask.unsqueeze(1)
    
    x = x.view(b,c,h,w)
    del grid_batch, grid_x, grid_y

    return x


def rand_translation(x, ratio=0.125):
    
    if x.dim()==4:
        b,c,h,w = x.size()
        x = x.view(b*c,1,h,w)
        
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    x = x.view(b,c,h,w)
    del grid_batch, grid_x, grid_y
    return x


@th.no_grad()
def add_flux_in_empty_target(deepLarts, Unet, fluxbatch, empty_target_images, cfg, rank):
    device = fluxbatch.device
    
    b, c, w, h = fluxbatch.size()
    savename=cfg.DIRECTORIES.JUWELS.CODEDIR + 'before_' + str(rank)
    # plot_grid(fluxbatch, batch_size = 64, gray=True, cmap='hot', normalize=True, save=True, savename=savename)
    fluxbatch = fluxbatch.view(b*c,1,w,h)

    # add vertical stripes, which are a common artifact in real target images
    if deepLarts.training_args["add_vertical_stripes"]==True:
        amp_start =  deepLarts.training_args["amp_start"]
        amp_end =  deepLarts.training_args["amp_end"]
        height_range = deepLarts.training_args["height_range"]
        width_range = deepLarts.training_args["width_range"]
                              
        
        cutout_size = (random.randint(height_range[0], height_range[1]), 
                       random.randint(width_range[0], width_range[1]))
        fluxbatch = add_random_stripes(fluxbatch, 
                                             amp_start=amp_start, 
                                             amp_end=amp_end,
                                             cutout_size=cutout_size)

        cutout_size = (random.randint(height_range[0], height_range[1]), 
                       random.randint(width_range[0], width_range[1]))
        fluxbatch = add_random_stripes(fluxbatch, 
                                             amp_start=amp_start, 
                                             amp_end=amp_end,
                                             cutout_size=cutout_size)

        cutout_size = (random.randint(height_range[0], height_range[1]), 
                       random.randint(width_range[0], width_range[1]))
        fluxbatch = add_random_stripes(fluxbatch, 
                                             amp_start=amp_start, 
                                             amp_end=amp_end,
                                             cutout_size=cutout_size)        
        
        
    if deepLarts.training_args["apply_overexposure"]==True:
        overeposure_max = th.rand(fluxbatch.size(0), device=device)*deepLarts.training_args["overexposure_clamp"]
        overeposure_max = overeposure_max.unsqueeze(-1).unsqueeze(-1).repeat((1,w,h)).unsqueeze(1)
        fluxbatch = th.clamp(fluxbatch, th.zeros_like(overeposure_max, device=device), overeposure_max)
        del overeposure_max
        
    # fluxbatch = utils.rand_translation(fluxbatch, ratio=0.2)
    
    # noise = th.rand_like(empty_target_images)*th.max(empty_target_images)*0
    
    mul = th.rand(b*c, device=device)*4 + 6
    mul = mul.view(-1, 1, 1, 1)

    idc = th.randperm(empty_target_images.size(0), device=device)

    fluxbatch = empty_target_images[idc,:,:][:b*c,:,:].unsqueeze(1) + mul*fluxbatch

    # plt.imshow(input_[0,0,:,:])
    # plt.title(mul)
    # plt.show()
    
    fluxbatch = Unet(fluxbatch)[:,0,:,:].unsqueeze(1)
    fluxbatch[fluxbatch<0]=0
    
    planex = cfg.AC.TARGET.TARGET6.PLANE_X 
    planey = cfg.AC.TARGET.TARGET6.PLANE_Y
    targetEvaluationPlaneSize = cfg.AC.TARGET.EVALUATION_SIZE
    fluxbatch = cropp_img_batch_around_mass_centers(cfg, fluxbatch, targetEvaluationPlaneSize, planex, planey)
    # unet_flux = utils.cropp_img_batch_around_mass_centers(cfg, unet_flux, targetEvaluationPlaneSize, planex, planey)[0,:,:]
    
    savename=cfg.DIRECTORIES.JUWELS.CODEDIR + 'after_' + str(rank)
    
    fluxsize = fluxbatch.size(-1)
    fluxbatch = fluxbatch.view(b,c,fluxsize,fluxsize)
    # plot_grid(fluxbatch, batch_size = 64, gray=True, cmap='hot', normalize=True, save=True, savename=savename)
    
    return fluxbatch
    # plt.imshow(fluxbatch[])


def random_crop_side(tensor, p_crop):


    # Zufällige Auswahl der Seite für jedes Bild
    
    b, c, w, h = tensor.size()
    choices = th.randint(13, (b*c,), device=tensor.device)

    tensor = tensor.view(b*c,1,w,h)
    
    # Ergebnis-Tensor initialisieren
    cropped_tensors = []
    
    assert 0 <= p_crop and p_crop <= 1 , "P crop must be between 0 and 1!" 
    
    for i in range(b*c):
        if p_crop < random.random():
            cropped_tensors.append(tensor[i:i+1, :, :, :]) # do nothing
            continue
        if choices[i] == 0:
            cropped_tensors.append(tensor[i:i+1, :, 1:, :])  # Remove top row
        elif choices[i] == 1:
            cropped_tensors.append(tensor[i:i+1, :, :-1, :])  # Remove bottom row
        elif choices[i] == 2:
            cropped_tensors.append(tensor[i:i+1, :, :, 1:])  # Remove left column
        elif choices[i] == 3:
            cropped_tensors.append(tensor[i:i+1, :, :, :-1])  # Remove right column
        elif choices[i] == 4:
            cropped_tensors.append(tensor[i:i+1, :, 1:, 1:])  # Remove top row, left column
        elif choices[i] == 5:
            cropped_tensors.append(tensor[i:i+1, :, 1:, :-1]) # Remove top row, right column
        elif choices[i] == 6:
            cropped_tensors.append(tensor[i:i+1, :, :-1, 1:])  # Remove bottom row, left column
        elif choices[i] == 7:
            cropped_tensors.append(tensor[i:i+1, :, :-1, :-1])  # Remove bottom row, right column
            
        elif choices[i] == 8:
            cropped_tensors.append(tensor[i:i+1, :, 1:-1, 1:])  # Remove top row, left column
        elif choices[i] == 9:
            cropped_tensors.append(tensor[i:i+1, :, 1:-1, :-1]) # Remove top row, right column
            
        elif choices[i] == 10:
            cropped_tensors.append(tensor[i:i+1, :, 1:, 1:-1])  # Remove bottom row, left column
        elif choices[i] == 11:
            cropped_tensors.append(tensor[i:i+1, :, :-1, 1:-1])  # Remove bottom row, right column
        elif choices[i] == 12:
            cropped_tensors.append(tensor[i:i+1, :, 1:-1, 1:-1])  # Remove bottom row, right column
        else:
            raise Exception("Something wrong in cropping function!")
            
        cropped_tensors[-1] = th.nn.functional.interpolate(cropped_tensors[-1], size=(w), mode='bilinear')
        
    cropped_tensor = th.cat(cropped_tensors, dim=0)
    cropped_tensor = cropped_tensor.view(b,c,w,h)

    return cropped_tensor


# def rand_contrast(x, scale):
    
#     b, c, w, h = x.size()
#     x = x.view(b*c,1,w,h)
    
#     x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
#     x = (x - x_mean) * (((torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5) * 2.0 * scale) + 1.0) + x_mean
#     x = x.view(b,c,w,h)
    
#     return x

def rand_contrast(x, scale, p):
    b, c, w, h = x.size()
    x = x.view(b * c, 1, w, h)

    # Calculate the number of images to adjust
    num_images = int(b * c * p)
    
    # Generate random indices for the images to adjust
    indices = torch.randperm(b * c)[:num_images]
    
    # Create a mask for images to adjust
    mask = torch.zeros(x.size(0), dtype=torch.bool)
    mask[indices] = True

    x_mean = x[mask].mean(dim=[1, 2, 3], keepdim=True)

    # Apply contrast adjustment only to selected images
    x[mask] = (x[mask] - x_mean) * (
        ((torch.rand(num_images, 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5) * 2.0 * scale) + 1.0
    ) + x_mean

    x = x.view(b, c, w, h)
    
    return x

def add_overexposure(fluxbatch, overexposure_clamp, p):
    device=fluxbatch.device
    batch_size, n_sunPos, fluxsize, _ = fluxbatch.size()
    
    
    fluxbatch = fluxbatch.view(batch_size*n_sunPos,1,fluxsize,fluxsize)
    
    flux_maximas = fluxbatch.view(batch_size*n_sunPos, -1).max(dim=1)[0]

    clamp_start = flux_maximas*overexposure_clamp

    difference = flux_maximas - clamp_start

    overeposure_max = clamp_start + th.rand(fluxbatch.size(0), device=device)*difference
    
    overeposure_max = overeposure_max.unsqueeze(-1).unsqueeze(-1).repeat((1,fluxsize,fluxsize)).unsqueeze(1)
    
    # Calculate the number of images to adjust
    num_images = int(batch_size * n_sunPos * p)
    # Generate random indices for the images to adjust
    indices = th.randperm(batch_size * n_sunPos)[:num_images]
    
    # Create a mask for images to adjust
    mask = torch.zeros(fluxbatch.size(0), dtype=torch.bool)
    mask[indices] = True
    
    fluxbatch[mask] = th.clamp(fluxbatch[mask], th.zeros_like(overeposure_max[mask], device=device), overeposure_max[mask])
    
    fluxbatch = fluxbatch.view(batch_size,n_sunPos,fluxsize,fluxsize)
    del overeposure_max
    return fluxbatch


# def give_dataLoader(dataset, batch_size, cluster, rank, world_size, subscript, shuffle):
    
#         if cluster:
#             sampler = th.utils.data.distributed.DistributedSampler(dataset, 
#                                                                    num_replicas=world_size,
#                                                                    rank=rank, shuffle=True)

#             dataloader = DataLoader(dataset, 
#                                     batch_size=batch_size, 
#                                     sampler=sampler,
#                                     num_workers=4,
#                                     pin_memory=True)
            

#             if rank==0:
#                 utilsDL.print_dataset_info(dataset, subscript)

#         else:
#             utilsDL.print_dataset_info(dataset, subscript)
            
#             dataloader = DataLoader(dataset, 
#                                     batch_size=batch_size, 
#                                     shuffle=shuffle)
            
#         return dataloader
    
def give_dataLoader(dataset, batch_size, cluster, rank, world_size, subscript, shuffle):
    
        if cluster:
            dataloader = DataLoader(dataset, 
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    drop_last=True)
            
            utilsDL.print_dataset_info(dataset, subscript)

            # if rank==0:
            #     utilsDL.print_dataset_info(dataset, subscript)

        else:
            # utilsDL.print_dataset_info(dataset, subscript)
            
            dataloader = DataLoader(dataset, 
                                    batch_size=batch_size, 
                                    shuffle=shuffle,
                                    drop_last=True)
            
        return dataloader


def calculate_distance(position, cfg):
    # if toreal==True:
    position = transform_helPos(th.tensor(position), cfg, direction='toreal')
    position = position.squeeze().squeeze()
    distance = np.sqrt(np.dot(position,position)).item()
    return distance

def calc_uncertainty(zcntrl_pred):
    std = np.std(zcntrl_pred, axis=0)
    return np.mean(std)

def apply_uncertainty(df, real_sim):
    grouped = df.groupby('helname')
    
    if real_sim == "real":
        df['uncertainties'] = grouped['zcntrl_pred_real'].transform(lambda x: calc_uncertainty(np.stack(x.values)))
    elif real_sim == "sim":
        df['uncertainties_sim'] = grouped['zcntrl_pred_sim'].transform(lambda x: calc_uncertainty(np.stack(x.values)))
    else:
        raise Exception("real or sim!")
    return df

def calculate_ssim(df, real):
    gt = df["zcntrl"]
    
    if real==True:
        pred = df["zcntrl_pred_real"]
    elif real == False:
        pred = df["zcntrl_pred_sim"]
    else:
        raise Exception 
    
    pred = facets_to_surface(th.tensor(pred)).numpy()[0,0,:,:]
    gt = facets_to_surface(th.tensor(gt)).numpy()[0,0,:,:]
    
    return ssim(pred, gt, data_range=0.006)


def give_ssim(pred, gt):
    pred = facets_to_surface(th.tensor(pred)).numpy()[0,0,:,:]
    gt = facets_to_surface(th.tensor(gt)).numpy()[0,0,:,:]
    
    return ssim(pred, gt, data_range=0.006)
    

def apply_acc_zcntrl(df, real):
    gt = df["zcntrl"]
    
    if real==True:
        pred = df["zcntrl_pred_real"]
    elif real == False:
        pred = df["zcntrl_pred_sim"]
    else:
        raise Exception 
    pred = facets_to_surface(th.tensor(pred))
    gt = facets_to_surface(th.tensor(gt))
    
    acc = give_accuracy(gt, pred)
    return acc

def apply_acc_defl(df, real):
    gt = df["flux_defl"]
    
    if real==True:
        pred = df["flux_pred_real"]
    elif real == False:
        pred = df["flux_ideal"]
    else:
        raise Exception 
    pred = th.tensor(pred)
    gt = th.tensor(gt)
    
    acc = give_accuracy(gt, pred, normalize=True)
    return acc.item()

def apply_acc(df, gt_string, pred_string, normalize=True):
    gt = df[gt_string]
    pred = df[pred_string]
    
    if isinstance(gt,np.ndarray):
        pred = th.tensor(pred)
        gt = th.tensor(gt)
        
        # print(pred.size())
        # print(gt.size())
        acc = give_accuracy(gt, pred, normalize=normalize)
        # print(acc)
        return acc.item()
    else:
        return None


def apply_acc_(df, gt_string, pred_string):
    
    gt = df[gt_string]
    pred = df[pred_string]
    if isinstance(gt,np.ndarray):
        pred = th.tensor(pred).unsqueeze(0).unsqueeze(0)
        gt = th.tensor(gt).unsqueeze(0).unsqueeze(0)
        print(pred.size())
        print(gt.size())
        acc = give_accuracy(gt, pred, normalize=True)
        return acc.item()
    else:
        return None
    
# Prüfen und Subtrahieren
def subtract_rows(df, row1_id, row2_id):
    row1 = df.iloc[row1_id]
    row2 = df.iloc[row2_id]
    
    # Prüfen, ob eine der beiden Zeilen `None` enthält
    if row1.isnull().any() or row2.isnull().any():
        return None  # Oder ein Standardwert wie np.nan
    
    # Subtraktion durchführen
    return row1 - row2


def shift_fluxes_to_aimpoint(images, 
                             translations, 
                             tracking_error_translation=None):
    
    if images.ndim == 2:  # (H, W)
        images = images.unsqueeze(0).unsqueeze(0)  # Batch- und Channel-Dimension hinzufügen
    elif images.ndim == 3:  # (C, H, W)
        images = images.unsqueeze(1)  # Nur Batch-Dimension hinzufügen
    elif images.ndim == 4:  # (B, C, H, W)
        pass  # Keine Änderung notwendig
    else:
        raise ValueError(f"Unerwartete Tensor-Dimensionen: {images.shape}. Erwartet: (H, W), (C, H, W), oder (B, C, H, W).")
    
    device = images.device
    N, C, H, W = images.size()
    # Normalisierte Translationen (Pixel -> [-1, 1])
    
    if isinstance(tracking_error_translation, th.Tensor):
        translations = translations + tracking_error_translation
        
    translations_normalized = translations / th.tensor([W / 2, H / 2], device=device)
    
    # Basisgitter erstellen
    base_grid = F.affine_grid(
        th.eye(2, 3, device=device).unsqueeze(0).repeat(N, 1, 1),  # Identitätsmatrix für jedes Bild
        size=(N, C, H, W),  # Zielgröße
        align_corners=True,
    )
    
    # Translation zu jedem Bild hinzufügen
    translated_grid = base_grid + translations_normalized.view(N, 1, 1, 2)
    
    # Translation auf die Bilder anwenden
    translated_images = F.grid_sample(
        images, translated_grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    return translated_images


def csv_to_tensor(file_path):
    """
    Lädt eine CSV-Datei mit Bilddaten und konvertiert sie in einen PyTorch-Tensor.
    Die Bildform wird automatisch erkannt.
    
    Args:
        file_path (str): Pfad zur CSV-Datei.
        
    Returns:
        torch.Tensor: Tensor mit Bilddaten, geformt als (N, H, W).
    """
    # CSV-Daten laden
    data = pd.read_csv(file_path, header=None, delimiter="\t", decimal=",")  # Header anpassen, falls erforderlich

    # In einen PyTorch-Tensor konvertieren
    
    data_tensor = th.tensor(data.values, dtype=th.float32)
    return data_tensor



def generate_gif_from_plots(save_dir, gif_name="output.gif", duration=300):
    """Erstellt ein GIF aus gespeicherten Plot-Bildern."""
    images = []
    for file in sorted(os.listdir(save_dir)):
        if file.endswith(".png"):
            images.append(Image.open(os.path.join(save_dir, file)))
    images[0].save(
        os.path.join(save_dir, gif_name),
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

def plot_equipotential_lines(flux_defl_sum, flux_iDLR_sum, flux_ideal_sum, cmap="hot", width=4.35, height=5.22,):

    """

    Plottet die Äquipotentiallinien der drei Flussdichten übereinander.

    

    Args:

        flux_defl_sum (torch.Tensor): Tensor der ersten Flussdichte (H, W).

        flux_iDLR_sum (torch.Tensor): Tensor der zweiten Flussdichte (H, W).

        flux_ideal_sum (torch.Tensor): Tensor der dritten Flussdichte (H, W).

        cmap (str): Colormap für die Hintergrundanzeige.

    """

    # Sicherstellen, dass die Eingaben 2D-Tensoren sind

    flux_defl_sum = flux_defl_sum.squeeze().cpu().numpy()

    flux_iDLR_sum = flux_iDLR_sum.squeeze().cpu().numpy()

    flux_ideal_sum = flux_ideal_sum.squeeze().cpu().numpy()



    # Hintergrundbild erstellen (z. B. Summierte Flüsse)

    # combined_flux = flux_defl_sum + flux_iDLR_sum + flux_ideal_sum



    # Plotten

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(flux_defl_sum, cmap=cmap)

    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label(r"Flux Density / $\frac{kW}{m^2}$", fontsize=14)



    # Äquipotentiallinien der einzelnen Flüsse

    ax.contour(flux_defl_sum, levels=15, colors="black", linewidths=1, linestyles="solid", label="Deflectometry")

    ax.contour(flux_iDLR_sum, levels=15, colors="red", linewidths=1, linestyles="solid", label="iDLR")

    ax.contour(flux_ideal_sum, levels=15, colors="gray", linewidths=1, linestyles="dotted", label="Ideal")



    # Achsen und Titel

    ax.set_title("Equipotential Lines of Flux Densities")
    
    ax.set_xticks([0, 14, 29])
    ax.set_xticklabels(["0", str(int(width/2)), str(int(width))])
    
    ax.set_yticks([0, 17, 35])
    ax.set_yticklabels(["0", str(int(height/2)), str(int(height))])
    
    ax.set_xlabel("width / m")

    ax.set_ylabel("height / m")



    # Legende hinzufügen

    handles = [

        plt.Line2D([], [], color="black", label="Deflectometry"),

        plt.Line2D([], [], color="red", label="iDLR"),

        plt.Line2D([], [], color="gray", label="Ideal"),

    ]

    ax.legend(handles=handles, loc="best")



    plt.show()


def getCosineEfficiency(positions, sun_position, aim_points):
    """
    Compute the cosine efficiency based on the cosine of the angle 
    between the sun vector and the aiming direction vector.

    Parameters:
    positions (torch.Tensor): Tensor of heliostat positions, shape (n, 3).
    sun_position (torch.Tensor): Tensor of the sun's position, shape (3,).
    aim_points (torch.Tensor): Tensor of aim points, shape (n, 3).

    Returns:
    torch.Tensor: Cosine efficiency for each heliostat, shape (n,).
    """
    # Compute aiming direction vectors (from heliostat to aim point)
    aim_directions = aim_points - positions

    # Compute sun vectors (from heliostat to sun)
    
    if sun_position.ndim == 1:
        sun_position = sun_position.unsqueeze(0).repeat(positions.size(0),1) 
        
    # sun_vectors = sun_position
    # Normalize the vectors
    aim_directions_norm = aim_directions / torch.norm(aim_directions, dim=1, keepdim=True)
    sun_position = sun_position / torch.norm(sun_position, dim=1, keepdim=True)

    # Compute dot product between normalized vectors
    dot_products = torch.sum(aim_directions_norm * sun_position, dim=1)

    # Clamp values to the range [-1, 1] to avoid numerical issues
    cosine_efficiency = torch.clamp(dot_products, -1.0, 1.0)

    return cosine_efficiency


def scale_flux_to_energy(cfg, fluxes, helPos, sunPos, aimPoints, nrays, 
                         n_missed_rays):
    
    """
    The normalised flux density is scaled to kw/m**2. 
    Importantly: The number rays for the simulation should not be changed (400)

    Parameters:
    helPos (torch.Tensor): Tensor of heliostat positions, shape (n, 3).
    sun_position (torch.Tensor): Tensor of the sun's position, shape (n, 3).
    aim_points (torch.Tensor): Tensor of aim points, shape (n, 3).
    n_missed_rays (torch.Tensor): Tensor of missed rays, shape (n, 3).

    Returns:
    torch.Tensor: Cosine efficiency for each heliostat, shape (n,).
    """
    
    assert fluxes.ndim==3
    assert helPos.ndim==1
    
    assert aimPoints.ndim==2
    
    nfluxes, W, H = fluxes.size()
    
    cosine_efficiency = getCosineEfficiency(helPos, sunPos, aimPoints)
    DNI = cfg.AC.SUN.DNI 
    A_heliostat = 4 * 1.6 * 1.25
    reflectivity = 0.9

    frac_missed = n_missed_rays / nrays
    
    # Die Gesamte Energie in der Flussdichtewird auf 1 normiert.
    fluxes = fluxes/th.sum(fluxes, dim=(1,2), keepdim=True)
    
    npixel = W * H
    area_receiver = cfg.AC.TARGET.RECEIVER.PLANE_X * cfg.AC.TARGET.RECEIVER.PLANE_Y
    pixelarea = area_receiver/npixel
    
    scale = DNI * reflectivity * A_heliostat * cosine_efficiency * (1- frac_missed) / pixelarea 

    fluxes = scale.unsqueeze(-1).unsqueeze(-1) * fluxes / 1000
    
    spillage = frac_missed * th.sum(fluxes, dim=(1,2))
    
    return fluxes, spillage 
    

def sample_rayleigh(shape, scale=1.0):
    """
    Generiert einen Torch-Tensor mit Stichproben aus einer Rayleigh-Verteilung.

    Args:
    - shape: Tuple, die die Form des Tensors angibt (z.B. (100,) für 100 Werte).
    - scale: Die Skalenparameter (σ) der Rayleigh-Verteilung.

    Returns:
    - Torch-Tensor mit Rayleigh-verteilten Stichproben.
    """
    uniform_samples = torch.rand(shape)  # Gleichverteilte Zufallszahlen (0, 1)
    rayleigh_samples = scale * torch.sqrt(-2 * torch.log(uniform_samples))
    return rayleigh_samples


def give_tracking_error_translations(helpos, tracking_error_mrad, receiverpos, area_receiver, pixel_receiver, return_pixel=True, give_from_rayleigh_distribution=True):
    """
    The normalised flux density is scaled to kw/m**2. 
    Importantly: The number rays for the simulation should not be changed (400)

    Parameters:
    helPos (torch.Tensor): Tensor of heliostat positions, shape (n, 3).
    tracking_error_mrad (float):
    tracking_error_mrad (torch.Tensor): position of receiver, shape (3).
    area_receiver (Touple):  shape (2).
    pixel_receiver (Touple):  shape (2).

    Returns:
    torch.Tensor: translation on receiver caused by tracking inaccuracies, shape (n,2).
    """
    nhels = helpos.size(0)

    distances_to_receiver = th.norm(helpos + receiverpos.unsqueeze(0).repeat( (nhels,1) ), dim=1)
    
    if not give_from_rayleigh_distribution:
        tracking_error_rad = tracking_error_mrad * 0.001
        translation_radius = np.tan(tracking_error_rad) * distances_to_receiver
    elif give_from_rayleigh_distribution:
        tracking_error_mrad = sample_rayleigh(nhels, scale=tracking_error_mrad)
        tracking_error_rad = tracking_error_mrad * 0.001
        translation_radius = th.tan(tracking_error_rad) * distances_to_receiver

    else:
        raise Exception
    angular_direction_of_shift = th.rand(nhels)*360
    angular_direction_of_shift = th.deg2rad(angular_direction_of_shift)
    
    tracking_translations_meter_x = translation_radius * th.cos(angular_direction_of_shift)
    tracking_translations_meter_y = translation_radius * th.sin(angular_direction_of_shift)
    
    m2pix_x = pixel_receiver[0]/area_receiver[0]
    m2pix_y = pixel_receiver[1]/area_receiver[1]
    
    tracking_translations_pixel_x = tracking_translations_meter_x * m2pix_x
    tracking_translations_pixel_y = tracking_translations_meter_y * m2pix_y

    tracking_translations_pixel = th.round(th.cat( [tracking_translations_pixel_x.unsqueeze(1),
                                                    tracking_translations_pixel_y.unsqueeze(1)], dim=1))
    
    if return_pixel:
        return tracking_translations_pixel
    
    elif not return_pixel:
        return th.cat( [tracking_translations_meter_x.unsqueeze(1),
                        tracking_translations_meter_y.unsqueeze(1)], dim=1)
    
    else:
        raise Exception
        
        
def mean_normals_by_quadrant(points: torch.Tensor, normals: torch.Tensor):
    """
    Berechnet den mittleren Normalenvektor für jeden Quadranten basierend auf XY-Koordinaten.
    
    :param points: Tensor der Form [N,2], der XY-Koordinaten enthält.
    :param normals: Tensor der Form [N,3], der Normalenvektoren enthält.
    :return: Dictionary mit den mittleren Normalenvektoren pro Quadrant.
    """
    
    quadrants = {
        "Q1": (points[:, 0] > 0) & (points[:, 1] > 0),
        "Q2": (points[:, 0] < 0) & (points[:, 1] > 0),
        "Q3": (points[:, 0] < 0) & (points[:, 1] < 0),
        "Q4": (points[:, 0] > 0) & (points[:, 1] < 0),
    }

    mean_normals = {}
    
    for quadrant, mask in quadrants.items():
        if mask.any():  # Prüfen, ob Punkte im Quadranten existieren
            mean_vector = normals[mask].mean(dim=0)  # Mittelwert berechnen
            mean_vector = mean_vector / mean_vector.norm()  # Normalisieren
            mean_normals[quadrant] = mean_vector
        else:
            mean_normals[quadrant] = torch.tensor([0.0, 0.0, 0.0])  # Falls leer, Null-Vektor

    return mean_normals
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

if __name__ == '__main__':
    
    vec = th.tensor([0,-1,0], dtype=th.float32)
    
    ae = vec_to_ae(vec)
    
    print(ae)
