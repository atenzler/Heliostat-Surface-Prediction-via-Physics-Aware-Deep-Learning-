"""
augment_surfaces.py

This script generates augmented heliostat surface datasets by applying:
- Rotation (Z-only, 180° facet reordering)
- Linear interpolation between surfaces
- Random sampling of real surfaces

The augmented datasets are split into training and test sets and saved to disk
as `.pt` and/or `.json` files.

Usage:
    python src.preprocessing.augment_surfaces

Inputs:
- Ideal XY grid from all_surfaces_without_canting.json
- Z-displacement .pt files from input directory
- Hardcoded list of test heliostat names

Outputs:
- Augmented training surfaces (torch tensors)
- Test surfaces (torch tensors)
- Saved under the specified output directory
"""



import torch
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def linear_interpolation(surface1, surface2, alpha=None):

    """
    Linearly interpolate between two surfaces.

    Args:
        surface1 (torch.Tensor or np.ndarray): First surface.
        surface2 (torch.Tensor or np.ndarray): Second surface.
        alpha (float, optional): Interpolation factor in [0, 1].
            If None, a random alpha is drawn.

    Returns:
        torch.Tensor or np.ndarray: Interpolated surface.
    """

    if alpha is None:
        alpha = np.random.uniform(0, 1)
    return alpha * surface1 + (1 - alpha) * surface2

def rotate_surface_z_only(z_surface):
    """
    Rotates the *entire* surface (Z only) by 180 degrees.
    This involves:
    1. Flipping the 8x8 grid inside each facet
    2. Reordering the 4 facets to simulate full surface rotation

    Assumes input shape is (4, 8, 8)
    """
    # Step 1: Flip each facet's grid
    flipped_facet_grids = torch.flip(z_surface, dims=[-2, -1])  # shape still (4, 8, 8)

    # Step 2: Reorder facets like a 2x2 tile rotated 180°
    # Mapping: [0,1,2,3] → [3,2,1,0]
    rotated_surface = flipped_facet_grids[[3, 2, 1, 0]]

    return rotated_surface


def generate_augmented_data(train_surfaces, target_size=160000):
    """Generates augmented surfaces using rotation and interpolation (Z only)."""
    augmented_data = []

    # Rotate all training surfaces (Z only)
    rotated_surfaces = [rotate_surface_z_only(s) for s in train_surfaces]
    augmented_data.extend(rotated_surfaces)

    # Generate new surfaces using interpolation
    while len(augmented_data) < target_size:
        indices = torch.randperm(train_surfaces.shape[0])[:2]
        s1, s2 = train_surfaces[indices[0]], train_surfaces[indices[1]]
        new_surface = linear_interpolation(s1, s2)
        augmented_data.append(new_surface)

    return torch.stack(augmented_data)


def plot_surface(surface, title="Surface"):
    """Plots a 3D scatter plot of the (X, Y, Z) coordinates in space."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Extract the Z component (the 3rd component) for the scatter plot
    X = surface[:, :, 0]
    Y = surface[:, :, 1]
    Z = surface[:, :, 2]  # Z is in the third component of the surface array

    # Plot as a scatter plot of XYZ points
    ax.scatter(X, Y, Z, c=Z, cmap='viridis', marker='o')

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def save_surfaces_to_json(surfaces, filename_prefix, max_surfaces_per_file, data_split_type):
    """
    Saves augmented surface data into multiple JSON files, each containing a maximum of max_surfaces_per_file surfaces.

    Args:
        surfaces (torch.Tensor): Tensor of shape (N, 4, 8, 8) containing augmented surfaces.
        filename_prefix (str): Prefix for the JSON file names (e.g., "augmented_surface_").
        max_surfaces_per_file (int): Maximum number of surfaces to store in one file.
    """

    num_surfaces = surfaces.shape[0]
    num_files = (num_surfaces + max_surfaces_per_file - 1) // max_surfaces_per_file  # Calculate how many files are needed

    # Initialize a global counter for surface names
    surface_counter = 1

    for file_idx in range(num_files):
        # Calculate start and end indices for the current file
        start_idx = file_idx * max_surfaces_per_file
        end_idx = min((file_idx + 1) * max_surfaces_per_file, num_surfaces)

        # Prepare dictionary for current file
        aug_surface_dict = {}
        for i in range(start_idx, end_idx):
            surface = surfaces[i]
            if data_split_type == "training":
                surface_name = f"AUG{surface_counter}"  # Naming each surface as AUG1, AUG2, ...
            elif data_split_type == "valid":
                surface_name = f"VAL{surface_counter}"
            elif data_split_type == "test":
                surface_name = f"TEST{surface_counter}"
            surface_counter += 1  # Increment the surface counter
            facets = {}

            for j in range(4):  # Each surface has 4 facets
                facet_key = f"facet_{j + 1}"  # Naming facets as facet_1, facet_2, ...
                facets[facet_key] = {
                    "facet_key": facet_key,
                    "control_points": surface[j].tolist()  # Store control points
                }

            # Store surface data
            aug_surface_dict[surface_name] = facets

        # Save the current file
        filename = f"{filename_prefix}_{file_idx}.json"
        with open(filename, "w") as f:
            json.dump(aug_surface_dict, f, indent=4)



def save_surfaces_to_pt(surfaces, directory, filename_prefix, data_split_type):
    """
    Saves each augmented surface as an individual .pt file.

    Args:
        surfaces (torch.Tensor): Tensor of shape (N, 4, 8, 8) containing augmented surfaces.
        directory (str): Path to the directory where files should be saved.
        filename_prefix (str): Prefix for the .pt file names (e.g., "augmented_surface").
        data_split_type (str): Type of data split ("training", "valid", "test").
    """

    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists

    for i in range(surfaces.shape[0]):
        if data_split_type == "training":
            filename = f"{filename_prefix}_AUG{i+1}.pt"
        elif data_split_type == "valid":
            filename = f"{filename_prefix}_VAL{i+1}.pt"
        elif data_split_type == "test":
            filename = f"{filename_prefix}_TEST{i+1}.pt"

        file_path = os.path.join(directory, filename)
        torch.save(surfaces[i].detach().clone(), file_path)  # Save each surface as a separate .pt file

        print(f"Saved surface {i+1} to {file_path}")

# Example usage
if __name__ == "__main__":

    start_time = time.time()
    target_size = 50000
    max_surfaces_per_file = 10000

    # Define your test heliostats by name only (no date parts)
    testset_names = [
        'AA31', 'AA35', 'AA44', 'AB42', 'AC30', 'AC35', 'AC38', 'AC42', 'AD31', 'AD33',
        'AD35', 'AD42', 'AD44', 'AE25', 'AE29', 'AE30', 'AF37', 'AF42', 'AF46',
        'AX56', 'AY26', 'AY32', 'AY37', 'AY60', 'BA27', 'BA63', 'BA65', 'BB41', 'BC62', 'BC66',
        "AB38", "AB39", "AB44", "AC37", "AC43", "AD30", "AD34", "AD36", "AD37", "AD40", "AD45", "AD46",
        "AE27", "AF32", "AF33", "AF34", "AF38", "AF39", "AF40", "AF44", "AF45", "AX54",
        "AY28", "AY35", "AY36", "AY38", "AY39", "AZ55", "BA28", "BA29", "BA61", "BC61", "BC64"
    ]
    testset_names = list(set(testset_names))  # remove duplicates

    json_path = r"all_surfaces_without_canting.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    # Load ideal grid just for ref/validation (not added again)
    ideal_grid = torch.tensor(data["AA23"]["facet_1"]["control_points"], device=device)
    ideal_grid_xy = ideal_grid[:, :, :2].unsqueeze(0).repeat(4, 1, 1, 1)

    # Path to .pt files
    z_disp_dir = r"...\z_displacements"

    train_surfaces_list = []
    test_surfaces_list = []

    for filename in os.listdir(z_disp_dir):
        if filename.endswith("_displacement_z.pt"):
            heliostat_name = filename.split("_")[0]  # e.g., "AA23" from "AA23_displacement_z.pt"

            full_path = os.path.join(z_disp_dir, filename)
            displacement_tensor = torch.load(full_path, map_location=device).clone()
            displacement_tensor[..., 0:2] = ideal_grid_xy  # Replace XY

            if heliostat_name in testset_names:
                test_surfaces_list.append(displacement_tensor)
            else:
                train_surfaces_list.append(displacement_tensor)

    # Convert to tensors
    train_surfaces = torch.stack(train_surfaces_list)
    test_surfaces = torch.stack(test_surfaces_list)


    # Augment only 90% of training data (Z only)
    num_real_to_include = int(0.1 * target_size)
    num_augmented = target_size - num_real_to_include

    # Extract only Z channel for augmentation
    train_z_only = train_surfaces[..., 2]  # Shape: (297, 4, 8, 8)
    augmented_z = generate_augmented_data(train_z_only, target_size=num_augmented)

    # Sample real Z surfaces (with replacement)
    real_samples_z = train_z_only[torch.randint(0, train_z_only.size(0), (num_real_to_include,))]

    # Merge
    combined_z = torch.cat([real_samples_z, augmented_z], dim=0)  # (100k, 4, 8, 8)

    # Reattach XY grid to Z
    device = combined_z.device
    ideal_grid_xy_expanded = ideal_grid_xy.unsqueeze(0).repeat(combined_z.size(0), 1, 1, 1, 1)  # (100k, 4, 8, 8, 2)
    augmented_surfaces_xyz = torch.cat([ideal_grid_xy_expanded, combined_z.unsqueeze(-1)], dim=-1)  # (100k, 4, 8, 8, 3)

    # Save
    directory = r"\save_augmented_surfaces"
    save_surfaces_to_pt(augmented_surfaces_xyz, filename_prefix="augmented_surface", data_split_type="training", directory=directory)
    save_surfaces_to_pt(test_surfaces, filename_prefix="test", data_split_type="test", directory=directory)



