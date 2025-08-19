"""
compute_z_displacements.py

Computes Z-displacement tensors for heliostat surfaces by comparing
real measured control points against ideal NURBS-fitted surfaces.

Workflow:
- Load all heliostat surface definitions from JSON.
- Load corresponding ideal NURBS surfaces from .pt files.
- Align real surfaces to ideal ones by facet permutation (XY alignment).
- Compute displacement tensors (real - ideal).
- Save each Z-displacement tensor as {heliostat_name}_displacement_z.pt.

Also tracks which heliostats have been processed and which remain unfinished.
"""


import itertools
import json
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def list_folders_in_directory(directory_path: str) -> list:
    # List all entries in the directory and filter to only include folders
    folder_list = [entry for entry in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, entry))]
    return folder_list

# Load JSON data into a Python dictionary
with open(r"...\all_heliostat_surfaces.json", 'r') as f:
    data_dict = json.load(f)

finished_helio_list = []

# Loop through each heliostat
for heliostat_name, heliostat_data in data_dict.items():
    try:
        # Try to load ideal surface
        ideal_surface_path = fr"...\{heliostat_name}_ideal_nurbs.pt"
        if not os.path.exists(ideal_surface_path):
            print(f"⚠️ Skipping {heliostat_name}: Ideal surface file not found.")
            continue

        ideal_surface = torch.load(ideal_surface_path)
        ideal_surface = ideal_surface.clone()

        # Prepare real surface and translations
        real_cp_list = []
        for i in range(4):
            real_facet = heliostat_data[f"facet_{i + 1}"]
            cp = torch.tensor(real_facet["control_points"])
            tvec = torch.tensor(real_facet["translation_vector"][:3])

            cp[:, :, 0] += tvec[0]
            cp[:, :, 1] += tvec[1]
            #cp[:, :, 2] += tvec[2]

            real_cp_list.append(cp)

        real_surface = torch.stack(real_cp_list)

        # Align facets based on XY match
        ideal_xy = ideal_surface[:, :, :, :2]
        real_xy = real_surface[:, :, :, :2]

        min_diff = float("inf")
        best_perm = None

        for perm in itertools.permutations(range(4)):
            permuted_real_xy = real_xy[list(perm)]
            diff = torch.sum((permuted_real_xy - ideal_xy) ** 2).item()
            if diff < min_diff:
                min_diff = diff
                best_perm = perm

        real_surface_aligned = real_surface[list(best_perm)]


        # Compute z-displacement
        z_displacement = real_surface_aligned - ideal_surface


        torch.save(z_displacement, fr"...\z_displacements\{heliostat_name}_displacement_z.pt")
        finished_helio_list.append(heliostat_name)
        print(f"Processed {heliostat_name}")

    except Exception as e:
        print(f"Error with {heliostat_name}: {e}")
        continue




