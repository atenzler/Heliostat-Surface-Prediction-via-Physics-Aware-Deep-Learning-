"""
generate_ideal_heliostats.py

Generates "ideal" heliostat surfaces by fitting NURBS representations based on
facet canting parameters. Each heliostat is reconstructed and saved as a tensor
(.pt) and plotted as a .png for inspection.

Workflow:
- Load list of heliostats to process from JSON.
- For each heliostat:
    - Extract canting and translation vectors.
    - Generate an ideal heliostat surface with SurfaceConverter.
    - Fit NURBS surfaces to approximate geometry.
    - Save results (.pt tensor and .png surface plot).

Inputs:
- heliostat property JSONs (canting info).
- unfinished_helios.json (list of heliostats to process).
- Ideal grid file (AA23.json).

Outputs:
- {heliostat_name}_ideal_nurbs.pt (control points tensor).
- {heliostat_name}_ideal_surface.png (surface plot).
"""


import json
import time
import torch
import os

from classify_heliostats import classify_and_plot_heliostats
from artist.util import config_dictionary
from artist.util.surface_converter import SurfaceConverter
from artist.util.utils import convert_3d_point_to_4d_format, convert_3d_direction_to_4d_format

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data_path = r".../heliostat_properties"
receiver_filtered_dict, receiver_helios, other_helios = classify_and_plot_heliostats(data_path, plot=False)

# File path to load the list from
input_path = r".../helios_to_process.json"

# Load the list
with open(input_path, "r") as f:
    heliostats_to_process = json.load(f)

# Initialize SurfaceConverter (once)
surface_converter = SurfaceConverter(step_size=1, max_epoch=20000,
                                     conversion_method=config_dictionary.convert_nurbs_from_points)

# Output directory for saving NURBS surfaces
output_dir = r".../ideal_surfaces"
os.makedirs(output_dir, exist_ok=True)

# Loop through filtered heliostats
for heliostat_name in heliostats_to_process:

    start_time = time.time()
    facets = receiver_helios[heliostat_name]["facets"]


    facet_translation_vectors = torch.stack([torch.tensor(facet["translation_vector"] + [0], device=device) for facet in facets])
    canting_e = torch.stack([torch.tensor(facet["canting_e"] + [0], device=device) for facet in facets])
    canting_n = torch.stack([torch.tensor(facet["canting_n"] + [0], device=device) for facet in facets])


    # Generate surface
    ideal_surface_points, ideal_surface_normals = surface_converter.generate_ideal_juelich_heliostat_surface(
        cantings_e=canting_e,
        cantings_n=canting_n,
        facet_translation_vectors=facet_translation_vectors,
        number_of_surface_points=2000,
        device=device
    )

    # Format for NURBS fitting
    ideal_surface_points = torch.stack(ideal_surface_points)
    ideal_surface_points = convert_3d_point_to_4d_format(ideal_surface_points, device=device)
    ideal_surface_normals = torch.stack(ideal_surface_normals)
    ideal_surface_normals = convert_3d_direction_to_4d_format(ideal_surface_normals, device=device)

    # JSON file with constant xy grid
    with open(r".../AA23.json","r") as f:
        data = json.load(f)

    ideal_grid = torch.tensor(data["facet_1"]["control_points"], device=device)
    ideal_grid_xy = ideal_grid[:, :, :2]
    ideal_grid_xy = ideal_grid_xy.unsqueeze(0).repeat(4, 1, 1, 1)

    # Fit NURBS surfaces
    ideal_nurbs_tensors = []
    for i in range(4):
        ideal_nurbs_surface = surface_converter.fit_nurbs_surface(
            surface_points=ideal_surface_points[i],
            surface_normals=ideal_surface_normals[i],
            conversion_method=config_dictionary.convert_nurbs_from_points,
            number_control_points_e=8,
            number_control_points_n=8,
            tolerance=3e-5,
            initial_learning_rate=1e-3,
            max_epoch=1000,
            degree_n=1,
            degree_e=1,
            optimize_only_z_cntrl_points=True,
            ideal_grid_xy=ideal_grid_xy[i],
            device=device
        )
        ideal_nurbs_tensor = ideal_nurbs_surface.control_points.to(device)
        ideal_nurbs_tensors.append(ideal_nurbs_tensor)

    # Stack and apply facet translations
    ideal_nurbs_tensor = torch.stack(ideal_nurbs_tensors).clone()
    for i in range(4):
        ideal_nurbs_tensor[i, :, :, 0] -= facet_translation_vectors[i, 0]
        ideal_nurbs_tensor[i, :, :, 1] -= facet_translation_vectors[i, 1]

    # Save
    save_path = os.path.join(output_dir, f"{heliostat_name}_ideal_nurbs.pt")
    torch.save(ideal_nurbs_tensor, save_path)
    filename_surface_plot = os.path.join(output_dir, f"{heliostat_name}_ideal_surface.png")

    end_time = time.time()
    duration = end_time - start_time


