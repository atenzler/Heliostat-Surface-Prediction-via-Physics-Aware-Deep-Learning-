"""
integrate_raytracer.py

Integration layer between the DeepLARTS neural network and the ARTIST raytracer.

This module provides:
- `get_position_and_canting`: Load heliostat geometry and canting from JSON.
- `overwrite_scenario`: Replace heliostat surfaces in an ARTIST scenario with NN-predicted Z-residuals.
- `raytracing`: Perform heliostat-based raytracing for given sun positions and aim points.

Usage:
- Called after NN predictions to simulate flux maps.
- Relies on ARTIST’s Scenario and HeliostatRayTracer classes.
- Used in `run_experiment.py` for validation of predicted heliostat surfaces.

Notes:
- Paths to scenarios and heliostat property JSONs are currently hardcoded.
- Adapt these to your own `data/` structure before running.
"""

import gc
import json
import torch
import math
import matplotlib.pyplot as plt

from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.util.utils import convert_wgs84_coordinates_to_local_enu, convert_3d_point_to_4d_format


def get_position_and_canting(helio_list, power_plant_position, dic):
      # Dictionary to store all heliostat data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for helio_name in helio_list:
        file_path = (
            fr"..\defl_data\{helio_name}\Properties\{helio_name}-heliostat-properties.json")
        with open(file_path, "r") as file:
            data = json.load(file)

        # Extract data into variables
        heliostat_position = torch.tensor(data["heliostat_position"])
        heliostat_position_enu = convert_wgs84_coordinates_to_local_enu(heliostat_position, power_plant_position, device=device)
        heliostat_position_enu = convert_3d_point_to_4d_format(heliostat_position_enu, device=device)

        print(data)

        # Define facet details
        facets = []
        for facet in data["facet_properties"]["facets"]:
            print(facet)
            translation_vector = facet["translation_vector"]
            canting_e = facet["canting_e"]
            canting_n = facet["canting_n"]
            facets.append({
                "translation_vector": translation_vector,
                "canting_e": canting_e,
                "canting_n": canting_n,
            })
        print(facets)

        dic[helio_name] = {
            "heliostat_position_enu": heliostat_position_enu,  # Convert tensor to list for easier handling
            "facets": facets
    }

    return dic


def overwrite_scenario(
    aim_point, heliostat_positions, z_cntrl_points, new_scenario,
    prototype_surface, xy_grid, batch_size, first_overwrite, device,
    heliostat_names=None, get_ideal_surface=None, translation_vector=None,
    surface_pool=None
):
    num_helios = z_cntrl_points.shape[0]
    all_surface_points = torch.zeros(num_helios, 10000, 4, device=device)
    all_surface_normals = torch.zeros(num_helios, 10000, 4, device=device)


    for batch_index in range(num_helios):
        surface = surface_pool[batch_index]  # Reuse instance instead of copying

        z_residuals = z_cntrl_points[batch_index].unsqueeze(-1)  # [4, 8, 8, 1]

        if heliostat_names and get_ideal_surface and translation_vector is not None:
            raw_name = heliostat_names[batch_index]
            name = raw_name[0] if isinstance(raw_name, tuple) else raw_name
            # Get CPU version and move to device
            ideal_surface_cpu = get_ideal_surface(name)
            ideal_surface = ideal_surface_cpu.to(device)
            del ideal_surface_cpu  #  Remove CPU reference immediately

            # Apply translation in a gradient-safe way (no in-place ops!)
            translation = translation_vector.view(4, 1, 1, 3)  # [4, 1, 1, 3]
            ideal_surface = ideal_surface - translation

            # Add Z residuals (broadcasting)
            ideal_surface[..., 2] += z_residuals.squeeze(-1)
            del z_residuals

            control_points = ideal_surface  # reuse
            del ideal_surface
        else:
            control_points = torch.cat([xy_grid, z_residuals], dim=3)  # [4,8,8,3]
            del z_residuals

        for i in range(4):
            surface.facets[i].control_points = control_points[i]
            if translation_vector is not None:
                surface.facets[i].translation_vector[:] = torch.cat(
                    [translation_vector[i], torch.tensor([0.0], device=device)]
                )

        del control_points

        pts, norms = surface.get_surface_points_and_normals(device=device)
        all_surface_points[batch_index].copy_(pts.reshape(-1, 4))
        all_surface_normals[batch_index].copy_(norms.reshape(-1, 4))

    new_scenario.heliostat_field.number_of_heliostats = num_helios
    new_scenario.heliostat_field.all_heliostat_positions = heliostat_positions
    new_scenario.heliostat_field.all_aim_points = aim_point.repeat(num_helios, 1)
    new_scenario.heliostat_field.all_surface_points = all_surface_points.to(device)
    new_scenario.heliostat_field.all_surface_normals = all_surface_normals.to(device)

    if first_overwrite:
        hf = new_scenario.heliostat_field
        hf.all_initial_orientations = hf.all_initial_orientations.repeat(num_helios, 1)
        hf.all_kinematic_deviation_parameters = hf.all_kinematic_deviation_parameters.repeat(num_helios, 1)
        hf.all_actuator_parameters = hf.all_actuator_parameters.repeat(num_helios, 1, 1)
        hf.all_aligned_heliostats = torch.tensor([0], device=device).repeat(num_helios)
        hf.all_preferred_reflection_directions = torch.zeros(num_helios, 4, device=device)
        hf.all_current_aligned_surface_points = torch.zeros(num_helios, 10000, 4, device=device)
        hf.all_current_aligned_surface_normals = torch.zeros(num_helios, 10000, 4, device=device)
    else:
        hf = new_scenario.heliostat_field
        hf.all_aligned_heliostats = torch.tensor([0], device=device).repeat(num_helios)
        hf.all_preferred_reflection_directions = torch.zeros(num_helios, 4, device=device)
        hf.all_current_aligned_surface_points = torch.zeros(num_helios, 10000, 4, device=device)
        hf.all_current_aligned_surface_normals = torch.zeros(num_helios, 10000, 4, device=device)

    # Clean up large temporary tensors
    del all_surface_points
    del all_surface_normals

    gc.collect()
    torch.cuda.empty_cache()

    return new_scenario, False



def raytracing(scenario, sun_positions, aim_point_area, batch_size, show_image, device, evalaute_test_data=False):
    # Align the heliostat.
    incident_ray_directions = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - sun_positions

    scenario.heliostat_field.align_surfaces_with_incident_ray_direction(
        incident_ray_direction=incident_ray_directions, device=device
    )

    # Define the raytracer.
    raytracer = HeliostatRayTracer(
        scenario=scenario,
        batch_size=batch_size,
        world_size=1,
        bitmap_resolution_e=64,
        bitmap_resolution_u=64
    )

    target_area = scenario.get_target_area(aim_point_area)    #only relevant for closer heliostats
    if evalaute_test_data == False:
        target_area.plane_e = 20.0
        target_area.plane_u = 20.0

    # Perform heliostat-based raytracing.
    images = raytracer.trace_rays(
        incident_ray_direction=incident_ray_directions, target_area=target_area, device=device
    )
    del raytracer
    gc.collect()
    torch.cuda.empty_cache()

    if show_image == True:
        images = images.cpu().detach()
        # Determine grid size (square root for nearly equal rows/cols)
        n_images = images.shape[0]  # Number of images
        grid_size = math.ceil(math.sqrt(n_images))  # Grid layout (rows × cols)

        # Create subplots
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

        # Flatten axes array for easy iteration
        axes = axes.flatten()

        # Loop through images and plot them
        for i in range(n_images):
            axes[i].imshow(images[i], cmap="inferno")
            axes[i].axis("off")  # Hide axes for better visualization

        # Hide any unused subplots
        for j in range(n_images, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    return images



