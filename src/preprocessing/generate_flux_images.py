"""
generate_flux_images.py

Generates flux images for training using ARTIST raytracer in parallel.

Main steps:
- Load scenario (.h5) with heliostat and receiver definitions.
- Load augmented surface tensors and ideal heliostat surfaces.
- Randomly assign heliostat positions and heliostat properties.
- Overwrite scenario with new surfaces/positions.
- Run raytracing for multiple sun positions.
- Save generated flux images, sun vectors, and metadata in chunks.

Outputs:
- images_chunk_XXX.pt         (raytraced images)
- sun_vectors_chunk_XXX.pt    (sun positions for each image)
- metadata_chunk_XXX.json     (mapping to heliostat/surface info)

Note:
- Parallel loading uses ThreadPoolExecutor for efficiency.
"""


from artist.util.utils import convert_3d_point_to_4d_format, convert_3d_direction_to_4d_format
from artist.util.scenario import Scenario
from artist.field.surface import Surface
from artist.raytracing.heliostat_tracing import HeliostatRayTracer

from classify_heliostats import classify_and_plot_heliostats
from sun_positions_utils import random_sun_positions_whole_year_new_positions

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import torch
import json
import itertools
import h5py
import os
import copy
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def align_facets_by_xy(ref_surface, target_surface):
    """
    Aligns the facet ordering of `target_surface` to match `ref_surface` based on XY positions.
    Assumes both tensors are shape (4, 8, 8, 3).
    """
    ref_xy = ref_surface[..., :2]
    target_xy = target_surface[..., :2]

    min_diff = float("inf")
    best_perm = list(range(4))

    for perm in itertools.permutations(range(4)):
        permuted = target_xy[list(perm)]
        diff = torch.sum((permuted - ref_xy) ** 2).item()
        if diff < min_diff:
            min_diff = diff
            best_perm = list(perm)

    return target_surface[best_perm]

def overwrite_scenario_xyz_surface(aim_point, all_positions_tensor, aug_surfaces_tensor, ideal_surfaces_tensor, new_scenario, prototype_surface,
                                    translation_vector_tensor, canting_e_tensor, canting_n_tensor, first_overwrite,  device):

    num_helios = len(aug_surfaces_tensor)
    all_surface_points = torch.zeros(num_helios, 10000, 4)
    all_surface_normals = torch.zeros(num_helios, 10000, 4)

    for j in range(num_helios):
        surface = copy.copy(prototype_surface)

        canting_e_helio = canting_e_tensor[j]
        canting_n_helio = canting_n_tensor[j]
        translation_vector_helio = translation_vector_tensor[j]
        ideal_surface_helio = ideal_surfaces_tensor[j]
        aug_surface_helio = aug_surfaces_tensor[j]

        for i in range(4):
            ideal_surface_helio[i, :, :, 0] = ideal_surface_helio[i, :, :, 0] - translation_vector_helio[i, 0]  # Apply translation in x
            ideal_surface_helio[i, :, :, 1] = ideal_surface_helio[i, :, :, 1] - translation_vector_helio[i, 1]  # Apply translation in y

        # Align facets of aug_surface to match ideal_surface
        aug_surface_helio = align_facets_by_xy(ideal_surface_helio, aug_surface_helio)

        control_points_helio = ideal_surface_helio.clone()
        control_points_helio[..., 2] += aug_surface_helio[..., 2]

        control_points_helio_plot = control_points_helio

        for i in range(4):
            control_points_helio_plot[i, :, :, 0] = control_points_helio_plot[i, :, :, 0] + translation_vector_helio[i, 0]  # Apply translation in x
            control_points_helio_plot[i, :, :, 1] = control_points_helio_plot[i, :, :, 1] + translation_vector_helio[i, 1]  # Apply translation in y
            control_points_helio_plot[i, :, :, 2] = control_points_helio_plot[i, :, :, 2] + translation_vector_helio[i, 2]  # Apply translation in z


        for i in range(4):
            surface.facet_list[i].control_points = control_points_helio[i]
            surface.facet_list[i].canting_e = canting_e_helio[i]
            surface.facet_list[i].canting_n = canting_n_helio[i]
            surface.facet_list[i].translation_vector = torch.zeros_like(translation_vector_helio[i])

        surface = Surface(surface)

        all_surface_points[j], all_surface_normals[j] = (tensor.reshape(-1, 4) for tensor in
                                                                             surface.get_surface_points_and_normals(
                                                                                 device=device))


    new_scenario.heliostat_field.number_of_heliostats = num_helios
    new_scenario.heliostat_field.all_heliostat_positions = all_positions_tensor
    new_scenario.heliostat_field.all_aim_points = aim_point
    new_scenario.heliostat_field.all_aim_points = new_scenario.heliostat_field.all_aim_points.repeat(num_helios, 1)
    new_scenario.heliostat_field.all_surface_points = all_surface_points.to(device=device)
    new_scenario.heliostat_field.all_surface_normals = all_surface_normals.to(device=device)
    if first_overwrite:
        new_scenario.heliostat_field.all_initial_orientations = new_scenario.heliostat_field.all_initial_orientations.repeat(num_helios, 1)
        new_scenario.heliostat_field.all_kinematic_deviation_parameters = new_scenario.heliostat_field.all_kinematic_deviation_parameters.repeat(num_helios, 1)
        new_scenario.heliostat_field.all_actuator_parameters = new_scenario.heliostat_field.all_actuator_parameters.repeat(num_helios, 1, 1)
    new_scenario.heliostat_field.all_aligned_heliostats = torch.tensor([0], device=device).repeat(num_helios)
    new_scenario.heliostat_field.all_preferred_reflection_directions = torch.zeros(num_helios, 4, device=device)
    new_scenario.heliostat_field.all_current_aligned_surface_points = torch.zeros(num_helios, 10000, 4, device = device)
    new_scenario.heliostat_field.all_current_aligned_surface_normals = torch.zeros(num_helios, 10000, 4, device = device)

    first_overwrite = False  # Prevent repeating

    return new_scenario, first_overwrite

def raytracing_parallel_same_sun_positions(scenario, sun_position, aim_point_area, num_heliostats, target_area, device):
    # Align the heliostat.

    incident_ray_direction = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - sun_position
    incident_ray_direction_norm = torch.nn.functional.normalize(incident_ray_direction, dim=0)

    scenario.heliostat_field.align_surfaces_with_incident_ray_direction(
        incident_ray_direction=incident_ray_direction_norm, device=device
    )

    # Define the raytracer.
    raytracer = HeliostatRayTracer(
        scenario=scenario,
        batch_size=num_heliostats,
        world_size=1,
    bitmap_resolution_e=64,
    bitmap_resolution_u=64
    )

    # Perform heliostat-based raytracing.
    images = raytracer.trace_rays(
        incident_ray_direction=incident_ray_direction, target_area=target_area, device=device
    )

    return images

if __name__ == "__main__":

    scenario_name = r".../new_artist_scenario.h5"
    # Load a scenario.
    with h5py.File(scenario_name, "r") as f:
        scenario, prototype_surface = Scenario.load_scenario_from_hdf5(scenario_file=f, control_points_available=True, device=device)

    power_plant_position=torch.tensor([50.91342112259258, 6.387824755874856, 87.0], device=device) #Info from paint

    scenario.light_sources.light_source_list[0].number_of_rays = 400

    aim_point_area = "receiver"
    aim_point_receiver = scenario.target_areas.target_area_list[1].center

    # Increase receiver size
    target_area = scenario.get_target_area(aim_point_area)


    data_path = r"/.../heliostat_properties"
    receiver_filtered_dict, receiver_helios, other_helios = classify_and_plot_heliostats(data_path, plot=False)

    # Path to your augmented surface directory
    aug_dir = r".../training_surfaces"

    heliostat_names = list(receiver_filtered_dict.keys())

    all_surfaces = []
    all_positions = []
    all_ideal_surfaces = []
    translation_vector_list = []
    canting_e_list = []
    canting_n_list = []
    chosen_helios_list = []
    images_list = []

    ideal_dir = ".../ideal_heliostats"
    ideal_surfaces = {}  # Dictionary to store {heliostat_name: surface_tensor}


    def load_ideal_surface(filename):
        if not filename.endswith("_ideal_nurbs.pt"):
            return None
        heliostat_name = filename.replace("_ideal_nurbs.pt", "")
        full_path = os.path.join(ideal_dir, filename)
        surface_tensor = torch.load(full_path, map_location=device)
        return heliostat_name, surface_tensor


    print("Loading ideal surfaces in parallel...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(load_ideal_surface, os.listdir(ideal_dir)))

    for res in results:
        if res:
            heliostat_name, tensor = res
            ideal_surfaces[heliostat_name] = tensor

    print(f"✅ Loaded {len(ideal_surfaces)} ideal surfaces.")


    lock = Lock()
    max_workers = 8


    def random_point_on_shell_with_bounds(receiver_position, min_dist, max_dist, device,
                                          east_bounds=(-50, 50), north_min=30.0, up_max=130.0):
        """
        Keeps trying until it finds a point that:
          - is between min_dist and max_dist from receiver_position (in 3D space),
          - has east within east_bounds,
          - has north > north_min,
          - has up < up_max.
        """
        receiver_position = torch.tensor(receiver_position, dtype=torch.float32, device=device)

        while True:
            direction = torch.randn(3, device=device)
            direction = direction / torch.norm(direction)

            dist = random.uniform(min_dist, max_dist)
            candidate = receiver_position + dist * direction

            east, north, up = candidate[0].item(), candidate[1].item(), candidate[2].item()

            if (east_bounds[0] <= east <= east_bounds[1]) and (north > north_min) and (up < up_max):
                return candidate


    def random_point_on_plane_with_bounds(receiver_position, min_dist, max_dist, device,
                                          east_bounds=(-50, 50), north_min=5):
        """
        Samples a random point on the same horizontal plane as the receiver (z=0),
        while ensuring:
          - Distance from receiver (in XY plane) is between min_dist and max_dist,
          - East (x) within east_bounds,
          - North (y) within north_bounds.
        """
        receiver_position = torch.tensor(receiver_position, dtype=torch.float32, device=device)

        while True:
            # Sample random angle and distance in XY-plane
            angle = torch.tensor(random.uniform(0, 2 * torch.pi), device=device)
            dist = torch.tensor(random.uniform(min_dist, max_dist), device=device)

            # Compute offset in XY-plane
            offset_x = dist * torch.cos(angle)
            offset_y = dist * torch.sin(angle)

            candidate = torch.tensor([
                receiver_position[0] + offset_x,
                receiver_position[1] + offset_y,
                0  # Force z = 0 plane
            ], device=device)

            east, north = candidate[0].item(), candidate[1].item()
            if east_bounds[0] <= east <= east_bounds[1] and north_min <= north:
                return candidate


    def load_and_assign(i, min_dist=10.0, max_dist=30.0, overwrite_position=True):

        # Only allow the first 10 surfaces  #todo: just for testing!
        #if i > 10:
        #    return None

        filename = f"test_TEST{i}.pt"
        full_path = os.path.join(aug_dir, filename)

        if not os.path.exists(full_path):
            return None  # Skip missing files

        surface_tensor = torch.load(full_path, map_location=device)

        chosen_helio = random.choice(heliostat_names)

        receiver_position = torch.tensor([0.038603, -0.50296, 55.227], device=device)

        if overwrite_position:
            new_position = random_point_on_plane_with_bounds(
                receiver_position, 5.0, 15.0, device,
                east_bounds=(-50, 50), north_min=5)

        else:
            original_position = receiver_filtered_dict[chosen_helio]["position"]
            new_position = torch.tensor(original_position, dtype=torch.float32, device=device)

        position_4d = convert_3d_point_to_4d_format(new_position, device=device)

        ideal_surface_tensor = ideal_surfaces[chosen_helio]
        heliostat_tvec_info = receiver_filtered_dict[chosen_helio]["facets"]

        trans_vecs, cant_e, cant_n = [], [], []
        for facet in heliostat_tvec_info:
            trans_vecs.append(torch.tensor(facet["translation_vector"], dtype=torch.float32, device=device))
            cant_e.append(torch.tensor(facet["canting_e"], dtype=torch.float32, device=device))
            cant_n.append(torch.tensor(facet["canting_n"], dtype=torch.float32, device=device))

        return {
            "surface": surface_tensor,
            "position_4d": position_4d,
            "ideal_surface": ideal_surface_tensor,
            "chosen_helio": chosen_helio,
            "trans_vecs": trans_vecs,
            "canting_e": cant_e,
            "canting_n": cant_n
        }


    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(load_and_assign, i, min_dist=5, max_dist= 15, overwrite_position = True) for i in range(1, 60)]

        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue  # Skip if file was missing

            all_surfaces.append(result["surface"])
            all_positions.append(result["position_4d"])
            all_ideal_surfaces.append(result["ideal_surface"])
            chosen_helios_list.append(result["chosen_helio"])
            translation_vector_list.extend(result["trans_vecs"])
            canting_e_list.extend(result["canting_e"])
            canting_n_list.extend(result["canting_n"])

    print("Finished parallel loading.")

    # Stack all into tensors
    aug_surfaces_tensor = torch.stack(all_surfaces)  # Shape: (100000, 4, 8, 8, 3)
    all_positions_tensor = torch.stack(all_positions)  # Shape: (100000, 4)
    all_ideal_surfaces_tensor = torch.stack(all_ideal_surfaces) # Shape: (100000, 4, 8, 8, 3)

    translation_vector_tensor = torch.stack(translation_vector_list)  # Shape: (num_heliostats * 4, 3)
    translation_vector_tensor = convert_3d_direction_to_4d_format(translation_vector_tensor, device=device)
    canting_e_tensor = torch.stack(canting_e_list)  # Shape: (num_heliostats * 4, 3)
    canting_e_tensor = convert_3d_direction_to_4d_format(canting_e_tensor, device=device)
    canting_n_tensor = torch.stack(canting_n_list)  # Shape: (num_heliostats * 4, 3)
    canting_n_tensor = convert_3d_direction_to_4d_format(canting_n_tensor, device=device)

    translation_vector_tensor = translation_vector_tensor.reshape(len(aug_surfaces_tensor), 4, 4)
    canting_e_tensor = canting_e_tensor.reshape(len(aug_surfaces_tensor), 4, 4)
    canting_n_tensor = canting_n_tensor.reshape(len(aug_surfaces_tensor), 4, 4)

    chunk_size = 8
    num_chunks = aug_surfaces_tensor.shape[0] // chunk_size

    first_overwrite = True
    output_base_dir = r".../flux_images"
    os.makedirs(output_base_dir, exist_ok=True)

    for i in range(num_chunks):

        # Chunk slicing
        aug_surfaces_chunk = aug_surfaces_tensor[i * chunk_size:(i + 1) * chunk_size]
        positions_chunk = all_positions_tensor[i * chunk_size:(i + 1) * chunk_size]
        ideal_surfaces_chunk = all_ideal_surfaces_tensor[i * chunk_size:(i + 1) * chunk_size]
        canting_e_chunk = canting_e_tensor[i * chunk_size:(i + 1) * chunk_size]
        canting_n_chunk = canting_n_tensor[i * chunk_size:(i + 1) * chunk_size]
        translation_vector_chunk = translation_vector_tensor[i * chunk_size:(i + 1) * chunk_size]

        # Overwrite scenario
        new_scenario, first_overwrite = overwrite_scenario_xyz_surface(
            aim_point_receiver, positions_chunk, aug_surfaces_chunk, ideal_surfaces_chunk, scenario,
            prototype_surface, translation_vector_chunk, canting_e_chunk, canting_n_chunk, first_overwrite, device
        )

        num_heliostats = aug_surfaces_chunk.shape[0]

        # Get sun positions for this chunk
        sun_vecs_list, extras_list, ae_list = random_sun_positions_whole_year_new_positions(8, device)
        sun_positions = torch.stack(sun_vecs_list)  # (8, 4)

        # Allocate tensor to hold all images (helios, sun_positions, H, W)
        images_tensor = torch.zeros((num_heliostats, 8, 64, 64), dtype=torch.float32)

        # Run raytracing for each sun position
        for j, sun_position in enumerate(sun_positions):
            with torch.no_grad():
                images = raytracing_parallel_same_sun_positions(
                    new_scenario, sun_position, aim_point_area, num_heliostats, target_area,
                    device=device
                )
            images_tensor[:, j] = images

        # Collect metadata
        metadata_list = []
        for h_idx in range(num_heliostats):
            metadata_list.append({
                "augmented_surface_file": f"augmented_surface_AUG{i * chunk_size + h_idx + 1}.pt",
                "heliostat_name": chosen_helios_list[i * chunk_size + h_idx],
                "position_enu": all_positions_tensor[i * chunk_size + h_idx].tolist()
            })

        # Save outputs
        chunk_id = f"{i + 1:03d}"

        torch.save(images_tensor, os.path.join(output_base_dir, f"images_chunk_{chunk_id}.pt"))
        torch.save(sun_positions, os.path.join(output_base_dir, f"sun_vectors_chunk_{chunk_id}.pt"))
        with open(os.path.join(output_base_dir, f"metadata_chunk_{chunk_id}.json"), "w") as f:
            json.dump(metadata_list, f, indent=2)

        print(f"Saved chunk {chunk_id}")


        # Free up memory
        del images_tensor, images, new_scenario
        torch.cuda.empty_cache()