"""
compute_nurbs_without_canting.py

Removes canting effects from heliostat NURBS surfaces by comparing
measured control points against an ideal surface.

Workflow:
- Load ideal XY grid from a reference JSON.
- Load facet canting and translation data from JSON.
- Generate ideal Juelich heliostat surfaces with SurfaceConverter.
- Fit NURBS surfaces to the ideal geometry.
- Compare with measured NURBS control points.
- Compute displacement tensors (Z-direction).
- Save results as JSON.

Outputs:
- all_surfaces_without_canting.json
"""


import json
import pathlib
import matplotlib.pyplot as plt
import torch

from artist.util import config_dictionary
from artist.util.configuration_classes import FacetConfig
from artist.util.surface_converter import SurfaceConverter
from artist.util.utils import convert_3d_point_to_4d_format, convert_3d_direction_to_4d_format

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_surface_plot(surface_points: torch.Tensor, title: str, filename: str):
    """
    Plot and save a set of 3D points as a scatter plot.

    Parameters
    ----------
    surface_points : torch.Tensor
        Tensor of shape (N, 4); only the first three dimensions are plotted.
    title : str
        Title for the plot.
    filename : str
        File path where the plot will be saved.
    """
    pts = surface_points[:, :3].detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
    ax.set_title(title)
    plt.savefig(filename)
    plt.close(fig)


def get_or_create_facet_list(cache_file_path: pathlib.Path) -> list[FacetConfig]:
    """
    Retrieves the facet list from a cache file if it exists. Otherwise, generates the facet list,
    saves it to the cache file, and returns the generated data.

    Parameters
    ----------
    cache_file_path : pathlib.Path
        File path where the facet list is cached.

    Returns
    -------
    any
        The facet list generated from the input files.
    """

    with torch.serialization.safe_globals([FacetConfig]):
        facet_list = torch.load(cache_file_path)
    print(f"Loading cached facet list from {cache_file_path}")

    torch.save(facet_list, cache_file_path)
    print(f"Facet list saved to {cache_file_path}")

    return facet_list


if __name__ == "__main__":

    # JSON file with constant xy grid
    with open(
            r".../all_heliostat_surfaces.json",
            "r") as f:
        data = json.load(f)

    ideal_grid = torch.tensor(data["AA23"]["facet_1"]["control_points"], device=device)
    ideal_grid_xy = ideal_grid[:, :, :2]
    ideal_grid_xy = ideal_grid_xy.unsqueeze(0).repeat(4, 1, 1, 1)

    nurbs_without_canting_data = {}

    with open(r".../heliostat_cant_tra_vec.json", "r") as f:
        heliostat_data = json.load(f)

    for heliostat_name, facets in heliostat_data.items():
        # Convert extracted data into tensors
        facet_translation_vectors = torch.stack(
            [torch.tensor(facet["translation_vector"] + [0], device=device) for facet in facets])
        canting_e = torch.stack([torch.tensor(facet["canting_e"] + [0], device=device) for facet in facets])
        canting_n = torch.stack([torch.tensor(facet["canting_n"] + [0], device=device) for facet in facets])

        surface_converter = SurfaceConverter(step_size=1, max_epoch=20000,
                                             conversion_method=config_dictionary.convert_nurbs_from_points)

        # Generate the ideal Juelich heliostat surface.
        ideal_surface_points, ideal_surface_normals = surface_converter.generate_ideal_juelich_heliostat_surface(
            cantings_e=canting_e,
            cantings_n=canting_n,
            facet_translation_vectors=facet_translation_vectors,
            number_of_surface_points=2000,
            device=device
        )# ideal_surface_points is returned as a tensor with shape (number_of_facets, points_per_facet, 4)

        cat_ideal_surface_for_plot = torch.cat(ideal_surface_points)

        ideal_surface_points = torch.stack(ideal_surface_points)
        ideal_surface_points = convert_3d_point_to_4d_format(ideal_surface_points, device=device)
        ideal_surface_normals = torch.stack(ideal_surface_normals)
        ideal_surface_normals = convert_3d_direction_to_4d_format(ideal_surface_normals, device=device)

        ideal_nurbs_tensors = []  # List to store control points of each facet



        for i in range(4):  # Assuming 4 facets
            ideal_nurbs_surface = surface_converter.fit_nurbs_surface(
                surface_points=ideal_surface_points[i],  # Select facet i
                surface_normals=ideal_surface_normals[i],  # Select normals for facet i
                conversion_method=config_dictionary.convert_nurbs_from_points,
                number_control_points_e=8,
                number_control_points_n=8,
                tolerance=3e-5,
                initial_learning_rate=1e-3,
                max_epoch=1000,
                degree_n= 1,
                degree_e= 1,
                optimize_only_z_cntrl_points = True,
                ideal_grid_xy = ideal_grid_xy[i],
                device=device
            )

            # Extract control points and move them to the desired device
            ideal_nurbs_tensor = ideal_nurbs_surface.control_points.to(device)
            ideal_nurbs_tensors.append(ideal_nurbs_tensor)  # Store in list

        # Stack into a single tensor of shape (4, 8, 8, 4)
        ideal_nurbs_tensor = torch.stack(ideal_nurbs_tensors)

        control_points_list = []
        for facet_name in ["facet_1", "facet_2", "facet_3", "facet_4"]:
            facet_control_points = torch.tensor(data[heliostat_name][facet_name]["control_points"], device=device)
            control_points_list.append(facet_control_points)

        nurbs_with_canting = torch.stack(control_points_list)
        facet_translation_vectors = facet_translation_vectors[:, :3]

        for i in range(4):
            nurbs_with_canting[i, :, :, 2] = nurbs_with_canting[i, :, :, 2] + facet_translation_vectors[i, 2]

        # 4. Compute displacement (only in Z direction)
        displacement_from_ideal_surface_z = nurbs_with_canting - ideal_nurbs_tensor
        displacement_from_ideal_surface_z[:, :, :, :2] = ideal_grid_xy

        # Convert displacement tensor to dictionary format
        nurbs_without_canting_data[heliostat_name] = {
            heliostat_name: {
                f"facet_{i + 1}": {
                    "facet_key": f"facet_{i + 1}",
                    "control_points": displacement_from_ideal_surface_z[i].detach().cpu().numpy().tolist()
                }
                for i in range(4)
            }
        }

    # Save the displacement data to a JSON file
    output_file = r".../all_surfaces_without_canting.json"
    with open(output_file, "w") as f:
        json.dump(nurbs_without_canting_data, f, indent=4)











