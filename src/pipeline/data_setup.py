"""
data_setup.py

Functions to set up datasets, scenarios, and model initialization
for DeepLARTS training and evaluation.

Includes:
- Surface loading utilities (augmented & ideal)
- Scenario + prototype surface loading (from ARTIST.h5)
- Model builders (with/without canting inputs)

Notes:
- File paths should be adapted to your local cluster/data layout.
- The functions here are imported into `run_experiment.py`
  to set up training reproducibly.

"""

import copy
import json
from weakref import WeakValueDictionary
import h5py

from func import *
from impl import *
from dataset import HeliostatChunkDataset, HeliostatTestDataset
from src.model import my_deepLarts
from artist.util.scenario import Scenario
from my_cfg import _C as _C_EIGHT
from src.model.network_parameters import (
    architecture_args, convolution_encoder_args, transformer_fusion_encoder_args,
    transformer_flux_encoder_args, styleGAN_args, training_args, data_args
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_augmented_surfaces(directory, device, limit=100000):
    augmented_surfaces = {}
    for i in range(1, limit + 1):
        file_name = f"augmented_surface_AUG{i}.pt"
        path = os.path.join(directory, file_name)
        if os.path.exists(path):
            tensor = torch.load(path, map_location=device)
            augmented_surfaces[f"AUG{i}"] = tensor.to(device)
    return augmented_surfaces

def load_augmented_surfaces_by_keys(directory, device, keys):
    from pathlib import Path
    directory = Path(directory)
    augmented_surfaces = {}
    for key in keys:
        filename = f"augmented_surface_{key}.pt"
        filepath = directory / filename
        if filepath.exists():
            augmented_surfaces[key] = torch.load(filepath).to(device)
        else:
            print(f"⚠️ File not found: {filename}")
    return augmented_surfaces

def load_ideal_surfaces(ideal_dir, device):
    ideal_surfaces = {}
    for filename in os.listdir(ideal_dir):
        if filename.endswith("_ideal_nurbs.pt"):
            heliostat_name = filename.replace("_ideal_nurbs.pt", "")
            surface_tensor = torch.load(os.path.join(ideal_dir, filename), map_location=device)
            ideal_surfaces[heliostat_name] = surface_tensor.to(device)
    return ideal_surfaces


def setup_data_and_scenario(num_inputs, keys, batch_size=8, use_canting_inputs=False, chunk_size=8, close_to_receiver=False):
    base_dir = ".../data/processed"
    ideal_dir = os.path.join(base_dir, "ideal_heliostats")
    scenario_path = os.path.join(base_dir, "new_artist_scenario.h5")
    json_path = os.path.join(base_dir, "all_heliostat_surfaces.json")

    augmented_surface_dir = os.path.join(base_dir, "training_surfaces")

    def get_aug_surface_loader(base_dir):
        def load_surface(key):
            path = os.path.join(base_dir, f"augmented_surface_{key}.pt")
            if os.path.exists(path):
                return torch.load(path, map_location="cpu")  # Use .to(device) later
            else:
                raise FileNotFoundError(f"Surface not found: {path}")

        return load_surface

    def get_ideal_surface_loader(base_dir):
        def load_surface(key):
            path = os.path.join(base_dir, f"{key}_ideal_nurbs.pt")
            if os.path.exists(path):
                return torch.load(path, map_location="cpu")  # Call .to(device) later
            else:
                raise FileNotFoundError(f"Ideal surface not found: {path}")

        return load_surface

    def get_ideal_surface_loader_lazy(base_dir):
        _cache = WeakValueDictionary()

        def load_surface(key):
            if key not in _cache:
                path = os.path.join(base_dir, f"{key}_ideal_nurbs.pt")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Surface not found: {path}")
                surface = torch.load(path, map_location="cpu")
                _cache[key] = surface
            return _cache[key]

        return load_surface

    get_aug_surface = get_aug_surface_loader(augmented_surface_dir)
    get_ideal_surface = get_ideal_surface_loader_lazy(ideal_dir)

    # Load static control point grid
    with open(json_path, "r") as f:
        data = json.load(f)
    reference_heliostat = "AA23"
    ideal_grid = torch.tensor(data[reference_heliostat]["facet_1"]["control_points"], device=device)
    ideal_grid_xy = ideal_grid[:, :, :2].unsqueeze(0).repeat(4, 1, 1, 1)

    translation_vector = torch.tensor([
        data[reference_heliostat][f"facet_{i}"]["translation_vector"][:3]
        for i in range(1, 5)
    ], dtype=torch.float32, device=device)

    # Load scenario
    with h5py.File(scenario_path, "r") as f:
        scenario, prototype_surface = Scenario.load_scenario_from_hdf5(
            scenario_file=f, control_points_available=True, device=device
        )

    aim_point_receiver = scenario.target_areas.target_area_list[1].center
    aim_point_area = "receiver"

    # Reduce rays to speed up training
    light_source = scenario.light_sources.light_source_list[0]
    light_source.number_of_rays = 400
    scenario.light_sources.light_source_list[0] = light_source
    if batch_size == 8:
        if use_canting_inputs:
            canting_path = os.path.join(base_dir, "heliostat_cant_tra_vec_pos_filtered.json")
            dataset = HeliostatChunkDataset(os.path.join(base_dir, "training_images"),
                                use_canting_inputs=True, canting_path=canting_path, limit=num_inputs,chunk_size=chunk_size)
        else:
            dataset = HeliostatChunkDataset(os.path.join(base_dir, "training_images"),
                                            limit=num_inputs, chunk_size=chunk_size)

        test_set = HeliostatTestDataset(os.path.join(base_dir, r"test_images"))
        valid_set = None



    return dataset, valid_set, test_set, copy.deepcopy(scenario), prototype_surface, get_ideal_surface, \
        translation_vector, ideal_grid_xy, aim_point_area, aim_point_receiver, get_aug_surface


def build_model(use_canting_inputs=False):
    architecture_args["use_canting_vecs"] = use_canting_inputs
    cfg = _C_EIGHT.clone()
    model = my_deepLarts.init_deepLarts(
        cfg,
        new_deepLarts=True,
        name_deepLarts="test_standard",
        load_from_deepLarts=None,
        architecture_args=architecture_args,
        convolution_encoder_args=convolution_encoder_args,
        transformer_fusion_encoder_args=transformer_fusion_encoder_args,
        transformer_flux_encoder_args=transformer_flux_encoder_args,
        styleGAN_args=styleGAN_args,
        data_args=data_args,
        training_args=training_args,
        device="cpu",
        timestamp="optuna",
        cluster=False,
        rank=-0
    )
    return model






