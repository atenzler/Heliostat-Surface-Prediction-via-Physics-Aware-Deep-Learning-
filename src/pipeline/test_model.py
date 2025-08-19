"""
test_model.py

Evaluation script for trained DeepLARTS models.

Features:
- Loads ideal surface grids, translation vectors, and ARTIST scenario
- Loads test dataset (flux images, sun positions, heliostat positions)
- Reconstructs predicted surfaces from NN outputs
- Optionally loads:
  * Best surface model (evaluated on surface reconstruction)
  * Best train model (evaluated on flux accuracy)
- Evaluates predictions with `evaluate_model_on_test_set`
- Produces plots (surface reconstructions, heatmaps, metrics)

Usage:
- Run after training to evaluate checkpoints
- Works with both untrained (debug) and trained models
- Requires access to ARTIST scenarios and preprocessed test surfaces
- Integration with ARTIST raytracer
"""


from weakref import WeakValueDictionary
import os
import h5py
import torch
import time
import json
from collections import OrderedDict

from func import *
from impl import *
from dataset import HeliostatTestDataset
from src.model import my_deepLarts
from artist.util.scenario import Scenario
from my_cfg import _C
from src.model.network_parameters import architecture_args, convolution_encoder_args, transformer_fusion_encoder_args, transformer_flux_encoder_args, styleGAN_args, training_args, test_args, data_args

def load_ideal_surfaces(ideal_dir, device):
    ideal_surfaces = {}
    for filename in os.listdir(ideal_dir):
        if filename.endswith("_ideal_nurbs.pt"):
            heliostat_name = filename.replace("_ideal_nurbs.pt", "")
            surface_tensor = torch.load(os.path.join(ideal_dir, filename), map_location=device)
            ideal_surfaces[heliostat_name] = surface_tensor.to(device)
    return ideal_surfaces

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()

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

    ideal_dir = r".../ideal_heliostats"
    get_ideal_surface = get_ideal_surface_loader_lazy(ideal_dir)

    if device == 'cuda':
        torch.backends.cuda.max_split_size = 1024
    print(device)

    # Scenario nur einmal einlesen
    scenario_name = r"...\new_artist_scenario.h5"

    # JSON file with constant xy grid + translation vector
    with open(r"...\all_heliostat_surfaces.json", "r") as f:
        data = json.load(f)

    reference_heliostat = "AA23"  # or any heliostat you trust as representative

    # Get control point grid (XY only)
    ideal_grid = torch.tensor(data[reference_heliostat]["facet_1"]["control_points"], device=device)
    ideal_grid_xy = ideal_grid[:, :, :2]
    ideal_grid_xy = ideal_grid_xy.unsqueeze(0).repeat(4, 1, 1, 1)  # shape: [4, 8, 8, 2]

    # Get translation vector per facet
    translation_vector_list = []
    for i in range(1, 5):  # facet_1 to facet_4
        key = f"facet_{i}"
        full_vec = data[reference_heliostat][key]["translation_vector"]  # includes 4 elements
        translation_vector_list.append(full_vec[:3])  # take only x, y, z

    translation_vector = torch.tensor(translation_vector_list, dtype=torch.float32, device=device)


    # Load a scenario.
    with h5py.File(scenario_name, "r") as f:
        new_scenario, prototype_surface = Scenario.load_scenario_from_hdf5(scenario_file=f, control_points_available=True, device=device)


    augmented_surface_dir = r"data\processed\test_surfaces"
    def get_aug_surface_loader(base_dir):
        def load_surface(key):
            path = os.path.join(base_dir, f"test_{key}.pt")
            if os.path.exists(path):
                return torch.load(path, map_location="cpu")  # Use .to(device) later
            else:
                raise FileNotFoundError(f"Surface not found: {path}")
        return load_surface
    get_aug_surface = get_aug_surface_loader(augmented_surface_dir)


    def load_test_surfaces(directory, device, limit=63):
        test_surfaces = {}
        for i in range(1, limit + 1):
            file_name = f"test_TEST{i}.pt"
            path = os.path.join(directory, file_name)
            if os.path.exists(path):
                tensor = torch.load(path, map_location=device)
                test_surfaces[f"TEST{i}"] = tensor.to(device)
        return test_surfaces


    test_surface_dir =r"data\processed\test_surfaces"
    test_surface_dict =load_test_surfaces(test_surface_dir, device)

    light_source = new_scenario.light_sources.light_source_list[0]
    light_source.number_of_rays = 400
    new_scenario.light_sources.light_source_list[0] = light_source

    aim_point_area = "multi_focus_tower"
    aim_point_receiver = new_scenario.target_areas.target_area_list[0].center

    #todo: care, synthetic datasets were canted to receiver

    #canting_vecs = False
    use_canting_inputs = False
    architecture_args["use_canting_vecs"] = use_canting_inputs

    my_cfg = _C.clone()

    model = my_deepLarts.init_deepLarts(my_cfg,
                                        new_deepLarts=True,
                                        name_deepLarts="test",
                                        load_from_deepLarts=None,
                                        architecture_args=architecture_args,
                                        convolution_encoder_args=convolution_encoder_args,
                                        transformer_fusion_encoder_args=transformer_fusion_encoder_args,
                                        transformer_flux_encoder_args=transformer_flux_encoder_args,
                                        styleGAN_args=styleGAN_args,
                                        data_args=data_args,
                                        training_args=training_args,
                                        device="cpu",
                                        timestamp="2025_02_20",
                                        cluster=False,
                                        rank=-0)
    features = "not applicable"
    nbr = "not applicable"

    model.to(device)
    model.eval()

    test_data_dir = r"...\test_images"
    test_set = HeliostatTestDataset(test_data_dir)

    #todo: uncomment when using canitng inputs
    #test_data_dir = r"test_images"
    #canting_path = r".../heliostat_cant_tra_vec_pos_filtered.json"
    #test_set = HeliostatTestDataset(test_data_dir, use_canting_inputs=use_canting_inputs, canting_path=canting_path)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    predicted_surfaces = []

    with torch.no_grad():
        for idx, (inputs, _) in enumerate(test_loader):
            if idx >= 1:
                break

            if use_canting_inputs:
                sun_vecs, flux_img, helio_pos_norm, canting_vecs, _, aug_surface_key = inputs
                canting_vecs = canting_vecs.to(device)
            else:
                sun_vecs, flux_img, helio_pos_norm, _, aug_surface_key = inputs
                canting_vecs = None

            sun_vecs = sun_vecs.to(device)
            flux_img = flux_img.to(device)
            heli_pos = helio_pos_norm.to(device)
            key = aug_surface_key[0]

            z_pred = model(flux_img, sun_vecs, heli_pos, targetID=None, canting_vecs=canting_vecs)[0][0]  # [4,8,8]
            # Step 1: Reconstruct full 3D surface (X, Y from grid + Z from prediction)
            surface = torch.cat([ideal_grid_xy, z_pred.unsqueeze(-1)], dim=-1)  # [4, 8, 8, 3]

            # Step 2: Apply x/y translation (zero out Z)
            translated_facets = []
            for f in range(4):
                translation = translation_vector[f].clone()
                translation[2] = 0.0  # only apply x and y shift
                translated = surface[f] + translation.view(1, 1, 3)
                translated_facets.append(translated)

            pred_surface = torch.stack(translated_facets, dim=0)  # [4, 8, 8, 3]
            predicted_surfaces.append(pred_surface.cpu())

    fig = plt.figure(figsize=(5 * len(predicted_surfaces), 5))

    plot_first_prediction = False

    if plot_first_prediction:
        for idx, surface in enumerate(predicted_surfaces):
            ax = fig.add_subplot(1, len(predicted_surfaces), idx + 1, projection='3d')
            ax.set_title(f"Untrained Prediction #{idx + 1}", fontsize=10)

            for f in range(4):
                facet = surface[f]
                X, Y, Z = facet[..., 0], facet[..., 1], facet[..., 2]
                ax.plot_surface(X, Y, Z, alpha=0.8, edgecolor='k')

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-0.01, 0.01)
            ax.view_init(elev=30, azim=60)

        plt.tight_layout()
        plt.show()

        surface_ref_cpu = get_aug_surface(key)
        surface_ref = surface_ref_cpu.to(device)
        del surface_ref_cpu

        z_gt = surface_ref[..., 2] - surface_ref[..., 2].mean()

        # Simulate 5 predictions from an untrained model
        pred_surfaces_list = [z_pred.squeeze().cpu()]  # list of [4, 8, 8] tensors
        gt_surfaces_list = [z_gt.squeeze().cpu()]  # same size for comparison

        print_surface_heatmaps(pred_surfaces_list, gt_surfaces_list, epoch=0, tag="UntrainedSurfaceHeatmaps")

    surface_model = True

    if surface_model:
        #Best Surface Model
        print("Best Surface Model:")
        checkpoint_path = r"...\best_surface_model.pth" # Path to your uploaded file
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")


        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            new_key = k.replace("module.", "")  # strip 'module.' if it exists
            new_state_dict[new_key] = v

        # Load into model
        model.load_state_dict(new_state_dict)
        model.to(device)


        canting_path = r".../heliostat_cant_tra_vec_pos_filtered.json"
        test_set = HeliostatTestDataset(test_data_dir, use_canting_inputs=use_canting_inputs, canting_path=canting_path)

        criterion = 'MAE'
        batch_size = 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_dir = fr"...\surface_test_results_{timestamp}"


        """  Test  """
        evaluate_model_on_test_set(test_set, model, device, create_loss(criterion),
                        aim_point_receiver, aim_point_area, new_scenario, ideal_grid_xy, translation_vector,
                        prototype_surface, batch_size, get_ideal_surface=get_ideal_surface, get_aug_surface=get_aug_surface,
                         use_canting_inputs=use_canting_inputs, log_dir=log_dir)


    else:

        #Best train model
        print("Best Train Model:")
        checkpoint_path = r"...\best_train_model.pth"  # Path to your uploaded file
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            new_key = k.replace("module.", "")  # strip 'module.' if it exists
            new_state_dict[new_key] = v

        # Load into model
        model.load_state_dict(new_state_dict)
        model.to(device)


        canting_path = r".../heliostat_cant_tra_vec_pos_filtered.json"
        test_set = HeliostatTestDataset(test_data_dir, use_canting_inputs = use_canting_inputs, canting_path = canting_path)

        criterion = 'MAE'
        batch_size = 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_dir = fr"...\train_test_results_{timestamp}"

        """  Test  """
        evaluate_model_on_test_set(test_set, model, device, create_loss(criterion),
                                   aim_point_receiver, aim_point_area, new_scenario, ideal_grid_xy, translation_vector,
                                   prototype_surface, batch_size, get_ideal_surface=get_ideal_surface, get_aug_surface=get_aug_surface,
                                    use_canting_inputs=use_canting_inputs, log_dir=log_dir)

