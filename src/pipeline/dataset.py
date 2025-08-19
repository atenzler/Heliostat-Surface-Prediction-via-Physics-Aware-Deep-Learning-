"""
dataset.py

PyTorch Dataset classes for heliostat surface prediction.

Includes:
- normalize_flux_image(): Utility to normalize flux images
- HeliostatChunkDataset: Main training dataset (loads from chunked flux/sun/meta files)
- HeliostatTestDataset: Dataset wrapper for testing

Features:
- Caches chunk files for efficient access
- Normalizes flux images and heliostat positions
- Supports optional facet canting vector inputs
- Handles cluster-safe distributed logging

Notes:
- Expects `images_chunk*.pt`, `sun_vectors_chunk*.pt`, and `metadata_chunk*.json` in `data/processed/flux_images/`
- Metadata JSON must include keys:
  `position_enu`, `heliostat_name`, and surface file references
"""

import torch
from torch.utils.data import Dataset
import os
import json
import gc
import torch.distributed as dist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize_flux_image(tensor):
    """Normalize flux images assuming they are [0, 255] or [0,1] floats."""
    if tensor.max() > 1.5:  # Heuristic: If max > 1.5, assume in [0,255]
        tensor = tensor / 255.0
    return tensor


class HeliostatChunkDataset(Dataset):
    def __init__(self, data_dir, limit=None, filter_by_distance=False, use_canting_inputs=False, canting_path=None, chunk_size=8):
        self.data_dir = data_dir
        self.index_map = []
        rank = dist.get_rank() if dist.is_initialized() else 0
        self.use_canting_inputs = use_canting_inputs
        self.min_vals = torch.tensor([-58.0915, 29.2537, 1.5110], device="cpu")
        self.max_vals = torch.tensor([157.8804, 243.5274, 2.0130], device="cpu")
        self.receiver_pos = torch.tensor([3.8603e-02, -5.0296e-01, 5.5227e+01], device="cpu")
        self.canting_path = canting_path

        if self.use_canting_inputs:
            with open(self.canting_path, 'r') as f:
                self.canting_lookup = json.load(f)

        self.image_chunks = sorted([f for f in os.listdir(data_dir) if f.startswith("images_chunk")])
        self.sun_vector_chunks = sorted([f for f in os.listdir(data_dir) if f.startswith("sun_vectors_chunk")])
        self.metadata_files = sorted([f for f in os.listdir(data_dir) if f.startswith("metadata_chunk")])
        assert len(self.image_chunks) == len(self.sun_vector_chunks) == len(self.metadata_files)

        self.chunk_size = chunk_size  # number of samples per chunk
        num_chunks = len(self.image_chunks)
        if limit is not None:
            max_samples = limit
        else:
            max_samples = num_chunks * chunk_size

        total_loaded = 0
        for i in range(num_chunks):
            meta_path = os.path.join(data_dir, self.metadata_files[i])
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            for j, meta in enumerate(metadata):
                if total_loaded >= max_samples:
                    break

                position = torch.tensor(meta["position_enu"][:3])
                if filter_by_distance:
                    distance = torch.norm(position - self.receiver_pos)
                    if not (10.0 <= distance <= 25.0):
                        continue

                self.index_map.append({
                    "img_chunk": self.image_chunks[i],
                    "sun_chunk": self.sun_vector_chunks[i],
                    "meta": meta,
                    "index_in_chunk": j
                })
                total_loaded += 1

        # Caching the most recent chunk
        self._cached_img_path = None
        self._cached_sun_path = None
        self._cached_imgs = None
        self._cached_suns = None

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        rank = dist.get_rank() if dist.is_initialized() else 0

        item = self.index_map[idx]
        meta = item["meta"]
        img_path = os.path.join(self.data_dir, item["img_chunk"])
        sun_path = os.path.join(self.data_dir, item["sun_chunk"])
        j = item["index_in_chunk"]

        if self._cached_img_path != img_path:
            if self._cached_imgs is not None:
                del self._cached_imgs
                gc.collect()
            try:
                self._cached_imgs = torch.load(img_path, map_location='cpu')
            except PermissionError as e:
                print(f"[Rank {rank}] Warning: Could not read {img_path} due to {e}. Returning dummy data.")
                # Return a dummy tensor and z_control to avoid crashing the trial
                dummy_flux = torch.zeros((8, 64, 64), dtype=torch.float32)
                dummy_sun_vecs = torch.zeros((8, 3), dtype=torch.float32)
                return (dummy_sun_vecs, dummy_flux, torch.zeros(3), "", ""), torch.zeros(4, 8, 8, 3)
            self._cached_img_path = img_path

        if self._cached_sun_path != sun_path:
            if self._cached_suns is not None:
                del self._cached_suns
                gc.collect()
            self._cached_suns = torch.load(sun_path,  map_location='cpu')[:, :3]
            self._cached_suns = self._cached_suns / (self._cached_suns.norm(dim=1, keepdim=True) + 1e-8)
            self._cached_sun_path = sun_path

        flux = self._cached_imgs[j].detach()
        flux = flux / (flux.max() + 1e-8)
        sun_vecs = self._cached_suns.detach()

        pos = torch.tensor(meta["position_enu"][:3], device="cpu")
        helio_pos_norm = (pos - self.min_vals) / (self.max_vals - self.min_vals + 1e-8)
        z_control = torch.zeros(4, 8, 8, 3, device="cpu")
        heliostat_name = meta["heliostat_name"]
        aug_key = meta["augmented_surface_file"].replace(".pt", "").replace("augmented_surface_", "")


        if self.use_canting_inputs:
            canting_entry = self.canting_lookup.get(heliostat_name)
            if canting_entry is None:
                raise KeyError(f"Canting vector for heliostat '{heliostat_name}' not found in canting lookup.")

            canting_vec_list = []
            for facet in canting_entry["facets"]:
                canting_vec_list.extend(facet["canting_e"])
                canting_vec_list.extend(facet["canting_n"])

            canting_vecs = torch.tensor(canting_vec_list, dtype=torch.float32)  # shape: [24]
            return (sun_vecs, flux, helio_pos_norm, canting_vecs, heliostat_name, aug_key), z_control
        else:
            return (sun_vecs, flux, helio_pos_norm, heliostat_name, aug_key), z_control


class HeliostatTestDataset(Dataset):
    def __init__(self, data_dir, limit=None, filter_by_distance = False, run_local_test=False, chunk_size=8,
             use_canting_inputs=False, canting_path=None):
        self.data_dir = data_dir
        self.samples = []
        self.run_local_test = run_local_test
        self.use_canting_inputs = use_canting_inputs
        self.canting_path = canting_path

        if self.use_canting_inputs:
            if not os.path.exists(canting_path):
                raise FileNotFoundError(f"Canting path not found: {canting_path}")
            with open(self.canting_path, 'r') as f:
                self.canting_lookup = json.load(f)

        # Load these from saved file if needed
        min_vals = torch.tensor([-58.0915, 29.2537, 1.5110])
        max_vals = torch.tensor([157.8804, 243.5274, 2.0130])

        # Collect all chunks
        self.image_chunks = sorted([f for f in os.listdir(data_dir) if f.startswith("images_chunk")])
        self.sun_vector_chunks = sorted([f for f in os.listdir(data_dir) if f.startswith("sun_vectors_chunk")])
        self.metadata_files = sorted([f for f in os.listdir(data_dir) if f.startswith("metadata_chunk")])

        assert len(self.image_chunks) == len(self.sun_vector_chunks) == len(self.metadata_files), "Mismatch in chunks"

        # Figure out how many full chunks we need to reach 'limit'
        if limit is not None:
            self.chunk_size = chunk_size
            num_chunks_needed = (limit + chunk_size - 1) // chunk_size  # Ceiling division
        else:
            num_chunks_needed = len(self.image_chunks)

        for i in range(num_chunks_needed):
            images = torch.load(os.path.join(data_dir, self.image_chunks[i]))  # [N, 8, 64, 64]
            sun_vecs = torch.load(os.path.join(data_dir, self.sun_vector_chunks[i]))[:, :3]  # [8, 3]

            with open(os.path.join(data_dir, self.metadata_files[i]), "r") as f:
                metadata = json.load(f)

            # normalization of sun position
            sun_vecs = sun_vecs / (torch.norm(sun_vecs, dim=1, keepdim=True) + 1e-8)  # shape: [N, 3]

            for j, meta in enumerate(metadata):
                if limit is not None and len(self.samples) >= limit:
                    return  # stop loading once limit is reached

                position = torch.tensor(meta["position_enu"][:3])
                flux = images[j]
                flux = flux / (flux.max() + 1e-8)


                z_control = torch.zeros(4, 8, 8, 3)
                heliostat_name = meta["heliostat_name"]
                if run_local_test:
                    test_surface_key = meta["test_surface_file"].replace(".pt", "").replace("test_", "")
                    #todo: change back to augmented_surface_file if not working!
                else:
                    test_surface_key = meta["augmented_surface_file"].replace(".pt", "").replace("test_", "")
                    #todo: change back to augmented_surface_file if not working!
                # Normalize heliostat position
                helio_pos_norm = (position - min_vals) / (max_vals - min_vals + 1e-8)

                if self.use_canting_inputs:
                    canting_entry = self.canting_lookup.get(heliostat_name)
                    if canting_entry is None:
                        raise KeyError(f"Canting vector for heliostat '{heliostat_name}' not found in canting lookup.")

                    canting_vec_list = []
                    for facet in canting_entry["facets"]:
                        canting_vec_list.extend(facet["canting_e"])
                        canting_vec_list.extend(facet["canting_n"])
                    canting_vecs = torch.tensor(canting_vec_list, dtype=torch.float32)  # shape [24]

                    self.samples.append(
                        (sun_vecs, flux, helio_pos_norm, z_control, heliostat_name, test_surface_key, canting_vecs))
                else:
                    self.samples.append((sun_vecs, flux, helio_pos_norm, z_control, heliostat_name, test_surface_key))



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_canting_inputs:
            sun_vecs, flux_img, helio_pos, z_control, heliostat_name, aug_surface_key, canting_vecs = self.samples[idx]
            return (sun_vecs, flux_img, helio_pos, canting_vecs, heliostat_name, aug_surface_key), z_control
        else:
            sun_vecs, flux_img, helio_pos, z_control, heliostat_name, aug_surface_key = self.samples[idx]
            return (sun_vecs, flux_img, helio_pos, heliostat_name, aug_surface_key), z_control
