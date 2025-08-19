"""
classify_heliostats.py

Classifies heliostats into receiver-canting and other categories based on
their JSON property files. Optionally plots their positions in local ENU
coordinates.

Workflow:
- Reads all heliostat property JSON files in a given directory.
- Extracts canting type, facet info, and WGS84 position.
- Converts positions to local ENU coordinates relative to a reference
  power plant position.
- Returns three dictionaries:
    1) receiver_filtered_dict (receiver-canting heliostats with East > -70),
    2) receiver_dict (all receiver-canting heliostats),
    3) other_dict (all other canting heliostats).

Optionally, a 2D scatter plot of heliostat positions can be displayed.
"""

import os
import json
import matplotlib.pyplot as plt
import torch

from artist.util.utils import convert_wgs84_coordinates_to_local_enu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def classify_and_plot_heliostats(data_dir, plot):
    receiver_filtered_dict = {}
    receiver_dict = {}
    other_dict = {}

    power_plant_position = torch.tensor([50.91342112259258, 6.387824755874856, 87.0], device=device)

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        json_file = os.path.join(folder_path, "Properties", f"{folder}-heliostat-properties.json")
        if not os.path.isfile(json_file):
            print(f" Skipping {folder}: No JSON file found.")
            continue

        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            canting_type = data.get("facet_properties", {}).get("canting_type", "")
            facets = data.get("facet_properties", {}).get("facets", [])
            position = torch.tensor(data.get("heliostat_position", None), device=device)
            position = convert_wgs84_coordinates_to_local_enu(position, power_plant_position, device=device)

            if position is None:
                print(f" No position info for {folder}.")
                continue

            heliostat_info = {
                "position": position,
                "facets": facets
            }

            if canting_type.lower() == "receiver canting":
                receiver_dict[folder] = heliostat_info
                if position[0].item() > -70:
                    receiver_filtered_dict[folder] = heliostat_info
            else:
                other_dict[folder] = heliostat_info

        except Exception as e:
            print(f" Error reading {json_file}: {e}")
            continue

    if plot:
        fig, ax = plt.subplots(figsize=(10, 7))

        if receiver_filtered_dict:
            positions = [v["position"].cpu().numpy() for v in receiver_filtered_dict.values()]
            east, north = zip(*[(pos[0], pos[1]) for pos in positions])
            ax.scatter(east, north, c='darkgreen', label='Heliostat Position')

        ax.set_title("Filtered Heliostat Positions")
        ax.set_xlabel("East")
        ax.set_ylabel("North")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    return receiver_filtered_dict, receiver_dict, other_dict

if __name__ == "__main__":
    data_path = r"heliostat_properties"
    receiver_filtered, receiver_helios, other_helios = classify_and_plot_heliostats(data_path, plot = False)

    print("\nFiltered Receiver Canting Heliostats (East > -70):", receiver_filtered)
    print("number of filtered receiver canting heliostats:", len(receiver_filtered))
    print("Receiver Canting Heliostats:", receiver_helios)
    print("number of receiver canting heliostats:", len(receiver_helios))
    print("Other Canting Heliostats:", other_helios)
    print("number of Other canting heliostats:", len(other_helios))
