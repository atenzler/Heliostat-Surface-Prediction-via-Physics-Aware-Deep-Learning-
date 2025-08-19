"""
heliostat_positions_utils.py

Utility functions for loading and generating heliostat positions.

Functions:
- give_helPos_on_field_list(): return a list of heliostat positions
  (synthetic grid, or from deflectometry list).
- give_helPos_larger_grid(): generate a larger synthetic grid of positions
  in front of the tower.
- give_defl_list(): load real heliostat positions with deflectometry data
  and plot them.

Note:
- Paths are currently hardcoded and should be adapted or replaced with config.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def list_folders_in_directory(directory_path: str) -> list:
    # List all entries in the directory and filter to only include folders
    folder_list = [entry for entry in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, entry))]
    return folder_list


def give_defl_list():
    hellist = list_folders_in_directory(r"defl_data")
    posdir = r'...\heliostat_position_dictionary.npy'
    helPos_dic = np.load(posdir, allow_pickle=True).item()

    helios_with_defl_dict = {}

    for hel_name in hellist:
        if hel_name in helPos_dic:
            # Assign the position from helPos_dic to the corresponding heliostat in the dictionary
            helios_with_defl_dict[hel_name] = helPos_dic[hel_name]

    # Extract East (x) and North (y) coordinates
    east_coords = []
    north_coords = []

    # Loop through the dictionary and extract the first two dimensions (East and North)
    for hel_name, position in helios_with_defl_dict.items():
        east_coords.append(position[0])  # Assuming East is the first element in the position
        north_coords.append(position[1])  # Assuming North is the second element in the position

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(east_coords, north_coords, c='blue', edgecolors='black', alpha=0.7, label="Heliostats")

    # Label the axes and add a title
    plt.xlabel("East / m", fontsize=14)
    plt.ylabel("North / m", fontsize=14)
    plt.title("Heliostat Positions (East vs North)", fontsize=16, fontweight='bold')

    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="best")

    # Display the plot
    plt.show()


    return helios_with_defl_dict
