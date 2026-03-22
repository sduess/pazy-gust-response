import numpy as np
import utils.helpers as utils

def get_reference_deformation_static(icase: int, folder_experimental_data: str) -> np.ndarray:
    """
    Loads experimental static deformation data for a given case.

    Args:
        icase (int): Case index (1-based).

    Returns:
        np.ndarray: Experimental data as a 2D array [y, z].
    """
    filepath = f"{folder_experimental_data}/static_results/steady_deflection_case{icase}.csv"
    data_deformation = np.loadtxt(filepath, skiprows=1, delimiter=',')
    
    return data_deformation


def get_reference_deformation_dynamic(icase: int, spanwise_position: float, wing_halfspan: float, folder_experimental_data: str) -> np.ndarray:
    """
    Loads reference experimental deformation data for comparison.

    Args:
        icase (int): Case number (starting from 1).
        spanwise_position (float): Normalized y/b location.
        wing_halfspan (float): Wing half-span.

    Returns:
        np.ndarray: Time history of deformation at the specified position.
    """
    filepath_str = "{}/dynamic_results/{}_gust{}.txt"
    data_deformation = np.loadtxt(filepath_str.format(folder_experimental_data, 'deflections', icase), skiprows=1, delimiter=',')/wing_halfspan/1000*100
    spanwise_coords = np.loadtxt(filepath_str.format(folder_experimental_data, 'coordinates', icase), skiprows=1, delimiter=',')/wing_halfspan/1000
    idx_spanwise_position = utils.find_index_of_closest_entry(spanwise_coords[0, 1:], spanwise_position) + 1

    return data_deformation[:,idx_spanwise_position]