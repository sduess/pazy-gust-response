import numpy as np
import json

def get_spanwise_and_displacement_coordinate(vertical: bool) -> tuple[int, int]:
    """
    Determines coordinate indices based on vertical motion setting.

    Args:
        vertical (bool): True for vertical displacement, False for lateral.

    Returns:
        tuple[int, int]: Indices for spanwise and displacement coordinates.
    """
    idx_spanwise_coord = 2 if vertical else 1
    idx_displacement_coord = 1 if vertical else 2
    return idx_spanwise_coord, idx_displacement_coord

def get_coordinate_dimensions() -> tuple[list[str], int]:
    """
    Returns the coordinate labels and number of spatial dimensions.

    Returns:
        tuple[list[str], int]: List of coordinate names and number of dimensions (typically 3).
    """
    return ['x', 'y', 'z'], 3


def get_timestep_str(its: int) -> str:
    """
    Formats the timestep index into SHARPy's expected string format.

    Args:
        its (int): Timestep index.

    Returns:
        str: Zero-padded timestep string (e.g., '00003').
    """
    return f"{its:05d}"


def find_index_of_closest_entry(array_values: np.ndarray, target_value: float) -> int:
    """
    Find the index of the array entry closest to a target value.

    Args:
        array_values (np.ndarray): 1D array of values.
        target_value (float): Value to search for.

    Returns:
        int: Index of the closest entry.
    """
    return int(np.argmin(np.abs(array_values - target_value)))

def export_dict_to_json_file(results: dict, file: str) -> None:
    """
    Exports a dictionary to a JSON file.

    Args:
        results (dict): Dictionary to write.
        file (str): Path to the output JSON file.
    """
    with open(file, "w") as f:
        json.dump(results, f, indent=4)

        
def get_corner_points(zeta_surf: np.ndarray, num_points: int) -> np.ndarray:
    """
    Converts a reshaped 3D surface array (zeta) into a list of 3D point coordinates.

    Args:
        zeta_surf (np.ndarray): A reshaped SHARPy zeta surface array of shape (3, N).
        num_points (int): Number of points (N).

    Returns:
        np.ndarray: Array of shape (N, 3) where each row is a (x, y, z) point.
    """
    return np.transpose(zeta_surf.reshape(3, num_points))


def write_data_to_file(data: np.ndarray, file_path: str, init_file: bool = False) -> None:
    """
    Writes a NumPy 2D array to a CSV file. Optionally overwrites the file.

    Args:
        data (np.ndarray): Array to write. Will be transposed before saving (shape [N, M] → [M, N]).
        file_path (str): Path to the output CSV file.
        init_file (bool, optional): If True, overwrite file. If False, append to it. Defaults to False.
    
    Notes:
        The data is transposed so that time steps go into columns, which is useful for time-history plotting.
    """
    file_mode = "w" if init_file else "a"

    with open(file_path, file_mode) as f:
        data_t = np.transpose(data)

        np.savetxt(f, data_t, delimiter=',')