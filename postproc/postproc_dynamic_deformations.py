import numpy as np
import os
import h5py as h5
import utils
from typing import Tuple, List
import scipy.signal as scipy_signal

import utils.plotting as plot_utils
import utils.h5tools as h5_utils
import utils.helpers as utils

def postproc_gust_induced_dynamic_deformation(
    result_folder: str,
    case_name: str,
    half_wingspan: float,
    data_output_file: str
) -> float:
    """
    Processes dynamic gust-induced deformation from SHARPy output and writes to .dat files.

    Args:
        result_folder (str): Path to the main result folder.
        case_name (str): Name of the simulation case.
        half_wingspan (float): Wing half span used for normalization.
        data_output_file (str): Format string for output text files.

    Returns:
        float: Time step used in the simulation.
    """
    dimensions, num_dimensions = utils.get_coordinate_dimensions()
    file = os.path.join(result_folder, case_name,case_name, 'savedata', case_name + '.data.h5')
    n_tsteps = h5_utils.get_num_timesteps(file)
    dt = h5_utils.get_time_step(file)
    num_nodes = h5_utils.get_number_of_structural_nodes(file)
    num_chordwise_nodes = h5_utils.get_number_of_chordwise_aero_nodes(file)
    structural_deformation = np.zeros((n_tsteps,num_nodes,num_dimensions))
   
    gust_quarter_chord = np.zeros((n_tsteps,num_dimensions))
    for its in range(n_tsteps):
        structural_deformation[its, ::] = h5_utils.read_structural_deformation(file, half_wingspan, h5_utils.get_timestep_str(its))
        gust_quarter_chord[its, :] = read_gust_quarter_chord(file, h5_utils.get_timestep_str(its), num_chordwise_nodes//4)
    for idim in range(num_dimensions):
        np.savetxt(data_output_file.format('structural_deformation', dimensions[idim]),structural_deformation[:, :, idim])
    np.savetxt(data_output_file.format('gust', 'quarter_chord'),gust_quarter_chord)
   
    return dt



def get_deformation_and_gust_period_for_position(
    spanwise_coords: np.ndarray,
    deformation: np.ndarray,
    position: float,
    gust_quarter_chord: np.ndarray,
    dt: float = 0.0,
    max_samples: int = 4000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Extracts the deformation time history and the gust velocity for one oscillation period
    at a specific spanwise position.

    Args:
        spanwise_coords (np.ndarray): 2D array of spanwise coordinates.
        deformation (np.ndarray): 2D array of node deformations (time x spanwise position).
        position (float): Normalized spanwise location (e.g., 0.9 for 90% span).
        gust_quarter_chord (np.ndarray): Time history of gust velocity vectors at quarter-chord.
        dt (float, optional): Time step (for optional debug output). Defaults to 0.0.
        max_samples (int, optional): Max time samples to consider. Defaults to 4000.

    Returns:
        Tuple:
            - np.ndarray: Full deformation history at selected spanwise position.
            - np.ndarray: Deformation during one oscillation period.
            - np.ndarray: Gust velocity during one oscillation period.
            - dict: Metrics including amplitude, mean, and phase shift.
    """
    # Locate closest spanwise index
    idx_spanwise_position = utils.find_index_of_closest_entry(spanwise_coords[0, :], position) 
    deformation_at_position = deformation[:max_samples, idx_spanwise_position]

    # Flip if necessary to keep values positive
    flipped = False
    if np.mean(deformation_at_position) < 0:
        deformation_at_position *= -1
        flipped = True

    deformation_at_position *= 100  # express in percent of half-span

    # Find local maxima
    local_maxima = scipy_signal.argrelextrema(deformation_at_position, np.greater)[0]
    if len(local_maxima) < 3:
        raise ValueError("Not enough oscillations detected to extract a period.")
        # TODO also check for convergence

    # Identify one period (last full cycle)
    i0, i1 = local_maxima[-3], local_maxima[-2]
    period_of_oscillation = deformation_at_position[i0:i1]
    period_of_gust_velocity = gust_quarter_chord[i0:i1]

    if flipped:
        period_of_gust_velocity *= -1

    phase_shift, idx_roll = compute_phase_shift(period_of_gust_velocity)
    idx_roll += int(3 * len(period_of_gust_velocity) // 4)

    period_of_oscillation = np.roll(period_of_oscillation, -idx_roll)
    period_of_gust_velocity = np.roll(period_of_gust_velocity, -idx_roll)

    max_val = float(np.max(period_of_oscillation))
    min_val = float(np.min(period_of_oscillation))

    metrics = {
        "amplitude": (max_val - min_val) / 2,
        "oscillation_mean": (max_val + min_val) / 2,
        "phase_shift": float(phase_shift)
    }

    return deformation_at_position, period_of_oscillation,period_of_gust_velocity,metrics 


def extract_converged_oscillation_period_from_time_history(
    data_output_file: str,
    vertical: bool,
    dt: float,
    icase: int,
    wing_halfspan: float,
    output_folder: str,
    case_name: str,
    experimental_data: str,
    spanwise_positions: List[float] = [0.9],
    plot: bool = True
) -> None:
    """
    Extracts one period of converged structural oscillation for each spanwise position
    and saves the deformation/gust velocity phase and amplitude metrics.

    Args:
        data_output_file (str): Format string for the output data files.
        vertical (bool): Whether the deformation is vertical or lateral.
        dt (float): Time step size.
        icase (int): Experimental case index (used for reference comparison).
        wing_halfspan (float): Wing half span for normalizing.
        output_folder (str): Path to save output JSON file.
        case_name (str): Name of the simulation case.
        spanwise_positions (List[float], optional): Normalized y/b locations. Defaults to [0.9].
        plot (bool, optional): Whether to show time history and period plots. Defaults to True.
    """
    spanwise_coords, deformation, gust_quarter_chord = load_time_histories(vertical, data_output_file)
    metrics_all_positions = {}

    for position in spanwise_positions:
        deformation_at_position, period_of_oscillation, period_of_gust_velocity, metrics = get_deformation_and_gust_period_for_position(
            spanwise_coords, deformation, position, gust_quarter_chord, dt=dt
        )

        metrics_all_positions[str(position)] = metrics

        if plot:
            plot_utils.plot_time_history_of_structural_deformation_and_gust_velocity(
                position, deformation_at_position, dt, gust_quarter_chord
            )
            plot_utils.plot_period_of_structural_deformation_and_gust(
                period_of_oscillation, period_of_gust_velocity, position, wing_halfspan, icase, experimental_data
            )

    json_path = os.path.join(output_folder, case_name + '_oscillation_metrics.json')
    utils.export_dict_to_json_file(metrics_all_positions, json_path)
   

def load_time_histories(vertical: bool, data_output_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads structural deformation and gust data from text files.

    Args:
        vertical (bool): Whether to extract vertical or lateral deformation.
        data_output_file (str): Format string for the output data files.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Spanwise coordinates
            - Deformation time history
            - Gust velocity history at quarter chord
    """
    dimensions, _ = utils.get_coordinate_dimensions()
    idx_spanwise_coord, idx_displacement_coord = utils.get_spanwise_and_displacement_coordinate(vertical)
    gust_quarter_chord = np.loadtxt(data_output_file.format('gust','quarter_chord'))[:,idx_displacement_coord]
    deformation = np.loadtxt(data_output_file.format('structural_deformation',dimensions[idx_displacement_coord]))
    spanwise_coords =np.loadtxt(data_output_file.format('structural_deformation',dimensions[idx_spanwise_coord]))
    return spanwise_coords, deformation, gust_quarter_chord


def read_gust_quarter_chord(file: str, ts_str: str, chordwise_node: int,
                             spanwise_node: int = 0, isurface: int = 0) -> np.ndarray:
    """
    Reads the gust velocity at the quarter-chord point from the SHARPy results.

    Args:
        file (str): Path to the HDF5 result file.
        ts_str (str): Timestep string formatted as '00000'.
        chordwise_node (int): Index of the chordwise node.
        spanwise_node (int, optional): Index of the spanwise node. Defaults to 0.
        isurface (int, optional): Index of the aerodynamic surface. Defaults to 0.

    Returns:
        np.ndarray: Gust velocity vector (3D).
    """
    with h5.File(file, "r") as f: 
        gust_data = np.transpose(
            np.array(f['data']['aero']['timestep_info'][ts_str]['u_ext']['_as_array'])[isurface, :, chordwise_node, spanwise_node])

    return gust_data



def compute_phase_shift(period_of_gust_velocity: np.ndarray) -> tuple[float, int]:
    """
    Compute the phase shift of the gust velocity signal.

    Args:
        period_of_gust_velocity (np.ndarray): Gust velocity over one oscillation period.

    Returns:
        Tuple[float, int]: Phase shift in radians and index of the max gust velocity.
    """
    idx_max = np.argmax(period_of_gust_velocity)
    num_ts_per_period = np.shape(period_of_gust_velocity)
    phase_shift = 2 * np.pi * (idx_max)/num_ts_per_period
    return phase_shift, idx_max

if __name__ == '__main__':
    result_folder = '../output/'
    case_name = 'pazy_vertical_case_2_polars0_effcor_0dynamic_m8_not_symmetric' #'pazy_dynamic_alpha_{}_polars{}_effcor_{}'
    output_folder = os.path.join('../results/extracted_data/'+case_name)

    experimental_data = os.path.join(os.path.abspath(os.path.dirname(os.path.realpath(__file__))),
                                     '../experimental_data')
    icase = 2
    data_output_file = os.path.join(output_folder,'dynamic_{}_{}.dat')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    pazy_half_span = 0.55 
    vertical = True
    dt = postproc_gust_induced_dynamic_deformation(result_folder, case_name, pazy_half_span,  data_output_file)
    extract_converged_oscillation_period_from_time_history(data_output_file, vertical, dt, icase,pazy_half_span, output_folder, case_name, experimental_data)
    