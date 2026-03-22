
import numpy as np
import h5py as h5
from sharpy.utils.datastructures import AeroTimeStepInfo

def read_structural_deformation(file: str, half_wingspan: float, ts_str: str) -> np.ndarray:
    """
    Reads and normalizes structural deformation data from SHARPy results.

    Args:
        file (str): Path to the HDF5 result file.
        half_wingspan (float): Wing half-span for normalization.
        ts_str (str): Timestep string.

    Returns:
        np.ndarray: Normalized node positions (Nx3).
    """
    with h5.File(file, "r") as f: 
        wing_deformation = np.array(f['data']['structure']['timestep_info'][ts_str]['pos'])
        wing_deformation /= half_wingspan
    return wing_deformation

def get_num_timesteps(file: str) -> int:
    """
    Reads the number of time steps from the SHARPy HDF5 results file.

    Args:
        file (str): Path to the HDF5 result file.

    Returns:
        int: Number of time steps.
    """
    with h5.File(file, "r") as f:
        return int(np.array(f['data']['ts']))
        
def get_time_step(file: str) -> float:
    """
    Extracts the simulation time step size from the HDF5 file.

    Args:
        file (str): Path to the HDF5 result file.

    Returns:
        float: Time step in seconds.
    """
    with h5.File(file, "r") as f:
        return float(np.array(f['data']['settings']['DynamicCoupled']['dt']))

def get_number_of_structural_nodes(file: str) -> int:
    """
    Retrieves the number of structural nodes from the first timestep.

    Args:
        file (str): Path to the HDF5 result file.

    Returns:
        int: Number of structural nodes.
    """
    with h5.File(file, "r") as f:
        return int(np.array(f['data']['structure']['timestep_info']['00000']['num_node']))


def get_number_of_chordwise_aero_nodes(file: str) -> int:
    """
    Retrieves the number of chordwise nodes of the wing from the first timestep.

    Args:
        file (str): Path to the HDF5 result file.

    Returns:
        int: Number of chordwise nodes.
    """
    with h5.File(file, "r") as f:
        return int(np.array(f['data']['aero']['timestep_info']['00000']['dimensions'])[0,0])

def get_timestep_str(its: int) -> str:
    """
    Formats the timestep index into SHARPy's expected string format.

    Args:
        its (int): Timestep index.

    Returns:
        str: Zero-padded timestep string (e.g., '00003').
    """
    return f"{its:05d}"


def get_current_aero_ts(file_dir: str, ini_info: AeroTimeStepInfo, ts: int) -> AeroTimeStepInfo:
    """
    Loads the aerodynamic time step data from the SHARPy HDF5 results file 
    into a new AeroTimeStepInfo object.

    Args:
        file_dir (str): Path of the SHARPy HDF5 result file.
        ini_info (AeroTimeStepInfo): Template aerodynamic timestep info object to copy.
        ts (int): Timestep index (e.g., 0, 1, 2, ...).

    Returns:
        AeroTimeStepInfo: Fully populated aerodynamic timestep object for the given timestep.
    
    Notes:
        Assumes global variable `h5_file` is set to the path of the SHARPy HDF5 file.
    """
    aero_tstep = ini_info.copy()
    ts_str = f'{ts:05d}'

    with h5.File(file_dir, "r") as f:
        for isurf in range(aero_tstep.n_surf):
            isurf_str = f'{isurf:05d}'
            aero_tstep.zeta[isurf] = np.array(f['data']['aero']['timestep_info'][ts_str]['zeta'][isurf_str])
            aero_tstep.zeta_star[isurf] = np.array(f['data']['aero']['timestep_info'][ts_str]['zeta_star'][isurf_str])
            aero_tstep.gamma[isurf] = np.array(f['data']['aero']['timestep_info'][ts_str]['gamma'][isurf_str])
            aero_tstep.gamma_star[isurf] = np.array(f['data']['aero']['timestep_info'][ts_str]['gamma_star'][isurf_str])
    
    return aero_tstep

def get_aero_dimensions(file_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieves the UVLM grid dimensions from the first aerodynamic timestep.

    Args:
        file_dir (str): Path of the SHARPy HDF5 result file.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - dimensions: UVLM main surface grid shape [n_chordwise, n_spanwise]
            - dimensions_star: Trailing wake grid shape [n_chordwise_wake, n_spanwise]
    """
    with h5.File(file_dir, "r") as f:
        return (
            np.array(f['data']['aero']['timestep_info']['00000']['dimensions'], dtype=int),
            np.array(f['data']['aero']['timestep_info']['00000']['dimensions_star'], dtype=int)
        )