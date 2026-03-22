from typing import Any, Dict, List, Optional, Union

import numpy as np


def apply_force_corrections(
    settings: Dict[str, Any],
    use_polars: bool = False,
    dynamic: bool = True,
    efficiency_correction: bool = False,
) -> Dict[str, Any]:
    """
    Adds polar or efficiency force corrections to static and dynamic coupled solvers.

    Args:
        settings: SHARPy configuration dictionary.
        use_polars: Whether to apply PolarCorrection.
        efficiency_correction: Whether to apply EfficiencyCorrection.

    Returns:
        The updated settings dictionary.
    """
    solvers = ["StaticCoupled"]
    if dynamic:
        solvers.append("DynamicCoupled")

    if efficiency_correction:
        for solver in solvers:
            settings[solver].update({"correct_forces_method": "EfficiencyCorrection"})

    if use_polars:
        polar_settings = {
            "cd_from_cl": "off",
            "correct_lift": "on",
            "moment_from_polar": "on",
            "aoa_cl0": [0.0, 0.0],
        }

        for solver in solvers:
            settings[solver]["correct_forces_method"] = "PolarCorrection"
            settings[solver]["correct_forces_settings"] = polar_settings
    return settings


def generate_polar_arrays(airfoils: Dict[int, Union[str, bytes]]) -> List[np.ndarray]:
    """
    Loads polar data for multiple airfoils from files.

    Each file must have 12 header rows followed by columns [AoA, Cl, Cd, Cm].
    AoA values greater than 1 are assumed to be in degrees and are converted to radians.

    Args:
        airfoils: Dictionary mapping surface index to polar file path.

    Returns:
        List of arrays of shape (n, 4) containing [AoA (rad), Cl, Cd, Cm] per airfoil.
    """
    polar_data = [None] * len(airfoils)

    for index, filename in airfoils.items():
        # Load first four columns after skipping 12 header lines
        data = np.loadtxt(filename, skiprows=12)[:, :4]
        # Convert AoA to radians if it seems to be in degrees
        if np.any(data[:, 0] > 1.0):
            data[:, 0] *= np.pi / 180

        polar_data[index] = data
    return polar_data


def setup_force_correction(
    polar_file: str, use_polars: bool = False, efficiency_correction: bool = True
) -> Optional[Dict[int, str]]:
    """
    Configures force correction options including polar data and efficiency correction.

    Args:
        polar_file: Path to the polar data file.
        use_polars: Enable polar-based aerodynamic correction.
        efficiency_correction: Enable empirical efficiency correction.

    Returns:
        Dictionary mapping surface index to polar file path, or None if polar correction is disabled.

    Raises:
        ValueError: If both polar and efficiency correction are enabled simultaneously.
    """
    if efficiency_correction and use_polars:
        raise ValueError("Cannot use both efficiency and polar correction.")

    if not use_polars:
        return None

    return {0: polar_file, 1: polar_file}
