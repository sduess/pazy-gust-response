from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class TestCaseSettings:
    """
    Parameters that define a specific aerodynamic gust test case.

    Attributes:
        alpha: Angle of attack in radians.
        u_inf: Freestream velocity [m/s].
        gust_T: Period of the gust cycle [s].
        frequency_gust_vane: Frequency of gust vane oscillation [Hz].
        gust_amplitude: Amplitude of the gust [unitless, multiplier of u_inf].
    """

    alpha: float
    u_inf: float
    gust_T: float
    frequency_gust_vane: float
    gust_amplitude: float


@dataclass
class PazyModelSettings:
    """
    Configuration settings for generating the Pazy wing structural and aerodynamic model.

    Attributes:
        skin_on: Whether the wing skin is present ('on' or 'off').
        discretisation_method: Discretisation method used ('michigan', 'even', etc.).
        model_id: Identifier string for the wing variant (e.g., 'delft').
        num_elem: Number of beam elements.
        surface_m: Number of aerodynamic panels along the chord.
        symmetry_condition: Whether symmetry boundary condition is applied.
    """

    skin_on: str
    discretisation_method: str
    model_id: str
    num_elem: int
    surface_m: int
    symmetry_condition: bool


@dataclass
class SimulationRunSettings:
    """
    Master dataclass to store all high-level simulation setup options.

    Attributes:
        case: Case identifier number.
        case_root: Root directory where case data will be saved.
        output_folder: Output directory for simulation results.
        gust_vanes: Whether to simulate gust vanes.
        symmetry_condition: Whether to enable SHARPy symmetry mode.
        test_case_settings: Instance defining gust parameters and freestream.
        airfoil_polar: Mapping of surface ID to polar data file paths.
        use_polars: Whether to enable polar correction in SHARPy.
        efficiency_correction: Whether to apply force correction for efficiency.
        model_id: Which version of the Pazy model to use.
        vertical: Whether the wing is vertically mounted.
        restart: Whether to restart from previous SHARPy results.
        only_gust_vanes: Whether to run a simulation with gust vanes only.
        specific_gust_input: Whether to load a user-defined gust profile.
        convection_scheme: The convection scheme for UVLM.
        gravity: Whether to include gravity in the simulation.
    """

    case: int
    case_root: str
    output_folder: str
    gust_vanes: bool
    symmetry_condition: bool
    test_case_settings: TestCaseSettings
    airfoil_polar: Optional[Dict[int, str]]
    use_polars: bool
    efficiency_correction: bool
    model_id: str
    vertical: bool
    restart: bool
    only_gust_vanes: bool
    specific_gust_input: bool
    convection_scheme: int
    gravity: bool
    rho: float = 1.205


@dataclass
class StaticSimulationSettings:
    """
    Configuration for a static aeroelastic simulation of the Pazy wing.

    Attributes:
        alpha_deg: Angle of attack in degrees.
        u_inf: Freestream velocity [m/s].
        case_root: Root directory where case files will be saved.
        output_folder: Output directory for simulation results.
        symmetry_condition: Whether to apply symmetry boundary condition.
        airfoil_polar: Mapping of surface index to polar file path.
        use_polars: Whether to enable polar-based aerodynamic correction.
        efficiency_correction: Whether to apply empirical efficiency correction.
        model_id: Which version of the Pazy model to use.
        vertical: Whether the wing is mounted vertically.
        gravity: Whether to include gravity.
        rho: Air density [kg/m^3].
    """

    alpha_deg: float
    u_inf: float
    case_root: str = "./cases/"
    output_folder: str = "./output/"
    symmetry_condition: bool = False
    airfoil_polar: Optional[Dict[int, str]] = None
    use_polars: bool = False
    efficiency_correction: bool = False
    model_id: str = "delft"
    vertical: bool = True
    gravity: bool = True
    rho: float = 1.205


def create_test_cases() -> Dict[int, TestCaseSettings]:
    test_cases = {
        1: TestCaseSettings(
            alpha=np.deg2rad(5.0),
            u_inf=18.3,
            gust_T=0.175439,
            frequency_gust_vane=5.7,
            gust_amplitude=0.81,
        ),
        2: TestCaseSettings(
            alpha=np.deg2rad(10.0),
            u_inf=18.3,
            gust_T=0.3125,
            frequency_gust_vane=3.2,
            gust_amplitude=0.65,
        ),
    }
    for case in test_cases.values():
        case.gust_T = 1.0 / case.frequency_gust_vane
    return test_cases
