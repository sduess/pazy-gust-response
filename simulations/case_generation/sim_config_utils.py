import os
from typing import Any, Dict

import numpy as np
import sharpy.utils.algebra as algebra


def setup_sharpy_flow(
    case_name: str,
    case_route: str,
    output_folder: str,
    static: bool = False,
    restart: bool = False,
) -> Dict[str, Any]:
    """
    Defines the SHARPy flow sequence and logging options.

    Args:
        case_name: Name of the SHARPy case.
        case_route: Path to the case directory.
        output_folder: Path where logs and outputs are stored.
        restart: Whether to restart from a previous simulation.

    Returns:
        A dictionary with SHARPy flow configuration.
    """
    if static:
        flow_sequence = [
            "BeamLoader",
            "AerogridLoader",
            "StaticCoupled",
            "BeamPlot",
            "AerogridPlot",
            "SaveData",
        ]
    else:
        flow_sequence = [
            "BeamLoader",
            "AerogridLoader",
            "BeamPlot",
            "AerogridPlot",
            "StaticCoupled",
            "AeroForcesCalculator",
            "LiftDistribution",
            "DynamicCoupled",
            "SaveData",
        ]
        if restart:
            flow_sequence = ["DynamicCoupled"]

    return {
        "flow": flow_sequence,
        "case": case_name,
        "route": case_route,
        "write_screen": "on",
        "write_log": "on",
        "log_folder": os.path.join(output_folder, case_name),
        "log_file": f"{case_name}.log",
    }


def get_beamloader_settings(alpha: float, vertical: bool = True) -> dict:
    """
    Generates BeamLoader settings with the appropriate orientation based on the wing placement.

    Args:
        alpha: Angle of attack in radians.
        vertical: True if wing is vertical (rotation around z-axis), False if horizontal.

    Returns:
        BeamLoader settings dictionary.
    """
    axis_index = 2 if vertical else 1
    rotation_vector = np.zeros(3)
    rotation_vector[axis_index] = alpha
    orientation = algebra.euler2quat(rotation_vector)

    return {"unsteady": "on", "orientation": orientation}


def setup_aerogrid_loader(
    u_inf: float, dt: float, surface_m: int, only_gust_vanes: bool = False
) -> Dict[str, Any]:
    """
    Prepares the AerogridLoader configuration.

    Args:
        u_inf: Freestream velocity.
        dt: Time step size.
        surface_m: Chordwise panels.
        only_gust_vanes: Flag indicating whether only gust vanes are present.

    Returns:
        A dictionary for AerogridLoader settings.
    """
    mstar = 20 if only_gust_vanes else 20 * surface_m
    return {
        "unsteady": "on",
        "aligned_grid": "on",
        "mstar": mstar,
        "wake_shape_generator": "StraightWake",
        "wake_shape_generator_input": {"u_inf": u_inf, "dt": dt},
    }


def setup_postprocessors(rho: float, u_inf: float) -> Dict[str, dict]:
    """
    Creates a dictionary of SHARPy postprocessor settings.

    This function defines standard postprocessors such as AeroForcesCalculator,
    LiftDistribution, BeamPlot, AerogridPlot, SaveData, and PickleData.

    Args:
        rho: Air density [kg/m^3].
        u_inf: Freestream velocity [m/s].

    Returns:
        Dictionary of postprocessor settings keyed by module name.
    """
    q_ref = 0.5 * rho * u_inf**2  # Dynamic pressure
    S_ref = 0.1 * 0.55  # Reference surface area, adjust as needed

    return {
        "AeroForcesCalculator": {
            "write_text_file": "on",
            "screen_output": "on",
            "coefficients": False,
            "q_ref": q_ref,
            "S_ref": S_ref,
        },
        "LiftDistribution": {"coefficients": True, "rho": rho},
        "BeamPlot": {},
        "AerogridPlot": {
            "include_rbm": "off",
            "include_applied_forces": "on",
            "minus_m_star": 0,
        },
        "SaveData": {"save_aero": True, "save_struct": True, "save_linear": True},
        "PickleData": {},
    }


def setup_static_settings(
    gravity: bool,
    rho: float,
    u_inf: float,
    dt: float,
    symmetry_condition: bool,
    symmetry_plane: int,
) -> Dict[str, Any]:
    """
    Generates solver settings for the static solver coupling steady aerodynamics
    and a nonlinear structural solver.

    Args:
        gravity: Whether to enable gravity.
        rho: Air density.
        u_inf: Freestream velocity.
        dt: Time step size.
        symmetry_condition: Symmetry condition flag.
        symmetry_plane: Symmetry plane axis (0=x, 1=y, 2=z).

    Returns:
        A dictionary with static solver settings.
    """
    settings_nonlinear_static = {
        "print_info": "off",
        "max_iterations": 1000,
        "num_load_steps": 5,
        "delta_curved": 1e-5,
        "min_delta": 1e-6,
        "gravity_on": gravity,
        "relaxation_factor": 0.1,
        "gravity": 9.81,
    }
    settings_static_uvlm = {
        "rho": rho,
        "print_info": True,
        "horseshoe": False,
        "num_cores": 4,
        "n_rollup": 0,
        "rollup_dt": dt,
        "rollup_aic_refresh": 1,
        "rollup_tolerance": 1e-4,
        "velocity_field_generator": "SteadyVelocityField",
        "velocity_field_input": {"u_inf": u_inf, "u_inf_direction": [1.0, 0.0, 0.0]},
        "vortex_radius": 1e-9,
        "symmetry_condition": symmetry_condition,
        "symmetry_plane": symmetry_plane,
    }
    return {
        "StaticCoupled": {
            "print_info": "on",
            "max_iter": 200,
            "n_load_steps": 8,
            "tolerance": 1e-5,
            "relaxation_factor": 0.2,
            "aero_solver": "StaticUvlm",
            "aero_solver_settings": settings_static_uvlm,
            "structural_solver": "NonLinearStatic",
            "structural_solver_settings": settings_nonlinear_static,
        }
    }


def setup_dynamic_coupled_settings(
    settings: dict,
    rho: float,
    u_inf: float,
    dt: float,
    n_time_steps: int,
    num_cores: int,
    gust_T: float,
    gust_amplitude: float,
    gust_component: int,
    convection_scheme: int,
    variable_wake: bool,
    symmetry_condition: bool,
    symmetry_plane: int,
    gravity: bool,
    tolerance: float,
    fsi_tolerance: float,
    case_id: int,
    specific_gust_input: bool,
    route_test_dir: str,
) -> dict:
    """
    Adds dynamic coupled solver settings to an existing SHARPy settings dictionary.

    This includes the StepUvlm aero solver, the NonLinearDynamicPrescribedStep structural solver,
    and the combined DynamicCoupled solver.

    Args:
        settings: The SHARPy config dictionary to update (used to forward SaveData settings).
        rho: Air density [kg/m^3].
        u_inf: Freestream velocity [m/s].
        dt: Time step size [s].
        n_time_steps: Number of dynamic time steps.
        num_cores: Number of CPU cores for UVLM.
        gust_T: Gust period [s].
        gust_amplitude: Gust intensity (fraction of u_inf).
        gust_component: Axis index of the gust velocity component.
        convection_scheme: Wake convection scheme index.
        variable_wake: Whether the wake uses variable spacing (disables CFL=1 enforcement).
        symmetry_condition: Whether to apply aerodynamic symmetry.
        symmetry_plane: Symmetry plane axis (0=x, 1=y, 2=z).
        gravity: Whether to include gravity.
        tolerance: Structural solver convergence tolerance.
        fsi_tolerance: FSI coupling convergence tolerance.
        case_id: Case identifier, used to locate a specific gust input file.
        specific_gust_input: Whether to load a user-defined gust profile from file.
        route_test_dir: Path to the simulation directory (used for gust input file lookup).

    Returns:
        DynamicCoupled solver settings dictionary.
    """
    velocity_field_input = {
        "u_inf": u_inf,
        "u_inf_direction": [1.0, 0.0, 0.0],
        "relative_motion": True,
        "offset": 10 * dt * u_inf,
        "gust_shape": "continuous_sin",
        "gust_parameters": {
            "gust_length": gust_T * u_inf,
            "gust_intensity": 2 * gust_amplitude,
            "gust_component": gust_component,
        },
    }

    if specific_gust_input:
        file_path = os.path.join(
            route_test_dir,
            f"04_gust_inputs/frozen_gust_input_pazy_vertical_case_{int(case_id)}_polars0_effcor_0dynamic_m8_gust_vanes_only.csv",
        )
        velocity_field_input = {
            "u_inf": u_inf,
            "u_inf_direction": [1.0, 0.0, 0.0],
            "relative_motion": True,
            "offset": 10 * dt * u_inf,
            "gust_shape": "time varying",
            "gust_parameters": {"file": file_path},
        }

    settings_step_uvlm = {
        "num_cores": num_cores,
        "convection_scheme": convection_scheme,
        "gamma_dot_filtering": 0,
        "cfl1": not variable_wake,
        "velocity_field_generator": "GustVelocityField",
        "velocity_field_input": velocity_field_input,
        "rho": rho,
        "n_time_steps": n_time_steps,
        "dt": dt,
        "symmetry_condition": symmetry_condition,
        "symmetry_plane": symmetry_plane,
    }

    nonlinear_dynamic_prescribed_step = {
        "print_info": "on",
        "max_iterations": 950,
        "delta_curved": 1e-1,
        "min_delta": tolerance,
        "newmark_damp": 1e-4,
        "gravity_on": gravity,
        "gravity": 9.81,
        "num_steps": n_time_steps,
        "dt": dt,
    }

    return {
        "structural_solver": "NonLinearDynamicPrescribedStep",
        "structural_solver_settings": nonlinear_dynamic_prescribed_step,
        "aero_solver": "StepUvlm",
        "aero_solver_settings": settings_step_uvlm,
        "fsi_substeps": 200,
        "fsi_tolerance": fsi_tolerance,
        "relaxation_factor": 0.0,
        "minimum_steps": 1,
        "relaxation_steps": 150,
        "final_relaxation_factor": 0.0,
        "n_time_steps": n_time_steps,
        "print_info": True,
        "dt": dt,
        "include_unsteady_force_contribution": True,
        "postprocessors": ["BeamLoads", "BeamPlot", "AerogridPlot", "SaveData"],
        "postprocessors_settings": {
            "BeamLoads": {"csv_output": "off"},
            "BeamPlot": {"include_rbm": "on", "include_applied_forces": "on"},
            "AerogridPlot": {"include_rbm": "on", "include_applied_forces": "on"},
            "SaveData": settings.get(
                "SaveData", {}
            ),  # TODO: get all settings from the postproc setting function
        },
    }
