import os
from typing import Any, Dict

import numpy as np


def apply_gust_vane_settings(
    settings: Dict[str, Any],
    cs_deflection_file: str,
    dt: float,
    u_inf: float,
    surface_m: int,
    vertical: bool,
    symmetry_condition: bool,
    only_gust_vanes: bool,
) -> None:
    """
    Updates the SHARPy configuration with gust vane definitions.

    Args:
        settings: SHARPy configuration dictionary.
        cs_deflection_file: Path to the gust vane deflection file.
        dt: Time step size.
        u_inf: Freestream velocity.
        surface_m: Chordwise discretisation.
        vertical: Orientation of the gust vanes (vertical or horizontal).
        symmetry_condition: Whether the problem is symmetric.
        only_gust_vanes: Whether gust vanes are the only gust source.

    Returns:
        None. Modifies the input settings dictionary in-place.
    """
    wake_length_vanes = 5
    gust_vane_parameters = {
        "M": surface_m * 3,
        "N": 40,
        "M_star": int(wake_length_vanes / (dt * u_inf)),
        "span": 5,
        "chord": 0.3,
        "control_surface_deflection_generator_settings": {
            "dt": dt,
            "deflection_file": cs_deflection_file,
        },
    }

    settings["AerogridLoader"]["gust_vanes"] = True
    settings["AerogridLoader"]["gust_vanes_generator_settings"] = {
        "n_vanes": 2,
        "streamwise_position": [-1.5, -1.5],
        "vertical_position": [-0.25, 0.25],
        "symmetry_condition": symmetry_condition,
        "vane_parameters": [gust_vane_parameters, gust_vane_parameters],
        "vertical": vertical,
    }

    if only_gust_vanes:
        # Push vanes far back (20 m) as workaround for SHARPy not supporting vane-only setups
        for i in range(
            settings["AerogridLoader"]["gust_vanes_generator_settings"]["n_vanes"]
        ):
            settings["AerogridLoader"]["gust_vanes_generator_settings"][
                "streamwise_position"
            ][i] -= 20
        settings["AerogridLoader"]["mstar"] = 20

    # Override velocity field for gust vanes
    stepuvlm_updates = {
        "convection_scheme": 3,
        "velocity_field_generator": "SteadyVelocityField",
        "velocity_field_input": {
            "u_inf": u_inf,
            "u_inf_direction": [1.0, 0.0, 0.0],
        },
    }
    settings["DynamicCoupled"]["aero_solver_settings"].update(stepuvlm_updates)


def write_deflection_file(
    n_tstep: int,
    dt: float,
    amplitude: float,
    frequency: float,
    mean: float,
    surface_m: int,
    route_test_dir: str,
) -> str:
    """
    Generates and writes a prescribed control surface deflection signal to a CSV file.

    The deflection is modeled as a sine wave and saved for use in gust vane simulations.

    Args:
        n_tstep: Number of time steps in the simulation.
        dt: Time step size [s].
        amplitude: Amplitude of the sinusoidal deflection [rad].
        frequency: Frequency of the sinusoidal deflection [Hz].
        mean: Mean deflection (currently unused).
        surface_m: Chordwise panel count, used to distinguish output file names.
        route_test_dir: Path to the directory where the output file will be saved.

    Returns:
        Path to the generated CSV file.
    """
    # Define output file path
    filename = f"cs_deflection_amplitude_{amplitude}_frequency_{frequency}_mean_0_m{surface_m}.csv"

    cs_deflection_file = os.path.join(
        route_test_dir, "gust_vane_deflection_files", filename
    )

    # Time vector
    time = np.linspace(0.0, n_tstep * dt, n_tstep)

    # Generate sine wave signal for deflection
    deflection = amplitude * np.sin(2 * np.pi * frequency * time)

    # Insert initial phase (currently 0 steps with 0 deflection)
    initial_deflection = 0.0
    n_tsteps_wo_deflection = 0
    if n_tsteps_wo_deflection > 0:
        deflection = np.insert(
            deflection, 0, initial_deflection * np.ones(n_tsteps_wo_deflection)
        )
        deflection = np.delete(
            deflection, list(range(n_tstep - n_tsteps_wo_deflection, n_tstep))
        )

    # Save to CSV
    np.savetxt(cs_deflection_file, deflection)

    return cs_deflection_file
