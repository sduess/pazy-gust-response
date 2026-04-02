import os

import case_generation.force_correction_utils as force_correction_utils
import case_generation.gust_vane_utils as gust_vane_utils
import case_generation.pazy_utils as pazy_utils
import case_generation.sim_config_utils as sim_config_utils
import configobj
import numpy as np
import sharpy.sharpy_main
from case_generation.settings_classes import (
    PazyModelSettings,
    SimulationRunSettings,
    create_test_cases,
)

_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
_PROJECT_ROOT = os.path.dirname(_DIR)

# Fixed simulation constants
_N_TSTEP = 6000
_CFL = 1
_TOLERANCE = 1e-6
_FSI_TOLERANCE = 1e-4
_NUM_CORES = 4


def _build_pazy_model_settings(sim: SimulationRunSettings) -> PazyModelSettings:
    return PazyModelSettings(
        skin_on="on",
        discretisation_method="even" if sim.only_gust_vanes else "michigan",
        model_id=sim.model_id,
        num_elem=4 if sim.only_gust_vanes else 2,
        surface_m=8,
        symmetry_condition=sim.symmetry_condition,
    )


def _build_case_name(sim: SimulationRunSettings, surface_m: int) -> str:
    name = (
        f"pazy_vertical_case_{sim.case}"
        f"_polars{int(sim.use_polars)}"
        f"_effcor_{int(sim.efficiency_correction)}"
        f"_dynamic_m{surface_m}"
    )
    if sim.gust_vanes:
        name += "_gust_vanes"
        if sim.only_gust_vanes:
            name += "_only"
    if sim.restart:
        name += "_restart"
    return name


def _build_settings(
    sim: SimulationRunSettings,
    case_name: str,
    case_route: str,
    pazy_model: PazyModelSettings,
    chord: float,
) -> dict:
    tc = sim.test_case_settings
    dt = _CFL * chord / pazy_model.surface_m / tc.u_inf
    gust_component = 1 if sim.vertical else 2
    symmetry_plane = 2 if sim.vertical else 1

    settings = {}

    settings["SHARPy"] = sim_config_utils.setup_sharpy_flow(
        case_name, case_route, sim.output_folder, static=False, restart=sim.restart
    )
    settings["BeamLoader"] = sim_config_utils.get_beamloader_settings(
        tc.alpha, sim.vertical
    )
    settings["AerogridLoader"] = sim_config_utils.setup_aerogrid_loader(
        tc.u_inf, dt, pazy_model.surface_m, sim.only_gust_vanes
    )
    settings.update(
        sim_config_utils.setup_static_settings(
            sim.gravity, sim.rho, tc.u_inf, dt, sim.symmetry_condition, symmetry_plane
        )
    )
    settings.update(sim_config_utils.setup_postprocessors(sim.rho, tc.u_inf))

    settings["Modal"] = {
        "NumLambda": 40,
        "rigid_body_modes": "off",
        "print_matrices": "off",
        "continuous_eigenvalues": "off",
        "write_modes_vtk": "off",
        "use_undamped_modes": "on",
    }

    cs_deflection_file = None
    if sim.gust_vanes:
        cs_deflection_file = gust_vane_utils.write_deflection_file(
            n_tstep=_N_TSTEP,
            dt=dt,
            amplitude=np.deg2rad(5.0),
            frequency=tc.frequency_gust_vane,
            mean=0.0,
            surface_m=pazy_model.surface_m,
            route_test_dir=_DIR,
        )

    settings["DynamicCoupled"] = sim_config_utils.setup_dynamic_coupled_settings(
        settings=settings,
        rho=sim.rho,
        u_inf=tc.u_inf,
        dt=dt,
        n_time_steps=_N_TSTEP,
        num_cores=_NUM_CORES,
        gust_T=tc.gust_T,
        gust_amplitude=tc.gust_amplitude,
        gust_component=gust_component,
        convection_scheme=sim.convection_scheme,
        variable_wake=False,
        symmetry_condition=sim.symmetry_condition,
        symmetry_plane=symmetry_plane,
        gravity=sim.gravity,
        tolerance=_TOLERANCE,
        fsi_tolerance=_FSI_TOLERANCE,
        case_id=sim.case,
        specific_gust_input=sim.specific_gust_input,
        route_test_dir=_DIR,
    )

    if sim.gust_vanes:
        gust_vane_utils.apply_gust_vane_settings(
            settings=settings,
            cs_deflection_file=cs_deflection_file,
            dt=dt,
            u_inf=tc.u_inf,
            surface_m=pazy_model.surface_m,
            vertical=sim.vertical,
            symmetry_condition=sim.symmetry_condition,
            only_gust_vanes=sim.only_gust_vanes,
        )

    settings = force_correction_utils.apply_force_corrections(
        settings=settings,
        use_polars=sim.use_polars,
        dynamic=True,
        efficiency_correction=sim.efficiency_correction,
    )

    return settings


def run_gust_response(sim: SimulationRunSettings):
    """Run a dynamic gust response simulation for the Pazy wing."""
    pazy_model = _build_pazy_model_settings(sim)
    case_name = _build_case_name(sim, pazy_model.surface_m)
    case_route = os.path.join(sim.case_root, case_name)
    os.makedirs(case_route, exist_ok=True)

    polar_arrays = None
    if sim.airfoil_polar is not None:
        polar_arrays = force_correction_utils.generate_polar_arrays(sim.airfoil_polar)

    pazy = pazy_utils.setup_pazy_model(
        case_name=case_name,
        case_route=case_route,
        pazy_settings=pazy_model,
        symmetry_condition=sim.symmetry_condition,
        polars=polar_arrays,
        vertical=sim.vertical,
    )

    settings = _build_settings(
        sim, case_name, case_route, pazy_model, chord=pazy.aero.main_chord
    )

    config = configobj.ConfigObj()
    config.filename = os.path.join(case_route, f"{case_name}.sharpy")
    for k, v in settings.items():
        config[k] = v
    config.write()

    return sharpy.sharpy_main.main(
        ["", os.path.join(case_route, f"{case_name}.sharpy")]
    )


def generate_dynamic_gust_response_pazy():
    """
    Run gust response cases from Duessler et al., AIAA Journal 2025 (DOI 10.2514/1.J038332).

    Set ``gust_vanes=True`` to simulate oscillating gust vanes as used in the wind tunnel
    experiment, or leave False to use SHARPy's built-in frozen gust model.
    """
    test_cases = create_test_cases()
    case_id = 1

    use_polars = False
    efficiency_correction = False
    airfoil_polar_file = os.path.join(
        _PROJECT_ROOT, "lib/pazy-model/src/airfoil_polars/xfoil_seq_re120000_naca0018.txt"
    )
    airfoil_polar = force_correction_utils.setup_force_correction(
        airfoil_polar_file, use_polars, efficiency_correction
    )

    sim = SimulationRunSettings(
        case=case_id,
        case_root=os.path.join(_PROJECT_ROOT, "cases"),
        output_folder=os.path.join(_PROJECT_ROOT, "output"),
        gust_vanes=False,
        symmetry_condition=True,
        test_case_settings=test_cases[case_id],
        airfoil_polar=airfoil_polar,
        use_polars=use_polars,
        efficiency_correction=efficiency_correction,
        model_id="delft",
        vertical=True,
        restart=False,
        only_gust_vanes=False,
        specific_gust_input=False,
        convection_scheme=2,
        gravity=True,
        rho=1.205,
    )

    run_gust_response(sim)


if __name__ == "__main__":
    generate_dynamic_gust_response_pazy()
