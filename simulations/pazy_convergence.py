import os

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

_N_TSTEP = 2000
_CFL = 1
_TOLERANCE = 1e-6
_FSI_TOLERANCE = 1e-4
_NUM_CORES = 4


def _build_convergence_case_name(
    sim: SimulationRunSettings,
    surface_m: int,
    n_vanes: int,
    m_vanes: int,
    wake_length_vanes: float,
) -> str:
    return (
        f"pazy_convergence_case_{sim.case}"
        f"_wingM{surface_m}"
        f"_vaneN{n_vanes}_vaneM{m_vanes}_vaneMstar{wake_length_vanes:.0f}m"
    )


def run_gust_vane_convergence(
    sim: SimulationRunSettings,
    surface_m: int,
    list_n_vanes: list,
    list_m_vanes: list,
    list_wake_length_vanes: list,
) -> None:
    """
    Run a parametric convergence study over gust vane discretisation.

    Loops over all combinations of spanwise panels (N), chordwise panels (M),
    and wake length for the gust vanes, keeping the wing discretisation fixed.

    Args:
        sim: Base simulation settings (test case, freestream, flags).
        surface_m: Chordwise panels on the wing (fixed across the study).
        list_n_vanes: Spanwise vane panel counts to sweep.
        list_m_vanes: Chordwise vane panel counts to sweep.
        list_wake_length_vanes: Vane wake lengths [m] to sweep.
    """
    tc = sim.test_case_settings
    gust_component = 1 if sim.vertical else 2
    symmetry_plane = 2 if sim.vertical else 1

    pazy_model = PazyModelSettings(
        skin_on="on",
        discretisation_method="michigan",
        model_id=sim.model_id,
        num_elem=2,
        surface_m=surface_m,
        symmetry_condition=sim.symmetry_condition,
    )

    for n in list_n_vanes:
        for m in list_m_vanes:
            for wake_length in list_wake_length_vanes:
                case_name = _build_convergence_case_name(
                    sim, surface_m, n, m, wake_length
                )
                case_route = os.path.join(sim.case_root, case_name)
                os.makedirs(case_route, exist_ok=True)

                pazy = pazy_utils.setup_pazy_model(
                    case_name=case_name,
                    case_route=case_route,
                    pazy_settings=pazy_model,
                    symmetry_condition=sim.symmetry_condition,
                    vertical=sim.vertical,
                )
                dt = _CFL * pazy.aero.main_chord / surface_m / tc.u_inf

                settings = {}
                settings["SHARPy"] = sim_config_utils.setup_sharpy_flow(
                    case_name, case_route, sim.output_folder
                )
                settings["BeamLoader"] = sim_config_utils.get_beamloader_settings(
                    tc.alpha, sim.vertical
                )
                settings["AerogridLoader"] = sim_config_utils.setup_aerogrid_loader(
                    tc.u_inf, dt, surface_m
                )
                settings.update(
                    sim_config_utils.setup_static_settings(
                        sim.gravity,
                        sim.rho,
                        tc.u_inf,
                        dt,
                        sim.symmetry_condition,
                        symmetry_plane,
                    )
                )
                settings.update(
                    sim_config_utils.setup_postprocessors(sim.rho, tc.u_inf)
                )

                cs_deflection_file = gust_vane_utils.write_deflection_file(
                    n_tstep=_N_TSTEP,
                    dt=dt,
                    amplitude=np.deg2rad(5.0),
                    frequency=tc.frequency_gust_vane,
                    mean=0.0,
                    surface_m=surface_m,
                    route_test_dir=_DIR,
                )

                settings["DynamicCoupled"] = (
                    sim_config_utils.setup_dynamic_coupled_settings(
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
                        specific_gust_input=False,
                        route_test_dir=_DIR,
                    )
                )

                gust_vane_utils.apply_gust_vane_settings(
                    settings=settings,
                    cs_deflection_file=cs_deflection_file,
                    dt=dt,
                    u_inf=tc.u_inf,
                    surface_m=surface_m,
                    vertical=sim.vertical,
                    symmetry_condition=sim.symmetry_condition,
                    only_gust_vanes=False,
                )
                # Override vane discretisation with convergence study values
                for vane_params in settings["AerogridLoader"][
                    "gust_vanes_generator_settings"
                ]["vane_parameters"]:
                    vane_params["M"] = m
                    vane_params["N"] = n
                    vane_params["M_star"] = int(wake_length / (dt * tc.u_inf))

                config = configobj.ConfigObj()
                config.filename = os.path.join(case_route, f"{case_name}.sharpy")
                for k, v in settings.items():
                    config[k] = v
                config.write()

                sharpy.sharpy_main.main(
                    ["", os.path.join(case_route, f"{case_name}.sharpy")]
                )


def main():
    test_cases = create_test_cases()
    case_id = 1

    sim = SimulationRunSettings(
        case=case_id,
        case_root="./cases/",
        output_folder="./output/",
        gust_vanes=True,
        symmetry_condition=True,
        test_case_settings=test_cases[case_id],
        airfoil_polar=None,
        use_polars=False,
        efficiency_correction=False,
        model_id="delft",
        vertical=True,
        restart=False,
        only_gust_vanes=False,
        specific_gust_input=False,
        convection_scheme=2,
        gravity=True,
    )

    run_gust_vane_convergence(
        sim=sim,
        surface_m=4,
        list_n_vanes=[10, 20, 40],
        list_m_vanes=[8, 16, 24],
        list_wake_length_vanes=[3, 5],
    )


if __name__ == "__main__":
    main()
