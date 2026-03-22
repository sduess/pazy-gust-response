import os

import case_generation.force_correction_utils as force_correction_utils
import case_generation.pazy_utils as pazy_utils
import case_generation.sim_config_utils as sim_config_utils
import configobj
import numpy as np
import sharpy.sharpy_main
from case_generation.settings_classes import PazyModelSettings, StaticSimulationSettings

_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

_CFL = 1


def run_static_coupled_simulation(sim: StaticSimulationSettings):
    """Run a static aeroelastic simulation for the Pazy wing."""
    alpha_rad = np.deg2rad(sim.alpha_deg)
    pazy_model = PazyModelSettings(
        skin_on="on",
        discretisation_method="michigan",
        model_id=sim.model_id,
        num_elem=2,
        surface_m=8,
        symmetry_condition=sim.symmetry_condition,
    )

    case_name = (
        f"pazy_steady_alpha_{sim.alpha_deg:g}"
        f"_polars{int(sim.use_polars)}"
        f"_effcor_{int(sim.efficiency_correction)}"
    )
    case_route = sim.case_root
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

    dt = _CFL * pazy.aero.main_chord / pazy_model.surface_m / sim.u_inf
    symmetry_plane = 2 if sim.vertical else 1

    settings = {}
    settings["SHARPy"] = sim_config_utils.setup_sharpy_flow(
        case_name, case_route, sim.output_folder, static=True
    )
    settings["BeamLoader"] = sim_config_utils.get_beamloader_settings(
        alpha_rad, sim.vertical
    )
    settings["AerogridLoader"] = sim_config_utils.setup_aerogrid_loader(
        sim.u_inf, dt, pazy_model.surface_m
    )
    settings.update(
        sim_config_utils.setup_static_settings(
            sim.gravity, sim.rho, sim.u_inf, dt, sim.symmetry_condition, symmetry_plane
        )
    )
    settings.update(sim_config_utils.setup_postprocessors(sim.rho, sim.u_inf))
    settings = force_correction_utils.apply_force_corrections(
        settings,
        sim.use_polars,
        dynamic=False,
        efficiency_correction=sim.efficiency_correction,
    )

    config = configobj.ConfigObj()
    config.filename = os.path.join(case_route, f"{case_name}.sharpy")
    for k, v in settings.items():
        config[k] = v
    config.write()

    return sharpy.sharpy_main.main(
        ["", os.path.join(case_route, f"{case_name}.sharpy")]
    )


def generate_static_coupled_wing_deformation():
    """
    Run static aeroelastic simulation.
    """
    use_polars = True
    efficiency_correction = False
    airfoil_polar_file = os.path.join(
        _DIR, "../lib/pazy-model/src/airfoil_polars/xfoil_seq_re120000_naca0018.txt"
    )
    airfoil_polar = force_correction_utils.setup_force_correction(
        airfoil_polar_file, use_polars, efficiency_correction
    )

    sim = StaticSimulationSettings(
        alpha_deg=5,
        u_inf=18.3,
        case_root="./cases/",
        output_folder="./output/",
        symmetry_condition=True,
        airfoil_polar=airfoil_polar,
        use_polars=use_polars,
        efficiency_correction=efficiency_correction,
        model_id="delft",
        vertical=True,
        gravity=True,
        rho=1.205,
    )

    run_static_coupled_simulation(sim)


if __name__ == "__main__":
    generate_static_coupled_wing_deformation()
