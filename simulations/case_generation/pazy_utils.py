from typing import List, Optional

import numpy as np
from pazy_wing_model import PazyWing

from case_generation.settings_classes import PazyModelSettings


def setup_pazy_model(
    case_name: str,
    case_route: str,
    pazy_settings: PazyModelSettings,
    symmetry_condition: bool = False,
    polars: Optional[List[np.ndarray]] = None,
    vertical: bool = True,
):
    """
    Creates and configures a PazyWing model.

    Args:
        case_name: Unique identifier for the simulation case.
        case_route: Path where SHARPy and model files will be stored.
        pazy_settings: Settings for model discretisation and structure.
        symmetry_condition: Whether to apply symmetry across the wing root.
        polars: Optional list of airfoil polar arrays.
        vertical: Whether the wing is mounted vertically.

    Returns:
        The generated and saved Pazy wing model.
    """

    pazy = PazyWing(case_name, case_route, vars(pazy_settings))
    pazy.generate_structure()

    if not symmetry_condition:
        print("mirror wing")
        pazy.structure.mirror_wing()

    if vertical:
        pazy.structure.rotate_wing()

    tip_load = 0  # Optional point load at the tip
    if tip_load > 0.0:
        mid_chord_b = (pazy.get_ea_reference_line() - 0.5) * 0.1
        pazy.structure.add_lumped_mass(
            (
                tip_load,
                pazy.structure.n_node // 2,
                np.zeros((3, 3)),
                np.array([0, mid_chord_b, 0]),
            )
        )
        if not symmetry_condition:
            pazy.structure.add_lumped_mass(
                (
                    tip_load,
                    pazy.structure.n_node // 2 + 1,
                    np.zeros((3, 3)),
                    np.array([0, mid_chord_b, 0]),
                )
            )

    pazy.generate_aero(polars=polars)
    pazy.save_files()

    return pazy
