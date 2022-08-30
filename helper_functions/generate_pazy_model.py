import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "lib/pazy-model"))
from pazy_wing_model import PazyWing


def setup_pazy_model(case_name, case_route, pazy_settings, symmetry_condition = False):
    pazy = PazyWing(case_name, case_route, pazy_settings)
    pazy.generate_structure()
    if not symmetry_condition:
        pazy.structure.mirror_wing()
    pazy.generate_aero()

    pazy.save_files()
    return pazy