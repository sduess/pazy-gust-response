import os

_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
_PROJECT_ROOT = os.path.dirname(_DIR)

import matplotlib.pyplot as plt
import numpy as np
import utils.experimental_data as reference_utils
import utils.h5tools as h5_utils
import utils.helpers as utils
import utils.plotting as plot_utils


def postproc_static_results(
    result_folder: str,
    case_format_str: str,
    half_wingspan: float,
    vertical: bool,
    output_plot_file: str,
    experimental_data: str,
) -> None:
    """
    Post-processes static deformation results by comparing experimental data to
    SHARPy simulation output and generates a plot.

    Args:
        result_folder (str): Directory containing simulation result folders.
        case_format_str (str): Format string for simulation case names.
        half_wingspan (float): Wing half-span, used for normalization.
        vertical (bool): If True, extract vertical deformation (y); else lateral (z).
        output_plot_file (str): Path to save the resulting plot as a PNG.
    """
    list_alpha = [5, 10]  # Angles of attack in degrees
    idx_spanwise_coord, idx_displacement_coord = (
        utils.get_spanwise_and_displacement_coordinate(vertical)
    )
    for ialpha, alpha in enumerate(list_alpha):
        ref_deformation = reference_utils.get_reference_deformation_static(
            ialpha + 1, experimental_data
        )
        plt.scatter(
            ref_deformation[:, 0],
            ref_deformation[:, 1],
            color=plot_utils.get_color_by_index(ialpha),
            marker="d",  # type: ignore[arg-type]
            label=r"Experiments - $\alpha$ = {} deg".format(alpha),
        )

        for use_polars in range(2):
            for use_eff_correction in range(2):
                if use_polars and use_eff_correction:
                    continue
                case_name = case_format_str.format(
                    alpha, use_polars, use_eff_correction
                )
                file = os.path.join(
                    result_folder,
                    case_name,
                    case_name,
                    "savedata",
                    case_name + ".data.h5",
                )
                wing_deformation = h5_utils.read_structural_deformation(
                    file, half_wingspan, h5_utils.get_timestep_str(0)
                )
                plt.plot(
                    np.abs(wing_deformation[:, idx_spanwise_coord]),
                    np.abs(wing_deformation[:, idx_displacement_coord]),
                    linestyle=plot_utils.get_linestyle(
                        bool(use_polars), bool(use_eff_correction)
                    ),
                    color=plot_utils.get_color_by_index(ialpha),
                    label=plot_utils.get_label(
                        alpha, bool(use_polars), bool(use_eff_correction)
                    ),
                )

    plt.grid()
    plt.ylabel("y/s")
    plt.xlabel("z/s")
    plt.legend()
    plt.savefig(output_plot_file)
    plt.show()


if __name__ == "__main__":
    output_folder = os.path.join(_PROJECT_ROOT, "output")
    case_format_str = "pazy_steady_alpha_{}_polars{}_effcor_{}"
    plot_file = os.path.join(_PROJECT_ROOT, "results", "plots", "static_deformation.png")
    experimental_data = os.path.join(_PROJECT_ROOT, "experimental_data")
    pazy_half_span = 0.55
    vertical = True

    postproc_static_results(
        output_folder,
        case_format_str,
        pazy_half_span,
        True,
        plot_file,
        experimental_data,
    )
