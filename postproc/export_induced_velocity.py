import os
import tempfile
from typing import Optional, Tuple

import h5py as h5
import numpy as np
import sharpy.aero.utils.uvlmlib as uvlmlib
import utils.h5tools as h5_utils
import utils.helpers as utils
from sharpy.utils.datastructures import AeroTimeStepInfo

_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

from pazy_wing_model import PazyWing  # noqa: E402


def get_pazy_geometry(model_id: str = "delft") -> tuple[float, float, float]:
    """
    Instantiates a minimal Pazy wing model to extract fixed geometric parameters.

    No files are written to disk; the model is used only for geometry extraction.

    Args:
        model_id: Wing model variant (e.g. 'delft', 'pazy').

    Returns:
        tuple: (chord [m], half_wingspan [m], ea_main [-])
    """
    pazy = PazyWing(
        "_geom_query",
        tempfile.gettempdir(),
        {"model_id": model_id, "num_elem": 2, "symmetry_condition": True},
    )
    pazy.generate_structure()
    pazy.generate_aero()
    return pazy.aero.main_chord, pazy.structure.y[-1], pazy.get_ea_reference_line()


def get_velocity_field_points(
    chord: float,
    wing_span: float,
    ea_main: float,
    spanwise_slices: bool = False,
    dict_spanwise_slices: Optional[dict] = None,
) -> np.ndarray:
    """
    Generates a set of 3D points for evaluating the induced velocity field.

    The function supports two modes:
    1. `spanwise_slices=True`: generates vertical slices at multiple spanwise positions.
    2. `spanwise_slices=False`: generates a 1D chordwise line at y=0 and z=0.

    Args:
        chord: Wing chord length [m].
        wing_span: Wing half-span [m].
        ea_main: Elastic axis position as fraction of chord from leading edge [-].
        spanwise_slices: Whether to generate vertical slices.
        dict_spanwise_slices: Required if `spanwise_slices=True`. Must contain:
            - 'n_points_horizontal': int
            - 'n_points_vertical': int
            - 'spanwise_positions': list of float
            - 'x_0', 'x_1': float (start and end in x-direction)
            - 'z_0', 'z_1': float (start and end in z-direction)

    Returns:
        np.ndarray: Array of shape (N, 3) where N is the number of velocity field points.
                    Each row represents a point (x, z, y).
    """
    if dict_spanwise_slices is None:
        dict_spanwise_slices = {}

    if spanwise_slices:
        nh = dict_spanwise_slices["n_points_horizontal"]
        nv = dict_spanwise_slices["n_points_vertical"]
        y_positions = dict_spanwise_slices["spanwise_positions"]
        x_0, x_1 = dict_spanwise_slices["x_0"], dict_spanwise_slices["x_1"]
        z_0, z_1 = dict_spanwise_slices["z_0"], dict_spanwise_slices["z_1"]

        num_slices = len(y_positions)
        total_points = num_slices * nh * nv
        velocity_field_points = np.zeros((total_points, 3))

        dx = (x_1 - x_0) / (nh - 1)
        dz = (z_1 - z_0) / (nv - 1)

        point_id = 0
        for y_rel in y_positions:
            y = y_rel * wing_span
            for i in range(nh):
                x = x_0 + i * dx
                for j in range(nv):
                    z = z_0 + j * dz
                    velocity_field_points[point_id] = [x, z, y]
                    point_id += 1

        print(
            f"Generated {total_points} velocity points across {num_slices} spanwise slices."
        )

    else:
        # Simple line in x-direction (used for debugging)
        num_points = 20
        velocity_field_points = np.zeros((num_points, 3))
        for i in range(num_points):
            x = -(ea_main + i / (num_points - 1)) * chord
            velocity_field_points[i, 0] = x

        print(f"Generated {num_points} velocity points along a single x-line.")

    return velocity_field_points


def load_wing_corner_points_for_time_interval(
    file_path: str,
    ts_start: int,
    nts: int,
) -> np.ndarray:
    """Load wing corner points (`zeta[0]`) from a SHARPy HDF5 results file.

    This function extracts the aero surface corner coordinates for a sequence
    of timesteps. The data is read from the SHARPy dataset:
    ``/data/aero/timestep_info/<ts>/zeta/00000``.

    The output array has shape ``(nts, *zeta_shape)``, where the first dimension
    corresponds to time.

    Args:
        file_path: Path to the `.h5` SHARPy results file.
        ts_start: Index of the first timestep to load.
        nts: Number of consecutive timesteps to load.

    Returns:
        np.ndarray: An array of shape ``(nts, *zeta_shape)`` containing
            the wing corner points at each timestep.
    """
    with h5.File(file_path, "r") as f:
        ts0_str = h5_utils.get_timestep_str(ts_start)
        ds0 = f["data"]["aero"]["timestep_info"][ts0_str]["zeta"]["00000"]

        zeta_shape: Tuple[int, ...] = ds0.shape
        dtype = ds0.dtype

        wing_corner_points = np.zeros((nts, *zeta_shape), dtype=dtype)

        for timestep_index in range(nts):
            ts_str = h5_utils.get_timestep_str(ts_start + timestep_index)
            wing_corner_points[timestep_index, ::] = np.array(
                f["data"]["aero"]["timestep_info"][ts_str]["zeta"]["00000"]
            )
        return wing_corner_points


def get_index_for_spanwise_position(
    spanwise_coordinates: np.ndarray, wing_position_in_percent: float
) -> int:
    normalized = spanwise_coordinates / np.max(spanwise_coordinates)
    return utils.find_index_of_closest_entry(normalized, wing_position_in_percent)


def export_induced_velocities(
    h5_file: str,
    output_folder: str,
    case: str,
    spanwise_slices: bool,
    use_collocation_points: bool,
    symmetry_condition: bool,
    dict_spanwise_slices: dict,
    ts_start: int,
    nts: int,
    chord: float,
    wing_span: float,
    ea_main: float,
) -> None:
    """
    Computes and exports the induced velocity field from SHARPy simulations.
    Supports structured grid slices and optional airfoil geometry output.

    Results are saved to a single HDF5 file:
        ``{output_folder}/{case}_induced_velocity.h5``

    The file contains:
        - ``velocity_points``: (N, 3) array of field point coordinates [x, z, y].
        - ``u_induced``: (nts, N, 3) array of induced velocity components [x, y, z].
        - ``airfoil_geometry/slice_{i}`` (optional): (nts, 2, n_chord+1) arrays.
        - Attributes: ts_start, nts, nh, nv, n_slices, x_0, x_1, z_0, z_1.

    Args:
        h5_file: Path to the SHARPy .data.h5 result file.
        output_folder: Directory to save the output HDF5 file.
        case: Case name used for output file naming.
        spanwise_slices: Whether to use spanwise slices for field extraction.
        use_collocation_points: If True, use panel collocation points for velocity field.
        symmetry_condition: If True, apply symmetry conditions in UVLM computation.
        dict_spanwise_slices: Dictionary defining the spanwise slice grid and geometry flags.
        ts_start: First timestep to process.
        nts: Number of timesteps to process.
        chord: Wing chord length [m].
        wing_span: Wing half-span [m].
        ea_main: Elastic axis position as fraction of chord from leading edge [-].
    """
    flag_airfoil_geometry = spanwise_slices and dict_spanwise_slices.get(
        "get_airfoil_geometry", False
    )

    list_indices_airfoil_geometry = []
    if spanwise_slices:
        velocity_field_points = get_velocity_field_points(
            chord, wing_span, ea_main, spanwise_slices, dict_spanwise_slices
        )
        if dict_spanwise_slices["get_airfoil_geometry"]:
            with h5.File(h5_file, "r") as f:
                spanwise_beam_coords = np.array(
                    f["data"]["structure"]["timestep_info"]["00000"]["pos"]
                )[2]
            for spanwise_position in dict_spanwise_slices["spanwise_positions"]:
                list_indices_airfoil_geometry.append(
                    get_index_for_spanwise_position(
                        spanwise_beam_coords, spanwise_position
                    )
                )
    else:
        velocity_field_points = get_velocity_field_points(chord, wing_span, ea_main)

    # Keep a reference to the original grid before the loop can overwrite it
    grid_velocity_points = velocity_field_points

    num_points = np.shape(velocity_field_points)[0]
    dimensions = h5_utils.get_aero_dimensions(h5_file)
    gust_vane_surfaces = [1, 2] if symmetry_condition else [2, 3]
    ini_info = AeroTimeStepInfo(
        dimensions[0],
        dimensions[1],
        gust_vane_surfaces=gust_vane_surfaces,
    )

    if use_collocation_points:
        num_points = (dimensions[0][0, 0] + 1) * (dimensions[0][0, 1] + 1)
    matrix_uind = np.zeros((nts, num_points, 3))

    airfoil_geometry_all: Optional[np.ndarray] = None
    wing_corner_points_time_history: Optional[np.ndarray] = None
    if flag_airfoil_geometry:
        wing_corner_points_time_history = load_wing_corner_points_for_time_interval(
            h5_file, ts_start, nts
        )
        assert wing_corner_points_time_history is not None
        n_geom_slices = len(list_indices_airfoil_geometry)
        n_chord_p1 = wing_corner_points_time_history.shape[2]
        airfoil_geometry_all = np.zeros((n_geom_slices, nts, 2, n_chord_p1))

    for counter_ts, ts in enumerate(range(ts_start, ts_start + nts)):
        print("ts: ", ts)
        timestep_info = h5_utils.get_current_aero_ts(h5_file, ini_info, ts)
        if use_collocation_points:
            velocity_field_points = utils.get_corner_points(
                timestep_info.zeta[0], num_points
            )
        matrix_uind[counter_ts, :, :] = (
            uvlmlib.uvlm_calculate_total_induced_velocity_at_points(
                timestep_info,
                velocity_field_points,
                1e-6,
                for_pos=np.zeros((6)),
                ncores=6,
                symmetry_condition=symmetry_condition,
                symmetry_plane=2,
            )
        )

        if flag_airfoil_geometry:
            assert airfoil_geometry_all is not None
            assert wing_corner_points_time_history is not None
            for counter_slice, span_index in enumerate(list_indices_airfoil_geometry):
                airfoil_geometry_all[counter_slice, counter_ts] = (
                    wing_corner_points_time_history[counter_ts, :2, :, span_index]
                )

    h5_out = os.path.join(output_folder, f"{case}_induced_velocity.h5")
    with h5.File(h5_out, "w") as f:
        f.create_dataset(
            "velocity_points",
            data=grid_velocity_points,
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "u_induced", data=matrix_uind, compression="gzip", compression_opts=4
        )
        if flag_airfoil_geometry:
            assert airfoil_geometry_all is not None
            for i, _ in enumerate(list_indices_airfoil_geometry):
                f.create_dataset(
                    f"airfoil_geometry/slice_{i}",
                    data=airfoil_geometry_all[i],
                    compression="gzip",
                    compression_opts=4,
                )
        f.attrs["ts_start"] = ts_start
        f.attrs["nts"] = nts
        f.attrs["nh"] = dict_spanwise_slices.get("n_points_horizontal", 0)
        f.attrs["nv"] = dict_spanwise_slices.get("n_points_vertical", 0)
        f.attrs["n_slices"] = len(dict_spanwise_slices.get("spanwise_positions", []))
        f.attrs["x_0"] = dict_spanwise_slices.get("x_0", 0.0)
        f.attrs["x_1"] = dict_spanwise_slices.get("x_1", 0.0)
        f.attrs["z_0"] = dict_spanwise_slices.get("z_0", 0.0)
        f.attrs["z_1"] = dict_spanwise_slices.get("z_1", 0.0)
    print(f"Saved induced velocity data → {h5_out}")


if __name__ == "__main__":
    chord, wing_span, ea_main = get_pazy_geometry(model_id="delft")

    result_folder = "../output"
    case = "pazy_vertical_case_1_polars0_effcor_0_dynamic_m8_gust_vanes"
    symmetry_condition = True
    use_collocation_points = False
    spanwise_slices = True
    dict_spanwise_slices = {
        "n_points_horizontal": 201,
        "n_points_vertical": 101,
        "spanwise_positions": [0.0],
        "x_0": -0.4,
        "x_1": 0.4,
        "z_0": -0.1,
        "z_1": 0.1,
        "get_airfoil_geometry": True,
    }
    ts_start = 3000
    nts = 900

    output_folder = os.path.join("../results/extracted_data/", case)
    os.makedirs(output_folder, exist_ok=True)
    h5_file = os.path.join(result_folder, case, case, "savedata", case + ".data.h5")

    export_induced_velocities(
        h5_file,
        output_folder,
        case,
        spanwise_slices,
        use_collocation_points,
        symmetry_condition,
        dict_spanwise_slices,
        ts_start,
        nts,
        chord,
        wing_span,
        ea_main,
    )
