"""Plotting tools for induced velocity field data from export_induced_velocity.py.

Data layout (HDF5 file: {case}_induced_velocity.h5)
----------------------------------------------------
velocity_points  — dataset, shape (N, 3), columns [x, z, y]
    Chordwise, vertical, spanwise.  Column ordering is non-standard (z before y).
    Points are ordered: outer-loop x (nh), inner-loop z (nv), per spanwise slice.

u_induced  — dataset, shape (nts, N, 3)
    Last axis: component index 0=x, 1=y, 2=z.

airfoil_geometry/slice_{i}  — dataset, shape (nts, 2, n_chord+1)
    axis-1 index 0 → x-coordinates along the chord.
    axis-1 index 1 → vertical (z) coordinates along the chord.
    Note: SHARPy zeta[:2] maps to (x, z) for this wing orientation.

Attributes on the root group: ts_start, nts, nh, nv, n_slices,
    x_0, x_1, z_0, z_1.

Grid structure (spanwise_slices=True)
--------------------------------------
    Total points  = n_slices × nh × nv
    For slice s, point (i, j): global index = s*nh*nv + i*nv + j
      where i ∈ [0, nh), j ∈ [0, nv).
"""

import os

_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
_PROJECT_ROOT = os.path.dirname(_DIR)
from pathlib import Path
from typing import Optional

import h5py as h5
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# ── column mapping in velocity_points (columns are [x, z, y]) ────────────────
_VP_COL: dict[str, int] = {"x": 0, "z": 1, "y": 2}
# component → index in u_induced last axis
_COMP_IDX: dict[str, int] = {"x": 0, "y": 1, "z": 2}


# ─────────────────────────────── I/O helpers ─────────────────────────────────


def _h5_path(data_dir: str, case: str) -> str:
    """Return the path to the induced-velocity HDF5 file."""
    return os.path.join(data_dir, f"{case}_induced_velocity.h5")


def load_velocity_points(data_dir: str, case: str) -> np.ndarray:
    """Load velocity field point coordinates from the HDF5 file.

    Returns:
        Array of shape (N, 3) with columns [x, z, y].
    """
    with h5.File(_h5_path(data_dir, case), "r") as f:
        return f["velocity_points"][:]


def load_induced_velocity(
    data_dir: str,
    case: str,
    component: str,
) -> np.ndarray:
    """Load the induced velocity for one component from the HDF5 file.

    Args:
        data_dir: Directory containing the HDF5 file.
        case: Case name prefix.
        component: Velocity component ('x', 'y', or 'z').

    Returns:
        Array of shape (nts, n_points) — rows = timestep, columns = point index.
    """
    idx = _COMP_IDX[component]
    with h5.File(_h5_path(data_dir, case), "r") as f:
        return f["u_induced"][:, :, idx]


def load_airfoil_geometry(data_dir: str, case: str, slice_idx: int) -> np.ndarray:
    """Load airfoil geometry for a spanwise slice from the HDF5 file.

    Args:
        data_dir: Directory containing the HDF5 file.
        case: Case name prefix.
        slice_idx: Slice index (0-based).

    Returns:
        Array of shape (nts, 2, n_chord+1).
        ``arr[t, 0, :]`` → x-coordinates along the chord at timestep t.
        ``arr[t, 1, :]`` → z-coordinates along the chord at timestep t.
    """
    with h5.File(_h5_path(data_dir, case), "r") as f:
        return f[f"airfoil_geometry/slice_{slice_idx}"][:]


# ─────────────────────────────── grid helpers ────────────────────────────────


def _get_grid_attrs(data_dir: str, case: str) -> tuple[int, int, int]:
    """Read nh, nv, n_slices from HDF5 file attributes.

    Returns:
        Tuple (nh, nv, n_slices).
    """
    with h5.File(_h5_path(data_dir, case), "r") as f:
        return int(f.attrs["nh"]), int(f.attrs["nv"]), int(f.attrs["n_slices"])


def _get_meshgrid_for_slice(
    velocity_points: np.ndarray,
    slice_idx: int,
    nh: int,
    nv: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return X, Z meshgrid arrays of shape (nv, nh) for plotting.

    The transpose puts z on the row axis (suitable for pcolormesh/contourf
    with x on the horizontal axis and z on the vertical axis).
    """
    start = slice_idx * nh * nv
    end = start + nh * nv
    x_flat = velocity_points[start:end, _VP_COL["x"]].reshape(nh, nv)
    z_flat = velocity_points[start:end, _VP_COL["z"]].reshape(nh, nv)
    # Transpose so axis-0 = z (nv rows) and axis-1 = x (nh cols)
    return x_flat.T, z_flat.T  # each shape (nv, nh)


# ─────────────────────────────── Graph 1 ─────────────────────────────────────


def plot_1d_gust_amplitude(
    data_dir: str,
    case: str,
    component: str,
    scan_axis: str,
    fixed_coords: dict[str, float],
    *,
    scan_range: Optional[tuple[float, float]] = None,
    chord: Optional[float] = None,
    d_v: Optional[float] = None,
    timestep: Optional[int] = None,
    amplitude_mode: str = "peak",
    tol: float = 1e-6,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot the induced velocity component along a 1-D line.

    Selects all grid points where two coordinates are fixed and plots the
    selected velocity component vs the remaining (scan) coordinate.

    Args:
        data_dir: Directory containing the HDF5 file.
        case: Case name prefix.
        component: Velocity component to plot ('x', 'y', or 'z').
        scan_axis: The coordinate that varies ('x', 'y', or 'z').
        fixed_coords: Dict of the two fixed coordinate values,
                      e.g. ``{'y': 0.0, 'z': 0.0}``.
        scan_range: Optional ``(min, max)`` limits on the scan coordinate
                    in **physical units** (applied before normalisation).
        chord: If given, normalise the x-coordinate by this value and label
               the axis as ``x/c``.
        d_v: If given, normalise the z-coordinate by this value and label
             the axis as ``z/d_v``.
        timestep: Timestep index (0-based within the exported range).
                  Required when ``amplitude_mode='timestep'``.
        amplitude_mode: How to reduce over time:
            ``'timestep'``  — value at *timestep*;
            ``'amplitude'`` — half peak-to-peak ``(max − min) / 2`` over all timesteps;
            ``'peak'``      — max absolute value over all timesteps;
            ``'rms'``       — RMS over all timesteps.
        tol: Absolute tolerance for coordinate matching.
        ax: Axes to plot into; a new figure is created if None.
        label: Legend label for the plotted line.
        save_path: If given, save the figure here.

    Returns:
        The :class:`~matplotlib.axes.Axes` used.
    """
    if scan_axis not in _VP_COL:
        raise ValueError(f"scan_axis must be one of {list(_VP_COL)}, got '{scan_axis}'")

    velocity_points = load_velocity_points(data_dir, case)
    u_ind = load_induced_velocity(data_dir, case, component)

    # ── find matching point indices ───────────────────────────────────────────
    mask = np.ones(velocity_points.shape[0], dtype=bool)
    for ax_name, val in fixed_coords.items():
        mask &= np.abs(velocity_points[:, _VP_COL[ax_name]] - val) < tol

    indices = np.where(mask)[0]
    if indices.size == 0:
        raise ValueError(
            f"No points match fixed_coords={fixed_coords} (tol={tol}). "
            "Check coordinate values or increase tol."
        )

    coord_vals = velocity_points[indices, _VP_COL[scan_axis]]
    order = np.argsort(coord_vals)
    indices, coord_vals = indices[order], coord_vals[order]

    if scan_range is not None:
        range_mask = (coord_vals >= scan_range[0]) & (coord_vals <= scan_range[1])
        indices, coord_vals = indices[range_mask], coord_vals[range_mask]
        if indices.size == 0:
            raise ValueError(
                f"No points remain after applying scan_range={scan_range}."
            )

    # ── reduce over time ──────────────────────────────────────────────────────
    vel_line = u_ind[:, indices]  # (nts, n_line_points)

    if amplitude_mode == "timestep":
        if timestep is None:
            raise ValueError("'timestep' must be given when amplitude_mode='timestep'.")
        vel_plot = vel_line[timestep, :]
        ylabel = f"$u_{{{component}}}$ [m/s] at ts={timestep}"
    elif amplitude_mode == "amplitude":
        vel_plot = (np.max(vel_line, axis=0) - np.min(vel_line, axis=0)) / 2.0
        ylabel = f"Amplitude $u_{{{component}}}$ [m/s]"
    elif amplitude_mode == "peak":
        vel_plot = np.max(np.abs(vel_line), axis=0)
        ylabel = f"Peak $|u_{{{component}}}|$ [m/s]"
    elif amplitude_mode == "rms":
        vel_plot = np.sqrt(np.mean(vel_line**2, axis=0))
        ylabel = f"RMS $u_{{{component}}}$ [m/s]"
    else:
        raise ValueError(f"Unknown amplitude_mode '{amplitude_mode}'.")

    # ── non-dimensionalise scan coordinate ───────────────────────────────────
    _scale = {"x": chord, "z": d_v}
    scale = _scale.get(scan_axis)
    if scale is not None:
        coord_plot = coord_vals / scale
        xlabel = r"$x/c$" if scan_axis == "x" else r"$z/d_\mathrm{v}$"
    else:
        coord_plot = coord_vals
        xlabel = f"${scan_axis}$ [m]"

    # ── plot ──────────────────────────────────────────────────────────────────
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(coord_plot, vel_plot, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"Induced $u_{{{component}}}$ along {scan_axis} — "
        + " @".join(f"{k}={v:.2f} m" for k, v in fixed_coords.items())
    )
    ax.grid(True)
    if label:
        ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


# ─────────────────────────────── Graph 2 ─────────────────────────────────────


def animate_velocity_field_in_plane(
    data_dir: str,
    case: str,
    velocity_component: str,
    slice_idx: int = 0,
    *,
    chord: Optional[float] = None,
    d_v: Optional[float] = None,
    airfoil_slice_idx: Optional[int] = None,
    ts_range: Optional[tuple[int, int]] = None,
    cmap: str = "RdBu_r",
    clim: Optional[tuple[float, float]] = None,
    colorbar_location: str = "right",
    fps: int = 10,
    save_path: Optional[str] = None,
) -> animation.FuncAnimation:
    """Animate the induced velocity in the x-z plane for one spanwise slice.

    Each frame shows a pcolormesh plot of the chosen velocity component.
    If airfoil geometry was exported it is overlaid on every frame.

    Args:
        data_dir: Directory containing the HDF5 file.
        case: Case name prefix.
        velocity_component: Component to animate ('x', 'y', or 'z').
        slice_idx: Index of the spanwise slice to show (0-based).
        chord: If given, normalise x-coordinates by this value and label
               the axis as ``x/c``.
        d_v: If given, normalise z-coordinates by this value and label
             the axis as ``z/d_v``.
        airfoil_slice_idx: If not None, overlay airfoil geometry for this
                           slice index.  Pass *slice_idx* to match the plane.
        ts_range: ``(first, last)`` timestep indices (exclusive on right) to
                  animate.  None uses all exported timesteps.
        cmap: Matplotlib colormap.
        clim: ``(vmin, vmax)`` for the colour scale.  If None the global
              symmetric extremum over all frames is used.
        colorbar_location: Where to place the colorbar: ``'right'`` (default,
                           vertical), ``'left'`` (vertical), ``'top'``
                           (horizontal), or ``'bottom'`` (horizontal).
        fps: Frames per second.
        save_path: Path to save the animation ('.gif' requires Pillow,
                   '.mp4' requires ffmpeg).

    Returns:
        The :class:`~matplotlib.animation.FuncAnimation` object.
    """
    nh, nv, _ = _get_grid_attrs(data_dir, case)
    velocity_points = load_velocity_points(data_dir, case)
    u_ind = load_induced_velocity(data_dir, case, velocity_component)

    nts_total = u_ind.shape[0]
    ts_start, ts_end = (0, nts_total) if ts_range is None else ts_range
    n_frames = ts_end - ts_start

    X, Z = _get_meshgrid_for_slice(velocity_points, slice_idx, nh, nv)
    # X, Z shape: (nv, nh)

    # ── non-dimensionalise spatial grids ─────────────────────────────────────
    x_scale = chord if chord is not None else 1.0
    z_scale = d_v if d_v is not None else 1.0
    X = X / x_scale
    Z = Z / z_scale
    xlabel = r"$x/c$" if chord is not None else r"$x$ [m]"
    zlabel = r"$z/d_\mathrm{v}$" if d_v is not None else r"$z$ [m]"

    start_pt = slice_idx * nh * nv
    end_pt = start_pt + nh * nv
    # vel: (n_frames, nv, nh) — transpose so rows=z, cols=x
    vel = (
        u_ind[ts_start:ts_end, start_pt:end_pt]
        .reshape(n_frames, nh, nv)
        .transpose(0, 2, 1)
    )

    if clim is not None:
        vmin, vmax = clim
    else:
        vabs = max(abs(float(vel.min())), abs(float(vel.max())))
        vmin, vmax = -vabs, vabs

    # ── optional airfoil geometry ─────────────────────────────────────────────
    airfoil_data = None
    if airfoil_slice_idx is not None:
        try:
            airfoil_data = load_airfoil_geometry(data_dir, case, airfoil_slice_idx)
        except (FileNotFoundError, KeyError):
            print(
                f"Warning: airfoil geometry for slice {airfoil_slice_idx} not found."
                " Airfoil overlay disabled."
            )

    # ── figure setup ─────────────────────────────────────────────────────────
    cbar_is_horizontal = colorbar_location in ("top", "bottom")
    fig, ax = plt.subplots(figsize=(9, 5) if not cbar_is_horizontal else (9, 6))
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(zlabel)

    # pcolormesh is used instead of contourf: it supports in-place data updates
    # via set_array() and has a stable API across all matplotlib versions.
    pc = ax.pcolormesh(X, Z, vel[0], cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    cbar_orientation = "horizontal" if cbar_is_horizontal else "vertical"
    cbar = fig.colorbar(
        pc,
        ax=ax,
        orientation=cbar_orientation,
        location=colorbar_location,
        label=f"$u_{{{velocity_component}}}$ [m/s]",
    )
    cbar.set_ticks(np.linspace(vmin, vmax, 5))

    (airfoil_line,) = ax.plot([], [], "k-", lw=1.5, label="airfoil")
    if airfoil_data is not None:
        ax.legend(loc="upper right", fontsize=8)

    title = ax.set_title("")

    def _update(frame: int):
        pc.set_array(vel[frame].ravel())
        title.set_text(
            f"Induced $u_{{{velocity_component}}}$  |  "
            f"slice {slice_idx}  |  timestep {ts_start + frame}"
        )
        if airfoil_data is not None:
            ts_global = ts_start + frame
            airfoil_line.set_data(
                airfoil_data[ts_global, 0, :] / x_scale,  # x/c along chord
                airfoil_data[ts_global, 1, :] / z_scale,  # z/d_v along chord
            )
        return [pc, airfoil_line, title]

    ani = animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=1000 / fps, blit=False
    )

    if save_path:
        ext = Path(save_path).suffix.lower()
        writer = "pillow" if ext == ".gif" else "ffmpeg"
        ani.save(save_path, writer=writer, fps=fps)
        print(f"Saved animation → {save_path}")

    return ani


# ─────────────────────────────── example usage ───────────────────────────────

if __name__ == "__main__":
    data_dir = os.path.join(_PROJECT_ROOT, "results", "extracted_data", "pazy_vertical_case_1_polars0_effcor_0_dynamic_m8_gust_vanes")
    case = "pazy_vertical_case_1_polars0_effcor_0_dynamic_m8_gust_vanes"

    chord = 0.1  # Pazy wing chord [m]
    d_v = 0.3  # distance between gust vanes [m]

    # ── Graph 1: amplitude |uy| along x (upstream half only) at z=0, y=0 ────
    # x-axis is non-dimensionalised by chord (x/c).
    plot_1d_gust_amplitude(
        data_dir=data_dir,
        case=case,
        component="y",
        scan_axis="x",
        fixed_coords={"z": 0.0, "y": 0.0},
        scan_range=(-0.24410, -0.044100),
        chord=chord,
        amplitude_mode="amplitude",
        save_path=os.path.join(_PROJECT_ROOT, "results", "extracted_data", "induced_vel_1d_peak_uz_vs_x.png"),
    )
    plt.show()

    # ── Graph 2: animation of uy in the x-z plane (slice 0) ─────────────────
    # Both axes non-dimensionalised (x/c, z/d_v); airfoil overlay included.
    # Pass clim=(vmin, vmax) to override the default global-extremum colour scale.
    ani = animate_velocity_field_in_plane(
        data_dir=data_dir,
        case=case,
        velocity_component="y",
        slice_idx=0,
        chord=chord,
        d_v=d_v,
        airfoil_slice_idx=0,
        ts_range=(0, 900),
        clim=(-4.0, 4.0),
        colorbar_location="bottom",
        fps=15,
        save_path=os.path.join(_PROJECT_ROOT, "results", "extracted_data", "induced_vel_animation.gif"),
    )
    plt.show()
