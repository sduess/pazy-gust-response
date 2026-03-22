import numpy as np
import matplotlib.pyplot as plt

import utils.experimental_data as reference_utils
def get_label(alpha: float, use_polars: bool, use_eff_correction: bool) -> str:
    """
    Constructs a label string based on aerodynamic correction settings.

    Args:
        alpha (float): Angle of attack in degrees.
        use_polars (bool): Whether polar correction is used.
        use_eff_correction (bool): Whether lift slope correction is used.

    Returns:
        str: Label string for plotting or legend.
    """
    label_str = r'$\alpha$ = {} deg '.format(alpha)
    if use_polars:
        label_str += ' + Polar Correction'
    elif use_eff_correction:
        label_str += r' + $c_{l_\alpha}$ Correction'
    return label_str

def get_linestyle(use_polars: bool, use_eff_correction: bool) -> str:
    """
    Returns a matplotlib line style based on aerodynamic correction usage.

    Args:
        use_polars (bool): Whether polar correction is enabled.
        use_eff_correction (bool): Whether lift slope correction is enabled.

    Returns:
        str: Matplotlib-compatible line style string.
    """
    list_linestyles = ['solid', 'dashed', 'dotted']
    if use_polars:
        idx = 1
    elif use_eff_correction:
        idx = 2
    else:
        idx = 0
    return list_linestyles[idx]
    
def get_color_by_index(index: int) -> str:
    """
    Returns a predefined matplotlib-compatible color string based on the given index.

    Args:
        index (int): Index of the color to retrieve (0-based).

    Returns:
        str: Color string (e.g., 'tab:blue') compatible with matplotlib.

    Raises:
        IndexError: If the index is out of range of the color list.
    """
    list_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    return list_colors[index]


def plot_time_history_of_structural_deformation_and_gust_velocity(
    position: float,
    deformation_at_position: np.ndarray,
    dt: float,
    gust_quarter_chord: np.ndarray
) -> None:
    """
    Plots the full time history of structural deformation and gust velocity.

    Args:
        position (float): Normalized y/b position.
        deformation_at_position (np.ndarray): Deformation signal.
        dt (float): Time step.
        gust_quarter_chord (np.ndarray): Gust velocity history.
    """
    plt.figure()
    plt.title(f"Time history of wing deformation @ y/b = {position:.2f}")
    plt.plot(np.arange(0,len(deformation_at_position))*dt,deformation_at_position)
    plt.plot(np.arange(0,len(deformation_at_position))*dt,gust_quarter_chord[:len(deformation_at_position)]+6)
    plt.grid()
    plt.xlabel('time, s')
    plt.ylabel(r'$\Delta$ z/b, %')
    plt.show()

def plot_period_of_structural_deformation_and_gust(
    period_of_oscillation: np.ndarray,
    period_of_gust_velocity: np.ndarray,
    position: float,
    wing_halfspan: float,icase: int,
    experimental_data: str
) -> None:
    """
    Plots one period of structural deformation and gust velocity with optional reference.

    Args:
        period_of_oscillation (np.ndarray): Deformation during one oscillation period.
        period_of_gust_velocity (np.ndarray): Gust velocity during one oscillation period.
        position (float): Normalized y/b location.
        wing_halfspan (float): Wing half-span.
    """
    fig, ax1 = plt.subplots()
    # ax1.title(f"Time period of wing deformation @ y/b = {position:.2f}")
    ax1.plot(np.arange(0,1,1./len(period_of_oscillation)),period_of_oscillation-np.mean(period_of_oscillation), color='tab:green', linestyle='--')
    ax1.set_xlabel('t/T')
    ax1.set_ylabel(r'$\Delta$ z/b, %')
    ax1.set_xlim([0,1])
    ax2 = ax1.twinx()
    ax2.set_ylabel('gust velocity, m/s')
    ax2.plot(np.arange(0,1,1./len(period_of_gust_velocity)),period_of_gust_velocity, color='tab:blue', linestyle='-')
    ax2.set_xlim([0,1])
    fig.tight_layout()
    reference_deformation = reference_utils.get_reference_deformation_dynamic(icase, position, wing_halfspan, experimental_data)
    ax1.plot(np.arange(0,1,1./len(reference_deformation)), reference_deformation-np.mean(reference_deformation), color='tab:green', linestyle='-')
    plt.show()