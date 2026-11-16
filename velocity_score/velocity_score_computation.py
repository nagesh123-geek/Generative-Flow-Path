import os
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ============================================================
# CONFIGURATION
# ============================================================

RUNS_DIR = "../runs"

DENSITY_DIR = "velocity_score_kl_analysis"

OUTPUT_DIR = "velocity_score_dynamics"

FLOWS = ["linear","trigonometric","spherical","operator"]

ARCHITECTURE = "transformer"

N_PAIRS = 19

N_SAMPLES = 250000

DPI = 600

EPS = 1e-12

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PUBLICATION STYLE
# ============================================================

plt.rcParams.update({

    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],

    "font.size": 14,
    "font.weight": "bold",

    "axes.labelweight": "bold",
    "axes.titleweight": "bold",

    "axes.linewidth": 2.5,

    "xtick.major.width": 2.5,
    "ytick.major.width": 2.5,

    "xtick.major.size": 8,
    "ytick.major.size": 8,

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white"

})


# ============================================================
# HELPER: LOAD TRAJECTORY AND VELOCITY
# ============================================================

def load_dynamics(flow):

    model_dir = os.path.join(RUNS_DIR,f"{flow}_{ARCHITECTURE}")

    trajectory_path = os.path.join(model_dir,f"trajectory_{N_SAMPLES}.npy")

    velocity_path = os.path.join(model_dir,f"velocity_{N_SAMPLES}.npy")

    time_path = os.path.join(model_dir,"trajectory_time.npy")

    for path in [trajectory_path,velocity_path,time_path]:

        if not os.path.exists(path):
            raise FileNotFoundError(path)

    trajectory = np.load( trajectory_path,mmap_mode="r")

    velocity = np.load(velocity_path, mmap_mode="r")

    time = np.load(time_path)

    if trajectory.shape != velocity.shape:

        raise ValueError(f"Trajectory and velocity shape mismatch: "f"{trajectory.shape} vs {velocity.shape}")

    if trajectory.ndim != 3:

        raise ValueError(f"Expected (T,N,D), got {trajectory.shape}")

    if trajectory.shape[2] != 38:

        raise ValueError(f"Expected 38 dimensions, got {trajectory.shape[2]}")

    if trajectory.shape[0] != len(time):

        raise ValueError("Time dimension does not match trajectory.")

    print(f"Trajectory: {trajectory.shape}")
    print(f"Velocity:   {velocity.shape}")
    print(f"Time:       {time.shape}")

    return trajectory, velocity, time


# ============================================================
# HELPER: CALCULATE SPATIAL SCORE DIFFERENCE
# ============================================================

def calculate_score_difference(p_t,p_ref,x_axis,y_axis):

    """
    Calculate:

    Delta s = grad(log(p_t)) - grad(log(p_ref))

    p_t and p_ref have shape (Nx, Ny).

    Returns:
        delta_s_x: (Nx, Ny)
        delta_s_y: (Nx, Ny)
    """

    log_pt = np.log(np.maximum(p_t, EPS))

    log_pref = np.log(np.maximum(p_ref, EPS))

    # Gradient along the two coordinate axes
    grad_pt_x, grad_pt_y = np.gradient(log_pt,x_axis,y_axis,edge_order=2)

    grad_pref_x, grad_pref_y = np.gradient(log_pref,x_axis,y_axis,edge_order=2)

    delta_s_x = grad_pt_x - grad_pref_x

    delta_s_y = grad_pt_y - grad_pref_y

    return delta_s_x, delta_s_y


# ============================================================
# HELPER: INTERPOLATE SCORE AT TRAJECTORY SAMPLES
# ============================================================

def interpolate_score(delta_s_x,delta_s_y,x_axis,y_axis,positions):

    """
    positions: (N,2)

    Returns:
        score_x: (N,)
        score_y: (N,)
    """

    # Keep interpolation coordinates inside the KDE grid.
    x = np.clip(positions[:, 0],x_axis[0],x_axis[-1])

    y = np.clip(positions[:, 1],y_axis[0],y_axis[-1])

    points = np.column_stack([x, y])

    interp_x = RegularGridInterpolator(
        (x_axis, y_axis),
        delta_s_x,
        bounds_error=False,
        fill_value=None)

    interp_y = RegularGridInterpolator(
        (x_axis, y_axis),
        delta_s_y,
        bounds_error=False,
        fill_value=None)

    score_x = interp_x(points)

    score_y = interp_y(points)

    return score_x, score_y


# ============================================================
# PROCESS ONE PAIR
# ============================================================

def analyze_pair(flow,pair,trajectory,velocity,time):

    density_path = os.path.join(
        DENSITY_DIR,
        flow,
        f"pair_{pair:02d}",
        "velocity_score_kl_analysis.npz" )

    if not os.path.exists(density_path):

        print(f"Missing density file: {density_path}")

        return None

    data = np.load(density_path)

    p_t_all = data["p_t"]

    p_ref = data["p_ref"]

    x_axis = data["x_axis"]

    y_axis = data["y_axis"]

    dimensions = data["psi_phi_dimensions"]

    dim_x = int(dimensions[0])

    dim_y = int(dimensions[1])

    T = len(time)

    # --------------------------------------------------------
    # OUTPUT ARRAYS
    # --------------------------------------------------------

    results = {

        "time": time.copy(),

        "v_psi": np.zeros(T),

        "v_phi": np.zeros(T),

        "ds_psi": np.zeros(T),

        "ds_phi": np.zeros(T),

        "vpsi_dspsi": np.zeros(T),

        "vphi_dsphi": np.zeros(T),

        "velocity_score_dot": np.zeros(T),

        "mean_abs_velocity": np.zeros(T),

        "mean_abs_score": np.zeros(T),

        "velocity_rms": np.zeros(T),

        "score_rms": np.zeros(T),

        "psi_dimension": dim_x,

        "phi_dimension": dim_y

    }

    # --------------------------------------------------------
    # TIME LOOP
    # --------------------------------------------------------

    for t in range(T):

        # Density at current time
        p_t = p_t_all[t]

        # Spatial score difference
        delta_s_x, delta_s_y = calculate_score_difference(p_t,p_ref,x_axis,y_axis)

        # Generated positions for this pair
        positions = np.column_stack([trajectory[t, :, dim_x], trajectory[t, :, dim_y]])

        # Velocity components
        v_x = np.asarray(velocity[t, :, dim_x])

        v_y = np.asarray(velocity[t, :, dim_y])

        # Score difference evaluated at samples
        ds_x, ds_y = interpolate_score( delta_s_x,delta_s_y,x_axis,y_axis,positions)

        # ----------------------------------------------------
        # MEAN VELOCITY COMPONENTS
        # ----------------------------------------------------

        results["v_psi"][t] = np.mean(v_x)

        results["v_phi"][t] = np.mean(v_y)

        # ----------------------------------------------------
        # MEAN SCORE DIFFERENCE COMPONENTS
        # ----------------------------------------------------

        results["ds_psi"][t] = np.mean(ds_x)

        results["ds_phi"][t] = np.mean(ds_y)

        # ----------------------------------------------------
        # VELOCITY-SCORE CONTRIBUTIONS
        # ----------------------------------------------------

        contribution_x = v_x * ds_x

        contribution_y = v_y * ds_y

        results["vpsi_dspsi"][t] = np.mean(contribution_x)

        results["vphi_dsphi"][t] = np.mean(contribution_y)

        results["velocity_score_dot"][t] = np.mean( contribution_x + contribution_y)

        # ----------------------------------------------------
        # MAGNITUDE STATISTICS
        # ----------------------------------------------------

        velocity_magnitude = np.sqrt(v_x**2 + v_y**2)

        score_magnitude = np.sqrt(ds_x**2 + ds_y**2)

        results["mean_abs_velocity"][t] = np.mean(velocity_magnitude)

        results["mean_abs_score"][t] = np.mean(score_magnitude)

        results["velocity_rms"][t] = np.sqrt(np.mean(velocity_magnitude**2))

        results["score_rms"][t] = np.sqrt(np.mean(score_magnitude**2))

        if t % 25 == 0:

            print(f"    Pair {pair:02d} | "
                  f"t = {time[t]:.3f} | "
                f"<v.ds> = "
                f"{results['velocity_score_dot'][t]:.6f}" )

    data.close()

    return results


# ============================================================
# COLOR MAP FOR PAIRS
# ============================================================

cmap = plt.cm.turbo

norm = Normalize(vmin=1,vmax=N_PAIRS)


# ============================================================
# PLOT ALL PAIRS FOR ONE FLOW
# ============================================================

def plot_flow(flow, all_results):

    output_flow = os.path.join(OUTPUT_DIR,flow)

    os.makedirs(output_flow, exist_ok=True)

    # Five panels
    quantities = [

        ("v_psi", r"$\langle v_\psi\rangle$", "Velocity"),

        ("v_phi", r"$\langle v_\phi\rangle$", "Velocity"),

        ("ds_psi", r"$\langle\Delta s_\psi\rangle$", "Score difference"),

        ("ds_phi", r"$\langle\Delta s_\phi\rangle$", "Score difference"),

        ("velocity_score_dot",
         r"$\langle\mathbf v\cdot\Delta\mathbf s\rangle$",
         "Velocity-score contribution")

    ]

    fig, axes = plt.subplots(3,2,figsize=(17, 15), sharex=True)

    axes = axes.flatten()

    for idx, (key, ylabel, title) in enumerate(quantities):

        ax = axes[idx]

        for pair in range(1, N_PAIRS + 1):

            if pair not in all_results:
                continue

            result = all_results[pair]

            color = cmap(norm(pair))

            ax.plot(
                result["time"],
                result[key],
                color=color,
                linewidth=2.0,
                alpha=0.9 )

        ax.set_ylabel(
            ylabel,
            fontsize=16,
            fontweight="bold")

        ax.set_title(
            title,
            fontsize=18,
            fontweight="bold")

        ax.tick_params(
            labelsize=13,
            width=2,
            length=7 )

        for spine in ax.spines.values():
            spine.set_linewidth(2.2)

        ax.grid(False)

    # Sixth panel unused
    axes[-1].axis("off")

    # --------------------------------------------------------
    # COMMON X LABEL
    # --------------------------------------------------------

    for ax in axes[:-1]:

        ax.set_xlabel(
            "Flow time",
            fontsize=16,
            fontweight="bold")

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

    # --------------------------------------------------------
    # PAIR COLOR BAR
    # --------------------------------------------------------

    scalar_map = ScalarMappable(
        norm=norm,
        cmap=cmap )

    cbar = fig.colorbar(
        scalar_map,
        ax=axes[:-1],
        fraction=0.025,
        pad=0.025)

    cbar.set_label(
        "ψ–φ Pair",
        fontsize=16,
        fontweight="bold)

    cbar.set_ticks(np.arange(1, N_PAIRS + 1))

    cbar.set_ticklabels([f"{i:02d}" for i in range(1, N_PAIRS + 1)])

    cbar.ax.tick_params(labelsize=11,width=2)

    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    fig.suptitle(
        f"{flow.capitalize()} Flow: Velocity and Score Dynamics",
        fontsize=23,
        fontweight="bold",
        y=0.99 )

    plt.tight_layout(rect=[0, 0, 0.94, 0.96] )

    fig.savefig(
        os.path.join(output_flow,f"{flow}_velocity_score_dynamics.png" ),dpi=DPI,bbox_inches="tight" )

    fig.savefig(os.path.join(output_flow,f"{flow}_velocity_score_dynamics.pdf"),bbox_inches="tight" )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

for flow in FLOWS:

    print("\n" + "=" * 70)
    print(f"FLOW: {flow.upper()}")
    print("=" * 70)

    trajectory, velocity, time = load_dynamics(flow)

    all_results = {}

    # --------------------------------------------------------
    # ALL 19 PAIRS
    # --------------------------------------------------------

    for pair in range(1, N_PAIRS + 1):

        print(f"\nAnalyzing pair {pair:02d}")

        result = analyze_pair(flow,pair,trajectory,velocity,time)

        if result is None:
            continue

        all_results[pair] = result

        # ----------------------------------------------------
        # SAVE INDIVIDUAL PAIR
        # ----------------------------------------------------

        pair_dir = os.path.join( OUTPUT_DIR,flow,f"pair_{pair:02d}")

        os.makedirs(pair_dir, exist_ok=True)

        np.savez_compressed(
            os.path.join( pair_dir, "velocity_score_dynamics.npz"),**result )

    # --------------------------------------------------------
    # SAVE COMBINED FLOW DATA
    # --------------------------------------------------------

    if len(all_results) > 0:

        combined = {}

        for pair, result in all_results.items():

            for key, value in result.items():

                combined[ f"pair_{pair:02d}_{key}"] = value

        combined_path = os.path.join(OUTPUT_DIR,flow,f"{flow}_all_pairs_velocity_score.npz")

        np.savez_compressed(combined_path,**combined)

        print(f"\nSaved combined data: {combined_path}")

        # ----------------------------------------------------
        # PLOT
        # ----------------------------------------------------

        plot_flow(flow, all_results)

        print(f"Plots saved for {flow}")

    del trajectory
    del velocity

# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("VELOCITY AND SCORE ANALYSIS COMPLETED")
print("=" * 70)

print(f"Output directory: {OUTPUT_DIR}")

print("Flows:", FLOWS)

print("Pairs per flow:", N_PAIRS)

print("Expected figures: 4")

print("=" * 70)
