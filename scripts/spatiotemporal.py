#!/usr/bin/env python
"""
Spatiotemporal comparison of Schrodinger bridge transport across four cases.

Reads the NetCDF output of the `anyakrakusuma` entropic optimal transport
solver for four scenarios, renders a single 4x3 multipanel figure comparing
the start, middle, and end of each bridge trajectory, and writes a plain-text
statistics report with all numbers needed for the manuscript.

Layout
    rows    : four transport cases
    columns : interpolation snapshots at t = 0, t ~ 0.5, t = 1
    panels  : labelled (a) - (l), no titles or subtitles
    legend  : one shared legend below the grid

Expected directory layout (script is run from the `scripts` folder)
    ../raw_data      input NetCDF files
    ../calculations  text statistics report
    ../figures       PDF / PNG / EPS figures

Author: Sandy H. S. Herho <sandy.herho@email.ucr.edu>
Date: 2026-07-06
"""

import matplotlib
matplotlib.use("Agg")

from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from netCDF4 import Dataset


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RAW_DATA_DIR = Path("../raw_data")
CALC_DIR = Path("../calculations")
FIG_DIR = Path("../figures")

FIG_STEM = "spatiotemporal_comparison"
REPORT_NAME = "spatiotemporal_statistics_report.txt"

CASES = [
    {"file": "case1_circle_to_circle.nc",           "label": "Circle to Circle"},
    {"file": "case2_spiral_to_gaussian_mixture.nc", "label": "Spiral to Gaussian Mixture"},
    {"file": "case3_moons_to_moons_rotated.nc",     "label": "Two Moons to Rotated Two Moons"},
    {"file": "case4_lissajous_to_trefoil.nc",       "label": "Lissajous to Trefoil"},
]

# Colours (muted, print friendly, white background)
C_SOURCE = "#4C72B0"   # source cloud mu
C_TARGET = "#C44E52"   # target cloud nu
C_BRIDGE = "#333333"   # evolving bridge state rho_t

DPI_RASTER = 600
FIG_SIZE = (7.2, 9.6)


# --------------------------------------------------------------------------- #
# Matplotlib style
# --------------------------------------------------------------------------- #
def configure_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.labelsize": 11,
    })


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_case(path):
    """Read the arrays and metadata needed from one NetCDF file."""
    with Dataset(path, "r") as nc:
        data = {
            "time": np.asarray(nc.variables["time"][:], dtype=float),
            "source": np.asarray(nc.variables["X_source"][:], dtype=float),
            "target": np.asarray(nc.variables["X_target"][:], dtype=float),
            "trajectory": np.asarray(nc.variables["trajectory"][:], dtype=float),
        }
        meta_keys = [
            "epsilon", "transport_cost", "converged", "n_iterations",
            "plan_entropy", "effective_sparsity", "marginal_fidelity",
            "bridge_diffusivity", "source_type", "target_type",
        ]
        data["meta"] = {k: getattr(nc, k, None) for k in meta_keys}
    return data


def cloud_at_time(trajectory, time, t_star):
    """Return the point cloud at an exact interpolation time t_star.

    If a stored frame coincides with t_star it is returned directly; otherwise
    the cloud is linearly interpolated between the two straddling frames so the
    snapshot lands on t_star exactly. This lets the middle column be evaluated
    at the same t for every case regardless of the frame grid. The endpoints
    t = 0 and t = 1 always coincide with the first and last stored frames.
    """
    time = np.asarray(time, dtype=float)
    exact = np.where(np.isclose(time, t_star))[0]
    if exact.size:
        idx = int(exact[0])
        return trajectory[idx].copy(), float(time[idx])

    k = int(np.clip(np.searchsorted(time, t_star), 1, len(time) - 1))
    t0, t1 = time[k - 1], time[k]
    w = (t_star - t0) / (t1 - t0)
    cloud = (1.0 - w) * trajectory[k - 1] + w * trajectory[k]
    return cloud, float(t_star)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def cloud_stats(points):
    """Basic descriptive statistics for an (N, 2) point cloud."""
    centroid = points.mean(axis=0)
    std = points.std(axis=0)
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    prange = pmax - pmin
    disp = float(np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1))))
    return {
        "n": points.shape[0],
        "mean_x": float(centroid[0]), "mean_y": float(centroid[1]),
        "std_x": float(std[0]), "std_y": float(std[1]),
        "min_x": float(pmin[0]), "max_x": float(pmax[0]),
        "min_y": float(pmin[1]), "max_y": float(pmax[1]),
        "range_x": float(prange[0]), "range_y": float(prange[1]),
        "rms_dispersion": disp,
        "bbox_area": float(prange[0] * prange[1]),
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):d}"
    return f"{float(value):.6f}"


def write_report(cases_data, snapshot_labels, out_path):
    """Assemble the plain-text statistics report."""
    row_order = [
        ("N points", "n"),
        ("Centroid x", "mean_x"),
        ("Centroid y", "mean_y"),
        ("Std x", "std_x"),
        ("Std y", "std_y"),
        ("Min x", "min_x"),
        ("Max x", "max_x"),
        ("Min y", "min_y"),
        ("Max y", "max_y"),
        ("Range x", "range_x"),
        ("Range y", "range_y"),
        ("RMS dispersion", "rms_dispersion"),
        ("Bounding-box area", "bbox_area"),
    ]

    bar = "=" * 76
    sub = "-" * 76
    lines = []
    lines.append(bar)
    lines.append(" SPATIOTEMPORAL COMPARISON OF SCHRODINGER BRIDGE TRANSPORT")
    lines.append(" Entropic optimal transport (anyakrakusuma) trajectory statistics")
    lines.append(sub)
    lines.append(f" Author    : Sandy H. S. Herho <sandy.herho@email.ucr.edu>")
    lines.append(f" Generated : {datetime.now().isoformat(timespec='seconds')}")
    lines.append(bar)
    lines.append("")
    lines.append("INPUT FILES")
    lines.append(sub)
    for i, case in enumerate(cases_data, 1):
        lines.append(f"  [{i}] {case['file']}   ({case['label']})")
    lines.append("")

    for i, case in enumerate(cases_data, 1):
        meta = case["data"]["meta"]
        stats = case["stats"]
        t_vals = case["t_values"]

        lines.append(bar)
        lines.append(f" CASE {i}: {case['label']}")
        lines.append(f" File: {case['file']}")
        lines.append(sub)
        lines.append(" Solver / coupling metadata")
        lines.append(f"   Source type                             = {meta['source_type']}")
        lines.append(f"   Target type                             = {meta['target_type']}")
        lines.append(f"   Entropic regularization  epsilon        = {_fmt(meta['epsilon'])}")
        lines.append(f"   Wasserstein cost         <C,pi>         = {_fmt(meta['transport_cost'])}")
        conv = "n/a" if meta["converged"] is None else ("True" if int(meta["converged"]) else "False")
        lines.append(f"   Converged                               = {conv}")
        lines.append(f"   Sinkhorn iterations                     = {_fmt(meta['n_iterations'])}")
        lines.append(f"   Plan entropy             H(pi)          = {_fmt(meta['plan_entropy'])}")
        lines.append(f"   Effective sparsity       exp(H)/(n m)   = {_fmt(meta['effective_sparsity'])}")
        lines.append(f"   Marginal fidelity                       = {_fmt(meta['marginal_fidelity'])}")
        lines.append(f"   Bridge diffusivity       epsilon/6      = {_fmt(meta['bridge_diffusivity'])}")
        lines.append("")
        lines.append(" Snapshot statistics (point cloud, N x 2)")
        lines.append(" " + sub[:75])

        header = f"   {'Quantity':<20}"
        for lab, tv in zip(snapshot_labels, t_vals):
            col = f"{lab} (t={tv:.3f})"
            header += f"{col:>18}"
        lines.append(header)
        lines.append(" " + sub[:75])

        for label, key in row_order:
            row = f"   {label:<20}"
            for s in stats:
                if key == "n":
                    row += f"{s[key]:>18d}"
                else:
                    row += f"{s[key]:>18.6f}"
            lines.append(row)
        lines.append("")

    # Compact citeable summary
    lines.append(bar)
    lines.append(" SUMMARY TABLE")
    lines.append(sub)
    head = (f"   {'Case':<8}{'epsilon':>10}{'W-cost':>12}{'H(pi)':>12}"
            f"{'EffSparse':>12}{'RMS(0)':>10}{'RMS(mid)':>10}{'RMS(1)':>10}")
    lines.append(head)
    lines.append(" " + sub[:75])
    for i, case in enumerate(cases_data, 1):
        meta = case["data"]["meta"]
        s0, sm, s1 = case["stats"]
        lines.append(
            f"   {('C' + str(i)):<8}"
            f"{float(meta['epsilon']):>10.4f}"
            f"{float(meta['transport_cost']):>12.6f}"
            f"{float(meta['plan_entropy']):>12.6f}"
            f"{float(meta['effective_sparsity']):>12.6f}"
            f"{s0['rms_dispersion']:>10.4f}"
            f"{sm['rms_dispersion']:>10.4f}"
            f"{s1['rms_dispersion']:>10.4f}"
        )
    lines.append(bar)
    lines.append("")

    out_path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def square_limits(points, margin=0.08):
    """Centred square axis limits so equal-aspect panels are not distorted."""
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    centre = 0.5 * (pmin + pmax)
    half = 0.5 * float(np.max(pmax - pmin)) * (1.0 + margin)
    half = max(half, 1e-6)
    return (centre[0] - half, centre[0] + half), (centre[1] - half, centre[1] + half)


def make_figure(cases_data, out_stem):
    letters = [f"({c})" for c in "abcdefghijkl"]
    n_rows, n_cols = len(cases_data), 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=FIG_SIZE)

    k = 0
    for r, case in enumerate(cases_data):
        src = case["data"]["source"]
        tgt = case["data"]["target"]
        traj = case["data"]["trajectory"]
        clouds = case["clouds"]

        all_pts = np.vstack([src, tgt, traj.reshape(-1, 2)])
        xlim, ylim = square_limits(all_pts)

        for c, pts in enumerate(clouds):
            ax = axes[r, c]

            # reference clouds (context in every panel)
            ax.scatter(src[:, 0], src[:, 1], s=8, c=C_SOURCE,
                       alpha=0.22, linewidths=0)
            ax.scatter(tgt[:, 0], tgt[:, 1], s=8, c=C_TARGET,
                       alpha=0.22, linewidths=0)

            # evolving bridge state (evaluated at exactly t = 0, 0.5, 1)
            ax.scatter(pts[:, 0], pts[:, 1], s=12, c=C_BRIDGE,
                       alpha=0.75, linewidths=0)

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))

            ax.text(0.5, 1.03, letters[k], transform=ax.transAxes,
                    va="bottom", ha="center", fontweight="bold", fontsize=11)

            if r == n_rows - 1:
                ax.set_xlabel(r"$x$")
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(r"$y$")
            else:
                ax.set_yticklabels([])

            k += 1

    handles = [
        Line2D([], [], linestyle="None", marker="o", markerfacecolor=C_SOURCE,
               markeredgecolor="none", markersize=7, label=r"Source $\mu$ ($t=0$)"),
        Line2D([], [], linestyle="None", marker="o", markerfacecolor=C_TARGET,
               markeredgecolor="none", markersize=7, label=r"Target $\nu$ ($t=1$)"),
        Line2D([], [], linestyle="None", marker="o", markerfacecolor=C_BRIDGE,
               markeredgecolor="none", markersize=7, label=r"Bridge state $\rho_t$"),
    ]

    fig.tight_layout(rect=(0.0, 0.045, 1.0, 1.0), h_pad=1.8)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.006), fontsize=10,
               handletextpad=0.3, columnspacing=1.8)

    for ext in ("pdf", "png", "eps"):
        fig.savefig(f"{out_stem}.{ext}", dpi=DPI_RASTER,
                    bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    configure_style()
    CALC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_labels = ["Start", "Middle", "End"]
    snapshot_times = [0.0, 0.5, 1.0]
    cases_data = []

    for case in CASES:
        path = RAW_DATA_DIR / case["file"]
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

        data = load_case(path)
        clouds, t_values = [], []
        for t_star in snapshot_times:
            cloud, t_used = cloud_at_time(data["trajectory"], data["time"], t_star)
            clouds.append(cloud)
            t_values.append(t_used)
        stats = [cloud_stats(c) for c in clouds]

        cases_data.append({
            "file": case["file"],
            "label": case["label"],
            "data": data,
            "clouds": clouds,
            "t_values": t_values,
            "stats": stats,
        })
        print(f"Loaded {case['file']}: {data['trajectory'].shape[1]} particles, "
              f"{data['trajectory'].shape[0]} frames")

    report_path = CALC_DIR / REPORT_NAME
    write_report(cases_data, snapshot_labels, report_path)
    print(f"Report written : {report_path}")

    fig_stem = FIG_DIR / FIG_STEM
    make_figure(cases_data, str(fig_stem))
    print(f"Figures written: {fig_stem}.pdf / .png / .eps")


if __name__ == "__main__":
    main()