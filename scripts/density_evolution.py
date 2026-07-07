#!/usr/bin/env python
"""
Marginal density evolution of the Schrodinger bridge.

For each of the four `anyakrakusuma` cases, the bridge marginal rho_t is
estimated with a Gaussian kernel density estimate at five interpolation times
(t = 0, 0.25, 0.5, 0.75, 1) and displayed as a 4x5 montage. This is the
probability-density counterpart of the sample-scatter figure: it shows where
mass concentrates and rarefies as one distribution morphs into the other.

Output
    ../figures/density_evolution.{pdf,png,eps}   4x5 KDE montage
    ../calculations/density_evolution.txt         numerical report

Everything is derived from the stored trajectory; intermediate times are
obtained by linear interpolation between the two straddling frames.

Author: Sandy H. S. Herho <sandy.herho@email.ucr.edu>
Date: 2026-07-06
"""

import matplotlib
matplotlib.use("Agg")

from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from netCDF4 import Dataset
from scipy.stats import gaussian_kde
from scipy.ndimage import label


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RAW_DATA_DIR = Path("../raw_data")
CALC_DIR = Path("../calculations")
FIG_DIR = Path("../figures")

FIG_STEM = "density_evolution"
REPORT_NAME = "density_evolution.txt"

CASES = [
    {"file": "case1_circle_to_circle.nc",           "label": "Circle to Circle"},
    {"file": "case2_spiral_to_gaussian_mixture.nc", "label": "Spiral to Gaussian Mixture"},
    {"file": "case3_moons_to_moons_rotated.nc",     "label": "Two Moons to Rotated Two Moons"},
    {"file": "case4_lissajous_to_trefoil.nc",       "label": "Lissajous to Trefoil"},
]

SNAP_TIMES = [0.0, 0.25, 0.50, 0.75, 1.0]
GRID = 140                 # KDE evaluation grid resolution per axis
DISPLAY_FLOOR = 0.03       # normalized density below this renders as white
REGION_LEVEL = 0.50        # super-level set for counting high-density regions
CMAP = "viridis"


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
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.labelsize": 10,
    })


# --------------------------------------------------------------------------- #
# I/O and interpolation
# --------------------------------------------------------------------------- #
def load_case(path):
    with Dataset(path, "r") as nc:
        traj = np.asarray(nc.variables["trajectory"][:], dtype=float)
        time = np.asarray(nc.variables["time"][:], dtype=float)
        src = np.asarray(nc.variables["X_source"][:], dtype=float)
        tgt = np.asarray(nc.variables["X_target"][:], dtype=float)
    return traj, time, src, tgt


def cloud_at_time(trajectory, time, t_star):
    exact = np.where(np.isclose(time, t_star))[0]
    if exact.size:
        return trajectory[int(exact[0])].copy()
    k = int(np.clip(np.searchsorted(time, t_star), 1, len(time) - 1))
    w = (t_star - time[k - 1]) / (time[k] - time[k - 1])
    return (1.0 - w) * trajectory[k - 1] + w * trajectory[k]


def square_extent(points, margin=0.08):
    pmin, pmax = points.min(axis=0), points.max(axis=0)
    centre = 0.5 * (pmin + pmax)
    half = 0.5 * float(np.max(pmax - pmin)) * (1.0 + margin)
    return (centre[0] - half, centre[0] + half,
            centre[1] - half, centre[1] + half)


# --------------------------------------------------------------------------- #
# Density estimation
# --------------------------------------------------------------------------- #
def kde_on_grid(points, extent):
    """Return normalized density on a GRID x GRID mesh plus the KDE bandwidth."""
    x0, x1, y0, y1 = extent
    xs = np.linspace(x0, x1, GRID)
    ys = np.linspace(y0, y1, GRID)
    XX, YY = np.meshgrid(xs, ys)
    kde = gaussian_kde(points.T)
    Z = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(GRID, GRID)
    Zmax = Z.max()
    Zn = Z / Zmax if Zmax > 0 else Z
    cell_area = (x1 - x0) * (y1 - y0) / (GRID * GRID)
    return Zn, float(kde.factor), xs, ys, cell_area


def count_regions(Zn, level=REGION_LEVEL):
    """Connected components of the super-level set {rho >= level*max}.

    Robust to ridge-like densities: a ring or curve counts as one region,
    separated blobs count individually.
    """
    _, n = label(Zn >= level)
    return int(n)


def effective_area(Zn, cell_area):
    """Participation-based effective support area exp(H[p]) * cell_area.

    A smooth, ridge-robust measure of how much area the density occupies,
    replacing the argmax peak location which is degenerate for rings/curves.
    """
    p = (Zn / Zn.sum()).ravel()
    nz = p > 0
    Hgrid = -np.sum(p[nz] * np.log(p[nz]))
    return float(np.exp(Hgrid) * cell_area)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(records, out_path):
    bar = "=" * 76
    sub = "-" * 76
    L = []
    L.append(bar)
    L.append(" MARGINAL DENSITY EVOLUTION OF THE SCHRODINGER BRIDGE")
    L.append(" Gaussian KDE of rho_t on a %d x %d grid; densities normalized to peak" % (GRID, GRID))
    L.append(sub)
    L.append(f" Author    : Sandy H. S. Herho <sandy.herho@email.ucr.edu>")
    L.append(f" Generated : {datetime.now().isoformat(timespec='seconds')}")
    L.append(f" Snapshot times : {', '.join(f'{t:.2f}' for t in SNAP_TIMES)}")
    L.append(f" Regions        : connected components of {{rho >= {REGION_LEVEL:.2f} max}}")
    L.append(f" Support area   : fraction of the panel with density >= {DISPLAY_FLOOR:.2f}")
    L.append(f" Eff. area      : exp(Shannon entropy of the density) x cell area")
    L.append(bar)
    L.append("")

    for i, rec in enumerate(records, 1):
        L.append(bar)
        L.append(f" CASE {i}: {rec['label']}")
        L.append(f" File: {rec['file']}   (KDE bandwidth factor = {rec['bw']:.4f})")
        L.append(sub)
        L.append(f"   {'t':>6}{'regions':>9}{'eff_area':>12}"
                 f"{'support_frac':>14}{'centroid_x':>12}{'centroid_y':>12}")
        L.append(" " + sub[:75])
        for s in rec["snaps"]:
            L.append(
                f"   {s['t']:>6.2f}{s['modes']:>9d}{s['eff_area']:>12.5f}"
                f"{s['support']:>14.5f}{s['centroid'][0]:>12.5f}{s['centroid'][1]:>12.5f}"
            )
        L.append("")

    L.append(bar)
    L.append(" HIGH-DENSITY REGION COUNT  (rows: cases, columns: snapshot times)")
    L.append(sub)
    header = f"   {'Case':<6}" + "".join(f"{f't={t:.2f}':>10}" for t in SNAP_TIMES)
    L.append(header)
    L.append(" " + sub[:75])
    for i, rec in enumerate(records, 1):
        row = f"   {('C' + str(i)):<6}" + "".join(f"{s['modes']:>10d}" for s in rec["snaps"])
        L.append(row)
    L.append(bar)
    L.append("")
    out_path.write_text("\n".join(L))


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(records, out_stem):
    n_rows, n_cols = len(records), len(SNAP_TIMES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.4, 7.2))
    letters = [f"({c})" for c in "abcdefghijklmnopqrst"]

    k = 0
    for r, rec in enumerate(records):
        extent = rec["extent"]
        for c, s in enumerate(rec["snaps"]):
            ax = axes[r, c]
            Zdisp = np.where(s["grid"] < DISPLAY_FLOOR, np.nan, s["grid"])
            ax.imshow(Zdisp, origin="lower", extent=extent, cmap=CMAP,
                      vmin=0.0, vmax=1.0, aspect="equal", interpolation="bilinear")
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.text(0.5, 1.04, letters[k], transform=ax.transAxes,
                    va="bottom", ha="center", fontweight="bold", fontsize=10)
            if r == n_rows - 1:
                ax.set_xlabel(r"$x$ [dimensionless]")
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(r"$y$ [dimensionless]")
            else:
                ax.set_yticklabels([])
            k += 1

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0), h_pad=1.6, w_pad=1.2)

    cax = fig.add_axes([0.30, 0.045, 0.40, 0.018])
    sm = cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap=CMAP)
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(r"normalized density $\rho_t / \max\,\rho_t$ [dimensionless]", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    # imshow embeds a raster; EPS stores it uncompressed, so use a lower
    # embedded-image dpi for EPS while keeping the raster formats crisp.
    save_dpi = {"pdf": 400, "png": 400, "eps": 150}
    for ext in ("pdf", "png", "eps"):
        fig.savefig(f"{out_stem}.{ext}", dpi=save_dpi[ext], bbox_inches="tight",
                    pad_inches=0.05, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    configure_style()
    CALC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for case in CASES:
        path = RAW_DATA_DIR / case["file"]
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        traj, time, src, tgt = load_case(path)

        clouds = [cloud_at_time(traj, time, t) for t in SNAP_TIMES]
        extent = square_extent(np.vstack([src, tgt] + clouds))

        snaps, bw = [], None
        for t, cloud in zip(SNAP_TIMES, clouds):
            Zn, factor, xs, ys, cell_area = kde_on_grid(cloud, extent)
            if bw is None:
                bw = factor
            support = float((Zn >= DISPLAY_FLOOR).sum()) / (GRID * GRID)
            snaps.append({
                "t": t, "grid": Zn,
                "modes": count_regions(Zn),
                "eff_area": effective_area(Zn, cell_area),
                "support": support,
                "centroid": (float(cloud[:, 0].mean()), float(cloud[:, 1].mean())),
            })

        records.append({"file": case["file"], "label": case["label"],
                        "extent": extent, "bw": bw, "snaps": snaps})
        print(f"Processed {case['file']}: {len(SNAP_TIMES)} density snapshots")

    write_report(records, CALC_DIR / REPORT_NAME)
    print(f"Report written : {CALC_DIR / REPORT_NAME}")
    make_figure(records, str(FIG_DIR / FIG_STEM))
    print(f"Figures written: {FIG_DIR / FIG_STEM}.pdf / .png / .eps")


if __name__ == "__main__":
    main()