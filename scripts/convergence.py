#!/usr/bin/env python
"""
Sinkhorn convergence of the entropic optimal transport solver.

Reads the stored marginal-constraint-violation history ||pi 1 - a||_1 from each
NetCDF file and displays it against iteration number, together with the
error normalized by its initial value so the geometric contraction rates of the
four cases can be compared on a common footing. A log-linear fit of the decay
gives an empirical per-iteration contraction factor for each case.

Note on the stored data: the solver records the marginal error every tenth
iteration, so the iteration axis is reconstructed as 10 * (record index). Cases
that converge almost immediately (here the symmetric circle-to-circle problem)
contain only a single recorded point and cannot be fitted.

Output
    ../figures/convergence.{pdf,png,eps}   1x2 convergence panels
    ../calculations/convergence.txt         numerical report

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
from netCDF4 import Dataset


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RAW_DATA_DIR = Path("../raw_data")
CALC_DIR = Path("../calculations")
FIG_DIR = Path("../figures")

FIG_STEM = "convergence"
REPORT_NAME = "convergence.txt"

CASES = [
    {"file": "case1_circle_to_circle.nc",           "label": "Circle to Circle"},
    {"file": "case2_spiral_to_gaussian_mixture.nc", "label": "Spiral to Gaussian Mixture"},
    {"file": "case3_moons_to_moons_rotated.nc",     "label": "Two Moons to Rotated Two Moons"},
    {"file": "case4_lissajous_to_trefoil.nc",       "label": "Lissajous to Trefoil"},
]

CASE_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
CASE_MARKERS = ["o", "s", "^", "D"]
RECORD_STRIDE = 10         # solver logs the marginal error every 10 iterations


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
# I/O and analysis
# --------------------------------------------------------------------------- #
def load_case(path):
    with Dataset(path, "r") as nc:
        err = np.asarray(nc.variables["marginal_errors"][:], dtype=float)
        eps = float(getattr(nc, "epsilon", np.nan))
        n_iter = int(getattr(nc, "n_iterations", -1))
        converged = bool(int(getattr(nc, "converged", 0)))
        tol = float(getattr(nc, "tolerance", np.nan))
    return err, eps, n_iter, converged, tol


def fit_rate(iters, err):
    """Log-linear fit: slope (log10/iter), intercept, per-iteration factor."""
    mask = err > 0
    if mask.sum() < 2:
        return None, None, None
    slope, intercept = np.polyfit(iters[mask], np.log10(err[mask]), 1)
    return float(slope), float(intercept), float(10.0 ** slope)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(records, out_path):
    bar = "=" * 76
    sub = "-" * 76
    L = []
    L.append(bar)
    L.append(" SINKHORN CONVERGENCE OF THE ENTROPIC OT SOLVER")
    L.append(" Marginal constraint violation ||pi 1 - a||_1 versus iteration")
    L.append(sub)
    L.append(f" Author    : Sandy H. S. Herho <sandy.herho@email.ucr.edu>")
    L.append(f" Generated : {datetime.now().isoformat(timespec='seconds')}")
    L.append(sub)
    L.append(f" NOTE: the marginal error is recorded every {RECORD_STRIDE} iterations, so the")
    L.append(" iteration axis is reconstructed as 10 * (record index). The per-iteration")
    L.append(" contraction factor is 10^slope of a log-linear fit to the decay.")
    L.append(bar)
    L.append("")

    for i, rc in enumerate(records, 1):
        L.append(bar)
        L.append(f" CASE {i}: {rc['label']}")
        L.append(f" File: {rc['file']}   (epsilon = {rc['eps']:.4f})")
        L.append(sub)
        L.append(f"   Converged                               = {rc['converged']}")
        L.append(f"   Iterations to convergence  [attr]       = {rc['n_iter']}")
        L.append(f"   Recorded error samples                  = {rc['n_samples']}")
        L.append(f"   Tolerance                               = {rc['tol']:.3e}")
        L.append(f"   Initial marginal error                  = {rc['err0']:.6e}")
        L.append(f"   Final marginal error                    = {rc['errf']:.6e}")
        if rc["slope"] is not None:
            L.append(f"   Log-linear slope       [log10/iter]     = {rc['slope']:.6e}")
            L.append(f"   Per-iteration contraction factor        = {rc['factor']:.6f}")
        else:
            L.append(f"   Log-linear slope                        = n/a (single sample)")
        L.append("")

    L.append(bar)
    L.append(" SUMMARY TABLE")
    L.append(sub)
    L.append(f"   {'Case':<6}{'eps':>8}{'n_iter':>9}{'err0':>13}{'err_final':>13}"
             f"{'contraction':>13}")
    L.append(" " + sub[:75])
    for i, rc in enumerate(records, 1):
        fac = f"{rc['factor']:>13.5f}" if rc["factor"] is not None else f"{'n/a':>13}"
        L.append(
            f"   {('C' + str(i)):<6}{rc['eps']:>8.3f}{rc['n_iter']:>9d}"
            f"{rc['err0']:>13.4e}{rc['errf']:>13.4e}{fac}"
        )
    L.append(bar)
    L.append("")
    out_path.write_text("\n".join(L))


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(records, out_stem):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    ax0, ax1 = axes

    # Only cases with a genuine decay curve (>= 3 recorded samples) are drawn;
    # trivially converged cases (single sample) are reported but not plotted.
    drawn = [rc for rc in records if rc["n_samples"] >= 3]

    for rc, color, marker in zip(drawn, CASE_COLORS, CASE_MARKERS):
        it, err = rc["iters"], rc["err"]
        ax0.semilogy(it, err, color=color, lw=1.2, marker=marker, ms=3.5,
                     markevery=max(1, len(it) // 12), mfc=color, mec=color,
                     label=rc["label"], zorder=3)
        # overlay the fitted geometric decay e0 * factor**iter
        if rc["slope"] is not None:
            fit = 10.0 ** (rc["intercept"] + rc["slope"] * it)
            ax0.semilogy(it, fit, color=color, lw=0.9, ls="--", alpha=0.9, zorder=2)
        # per-iteration contraction ratio (should be ~constant if geometric)
        ratio = (err[1:] / err[:-1]) ** (1.0 / RECORD_STRIDE)
        ax1.plot(it[1:], ratio, color=color, lw=1.2, marker=marker, ms=3.5,
                 markevery=max(1, len(it) // 12), mfc=color, mec=color)
        if rc["factor"] is not None:
            ax1.axhline(rc["factor"], color=color, lw=0.8, ls=":", alpha=0.8)

    ax0.set_xlabel("iteration [dimensionless]")
    ax0.set_ylabel("marginal error [dimensionless]")
    ax1.set_xlabel("iteration [dimensionless]")
    ax1.set_ylabel("contraction factor [dimensionless]")
    ax1.set_ylim(0.90, 1.0)

    for ax, lab in zip(axes, ["(a)", "(b)"]):
        ax.text(0.5, 1.04, lab, transform=ax.transAxes, va="bottom",
                ha="center", fontweight="bold", fontsize=11)
        ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)

    handles = [Line2D([], [], color=c, lw=1.4, marker=m, ms=4, label=rc["label"])
               for c, m, rc in zip(CASE_COLORS, CASE_MARKERS, drawn)]
    handles.append(Line2D([], [], color="0.4", lw=0.9, ls="--",
                          label="geometric fit"))
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0), w_pad=2.0)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=8.5,
               handletextpad=0.5, columnspacing=1.4)

    for ext in ("pdf", "png", "eps"):
        fig.savefig(f"{out_stem}.{ext}", dpi=600, bbox_inches="tight",
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
        err, eps, n_iter, converged, tol = load_case(path)
        iters = RECORD_STRIDE * np.arange(len(err))
        slope, intercept, factor = fit_rate(iters, err)
        records.append({
            "file": case["file"], "label": case["label"], "eps": eps,
            "n_iter": n_iter, "converged": converged, "tol": tol,
            "iters": iters, "err": err, "n_samples": len(err),
            "err0": float(err[0]), "errf": float(err[-1]),
            "slope": slope, "intercept": intercept, "factor": factor,
        })
        print(f"Processed {case['file']}: {len(err)} recorded error samples")

    write_report(records, CALC_DIR / REPORT_NAME)
    print(f"Report written : {CALC_DIR / REPORT_NAME}")
    make_figure(records, str(FIG_DIR / FIG_STEM))
    print(f"Figures written: {FIG_DIR / FIG_STEM}.pdf / .png / .eps")


if __name__ == "__main__":
    main()
