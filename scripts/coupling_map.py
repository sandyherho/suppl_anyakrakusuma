#!/usr/bin/env python
"""
Entropic transport map from the stored optimal coupling.

For each case the barycentric projection of the entropic plan,
    T(x_i) = E[Y | X = x_i] = sum_j P(j|i) y_j,
is reconstructed from the stored transport_plan and drawn as displacement
arrows from each source point to its conditional-mean target. This is the
entropic analogue of the Monge map and shows the direction and magnitude of
mass transport that the Sinkhorn solution encodes.

Important caveat, stated in the report as well: the solver stores the plan
subsampled to at most 500 x 500 with strided indices when n exceeds 500, and
the retained rows are not renormalized. This script renormalizes the retained
rows for display, so for n > 500 the map is computed over a subset of source
and target points and is therefore approximate. For exact maps, archive or
recompute the full plan.

Output
    ../figures/coupling_map.{pdf,png,eps}   2x2 transport-map panels
    ../calculations/coupling_map.txt         numerical report

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

FIG_STEM = "coupling_map"
REPORT_NAME = "coupling_map.txt"

CASES = [
    {"file": "case1_circle_to_circle.nc",           "label": "Circle to Circle"},
    {"file": "case2_spiral_to_gaussian_mixture.nc", "label": "Spiral to Gaussian Mixture"},
    {"file": "case3_moons_to_moons_rotated.nc",     "label": "Two Moons to Rotated Two Moons"},
    {"file": "case4_lissajous_to_trefoil.nc",       "label": "Lissajous to Trefoil"},
]

C_SOURCE = "#4C72B0"
C_TARGET = "#C44E52"
C_ARROW = "#2E2E2E"
N_ARROWS = 90              # arrows drawn per panel (subsampled for legibility)


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
    with Dataset(path, "r") as nc:
        plan = np.asarray(nc.variables["transport_plan"][:], dtype=float)
        src = np.asarray(nc.variables["X_source"][:], dtype=float)
        tgt = np.asarray(nc.variables["X_target"][:], dtype=float)
        eps = float(getattr(nc, "epsilon", np.nan))
        stored_H = float(getattr(nc, "plan_entropy", np.nan))
    return plan, src, tgt, eps, stored_H


def align_plan(plan, src, tgt):
    """Match source/target points to plan rows/cols, handling subsampling."""
    n = src.shape[0]
    p = plan.shape[0]
    if p < n:
        idx = np.linspace(0, n - 1, p, dtype=int)
        return plan, src[idx], tgt[idx], True, p, n
    return plan, src, tgt, False, p, n


def barycentric_map(plan, tgt_pts):
    """Row-normalized conditional means E[Y|X=i] and the conditionals."""
    row_sums = plan.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-300, 1.0, row_sums)
    cond = plan / row_sums                        # P(Y | X = i)
    ybar = cond @ tgt_pts
    return cond, ybar


def square_limits(points, margin=0.08):
    pmin, pmax = points.min(axis=0), points.max(axis=0)
    centre = 0.5 * (pmin + pmax)
    half = 0.5 * float(np.max(pmax - pmin)) * (1.0 + margin)
    return (centre[0] - half, centre[0] + half), (centre[1] - half, centre[1] + half)


def neg_xlogx(p):
    """Elementwise -p*log(p) with the 0*log(0)=0 convention, no log(0) warning."""
    out = np.zeros_like(p, dtype=float)
    nz = p > 0
    out[nz] = -p[nz] * np.log(p[nz])
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(records, out_path):
    bar = "=" * 76
    sub = "-" * 76
    L = []
    L.append(bar)
    L.append(" ENTROPIC TRANSPORT MAP FROM THE STORED COUPLING")
    L.append(" Barycentric projection T(x) = E[Y|X=x] and conditional diagnostics")
    L.append(sub)
    L.append(f" Author    : Sandy H. S. Herho <sandy.herho@email.ucr.edu>")
    L.append(f" Generated : {datetime.now().isoformat(timespec='seconds')}")
    L.append(sub)
    L.append(" NOTE: quantities below are computed on the stored transport_plan.")
    L.append(" When the plan is subsampled (n > 500) the conditionals are over a")
    L.append(" strided subset of targets and are therefore approximate.")
    L.append(bar)
    L.append("")

    for i, rc in enumerate(records, 1):
        L.append(bar)
        L.append(f" CASE {i}: {rc['label']}")
        L.append(f" File: {rc['file']}   (epsilon = {rc['eps']:.4f})")
        L.append(sub)
        if rc["subsampled"]:
            L.append(f"   Plan storage        : SUBSAMPLED  {rc['p']} x {rc['p']}  (n = {rc['n']})")
        else:
            L.append(f"   Plan storage        : FULL  {rc['p']} x {rc['p']}")
        L.append("")
        L.append(" Conditional coupling P(Y|X=i)")
        L.append(f"   Mean row entropy H_i      [nats]        = {rc['H_mean']:.6f}")
        L.append(f"   Row entropy spread (std)  [nats]        = {rc['H_std']:.6f}")
        L.append(f"   Mean effective targets  <exp(H_i)>      = {rc['perplexity']:.4f}")
        L.append(f"   Mean peak conditional prob  max_j P(j|i)= {rc['max_prob']:.6f}")
        L.append("")
        L.append(" Barycentric transport displacement  |T(x) - x|")
        L.append(f"   Mean displacement                       = {rc['disp_mean']:.6f}")
        L.append(f"   Median displacement                     = {rc['disp_med']:.6f}")
        L.append(f"   Max displacement                        = {rc['disp_max']:.6f}")
        L.append("")
        L.append(" Reference")
        L.append(f"   Stored full-plan entropy H(pi)  [attr]  = {rc['stored_H']:.6f}")
        L.append(f"   (block-normalized entropy, this subset) = {rc['block_H']:.6f}")
        L.append("")

    L.append(bar)
    L.append(" SUMMARY TABLE")
    L.append(sub)
    L.append(f"   {'Case':<6}{'eps':>8}{'eff_targets':>13}{'maxprob':>10}"
             f"{'disp_mean':>12}{'disp_max':>12}")
    L.append(" " + sub[:75])
    for i, rc in enumerate(records, 1):
        L.append(
            f"   {('C' + str(i)):<6}{rc['eps']:>8.3f}{rc['perplexity']:>13.4f}"
            f"{rc['max_prob']:>10.5f}{rc['disp_mean']:>12.6f}{rc['disp_max']:>12.6f}"
        )
    L.append(bar)
    L.append("")
    out_path.write_text("\n".join(L))


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(records, out_stem):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2))
    letters = ["(a)", "(b)", "(c)", "(d)"]

    for p, (ax, rc) in enumerate(zip(axes.flat, records)):
        src, tgt, ybar = rc["src"], rc["tgt"], rc["ybar"]

        ax.scatter(src[:, 0], src[:, 1], s=7, c=C_SOURCE, alpha=0.22, linewidths=0)
        ax.scatter(tgt[:, 0], tgt[:, 1], s=7, c=C_TARGET, alpha=0.22, linewidths=0)

        sub = np.linspace(0, src.shape[0] - 1, min(N_ARROWS, src.shape[0]), dtype=int)
        disp = ybar[sub] - src[sub]
        ax.quiver(src[sub, 0], src[sub, 1], disp[:, 0], disp[:, 1],
                  angles="xy", scale_units="xy", scale=1.0, width=0.004,
                  headwidth=4, headlength=5, color=C_ARROW, alpha=0.85, zorder=3)

        xlim, ylim = square_limits(np.vstack([src, tgt, ybar]))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.text(0.5, 1.03, letters[p], transform=ax.transAxes,
                va="bottom", ha="center", fontweight="bold", fontsize=11)
        if p in (2, 3):
            ax.set_xlabel(r"$x$ [dimensionless]")
        if p in (0, 2):
            ax.set_ylabel(r"$y$ [dimensionless]")

    handles = [
        Line2D([], [], linestyle="None", marker="o", markerfacecolor=C_SOURCE,
               markeredgecolor="none", markersize=7, label=r"Source $\mu$"),
        Line2D([], [], linestyle="None", marker="o", markerfacecolor=C_TARGET,
               markeredgecolor="none", markersize=7, label=r"Target $\nu$"),
        Line2D([], [], color=C_ARROW, lw=1.4, marker=">", markersize=5,
               label=r"Transport $T(x)=\mathbb{E}[Y|X{=}x]$"),
    ]
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0), h_pad=1.8, w_pad=2.0)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.006), fontsize=9,
               handletextpad=0.4, columnspacing=1.6)

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
        plan, src_full, tgt_full, eps, stored_H = load_case(path)
        plan, src, tgt, subsampled, p, n = align_plan(plan, src_full, tgt_full)
        cond, ybar = barycentric_map(plan, tgt)

        row_H = neg_xlogx(cond).sum(axis=1)
        disp = np.linalg.norm(ybar - src, axis=1)
        block = plan / plan.sum()
        block_H = float(neg_xlogx(block).sum())

        records.append({
            "file": case["file"], "label": case["label"], "eps": eps,
            "subsampled": subsampled, "p": p, "n": n,
            "src": src, "tgt": tgt, "ybar": ybar,
            "H_mean": float(row_H.mean()), "H_std": float(row_H.std()),
            "perplexity": float(np.exp(row_H).mean()),
            "max_prob": float(cond.max(axis=1).mean()),
            "disp_mean": float(disp.mean()), "disp_med": float(np.median(disp)),
            "disp_max": float(disp.max()),
            "stored_H": stored_H, "block_H": block_H,
        })
        tag = "subsampled" if subsampled else "full"
        print(f"Processed {case['file']}: plan {p}x{p} ({tag})")

    write_report(records, CALC_DIR / REPORT_NAME)
    print(f"Report written : {CALC_DIR / REPORT_NAME}")
    make_figure(records, str(FIG_DIR / FIG_STEM))
    print(f"Figures written: {FIG_DIR / FIG_STEM}.pdf / .png / .eps")


if __name__ == "__main__":
    main()