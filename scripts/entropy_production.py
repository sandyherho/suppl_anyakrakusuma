#!/usr/bin/env python
"""
Entropy production and geometric evolution along the Schrodinger bridge.

Reads the four `anyakrakusuma` NetCDF trajectories and, for every stored frame,
estimates the differential entropy of the marginal rho_t together with a set of
second-moment shape descriptors (RMS dispersion, covariance-ellipse
eccentricity and orientation). The differential entropy is obtained with the
Kozachenko-Leonenko k-nearest-neighbour estimator, which is well suited to the
thin, curved point clouds produced here.

Output
    ../figures/entropy_production.{pdf,png,eps}   2x2 line panels
    ../calculations/entropy_production.txt         numerical report

All quantities are derived only from the stored trajectory and time arrays.

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
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RAW_DATA_DIR = Path("../raw_data")
CALC_DIR = Path("../calculations")
FIG_DIR = Path("../figures")

FIG_STEM = "entropy_production"
REPORT_NAME = "entropy_production.txt"

CASES = [
    {"file": "case1_circle_to_circle.nc",           "label": "Circle to Circle"},
    {"file": "case2_spiral_to_gaussian_mixture.nc", "label": "Spiral to Gaussian Mixture"},
    {"file": "case3_moons_to_moons_rotated.nc",     "label": "Two Moons to Rotated Two Moons"},
    {"file": "case4_lissajous_to_trefoil.nc",       "label": "Lissajous to Trefoil"},
]

CASE_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
K_NEIGHBOURS = 5
ECC_MIN = 0.25          # orientation resolved only above this anisotropy
BOOT_B = 80             # subsampling replicates for the uncertainty bands
BOOT_FRAC = 0.80        # subsample size as a fraction of N (no replacement)
CI_Z = 1.96             # 95% Gaussian band
BOOT_SEED = 0


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
# Estimators
# --------------------------------------------------------------------------- #
def knn_differential_entropy(points, k=K_NEIGHBOURS):
    """Kozachenko-Leonenko differential entropy estimate (nats)."""
    n, d = points.shape
    tree = cKDTree(points)
    dist, _ = tree.query(points, k=k + 1)          # first neighbour is the point itself
    r = np.maximum(dist[:, -1], 1e-12)
    log_unit_ball = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1.0)
    return float(-digamma(k) + digamma(n) + log_unit_ball + (d / n) * np.sum(np.log(r)))


def ellipse_descriptors(points):
    """Return RMS dispersion, eccentricity and principal-axis angle (radians)."""
    cov = np.cov(points, rowvar=False)
    cxx, cyy, cxy = cov[0, 0], cov[1, 1], cov[0, 1]
    evals = np.linalg.eigvalsh(cov)                # ascending
    lam_min, lam_max = float(evals[0]), float(evals[1])
    rms = float(np.sqrt(lam_min + lam_max))
    ecc = float(np.sqrt(max(0.0, 1.0 - lam_min / lam_max))) if lam_max > 0 else 0.0
    angle2 = np.arctan2(2.0 * cxy, cxx - cyy)      # doubled angle, resolves axis wrap
    return rms, ecc, lam_min, lam_max, angle2


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_case(path):
    with Dataset(path, "r") as nc:
        traj = np.asarray(nc.variables["trajectory"][:], dtype=float)
        time = np.asarray(nc.variables["time"][:], dtype=float)
        eps = float(getattr(nc, "epsilon", np.nan))
    return traj, time, eps


def analyse_case(traj, time):
    """Per-frame descriptors with subsampling-bootstrap 95% uncertainty bands.

    Uncertainty is estimated by m-out-of-n subsampling without replacement
    (replacement would create coincident points and corrupt the k-NN entropy).
    The half-width at full sample size N is std_sub * sqrt(m/N) * z, following
    the sqrt(n) scaling of the estimators.
    """
    nframes = traj.shape[0]
    n = traj.shape[1]
    m = max(10, int(BOOT_FRAC * n))
    scale = np.sqrt(m / n)
    rng = np.random.default_rng(BOOT_SEED)

    H = np.empty(nframes); rms = np.empty(nframes)
    ecc = np.empty(nframes); angle2 = np.empty(nframes)
    H_hw = np.empty(nframes); rms_hw = np.empty(nframes); ecc_hw = np.empty(nframes)
    ang_hw = np.empty(nframes)

    for f in range(nframes):
        pts = traj[f]
        H[f] = knn_differential_entropy(pts)
        rms[f], ecc[f], _, _, angle2[f] = ellipse_descriptors(pts)

        Hs = np.empty(BOOT_B); Rs = np.empty(BOOT_B)
        Es = np.empty(BOOT_B); A2 = np.empty(BOOT_B)
        for b in range(BOOT_B):
            p = pts[rng.choice(n, m, replace=False)]
            Hs[b] = knn_differential_entropy(p)
            Rs[b], Es[b], _, _, A2[b] = ellipse_descriptors(p)
        H_hw[f] = CI_Z * Hs.std() * scale
        rms_hw[f] = CI_Z * Rs.std() * scale
        ecc_hw[f] = CI_Z * Es.std() * scale
        # circular spread of the doubled principal-axis angle across subsamples
        R = np.abs(np.mean(np.exp(1j * A2)))
        sigma_circ = np.sqrt(-2.0 * np.log(R)) if R > 1e-12 else np.inf
        ang_hw[f] = np.degrees(0.5 * sigma_circ * scale) * CI_Z

    # Orientation is only meaningful for anisotropic clouds; unwrap the
    # doubled angle over the reliable frames only and mask the rest.
    angle_deg = np.full(nframes, np.nan)
    ang_band = np.full(nframes, np.nan)
    reliable = np.where(ecc >= ECC_MIN)[0]
    if reliable.size:
        angle_deg[reliable] = np.degrees(0.5 * np.unwrap(angle2[reliable]))
        ang_band[reliable] = ang_hw[reliable]
    return {"H": H, "rms": rms, "ecc": ecc, "angle": angle_deg,
            "H_hw": H_hw, "rms_hw": rms_hw, "ecc_hw": ecc_hw,
            "angle_hw": ang_band}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def diffusive_entropy_term(eps, t, d=2):
    """Differential entropy of the pure bridge-noise term, sigma^2 = eps t (1-t)."""
    var = eps * t * (1.0 - t)
    if var <= 0:
        return float("-inf")
    return float((d / 2.0) * np.log(2.0 * np.pi * np.e * var))


def interp_at(time, series, t_star):
    return float(np.interp(t_star, time, series))


def write_report(results, out_path):
    bar = "=" * 76
    sub = "-" * 76
    L = []
    L.append(bar)
    L.append(" ENTROPY PRODUCTION AND GEOMETRIC EVOLUTION ALONG THE BRIDGE")
    L.append(" Kozachenko-Leonenko differential entropy (k = %d, nats) and" % K_NEIGHBOURS)
    L.append(" covariance-ellipse descriptors, per frame")
    L.append(sub)
    L.append(f" Author    : Sandy H. S. Herho <sandy.herho@email.ucr.edu>")
    L.append(f" Generated : {datetime.now().isoformat(timespec='seconds')}")
    L.append(f" Uncertainty : 95% bands from {BOOT_B} subsamples "
             f"at m/N = {BOOT_FRAC:.2f}, scaled to full N")
    L.append(bar)
    L.append("")

    for i, r in enumerate(results, 1):
        t = r["time"]
        d = r["desc"]
        H0, Hm, H1 = (interp_at(t, d["H"], v) for v in (0.0, 0.5, 1.0))
        rms0, rmsm, rms1 = (interp_at(t, d["rms"], v) for v in (0.0, 0.5, 1.0))
        i_peak = int(np.argmax(d["rms"]))
        ang = d["angle"]
        defined = np.where(np.isfinite(ang))[0]
        if defined.size:
            ang0, ang1 = ang[defined[0]], ang[defined[-1]]
            ang_str = f"{ang0:.4f} / {ang1:.4f}"
            sweep_str = f"{ang1 - ang0:.4f}"
        else:
            ang0 = ang1 = np.nan
            ang_str = "undefined (near-isotropic cloud)"
            sweep_str = "undefined"
        H_diff_mid = diffusive_entropy_term(r["eps"], 0.5)

        L.append(bar)
        L.append(f" CASE {i}: {r['label']}")
        L.append(f" File: {r['file']}   (epsilon = {r['eps']:.4f}, frames = {len(t)})")
        L.append(sub)
        L.append(" Differential entropy H(rho_t)  [nats]")
        L.append(f"   H(t=0)                                  = {H0:.6f}")
        L.append(f"   H(t=0.5)                                = {Hm:.6f}")
        L.append(f"   H(t=1)                                  = {H1:.6f}")
        L.append(f"   Net entropy production  H(1) - H(0)     = {H1 - H0:.6f}")
        L.append(f"   Peak entropy  max_t H(rho_t)            = {np.max(d['H']):.6f}  at t = {t[int(np.argmax(d['H']))]:.4f}")
        L.append(f"   Mean 95% band half-width (H)            = {np.mean(d['H_hw']):.6f}")
        L.append(f"   Diffusive term (d/2)ln(2*pi*e*eps/4)    = {H_diff_mid:.6f}   [noise-only reference at t=0.5]")
        L.append("")
        L.append(" RMS dispersion  sqrt(trace Cov)")
        L.append(f"   RMS(t=0) / RMS(0.5) / RMS(1)            = {rms0:.6f} / {rmsm:.6f} / {rms1:.6f}")
        L.append(f"   Peak RMS dispersion                     = {d['rms'][i_peak]:.6f}  at t = {t[i_peak]:.4f}")
        L.append(f"   Mean 95% band half-width (RMS)          = {np.mean(d['rms_hw']):.6f}")
        L.append("")
        L.append(" Covariance-ellipse geometry")
        L.append(f"   Eccentricity  range [min, max]          = [{np.min(d['ecc']):.6f}, {np.max(d['ecc']):.6f}]")
        L.append(f"   Orientation resolved frames (ecc>={ECC_MIN:.2f})  = {int(np.isfinite(d['angle']).sum())} / {len(d['angle'])}")
        L.append(f"   Orientation angle  first / last  [deg]  = {ang_str}")
        L.append(f"   Total orientation sweep       [deg]     = {sweep_str}")
        ohw = d["angle_hw"][np.isfinite(d["angle_hw"])]
        ohw_str = f"{float(np.mean(ohw)):.4f}" if ohw.size else "n/a"
        L.append(f"   Mean 95% orientation half-width [deg]   = {ohw_str}")
        L.append("")

    L.append(bar)
    L.append(" SUMMARY TABLE")
    L.append(sub)
    L.append(f"   {'Case':<6}{'eps':>8}{'H(0)':>10}{'H(0.5)':>10}{'H(1)':>10}"
             f"{'dH':>10}{'RMSpeak':>10}{'sweep_deg':>11}")
    L.append(" " + sub[:75])
    for i, r in enumerate(results, 1):
        t, d = r["time"], r["desc"]
        H0, H1 = interp_at(t, d["H"], 0.0), interp_at(t, d["H"], 1.0)
        ang = d["angle"][np.isfinite(d["angle"])]
        sweep = (ang[-1] - ang[0]) if ang.size else np.nan
        sweep_txt = f"{sweep:>11.3f}" if np.isfinite(sweep) else f"{'n/a':>11}"
        L.append(
            f"   {('C' + str(i)):<6}{r['eps']:>8.3f}"
            f"{H0:>10.4f}{interp_at(t, d['H'], 0.5):>10.4f}{H1:>10.4f}"
            f"{H1 - H0:>10.4f}{np.max(d['rms']):>10.4f}"
            f"{sweep_txt}"
        )
    L.append(bar)
    L.append("")
    out_path.write_text("\n".join(L))


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(results, out_stem):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    letters = ["(a)", "(b)", "(c)", "(d)"]
    keys = ["H", "rms", "ecc", "angle"]
    ylabels = [r"$H(\rho_t)$ [nats]", r"RMS dispersion [dimensionless]",
               r"eccentricity [dimensionless]", r"orientation [deg]"]

    hw_keys = {0: "H_hw", 1: "rms_hw", 2: "ecc_hw", 3: "angle_hw"}
    for p, ax in enumerate(axes.flat):
        for r, color in zip(results, CASE_COLORS):
            y = r["desc"][keys[p]]
            if p in hw_keys:
                hw = r["desc"][hw_keys[p]]
                ax.fill_between(r["time"], y - hw, y + hw,
                                color=color, alpha=0.16, linewidth=0)
            ax.plot(r["time"], y, color=color, lw=1.4)
        ax.set_ylabel(ylabels[p])
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.text(0.5, 1.03, letters[p], transform=ax.transAxes,
                va="bottom", ha="center", fontweight="bold", fontsize=11)
        if p in (2, 3):
            ax.set_xlabel(r"interpolation time $t$ [dimensionless]")

    handles = [Line2D([], [], color=c, lw=1.6, label=r["label"])
               for c, r in zip(CASE_COLORS, results)]
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0), h_pad=1.8, w_pad=2.0)
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.006), fontsize=9,
               handletextpad=0.5, columnspacing=1.8)

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

    results = []
    for case in CASES:
        path = RAW_DATA_DIR / case["file"]
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        traj, time, eps = load_case(path)
        desc = analyse_case(traj, time)
        results.append({"file": case["file"], "label": case["label"],
                        "time": time, "eps": eps, "desc": desc})
        print(f"Analysed {case['file']}: {traj.shape[0]} frames, "
              f"{traj.shape[1]} particles")

    write_report(results, CALC_DIR / REPORT_NAME)
    print(f"Report written : {CALC_DIR / REPORT_NAME}")
    make_figure(results, str(FIG_DIR / FIG_STEM))
    print(f"Figures written: {FIG_DIR / FIG_STEM}.pdf / .png / .eps")


if __name__ == "__main__":
    main()