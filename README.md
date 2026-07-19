# Supplementary Data Analysis Scripts

### *anyakrakusuma: A Python Library for Entropic Schrödinger Bridges on Idealized Geometries*

This repository holds the post-processing scripts that turn the raw NetCDF
solver output of **`anyakrakusuma`** into the diagnostic figures and numerical
reports of the accompanying manuscript. Each script reads the four archived
solver runs, computes one family of diagnostics, and writes both a
publication-quality figure (PDF, PNG, EPS) and a plain-text numerical report.
The scripts are read-only with respect to the solver: they consume the
archived `.nc` files and never modify them.

## Scripts

All four live in `scripts/` and are run from that directory. Each reads the
four case files from `../raw_data/`, writes figures to `../figures/`, and
writes a text report to `../calculations/`.

| Script | Reads from each `.nc` | Produces |
|---|---|---|
| `convergence.py` | `marginal_errors`, `epsilon`, `n_iterations`, `converged`, `tolerance` | Sinkhorn residual decay and per-iteration contraction factor (log-linear fit); 1×2 panel |
| `coupling_map.py` | `transport_plan`, `X_source`, `X_target`, `epsilon`, `plan_entropy` | Barycentric transport map `T(x) = E[Y|X=x]`, conditional row entropy, perplexity, and displacement; 2×2 panel |
| `density_evolution.py` | `trajectory`, `time`, `X_source`, `X_target` | Gaussian-KDE marginals at t = 0, 0.25, 0.5, 0.75, 1, with high-density region counts and effective support area; 4×5 montage |
| `entropy_production.py` | `trajectory`, `time`, `epsilon` | Per-frame Kozachenko–Leonenko differential entropy and covariance-ellipse descriptors (RMS dispersion, eccentricity, principal-axis orientation) with subsampling uncertainty bands; 2×2 panel |

## Four cases

Each script processes the same four idealized transport problems:

1. `case1_circle_to_circle.nc` — circle to circle (dilation)
2. `case2_spiral_to_gaussian_mixture.nc` — spiral to Gaussian mixture (fragmentation)
3. `case3_moons_to_moons_rotated.nc` — two moons to rotated two moons (reorientation)
4. `case4_lissajous_to_trefoil.nc` — Lissajous curve to trefoil (deformation)

## Requirements

Python 3.9 or newer, with:

```bash
pip install numpy scipy matplotlib netCDF4
```

## Usage

Download the archived NetCDF files from the OSF repository (see below) into a
`raw_data/` directory at the repository root, then run any script from inside
`scripts/`:

```bash
mkdir -p raw_data calculations figures
# place the four .nc files in raw_data/

cd scripts
python convergence.py
python coupling_map.py
python density_evolution.py
python entropy_production.py
```

The `figures/` and `calculations/` directories are created automatically if
absent. A script exits with a clear error if a required `.nc` file is missing.

### Directory layout

```
.
├── scripts/          # the four analysis scripts (tracked)
├── raw_data/         # input NetCDF files (not tracked; from OSF)
├── figures/          # generated figures     (not tracked)
└── calculations/     # generated text reports (not tracked)
```

Only the scripts are version-controlled; all inputs and generated outputs are
git-ignored and distributed through the OSF archive.

## Related resources

- **Library source code:** https://github.com/sandyherho/anyakrakusuma
- **Install from PyPI:** https://pypi.org/project/anyakrakusuma/
- **Archived data (raw NetCDF, computed metrics, run logs, figures):** https://doi.org/10.17605/OSF.IO/VQWF4

## Authors

Sandy H. S. Herho, Dasapta E. Irawan, Agus W. Jatmiko, Sito F. Biosa,
Candrasa Surya Dharma, Edi Riawan, Astyka Pamumpuni, Rendy D. Kartiko,
Rusmawan Suwarman, and Deny J. Puradimaja.

Correspondence: Sandy H. S. Herho — <sandy.herho@email.ucr.edu>

## License

MIT License - See [LICENSE](LICENSE) for details.