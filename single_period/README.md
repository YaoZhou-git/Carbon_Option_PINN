# Single-period carbon-option calculations

Source notebooks, saved models and results for Sections 4.1--4.3 and Table 1.

## Layout

- `notebooks/`: principal training, simulation, comparison and analysis notebooks.
- `pinn_results/`: baseline checkpoints, seven maturity runs and Monte Carlo results. Keep each result beside its `simulation_parameters.txt`.
- `reference_figures/`: saved allowance-distribution, PINN/MC comparison and maturity figures.
- `reference_tables/`: boundary diagnostics and Table 1 values.
- `entrypoints/`: baseline training and parameter-scan notebooks.
- `legacy/`: earlier model and analysis variants.
- `generated/`: output directory created by training and plotting.

Install dependencies from the repository root's requirements files.

## Figures and diagnostics

| Result | Notebook / entry |
|---|---|
| Figure 2: allowance-price distributions | `notebooks/A_distribution_simulation.ipynb`, code cell 4 (zero-based; run preceding imports first) |
| Figure 3: PINN and Monte Carlo comparison | `notebooks/compare.ipynb` |
| Figure 4: maturity curves | `notebooks/Sensitivity Analysis .ipynb`, code cell 2 |
| Table 1: saved boundary diagnostics | `notebooks/Result PINN Solution.ipynb` and `reference_tables/` |
| Table 1: ATM normalization | `python single_period/atm_reference.py` from the repository root |

The maturity figure uses the seven saved runs at tau = 0.5, 1, 1.5, 2, 2.5, 3 and 3.5, with alpha = 1.2 and beta = 0.5. The allowance-path simulation has no fixed random seed.

The saved Figure 3 image is supplied separately: its exact checkpoint/Monte Carlo pair is not identified in the saved records. The included Monte Carlo source and metadata specify 300 steps at tau = 3; the manuscript comparison describes 400 steps.

For Table 1, the August checkpoint uses a 16-32-16 network; the later baseline trainer uses 64-64-64. The diagnostic notebook evaluates terminal and initial values at A = 0. Its parameter reader can fall back to alpha = 0.8 when metadata is absent, while the checkpoint metadata specifies alpha = 1.2. Keep the saved metadata with the corresponding checkpoint. The legacy CSV also contains an upper-bound check outside Table 1.

## Training

Run from the repository root:

```text
python -m jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=-1 single_period/entrypoints/train_baseline.ipynb --output train_baseline.executed.ipynb --output-dir single_period/generated
```

For the parameter scan:

```text
python -m jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=-1 single_period/entrypoints/train_large_scan.ipynb --output train_large_scan.executed.ipynb --output-dir single_period/generated
```

The scan uses `SCAN_ALPHA_BETA=True`, `SCAN_TAU=False` and 51 x 51 parameter combinations. It performs new training and does not generate the seven saved maturity runs in its default configuration. Earlier notebooks in `legacy/` may require additional historical inputs.
