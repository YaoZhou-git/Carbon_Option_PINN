# Multi-period carbon-option calculations

This directory contains the current four-factor PINN calculations, trained models,
saved synthetic results, independent reference implementations, and plotting code.


Run commands from this directory with Python 3.10 or newer. Install dependencies
with `python -m pip install -r requirements.txt`. No external pretrained model is
downloaded. The original recorded inference environment used Python 3.10.16,
PyTorch 2.7.0+cu128 and NumPy 2.2.5. The original Sobol run used Python 3.12.4 and
NumPy 1.26.4. This dependency list is not an exact environment lock.

## Lightweight checks and saved-result reproduction

```text
python smoke_check.py
python -m unittest discover -s core -p "test_factor*.py" -v
python -m unittest discover -s auxiliary -p "test_generalized_reference.py" -v
python primary/infer_existing_pinn.py
python restored_plot/plot_restored_figure5.py
python economic_plots/plot_pinn_economic_44.py
python primary/plot_pinn_primary_44.py
```

The smoke check loads all four baseline models and both matched models on CPU and
compares six prices with existing saved outputs. It performs no training or
reference integration. The 13 existing pure-model tests check the model, PINN and
generalized reference integrator; the separate historical-market-data audit tests
are excluded. The inference driver evaluates the existing 96-state batch and
automatic derivatives using saved references.

Generated files go under `outputs/`. The current manuscript figures are:

- Figure 5: `outputs/restored_figure5/fig/multiperiod_curve_distribution_restored.*`.
- Figure 6: `outputs/economic_figures/fig/multiperiod_quote_scans_pinn_only.*`.
- Appendix D validation: `outputs/primary_figures/fig/multiperiod_pinn_validation.*`.

The plotting drivers also retain their original alternative views and Chinese
exports. They read frozen CSVs under `primary/source_data/` and `reference/data/`;
they do not invoke models or integration. Windows Chinese font loading is optional
when available. On other systems, install an appropriate Chinese font for the Chinese
exports and Arial for matching English typography; font substitution can change
the rendered appearance without changing data.

## Model and result sources

`core/factor_model.py` and `core/factor_pinn.py` are the local model implementation
and baseline training source. `models/baseline/` contains N=1,...,4 models. Training used seeds
20260917+N, 10,000 Adam updates, batch size 2,048 and cosine learning rate
1e-3 to 5e-5. Recorded N4 time was 687.714 seconds.

`primary/train_matched.py` warm-starts the N4 baseline for flat and upward curves,
using seeds 20261111 and 20261112, 2,500 additional updates, batch size 2,048 and
cosine learning rate 1e-4 to 1e-5. It preserves the original explicit beta-buffer
restoration after loading the baseline. Existing weights are in `models/matched/`.
The original training driver requires CUDA. Training metadata are in `primary/saved/` and `models/baseline/provenance/`.

`primary/infer_existing_pinn.py` preserves the original forward map, factor/quote
Jacobian and automatic differentiation. It reads the baseline checkpoint, the 93
saved quote locations and the saved reference prices/gradients. Its checkpoint
inventory uses the Python standard library and is limited to included models.

The methods used for the manuscript results must be distinguished:

| Result | Source and method |
| --- | --- |
| Figure 5(a), risk freezing, variance matching, N comparisons, quote references | `reference/scripts/run_analysis.py`: exact latent Gaussian transitions and scrambled Sobol integration; eight sequences, 2^17 points for main cases and 2^14 for scans. |
| Figure 5(b), Figure 6 and reference-state quote gradients | Existing baseline/matched PINNs and saved automatic derivatives; final figures use their frozen CSVs. |
| Current 24-state validation, maximum absolute 0.0464 and relative 1.194% | `reference/scripts/supplement_analysis.py`: compares 24 saved PINN outputs with eight independently scrambled Sobol sequences of 2^17 points. |
| Original 24-state verification | `legacy/export_pinn.py`: deterministic conditional Gaussian quadrature (order 80, inner 256), distinct from the current Sobol validation. |
| Continuation B5=8 value 16.1664; penalty-path values 16.4678 and 16.2034 | `auxiliary/run_robustness.py`: conditional Gaussian quadrature, order 40 and inner 160. These particular auxiliary numbers are not Sobol outputs. |

## Optional expensive regeneration

These commands are manual and are not part of the lightweight smoke check:

```text
python reference/scripts/run_analysis.py
python reference/scripts/supplement_analysis.py
python auxiliary/run_robustness.py
python legacy/export_pinn.py
python core/factor_pinn.py --epochs 10000 --batch 2048 --dimensions 1 2 3 4
python primary/train_matched.py
```

The reference drivers perform many integration scenarios, not just one figure.
They overwrite synthetic reference CSVs in the `reference/data/` directory.
The supplementary driver executes its analyses at module top level; use it as a
script rather than importing it as a passive utility. Its saved 24-point PINN input
is included as `reference/data/saved_PINN_input.csv`.

The auxiliary driver also includes its historical held-out checks; outputs go to
`outputs/auxiliary/`. The legacy export writes to `outputs/legacy/`. Retraining
overwrites the corresponding model files; matched training logs go to
`outputs/matched_training/`. Plotting defaults remain tied to frozen paper-result
CSVs, so newly trained weights do not silently replace the published plot inputs.

Active model defaults are defined in `Model`; `parameters.json` records the baseline settings.
