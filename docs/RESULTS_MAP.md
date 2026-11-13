# Manuscript results and code entries

Calculation entries for the manuscript figures, tables and supplementary results.

| Manuscript item | Location in this folder | Role / method |
|---|---|---|
| Figure 1 | Descriptive PINN framework diagram | No scientific calculation entry is required for this schematic. |
| Figure 2 / Section 4.1 | single_period/notebooks/A_distribution_simulation.ipynb | Price-path simulation and KDE. The saved model-generated image matches the manuscript asset by SHA256; the original simulation does not fix a random seed. |
| Figure 3 / Section 4.2 | single_period/notebooks/compare.ipynb and associated saved runs | Original candidate PINN/MC comparison sources and exact archived manuscript image. The saved record does not establish the exact checkpoint/MC pair behind that image. See the single-period README. |
| Table 1 original diagnostics | single_period/notebooks/Result PINN Solution.ipynb; reference_tables/ | Preserved diagnostic source and August checkpoint. Original definitions and legacy parameter fallback are documented. |
| Table 1 ATM normalization | single_period/atm_reference.py | A standalone wrapper recording the previously used one-dimensional conditional-Gaussian quadrature: 10.129459296835025 and percentages 0.0872702, 0.3077163, 0.2606260. This does not redefine the three original diagnostics. |
| Figure 4 / Section 4.3 | single_period/notebooks/Sensitivity Analysis .ipynb | Reads the seven saved maturity runs. The archived image matches the manuscript asset by SHA256. |
| Figure 5(a) | multi_period/reference/scripts/run_analysis.py | Exact latent transitions with scrambled Sobol integration; frozen distribution input supplied. |
| Figure 5(b) | multi_period/primary/; models/baseline/ and models/matched/ | Existing four-factor PINN predictions for baseline and variance-matched parameters. |
| Figure 5 final plotting | multi_period/restored_plot/plot_restored_figure5.py | Draws the saved distribution and PINN price panels. |
| Figure 6 / quote sensitivities | multi_period/primary/infer_existing_pinn.py; economic_plots/plot_pinn_economic_44.py | Existing-network prices and automatic derivatives, with other futures quotes fixed. |
| Appendix D prices and Greeks | multi_period/primary/source_data/; primary/plot_pinn_primary_44.py | Five principal price cases, reference-state Delta comparison, and archived validation plot. |
| Appendix D current 24-point references | multi_period/reference/scripts/supplement_analysis.py | Independently scrambled Sobol integration. Distinct from the old deterministic quadrature export in multi_period/legacy/. |
| Factor-risk freezing and cross-N references | multi_period/reference/scripts/run_analysis.py | Conditional-Gaussian Sobol calculations; terminal closure changes with N. |
| Continuation / penalty-path auxiliary numbers | multi_period/auxiliary/run_robustness.py | Deterministic conditional-Gaussian quadrature, order 40 / inner 160. These particular saved outputs are not Sobol estimates. |
| Figure 7 / historical sample | empirical/src/plot_price_history.py | Requires separately obtained licensed observations; no original price plot is included. |
| Table 2 | empirical/src/run_pilot.py and pilot_estimator.py | Joint two-period likelihood; current indexed separate-scale specification. |
| Figure 8 | empirical/src/calibrated_pricing.py and plot_indexed_results.py | Conditional Gaussian pricing and risk timing under the fitted diffusion. Observation-level inputs/results stay local. |
| Appendix C | empirical/src/pilot_estimator.py | State transform, exact transition moments, profiled drift and price likelihood. |
| Appendix E | empirical/src/robust_inference.py and run_pilot.py | HAC summaries, common/separate beta and relaxed-alpha comparisons. |
| Black (1976) discussion control | controls/black76/black76_demo.py | Optional analytic futures-option control; not a carbon PINN experiment. |

## Training entries

- Single-period: output-cleared source and extracted opt-in cells in single_period/entrypoints/. The baseline training source and the historical Table 1 checkpoint are different archived versions; their distinction is preserved.
- N=1--4: multi_period/core/factor_pinn.py, with original architecture and recorded training settings.
- Matched initial variance: multi_period/primary/train_matched.py, warm-starting the baseline and restoring the scenario-specific beta buffers.

See each component README for commands and configuration details.
