# Empirical estimation and calibrated option valuation

This folder contains the empirical code used by §4.5, likelihood Appendix C, and supplementary-estimation Appendix E of the manuscript. It includes the current Dec22/Dec23 workflow and separately labelled historical MATLAB/notebook sources.

This directory contains code and selected aggregate reference results. **Licensed raw futures observations, dated innovations, daily transformed prices, daily state-dependent risk tables, and observed-price figure artifacts are excluded.**

## Paper-to-code map

| Paper item | Entry / source | Result selection |
|---|---|---|
| Table 2: six estimated parameters | `src/run_pilot.py`, `src/pilot_estimator.py` | `indexed_scenario_separate_beta_binary_clock` |
| Figure 7: observed futures prices | `src/plot_price_history.py` | 256 common daily Dec22/Dec23 closes during 2021; requires licensed local data |
| Figure 8a: fitted factor risk rates | `src/calibrated_pricing.py`, then `src/plot_indexed_results.py` | `indexed_separate`, columns `date,z1,z2` |
| Figure 8b: model-implied call values | Same pricing/plotting entries | `indexed_separate`, 15 remaining-maturity points at the 30 June 2021 reference state |
| Appendix C: exact transition likelihood | `src/pilot_estimator.py` | `latent_states`, `integrated_clock`, `transition`, `profile_drifts`, `likelihood_components` |
| Appendix E: HAC(4/8/12) | `src/robust_inference.py` | `indexed_sensitivity` in the historical result naming scheme |
| Appendix E: common/separate β | `src/run_pilot.py` | `indexed_scenario_binary_clock` versus `indexed_scenario_separate_beta_binary_clock` |
| Appendix E: relaxed α and multiple starts | `src/run_pilot.py` | `indexed_scenario_separate_beta_positive_clock` versus the binary-clock counterpart |

The shared α is maintained in all current empirical specifications. The relaxed-α comparison changes its lower bound from 1 to 0.000001; it does not estimate two independent exponents or test α₁=α₂.

## Inputs and fixed calibration settings

Obtain the following workbooks through your own authorized data access and store them outside the repository:

| Contract | Instrument identifier in original workbook | Expected original filename |
|---|---|---|
| December 2022 EUA futures | `FEUAZ2^2` | `EUA FUTURE DEC 22价格历史_20260126_1449.xlsx` |
| December 2023 EUA futures | `FEUAZ3^2` | `EUA FUTURE DEC 23价格历史_20260126_1449.xlsx` |

The reader uses `Sheet 1`, column A for dates and column B for closing prices. The original header rows are 32 and 33, respectively. It detects valid date/positive numeric-close rows, sorts the common dates, and keeps calendar 2021. It does not clip factors or delete observations according to the sign of a transformed factor. The expected sample is 256 paired observations from 4 January to 30 December 2021 and 255 transitions. Source filenames and SHA-256 values are recorded in `reference/source_metadata.json`; equivalent re-exports can have different workbook hashes and should be documented before changing the figure integrity check.

The preserved scientific settings are π₁=121.01, π₂=128.71, κ=exp(-0.03), contract-expiry proxies 19 December 2022 and 18 December 2023, and ACT/365.25 time. The primary model has one α, separate β₁/β₂, separate historical drift shifts h₁/h₂, and correlated latent innovations. These are conditional calibration assumptions. The code does not establish that contract expiry equals a legal surrender date or identify a policy effect.

Risk-neutral valuation removes h₁/h₂ while retaining the latent drift 0.5 zⱼ ξⱼ and the fitted diffusion/dependence structure. The reference state is 30 June 2021, A₁=K=56.78, A₂=57.49, with valuation rate 3%. The archived model call values at 0.25, 0.50 and 1.00 years are 5.939265, 9.042207 and 15.254942. These are calculated option values, not observed option-market quotes.

## Run only an explicitly selected stage

From the repository root:

```text
python -m pip install -r empirical/requirements.txt
python empirical/run_empirical.py --help
```

Invoking the launcher without `--stage` prints help and performs no estimation. Supply your licensed data location using `--data-dir` or `CARBON_OPTION_DATA_DIR`. You can instead provide `--dec22-file` and `--dec23-file` for explicit filenames. Use `--output-dir` or `CARBON_OPTION_RESULTS_DIR` to choose a local results directory; the default is the ignored `empirical/results/` folder.

```text
python empirical/run_empirical.py --stage estimate --data-dir "<licensed-data-directory>"
python empirical/run_empirical.py --stage hac --data-dir "<licensed-data-directory>"
python empirical/run_empirical.py --stage price --data-dir "<licensed-data-directory>"
python empirical/run_empirical.py --stage fig7 --data-dir "<licensed-data-directory>"
python empirical/run_empirical.py --stage fig8
```

Use the same `--output-dir` for every stage when overriding the default. `fig8` reads the locally generated pricing files and does not itself read a workbook. `fig7` can be run without re-estimating; it uses the curated source hashes when no locally generated provenance file is present. Figure commands write only to the results directory and do not modify a manuscript folder.

The default `estimate` stage runs the three specifications required by the current table, common-β comparison, and relaxed-α comparison. It keeps the original optimizer, bounds, start points, transition integrals and likelihood. It does not run the old π=100 cases or the α-profile grid. The default `hac` stage evaluates the current indexed fit at Bartlett bandwidths 4, 8 and 12 and checks both original finite-difference steps; it does not run the additional 2.5%/4% rate re-estimation.

`--all-historical-scenarios` explicitly restores those additional original scenarios for the relevant stage. They are not needed for the current manuscript. This option should be used consistently across estimation and downstream stages because the extra pricing cases require the corresponding extra fit results.

The optional external-discount diagnostic can be invoked separately:

```text
python empirical/run_empirical.py --stage external-audit --data-dir "<licensed-data-directory>"
```

That stage reads local licensed futures and downloads public ECB AAA curve series if no local cache exists. It performs no model estimation. Its metadata is optional for HAC; missing metadata is explicitly marked as not run rather than replaced with an invented diagnostic.

For plot language, use `--language en`, `zh`, or `both`; English is the default. `--font` selects an installed font. Font fallback affects presentation only, and Chinese glyphs require a suitable installed font.

## Generated outputs

| Local output directory | Principal files |
|---|---|
| `results/estimation/` | `conditional_estimation_results.json`, `conditional_parameter_estimates.csv`, `sample_provenance.json`, diagnostics and dated innovations |
| `results/inference/` | `robust_inference_results.json/.csv` |
| `results/pricing/` | `calibrated_option_prices.csv`, `calibrated_pricing_results.json`, private `calibrated_risk_contributions.csv` |
| `results/figures/price_history/` | Figure 7 exports and integrity metadata; these display licensed observations |
| `results/figures/indexed/` | Figure 8 exports and plotting metadata |
| `results/ecb_cache/` | Optional public ECB curve cache |

Generated estimation/pricing JSON files can contain nested optimizer results, metadata and local paths. Daily innovations and the full risk CSV should stay local. The risk CSV includes bⱼ and state-dependent variance components for each observation date, not just deterministic time functions. Do not commit it as an apparently harmless transformed-data file. `.gitignore` excludes the default results tree; apply equivalent exclusions when choosing another output directory.

## Included aggregate reference results

`reference/` contains aggregate estimation and pricing results:

- `fit_summary.json`, `parameter_comparisons.csv`: three aggregate indexed fit summaries.
- `optimizer_runs.csv`: fitted parameter coordinates and objective values from individual starts, with no observation/residual arrays.
- `hac_standard_errors.csv`, `inference_summary.json`: current HAC estimates, intervals and numerical-stability summaries.
- `model_option_prices.csv`, `risk_pricing_summary.json`: the current model call curve and aggregate risk shares at the reported calibration settings.
- `source_metadata.json`: source filenames, instrument identities, hashes and sample counts, with local absolute paths removed.

These reference values support comparison with a future authorized rerun. Rebuilding the empirical figures requires the licensed observations and locally generated pricing results.

## Historical sources

`legacy/` contains the original related MATLAB sources and an output-cleared `EUA price.ipynb`. Their input paths use `CARBON_OPTION_DATA_DIR`; their required old datasets are not supplied. They concern other periods/series and are not the source of Table 2 or Figures 7–8. In particular, the old two-period MATLAB code clips transformed factors; the current Python estimator rejects out-of-domain inputs instead. The current Python workflow does not use the legacy clipping.

The legacy notebook requires `pandas` in addition to plotting/Excel packages. MATLAB sources may require Statistics and Machine Learning, Optimization or Econometrics Toolbox functions.
