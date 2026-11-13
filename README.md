# Carbon Option Pricing

Code for *Physics-Informed Neural Networks for carbon options: Multi-period pricing and Intertemporal Risk*.

[中文说明](README_ZH.md)

## Contents

| Folder | Contents |
|---|---|
| [single_period](single_period/README.md) | Single-period PINN and Monte Carlo models, allowance-price dynamics, maturity analysis and saved results |
| [multi_period](multi_period/README.md) | One- to four-factor models, PINN training and inference, variance matching, quote sensitivities and reference pricing |
| [empirical](empirical/README.md) | Two-period likelihood estimation, inference, calibrated pricing and empirical figures |
| [controls/black76](controls/black76/README.md) | Black (1976) futures-option maturity comparison |
| [data](data/README.md) | Market-data access and input requirements |
| [docs/RESULTS_MAP.md](docs/RESULTS_MAP.md) | Manuscript figures and tables mapped to calculation entries |

## Installation

Use Python 3.10 or newer. The PINN inference environment used Python 3.10.16 and PyTorch 2.7.0. Install the CPU or CUDA build appropriate for your machine.

```text
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment before installing dependencies: `.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on Linux/macOS.

For the notebooks:

```text
python -m pip install -r requirements-notebooks.txt
```

The MATLAB sources in `empirical/legacy/` require MATLAB and the relevant statistical and optimization toolboxes.

## Running the code

Follow each component's README for inference, plotting, training and estimation commands. Saved checkpoints and model-generated reference outputs are included. Plotting scripts use saved numerical inputs; training and estimation have separate entry points.

The single-period dynamics and two-period banking/withdrawal construction follow the Carmona--Hinz reduced-form framework. The multi-period implementation uses the recursive factor model described in the manuscript. PINN predictions, automatic derivatives, Sobol integration and deterministic Gaussian quadrature are identified in the result map.

## Market data

The empirical analysis uses Refinitiv Datastream observations accessed through an institutional subscription. Original observations and observation-level transformations are subject to the provider's licence and are not redistributed. Researchers with appropriate access can supply the data through the interface documented in [empirical/README.md](empirical/README.md).

Aggregate estimation summaries and model-generated reference outputs are included. Keep licensed inputs and observation-level results outside version control.

## Citation

Please cite the associated manuscript when using this code.
