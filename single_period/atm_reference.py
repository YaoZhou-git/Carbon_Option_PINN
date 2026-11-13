"""Standalone Gaussian-quadrature wrapper for the paper's Table 1 ATM scale.

This records the calculation previously used to normalize the three saved
diagnostics. It neither reloads a PINN nor changes their definitions.
"""
from __future__ import annotations

import json
import math

from scipy.integrate import quad
from scipy.special import ndtr, ndtri


def main() -> None:
    penalty, alpha, beta, rate = 100.0, 1.2, 0.5, 0.05
    compliance, expiry, initial, strike = 4.0, 3.0, 50.0, 50.0
    clock = beta / (1.0 - alpha) * (
        compliance ** (1.0 - alpha) - (compliance - expiry) ** (1.0 - alpha)
    )
    mean = math.exp(clock / 2.0) * ndtri(initial / penalty)
    deviation = math.sqrt(math.expm1(clock))
    cutoff = (ndtri(strike / penalty) - mean) / deviation

    def integrand(x: float) -> float:
        payoff = penalty * ndtr(mean + deviation * x) - strike
        return payoff * math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    undiscounted, quadrature_error = quad(
        integrand, cutoff, math.inf, epsabs=1e-11, epsrel=1e-11
    )
    value = math.exp(-rate * expiry) * undiscounted
    diagnostics = [-0.00884, 0.03117, -0.02640]
    print(json.dumps({
        "method": "one-dimensional conditional Gaussian quadrature",
        "parameters": {
            "pi": penalty, "alpha": alpha, "beta": beta, "r": rate,
            "T": compliance, "tau": expiry, "A0": initial, "K": strike
        },
        "atm_call_value": value,
        "quadrature_absolute_error_estimate": math.exp(-rate * expiry) * quadrature_error,
        "original_reported_diagnostics": diagnostics,
        "normalized_error_percent": [100.0 * abs(x) / value for x in diagnostics],
        "role": "Reference-scale wrapper; original diagnostics are read as reported literals."
    }, indent=2))


if __name__ == "__main__":
    main()
