"""Four pre-specified robustness checks for the revised carbon-option paper.

This script reads the frozen 2026-09-17 N=4 PINN checkpoint and the published
model implementation. It never trains a model or changes the original package.
All outputs are theoretical scenario values; no option market quotes are used.

Run from any directory with Python 3.10+ and numpy, scipy, torch installed:
    python run_robustness.py
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PACKAGE = ROOT / "core"
OUTPUT = ROOT / "outputs" / "auxiliary"
BASELINE = ROOT / "models" / "baseline"
sys.path.insert(0, str(PACKAGE))
from factor_model import (  # noqa: E402
    Model,
    conditional_call,
    covariance,
    factors,
    matrix,
    normal_grid,
    value,
)
from factor_pinn import FactorPINN  # noqa: E402

BASE = Model()
FORWARD = (60.0, 45.0, 30.0, 20.0)
ORDER = 40
INNER = 160


def save_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {name}")
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generalized_value(
    quotes: tuple[float, ...] | list[float],
    cfg: Model,
    penalty_path: tuple[float, ...],
    *,
    continuation: float = 0.0,
    terminal_kappa: float = 0.0,
    t: float = 0.0,
    order: int = ORDER,
    inner: int = INNER,
) -> float:
    """Conditional Gaussian call value with deterministic pi_j and terminal tail.

    For n=4 and continuation B5, A4=X4+terminal_kappa*B5. B5 is constant
    throughout the option's remaining life and is not an observed fifth quote.
    X1..X3 follow the same recursion as in the original N=4 specification.
    """
    a = np.asarray(quotes, dtype=float)
    n = len(a)
    if n < 1 or n > 4:
        raise ValueError("This check covers N=1,...,4")
    pi = np.asarray(penalty_path[:n], dtype=float)
    if len(pi) != n or np.any(pi <= 0) or pi[0] != cfg.pi:
        raise ValueError("The positive penalty path must have pi_1=100")
    x = a.copy()
    if n > 1:
        x[:-1] -= np.asarray(cfg.kappa[: n - 1]) * a[1:]
    x[-1] -= terminal_kappa * continuation
    if np.any(x <= 0.0) or np.any(x >= pi):
        raise ValueError(f"Outside model domain: X={x}, pi={pi}")
    if not 0.0 <= t < cfg.tau:
        raise ValueError("The valuation time must precede option maturity")

    L = matrix(cfg, n)
    tail_weight = float(np.prod(cfg.kappa[: n - 1]) * terminal_kappa)
    tail_value = tail_weight * continuation
    assert np.isclose(float(L[0] @ x + tail_value), a[0], atol=1e-12)
    scales, V = covariance(cfg, n, float(t), float(cfg.tau))
    m = scales * ndtri(x / pi)
    if n == 2:
        # The one-dimensional future contribution has a kink where it alone
        # reaches K. The source model uses adaptive quadrature for this case;
        # a modest Hermite grid would bias the generalized N=2 comparison.
        sd_future = float(np.sqrt(V[1, 1]))
        slope = float(V[0, 1] / sd_future)
        conditional_var = float(V[0, 0] - slope * slope)
        future_weight = float(L[0, 1])

        def integrand(z: float) -> float:
            future_value = future_weight * pi[1] * ndtr(m[1] + sd_future * z)
            offset = np.array([future_value + tail_value])
            conditional_mean = np.array([m[0] + slope * z])
            call = conditional_call(conditional_mean, conditional_var, offset, cfg, inner)[0]
            return float(call * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi))

        break_fraction = (cfg.K - tail_value) / (future_weight * pi[1])
        points = []
        if 0.0 < break_fraction < 1.0:
            kink = float((ndtri(break_fraction) - m[1]) / sd_future)
            if -9.0 < kink < 9.0:
                points = [kink]
        expectation = quad(integrand, -9.0, 9.0, epsabs=2e-7,
                           epsrel=2e-8, points=points, limit=200)[0]
        return float(np.exp(-cfg.r * (cfg.tau - t)) * expectation)
    if n == 1:
        mu = np.array([m[0]])
        var = float(V[0, 0])
        b = np.array([tail_value])
        weights = np.array([1.0])
    else:
        nodes, weights = normal_grid(n - 1, order)
        VF = V[1:, 1:]
        yf = m[1:] + nodes @ np.linalg.cholesky(VF).T
        coef = np.linalg.solve(VF, V[1:, 0])
        mu = m[0] + (yf - m[1:]) @ coef
        var = float(V[0, 0] - V[0, 1:] @ coef)
        b = (ndtr(yf) * pi[1:]) @ L[0, 1:] + tail_value
    expected = conditional_call(mu, var, b, cfg, inner)
    return float(np.exp(-cfg.r * (cfg.tau - t)) * (weights @ expected))


def reference_holdout_states() -> list[dict]:
    """Independent, scrambled Sobol draws; none belong to training batches."""
    broad = qmc.Sobol(d=5, scramble=True, seed=20260923).random_base2(m=5)
    local_candidates = qmc.Sobol(d=5, scramble=True, seed=20260924).random_base2(m=9)
    rows = []
    for i, u in enumerate(broad):
        x = BASE.pi * (0.025 + 0.95 * u[:4])
        a = matrix(BASE, 4) @ x
        t = 0.0 if i < 16 else 2.5 * float(u[4])
        rows.append({"group": "broad_factor_domain", "index": i + 1, "A": a, "X": x, "t": t})
    local = []
    for u in local_candidates:
        a = np.array([40 + 60 * u[0], 20 + 40 * u[1], 10 + 30 * u[2], 5 + 20 * u[3]])
        try:
            x = factors(a, BASE)
        except ValueError:
            continue
        local.append((a, x, float(u[4])))
        if len(local) == 32:
            break
    if len(local) != 32:
        raise RuntimeError("Insufficient admissible local held-out draws")
    for i, (a, x, time_u) in enumerate(local):
        t = 0.0 if i < 16 else 2.5 * time_u
        rows.append({"group": "economic_curve_box", "index": i + 1, "A": a, "X": x, "t": t})
    return rows


def check_n4_holdout() -> tuple[list[dict], list[dict], dict]:
    path = BASELINE / "N4_factor_pinn.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["N"] == 4
    net = FactorPINN(BASE, 4).cpu().eval()
    net.load_state_dict(checkpoint["state_dict"])
    states = reference_holdout_states()
    rows = []
    for i, item in enumerate(states):
        a, x, t = item["A"], item["X"], item["t"]
        tensor = torch.from_numpy(np.asarray([np.r_[x / BASE.pi, t]], dtype=np.float32))
        with torch.no_grad():
            pinn = float(net(tensor).item() * BASE.pi)
        gaussian = value(tuple(a), BASE, t=t, order=ORDER, inner=INNER)
        lower = float(np.exp(-BASE.r * (BASE.tau - t)) * max(a[0] - BASE.K, 0.0))
        upper = float(np.exp(-BASE.r * (BASE.tau - t)) * a[0])
        abs_error = abs(pinn - gaussian)
        rows.append({
            "group": item["group"], "sample": item["index"], "t": t,
            **{f"X{j+1}": float(x[j]) for j in range(4)},
            **{f"A{j+1}": float(a[j]) for j in range(4)},
            "PINN_C": pinn, "Gaussian_C": gaussian,
            "absolute_error": abs_error,
            "relative_error": abs_error / gaussian if gaussian > 0 else float("nan"),
            "discounted_intrinsic_bound": lower,
            "discounted_underlying_bound": upper,
            "lower_bound_violation": int(pinn < lower - 1e-6),
            "upper_bound_violation": int(pinn > upper + 1e-6),
        })
        if (i + 1) % 16 == 0:
            print(f"N=4 holdout reference: {i+1}/{len(states)}", flush=True)
    save_csv("n4_heldout_states.csv", rows)
    summary = []
    for group in ("broad_factor_domain", "economic_curve_box", "all"):
        sub = [r for r in rows if group == "all" or r["group"] == group]
        errors = np.array([r["absolute_error"] for r in sub])
        rel = np.array([r["relative_error"] for r in sub])
        summary.append({
            "group": group, "states": len(sub),
            "MAE": float(np.mean(errors)),
            "RMSE": float(np.sqrt(np.mean(errors * errors))),
            "max_absolute_error": float(np.max(errors)),
            "max_relative_error": float(np.max(rel)),
            "max_relative_error_reference_C": float(sub[int(np.argmax(rel))]["Gaussian_C"]),
            "lower_bound_violations": sum(r["lower_bound_violation"] for r in sub),
            "upper_bound_violations": sum(r["upper_bound_violation"] for r in sub),
            "max_lower_bound_shortfall": max(
                max(r["discounted_intrinsic_bound"] - r["PINN_C"], 0.0) for r in sub
            ),
            "max_upper_bound_excess": max(
                max(r["PINN_C"] - r["discounted_underlying_bound"], 0.0) for r in sub
            ),
        })
    save_csv("n4_holdout_summary.csv", summary)
    # A transparent quadrature refinement check at the worst eight points.
    worst = sorted(rows, key=lambda row: row["absolute_error"], reverse=True)[:8]
    refine = []
    for row in worst:
        a = tuple(row[f"A{j}"] for j in range(1, 5))
        finer = value(a, BASE, t=row["t"], order=48, inner=192)
        refine.append({
            "group": row["group"], "sample": row["sample"],
            "Gaussian_40x160": row["Gaussian_C"], "Gaussian_48x192": finer,
            "absolute_refinement_change": abs(finer - row["Gaussian_C"]),
        })
    save_csv("n4_reference_refinement.csv", refine)
    return rows, summary, {"max_reference_refinement_change": max(x["absolute_refinement_change"] for x in refine)}


def check_terminal_closure() -> list[dict]:
    terminal_kappa = float(np.exp(-0.2))
    n3 = value(FORWARD[:3], BASE, order=ORDER, inner=INNER)
    baseline = value(FORWARD, BASE, order=ORDER, inner=INNER)
    rows = []
    for b5 in (0.0, 8.0, 12.0, 16.0):
        result = generalized_value(
            FORWARD, BASE, (100.0,) * 4,
            continuation=b5, terminal_kappa=terminal_kappa,
        )
        rows.append({
            "B5_deterministic": b5, "kappa4": terminal_kappa,
            "X4": FORWARD[3] - terminal_kappa * b5,
            "C4": result, "difference_from_terminal_closure": result - baseline,
            "C3_fixed": n3, "C4_minus_C3": result - n3,
        })
    if abs(rows[0]["C4"] - baseline) > 0.003:
        raise AssertionError("Generalized closure does not reproduce baseline")
    save_csv("terminal_closure.csv", rows)
    return rows


def check_heterogeneous_parameters() -> list[dict]:
    cases = [
        ("common_baseline", (1.2,) * 4, (0.8,) * 4),
        ("later_alpha_lower", (1.2, 1.05, 1.10, 1.15), (0.8,) * 4),
        ("later_alpha_higher", (1.2, 1.25, 1.30, 1.35), (0.8,) * 4),
        ("later_beta_lower", (1.2,) * 4, (0.8, 0.72, 0.64, 0.56)),
        ("later_beta_higher", (1.2,) * 4, (0.8, 0.88, 0.96, 1.04)),
    ]
    rows = []
    for name, alphas, betas in cases:
        cfg = replace(BASE, alpha=alphas, beta=betas)
        cs = [value(FORWARD[:n], cfg, order=ORDER, inner=INNER) for n in range(1, 5)]
        rows.append({
            "case": name, "alpha": ",".join(str(x) for x in alphas),
            "beta": ",".join(str(x) for x in betas),
            **{f"C{j+1}": cs[j] for j in range(4)},
            "C2_minus_C1": cs[1] - cs[0],
            "C3_minus_C2": cs[2] - cs[1],
            "C4_minus_C3": cs[3] - cs[2],
        })
    save_csv("heterogeneous_parameters.csv", rows)
    return rows


def check_penalty_path() -> list[dict]:
    cases = [
        ("common_100", (100.0, 100.0, 100.0, 100.0)),
        ("rising_5_per_period", (100.0, 105.0, 110.0, 115.0)),
        ("falling_5_per_period", (100.0, 95.0, 90.0, 85.0)),
    ]
    baseline = [value(FORWARD[:n], BASE, order=ORDER, inner=INNER) for n in range(1, 5)]
    rows = []
    for name, pi in cases:
        cs = [generalized_value(FORWARD[:n], BASE, pi) for n in range(1, 5)]
        xs = factors(FORWARD, BASE)
        assert all(xs[j] < pi[j] for j in range(4))
        rows.append({
            "case": name, "penalty_path": ",".join(str(int(x)) for x in pi),
            "min_distance_to_upper_factor_boundary": min(float(pi[j] - xs[j]) for j in range(4)),
            **{f"C{j+1}": cs[j] for j in range(4)},
            "C4_difference_from_common": cs[3] - baseline[3],
            "C4_minus_C3": cs[3] - cs[2],
        })
    if max(abs(rows[0][f"C{n}"] - baseline[n - 1]) for n in range(1, 5)) > 0.003:
        raise AssertionError("Generalized penalties do not reproduce baseline")
    save_csv("deterministic_penalty_paths.csv", rows)
    return rows


def latex_table(name: str, caption: str, header: list[str], rows: list[list[str]],
                *, label: str, columns: str | None = None, note: str = "") -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering\\small\\setlength{\\tabcolsep}{3pt}",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        "\\begin{tabular}{" + (columns or "l" + "r" * (len(header) - 1)) + "}",
        "\\toprule",
        " & ".join(header) + " " + "\\" * 2,
        "\\midrule",
    ]
    lines.extend(" & ".join(row) + " " + "\\" * 2 for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    if note:
        lines.extend(["\\par\\smallskip\\footnotesize " + note])
    lines.extend(["\\end{table}", ""])
    (OUTPUT / name).write_text("\n".join(lines), encoding="utf-8")


def make_tables(holdout: list[dict], closure: list[dict], hetero: list[dict], penalties: list[dict]) -> None:
    latex_table(
        "table_robustness_n4_holdout.tex",
        "Independent four-factor PINN evaluation on admissible held-out states.",
        ["Region", "$n$", "MAE", "RMSE", "Max.", "Max. (\\%)", "Viol.", "Short."],
        [[{"broad_factor_domain": "Broad",
           "economic_curve_box": "Curve box", "all": "All"}[r["group"]],
          str(r["states"]), f'{r["MAE"]:.4f}',
          f'{r["RMSE"]:.4f}', f'{r["max_absolute_error"]:.4f}',
          f'{100*r["max_relative_error"]:.2f}',
          str(r["lower_bound_violations"] + r["upper_bound_violations"]),
          f'{r["max_lower_bound_shortfall"]:.4f}'] for r in holdout],
        label="tab:robust-holdout",
        note=("Broad: 32 interior factor-domain draws; curve box: 32 admissible "
              "forward-curve draws. Max. is maximum absolute price error; Max. "
              "(\\%) is maximum pointwise relative error against the positive Gaussian "
              "price. Viol. counts violations of either call price bound, and Short. "
              "is maximum discounted intrinsic-value shortfall. All nine violations "
              "are lower-bound failures in deep-in-the-money broad draws; no "
              "upper-bound failures occurred."),
    )
    latex_table(
        "table_robustness_closure.tex",
        "First-period option value under deterministic final-period continuation scenarios.",
        ["$B_5$", "$X_4$", "$C_4$", "$C_4-C_4^{(0)}$", "$C_4-C_3$"],
        [[f'{r["B5_deterministic"]:.0f}', f'{r["X4"]:.3f}', f'{r["C4"]:.3f}',
          f'{r["difference_from_terminal_closure"]:+.3f}', f'{r["C4_minus_C3"]:+.3f}'] for r in closure],
        label="tab:robust-closure",
        note=("$A_4=X_4+\\kappa_4 B_5$ with $\\kappa_4=e^{-0.2}$ and fixed "
              "$(A_1,A_2,A_3,A_4)=(60,45,30,20)$. $B_5$ is a deterministic "
              "external continuation input, not an observed fifth contract."),
    )
    latex_table(
        "table_robustness_heterogeneity.tex",
        "Illustrative period-specific diffusion intensities, with $\\alpha_j\\geq1$.",
        ["Scenario", "$(\\alpha_2,\\alpha_3,\\alpha_4)$",
         "$(\\beta_2,\\beta_3,\\beta_4)$", "$C_2$", "$C_3$", "$C_4$", "$C_4-C_3$"],
        [[{"common_baseline": "Common", "later_alpha_lower": "Lower $\\alpha$",
           "later_alpha_higher": "Higher $\\alpha$", "later_beta_lower": "Lower $\\beta$",
           "later_beta_higher": "Higher $\\beta$"}[r["case"]],
          "(" + ",".join(r["alpha"].split(",")[1:]) + ")",
          "(" + ",".join(r["beta"].split(",")[1:]) + ")",
          *[f'{r[f"C{n}"]:.3f}' for n in range(2, 5)],
          f'{r["C4_minus_C3"]:+.3f}'] for r in hetero],
        label="tab:robust-hetero",
        columns="lccrrrr",
        note=("$\\alpha_1=1.2$, $\\beta_1=0.8$ and $C_1=17.065$ in every "
              "row; all other parameters and the forward curve are fixed. "
              "All $\\alpha_j\\geq1$. These are sensitivity settings, not fitted parameters."),
    )
    latex_table(
        "table_robustness_penalty_paths.tex",
        "Deterministic penalty-scale paths, holding $\\pi_1=100$ and futures quotes fixed.",
        ["$\\boldsymbol{\\pi}$", "$C_1$", "$C_2$", "$C_3$", "$C_4$", "$C_4-C_4^{(0)}$"],
        [["(" + r["penalty_path"] + ")", *[f'{r[f"C{n}"]:.3f}' for n in range(1, 5)],
          f'{r["C4_difference_from_common"]:+.3f}'] for r in penalties],
        label="tab:robust-penalty",
        note=("All recursive factors remain strictly inside $(0,\\pi_j)$. "
              "These stylized paths are not matched to reporting-year penalty schedules."),
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not PACKAGE.exists():
        raise FileNotFoundError(PACKAGE)
    holdout_rows, holdout_summary, refinement = check_n4_holdout()
    closure = check_terminal_closure()
    hetero = check_heterogeneous_parameters()
    penalties = check_penalty_path()
    make_tables(holdout_summary, closure, hetero, penalties)
    meta = {
        "source_checkpoint": str(BASELINE / "N4_factor_pinn.pt"),
        "source_model": str(PACKAGE / "factor_model.py"),
        "no_retraining": True,
        "no_option_market_data": True,
        "no_N_greater_than_3_MC_benchmark": True,
        "Gaussian_quadrature_order": ORDER,
        "conditional_inner_order": INNER,
        "heldout_seed_broad": 20260923,
        "heldout_seed_economic_curve": 20260924,
        "heldout_count": len(holdout_rows),
        "reference_refinement": refinement,
        "baseline": {
            "A": FORWARD, "tau": BASE.tau, "T": BASE.T,
            "pi": BASE.pi, "K": BASE.K, "r": BASE.r,
            "alpha": BASE.alpha, "beta": BASE.beta,
            "rho": BASE.rho, "kappa": BASE.kappa,
        },
    }
    (OUTPUT / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({
        "holdout": holdout_summary,
        "refinement": refinement,
        "closure": closure,
        "heterogeneity": hetero,
        "penalty_paths": penalties,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
