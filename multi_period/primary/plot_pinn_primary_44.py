"""Draw the three saved-result figures for subsection 4.4, without model calls."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


PACKAGE = Path(__file__).resolve().parents[1]
OUT = PACKAGE / 'outputs' / 'primary_figures'
PREVIOUS = PACKAGE / "reference" / "data"
SAVED = PACKAGE / "primary" / "saved"
FIG = OUT / "fig"
SOURCE = OUT / "source_data"
CURVES = ["downward", "flat", "upward"]
CURVES_ZH = ["\u4e0b\u503e", "\u5e73\u5766", "\u4e0a\u503e"]
COLORS = ["#27649A", "#C17A29", "#41826B", "#8562A5"]
STYLES = ["-", "--", "-.", ":"]

# Contract: retain the distributional evidence and the independent validation;
# use PINN prices for both parameter groups and quote scans. Sobol results are
# reference points, and their sampling uncertainty is not PINN uncertainty.
# Preserve the three quantitative grids, their dimensions, and editable exports.
def load_data(baseline_path, scan_path, matched_path):
    hist = pd.read_csv(PREVIOUS / "terminal_distribution.csv")
    original = pd.read_csv(PREVIOUS / "matched_initial_variance.csv")
    check = pd.read_csv(PREVIOUS / "pinn_independent_recheck.csv")
    old_scans = pd.read_csv(PREVIOUS / "future_quote_scans.csv")
    baseline = pd.read_csv(baseline_path).set_index("curve").loc[CURVES]
    scans = pd.read_csv(scan_path).sort_values(["period", "quote"])
    matched = pd.read_csv(matched_path).set_index("curve").loc[CURVES[1:]]
    assert len(hist) == 372 and len(check) == 24
    assert len(scans) == 93 and scans.groupby("period").size().to_dict() == {2: 31, 3: 31, 4: 31}
    np.testing.assert_allclose(scans[["period", "quote", "C"]],
                               old_scans.sort_values(["period", "quote"])[["period", "quote", "C"]],
                               rtol=0, atol=1e-12)
    prices = pd.DataFrame({"curve": CURVES,
        "baseline_PINN": baseline.PINN_C.to_numpy(),
        "matched_PINN": [baseline.loc["downward", "PINN_C"], *matched.PINN_price.to_numpy()],
        "baseline_Sobol": original.C_original.to_numpy(),
        "matched_Sobol": original.C_matched.to_numpy()})
    assert np.isfinite(prices.iloc[:, 1:].to_numpy()).all()
    assert np.isfinite(scans.PINN_C.to_numpy()).all()
    return hist, prices, scans, check


def label(ax, letter):
    ax.text(-.12, 1.04, letter, transform=ax.transAxes, weight="bold", fontsize=11)


def save(fig, name):
    fig.tight_layout(pad=1.15, w_pad=1.7)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw(language, hist, prices, scans, check):
    zh = language == "ZH"
    plt.rcParams["font.family"] = "Microsoft YaHei" if zh else "Arial"
    suffix = "_ZH" if zh else ""
    names = CURVES_ZH if zh else ["Downward", "Flat", "Upward"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.95),
                             gridspec_kw={"width_ratios": [1.12, 1]})
    for j, name in enumerate(CURVES_ZH):
        s = hist[hist.curve == name]
        axes[0].plot((s.left+s.right)/2, s.density, color=COLORS[j],
                     ls=STYLES[j], label=names[j], lw=1.5)
    axes[0].axvline(50, color="#777777", ls=":", lw=.9)
    axes[0].text(88, .021, r"$K=50$", color="#555555", fontsize=8)
    axes[0].set(xlim=(0, 310),
        xlabel=r"期权到期时的 $A_1(\tau)$" if zh else r"First-period futures price $A_1(\tau)$",
        ylabel="概率密度" if zh else "Density")
    axes[0].legend(loc="upper right")
    x = np.arange(3)
    width = .35
    axes[1].bar(x-width/2, prices.baseline_PINN, width, color="#27649A",
                label="PINN：基准参数" if zh else "PINN: baseline")
    axes[1].bar(x+width/2, prices.matched_PINN, width, facecolor="#ACBBC8",
                edgecolor="#5B6D7C", hatch="///", linewidth=.45,
                label="PINN：匹配初始方差" if zh else "PINN: matched initial variance")
    for dx, column in [(-width/2, "baseline_Sobol"), (width/2, "matched_Sobol")]:
        axes[1].scatter(x+dx, prices[column], marker="o", facecolors="white",
                        edgecolors="#333333", linewidths=.65, s=15, zorder=4,
                        label=("Sobol 参考" if zh else "Sobol reference") if dx < 0 else None)
    for i, row in prices.iterrows():
        for dx, value in [(-width/2, row.baseline_PINN), (width/2, row.matched_PINN)]:
            axes[1].text(i+dx, value+.3, f"{value:.2f}", ha="center", fontsize=7)
    axes[1].set(xticks=x, xticklabels=names, ylim=(0, 22.5),
                ylabel="看涨期权价值" if zh else "Call value")
    axes[1].legend(loc="upper right", fontsize=6.7)
    for j, a in enumerate(axes):
        label(a, chr(97+j))
    save(fig, "multiperiod_curve_distribution_pinn_primary"+suffix)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharey=True)
    for j, a in enumerate(axes):
        s = scans[scans.period == j+2]
        a.plot(s.quote, s.PINN_C, color=COLORS[j], lw=1.6, label="PINN")
        a.scatter(s.quote, s.C, facecolors="white", edgecolors=COLORS[j],
                  linewidths=.6, s=12, zorder=3,
                  label="Sobol 参考" if zh else "Sobol reference")
        a.scatter([45, 30, 20][j], prices.baseline_PINN.iloc[0], color=COLORS[j],
                  edgecolor="white", linewidth=.5, s=27, zorder=4)
        a.set(xlabel=f"第 {j+2} 期报价 $A_{j+2}$" if zh else f"Futures quote $A_{j+2}$")
        label(a, chr(97+j))
    axes[0].set_ylabel("看涨期权价值" if zh else "Call value")
    axes[0].legend(loc="lower left", fontsize=7.1)
    save(fig, "multiperiod_quote_scans_pinn_primary"+suffix)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.95))
    markers = ["o", "s", "^", "D"]
    for j in range(4):
        s = check[check.N == j+1]
        axes[0].plot(s.A1, s.new_QMC, color=COLORS[j], ls=STYLES[j],
                     label=f"$N={j+1}$", lw=1.3)
        axes[0].scatter(s.A1, s.saved_PINN, color=COLORS[j], marker=markers[j],
                        facecolor="none", s=28, linewidth=.9)
        axes[1].plot(s.A1, abs(s.PINN_minus_new_QMC), color=COLORS[j], ls=STYLES[j],
                     marker=markers[j], ms=3.6, label=f"$N={j+1}$", lw=1.2)
    axes[0].legend(ncol=2, loc="upper left")
    axes[0].text(.04, .61, "线：参考价格\n点：PINN" if zh else "Lines: reference\nMarkers: PINN",
                 transform=axes[0].transAxes, fontsize=7.5)
    axes[0].set_ylabel("看涨期权价值" if zh else "Call value")
    axes[1].set_ylabel("绝对价格差" if zh else "Absolute price difference")
    axes[1].legend(ncol=2, loc="upper right")
    for j, a in enumerate(axes):
        a.set(xlabel="第一期报价 $A_1$" if zh else "First-period futures quote $A_1$",
              xticks=[40, 50, 60, 70, 80, 90])
        label(a, chr(97+j))
    save(fig, "multiperiod_pinn_validation"+suffix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=SAVED/"three_curve_prices_and_gradients.csv")
    parser.add_argument("--scans", type=Path, default=SAVED/"future_quote_scans_pinn_exact_saved_points.csv")
    parser.add_argument("--matched", type=Path, default=SAVED/"matched_pinn_prices.csv")
    args = parser.parse_args()
    hist, prices, scans, check = load_data(args.baseline, args.scans, args.matched)
    FIG.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    for path in [PREVIOUS/"terminal_distribution.csv", PREVIOUS/"matched_initial_variance.csv",
                 PREVIOUS/"pinn_independent_recheck.csv", PREVIOUS/"future_quote_scans.csv",
                 args.baseline, args.scans, args.matched]:
        target = SOURCE/path.name
        if path.resolve() != target.resolve():
            shutil.copy2(path, target)
    prices.to_csv(SOURCE/"figure_curve_prices.csv", index=False)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
        "legend.frameon": False, "legend.fontsize": 7.5, "pdf.fonttype": 42,
        "svg.fonttype": "none", "axes.unicode_minus": False,
        "mathtext.fontset": "dejavusans", "savefig.facecolor": "white"})
    draw("EN", hist, prices, scans, check)
    draw("ZH", hist, prices, scans, check)
    manifest = {"figure_count": 3, "languages": ["EN", "ZH"],
        "formats": ["pdf", "svg", "png"], "new_training": False,
        "new_integration": False, "distribution_method": "Saved Gaussian-Sobol histogram",
        "price_method": "PINN", "reference_markers": "Saved Sobol prices",
        "scan_uncertainty_band": False, "validation_states": 24,
        "validation_max_relative_error_percent": float(check.relative_absolute_difference.max()*100),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in SOURCE.glob("*.csv")}}
    (OUT/"figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figures": len(list(FIG.glob("*"))), "prices": prices.to_dict("records")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
