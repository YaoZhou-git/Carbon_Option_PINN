"""Redraw two economic-result figures using saved PINN prices only."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


PACKAGE = Path(__file__).resolve().parents[1]
OUT = PACKAGE / 'outputs' / 'economic_figures'
INPUT = PACKAGE / "primary" / "source_data"
FIG = OUT / "fig"
CURVES = ["downward", "flat", "upward"]
CURVES_ZH = ["\u4e0b\u503e", "\u5e73\u5766", "\u4e0a\u503e"]
COLORS = ["#27649A", "#C17A29", "#41826B"]
INPUTS = [INPUT / "figure_curve_prices.csv",
          INPUT / "future_quote_scans_pinn_exact_saved_points.csv"]

# Figure contract: the saved PINN prices show the effects of the futures-curve
# shape and the initial-variance matching; quote scans show the response to
# each later-period quote. These are quantitative grids using the established
# Python style, with editable PDF/SVG and 300-dpi PNG exports in EN and ZH.
# No model evaluation, integration, smoothing, or price modification is used.
def load_data():
    prices = pd.read_csv(INPUTS[0], usecols=["curve", "baseline_PINN", "matched_PINN"])
    prices = prices.set_index("curve").loc[CURVES].reset_index()
    scans = pd.read_csv(INPUTS[1], usecols=["period", "quote", "PINN_C"])
    scans = scans.sort_values(["period", "quote"])
    assert len(prices) == 3 and not prices.curve.duplicated().any()
    assert len(scans) == 93
    assert scans.groupby("period").size().to_dict() == {2: 31, 3: 31, 4: 31}
    assert np.isfinite(prices[["baseline_PINN", "matched_PINN"]].to_numpy()).all()
    assert np.isfinite(scans[["quote", "PINN_C"]].to_numpy()).all()
    assert prices.loc[0, "baseline_PINN"] == prices.loc[0, "matched_PINN"]
    return prices, scans


def panel_label(ax, letter):
    ax.text(-.12, 1.04, letter, transform=ax.transAxes, weight="bold", fontsize=11)


def save(fig, name):
    fig.tight_layout(pad=1.15, w_pad=1.7)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw(language, prices, scans):
    zh = language == "ZH"
    plt.rcParams["font.family"] = "Microsoft YaHei" if zh else "Arial"
    suffix = "_ZH" if zh else ""
    names = CURVES_ZH if zh else ["Downward", "Flat", "Upward"]

    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    x = np.arange(3)
    width = .35
    ax.bar(x-width/2, prices.baseline_PINN, width, color=COLORS[0],
           label="\u57fa\u51c6\u53c2\u6570" if zh else "Baseline")
    ax.bar(x+width/2, prices.matched_PINN, width, facecolor="#ACBBC8",
           edgecolor="#5B6D7C", hatch="///", linewidth=.45,
           label="\u5339\u914d\u521d\u59cb\u65b9\u5dee" if zh else "Matched initial variance")
    for i, row in prices.iterrows():
        for dx, value in [(-width/2, row.baseline_PINN), (width/2, row.matched_PINN)]:
            ax.text(i+dx, value+.3, f"{value:.2f}", ha="center", fontsize=7)
    ax.set(xticks=x, xticklabels=names, ylim=(0, 22.5),
           ylabel="\u770b\u6da8\u671f\u6743\u4ef7\u503c" if zh else "Call value")
    ax.legend(loc="upper right", fontsize=7.5)
    save(fig, "multiperiod_curve_prices_pinn_only"+suffix)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharey=True)
    for j, ax in enumerate(axes):
        scan = scans[scans.period == j+2]
        ax.plot(scan.quote, scan.PINN_C, color=COLORS[j], lw=1.6)
        ax.scatter([45, 30, 20][j], prices.baseline_PINN.iloc[0], color=COLORS[j],
                   edgecolor="white", linewidth=.5, s=27, zorder=4)
        ax.set(xlabel=f"\u7b2c {j+2} \u671f\u62a5\u4ef7 $A_{j+2}$" if zh else f"Futures quote $A_{j+2}$")
        panel_label(ax, chr(97+j))
    axes[0].set_ylabel("\u770b\u6da8\u671f\u6743\u4ef7\u503c" if zh else "Call value")
    save(fig, "multiperiod_quote_scans_pinn_only"+suffix)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    input_hashes = {path.name: sha256(path) for path in INPUTS}
    prices, scans = load_data()
    FIG.mkdir(parents=True, exist_ok=True)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
        "legend.frameon": False, "legend.fontsize": 7.5, "pdf.fonttype": 42,
        "svg.fonttype": "none", "axes.unicode_minus": False,
        "mathtext.fontset": "dejavusans", "savefig.facecolor": "white"})
    draw("EN", prices, scans)
    draw("ZH", prices, scans)
    assert input_hashes == {path.name: sha256(path) for path in INPUTS}
    outputs = sorted(path.name for path in FIG.iterdir() if path.suffix in [".pdf", ".svg", ".png"])
    manifest = {
        "figure_count": 2,
        "languages": ["EN", "ZH"],
        "formats": ["pdf", "svg", "png"],
        "price_method": "saved PINN outputs",
        "new_training": False,
        "new_inference": False,
        "new_integration": False,
        "source_sha256": input_hashes,
        "figure_5_input_columns": ["curve", "baseline_PINN", "matched_PINN"],
        "figure_6_input_columns": ["period", "quote", "PINN_C"],
        "baseline_markers": {"A2": 45, "A3": 30, "A4": 20,
                             "PINN_C": float(prices.baseline_PINN.iloc[0])},
        "outputs": outputs,
    }
    (OUT / "economic_figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_count": len(outputs), "saved_PINN_prices": prices.to_dict("records")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
