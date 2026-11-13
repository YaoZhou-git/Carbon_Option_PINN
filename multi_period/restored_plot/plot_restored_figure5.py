"""Restore Figure 5 from saved histogram and PINN prices; plotting only."""
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
OUT = PACKAGE / 'outputs' / 'restored_figure5'
INPUT = PACKAGE / "primary" / "source_data"
FIG = OUT / "fig"
INPUTS = [INPUT / "terminal_distribution.csv", INPUT / "figure_curve_prices.csv"]
CURVES = ["downward", "flat", "upward"]
CURVES_ZH = ["\u4e0b\u503e", "\u5e73\u5766", "\u4e0a\u503e"]
COLORS = ["#27649A", "#C17A29", "#41826B"]
STYLES = ["-", "--", "-."]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(language, hist, prices):
    zh = language == "ZH"
    plt.rcParams["font.family"] = "Microsoft YaHei" if zh else "Arial"
    names = CURVES_ZH if zh else ["Downward", "Flat", "Upward"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.95),
                             gridspec_kw={"width_ratios": [1.12, 1]})
    # Identical saved bins, densities, styles, limits, and strike to the old panel a.
    for j, name in enumerate(CURVES_ZH):
        s = hist[hist.curve == name]
        axes[0].plot((s.left+s.right)/2, s.density, color=COLORS[j],
                     ls=STYLES[j], label=names[j], lw=1.5)
    axes[0].axvline(50, color="#777777", ls=":", lw=.9)
    axes[0].text(88, .021, r"$K=50$", color="#555555", fontsize=8)
    axes[0].set(xlim=(0, 310),
        xlabel="\u671f\u6743\u5230\u671f\u65f6\u7684 " + r"$A_1(\tau)$" if zh
               else r"First-period futures price $A_1(\tau)$",
        ylabel="\u6982\u7387\u5bc6\u5ea6" if zh else "Density")
    axes[0].legend(loc="upper right")

    x = np.arange(3)
    width = .35
    axes[1].bar(x-width/2, prices.baseline_PINN, width, color=COLORS[0],
                label="\u57fa\u51c6\u53c2\u6570" if zh else "Baseline")
    axes[1].bar(x+width/2, prices.matched_PINN, width, facecolor="#ACBBC8",
                edgecolor="#5B6D7C", hatch="///", linewidth=.45,
                label="\u5339\u914d\u521d\u59cb\u65b9\u5dee" if zh else "Matched initial variance")
    for i, row in prices.iterrows():
        for dx, value in [(-width/2, row.baseline_PINN), (width/2, row.matched_PINN)]:
            axes[1].text(i+dx, value+.3, f"{value:.2f}", ha="center", fontsize=7)
    axes[1].set(xticks=x, xticklabels=names, ylim=(0, 22.5),
                ylabel="\u770b\u6da8\u671f\u6743\u4ef7\u503c" if zh else "Call value")
    axes[1].legend(loc="upper right", fontsize=6.7)
    for j, ax in enumerate(axes):
        ax.text(-.12, 1.04, chr(97+j), transform=ax.transAxes,
                weight="bold", fontsize=11)
    assert len(axes[0].lines) == 4
    assert len(axes[1].patches) == 6 and not axes[1].collections
    assert len(axes[1].get_legend_handles_labels()[1]) == 2
    fig.tight_layout(pad=1.15, w_pad=1.7)
    name = "multiperiod_curve_distribution_restored" + ("_ZH" if zh else "")
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    input_hashes = {path.name: sha256(path) for path in INPUTS}
    hist = pd.read_csv(INPUTS[0])
    prices = pd.read_csv(INPUTS[1], usecols=["curve", "baseline_PINN", "matched_PINN"])
    prices = prices.set_index("curve").loc[CURVES].reset_index()
    assert len(hist) == 372 and hist.groupby("curve").size().to_dict() == {
        name: 124 for name in CURVES_ZH}
    assert len(prices) == 3 and np.isfinite(prices.iloc[:, 1:].to_numpy()).all()
    FIG.mkdir(parents=True, exist_ok=True)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
        "legend.frameon": False, "legend.fontsize": 7.5, "pdf.fonttype": 42,
        "svg.fonttype": "none", "axes.unicode_minus": False,
        "mathtext.fontset": "dejavusans", "savefig.facecolor": "white"})
    for language in ["EN", "ZH"]:
        draw(language, hist, prices)
    assert input_hashes == {path.name: sha256(path) for path in INPUTS}
    outputs = sorted(path.name for path in FIG.iterdir()
                     if path.name.startswith("multiperiod_curve_distribution_restored"))
    assert len(outputs) == 6
    manifest = {
        "figure_count": 1, "panels": ["a: saved Gaussian-Sobol distribution", "b: saved PINN prices"],
        "languages": ["EN", "ZH"], "formats": ["pdf", "svg", "png"],
        "new_training": False, "new_inference": False, "new_integration": False,
        "histogram_rows": len(hist), "bins_per_curve": 124,
        "histogram_modified": False, "strike": 50,
        "price_columns": ["baseline_PINN", "matched_PINN"],
        "price_reference_markers": False,
        "saved_PINN_prices": prices.to_dict("records"),
        "source_directory": str(INPUT), "source_sha256": input_hashes,
        "source_hashes_unchanged": True, "outputs": outputs,
    }
    (OUT / "restored_figure5_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
