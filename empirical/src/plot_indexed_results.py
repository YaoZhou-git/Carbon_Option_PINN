"""Indexed-penalty empirical panels from the saved, unmodified result tables."""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager


from release_paths import results_dir, figure_language, figure_font
HERE = results_dir()/'figures'/'indexed'
SOURCE = results_dir()/'pricing'
FIG = HERE
BLUE, ORANGE = '#286DA8', '#D58A32'

CONTRACT = {
    'conclusion': 'The indexed calibration gives distinct period risk clocks and an increasing call-price term structure at the stated valuation state.',
    'panel_a': 'Two calibrated risk-clock intensities on all 256 observation dates; these are z_j rather than price diffusion coefficients.',
    'panel_b': 'One indexed-separate call-price curve over all 15 saved horizons; no other model is included.',
    'archetype': 'quantitative grid',
    'backend': 'Python matplotlib',
    'size_inches': [183/25.4, 3.35],
    'exports': ['editable-text PDF', 'editable-text SVG', '300-dpi PNG'],
    'uncertainty': 'Deterministic calibrated outputs; no confidence interval is plotted.',
}


def read_rows(name):
    path = SOURCE/name
    with path.open(encoding='utf-8', newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row['model'] == 'indexed_separate']
    return path, rows


def draw(risk, prices, zh=False):
    font = figure_font(zh)
    font_manager.findfont(font, fallback_to_default=True)
    plt.rcParams.update({
        'font.family': font,
        'font.size': 8.5,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.5,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.linewidth': .7,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'mathtext.fontset': 'dejavusans',
        'axes.unicode_minus': False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(183/25.4, 3.35))
    fig.subplots_adjust(left=.085, right=.985, bottom=.20, top=.84, wspace=.32)
    dates = [datetime.fromisoformat(row['date']) for row in risk]
    ref = [i for i, d in enumerate(dates) if d.date().isoformat() == '2021-06-30'][0]
    for name, color, label in [('z1', BLUE, r'$z_1(t)$'), ('z2', ORANGE, r'$z_2(t)$')]:
        values = [float(row[name]) for row in risk]
        axes[0].plot(dates, values, color=color, label=label, lw=1.65)
        axes[0].scatter([dates[ref]], [values[ref]], color=color, s=22, zorder=4)
    axes[0].axvline(dates[ref], color='#9A9A9A', lw=.65, ls=(0, (3, 3)), zorder=0)
    axes[0].set_xlim(datetime(2021, 1, 1), datetime(2021, 12, 31))
    ticks = [datetime(2021, month, 1) for month in [1, 4, 7, 10]]+[datetime(2021, 12, 31)]
    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels(['1月', '4月', '7月', '10月', '12月'] if zh else ['Jan', 'Apr', 'Jul', 'Oct', 'Dec'])
    axes[0].set_ylim(0, .66)
    axes[0].set_xlabel('观测日期（2021年）' if zh else 'Observation date (2021)')
    axes[0].set_ylabel('风险时钟强度（年$^{-1}$）' if zh else r'Risk-clock intensity (year$^{-1}$)')
    axes[0].set_title('分期风险时钟' if zh else 'Period-specific risk clocks', loc='left', pad=13)
    axes[0].legend(loc='upper left', frameon=False)

    horizons = np.array([float(row['remaining_years']) for row in prices])
    calls = np.array([float(row['call_price']) for row in prices])
    axes[1].plot(horizons, calls, color=BLUE, lw=1.8)
    for horizon in [.25, .5, 1.]:
        index = np.where(np.isclose(horizons, horizon))[0][0]
        value = calls[index]
        axes[1].scatter([horizon], [value], color=ORANGE, s=23, zorder=4)
        offset = (7, -13) if horizon < 1 else (-5, 8)
        axes[1].annotate(f'{value:.3f}', (horizon, value), xytext=offset,
                         textcoords='offset points', ha='left' if horizon < 1 else 'right',
                         va='center', fontsize=8.5, color='#333333')
    axes[1].set_xlim(0, 1.25)
    axes[1].set_ylim(0, 20)
    axes[1].set_xticks([0, .25, .5, .75, 1., 1.25])
    axes[1].set_yticks([0, 5, 10, 15, 20])
    axes[1].set_xlabel('期权剩余期限（年）' if zh else 'Remaining option maturity (years)')
    axes[1].set_ylabel('看涨期权价格（欧元/吨）' if zh else 'Call price (EUR/t)')
    axes[1].set_title('校准后的期权期限结构' if zh else 'Calibrated call-price term structure', loc='left', pad=13)
    for panel, ax in zip(['a', 'b'], axes):
        ax.text(-.20, 1.095, panel, transform=ax.transAxes, weight='bold', fontsize=10, va='bottom')
        ax.tick_params(direction='out', length=3, width=.65)
    fig.text(.535, .055,
             r'2021-06-30；$K=56.78$；$r=3\%$' if zh else r'30 Jun 2021; $K=56.78$; $r=3\%$',
             fontsize=8, color='#4A4A4A')
    suffix = '_ZH' if zh else ''
    stem = FIG/f'calibrated_risk_pricing_indexed_20261109{suffix}'
    paths = []
    for extension in ['pdf', 'svg', 'png']:
        path = stem.with_suffix('.'+extension)
        fig.savefig(path, dpi=300, facecolor='white')
        paths.append(path)
    plt.close(fig)
    return paths


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    risk_path, risk = read_rows('calibrated_risk_contributions.csv')
    price_path, prices = read_rows('calibrated_option_prices.csv')
    assert len(risk) == 256 and len(prices) == 15
    assert len({row['date'] for row in risk}) == 256
    assert risk[0]['date'] == '2021-01-04' and risk[-1]['date'] == '2021-12-30'
    assert all(float(b['remaining_years']) > float(a['remaining_years']) for a, b in zip(prices, prices[1:]))
    assert all(float(row['strike']) == 56.78 for row in prices)
    point_checks = {}
    for horizon, rounded in [(.25, 5.939), (.5, 9.042), (1., 15.255)]:
        row = next(row for row in prices if abs(float(row['remaining_years'])-horizon) < 1e-10)
        assert round(float(row['call_price']), 3) == rounded
        point_checks[str(horizon)] = float(row['call_price'])
    language=figure_language()
    paths=[]
    if language in ('en','both'):paths+=draw(risk,prices)
    if language in ('zh','both'):paths+=draw(risk,prices,zh=True)
    svg_checks = {}
    for path in paths:
        if path.suffix == '.svg':
            raw = path.read_text(encoding='utf-8')
            count = len(re.findall(r'<text\b', raw))
            assert count >= 20
            svg_checks[path.name] = {'text_element_count': count, 'editable_text': True}
    report = {'figure_contract': CONTRACT, 'n_observation_dates': len(risk), 'n_maturity_points': len(prices),
              'model': 'indexed_separate', 'reference_date': '2021-06-30',
              'reference_futures_prices': [56.78, 57.49], 'call_checks': point_checks,
              'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [risk_path, price_path]},
              'svg_checks': svg_checks, 'outputs': [str(path) for path in paths],
              'visual_qa': 'Pending inspection of both PNG previews.'}
    (HERE/'figure_indexed_verification.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
