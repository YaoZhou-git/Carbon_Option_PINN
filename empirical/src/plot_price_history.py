"""Daily EUA futures closes used by the existing 2021 two-period estimation.

Figure contract: give the reader the observed price paths underlying the inverse
problem. One descriptive line panel, preserving all 256 common-date observations.
Blue/orange identify contracts; a dash pattern also distinguishes overlapping
curves. No smoothing, interpolation, monthly averaging or factor-sign filtering.
Exports: 183-mm vector PDF/SVG, 600-dpi TIFF, and 300-dpi PNG preview.
"""
from pathlib import Path
from datetime import date
import sys,json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

from release_paths import results_dir, PACKAGE_ROOT
OUT=results_dir()/'figures'/'price_history'
PILOT=results_dir()/'estimation'
from run_pilot import read_source


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    days,prices,info=read_source()
    expected_path=PILOT/'sample_provenance.json'
    if not expected_path.exists():expected_path=PACKAGE_ROOT/'reference'/'source_metadata.json'
    expected=json.loads(expected_path.read_text(encoding='utf-8'))
    assert len(days)==256 and days[0]==date(2021,1,4) and days[-1]==date(2021,12,30)
    assert np.all(prices>0) and all(a<b for a,b in zip(days,days[1:]))
    assert {k:v['sha256'] for k,v in info['files'].items()}=={k:v['sha256'] for k,v in expected['files'].items()}
    for rec in info['files'].values():assert hashlib.sha256(Path(rec['file']).read_bytes()).hexdigest()==rec['sha256']
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':9,'axes.labelsize':10,'xtick.labelsize':8.5,'ytick.labelsize':9,'legend.fontsize':9,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
    width=183/25.4
    fig,ax=plt.subplots(figsize=(width,3.35))
    ax.plot(days,prices[:,0],color='#0072BD',lw=1.35,label=r'EUA Dec 2022 ($A_1$)',zorder=2)
    ax.plot(days,prices[:,1],color='#D95319',lw=1.25,ls=(0,(4,2)),label=r'EUA Dec 2023 ($A_2$)',zorder=3)
    ax.set_xlim(date(2021,1,1),date(2021,12,31));ax.set_ylim(29,94)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
    ax.tick_params(axis='both',which='major',width=.6,length=3.5)
    for label in ax.get_xticklabels():label.set_rotation(45);label.set_ha('right')
    ax.set_xlabel('Observation Date',labelpad=6)
    ax.set_ylabel('Closing Price (EUR/tonne)',labelpad=6)
    ax.legend(loc='lower right',frameon=True,fancybox=False,edgecolor='#BBBBBB',facecolor='white',framealpha=.95)
    fig.subplots_adjust(left=.09,right=.985,bottom=.235,top=.965)
    stem='futures_prices_dec22_dec23_2021'
    for ext,dpi in [('png',300),('pdf',300),('svg',300),('tiff',600)]:
        fig.savefig(OUT/f'{stem}.{ext}',dpi=dpi)
    assert len(ax.lines)==2
    for j in range(2):assert np.array_equal(np.asarray(ax.lines[j].get_ydata()),prices[:,j])
    plt.close(fig)
    metadata={'figure_role':'Observed price paths for the current inverse-estimation sample',
        'contracts':['EUA Dec 2022','EUA Dec 2023'],'frequency':'daily closing prices; monthly ticks',
        'n_paired_dates':len(days),'observation_start':days[0].isoformat(),'observation_end':days[-1].isoformat(),
        'smoothing':False,'interpolation':False,'monthly_aggregation':False,'factor_sign_filtering':False,
        'raw_inputs_unchanged':True,'input_hashes':info['files'],
        'plot_ydata_identical_to_estimation_input':True,
        'price_ranges':info['price_ranges'],'log_return_correlation':float(np.corrcoef(np.diff(np.log(prices),axis=0).T)[0,1]),
        'exports':{ext:{'bytes':(OUT/f'{stem}.{ext}').stat().st_size,'sha256':hashlib.sha256((OUT/f'{stem}.{ext}').read_bytes()).hexdigest()} for ext in ['png','pdf','svg','tiff']}}
    (OUT/'FIGURE_VERIFICATION.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    snippet=r'''\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\linewidth]{fig/futures_prices_dec22_dec23_2021.pdf}
\caption{\textcolor{red}{Daily closing prices of EUA futures expiring in December 2022 and December 2023, from 4 January to 30 December 2021. The sample contains 256 paired daily observations; the horizontal axis uses monthly ticks.}}
\label{fig:futures_price}
\end{figure}
'''
    (OUT/'price_figure_insertion.tex').write_text(snippet,encoding='utf-8')
    print(json.dumps({'n':len(days),'start':str(days[0]),'end':str(days[-1]),'price_ranges':info['price_ranges'],
        'data_match':True,'exports':['pdf','png','svg','tiff']},ensure_ascii=False))


if __name__=='__main__':main()
