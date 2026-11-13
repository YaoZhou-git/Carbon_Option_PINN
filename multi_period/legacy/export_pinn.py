"""Export fixed-maturity PINN slices and fresh independent price diagnostics."""
import sys
from pathlib import Path
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'core'))
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from factor_model import Model,factors,value
from factor_pinn import FactorPINN
from run_experiments import COLORS,STYLES,save,label

ROOT=PACKAGE/'outputs'/'legacy'
ROOT.mkdir(parents=True,exist_ok=True)


def main():
    (ROOT/"pinn").mkdir(parents=True,exist_ok=True)
    cfg=Model();rows=[];diagnostics=[];references=[];nets={}
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
    for n in range(1,5):
        checkpoint=torch.load(PACKAGE/'models'/'baseline'/f'N{n}_factor_pinn.pt',map_location='cpu',weights_only=False)
        assert checkpoint['parameters']==cfg.__dict__,'Checkpoint parameters differ from the current configuration; retrain rather than rescaling old prices.'
        net=FactorPINN(cfg,n);net.load_state_dict(checkpoint['state_dict']);net.eval();nets[n]=net
        grid=np.arange(40.,96.,2.5)
        states=np.array([np.r_[factors([a1,45.,30.,20.][:n],cfg)/cfg.pi,0] for a1 in grid])
        with torch.no_grad():cs=net(torch.tensor(states,dtype=torch.float32)).numpy().ravel()*cfg.pi
        axs[0].plot(grid,cs,color=COLORS[n-1],ls=STYLES[n-1],label=f'N = {n}')
        for a1,c in zip(grid,cs):
            lower=np.exp(-.15)*max(a1-50,0);upper=np.exp(-.15)*a1
            rows.append({'N':n,'A1':a1,'PINN_C':float(c),'lower_bound':lower,'upper_bound':upper})
            assert lower-1e-4<=c<=upper+1e-4,('Displayed PINN bound violation',n,a1,c,lower,upper)
        for a1 in (40.,50.,60.,70.,80.,90.):
            c=float(cs[int(np.flatnonzero(grid==a1)[0])])
            q=value([a1,45.,30.,20.][:n],cfg,order=80,inner=256)
            diagnostics.append({'N':n,'A1':a1,'PINN_C':c,'Gaussian_C':q,'difference':c-q,
                                'lower_bound_shortfall':max(np.exp(-.15)*max(a1-50,0)-c,0)})
            if a1==60:references.append({'N':n,'PINN_C':c,'Gaussian_C':q,'absolute_difference':abs(c-q)})
    for name,rr in [('fig6_factor_pinn_slice',rows),('pinn_independent_diagnostics',diagnostics),('table4_pinn_reference',references)]:
        with (ROOT/'source_data'/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=rr[0]);w.writeheader();w.writerows(rr)
    for i,n in enumerate((2,3,4)):
        vals=np.array([x['PINN_C'] for x in rows if x['N']==n])
        previous=np.array([x['PINN_C'] for x in rows if x['N']==n-1])
        axs[1].plot(grid,vals-previous,color=COLORS[n-1],ls=STYLES[n-1],label=f'$C_{n}-C_{n-1}$')
    axs[0].set(xlabel='Current futures price, $A_1$',ylabel='PINN call value, $C_N$')
    axs[1].set(xlabel='Current futures price, $A_1$',ylabel='Incremental call value')
    axs[1].axhline(0,color='#777777',lw=.6)
    for j,ax in enumerate(axs):ax.legend();label(ax,chr(97+j))
    save('fig6_factor_pinn_slice',fig)
    tex=[r'\begin{tabular}{rrr}',r'\toprule',r'$N$ & PINN price $C_N$ & $C_N-C_{N-1}$ \\',r'\midrule']
    for i,x in enumerate(references):
        increment='--' if i==0 else f'{x["PINN_C"]-references[i-1]["PINN_C"]:.3f}'
        tex.append(f'{x["N"]} & {x["PINN_C"]:.3f} & {increment}'+r' \\')
    tex.extend([r'\bottomrule',r'\end{tabular}'])
    (ROOT/'tables'/'table4_pinn_reference.tex').write_text('\n'.join(tex),encoding='utf-8')
    summary={'displayed_N':list(nets),'architecture':'3 hidden Tanh layers, width 64; smooth trial function plus learned PDE correction',
      'fixed_maturity':3,'time_domain':[0,3],'factors_domain':[0,100],
      'terminal_condition':'hard imposed to float precision','MC_used':False,
      'max_displayed_bound_shortfall':max(max(x['lower_bound']-x['PINN_C'],x['PINN_C']-x['upper_bound'],0) for x in rows),
      'max_independent_slice_absolute_difference':max(abs(x['difference']) for x in diagnostics),
      'reference_prices':references,'scope':'One fixed-maturity theoretical profile; no high-dimensional superiority claim.'}
    (ROOT/'pinn'/'final_diagnostics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    old=ROOT/'pinn'/'price_slice.csv';backup=ROOT/'pinn'/'training_price_slice.csv'
    if old.exists() and not backup.exists():shutil.copy2(old,backup)
    # The training-time diagnostic used a lower quadrature order. Preserve it
    # separately and make the canonical final slice use the fresh refinement.
    with old.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=diagnostics[0]);w.writeheader();w.writerows(diagnostics)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
