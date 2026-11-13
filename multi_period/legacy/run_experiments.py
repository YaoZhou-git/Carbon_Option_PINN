"""Reproducible, data-free economic comparative statics; Python backend."""
import sys
from pathlib import Path
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'core'))
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import csv
import json
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from factor_model import Model,value,boundary_probability,factors

ROOT=PACKAGE/'outputs'/'legacy'
ROOT.mkdir(parents=True,exist_ok=True)
FIG=ROOT/'figures';DATA=ROOT/'source_data';TABLE=ROOT/'tables'
for p in (FIG,DATA,TABLE):p.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':9,
 'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.7,
 'xtick.direction':'out','ytick.direction':'out','savefig.facecolor':'white'})
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']
STYLES=['-','--','-.',':']


def write(name,rows):
    with (DATA/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


def save(name,fig):
    fig.tight_layout(pad=1.3)
    for ext in ('pdf','svg','png','tiff'):
        kwargs={'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}
        fig.savefig(FIG/f'{name}.{ext}',dpi=600 if ext=='tiff' else 240,**kwargs)
    plt.close(fig)


def label(ax,text):ax.text(-.14,1.04,text,transform=ax.transAxes,fontweight='bold',fontsize=10)


def main():
    cfg=Model();base=[60.,45.,30.,20.]
    (ROOT/'parameters.json').write_text(json.dumps(asdict(cfg),indent=2),encoding='utf-8')
    curves={'Downward':[60.,45.,30.,20.],'Flat':[60.]*4,'Upward':[60.,65.,70.,75.]}
    rows=[];grid=np.arange(40.,96.,2.5)
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
    for n in range(1,5):
        cs=[]
        for a1 in grid:
            a=[a1,*base[1:]][:n];c=value(a,cfg);cs.append(c)
            rows.append({'A1':a1,'A2':45,'A3':30,'A4':20,'N':n,'t':0,'tau':3,'C':c,
                         'discounted_intrinsic':np.exp(-.15)*max(a1-50,0)})
        axs[0].plot(grid,cs,color=COLORS[n-1],ls=STYLES[n-1],label=f'N = {n}')
    axs[0].set(xlabel='Current futures price, $A_1$',ylabel='First-period call value, $C_N$')
    axs[0].legend(loc='upper left');label(axs[0],'a')
    table=[]
    for j,(name,a) in enumerate(curves.items()):
        cs=[value(a[:n],cfg,order=80,inner=256) for n in range(1,5)]
        axs[1].plot(range(1,5),cs,marker=['o','s','^'][j],ms=4,color=COLORS[j],ls=STYLES[j],label=name)
        table.append({'curve':name,'A1':a[0],'A2':a[1],'A3':a[2],'A4':a[3],
                      **{f'C{n}':cs[n-1] for n in range(1,5)},
                      **{f'Delta{n}':cs[n-1]-cs[n-2] for n in range(2,5)}})
    axs[1].set(xlabel='Number of compliance periods, $N$',ylabel='First-period call value',xticks=[1,2,3,4])
    axs[1].legend(loc='best');label(axs[1],'b')
    write('fig1_period_comparison',rows);write('table1_forward_curves',table)
    save('fig1_period_comparison',fig)
    tex=['% Requires booktabs. Model-implied values, not observed option prices.',
     r'\begin{tabular}{lrrrrrrrr}',r'\toprule',
     r'Forward curve & $A_2$ & $A_3$ & $A_4$ & $C_1$ & $C_2$ & $C_3$ & $C_4$ & $C_4-C_1$ \\',r'\midrule']
    for x in table:
        vals=[x[k] for k in ('A2','A3','A4','C1','C2','C3','C4')]+[x['C4']-x['C1']]
        tex.append(x['curve']+' & '+' & '.join(f'{v:.3f}' for v in vals)+r' \\')
    tex.extend([r'\bottomrule',r'\end{tabular}'])
    (TABLE/'table1_forward_curves.tex').write_text('\n'.join(tex),encoding='utf-8')
    print('Figure 1 complete',table,flush=True)

    rows=[];fig,axs=plt.subplots(1,3,figsize=(7.2,2.55))
    for j in range(1,4):
        low=cfg.kappa[j]*base[j+1]+1 if j<3 else 2.
        high=min(base[j-1]/cfg.kappa[j-1]-1,99.)
        xs=np.linspace(low,high,21)
        for n in range(j+1,5):
            cs=[]
            for x in xs:
                a=base.copy();a[j]=x;c=value(a[:n],cfg);cs.append(c)
                rows.append({'varied_period':j+1,'varied_price':x,'N':n,'C':c})
            axs[j-1].plot(xs,cs,color=COLORS[n-1],ls=STYLES[n-1],label=f'N = {n}')
        axs[j-1].set(xlabel=f'Future price, $A_{j+1}$',ylabel='Call value' if j==1 else '')
        axs[j-1].legend(fontsize=7);label(axs[j-1],chr(96+j))
    write('fig2_future_price_effects',rows);save('fig2_future_price_effects',fig)
    print('Figure 2 complete',flush=True)

    rows=[];fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
    taus=np.linspace(.1,3.9,31)
    for j,a1 in enumerate((45.,60.,90.)):
        cs=[value([a1,*base[1:]],cfg,tau=s) for s in taus]
        us=np.array(cs)*np.exp(cfg.r*taus)
        axs[0].plot(taus,cs,color=COLORS[j],ls=STYLES[j],label=f'$A_1={a1:g}$')
        axs[1].plot(taus,us,color=COLORS[j],ls=STYLES[j],label=f'$A_1={a1:g}$')
        for tau,c,u in zip(taus,cs,us):rows.append({'A1':a1,'N':4,'tau':tau,'C':c,'undiscounted_payoff':u})
    axs[0].set(xlabel='Option maturity, $\\tau$',ylabel='Discounted call value')
    axs[1].set(xlabel='Option maturity, $\\tau$',ylabel='Expected payoff, $e^{r\\tau}C$')
    for j,ax in enumerate(axs):ax.legend();label(ax,chr(97+j))
    write('fig3_maturity_discounting',rows);save('fig3_maturity_discounting',fig)
    print('Figure 3 complete',flush=True)

    rows=[];fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
    sub=replace(cfg,alpha=(.4779,)*4,beta=(.8*4**(.4779-1.2),)*4)
    taus=np.r_[np.linspace(.1,3.8,24),3.9,3.99,3.999,3.9999]
    for j,(name,p) in enumerate((('Binary-admissible',cfg),('Subcritical (matched initial variance)',sub))):
        probs=[];cs=[]
        for tau in taus:
            prob=boundary_probability([60.],p,tau)[0];c=value([60.],p,tau=tau)
            probs.append(prob);cs.append(c)
            rows.append({'regime':name,'alpha':p.alpha[0],'beta':p.beta[0],'tau':tau,'near_boundary_probability':prob,'C':c})
        axs[0].plot(taus,probs,color=COLORS[j],ls=STYLES[j],label=name)
        axs[1].plot(taus,cs,color=COLORS[j],ls=STYLES[j],label=name)
    axs[0].set(xlabel='Option maturity, $\\tau$',ylabel='$P(A_1/\\pi<0.01\\ \mathrm{or}>0.99)$',ylim=(0,1))
    axs[1].set(xlabel='Option maturity, $\\tau$',ylabel='Single-period call value')
    for j,ax in enumerate(axs):label(ax,chr(97+j))
    axs[1].legend(loc='best',fontsize=6.4)
    write('fig4_terminal_regimes',rows);save('fig4_terminal_regimes',fig)
    print('Figure 4 complete',flush=True)

    rows=[];fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
    for n in (2,3,4):
        rs=np.linspace(-.25,.75,21);cs=[]
        for rho in rs:
            c=value(base[:n],replace(cfg,rho=float(rho)));cs.append(c)
            rows.append({'parameter':'rho','value':rho,'N':n,'C':c})
        axs[0].plot(rs,cs,color=COLORS[n-1],ls=STYLES[n-1],label=f'N = {n}')
        # kappa is the manuscript's deterministic inter-period discount factor,
        # not an unrestricted measure of banking flexibility. This sensitivity
        # varies future-period discount rates, keeping rates before tau fixed.
        ks=np.linspace(.65,.95,21);cs=[]
        for k in ks:
            c=value(base[:n],replace(cfg,kappa=(float(k),)*3));cs.append(c)
            rows.append({'parameter':'kappa','value':k,'N':n,'C':c})
        axs[1].plot(ks,cs,color=COLORS[n-1],ls=STYLES[n-1],label=f'N = {n}')
    axs[0].set(xlabel='Cross-factor correlation, $\\rho$',ylabel='First-period call value')
    axs[1].set(xlabel='Inter-period discount factor, $\\kappa$',ylabel='First-period call value')
    for j,ax in enumerate(axs):ax.legend();label(ax,chr(97+j))
    write('fig5_dependence_transfer',rows);save('fig5_dependence_transfer',fig)
    print('Figure 5 complete',flush=True)

    # Order refinement and economic restrictions on displayed states.
    checks=[]
    for name,a in curves.items():
        for n in range(1,5):
            c24=value(a[:n],cfg,order=24,inner=96)
            c44=value(a[:n],cfg,order=44,inner=160)
            checks.append({'curve':name,'N':n,'C_order24':c24,'C_order44':c44,'absolute_change':abs(c44-c24),
                           'lower_bound':np.exp(-.15)*10,'upper_bound':np.exp(-.15)*60})
    write('quadrature_refinement',checks)
    legacy=replace(cfg,T=(4.,)*4,kappa=(.5,)*3)
    write('legacy_corrected_domain',[{'N':n,'C':value(base[:n],legacy,order=44)} for n in range(1,5)])
    (ROOT/'numerical_summary.json').write_text(json.dumps({'forward_curves':table,
     'max_reference_refinement':max(x['absolute_change'] for x in checks),
     'future_sensitivity_note':'All non-varied futures, t, tau, K and coefficients are held fixed.',
     'no_MC_used':True},indent=2),encoding='utf-8')
    print('All data-free figures exported',flush=True)


if __name__=='__main__':main()
