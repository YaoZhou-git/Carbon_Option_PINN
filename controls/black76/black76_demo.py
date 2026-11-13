"""Black (1976) maturity-curve crossing, with a zero-rate control.

European call, fixed underlying futures contract, cash payoff at option expiry.
No carbon mechanism, price bound, or state-dependent volatility is introduced.
"""
from pathlib import Path
import csv,json
import numpy as np
from scipy.special import ndtr
from scipy.optimize import brentq
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent
K=50.;R=.05;SIGMA=.2
TAUS=np.arange(.5,3.51,.5)
GRID=np.linspace(0.,100.,1001)

def black_call(F,tau,r=R):
    F=np.asarray(F,dtype=float)
    safe=np.maximum(F,np.finfo(float).tiny)
    d1=(np.log(safe/K)+.5*SIGMA**2*tau)/(SIGMA*np.sqrt(tau))
    d2=d1-SIGMA*np.sqrt(tau)
    c=np.exp(-r*tau)*(F*ndtr(d1)-K*ndtr(d2))
    return np.where(F>0,c,0.)

cjk_font = Path('C:/Windows/Fonts/msyh.ttc')
if cjk_font.exists():
    font_manager.fontManager.addfont(str(cjk_font))
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'font.size':12,
    'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,
    'pdf.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'dejavusans'})
colors=plt.colormaps['viridis'](np.linspace(.03,.92,len(TAUS)))
root=brentq(lambda F:float(black_call(F,3.5)-black_call(F,.5)),50.,100.,xtol=1e-12)
cross=float(black_call(root,.5))

fig,ax=plt.subplots(figsize=(9.2,6.1))
for tau,color in zip(TAUS,colors):
    ax.plot(GRID,black_call(GRID,tau),lw=2.2,color=color,label=rf'$\tau={tau:.1f}$ 年')
ax.axvline(K,color='#BD5353',ls='--',lw=1.1)
ax.text(K+1,1.1,'K=50',color='#BD5353',fontsize=11)
ax.plot(root,cross,'o',ms=6,mfc='white',mec='#333',mew=1.2,zorder=5)
ax.annotate(f'τ=0.5 与 3.5 的交叉\nF≈{root:.2f}',xy=(root,cross),xytext=(39,34),
    arrowprops={'arrowstyle':'->','color':'#444','lw':1},fontsize=11,
    bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
ax.annotate('低价区：长期曲线在上',xy=(40,float(black_call(40,3.5))),xytext=(10,15),
    arrowprops={'arrowstyle':'->','color':'#555'},fontsize=11)
ax.annotate('高价区：短期曲线在上',xy=(94,float(black_call(94,.5))),xytext=(64,46),
    arrowprops={'arrowstyle':'->','color':'#555'},fontsize=11)
ax.set(xlabel='当前期货价格 F',ylabel='t=0 的欧式看涨期权价格 C',xlim=(0,100),ylim=(0,51))
fig.suptitle('Black（1976）：不同到期曲线的排序反转',y=.98,fontsize=17)
fig.text(.5,.918,'K=50，r=5%，σ=20%；同一期货合约，改变期权到期时间',
    ha='center',va='center',fontsize=10,color='#555')
ax.grid(alpha=.16);ax.legend(loc='upper left',fontsize=10)
fig.subplots_adjust(left=.10,right=.98,bottom=.12,top=.86)
for ext in ['png','svg','pdf']:fig.savefig(ROOT/f'black76_maturity_crossing.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,r in zip(axes,[.05,0.]):
    for tau,color in zip(TAUS,colors):ax.plot(GRID,black_call(GRID,tau,r),color=color,lw=1.7,label=f'τ={tau:g}')
    ax.set(xlabel='当前期货价格 F',ylabel='看涨期权价格 C',title=f'r={r:.0%}',xlim=(0,100),ylim=(0,52))
    ax.grid(alpha=.16)
axes[0].legend(fontsize=8,loc='upper left')
axes[1].text(.04,.92,'去除贴现后，长期曲线不低于短期',transform=axes[1].transAxes,fontsize=10)
fig.suptitle('控制实验：相同的期货动态，仅改变现金流贴现率',fontsize=13)
fig.tight_layout(pad=1.5)
for ext in ['png','svg','pdf']:fig.savefig(ROOT/f'black76_zero_rate_control.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)

with (ROOT/'curve_data.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['F','tau','r','sigma','K','call'])
    for r in [.05,0.]:
        for tau in TAUS:
            for F,C in zip(GRID,black_call(GRID,tau,r)):w.writerow([F,tau,r,SIGMA,K,C])
table=[{'F':F,'short_tau_0.5':float(black_call(F,.5)),
    'long_tau_3.5':float(black_call(F,3.5)),
    'long_minus_short':float(black_call(F,3.5)-black_call(F,.5))} for F in [60.,90.]]
numerical=[]
for F in [20.,60.,90.]:
    for tau in [.5,3.5]:
        def integrand(z):
            Ft=F*np.exp(-.5*SIGMA**2*tau+SIGMA*np.sqrt(tau)*z)
            return max(Ft-K,0)*np.exp(-z*z/2)/np.sqrt(2*np.pi)
        cutoff=(np.log(K/F)+.5*SIGMA**2*tau)/(SIGMA*np.sqrt(tau))
        independent=np.exp(-R*tau)*quad(integrand,max(cutoff,-12),12,epsabs=1e-11,epsrel=1e-11)[0]
        numerical.append(abs(independent-float(black_call(F,tau))))
values=np.array([black_call(GRID,tau) for tau in TAUS])
zero=np.array([black_call(GRID,tau,0.) for tau in TAUS])
assert np.diff(values,axis=1).min()>-1e-12
assert np.diff(zero,axis=0).min()>-1e-12
assert max(numerical)<1e-8
assert abs(float(black_call(root,3.5)-black_call(root,.5)))<1e-10
assert table[0]['long_minus_short']>0 and table[1]['long_minus_short']<0
assert np.all(values>=np.exp(-R*TAUS[:,None])*np.maximum(GRID-K,0)-1e-10)
assert np.all(values<=np.exp(-R*TAUS[:,None])*GRID+1e-10)
out={'K':K,'r':R,'sigma':SIGMA,'maturities_years':TAUS.tolist(),
    'comparison':'same fixed futures contract; option payoff settled at option expiry',
    'crossing_short_long_F':root,'crossing_call':cross,'example_table':table,
    'independent_gaussian_integration_max_error':max(numerical),
    'all_fixed_maturity_curves_increasing':True,'zero_rate_maturity_order_verified':True,
    'bounds_verified':True,'status':'passed'}
(ROOT/'verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
