"""Question-driven four-period experiments. Exact transitions; independent RQMC.

No historical source is changed. Seeds and all scenarios are retained. Numerical
intervals concern integration error only, not parameter or market uncertainty.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys, json, csv, time, hashlib
from pathlib import Path
from dataclasses import replace, asdict
from functools import lru_cache
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.integrate import quad
from scipy.stats import qmc, t as student_t

ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=Path(__file__).with_name('reference_factor_model.py')
from reference_factor_model import Model, value as legacy_value, covariance as legacy_covariance
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
BASE=Model(); A=np.array([60.,45.,30.,20.])
CURVES={'下倾':A, '平坦':np.array([60.]*4),'上倾':np.array([60.,65.,70.,75.])}

def write(name,rows):
    with (DATA/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(name,len(rows),flush=True)

def weights(c,n):
    return np.r_[1.,np.cumprod(c.kappa[:n-1])]

def xfactor(a,c):
    a=np.asarray(a,float);x=a.copy()
    x[:-1]-=np.asarray(c.kappa[:len(a)-1])*a[1:]
    if not np.all((x>0)&(x<c.pi)): raise ValueError(('Outside domain',a,x))
    return x

def Z(c,j,s):
    T,al,be=c.T[j],c.alpha[j],c.beta[j]
    if abs(al-1)<1e-12:return be*np.log(T/(T-s))
    return be*(T**(1-al)-(T-s)**(1-al))/(1-al)

@lru_cache(maxsize=512)
def trans(c,n):
    if not 0<c.tau<min(c.T[:n]):raise ValueError('Require tau<T1')
    if n>1 and not -1/(n-1)<c.rho<1:raise ValueError('Non-SPD correlation')
    z=np.array([Z(c,j,c.tau) for j in range(n)])
    M=np.exp(.5*z);V=np.diag(np.expm1(z))
    for i in range(n):
        for j in range(i+1,n):
            def integrand(s):
                zi=c.beta[i]*(c.T[i]-s)**(-c.alpha[i])
                zj=c.beta[j]*(c.T[j]-s)**(-c.alpha[j])
                return np.sqrt(zi*zj)*np.exp(.5*(z[i]-Z(c,i,s)+z[j]-Z(c,j,s)))
            V[i,j]=V[j,i]=c.rho*quad(integrand,0,c.tau,epsabs=1e-10)[0]
    return M,V

@lru_cache(maxsize=8)
def normals(power,reps,n):
    return np.array([ndtri(qmc.Sobol(n,scramble=True,seed=731001+101*r).random_base2(power)) for r in range(reps)])

def terminals(a,c,power=14,reps=8):
    x=xfactor(a,c);n=len(x);M,V=trans(c,n)
    y=M*ndtri(x/c.pi)+normals(power,reps,n)@np.linalg.cholesky(V).T
    return c.pi*ndtr(y)@weights(c,n)

def estimate(a,c=BASE,power=14,reps=8,details=False):
    s=terminals(a,c,power,reps)
    estimates=np.exp(-c.r*c.tau)*np.maximum(s-c.K,0).mean(axis=1)
    mean=float(estimates.mean());se=float(estimates.std(ddof=1)/np.sqrt(reps))
    out={'C':mean,'se':se,'halfwidth95':float(student_t.ppf(.975,reps-1)*se)}
    if details:
        out.update(mean_A=float(s.mean()),variance=float(s.var()),p_ITM=float((s>c.K).mean()),
            convexity_undiscounted=mean*np.exp(c.r*c.tau)-max(a[0]-c.K,0),
            TV=mean-np.exp(-c.r*c.tau)*max(a[0]-c.K,0))
    return out

def delta(a,c,j,h=.05,power=15,reps=8):
    up=np.array(a,float);down=up.copy();up[j]+=h;down[j]-=h
    su=terminals(up,c,power,reps);sd=terminals(down,c,power,reps)
    d=np.exp(-c.r*c.tau)*(np.maximum(su-c.K,0)-np.maximum(sd-c.K,0)).mean(axis=1)/(2*h)
    return float(d.mean()),float(d.std(ddof=1)/np.sqrt(reps))

def ivar(a,c):
    x=xfactor(a,c);w=weights(c,len(a));sig=c.pi*np.exp(-.5*ndtri(x/c.pi)**2)/np.sqrt(2*np.pi)
    sig*=np.sqrt(np.array(c.beta[:len(a)])*np.array(c.T[:len(a)])**(-np.array(c.alpha[:len(a)])))
    d=w*sig;R=np.full((len(a),len(a)),c.rho);np.fill_diagonal(R,1)
    allocation=d*(R@d)
    return float(allocation.sum()),allocation

def main():
    start=time.time();meta={'base':asdict(BASE),'curve_scenarios':{k:v.tolist() for k,v in CURVES.items()},
        'seed_rule':'731001+101*r; independently scrambled Sobol; common random numbers across scenarios',
        'legacy_factor_model_sha256':hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),
        'software':{'python':sys.version,'numpy':np.__version__}}
    (ROOT/'run_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    checks=[]
    for n in range(1,5):
        M,V=trans(BASE,n);m0,v0=legacy_covariance(BASE,n,0.,3.)
        checks.append({'check':f'covariance_N{n}','error':float(np.max(np.abs(V-v0)))})
        x=xfactor(A[:n],BASE)
        recovered=BASE.pi*ndtr(M*ndtri(x/BASE.pi)/np.sqrt(1+np.diag(V)))
        checks.append({'check':f'analytic_martingale_N{n}','error':float(np.max(np.abs(x-recovered)))})
    rows=[]
    for label,a in CURVES.items():
        for n in range(1,5):
            c=estimate(a[:n],power=17,reps=8,details=True)
            var,alloc=ivar(a[:n],BASE)
            rows.append({'curve':label,'N':n,**{f'A{j+1}':float(a[j]) for j in range(4)},**c,'instant_variance':var})
    write('curve_N_comparison',rows)
    # Numerical verification across methods and two resolutions, not merely saved checks.
    nums=[]
    for label,a in CURVES.items():
        e=estimate(a,power=17,reps=8);f=estimate(a,power=18,reps=8)
        q32=legacy_value(a,BASE,order=32,inner=96)
        q48=legacy_value(a,BASE,order=48,inner=128)
        nums.append({'curve':label,'QMC17':e['C'],'QMC18':f['C'],'halfwidth95_18':f['halfwidth95'],
            'quadrature32':q32,'quadrature48':q48,'quad_change':q48-q32,'QMC18_minus_quad48':f['C']-q48})
    write('independent_verification',nums)
    write('analytic_checks',checks)
    # Distribution evidence on a fixed N=4 model.
    rows=[]
    for label,a in CURVES.items():
        s=terminals(a,BASE,17,8).ravel();counts,edges=np.histogram(s,bins=np.linspace(0,310,125),density=False)
        for k,count in enumerate(counts):rows.append({'curve':label,'left':edges[k],'right':edges[k+1],
            'mass':count/len(s),'density':count/len(s)/(edges[k+1]-edges[k])})
    write('terminal_distribution',rows)
    # Curves matched on initial instantaneous variance of A1.
    target=ivar(A,BASE)[0];rows=[]
    for label,a in CURVES.items():
        orig=ivar(a,BASE)[0];scale=target/orig
        c=replace(BASE,beta=tuple(b*scale for b in BASE.beta))
        rows.append({'curve':label,'variance_original':orig,'variance_matched':ivar(a,c)[0],
            'beta_multiplier':scale,'C_original':estimate(a,power=17)['C'],
            'C_matched':estimate(a,c,power=17)['C'],'matched_halfwidth95':estimate(a,c,power=17)['halfwidth95']})
    write('matched_initial_variance',rows)
    # Single quotation scans with all other quotes held fixed in the valid domain.
    rows=[]
    for j in range(1,4):
        low=BASE.kappa[j]*A[j+1]+1 if j<3 else 1.
        high=min((A[j-1]-1)/BASE.kappa[j-1],99.)
        for p in np.linspace(low,high,31):
            a=A.copy();a[j]=p;c=estimate(a,power=14)
            rows.append({'period':j+1,'quote':p,**c,'X_left':xfactor(a,BASE)[j-1],'X_right':xfactor(a,BASE)[j]})
    write('future_quote_scans',rows)
    # Chain rule attribution: C_Aj=C_Xj-kappa(j-1) C_X(j-1).
    rows=[];x=xfactor(A,BASE);w=weights(BASE,4)
    def a_from_x(v):
        a=v.copy()
        for j in range(2,-1,-1):a[j]+=BASE.kappa[j]*a[j+1]
        return a
    dx=[]
    for j in range(4):
        u=x.copy();v=x.copy();u[j]+=.05;v[j]-=.05
        aup=a_from_x(u);adown=a_from_x(v)
        su=terminals(aup,BASE,17,8);sd=terminals(adown,BASE,17,8)
        d=np.exp(-.15)*(np.maximum(su-50,0)-np.maximum(sd-50,0)).mean(axis=1)/.1
        dx.append(float(d.mean()))
    var,alloc=ivar(A,BASE)
    for j in range(4):
        da,se=delta(A,BASE,j,power=17);fine,_=delta(A,BASE,j,h=.025,power=17)
        offset=-BASE.kappa[j-1]*dx[j-1] if j else 0.
        rows.append({'period':j+1,'X':x[j],'weight':w[j],'variance_share':alloc[j]/var,
            'delta_A':da,'delta_se':se,'delta_halfwidth95':2.364624*se,'delta_A_h025':fine,
            'delta_X':dx[j],'adjacent_offset':offset,'chain_delta':dx[j]+offset,
            'chain_residual':da-dx[j]-offset})
    write('risk_decomposition',rows)
    # State and dependence interactions, including counterexamples, with all points retained.
    rows=[]
    for label,a in CURVES.items():
        for rho in [-.25,0.,.3,.7]:
            for K in [30.,50.,70.]:
                c=replace(BASE,rho=rho,K=K);p=estimate(a,c,power=15)
                r={'curve':label,'rho':rho,'K':K,**p}
                for j in range(1,4):
                    d,se=delta(a,c,j,power=15);r[f'delta_A{j+1}']=d;r[f'se_A{j+1}']=se
                rows.append(r)
    write('state_dependence_grid',rows)
    rows=[]
    for rho in np.linspace(-.3,.8,23):
        c=replace(BASE,rho=float(rho))
        for label,a in CURVES.items():rows.append({'curve':label,'rho':rho,**estimate(a,c,power=14)})
    write('correlation_curves',rows)
    # Controlled parameter perturbations with alpha/T normalized to identical z_j(0).
    rows=[];baseprice=estimate(A,power=17)['C'];z0=np.array(BASE.beta)*np.array(BASE.T)**(-np.array(BASE.alpha))
    for j in range(4):
        for mult in [.5,1.,1.5,2.]:
            b=list(BASE.beta);b[j]*=mult;c=replace(BASE,beta=tuple(b));p=estimate(A,c,power=15)
            rows.append({'family':'beta','period':j+1,'setting':mult,'description':'relative beta',**p,'difference':p['C']-baseprice})
        for al in [.6,1.,1.2,1.6,2.]:
            av=list(BASE.alpha);av[j]=al;b=list(BASE.beta);b[j]=z0[j]*BASE.T[j]**al
            c=replace(BASE,alpha=tuple(av),beta=tuple(b));p=estimate(A,c,power=15)
            rows.append({'family':'alpha_matched_z0','period':j+1,'setting':al,'description':'initial factor variance held fixed',**p,'difference':p['C']-baseprice})
    for gap in [2.,3.,4.,6.,8.]:
        T=(4.,4.+gap,4.+2*gap,4.+3*gap)
        b=tuple(z0[j]*T[j]**BASE.alpha[j] for j in range(4))
        # Explicit diagnostic: fixed weights isolates diffusion horizon, not a new discount policy.
        c=replace(BASE,T=T,beta=b);p=estimate(A,c,power=15)
        rows.append({'family':'T_gap_matched_z0_fixed_weights','period':0,'setting':gap,
            'description':'diffusion-clock diagnostic; kappa fixed',**p,'difference':p['C']-baseprice})
    write('parameter_perturbations',rows)
    # Common rho vs beta_future interaction.
    rows=[]
    for rho in [-.25,0.,.3,.7]:
        for mult in [.5,1.,2.]:
            c=replace(BASE,rho=rho,beta=(.8,)+tuple(.8*mult for _ in range(3)))
            rows.append({'rho':rho,'future_beta_multiplier':mult,**estimate(A,c,power=16)})
    write('parameter_interactions',rows)
    # Maturity diagnostics: cashflow r changes only; kappa and dynamics kept fixed.
    rows=[]
    for a1 in [45.,60.,90.]:
        a=np.array([a1,45,30,20.])
        for tau in np.linspace(.1,3.9,39):
            c=replace(BASE,tau=float(tau));p=estimate(a,c,power=15)
            U=p['C']*np.exp(c.r*tau);TV=p['C']-np.exp(-c.r*tau)*max(a1-50,0)
            rows.append({'A1':a1,'tau':tau,**p,'U':U,'TV':TV,
                'undiscounted_convexity':U-max(a1-50,0),'C_r0_same_dynamics':U})
    write('maturity_decomposition',rows)
    # Binary and subcritical clocks matched at t=0: independent single-factor high-price diagnostic.
    rows=[]
    for alpha in [.4779,1.,1.2,1.6]:
        beta=.8*4**(alpha-1.2)
        for tau in [.1,.5,1.,1.5,2.,2.5,3.,3.5,3.8,3.95,3.99]:
            c=replace(BASE,alpha=(alpha,)*4,beta=(beta,)*4,tau=tau)
            p=estimate([90.],c,power=17)
            rows.append({'alpha':alpha,'beta':beta,'tau':tau,**p,'U':p['C']*np.exp(.05*tau),
                'TV':p['C']-40*np.exp(-.05*tau)})
    write('maturity_clock_regimes',rows)
    # Hold a common N=4 full model, freeze a selected factor's stochasticity.
    rows=[]
    for j in range(4):
        b=list(BASE.beta);b[j]*=1e-8;c=replace(BASE,beta=tuple(b));p=estimate(A,c,power=17)
        rows.append({'frozen_period':j+1,**p,'baseline_C':baseprice,'difference':p['C']-baseprice,
            'meaning':'near-frozen factor at fixed current quotes and N=4; covariance also changes'})
    write('factor_freezing',rows)
    summary={'elapsed_seconds':time.time()-start,'baseline_C':baseprice,
        'verification_max_abs_covariance_error':max(x['error'] for x in checks),
        'all_scenarios_in_valid_domain':True,'status':'completed'}
    (ROOT/'run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary,flush=True)

if __name__=='__main__': main()
