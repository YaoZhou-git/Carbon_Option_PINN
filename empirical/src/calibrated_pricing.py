"""Risk contributions and Q prices under the explicitly fixed two-period mapping.

The raw closing-price workbooks and original estimator are read only.
Conditional integration is deterministic; independent Sobol checks are separate.
"""
from pathlib import Path
from datetime import date
import sys, json, csv, time
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.integrate import quad
from scipy.stats import norm

from release_paths import results_dir, full_run
OUT = results_dir()/'pricing'
PILOT = results_dir()/'estimation'
import pilot_estimator as pe
from run_pilot import read_source


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csvwrite(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def q_gaussian(a, b, T, alpha, beta, rho, xi):
    """Stable scaled Gaussian Y=xi(tau)/M, avoiding exponential covariances."""
    H=pe.integrated_clock(a, b, T, alpha, beta)
    M=np.exp(H/2)
    V=np.zeros((2,2));V[0,0]=-np.expm1(-H[0]);V[1,1]=-np.expm1(-H[1])
    def kernel(s):
        hs=pe.integrated_clock(a,s,T,alpha,beta)
        z=beta*(T-s)**(-alpha)
        return np.sqrt(z[0]*z[1])*np.exp(-.5*hs.sum())
    V[0,1]=V[1,0]=rho*quad(kernel,a,b,epsabs=2e-11,epsrel=2e-11)[0]
    assert np.linalg.eigvalsh(V)[0]>0
    return np.asarray(xi),V,M


def conditional_call(mean, sd, strike, pi, eps=1e-9):
    if strike<=0:return pi*ndtr(mean/np.sqrt(1+sd*sd))-strike
    if strike>=pi:return 0.
    if sd<1e-12:return max(pi*ndtr(mean)-strike,0.)
    cut=(ndtri(strike/pi)-mean)/sd
    if cut>11:return 0.
    if cut < -11:return pi*ndtr(mean/np.sqrt(1+sd*sd))-strike
    # Explicit split at the CDF transition remains accurate for large sd.
    f=lambda u:(pi*ndtr(mean+sd*u)-strike)*norm.pdf(u)
    points=[max(cut,-11),11.]
    for v in [-8.,-4.,0.,4.,8.]:
        u=(v-mean)/sd
        if points[0]<u<points[-1]:points.append(u)
    points=sorted(set(points))
    return sum(quad(f,x,y,epsabs=eps/len(points),epsrel=2e-10,limit=100)[0] for x,y in zip(points,points[1:]))


def call_price(a,b,T,alpha,beta,rho,xi,pi,kappa,K,r,eps=2e-8):
    mean,V,M=q_gaussian(a,b,T,alpha,np.array(beta),rho,xi)
    s2=np.sqrt(V[1,1]);c=V[0,1]/s2
    s1=np.sqrt(V[0,0]-c*c)
    def f(v):
        y2=M[1]*(mean[1]+s2*v)
        strike=K-kappa*pi[1]*ndtr(y2)
        return conditional_call(M[0]*(mean[0]+c*v), M[0]*s1, strike, pi[0], eps=eps*.05)*norm.pdf(v)
    points=[-10.,10.]
    # Price kink transitions and regions of changing probit curvature.
    for q in [K/(kappa*pi[1]),(K-pi[0])/(kappa*pi[1])]:
        if 0<q<1:
            v=(ndtri(q)/M[1]-mean[1])/s2
            if -10<v<10:points.append(v)
    points=sorted(set(points))
    integral=sum(quad(f,x,y,epsabs=eps/len(points),epsrel=2e-9,limit=140)[0] for x,y in zip(points,points[1:]))
    return np.exp(-r*(b-a))*integral


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.time();days,A,provenance=read_source()
    t=np.array([(d-days[0]).days/365.25 for d in days])
    T=np.array([(date(2022,12,19)-days[0]).days/365.25,(date(2023,12,18)-days[0]).days/365.25])
    kappa=float(np.exp(-.03));r=.03
    results=json.loads((PILOT/'conditional_estimation_results.json').read_text(encoding='utf-8'))
    ref=max(i for i,d in enumerate(days) if d<=date(2021,6,30))
    rows=[];riskrows=[];summaries={};checks=[]
    cases = [
        ('baseline_separate','legacy100_separate_beta_binary_clock',[100.,100.]),
        ('baseline_common','legacy100_binary_clock',[100.,100.]),
        ('indexed_separate','indexed_scenario_separate_beta_binary_clock',[121.01,128.71]),
        ('indexed_common','indexed_scenario_binary_clock',[121.01,128.71])]
    if not full_run():cases=[case for case in cases if case[0]=='indexed_separate']
    for label,key,pi in cases:
        fit=results[key];alpha=fit['alpha'];beta=np.array([fit.get('beta1',fit.get('beta')),fit.get('beta2',fit.get('beta'))]);rho=fit['rho']
        xi,_=pe.latent_states(A,kappa,pi)
        z=beta*(T-t[:,None])**(-alpha)
        b=np.asarray(pi)*norm.pdf(xi)*np.sqrt(z)
        own=b[:,0]**2;later=(kappa*b[:,1])**2;cross=2*kappa*rho*b[:,0]*b[:,1];total=own+later+cross
        for i,d in enumerate(days):
            riskrows.append({'model':label,'date':d.isoformat(),'z1':float(z[i,0]),'z2':float(z[i,1]),
                'b1':float(b[i,0]),'b2':float(b[i,1]),'own_variance':float(own[i]),'later_variance':float(later[i]),
                'cross_variance':float(cross[i]),'total_variance':float(total[i]),'own_share':float(own[i]/total[i]),
                'later_share':float(later[i]/total[i]),'cross_share':float(cross[i]/total[i]),
                'later_euler_share':float((later[i]+cross[i]/2)/total[i])})
        summaries[label]={'fit_key':key,'alpha':alpha,'beta':beta.tolist(),'rho':rho,'penalty':pi,
            'mean_own_share':float(np.mean(own/total)),'mean_later_share':float(np.mean(later/total)),
            'mean_cross_share':float(np.mean(cross/total)),'mean_later_euler_share':float(np.mean((later+cross/2)/total)),
            'later_euler_share_min_max':[float(np.min((later+cross/2)/total)),float(np.max((later+cross/2)/total))],
            'reference_risk':{k:v for k,v in riskrows[-len(days)+ref].items() if k not in ['model','date']}}
        for horizon in sorted(set([.05,.1,.15,.2,.25,.3,.4,.5,.6,.7,.8,.9,1.,1.1,1.2])):
            price=call_price(t[ref],t[ref]+horizon,T,alpha,beta,rho,xi[ref],pi,kappa,A[ref,0],r)
            rows.append({'model':label,'remaining_years':horizon,'strike':float(A[ref,0]),'call_price':float(price)})
        for horizon in [.25,.5,1.]:
            tight=call_price(t[ref],t[ref]+horizon,T,alpha,beta,rho,xi[ref],pi,kappa,A[ref,0],r,eps=2e-10)
            orig=next(x['call_price'] for x in rows if x['model']==label and x['remaining_years']==horizon)
            checks.append({'model':label,'horizon':horizon,'tolerance_price_difference':float(abs(tight-orig))})
        # Separate correlation channel at unchanged diffusion scales and current quotes.
        if label=='baseline_separate':
            for horizon in [.25,.5,1.]:
                for rho_test in [0.,rho]:
                    price=call_price(t[ref],t[ref]+horizon,T,alpha,beta,rho_test,xi[ref],pi,kappa,A[ref,0],r)
                    checks.append({'channel':'fixed_scales_correlation','horizon':horizon,'rho':rho_test,'price':float(price)})
    csvwrite('calibrated_risk_contributions.csv',riskrows)
    csvwrite('calibrated_option_prices.csv',rows)
    dump('calibrated_pricing_results.json',{'sample':provenance,'reference_date':days[ref].isoformat(),'reference_quotes':A[ref].tolist(),
        'risk_clock_dates':['2022-12-19','2023-12-18'],'reference_years_to_dates':(T-t[ref]).tolist(),
        'kappa':kappa,'r':r,'risk_neutral_drift':'.5*z*xi; historical h removed',
        'integration':'Nested deterministic normal integrals, adaptive Gauss-Kronrod; scaled latent covariance computed by direct quadrature.',
        'models':summaries,'checks':checks,'seconds':time.time()-start})
    print(json.dumps({'reference_date':days[ref].isoformat(),'reference_quotes':A[ref].tolist(),'seconds':time.time()-start,'models':summaries,'checks':checks},indent=2),flush=True)


if __name__=='__main__':main()
