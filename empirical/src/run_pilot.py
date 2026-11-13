"""Read original closing prices and estimate explicitly conditional diagnostics.

This script does not assert that a December maturity is a compliance date or
that the legacy 3% discount is a market-calibrated input.
"""
from pathlib import Path
from datetime import date,datetime
import csv,json,hashlib,time,sys,warnings
import numpy as np
from openpyxl import load_workbook
from scipy.stats import jarque_bera,chi2
import pilot_estimator as pe

from release_paths import results_dir, contract_path, full_run
OUT=results_dir()/'estimation'
warnings.filterwarnings('ignore',message='Workbook contains no default style')

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def csvwrite(name,rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def read_source():
    series={};files={}
    for year in [2022,2023]:
        path=contract_path(year)
        book=load_workbook(path,read_only=True,data_only=True)
        rows={}
        for row in book.active.iter_rows(values_only=True):
            if isinstance(row[0],datetime) and isinstance(row[1],(int,float)) and row[1]>0:
                d=row[0].date()
                assert d not in rows
                rows[d]={'close':float(row[1]),'volume':row[7],'oi':row[8],'bid':row[9],'ask':row[10]}
        book.close();series[year]=rows
        files[str(year)]={'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    common=sorted(set(series[2022])&set(series[2023]))
    days=[d for d in common if d.year==2021]
    A=np.array([[series[y][d]['close'] for y in [2022,2023]] for d in days])
    info={'sample':'All common positive closing prices in calendar 2021; no factor-sign selection',
        'n':len(days),'start':days[0].isoformat(),'end':days[-1].isoformat(),
        'files':files,'dates_strictly_ascending':all(a<b for a,b in zip(days,days[1:])),
        'volume_missing':[sum(series[y][d]['volume'] is None for d in days) for y in [2022,2023]],
        'volume_reported_zero':[sum(series[y][d]['volume']==0 for d in days) for y in [2022,2023]],
        'price_ranges':[[float(A[:,j].min()),float(A[:,j].max())] for j in range(2)]}
    return days,A,info

def diagnose(residual,V):
    white=np.linalg.solve(np.linalg.cholesky(V),residual[:,:,None]).squeeze(-1)
    marginal=residual/np.sqrt(np.diagonal(V,axis1=1,axis2=2))
    result={'n_innovations':len(white),'whitened_cross_correlation':float(np.corrcoef(white.T)[0,1]),
        'marginal_standardized_innovation_correlation':float(np.corrcoef(marginal.T)[0,1]),'factors':[]}
    for j in range(2):
        x=white[:,j];centered=x-x.mean();n=len(x)
        acf=[float(centered[k:]@centered[:-k]/(centered@centered)) for k in range(1,11)]
        jb=jarque_bera(x)
        Q=n*(n+2)*sum(acf[k-1]**2/(n-k) for k in range(1,11))
        result['factors'].append({'factor':j+1,'whitened_mean':float(x.mean()),'whitened_sd':float(x.std(ddof=1)),
            'acf_lag1':acf[0],'ljung_box_Q10':float(Q),'ljung_box_reference_p10':float(chi2.sf(Q,10)),
            'jarque_bera':float(jb.statistic),'jarque_bera_reference_p':float(jb.pvalue),
            'abs_innovation_above3_fraction':float(np.mean(abs(x)>3))})
    result['p_value_scope']='Reference asymptotic diagnostics, not parameter-estimation-adjusted or multiplicity-adjusted inference.'
    return result,white

def local_information(fit,times,xi,T):
    if fit['alpha_at_lower_bound'] or fit['alpha_at_upper_bound']:
        return {'status':'not_reported_at_parameter_boundary'}
    separate='beta1' in fit
    q=np.array([fit['alpha'],np.log(fit['beta1']),np.log(fit['beta2']),fit['h1'],fit['h2'],np.arctanh(fit['rho'])]) if separate else np.array([fit['alpha'],np.log(fit['beta']),fit['h1'],fit['h2'],np.arctanh(fit['rho'])])
    def full(v):
        if separate:return pe.likelihood_components(times,xi,T,v[0],np.exp(v[1:3]),np.tanh(v[5]),v[3:5])[0]
        return pe.likelihood_components(times,xi,T,v[0],np.exp(v[1]),np.tanh(v[4]),v[2:4])[0]
    dim=len(q);steps=2e-4*np.maximum(1.,abs(q));base=full(q);H=np.empty((dim,dim))
    for i in range(dim):
        ei=np.eye(dim)[i]*steps[i]
        H[i,i]=(full(q+ei)-2*base+full(q-ei))/steps[i]**2
        for j in range(i):
            ej=np.eye(dim)[j]*steps[j]
            H[i,j]=H[j,i]=(full(q+ei+ej)-full(q+ei-ej)-full(q-ei+ej)+full(q-ei-ej))/(4*steps[i]*steps[j])
    eigen=np.linalg.eigvalsh(H)
    out={'eigenvalues':eigen.tolist(),'status':'non_positive_local_information'}
    if eigen[0]>0:
        se=np.sqrt(np.diag(np.linalg.inv(H)))
        names=['alpha','beta1','beta2','h1','h2','rho'] if separate else ['alpha','beta','h1','h2','rho']
        scale=np.array([1.,fit['beta1'],fit['beta2'],1.,1.,1.-fit['rho']**2]) if separate else np.array([1.,fit['beta'],1.,1.,1.-fit['rho']**2])
        errors=dict(zip(names,(se*scale).tolist()))
        out.update({'status':'model_based_local_information_only','condition_number':float(eigen[-1]/eigen[0]),
            'standard_errors':errors,'wald95_conditional_model_only':{name:[float(fit[name]-1.96*errors[name]),float(fit[name]+1.96*errors[name])] for name in names}})
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    begin=time.time();days,A,sample=read_source()
    assert len(days)==256
    t=np.array([(d-days[0]).days/365.25 for d in days])
    T=np.array([(date(2022,12,19)-days[0]).days/365.25,(date(2023,12,18)-days[0]).days/365.25])
    kappa=float(np.exp(-.03))
    support=[]
    for rate in [-.01,0.,.01,.015,.018,.02,.025,.03,.04]:
        kap=np.exp(-rate*(date(2023,12,18)-date(2022,12,19)).days/365.25)
        X=A.copy();X[:,0]-=kap*A[:,1]
        support.append({'diagnostic_flat_forward_rate':rate,'kappa':float(kap),'n':len(A),
            'positive_first_factor':int(np.sum(X[:,0]>0)),'first_factor_min':float(X[:,0].min()),'first_factor_max':float(X[:,0].max())})
    csvwrite('support_sensitivity.csv',support)
    X=A[:,0]-kappa*A[:,1]
    sample.update({'legacy_kappa':kappa,'legacy_X1_min':float(X.min()),'legacy_X1_max':float(X.max()),
        'constant_kappa_required_strictly_below':float(np.min(A[:,0]/A[:,1])),
        'contract_date_proxies':['2022-12-19','2023-12-18'],'year_fraction_basis':'ACT/365.25',
        'time_gap_days_min_max':[int(min((b-a).days for a,b in zip(days,days[1:]))),int(max((b-a).days for a,b in zip(days,days[1:])))],
        'source_prices_exported':False})
    dump('sample_provenance.json',sample)
    results={};parameters=[];profile=[];diagnostics=[]
    penalty_cases=[('indexed_scenario',np.array([121.01,128.71]))]
    if full_run():penalty_cases.insert(0,('legacy100',np.array([100.,100.])))
    for penalty_label,pi in penalty_cases:
        xi,logj=pe.latent_states(A,kappa,pi)
        for family,lower in ([('positive_clock',1e-6),('binary_clock',1.)] if full_run() else [('binary_clock',1.)]):
            name=penalty_label+'_'+family
            summary,residual,V=pe.fit(t,xi,T,alpha_min=lower)
            summary.update({'penalty':pi.tolist(),'kappa':kappa,'alpha_min':lower,
                'scope':'Conditional diagnostic under the legacy fixed 3% assumption and contract-date proxies. Not market structural calibration.',
                'loglik_price':float(-summary['nll_latent']+logj[1:].sum()),'n_observations':len(A),'n_transitions':len(A)-1})
            summary['local_information']=local_information(summary,t,xi,T)
            summary['innovation_diagnostics'],white=diagnose(residual,V)
            results[name]=summary
            params={'model':name,'beta1':summary['beta'],'beta2':summary['beta'],**{k:summary[k] for k in ['alpha','h1','h2','rho','nll_latent','loglik_price','optimizer_success','gradient_norm','projected_gradient_norm','alpha_at_lower_bound','alpha_at_upper_bound']}}
            parameters.append(params)
            diagnostics.extend([{'model':name,**x} for x in summary['innovation_diagnostics']['factors']])
            csvwrite(name+'_innovations.csv',[{'date':days[i+1].isoformat(),'white1':float(w[0]),'white2':float(w[1])} for i,w in enumerate(white)])
            print(json.dumps(params,ensure_ascii=False),flush=True)
        for family,lower in [('positive_clock',1e-6),('binary_clock',1.)]:
            name=penalty_label+'_separate_beta_'+family
            summary,residual,V=pe.fit_separate_scales(t,xi,T,alpha_min=lower)
            summary.update({'penalty':pi.tolist(),'kappa':kappa,'alpha_min':lower,
                'scope':'Conditional diagnostic under legacy 3% and contract-date proxies; beta1 and beta2 distinct, common alpha. Not market structural calibration.',
                'loglik_price':float(-summary['nll_latent']+logj[1:].sum()),'n_observations':len(A),'n_transitions':len(A)-1,
                'local_information':local_information(summary,t,xi,T)})
            summary['innovation_diagnostics'],white=diagnose(residual,V)
            results[name]=summary
            params={'model':name,**{k:summary[k] for k in ['alpha','beta1','beta2','h1','h2','rho','nll_latent','loglik_price','optimizer_success','gradient_norm','projected_gradient_norm','alpha_at_lower_bound','alpha_at_upper_bound']}}
            parameters.append(params)
            diagnostics.extend([{'model':name,**x} for x in summary['innovation_diagnostics']['factors']])
            csvwrite(name+'_innovations.csv',[{'date':days[i+1].isoformat(),'white1':float(w[0]),'white2':float(w[1])} for i,w in enumerate(white)])
            print(json.dumps(params,ensure_ascii=False),flush=True)
        for alpha in ([0.,.1,.25,.5,.75,1.,1.25,1.5,2.,2.5,3.,4.,5.,6.,8.] if full_run() else []):
            summary,_,_=pe.fit(t,xi,T,fixed_alpha=alpha,starts=[(alpha,.1,0.),(alpha,1.,.5)])
            profile.append({'penalty':penalty_label,'alpha':alpha,'beta1':summary['beta'],'beta2':summary['beta'],'rho':summary['rho'],
                'h1':summary['h1'],'h2':summary['h2'],'nll_latent':summary['nll_latent'],'success':summary['optimizer_success']})
            hs,_,_=pe.fit_separate_scales(t,xi,T,fixed_alpha=alpha)
            profile.append({'penalty':penalty_label+'_separate_beta','alpha':alpha,'beta1':hs['beta1'],'beta2':hs['beta2'],'rho':hs['rho'],
                'h1':hs['h1'],'h2':hs['h2'],'nll_latent':hs['nll_latent'],'success':hs['optimizer_success']})
        dump('conditional_estimation_results.json',results)
    csvwrite('conditional_parameter_estimates.csv',parameters)
    if profile:csvwrite('alpha_profile.csv',profile)
    csvwrite('innovation_diagnostics.csv',diagnostics)
    dump('run_metadata.json',{'python':sys.version,'numpy':np.__version__,'seconds':time.time()-begin,
        'source':'Original workbook closing prices','mapping_gate':'See separate institutional and external-discount audit. Conditional fits do not pass this gate by construction.',
        'normalizer':'Factor-price transformation Jacobian included in price log likelihood; no invalid-factor clipping.',
        'clock_integrals':'Analytic H and diagonal variance; 12-node Gauss-Legendre cross covariance and h drift integrals.',
        'tests':'python -m unittest -v test_pilot_estimator.py','entry':'python run_pilot.py'})
    print('Completed conditional diagnostics in %.2f seconds'%(time.time()-begin),flush=True)

if __name__=='__main__':main()
