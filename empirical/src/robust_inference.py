"""Full-parameter HAC inference for conditional two-factor Gaussian QMLE.

The licensed source prices are read in place; no daily prices or scores are
exported. Fixed kappa and contract-date proxies define each conditional model.
The external ECB AAA proxy's state-domain failure is reported separately.
"""
from pathlib import Path
from datetime import date, datetime, timezone
import csv, hashlib, json, sys, time
import numpy as np
from scipy.stats import norm

from release_paths import results_dir, full_run
OUT=results_dir()/'inference'
PILOT=results_dir()/'estimation'
import pilot_estimator as pe
import run_pilot as rp

NAMES=['alpha','beta1','beta2','h1','h2','rho']
QNAMES=['alpha','logbeta1','logbeta2','h1','h2','atanh_rho']
BANDWIDTHS=[4,8,12]
STEPS=[1e-4,2e-4]

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def q_from_fit(fit):
    return np.array([fit['alpha'],np.log(fit['beta1']),np.log(fit['beta2']),fit['h1'],fit['h2'],np.arctanh(fit['rho'])])

def natural(q):
    v=q.copy();v[1:3]=np.exp(v[1:3]);v[5]=np.tanh(v[5]);return v

def nll_contributions(q,times,xi,T):
    M,g,V=pe.transition(times[:-1],times[1:],T,q[0],np.exp(q[1:3]),np.tanh(q[5]))
    residual=xi[1:]-M*xi[:-1]-g*q[3:5]
    sign,logdet=np.linalg.slogdet(V)
    if not np.all(sign>0):raise ValueError('Nonpositive covariance')
    return .5*(2*np.log(2*np.pi)+logdet+np.einsum('ni,nij,nj->n',residual,np.linalg.inv(V),residual))

def derivatives(q,times,xi,T,relative_step):
    steps=relative_step*np.maximum(1.,abs(q));dim=len(q)
    origin=nll_contributions(q,times,xi,T)
    scores=np.empty((len(origin),dim));H=np.empty((dim,dim))
    for i in range(dim):
        ei=np.eye(dim)[i]*steps[i]
        plus=nll_contributions(q+ei,times,xi,T);minus=nll_contributions(q-ei,times,xi,T)
        scores[:,i]=-(plus-minus)/(2*steps[i])
        H[i,i]=np.sum(plus-2*origin+minus)/steps[i]**2
        for j in range(i):
            ej=np.eye(dim)[j]*steps[j]
            cross=(nll_contributions(q+ei+ej,times,xi,T)-nll_contributions(q+ei-ej,times,xi,T)
                  -nll_contributions(q-ei+ej,times,xi,T)+nll_contributions(q-ei-ej,times,xi,T))
            H[i,j]=H[j,i]=np.sum(cross)/(4*steps[i]*steps[j])
    M,g,V=pe.transition(times[:-1],times[1:],T,q[0],np.exp(q[1:3]),np.tanh(q[5]))
    residual=xi[1:]-M*xi[:-1]-g*q[3:5]
    analytic_h=g*np.einsum('nij,nj->ni',np.linalg.inv(V),residual)
    return scores,H,{'relative_step':relative_step,'parameter_steps':steps.tolist(),
                     'score_sum':scores.sum(axis=0).tolist(),
                     'score_sum_norm':float(np.linalg.norm(scores.sum(axis=0))),
                     'max_h_score_difference_from_analytic':float(np.max(abs(scores[:,3:5]-analytic_h))),
                     'hessian_eigenvalues':np.linalg.eigvalsh(H).tolist(),
                     'hessian_condition_number':float(np.linalg.cond(H))}

def bartlett_meat(scores,bandwidth):
    # Observations, not calendar-day lags. Centering changes a converged MLE's
    # scores only by numerical error. No additional n/(n-k) factor is applied.
    x=scores-scores.mean(axis=0);B=x.T@x
    for lag in range(1,bandwidth+1):
        gamma=x[lag:].T@x[:-lag]
        B+=(1.-lag/(bandwidth+1.))*(gamma+gamma.T)
    return (B+B.T)/2.

def transformed_interval(q,se_q):
    lo=q-1.959963984540054*se_q;hi=q+1.959963984540054*se_q
    lo[1:3]=np.exp(lo[1:3]);hi[1:3]=np.exp(hi[1:3])
    lo[5]=np.tanh(lo[5]);hi[5]=np.tanh(hi[5])
    return {name:[float(a),float(b)] for name,a,b in zip(NAMES,lo,hi)}

def sandwich(q,scores,H,bandwidth):
    eigenH=np.linalg.eigvalsh(H)
    if eigenH.min()<=0:raise ValueError('Observed Hessian is not positive definite')
    inverseH=np.linalg.inv(H);B=bartlett_meat(scores,bandwidth)
    covariance_q=inverseH@B@inverseH
    covariance_q=(covariance_q+covariance_q.T)/2.
    if np.diag(covariance_q).min()<=0:raise ValueError('Nonpositive sandwich variances')
    point=natural(q);D=np.diag([1.,point[1],point[2],1.,1.,1.-point[5]**2])
    covariance=D@covariance_q@D
    se=np.sqrt(np.diag(covariance));se_q=np.sqrt(np.diag(covariance_q))
    model_se=np.sqrt(np.diag(D@inverseH@D))
    contrast=np.array([0.,-1.,1.,0.,0.,0.]);logratio=float(contrast@q)
    ratio_se=float(np.sqrt(contrast@covariance_q@contrast))
    return {'bandwidth':bandwidth,'hac_standard_errors':dict(zip(NAMES,se.tolist())),
            'model_based_standard_errors':dict(zip(NAMES,model_se.tolist())),
            'natural_scale_wald95':{name:[float(x-1.959963984540054*s),float(x+1.959963984540054*s)] for name,x,s in zip(NAMES,point,se)},
            'transform_respecting_wald95':transformed_interval(q,se_q),
            'beta2_over_beta1':{'estimate':float(np.exp(logratio)),
                'log_ratio_se':ratio_se,'wald95':[float(np.exp(logratio-1.959963984540054*ratio_se)),float(np.exp(logratio+1.959963984540054*ratio_se))]},
            'covariance_q':covariance_q.tolist(),'covariance_natural':covariance.tolist(),
            'meat_eigenvalues':np.linalg.eigvalsh(B).tolist()}

def range_stats(A,kappa,pi):
    X=A.copy();X[:,0]-=kappa*A[:,1]
    return {'observations':len(X),'legal_factor_rows':int(np.sum(np.all((X>0)&(X<pi),axis=1))),
            'X1_range':[float(X[:,0].min()),float(X[:,0].max())]}

def main():
    start=time.time();OUT.mkdir(parents=True,exist_ok=True)
    days,A,source=rp.read_source()
    times=np.array([(d-days[0]).days/365.25 for d in days])
    T=np.array([(date(2022,12,19)-days[0]).days/365.25,(date(2023,12,18)-days[0]).days/365.25])
    old=json.loads((PILOT/'conditional_estimation_results.json').read_text(encoding='utf-8'))
    kappa=float(np.exp(-.03));rows=[]
    external_path=PILOT/'external_kappa_audit_summary.json'
    external=json.loads(external_path.read_text(encoding='utf-8')) if external_path.exists() else {'ecb_AAA_forward_proxy':{'status':'Optional external-domain audit not run; no proxy result claimed'}}
    result={'generated_utc':datetime.now(timezone.utc).isoformat(),
       'method':{'parameter_order':NAMES,'optimization_coordinate_order':QNAMES,
          'covariance':'H^{-1} B_L H^{-1}; H=sum of negative-log-likelihood Hessians; B_L=Bartlett HAC sum of centered full six-parameter scores',
          'bandwidths':BANDWIDTHS,'lag_unit':'successive common observations, not calendar days',
          'finite_sample_n_over_n_minus_k_correction':False,
          'primary_relative_fd_step':STEPS[0],'stability_check_relative_fd_step':STEPS[1],
          'physical_latent_sde':'d xi_j = [0.5 z_j(t) xi_j + h_j sqrt(z_j(t))] dt + sqrt(z_j(t)) dW_j^P; d[W_1^P,W_2^P]_t=rho dt',
          'h_interpretation':'h_j is the physical latent drift shift relative to the zero-drift-factor pricing measure, in correlated Brownian coordinates; the independent-Brownian market-price-of-risk vector is R^{-1/2} h',
          'girsanov_if_needed':'dQ/dP = exp[- integral h^T R^{-1} dW^P - 0.5 integral h^T R^{-1} h dt], for fixed nonsingular R and constant h',
          'transition':'Exact Gaussian mean/covariance integrals; analytic integrated clocks/diagonal variances; 12-point Gauss-Legendre drift and cross-covariance integrals',
          'price_jacobian':'Included in the price log likelihood; constant in these six parameters because kappa and pi_j are fixed, so contributes zero score and Hessian',
          'interval_transformations':'For transform_respecting_wald95, log-beta intervals are exponentiated and Fisher-rho intervals mapped by tanh; alpha and h use identity coordinates. Alpha intervals are not clipped at the estimation lower bound 1.',
          'interval_scope':'Conditional asymptotic QMLE/HAC inference for fixed kappa, date proxies, and penalty scales; no inference about unidentified mapping inputs'},
       'sample':{'start':str(days[0]),'end':str(days[-1]),'observations':len(days),'transitions':len(days)-1,
           'price_files_sha256':{name:item['sha256'] for name,item in source['files'].items()},
           'no_factor_sign_subsampling_or_clipping':True,'daily_source_prices_exported':False},
       'conditional_inputs':{'kappa':kappa,'historical_rate_assumption':.03,'date_proxies':['2022-12-19','2023-12-18'],
                            'primary_pi':[100.,100.],'indexed_sensitivity_pi':[121.01,128.71]},
       'external_mapping_diagnostic':external['ecb_AAA_forward_proxy'],
       'models':{}}
    cases=[('indexed_sensitivity','indexed_scenario_separate_beta_binary_clock',np.array([121.01,128.71]))]
    if full_run():cases.insert(0,('primary_pi100','legacy100_separate_beta_binary_clock',np.array([100.,100.])))
    for label,source_name,pi in cases:
        fit=old[source_name];q=q_from_fit(fit);xi,logj=pe.latent_states(A,kappa,pi)
        contributions=nll_contributions(q,times,xi,T)
        assert abs(contributions.sum()-fit['nll_latent'])<1e-8
        assert abs(-contributions.sum()+logj[1:].sum()-fit['loglik_price'])<1e-8
        derivatives_at_steps=[derivatives(q,times,xi,T,step) for step in STEPS]
        answers=[]
        for scores,H,checks in derivatives_at_steps:
            answers.append({'checks':checks,'hessian_q':H.tolist(),
                            'bandwidth_results':{str(L):sandwich(q,scores,H,L) for L in BANDWIDTHS}})
        H0=np.array(answers[0]['hessian_q']);H1=np.array(answers[1]['hessian_q'])
        relative_hessian_difference=float(np.linalg.norm(H0-H1)/np.linalg.norm(H0))
        se_differences={}
        for L in BANDWIDTHS:
            se0=np.array(list(answers[0]['bandwidth_results'][str(L)]['hac_standard_errors'].values()))
            se1=np.array(list(answers[1]['bandwidth_results'][str(L)]['hac_standard_errors'].values()))
            se_differences[str(L)]=float(np.max(abs(se0-se1)/se0))
        stability={'relative_hessian_frobenius_difference':relative_hessian_difference,
                   'max_relative_HAC_SE_difference_by_bandwidth':se_differences,
                   'passed':bool(relative_hessian_difference<1e-3 and max(se_differences.values())<1e-3
                                 and max(a['checks']['max_h_score_difference_from_analytic'] for a in answers)<1e-6)}
        model={'source_fit_key':source_name,'estimates':{name:float(fit[name]) for name in NAMES},
               'pi':pi.tolist(),'alpha_constraint':'>=1','alpha_at_lower_bound':fit['alpha_at_lower_bound'],
               'alpha_at_upper_bound':fit['alpha_at_upper_bound'],'loglik_price':fit['loglik_price'],
               'domain':range_stats(A,kappa,pi),'stability':stability,
               'primary':answers[0],'second_step_check':answers[1]}
        result['models'][label]=model
        for L,data in answers[0]['bandwidth_results'].items():
            for parameter in NAMES:
                rows.append({'model':label,'bandwidth':int(L),'parameter':parameter,
                    'estimate':model['estimates'][parameter],'model_based_se':data['model_based_standard_errors'][parameter],
                    'hac_se':data['hac_standard_errors'][parameter],
                    'wald95_natural_lower':data['natural_scale_wald95'][parameter][0],
                    'wald95_natural_upper':data['natural_scale_wald95'][parameter][1],
                    'wald95_transform_lower':data['transform_respecting_wald95'][parameter][0],
                    'wald95_transform_upper':data['transform_respecting_wald95'][parameter][1]})
        print(json.dumps({'model':label,'estimates':model['estimates'],'stability':stability,
                          'HAC8':answers[0]['bandwidth_results']['8']['hac_standard_errors']},ensure_ascii=False),flush=True)
    result['seconds']=time.time()-start
    dump('robust_inference_results.json',result)
    with (OUT/'robust_inference_results.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    sensitivity={'scope':'Fixed-kappa conditional sensitivity; rates below are assumptions, not external market estimates',
                 'pi':[100.,100.],'alpha_constraint':'>=1','models':{},
                 'external_mapping_diagnostic':external['ecb_AAA_forward_proxy']}
    for rate in ([.025,.04] if full_run() else []):
        kap=float(np.exp(-rate));pi=np.array([100.,100.]);domain=range_stats(A,kap,pi)
        item={'assumed_one_year_continuous_rate':rate,'kappa':kap,'domain':domain}
        if domain['legal_factor_rows']==len(A):
            xi,logj=pe.latent_states(A,kap,pi)
            fit,_,_=pe.fit_separate_scales(times,xi,T,alpha_min=1.)
            fit['loglik_price']=float(-fit['nll_latent']+logj[1:].sum())
            item['fit']=fit
        else:item['status']='No estimation because observed factors violate the state domain'
        sensitivity['models'][str(rate)]=item
        print(json.dumps({'rate_sensitivity':rate,**item},ensure_ascii=False),flush=True)
    if full_run():dump('rate_sensitivity.json',sensitivity)
    print('Completed robust inference.'+(' Historical rate sensitivity also completed.' if full_run() else ''),flush=True)

if __name__=='__main__':main()
