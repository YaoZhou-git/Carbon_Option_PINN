"""Two-factor exact-transition estimation for a fixed, externally specified mapping.

No factor clipping or admissibility-based subsampling is performed here.
The physical latent drift is 0.5*z*xi + h_j*sqrt(z).
"""
import numpy as np
from scipy.special import ndtri
from scipy.optimize import minimize

_nodes,_weights=np.polynomial.legendre.leggauss(12)


def integrated_clock(a,b,T,alpha,beta):
    a,b,T=np.asarray(a),np.asarray(b),np.asarray(T)
    if np.any(b>=T) or np.any(a>b) or np.any(np.asarray(beta)<=0):
        raise ValueError('Require a<=b<T and positive beta')
    p=1.-alpha
    logratio=np.log1p((b-a)/(T-b))
    if abs(p)<1e-8:
        return beta*logratio
    return beta*np.power(T-b,p)*np.expm1(p*logratio)/p


def transition(a,b,T,alpha,beta,rho):
    a=np.atleast_1d(a).astype(float);b=np.atleast_1d(b).astype(float)
    T=np.asarray(T,dtype=float)
    if not -1<rho<1 or alpha<0:
        raise ValueError('Invalid alpha or correlation')
    H=integrated_clock(a[:,None],b[:,None],T,alpha,beta)
    M=np.exp(H/2.)
    s=(a[:,None]+b[:,None])/2.+(b-a)[:,None]*_nodes[None,:]/2.
    weights=(b-a)[:,None]*_weights[None,:]/2.
    Hsb=integrated_clock(s[:,:,None],b[:,None,None],T,alpha,beta)
    kernel=np.exp(Hsb/2.)*np.sqrt(beta*np.power(T-s[:,:,None],-alpha))
    g=np.sum(weights[:,:,None]*kernel,axis=1)
    V=np.zeros((len(a),2,2))
    V[:,0,0]=np.expm1(H[:,0]);V[:,1,1]=np.expm1(H[:,1])
    V[:,0,1]=V[:,1,0]=rho*np.sum(weights*kernel[:,:,0]*kernel[:,:,1],axis=1)
    return M,g,V


def latent_states(A,kappa,pi):
    A=np.asarray(A,dtype=float);pi=np.asarray(pi,dtype=float)
    X=A.copy();X[:,0]-=kappa*A[:,1]
    if np.any(X<=0) or np.any(X>=pi):
        raise ValueError('Observed factors outside (0,pi); no clipping is allowed')
    xi=ndtri(X/pi)
    logphi=-.5*xi**2-.5*np.log(2*np.pi)
    logj=-np.sum(np.log(pi)+logphi,axis=1)
    return xi,logj


def profile_drifts(prev,nxt,M,g,V):
    inv=np.linalg.inv(V)
    target=nxt-M*prev
    information=np.einsum('ni,nij,nj->ij',g,inv,g)
    score=np.sum(g*np.einsum('nij,nj->ni',inv,target),axis=0)
    return np.linalg.solve(information,score)


def likelihood_components(times,xi,T,alpha,beta,rho,h=None):
    M,g,V=transition(times[:-1],times[1:],T,alpha,beta,rho)
    if h is None:h=profile_drifts(xi[:-1],xi[1:],M,g,V)
    residual=xi[1:]-M*xi[:-1]-g*np.asarray(h)
    sign,logdet=np.linalg.slogdet(V)
    if np.any(sign<=0):raise ValueError('Non-positive covariance')
    inv=np.linalg.inv(V)
    nll=.5*np.sum(2*np.log(2*np.pi)+logdet+np.einsum('ni,nij,nj->n',residual,inv,residual))
    return float(nll),np.asarray(h),residual,V


def objective(q,times,xi,T,fixed_alpha=None):
    try:
        if fixed_alpha is None:
            alpha,logbeta,eta=q
        else:
            alpha=fixed_alpha;logbeta,eta=q
        beta=np.exp(logbeta);rho=np.tanh(eta)
        if not np.isfinite(beta):return 1e100
        return likelihood_components(times,xi,T,alpha,beta,rho)[0]
    except (ValueError,np.linalg.LinAlgError,FloatingPointError,OverflowError):
        return 1e100


def projected_gradient_norm(result,bounds):
    g=np.asarray(result.jac).copy()
    for j,(lo,hi) in enumerate(bounds):
        if (result.x[j]-lo<1e-5 and g[j]>0) or (hi-result.x[j]<1e-5 and g[j]<0):g[j]=0.
    return float(np.linalg.norm(g))


def fit(times,xi,T,alpha_min=0.,fixed_alpha=None,starts=None):
    if starts is None:
        starts=[(.2,.1,0.),(.7,.3,.3),(1.2,.8,0.),(2.,2.,-.3),(4.,5.,.5),(1.,.05,.7)]
    bounds=[(alpha_min,8.),(-16.,9.),(-4.95,4.95)] if fixed_alpha is None else [(-16.,9.),(-4.95,4.95)]
    runs=[]
    for alpha,beta,rho in starts:
        q=[max(alpha,alpha_min),np.log(beta),np.arctanh(rho)] if fixed_alpha is None else [np.log(beta),np.arctanh(rho)]
        result=minimize(objective,q,args=(times,xi,T,fixed_alpha),method='L-BFGS-B',bounds=bounds,
            options={'maxiter':1000,'ftol':1e-12,'gtol':1e-6,'maxls':40})
        runs.append(result)
    best=min(runs,key=lambda x:x.fun)
    if fixed_alpha is None:alpha,logbeta,eta=best.x
    else:alpha=fixed_alpha;logbeta,eta=best.x
    beta=np.exp(logbeta);rho=np.tanh(eta)
    nll,h,residual,V=likelihood_components(times,xi,T,alpha,beta,rho)
    summary={'alpha':float(alpha),'beta':float(beta),'rho':float(rho),'h1':float(h[0]),'h2':float(h[1]),
        'nll_latent':nll,'optimizer_success':bool(best.success),'message':str(best.message),
        'gradient_norm':float(np.linalg.norm(best.jac)),
        'projected_gradient_norm':projected_gradient_norm(best,bounds),
        'alpha_at_lower_bound':bool(fixed_alpha is None and alpha-alpha_min<1e-5),
        'alpha_at_upper_bound':bool(fixed_alpha is None and 8.-alpha<1e-5),
        'runs':[{'nll':float(r.fun),'success':bool(r.success),'iterations':int(r.nit),'q':r.x.tolist()} for r in runs]}
    return summary,residual,V


def fit_separate_scales(times,xi,T,alpha_min=0.,fixed_alpha=None):
    def obj(q):
        if fixed_alpha is None:alpha=q[0];logs=q[1:3];eta=q[3]
        else:alpha=fixed_alpha;logs=q[:2];eta=q[2]
        try:return likelihood_components(times,xi,T,alpha,np.exp(logs),np.tanh(eta))[0]
        except (ValueError,np.linalg.LinAlgError,FloatingPointError,OverflowError):return 1e100
    bounds=([(alpha_min,8.)] if fixed_alpha is None else [])+[(-16.,9.),(-16.,9.),(-4.95,4.95)]
    runs=[]
    for alpha,b1,b2,rho in [(.2,.1,.3,.3),(1.2,.3,.8,.6),(3.,1.,2.,0.),(.7,.5,.1,-.2)]:
        q=([max(alpha,alpha_min)] if fixed_alpha is None else [])+[np.log(b1),np.log(b2),np.arctanh(rho)]
        runs.append(minimize(obj,q,method='L-BFGS-B',bounds=bounds,options={'maxiter':1500,'ftol':1e-12,'gtol':1e-6,'maxls':40}))
    best=min(runs,key=lambda r:r.fun)
    if fixed_alpha is None:alpha=best.x[0];logs=best.x[1:3];eta=best.x[3]
    else:alpha=fixed_alpha;logs=best.x[:2];eta=best.x[2]
    beta=np.exp(logs);rho=np.tanh(eta)
    nll,h,residual,V=likelihood_components(times,xi,T,alpha,beta,rho)
    summary={'alpha':float(alpha),'beta1':float(beta[0]),'beta2':float(beta[1]),'rho':float(rho),
        'h1':float(h[0]),'h2':float(h[1]),'nll_latent':nll,'optimizer_success':bool(best.success),
        'gradient_norm':float(np.linalg.norm(best.jac)),'projected_gradient_norm':projected_gradient_norm(best,bounds),
        'alpha_at_lower_bound':bool(fixed_alpha is None and alpha-alpha_min<1e-5),
        'alpha_at_upper_bound':bool(fixed_alpha is None and 8.-alpha<1e-5),
        'runs':[{'nll':float(r.fun),'success':bool(r.success),'iterations':int(r.nit),'q':r.x.tolist()} for r in runs]}
    return summary,residual,V
