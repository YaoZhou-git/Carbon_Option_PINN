"""Old recursive reduced-form model, solved in its bounded penalty factors.

No Monte Carlo, data clipping, price-box truncation or endogenous abatement
state is introduced. The deterministic Gaussian expectation solves the same
backward equation as the factor-coordinate PINN in this package.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri


@dataclass(frozen=True)
class Model:
    pi: float = 100.
    r: float = .05
    K: float = 50.
    T: tuple = (4.,8.,12.,16.)
    alpha: tuple = (1.2,)*4
    beta: tuple = (.8,)*4
    kappa: tuple = (float(np.exp(-.2)),)*3
    rho: float = .3
    tau: float = 3.


def matrix(cfg, n):
    L=np.eye(n)
    for i in range(n):
        for j in range(i+1,n): L[i,j]=L[i,j-1]*cfg.kappa[j-1]
    return L


def factors(a,cfg):
    a=np.asarray(a,dtype=float)
    x=a.copy()
    x[:-1]-=np.asarray(cfg.kappa[:len(a)-1])*a[1:]
    if np.any(x<=0) or np.any(x>=cfg.pi):
        raise ValueError(f'Invalid recursive factors {x}; require 0<X_j<pi, without clipping')
    return x


def prices(x,cfg): return matrix(cfg,len(x))@np.asarray(x)


def integrated_z(cfg,j,t,s):
    T,alpha,beta=cfg.T[j],cfg.alpha[j],cfg.beta[j]
    if abs(alpha-1)<1e-12: return beta*np.log((T-t)/(T-s))
    return beta*((T-t)**(1-alpha)-(T-s)**(1-alpha))/(1-alpha)


@lru_cache(maxsize=256)
def covariance(cfg,n,t,tau):
    if not 0<=t<=tau<min(cfg.T[:n]):
        raise ValueError('Require 0<=t<=tau<first compliance horizon')
    if n>1 and not -1/(n-1)<cfg.rho<1: raise ValueError('Invalid equicorrelation')
    Z=np.array([integrated_z(cfg,j,t,tau) for j in range(n)])
    M=np.exp(Z/2)
    V=np.diag(np.expm1(Z))
    for i in range(n):
        for j in range(i+1,n):
            def f(s):
                zi=cfg.beta[i]*(cfg.T[i]-s)**(-cfg.alpha[i])
                zj=cfg.beta[j]*(cfg.T[j]-s)**(-cfg.alpha[j])
                return np.sqrt(zi*zj)*np.exp(-.5*(integrated_z(cfg,i,t,s)+integrated_z(cfg,j,t,s)))
            V[i,j]=V[j,i]=cfg.rho*M[i]*M[j]*quad(f,t,tau,epsabs=1e-11,epsrel=1e-11)[0]
    return M,V


def transition(x,cfg,t,tau):
    M,V=covariance(cfg,len(x),float(t),float(tau))
    return M*ndtri(np.asarray(x)/cfg.pi),V


@lru_cache(maxsize=32)
def normal_grid(d,order):
    if d==0: return np.zeros((1,0)),np.ones(1)
    nodes,w=np.polynomial.hermite.hermgauss(order)
    nodes=nodes*np.sqrt(2);w=w/np.sqrt(np.pi)
    grids=np.meshgrid(*([nodes]*d),indexing='ij')
    weights=np.meshgrid(*([w]*d),indexing='ij')
    return np.stack([g.ravel() for g in grids],axis=1),np.prod(np.stack(weights),axis=0).ravel()


@lru_cache(maxsize=8)
def uniform_grid(order):
    n,w=np.polynomial.legendre.leggauss(order)
    return (n+1)/2,w/2


def conditional_call(mu,var,b,cfg,inner):
    mu=np.atleast_1d(mu);b=np.atleast_1d(b)
    sd=np.sqrt(max(var,0.))
    expected=np.zeros(len(mu))
    full=b>=cfg.K
    expected[full]=cfg.pi*ndtr(mu[full]/np.sqrt(1+var))+b[full]-cfg.K
    partial=(b<cfg.K)&(b+cfg.pi>cfg.K)
    if np.any(partial):
        k=(cfg.K-b[partial])/cfg.pi
        d=(ndtri(k)-mu[partial])/sd
        tail=ndtr(-d)
        u,uw=uniform_grid(inner)
        z=ndtri(np.clip(1-tail[:,None]*u,1e-15,1-1e-15))
        payoff=cfg.pi*ndtr(mu[partial,None]+sd*z)+b[partial,None]-cfg.K
        expected[partial]=tail*(payoff@uw)
    return expected


def value(a,cfg=Model(),t=0.,tau=None,order=44,inner=128):
    tau=cfg.tau if tau is None else float(tau)
    x=factors(a,cfg);n=len(x)
    if t==tau: return float(max(a[0]-cfg.K,0))
    m,V=transition(x,cfg,t,tau)
    if n==2:
        # The sharp but smooth transition when the future component alone
        # exceeds K converges slowly under low-order Hermite rules. Resolve
        # the remaining one-dimensional integral adaptively, deterministically.
        s=np.sqrt(V[1,1]);q=V[0,1]/s
        var=V[0,0]-q*q;weight=matrix(cfg,n)[0,1]
        def f(z):
            b=cfg.pi*weight*ndtr(m[1]+s*z)
            return conditional_call(m[0]+q*z,var,b,cfg,inner)[0]*np.exp(-z*z/2)/np.sqrt(2*np.pi)
        points=[]
        if 0<cfg.K/(cfg.pi*weight)<1:
            point=(ndtri(cfg.K/(cfg.pi*weight))-m[1])/s
            if -9<point<9:points=[point]
        ans=quad(f,-9,9,epsabs=2e-7,epsrel=2e-8,points=points,limit=200)[0]
        return float(np.exp(-cfg.r*(tau-t))*ans)
    nodes,w=normal_grid(n-1,order)
    if n>1:
        VF=V[1:,1:]
        yf=m[1:]+nodes@np.linalg.cholesky(VF).T
        q=np.linalg.solve(VF,V[1:,0])
        mu=m[0]+(yf-m[1:])@q
        var=V[0,0]-V[0,1:]@q
        b=cfg.pi*ndtr(yf)@matrix(cfg,n)[0,1:]
    else:
        mu=np.array([m[0]]);var=V[0,0];b=np.array([0.])
    expected=conditional_call(mu,var,b,cfg,inner)
    return float(np.exp(-cfg.r*(tau-t))*(w@expected))


def boundary_probability(a,cfg,tau,epsilon=.01):
    x=factors(a,cfg);m,V=transition(x,cfg,0,tau)
    sd=np.sqrt(np.diag(V))
    return ndtr((ndtri(epsilon)-m)/sd)+ndtr((m-ndtri(1-epsilon))/sd)
