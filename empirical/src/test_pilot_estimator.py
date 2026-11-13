import unittest
import numpy as np
from scipy.integrate import quad

try:
    import pilot_estimator as pe
except ModuleNotFoundError:
    pe=None


class TransitionTests(unittest.TestCase):
    def require_implementation(self):
        self.assertIsNotNone(pe,'The pilot estimator is not implemented yet')

    def test_integrated_clock_matches_independent_quadrature(self):
        self.require_implementation()
        for alpha in [0.2,0.999999,1.0,1.000001,1.8,4.0]:
            actual=float(pe.integrated_clock(0.3,0.9,2.2,alpha,.7))
            expected=quad(lambda s:.7*(2.2-s)**(-alpha),.3,.9,epsabs=1e-12)[0]
            self.assertAlmostEqual(actual,expected,places=9)

    def test_transition_matches_independent_scipy_integrals(self):
        self.require_implementation()
        a,b=.2,.23;T=np.array([1.7,3.1]);alpha=1.35;beta=.8;rho=-.35
        M,g,V=pe.transition(np.array([a]),np.array([b]),T,alpha,beta,rho)
        z=lambda j,s:beta*(T[j]-s)**(-alpha)
        H=lambda j,s:quad(lambda u:z(j,u),s,b,epsabs=1e-12)[0]
        for j in range(2):
            self.assertAlmostEqual(M[0,j],np.exp(H(j,a)/2),places=10)
            oracle=quad(lambda s:np.exp(H(j,s)/2)*np.sqrt(z(j,s)),a,b,epsabs=1e-12)[0]
            self.assertAlmostEqual(g[0,j],oracle,places=10)
            for k in range(2):
                corr=1. if j==k else rho
                oracle=corr*quad(lambda s:np.exp((H(j,s)+H(k,s))/2)*np.sqrt(z(j,s)*z(k,s)),a,b,epsabs=1e-12)[0]
                self.assertAlmostEqual(V[0,j,k],oracle,places=10)

    def test_invalid_factor_is_rejected_not_clipped(self):
        self.require_implementation()
        with self.assertRaises(ValueError):pe.latent_states(np.array([[60.,65.]]),1.,np.array([121.01,128.71]))

    def test_period_specific_scales_match_independent_covariance(self):
        self.require_implementation()
        a,b=.2,.23;T=np.array([1.7,3.1]);alpha=1.35;beta=np.array([.4,1.2]);rho=.45
        M,g,V=pe.transition(np.array([a]),np.array([b]),T,alpha,beta,rho)
        z=lambda j,s:beta[j]*(T[j]-s)**(-alpha)
        H=lambda j,s:quad(lambda u:z(j,u),s,b,epsabs=1e-12)[0]
        oracle=rho*quad(lambda s:np.exp((H(0,s)+H(1,s))/2)*np.sqrt(z(0,s)*z(1,s)),a,b,epsabs=1e-12)[0]
        self.assertAlmostEqual(V[0,0,1],oracle,places=10)

    def test_profiled_drifts_satisfy_gls_normal_equations(self):
        self.require_implementation()
        rng=np.random.default_rng(42)
        prev=rng.normal(size=(12,2));a=np.arange(12)/365.;b=a+1/365.
        M,g,V=pe.transition(a,b,np.array([2.,3.]),1.2,.8,.3)
        nxt=M*prev+g*np.array([.2,-.1])+np.einsum('nij,nj->ni',np.linalg.cholesky(V),rng.normal(size=(12,2)))
        h=pe.profile_drifts(prev,nxt,M,g,V)
        residual=nxt-M*prev-g*h
        normal=np.sum(g*np.einsum('nij,nj->ni',np.linalg.inv(V),residual),axis=0)
        np.testing.assert_allclose(normal,0.,atol=1e-9)

    def test_price_jacobian_matches_finite_difference_determinant(self):
        self.require_implementation()
        A=np.array([[60.,45.]]);pi=np.array([121.01,128.71]);kappa=np.exp(-.03)
        xi,logj=pe.latent_states(A,kappa,pi)
        eps=1e-5;J=np.empty((2,2))
        for j in range(2):
            up=A.copy();down=A.copy();up[0,j]+=eps;down[0,j]-=eps
            J[:,j]=(pe.latent_states(up,kappa,pi)[0][0]-pe.latent_states(down,kappa,pi)[0][0])/(2*eps)
        self.assertAlmostEqual(float(logj[0]),float(np.log(abs(np.linalg.det(J)))),places=8)

if __name__=='__main__':unittest.main(verbosity=2)
