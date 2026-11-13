import unittest
import numpy as np
from factor_model import Model, factors, prices, transition, value


class FactorModelTests(unittest.TestCase):
    def test_factor_domain_not_price_box(self):
        cfg = Model()
        x = np.array([80., 80., 80., 80.])
        a = prices(x, cfg)
        self.assertGreater(a[0], cfg.pi)
        np.testing.assert_allclose(factors(a, cfg), x)

    def test_reject_negative_gap_without_clipping(self):
        with self.assertRaises(ValueError):
            factors([60., 80.], Model())

    def test_exact_martingale_mean(self):
        from scipy.special import ndtr
        cfg = Model()
        a = [60., 45., 30., 20.]
        x = factors(a, cfg)
        m, v = transition(x, cfg, 0., 3.)
        np.testing.assert_allclose(cfg.pi*ndtr(m/np.sqrt(1+np.diag(v))), x, atol=1e-10)

    def test_no_arbitrage(self):
        cfg = Model()
        for n in range(1,5):
            for a1 in (40.,60.,90.):
                a = [a1,45.,30.,20.][:n]
                c = value(a,cfg)
                self.assertGreaterEqual(c+1e-6, np.exp(-cfg.r*3)*max(a1-cfg.K,0))
                self.assertLessEqual(c, np.exp(-cfg.r*3)*a1+1e-6)

    def test_maturity_convex_order(self):
        cfg = Model()
        vals = [value([90.,45.],cfg,tau=s)*np.exp(cfg.r*s) for s in (.1,1.,2.,3.,3.8)]
        self.assertTrue(np.all(np.diff(vals)>-1e-3))

    def test_strike_monotonicity_convexity(self):
        from dataclasses import replace
        cfg = Model()
        vals = [value([60.,45.],replace(cfg,K=k)) for k in (40,50,60)]
        self.assertTrue(np.all(np.diff(vals)<0))
        self.assertGreaterEqual(vals[0]-2*vals[1]+vals[2],-1e-4)

    def test_quadrature_convergence(self):
        cfg = Model()
        c1 = value([60.,45.,30.,20.],cfg,order=24)
        c2 = value([60.,45.,30.,20.],cfg,order=40)
        self.assertLess(abs(c1-c2),.02)

    def test_tau_parameter_and_terminal(self):
        cfg = Model()
        self.assertEqual(value([60.,45.],cfg,t=3,tau=3),10.)
        with self.assertRaises(ValueError):
            value([60.],cfg,tau=4)


if __name__=='__main__': unittest.main()
