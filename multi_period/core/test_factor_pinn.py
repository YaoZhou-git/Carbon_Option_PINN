import unittest
import numpy as np
import torch
from factor_model import Model,matrix
from factor_pinn import FactorPINN,residual


class PINNTests(unittest.TestCase):
    def test_terminal_at_strike_is_exactly_zero(self):
        net=FactorPINN(Model(),1)
        self.assertEqual(net(torch.tensor([[.5,3.]])).item(),0.)

    def test_exact_terminal_condition(self):
        cfg=Model();net=FactorPINN(cfg,4)
        y=torch.rand(64,4)*.98+.01
        t=torch.full((64,1),cfg.tau)
        actual=net(torch.cat([y,t],1)).detach().numpy().ravel()
        expected=np.maximum(y.numpy()@matrix(cfg,4)[0]-.5,0)
        np.testing.assert_allclose(actual,expected,atol=2e-5)

    def test_full_box_contains_prices_above_pi(self):
        cfg=Model();net=FactorPINN(cfg,4)
        a=(torch.ones(1,4)*.8)@net.weights
        self.assertGreater(a.item(),1)

    def test_finite_full_covariance_residual(self):
        cfg=Model();net=FactorPINN(cfg,4)
        state=torch.cat([torch.rand(48,4)*.96+.02,torch.rand(48,1)*2.9],1)
        r=residual(net,state)
        self.assertTrue(torch.isfinite(r).all())
        r.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))


if __name__=='__main__':unittest.main()
