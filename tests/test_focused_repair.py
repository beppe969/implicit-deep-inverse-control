"""Regressions for fixed plant centers and per-query certified stopping."""
from __future__ import annotations
import sys
from pathlib import Path
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import experiments as ex


def load_main():
    ck = torch.load(ROOT / 'results/narx_models.pt', weights_only=True, map_location='cpu')
    sd = ck['variant:main']
    model = ex.AlgebraicImplicitMap(10, 2, 48, sd['D_u'], sd['D_chi'], kappa_i=.68)
    model.load_state_dict(sd)
    model.eval()
    return model


class StoppingTests(unittest.TestCase):
    def setUp(self):
        # One-dimensional arithmetic isolates stopping from BLAS roundoff.
        self.model = ex.AlgebraicImplicitMap(1, 1, 1, [[1.]], [[.2]], kappa_i=.9).double()
        with torch.no_grad():
            self.model.B_u.fill_(-1.)
            self.model.B_chi.fill_(.5)
            self.model.C_raw.fill_(1.8)
            self.model.b.zero_()
            self.model.c.zero_()
        self.model.eval()
        self.chi = torch.tensor([[.2], [.4], [-.3], [.01]], dtype=torch.float64)
        self.y = torch.tensor([[.15], [-.05], [.8], [.3]], dtype=torch.float64)

    def test_individual_batch_permutation_and_chunking(self):
        # A loose tolerance exposes the original extra-updates dependence.
        for tol in (.01, 1e-12):
            u, info = self.model.inverse(self.chi, self.y, tol=tol, return_info=True)
            single = [self.model.inverse(self.chi[j:j+1], self.y[j:j+1], tol=tol,
                                         return_info=True) for j in range(len(self.y))]
            self.assertTrue(torch.equal(u, torch.cat([x[0] for x in single])))
            self.assertTrue(torch.equal(info['h'], torch.cat([x[1]['h'] for x in single])))
            self.assertTrue(torch.equal(info['iterations'], torch.cat([x[1]['iterations'] for x in single])))
            perm = torch.tensor([3, 1, 0, 2])
            up, ip = self.model.inverse(self.chi[perm], self.y[perm], tol=tol, return_info=True)
            self.assertTrue(torch.equal(u[perm], up))
            self.assertTrue(torch.equal(info['h'][perm], ip['h']))
            chunks = [self.model.inverse(self.chi[j:j+2], self.y[j:j+2], tol=tol) for j in (0, 2)]
            self.assertTrue(torch.equal(u, torch.cat(chunks)))
            self.assertTrue(torch.all(info['residual'] <= tol))

    def test_returned_residual_and_first_crossing(self):
        tol = .01
        _, info = self.model.inverse(self.chi, self.y, tol=tol, return_info=True)
        _, rhs = self.model._inverse_affine(self.chi, self.y)
        h = torch.zeros_like(info['h'])
        for k in range(1, int(info['iterations'].max())+1):
            hn = torch.tanh(h @ self.model.inverse_A().T + rhs)
            r = torch.max(torch.abs(hn-h), dim=1).values
            mask = info['iterations'] == k
            self.assertTrue(torch.equal(h[mask], info['h'][mask]))
            self.assertTrue(torch.all(r[mask] <= tol))
            self.assertTrue(torch.all(r[info['iterations'] > k] > tol))
            h = hn
        self.assertTrue(torch.all(info['converged']))

    def test_cap_exposed_and_uncertified_default_rejected(self):
        with self.assertRaises(RuntimeError):
            self.model.inverse(self.chi, self.y, tol=1e-15, max_iter=1)
        _, info = self.model.inverse(self.chi, self.y, tol=1e-15, max_iter=1, return_info=True)
        self.assertFalse(torch.all(info['converged']))
        self.assertTrue(torch.equal(info['h'], torch.zeros_like(info['h'])))
        self.assertTrue(torch.all(info['iterations'] == 1))

    def test_actual_model_batches_agree_to_roundoff(self):
        model = load_main()
        gen = torch.Generator().manual_seed(20261013)
        chi = .4 * torch.randn(31, 10, generator=gen)
        y = .3 * torch.randn(31, 2, generator=gen)
        u, info = model.inverse(chi, y, tol=2e-7, return_info=True)
        pairs = [model.inverse(chi[j:j+1], y[j:j+1], tol=2e-7, return_info=True)
                 for j in range(len(y))]
        us = torch.cat([p[0] for p in pairs])
        # Batch-size-dependent BLAS rounding is distinct from extra solver updates.
        torch.testing.assert_close(u, us, rtol=0, atol=2e-7)
        self.assertTrue(torch.all(info['converged']))
        self.assertTrue(torch.all(info['residual'] <= 2e-7))


class PhysicalPlantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_main()
        cls.ref = ex.build_reference(cls.model, 15)
        cls.F = .1*np.eye(2)
        cls.Ec = np.full((2, 10), .005)
        cls.Eu = np.full((2, 2), .005)

    def test_certificate_reference_does_not_change_physical_plant(self):
        yr, cr, ur, _ = self.ref
        args = ('implicit inverse', self.model, {}, yr)
        tail = (self.F, self.Ec, self.Eu, np.full(2,.004), np.full(2,.003))
        kw = dict(nruns=5, seed=39125, plant_chi_ref=cr, plant_u_ref=ur)
        _, a = ex.closed_loop_narx(*args, cr, ur, *tail, **kw)
        _, b = ex.closed_loop_narx(*args, cr+.1, ur-.2, *tail, **kw)
        for key in ['y', 'ym', 'u', 'chi']:
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_allclose(b['u_dev']-a['u_dev'], .2, atol=1e-15)

    def test_center_allowance_dominates_recentering_difference(self):
        _, cr, ur, _ = self.ref
        rng = np.random.default_rng(7319)
        ca = cr + .2*rng.normal(size=cr.shape)
        ua = ur + .2*rng.normal(size=ur.shape)
        allowance = ex.uncertainty_center_allowance(ca,ua,cr,ur,self.Ec,self.Eu)
        expected = np.max(np.abs(ca[:-1]-cr[:-1])@self.Ec.T + np.abs(ua-ur)@self.Eu.T,axis=0)
        np.testing.assert_array_equal(allowance,expected)
        np.testing.assert_array_equal(ex.uncertainty_center_allowance(cr,ur,cr,ur,self.Ec,self.Eu),np.zeros(2))
        q = rng.normal(size=ca[:-1].shape)
        u = rng.normal(size=ua.shape)
        lhs=np.abs(q-cr[:-1])@self.Ec.T + np.abs(u-ur)@self.Eu.T
        rhs=np.abs(q-ca[:-1])@self.Ec.T + np.abs(u-ua)@self.Eu.T + allowance
        self.assertLessEqual(float(np.max(lhs-rhs)),1e-14)

    def test_partial_center_rejected(self):
        yr,cr,ur,_ = self.ref
        with self.assertRaises(ValueError):
            ex.closed_loop_narx('implicit inverse',self.model,{},yr,cr,ur,self.F,self.Ec,self.Eu,
                               np.full(2,.004),np.full(2,.003),nruns=2,plant_chi_ref=cr)


if __name__ == '__main__':
    unittest.main()
