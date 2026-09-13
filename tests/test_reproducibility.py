"""Fast scientific and packaging regressions (no training and no plotting)."""
from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'scripts'), str(ROOT)]
import numpy as np
import torch
import experiments as ex
from audit_light_revision import independent_equation_tests
from export_tables import export_tables
from repro_support import compare_results, export_metrics, verify_inputs, write_json
from reproduce import fresh_output

ARCHIVE = json.loads((ROOT / 'results/all_results.json').read_text())


def load_narx(name: str = 'main', *, double: bool = False):
    weights = torch.load(ROOT / 'results/narx_models.pt', weights_only=True, map_location='cpu')
    sd = weights['variant:' + name]
    model = ex.AlgebraicImplicitMap(10, 2, 48, sd['D_u'], sd['D_chi'], kappa_i=.68,
                                    input_features='B_u=0' not in name,
                                    kappa_f=.35 if 'A!=0' in name else 0.)
    model.load_state_dict(sd, strict=True)
    model.eval()
    if double:
        model.double()
        model.D_u_inv.copy_(torch.linalg.inv(model.D_u))
    return model


class ScientificTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_narx(double=True)
        gen = torch.Generator().manual_seed(421)
        cls.chi = torch.randn((24, 10), generator=gen, dtype=torch.float64) * .2
        cls.target = torch.randn((24, 2), generator=gen, dtype=torch.float64) * .2

    def test_reference_manifest(self):
        self.assertEqual(verify_inputs(ROOT)['status'], 'passed')

    def test_forward_inverse_cycle_double(self):
        u = self.model.inverse(self.chi, self.target, tol=1e-13)
        with torch.no_grad():
            error = torch.max(torch.abs(self.model(self.chi, u) - self.target)).item()
        self.assertLess(error, 1e-11)

    def test_cyclic_forward_inverse_cycle_double(self):
        model = load_narx('implicit forward (A!=0)', double=True)
        u = model.inverse(self.chi, self.target, tol=1e-13)
        # Use a tight forward solve for this double-precision identity test.
        hidden = model.hidden_forward(self.chi, u, tol=1e-13)
        with torch.no_grad():
            predicted = hidden @ model.effective_C().T + self.chi @ model.D_chi.T + u @ model.D_u.T + model.c
        self.assertLess(torch.max(torch.abs(predicted - self.target)).item(), 1e-11)

    def test_bi_contraction(self):
        for name in ARCHIVE['narx']['variants']:
            with self.subTest(name=name):
                sens = load_narx(name).sensitivity_matrices()
                self.assertLess(sens['rho_absA'], 1.)
                self.assertLess(sens['rho_absAi'], 1.)
                self.assertGreaterEqual(np.min(sens['S_chi']), -1e-14)

    def test_inverse_sensitivity(self):
        a, b = self.chi + .031, self.target - .027
        u1 = self.model.inverse(self.chi, self.target, tol=1e-13)
        u2 = self.model.inverse(a, b, tol=1e-13)
        sens = self.model.sensitivity_matrices()
        bound = np.abs((a-self.chi).numpy()) @ sens['S_chi'].T + np.abs((b-self.target).numpy()) @ sens['S_y'].T
        self.assertLess(np.max(np.abs((u2-u1).numpy()) - bound), 1e-11)

    def test_solver_residual_control_and_output_bounds(self):
        model = self.model
        u, info = model.inverse(self.chi, self.target, tol=1e-13, return_info=True)
        gen = torch.Generator().manual_seed(729)
        hidden = info['h'] + .04 * torch.randn(info['h'].shape, generator=gen, dtype=torch.float64)
        ybar, rhs = model._inverse_affine(self.chi, self.target)
        with torch.no_grad():
            approximate_u = (ybar - hidden @ model.effective_C().T) @ model.D_u_inv.T
            residual = torch.abs(hidden - torch.tanh(hidden @ model.inverse_A().T + rhs)).numpy()
            output_error = torch.abs(model(self.chi, approximate_u) - self.target).numpy()
        sens = model.sensitivity_matrices()
        self.assertLess(np.max(np.abs((approximate_u-u).numpy()) - residual @ sens['K_u'].T), 1e-11)
        self.assertLess(np.max(output_error - residual @ sens['K_y'].T), 1e-11)

    def test_main_certificate_and_tubes(self):
        n = ARCHIVE['narx']
        cert = ex.certificate(load_narx().sensitivity_matrices(), np.asarray(n['F']),
                              np.asarray(n['E_chi']), np.asarray(n['E_u']), n['d_bar_total'],
                              np.full(2, .003), np.full(48, 2e-7))
        self.assertLess(cert['rho_H'], 1.)
        np.testing.assert_allclose(cert['H'], n['H'], atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(cert['y_tube'], n['output_tube'], atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(cert['u_tube'], n['input_deviation_tube'], atol=1e-8, rtol=1e-5)

    def test_uncertified_ablation_is_reported(self):
        n = ARCHIVE['narx']['variants']['control-affine (B_u=0)']
        self.assertGreater(n['rho_H'], 1.)
        self.assertTrue(all(not np.isfinite(x) for x in n['u_tube']))

    def test_shift_register(self):
        J, Gy, Gu, Py = ex.narx_shift_matrices()
        q = torch.arange(10, dtype=ex.DTYPE).reshape(1, -1)
        y = torch.tensor([[11., 12.]])
        u = torch.tensor([[13., 14.]])
        shifted = ex.narx_shift(q, y, u).numpy()
        np.testing.assert_allclose(shifted, q.numpy() @ J.T + y.numpy() @ Gy.T + u.numpy() @ Gu.T)
        np.testing.assert_allclose(q.numpy() @ Py.T, q.numpy()[:, :2])

    def test_paired_bootstrap_deterministic(self):
        a = np.array([1., 2., 3., 4.])
        result = ex.paired_bootstrap_ci(a, a+.5, nboot=100, seed=3)
        self.assertEqual(result['mean_difference'], -.5)
        self.assertEqual(result['ci95_low'], -.5)
        self.assertEqual(result['ci95_high'], -.5)

    def test_independent_numpy_equations(self):
        result = independent_equation_tests()
        self.assertEqual(result['dense_cyclic_models'], 36)
        self.assertLess(result['maximum_forward_inverse_cycle_error'], 1e-11)
        self.assertLess(max(result['maximum_inequality_defects'].values()), 1e-11)


class PackagingTests(unittest.TestCase):
    def test_csv_quotes_variant_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'metrics.csv'
            export_metrics({'narx': {'name, with comma': {'value': 3.}}}, path)
            with path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.reader(stream))
            self.assertTrue(all(len(row) == 3 for row in rows))
            self.assertEqual(rows[1][1], 'name, with comma/value')

    def test_nonfinite_to_json_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.json'
            write_json(path, {'unavailable': float('nan')})
            self.assertEqual(json.loads(path.read_text()), {'unavailable': None})
            self.assertNotIn('NaN', path.read_text())

    def test_scientific_difference_fails_comparison(self):
        a = {'narx': {'rmse_mean': .005, 'controller_time': .001}}
        b = {'narx': {'rmse_mean': .01, 'controller_time': .002}}
        self.assertEqual(compare_results(a, b)['scientific_values_failed'], 1)

    def test_timing_differences_are_diagnostic(self):
        a = {'narx': {'rmse_mean': .005, 'controller_time': .001}}
        b = {'narx': {'rmse_mean': .005, 'controller_time': .2}}
        r = compare_results(a, b)
        self.assertEqual(r['status'], 'passed')
        self.assertEqual(r['diagnostic_values_excluded_from_gate'], 1)

    def test_missing_scientific_values_fail(self):
        result = compare_results({'static': {'a': 1., 'b': 2.}}, {'static': {'a': 1.}})
        self.assertEqual(result['status'], 'differences_detected')
        self.assertEqual(result['missing_reference_fields'], ['static/b'])

    def test_nonfinite_expected_values_match_null(self):
        result = compare_results({'narx': {'u_tube': [float('nan')]}}, {'narx': {'u_tube': [None]}})
        self.assertEqual(result['status'], 'passed')

    def test_output_overwrite_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                fresh_output(tmp, 'test')

    def test_all_four_tables_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            export_tables(ARCHIVE, folder)
            self.assertEqual(len(list(folder.glob('table_*.csv'))), 4)
            self.assertEqual(len(list(folder.glob('table_*.tex'))), 4)
            text = (folder / 'TABLES.md').read_text()
            for label in ['Table I.', 'Table II.', 'Table III.', 'Table IV.']:
                self.assertIn(label, text)
            self.assertNotIn('NaN', (folder / 'results_portable.json').read_text())

    def test_saved_main_and_variant_weights_agree(self):
        ck = torch.load(ROOT / 'results/narx_models.pt', weights_only=True, map_location='cpu')
        self.assertEqual(set(ck['model']), set(ck['variant:main']))
        self.assertTrue(all(torch.equal(ck['model'][k], ck['variant:main'][k]) for k in ck['model']))


if __name__ == '__main__':
    unittest.main()
