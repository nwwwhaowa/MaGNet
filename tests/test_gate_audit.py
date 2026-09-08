"""Controlled tests for attribution, masking, losses and full-forward policies."""
import csv
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from test_stage1a import tiny_model, toy_inputs, toy_cost, make_aux, oracle
from models.submodules.geometry_gate import GeometryGate
from utils.gate_oracle import oracle_loss
from utils.gate_audit import apply_gate_policy, FirstStepAudit, run_gate_audit
import train_StructMaGNet as training


class GateAuditTests(unittest.TestCase):
    def test_policies_preserve_support_distribution_and_rng(self):
        g = torch.tensor([.1, .2, .9, .8, .3]).view(1, 1, 1, 5)
        valid = torch.tensor([1, 1, 0, 1, 1]).bool().view_as(g)
        mean = apply_gate_policy(g, valid, 'image_mean')
        torch.testing.assert_close(mean[valid], torch.full((4,), .35))
        self.assertEqual(mean[~valid].item(), 0.)
        state = torch.get_rng_state().clone()
        shuffled = apply_gate_policy(g, valid, 'shuffle')
        torch.testing.assert_close(shuffled[valid].sort().values, g[valid].sort().values)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        torch.testing.assert_close(shuffled, apply_gate_policy(g, valid, 'shuffle'))
        for p in ('image_mean', 'shuffle', 'fixed'):
            self.assertEqual(apply_gate_policy(g, valid & False, p).sum().item(), 0.)
        with self.assertRaises(ValueError):
            apply_gate_policy(g, valid, 'fixed', 1.1)

    def test_init_bias_compatibility_and_gradients(self):
        for bias in (4., 0.):
            gate = GeometryGate(init_bias=bias)
            x = torch.ones(1, 4, 3, 3)
            y = gate(x)
            torch.testing.assert_close(y, torch.full_like(y, torch.sigmoid(torch.tensor(bias)).item()))
            (y-.2).square().mean().backward()
            self.assertGreater(gate.net[-1].bias.grad.abs().item(), 0.)
            self.assertEqual(gate.net[0].weight.grad.abs().sum().item(), 0.)
        old, new = GeometryGate(), GeometryGate(init_bias=0.)
        new.load_state_dict(old.state_dict(), strict=True)
        torch.testing.assert_close(old(x), new(x))

    def test_loss_gradient_matches_independent_reference(self):
        for kind in ('smooth_l1', 'mse', 'depth_mse'):
            o = oracle(make_aux([1, 1], [2, 3], [.3, .7]), [1.2, 1.4])
            loss = oracle_loss(o, kind)[0]
            loss.backward()
            g = torch.tensor([.3, .7], requires_grad=True)
            if kind == 'smooth_l1':
                expected = torch.nn.functional.smooth_l1_loss(g, torch.tensor([.2, .2]), beta=.1)
            elif kind == 'mse':
                expected = ((g-torch.tensor([.2, .2]))**2).mean()
            else:
                expected = ((1+g*torch.tensor([1., 2.])-torch.tensor([1.2, 1.4]))**2).mean()
            expected.backward()
            torch.testing.assert_close(loss, expected)
            torch.testing.assert_close(o['gate'].grad.flatten(), g.grad)
            empty = oracle(make_aux([1], [1], [.5]), [1])
            self.assertEqual(oracle_loss(empty, kind)[0].item(), 0.)

    def test_perfect_pixel_selection_beats_equal_mean_and_fixed_controls(self):
        gate = GeometryGate(init_bias=0.)
        o = oracle(make_aux([1, 1, 1, 1], [2, 2, 2, 2], [0, 1, 0, 1]), [1, 2, 1, 2])
        inp = torch.zeros(1, 4, 1, 4)
        inp[:, 0] = o['target'][:, 0]
        aux = {'geometry_valid': [torch.ones_like(o['valid'])], 'gate_input': [inp]}
        audit = FirstStepAudit(gate, 0.)
        with torch.no_grad():
            audit(o, aux, 0)
        rows = {r['policy']: r for r in audit.rows()}
        self.assertEqual(rows['learned']['fused_rmse_low'], 0.)
        self.assertEqual(rows['image_mean']['fused_rmse_low'], .5)
        self.assertEqual(rows['fixed_0.5']['fused_rmse_low'], .5)
        self.assertEqual(rows['fixed_1']['valid_pixels'], 4)
        features = {r['feature']: r for r in audit.features.rows(0.)}
        self.assertAlmostEqual(features['entropy']['corr_oracle_target'], 1.)
        self.assertTrue(math.isnan(features['rotation_radians']['corr_oracle_target']))

    def test_real_forward_fixed_zero_and_one(self):
        m = tiny_model().eval()
        m.gate_policy, m.gate_fixed_value = 'fixed', 0.
        with patch('models.STRUCTMAGNET.homography.est_costvolume_CW', toy_cost):
            _, aux = m(*toy_inputs(), mode='test', return_aux=True)
            torch.testing.assert_close(aux['gated_gmm_lowres'][-1], aux['mono_gmm'])
            self.assertEqual(len(aux['gate_input']), 3)
            m.gate_fixed_value = 1.
            _, aux = m(*toy_inputs(), mode='test', return_aux=True)
            torch.testing.assert_close(aux['gated_gmm_lowres'][-1], aux['ungated_gmm'][-1])
            m.train()
            with self.assertRaisesRegex(RuntimeError, 'evaluation-only'):
                m(*toy_inputs(), return_aux=True)

    def test_audit_end_to_end_with_real_validation_and_toy_backbone(self):
        m = tiny_model().eval()
        frame = {'img': torch.ones(1, 3, 3, 4), 'gt_dmap': torch.full((1, 1, 3, 4), 2.2),
                 'extM': torch.eye(4).unsqueeze(0)}
        loader = [([frame]*3, {})]
        args = SimpleNamespace(min_depth=.001, max_depth=10., gate_min_delta=.01)
        with tempfile.TemporaryDirectory() as tmp:
            cli = SimpleNamespace(output_dir=tmp, val_max_samples=1, gate_audit_full=True)
            with patch('models.STRUCTMAGNET.homography.est_costvolume_CW', toy_cost):
                run_gate_audit(m, loader, torch.device('cpu'), args, cli, training.validate)
            with (Path(tmp)/'logs/gate_audit_full.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 24)
            self.assertTrue(all(int(r['samples']) == 1 for r in rows))
            with (Path(tmp)/'logs/gate_audit_first_step.csv').open() as f:
                first = list(csv.DictReader(f))
            self.assertEqual(len(first), 39)
            self.assertEqual(m.gate_policy, 'learned')


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
