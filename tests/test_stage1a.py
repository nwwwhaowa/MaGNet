"""CPU regressions for Stage 1A; no data, pretrained weights, or network needed."""
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from utils.gate_oracle import first_gate_oracle, oracle_loss, OracleAccumulator
from models.submodules.geometry_gate import GeometryGate, cost_volume_statistics
from models.STRUCTMAGNET import STRUCTMAGNET, GNET, upsample_depth_via_mask
from data.dataloader_7scenes_train import balanced_validation_indices
import train_StructMaGNet as training


def make_aux(mono, mv, gate, support=None):
    def tensor(x):
        return torch.as_tensor(x, dtype=torch.float32).reshape(1, 1, 1, -1)
    m, v, g = tensor(mono), tensor(mv), tensor(gate).requires_grad_()
    out = {'mono_gmm': torch.cat([m, torch.ones_like(m)], 1),
           'ungated_gmm': [torch.cat([v, torch.ones_like(v)], 1)],
           'geometry_gate': [g]}
    if support is not None:
        out['geometry_valid'] = [tensor(support).bool()]
    return out


def oracle(aux, gt, min_delta=.01):
    gt = torch.as_tensor(gt, dtype=torch.float32).reshape(aux['geometry_gate'][0].shape)
    return first_gate_oracle(aux, gt, .001, 10., min_delta)


class OracleTests(unittest.TestCase):
    def test_interpolation_optimum_and_first_gate_gradient(self):
        aux = make_aux([2, 2, 2, 2], [4, 4, 4, 1], [.8]*4)
        aux['geometry_gate'].append(torch.full((1, 1, 1, 4), .5, requires_grad=True))
        o = oracle(aux, [1, 3, 5, 1.75])
        torch.testing.assert_close(o['target'].flatten(), torch.tensor([0., .5, 1., .25]))
        self.assertFalse(o['target'].requires_grad)
        oracle_loss(o)[0].backward()
        self.assertIsNotNone(aux['geometry_gate'][0].grad)
        self.assertIsNone(aux['geometry_gate'][1].grad)
        # Brute-force independent reference for the constrained interpolation.
        candidates = torch.linspace(0, 1, 1001)
        for j, expected in enumerate(o['target'].flatten()):
            err = (o['mono'].flatten()[j] + candidates *
                   (o['mv']-o['mono']).flatten()[j] - o['gt'].flatten()[j]).square()
            self.assertAlmostEqual(candidates[err.argmin()].item(), expected.item(), places=3)

    def test_promote_before_half_precision_subtraction(self):
        aux = make_aux([-60000, 1], [60000, 2], [.2, .8])
        aux['mono_gmm'] = aux['mono_gmm'].half()
        aux['ungated_gmm'][0] = aux['ungated_gmm'][0].half()
        o = oracle(aux, [2, 1.25])
        self.assertTrue(torch.isfinite(o['target']).all())
        self.assertAlmostEqual(o['target'].flatten()[0].item(), 60002/120000, places=6)
        self.assertAlmostEqual(o['target'].flatten()[1].item(), .25, places=6)

    def test_invalid_and_empty_support(self):
        aux = make_aux([1, 1, float('nan'), 1], [1, 2, 3, 2], [.5]*4,
                       [True, False, True, True])
        o = oracle(aux, [2, 2, 2, 0])
        self.assertFalse(o['valid'].any())
        loss, mean, std = oracle_loss(o)
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(math.isnan(mean.item()) and math.isnan(std.item()))
        loss.backward()
        torch.testing.assert_close(aux['geometry_gate'][0].grad, torch.zeros(1, 1, 1, 4))

    def test_nonfinite_gate_is_not_hidden(self):
        with self.assertRaisesRegex(RuntimeError, 'Non-finite GeometryGate'):
            oracle(make_aux([1], [2], [float('nan')]), [1.5])

    def test_pixel_weighted_statistics_ignore_empty_images(self):
        stats = OracleAccumulator()
        stats.update(oracle(make_aux([1], [2], [.5]), [0]))
        stats.update(oracle(make_aux([1], [2], [0]), [1]))
        stats.update(oracle(make_aux([1]*3, [2]*3, [1]*3), [2]*3))
        r = stats.compute()
        self.assertEqual(r['valid_pixels'], 4)
        self.assertEqual(r['empty_samples'], 1)
        self.assertAlmostEqual(r['valid_fraction'], .8)
        self.assertAlmostEqual(r['target_mean'], .75)
        self.assertAlmostEqual(r['corr'], 1.)
        self.assertEqual(r['oracle_rmse_low'], 0.)
        self.assertEqual(r['fused_rmse_low'], 0.)
        self.assertLess(r['oracle_rmse_low'], r['mono_rmse_low'])

    def test_zero_variance_correlation_is_undefined(self):
        stats = OracleAccumulator()
        stats.update(oracle(make_aux([1, 1], [2, 2], [.5, .5]), [1, 2]))
        self.assertTrue(math.isnan(stats.compute()['corr']))


class ToyD(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
    def forward(self, x):
        self.bn(x)
        n, _, h, w = x.shape
        return (torch.cat([x.new_full((n, 1, h, w), 2.),
                           x.new_full((n, 1, h, w), .5)], 1),
                x.new_ones(n, 256, h, w))


class ToyMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(9))
    def forward(self, x):
        return self.bias.view(1, 9, 1, 1).expand(x.shape[0], 9, *x.shape[-2:])


def tiny_model():
    # Exercise the real forward/update logic with cheap deterministic feature providers.
    m = STRUCTMAGNET.__new__(STRUCTMAGNET)
    nn.Module.__init__(m)
    m.d_net, m.f_net = ToyD(), nn.Conv2d(3, 2, 1)
    m.g_net, m.geometry_gate, m.mask_head = GNET(259), GeometryGate(), ToyMask()
    for p in m.g_net.parameters():
        nn.init.zeros_(p)
    m.g_net.gnet[-1].bias.data[0] = 1.
    m.train_iter = m.test_iter = 3
    m.k_list, m.weighting, m.downsample_ratio = [-1., 0., 1.], 'CW5', 1
    m.upsample_depth = upsample_depth_via_mask
    training.freeze_phase_a(m)
    training.set_phase_a_train_mode(m)
    return m


def toy_inputs(batch=1):
    return (torch.ones(batch, 3, 3, 4), torch.ones(batch, 3, 3, 4),
            torch.eye(4).view(1, 1, 4, 4).repeat(batch, 1, 1, 1),
            torch.ones(batch, 1, dtype=torch.int), {})


def toy_cost(depth, *args):
    return torch.ones_like(depth)


class ForwardTests(unittest.TestCase):
    def test_open_gate_preserves_tiny_positive_sigma(self):
        class OpenGate(nn.Module):
            def forward(self, x):
                return torch.ones_like(x[:, :1])
        m = tiny_model()
        m.geometry_gate = OpenGate()
        m.g_net.gnet[-1].bias.data[1] = -30.
        with patch('models.STRUCTMAGNET.homography.est_costvolume_CW', side_effect=toy_cost):
            pred, aux = m(*toy_inputs(), return_aux=True)
        self.assertTrue((pred[-1][:, 1:] > 0).all())
        torch.testing.assert_close(aux['gated_gmm_lowres'][0][:, 1:],
                                   aux['ungated_gmm'][0][:, 1:], rtol=0, atol=0)

    def test_first_gate_warmup_only_changes_gate(self):
        torch.manual_seed(4)
        m = tiny_model()
        before = {k: v.clone() for k, v in m.state_dict().items()}
        opt = torch.optim.AdamW(m.geometry_gate.parameters(), lr=.02)
        losses = []
        with patch('models.STRUCTMAGNET.homography.est_costvolume_CW', side_effect=toy_cost):
            for _ in range(8):
                opt.zero_grad()
                preds, aux = m(*toy_inputs(), return_aux=True)
                self.assertEqual(len(preds), 3)
                o = oracle(aux, torch.full((1, 1, 3, 4), 2.1))
                loss = oracle_loss(o)[0]
                losses.append(loss.item())
                loss.backward()
                opt.step()
        self.assertLess(losses[-1], losses[0])
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in m.state_dict().items()
                            if k.startswith('geometry_gate.')))
        for k, v in m.state_dict().items():
            if not k.startswith('geometry_gate.'):
                self.assertTrue(torch.equal(before[k], v), k)
        for name, p in m.named_parameters():
            if not name.startswith('geometry_gate.'):
                self.assertIsNone(p.grad, name)

    def test_no_valid_view_exactly_preserves_monocular_gaussian(self):
        m = tiny_model()
        inp = list(toy_inputs()); inp[3].zero_()
        with patch('models.STRUCTMAGNET.homography.est_costvolume_CW', side_effect=toy_cost):
            _, aux = m(*inp, return_aux=True)
        for gate, gmm, valid in zip(aux['geometry_gate'], aux['gated_gmm_lowres'],
                                    aux['geometry_valid']):
            self.assertEqual(gate.count_nonzero().item(), 0)
            self.assertFalse(valid.any())
            torch.testing.assert_close(gmm, aux['mono_gmm'], rtol=0, atol=0)

    def test_nonfinite_matching_falls_back_without_nan(self):
        m = tiny_model()
        with patch('models.STRUCTMAGNET.homography.est_costvolume_CW',
                   side_effect=lambda depth, *args: torch.full_like(depth, float('nan'))):
            pred, aux = m(*toy_inputs(), return_aux=True)
        self.assertTrue(torch.isfinite(pred[-1]).all())
        torch.testing.assert_close(aux['gated_gmm_lowres'][-1], aux['mono_gmm'])
        self.assertFalse(aux['geometry_valid'][0].any())

    def test_entropy_half_precision_is_finite_and_preserves_similarity_sign(self):
        volume = torch.tensor([1000., -1000., 0.], dtype=torch.float16).reshape(1, 3, 1, 1)
        entropy, peak = cost_volume_statistics(volume)
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertAlmostEqual(entropy.item(), 0.)
        self.assertAlmostEqual(peak.item(), 1.)
        e, p = cost_volume_statistics(torch.zeros(1, 5, 1, 1).half())
        self.assertAlmostEqual(e.item(), 1., places=6)
        self.assertAlmostEqual(p.item(), .2, places=6)
        e1, p1 = cost_volume_statistics(torch.zeros(1, 1, 1, 1))
        self.assertEqual(e1.item(), 0.)
        self.assertEqual(p1.item(), 1.)


class ProtocolTests(unittest.TestCase):
    def test_validation_counts_images_not_batches_and_ignores_empty_oracles(self):
        class ValidationModel(nn.Module):
            def forward(self, ref, src, poses, valid, intrinsics, **kwargs):
                b = ref.shape[0]
                # First sample has an unidentifiable gate (mono == mv).
                mono = ref.new_ones(b, 1, 1, 1)
                mv = 1 + ref[:, :1, :1, :1]
                aux = {'mono_gmm': torch.cat([mono, mono], 1),
                       'ungated_gmm': [torch.cat([mv, mono], 1)],
                       'geometry_gate': [mono*.25]*3}
                return [torch.cat([mono, mono], 1)], aux

        def frame(values, depths):
            b = len(values)
            return {'img': torch.tensor(values).float().view(b, 1, 1, 1).expand(b, 3, 1, 1),
                    'gt_dmap': torch.tensor(depths).float().view(b, 1, 1, 1),
                    'extM': torch.eye(4).unsqueeze(0).repeat(b, 1, 1)}
        frames = [frame([0, 1], [1, 3]) for _ in range(3)]
        args = SimpleNamespace(min_depth=.001, max_depth=10., gate_min_delta=.01)
        result = training.validate(ValidationModel(), [(frames, {})], torch.device('cpu'),
                                   args, fixed_deg=0., max_samples=0)
        self.assertEqual(result['samples'], 2)
        self.assertAlmostEqual(result['rmse'], 1.)  # mean(0, 2), not sqrt(mean(0, 4))
        self.assertEqual(result['oracle_empty_samples'], 1)
        self.assertEqual(result['oracle_valid_pixels'], 1)
        self.assertAlmostEqual(result['gate_target_mean'], 1.)
        result = training.validate(ValidationModel(), [(frames, {})], torch.device('cpu'),
                                   args, fixed_deg=0., max_samples=1)
        self.assertEqual(result['samples'], 1)
        self.assertEqual(result['rmse'], 0.)
        self.assertTrue(math.isnan(result['gate_target_mean']))

    def test_rotation_noise_preserves_translation_and_so3(self):
        torch.manual_seed(123)
        poses = torch.eye(4).view(1, 1, 4, 4).repeat(3, 4, 1, 1)
        poses[..., :3, 3] = torch.randn(3, 4, 3)
        before = poses.clone()
        for noise in (lambda x: training.sample_train_rotation_noise(x),
                      lambda x: training.fixed_validation_rotation_noise(x, 5.)):
            out, _, _ = noise(poses)
            torch.testing.assert_close(out[..., :3, 3], before[..., :3, 3], rtol=0, atol=0)
            r = out[..., :3, :3]
            torch.testing.assert_close(r.transpose(-1, -2) @ r, before[..., :3, :3], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(torch.linalg.det(r), torch.ones(3, 4))
        torch.testing.assert_close(poses, before, rtol=0, atol=0)

    def test_short_validation_covers_both_scenes(self):
        samples = [(s, 1, i) for s in ('chess', 'office') for i in range(100)]
        selected = balanced_validation_indices(samples, 32)
        self.assertEqual(len(set(selected)), 32)
        self.assertEqual(sum(samples[i][0] == 'office' for i in selected), 16)
        self.assertIn(0, selected); self.assertIn(199, selected)
        self.assertEqual(selected, balanced_validation_indices(samples, 32))
        with self.assertRaises(ValueError):
            balanced_validation_indices(samples, 1)
        self.assertEqual(len(balanced_validation_indices(samples, 0)), 200)

    def test_incomplete_checkpoint_fails_before_loading(self):
        m = tiny_model()
        before = {k: v.clone() for k, v in m.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'bad.pt'
            torch.save({'model': {'g_net.gnet.0.weight': torch.ones(1)}}, p)
            with self.assertRaisesRegex(RuntimeError, 'Incomplete MaGNet'):
                training.load_compatible_backbone(p, m)
        for k, v in m.state_dict().items():
            self.assertTrue(torch.equal(before[k], v))

    def test_checkpoint_best_score_roundtrip_and_gate_only(self):
        m = tiny_model()
        opt = torch.optim.AdamW(m.geometry_gate.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'last.pt'
            training.save_gate_checkpoint(p, m, opt, 3, 24, {'score': 9.},
                                          SimpleNamespace(seed=1), best_score=2.)
            ckpt = training.load_gate_checkpoint(p, m, opt)
        self.assertEqual(ckpt['best_score'], 2.)
        self.assertEqual(ckpt['metrics']['score'], 9.)
        self.assertTrue(all(k.startswith('geometry_gate.') for k in ckpt['model']))
        self.assertEqual(training.resume_best_score(ckpt, SimpleNamespace(seed=1)), 2.)
        self.assertIsNone(training.resume_best_score(ckpt, SimpleNamespace(val_max_samples=100)))
        legacy = {k: v for k, v in ckpt.items() if k != 'validation_protocol'}
        self.assertIsNone(training.resume_best_score(legacy, SimpleNamespace(seed=1)))

    def test_csv_rejects_mixed_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'train.csv'
            training.append_csv(p, ['step'], {'step': 1})
            with self.assertRaisesRegex(ValueError, 'schema mismatch'):
                training.append_csv(p, ['epoch'], {'epoch': 1})


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
