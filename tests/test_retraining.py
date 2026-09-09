"""CPU regressions; no ScanNet, downloaded weights or GPU required."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from models.STRUCTMAGNET import STRUCTMAGNET
from models.submodules.geometry_gate import cost_volume_statistics
from utils.losses import MagnetLoss
from utils.retraining import (configure_training, depth_loss, load_backbone,
                              perturb_rotations, restore_checkpoint, save_checkpoint)


class FakeD(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 256, 1), nn.BatchNorm2d(256))

    def forward(self, x):
        features = self.features(x)
        gaussian = torch.cat([torch.ones_like(x[:, :1]) * 2, torch.ones_like(x[:, :1])], 1)
        return gaussian, features


class FakeF(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.features = nn.Conv2d(3, 64, 1)

    def forward(self, x):
        return self.features(x)


def make_model(gate_mode='learned'):
    args = SimpleNamespace(DNET_ckpt='', FNET_ckpt='', MAGNET_sampling_range=3,
                           MAGNET_num_samples=3, MAGNET_mvs_weighting='CW5',
                           MAGNET_num_train_iter=2, MAGNET_num_test_iter=2,
                           dpv_height=4, dpv_width=4, downsample_ratio=1,
                           gate_mode=gate_mode)
    with patch('models.STRUCTMAGNET.DNET', FakeD), patch('models.STRUCTMAGNET.FNET', FakeF), \
         patch('models.STRUCTMAGNET.load_checkpoint', side_effect=lambda path, model: model), \
         patch.object(STRUCTMAGNET, 'depth_sampling', return_value=[-1., 0., 1.]):
        model = STRUCTMAGNET(args)
    return model


def forward(model, valid=True):
    # Exercise real homography warping with tiny feature maps and identity K.
    ref, sources = torch.randn(1, 3, 4, 4), torch.randn(2, 3, 4, 4)
    poses = torch.eye(4).repeat(1, 2, 1, 1)
    poses[..., 0, 3] = .1
    yy, xx = torch.meshgrid(torch.arange(4), torch.arange(4), indexing='ij')
    rays = torch.stack([xx + .5, yy + .5, torch.ones_like(xx)], 0).float().reshape(1, 3, -1)
    intrinsics = dict(intM=torch.eye(3).unsqueeze(0), unit_ray_array_2D=rays)
    return model(ref, sources, poses, torch.full((1, 2), float(valid)), intrinsics,
                 return_aux=True, rot_unc=None)


class RetrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_rotation_mixture_and_translation(self):
        poses = torch.eye(4).repeat(512, 4, 1, 1)
        poses[..., :3, 3] = torch.randn(512, 4, 3)
        noisy, angles = perturb_rotations(poses, generator=torch.Generator().manual_seed(3))
        torch.testing.assert_close(noisy[..., :3, 3], poses[..., :3, 3], rtol=0, atol=0)
        torch.testing.assert_close(noisy[..., 3, :], poses[..., 3, :], rtol=0, atol=0)
        rotation = noisy[..., :3, :3]
        torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation,
                                   torch.eye(3).expand_as(rotation), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(512, 4))
        actual = torch.rad2deg(torch.acos(((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))
        torch.testing.assert_close(actual[angles > .1], angles[angles > .1], atol=.02, rtol=.01)
        fraction = (angles == 0).all(1).float().mean().item()
        self.assertTrue(.4 < fraction < .6)
        self.assertLessEqual(angles.max(), 5.)
        clean, _ = perturb_rotations(poses, clean_probability=1.)
        torch.testing.assert_close(clean, poses, rtol=0, atol=0)

    def test_noise_generator_is_independent(self):
        poses = torch.eye(4).repeat(3, 2, 1, 1)
        one = perturb_rotations(poses, generator=torch.Generator().manual_seed(9))[0]
        torch.randn(1000)  # model initialization must not change augmentation
        two = perturb_rotations(poses, generator=torch.Generator().manual_seed(9))[0]
        torch.testing.assert_close(one, two, rtol=0, atol=0)
        fixed = perturb_rotations(poses, fixed_degrees=8., generator=torch.Generator().manual_seed(9))
        self.assertTrue((fixed[1] == 8).all())

    def test_depth_loss_matches_original_and_rejects_invalid_predictions(self):
        gt = torch.tensor([[[[2., 0., 3.]]]])
        predictions = [torch.tensor([[[[1., 1., 2.]], [[.4, 1., .7]]]], requires_grad=True),
                       torch.tensor([[[[1.5, 1., 2.5]], [[.5, 1., .8]]]], requires_grad=True)]
        original = MagnetLoss(SimpleNamespace(loss_fn='gaussian', loss_gamma=.8))
        expected = original(predictions, gt, gt > .001)
        actual = depth_loss(predictions, gt)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        for pred in predictions:
            self.assertGreater(pred.grad.abs().sum().item(), 0)
        bad = predictions[0].detach().clone()
        bad[0, 0, 0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            depth_loss([bad], gt)
        with self.assertRaises(ValueError):
            depth_loss(predictions, torch.zeros_like(gt))

    def test_depth_gradients_reach_all_trainable_modules(self):
        model = make_model()
        configure_training(model)
        self.assertFalse(model.d_net.training)
        self.assertFalse(model.f_net.training)
        predictions, aux = forward(model)
        depth_loss(predictions, torch.ones(1, 1, 4, 4) * 3).backward()
        for module in (model.g_net, model.mask_head, model.geometry_gate):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()
                                   if p.grad is not None), 0.)
        for module in (model.d_net, model.f_net):
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in module.parameters()))
        self.assertEqual(len(aux['depth_update']), 2)

    def test_gate_off_matches_raw_proposal_and_no_source_falls_back(self):
        model = make_model('off')
        configure_training(model)
        _, aux = forward(model)
        for raw, gated in zip(aux['ungated_gmm'], aux['gated_gmm_lowres']):
            torch.testing.assert_close(raw, gated)
        self.assertTrue(all(not p.requires_grad for p in model.geometry_gate.parameters()))
        _, aux = forward(model, valid=False)
        for gate, gated in zip(aux['geometry_gate'], aux['gated_gmm_lowres']):
            self.assertEqual(gate.count_nonzero().item(), 0)
            torch.testing.assert_close(gated, aux['mono_gmm'])

    def test_mixed_precision_statistics_are_finite(self):
        entropy, peak = cost_volume_statistics(torch.tensor([[[[65000.]], [[-65000.]]]], dtype=torch.float16))
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue(torch.isfinite(peak).all())

    def test_validation_uses_no_noise_labels_and_pixel_weighted_metrics(self):
        import train_StructMaGNet_scannet as training

        class PredictionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.poses = []

            def forward(self, ref, sources, poses, valid, intrinsics, **kwargs):
                assert kwargs['rot_unc'] is None
                self.poses.append(poses.clone())
                mu = torch.full((1, 1, 1, 2), 2.)
                return [torch.cat([mu, torch.ones_like(mu)], 1)], {
                    'geometry_gate': [torch.ones_like(mu) * .5],
                    'depth_update': [torch.ones_like(mu) * .1]}

        def batch(gt):
            return (None, None, torch.eye(4).repeat(1, 2, 1, 1), None, None,
                    torch.tensor(gt).reshape(1, 1, 1, 2))

        data = [batch([1., 0.]), batch([4., 4.])]
        cli = SimpleNamespace(val_degrees=[0., 5., 8.], seed=3, amp=False,
                              val_max_samples=0, min_depth=.001, max_depth=10.)
        model = PredictionModel()
        rng = torch.get_rng_state().clone()
        with patch.object(training, 'prepare_batch', side_effect=lambda b, device: b):
            first = training.validate(model, data, 'cpu', cli)
            second = training.validate(model, data, 'cpu', cli)
        self.assertEqual(first, second)
        self.assertEqual(first[0]['valid_pixels'], 3)
        self.assertAlmostEqual(first[0]['rmse'], 3 ** .5)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for i in range(6):
            torch.testing.assert_close(model.poses[i], model.poses[i + 6], rtol=0, atol=0)

    def test_checkpoint_restores_all_updated_modules_optimizer_and_rng(self):
        model = make_model()
        optimizer = torch.optim.AdamW(configure_training(model), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        prediction, _ = forward(model)
        depth_loss(prediction, torch.ones(1, 1, 4, 4) * 3).backward()
        optimizer.step()
        rng, loader_rng = torch.Generator().manual_seed(7), torch.Generator().manual_seed(8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'last.pt'
            save_checkpoint(path, model, optimizer, scaler, 2, 20, .4,
                            {'gate_mode': 'learned'}, rng, loader_rng)
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        p.add_(10)
            restored = restore_checkpoint(path, model, optimizer, scaler, rng, loader_rng)
            self.assertEqual(restored['epoch'], 2)
            self.assertEqual(restored['step'], 20)
            for k, v in model.state_dict().items():
                torch.testing.assert_close(v, expected[k])
            self.assertTrue(optimizer.state)
            self.assertTrue(torch.equal(rng.get_state(), torch.Generator().manual_seed(7).get_state()))
            model.gate_mode = 'off'
            with self.assertRaisesRegex(ValueError, 'gate_mode'):
                restore_checkpoint(path, model)
            torch.save({'model': {}}, path)
            with self.assertRaisesRegex(ValueError, 'missing compatible'):
                load_backbone(path, model)


if __name__ == '__main__':
    unittest.main()
