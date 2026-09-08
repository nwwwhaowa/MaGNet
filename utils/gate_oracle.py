"""First-step interpolation oracle shared by Stage 1A training and evaluation.

This is a supervised low-resolution diagnostic, not a deployable gate input.
Ground-truth depth is used only here, never by STRUCTMAGNET.forward.
"""

import math

import torch
import torch.nn.functional as F


def first_gate_oracle(aux, gt_depth, min_depth, max_depth, min_delta=0.01):
    """Return gate, detached target, mask, and endpoints in FP32.

    The exact constrained least-squares solution is clamp((gt-mono)/delta).
    Masking |delta| <= min_delta makes division safe without adding a ridge
    term that biases the coefficient. All arithmetic is promoted BEFORE the
    subtraction/division (casting an already-overflowed FP16 target is too late).
    """
    if not (0 < min_depth < max_depth) or not math.isfinite(max_depth):
        raise ValueError('Expected finite 0 < min_depth < max_depth')
    if not math.isfinite(min_delta) or min_delta <= 0:
        raise ValueError('gate_min_delta must be finite and positive')
    gates, proposals = aux.get('geometry_gate'), aux.get('ungated_gmm')
    if not isinstance(gates, (list, tuple)) or not gates:
        raise RuntimeError('aux[geometry_gate] must be a non-empty list')
    if not isinstance(proposals, (list, tuple)) or not proposals:
        raise RuntimeError('aux[ungated_gmm] must be a non-empty list')
    if aux.get('mono_gmm') is None:
        raise RuntimeError('aux[mono_gmm] is required')

    gate = gates[0].float()
    with torch.no_grad():
        mono = aux['mono_gmm'][:, :1].detach().float()
        mv = proposals[0][:, :1].detach().float()
        gt = F.interpolate(gt_depth.detach().float(), size=gate.shape[-2:],
                           mode='nearest')
        if gate.shape != mono.shape or gate.shape != mv.shape or gate.shape != gt.shape:
            raise ValueError('Oracle gate, endpoints, and resized GT must have the same shape')
        delta = mv - mono
        valid = ((gt > min_depth) & (gt < max_depth) & torch.isfinite(gt)
                 & torch.isfinite(mono) & torch.isfinite(mv)
                 & torch.isfinite(delta) & (delta.abs() > min_delta))
        support = aux.get('geometry_valid')
        if support is not None:
            if not isinstance(support, (list, tuple)) or not support:
                raise RuntimeError('aux[geometry_valid] must be a non-empty list')
            if support[0].shape != gate.shape:
                raise ValueError('geometry_valid must match the gate shape')
            valid = valid & support[0].bool()
        # Invalid pixels never participate in arithmetic or loss reductions.
        target = torch.zeros_like(mono)
        target[valid] = ((gt[valid] - mono[valid]) / delta[valid]).clamp(0, 1)
    if not torch.isfinite(gate).all():
        raise RuntimeError('Non-finite GeometryGate output; refusing to hide it in the oracle loss')
    if ((gate < 0) | (gate > 1)).any():
        raise RuntimeError('GeometryGate output must lie in [0, 1]')
    return {'gate': gate, 'target': target, 'valid': valid,
            'mono': mono, 'mv': mv, 'gt': gt}


def oracle_loss(oracle, kind='smooth_l1'):
    """Smooth-L1 on identifiable pixels; empty masks return graph-connected zero.

    The caller must skip the optimizer step on an empty mask, because AdamW
    can change parameters via momentum/weight decay even with a zero gradient.
    """
    if kind not in ('smooth_l1', 'mse', 'depth_mse'):
        raise ValueError('Unknown gate loss: ' + kind)
    gate, target, valid = (oracle[k] for k in ('gate', 'target', 'valid'))
    if not valid.any():
        nan = gate.new_tensor(float('nan'))
        return gate.sum() * 0.0, nan, nan
    target_v = target[valid]
    if kind == 'smooth_l1':
        loss = F.smooth_l1_loss(gate[valid], target_v, beta=0.1)
    elif kind == 'mse':
        loss = F.mse_loss(gate[valid], target_v)
    else:
        fused = oracle['mono'][valid] + gate[valid] * (
            oracle['mv'][valid] - oracle['mono'][valid])
        loss = F.mse_loss(fused, oracle['gt'][valid])
    return loss, target_v.mean(), target_v.std(unbiased=False)


class OracleAccumulator:
    """Pixel-weighted streaming statistics; empty images do not poison a run.

    RMSEs below share the same identifiable, nearest-resized GT pixel set.
    They must not be compared numerically with the full-resolution depth metric.
    """

    def __init__(self):
        self.count = self.total_pixels = self.images = self.empty_images = 0
        self.sums = dict.fromkeys(
            ('g', 't', 'gg', 'tt', 'gt', 'abs', 'mono_sq', 'mv_sq',
             'fused_sq', 'oracle_sq', 'closed', 'open'), 0.0)

    def update(self, oracle):
        valid = oracle['valid']
        for mask in valid:
            self.images += 1
            self.empty_images += int(not mask.any().item())
        self.total_pixels += valid.numel()
        n = int(valid.sum().item())
        if not n:
            return
        g, t, mono, mv, gt = [oracle[k][valid].detach().double()
                              for k in ('gate', 'target', 'mono', 'mv', 'gt')]
        self.count += n
        delta = mv - mono
        vals = (g.sum(), t.sum(), g.square().sum(), t.square().sum(),
                (g*t).sum(), (g-t).abs().sum(), (mono-gt).square().sum(),
                (mv-gt).square().sum(), (mono+g*delta-gt).square().sum(),
                (mono+t*delta-gt).square().sum(), (t == 0).sum(), (t == 1).sum())
        for key, value in zip(self.sums, vals):
            self.sums[key] += float(value.item())

    def compute(self):
        result = {'valid_pixels': self.count, 'total_pixels': self.total_pixels,
                  'valid_fraction': self.count / self.total_pixels if self.total_pixels else 0.0,
                  'empty_samples': self.empty_images}
        names = ('target_mean', 'target_std', 'mae', 'corr', 'mono_rmse_low',
                 'mv_rmse_low', 'fused_rmse_low', 'oracle_rmse_low',
                 'target_closed_fraction', 'target_open_fraction')
        if not self.count:
            result.update(dict.fromkeys(names, float('nan')))
            return result
        n, s = self.count, self.sums
        g_var = max(0.0, s['gg'] - s['g']**2/n)
        t_var = max(0.0, s['tt'] - s['t']**2/n)
        denom = math.sqrt(g_var*t_var)
        corr = (s['gt'] - s['g']*s['t']/n)/denom if denom > 1e-12 else float('nan')
        result.update(target_mean=s['t']/n, target_std=math.sqrt(t_var/n),
                      mae=s['abs']/n, corr=max(-1.0, min(1.0, corr)) if math.isfinite(corr) else corr,
                      target_closed_fraction=s['closed']/n, target_open_fraction=s['open']/n)
        for output, key in (('mono', 'mono_sq'), ('mv', 'mv_sq'),
                            ('fused', 'fused_sq'), ('oracle', 'oracle_sq')):
            result[output+'_rmse_low'] = math.sqrt(s[key]/n)
        return result
