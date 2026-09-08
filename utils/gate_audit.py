"""Paired gate diagnostics. Ground truth is used for scoring only.

First-step controls share frozen endpoints and masks. Full-trajectory controls
rerun sampling/refinement, so their later proposals need not match.
"""
import csv
import json
import math
from pathlib import Path

import torch

from utils.gate_oracle import OracleAccumulator, oracle_loss

FEATURES = ('entropy', 'peak', 'mono_sigma', 'rotation_radians')
FIXED = (0., .25, .5, .75, 1.)


def apply_gate_policy(gate, support, policy, value=.5):
    """No GT dependence; means/permutations use geometry-supported pixels only."""
    if policy == 'learned':
        return gate
    if policy == 'fixed':
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Fixed gate must be finite and in [0, 1]')
        return torch.where(support, torch.full_like(gate, value), torch.zeros_like(gate))
    if policy not in ('image_mean', 'shuffle'):
        raise ValueError('Unknown gate policy: ' + policy)
    out = torch.zeros_like(gate)
    for b in range(gate.shape[0]):
        mask = support[b]
        values = gate[b][mask]
        if not values.numel():
            continue
        if policy == 'image_mean':
            out[b][mask] = values.mean()
        else:
            # Local generator avoids altering training/data RNG state. This one
            # deterministic shuffle is a diagnostic, not a significance test.
            rng = torch.Generator(device=gate.device).manual_seed(1729 + b)
            out[b][mask] = values[torch.randperm(values.numel(), generator=rng,
                                               device=gate.device)]
    return out


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class FeatureMoments:
    """Streaming pixel Pearson correlations and within-image spatial variation."""
    def __init__(self):
        self.n = self.images = 0
        self.sum = torch.zeros(6, dtype=torch.float64)
        self.cross = torch.zeros(6, 6, dtype=torch.float64)
        self.lo = torch.full((6,), float('inf'), dtype=torch.float64)
        self.hi = -self.lo
        self.spatial = torch.zeros(6, dtype=torch.float64)

    def update(self, inputs, oracle):
        mask = oracle['valid'].flatten()
        if not mask.any():
            return
        x = torch.cat([inputs, oracle['gate'], oracle['target']], dim=1)
        x = x[0].reshape(6, -1)[:, mask].double()
        if not torch.isfinite(x).all():
            raise RuntimeError('Nonfinite diagnostic input')
        self.n += x.shape[1]
        self.images += 1
        self.sum += x.sum(1).cpu()
        self.cross += (x @ x.T).cpu()
        self.lo = torch.minimum(self.lo, x.amin(1).cpu())
        self.hi = torch.maximum(self.hi, x.amax(1).cpu())
        self.spatial += x.std(1, unbiased=False).cpu()

    def rows(self, noise):
        if not self.n:
            return []
        mean = self.sum / self.n
        cov = self.cross / self.n - mean[:, None] * mean[None, :]
        std = cov.diag().clamp_min(0).sqrt()
        names = FEATURES + ('gate', 'oracle_target')
        rows = []
        for j, name in enumerate(names):
            row = dict(noise_deg=noise, feature=name, pixels=self.n,
                       mean=mean[j].item(), std=std[j].item(), min=self.lo[j].item(),
                       max=self.hi[j].item(), mean_image_spatial_std=(self.spatial[j]/self.images).item())
            for k, other in enumerate(names):
                denom = (std[j]*std[k]).item()
                row['corr_' + other] = max(-1., min(1., cov[j, k].item()/denom)) if denom > 1e-12 else float('nan')
            rows.append(row)
        return rows


class FirstStepAudit:
    def __init__(self, gate_module, noise):
        self.gate_module, self.noise = gate_module, noise
        self.stats, self.losses = {}, {}
        self.features = FeatureMoments()
        self.per_image = []

    def __call__(self, oracle, aux, image_index):
        gate, support = oracle['gate'], aux['geometry_valid'][0]
        inputs = aux['gate_input'][0]
        self.features.update(inputs, oracle)
        variants = {'learned': gate}
        for value in FIXED:
            variants[f'fixed_{value:g}'] = apply_gate_policy(gate, support, 'fixed', value)
        for policy in ('image_mean', 'shuffle'):
            variants[policy] = apply_gate_policy(gate, support, policy)
        # Remove spatial information channel-by-channel while retaining that
        # image's level; rotation is already spatially constant, so also test zero.
        for j, name in enumerate(FEATURES):
            altered = inputs.clone()
            for b in range(inputs.shape[0]):
                mask = support[b, 0]
                if mask.any():
                    altered[b, j] = inputs[b, j][mask].mean()
            pred = self.gate_module(altered).float()
            variants['mean_input_' + name] = torch.where(support, pred, torch.zeros_like(pred))
        altered = inputs.clone()
        altered[:, 3] = 0
        pred = self.gate_module(altered).float()
        variants['zero_rotation_input'] = torch.where(support, pred, torch.zeros_like(pred))
        for name, g in variants.items():
            o = dict(oracle, gate=g)
            self.stats.setdefault(name, OracleAccumulator()).update(o)
            count = int(o['valid'].sum().item())
            sums = self.losses.setdefault(name, dict(pixels=0, smooth_l1=0., mse=0., depth_mse=0., sigmoid_slope=0.))
            sums['pixels'] += count
            if count:
                for loss in ('smooth_l1', 'mse', 'depth_mse'):
                    sums[loss] += oracle_loss(o, loss)[0].item()*count
                sums['sigmoid_slope'] += (g[o['valid']]*(1-g[o['valid']])).sum().item()
            one = OracleAccumulator()
            one.update(o)
            self.per_image.append(dict(noise_deg=self.noise, image_index=image_index,
                                       policy=name, **one.compute()))

    def rows(self):
        rows = []
        for name, stats in self.stats.items():
            sums = self.losses[name]
            metrics = {k: v/sums['pixels'] if sums['pixels'] else float('nan')
                       for k, v in sums.items() if k != 'pixels'}
            rows.append(dict(noise_deg=self.noise, policy=name, **stats.compute(), **metrics))
        return rows


def run_gate_audit(model, loader, device, args, cli, validate, checkpoint_args=None):
    output = Path(cli.output_dir) / 'logs'
    output.mkdir(parents=True, exist_ok=True)
    first, inputs, per_image, full = [], [], [], []
    original_policy = getattr(model, 'gate_policy', 'learned')
    original_value = getattr(model, 'gate_fixed_value', .5)
    try:
        for noise in (0., 5., 8.):
            model.gate_policy = 'learned'
            audit = FirstStepAudit(model.geometry_gate, noise)
            result = validate(model, loader, device, args, noise, cli.val_max_samples,
                              oracle_callback=audit)
            first.extend(audit.rows())
            inputs.extend(audit.features.rows(noise))
            per_image.extend(audit.per_image)
            full.append(dict(policy='learned', **result))
            print(f'Gate audit: {noise:g} deg learned + first-step controls complete', flush=True)
            if cli.gate_audit_full:
                policies = [(f'fixed_{v:g}', 'fixed', v) for v in FIXED]
                policies += [('image_mean', 'image_mean', .5), ('shuffle', 'shuffle', .5)]
                for name, policy, value in policies:
                    model.gate_policy, model.gate_fixed_value = policy, value
                    result = validate(model, loader, device, args, noise, cli.val_max_samples)
                    full.append(dict(policy=name, **result))
                    print(f'Gate audit: {noise:g} deg {name} RMSE={result["rmse"]:.6f}', flush=True)
            # Save completed noise levels, even if a later evaluation fails.
            write_csv(output / 'gate_audit_first_step.csv', first)
            write_csv(output / 'gate_audit_inputs.csv', inputs)
            write_csv(output / 'gate_audit_per_image.csv', per_image)
            write_csv(output / 'gate_audit_full.csv', full)
        metadata = dict(protocol='gate-audit-v1', args=vars(cli),
                        checkpoint_training_args=checkpoint_args,
                        first_step='shared frozen endpoints; shared identifiable mask; pixel weighted',
                        full='rerun all iterations; mean per-image final full-resolution metrics',
                        shuffle_seed=1729, ground_truth='scoring and oracle targets only')
        (output / 'gate_audit_config.json').write_text(json.dumps(metadata, indent=2))
        print('Gate audit saved:', output)
    finally:
        model.gate_policy, model.gate_fixed_value = original_policy, original_value
