"""Small, depth-supervised retraining helpers; no gate oracle or pose labels."""
import math

import torch


def perturb_rotations(poses, clean_probability=0.5, max_degrees=5.0,
                      generator=None, fixed_degrees=None):
    """Right-multiply reference-to-source rotations; copy translation exactly.

    A Bernoulli draw keeps each complete sample clean. Other samples receive
    independent uniform [0, max_degrees] angles and uniform sphere axes per view.
    Returned angles are diagnostics ONLY, never inputs to the depth network.
    A caller-owned generator makes evaluation repeatable without changing the
    training RNG. Batch size one uses the same mixture over successive steps.
    """
    if poses.ndim != 4 or poses.shape[-2:] != (4, 4):
        raise ValueError('Expected poses [B, V, 4, 4]')
    if not 0 <= clean_probability <= 1 or not math.isfinite(max_degrees) or max_degrees < 0:
        raise ValueError('Invalid clean probability or rotation range')
    if fixed_degrees is not None and (not math.isfinite(fixed_degrees) or fixed_degrees < 0):
        raise ValueError('fixed_degrees must be finite and nonnegative')
    b, v = poses.shape[:2]
    # CPU generator decouples noise from model initialization, AMP and dropout.
    if fixed_degrees is None:
        active = torch.rand(b, 1, generator=generator) >= clean_probability
        angles = torch.rand(b, v, generator=generator) * max_degrees * active
    else:
        angles = torch.full((b, v), float(fixed_degrees))
    axes = torch.randn(b, v, 3, generator=generator)
    axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    rotvec = axes * torch.deg2rad(angles).unsqueeze(-1)
    x, y, z = rotvec.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(b, v, 3, 3)
    delta = torch.matrix_exp(skew).to(device=poses.device, dtype=poses.dtype)
    result = poses.clone()
    result[..., :3, :3] = poses[..., :3, :3] @ delta
    return result, angles


def depth_loss(predictions, gt, min_depth=1e-3, max_depth=10.0, gamma=0.8):
    """Original MaGNet discounted Gaussian NLL, computed in FP32.

    Only GT determines the evaluation mask. Non-finite predictions on valid GT
    raise an error rather than making a failing model look better by masking it.
    The original variance floor is retained; no oracle or auxiliary gate loss.
    """
    valid = torch.isfinite(gt) & (gt > min_depth) & (gt < max_depth)
    if not valid.any() or not predictions:
        raise ValueError('Depth loss requires valid GT and nonempty predictions')
    target = gt.float()[valid]
    loss = target.new_zeros(())
    for i, pred in enumerate(predictions):
        mu, sigma = pred.float().split(1, dim=1)
        mu, sigma = mu[valid], sigma[valid]
        if not torch.isfinite(mu).all() or not torch.isfinite(sigma).all() or (sigma <= 0).any():
            raise FloatingPointError('Invalid depth Gaussian on valid GT')
        var = sigma.square().clamp_min(1e-10)
        nll = (mu - target).square() / (2 * var) + 0.5 * var.log()
        loss = loss + gamma ** (len(predictions) - i - 1) * nll.mean()
    return loss


def configure_training(model):
    """Keep pretrained D/F fixed; train original G/upsampling plus optional gate."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(('g_net.', 'mask_head.')) or
                                 (model.gate_mode == 'learned' and name.startswith('geometry_gate.')))
    set_train_mode(model)
    return [p for p in model.parameters() if p.requires_grad]


def set_train_mode(model):
    model.train()
    model.d_net.eval()
    model.f_net.eval()


def load_file(path):
    # Full training checkpoints include optimizer/RNG state. Only load trusted files.
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # Jetson installations with older PyTorch
        return torch.load(path, map_location='cpu')


def load_backbone(path, model):
    checkpoint = load_file(path)
    state = checkpoint.get('model', checkpoint)
    state = {k[len('module.'):] if k.startswith('module.') else k: v
             for k, v in state.items()}
    # D/F checkpoints have already been loaded strictly by STRUCTMAGNET.
    required = {k: v for k, v in model.state_dict().items()
                if k.startswith(('g_net.', 'mask_head.'))}
    missing = [k for k, v in required.items() if k not in state or state[k].shape != v.shape]
    if missing:
        raise ValueError('MaGNet checkpoint is missing compatible tensors: ' + ', '.join(missing))
    model.load_state_dict({k: state[k] for k in required}, strict=False)


def save_checkpoint(path, model, optimizer, scaler, epoch, step, best_score, config,
                    noise_generator, loader_generator):
    """Store every updated module; a gate-only file cannot resume G-Net training."""
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if k.startswith(('g_net.', 'mask_head.', 'geometry_gate.'))}
    checkpoint = dict(format='scannet-depth-v1', model=state, optimizer=optimizer.state_dict(),
                      scaler=scaler.state_dict(), epoch=epoch, step=step,
                      best_score=best_score, config=dict(config),
                      noise_rng=noise_generator.get_state(),
                      loader_rng=loader_generator.get_state(), torch_rng=torch.get_rng_state())
    if torch.cuda.is_available():
        checkpoint['cuda_rng'] = torch.cuda.get_rng_state_all()
    temporary = str(path) + '.tmp'
    torch.save(checkpoint, temporary)
    import os
    os.replace(temporary, path)


def restore_checkpoint(path, model, optimizer=None, scaler=None,
                       noise_generator=None, loader_generator=None):
    checkpoint = load_file(path)
    if checkpoint.get('format') != 'scannet-depth-v1':
        raise ValueError('Expected a scannet-depth-v1 checkpoint, not a gate-only warm-up')
    if checkpoint['config']['gate_mode'] != model.gate_mode:
        raise ValueError('Checkpoint gate_mode differs from the requested model')
    expected = {k for k in model.state_dict()
                if k.startswith(('g_net.', 'mask_head.', 'geometry_gate.'))}
    if set(checkpoint['model']) != expected:
        raise ValueError('Retraining checkpoint does not contain all updated modules')
    model.load_state_dict(checkpoint['model'], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler'])
    if noise_generator is not None:
        noise_generator.set_state(checkpoint['noise_rng'])
    if loader_generator is not None:
        loader_generator.set_state(checkpoint['loader_rng'])
    if optimizer is not None:
        torch.set_rng_state(checkpoint['torch_rng'])
        if 'cuda_rng' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    return checkpoint
