"""Paper equations (3), (5), (8): right-tangent Gaussian SO(3) sampling.

This standalone module is not yet wired into STRUCTMAGNET or its data loader.
Covariance is an INPUT from a declared provider, not an injected noise label.
Units: covariance in radians squared, tangent vectors in radians.
"""
import torch


def sample_rotation_covariance(rotations, covariance, num_samples, *, generator=None):
    """Return (sampled_R, tangent_vectors, weights).

    Inputs are [B,V,3,3]. Outputs are [Q,B,V,3,3], [Q,B,V,3], [Q].
    R_q = R @ Exp([xi_q]x), xi_q = L @ z_q, z_q iid N(0,I).
    An exactly zero covariance gives exactly the nominal rotation. Otherwise
    covariance must be positive definite; no silent jitter/clamping is applied.
    Use a dedicated generator on rotations.device and persist its state.
    """
    if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples < 1:
        raise ValueError('num_samples must be a positive integer')
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError('rotations must have shape [B,V,3,3]')
    if covariance.shape != rotations.shape:
        raise ValueError('covariance must match rotations [B,V,3,3]')
    if not rotations.is_floating_point() or not covariance.is_floating_point():
        raise ValueError('rotations and covariance must be floating point')
    if covariance.device != rotations.device:
        raise ValueError('rotations and covariance must be on the same device')
    if rotations.numel() == 0:
        raise ValueError('batch and view dimensions must be nonempty')
    dtype = torch.float64 if rotations.dtype == torch.float64 else torch.float32
    r, cov = rotations.to(dtype), covariance.to(dtype)
    if not torch.isfinite(r).all() or not torch.isfinite(cov).all():
        raise ValueError('rotations and covariance must be finite')
    eye = torch.eye(3, dtype=dtype, device=r.device)
    tolerance = 1e-5 if dtype == torch.float64 else 2e-4
    if not torch.allclose(r.transpose(-1, -2) @ r, eye.expand_as(r), atol=tolerance, rtol=0):
        raise ValueError('rotations must be orthogonal')
    if not torch.allclose(torch.linalg.det(r), torch.ones_like(r[..., 0, 0]), atol=tolerance, rtol=0):
        raise ValueError('rotations must have determinant +1')
    scale = cov.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(torch.finfo(dtype).tiny)
    if ((cov - cov.transpose(-1, -2)).abs() > scale * 1e-5).any():
        raise ValueError('covariance must be symmetric')
    cov = (cov + cov.transpose(-1, -2)) * 0.5
    zero = cov.eq(0).all(dim=-1).all(dim=-1)
    # Cholesky requires SPD; zero is handled as a deterministic special case.
    factor, info = torch.linalg.cholesky_ex(torch.where(zero[..., None, None], eye, cov))
    if info.ne(0).any():
        raise ValueError('nonzero covariance must be positive definite')
    factor = torch.where(zero[..., None, None], torch.zeros_like(factor), factor)
    z = torch.randn((num_samples,) + r.shape[:-2] + (3,), device=r.device, dtype=dtype, generator=generator)
    xi = (factor.unsqueeze(0) @ z.unsqueeze(-1)).squeeze(-1)
    x, y, zz = xi.unbind(-1)
    o = torch.zeros_like(x)
    skew = torch.stack((o, -zz, y, zz, o, -x, -y, x, o), dim=-1).reshape(xi.shape[:-1] + (3, 3))
    sampled = r.unsqueeze(0) @ torch.matrix_exp(skew)
    sampled = torch.where(zero[None, ..., None, None], r.unsqueeze(0), sampled)
    weights = torch.full((num_samples,), 1.0 / num_samples, device=r.device, dtype=dtype)
    return sampled, xi, weights
