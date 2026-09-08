import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cost_volume_statistics(cost_volume, eps=1e-8):
    """
    Args:
        cost_volume: [B, D, H, W], MaGNet similarity scores (higher is better)

    Returns:
        entropy: [B, 1, H, W], normalized to [0, 1]
        peak:    [B, 1, H, W]
    """
    # FP16 rounds tiny probabilities/eps to zero; 0*log(0) is then NaN.
    # Keep the original similarity sign, rather than treating scores as costs.
    log_prob = F.log_softmax(cost_volume.float(), dim=1)
    prob = log_prob.exp()
    entropy = -torch.sum(
        prob * log_prob,
        dim=1,
        keepdim=True,
    )

    # normalize entropy to approximately [0, 1]
    if cost_volume.shape[1] > 1:
        entropy = entropy / math.log(cost_volume.shape[1])

    peak = torch.max(
        prob,
        dim=1,
        keepdim=True,
    )[0]

    return entropy.clamp(0.0, 1.0), peak


def rotation_uncertainty_map(
    rot_unc,
    batch_size,
    height,
    width,
    device,
    dtype,
):
    """
    rot_unc:
        None
        [B]
        [B, V]

    Unit: radians.
    """

    if rot_unc is None:
        return torch.zeros(
            batch_size, 1, height, width,
            device=device,
            dtype=dtype,
        )

    if not torch.is_tensor(rot_unc):
        rot_unc = torch.as_tensor(
            rot_unc,
            device=device,
            dtype=dtype,
        )
    else:
        rot_unc = rot_unc.to(
            device=device,
            dtype=dtype,
        )

    if rot_unc.ndim == 0:
        rot_unc = rot_unc.repeat(batch_size)

    elif rot_unc.ndim == 2:
        # mean uncertainty over source views
        rot_unc = rot_unc.mean(dim=1)

    elif rot_unc.ndim != 1:
        raise ValueError(
            "rot_unc must be None, [B], or [B, V], "
            f"but got {tuple(rot_unc.shape)}"
        )

    if rot_unc.shape[0] != batch_size:
        raise ValueError(
            f"rot_unc batch={rot_unc.shape[0]}, "
            f"expected {batch_size}"
        )

    rot_unc = rot_unc.view(
        batch_size, 1, 1, 1
    )

    return rot_unc.expand(
        -1, -1, height, width
    )


class GeometryGate(nn.Module):
    """
    Input channels:
        0: cost-volume entropy
        1: cost-volume peak
        2: monocular sigma
        3: rotation uncertainty
    """

    def __init__(self, ch_in=4, hidden_dim=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                ch_in, hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_dim, hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_dim, 1,
                kernel_size=1,
            ),
        )

        self.reset_parameters()

    def reset_parameters(self):
        # Make the initial model behave almost like original MaGNet.
        final_conv = self.net[-1]

        nn.init.zeros_(final_conv.weight)
        nn.init.constant_(final_conv.bias, 4.0)

    def forward(self, x):
        return torch.sigmoid(self.net(x))
