import torch


def skew(v):
    """
    v: (..., 3)
    return: (..., 3, 3)
    """
    vx, vy, vz = v.unbind(dim=-1)

    O = torch.zeros_like(vx)

    K = torch.stack([
        O,  -vz,  vy,
        vz,  O,  -vx,
        -vy, vx,  O,
    ], dim=-1)

    return K.reshape(*v.shape[:-1], 3, 3)


def so3_exp(rotvec):
    """
    rotvec: (..., 3), radians
    """
    return torch.matrix_exp(skew(rotvec))


def generate_rotation_hypotheses(
    R,
    rot_hyp_vec,
):
    """
    R:
        B x V x 3 x 3

    rot_hyp_vec:
        B x V x 3
        principal rotation perturbation vector, radians

    Returns:
        rotations: list of 3 tensors
        weights:   tuple
    """

    dR_pos = so3_exp(rot_hyp_vec)
    dR_neg = so3_exp(-rot_hyp_vec)

    # Right perturbation convention:
    # R_q = R Exp(delta theta)
    R_neg = torch.matmul(R, dR_neg)
    R_nom = R
    R_pos = torch.matmul(R, dR_pos)

    rotations = [
        R_neg,
        R_nom,
        R_pos,
    ]

    weights = (
        0.25,
        0.50,
        0.25,
    )

    return rotations, weights
