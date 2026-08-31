import torch

import models.submodules.homography as homography

from models.submodules.rotation_utils import (
    generate_rotation_hypotheses,
)


def est_costvolume_CW_marginalized(
    depth_volume,
    ref_feat,
    nghbr_feat,
    ref_gmm,
    nghbr_gmm,
    Rs_src,
    ts_src,
    is_valid,
    cam_intrins,
    thres,
    rot_hyp_vec=None,
):
    """
    Pose-marginalized version of MaGNet CW cost volume.

    Rs_src:
        B x V x 3 x 3

    rot_hyp_vec:
        B x V x 3
    """

    # Exact fallback to original MaGNet
    if rot_hyp_vec is None:
        return homography.est_costvolume_CW(
            depth_volume,
            ref_feat,
            nghbr_feat,
            ref_gmm,
            nghbr_gmm,
            Rs_src,
            ts_src,
            is_valid,
            cam_intrins,
            thres,
        )

    rotations, weights = generate_rotation_hypotheses(
        Rs_src,
        rot_hyp_vec,
    )

    cost_volume_marg = None

    for R_q, w_q in zip(rotations, weights):

        cost_q = homography.est_costvolume_CW(
            depth_volume,
            ref_feat,
            nghbr_feat,
            ref_gmm,
            nghbr_gmm,
            R_q,
            ts_src,
            is_valid,
            cam_intrins,
            thres,
        )

        if cost_volume_marg is None:
            cost_volume_marg = w_q * cost_q
        else:
            cost_volume_marg = (
                cost_volume_marg
                + w_q * cost_q
            )

    return cost_volume_marg
