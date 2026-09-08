import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np

from models.DNET import DNET
from models.FNET import FNET
import models.submodules.homography as homography
from models.submodules.geometry_gate import (
    GeometryGate,
    cost_volume_statistics,
    rotation_uncertainty_map,
)


def upsample_depth_via_mask(depth, up_mask, k):
    """Learned convex upsampling used by the original MaGNet."""
    N, o_dim, H, W = depth.shape
    up_mask = up_mask.view(N, 1, 9, k, k, H, W)
    up_mask = torch.softmax(up_mask, dim=2)

    up_depth = F.unfold(depth, [3, 3], padding=1)
    up_depth = up_depth.view(N, o_dim, 9, 1, 1, H, W)
    up_depth = torch.sum(up_mask * up_depth, dim=2)

    up_depth = up_depth.permute(0, 1, 4, 2, 5, 3)
    return up_depth.reshape(N, o_dim, k * H, k * W)


def load_checkpoint(fpath, model):
    """Strict loader used for the original D-Net/F-Net checkpoints."""
    ckpt = torch.load(fpath, map_location='cpu')
    if 'model' in ckpt:
        ckpt = ckpt['model']

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        load_dict[k] = v

    model.load_state_dict(load_dict)
    return model


class GNET(nn.Module):
    def __init__(self, ch_in, ch_out=2):
        super().__init__()
        h_dim = 128
        self.gnet = nn.Sequential(
            nn.Conv2d(ch_in, h_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, h_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, h_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, ch_out, 1),
        )

    def predict_params(self, cost_volume):
        """Return original MaGNet normalized mean residual and sigma scale."""
        d_output = self.gnet(cost_volume)
        # Gaussian parameter arithmetic stays FP32 under autocast.
        mu_res, sigma_raw = torch.split(d_output.float(), 1, dim=1)
        sigma_scale = F.elu(sigma_raw) + 1.0 + 1e-10
        return mu_res, sigma_scale

    def forward(self, cost_volume, ref_gmm):
        mu_0, sigma_0 = torch.split(ref_gmm, 1, dim=1)
        mu_res, sigma_scale = self.predict_params(cost_volume)
        mu_new = mu_0 + mu_res * sigma_0
        sigma_new = sigma_scale * sigma_0
        return torch.cat([mu_new, sigma_new], dim=1)


class STRUCTMAGNET(nn.Module):
    """Stage-1 StructMaGNet: original MaGNet + GeometryGate."""

    def __init__(self, args):
        super().__init__()
        self.args = args

        print('loading DNET...{}'.format(args.DNET_ckpt))
        self.d_net = DNET(args, dnet=False)
        self.d_net = load_checkpoint(args.DNET_ckpt, self.d_net)
        for param in self.d_net.parameters():
            param.requires_grad = False
        self.d_net.eval()

        print('loading FNET... {}'.format(args.FNET_ckpt))
        self.f_net = FNET(args)
        self.f_net = load_checkpoint(args.FNET_ckpt, self.f_net)
        for param in self.f_net.parameters():
            param.requires_grad = False
        self.f_net.eval()

        self.sampling_range = args.MAGNET_sampling_range
        self.n_samples = args.MAGNET_num_samples
        self.weighting = args.MAGNET_mvs_weighting
        self.train_iter = args.MAGNET_num_train_iter
        self.test_iter = args.MAGNET_num_test_iter
        self.dpv_height = args.dpv_height
        self.dpv_width = args.dpv_width
        self.k_list = self.depth_sampling()
        self.downsample_ratio = args.downsample_ratio

        dnet_fdim = 256
        self.g_net = GNET(
            ch_in=dnet_fdim + self.n_samples,
            ch_out=2,
        )

        self.geometry_gate = GeometryGate(
            ch_in=4,
            hidden_dim=32,
            init_bias=getattr(args, 'gate_init_bias', 4.0),
        )

        h_dim = 128
        self.mask_head = nn.Sequential(
            nn.Conv2d(dnet_fdim, h_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, h_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, h_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                h_dim,
                9 * self.downsample_ratio * self.downsample_ratio,
                1,
            ),
        )
        self.upsample_depth = upsample_depth_via_mask

    def depth_sampling(self):
        from scipy.special import erf
        from scipy.stats import norm

        p_total = erf(self.sampling_range / np.sqrt(2))
        idx_list = np.arange(0, self.n_samples + 1)
        p_list = (
            (1 - p_total) / 2
            + ((idx_list / self.n_samples) * p_total)
        )
        k_list = norm.ppf(p_list)
        k_list = (k_list[1:] + k_list[:-1]) / 2
        return list(k_list)

    def forward(
        self,
        ref_img,
        nghbr_imgs,
        nghbr_poses,
        is_valid,
        cam_intrins,
        mode='train',
        rot_unc=None,
        return_aux=False,
    ):
        B = ref_img.shape[0]

        # D-Net/F-Net remain frozen in Stage-1.
        with torch.no_grad():
            mono_gmms, x_d3 = self.d_net(
                torch.cat((ref_img, nghbr_imgs), dim=0)
            )
            mono_gmms = mono_gmms.detach().float()

            ref_gmms = mono_gmms[:B, ...]
            _, mono_sigma = torch.split(ref_gmms, 1, dim=1)
            mono_sigma = mono_sigma.detach()

            x_d3 = x_d3[:B, ...].detach()
            nghbr_gmms = mono_gmms[B:, ...]

            feat_4 = self.f_net(
                torch.cat((ref_img, nghbr_imgs), dim=0)
            )
            ref_feat_4 = feat_4[:B, ...].detach()
            nghbr_feat_4 = feat_4[B:, ...].detach()

        Rs_src = nghbr_poses[:, :, :3, :3]
        ts_src = nghbr_poses[:, :, :3, 3]

        # Keep the initial monocular Gaussian at index 0, matching MaGNet.
        pred_low_list = [ref_gmms]
        gate_list = []
        entropy_list = []
        gate_input_list = []
        peak_list = []
        ungated_gmm_list = []
        geometry_valid_list = []
        # Frame-level pose availability. Per-pixel visibility/occlusion support
        # remains a separate future matching change; a flat cost is not a mask.
        has_source = is_valid.to(ref_gmms.device).eq(1).any(dim=1).view(B, 1, 1, 1)

        n_iter = self.train_iter if mode == 'train' else self.test_iter

        for _ in range(n_iter):
            ref_mu, ref_sigma = torch.split(
                pred_low_list[-1].detach(),
                1,
                dim=1,
            )

            depth_volume = [
                ref_mu + ref_sigma * k
                for k in self.k_list
            ]
            depth_volume = torch.cat(depth_volume, dim=1)

            thres = int(self.weighting.split('CW')[1])
            cost_volume = homography.est_costvolume_CW(
                depth_volume,
                ref_feat_4,
                nghbr_feat_4,
                ref_gmms,
                nghbr_gmms,
                Rs_src,
                ts_src,
                is_valid,
                cam_intrins,
                thres,
            )

            # Some 7-Scenes warps can yield non-finite matching values
            # near invalid/out-of-view projections. Prevent those values
            # from contaminating G-Net or GeometryGate.
            cost_volume_safe = torch.nan_to_num(
                cost_volume.detach(),
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )

            gnet_input = torch.cat(
                [cost_volume_safe, x_d3],
                dim=1,
            )
            gnet_input = torch.nan_to_num(
                gnet_input,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )

            prev_gmm = pred_low_list[-1].detach()
            prev_mu, prev_sigma = torch.split(
                prev_gmm,
                1,
                dim=1,
            )

            # Frozen original MaGNet proposal.
            mu_res, sigma_scale = self.g_net.predict_params(
                gnet_input
            )

            raw_mu = prev_mu + mu_res * prev_sigma
            raw_sigma = sigma_scale * prev_sigma
            raw_mv_gmm = torch.cat(
                [raw_mu, raw_sigma],
                dim=1,
            )

            cost_entropy, cost_peak = cost_volume_statistics(
                cost_volume_safe
            )

            rot_unc_map = rotation_uncertainty_map(
                rot_unc=rot_unc,
                batch_size=B,
                height=cost_volume.shape[2],
                width=cost_volume.shape[3],
                device=cost_volume.device,
                dtype=cost_volume.dtype,
            )

            gate_input = torch.cat(
                [
                    cost_entropy,
                    cost_peak,
                    mono_sigma,
                    rot_unc_map,
                ],
                dim=1,
            )
            gate_input = torch.nan_to_num(
                gate_input,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )

            g_geo = self.geometry_gate(gate_input)

            # Smooth-L1 does not require clipping away from 0/1. Do not hide
            # non-finite gate outputs: the shared oracle checker reports them.
            g_geo = g_geo.float()
            geometry_valid = (has_source
                              & torch.isfinite(cost_volume).all(dim=1, keepdim=True)
                              & torch.isfinite(raw_mv_gmm).all(dim=1, keepdim=True)
                              & (raw_sigma > 0))
            g_geo = torch.where(geometry_valid, g_geo, torch.zeros_like(g_geo))
            policy = getattr(self, 'gate_policy', 'learned')
            if policy != 'learned':
                if self.training:
                    raise RuntimeError('Gate policies are evaluation-only')
                from utils.gate_audit import apply_gate_policy
                g_geo = apply_gate_policy(g_geo, geometry_valid, policy,
                                          getattr(self, 'gate_fixed_value', 0.5))
            # Sanitize invalid proposals BEFORE multiplication: 0*NaN is NaN.
            safe_mu = torch.where(geometry_valid, raw_mu, prev_mu)
            safe_sigma = torch.where(geometry_valid, raw_sigma, prev_sigma)
            mu_new = prev_mu + g_geo * (safe_mu - prev_mu)
            # Convex form avoids cancellation to zero when g==1 and the
            # proposal sigma is tiny but positive (common with saturated AMP gates).
            sigma_new = (1.0 - g_geo) * prev_sigma + g_geo * safe_sigma

            new_pred = torch.cat(
                [mu_new, sigma_new],
                dim=1,
            )

            # IMPORTANT: append INSIDE the iterative refinement loop.
            pred_low_list.append(new_pred)
            gate_list.append(g_geo)
            if return_aux:
                gate_input_list.append(gate_input.detach())
            entropy_list.append(cost_entropy)
            peak_list.append(cost_peak)
            ungated_gmm_list.append(raw_mv_gmm.detach())
            geometry_valid_list.append(geometry_valid.detach())

        mask = self.mask_head(x_d3)
        pred_list = [
            self.upsample_depth(
                pred,
                mask,
                self.downsample_ratio,
            )
            for pred in pred_low_list[1:]
        ]

        if return_aux:
            aux = {
                'geometry_gate': gate_list,
                'gate_input': gate_input_list,
                'cost_entropy': entropy_list,
                'cost_peak': peak_list,
                'rot_unc': rot_unc,
                'mono_gmm': ref_gmms,
                # Original MaGNet proposal before GeometryGate at each iteration.
                'ungated_gmm': ungated_gmm_list,
                'geometry_valid': geometry_valid_list,
                # Useful for debugging / future losses.
                'gated_gmm_lowres': pred_low_list[1:],
            }
            return pred_list, aux

        return pred_list


class MAGNET_F(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.f_net = FNET(args)

    def forward(
        self,
        ref_img,
        nghbr_imgs,
        nghbr_poses,
        is_valid,
        cam_intrins,
        d_center,
    ):
        B = ref_img.shape[0]

        feat_4 = self.f_net(
            torch.cat((ref_img, nghbr_imgs), dim=0)
        )
        ref_feat_4 = feat_4[:B, ...]
        nghbr_feat_4 = feat_4[B:, ...]

        Rs_src = nghbr_poses[:, :, :3, :3]
        ts_src = nghbr_poses[:, :, :3, 3]

        return homography.est_costvolume_F(
            d_center,
            ref_feat_4,
            nghbr_feat_4,
            Rs_src,
            ts_src,
            is_valid,
            cam_intrins,
        )
