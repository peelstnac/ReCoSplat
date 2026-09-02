"""Relative camera-pose accuracy, trajectory error, and error-curve AUC."""

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from ..model.geometry import se3_inverse


def umeyama_sim3(
    src: Float[Tensor, "n 3"],
    dst: Float[Tensor, "n 3"],
) -> tuple[Tensor, Float[Tensor, "3 3"], Float[Tensor, "3"]]:
    """Least-squares similarity transform (s, R, t) with dst ~= s * R @ src + t
    (Umeyama 1991). Runs in float64 for stability; degenerate point sets (all
    coincident) fall back to s=1."""
    src = src.to(torch.float64)
    dst = dst.to(torch.float64)
    n = src.shape[0]

    mu_src = src.mean(0)
    mu_dst = dst.mean(0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = dst_c.T @ src_c / n
    u, d, vt = torch.linalg.svd(cov)
    s_mat = torch.eye(3, dtype=torch.float64, device=src.device)
    if torch.det(u) * torch.det(vt) < 0:
        s_mat[2, 2] = -1.0

    rotation = u @ s_mat @ vt
    var_src = (src_c**2).sum() / n
    if var_src < 1e-12:
        scale = torch.ones((), dtype=torch.float64, device=src.device)
    else:
        scale = (d * s_mat.diagonal()).sum() / var_src
    translation = mu_dst - scale * rotation @ mu_src
    return scale, rotation, translation


def orient_extrinsics(extrinsics: Float[Tensor, "v 4 4"]) -> Float[Tensor, "v 4 4"]:
    """Re-anchor a camera-to-world trajectory to its first view."""
    w2c_v1 = se3_inverse(extrinsics[0])
    return torch.einsum("ij, njk -> nik", w2c_v1, extrinsics)


def compute_pairwise_relative_poses(
    pred_poses: Float[Tensor, "b v 4 4"],
    target_poses: Float[Tensor, "b v 4 4"],
) -> tuple[Float[Tensor, "b n 4 4"], Float[Tensor, "b n 4 4"]]:
    """Compute relative poses for all ordered pairs of distinct views."""
    _, v, _, _ = pred_poses.shape
    device = pred_poses.device

    i_indices = torch.arange(v, device=device).repeat_interleave(v - 1)
    j_indices = torch.cat(
        [
            torch.cat([torch.arange(j, device=device), torch.arange(j + 1, v, device=device)])
            for j in range(v)
        ]
    )

    pred_rel = torch.linalg.inv(pred_poses[:, i_indices]) @ pred_poses[:, j_indices]
    target_rel = torch.linalg.inv(target_poses[:, i_indices]) @ target_poses[:, j_indices]
    return pred_rel, target_rel


@torch.no_grad()
def compute_rra_rta_errors(
    pred_poses: Float[Tensor, "v 4 4"],
    gt_poses: Float[Tensor, "v 4 4"],
) -> tuple[Float[Tensor, " n"], Float[Tensor, " n"]]:
    """Compute per-pair rotation and translation-direction errors in degrees."""
    pred = orient_extrinsics(pred_poses.to(torch.float64))
    gt = orient_extrinsics(gt_poses.to(torch.float64))

    pred_rel, gt_rel = compute_pairwise_relative_poses(pred.unsqueeze(0), gt.unsqueeze(0))
    pred_r, gt_r = pred_rel[0, :, :3, :3], gt_rel[0, :, :3, :3]
    pred_t, gt_t = pred_rel[0, :, :3, 3], gt_rel[0, :, :3, 3]

    m = torch.einsum("nij,njk->nik", pred_r.transpose(-1, -2), gt_r)
    cos_r = (m.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    err_r = torch.rad2deg(torch.abs(torch.acos(torch.clamp(cos_r, -1.0, 1.0))))

    denom = torch.norm(pred_t, dim=-1) * torch.norm(gt_t, dim=-1)
    cos_t = torch.clamp((pred_t * gt_t).sum(-1) / denom, -1.0, 1.0)
    err_t = torch.rad2deg(torch.acos(cos_t))
    err_t = torch.where(torch.norm(pred_t - gt_t, dim=-1) < 1e-6, torch.zeros_like(err_t), err_t)
    err_t = torch.minimum(err_t, 180 - err_t)

    return err_r, err_t


@torch.no_grad()
def compute_ate(
    pred_poses: Float[Tensor, "v 4 4"],
    gt_poses: Float[Tensor, "v 4 4"],
) -> float:
    """Absolute trajectory error: RMSE of camera centers after similarity (sim3)
    alignment of the predicted centers onto GT. GT is in the dataset's normalized
    scale (max pairwise context distance = 1), so the value reads as a fraction of
    the scene extent."""
    pred_centers = pred_poses[:, :3, 3].to(torch.float64)
    gt_centers = gt_poses[:, :3, 3].to(torch.float64)
    s, r, t = umeyama_sim3(pred_centers, gt_centers)
    aligned = s * pred_centers @ r.T + t
    return torch.sqrt(((aligned - gt_centers) ** 2).sum(-1).mean()).item()


@torch.no_grad()
def scene_max_pose_error(
    pred_poses: Float[Tensor, "v 4 4"],
    gt_poses: Float[Tensor, "v 4 4"],
) -> float:
    """Return the mean per-view maximum of rotation and translation errors."""
    pred = orient_extrinsics(pred_poses.to(torch.float64))[1:]
    gt = orient_extrinsics(gt_poses.to(torch.float64))[1:]

    m = torch.einsum("nij,njk->nik", pred[:, :3, :3].transpose(-1, -2), gt[:, :3, :3])
    cos_r = (m.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    err_r = torch.rad2deg(torch.abs(torch.acos(torch.clamp(cos_r, -1.0, 1.0))))

    pred_t, gt_t = pred[:, :3, 3], gt[:, :3, 3]
    denom = torch.norm(pred_t, dim=-1) * torch.norm(gt_t, dim=-1)
    cos_t = torch.clamp((pred_t * gt_t).sum(-1) / denom, -1.0, 1.0)
    err_t = torch.rad2deg(torch.acos(cos_t))
    err_t = torch.where(
        torch.norm(pred_t - gt_t, dim=-1) < 1e-6, torch.zeros_like(err_t), err_t
    )
    err_t = torch.minimum(err_t, 180 - err_t)

    return torch.maximum(err_t, err_r).mean().item()


def accuracy_at(errors: Float[Tensor, " n"], threshold_deg: float) -> float:
    """Fraction of pairs with error below the threshold (RRA@tau / RTA@tau)."""
    return (errors < threshold_deg).float().mean().item()


def pose_auc(errors, thresholds) -> list[float]:
    """Compute recall-to-error AUC at each threshold."""
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs
