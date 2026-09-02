"""Per-view photometric camera-pose refinement with SE(3) updates."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

if TYPE_CHECKING:
    from ..model.rendering.splatting_cuda import DecoderSplattingCUDA
    from ..model.rendering.splatting_gsplat import DecoderSplattingGSPlat
    from ..model.adapter.types import Gaussians


@dataclass
class PoseOptCfg:
    steps: int
    rot_lr: float
    trans_lr: float
    mse_weight: float
    lpips_weight: float
    renderer: Literal["gsplat", "splatting_cuda"] = "splatting_cuda"



def _skew(x: Float[Tensor, "... 3"]) -> Float[Tensor, "... 3 3"]:
    zero = torch.zeros_like(x[..., 0])
    return torch.stack(
        [
            torch.stack([zero, -x[..., 2], x[..., 1]], dim=-1),
            torch.stack([x[..., 2], zero, -x[..., 0]], dim=-1),
            torch.stack([-x[..., 1], x[..., 0], zero], dim=-1),
        ],
        dim=-2,
    )


def se3_exp(tau: Float[Tensor, "... 6"]) -> Float[Tensor, "... 4 4"]:
    """Compute batched SE(3) exponential maps for [translation, rotation] vectors."""
    rho, theta = tau[..., :3], tau[..., 3:]
    w = _skew(theta)
    w2 = w @ w
    angle = theta.norm(dim=-1, keepdim=True).unsqueeze(-1)
    eye = torch.eye(3, dtype=tau.dtype, device=tau.device).expand(*tau.shape[:-1], 3, 3)

    small = angle < 1e-5
    safe = torch.where(small, torch.ones_like(angle), angle)
    sin, cos = torch.sin(safe), torch.cos(safe)

    rot = torch.where(
        small,
        eye + w + 0.5 * w2,
        eye + (sin / safe) * w + ((1 - cos) / safe**2) * w2,
    )
    v_mat = torch.where(
        small,
        eye + 0.5 * w + (1.0 / 6.0) * w2,
        eye + ((1 - cos) / safe**2) * w + ((safe - sin) / safe**3) * w2,
    )
    t = (v_mat @ rho.unsqueeze(-1)).squeeze(-1)

    out = torch.zeros(*tau.shape[:-1], 4, 4, dtype=tau.dtype, device=tau.device)
    out[..., :3, :3] = rot
    out[..., :3, 3] = t
    out[..., 3, 3] = 1.0
    return out


def apply_pose_delta(
    extrinsics: Float[Tensor, "b v 4 4"],
    cam_trans_delta: Float[Tensor, "b v 3"],
    cam_rot_delta: Float[Tensor, "b v 3"],
) -> Float[Tensor, "b v 4 4"]:
    """Apply differentiable pose deltas in world-to-camera space and return c2w."""
    tau = torch.cat([cam_trans_delta, cam_rot_delta], dim=-1)
    w2c = torch.linalg.inv(extrinsics)
    new_w2c = se3_exp(tau) @ w2c
    return torch.linalg.inv(new_w2c)



def optimize_poses(
    decoder: "DecoderSplattingCUDA | DecoderSplattingGSPlat",
    gaussians: "Gaussians",
    target_views: dict,
    init_extrinsics: Float[Tensor, "b t 4 4"],
    cfg: PoseOptCfg,
) -> Float[Tensor, "b t 4 4"]:
    """Refine target camera poses while keeping Gaussians and model weights fixed.

    The target-view dictionary must contain image, intrinsics, near, and far tensors.
    """
    if cfg.steps == 0:
        return init_extrinsics.detach().clone()

    from .metrics import get_lpips

    image = target_views["image"]
    b, t, _, h, w = image.shape
    device = image.device

    lpips_model = get_lpips(device) if cfg.lpips_weight > 0 else None

    if cfg.renderer == "splatting_cuda":
        from ..model.rendering.splatting_cuda import DecoderSplattingCUDA

        cuda_decoder = DecoderSplattingCUDA(decoder.cfg).to(device).eval()
    elif cfg.renderer == "gsplat":
        cuda_decoder = None
    else:
        raise ValueError(f"unknown pose_opt renderer {cfg.renderer!r}")

    # The custom rasterizer path uses TF32 matmuls and restores the caller's state.
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    if cuda_decoder is not None:
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with torch.enable_grad():
            cam_rot_delta = nn.Parameter(torch.zeros([b, t, 3], device=device))
            cam_trans_delta = nn.Parameter(torch.zeros([b, t, 3], device=device))
            optimizer = torch.optim.Adam(
                [
                    {"params": [cam_rot_delta], "lr": cfg.rot_lr},
                    {"params": [cam_trans_delta], "lr": cfg.trans_lr},
                ]
            )

            extrinsics = init_extrinsics.detach().clone()
            for _ in range(cfg.steps):
                optimizer.zero_grad()
                if cuda_decoder is not None:
                    output = cuda_decoder.forward(
                        gaussians,
                        extrinsics,
                        target_views["intrinsics"],
                        target_views["near"],
                        target_views["far"],
                        (h, w),
                        cam_rot_delta=cam_rot_delta,
                        cam_trans_delta=cam_trans_delta,
                    )
                else:
                    output = decoder.forward(
                        gaussians,
                        apply_pose_delta(extrinsics, cam_trans_delta, cam_rot_delta),
                        target_views["intrinsics"],
                        target_views["near"],
                        target_views["far"],
                        (h, w),
                    )

                loss = cfg.mse_weight * (output.color - image).square().mean()
                if lpips_model is not None:
                    lpips_value = lpips_model.forward(
                        rearrange(output.color, "b v c h w -> (b v) c h w"),
                        rearrange(image, "b v c h w -> (b v) c h w"),
                        normalize=True,
                    ).mean()
                    loss = loss + cfg.lpips_weight * lpips_value

                loss.backward()
                with torch.no_grad():
                    optimizer.step()
                    extrinsics = apply_pose_delta(extrinsics, cam_trans_delta, cam_rot_delta)
                    cam_rot_delta.data.fill_(0)
                    cam_trans_delta.data.fill_(0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    return extrinsics.detach()
