"""Export predicted Gaussians in the standard 3DGS PLY layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .model.adapter.types import Gaussians

SH_C0 = 0.28209479177387814


def matrix_to_quaternion_wxyz(matrix: Tensor) -> Tensor:
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    q_abs = torch.sqrt(
        torch.clamp_min(
            torch.stack(
                (
                    1 + m00 + m11 + m22,
                    1 + m00 - m11 - m22,
                    1 - m00 + m11 - m22,
                    1 - m00 - m11 + m22,
                ),
                dim=-1,
            ),
            0,
        )
    )
    candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )
    candidates /= 2 * q_abs.clamp_min(0.1).unsqueeze(-1)
    choice = q_abs.argmax(dim=-1)
    batch = torch.arange(choice.numel(), device=choice.device)
    quaternions = candidates.reshape(-1, 4, 4)[batch, choice.reshape(-1)]
    quaternions = quaternions.reshape(*choice.shape, 4)
    return quaternions / quaternions.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def gaussian_ply_attributes(
    gaussians: Gaussians,
    opacity_threshold: float = 0.005,
    chunk_size: int = 262_144,
) -> np.ndarray:
    if gaussians.means.shape[0] != 1:
        raise ValueError("PLY export requires batch size one")
    keep = gaussians.opacities[0] > opacity_threshold
    count = int(keep.sum().item())
    attributes = np.empty((count, 62), dtype="<f4")
    cursor = 0
    for start in range(0, keep.numel(), chunk_size):
        end = min(start + chunk_size, keep.numel())
        mask = keep[start:end]
        amount = int(mask.sum().item())
        if amount == 0:
            continue

        covariance = gaussians.covariances[0, start:end][mask].float()
        eigenvalues, rotation = torch.linalg.eigh(covariance)
        rotation = rotation.clone()
        reflected = torch.linalg.det(rotation) < 0
        rotation[reflected, :, 0] *= -1
        scales = eigenvalues.clamp_min(1e-12).sqrt().log()
        quaternions = matrix_to_quaternion_wxyz(rotation)
        opacity = gaussians.opacities[0, start:end][mask].float().clamp(1e-6, 1 - 1e-6)
        opacity = torch.logit(opacity).unsqueeze(-1)

        block = torch.cat(
            (
                gaussians.means[0, start:end][mask].float(),
                torch.zeros((amount, 3), device=covariance.device),
                gaussians.harmonics[0, start:end, :, 0][mask].float(),
                torch.zeros((amount, 45), device=covariance.device),
                opacity,
                scales,
                quaternions,
            ),
            dim=-1,
        )
        attributes[cursor : cursor + amount] = block.detach().cpu().numpy()
        cursor += amount
    return attributes


def write_gaussian_ply(path: Path, attributes: np.ndarray) -> None:
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names.extend(f"f_rest_{index}" for index in range(45))
    names.extend(
        ("opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
    )
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(attributes)}"]
    header.extend(f"property float {name}" for name in names)
    header.extend(("end_header", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        attributes.astype("<f4", copy=False).tofile(handle)


def write_input_ply(path: Path, attributes: np.ndarray, max_points: int = 100_000) -> None:
    if len(attributes) > max_points:
        indices = np.linspace(0, len(attributes) - 1, max_points, dtype=np.int64)
        attributes = attributes[indices]
    dtype = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    vertices = np.empty(len(attributes), dtype=dtype)
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = attributes[:, index]
    for name in ("nx", "ny", "nz"):
        vertices[name] = 0
    colors = np.clip(0.5 + SH_C0 * attributes[:, 6:9], 0, 1)
    colors = np.round(colors * 255).astype(np.uint8)
    vertices["red"], vertices["green"], vertices["blue"] = colors.T

    header = [
        "ply", "format binary_little_endian 1.0", f"element vertex {len(vertices)}",
        "property float x", "property float y", "property float z",
        "property float nx", "property float ny", "property float nz",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header", "",
    ]
    with path.open("wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        vertices.tofile(handle)
