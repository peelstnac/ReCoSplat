"""Camera normalization and field-of-view helpers."""

import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor


def camera_normalization(
    pivotal_pose: Float[Tensor, "1 4 4"],
    poses: Float[Tensor, "n 4 4"],
) -> Float[Tensor, "n 4 4"]:
    """Re-center all poses so the pivotal camera becomes the identity."""
    canonical_camera_extrinsics = torch.tensor(
        [[
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]],
        dtype=torch.float32,
        device=pivotal_pose.device,
    )
    pivotal_pose_inv = torch.inverse(pivotal_pose)
    camera_norm_matrix = torch.bmm(canonical_camera_extrinsics, pivotal_pose_inv)

    poses = torch.bmm(camera_norm_matrix.repeat(poses.shape[0], 1, 1), poses)

    return poses


def get_fov(intrinsics: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 2"]:
    """Field of view (radians) from normalized intrinsics."""
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)
