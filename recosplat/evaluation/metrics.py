"""PSNR, SSIM, and LPIPS image metrics."""

from functools import cache

import torch
from einops import reduce
from jaxtyping import Float
from skimage.metrics import structural_similarity
from torch import Tensor


def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@cache
def get_lpips(device: torch.device):
    from lpips import LPIPS

    model = LPIPS(net="vgg").to(device)
    # Frozen: gradients may flow *through* it (pose opt differentiates the rendered
    # input) but never *into* its weights.
    model.requires_grad_(False)
    return model


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
    chunk_size: int = 50,
) -> Float[Tensor, " batch"]:
    lpips_model = get_lpips(predicted.device)
    results = []
    for gt_chunk, pred_chunk in zip(ground_truth.split(chunk_size), predicted.split(chunk_size)):
        value = lpips_model.forward(gt_chunk, pred_chunk, normalize=True)
        results.append(value[:, 0, 0, 0])
    return torch.cat(results)


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)
