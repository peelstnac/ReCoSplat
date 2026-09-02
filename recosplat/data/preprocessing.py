"""Utilities for creating evaluation chunks."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.io import decode_image, encode_png

from .chunk_dataset import convert_poses
from .shims.crop_shim import rescale_and_crop

TARGET_SHAPE = (224, 224)


def evaluation_requirements(index_dir: Path) -> dict[str, int]:
    """Return the largest frame index required for each evaluation scene."""
    requirements: dict[str, int] = {}
    paths = sorted(index_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no evaluation indices found in {index_dir}")
    for path in paths:
        with path.open("r") as file:
            entries = json.load(file)
        for key, entry in entries.items():
            if entry is None:
                continue
            indices = [*entry["context"], *entry["target"]]
            requirements[key] = max(requirements.get(key, -1), max(indices))
    return requirements


def camera_rows(
    normalized_intrinsics: tuple[float, float, float, float],
    camera_to_world: Iterable[np.ndarray],
) -> Tensor:
    fx, fy, cx, cy = normalized_intrinsics
    rows = []
    for c2w in camera_to_world:
        c2w = np.asarray(c2w, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"expected a 4x4 camera-to-world matrix, got {c2w.shape}")
        w2c = np.linalg.inv(c2w)
        rows.append([fx, fy, cx, cy, 0.0, 0.0, *w2c[:3].reshape(-1)])
    return torch.tensor(np.asarray(rows), dtype=torch.float32)


def _convert_image(
    path: Path,
    intrinsics: Tensor,
    source_shape: tuple[int, int],
    crop_border: int,
) -> tuple[Tensor, Tensor]:
    if crop_border:
        with Image.open(path) as source:
            if source.size != (source_shape[1], source_shape[0]):
                raise ValueError(
                    f"{path} has size {source.size}, expected "
                    f"{(source_shape[1], source_shape[0])}"
                )
            cropped = source.crop(
                (
                    crop_border,
                    crop_border,
                    source_shape[1] - crop_border,
                    source_shape[0] - crop_border,
                )
            )
            buffer = BytesIO()
            cropped.save(buffer, format="PNG")
        encoded = torch.frombuffer(bytearray(buffer.getvalue()), dtype=torch.uint8)
    else:
        encoded = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8)
    image = decode_image(encoded)
    expected_shape = (
        source_shape[0] - 2 * crop_border,
        source_shape[1] - 2 * crop_border,
    )
    if image.shape != (3, *expected_shape):
        raise ValueError(
            f"{path} has shape {tuple(image.shape)}, expected {(3, *expected_shape)}"
        )
    resized, adjusted = rescale_and_crop(
        image.to(torch.float32)[None] / 255.0,
        intrinsics[None],
        TARGET_SHAPE,
    )
    output = (resized[0] * 255).clip(0, 255).to(torch.uint8)
    return encode_png(output), adjusted[0]


def convert_images(
    paths: list[Path],
    cameras: Tensor,
    source_shape: tuple[int, int],
    crop_border: int,
    workers: int,
) -> tuple[list[Tensor], Tensor]:
    if len(paths) != len(cameras):
        raise ValueError(f"found {len(paths)} images but {len(cameras)} cameras")
    _, intrinsics = convert_poses(cameras)

    def convert(item: tuple[Path, Tensor]) -> tuple[Tensor, Tensor]:
        path, intrinsic = item
        return _convert_image(path, intrinsic, source_shape, crop_border)

    items = list(zip(paths, intrinsics, strict=True))
    if workers == 1:
        converted = list(map(convert, items))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            converted = list(pool.map(convert, items))

    images = [image for image, _ in converted]
    adjusted = torch.stack([intrinsic for _, intrinsic in converted])
    output_cameras = cameras.clone()
    output_cameras[:, 0] = adjusted[:, 0, 0]
    output_cameras[:, 1] = adjusted[:, 1, 1]
    output_cameras[:, 2] = adjusted[:, 0, 2]
    output_cameras[:, 3] = adjusted[:, 1, 2]
    return images, output_cameras


def image_files_by_index(directory: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        try:
            index = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        if index in result:
            raise ValueError(f"duplicate frame index {index} in {directory}")
        result[index] = path
    if not result:
        raise FileNotFoundError(f"no indexed images found in {directory}")
    return result


def image_shape(path: Path) -> tuple[int, int]:
    encoded = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8)
    _, height, width = decode_image(encoded).shape
    return height, width


class ChunkWriter:
    def __init__(self, output_root: Path, target_bytes: int) -> None:
        self.output_dir = output_root / "test"
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_bytes = target_bytes
        self.chunk: list[dict] = []
        self.chunk_bytes = 0
        self.chunk_index = 0
        self.scene_count = 0
        self.frame_count = 0

    def add(self, scene: dict) -> None:
        scene_bytes = sum(image.numel() for image in scene["images"])
        if self.chunk and self.chunk_bytes + scene_bytes > self.target_bytes:
            self._flush()
        self.chunk.append(scene)
        self.chunk_bytes += scene_bytes
        self.scene_count += 1
        self.frame_count += len(scene["images"])

    def finish(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self.chunk:
            return
        path = self.output_dir / f"{self.chunk_index:06d}.torch"
        temporary = path.with_suffix(f".torch.tmp.{os.getpid()}")
        torch.save(self.chunk, temporary)
        os.replace(temporary, path)
        print(f"wrote {path} ({len(self.chunk)} scenes)", flush=True)
        self.chunk = []
        self.chunk_bytes = 0
        self.chunk_index += 1


def require_scene_coverage(found: set[str], required: set[str]) -> None:
    missing = sorted(required - found)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", and {len(missing) - 5} more"
        raise FileNotFoundError(f"missing {len(missing)} evaluation scenes: {preview}{suffix}")


def select_existing_frames(
    frame_paths: list[Path], cameras: Tensor, largest_required_index: int
) -> tuple[list[Path], Tensor]:
    required_length = largest_required_index + 1
    if len(frame_paths) < required_length or len(cameras) < required_length:
        raise ValueError(
            f"evaluation requires frame {largest_required_index}, but only "
            f"{min(len(frame_paths), len(cameras))} aligned frames are available"
        )
    return frame_paths[:required_length], cameras[:required_length]
