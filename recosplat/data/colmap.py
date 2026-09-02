"""Load registered images and cameras from an undistorted COLMAP scene."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]

    def intrinsic_matrix(self) -> np.ndarray:
        if self.model == "SIMPLE_PINHOLE":
            f, cx, cy = self.params
            fx, fy = f, f
        elif self.model == "PINHOLE":
            fx, fy, cx, cy = self.params
        else:
            raise ValueError(
                f"camera {self.camera_id} uses {self.model}; run COLMAP image_undistorter "
                "to produce PINHOLE or SIMPLE_PINHOLE cameras"
            )
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    camera_id: int
    name: str
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]

    def c2w(self) -> np.ndarray:
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = qvec_to_rotmat(self.qvec)
        w2c[:3, 3] = self.tvec
        return np.linalg.inv(w2c)


@dataclass
class PreparedView:
    image_id: int
    camera_id: int
    name: str
    image: Tensor
    intrinsics: Tensor
    c2w: Tensor


_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def qvec_to_rotmat(qvec: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def rotmat_to_qvec(rotation: np.ndarray) -> np.ndarray:
    m = rotation
    q = np.array(
        [
            np.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2,
            np.copysign(np.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2]),
            np.copysign(np.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0]),
            np.copysign(np.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1]),
        ]
    )
    q /= np.linalg.norm(q)
    return q


def _read_exact(handle, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise ValueError("truncated COLMAP binary file")
    return value


def _read_c_string(handle) -> str:
    value = bytearray()
    while True:
        char = _read_exact(handle, 1)
        if char == b"\x00":
            return value.decode("utf-8")
        value.extend(char)


def read_cameras_binary(path: Path) -> dict[int, ColmapCamera]:
    cameras = {}
    with path.open("rb") as handle:
        (count,) = struct.unpack("<Q", _read_exact(handle, 8))
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack(
                "<iiQQ", _read_exact(handle, 24)
            )
            if model_id not in _CAMERA_MODELS:
                raise ValueError(f"unsupported COLMAP camera model id: {model_id}")
            model, num_params = _CAMERA_MODELS[model_id]
            params = struct.unpack(f"<{num_params}d", _read_exact(handle, 8 * num_params))
            cameras[camera_id] = ColmapCamera(camera_id, model, width, height, params)
    return cameras


def read_images_binary(path: Path) -> dict[int, ColmapImage]:
    images = {}
    with path.open("rb") as handle:
        (count,) = struct.unpack("<Q", _read_exact(handle, 8))
        for _ in range(count):
            values = struct.unpack("<idddddddi", _read_exact(handle, 64))
            image_id = values[0]
            qvec = tuple(values[1:5])
            tvec = tuple(values[5:8])
            camera_id = values[8]
            name = _read_c_string(handle)
            (num_points,) = struct.unpack("<Q", _read_exact(handle, 8))
            handle.seek(24 * num_points, 1)
            images[image_id] = ColmapImage(image_id, camera_id, name, qvec, tvec)
    return images


def read_cameras_text(path: Path) -> dict[int, ColmapCamera]:
    cameras = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        camera_id = int(fields[0])
        cameras[camera_id] = ColmapCamera(
            camera_id,
            fields[1],
            int(fields[2]),
            int(fields[3]),
            tuple(float(value) for value in fields[4:]),
        )
    return cameras


def read_images_text(path: Path) -> dict[int, ColmapImage]:
    lines = [line.rstrip("\n") for line in path.open() if not line.lstrip().startswith("#")]
    images = {}
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line:
            continue
        fields = line.split()
        if len(fields) < 10:
            raise ValueError(f"invalid COLMAP image record: {line}")
        image_id = int(fields[0])
        images[image_id] = ColmapImage(
            image_id,
            int(fields[8]),
            fields[9],
            tuple(float(value) for value in fields[1:5]),
            tuple(float(value) for value in fields[5:8]),
        )
        if index < len(lines):
            index += 1
    return images


def find_sparse_model(scene_path: Path) -> Path:
    candidates = (scene_path / "sparse" / "0", scene_path / "sparse")
    for candidate in candidates:
        binary = (candidate / "cameras.bin").is_file() and (candidate / "images.bin").is_file()
        text = (candidate / "cameras.txt").is_file() and (candidate / "images.txt").is_file()
        if binary or text:
            return candidate
    raise FileNotFoundError(f"no COLMAP model found under {scene_path / 'sparse'}")


def load_colmap_model(
    scene_path: Path,
) -> tuple[dict[int, ColmapCamera], dict[int, ColmapImage]]:
    sparse_path = find_sparse_model(scene_path)
    if (sparse_path / "cameras.bin").is_file():
        cameras = read_cameras_binary(sparse_path / "cameras.bin")
        images = read_images_binary(sparse_path / "images.bin")
    else:
        cameras = read_cameras_text(sparse_path / "cameras.txt")
        images = read_images_text(sparse_path / "images.txt")
    for image in images.values():
        if image.camera_id not in cameras:
            raise ValueError(f"image {image.name} references missing camera {image.camera_id}")
        cameras[image.camera_id].intrinsic_matrix()
    return cameras, images


def select_images(
    images: dict[int, ColmapImage],
    image_list: Path | None,
    stride: int,
    max_views: int | None,
) -> list[ColmapImage]:
    if stride < 1:
        raise ValueError("stride must be at least one")
    by_name = {image.name: image for image in images.values()}
    if image_list is None:
        selected = sorted(images.values(), key=lambda image: image.name)
    else:
        names = [
            line.strip()
            for line in image_list.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(f"images are not registered by COLMAP: {missing}")
        selected = [by_name[name] for name in names]
    selected = selected[::stride]
    return selected if max_views is None else selected[:max_views]


def _prepare_image(image: Image.Image, intrinsics: np.ndarray, size: int) -> tuple[Tensor, Tensor]:
    width, height = image.size
    scale = max(size / height, size / width)
    scaled_width = round(width * scale)
    scaled_height = round(height * scale)
    scale_x, scale_y = scaled_width / width, scaled_height / height
    left = (scaled_width - size) // 2
    top = (scaled_height - size) // 2

    image = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
    image = image.crop((left, top, left + size, top + size))
    image_array = np.asarray(image, dtype=np.uint8).copy()
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0

    k = intrinsics.copy()
    k[0] *= scale_x
    k[1] *= scale_y
    k[0, 2] -= left
    k[1, 2] -= top
    k[0] /= size
    k[1] /= size
    return image_tensor, torch.from_numpy(k).float()


def prepare_views(
    scene_path: Path,
    images_dir: str,
    selected: list[ColmapImage],
    cameras: dict[int, ColmapCamera],
    size: int = 224,
) -> list[PreparedView]:
    if len(selected) < 2:
        raise ValueError("at least two registered images are required")
    views = []
    for record in selected:
        camera = cameras[record.camera_id]
        path = scene_path / images_dir / record.name
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as raw:
            image = raw.convert("RGB")
        if image.size != (camera.width, camera.height):
            raise ValueError(
                f"{record.name} is {image.size[0]}x{image.size[1]}, but camera "
                f"{camera.camera_id} declares {camera.width}x{camera.height}"
            )
        image_tensor, intrinsics = _prepare_image(image, camera.intrinsic_matrix(), size)
        views.append(
            PreparedView(
                record.image_id,
                record.camera_id,
                record.name,
                image_tensor,
                intrinsics,
                torch.from_numpy(record.c2w()).float(),
            )
        )

    c2w = torch.stack([view.c2w for view in views])
    canonical = torch.linalg.inv(c2w[0]) @ c2w
    scale = torch.pdist(canonical[:, :3, 3]).max()
    if not torch.isfinite(scale) or scale <= 1e-6:
        raise ValueError("selected cameras have no usable translation baseline")
    canonical[:, :3, 3] /= scale
    for view, pose in zip(views, canonical):
        view.c2w = pose
    return views


def save_viewer_cameras(output_dir: Path, views: list[PreparedView], c2w: Tensor) -> None:
    image_root = output_dir / "images"
    sparse_root = output_dir / "sparse" / "0"
    image_root.mkdir(parents=True, exist_ok=True)
    sparse_root.mkdir(parents=True, exist_ok=True)

    cameras_json = []
    camera_rows: dict[int, tuple[float, ...]] = {}
    image_rows = []
    for index, (view, pose_tensor) in enumerate(zip(views, c2w)):
        pose = pose_tensor.detach().cpu().double().numpy()
        k = view.intrinsics.detach().cpu().double().numpy().copy()
        k[0] *= view.image.shape[-1]
        k[1] *= view.image.shape[-2]
        output_image = image_root / view.name
        output_image = output_image.with_suffix(".png")
        output_image.parent.mkdir(parents=True, exist_ok=True)
        array = (view.image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(array).save(output_image)

        output_name = output_image.relative_to(image_root).as_posix()
        cameras_json.append(
            {
                "id": index,
                "img_name": output_name,
                "width": view.image.shape[-1],
                "height": view.image.shape[-2],
                "position": pose[:3, 3].tolist(),
                "rotation": pose[:3, :3].tolist(),
                "fx": float(k[0, 0]),
                "fy": float(k[1, 1]),
                "cx": float(k[0, 2]),
                "cy": float(k[1, 2]),
            }
        )

        camera_rows[view.camera_id] = (
            float(k[0, 0]),
            float(k[1, 1]),
            float(k[0, 2]),
            float(k[1, 2]),
        )
        w2c = np.linalg.inv(pose)
        qvec = rotmat_to_qvec(w2c[:3, :3])
        image_rows.append((view, output_name, qvec, w2c[:3, 3]))

    (output_dir / "cameras.json").write_text(json.dumps(cameras_json, indent=2) + "\n")
    with (sparse_root / "cameras.txt").open("w") as handle:
        handle.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for camera_id, params in sorted(camera_rows.items()):
            handle.write(f"{camera_id} PINHOLE 224 224 {' '.join(map(str, params))}\n")
    with (sparse_root / "images.txt").open("w") as handle:
        handle.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for view, name, qvec, tvec in image_rows:
            values = [view.image_id, *qvec, *tvec, view.camera_id, name]
            handle.write(" ".join(map(str, values)) + "\n\n")
    (sparse_root / "points3D.txt").write_text(
        "# 3D point list is empty; geometry is stored in point_cloud.ply\n"
    )
