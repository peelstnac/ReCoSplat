"""Convert supported evaluation datasets to ReCoSplat's 224x224 chunk format."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from recosplat.data.preprocessing import (
    ChunkWriter,
    camera_rows,
    convert_images,
    evaluation_requirements,
    image_files_by_index,
    image_shape,
    require_scene_coverage,
    select_existing_frames,
)

BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])

INDEX_DIRECTORIES = {
    "dl3dv": REPOSITORY_ROOT / "assets" / "DL3DV" / "nvs",
    "scannet": REPOSITORY_ROOT / "assets" / "ScanNet" / "nvs",
    "scannetpp": REPOSITORY_ROOT / "assets" / "ScanNet++" / "nvs",
}


def nerfstudio_scene(
    scene_root: Path,
    largest_required_index: int,
    workers: int,
    image_directory: Path,
    metadata_path: Path,
) -> dict:
    with metadata_path.open("r") as file:
        metadata = json.load(file)

    height, width = int(metadata["h"]), int(metadata["w"])
    normalized_intrinsics = (
        float(metadata["fl_x"]) / width,
        float(metadata["fl_y"]) / height,
        float(metadata["cx"]) / width,
        float(metadata["cy"]) / height,
    )
    indexed_images = image_files_by_index(image_directory)
    paths = []
    camera_to_world = []
    for frame in metadata["frames"]:
        index = int(Path(frame["file_path"]).stem.split("_")[-1])
        if index not in indexed_images:
            raise FileNotFoundError(f"missing frame {index} in {image_directory}")
        paths.append(indexed_images[index])
        camera_to_world.append(np.asarray(frame["transform_matrix"]) @ BLENDER_TO_OPENCV)

    cameras = camera_rows(normalized_intrinsics, camera_to_world)
    paths, cameras = select_existing_frames(paths, cameras, largest_required_index)
    images, cameras = convert_images(paths, cameras, image_shape(paths[0]), 0, workers)
    return {"key": scene_root.name, "cameras": cameras, "images": images}


def convert_dl3dv(
    input_root: Path, requirements: dict[str, int], workers: int, writer: ChunkWriter
) -> set[str]:
    found = {
        key
        for key in requirements
        if (input_root / key / "nerfstudio" / "images_4").is_dir()
        and (input_root / key / "nerfstudio" / "transforms.json").is_file()
    }
    require_scene_coverage(found, set(requirements))
    for key in sorted(requirements):
        scene_root = input_root / key
        image_directory = scene_root / "nerfstudio" / "images_4"
        print(f"converting {key}", flush=True)
        writer.add(
            nerfstudio_scene(
                scene_root,
                requirements[key],
                workers,
                image_directory,
                scene_root / "nerfstudio" / "transforms.json",
            )
        )
    return found


def convert_scannetpp(
    input_root: Path, requirements: dict[str, int], workers: int, writer: ChunkWriter
) -> set[str]:
    found = {
        key
        for key in requirements
        if (input_root / key / "iphone" / "rgb").is_dir()
        and (input_root / key / "iphone" / "nerfstudio" / "transforms.json").is_file()
    }
    require_scene_coverage(found, set(requirements))
    for key in sorted(requirements):
        scene_root = input_root / key
        image_directory = scene_root / "iphone" / "rgb"
        print(f"converting {key}", flush=True)
        writer.add(
            nerfstudio_scene(
                scene_root,
                requirements[key],
                workers,
                image_directory,
                scene_root / "iphone" / "nerfstudio" / "transforms.json",
            )
        )
    return found


def convert_scannet_scene(
    scene_root: Path, largest_required_index: int, workers: int
) -> dict:
    image_directory = scene_root / "color"
    indexed_images = image_files_by_index(image_directory)
    extrinsics = np.load(scene_root / "extrinsics.npy")
    intrinsic = np.loadtxt(scene_root / "intrinsic" / "intrinsic_color.txt")
    if intrinsic.shape != (4, 4):
        raise ValueError(f"invalid intrinsic matrix in {scene_root}")

    first_image = next(iter(indexed_images.values()))
    height, width = image_shape(first_image)
    border = 20
    cropped_height = height - 2 * border
    cropped_width = width - 2 * border
    normalized_intrinsics = (
        float(intrinsic[0, 0]) / cropped_width,
        float(intrinsic[1, 1]) / cropped_height,
        (float(intrinsic[0, 2]) - border) / cropped_width,
        (float(intrinsic[1, 2]) - border) / cropped_height,
    )

    paths = []
    camera_to_world = []
    for index, c2w in enumerate(extrinsics):
        if np.isinf(c2w).any():
            continue
        if index not in indexed_images:
            raise FileNotFoundError(f"missing frame {index} in {image_directory}")
        paths.append(indexed_images[index])
        camera_to_world.append(c2w)

    cameras = camera_rows(normalized_intrinsics, camera_to_world)
    paths, cameras = select_existing_frames(paths, cameras, largest_required_index)
    images, cameras = convert_images(paths, cameras, (height, width), border, workers)
    return {"key": scene_root.name, "cameras": cameras, "images": images}


def convert_scannet(
    input_root: Path, requirements: dict[str, int], workers: int, writer: ChunkWriter
) -> set[str]:
    scene_parent = input_root / "test" if (input_root / "test").is_dir() else input_root
    found = {
        key
        for key in requirements
        if (scene_parent / key / "color").is_dir()
        and (scene_parent / key / "extrinsics.npy").is_file()
        and (scene_parent / key / "intrinsic" / "intrinsic_color.txt").is_file()
    }
    require_scene_coverage(found, set(requirements))
    for key in sorted(requirements):
        scene_root = scene_parent / key
        print(f"converting {key}", flush=True)
        writer.add(convert_scannet_scene(scene_root, requirements[key], workers))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(INDEX_DIRECTORIES))
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size-mb", type=int, default=200)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.chunk_size_mb < 1:
        parser.error("--chunk-size-mb must be at least 1")
    input_root = args.input_root.resolve()
    if not input_root.is_dir():
        parser.error(f"input root does not exist: {input_root}")

    requirements = evaluation_requirements(INDEX_DIRECTORIES[args.dataset])
    writer = ChunkWriter(args.output_root.resolve(), args.chunk_size_mb * 1_000_000)
    converters = {
        "dl3dv": convert_dl3dv,
        "scannet": convert_scannet,
        "scannetpp": convert_scannetpp,
    }
    found = converters[args.dataset](input_root, requirements, args.workers, writer)
    require_scene_coverage(found, set(requirements))
    writer.finish()
    print(
        f"converted {writer.scene_count} scenes and {writer.frame_count} frames "
        f"into {writer.chunk_index} chunks",
        flush=True,
    )


if __name__ == "__main__":
    main()
