"""Generate a small, fixed-pose NeRF scene and an independent test split."""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def unit(value):
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-8)


def make_split(directory, split, views, offset, size):
    relative = Path(split)
    (directory / relative).mkdir(parents=True, exist_ok=True)
    focal = size * 1.1
    x, y = np.meshgrid((np.arange(size) + 0.5 - size / 2) / focal,
                       (size / 2 - np.arange(size) - 0.5) / focal)
    frames = []
    for i in range(views):
        angle = 2 * np.pi * (i + offset) / views
        origin = np.array([2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.45 + 0.25 * np.sin(angle * 3)])
        back = unit(origin)
        right = unit(np.cross([0, 0, 1], back))
        up = np.cross(back, right)
        direction = unit(x[..., None] * right + y[..., None] * up - back)
        depth = np.full((size, size), np.inf)
        image = np.full((size, size, 3), 12.0)
        for center, radius, color in [
            ([-0.29, 0.04, 0], 0.39, [238, 114, 49]),
            ([0.33, 0.02, 0.09], 0.31, [37, 162, 219]),
            ([0.06, -0.27, -0.21], 0.22, [225, 196, 71]),
        ]:
            delta = origin - center
            b = direction @ delta
            discriminant = b * b - (delta @ delta - radius * radius)
            t = -b - np.sqrt(np.maximum(discriminant, 0))
            hits = (discriminant >= 0) & (t > 0) & (t < depth)
            normal = (delta + t[hits, None] * direction[hits]) / radius
            illumination = 0.27 + 0.73 * np.maximum(0, normal @ unit(np.array([0.4, -0.4, 1.0])))
            image[hits] = illumination[:, None] * color
            depth[hits] = t[hits]
        path = relative / f"frame-{i:02}.png"
        Image.fromarray(image.astype(np.uint8)).save(directory / path)
        pose = np.eye(4)
        pose[:3] = np.column_stack([right, up, back, origin])
        frames.append({"file_path": str(path), "transform_matrix": pose.tolist()})
    transform = {"w": size, "h": size, "fl_x": focal, "fl_y": focal, "cx": size / 2, "cy": size / 2,
                 "camera_angle_x": 2 * np.arctan(size / (2 * focal)), "aabb_scale": 1, "frames": frames}
    (directory / f"{split}.json").write_text(json.dumps(transform))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    make_split(args.directory, "train", 24, 0, 320)
    make_split(args.directory, "test", 8, 0.5, 320)
