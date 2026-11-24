"""Write the indexed, vertex-colored mesh returned by pyngp as GLB."""

import json
from pathlib import Path
import struct

import numpy as np


def write_glb(path, mesh):
    faces = np.asarray(mesh["F"], dtype=np.int64)
    if not len(faces):
        raise ValueError("The iso level produced no mesh. Try another density threshold.")
    used, remap = np.unique(faces[:, ::-1], return_inverse=True)
    vertices = np.asarray(mesh["V"][used], dtype="<f4")
    if not np.isfinite(vertices).all():
        raise ValueError("The extracted mesh contains invalid vertices.")
    # Triangle-soup extraction gives adjacent triangles identical edge vertices.
    # Weld them before export and accumulate area-weighted vertex normals.
    vertices, first, weld = np.unique(vertices, axis=0, return_index=True, return_inverse=True)
    faces = weld[remap.reshape(-1)].reshape(-1, 3).astype("<u4")
    normals = np.zeros_like(vertices)
    triangles = vertices[faces]
    weighted = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], weighted)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0
    normals[valid] /= lengths[valid, None]
    normals[~valid] = (0, 1, 0)
    srgb = np.clip(np.nan_to_num(mesh["C"][used[first]]), 0, 1)
    colors = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4).astype("<f4")

    payload = bytearray()
    views = []
    accessors = []
    arrays = [(vertices, "VEC3", 5126), (normals, "VEC3", 5126),
              (colors, "VEC3", 5126), (faces.ravel(), "SCALAR", 5125)]
    for index, (array, kind, component) in enumerate(arrays):
        views.append({"buffer": 0, "byteOffset": len(payload), "byteLength": array.nbytes,
                      "target": 34963 if index == 3 else 34962})
        accessors.append({"bufferView": index, "componentType": component,
                          "count": len(array), "type": kind})
        payload.extend(array.tobytes())
    accessors[0].update(min=vertices.min(axis=0).tolist(), max=vertices.max(axis=0).tolist())

    document = {
        "asset": {"version": "2.0", "generator": "NeRF2Mesh"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2},
                                   "indices": 3, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"metallicFactor": 0, "roughnessFactor": 0.9}}],
        "buffers": [{"byteLength": len(payload)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    metadata = json.dumps(document, separators=(",", ":")).encode()
    metadata += b" " * (-len(metadata) % 4)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".glb.tmp")
    with temporary.open("wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, 28 + len(metadata) + len(payload)))
        handle.write(struct.pack("<I4s", len(metadata), b"JSON"))
        handle.write(metadata)
        handle.write(struct.pack("<I4s", len(payload), b"BIN\0"))
        handle.write(payload)
    temporary.replace(path)
    return len(faces)
