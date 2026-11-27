"""COLMAP subprocess pipeline and headless NeRF artifact generation."""

from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.getenv("NGP_BUILD_DIR", ROOT / "build")).resolve()


def run(directory, *arguments, env=None):
    with (directory / "worker.log").open("a", encoding="utf-8") as log:
        log.write(f"\n{arguments[0]} {arguments[1]}\n")
        log.flush()
        subprocess.run(arguments, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL, check=True)


def build_poses(directory):
    run(directory, sys.executable, str(Path(__file__).resolve()), "poses", str(directory))
    return len(json.loads((directory / "transforms.json").read_text())["frames"])


def reconstruct(directory):
    import pycolmap

    database = directory / "capture.db"
    images = directory / "images"
    sparse = directory / "sparse"
    sparse.mkdir()
    pycolmap.extract_features(database, images, device=pycolmap.Device.cpu)
    pycolmap.match_exhaustive(database, device=pycolmap.Device.cpu)
    models = pycolmap.incremental_mapping(database, images, sparse)
    best = max(models.values(), key=lambda model: model.num_reg_images(), default=None)
    if best is None or best.num_reg_images() < 3:
        print("COLMAP registered fewer than three photos. Capture more overlapping views.", file=sys.stderr)
        return 1
    poses = directory / "poses"
    poses.mkdir()
    best.write_text(poses)
    run(directory, sys.executable, str(ROOT / "scripts/colmap2nerf.py"),
        "--images", "images", "--text", str(poses), "--out", "transforms.json", "--aabb_scale", "4")
    return 0


def train(directory, steps):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(BUILD) + os.pathsep + environment.get("PYTHONPATH", "")
    with (directory / "worker.log").open("a", encoding="utf-8") as log:
        subprocess.run([sys.executable, str(ROOT / "scripts/run.py"), "--scene", str(directory / "transforms.json"),
                        "--n_steps", str(steps), "--save_snapshot", str(directory / "scene.ingp")],
                       cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL, check=True)


def export(directory, resolution):
    # Keep export headless even when pyngp was built with desktop GUI support.
    # Otherwise Testbed can probe the desktop OpenGL device during initialization.
    environment = os.environ.copy()
    environment.pop("DISPLAY", None)
    environment.pop("WAYLAND_DISPLAY", None)
    # The child also releases its CUDA allocations before the next job.
    run(directory, sys.executable, str(Path(__file__).resolve()), "export", str(directory), str(resolution), env=environment)
    return json.loads((directory / "artifacts.json").read_text())


def render_previews(testbed, directory):
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp
    from common import write_image

    dataset = testbed.nerf.training.dataset
    if dataset.n_images < 2:
        return []

    cameras = []
    for index in range(dataset.n_images):
        testbed.set_camera_to_training_view(index)
        cameras.append(testbed.camera_matrix.copy())

    positions = np.asarray(cameras)[:, :, 3]
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    distances[distances <= 1e-6] = np.inf
    pairs = sorted({
        tuple(sorted((a, b))) for a, b in enumerate(distances.argmin(axis=1)) if np.isfinite(distances[a, b])
    })
    if not pairs:
        return []

    testbed.render_ground_truth = False
    frames = []
    for index in np.linspace(0, len(pairs) - 1, min(6, len(pairs)), dtype=int):
        a, b = pairs[index]
        rotation = Rotation.from_matrix(np.stack((cameras[a][:, :3], cameras[b][:, :3])))
        midpoint = np.empty((3, 4), dtype=np.float32)
        midpoint[:, :3] = Slerp([0, 1], rotation)(0.5).as_matrix()
        midpoint[:, 3] = (positions[a] + positions[b]) * 0.5

        # Use one source camera's lens settings with the interpolated pose.
        testbed.set_camera_to_training_view(a)
        testbed.camera_matrix = midpoint
        width, height = dataset.metadata[a].resolution
        scale = 720 / max(width, height)
        image = testbed.render(max(1, round(width * scale)), max(1, round(height * scale)), 8, True)
        name = f"view-{len(frames)}.png"
        write_image(str(directory / name), image)
        frames.append(name)
    return frames


def export_artifacts(directory, resolution):
    sys.path[:0] = [str(BUILD), str(ROOT / "scripts")]
    import pyngp as ngp
    from mesh_glb import write_glb

    testbed = ngp.Testbed()
    testbed.root_dir = str(ROOT)
    testbed.load_training_data(str(directory / "transforms.json"))
    testbed.load_snapshot(str(directory / "scene.ingp"))
    testbed.shall_train = False
    testbed.use_distance_skip = True
    testbed.sparse_mesh_tile = 8
    mesh = testbed.compute_marching_cubes_mesh([resolution] * 3, thresh=2.5)
    triangles = write_glb(directory / "mesh.glb", mesh)
    frames = render_previews(testbed, directory)
    (directory / "artifacts.json").write_text(json.dumps({"triangles": triangles, "frames": frames}))


if __name__ == "__main__":
    if sys.argv[1] == "poses":
        sys.exit(reconstruct(Path(sys.argv[2])))
    export_artifacts(Path(sys.argv[2]), int(sys.argv[3]))
