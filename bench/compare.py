"""Check paired outputs and record CUDA-event and NVML measurements."""

import argparse
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np
import pyngp as ngp
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import compute_error, linear_to_srgb, write_image
from baseline import NvmlDevice, measure_with_nvml, memory_rows, write_rows


RENDER_VARIANTS = ("no_skip", "exact_skip")
RENDER_CHECK_VARIANTS = (*RENDER_VARIANTS, "voxel_stable")


def configure_render(model, variant):
    model.disable_ray_skipping = variant == "no_skip"
    model.stable_ray_steps = True
    model.use_distance_skip = variant == "exact_skip"


def power_source():
    supplies = Path("/sys/class/power_supply")
    try:
        if any(path.read_text().strip() == "1" for path in supplies.glob("*/online")):
            return "external_power"
        if any(path.read_text().strip() == "Discharging" for path in supplies.glob("*/status")):
            return "battery"
    except OSError:
        pass
    return "unknown_power"


def load(scene, snapshot):
    model = ngp.Testbed()
    model.root_dir = str(ROOT)
    if scene:
        model.load_training_data(scene)
    model.load_snapshot(snapshot)
    model.shall_train = False
    return model


def report_time(label, times):
    print(f"{label}: {statistics.median(times):.4f} ms median ({min(times):.4f}–{max(times):.4f} ms)", flush=True)


def render_pair(args):
    start_power = power_source()
    model = load(args.test_scene or args.scene, args.snapshot)
    model.jit_fusion = args.jit
    model.snap_to_pixel_centers = True
    model.count_empty_advances = True
    n_views = model.nerf.training.dataset.n_images
    if not n_views:
        raise ValueError("The dataset has no views.")
    rows = []
    output = Path("results") / f"render_{args.name}_{'fused' if args.jit else 'unfused'}_no_skip"
    output.mkdir(parents=True, exist_ok=True)
    resolution = f"{args.width}x{args.height}"
    reference_label = "held_out" if args.test_scene else "training_view"
    if not args.jit:
        args.name += " (unfused)"
    count = min(args.views, n_views)
    for run, view in enumerate(np.linspace(0, n_views - 1, count, dtype=int)):
        model.set_camera_to_training_view(int(view))
        model.render_ground_truth = True
        ref = model.render(args.width, args.height, 1, True)
        model.render_ground_truth = False
        captures = {}
        for variant in RENDER_CHECK_VARIANTS:
            configure_render(model, variant)
            image = model.render(args.width, args.height, 1, True)
            if model.jit_fusion != args.jit:
                raise RuntimeError("JIT fusion fell back during a requested fused comparison.")
            captures[variant] = image
            if variant not in RENDER_VARIANTS:
                continue
            rgb = np.clip(linear_to_srgb(image[..., :3]), 0, 1)
            target = np.clip(linear_to_srgb(ref[..., :3]), 0, 1)
            mse = float(np.mean((rgb - target) ** 2))
            rows.extend([
                ["ray_quality", args.name, resolution, variant, run, f"{reference_label}_psnr", -10 * np.log10(mse) if mse else "inf", "dB"],
                ["ray_quality", args.name, resolution, variant, run, f"{reference_label}_ssim", compute_error("SSIM", rgb, target), "ratio"],
                ["ray_quality", args.name, resolution, variant, run, "empty_advances_per_pixel_ray", np.mean(model.empty_advances()), "advances"],
            ])
            if run == 0:
                write_image(str(output / f"{variant}.png"), image)
        for label, variant in (("stable", "voxel_stable"), ("no_skip", "no_skip")):
            delta = np.abs(captures["exact_skip"] - captures[variant])
            mse = float(np.mean(delta[..., :3] ** 2))
            print(f"view {view} against {label}: max RGBA error {delta.max():.6g}; RGB RMSE {np.sqrt(mse):.6g}", flush=True)
            # Apply the existing per-pixel regression check to fixed-grid voxel
            # skipping. The sample-by-sample ablation can expose isolated
            # pixel differences, so record its peak error and
            # check the image-wide error against the same RMSE tolerance.
            if not np.all(np.isfinite(delta)) or np.sqrt(mse) > 2e-4 or (label == "stable" and delta.max() > 0.005):
                raise RuntimeError(f"Distance skipping changed the {label} render beyond tolerance.")
            rows.extend([
                ["ray_quality", args.name, resolution, "exact_skip", run, f"{label}_max_rgba_error", delta.max(), "ratio"],
                ["ray_quality", args.name, resolution, "exact_skip", run, f"{label}_psnr", -10 * np.log10(mse) if mse else "inf", "dB"],
                ["ray_quality", args.name, resolution, "exact_skip", run, f"{label}_ssim", compute_error("SSIM", captures["exact_skip"][..., :3], captures[variant][..., :3]), "ratio"],
            ])

    model.count_empty_advances = False
    model.set_camera_to_training_view(0)
    variant_times = {variant: [] for variant in RENDER_VARIANTS}
    for variant in RENDER_VARIANTS:
        configure_render(model, variant)
        model.benchmark_render(args.width, args.height, 2, 1)  # Discard all three frames.
    for run in range(args.runs):
        # Rotate the order so changes in GPU clocks affect each mode across the run.
        first = run % len(RENDER_VARIANTS)
        for variant in RENDER_VARIANTS[first:] + RENDER_VARIANTS[:first]:
            configure_render(model, variant)
            elapsed, = model.benchmark_render(args.width, args.height, 0, 1)
            if model.jit_fusion != args.jit:
                raise RuntimeError("JIT fusion fell back during a requested fused timing run.")
            variant_times[variant].append(elapsed)
            rows.append(["ray_render", args.name, resolution, variant, run, "gpu_time", elapsed, "ms"])
    for variant, times in variant_times.items():
        report_time(variant, times)
    end_power = power_source()
    power = start_power if start_power == end_power else "changed_power"
    rows.append(["ray_conditions", args.name, resolution, power, -1, "power_source", 1, "flag"])
    write_rows(output.with_suffix(".csv"), rows)


def mesh_worker(args):
    model = load(None, args.snapshot)
    model.sparse_mesh_tile = args.tile
    resolution = int(args.resolution)
    try:
        if args.worker == "geometry":
            mesh = model.compute_marching_cubes_mesh([resolution] * 3, thresh=args.threshold)
            faces = mesh["F"]
            if not len(faces):
                raise RuntimeError("The chosen iso level produced no triangles.")
            shape = (len(faces), 3, 3)
            triangles = np.lib.format.open_memmap(args.path, mode="w+", dtype=np.float32, shape=shape)
            for start in range(0, len(faces), 1_000_000):
                stop = min(len(faces), start + 1_000_000)
                chunk = mesh["V"][faces[start:stop]]
                if not np.isfinite(chunk).all():
                    raise RuntimeError("Mesh contains non-finite coordinates.")
                triangles[start:stop] = chunk
            triangles.flush()
        else:
            device = NvmlDevice(args.gpu)
            try:
                times, memory = measure_with_nvml(device, 10, lambda: model.benchmark_dense_marching_cubes(
                    resolution, args.threshold, 3, args.runs, tile=args.tile))
            finally:
                device.close()
            variant = "sparse" if args.tile else "dense"
            grid = f"{resolution}³"
            report_time(f"{variant} {grid}", times)
            rows = [["marching_cubes", args.name, grid, variant, i, "gpu_time", time, "ms"] for i, time in enumerate(times)]
            rows += [["marching_cubes", *row[1:]] for row in memory_rows(args.name, grid, variant, memory, 10)]
            write_rows(Path(args.path), rows)
    except RuntimeError as error:
        if any(message in str(error).lower() for message in ("out of memory", "memory allocation failed", "cuda_error_out_of_memory")):
            print(f"GPU allocation failed at {resolution}³: {error}", file=sys.stderr)
            return 88
        raise
    return 0


def triangle_keys(triangles):
    words = np.ascontiguousarray(triangles).view(np.uint32).reshape(-1, 9)
    keys = np.zeros(len(words), dtype=np.uint64)
    for field in words.T:
        keys *= np.uint64(1099511628211)
        keys ^= field
    return keys


def corresponding_triangles(sample, sparse):
    keys, unique, inverse = np.unique(triangle_keys(sample), return_index=True, return_inverse=True)
    found = np.zeros(len(keys), dtype=bool)
    expected = sample[unique]
    matched = np.empty_like(expected)
    for start in range(0, len(sparse), 1_000_000):
        chunk = sparse[start:start + 1_000_000]
        values = triangle_keys(chunk)
        indices = np.minimum(np.searchsorted(keys, values), len(keys) - 1)
        candidates = np.flatnonzero(keys[indices] == values)
        selected = indices[candidates]
        exact = np.all(chunk[candidates] == expected[selected], axis=(1, 2))
        found[selected[exact]] = True
        matched[selected[exact]] = chunk[candidates[exact]]
        if found.all():
            return matched[inverse]
    # Allow small arithmetic differences to be compared by geometric proximity.
    matching = cKDTree(sparse.mean(axis=1)).query(sample.mean(axis=1), workers=2)[1]
    return sparse[matching]


def geometry_match(dense, sparse):
    if not np.isfinite(dense).all() or not np.isfinite(sparse).all():
        raise RuntimeError("Mesh contains non-finite coordinates.")
    diag = float(np.linalg.norm(dense.max(axis=(0, 1)) - dense.min(axis=(0, 1))))
    if diag == 0 or len(dense) != len(sparse):
        raise RuntimeError(f"Triangle counts differ: dense {len(dense)}, sparse {len(sparse)}.")
    # Pair triangles by centroid. Common barycentric samples remove sampling jitter
    # when both algorithms emit the same triangle. The KD queries are bidirectional.
    rng = np.random.default_rng(1337)
    area = np.empty(len(dense), dtype=np.float64)
    for start in range(0, len(dense), 1_000_000):
        end = min(len(dense), start + 1_000_000)
        tile = dense[start:end]
        area[start:end] = np.linalg.norm(np.cross(tile[:, 1] - tile[:, 0], tile[:, 2] - tile[:, 0]), axis=1)
    total_area = float(area.sum())
    if total_area <= 0:
        raise RuntimeError("The dense mesh has no surface area.")
    area /= total_area
    first = dense[rng.choice(len(dense), 100_000, p=area)]
    del area
    second = corresponding_triangles(first, sparse)
    nearest = ((first[:, :, None] - second[:, None]) ** 2).sum(axis=-1).argmin(axis=-1)
    second = np.take_along_axis(second, nearest[:, :, None], axis=1)
    u, v = rng.random((2, len(first), 1))
    root = np.sqrt(u)
    weights = np.concatenate((1 - root, root * (1 - v), root * v), axis=1)[..., None]
    a = (first * weights).sum(axis=1)
    b = (second * weights).sum(axis=1)
    distances = [cKDTree(target).query(points, workers=2)[0] for points, target in ((a, b), (b, a))]
    chamfer = float((distances[0].mean() + distances[1].mean()) * 0.5)
    vertex_error = float(np.linalg.norm(first - second, axis=-1).max())
    if chamfer > diag * 1e-4 or vertex_error > diag * 1e-3:
        raise RuntimeError(f"Mesh agreement failed: Chamfer {chamfer:g}, max paired vertex error {vertex_error:g}.")
    return chamfer


def mesh_pair(args):
    import csv
    import tempfile

    output = Path("results")
    output.mkdir(exist_ok=True)
    rows = []
    for grid in args.resolutions:
        with tempfile.TemporaryDirectory(prefix="mesh-check-", dir=output) as name:
            directory = Path(name)

            def run(variant, mode, extension):
                path = directory / f"{variant}.{extension}"
                log = output / f"mesh_{args.name}_{grid}_{variant}_{mode}.log"
                with log.open("w") as handle:
                    code = subprocess.run([
                        sys.executable, __file__, "worker", "--worker", mode, "--name", args.name, "--scene", args.scene,
                        "--snapshot", args.snapshot, "--resolution", str(grid), "--path", str(path),
                        "--tile", str(args.tile if variant == "sparse" else 0),
                        "--runs", str(args.runs), "--threshold", str(args.threshold), "--gpu", str(args.gpu),
                    ], stdout=handle, stderr=subprocess.STDOUT).returncode
                if code == 88:
                    print(f"{variant} {grid}³ {mode}: GPU out of memory ({log})", flush=True)
                    return None
                if code:
                    raise RuntimeError(f"{variant} {grid}³ {mode} exited with {code}. See {log}.")
                if mode == "timing":
                    with path.open() as handle:
                        values = [float(row["value"]) for row in csv.DictReader(handle) if row["metric"] == "gpu_time"]
                    report_time(f"{variant} {grid}³", values)
                return path

            geometry = {variant: run(variant, "geometry", "npy") for variant in ("dense", "sparse")}
            for variant, path in geometry.items():
                if path is None:
                    rows.append(["marching_cubes", args.name, f"{grid}³", variant, -1, "oom", 1, "boolean"])

            if geometry["dense"] and geometry["sparse"]:
                dense = np.load(geometry["dense"], mmap_mode="r")
                sparse = np.load(geometry["sparse"], mmap_mode="r")
                chamfer = geometry_match(dense, sparse)
                print(f"{grid}³: equal triangle counts ({len(dense)}), Chamfer {chamfer:.6g}", flush=True)
                rows.append(["mesh_quality", args.name, f"{grid}³", "sparse", -1, "chamfer", chamfer, "scene units"])
                for variant, triangles in [("dense", dense), ("sparse", sparse)]:
                    rows.append(["mesh_quality", args.name, f"{grid}³", variant, -1, "triangles", len(triangles), "triangles"])
                del dense, sparse
            elif geometry["sparse"] and not geometry["dense"]:
                print(f"{grid}³: dense OOM. Sparse timing has no paired dense correctness check at this resolution.", flush=True)

            for variant, path in geometry.items():
                if path is None:
                    continue
                timings = run(variant, "timing", "csv")
                if timings is None:
                    rows.append(["marching_cubes", args.name, f"{grid}³", variant, -1, "oom", 1, "boolean"])
                else:
                    with timings.open() as handle:
                        rows.extend(list(row.values()) for row in csv.DictReader(handle))
        write_rows(Path("results") / f"mesh_{args.name}.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["render", "mesh", "worker"])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--test-scene", help="Use this independent camera/ground-truth split for render comparisons.")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[256, 512, 768, 1024])
    parser.add_argument("--tile", type=int, choices=[0, 8, 16], default=8)
    parser.add_argument("--threshold", type=float, default=2.5)
    parser.add_argument("--worker", choices=["geometry", "timing"])
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--path")
    args = parser.parse_args()
    if args.runs < 10 or args.views < 1:
        parser.error("Provide at least 10 timed runs and one view.")
    if args.mode == "render":
        render_pair(args)
    elif args.mode == "mesh":
        mesh_pair(args)
    else:
        return mesh_worker(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
