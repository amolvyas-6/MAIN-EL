"""Compare both GPU transforms with a brute-force Euclidean reference."""

from pathlib import Path
import sys

import numpy as np
import pyngp as ngp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.baseline import write_rows


def brute_force(mask):
    seeds = np.argwhere(mask)
    if not len(seeds):
        return np.full(mask.shape, np.inf, dtype=np.float32)
    points = np.indices(mask.shape).reshape(3, -1).T
    differences = points[:, None] - seeds[None]
    return np.sqrt(np.min(np.sum(differences ** 2, axis=-1), axis=1)).astype(np.float32).reshape(mask.shape)


def compare(mask, expected, fixture):
    exact, exact_ms = ngp.measure_distance_transform(mask, False)
    flooded, jfa_ms = ngp.measure_distance_transform(mask, True)
    np.testing.assert_array_equal(exact, expected, err_msg=fixture)
    assert np.array_equal(np.isinf(flooded), np.isinf(expected)), fixture
    finite = np.isfinite(expected)
    errors = flooded[finite] - expected[finite]
    assert np.all(errors >= 0), fixture
    source = "distance_transform", fixture, str(mask.shape[0])
    rows = [
        [*source, variant, i, "gpu_time", value, "ms"]
        for variant, times in (("exact", exact_ms), ("jfa", jfa_ms))
        for i, value in enumerate(times)
    ]
    if errors.size:
        rows.extend([
            [*source, "jfa", -1, "max_overestimate", float(errors.max()), "voxels"],
            [*source, "jfa", -1, "mean_overestimate", float(errors.mean()), "voxels"],
            [*source, "jfa", -1, "wrong_voxels", int(np.count_nonzero(errors)), "voxels"],
        ])
    return rows


def main():
    rng = np.random.default_rng(1337)
    fixtures = [
        ("empty", np.zeros((8, 8, 8), dtype=bool)),
        ("full", np.ones((8, 8, 8), dtype=bool)),
        ("single_voxel", np.ones((1, 1, 1), dtype=bool)),
    ]
    fixtures += [(f"random_{n}", rng.random((n, n, n)) < 0.065) for n in (5, 8, 16, 24)]
    rows = []
    for name, mask in fixtures:
        rows.extend(compare(mask, brute_force(mask), name))

    # Full occupancy-grid resolution, with exact analytic ground truth.
    mask = np.zeros((128, 128, 128), dtype=bool)
    mask[63, 53, 91] = True
    axes = np.ogrid[:128, :128, :128]
    expected = np.sqrt(sum((axis - coordinate) ** 2 for axis, coordinate in zip(axes, (63, 53, 91)))).astype(np.float32)
    rows.extend(compare(mask, expected, "single_seed_128"))
    path = Path("results/distance_transform.csv")
    write_rows(path, rows)
    print(f"Exact EDT passed {len(fixtures) + 1} fixtures. Timings and JFA errors saved to {path}.")


if __name__ == "__main__":
    main()
