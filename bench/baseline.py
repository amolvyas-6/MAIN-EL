#!/usr/bin/env python3
"""Record Phase 0 Instant-NGP baseline render, mesh, and memory measurements."""

import argparse
import csv
import ctypes
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import pyngp as ngp


CSV_COLUMNS = ["experiment", "scene", "resolution", "variant", "run", "metric", "value", "unit"]
MIB = 1024 * 1024


class NvmlMemoryInfo(ctypes.Structure):
	_fields_ = [
		("total", ctypes.c_ulonglong),
		("free", ctypes.c_ulonglong),
		("used", ctypes.c_ulonglong),
	]


class NvmlDevice:
	"""Minimal NVML wrapper so the benchmark needs no Python package install."""

	def __init__(self, gpu_index: int):
		try:
			self._lib = ctypes.CDLL("libnvidia-ml.so.1")
		except OSError as error:
			raise RuntimeError("Could not load libnvidia-ml.so.1 for NVML memory sampling.") from error

		self._lib.nvmlInit_v2.argtypes = []
		self._lib.nvmlInit_v2.restype = ctypes.c_uint
		self._lib.nvmlShutdown.argtypes = []
		self._lib.nvmlShutdown.restype = ctypes.c_uint
		self._lib.nvmlErrorString.argtypes = [ctypes.c_uint]
		self._lib.nvmlErrorString.restype = ctypes.c_char_p

		try:
			get_handle = self._lib.nvmlDeviceGetHandleByIndex_v2
		except AttributeError:
			get_handle = self._lib.nvmlDeviceGetHandleByIndex
		get_handle.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
		get_handle.restype = ctypes.c_uint

		self._lib.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(NvmlMemoryInfo)]
		self._lib.nvmlDeviceGetMemoryInfo.restype = ctypes.c_uint

		self._check(self._lib.nvmlInit_v2(), "nvmlInit_v2")
		self._handle = ctypes.c_void_p()
		self._check(get_handle(gpu_index, ctypes.byref(self._handle)), "nvmlDeviceGetHandleByIndex")

	def _check(self, status: int, operation: str) -> None:
		if status == 0:
			return
		message = self._lib.nvmlErrorString(status)
		message_text = message.decode("utf-8", errors="replace") if message else f"status {status}"
		raise RuntimeError(f"{operation} failed: {message_text}")

	def used_bytes(self) -> int:
		info = NvmlMemoryInfo()
		self._check(self._lib.nvmlDeviceGetMemoryInfo(self._handle, ctypes.byref(info)), "nvmlDeviceGetMemoryInfo")
		return int(info.used)

	def close(self) -> None:
		if self._lib is not None:
			self._check(self._lib.nvmlShutdown(), "nvmlShutdown")
			self._lib = None


@dataclass
class MemoryStats:
	start_bytes: int
	peak_bytes: int
	sample_count: int


class NvmlPeakSampler:
	def __init__(self, device: NvmlDevice, interval_ms: float):
		self._device = device
		self._interval_seconds = interval_ms / 1000.0
		self._values = []
		self._lock = threading.Lock()
		self._stop = threading.Event()
		self._thread = None
		self._error = None

	def _sample_once(self) -> None:
		try:
			value = self._device.used_bytes()
			with self._lock:
				self._values.append(value)
		except Exception as error:  # Propagate the sampler failure on stop().
			self._error = error
			self._stop.set()

	def _run(self) -> None:
		while not self._stop.wait(self._interval_seconds):
			self._sample_once()

	def start(self) -> None:
		self._sample_once()
		self._thread = threading.Thread(target=self._run, name="nvml-peak-sampler", daemon=True)
		self._thread.start()

	def stop(self) -> MemoryStats:
		self._stop.set()
		if self._thread is not None:
			self._thread.join()
		self._sample_once()
		if self._error is not None:
			raise RuntimeError("NVML memory sampling failed.") from self._error
		with self._lock:
			if not self._values:
				raise RuntimeError("NVML memory sampler recorded no values.")
			return MemoryStats(self._values[0], max(self._values), len(self._values))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--scene", type=Path, required=True, help="NeRF dataset directory containing transforms.json.")
	parser.add_argument("--snapshot", type=Path, required=True, help="Trained .ingp or .msgpack snapshot to load.")
	parser.add_argument("--output", type=Path, required=True, help="CSV path for raw baseline measurements.")
	parser.add_argument("--mesh-output", type=Path, required=True, help="PLY or OBJ path for the saved dense baseline mesh.")
	parser.add_argument("--scene-name", default="", help="CSV scene label; defaults to the scene directory name.")
	parser.add_argument("--width", type=int, default=800, help="Fixed render width.")
	parser.add_argument("--height", type=int, default=600, help="Fixed render height.")
	parser.add_argument("--view", type=int, default=0, help="Fixed training-view index used for rendering.")
	parser.add_argument("--warmup-runs", type=int, default=3, help="Discarded warm-up runs before each timed series.")
	parser.add_argument("--runs", type=int, default=10, help="Number of raw timed runs; must be at least 10.")
	parser.add_argument("--mesh-res", type=int, default=256, help="Cubic dense marching-cubes resolution; must be divisible by 16.")
	parser.add_argument("--mesh-thresh", type=float, default=2.5, help="Dense marching-cubes density threshold.")
	parser.add_argument("--gpu-index", type=int, default=0, help="Physical NVML GPU index to sample.")
	parser.add_argument("--nvml-poll-ms", type=float, default=10.0, help="NVML device-memory polling period in milliseconds.")
	args = parser.parse_args()

	if args.width <= 0 or args.height <= 0:
		parser.error("--width and --height must be positive.")
	if args.view < 0:
		parser.error("--view must be non-negative.")
	if args.warmup_runs < 0:
		parser.error("--warmup-runs must be non-negative.")
	if args.runs < 10:
		parser.error("--runs must be at least 10 under the project benchmark protocol.")
	if args.mesh_res <= 0 or args.mesh_res % 16 != 0:
		parser.error("--mesh-res must be positive and divisible by 16.")
	if args.gpu_index < 0:
		parser.error("--gpu-index must be non-negative.")
	if args.nvml_poll_ms <= 0:
		parser.error("--nvml-poll-ms must be positive.")
	if not args.scene.is_dir():
		parser.error(f"Scene directory does not exist: {args.scene}")
	if not args.snapshot.is_file():
		parser.error(f"Snapshot does not exist: {args.snapshot}")
	return args


def measure_with_nvml(device: NvmlDevice, interval_ms: float, operation):
	sampler = NvmlPeakSampler(device, interval_ms)
	sampler.start()
	try:
		result = operation()
	except Exception:
		sampler.stop()
		raise
	return result, sampler.stop()


def memory_rows(scene: str, resolution: str, variant: str, stats: MemoryStats, poll_ms: float):
	peak_mib = stats.peak_bytes / MIB
	delta_mib = (stats.peak_bytes - stats.start_bytes) / MIB
	return [
		["baseline", scene, resolution, variant, -1, "nvml_peak_gpu_memory_used", peak_mib, "MiB"],
		["baseline", scene, resolution, variant, -1, "nvml_peak_gpu_memory_delta", delta_mib, "MiB"],
		["baseline", scene, resolution, variant, -1, "nvml_memory_sample_period", poll_ms, "ms"],
		["baseline", scene, resolution, variant, -1, "nvml_memory_sample_count", stats.sample_count, "samples"],
	]


def write_rows(path: Path, rows) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle)
		writer.writerow(CSV_COLUMNS)
		writer.writerows(rows)


def main() -> None:
	args = parse_args()
	repo_root = Path(__file__).resolve().parents[1]
	scene = args.scene_name or args.scene.resolve().name
	render_resolution = f"{args.width}x{args.height}"
	mesh_resolution = f"{args.mesh_res}x{args.mesh_res}x{args.mesh_res}"

	testbed = ngp.Testbed()
	testbed.root_dir = str(repo_root)
	testbed.load_training_data(str(args.scene))
	testbed.load_snapshot(str(args.snapshot))
	testbed.set_camera_to_training_view(args.view)

	device = NvmlDevice(args.gpu_index)
	try:
		render_ms, render_memory = measure_with_nvml(
			device,
			args.nvml_poll_ms,
			lambda: testbed.benchmark_render(args.width, args.height, args.warmup_runs, args.runs),
		)
		mesh_ms, mesh_memory = measure_with_nvml(
			device,
			args.nvml_poll_ms,
			lambda: testbed.benchmark_dense_marching_cubes(
				args.mesh_res, args.mesh_thresh, args.warmup_runs, args.runs
			),
		)

		args.mesh_output.parent.mkdir(parents=True, exist_ok=True)
		testbed.compute_and_save_marching_cubes_mesh(
			str(args.mesh_output), [args.mesh_res, args.mesh_res, args.mesh_res], thresh=args.mesh_thresh
		)
	finally:
		device.close()

	rows = []
	rows.extend(
		["baseline", scene, render_resolution, "baseline", run, "render_gpu_time", float(value), "ms"]
		for run, value in enumerate(render_ms)
	)
	rows.extend(memory_rows(scene, render_resolution, "baseline", render_memory, args.nvml_poll_ms))
	rows.extend(
		["baseline", scene, mesh_resolution, "dense_marching_cubes", run, "dense_mesh_extraction_gpu_time", float(value), "ms"]
		for run, value in enumerate(mesh_ms)
	)
	rows.extend(memory_rows(scene, mesh_resolution, "dense_marching_cubes", mesh_memory, args.nvml_poll_ms))
	write_rows(args.output, rows)

	print(f"Wrote {len(rows)} raw measurements to {args.output}")
	print(f"Saved dense baseline mesh to {args.mesh_output}")
	print("Timing method: CUDA events in the testbed. Memory method: NVML device-memory polling.")


if __name__ == "__main__":
	main()
