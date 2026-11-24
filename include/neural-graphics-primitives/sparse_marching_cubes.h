#pragma once

#include <neural-graphics-primitives/bounding_box.cuh>
#include <tiny-cuda-nn/gpu_memory.h>
#include <functional>

namespace ngp {

struct NerfPosition;

void sparse_marching_cubes_gpu(
	cudaStream_t stream, BoundingBox render_aabb, mat3 render_aabb_to_local, BoundingBox train_aabb,
	ivec3 resolution, float threshold, int tile_size, const float* occupancy, uint32_t max_mip,
	const std::function<void(const NerfPosition*, float*, uint32_t)>& evaluate,
	tcnn::GPUMemory<vec3>& vertices, tcnn::GPUMemory<uint32_t>& indices
);

}
