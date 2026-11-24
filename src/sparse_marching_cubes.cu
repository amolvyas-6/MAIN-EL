#include <neural-graphics-primitives/sparse_marching_cubes.h>
#include <neural-graphics-primitives/nerf_device.cuh>

#include <thrust/scan.h>
#include <thrust/transform_reduce.h>
#include <thrust/system/cuda/execution_policy.h>

namespace ngp {
namespace {

__constant__ int8_t triangle_table[256][16] =
#include "marching_cubes_table.inl"
;

constexpr int reduction_threads = 128;
constexpr uint32_t evaluation_batch = 1 << 20;

struct Lattice {
	ivec3 resolution;
	ivec3 tiles;
	int size;
	BoundingBox box;
	mat3 to_world;

	__device__ ivec3 tile_origin(uint32_t index) const {
		int id = index;
		return ivec3{id % tiles.x, id / tiles.x % tiles.y, id / (tiles.x * tiles.y)} * size;
	}

	__device__ vec3 sample_point(ivec3 point) const {
		return to_world * (vec3(point) / vec3(resolution - 1) * (box.max - box.min) + box.min);
	}
};

__device__ ivec3 local_point(uint32_t index, int side) {
	int id = index;
	return {id % side, id / side % side, id / (side * side)};
}

// Use the same inclusive occupancy threshold as the dense sample mask.
// Test a coarse superset of all voxels the tile and its halo can touch.
__global__ void occupancy_flags(
	uint32_t count, Lattice lattice, const float* occupancy, uint32_t max_mip, uint32_t* flags
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count) {
		return;
	}
	ivec3 origin = lattice.tile_origin(i);
	ivec3 end = min(origin + lattice.size, lattice.resolution - 1);
	vec3 lower = lattice.sample_point(origin);
	vec3 upper = lower;
	for (int corner = 1; corner < 8; ++corner) {
		vec3 pos = lattice.sample_point({corner & 1 ? end.x : origin.x, corner & 2 ? end.y : origin.y, corner & 4 ? end.z : origin.z});
		lower = min(lower, pos);
		upper = max(upper, pos);
	}
	float nearest = max(max(lower - 0.5f, 0.5f - upper));
	float farthest = max(max(abs(lower - 0.5f), abs(upper - 0.5f)));
	for (uint32_t mip = 0; mip <= max_mip; ++mip) {
		float radius = scalbnf(0.5f, int(mip));
		if ((mip < max_mip && nearest > radius + 1e-5f) || (mip > 0 && farthest < radius * 0.5f - 1e-5f)) {
			continue;
		}
		float scale = NERF_GRIDSIZE() / (2.0f * radius);
		// Expand by one cell for boundary rounding in warp/unwarp.
		ivec3 lo = clamp(ivec3(floor((lower - 0.5f) * scale + NERF_GRIDSIZE() * 0.5f)) - 1, 0, int(NERF_GRIDSIZE()) - 1);
		ivec3 hi = clamp(ivec3(floor((upper - 0.5f) * scale + NERF_GRIDSIZE() * 0.5f)) + 1, 0, int(NERF_GRIDSIZE()) - 1);
		for (int z = lo.z; z <= hi.z; ++z) {
			for (int y = lo.y; y <= hi.y; ++y) {
				for (int x = lo.x; x <= hi.x; ++x) {
					if (!(occupancy[grid_mip_offset(mip) + morton3D(x, y, z)] < NERF_MIN_OPTICAL_THICKNESS())) {
						flags[i] = 1;
						return;
					}
				}
			}
		}
	}
	flags[i] = 0;
}

__global__ void scatter_tiles(uint32_t count, const uint32_t* offsets, uint32_t* tiles) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < count && offsets[i + 1] > offsets[i]) {
		tiles[offsets[i]] = i;
	}
}

uint32_t scan_total(cudaStream_t stream, GPUMemory<uint32_t>& offsets, uint32_t count, uint32_t max_per_item = 1) {
	if (uint64_t(count) * max_per_item >= std::numeric_limits<uint32_t>::max()) {
		uint64_t total = thrust::transform_reduce(
			thrust::cuda::par.on(stream), offsets.data(), offsets.data() + count,
			[] __host__ __device__ (uint32_t value) -> uint64_t { return value; }, uint64_t(0), thrust::plus<uint64_t>{}
		);
		if (total >= std::numeric_limits<uint32_t>::max() - BATCH_SIZE_GRANULARITY) {
			throw std::runtime_error{"Sparse mesh output exceeds the 32-bit index budget."};
		}
	}
	CUDA_CHECK_THROW(cudaMemsetAsync(offsets.data() + count, 0, sizeof(uint32_t), stream));
	thrust::exclusive_scan(thrust::cuda::par.on(stream), offsets.data(), offsets.data() + count + 1, offsets.data());
	uint32_t total;
	CUDA_CHECK_THROW(cudaMemcpyAsync(&total, offsets.data() + count, sizeof(total), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	return total;
}

__global__ void sample_tiles(
	uint32_t count, Lattice lattice, BoundingBox train_aabb, const uint32_t* tiles,
	uint32_t offset, uint32_t total, NerfPosition* positions
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count) {
		return;
	}
	int halo = lattice.size + 1;
	int samples_per_tile = halo * halo * halo;
	uint32_t sample = min(i + offset, total - 1);
	ivec3 point = lattice.tile_origin(tiles[sample / samples_per_tile]) + local_point(sample % samples_per_tile, halo);
	vec3 pos = lattice.sample_point(min(point, lattice.resolution - 1));
	positions[i] = {warp_position(pos, train_aabb), warp_dt(MIN_CONE_STEPSIZE())};
}

__global__ void surface_flags(uint32_t count, int samples, float threshold, const float* density, uint32_t* active) {
	__shared__ uint32_t signs[reduction_threads];
	// Grid stride allows a large number of tiles while bounding launch size.
	for (uint32_t tile = blockIdx.x; tile < count; tile += gridDim.x) {
		uint32_t sides = 0;
		for (int i = threadIdx.x; i < samples; i += blockDim.x) {
			sides |= density[size_t(tile) * samples + i] > threshold ? 1u : 2u;
		}
		signs[threadIdx.x] = sides;
		__syncthreads();
		for (int stride = reduction_threads / 2; stride > 0; stride /= 2) {
			if (threadIdx.x < stride) {
				signs[threadIdx.x] |= signs[threadIdx.x + stride];
			}
			__syncthreads();
		}
		if (threadIdx.x == 0) {
			active[tile] = signs[0] == 3;
		}
		__syncthreads();
	}
}

__device__ ivec3 corner_offset(int corner) {
	return {(corner ^ (corner >> 1)) & 1, (corner >> 1) & 1, (corner >> 2) & 1};
}

__device__ float corner_density(const float* density, ivec3 corner, int halo) {
	return density[corner.x + halo * (corner.y + halo * corner.z)];
}

__global__ void classify(
	uint32_t count, Lattice lattice, const uint32_t* tiles, const uint32_t* surface,
	const float* densities, float threshold, uint8_t* cases, uint32_t* offsets
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count) {
		return;
	}
	int cells = lattice.size * lattice.size * lattice.size;
	uint32_t tile = surface[i / cells];
	ivec3 local = local_point(i % cells, lattice.size);
	ivec3 point = lattice.tile_origin(tiles[tile]) + local;
	int halo = lattice.size + 1;
	uint32_t mask = 0;
	if (max(point - lattice.resolution + 2) <= 0) {
		const float* density = densities + size_t(tile) * halo * halo * halo;
		for (int corner = 0; corner < 8; ++corner) {
			mask |= uint32_t(corner_density(density, local + corner_offset(corner), halo) > threshold) << corner;
		}
	}
	cases[i] = mask;
	uint32_t vertices = 0;
	for (; vertices < 15 && triangle_table[mask][vertices] >= 0; vertices += 3) {}
	offsets[i] = vertices;
}

__global__ void classify_edges(
	uint32_t count, Lattice lattice, const uint32_t* tiles, const uint32_t* surface, const float* densities,
	float threshold, uint8_t* masks, uint32_t* offsets
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count) {
		return;
	}
	int halo = lattice.size + 1;
	uint32_t samples = halo * halo * halo;
	uint32_t tile = surface[i / samples];
	ivec3 local = local_point(i % samples, halo);
	ivec3 point = lattice.tile_origin(tiles[tile]) + local;
	const float* density = densities + size_t(tile) * samples;
	bool inside = density[i % samples] > threshold;
	uint32_t mask = 0;
	if (max(point - lattice.resolution + 1) <= 0) {
		for (int axis = 0; axis < 3; ++axis) {
			ivec3 next = local;
			++next[axis];
			if (next[axis] < halo && point[axis] + 1 < lattice.resolution[axis]) {
				mask |= uint32_t(inside != (corner_density(density, next, halo) > threshold)) << axis;
			}
		}
	}
	masks[i] = mask;
	offsets[i] = __popc(mask);
}

__global__ void emit_vertices(
	uint32_t count, Lattice lattice, const uint32_t* tiles, const uint32_t* surface, const float* densities,
	float threshold, const uint8_t* masks, const uint32_t* offsets, vec3* vertices
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count || !masks[i]) {
		return;
	}
	int halo = lattice.size + 1;
	uint32_t samples = halo * halo * halo;
	uint32_t tile = surface[i / samples];
	ivec3 local = local_point(i % samples, halo);
	ivec3 point = lattice.tile_origin(tiles[tile]) + local;
	const float* density = densities + size_t(tile) * samples;
	float da = density[i % samples];
	vec3 scale = (lattice.box.max - lattice.box.min) / vec3(lattice.resolution - 1);
	uint32_t vertex = offsets[i];
	for (int axis = 0; axis < 3; ++axis) {
		if (!(masks[i] & (1u << axis))) {
			continue;
		}
		ivec3 next = local;
		++next[axis];
		float db = corner_density(density, next, halo);
		vec3 p = vec3(point);
		p[axis] += (threshold - da) / (db - da);
		vertices[vertex++] = lattice.to_world * (p * scale + lattice.box.min);
	}
}

__global__ void emit_faces(
	uint32_t count, int tile_size, const uint8_t* cases, const uint32_t* face_offsets,
	const uint8_t* edge_masks, const uint32_t* edge_offsets, uint32_t* indices
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count || face_offsets[i] == face_offsets[i + 1]) {
		return;
	}
	int cells = tile_size * tile_size * tile_size;
	int halo = tile_size + 1;
	ivec3 local = local_point(i % cells, tile_size);
	uint32_t sample_base = i / cells * halo * halo * halo;
	uint32_t start = face_offsets[i];
	for (uint32_t j = start; j < face_offsets[i + 1]; ++j) {
		int edge = triangle_table[cases[i]][j - start];
		int first = edge < 8 ? edge : edge - 8;
		int second = edge < 8 ? (edge & 4) + ((edge + 1) & 3) : edge - 4;
		ivec3 start_point = local + min(corner_offset(first), corner_offset(second));
		uint32_t sample = sample_base + start_point.x + halo * (start_point.y + halo * start_point.z);
		int axis = edge >= 8 ? 2 : edge & 1;
		indices[j] = edge_offsets[sample] + __popc(edge_masks[sample] & ((1u << axis) - 1));
	}
}

} // namespace

void sparse_marching_cubes_gpu(
	cudaStream_t stream, BoundingBox render_aabb, mat3 render_aabb_to_local, BoundingBox train_aabb,
	ivec3 resolution, float threshold, int tile_size, const float* occupancy, uint32_t max_mip,
	const std::function<void(const NerfPosition*, float*, uint32_t)>& evaluate,
	GPUMemory<vec3>& vertices, GPUMemory<uint32_t>& indices
) {
	if (tile_size != 8 && tile_size != 16) {
		throw std::invalid_argument{"Sparse marching cubes tile size must be 8 or 16."};
	}
	if (!occupancy || !std::isfinite(threshold) || threshold <= -10000.0f || max(resolution) > 1024) {
		throw std::invalid_argument{"Sparse marching cubes requires a trained NeRF, a finite iso level above -10000, and a resolution up to 1024."};
	}
	ivec3 tiles = (resolution + tile_size - 2) / tile_size;
	Lattice lattice{resolution, tiles, tile_size, render_aabb, transpose(render_aabb_to_local)};
	uint32_t tile_count = product(tiles);
	GPUMemory<uint32_t> occupancy_offsets(tile_count + 1);
	linear_kernel(occupancy_flags, 0, stream, tile_count, lattice, occupancy, max_mip, occupancy_offsets.data());
	uint32_t candidate_count = scan_total(stream, occupancy_offsets, tile_count);
	GPUMemory<uint32_t> candidates(candidate_count);
	linear_kernel(scatter_tiles, 0, stream, tile_count, occupancy_offsets.data(), candidates.data());

	vertices.resize(0);
	indices.resize(0);
	if (!candidate_count) {
		return;
	}

	int halo = tile_size + 1;
	uint32_t samples_per_tile = halo * halo * halo;
	uint32_t total = candidate_count * samples_per_tile;
	GPUMemory<float> densities(next_multiple(total, BATCH_SIZE_GRANULARITY));
	GPUMemory<NerfPosition> positions(min(evaluation_batch, next_multiple(total, BATCH_SIZE_GRANULARITY)));
	for (uint32_t offset = 0; offset < total; offset += evaluation_batch) {
		uint32_t batch = next_multiple(min(total - offset, evaluation_batch), BATCH_SIZE_GRANULARITY);
		linear_kernel(sample_tiles, 0, stream, batch, lattice, train_aabb, candidates.data(), offset, total, positions.data());
		evaluate(positions.data(), densities.data() + offset, batch);
	}
	GPUMemory<uint32_t> surface_offsets(candidate_count + 1);
	surface_flags<<<min(candidate_count, 65535u), reduction_threads, 0, stream>>>(
		candidate_count, samples_per_tile, threshold, densities.data(), surface_offsets.data()
	);
	uint32_t surface_count = scan_total(stream, surface_offsets, candidate_count);
	if (!surface_count) {
		return;
	}
	GPUMemory<uint32_t> surfaces(surface_count);
	linear_kernel(scatter_tiles, 0, stream, candidate_count, surface_offsets.data(), surfaces.data());
	uint32_t cell_count = surface_count * tile_size * tile_size * tile_size;
	GPUMemory<uint32_t> offsets(cell_count + 1);
	GPUMemory<uint8_t> cases(cell_count);
	linear_kernel(classify, 0, stream, cell_count, lattice, candidates.data(), surfaces.data(), densities.data(), threshold, cases.data(), offsets.data());
	uint32_t count = scan_total(stream, offsets, cell_count, 15);
	if (!count) {
		return;
	}
	uint32_t edge_count = surface_count * samples_per_tile;
	GPUMemory<uint32_t> edge_offsets(edge_count + 1);
	GPUMemory<uint8_t> edge_masks(edge_count);
	linear_kernel(classify_edges, 0, stream, edge_count, lattice, candidates.data(), surfaces.data(), densities.data(), threshold,
		edge_masks.data(), edge_offsets.data());
	uint32_t vertex_count = scan_total(stream, edge_offsets, edge_count, 3);
	vertices.resize(next_multiple(vertex_count, BATCH_SIZE_GRANULARITY));
	vertices.memset(0);
	indices.resize(count);
	linear_kernel(emit_vertices, 0, stream, edge_count, lattice, candidates.data(), surfaces.data(), densities.data(), threshold,
		edge_masks.data(), edge_offsets.data(), vertices.data());
	linear_kernel(emit_faces, 0, stream, cell_count, tile_size, cases.data(), offsets.data(),
		edge_masks.data(), edge_offsets.data(), indices.data());
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}

} // namespace ngp
