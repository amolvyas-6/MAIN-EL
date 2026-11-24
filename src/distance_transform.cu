#include <neural-graphics-primitives/distance_transform.h>
#include <neural-graphics-primitives/nerf_device.cuh>

namespace ngp {
namespace {

constexpr int no_seed = 1 << 28;
constexpr int lines_per_block = 32;

__device__ bool occupied(const uint8_t* occupancy, int index, int side, bool packed) {
	if (!packed) {
		return occupancy[index] != 0;
	}
	int volume = side * side * side;
	int cell = index % volume;
	int morton = morton3D(cell % side, cell / side % side, cell / (side * side));
	return occupancy[(index / volume * volume + morton) / 8] & (1 << (morton % 8));
}

__global__ void init_transform(uint32_t count, const uint8_t* occupancy, int side, bool packed, bool jfa, int* scratch) {
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < count) {
		bool seed = occupied(occupancy, i, side, packed);
		scratch[i] = jfa ? (seed ? int(i) % (side * side * side) : -1) : (seed ? 0 : no_seed);
	}
}

// Each thread builds one lower envelope. Put the lines in shared memory
// with line number as the fastest index so neighbouring threads share banks evenly.
__global__ void transform_axis(int lines, int side, int axis, const int* input, int* output) {
	extern __shared__ int workspace[];
	int* sites = workspace + threadIdx.x;
	int* starts = workspace + side * lines_per_block + threadIdx.x;
	int line = blockIdx.x * blockDim.x + threadIdx.x;
	if (line >= lines) {
		return;
	}
	int u = line % side;
	int v = line / side % side;
	int base = line / (side * side) * side * side * side;
	int stride = axis == 0 ? 1 : (axis == 1 ? side : side * side);
	base += axis == 0 ? u * side + v * side * side : (axis == 1 ? u + v * side * side : u + v * side);

	int top = -1;
	for (int q = 0; q < side; ++q) {
		int cost = input[base + q * stride];
		if (cost == no_seed) {
			continue;
		}
		int start = 0;
		while (top >= 0) {
			int p = sites[top * lines_per_block];
			int numerator = (cost + q * q) - (input[base + p * stride] + p * p);
			int denominator = 2 * (q - p);
			// Integer ceiling of the crossing point, including negative values.
			start = numerator / denominator + (numerator % denominator > 0);
			if (start > starts[top * lines_per_block]) {
				break;
			}
			--top;
		}
		++top;
		sites[top * lines_per_block] = q;
		starts[top * lines_per_block] = top == 0 ? 0 : start;
	}

	int active = 0;
	for (int x = 0; x < side; ++x) {
		while (active < top && starts[(active + 1) * lines_per_block] <= x) {
			++active;
		}
		int distance = no_seed;
		if (top >= 0) {
			int p = sites[active * lines_per_block];
			distance = input[base + p * stride] + (x - p) * (x - p);
		}
		output[base + x * stride] = distance;
	}
}

__device__ int squared_distance(int seed, ivec3 point, int side) {
	if (seed < 0) {
		return no_seed;
	}
	ivec3 delta = ivec3{seed % side, seed / side % side, seed / (side * side)} - point;
	return delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
}

__global__ void flood(uint32_t count, int side, int step, const int* input, int* output) {
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= count) {
		return;
	}
	int volume = side * side * side;
	int cell = i % volume;
	ivec3 point{cell % side, cell / side % side, cell / (side * side)};
	int base = i / volume * volume;
	int best_seed = input[i];
	int best_dist = squared_distance(best_seed, point, side);
	for (int z = -step; z <= step; z += step) {
		for (int y = -step; y <= step; y += step) {
			for (int x = -step; x <= step; x += step) {
				ivec3 p = point + ivec3{x, y, z};
				if (p.x < 0 || p.y < 0 || p.z < 0 || p.x >= side || p.y >= side || p.z >= side) {
					continue;
				}
				int seed = input[base + p.x + side * (p.y + side * p.z)];
				int dist = squared_distance(seed, point, side);
				if (dist < best_dist) {
					best_seed = seed;
					best_dist = dist;
				}
			}
		}
	}
	output[i] = best_seed;
}

__global__ void finish_transform(uint32_t count, int side, bool jfa, const int* input, float* output) {
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= count) {
		return;
	}
	int cell = i % (side * side * side);
	int distance = jfa ? squared_distance(input[i], {cell % side, cell / side % side, cell / (side * side)}, side) : input[i];
	output[i] = distance == no_seed ? std::numeric_limits<float>::infinity() : __fsqrt_rn(float(distance));
}

} // namespace

void distance_transform(
	cudaStream_t stream, const uint8_t* occupancy, int side, int cascades, bool packed_morton,
	bool jump_flood, GPUMemory<float>& distances
) {
	if (side < 1 || side > 128 || cascades < 1 || cascades > 8 || (packed_morton && side != 128)) {
		throw std::invalid_argument{"Invalid distance transform dimensions."};
	}
	uint32_t count = side * side * side * cascades;
	distances.resize(count);
	GPUMemoryArena::Allocation allocation;
	auto buffers = allocate_workspace_and_distribute<int, int>(stream, &allocation, count, count);
	int* input = std::get<0>(buffers);
	int* output = std::get<1>(buffers);
	linear_kernel(init_transform, 0, stream, count, occupancy, side, packed_morton, jump_flood, input);
	if (jump_flood) {
		int step = 1;
		while (step * 2 < side) {
			step *= 2;
		}
		for (; step > 0 && side > 1; step /= 2) {
			linear_kernel(flood, 0, stream, count, side, step, input, output);
			std::swap(input, output);
		}
	} else {
		int lines = side * side * cascades;
		for (int axis = 0; axis < 3; ++axis) {
			transform_axis<<<div_round_up(lines, lines_per_block), lines_per_block, 2 * side * lines_per_block * sizeof(int), stream>>>(
				lines, side, axis, input, output
			);
			std::swap(input, output);
		}
	}
	linear_kernel(finish_transform, 0, stream, count, side, jump_flood, input, distances.data());
	CUDA_CHECK_THROW(cudaGetLastError());
}

} // namespace ngp
