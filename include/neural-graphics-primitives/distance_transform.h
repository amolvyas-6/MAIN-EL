#pragma once

#include <tiny-cuda-nn/gpu_memory.h>

namespace ngp {

// Output is row-major, x fastest, in voxel units. Input is either a row-major byte
// mask or one or more cascades of packed Morton-ordered occupancy bits.
void distance_transform(
	cudaStream_t stream, const uint8_t* occupancy, int side, int cascades, bool packed_morton,
	bool jump_flood, tcnn::GPUMemory<float>& distances
);

}
