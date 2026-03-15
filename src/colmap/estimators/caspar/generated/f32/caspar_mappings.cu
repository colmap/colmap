#include "caspar_mappings.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// We use shared memory to improve the memory access.
// A smaller block size of 32 allows for larger nodetypes.
constexpr int block_size = 32;

namespace caspar {

__global__
__launch_bounds__(block_size, 1) void ConstPinholeCalib_stacked_to_caspar_kernel(
    const float* const __restrict__ stacked_data,
    float* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 4];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 4 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 4;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 4] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float* stacked_local_ptr = stacked_data_local + threadIdx.x * 4;
    float* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4*>(out_ptr)[0] = reinterpret_cast<float4*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholeCalib_caspar_to_stacked_kernel(
    const float* const __restrict__ cas_data,
    float* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 4];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float* stacked_local_ptr = stacked_data_local + threadIdx.x * 4;
    const float* in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4*>(data)[0] =
        reinterpret_cast<const float4*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
    stacked_local_ptr[3] = data[3];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 4 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 4;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 4];
  }
}

cudaError_t ConstPinholeCalib_stacked_to_caspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholeCalib_stacked_to_caspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPinholeCalib_caspar_to_stacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholeCalib_caspar_to_stacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstPixel_stacked_to_caspar_kernel(
    const float* const __restrict__ stacked_data,
    float* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 2];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 2] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    float* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2*>(out_ptr)[0] = reinterpret_cast<float2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPixel_caspar_to_stacked_kernel(
    const float* const __restrict__ cas_data,
    float* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const float* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2*>(data)[0] =
        reinterpret_cast<const float2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 2];
  }
}

cudaError_t ConstPixel_stacked_to_caspar(const float* stacked_data,
                                         float* cas_data,
                                         const unsigned int cas_stride,
                                         const unsigned int cas_offset,
                                         const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPixel_stacked_to_caspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPixel_caspar_to_stacked(const float* cas_data,
                                         float* stacked_data,
                                         const unsigned int cas_stride,
                                         const unsigned int cas_offset,
                                         const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPixel_caspar_to_stacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstPoint_stacked_to_caspar_kernel(
    const float* const __restrict__ stacked_data,
    float* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 3];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 3] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float* stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    float* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4*>(out_ptr)[0] = reinterpret_cast<float4*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPoint_caspar_to_stacked_kernel(
    const float* const __restrict__ cas_data,
    float* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 3];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float* stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    const float* in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4*>(data)[0] =
        reinterpret_cast<const float4*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 3];
  }
}

cudaError_t ConstPoint_stacked_to_caspar(const float* stacked_data,
                                         float* cas_data,
                                         const unsigned int cas_stride,
                                         const unsigned int cas_offset,
                                         const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPoint_stacked_to_caspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPoint_caspar_to_stacked(const float* cas_data,
                                         float* stacked_data,
                                         const unsigned int cas_stride,
                                         const unsigned int cas_offset,
                                         const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPoint_caspar_to_stacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstPose_stacked_to_caspar_kernel(
    const float* const __restrict__ stacked_data,
    float* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 7];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 7 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 7;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 7] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float* stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    float* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4*>(out_ptr)[0] = reinterpret_cast<float4*>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];
    data[2] = stacked_local_ptr[6];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4*>(out_ptr)[0] = reinterpret_cast<float4*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPose_caspar_to_stacked_kernel(
    const float* const __restrict__ cas_data,
    float* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 7];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float* stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    const float* in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4*>(data)[0] =
        reinterpret_cast<const float4*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
    stacked_local_ptr[3] = data[3];
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4*>(data)[0] =
        reinterpret_cast<const float4*>(in_ptr)[0];
    stacked_local_ptr[4] = data[0];
    stacked_local_ptr[5] = data[1];
    stacked_local_ptr[6] = data[2];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 7 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 7;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 7];
  }
}

cudaError_t ConstPose_stacked_to_caspar(const float* stacked_data,
                                        float* cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPose_stacked_to_caspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPose_caspar_to_stacked(const float* cas_data,
                                        float* stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPose_caspar_to_stacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialCalib_stacked_to_caspar_kernel(
    const float* const __restrict__ stacked_data,
    float* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 4];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 4 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 4;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 4] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float* stacked_local_ptr = stacked_data_local + threadIdx.x * 4;
    float* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4*>(out_ptr)[0] = reinterpret_cast<float4*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialCalib_caspar_to_stacked_kernel(
    const float* const __restrict__ cas_data,
    float* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 4];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float* stacked_local_ptr = stacked_data_local + threadIdx.x * 4;
    const float* in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4*>(data)[0] =
        reinterpret_cast<const float4*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
    stacked_local_ptr[3] = data[3];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 4 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 4;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 4];
  }
}

cudaError_t ConstSimpleRadialCalib_stacked_to_caspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialCalib_stacked_to_caspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstSimpleRadialCalib_caspar_to_stacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialCalib_caspar_to_stacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void PinholeCalib_stacked_to_caspar_kernel(
    const float* const __restrict__ stacked_data,
    float* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 4];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 4 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 4;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 4] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float* stacked_local_ptr = stacked_data_local + threadIdx.x * 4;
    float* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4*>(out_ptr)[0] = reinterpret_cast<float4*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void PinholeCalib_caspar_to_stacked_kernel(
    const float* const __restrict__ cas_data,
    float* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 4];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float* stacked_local_ptr = stacked_data_local + threadIdx.x * 4;
    const float* in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4*>(data)[0] =
        reinterpret_cast<const float4*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
    stacked_local_ptr[3] = data[3];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 4 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 4;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 4];
  }
}

cudaError_t PinholeCalib_stacked_to_caspar(const float* stacked_data,
                                           float* cas_data,
                                           const unsigned int cas_stride,
                                           const unsigned int cas_offset,
                                           const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholeCalib_stacked_to_caspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t PinholeCalib_caspar_to_stacked(const float* cas_data,
                                           float* stacked_data,
                                           const unsigned int cas_stride,
                                           const unsigned int cas_offset,
                                           const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholeCalib_caspar_to_stacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__ __launch_bounds__(block_size, 1) void Point_stacked_to_caspar_kernel(
    const float* const __restrict__ stacked_data,
    float* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 3];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 3] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float* stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    float* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4*>(out_ptr)[0] = reinterpret_cast<float4*>(data)[0];
  }
}

__global__ __launch_bounds__(block_size, 1) void Point_caspar_to_stacked_kernel(
    const float* const __restrict__ cas_data,
    float* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 3];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float* stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    const float* in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4*>(data)[0] =
        reinterpret_cast<const float4*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 3];
  }
}

cudaError_t Point_stacked_to_caspar(const float* stacked_data,
                                    float* cas_data,
                                    const unsigned int cas_stride,
                                    const unsigned int cas_offset,
                                    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  Point_stacked_to_caspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t Point_caspar_to_stacked(const float* cas_data,
                                    float* stacked_data,
                                    const unsigned int cas_stride,
                                    const unsigned int cas_offset,
                                    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  Point_caspar_to_stacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__ __launch_bounds__(block_size, 1) void Pose_stacked_to_caspar_kernel(
    const float* const __restrict__ stacked_data,
    float* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 7];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 7 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 7;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 7] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float* stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    float* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4*>(out_ptr)[0] = reinterpret_cast<float4*>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];
    data[2] = stacked_local_ptr[6];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4*>(out_ptr)[0] = reinterpret_cast<float4*>(data)[0];
  }
}

__global__ __launch_bounds__(block_size, 1) void Pose_caspar_to_stacked_kernel(
    const float* const __restrict__ cas_data,
    float* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 7];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float* stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    const float* in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4*>(data)[0] =
        reinterpret_cast<const float4*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
    stacked_local_ptr[3] = data[3];
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4*>(data)[0] =
        reinterpret_cast<const float4*>(in_ptr)[0];
    stacked_local_ptr[4] = data[0];
    stacked_local_ptr[5] = data[1];
    stacked_local_ptr[6] = data[2];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 7 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 7;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 7];
  }
}

cudaError_t Pose_stacked_to_caspar(const float* stacked_data,
                                   float* cas_data,
                                   const unsigned int cas_stride,
                                   const unsigned int cas_offset,
                                   const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  Pose_stacked_to_caspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t Pose_caspar_to_stacked(const float* cas_data,
                                   float* stacked_data,
                                   const unsigned int cas_stride,
                                   const unsigned int cas_offset,
                                   const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  Pose_caspar_to_stacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialCalib_stacked_to_caspar_kernel(
    const float* const __restrict__ stacked_data,
    float* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 4];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 4 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 4;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 4] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float* stacked_local_ptr = stacked_data_local + threadIdx.x * 4;
    float* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4*>(out_ptr)[0] = reinterpret_cast<float4*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialCalib_caspar_to_stacked_kernel(
    const float* const __restrict__ cas_data,
    float* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 4];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float* stacked_local_ptr = stacked_data_local + threadIdx.x * 4;
    const float* in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4*>(data)[0] =
        reinterpret_cast<const float4*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
    stacked_local_ptr[3] = data[3];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 4 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 4;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 4];
  }
}

cudaError_t SimpleRadialCalib_stacked_to_caspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialCalib_stacked_to_caspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t SimpleRadialCalib_caspar_to_stacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialCalib_caspar_to_stacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

}  // namespace caspar