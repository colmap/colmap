#include <stdio.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include "caspar_mappings.h"

namespace cg = cooperative_groups;

// We use shared memory to improve the memory access.
// A smaller block size of 32 allows for larger nodetypes.
constexpr int block_size = 32;

namespace caspar {

__global__
__launch_bounds__(block_size, 1) void ConstOpenCVFocalAndExtraStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 6];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 6 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 6;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 6] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 6;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float2 *>(out_ptr)[0] =
        reinterpret_cast<float2 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstOpenCVFocalAndExtraCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 6];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 6;
    const float *in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
    stacked_local_ptr[3] = data[3];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float2 *>(data)[0] =
        reinterpret_cast<const float2 *>(in_ptr)[0];
    stacked_local_ptr[4] = data[0];
    stacked_local_ptr[5] = data[1];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 6 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 6;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 6];
  }
}

cudaError_t ConstOpenCVFocalAndExtraStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstOpenCVFocalAndExtraStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstOpenCVFocalAndExtraCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstOpenCVFocalAndExtraCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstOpenCVPoseStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];
    data[2] = stacked_local_ptr[6];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstOpenCVPoseCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 7];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    const float *in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
    stacked_local_ptr[3] = data[3];
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
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

cudaError_t ConstOpenCVPoseStackedToCaspar(const float *stacked_data,
                                           float *cas_data,
                                           const unsigned int cas_stride,
                                           const unsigned int cas_offset,
                                           const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstOpenCVPoseStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstOpenCVPoseCasparToStacked(const float *cas_data,
                                           float *stacked_data,
                                           const unsigned int cas_stride,
                                           const unsigned int cas_offset,
                                           const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstOpenCVPoseCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstOpenCVPrincipalPointStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(out_ptr)[0] =
        reinterpret_cast<float2 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstOpenCVPrincipalPointCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const float *in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(data)[0] =
        reinterpret_cast<const float2 *>(in_ptr)[0];
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

cudaError_t ConstOpenCVPrincipalPointStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstOpenCVPrincipalPointStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstOpenCVPrincipalPointCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstOpenCVPrincipalPointCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstOpenCVSensorFromRigStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];
    data[2] = stacked_local_ptr[6];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstOpenCVSensorFromRigCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 7];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    const float *in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
    stacked_local_ptr[3] = data[3];
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
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

cudaError_t ConstOpenCVSensorFromRigStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstOpenCVSensorFromRigStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstOpenCVSensorFromRigCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstOpenCVSensorFromRigCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholeFocalStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(out_ptr)[0] =
        reinterpret_cast<float2 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholeFocalCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const float *in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(data)[0] =
        reinterpret_cast<const float2 *>(in_ptr)[0];
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

cudaError_t ConstPinholeFocalStackedToCaspar(const float *stacked_data,
                                             float *cas_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholeFocalStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPinholeFocalCasparToStacked(const float *cas_data,
                                             float *stacked_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholeFocalCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholePoseStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];
    data[2] = stacked_local_ptr[6];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholePoseCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 7];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    const float *in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
    stacked_local_ptr[3] = data[3];
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
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

cudaError_t ConstPinholePoseStackedToCaspar(const float *stacked_data,
                                            float *cas_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholePoseStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPinholePoseCasparToStacked(const float *cas_data,
                                            float *stacked_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholePoseCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholePrincipalPointStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(out_ptr)[0] =
        reinterpret_cast<float2 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholePrincipalPointCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const float *in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(data)[0] =
        reinterpret_cast<const float2 *>(in_ptr)[0];
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

cudaError_t ConstPinholePrincipalPointStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholePrincipalPointStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPinholePrincipalPointCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholePrincipalPointCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholeSensorFromRigStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];
    data[2] = stacked_local_ptr[6];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholeSensorFromRigCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 7];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    const float *in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
    stacked_local_ptr[3] = data[3];
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
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

cudaError_t ConstPinholeSensorFromRigStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholeSensorFromRigStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPinholeSensorFromRigCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholeSensorFromRigCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstPixelStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(out_ptr)[0] =
        reinterpret_cast<float2 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPixelCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const float *in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(data)[0] =
        reinterpret_cast<const float2 *>(in_ptr)[0];
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

cudaError_t ConstPixelStackedToCaspar(const float *stacked_data,
                                      float *cas_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPixelStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPixelCasparToStacked(const float *cas_data,
                                      float *stacked_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPixelCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstPointStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPointCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 3];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    const float *in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
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

cudaError_t ConstPointStackedToCaspar(const float *stacked_data,
                                      float *cas_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPointStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPointCasparToStacked(const float *cas_data,
                                      float *stacked_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPointCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialFocalAndExtraStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(out_ptr)[0] =
        reinterpret_cast<float2 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialFocalAndExtraCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const float *in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(data)[0] =
        reinterpret_cast<const float2 *>(in_ptr)[0];
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

cudaError_t ConstSimpleRadialFocalAndExtraStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialFocalAndExtraStackedToCaspar_kernel<<<num_blocks,
                                                         block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstSimpleRadialFocalAndExtraCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialFocalAndExtraCasparToStacked_kernel<<<num_blocks,
                                                         block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialPoseStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];
    data[2] = stacked_local_ptr[6];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialPoseCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 7];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    const float *in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
    stacked_local_ptr[3] = data[3];
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
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

cudaError_t ConstSimpleRadialPoseStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialPoseStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstSimpleRadialPoseCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialPoseCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialPrincipalPointStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(out_ptr)[0] =
        reinterpret_cast<float2 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialPrincipalPointCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const float *in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(data)[0] =
        reinterpret_cast<const float2 *>(in_ptr)[0];
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

cudaError_t ConstSimpleRadialPrincipalPointStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialPrincipalPointStackedToCaspar_kernel<<<num_blocks,
                                                          block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstSimpleRadialPrincipalPointCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialPrincipalPointCasparToStacked_kernel<<<num_blocks,
                                                          block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialSensorFromRigStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];
    data[2] = stacked_local_ptr[6];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialSensorFromRigCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 7];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    const float *in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
    stacked_local_ptr[3] = data[3];
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
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

cudaError_t ConstSimpleRadialSensorFromRigStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialSensorFromRigStackedToCaspar_kernel<<<num_blocks,
                                                         block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstSimpleRadialSensorFromRigCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialSensorFromRigCasparToStacked_kernel<<<num_blocks,
                                                         block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void OpenCVCalibStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 8];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 8 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 8;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 8] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 8;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];
    data[2] = stacked_local_ptr[6];
    data[3] = stacked_local_ptr[7];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void OpenCVCalibCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 8];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 8;
    const float *in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
    stacked_local_ptr[3] = data[3];
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
    stacked_local_ptr[4] = data[0];
    stacked_local_ptr[5] = data[1];
    stacked_local_ptr[6] = data[2];
    stacked_local_ptr[7] = data[3];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 8 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 8;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 8];
  }
}

cudaError_t OpenCVCalibStackedToCaspar(const float *stacked_data,
                                       float *cas_data,
                                       const unsigned int cas_stride,
                                       const unsigned int cas_offset,
                                       const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  OpenCVCalibStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t OpenCVCalibCasparToStacked(const float *cas_data,
                                       float *stacked_data,
                                       const unsigned int cas_stride,
                                       const unsigned int cas_offset,
                                       const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  OpenCVCalibCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void OpenCVFocalAndExtraStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 6];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 6 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 6;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 6] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 6;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float2 *>(out_ptr)[0] =
        reinterpret_cast<float2 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void OpenCVFocalAndExtraCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 6];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 6;
    const float *in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
    stacked_local_ptr[3] = data[3];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float2 *>(data)[0] =
        reinterpret_cast<const float2 *>(in_ptr)[0];
    stacked_local_ptr[4] = data[0];
    stacked_local_ptr[5] = data[1];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 6 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 6;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 6];
  }
}

cudaError_t OpenCVFocalAndExtraStackedToCaspar(const float *stacked_data,
                                               float *cas_data,
                                               const unsigned int cas_stride,
                                               const unsigned int cas_offset,
                                               const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  OpenCVFocalAndExtraStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t OpenCVFocalAndExtraCasparToStacked(const float *cas_data,
                                               float *stacked_data,
                                               const unsigned int cas_stride,
                                               const unsigned int cas_offset,
                                               const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  OpenCVFocalAndExtraCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void OpenCVPoseStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];
    data[2] = stacked_local_ptr[6];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void OpenCVPoseCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 7];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    const float *in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
    stacked_local_ptr[3] = data[3];
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
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

cudaError_t OpenCVPoseStackedToCaspar(const float *stacked_data,
                                      float *cas_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  OpenCVPoseStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t OpenCVPoseCasparToStacked(const float *cas_data,
                                      float *stacked_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  OpenCVPoseCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void OpenCVPrincipalPointStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(out_ptr)[0] =
        reinterpret_cast<float2 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void OpenCVPrincipalPointCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const float *in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(data)[0] =
        reinterpret_cast<const float2 *>(in_ptr)[0];
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

cudaError_t OpenCVPrincipalPointStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  OpenCVPrincipalPointStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t OpenCVPrincipalPointCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  OpenCVPrincipalPointCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void PinholeCalibStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 4;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void PinholeCalibCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 4];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 4;
    const float *in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
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

cudaError_t PinholeCalibStackedToCaspar(const float *stacked_data,
                                        float *cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholeCalibStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t PinholeCalibCasparToStacked(const float *cas_data,
                                        float *stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholeCalibCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void PinholeFocalStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(out_ptr)[0] =
        reinterpret_cast<float2 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void PinholeFocalCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const float *in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(data)[0] =
        reinterpret_cast<const float2 *>(in_ptr)[0];
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

cudaError_t PinholeFocalStackedToCaspar(const float *stacked_data,
                                        float *cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholeFocalStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t PinholeFocalCasparToStacked(const float *cas_data,
                                        float *stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholeFocalCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void PinholePoseStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];
    data[2] = stacked_local_ptr[6];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void PinholePoseCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 7];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    const float *in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
    stacked_local_ptr[3] = data[3];
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
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

cudaError_t PinholePoseStackedToCaspar(const float *stacked_data,
                                       float *cas_data,
                                       const unsigned int cas_stride,
                                       const unsigned int cas_offset,
                                       const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholePoseStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t PinholePoseCasparToStacked(const float *cas_data,
                                       float *stacked_data,
                                       const unsigned int cas_stride,
                                       const unsigned int cas_offset,
                                       const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholePoseCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void PinholePrincipalPointStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(out_ptr)[0] =
        reinterpret_cast<float2 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void PinholePrincipalPointCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const float *in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(data)[0] =
        reinterpret_cast<const float2 *>(in_ptr)[0];
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

cudaError_t PinholePrincipalPointStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholePrincipalPointStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t PinholePrincipalPointCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholePrincipalPointCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__ __launch_bounds__(block_size, 1) void PointStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
  }
}

__global__ __launch_bounds__(block_size, 1) void PointCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 3];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    const float *in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
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

cudaError_t PointStackedToCaspar(const float *stacked_data, float *cas_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PointStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t PointCasparToStacked(const float *cas_data, float *stacked_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PointCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialCalibStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 4;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialCalibCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 4];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 4;
    const float *in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
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

cudaError_t SimpleRadialCalibStackedToCaspar(const float *stacked_data,
                                             float *cas_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialCalibStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t SimpleRadialCalibCasparToStacked(const float *cas_data,
                                             float *stacked_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialCalibCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialFocalAndExtraStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(out_ptr)[0] =
        reinterpret_cast<float2 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialFocalAndExtraCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const float *in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(data)[0] =
        reinterpret_cast<const float2 *>(in_ptr)[0];
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

cudaError_t SimpleRadialFocalAndExtraStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialFocalAndExtraStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t SimpleRadialFocalAndExtraCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialFocalAndExtraCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialPoseStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];
    data[2] = stacked_local_ptr[2];
    data[3] = stacked_local_ptr[3];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];
    data[2] = stacked_local_ptr[6];

    out_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4 *>(out_ptr)[0] =
        reinterpret_cast<float4 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialPoseCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 7];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    const float *in_ptr;
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    stacked_local_ptr[2] = data[2];
    stacked_local_ptr[3] = data[3];
    in_ptr = cas_data + 4 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<float4 *>(data)[0] =
        reinterpret_cast<const float4 *>(in_ptr)[0];
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

cudaError_t SimpleRadialPoseStackedToCaspar(const float *stacked_data,
                                            float *cas_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialPoseStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t SimpleRadialPoseCasparToStacked(const float *cas_data,
                                            float *stacked_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialPoseCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialPrincipalPointStackedToCaspar_kernel(
    const float *const __restrict__ stacked_data,
    float *const __restrict__ cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
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
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    float *out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(out_ptr)[0] =
        reinterpret_cast<float2 *>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialPrincipalPointCasparToStacked_kernel(
    const float *const __restrict__ cas_data,
    float *const __restrict__ stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {

  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    float data[4] = {0, 0, 0, 0};
    float *stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const float *in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<float2 *>(data)[0] =
        reinterpret_cast<const float2 *>(in_ptr)[0];
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

cudaError_t SimpleRadialPrincipalPointStackedToCaspar(
    const float *stacked_data, float *cas_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialPrincipalPointStackedToCaspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t SimpleRadialPrincipalPointCasparToStacked(
    const float *cas_data, float *stacked_data, const unsigned int cas_stride,
    const unsigned int cas_offset, const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialPrincipalPointCasparToStacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

} // namespace caspar