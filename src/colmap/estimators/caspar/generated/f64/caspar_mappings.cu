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
__launch_bounds__(block_size, 1) void ConstPinholeFocalAndExtra_stacked_to_caspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 2] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholeFocalAndExtra_caspar_to_stacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
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

cudaError_t ConstPinholeFocalAndExtra_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholeFocalAndExtra_stacked_to_caspar_kernel<<<num_blocks,
                                                       block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPinholeFocalAndExtra_caspar_to_stacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholeFocalAndExtra_caspar_to_stacked_kernel<<<num_blocks,
                                                       block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholePrincipalPoint_stacked_to_caspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 2] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPinholePrincipalPoint_caspar_to_stacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
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

cudaError_t ConstPinholePrincipalPoint_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholePrincipalPoint_stacked_to_caspar_kernel<<<num_blocks,
                                                        block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPinholePrincipalPoint_caspar_to_stacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPinholePrincipalPoint_caspar_to_stacked_kernel<<<num_blocks,
                                                        block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstPixel_stacked_to_caspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 2] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPixel_caspar_to_stacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
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

cudaError_t ConstPixel_stacked_to_caspar(const double* stacked_data,
                                         double* cas_data,
                                         const unsigned int cas_stride,
                                         const unsigned int cas_offset,
                                         const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPixel_stacked_to_caspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPixel_caspar_to_stacked(const double* cas_data,
                                         double* stacked_data,
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
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 3];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 3] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];

    out_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    out_ptr[0] = data[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPoint_caspar_to_stacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 3];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    data[0] = in_ptr[0];
    stacked_local_ptr[2] = data[0];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 3];
  }
}

cudaError_t ConstPoint_stacked_to_caspar(const double* stacked_data,
                                         double* cas_data,
                                         const unsigned int cas_stride,
                                         const unsigned int cas_offset,
                                         const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPoint_stacked_to_caspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPoint_caspar_to_stacked(const double* cas_data,
                                         double* stacked_data,
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
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 7];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 7 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 7;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 7] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];
    data[1] = stacked_local_ptr[3];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[6];

    out_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    out_ptr[0] = data[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstPose_caspar_to_stacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 7];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[2] = data[0];
    stacked_local_ptr[3] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[4] = data[0];
    stacked_local_ptr[5] = data[1];
    in_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    data[0] = in_ptr[0];
    stacked_local_ptr[6] = data[0];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 7 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 7;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 7];
  }
}

cudaError_t ConstPose_stacked_to_caspar(const double* stacked_data,
                                        double* cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPose_stacked_to_caspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstPose_caspar_to_stacked(const double* cas_data,
                                        double* stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstPose_caspar_to_stacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialFocalAndExtra_stacked_to_caspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 2] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialFocalAndExtra_caspar_to_stacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
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

cudaError_t ConstSimpleRadialFocalAndExtra_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialFocalAndExtra_stacked_to_caspar_kernel<<<num_blocks,
                                                            block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstSimpleRadialFocalAndExtra_caspar_to_stacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialFocalAndExtra_caspar_to_stacked_kernel<<<num_blocks,
                                                            block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialPrincipalPoint_stacked_to_caspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 2] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void ConstSimpleRadialPrincipalPoint_caspar_to_stacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
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

cudaError_t ConstSimpleRadialPrincipalPoint_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialPrincipalPoint_stacked_to_caspar_kernel<<<num_blocks,
                                                             block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ConstSimpleRadialPrincipalPoint_caspar_to_stacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ConstSimpleRadialPrincipalPoint_caspar_to_stacked_kernel<<<num_blocks,
                                                             block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void PinholeFocalAndExtra_stacked_to_caspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 2] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void PinholeFocalAndExtra_caspar_to_stacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
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

cudaError_t PinholeFocalAndExtra_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholeFocalAndExtra_stacked_to_caspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t PinholeFocalAndExtra_caspar_to_stacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholeFocalAndExtra_caspar_to_stacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void PinholePrincipalPoint_stacked_to_caspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 2] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void PinholePrincipalPoint_caspar_to_stacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
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

cudaError_t PinholePrincipalPoint_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholePrincipalPoint_stacked_to_caspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t PinholePrincipalPoint_caspar_to_stacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  PinholePrincipalPoint_caspar_to_stacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__ __launch_bounds__(block_size, 1) void Point_stacked_to_caspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 3];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 3] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];

    out_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    out_ptr[0] = data[0];
  }
}

__global__ __launch_bounds__(block_size, 1) void Point_caspar_to_stacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 3];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    data[0] = in_ptr[0];
    stacked_local_ptr[2] = data[0];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 3];
  }
}

cudaError_t Point_stacked_to_caspar(const double* stacked_data,
                                    double* cas_data,
                                    const unsigned int cas_stride,
                                    const unsigned int cas_offset,
                                    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  Point_stacked_to_caspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t Point_caspar_to_stacked(const double* cas_data,
                                    double* stacked_data,
                                    const unsigned int cas_stride,
                                    const unsigned int cas_offset,
                                    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  Point_caspar_to_stacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__ __launch_bounds__(block_size, 1) void Pose_stacked_to_caspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 7];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 7 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 7;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 7] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];
    data[1] = stacked_local_ptr[3];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[6];

    out_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    out_ptr[0] = data[0];
  }
}

__global__ __launch_bounds__(block_size, 1) void Pose_caspar_to_stacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 7];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 7;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[2] = data[0];
    stacked_local_ptr[3] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[4] = data[0];
    stacked_local_ptr[5] = data[1];
    in_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    data[0] = in_ptr[0];
    stacked_local_ptr[6] = data[0];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 7 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 7;
       target += blockDim.x) {
    stacked_data[target] =
        stacked_data_local[target - (blockIdx.x * blockDim.x) * 7];
  }
}

cudaError_t Pose_stacked_to_caspar(const double* stacked_data,
                                   double* cas_data,
                                   const unsigned int cas_stride,
                                   const unsigned int cas_offset,
                                   const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  Pose_stacked_to_caspar_kernel<<<num_blocks, block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t Pose_caspar_to_stacked(const double* cas_data,
                                   double* stacked_data,
                                   const unsigned int cas_stride,
                                   const unsigned int cas_offset,
                                   const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  Pose_caspar_to_stacked_kernel<<<num_blocks, block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialFocalAndExtra_stacked_to_caspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 2] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialFocalAndExtra_caspar_to_stacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
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

cudaError_t SimpleRadialFocalAndExtra_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialFocalAndExtra_stacked_to_caspar_kernel<<<num_blocks,
                                                       block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t SimpleRadialFocalAndExtra_caspar_to_stacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialFocalAndExtra_caspar_to_stacked_kernel<<<num_blocks,
                                                       block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialPrincipalPoint_stacked_to_caspar_kernel(
    const double* const __restrict__ stacked_data,
    double* const __restrict__ cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2;
       target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 2] =
        stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] =
        reinterpret_cast<double2*>(data)[0];
  }
}

__global__
__launch_bounds__(block_size, 1) void SimpleRadialPrincipalPoint_caspar_to_stacked_kernel(
    const double* const __restrict__ cas_data,
    double* const __restrict__ stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] =
        reinterpret_cast<const double2*>(in_ptr)[0];
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

cudaError_t SimpleRadialPrincipalPoint_stacked_to_caspar(
    const double* stacked_data,
    double* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialPrincipalPoint_stacked_to_caspar_kernel<<<num_blocks,
                                                        block_size>>>(
      stacked_data, cas_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t SimpleRadialPrincipalPoint_caspar_to_stacked(
    const double* cas_data,
    double* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  SimpleRadialPrincipalPoint_caspar_to_stacked_kernel<<<num_blocks,
                                                        block_size>>>(
      cas_data, stacked_data, cas_stride, cas_offset, num_objects);

  return cudaGetLastError();
}

}  // namespace caspar