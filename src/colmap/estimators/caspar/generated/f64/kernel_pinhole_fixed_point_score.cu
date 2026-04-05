#include "kernel_pinhole_fixed_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_point_score_kernel(double* pose,
                                     unsigned int pose_num_alloc,
                                     SharedIndex* pose_indices,
                                     double* calib,
                                     unsigned int calib_num_alloc,
                                     SharedIndex* calib_indices,
                                     double* pixel,
                                     unsigned int pixel_num_alloc,
                                     double* point,
                                     unsigned int point_num_alloc,
                                     double* const out_rTr,
                                     size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27;
  load_shared<2, double, double>(
      calib, 2 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
  };
  load_shared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r0, r5);
  };
  __syncthreads();
  load_shared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9);
  };
  load_shared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r10, r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = -2.00000000000000000e+00;
    r13 = r11 * r12;
  };
  load_shared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r14, r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = 2.00000000000000000e+00;
    r17 = r14 * r16;
    r18 = r15 * r17;
    r19 = fma(r10, r13, r18);
    r19 = fma(r9, r19, r6);
    read_idx_1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r6);
    r20 = r15 * r11;
    r21 = r10 * r17;
    r20 = fma(r16, r20, r21);
    r22 = r10 * r10;
    r22 = r12 * r22;
    r23 = 1.00000000000000000e+00;
    r24 = r15 * r15;
    r24 = fma(r12, r24, r23);
    r25 = r22 + r24;
    r19 = fma(r6, r20, r19);
    r19 = fma(r8, r25, r19);
    r25 = r0 * r19;
    r20 = 1.00000000000000008e-15;
  };
  load_shared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r26);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r27 = r15 * r10;
    r27 = r27 * r16;
    r17 = fma(r11, r17, r27);
    r17 = fma(r9, r17, r26);
    r21 = fma(r15, r13, r21);
    r26 = r14 * r14;
    r26 = r26 * r12;
    r24 = r26 + r24;
    r17 = fma(r8, r21, r17);
    r17 = fma(r6, r24, r17);
    r24 = copysign(1.0, r17);
    r24 = fma(r20, r24, r17);
    r24 = 1.0 / r24;
    r2 = fma(r24, r25, r2);
    r4 = fma(r3, r4, r1);
    r3 = r10 * r11;
    r3 = fma(r16, r3, r18);
    r3 = fma(r8, r3, r7);
    r13 = fma(r14, r13, r27);
    r22 = r23 + r22;
    r22 = r22 + r26;
    r3 = fma(r6, r13, r3);
    r3 = fma(r9, r22, r3);
    r22 = r5 * r3;
    r4 = fma(r24, r22, r4);
    r4 = fma(r4, r4, r2 * r2);
  };
  sum_store<double>(out_rTr_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r4);
  sum_flush_final<double>(out_rTr_local, out_rTr, 1);
}

void pinhole_fixed_point_score(double* pose,
                               unsigned int pose_num_alloc,
                               SharedIndex* pose_indices,
                               double* calib,
                               unsigned int calib_num_alloc,
                               SharedIndex* calib_indices,
                               double* pixel,
                               unsigned int pixel_num_alloc,
                               double* point,
                               unsigned int point_num_alloc,
                               double* const out_rTr,
                               size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_fixed_point_score_kernel<<<n_blocks, 1024>>>(pose,
                                                       pose_num_alloc,
                                                       pose_indices,
                                                       calib,
                                                       calib_num_alloc,
                                                       calib_indices,
                                                       pixel,
                                                       pixel_num_alloc,
                                                       point,
                                                       point_num_alloc,
                                                       out_rTr,
                                                       problem_size);
}

}  // namespace caspar