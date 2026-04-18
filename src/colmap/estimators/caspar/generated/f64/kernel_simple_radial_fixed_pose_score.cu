#include "kernel_simple_radial_fixed_pose_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_pose_score_kernel(double* focal,
                                          unsigned int focal_num_alloc,
                                          SharedIndex* focal_indices,
                                          double* extra_calib,
                                          unsigned int extra_calib_num_alloc,
                                          SharedIndex* extra_calib_indices,
                                          double* point,
                                          unsigned int point_num_alloc,
                                          SharedIndex* point_indices,
                                          double* pixel,
                                          unsigned int pixel_num_alloc,
                                          double* pose,
                                          unsigned int pose_num_alloc,
                                          double* const out_rTr,
                                          size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex focal_indices_loc[1024];
  focal_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex extra_calib_indices_loc[1024];
  extra_calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? extra_calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26;
  load_shared<2, double, double>(extra_calib,
                                 0 * extra_calib_num_alloc,
                                 extra_calib_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          extra_calib_indices_loc[threadIdx.x].target,
                          r0,
                          r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
    read_idx_2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r0, r5);
  };
  load_shared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r8, r9);
    r10 = -2.00000000000000000e+00;
    r11 = r9 * r10;
    read_idx_2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r12, r13);
    r14 = 2.00000000000000000e+00;
    r15 = r12 * r14;
    r16 = r13 * r15;
    r17 = fma(r8, r11, r16);
    r17 = fma(r7, r17, r0);
  };
  load_shared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r0);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r18 = r13 * r9;
    r19 = r8 * r15;
    r18 = fma(r14, r18, r19);
    r20 = r8 * r8;
    r20 = r10 * r20;
    r21 = 1.00000000000000000e+00;
    r22 = r13 * r13;
    r22 = fma(r10, r22, r21);
    r23 = r20 + r22;
    r17 = fma(r0, r18, r17);
    r17 = fma(r6, r23, r17);
  };
  load_shared<1, double, double>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, focal_indices_loc[threadIdx.x].target, r23);
  };
  __syncthreads();
  load_shared<1, double, double>(extra_calib,
                                 2 * extra_calib_num_alloc,
                                 extra_calib_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared,
                          extra_calib_indices_loc[threadIdx.x].target,
                          r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r24 = 1.00000000000000008e-15;
    read_idx_1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r25);
    r26 = r13 * r8;
    r26 = r26 * r14;
    r15 = fma(r9, r15, r26);
    r15 = fma(r7, r15, r25);
    r19 = fma(r13, r11, r19);
    r25 = r12 * r12;
    r25 = r25 * r10;
    r22 = r25 + r22;
    r15 = fma(r6, r19, r15);
    r15 = fma(r0, r22, r15);
    r22 = copysign(1.0, r15);
    r22 = fma(r24, r22, r15);
    r24 = r22 * r22;
    r24 = 1.0 / r24;
    r15 = r17 * r17;
    r19 = r8 * r9;
    r19 = fma(r14, r19, r16);
    r19 = fma(r6, r19, r5);
    r11 = fma(r12, r11, r26);
    r20 = r21 + r20;
    r20 = r20 + r25;
    r19 = fma(r0, r11, r19);
    r19 = fma(r7, r20, r19);
    r20 = r19 * r19;
    r20 = fma(r24, r20, r24 * r15);
    r20 = fma(r18, r20, r21);
    r20 = r23 * r20;
    r22 = 1.0 / r22;
    r20 = r20 * r22;
    r2 = fma(r17, r20, r2);
    r4 = fma(r3, r4, r1);
    r4 = fma(r19, r20, r4);
    r4 = fma(r4, r4, r2 * r2);
  };
  sum_store<double>(out_rTr_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r4);
  sum_flush_final<double>(out_rTr_local, out_rTr, 1);
}

void simple_radial_fixed_pose_score(double* focal,
                                    unsigned int focal_num_alloc,
                                    SharedIndex* focal_indices,
                                    double* extra_calib,
                                    unsigned int extra_calib_num_alloc,
                                    SharedIndex* extra_calib_indices,
                                    double* point,
                                    unsigned int point_num_alloc,
                                    SharedIndex* point_indices,
                                    double* pixel,
                                    unsigned int pixel_num_alloc,
                                    double* pose,
                                    unsigned int pose_num_alloc,
                                    double* const out_rTr,
                                    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_pose_score_kernel<<<n_blocks, 1024>>>(
      focal,
      focal_num_alloc,
      focal_indices,
      extra_calib,
      extra_calib_num_alloc,
      extra_calib_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      out_rTr,
      problem_size);
}

}  // namespace caspar