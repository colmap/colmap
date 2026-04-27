#include "kernel_simple_radial_fixed_focal_and_extra_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_focal_and_extra_score_kernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        double* const out_rTr,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26;
  load_shared<2, double, double>(principal_point,
                                 0 * principal_point_num_alloc,
                                 principal_point_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          principal_point_indices_loc[threadIdx.x].target,
                          r0,
                          r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r3 = fma(r3, r4, r1);
  };
  load_shared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r1, r5);
  };
  __syncthreads();
  load_shared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  load_shared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = r8 * r9;
    r11 = 2.00000000000000000e+00;
    r10 = r10 * r11;
  };
  load_shared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r12, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r14 = r12 * r11;
    r15 = fma(r13, r14, r10);
    r15 = fma(r6, r15, r5);
  };
  load_shared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = r9 * r14;
    r17 = -2.00000000000000000e+00;
    r18 = r13 * r17;
    r19 = fma(r8, r18, r16);
    r20 = r12 * r12;
    r20 = r20 * r17;
    r21 = 1.00000000000000000e+00;
    r22 = r8 * r8;
    r22 = fma(r17, r22, r21);
    r23 = r20 + r22;
    r15 = fma(r5, r19, r15);
    r15 = fma(r7, r23, r15);
    read_idx_2<1024, double, double, double2>(focal_and_extra,
                                              0 * focal_and_extra_num_alloc,
                                              global_thread_idx,
                                              r23,
                                              r19);
    r24 = 1.00000000000000008e-15;
  };
  load_shared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r25);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r26 = r8 * r13;
    r26 = fma(r11, r26, r16);
    r26 = fma(r7, r26, r25);
    r14 = r8 * r14;
    r25 = fma(r9, r18, r14);
    r16 = r9 * r9;
    r16 = r17 * r16;
    r22 = r16 + r22;
    r26 = fma(r6, r25, r26);
    r26 = fma(r5, r22, r26);
    r22 = copysign(1.0, r26);
    r22 = fma(r24, r22, r26);
    r24 = r22 * r22;
    r24 = 1.0 / r24;
    r18 = fma(r12, r18, r10);
    r18 = fma(r7, r18, r1);
    r7 = r9 * r13;
    r7 = fma(r11, r7, r14);
    r20 = r21 + r20;
    r20 = r20 + r16;
    r18 = fma(r5, r7, r18);
    r18 = fma(r6, r20, r18);
    r20 = r18 * r18;
    r6 = r15 * r15;
    r6 = fma(r24, r6, r24 * r20);
    r6 = fma(r19, r6, r21);
    r6 = r23 * r6;
    r22 = 1.0 / r22;
    r6 = r6 * r22;
    r3 = fma(r15, r6, r3);
    r4 = fma(r2, r4, r0);
    r4 = fma(r18, r6, r4);
    r4 = fma(r4, r4, r3 * r3);
  };
  sum_store<double>(out_rTr_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r4);
  sum_flush_final<double>(out_rTr_local, out_rTr, 1);
}

void simple_radial_fixed_focal_and_extra_score(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    double* const out_rTr,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_focal_and_extra_score_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
      out_rTr,
      problem_size);
}

}  // namespace caspar