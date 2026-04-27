#include "kernel_simple_radial_fixed_principal_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_principal_point_jtjnjtr_direct_kernel(
        double* pose_njtr,
        unsigned int pose_njtr_num_alloc,
        SharedIndex* pose_njtr_indices,
        double* pose_jac,
        unsigned int pose_jac_num_alloc,
        double* focal_and_extra_njtr,
        unsigned int focal_and_extra_njtr_num_alloc,
        SharedIndex* focal_and_extra_njtr_indices,
        double* focal_and_extra_jac,
        unsigned int focal_and_extra_jac_num_alloc,
        double* point_njtr,
        unsigned int point_njtr_num_alloc,
        SharedIndex* point_njtr_indices,
        double* point_jac,
        unsigned int point_jac_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        double* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_njtr_indices_loc[1024];
  pose_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex focal_and_extra_njtr_indices_loc[1024];
  focal_and_extra_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex point_njtr_indices_loc[1024];
  point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1);
  };
  load_shared<1, double, double>(point_njtr,
                                 2 * point_njtr_num_alloc,
                                 point_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, point_njtr_indices_loc[threadIdx.x].target, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        point_jac, 4 * point_jac_num_alloc, global_thread_idx, r3, r4);
  };
  load_shared<2, double, double>(point_njtr,
                                 0 * point_njtr_num_alloc,
                                 point_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          point_njtr_indices_loc[threadIdx.x].target,
                          r5,
                          r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        point_jac, 2 * point_jac_num_alloc, global_thread_idx, r7, r8);
    r9 = fma(r6, r7, r2 * r3);
    read_idx_2<1024, double, double, double2>(
        point_jac, 0 * point_jac_num_alloc, global_thread_idx, r10, r11);
    r9 = fma(r5, r10, r9);
  };
  load_shared<2, double, double>(focal_and_extra_njtr,
                                 0 * focal_and_extra_njtr_num_alloc,
                                 focal_and_extra_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                          r12,
                          r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(focal_and_extra_jac,
                                              0 * focal_and_extra_jac_num_alloc,
                                              global_thread_idx,
                                              r14,
                                              r15);
    read_idx_2<1024, double, double, double2>(focal_and_extra_jac,
                                              2 * focal_and_extra_jac_num_alloc,
                                              global_thread_idx,
                                              r16,
                                              r17);
    r18 = fma(r13, r16, r12 * r14);
    r19 = r9 + r18;
    r6 = fma(r6, r8, r2 * r4);
    r6 = fma(r5, r11, r6);
    r13 = fma(r13, r17, r12 * r15);
    r12 = r6 + r13;
    r5 = fma(r1, r12, r0 * r19);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 2 * pose_jac_num_alloc, global_thread_idx, r2, r20);
    r21 = fma(r20, r12, r2 * r19);
    write_sum_2<double, double>((double*)inout_shared, r5, r21);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r21, r5);
    r22 = fma(r5, r12, r21 * r19);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 6 * pose_jac_num_alloc, global_thread_idx, r23, r24);
    r25 = fma(r24, r12, r23 * r19);
    write_sum_2<double, double>((double*)inout_shared, r22, r25);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r25, r22);
    r26 = fma(r22, r12, r25 * r19);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 10 * pose_jac_num_alloc, global_thread_idx, r27, r28);
    r19 = fma(r27, r19, r28 * r12);
    write_sum_2<double, double>((double*)inout_shared, r26, r19);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              4 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  load_shared<2, double, double>(pose_njtr,
                                 4 * pose_njtr_num_alloc,
                                 pose_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          pose_njtr_indices_loc[threadIdx.x].target,
                          r19,
                          r26);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r25 = fma(r19, r25, r26 * r27);
  };
  load_shared<2, double, double>(pose_njtr,
                                 2 * pose_njtr_num_alloc,
                                 pose_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          pose_njtr_indices_loc[threadIdx.x].target,
                          r27,
                          r12);
  };
  __syncthreads();
  load_shared<2, double, double>(pose_njtr,
                                 0 * pose_njtr_num_alloc,
                                 pose_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          pose_njtr_indices_loc[threadIdx.x].target,
                          r29,
                          r30);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r25 = fma(r27, r21, r25);
    r25 = fma(r12, r23, r25);
    r25 = fma(r29, r0, r25);
    r25 = fma(r30, r2, r25);
    r9 = r25 + r9;
    r24 = fma(r12, r24, r26 * r28);
    r24 = fma(r27, r5, r24);
    r24 = fma(r29, r1, r24);
    r24 = fma(r30, r20, r24);
    r24 = fma(r19, r22, r24);
    r6 = r24 + r6;
    r15 = fma(r15, r6, r14 * r9);
    r6 = fma(r17, r6, r16 * r9);
    write_sum_2<double, double>((double*)inout_shared, r15, r6);
  };
  flush_sum_shared<2, double>(out_focal_and_extra_njtr,
                              0 * out_focal_and_extra_njtr_num_alloc,
                              focal_and_extra_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = r25 + r18;
    r13 = r24 + r13;
    r11 = fma(r11, r13, r10 * r18);
    r8 = fma(r8, r13, r7 * r18);
    write_sum_2<double, double>((double*)inout_shared, r11, r8);
  };
  flush_sum_shared<2, double>(out_point_njtr,
                              0 * out_point_njtr_num_alloc,
                              point_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = fma(r4, r13, r3 * r18);
    write_sum_1<double, double>((double*)inout_shared, r13);
  };
  flush_sum_shared<1, double>(out_point_njtr,
                              2 * out_point_njtr_num_alloc,
                              point_njtr_indices_loc,
                              (double*)inout_shared);
}

void simple_radial_fixed_principal_point_jtjnjtr_direct(
    double* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    double* pose_jac,
    unsigned int pose_jac_num_alloc,
    double* focal_and_extra_njtr,
    unsigned int focal_and_extra_njtr_num_alloc,
    SharedIndex* focal_and_extra_njtr_indices,
    double* focal_and_extra_jac,
    unsigned int focal_and_extra_jac_num_alloc,
    double* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    double* point_jac,
    unsigned int point_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_principal_point_jtjnjtr_direct_kernel<<<n_blocks, 1024>>>(
      pose_njtr,
      pose_njtr_num_alloc,
      pose_njtr_indices,
      pose_jac,
      pose_jac_num_alloc,
      focal_and_extra_njtr,
      focal_and_extra_njtr_num_alloc,
      focal_and_extra_njtr_indices,
      focal_and_extra_jac,
      focal_and_extra_jac_num_alloc,
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar