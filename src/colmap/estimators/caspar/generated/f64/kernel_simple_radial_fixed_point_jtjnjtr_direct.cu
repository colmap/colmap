#include "kernel_simple_radial_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_point_jtjnjtr_direct_kernel(
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
        double* principal_point_njtr,
        unsigned int principal_point_njtr_num_alloc,
        SharedIndex* principal_point_njtr_indices,
        double* principal_point_jac,
        unsigned int principal_point_jac_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        double* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
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

  __shared__ SharedIndex principal_point_njtr_indices_loc[1024];
  principal_point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1);
  };
  load_shared<2, double, double>(principal_point_njtr,
                                 0 * principal_point_njtr_num_alloc,
                                 principal_point_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          principal_point_njtr_indices_loc[threadIdx.x].target,
                          r2,
                          r3);
  };
  __syncthreads();
  load_shared<2, double, double>(focal_and_extra_njtr,
                                 0 * focal_and_extra_njtr_num_alloc,
                                 focal_and_extra_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                          r4,
                          r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(focal_and_extra_jac,
                                              0 * focal_and_extra_jac_num_alloc,
                                              global_thread_idx,
                                              r6,
                                              r7);
    read_idx_2<1024, double, double, double2>(focal_and_extra_jac,
                                              2 * focal_and_extra_jac_num_alloc,
                                              global_thread_idx,
                                              r8,
                                              r9);
    r10 = fma(r5, r8, r4 * r6);
    r11 = r2 + r10;
    r5 = fma(r5, r9, r4 * r7);
    r4 = r3 + r5;
    r12 = fma(r1, r4, r0 * r11);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 2 * pose_jac_num_alloc, global_thread_idx, r13, r14);
    r15 = fma(r14, r4, r13 * r11);
    write_sum_2<double, double>((double*)inout_shared, r12, r15);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r15, r12);
    r16 = fma(r12, r4, r15 * r11);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 6 * pose_jac_num_alloc, global_thread_idx, r17, r18);
    r19 = fma(r18, r4, r17 * r11);
    write_sum_2<double, double>((double*)inout_shared, r16, r19);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r19, r16);
    r20 = fma(r16, r4, r19 * r11);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 10 * pose_jac_num_alloc, global_thread_idx, r21, r22);
    r11 = fma(r21, r11, r22 * r4);
    write_sum_2<double, double>((double*)inout_shared, r20, r11);
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
                          r11,
                          r20);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r19 = fma(r11, r19, r20 * r21);
  };
  load_shared<2, double, double>(pose_njtr,
                                 2 * pose_njtr_num_alloc,
                                 pose_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          pose_njtr_indices_loc[threadIdx.x].target,
                          r21,
                          r4);
  };
  __syncthreads();
  load_shared<2, double, double>(pose_njtr,
                                 0 * pose_njtr_num_alloc,
                                 pose_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          pose_njtr_indices_loc[threadIdx.x].target,
                          r23,
                          r24);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r19 = fma(r21, r15, r19);
    r19 = fma(r4, r17, r19);
    r19 = fma(r23, r0, r19);
    r19 = fma(r24, r13, r19);
    r2 = r2 + r19;
    r18 = fma(r4, r18, r20 * r22);
    r18 = fma(r21, r12, r18);
    r18 = fma(r23, r1, r18);
    r18 = fma(r24, r14, r18);
    r18 = fma(r11, r16, r18);
    r3 = r3 + r18;
    r7 = fma(r7, r3, r6 * r2);
    r3 = fma(r9, r3, r8 * r2);
    write_sum_2<double, double>((double*)inout_shared, r7, r3);
  };
  flush_sum_shared<2, double>(out_focal_and_extra_njtr,
                              0 * out_focal_and_extra_njtr_num_alloc,
                              focal_and_extra_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = r19 + r10;
    r5 = r18 + r5;
    write_sum_2<double, double>((double*)inout_shared, r10, r5);
  };
  flush_sum_shared<2, double>(out_principal_point_njtr,
                              0 * out_principal_point_njtr_num_alloc,
                              principal_point_njtr_indices_loc,
                              (double*)inout_shared);
}

void simple_radial_fixed_point_jtjnjtr_direct(
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
    double* principal_point_njtr,
    unsigned int principal_point_njtr_num_alloc,
    SharedIndex* principal_point_njtr_indices,
    double* principal_point_jac,
    unsigned int principal_point_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    double* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_point_jtjnjtr_direct_kernel<<<n_blocks, 1024>>>(
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
      principal_point_njtr,
      principal_point_njtr_num_alloc,
      principal_point_njtr_indices,
      principal_point_jac,
      principal_point_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar