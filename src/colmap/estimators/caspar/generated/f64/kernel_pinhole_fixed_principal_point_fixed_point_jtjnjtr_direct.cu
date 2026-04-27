#include "kernel_pinhole_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_principal_point_fixed_point_jtjnjtr_direct_kernel(
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
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1);
  };
  load_shared<2, double, double>(focal_and_extra_njtr,
                                 0 * focal_and_extra_njtr_num_alloc,
                                 focal_and_extra_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                          r2,
                          r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(focal_and_extra_jac,
                                              0 * focal_and_extra_jac_num_alloc,
                                              global_thread_idx,
                                              r4,
                                              r5);
    r2 = r2 * r4;
    r3 = r3 * r5;
    r6 = fma(r1, r3, r0 * r2);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 2 * pose_jac_num_alloc, global_thread_idx, r7, r8);
    r9 = fma(r8, r3, r7 * r2);
    write_sum_2<double, double>((double*)inout_shared, r6, r9);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 6 * pose_jac_num_alloc, global_thread_idx, r9, r6);
    r10 = r9 * r2;
    read_idx_2<1024, double, double, double2>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r11, r12);
    r13 = fma(r12, r3, r11 * r2);
    write_sum_2<double, double>((double*)inout_shared, r13, r10);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = r6 * r3;
    read_idx_2<1024, double, double, double2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r13, r14);
    r2 = fma(r13, r2, r14 * r3);
    write_sum_2<double, double>((double*)inout_shared, r10, r2);
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
                          r2,
                          r10);
  };
  __syncthreads();
  load_shared<2, double, double>(pose_njtr,
                                 2 * pose_njtr_num_alloc,
                                 pose_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          pose_njtr_indices_loc[threadIdx.x].target,
                          r3,
                          r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r11 = fma(r3, r11, r10 * r13);
  };
  load_shared<2, double, double>(pose_njtr,
                                 0 * pose_njtr_num_alloc,
                                 pose_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          pose_njtr_indices_loc[threadIdx.x].target,
                          r13,
                          r16);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r11 = fma(r15, r9, r11);
    r11 = fma(r13, r0, r11);
    r11 = fma(r16, r7, r11);
    r11 = r4 * r11;
    r12 = fma(r3, r12, r10 * r14);
    r12 = fma(r13, r1, r12);
    r12 = fma(r16, r8, r12);
    r12 = fma(r2, r6, r12);
    r12 = r5 * r12;
    write_sum_2<double, double>((double*)inout_shared, r11, r12);
  };
  flush_sum_shared<2, double>(out_focal_and_extra_njtr,
                              0 * out_focal_and_extra_njtr_num_alloc,
                              focal_and_extra_njtr_indices_loc,
                              (double*)inout_shared);
}

void pinhole_fixed_principal_point_fixed_point_jtjnjtr_direct(
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
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_fixed_principal_point_fixed_point_jtjnjtr_direct_kernel<<<n_blocks,
                                                                    1024>>>(
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
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar