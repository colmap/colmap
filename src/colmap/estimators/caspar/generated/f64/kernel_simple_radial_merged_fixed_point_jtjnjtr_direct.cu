#include "kernel_simple_radial_merged_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_merged_fixed_point_jtjnjtr_direct_kernel(
        double* pose_njtr,
        unsigned int pose_njtr_num_alloc,
        SharedIndex* pose_njtr_indices,
        double* pose_jac,
        unsigned int pose_jac_num_alloc,
        double* calib_njtr,
        unsigned int calib_njtr_num_alloc,
        SharedIndex* calib_njtr_indices,
        double* calib_jac,
        unsigned int calib_jac_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_njtr_indices_loc[1024];
  pose_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex calib_njtr_indices_loc[1024];
  calib_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1);
  };
  load_shared<2, double, double>(calib_njtr,
                                 2 * calib_njtr_num_alloc,
                                 calib_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          calib_njtr_indices_loc[threadIdx.x].target,
                          r2,
                          r3);
  };
  __syncthreads();
  load_shared<2, double, double>(calib_njtr,
                                 0 * calib_njtr_num_alloc,
                                 calib_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          calib_njtr_indices_loc[threadIdx.x].target,
                          r4,
                          r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        calib_jac, 0 * calib_jac_num_alloc, global_thread_idx, r6, r7);
    r2 = fma(r4, r6, r2);
    read_idx_2<1024, double, double, double2>(
        calib_jac, 2 * calib_jac_num_alloc, global_thread_idx, r8, r9);
    r2 = fma(r5, r8, r2);
    r4 = fma(r4, r7, r3);
    r4 = fma(r5, r9, r4);
    r5 = fma(r1, r4, r0 * r2);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 2 * pose_jac_num_alloc, global_thread_idx, r3, r10);
    r11 = fma(r10, r4, r3 * r2);
    write_sum_2<double, double>((double*)inout_shared, r5, r11);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r11, r5);
    r12 = fma(r5, r4, r11 * r2);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 6 * pose_jac_num_alloc, global_thread_idx, r13, r14);
    r15 = fma(r14, r4, r13 * r2);
    write_sum_2<double, double>((double*)inout_shared, r12, r15);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r15, r12);
    r16 = fma(r12, r4, r15 * r2);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 10 * pose_jac_num_alloc, global_thread_idx, r17, r18);
    r2 = fma(r17, r2, r18 * r4);
    write_sum_2<double, double>((double*)inout_shared, r16, r2);
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
                          r16);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = fma(r2, r15, r16 * r17);
  };
  load_shared<2, double, double>(pose_njtr,
                                 2 * pose_njtr_num_alloc,
                                 pose_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          pose_njtr_indices_loc[threadIdx.x].target,
                          r17,
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
                          r19,
                          r20);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = fma(r17, r11, r15);
    r15 = fma(r4, r13, r15);
    r15 = fma(r19, r0, r15);
    r15 = fma(r20, r3, r15);
    r14 = fma(r4, r14, r16 * r18);
    r14 = fma(r17, r5, r14);
    r14 = fma(r19, r1, r14);
    r14 = fma(r20, r10, r14);
    r14 = fma(r2, r12, r14);
    r7 = fma(r7, r14, r6 * r15);
    r9 = fma(r9, r14, r8 * r15);
    write_sum_2<double, double>((double*)inout_shared, r7, r9);
  };
  flush_sum_shared<2, double>(out_calib_njtr,
                              0 * out_calib_njtr_num_alloc,
                              calib_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<double, double>((double*)inout_shared, r15, r14);
  };
  flush_sum_shared<2, double>(out_calib_njtr,
                              2 * out_calib_njtr_num_alloc,
                              calib_njtr_indices_loc,
                              (double*)inout_shared);
}

void simple_radial_merged_fixed_point_jtjnjtr_direct(
    double* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    double* pose_jac,
    unsigned int pose_jac_num_alloc,
    double* calib_njtr,
    unsigned int calib_njtr_num_alloc,
    SharedIndex* calib_njtr_indices,
    double* calib_jac,
    unsigned int calib_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_merged_fixed_point_jtjnjtr_direct_kernel<<<n_blocks, 1024>>>(
      pose_njtr,
      pose_njtr_num_alloc,
      pose_njtr_indices,
      pose_jac,
      pose_jac_num_alloc,
      calib_njtr,
      calib_njtr_num_alloc,
      calib_njtr_indices,
      calib_jac,
      calib_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar