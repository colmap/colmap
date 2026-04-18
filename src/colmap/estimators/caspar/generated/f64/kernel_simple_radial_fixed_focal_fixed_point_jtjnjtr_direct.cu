#include "kernel_simple_radial_fixed_focal_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_focal_fixed_point_jtjnjtr_direct_kernel(
        double* pose_njtr,
        unsigned int pose_njtr_num_alloc,
        SharedIndex* pose_njtr_indices,
        double* pose_jac,
        unsigned int pose_jac_num_alloc,
        double* extra_calib_njtr,
        unsigned int extra_calib_njtr_num_alloc,
        SharedIndex* extra_calib_njtr_indices,
        double* extra_calib_jac,
        unsigned int extra_calib_jac_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_extra_calib_njtr,
        unsigned int out_extra_calib_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_njtr_indices_loc[1024];
  pose_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex extra_calib_njtr_indices_loc[1024];
  extra_calib_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? extra_calib_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1);
  };
  load_shared<2, double, double>(extra_calib_njtr,
                                 0 * extra_calib_njtr_num_alloc,
                                 extra_calib_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          extra_calib_njtr_indices_loc[threadIdx.x].target,
                          r2,
                          r3);
  };
  __syncthreads();
  load_shared<1, double, double>(extra_calib_njtr,
                                 2 * extra_calib_njtr_num_alloc,
                                 extra_calib_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared,
                          extra_calib_njtr_indices_loc[threadIdx.x].target,
                          r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(extra_calib_jac,
                                              0 * extra_calib_jac_num_alloc,
                                              global_thread_idx,
                                              r5,
                                              r6);
    r3 = fma(r4, r6, r3);
    r4 = fma(r4, r5, r2);
    r2 = fma(r0, r4, r1 * r3);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 2 * pose_jac_num_alloc, global_thread_idx, r7, r8);
    r9 = fma(r7, r4, r8 * r3);
    write_sum_2<double, double>((double*)inout_shared, r2, r9);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r9, r2);
    r10 = fma(r9, r4, r2 * r3);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 6 * pose_jac_num_alloc, global_thread_idx, r11, r12);
    r13 = fma(r11, r4, r12 * r3);
    write_sum_2<double, double>((double*)inout_shared, r10, r13);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r13, r10);
    r14 = fma(r13, r4, r10 * r3);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 10 * pose_jac_num_alloc, global_thread_idx, r15, r16);
    r4 = fma(r15, r4, r16 * r3);
    write_sum_2<double, double>((double*)inout_shared, r14, r4);
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
                          r4,
                          r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r13 = fma(r4, r13, r14 * r15);
  };
  load_shared<2, double, double>(pose_njtr,
                                 2 * pose_njtr_num_alloc,
                                 pose_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          pose_njtr_indices_loc[threadIdx.x].target,
                          r15,
                          r3);
  };
  __syncthreads();
  load_shared<2, double, double>(pose_njtr,
                                 0 * pose_njtr_num_alloc,
                                 pose_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          pose_njtr_indices_loc[threadIdx.x].target,
                          r17,
                          r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r13 = fma(r15, r9, r13);
    r13 = fma(r3, r11, r13);
    r13 = fma(r17, r0, r13);
    r13 = fma(r18, r7, r13);
    r12 = fma(r3, r12, r14 * r16);
    r12 = fma(r15, r2, r12);
    r12 = fma(r17, r1, r12);
    r12 = fma(r18, r8, r12);
    r12 = fma(r4, r10, r12);
    write_sum_2<double, double>((double*)inout_shared, r13, r12);
  };
  flush_sum_shared<2, double>(out_extra_calib_njtr,
                              0 * out_extra_calib_njtr_num_alloc,
                              extra_calib_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fma(r6, r12, r5 * r13);
    write_sum_1<double, double>((double*)inout_shared, r12);
  };
  flush_sum_shared<1, double>(out_extra_calib_njtr,
                              2 * out_extra_calib_njtr_num_alloc,
                              extra_calib_njtr_indices_loc,
                              (double*)inout_shared);
}

void simple_radial_fixed_focal_fixed_point_jtjnjtr_direct(
    double* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    double* pose_jac,
    unsigned int pose_jac_num_alloc,
    double* extra_calib_njtr,
    unsigned int extra_calib_njtr_num_alloc,
    SharedIndex* extra_calib_njtr_indices,
    double* extra_calib_jac,
    unsigned int extra_calib_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_extra_calib_njtr,
    unsigned int out_extra_calib_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_focal_fixed_point_jtjnjtr_direct_kernel<<<n_blocks,
                                                                1024>>>(
      pose_njtr,
      pose_njtr_num_alloc,
      pose_njtr_indices,
      pose_jac,
      pose_jac_num_alloc,
      extra_calib_njtr,
      extra_calib_njtr_num_alloc,
      extra_calib_njtr_indices,
      extra_calib_jac,
      extra_calib_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_extra_calib_njtr,
      out_extra_calib_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar