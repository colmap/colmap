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
        double* focal_njtr,
        unsigned int focal_njtr_num_alloc,
        SharedIndex* focal_njtr_indices,
        double* focal_jac,
        unsigned int focal_jac_num_alloc,
        double* extra_calib_njtr,
        unsigned int extra_calib_njtr_num_alloc,
        SharedIndex* extra_calib_njtr_indices,
        double* extra_calib_jac,
        unsigned int extra_calib_jac_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
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

  __shared__ SharedIndex focal_njtr_indices_loc[1024];
  focal_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex extra_calib_njtr_indices_loc[1024];
  extra_calib_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? extra_calib_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1);
  };
  load_shared<1, double, double>(focal_njtr,
                                 0 * focal_njtr_num_alloc,
                                 focal_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, focal_njtr_indices_loc[threadIdx.x].target, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        focal_jac, 0 * focal_jac_num_alloc, global_thread_idx, r3, r4);
    r5 = r2 * r4;
  };
  load_shared<2, double, double>(extra_calib_njtr,
                                 0 * extra_calib_njtr_num_alloc,
                                 extra_calib_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          extra_calib_njtr_indices_loc[threadIdx.x].target,
                          r6,
                          r7);
  };
  __syncthreads();
  load_shared<1, double, double>(extra_calib_njtr,
                                 2 * extra_calib_njtr_num_alloc,
                                 extra_calib_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared,
                          extra_calib_njtr_indices_loc[threadIdx.x].target,
                          r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(extra_calib_jac,
                                              0 * extra_calib_jac_num_alloc,
                                              global_thread_idx,
                                              r9,
                                              r10);
    r7 = fma(r8, r10, r7);
    r11 = r5 + r7;
    r2 = r2 * r3;
    r8 = fma(r8, r9, r6);
    r6 = r2 + r8;
    r12 = fma(r0, r6, r1 * r11);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 2 * pose_jac_num_alloc, global_thread_idx, r13, r14);
    r15 = fma(r13, r6, r14 * r11);
    write_sum_2<double, double>((double*)inout_shared, r12, r15);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r15, r12);
    r16 = fma(r15, r6, r12 * r11);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 6 * pose_jac_num_alloc, global_thread_idx, r17, r18);
    r19 = fma(r17, r6, r18 * r11);
    write_sum_2<double, double>((double*)inout_shared, r16, r19);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r19, r16);
    r20 = fma(r19, r6, r16 * r11);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 10 * pose_jac_num_alloc, global_thread_idx, r21, r22);
    r6 = fma(r21, r6, r22 * r11);
    write_sum_2<double, double>((double*)inout_shared, r20, r6);
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
                          r6,
                          r20);
  };
  __syncthreads();
  load_shared<2, double, double>(pose_njtr,
                                 2 * pose_njtr_num_alloc,
                                 pose_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          pose_njtr_indices_loc[threadIdx.x].target,
                          r11,
                          r23);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r18 = fma(r23, r18, r20 * r22);
  };
  load_shared<2, double, double>(pose_njtr,
                                 0 * pose_njtr_num_alloc,
                                 pose_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          pose_njtr_indices_loc[threadIdx.x].target,
                          r22,
                          r24);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r18 = fma(r11, r12, r18);
    r18 = fma(r22, r1, r18);
    r18 = fma(r24, r14, r18);
    r18 = fma(r6, r16, r18);
    r7 = r18 + r7;
    r19 = fma(r6, r19, r20 * r21);
    r19 = fma(r11, r15, r19);
    r19 = fma(r23, r17, r19);
    r19 = fma(r22, r0, r19);
    r19 = fma(r24, r13, r19);
    r8 = r19 + r8;
    r8 = fma(r3, r8, r4 * r7);
    write_sum_1<double, double>((double*)inout_shared, r8);
  };
  flush_sum_shared<1, double>(out_focal_njtr,
                              0 * out_focal_njtr_num_alloc,
                              focal_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = r2 + r19;
    r18 = r5 + r18;
    write_sum_2<double, double>((double*)inout_shared, r19, r18);
  };
  flush_sum_shared<2, double>(out_extra_calib_njtr,
                              0 * out_extra_calib_njtr_num_alloc,
                              extra_calib_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fma(r9, r19, r10 * r18);
    write_sum_1<double, double>((double*)inout_shared, r19);
  };
  flush_sum_shared<1, double>(out_extra_calib_njtr,
                              2 * out_extra_calib_njtr_num_alloc,
                              extra_calib_njtr_indices_loc,
                              (double*)inout_shared);
}

void simple_radial_fixed_point_jtjnjtr_direct(
    double* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    double* pose_jac,
    unsigned int pose_jac_num_alloc,
    double* focal_njtr,
    unsigned int focal_njtr_num_alloc,
    SharedIndex* focal_njtr_indices,
    double* focal_jac,
    unsigned int focal_jac_num_alloc,
    double* extra_calib_njtr,
    unsigned int extra_calib_njtr_num_alloc,
    SharedIndex* extra_calib_njtr_indices,
    double* extra_calib_jac,
    unsigned int extra_calib_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    double* const out_extra_calib_njtr,
    unsigned int out_extra_calib_njtr_num_alloc,
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
      focal_njtr,
      focal_njtr_num_alloc,
      focal_njtr_indices,
      focal_jac,
      focal_jac_num_alloc,
      extra_calib_njtr,
      extra_calib_njtr_num_alloc,
      extra_calib_njtr_indices,
      extra_calib_jac,
      extra_calib_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      out_extra_calib_njtr,
      out_extra_calib_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar