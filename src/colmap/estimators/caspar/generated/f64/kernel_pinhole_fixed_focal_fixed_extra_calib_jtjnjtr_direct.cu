#include "kernel_pinhole_fixed_focal_fixed_extra_calib_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_focal_fixed_extra_calib_jtjnjtr_direct_kernel(
        double* pose_njtr,
        unsigned int pose_njtr_num_alloc,
        SharedIndex* pose_njtr_indices,
        double* pose_jac,
        unsigned int pose_jac_num_alloc,
        double* point_njtr,
        unsigned int point_njtr_num_alloc,
        SharedIndex* point_njtr_indices,
        double* point_jac,
        unsigned int point_jac_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
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

  __shared__ SharedIndex point_njtr_indices_loc[1024];
  point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20;

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
    r6 = fma(r6, r8, r2 * r4);
    r6 = fma(r5, r11, r6);
    r5 = fma(r1, r6, r0 * r9);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 2 * pose_jac_num_alloc, global_thread_idx, r2, r12);
    r13 = fma(r12, r6, r2 * r9);
    write_sum_2<double, double>((double*)inout_shared, r5, r13);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r13, r5);
    r14 = fma(r5, r6, r13 * r9);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 6 * pose_jac_num_alloc, global_thread_idx, r15, r16);
    r17 = r15 * r9;
    write_sum_2<double, double>((double*)inout_shared, r14, r17);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = r16 * r6;
    read_idx_2<1024, double, double, double2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r14, r18);
    r9 = fma(r14, r9, r18 * r6);
    write_sum_2<double, double>((double*)inout_shared, r17, r9);
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
                          r9,
                          r17);
  };
  __syncthreads();
  load_shared<2, double, double>(pose_njtr,
                                 2 * pose_njtr_num_alloc,
                                 pose_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          pose_njtr_indices_loc[threadIdx.x].target,
                          r6,
                          r19);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r5 = fma(r6, r5, r17 * r18);
  };
  load_shared<2, double, double>(pose_njtr,
                                 0 * pose_njtr_num_alloc,
                                 pose_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          pose_njtr_indices_loc[threadIdx.x].target,
                          r18,
                          r20);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r5 = fma(r18, r1, r5);
    r5 = fma(r20, r12, r5);
    r5 = fma(r9, r16, r5);
    r13 = fma(r6, r13, r17 * r14);
    r13 = fma(r19, r15, r13);
    r13 = fma(r18, r0, r13);
    r13 = fma(r20, r2, r13);
    r10 = fma(r10, r13, r11 * r5);
    r7 = fma(r7, r13, r8 * r5);
    write_sum_2<double, double>((double*)inout_shared, r10, r7);
  };
  flush_sum_shared<2, double>(out_point_njtr,
                              0 * out_point_njtr_num_alloc,
                              point_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = fma(r3, r13, r4 * r5);
    write_sum_1<double, double>((double*)inout_shared, r13);
  };
  flush_sum_shared<1, double>(out_point_njtr,
                              2 * out_point_njtr_num_alloc,
                              point_njtr_indices_loc,
                              (double*)inout_shared);
}

void pinhole_fixed_focal_fixed_extra_calib_jtjnjtr_direct(
    double* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    double* pose_jac,
    unsigned int pose_jac_num_alloc,
    double* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    double* point_jac,
    unsigned int point_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_fixed_focal_fixed_extra_calib_jtjnjtr_direct_kernel<<<n_blocks,
                                                                1024>>>(
      pose_njtr,
      pose_njtr_num_alloc,
      pose_njtr_indices,
      pose_jac,
      pose_jac_num_alloc,
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar