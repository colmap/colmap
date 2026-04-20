#include "kernel_pinhole_fixed_focal_and_extra_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_focal_and_extra_fixed_point_jtjnjtr_direct_kernel(
        double* pose_njtr,
        unsigned int pose_njtr_num_alloc,
        SharedIndex* pose_njtr_indices,
        double* pose_jac,
        unsigned int pose_jac_num_alloc,
        double* principal_point_njtr,
        unsigned int principal_point_njtr_num_alloc,
        SharedIndex* principal_point_njtr_indices,
        double* principal_point_jac,
        unsigned int principal_point_jac_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
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

  __shared__ SharedIndex principal_point_njtr_indices_loc[1024];
  principal_point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14;

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
  if (global_thread_idx < problem_size) {
    r4 = fma(r1, r3, r0 * r2);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 2 * pose_jac_num_alloc, global_thread_idx, r5, r6);
    r7 = fma(r6, r3, r5 * r2);
    write_sum_2<double, double>((double*)inout_shared, r4, r7);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r7, r4);
    r8 = fma(r4, r3, r7 * r2);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 6 * pose_jac_num_alloc, global_thread_idx, r9, r10);
    r11 = r9 * r2;
    write_sum_2<double, double>((double*)inout_shared, r8, r11);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = r10 * r3;
    read_idx_2<1024, double, double, double2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r8, r12);
    r3 = fma(r12, r3, r8 * r2);
    write_sum_2<double, double>((double*)inout_shared, r11, r3);
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
                          r3,
                          r11);
  };
  __syncthreads();
  load_shared<2, double, double>(pose_njtr,
                                 2 * pose_njtr_num_alloc,
                                 pose_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          pose_njtr_indices_loc[threadIdx.x].target,
                          r2,
                          r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r7 = fma(r2, r7, r11 * r8);
  };
  load_shared<2, double, double>(pose_njtr,
                                 0 * pose_njtr_num_alloc,
                                 pose_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          pose_njtr_indices_loc[threadIdx.x].target,
                          r8,
                          r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r7 = fma(r13, r9, r7);
    r7 = fma(r8, r0, r7);
    r7 = fma(r14, r5, r7);
    r4 = fma(r2, r4, r11 * r12);
    r4 = fma(r8, r1, r4);
    r4 = fma(r14, r6, r4);
    r4 = fma(r3, r10, r4);
    write_sum_2<double, double>((double*)inout_shared, r7, r4);
  };
  flush_sum_shared<2, double>(out_principal_point_njtr,
                              0 * out_principal_point_njtr_num_alloc,
                              principal_point_njtr_indices_loc,
                              (double*)inout_shared);
}

void pinhole_fixed_focal_and_extra_fixed_point_jtjnjtr_direct(
    double* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    double* pose_jac,
    unsigned int pose_jac_num_alloc,
    double* principal_point_njtr,
    unsigned int principal_point_njtr_num_alloc,
    SharedIndex* principal_point_njtr_indices,
    double* principal_point_jac,
    unsigned int principal_point_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_fixed_focal_and_extra_fixed_point_jtjnjtr_direct_kernel<<<n_blocks,
                                                                    1024>>>(
      pose_njtr,
      pose_njtr_num_alloc,
      pose_njtr_indices,
      pose_jac,
      pose_jac_num_alloc,
      principal_point_njtr,
      principal_point_njtr_num_alloc,
      principal_point_njtr_indices,
      principal_point_jac,
      principal_point_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar