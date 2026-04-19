#include "kernel_pinhole_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) pinhole_jtjnjtr_direct_kernel(
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
    double* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    double* point_jac,
    unsigned int point_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    double* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
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

  __shared__ SharedIndex principal_point_njtr_indices_loc[1024];
  principal_point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_njtr_indices[global_thread_idx]
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
    r9 = fma(r6, r8, r2 * r4);
    read_idx_2<1024, double, double, double2>(
        point_jac, 0 * point_jac_num_alloc, global_thread_idx, r10, r11);
    r9 = fma(r5, r11, r9);
  };
  load_shared<2, double, double>(principal_point_njtr,
                                 0 * principal_point_njtr_num_alloc,
                                 principal_point_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          principal_point_njtr_indices_loc[threadIdx.x].target,
                          r12,
                          r13);
  };
  __syncthreads();
  load_shared<2, double, double>(focal_and_extra_njtr,
                                 0 * focal_and_extra_njtr_num_alloc,
                                 focal_and_extra_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                          r14,
                          r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(focal_and_extra_jac,
                                              0 * focal_and_extra_jac_num_alloc,
                                              global_thread_idx,
                                              r16,
                                              r17);
    r15 = r15 * r17;
    r18 = r13 + r15;
    r19 = r9 + r18;
    r6 = fma(r6, r7, r2 * r3);
    r6 = fma(r5, r10, r6);
    r14 = r14 * r16;
    r5 = r12 + r14;
    r2 = r6 + r5;
    r20 = fma(r0, r2, r1 * r19);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 2 * pose_jac_num_alloc, global_thread_idx, r21, r22);
    r23 = fma(r21, r2, r22 * r19);
    write_sum_2<double, double>((double*)inout_shared, r20, r23);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r23, r20);
    r24 = fma(r23, r2, r20 * r19);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 6 * pose_jac_num_alloc, global_thread_idx, r25, r26);
    r27 = r25 * r2;
    write_sum_2<double, double>((double*)inout_shared, r24, r27);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = r26 * r19;
    read_idx_2<1024, double, double, double2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r24, r28);
    r19 = fma(r28, r19, r24 * r2);
    write_sum_2<double, double>((double*)inout_shared, r27, r19);
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
                          r27);
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
                          r29);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r23 = fma(r2, r23, r27 * r24);
  };
  load_shared<2, double, double>(pose_njtr,
                                 0 * pose_njtr_num_alloc,
                                 pose_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          pose_njtr_indices_loc[threadIdx.x].target,
                          r24,
                          r30);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r23 = fma(r29, r25, r23);
    r23 = fma(r24, r0, r23);
    r23 = fma(r30, r21, r23);
    r6 = r23 + r6;
    r12 = r12 + r6;
    r12 = r16 * r12;
    r20 = fma(r2, r20, r27 * r28);
    r20 = fma(r24, r1, r20);
    r20 = fma(r30, r22, r20);
    r20 = fma(r19, r26, r20);
    r9 = r20 + r9;
    r13 = r13 + r9;
    r13 = r17 * r13;
    write_sum_2<double, double>((double*)inout_shared, r12, r13);
  };
  flush_sum_shared<2, double>(out_focal_and_extra_njtr,
                              0 * out_focal_and_extra_njtr_num_alloc,
                              focal_and_extra_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r14 + r6;
    r9 = r15 + r9;
    write_sum_2<double, double>((double*)inout_shared, r6, r9);
  };
  flush_sum_shared<2, double>(out_principal_point_njtr,
                              0 * out_principal_point_njtr_num_alloc,
                              principal_point_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = r20 + r18;
    r5 = r23 + r5;
    r10 = fma(r10, r5, r11 * r18);
    r7 = fma(r7, r5, r8 * r18);
    write_sum_2<double, double>((double*)inout_shared, r10, r7);
  };
  flush_sum_shared<2, double>(out_point_njtr,
                              0 * out_point_njtr_num_alloc,
                              point_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r5 = fma(r3, r5, r4 * r18);
    write_sum_1<double, double>((double*)inout_shared, r5);
  };
  flush_sum_shared<1, double>(out_point_njtr,
                              2 * out_point_njtr_num_alloc,
                              point_njtr_indices_loc,
                              (double*)inout_shared);
}

void pinhole_jtjnjtr_direct(double* pose_njtr,
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
                            double* point_njtr,
                            unsigned int point_njtr_num_alloc,
                            SharedIndex* point_njtr_indices,
                            double* point_jac,
                            unsigned int point_jac_num_alloc,
                            double* const out_pose_njtr,
                            unsigned int out_pose_njtr_num_alloc,
                            double* const out_focal_and_extra_njtr,
                            unsigned int out_focal_and_extra_njtr_num_alloc,
                            double* const out_principal_point_njtr,
                            unsigned int out_principal_point_njtr_num_alloc,
                            double* const out_point_njtr,
                            unsigned int out_point_njtr_num_alloc,
                            size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_jtjnjtr_direct_kernel<<<n_blocks, 1024>>>(
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
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar