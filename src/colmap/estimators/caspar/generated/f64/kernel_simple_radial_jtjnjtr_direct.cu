#include "kernel_simple_radial_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) simple_radial_jtjnjtr_direct_kernel(
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
    double* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    double* point_jac,
    unsigned int point_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    double* const out_extra_calib_njtr,
    unsigned int out_extra_calib_njtr_num_alloc,
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

  __shared__ SharedIndex point_njtr_indices_loc[1024];
  point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34;

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
    r5 = r2 * r3;
  };
  load_shared<1, double, double>(point_njtr,
                                 2 * point_njtr_num_alloc,
                                 point_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, point_njtr_indices_loc[threadIdx.x].target, r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        point_jac, 4 * point_jac_num_alloc, global_thread_idx, r7, r8);
  };
  load_shared<2, double, double>(point_njtr,
                                 0 * point_njtr_num_alloc,
                                 point_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          point_njtr_indices_loc[threadIdx.x].target,
                          r9,
                          r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        point_jac, 2 * point_jac_num_alloc, global_thread_idx, r11, r12);
    r13 = fma(r10, r11, r6 * r7);
    read_idx_2<1024, double, double, double2>(
        point_jac, 0 * point_jac_num_alloc, global_thread_idx, r14, r15);
    r13 = fma(r9, r14, r13);
  };
  load_shared<2, double, double>(extra_calib_njtr,
                                 0 * extra_calib_njtr_num_alloc,
                                 extra_calib_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          extra_calib_njtr_indices_loc[threadIdx.x].target,
                          r16,
                          r17);
  };
  __syncthreads();
  load_shared<1, double, double>(extra_calib_njtr,
                                 2 * extra_calib_njtr_num_alloc,
                                 extra_calib_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared,
                          extra_calib_njtr_indices_loc[threadIdx.x].target,
                          r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(extra_calib_jac,
                                              0 * extra_calib_jac_num_alloc,
                                              global_thread_idx,
                                              r19,
                                              r20);
    r16 = fma(r18, r19, r16);
    r21 = r13 + r16;
    r22 = r5 + r21;
    r2 = r2 * r4;
    r10 = fma(r10, r12, r6 * r8);
    r10 = fma(r9, r15, r10);
    r18 = fma(r18, r20, r17);
    r17 = r10 + r18;
    r9 = r2 + r17;
    r6 = fma(r1, r9, r0 * r22);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 2 * pose_jac_num_alloc, global_thread_idx, r23, r24);
    r25 = fma(r24, r9, r23 * r22);
    write_sum_2<double, double>((double*)inout_shared, r6, r25);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r25, r6);
    r26 = fma(r6, r9, r25 * r22);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 6 * pose_jac_num_alloc, global_thread_idx, r27, r28);
    r29 = fma(r28, r9, r27 * r22);
    write_sum_2<double, double>((double*)inout_shared, r26, r29);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r29, r26);
    r30 = fma(r26, r9, r29 * r22);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 10 * pose_jac_num_alloc, global_thread_idx, r31, r32);
    r22 = fma(r31, r22, r32 * r9);
    write_sum_2<double, double>((double*)inout_shared, r30, r22);
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
                          r22,
                          r30);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r29 = fma(r22, r29, r30 * r31);
  };
  load_shared<2, double, double>(pose_njtr,
                                 2 * pose_njtr_num_alloc,
                                 pose_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          pose_njtr_indices_loc[threadIdx.x].target,
                          r31,
                          r9);
  };
  __syncthreads();
  load_shared<2, double, double>(pose_njtr,
                                 0 * pose_njtr_num_alloc,
                                 pose_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          pose_njtr_indices_loc[threadIdx.x].target,
                          r33,
                          r34);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r29 = fma(r31, r25, r29);
    r29 = fma(r9, r27, r29);
    r29 = fma(r33, r0, r29);
    r29 = fma(r34, r23, r29);
    r21 = r29 + r21;
    r28 = fma(r9, r28, r30 * r32);
    r28 = fma(r31, r6, r28);
    r28 = fma(r33, r1, r28);
    r28 = fma(r34, r24, r28);
    r28 = fma(r22, r26, r28);
    r17 = r28 + r17;
    r17 = fma(r4, r17, r3 * r21);
    write_sum_1<double, double>((double*)inout_shared, r17);
  };
  flush_sum_shared<1, double>(out_focal_njtr,
                              0 * out_focal_njtr_num_alloc,
                              focal_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = r5 + r29;
    r13 = r13 + r29;
    r28 = r2 + r28;
    r10 = r10 + r28;
    write_sum_2<double, double>((double*)inout_shared, r13, r10);
  };
  flush_sum_shared<2, double>(out_extra_calib_njtr,
                              0 * out_extra_calib_njtr_num_alloc,
                              extra_calib_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fma(r20, r10, r19 * r13);
    write_sum_1<double, double>((double*)inout_shared, r10);
  };
  flush_sum_shared<1, double>(out_extra_calib_njtr,
                              2 * out_extra_calib_njtr_num_alloc,
                              extra_calib_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = r16 + r29;
    r28 = r18 + r28;
    r15 = fma(r15, r28, r14 * r29);
    r12 = fma(r12, r28, r11 * r29);
    write_sum_2<double, double>((double*)inout_shared, r15, r12);
  };
  flush_sum_shared<2, double>(out_point_njtr,
                              0 * out_point_njtr_num_alloc,
                              point_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = fma(r8, r28, r7 * r29);
    write_sum_1<double, double>((double*)inout_shared, r28);
  };
  flush_sum_shared<1, double>(out_point_njtr,
                              2 * out_point_njtr_num_alloc,
                              point_njtr_indices_loc,
                              (double*)inout_shared);
}

void simple_radial_jtjnjtr_direct(double* pose_njtr,
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
                                  double* point_njtr,
                                  unsigned int point_njtr_num_alloc,
                                  SharedIndex* point_njtr_indices,
                                  double* point_jac,
                                  unsigned int point_jac_num_alloc,
                                  double* const out_pose_njtr,
                                  unsigned int out_pose_njtr_num_alloc,
                                  double* const out_focal_njtr,
                                  unsigned int out_focal_njtr_num_alloc,
                                  double* const out_extra_calib_njtr,
                                  unsigned int out_extra_calib_njtr_num_alloc,
                                  double* const out_point_njtr,
                                  unsigned int out_point_njtr_num_alloc,
                                  size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_jtjnjtr_direct_kernel<<<n_blocks, 1024>>>(
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
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      out_extra_calib_njtr,
      out_extra_calib_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar