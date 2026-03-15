#include "kernel_simple_radial_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_jtjnjtr_direct_kernel(double* pose_njtr,
                                        unsigned int pose_njtr_num_alloc,
                                        SharedIndex* pose_njtr_indices,
                                        double* pose_jac,
                                        unsigned int pose_jac_num_alloc,
                                        double* calib_njtr,
                                        unsigned int calib_njtr_num_alloc,
                                        SharedIndex* calib_njtr_indices,
                                        double* calib_jac,
                                        unsigned int calib_jac_num_alloc,
                                        double* point_njtr,
                                        unsigned int point_njtr_num_alloc,
                                        SharedIndex* point_njtr_indices,
                                        double* point_jac,
                                        unsigned int point_jac_num_alloc,
                                        double* const out_pose_njtr,
                                        unsigned int out_pose_njtr_num_alloc,
                                        double* const out_calib_njtr,
                                        unsigned int out_calib_njtr_num_alloc,
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

  __shared__ SharedIndex calib_njtr_indices_loc[1024];
  calib_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_njtr_indices[global_thread_idx]
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
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        calib_jac, 2 * calib_jac_num_alloc, global_thread_idx, r4, r5);
    r2 = fma(r3, r5, r2);
  };
  load_shared<2, double, double>(calib_njtr,
                                 0 * calib_njtr_num_alloc,
                                 calib_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          calib_njtr_indices_loc[threadIdx.x].target,
                          r6,
                          r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        calib_jac, 0 * calib_jac_num_alloc, global_thread_idx, r8, r9);
    r2 = fma(r6, r9, r2);
  };
  load_shared<1, double, double>(point_njtr,
                                 2 * point_njtr_num_alloc,
                                 point_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, point_njtr_indices_loc[threadIdx.x].target, r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        point_jac, 4 * point_jac_num_alloc, global_thread_idx, r11, r12);
  };
  load_shared<2, double, double>(point_njtr,
                                 0 * point_njtr_num_alloc,
                                 point_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          point_njtr_indices_loc[threadIdx.x].target,
                          r13,
                          r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        point_jac, 2 * point_jac_num_alloc, global_thread_idx, r15, r16);
    r17 = fma(r14, r16, r10 * r12);
    read_idx_2<1024, double, double, double2>(
        point_jac, 0 * point_jac_num_alloc, global_thread_idx, r18, r19);
    r17 = fma(r13, r19, r17);
    r20 = r2 + r17;
    r3 = fma(r3, r4, r7);
    r3 = fma(r6, r8, r3);
    r14 = fma(r14, r15, r10 * r11);
    r14 = fma(r13, r18, r14);
    r13 = r3 + r14;
    r10 = fma(r0, r13, r1 * r20);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 2 * pose_jac_num_alloc, global_thread_idx, r6, r7);
    r21 = fma(r6, r13, r7 * r20);
    write_sum_2<double, double>((double*)inout_shared, r10, r21);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r21, r10);
    r22 = fma(r21, r13, r10 * r20);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 6 * pose_jac_num_alloc, global_thread_idx, r23, r24);
    r25 = fma(r23, r13, r24 * r20);
    write_sum_2<double, double>((double*)inout_shared, r22, r25);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r25, r22);
    r26 = fma(r25, r13, r22 * r20);
    read_idx_2<1024, double, double, double2>(
        pose_jac, 10 * pose_jac_num_alloc, global_thread_idx, r27, r28);
    r13 = fma(r27, r13, r28 * r20);
    write_sum_2<double, double>((double*)inout_shared, r26, r13);
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
                          r13,
                          r26);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r25 = fma(r13, r25, r26 * r27);
  };
  load_shared<2, double, double>(pose_njtr,
                                 2 * pose_njtr_num_alloc,
                                 pose_njtr_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          pose_njtr_indices_loc[threadIdx.x].target,
                          r27,
                          r20);
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
    r25 = fma(r20, r23, r25);
    r25 = fma(r29, r0, r25);
    r25 = fma(r30, r6, r25);
    r14 = r25 + r14;
    r24 = fma(r20, r24, r26 * r28);
    r24 = fma(r27, r10, r24);
    r24 = fma(r29, r1, r24);
    r24 = fma(r30, r7, r24);
    r24 = fma(r13, r22, r24);
    r17 = r24 + r17;
    r9 = fma(r9, r17, r8 * r14);
    write_sum_2<double, double>((double*)inout_shared, r9, r14);
  };
  flush_sum_shared<2, double>(out_calib_njtr,
                              0 * out_calib_njtr_num_alloc,
                              calib_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r5 = fma(r5, r17, r4 * r14);
    write_sum_2<double, double>((double*)inout_shared, r17, r5);
  };
  flush_sum_shared<2, double>(out_calib_njtr,
                              2 * out_calib_njtr_num_alloc,
                              calib_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r25 + r3;
    r2 = r24 + r2;
    r19 = fma(r19, r2, r18 * r3);
    r16 = fma(r16, r2, r15 * r3);
    write_sum_2<double, double>((double*)inout_shared, r19, r16);
  };
  flush_sum_shared<2, double>(out_point_njtr,
                              0 * out_point_njtr_num_alloc,
                              point_njtr_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = fma(r12, r2, r11 * r3);
    write_sum_1<double, double>((double*)inout_shared, r2);
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
                                  double* calib_njtr,
                                  unsigned int calib_njtr_num_alloc,
                                  SharedIndex* calib_njtr_indices,
                                  double* calib_jac,
                                  unsigned int calib_jac_num_alloc,
                                  double* point_njtr,
                                  unsigned int point_njtr_num_alloc,
                                  SharedIndex* point_njtr_indices,
                                  double* point_jac,
                                  unsigned int point_jac_num_alloc,
                                  double* const out_pose_njtr,
                                  unsigned int out_pose_njtr_num_alloc,
                                  double* const out_calib_njtr,
                                  unsigned int out_calib_njtr_num_alloc,
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
      calib_njtr,
      calib_njtr_num_alloc,
      calib_njtr_indices,
      calib_jac,
      calib_jac_num_alloc,
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar