#include "kernel_pinhole_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first_kernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* focal,
        unsigned int focal_num_alloc,
        double* extra_calib,
        unsigned int extra_calib_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        extra_calib, 0 * extra_calib_num_alloc, global_thread_idx, r0, r1);
    read_idx_2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
    r0 = 1.00000000000000008e-15;
  };
  load_shared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r6, r7);
  };
  load_shared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  load_shared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r10, r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = r9 * r10;
    r13 = 2.00000000000000000e+00;
    r12 = r12 * r13;
    r14 = r8 * r13;
    r15 = r11 * r14;
    r16 = r12 + r15;
    r5 = fma(r7, r16, r5);
    r17 = r9 * r11;
    r18 = -2.00000000000000000e+00;
    r17 = r17 * r18;
    r19 = r10 * r14;
    r20 = r17 + r19;
    read_idx_1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r21);
    r22 = r8 * r8;
    r22 = r22 * r18;
    r23 = 1.00000000000000000e+00;
    r24 = r9 * r9;
    r25 = fma(r18, r24, r23);
    r26 = r22 + r25;
    r5 = fma(r6, r20, r5);
    r5 = fma(r21, r26, r5);
    r26 = copysign(1.0, r5);
    r26 = fma(r0, r26, r5);
    r0 = 1.0 / r26;
    read_idx_2<1024, double, double, double2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r5, r27);
  };
  load_shared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r28, r29);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r30 = r10 * r18;
    r31 = r11 * r30;
    r14 = r9 * r14;
    r32 = r31 + r14;
    r28 = fma(r7, r32, r28);
    r33 = r9 * r11;
    r33 = r33 * r13;
    r19 = r33 + r19;
    r34 = r10 * r30;
    r25 = r34 + r25;
    r28 = fma(r21, r19, r28);
    r28 = fma(r6, r25, r28);
    r25 = r5 * r28;
    r2 = fma(r0, r25, r2);
    r3 = fma(r3, r4, r1);
    r1 = r10 * r11;
    r1 = r1 * r13;
    r14 = r1 + r14;
    r29 = fma(r6, r14, r29);
    r13 = r8 * r11;
    r13 = r13 * r18;
    r12 = r12 + r13;
    r34 = r23 + r34;
    r34 = r34 + r22;
    r29 = fma(r21, r12, r29);
    r29 = fma(r7, r34, r29);
    r34 = r27 * r29;
    r3 = fma(r0, r34, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r22 = fma(r2, r2, r3 * r3);
  };
  sum_store<double>(out_rTr_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r22);
  if (global_thread_idx < problem_size) {
    r22 = r4 * r3;
    r23 = r10 * r10;
    r11 = r11 * r11;
    r35 = r4 * r11;
    r36 = r23 + r35;
    r37 = r8 * r8;
    r38 = r4 * r24;
    r39 = r37 + r38;
    r40 = r36 + r39;
    r40 = fma(r21, r40, r7 * r12);
    r12 = r27 * r40;
    r41 = r9 * r30;
    r13 = r13 + r41;
    r38 = r23 + r38;
    r23 = r8 * r8;
    r23 = r23 * r4;
    r42 = r11 + r23;
    r38 = r38 + r42;
    r38 = fma(r7, r38, r21 * r13);
    r13 = r26 * r26;
    r43 = 1.0 / r13;
    r44 = r4 * r43;
    r45 = r38 * r44;
    r45 = fma(r34, r45, r0 * r12);
    r12 = r4 * r2;
    r46 = r44 * r25;
    r9 = r8 * r9;
    r9 = r9 * r18;
    r1 = r1 + r9;
    r19 = fma(r7, r19, r21 * r1);
    r1 = r5 * r19;
    r1 = fma(r0, r1, r38 * r46);
    r12 = fma(r1, r12, r45 * r22);
    r22 = r4 * r2;
    r30 = r8 * r30;
    r17 = r17 + r30;
    r10 = r10 * r10;
    r10 = r10 * r4;
    r11 = r11 + r10;
    r11 = r11 + r39;
    r11 = fma(r21, r11, r6 * r17);
    r17 = r5 * r11;
    r35 = r37 + r35;
    r10 = r24 + r10;
    r35 = r35 + r10;
    r35 = fma(r6, r35, r21 * r20);
    r17 = fma(r35, r46, r0 * r17);
    r20 = r4 * r3;
    r37 = r35 * r44;
    r41 = r15 + r41;
    r41 = fma(r6, r41, r21 * r14);
    r14 = r27 * r41;
    r14 = fma(r0, r14, r34 * r37);
    r20 = fma(r14, r20, r17 * r22);
    write_sum_2<double, double>((double*)inout_shared, r12, r20);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r4 * r2;
    r30 = r33 + r30;
    r30 = fma(r7, r30, r6 * r16);
    r23 = r24 + r23;
    r23 = r23 + r36;
    r23 = fma(r7, r23, r6 * r32);
    r32 = r5 * r23;
    r32 = fma(r0, r32, r30 * r46);
    r36 = r4 * r3;
    r24 = r30 * r44;
    r31 = r9 + r31;
    r10 = r42 + r10;
    r10 = fma(r6, r10, r7 * r31);
    r6 = r27 * r10;
    r6 = fma(r0, r6, r34 * r24);
    r36 = fma(r6, r36, r32 * r20);
    r20 = r5 * r4;
    r20 = r20 * r2;
    r20 = r20 * r0;
    write_sum_2<double, double>((double*)inout_shared, r36, r20);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r27 * r4;
    r20 = r20 * r3;
    r20 = r20 * r0;
    r36 = r3 * r43;
    r24 = r2 * r43;
    r24 = fma(r25, r24, r34 * r36);
    write_sum_2<double, double>((double*)inout_shared, r20, r24);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              4 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r45, r45, r1 * r1);
    r20 = fma(r17, r17, r14 * r14);
    write_sum_2<double, double>((double*)inout_shared, r24, r20);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              0 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r5 * r5;
    r20 = r20 * r43;
    r24 = fma(r6, r6, r32 * r32);
    write_sum_2<double, double>((double*)inout_shared, r24, r20);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              2 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r27 * r27;
    r20 = r20 * r43;
    r24 = r5 * r28;
    r13 = r26 * r13;
    r26 = r26 * r13;
    r26 = 1.0 / r26;
    r24 = r24 * r26;
    r36 = r29 * r26;
    r31 = r27 * r34;
    r36 = fma(r31, r36, r25 * r24);
    write_sum_2<double, double>((double*)inout_shared, r20, r36);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              4 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = fma(r45, r14, r1 * r17);
    r20 = fma(r1, r32, r45 * r6);
    write_sum_2<double, double>((double*)inout_shared, r36, r20);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              0 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r5 * r1;
    r20 = r20 * r0;
    r36 = r27 * r45;
    r36 = r36 * r0;
    write_sum_2<double, double>((double*)inout_shared, r20, r36);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              2 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = fma(r14, r6, r17 * r32);
    r20 = r45 * r44;
    r1 = fma(r1, r46, r34 * r20);
    write_sum_2<double, double>((double*)inout_shared, r1, r36);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              4 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = r5 * r17;
    r36 = r36 * r0;
    r1 = r27 * r14;
    r1 = r1 * r0;
    write_sum_2<double, double>((double*)inout_shared, r36, r1);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              6 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r5 * r32;
    r1 = r1 * r0;
    r36 = r14 * r44;
    r17 = fma(r17, r46, r34 * r36);
    write_sum_2<double, double>((double*)inout_shared, r17, r1);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              8 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r27 * r6;
    r1 = r1 * r0;
    r0 = r6 * r44;
    r0 = fma(r34, r0, r32 * r46);
    write_sum_2<double, double>((double*)inout_shared, r1, r0);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              10 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
    r25 = r5 * r25;
    r13 = 1.0 / r13;
    r13 = r4 * r13;
    r25 = r25 * r13;
    write_sum_2<double, double>((double*)inout_shared, r0, r25);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              12 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r13 * r31;
    write_sum_1<double, double>((double*)inout_shared, r31);
  };
  flush_sum_shared<1, double>(out_pose_precond_tril,
                              14 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  sum_flush_final<double>(out_rTr_local, out_rTr, 1);
}

void pinhole_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* focal,
    unsigned int focal_num_alloc,
    double* extra_calib,
    unsigned int extra_calib_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first_kernel<<<
      n_blocks,
      1024>>>(pose,
              pose_num_alloc,
              pose_indices,
              pixel,
              pixel_num_alloc,
              focal,
              focal_num_alloc,
              extra_calib,
              extra_calib_num_alloc,
              point,
              point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_rTr,
              out_pose_njtr,
              out_pose_njtr_num_alloc,
              out_pose_precond_diag,
              out_pose_precond_diag_num_alloc,
              out_pose_precond_tril,
              out_pose_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar