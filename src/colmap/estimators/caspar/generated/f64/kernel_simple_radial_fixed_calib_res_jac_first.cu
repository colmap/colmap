#include "kernel_simple_radial_fixed_calib_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_calib_res_jac_first_kernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* calib,
        unsigned int calib_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
        double* out_pose_jac,
        unsigned int out_pose_jac_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        double* out_point_jac,
        unsigned int out_point_jac_num_alloc,
        double* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        double* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        double* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        calib, 0 * calib_num_alloc, global_thread_idx, r0, r1);
    read_idx_2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r1);
  };
  load_shared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r1, r5);
  };
  __syncthreads();
  load_shared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  load_shared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = r8 * r9;
    r11 = -2.00000000000000000e+00;
    r10 = r10 * r11;
  };
  load_shared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r12, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r14 = r12 * r13;
    r15 = 2.00000000000000000e+00;
    r14 = r14 * r15;
    r16 = r10 + r14;
    r1 = fma(r7, r16, r1);
  };
  load_shared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r17);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r18 = r12 * r8;
    r18 = r18 * r15;
    r19 = r13 * r9;
    r19 = r19 * r15;
    r20 = r18 + r19;
    r21 = r8 * r8;
    r22 = r11 * r21;
    r23 = 1.00000000000000000e+00;
    r24 = r13 * r13;
    r25 = fma(r11, r24, r23);
    r26 = r22 + r25;
    r1 = fma(r17, r20, r1);
    r1 = fma(r6, r26, r1);
    r27 = r0 * r1;
    read_idx_2<1024, double, double, double2>(
        calib, 2 * calib_num_alloc, global_thread_idx, r28, r29);
    r30 = r1 * r1;
    r31 = 1.00000000000000008e-15;
  };
  load_shared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r33 = r13 * r8;
    r33 = r33 * r15;
    r34 = r12 * r9;
    r34 = r34 * r15;
    r35 = r33 + r34;
    r32 = fma(r7, r35, r32);
    r36 = r13 * r9;
    r36 = r36 * r11;
    r18 = r18 + r36;
    r37 = r12 * r12;
    r38 = r11 * r37;
    r25 = r38 + r25;
    r32 = fma(r6, r18, r32);
    r32 = fma(r17, r25, r32);
    r39 = copysign(1.0, r32);
    r39 = fma(r31, r39, r32);
    r31 = r39 * r39;
    r32 = 1.0 / r31;
    r40 = r8 * r9;
    r40 = r40 * r15;
    r14 = r14 + r40;
    r5 = fma(r6, r14, r5);
    r41 = r12 * r9;
    r41 = r41 * r11;
    r33 = r33 + r41;
    r22 = r23 + r22;
    r22 = r22 + r38;
    r5 = fma(r17, r33, r5);
    r5 = fma(r7, r22, r5);
    r38 = r5 * r5;
    r38 = fma(r32, r38, r32 * r30);
    r38 = fma(r29, r38, r23);
    r23 = 1.0 / r39;
    r30 = r38 * r23;
    r2 = fma(r27, r30, r2);
    r3 = fma(r3, r4, r28);
    r28 = r0 * r38;
    r28 = r28 * r23;
    r3 = fma(r5, r28, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r30 = fma(r2, r2, r3 * r3);
  };
  sum_store<double>(out_rTr_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r30);
  if (global_thread_idx < problem_size) {
    r30 = r13 * r8;
    r30 = r30 * r11;
    r41 = r41 + r30;
    r42 = r4 * r24;
    r43 = r21 + r42;
    r9 = r9 * r9;
    r44 = r4 * r37;
    r45 = r9 + r44;
    r46 = r43 + r45;
    r46 = fma(r7, r46, r17 * r41);
    r41 = r1 * r1;
    r41 = r11 * r41;
    r31 = r39 * r31;
    r39 = 1.0 / r31;
    r41 = r41 * r39;
    r47 = r5 * r5;
    r47 = r11 * r47;
    r47 = r47 * r39;
    r48 = fma(r46, r47, r46 * r41);
    r49 = r15 * r5;
    r50 = r4 * r9;
    r51 = r37 + r50;
    r43 = r43 + r51;
    r43 = fma(r17, r43, r7 * r33);
    r49 = r49 * r43;
    r48 = fma(r32, r49, r48);
    r52 = r15 * r1;
    r13 = r12 * r13;
    r13 = r13 * r11;
    r40 = r40 + r13;
    r40 = fma(r7, r20, r17 * r40);
    r52 = r52 * r40;
    r48 = fma(r32, r52, r48);
    r52 = r29 * r48;
    r52 = r52 * r23;
    r40 = fma(r40, r28, r27 * r52);
    r52 = r46 * r27;
    r38 = r38 * r4;
    r38 = r38 * r32;
    r40 = fma(r38, r52, r40);
    r52 = r0 * r48;
    r49 = r29 * r5;
    r52 = r52 * r23;
    r52 = fma(r49, r52, r43 * r28);
    r43 = r0 * r5;
    r43 = r43 * r46;
    r52 = fma(r38, r43, r52);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r40, r52);
    r8 = r12 * r8;
    r8 = r8 * r11;
    r36 = r36 + r8;
    r9 = r37 + r9;
    r37 = r4 * r21;
    r9 = r9 + r42;
    r9 = r9 + r37;
    r9 = fma(r17, r9, r6 * r36);
    r37 = r24 + r37;
    r51 = r51 + r37;
    r51 = fma(r6, r51, r17 * r18);
    r36 = r15 * r5;
    r30 = r34 + r30;
    r30 = fma(r6, r30, r17 * r14);
    r36 = r36 * r30;
    r36 = fma(r32, r36, r51 * r47);
    r17 = r15 * r1;
    r17 = r17 * r9;
    r36 = fma(r32, r17, r36);
    r36 = fma(r51, r41, r36);
    r17 = r29 * r36;
    r17 = r17 * r23;
    r17 = fma(r27, r17, r9 * r28);
    r9 = r51 * r27;
    r17 = fma(r38, r9, r17);
    r9 = r0 * r5;
    r9 = r9 * r51;
    r30 = fma(r30, r28, r38 * r9);
    r9 = r0 * r36;
    r9 = r9 * r23;
    r30 = fma(r49, r9, r30);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r17, r30);
    r21 = r24 + r21;
    r21 = r21 + r44;
    r21 = r21 + r50;
    r21 = fma(r7, r21, r6 * r16);
    r50 = r15 * r5;
    r13 = r10 + r13;
    r37 = r45 + r37;
    r37 = fma(r6, r37, r7 * r13);
    r50 = r50 * r37;
    r8 = r19 + r8;
    r8 = fma(r7, r8, r6 * r35);
    r50 = fma(r8, r41, r32 * r50);
    r7 = r15 * r1;
    r7 = r7 * r21;
    r50 = fma(r32, r7, r50);
    r50 = fma(r8, r47, r50);
    r7 = r29 * r50;
    r7 = r7 * r23;
    r7 = fma(r27, r7, r21 * r28);
    r21 = r8 * r27;
    r7 = fma(r38, r21, r7);
    r21 = r0 * r5;
    r21 = r21 * r8;
    r37 = fma(r37, r28, r38 * r21);
    r21 = r0 * r50;
    r21 = r21 * r23;
    r37 = fma(r49, r21, r37);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r7, r37);
    r21 = r0 * r29;
    r21 = r21 * r15;
    r21 = r21 * r1;
    r21 = r21 * r5;
    r21 = r21 * r39;
    r6 = r29 * r15;
    r6 = r6 * r1;
    r6 = r6 * r39;
    r6 = fma(r27, r6, r28);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r6, r21);
    r19 = r0 * r15;
    r19 = r19 * r5;
    r19 = r19 * r39;
    r19 = fma(r49, r19, r28);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r21, r19);
    r13 = r41 + r47;
    r45 = r29 * r13;
    r45 = r45 * r23;
    r45 = fma(r27, r45, r27 * r38);
    r10 = r0 * r13;
    r10 = r10 * r23;
    r44 = r0 * r5;
    r44 = fma(r38, r44, r49 * r10);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r45, r44);
    r10 = r4 * r3;
    r24 = r4 * r2;
    r24 = fma(r40, r24, r52 * r10);
    r10 = r4 * r2;
    r9 = r4 * r3;
    r9 = fma(r30, r9, r17 * r10);
    write_sum_2<double, double>((double*)inout_shared, r24, r9);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = r4 * r3;
    r24 = r4 * r2;
    r24 = fma(r7, r24, r37 * r9);
    r9 = r4 * r2;
    r39 = r11 * r39;
    r11 = r3 * r39;
    r10 = r27 * r49;
    r11 = fma(r10, r11, r6 * r9);
    write_sum_2<double, double>((double*)inout_shared, r24, r11);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = r4 * r2;
    r24 = r4 * r3;
    r24 = fma(r44, r24, r45 * r11);
    r11 = r4 * r3;
    r9 = r2 * r39;
    r9 = fma(r10, r9, r19 * r11);
    write_sum_2<double, double>((double*)inout_shared, r9, r24);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              4 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r52, r52, r40 * r40);
    r9 = fma(r17, r17, r30 * r30);
    write_sum_2<double, double>((double*)inout_shared, r24, r9);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              0 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = fma(r37, r37, r7 * r7);
    r24 = r0 * r29;
    r11 = 4.00000000000000000e+00;
    r31 = r31 * r31;
    r31 = 1.0 / r31;
    r24 = r24 * r1;
    r24 = r24 * r5;
    r24 = r24 * r11;
    r24 = r24 * r31;
    r24 = r24 * r10;
    r10 = fma(r6, r6, r24);
    write_sum_2<double, double>((double*)inout_shared, r9, r10);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              2 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r19, r19, r24);
    r10 = fma(r45, r45, r44 * r44);
    write_sum_2<double, double>((double*)inout_shared, r24, r10);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              4 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fma(r40, r17, r52 * r30);
    r24 = fma(r40, r7, r52 * r37);
    write_sum_2<double, double>((double*)inout_shared, r10, r24);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              0 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r52, r21, r40 * r6);
    r10 = fma(r40, r21, r52 * r19);
    write_sum_2<double, double>((double*)inout_shared, r24, r10);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              2 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fma(r40, r45, r52 * r44);
    r52 = fma(r17, r7, r30 * r37);
    write_sum_2<double, double>((double*)inout_shared, r40, r52);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              4 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fma(r30, r21, r17 * r6);
    r40 = fma(r17, r21, r30 * r19);
    write_sum_2<double, double>((double*)inout_shared, r52, r40);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              6 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = fma(r17, r45, r30 * r44);
    r30 = fma(r37, r21, r7 * r6);
    write_sum_2<double, double>((double*)inout_shared, r17, r30);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              8 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r37, r44, r7 * r45);
    r7 = fma(r7, r21, r37 * r19);
    write_sum_2<double, double>((double*)inout_shared, r7, r30);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              10 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r6, r21, r19 * r21);
    r6 = fma(r44, r21, r6 * r45);
    write_sum_2<double, double>((double*)inout_shared, r30, r6);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              12 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = fma(r45, r21, r19 * r44);
    write_sum_1<double, double>((double*)inout_shared, r21);
  };
  flush_sum_shared<1, double>(out_pose_precond_tril,
                              14 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = r15 * r26;
    r21 = r21 * r1;
    r21 = fma(r32, r21, r18 * r41);
    r45 = r15 * r14;
    r45 = r45 * r5;
    r21 = fma(r32, r45, r21);
    r21 = fma(r18, r47, r21);
    r45 = r29 * r21;
    r45 = r45 * r23;
    r45 = fma(r26, r28, r27 * r45);
    r18 = r18 * r38;
    r45 = fma(r27, r18, r45);
    r44 = r0 * r5;
    r44 = fma(r14, r28, r18 * r44);
    r18 = r0 * r21;
    r18 = r18 * r23;
    r44 = fma(r49, r18, r44);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               0 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r45,
                                               r44);
    r18 = r35 * r27;
    r18 = fma(r16, r28, r38 * r18);
    r19 = r15 * r16;
    r19 = r19 * r1;
    r19 = fma(r35, r41, r32 * r19);
    r6 = r15 * r22;
    r6 = r6 * r5;
    r19 = fma(r32, r6, r19);
    r19 = fma(r35, r47, r19);
    r6 = r29 * r19;
    r6 = r6 * r23;
    r18 = fma(r27, r6, r18);
    r6 = r0 * r35;
    r6 = r6 * r5;
    r6 = fma(r22, r28, r38 * r6);
    r30 = r0 * r19;
    r30 = r30 * r23;
    r6 = fma(r49, r30, r6);
    write_idx_2<1024, double, double, double2>(
        out_point_jac, 2 * out_point_jac_num_alloc, global_thread_idx, r18, r6);
    r30 = r15 * r20;
    r30 = r30 * r1;
    r41 = fma(r25, r41, r32 * r30);
    r30 = r15 * r33;
    r30 = r30 * r5;
    r41 = fma(r32, r30, r41);
    r41 = fma(r25, r47, r41);
    r47 = r29 * r41;
    r47 = r47 * r23;
    r47 = fma(r20, r28, r27 * r47);
    r30 = r25 * r27;
    r47 = fma(r38, r30, r47);
    r30 = r0 * r41;
    r30 = r30 * r23;
    r28 = fma(r33, r28, r49 * r30);
    r30 = r0 * r25;
    r30 = r30 * r5;
    r28 = fma(r38, r30, r28);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               4 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r47,
                                               r28);
    r30 = r4 * r2;
    r38 = r4 * r3;
    r38 = fma(r44, r38, r45 * r30);
    r30 = r4 * r2;
    r49 = r4 * r3;
    r49 = fma(r6, r49, r18 * r30);
    write_sum_2<double, double>((double*)inout_shared, r38, r49);
  };
  flush_sum_shared<2, double>(out_point_njtr,
                              0 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r4 * r2;
    r38 = r4 * r3;
    r38 = fma(r28, r38, r47 * r49);
    write_sum_1<double, double>((double*)inout_shared, r38);
  };
  flush_sum_shared<1, double>(out_point_njtr,
                              2 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = fma(r44, r44, r45 * r45);
    r49 = fma(r6, r6, r18 * r18);
    write_sum_2<double, double>((double*)inout_shared, r38, r49);
  };
  flush_sum_shared<2, double>(out_point_precond_diag,
                              0 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = fma(r47, r47, r28 * r28);
    write_sum_1<double, double>((double*)inout_shared, r49);
  };
  flush_sum_shared<1, double>(out_point_precond_diag,
                              2 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = fma(r45, r18, r44 * r6);
    r44 = fma(r44, r28, r45 * r47);
    write_sum_2<double, double>((double*)inout_shared, r49, r44);
  };
  flush_sum_shared<2, double>(out_point_precond_tril,
                              0 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = fma(r6, r28, r18 * r47);
    write_sum_1<double, double>((double*)inout_shared, r28);
  };
  flush_sum_shared<1, double>(out_point_precond_tril,
                              2 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  sum_flush_final<double>(out_rTr_local, out_rTr, 1);
}

void simple_radial_fixed_calib_res_jac_first(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* calib,
    unsigned int calib_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    double* out_point_jac,
    unsigned int out_point_jac_num_alloc,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    double* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_calib_res_jac_first_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      calib,
      calib_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      out_point_jac,
      out_point_jac_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar