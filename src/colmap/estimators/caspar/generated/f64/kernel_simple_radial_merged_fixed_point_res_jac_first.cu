#include "kernel_simple_radial_merged_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_merged_fixed_point_res_jac_first_kernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* point,
        unsigned int point_num_alloc,
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
        double* out_calib_jac,
        unsigned int out_calib_jac_num_alloc,
        double* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        double* const out_calib_precond_diag,
        unsigned int out_calib_precond_diag_num_alloc,
        double* const out_calib_precond_tril,
        unsigned int out_calib_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52;
  load_shared<2, double, double>(
      calib, 2 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
  };
  load_shared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r0, r5);
  };
  __syncthreads();
  load_shared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9);
  };
  load_shared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r10, r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = r10 * r11;
    r13 = 2.00000000000000000e+00;
    r12 = r12 * r13;
  };
  load_shared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r14, r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = -2.00000000000000000e+00;
    r17 = r14 * r16;
    r18 = r15 * r17;
    r19 = r12 + r18;
    r6 = fma(r9, r19, r6);
    read_idx_1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r20);
    r21 = r10 * r14;
    r21 = r21 * r13;
    r22 = r11 * r15;
    r22 = r22 * r13;
    r23 = r21 + r22;
    r24 = r14 * r17;
    r25 = 1.00000000000000000e+00;
    r26 = r11 * r11;
    r27 = fma(r16, r26, r25);
    r28 = r24 + r27;
    r6 = fma(r20, r23, r6);
    r6 = fma(r8, r28, r6);
    r28 = r0 * r6;
    r29 = r6 * r6;
    r30 = 1.00000000000000008e-15;
  };
  load_shared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r31);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r32 = r11 * r14;
    r32 = r32 * r13;
    r33 = r10 * r15;
    r33 = r33 * r13;
    r34 = r32 + r33;
    r31 = fma(r9, r34, r31);
    r35 = r11 * r15;
    r35 = r35 * r16;
    r21 = r21 + r35;
    r36 = r10 * r10;
    r37 = r16 * r36;
    r27 = r37 + r27;
    r31 = fma(r8, r21, r31);
    r31 = fma(r20, r27, r31);
    r27 = copysign(1.0, r31);
    r27 = fma(r30, r27, r31);
    r30 = r27 * r27;
    r31 = 1.0 / r30;
    r38 = r14 * r15;
    r38 = r38 * r13;
    r12 = r12 + r38;
    r7 = fma(r8, r12, r7);
    r39 = r10 * r15;
    r39 = r39 * r16;
    r32 = r32 + r39;
    r24 = r25 + r24;
    r24 = r24 + r37;
    r7 = fma(r20, r32, r7);
    r7 = fma(r9, r24, r7);
    r24 = r7 * r7;
    r24 = fma(r31, r24, r31 * r29);
    r29 = fma(r5, r24, r25);
    r37 = 1.0 / r27;
    r40 = r29 * r37;
    r2 = fma(r28, r40, r2);
    r3 = fma(r3, r4, r1);
    r1 = r0 * r29;
    r1 = r1 * r37;
    r3 = fma(r7, r1, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r41 = fma(r3, r3, r2 * r2);
  };
  sum_store<double>(out_rTr_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r41);
  if (global_thread_idx < problem_size) {
    r41 = r11 * r17;
    r39 = r39 + r41;
    r42 = r14 * r14;
    r43 = r4 * r26;
    r44 = r42 + r43;
    r15 = r15 * r15;
    r45 = r4 * r36;
    r46 = r15 + r45;
    r47 = r44 + r46;
    r47 = fma(r9, r47, r20 * r39);
    r39 = r47 * r28;
    r48 = r29 * r4;
    r48 = r48 * r31;
    r49 = r16 * r6;
    r30 = r27 * r30;
    r27 = 1.0 / r30;
    r49 = r49 * r6;
    r49 = r49 * r47;
    r50 = r16 * r7;
    r50 = r50 * r7;
    r50 = r50 * r47;
    r50 = fma(r27, r50, r27 * r49);
    r49 = r13 * r7;
    r51 = r4 * r15;
    r52 = r36 + r51;
    r44 = r44 + r52;
    r44 = fma(r20, r44, r9 * r32);
    r49 = r49 * r44;
    r50 = fma(r31, r49, r50);
    r32 = r13 * r6;
    r11 = r10 * r11;
    r11 = r11 * r16;
    r38 = r38 + r11;
    r23 = fma(r9, r23, r20 * r38);
    r32 = r32 * r23;
    r50 = fma(r31, r32, r50);
    r32 = r5 * r50;
    r32 = r32 * r37;
    r32 = fma(r28, r32, r48 * r39);
    r32 = fma(r23, r1, r32);
    r23 = r0 * r31;
    r23 = r23 * r7;
    r23 = r23 * r29;
    r23 = r23 * r4;
    r44 = fma(r47, r23, r44 * r1);
    r39 = r0 * r50;
    r49 = r5 * r7;
    r39 = r39 * r37;
    r44 = fma(r49, r39, r44);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r32, r44);
    r39 = r16 * r7;
    r14 = r14 * r14;
    r14 = r14 * r4;
    r38 = r26 + r14;
    r52 = r52 + r38;
    r52 = fma(r8, r52, r20 * r21);
    r39 = r39 * r7;
    r39 = r39 * r52;
    r21 = r13 * r7;
    r41 = r33 + r41;
    r41 = fma(r8, r41, r20 * r12);
    r21 = r21 * r41;
    r21 = fma(r31, r21, r27 * r39);
    r39 = r16 * r6;
    r39 = r39 * r6;
    r39 = r39 * r52;
    r21 = fma(r27, r39, r21);
    r12 = r13 * r6;
    r17 = r10 * r17;
    r35 = r35 + r17;
    r15 = r36 + r15;
    r15 = r15 + r43;
    r15 = r15 + r14;
    r15 = fma(r20, r15, r8 * r35);
    r12 = r12 * r15;
    r21 = fma(r31, r12, r21);
    r12 = r5 * r21;
    r12 = r12 * r37;
    r39 = r52 * r28;
    r39 = fma(r48, r39, r28 * r12);
    r39 = fma(r15, r1, r39);
    r41 = fma(r41, r1, r52 * r23);
    r15 = r0 * r21;
    r15 = r15 * r37;
    r41 = fma(r49, r15, r41);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r39, r41);
    r15 = r13 * r7;
    r18 = r11 + r18;
    r38 = r46 + r38;
    r38 = fma(r8, r38, r9 * r18);
    r15 = r15 * r38;
    r18 = r16 * r6;
    r17 = r22 + r17;
    r17 = fma(r9, r17, r8 * r34);
    r18 = r18 * r6;
    r18 = r18 * r17;
    r18 = fma(r27, r18, r31 * r15);
    r15 = r16 * r7;
    r15 = r15 * r7;
    r15 = r15 * r17;
    r18 = fma(r27, r15, r18);
    r34 = r13 * r6;
    r42 = r26 + r42;
    r42 = r42 + r45;
    r42 = r42 + r51;
    r42 = fma(r9, r42, r8 * r19);
    r34 = r34 * r42;
    r18 = fma(r31, r34, r18);
    r34 = r5 * r18;
    r34 = r34 * r37;
    r15 = r17 * r28;
    r15 = fma(r48, r15, r28 * r34);
    r15 = fma(r42, r1, r15);
    r38 = fma(r38, r1, r17 * r23);
    r42 = r0 * r18;
    r42 = r42 * r37;
    r38 = fma(r49, r42, r38);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r15, r38);
    r42 = r0 * r5;
    r42 = r42 * r13;
    r42 = r42 * r6;
    r42 = r42 * r7;
    r42 = r42 * r27;
    r34 = r27 * r28;
    r9 = r13 * r34;
    r19 = r5 * r6;
    r9 = fma(r19, r9, r1);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r9, r42);
    r8 = r0 * r13;
    r8 = r8 * r7;
    r8 = r8 * r27;
    r8 = fma(r49, r8, r1);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r42, r8);
    r1 = r16 * r6;
    r1 = r1 * r6;
    r51 = r16 * r7;
    r51 = r51 * r7;
    r51 = fma(r27, r51, r27 * r1);
    r1 = r5 * r51;
    r1 = r1 * r37;
    r1 = fma(r28, r1, r28 * r48);
    r48 = r0 * r51;
    r48 = r48 * r37;
    r48 = fma(r49, r48, r23);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r1, r48);
    r23 = r4 * r2;
    r27 = r4 * r3;
    r27 = fma(r44, r27, r32 * r23);
    r23 = r4 * r2;
    r45 = r4 * r3;
    r45 = fma(r41, r45, r39 * r23);
    write_sum_2<double, double>((double*)inout_shared, r27, r45);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = r4 * r3;
    r27 = r4 * r2;
    r27 = fma(r15, r27, r38 * r45);
    r45 = r4 * r2;
    r23 = r16 * r3;
    r26 = r49 * r34;
    r23 = fma(r26, r23, r9 * r45);
    write_sum_2<double, double>((double*)inout_shared, r27, r23);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = r4 * r3;
    r27 = r4 * r2;
    r27 = fma(r1, r27, r48 * r23);
    r23 = r4 * r3;
    r45 = r16 * r2;
    r45 = fma(r26, r45, r8 * r23);
    write_sum_2<double, double>((double*)inout_shared, r45, r27);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              4 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r32, r32, r44 * r44);
    r45 = fma(r41, r41, r39 * r39);
    write_sum_2<double, double>((double*)inout_shared, r27, r45);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              0 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fma(r15, r15, r38 * r38);
    r27 = 4.00000000000000000e+00;
    r30 = r30 * r30;
    r30 = 1.0 / r30;
    r30 = r27 * r30;
    r27 = r0 * r7;
    r30 = r30 * r28;
    r30 = r30 * r49;
    r30 = r30 * r19;
    r30 = r30 * r27;
    r19 = fma(r9, r9, r30);
    write_sum_2<double, double>((double*)inout_shared, r45, r19);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              2 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r8, r8, r30);
    r19 = fma(r48, r48, r1 * r1);
    write_sum_2<double, double>((double*)inout_shared, r30, r19);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              4 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fma(r32, r39, r44 * r41);
    r30 = fma(r32, r15, r44 * r38);
    write_sum_2<double, double>((double*)inout_shared, r19, r30);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              0 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r44, r42, r32 * r9);
    r19 = fma(r32, r42, r44 * r8);
    write_sum_2<double, double>((double*)inout_shared, r30, r19);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              2 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = fma(r44, r48, r32 * r1);
    r32 = fma(r39, r15, r41 * r38);
    write_sum_2<double, double>((double*)inout_shared, r44, r32);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              4 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r41, r42, r39 * r9);
    r44 = fma(r39, r42, r41 * r8);
    write_sum_2<double, double>((double*)inout_shared, r32, r44);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              6 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r41, r48, r39 * r1);
    r39 = fma(r38, r42, r15 * r9);
    write_sum_2<double, double>((double*)inout_shared, r41, r39);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              8 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r38, r48, r15 * r1);
    r15 = fma(r15, r42, r38 * r8);
    write_sum_2<double, double>((double*)inout_shared, r15, r39);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              10 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r9, r42, r8 * r42);
    r9 = fma(r48, r42, r9 * r1);
    write_sum_2<double, double>((double*)inout_shared, r39, r9);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              12 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = fma(r1, r42, r8 * r48);
    write_sum_1<double, double>((double*)inout_shared, r42);
  };
  flush_sum_shared<1, double>(out_pose_precond_tril,
                              14 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r6 * r40;
    r1 = r7 * r40;
    write_idx_2<1024, double, double, double2>(
        out_calib_jac, 0 * out_calib_jac_num_alloc, global_thread_idx, r42, r1);
    r48 = r24 * r37;
    r48 = r48 * r28;
    r8 = r0 * r7;
    r8 = r8 * r24;
    r8 = r8 * r37;
    write_idx_2<1024, double, double, double2>(
        out_calib_jac, 2 * out_calib_jac_num_alloc, global_thread_idx, r48, r8);
    r9 = r7 * r4;
    r9 = r9 * r3;
    r39 = r6 * r4;
    r39 = r39 * r2;
    r39 = fma(r40, r39, r40 * r9);
    r9 = r24 * r4;
    r9 = r9 * r2;
    r9 = r9 * r37;
    r40 = r0 * r7;
    r40 = r40 * r24;
    r40 = r40 * r4;
    r40 = r40 * r3;
    r40 = fma(r37, r40, r28 * r9);
    write_sum_2<double, double>((double*)inout_shared, r39, r40);
  };
  flush_sum_shared<2, double>(out_calib_njtr,
                              0 * out_calib_njtr_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r4 * r2;
    r39 = r4 * r3;
    write_sum_2<double, double>((double*)inout_shared, r40, r39);
  };
  flush_sum_shared<2, double>(out_calib_njtr,
                              2 * out_calib_njtr_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r7 * r7;
    r39 = r39 * r29;
    r39 = r39 * r29;
    r40 = r6 * r6;
    r40 = r40 * r29;
    r40 = r40 * r29;
    r40 = fma(r31, r40, r31 * r39);
    r39 = r0 * r31;
    r9 = r24 * r24;
    r39 = r39 * r9;
    r9 = r7 * r39;
    r37 = r6 * r28;
    r37 = fma(r39, r37, r27 * r9);
    write_sum_2<double, double>((double*)inout_shared, r40, r37);
  };
  flush_sum_shared<2, double>(out_calib_precond_diag,
                              0 * out_calib_precond_diag_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<double, double>((double*)inout_shared, r25, r25);
  };
  flush_sum_shared<2, double>(out_calib_precond_diag,
                              2 * out_calib_precond_diag_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r6 * r24;
    r25 = r25 * r29;
    r25 = r25 * r31;
    r37 = r0 * r7;
    r37 = r37 * r7;
    r37 = r37 * r24;
    r37 = r37 * r29;
    r37 = fma(r31, r37, r28 * r25);
    write_sum_2<double, double>((double*)inout_shared, r37, r42);
  };
  flush_sum_shared<2, double>(out_calib_precond_tril,
                              0 * out_calib_precond_tril_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<double, double>((double*)inout_shared, r1, r48);
  };
  flush_sum_shared<2, double>(out_calib_precond_tril,
                              2 * out_calib_precond_tril_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = 0.00000000000000000e+00;
    write_sum_2<double, double>((double*)inout_shared, r8, r48);
  };
  flush_sum_shared<2, double>(out_calib_precond_tril,
                              4 * out_calib_precond_tril_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  sum_flush_final<double>(out_rTr_local, out_rTr, 1);
}

void simple_radial_merged_fixed_point_res_jac_first(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* point,
    unsigned int point_num_alloc,
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
    double* out_calib_jac,
    unsigned int out_calib_jac_num_alloc,
    double* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    double* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    double* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_merged_fixed_point_res_jac_first_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
      point,
      point_num_alloc,
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
      out_calib_jac,
      out_calib_jac_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar