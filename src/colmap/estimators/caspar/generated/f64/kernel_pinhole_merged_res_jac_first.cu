#include "kernel_pinhole_merged_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) pinhole_merged_res_jac_first_kernel(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
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
  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
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
      r46, r47, r48, r49, r50, r51;
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
    r0 = 1.00000000000000008e-15;
  };
  load_shared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r5);
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
  };
  load_shared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r21);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r22 = r8 * r8;
    r22 = r22 * r18;
    r23 = 1.00000000000000000e+00;
    r24 = r9 * r9;
    r25 = fma(r18, r24, r23);
    r26 = r22 + r25;
    r5 = fma(r6, r20, r5);
    r5 = fma(r21, r26, r5);
    r27 = copysign(1.0, r5);
    r27 = fma(r0, r27, r5);
    r0 = 1.0 / r27;
  };
  load_shared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r5, r28);
  };
  __syncthreads();
  load_shared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r29, r30);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r31 = r10 * r18;
    r32 = r11 * r31;
    r14 = r9 * r14;
    r33 = r32 + r14;
    r29 = fma(r7, r33, r29);
    r34 = r9 * r11;
    r34 = r34 * r13;
    r19 = r34 + r19;
    r35 = r10 * r31;
    r25 = r35 + r25;
    r29 = fma(r21, r19, r29);
    r29 = fma(r6, r25, r29);
    r36 = r5 * r29;
    r2 = fma(r0, r36, r2);
    r3 = fma(r3, r4, r1);
    r1 = r10 * r11;
    r1 = r1 * r13;
    r14 = r1 + r14;
    r30 = fma(r6, r14, r30);
    r13 = r8 * r11;
    r13 = r13 * r18;
    r12 = r12 + r13;
    r35 = r23 + r35;
    r35 = r35 + r22;
    r30 = fma(r21, r12, r30);
    r30 = fma(r7, r35, r30);
    r22 = r28 * r30;
    r3 = fma(r0, r22, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r37 = fma(r3, r3, r2 * r2);
  };
  sum_store<double>(out_rTr_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r37);
  if (global_thread_idx < problem_size) {
    r37 = r8 * r9;
    r37 = r37 * r18;
    r1 = r1 + r37;
    r1 = fma(r7, r19, r21 * r1);
    r18 = r5 * r1;
    r9 = r9 * r31;
    r13 = r13 + r9;
    r38 = r10 * r10;
    r39 = r4 * r24;
    r40 = r38 + r39;
    r11 = r11 * r11;
    r41 = r8 * r8;
    r41 = r41 * r4;
    r42 = r11 + r41;
    r43 = r40 + r42;
    r43 = fma(r7, r43, r21 * r13);
    r13 = r5 * r29;
    r44 = r27 * r27;
    r45 = 1.0 / r44;
    r13 = r13 * r4;
    r13 = r13 * r45;
    r18 = fma(r43, r13, r0 * r18);
    r46 = r4 * r45;
    r47 = r43 * r46;
    r48 = r8 * r8;
    r49 = r4 * r11;
    r50 = r48 + r49;
    r40 = r40 + r50;
    r40 = fma(r21, r40, r7 * r12);
    r51 = r28 * r40;
    r51 = fma(r0, r51, r22 * r47);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r18, r51);
    r10 = r10 * r10;
    r10 = r10 * r4;
    r47 = r24 + r10;
    r50 = r50 + r47;
    r50 = fma(r6, r50, r21 * r20);
    r31 = r8 * r31;
    r17 = r17 + r31;
    r11 = r48 + r11;
    r11 = r11 + r39;
    r11 = r11 + r10;
    r11 = fma(r21, r11, r6 * r17);
    r17 = r5 * r11;
    r17 = fma(r0, r17, r50 * r13);
    r9 = r15 + r9;
    r9 = fma(r6, r9, r21 * r14);
    r21 = r28 * r9;
    r15 = r50 * r46;
    r15 = fma(r22, r15, r0 * r21);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r17, r15);
    r38 = r24 + r38;
    r38 = r38 + r41;
    r38 = r38 + r49;
    r38 = fma(r7, r38, r6 * r33);
    r49 = r5 * r38;
    r31 = r34 + r31;
    r31 = fma(r7, r31, r6 * r16);
    r49 = fma(r31, r13, r0 * r49);
    r32 = r37 + r32;
    r47 = r42 + r47;
    r47 = fma(r6, r47, r7 * r32);
    r6 = r28 * r47;
    r32 = r31 * r46;
    r32 = fma(r22, r32, r0 * r6);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r49, r32);
    r6 = r5 * r0;
    r7 = r28 * r0;
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r6, r7);
    r7 = r46 * r22;
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r13, r7);
    r7 = r4 * r2;
    r6 = r4 * r3;
    r6 = fma(r51, r6, r18 * r7);
    r7 = r4 * r2;
    r42 = r4 * r3;
    r42 = fma(r15, r42, r17 * r7);
    write_sum_2<double, double>((double*)inout_shared, r6, r42);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r4 * r2;
    r6 = r4 * r3;
    r6 = fma(r32, r6, r49 * r42);
    r42 = r4 * r2;
    r42 = r42 * r0;
    r7 = r5 * r42;
    write_sum_2<double, double>((double*)inout_shared, r6, r7);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r28 * r4;
    r7 = r7 * r3;
    r7 = r7 * r0;
    r6 = r3 * r45;
    r37 = r2 * r45;
    r37 = fma(r36, r37, r22 * r6);
    write_sum_2<double, double>((double*)inout_shared, r7, r37);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              4 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fma(r18, r18, r51 * r51);
    r7 = fma(r15, r15, r17 * r17);
    write_sum_2<double, double>((double*)inout_shared, r37, r7);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              0 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r5 * r5;
    r7 = r7 * r45;
    r37 = fma(r32, r32, r49 * r49);
    write_sum_2<double, double>((double*)inout_shared, r37, r7);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              2 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r28 * r28;
    r7 = r7 * r45;
    r44 = r27 * r44;
    r27 = r27 * r44;
    r27 = 1.0 / r27;
    r37 = r30 * r27;
    r6 = r28 * r22;
    r34 = r5 * r29;
    r34 = r34 * r27;
    r34 = fma(r36, r34, r6 * r37);
    write_sum_2<double, double>((double*)inout_shared, r7, r34);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              4 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fma(r51, r15, r18 * r17);
    r7 = fma(r18, r49, r51 * r32);
    write_sum_2<double, double>((double*)inout_shared, r34, r7);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              0 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r5 * r18;
    r7 = r7 * r0;
    r34 = r28 * r51;
    r34 = r34 * r0;
    write_sum_2<double, double>((double*)inout_shared, r7, r34);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              2 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fma(r17, r49, r15 * r32);
    r7 = r51 * r46;
    r18 = fma(r18, r13, r22 * r7);
    write_sum_2<double, double>((double*)inout_shared, r18, r34);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              4 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r5 * r17;
    r34 = r34 * r0;
    r18 = r28 * r15;
    r18 = r18 * r0;
    write_sum_2<double, double>((double*)inout_shared, r34, r18);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              6 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = r5 * r49;
    r18 = r18 * r0;
    r34 = r15 * r46;
    r34 = fma(r22, r34, r17 * r13);
    write_sum_2<double, double>((double*)inout_shared, r34, r18);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              8 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = r28 * r32;
    r18 = r18 * r0;
    r34 = r32 * r46;
    r49 = fma(r49, r13, r22 * r34);
    write_sum_2<double, double>((double*)inout_shared, r18, r49);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              10 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = 0.00000000000000000e+00;
    r18 = r5 * r4;
    r44 = 1.0 / r44;
    r18 = r18 * r44;
    r18 = r18 * r36;
    write_sum_2<double, double>((double*)inout_shared, r49, r18);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              12 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = r4 * r44;
    r44 = r44 * r6;
    write_sum_1<double, double>((double*)inout_shared, r44);
  };
  flush_sum_shared<1, double>(out_pose_precond_tril,
                              14 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = r29 * r0;
    r6 = r30 * r0;
    write_idx_2<1024, double, double, double2>(
        out_calib_jac, 0 * out_calib_jac_num_alloc, global_thread_idx, r44, r6);
    r42 = r29 * r42;
    r18 = r4 * r30;
    r18 = r18 * r3;
    r18 = r18 * r0;
    write_sum_2<double, double>((double*)inout_shared, r42, r18);
  };
  flush_sum_shared<2, double>(out_calib_njtr,
                              0 * out_calib_njtr_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = r4 * r2;
    r42 = r4 * r3;
    write_sum_2<double, double>((double*)inout_shared, r18, r42);
  };
  flush_sum_shared<2, double>(out_calib_njtr,
                              2 * out_calib_njtr_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r29 * r29;
    r42 = r42 * r45;
    r18 = r30 * r30;
    r18 = r18 * r45;
    write_sum_2<double, double>((double*)inout_shared, r42, r18);
  };
  flush_sum_shared<2, double>(out_calib_precond_diag,
                              0 * out_calib_precond_diag_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<double, double>((double*)inout_shared, r23, r23);
  };
  flush_sum_shared<2, double>(out_calib_precond_diag,
                              2 * out_calib_precond_diag_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<double, double>((double*)inout_shared, r49, r44);
  };
  flush_sum_shared<2, double>(out_calib_precond_tril,
                              0 * out_calib_precond_tril_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<double, double>((double*)inout_shared, r6, r49);
  };
  flush_sum_shared<2, double>(out_calib_precond_tril,
                              4 * out_calib_precond_tril_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r5 * r25;
    r49 = fma(r20, r13, r0 * r49);
    r6 = r28 * r14;
    r44 = r20 * r46;
    r44 = fma(r22, r44, r0 * r6);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               0 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r49,
                                               r44);
    r6 = r5 * r33;
    r6 = fma(r16, r13, r0 * r6);
    r23 = r28 * r35;
    r18 = r16 * r46;
    r18 = fma(r22, r18, r0 * r23);
    write_idx_2<1024, double, double, double2>(
        out_point_jac, 2 * out_point_jac_num_alloc, global_thread_idx, r6, r18);
    r23 = r5 * r19;
    r23 = fma(r0, r23, r26 * r13);
    r13 = r28 * r12;
    r42 = r26 * r46;
    r42 = fma(r22, r42, r0 * r13);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               4 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r23,
                                               r42);
    r13 = r4 * r3;
    r22 = r4 * r2;
    r22 = fma(r49, r22, r44 * r13);
    r13 = r4 * r2;
    r0 = r4 * r3;
    r0 = fma(r18, r0, r6 * r13);
    write_sum_2<double, double>((double*)inout_shared, r22, r0);
  };
  flush_sum_shared<2, double>(out_point_njtr,
                              0 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r4 * r3;
    r22 = r4 * r2;
    r22 = fma(r23, r22, r42 * r0);
    write_sum_1<double, double>((double*)inout_shared, r22);
  };
  flush_sum_shared<1, double>(out_point_njtr,
                              2 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = fma(r44, r44, r49 * r49);
    r0 = fma(r6, r6, r18 * r18);
    write_sum_2<double, double>((double*)inout_shared, r22, r0);
  };
  flush_sum_shared<2, double>(out_point_precond_diag,
                              0 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = fma(r42, r42, r23 * r23);
    write_sum_1<double, double>((double*)inout_shared, r0);
  };
  flush_sum_shared<1, double>(out_point_precond_diag,
                              2 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = fma(r49, r6, r44 * r18);
    r44 = fma(r44, r42, r49 * r23);
    write_sum_2<double, double>((double*)inout_shared, r0, r44);
  };
  flush_sum_shared<2, double>(out_point_precond_tril,
                              0 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = fma(r18, r42, r6 * r23);
    write_sum_1<double, double>((double*)inout_shared, r42);
  };
  flush_sum_shared<1, double>(out_point_precond_tril,
                              2 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  sum_flush_final<double>(out_rTr_local, out_rTr, 1);
}

void pinhole_merged_res_jac_first(double* pose,
                                  unsigned int pose_num_alloc,
                                  SharedIndex* pose_indices,
                                  double* calib,
                                  unsigned int calib_num_alloc,
                                  SharedIndex* calib_indices,
                                  double* point,
                                  unsigned int point_num_alloc,
                                  SharedIndex* point_indices,
                                  double* pixel,
                                  unsigned int pixel_num_alloc,
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
  pinhole_merged_res_jac_first_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      calib,
      calib_num_alloc,
      calib_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
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