#include "kernel_simple_radial_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_point_res_jac_kernel(
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50;
  load_shared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
  if (global_thread_idx < problem_size) {
    r10 = r8 * r9;
    r11 = 2.00000000000000000e+00;
    r10 = r10 * r11;
  };
  load_shared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r12, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r14 = -2.00000000000000000e+00;
    r15 = r12 * r14;
    r16 = r13 * r15;
    r17 = r10 + r16;
    r1 = fma(r7, r17, r1);
    read_idx_1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r18);
    r19 = r8 * r12;
    r19 = r19 * r11;
    r20 = r9 * r13;
    r20 = r20 * r11;
    r21 = r19 + r20;
    r22 = r12 * r15;
    r23 = 1.00000000000000000e+00;
    r24 = r9 * r9;
    r25 = fma(r14, r24, r23);
    r26 = r22 + r25;
    r1 = fma(r18, r21, r1);
    r1 = fma(r6, r26, r1);
    r26 = r0 * r1;
  };
  load_shared<2, double, double>(
      calib, 2 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r27, r28);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r29 = r1 * r1;
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
    r32 = r9 * r12;
    r32 = r32 * r11;
    r33 = r8 * r13;
    r33 = r33 * r11;
    r34 = r32 + r33;
    r31 = fma(r7, r34, r31);
    r35 = r9 * r13;
    r35 = r35 * r14;
    r19 = r19 + r35;
    r36 = r8 * r8;
    r37 = r14 * r36;
    r25 = r37 + r25;
    r31 = fma(r6, r19, r31);
    r31 = fma(r18, r25, r31);
    r25 = copysign(1.0, r31);
    r25 = fma(r30, r25, r31);
    r30 = r25 * r25;
    r31 = 1.0 / r30;
    r38 = r12 * r13;
    r38 = r38 * r11;
    r10 = r10 + r38;
    r5 = fma(r6, r10, r5);
    r39 = r8 * r13;
    r39 = r39 * r14;
    r32 = r32 + r39;
    r22 = r23 + r22;
    r22 = r22 + r37;
    r5 = fma(r18, r32, r5);
    r5 = fma(r7, r22, r5);
    r22 = r5 * r5;
    r22 = fma(r31, r22, r31 * r29);
    r29 = fma(r28, r22, r23);
    r37 = 1.0 / r25;
    r40 = r29 * r37;
    r2 = fma(r26, r40, r2);
    r3 = fma(r3, r4, r27);
    r27 = r0 * r29;
    r27 = r27 * r37;
    r3 = fma(r5, r27, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r41 = r14 * r1;
    r42 = r9 * r15;
    r39 = r39 + r42;
    r43 = r12 * r12;
    r44 = r4 * r24;
    r45 = r43 + r44;
    r13 = r13 * r13;
    r46 = r4 * r36;
    r47 = r13 + r46;
    r48 = r45 + r47;
    r48 = fma(r7, r48, r18 * r39);
    r30 = r25 * r30;
    r25 = 1.0 / r30;
    r41 = r41 * r1;
    r41 = r41 * r48;
    r39 = r14 * r5;
    r39 = r39 * r5;
    r39 = r39 * r48;
    r39 = fma(r25, r39, r25 * r41);
    r41 = r11 * r5;
    r49 = r4 * r13;
    r50 = r36 + r49;
    r45 = r45 + r50;
    r45 = fma(r18, r45, r7 * r32);
    r41 = r41 * r45;
    r39 = fma(r31, r41, r39);
    r32 = r11 * r1;
    r9 = r8 * r9;
    r9 = r9 * r14;
    r38 = r38 + r9;
    r21 = fma(r7, r21, r18 * r38);
    r32 = r32 * r21;
    r39 = fma(r31, r32, r39);
    r32 = r28 * r39;
    r32 = r32 * r37;
    r21 = fma(r21, r27, r26 * r32);
    r32 = r48 * r26;
    r41 = r29 * r4;
    r41 = r41 * r31;
    r21 = fma(r41, r32, r21);
    r32 = r0 * r39;
    r38 = r28 * r5;
    r32 = r32 * r37;
    r32 = fma(r38, r32, r45 * r27);
    r45 = r0 * r31;
    r45 = r45 * r5;
    r45 = r45 * r29;
    r45 = r45 * r4;
    r32 = fma(r48, r45, r32);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r21, r32);
    r15 = r8 * r15;
    r35 = r35 + r15;
    r13 = r36 + r13;
    r12 = r12 * r12;
    r12 = r12 * r4;
    r13 = r13 + r44;
    r13 = r13 + r12;
    r13 = fma(r18, r13, r6 * r35);
    r35 = r14 * r5;
    r12 = r24 + r12;
    r50 = r50 + r12;
    r50 = fma(r6, r50, r18 * r19);
    r35 = r35 * r5;
    r35 = r35 * r50;
    r19 = r11 * r5;
    r42 = r33 + r42;
    r42 = fma(r6, r42, r18 * r10);
    r19 = r19 * r42;
    r19 = fma(r31, r19, r25 * r35);
    r35 = r14 * r1;
    r35 = r35 * r1;
    r35 = r35 * r50;
    r19 = fma(r25, r35, r19);
    r10 = r11 * r1;
    r10 = r10 * r13;
    r19 = fma(r31, r10, r19);
    r10 = r28 * r19;
    r10 = r10 * r37;
    r10 = fma(r26, r10, r13 * r27);
    r13 = r50 * r26;
    r10 = fma(r41, r13, r10);
    r42 = fma(r42, r27, r50 * r45);
    r13 = r0 * r19;
    r13 = r13 * r37;
    r42 = fma(r38, r13, r42);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r10, r42);
    r43 = r24 + r43;
    r43 = r43 + r46;
    r43 = r43 + r49;
    r43 = fma(r7, r43, r6 * r17);
    r17 = r11 * r5;
    r16 = r9 + r16;
    r12 = r47 + r12;
    r12 = fma(r6, r12, r7 * r16);
    r17 = r17 * r12;
    r16 = r14 * r1;
    r15 = r20 + r15;
    r15 = fma(r7, r15, r6 * r34);
    r16 = r16 * r1;
    r16 = r16 * r15;
    r16 = fma(r25, r16, r31 * r17);
    r17 = r14 * r5;
    r17 = r17 * r5;
    r17 = r17 * r15;
    r16 = fma(r25, r17, r16);
    r7 = r11 * r1;
    r7 = r7 * r43;
    r16 = fma(r31, r7, r16);
    r7 = r28 * r16;
    r7 = r7 * r37;
    r7 = fma(r26, r7, r43 * r27);
    r43 = r15 * r26;
    r7 = fma(r41, r43, r7);
    r12 = fma(r12, r27, r15 * r45);
    r43 = r0 * r16;
    r43 = r43 * r37;
    r12 = fma(r38, r43, r12);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r7, r12);
    r43 = r0 * r28;
    r43 = r43 * r11;
    r43 = r43 * r1;
    r43 = r43 * r5;
    r43 = r43 * r25;
    r17 = r25 * r26;
    r34 = r11 * r17;
    r6 = r28 * r1;
    r34 = fma(r6, r34, r27);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r34, r43);
    r20 = r0 * r11;
    r20 = r20 * r5;
    r20 = r20 * r25;
    r20 = fma(r38, r20, r27);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r43, r20);
    r27 = r14 * r1;
    r27 = r27 * r1;
    r47 = r14 * r5;
    r47 = r47 * r5;
    r47 = fma(r25, r47, r25 * r27);
    r27 = r28 * r47;
    r27 = r27 * r37;
    r27 = fma(r26, r27, r26 * r41);
    r41 = r0 * r47;
    r41 = r41 * r37;
    r41 = fma(r38, r41, r45);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r27, r41);
    r45 = r4 * r3;
    r25 = r4 * r2;
    r25 = fma(r21, r25, r32 * r45);
    r45 = r4 * r2;
    r9 = r4 * r3;
    r9 = fma(r42, r9, r10 * r45);
    write_sum_2<double, double>((double*)inout_shared, r25, r9);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = r4 * r3;
    r25 = r4 * r2;
    r25 = fma(r7, r25, r12 * r9);
    r9 = r4 * r2;
    r45 = r14 * r3;
    r49 = r38 * r17;
    r45 = fma(r49, r45, r34 * r9);
    write_sum_2<double, double>((double*)inout_shared, r25, r45);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = r4 * r2;
    r25 = r4 * r3;
    r25 = fma(r41, r25, r27 * r45);
    r45 = r4 * r3;
    r9 = r14 * r2;
    r9 = fma(r49, r9, r20 * r45);
    write_sum_2<double, double>((double*)inout_shared, r9, r25);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              4 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = fma(r32, r32, r21 * r21);
    r9 = fma(r10, r10, r42 * r42);
    write_sum_2<double, double>((double*)inout_shared, r25, r9);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              0 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = fma(r12, r12, r7 * r7);
    r25 = 4.00000000000000000e+00;
    r30 = r30 * r30;
    r30 = 1.0 / r30;
    r30 = r25 * r30;
    r25 = r0 * r5;
    r30 = r30 * r26;
    r30 = r30 * r38;
    r30 = r30 * r6;
    r30 = r30 * r25;
    r6 = fma(r34, r34, r30);
    write_sum_2<double, double>((double*)inout_shared, r9, r6);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              2 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r20, r20, r30);
    r6 = fma(r27, r27, r41 * r41);
    write_sum_2<double, double>((double*)inout_shared, r30, r6);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              4 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r21, r10, r32 * r42);
    r30 = fma(r21, r7, r32 * r12);
    write_sum_2<double, double>((double*)inout_shared, r6, r30);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              0 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r32, r43, r21 * r34);
    r6 = fma(r21, r43, r32 * r20);
    write_sum_2<double, double>((double*)inout_shared, r30, r6);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              2 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = fma(r21, r27, r32 * r41);
    r32 = fma(r10, r7, r42 * r12);
    write_sum_2<double, double>((double*)inout_shared, r21, r32);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              4 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r42, r43, r10 * r34);
    r21 = fma(r10, r43, r42 * r20);
    write_sum_2<double, double>((double*)inout_shared, r32, r21);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              6 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fma(r10, r27, r42 * r41);
    r42 = fma(r12, r43, r7 * r34);
    write_sum_2<double, double>((double*)inout_shared, r10, r42);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              8 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = fma(r12, r41, r7 * r27);
    r7 = fma(r7, r43, r12 * r20);
    write_sum_2<double, double>((double*)inout_shared, r7, r42);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              10 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = fma(r34, r43, r20 * r43);
    r34 = fma(r41, r43, r34 * r27);
    write_sum_2<double, double>((double*)inout_shared, r42, r34);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              12 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = fma(r27, r43, r20 * r41);
    write_sum_1<double, double>((double*)inout_shared, r43);
  };
  flush_sum_shared<1, double>(out_pose_precond_tril,
                              14 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r1 * r40;
    r27 = r5 * r40;
    write_idx_2<1024, double, double, double2>(out_calib_jac,
                                               0 * out_calib_jac_num_alloc,
                                               global_thread_idx,
                                               r43,
                                               r27);
    r41 = r22 * r37;
    r41 = r41 * r26;
    r20 = r0 * r5;
    r20 = r20 * r22;
    r20 = r20 * r37;
    write_idx_2<1024, double, double, double2>(out_calib_jac,
                                               2 * out_calib_jac_num_alloc,
                                               global_thread_idx,
                                               r41,
                                               r20);
    r34 = r4 * r2;
    r42 = r5 * r4;
    r42 = r42 * r3;
    r7 = r1 * r4;
    r7 = r7 * r2;
    r7 = fma(r40, r7, r40 * r42);
    write_sum_2<double, double>((double*)inout_shared, r7, r34);
  };
  flush_sum_shared<2, double>(out_calib_njtr,
                              0 * out_calib_njtr_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r4 * r3;
    r7 = r0 * r5;
    r7 = r7 * r22;
    r7 = r7 * r4;
    r7 = r7 * r3;
    r42 = r22 * r4;
    r42 = r42 * r2;
    r42 = r42 * r37;
    r42 = fma(r26, r42, r37 * r7);
    write_sum_2<double, double>((double*)inout_shared, r34, r42);
  };
  flush_sum_shared<2, double>(out_calib_njtr,
                              2 * out_calib_njtr_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r5 * r5;
    r42 = r42 * r29;
    r42 = r42 * r29;
    r34 = r1 * r1;
    r34 = r34 * r29;
    r34 = r34 * r29;
    r34 = fma(r31, r34, r31 * r42);
    write_sum_2<double, double>((double*)inout_shared, r34, r23);
  };
  flush_sum_shared<2, double>(out_calib_precond_diag,
                              0 * out_calib_precond_diag_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r0 * r31;
    r42 = r22 * r22;
    r34 = r34 * r42;
    r42 = r5 * r34;
    r7 = r1 * r26;
    r7 = fma(r34, r7, r25 * r42);
    write_sum_2<double, double>((double*)inout_shared, r23, r7);
  };
  flush_sum_shared<2, double>(out_calib_precond_diag,
                              2 * out_calib_precond_diag_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<double, double>((double*)inout_shared, r43, r27);
  };
  flush_sum_shared<2, double>(out_calib_precond_tril,
                              0 * out_calib_precond_tril_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = 0.00000000000000000e+00;
    r43 = r0 * r5;
    r43 = r43 * r5;
    r43 = r43 * r22;
    r43 = r43 * r29;
    r7 = r1 * r22;
    r7 = r7 * r29;
    r7 = r7 * r31;
    r7 = fma(r26, r7, r31 * r43);
    write_sum_2<double, double>((double*)inout_shared, r7, r27);
  };
  flush_sum_shared<2, double>(out_calib_precond_tril,
                              2 * out_calib_precond_tril_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<double, double>((double*)inout_shared, r41, r20);
  };
  flush_sum_shared<2, double>(out_calib_precond_tril,
                              4 * out_calib_precond_tril_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
}

void simple_radial_fixed_point_res_jac(
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
  simple_radial_fixed_point_res_jac_kernel<<<n_blocks, 1024>>>(
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