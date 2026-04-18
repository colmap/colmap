#include "kernel_simple_radial_fixed_focal_fixed_extra_calib_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_focal_fixed_extra_calib_res_jac_kernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* focal,
        unsigned int focal_num_alloc,
        double* extra_calib,
        unsigned int extra_calib_num_alloc,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        extra_calib, 0 * extra_calib_num_alloc, global_thread_idx, r0, r1);
    read_idx_2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
    read_idx_1<1024, double, double, double>(
        focal, 0 * focal_num_alloc, global_thread_idx, r0);
  };
  load_shared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r5, r6);
  };
  __syncthreads();
  load_shared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r7, r8);
  };
  __syncthreads();
  load_shared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r9, r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r11 = r9 * r10;
    r12 = -2.00000000000000000e+00;
    r11 = r11 * r12;
  };
  load_shared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r13, r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = r13 * r14;
    r16 = 2.00000000000000000e+00;
    r15 = r15 * r16;
    r17 = r11 + r15;
    r5 = fma(r8, r17, r5);
  };
  load_shared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r19 = r13 * r9;
    r19 = r19 * r16;
    r20 = r14 * r10;
    r20 = r20 * r16;
    r21 = r19 + r20;
    r22 = r9 * r9;
    r23 = r12 * r22;
    r24 = 1.00000000000000000e+00;
    r25 = r14 * r14;
    r26 = fma(r12, r25, r24);
    r27 = r23 + r26;
    r5 = fma(r18, r21, r5);
    r5 = fma(r7, r27, r5);
    r28 = r0 * r5;
    read_idx_1<1024, double, double, double>(
        extra_calib, 2 * extra_calib_num_alloc, global_thread_idx, r29);
    r30 = r5 * r5;
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
    r33 = r14 * r9;
    r33 = r33 * r16;
    r34 = r13 * r10;
    r34 = r34 * r16;
    r35 = r33 + r34;
    r32 = fma(r8, r35, r32);
    r36 = r14 * r10;
    r36 = r36 * r12;
    r19 = r19 + r36;
    r37 = r13 * r13;
    r38 = r12 * r37;
    r26 = r38 + r26;
    r32 = fma(r7, r19, r32);
    r32 = fma(r18, r26, r32);
    r39 = copysign(1.0, r32);
    r39 = fma(r31, r39, r32);
    r31 = r39 * r39;
    r32 = 1.0 / r31;
    r40 = r9 * r10;
    r40 = r40 * r16;
    r15 = r15 + r40;
    r6 = fma(r7, r15, r6);
    r41 = r13 * r10;
    r41 = r41 * r12;
    r33 = r33 + r41;
    r23 = r24 + r23;
    r23 = r23 + r38;
    r6 = fma(r18, r33, r6);
    r6 = fma(r8, r23, r6);
    r38 = r6 * r6;
    r38 = fma(r32, r38, r32 * r30);
    r38 = fma(r29, r38, r24);
    r24 = 1.0 / r39;
    r30 = r38 * r24;
    r2 = fma(r28, r30, r2);
    r3 = fma(r3, r4, r1);
    r1 = r0 * r38;
    r1 = r1 * r24;
    r3 = fma(r6, r1, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r30 = r14 * r9;
    r30 = r30 * r12;
    r41 = r41 + r30;
    r42 = r4 * r25;
    r43 = r22 + r42;
    r10 = r10 * r10;
    r44 = r4 * r37;
    r45 = r10 + r44;
    r46 = r43 + r45;
    r46 = fma(r8, r46, r18 * r41);
    r41 = r46 * r28;
    r38 = r38 * r4;
    r38 = r38 * r32;
    r14 = r13 * r14;
    r14 = r14 * r12;
    r40 = r40 + r14;
    r40 = fma(r8, r21, r18 * r40);
    r41 = fma(r40, r1, r38 * r41);
    r47 = r5 * r5;
    r47 = r12 * r47;
    r31 = r39 * r31;
    r39 = 1.0 / r31;
    r47 = r47 * r39;
    r48 = r6 * r6;
    r48 = r12 * r48;
    r48 = r48 * r39;
    r49 = fma(r46, r48, r46 * r47);
    r50 = r16 * r6;
    r51 = r4 * r10;
    r52 = r37 + r51;
    r43 = r43 + r52;
    r43 = fma(r18, r43, r8 * r33);
    r50 = r50 * r43;
    r49 = fma(r32, r50, r49);
    r53 = r16 * r5;
    r53 = r53 * r40;
    r49 = fma(r32, r53, r49);
    r53 = r29 * r49;
    r53 = r53 * r24;
    r41 = fma(r28, r53, r41);
    r53 = r0 * r49;
    r50 = r29 * r6;
    r53 = r53 * r24;
    r43 = fma(r43, r1, r50 * r53);
    r53 = r0 * r6;
    r53 = r53 * r46;
    r43 = fma(r38, r53, r43);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r41, r43);
    r53 = r4 * r22;
    r40 = r25 + r53;
    r52 = r52 + r40;
    r52 = fma(r7, r52, r18 * r19);
    r54 = r16 * r6;
    r30 = r34 + r30;
    r30 = fma(r7, r30, r18 * r15);
    r54 = r54 * r30;
    r54 = fma(r32, r54, r52 * r48);
    r34 = r16 * r5;
    r9 = r13 * r9;
    r9 = r9 * r12;
    r36 = r36 + r9;
    r10 = r37 + r10;
    r10 = r10 + r42;
    r10 = r10 + r53;
    r10 = fma(r18, r10, r7 * r36);
    r34 = r34 * r10;
    r54 = fma(r32, r34, r54);
    r54 = fma(r52, r47, r54);
    r34 = r29 * r54;
    r34 = r34 * r24;
    r18 = r52 * r28;
    r18 = fma(r38, r18, r28 * r34);
    r18 = fma(r10, r1, r18);
    r10 = r0 * r6;
    r10 = r10 * r52;
    r30 = fma(r30, r1, r38 * r10);
    r10 = r0 * r54;
    r10 = r10 * r24;
    r30 = fma(r50, r10, r30);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r18, r30);
    r9 = r20 + r9;
    r9 = fma(r8, r9, r7 * r35);
    r20 = r9 * r28;
    r22 = r25 + r22;
    r22 = r22 + r44;
    r22 = r22 + r51;
    r22 = fma(r8, r22, r7 * r17);
    r20 = fma(r22, r1, r38 * r20);
    r51 = r16 * r6;
    r14 = r11 + r14;
    r40 = r45 + r40;
    r40 = fma(r7, r40, r8 * r14);
    r51 = r51 * r40;
    r51 = fma(r9, r47, r32 * r51);
    r7 = r16 * r5;
    r7 = r7 * r22;
    r51 = fma(r32, r7, r51);
    r51 = fma(r9, r48, r51);
    r7 = r29 * r51;
    r7 = r7 * r24;
    r20 = fma(r28, r7, r20);
    r7 = r0 * r6;
    r7 = r7 * r9;
    r7 = fma(r38, r7, r40 * r1);
    r40 = r0 * r51;
    r40 = r40 * r24;
    r7 = fma(r50, r40, r7);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r20, r7);
    r40 = r0 * r29;
    r40 = r40 * r16;
    r40 = r40 * r5;
    r40 = r40 * r6;
    r40 = r40 * r39;
    r22 = r29 * r16;
    r22 = r22 * r5;
    r22 = r22 * r39;
    r22 = fma(r28, r22, r1);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r22, r40);
    r14 = r0 * r16;
    r14 = r14 * r6;
    r14 = r14 * r39;
    r14 = fma(r50, r14, r1);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r40, r14);
    r8 = r47 + r48;
    r45 = r29 * r8;
    r45 = r45 * r24;
    r45 = fma(r28, r45, r28 * r38);
    r11 = r0 * r6;
    r44 = r0 * r8;
    r44 = r44 * r24;
    r44 = fma(r50, r44, r38 * r11);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r45, r44);
    r11 = r4 * r2;
    r25 = r4 * r3;
    r25 = fma(r43, r25, r41 * r11);
    r11 = r4 * r2;
    r10 = r4 * r3;
    r10 = fma(r30, r10, r18 * r11);
    write_sum_2<double, double>((double*)inout_shared, r25, r10);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = r4 * r3;
    r25 = r4 * r2;
    r25 = fma(r20, r25, r7 * r10);
    r10 = r4 * r2;
    r39 = r12 * r39;
    r12 = r3 * r39;
    r11 = r28 * r50;
    r12 = fma(r11, r12, r22 * r10);
    write_sum_2<double, double>((double*)inout_shared, r25, r12);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = r4 * r3;
    r25 = r4 * r2;
    r25 = fma(r45, r25, r44 * r12);
    r12 = r4 * r3;
    r10 = r2 * r39;
    r10 = fma(r11, r10, r14 * r12);
    write_sum_2<double, double>((double*)inout_shared, r10, r25);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              4 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = fma(r41, r41, r43 * r43);
    r10 = fma(r18, r18, r30 * r30);
    write_sum_2<double, double>((double*)inout_shared, r25, r10);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              0 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fma(r7, r7, r20 * r20);
    r25 = r0 * r29;
    r12 = 4.00000000000000000e+00;
    r31 = r31 * r31;
    r31 = 1.0 / r31;
    r25 = r25 * r5;
    r25 = r25 * r6;
    r25 = r25 * r12;
    r25 = r25 * r31;
    r25 = r25 * r11;
    r11 = fma(r22, r22, r25);
    write_sum_2<double, double>((double*)inout_shared, r10, r11);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              2 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = fma(r14, r14, r25);
    r11 = fma(r44, r44, r45 * r45);
    write_sum_2<double, double>((double*)inout_shared, r25, r11);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              4 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fma(r41, r18, r43 * r30);
    r25 = fma(r43, r7, r41 * r20);
    write_sum_2<double, double>((double*)inout_shared, r11, r25);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              0 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = fma(r43, r40, r41 * r22);
    r11 = fma(r41, r40, r43 * r14);
    write_sum_2<double, double>((double*)inout_shared, r25, r11);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              2 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r41, r45, r43 * r44);
    r43 = fma(r30, r7, r18 * r20);
    write_sum_2<double, double>((double*)inout_shared, r41, r43);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              4 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = fma(r30, r40, r18 * r22);
    r41 = fma(r18, r40, r30 * r14);
    write_sum_2<double, double>((double*)inout_shared, r43, r41);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              6 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r30, r44, r18 * r45);
    r18 = fma(r7, r40, r20 * r22);
    write_sum_2<double, double>((double*)inout_shared, r30, r18);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              8 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = fma(r7, r44, r20 * r45);
    r20 = fma(r20, r40, r7 * r14);
    write_sum_2<double, double>((double*)inout_shared, r20, r18);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              10 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = fma(r14, r40, r22 * r40);
    r22 = fma(r44, r40, r22 * r45);
    write_sum_2<double, double>((double*)inout_shared, r18, r22);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              12 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fma(r45, r40, r14 * r44);
    write_sum_1<double, double>((double*)inout_shared, r40);
  };
  flush_sum_shared<1, double>(out_pose_precond_tril,
                              14 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r19 * r28;
    r40 = fma(r38, r40, r27 * r1);
    r45 = r16 * r27;
    r45 = r45 * r5;
    r45 = fma(r32, r45, r19 * r47);
    r44 = r16 * r15;
    r44 = r44 * r6;
    r45 = fma(r32, r44, r45);
    r45 = fma(r19, r48, r45);
    r44 = r29 * r45;
    r44 = r44 * r24;
    r40 = fma(r28, r44, r40);
    r44 = r0 * r19;
    r44 = r44 * r6;
    r14 = r0 * r45;
    r14 = r14 * r24;
    r14 = fma(r50, r14, r38 * r44);
    r14 = fma(r15, r1, r14);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               0 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r40,
                                               r14);
    r44 = r16 * r17;
    r44 = r44 * r5;
    r44 = fma(r35, r47, r32 * r44);
    r22 = r16 * r23;
    r22 = r22 * r6;
    r44 = fma(r32, r22, r44);
    r44 = fma(r35, r48, r44);
    r22 = r29 * r44;
    r22 = r22 * r24;
    r22 = fma(r17, r1, r28 * r22);
    r18 = r35 * r28;
    r22 = fma(r38, r18, r22);
    r18 = r0 * r44;
    r18 = r18 * r24;
    r18 = fma(r23, r1, r50 * r18);
    r20 = r0 * r35;
    r20 = r20 * r6;
    r18 = fma(r38, r20, r18);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               2 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r22,
                                               r18);
    r20 = r16 * r21;
    r20 = r20 * r5;
    r47 = fma(r26, r47, r32 * r20);
    r20 = r16 * r33;
    r20 = r20 * r6;
    r47 = fma(r32, r20, r47);
    r47 = fma(r26, r48, r47);
    r48 = r29 * r47;
    r48 = r48 * r24;
    r48 = fma(r28, r48, r21 * r1);
    r38 = r26 * r38;
    r48 = fma(r28, r38, r48);
    r26 = r0 * r6;
    r26 = fma(r38, r26, r33 * r1);
    r1 = r0 * r47;
    r1 = r1 * r24;
    r26 = fma(r50, r1, r26);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               4 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r48,
                                               r26);
    r1 = r4 * r3;
    r50 = r4 * r2;
    r50 = fma(r40, r50, r14 * r1);
    r1 = r4 * r3;
    r24 = r4 * r2;
    r24 = fma(r22, r24, r18 * r1);
    write_sum_2<double, double>((double*)inout_shared, r50, r24);
  };
  flush_sum_shared<2, double>(out_point_njtr,
                              0 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = r4 * r3;
    r50 = r4 * r2;
    r50 = fma(r48, r50, r26 * r24);
    write_sum_1<double, double>((double*)inout_shared, r50);
  };
  flush_sum_shared<1, double>(out_point_njtr,
                              2 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = fma(r40, r40, r14 * r14);
    r24 = fma(r22, r22, r18 * r18);
    write_sum_2<double, double>((double*)inout_shared, r50, r24);
  };
  flush_sum_shared<2, double>(out_point_precond_diag,
                              0 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r48, r48, r26 * r26);
    write_sum_1<double, double>((double*)inout_shared, r24);
  };
  flush_sum_shared<1, double>(out_point_precond_diag,
                              2 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r14, r18, r40 * r22);
    r14 = fma(r14, r26, r40 * r48);
    write_sum_2<double, double>((double*)inout_shared, r24, r14);
  };
  flush_sum_shared<2, double>(out_point_precond_tril,
                              0 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = fma(r22, r48, r18 * r26);
    write_sum_1<double, double>((double*)inout_shared, r48);
  };
  flush_sum_shared<1, double>(out_point_precond_tril,
                              2 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
}

void simple_radial_fixed_focal_fixed_extra_calib_res_jac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* focal,
    unsigned int focal_num_alloc,
    double* extra_calib,
    unsigned int extra_calib_num_alloc,
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
  simple_radial_fixed_focal_fixed_extra_calib_res_jac_kernel<<<n_blocks,
                                                               1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      focal,
      focal_num_alloc,
      extra_calib,
      extra_calib_num_alloc,
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