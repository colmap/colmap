#include "kernel_simple_radial_fixed_focal_and_extra_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_focal_and_extra_fixed_principal_point_res_jac_kernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
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
    read_idx_2<1024, double, double, double2>(principal_point,
                                              0 * principal_point_num_alloc,
                                              global_thread_idx,
                                              r0,
                                              r1);
    read_idx_2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
    read_idx_2<1024, double, double, double2>(focal_and_extra,
                                              0 * focal_and_extra_num_alloc,
                                              global_thread_idx,
                                              r0,
                                              r5);
  };
  load_shared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  load_shared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r8, r9);
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
    r12 = r10 * r11;
    r13 = -2.00000000000000000e+00;
    r12 = r12 * r13;
  };
  load_shared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r14, r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = r14 * r15;
    r17 = 2.00000000000000000e+00;
    r16 = r16 * r17;
    r18 = r12 + r16;
    r6 = fma(r9, r18, r6);
  };
  load_shared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r19);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r20 = r14 * r10;
    r20 = r20 * r17;
    r21 = r15 * r11;
    r21 = r21 * r17;
    r22 = r20 + r21;
    r23 = r10 * r10;
    r24 = r13 * r23;
    r25 = 1.00000000000000000e+00;
    r26 = r15 * r15;
    r27 = fma(r13, r26, r25);
    r28 = r24 + r27;
    r6 = fma(r19, r22, r6);
    r6 = fma(r8, r28, r6);
    r29 = r0 * r6;
    r30 = r6 * r6;
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
    r33 = r15 * r10;
    r33 = r33 * r17;
    r34 = r14 * r11;
    r34 = r34 * r17;
    r35 = r33 + r34;
    r32 = fma(r9, r35, r32);
    r36 = r15 * r11;
    r36 = r36 * r13;
    r20 = r20 + r36;
    r37 = r14 * r14;
    r38 = r13 * r37;
    r27 = r38 + r27;
    r32 = fma(r8, r20, r32);
    r32 = fma(r19, r27, r32);
    r39 = copysign(1.0, r32);
    r39 = fma(r31, r39, r32);
    r31 = r39 * r39;
    r32 = 1.0 / r31;
    r40 = r10 * r11;
    r40 = r40 * r17;
    r16 = r16 + r40;
    r7 = fma(r8, r16, r7);
    r41 = r14 * r11;
    r41 = r41 * r13;
    r33 = r33 + r41;
    r24 = r25 + r24;
    r24 = r24 + r38;
    r7 = fma(r19, r33, r7);
    r7 = fma(r9, r24, r7);
    r38 = r7 * r7;
    r38 = fma(r32, r38, r32 * r30);
    r38 = fma(r5, r38, r25);
    r25 = 1.0 / r39;
    r30 = r38 * r25;
    r2 = fma(r29, r30, r2);
    r3 = fma(r3, r4, r1);
    r1 = r0 * r38;
    r1 = r1 * r25;
    r3 = fma(r7, r1, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r30 = r15 * r10;
    r30 = r30 * r13;
    r41 = r41 + r30;
    r42 = r4 * r26;
    r43 = r23 + r42;
    r11 = r11 * r11;
    r44 = r4 * r37;
    r45 = r11 + r44;
    r46 = r43 + r45;
    r46 = fma(r9, r46, r19 * r41);
    r41 = r6 * r6;
    r41 = r13 * r41;
    r31 = r39 * r31;
    r39 = 1.0 / r31;
    r41 = r41 * r39;
    r47 = r7 * r7;
    r47 = r13 * r47;
    r47 = r47 * r39;
    r48 = fma(r46, r47, r46 * r41);
    r49 = r17 * r7;
    r50 = r4 * r11;
    r51 = r37 + r50;
    r43 = r43 + r51;
    r43 = fma(r19, r43, r9 * r33);
    r49 = r49 * r43;
    r48 = fma(r32, r49, r48);
    r52 = r17 * r6;
    r15 = r14 * r15;
    r15 = r15 * r13;
    r40 = r40 + r15;
    r40 = fma(r9, r22, r19 * r40);
    r52 = r52 * r40;
    r48 = fma(r32, r52, r48);
    r52 = r5 * r48;
    r52 = r52 * r25;
    r40 = fma(r40, r1, r29 * r52);
    r52 = r46 * r29;
    r38 = r38 * r4;
    r38 = r38 * r32;
    r40 = fma(r38, r52, r40);
    r52 = r0 * r48;
    r49 = r5 * r7;
    r52 = r52 * r25;
    r43 = fma(r43, r1, r49 * r52);
    r52 = r0 * r7;
    r52 = r52 * r46;
    r43 = fma(r38, r52, r43);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r40, r43);
    r52 = r4 * r23;
    r53 = r26 + r52;
    r51 = r51 + r53;
    r51 = fma(r8, r51, r19 * r20);
    r54 = r17 * r7;
    r30 = r34 + r30;
    r30 = fma(r8, r30, r19 * r16);
    r54 = r54 * r30;
    r54 = fma(r32, r54, r51 * r47);
    r34 = r17 * r6;
    r10 = r14 * r10;
    r10 = r10 * r13;
    r36 = r36 + r10;
    r11 = r37 + r11;
    r11 = r11 + r42;
    r11 = r11 + r52;
    r11 = fma(r19, r11, r8 * r36);
    r34 = r34 * r11;
    r54 = fma(r32, r34, r54);
    r54 = fma(r51, r41, r54);
    r34 = r5 * r54;
    r34 = r34 * r25;
    r11 = fma(r11, r1, r29 * r34);
    r34 = r51 * r29;
    r11 = fma(r38, r34, r11);
    r34 = r0 * r7;
    r34 = r34 * r51;
    r34 = fma(r38, r34, r30 * r1);
    r30 = r0 * r54;
    r30 = r30 * r25;
    r34 = fma(r49, r30, r34);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r11, r34);
    r30 = r17 * r7;
    r15 = r12 + r15;
    r53 = r45 + r53;
    r53 = fma(r8, r53, r9 * r15);
    r30 = r30 * r53;
    r10 = r21 + r10;
    r10 = fma(r9, r10, r8 * r35);
    r30 = fma(r10, r41, r32 * r30);
    r21 = r17 * r6;
    r23 = r26 + r23;
    r23 = r23 + r44;
    r23 = r23 + r50;
    r23 = fma(r9, r23, r8 * r18);
    r21 = r21 * r23;
    r30 = fma(r32, r21, r30);
    r30 = fma(r10, r47, r30);
    r21 = r5 * r30;
    r21 = r21 * r25;
    r10 = r10 * r38;
    r21 = fma(r29, r10, r29 * r21);
    r21 = fma(r23, r1, r21);
    r23 = r0 * r7;
    r9 = r0 * r30;
    r9 = r9 * r25;
    r9 = fma(r49, r9, r10 * r23);
    r9 = fma(r53, r1, r9);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r21, r9);
    r53 = r0 * r5;
    r53 = r53 * r17;
    r53 = r53 * r6;
    r53 = r53 * r7;
    r53 = r53 * r39;
    r23 = r5 * r17;
    r23 = r23 * r6;
    r23 = r23 * r39;
    r23 = fma(r29, r23, r1);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r23, r53);
    r10 = r0 * r17;
    r10 = r10 * r7;
    r10 = r10 * r39;
    r10 = fma(r49, r10, r1);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r53, r10);
    r8 = r41 + r47;
    r50 = r5 * r8;
    r50 = r50 * r25;
    r50 = fma(r29, r38, r29 * r50);
    r44 = r0 * r8;
    r44 = r44 * r25;
    r26 = r0 * r7;
    r26 = fma(r38, r26, r49 * r44);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r50, r26);
    r44 = r4 * r2;
    r15 = r4 * r3;
    r15 = fma(r43, r15, r40 * r44);
    r44 = r4 * r2;
    r45 = r4 * r3;
    r45 = fma(r34, r45, r11 * r44);
    write_sum_2<double, double>((double*)inout_shared, r15, r45);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = r4 * r2;
    r15 = r4 * r3;
    r15 = fma(r9, r15, r21 * r45);
    r45 = r4 * r2;
    r39 = r13 * r39;
    r13 = r3 * r39;
    r44 = r29 * r49;
    r13 = fma(r44, r13, r23 * r45);
    write_sum_2<double, double>((double*)inout_shared, r15, r13);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r4 * r3;
    r15 = r4 * r2;
    r15 = fma(r50, r15, r26 * r13);
    r13 = r4 * r3;
    r45 = r2 * r39;
    r45 = fma(r44, r45, r10 * r13);
    write_sum_2<double, double>((double*)inout_shared, r45, r15);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              4 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fma(r43, r43, r40 * r40);
    r45 = fma(r11, r11, r34 * r34);
    write_sum_2<double, double>((double*)inout_shared, r15, r45);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              0 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fma(r21, r21, r9 * r9);
    r15 = r0 * r5;
    r13 = 4.00000000000000000e+00;
    r31 = r31 * r31;
    r31 = 1.0 / r31;
    r15 = r15 * r6;
    r15 = r15 * r7;
    r15 = r15 * r13;
    r15 = r15 * r31;
    r15 = r15 * r44;
    r44 = fma(r23, r23, r15);
    write_sum_2<double, double>((double*)inout_shared, r45, r44);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              2 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fma(r10, r10, r15);
    r44 = fma(r50, r50, r26 * r26);
    write_sum_2<double, double>((double*)inout_shared, r15, r44);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              4 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = fma(r40, r11, r43 * r34);
    r15 = fma(r40, r21, r43 * r9);
    write_sum_2<double, double>((double*)inout_shared, r44, r15);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              0 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fma(r43, r53, r40 * r23);
    r44 = fma(r40, r53, r43 * r10);
    write_sum_2<double, double>((double*)inout_shared, r15, r44);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              2 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fma(r40, r50, r43 * r26);
    r43 = fma(r34, r9, r11 * r21);
    write_sum_2<double, double>((double*)inout_shared, r40, r43);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              4 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = fma(r34, r53, r11 * r23);
    r40 = fma(r11, r53, r34 * r10);
    write_sum_2<double, double>((double*)inout_shared, r43, r40);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              6 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fma(r34, r26, r11 * r50);
    r11 = fma(r9, r53, r21 * r23);
    write_sum_2<double, double>((double*)inout_shared, r34, r11);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              8 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fma(r21, r50, r9 * r26);
    r21 = fma(r21, r53, r9 * r10);
    write_sum_2<double, double>((double*)inout_shared, r21, r11);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              10 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fma(r10, r53, r23 * r53);
    r23 = fma(r26, r53, r23 * r50);
    write_sum_2<double, double>((double*)inout_shared, r11, r23);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              12 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = fma(r50, r53, r10 * r26);
    write_sum_1<double, double>((double*)inout_shared, r53);
  };
  flush_sum_shared<1, double>(out_pose_precond_tril,
                              14 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = r20 * r29;
    r53 = fma(r38, r53, r28 * r1);
    r50 = r17 * r28;
    r50 = r50 * r6;
    r50 = fma(r32, r50, r20 * r41);
    r26 = r17 * r16;
    r26 = r26 * r7;
    r50 = fma(r32, r26, r50);
    r50 = fma(r20, r47, r50);
    r26 = r5 * r50;
    r26 = r26 * r25;
    r53 = fma(r29, r26, r53);
    r26 = r0 * r50;
    r26 = r26 * r25;
    r10 = r0 * r20;
    r10 = r10 * r7;
    r10 = fma(r38, r10, r49 * r26);
    r10 = fma(r16, r1, r10);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               0 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r53,
                                               r10);
    r26 = r35 * r29;
    r26 = fma(r18, r1, r38 * r26);
    r23 = r17 * r18;
    r23 = r23 * r6;
    r23 = fma(r35, r41, r32 * r23);
    r11 = r17 * r24;
    r11 = r11 * r7;
    r23 = fma(r32, r11, r23);
    r23 = fma(r35, r47, r23);
    r11 = r5 * r23;
    r11 = r11 * r25;
    r26 = fma(r29, r11, r26);
    r11 = r0 * r23;
    r11 = r11 * r25;
    r11 = fma(r24, r1, r49 * r11);
    r21 = r0 * r35;
    r21 = r21 * r7;
    r11 = fma(r38, r21, r11);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               2 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r26,
                                               r11);
    r21 = r27 * r29;
    r21 = fma(r22, r1, r38 * r21);
    r9 = r17 * r22;
    r9 = r9 * r6;
    r41 = fma(r27, r41, r32 * r9);
    r9 = r17 * r33;
    r9 = r9 * r7;
    r41 = fma(r32, r9, r41);
    r41 = fma(r27, r47, r41);
    r47 = r5 * r41;
    r47 = r47 * r25;
    r21 = fma(r29, r47, r21);
    r47 = r0 * r41;
    r47 = r47 * r25;
    r1 = fma(r33, r1, r49 * r47);
    r47 = r0 * r27;
    r47 = r47 * r7;
    r1 = fma(r38, r47, r1);
    write_idx_2<1024, double, double, double2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r21, r1);
    r47 = r4 * r3;
    r38 = r4 * r2;
    r38 = fma(r53, r38, r10 * r47);
    r47 = r4 * r3;
    r49 = r4 * r2;
    r49 = fma(r26, r49, r11 * r47);
    write_sum_2<double, double>((double*)inout_shared, r38, r49);
  };
  flush_sum_shared<2, double>(out_point_njtr,
                              0 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r4 * r2;
    r38 = r4 * r3;
    r38 = fma(r1, r38, r21 * r49);
    write_sum_1<double, double>((double*)inout_shared, r38);
  };
  flush_sum_shared<1, double>(out_point_njtr,
                              2 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = fma(r10, r10, r53 * r53);
    r49 = fma(r26, r26, r11 * r11);
    write_sum_2<double, double>((double*)inout_shared, r38, r49);
  };
  flush_sum_shared<2, double>(out_point_precond_diag,
                              0 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = fma(r1, r1, r21 * r21);
    write_sum_1<double, double>((double*)inout_shared, r49);
  };
  flush_sum_shared<1, double>(out_point_precond_diag,
                              2 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = fma(r10, r11, r53 * r26);
    r53 = fma(r53, r21, r10 * r1);
    write_sum_2<double, double>((double*)inout_shared, r49, r53);
  };
  flush_sum_shared<2, double>(out_point_precond_tril,
                              0 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = fma(r26, r21, r11 * r1);
    write_sum_1<double, double>((double*)inout_shared, r21);
  };
  flush_sum_shared<1, double>(out_point_precond_tril,
                              2 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
}

void simple_radial_fixed_focal_and_extra_fixed_principal_point_res_jac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
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
  simple_radial_fixed_focal_and_extra_fixed_principal_point_res_jac_kernel<<<
      n_blocks,
      1024>>>(pose,
              pose_num_alloc,
              pose_indices,
              point,
              point_num_alloc,
              point_indices,
              pixel,
              pixel_num_alloc,
              focal_and_extra,
              focal_and_extra_num_alloc,
              principal_point,
              principal_point_num_alloc,
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