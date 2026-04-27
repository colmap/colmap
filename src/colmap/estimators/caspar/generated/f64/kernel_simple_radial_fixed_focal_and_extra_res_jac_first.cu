#include "kernel_simple_radial_fixed_focal_and_extra_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_focal_and_extra_res_jac_first_kernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
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
        double* out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        double* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        double* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        double* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
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
  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
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
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55;
  load_shared<2, double, double>(principal_point,
                                 0 * principal_point_num_alloc,
                                 principal_point_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          principal_point_indices_loc[threadIdx.x].target,
                          r0,
                          r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
    r30 = 1.0 / r39;
    r42 = r38 * r30;
    r2 = fma(r29, r42, r2);
    r3 = fma(r3, r4, r1);
    r1 = r0 * r38;
    r1 = r1 * r30;
    r3 = fma(r7, r1, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r42 = fma(r2, r2, r3 * r3);
  };
  sum_store<double>(out_rTr_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r42);
  if (global_thread_idx < problem_size) {
    r42 = r15 * r10;
    r42 = r42 * r13;
    r41 = r41 + r42;
    r43 = r4 * r26;
    r44 = r23 + r43;
    r11 = r11 * r11;
    r45 = r4 * r37;
    r46 = r11 + r45;
    r47 = r44 + r46;
    r47 = fma(r9, r47, r19 * r41);
    r41 = r6 * r6;
    r41 = r13 * r41;
    r31 = r39 * r31;
    r39 = 1.0 / r31;
    r41 = r41 * r39;
    r48 = r7 * r7;
    r48 = r13 * r48;
    r48 = r48 * r39;
    r49 = fma(r47, r48, r47 * r41);
    r50 = r17 * r7;
    r51 = r4 * r11;
    r52 = r37 + r51;
    r44 = r44 + r52;
    r44 = fma(r19, r44, r9 * r33);
    r50 = r50 * r44;
    r49 = fma(r32, r50, r49);
    r53 = r17 * r6;
    r15 = r14 * r15;
    r15 = r15 * r13;
    r40 = r40 + r15;
    r40 = fma(r9, r22, r19 * r40);
    r53 = r53 * r40;
    r49 = fma(r32, r53, r49);
    r53 = r5 * r49;
    r53 = r53 * r30;
    r40 = fma(r40, r1, r29 * r53);
    r53 = r47 * r29;
    r38 = r38 * r4;
    r38 = r38 * r32;
    r40 = fma(r38, r53, r40);
    r53 = r0 * r49;
    r50 = r5 * r7;
    r53 = r53 * r30;
    r44 = fma(r44, r1, r50 * r53);
    r53 = r0 * r7;
    r53 = r53 * r47;
    r44 = fma(r38, r53, r44);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r40, r44);
    r53 = r4 * r23;
    r54 = r26 + r53;
    r52 = r52 + r54;
    r52 = fma(r8, r52, r19 * r20);
    r55 = r17 * r7;
    r42 = r34 + r42;
    r42 = fma(r8, r42, r19 * r16);
    r55 = r55 * r42;
    r55 = fma(r32, r55, r52 * r48);
    r34 = r17 * r6;
    r10 = r14 * r10;
    r10 = r10 * r13;
    r36 = r36 + r10;
    r11 = r37 + r11;
    r11 = r11 + r43;
    r11 = r11 + r53;
    r11 = fma(r19, r11, r8 * r36);
    r34 = r34 * r11;
    r55 = fma(r32, r34, r55);
    r55 = fma(r52, r41, r55);
    r34 = r5 * r55;
    r34 = r34 * r30;
    r11 = fma(r11, r1, r29 * r34);
    r34 = r52 * r29;
    r11 = fma(r38, r34, r11);
    r34 = r0 * r7;
    r34 = r34 * r52;
    r34 = fma(r38, r34, r42 * r1);
    r42 = r0 * r55;
    r42 = r42 * r30;
    r34 = fma(r50, r42, r34);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r11, r34);
    r42 = r17 * r7;
    r15 = r12 + r15;
    r54 = r46 + r54;
    r54 = fma(r8, r54, r9 * r15);
    r42 = r42 * r54;
    r10 = r21 + r10;
    r10 = fma(r9, r10, r8 * r35);
    r42 = fma(r10, r41, r32 * r42);
    r21 = r17 * r6;
    r23 = r26 + r23;
    r23 = r23 + r45;
    r23 = r23 + r51;
    r23 = fma(r9, r23, r8 * r18);
    r21 = r21 * r23;
    r42 = fma(r32, r21, r42);
    r42 = fma(r10, r48, r42);
    r21 = r5 * r42;
    r21 = r21 * r30;
    r9 = r10 * r29;
    r9 = fma(r38, r9, r29 * r21);
    r9 = fma(r23, r1, r9);
    r23 = r0 * r7;
    r23 = r23 * r10;
    r21 = r0 * r42;
    r21 = r21 * r30;
    r21 = fma(r50, r21, r38 * r23);
    r21 = fma(r54, r1, r21);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r9, r21);
    r54 = r0 * r5;
    r54 = r54 * r17;
    r54 = r54 * r6;
    r54 = r54 * r7;
    r54 = r54 * r39;
    r23 = r5 * r17;
    r23 = r23 * r6;
    r23 = r23 * r39;
    r23 = fma(r29, r23, r1);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r23, r54);
    r8 = r0 * r17;
    r8 = r8 * r7;
    r8 = r8 * r39;
    r8 = fma(r50, r8, r1);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r54, r8);
    r51 = r41 + r48;
    r45 = r5 * r51;
    r45 = r45 * r30;
    r45 = fma(r29, r38, r29 * r45);
    r26 = r0 * r51;
    r26 = r26 * r30;
    r15 = r0 * r7;
    r15 = fma(r38, r15, r50 * r26);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r45, r15);
    r26 = r4 * r2;
    r46 = r4 * r3;
    r46 = fma(r44, r46, r40 * r26);
    r26 = r4 * r2;
    r12 = r4 * r3;
    r12 = fma(r34, r12, r11 * r26);
    write_sum_2<double, double>((double*)inout_shared, r46, r12);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = r4 * r2;
    r46 = r4 * r3;
    r46 = fma(r21, r46, r9 * r12);
    r12 = r4 * r2;
    r39 = r13 * r39;
    r13 = r3 * r39;
    r26 = r29 * r50;
    r13 = fma(r26, r13, r23 * r12);
    write_sum_2<double, double>((double*)inout_shared, r46, r13);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r4 * r3;
    r46 = r4 * r2;
    r46 = fma(r45, r46, r15 * r13);
    r13 = r4 * r3;
    r12 = r2 * r39;
    r12 = fma(r26, r12, r8 * r13);
    write_sum_2<double, double>((double*)inout_shared, r12, r46);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              4 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fma(r44, r44, r40 * r40);
    r12 = fma(r11, r11, r34 * r34);
    write_sum_2<double, double>((double*)inout_shared, r46, r12);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              0 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fma(r9, r9, r21 * r21);
    r46 = r0 * r5;
    r13 = 4.00000000000000000e+00;
    r31 = r31 * r31;
    r31 = 1.0 / r31;
    r46 = r46 * r6;
    r46 = r46 * r7;
    r46 = r46 * r13;
    r46 = r46 * r31;
    r46 = r46 * r26;
    r26 = fma(r23, r23, r46);
    write_sum_2<double, double>((double*)inout_shared, r12, r26);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              2 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fma(r8, r8, r46);
    r26 = fma(r45, r45, r15 * r15);
    write_sum_2<double, double>((double*)inout_shared, r46, r26);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              4 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fma(r40, r11, r44 * r34);
    r46 = fma(r40, r9, r44 * r21);
    write_sum_2<double, double>((double*)inout_shared, r26, r46);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              0 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fma(r44, r54, r40 * r23);
    r26 = fma(r40, r54, r44 * r8);
    write_sum_2<double, double>((double*)inout_shared, r46, r26);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              2 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fma(r40, r45, r44 * r15);
    r44 = fma(r34, r21, r11 * r9);
    write_sum_2<double, double>((double*)inout_shared, r40, r44);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              4 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = fma(r34, r54, r11 * r23);
    r40 = fma(r11, r54, r34 * r8);
    write_sum_2<double, double>((double*)inout_shared, r44, r40);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              6 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fma(r34, r15, r11 * r45);
    r11 = fma(r21, r54, r9 * r23);
    write_sum_2<double, double>((double*)inout_shared, r34, r11);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              8 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fma(r9, r45, r21 * r15);
    r9 = fma(r9, r54, r21 * r8);
    write_sum_2<double, double>((double*)inout_shared, r9, r11);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              10 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fma(r8, r54, r23 * r54);
    r23 = fma(r15, r54, r23 * r45);
    write_sum_2<double, double>((double*)inout_shared, r11, r23);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              12 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = fma(r45, r54, r8 * r15);
    write_sum_1<double, double>((double*)inout_shared, r54);
  };
  flush_sum_shared<1, double>(out_pose_precond_tril,
                              14 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = r4 * r2;
    r45 = r4 * r3;
    write_sum_2<double, double>((double*)inout_shared, r54, r45);
  };
  flush_sum_shared<2, double>(out_principal_point_njtr,
                              0 * out_principal_point_njtr_num_alloc,
                              principal_point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<double, double>((double*)inout_shared, r25, r25);
  };
  flush_sum_shared<2, double>(out_principal_point_precond_diag,
                              0 * out_principal_point_precond_diag_num_alloc,
                              principal_point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r20 * r38;
    r45 = fma(r29, r25, r28 * r1);
    r54 = r17 * r28;
    r54 = r54 * r6;
    r54 = fma(r32, r54, r20 * r41);
    r15 = r17 * r16;
    r15 = r15 * r7;
    r54 = fma(r32, r15, r54);
    r54 = fma(r20, r48, r54);
    r20 = r5 * r54;
    r20 = r20 * r30;
    r45 = fma(r29, r20, r45);
    r20 = r0 * r54;
    r20 = r20 * r30;
    r15 = r0 * r7;
    r15 = fma(r25, r15, r50 * r20);
    r15 = fma(r16, r1, r15);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               0 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r45,
                                               r15);
    r20 = r35 * r29;
    r20 = fma(r18, r1, r38 * r20);
    r25 = r17 * r18;
    r25 = r25 * r6;
    r25 = fma(r35, r41, r32 * r25);
    r8 = r17 * r24;
    r8 = r8 * r7;
    r25 = fma(r32, r8, r25);
    r25 = fma(r35, r48, r25);
    r8 = r5 * r25;
    r8 = r8 * r30;
    r20 = fma(r29, r8, r20);
    r8 = r0 * r25;
    r8 = r8 * r30;
    r8 = fma(r24, r1, r50 * r8);
    r23 = r0 * r35;
    r23 = r23 * r7;
    r8 = fma(r38, r23, r8);
    write_idx_2<1024, double, double, double2>(
        out_point_jac, 2 * out_point_jac_num_alloc, global_thread_idx, r20, r8);
    r23 = r27 * r29;
    r23 = fma(r22, r1, r38 * r23);
    r11 = r17 * r22;
    r11 = r11 * r6;
    r41 = fma(r27, r41, r32 * r11);
    r11 = r17 * r33;
    r11 = r11 * r7;
    r41 = fma(r32, r11, r41);
    r41 = fma(r27, r48, r41);
    r48 = r5 * r41;
    r48 = r48 * r30;
    r23 = fma(r29, r48, r23);
    r48 = r0 * r41;
    r48 = r48 * r30;
    r1 = fma(r33, r1, r50 * r48);
    r48 = r0 * r27;
    r48 = r48 * r7;
    r1 = fma(r38, r48, r1);
    write_idx_2<1024, double, double, double2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r23, r1);
    r48 = r4 * r3;
    r38 = r4 * r2;
    r38 = fma(r45, r38, r15 * r48);
    r48 = r4 * r3;
    r50 = r4 * r2;
    r50 = fma(r20, r50, r8 * r48);
    write_sum_2<double, double>((double*)inout_shared, r38, r50);
  };
  flush_sum_shared<2, double>(out_point_njtr,
                              0 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r4 * r2;
    r38 = r4 * r3;
    r38 = fma(r1, r38, r23 * r50);
    write_sum_1<double, double>((double*)inout_shared, r38);
  };
  flush_sum_shared<1, double>(out_point_njtr,
                              2 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = fma(r15, r15, r45 * r45);
    r50 = fma(r20, r20, r8 * r8);
    write_sum_2<double, double>((double*)inout_shared, r38, r50);
  };
  flush_sum_shared<2, double>(out_point_precond_diag,
                              0 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = fma(r1, r1, r23 * r23);
    write_sum_1<double, double>((double*)inout_shared, r50);
  };
  flush_sum_shared<1, double>(out_point_precond_diag,
                              2 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = fma(r15, r8, r45 * r20);
    r45 = fma(r45, r23, r15 * r1);
    write_sum_2<double, double>((double*)inout_shared, r50, r45);
  };
  flush_sum_shared<2, double>(out_point_precond_tril,
                              0 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r20, r23, r8 * r1);
    write_sum_1<double, double>((double*)inout_shared, r23);
  };
  flush_sum_shared<1, double>(out_point_precond_tril,
                              2 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  sum_flush_final<double>(out_rTr_local, out_rTr, 1);
}

void simple_radial_fixed_focal_and_extra_res_jac_first(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
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
    double* out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    double* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    double* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    double* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
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
  simple_radial_fixed_focal_and_extra_res_jac_first_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
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
      out_principal_point_jac,
      out_principal_point_jac_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc,
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