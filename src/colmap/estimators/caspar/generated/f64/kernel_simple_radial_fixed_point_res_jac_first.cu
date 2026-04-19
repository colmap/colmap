#include "kernel_simple_radial_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_point_res_jac_first_kernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        SharedIndex* focal_and_extra_indices,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
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
        double* out_focal_and_extra_jac,
        unsigned int out_focal_and_extra_jac_num_alloc,
        double* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        double* const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        double* const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
        double* out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        double* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        double* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        double* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex focal_and_extra_indices_loc[1024];
  focal_and_extra_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52;
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
  };
  load_shared<2, double, double>(focal_and_extra,
                                 0 * focal_and_extra_num_alloc,
                                 focal_and_extra_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          focal_and_extra_indices_loc[threadIdx.x].target,
                          r0,
                          r5);
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
    r41 = fma(r2, r2, r3 * r3);
  };
  sum_store<double>(out_rTr_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r41);
  if (global_thread_idx < problem_size) {
    r41 = r16 * r6;
    r42 = r11 * r17;
    r39 = r39 + r42;
    r43 = r14 * r14;
    r44 = r4 * r26;
    r45 = r43 + r44;
    r15 = r15 * r15;
    r46 = r4 * r36;
    r47 = r15 + r46;
    r48 = r45 + r47;
    r48 = fma(r9, r48, r20 * r39);
    r30 = r27 * r30;
    r27 = 1.0 / r30;
    r41 = r41 * r6;
    r41 = r41 * r48;
    r39 = r16 * r7;
    r39 = r39 * r7;
    r39 = r39 * r48;
    r39 = fma(r27, r39, r27 * r41);
    r41 = r13 * r7;
    r49 = r4 * r15;
    r50 = r36 + r49;
    r45 = r45 + r50;
    r45 = fma(r20, r45, r9 * r32);
    r41 = r41 * r45;
    r39 = fma(r31, r41, r39);
    r32 = r13 * r6;
    r11 = r10 * r11;
    r11 = r11 * r16;
    r38 = r38 + r11;
    r23 = fma(r9, r23, r20 * r38);
    r32 = r32 * r23;
    r39 = fma(r31, r32, r39);
    r32 = r5 * r39;
    r32 = r32 * r37;
    r23 = fma(r23, r1, r28 * r32);
    r32 = r48 * r28;
    r41 = r29 * r4;
    r41 = r41 * r31;
    r23 = fma(r41, r32, r23);
    r32 = r0 * r39;
    r38 = r5 * r7;
    r32 = r32 * r37;
    r45 = fma(r45, r1, r38 * r32);
    r32 = r0 * r31;
    r32 = r32 * r7;
    r32 = r32 * r29;
    r32 = r32 * r4;
    r45 = fma(r48, r32, r45);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r23, r45);
    r51 = r16 * r7;
    r14 = r14 * r14;
    r14 = r14 * r4;
    r52 = r26 + r14;
    r50 = r50 + r52;
    r50 = fma(r8, r50, r20 * r21);
    r51 = r51 * r7;
    r51 = r51 * r50;
    r21 = r13 * r7;
    r42 = r33 + r42;
    r42 = fma(r8, r42, r20 * r12);
    r21 = r21 * r42;
    r21 = fma(r31, r21, r27 * r51);
    r51 = r16 * r6;
    r51 = r51 * r6;
    r51 = r51 * r50;
    r21 = fma(r27, r51, r21);
    r12 = r13 * r6;
    r17 = r10 * r17;
    r35 = r35 + r17;
    r15 = r36 + r15;
    r15 = r15 + r44;
    r15 = r15 + r14;
    r15 = fma(r20, r15, r8 * r35);
    r12 = r12 * r15;
    r21 = fma(r31, r12, r21);
    r12 = r5 * r21;
    r12 = r12 * r37;
    r15 = fma(r15, r1, r28 * r12);
    r12 = r50 * r28;
    r15 = fma(r41, r12, r15);
    r42 = fma(r50, r32, r42 * r1);
    r12 = r0 * r21;
    r12 = r12 * r37;
    r42 = fma(r38, r12, r42);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r15, r42);
    r12 = r13 * r7;
    r18 = r11 + r18;
    r52 = r47 + r52;
    r52 = fma(r8, r52, r9 * r18);
    r12 = r12 * r52;
    r18 = r16 * r6;
    r17 = r22 + r17;
    r17 = fma(r9, r17, r8 * r34);
    r18 = r18 * r6;
    r18 = r18 * r17;
    r18 = fma(r27, r18, r31 * r12);
    r12 = r16 * r7;
    r12 = r12 * r7;
    r12 = r12 * r17;
    r18 = fma(r27, r12, r18);
    r34 = r13 * r6;
    r43 = r26 + r43;
    r43 = r43 + r46;
    r43 = r43 + r49;
    r43 = fma(r9, r43, r8 * r19);
    r34 = r34 * r43;
    r18 = fma(r31, r34, r18);
    r34 = r5 * r18;
    r34 = r34 * r37;
    r12 = r17 * r28;
    r12 = fma(r41, r12, r28 * r34);
    r12 = fma(r43, r1, r12);
    r43 = r0 * r18;
    r43 = r43 * r37;
    r43 = fma(r38, r43, r17 * r32);
    r43 = fma(r52, r1, r43);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r12, r43);
    r52 = r0 * r5;
    r52 = r52 * r13;
    r52 = r52 * r6;
    r52 = r52 * r7;
    r52 = r52 * r27;
    r34 = r27 * r28;
    r9 = r13 * r34;
    r19 = r5 * r6;
    r9 = fma(r19, r9, r1);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r9, r52);
    r8 = r0 * r13;
    r8 = r8 * r7;
    r8 = r8 * r27;
    r8 = fma(r38, r8, r1);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r52, r8);
    r1 = r16 * r6;
    r1 = r1 * r6;
    r49 = r16 * r7;
    r49 = r49 * r7;
    r49 = fma(r27, r49, r27 * r1);
    r1 = r5 * r49;
    r1 = r1 * r37;
    r41 = fma(r28, r41, r28 * r1);
    r1 = r0 * r49;
    r1 = r1 * r37;
    r1 = fma(r38, r1, r32);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r41, r1);
    r32 = r4 * r2;
    r27 = r4 * r3;
    r27 = fma(r45, r27, r23 * r32);
    r32 = r4 * r2;
    r46 = r4 * r3;
    r46 = fma(r42, r46, r15 * r32);
    write_sum_2<double, double>((double*)inout_shared, r27, r46);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r4 * r2;
    r27 = r4 * r3;
    r27 = fma(r43, r27, r12 * r46);
    r46 = r4 * r2;
    r32 = r16 * r3;
    r26 = r38 * r34;
    r32 = fma(r26, r32, r9 * r46);
    write_sum_2<double, double>((double*)inout_shared, r27, r32);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r4 * r3;
    r27 = r4 * r2;
    r27 = fma(r41, r27, r1 * r32);
    r32 = r4 * r3;
    r46 = r16 * r2;
    r46 = fma(r26, r46, r8 * r32);
    write_sum_2<double, double>((double*)inout_shared, r46, r27);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              4 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r45, r45, r23 * r23);
    r46 = fma(r15, r15, r42 * r42);
    write_sum_2<double, double>((double*)inout_shared, r27, r46);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              0 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fma(r12, r12, r43 * r43);
    r27 = 4.00000000000000000e+00;
    r30 = r30 * r30;
    r30 = 1.0 / r30;
    r30 = r27 * r30;
    r27 = r0 * r7;
    r30 = r30 * r28;
    r30 = r30 * r38;
    r30 = r30 * r19;
    r30 = r30 * r27;
    r19 = fma(r9, r9, r30);
    write_sum_2<double, double>((double*)inout_shared, r46, r19);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              2 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r8, r8, r30);
    r19 = fma(r41, r41, r1 * r1);
    write_sum_2<double, double>((double*)inout_shared, r30, r19);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              4 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fma(r23, r15, r45 * r42);
    r30 = fma(r23, r12, r45 * r43);
    write_sum_2<double, double>((double*)inout_shared, r19, r30);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              0 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r45, r52, r23 * r9);
    r19 = fma(r23, r52, r45 * r8);
    write_sum_2<double, double>((double*)inout_shared, r30, r19);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              2 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r23, r41, r45 * r1);
    r45 = fma(r42, r43, r15 * r12);
    write_sum_2<double, double>((double*)inout_shared, r23, r45);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              4 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fma(r42, r52, r15 * r9);
    r23 = fma(r15, r52, r42 * r8);
    write_sum_2<double, double>((double*)inout_shared, r45, r23);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              6 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = fma(r42, r1, r15 * r41);
    r15 = fma(r43, r52, r12 * r9);
    write_sum_2<double, double>((double*)inout_shared, r42, r15);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              8 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fma(r12, r41, r43 * r1);
    r12 = fma(r12, r52, r43 * r8);
    write_sum_2<double, double>((double*)inout_shared, r12, r15);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              10 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fma(r8, r52, r9 * r52);
    r9 = fma(r1, r52, r9 * r41);
    write_sum_2<double, double>((double*)inout_shared, r15, r9);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              12 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fma(r41, r52, r8 * r1);
    write_sum_1<double, double>((double*)inout_shared, r52);
  };
  flush_sum_shared<1, double>(out_pose_precond_tril,
                              14 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = r6 * r40;
    r41 = r7 * r40;
    write_idx_2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        0 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r52,
        r41);
    r41 = r24 * r37;
    r41 = r41 * r28;
    r52 = r0 * r7;
    r52 = r52 * r24;
    r52 = r52 * r37;
    write_idx_2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        2 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r41,
        r52);
    r52 = r7 * r4;
    r52 = r52 * r3;
    r41 = r6 * r4;
    r41 = r41 * r2;
    r41 = fma(r40, r41, r40 * r52);
    r52 = r24 * r4;
    r52 = r52 * r2;
    r52 = r52 * r37;
    r40 = r0 * r7;
    r40 = r40 * r24;
    r40 = r40 * r4;
    r40 = r40 * r3;
    r40 = fma(r37, r40, r28 * r52);
    write_sum_2<double, double>((double*)inout_shared, r41, r40);
  };
  flush_sum_shared<2, double>(out_focal_and_extra_njtr,
                              0 * out_focal_and_extra_njtr_num_alloc,
                              focal_and_extra_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r7 * r7;
    r40 = r40 * r29;
    r40 = r40 * r29;
    r41 = r6 * r6;
    r41 = r41 * r29;
    r41 = r41 * r29;
    r41 = fma(r31, r41, r31 * r40);
    r40 = r6 * r28;
    r52 = r0 * r31;
    r37 = r24 * r24;
    r52 = r52 * r37;
    r37 = r7 * r52;
    r37 = fma(r27, r37, r52 * r40);
    write_sum_2<double, double>((double*)inout_shared, r41, r37);
  };
  flush_sum_shared<2, double>(out_focal_and_extra_precond_diag,
                              0 * out_focal_and_extra_precond_diag_num_alloc,
                              focal_and_extra_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = r0 * r7;
    r37 = r37 * r7;
    r37 = r37 * r24;
    r37 = r37 * r29;
    r41 = r6 * r24;
    r41 = r41 * r29;
    r41 = r41 * r31;
    r41 = fma(r28, r41, r31 * r37);
    write_sum_1<double, double>((double*)inout_shared, r41);
  };
  flush_sum_shared<1, double>(out_focal_and_extra_precond_tril,
                              0 * out_focal_and_extra_precond_tril_num_alloc,
                              focal_and_extra_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = r4 * r2;
    r37 = r4 * r3;
    write_sum_2<double, double>((double*)inout_shared, r41, r37);
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
  sum_flush_final<double>(out_rTr_local, out_rTr, 1);
}

void simple_radial_fixed_point_res_jac_first(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
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
    double* out_focal_and_extra_jac,
    unsigned int out_focal_and_extra_jac_num_alloc,
    double* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    double* const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    double* const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
    double* out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    double* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    double* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    double* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_point_res_jac_first_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      focal_and_extra,
      focal_and_extra_num_alloc,
      focal_and_extra_indices,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
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
      out_focal_and_extra_jac,
      out_focal_and_extra_jac_num_alloc,
      out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc,
      out_focal_and_extra_precond_diag,
      out_focal_and_extra_precond_diag_num_alloc,
      out_focal_and_extra_precond_tril,
      out_focal_and_extra_precond_tril_num_alloc,
      out_principal_point_jac,
      out_principal_point_jac_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar