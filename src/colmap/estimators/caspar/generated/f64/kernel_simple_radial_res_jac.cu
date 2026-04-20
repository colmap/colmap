#include "kernel_simple_radial_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) simple_radial_res_jac_kernel(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
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
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58;
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
    r30 = fma(r5, r38, r25);
    r42 = 1.0 / r39;
    r43 = r30 * r42;
    r2 = fma(r29, r43, r2);
    r3 = fma(r3, r4, r1);
    r1 = r0 * r30;
    r1 = r1 * r42;
    r3 = fma(r7, r1, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r44 = r15 * r10;
    r44 = r44 * r13;
    r41 = r41 + r44;
    r45 = r4 * r26;
    r46 = r23 + r45;
    r11 = r11 * r11;
    r47 = r4 * r37;
    r48 = r11 + r47;
    r49 = r46 + r48;
    r49 = fma(r9, r49, r19 * r41);
    r41 = r6 * r6;
    r41 = r13 * r41;
    r31 = r39 * r31;
    r39 = 1.0 / r31;
    r41 = r41 * r39;
    r50 = r7 * r7;
    r50 = r13 * r50;
    r50 = r50 * r39;
    r51 = fma(r49, r50, r49 * r41);
    r52 = r17 * r7;
    r53 = r4 * r11;
    r54 = r37 + r53;
    r46 = r46 + r54;
    r46 = fma(r19, r46, r9 * r33);
    r52 = r52 * r46;
    r51 = fma(r32, r52, r51);
    r55 = r17 * r6;
    r15 = r14 * r15;
    r15 = r15 * r13;
    r40 = r40 + r15;
    r40 = fma(r9, r22, r19 * r40);
    r55 = r55 * r40;
    r51 = fma(r32, r55, r51);
    r55 = r5 * r51;
    r55 = r55 * r42;
    r40 = fma(r40, r1, r29 * r55);
    r55 = r49 * r29;
    r52 = r30 * r4;
    r52 = r52 * r32;
    r40 = fma(r52, r55, r40);
    r55 = r0 * r51;
    r56 = r5 * r7;
    r55 = r55 * r42;
    r46 = fma(r46, r1, r56 * r55);
    r55 = r0 * r7;
    r55 = r55 * r49;
    r46 = fma(r52, r55, r46);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r40, r46);
    r55 = r4 * r23;
    r57 = r26 + r55;
    r54 = r54 + r57;
    r54 = fma(r8, r54, r19 * r20);
    r58 = r17 * r7;
    r44 = r34 + r44;
    r44 = fma(r8, r44, r19 * r16);
    r58 = r58 * r44;
    r58 = fma(r32, r58, r54 * r50);
    r34 = r17 * r6;
    r10 = r14 * r10;
    r10 = r10 * r13;
    r36 = r36 + r10;
    r11 = r37 + r11;
    r11 = r11 + r45;
    r11 = r11 + r55;
    r11 = fma(r19, r11, r8 * r36);
    r34 = r34 * r11;
    r58 = fma(r32, r34, r58);
    r58 = fma(r54, r41, r58);
    r34 = r5 * r58;
    r34 = r34 * r42;
    r11 = fma(r11, r1, r29 * r34);
    r34 = r54 * r29;
    r11 = fma(r52, r34, r11);
    r34 = r0 * r7;
    r34 = r34 * r54;
    r34 = fma(r52, r34, r44 * r1);
    r44 = r0 * r58;
    r44 = r44 * r42;
    r34 = fma(r56, r44, r34);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r11, r34);
    r44 = r17 * r7;
    r15 = r12 + r15;
    r57 = r48 + r57;
    r57 = fma(r8, r57, r9 * r15);
    r44 = r44 * r57;
    r10 = r21 + r10;
    r10 = fma(r9, r10, r8 * r35);
    r44 = fma(r10, r41, r32 * r44);
    r21 = r17 * r6;
    r23 = r26 + r23;
    r23 = r23 + r47;
    r23 = r23 + r53;
    r23 = fma(r9, r23, r8 * r18);
    r21 = r21 * r23;
    r44 = fma(r32, r21, r44);
    r44 = fma(r10, r50, r44);
    r21 = r5 * r44;
    r21 = r21 * r42;
    r10 = r10 * r52;
    r21 = fma(r29, r10, r29 * r21);
    r21 = fma(r23, r1, r21);
    r23 = r0 * r7;
    r9 = r0 * r44;
    r9 = r9 * r42;
    r9 = fma(r56, r9, r10 * r23);
    r9 = fma(r57, r1, r9);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r21, r9);
    r57 = r0 * r5;
    r57 = r57 * r17;
    r57 = r57 * r6;
    r57 = r57 * r7;
    r57 = r57 * r39;
    r10 = r5 * r17;
    r10 = r10 * r6;
    r10 = r10 * r39;
    r10 = fma(r29, r10, r1);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r10, r57);
    r8 = r0 * r17;
    r8 = r8 * r7;
    r8 = r8 * r39;
    r8 = fma(r56, r8, r1);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r57, r8);
    r53 = r41 + r50;
    r47 = r5 * r53;
    r47 = r47 * r42;
    r47 = fma(r29, r52, r29 * r47);
    r26 = r0 * r53;
    r26 = r26 * r42;
    r15 = r0 * r7;
    r15 = fma(r52, r15, r56 * r26);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r47, r15);
    r26 = r4 * r2;
    r48 = r4 * r3;
    r48 = fma(r46, r48, r40 * r26);
    r26 = r4 * r2;
    r12 = r4 * r3;
    r12 = fma(r34, r12, r11 * r26);
    write_sum_2<double, double>((double*)inout_shared, r48, r12);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = r4 * r2;
    r48 = r4 * r3;
    r48 = fma(r9, r48, r21 * r12);
    r12 = r4 * r2;
    r39 = r13 * r39;
    r13 = r3 * r39;
    r26 = r29 * r56;
    r13 = fma(r26, r13, r10 * r12);
    write_sum_2<double, double>((double*)inout_shared, r48, r13);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r4 * r3;
    r48 = r4 * r2;
    r48 = fma(r47, r48, r15 * r13);
    r13 = r4 * r3;
    r12 = r2 * r39;
    r12 = fma(r26, r12, r8 * r13);
    write_sum_2<double, double>((double*)inout_shared, r12, r48);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              4 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = fma(r46, r46, r40 * r40);
    r12 = fma(r11, r11, r34 * r34);
    write_sum_2<double, double>((double*)inout_shared, r48, r12);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              0 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fma(r21, r21, r9 * r9);
    r48 = r0 * r5;
    r13 = 4.00000000000000000e+00;
    r31 = r31 * r31;
    r31 = 1.0 / r31;
    r48 = r48 * r6;
    r48 = r48 * r7;
    r48 = r48 * r13;
    r48 = r48 * r31;
    r48 = r48 * r26;
    r26 = fma(r10, r10, r48);
    write_sum_2<double, double>((double*)inout_shared, r12, r26);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              2 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = fma(r8, r8, r48);
    r26 = fma(r47, r47, r15 * r15);
    write_sum_2<double, double>((double*)inout_shared, r48, r26);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              4 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fma(r40, r11, r46 * r34);
    r48 = fma(r40, r21, r46 * r9);
    write_sum_2<double, double>((double*)inout_shared, r26, r48);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              0 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = fma(r46, r57, r40 * r10);
    r26 = fma(r40, r57, r46 * r8);
    write_sum_2<double, double>((double*)inout_shared, r48, r26);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              2 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fma(r40, r47, r46 * r15);
    r46 = fma(r34, r9, r11 * r21);
    write_sum_2<double, double>((double*)inout_shared, r40, r46);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              4 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fma(r34, r57, r11 * r10);
    r40 = fma(r11, r57, r34 * r8);
    write_sum_2<double, double>((double*)inout_shared, r46, r40);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              6 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fma(r34, r15, r11 * r47);
    r11 = fma(r9, r57, r21 * r10);
    write_sum_2<double, double>((double*)inout_shared, r34, r11);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              8 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fma(r21, r47, r9 * r15);
    r21 = fma(r21, r57, r9 * r8);
    write_sum_2<double, double>((double*)inout_shared, r21, r11);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              10 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fma(r8, r57, r10 * r57);
    r10 = fma(r15, r57, r10 * r47);
    write_sum_2<double, double>((double*)inout_shared, r11, r10);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              12 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fma(r47, r57, r8 * r15);
    write_sum_1<double, double>((double*)inout_shared, r57);
  };
  flush_sum_shared<1, double>(out_pose_precond_tril,
                              14 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = r6 * r43;
    r47 = r7 * r43;
    write_idx_2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        0 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r57,
        r47);
    r47 = r38 * r42;
    r47 = r47 * r29;
    r57 = r0 * r7;
    r57 = r57 * r38;
    r57 = r57 * r42;
    write_idx_2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        2 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r47,
        r57);
    r57 = r7 * r4;
    r57 = r57 * r3;
    r47 = r6 * r4;
    r47 = r47 * r2;
    r47 = fma(r43, r47, r43 * r57);
    r57 = r38 * r4;
    r57 = r57 * r2;
    r57 = r57 * r42;
    r43 = r0 * r7;
    r43 = r43 * r38;
    r43 = r43 * r4;
    r43 = r43 * r3;
    r43 = fma(r42, r43, r29 * r57);
    write_sum_2<double, double>((double*)inout_shared, r47, r43);
  };
  flush_sum_shared<2, double>(out_focal_and_extra_njtr,
                              0 * out_focal_and_extra_njtr_num_alloc,
                              focal_and_extra_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r7 * r7;
    r43 = r43 * r30;
    r43 = r43 * r30;
    r47 = r6 * r6;
    r47 = r47 * r30;
    r47 = r47 * r30;
    r47 = fma(r32, r47, r32 * r43);
    r43 = r6 * r29;
    r57 = r0 * r32;
    r15 = r38 * r38;
    r57 = r57 * r15;
    r15 = r7 * r57;
    r15 = fma(r23, r15, r57 * r43);
    write_sum_2<double, double>((double*)inout_shared, r47, r15);
  };
  flush_sum_shared<2, double>(out_focal_and_extra_precond_diag,
                              0 * out_focal_and_extra_precond_diag_num_alloc,
                              focal_and_extra_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = r0 * r7;
    r15 = r15 * r7;
    r15 = r15 * r38;
    r15 = r15 * r30;
    r47 = r6 * r38;
    r47 = r47 * r30;
    r47 = r47 * r32;
    r47 = fma(r29, r47, r32 * r15);
    write_sum_1<double, double>((double*)inout_shared, r47);
  };
  flush_sum_shared<1, double>(out_focal_and_extra_precond_tril,
                              0 * out_focal_and_extra_precond_tril_num_alloc,
                              focal_and_extra_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = r4 * r2;
    r15 = r4 * r3;
    write_sum_2<double, double>((double*)inout_shared, r47, r15);
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
    r25 = r20 * r29;
    r25 = fma(r52, r25, r28 * r1);
    r15 = r17 * r28;
    r15 = r15 * r6;
    r15 = fma(r32, r15, r20 * r41);
    r47 = r17 * r16;
    r47 = r47 * r7;
    r15 = fma(r32, r47, r15);
    r15 = fma(r20, r50, r15);
    r47 = r5 * r15;
    r47 = r47 * r42;
    r25 = fma(r29, r47, r25);
    r47 = r0 * r15;
    r47 = r47 * r42;
    r30 = r0 * r20;
    r30 = r30 * r7;
    r30 = fma(r52, r30, r56 * r47);
    r30 = fma(r16, r1, r30);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               0 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r25,
                                               r30);
    r47 = r35 * r29;
    r47 = fma(r18, r1, r52 * r47);
    r43 = r17 * r18;
    r43 = r43 * r6;
    r43 = fma(r35, r41, r32 * r43);
    r23 = r17 * r24;
    r23 = r23 * r7;
    r43 = fma(r32, r23, r43);
    r43 = fma(r35, r50, r43);
    r23 = r5 * r43;
    r23 = r23 * r42;
    r47 = fma(r29, r23, r47);
    r23 = r0 * r43;
    r23 = r23 * r42;
    r23 = fma(r24, r1, r56 * r23);
    r8 = r0 * r35;
    r8 = r8 * r7;
    r23 = fma(r52, r8, r23);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               2 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r47,
                                               r23);
    r8 = r27 * r29;
    r8 = fma(r22, r1, r52 * r8);
    r10 = r17 * r22;
    r10 = r10 * r6;
    r41 = fma(r27, r41, r32 * r10);
    r10 = r17 * r33;
    r10 = r10 * r7;
    r41 = fma(r32, r10, r41);
    r41 = fma(r27, r50, r41);
    r50 = r5 * r41;
    r50 = r50 * r42;
    r8 = fma(r29, r50, r8);
    r50 = r0 * r41;
    r50 = r50 * r42;
    r1 = fma(r33, r1, r56 * r50);
    r50 = r0 * r27;
    r50 = r50 * r7;
    r1 = fma(r52, r50, r1);
    write_idx_2<1024, double, double, double2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r8, r1);
    r50 = r4 * r3;
    r52 = r4 * r2;
    r52 = fma(r25, r52, r30 * r50);
    r50 = r4 * r3;
    r56 = r4 * r2;
    r56 = fma(r47, r56, r23 * r50);
    write_sum_2<double, double>((double*)inout_shared, r52, r56);
  };
  flush_sum_shared<2, double>(out_point_njtr,
                              0 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = r4 * r2;
    r52 = r4 * r3;
    r52 = fma(r1, r52, r8 * r56);
    write_sum_1<double, double>((double*)inout_shared, r52);
  };
  flush_sum_shared<1, double>(out_point_njtr,
                              2 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fma(r30, r30, r25 * r25);
    r56 = fma(r47, r47, r23 * r23);
    write_sum_2<double, double>((double*)inout_shared, r52, r56);
  };
  flush_sum_shared<2, double>(out_point_precond_diag,
                              0 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = fma(r1, r1, r8 * r8);
    write_sum_1<double, double>((double*)inout_shared, r56);
  };
  flush_sum_shared<1, double>(out_point_precond_diag,
                              2 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = fma(r30, r23, r25 * r47);
    r25 = fma(r25, r8, r30 * r1);
    write_sum_2<double, double>((double*)inout_shared, r56, r25);
  };
  flush_sum_shared<2, double>(out_point_precond_tril,
                              0 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = fma(r47, r8, r23 * r1);
    write_sum_1<double, double>((double*)inout_shared, r8);
  };
  flush_sum_shared<1, double>(out_point_precond_tril,
                              2 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
}

void simple_radial_res_jac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
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
  simple_radial_res_jac_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      focal_and_extra,
      focal_and_extra_num_alloc,
      focal_and_extra_indices,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
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