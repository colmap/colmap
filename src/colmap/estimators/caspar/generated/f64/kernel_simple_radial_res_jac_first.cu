#include "kernel_simple_radial_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) simple_radial_res_jac_first_kernel(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    double* extra_calib,
    unsigned int extra_calib_num_alloc,
    SharedIndex* extra_calib_indices,
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
    double* out_focal_jac,
    unsigned int out_focal_jac_num_alloc,
    double* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    double* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    double* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
    double* out_extra_calib_jac,
    unsigned int out_extra_calib_jac_num_alloc,
    double* const out_extra_calib_njtr,
    unsigned int out_extra_calib_njtr_num_alloc,
    double* const out_extra_calib_precond_diag,
    unsigned int out_extra_calib_precond_diag_num_alloc,
    double* const out_extra_calib_precond_tril,
    unsigned int out_extra_calib_precond_tril_num_alloc,
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
  __shared__ SharedIndex focal_indices_loc[1024];
  focal_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex extra_calib_indices_loc[1024];
  extra_calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? extra_calib_indices[global_thread_idx]
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
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58;
  load_shared<2, double, double>(extra_calib,
                                 0 * extra_calib_num_alloc,
                                 extra_calib_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          extra_calib_indices_loc[threadIdx.x].target,
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
  load_shared<1, double, double>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, focal_indices_loc[threadIdx.x].target, r0);
  };
  __syncthreads();
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
  };
  load_shared<1, double, double>(extra_calib,
                                 2 * extra_calib_num_alloc,
                                 extra_calib_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared,
                          extra_calib_indices_loc[threadIdx.x].target,
                          r29);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
    r30 = fma(r29, r38, r24);
    r42 = 1.0 / r39;
    r43 = r30 * r42;
    r2 = fma(r28, r43, r2);
    r3 = fma(r3, r4, r1);
    r1 = r0 * r30;
    r1 = r1 * r42;
    r3 = fma(r6, r1, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r44 = fma(r3, r3, r2 * r2);
  };
  sum_store<double>(out_rTr_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r44);
  if (global_thread_idx < problem_size) {
    r44 = r14 * r9;
    r44 = r44 * r12;
    r41 = r41 + r44;
    r45 = r4 * r25;
    r46 = r22 + r45;
    r10 = r10 * r10;
    r47 = r4 * r37;
    r48 = r10 + r47;
    r49 = r46 + r48;
    r49 = fma(r8, r49, r18 * r41);
    r41 = r49 * r28;
    r50 = r30 * r4;
    r50 = r50 * r32;
    r14 = r13 * r14;
    r14 = r14 * r12;
    r40 = r40 + r14;
    r40 = fma(r8, r21, r18 * r40);
    r41 = fma(r40, r1, r50 * r41);
    r51 = r5 * r5;
    r51 = r12 * r51;
    r31 = r39 * r31;
    r39 = 1.0 / r31;
    r51 = r51 * r39;
    r52 = r6 * r6;
    r52 = r12 * r52;
    r52 = r52 * r39;
    r53 = fma(r49, r52, r49 * r51);
    r54 = r16 * r6;
    r55 = r4 * r10;
    r56 = r37 + r55;
    r46 = r46 + r56;
    r46 = fma(r18, r46, r8 * r33);
    r54 = r54 * r46;
    r53 = fma(r32, r54, r53);
    r57 = r16 * r5;
    r57 = r57 * r40;
    r53 = fma(r32, r57, r53);
    r57 = r29 * r53;
    r57 = r57 * r42;
    r41 = fma(r28, r57, r41);
    r57 = r0 * r53;
    r54 = r29 * r6;
    r57 = r57 * r42;
    r46 = fma(r46, r1, r54 * r57);
    r57 = r0 * r6;
    r57 = r57 * r49;
    r46 = fma(r50, r57, r46);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r41, r46);
    r57 = r4 * r22;
    r40 = r25 + r57;
    r56 = r56 + r40;
    r56 = fma(r7, r56, r18 * r19);
    r58 = r16 * r6;
    r44 = r34 + r44;
    r44 = fma(r7, r44, r18 * r15);
    r58 = r58 * r44;
    r58 = fma(r32, r58, r56 * r52);
    r34 = r16 * r5;
    r9 = r13 * r9;
    r9 = r9 * r12;
    r36 = r36 + r9;
    r10 = r37 + r10;
    r10 = r10 + r45;
    r10 = r10 + r57;
    r10 = fma(r18, r10, r7 * r36);
    r34 = r34 * r10;
    r58 = fma(r32, r34, r58);
    r58 = fma(r56, r51, r58);
    r34 = r29 * r58;
    r34 = r34 * r42;
    r18 = r56 * r28;
    r18 = fma(r50, r18, r28 * r34);
    r18 = fma(r10, r1, r18);
    r10 = r0 * r6;
    r10 = r10 * r56;
    r44 = fma(r44, r1, r50 * r10);
    r10 = r0 * r58;
    r10 = r10 * r42;
    r44 = fma(r54, r10, r44);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r18, r44);
    r9 = r20 + r9;
    r9 = fma(r8, r9, r7 * r35);
    r20 = r9 * r28;
    r22 = r25 + r22;
    r22 = r22 + r47;
    r22 = r22 + r55;
    r22 = fma(r8, r22, r7 * r17);
    r20 = fma(r22, r1, r50 * r20);
    r55 = r16 * r6;
    r14 = r11 + r14;
    r40 = r48 + r40;
    r40 = fma(r7, r40, r8 * r14);
    r55 = r55 * r40;
    r55 = fma(r9, r51, r32 * r55);
    r7 = r16 * r5;
    r7 = r7 * r22;
    r55 = fma(r32, r7, r55);
    r55 = fma(r9, r52, r55);
    r7 = r29 * r55;
    r7 = r7 * r42;
    r20 = fma(r28, r7, r20);
    r7 = r0 * r6;
    r7 = r7 * r9;
    r7 = fma(r50, r7, r40 * r1);
    r40 = r0 * r55;
    r40 = r40 * r42;
    r7 = fma(r54, r40, r7);
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
    r14 = fma(r54, r14, r1);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r40, r14);
    r8 = r51 + r52;
    r48 = r29 * r8;
    r48 = r48 * r42;
    r48 = fma(r28, r48, r28 * r50);
    r11 = r0 * r6;
    r47 = r0 * r8;
    r47 = r47 * r42;
    r47 = fma(r54, r47, r50 * r11);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r48, r47);
    r11 = r4 * r2;
    r25 = r4 * r3;
    r25 = fma(r46, r25, r41 * r11);
    r11 = r4 * r2;
    r10 = r4 * r3;
    r10 = fma(r44, r10, r18 * r11);
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
    r11 = r28 * r54;
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
    r25 = fma(r48, r25, r47 * r12);
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
    r25 = fma(r41, r41, r46 * r46);
    r10 = fma(r18, r18, r44 * r44);
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
    r11 = fma(r47, r47, r48 * r48);
    write_sum_2<double, double>((double*)inout_shared, r25, r11);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              4 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fma(r41, r18, r46 * r44);
    r25 = fma(r46, r7, r41 * r20);
    write_sum_2<double, double>((double*)inout_shared, r11, r25);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              0 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = fma(r46, r40, r41 * r22);
    r11 = fma(r41, r40, r46 * r14);
    write_sum_2<double, double>((double*)inout_shared, r25, r11);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              2 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r41, r48, r46 * r47);
    r46 = fma(r44, r7, r18 * r20);
    write_sum_2<double, double>((double*)inout_shared, r41, r46);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              4 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fma(r44, r40, r18 * r22);
    r41 = fma(r18, r40, r44 * r14);
    write_sum_2<double, double>((double*)inout_shared, r46, r41);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              6 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = fma(r44, r47, r18 * r48);
    r18 = fma(r7, r40, r20 * r22);
    write_sum_2<double, double>((double*)inout_shared, r44, r18);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              8 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = fma(r7, r47, r20 * r48);
    r20 = fma(r20, r40, r7 * r14);
    write_sum_2<double, double>((double*)inout_shared, r20, r18);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              10 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = fma(r14, r40, r22 * r40);
    r22 = fma(r47, r40, r22 * r48);
    write_sum_2<double, double>((double*)inout_shared, r18, r22);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              12 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fma(r48, r40, r14 * r47);
    write_sum_1<double, double>((double*)inout_shared, r40);
  };
  flush_sum_shared<1, double>(out_pose_precond_tril,
                              14 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r5 * r43;
    r48 = r6 * r43;
    write_idx_2<1024, double, double, double2>(out_focal_jac,
                                               0 * out_focal_jac_num_alloc,
                                               global_thread_idx,
                                               r40,
                                               r48);
    r48 = r5 * r4;
    r48 = r48 * r2;
    r40 = r6 * r4;
    r40 = r40 * r3;
    r40 = fma(r43, r40, r43 * r48);
    write_sum_1<double, double>((double*)inout_shared, r40);
  };
  flush_sum_shared<1, double>(out_focal_njtr,
                              0 * out_focal_njtr_num_alloc,
                              focal_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r6 * r6;
    r40 = r40 * r30;
    r40 = r40 * r30;
    r48 = r5 * r5;
    r48 = r48 * r30;
    r48 = r48 * r30;
    r48 = fma(r32, r48, r32 * r40);
    write_sum_1<double, double>((double*)inout_shared, r48);
  };
  flush_sum_shared<1, double>(out_focal_precond_diag,
                              0 * out_focal_precond_diag_num_alloc,
                              focal_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r38 * r42;
    r48 = r48 * r28;
    r40 = r0 * r6;
    r40 = r40 * r38;
    r40 = r40 * r42;
    write_idx_2<1024, double, double, double2>(
        out_extra_calib_jac,
        0 * out_extra_calib_jac_num_alloc,
        global_thread_idx,
        r48,
        r40);
    r30 = r4 * r2;
    r43 = r4 * r3;
    write_sum_2<double, double>((double*)inout_shared, r30, r43);
  };
  flush_sum_shared<2, double>(out_extra_calib_njtr,
                              0 * out_extra_calib_njtr_num_alloc,
                              extra_calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r0 * r6;
    r43 = r43 * r38;
    r43 = r43 * r4;
    r43 = r43 * r3;
    r30 = r38 * r4;
    r30 = r30 * r2;
    r30 = r30 * r42;
    r30 = fma(r28, r30, r42 * r43);
    write_sum_1<double, double>((double*)inout_shared, r30);
  };
  flush_sum_shared<1, double>(out_extra_calib_njtr,
                              2 * out_extra_calib_njtr_num_alloc,
                              extra_calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<double, double>((double*)inout_shared, r24, r24);
  };
  flush_sum_shared<2, double>(out_extra_calib_precond_diag,
                              0 * out_extra_calib_precond_diag_num_alloc,
                              extra_calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = r5 * r28;
    r30 = r0 * r32;
    r43 = r38 * r38;
    r30 = r30 * r43;
    r43 = r6 * r30;
    r47 = r0 * r6;
    r43 = fma(r47, r43, r30 * r24);
    write_sum_1<double, double>((double*)inout_shared, r43);
  };
  flush_sum_shared<1, double>(out_extra_calib_precond_diag,
                              2 * out_extra_calib_precond_diag_num_alloc,
                              extra_calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = 0.00000000000000000e+00;
    write_sum_2<double, double>((double*)inout_shared, r43, r48);
  };
  flush_sum_shared<2, double>(out_extra_calib_precond_tril,
                              0 * out_extra_calib_precond_tril_num_alloc,
                              extra_calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_1<double, double>((double*)inout_shared, r40);
  };
  flush_sum_shared<1, double>(out_extra_calib_precond_tril,
                              2 * out_extra_calib_precond_tril_num_alloc,
                              extra_calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r19 * r28;
    r40 = fma(r50, r40, r27 * r1);
    r48 = r16 * r27;
    r48 = r48 * r5;
    r48 = fma(r32, r48, r19 * r51);
    r43 = r16 * r15;
    r43 = r43 * r6;
    r48 = fma(r32, r43, r48);
    r48 = fma(r19, r52, r48);
    r43 = r29 * r48;
    r43 = r43 * r42;
    r40 = fma(r28, r43, r40);
    r43 = r0 * r19;
    r43 = r43 * r6;
    r24 = r0 * r48;
    r24 = r24 * r42;
    r24 = fma(r54, r24, r50 * r43);
    r24 = fma(r15, r1, r24);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               0 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r40,
                                               r24);
    r43 = r16 * r17;
    r43 = r43 * r5;
    r43 = fma(r35, r51, r32 * r43);
    r14 = r16 * r23;
    r14 = r14 * r6;
    r43 = fma(r32, r14, r43);
    r43 = fma(r35, r52, r43);
    r14 = r29 * r43;
    r14 = r14 * r42;
    r14 = fma(r17, r1, r28 * r14);
    r35 = r35 * r50;
    r14 = fma(r28, r35, r14);
    r22 = r0 * r43;
    r22 = r22 * r42;
    r22 = fma(r23, r1, r54 * r22);
    r22 = fma(r35, r47, r22);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               2 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r14,
                                               r22);
    r47 = r16 * r21;
    r47 = r47 * r5;
    r51 = fma(r26, r51, r32 * r47);
    r47 = r16 * r33;
    r47 = r47 * r6;
    r51 = fma(r32, r47, r51);
    r51 = fma(r26, r52, r51);
    r52 = r29 * r51;
    r52 = r52 * r42;
    r52 = fma(r28, r52, r21 * r1);
    r47 = r26 * r28;
    r52 = fma(r50, r47, r52);
    r47 = r0 * r26;
    r47 = r47 * r6;
    r47 = fma(r50, r47, r33 * r1);
    r1 = r0 * r51;
    r1 = r1 * r42;
    r47 = fma(r54, r1, r47);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               4 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r52,
                                               r47);
    r1 = r4 * r3;
    r54 = r4 * r2;
    r54 = fma(r40, r54, r24 * r1);
    r1 = r4 * r3;
    r42 = r4 * r2;
    r42 = fma(r14, r42, r22 * r1);
    write_sum_2<double, double>((double*)inout_shared, r54, r42);
  };
  flush_sum_shared<2, double>(out_point_njtr,
                              0 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r4 * r3;
    r54 = r4 * r2;
    r54 = fma(r52, r54, r47 * r42);
    write_sum_1<double, double>((double*)inout_shared, r54);
  };
  flush_sum_shared<1, double>(out_point_njtr,
                              2 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = fma(r40, r40, r24 * r24);
    r42 = fma(r14, r14, r22 * r22);
    write_sum_2<double, double>((double*)inout_shared, r54, r42);
  };
  flush_sum_shared<2, double>(out_point_precond_diag,
                              0 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = fma(r52, r52, r47 * r47);
    write_sum_1<double, double>((double*)inout_shared, r42);
  };
  flush_sum_shared<1, double>(out_point_precond_diag,
                              2 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = fma(r24, r22, r40 * r14);
    r24 = fma(r24, r47, r40 * r52);
    write_sum_2<double, double>((double*)inout_shared, r42, r24);
  };
  flush_sum_shared<2, double>(out_point_precond_tril,
                              0 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fma(r14, r52, r22 * r47);
    write_sum_1<double, double>((double*)inout_shared, r52);
  };
  flush_sum_shared<1, double>(out_point_precond_tril,
                              2 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  sum_flush_final<double>(out_rTr_local, out_rTr, 1);
}

void simple_radial_res_jac_first(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    double* extra_calib,
    unsigned int extra_calib_num_alloc,
    SharedIndex* extra_calib_indices,
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
    double* out_focal_jac,
    unsigned int out_focal_jac_num_alloc,
    double* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    double* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    double* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
    double* out_extra_calib_jac,
    unsigned int out_extra_calib_jac_num_alloc,
    double* const out_extra_calib_njtr,
    unsigned int out_extra_calib_njtr_num_alloc,
    double* const out_extra_calib_precond_diag,
    unsigned int out_extra_calib_precond_diag_num_alloc,
    double* const out_extra_calib_precond_tril,
    unsigned int out_extra_calib_precond_tril_num_alloc,
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
  simple_radial_res_jac_first_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      focal,
      focal_num_alloc,
      focal_indices,
      extra_calib,
      extra_calib_num_alloc,
      extra_calib_indices,
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
      out_focal_jac,
      out_focal_jac_num_alloc,
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      out_focal_precond_diag,
      out_focal_precond_diag_num_alloc,
      out_focal_precond_tril,
      out_focal_precond_tril_num_alloc,
      out_extra_calib_jac,
      out_extra_calib_jac_num_alloc,
      out_extra_calib_njtr,
      out_extra_calib_njtr_num_alloc,
      out_extra_calib_precond_diag,
      out_extra_calib_precond_diag_num_alloc,
      out_extra_calib_precond_tril,
      out_extra_calib_precond_tril_num_alloc,
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