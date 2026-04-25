#include "kernel_simple_radial_merged_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) simple_radial_merged_res_jac_kernel(
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58;
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
    r41 = r49 * r29;
    r50 = r30 * r4;
    r50 = r50 * r32;
    r51 = r6 * r6;
    r51 = r13 * r51;
    r31 = r39 * r31;
    r39 = 1.0 / r31;
    r51 = r51 * r39;
    r52 = r7 * r7;
    r52 = r13 * r52;
    r52 = r52 * r39;
    r53 = fma(r49, r52, r49 * r51);
    r54 = r17 * r7;
    r55 = r4 * r11;
    r56 = r37 + r55;
    r46 = r46 + r56;
    r46 = fma(r19, r46, r9 * r33);
    r54 = r54 * r46;
    r53 = fma(r32, r54, r53);
    r57 = r17 * r6;
    r15 = r14 * r15;
    r15 = r15 * r13;
    r40 = r40 + r15;
    r40 = fma(r9, r22, r19 * r40);
    r57 = r57 * r40;
    r53 = fma(r32, r57, r53);
    r57 = r5 * r53;
    r57 = r57 * r42;
    r57 = fma(r29, r57, r50 * r41);
    r57 = fma(r40, r1, r57);
    r40 = r0 * r7;
    r40 = r40 * r49;
    r40 = fma(r50, r40, r46 * r1);
    r46 = r0 * r53;
    r41 = r5 * r7;
    r46 = r46 * r42;
    r40 = fma(r41, r46, r40);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r57, r40);
    r46 = r4 * r23;
    r54 = r26 + r46;
    r56 = r56 + r54;
    r56 = fma(r8, r56, r19 * r20);
    r58 = r17 * r7;
    r44 = r34 + r44;
    r44 = fma(r8, r44, r19 * r16);
    r58 = r58 * r44;
    r58 = fma(r32, r58, r56 * r52);
    r34 = r17 * r6;
    r10 = r14 * r10;
    r10 = r10 * r13;
    r36 = r36 + r10;
    r11 = r37 + r11;
    r11 = r11 + r45;
    r11 = r11 + r46;
    r11 = fma(r19, r11, r8 * r36);
    r34 = r34 * r11;
    r58 = fma(r32, r34, r58);
    r58 = fma(r56, r51, r58);
    r34 = r5 * r58;
    r34 = r34 * r42;
    r19 = r56 * r29;
    r19 = fma(r50, r19, r29 * r34);
    r19 = fma(r11, r1, r19);
    r11 = r0 * r7;
    r11 = r11 * r56;
    r44 = fma(r44, r1, r50 * r11);
    r11 = r0 * r58;
    r11 = r11 * r42;
    r44 = fma(r41, r11, r44);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r19, r44);
    r11 = r17 * r7;
    r15 = r12 + r15;
    r54 = r48 + r54;
    r54 = fma(r8, r54, r9 * r15);
    r11 = r11 * r54;
    r10 = r21 + r10;
    r10 = fma(r9, r10, r8 * r35);
    r11 = fma(r10, r51, r32 * r11);
    r21 = r17 * r6;
    r23 = r26 + r23;
    r23 = r23 + r47;
    r23 = r23 + r55;
    r23 = fma(r9, r23, r8 * r18);
    r21 = r21 * r23;
    r11 = fma(r32, r21, r11);
    r11 = fma(r10, r52, r11);
    r21 = r5 * r11;
    r21 = r21 * r42;
    r9 = r10 * r29;
    r9 = fma(r50, r9, r29 * r21);
    r9 = fma(r23, r1, r9);
    r23 = r0 * r7;
    r23 = r23 * r10;
    r54 = fma(r54, r1, r50 * r23);
    r23 = r0 * r11;
    r23 = r23 * r42;
    r54 = fma(r41, r23, r54);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r9, r54);
    r23 = r0 * r5;
    r23 = r23 * r17;
    r23 = r23 * r6;
    r23 = r23 * r7;
    r23 = r23 * r39;
    r21 = r5 * r17;
    r21 = r21 * r6;
    r21 = r21 * r39;
    r21 = fma(r29, r21, r1);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r21, r23);
    r8 = r0 * r17;
    r8 = r8 * r7;
    r8 = r8 * r39;
    r8 = fma(r41, r8, r1);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r23, r8);
    r55 = r51 + r52;
    r47 = r5 * r55;
    r47 = r47 * r42;
    r47 = fma(r29, r47, r29 * r50);
    r26 = r0 * r7;
    r15 = r0 * r55;
    r15 = r15 * r42;
    r15 = fma(r41, r15, r50 * r26);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r47, r15);
    r26 = r4 * r2;
    r48 = r4 * r3;
    r48 = fma(r40, r48, r57 * r26);
    r26 = r4 * r2;
    r12 = r4 * r3;
    r12 = fma(r44, r12, r19 * r26);
    write_sum_2<double, double>((double*)inout_shared, r48, r12);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = r4 * r3;
    r48 = r4 * r2;
    r48 = fma(r9, r48, r54 * r12);
    r12 = r4 * r2;
    r39 = r13 * r39;
    r13 = r3 * r39;
    r26 = r29 * r41;
    r13 = fma(r26, r13, r21 * r12);
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
    r48 = fma(r57, r57, r40 * r40);
    r12 = fma(r44, r44, r19 * r19);
    write_sum_2<double, double>((double*)inout_shared, r48, r12);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              0 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fma(r9, r9, r54 * r54);
    r48 = r0 * r5;
    r13 = 4.00000000000000000e+00;
    r31 = r31 * r31;
    r31 = 1.0 / r31;
    r48 = r48 * r6;
    r48 = r48 * r7;
    r48 = r48 * r13;
    r48 = r48 * r31;
    r48 = r48 * r26;
    r26 = fma(r21, r21, r48);
    write_sum_2<double, double>((double*)inout_shared, r12, r26);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              2 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = fma(r8, r8, r48);
    r26 = fma(r15, r15, r47 * r47);
    write_sum_2<double, double>((double*)inout_shared, r48, r26);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              4 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fma(r57, r19, r40 * r44);
    r48 = fma(r57, r9, r40 * r54);
    write_sum_2<double, double>((double*)inout_shared, r26, r48);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              0 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = fma(r40, r23, r57 * r21);
    r26 = fma(r57, r23, r40 * r8);
    write_sum_2<double, double>((double*)inout_shared, r48, r26);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              2 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fma(r40, r15, r57 * r47);
    r57 = fma(r19, r9, r44 * r54);
    write_sum_2<double, double>((double*)inout_shared, r40, r57);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              4 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fma(r44, r23, r19 * r21);
    r40 = fma(r19, r23, r44 * r8);
    write_sum_2<double, double>((double*)inout_shared, r57, r40);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              6 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = fma(r44, r15, r19 * r47);
    r19 = fma(r54, r23, r9 * r21);
    write_sum_2<double, double>((double*)inout_shared, r44, r19);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              8 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fma(r54, r15, r9 * r47);
    r9 = fma(r9, r23, r54 * r8);
    write_sum_2<double, double>((double*)inout_shared, r9, r19);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              10 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fma(r21, r23, r8 * r23);
    r21 = fma(r15, r23, r21 * r47);
    write_sum_2<double, double>((double*)inout_shared, r19, r21);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              12 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r47, r23, r8 * r15);
    write_sum_1<double, double>((double*)inout_shared, r23);
  };
  flush_sum_shared<1, double>(out_pose_precond_tril,
                              14 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = r6 * r43;
    r47 = r7 * r43;
    write_idx_2<1024, double, double, double2>(out_calib_jac,
                                               0 * out_calib_jac_num_alloc,
                                               global_thread_idx,
                                               r23,
                                               r47);
    r15 = r38 * r42;
    r15 = r15 * r29;
    r8 = r0 * r7;
    r8 = r8 * r38;
    r8 = r8 * r42;
    write_idx_2<1024, double, double, double2>(
        out_calib_jac, 2 * out_calib_jac_num_alloc, global_thread_idx, r15, r8);
    r21 = r7 * r4;
    r21 = r21 * r3;
    r19 = r6 * r4;
    r19 = r19 * r2;
    r19 = fma(r43, r19, r43 * r21);
    r21 = r38 * r4;
    r21 = r21 * r2;
    r21 = r21 * r42;
    r43 = r0 * r7;
    r43 = r43 * r38;
    r43 = r43 * r4;
    r43 = r43 * r3;
    r43 = fma(r42, r43, r29 * r21);
    write_sum_2<double, double>((double*)inout_shared, r19, r43);
  };
  flush_sum_shared<2, double>(out_calib_njtr,
                              0 * out_calib_njtr_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r4 * r2;
    r19 = r4 * r3;
    write_sum_2<double, double>((double*)inout_shared, r43, r19);
  };
  flush_sum_shared<2, double>(out_calib_njtr,
                              2 * out_calib_njtr_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = r7 * r7;
    r19 = r19 * r30;
    r19 = r19 * r30;
    r43 = r6 * r6;
    r43 = r43 * r30;
    r43 = r43 * r30;
    r43 = fma(r32, r43, r32 * r19);
    r19 = r0 * r32;
    r21 = r38 * r38;
    r19 = r19 * r21;
    r21 = r7 * r19;
    r9 = r0 * r7;
    r54 = r6 * r29;
    r54 = fma(r19, r54, r9 * r21);
    write_sum_2<double, double>((double*)inout_shared, r43, r54);
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
    r25 = r6 * r38;
    r25 = r25 * r30;
    r25 = r25 * r32;
    r54 = r0 * r7;
    r54 = r54 * r7;
    r54 = r54 * r38;
    r54 = r54 * r30;
    r54 = fma(r32, r54, r29 * r25);
    write_sum_2<double, double>((double*)inout_shared, r54, r23);
  };
  flush_sum_shared<2, double>(out_calib_precond_tril,
                              0 * out_calib_precond_tril_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<double, double>((double*)inout_shared, r47, r15);
  };
  flush_sum_shared<2, double>(out_calib_precond_tril,
                              2 * out_calib_precond_tril_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = 0.00000000000000000e+00;
    write_sum_2<double, double>((double*)inout_shared, r8, r15);
  };
  flush_sum_shared<2, double>(out_calib_precond_tril,
                              4 * out_calib_precond_tril_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = r17 * r28;
    r15 = r15 * r6;
    r15 = fma(r32, r15, r20 * r51);
    r8 = r17 * r16;
    r8 = r8 * r7;
    r15 = fma(r32, r8, r15);
    r15 = fma(r20, r52, r15);
    r8 = r5 * r15;
    r8 = r8 * r42;
    r8 = fma(r28, r1, r29 * r8);
    r47 = r20 * r29;
    r8 = fma(r50, r47, r8);
    r47 = r0 * r15;
    r47 = r47 * r42;
    r47 = fma(r41, r47, r16 * r1);
    r23 = r0 * r20;
    r23 = r23 * r7;
    r47 = fma(r50, r23, r47);
    write_idx_2<1024, double, double, double2>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r8, r47);
    r23 = r17 * r18;
    r23 = r23 * r6;
    r23 = fma(r35, r51, r32 * r23);
    r54 = r17 * r24;
    r54 = r54 * r7;
    r23 = fma(r32, r54, r23);
    r23 = fma(r35, r52, r23);
    r54 = r5 * r23;
    r54 = r54 * r42;
    r54 = fma(r29, r54, r18 * r1);
    r35 = r35 * r50;
    r54 = fma(r29, r35, r54);
    r9 = fma(r24, r1, r35 * r9);
    r35 = r0 * r23;
    r35 = r35 * r42;
    r9 = fma(r41, r35, r9);
    write_idx_2<1024, double, double, double2>(
        out_point_jac, 2 * out_point_jac_num_alloc, global_thread_idx, r54, r9);
    r35 = r17 * r22;
    r35 = r35 * r6;
    r51 = fma(r27, r51, r32 * r35);
    r35 = r17 * r33;
    r35 = r35 * r7;
    r51 = fma(r32, r35, r51);
    r51 = fma(r27, r52, r51);
    r52 = r5 * r51;
    r52 = r52 * r42;
    r35 = r27 * r29;
    r35 = fma(r50, r35, r29 * r52);
    r35 = fma(r22, r1, r35);
    r52 = r0 * r51;
    r52 = r52 * r42;
    r1 = fma(r33, r1, r41 * r52);
    r52 = r0 * r27;
    r52 = r52 * r7;
    r1 = fma(r50, r52, r1);
    write_idx_2<1024, double, double, double2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r35, r1);
    r52 = r4 * r3;
    r50 = r4 * r2;
    r50 = fma(r8, r50, r47 * r52);
    r52 = r4 * r3;
    r41 = r4 * r2;
    r41 = fma(r54, r41, r9 * r52);
    write_sum_2<double, double>((double*)inout_shared, r50, r41);
  };
  flush_sum_shared<2, double>(out_point_njtr,
                              0 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = r4 * r3;
    r50 = r4 * r2;
    r50 = fma(r35, r50, r1 * r41);
    write_sum_1<double, double>((double*)inout_shared, r50);
  };
  flush_sum_shared<1, double>(out_point_njtr,
                              2 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = fma(r8, r8, r47 * r47);
    r41 = fma(r54, r54, r9 * r9);
    write_sum_2<double, double>((double*)inout_shared, r50, r41);
  };
  flush_sum_shared<2, double>(out_point_precond_diag,
                              0 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r1, r1, r35 * r35);
    write_sum_1<double, double>((double*)inout_shared, r41);
  };
  flush_sum_shared<1, double>(out_point_precond_diag,
                              2 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r47, r9, r8 * r54);
    r47 = fma(r47, r1, r8 * r35);
    write_sum_2<double, double>((double*)inout_shared, r41, r47);
  };
  flush_sum_shared<2, double>(out_point_precond_tril,
                              0 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fma(r54, r35, r9 * r1);
    write_sum_1<double, double>((double*)inout_shared, r35);
  };
  flush_sum_shared<1, double>(out_point_precond_tril,
                              2 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
}

void simple_radial_merged_res_jac(double* pose,
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
  simple_radial_merged_res_jac_kernel<<<n_blocks, 1024>>>(
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