#include "kernel_simple_radial_fixed_focal_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_focal_fixed_point_res_jac_kernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* extra_calib,
        unsigned int extra_calib_num_alloc,
        SharedIndex* extra_calib_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* focal,
        unsigned int focal_num_alloc,
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
        double* out_extra_calib_jac,
        unsigned int out_extra_calib_jac_num_alloc,
        double* const out_extra_calib_njtr,
        unsigned int out_extra_calib_njtr_num_alloc,
        double* const out_extra_calib_precond_diag,
        unsigned int out_extra_calib_precond_diag_num_alloc,
        double* const out_extra_calib_precond_tril,
        unsigned int out_extra_calib_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex extra_calib_indices_loc[1024];
  extra_calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? extra_calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50;
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
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r7, r8);
  };
  load_shared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r9, r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r11 = r9 * r10;
    r12 = 2.00000000000000000e+00;
    r11 = r11 * r12;
  };
  load_shared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r13, r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = -2.00000000000000000e+00;
    r16 = r13 * r15;
    r17 = r14 * r16;
    r18 = r11 + r17;
    r5 = fma(r8, r18, r5);
    read_idx_1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r19);
    r20 = r9 * r13;
    r20 = r20 * r12;
    r21 = r10 * r14;
    r21 = r21 * r12;
    r22 = r20 + r21;
    r23 = r13 * r16;
    r24 = 1.00000000000000000e+00;
    r25 = r10 * r10;
    r26 = fma(r15, r25, r24);
    r27 = r23 + r26;
    r5 = fma(r19, r22, r5);
    r5 = fma(r7, r27, r5);
    r27 = r0 * r5;
  };
  load_shared<1, double, double>(extra_calib,
                                 2 * extra_calib_num_alloc,
                                 extra_calib_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared,
                          extra_calib_indices_loc[threadIdx.x].target,
                          r28);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r29 = r5 * r5;
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
    r32 = r10 * r13;
    r32 = r32 * r12;
    r33 = r9 * r14;
    r33 = r33 * r12;
    r34 = r32 + r33;
    r31 = fma(r8, r34, r31);
    r35 = r10 * r14;
    r35 = r35 * r15;
    r20 = r20 + r35;
    r36 = r9 * r9;
    r37 = r15 * r36;
    r26 = r37 + r26;
    r31 = fma(r7, r20, r31);
    r31 = fma(r19, r26, r31);
    r26 = copysign(1.0, r31);
    r26 = fma(r30, r26, r31);
    r30 = r26 * r26;
    r31 = 1.0 / r30;
    r38 = r13 * r14;
    r38 = r38 * r12;
    r11 = r11 + r38;
    r6 = fma(r7, r11, r6);
    r39 = r9 * r14;
    r39 = r39 * r15;
    r32 = r32 + r39;
    r23 = r24 + r23;
    r23 = r23 + r37;
    r6 = fma(r19, r32, r6);
    r6 = fma(r8, r23, r6);
    r23 = r6 * r6;
    r23 = fma(r31, r23, r31 * r29);
    r29 = fma(r28, r23, r24);
    r37 = 1.0 / r26;
    r40 = r29 * r37;
    r2 = fma(r27, r40, r2);
    r3 = fma(r3, r4, r1);
    r1 = r0 * r29;
    r1 = r1 * r37;
    r3 = fma(r6, r1, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r40 = r10 * r16;
    r39 = r39 + r40;
    r41 = r13 * r13;
    r42 = r4 * r25;
    r43 = r41 + r42;
    r14 = r14 * r14;
    r44 = r4 * r36;
    r45 = r14 + r44;
    r46 = r43 + r45;
    r46 = fma(r8, r46, r19 * r39);
    r39 = r46 * r27;
    r47 = r29 * r4;
    r47 = r47 * r31;
    r10 = r9 * r10;
    r10 = r10 * r15;
    r38 = r38 + r10;
    r22 = fma(r8, r22, r19 * r38);
    r39 = fma(r22, r1, r47 * r39);
    r38 = r15 * r5;
    r30 = r26 * r30;
    r26 = 1.0 / r30;
    r38 = r38 * r5;
    r38 = r38 * r46;
    r48 = r15 * r6;
    r48 = r48 * r6;
    r48 = r48 * r46;
    r48 = fma(r26, r48, r26 * r38);
    r38 = r12 * r6;
    r49 = r4 * r14;
    r50 = r36 + r49;
    r43 = r43 + r50;
    r43 = fma(r19, r43, r8 * r32);
    r38 = r38 * r43;
    r48 = fma(r31, r38, r48);
    r32 = r12 * r5;
    r32 = r32 * r22;
    r48 = fma(r31, r32, r48);
    r32 = r28 * r48;
    r32 = r32 * r37;
    r39 = fma(r27, r32, r39);
    r32 = r0 * r48;
    r38 = r28 * r6;
    r32 = r32 * r37;
    r43 = fma(r43, r1, r38 * r32);
    r32 = r0 * r31;
    r32 = r32 * r6;
    r32 = r32 * r29;
    r32 = r32 * r4;
    r43 = fma(r46, r32, r43);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r39, r43);
    r29 = r15 * r6;
    r13 = r13 * r13;
    r13 = r13 * r4;
    r22 = r25 + r13;
    r50 = r50 + r22;
    r50 = fma(r7, r50, r19 * r20);
    r29 = r29 * r6;
    r29 = r29 * r50;
    r20 = r12 * r6;
    r40 = r33 + r40;
    r40 = fma(r7, r40, r19 * r11);
    r20 = r20 * r40;
    r20 = fma(r31, r20, r26 * r29);
    r29 = r15 * r5;
    r29 = r29 * r5;
    r29 = r29 * r50;
    r20 = fma(r26, r29, r20);
    r11 = r12 * r5;
    r16 = r9 * r16;
    r35 = r35 + r16;
    r14 = r36 + r14;
    r14 = r14 + r42;
    r14 = r14 + r13;
    r14 = fma(r19, r14, r7 * r35);
    r11 = r11 * r14;
    r20 = fma(r31, r11, r20);
    r11 = r28 * r20;
    r11 = r11 * r37;
    r29 = r50 * r27;
    r29 = fma(r47, r29, r27 * r11);
    r29 = fma(r14, r1, r29);
    r40 = fma(r40, r1, r50 * r32);
    r14 = r0 * r20;
    r14 = r14 * r37;
    r40 = fma(r38, r14, r40);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r29, r40);
    r16 = r21 + r16;
    r16 = fma(r8, r16, r7 * r34);
    r34 = r16 * r27;
    r41 = r25 + r41;
    r41 = r41 + r44;
    r41 = r41 + r49;
    r41 = fma(r8, r41, r7 * r18);
    r34 = fma(r41, r1, r47 * r34);
    r18 = r12 * r6;
    r17 = r10 + r17;
    r22 = r45 + r22;
    r22 = fma(r7, r22, r8 * r17);
    r18 = r18 * r22;
    r7 = r15 * r5;
    r7 = r7 * r5;
    r7 = r7 * r16;
    r7 = fma(r26, r7, r31 * r18);
    r18 = r15 * r6;
    r18 = r18 * r6;
    r18 = r18 * r16;
    r7 = fma(r26, r18, r7);
    r17 = r12 * r5;
    r17 = r17 * r41;
    r7 = fma(r31, r17, r7);
    r17 = r28 * r7;
    r17 = r17 * r37;
    r34 = fma(r27, r17, r34);
    r22 = fma(r16, r32, r22 * r1);
    r17 = r0 * r7;
    r17 = r17 * r37;
    r22 = fma(r38, r17, r22);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r34, r22);
    r17 = r28 * r0;
    r17 = r17 * r12;
    r17 = r17 * r5;
    r17 = r17 * r6;
    r17 = r17 * r26;
    r18 = r26 * r27;
    r41 = r12 * r18;
    r8 = r28 * r5;
    r41 = fma(r8, r41, r1);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r41, r17);
    r45 = r0 * r12;
    r45 = r45 * r6;
    r45 = r45 * r26;
    r45 = fma(r38, r45, r1);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r17, r45);
    r1 = r15 * r5;
    r1 = r1 * r5;
    r10 = r15 * r6;
    r10 = r10 * r6;
    r10 = fma(r26, r10, r26 * r1);
    r1 = r28 * r10;
    r1 = r1 * r37;
    r1 = fma(r27, r1, r27 * r47);
    r47 = r0 * r10;
    r47 = r47 * r37;
    r47 = fma(r38, r47, r32);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r1, r47);
    r32 = r4 * r2;
    r26 = r4 * r3;
    r26 = fma(r43, r26, r39 * r32);
    r32 = r4 * r2;
    r49 = r4 * r3;
    r49 = fma(r40, r49, r29 * r32);
    write_sum_2<double, double>((double*)inout_shared, r26, r49);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r4 * r3;
    r26 = r4 * r2;
    r26 = fma(r34, r26, r22 * r49);
    r49 = r4 * r2;
    r32 = r15 * r3;
    r44 = r38 * r18;
    r32 = fma(r44, r32, r41 * r49);
    write_sum_2<double, double>((double*)inout_shared, r26, r32);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r4 * r3;
    r26 = r4 * r2;
    r26 = fma(r1, r26, r47 * r32);
    r32 = r4 * r3;
    r49 = r15 * r2;
    r49 = fma(r44, r49, r45 * r32);
    write_sum_2<double, double>((double*)inout_shared, r49, r26);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              4 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fma(r39, r39, r43 * r43);
    r49 = fma(r29, r29, r40 * r40);
    write_sum_2<double, double>((double*)inout_shared, r26, r49);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              0 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = fma(r22, r22, r34 * r34);
    r26 = 4.00000000000000000e+00;
    r30 = r30 * r30;
    r30 = 1.0 / r30;
    r30 = r26 * r30;
    r26 = r0 * r6;
    r30 = r30 * r27;
    r30 = r30 * r38;
    r30 = r30 * r8;
    r30 = r30 * r26;
    r8 = fma(r41, r41, r30);
    write_sum_2<double, double>((double*)inout_shared, r49, r8);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              2 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r45, r45, r30);
    r8 = fma(r47, r47, r1 * r1);
    write_sum_2<double, double>((double*)inout_shared, r30, r8);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              4 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = fma(r39, r29, r43 * r40);
    r30 = fma(r43, r22, r39 * r34);
    write_sum_2<double, double>((double*)inout_shared, r8, r30);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              0 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r43, r17, r39 * r41);
    r8 = fma(r39, r17, r43 * r45);
    write_sum_2<double, double>((double*)inout_shared, r30, r8);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              2 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r39, r1, r43 * r47);
    r43 = fma(r40, r22, r29 * r34);
    write_sum_2<double, double>((double*)inout_shared, r39, r43);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              4 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = fma(r40, r17, r29 * r41);
    r39 = fma(r29, r17, r40 * r45);
    write_sum_2<double, double>((double*)inout_shared, r43, r39);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              6 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fma(r40, r47, r29 * r1);
    r29 = fma(r22, r17, r34 * r41);
    write_sum_2<double, double>((double*)inout_shared, r40, r29);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              8 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fma(r22, r47, r34 * r1);
    r34 = fma(r34, r17, r22 * r45);
    write_sum_2<double, double>((double*)inout_shared, r34, r29);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              10 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fma(r45, r17, r41 * r17);
    r41 = fma(r47, r17, r41 * r1);
    write_sum_2<double, double>((double*)inout_shared, r29, r41);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              12 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = fma(r1, r17, r45 * r47);
    write_sum_1<double, double>((double*)inout_shared, r17);
  };
  flush_sum_shared<1, double>(out_pose_precond_tril,
                              14 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = r23 * r37;
    r17 = r17 * r27;
    r1 = r0 * r6;
    r1 = r1 * r23;
    r1 = r1 * r37;
    write_idx_2<1024, double, double, double2>(
        out_extra_calib_jac,
        0 * out_extra_calib_jac_num_alloc,
        global_thread_idx,
        r17,
        r1);
    r47 = r4 * r2;
    r45 = r4 * r3;
    write_sum_2<double, double>((double*)inout_shared, r47, r45);
  };
  flush_sum_shared<2, double>(out_extra_calib_njtr,
                              0 * out_extra_calib_njtr_num_alloc,
                              extra_calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = r0 * r6;
    r45 = r45 * r23;
    r45 = r45 * r4;
    r45 = r45 * r3;
    r47 = r23 * r4;
    r47 = r47 * r2;
    r47 = r47 * r37;
    r47 = fma(r27, r47, r37 * r45);
    write_sum_1<double, double>((double*)inout_shared, r47);
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
    r24 = r5 * r27;
    r31 = r0 * r31;
    r47 = r23 * r23;
    r31 = r31 * r47;
    r47 = r6 * r31;
    r47 = fma(r26, r47, r31 * r24);
    write_sum_1<double, double>((double*)inout_shared, r47);
  };
  flush_sum_shared<1, double>(out_extra_calib_precond_diag,
                              2 * out_extra_calib_precond_diag_num_alloc,
                              extra_calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = 0.00000000000000000e+00;
    write_sum_2<double, double>((double*)inout_shared, r47, r17);
  };
  flush_sum_shared<2, double>(out_extra_calib_precond_tril,
                              0 * out_extra_calib_precond_tril_num_alloc,
                              extra_calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_1<double, double>((double*)inout_shared, r1);
  };
  flush_sum_shared<1, double>(out_extra_calib_precond_tril,
                              2 * out_extra_calib_precond_tril_num_alloc,
                              extra_calib_indices_loc,
                              (double*)inout_shared);
}

void simple_radial_fixed_focal_fixed_point_res_jac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* extra_calib,
    unsigned int extra_calib_num_alloc,
    SharedIndex* extra_calib_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* focal,
    unsigned int focal_num_alloc,
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
    double* out_extra_calib_jac,
    unsigned int out_extra_calib_jac_num_alloc,
    double* const out_extra_calib_njtr,
    unsigned int out_extra_calib_njtr_num_alloc,
    double* const out_extra_calib_precond_diag,
    unsigned int out_extra_calib_precond_diag_num_alloc,
    double* const out_extra_calib_precond_tril,
    unsigned int out_extra_calib_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_focal_fixed_point_res_jac_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      extra_calib,
      extra_calib_num_alloc,
      extra_calib_indices,
      pixel,
      pixel_num_alloc,
      focal,
      focal_num_alloc,
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
      out_extra_calib_jac,
      out_extra_calib_jac_num_alloc,
      out_extra_calib_njtr,
      out_extra_calib_njtr_num_alloc,
      out_extra_calib_precond_diag,
      out_extra_calib_precond_diag_num_alloc,
      out_extra_calib_precond_tril,
      out_extra_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar