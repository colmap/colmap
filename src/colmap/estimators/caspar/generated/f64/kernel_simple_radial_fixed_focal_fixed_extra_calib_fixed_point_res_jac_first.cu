#include "kernel_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first_kernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* focal,
        unsigned int focal_num_alloc,
        double* extra_calib,
        unsigned int extra_calib_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50;

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
    read_idx_1<1024, double, double, double>(
        extra_calib, 2 * extra_calib_num_alloc, global_thread_idx, r28);
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
    r23 = fma(r28, r23, r24);
    r24 = 1.0 / r26;
    r29 = r23 * r24;
    r2 = fma(r27, r29, r2);
    r3 = fma(r3, r4, r1);
    r1 = r0 * r23;
    r1 = r1 * r24;
    r3 = fma(r6, r1, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r29 = fma(r3, r3, r2 * r2);
  };
  sum_store<double>(out_rTr_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r29);
  if (global_thread_idx < problem_size) {
    r29 = r4 * r2;
    r37 = r10 * r16;
    r39 = r39 + r37;
    r40 = r13 * r13;
    r41 = r4 * r25;
    r42 = r40 + r41;
    r14 = r14 * r14;
    r43 = r4 * r36;
    r44 = r14 + r43;
    r45 = r42 + r44;
    r45 = fma(r8, r45, r19 * r39);
    r39 = r45 * r27;
    r46 = r23 * r4;
    r46 = r46 * r31;
    r10 = r9 * r10;
    r10 = r10 * r15;
    r38 = r38 + r10;
    r22 = fma(r8, r22, r19 * r38);
    r39 = fma(r22, r1, r46 * r39);
    r38 = r15 * r5;
    r30 = r26 * r30;
    r26 = 1.0 / r30;
    r38 = r38 * r5;
    r38 = r38 * r45;
    r47 = r15 * r6;
    r47 = r47 * r6;
    r47 = r47 * r45;
    r47 = fma(r26, r47, r26 * r38);
    r38 = r12 * r6;
    r48 = r4 * r14;
    r49 = r36 + r48;
    r42 = r42 + r49;
    r42 = fma(r19, r42, r8 * r32);
    r38 = r38 * r42;
    r47 = fma(r31, r38, r47);
    r32 = r12 * r5;
    r32 = r32 * r22;
    r47 = fma(r31, r32, r47);
    r32 = r28 * r47;
    r32 = r32 * r24;
    r39 = fma(r27, r32, r39);
    r32 = r4 * r3;
    r38 = r0 * r47;
    r22 = r28 * r6;
    r38 = r38 * r24;
    r42 = fma(r42, r1, r22 * r38);
    r38 = r0 * r31;
    r38 = r38 * r6;
    r38 = r38 * r23;
    r38 = r38 * r4;
    r42 = fma(r45, r38, r42);
    r32 = fma(r42, r32, r39 * r29);
    r29 = r4 * r2;
    r23 = r15 * r6;
    r13 = r13 * r13;
    r13 = r13 * r4;
    r50 = r25 + r13;
    r49 = r49 + r50;
    r49 = fma(r7, r49, r19 * r20);
    r23 = r23 * r6;
    r23 = r23 * r49;
    r20 = r12 * r6;
    r37 = r33 + r37;
    r37 = fma(r7, r37, r19 * r11);
    r20 = r20 * r37;
    r20 = fma(r31, r20, r26 * r23);
    r23 = r15 * r5;
    r23 = r23 * r5;
    r23 = r23 * r49;
    r20 = fma(r26, r23, r20);
    r11 = r12 * r5;
    r16 = r9 * r16;
    r35 = r35 + r16;
    r14 = r36 + r14;
    r14 = r14 + r41;
    r14 = r14 + r13;
    r14 = fma(r19, r14, r7 * r35);
    r11 = r11 * r14;
    r20 = fma(r31, r11, r20);
    r11 = r28 * r20;
    r11 = r11 * r24;
    r23 = r49 * r27;
    r23 = fma(r46, r23, r27 * r11);
    r23 = fma(r14, r1, r23);
    r14 = r4 * r3;
    r37 = fma(r37, r1, r49 * r38);
    r11 = r0 * r20;
    r11 = r11 * r24;
    r37 = fma(r22, r11, r37);
    r14 = fma(r37, r14, r23 * r29);
    write_sum_2<double, double>((double*)inout_shared, r32, r14);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r14 = r4 * r3;
    r17 = r10 + r17;
    r50 = r44 + r50;
    r50 = fma(r7, r50, r8 * r17);
    r16 = r21 + r16;
    r16 = fma(r8, r16, r7 * r34);
    r34 = fma(r16, r38, r50 * r1);
    r21 = r12 * r6;
    r21 = r21 * r50;
    r50 = r15 * r5;
    r50 = r50 * r5;
    r50 = r50 * r16;
    r50 = fma(r26, r50, r31 * r21);
    r21 = r15 * r6;
    r21 = r21 * r6;
    r21 = r21 * r16;
    r50 = fma(r26, r21, r50);
    r17 = r12 * r5;
    r40 = r25 + r40;
    r40 = r40 + r43;
    r40 = r40 + r48;
    r40 = fma(r8, r40, r7 * r18);
    r17 = r17 * r40;
    r50 = fma(r31, r17, r50);
    r17 = r0 * r50;
    r17 = r17 * r24;
    r34 = fma(r22, r17, r34);
    r17 = r4 * r2;
    r21 = r16 * r27;
    r40 = fma(r40, r1, r46 * r21);
    r21 = r28 * r50;
    r21 = r21 * r24;
    r40 = fma(r27, r21, r40);
    r17 = fma(r40, r17, r34 * r14);
    r14 = r4 * r2;
    r21 = r26 * r27;
    r31 = r12 * r21;
    r8 = r28 * r5;
    r31 = fma(r8, r31, r1);
    r18 = r15 * r3;
    r7 = r22 * r21;
    r18 = fma(r7, r18, r31 * r14);
    write_sum_2<double, double>((double*)inout_shared, r17, r18);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = r4 * r3;
    r17 = r15 * r5;
    r17 = r17 * r5;
    r14 = r15 * r6;
    r14 = r14 * r6;
    r14 = fma(r26, r14, r26 * r17);
    r17 = r0 * r14;
    r17 = r17 * r24;
    r17 = fma(r22, r17, r38);
    r38 = r4 * r2;
    r48 = r28 * r14;
    r48 = r48 * r24;
    r48 = fma(r27, r48, r27 * r46);
    r38 = fma(r48, r38, r17 * r18);
    r18 = r4 * r3;
    r46 = r0 * r12;
    r46 = r46 * r6;
    r46 = r46 * r26;
    r46 = fma(r22, r46, r1);
    r1 = r15 * r2;
    r1 = fma(r7, r1, r46 * r18);
    write_sum_2<double, double>((double*)inout_shared, r1, r38);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              4 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = fma(r39, r39, r42 * r42);
    r1 = fma(r23, r23, r37 * r37);
    write_sum_2<double, double>((double*)inout_shared, r38, r1);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              0 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fma(r34, r34, r40 * r40);
    r38 = r0 * r6;
    r18 = 4.00000000000000000e+00;
    r30 = r30 * r30;
    r30 = 1.0 / r30;
    r38 = r38 * r18;
    r38 = r38 * r30;
    r38 = r38 * r27;
    r38 = r38 * r22;
    r38 = r38 * r8;
    r8 = fma(r31, r31, r38);
    write_sum_2<double, double>((double*)inout_shared, r1, r8);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              2 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = fma(r46, r46, r38);
    r8 = fma(r17, r17, r48 * r48);
    write_sum_2<double, double>((double*)inout_shared, r38, r8);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              4 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = fma(r39, r23, r42 * r37);
    r38 = fma(r42, r34, r39 * r40);
    write_sum_2<double, double>((double*)inout_shared, r8, r38);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              0 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r12 * r7;
    r38 = fma(r42, r7, r39 * r31);
    r8 = fma(r39, r7, r42 * r46);
    write_sum_2<double, double>((double*)inout_shared, r38, r8);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              2 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r39, r48, r42 * r17);
    r42 = fma(r37, r34, r23 * r40);
    write_sum_2<double, double>((double*)inout_shared, r39, r42);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              4 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = fma(r37, r7, r23 * r31);
    r39 = fma(r23, r7, r37 * r46);
    write_sum_2<double, double>((double*)inout_shared, r42, r39);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              6 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fma(r37, r17, r23 * r48);
    r23 = fma(r34, r7, r40 * r31);
    write_sum_2<double, double>((double*)inout_shared, r37, r23);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              8 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r34, r17, r40 * r48);
    r40 = fma(r40, r7, r34 * r46);
    write_sum_2<double, double>((double*)inout_shared, r40, r23);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              10 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r46, r7, r31 * r7);
    r31 = fma(r17, r7, r31 * r48);
    write_sum_2<double, double>((double*)inout_shared, r23, r31);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              12 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = fma(r48, r7, r46 * r17);
    write_sum_1<double, double>((double*)inout_shared, r7);
  };
  flush_sum_shared<1, double>(out_pose_precond_tril,
                              14 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  sum_flush_final<double>(out_rTr_local, out_rTr, 1);
}

void simple_radial_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* focal,
    unsigned int focal_num_alloc,
    double* extra_calib,
    unsigned int extra_calib_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first_kernel<<<
      n_blocks,
      1024>>>(pose,
              pose_num_alloc,
              pose_indices,
              pixel,
              pixel_num_alloc,
              focal,
              focal_num_alloc,
              extra_calib,
              extra_calib_num_alloc,
              point,
              point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_rTr,
              out_pose_njtr,
              out_pose_njtr_num_alloc,
              out_pose_precond_diag,
              out_pose_precond_diag_num_alloc,
              out_pose_precond_tril,
              out_pose_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar