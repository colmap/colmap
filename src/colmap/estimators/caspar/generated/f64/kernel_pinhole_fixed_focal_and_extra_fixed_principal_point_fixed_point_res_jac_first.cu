#include "kernel_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac_first_kernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
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
      r46, r47;

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
    r0 = 1.00000000000000008e-15;
  };
  load_shared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r5);
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
  load_shared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r10, r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = r9 * r10;
    r13 = 2.00000000000000000e+00;
    r12 = r12 * r13;
    r14 = r8 * r13;
    r15 = r11 * r14;
    r16 = r12 + r15;
    r5 = fma(r7, r16, r5);
    r17 = r9 * r11;
    r18 = -2.00000000000000000e+00;
    r17 = r17 * r18;
    r19 = r10 * r14;
    r20 = r17 + r19;
    read_idx_1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r21);
    r22 = r8 * r8;
    r22 = r22 * r18;
    r23 = 1.00000000000000000e+00;
    r24 = r9 * r9;
    r25 = fma(r18, r24, r23);
    r26 = r22 + r25;
    r5 = fma(r6, r20, r5);
    r5 = fma(r21, r26, r5);
    r26 = copysign(1.0, r5);
    r26 = fma(r0, r26, r5);
    r0 = 1.0 / r26;
    read_idx_2<1024, double, double, double2>(focal_and_extra,
                                              0 * focal_and_extra_num_alloc,
                                              global_thread_idx,
                                              r5,
                                              r27);
  };
  load_shared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r28, r29);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r30 = r10 * r18;
    r31 = r11 * r30;
    r14 = r9 * r14;
    r32 = r31 + r14;
    r28 = fma(r7, r32, r28);
    r33 = r9 * r11;
    r33 = r33 * r13;
    r19 = r33 + r19;
    r34 = r10 * r30;
    r25 = r34 + r25;
    r28 = fma(r21, r19, r28);
    r28 = fma(r6, r25, r28);
    r25 = r5 * r28;
    r2 = fma(r0, r25, r2);
    r3 = fma(r3, r4, r1);
    r1 = r10 * r11;
    r1 = r1 * r13;
    r14 = r1 + r14;
    r29 = fma(r6, r14, r29);
    r13 = r8 * r11;
    r13 = r13 * r18;
    r12 = r12 + r13;
    r34 = r23 + r34;
    r34 = r34 + r22;
    r29 = fma(r21, r12, r29);
    r29 = fma(r7, r34, r29);
    r34 = r27 * r29;
    r3 = fma(r0, r34, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r22 = fma(r2, r2, r3 * r3);
  };
  sum_store<double>(out_rTr_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r22);
  if (global_thread_idx < problem_size) {
    r22 = r4 * r2;
    r23 = r9 * r30;
    r13 = r13 + r23;
    r35 = r10 * r10;
    r36 = r4 * r24;
    r37 = r35 + r36;
    r11 = r11 * r11;
    r38 = r8 * r8;
    r38 = r38 * r4;
    r39 = r11 + r38;
    r40 = r37 + r39;
    r40 = fma(r7, r40, r21 * r13);
    r13 = r26 * r26;
    r41 = 1.0 / r13;
    r42 = r4 * r41;
    r43 = r42 * r25;
    r9 = r8 * r9;
    r9 = r9 * r18;
    r1 = r1 + r9;
    r19 = fma(r7, r19, r21 * r1);
    r1 = r5 * r19;
    r1 = fma(r0, r1, r40 * r43);
    r18 = r4 * r3;
    r44 = r40 * r42;
    r45 = r8 * r8;
    r46 = r4 * r11;
    r47 = r45 + r46;
    r37 = r37 + r47;
    r37 = fma(r21, r37, r7 * r12);
    r12 = r27 * r37;
    r12 = fma(r0, r12, r34 * r44);
    r18 = fma(r12, r18, r1 * r22);
    r22 = r4 * r3;
    r23 = r15 + r23;
    r23 = fma(r6, r23, r21 * r14);
    r14 = r27 * r23;
    r10 = r10 * r10;
    r10 = r10 * r4;
    r15 = r24 + r10;
    r47 = r47 + r15;
    r47 = fma(r6, r47, r21 * r20);
    r20 = r47 * r42;
    r20 = fma(r34, r20, r0 * r14);
    r14 = r4 * r2;
    r30 = r8 * r30;
    r17 = r17 + r30;
    r11 = r45 + r11;
    r11 = r11 + r36;
    r11 = r11 + r10;
    r11 = fma(r21, r11, r6 * r17);
    r21 = r5 * r11;
    r21 = fma(r47, r43, r0 * r21);
    r14 = fma(r21, r14, r20 * r22);
    write_sum_2<double, double>((double*)inout_shared, r18, r14);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r14 = r4 * r3;
    r30 = r33 + r30;
    r30 = fma(r7, r30, r6 * r16);
    r16 = r30 * r42;
    r31 = r9 + r31;
    r15 = r39 + r15;
    r15 = fma(r6, r15, r7 * r31);
    r31 = r27 * r15;
    r31 = fma(r0, r31, r34 * r16);
    r16 = r4 * r2;
    r35 = r24 + r35;
    r35 = r35 + r38;
    r35 = r35 + r46;
    r35 = fma(r7, r35, r6 * r32);
    r7 = r5 * r35;
    r7 = fma(r0, r7, r30 * r43);
    r16 = fma(r7, r16, r31 * r14);
    r14 = r5 * r4;
    r14 = r14 * r2;
    r14 = r14 * r0;
    write_sum_2<double, double>((double*)inout_shared, r16, r14);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r14 = r27 * r4;
    r14 = r14 * r3;
    r14 = r14 * r0;
    r16 = r2 * r41;
    r32 = r3 * r41;
    r32 = fma(r34, r32, r25 * r16);
    write_sum_2<double, double>((double*)inout_shared, r14, r32);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              4 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r1, r1, r12 * r12);
    r14 = fma(r21, r21, r20 * r20);
    write_sum_2<double, double>((double*)inout_shared, r32, r14);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              0 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r14 = r5 * r5;
    r14 = r14 * r41;
    r32 = fma(r31, r31, r7 * r7);
    write_sum_2<double, double>((double*)inout_shared, r32, r14);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              2 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r14 = r27 * r27;
    r14 = r14 * r41;
    r13 = r26 * r13;
    r26 = r26 * r13;
    r26 = 1.0 / r26;
    r32 = r29 * r26;
    r16 = r27 * r34;
    r6 = r5 * r28;
    r6 = r6 * r26;
    r6 = fma(r25, r6, r16 * r32);
    write_sum_2<double, double>((double*)inout_shared, r14, r6);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              4 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r12, r20, r1 * r21);
    r14 = fma(r1, r7, r12 * r31);
    write_sum_2<double, double>((double*)inout_shared, r6, r14);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              0 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r14 = r5 * r1;
    r14 = r14 * r0;
    r6 = r27 * r12;
    r6 = r6 * r0;
    write_sum_2<double, double>((double*)inout_shared, r14, r6);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              2 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r20, r31, r21 * r7);
    r14 = r12 * r42;
    r1 = fma(r1, r43, r34 * r14);
    write_sum_2<double, double>((double*)inout_shared, r1, r6);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              4 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r5 * r21;
    r6 = r6 * r0;
    r1 = r27 * r20;
    r1 = r1 * r0;
    write_sum_2<double, double>((double*)inout_shared, r6, r1);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              6 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r5 * r7;
    r1 = r1 * r0;
    r6 = r20 * r42;
    r6 = fma(r34, r6, r21 * r43);
    write_sum_2<double, double>((double*)inout_shared, r6, r1);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              8 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r27 * r31;
    r1 = r1 * r0;
    r0 = r31 * r42;
    r43 = fma(r7, r43, r34 * r0);
    write_sum_2<double, double>((double*)inout_shared, r1, r43);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              10 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = 0.00000000000000000e+00;
    r25 = r5 * r25;
    r13 = 1.0 / r13;
    r13 = r4 * r13;
    r25 = r25 * r13;
    write_sum_2<double, double>((double*)inout_shared, r43, r25);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              12 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = r13 * r16;
    write_sum_1<double, double>((double*)inout_shared, r16);
  };
  flush_sum_shared<1, double>(out_pose_precond_tril,
                              14 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  sum_flush_final<double>(out_rTr_local, out_rTr, 1);
}

void pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac_first(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
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
  pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac_first_kernel<<<
      n_blocks,
      1024>>>(pose,
              pose_num_alloc,
              pose_indices,
              pixel,
              pixel_num_alloc,
              focal_and_extra,
              focal_and_extra_num_alloc,
              principal_point,
              principal_point_num_alloc,
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