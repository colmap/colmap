#include "kernel_simple_radial_fixed_pose_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_pose_fixed_principal_point_res_jac_first_kernel(
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        SharedIndex* focal_and_extra_indices,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* pose,
        unsigned int pose_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
        double* out_focal_and_extra_jac,
        unsigned int out_focal_and_extra_jac_num_alloc,
        double* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        double* const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        double* const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
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

  __shared__ SharedIndex focal_and_extra_indices_loc[1024];
  focal_and_extra_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35;

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
    read_idx_2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r0, r5);
  };
  load_shared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = -2.00000000000000000e+00;
    read_idx_2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r9, r10);
    r11 = r9 * r10;
    read_idx_2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r12, r13);
    r14 = 2.00000000000000000e+00;
    r15 = r12 * r14;
    r16 = r13 * r15;
    r17 = fma(r8, r11, r16);
    r0 = fma(r7, r17, r0);
  };
  load_shared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r19 = r13 * r10;
    r20 = r9 * r15;
    r19 = fma(r14, r19, r20);
    r21 = r9 * r9;
    r21 = r21 * r8;
    r22 = 1.00000000000000000e+00;
    r23 = r13 * r13;
    r23 = fma(r8, r23, r22);
    r24 = r21 + r23;
    r0 = fma(r18, r19, r0);
    r0 = fma(r6, r24, r0);
  };
  load_shared<2, double, double>(focal_and_extra,
                                 0 * focal_and_extra_num_alloc,
                                 focal_and_extra_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>((double*)inout_shared,
                          focal_and_extra_indices_loc[threadIdx.x].target,
                          r25,
                          r26);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r27 = r0 * r0;
    r28 = 1.00000000000000008e-15;
    read_idx_1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r29);
    r9 = r13 * r9;
    r9 = r9 * r14;
    r15 = fma(r10, r15, r9);
    r29 = fma(r7, r15, r29);
    r30 = r13 * r10;
    r30 = fma(r8, r30, r20);
    r20 = r12 * r12;
    r20 = r20 * r8;
    r23 = r20 + r23;
    r29 = fma(r6, r30, r29);
    r29 = fma(r18, r23, r29);
    r31 = copysign(1.0, r29);
    r31 = fma(r28, r31, r29);
    r28 = r31 * r31;
    r29 = 1.0 / r28;
    r27 = r27 * r29;
    r11 = fma(r14, r11, r16);
    r6 = fma(r6, r11, r5);
    r5 = r12 * r10;
    r5 = fma(r8, r5, r9);
    r21 = r22 + r21;
    r21 = r21 + r20;
    r6 = fma(r18, r5, r6);
    r6 = fma(r7, r21, r6);
    r7 = r6 * r6;
    r18 = r29 * r7;
    r20 = r27 + r18;
    r22 = fma(r26, r20, r22);
    r9 = r0 * r22;
    r16 = 1.0 / r31;
    r32 = r25 * r16;
    r2 = fma(r32, r9, r2);
    r3 = fma(r3, r4, r1);
    r1 = r6 * r22;
    r3 = fma(r32, r1, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = fma(r2, r2, r3 * r3);
  };
  sum_store<double>(out_rTr_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r1);
  if (global_thread_idx < problem_size) {
    r1 = r0 * r22;
    r1 = r1 * r16;
    r9 = r6 * r22;
    r9 = r9 * r16;
    write_idx_2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        0 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r1,
        r9);
    r9 = r0 * r20;
    r9 = r9 * r32;
    r1 = r6 * r20;
    r1 = r1 * r32;
    write_idx_2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        2 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r9,
        r1);
    r1 = r6 * r3;
    r9 = r22 * r4;
    r1 = r1 * r16;
    r33 = r0 * r2;
    r33 = r33 * r16;
    r33 = fma(r9, r33, r9 * r1);
    r1 = r0 * r20;
    r1 = r1 * r4;
    r1 = r1 * r2;
    r16 = r6 * r20;
    r16 = r16 * r4;
    r16 = r16 * r3;
    r16 = fma(r32, r16, r32 * r1);
    write_sum_2<double, double>((double*)inout_shared, r33, r16);
  };
  flush_sum_shared<2, double>(out_focal_and_extra_njtr,
                              0 * out_focal_and_extra_njtr_num_alloc,
                              focal_and_extra_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = r22 * r22;
    r33 = r22 * r27;
    r16 = fma(r22, r33, r18 * r16);
    r1 = r25 * r20;
    r34 = r25 * r20;
    r1 = r1 * r34;
    r1 = fma(r18, r1, r27 * r1);
    write_sum_2<double, double>((double*)inout_shared, r16, r1);
  };
  flush_sum_shared<2, double>(out_focal_and_extra_precond_diag,
                              0 * out_focal_and_extra_precond_diag_num_alloc,
                              focal_and_extra_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r22 * r18;
    r33 = fma(r34, r33, r34 * r1);
    write_sum_1<double, double>((double*)inout_shared, r33);
  };
  flush_sum_shared<1, double>(out_focal_and_extra_precond_tril,
                              0 * out_focal_and_extra_precond_tril_num_alloc,
                              focal_and_extra_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = r24 * r22;
    r34 = r0 * r29;
    r1 = r30 * r34;
    r9 = r25 * r9;
    r1 = fma(r9, r1, r32 * r33);
    r33 = r0 * r0;
    r28 = r31 * r28;
    r28 = 1.0 / r28;
    r28 = r8 * r28;
    r33 = r33 * r30;
    r8 = r14 * r24;
    r8 = fma(r34, r8, r28 * r33);
    r33 = r14 * r11;
    r33 = r33 * r6;
    r8 = fma(r29, r33, r8);
    r31 = r30 * r7;
    r8 = fma(r28, r31, r8);
    r31 = r0 * r8;
    r26 = r26 * r32;
    r1 = fma(r26, r31, r1);
    r31 = r6 * r8;
    r33 = r6 * r29;
    r33 = r33 * r9;
    r31 = fma(r30, r33, r26 * r31);
    r25 = r11 * r22;
    r31 = fma(r32, r25, r31);
    write_idx_2<1024, double, double, double2>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r1, r31);
    r25 = r15 * r34;
    r16 = r17 * r22;
    r16 = fma(r32, r16, r9 * r25);
    r25 = r14 * r17;
    r27 = r0 * r0;
    r35 = r15 * r28;
    r27 = fma(r35, r27, r34 * r25);
    r25 = r14 * r21;
    r25 = r25 * r6;
    r27 = fma(r29, r25, r27);
    r27 = fma(r7, r35, r27);
    r27 = r27 * r26;
    r16 = fma(r0, r27, r16);
    r25 = r21 * r22;
    r25 = fma(r32, r25, r6 * r27);
    r25 = fma(r15, r33, r25);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               2 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r16,
                                               r25);
    r27 = r23 * r34;
    r35 = r19 * r22;
    r35 = fma(r32, r35, r9 * r27);
    r27 = r14 * r19;
    r9 = r0 * r0;
    r9 = r9 * r23;
    r9 = fma(r28, r9, r34 * r27);
    r27 = r14 * r5;
    r27 = r27 * r6;
    r9 = fma(r29, r27, r9);
    r29 = r23 * r7;
    r9 = fma(r28, r29, r9);
    r29 = r0 * r9;
    r35 = fma(r26, r29, r35);
    r29 = r6 * r9;
    r27 = r5 * r22;
    r27 = fma(r32, r27, r26 * r29);
    r27 = fma(r23, r33, r27);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               4 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r35,
                                               r27);
    r33 = r4 * r3;
    r29 = r4 * r2;
    r29 = fma(r1, r29, r31 * r33);
    r33 = r4 * r3;
    r32 = r4 * r2;
    r32 = fma(r16, r32, r25 * r33);
    write_sum_2<double, double>((double*)inout_shared, r29, r32);
  };
  flush_sum_shared<2, double>(out_point_njtr,
                              0 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r4 * r2;
    r29 = r4 * r3;
    r29 = fma(r27, r29, r35 * r32);
    write_sum_1<double, double>((double*)inout_shared, r29);
  };
  flush_sum_shared<1, double>(out_point_njtr,
                              2 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fma(r31, r31, r1 * r1);
    r32 = fma(r16, r16, r25 * r25);
    write_sum_2<double, double>((double*)inout_shared, r29, r32);
  };
  flush_sum_shared<2, double>(out_point_precond_diag,
                              0 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r27, r27, r35 * r35);
    write_sum_1<double, double>((double*)inout_shared, r32);
  };
  flush_sum_shared<1, double>(out_point_precond_diag,
                              2 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r31, r25, r1 * r16);
    r1 = fma(r1, r35, r31 * r27);
    write_sum_2<double, double>((double*)inout_shared, r32, r1);
  };
  flush_sum_shared<2, double>(out_point_precond_tril,
                              0 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fma(r16, r35, r25 * r27);
    write_sum_1<double, double>((double*)inout_shared, r35);
  };
  flush_sum_shared<1, double>(out_point_precond_tril,
                              2 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  sum_flush_final<double>(out_rTr_local, out_rTr, 1);
}

void simple_radial_fixed_pose_fixed_principal_point_res_jac_first(
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* out_focal_and_extra_jac,
    unsigned int out_focal_and_extra_jac_num_alloc,
    double* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    double* const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    double* const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
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
  simple_radial_fixed_pose_fixed_principal_point_res_jac_first_kernel<<<
      n_blocks,
      1024>>>(focal_and_extra,
              focal_and_extra_num_alloc,
              focal_and_extra_indices,
              point,
              point_num_alloc,
              point_indices,
              pixel,
              pixel_num_alloc,
              pose,
              pose_num_alloc,
              principal_point,
              principal_point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_rTr,
              out_focal_and_extra_jac,
              out_focal_and_extra_jac_num_alloc,
              out_focal_and_extra_njtr,
              out_focal_and_extra_njtr_num_alloc,
              out_focal_and_extra_precond_diag,
              out_focal_and_extra_precond_diag_num_alloc,
              out_focal_and_extra_precond_tril,
              out_focal_and_extra_precond_tril_num_alloc,
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