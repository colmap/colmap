#include "kernel_simple_radial_merged_fixed_pose_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_merged_fixed_pose_res_jac_kernel(
        double* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* pose,
        unsigned int pose_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
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
      r31, r32, r33, r34, r35, r36, r37, r38, r39;
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
  load_shared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r25, r26);
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
    r9 = fma(r26, r20, r22);
    r16 = r0 * r9;
    r32 = 1.0 / r31;
    r33 = r25 * r32;
    r2 = fma(r33, r16, r2);
    r3 = fma(r3, r4, r1);
    r1 = r6 * r9;
    r3 = fma(r33, r1, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = r0 * r9;
    r1 = r1 * r32;
    r16 = r6 * r9;
    r16 = r16 * r32;
    write_idx_2<1024, double, double, double2>(
        out_calib_jac, 0 * out_calib_jac_num_alloc, global_thread_idx, r1, r16);
    r34 = r0 * r20;
    r34 = r34 * r33;
    r35 = r6 * r20;
    r35 = r35 * r33;
    write_idx_2<1024, double, double, double2>(out_calib_jac,
                                               2 * out_calib_jac_num_alloc,
                                               global_thread_idx,
                                               r34,
                                               r35);
    r36 = r6 * r3;
    r37 = r9 * r4;
    r36 = r36 * r32;
    r38 = r0 * r2;
    r38 = r38 * r32;
    r38 = fma(r37, r38, r37 * r36);
    r36 = r0 * r20;
    r36 = r36 * r4;
    r36 = r36 * r2;
    r32 = r6 * r20;
    r32 = r32 * r4;
    r32 = r32 * r3;
    r32 = fma(r33, r32, r33 * r36);
    write_sum_2<double, double>((double*)inout_shared, r38, r32);
  };
  flush_sum_shared<2, double>(out_calib_njtr,
                              0 * out_calib_njtr_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r4 * r2;
    r38 = r4 * r3;
    write_sum_2<double, double>((double*)inout_shared, r32, r38);
  };
  flush_sum_shared<2, double>(out_calib_njtr,
                              2 * out_calib_njtr_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = r9 * r9;
    r32 = r9 * r27;
    r38 = fma(r9, r32, r18 * r38);
    r36 = r25 * r20;
    r39 = r25 * r20;
    r36 = r36 * r39;
    r36 = fma(r27, r36, r18 * r36);
    write_sum_2<double, double>((double*)inout_shared, r38, r36);
  };
  flush_sum_shared<2, double>(out_calib_precond_diag,
                              0 * out_calib_precond_diag_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<double, double>((double*)inout_shared, r22, r22);
  };
  flush_sum_shared<2, double>(out_calib_precond_diag,
                              2 * out_calib_precond_diag_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = r9 * r18;
    r22 = fma(r39, r22, r39 * r32);
    write_sum_2<double, double>((double*)inout_shared, r22, r1);
  };
  flush_sum_shared<2, double>(out_calib_precond_tril,
                              0 * out_calib_precond_tril_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<double, double>((double*)inout_shared, r16, r34);
  };
  flush_sum_shared<2, double>(out_calib_precond_tril,
                              2 * out_calib_precond_tril_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = 0.00000000000000000e+00;
    write_sum_2<double, double>((double*)inout_shared, r35, r34);
  };
  flush_sum_shared<2, double>(out_calib_precond_tril,
                              4 * out_calib_precond_tril_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r0 * r0;
    r28 = r31 * r28;
    r28 = 1.0 / r28;
    r28 = r8 * r28;
    r34 = r34 * r30;
    r8 = r14 * r24;
    r31 = r0 * r29;
    r8 = fma(r31, r8, r28 * r34);
    r34 = r14 * r11;
    r34 = r34 * r6;
    r8 = fma(r29, r34, r8);
    r35 = r30 * r7;
    r8 = fma(r28, r35, r8);
    r35 = r0 * r8;
    r26 = r26 * r33;
    r34 = r24 * r9;
    r34 = fma(r33, r34, r26 * r35);
    r35 = r30 * r31;
    r37 = r25 * r37;
    r34 = fma(r37, r35, r34);
    r35 = r11 * r9;
    r25 = r6 * r8;
    r25 = fma(r26, r25, r33 * r35);
    r35 = r6 * r29;
    r35 = r35 * r37;
    r25 = fma(r30, r35, r25);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               0 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r34,
                                               r25);
    r16 = r17 * r9;
    r1 = r14 * r17;
    r22 = r0 * r0;
    r32 = r15 * r28;
    r22 = fma(r32, r22, r31 * r1);
    r1 = r14 * r21;
    r1 = r1 * r6;
    r22 = fma(r29, r1, r22);
    r22 = fma(r7, r32, r22);
    r22 = r22 * r26;
    r16 = fma(r0, r22, r33 * r16);
    r1 = r15 * r31;
    r16 = fma(r37, r1, r16);
    r1 = r21 * r9;
    r1 = fma(r33, r1, r15 * r35);
    r1 = fma(r6, r22, r1);
    write_idx_2<1024, double, double, double2>(
        out_point_jac, 2 * out_point_jac_num_alloc, global_thread_idx, r16, r1);
    r22 = r14 * r19;
    r32 = r0 * r0;
    r32 = r32 * r23;
    r32 = fma(r28, r32, r31 * r22);
    r22 = r14 * r5;
    r22 = r22 * r6;
    r32 = fma(r29, r22, r32);
    r29 = r23 * r7;
    r32 = fma(r28, r29, r32);
    r29 = r0 * r32;
    r22 = r23 * r31;
    r22 = fma(r37, r22, r26 * r29);
    r29 = r19 * r9;
    r22 = fma(r33, r29, r22);
    r29 = r6 * r32;
    r37 = r5 * r9;
    r37 = fma(r33, r37, r26 * r29);
    r37 = fma(r23, r35, r37);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               4 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r22,
                                               r37);
    r35 = r4 * r3;
    r29 = r4 * r2;
    r29 = fma(r34, r29, r25 * r35);
    r35 = r4 * r3;
    r33 = r4 * r2;
    r33 = fma(r16, r33, r1 * r35);
    write_sum_2<double, double>((double*)inout_shared, r29, r33);
  };
  flush_sum_shared<2, double>(out_point_njtr,
                              0 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = r4 * r3;
    r29 = r4 * r2;
    r29 = fma(r22, r29, r37 * r33);
    write_sum_1<double, double>((double*)inout_shared, r29);
  };
  flush_sum_shared<1, double>(out_point_njtr,
                              2 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fma(r34, r34, r25 * r25);
    r33 = fma(r16, r16, r1 * r1);
    write_sum_2<double, double>((double*)inout_shared, r29, r33);
  };
  flush_sum_shared<2, double>(out_point_precond_diag,
                              0 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fma(r37, r37, r22 * r22);
    write_sum_1<double, double>((double*)inout_shared, r33);
  };
  flush_sum_shared<1, double>(out_point_precond_diag,
                              2 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fma(r25, r1, r34 * r16);
    r25 = fma(r25, r37, r34 * r22);
    write_sum_2<double, double>((double*)inout_shared, r33, r25);
  };
  flush_sum_shared<2, double>(out_point_precond_tril,
                              0 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = fma(r16, r22, r1 * r37);
    write_sum_1<double, double>((double*)inout_shared, r22);
  };
  flush_sum_shared<1, double>(out_point_precond_tril,
                              2 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
}

void simple_radial_merged_fixed_pose_res_jac(
    double* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
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
  simple_radial_merged_fixed_pose_res_jac_kernel<<<n_blocks, 1024>>>(
      calib,
      calib_num_alloc,
      calib_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      out_res,
      out_res_num_alloc,
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