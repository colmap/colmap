#include "kernel_simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac_kernel(
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* pose,
        unsigned int pose_num_alloc,
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        double* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        double* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37;

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
    r25 = 1.00000000000000008e-15;
    read_idx_1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r26);
    r9 = r13 * r9;
    r9 = r9 * r14;
    r15 = fma(r10, r15, r9);
    r26 = fma(r7, r15, r26);
    r27 = r13 * r10;
    r27 = fma(r8, r27, r20);
    r20 = r12 * r12;
    r20 = r20 * r8;
    r23 = r20 + r23;
    r26 = fma(r6, r27, r26);
    r26 = fma(r18, r23, r26);
    r28 = copysign(1.0, r26);
    r28 = fma(r25, r28, r26);
    r25 = 1.0 / r28;
    read_idx_2<1024, double, double, double2>(focal_and_extra,
                                              0 * focal_and_extra_num_alloc,
                                              global_thread_idx,
                                              r26,
                                              r29);
    r30 = r28 * r28;
    r31 = 1.0 / r30;
    r32 = r0 * r31;
    r11 = fma(r14, r11, r16);
    r6 = fma(r6, r11, r5);
    r5 = r12 * r10;
    r5 = fma(r8, r5, r9);
    r21 = r22 + r21;
    r21 = r21 + r20;
    r6 = fma(r18, r5, r6);
    r6 = fma(r7, r21, r6);
    r7 = r6 * r6;
    r18 = fma(r31, r7, r0 * r32);
    r18 = fma(r29, r18, r22);
    r18 = r26 * r18;
    r22 = r25 * r18;
    r2 = fma(r0, r22, r2);
    r3 = fma(r3, r4, r1);
    r3 = fma(r6, r22, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = r4 * r3;
    r20 = r26 * r29;
    r9 = r0 * r0;
    r30 = r28 * r30;
    r30 = 1.0 / r30;
    r30 = r8 * r30;
    r9 = r9 * r30;
    r8 = r14 * r24;
    r8 = fma(r32, r8, r27 * r9);
    r28 = r14 * r11;
    r28 = r28 * r6;
    r8 = fma(r31, r28, r8);
    r16 = r27 * r30;
    r8 = fma(r7, r16, r8);
    r20 = r20 * r8;
    r20 = r20 * r25;
    r8 = r27 * r6;
    r8 = r8 * r4;
    r8 = r8 * r31;
    r8 = fma(r18, r8, r6 * r20);
    r8 = fma(r11, r22, r8);
    r16 = r4 * r2;
    r28 = r4 * r18;
    r28 = r28 * r32;
    r33 = fma(r27, r28, r24 * r22);
    r33 = fma(r0, r20, r33);
    r16 = fma(r33, r16, r8 * r1);
    r1 = r4 * r3;
    r20 = r26 * r29;
    r34 = r14 * r17;
    r34 = fma(r15, r9, r32 * r34);
    r35 = r15 * r30;
    r34 = fma(r7, r35, r34);
    r36 = r14 * r21;
    r36 = r36 * r6;
    r34 = fma(r31, r36, r34);
    r20 = r20 * r6;
    r20 = r20 * r34;
    r20 = fma(r21, r22, r25 * r20);
    r36 = r15 * r6;
    r36 = r36 * r4;
    r36 = r36 * r31;
    r20 = fma(r18, r36, r20);
    r36 = r4 * r2;
    r35 = fma(r17, r22, r15 * r28);
    r37 = r26 * r29;
    r37 = r37 * r0;
    r37 = r37 * r34;
    r35 = fma(r25, r37, r35);
    r36 = fma(r35, r36, r20 * r1);
    write_sum_2<double, double>((double*)inout_shared, r16, r36);
  };
  flush_sum_shared<2, double>(out_point_njtr,
                              0 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = r4 * r2;
    r28 = fma(r19, r22, r23 * r28);
    r16 = r26 * r29;
    r1 = r14 * r19;
    r9 = fma(r23, r9, r32 * r1);
    r1 = r14 * r5;
    r1 = r1 * r6;
    r9 = fma(r31, r1, r9);
    r32 = r23 * r30;
    r9 = fma(r7, r32, r9);
    r16 = r16 * r0;
    r16 = r16 * r9;
    r28 = fma(r25, r16, r28);
    r16 = r4 * r3;
    r0 = r26 * r29;
    r0 = r0 * r6;
    r0 = r0 * r9;
    r22 = fma(r5, r22, r25 * r0);
    r0 = r23 * r6;
    r0 = r0 * r4;
    r0 = r0 * r31;
    r22 = fma(r18, r0, r22);
    r16 = fma(r22, r16, r28 * r36);
    write_sum_1<double, double>((double*)inout_shared, r16);
  };
  flush_sum_shared<1, double>(out_point_njtr,
                              2 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = fma(r8, r8, r33 * r33);
    r36 = fma(r35, r35, r20 * r20);
    write_sum_2<double, double>((double*)inout_shared, r16, r36);
  };
  flush_sum_shared<2, double>(out_point_precond_diag,
                              0 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = fma(r22, r22, r28 * r28);
    write_sum_1<double, double>((double*)inout_shared, r36);
  };
  flush_sum_shared<1, double>(out_point_precond_diag,
                              2 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = fma(r8, r20, r33 * r35);
    r33 = fma(r33, r28, r8 * r22);
    write_sum_2<double, double>((double*)inout_shared, r36, r33);
  };
  flush_sum_shared<2, double>(out_point_precond_tril,
                              0 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = fma(r35, r28, r20 * r22);
    write_sum_1<double, double>((double*)inout_shared, r28);
  };
  flush_sum_shared<1, double>(out_point_precond_tril,
                              2 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
}

void simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac(
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
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
  simple_radial_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac_kernel<<<
      n_blocks,
      1024>>>(point,
              point_num_alloc,
              point_indices,
              pixel,
              pixel_num_alloc,
              pose,
              pose_num_alloc,
              focal_and_extra,
              focal_and_extra_num_alloc,
              principal_point,
              principal_point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_point_njtr,
              out_point_njtr_num_alloc,
              out_point_precond_diag,
              out_point_precond_diag_num_alloc,
              out_point_precond_tril,
              out_point_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar