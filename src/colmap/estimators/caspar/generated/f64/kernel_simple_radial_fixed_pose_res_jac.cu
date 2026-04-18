#include "kernel_simple_radial_fixed_pose_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_pose_res_jac_kernel(
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
        double* pose,
        unsigned int pose_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36;
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
  load_shared<1, double, double>(extra_calib,
                                 2 * extra_calib_num_alloc,
                                 extra_calib_indices_loc,
                                 (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared,
                          extra_calib_indices_loc[threadIdx.x].target,
                          r25);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r26 = 1.00000000000000008e-15;
    read_idx_1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r27);
    r9 = r13 * r9;
    r9 = r9 * r14;
    r15 = fma(r10, r15, r9);
    r27 = fma(r7, r15, r27);
    r28 = r13 * r10;
    r28 = fma(r8, r28, r20);
    r20 = r12 * r12;
    r20 = r20 * r8;
    r23 = r20 + r23;
    r27 = fma(r6, r28, r27);
    r27 = fma(r18, r23, r27);
    r29 = copysign(1.0, r27);
    r29 = fma(r26, r29, r27);
    r26 = r29 * r29;
    r27 = 1.0 / r26;
    r30 = r0 * r27;
    r11 = fma(r14, r11, r16);
    r6 = fma(r6, r11, r5);
    r5 = r12 * r10;
    r5 = fma(r8, r5, r9);
    r21 = r22 + r21;
    r21 = r21 + r20;
    r6 = fma(r18, r5, r6);
    r6 = fma(r7, r21, r6);
    r7 = r6 * r6;
    r18 = fma(r27, r7, r0 * r30);
    r20 = fma(r25, r18, r22);
    r9 = r0 * r20;
  };
  load_shared<1, double, double>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, focal_indices_loc[threadIdx.x].target, r16);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r31 = 1.0 / r29;
    r32 = r16 * r31;
    r2 = fma(r32, r9, r2);
    r3 = fma(r3, r4, r1);
    r1 = r6 * r20;
    r3 = fma(r32, r1, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = r0 * r20;
    r1 = r1 * r31;
    r9 = r6 * r20;
    r9 = r9 * r31;
    write_idx_2<1024, double, double, double2>(
        out_focal_jac, 0 * out_focal_jac_num_alloc, global_thread_idx, r1, r9);
    r9 = r0 * r2;
    r1 = r20 * r4;
    r9 = r9 * r31;
    r33 = r6 * r3;
    r33 = r33 * r31;
    r33 = fma(r1, r33, r1 * r9);
    write_sum_1<double, double>((double*)inout_shared, r33);
  };
  flush_sum_shared<1, double>(out_focal_njtr,
                              0 * out_focal_njtr_num_alloc,
                              focal_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = r20 * r20;
    r33 = r33 * r27;
    r9 = r0 * r20;
    r9 = r9 * r20;
    r9 = fma(r30, r9, r7 * r33);
    write_sum_1<double, double>((double*)inout_shared, r9);
  };
  flush_sum_shared<1, double>(out_focal_precond_diag,
                              0 * out_focal_precond_diag_num_alloc,
                              focal_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = r0 * r18;
    r9 = r9 * r32;
    r33 = r6 * r18;
    r33 = r33 * r32;
    write_idx_2<1024, double, double, double2>(
        out_extra_calib_jac,
        0 * out_extra_calib_jac_num_alloc,
        global_thread_idx,
        r9,
        r33);
    r31 = r4 * r2;
    r34 = r4 * r3;
    write_sum_2<double, double>((double*)inout_shared, r31, r34);
  };
  flush_sum_shared<2, double>(out_extra_calib_njtr,
                              0 * out_extra_calib_njtr_num_alloc,
                              extra_calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r6 * r18;
    r34 = r34 * r4;
    r34 = r34 * r3;
    r31 = r0 * r18;
    r31 = r31 * r4;
    r31 = r31 * r2;
    r31 = fma(r32, r31, r32 * r34);
    write_sum_1<double, double>((double*)inout_shared, r31);
  };
  flush_sum_shared<1, double>(out_extra_calib_njtr,
                              2 * out_extra_calib_njtr_num_alloc,
                              extra_calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<double, double>((double*)inout_shared, r22, r22);
  };
  flush_sum_shared<2, double>(out_extra_calib_precond_diag,
                              0 * out_extra_calib_precond_diag_num_alloc,
                              extra_calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = r0 * r30;
    r31 = r16 * r16;
    r34 = r18 * r18;
    r31 = r31 * r34;
    r34 = r27 * r7;
    r34 = fma(r31, r34, r31 * r22);
    write_sum_1<double, double>((double*)inout_shared, r34);
  };
  flush_sum_shared<1, double>(out_extra_calib_precond_diag,
                              2 * out_extra_calib_precond_diag_num_alloc,
                              extra_calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = 0.00000000000000000e+00;
    write_sum_2<double, double>((double*)inout_shared, r34, r9);
  };
  flush_sum_shared<2, double>(out_extra_calib_precond_tril,
                              0 * out_extra_calib_precond_tril_num_alloc,
                              extra_calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_1<double, double>((double*)inout_shared, r33);
  };
  flush_sum_shared<1, double>(out_extra_calib_precond_tril,
                              2 * out_extra_calib_precond_tril_num_alloc,
                              extra_calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = r24 * r20;
    r9 = r28 * r30;
    r1 = r16 * r1;
    r9 = fma(r1, r9, r32 * r33);
    r33 = r0 * r0;
    r26 = r29 * r26;
    r26 = 1.0 / r26;
    r26 = r8 * r26;
    r33 = r33 * r26;
    r8 = r14 * r24;
    r8 = fma(r30, r8, r28 * r33);
    r29 = r14 * r11;
    r29 = r29 * r6;
    r8 = fma(r27, r29, r8);
    r16 = r28 * r7;
    r8 = fma(r26, r16, r8);
    r16 = r0 * r8;
    r25 = r25 * r32;
    r9 = fma(r25, r16, r9);
    r16 = r6 * r27;
    r16 = r16 * r1;
    r29 = r6 * r8;
    r29 = fma(r25, r29, r28 * r16);
    r34 = r11 * r20;
    r29 = fma(r32, r34, r29);
    write_idx_2<1024, double, double, double2>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r9, r29);
    r34 = r14 * r17;
    r34 = fma(r15, r33, r30 * r34);
    r22 = r15 * r7;
    r34 = fma(r26, r22, r34);
    r31 = r14 * r21;
    r31 = r31 * r6;
    r34 = fma(r27, r31, r34);
    r34 = r34 * r25;
    r31 = r17 * r20;
    r31 = fma(r32, r31, r0 * r34);
    r22 = r15 * r30;
    r31 = fma(r1, r22, r31);
    r22 = r21 * r20;
    r22 = fma(r32, r22, r6 * r34);
    r22 = fma(r15, r16, r22);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               2 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r31,
                                               r22);
    r34 = r19 * r20;
    r35 = r14 * r19;
    r33 = fma(r23, r33, r30 * r35);
    r35 = r14 * r5;
    r35 = r35 * r6;
    r33 = fma(r27, r35, r33);
    r36 = r23 * r7;
    r33 = fma(r26, r36, r33);
    r36 = r0 * r33;
    r36 = fma(r25, r36, r32 * r34);
    r34 = r23 * r30;
    r36 = fma(r1, r34, r36);
    r34 = r5 * r20;
    r16 = fma(r23, r16, r32 * r34);
    r34 = r6 * r33;
    r16 = fma(r25, r34, r16);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               4 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r36,
                                               r16);
    r34 = r4 * r3;
    r25 = r4 * r2;
    r25 = fma(r9, r25, r29 * r34);
    r34 = r4 * r3;
    r32 = r4 * r2;
    r32 = fma(r31, r32, r22 * r34);
    write_sum_2<double, double>((double*)inout_shared, r25, r32);
  };
  flush_sum_shared<2, double>(out_point_njtr,
                              0 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r4 * r3;
    r25 = r4 * r2;
    r25 = fma(r36, r25, r16 * r32);
    write_sum_1<double, double>((double*)inout_shared, r25);
  };
  flush_sum_shared<1, double>(out_point_njtr,
                              2 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = fma(r9, r9, r29 * r29);
    r32 = fma(r31, r31, r22 * r22);
    write_sum_2<double, double>((double*)inout_shared, r25, r32);
  };
  flush_sum_shared<2, double>(out_point_precond_diag,
                              0 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r36, r36, r16 * r16);
    write_sum_1<double, double>((double*)inout_shared, r32);
  };
  flush_sum_shared<1, double>(out_point_precond_diag,
                              2 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r29, r22, r9 * r31);
    r29 = fma(r29, r16, r9 * r36);
    write_sum_2<double, double>((double*)inout_shared, r32, r29);
  };
  flush_sum_shared<2, double>(out_point_precond_tril,
                              0 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = fma(r31, r36, r22 * r16);
    write_sum_1<double, double>((double*)inout_shared, r36);
  };
  flush_sum_shared<1, double>(out_point_precond_tril,
                              2 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
}

void simple_radial_fixed_pose_res_jac(
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
    double* pose,
    unsigned int pose_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
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
  simple_radial_fixed_pose_res_jac_kernel<<<n_blocks, 1024>>>(
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
      pose,
      pose_num_alloc,
      out_res,
      out_res_num_alloc,
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