#include "kernel_pinhole_fixed_pose_fixed_focal_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_pose_fixed_focal_res_jac_first_kernel(
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
        double* focal,
        unsigned int focal_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
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
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30;
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
    r0 = 1.00000000000000008e-15;
    read_idx_1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r5);
  };
  load_shared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r8, r9);
    read_idx_2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r10, r11);
    r12 = r9 * r10;
    r13 = 2.00000000000000000e+00;
    r12 = r12 * r13;
    r14 = r8 * r13;
    r15 = fma(r11, r14, r12);
    r5 = fma(r7, r15, r5);
    r16 = r10 * r14;
    r17 = -2.00000000000000000e+00;
    r18 = r11 * r17;
    r19 = fma(r9, r18, r16);
  };
  load_shared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r20);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r21 = r8 * r8;
    r21 = r21 * r17;
    r22 = 1.00000000000000000e+00;
    r23 = r9 * r9;
    r23 = fma(r17, r23, r22);
    r24 = r21 + r23;
    r5 = fma(r6, r19, r5);
    r5 = fma(r20, r24, r5);
    r25 = copysign(1.0, r5);
    r25 = fma(r0, r25, r5);
    r0 = 1.0 / r25;
    read_idx_2<1024, double, double, double2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r5, r26);
    read_idx_2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r27, r28);
    r14 = r9 * r14;
    r29 = fma(r10, r18, r14);
    r27 = fma(r7, r29, r27);
    r30 = r9 * r11;
    r30 = fma(r13, r30, r16);
    r16 = r10 * r10;
    r16 = r17 * r16;
    r23 = r16 + r23;
    r27 = fma(r20, r30, r27);
    r27 = fma(r6, r23, r27);
    r27 = r5 * r27;
    r2 = fma(r0, r27, r2);
    r3 = fma(r3, r4, r1);
    r1 = r10 * r11;
    r1 = fma(r13, r1, r14);
    r6 = fma(r6, r1, r28);
    r18 = fma(r8, r18, r12);
    r16 = r22 + r16;
    r16 = r16 + r21;
    r6 = fma(r20, r18, r6);
    r6 = fma(r7, r16, r6);
    r6 = r26 * r6;
    r3 = fma(r0, r6, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r7 = fma(r2, r2, r3 * r3);
  };
  sum_store<double>(out_rTr_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r7);
  if (global_thread_idx < problem_size) {
    r7 = r4 * r2;
    r20 = r4 * r3;
    write_sum_2<double, double>((double*)inout_shared, r7, r20);
  };
  flush_sum_shared<2, double>(out_extra_calib_njtr,
                              0 * out_extra_calib_njtr_num_alloc,
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
    r25 = r25 * r25;
    r25 = 1.0 / r25;
    r25 = r4 * r25;
    r27 = r25 * r27;
    r22 = r5 * r23;
    r22 = fma(r0, r22, r19 * r27);
    r20 = r26 * r1;
    r7 = r19 * r25;
    r7 = fma(r6, r7, r0 * r20);
    write_idx_2<1024, double, double, double2>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r22, r7);
    r20 = r5 * r29;
    r20 = fma(r0, r20, r15 * r27);
    r21 = r15 * r25;
    r8 = r26 * r16;
    r8 = fma(r0, r8, r6 * r21);
    write_idx_2<1024, double, double, double2>(
        out_point_jac, 2 * out_point_jac_num_alloc, global_thread_idx, r20, r8);
    r21 = r5 * r30;
    r21 = fma(r0, r21, r24 * r27);
    r27 = r24 * r25;
    r12 = r26 * r18;
    r12 = fma(r0, r12, r6 * r27);
    write_idx_2<1024, double, double, double2>(out_point_jac,
                                               4 * out_point_jac_num_alloc,
                                               global_thread_idx,
                                               r21,
                                               r12);
    r27 = r4 * r3;
    r0 = r4 * r2;
    r0 = fma(r22, r0, r7 * r27);
    r27 = r4 * r3;
    r6 = r4 * r2;
    r6 = fma(r20, r6, r8 * r27);
    write_sum_2<double, double>((double*)inout_shared, r0, r6);
  };
  flush_sum_shared<2, double>(out_point_njtr,
                              0 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r4 * r3;
    r0 = r4 * r2;
    r0 = fma(r21, r0, r12 * r6);
    write_sum_1<double, double>((double*)inout_shared, r0);
  };
  flush_sum_shared<1, double>(out_point_njtr,
                              2 * out_point_njtr_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = fma(r22, r22, r7 * r7);
    r6 = fma(r20, r20, r8 * r8);
    write_sum_2<double, double>((double*)inout_shared, r0, r6);
  };
  flush_sum_shared<2, double>(out_point_precond_diag,
                              0 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r21, r21, r12 * r12);
    write_sum_1<double, double>((double*)inout_shared, r6);
  };
  flush_sum_shared<1, double>(out_point_precond_diag,
                              2 * out_point_precond_diag_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r7, r8, r22 * r20);
    r7 = fma(r7, r12, r22 * r21);
    write_sum_2<double, double>((double*)inout_shared, r6, r7);
  };
  flush_sum_shared<2, double>(out_point_precond_tril,
                              0 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = fma(r20, r21, r8 * r12);
    write_sum_1<double, double>((double*)inout_shared, r21);
  };
  flush_sum_shared<1, double>(out_point_precond_tril,
                              2 * out_point_precond_tril_num_alloc,
                              point_indices_loc,
                              (double*)inout_shared);
  sum_flush_final<double>(out_rTr_local, out_rTr, 1);
}

void pinhole_fixed_pose_fixed_focal_res_jac_first(
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
    double* focal,
    unsigned int focal_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
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
  pinhole_fixed_pose_fixed_focal_res_jac_first_kernel<<<n_blocks, 1024>>>(
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
      focal,
      focal_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
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