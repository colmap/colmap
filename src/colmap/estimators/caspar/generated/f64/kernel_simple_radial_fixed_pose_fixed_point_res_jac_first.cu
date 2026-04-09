#include "kernel_simple_radial_fixed_pose_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_pose_fixed_point_res_jac_first_kernel(
        double* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* pose,
        unsigned int pose_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
        double* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        double* const out_calib_precond_diag,
        unsigned int out_calib_precond_diag_num_alloc,
        double* const out_calib_precond_tril,
        unsigned int out_calib_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27;
  load_shared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r1);
    read_idx_2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r1, r5);
    read_idx_2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r6, r7);
    read_idx_2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r8, r9);
    r10 = -2.00000000000000000e+00;
    r11 = r9 * r10;
    read_idx_2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r12, r13);
    r14 = 2.00000000000000000e+00;
    r15 = r12 * r14;
    r16 = r13 * r15;
    r17 = fma(r8, r11, r16);
    r17 = fma(r7, r17, r1);
    read_idx_1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r1);
    r18 = r13 * r9;
    r19 = r8 * r15;
    r18 = fma(r14, r18, r19);
    r20 = r8 * r8;
    r20 = r10 * r20;
    r21 = 1.00000000000000000e+00;
    r22 = r13 * r13;
    r22 = fma(r10, r22, r21);
    r23 = r20 + r22;
    r17 = fma(r1, r18, r17);
    r17 = fma(r6, r23, r17);
    r23 = r0 * r17;
  };
  load_shared<2, double, double>(
      calib, 2 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r18, r24);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r25 = 1.00000000000000008e-15;
    read_idx_1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r26);
    r27 = r13 * r8;
    r27 = r27 * r14;
    r15 = fma(r9, r15, r27);
    r15 = fma(r7, r15, r26);
    r19 = fma(r13, r11, r19);
    r26 = r12 * r12;
    r26 = r26 * r10;
    r22 = r26 + r22;
    r15 = fma(r6, r19, r15);
    r15 = fma(r1, r22, r15);
    r22 = copysign(1.0, r15);
    r22 = fma(r25, r22, r15);
    r25 = r22 * r22;
    r25 = 1.0 / r25;
    r15 = r17 * r17;
    r19 = r8 * r9;
    r19 = fma(r14, r19, r16);
    r19 = fma(r6, r19, r5);
    r11 = fma(r12, r11, r27);
    r20 = r21 + r20;
    r20 = r20 + r26;
    r19 = fma(r1, r11, r19);
    r19 = fma(r7, r20, r19);
    r20 = r19 * r19;
    r7 = fma(r25, r20, r25 * r15);
    r24 = fma(r24, r7, r21);
    r22 = 1.0 / r22;
    r11 = r24 * r22;
    r2 = fma(r11, r23, r2);
    r3 = fma(r3, r4, r18);
    r18 = r0 * r19;
    r3 = fma(r11, r18, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r18 = fma(r2, r2, r3 * r3);
  };
  sum_store<double>(out_rTr_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r18);
  if (global_thread_idx < problem_size) {
    r18 = r4 * r2;
    r3 = r4 * r3;
    r23 = r19 * r3;
    r1 = r17 * r4;
    r1 = r1 * r2;
    r1 = fma(r11, r1, r11 * r23);
    write_sum_2<double, double>((double*)inout_shared, r1, r18);
  };
  flush_sum_shared<2, double>(out_calib_njtr,
                              0 * out_calib_njtr_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = r0 * r7;
    r1 = r22 * r18;
    r26 = r17 * r4;
    r26 = r26 * r2;
    r26 = r26 * r22;
    r26 = fma(r18, r26, r23 * r1);
    write_sum_2<double, double>((double*)inout_shared, r3, r26);
  };
  flush_sum_shared<2, double>(out_calib_njtr,
                              2 * out_calib_njtr_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = r24 * r24;
    r26 = r26 * r25;
    r26 = fma(r15, r26, r20 * r26);
    write_sum_2<double, double>((double*)inout_shared, r26, r21);
  };
  flush_sum_shared<2, double>(out_calib_precond_diag,
                              0 * out_calib_precond_diag_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r0 * r7;
    r25 = r25 * r18;
    r7 = r7 * r25;
    r7 = fma(r15, r7, r20 * r7);
    write_sum_2<double, double>((double*)inout_shared, r21, r7);
  };
  flush_sum_shared<2, double>(out_calib_precond_diag,
                              2 * out_calib_precond_diag_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r17 * r11;
    r11 = r19 * r11;
    write_sum_2<double, double>((double*)inout_shared, r7, r11);
  };
  flush_sum_shared<2, double>(out_calib_precond_tril,
                              0 * out_calib_precond_tril_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = 0.00000000000000000e+00;
    r7 = r24 * r20;
    r21 = r24 * r15;
    r21 = fma(r25, r21, r25 * r7);
    write_sum_2<double, double>((double*)inout_shared, r21, r11);
  };
  flush_sum_shared<2, double>(out_calib_precond_tril,
                              2 * out_calib_precond_tril_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = r17 * r22;
    r11 = r11 * r18;
    r21 = r19 * r22;
    r21 = r21 * r18;
    write_sum_2<double, double>((double*)inout_shared, r11, r21);
  };
  flush_sum_shared<2, double>(out_calib_precond_tril,
                              4 * out_calib_precond_tril_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  sum_flush_final<double>(out_rTr_local, out_rTr, 1);
}

void simple_radial_fixed_pose_fixed_point_res_jac_first(
    double* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    double* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    double* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_pose_fixed_point_res_jac_first_kernel<<<n_blocks, 1024>>>(
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar