#include "kernel_pinhole_fixed_pose_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_pose_fixed_point_res_jac_kernel(
        double* focal,
        unsigned int focal_num_alloc,
        SharedIndex* focal_indices,
        double* extra_calib,
        unsigned int extra_calib_num_alloc,
        SharedIndex* extra_calib_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* pose,
        unsigned int pose_num_alloc,
        double* point,
        unsigned int point_num_alloc,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26;
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
  };
  load_shared<2, double, double>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, focal_indices_loc[threadIdx.x].target, r0, r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r6, r7);
    read_idx_2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9);
    read_idx_2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r10, r11);
    r12 = -2.00000000000000000e+00;
    r13 = r11 * r12;
    read_idx_2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r14, r15);
    r16 = 2.00000000000000000e+00;
    r17 = r14 * r16;
    r18 = r15 * r17;
    r19 = fma(r10, r13, r18);
    r19 = fma(r9, r19, r6);
    read_idx_1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r6);
    r20 = r15 * r11;
    r21 = r10 * r17;
    r20 = fma(r16, r20, r21);
    r22 = r10 * r10;
    r22 = r12 * r22;
    r23 = 1.00000000000000000e+00;
    r24 = r15 * r15;
    r24 = fma(r12, r24, r23);
    r25 = r22 + r24;
    r19 = fma(r6, r20, r19);
    r19 = fma(r8, r25, r19);
    r25 = 1.00000000000000008e-15;
    read_idx_1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r20);
    r26 = r15 * r10;
    r26 = r26 * r16;
    r17 = fma(r11, r17, r26);
    r17 = fma(r9, r17, r20);
    r21 = fma(r15, r13, r21);
    r20 = r14 * r14;
    r20 = r20 * r12;
    r24 = r20 + r24;
    r17 = fma(r8, r21, r17);
    r17 = fma(r6, r24, r17);
    r24 = copysign(1.0, r17);
    r24 = fma(r25, r24, r17);
    r25 = 1.0 / r24;
    r17 = r19 * r25;
    r2 = fma(r0, r17, r2);
    r3 = fma(r3, r4, r1);
    r1 = r10 * r11;
    r1 = fma(r16, r1, r18);
    r1 = fma(r8, r1, r7);
    r13 = fma(r14, r13, r26);
    r22 = r23 + r22;
    r22 = r22 + r20;
    r1 = fma(r6, r13, r1);
    r1 = fma(r9, r22, r1);
    r22 = r5 * r1;
    r3 = fma(r25, r22, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r22 = r1 * r25;
    write_idx_2<1024, double, double, double2>(out_focal_jac,
                                               0 * out_focal_jac_num_alloc,
                                               global_thread_idx,
                                               r17,
                                               r22);
    r2 = r4 * r2;
    r17 = r17 * r2;
    r22 = r4 * r1;
    r22 = r22 * r3;
    r22 = r22 * r25;
    write_sum_2<double, double>((double*)inout_shared, r17, r22);
  };
  flush_sum_shared<2, double>(out_focal_njtr,
                              0 * out_focal_njtr_num_alloc,
                              focal_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = r19 * r19;
    r24 = r24 * r24;
    r24 = 1.0 / r24;
    r19 = r19 * r24;
    r22 = r1 * r1;
    r22 = r24 * r22;
    write_sum_2<double, double>((double*)inout_shared, r19, r22);
  };
  flush_sum_shared<2, double>(out_focal_precond_diag,
                              0 * out_focal_precond_diag_num_alloc,
                              focal_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r4 * r3;
    write_sum_2<double, double>((double*)inout_shared, r2, r3);
  };
  flush_sum_shared<2, double>(out_extra_calib_njtr,
                              0 * out_extra_calib_njtr_num_alloc,
                              extra_calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<double, double>((double*)inout_shared, r23, r23);
  };
  flush_sum_shared<2, double>(out_extra_calib_precond_diag,
                              0 * out_extra_calib_precond_diag_num_alloc,
                              extra_calib_indices_loc,
                              (double*)inout_shared);
}

void pinhole_fixed_pose_fixed_point_res_jac(
    double* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    double* extra_calib,
    unsigned int extra_calib_num_alloc,
    SharedIndex* extra_calib_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* point,
    unsigned int point_num_alloc,
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
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_fixed_pose_fixed_point_res_jac_kernel<<<n_blocks, 1024>>>(
      focal,
      focal_num_alloc,
      focal_indices,
      extra_calib,
      extra_calib_num_alloc,
      extra_calib_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      point,
      point_num_alloc,
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
      problem_size);
}

}  // namespace caspar