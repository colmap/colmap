#include "kernel_simple_radial_fixed_pose_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_pose_fixed_point_res_jac_kernel(
        float* focal,
        unsigned int focal_num_alloc,
        SharedIndex* focal_indices,
        float* extra_calib,
        unsigned int extra_calib_num_alloc,
        SharedIndex* extra_calib_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* out_focal_jac,
        unsigned int out_focal_jac_num_alloc,
        float* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        float* const out_focal_precond_diag,
        unsigned int out_focal_precond_diag_num_alloc,
        float* const out_focal_precond_tril,
        unsigned int out_focal_precond_tril_num_alloc,
        float* out_extra_calib_jac,
        unsigned int out_extra_calib_jac_num_alloc,
        float* const out_extra_calib_njtr,
        unsigned int out_extra_calib_njtr_num_alloc,
        float* const out_extra_calib_precond_diag,
        unsigned int out_extra_calib_precond_diag_num_alloc,
        float* const out_extra_calib_precond_tril,
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25;
  load_shared<3, float, float>(extra_calib,
                               0 * extra_calib_num_alloc,
                               extra_calib_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>((float*)inout_shared,
                         extra_calib_indices_loc[threadIdx.x].target,
                         r0,
                         r1,
                         r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r3, r4);
    r5 = -1.00000000000000000e+00;
    r3 = fmaf(r3, r5, r0);
    read_idx_3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r0, r6, r7);
    read_idx_3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9, r10);
    read_idx_4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r11, r12, r13, r14);
    r15 = -2.00000000000000000e+00;
    r16 = r14 * r15;
    r17 = 2.00000000000000000e+00;
    r18 = r11 * r17;
    r19 = r12 * r18;
    r20 = fmaf(r13, r16, r19);
    r20 = fmaf(r9, r20, r0);
    r0 = r12 * r14;
    r21 = r13 * r18;
    r0 = fmaf(r17, r0, r21);
    r22 = r13 * r13;
    r22 = r15 * r22;
    r23 = 1.00000000000000000e+00;
    r24 = r12 * r12;
    r24 = fmaf(r15, r24, r23);
    r25 = r22 + r24;
    r20 = fmaf(r10, r0, r20);
    r20 = fmaf(r8, r25, r20);
    r25 = r13 * r14;
    r25 = fmaf(r17, r25, r19);
    r25 = fmaf(r8, r25, r6);
    r6 = r12 * r13;
    r6 = r6 * r17;
    r17 = fmaf(r11, r16, r6);
    r22 = r23 + r22;
    r11 = r11 * r11;
    r11 = r11 * r15;
    r22 = r22 + r11;
    r25 = fmaf(r10, r17, r25);
    r25 = fmaf(r9, r22, r25);
    r22 = r25 * r25;
    r17 = 9.99999999999999955e-07;
    r18 = fmaf(r14, r18, r6);
    r18 = fmaf(r9, r18, r7);
    r16 = fmaf(r12, r16, r21);
    r24 = r11 + r24;
    r18 = fmaf(r8, r16, r18);
    r18 = fmaf(r10, r24, r18);
    r24 = copysign(1.0, r18);
    r24 = fmaf(r17, r24, r18);
    r17 = r24 * r24;
    r17 = 1.0 / r17;
    r22 = r22 * r17;
    r18 = r20 * r20;
    r10 = fmaf(r17, r18, r22);
    r2 = fmaf(r2, r10, r23);
    r16 = r20 * r2;
  };
  load_shared<1, float, float>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>(
        (float*)inout_shared, focal_indices_loc[threadIdx.x].target, r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r24 = 1.0 / r24;
    r11 = r8 * r24;
    r3 = fmaf(r11, r16, r3);
    r4 = fmaf(r4, r5, r1);
    r1 = r25 * r2;
    r4 = fmaf(r11, r1, r4);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r3, r4);
    r1 = r20 * r2;
    r1 = r1 * r24;
    r16 = r25 * r2;
    r16 = r16 * r24;
    write_idx_2<1024, float, float, float2>(
        out_focal_jac, 0 * out_focal_jac_num_alloc, global_thread_idx, r1, r16);
    r16 = r2 * r24;
    r4 = r5 * r4;
    r1 = r25 * r4;
    r21 = r20 * r2;
    r21 = r21 * r5;
    r21 = r21 * r3;
    r21 = fmaf(r24, r21, r1 * r16);
    write_sum_1<float, float>((float*)inout_shared, r21);
  };
  flush_sum_shared<1, float>(out_focal_njtr,
                             0 * out_focal_njtr_num_alloc,
                             focal_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = r2 * r2;
    r16 = r17 * r18;
    r16 = fmaf(r21, r16, r21 * r22);
    write_sum_1<float, float>((float*)inout_shared, r16);
  };
  flush_sum_shared<1, float>(out_focal_precond_diag,
                             0 * out_focal_precond_diag_num_alloc,
                             focal_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = r8 * r20;
    r16 = r16 * r10;
    r16 = r16 * r24;
    r11 = r10 * r11;
    r21 = r25 * r11;
    write_idx_2<1024, float, float, float2>(out_extra_calib_jac,
                                            0 * out_extra_calib_jac_num_alloc,
                                            global_thread_idx,
                                            r16,
                                            r21);
    r9 = r5 * r3;
    r7 = r5 * r3;
    r7 = fmaf(r16, r7, r11 * r1);
    write_sum_3<float, float>((float*)inout_shared, r9, r4, r7);
  };
  flush_sum_shared<3, float>(out_extra_calib_njtr,
                             0 * out_extra_calib_njtr_num_alloc,
                             extra_calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = r8 * r8;
    r8 = r8 * r10;
    r8 = r8 * r10;
    r10 = r17 * r18;
    r10 = fmaf(r8, r10, r22 * r8);
    write_sum_3<float, float>((float*)inout_shared, r23, r23, r10);
  };
  flush_sum_shared<3, float>(out_extra_calib_precond_diag,
                             0 * out_extra_calib_precond_diag_num_alloc,
                             extra_calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = 0.00000000000000000e+00;
    write_sum_3<float, float>((float*)inout_shared, r10, r16, r21);
  };
  flush_sum_shared<3, float>(out_extra_calib_precond_tril,
                             0 * out_extra_calib_precond_tril_num_alloc,
                             extra_calib_indices_loc,
                             (float*)inout_shared);
}

void simple_radial_fixed_pose_fixed_point_res_jac(
    float* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    float* extra_calib,
    unsigned int extra_calib_num_alloc,
    SharedIndex* extra_calib_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* out_focal_jac,
    unsigned int out_focal_jac_num_alloc,
    float* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    float* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    float* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
    float* out_extra_calib_jac,
    unsigned int out_extra_calib_jac_num_alloc,
    float* const out_extra_calib_njtr,
    unsigned int out_extra_calib_njtr_num_alloc,
    float* const out_extra_calib_precond_diag,
    unsigned int out_extra_calib_precond_diag_num_alloc,
    float* const out_extra_calib_precond_tril,
    unsigned int out_extra_calib_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_pose_fixed_point_res_jac_kernel<<<n_blocks, 1024>>>(
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