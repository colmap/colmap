#include "kernel_simple_radial_fixed_pose_fixed_extra_calib_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_pose_fixed_extra_calib_fixed_point_res_jac_kernel(
        float* focal,
        unsigned int focal_num_alloc,
        SharedIndex* focal_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* extra_calib,
        unsigned int extra_calib_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        float* const out_focal_precond_diag,
        unsigned int out_focal_precond_diag_num_alloc,
        float* const out_focal_precond_tril,
        unsigned int out_focal_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ SharedIndex focal_indices_loc[1024];
  focal_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26;

  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        extra_calib, 0 * extra_calib_num_alloc, global_thread_idx, r0, r1, r2);
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
  };
  load_shared<1, float, float>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>(
        (float*)inout_shared, focal_indices_loc[threadIdx.x].target, r25);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = 9.99999999999999955e-07;
    r26 = r12 * r13;
    r26 = r26 * r17;
    r18 = fmaf(r14, r18, r26);
    r18 = fmaf(r9, r18, r7);
    r21 = fmaf(r12, r16, r21);
    r7 = r11 * r11;
    r7 = r7 * r15;
    r24 = r7 + r24;
    r18 = fmaf(r8, r21, r18);
    r18 = fmaf(r10, r24, r18);
    r24 = copysign(1.0, r18);
    r24 = fmaf(r0, r24, r18);
    r0 = r24 * r24;
    r0 = 1.0 / r0;
    r18 = r13 * r14;
    r18 = fmaf(r17, r18, r19);
    r18 = fmaf(r8, r18, r6);
    r16 = fmaf(r11, r16, r26);
    r22 = r23 + r22;
    r22 = r22 + r7;
    r18 = fmaf(r10, r16, r18);
    r18 = fmaf(r9, r22, r18);
    r22 = r18 * r18;
    r9 = r20 * r20;
    r16 = fmaf(r0, r9, r0 * r22);
    r16 = fmaf(r2, r16, r23);
    r24 = 1.0 / r24;
    r24 = r16 * r24;
    r25 = r25 * r24;
    r3 = fmaf(r20, r25, r3);
    r4 = fmaf(r4, r5, r1);
    r4 = fmaf(r18, r25, r4);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r3, r4);
    r25 = r18 * r5;
    r25 = r25 * r4;
    r4 = r20 * r5;
    r4 = r4 * r3;
    r4 = fmaf(r24, r4, r24 * r25);
    write_sum_1<float, float>((float*)inout_shared, r4);
  };
  flush_sum_shared<1, float>(out_focal_njtr,
                             0 * out_focal_njtr_num_alloc,
                             focal_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = r16 * r16;
    r16 = r16 * r0;
    r16 = fmaf(r9, r16, r22 * r16);
    write_sum_1<float, float>((float*)inout_shared, r16);
  };
  flush_sum_shared<1, float>(out_focal_precond_diag,
                             0 * out_focal_precond_diag_num_alloc,
                             focal_indices_loc,
                             (float*)inout_shared);
}

void simple_radial_fixed_pose_fixed_extra_calib_fixed_point_res_jac(
    float* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* extra_calib,
    unsigned int extra_calib_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    float* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    float* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_pose_fixed_extra_calib_fixed_point_res_jac_kernel<<<
      n_blocks,
      1024>>>(focal,
              focal_num_alloc,
              focal_indices,
              pixel,
              pixel_num_alloc,
              pose,
              pose_num_alloc,
              extra_calib,
              extra_calib_num_alloc,
              point,
              point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_focal_njtr,
              out_focal_njtr_num_alloc,
              out_focal_precond_diag,
              out_focal_precond_diag_num_alloc,
              out_focal_precond_tril,
              out_focal_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar