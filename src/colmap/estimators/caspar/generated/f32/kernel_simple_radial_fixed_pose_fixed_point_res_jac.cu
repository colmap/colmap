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
        float* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        float* const out_calib_precond_diag,
        unsigned int out_calib_precond_diag_num_alloc,
        float* const out_calib_precond_tril,
        unsigned int out_calib_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27;
  load_shared<4, float, float>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         calib_indices_loc[threadIdx.x].target,
                         r0,
                         r1,
                         r2,
                         r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r4 = fmaf(r4, r6, r1);
    read_idx_3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r1, r7, r8);
    read_idx_3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r9, r10, r11);
    read_idx_4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r12, r13, r14, r15);
    r16 = -2.00000000000000000e+00;
    r17 = r15 * r16;
    r18 = 2.00000000000000000e+00;
    r19 = r12 * r18;
    r20 = r13 * r19;
    r21 = fmaf(r14, r17, r20);
    r21 = fmaf(r10, r21, r1);
    r1 = r13 * r15;
    r22 = r14 * r19;
    r1 = fmaf(r18, r1, r22);
    r23 = r14 * r14;
    r23 = r16 * r23;
    r24 = 1.00000000000000000e+00;
    r25 = r13 * r13;
    r25 = fmaf(r16, r25, r24);
    r26 = r23 + r25;
    r21 = fmaf(r11, r1, r21);
    r21 = fmaf(r9, r26, r21);
    r26 = r0 * r21;
    r1 = 9.99999999999999955e-07;
    r27 = r13 * r14;
    r27 = r27 * r18;
    r19 = fmaf(r15, r19, r27);
    r19 = fmaf(r10, r19, r8);
    r22 = fmaf(r13, r17, r22);
    r8 = r12 * r12;
    r8 = r8 * r16;
    r25 = r8 + r25;
    r19 = fmaf(r9, r22, r19);
    r19 = fmaf(r11, r25, r19);
    r25 = copysign(1.0, r19);
    r25 = fmaf(r1, r25, r19);
    r1 = r25 * r25;
    r1 = 1.0 / r1;
    r19 = r14 * r15;
    r19 = fmaf(r18, r19, r20);
    r19 = fmaf(r9, r19, r7);
    r17 = fmaf(r12, r17, r27);
    r23 = r24 + r23;
    r23 = r23 + r8;
    r19 = fmaf(r11, r17, r19);
    r19 = fmaf(r10, r23, r19);
    r23 = r19 * r19;
    r10 = r21 * r21;
    r17 = fmaf(r1, r10, r1 * r23);
    r3 = fmaf(r3, r17, r24);
    r25 = 1.0 / r25;
    r11 = r3 * r25;
    r4 = fmaf(r11, r26, r4);
    r5 = fmaf(r5, r6, r2);
    r2 = r0 * r19;
    r5 = fmaf(r11, r2, r5);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5);
    r2 = r6 * r4;
    r5 = r6 * r5;
    r26 = r19 * r5;
    r8 = r21 * r6;
    r8 = r8 * r4;
    r8 = fmaf(r11, r8, r11 * r26);
    r12 = r0 * r17;
    r27 = r25 * r12;
    r9 = r21 * r6;
    r9 = r9 * r4;
    r9 = r9 * r25;
    r9 = fmaf(r12, r9, r26 * r27);
    write_sum_4<float, float>((float*)inout_shared, r8, r2, r5, r9);
  };
  flush_sum_shared<4, float>(out_calib_njtr,
                             0 * out_calib_njtr_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = r3 * r3;
    r9 = r9 * r1;
    r9 = fmaf(r10, r9, r23 * r9);
    r17 = r0 * r17;
    r1 = r1 * r12;
    r17 = r17 * r1;
    r17 = fmaf(r10, r17, r23 * r17);
    write_sum_4<float, float>((float*)inout_shared, r9, r24, r24, r17);
  };
  flush_sum_shared<4, float>(out_calib_precond_diag,
                             0 * out_calib_precond_diag_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = 0.00000000000000000e+00;
    r24 = r21 * r11;
    r11 = r19 * r11;
    r9 = r3 * r10;
    r5 = r3 * r23;
    r5 = fmaf(r1, r5, r1 * r9);
    write_sum_4<float, float>((float*)inout_shared, r24, r11, r5, r17);
  };
  flush_sum_shared<4, float>(out_calib_precond_tril,
                             0 * out_calib_precond_tril_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = r21 * r25;
    r17 = r17 * r12;
    r5 = r19 * r25;
    r5 = r5 * r12;
    write_sum_2<float, float>((float*)inout_shared, r17, r5);
  };
  flush_sum_shared<2, float>(out_calib_precond_tril,
                             4 * out_calib_precond_tril_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
}

void simple_radial_fixed_pose_fixed_point_res_jac(
    float* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    float* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    float* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_pose_fixed_point_res_jac_kernel<<<n_blocks, 1024>>>(
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
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar