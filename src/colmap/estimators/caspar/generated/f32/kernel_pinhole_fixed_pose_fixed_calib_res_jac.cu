#include "kernel_pinhole_fixed_pose_fixed_calib_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_pose_fixed_calib_res_jac_kernel(
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* calib,
        unsigned int calib_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        float* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        float* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        calib, 0 * calib_num_alloc, global_thread_idx, r0, r1, r2, r3);
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r4 = fmaf(r4, r6, r2);
    r2 = 9.99999999999999955e-07;
    read_idx_3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r7, r8, r9);
  };
  load_shared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>((float*)inout_shared,
                         point_indices_loc[threadIdx.x].target,
                         r10,
                         r11,
                         r12);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r13, r14, r15, r16);
    r17 = r14 * r15;
    r18 = 2.00000000000000000e+00;
    r17 = r17 * r18;
    r19 = r13 * r18;
    r20 = fmaf(r16, r19, r17);
    r9 = fmaf(r11, r20, r9);
    r21 = r15 * r19;
    r22 = -2.00000000000000000e+00;
    r23 = r16 * r22;
    r24 = fmaf(r14, r23, r21);
    r25 = r13 * r13;
    r25 = r25 * r22;
    r26 = 1.00000000000000000e+00;
    r27 = r14 * r14;
    r27 = fmaf(r22, r27, r26);
    r28 = r25 + r27;
    r9 = fmaf(r10, r24, r9);
    r9 = fmaf(r12, r28, r9);
    r29 = copysign(1.0, r9);
    r29 = fmaf(r2, r29, r9);
    r2 = 1.0 / r29;
    r19 = r14 * r19;
    r9 = fmaf(r15, r23, r19);
    r7 = fmaf(r11, r9, r7);
    r30 = r14 * r16;
    r30 = fmaf(r18, r30, r21);
    r21 = r15 * r15;
    r21 = r22 * r21;
    r27 = r21 + r27;
    r7 = fmaf(r12, r30, r7);
    r7 = fmaf(r10, r27, r7);
    r7 = r0 * r7;
    r4 = fmaf(r2, r7, r4);
    r5 = fmaf(r5, r6, r3);
    r3 = r15 * r16;
    r3 = fmaf(r18, r3, r19);
    r10 = fmaf(r10, r3, r8);
    r23 = fmaf(r13, r23, r17);
    r21 = r26 + r21;
    r21 = r21 + r25;
    r10 = fmaf(r12, r23, r10);
    r10 = fmaf(r11, r21, r10);
    r10 = r1 * r10;
    r5 = fmaf(r2, r10, r5);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5);
    r11 = r6 * r5;
    r12 = r1 * r3;
    r29 = r29 * r29;
    r29 = 1.0 / r29;
    r29 = r6 * r29;
    r25 = r24 * r29;
    r25 = fmaf(r10, r25, r2 * r12);
    r12 = r6 * r4;
    r26 = r0 * r27;
    r7 = r29 * r7;
    r26 = fmaf(r24, r7, r2 * r26);
    r12 = fmaf(r26, r12, r25 * r11);
    r11 = r6 * r5;
    r13 = r1 * r21;
    r17 = r20 * r29;
    r17 = fmaf(r10, r17, r2 * r13);
    r13 = r6 * r4;
    r8 = r0 * r9;
    r8 = fmaf(r20, r7, r2 * r8);
    r13 = fmaf(r8, r13, r17 * r11);
    r11 = r6 * r5;
    r19 = r1 * r23;
    r18 = r28 * r29;
    r18 = fmaf(r10, r18, r2 * r19);
    r19 = r6 * r4;
    r10 = r0 * r30;
    r7 = fmaf(r28, r7, r2 * r10);
    r19 = fmaf(r7, r19, r18 * r11);
    write_sum_3<float, float>((float*)inout_shared, r12, r13, r19);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fmaf(r26, r26, r25 * r25);
    r13 = fmaf(r8, r8, r17 * r17);
    r12 = fmaf(r18, r18, r7 * r7);
    write_sum_3<float, float>((float*)inout_shared, r19, r13, r12);
  };
  flush_sum_shared<3, float>(out_point_precond_diag,
                             0 * out_point_precond_diag_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fmaf(r26, r8, r25 * r17);
    r25 = fmaf(r25, r18, r26 * r7);
    r18 = fmaf(r17, r18, r8 * r7);
    write_sum_3<float, float>((float*)inout_shared, r12, r25, r18);
  };
  flush_sum_shared<3, float>(out_point_precond_tril,
                             0 * out_point_precond_tril_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
}

void pinhole_fixed_pose_fixed_calib_res_jac(
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* calib,
    unsigned int calib_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    float* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_fixed_pose_fixed_calib_res_jac_kernel<<<n_blocks, 1024>>>(
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      calib,
      calib_num_alloc,
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