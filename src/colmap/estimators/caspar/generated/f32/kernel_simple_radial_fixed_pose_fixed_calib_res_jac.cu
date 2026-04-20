#include "kernel_simple_radial_fixed_pose_fixed_calib_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_pose_fixed_calib_res_jac_kernel(
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
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        calib, 0 * calib_num_alloc, global_thread_idx, r0, r1, r2, r3);
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r4 = fmaf(r4, r6, r1);
    read_idx_3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r1, r7, r8);
  };
  load_shared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>((float*)inout_shared,
                         point_indices_loc[threadIdx.x].target,
                         r9,
                         r10,
                         r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = -2.00000000000000000e+00;
    read_idx_4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r13, r14, r15, r16);
    r17 = r15 * r16;
    r18 = 2.00000000000000000e+00;
    r19 = r13 * r18;
    r20 = r14 * r19;
    r21 = fmaf(r12, r17, r20);
    r1 = fmaf(r10, r21, r1);
    r22 = r14 * r16;
    r23 = r15 * r19;
    r22 = fmaf(r18, r22, r23);
    r24 = r15 * r15;
    r24 = r24 * r12;
    r25 = 1.00000000000000000e+00;
    r26 = r14 * r14;
    r26 = fmaf(r12, r26, r25);
    r27 = r24 + r26;
    r1 = fmaf(r11, r22, r1);
    r1 = fmaf(r9, r27, r1);
    r28 = 9.99999999999999955e-07;
    r15 = r14 * r15;
    r15 = r15 * r18;
    r19 = fmaf(r16, r19, r15);
    r8 = fmaf(r10, r19, r8);
    r29 = r14 * r16;
    r29 = fmaf(r12, r29, r23);
    r23 = r13 * r13;
    r23 = r23 * r12;
    r26 = r23 + r26;
    r8 = fmaf(r9, r29, r8);
    r8 = fmaf(r11, r26, r8);
    r30 = copysign(1.0, r8);
    r30 = fmaf(r28, r30, r8);
    r28 = 1.0 / r30;
    r17 = fmaf(r18, r17, r20);
    r9 = fmaf(r9, r17, r7);
    r7 = r13 * r16;
    r7 = fmaf(r12, r7, r15);
    r24 = r25 + r24;
    r24 = r24 + r23;
    r9 = fmaf(r11, r7, r9);
    r9 = fmaf(r10, r24, r9);
    r10 = r30 * r30;
    r11 = 1.0 / r10;
    r23 = r9 * r11;
    r15 = r1 * r1;
    r20 = fmaf(r11, r15, r9 * r23);
    r20 = fmaf(r3, r20, r25);
    r20 = r0 * r20;
    r25 = r28 * r20;
    r4 = fmaf(r1, r25, r4);
    r5 = fmaf(r5, r6, r2);
    r5 = fmaf(r9, r25, r5);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5);
    r2 = r6 * r4;
    r8 = r1 * r29;
    r8 = r8 * r6;
    r8 = r8 * r11;
    r8 = fmaf(r27, r25, r20 * r8);
    r31 = r0 * r3;
    r10 = r30 * r10;
    r10 = 1.0 / r10;
    r10 = r12 * r10;
    r12 = r29 * r10;
    r30 = r18 * r27;
    r30 = r30 * r1;
    r30 = fmaf(r11, r30, r15 * r12);
    r12 = r9 * r9;
    r12 = r12 * r10;
    r32 = r18 * r17;
    r30 = fmaf(r23, r32, r30);
    r30 = fmaf(r29, r12, r30);
    r31 = r31 * r30;
    r31 = r31 * r28;
    r8 = fmaf(r1, r31, r8);
    r30 = r6 * r5;
    r32 = r6 * r20;
    r32 = r32 * r23;
    r31 = fmaf(r29, r32, r9 * r31);
    r31 = fmaf(r17, r25, r31);
    r30 = fmaf(r31, r30, r8 * r2);
    r2 = r6 * r5;
    r33 = r0 * r3;
    r34 = r18 * r21;
    r34 = r34 * r1;
    r35 = r19 * r10;
    r35 = fmaf(r15, r35, r11 * r34);
    r34 = r18 * r24;
    r35 = fmaf(r23, r34, r35);
    r35 = fmaf(r19, r12, r35);
    r33 = r33 * r9;
    r33 = r33 * r35;
    r33 = fmaf(r28, r33, r19 * r32);
    r33 = fmaf(r24, r25, r33);
    r34 = r6 * r4;
    r36 = r1 * r19;
    r36 = r36 * r6;
    r36 = r36 * r11;
    r36 = fmaf(r21, r25, r20 * r36);
    r37 = r0 * r3;
    r37 = r37 * r1;
    r37 = r37 * r35;
    r36 = fmaf(r28, r37, r36);
    r34 = fmaf(r36, r34, r33 * r2);
    r2 = r6 * r4;
    r37 = r0 * r3;
    r35 = r18 * r22;
    r35 = r35 * r1;
    r38 = r26 * r10;
    r38 = fmaf(r15, r38, r11 * r35);
    r35 = r18 * r7;
    r38 = fmaf(r23, r35, r38);
    r38 = fmaf(r26, r12, r38);
    r37 = r37 * r1;
    r37 = r37 * r38;
    r12 = r1 * r26;
    r12 = r12 * r6;
    r12 = r12 * r11;
    r12 = fmaf(r20, r12, r28 * r37);
    r12 = fmaf(r22, r25, r12);
    r37 = r6 * r5;
    r25 = fmaf(r7, r25, r26 * r32);
    r32 = r0 * r3;
    r32 = r32 * r9;
    r32 = r32 * r38;
    r25 = fmaf(r28, r32, r25);
    r37 = fmaf(r25, r37, r12 * r2);
    write_sum_3<float, float>((float*)inout_shared, r30, r34, r37);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fmaf(r8, r8, r31 * r31);
    r34 = fmaf(r36, r36, r33 * r33);
    r30 = fmaf(r25, r25, r12 * r12);
    write_sum_3<float, float>((float*)inout_shared, r37, r34, r30);
  };
  flush_sum_shared<3, float>(out_point_precond_diag,
                             0 * out_point_precond_diag_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fmaf(r8, r36, r31 * r33);
    r8 = fmaf(r8, r12, r31 * r25);
    r25 = fmaf(r33, r25, r36 * r12);
    write_sum_3<float, float>((float*)inout_shared, r30, r8, r25);
  };
  flush_sum_shared<3, float>(out_point_precond_tril,
                             0 * out_point_precond_tril_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
}

void simple_radial_fixed_pose_fixed_calib_res_jac(
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
  simple_radial_fixed_pose_fixed_calib_res_jac_kernel<<<n_blocks, 1024>>>(
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