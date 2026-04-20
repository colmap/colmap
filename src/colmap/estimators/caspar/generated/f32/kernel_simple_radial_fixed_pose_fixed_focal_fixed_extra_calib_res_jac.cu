#include "kernel_simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_res_jac_kernel(
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        float* extra_calib,
        unsigned int extra_calib_num_alloc,
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
    read_idx_3<1024, float, float, float4>(
        extra_calib, 0 * extra_calib_num_alloc, global_thread_idx, r0, r1, r2);
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r3, r4);
    r5 = -1.00000000000000000e+00;
    r3 = fmaf(r3, r5, r0);
    read_idx_3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r0, r6, r7);
  };
  load_shared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>((float*)inout_shared,
                         point_indices_loc[threadIdx.x].target,
                         r8,
                         r9,
                         r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r11 = -2.00000000000000000e+00;
    read_idx_4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r12, r13, r14, r15);
    r16 = r14 * r15;
    r17 = 2.00000000000000000e+00;
    r18 = r12 * r17;
    r19 = r13 * r18;
    r20 = fmaf(r11, r16, r19);
    r0 = fmaf(r9, r20, r0);
    r21 = r13 * r15;
    r22 = r14 * r18;
    r21 = fmaf(r17, r21, r22);
    r23 = r14 * r14;
    r23 = r23 * r11;
    r24 = 1.00000000000000000e+00;
    r25 = r13 * r13;
    r25 = fmaf(r11, r25, r24);
    r26 = r23 + r25;
    r0 = fmaf(r10, r21, r0);
    r0 = fmaf(r8, r26, r0);
    r27 = 9.99999999999999955e-07;
    r14 = r13 * r14;
    r14 = r14 * r17;
    r18 = fmaf(r15, r18, r14);
    r7 = fmaf(r9, r18, r7);
    r28 = r13 * r15;
    r28 = fmaf(r11, r28, r22);
    r22 = r12 * r12;
    r22 = r22 * r11;
    r25 = r22 + r25;
    r7 = fmaf(r8, r28, r7);
    r7 = fmaf(r10, r25, r7);
    r29 = copysign(1.0, r7);
    r29 = fmaf(r27, r29, r7);
    r27 = 1.0 / r29;
    read_idx_1<1024, float, float, float>(
        focal, 0 * focal_num_alloc, global_thread_idx, r7);
    r16 = fmaf(r17, r16, r19);
    r8 = fmaf(r8, r16, r6);
    r6 = r12 * r15;
    r6 = fmaf(r11, r6, r14);
    r23 = r24 + r23;
    r23 = r23 + r22;
    r8 = fmaf(r10, r6, r8);
    r8 = fmaf(r9, r23, r8);
    r9 = r29 * r29;
    r10 = 1.0 / r9;
    r22 = r8 * r10;
    r14 = r0 * r0;
    r19 = fmaf(r10, r14, r8 * r22);
    r19 = fmaf(r2, r19, r24);
    r19 = r7 * r19;
    r24 = r27 * r19;
    r3 = fmaf(r0, r24, r3);
    r4 = fmaf(r4, r5, r1);
    r4 = fmaf(r8, r24, r4);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r3, r4);
    r1 = r5 * r3;
    r30 = r7 * r2;
    r9 = r29 * r9;
    r9 = 1.0 / r9;
    r9 = r11 * r9;
    r11 = r28 * r9;
    r29 = r17 * r26;
    r29 = r29 * r0;
    r29 = fmaf(r10, r29, r14 * r11);
    r11 = r8 * r8;
    r11 = r11 * r9;
    r31 = r17 * r16;
    r29 = fmaf(r22, r31, r29);
    r29 = fmaf(r28, r11, r29);
    r30 = r30 * r29;
    r30 = r30 * r27;
    r29 = fmaf(r26, r24, r0 * r30);
    r31 = r0 * r28;
    r31 = r31 * r5;
    r31 = r31 * r10;
    r29 = fmaf(r19, r31, r29);
    r31 = r5 * r4;
    r32 = r5 * r19;
    r32 = r32 * r22;
    r30 = fmaf(r8, r30, r28 * r32);
    r30 = fmaf(r16, r24, r30);
    r31 = fmaf(r30, r31, r29 * r1);
    r1 = r5 * r3;
    r33 = r7 * r2;
    r34 = r17 * r20;
    r34 = r34 * r0;
    r35 = r18 * r9;
    r35 = fmaf(r14, r35, r10 * r34);
    r34 = r17 * r23;
    r35 = fmaf(r22, r34, r35);
    r35 = fmaf(r18, r11, r35);
    r33 = r33 * r0;
    r33 = r33 * r35;
    r33 = fmaf(r20, r24, r27 * r33);
    r34 = r0 * r18;
    r34 = r34 * r5;
    r34 = r34 * r10;
    r33 = fmaf(r19, r34, r33);
    r34 = r5 * r4;
    r36 = fmaf(r18, r32, r23 * r24);
    r37 = r7 * r2;
    r37 = r37 * r8;
    r37 = r37 * r35;
    r36 = fmaf(r27, r37, r36);
    r34 = fmaf(r36, r34, r33 * r1);
    r1 = r5 * r3;
    r37 = r0 * r25;
    r37 = r37 * r5;
    r37 = r37 * r10;
    r37 = fmaf(r19, r37, r21 * r24);
    r19 = r7 * r2;
    r35 = r17 * r21;
    r35 = r35 * r0;
    r38 = r25 * r9;
    r38 = fmaf(r14, r38, r10 * r35);
    r35 = r17 * r6;
    r38 = fmaf(r22, r35, r38);
    r38 = fmaf(r25, r11, r38);
    r19 = r19 * r0;
    r19 = r19 * r38;
    r37 = fmaf(r27, r19, r37);
    r19 = r5 * r4;
    r24 = fmaf(r6, r24, r25 * r32);
    r32 = r7 * r2;
    r32 = r32 * r8;
    r32 = r32 * r38;
    r24 = fmaf(r27, r32, r24);
    r19 = fmaf(r24, r19, r37 * r1);
    write_sum_3<float, float>((float*)inout_shared, r31, r34, r19);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fmaf(r30, r30, r29 * r29);
    r34 = fmaf(r36, r36, r33 * r33);
    r31 = fmaf(r24, r24, r37 * r37);
    write_sum_3<float, float>((float*)inout_shared, r19, r34, r31);
  };
  flush_sum_shared<3, float>(out_point_precond_diag,
                             0 * out_point_precond_diag_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r30, r36, r29 * r33);
    r30 = fmaf(r30, r24, r29 * r37);
    r24 = fmaf(r36, r24, r33 * r37);
    write_sum_3<float, float>((float*)inout_shared, r31, r30, r24);
  };
  flush_sum_shared<3, float>(out_point_precond_tril,
                             0 * out_point_precond_tril_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
}

void simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_res_jac(
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* extra_calib,
    unsigned int extra_calib_num_alloc,
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
  simple_radial_fixed_pose_fixed_focal_fixed_extra_calib_res_jac_kernel<<<
      n_blocks,
      1024>>>(point,
              point_num_alloc,
              point_indices,
              pixel,
              pixel_num_alloc,
              pose,
              pose_num_alloc,
              focal,
              focal_num_alloc,
              extra_calib,
              extra_calib_num_alloc,
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