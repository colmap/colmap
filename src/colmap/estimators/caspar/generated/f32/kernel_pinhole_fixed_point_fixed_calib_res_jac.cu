#include "kernel_pinhole_fixed_point_fixed_calib_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_point_fixed_calib_res_jac_kernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* calib,
        unsigned int calib_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        calib, 0 * calib_num_alloc, global_thread_idx, r0, r1, r2, r3);
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r4 = fmaf(r4, r6, r2);
    r2 = 9.99999999999999955e-07;
  };
  load_shared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>(
        (float*)inout_shared, pose_indices_loc[threadIdx.x].target, r7, r8, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r10, r11, r12);
  };
  load_shared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         pose_indices_loc[threadIdx.x].target,
                         r13,
                         r14,
                         r15,
                         r16);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r17 = r14 * r15;
    r18 = 2.00000000000000000e+00;
    r17 = r17 * r18;
    r19 = r13 * r18;
    r20 = r16 * r19;
    r21 = r17 + r20;
    r9 = fmaf(r11, r21, r9);
    r22 = r14 * r16;
    r23 = -2.00000000000000000e+00;
    r22 = r22 * r23;
    r24 = r15 * r19;
    r25 = r22 + r24;
    r26 = r13 * r13;
    r26 = r26 * r23;
    r27 = 1.00000000000000000e+00;
    r28 = r14 * r14;
    r29 = fmaf(r23, r28, r27);
    r30 = r26 + r29;
    r9 = fmaf(r10, r25, r9);
    r9 = fmaf(r12, r30, r9);
    r30 = copysign(1.0, r9);
    r30 = fmaf(r2, r30, r9);
    r2 = 1.0 / r30;
    r9 = r15 * r23;
    r31 = r16 * r9;
    r19 = r14 * r19;
    r32 = r31 + r19;
    r7 = fmaf(r11, r32, r7);
    r33 = r14 * r16;
    r33 = r33 * r18;
    r24 = r33 + r24;
    r34 = r15 * r9;
    r29 = r34 + r29;
    r7 = fmaf(r12, r24, r7);
    r7 = fmaf(r10, r29, r7);
    r29 = r0 * r7;
    r4 = fmaf(r2, r29, r4);
    r5 = fmaf(r5, r6, r3);
    r3 = r15 * r16;
    r3 = r3 * r18;
    r19 = r3 + r19;
    r8 = fmaf(r10, r19, r8);
    r18 = r13 * r16;
    r18 = r18 * r23;
    r17 = r17 + r18;
    r34 = r27 + r34;
    r34 = r34 + r26;
    r8 = fmaf(r12, r17, r8);
    r8 = fmaf(r11, r34, r8);
    r34 = r1 * r8;
    r5 = fmaf(r2, r34, r5);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5);
    r26 = r6 * r4;
    r27 = r13 * r14;
    r27 = r27 * r23;
    r3 = r3 + r27;
    r24 = fmaf(r11, r24, r12 * r3);
    r3 = r0 * r24;
    r14 = r14 * r9;
    r18 = r18 + r14;
    r23 = r15 * r15;
    r35 = r6 * r28;
    r36 = r23 + r35;
    r16 = r16 * r16;
    r37 = r13 * r13;
    r37 = r37 * r6;
    r38 = r16 + r37;
    r39 = r36 + r38;
    r39 = fmaf(r11, r39, r12 * r18);
    r18 = r30 * r30;
    r40 = 1.0 / r18;
    r41 = r6 * r40;
    r42 = r41 * r29;
    r3 = fmaf(r39, r42, r2 * r3);
    r43 = r6 * r5;
    r44 = r39 * r41;
    r45 = r13 * r13;
    r46 = r6 * r16;
    r47 = r45 + r46;
    r36 = r36 + r47;
    r36 = fmaf(r12, r36, r11 * r17);
    r17 = r1 * r36;
    r17 = fmaf(r2, r17, r34 * r44);
    r43 = fmaf(r17, r43, r3 * r26);
    r26 = r6 * r4;
    r9 = r13 * r9;
    r22 = r22 + r9;
    r16 = r45 + r16;
    r15 = r15 * r15;
    r15 = r15 * r6;
    r16 = r16 + r35;
    r16 = r16 + r15;
    r16 = fmaf(r12, r16, r10 * r22);
    r22 = r0 * r16;
    r15 = r28 + r15;
    r47 = r47 + r15;
    r47 = fmaf(r10, r47, r12 * r25);
    r22 = fmaf(r47, r42, r2 * r22);
    r25 = r6 * r5;
    r14 = r20 + r14;
    r14 = fmaf(r10, r14, r12 * r19);
    r19 = r1 * r14;
    r12 = r47 * r41;
    r12 = fmaf(r34, r12, r2 * r19);
    r25 = fmaf(r12, r25, r22 * r26);
    r26 = r6 * r5;
    r9 = r33 + r9;
    r9 = fmaf(r11, r9, r10 * r21);
    r21 = r9 * r41;
    r31 = r27 + r31;
    r15 = r38 + r15;
    r15 = fmaf(r10, r15, r11 * r31);
    r31 = r1 * r15;
    r31 = fmaf(r2, r31, r34 * r21);
    r21 = r6 * r4;
    r23 = r28 + r23;
    r23 = r23 + r37;
    r23 = r23 + r46;
    r23 = fmaf(r11, r23, r10 * r32);
    r11 = r0 * r23;
    r11 = fmaf(r9, r42, r2 * r11);
    r21 = fmaf(r11, r21, r31 * r26);
    r26 = r0 * r6;
    r26 = r26 * r4;
    r26 = r26 * r2;
    write_sum_4<float, float>((float*)inout_shared, r43, r25, r21, r26);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = r1 * r6;
    r26 = r26 * r5;
    r26 = r26 * r2;
    r21 = r4 * r40;
    r25 = r5 * r40;
    r25 = fmaf(r34, r25, r29 * r21);
    write_sum_2<float, float>((float*)inout_shared, r26, r25);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r0 * r0;
    r25 = r25 * r40;
    r26 = fmaf(r3, r3, r17 * r17);
    r21 = fmaf(r22, r22, r12 * r12);
    r43 = fmaf(r31, r31, r11 * r11);
    write_sum_4<float, float>((float*)inout_shared, r26, r21, r43, r25);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r1 * r1;
    r25 = r25 * r40;
    r43 = r0 * r7;
    r18 = r30 * r18;
    r30 = r30 * r18;
    r30 = 1.0 / r30;
    r43 = r43 * r30;
    r21 = r8 * r30;
    r26 = r1 * r34;
    r21 = fmaf(r26, r21, r29 * r43);
    write_sum_2<float, float>((float*)inout_shared, r25, r21);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = fmaf(r17, r12, r3 * r22);
    r25 = fmaf(r17, r31, r3 * r11);
    r43 = r0 * r3;
    r43 = r43 * r2;
    r32 = r1 * r17;
    r32 = r32 * r2;
    write_sum_4<float, float>((float*)inout_shared, r21, r25, r43, r32);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fmaf(r22, r11, r12 * r31);
    r43 = r0 * r22;
    r43 = r43 * r2;
    r25 = r1 * r12;
    r25 = r25 * r2;
    r21 = r17 * r41;
    r21 = fmaf(r34, r21, r3 * r42);
    write_sum_4<float, float>((float*)inout_shared, r21, r32, r43, r25);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r0 * r11;
    r25 = r25 * r2;
    r43 = r1 * r31;
    r43 = r43 * r2;
    r2 = r12 * r41;
    r22 = fmaf(r22, r42, r34 * r2);
    r2 = r31 * r41;
    r42 = fmaf(r11, r42, r34 * r2);
    write_sum_4<float, float>((float*)inout_shared, r22, r25, r43, r42);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = 0.00000000000000000e+00;
    r29 = r0 * r29;
    r18 = 1.0 / r18;
    r18 = r6 * r18;
    r29 = r29 * r18;
    r26 = r18 * r26;
    write_sum_3<float, float>((float*)inout_shared, r42, r29, r26);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
}

void pinhole_fixed_point_fixed_calib_res_jac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* calib,
    unsigned int calib_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_fixed_point_fixed_calib_res_jac_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      pixel,
      pixel_num_alloc,
      calib,
      calib_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar