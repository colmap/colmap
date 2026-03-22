#include "kernel_simple_radial_fixed_point_fixed_calib_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_point_fixed_calib_res_jac_kernel(
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
      r46, r47, r48, r49;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        calib, 0 * calib_num_alloc, global_thread_idx, r0, r1, r2, r3);
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r4 = fmaf(r4, r6, r1);
  };
  load_shared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>(
        (float*)inout_shared, pose_indices_loc[threadIdx.x].target, r1, r7, r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r9, r10, r11);
  };
  load_shared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         pose_indices_loc[threadIdx.x].target,
                         r12,
                         r13,
                         r14,
                         r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = r12 * r13;
    r17 = 2.00000000000000000e+00;
    r16 = r16 * r17;
    r18 = -2.00000000000000000e+00;
    r19 = r14 * r18;
    r20 = r15 * r19;
    r21 = r16 + r20;
    r1 = fmaf(r10, r21, r1);
    r22 = r12 * r14;
    r22 = r22 * r17;
    r23 = r13 * r15;
    r23 = r23 * r17;
    r24 = r22 + r23;
    r25 = r14 * r19;
    r26 = 1.00000000000000000e+00;
    r27 = r13 * r13;
    r28 = fmaf(r18, r27, r26);
    r29 = r25 + r28;
    r1 = fmaf(r11, r24, r1);
    r1 = fmaf(r9, r29, r1);
    r29 = r0 * r1;
    r30 = r14 * r15;
    r30 = r30 * r17;
    r16 = r16 + r30;
    r7 = fmaf(r9, r16, r7);
    r31 = r13 * r14;
    r31 = r31 * r17;
    r32 = r12 * r15;
    r32 = r32 * r18;
    r33 = r31 + r32;
    r25 = r26 + r25;
    r34 = r12 * r12;
    r35 = r18 * r34;
    r25 = r25 + r35;
    r7 = fmaf(r11, r33, r7);
    r7 = fmaf(r10, r25, r7);
    r25 = r7 * r7;
    r36 = 9.99999999999999955e-07;
    r37 = r12 * r15;
    r37 = r37 * r17;
    r31 = r31 + r37;
    r8 = fmaf(r10, r31, r8);
    r38 = r13 * r15;
    r38 = r38 * r18;
    r22 = r22 + r38;
    r28 = r35 + r28;
    r8 = fmaf(r9, r22, r8);
    r8 = fmaf(r11, r28, r8);
    r28 = copysign(1.0, r8);
    r28 = fmaf(r36, r28, r8);
    r36 = r28 * r28;
    r8 = 1.0 / r36;
    r35 = r1 * r1;
    r35 = fmaf(r8, r35, r8 * r25);
    r35 = fmaf(r3, r35, r26);
    r26 = 1.0 / r28;
    r25 = r35 * r26;
    r4 = fmaf(r29, r25, r4);
    r5 = fmaf(r5, r6, r2);
    r2 = r0 * r35;
    r2 = r2 * r26;
    r5 = fmaf(r7, r2, r5);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5);
    r25 = r6 * r4;
    r39 = r17 * r7;
    r40 = r14 * r14;
    r15 = r15 * r15;
    r41 = r6 * r15;
    r42 = r40 + r41;
    r43 = r6 * r27;
    r44 = r34 + r43;
    r45 = r42 + r44;
    r45 = fmaf(r11, r45, r10 * r33);
    r39 = r39 * r45;
    r33 = r18 * r7;
    r46 = r13 * r19;
    r32 = r32 + r46;
    r43 = r40 + r43;
    r40 = r6 * r34;
    r47 = r15 + r40;
    r43 = r43 + r47;
    r43 = fmaf(r10, r43, r11 * r32);
    r36 = r28 * r36;
    r28 = 1.0 / r36;
    r33 = r33 * r7;
    r33 = r33 * r43;
    r33 = fmaf(r28, r33, r8 * r39);
    r39 = r18 * r1;
    r39 = r39 * r1;
    r39 = r39 * r43;
    r33 = fmaf(r28, r39, r33);
    r32 = r17 * r1;
    r13 = r12 * r13;
    r13 = r13 * r18;
    r30 = r30 + r13;
    r24 = fmaf(r10, r24, r11 * r30);
    r32 = r32 * r24;
    r33 = fmaf(r8, r32, r33);
    r32 = r3 * r33;
    r32 = r32 * r26;
    r24 = fmaf(r24, r2, r29 * r32);
    r32 = r43 * r29;
    r39 = r35 * r6;
    r39 = r39 * r8;
    r24 = fmaf(r39, r32, r24);
    r32 = r6 * r5;
    r30 = r0 * r33;
    r48 = r3 * r7;
    r30 = r30 * r26;
    r49 = r0 * r7;
    r49 = r49 * r8;
    r49 = r49 * r35;
    r49 = r49 * r6;
    r30 = fmaf(r43, r49, r48 * r30);
    r30 = fmaf(r45, r2, r30);
    r32 = fmaf(r30, r32, r24 * r25);
    r25 = r6 * r4;
    r45 = r17 * r7;
    r46 = r37 + r46;
    r46 = fmaf(r9, r46, r11 * r16);
    r45 = r45 * r46;
    r16 = r17 * r1;
    r19 = r12 * r19;
    r38 = r38 + r19;
    r14 = r14 * r14;
    r14 = r14 * r6;
    r15 = r15 + r14;
    r15 = r15 + r44;
    r15 = fmaf(r11, r15, r9 * r38);
    r16 = r16 * r15;
    r16 = fmaf(r8, r16, r8 * r45);
    r45 = r18 * r1;
    r41 = r34 + r41;
    r14 = r27 + r14;
    r41 = r41 + r14;
    r41 = fmaf(r9, r41, r11 * r22);
    r45 = r45 * r1;
    r45 = r45 * r41;
    r16 = fmaf(r28, r45, r16);
    r22 = r18 * r7;
    r22 = r22 * r7;
    r22 = r22 * r41;
    r16 = fmaf(r28, r22, r16);
    r22 = r3 * r16;
    r22 = r22 * r26;
    r45 = r41 * r29;
    r45 = fmaf(r39, r45, r29 * r22);
    r45 = fmaf(r15, r2, r45);
    r15 = r6 * r5;
    r22 = r0 * r16;
    r22 = r22 * r26;
    r22 = fmaf(r41, r49, r48 * r22);
    r22 = fmaf(r46, r2, r22);
    r15 = fmaf(r22, r15, r45 * r25);
    r25 = r6 * r4;
    r46 = r17 * r1;
    r40 = r27 + r40;
    r40 = r40 + r42;
    r40 = fmaf(r10, r40, r9 * r21);
    r46 = r46 * r40;
    r21 = r18 * r7;
    r19 = r23 + r19;
    r19 = fmaf(r10, r19, r9 * r31);
    r21 = r21 * r7;
    r21 = r21 * r19;
    r21 = fmaf(r28, r21, r8 * r46);
    r46 = r18 * r1;
    r46 = r46 * r1;
    r46 = r46 * r19;
    r21 = fmaf(r28, r46, r21);
    r31 = r17 * r7;
    r20 = r13 + r20;
    r14 = r47 + r14;
    r14 = fmaf(r9, r14, r10 * r20);
    r31 = r31 * r14;
    r21 = fmaf(r8, r31, r21);
    r31 = r3 * r21;
    r31 = r31 * r26;
    r40 = fmaf(r40, r2, r29 * r31);
    r31 = r19 * r29;
    r40 = fmaf(r39, r31, r40);
    r31 = r6 * r5;
    r46 = r0 * r21;
    r46 = r46 * r26;
    r46 = fmaf(r48, r46, r14 * r2);
    r46 = fmaf(r19, r49, r46);
    r31 = fmaf(r46, r31, r40 * r25);
    r25 = r6 * r4;
    r14 = r28 * r29;
    r8 = r17 * r14;
    r9 = r3 * r1;
    r8 = fmaf(r9, r8, r2);
    r20 = r18 * r5;
    r10 = r48 * r14;
    r20 = fmaf(r10, r20, r8 * r25);
    write_sum_4<float, float>((float*)inout_shared, r32, r15, r31, r20);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r6 * r5;
    r31 = r18 * r7;
    r31 = r31 * r7;
    r15 = r18 * r1;
    r15 = r15 * r1;
    r15 = fmaf(r28, r15, r28 * r31);
    r31 = r0 * r15;
    r31 = r31 * r26;
    r31 = fmaf(r48, r31, r49);
    r49 = r6 * r4;
    r32 = r3 * r15;
    r32 = r32 * r26;
    r39 = fmaf(r29, r39, r29 * r32);
    r49 = fmaf(r39, r49, r31 * r20);
    r20 = r6 * r5;
    r32 = r0 * r17;
    r32 = r32 * r7;
    r32 = r32 * r28;
    r32 = fmaf(r48, r32, r2);
    r2 = r18 * r4;
    r2 = fmaf(r10, r2, r32 * r20);
    write_sum_2<float, float>((float*)inout_shared, r2, r49);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = fmaf(r30, r30, r24 * r24);
    r2 = fmaf(r22, r22, r45 * r45);
    r20 = fmaf(r46, r46, r40 * r40);
    r28 = r0 * r7;
    r26 = 4.00000000000000000e+00;
    r36 = r36 * r36;
    r36 = 1.0 / r36;
    r28 = r28 * r26;
    r28 = r28 * r36;
    r28 = r28 * r29;
    r28 = r28 * r48;
    r28 = r28 * r9;
    r9 = fmaf(r8, r8, r28);
    write_sum_4<float, float>((float*)inout_shared, r49, r2, r20, r9);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = fmaf(r32, r32, r28);
    r9 = fmaf(r39, r39, r31 * r31);
    write_sum_2<float, float>((float*)inout_shared, r28, r9);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = fmaf(r24, r45, r30 * r22);
    r28 = fmaf(r30, r46, r24 * r40);
    r10 = r17 * r10;
    r20 = fmaf(r30, r10, r24 * r8);
    r2 = fmaf(r24, r10, r30 * r32);
    write_sum_4<float, float>((float*)inout_shared, r9, r28, r20, r2);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fmaf(r30, r31, r24 * r39);
    r24 = fmaf(r45, r40, r22 * r46);
    r2 = fmaf(r22, r10, r45 * r8);
    r20 = fmaf(r45, r10, r22 * r32);
    write_sum_4<float, float>((float*)inout_shared, r30, r24, r2, r20);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fmaf(r45, r39, r22 * r31);
    r22 = fmaf(r46, r31, r40 * r39);
    r20 = fmaf(r46, r10, r40 * r8);
    r40 = fmaf(r40, r10, r46 * r32);
    write_sum_4<float, float>((float*)inout_shared, r45, r20, r40, r22);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = fmaf(r32, r10, r8 * r10);
    r8 = fmaf(r31, r10, r8 * r39);
    r10 = fmaf(r39, r10, r32 * r31);
    write_sum_3<float, float>((float*)inout_shared, r22, r8, r10);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
}

void simple_radial_fixed_point_fixed_calib_res_jac(
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
  simple_radial_fixed_point_fixed_calib_res_jac_kernel<<<n_blocks, 1024>>>(
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