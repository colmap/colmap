#include "kernel_simple_radial_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_point_res_jac_kernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* out_pose_jac,
        unsigned int out_pose_jac_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        float* out_calib_jac,
        unsigned int out_calib_jac_num_alloc,
        float* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        float* const out_calib_precond_diag,
        unsigned int out_calib_precond_diag_num_alloc,
        float* const out_calib_precond_tril,
        unsigned int out_calib_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50;
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
    r25 = fmaf(r3, r35, r26);
    r39 = 1.0 / r28;
    r40 = r25 * r39;
    r4 = fmaf(r29, r40, r4);
    r5 = fmaf(r5, r6, r2);
    r2 = r0 * r25;
    r2 = r2 * r39;
    r5 = fmaf(r7, r2, r5);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5);
    r41 = r17 * r7;
    r42 = r14 * r14;
    r15 = r15 * r15;
    r43 = r6 * r15;
    r44 = r42 + r43;
    r45 = r6 * r27;
    r46 = r34 + r45;
    r47 = r44 + r46;
    r47 = fmaf(r11, r47, r10 * r33);
    r41 = r41 * r47;
    r33 = r18 * r7;
    r48 = r13 * r19;
    r32 = r32 + r48;
    r45 = r42 + r45;
    r42 = r6 * r34;
    r49 = r15 + r42;
    r45 = r45 + r49;
    r45 = fmaf(r10, r45, r11 * r32);
    r36 = r28 * r36;
    r28 = 1.0 / r36;
    r33 = r33 * r7;
    r33 = r33 * r45;
    r33 = fmaf(r28, r33, r8 * r41);
    r41 = r18 * r1;
    r41 = r41 * r1;
    r41 = r41 * r45;
    r33 = fmaf(r28, r41, r33);
    r32 = r17 * r1;
    r13 = r12 * r13;
    r13 = r13 * r18;
    r30 = r30 + r13;
    r24 = fmaf(r10, r24, r11 * r30);
    r32 = r32 * r24;
    r33 = fmaf(r8, r32, r33);
    r32 = r3 * r33;
    r32 = r32 * r39;
    r24 = fmaf(r24, r2, r29 * r32);
    r32 = r45 * r29;
    r41 = r25 * r6;
    r41 = r41 * r8;
    r24 = fmaf(r41, r32, r24);
    r32 = r0 * r33;
    r30 = r3 * r7;
    r32 = r32 * r39;
    r50 = r0 * r7;
    r50 = r50 * r8;
    r50 = r50 * r25;
    r50 = r50 * r6;
    r32 = fmaf(r45, r50, r30 * r32);
    r32 = fmaf(r47, r2, r32);
    r47 = r17 * r7;
    r48 = r37 + r48;
    r48 = fmaf(r9, r48, r11 * r16);
    r47 = r47 * r48;
    r16 = r17 * r1;
    r19 = r12 * r19;
    r38 = r38 + r19;
    r14 = r14 * r14;
    r14 = r14 * r6;
    r15 = r15 + r14;
    r15 = r15 + r46;
    r15 = fmaf(r11, r15, r9 * r38);
    r16 = r16 * r15;
    r16 = fmaf(r8, r16, r8 * r47);
    r47 = r18 * r1;
    r43 = r34 + r43;
    r14 = r27 + r14;
    r43 = r43 + r14;
    r43 = fmaf(r9, r43, r11 * r22);
    r47 = r47 * r1;
    r47 = r47 * r43;
    r16 = fmaf(r28, r47, r16);
    r22 = r18 * r7;
    r22 = r22 * r7;
    r22 = r22 * r43;
    r16 = fmaf(r28, r22, r16);
    r22 = r3 * r16;
    r22 = r22 * r39;
    r47 = r43 * r29;
    r47 = fmaf(r41, r47, r29 * r22);
    r47 = fmaf(r15, r2, r47);
    r15 = r0 * r16;
    r15 = r15 * r39;
    r15 = fmaf(r43, r50, r30 * r15);
    r15 = fmaf(r48, r2, r15);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            0 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r24,
                                            r32,
                                            r47,
                                            r15);
    r48 = r0 * r3;
    r48 = r48 * r17;
    r48 = r48 * r1;
    r48 = r48 * r7;
    r48 = r48 * r28;
    r22 = r17 * r1;
    r42 = r27 + r42;
    r42 = r42 + r44;
    r42 = fmaf(r10, r42, r9 * r21);
    r22 = r22 * r42;
    r21 = r18 * r7;
    r19 = r23 + r19;
    r19 = fmaf(r10, r19, r9 * r31);
    r21 = r21 * r7;
    r21 = r21 * r19;
    r21 = fmaf(r28, r21, r8 * r22);
    r22 = r18 * r1;
    r22 = r22 * r1;
    r22 = r22 * r19;
    r21 = fmaf(r28, r22, r21);
    r31 = r17 * r7;
    r20 = r13 + r20;
    r14 = r49 + r14;
    r14 = fmaf(r9, r14, r10 * r20);
    r31 = r31 * r14;
    r21 = fmaf(r8, r31, r21);
    r31 = r3 * r21;
    r31 = r31 * r39;
    r42 = fmaf(r42, r2, r29 * r31);
    r31 = r19 * r29;
    r42 = fmaf(r41, r31, r42);
    r31 = r0 * r21;
    r31 = r31 * r39;
    r31 = fmaf(r30, r31, r14 * r2);
    r31 = fmaf(r19, r50, r31);
    r14 = r28 * r29;
    r22 = r17 * r14;
    r9 = r3 * r1;
    r22 = fmaf(r9, r22, r2);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            4 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r42,
                                            r31,
                                            r22,
                                            r48);
    r20 = r0 * r17;
    r20 = r20 * r7;
    r20 = r20 * r28;
    r20 = fmaf(r30, r20, r2);
    r2 = r18 * r7;
    r2 = r2 * r7;
    r10 = r18 * r1;
    r10 = r10 * r1;
    r10 = fmaf(r28, r10, r28 * r2);
    r2 = r3 * r10;
    r2 = r2 * r39;
    r41 = fmaf(r29, r41, r29 * r2);
    r2 = r0 * r10;
    r2 = r2 * r39;
    r2 = fmaf(r30, r2, r50);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            8 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r48,
                                            r20,
                                            r41,
                                            r2);
    r50 = r6 * r4;
    r28 = r6 * r5;
    r28 = fmaf(r32, r28, r24 * r50);
    r50 = r6 * r4;
    r49 = r6 * r5;
    r49 = fmaf(r15, r49, r47 * r50);
    r50 = r6 * r4;
    r13 = r6 * r5;
    r13 = fmaf(r31, r13, r42 * r50);
    r50 = r6 * r4;
    r23 = r18 * r5;
    r44 = r30 * r14;
    r23 = fmaf(r44, r23, r22 * r50);
    write_sum_4<float, float>((float*)inout_shared, r28, r49, r13, r23);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = r6 * r5;
    r13 = r6 * r4;
    r13 = fmaf(r41, r13, r2 * r23);
    r23 = r6 * r5;
    r49 = r18 * r4;
    r49 = fmaf(r44, r49, r20 * r23);
    write_sum_2<float, float>((float*)inout_shared, r49, r13);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = fmaf(r32, r32, r24 * r24);
    r49 = fmaf(r15, r15, r47 * r47);
    r23 = fmaf(r31, r31, r42 * r42);
    r44 = 4.00000000000000000e+00;
    r36 = r36 * r36;
    r36 = 1.0 / r36;
    r36 = r44 * r36;
    r44 = r0 * r7;
    r36 = r36 * r29;
    r36 = r36 * r30;
    r36 = r36 * r9;
    r36 = r36 * r44;
    r9 = fmaf(r22, r22, r36);
    write_sum_4<float, float>((float*)inout_shared, r13, r49, r23, r9);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = fmaf(r20, r20, r36);
    r9 = fmaf(r41, r41, r2 * r2);
    write_sum_2<float, float>((float*)inout_shared, r36, r9);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = fmaf(r24, r47, r32 * r15);
    r36 = fmaf(r32, r31, r24 * r42);
    r23 = fmaf(r32, r48, r24 * r22);
    r49 = fmaf(r24, r48, r32 * r20);
    write_sum_4<float, float>((float*)inout_shared, r9, r36, r23, r49);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fmaf(r32, r2, r24 * r41);
    r24 = fmaf(r47, r42, r15 * r31);
    r49 = fmaf(r15, r48, r47 * r22);
    r23 = fmaf(r47, r48, r15 * r20);
    write_sum_4<float, float>((float*)inout_shared, r32, r24, r49, r23);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fmaf(r47, r41, r15 * r2);
    r15 = fmaf(r31, r2, r42 * r41);
    r23 = fmaf(r31, r48, r42 * r22);
    r42 = fmaf(r42, r48, r31 * r20);
    write_sum_4<float, float>((float*)inout_shared, r47, r23, r42, r15);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r20, r48, r22 * r48);
    r22 = fmaf(r2, r48, r22 * r41);
    r48 = fmaf(r41, r48, r20 * r2);
    write_sum_3<float, float>((float*)inout_shared, r15, r22, r48);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r1 * r40;
    r22 = r7 * r40;
    r15 = r35 * r39;
    r15 = r15 * r29;
    r41 = r0 * r7;
    r41 = r41 * r35;
    r41 = r41 * r39;
    write_idx_4<1024, float, float, float4>(out_calib_jac,
                                            0 * out_calib_jac_num_alloc,
                                            global_thread_idx,
                                            r48,
                                            r22,
                                            r15,
                                            r41);
    r2 = r6 * r4;
    r20 = r6 * r5;
    r42 = r7 * r6;
    r42 = r42 * r5;
    r23 = r1 * r6;
    r23 = r23 * r4;
    r23 = fmaf(r40, r23, r40 * r42);
    r42 = r0 * r7;
    r42 = r42 * r35;
    r42 = r42 * r6;
    r42 = r42 * r5;
    r40 = r35 * r6;
    r40 = r40 * r4;
    r40 = r40 * r39;
    r40 = fmaf(r29, r40, r39 * r42);
    write_sum_4<float, float>((float*)inout_shared, r23, r2, r20, r40);
  };
  flush_sum_shared<4, float>(out_calib_njtr,
                             0 * out_calib_njtr_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r7 * r7;
    r40 = r40 * r25;
    r40 = r40 * r25;
    r20 = r1 * r1;
    r20 = r20 * r25;
    r20 = r20 * r25;
    r20 = fmaf(r8, r20, r8 * r40);
    r40 = r0 * r8;
    r2 = r35 * r35;
    r40 = r40 * r2;
    r2 = r7 * r40;
    r23 = r1 * r29;
    r23 = fmaf(r40, r23, r44 * r2);
    write_sum_4<float, float>((float*)inout_shared, r20, r26, r26, r23);
  };
  flush_sum_shared<4, float>(out_calib_precond_diag,
                             0 * out_calib_precond_diag_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = 0.00000000000000000e+00;
    r26 = r1 * r35;
    r26 = r26 * r25;
    r26 = r26 * r8;
    r20 = r0 * r7;
    r20 = r20 * r7;
    r20 = r20 * r35;
    r20 = r20 * r25;
    r20 = fmaf(r8, r20, r29 * r26);
    write_sum_4<float, float>((float*)inout_shared, r48, r22, r20, r23);
  };
  flush_sum_shared<4, float>(out_calib_precond_tril,
                             0 * out_calib_precond_tril_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<float, float>((float*)inout_shared, r15, r41);
  };
  flush_sum_shared<2, float>(out_calib_precond_tril,
                             4 * out_calib_precond_tril_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
}

void simple_radial_fixed_point_res_jac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    float* out_calib_jac,
    unsigned int out_calib_jac_num_alloc,
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
  simple_radial_fixed_point_res_jac_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      out_calib_jac,
      out_calib_jac_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar