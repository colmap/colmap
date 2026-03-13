#include "kernel_pinhole_fixed_calib_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_calib_res_jac_first_kernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* calib,
        unsigned int calib_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* out_pose_jac,
        unsigned int out_pose_jac_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        float* out_point_jac,
        unsigned int out_point_jac_num_alloc,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        float* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        float* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50;

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
    r31 = copysign(1.0, r9);
    r31 = fmaf(r2, r31, r9);
    r2 = 1.0 / r31;
    r9 = r15 * r23;
    r32 = r16 * r9;
    r19 = r14 * r19;
    r33 = r32 + r19;
    r7 = fmaf(r11, r33, r7);
    r34 = r14 * r16;
    r34 = r34 * r18;
    r24 = r34 + r24;
    r35 = r15 * r9;
    r29 = r35 + r29;
    r7 = fmaf(r12, r24, r7);
    r7 = fmaf(r10, r29, r7);
    r36 = r0 * r7;
    r4 = fmaf(r2, r36, r4);
    r5 = fmaf(r5, r6, r3);
    r3 = r15 * r16;
    r3 = r3 * r18;
    r19 = r3 + r19;
    r8 = fmaf(r10, r19, r8);
    r18 = r13 * r16;
    r18 = r18 * r23;
    r17 = r17 + r18;
    r35 = r27 + r35;
    r35 = r35 + r26;
    r8 = fmaf(r12, r17, r8);
    r8 = fmaf(r11, r35, r8);
    r26 = r1 * r8;
    r5 = fmaf(r2, r26, r5);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5);
    r27 = fmaf(r4, r4, r5 * r5);
  };
  sum_store<float>(out_rTr_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r27);
  if (global_thread_idx < problem_size) {
    r27 = r13 * r14;
    r27 = r27 * r23;
    r3 = r3 + r27;
    r3 = fmaf(r11, r24, r12 * r3);
    r23 = r0 * r3;
    r14 = r14 * r9;
    r18 = r18 + r14;
    r37 = r15 * r15;
    r38 = r6 * r28;
    r39 = r37 + r38;
    r16 = r16 * r16;
    r40 = r13 * r13;
    r40 = r40 * r6;
    r41 = r16 + r40;
    r42 = r39 + r41;
    r42 = fmaf(r11, r42, r12 * r18);
    r18 = r0 * r7;
    r43 = r31 * r31;
    r44 = 1.0 / r43;
    r18 = r18 * r6;
    r18 = r18 * r44;
    r23 = fmaf(r42, r18, r2 * r23);
    r45 = r6 * r44;
    r46 = r42 * r45;
    r47 = r13 * r13;
    r48 = r6 * r16;
    r49 = r47 + r48;
    r39 = r39 + r49;
    r39 = fmaf(r12, r39, r11 * r17);
    r50 = r1 * r39;
    r50 = fmaf(r2, r50, r26 * r46);
    r9 = r13 * r9;
    r22 = r22 + r9;
    r16 = r47 + r16;
    r15 = r15 * r15;
    r15 = r15 * r6;
    r16 = r16 + r38;
    r16 = r16 + r15;
    r16 = fmaf(r12, r16, r10 * r22);
    r22 = r0 * r16;
    r15 = r28 + r15;
    r49 = r49 + r15;
    r49 = fmaf(r10, r49, r12 * r25);
    r22 = fmaf(r49, r18, r2 * r22);
    r14 = r20 + r14;
    r14 = fmaf(r10, r14, r12 * r19);
    r12 = r1 * r14;
    r20 = r49 * r45;
    r20 = fmaf(r26, r20, r2 * r12);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            0 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r23,
                                            r50,
                                            r22,
                                            r20);
    r12 = r0 * r2;
    r38 = r1 * r2;
    r37 = r28 + r37;
    r37 = r37 + r40;
    r37 = r37 + r48;
    r37 = fmaf(r11, r37, r10 * r33);
    r48 = r0 * r37;
    r9 = r34 + r9;
    r9 = fmaf(r11, r9, r10 * r21);
    r48 = fmaf(r9, r18, r2 * r48);
    r34 = r9 * r45;
    r32 = r27 + r32;
    r15 = r41 + r15;
    r15 = fmaf(r10, r15, r11 * r32);
    r10 = r1 * r15;
    r10 = fmaf(r2, r10, r26 * r34);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            4 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r48,
                                            r10,
                                            r12,
                                            r38);
    r38 = r45 * r26;
    write_idx_2<1024, float, float, float2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r18, r38);
    r38 = r6 * r4;
    r12 = r6 * r5;
    r12 = fmaf(r50, r12, r23 * r38);
    r38 = r6 * r4;
    r34 = r6 * r5;
    r34 = fmaf(r20, r34, r22 * r38);
    r38 = r6 * r5;
    r32 = r6 * r4;
    r32 = fmaf(r48, r32, r10 * r38);
    r38 = r0 * r6;
    r38 = r38 * r4;
    r38 = r38 * r2;
    write_sum_4<float, float>((float*)inout_shared, r12, r34, r32, r38);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = r1 * r6;
    r38 = r38 * r5;
    r38 = r38 * r2;
    r32 = r4 * r44;
    r34 = r5 * r44;
    r34 = fmaf(r26, r34, r36 * r32);
    write_sum_2<float, float>((float*)inout_shared, r38, r34);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r0 * r0;
    r34 = r34 * r44;
    r38 = fmaf(r23, r23, r50 * r50);
    r32 = fmaf(r22, r22, r20 * r20);
    r12 = fmaf(r10, r10, r48 * r48);
    write_sum_4<float, float>((float*)inout_shared, r38, r32, r12, r34);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r1 * r1;
    r34 = r34 * r44;
    r12 = r0 * r7;
    r43 = r31 * r43;
    r31 = r31 * r43;
    r31 = 1.0 / r31;
    r12 = r12 * r31;
    r32 = r8 * r31;
    r38 = r1 * r26;
    r32 = fmaf(r38, r32, r36 * r12);
    write_sum_2<float, float>((float*)inout_shared, r34, r32);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fmaf(r50, r20, r23 * r22);
    r34 = fmaf(r50, r10, r23 * r48);
    r12 = r0 * r23;
    r12 = r12 * r2;
    r11 = r1 * r50;
    r11 = r11 * r2;
    write_sum_4<float, float>((float*)inout_shared, r32, r34, r12, r11);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r22, r48, r20 * r10);
    r12 = r0 * r22;
    r12 = r12 * r2;
    r34 = r1 * r20;
    r34 = r34 * r2;
    r32 = r50 * r45;
    r32 = fmaf(r26, r32, r23 * r18);
    write_sum_4<float, float>((float*)inout_shared, r32, r11, r12, r34);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r0 * r48;
    r34 = r34 * r2;
    r12 = r1 * r10;
    r12 = r12 * r2;
    r11 = r20 * r45;
    r22 = fmaf(r22, r18, r26 * r11);
    r11 = r10 * r45;
    r48 = fmaf(r48, r18, r26 * r11);
    write_sum_4<float, float>((float*)inout_shared, r22, r34, r12, r48);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = 0.00000000000000000e+00;
    r36 = r0 * r36;
    r43 = 1.0 / r43;
    r43 = r6 * r43;
    r36 = r36 * r43;
    r38 = r43 * r38;
    write_sum_3<float, float>((float*)inout_shared, r48, r36, r38);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = r0 * r29;
    r38 = fmaf(r25, r18, r2 * r38);
    r36 = r1 * r19;
    r48 = r25 * r45;
    r48 = fmaf(r26, r48, r2 * r36);
    r36 = r0 * r33;
    r36 = fmaf(r21, r18, r2 * r36);
    r43 = r1 * r35;
    r12 = r21 * r45;
    r12 = fmaf(r26, r12, r2 * r43);
    write_idx_4<1024, float, float, float4>(out_point_jac,
                                            0 * out_point_jac_num_alloc,
                                            global_thread_idx,
                                            r38,
                                            r48,
                                            r36,
                                            r12);
    r43 = r0 * r24;
    r18 = fmaf(r30, r18, r2 * r43);
    r43 = r1 * r17;
    r34 = r30 * r45;
    r34 = fmaf(r26, r34, r2 * r43);
    write_idx_2<1024, float, float, float2>(out_point_jac,
                                            4 * out_point_jac_num_alloc,
                                            global_thread_idx,
                                            r18,
                                            r34);
    r43 = r6 * r5;
    r26 = r6 * r4;
    r26 = fmaf(r38, r26, r48 * r43);
    r43 = r6 * r5;
    r2 = r6 * r4;
    r2 = fmaf(r36, r2, r12 * r43);
    r43 = r6 * r5;
    r22 = r6 * r4;
    r22 = fmaf(r18, r22, r34 * r43);
    write_sum_3<float, float>((float*)inout_shared, r26, r2, r22);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = fmaf(r38, r38, r48 * r48);
    r2 = fmaf(r36, r36, r12 * r12);
    r26 = fmaf(r34, r34, r18 * r18);
    write_sum_3<float, float>((float*)inout_shared, r22, r2, r26);
  };
  flush_sum_shared<3, float>(out_point_precond_diag,
                             0 * out_point_precond_diag_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fmaf(r38, r36, r48 * r12);
    r48 = fmaf(r48, r34, r38 * r18);
    r34 = fmaf(r12, r34, r36 * r18);
    write_sum_3<float, float>((float*)inout_shared, r26, r48, r34);
  };
  flush_sum_shared<3, float>(out_point_precond_tril,
                             0 * out_point_precond_tril_num_alloc,
                             point_indices_loc,
                             (float*)inout_shared);
  sum_flush_final<float>(out_rTr_local, out_rTr, 1);
}

void pinhole_fixed_calib_res_jac_first(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* calib,
    unsigned int calib_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    float* out_point_jac,
    unsigned int out_point_jac_num_alloc,
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
  pinhole_fixed_calib_res_jac_first_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      calib,
      calib_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      out_point_jac,
      out_point_jac_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar