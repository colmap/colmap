#include "kernel_pinhole_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_point_res_jac_first_kernel(
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
        float* const out_rTr,
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

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46;
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
    r26 = fmaf(r4, r4, r5 * r5);
  };
  sum_store<float>(out_rTr_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r26);
  if (global_thread_idx < problem_size) {
    r26 = r13 * r14;
    r26 = r26 * r23;
    r3 = r3 + r26;
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
    r18 = r0 * r7;
    r40 = r30 * r30;
    r41 = 1.0 / r40;
    r18 = r18 * r6;
    r18 = r18 * r41;
    r3 = fmaf(r39, r18, r2 * r3);
    r42 = r6 * r41;
    r43 = r39 * r42;
    r44 = r13 * r13;
    r45 = r6 * r16;
    r46 = r44 + r45;
    r36 = r36 + r46;
    r36 = fmaf(r12, r36, r11 * r17);
    r17 = r1 * r36;
    r17 = fmaf(r2, r17, r34 * r43);
    r9 = r13 * r9;
    r22 = r22 + r9;
    r16 = r44 + r16;
    r15 = r15 * r15;
    r15 = r15 * r6;
    r16 = r16 + r35;
    r16 = r16 + r15;
    r16 = fmaf(r12, r16, r10 * r22);
    r22 = r0 * r16;
    r15 = r28 + r15;
    r46 = r46 + r15;
    r46 = fmaf(r10, r46, r12 * r25);
    r22 = fmaf(r46, r18, r2 * r22);
    r14 = r20 + r14;
    r14 = fmaf(r10, r14, r12 * r19);
    r19 = r1 * r14;
    r12 = r46 * r42;
    r12 = fmaf(r34, r12, r2 * r19);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            0 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r3,
                                            r17,
                                            r22,
                                            r12);
    r19 = r0 * r2;
    r20 = r1 * r2;
    r23 = r28 + r23;
    r23 = r23 + r37;
    r23 = r23 + r45;
    r23 = fmaf(r11, r23, r10 * r32);
    r32 = r0 * r23;
    r9 = r33 + r9;
    r9 = fmaf(r11, r9, r10 * r21);
    r32 = fmaf(r9, r18, r2 * r32);
    r21 = r9 * r42;
    r31 = r26 + r31;
    r15 = r38 + r15;
    r15 = fmaf(r10, r15, r11 * r31);
    r10 = r1 * r15;
    r10 = fmaf(r2, r10, r34 * r21);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            4 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r32,
                                            r10,
                                            r19,
                                            r20);
    r20 = r42 * r34;
    write_idx_2<1024, float, float, float2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r18, r20);
    r20 = r6 * r4;
    r19 = r6 * r5;
    r19 = fmaf(r17, r19, r3 * r20);
    r20 = r6 * r4;
    r21 = r6 * r5;
    r21 = fmaf(r12, r21, r22 * r20);
    r20 = r6 * r5;
    r31 = r6 * r4;
    r31 = fmaf(r32, r31, r10 * r20);
    r20 = r6 * r4;
    r20 = r20 * r2;
    r11 = r0 * r20;
    write_sum_4<float, float>((float*)inout_shared, r19, r21, r31, r11);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = r1 * r6;
    r11 = r11 * r5;
    r11 = r11 * r2;
    r31 = r4 * r41;
    r21 = r5 * r41;
    r21 = fmaf(r34, r21, r29 * r31);
    write_sum_2<float, float>((float*)inout_shared, r11, r21);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = r0 * r0;
    r21 = r21 * r41;
    r11 = fmaf(r3, r3, r17 * r17);
    r31 = fmaf(r22, r22, r12 * r12);
    r19 = fmaf(r10, r10, r32 * r32);
    write_sum_4<float, float>((float*)inout_shared, r11, r31, r19, r21);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = r1 * r1;
    r21 = r21 * r41;
    r19 = r0 * r7;
    r40 = r30 * r40;
    r30 = r30 * r40;
    r30 = 1.0 / r30;
    r19 = r19 * r30;
    r31 = r8 * r30;
    r11 = r1 * r34;
    r31 = fmaf(r11, r31, r29 * r19);
    write_sum_2<float, float>((float*)inout_shared, r21, r31);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r17, r12, r3 * r22);
    r21 = fmaf(r17, r10, r3 * r32);
    r19 = r0 * r3;
    r19 = r19 * r2;
    r38 = r1 * r17;
    r38 = r38 * r2;
    write_sum_4<float, float>((float*)inout_shared, r31, r21, r19, r38);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = fmaf(r22, r32, r12 * r10);
    r19 = r0 * r22;
    r19 = r19 * r2;
    r21 = r1 * r12;
    r21 = r21 * r2;
    r31 = r17 * r42;
    r31 = fmaf(r34, r31, r3 * r18);
    write_sum_4<float, float>((float*)inout_shared, r31, r38, r19, r21);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = r0 * r32;
    r21 = r21 * r2;
    r19 = r1 * r10;
    r19 = r19 * r2;
    r38 = r12 * r42;
    r22 = fmaf(r22, r18, r34 * r38);
    r38 = r10 * r42;
    r18 = fmaf(r32, r18, r34 * r38);
    write_sum_4<float, float>((float*)inout_shared, r22, r21, r19, r18);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = 0.00000000000000000e+00;
    r19 = r0 * r6;
    r40 = 1.0 / r40;
    r19 = r19 * r40;
    r19 = r19 * r29;
    r40 = r6 * r40;
    r40 = r40 * r11;
    write_sum_3<float, float>((float*)inout_shared, r18, r19, r40);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r7 * r2;
    r19 = r8 * r2;
    write_idx_2<1024, float, float, float2>(out_calib_jac,
                                            0 * out_calib_jac_num_alloc,
                                            global_thread_idx,
                                            r40,
                                            r19);
    r11 = r6 * r4;
    r29 = r6 * r5;
    r20 = r7 * r20;
    r21 = r6 * r8;
    r21 = r21 * r5;
    r21 = r21 * r2;
    write_sum_4<float, float>((float*)inout_shared, r20, r21, r11, r29);
  };
  flush_sum_shared<4, float>(out_calib_njtr,
                             0 * out_calib_njtr_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = r7 * r7;
    r29 = r29 * r41;
    r11 = r8 * r8;
    r11 = r11 * r41;
    write_sum_4<float, float>((float*)inout_shared, r29, r11, r27, r27);
  };
  flush_sum_shared<4, float>(out_calib_precond_diag,
                             0 * out_calib_precond_diag_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_4<float, float>((float*)inout_shared, r18, r40, r18, r18);
  };
  flush_sum_shared<4, float>(out_calib_precond_tril,
                             0 * out_calib_precond_tril_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<float, float>((float*)inout_shared, r19, r18);
  };
  flush_sum_shared<2, float>(out_calib_precond_tril,
                             4 * out_calib_precond_tril_num_alloc,
                             calib_indices_loc,
                             (float*)inout_shared);
  sum_flush_final<float>(out_rTr_local, out_rTr, 1);
}

void pinhole_fixed_point_res_jac_first(
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
    float* const out_rTr,
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
  pinhole_fixed_point_res_jac_first_kernel<<<n_blocks, 1024>>>(
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
      out_rTr,
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