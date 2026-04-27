#include "kernel_pinhole_fixed_focal_and_extra_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_focal_and_extra_fixed_point_res_jac_first_kernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
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
        float* out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        float* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        float* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        float* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46;
  load_shared<2, float, float>(principal_point,
                               0 * principal_point_num_alloc,
                               principal_point_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<float>((float*)inout_shared,
                         principal_point_indices_loc[threadIdx.x].target,
                         r0,
                         r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
    r0 = 9.99999999999999955e-07;
  };
  load_shared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>(
        (float*)inout_shared, pose_indices_loc[threadIdx.x].target, r5, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9, r10);
  };
  load_shared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         pose_indices_loc[threadIdx.x].target,
                         r11,
                         r12,
                         r13,
                         r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = r12 * r13;
    r16 = 2.00000000000000000e+00;
    r15 = r15 * r16;
    r17 = r11 * r16;
    r18 = r14 * r17;
    r19 = r15 + r18;
    r7 = fmaf(r9, r19, r7);
    r20 = r12 * r14;
    r21 = -2.00000000000000000e+00;
    r20 = r20 * r21;
    r22 = r13 * r17;
    r23 = r20 + r22;
    r24 = r11 * r11;
    r24 = r24 * r21;
    r25 = 1.00000000000000000e+00;
    r26 = r12 * r12;
    r27 = fmaf(r21, r26, r25);
    r28 = r24 + r27;
    r7 = fmaf(r8, r23, r7);
    r7 = fmaf(r10, r28, r7);
    r28 = copysign(1.0, r7);
    r28 = fmaf(r0, r28, r7);
    r0 = 1.0 / r28;
    read_idx_2<1024, float, float, float2>(focal_and_extra,
                                           0 * focal_and_extra_num_alloc,
                                           global_thread_idx,
                                           r7,
                                           r29);
    r30 = r13 * r21;
    r31 = r14 * r30;
    r17 = r12 * r17;
    r32 = r31 + r17;
    r5 = fmaf(r9, r32, r5);
    r33 = r12 * r14;
    r33 = r33 * r16;
    r22 = r33 + r22;
    r34 = r13 * r30;
    r27 = r34 + r27;
    r5 = fmaf(r10, r22, r5);
    r5 = fmaf(r8, r27, r5);
    r27 = r7 * r5;
    r2 = fmaf(r0, r27, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r13 * r14;
    r1 = r1 * r16;
    r17 = r1 + r17;
    r6 = fmaf(r8, r17, r6);
    r16 = r11 * r14;
    r16 = r16 * r21;
    r15 = r15 + r16;
    r34 = r25 + r34;
    r34 = r34 + r24;
    r6 = fmaf(r10, r15, r6);
    r6 = fmaf(r9, r34, r6);
    r34 = r29 * r6;
    r3 = fmaf(r0, r34, r3);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r24 = fmaf(r3, r3, r2 * r2);
  };
  sum_store<float>(out_rTr_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r24);
  if (global_thread_idx < problem_size) {
    r24 = r12 * r30;
    r16 = r16 + r24;
    r35 = r13 * r13;
    r36 = r4 * r26;
    r37 = r35 + r36;
    r14 = r14 * r14;
    r38 = r11 * r11;
    r38 = r38 * r4;
    r39 = r14 + r38;
    r40 = r37 + r39;
    r40 = fmaf(r9, r40, r10 * r16);
    r16 = r7 * r4;
    r41 = r28 * r28;
    r42 = 1.0 / r41;
    r16 = r16 * r5;
    r16 = r16 * r42;
    r12 = r11 * r12;
    r12 = r12 * r21;
    r1 = r1 + r12;
    r22 = fmaf(r9, r22, r10 * r1);
    r1 = r7 * r22;
    r1 = fmaf(r0, r1, r40 * r16);
    r21 = r11 * r11;
    r43 = r4 * r14;
    r44 = r21 + r43;
    r37 = r37 + r44;
    r37 = fmaf(r10, r37, r9 * r15);
    r15 = r29 * r37;
    r45 = r4 * r42;
    r46 = r40 * r45;
    r46 = fmaf(r34, r46, r0 * r15);
    r13 = r13 * r13;
    r13 = r13 * r4;
    r15 = r26 + r13;
    r44 = r44 + r15;
    r44 = fmaf(r8, r44, r10 * r23);
    r30 = r11 * r30;
    r20 = r20 + r30;
    r14 = r21 + r14;
    r14 = r14 + r36;
    r14 = r14 + r13;
    r14 = fmaf(r10, r14, r8 * r20);
    r20 = r7 * r14;
    r20 = fmaf(r0, r20, r44 * r16);
    r13 = r44 * r45;
    r24 = r18 + r24;
    r24 = fmaf(r8, r24, r10 * r17);
    r17 = r29 * r24;
    r17 = fmaf(r0, r17, r34 * r13);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            0 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r1,
                                            r46,
                                            r20,
                                            r17);
    r13 = r7 * r0;
    r10 = r29 * r0;
    r30 = r33 + r30;
    r30 = fmaf(r9, r30, r8 * r19);
    r35 = r26 + r35;
    r35 = r35 + r38;
    r35 = r35 + r43;
    r35 = fmaf(r9, r35, r8 * r32);
    r32 = r7 * r35;
    r32 = fmaf(r0, r32, r30 * r16);
    r43 = r30 * r45;
    r31 = r12 + r31;
    r15 = r39 + r15;
    r15 = fmaf(r8, r15, r9 * r31);
    r8 = r29 * r15;
    r8 = fmaf(r0, r8, r34 * r43);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            4 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r32,
                                            r8,
                                            r13,
                                            r10);
    r10 = r45 * r34;
    write_idx_2<1024, float, float, float2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r16, r10);
    r10 = r4 * r3;
    r13 = r4 * r2;
    r13 = fmaf(r1, r13, r46 * r10);
    r10 = r4 * r2;
    r43 = r4 * r3;
    r43 = fmaf(r17, r43, r20 * r10);
    r10 = r4 * r2;
    r31 = r4 * r3;
    r31 = fmaf(r8, r31, r32 * r10);
    r10 = r7 * r4;
    r10 = r10 * r2;
    r10 = r10 * r0;
    write_sum_4<float, float>((float*)inout_shared, r13, r43, r31, r10);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = r29 * r4;
    r10 = r10 * r3;
    r10 = r10 * r0;
    r31 = r3 * r42;
    r43 = r2 * r42;
    r43 = fmaf(r27, r43, r34 * r31);
    write_sum_2<float, float>((float*)inout_shared, r10, r43);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r7 * r7;
    r43 = r43 * r42;
    r10 = fmaf(r1, r1, r46 * r46);
    r31 = fmaf(r17, r17, r20 * r20);
    r13 = fmaf(r32, r32, r8 * r8);
    write_sum_4<float, float>((float*)inout_shared, r10, r31, r13, r43);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r29 * r29;
    r43 = r43 * r42;
    r41 = r28 * r41;
    r28 = r28 * r41;
    r28 = 1.0 / r28;
    r13 = r6 * r28;
    r31 = r29 * r34;
    r10 = r7 * r5;
    r10 = r10 * r28;
    r10 = fmaf(r27, r10, r31 * r13);
    write_sum_2<float, float>((float*)inout_shared, r43, r10);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fmaf(r46, r17, r1 * r20);
    r43 = fmaf(r1, r32, r46 * r8);
    r13 = r7 * r1;
    r13 = r13 * r0;
    r9 = r29 * r46;
    r9 = r9 * r0;
    write_sum_4<float, float>((float*)inout_shared, r10, r43, r13, r9);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = fmaf(r17, r8, r20 * r32);
    r13 = r7 * r20;
    r13 = r13 * r0;
    r43 = r29 * r17;
    r43 = r43 * r0;
    r10 = r46 * r45;
    r1 = fmaf(r1, r16, r34 * r10);
    write_sum_4<float, float>((float*)inout_shared, r1, r9, r13, r43);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r7 * r32;
    r43 = r43 * r0;
    r13 = r29 * r8;
    r13 = r13 * r0;
    r0 = r17 * r45;
    r20 = fmaf(r20, r16, r34 * r0);
    r0 = r8 * r45;
    r0 = fmaf(r34, r0, r32 * r16);
    write_sum_4<float, float>((float*)inout_shared, r20, r43, r13, r0);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
    r27 = r7 * r27;
    r41 = 1.0 / r41;
    r41 = r4 * r41;
    r27 = r27 * r41;
    r31 = r41 * r31;
    write_sum_3<float, float>((float*)inout_shared, r0, r27, r31);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r4 * r2;
    r27 = r4 * r3;
    write_sum_2<float, float>((float*)inout_shared, r31, r27);
  };
  flush_sum_shared<2, float>(out_principal_point_njtr,
                             0 * out_principal_point_njtr_num_alloc,
                             principal_point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<float, float>((float*)inout_shared, r25, r25);
  };
  flush_sum_shared<2, float>(out_principal_point_precond_diag,
                             0 * out_principal_point_precond_diag_num_alloc,
                             principal_point_indices_loc,
                             (float*)inout_shared);
  sum_flush_final<float>(out_rTr_local, out_rTr, 1);
}

void pinhole_fixed_focal_and_extra_fixed_point_res_jac_first(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
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
    float* out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    float* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    float* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_fixed_focal_and_extra_fixed_point_res_jac_first_kernel<<<n_blocks,
                                                                   1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      pixel,
      pixel_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
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
      out_principal_point_jac,
      out_principal_point_jac_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar