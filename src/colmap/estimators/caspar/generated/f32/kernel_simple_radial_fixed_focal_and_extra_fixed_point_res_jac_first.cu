#include "kernel_simple_radial_fixed_focal_and_extra_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_focal_and_extra_fixed_point_res_jac_first_kernel(
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
      r46, r47, r48, r49;
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
    read_idx_2<1024, float, float, float2>(focal_and_extra,
                                           0 * focal_and_extra_num_alloc,
                                           global_thread_idx,
                                           r0,
                                           r5);
  };
  load_shared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>(
        (float*)inout_shared, pose_indices_loc[threadIdx.x].target, r6, r7, r8);
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
    r6 = fmaf(r10, r21, r6);
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
    r6 = fmaf(r11, r24, r6);
    r6 = fmaf(r9, r29, r6);
    r29 = r0 * r6;
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
    r35 = r6 * r6;
    r35 = fmaf(r8, r35, r8 * r25);
    r35 = fmaf(r5, r35, r26);
    r25 = 1.0 / r28;
    r39 = r35 * r25;
    r2 = fmaf(r29, r39, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r0 * r35;
    r1 = r1 * r25;
    r3 = fmaf(r7, r1, r3);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r39 = fmaf(r3, r3, r2 * r2);
  };
  sum_store<float>(out_rTr_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r39);
  if (global_thread_idx < problem_size) {
    r39 = r12 * r13;
    r39 = r39 * r18;
    r30 = r30 + r39;
    r24 = fmaf(r10, r24, r11 * r30);
    r13 = r13 * r19;
    r32 = r32 + r13;
    r30 = r14 * r14;
    r40 = r4 * r27;
    r41 = r30 + r40;
    r15 = r15 * r15;
    r42 = r4 * r34;
    r43 = r15 + r42;
    r44 = r41 + r43;
    r44 = fmaf(r10, r44, r11 * r32);
    r32 = r44 * r29;
    r45 = r35 * r4;
    r45 = r45 * r8;
    r32 = fmaf(r45, r32, r24 * r1);
    r46 = r17 * r7;
    r47 = r4 * r15;
    r48 = r34 + r47;
    r41 = r41 + r48;
    r41 = fmaf(r11, r41, r10 * r33);
    r46 = r46 * r41;
    r33 = r18 * r7;
    r36 = r28 * r36;
    r28 = 1.0 / r36;
    r33 = r33 * r7;
    r33 = r33 * r44;
    r33 = fmaf(r28, r33, r8 * r46);
    r46 = r18 * r6;
    r46 = r46 * r6;
    r46 = r46 * r44;
    r33 = fmaf(r28, r46, r33);
    r49 = r17 * r6;
    r49 = r49 * r24;
    r33 = fmaf(r8, r49, r33);
    r49 = r5 * r33;
    r49 = r49 * r25;
    r32 = fmaf(r29, r49, r32);
    r49 = r0 * r33;
    r46 = r5 * r7;
    r49 = r49 * r25;
    r41 = fmaf(r41, r1, r46 * r49);
    r49 = r0 * r7;
    r49 = r49 * r8;
    r49 = r49 * r35;
    r49 = r49 * r4;
    r41 = fmaf(r44, r49, r41);
    r35 = r17 * r7;
    r13 = r37 + r13;
    r13 = fmaf(r9, r13, r11 * r16);
    r35 = r35 * r13;
    r16 = r17 * r6;
    r19 = r12 * r19;
    r38 = r38 + r19;
    r15 = r34 + r15;
    r14 = r14 * r14;
    r14 = r14 * r4;
    r15 = r15 + r40;
    r15 = r15 + r14;
    r15 = fmaf(r11, r15, r9 * r38);
    r16 = r16 * r15;
    r16 = fmaf(r8, r16, r8 * r35);
    r35 = r18 * r6;
    r14 = r27 + r14;
    r48 = r48 + r14;
    r48 = fmaf(r9, r48, r11 * r22);
    r35 = r35 * r6;
    r35 = r35 * r48;
    r16 = fmaf(r28, r35, r16);
    r22 = r18 * r7;
    r22 = r22 * r7;
    r22 = r22 * r48;
    r16 = fmaf(r28, r22, r16);
    r22 = r5 * r16;
    r22 = r22 * r25;
    r35 = r48 * r29;
    r35 = fmaf(r45, r35, r29 * r22);
    r35 = fmaf(r15, r1, r35);
    r15 = r0 * r16;
    r15 = r15 * r25;
    r15 = fmaf(r46, r15, r13 * r1);
    r15 = fmaf(r48, r49, r15);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            0 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r32,
                                            r41,
                                            r35,
                                            r15);
    r13 = r0 * r5;
    r13 = r13 * r17;
    r13 = r13 * r6;
    r13 = r13 * r7;
    r13 = r13 * r28;
    r30 = r27 + r30;
    r30 = r30 + r42;
    r30 = r30 + r47;
    r30 = fmaf(r10, r30, r9 * r21);
    r19 = r23 + r19;
    r19 = fmaf(r10, r19, r9 * r31);
    r31 = r19 * r29;
    r31 = fmaf(r45, r31, r30 * r1);
    r23 = r17 * r6;
    r23 = r23 * r30;
    r30 = r18 * r7;
    r30 = r30 * r7;
    r30 = r30 * r19;
    r30 = fmaf(r28, r30, r8 * r23);
    r23 = r18 * r6;
    r23 = r23 * r6;
    r23 = r23 * r19;
    r30 = fmaf(r28, r23, r30);
    r21 = r17 * r7;
    r20 = r39 + r20;
    r14 = r43 + r14;
    r14 = fmaf(r9, r14, r10 * r20);
    r21 = r21 * r14;
    r30 = fmaf(r8, r21, r30);
    r21 = r5 * r30;
    r21 = r21 * r25;
    r31 = fmaf(r29, r21, r31);
    r21 = r0 * r30;
    r21 = r21 * r25;
    r21 = fmaf(r46, r21, r14 * r1);
    r21 = fmaf(r19, r49, r21);
    r14 = r28 * r29;
    r23 = r17 * r14;
    r8 = r5 * r6;
    r23 = fmaf(r8, r23, r1);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            4 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r31,
                                            r21,
                                            r23,
                                            r13);
    r9 = r0 * r17;
    r9 = r9 * r7;
    r9 = r9 * r28;
    r9 = fmaf(r46, r9, r1);
    r1 = r18 * r7;
    r1 = r1 * r7;
    r20 = r18 * r6;
    r20 = r20 * r6;
    r20 = fmaf(r28, r20, r28 * r1);
    r1 = r5 * r20;
    r1 = r1 * r25;
    r1 = fmaf(r29, r1, r29 * r45);
    r45 = r0 * r20;
    r45 = r45 * r25;
    r45 = fmaf(r46, r45, r49);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            8 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r13,
                                            r9,
                                            r1,
                                            r45);
    r49 = r4 * r2;
    r25 = r4 * r3;
    r25 = fmaf(r41, r25, r32 * r49);
    r49 = r4 * r2;
    r28 = r4 * r3;
    r28 = fmaf(r15, r28, r35 * r49);
    r49 = r4 * r3;
    r10 = r4 * r2;
    r10 = fmaf(r31, r10, r21 * r49);
    r49 = r4 * r2;
    r43 = r18 * r3;
    r39 = r46 * r14;
    r43 = fmaf(r39, r43, r23 * r49);
    write_sum_4<float, float>((float*)inout_shared, r25, r28, r10, r43);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r4 * r3;
    r10 = r4 * r2;
    r10 = fmaf(r1, r10, r45 * r43);
    r43 = r4 * r3;
    r28 = r18 * r2;
    r28 = fmaf(r39, r28, r9 * r43);
    write_sum_2<float, float>((float*)inout_shared, r28, r10);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fmaf(r41, r41, r32 * r32);
    r28 = fmaf(r15, r15, r35 * r35);
    r43 = fmaf(r31, r31, r21 * r21);
    r39 = r0 * r7;
    r25 = 4.00000000000000000e+00;
    r36 = r36 * r36;
    r36 = 1.0 / r36;
    r39 = r39 * r25;
    r39 = r39 * r36;
    r39 = r39 * r29;
    r39 = r39 * r46;
    r39 = r39 * r8;
    r8 = fmaf(r23, r23, r39);
    write_sum_4<float, float>((float*)inout_shared, r10, r28, r43, r8);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fmaf(r9, r9, r39);
    r8 = fmaf(r1, r1, r45 * r45);
    write_sum_2<float, float>((float*)inout_shared, r39, r8);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = fmaf(r32, r35, r41 * r15);
    r39 = fmaf(r32, r31, r41 * r21);
    r43 = fmaf(r41, r13, r32 * r23);
    r28 = fmaf(r32, r13, r41 * r9);
    write_sum_4<float, float>((float*)inout_shared, r8, r39, r43, r28);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fmaf(r32, r1, r41 * r45);
    r41 = fmaf(r15, r21, r35 * r31);
    r28 = fmaf(r15, r13, r35 * r23);
    r43 = fmaf(r35, r13, r15 * r9);
    write_sum_4<float, float>((float*)inout_shared, r32, r41, r28, r43);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r15, r45, r35 * r1);
    r35 = fmaf(r21, r45, r31 * r1);
    r43 = fmaf(r21, r13, r31 * r23);
    r31 = fmaf(r31, r13, r21 * r9);
    write_sum_4<float, float>((float*)inout_shared, r15, r43, r31, r35);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fmaf(r23, r13, r9 * r13);
    r23 = fmaf(r45, r13, r23 * r1);
    r13 = fmaf(r1, r13, r9 * r45);
    write_sum_3<float, float>((float*)inout_shared, r35, r23, r13);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r4 * r2;
    r23 = r4 * r3;
    write_sum_2<float, float>((float*)inout_shared, r13, r23);
  };
  flush_sum_shared<2, float>(out_principal_point_njtr,
                             0 * out_principal_point_njtr_num_alloc,
                             principal_point_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<float, float>((float*)inout_shared, r26, r26);
  };
  flush_sum_shared<2, float>(out_principal_point_precond_diag,
                             0 * out_principal_point_precond_diag_num_alloc,
                             principal_point_indices_loc,
                             (float*)inout_shared);
  sum_flush_final<float>(out_rTr_local, out_rTr, 1);
}

void simple_radial_fixed_focal_and_extra_fixed_point_res_jac_first(
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
  simple_radial_fixed_focal_and_extra_fixed_point_res_jac_first_kernel<<<
      n_blocks,
      1024>>>(pose,
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