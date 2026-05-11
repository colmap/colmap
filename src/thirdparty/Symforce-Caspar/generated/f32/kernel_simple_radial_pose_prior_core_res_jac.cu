#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_pose_prior_core_res_jac.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialPosePriorCoreResJacKernel(
        float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
        float *prior_position, unsigned int prior_position_num_alloc,
        float *sqrt_info, unsigned int sqrt_info_num_alloc, float *out_res,
        unsigned int out_res_num_alloc, float *const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float *const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float *const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sqrt_info, 0 * sqrt_info_num_alloc,
                                         global_thread_idx, r0, r1, r2, r3);
    ReadIdx3<1024, float, float, float4>(prior_position,
                                         0 * prior_position_num_alloc,
                                         global_thread_idx, r4, r5, r6);
    r7 = -1.00000000000000000e+00;
  };
  LoadShared<3, float, float>(pose, 4 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r8, r9, r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r11 = -2.00000000000000000e+00;
  };
  LoadShared<4, float, float>(pose, 0 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r12, r13, r14,
                       r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = r14 * r14;
    r17 = r11 * r16;
    r18 = 1.00000000000000000e+00;
    r19 = r12 * r12;
    r20 = fmaf(r11, r19, r18);
    r21 = r17 + r20;
    r22 = r12 * r15;
    r23 = 2.00000000000000000e+00;
    r22 = r22 * r23;
    r24 = r13 * r23;
    r25 = r14 * r24;
    r26 = r22 + r25;
    r26 = r10 * r26;
    r21 = fmaf(r9, r21, r26);
    r27 = r15 * r11;
    r28 = r14 * r27;
    r29 = r12 * r24;
    r30 = r28 + r29;
    r30 = r8 * r30;
    r21 = r21 + r30;
    r21 = fmaf(r7, r21, r5 * r7);
    r5 = r13 * r13;
    r5 = r5 * r11;
    r20 = r5 + r20;
    r31 = r12 * r27;
    r25 = r25 + r31;
    r20 = fmaf(r9, r25, r10 * r20);
    r32 = r12 * r14;
    r32 = r32 * r23;
    r33 = r15 * r24;
    r34 = r32 + r33;
    r20 = fmaf(r8, r34, r20);
    r20 = fmaf(r7, r20, r6 * r7);
    r6 = fmaf(r2, r20, r1 * r21);
    r17 = r18 + r17;
    r17 = r17 + r5;
    r5 = r14 * r15;
    r5 = r5 * r23;
    r29 = r5 + r29;
    r17 = fmaf(r9, r29, r8 * r17);
    r27 = r13 * r27;
    r32 = r32 + r27;
    r17 = fmaf(r10, r32, r17);
    r17 = fmaf(r7, r17, r4 * r7);
    r6 = fmaf(r0, r17, r6);
    ReadIdx2<1024, float, float, float2>(sqrt_info, 4 * sqrt_info_num_alloc,
                                         global_thread_idx, r17, r4);
    r21 = fmaf(r17, r20, r3 * r21);
    r18 = r4 * r20;
    WriteIdx3<1024, float, float, float4>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r6, r21, r18);
    r18 = r7 * r21;
    r35 = r10 * r7;
    r36 = r13 * r14;
    r36 = r36 * r11;
    r31 = r36 + r31;
    r37 = r8 * r7;
    r38 = r12 * r13;
    r38 = r38 * r11;
    r5 = r5 + r38;
    r37 = fmaf(r5, r37, r31 * r35);
    r9 = r9 * r7;
    r35 = r15 * r15;
    r35 = r35 * r7;
    r39 = r19 + r35;
    r40 = r13 * r13;
    r40 = r40 * r7;
    r41 = r16 + r40;
    r42 = r39 + r41;
    r37 = fmaf(r42, r9, r37);
    r42 = r10 * r7;
    r15 = r15 * r15;
    r43 = r7 * r19;
    r44 = r15 + r43;
    r41 = r41 + r44;
    r45 = r8 * r7;
    r45 = fmaf(r34, r45, r41 * r42);
    r45 = fmaf(r25, r9, r45);
    r25 = fmaf(r3, r45, r17 * r37);
    r42 = r7 * r6;
    r45 = fmaf(r1, r45, r2 * r37);
    r42 = fmaf(r45, r42, r25 * r18);
    r18 = r7 * r20;
    r4 = r4 * r4;
    r18 = r18 * r37;
    r42 = fmaf(r4, r18, r42);
    r18 = r7 * r6;
    r34 = r8 * r7;
    r15 = r19 + r15;
    r41 = r7 * r16;
    r15 = r15 + r40;
    r15 = r15 + r41;
    r40 = r10 * r7;
    r40 = fmaf(r32, r40, r15 * r34);
    r40 = fmaf(r29, r9, r40);
    r29 = r10 * r7;
    r34 = r13 * r13;
    r41 = r34 + r41;
    r39 = r39 + r41;
    r32 = r8 * r7;
    r14 = r12 * r14;
    r14 = r14 * r11;
    r27 = r14 + r27;
    r32 = fmaf(r27, r32, r39 * r29);
    r36 = r22 + r36;
    r32 = fmaf(r36, r9, r32);
    r32 = fmaf(r0, r32, r2 * r40);
    r22 = r7 * r21;
    r29 = r17 * r40;
    r22 = fmaf(r29, r22, r32 * r18);
    r18 = r7 * r20;
    r18 = r18 * r40;
    r22 = fmaf(r4, r18, r22);
    r18 = r7 * r6;
    r41 = r44 + r41;
    r26 = fmaf(r7, r26, r41 * r9);
    r26 = fmaf(r7, r30, r26);
    r30 = r8 * r7;
    r34 = r16 + r34;
    r34 = r34 + r35;
    r34 = r34 + r43;
    r43 = r10 * r7;
    r33 = r14 + r33;
    r43 = fmaf(r33, r43, r34 * r30);
    r28 = r38 + r28;
    r43 = fmaf(r28, r9, r43);
    r26 = fmaf(r1, r43, r0 * r26);
    r9 = r7 * r21;
    r38 = r3 * r43;
    r9 = fmaf(r38, r9, r26 * r18);
    r18 = r7 * r21;
    r30 = fmaf(r17, r27, r3 * r5);
    r34 = r7 * r6;
    r16 = r23 * r16;
    r24 = fmaf(r13, r24, r7);
    r13 = r16 + r24;
    r5 = fmaf(r1, r5, r0 * r13);
    r5 = fmaf(r2, r27, r5);
    r34 = fmaf(r5, r34, r30 * r18);
    r18 = r7 * r20;
    r18 = r18 * r27;
    r34 = fmaf(r4, r18, r34);
    WriteSum4<float, float>((float *)inout_shared, r42, r22, r9, r34);
  };
  FlushSumShared<4, float>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r7 * r21;
    r16 = r7 + r16;
    r19 = r23 * r19;
    r16 = r16 + r19;
    r23 = fmaf(r17, r36, r3 * r16);
    r9 = r7 * r6;
    r16 = fmaf(r2, r36, r1 * r16);
    r16 = fmaf(r0, r28, r16);
    r9 = fmaf(r16, r9, r23 * r34);
    r34 = r7 * r20;
    r34 = r34 * r36;
    r9 = fmaf(r4, r34, r9);
    r34 = r7 * r21;
    r24 = r19 + r24;
    r19 = fmaf(r3, r31, r17 * r24);
    r28 = r7 * r6;
    r31 = fmaf(r1, r31, r2 * r24);
    r31 = fmaf(r0, r33, r31);
    r28 = fmaf(r31, r28, r19 * r34);
    r34 = r7 * r20;
    r33 = r24 * r4;
    r28 = fmaf(r33, r34, r28);
    WriteSum2<float, float>((float *)inout_shared, r9, r28);
  };
  FlushSumShared<2, float>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = fmaf(r45, r45, r25 * r25);
    r9 = r37 * r37;
    r28 = fmaf(r4, r9, r28);
    r9 = r40 * r40;
    r9 = fmaf(r4, r9, r32 * r32);
    r34 = r17 * r40;
    r9 = fmaf(r29, r34, r9);
    r34 = r3 * r43;
    r34 = fmaf(r38, r34, r26 * r26);
    r0 = fmaf(r30, r30, r5 * r5);
    r1 = r27 * r27;
    r0 = fmaf(r4, r1, r0);
    WriteSum4<float, float>((float *)inout_shared, r28, r9, r34, r0);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = fmaf(r16, r16, r23 * r23);
    r34 = r36 * r36;
    r0 = fmaf(r4, r34, r0);
    r34 = fmaf(r19, r19, r31 * r31);
    r34 = fmaf(r24, r33, r34);
    WriteSum2<float, float>((float *)inout_shared, r0, r34);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fmaf(r25, r29, r45 * r32);
    r0 = r37 * r40;
    r34 = fmaf(r4, r0, r34);
    r0 = fmaf(r25, r38, r45 * r26);
    r24 = fmaf(r45, r5, r25 * r30);
    r9 = r37 * r27;
    r24 = fmaf(r4, r9, r24);
    r9 = fmaf(r45, r16, r25 * r23);
    r28 = r37 * r36;
    r9 = fmaf(r4, r28, r9);
    WriteSum4<float, float>((float *)inout_shared, r34, r0, r24, r9);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fmaf(r45, r31, r25 * r19);
    r45 = fmaf(r37, r33, r45);
    r25 = fmaf(r29, r38, r32 * r26);
    r9 = r40 * r27;
    r9 = fmaf(r4, r9, r32 * r5);
    r9 = fmaf(r30, r29, r9);
    r24 = fmaf(r23, r29, r32 * r16);
    r0 = r40 * r36;
    r24 = fmaf(r4, r0, r24);
    WriteSum4<float, float>((float *)inout_shared, r45, r25, r9, r24);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fmaf(r19, r29, r32 * r31);
    r29 = fmaf(r40, r33, r29);
    r32 = fmaf(r30, r38, r26 * r5);
    r24 = fmaf(r23, r38, r26 * r16);
    r38 = fmaf(r19, r38, r26 * r31);
    WriteSum4<float, float>((float *)inout_shared, r29, r32, r24, r38);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = fmaf(r30, r23, r5 * r16);
    r24 = r36 * r27;
    r38 = fmaf(r4, r24, r38);
    r5 = fmaf(r5, r31, r30 * r19);
    r5 = fmaf(r27, r33, r5);
    r31 = fmaf(r16, r31, r23 * r19);
    r31 = fmaf(r36, r33, r31);
    WriteSum3<float, float>((float *)inout_shared, r38, r5, r31);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
}

void SimpleRadialPosePriorCoreResJac(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *prior_position, unsigned int prior_position_num_alloc,
    float *sqrt_info, unsigned int sqrt_info_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, float *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialPosePriorCoreResJacKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, prior_position,
      prior_position_num_alloc, sqrt_info, sqrt_info_num_alloc, out_res,
      out_res_num_alloc, out_pose_njtr, out_pose_njtr_num_alloc,
      out_pose_precond_diag, out_pose_precond_diag_num_alloc,
      out_pose_precond_tril, out_pose_precond_tril_num_alloc, problem_size);
}

} // namespace caspar