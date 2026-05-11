#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_res_jac.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointResJacKernel(
        float *point, unsigned int point_num_alloc, SharedIndex *point_indices,
        float *pixel, unsigned int pixel_num_alloc, float *pose,
        unsigned int pose_num_alloc, float *focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc, float *principal_point,
        unsigned int principal_point_num_alloc, float *out_res,
        unsigned int out_res_num_alloc, float *const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        float *const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        float *const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {
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
    ReadIdx2<1024, float, float, float2>(principal_point,
                                         0 * principal_point_num_alloc,
                                         global_thread_idx, r0, r1);
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
    ReadIdx3<1024, float, float, float4>(pose, 4 * pose_num_alloc,
                                         global_thread_idx, r0, r5, r6);
  };
  LoadShared<3, float, float>(point, 0 * point_num_alloc, point_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       point_indices_loc[threadIdx.x].target, r7, r8, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = -2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(pose, 0 * pose_num_alloc,
                                         global_thread_idx, r11, r12, r13, r14);
    r15 = r13 * r14;
    r16 = 2.00000000000000000e+00;
    r17 = r11 * r16;
    r18 = r12 * r17;
    r19 = fmaf(r10, r15, r18);
    r0 = fmaf(r8, r19, r0);
    r20 = r12 * r14;
    r21 = r13 * r17;
    r20 = fmaf(r16, r20, r21);
    r22 = r13 * r13;
    r22 = r22 * r10;
    r23 = 1.00000000000000000e+00;
    r24 = r12 * r12;
    r24 = fmaf(r10, r24, r23);
    r25 = r22 + r24;
    r0 = fmaf(r9, r20, r0);
    r0 = fmaf(r7, r25, r0);
    r26 = 9.99999999999999955e-07;
    r13 = r12 * r13;
    r13 = r13 * r16;
    r17 = fmaf(r14, r17, r13);
    r6 = fmaf(r8, r17, r6);
    r27 = r12 * r14;
    r27 = fmaf(r10, r27, r21);
    r21 = r11 * r11;
    r21 = r21 * r10;
    r24 = r21 + r24;
    r6 = fmaf(r7, r27, r6);
    r6 = fmaf(r9, r24, r6);
    r28 = copysign(1.0, r6);
    r28 = fmaf(r26, r28, r6);
    r26 = 1.0 / r28;
    ReadIdx2<1024, float, float, float2>(focal_and_distortion,
                                         0 * focal_and_distortion_num_alloc,
                                         global_thread_idx, r6, r29);
    r15 = fmaf(r16, r15, r18);
    r7 = fmaf(r7, r15, r5);
    r5 = r11 * r14;
    r5 = fmaf(r10, r5, r13);
    r22 = r23 + r22;
    r22 = r22 + r21;
    r7 = fmaf(r9, r5, r7);
    r7 = fmaf(r8, r22, r7);
    r8 = r28 * r28;
    r9 = 1.0 / r8;
    r21 = r7 * r9;
    r13 = r0 * r0;
    r18 = fmaf(r9, r13, r7 * r21);
    r18 = fmaf(r29, r18, r23);
    r18 = r6 * r18;
    r23 = r26 * r18;
    r2 = fmaf(r0, r23, r2);
    r3 = fmaf(r3, r4, r1);
    r3 = fmaf(r7, r23, r3);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r2, r3);
    r1 = r4 * r2;
    r30 = r0 * r27;
    r30 = r30 * r4;
    r30 = r30 * r9;
    r31 = r6 * r29;
    r8 = r28 * r8;
    r8 = 1.0 / r8;
    r8 = r10 * r8;
    r10 = r27 * r8;
    r28 = r16 * r25;
    r28 = r28 * r0;
    r28 = fmaf(r9, r28, r13 * r10);
    r10 = r7 * r7;
    r10 = r10 * r8;
    r32 = r16 * r15;
    r28 = fmaf(r21, r32, r28);
    r28 = fmaf(r27, r10, r28);
    r31 = r31 * r28;
    r31 = r31 * r26;
    r30 = fmaf(r0, r31, r18 * r30);
    r30 = fmaf(r25, r23, r30);
    r28 = r4 * r3;
    r32 = r4 * r18;
    r32 = r32 * r21;
    r31 = fmaf(r27, r32, r7 * r31);
    r31 = fmaf(r15, r23, r31);
    r28 = fmaf(r31, r28, r30 * r1);
    r1 = r4 * r2;
    r33 = r0 * r17;
    r33 = r33 * r4;
    r33 = r33 * r9;
    r33 = fmaf(r19, r23, r18 * r33);
    r34 = r6 * r29;
    r35 = r16 * r19;
    r35 = r35 * r0;
    r36 = r17 * r8;
    r36 = fmaf(r13, r36, r9 * r35);
    r35 = r16 * r22;
    r36 = fmaf(r21, r35, r36);
    r36 = fmaf(r17, r10, r36);
    r34 = r34 * r0;
    r34 = r34 * r36;
    r33 = fmaf(r26, r34, r33);
    r34 = r4 * r3;
    r35 = r6 * r29;
    r35 = r35 * r7;
    r35 = r35 * r36;
    r35 = fmaf(r26, r35, r17 * r32);
    r35 = fmaf(r22, r23, r35);
    r34 = fmaf(r35, r34, r33 * r1);
    r1 = r4 * r3;
    r32 = fmaf(r24, r32, r5 * r23);
    r36 = r6 * r29;
    r37 = r16 * r20;
    r37 = r37 * r0;
    r38 = r24 * r8;
    r38 = fmaf(r13, r38, r9 * r37);
    r37 = r16 * r5;
    r38 = fmaf(r21, r37, r38);
    r38 = fmaf(r24, r10, r38);
    r36 = r36 * r7;
    r36 = r36 * r38;
    r32 = fmaf(r26, r36, r32);
    r36 = r4 * r2;
    r7 = r6 * r29;
    r7 = r7 * r0;
    r7 = r7 * r38;
    r7 = fmaf(r26, r7, r20 * r23);
    r23 = r0 * r24;
    r23 = r23 * r4;
    r23 = r23 * r9;
    r7 = fmaf(r18, r23, r7);
    r36 = fmaf(r7, r36, r32 * r1);
    WriteSum3<float, float>((float *)inout_shared, r28, r34, r36);
  };
  FlushSumShared<3, float>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = fmaf(r30, r30, r31 * r31);
    r34 = fmaf(r35, r35, r33 * r33);
    r28 = fmaf(r7, r7, r32 * r32);
    WriteSum3<float, float>((float *)inout_shared, r36, r34, r28);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = fmaf(r31, r35, r30 * r33);
    r31 = fmaf(r31, r32, r30 * r7);
    r32 = fmaf(r35, r32, r33 * r7);
    WriteSum3<float, float>((float *)inout_shared, r28, r31, r32);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc, (float *)inout_shared);
}

void SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointResJac(
    float *point, unsigned int point_num_alloc, SharedIndex *point_indices,
    float *pixel, unsigned int pixel_num_alloc, float *pose,
    unsigned int pose_num_alloc, float *focal_and_distortion,
    unsigned int focal_and_distortion_num_alloc, float *principal_point,
    unsigned int principal_point_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *const out_point_njtr,
    unsigned int out_point_njtr_num_alloc, float *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointResJacKernel<<<
      n_blocks, 1024>>>(
      point, point_num_alloc, point_indices, pixel, pixel_num_alloc, pose,
      pose_num_alloc, focal_and_distortion, focal_and_distortion_num_alloc,
      principal_point, principal_point_num_alloc, out_res, out_res_num_alloc,
      out_point_njtr, out_point_njtr_num_alloc, out_point_precond_diag,
      out_point_precond_diag_num_alloc, out_point_precond_tril,
      out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar