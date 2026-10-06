#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_opencv_split_fixed_pose_fixed_principal_point_score.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedPoseFixedPrincipalPointScoreKernel(
        float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        float *focal_and_extra, unsigned int focal_and_extra_num_alloc,
        SharedIndex *focal_and_extra_indices, float *point,
        unsigned int point_num_alloc, SharedIndex *point_indices, float *pixel,
        unsigned int pixel_num_alloc, float *pose, unsigned int pose_num_alloc,
        float *principal_point, unsigned int principal_point_num_alloc,
        float *const out_rTr, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex focal_and_extra_indices_loc[1024];
  focal_and_extra_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_indices[global_thread_idx]
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
      r46, r47;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(principal_point,
                                         0 * principal_point_num_alloc,
                                         global_thread_idx, r0, r1);
  };
  LoadShared<4, float, float>(focal_and_extra, 0 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target, r2, r3,
                       r4, r5);
  };
  __syncthreads();
  LoadShared<2, float, float>(focal_and_extra, 4 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float *)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = 9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r9, r10, r11);
  };
  LoadShared<3, float, float>(point, 0 * point_num_alloc, point_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       point_indices_loc[threadIdx.x].target, r12, r13, r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = -2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r16, r17, r18, r19);
    ReadIdx4<1024, float, float, float4>(pose, 0 * pose_num_alloc,
                                         global_thread_idx, r20, r21, r22, r23);
    r24 = r16 * r22;
    r25 = -1.00000000000000000e+00;
    r24 = fmaf(r25, r24, r19 * r21);
    r24 = fmaf(r17, r23, r24);
    r24 = fmaf(r18, r20, r24);
    r26 = r15 * r24;
    r27 = fmaf(r17, r21, r16 * r20);
    r27 = fmaf(r18, r22, r27);
    r27 = fmaf(r25, r27, r19 * r23);
    r28 = fmaf(r16, r21, r19 * r22);
    r29 = r17 * r20;
    r28 = fmaf(r25, r29, r28);
    r28 = fmaf(r18, r23, r28);
    r29 = 2.00000000000000000e+00;
    r23 = fmaf(r16, r23, r19 * r20);
    r30 = r18 * r21;
    r23 = fmaf(r25, r30, r23);
    r23 = fmaf(r17, r22, r23);
    r30 = r29 * r23;
    r31 = r28 * r30;
    r26 = fmaf(r27, r26, r31);
    r26 = fmaf(r12, r26, r11);
    ReadIdx3<1024, float, float, float4>(pose, 4 * pose_num_alloc,
                                         global_thread_idx, r11, r32, r33);
    r34 = r16 * r18;
    r34 = r34 * r29;
    r35 = r19 * r15;
    r36 = fmaf(r17, r35, r34);
    r37 = r17 * r17;
    r37 = r15 * r37;
    r38 = 1.00000000000000000e+00;
    r39 = r16 * r16;
    r39 = fmaf(r15, r39, r38);
    r40 = r37 + r39;
    r41 = r17 * r18;
    r41 = r41 * r29;
    r42 = r16 * r19;
    r42 = fmaf(r29, r42, r41);
    r43 = r29 * r28;
    r43 = r43 * r24;
    r44 = fmaf(r27, r30, r43);
    r45 = r23 * r23;
    r45 = r45 * r15;
    r46 = r38 + r45;
    r47 = r24 * r24;
    r47 = r15 * r47;
    r46 = r46 + r47;
    r26 = fmaf(r11, r36, r26);
    r26 = fmaf(r33, r40, r26);
    r26 = fmaf(r32, r42, r26);
    r26 = fmaf(r13, r44, r26);
    r26 = fmaf(r14, r46, r26);
    r46 = copysign(1.0, r26);
    r46 = fmaf(r8, r46, r26);
    r8 = r46 * r46;
    r8 = 1.0 / r8;
    r26 = r28 * r28;
    r26 = r26 * r15;
    r44 = r38 + r26;
    r44 = r44 + r47;
    r44 = fmaf(r12, r44, r9);
    r30 = r24 * r30;
    r28 = r28 * r27;
    r9 = fmaf(r15, r28, r30);
    r47 = r29 * r24;
    r47 = fmaf(r27, r47, r31);
    r31 = r17 * r19;
    r31 = fmaf(r29, r31, r34);
    r34 = r16 * r17;
    r34 = r34 * r29;
    r42 = fmaf(r18, r35, r34);
    r40 = r18 * r18;
    r40 = r15 * r40;
    r36 = r38 + r40;
    r36 = r36 + r37;
    r44 = fmaf(r13, r9, r44);
    r44 = fmaf(r14, r47, r44);
    r44 = fmaf(r33, r31, r44);
    r44 = fmaf(r32, r42, r44);
    r44 = fmaf(r11, r36, r44);
    r36 = r44 * r44;
    r36 = r8 * r36;
    r42 = 3.00000000000000000e+00;
    r28 = fmaf(r29, r28, r30);
    r28 = fmaf(r12, r28, r10);
    r12 = r18 * r19;
    r12 = fmaf(r29, r12, r34);
    r39 = r40 + r39;
    r35 = fmaf(r16, r35, r41);
    r41 = r23 * r15;
    r41 = fmaf(r27, r41, r43);
    r45 = r38 + r45;
    r45 = r45 + r26;
    r28 = fmaf(r11, r12, r28);
    r28 = fmaf(r32, r39, r28);
    r28 = fmaf(r33, r35, r28);
    r28 = fmaf(r14, r41, r28);
    r28 = fmaf(r13, r45, r28);
    r45 = r28 * r28;
    r45 = r8 * r45;
    r13 = fmaf(r42, r45, r36);
    r46 = 1.0 / r46;
    r13 = fmaf(r28, r46, r6 * r13);
    r41 = r7 * r29;
    r8 = r28 * r8;
    r41 = r41 * r44;
    r13 = fmaf(r8, r41, r13);
    r14 = r36 + r45;
    r35 = r14 * r14;
    r35 = fmaf(r5, r35, r4 * r14);
    r35 = r35 * r46;
    r13 = fmaf(r28, r35, r13);
    r13 = fmaf(r3, r13, r1);
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r3, r1);
    r13 = fmaf(r1, r25, r13);
    r36 = fmaf(r42, r36, r45);
    r46 = fmaf(r44, r46, r7 * r36);
    r36 = r6 * r29;
    r36 = r36 * r44;
    r46 = fmaf(r8, r36, r46);
    r46 = fmaf(r44, r35, r46);
    r46 = fmaf(r2, r46, r0);
    r46 = fmaf(r3, r25, r46);
    r46 = fmaf(r46, r46, r13 * r13);
  };
  // See kernel_opencv_split_fixed_principal_point_score.cu: r46 is only
  // assigned inside the guard above, so for padding-lane threads beyond
  // problem_size it is otherwise read uninitialized here (UB, can be NaN).
  if (global_thread_idx >= problem_size) {
    r46 = 0.0f;
  }
  SumStore<float>(out_rTr_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r46);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void OpencvSplitFixedPoseFixedPrincipalPointScore(
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *focal_and_extra, unsigned int focal_and_extra_num_alloc,
    SharedIndex *focal_and_extra_indices, float *point,
    unsigned int point_num_alloc, SharedIndex *point_indices, float *pixel,
    unsigned int pixel_num_alloc, float *pose, unsigned int pose_num_alloc,
    float *principal_point, unsigned int principal_point_num_alloc,
    float *const out_rTr, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvSplitFixedPoseFixedPrincipalPointScoreKernel<<<n_blocks, 1024>>>(
      sensor_from_rig, sensor_from_rig_num_alloc, focal_and_extra,
      focal_and_extra_num_alloc, focal_and_extra_indices, point,
      point_num_alloc, point_indices, pixel, pixel_num_alloc, pose,
      pose_num_alloc, principal_point, principal_point_num_alloc, out_rTr,
      problem_size);
}

} // namespace caspar