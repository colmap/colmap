#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_opencv_fixed_pose_fixed_point_score.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) OpencvFixedPoseFixedPointScoreKernel(
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    float *pixel, unsigned int pixel_num_alloc, float *pose,
    unsigned int pose_num_alloc, float *point, unsigned int point_num_alloc,
    float *const out_rTr, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48;
  LoadShared<4, float, float>(calib, 4 * calib_num_alloc, calib_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       calib_indices_loc[threadIdx.x].target, r0, r1, r2, r3);
  };
  __syncthreads();
  LoadShared<4, float, float>(calib, 0 * calib_num_alloc, calib_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       calib_indices_loc[threadIdx.x].target, r4, r5, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = 9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r9, r10, r11);
    ReadIdx3<1024, float, float, float4>(point, 0 * point_num_alloc,
                                         global_thread_idx, r12, r13, r14);
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r15, r16, r17, r18);
    ReadIdx4<1024, float, float, float4>(pose, 0 * pose_num_alloc,
                                         global_thread_idx, r19, r20, r21, r22);
    r23 = fmaf(r15, r20, r18 * r21);
    r24 = r16 * r19;
    r25 = -1.00000000000000000e+00;
    r23 = fmaf(r25, r24, r23);
    r23 = fmaf(r17, r22, r23);
    r24 = 2.00000000000000000e+00;
    r26 = fmaf(r15, r22, r18 * r19);
    r27 = r17 * r20;
    r26 = fmaf(r25, r27, r26);
    r26 = fmaf(r16, r21, r26);
    r27 = r24 * r26;
    r28 = r23 * r27;
    r29 = r15 * r21;
    r29 = fmaf(r25, r29, r18 * r20);
    r29 = fmaf(r16, r22, r29);
    r29 = fmaf(r17, r19, r29);
    r30 = -2.00000000000000000e+00;
    r31 = fmaf(r16, r20, r15 * r19);
    r31 = fmaf(r17, r21, r31);
    r31 = fmaf(r25, r31, r18 * r22);
    r22 = r30 * r31;
    r32 = fmaf(r29, r22, r28);
    r32 = fmaf(r12, r32, r11);
    ReadIdx3<1024, float, float, float4>(pose, 4 * pose_num_alloc,
                                         global_thread_idx, r11, r33, r34);
    r35 = r15 * r17;
    r35 = r35 * r24;
    r36 = r16 * r18;
    r37 = fmaf(r30, r36, r35);
    r38 = r15 * r15;
    r38 = r30 * r38;
    r39 = 1.00000000000000000e+00;
    r40 = r16 * r16;
    r40 = fmaf(r30, r40, r39);
    r41 = r38 + r40;
    r42 = r16 * r17;
    r42 = r42 * r24;
    r43 = r15 * r18;
    r43 = fmaf(r24, r43, r42);
    r44 = r24 * r23;
    r44 = r44 * r29;
    r45 = fmaf(r31, r27, r44);
    r46 = r29 * r29;
    r46 = r30 * r46;
    r47 = r39 + r46;
    r48 = r26 * r26;
    r48 = r48 * r30;
    r47 = r47 + r48;
    r32 = fmaf(r11, r37, r32);
    r32 = fmaf(r34, r41, r32);
    r32 = fmaf(r33, r43, r32);
    r32 = fmaf(r13, r45, r32);
    r32 = fmaf(r14, r47, r32);
    r47 = copysign(1.0, r32);
    r47 = fmaf(r8, r47, r32);
    r8 = r47 * r47;
    r8 = 1.0 / r8;
    r32 = r24 * r23;
    r27 = r29 * r27;
    r32 = fmaf(r31, r32, r27);
    r32 = fmaf(r12, r32, r10);
    r10 = r15 * r16;
    r10 = r10 * r24;
    r45 = r17 * r18;
    r45 = fmaf(r24, r45, r10);
    r43 = r17 * r17;
    r43 = r30 * r43;
    r41 = r39 + r43;
    r41 = r41 + r38;
    r38 = r15 * r18;
    r38 = fmaf(r30, r38, r42);
    r26 = fmaf(r26, r22, r44);
    r44 = r23 * r23;
    r44 = r30 * r44;
    r42 = r39 + r44;
    r42 = r42 + r48;
    r32 = fmaf(r11, r45, r32);
    r32 = fmaf(r33, r41, r32);
    r32 = fmaf(r34, r38, r32);
    r32 = fmaf(r14, r26, r32);
    r32 = fmaf(r13, r42, r32);
    r42 = r32 * r32;
    r42 = r8 * r42;
    r26 = 3.00000000000000000e+00;
    r46 = r39 + r46;
    r46 = r46 + r44;
    r46 = fmaf(r12, r46, r9);
    r22 = fmaf(r23, r22, r27);
    r27 = r24 * r29;
    r27 = fmaf(r31, r27, r28);
    r36 = fmaf(r24, r36, r35);
    r35 = r17 * r18;
    r35 = fmaf(r30, r35, r10);
    r40 = r43 + r40;
    r46 = fmaf(r13, r22, r46);
    r46 = fmaf(r14, r27, r46);
    r46 = fmaf(r34, r36, r46);
    r46 = fmaf(r33, r35, r46);
    r46 = fmaf(r11, r40, r46);
    r40 = r46 * r46;
    r40 = r8 * r40;
    r11 = fmaf(r26, r40, r42);
    r47 = 1.0 / r47;
    r11 = fmaf(r46, r47, r1 * r11);
    r35 = r42 + r40;
    r33 = r35 * r35;
    r33 = fmaf(r7, r33, r6 * r35);
    r33 = r33 * r47;
    r7 = r0 * r24;
    r8 = r46 * r8;
    r7 = r7 * r32;
    r11 = fmaf(r8, r7, r11);
    r11 = fmaf(r46, r33, r11);
    r11 = fmaf(r4, r11, r2);
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r4, r2);
    r11 = fmaf(r4, r25, r11);
    r42 = fmaf(r26, r42, r40);
    r47 = fmaf(r32, r47, r0 * r42);
    r42 = r1 * r24;
    r42 = r42 * r32;
    r47 = fmaf(r8, r42, r47);
    r47 = fmaf(r32, r33, r47);
    r47 = fmaf(r5, r47, r3);
    r47 = fmaf(r2, r25, r47);
    r47 = fmaf(r47, r47, r11 * r11);
  };
  SumStore<float>(out_rTr_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r47);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void OpencvFixedPoseFixedPointScore(float *sensor_from_rig,
                                    unsigned int sensor_from_rig_num_alloc,
                                    float *calib, unsigned int calib_num_alloc,
                                    SharedIndex *calib_indices, float *pixel,
                                    unsigned int pixel_num_alloc, float *pose,
                                    unsigned int pose_num_alloc, float *point,
                                    unsigned int point_num_alloc,
                                    float *const out_rTr, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvFixedPoseFixedPointScoreKernel<<<n_blocks, 1024>>>(
      sensor_from_rig, sensor_from_rig_num_alloc, calib, calib_num_alloc,
      calib_indices, pixel, pixel_num_alloc, pose, pose_num_alloc, point,
      point_num_alloc, out_rTr, problem_size);
}

} // namespace caspar