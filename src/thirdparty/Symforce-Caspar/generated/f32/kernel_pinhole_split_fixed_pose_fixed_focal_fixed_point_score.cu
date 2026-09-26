#include "cuda_to_hip.h"

#include "kernel_pinhole_split_fixed_pose_fixed_focal_fixed_point_score.h"
#include "memops.cuh"

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPoseFixedFocalFixedPointScoreKernel(
        float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        float *principal_point, unsigned int principal_point_num_alloc,
        SharedIndex *principal_point_indices, float *pixel,
        unsigned int pixel_num_alloc, float *pose, unsigned int pose_num_alloc,
        float *focal, unsigned int focal_num_alloc, float *point,
        unsigned int point_num_alloc, float *const out_rTr,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45;
  LoadShared<2, float, float>(principal_point, 0 * principal_point_num_alloc,
                              principal_point_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float *)inout_shared,
                       principal_point_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r3 = fmaf(r3, r4, r1);
    ReadIdx2<1024, float, float, float2>(focal, 0 * focal_num_alloc,
                                         global_thread_idx, r1, r5);
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r6, r7, r8);
    ReadIdx3<1024, float, float, float4>(point, 0 * point_num_alloc,
                                         global_thread_idx, r9, r10, r11);
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r12, r13, r14, r15);
    ReadIdx4<1024, float, float, float4>(pose, 0 * pose_num_alloc,
                                         global_thread_idx, r16, r17, r18, r19);
    r20 = r12 * r18;
    r20 = fmaf(r4, r20, r15 * r17);
    r20 = fmaf(r13, r19, r20);
    r20 = fmaf(r14, r16, r20);
    r21 = 2.00000000000000000e+00;
    r22 = fmaf(r12, r19, r15 * r16);
    r23 = r14 * r17;
    r22 = fmaf(r4, r23, r22);
    r22 = fmaf(r13, r18, r22);
    r23 = r21 * r22;
    r24 = r20 * r23;
    r25 = fmaf(r12, r17, r15 * r18);
    r26 = r13 * r16;
    r25 = fmaf(r4, r26, r25);
    r25 = fmaf(r14, r19, r25);
    r26 = fmaf(r13, r17, r12 * r16);
    r26 = fmaf(r14, r18, r26);
    r26 = fmaf(r4, r26, r15 * r19);
    r19 = r25 * r26;
    r27 = fmaf(r21, r19, r24);
    r27 = fmaf(r9, r27, r7);
    ReadIdx3<1024, float, float, float4>(pose, 4 * pose_num_alloc,
                                         global_thread_idx, r7, r28, r29);
    r30 = r14 * r15;
    r31 = r12 * r13;
    r31 = r31 * r21;
    r30 = fmaf(r21, r30, r31);
    r32 = -2.00000000000000000e+00;
    r33 = r14 * r14;
    r33 = r32 * r33;
    r34 = 1.00000000000000000e+00;
    r35 = r12 * r12;
    r35 = fmaf(r32, r35, r34);
    r36 = r33 + r35;
    r37 = r13 * r14;
    r37 = r37 * r21;
    r38 = r15 * r32;
    r39 = fmaf(r12, r38, r37);
    r40 = r21 * r25;
    r40 = r40 * r20;
    r41 = r22 * r32;
    r41 = fmaf(r26, r41, r40);
    r42 = r22 * r22;
    r42 = r42 * r32;
    r43 = r34 + r42;
    r44 = r25 * r25;
    r44 = r44 * r32;
    r43 = r43 + r44;
    r27 = fmaf(r7, r30, r27);
    r27 = fmaf(r28, r36, r27);
    r27 = fmaf(r29, r39, r27);
    r27 = fmaf(r11, r41, r27);
    r27 = fmaf(r10, r43, r27);
    r43 = r5 * r27;
    r41 = 9.99999999999999955e-07;
    r39 = r32 * r20;
    r25 = r25 * r23;
    r39 = fmaf(r26, r39, r25);
    r39 = fmaf(r9, r39, r8);
    r8 = r12 * r14;
    r8 = r8 * r21;
    r36 = fmaf(r13, r38, r8);
    r30 = r13 * r13;
    r30 = r32 * r30;
    r35 = r30 + r35;
    r45 = r12 * r15;
    r45 = fmaf(r21, r45, r37);
    r23 = fmaf(r26, r23, r40);
    r42 = r34 + r42;
    r40 = r20 * r20;
    r40 = r32 * r40;
    r42 = r42 + r40;
    r39 = fmaf(r7, r36, r39);
    r39 = fmaf(r29, r35, r39);
    r39 = fmaf(r28, r45, r39);
    r39 = fmaf(r10, r23, r39);
    r39 = fmaf(r11, r42, r39);
    r42 = copysign(1.0, r39);
    r42 = fmaf(r41, r42, r39);
    r42 = 1.0 / r42;
    r3 = fmaf(r42, r43, r3);
    r4 = fmaf(r2, r4, r0);
    r44 = r34 + r44;
    r44 = r44 + r40;
    r44 = fmaf(r9, r44, r6);
    r19 = fmaf(r32, r19, r24);
    r24 = r21 * r20;
    r24 = fmaf(r26, r24, r25);
    r25 = r13 * r15;
    r25 = fmaf(r21, r25, r8);
    r38 = fmaf(r14, r38, r31);
    r33 = r34 + r33;
    r33 = r33 + r30;
    r44 = fmaf(r10, r19, r44);
    r44 = fmaf(r11, r24, r44);
    r44 = fmaf(r29, r25, r44);
    r44 = fmaf(r28, r38, r44);
    r44 = fmaf(r7, r33, r44);
    r33 = r1 * r44;
    r4 = fmaf(r42, r33, r4);
    r4 = fmaf(r4, r4, r3 * r3);
  };
  SumStore<float>(out_rTr_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r4);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedPoseFixedFocalFixedPointScore(
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *principal_point, unsigned int principal_point_num_alloc,
    SharedIndex *principal_point_indices, float *pixel,
    unsigned int pixel_num_alloc, float *pose, unsigned int pose_num_alloc,
    float *focal, unsigned int focal_num_alloc, float *point,
    unsigned int point_num_alloc, float *const out_rTr, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedPoseFixedFocalFixedPointScoreKernel<<<n_blocks, 1024>>>(
      sensor_from_rig, sensor_from_rig_num_alloc, principal_point,
      principal_point_num_alloc, principal_point_indices, pixel,
      pixel_num_alloc, pose, pose_num_alloc, focal, focal_num_alloc, point,
      point_num_alloc, out_rTr, problem_size);
}

} // namespace caspar