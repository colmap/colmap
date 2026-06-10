#include "kernel_spherical_fixed_pose_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SphericalFixedPoseScoreKernel(float* sensor_from_rig,
                                  unsigned int sensor_from_rig_num_alloc,
                                  float* wh,
                                  unsigned int wh_num_alloc,
                                  float* point,
                                  unsigned int point_num_alloc,
                                  SharedIndex* point_indices,
                                  float* pixel,
                                  unsigned int pixel_num_alloc,
                                  float* pose,
                                  unsigned int pose_num_alloc,
                                  float* const out_rTr,
                                  size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        wh, 0 * wh_num_alloc, global_thread_idx, r0, r1);
    r2 = 5.00000000000000000e-01;
    r3 = 1.59154943091895346e-01;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r4,
                                         r5,
                                         r6);
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r7,
                       r8,
                       r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = -2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r11,
                                         r12,
                                         r13,
                                         r14);
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r15, r16, r17, r18);
    r19 = r11 * r17;
    r20 = -1.00000000000000000e+00;
    r19 = fmaf(r20, r19, r14 * r16);
    r19 = fmaf(r12, r18, r19);
    r19 = fmaf(r13, r15, r19);
    r21 = r10 * r19;
    r21 = r21 * r19;
    r22 = 1.00000000000000000e+00;
    r23 = fmaf(r11, r16, r14 * r17);
    r24 = r12 * r15;
    r23 = fmaf(r20, r24, r23);
    r23 = fmaf(r13, r18, r23);
    r24 = r23 * r23;
    r24 = fmaf(r10, r24, r22);
    r25 = r21 + r24;
    r25 = fmaf(r7, r25, r4);
    r4 = fmaf(r11, r18, r14 * r15);
    r26 = r13 * r16;
    r4 = fmaf(r20, r26, r4);
    r4 = fmaf(r12, r17, r4);
    r26 = 2.00000000000000000e+00;
    r27 = r26 * r19;
    r28 = r4 * r27;
    r29 = fmaf(r12, r16, r11 * r15);
    r29 = fmaf(r13, r17, r29);
    r29 = fmaf(r20, r29, r14 * r18);
    r18 = r10 * r29;
    r30 = fmaf(r23, r18, r28);
    r31 = r23 * r26;
    r31 = r31 * r4;
    r32 = fmaf(r29, r27, r31);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r33, r34, r35);
    r36 = r11 * r13;
    r36 = r36 * r26;
    r37 = r12 * r14;
    r38 = fmaf(r26, r37, r36);
    r39 = r13 * r14;
    r40 = r11 * r12;
    r40 = r40 * r26;
    r39 = fmaf(r10, r39, r40);
    r41 = r13 * r13;
    r41 = r10 * r41;
    r42 = r22 + r41;
    r43 = r12 * r12;
    r43 = r43 * r10;
    r42 = r42 + r43;
    r25 = fmaf(r8, r30, r25);
    r25 = fmaf(r9, r32, r25);
    r25 = fmaf(r35, r38, r25);
    r25 = fmaf(r34, r39, r25);
    r25 = fmaf(r33, r42, r25);
    r42 = 9.99999999999999955e-07;
    r19 = fmaf(r19, r18, r31);
    r19 = fmaf(r7, r19, r6);
    r37 = fmaf(r10, r37, r36);
    r43 = r22 + r43;
    r36 = r11 * r11;
    r36 = r10 * r36;
    r43 = r43 + r36;
    r6 = r12 * r13;
    r6 = r6 * r26;
    r31 = r11 * r14;
    r31 = fmaf(r26, r31, r6);
    r39 = r26 * r4;
    r27 = r23 * r27;
    r39 = fmaf(r29, r39, r27);
    r21 = r22 + r21;
    r38 = r4 * r4;
    r38 = r10 * r38;
    r21 = r21 + r38;
    r19 = fmaf(r33, r37, r19);
    r19 = fmaf(r35, r43, r19);
    r19 = fmaf(r34, r31, r19);
    r19 = fmaf(r8, r39, r19);
    r19 = fmaf(r9, r21, r19);
    r21 = copysignf(r42, r19);
    r21 = r21 + r19;
    r21 = atan2f(r25, r21);
    r21 = fmaf(r3, r21, r2);
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r3, r39);
    r3 = fmaf(r3, r20, r0 * r21);
    r21 = -3.18309886183790691e-01;
    r0 = r23 * r26;
    r0 = fmaf(r29, r0, r28);
    r0 = fmaf(r7, r0, r5);
    r7 = r13 * r14;
    r7 = fmaf(r26, r7, r40);
    r41 = r22 + r41;
    r41 = r41 + r36;
    r36 = r11 * r14;
    r36 = fmaf(r10, r36, r6);
    r18 = fmaf(r4, r18, r27);
    r24 = r38 + r24;
    r0 = fmaf(r33, r7, r0);
    r0 = fmaf(r34, r41, r0);
    r0 = fmaf(r35, r36, r0);
    r0 = fmaf(r9, r18, r0);
    r0 = fmaf(r8, r24, r0);
    r0 = r20 * r0;
    r25 = fmaf(r25, r25, r42);
    r25 = fmaf(r19, r19, r25);
    r25 = sqrtf(r25);
    r42 = copysignf(r42, r25);
    r25 = r42 + r25;
    r25 = atan2f(r0, r25);
    r25 = fmaf(r21, r25, r2);
    r20 = fmaf(r39, r20, r1 * r25);
    r20 = fmaf(r20, r20, r3 * r3);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r20);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void SphericalFixedPoseScore(float* sensor_from_rig,
                             unsigned int sensor_from_rig_num_alloc,
                             float* wh,
                             unsigned int wh_num_alloc,
                             float* point,
                             unsigned int point_num_alloc,
                             SharedIndex* point_indices,
                             float* pixel,
                             unsigned int pixel_num_alloc,
                             float* pose,
                             unsigned int pose_num_alloc,
                             float* const out_rTr,
                             size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SphericalFixedPoseScoreKernel<<<n_blocks, 1024>>>(sensor_from_rig,
                                                    sensor_from_rig_num_alloc,
                                                    wh,
                                                    wh_num_alloc,
                                                    point,
                                                    point_num_alloc,
                                                    point_indices,
                                                    pixel,
                                                    pixel_num_alloc,
                                                    pose,
                                                    pose_num_alloc,
                                                    out_rTr,
                                                    problem_size);
}

}  // namespace caspar