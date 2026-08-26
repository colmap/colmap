#include "kernel_pinhole_fixed_pose_fixed_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedPoseFixedPointScoreKernel(
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* pose,
        unsigned int pose_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* const out_rTr,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45;
  LoadShared<2, double, double>(
      calib, 2 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r3 = fma(r3, r4, r1);
  };
  LoadShared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r1, r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r6,
                                            r7);
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r10,
                                            r11);
    ReadIdx2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r12, r13);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r14,
                                            r15);
    ReadIdx2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r16, r17);
    r18 = r14 * r16;
    r18 = fma(r4, r18, r11 * r13);
    r18 = fma(r15, r17, r18);
    r18 = fma(r10, r12, r18);
    r19 = 2.00000000000000000e+00;
    r20 = fma(r14, r17, r11 * r12);
    r21 = r10 * r13;
    r20 = fma(r4, r21, r20);
    r20 = fma(r15, r16, r20);
    r21 = r19 * r20;
    r22 = r18 * r21;
    r23 = fma(r14, r13, r11 * r16);
    r24 = r15 * r12;
    r23 = fma(r4, r24, r23);
    r23 = fma(r10, r17, r23);
    r24 = fma(r15, r13, r14 * r12);
    r24 = fma(r10, r16, r24);
    r24 = fma(r4, r24, r11 * r17);
    r17 = r23 * r24;
    r25 = fma(r19, r17, r22);
    r25 = fma(r8, r25, r7);
    ReadIdx2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r7, r26);
    r27 = r10 * r11;
    r28 = r14 * r15;
    r28 = r28 * r19;
    r27 = fma(r19, r27, r28);
    r29 = -2.00000000000000000e+00;
    r30 = r10 * r10;
    r30 = r29 * r30;
    r31 = 1.00000000000000000e+00;
    r32 = r14 * r14;
    r32 = fma(r29, r32, r31);
    r33 = r30 + r32;
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r34);
    r35 = r15 * r10;
    r35 = r35 * r19;
    r36 = r11 * r29;
    r37 = fma(r14, r36, r35);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r38);
    r39 = r19 * r23;
    r39 = r39 * r18;
    r40 = r20 * r29;
    r40 = fma(r24, r40, r39);
    r41 = r20 * r20;
    r41 = r41 * r29;
    r42 = r31 + r41;
    r43 = r23 * r23;
    r43 = r43 * r29;
    r42 = r42 + r43;
    r25 = fma(r7, r27, r25);
    r25 = fma(r26, r33, r25);
    r25 = fma(r34, r37, r25);
    r25 = fma(r38, r40, r25);
    r25 = fma(r9, r42, r25);
    r42 = r5 * r25;
    r40 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r37);
    r33 = r29 * r18;
    r23 = r23 * r21;
    r33 = fma(r24, r33, r23);
    r33 = fma(r8, r33, r37);
    r37 = r14 * r10;
    r37 = r37 * r19;
    r27 = fma(r15, r36, r37);
    r44 = r15 * r15;
    r44 = r29 * r44;
    r32 = r44 + r32;
    r45 = r14 * r11;
    r45 = fma(r19, r45, r35);
    r21 = fma(r24, r21, r39);
    r41 = r31 + r41;
    r39 = r18 * r18;
    r39 = r29 * r39;
    r41 = r41 + r39;
    r33 = fma(r7, r27, r33);
    r33 = fma(r34, r32, r33);
    r33 = fma(r26, r45, r33);
    r33 = fma(r9, r21, r33);
    r33 = fma(r38, r41, r33);
    r41 = copysign(1.0, r33);
    r41 = fma(r40, r41, r33);
    r41 = 1.0 / r41;
    r3 = fma(r41, r42, r3);
    r4 = fma(r2, r4, r0);
    r43 = r31 + r43;
    r43 = r43 + r39;
    r43 = fma(r8, r43, r6);
    r17 = fma(r29, r17, r22);
    r22 = r19 * r18;
    r22 = fma(r24, r22, r23);
    r23 = r15 * r11;
    r23 = fma(r19, r23, r37);
    r36 = fma(r10, r36, r28);
    r30 = r31 + r30;
    r30 = r30 + r44;
    r43 = fma(r9, r17, r43);
    r43 = fma(r38, r22, r43);
    r43 = fma(r34, r23, r43);
    r43 = fma(r26, r36, r43);
    r43 = fma(r7, r30, r43);
    r30 = r1 * r43;
    r4 = fma(r41, r30, r4);
    r4 = fma(r4, r4, r3 * r3);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r4);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeFixedPoseFixedPointScore(double* sensor_from_rig,
                                     unsigned int sensor_from_rig_num_alloc,
                                     double* calib,
                                     unsigned int calib_num_alloc,
                                     SharedIndex* calib_indices,
                                     double* pixel,
                                     unsigned int pixel_num_alloc,
                                     double* pose,
                                     unsigned int pose_num_alloc,
                                     double* point,
                                     unsigned int point_num_alloc,
                                     double* const out_rTr,
                                     size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFixedPoseFixedPointScoreKernel<<<n_blocks, 1024>>>(
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      point,
      point_num_alloc,
      out_rTr,
      problem_size);
}

}  // namespace caspar