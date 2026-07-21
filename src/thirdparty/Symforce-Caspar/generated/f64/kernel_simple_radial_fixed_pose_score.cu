#include "kernel_simple_radial_fixed_pose_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialFixedPoseScoreKernel(double* sensor_from_rig,
                                     unsigned int sensor_from_rig_num_alloc,
                                     double* calib,
                                     unsigned int calib_num_alloc,
                                     SharedIndex* calib_indices,
                                     double* point,
                                     unsigned int point_num_alloc,
                                     SharedIndex* point_indices,
                                     double* pixel,
                                     unsigned int pixel_num_alloc,
                                     double* pose,
                                     unsigned int pose_num_alloc,
                                     double* const out_rTr,
                                     size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44;
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
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r1,
                                            r5);
  };
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r8,
                                            r9);
    ReadIdx2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r10, r11);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r12,
                                            r13);
    ReadIdx2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r14, r15);
    r16 = r12 * r14;
    r16 = fma(r4, r16, r9 * r11);
    r16 = fma(r13, r15, r16);
    r16 = fma(r8, r10, r16);
    r17 = 2.00000000000000000e+00;
    r18 = fma(r12, r15, r9 * r10);
    r19 = r8 * r11;
    r18 = fma(r4, r19, r18);
    r18 = fma(r13, r14, r18);
    r19 = r17 * r18;
    r20 = r16 * r19;
    r21 = fma(r12, r11, r9 * r14);
    r22 = r13 * r10;
    r21 = fma(r4, r22, r21);
    r21 = fma(r8, r15, r21);
    r22 = fma(r13, r11, r12 * r10);
    r22 = fma(r8, r14, r22);
    r22 = fma(r4, r22, r9 * r15);
    r15 = r21 * r22;
    r23 = fma(r17, r15, r20);
    r23 = fma(r6, r23, r5);
    ReadIdx2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r5, r24);
    r25 = r8 * r9;
    r26 = r12 * r13;
    r26 = r26 * r17;
    r25 = fma(r17, r25, r26);
    r27 = -2.00000000000000000e+00;
    r28 = r8 * r8;
    r28 = r27 * r28;
    r29 = 1.00000000000000000e+00;
    r30 = r12 * r12;
    r30 = fma(r27, r30, r29);
    r31 = r28 + r30;
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r32);
    r33 = r13 * r8;
    r33 = r33 * r17;
    r34 = r9 * r27;
    r35 = fma(r12, r34, r33);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r36);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r37 = r17 * r21;
    r37 = r37 * r16;
    r38 = r18 * r27;
    r38 = fma(r22, r38, r37);
    r39 = r18 * r18;
    r39 = r39 * r27;
    r40 = r29 + r39;
    r41 = r21 * r21;
    r41 = r41 * r27;
    r40 = r40 + r41;
    r23 = fma(r5, r25, r23);
    r23 = fma(r24, r31, r23);
    r23 = fma(r32, r35, r23);
    r23 = fma(r36, r38, r23);
    r23 = fma(r7, r40, r23);
  };
  LoadShared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r40, r38);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r35 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r31);
    r25 = r27 * r16;
    r21 = r21 * r19;
    r25 = fma(r22, r25, r21);
    r25 = fma(r6, r25, r31);
    r31 = r12 * r8;
    r31 = r31 * r17;
    r42 = fma(r13, r34, r31);
    r43 = r13 * r13;
    r43 = r27 * r43;
    r30 = r43 + r30;
    r44 = r12 * r9;
    r44 = fma(r17, r44, r33);
    r19 = fma(r22, r19, r37);
    r39 = r29 + r39;
    r37 = r16 * r16;
    r37 = r27 * r37;
    r39 = r39 + r37;
    r25 = fma(r5, r42, r25);
    r25 = fma(r32, r30, r25);
    r25 = fma(r24, r44, r25);
    r25 = fma(r7, r19, r25);
    r25 = fma(r36, r39, r25);
    r39 = copysign(1.0, r25);
    r39 = fma(r35, r39, r25);
    r35 = r39 * r39;
    r35 = 1.0 / r35;
    r41 = r29 + r41;
    r41 = r41 + r37;
    r41 = fma(r6, r41, r1);
    r15 = fma(r27, r15, r20);
    r20 = r17 * r16;
    r20 = fma(r22, r20, r21);
    r21 = r13 * r9;
    r21 = fma(r17, r21, r31);
    r34 = fma(r8, r34, r26);
    r28 = r29 + r28;
    r28 = r28 + r43;
    r41 = fma(r7, r15, r41);
    r41 = fma(r36, r20, r41);
    r41 = fma(r32, r21, r41);
    r41 = fma(r24, r34, r41);
    r41 = fma(r5, r28, r41);
    r28 = r41 * r41;
    r5 = r23 * r23;
    r5 = fma(r35, r5, r35 * r28);
    r5 = fma(r38, r5, r29);
    r5 = r40 * r5;
    r39 = 1.0 / r39;
    r5 = r5 * r39;
    r3 = fma(r23, r5, r3);
    r4 = fma(r2, r4, r0);
    r4 = fma(r41, r5, r4);
    r4 = fma(r4, r4, r3 * r3);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r4);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialFixedPoseScore(double* sensor_from_rig,
                                unsigned int sensor_from_rig_num_alloc,
                                double* calib,
                                unsigned int calib_num_alloc,
                                SharedIndex* calib_indices,
                                double* point,
                                unsigned int point_num_alloc,
                                SharedIndex* point_indices,
                                double* pixel,
                                unsigned int pixel_num_alloc,
                                double* pose,
                                unsigned int pose_num_alloc,
                                double* const out_rTr,
                                size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialFixedPoseScoreKernel<<<n_blocks, 1024>>>(
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      calib,
      calib_num_alloc,
      calib_indices,
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