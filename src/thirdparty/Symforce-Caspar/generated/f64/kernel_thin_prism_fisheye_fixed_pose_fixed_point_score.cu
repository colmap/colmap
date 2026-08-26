#include "kernel_thin_prism_fisheye_fixed_pose_fixed_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFixedPoseFixedPointScoreKernel(
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
  LoadShared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r2, r3);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      calib, 6 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r4, r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r6 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r7);
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r10,
                                            r11);
    ReadIdx2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r12, r13);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r14,
                                            r15);
    ReadIdx2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r16, r17);
    r18 = fma(r14, r17, r11 * r12);
    r19 = r15 * r16;
    r20 = -1.00000000000000000e+00;
    r18 = fma(r20, r19, r18);
    r18 = fma(r10, r13, r18);
    r19 = 2.00000000000000000e+00;
    r21 = fma(r14, r13, r11 * r16);
    r22 = r10 * r17;
    r21 = fma(r20, r22, r21);
    r21 = fma(r15, r12, r21);
    r22 = r19 * r21;
    r23 = r18 * r22;
    r24 = r14 * r12;
    r24 = fma(r20, r24, r11 * r17);
    r24 = fma(r15, r13, r24);
    r24 = fma(r10, r16, r24);
    r25 = -2.00000000000000000e+00;
    r26 = fma(r15, r17, r14 * r16);
    r26 = fma(r10, r12, r26);
    r26 = fma(r20, r26, r11 * r13);
    r13 = r25 * r26;
    r27 = fma(r24, r13, r23);
    r27 = fma(r8, r27, r7);
    ReadIdx2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r7, r28);
    r29 = r14 * r10;
    r29 = r29 * r19;
    r30 = r15 * r11;
    r31 = fma(r25, r30, r29);
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r32);
    r33 = 1.00000000000000000e+00;
    r34 = r15 * r15;
    r34 = r34 * r25;
    r35 = r33 + r34;
    r36 = r14 * r14;
    r36 = r25 * r36;
    r35 = r35 + r36;
    r37 = r15 * r10;
    r37 = r37 * r19;
    r38 = r14 * r11;
    r38 = fma(r19, r38, r37);
    r39 = r19 * r18;
    r39 = r39 * r24;
    r40 = fma(r26, r22, r39);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r41);
    r42 = r24 * r24;
    r42 = r25 * r42;
    r43 = r33 + r42;
    r44 = r21 * r21;
    r44 = r44 * r25;
    r43 = r43 + r44;
    r27 = fma(r7, r31, r27);
    r27 = fma(r32, r35, r27);
    r27 = fma(r28, r38, r27);
    r27 = fma(r9, r40, r27);
    r27 = fma(r41, r43, r27);
    r43 = copysign(1.0, r27);
    r43 = fma(r6, r43, r27);
    r27 = r43 * r43;
    r27 = 1.0 / r27;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r40,
                                            r38);
    r42 = r33 + r42;
    r35 = r18 * r18;
    r35 = r25 * r35;
    r42 = r42 + r35;
    r42 = fma(r8, r42, r40);
    r22 = r24 * r22;
    r40 = fma(r18, r13, r22);
    r31 = r19 * r24;
    r31 = fma(r26, r31, r23);
    r30 = fma(r19, r30, r29);
    r29 = r10 * r11;
    r23 = r14 * r15;
    r23 = r23 * r19;
    r29 = fma(r25, r29, r23);
    r45 = r10 * r10;
    r45 = fma(r25, r45, r33);
    r34 = r34 + r45;
    r42 = fma(r9, r40, r42);
    r42 = fma(r41, r31, r42);
    r42 = fma(r32, r30, r42);
    r42 = fma(r28, r29, r42);
    r42 = fma(r7, r34, r42);
    r34 = r42 * r42;
    r29 = r19 * r18;
    r29 = fma(r26, r29, r22);
    r29 = fma(r8, r29, r38);
    r8 = r10 * r11;
    r8 = fma(r19, r8, r23);
    r45 = r36 + r45;
    r36 = r14 * r11;
    r36 = fma(r25, r36, r37);
    r13 = fma(r21, r13, r39);
    r35 = r33 + r35;
    r35 = r35 + r44;
    r29 = fma(r7, r8, r29);
    r29 = fma(r28, r45, r29);
    r29 = fma(r32, r36, r29);
    r29 = fma(r41, r13, r29);
    r29 = fma(r9, r35, r29);
    r35 = r29 * r29;
    r9 = fma(r27, r35, r27 * r34);
    r9 = sqrt(r9);
    r13 = copysign(1.0, r9);
    r13 = fma(r6, r13, r9);
    r6 = r13 * r13;
    r6 = 1.0 / r6;
    r6 = r27 * r6;
    r9 = atan(r9);
    r27 = r9 * r9;
    r6 = r6 * r27;
    r27 = r6 * r35;
    r41 = 3.00000000000000000e+00;
    r41 = r41 * r6;
    r36 = fma(r34, r41, r27);
  };
  LoadShared<2, double, double>(
      calib, 10 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r32, r45);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r34 = r6 * r34;
    r27 = r27 + r34;
    r32 = fma(r32, r27, r5 * r36);
  };
  LoadShared<2, double, double>(
      calib, 4 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r36, r28);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = r27 * r27;
    r28 = fma(r28, r8, r36 * r27);
  };
  LoadShared<2, double, double>(
      calib, 8 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r36, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r44 = r8 * r8;
    r8 = r27 * r8;
    r28 = fma(r7, r44, r28);
    r28 = fma(r36, r8, r28);
    r43 = 1.0 / r43;
    r43 = r9 * r43;
    r13 = 1.0 / r13;
    r43 = r43 * r13;
    r28 = r28 * r43;
    r13 = r4 * r19;
    r13 = r13 * r42;
    r13 = r13 * r29;
    r32 = fma(r6, r13, r32);
    r32 = fma(r42, r28, r32);
    r32 = fma(r42, r43, r32);
    r32 = fma(r2, r32, r0);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r0);
    r32 = fma(r2, r20, r32);
    r41 = fma(r35, r41, r34);
    r27 = fma(r45, r27, r4 * r41);
    r45 = r5 * r19;
    r45 = r45 * r42;
    r45 = r45 * r29;
    r27 = fma(r6, r45, r27);
    r27 = fma(r29, r28, r27);
    r27 = fma(r29, r43, r27);
    r27 = fma(r3, r27, r1);
    r27 = fma(r0, r20, r27);
    r27 = fma(r27, r27, r32 * r32);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r27);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void ThinPrismFisheyeFixedPoseFixedPointScore(
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
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFixedPoseFixedPointScoreKernel<<<n_blocks, 1024>>>(
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