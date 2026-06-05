#include "kernel_simple_radial_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointScoreKernel(
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* pose,
        unsigned int pose_num_alloc,
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* const out_rTr,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(principal_point,
                                            0 * principal_point_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r0,
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
    r8 = -2.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r9,
                                            r10);
    ReadIdx2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r11, r12);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r13,
                                            r14);
    ReadIdx2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r15, r16);
    r17 = fma(r13, r16, r10 * r11);
    r18 = r14 * r15;
    r17 = fma(r4, r18, r17);
    r17 = fma(r9, r12, r17);
    r18 = r17 * r17;
    r18 = r8 * r18;
    r19 = 1.00000000000000000e+00;
    r20 = r13 * r11;
    r20 = fma(r4, r20, r10 * r16);
    r20 = fma(r14, r12, r20);
    r20 = fma(r9, r15, r20);
    r21 = r20 * r20;
    r21 = fma(r8, r21, r19);
    r22 = r18 + r21;
    r22 = fma(r6, r22, r0);
    r0 = 2.00000000000000000e+00;
    r23 = fma(r13, r12, r10 * r15);
    r24 = r9 * r16;
    r23 = fma(r4, r24, r23);
    r23 = fma(r14, r11, r23);
    r24 = r0 * r23;
    r25 = r20 * r24;
    r26 = fma(r14, r16, r13 * r15);
    r26 = fma(r9, r11, r26);
    r26 = fma(r4, r26, r10 * r12);
    r12 = r8 * r26;
    r27 = fma(r17, r12, r25);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r28);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r29 = r0 * r20;
    r30 = r17 * r24;
    r29 = fma(r26, r29, r30);
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r31);
    r32 = r13 * r9;
    r32 = r32 * r0;
    r33 = r14 * r10;
    r34 = fma(r0, r33, r32);
    ReadIdx2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r35, r36);
    r37 = r9 * r10;
    r38 = r13 * r14;
    r38 = r38 * r0;
    r37 = fma(r8, r37, r38);
    r39 = r14 * r14;
    r39 = r39 * r8;
    r40 = r19 + r39;
    r41 = r9 * r9;
    r41 = r8 * r41;
    r40 = r40 + r41;
    r22 = fma(r7, r27, r22);
    r22 = fma(r28, r29, r22);
    r22 = fma(r31, r34, r22);
    r22 = fma(r36, r37, r22);
    r22 = fma(r35, r40, r22);
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            0 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r40,
                                            r37);
    r34 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r29);
    r30 = fma(r20, r12, r30);
    r30 = fma(r6, r30, r29);
    r33 = fma(r8, r33, r32);
    r39 = r19 + r39;
    r32 = r13 * r13;
    r32 = r8 * r32;
    r39 = r39 + r32;
    r29 = r14 * r9;
    r29 = r29 * r0;
    r27 = r13 * r10;
    r27 = fma(r0, r27, r29);
    r42 = r0 * r17;
    r42 = r42 * r20;
    r24 = fma(r26, r24, r42);
    r43 = r23 * r23;
    r43 = r43 * r8;
    r21 = r43 + r21;
    r30 = fma(r35, r33, r30);
    r30 = fma(r31, r39, r30);
    r30 = fma(r36, r27, r30);
    r30 = fma(r7, r24, r30);
    r30 = fma(r28, r21, r30);
    r21 = copysign(1.0, r30);
    r21 = fma(r34, r21, r30);
    r34 = r21 * r21;
    r34 = 1.0 / r34;
    r30 = r22 * r22;
    r24 = r0 * r17;
    r24 = fma(r26, r24, r25);
    r24 = fma(r6, r24, r5);
    r6 = r9 * r10;
    r6 = fma(r0, r6, r38);
    r41 = r19 + r41;
    r41 = r41 + r32;
    r32 = r13 * r10;
    r32 = fma(r8, r32, r29);
    r12 = fma(r23, r12, r42);
    r18 = r19 + r18;
    r18 = r18 + r43;
    r24 = fma(r35, r6, r24);
    r24 = fma(r36, r41, r24);
    r24 = fma(r31, r32, r24);
    r24 = fma(r28, r12, r24);
    r24 = fma(r7, r18, r24);
    r18 = r24 * r24;
    r18 = fma(r34, r18, r34 * r30);
    r18 = fma(r37, r18, r19);
    r18 = r40 * r18;
    r21 = 1.0 / r21;
    r18 = r18 * r21;
    r2 = fma(r22, r18, r2);
    r4 = fma(r3, r4, r1);
    r4 = fma(r24, r18, r4);
    r4 = fma(r4, r4, r2 * r2);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r4);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointScore(
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* const out_rTr,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointScoreKernel<<<
      n_blocks,
      1024>>>(sensor_from_rig,
              sensor_from_rig_num_alloc,
              point,
              point_num_alloc,
              point_indices,
              pixel,
              pixel_num_alloc,
              pose,
              pose_num_alloc,
              focal_and_extra,
              focal_and_extra_num_alloc,
              principal_point,
              principal_point_num_alloc,
              out_rTr,
              problem_size);
}

}  // namespace caspar