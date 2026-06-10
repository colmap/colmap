#include "kernel_spherical_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SphericalScoreKernel(double* pose,
                         unsigned int pose_num_alloc,
                         SharedIndex* pose_indices,
                         double* sensor_from_rig,
                         unsigned int sensor_from_rig_num_alloc,
                         double* wh,
                         unsigned int wh_num_alloc,
                         double* point,
                         unsigned int point_num_alloc,
                         SharedIndex* point_indices,
                         double* pixel,
                         unsigned int pixel_num_alloc,
                         double* const out_rTr,
                         size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        wh, 0 * wh_num_alloc, global_thread_idx, r0, r1);
    r2 = 5.00000000000000000e-01;
    r3 = 1.59154943091895346e-01;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r4,
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
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r9, r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r11,
                                            r12);
  };
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r13, r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r15,
                                            r16);
    r17 = r13 * r15;
    r18 = -1.00000000000000000e+00;
    r17 = fma(r18, r17, r10 * r12);
    r17 = fma(r14, r16, r17);
    r17 = fma(r9, r11, r17);
    r19 = r8 * r17;
    r19 = r19 * r17;
    r20 = 1.00000000000000000e+00;
    r21 = fma(r10, r15, r13 * r12);
    r22 = r9 * r16;
    r21 = fma(r18, r22, r21);
    r21 = fma(r14, r11, r21);
    r22 = r21 * r21;
    r22 = fma(r8, r22, r20);
    r23 = r19 + r22;
    r23 = fma(r6, r23, r4);
    r4 = fma(r14, r15, r9 * r12);
    r24 = r10 * r11;
    r4 = fma(r18, r24, r4);
    r4 = fma(r13, r16, r4);
    r24 = 2.00000000000000000e+00;
    r25 = r24 * r17;
    r26 = r4 * r25;
    r27 = fma(r10, r16, r9 * r15);
    r27 = fma(r13, r11, r27);
    r27 = fma(r18, r27, r14 * r12);
    r14 = r8 * r27;
    r28 = fma(r21, r14, r26);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r29);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r30 = r21 * r24;
    r30 = r30 * r4;
    r31 = fma(r27, r25, r30);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r33 = r15 * r11;
    r33 = r33 * r24;
    r34 = r16 * r12;
    r35 = fma(r24, r34, r33);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r36, r37);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r38 = r11 * r12;
    r39 = r15 * r16;
    r39 = r39 * r24;
    r38 = fma(r8, r38, r39);
    r40 = r11 * r11;
    r40 = r8 * r40;
    r41 = r20 + r40;
    r42 = r16 * r16;
    r42 = r42 * r8;
    r41 = r41 + r42;
    r23 = fma(r7, r28, r23);
    r23 = fma(r29, r31, r23);
    r23 = fma(r32, r35, r23);
    r23 = fma(r37, r38, r23);
    r23 = fma(r36, r41, r23);
    r41 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r38);
    r17 = fma(r17, r14, r30);
    r17 = fma(r6, r17, r38);
    r34 = fma(r8, r34, r33);
    r42 = r20 + r42;
    r33 = r15 * r15;
    r33 = r8 * r33;
    r42 = r42 + r33;
    r38 = r16 * r11;
    r38 = r38 * r24;
    r30 = r15 * r12;
    r30 = fma(r24, r30, r38);
    r35 = r24 * r4;
    r25 = r21 * r25;
    r35 = fma(r27, r35, r25);
    r19 = r20 + r19;
    r31 = r4 * r4;
    r31 = r8 * r31;
    r19 = r19 + r31;
    r17 = fma(r36, r34, r17);
    r17 = fma(r32, r42, r17);
    r17 = fma(r37, r30, r17);
    r17 = fma(r7, r35, r17);
    r17 = fma(r29, r19, r17);
    r19 = copysign(r41, r17);
    r19 = r19 + r17;
    r19 = atan2(r23, r19);
    r19 = fma(r3, r19, r2);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r3, r35);
    r3 = fma(r3, r18, r0 * r19);
    r19 = -3.18309886183790691e-01;
    r0 = r21 * r24;
    r0 = fma(r27, r0, r26);
    r0 = fma(r6, r0, r5);
    r6 = r11 * r12;
    r6 = fma(r24, r6, r39);
    r40 = r20 + r40;
    r40 = r40 + r33;
    r33 = r15 * r12;
    r33 = fma(r8, r33, r38);
    r14 = fma(r4, r14, r25);
    r22 = r31 + r22;
    r0 = fma(r36, r6, r0);
    r0 = fma(r37, r40, r0);
    r0 = fma(r32, r33, r0);
    r0 = fma(r29, r14, r0);
    r0 = fma(r7, r22, r0);
    r0 = r18 * r0;
    r23 = fma(r23, r23, r41);
    r23 = fma(r17, r17, r23);
    r23 = sqrt(r23);
    r41 = copysign(r41, r23);
    r23 = r41 + r23;
    r23 = atan2(r0, r23);
    r23 = fma(r19, r23, r2);
    r18 = fma(r35, r18, r1 * r23);
    r18 = fma(r18, r18, r3 * r3);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r18);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SphericalScore(double* pose,
                    unsigned int pose_num_alloc,
                    SharedIndex* pose_indices,
                    double* sensor_from_rig,
                    unsigned int sensor_from_rig_num_alloc,
                    double* wh,
                    unsigned int wh_num_alloc,
                    double* point,
                    unsigned int point_num_alloc,
                    SharedIndex* point_indices,
                    double* pixel,
                    unsigned int pixel_num_alloc,
                    double* const out_rTr,
                    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SphericalScoreKernel<<<n_blocks, 1024>>>(pose,
                                           pose_num_alloc,
                                           pose_indices,
                                           sensor_from_rig,
                                           sensor_from_rig_num_alloc,
                                           wh,
                                           wh_num_alloc,
                                           point,
                                           point_num_alloc,
                                           point_indices,
                                           pixel,
                                           pixel_num_alloc,
                                           out_rTr,
                                           problem_size);
}

}  // namespace caspar