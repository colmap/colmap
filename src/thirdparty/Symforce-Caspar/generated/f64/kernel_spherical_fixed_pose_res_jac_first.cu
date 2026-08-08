#include "kernel_spherical_fixed_pose_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SphericalFixedPoseResJacFirstKernel(
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* wh,
    unsigned int wh_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    double* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
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
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46;

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
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r9,
                                            r10);
    ReadIdx2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r11, r12);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r13,
                                            r14);
    ReadIdx2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r15, r16);
    r17 = r13 * r15;
    r18 = -1.00000000000000000e+00;
    r17 = fma(r18, r17, r10 * r12);
    r17 = fma(r14, r16, r17);
    r17 = fma(r9, r11, r17);
    r19 = r8 * r17;
    r19 = r19 * r17;
    r20 = 1.00000000000000000e+00;
    r21 = fma(r13, r12, r10 * r15);
    r22 = r14 * r11;
    r21 = fma(r18, r22, r21);
    r21 = fma(r9, r16, r21);
    r22 = r21 * r21;
    r22 = fma(r8, r22, r20);
    r23 = r19 + r22;
    r4 = fma(r6, r23, r4);
    r24 = fma(r13, r16, r10 * r11);
    r25 = r9 * r12;
    r24 = fma(r18, r25, r24);
    r24 = fma(r14, r15, r24);
    r25 = 2.00000000000000000e+00;
    r26 = r25 * r17;
    r27 = r24 * r26;
    r28 = fma(r14, r12, r13 * r11);
    r28 = fma(r9, r15, r28);
    r28 = fma(r18, r28, r10 * r16);
    r16 = r8 * r28;
    r29 = fma(r21, r16, r27);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r30);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r31 = r21 * r25;
    r31 = r31 * r24;
    r32 = fma(r28, r26, r31);
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r33);
    r34 = r13 * r9;
    r34 = r34 * r25;
    r35 = r14 * r10;
    r36 = fma(r25, r35, r34);
    ReadIdx2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r37, r38);
    r39 = r9 * r10;
    r40 = r13 * r14;
    r40 = r40 * r25;
    r39 = fma(r8, r39, r40);
    r41 = r9 * r9;
    r41 = r8 * r41;
    r42 = r20 + r41;
    r43 = r14 * r14;
    r43 = r43 * r8;
    r42 = r42 + r43;
    r4 = fma(r7, r29, r4);
    r4 = fma(r30, r32, r4);
    r4 = fma(r33, r36, r4);
    r4 = fma(r38, r39, r4);
    r4 = fma(r37, r42, r4);
    r42 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r39);
    r17 = fma(r17, r16, r31);
    r39 = fma(r6, r17, r39);
    r35 = fma(r8, r35, r34);
    r43 = r20 + r43;
    r34 = r13 * r13;
    r34 = r8 * r34;
    r43 = r43 + r34;
    r31 = r14 * r9;
    r31 = r31 * r25;
    r36 = r13 * r10;
    r36 = fma(r25, r36, r31);
    r44 = r25 * r24;
    r26 = r21 * r26;
    r44 = fma(r28, r44, r26);
    r19 = r20 + r19;
    r45 = r24 * r24;
    r45 = r8 * r45;
    r19 = r19 + r45;
    r39 = fma(r37, r35, r39);
    r39 = fma(r33, r43, r39);
    r39 = fma(r38, r36, r39);
    r39 = fma(r7, r44, r39);
    r39 = fma(r30, r19, r39);
    r36 = copysign(r42, r39);
    r36 = r36 + r39;
    r43 = atan2(r4, r36);
    r43 = fma(r3, r43, r2);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r3, r35);
    r3 = fma(r3, r18, r0 * r43);
    r43 = -3.18309886183790691e-01;
    r46 = r21 * r25;
    r46 = fma(r28, r46, r27);
    r6 = fma(r6, r46, r5);
    r5 = r9 * r10;
    r5 = fma(r25, r5, r40);
    r41 = r20 + r41;
    r41 = r41 + r34;
    r34 = r13 * r10;
    r34 = fma(r8, r34, r31);
    r16 = fma(r24, r16, r26);
    r22 = r45 + r22;
    r6 = fma(r37, r5, r6);
    r6 = fma(r38, r41, r6);
    r6 = fma(r33, r34, r6);
    r6 = fma(r30, r16, r6);
    r6 = fma(r7, r22, r6);
    r7 = r18 * r6;
    r30 = r4 * r4;
    r34 = r42 + r30;
    r34 = fma(r39, r39, r34);
    r33 = sqrt(r34);
    r42 = copysign(r42, r33);
    r33 = r42 + r33;
    r7 = atan2(r7, r33);
    r7 = fma(r43, r7, r2);
    r35 = fma(r35, r18, r1 * r7);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r3, r35);
    r7 = fma(r35, r35, r3 * r3);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r7);
  if (global_thread_idx < problem_size) {
    r7 = 3.18309886183790691e-01;
    r7 = r35 * r7;
    r35 = r33 * r33;
    r43 = fma(r6, r6, r35);
    r42 = 1.0 / r43;
    r41 = r1 * r35;
    r7 = r7 * r42;
    r7 = r7 * r41;
    r42 = r25 * r23;
    r38 = r25 * r17;
    r38 = fma(r39, r38, r4 * r42);
    r6 = r2 * r6;
    r2 = 1.0 / r35;
    r34 = rsqrt(r34);
    r6 = r6 * r2;
    r6 = r6 * r34;
    r34 = r18 * r46;
    r33 = 1.0 / r33;
    r34 = fma(r33, r34, r38 * r6);
    r38 = -1.59154943091895346e-01;
    r38 = r3 * r38;
    r3 = r36 * r36;
    r30 = r30 + r3;
    r2 = 1.0 / r30;
    r42 = r0 * r3;
    r38 = r38 * r2;
    r38 = r38 * r42;
    r2 = r18 * r4;
    r5 = 1.0 / r3;
    r2 = r2 * r5;
    r36 = 1.0 / r36;
    r5 = fma(r23, r36, r17 * r2);
    r37 = fma(r5, r38, r34 * r7);
    r45 = r25 * r44;
    r26 = r25 * r29;
    r26 = fma(r4, r26, r39 * r45);
    r45 = r18 * r22;
    r45 = fma(r33, r45, r26 * r6);
    r26 = fma(r29, r36, r44 * r2);
    r31 = fma(r26, r38, r45 * r7);
    WriteSum2<double, double>((double*)inout_shared, r37, r31);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = fma(r19, r2, r32 * r36);
    r36 = r18 * r16;
    r31 = r25 * r32;
    r37 = r25 * r19;
    r37 = fma(r39, r37, r4 * r31);
    r6 = fma(r37, r6, r33 * r36);
    r7 = fma(r6, r7, r2 * r38);
    WriteSum1<double, double>((double*)inout_shared, r7);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = 2.53302959105844473e-02;
    r7 = r0 * r7;
    r30 = r30 * r30;
    r30 = 1.0 / r30;
    r7 = r7 * r30;
    r7 = r7 * r42;
    r7 = r7 * r3;
    r3 = r5 * r7;
    r42 = 1.01321183642337789e-01;
    r42 = r1 * r42;
    r43 = r43 * r43;
    r43 = 1.0 / r43;
    r42 = r42 * r43;
    r42 = r42 * r41;
    r42 = r42 * r35;
    r35 = r34 * r42;
    r34 = fma(r34, r35, r5 * r3);
    r5 = r26 * r26;
    r41 = r45 * r45;
    r41 = fma(r42, r41, r7 * r5);
    WriteSum2<double, double>((double*)inout_shared, r34, r41);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = r6 * r6;
    r34 = r2 * r2;
    r34 = fma(r7, r34, r42 * r41);
    WriteSum1<double, double>((double*)inout_shared, r34);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fma(r26, r3, r45 * r35);
    r3 = fma(r2, r3, r6 * r35);
    WriteSum2<double, double>((double*)inout_shared, r34, r3);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r45 * r6;
    r34 = r26 * r2;
    r34 = fma(r7, r34, r42 * r3);
    WriteSum1<double, double>((double*)inout_shared, r34);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SphericalFixedPoseResJacFirst(
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* wh,
    unsigned int wh_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    double* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SphericalFixedPoseResJacFirstKernel<<<n_blocks, 1024>>>(
      sensor_from_rig,
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
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar