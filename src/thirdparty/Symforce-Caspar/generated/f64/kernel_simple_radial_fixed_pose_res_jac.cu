#include "kernel_simple_radial_fixed_pose_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SimpleRadialFixedPoseResJacKernel(
    double* sensor_from_rig,
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
    double* out_res,
    unsigned int out_res_num_alloc,
    double* out_calib_jac,
    unsigned int out_calib_jac_num_alloc,
    double* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    double* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    double* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
    double* out_point_jac,
    unsigned int out_point_jac_num_alloc,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    double* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46;
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
    r0 = fma(r6, r22, r0);
    r23 = r17 * r8;
    r24 = fma(r14, r16, r13 * r15);
    r24 = fma(r9, r11, r24);
    r24 = fma(r4, r24, r10 * r12);
    r25 = 2.00000000000000000e+00;
    r12 = fma(r13, r12, r10 * r15);
    r26 = r9 * r16;
    r12 = fma(r4, r26, r12);
    r12 = fma(r14, r11, r12);
    r26 = r25 * r12;
    r27 = r20 * r26;
    r23 = fma(r24, r23, r27);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r28);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r29 = r25 * r20;
    r30 = r17 * r26;
    r29 = fma(r24, r29, r30);
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r31);
    r32 = r13 * r9;
    r32 = r32 * r25;
    r33 = r14 * r10;
    r34 = fma(r25, r33, r32);
    ReadIdx2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r35, r36);
    r37 = r9 * r10;
    r38 = r13 * r14;
    r38 = r38 * r25;
    r37 = fma(r8, r37, r38);
    r39 = r14 * r14;
    r39 = r39 * r8;
    r40 = r19 + r39;
    r41 = r9 * r9;
    r41 = r8 * r41;
    r40 = r40 + r41;
    r0 = fma(r7, r23, r0);
    r0 = fma(r28, r29, r0);
    r0 = fma(r31, r34, r0);
    r0 = fma(r36, r37, r0);
    r0 = fma(r35, r40, r0);
  };
  LoadShared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r40, r37);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r34 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r42);
    r43 = r8 * r20;
    r43 = fma(r24, r43, r30);
    r42 = fma(r6, r43, r42);
    r33 = fma(r8, r33, r32);
    r39 = r19 + r39;
    r32 = r13 * r13;
    r32 = r8 * r32;
    r39 = r39 + r32;
    r30 = r14 * r9;
    r30 = r30 * r25;
    r44 = r13 * r10;
    r44 = fma(r25, r44, r30);
    r45 = r25 * r17;
    r45 = r45 * r20;
    r26 = fma(r24, r26, r45);
    r46 = r12 * r12;
    r46 = r46 * r8;
    r21 = r46 + r21;
    r42 = fma(r35, r33, r42);
    r42 = fma(r31, r39, r42);
    r42 = fma(r36, r44, r42);
    r42 = fma(r7, r26, r42);
    r42 = fma(r28, r21, r42);
    r44 = copysign(1.0, r42);
    r44 = fma(r34, r44, r42);
    r34 = r44 * r44;
    r42 = 1.0 / r34;
    r39 = r0 * r0;
    r39 = r42 * r39;
    r33 = r25 * r17;
    r33 = fma(r24, r33, r27);
    r6 = fma(r6, r33, r5);
    r5 = r9 * r10;
    r5 = fma(r25, r5, r38);
    r41 = r19 + r41;
    r41 = r41 + r32;
    r32 = r13 * r10;
    r32 = fma(r8, r32, r30);
    r30 = r12 * r8;
    r30 = fma(r24, r30, r45);
    r18 = r19 + r18;
    r18 = r18 + r46;
    r6 = fma(r35, r5, r6);
    r6 = fma(r36, r41, r6);
    r6 = fma(r31, r32, r6);
    r6 = fma(r28, r30, r6);
    r6 = fma(r7, r18, r6);
    r7 = r6 * r6;
    r28 = r42 * r7;
    r32 = r39 + r28;
    r31 = fma(r37, r32, r19);
    r41 = r0 * r31;
    r36 = 1.0 / r44;
    r5 = r40 * r36;
    r2 = fma(r5, r41, r2);
    r3 = fma(r3, r4, r1);
    r1 = r6 * r31;
    r3 = fma(r5, r1, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = r0 * r31;
    r1 = r1 * r36;
    r41 = r6 * r31;
    r41 = r41 * r36;
    WriteIdx2<1024, double, double, double2>(
        out_calib_jac, 0 * out_calib_jac_num_alloc, global_thread_idx, r1, r41);
    r35 = r0 * r32;
    r35 = r35 * r5;
    r46 = r6 * r32;
    r46 = r46 * r5;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             2 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r35,
                                             r46);
    r45 = r6 * r3;
    r24 = r4 * r31;
    r45 = r45 * r36;
    r38 = r0 * r2;
    r38 = r38 * r36;
    r38 = fma(r24, r38, r24 * r45);
    r45 = r4 * r6;
    r45 = r45 * r32;
    r45 = r45 * r3;
    r36 = r4 * r0;
    r36 = r36 * r32;
    r36 = r36 * r2;
    r36 = fma(r5, r36, r5 * r45);
    WriteSum2<double, double>((double*)inout_shared, r38, r36);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = r4 * r2;
    r38 = r4 * r3;
    WriteSum2<double, double>((double*)inout_shared, r36, r38);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = r31 * r39;
    r36 = r31 * r31;
    r36 = fma(r28, r36, r31 * r38);
    r45 = r40 * r32;
    r32 = r40 * r32;
    r45 = r45 * r32;
    r45 = fma(r39, r45, r28 * r45);
    WriteSum2<double, double>((double*)inout_shared, r36, r45);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            0 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r19, r19);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            2 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = r31 * r28;
    r19 = fma(r32, r19, r32 * r38);
    WriteSum2<double, double>((double*)inout_shared, r19, r1);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r41, r35);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            2 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = 0.00000000000000000e+00;
    WriteSum2<double, double>((double*)inout_shared, r46, r35);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r0 * r42;
    r46 = r43 * r35;
    r24 = r40 * r24;
    r40 = r22 * r31;
    r40 = fma(r5, r40, r24 * r46);
    r46 = r25 * r22;
    r34 = r44 * r34;
    r34 = 1.0 / r34;
    r34 = r8 * r34;
    r44 = r43 * r34;
    r46 = fma(r7, r44, r35 * r46);
    r41 = r0 * r0;
    r46 = fma(r44, r41, r46);
    r44 = r25 * r33;
    r44 = r44 * r6;
    r46 = fma(r42, r44, r46);
    r37 = r37 * r5;
    r46 = r46 * r37;
    r40 = fma(r0, r46, r40);
    r44 = r33 * r31;
    r44 = fma(r5, r44, r6 * r46);
    r46 = r6 * r42;
    r46 = r46 * r24;
    r44 = fma(r43, r46, r44);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r40,
                                             r44);
    r41 = r23 * r31;
    r1 = r26 * r0;
    r1 = r1 * r0;
    r19 = r25 * r23;
    r19 = fma(r35, r19, r34 * r1);
    r1 = r25 * r18;
    r1 = r1 * r6;
    r19 = fma(r42, r1, r19);
    r38 = r26 * r7;
    r19 = fma(r34, r38, r19);
    r38 = r0 * r19;
    r38 = fma(r37, r38, r5 * r41);
    r41 = r26 * r35;
    r38 = fma(r24, r41, r38);
    r41 = r18 * r31;
    r41 = fma(r5, r41, r26 * r46);
    r1 = r6 * r19;
    r41 = fma(r37, r1, r41);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r38,
                                             r41);
    r1 = r25 * r29;
    r32 = r21 * r7;
    r32 = fma(r34, r32, r35 * r1);
    r1 = r21 * r0;
    r1 = r1 * r0;
    r32 = fma(r34, r1, r32);
    r34 = r25 * r30;
    r34 = r34 * r6;
    r32 = fma(r42, r34, r32);
    r34 = r0 * r32;
    r1 = r29 * r31;
    r1 = fma(r5, r1, r37 * r34);
    r34 = r21 * r35;
    r1 = fma(r24, r34, r1);
    r34 = r30 * r31;
    r24 = r6 * r32;
    r24 = fma(r37, r24, r5 * r34);
    r24 = fma(r21, r46, r24);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r1, r24);
    r46 = r4 * r3;
    r34 = r4 * r2;
    r34 = fma(r40, r34, r44 * r46);
    r46 = r4 * r2;
    r37 = r4 * r3;
    r37 = fma(r41, r37, r38 * r46);
    WriteSum2<double, double>((double*)inout_shared, r34, r37);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = r4 * r2;
    r34 = r4 * r3;
    r34 = fma(r24, r34, r1 * r37);
    WriteSum1<double, double>((double*)inout_shared, r34);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fma(r40, r40, r44 * r44);
    r37 = fma(r38, r38, r41 * r41);
    WriteSum2<double, double>((double*)inout_shared, r34, r37);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fma(r1, r1, r24 * r24);
    WriteSum1<double, double>((double*)inout_shared, r37);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fma(r44, r41, r40 * r38);
    r44 = fma(r44, r24, r40 * r1);
    WriteSum2<double, double>((double*)inout_shared, r37, r44);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fma(r38, r1, r41 * r24);
    WriteSum1<double, double>((double*)inout_shared, r1);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
}

void SimpleRadialFixedPoseResJac(double* sensor_from_rig,
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
                                 double* out_res,
                                 unsigned int out_res_num_alloc,
                                 double* out_calib_jac,
                                 unsigned int out_calib_jac_num_alloc,
                                 double* const out_calib_njtr,
                                 unsigned int out_calib_njtr_num_alloc,
                                 double* const out_calib_precond_diag,
                                 unsigned int out_calib_precond_diag_num_alloc,
                                 double* const out_calib_precond_tril,
                                 unsigned int out_calib_precond_tril_num_alloc,
                                 double* out_point_jac,
                                 unsigned int out_point_jac_num_alloc,
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
  SimpleRadialFixedPoseResJacKernel<<<n_blocks, 1024>>>(
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
      out_res,
      out_res_num_alloc,
      out_calib_jac,
      out_calib_jac_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      out_point_jac,
      out_point_jac_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar