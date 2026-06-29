#include "kernel_simple_radial_split_fixed_pose_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPoseFixedPrincipalPointResJacFirstKernel(
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        SharedIndex* focal_and_extra_indices,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* pose,
        unsigned int pose_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
        double* out_focal_and_extra_jac,
        unsigned int out_focal_and_extra_jac_num_alloc,
        double* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        double* const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        double* const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
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

  __shared__ SharedIndex focal_and_extra_indices_loc[1024];
  focal_and_extra_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
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
  LoadShared<2, double, double>(focal_and_extra,
                                0 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target,
                        r40,
                        r37);
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
    r19 = fma(r37, r32, r19);
    r31 = r0 * r19;
    r41 = 1.0 / r44;
    r36 = r40 * r41;
    r2 = fma(r36, r31, r2);
    r3 = fma(r3, r4, r1);
    r1 = r6 * r19;
    r3 = fma(r36, r1, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = fma(r3, r3, r2 * r2);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r1);
  if (global_thread_idx < problem_size) {
    r1 = r0 * r19;
    r1 = r1 * r41;
    r31 = r6 * r19;
    r31 = r31 * r41;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        0 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r1,
        r31);
    r31 = r0 * r32;
    r31 = r31 * r36;
    r1 = r6 * r32;
    r1 = r1 * r36;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        2 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r31,
        r1);
    r1 = r0 * r2;
    r31 = r4 * r19;
    r1 = r1 * r41;
    r5 = r6 * r3;
    r5 = r5 * r41;
    r5 = fma(r31, r5, r31 * r1);
    r1 = r4 * r6;
    r1 = r1 * r32;
    r1 = r1 * r3;
    r41 = r4 * r0;
    r41 = r41 * r32;
    r41 = r41 * r2;
    r41 = fma(r36, r41, r36 * r1);
    WriteSum2<double, double>((double*)inout_shared, r5, r41);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            0 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = r19 * r39;
    r5 = r19 * r19;
    r5 = fma(r28, r5, r19 * r41);
    r1 = r40 * r32;
    r32 = r40 * r32;
    r1 = r1 * r32;
    r1 = fma(r39, r1, r28 * r1);
    WriteSum2<double, double>((double*)inout_shared, r5, r1);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            0 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r19 * r28;
    r1 = fma(r32, r1, r32 * r41);
    WriteSum1<double, double>((double*)inout_shared, r1);
  };
  FlushSumShared<1, double>(out_focal_and_extra_precond_tril,
                            0 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r25 * r22;
    r41 = r0 * r42;
    r34 = r44 * r34;
    r34 = 1.0 / r34;
    r34 = r8 * r34;
    r44 = r43 * r34;
    r1 = fma(r7, r44, r41 * r1);
    r32 = r0 * r0;
    r1 = fma(r44, r32, r1);
    r44 = r25 * r33;
    r44 = r44 * r6;
    r1 = fma(r42, r44, r1);
    r37 = r37 * r36;
    r1 = r1 * r37;
    r44 = r43 * r41;
    r31 = r40 * r31;
    r44 = fma(r31, r44, r0 * r1);
    r40 = r22 * r19;
    r44 = fma(r36, r40, r44);
    r40 = r6 * r42;
    r40 = r40 * r31;
    r1 = fma(r6, r1, r43 * r40);
    r32 = r33 * r19;
    r1 = fma(r36, r32, r1);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r44, r1);
    r32 = r26 * r0;
    r32 = r32 * r0;
    r5 = r25 * r23;
    r5 = fma(r41, r5, r34 * r32);
    r32 = r25 * r18;
    r32 = r32 * r6;
    r5 = fma(r42, r32, r5);
    r39 = r26 * r7;
    r5 = fma(r34, r39, r5);
    r39 = r0 * r5;
    r32 = r23 * r19;
    r32 = fma(r36, r32, r37 * r39);
    r39 = r26 * r41;
    r32 = fma(r31, r39, r32);
    r39 = r6 * r5;
    r39 = fma(r26, r40, r37 * r39);
    r35 = r18 * r19;
    r39 = fma(r36, r35, r39);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r32,
                                             r39);
    r35 = r25 * r29;
    r46 = r21 * r7;
    r46 = fma(r34, r46, r41 * r35);
    r35 = r21 * r0;
    r35 = r35 * r0;
    r46 = fma(r34, r35, r46);
    r34 = r25 * r30;
    r34 = r34 * r6;
    r46 = fma(r42, r34, r46);
    r34 = r0 * r46;
    r35 = r29 * r19;
    r35 = fma(r36, r35, r37 * r34);
    r34 = r21 * r41;
    r35 = fma(r31, r34, r35);
    r34 = r6 * r46;
    r40 = fma(r21, r40, r37 * r34);
    r34 = r30 * r19;
    r40 = fma(r36, r34, r40);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r35,
                                             r40);
    r34 = r4 * r2;
    r36 = r4 * r3;
    r36 = fma(r1, r36, r44 * r34);
    r34 = r4 * r3;
    r37 = r4 * r2;
    r37 = fma(r32, r37, r39 * r34);
    WriteSum2<double, double>((double*)inout_shared, r36, r37);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = r4 * r2;
    r36 = r4 * r3;
    r36 = fma(r40, r36, r35 * r37);
    WriteSum1<double, double>((double*)inout_shared, r36);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = fma(r1, r1, r44 * r44);
    r37 = fma(r39, r39, r32 * r32);
    WriteSum2<double, double>((double*)inout_shared, r36, r37);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fma(r35, r35, r40 * r40);
    WriteSum1<double, double>((double*)inout_shared, r37);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fma(r1, r39, r44 * r32);
    r1 = fma(r1, r40, r44 * r35);
    WriteSum2<double, double>((double*)inout_shared, r37, r1);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fma(r32, r35, r39 * r40);
    WriteSum1<double, double>((double*)inout_shared, r35);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedPoseFixedPrincipalPointResJacFirst(
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* out_focal_and_extra_jac,
    unsigned int out_focal_and_extra_jac_num_alloc,
    double* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    double* const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    double* const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
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
  SimpleRadialSplitFixedPoseFixedPrincipalPointResJacFirstKernel<<<n_blocks,
                                                                   1024>>>(
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
      focal_and_extra_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      principal_point,
      principal_point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_focal_and_extra_jac,
      out_focal_and_extra_jac_num_alloc,
      out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc,
      out_focal_and_extra_precond_diag,
      out_focal_and_extra_precond_diag_num_alloc,
      out_focal_and_extra_precond_tril,
      out_focal_and_extra_precond_tril_num_alloc,
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