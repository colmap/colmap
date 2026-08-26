#include "kernel_thin_prism_fisheye_split_fixed_pose_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointResJacKernel(
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        SharedIndex* focal_and_extra_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* pose,
        unsigned int pose_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        double* const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        double* const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex focal_and_extra_indices_loc[1024];
  focal_and_extra_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(principal_point,
                                            0 * principal_point_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1);
  };
  LoadShared<2, double, double>(focal_and_extra,
                                0 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target,
                        r2,
                        r3);
  };
  __syncthreads();
  LoadShared<2, double, double>(focal_and_extra,
                                4 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target,
                        r4,
                        r5);
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
    r10 = 2.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r11,
                                            r12);
    ReadIdx2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r13, r14);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r15,
                                            r16);
    ReadIdx2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r17, r18);
    r19 = fma(r15, r18, r12 * r13);
    r20 = r11 * r14;
    r21 = -1.00000000000000000e+00;
    r19 = fma(r21, r20, r19);
    r19 = fma(r16, r17, r19);
    r20 = r10 * r19;
    r22 = r15 * r17;
    r22 = fma(r21, r22, r12 * r14);
    r22 = fma(r16, r18, r22);
    r22 = fma(r11, r13, r22);
    r20 = r20 * r22;
    r23 = fma(r15, r14, r12 * r17);
    r24 = r16 * r13;
    r23 = fma(r21, r24, r23);
    r23 = fma(r11, r18, r23);
    r24 = r10 * r23;
    r25 = fma(r16, r14, r15 * r13);
    r25 = fma(r11, r17, r25);
    r25 = fma(r21, r25, r12 * r18);
    r24 = fma(r25, r24, r20);
    r24 = fma(r8, r24, r7);
    ReadIdx2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r7, r18);
    r26 = r15 * r16;
    r26 = r26 * r10;
    r27 = r11 * r12;
    r27 = fma(r10, r27, r26);
    r28 = -2.00000000000000000e+00;
    r29 = r15 * r15;
    r29 = r28 * r29;
    r30 = 1.00000000000000000e+00;
    r31 = r11 * r11;
    r31 = fma(r28, r31, r30);
    r32 = r29 + r31;
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r33);
    r34 = r16 * r11;
    r34 = r34 * r10;
    r35 = r15 * r12;
    r35 = fma(r28, r35, r34);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r36);
    r37 = r10 * r23;
    r37 = r37 * r22;
    r38 = r28 * r25;
    r39 = fma(r19, r38, r37);
    r40 = r23 * r23;
    r40 = r28 * r40;
    r41 = r30 + r40;
    r42 = r19 * r19;
    r42 = r28 * r42;
    r41 = r41 + r42;
    r24 = fma(r7, r27, r24);
    r24 = fma(r18, r32, r24);
    r24 = fma(r33, r35, r24);
    r24 = fma(r36, r39, r24);
    r24 = fma(r9, r41, r24);
    r41 = r24 * r24;
    r39 = r22 * r22;
    r39 = r28 * r39;
    r35 = r30 + r39;
    r35 = r35 + r40;
    r35 = fma(r8, r35, r6);
    r20 = fma(r23, r38, r20);
    r6 = r10 * r23;
    r6 = r6 * r19;
    r40 = r10 * r22;
    r40 = fma(r25, r40, r6);
    r32 = r15 * r11;
    r32 = r32 * r10;
    r27 = r16 * r12;
    r43 = fma(r10, r27, r32);
    r44 = r11 * r12;
    r44 = fma(r28, r44, r26);
    r26 = r16 * r16;
    r26 = r26 * r28;
    r31 = r26 + r31;
    r35 = fma(r9, r20, r35);
    r35 = fma(r36, r40, r35);
    r35 = fma(r33, r43, r35);
    r35 = fma(r18, r44, r35);
    r35 = fma(r7, r31, r35);
    r31 = r35 * r35;
    r44 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r43);
    r38 = fma(r22, r38, r6);
    r38 = fma(r8, r38, r43);
    r27 = fma(r28, r27, r32);
    r26 = r30 + r26;
    r26 = r26 + r29;
    r29 = r15 * r12;
    r29 = fma(r10, r29, r34);
    r34 = r10 * r19;
    r34 = fma(r25, r34, r37);
    r39 = r30 + r39;
    r39 = r39 + r42;
    r38 = fma(r7, r27, r38);
    r38 = fma(r33, r26, r38);
    r38 = fma(r18, r29, r38);
    r38 = fma(r9, r34, r38);
    r38 = fma(r36, r39, r38);
    r39 = copysign(1.0, r38);
    r39 = fma(r44, r39, r38);
    r38 = r39 * r39;
    r36 = 1.0 / r38;
    r34 = r24 * r24;
    r34 = fma(r36, r34, r36 * r31);
    r34 = sqrt(r34);
    r31 = atan(r34);
    r9 = copysign(1.0, r34);
    r9 = fma(r44, r9, r34);
    r44 = r9 * r9;
    r34 = 1.0 / r44;
    r34 = r36 * r34;
    r36 = r31 * r34;
    r41 = r41 * r31;
    r41 = r41 * r36;
    r29 = 3.00000000000000000e+00;
    r18 = r35 * r29;
    r26 = r35 * r31;
    r18 = r18 * r26;
    r18 = fma(r36, r18, r41);
  };
  LoadShared<2, double, double>(focal_and_extra,
                                8 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target,
                        r33,
                        r27);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r7 = r35 * r26;
    r7 = r7 * r36;
    r41 = r41 + r7;
    r33 = fma(r33, r41, r5 * r18);
    r42 = r24 * r31;
    r30 = r4 * r42;
    r37 = r10 * r26;
    r25 = r34 * r37;
    r33 = fma(r25, r30, r33);
  };
  LoadShared<2, double, double>(focal_and_extra,
                                2 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target,
                        r32,
                        r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r43 = r41 * r41;
    r8 = fma(r8, r43, r32 * r41);
  };
  LoadShared<2, double, double>(focal_and_extra,
                                6 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target,
                        r32,
                        r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r40 = r43 * r43;
    r20 = r41 * r43;
    r8 = fma(r6, r40, r8);
    r8 = fma(r32, r20, r8);
    r32 = r8 * r26;
    r6 = 1.0 / r39;
    r45 = 1.0 / r9;
    r46 = r6 * r45;
    r33 = fma(r46, r32, r33);
    r33 = fma(r26, r46, r33);
    r0 = fma(r2, r33, r0);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r32, r30);
    r0 = fma(r32, r21, r0);
    r32 = r24 * r24;
    r32 = r32 * r31;
    r32 = r32 * r29;
    r32 = fma(r36, r32, r7);
    r27 = fma(r27, r41, r4 * r32);
    r7 = r8 * r46;
    r27 = fma(r42, r7, r27);
    r47 = r5 * r42;
    r27 = fma(r25, r47, r27);
    r27 = fma(r46, r42, r27);
    r1 = fma(r3, r27, r1);
    r1 = fma(r30, r21, r1);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r0, r1);
    r30 = r21 * r27;
    r30 = r30 * r1;
    r47 = r21 * r0;
    r7 = r33 * r47;
    WriteSum2<double, double>((double*)inout_shared, r7, r30);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            0 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = r21 * r41;
    r7 = r3 * r42;
    r30 = r30 * r1;
    r30 = r30 * r46;
    r48 = r2 * r26;
    r49 = r48 * r47;
    r50 = r46 * r49;
    r30 = fma(r41, r50, r7 * r30);
    r51 = r21 * r1;
    r52 = r43 * r46;
    r51 = r51 * r7;
    r49 = fma(r52, r49, r52 * r51);
    WriteSum2<double, double>((double*)inout_shared, r30, r49);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            2 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r3 * r21;
    r49 = r49 * r32;
    r30 = r28 * r0;
    r30 = r30 * r42;
    r30 = r30 * r48;
    r30 = fma(r34, r30, r1 * r49);
    r49 = r2 * r18;
    r51 = r28 * r1;
    r51 = r51 * r26;
    r51 = r51 * r7;
    r51 = fma(r34, r51, r47 * r49);
    WriteSum2<double, double>((double*)inout_shared, r30, r51);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            4 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = r21 * r1;
    r51 = r51 * r20;
    r51 = r51 * r46;
    r51 = fma(r20, r50, r7 * r51);
    r30 = r21 * r1;
    r30 = r30 * r46;
    r30 = r30 * r7;
    r50 = fma(r40, r50, r40 * r30);
    WriteSum2<double, double>((double*)inout_shared, r51, r50);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            6 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r3 * r21;
    r50 = r50 * r41;
    r50 = r50 * r1;
    r51 = r2 * r41;
    r51 = r51 * r47;
    WriteSum2<double, double>((double*)inout_shared, r51, r50);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            8 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r33 * r33;
    r51 = r27 * r27;
    WriteSum2<double, double>((double*)inout_shared, r50, r51);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            0 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = r2 * r48;
    r50 = r35 * r51;
    r47 = r36 * r50;
    r30 = r24 * r24;
    r49 = r3 * r3;
    r30 = r30 * r31;
    r30 = r30 * r43;
    r30 = r30 * r36;
    r30 = fma(r49, r30, r43 * r47);
    r53 = r40 * r36;
    r54 = r3 * r7;
    r55 = r24 * r54;
    r53 = fma(r55, r53, r40 * r47);
    WriteSum2<double, double>((double*)inout_shared, r30, r53);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            2 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = r24 * r42;
    r56 = r31 * r31;
    r57 = 4.00000000000000000e+00;
    r38 = r39 * r38;
    r39 = r39 * r38;
    r39 = 1.0 / r39;
    r44 = r9 * r44;
    r9 = r9 * r44;
    r9 = 1.0 / r9;
    r56 = r56 * r57;
    r56 = r56 * r39;
    r56 = r56 * r9;
    r30 = r30 * r50;
    r9 = r32 * r32;
    r49 = fma(r9, r49, r56 * r30);
    r9 = r35 * r26;
    r9 = r9 * r55;
    r30 = r2 * r2;
    r39 = r18 * r18;
    r30 = fma(r39, r30, r56 * r9);
    WriteSum2<double, double>((double*)inout_shared, r49, r30);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            4 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = r20 * r20;
    r49 = r36 * r55;
    r49 = fma(r30, r49, r30 * r47);
    r9 = r40 * r40;
    r39 = r36 * r55;
    r39 = fma(r9, r39, r47 * r9);
    WriteSum2<double, double>((double*)inout_shared, r49, r39);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            6 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r2 * r2;
    r39 = r39 * r43;
    r9 = r3 * r3;
    r9 = r9 * r43;
    WriteSum2<double, double>((double*)inout_shared, r39, r9);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            8 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = 0.00000000000000000e+00;
    r39 = r41 * r33;
    r39 = r39 * r46;
    r39 = r39 * r48;
    WriteSum2<double, double>((double*)inout_shared, r9, r39);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            0 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r33 * r48;
    r39 = r39 * r52;
    r56 = r10 * r33;
    r56 = r56 * r42;
    r56 = r56 * r48;
    r56 = r56 * r34;
    WriteSum2<double, double>((double*)inout_shared, r39, r56);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            2 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = r2 * r18;
    r56 = r56 * r33;
    r39 = r33 * r20;
    r39 = r39 * r46;
    r39 = r39 * r48;
    WriteSum2<double, double>((double*)inout_shared, r56, r39);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            4 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r2 * r41;
    r39 = r39 * r33;
    r33 = r33 * r46;
    r33 = r33 * r48;
    r33 = r33 * r40;
    WriteSum2<double, double>((double*)inout_shared, r33, r39);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            6 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r41 * r27;
    r39 = r39 * r46;
    r39 = r39 * r7;
    WriteSum2<double, double>((double*)inout_shared, r9, r39);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            8 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r3 * r32;
    r39 = r39 * r27;
    r33 = r27 * r7;
    r48 = r52 * r33;
    WriteSum2<double, double>((double*)inout_shared, r48, r39);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            10 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = r25 * r33;
    r39 = r27 * r20;
    r39 = r39 * r46;
    r39 = r39 * r7;
    WriteSum2<double, double>((double*)inout_shared, r33, r39);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            12 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r27 * r46;
    r39 = r39 * r7;
    r39 = r39 * r40;
    WriteSum2<double, double>((double*)inout_shared, r39, r9);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            14 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = r3 * r41;
    r9 = r9 * r27;
    r27 = r20 * r55;
    r39 = fma(r36, r27, r20 * r47);
    WriteSum2<double, double>((double*)inout_shared, r9, r39);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            16 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r41 * r32;
    r39 = r39 * r46;
    r9 = r10 * r42;
    r38 = 1.0 / r38;
    r38 = r31 * r38;
    r44 = 1.0 / r44;
    r38 = r38 * r44;
    r9 = r9 * r50;
    r9 = r9 * r38;
    r39 = fma(r41, r9, r54 * r39);
    r50 = r18 * r41;
    r50 = r50 * r46;
    r44 = r41 * r55;
    r44 = r44 * r37;
    r44 = fma(r38, r44, r51 * r50);
    WriteSum2<double, double>((double*)inout_shared, r39, r44);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            18 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = r41 * r40;
    r39 = r36 * r44;
    r39 = fma(r55, r39, r44 * r47);
    WriteSum2<double, double>((double*)inout_shared, r53, r39);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            20 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r35 * r31;
    r53 = r2 * r2;
    r31 = r31 * r43;
    r31 = r31 * r6;
    r31 = r31 * r45;
    r31 = r31 * r53;
    r53 = r54 * r52;
    WriteSum2<double, double>((double*)inout_shared, r31, r53);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            22 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = r32 * r54;
    r53 = fma(r43, r9, r52 * r53);
    r52 = r43 * r55;
    r52 = r52 * r37;
    r52 = fma(r38, r52, r18 * r31);
    WriteSum2<double, double>((double*)inout_shared, r53, r52);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            24 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r39, r49);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            26 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r20 * r46;
    r49 = r49 * r51;
    r39 = r20 * r46;
    r39 = r39 * r54;
    WriteSum2<double, double>((double*)inout_shared, r49, r39);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            28 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r10 * r18;
    r39 = r39 * r42;
    r39 = r39 * r34;
    r25 = r54 * r25;
    r39 = fma(r32, r25, r51 * r39);
    r49 = r32 * r20;
    r49 = r49 * r46;
    r49 = fma(r20, r9, r54 * r49);
    WriteSum2<double, double>((double*)inout_shared, r39, r49);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            30 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r10 * r41;
    r49 = r49 * r42;
    r49 = r49 * r34;
    r49 = r49 * r51;
    r34 = r32 * r46;
    r34 = r34 * r40;
    r9 = fma(r40, r9, r54 * r34);
    WriteSum2<double, double>((double*)inout_shared, r9, r49);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            32 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r3 * r3;
    r49 = r49 * r41;
    r49 = r49 * r32;
    r9 = r18 * r20;
    r9 = r9 * r46;
    r34 = r37 * r38;
    r34 = fma(r27, r34, r51 * r9);
    WriteSum2<double, double>((double*)inout_shared, r49, r34);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            34 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r2 * r2;
    r34 = r34 * r18;
    r34 = r34 * r41;
    r49 = r18 * r46;
    r49 = r49 * r40;
    r9 = r40 * r55;
    r9 = r9 * r37;
    r9 = fma(r38, r9, r51 * r49);
    WriteSum2<double, double>((double*)inout_shared, r9, r34);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            36 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r41 * r25;
    r30 = r41 * r30;
    r34 = r36 * r55;
    r34 = fma(r30, r34, r47 * r30);
    WriteSum2<double, double>((double*)inout_shared, r25, r34);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            38 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r46 * r40;
    r34 = r34 * r51;
    r25 = r46 * r40;
    r25 = r25 * r54;
    WriteSum2<double, double>((double*)inout_shared, r34, r25);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            40 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = r46 * r51;
    r51 = r51 * r44;
    r25 = r46 * r54;
    r25 = r25 * r44;
    WriteSum2<double, double>((double*)inout_shared, r51, r25);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            42 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
}

void ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointResJac(
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    double* const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    double* const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointFixedPointResJacKernel<<<
      n_blocks,
      1024>>>(sensor_from_rig,
              sensor_from_rig_num_alloc,
              focal_and_extra,
              focal_and_extra_num_alloc,
              focal_and_extra_indices,
              pixel,
              pixel_num_alloc,
              pose,
              pose_num_alloc,
              principal_point,
              principal_point_num_alloc,
              point,
              point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_focal_and_extra_njtr,
              out_focal_and_extra_njtr_num_alloc,
              out_focal_and_extra_precond_diag,
              out_focal_and_extra_precond_diag_num_alloc,
              out_focal_and_extra_precond_tril,
              out_focal_and_extra_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar