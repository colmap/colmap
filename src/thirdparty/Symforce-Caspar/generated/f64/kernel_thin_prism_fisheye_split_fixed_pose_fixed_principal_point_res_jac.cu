#include "kernel_thin_prism_fisheye_split_fixed_pose_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointResJacKernel(
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97, r98, r99;

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
  };
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
    r7 = fma(r8, r24, r7);
    ReadIdx2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r18, r26);
    r27 = r15 * r16;
    r27 = r27 * r10;
    r28 = r11 * r12;
    r28 = fma(r10, r28, r27);
    r29 = -2.00000000000000000e+00;
    r30 = r15 * r15;
    r30 = r29 * r30;
    r31 = 1.00000000000000000e+00;
    r32 = r11 * r11;
    r32 = fma(r29, r32, r31);
    r33 = r30 + r32;
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r34);
    r35 = r16 * r11;
    r35 = r35 * r10;
    r36 = r15 * r12;
    r36 = fma(r29, r36, r35);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r37);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r38 = r10 * r23;
    r38 = r38 * r22;
    r39 = r19 * r25;
    r40 = fma(r29, r39, r38);
    r41 = r23 * r23;
    r41 = r29 * r41;
    r42 = r31 + r41;
    r43 = r19 * r19;
    r43 = r43 * r29;
    r42 = r42 + r43;
    r7 = fma(r18, r28, r7);
    r7 = fma(r26, r33, r7);
    r7 = fma(r34, r36, r7);
    r7 = fma(r37, r40, r7);
    r7 = fma(r9, r42, r7);
    r36 = r22 * r22;
    r36 = r29 * r36;
    r33 = r31 + r36;
    r33 = r33 + r41;
    r6 = fma(r8, r33, r6);
    r41 = r23 * r29;
    r41 = fma(r25, r41, r20);
    r20 = r10 * r23;
    r20 = r20 * r19;
    r19 = r10 * r22;
    r19 = fma(r25, r19, r20);
    r28 = r15 * r11;
    r28 = r28 * r10;
    r44 = r16 * r12;
    r45 = fma(r10, r44, r28);
    r46 = r11 * r12;
    r46 = fma(r29, r46, r27);
    r27 = r16 * r16;
    r27 = r27 * r29;
    r32 = r27 + r32;
    r6 = fma(r9, r41, r6);
    r6 = fma(r37, r19, r6);
    r6 = fma(r34, r45, r6);
    r6 = fma(r26, r46, r6);
    r6 = fma(r18, r32, r6);
    r32 = r6 * r6;
    r46 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r45);
    r47 = r29 * r22;
    r47 = fma(r25, r47, r20);
    r8 = fma(r8, r47, r45);
    r44 = fma(r29, r44, r28);
    r27 = r31 + r27;
    r27 = r27 + r30;
    r30 = r15 * r12;
    r30 = fma(r10, r30, r35);
    r39 = fma(r10, r39, r38);
    r36 = r31 + r36;
    r36 = r36 + r43;
    r8 = fma(r18, r44, r8);
    r8 = fma(r34, r27, r8);
    r8 = fma(r26, r30, r8);
    r8 = fma(r9, r39, r8);
    r8 = fma(r37, r36, r8);
    r37 = copysign(1.0, r8);
    r37 = fma(r46, r37, r8);
    r8 = r37 * r37;
    r9 = 1.0 / r8;
    r30 = r7 * r7;
    r30 = fma(r9, r30, r9 * r32);
    r32 = sqrt(r30);
    r26 = atan(r32);
    r27 = r7 * r26;
    r34 = copysign(1.0, r32);
    r34 = fma(r46, r34, r32);
    r46 = r34 * r34;
    r32 = 1.0 / r46;
    r44 = r9 * r32;
    r18 = r7 * r26;
    r27 = r27 * r44;
    r27 = r27 * r18;
    r43 = 3.00000000000000000e+00;
    r38 = r6 * r6;
    r38 = r9 * r38;
    r35 = r26 * r26;
    r38 = r38 * r32;
    r38 = r38 * r35;
    r35 = fma(r43, r38, r27);
  };
  LoadShared<2, double, double>(focal_and_extra,
                                8 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target,
                        r28,
                        r45);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r27 = r38 + r27;
    r20 = fma(r28, r27, r5 * r35);
    r25 = r4 * r10;
    r48 = r6 * r26;
    r49 = r48 * r44;
    r25 = r25 * r18;
    r20 = fma(r49, r25, r20);
  };
  LoadShared<2, double, double>(focal_and_extra,
                                2 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target,
                        r50,
                        r51);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r52 = r27 * r27;
    r53 = fma(r51, r52, r50 * r27);
  };
  LoadShared<2, double, double>(focal_and_extra,
                                6 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target,
                        r54,
                        r55);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r56 = r52 * r52;
    r57 = r27 * r52;
    r53 = fma(r55, r56, r53);
    r53 = fma(r54, r57, r53);
    r58 = 1.0 / r37;
    r59 = 1.0 / r34;
    r60 = r58 * r59;
    r61 = r53 * r60;
    r20 = fma(r48, r61, r20);
    r20 = fma(r48, r60, r20);
    r0 = fma(r2, r20, r0);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r25, r62);
    r0 = fma(r25, r21, r0);
    r25 = r7 * r26;
    r25 = r25 * r43;
    r25 = r25 * r44;
    r25 = fma(r18, r25, r38);
    r63 = fma(r45, r27, r4 * r25);
    r64 = r5 * r10;
    r64 = r64 * r18;
    r63 = fma(r49, r64, r63);
    r63 = fma(r18, r61, r63);
    r63 = fma(r60, r18, r63);
    r1 = fma(r3, r63, r1);
    r1 = fma(r62, r21, r1);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r0, r1);
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        0 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r20,
        r63);
    r62 = r2 * r27;
    r62 = r62 * r48;
    r62 = r62 * r60;
    r64 = r27 * r60;
    r65 = r3 * r18;
    r64 = r64 * r65;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        2 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r62,
        r64);
    r64 = r2 * r48;
    r64 = r64 * r60;
    r64 = r64 * r52;
    r62 = r60 * r52;
    r62 = r62 * r65;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        4 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r64,
        r62);
    r62 = r3 * r25;
    r64 = r2 * r10;
    r64 = r64 * r18;
    r64 = r64 * r49;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        6 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r64,
        r62);
    r62 = r2 * r35;
    r64 = r10 * r49;
    r64 = r64 * r65;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        8 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r62,
        r64);
    r64 = r2 * r48;
    r64 = r64 * r60;
    r64 = r64 * r57;
    r62 = r60 * r57;
    r62 = r62 * r65;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        10 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r64,
        r62);
    r62 = r2 * r48;
    r62 = r62 * r60;
    r62 = r62 * r56;
    r64 = r60 * r65;
    r64 = r64 * r56;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        12 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r62,
        r64);
    r64 = r2 * r27;
    r62 = r3 * r27;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        14 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r64,
        r62);
    r62 = r21 * r63;
    r62 = r62 * r1;
    r64 = r21 * r0;
    r66 = r20 * r64;
    WriteSum2<double, double>((double*)inout_shared, r66, r62);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            0 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = r21 * r27;
    r62 = r62 * r1;
    r62 = r62 * r60;
    r66 = r27 * r48;
    r64 = r2 * r64;
    r66 = r66 * r60;
    r66 = fma(r64, r66, r65 * r62);
    r62 = r21 * r1;
    r62 = r62 * r60;
    r62 = r62 * r52;
    r67 = r48 * r60;
    r67 = r67 * r52;
    r67 = fma(r64, r67, r65 * r62);
    WriteSum2<double, double>((double*)inout_shared, r66, r67);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            2 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = r3 * r21;
    r67 = r67 * r25;
    r66 = r2 * r29;
    r66 = r66 * r0;
    r66 = r66 * r18;
    r66 = fma(r49, r66, r1 * r67);
    r67 = r29 * r1;
    r67 = r67 * r49;
    r67 = fma(r65, r67, r35 * r64);
    WriteSum2<double, double>((double*)inout_shared, r66, r67);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            4 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = r21 * r1;
    r67 = r67 * r60;
    r67 = r67 * r57;
    r66 = r48 * r60;
    r66 = r66 * r57;
    r66 = fma(r64, r66, r65 * r67);
    r67 = r21 * r1;
    r67 = r67 * r60;
    r67 = r67 * r65;
    r0 = r48 * r60;
    r0 = r0 * r56;
    r0 = fma(r64, r0, r56 * r67);
    WriteSum2<double, double>((double*)inout_shared, r66, r0);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            6 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r3 * r21;
    r0 = r0 * r27;
    r0 = r0 * r1;
    r66 = r27 * r64;
    WriteSum2<double, double>((double*)inout_shared, r66, r0);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            8 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r20 * r20;
    r66 = r63 * r63;
    WriteSum2<double, double>((double*)inout_shared, r0, r66);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            0 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = r2 * r2;
    r38 = r66 * r38;
    r0 = r7 * r26;
    r67 = r3 * r65;
    r0 = r0 * r44;
    r0 = r0 * r52;
    r0 = fma(r67, r0, r52 * r38);
    r62 = r7 * r26;
    r62 = r62 * r44;
    r62 = r62 * r56;
    r62 = fma(r67, r62, r56 * r38);
    WriteSum2<double, double>((double*)inout_shared, r0, r62);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            2 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r48 * r66;
    r68 = r18 * r0;
    r69 = r6 * r7;
    r70 = 4.00000000000000000e+00;
    r8 = r37 * r8;
    r37 = r37 * r8;
    r37 = 1.0 / r37;
    r46 = r34 * r46;
    r34 = r34 * r46;
    r34 = 1.0 / r34;
    r69 = r69 * r26;
    r69 = r69 * r26;
    r69 = r69 * r70;
    r69 = r69 * r37;
    r69 = r69 * r34;
    r34 = r3 * r3;
    r34 = r34 * r25;
    r34 = fma(r25, r34, r69 * r68);
    r68 = r48 * r67;
    r37 = r35 * r35;
    r37 = fma(r66, r37, r69 * r68);
    WriteSum2<double, double>((double*)inout_shared, r34, r37);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            4 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = r57 * r57;
    r34 = r7 * r26;
    r34 = r34 * r44;
    r34 = r34 * r67;
    r34 = fma(r37, r34, r37 * r38);
    r68 = r56 * r56;
    r69 = r7 * r26;
    r69 = r69 * r44;
    r69 = r69 * r67;
    r69 = fma(r68, r69, r38 * r68);
    WriteSum2<double, double>((double*)inout_shared, r34, r69);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            6 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = r52 * r66;
    r68 = r3 * r3;
    r68 = r68 * r52;
    WriteSum2<double, double>((double*)inout_shared, r69, r68);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            8 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = 0.00000000000000000e+00;
    r69 = r2 * r27;
    r69 = r69 * r20;
    r69 = r69 * r48;
    r69 = r69 * r60;
    WriteSum2<double, double>((double*)inout_shared, r68, r69);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            0 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = r2 * r20;
    r69 = r69 * r48;
    r69 = r69 * r60;
    r69 = r69 * r52;
    r71 = r2 * r10;
    r71 = r71 * r20;
    r71 = r71 * r18;
    r71 = r71 * r49;
    WriteSum2<double, double>((double*)inout_shared, r69, r71);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            2 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = r2 * r35;
    r71 = r71 * r20;
    r69 = r2 * r20;
    r69 = r69 * r48;
    r69 = r69 * r60;
    r69 = r69 * r57;
    WriteSum2<double, double>((double*)inout_shared, r71, r69);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            4 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = r2 * r27;
    r69 = r69 * r20;
    r20 = r2 * r20;
    r20 = r20 * r48;
    r20 = r20 * r60;
    r20 = r20 * r56;
    WriteSum2<double, double>((double*)inout_shared, r20, r69);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            6 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = r27 * r63;
    r69 = r69 * r60;
    r69 = r69 * r65;
    WriteSum2<double, double>((double*)inout_shared, r68, r69);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            8 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = r3 * r25;
    r69 = r69 * r63;
    r20 = r63 * r60;
    r20 = r20 * r52;
    r20 = r20 * r65;
    WriteSum2<double, double>((double*)inout_shared, r20, r69);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            10 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = r10 * r63;
    r69 = r69 * r49;
    r69 = r69 * r65;
    r20 = r63 * r60;
    r20 = r20 * r57;
    r20 = r20 * r65;
    WriteSum2<double, double>((double*)inout_shared, r69, r20);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            12 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r63 * r60;
    r20 = r20 * r65;
    r20 = r20 * r56;
    WriteSum2<double, double>((double*)inout_shared, r20, r68);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            14 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = r3 * r27;
    r68 = r68 * r63;
    r63 = r7 * r26;
    r63 = r63 * r44;
    r63 = r63 * r57;
    r63 = fma(r67, r63, r57 * r38);
    WriteSum2<double, double>((double*)inout_shared, r68, r63);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            16 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = r25 * r67;
    r68 = r60 * r63;
    r20 = r6 * r27;
    r8 = 1.0 / r8;
    r46 = 1.0 / r46;
    r65 = r10 * r26;
    r20 = r20 * r8;
    r20 = r20 * r46;
    r20 = r20 * r18;
    r20 = r20 * r65;
    r20 = fma(r0, r20, r27 * r68);
    r69 = r60 * r0;
    r71 = r35 * r69;
    r72 = r7 * r27;
    r72 = r72 * r8;
    r72 = r72 * r46;
    r72 = r72 * r48;
    r72 = r72 * r65;
    r72 = fma(r67, r72, r27 * r71);
    WriteSum2<double, double>((double*)inout_shared, r20, r72);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            18 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r27 * r56;
    r20 = r7 * r26;
    r20 = r20 * r44;
    r20 = r20 * r67;
    r20 = fma(r72, r20, r72 * r38);
    WriteSum2<double, double>((double*)inout_shared, r62, r20);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            20 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = r52 * r69;
    r73 = r60 * r52;
    r73 = r73 * r67;
    WriteSum2<double, double>((double*)inout_shared, r62, r73);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            22 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = r6 * r8;
    r73 = r73 * r46;
    r73 = r73 * r18;
    r73 = r73 * r52;
    r73 = r73 * r65;
    r73 = fma(r0, r73, r52 * r68);
    r62 = r7 * r8;
    r62 = r62 * r46;
    r62 = r62 * r48;
    r62 = r62 * r52;
    r62 = r62 * r65;
    r62 = fma(r67, r62, r52 * r71);
    WriteSum2<double, double>((double*)inout_shared, r73, r62);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            24 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r20, r34);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            26 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r57 * r69;
    r20 = r60 * r57;
    r20 = r20 * r67;
    WriteSum2<double, double>((double*)inout_shared, r34, r20);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            28 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r10 * r35;
    r20 = r20 * r18;
    r20 = r20 * r49;
    r34 = r10 * r49;
    r34 = fma(r63, r34, r66 * r20);
    r20 = r6 * r8;
    r20 = r20 * r46;
    r20 = r20 * r18;
    r20 = r20 * r57;
    r20 = r20 * r65;
    r20 = fma(r0, r20, r57 * r68);
    WriteSum2<double, double>((double*)inout_shared, r34, r20);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            30 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r10 * r27;
    r20 = r20 * r18;
    r20 = r20 * r49;
    r20 = r20 * r66;
    r34 = r6 * r8;
    r34 = r34 * r46;
    r34 = r34 * r18;
    r34 = r34 * r56;
    r34 = r34 * r65;
    r34 = fma(r0, r34, r56 * r68);
    WriteSum2<double, double>((double*)inout_shared, r34, r20);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            32 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r3 * r3;
    r20 = r20 * r27;
    r20 = r20 * r25;
    r25 = r7 * r8;
    r25 = r25 * r46;
    r25 = r25 * r48;
    r25 = r25 * r57;
    r25 = r25 * r65;
    r25 = fma(r67, r25, r57 * r71);
    WriteSum2<double, double>((double*)inout_shared, r20, r25);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            34 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r35 * r27;
    r25 = r25 * r66;
    r20 = r7 * r8;
    r20 = r20 * r46;
    r20 = r20 * r48;
    r20 = r20 * r56;
    r20 = r20 * r65;
    r20 = fma(r67, r20, r56 * r71);
    WriteSum2<double, double>((double*)inout_shared, r20, r25);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            36 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r10 * r27;
    r25 = r25 * r49;
    r25 = r25 * r67;
    r37 = r27 * r37;
    r20 = r7 * r26;
    r20 = r20 * r44;
    r20 = r20 * r67;
    r20 = fma(r37, r20, r38 * r37);
    WriteSum2<double, double>((double*)inout_shared, r25, r20);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            38 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r56 * r69;
    r56 = r60 * r56;
    r56 = r56 * r67;
    WriteSum2<double, double>((double*)inout_shared, r20, r56);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            40 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = r72 * r69;
    r56 = r60 * r67;
    r56 = r56 * r72;
    WriteSum2<double, double>((double*)inout_shared, r69, r56);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            42 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = r26 * r32;
    r69 = r7 * r7;
    r72 = r29 * r8;
    r69 = r69 * r72;
    r20 = r26 * r69;
    r56 = r56 * r20;
    r25 = r21 * r7;
    r37 = r10 * r33;
    r37 = r37 * r6;
    r37 = fma(r47, r69, r9 * r37);
    r38 = r47 * r6;
    r38 = r38 * r6;
    r37 = fma(r72, r38, r37);
    r71 = r10 * r24;
    r71 = r71 * r7;
    r37 = fma(r9, r71, r37);
    r71 = rsqrt(r30);
    r25 = r25 * r7;
    r25 = r25 * r26;
    r25 = r25 * r26;
    r25 = r25 * r37;
    r25 = r25 * r9;
    r25 = r25 * r46;
    r25 = fma(r71, r25, r47 * r56);
    r30 = r31 + r30;
    r30 = 1.0 / r30;
    r30 = r30 * r71;
    r31 = r7 * r30;
    r38 = r37 * r31;
    r34 = r44 * r18;
    r25 = fma(r34, r38, r25);
    r68 = r65 * r34;
    r25 = fma(r24, r68, r25);
    r38 = r47 * r6;
    r38 = r38 * r26;
    r38 = r38 * r32;
    r38 = r38 * r48;
    r63 = r37 * r30;
    r62 = r6 * r49;
    r63 = fma(r62, r63, r72 * r38);
    r38 = r33 * r49;
    r63 = fma(r65, r38, r63);
    r73 = r21 * r6;
    r73 = r73 * r26;
    r73 = r73 * r37;
    r73 = r73 * r9;
    r73 = r73 * r46;
    r73 = r73 * r71;
    r63 = fma(r48, r73, r63);
    r73 = r25 + r63;
    r38 = r47 * r26;
    r74 = -6.00000000000000000e+00;
    r38 = r38 * r74;
    r38 = r38 * r32;
    r38 = r38 * r8;
    r75 = r6 * r48;
    r76 = r43 * r37;
    r76 = r76 * r30;
    r76 = fma(r62, r76, r38 * r75);
    r77 = r33 * r26;
    r78 = 6.00000000000000000e+00;
    r77 = r77 * r78;
    r76 = fma(r49, r77, r76);
    r79 = r6 * r37;
    r80 = -3.00000000000000000e+00;
    r80 = r26 * r80;
    r80 = r80 * r9;
    r80 = r80 * r46;
    r80 = r80 * r71;
    r79 = r79 * r48;
    r76 = fma(r80, r79, r76);
    r76 = r76 + r25;
    r76 = fma(r5, r76, r28 * r73);
    r25 = r6 * r30;
    r79 = 5.00000000000000000e-01;
    r77 = r79 * r61;
    r25 = r25 * r77;
    r81 = r21 * r47;
    r81 = r81 * r53;
    r81 = r81 * r9;
    r81 = r81 * r59;
    r76 = fma(r48, r81, r76);
    r82 = r4 * r37;
    r83 = r10 * r49;
    r83 = r83 * r31;
    r76 = fma(r83, r82, r76);
    r84 = r4 * r29;
    r84 = r84 * r37;
    r84 = r84 * r9;
    r84 = r84 * r46;
    r84 = r84 * r71;
    r84 = r84 * r48;
    r76 = fma(r18, r84, r76);
    r85 = r6 * r79;
    r85 = r85 * r37;
    r85 = r85 * r60;
    r76 = fma(r30, r85, r76);
    r86 = r4 * r68;
    r87 = r21 * r47;
    r87 = r87 * r9;
    r87 = r87 * r59;
    r76 = fma(r48, r87, r76);
    r88 = -5.00000000000000000e-01;
    r89 = r88 * r37;
    r89 = r89 * r32;
    r89 = r89 * r58;
    r89 = r89 * r71;
    r90 = r53 * r89;
    r91 = r51 * r10;
    r91 = r91 * r27;
    r91 = fma(r50, r73, r73 * r91);
    r70 = r55 * r70;
    r70 = r70 * r57;
    r54 = r54 * r43;
    r54 = r54 * r52;
    r91 = fma(r73, r70, r91);
    r91 = fma(r73, r54, r91);
    r52 = r91 * r48;
    r76 = fma(r60, r52, r76);
    r57 = r33 * r26;
    r76 = fma(r60, r57, r76);
    r55 = r4 * r47;
    r92 = -4.00000000000000000e+00;
    r92 = r92 * r32;
    r92 = r92 * r8;
    r92 = r92 * r48;
    r92 = r92 * r18;
    r76 = fma(r92, r55, r76);
    r93 = r4 * r24;
    r93 = r93 * r49;
    r76 = fma(r65, r93, r76);
    r94 = r33 * r26;
    r76 = fma(r61, r94, r76);
    r76 = fma(r37, r25, r76);
    r76 = fma(r33, r86, r76);
    r76 = fma(r48, r90, r76);
    r76 = fma(r48, r89, r76);
    r94 = r2 * r76;
    r93 = r7 * r38;
    r55 = r37 * r80;
    r55 = fma(r20, r55, r18 * r93);
    r93 = r43 * r37;
    r93 = r93 * r31;
    r55 = fma(r34, r93, r55);
    r57 = r24 * r26;
    r57 = r57 * r78;
    r57 = r57 * r44;
    r55 = fma(r18, r57, r55);
    r55 = r55 + r63;
    r55 = fma(r4, r55, r45 * r73);
    r73 = r91 * r60;
    r55 = fma(r18, r73, r55);
    r63 = r79 * r37;
    r63 = r63 * r60;
    r55 = fma(r31, r63, r55);
    r57 = r24 * r26;
    r55 = fma(r61, r57, r55);
    r93 = r21 * r47;
    r93 = r93 * r7;
    r93 = r93 * r26;
    r93 = r93 * r9;
    r55 = fma(r59, r93, r55);
    r52 = r5 * r37;
    r55 = fma(r83, r52, r55);
    r87 = r5 * r29;
    r87 = r87 * r37;
    r87 = r87 * r9;
    r87 = r87 * r46;
    r87 = r87 * r71;
    r87 = r87 * r48;
    r55 = fma(r18, r87, r55);
    r85 = r5 * r33;
    r55 = fma(r68, r85, r55);
    r84 = r7 * r26;
    r55 = fma(r89, r84, r55);
    r89 = r37 * r31;
    r55 = fma(r77, r89, r55);
    r82 = r24 * r26;
    r55 = fma(r60, r82, r55);
    r81 = r21 * r47;
    r81 = r81 * r7;
    r81 = r81 * r26;
    r81 = r81 * r53;
    r81 = r81 * r9;
    r55 = fma(r59, r81, r55);
    r95 = r5 * r92;
    r96 = r5 * r24;
    r96 = r96 * r49;
    r55 = fma(r65, r96, r55);
    r55 = fma(r90, r18, r55);
    r55 = fma(r47, r95, r55);
    r96 = r3 * r55;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r94,
                                             r96);
    r96 = r41 * r26;
    r96 = r96 * r78;
    r94 = r39 * r6;
    r94 = r94 * r26;
    r94 = r94 * r74;
    r94 = r94 * r32;
    r94 = r94 * r8;
    r94 = fma(r48, r94, r49 * r96);
    r96 = r39 * r6;
    r96 = r96 * r6;
    r81 = r10 * r41;
    r81 = r81 * r6;
    r81 = fma(r9, r81, r72 * r96);
    r96 = r10 * r42;
    r96 = r96 * r7;
    r81 = fma(r9, r96, r81);
    r81 = fma(r39, r69, r81);
    r96 = r81 * r80;
    r90 = r43 * r81;
    r90 = r90 * r30;
    r94 = fma(r62, r90, r94);
    r82 = r81 * r31;
    r82 = fma(r34, r82, r39 * r56);
    r89 = r21 * r7;
    r89 = r89 * r7;
    r89 = r89 * r26;
    r89 = r89 * r26;
    r89 = r89 * r81;
    r89 = r89 * r9;
    r89 = r89 * r46;
    r82 = fma(r71, r89, r82);
    r82 = fma(r42, r68, r82);
    r94 = fma(r96, r75, r94);
    r94 = r94 + r82;
    r90 = r41 * r49;
    r75 = r39 * r6;
    r75 = r75 * r26;
    r75 = r75 * r32;
    r75 = r75 * r48;
    r75 = fma(r72, r75, r65 * r90);
    r90 = r21 * r6;
    r90 = r90 * r26;
    r90 = r90 * r81;
    r90 = r90 * r9;
    r90 = r90 * r46;
    r90 = r90 * r71;
    r75 = fma(r48, r90, r75);
    r89 = r81 * r30;
    r75 = fma(r62, r89, r75);
    r82 = r82 + r75;
    r94 = fma(r28, r82, r5 * r94);
    r89 = r21 * r39;
    r89 = r89 * r53;
    r89 = r89 * r9;
    r89 = r89 * r59;
    r94 = fma(r48, r89, r94);
    r90 = r41 * r26;
    r94 = fma(r61, r90, r94);
    r84 = r6 * r79;
    r84 = r84 * r81;
    r84 = r84 * r60;
    r94 = fma(r30, r84, r94);
    r85 = r51 * r10;
    r85 = r85 * r27;
    r85 = fma(r82, r85, r50 * r82);
    r85 = fma(r82, r54, r85);
    r85 = fma(r82, r70, r85);
    r87 = r85 * r48;
    r94 = fma(r60, r87, r94);
    r52 = r21 * r39;
    r52 = r52 * r9;
    r52 = r52 * r59;
    r94 = fma(r48, r52, r94);
    r93 = r4 * r42;
    r93 = r93 * r49;
    r94 = fma(r65, r93, r94);
    r57 = r41 * r26;
    r94 = fma(r60, r57, r94);
    r63 = r88 * r81;
    r63 = r63 * r32;
    r63 = r63 * r58;
    r63 = r63 * r71;
    r94 = fma(r48, r63, r94);
    r73 = r4 * r81;
    r94 = fma(r83, r73, r94);
    r97 = r4 * r29;
    r97 = r97 * r81;
    r97 = r97 * r9;
    r97 = r97 * r46;
    r97 = r97 * r71;
    r97 = r97 * r48;
    r94 = fma(r18, r97, r94);
    r98 = r4 * r39;
    r94 = fma(r92, r98, r94);
    r99 = r53 * r88;
    r99 = r99 * r81;
    r99 = r99 * r32;
    r99 = r99 * r58;
    r99 = r99 * r71;
    r94 = fma(r48, r99, r94);
    r94 = fma(r81, r25, r94);
    r94 = fma(r41, r86, r94);
    r99 = r2 * r94;
    r98 = r39 * r7;
    r98 = r98 * r7;
    r98 = r98 * r26;
    r98 = r98 * r26;
    r98 = r98 * r74;
    r98 = r98 * r32;
    r97 = r43 * r81;
    r97 = r97 * r31;
    r97 = fma(r34, r97, r8 * r98);
    r98 = r42 * r26;
    r98 = r98 * r78;
    r98 = r98 * r44;
    r97 = fma(r18, r98, r97);
    r97 = fma(r20, r96, r97);
    r97 = r97 + r75;
    r82 = fma(r45, r82, r4 * r97);
    r97 = r21 * r39;
    r97 = r97 * r7;
    r97 = r97 * r26;
    r97 = r97 * r9;
    r82 = fma(r59, r97, r82);
    r75 = r21 * r39;
    r75 = r75 * r7;
    r75 = r75 * r26;
    r75 = r75 * r53;
    r75 = r75 * r9;
    r82 = fma(r59, r75, r82);
    r96 = r79 * r81;
    r96 = r96 * r60;
    r82 = fma(r31, r96, r82);
    r98 = r42 * r26;
    r82 = fma(r61, r98, r82);
    r73 = r5 * r42;
    r73 = r73 * r49;
    r82 = fma(r65, r73, r82);
    r63 = r7 * r26;
    r63 = r63 * r88;
    r63 = r63 * r81;
    r63 = r63 * r32;
    r63 = r63 * r58;
    r82 = fma(r71, r63, r82);
    r57 = r5 * r41;
    r82 = fma(r68, r57, r82);
    r93 = r5 * r81;
    r82 = fma(r83, r93, r82);
    r52 = r5 * r29;
    r52 = r52 * r81;
    r52 = r52 * r9;
    r52 = r52 * r46;
    r52 = r52 * r71;
    r52 = r52 * r48;
    r82 = fma(r18, r52, r82);
    r87 = r42 * r26;
    r82 = fma(r60, r87, r82);
    r84 = r81 * r31;
    r82 = fma(r77, r84, r82);
    r90 = r85 * r60;
    r82 = fma(r18, r90, r82);
    r89 = r7 * r26;
    r89 = r89 * r53;
    r89 = r89 * r88;
    r89 = r89 * r81;
    r89 = r89 * r32;
    r89 = r89 * r58;
    r82 = fma(r71, r89, r82);
    r82 = fma(r39, r95, r82);
    r89 = r3 * r82;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r99,
                                             r89);
    r89 = r36 * r6;
    r89 = r89 * r26;
    r89 = r89 * r74;
    r89 = r89 * r32;
    r89 = r89 * r8;
    r99 = r10 * r19;
    r99 = r99 * r6;
    r69 = fma(r36, r69, r9 * r99);
    r99 = r36 * r6;
    r99 = r99 * r6;
    r69 = fma(r72, r99, r69);
    r90 = r10 * r40;
    r90 = r90 * r7;
    r69 = fma(r9, r90, r69);
    r90 = r6 * r69;
    r90 = r90 * r48;
    r90 = fma(r80, r90, r48 * r89);
    r89 = r43 * r69;
    r89 = r89 * r30;
    r90 = fma(r62, r89, r90);
    r99 = r19 * r26;
    r99 = r99 * r78;
    r90 = fma(r49, r99, r90);
    r84 = r21 * r7;
    r84 = r84 * r7;
    r84 = r84 * r26;
    r84 = r84 * r26;
    r84 = r84 * r69;
    r84 = r84 * r9;
    r84 = r84 * r46;
    r84 = fma(r71, r84, r40 * r68);
    r87 = r69 * r31;
    r84 = fma(r34, r87, r84);
    r84 = fma(r36, r56, r84);
    r90 = r90 + r84;
    r99 = r36 * r6;
    r99 = r99 * r26;
    r99 = r99 * r32;
    r99 = r99 * r48;
    r89 = r21 * r6;
    r89 = r89 * r26;
    r89 = r89 * r69;
    r89 = r89 * r9;
    r89 = r89 * r46;
    r89 = r89 * r71;
    r89 = fma(r48, r89, r72 * r99);
    r99 = r69 * r30;
    r89 = fma(r62, r99, r89);
    r62 = r19 * r49;
    r89 = fma(r65, r62, r89);
    r84 = r84 + r89;
    r28 = fma(r28, r84, r5 * r90);
    r90 = r19 * r26;
    r28 = fma(r61, r90, r28);
    r62 = r21 * r36;
    r62 = r62 * r9;
    r62 = r62 * r59;
    r28 = fma(r48, r62, r28);
    r99 = r4 * r40;
    r99 = r99 * r49;
    r28 = fma(r65, r99, r28);
    r72 = r4 * r29;
    r72 = r72 * r69;
    r72 = r72 * r9;
    r72 = r72 * r46;
    r72 = r72 * r71;
    r72 = r72 * r48;
    r28 = fma(r18, r72, r28);
    r56 = r88 * r69;
    r56 = r56 * r32;
    r56 = r56 * r58;
    r56 = r56 * r71;
    r28 = fma(r48, r56, r28);
    r87 = r6 * r79;
    r87 = r87 * r69;
    r87 = r87 * r60;
    r28 = fma(r30, r87, r28);
    r83 = r69 * r83;
    r52 = r4 * r36;
    r28 = fma(r92, r52, r28);
    r92 = r19 * r26;
    r28 = fma(r60, r92, r28);
    r93 = r53 * r88;
    r93 = r93 * r69;
    r93 = r93 * r32;
    r93 = r93 * r58;
    r93 = r93 * r71;
    r28 = fma(r48, r93, r28);
    r57 = r21 * r36;
    r57 = r57 * r53;
    r57 = r57 * r9;
    r57 = r57 * r59;
    r28 = fma(r48, r57, r28);
    r63 = r51 * r10;
    r63 = r63 * r27;
    r63 = fma(r84, r63, r50 * r84);
    r63 = fma(r84, r70, r63);
    r63 = fma(r84, r54, r63);
    r54 = r63 * r48;
    r28 = fma(r60, r54, r28);
    r28 = fma(r69, r25, r28);
    r28 = fma(r4, r83, r28);
    r28 = fma(r19, r86, r28);
    r54 = r2 * r28;
    r57 = r40 * r26;
    r57 = r57 * r78;
    r57 = r57 * r44;
    r44 = r69 * r80;
    r44 = fma(r20, r44, r18 * r57);
    r57 = r43 * r69;
    r57 = r57 * r31;
    r44 = fma(r34, r57, r44);
    r34 = r36 * r7;
    r34 = r34 * r7;
    r34 = r34 * r26;
    r34 = r34 * r26;
    r34 = r34 * r74;
    r34 = r34 * r32;
    r44 = fma(r8, r34, r44);
    r44 = r44 + r89;
    r44 = fma(r4, r44, r45 * r84);
    r84 = r5 * r40;
    r84 = r84 * r49;
    r44 = fma(r65, r84, r44);
    r65 = r5 * r29;
    r65 = r65 * r69;
    r65 = r65 * r9;
    r65 = r65 * r46;
    r65 = r65 * r71;
    r65 = r65 * r48;
    r44 = fma(r18, r65, r44);
    r46 = r7 * r26;
    r46 = r46 * r88;
    r46 = r46 * r69;
    r46 = r46 * r32;
    r46 = r46 * r58;
    r44 = fma(r71, r46, r44);
    r45 = r7 * r26;
    r45 = r45 * r53;
    r45 = r45 * r88;
    r45 = r45 * r69;
    r45 = r45 * r32;
    r45 = r45 * r58;
    r44 = fma(r71, r45, r44);
    r71 = r69 * r31;
    r44 = fma(r77, r71, r44);
    r77 = r21 * r36;
    r77 = r77 * r7;
    r77 = r77 * r26;
    r77 = r77 * r53;
    r77 = r77 * r9;
    r44 = fma(r59, r77, r44);
    r58 = r40 * r26;
    r44 = fma(r61, r58, r44);
    r61 = r5 * r19;
    r44 = fma(r68, r61, r44);
    r68 = r40 * r26;
    r44 = fma(r60, r68, r44);
    r32 = r63 * r60;
    r44 = fma(r18, r32, r44);
    r89 = r79 * r69;
    r89 = r89 * r60;
    r44 = fma(r31, r89, r44);
    r34 = r21 * r36;
    r34 = r34 * r7;
    r34 = r34 * r26;
    r34 = r34 * r9;
    r44 = fma(r59, r34, r44);
    r44 = fma(r5, r83, r44);
    r44 = fma(r36, r95, r44);
    r34 = r3 * r44;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r54,
                                             r34);
    r34 = r3 * r21;
    r34 = r34 * r1;
    r34 = fma(r76, r64, r55 * r34);
    r54 = r3 * r21;
    r54 = r54 * r1;
    r54 = fma(r94, r64, r82 * r54);
    WriteSum2<double, double>((double*)inout_shared, r34, r54);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = r3 * r21;
    r54 = r54 * r1;
    r64 = fma(r28, r64, r44 * r54);
    WriteSum1<double, double>((double*)inout_shared, r64);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = r3 * r3;
    r54 = r55 * r55;
    r34 = r76 * r76;
    r34 = fma(r66, r34, r54 * r64);
    r54 = r82 * r82;
    r89 = r94 * r94;
    r89 = fma(r66, r89, r54 * r64);
    WriteSum2<double, double>((double*)inout_shared, r34, r89);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r89 = r44 * r44;
    r34 = r28 * r28;
    r34 = fma(r66, r34, r89 * r64);
    WriteSum1<double, double>((double*)inout_shared, r34);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r3 * r3;
    r34 = r34 * r55;
    r64 = r76 * r94;
    r64 = fma(r66, r64, r82 * r34);
    r34 = r3 * r3;
    r34 = r34 * r55;
    r55 = r76 * r28;
    r55 = fma(r66, r55, r44 * r34);
    WriteSum2<double, double>((double*)inout_shared, r64, r55);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = r3 * r3;
    r55 = r55 * r82;
    r82 = r94 * r28;
    r82 = fma(r66, r82, r44 * r55);
    WriteSum1<double, double>((double*)inout_shared, r82);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
}

void ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointResJac(
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
  ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointResJacKernel<<<n_blocks,
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