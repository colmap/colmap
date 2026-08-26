#include "kernel_thin_prism_fisheye_fixed_pose_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFixedPoseResJacKernel(
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
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102;
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
        pose, 2 * pose_num_alloc, global_thread_idx, r13, r14);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r15,
                                            r16);
    ReadIdx2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r17, r18);
    r19 = fma(r15, r18, r12 * r13);
    r20 = r16 * r17;
    r21 = -1.00000000000000000e+00;
    r19 = fma(r21, r20, r19);
    r19 = fma(r11, r14, r19);
    r20 = r10 * r19;
    r22 = fma(r15, r14, r12 * r17);
    r23 = r11 * r18;
    r22 = fma(r21, r23, r22);
    r22 = fma(r16, r13, r22);
    r20 = r20 * r22;
    r23 = -2.00000000000000000e+00;
    r24 = r15 * r13;
    r24 = fma(r21, r24, r12 * r18);
    r24 = fma(r16, r14, r24);
    r24 = fma(r11, r17, r24);
    r25 = r23 * r24;
    r26 = fma(r16, r18, r15 * r17);
    r26 = fma(r11, r13, r26);
    r26 = fma(r21, r26, r12 * r14);
    r25 = fma(r26, r25, r20);
    r7 = fma(r8, r25, r7);
    ReadIdx2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r14, r27);
    r28 = r15 * r11;
    r28 = r28 * r10;
    r29 = r16 * r12;
    r30 = fma(r23, r29, r28);
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r31);
    r32 = r15 * r15;
    r32 = r23 * r32;
    r33 = 1.00000000000000000e+00;
    r34 = r16 * r16;
    r34 = fma(r23, r34, r33);
    r35 = r32 + r34;
    r36 = r16 * r11;
    r36 = r36 * r10;
    r37 = r15 * r12;
    r37 = fma(r10, r37, r36);
    r38 = r10 * r19;
    r38 = r38 * r24;
    r39 = r22 * r26;
    r40 = fma(r10, r39, r38);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r41);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r42 = r24 * r24;
    r42 = r23 * r42;
    r43 = r33 + r42;
    r44 = r22 * r22;
    r44 = r44 * r23;
    r43 = r43 + r44;
    r7 = fma(r14, r30, r7);
    r7 = fma(r31, r35, r7);
    r7 = fma(r27, r37, r7);
    r7 = fma(r9, r40, r7);
    r7 = fma(r41, r43, r7);
    r37 = copysign(1.0, r7);
    r37 = fma(r6, r37, r7);
    r7 = r37 * r37;
    r35 = 1.0 / r7;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r30,
                                            r45);
    r22 = r10 * r22;
    r22 = r22 * r24;
    r46 = r10 * r19;
    r46 = fma(r26, r46, r22);
    r45 = fma(r8, r46, r45);
    r47 = r15 * r16;
    r47 = r47 * r10;
    r48 = r11 * r12;
    r48 = fma(r10, r48, r47);
    r49 = r11 * r11;
    r49 = r23 * r49;
    r50 = r33 + r49;
    r50 = r50 + r32;
    r32 = r15 * r12;
    r32 = fma(r23, r32, r36);
    r39 = fma(r23, r39, r38);
    r38 = r19 * r19;
    r38 = r23 * r38;
    r36 = r33 + r38;
    r36 = r36 + r44;
    r45 = fma(r14, r48, r45);
    r45 = fma(r27, r50, r45);
    r45 = fma(r31, r32, r45);
    r45 = fma(r41, r39, r45);
    r45 = fma(r9, r36, r45);
    r32 = r45 * r45;
    r32 = r35 * r32;
    r42 = r33 + r42;
    r42 = r42 + r38;
    r8 = fma(r8, r42, r30);
    r30 = r19 * r23;
    r30 = fma(r26, r30, r22);
    r22 = r10 * r24;
    r22 = fma(r26, r22, r20);
    r29 = fma(r10, r29, r28);
    r28 = r11 * r12;
    r28 = fma(r23, r28, r47);
    r34 = r49 + r34;
    r8 = fma(r9, r30, r8);
    r8 = fma(r41, r22, r8);
    r8 = fma(r31, r29, r8);
    r8 = fma(r27, r28, r8);
    r8 = fma(r14, r34, r8);
    r34 = r8 * r8;
    r14 = r45 * r45;
    r14 = fma(r35, r14, r35 * r34);
    r34 = sqrt(r14);
    r28 = copysign(1.0, r34);
    r28 = fma(r6, r28, r34);
    r6 = r28 * r28;
    r27 = 1.0 / r6;
    r34 = atan(r34);
    r29 = r34 * r34;
    r32 = r32 * r27;
    r32 = r32 * r29;
    r29 = r8 * r34;
    r31 = 3.00000000000000000e+00;
    r41 = r35 * r27;
    r9 = r8 * r34;
    r29 = r29 * r31;
    r29 = r29 * r41;
    r29 = fma(r9, r29, r32);
  };
  LoadShared<2, double, double>(
      calib, 10 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r49, r47);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r20 = r8 * r34;
    r20 = r20 * r41;
    r20 = r20 * r9;
    r26 = r32 + r20;
    r38 = fma(r49, r26, r5 * r29);
  };
  LoadShared<2, double, double>(
      calib, 4 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r50, r48);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r44 = r26 * r26;
    r51 = fma(r48, r44, r50 * r26);
  };
  LoadShared<2, double, double>(
      calib, 8 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r52, r53);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r54 = r44 * r44;
    r55 = r26 * r44;
    r51 = fma(r53, r54, r51);
    r51 = fma(r52, r55, r51);
    r56 = 1.0 / r37;
    r57 = 1.0 / r28;
    r58 = r56 * r57;
    r59 = r51 * r58;
    r60 = r4 * r10;
    r61 = r45 * r34;
    r62 = r61 * r41;
    r60 = r60 * r9;
    r38 = fma(r62, r60, r38);
    r38 = fma(r9, r59, r38);
    r38 = fma(r58, r9, r38);
    r0 = fma(r2, r38, r0);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r60, r63);
    r0 = fma(r60, r21, r0);
    r20 = fma(r31, r32, r20);
    r60 = fma(r47, r26, r4 * r20);
    r64 = r5 * r10;
    r64 = r64 * r9;
    r60 = fma(r62, r64, r60);
    r60 = fma(r61, r59, r60);
    r60 = fma(r61, r58, r60);
    r1 = fma(r3, r60, r1);
    r1 = fma(r63, r21, r1);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r0, r1);
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             0 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r38,
                                             r60);
    r63 = r26 * r58;
    r64 = r2 * r9;
    r63 = r63 * r64;
    r65 = r3 * r26;
    r65 = r65 * r61;
    r65 = r65 * r58;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             2 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r63,
                                             r65);
    r66 = r58 * r44;
    r66 = r66 * r64;
    r67 = r3 * r61;
    r67 = r67 * r58;
    r67 = r67 * r44;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             4 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r66,
                                             r67);
    r68 = r3 * r20;
    r69 = r10 * r62;
    r69 = r69 * r64;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             6 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r69,
                                             r68);
    r70 = r2 * r29;
    r71 = r3 * r10;
    r71 = r71 * r9;
    r71 = r71 * r62;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             8 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r70,
                                             r71);
    r72 = r58 * r55;
    r72 = r72 * r64;
    r73 = r3 * r61;
    r73 = r73 * r58;
    r73 = r73 * r55;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             10 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r72,
                                             r73);
    r74 = r58 * r64;
    r74 = r74 * r54;
    r75 = r3 * r61;
    r75 = r75 * r58;
    r75 = r75 * r54;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             12 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r74,
                                             r75);
    r76 = r2 * r26;
    r77 = r3 * r26;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             14 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r76,
                                             r77);
    r78 = r21 * r60;
    r78 = r78 * r1;
    r79 = r21 * r0;
    r80 = r38 * r79;
    WriteSum2<double, double>((double*)inout_shared, r80, r78);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = r21 * r1;
    WriteSum2<double, double>((double*)inout_shared, r79, r78);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = r26 * r58;
    r78 = r78 * r64;
    r80 = r3 * r21;
    r80 = r80 * r26;
    r80 = r80 * r1;
    r80 = r80 * r61;
    r80 = fma(r58, r80, r79 * r78);
    r78 = r3 * r21;
    r78 = r78 * r1;
    r78 = r78 * r61;
    r78 = r78 * r58;
    r81 = r58 * r44;
    r81 = r81 * r64;
    r81 = fma(r79, r81, r44 * r78);
    WriteSum2<double, double>((double*)inout_shared, r80, r81);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            4 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = r3 * r21;
    r81 = r81 * r20;
    r80 = r23 * r0;
    r80 = r80 * r62;
    r80 = fma(r64, r80, r1 * r81);
    r81 = r2 * r79;
    r78 = r3 * r23;
    r78 = r78 * r1;
    r78 = r78 * r9;
    r78 = fma(r62, r78, r29 * r81);
    WriteSum2<double, double>((double*)inout_shared, r80, r78);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            6 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = r3 * r21;
    r78 = r78 * r1;
    r78 = r78 * r61;
    r78 = r78 * r58;
    r80 = r58 * r55;
    r80 = r80 * r64;
    r80 = fma(r79, r80, r55 * r78);
    r78 = r3 * r21;
    r78 = r78 * r1;
    r78 = r78 * r61;
    r78 = r78 * r58;
    r82 = r58 * r64;
    r82 = r82 * r54;
    r82 = fma(r79, r82, r54 * r78);
    WriteSum2<double, double>((double*)inout_shared, r80, r82);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            8 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r82 = r3 * r21;
    r82 = r82 * r26;
    r82 = r82 * r1;
    r80 = r26 * r81;
    WriteSum2<double, double>((double*)inout_shared, r80, r82);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            10 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r82 = r38 * r38;
    r80 = r60 * r60;
    WriteSum2<double, double>((double*)inout_shared, r82, r80);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            0 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r33, r33);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            2 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = r3 * r3;
    r32 = r80 * r32;
    r82 = r8 * r34;
    r78 = r2 * r64;
    r82 = r82 * r41;
    r82 = r82 * r44;
    r82 = fma(r78, r82, r44 * r32);
    r79 = r8 * r34;
    r79 = r79 * r41;
    r79 = r79 * r54;
    r79 = fma(r78, r79, r54 * r32);
    WriteSum2<double, double>((double*)inout_shared, r82, r79);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            4 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r82 = r61 * r78;
    r83 = r8 * r45;
    r84 = 4.00000000000000000e+00;
    r7 = r37 * r7;
    r37 = r37 * r7;
    r37 = 1.0 / r37;
    r6 = r28 * r6;
    r28 = r28 * r6;
    r28 = 1.0 / r28;
    r83 = r83 * r34;
    r83 = r83 * r34;
    r83 = r83 * r84;
    r83 = r83 * r37;
    r83 = r83 * r28;
    r28 = r20 * r20;
    r28 = fma(r80, r28, r83 * r82);
    r82 = r61 * r80;
    r37 = r9 * r82;
    r85 = r2 * r2;
    r85 = r85 * r29;
    r85 = fma(r29, r85, r83 * r37);
    WriteSum2<double, double>((double*)inout_shared, r28, r85);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            6 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = r55 * r55;
    r28 = r8 * r34;
    r28 = r28 * r41;
    r28 = r28 * r78;
    r28 = fma(r85, r28, r85 * r32);
    r37 = r54 * r54;
    r83 = r8 * r34;
    r83 = r83 * r41;
    r83 = r83 * r78;
    r83 = fma(r37, r83, r32 * r37);
    WriteSum2<double, double>((double*)inout_shared, r28, r83);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            8 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = r2 * r2;
    r83 = r83 * r44;
    r37 = r44 * r80;
    WriteSum2<double, double>((double*)inout_shared, r83, r37);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            10 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = 0.00000000000000000e+00;
    WriteSum2<double, double>((double*)inout_shared, r37, r38);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = r26 * r38;
    r83 = r83 * r58;
    r83 = r83 * r64;
    WriteSum2<double, double>((double*)inout_shared, r37, r83);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            2 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = r38 * r58;
    r83 = r83 * r44;
    r83 = r83 * r64;
    r86 = r10 * r38;
    r86 = r86 * r62;
    r86 = r86 * r64;
    WriteSum2<double, double>((double*)inout_shared, r83, r86);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r86 = r2 * r29;
    r86 = r86 * r38;
    r83 = r38 * r58;
    r83 = r83 * r55;
    r83 = r83 * r64;
    WriteSum2<double, double>((double*)inout_shared, r86, r83);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            6 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = r2 * r26;
    r83 = r83 * r38;
    r38 = r38 * r58;
    r38 = r38 * r64;
    r38 = r38 * r54;
    WriteSum2<double, double>((double*)inout_shared, r38, r83);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            8 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = r3 * r26;
    r83 = r83 * r60;
    r83 = r83 * r61;
    r83 = r83 * r58;
    WriteSum2<double, double>((double*)inout_shared, r60, r83);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            12 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = r3 * r20;
    r83 = r83 * r60;
    r38 = r3 * r60;
    r38 = r38 * r61;
    r38 = r38 * r58;
    r38 = r38 * r44;
    WriteSum2<double, double>((double*)inout_shared, r38, r83);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            14 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = r3 * r10;
    r83 = r83 * r60;
    r83 = r83 * r9;
    r83 = r83 * r62;
    r38 = r3 * r60;
    r38 = r38 * r61;
    r38 = r38 * r58;
    r38 = r38 * r55;
    WriteSum2<double, double>((double*)inout_shared, r83, r38);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            16 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = r3 * r60;
    r38 = r38 * r61;
    r38 = r38 * r58;
    r38 = r38 * r54;
    WriteSum2<double, double>((double*)inout_shared, r38, r37);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            18 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = r3 * r26;
    r38 = r38 * r60;
    WriteSum2<double, double>((double*)inout_shared, r38, r37);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            20 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r63, r66);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            22 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r69, r70);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            24 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r72, r74);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            26 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r76, r37);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            28 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r65, r67);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            30 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r68, r71);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            32 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r73, r75);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            34 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r37, r77);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            36 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = r8 * r34;
    r77 = r77 * r41;
    r77 = r77 * r55;
    r77 = fma(r78, r77, r55 * r32);
    r75 = r58 * r82;
    r73 = r20 * r75;
    r71 = r8 * r26;
    r7 = 1.0 / r7;
    r6 = 1.0 / r6;
    r68 = r10 * r34;
    r71 = r71 * r7;
    r71 = r71 * r6;
    r71 = r71 * r61;
    r71 = r71 * r68;
    r71 = fma(r78, r71, r26 * r73);
    WriteSum2<double, double>((double*)inout_shared, r77, r71);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            38 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = r45 * r26;
    r71 = r71 * r7;
    r71 = r71 * r6;
    r71 = r71 * r9;
    r71 = r71 * r68;
    r77 = r29 * r78;
    r67 = r58 * r77;
    r71 = fma(r26, r67, r82 * r71);
    WriteSum2<double, double>((double*)inout_shared, r71, r79);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            40 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = r58 * r44;
    r79 = r79 * r78;
    r71 = r26 * r54;
    r65 = r8 * r34;
    r65 = r65 * r41;
    r65 = r65 * r78;
    r65 = fma(r71, r65, r71 * r32);
    WriteSum2<double, double>((double*)inout_shared, r65, r79);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            42 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = r44 * r75;
    r76 = r8 * r7;
    r76 = r76 * r6;
    r76 = r76 * r61;
    r76 = r76 * r44;
    r76 = r76 * r68;
    r76 = fma(r78, r76, r44 * r73);
    WriteSum2<double, double>((double*)inout_shared, r79, r76);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            44 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = r45 * r7;
    r76 = r76 * r6;
    r76 = r76 * r9;
    r76 = r76 * r44;
    r76 = r76 * r68;
    r76 = fma(r44, r67, r82 * r76);
    WriteSum2<double, double>((double*)inout_shared, r76, r65);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            46 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = r58 * r55;
    r65 = r65 * r78;
    WriteSum2<double, double>((double*)inout_shared, r28, r65);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            48 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = r55 * r75;
    r28 = r10 * r62;
    r76 = r10 * r20;
    r76 = r76 * r9;
    r76 = r76 * r62;
    r76 = fma(r80, r76, r77 * r28);
    WriteSum2<double, double>((double*)inout_shared, r65, r76);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            50 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = r8 * r7;
    r76 = r76 * r6;
    r76 = r76 * r61;
    r76 = r76 * r55;
    r76 = r76 * r68;
    r76 = fma(r78, r76, r55 * r73);
    r65 = r8 * r7;
    r65 = r65 * r6;
    r65 = r65 * r61;
    r65 = r65 * r54;
    r65 = r65 * r68;
    r65 = fma(r78, r65, r54 * r73);
    WriteSum2<double, double>((double*)inout_shared, r76, r65);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            52 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = r10 * r26;
    r65 = r65 * r62;
    r65 = r65 * r78;
    r76 = r26 * r20;
    r76 = r76 * r80;
    WriteSum2<double, double>((double*)inout_shared, r65, r76);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            54 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = r45 * r7;
    r76 = r76 * r6;
    r76 = r76 * r9;
    r76 = r76 * r55;
    r76 = r76 * r68;
    r76 = fma(r55, r67, r82 * r76);
    r65 = r45 * r7;
    r65 = r65 * r6;
    r65 = r65 * r9;
    r65 = r65 * r54;
    r65 = r65 * r68;
    r67 = fma(r54, r67, r82 * r65);
    WriteSum2<double, double>((double*)inout_shared, r76, r67);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            56 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = r2 * r2;
    r67 = r67 * r26;
    r67 = r67 * r29;
    r29 = r10 * r26;
    r29 = r29 * r9;
    r29 = r29 * r62;
    r29 = r29 * r80;
    WriteSum2<double, double>((double*)inout_shared, r67, r29);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            58 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = r58 * r54;
    r29 = r29 * r78;
    r85 = r26 * r85;
    r67 = r8 * r34;
    r67 = r67 * r41;
    r67 = r67 * r78;
    r67 = fma(r85, r67, r32 * r85);
    WriteSum2<double, double>((double*)inout_shared, r67, r29);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            60 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = r54 * r75;
    r29 = r58 * r78;
    r29 = r29 * r71;
    WriteSum2<double, double>((double*)inout_shared, r54, r29);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            62 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = r71 * r75;
    WriteSum2<double, double>((double*)inout_shared, r75, r37);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            64 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = r8 * r8;
    r75 = r25 * r34;
    r71 = -6.00000000000000000e+00;
    r75 = r75 * r71;
    r75 = r75 * r27;
    r75 = r75 * r7;
    r37 = r37 * r34;
    r29 = r10 * r42;
    r29 = r29 * r8;
    r54 = r25 * r45;
    r67 = r23 * r7;
    r54 = r54 * r45;
    r54 = fma(r67, r54, r35 * r29);
    r29 = r8 * r8;
    r29 = r29 * r67;
    r85 = r10 * r46;
    r85 = r85 * r45;
    r54 = fma(r35, r85, r54);
    r54 = fma(r25, r29, r54);
    r85 = r31 * r54;
    r32 = rsqrt(r14);
    r14 = r33 + r14;
    r14 = 1.0 / r14;
    r14 = r32 * r14;
    r33 = r8 * r14;
    r76 = r41 * r9;
    r85 = r85 * r33;
    r85 = fma(r76, r85, r75 * r37);
    r37 = r42 * r34;
    r65 = 6.00000000000000000e+00;
    r37 = r37 * r65;
    r37 = r37 * r41;
    r85 = fma(r9, r37, r85);
    r73 = -3.00000000000000000e+00;
    r73 = r34 * r73;
    r73 = r73 * r35;
    r73 = r73 * r6;
    r73 = r73 * r32;
    r28 = r54 * r73;
    r77 = r34 * r29;
    r85 = fma(r77, r28, r85);
    r79 = r25 * r45;
    r79 = r79 * r34;
    r79 = r79 * r27;
    r79 = r79 * r61;
    r74 = r21 * r45;
    r74 = r74 * r34;
    r74 = r74 * r54;
    r74 = r74 * r35;
    r74 = r74 * r6;
    r74 = r74 * r32;
    r74 = fma(r61, r74, r67 * r79);
    r79 = r54 * r14;
    r72 = r45 * r62;
    r74 = fma(r72, r79, r74);
    r70 = r46 * r62;
    r74 = fma(r68, r70, r74);
    r85 = r85 + r74;
    r28 = r34 * r27;
    r28 = r28 * r77;
    r37 = r54 * r33;
    r37 = fma(r76, r37, r25 * r28);
    r70 = r68 * r76;
    r79 = r21 * r8;
    r79 = r79 * r8;
    r79 = r79 * r34;
    r79 = r79 * r34;
    r79 = r79 * r54;
    r79 = r79 * r35;
    r79 = r79 * r6;
    r37 = fma(r32, r79, r37);
    r37 = fma(r42, r70, r37);
    r74 = r74 + r37;
    r85 = fma(r49, r74, r5 * r85);
    r79 = r4 * r25;
    r69 = -4.00000000000000000e+00;
    r69 = r69 * r27;
    r69 = r69 * r7;
    r69 = r69 * r61;
    r69 = r69 * r9;
    r85 = fma(r69, r79, r85);
    r66 = r48 * r10;
    r66 = r66 * r26;
    r66 = fma(r74, r66, r50 * r74);
    r52 = r52 * r31;
    r52 = r52 * r44;
    r84 = r53 * r84;
    r84 = r84 * r55;
    r66 = fma(r74, r52, r66);
    r66 = fma(r74, r84, r66);
    r53 = r66 * r58;
    r85 = fma(r9, r53, r85);
    r63 = r4 * r54;
    r38 = r10 * r62;
    r38 = r38 * r33;
    r85 = fma(r38, r63, r85);
    r60 = r21 * r25;
    r60 = r60 * r8;
    r60 = r60 * r34;
    r60 = r60 * r51;
    r60 = r60 * r35;
    r85 = fma(r57, r60, r85);
    r83 = r4 * r70;
    r86 = r42 * r34;
    r85 = fma(r59, r86, r85);
    r87 = 5.00000000000000000e-01;
    r88 = r87 * r54;
    r88 = r88 * r58;
    r85 = fma(r33, r88, r85);
    r89 = r54 * r33;
    r90 = r87 * r59;
    r85 = fma(r90, r89, r85);
    r91 = r21 * r25;
    r91 = r91 * r8;
    r91 = r91 * r34;
    r91 = r91 * r35;
    r85 = fma(r57, r91, r85);
    r92 = r4 * r42;
    r92 = r92 * r62;
    r85 = fma(r68, r92, r85);
    r93 = r42 * r34;
    r85 = fma(r58, r93, r85);
    r94 = r4 * r23;
    r94 = r94 * r54;
    r94 = r94 * r35;
    r94 = r94 * r6;
    r94 = r94 * r32;
    r94 = r94 * r61;
    r85 = fma(r9, r94, r85);
    r95 = r8 * r34;
    r96 = -5.00000000000000000e-01;
    r97 = r96 * r54;
    r97 = r97 * r27;
    r97 = r97 * r56;
    r97 = r97 * r32;
    r95 = r95 * r51;
    r85 = fma(r97, r95, r85);
    r98 = r8 * r34;
    r85 = fma(r97, r98, r85);
    r85 = fma(r46, r83, r85);
    r98 = r2 * r85;
    r95 = r45 * r61;
    r94 = r45 * r54;
    r94 = r94 * r61;
    r94 = fma(r73, r94, r75 * r95);
    r95 = r31 * r54;
    r95 = r95 * r14;
    r94 = fma(r72, r95, r94);
    r75 = r46 * r34;
    r75 = r75 * r65;
    r94 = fma(r62, r75, r94);
    r94 = r94 + r37;
    r74 = fma(r47, r74, r4 * r94);
    r94 = r45 * r27;
    r94 = r94 * r34;
    r94 = r94 * r56;
    r94 = r94 * r96;
    r94 = r94 * r32;
    r94 = r94 * r54;
    r37 = r45 * r87;
    r37 = r37 * r54;
    r37 = r37 * r58;
    r74 = fma(r14, r37, r74);
    r75 = r45 * r14;
    r75 = r75 * r90;
    r95 = r5 * r69;
    r93 = r21 * r25;
    r93 = r93 * r35;
    r93 = r93 * r57;
    r74 = fma(r61, r93, r74);
    r92 = r5 * r54;
    r74 = fma(r38, r92, r74);
    r91 = r21 * r25;
    r91 = r91 * r51;
    r91 = r91 * r35;
    r91 = r91 * r57;
    r74 = fma(r61, r91, r74);
    r89 = r46 * r34;
    r74 = fma(r59, r89, r74);
    r88 = r5 * r46;
    r74 = fma(r70, r88, r74);
    r86 = r66 * r61;
    r74 = fma(r58, r86, r74);
    r60 = r46 * r34;
    r74 = fma(r58, r60, r74);
    r63 = r5 * r42;
    r63 = r63 * r62;
    r74 = fma(r68, r63, r74);
    r53 = r5 * r23;
    r53 = r53 * r54;
    r53 = r53 * r35;
    r53 = r53 * r6;
    r53 = r53 * r32;
    r53 = r53 * r61;
    r74 = fma(r9, r53, r74);
    r74 = r74 + r94;
    r74 = fma(r54, r75, r74);
    r74 = fma(r25, r95, r74);
    r74 = fma(r51, r94, r74);
    r53 = r3 * r74;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r98,
                                             r53);
    r53 = r30 * r34;
    r53 = r53 * r65;
    r53 = r53 * r41;
    r98 = r40 * r8;
    r98 = r98 * r8;
    r98 = r98 * r34;
    r98 = r98 * r34;
    r98 = r98 * r71;
    r98 = r98 * r27;
    r98 = fma(r7, r98, r9 * r53);
    r53 = r10 * r30;
    r53 = r53 * r8;
    r53 = fma(r35, r53, r40 * r29);
    r63 = r10 * r36;
    r63 = r63 * r45;
    r53 = fma(r35, r63, r53);
    r60 = r40 * r45;
    r60 = r60 * r45;
    r53 = fma(r67, r60, r53);
    r60 = r53 * r73;
    r98 = fma(r77, r60, r98);
    r63 = r31 * r53;
    r63 = r63 * r33;
    r98 = fma(r76, r63, r98);
    r86 = r40 * r45;
    r86 = r86 * r34;
    r86 = r86 * r27;
    r86 = r86 * r61;
    r94 = r53 * r14;
    r94 = fma(r72, r94, r67 * r86);
    r86 = r36 * r62;
    r94 = fma(r68, r86, r94);
    r88 = r21 * r45;
    r88 = r88 * r34;
    r88 = r88 * r53;
    r88 = r88 * r35;
    r88 = r88 * r6;
    r88 = r88 * r32;
    r94 = fma(r61, r88, r94);
    r98 = r98 + r94;
    r63 = fma(r40, r28, r30 * r70);
    r60 = r21 * r8;
    r60 = r60 * r8;
    r60 = r60 * r34;
    r60 = r60 * r34;
    r60 = r60 * r53;
    r60 = r60 * r35;
    r60 = r60 * r6;
    r63 = fma(r32, r60, r63);
    r88 = r53 * r33;
    r63 = fma(r76, r88, r63);
    r94 = r94 + r63;
    r98 = fma(r49, r94, r5 * r98);
    r88 = r87 * r53;
    r88 = r88 * r58;
    r98 = fma(r33, r88, r98);
    r60 = r30 * r34;
    r98 = fma(r59, r60, r98);
    r86 = r53 * r38;
    r89 = r53 * r33;
    r98 = fma(r90, r89, r98);
    r91 = r21 * r40;
    r91 = r91 * r8;
    r91 = r91 * r34;
    r91 = r91 * r35;
    r98 = fma(r57, r91, r98);
    r92 = r30 * r34;
    r98 = fma(r58, r92, r98);
    r93 = r8 * r34;
    r93 = r93 * r51;
    r93 = r93 * r96;
    r93 = r93 * r53;
    r93 = r93 * r27;
    r93 = r93 * r56;
    r98 = fma(r32, r93, r98);
    r37 = r4 * r23;
    r37 = r37 * r53;
    r37 = r37 * r35;
    r37 = r37 * r6;
    r37 = r37 * r32;
    r37 = r37 * r61;
    r98 = fma(r9, r37, r98);
    r79 = r48 * r10;
    r79 = r79 * r26;
    r79 = fma(r94, r79, r50 * r94);
    r79 = fma(r94, r52, r79);
    r79 = fma(r94, r84, r79);
    r97 = r79 * r58;
    r98 = fma(r9, r97, r98);
    r99 = r8 * r34;
    r99 = r99 * r96;
    r99 = r99 * r53;
    r99 = r99 * r27;
    r99 = r99 * r56;
    r98 = fma(r32, r99, r98);
    r100 = r21 * r40;
    r100 = r100 * r8;
    r100 = r100 * r34;
    r100 = r100 * r51;
    r100 = r100 * r35;
    r98 = fma(r57, r100, r98);
    r101 = r4 * r40;
    r98 = fma(r69, r101, r98);
    r102 = r4 * r30;
    r102 = r102 * r62;
    r98 = fma(r68, r102, r98);
    r98 = fma(r36, r83, r98);
    r98 = fma(r4, r86, r98);
    r102 = r2 * r98;
    r101 = r40 * r45;
    r101 = r101 * r34;
    r101 = r101 * r71;
    r101 = r101 * r27;
    r101 = r101 * r7;
    r100 = r31 * r53;
    r100 = r100 * r14;
    r100 = fma(r72, r100, r61 * r101);
    r101 = r36 * r34;
    r101 = r101 * r65;
    r100 = fma(r62, r101, r100);
    r99 = r45 * r53;
    r99 = r99 * r61;
    r100 = fma(r73, r99, r100);
    r100 = r100 + r63;
    r94 = fma(r47, r94, r4 * r100);
    r100 = r21 * r40;
    r100 = r100 * r35;
    r100 = r100 * r57;
    r94 = fma(r61, r100, r94);
    r63 = r51 * r96;
    r63 = r63 * r53;
    r63 = r63 * r27;
    r63 = r63 * r56;
    r63 = r63 * r32;
    r94 = fma(r61, r63, r94);
    r99 = r5 * r36;
    r94 = fma(r70, r99, r94);
    r101 = r45 * r87;
    r101 = r101 * r53;
    r101 = r101 * r58;
    r94 = fma(r14, r101, r94);
    r97 = r36 * r34;
    r94 = fma(r59, r97, r94);
    r37 = r21 * r40;
    r37 = r37 * r51;
    r37 = r37 * r35;
    r37 = r37 * r57;
    r94 = fma(r61, r37, r94);
    r93 = r5 * r23;
    r93 = r93 * r53;
    r93 = r93 * r35;
    r93 = r93 * r6;
    r93 = r93 * r32;
    r93 = r93 * r61;
    r94 = fma(r9, r93, r94);
    r92 = r96 * r53;
    r92 = r92 * r27;
    r92 = r92 * r56;
    r92 = r92 * r32;
    r94 = fma(r61, r92, r94);
    r91 = r79 * r61;
    r94 = fma(r58, r91, r94);
    r89 = r36 * r34;
    r94 = fma(r58, r89, r94);
    r60 = r5 * r30;
    r60 = r60 * r62;
    r94 = fma(r68, r60, r94);
    r94 = fma(r53, r75, r94);
    r94 = fma(r5, r86, r94);
    r94 = fma(r40, r95, r94);
    r60 = r3 * r94;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r102,
                                             r60);
    r60 = r43 * r8;
    r60 = r60 * r8;
    r60 = r60 * r34;
    r60 = r60 * r34;
    r60 = r60 * r71;
    r60 = r60 * r27;
    r102 = r10 * r22;
    r102 = r102 * r8;
    r89 = r43 * r45;
    r89 = r89 * r45;
    r89 = fma(r67, r89, r35 * r102);
    r102 = r10 * r39;
    r102 = r102 * r45;
    r89 = fma(r35, r102, r89);
    r89 = fma(r43, r29, r89);
    r102 = r89 * r73;
    r77 = fma(r77, r102, r7 * r60);
    r60 = r31 * r89;
    r60 = r60 * r33;
    r77 = fma(r76, r60, r77);
    r29 = r22 * r34;
    r29 = r29 * r65;
    r29 = r29 * r41;
    r77 = fma(r9, r29, r77);
    r41 = r39 * r62;
    r91 = r21 * r45;
    r91 = r91 * r34;
    r91 = r91 * r89;
    r91 = r91 * r35;
    r91 = r91 * r6;
    r91 = r91 * r32;
    r91 = fma(r61, r91, r68 * r41);
    r41 = r89 * r14;
    r91 = fma(r72, r41, r91);
    r92 = r43 * r45;
    r92 = r92 * r34;
    r92 = r92 * r27;
    r92 = r92 * r61;
    r91 = fma(r67, r92, r91);
    r77 = r77 + r91;
    r29 = r21 * r8;
    r29 = r29 * r8;
    r29 = r29 * r34;
    r29 = r29 * r34;
    r29 = r29 * r89;
    r29 = r29 * r35;
    r29 = r29 * r6;
    r29 = fma(r32, r29, r43 * r28);
    r28 = r89 * r33;
    r29 = fma(r76, r28, r29);
    r29 = fma(r22, r70, r29);
    r91 = r91 + r29;
    r49 = fma(r49, r91, r5 * r77);
    r77 = r8 * r34;
    r77 = r77 * r51;
    r77 = r77 * r96;
    r77 = r77 * r89;
    r77 = r77 * r27;
    r77 = r77 * r56;
    r49 = fma(r32, r77, r49);
    r28 = r21 * r43;
    r28 = r28 * r8;
    r28 = r28 * r34;
    r28 = r28 * r35;
    r49 = fma(r57, r28, r49);
    r76 = r8 * r34;
    r76 = r76 * r96;
    r76 = r76 * r89;
    r76 = r76 * r27;
    r76 = r76 * r56;
    r49 = fma(r32, r76, r49);
    r60 = r87 * r89;
    r60 = r60 * r58;
    r49 = fma(r33, r60, r49);
    r92 = r4 * r43;
    r49 = fma(r69, r92, r49);
    r69 = r4 * r22;
    r69 = r69 * r62;
    r49 = fma(r68, r69, r49);
    r41 = r4 * r89;
    r49 = fma(r38, r41, r49);
    r67 = r21 * r43;
    r67 = r67 * r8;
    r67 = r67 * r34;
    r67 = r67 * r51;
    r67 = r67 * r35;
    r49 = fma(r57, r67, r49);
    r93 = r4 * r23;
    r93 = r93 * r89;
    r93 = r93 * r35;
    r93 = r93 * r6;
    r93 = r93 * r32;
    r93 = r93 * r61;
    r49 = fma(r9, r93, r49);
    r37 = r22 * r34;
    r49 = fma(r58, r37, r49);
    r97 = r48 * r10;
    r97 = r97 * r26;
    r50 = fma(r50, r91, r91 * r97);
    r50 = fma(r91, r52, r50);
    r50 = fma(r91, r84, r50);
    r84 = r50 * r58;
    r49 = fma(r9, r84, r49);
    r52 = r89 * r33;
    r49 = fma(r90, r52, r49);
    r90 = r22 * r34;
    r49 = fma(r59, r90, r49);
    r49 = fma(r39, r83, r49);
    r90 = r2 * r49;
    r52 = r39 * r34;
    r52 = r52 * r65;
    r65 = r45 * r61;
    r65 = fma(r102, r65, r62 * r52);
    r52 = r31 * r89;
    r52 = r52 * r14;
    r65 = fma(r72, r52, r65);
    r72 = r43 * r45;
    r72 = r72 * r34;
    r72 = r72 * r71;
    r72 = r72 * r27;
    r72 = r72 * r7;
    r65 = fma(r61, r72, r65);
    r65 = r65 + r29;
    r91 = fma(r47, r91, r4 * r65);
    r47 = r51 * r96;
    r47 = r47 * r89;
    r47 = r47 * r27;
    r47 = r47 * r56;
    r47 = r47 * r32;
    r91 = fma(r61, r47, r91);
    r65 = r96 * r89;
    r65 = r65 * r27;
    r65 = r65 * r56;
    r65 = r65 * r32;
    r91 = fma(r61, r65, r91);
    r56 = r5 * r22;
    r56 = r56 * r62;
    r91 = fma(r68, r56, r91);
    r68 = r5 * r89;
    r91 = fma(r38, r68, r91);
    r38 = r21 * r43;
    r38 = r38 * r51;
    r38 = r38 * r35;
    r38 = r38 * r57;
    r91 = fma(r61, r38, r91);
    r27 = r50 * r61;
    r91 = fma(r58, r27, r91);
    r29 = r5 * r39;
    r91 = fma(r70, r29, r91);
    r70 = r5 * r23;
    r70 = r70 * r89;
    r70 = r70 * r35;
    r70 = r70 * r6;
    r70 = r70 * r32;
    r70 = r70 * r61;
    r91 = fma(r9, r70, r91);
    r32 = r39 * r34;
    r91 = fma(r58, r32, r91);
    r6 = r45 * r87;
    r6 = r6 * r89;
    r6 = r6 * r58;
    r91 = fma(r14, r6, r91);
    r72 = r39 * r34;
    r91 = fma(r59, r72, r91);
    r59 = r21 * r43;
    r59 = r59 * r35;
    r59 = r59 * r57;
    r91 = fma(r61, r59, r91);
    r91 = fma(r43, r95, r91);
    r91 = fma(r89, r75, r91);
    r59 = r3 * r91;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r90,
                                             r59);
    r59 = r3 * r21;
    r59 = r59 * r1;
    r59 = fma(r85, r81, r74 * r59);
    r90 = r3 * r21;
    r90 = r90 * r1;
    r90 = fma(r98, r81, r94 * r90);
    WriteSum2<double, double>((double*)inout_shared, r59, r90);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r90 = r3 * r21;
    r90 = r90 * r1;
    r81 = fma(r49, r81, r91 * r90);
    WriteSum1<double, double>((double*)inout_shared, r81);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = r2 * r2;
    r90 = r85 * r85;
    r1 = r74 * r74;
    r1 = fma(r80, r1, r90 * r81);
    r90 = r94 * r94;
    r59 = r98 * r98;
    r59 = fma(r59, r81, r80 * r90);
    WriteSum2<double, double>((double*)inout_shared, r1, r59);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = r91 * r91;
    r1 = r49 * r49;
    r81 = fma(r1, r81, r80 * r59);
    WriteSum1<double, double>((double*)inout_shared, r81);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = r2 * r2;
    r81 = r81 * r85;
    r1 = r74 * r94;
    r1 = fma(r80, r1, r98 * r81);
    r81 = r74 * r91;
    r59 = r2 * r2;
    r59 = r59 * r85;
    r59 = fma(r49, r59, r80 * r81);
    WriteSum2<double, double>((double*)inout_shared, r1, r59);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = r94 * r91;
    r1 = r2 * r2;
    r1 = r1 * r98;
    r1 = fma(r49, r1, r80 * r59);
    WriteSum1<double, double>((double*)inout_shared, r1);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
}

void ThinPrismFisheyeFixedPoseResJac(
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
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFixedPoseResJacKernel<<<n_blocks, 1024>>>(
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