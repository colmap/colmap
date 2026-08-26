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
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* out_calib_jac,
        unsigned int out_calib_jac_num_alloc,
        float* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        float* const out_calib_precond_diag,
        unsigned int out_calib_precond_diag_num_alloc,
        float* const out_calib_precond_tril,
        unsigned int out_calib_precond_tril_num_alloc,
        float* out_point_jac,
        unsigned int out_point_jac_num_alloc,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        float* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        float* const out_point_precond_tril,
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102, r103, r104,
      r105, r106;
  LoadShared<4, float, float>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r0,
                       r1,
                       r2,
                       r3);
  };
  __syncthreads();
  LoadShared<4, float, float>(
      calib, 4 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r4,
                       r5,
                       r6,
                       r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = 9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r9,
                                         r10,
                                         r11);
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r12,
                       r13,
                       r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = 2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r16,
                                         r17,
                                         r18,
                                         r19);
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r20, r21, r22, r23);
    r24 = fmaf(r16, r21, r19 * r22);
    r25 = r17 * r20;
    r26 = -1.00000000000000000e+00;
    r24 = fmaf(r26, r25, r24);
    r24 = fmaf(r18, r23, r24);
    r25 = r15 * r24;
    r27 = fmaf(r16, r23, r19 * r20);
    r28 = r18 * r21;
    r27 = fmaf(r26, r28, r27);
    r27 = fmaf(r17, r22, r27);
    r25 = r25 * r27;
    r28 = -2.00000000000000000e+00;
    r29 = r16 * r22;
    r29 = fmaf(r26, r29, r19 * r21);
    r29 = fmaf(r17, r23, r29);
    r29 = fmaf(r18, r20, r29);
    r30 = fmaf(r17, r21, r16 * r20);
    r30 = fmaf(r18, r22, r30);
    r30 = fmaf(r26, r30, r19 * r23);
    r23 = r29 * r30;
    r31 = fmaf(r28, r23, r25);
    r11 = fmaf(r12, r31, r11);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r32, r33, r34);
    r35 = r16 * r18;
    r35 = r35 * r15;
    r36 = r17 * r19;
    r37 = fmaf(r28, r36, r35);
    r38 = r17 * r17;
    r38 = r38 * r28;
    r39 = 1.00000000000000000e+00;
    r40 = r16 * r16;
    r40 = fmaf(r28, r40, r39);
    r41 = r38 + r40;
    r42 = r17 * r18;
    r42 = r42 * r15;
    r43 = r16 * r19;
    r43 = fmaf(r15, r43, r42);
    r44 = r15 * r24;
    r44 = r44 * r29;
    r45 = r15 * r27;
    r45 = fmaf(r30, r45, r44);
    r46 = r28 * r29;
    r46 = r46 * r29;
    r47 = r39 + r46;
    r48 = r27 * r27;
    r48 = r28 * r48;
    r47 = r47 + r48;
    r11 = fmaf(r32, r37, r11);
    r11 = fmaf(r34, r41, r11);
    r11 = fmaf(r33, r43, r11);
    r11 = fmaf(r13, r45, r11);
    r11 = fmaf(r14, r47, r11);
    r43 = copysign(1.0, r11);
    r43 = fmaf(r8, r43, r11);
    r11 = r43 * r43;
    r41 = 1.0 / r11;
    r37 = r15 * r27;
    r37 = r37 * r29;
    r29 = r15 * r24;
    r29 = fmaf(r30, r29, r37);
    r10 = fmaf(r12, r29, r10);
    r49 = r16 * r17;
    r49 = r49 * r15;
    r50 = r18 * r19;
    r50 = fmaf(r15, r50, r49);
    r51 = r18 * r18;
    r51 = r28 * r51;
    r40 = r51 + r40;
    r52 = r16 * r19;
    r52 = fmaf(r28, r52, r42);
    r42 = r27 * r28;
    r42 = fmaf(r30, r42, r44);
    r48 = r39 + r48;
    r44 = r24 * r24;
    r44 = r28 * r44;
    r48 = r48 + r44;
    r10 = fmaf(r32, r50, r10);
    r10 = fmaf(r33, r40, r10);
    r10 = fmaf(r34, r52, r10);
    r10 = fmaf(r14, r42, r10);
    r10 = fmaf(r13, r48, r10);
    r52 = r10 * r10;
    r52 = r41 * r52;
    r46 = r39 + r46;
    r46 = r46 + r44;
    r12 = fmaf(r12, r46, r9);
    r9 = r24 * r28;
    r9 = fmaf(r30, r9, r37);
    r23 = fmaf(r15, r23, r25);
    r36 = fmaf(r15, r36, r35);
    r35 = r18 * r19;
    r35 = fmaf(r28, r35, r49);
    r38 = r39 + r38;
    r38 = r38 + r51;
    r12 = fmaf(r13, r9, r12);
    r12 = fmaf(r14, r23, r12);
    r12 = fmaf(r34, r36, r12);
    r12 = fmaf(r33, r35, r12);
    r12 = fmaf(r32, r38, r12);
    r38 = r12 * r12;
    r32 = r10 * r10;
    r32 = fmaf(r41, r32, r41 * r38);
    r38 = sqrtf(r32);
    r35 = copysign(1.0, r38);
    r35 = fmaf(r8, r35, r38);
    r8 = r35 * r35;
    r33 = 1.0 / r8;
    r38 = atanf(r38);
    r36 = r38 * r38;
    r52 = r52 * r33;
    r52 = r52 * r36;
    r36 = r12 * r38;
    r34 = 3.00000000000000000e+00;
    r14 = r41 * r33;
    r13 = r12 * r38;
    r36 = r36 * r34;
    r36 = r36 * r14;
    r36 = fmaf(r13, r36, r52);
  };
  LoadShared<4, float, float>(
      calib, 8 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r51,
                       r49,
                       r25,
                       r37);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r30 = r12 * r38;
    r30 = r30 * r14;
    r30 = r30 * r13;
    r44 = r52 + r30;
    r40 = fmaf(r25, r44, r7 * r36);
    r50 = r6 * r15;
    r53 = r10 * r38;
    r54 = r53 * r14;
    r50 = r50 * r13;
    r40 = fmaf(r54, r50, r40);
    r55 = r44 * r44;
    r56 = fmaf(r5, r55, r4 * r44);
    r57 = r55 * r55;
    r58 = r44 * r55;
    r56 = fmaf(r49, r57, r56);
    r56 = fmaf(r51, r58, r56);
    r59 = 1.0 / r43;
    r60 = 1.0 / r35;
    r61 = r59 * r60;
    r62 = r56 * r61;
    r40 = fmaf(r13, r62, r40);
    r40 = fmaf(r61, r13, r40);
    r2 = fmaf(r0, r40, r2);
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r50, r63);
    r2 = fmaf(r50, r26, r2);
    r30 = fmaf(r34, r52, r30);
    r50 = fmaf(r37, r44, r6 * r30);
    r64 = r7 * r15;
    r64 = r64 * r13;
    r50 = fmaf(r54, r64, r50);
    r50 = fmaf(r53, r61, r50);
    r50 = fmaf(r53, r62, r50);
    r3 = fmaf(r1, r50, r3);
    r3 = fmaf(r63, r26, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r63 = r44 * r61;
    r64 = r0 * r13;
    r63 = r63 * r64;
    r65 = r1 * r44;
    r65 = r65 * r53;
    r65 = r65 * r61;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          0 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r40,
                                          r50,
                                          r63,
                                          r65);
    r66 = r1 * r30;
    r67 = r61 * r55;
    r67 = r67 * r64;
    r68 = r1 * r53;
    r68 = r68 * r61;
    r68 = r68 * r55;
    r69 = r15 * r54;
    r69 = r69 * r64;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          4 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r67,
                                          r68,
                                          r69,
                                          r66);
    r70 = r0 * r36;
    r71 = r1 * r15;
    r71 = r71 * r13;
    r71 = r71 * r54;
    r72 = r61 * r58;
    r72 = r72 * r64;
    r73 = r1 * r53;
    r73 = r73 * r61;
    r73 = r73 * r58;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          8 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r70,
                                          r71,
                                          r72,
                                          r73);
    r74 = r0 * r44;
    r75 = r1 * r44;
    r76 = r61 * r64;
    r76 = r76 * r57;
    r77 = r1 * r53;
    r77 = r77 * r61;
    r77 = r77 * r57;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          12 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r76,
                                          r77,
                                          r74,
                                          r75);
    r78 = r26 * r50;
    r78 = r78 * r3;
    r79 = r26 * r2;
    r80 = r26 * r3;
    r81 = r40 * r79;
    WriteSum4<float, float>((float*)inout_shared, r81, r78, r79, r80);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = r1 * r26;
    r80 = r80 * r44;
    r80 = r80 * r3;
    r80 = r80 * r53;
    r78 = r44 * r61;
    r78 = r78 * r64;
    r78 = fmaf(r79, r78, r61 * r80);
    r80 = r1 * r26;
    r80 = r80 * r3;
    r80 = r80 * r53;
    r80 = r80 * r61;
    r81 = r61 * r55;
    r81 = r81 * r64;
    r81 = fmaf(r79, r81, r55 * r80);
    r80 = r1 * r26;
    r80 = r80 * r30;
    r82 = r28 * r2;
    r82 = r82 * r54;
    r82 = fmaf(r64, r82, r3 * r80);
    r80 = r0 * r79;
    r83 = r1 * r28;
    r83 = r83 * r3;
    r83 = r83 * r13;
    r83 = fmaf(r54, r83, r36 * r80);
    WriteSum4<float, float>((float*)inout_shared, r78, r81, r82, r83);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           4 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = r1 * r26;
    r83 = r83 * r44;
    r83 = r83 * r3;
    r82 = r44 * r80;
    r81 = r1 * r26;
    r81 = r81 * r3;
    r81 = r81 * r53;
    r81 = r81 * r61;
    r78 = r61 * r58;
    r78 = r78 * r64;
    r78 = fmaf(r79, r78, r58 * r81);
    r81 = r1 * r26;
    r81 = r81 * r3;
    r81 = r81 * r53;
    r81 = r81 * r61;
    r84 = r61 * r64;
    r84 = r84 * r57;
    r84 = fmaf(r79, r84, r57 * r81);
    WriteSum4<float, float>((float*)inout_shared, r78, r84, r82, r83);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           8 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = r40 * r40;
    r82 = r50 * r50;
    WriteSum4<float, float>((float*)inout_shared, r83, r82, r39, r39);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r82 = r12 * r38;
    r83 = r0 * r64;
    r82 = r82 * r14;
    r82 = r82 * r55;
    r84 = r1 * r1;
    r52 = r84 * r52;
    r82 = fmaf(r55, r52, r83 * r82);
    r78 = r12 * r38;
    r78 = r78 * r14;
    r78 = r78 * r57;
    r78 = fmaf(r57, r52, r83 * r78);
    r81 = r30 * r30;
    r79 = r53 * r83;
    r85 = r12 * r10;
    r86 = 4.00000000000000000e+00;
    r11 = r43 * r11;
    r43 = r43 * r11;
    r43 = 1.0 / r43;
    r8 = r35 * r8;
    r35 = r35 * r8;
    r35 = 1.0 / r35;
    r85 = r85 * r38;
    r85 = r85 * r38;
    r85 = r85 * r86;
    r85 = r85 * r43;
    r85 = r85 * r35;
    r79 = fmaf(r85, r79, r84 * r81);
    r81 = r0 * r0;
    r81 = r81 * r36;
    r35 = r53 * r84;
    r43 = r13 * r35;
    r43 = fmaf(r85, r43, r36 * r81);
    WriteSum4<float, float>((float*)inout_shared, r82, r78, r79, r43);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           4 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r0 * r0;
    r43 = r43 * r55;
    r79 = r55 * r84;
    r82 = r12 * r38;
    r81 = r58 * r58;
    r82 = r82 * r14;
    r82 = r82 * r83;
    r82 = fmaf(r81, r52, r81 * r82);
    r85 = r12 * r38;
    r87 = r57 * r57;
    r85 = r85 * r14;
    r85 = r85 * r83;
    r87 = fmaf(r52, r87, r87 * r85);
    WriteSum4<float, float>((float*)inout_shared, r82, r87, r43, r79);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           8 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = 0.00000000000000000e+00;
    r43 = r44 * r40;
    r43 = r43 * r61;
    r43 = r43 * r64;
    WriteSum4<float, float>((float*)inout_shared, r79, r40, r79, r43);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r0 * r36;
    r43 = r43 * r40;
    r87 = r40 * r61;
    r87 = r87 * r55;
    r87 = r87 * r64;
    r85 = r15 * r40;
    r85 = r85 * r54;
    r85 = r85 * r64;
    r88 = r40 * r61;
    r88 = r88 * r58;
    r88 = r88 * r64;
    WriteSum4<float, float>((float*)inout_shared, r87, r85, r43, r88);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r88 = r0 * r44;
    r88 = r88 * r40;
    r40 = r40 * r61;
    r40 = r40 * r64;
    r40 = r40 * r57;
    WriteSum4<float, float>((float*)inout_shared, r40, r88, r79, r79);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           8 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r88 = r1 * r30;
    r88 = r88 * r50;
    r40 = r1 * r44;
    r40 = r40 * r50;
    r40 = r40 * r53;
    r40 = r40 * r61;
    r43 = r1 * r50;
    r43 = r43 * r53;
    r43 = r43 * r61;
    r43 = r43 * r55;
    WriteSum4<float, float>((float*)inout_shared, r50, r40, r43, r88);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           12 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r88 = r1 * r15;
    r88 = r88 * r50;
    r88 = r88 * r13;
    r88 = r88 * r54;
    r43 = r1 * r50;
    r43 = r43 * r53;
    r43 = r43 * r61;
    r43 = r43 * r58;
    r40 = r1 * r50;
    r40 = r40 * r53;
    r40 = r40 * r61;
    r40 = r40 * r57;
    WriteSum4<float, float>((float*)inout_shared, r88, r43, r40, r79);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           16 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r1 * r44;
    r40 = r40 * r50;
    WriteSum4<float, float>((float*)inout_shared, r40, r79, r63, r67);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           20 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float*)inout_shared, r69, r70, r72, r76);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           24 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float*)inout_shared, r74, r79, r65, r68);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           28 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float*)inout_shared, r66, r71, r73, r77);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           32 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = r12 * r38;
    r77 = r77 * r14;
    r77 = r77 * r58;
    r77 = fmaf(r58, r52, r83 * r77);
    r73 = r12 * r44;
    r11 = 1.0 / r11;
    r8 = 1.0 / r8;
    r71 = r15 * r38;
    r73 = r73 * r11;
    r73 = r73 * r8;
    r73 = r73 * r53;
    r73 = r73 * r71;
    r66 = r61 * r35;
    r68 = r30 * r66;
    r73 = fmaf(r44, r68, r83 * r73);
    WriteSum4<float, float>((float*)inout_shared, r79, r75, r77, r73);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           36 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = r61 * r55;
    r73 = r73 * r83;
    r77 = r36 * r83;
    r75 = r61 * r77;
    r65 = r10 * r44;
    r65 = r65 * r11;
    r65 = r65 * r8;
    r65 = r65 * r13;
    r65 = r65 * r71;
    r65 = fmaf(r35, r65, r44 * r75);
    r74 = r12 * r38;
    r76 = r44 * r57;
    r74 = r74 * r14;
    r74 = r74 * r83;
    r74 = fmaf(r76, r52, r76 * r74);
    WriteSum4<float, float>((float*)inout_shared, r65, r78, r74, r73);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           40 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = r55 * r66;
    r78 = r12 * r11;
    r78 = r78 * r8;
    r78 = r78 * r53;
    r78 = r78 * r55;
    r78 = r78 * r71;
    r78 = fmaf(r55, r68, r83 * r78);
    r65 = r10 * r11;
    r65 = r65 * r8;
    r65 = r65 * r13;
    r65 = r65 * r55;
    r65 = r65 * r71;
    r65 = fmaf(r35, r65, r55 * r75);
    WriteSum4<float, float>((float*)inout_shared, r73, r78, r65, r74);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           44 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = r61 * r58;
    r74 = r74 * r83;
    r65 = r58 * r66;
    r78 = r15 * r54;
    r73 = r15 * r30;
    r73 = r73 * r13;
    r73 = r73 * r54;
    r73 = fmaf(r84, r73, r77 * r78);
    WriteSum4<float, float>((float*)inout_shared, r82, r74, r65, r73);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           48 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = r15 * r44;
    r73 = r73 * r54;
    r73 = r73 * r83;
    r65 = r44 * r30;
    r65 = r65 * r84;
    r74 = r12 * r11;
    r74 = r74 * r8;
    r74 = r74 * r53;
    r74 = r74 * r58;
    r74 = r74 * r71;
    r74 = fmaf(r58, r68, r83 * r74);
    r82 = r12 * r11;
    r82 = r82 * r8;
    r82 = r82 * r53;
    r82 = r82 * r57;
    r82 = r82 * r71;
    r68 = fmaf(r57, r68, r83 * r82);
    WriteSum4<float, float>((float*)inout_shared, r74, r68, r73, r65);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           52 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = r0 * r0;
    r65 = r65 * r36;
    r65 = r65 * r44;
    r36 = r15 * r44;
    r36 = r36 * r13;
    r36 = r36 * r54;
    r36 = r36 * r84;
    r73 = r10 * r11;
    r73 = r73 * r8;
    r73 = r73 * r13;
    r73 = r73 * r58;
    r73 = r73 * r71;
    r73 = fmaf(r35, r73, r58 * r75);
    r68 = r10 * r11;
    r68 = r68 * r8;
    r68 = r68 * r13;
    r68 = r68 * r57;
    r68 = r68 * r71;
    r68 = fmaf(r35, r68, r57 * r75);
    WriteSum4<float, float>((float*)inout_shared, r73, r68, r65, r36);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           56 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = r61 * r57;
    r36 = r36 * r83;
    r57 = r57 * r66;
    r65 = r61 * r83;
    r65 = r65 * r76;
    r68 = r12 * r38;
    r81 = r44 * r81;
    r68 = r68 * r14;
    r68 = r68 * r83;
    r81 = fmaf(r52, r81, r81 * r68);
    WriteSum4<float, float>((float*)inout_shared, r81, r36, r57, r65);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           60 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = r76 * r66;
    WriteSum2<float, float>((float*)inout_shared, r66, r79);
  };
  FlushSumShared<2, float>(out_calib_precond_tril,
                           64 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = r26 * r10;
    r66 = r15 * r29;
    r66 = r66 * r10;
    r76 = r15 * r46;
    r76 = r76 * r12;
    r76 = fmaf(r41, r76, r41 * r66);
    r66 = r12 * r12;
    r65 = r28 * r11;
    r66 = r66 * r65;
    r57 = r31 * r10;
    r57 = r57 * r10;
    r76 = fmaf(r65, r57, r76);
    r76 = fmaf(r31, r66, r76);
    r57 = rsqrtf(r32);
    r79 = r79 * r38;
    r79 = r79 * r76;
    r79 = r79 * r41;
    r79 = r79 * r8;
    r79 = r79 * r57;
    r32 = r39 + r32;
    r32 = 1.0 / r32;
    r32 = r57 * r32;
    r39 = r76 * r32;
    r36 = r10 * r54;
    r39 = fmaf(r36, r39, r53 * r79);
    r79 = r31 * r10;
    r79 = r79 * r38;
    r79 = r79 * r33;
    r79 = r79 * r53;
    r39 = fmaf(r65, r79, r39);
    r81 = r29 * r54;
    r39 = fmaf(r71, r81, r39);
    r81 = r14 * r13;
    r79 = r71 * r81;
    r52 = r38 * r33;
    r68 = r38 * r66;
    r52 = r52 * r68;
    r73 = fmaf(r31, r52, r46 * r79);
    r75 = r12 * r32;
    r74 = r76 * r75;
    r73 = fmaf(r81, r74, r73);
    r82 = r26 * r12;
    r82 = r82 * r12;
    r82 = r82 * r38;
    r82 = r82 * r38;
    r82 = r82 * r76;
    r82 = r82 * r41;
    r82 = r82 * r8;
    r73 = fmaf(r57, r82, r73);
    r82 = r39 + r73;
    r74 = r46 * r38;
    r78 = 6.00000000000000000e+00;
    r74 = r74 * r78;
    r74 = r74 * r14;
    r77 = r31 * r38;
    r72 = -6.00000000000000000e+00;
    r77 = r77 * r72;
    r77 = r77 * r33;
    r77 = r77 * r11;
    r70 = r12 * r77;
    r70 = fmaf(r13, r70, r13 * r74);
    r74 = r34 * r76;
    r74 = r74 * r75;
    r70 = fmaf(r81, r74, r70);
    r69 = -3.00000000000000000e+00;
    r69 = r38 * r69;
    r69 = r69 * r41;
    r69 = r69 * r8;
    r69 = r69 * r57;
    r67 = r76 * r69;
    r70 = fmaf(r68, r67, r70);
    r70 = r70 + r39;
    r70 = fmaf(r7, r70, r25 * r82);
    r39 = -5.00000000000000000e-01;
    r67 = r39 * r76;
    r67 = r67 * r33;
    r67 = r67 * r59;
    r67 = r67 * r57;
    r74 = r56 * r67;
    r63 = r76 * r75;
    r40 = 5.00000000000000000e-01;
    r50 = r40 * r62;
    r70 = fmaf(r50, r63, r70);
    r43 = r6 * r46;
    r43 = r43 * r54;
    r70 = fmaf(r71, r43, r70);
    r88 = r26 * r31;
    r88 = r88 * r12;
    r88 = r88 * r38;
    r88 = r88 * r41;
    r70 = fmaf(r60, r88, r70);
    r85 = r6 * r28;
    r85 = r85 * r76;
    r85 = r85 * r41;
    r85 = r85 * r8;
    r85 = r85 * r57;
    r85 = r85 * r53;
    r70 = fmaf(r13, r85, r70);
    r87 = r6 * r76;
    r89 = r15 * r54;
    r89 = r89 * r75;
    r70 = fmaf(r89, r87, r70);
    r90 = r12 * r38;
    r70 = fmaf(r67, r90, r70);
    r91 = r46 * r38;
    r70 = fmaf(r62, r91, r70);
    r92 = r46 * r38;
    r70 = fmaf(r61, r92, r70);
    r93 = r6 * r31;
    r94 = -4.00000000000000000e+00;
    r94 = r94 * r33;
    r94 = r94 * r11;
    r94 = r94 * r53;
    r94 = r94 * r13;
    r70 = fmaf(r94, r93, r70);
    r95 = r6 * r79;
    r96 = r26 * r31;
    r96 = r96 * r12;
    r96 = r96 * r38;
    r96 = r96 * r56;
    r96 = r96 * r41;
    r70 = fmaf(r60, r96, r70);
    r97 = r5 * r15;
    r97 = r97 * r44;
    r97 = fmaf(r82, r97, r4 * r82);
    r86 = r49 * r86;
    r86 = r86 * r58;
    r51 = r51 * r34;
    r51 = r51 * r55;
    r97 = fmaf(r82, r86, r97);
    r97 = fmaf(r82, r51, r97);
    r49 = r97 * r61;
    r70 = fmaf(r13, r49, r70);
    r98 = r40 * r76;
    r98 = r98 * r61;
    r70 = fmaf(r75, r98, r70);
    r70 = fmaf(r74, r13, r70);
    r70 = fmaf(r29, r95, r70);
    r98 = r0 * r70;
    r49 = r10 * r76;
    r49 = r49 * r53;
    r96 = r34 * r76;
    r96 = r96 * r32;
    r96 = fmaf(r36, r96, r69 * r49);
    r49 = r10 * r53;
    r93 = r29 * r38;
    r93 = r93 * r78;
    r96 = fmaf(r54, r93, r96);
    r96 = fmaf(r77, r49, r96);
    r96 = r96 + r73;
    r96 = fmaf(r6, r96, r37 * r82);
    r82 = r10 * r32;
    r82 = r82 * r50;
    r73 = r26 * r31;
    r73 = r73 * r56;
    r73 = r73 * r41;
    r73 = r73 * r60;
    r96 = fmaf(r53, r73, r96);
    r93 = r7 * r46;
    r93 = r93 * r54;
    r96 = fmaf(r71, r93, r96);
    r92 = r7 * r28;
    r92 = r92 * r76;
    r92 = r92 * r41;
    r92 = r92 * r8;
    r92 = r92 * r57;
    r92 = r92 * r53;
    r96 = fmaf(r13, r92, r96);
    r91 = r7 * r76;
    r96 = fmaf(r89, r91, r96);
    r90 = r29 * r38;
    r96 = fmaf(r62, r90, r96);
    r87 = r10 * r40;
    r87 = r87 * r76;
    r87 = r87 * r61;
    r96 = fmaf(r32, r87, r96);
    r85 = r7 * r94;
    r88 = r7 * r29;
    r96 = fmaf(r79, r88, r96);
    r43 = r26 * r31;
    r43 = r43 * r41;
    r43 = r43 * r60;
    r96 = fmaf(r53, r43, r96);
    r63 = r29 * r38;
    r96 = fmaf(r61, r63, r96);
    r99 = r97 * r53;
    r96 = fmaf(r61, r99, r96);
    r96 = fmaf(r76, r82, r96);
    r96 = fmaf(r53, r67, r96);
    r96 = fmaf(r31, r85, r96);
    r96 = fmaf(r53, r74, r96);
    r74 = r1 * r96;
    r99 = r45 * r10;
    r99 = r99 * r10;
    r63 = r15 * r9;
    r63 = r63 * r12;
    r63 = fmaf(r41, r63, r65 * r99);
    r99 = r15 * r48;
    r99 = r99 * r10;
    r63 = fmaf(r41, r99, r63);
    r63 = fmaf(r45, r66, r63);
    r99 = r34 * r63;
    r99 = r99 * r75;
    r43 = r45 * r12;
    r43 = r43 * r12;
    r43 = r43 * r38;
    r43 = r43 * r38;
    r43 = r43 * r72;
    r43 = r43 * r33;
    r43 = fmaf(r11, r43, r81 * r99);
    r99 = r63 * r69;
    r43 = fmaf(r68, r99, r43);
    r88 = r9 * r38;
    r88 = r88 * r78;
    r88 = r88 * r14;
    r43 = fmaf(r13, r88, r43);
    r87 = r26 * r10;
    r87 = r87 * r38;
    r87 = r87 * r63;
    r87 = r87 * r41;
    r87 = r87 * r8;
    r87 = r87 * r57;
    r67 = r48 * r54;
    r67 = fmaf(r71, r67, r53 * r87);
    r87 = r63 * r32;
    r67 = fmaf(r36, r87, r67);
    r90 = r45 * r10;
    r90 = r90 * r38;
    r90 = r90 * r33;
    r90 = r90 * r53;
    r67 = fmaf(r65, r90, r67);
    r43 = r43 + r67;
    r88 = r63 * r75;
    r88 = fmaf(r45, r52, r81 * r88);
    r99 = r26 * r12;
    r99 = r99 * r12;
    r99 = r99 * r38;
    r99 = r99 * r38;
    r99 = r99 * r63;
    r99 = r99 * r41;
    r99 = r99 * r8;
    r88 = fmaf(r57, r99, r88);
    r88 = fmaf(r9, r79, r88);
    r67 = r67 + r88;
    r43 = fmaf(r25, r67, r7 * r43);
    r99 = r12 * r38;
    r99 = r99 * r39;
    r99 = r99 * r63;
    r99 = r99 * r33;
    r99 = r99 * r59;
    r43 = fmaf(r57, r99, r43);
    r90 = r26 * r45;
    r90 = r90 * r12;
    r90 = r90 * r38;
    r90 = r90 * r41;
    r43 = fmaf(r60, r90, r43);
    r87 = r26 * r45;
    r87 = r87 * r12;
    r87 = r87 * r38;
    r87 = r87 * r56;
    r87 = r87 * r41;
    r43 = fmaf(r60, r87, r43);
    r91 = r9 * r38;
    r43 = fmaf(r62, r91, r43);
    r92 = r6 * r28;
    r92 = r92 * r63;
    r92 = r92 * r41;
    r92 = r92 * r8;
    r92 = r92 * r57;
    r92 = r92 * r53;
    r43 = fmaf(r13, r92, r43);
    r93 = r6 * r45;
    r43 = fmaf(r94, r93, r43);
    r73 = r40 * r63;
    r73 = r73 * r61;
    r43 = fmaf(r75, r73, r43);
    r100 = r63 * r75;
    r43 = fmaf(r50, r100, r43);
    r101 = r9 * r38;
    r43 = fmaf(r61, r101, r43);
    r102 = r12 * r38;
    r102 = r102 * r56;
    r102 = r102 * r39;
    r102 = r102 * r63;
    r102 = r102 * r33;
    r102 = r102 * r59;
    r43 = fmaf(r57, r102, r43);
    r103 = r5 * r15;
    r103 = r103 * r44;
    r103 = fmaf(r4, r67, r67 * r103);
    r103 = fmaf(r67, r86, r103);
    r103 = fmaf(r67, r51, r103);
    r104 = r103 * r61;
    r43 = fmaf(r13, r104, r43);
    r105 = r6 * r9;
    r105 = r105 * r54;
    r43 = fmaf(r71, r105, r43);
    r106 = r63 * r89;
    r43 = fmaf(r48, r95, r43);
    r43 = fmaf(r6, r106, r43);
    r105 = r0 * r43;
    r104 = r10 * r63;
    r104 = r104 * r53;
    r102 = r48 * r38;
    r102 = r102 * r78;
    r102 = fmaf(r54, r102, r69 * r104);
    r104 = r34 * r63;
    r104 = r104 * r32;
    r102 = fmaf(r36, r104, r102);
    r101 = r45 * r10;
    r101 = r101 * r38;
    r101 = r101 * r72;
    r101 = r101 * r33;
    r101 = r101 * r11;
    r102 = fmaf(r53, r101, r102);
    r102 = r102 + r88;
    r67 = fmaf(r37, r67, r6 * r102);
    r102 = r7 * r48;
    r67 = fmaf(r79, r102, r67);
    r88 = r39 * r63;
    r88 = r88 * r33;
    r88 = r88 * r59;
    r88 = r88 * r57;
    r67 = fmaf(r53, r88, r67);
    r101 = r48 * r38;
    r67 = fmaf(r62, r101, r67);
    r104 = r48 * r38;
    r67 = fmaf(r61, r104, r67);
    r100 = r10 * r40;
    r100 = r100 * r63;
    r100 = r100 * r61;
    r67 = fmaf(r32, r100, r67);
    r73 = r7 * r28;
    r73 = r73 * r63;
    r73 = r73 * r41;
    r73 = r73 * r8;
    r73 = r73 * r57;
    r73 = r73 * r53;
    r67 = fmaf(r13, r73, r67);
    r93 = r103 * r53;
    r67 = fmaf(r61, r93, r67);
    r92 = r26 * r45;
    r92 = r92 * r56;
    r92 = r92 * r41;
    r92 = r92 * r60;
    r67 = fmaf(r53, r92, r67);
    r91 = r26 * r45;
    r91 = r91 * r41;
    r91 = r91 * r60;
    r67 = fmaf(r53, r91, r67);
    r87 = r56 * r39;
    r87 = r87 * r63;
    r87 = r87 * r33;
    r87 = r87 * r59;
    r87 = r87 * r57;
    r67 = fmaf(r53, r87, r67);
    r90 = r7 * r9;
    r90 = r90 * r54;
    r67 = fmaf(r71, r90, r67);
    r67 = fmaf(r45, r85, r67);
    r67 = fmaf(r63, r82, r67);
    r67 = fmaf(r7, r106, r67);
    r106 = r1 * r67;
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r98,
                                          r74,
                                          r105,
                                          r106);
    r106 = r26 * r10;
    r105 = r15 * r42;
    r105 = r105 * r10;
    r74 = r47 * r10;
    r74 = r74 * r10;
    r74 = fmaf(r65, r74, r41 * r105);
    r105 = r15 * r23;
    r105 = r105 * r12;
    r74 = fmaf(r41, r105, r74);
    r74 = fmaf(r47, r66, r74);
    r106 = r106 * r38;
    r106 = r106 * r74;
    r106 = r106 * r41;
    r106 = r106 * r8;
    r106 = r106 * r57;
    r105 = r42 * r54;
    r105 = fmaf(r71, r105, r53 * r106);
    r106 = r74 * r32;
    r105 = fmaf(r36, r106, r105);
    r66 = r47 * r10;
    r66 = r66 * r38;
    r66 = r66 * r33;
    r66 = r66 * r53;
    r105 = fmaf(r65, r66, r105);
    r66 = r74 * r75;
    r66 = fmaf(r81, r66, r23 * r79);
    r106 = r26 * r12;
    r106 = r106 * r12;
    r106 = r106 * r38;
    r106 = r106 * r38;
    r106 = r106 * r74;
    r106 = r106 * r41;
    r106 = r106 * r8;
    r66 = fmaf(r57, r106, r66);
    r66 = fmaf(r47, r52, r66);
    r106 = r105 + r66;
    r52 = r23 * r38;
    r52 = r52 * r78;
    r52 = r52 * r14;
    r14 = r34 * r74;
    r14 = r14 * r75;
    r14 = fmaf(r81, r14, r13 * r52);
    r52 = r47 * r12;
    r52 = r52 * r12;
    r52 = r52 * r38;
    r52 = r52 * r38;
    r52 = r52 * r72;
    r52 = r52 * r33;
    r14 = fmaf(r11, r52, r14);
    r81 = r74 * r69;
    r14 = fmaf(r68, r81, r14);
    r14 = r14 + r105;
    r14 = fmaf(r7, r14, r25 * r106);
    r25 = r6 * r47;
    r14 = fmaf(r94, r25, r14);
    r94 = r5 * r15;
    r94 = r94 * r44;
    r4 = fmaf(r4, r106, r106 * r94);
    r4 = fmaf(r106, r86, r4);
    r4 = fmaf(r106, r51, r4);
    r51 = r4 * r61;
    r14 = fmaf(r13, r51, r14);
    r86 = r6 * r28;
    r86 = r86 * r74;
    r86 = r86 * r41;
    r86 = r86 * r8;
    r86 = r86 * r57;
    r86 = r86 * r53;
    r14 = fmaf(r13, r86, r14);
    r94 = r6 * r74;
    r14 = fmaf(r89, r94, r14);
    r105 = r26 * r47;
    r105 = r105 * r12;
    r105 = r105 * r38;
    r105 = r105 * r41;
    r14 = fmaf(r60, r105, r14);
    r68 = r12 * r38;
    r68 = r68 * r39;
    r68 = r68 * r74;
    r68 = r68 * r33;
    r68 = r68 * r59;
    r14 = fmaf(r57, r68, r14);
    r52 = r26 * r47;
    r52 = r52 * r12;
    r52 = r52 * r38;
    r52 = r52 * r56;
    r52 = r52 * r41;
    r14 = fmaf(r60, r52, r14);
    r65 = r23 * r38;
    r14 = fmaf(r61, r65, r14);
    r98 = r12 * r38;
    r98 = r98 * r56;
    r98 = r98 * r39;
    r98 = r98 * r74;
    r98 = r98 * r33;
    r98 = r98 * r59;
    r14 = fmaf(r57, r98, r14);
    r90 = r23 * r38;
    r14 = fmaf(r62, r90, r14);
    r87 = r74 * r75;
    r14 = fmaf(r50, r87, r14);
    r50 = r40 * r74;
    r50 = r50 * r61;
    r14 = fmaf(r75, r50, r14);
    r91 = r6 * r23;
    r91 = r91 * r54;
    r14 = fmaf(r71, r91, r14);
    r14 = fmaf(r42, r95, r14);
    r91 = r0 * r14;
    r50 = r42 * r38;
    r50 = r50 * r78;
    r50 = fmaf(r54, r50, r81 * r49);
    r49 = r34 * r74;
    r49 = r49 * r32;
    r50 = fmaf(r36, r49, r50);
    r36 = r47 * r10;
    r36 = r36 * r38;
    r36 = r36 * r72;
    r36 = r36 * r33;
    r36 = r36 * r11;
    r50 = fmaf(r53, r36, r50);
    r50 = r50 + r66;
    r50 = fmaf(r6, r50, r37 * r106);
    r106 = r39 * r74;
    r106 = r106 * r33;
    r106 = r106 * r59;
    r106 = r106 * r57;
    r50 = fmaf(r53, r106, r50);
    r37 = r26 * r47;
    r37 = r37 * r56;
    r37 = r37 * r41;
    r37 = r37 * r60;
    r50 = fmaf(r53, r37, r50);
    r66 = r42 * r38;
    r50 = fmaf(r61, r66, r50);
    r36 = r7 * r28;
    r36 = r36 * r74;
    r36 = r36 * r41;
    r36 = r36 * r8;
    r36 = r36 * r57;
    r36 = r36 * r53;
    r50 = fmaf(r13, r36, r50);
    r8 = r7 * r74;
    r50 = fmaf(r89, r8, r50);
    r89 = r4 * r53;
    r50 = fmaf(r61, r89, r50);
    r49 = r7 * r42;
    r50 = fmaf(r79, r49, r50);
    r79 = r56 * r39;
    r79 = r79 * r74;
    r79 = r79 * r33;
    r79 = r79 * r59;
    r79 = r79 * r57;
    r50 = fmaf(r53, r79, r50);
    r57 = r42 * r38;
    r50 = fmaf(r62, r57, r50);
    r62 = r7 * r23;
    r62 = r62 * r54;
    r50 = fmaf(r71, r62, r50);
    r71 = r26 * r47;
    r71 = r71 * r41;
    r71 = r71 * r60;
    r50 = fmaf(r53, r71, r50);
    r60 = r10 * r40;
    r60 = r60 * r74;
    r60 = r60 * r61;
    r50 = fmaf(r32, r60, r50);
    r50 = fmaf(r74, r82, r50);
    r50 = fmaf(r47, r85, r50);
    r60 = r1 * r50;
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r91,
                                          r60);
    r60 = r1 * r26;
    r60 = r60 * r3;
    r60 = fmaf(r70, r80, r96 * r60);
    r91 = r1 * r26;
    r91 = r91 * r3;
    r91 = fmaf(r43, r80, r67 * r91);
    r71 = r1 * r26;
    r71 = r71 * r3;
    r80 = fmaf(r14, r80, r50 * r71);
    WriteSum3<float, float>((float*)inout_shared, r60, r91, r80);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = r96 * r96;
    r91 = r0 * r0;
    r60 = r70 * r70;
    r60 = fmaf(r60, r91, r84 * r80);
    r80 = r43 * r43;
    r71 = r67 * r67;
    r71 = fmaf(r84, r71, r80 * r91);
    r80 = r50 * r50;
    r3 = r14 * r14;
    r91 = fmaf(r3, r91, r84 * r80);
    WriteSum3<float, float>((float*)inout_shared, r60, r71, r91);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r91 = r96 * r67;
    r71 = r0 * r0;
    r71 = r71 * r70;
    r71 = fmaf(r43, r71, r84 * r91);
    r91 = r96 * r50;
    r60 = r0 * r0;
    r60 = r60 * r70;
    r60 = fmaf(r14, r60, r84 * r91);
    r91 = r67 * r50;
    r70 = r0 * r0;
    r70 = r70 * r43;
    r70 = fmaf(r14, r70, r84 * r91);
    WriteSum3<float, float>((float*)inout_shared, r71, r60, r70);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void ThinPrismFisheyeFixedPoseResJac(
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* out_calib_jac,
    unsigned int out_calib_jac_num_alloc,
    float* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    float* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    float* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
    float* out_point_jac,
    unsigned int out_point_jac_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    float* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float* const out_point_precond_tril,
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