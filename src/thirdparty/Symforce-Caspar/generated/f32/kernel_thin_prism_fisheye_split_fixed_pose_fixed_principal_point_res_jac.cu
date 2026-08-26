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
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        SharedIndex* focal_and_extra_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* out_focal_and_extra_jac,
        unsigned int out_focal_and_extra_jac_num_alloc,
        float* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        float* const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        float* const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(principal_point,
                                         0 * principal_point_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1);
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
  };
  LoadShared<4, float, float>(focal_and_extra,
                              0 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r0,
                       r5,
                       r6,
                       r7);
  };
  __syncthreads();
  LoadShared<2, float, float>(focal_and_extra,
                              8 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r8,
                       r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = 9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r11,
                                         r12,
                                         r13);
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r14,
                       r15,
                       r16);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r17 = 2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r18,
                                         r19,
                                         r20,
                                         r21);
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r22, r23, r24, r25);
    r26 = fmaf(r18, r23, r21 * r24);
    r27 = r19 * r22;
    r26 = fmaf(r4, r27, r26);
    r26 = fmaf(r20, r25, r26);
    r27 = r17 * r26;
    r28 = fmaf(r18, r25, r21 * r22);
    r29 = r20 * r23;
    r28 = fmaf(r4, r29, r28);
    r28 = fmaf(r19, r24, r28);
    r27 = r27 * r28;
    r29 = -2.00000000000000000e+00;
    r30 = r18 * r24;
    r30 = fmaf(r4, r30, r21 * r23);
    r30 = fmaf(r19, r25, r30);
    r30 = fmaf(r20, r22, r30);
    r31 = r29 * r30;
    r32 = fmaf(r19, r23, r18 * r22);
    r32 = fmaf(r20, r24, r32);
    r32 = fmaf(r4, r32, r21 * r25);
    r31 = fmaf(r32, r31, r27);
    r13 = fmaf(r14, r31, r13);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r25, r33, r34);
    r35 = r18 * r20;
    r35 = r35 * r17;
    r36 = r19 * r21;
    r37 = fmaf(r29, r36, r35);
    r38 = r18 * r18;
    r38 = r29 * r38;
    r39 = 1.00000000000000000e+00;
    r40 = r19 * r19;
    r40 = fmaf(r29, r40, r39);
    r41 = r38 + r40;
    r42 = r19 * r20;
    r42 = r42 * r17;
    r43 = r18 * r21;
    r43 = fmaf(r17, r43, r42);
    r44 = r17 * r26;
    r44 = r44 * r30;
    r45 = r28 * r32;
    r46 = fmaf(r17, r45, r44);
    r47 = r30 * r30;
    r47 = r29 * r47;
    r48 = r39 + r47;
    r49 = r28 * r28;
    r49 = r49 * r29;
    r48 = r48 + r49;
    r13 = fmaf(r25, r37, r13);
    r13 = fmaf(r34, r41, r13);
    r13 = fmaf(r33, r43, r13);
    r13 = fmaf(r15, r46, r13);
    r13 = fmaf(r16, r48, r13);
    r43 = copysign(1.0, r13);
    r43 = fmaf(r10, r43, r13);
    r13 = r43 * r43;
    r41 = 1.0 / r13;
    r47 = r39 + r47;
    r37 = r26 * r26;
    r37 = r29 * r37;
    r47 = r47 + r37;
    r11 = fmaf(r14, r47, r11);
    r28 = r17 * r28;
    r28 = r28 * r30;
    r50 = r26 * r29;
    r50 = fmaf(r32, r50, r28);
    r51 = r17 * r30;
    r51 = fmaf(r32, r51, r27);
    r36 = fmaf(r17, r36, r35);
    r35 = r20 * r21;
    r27 = r18 * r19;
    r27 = r27 * r17;
    r35 = fmaf(r29, r35, r27);
    r52 = r20 * r20;
    r52 = r29 * r52;
    r40 = r52 + r40;
    r11 = fmaf(r15, r50, r11);
    r11 = fmaf(r16, r51, r11);
    r11 = fmaf(r34, r36, r11);
    r11 = fmaf(r33, r35, r11);
    r11 = fmaf(r25, r40, r11);
    r40 = r11 * r11;
    r40 = r41 * r40;
    r35 = r11 * r11;
    r36 = r17 * r26;
    r36 = fmaf(r32, r36, r28);
    r14 = fmaf(r14, r36, r12);
    r12 = r20 * r21;
    r12 = fmaf(r17, r12, r27);
    r52 = r39 + r52;
    r52 = r52 + r38;
    r38 = r18 * r21;
    r38 = fmaf(r29, r38, r42);
    r45 = fmaf(r29, r45, r44);
    r37 = r39 + r37;
    r37 = r37 + r49;
    r14 = fmaf(r25, r12, r14);
    r14 = fmaf(r33, r52, r14);
    r14 = fmaf(r34, r38, r14);
    r14 = fmaf(r16, r45, r14);
    r14 = fmaf(r15, r37, r14);
    r15 = r14 * r14;
    r15 = fmaf(r41, r15, r41 * r35);
    r35 = sqrtf(r15);
    r16 = copysign(1.0, r35);
    r16 = fmaf(r10, r16, r35);
    r10 = r16 * r16;
    r38 = 1.0 / r10;
    r35 = atanf(r35);
    r34 = r35 * r35;
    r40 = r40 * r38;
    r40 = r40 * r34;
    r34 = r14 * r35;
    r52 = r41 * r38;
    r33 = r14 * r35;
    r34 = r34 * r52;
    r34 = r34 * r33;
    r12 = r40 + r34;
  };
  LoadShared<4, float, float>(focal_and_extra,
                              4 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r25,
                       r49,
                       r44,
                       r42);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r27 = 3.00000000000000000e+00;
    r34 = fmaf(r27, r40, r34);
    r28 = fmaf(r49, r34, r8 * r12);
    r32 = r25 * r17;
    r53 = r11 * r35;
    r54 = r53 * r52;
    r32 = r32 * r33;
    r28 = fmaf(r54, r32, r28);
    r55 = r12 * r12;
    r56 = r12 * r55;
    r57 = fmaf(r44, r56, r6 * r12);
    r58 = r55 * r55;
    r57 = fmaf(r42, r58, r57);
    r57 = fmaf(r7, r55, r57);
    r59 = 1.0 / r43;
    r60 = 1.0 / r16;
    r61 = r59 * r60;
    r62 = r57 * r61;
    r28 = fmaf(r53, r62, r28);
    r28 = fmaf(r53, r61, r28);
    r2 = fmaf(r0, r28, r2);
    r32 = r14 * r35;
    r32 = r32 * r27;
    r32 = r32 * r52;
    r32 = fmaf(r33, r32, r40);
    r63 = fmaf(r25, r32, r9 * r12);
    r64 = r49 * r17;
    r64 = r64 * r33;
    r63 = fmaf(r54, r64, r63);
    r63 = fmaf(r33, r62, r63);
    r63 = fmaf(r61, r33, r63);
    r1 = fmaf(r5, r63, r1);
    r1 = fmaf(r3, r4, r1);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r1);
    r3 = r0 * r12;
    r3 = r3 * r53;
    r3 = r3 * r61;
    r64 = r12 * r61;
    r65 = r5 * r33;
    r64 = r64 * r65;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          0 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r28,
                                          r63,
                                          r3,
                                          r64);
    r64 = r5 * r32;
    r3 = r0 * r53;
    r3 = r3 * r61;
    r3 = r3 * r55;
    r66 = r61 * r55;
    r66 = r66 * r65;
    r67 = r0 * r17;
    r67 = r67 * r33;
    r67 = r67 * r54;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          4 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r3,
                                          r66,
                                          r67,
                                          r64);
    r64 = r0 * r34;
    r67 = r17 * r54;
    r67 = r67 * r65;
    r66 = r0 * r53;
    r66 = r66 * r61;
    r66 = r66 * r56;
    r3 = r61 * r56;
    r3 = r3 * r65;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          8 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r64,
                                          r67,
                                          r66,
                                          r3);
    r3 = r0 * r12;
    r66 = r5 * r12;
    r67 = r0 * r53;
    r67 = r67 * r61;
    r67 = r67 * r58;
    r64 = r61 * r65;
    r64 = r64 * r58;
    WriteIdx4<1024, float, float, float4>(
        out_focal_and_extra_jac,
        12 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r67,
        r64,
        r3,
        r66);
    r66 = r4 * r63;
    r66 = r66 * r1;
    r3 = r4 * r2;
    r64 = r28 * r3;
    r67 = r12 * r53;
    r3 = r0 * r3;
    r67 = r67 * r61;
    r68 = r4 * r12;
    r68 = r68 * r1;
    r68 = r68 * r61;
    r68 = fmaf(r65, r68, r3 * r67);
    r67 = r53 * r61;
    r67 = r67 * r55;
    r69 = r4 * r1;
    r69 = r69 * r61;
    r69 = r69 * r55;
    r69 = fmaf(r65, r69, r3 * r67);
    WriteSum4<float, float>((float*)inout_shared, r64, r66, r68, r69);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           0 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = r5 * r4;
    r69 = r69 * r32;
    r68 = r0 * r29;
    r68 = r68 * r2;
    r68 = r68 * r33;
    r68 = fmaf(r54, r68, r1 * r69);
    r69 = r29 * r1;
    r69 = r69 * r54;
    r69 = fmaf(r34, r3, r65 * r69);
    r2 = r53 * r61;
    r2 = r2 * r56;
    r66 = r4 * r1;
    r66 = r66 * r61;
    r66 = r66 * r56;
    r66 = fmaf(r65, r66, r3 * r2);
    r2 = r53 * r61;
    r2 = r2 * r58;
    r64 = r4 * r1;
    r64 = r64 * r61;
    r64 = r64 * r65;
    r64 = fmaf(r58, r64, r3 * r2);
    WriteSum4<float, float>((float*)inout_shared, r68, r69, r66, r64);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           4 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = r5 * r4;
    r64 = r64 * r12;
    r64 = r64 * r1;
    r66 = r12 * r3;
    WriteSum2<float, float>((float*)inout_shared, r66, r64);
  };
  FlushSumShared<2, float>(out_focal_and_extra_njtr,
                           8 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = r28 * r28;
    r66 = r63 * r63;
    r69 = r14 * r35;
    r68 = r5 * r65;
    r69 = r69 * r52;
    r69 = r69 * r55;
    r2 = r0 * r0;
    r40 = r2 * r40;
    r69 = fmaf(r55, r40, r68 * r69);
    r67 = r14 * r35;
    r67 = r67 * r52;
    r67 = r67 * r58;
    r67 = fmaf(r58, r40, r68 * r67);
    WriteSum4<float, float>((float*)inout_shared, r64, r66, r69, r67);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_diag,
                           0 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = r53 * r2;
    r66 = r33 * r69;
    r64 = r11 * r14;
    r70 = 4.00000000000000000e+00;
    r13 = r43 * r13;
    r43 = r43 * r13;
    r43 = 1.0 / r43;
    r10 = r16 * r10;
    r16 = r16 * r10;
    r16 = 1.0 / r16;
    r64 = r64 * r35;
    r64 = r64 * r35;
    r64 = r64 * r70;
    r64 = r64 * r43;
    r64 = r64 * r16;
    r16 = r5 * r5;
    r16 = r16 * r32;
    r16 = fmaf(r32, r16, r64 * r66);
    r66 = r53 * r68;
    r43 = r34 * r34;
    r43 = fmaf(r2, r43, r64 * r66);
    r66 = r14 * r35;
    r64 = r56 * r56;
    r66 = r66 * r52;
    r66 = r66 * r68;
    r66 = fmaf(r64, r40, r64 * r66);
    r71 = r14 * r35;
    r72 = r58 * r58;
    r71 = r71 * r52;
    r71 = r71 * r68;
    r72 = fmaf(r40, r72, r72 * r71);
    WriteSum4<float, float>((float*)inout_shared, r16, r43, r66, r72);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_diag,
                           4 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r55 * r2;
    r43 = r5 * r5;
    r43 = r43 * r55;
    WriteSum2<float, float>((float*)inout_shared, r72, r43);
  };
  FlushSumShared<2, float>(out_focal_and_extra_precond_diag,
                           8 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = 0.00000000000000000e+00;
    r72 = r0 * r12;
    r72 = r72 * r28;
    r72 = r72 * r53;
    r72 = r72 * r61;
    r16 = r0 * r28;
    r16 = r16 * r53;
    r16 = r16 * r61;
    r16 = r16 * r55;
    r71 = r0 * r17;
    r71 = r71 * r28;
    r71 = r71 * r33;
    r71 = r71 * r54;
    WriteSum4<float, float>((float*)inout_shared, r43, r72, r16, r71);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           0 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = r0 * r34;
    r71 = r71 * r28;
    r16 = r0 * r12;
    r16 = r16 * r28;
    r72 = r0 * r28;
    r72 = r72 * r53;
    r72 = r72 * r61;
    r72 = r72 * r56;
    r28 = r0 * r28;
    r28 = r28 * r53;
    r28 = r28 * r61;
    r28 = r28 * r58;
    WriteSum4<float, float>((float*)inout_shared, r71, r72, r28, r16);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           4 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = r5 * r32;
    r16 = r16 * r63;
    r28 = r12 * r63;
    r28 = r28 * r61;
    r28 = r28 * r65;
    r72 = r63 * r61;
    r72 = r72 * r55;
    r72 = r72 * r65;
    WriteSum4<float, float>((float*)inout_shared, r43, r28, r72, r16);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           8 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = r17 * r63;
    r16 = r16 * r54;
    r16 = r16 * r65;
    r72 = r63 * r61;
    r72 = r72 * r56;
    r72 = r72 * r65;
    r28 = r63 * r61;
    r28 = r28 * r65;
    r28 = r28 * r58;
    WriteSum4<float, float>((float*)inout_shared, r16, r72, r28, r43);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           12 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r5 * r12;
    r43 = r43 * r63;
    r63 = r14 * r35;
    r63 = r63 * r52;
    r63 = r63 * r56;
    r63 = fmaf(r56, r40, r68 * r63);
    r28 = r11 * r12;
    r13 = 1.0 / r13;
    r10 = 1.0 / r10;
    r72 = r17 * r35;
    r28 = r28 * r13;
    r28 = r28 * r10;
    r28 = r28 * r33;
    r28 = r28 * r72;
    r16 = r32 * r68;
    r65 = r61 * r16;
    r28 = fmaf(r12, r65, r69 * r28);
    r71 = r61 * r69;
    r73 = r34 * r71;
    r74 = r14 * r12;
    r74 = r74 * r13;
    r74 = r74 * r10;
    r74 = r74 * r53;
    r74 = r74 * r72;
    r74 = fmaf(r68, r74, r12 * r73);
    WriteSum4<float, float>((float*)inout_shared, r43, r63, r28, r74);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           16 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = r55 * r71;
    r28 = r61 * r55;
    r28 = r28 * r68;
    r63 = r14 * r35;
    r43 = r12 * r58;
    r63 = r63 * r52;
    r63 = r63 * r68;
    r63 = fmaf(r43, r40, r43 * r63);
    WriteSum4<float, float>((float*)inout_shared, r67, r63, r74, r28);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           20 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = r11 * r13;
    r28 = r28 * r10;
    r28 = r28 * r33;
    r28 = r28 * r55;
    r28 = r28 * r72;
    r28 = fmaf(r55, r65, r69 * r28);
    r74 = r14 * r13;
    r74 = r74 * r10;
    r74 = r74 * r53;
    r74 = r74 * r55;
    r74 = r74 * r72;
    r74 = fmaf(r68, r74, r55 * r73);
    WriteSum4<float, float>((float*)inout_shared, r28, r74, r63, r66);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           24 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = r56 * r71;
    r63 = r61 * r56;
    r63 = r63 * r68;
    r74 = r17 * r54;
    r28 = r17 * r34;
    r28 = r28 * r33;
    r28 = r28 * r54;
    r28 = fmaf(r2, r28, r16 * r74);
    r74 = r11 * r13;
    r74 = r74 * r10;
    r74 = r74 * r33;
    r74 = r74 * r56;
    r74 = r74 * r72;
    r74 = fmaf(r56, r65, r69 * r74);
    WriteSum4<float, float>((float*)inout_shared, r66, r63, r28, r74);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           28 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = r17 * r12;
    r74 = r74 * r33;
    r74 = r74 * r54;
    r74 = r74 * r2;
    r28 = r5 * r5;
    r28 = r28 * r12;
    r28 = r28 * r32;
    r32 = r11 * r13;
    r32 = r32 * r10;
    r32 = r32 * r33;
    r32 = r32 * r58;
    r32 = r32 * r72;
    r65 = fmaf(r58, r65, r69 * r32);
    r32 = r14 * r13;
    r32 = r32 * r10;
    r32 = r32 * r53;
    r32 = r32 * r56;
    r32 = r32 * r72;
    r32 = fmaf(r68, r32, r56 * r73);
    WriteSum4<float, float>((float*)inout_shared, r65, r74, r28, r32);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           32 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r12 * r34;
    r32 = r32 * r2;
    r28 = r17 * r12;
    r28 = r28 * r54;
    r28 = r28 * r68;
    r74 = r14 * r13;
    r74 = r74 * r10;
    r74 = r74 * r53;
    r74 = r74 * r58;
    r74 = r74 * r72;
    r74 = fmaf(r68, r74, r58 * r73);
    r73 = r14 * r35;
    r64 = r12 * r64;
    r73 = r73 * r52;
    r73 = r73 * r68;
    r64 = fmaf(r40, r64, r64 * r73);
    WriteSum4<float, float>((float*)inout_shared, r74, r32, r28, r64);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           36 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = r58 * r71;
    r58 = r61 * r58;
    r58 = r58 * r68;
    r71 = r43 * r71;
    r28 = r61 * r68;
    r28 = r28 * r43;
    WriteSum4<float, float>((float*)inout_shared, r64, r58, r71, r28);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           40 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = r47 * r35;
    r71 = 6.00000000000000000e+00;
    r28 = r28 * r71;
    r58 = r31 * r35;
    r64 = -6.00000000000000000e+00;
    r58 = r58 * r64;
    r58 = r58 * r38;
    r58 = r58 * r13;
    r43 = r11 * r53;
    r28 = fmaf(r58, r43, r54 * r28);
    r32 = r17 * r36;
    r32 = r32 * r14;
    r74 = r17 * r47;
    r74 = r74 * r11;
    r74 = fmaf(r41, r74, r41 * r32);
    r32 = r31 * r11;
    r40 = r29 * r13;
    r32 = r32 * r11;
    r74 = fmaf(r40, r32, r74);
    r73 = r14 * r14;
    r73 = r73 * r40;
    r74 = fmaf(r31, r73, r74);
    r32 = r27 * r74;
    r65 = rsqrtf(r15);
    r15 = r39 + r15;
    r15 = 1.0 / r15;
    r15 = r65 * r15;
    r39 = r11 * r54;
    r32 = r32 * r15;
    r28 = fmaf(r39, r32, r28);
    r63 = r11 * r74;
    r66 = -3.00000000000000000e+00;
    r66 = r35 * r66;
    r66 = r66 * r41;
    r66 = r66 * r10;
    r66 = r66 * r65;
    r63 = r63 * r53;
    r28 = fmaf(r66, r63, r28);
    r16 = r4 * r14;
    r16 = r16 * r14;
    r16 = r16 * r35;
    r16 = r16 * r35;
    r16 = r16 * r74;
    r16 = r16 * r41;
    r16 = r16 * r10;
    r67 = r14 * r15;
    r75 = r74 * r67;
    r76 = r52 * r33;
    r75 = fmaf(r76, r75, r65 * r16);
    r16 = r35 * r38;
    r77 = r35 * r73;
    r16 = r16 * r77;
    r78 = r72 * r76;
    r75 = fmaf(r31, r16, r75);
    r75 = fmaf(r36, r78, r75);
    r28 = r28 + r75;
    r63 = r47 * r54;
    r32 = r31 * r11;
    r32 = r32 * r35;
    r32 = r32 * r38;
    r32 = r32 * r53;
    r32 = fmaf(r40, r32, r72 * r63);
    r63 = r74 * r15;
    r32 = fmaf(r39, r63, r32);
    r79 = r4 * r11;
    r79 = r79 * r35;
    r79 = r79 * r74;
    r79 = r79 * r41;
    r79 = r79 * r10;
    r79 = r79 * r65;
    r32 = fmaf(r53, r79, r32);
    r75 = r75 + r32;
    r28 = fmaf(r8, r75, r49 * r28);
    r79 = r47 * r35;
    r28 = fmaf(r62, r79, r28);
    r63 = -5.00000000000000000e-01;
    r80 = r63 * r74;
    r80 = r80 * r38;
    r80 = r80 * r59;
    r80 = r80 * r65;
    r81 = r57 * r80;
    r82 = r4 * r31;
    r82 = r82 * r57;
    r82 = r82 * r41;
    r82 = r82 * r60;
    r28 = fmaf(r53, r82, r28);
    r83 = r4 * r31;
    r83 = r83 * r41;
    r83 = r83 * r60;
    r28 = fmaf(r53, r83, r28);
    r84 = r25 * r36;
    r84 = r84 * r54;
    r28 = fmaf(r72, r84, r28);
    r85 = r7 * r17;
    r85 = r85 * r12;
    r85 = fmaf(r6, r75, r75 * r85);
    r70 = r42 * r70;
    r70 = r70 * r56;
    r44 = r44 * r27;
    r44 = r44 * r55;
    r85 = fmaf(r75, r70, r85);
    r85 = fmaf(r75, r44, r85);
    r55 = r85 * r53;
    r28 = fmaf(r61, r55, r28);
    r56 = r11 * r15;
    r42 = 5.00000000000000000e-01;
    r86 = r42 * r62;
    r56 = r56 * r86;
    r87 = r47 * r35;
    r28 = fmaf(r61, r87, r28);
    r88 = r25 * r31;
    r89 = -4.00000000000000000e+00;
    r89 = r89 * r38;
    r89 = r89 * r13;
    r89 = r89 * r53;
    r89 = r89 * r33;
    r28 = fmaf(r89, r88, r28);
    r90 = r25 * r29;
    r90 = r90 * r74;
    r90 = r90 * r41;
    r90 = r90 * r10;
    r90 = r90 * r65;
    r90 = r90 * r53;
    r28 = fmaf(r33, r90, r28);
    r91 = r25 * r78;
    r92 = r25 * r74;
    r93 = r17 * r54;
    r93 = r93 * r67;
    r28 = fmaf(r93, r92, r28);
    r94 = r11 * r42;
    r94 = r94 * r74;
    r94 = r94 * r61;
    r28 = fmaf(r15, r94, r28);
    r28 = fmaf(r53, r81, r28);
    r28 = fmaf(r74, r56, r28);
    r28 = fmaf(r53, r80, r28);
    r28 = fmaf(r47, r91, r28);
    r94 = r0 * r28;
    r92 = r74 * r66;
    r90 = r27 * r74;
    r90 = r90 * r67;
    r90 = fmaf(r76, r90, r77 * r92);
    r92 = r14 * r58;
    r90 = fmaf(r33, r92, r90);
    r88 = r36 * r35;
    r88 = r88 * r71;
    r88 = r88 * r52;
    r90 = fmaf(r33, r88, r90);
    r90 = r90 + r32;
    r75 = fmaf(r9, r75, r25 * r90);
    r90 = r74 * r67;
    r75 = fmaf(r86, r90, r75);
    r32 = r4 * r31;
    r32 = r32 * r14;
    r32 = r32 * r35;
    r32 = r32 * r57;
    r32 = r32 * r41;
    r75 = fmaf(r60, r32, r75);
    r88 = r49 * r36;
    r88 = r88 * r54;
    r75 = fmaf(r72, r88, r75);
    r92 = r49 * r89;
    r87 = r14 * r35;
    r75 = fmaf(r80, r87, r75);
    r80 = r49 * r29;
    r80 = r80 * r74;
    r80 = r80 * r41;
    r80 = r80 * r10;
    r80 = r80 * r65;
    r80 = r80 * r53;
    r75 = fmaf(r33, r80, r75);
    r55 = r49 * r47;
    r75 = fmaf(r78, r55, r75);
    r84 = r42 * r74;
    r84 = r84 * r61;
    r75 = fmaf(r67, r84, r75);
    r83 = r36 * r35;
    r75 = fmaf(r62, r83, r75);
    r82 = r4 * r31;
    r82 = r82 * r14;
    r82 = r82 * r35;
    r82 = r82 * r41;
    r75 = fmaf(r60, r82, r75);
    r79 = r49 * r74;
    r75 = fmaf(r93, r79, r75);
    r95 = r36 * r35;
    r75 = fmaf(r61, r95, r75);
    r96 = r85 * r61;
    r75 = fmaf(r33, r96, r75);
    r75 = fmaf(r81, r33, r75);
    r75 = fmaf(r31, r92, r75);
    r96 = r5 * r75;
    r95 = r17 * r50;
    r95 = r95 * r11;
    r95 = fmaf(r41, r95, r46 * r73);
    r79 = r17 * r37;
    r79 = r79 * r14;
    r95 = fmaf(r41, r79, r95);
    r82 = r46 * r11;
    r82 = r82 * r11;
    r95 = fmaf(r40, r82, r95);
    r82 = r95 * r15;
    r79 = r46 * r11;
    r79 = r79 * r35;
    r79 = r79 * r38;
    r79 = r79 * r53;
    r79 = fmaf(r40, r79, r39 * r82);
    r82 = r4 * r11;
    r82 = r82 * r35;
    r82 = r82 * r95;
    r82 = r82 * r41;
    r82 = r82 * r10;
    r82 = r82 * r65;
    r79 = fmaf(r53, r82, r79);
    r83 = r50 * r54;
    r79 = fmaf(r72, r83, r79);
    r83 = r4 * r14;
    r83 = r83 * r14;
    r83 = r83 * r35;
    r83 = r83 * r35;
    r83 = r83 * r95;
    r83 = r83 * r41;
    r83 = r83 * r10;
    r83 = fmaf(r37, r78, r65 * r83);
    r82 = r95 * r67;
    r83 = fmaf(r76, r82, r83);
    r83 = fmaf(r46, r16, r83);
    r82 = r79 + r83;
    r84 = r27 * r95;
    r84 = r84 * r15;
    r55 = r46 * r11;
    r55 = r55 * r35;
    r55 = r55 * r64;
    r55 = r55 * r38;
    r55 = r55 * r13;
    r55 = fmaf(r53, r55, r39 * r84);
    r84 = r95 * r66;
    r80 = r50 * r35;
    r80 = r80 * r71;
    r55 = fmaf(r54, r80, r55);
    r55 = fmaf(r84, r43, r55);
    r55 = r55 + r83;
    r55 = fmaf(r49, r55, r8 * r82);
    r83 = r25 * r29;
    r83 = r83 * r95;
    r83 = r83 * r41;
    r83 = r83 * r10;
    r83 = r83 * r65;
    r83 = r83 * r53;
    r55 = fmaf(r33, r83, r55);
    r80 = r63 * r95;
    r80 = r80 * r38;
    r80 = r80 * r59;
    r80 = r80 * r65;
    r55 = fmaf(r53, r80, r55);
    r43 = r4 * r46;
    r43 = r43 * r41;
    r43 = r43 * r60;
    r55 = fmaf(r53, r43, r55);
    r87 = r4 * r46;
    r87 = r87 * r57;
    r87 = r87 * r41;
    r87 = r87 * r60;
    r55 = fmaf(r53, r87, r55);
    r88 = r7 * r17;
    r88 = r88 * r12;
    r88 = fmaf(r82, r88, r6 * r82);
    r88 = fmaf(r82, r70, r88);
    r88 = fmaf(r82, r44, r88);
    r32 = r88 * r53;
    r55 = fmaf(r61, r32, r55);
    r81 = r57 * r63;
    r81 = r81 * r95;
    r81 = r81 * r38;
    r81 = r81 * r59;
    r81 = r81 * r65;
    r55 = fmaf(r53, r81, r55);
    r90 = r11 * r42;
    r90 = r90 * r95;
    r90 = r90 * r61;
    r55 = fmaf(r15, r90, r55);
    r97 = r50 * r35;
    r55 = fmaf(r62, r97, r55);
    r98 = r25 * r37;
    r98 = r98 * r54;
    r55 = fmaf(r72, r98, r55);
    r99 = r50 * r35;
    r55 = fmaf(r61, r99, r55);
    r100 = r25 * r46;
    r55 = fmaf(r89, r100, r55);
    r101 = r25 * r95;
    r55 = fmaf(r93, r101, r55);
    r55 = fmaf(r50, r91, r55);
    r55 = fmaf(r95, r56, r55);
    r101 = r0 * r55;
    r100 = r37 * r35;
    r100 = r100 * r71;
    r100 = r100 * r52;
    r100 = fmaf(r33, r100, r77 * r84);
    r84 = r27 * r95;
    r84 = r84 * r67;
    r100 = fmaf(r76, r84, r100);
    r99 = r46 * r14;
    r99 = r99 * r14;
    r99 = r99 * r35;
    r99 = r99 * r35;
    r99 = r99 * r64;
    r99 = r99 * r38;
    r100 = fmaf(r13, r99, r100);
    r100 = r100 + r79;
    r100 = fmaf(r25, r100, r9 * r82);
    r82 = r49 * r50;
    r100 = fmaf(r78, r82, r100);
    r79 = r49 * r95;
    r100 = fmaf(r93, r79, r100);
    r99 = r49 * r29;
    r99 = r99 * r95;
    r99 = r99 * r41;
    r99 = r99 * r10;
    r99 = r99 * r65;
    r99 = r99 * r53;
    r100 = fmaf(r33, r99, r100);
    r84 = r14 * r35;
    r84 = r84 * r57;
    r84 = r84 * r63;
    r84 = r84 * r95;
    r84 = r84 * r38;
    r84 = r84 * r59;
    r100 = fmaf(r65, r84, r100);
    r98 = r14 * r35;
    r98 = r98 * r63;
    r98 = r98 * r95;
    r98 = r98 * r38;
    r98 = r98 * r59;
    r100 = fmaf(r65, r98, r100);
    r97 = r88 * r61;
    r100 = fmaf(r33, r97, r100);
    r90 = r37 * r35;
    r100 = fmaf(r61, r90, r100);
    r81 = r4 * r46;
    r81 = r81 * r14;
    r81 = r81 * r35;
    r81 = r81 * r57;
    r81 = r81 * r41;
    r100 = fmaf(r60, r81, r100);
    r32 = r37 * r35;
    r100 = fmaf(r62, r32, r100);
    r87 = r42 * r95;
    r87 = r87 * r61;
    r100 = fmaf(r67, r87, r100);
    r43 = r49 * r37;
    r43 = r43 * r54;
    r100 = fmaf(r72, r43, r100);
    r80 = r95 * r67;
    r100 = fmaf(r86, r80, r100);
    r83 = r4 * r46;
    r83 = r83 * r14;
    r83 = r83 * r35;
    r83 = r83 * r41;
    r100 = fmaf(r60, r83, r100);
    r100 = fmaf(r46, r92, r100);
    r83 = r5 * r100;
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r94,
                                          r96,
                                          r101,
                                          r83);
    r83 = r4 * r14;
    r101 = r17 * r45;
    r101 = r101 * r14;
    r73 = fmaf(r48, r73, r41 * r101);
    r101 = r48 * r11;
    r101 = r101 * r11;
    r73 = fmaf(r40, r101, r73);
    r96 = r17 * r51;
    r96 = r96 * r11;
    r73 = fmaf(r41, r96, r73);
    r83 = r83 * r14;
    r83 = r83 * r35;
    r83 = r83 * r35;
    r83 = r83 * r73;
    r83 = r83 * r41;
    r83 = r83 * r10;
    r83 = fmaf(r45, r78, r65 * r83);
    r96 = r73 * r67;
    r83 = fmaf(r76, r96, r83);
    r83 = fmaf(r48, r16, r83);
    r16 = r51 * r54;
    r96 = r73 * r15;
    r96 = fmaf(r39, r96, r72 * r16);
    r16 = r48 * r11;
    r16 = r16 * r35;
    r16 = r16 * r38;
    r16 = r16 * r53;
    r96 = fmaf(r40, r16, r96);
    r40 = r4 * r11;
    r40 = r40 * r35;
    r40 = r40 * r73;
    r40 = r40 * r41;
    r40 = r40 * r10;
    r40 = r40 * r65;
    r96 = fmaf(r53, r40, r96);
    r40 = r83 + r96;
    r16 = r51 * r35;
    r16 = r16 * r71;
    r101 = r27 * r73;
    r101 = r101 * r15;
    r101 = fmaf(r39, r101, r54 * r16);
    r16 = r48 * r11;
    r16 = r16 * r35;
    r16 = r16 * r64;
    r16 = r16 * r38;
    r16 = r16 * r13;
    r101 = fmaf(r53, r16, r101);
    r39 = r11 * r73;
    r39 = r39 * r53;
    r101 = fmaf(r66, r39, r101);
    r101 = r101 + r83;
    r101 = fmaf(r49, r101, r8 * r40);
    r8 = r25 * r45;
    r8 = r8 * r54;
    r101 = fmaf(r72, r8, r101);
    r93 = r73 * r93;
    r83 = r57 * r63;
    r83 = r83 * r73;
    r83 = r83 * r38;
    r83 = r83 * r59;
    r83 = r83 * r65;
    r101 = fmaf(r53, r83, r101);
    r39 = r4 * r48;
    r39 = r39 * r57;
    r39 = r39 * r41;
    r39 = r39 * r60;
    r101 = fmaf(r53, r39, r101);
    r16 = r7 * r17;
    r16 = r16 * r12;
    r16 = fmaf(r40, r16, r6 * r40);
    r16 = fmaf(r40, r70, r16);
    r16 = fmaf(r40, r44, r16);
    r44 = r16 * r53;
    r101 = fmaf(r61, r44, r101);
    r70 = r4 * r48;
    r70 = r70 * r41;
    r70 = r70 * r60;
    r101 = fmaf(r53, r70, r101);
    r6 = r63 * r73;
    r6 = r6 * r38;
    r6 = r6 * r59;
    r6 = r6 * r65;
    r101 = fmaf(r53, r6, r101);
    r94 = r51 * r35;
    r101 = fmaf(r61, r94, r101);
    r80 = r51 * r35;
    r101 = fmaf(r62, r80, r101);
    r43 = r25 * r29;
    r43 = r43 * r73;
    r43 = r43 * r41;
    r43 = r43 * r10;
    r43 = r43 * r65;
    r43 = r43 * r53;
    r101 = fmaf(r33, r43, r101);
    r87 = r11 * r42;
    r87 = r87 * r73;
    r87 = r87 * r61;
    r101 = fmaf(r15, r87, r101);
    r32 = r25 * r48;
    r101 = fmaf(r89, r32, r101);
    r101 = fmaf(r25, r93, r101);
    r101 = fmaf(r73, r56, r101);
    r101 = fmaf(r51, r91, r101);
    r91 = r0 * r101;
    r32 = r73 * r66;
    r87 = r45 * r35;
    r87 = r87 * r71;
    r87 = r87 * r52;
    r87 = fmaf(r33, r87, r77 * r32);
    r32 = r27 * r73;
    r32 = r32 * r67;
    r87 = fmaf(r76, r32, r87);
    r76 = r48 * r14;
    r76 = r76 * r14;
    r76 = r76 * r35;
    r76 = r76 * r35;
    r76 = r76 * r64;
    r76 = r76 * r38;
    r87 = fmaf(r13, r76, r87);
    r87 = r87 + r96;
    r87 = fmaf(r25, r87, r9 * r40);
    r40 = r49 * r45;
    r40 = r40 * r54;
    r87 = fmaf(r72, r40, r87);
    r72 = r4 * r48;
    r72 = r72 * r14;
    r72 = r72 * r35;
    r72 = r72 * r57;
    r72 = r72 * r41;
    r87 = fmaf(r60, r72, r87);
    r9 = r14 * r35;
    r9 = r9 * r63;
    r9 = r9 * r73;
    r9 = r9 * r38;
    r9 = r9 * r59;
    r87 = fmaf(r65, r9, r87);
    r96 = r45 * r35;
    r87 = fmaf(r61, r96, r87);
    r76 = r16 * r61;
    r87 = fmaf(r33, r76, r87);
    r32 = r49 * r29;
    r32 = r32 * r73;
    r32 = r32 * r41;
    r32 = r32 * r10;
    r32 = r32 * r65;
    r32 = r32 * r53;
    r87 = fmaf(r33, r32, r87);
    r10 = r14 * r35;
    r10 = r10 * r57;
    r10 = r10 * r63;
    r10 = r10 * r73;
    r10 = r10 * r38;
    r10 = r10 * r59;
    r87 = fmaf(r65, r10, r87);
    r65 = r73 * r67;
    r87 = fmaf(r86, r65, r87);
    r86 = r45 * r35;
    r87 = fmaf(r62, r86, r87);
    r62 = r49 * r51;
    r87 = fmaf(r78, r62, r87);
    r78 = r4 * r48;
    r78 = r78 * r14;
    r78 = r78 * r35;
    r78 = r78 * r41;
    r87 = fmaf(r60, r78, r87);
    r60 = r42 * r73;
    r60 = r60 * r61;
    r87 = fmaf(r67, r60, r87);
    r87 = fmaf(r49, r93, r87);
    r87 = fmaf(r48, r92, r87);
    r60 = r5 * r87;
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r91,
                                          r60);
    r60 = r5 * r4;
    r60 = r60 * r1;
    r60 = fmaf(r28, r3, r75 * r60);
    r91 = r5 * r4;
    r91 = r91 * r1;
    r91 = fmaf(r55, r3, r100 * r91);
    r78 = r5 * r4;
    r78 = r78 * r1;
    r3 = fmaf(r101, r3, r87 * r78);
    WriteSum3<float, float>((float*)inout_shared, r60, r91, r3);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r28 * r28;
    r91 = r5 * r5;
    r60 = r75 * r75;
    r60 = fmaf(r60, r91, r2 * r3);
    r3 = r100 * r100;
    r78 = r55 * r55;
    r78 = fmaf(r2, r78, r3 * r91);
    r3 = r87 * r87;
    r62 = r101 * r101;
    r62 = fmaf(r2, r62, r3 * r91);
    WriteSum3<float, float>((float*)inout_shared, r60, r78, r62);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = r5 * r5;
    r62 = r62 * r75;
    r78 = r28 * r55;
    r78 = fmaf(r2, r78, r100 * r62);
    r62 = r28 * r101;
    r60 = r5 * r5;
    r60 = r60 * r75;
    r60 = fmaf(r87, r60, r2 * r62);
    r62 = r5 * r5;
    r62 = r62 * r100;
    r100 = r55 * r101;
    r100 = fmaf(r2, r100, r87 * r62);
    WriteSum3<float, float>((float*)inout_shared, r78, r60, r100);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void ThinPrismFisheyeSplitFixedPoseFixedPrincipalPointResJac(
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* out_focal_and_extra_jac,
    unsigned int out_focal_and_extra_jac_num_alloc,
    float* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    float* const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    float* const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
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