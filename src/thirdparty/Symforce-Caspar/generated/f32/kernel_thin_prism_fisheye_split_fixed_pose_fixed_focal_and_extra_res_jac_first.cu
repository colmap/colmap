#include "kernel_thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraResJacFirstKernel(
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        float* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        float* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        float* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
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

  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97;
  LoadShared<2, float, float>(principal_point,
                              0 * principal_point_num_alloc,
                              principal_point_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       principal_point_indices_loc[threadIdx.x].target,
                       r0,
                       r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
    ReadIdx4<1024, float, float, float4>(focal_and_extra,
                                         0 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r5,
                                         r6,
                                         r7);
    ReadIdx2<1024, float, float, float2>(focal_and_extra,
                                         8 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r8,
                                         r9);
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r10,
                                         r11,
                                         r12);
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r13,
                       r14,
                       r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = 2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r17,
                                         r18,
                                         r19,
                                         r20);
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r21, r22, r23, r24);
    r25 = fmaf(r17, r24, r20 * r21);
    r26 = r19 * r22;
    r25 = fmaf(r4, r26, r25);
    r25 = fmaf(r18, r23, r25);
    r26 = r16 * r25;
    r27 = r17 * r23;
    r27 = fmaf(r4, r27, r20 * r22);
    r27 = fmaf(r18, r24, r27);
    r27 = fmaf(r19, r21, r27);
    r26 = r26 * r27;
    r28 = fmaf(r17, r22, r20 * r23);
    r29 = r18 * r21;
    r28 = fmaf(r4, r29, r28);
    r28 = fmaf(r19, r24, r28);
    r29 = r16 * r28;
    r30 = fmaf(r18, r22, r17 * r21);
    r30 = fmaf(r19, r23, r30);
    r30 = fmaf(r4, r30, r20 * r24);
    r29 = fmaf(r30, r29, r26);
    r11 = fmaf(r13, r29, r11);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r24, r31, r32);
    r33 = r17 * r18;
    r33 = r33 * r16;
    r34 = r19 * r20;
    r34 = fmaf(r16, r34, r33);
    r35 = -2.00000000000000000e+00;
    r36 = r17 * r17;
    r36 = r35 * r36;
    r37 = 1.00000000000000000e+00;
    r38 = r19 * r19;
    r38 = fmaf(r35, r38, r37);
    r39 = r36 + r38;
    r40 = r18 * r19;
    r40 = r40 * r16;
    r41 = r17 * r20;
    r41 = fmaf(r35, r41, r40);
    r42 = r16 * r28;
    r42 = r42 * r27;
    r43 = r25 * r30;
    r44 = fmaf(r35, r43, r42);
    r45 = r28 * r28;
    r45 = r35 * r45;
    r46 = r37 + r45;
    r47 = r25 * r25;
    r47 = r47 * r35;
    r46 = r46 + r47;
    r11 = fmaf(r24, r34, r11);
    r11 = fmaf(r31, r39, r11);
    r11 = fmaf(r32, r41, r11);
    r11 = fmaf(r15, r44, r11);
    r11 = fmaf(r14, r46, r11);
    r41 = r11 * r11;
    r39 = 9.99999999999999955e-07;
    r34 = r27 * r27;
    r34 = r35 * r34;
    r48 = r37 + r34;
    r48 = r48 + r45;
    r10 = fmaf(r13, r48, r10);
    r45 = r28 * r35;
    r45 = fmaf(r30, r45, r26);
    r26 = r16 * r28;
    r26 = r26 * r25;
    r25 = r16 * r27;
    r25 = fmaf(r30, r25, r26);
    r49 = r17 * r19;
    r49 = r49 * r16;
    r50 = r18 * r20;
    r51 = fmaf(r16, r50, r49);
    r52 = r19 * r20;
    r52 = fmaf(r35, r52, r33);
    r33 = r18 * r18;
    r33 = r33 * r35;
    r38 = r33 + r38;
    r10 = fmaf(r14, r45, r10);
    r10 = fmaf(r15, r25, r10);
    r10 = fmaf(r32, r51, r10);
    r10 = fmaf(r31, r52, r10);
    r10 = fmaf(r24, r38, r10);
    r38 = r10 * r10;
    r52 = r35 * r27;
    r52 = fmaf(r30, r52, r26);
    r13 = fmaf(r13, r52, r12);
    r50 = fmaf(r35, r50, r49);
    r33 = r37 + r33;
    r33 = r33 + r36;
    r36 = r17 * r20;
    r36 = fmaf(r16, r36, r40);
    r43 = fmaf(r16, r43, r42);
    r34 = r37 + r34;
    r34 = r34 + r47;
    r13 = fmaf(r24, r50, r13);
    r13 = fmaf(r32, r33, r13);
    r13 = fmaf(r31, r36, r13);
    r13 = fmaf(r14, r43, r13);
    r13 = fmaf(r15, r34, r13);
    r15 = copysign(1.0, r13);
    r15 = fmaf(r39, r15, r13);
    r13 = r15 * r15;
    r14 = 1.0 / r13;
    r36 = r11 * r11;
    r36 = fmaf(r14, r36, r14 * r38);
    r38 = sqrtf(r36);
    r31 = copysign(1.0, r38);
    r31 = fmaf(r39, r31, r38);
    r39 = r31 * r31;
    r33 = 1.0 / r39;
    r38 = atanf(r38);
    r32 = r38 * r14;
    r50 = r38 * r32;
    r41 = r41 * r33;
    r41 = r41 * r50;
    r24 = r10 * r33;
    r47 = r10 * r24;
    r42 = r50 * r47;
    r40 = r41 + r42;
    ReadIdx4<1024, float, float, float4>(focal_and_extra,
                                         4 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r49,
                                         r12,
                                         r26,
                                         r30);
    r51 = 3.00000000000000000e+00;
    r53 = r51 * r50;
    r53 = fmaf(r47, r53, r41);
    r53 = fmaf(r12, r53, r8 * r40);
    r41 = r49 * r11;
    r54 = r16 * r50;
    r41 = r41 * r24;
    r53 = fmaf(r54, r41, r53);
    r55 = r40 * r40;
    r56 = r40 * r55;
    r57 = fmaf(r26, r56, r6 * r40);
    r56 = r30 * r56;
    r57 = fmaf(r40, r56, r57);
    r57 = fmaf(r7, r55, r57);
    r30 = 1.0 / r15;
    r58 = 1.0 / r31;
    r59 = r30 * r58;
    r60 = r38 * r59;
    r61 = r57 * r60;
    r53 = fmaf(r10, r61, r53);
    r53 = fmaf(r10, r60, r53);
    r2 = fmaf(r0, r53, r2);
    r53 = r11 * r11;
    r53 = r53 * r51;
    r53 = r53 * r33;
    r53 = fmaf(r50, r53, r42);
    r53 = fmaf(r49, r53, r9 * r40);
    r42 = r12 * r11;
    r42 = r42 * r24;
    r53 = fmaf(r54, r42, r53);
    r53 = fmaf(r11, r61, r53);
    r53 = fmaf(r11, r60, r53);
    r53 = fmaf(r5, r53, r1);
    r53 = fmaf(r3, r4, r53);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r53);
    r3 = fmaf(r53, r53, r2 * r2);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r3);
  if (global_thread_idx < problem_size) {
    r3 = r4 * r2;
    r1 = r4 * r53;
    WriteSum2<float, float>((float*)inout_shared, r3, r1);
  };
  FlushSumShared<2, float>(out_principal_point_njtr,
                           0 * out_principal_point_njtr_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<float, float>((float*)inout_shared, r37, r37);
  };
  FlushSumShared<2, float>(out_principal_point_precond_diag,
                           0 * out_principal_point_precond_diag_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r10 * r33;
    r3 = -5.00000000000000000e-01;
    r42 = r16 * r29;
    r42 = r42 * r11;
    r41 = r16 * r48;
    r41 = r41 * r10;
    r41 = fmaf(r14, r41, r14 * r42);
    r42 = r52 * r10;
    r13 = r15 * r13;
    r13 = 1.0 / r13;
    r15 = r35 * r13;
    r42 = r42 * r10;
    r41 = fmaf(r15, r42, r41);
    r62 = r11 * r11;
    r62 = r62 * r15;
    r41 = fmaf(r52, r62, r41);
    r42 = rsqrtf(r36);
    r1 = r1 * r38;
    r1 = r1 * r30;
    r1 = r1 * r3;
    r1 = r1 * r41;
    r1 = r1 * r42;
    r63 = 6.00000000000000000e+00;
    r64 = r48 * r63;
    r64 = r64 * r50;
    r65 = -6.00000000000000000e+00;
    r66 = r65 * r13;
    r67 = r38 * r38;
    r68 = r47 * r67;
    r66 = r66 * r68;
    r64 = fmaf(r52, r66, r24 * r64);
    r69 = r51 * r41;
    r36 = r37 + r36;
    r36 = 1.0 / r36;
    r37 = r36 * r32;
    r69 = r69 * r42;
    r69 = r69 * r37;
    r64 = fmaf(r47, r69, r64);
    r70 = r10 * r10;
    r71 = -3.00000000000000000e+00;
    r39 = r31 * r39;
    r39 = 1.0 / r39;
    r70 = r70 * r71;
    r70 = r70 * r41;
    r70 = r70 * r42;
    r70 = r70 * r39;
    r64 = fmaf(r50, r70, r64);
    r31 = r4 * r11;
    r72 = r39 * r50;
    r73 = r11 * r42;
    r72 = r72 * r73;
    r31 = r31 * r41;
    r74 = r41 * r73;
    r75 = r11 * r33;
    r74 = r74 * r37;
    r74 = fmaf(r75, r74, r72 * r31);
    r67 = r33 * r67;
    r67 = r67 * r62;
    r31 = r54 * r75;
    r74 = fmaf(r52, r67, r74);
    r74 = fmaf(r29, r31, r74);
    r64 = r64 + r74;
    r64 = fmaf(r12, r64, r1);
    r70 = r48 * r24;
    r69 = r52 * r15;
    r69 = fmaf(r68, r69, r54 * r70);
    r70 = r41 * r42;
    r70 = r70 * r37;
    r69 = fmaf(r47, r70, r69);
    r76 = r4 * r10;
    r76 = r76 * r10;
    r76 = r76 * r41;
    r76 = r76 * r42;
    r76 = r76 * r39;
    r69 = fmaf(r50, r76, r69);
    r74 = r74 + r69;
    r76 = r10 * r57;
    r70 = r4 * r52;
    r70 = r70 * r58;
    r70 = r70 * r32;
    r64 = fmaf(r70, r76, r64);
    r77 = r49 * r29;
    r77 = r77 * r24;
    r64 = fmaf(r54, r77, r64);
    r78 = r7 * r16;
    r78 = r78 * r40;
    r78 = fmaf(r6, r74, r74 * r78);
    r79 = 4.00000000000000000e+00;
    r56 = r79 * r56;
    r26 = r26 * r51;
    r26 = r26 * r55;
    r78 = fmaf(r74, r56, r78);
    r78 = fmaf(r74, r26, r78);
    r55 = r10 * r78;
    r64 = fmaf(r60, r55, r64);
    r79 = 5.00000000000000000e-01;
    r80 = r10 * r79;
    r80 = r80 * r42;
    r80 = r80 * r36;
    r80 = r80 * r59;
    r81 = r57 * r80;
    r82 = r49 * r11;
    r83 = -4.00000000000000000e+00;
    r82 = r82 * r38;
    r82 = r82 * r38;
    r82 = r82 * r83;
    r82 = r82 * r13;
    r82 = r82 * r24;
    r84 = r49 * r35;
    r84 = r84 * r10;
    r84 = r84 * r41;
    r64 = fmaf(r72, r84, r64);
    r85 = r49 * r48;
    r64 = fmaf(r31, r85, r64);
    r86 = r49 * r41;
    r87 = r16 * r24;
    r87 = r87 * r73;
    r87 = r87 * r37;
    r64 = fmaf(r87, r86, r64);
    r64 = fmaf(r8, r74, r64);
    r64 = fmaf(r48, r61, r64);
    r64 = fmaf(r57, r1, r64);
    r64 = fmaf(r10, r70, r64);
    r64 = fmaf(r41, r81, r64);
    r64 = fmaf(r48, r60, r64);
    r64 = fmaf(r52, r82, r64);
    r64 = fmaf(r41, r80, r64);
    r86 = r0 * r64;
    r85 = r11 * r71;
    r85 = r85 * r41;
    r84 = r51 * r41;
    r84 = r84 * r73;
    r84 = r84 * r37;
    r84 = fmaf(r75, r84, r72 * r85);
    r85 = r52 * r11;
    r85 = r85 * r11;
    r85 = r85 * r38;
    r85 = r85 * r38;
    r85 = r85 * r65;
    r85 = r85 * r33;
    r84 = fmaf(r13, r85, r84);
    r55 = r29 * r11;
    r55 = r55 * r63;
    r55 = r55 * r33;
    r84 = fmaf(r50, r55, r84);
    r84 = r84 + r69;
    r74 = fmaf(r9, r74, r49 * r84);
    r84 = r79 * r57;
    r84 = r84 * r41;
    r84 = r84 * r36;
    r84 = r84 * r73;
    r74 = fmaf(r59, r84, r74);
    r69 = r57 * r33;
    r55 = r38 * r3;
    r55 = r55 * r41;
    r55 = r55 * r30;
    r69 = r69 * r73;
    r74 = fmaf(r55, r69, r74);
    r85 = r11 * r57;
    r74 = fmaf(r70, r85, r74);
    r77 = r12 * r29;
    r77 = r77 * r24;
    r74 = fmaf(r54, r77, r74);
    r76 = r12 * r52;
    r76 = r76 * r11;
    r76 = r76 * r38;
    r76 = r76 * r38;
    r76 = r76 * r83;
    r76 = r76 * r13;
    r74 = fmaf(r24, r76, r74);
    r1 = r33 * r73;
    r74 = fmaf(r55, r1, r74);
    r55 = r12 * r35;
    r55 = r55 * r10;
    r55 = r55 * r41;
    r74 = fmaf(r72, r55, r74);
    r88 = r12 * r31;
    r89 = r79 * r41;
    r89 = r89 * r36;
    r89 = r89 * r73;
    r74 = fmaf(r59, r89, r74);
    r90 = r12 * r41;
    r74 = fmaf(r87, r90, r74);
    r91 = r11 * r78;
    r74 = fmaf(r60, r91, r74);
    r74 = fmaf(r48, r88, r74);
    r74 = fmaf(r29, r61, r74);
    r74 = fmaf(r11, r70, r74);
    r74 = fmaf(r29, r60, r74);
    r91 = r5 * r74;
    r90 = r16 * r45;
    r90 = r90 * r10;
    r90 = fmaf(r14, r90, r43 * r62);
    r70 = r16 * r46;
    r70 = r70 * r11;
    r90 = fmaf(r14, r70, r90);
    r89 = r43 * r10;
    r89 = r89 * r10;
    r90 = fmaf(r15, r89, r90);
    r89 = r90 * r42;
    r89 = r89 * r37;
    r70 = r43 * r15;
    r70 = fmaf(r68, r70, r47 * r89);
    r89 = r4 * r10;
    r89 = r89 * r10;
    r89 = r89 * r90;
    r89 = r89 * r42;
    r89 = r89 * r39;
    r70 = fmaf(r50, r89, r70);
    r55 = r45 * r24;
    r70 = fmaf(r54, r55, r70);
    r55 = r4 * r11;
    r55 = r55 * r90;
    r55 = fmaf(r46, r31, r72 * r55);
    r89 = r90 * r73;
    r89 = r89 * r37;
    r55 = fmaf(r75, r89, r55);
    r55 = fmaf(r43, r67, r55);
    r89 = r70 + r55;
    r1 = r51 * r90;
    r1 = r1 * r42;
    r1 = r1 * r37;
    r1 = fmaf(r43, r66, r47 * r1);
    r76 = r10 * r10;
    r77 = r71 * r90;
    r76 = r76 * r42;
    r76 = r76 * r39;
    r76 = r76 * r50;
    r1 = fmaf(r77, r76, r1);
    r85 = r45 * r63;
    r85 = r85 * r50;
    r1 = fmaf(r24, r85, r1);
    r1 = r1 + r55;
    r1 = fmaf(r12, r1, r8 * r89);
    r55 = r49 * r45;
    r1 = fmaf(r31, r55, r1);
    r85 = r49 * r35;
    r85 = r85 * r10;
    r85 = r85 * r90;
    r1 = fmaf(r72, r85, r1);
    r76 = r38 * r3;
    r76 = r76 * r90;
    r76 = r76 * r30;
    r76 = r76 * r42;
    r1 = fmaf(r24, r76, r1);
    r69 = r4 * r43;
    r69 = r69 * r10;
    r69 = r69 * r58;
    r1 = fmaf(r32, r69, r1);
    r84 = r4 * r43;
    r84 = r84 * r10;
    r84 = r84 * r57;
    r84 = r84 * r58;
    r1 = fmaf(r32, r84, r1);
    r92 = r7 * r16;
    r92 = r92 * r40;
    r92 = fmaf(r89, r92, r6 * r89);
    r92 = fmaf(r89, r56, r92);
    r92 = fmaf(r89, r26, r92);
    r93 = r10 * r92;
    r1 = fmaf(r60, r93, r1);
    r94 = r38 * r57;
    r94 = r94 * r3;
    r94 = r94 * r90;
    r94 = r94 * r30;
    r94 = r94 * r42;
    r1 = fmaf(r24, r94, r1);
    r95 = r49 * r46;
    r95 = r95 * r24;
    r1 = fmaf(r54, r95, r1);
    r96 = r49 * r90;
    r1 = fmaf(r87, r96, r1);
    r1 = fmaf(r90, r80, r1);
    r1 = fmaf(r90, r81, r1);
    r1 = fmaf(r45, r61, r1);
    r1 = fmaf(r45, r60, r1);
    r1 = fmaf(r43, r82, r1);
    r96 = r0 * r1;
    r95 = r11 * r72;
    r94 = r46 * r11;
    r94 = r94 * r63;
    r94 = r94 * r33;
    r94 = fmaf(r50, r94, r77 * r95);
    r95 = r51 * r90;
    r95 = r95 * r73;
    r95 = r95 * r37;
    r94 = fmaf(r75, r95, r94);
    r77 = r43 * r11;
    r77 = r77 * r11;
    r77 = r77 * r38;
    r77 = r77 * r38;
    r77 = r77 * r65;
    r77 = r77 * r33;
    r94 = fmaf(r13, r77, r94);
    r94 = r94 + r70;
    r94 = fmaf(r49, r94, r9 * r89);
    r89 = r12 * r90;
    r94 = fmaf(r87, r89, r94);
    r70 = r12 * r35;
    r70 = r70 * r10;
    r70 = r70 * r90;
    r94 = fmaf(r72, r70, r94);
    r77 = r38 * r57;
    r77 = r77 * r3;
    r77 = r77 * r90;
    r77 = r77 * r33;
    r77 = r77 * r30;
    r94 = fmaf(r73, r77, r94);
    r95 = r38 * r3;
    r95 = r95 * r90;
    r95 = r95 * r33;
    r95 = r95 * r30;
    r94 = fmaf(r73, r95, r94);
    r93 = r11 * r92;
    r94 = fmaf(r60, r93, r94);
    r84 = r4 * r43;
    r84 = r84 * r11;
    r84 = r84 * r57;
    r84 = r84 * r58;
    r94 = fmaf(r32, r84, r94);
    r69 = r79 * r90;
    r69 = r69 * r36;
    r69 = r69 * r73;
    r94 = fmaf(r59, r69, r94);
    r76 = r12 * r46;
    r76 = r76 * r24;
    r94 = fmaf(r54, r76, r94);
    r85 = r79 * r57;
    r85 = r85 * r90;
    r85 = r85 * r36;
    r85 = r85 * r73;
    r94 = fmaf(r59, r85, r94);
    r55 = r4 * r43;
    r55 = r55 * r11;
    r55 = r55 * r58;
    r94 = fmaf(r32, r55, r94);
    r97 = r12 * r43;
    r97 = r97 * r11;
    r97 = r97 * r38;
    r97 = r97 * r38;
    r97 = r97 * r83;
    r97 = r97 * r13;
    r94 = fmaf(r24, r97, r94);
    r94 = fmaf(r45, r88, r94);
    r94 = fmaf(r46, r60, r94);
    r94 = fmaf(r46, r61, r94);
    r97 = r5 * r94;
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r86,
                                          r91,
                                          r96,
                                          r97);
    r97 = r4 * r11;
    r96 = r16 * r44;
    r96 = r96 * r11;
    r62 = fmaf(r34, r62, r14 * r96);
    r96 = r34 * r10;
    r96 = r96 * r10;
    r62 = fmaf(r15, r96, r62);
    r91 = r16 * r25;
    r91 = r91 * r10;
    r62 = fmaf(r14, r91, r62);
    r91 = r62 * r72;
    r97 = fmaf(r44, r31, r91 * r97);
    r96 = r62 * r73;
    r96 = r96 * r37;
    r97 = fmaf(r75, r96, r97);
    r97 = fmaf(r34, r67, r97);
    r67 = r25 * r24;
    r96 = r62 * r42;
    r96 = r96 * r37;
    r96 = fmaf(r47, r96, r54 * r67);
    r67 = r34 * r15;
    r96 = fmaf(r68, r67, r96);
    r68 = r4 * r10;
    r68 = r68 * r10;
    r68 = r68 * r62;
    r68 = r68 * r42;
    r68 = r68 * r39;
    r96 = fmaf(r50, r68, r96);
    r68 = r97 + r96;
    r67 = r25 * r63;
    r67 = r67 * r50;
    r14 = r51 * r62;
    r14 = r14 * r42;
    r14 = r14 * r37;
    r14 = fmaf(r47, r14, r24 * r67);
    r67 = r10 * r10;
    r67 = r67 * r71;
    r67 = r67 * r62;
    r67 = r67 * r42;
    r67 = r67 * r39;
    r14 = fmaf(r50, r67, r14);
    r14 = fmaf(r34, r66, r14);
    r14 = r14 + r97;
    r14 = fmaf(r12, r14, r8 * r68);
    r8 = r49 * r44;
    r8 = r8 * r24;
    r14 = fmaf(r54, r8, r14);
    r97 = r49 * r62;
    r14 = fmaf(r87, r97, r14);
    r67 = r38 * r57;
    r67 = r67 * r3;
    r67 = r67 * r62;
    r67 = r67 * r30;
    r67 = r67 * r42;
    r14 = fmaf(r24, r67, r14);
    r66 = r4 * r34;
    r66 = r66 * r10;
    r66 = r66 * r57;
    r66 = r66 * r58;
    r14 = fmaf(r32, r66, r14);
    r39 = r7 * r16;
    r39 = r39 * r40;
    r39 = fmaf(r68, r39, r6 * r68);
    r39 = fmaf(r68, r56, r39);
    r39 = fmaf(r68, r26, r39);
    r26 = r10 * r39;
    r14 = fmaf(r60, r26, r14);
    r56 = r4 * r34;
    r56 = r56 * r10;
    r56 = r56 * r58;
    r14 = fmaf(r32, r56, r14);
    r6 = r38 * r3;
    r6 = r6 * r62;
    r6 = r6 * r30;
    r6 = r6 * r42;
    r14 = fmaf(r24, r6, r14);
    r40 = r35 * r10;
    r40 = r40 * r91;
    r47 = r49 * r25;
    r14 = fmaf(r31, r47, r14);
    r14 = fmaf(r62, r81, r14);
    r14 = fmaf(r25, r60, r14);
    r14 = fmaf(r25, r61, r14);
    r14 = fmaf(r49, r40, r14);
    r14 = fmaf(r62, r80, r14);
    r14 = fmaf(r34, r82, r14);
    r47 = r0 * r14;
    r82 = r11 * r71;
    r80 = r44 * r11;
    r80 = r80 * r63;
    r80 = r80 * r33;
    r80 = fmaf(r50, r80, r91 * r82);
    r82 = r51 * r62;
    r82 = r82 * r73;
    r82 = r82 * r37;
    r80 = fmaf(r75, r82, r80);
    r75 = r34 * r11;
    r75 = r75 * r11;
    r75 = r75 * r38;
    r75 = r75 * r38;
    r75 = r75 * r65;
    r75 = r75 * r33;
    r80 = fmaf(r13, r75, r80);
    r80 = r80 + r96;
    r80 = fmaf(r49, r80, r9 * r68);
    r68 = r12 * r44;
    r68 = r68 * r24;
    r80 = fmaf(r54, r68, r80);
    r54 = r4 * r34;
    r54 = r54 * r11;
    r54 = r54 * r57;
    r54 = r54 * r58;
    r80 = fmaf(r32, r54, r80);
    r9 = r38 * r3;
    r9 = r9 * r62;
    r9 = r9 * r33;
    r9 = r9 * r30;
    r80 = fmaf(r73, r9, r80);
    r96 = r12 * r62;
    r80 = fmaf(r87, r96, r80);
    r87 = r11 * r39;
    r80 = fmaf(r60, r87, r80);
    r75 = r12 * r34;
    r75 = r75 * r11;
    r75 = r75 * r38;
    r75 = r75 * r38;
    r75 = r75 * r83;
    r75 = r75 * r13;
    r80 = fmaf(r24, r75, r80);
    r13 = r38 * r57;
    r13 = r13 * r3;
    r13 = r13 * r62;
    r13 = r13 * r33;
    r13 = r13 * r30;
    r80 = fmaf(r73, r13, r80);
    r30 = r79 * r57;
    r30 = r30 * r62;
    r30 = r30 * r36;
    r30 = r30 * r73;
    r80 = fmaf(r59, r30, r80);
    r83 = r4 * r34;
    r83 = r83 * r11;
    r83 = r83 * r58;
    r80 = fmaf(r32, r83, r80);
    r32 = r79 * r62;
    r32 = r32 * r36;
    r32 = r32 * r73;
    r80 = fmaf(r59, r32, r80);
    r80 = fmaf(r44, r60, r80);
    r80 = fmaf(r12, r40, r80);
    r80 = fmaf(r44, r61, r80);
    r80 = fmaf(r25, r88, r80);
    r32 = r5 * r80;
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r47,
                                          r32);
    r32 = r5 * r4;
    r32 = r32 * r53;
    r47 = r0 * r4;
    r47 = r47 * r2;
    r47 = fmaf(r64, r47, r74 * r32);
    r32 = r0 * r4;
    r32 = r32 * r2;
    r83 = r5 * r4;
    r83 = r83 * r53;
    r83 = fmaf(r94, r83, r1 * r32);
    r32 = r0 * r4;
    r32 = r32 * r2;
    r2 = r5 * r4;
    r2 = r2 * r53;
    r2 = fmaf(r80, r2, r14 * r32);
    WriteSum3<float, float>((float*)inout_shared, r47, r83, r2);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r64 * r64;
    r83 = r0 * r0;
    r47 = r5 * r5;
    r32 = r74 * r47;
    r74 = fmaf(r74, r32, r83 * r2);
    r2 = r94 * r94;
    r53 = r1 * r83;
    r1 = fmaf(r1, r53, r47 * r2);
    r2 = r80 * r80;
    r88 = r14 * r14;
    r88 = fmaf(r83, r88, r47 * r2);
    WriteSum3<float, float>((float*)inout_shared, r74, r1, r88);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r88 = fmaf(r64, r53, r94 * r32);
    r1 = r64 * r14;
    r32 = fmaf(r80, r32, r83 * r1);
    r1 = r94 * r80;
    r53 = fmaf(r14, r53, r47 * r1);
    WriteSum3<float, float>((float*)inout_shared, r88, r32, r53);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraResJacFirst(
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    float* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    float* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
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
  ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraResJacFirstKernel<<<n_blocks,
                                                                      1024>>>(
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      pose,
      pose_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_principal_point_jac,
      out_principal_point_jac_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc,
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