#include "kernel_thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointResJacFirstKernel(
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        float* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        float* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

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
    r3 = r5 * r4;
    r1 = r11 * r33;
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
    r42 = -5.00000000000000000e-01;
    r63 = rsqrtf(r36);
    r1 = r1 * r38;
    r1 = r1 * r30;
    r1 = r1 * r41;
    r1 = r1 * r42;
    r1 = r1 * r63;
    r64 = -3.00000000000000000e+00;
    r65 = r11 * r64;
    r39 = r31 * r39;
    r39 = 1.0 / r39;
    r31 = r39 * r50;
    r66 = r11 * r63;
    r31 = r31 * r66;
    r65 = r65 * r41;
    r67 = r51 * r41;
    r36 = r37 + r36;
    r36 = 1.0 / r36;
    r37 = r36 * r32;
    r68 = r11 * r33;
    r67 = r67 * r66;
    r67 = r67 * r37;
    r67 = fmaf(r68, r67, r31 * r65);
    r65 = r52 * r11;
    r69 = -6.00000000000000000e+00;
    r65 = r65 * r11;
    r65 = r65 * r38;
    r65 = r65 * r38;
    r65 = r65 * r69;
    r65 = r65 * r33;
    r67 = fmaf(r13, r65, r67);
    r70 = r29 * r11;
    r71 = 6.00000000000000000e+00;
    r70 = r70 * r71;
    r70 = r70 * r33;
    r67 = fmaf(r50, r70, r67);
    r72 = r48 * r24;
    r73 = r52 * r15;
    r74 = r38 * r38;
    r75 = r47 * r74;
    r73 = fmaf(r75, r73, r54 * r72);
    r72 = r41 * r63;
    r72 = r72 * r37;
    r73 = fmaf(r47, r72, r73);
    r76 = r4 * r10;
    r76 = r76 * r10;
    r76 = r76 * r41;
    r76 = r76 * r63;
    r76 = r76 * r39;
    r73 = fmaf(r50, r76, r73);
    r67 = r67 + r73;
    r67 = fmaf(r49, r67, r1);
    r70 = r4 * r11;
    r70 = r70 * r41;
    r65 = r41 * r66;
    r65 = r65 * r37;
    r65 = fmaf(r68, r65, r31 * r70);
    r74 = r33 * r74;
    r74 = r74 * r62;
    r70 = r54 * r68;
    r65 = fmaf(r52, r74, r65);
    r65 = fmaf(r29, r70, r65);
    r73 = r73 + r65;
    r76 = 5.00000000000000000e-01;
    r72 = r76 * r57;
    r72 = r72 * r41;
    r72 = r72 * r36;
    r72 = r72 * r66;
    r67 = fmaf(r59, r72, r67);
    r77 = r11 * r57;
    r78 = r4 * r52;
    r78 = r78 * r58;
    r78 = r78 * r32;
    r67 = fmaf(r78, r77, r67);
    r79 = r12 * r29;
    r79 = r79 * r24;
    r67 = fmaf(r54, r79, r67);
    r80 = r12 * r11;
    r81 = -4.00000000000000000e+00;
    r80 = r80 * r38;
    r80 = r80 * r38;
    r80 = r80 * r81;
    r80 = r80 * r13;
    r80 = r80 * r24;
    r82 = r12 * r35;
    r82 = r82 * r10;
    r82 = r82 * r41;
    r67 = fmaf(r31, r82, r67);
    r83 = r12 * r48;
    r67 = fmaf(r70, r83, r67);
    r84 = r76 * r41;
    r84 = r84 * r36;
    r84 = r84 * r66;
    r67 = fmaf(r59, r84, r67);
    r85 = r12 * r41;
    r86 = r16 * r24;
    r86 = r86 * r66;
    r86 = r86 * r37;
    r67 = fmaf(r86, r85, r67);
    r87 = r7 * r16;
    r87 = r87 * r40;
    r87 = fmaf(r6, r73, r73 * r87);
    r88 = 4.00000000000000000e+00;
    r56 = r88 * r56;
    r26 = r26 * r51;
    r26 = r26 * r55;
    r87 = fmaf(r73, r56, r87);
    r87 = fmaf(r73, r26, r87);
    r55 = r11 * r87;
    r67 = fmaf(r60, r55, r67);
    r67 = fmaf(r9, r73, r67);
    r67 = fmaf(r57, r1, r67);
    r67 = fmaf(r52, r80, r67);
    r67 = fmaf(r29, r61, r67);
    r67 = fmaf(r11, r78, r67);
    r67 = fmaf(r29, r60, r67);
    r3 = r3 * r53;
    r55 = r0 * r4;
    r85 = r48 * r71;
    r85 = r85 * r50;
    r84 = r69 * r13;
    r84 = r84 * r75;
    r85 = fmaf(r52, r84, r24 * r85);
    r83 = r51 * r41;
    r83 = r83 * r63;
    r83 = r83 * r37;
    r85 = fmaf(r47, r83, r85);
    r82 = r10 * r10;
    r82 = r82 * r64;
    r82 = r82 * r41;
    r82 = r82 * r63;
    r82 = r82 * r39;
    r85 = fmaf(r50, r82, r85);
    r85 = r85 + r65;
    r73 = fmaf(r8, r73, r12 * r85);
    r85 = r57 * r63;
    r65 = r38 * r41;
    r65 = r65 * r42;
    r65 = r65 * r30;
    r85 = r85 * r24;
    r73 = fmaf(r65, r85, r73);
    r82 = r10 * r57;
    r73 = fmaf(r78, r82, r73);
    r83 = r49 * r29;
    r83 = r83 * r24;
    r73 = fmaf(r54, r83, r73);
    r79 = r10 * r87;
    r73 = fmaf(r60, r79, r73);
    r77 = r10 * r76;
    r77 = r77 * r36;
    r77 = r77 * r63;
    r77 = r77 * r59;
    r1 = r57 * r77;
    r72 = r63 * r24;
    r73 = fmaf(r65, r72, r73);
    r65 = r49 * r52;
    r65 = r65 * r11;
    r65 = r65 * r38;
    r65 = r65 * r38;
    r65 = r65 * r81;
    r65 = r65 * r13;
    r73 = fmaf(r24, r65, r73);
    r88 = r49 * r35;
    r88 = r88 * r10;
    r88 = r88 * r41;
    r73 = fmaf(r31, r88, r73);
    r89 = r49 * r70;
    r90 = r49 * r41;
    r73 = fmaf(r86, r90, r73);
    r73 = fmaf(r48, r61, r73);
    r73 = fmaf(r10, r78, r73);
    r73 = fmaf(r41, r1, r73);
    r73 = fmaf(r48, r60, r73);
    r73 = fmaf(r48, r89, r73);
    r73 = fmaf(r41, r77, r73);
    r55 = r55 * r2;
    r55 = fmaf(r73, r55, r67 * r3);
    r3 = r0 * r4;
    r90 = r16 * r45;
    r90 = r90 * r10;
    r90 = fmaf(r14, r90, r43 * r62);
    r88 = r16 * r46;
    r88 = r88 * r11;
    r90 = fmaf(r14, r88, r90);
    r65 = r43 * r10;
    r65 = r65 * r10;
    r90 = fmaf(r15, r65, r90);
    r65 = r90 * r63;
    r65 = r65 * r37;
    r88 = r43 * r15;
    r88 = fmaf(r75, r88, r47 * r65);
    r65 = r4 * r10;
    r65 = r65 * r10;
    r65 = r65 * r90;
    r65 = r65 * r63;
    r65 = r65 * r39;
    r88 = fmaf(r50, r65, r88);
    r72 = r45 * r24;
    r88 = fmaf(r54, r72, r88);
    r72 = r4 * r11;
    r72 = r72 * r90;
    r72 = fmaf(r46, r70, r31 * r72);
    r65 = r90 * r66;
    r65 = r65 * r37;
    r72 = fmaf(r68, r65, r72);
    r72 = fmaf(r43, r74, r72);
    r65 = r88 + r72;
    r79 = r51 * r90;
    r79 = r79 * r63;
    r79 = r79 * r37;
    r79 = fmaf(r43, r84, r47 * r79);
    r83 = r10 * r10;
    r78 = r64 * r90;
    r83 = r83 * r63;
    r83 = r83 * r39;
    r83 = r83 * r50;
    r79 = fmaf(r78, r83, r79);
    r82 = r45 * r71;
    r82 = r82 * r50;
    r79 = fmaf(r24, r82, r79);
    r79 = r79 + r72;
    r79 = fmaf(r12, r79, r8 * r65);
    r72 = r49 * r35;
    r72 = r72 * r10;
    r72 = r72 * r90;
    r79 = fmaf(r31, r72, r79);
    r82 = r38 * r42;
    r82 = r82 * r90;
    r82 = r82 * r30;
    r82 = r82 * r63;
    r79 = fmaf(r24, r82, r79);
    r83 = r4 * r43;
    r83 = r83 * r10;
    r83 = r83 * r58;
    r79 = fmaf(r32, r83, r79);
    r85 = r4 * r43;
    r85 = r85 * r10;
    r85 = r85 * r57;
    r85 = r85 * r58;
    r79 = fmaf(r32, r85, r79);
    r91 = r7 * r16;
    r91 = r91 * r40;
    r91 = fmaf(r65, r91, r6 * r65);
    r91 = fmaf(r65, r56, r91);
    r91 = fmaf(r65, r26, r91);
    r92 = r10 * r91;
    r79 = fmaf(r60, r92, r79);
    r93 = r38 * r57;
    r93 = r93 * r42;
    r93 = r93 * r90;
    r93 = r93 * r30;
    r93 = r93 * r63;
    r79 = fmaf(r24, r93, r79);
    r94 = r49 * r46;
    r94 = r94 * r24;
    r79 = fmaf(r54, r94, r79);
    r95 = r49 * r43;
    r95 = r95 * r11;
    r95 = r95 * r38;
    r95 = r95 * r38;
    r95 = r95 * r81;
    r95 = r95 * r13;
    r79 = fmaf(r24, r95, r79);
    r96 = r49 * r90;
    r79 = fmaf(r86, r96, r79);
    r79 = fmaf(r45, r89, r79);
    r79 = fmaf(r90, r77, r79);
    r79 = fmaf(r90, r1, r79);
    r79 = fmaf(r45, r61, r79);
    r79 = fmaf(r45, r60, r79);
    r3 = r3 * r2;
    r96 = r5 * r4;
    r95 = r11 * r31;
    r94 = r46 * r11;
    r94 = r94 * r71;
    r94 = r94 * r33;
    r94 = fmaf(r50, r94, r78 * r95);
    r95 = r51 * r90;
    r95 = r95 * r66;
    r95 = r95 * r37;
    r94 = fmaf(r68, r95, r94);
    r78 = r43 * r11;
    r78 = r78 * r11;
    r78 = r78 * r38;
    r78 = r78 * r38;
    r78 = r78 * r69;
    r78 = r78 * r33;
    r94 = fmaf(r13, r78, r94);
    r94 = r94 + r88;
    r94 = fmaf(r49, r94, r9 * r65);
    r65 = r12 * r45;
    r94 = fmaf(r70, r65, r94);
    r88 = r12 * r90;
    r94 = fmaf(r86, r88, r94);
    r78 = r12 * r35;
    r78 = r78 * r10;
    r78 = r78 * r90;
    r94 = fmaf(r31, r78, r94);
    r95 = r38 * r57;
    r95 = r95 * r42;
    r95 = r95 * r90;
    r95 = r95 * r33;
    r95 = r95 * r30;
    r94 = fmaf(r66, r95, r94);
    r93 = r38 * r42;
    r93 = r93 * r90;
    r93 = r93 * r33;
    r93 = r93 * r30;
    r94 = fmaf(r66, r93, r94);
    r92 = r11 * r91;
    r94 = fmaf(r60, r92, r94);
    r85 = r4 * r43;
    r85 = r85 * r11;
    r85 = r85 * r57;
    r85 = r85 * r58;
    r94 = fmaf(r32, r85, r94);
    r83 = r76 * r90;
    r83 = r83 * r36;
    r83 = r83 * r66;
    r94 = fmaf(r59, r83, r94);
    r82 = r12 * r46;
    r82 = r82 * r24;
    r94 = fmaf(r54, r82, r94);
    r72 = r76 * r57;
    r72 = r72 * r90;
    r72 = r72 * r36;
    r72 = r72 * r66;
    r94 = fmaf(r59, r72, r94);
    r97 = r4 * r43;
    r97 = r97 * r11;
    r97 = r97 * r58;
    r94 = fmaf(r32, r97, r94);
    r94 = fmaf(r46, r60, r94);
    r94 = fmaf(r46, r61, r94);
    r94 = fmaf(r43, r80, r94);
    r96 = r96 * r53;
    r96 = fmaf(r94, r96, r79 * r3);
    r3 = r0 * r4;
    r97 = r4 * r11;
    r72 = r16 * r44;
    r72 = r72 * r11;
    r62 = fmaf(r34, r62, r14 * r72);
    r72 = r34 * r10;
    r72 = r72 * r10;
    r62 = fmaf(r15, r72, r62);
    r82 = r16 * r25;
    r82 = r82 * r10;
    r62 = fmaf(r14, r82, r62);
    r82 = r62 * r31;
    r97 = fmaf(r44, r70, r82 * r97);
    r72 = r62 * r66;
    r72 = r72 * r37;
    r97 = fmaf(r68, r72, r97);
    r97 = fmaf(r34, r74, r97);
    r74 = r25 * r24;
    r72 = r62 * r63;
    r72 = r72 * r37;
    r72 = fmaf(r47, r72, r54 * r74);
    r74 = r34 * r15;
    r72 = fmaf(r75, r74, r72);
    r75 = r4 * r10;
    r75 = r75 * r10;
    r75 = r75 * r62;
    r75 = r75 * r63;
    r75 = r75 * r39;
    r72 = fmaf(r50, r75, r72);
    r75 = r97 + r72;
    r74 = r25 * r71;
    r74 = r74 * r50;
    r14 = r51 * r62;
    r14 = r14 * r63;
    r14 = r14 * r37;
    r14 = fmaf(r47, r14, r24 * r74);
    r74 = r10 * r10;
    r74 = r74 * r64;
    r74 = r74 * r62;
    r74 = r74 * r63;
    r74 = r74 * r39;
    r14 = fmaf(r50, r74, r14);
    r14 = fmaf(r34, r84, r14);
    r14 = r14 + r97;
    r14 = fmaf(r12, r14, r8 * r75);
    r8 = r49 * r44;
    r8 = r8 * r24;
    r14 = fmaf(r54, r8, r14);
    r97 = r49 * r62;
    r14 = fmaf(r86, r97, r14);
    r74 = r38 * r57;
    r74 = r74 * r42;
    r74 = r74 * r62;
    r74 = r74 * r30;
    r74 = r74 * r63;
    r14 = fmaf(r24, r74, r14);
    r84 = r4 * r34;
    r84 = r84 * r10;
    r84 = r84 * r57;
    r84 = r84 * r58;
    r14 = fmaf(r32, r84, r14);
    r39 = r7 * r16;
    r39 = r39 * r40;
    r39 = fmaf(r75, r39, r6 * r75);
    r39 = fmaf(r75, r56, r39);
    r39 = fmaf(r75, r26, r39);
    r26 = r10 * r39;
    r14 = fmaf(r60, r26, r14);
    r56 = r4 * r34;
    r56 = r56 * r10;
    r56 = r56 * r58;
    r14 = fmaf(r32, r56, r14);
    r6 = r38 * r42;
    r6 = r6 * r62;
    r6 = r6 * r30;
    r6 = r6 * r63;
    r14 = fmaf(r24, r6, r14);
    r40 = r35 * r10;
    r40 = r40 * r82;
    r47 = r49 * r34;
    r47 = r47 * r11;
    r47 = r47 * r38;
    r47 = r47 * r38;
    r47 = r47 * r81;
    r47 = r47 * r13;
    r14 = fmaf(r24, r47, r14);
    r14 = fmaf(r62, r1, r14);
    r14 = fmaf(r25, r60, r14);
    r14 = fmaf(r25, r61, r14);
    r14 = fmaf(r49, r40, r14);
    r14 = fmaf(r62, r77, r14);
    r14 = fmaf(r25, r89, r14);
    r3 = r3 * r2;
    r2 = r5 * r4;
    r89 = r11 * r64;
    r47 = r44 * r11;
    r47 = r47 * r71;
    r47 = r47 * r33;
    r47 = fmaf(r50, r47, r82 * r89);
    r89 = r51 * r62;
    r89 = r89 * r66;
    r89 = r89 * r37;
    r47 = fmaf(r68, r89, r47);
    r68 = r34 * r11;
    r68 = r68 * r11;
    r68 = r68 * r38;
    r68 = r68 * r38;
    r68 = r68 * r69;
    r68 = r68 * r33;
    r47 = fmaf(r13, r68, r47);
    r47 = r47 + r72;
    r47 = fmaf(r49, r47, r9 * r75);
    r75 = r12 * r44;
    r75 = r75 * r24;
    r47 = fmaf(r54, r75, r47);
    r54 = r4 * r34;
    r54 = r54 * r11;
    r54 = r54 * r57;
    r54 = r54 * r58;
    r47 = fmaf(r32, r54, r47);
    r9 = r38 * r42;
    r9 = r9 * r62;
    r9 = r9 * r33;
    r9 = r9 * r30;
    r47 = fmaf(r66, r9, r47);
    r72 = r12 * r62;
    r47 = fmaf(r86, r72, r47);
    r86 = r11 * r39;
    r47 = fmaf(r60, r86, r47);
    r68 = r38 * r57;
    r68 = r68 * r42;
    r68 = r68 * r62;
    r68 = r68 * r33;
    r68 = r68 * r30;
    r47 = fmaf(r66, r68, r47);
    r30 = r76 * r57;
    r30 = r30 * r62;
    r30 = r30 * r36;
    r30 = r30 * r66;
    r47 = fmaf(r59, r30, r47);
    r33 = r12 * r25;
    r47 = fmaf(r70, r33, r47);
    r70 = r4 * r34;
    r70 = r70 * r11;
    r70 = r70 * r58;
    r47 = fmaf(r32, r70, r47);
    r32 = r76 * r62;
    r32 = r32 * r36;
    r32 = r32 * r66;
    r47 = fmaf(r59, r32, r47);
    r47 = fmaf(r44, r60, r47);
    r47 = fmaf(r12, r40, r47);
    r47 = fmaf(r34, r80, r47);
    r47 = fmaf(r44, r61, r47);
    r2 = r2 * r53;
    r2 = fmaf(r47, r2, r14 * r3);
    WriteSum3<float, float>((float*)inout_shared, r55, r96, r2);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r73 * r73;
    r96 = r0 * r0;
    r55 = r67 * r67;
    r3 = r5 * r5;
    r55 = fmaf(r3, r55, r96 * r2);
    r2 = r94 * r94;
    r53 = r79 * r79;
    r53 = fmaf(r96, r53, r3 * r2);
    r2 = r47 * r3;
    r32 = r14 * r96;
    r14 = fmaf(r14, r32, r47 * r2);
    WriteSum3<float, float>((float*)inout_shared, r55, r53, r14);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r14 = r67 * r94;
    r53 = r73 * r79;
    r53 = fmaf(r96, r53, r3 * r14);
    r14 = fmaf(r67, r2, r73 * r32);
    r32 = fmaf(r79, r32, r94 * r2);
    WriteSum3<float, float>((float*)inout_shared, r53, r14, r32);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointResJacFirst(
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
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
  ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraFixedPrincipalPointResJacFirstKernel<<<
      n_blocks,
      1024>>>(sensor_from_rig,
              sensor_from_rig_num_alloc,
              point,
              point_num_alloc,
              point_indices,
              pixel,
              pixel_num_alloc,
              pose,
              pose_num_alloc,
              focal_and_extra,
              focal_and_extra_num_alloc,
              principal_point,
              principal_point_num_alloc,
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