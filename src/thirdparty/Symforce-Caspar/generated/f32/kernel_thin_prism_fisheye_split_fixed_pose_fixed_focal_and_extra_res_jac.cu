#include "kernel_thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraResJacKernel(
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
    r1 = 6.00000000000000000e+00;
    r3 = r48 * r1;
    r3 = r3 * r50;
    r42 = -6.00000000000000000e+00;
    r13 = r15 * r13;
    r13 = 1.0 / r13;
    r15 = r42 * r13;
    r41 = r38 * r38;
    r62 = r47 * r41;
    r15 = r15 * r62;
    r3 = fmaf(r52, r15, r24 * r3);
    r63 = r16 * r29;
    r63 = r63 * r11;
    r64 = r16 * r48;
    r64 = r64 * r10;
    r64 = fmaf(r14, r64, r14 * r63);
    r63 = r52 * r10;
    r65 = r35 * r13;
    r63 = r63 * r10;
    r64 = fmaf(r65, r63, r64);
    r66 = r11 * r11;
    r66 = r66 * r65;
    r64 = fmaf(r52, r66, r64);
    r63 = r51 * r64;
    r67 = rsqrtf(r36);
    r36 = r37 + r36;
    r36 = 1.0 / r36;
    r37 = r36 * r32;
    r63 = r63 * r67;
    r63 = r63 * r37;
    r3 = fmaf(r47, r63, r3);
    r68 = r10 * r10;
    r69 = -3.00000000000000000e+00;
    r39 = r31 * r39;
    r39 = 1.0 / r39;
    r68 = r68 * r69;
    r68 = r68 * r64;
    r68 = r68 * r67;
    r68 = r68 * r39;
    r3 = fmaf(r50, r68, r3);
    r31 = r4 * r11;
    r70 = r39 * r50;
    r71 = r11 * r67;
    r70 = r70 * r71;
    r31 = r31 * r64;
    r72 = r64 * r71;
    r73 = r11 * r33;
    r72 = r72 * r37;
    r72 = fmaf(r73, r72, r70 * r31);
    r41 = r33 * r41;
    r41 = r41 * r66;
    r31 = r54 * r73;
    r72 = fmaf(r52, r41, r72);
    r72 = fmaf(r29, r31, r72);
    r3 = r3 + r72;
    r68 = r48 * r24;
    r63 = r52 * r65;
    r63 = fmaf(r62, r63, r54 * r68);
    r68 = r64 * r67;
    r68 = r68 * r37;
    r63 = fmaf(r47, r68, r63);
    r74 = r4 * r10;
    r74 = r74 * r10;
    r74 = r74 * r64;
    r74 = r74 * r67;
    r74 = r74 * r39;
    r63 = fmaf(r50, r74, r63);
    r72 = r72 + r63;
    r3 = fmaf(r8, r72, r12 * r3);
    r74 = r57 * r67;
    r68 = -5.00000000000000000e-01;
    r75 = r38 * r68;
    r75 = r75 * r64;
    r75 = r75 * r30;
    r74 = r74 * r24;
    r3 = fmaf(r75, r74, r3);
    r76 = r10 * r57;
    r77 = r4 * r52;
    r77 = r77 * r58;
    r77 = r77 * r32;
    r3 = fmaf(r77, r76, r3);
    r78 = r49 * r29;
    r78 = r78 * r24;
    r3 = fmaf(r54, r78, r3);
    r79 = r7 * r16;
    r79 = r79 * r40;
    r79 = fmaf(r6, r72, r72 * r79);
    r80 = 4.00000000000000000e+00;
    r56 = r80 * r56;
    r26 = r26 * r51;
    r26 = r26 * r55;
    r79 = fmaf(r72, r56, r79);
    r79 = fmaf(r72, r26, r79);
    r55 = r10 * r79;
    r3 = fmaf(r60, r55, r3);
    r80 = 5.00000000000000000e-01;
    r81 = r10 * r80;
    r81 = r81 * r67;
    r81 = r81 * r36;
    r81 = r81 * r59;
    r82 = r57 * r81;
    r83 = r67 * r24;
    r3 = fmaf(r75, r83, r3);
    r75 = r49 * r11;
    r84 = -4.00000000000000000e+00;
    r75 = r75 * r38;
    r75 = r75 * r38;
    r75 = r75 * r84;
    r75 = r75 * r13;
    r75 = r75 * r24;
    r85 = r49 * r35;
    r85 = r85 * r10;
    r85 = r85 * r64;
    r3 = fmaf(r70, r85, r3);
    r86 = r49 * r48;
    r3 = fmaf(r31, r86, r3);
    r87 = r49 * r64;
    r88 = r16 * r24;
    r88 = r88 * r71;
    r88 = r88 * r37;
    r3 = fmaf(r88, r87, r3);
    r3 = fmaf(r48, r61, r3);
    r3 = fmaf(r10, r77, r3);
    r3 = fmaf(r64, r82, r3);
    r3 = fmaf(r48, r60, r3);
    r3 = fmaf(r52, r75, r3);
    r3 = fmaf(r64, r81, r3);
    r87 = r0 * r3;
    r86 = r11 * r33;
    r86 = r86 * r38;
    r86 = r86 * r30;
    r86 = r86 * r68;
    r86 = r86 * r64;
    r86 = r86 * r67;
    r85 = r11 * r69;
    r85 = r85 * r64;
    r83 = r51 * r64;
    r83 = r83 * r71;
    r83 = r83 * r37;
    r83 = fmaf(r73, r83, r70 * r85);
    r85 = r52 * r11;
    r85 = r85 * r11;
    r85 = r85 * r38;
    r85 = r85 * r38;
    r85 = r85 * r42;
    r85 = r85 * r33;
    r83 = fmaf(r13, r85, r83);
    r55 = r29 * r11;
    r55 = r55 * r1;
    r55 = r55 * r33;
    r83 = fmaf(r50, r55, r83);
    r83 = r83 + r63;
    r83 = fmaf(r49, r83, r86);
    r63 = r80 * r57;
    r63 = r63 * r64;
    r63 = r63 * r36;
    r63 = r63 * r71;
    r83 = fmaf(r59, r63, r83);
    r55 = r11 * r57;
    r83 = fmaf(r77, r55, r83);
    r85 = r12 * r29;
    r85 = r85 * r24;
    r83 = fmaf(r54, r85, r83);
    r78 = r12 * r52;
    r78 = r78 * r11;
    r78 = r78 * r38;
    r78 = r78 * r38;
    r78 = r78 * r84;
    r78 = r78 * r13;
    r83 = fmaf(r24, r78, r83);
    r76 = r12 * r35;
    r76 = r76 * r10;
    r76 = r76 * r64;
    r83 = fmaf(r70, r76, r83);
    r74 = r12 * r31;
    r89 = r80 * r64;
    r89 = r89 * r36;
    r89 = r89 * r71;
    r83 = fmaf(r59, r89, r83);
    r90 = r12 * r64;
    r83 = fmaf(r88, r90, r83);
    r91 = r11 * r79;
    r83 = fmaf(r60, r91, r83);
    r83 = fmaf(r9, r72, r83);
    r83 = fmaf(r57, r86, r83);
    r83 = fmaf(r48, r74, r83);
    r83 = fmaf(r29, r61, r83);
    r83 = fmaf(r11, r77, r83);
    r83 = fmaf(r29, r60, r83);
    r91 = r5 * r83;
    r90 = r16 * r45;
    r90 = r90 * r10;
    r90 = fmaf(r14, r90, r43 * r66);
    r77 = r16 * r46;
    r77 = r77 * r11;
    r90 = fmaf(r14, r77, r90);
    r89 = r43 * r10;
    r89 = r89 * r10;
    r90 = fmaf(r65, r89, r90);
    r89 = r90 * r67;
    r89 = r89 * r37;
    r77 = r43 * r65;
    r77 = fmaf(r62, r77, r47 * r89);
    r89 = r4 * r10;
    r89 = r89 * r10;
    r89 = r89 * r90;
    r89 = r89 * r67;
    r89 = r89 * r39;
    r77 = fmaf(r50, r89, r77);
    r76 = r45 * r24;
    r77 = fmaf(r54, r76, r77);
    r76 = r4 * r11;
    r76 = r76 * r90;
    r76 = fmaf(r46, r31, r70 * r76);
    r89 = r90 * r71;
    r89 = r89 * r37;
    r76 = fmaf(r73, r89, r76);
    r76 = fmaf(r43, r41, r76);
    r89 = r77 + r76;
    r78 = r51 * r90;
    r78 = r78 * r67;
    r78 = r78 * r37;
    r78 = fmaf(r43, r15, r47 * r78);
    r85 = r10 * r10;
    r55 = r69 * r90;
    r85 = r85 * r67;
    r85 = r85 * r39;
    r85 = r85 * r50;
    r78 = fmaf(r55, r85, r78);
    r86 = r45 * r1;
    r86 = r86 * r50;
    r78 = fmaf(r24, r86, r78);
    r78 = r78 + r76;
    r78 = fmaf(r12, r78, r8 * r89);
    r76 = r49 * r45;
    r78 = fmaf(r31, r76, r78);
    r86 = r49 * r35;
    r86 = r86 * r10;
    r86 = r86 * r90;
    r78 = fmaf(r70, r86, r78);
    r85 = r38 * r68;
    r85 = r85 * r90;
    r85 = r85 * r30;
    r85 = r85 * r67;
    r78 = fmaf(r24, r85, r78);
    r63 = r4 * r43;
    r63 = r63 * r10;
    r63 = r63 * r58;
    r78 = fmaf(r32, r63, r78);
    r72 = r4 * r43;
    r72 = r72 * r10;
    r72 = r72 * r57;
    r72 = r72 * r58;
    r78 = fmaf(r32, r72, r78);
    r92 = r7 * r16;
    r92 = r92 * r40;
    r92 = fmaf(r89, r92, r6 * r89);
    r92 = fmaf(r89, r56, r92);
    r92 = fmaf(r89, r26, r92);
    r93 = r10 * r92;
    r78 = fmaf(r60, r93, r78);
    r94 = r38 * r57;
    r94 = r94 * r68;
    r94 = r94 * r90;
    r94 = r94 * r30;
    r94 = r94 * r67;
    r78 = fmaf(r24, r94, r78);
    r95 = r49 * r46;
    r95 = r95 * r24;
    r78 = fmaf(r54, r95, r78);
    r96 = r49 * r90;
    r78 = fmaf(r88, r96, r78);
    r78 = fmaf(r90, r81, r78);
    r78 = fmaf(r90, r82, r78);
    r78 = fmaf(r45, r61, r78);
    r78 = fmaf(r45, r60, r78);
    r78 = fmaf(r43, r75, r78);
    r96 = r0 * r78;
    r95 = r11 * r70;
    r94 = r46 * r11;
    r94 = r94 * r1;
    r94 = r94 * r33;
    r94 = fmaf(r50, r94, r55 * r95);
    r95 = r51 * r90;
    r95 = r95 * r71;
    r95 = r95 * r37;
    r94 = fmaf(r73, r95, r94);
    r55 = r43 * r11;
    r55 = r55 * r11;
    r55 = r55 * r38;
    r55 = r55 * r38;
    r55 = r55 * r42;
    r55 = r55 * r33;
    r94 = fmaf(r13, r55, r94);
    r94 = r94 + r77;
    r94 = fmaf(r49, r94, r9 * r89);
    r89 = r12 * r90;
    r94 = fmaf(r88, r89, r94);
    r77 = r12 * r35;
    r77 = r77 * r10;
    r77 = r77 * r90;
    r94 = fmaf(r70, r77, r94);
    r55 = r38 * r57;
    r55 = r55 * r68;
    r55 = r55 * r90;
    r55 = r55 * r33;
    r55 = r55 * r30;
    r94 = fmaf(r71, r55, r94);
    r95 = r38 * r68;
    r95 = r95 * r90;
    r95 = r95 * r33;
    r95 = r95 * r30;
    r94 = fmaf(r71, r95, r94);
    r93 = r11 * r92;
    r94 = fmaf(r60, r93, r94);
    r72 = r4 * r43;
    r72 = r72 * r11;
    r72 = r72 * r57;
    r72 = r72 * r58;
    r94 = fmaf(r32, r72, r94);
    r63 = r80 * r90;
    r63 = r63 * r36;
    r63 = r63 * r71;
    r94 = fmaf(r59, r63, r94);
    r85 = r12 * r46;
    r85 = r85 * r24;
    r94 = fmaf(r54, r85, r94);
    r86 = r80 * r57;
    r86 = r86 * r90;
    r86 = r86 * r36;
    r86 = r86 * r71;
    r94 = fmaf(r59, r86, r94);
    r76 = r4 * r43;
    r76 = r76 * r11;
    r76 = r76 * r58;
    r94 = fmaf(r32, r76, r94);
    r97 = r12 * r43;
    r97 = r97 * r11;
    r97 = r97 * r38;
    r97 = r97 * r38;
    r97 = r97 * r84;
    r97 = r97 * r13;
    r94 = fmaf(r24, r97, r94);
    r94 = fmaf(r45, r74, r94);
    r94 = fmaf(r46, r60, r94);
    r94 = fmaf(r46, r61, r94);
    r97 = r5 * r94;
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r87,
                                          r91,
                                          r96,
                                          r97);
    r97 = r4 * r11;
    r96 = r16 * r44;
    r96 = r96 * r11;
    r66 = fmaf(r34, r66, r14 * r96);
    r96 = r34 * r10;
    r96 = r96 * r10;
    r66 = fmaf(r65, r96, r66);
    r91 = r16 * r25;
    r91 = r91 * r10;
    r66 = fmaf(r14, r91, r66);
    r91 = r66 * r70;
    r97 = fmaf(r44, r31, r91 * r97);
    r96 = r66 * r71;
    r96 = r96 * r37;
    r97 = fmaf(r73, r96, r97);
    r97 = fmaf(r34, r41, r97);
    r41 = r25 * r24;
    r96 = r66 * r67;
    r96 = r96 * r37;
    r96 = fmaf(r47, r96, r54 * r41);
    r41 = r34 * r65;
    r96 = fmaf(r62, r41, r96);
    r62 = r4 * r10;
    r62 = r62 * r10;
    r62 = r62 * r66;
    r62 = r62 * r67;
    r62 = r62 * r39;
    r96 = fmaf(r50, r62, r96);
    r62 = r97 + r96;
    r41 = r25 * r1;
    r41 = r41 * r50;
    r14 = r51 * r66;
    r14 = r14 * r67;
    r14 = r14 * r37;
    r14 = fmaf(r47, r14, r24 * r41);
    r41 = r10 * r10;
    r41 = r41 * r69;
    r41 = r41 * r66;
    r41 = r41 * r67;
    r41 = r41 * r39;
    r14 = fmaf(r50, r41, r14);
    r14 = fmaf(r34, r15, r14);
    r14 = r14 + r97;
    r14 = fmaf(r12, r14, r8 * r62);
    r8 = r49 * r44;
    r8 = r8 * r24;
    r14 = fmaf(r54, r8, r14);
    r97 = r49 * r66;
    r14 = fmaf(r88, r97, r14);
    r41 = r38 * r57;
    r41 = r41 * r68;
    r41 = r41 * r66;
    r41 = r41 * r30;
    r41 = r41 * r67;
    r14 = fmaf(r24, r41, r14);
    r15 = r4 * r34;
    r15 = r15 * r10;
    r15 = r15 * r57;
    r15 = r15 * r58;
    r14 = fmaf(r32, r15, r14);
    r39 = r7 * r16;
    r39 = r39 * r40;
    r39 = fmaf(r62, r39, r6 * r62);
    r39 = fmaf(r62, r56, r39);
    r39 = fmaf(r62, r26, r39);
    r26 = r10 * r39;
    r14 = fmaf(r60, r26, r14);
    r56 = r4 * r34;
    r56 = r56 * r10;
    r56 = r56 * r58;
    r14 = fmaf(r32, r56, r14);
    r6 = r38 * r68;
    r6 = r6 * r66;
    r6 = r6 * r30;
    r6 = r6 * r67;
    r14 = fmaf(r24, r6, r14);
    r40 = r35 * r10;
    r40 = r40 * r91;
    r47 = r49 * r25;
    r14 = fmaf(r31, r47, r14);
    r14 = fmaf(r66, r82, r14);
    r14 = fmaf(r25, r60, r14);
    r14 = fmaf(r25, r61, r14);
    r14 = fmaf(r49, r40, r14);
    r14 = fmaf(r66, r81, r14);
    r14 = fmaf(r34, r75, r14);
    r47 = r0 * r14;
    r75 = r11 * r69;
    r81 = r44 * r11;
    r81 = r81 * r1;
    r81 = r81 * r33;
    r81 = fmaf(r50, r81, r91 * r75);
    r75 = r51 * r66;
    r75 = r75 * r71;
    r75 = r75 * r37;
    r81 = fmaf(r73, r75, r81);
    r73 = r34 * r11;
    r73 = r73 * r11;
    r73 = r73 * r38;
    r73 = r73 * r38;
    r73 = r73 * r42;
    r73 = r73 * r33;
    r81 = fmaf(r13, r73, r81);
    r81 = r81 + r96;
    r81 = fmaf(r49, r81, r9 * r62);
    r62 = r12 * r44;
    r62 = r62 * r24;
    r81 = fmaf(r54, r62, r81);
    r54 = r4 * r34;
    r54 = r54 * r11;
    r54 = r54 * r57;
    r54 = r54 * r58;
    r81 = fmaf(r32, r54, r81);
    r9 = r38 * r68;
    r9 = r9 * r66;
    r9 = r9 * r33;
    r9 = r9 * r30;
    r81 = fmaf(r71, r9, r81);
    r96 = r12 * r66;
    r81 = fmaf(r88, r96, r81);
    r88 = r11 * r39;
    r81 = fmaf(r60, r88, r81);
    r73 = r12 * r34;
    r73 = r73 * r11;
    r73 = r73 * r38;
    r73 = r73 * r38;
    r73 = r73 * r84;
    r73 = r73 * r13;
    r81 = fmaf(r24, r73, r81);
    r13 = r38 * r57;
    r13 = r13 * r68;
    r13 = r13 * r66;
    r13 = r13 * r33;
    r13 = r13 * r30;
    r81 = fmaf(r71, r13, r81);
    r30 = r80 * r57;
    r30 = r30 * r66;
    r30 = r30 * r36;
    r30 = r30 * r71;
    r81 = fmaf(r59, r30, r81);
    r33 = r4 * r34;
    r33 = r33 * r11;
    r33 = r33 * r58;
    r81 = fmaf(r32, r33, r81);
    r32 = r80 * r66;
    r32 = r32 * r36;
    r32 = r32 * r71;
    r81 = fmaf(r59, r32, r81);
    r81 = fmaf(r44, r60, r81);
    r81 = fmaf(r12, r40, r81);
    r81 = fmaf(r44, r61, r81);
    r81 = fmaf(r25, r74, r81);
    r32 = r5 * r81;
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r47,
                                          r32);
    r32 = r5 * r4;
    r32 = r32 * r53;
    r47 = r0 * r4;
    r47 = r47 * r2;
    r47 = fmaf(r3, r47, r83 * r32);
    r32 = r0 * r4;
    r32 = r32 * r2;
    r33 = r5 * r4;
    r33 = r33 * r53;
    r33 = fmaf(r94, r33, r78 * r32);
    r32 = r0 * r4;
    r32 = r32 * r2;
    r2 = r5 * r4;
    r2 = r2 * r53;
    r2 = fmaf(r81, r2, r14 * r32);
    WriteSum3<float, float>((float*)inout_shared, r47, r33, r2);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r3 * r3;
    r33 = r0 * r0;
    r47 = r5 * r5;
    r32 = r83 * r47;
    r83 = fmaf(r83, r32, r33 * r2);
    r2 = r94 * r94;
    r53 = r78 * r33;
    r78 = fmaf(r78, r53, r47 * r2);
    r2 = r81 * r81;
    r74 = r14 * r14;
    r74 = fmaf(r33, r74, r47 * r2);
    WriteSum3<float, float>((float*)inout_shared, r83, r78, r74);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fmaf(r3, r53, r94 * r32);
    r78 = r3 * r14;
    r32 = fmaf(r81, r32, r33 * r78);
    r78 = r94 * r81;
    r53 = fmaf(r14, r53, r47 * r78);
    WriteSum3<float, float>((float*)inout_shared, r74, r32, r53);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraResJac(
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
  ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraResJacKernel<<<n_blocks,
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