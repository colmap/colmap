#include "kernel_simple_radial_split_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPrincipalPointResJacKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc,
        SharedIndex* focal_and_distortion_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* out_pose_jac,
        unsigned int out_pose_jac_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        float* out_focal_and_distortion_jac,
        unsigned int out_focal_and_distortion_jac_num_alloc,
        float* const out_focal_and_distortion_njtr,
        unsigned int out_focal_and_distortion_njtr_num_alloc,
        float* const out_focal_and_distortion_precond_diag,
        unsigned int out_focal_and_distortion_precond_diag_num_alloc,
        float* const out_focal_and_distortion_precond_tril,
        unsigned int out_focal_and_distortion_precond_tril_num_alloc,
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

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex focal_and_distortion_indices_loc[1024];
  focal_and_distortion_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_distortion_indices[global_thread_idx]
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
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90;

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
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r5,
                                         r6);
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r7,
                       r8,
                       r9);
  };
  __syncthreads();
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r10,
                       r11,
                       r12,
                       r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r14,
                                         r15,
                                         r16,
                                         r17);
    r18 = fmaf(r11, r14, r12 * r17);
    r19 = r10 * r15;
    r18 = fmaf(r4, r19, r18);
    r18 = fmaf(r13, r16, r18);
    r19 = r18 * r18;
    r20 = -2.00000000000000000e+00;
    r19 = r19 * r20;
    r21 = 1.00000000000000000e+00;
    r22 = fmaf(r13, r15, r11 * r17);
    r23 = r10 * r16;
    r24 = r12 * r14;
    r22 = r22 + r23;
    r22 = fmaf(r4, r24, r22);
    r25 = r20 * r22;
    r25 = fmaf(r22, r25, r21);
    r26 = r19 + r25;
    r0 = fmaf(r7, r26, r0);
    r27 = 2.00000000000000000e+00;
    r28 = fmaf(r13, r14, r10 * r17);
    r29 = r11 * r16;
    r28 = fmaf(r4, r29, r28);
    r28 = fmaf(r12, r15, r28);
    r29 = r27 * r28;
    r29 = r29 * r22;
    r30 = r18 * r20;
    r31 = fmaf(r11, r15, r10 * r14);
    r31 = fmaf(r12, r16, r31);
    r31 = fmaf(r4, r31, r13 * r17);
    r30 = fmaf(r31, r30, r29);
    r32 = r27 * r18;
    r32 = r32 * r28;
    r33 = r27 * r31;
    r34 = fmaf(r22, r33, r32);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r35,
                       r36,
                       r37);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r38 = r14 * r16;
    r38 = r38 * r27;
    r39 = r15 * r17;
    r40 = fmaf(r27, r39, r38);
    r41 = r16 * r17;
    r42 = r14 * r15;
    r42 = r42 * r27;
    r41 = fmaf(r20, r41, r42);
    r43 = r15 * r15;
    r43 = r43 * r20;
    r44 = r21 + r43;
    r45 = r16 * r16;
    r45 = r45 * r20;
    r44 = r44 + r45;
    r0 = fmaf(r8, r30, r0);
    r0 = fmaf(r9, r34, r0);
    r0 = fmaf(r37, r40, r0);
    r0 = fmaf(r36, r41, r0);
    r0 = fmaf(r35, r44, r0);
  };
  LoadShared<2, float, float>(focal_and_distortion,
                              0 * focal_and_distortion_num_alloc,
                              focal_and_distortion_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       focal_and_distortion_indices_loc[threadIdx.x].target,
                       r46,
                       r47);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r48 = 9.99999999999999955e-07;
    r49 = r20 * r22;
    r49 = fmaf(r31, r49, r32);
    r6 = fmaf(r7, r49, r6);
    r39 = fmaf(r20, r39, r38);
    r43 = r21 + r43;
    r38 = r14 * r14;
    r38 = r38 * r20;
    r43 = r43 + r38;
    r32 = r15 * r16;
    r32 = r32 * r27;
    r50 = r14 * r17;
    r50 = fmaf(r27, r50, r32);
    r51 = r27 * r18;
    r51 = r51 * r22;
    r52 = fmaf(r28, r33, r51);
    r53 = r28 * r28;
    r53 = r53 * r20;
    r25 = r53 + r25;
    r6 = fmaf(r35, r39, r6);
    r6 = fmaf(r37, r43, r6);
    r6 = fmaf(r36, r50, r6);
    r6 = fmaf(r8, r52, r6);
    r6 = fmaf(r9, r25, r6);
    r54 = copysign(1.0, r6);
    r54 = fmaf(r48, r54, r6);
    r48 = r54 * r54;
    r6 = 1.0 / r48;
    r55 = r0 * r6;
    r29 = fmaf(r18, r33, r29);
    r5 = fmaf(r7, r29, r5);
    r56 = r16 * r17;
    r56 = fmaf(r27, r56, r42);
    r45 = r21 + r45;
    r45 = r45 + r38;
    r38 = r14 * r17;
    r38 = fmaf(r20, r38, r32);
    r32 = r28 * r20;
    r32 = fmaf(r31, r32, r51);
    r19 = r21 + r19;
    r19 = r19 + r53;
    r5 = fmaf(r35, r56, r5);
    r5 = fmaf(r36, r45, r5);
    r5 = fmaf(r37, r38, r5);
    r5 = fmaf(r9, r32, r5);
    r5 = fmaf(r8, r19, r5);
    r37 = r5 * r5;
    r36 = fmaf(r6, r37, r0 * r55);
    r21 = fmaf(r47, r36, r21);
    r35 = r0 * r21;
    r53 = 1.0 / r54;
    r51 = r46 * r53;
    r2 = fmaf(r51, r35, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r5 * r21;
    r3 = fmaf(r51, r1, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = r10 * r17;
    r35 = -5.00000000000000000e-01;
    r42 = r13 * r14;
    r42 = fmaf(r35, r42, r35 * r1);
    r1 = r12 * r15;
    r42 = fmaf(r35, r1, r42);
    r57 = r11 * r16;
    r58 = 5.00000000000000000e-01;
    r42 = fmaf(r58, r57, r42);
    r57 = r22 * r42;
    r1 = r12 * r17;
    r59 = r11 * r14;
    r59 = fmaf(r58, r59, r58 * r1);
    r1 = r10 * r15;
    r59 = fmaf(r35, r1, r59);
    r60 = r13 * r58;
    r59 = fmaf(r16, r60, r59);
    r1 = fmaf(r59, r33, r27 * r57);
    r61 = r27 * r28;
    r62 = r13 * r15;
    r63 = r11 * r35;
    r62 = fmaf(r17, r63, r35 * r62);
    r62 = fmaf(r58, r24, r62);
    r62 = fmaf(r35, r23, r62);
    r64 = r27 * r18;
    r65 = r10 * r14;
    r66 = r12 * r16;
    r66 = fmaf(r35, r66, r35 * r65);
    r66 = fmaf(r17, r60, r66);
    r66 = fmaf(r15, r63, r66);
    r64 = r64 * r66;
    r61 = fmaf(r62, r61, r64);
    r1 = r1 + r61;
    r65 = r27 * r22;
    r65 = r65 * r66;
    r67 = r27 * r28;
    r67 = r67 * r59;
    r68 = r65 + r67;
    r69 = r18 * r20;
    r68 = fmaf(r42, r69, r68);
    r70 = r20 * r31;
    r68 = fmaf(r62, r70, r68);
    r68 = fmaf(r8, r68, r9 * r1);
    r1 = r22 * r59;
    r70 = -4.00000000000000000e+00;
    r1 = r1 * r70;
    r69 = r18 * r62;
    r71 = r70 * r69;
    r72 = r1 + r71;
    r68 = fmaf(r7, r72, r68);
    r72 = r21 * r68;
    r73 = r27 * r22;
    r73 = r73 * r62;
    r74 = r27 * r18;
    r74 = fmaf(r59, r74, r73);
    r75 = r27 * r28;
    r75 = r75 * r42;
    r76 = r66 * r33;
    r77 = r75 + r76;
    r78 = r74 + r77;
    r79 = r20 * r31;
    r79 = fmaf(r20, r57, r59 * r79);
    r79 = r79 + r61;
    r79 = fmaf(r7, r79, r8 * r78);
    r78 = r28 * r70;
    r59 = r66 * r78;
    r1 = r1 + r59;
    r79 = fmaf(r9, r1, r79);
    r1 = r79 * r55;
    r80 = r4 * r21;
    r81 = r46 * r80;
    r1 = fmaf(r81, r1, r51 * r72);
    r72 = r27 * r5;
    r82 = r28 * r20;
    r83 = r20 * r31;
    r83 = r83 * r66;
    r82 = fmaf(r42, r82, r83);
    r82 = r82 + r74;
    r59 = r71 + r59;
    r59 = fmaf(r8, r59, r9 * r82);
    r67 = fmaf(r62, r33, r67);
    r82 = r27 * r18;
    r82 = fmaf(r42, r82, r65);
    r67 = r67 + r82;
    r59 = fmaf(r7, r67, r59);
    r72 = r72 * r59;
    r67 = r27 * r68;
    r67 = fmaf(r55, r67, r6 * r72);
    r48 = r54 * r48;
    r48 = 1.0 / r48;
    r48 = r20 * r48;
    r54 = r79 * r48;
    r67 = fmaf(r37, r54, r67);
    r72 = r0 * r0;
    r72 = r72 * r48;
    r67 = fmaf(r79, r72, r67);
    r47 = r47 * r51;
    r67 = r67 * r47;
    r1 = fmaf(r0, r67, r1);
    r54 = r21 * r59;
    r65 = r5 * r6;
    r65 = r65 * r81;
    r54 = fmaf(r79, r65, r51 * r54);
    r54 = fmaf(r5, r67, r54);
    r67 = r20 * r22;
    r67 = fmaf(r62, r67, r83);
    r71 = r27 * r18;
    r74 = r12 * r17;
    r84 = r10 * r15;
    r84 = fmaf(r58, r84, r35 * r74);
    r74 = r13 * r16;
    r84 = fmaf(r35, r74, r84);
    r84 = fmaf(r14, r63, r84);
    r71 = r71 * r84;
    r74 = r27 * r28;
    r85 = r10 * r17;
    r86 = r12 * r15;
    r86 = fmaf(r58, r86, r58 * r85);
    r86 = fmaf(r14, r60, r86);
    r86 = fmaf(r16, r63, r86);
    r74 = fmaf(r86, r74, r71);
    r67 = r67 + r74;
    r63 = r27 * r22;
    r63 = r63 * r86;
    r85 = fmaf(r84, r33, r63);
    r85 = r85 + r61;
    r85 = fmaf(r8, r85, r7 * r67);
    r67 = r22 * r66;
    r67 = r67 * r70;
    r61 = r84 * r78;
    r87 = r67 + r61;
    r85 = fmaf(r9, r87, r85);
    r87 = r85 * r55;
    r76 = r73 + r76;
    r76 = r76 + r74;
    r74 = r18 * r70;
    r74 = r74 * r86;
    r67 = r67 + r74;
    r67 = fmaf(r7, r67, r9 * r76);
    r76 = r20 * r31;
    r76 = fmaf(r20, r69, r86 * r76);
    r73 = r27 * r28;
    r73 = r73 * r66;
    r88 = r27 * r22;
    r88 = fmaf(r84, r88, r73);
    r76 = r76 + r88;
    r67 = fmaf(r8, r76, r67);
    r76 = r21 * r67;
    r76 = fmaf(r51, r76, r81 * r87);
    r87 = r27 * r67;
    r87 = fmaf(r85, r72, r55 * r87);
    r89 = r85 * r48;
    r87 = fmaf(r37, r89, r87);
    r90 = r27 * r5;
    r63 = r64 + r63;
    r64 = r28 * r20;
    r63 = fmaf(r62, r64, r63);
    r62 = r20 * r31;
    r63 = fmaf(r84, r62, r63);
    r86 = fmaf(r86, r33, r27 * r69);
    r86 = r86 + r88;
    r86 = fmaf(r7, r86, r9 * r63);
    r61 = r74 + r61;
    r86 = fmaf(r8, r61, r86);
    r90 = r90 * r86;
    r87 = fmaf(r6, r90, r87);
    r90 = r0 * r87;
    r76 = fmaf(r47, r90, r76);
    r90 = r21 * r86;
    r89 = r5 * r87;
    r89 = fmaf(r47, r89, r51 * r90);
    r89 = fmaf(r85, r65, r89);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r1,
                                          r54,
                                          r76,
                                          r89);
    r90 = r11 * r17;
    r24 = fmaf(r35, r24, r58 * r90);
    r24 = fmaf(r15, r60, r24);
    r24 = fmaf(r58, r23, r24);
    r78 = r24 * r78;
    r57 = r70 * r57;
    r23 = r78 + r57;
    r58 = r27 * r18;
    r58 = r58 * r24;
    r73 = r73 + r58;
    r60 = r20 * r22;
    r73 = fmaf(r84, r60, r73);
    r35 = r20 * r31;
    r73 = fmaf(r42, r35, r73);
    r73 = fmaf(r7, r73, r9 * r23);
    r23 = r27 * r28;
    r23 = fmaf(r24, r33, r84 * r23);
    r23 = r23 + r82;
    r73 = fmaf(r8, r23, r73);
    r23 = r73 * r55;
    r83 = r75 + r83;
    r75 = r27 * r22;
    r75 = r75 * r24;
    r35 = r18 * r20;
    r83 = fmaf(r84, r35, r83);
    r83 = r83 + r75;
    r66 = r18 * r66;
    r66 = r66 * r70;
    r57 = r66 + r57;
    r57 = fmaf(r7, r57, r8 * r83);
    r33 = fmaf(r42, r33, r58);
    r33 = r33 + r88;
    r57 = fmaf(r9, r33, r57);
    r33 = r21 * r57;
    r33 = fmaf(r51, r33, r81 * r23);
    r23 = r27 * r57;
    r23 = fmaf(r55, r23, r73 * r72);
    r88 = r73 * r48;
    r23 = fmaf(r37, r88, r23);
    r42 = r27 * r5;
    r75 = r71 + r75;
    r75 = r75 + r77;
    r77 = r28 * r20;
    r71 = r20 * r31;
    r71 = fmaf(r24, r71, r84 * r77);
    r71 = r71 + r82;
    r71 = fmaf(r9, r71, r7 * r75);
    r78 = r66 + r78;
    r71 = fmaf(r8, r78, r71);
    r42 = r42 * r71;
    r23 = fmaf(r6, r42, r23);
    r42 = r0 * r23;
    r33 = fmaf(r47, r42, r33);
    r42 = r5 * r23;
    r42 = fmaf(r73, r65, r47 * r42);
    r88 = r21 * r71;
    r42 = fmaf(r51, r88, r42);
    r88 = r39 * r48;
    r78 = r27 * r56;
    r78 = r78 * r5;
    r78 = fmaf(r6, r78, r37 * r88);
    r88 = r27 * r44;
    r78 = fmaf(r55, r88, r78);
    r78 = fmaf(r39, r72, r78);
    r88 = r0 * r78;
    r8 = r44 * r21;
    r8 = fmaf(r51, r8, r47 * r88);
    r88 = r39 * r55;
    r8 = fmaf(r81, r88, r8);
    r88 = r56 * r21;
    r66 = r5 * r78;
    r66 = fmaf(r47, r66, r51 * r88);
    r66 = fmaf(r39, r65, r66);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r33,
                                          r42,
                                          r8,
                                          r66);
    r88 = r50 * r55;
    r9 = r50 * r48;
    r75 = r27 * r45;
    r75 = r75 * r5;
    r75 = fmaf(r6, r75, r37 * r9);
    r9 = r27 * r41;
    r75 = fmaf(r55, r9, r75);
    r75 = fmaf(r50, r72, r75);
    r9 = r0 * r75;
    r9 = fmaf(r47, r9, r81 * r88);
    r88 = r41 * r21;
    r9 = fmaf(r51, r88, r9);
    r88 = r5 * r75;
    r7 = r45 * r21;
    r7 = fmaf(r51, r7, r47 * r88);
    r7 = fmaf(r50, r65, r7);
    r88 = r27 * r38;
    r88 = r88 * r5;
    r82 = r43 * r48;
    r82 = fmaf(r37, r82, r6 * r88);
    r88 = r27 * r40;
    r82 = fmaf(r55, r88, r82);
    r82 = fmaf(r43, r72, r82);
    r88 = r0 * r82;
    r77 = r40 * r21;
    r77 = fmaf(r51, r77, r47 * r88);
    r88 = r43 * r55;
    r77 = fmaf(r81, r88, r77);
    r88 = r5 * r82;
    r24 = r38 * r21;
    r24 = fmaf(r51, r24, r47 * r88);
    r24 = fmaf(r43, r65, r24);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r9,
                                          r7,
                                          r77,
                                          r24);
    r88 = r4 * r3;
    r84 = r4 * r2;
    r84 = fmaf(r1, r84, r54 * r88);
    r88 = r4 * r3;
    r58 = r4 * r2;
    r58 = fmaf(r76, r58, r89 * r88);
    r88 = r4 * r3;
    r83 = r4 * r2;
    r83 = fmaf(r33, r83, r42 * r88);
    r88 = r4 * r2;
    r70 = r4 * r3;
    r70 = fmaf(r66, r70, r8 * r88);
    WriteSum4<float, float>((float*)inout_shared, r84, r58, r83, r70);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = r4 * r2;
    r83 = r4 * r3;
    r83 = fmaf(r7, r83, r9 * r70);
    r70 = r4 * r2;
    r58 = r4 * r3;
    r58 = fmaf(r24, r58, r77 * r70);
    WriteSum2<float, float>((float*)inout_shared, r83, r58);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = fmaf(r54, r54, r1 * r1);
    r83 = fmaf(r89, r89, r76 * r76);
    r70 = fmaf(r33, r33, r42 * r42);
    r84 = fmaf(r8, r8, r66 * r66);
    WriteSum4<float, float>((float*)inout_shared, r58, r83, r70, r84);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r84 = fmaf(r9, r9, r7 * r7);
    r70 = fmaf(r77, r77, r24 * r24);
    WriteSum2<float, float>((float*)inout_shared, r84, r70);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fmaf(r1, r76, r54 * r89);
    r84 = fmaf(r54, r42, r1 * r33);
    r83 = fmaf(r1, r8, r54 * r66);
    r58 = fmaf(r1, r9, r54 * r7);
    WriteSum4<float, float>((float*)inout_shared, r70, r84, r83, r58);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fmaf(r1, r77, r54 * r24);
    r54 = fmaf(r89, r42, r76 * r33);
    r58 = fmaf(r76, r8, r89 * r66);
    r83 = fmaf(r89, r7, r76 * r9);
    WriteSum4<float, float>((float*)inout_shared, r1, r54, r58, r83);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r89 = fmaf(r89, r24, r76 * r77);
    r76 = fmaf(r42, r66, r33 * r8);
    r83 = fmaf(r42, r7, r33 * r9);
    r42 = fmaf(r42, r24, r33 * r77);
    WriteSum4<float, float>((float*)inout_shared, r89, r76, r83, r42);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = fmaf(r8, r9, r66 * r7);
    r66 = fmaf(r66, r24, r8 * r77);
    r77 = fmaf(r9, r77, r7 * r24);
    WriteSum3<float, float>((float*)inout_shared, r42, r66, r77);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = r0 * r21;
    r77 = r77 * r53;
    r66 = r5 * r21;
    r66 = r66 * r53;
    r42 = r0 * r36;
    r42 = r42 * r51;
    r9 = r5 * r36;
    r9 = r9 * r51;
    WriteIdx4<1024, float, float, float4>(
        out_focal_and_distortion_jac,
        0 * out_focal_and_distortion_jac_num_alloc,
        global_thread_idx,
        r77,
        r66,
        r42,
        r9);
    r9 = r5 * r3;
    r9 = r9 * r53;
    r42 = r0 * r2;
    r42 = r42 * r53;
    r42 = fmaf(r80, r42, r80 * r9);
    r9 = r4 * r0;
    r9 = r9 * r36;
    r9 = r9 * r2;
    r80 = r4 * r5;
    r80 = r80 * r36;
    r80 = r80 * r3;
    r80 = fmaf(r51, r80, r51 * r9);
    WriteSum2<float, float>((float*)inout_shared, r42, r80);
  };
  FlushSumShared<2, float>(out_focal_and_distortion_njtr,
                           0 * out_focal_and_distortion_njtr_num_alloc,
                           focal_and_distortion_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = r21 * r6;
    r80 = r80 * r37;
    r42 = r0 * r21;
    r42 = r42 * r21;
    r42 = fmaf(r55, r42, r21 * r80);
    r9 = r0 * r55;
    r53 = r46 * r46;
    r66 = r36 * r36;
    r53 = r53 * r66;
    r66 = r6 * r37;
    r66 = fmaf(r53, r66, r53 * r9);
    WriteSum2<float, float>((float*)inout_shared, r42, r66);
  };
  FlushSumShared<2, float>(out_focal_and_distortion_precond_diag,
                           0 * out_focal_and_distortion_precond_diag_num_alloc,
                           focal_and_distortion_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = r46 * r0;
    r66 = r66 * r36;
    r66 = r66 * r21;
    r42 = r46 * r36;
    r42 = fmaf(r80, r42, r55 * r66);
    WriteSum1<float, float>((float*)inout_shared, r42);
  };
  FlushSumShared<1, float>(out_focal_and_distortion_precond_tril,
                           0 * out_focal_and_distortion_precond_tril_num_alloc,
                           focal_and_distortion_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = r49 * r55;
    r66 = r26 * r21;
    r66 = fmaf(r51, r66, r81 * r42);
    r42 = r27 * r29;
    r42 = r42 * r5;
    r80 = r27 * r26;
    r80 = fmaf(r55, r80, r6 * r42);
    r42 = r49 * r48;
    r80 = fmaf(r37, r42, r80);
    r80 = fmaf(r49, r72, r80);
    r42 = r0 * r80;
    r66 = fmaf(r47, r42, r66);
    r42 = r5 * r80;
    r42 = fmaf(r47, r42, r49 * r65);
    r9 = r29 * r21;
    r42 = fmaf(r51, r9, r42);
    r9 = r52 * r55;
    r53 = r30 * r21;
    r53 = fmaf(r51, r53, r81 * r9);
    r9 = r52 * r48;
    r77 = r27 * r30;
    r77 = fmaf(r55, r77, r37 * r9);
    r9 = r27 * r19;
    r9 = r9 * r5;
    r77 = fmaf(r6, r9, r77);
    r77 = fmaf(r52, r72, r77);
    r9 = r0 * r77;
    r53 = fmaf(r47, r9, r53);
    r9 = r19 * r21;
    r9 = fmaf(r52, r65, r51 * r9);
    r24 = r5 * r77;
    r9 = fmaf(r47, r24, r9);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r66,
                                          r42,
                                          r53,
                                          r9);
    r24 = r25 * r55;
    r7 = r34 * r21;
    r7 = fmaf(r51, r7, r81 * r24);
    r24 = r27 * r32;
    r24 = r24 * r5;
    r81 = r25 * r48;
    r81 = fmaf(r37, r81, r6 * r24);
    r24 = r27 * r34;
    r81 = fmaf(r55, r24, r81);
    r81 = fmaf(r25, r72, r81);
    r24 = r0 * r81;
    r7 = fmaf(r47, r24, r7);
    r24 = r5 * r81;
    r72 = r32 * r21;
    r72 = fmaf(r51, r72, r47 * r24);
    r72 = fmaf(r25, r65, r72);
    WriteIdx2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r7, r72);
    r65 = r4 * r3;
    r24 = r4 * r2;
    r24 = fmaf(r66, r24, r42 * r65);
    r65 = r4 * r3;
    r51 = r4 * r2;
    r51 = fmaf(r53, r51, r9 * r65);
    r65 = r4 * r2;
    r47 = r4 * r3;
    r47 = fmaf(r72, r47, r7 * r65);
    WriteSum3<float, float>((float*)inout_shared, r24, r51, r47);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fmaf(r66, r66, r42 * r42);
    r51 = fmaf(r53, r53, r9 * r9);
    r24 = fmaf(r72, r72, r7 * r7);
    WriteSum3<float, float>((float*)inout_shared, r47, r51, r24);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fmaf(r66, r53, r42 * r9);
    r66 = fmaf(r66, r7, r42 * r72);
    r72 = fmaf(r9, r72, r53 * r7);
    WriteSum3<float, float>((float*)inout_shared, r24, r66, r72);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void SimpleRadialSplitFixedPrincipalPointResJac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* focal_and_distortion,
    unsigned int focal_and_distortion_num_alloc,
    SharedIndex* focal_and_distortion_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    float* out_focal_and_distortion_jac,
    unsigned int out_focal_and_distortion_jac_num_alloc,
    float* const out_focal_and_distortion_njtr,
    unsigned int out_focal_and_distortion_njtr_num_alloc,
    float* const out_focal_and_distortion_precond_diag,
    unsigned int out_focal_and_distortion_precond_diag_num_alloc,
    float* const out_focal_and_distortion_precond_tril,
    unsigned int out_focal_and_distortion_precond_tril_num_alloc,
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
  SimpleRadialSplitFixedPrincipalPointResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      focal_and_distortion,
      focal_and_distortion_num_alloc,
      focal_and_distortion_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      principal_point,
      principal_point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      out_focal_and_distortion_jac,
      out_focal_and_distortion_jac_num_alloc,
      out_focal_and_distortion_njtr,
      out_focal_and_distortion_njtr_num_alloc,
      out_focal_and_distortion_precond_diag,
      out_focal_and_distortion_precond_diag_num_alloc,
      out_focal_and_distortion_precond_tril,
      out_focal_and_distortion_precond_tril_num_alloc,
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