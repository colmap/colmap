#include "kernel_simple_radial_split_fixed_focal_and_distortion_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndDistortionResJacKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
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
        float* focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc,
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

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

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
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88;
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
    r46 = 9.99999999999999955e-07;
    r47 = r20 * r22;
    r47 = fmaf(r31, r47, r32);
    r6 = fmaf(r7, r47, r6);
    r39 = fmaf(r20, r39, r38);
    r43 = r21 + r43;
    r38 = r14 * r14;
    r38 = r38 * r20;
    r43 = r43 + r38;
    r32 = r15 * r16;
    r32 = r32 * r27;
    r48 = r14 * r17;
    r48 = fmaf(r27, r48, r32);
    r49 = r27 * r18;
    r49 = r49 * r22;
    r50 = fmaf(r28, r33, r49);
    r51 = r28 * r28;
    r51 = r51 * r20;
    r25 = r51 + r25;
    r6 = fmaf(r35, r39, r6);
    r6 = fmaf(r37, r43, r6);
    r6 = fmaf(r36, r48, r6);
    r6 = fmaf(r8, r50, r6);
    r6 = fmaf(r9, r25, r6);
    r52 = copysign(1.0, r6);
    r52 = fmaf(r46, r52, r6);
    r46 = 1.0 / r52;
    ReadIdx2<1024, float, float, float2>(focal_and_distortion,
                                         0 * focal_and_distortion_num_alloc,
                                         global_thread_idx,
                                         r6,
                                         r53);
    r54 = r52 * r52;
    r55 = 1.0 / r54;
    r56 = r0 * r55;
    r29 = fmaf(r18, r33, r29);
    r5 = fmaf(r7, r29, r5);
    r57 = r16 * r17;
    r57 = fmaf(r27, r57, r42);
    r45 = r21 + r45;
    r45 = r45 + r38;
    r38 = r14 * r17;
    r38 = fmaf(r20, r38, r32);
    r32 = r28 * r20;
    r32 = fmaf(r31, r32, r49);
    r19 = r21 + r19;
    r19 = r19 + r51;
    r5 = fmaf(r35, r57, r5);
    r5 = fmaf(r36, r45, r5);
    r5 = fmaf(r37, r38, r5);
    r5 = fmaf(r9, r32, r5);
    r5 = fmaf(r8, r19, r5);
    r37 = r5 * r5;
    r36 = fmaf(r55, r37, r0 * r56);
    r36 = fmaf(r53, r36, r21);
    r36 = r6 * r36;
    r35 = r46 * r36;
    r2 = fmaf(r0, r35, r2);
    r3 = fmaf(r3, r4, r1);
    r3 = fmaf(r5, r35, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = r10 * r17;
    r51 = -5.00000000000000000e-01;
    r49 = r13 * r14;
    r49 = fmaf(r51, r49, r51 * r1);
    r1 = r12 * r15;
    r49 = fmaf(r51, r1, r49);
    r42 = r11 * r16;
    r58 = 5.00000000000000000e-01;
    r49 = fmaf(r58, r42, r49);
    r42 = r22 * r49;
    r1 = r12 * r17;
    r59 = r11 * r14;
    r59 = fmaf(r58, r59, r58 * r1);
    r1 = r10 * r15;
    r59 = fmaf(r51, r1, r59);
    r60 = r13 * r58;
    r59 = fmaf(r16, r60, r59);
    r1 = fmaf(r59, r33, r27 * r42);
    r61 = r27 * r28;
    r62 = r13 * r15;
    r63 = r11 * r51;
    r62 = fmaf(r17, r63, r51 * r62);
    r62 = fmaf(r58, r24, r62);
    r62 = fmaf(r51, r23, r62);
    r64 = r27 * r18;
    r65 = r10 * r14;
    r66 = r12 * r16;
    r66 = fmaf(r51, r66, r51 * r65);
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
    r68 = fmaf(r49, r69, r68);
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
    r72 = r27 * r22;
    r72 = r72 * r62;
    r73 = r27 * r18;
    r73 = fmaf(r59, r73, r72);
    r74 = r27 * r28;
    r74 = r74 * r49;
    r75 = r66 * r33;
    r76 = r74 + r75;
    r77 = r73 + r76;
    r78 = r20 * r31;
    r78 = fmaf(r20, r42, r59 * r78);
    r78 = r78 + r61;
    r78 = fmaf(r7, r78, r8 * r77);
    r77 = r28 * r70;
    r59 = r66 * r77;
    r1 = r1 + r59;
    r78 = fmaf(r9, r1, r78);
    r1 = r4 * r36;
    r1 = r1 * r56;
    r79 = fmaf(r78, r1, r68 * r35);
    r80 = r6 * r53;
    r81 = r27 * r5;
    r82 = r28 * r20;
    r83 = r20 * r31;
    r83 = r83 * r66;
    r82 = fmaf(r49, r82, r83);
    r82 = r82 + r73;
    r59 = r71 + r59;
    r59 = fmaf(r8, r59, r9 * r82);
    r67 = fmaf(r62, r33, r67);
    r82 = r27 * r18;
    r82 = fmaf(r49, r82, r65);
    r67 = r67 + r82;
    r59 = fmaf(r7, r67, r59);
    r81 = r81 * r59;
    r67 = r27 * r68;
    r67 = fmaf(r56, r67, r55 * r81);
    r54 = r52 * r54;
    r54 = 1.0 / r54;
    r54 = r20 * r54;
    r52 = r78 * r54;
    r67 = fmaf(r37, r52, r67);
    r81 = r0 * r0;
    r81 = r81 * r54;
    r67 = fmaf(r78, r81, r67);
    r80 = r80 * r67;
    r80 = r80 * r46;
    r79 = fmaf(r0, r80, r79);
    r67 = r4 * r5;
    r67 = r67 * r78;
    r67 = r67 * r55;
    r67 = fmaf(r36, r67, r59 * r35);
    r67 = fmaf(r5, r80, r67);
    r80 = r20 * r22;
    r80 = fmaf(r62, r80, r83);
    r59 = r27 * r18;
    r52 = r12 * r17;
    r65 = r10 * r15;
    r65 = fmaf(r58, r65, r51 * r52);
    r52 = r13 * r16;
    r65 = fmaf(r51, r52, r65);
    r65 = fmaf(r14, r63, r65);
    r59 = r59 * r65;
    r52 = r27 * r28;
    r71 = r10 * r17;
    r73 = r12 * r15;
    r73 = fmaf(r58, r73, r58 * r71);
    r73 = fmaf(r14, r60, r73);
    r73 = fmaf(r16, r63, r73);
    r52 = fmaf(r73, r52, r59);
    r80 = r80 + r52;
    r63 = r27 * r22;
    r63 = r63 * r73;
    r71 = fmaf(r65, r33, r63);
    r71 = r71 + r61;
    r71 = fmaf(r8, r71, r7 * r80);
    r80 = r22 * r66;
    r80 = r80 * r70;
    r61 = r65 * r77;
    r84 = r80 + r61;
    r71 = fmaf(r9, r84, r71);
    r75 = r72 + r75;
    r75 = r75 + r52;
    r52 = r18 * r70;
    r52 = r52 * r73;
    r80 = r80 + r52;
    r80 = fmaf(r7, r80, r9 * r75);
    r75 = r20 * r31;
    r75 = fmaf(r20, r69, r73 * r75);
    r72 = r27 * r28;
    r72 = r72 * r66;
    r84 = r27 * r22;
    r84 = fmaf(r65, r84, r72);
    r75 = r75 + r84;
    r80 = fmaf(r8, r75, r80);
    r75 = fmaf(r80, r35, r71 * r1);
    r85 = r6 * r53;
    r86 = r27 * r80;
    r86 = fmaf(r71, r81, r56 * r86);
    r87 = r71 * r54;
    r86 = fmaf(r37, r87, r86);
    r88 = r27 * r5;
    r63 = r64 + r63;
    r64 = r28 * r20;
    r63 = fmaf(r62, r64, r63);
    r62 = r20 * r31;
    r63 = fmaf(r65, r62, r63);
    r73 = fmaf(r73, r33, r27 * r69);
    r73 = r73 + r84;
    r73 = fmaf(r7, r73, r9 * r63);
    r61 = r52 + r61;
    r73 = fmaf(r8, r61, r73);
    r88 = r88 * r73;
    r86 = fmaf(r55, r88, r86);
    r85 = r85 * r0;
    r85 = r85 * r86;
    r75 = fmaf(r46, r85, r75);
    r85 = r6 * r53;
    r85 = r85 * r5;
    r85 = r85 * r86;
    r85 = fmaf(r46, r85, r73 * r35);
    r73 = r4 * r5;
    r73 = r73 * r71;
    r73 = r73 * r55;
    r85 = fmaf(r36, r73, r85);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r79,
                                          r67,
                                          r75,
                                          r85);
    r73 = r11 * r17;
    r24 = fmaf(r51, r24, r58 * r73);
    r24 = fmaf(r15, r60, r24);
    r24 = fmaf(r58, r23, r24);
    r77 = r24 * r77;
    r42 = r70 * r42;
    r23 = r77 + r42;
    r58 = r27 * r18;
    r58 = r58 * r24;
    r72 = r72 + r58;
    r60 = r20 * r22;
    r72 = fmaf(r65, r60, r72);
    r51 = r20 * r31;
    r72 = fmaf(r49, r51, r72);
    r72 = fmaf(r7, r72, r9 * r23);
    r23 = r27 * r28;
    r23 = fmaf(r24, r33, r65 * r23);
    r23 = r23 + r82;
    r72 = fmaf(r8, r23, r72);
    r83 = r74 + r83;
    r74 = r27 * r22;
    r74 = r74 * r24;
    r23 = r18 * r20;
    r83 = fmaf(r65, r23, r83);
    r83 = r83 + r74;
    r66 = r18 * r66;
    r66 = r66 * r70;
    r42 = r66 + r42;
    r42 = fmaf(r7, r42, r8 * r83);
    r33 = fmaf(r49, r33, r58);
    r33 = r33 + r84;
    r42 = fmaf(r9, r33, r42);
    r33 = fmaf(r42, r35, r72 * r1);
    r84 = r6 * r53;
    r49 = r27 * r42;
    r49 = fmaf(r56, r49, r72 * r81);
    r58 = r72 * r54;
    r49 = fmaf(r37, r58, r49);
    r83 = r27 * r5;
    r74 = r59 + r74;
    r74 = r74 + r76;
    r76 = r28 * r20;
    r59 = r20 * r31;
    r59 = fmaf(r24, r59, r65 * r76);
    r59 = r59 + r82;
    r59 = fmaf(r9, r59, r7 * r74);
    r77 = r66 + r77;
    r59 = fmaf(r8, r77, r59);
    r83 = r83 * r59;
    r49 = fmaf(r55, r83, r49);
    r84 = r84 * r0;
    r84 = r84 * r49;
    r33 = fmaf(r46, r84, r33);
    r84 = r6 * r53;
    r84 = r84 * r5;
    r84 = r84 * r49;
    r49 = r4 * r5;
    r49 = r49 * r72;
    r49 = r49 * r55;
    r49 = fmaf(r36, r49, r46 * r84);
    r49 = fmaf(r59, r35, r49);
    r59 = r6 * r53;
    r84 = r39 * r54;
    r83 = r27 * r57;
    r83 = r83 * r5;
    r83 = fmaf(r55, r83, r37 * r84);
    r84 = r27 * r44;
    r83 = fmaf(r56, r84, r83);
    r83 = fmaf(r39, r81, r83);
    r59 = r59 * r0;
    r59 = r59 * r83;
    r59 = fmaf(r44, r35, r46 * r59);
    r59 = fmaf(r39, r1, r59);
    r84 = r6 * r53;
    r84 = r84 * r5;
    r84 = r84 * r83;
    r84 = fmaf(r46, r84, r57 * r35);
    r83 = r4 * r39;
    r83 = r83 * r5;
    r83 = r83 * r55;
    r84 = fmaf(r36, r83, r84);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r33,
                                          r49,
                                          r59,
                                          r84);
    r83 = r6 * r53;
    r58 = r48 * r54;
    r77 = r27 * r45;
    r77 = r77 * r5;
    r77 = fmaf(r55, r77, r37 * r58);
    r58 = r27 * r41;
    r77 = fmaf(r56, r58, r77);
    r77 = fmaf(r48, r81, r77);
    r83 = r83 * r0;
    r83 = r83 * r77;
    r83 = fmaf(r46, r83, r48 * r1);
    r83 = fmaf(r41, r35, r83);
    r58 = r6 * r53;
    r58 = r58 * r5;
    r58 = r58 * r77;
    r58 = fmaf(r45, r35, r46 * r58);
    r77 = r4 * r48;
    r77 = r77 * r5;
    r77 = r77 * r55;
    r58 = fmaf(r36, r77, r58);
    r77 = r6 * r53;
    r8 = r27 * r38;
    r8 = r8 * r5;
    r66 = r43 * r54;
    r66 = fmaf(r37, r66, r55 * r8);
    r8 = r27 * r40;
    r66 = fmaf(r56, r8, r66);
    r66 = fmaf(r43, r81, r66);
    r77 = r77 * r0;
    r77 = r77 * r66;
    r77 = fmaf(r40, r35, r46 * r77);
    r77 = fmaf(r43, r1, r77);
    r8 = r6 * r53;
    r8 = r8 * r5;
    r8 = r8 * r66;
    r8 = fmaf(r38, r35, r46 * r8);
    r66 = r4 * r43;
    r66 = r66 * r5;
    r66 = r66 * r55;
    r8 = fmaf(r36, r66, r8);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r83,
                                          r58,
                                          r77,
                                          r8);
    r66 = r4 * r2;
    r3 = r4 * r3;
    r66 = fmaf(r67, r3, r79 * r66);
    r9 = r4 * r2;
    r9 = fmaf(r85, r3, r75 * r9);
    r74 = r4 * r2;
    r74 = fmaf(r49, r3, r33 * r74);
    r7 = r4 * r2;
    r7 = fmaf(r84, r3, r59 * r7);
    WriteSum4<float, float>((float*)inout_shared, r66, r9, r74, r7);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r4 * r2;
    r7 = fmaf(r58, r3, r83 * r7);
    r74 = r4 * r2;
    r74 = fmaf(r8, r3, r77 * r74);
    WriteSum2<float, float>((float*)inout_shared, r7, r74);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fmaf(r67, r67, r79 * r79);
    r7 = fmaf(r85, r85, r75 * r75);
    r9 = fmaf(r33, r33, r49 * r49);
    r66 = fmaf(r59, r59, r84 * r84);
    WriteSum4<float, float>((float*)inout_shared, r74, r7, r9, r66);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = fmaf(r83, r83, r58 * r58);
    r9 = fmaf(r77, r77, r8 * r8);
    WriteSum2<float, float>((float*)inout_shared, r66, r9);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = fmaf(r79, r75, r67 * r85);
    r66 = fmaf(r67, r49, r79 * r33);
    r7 = fmaf(r79, r59, r67 * r84);
    r74 = fmaf(r79, r83, r67 * r58);
    WriteSum4<float, float>((float*)inout_shared, r9, r66, r7, r74);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = fmaf(r79, r77, r67 * r8);
    r67 = fmaf(r85, r49, r75 * r33);
    r74 = fmaf(r75, r59, r85 * r84);
    r7 = fmaf(r85, r58, r75 * r83);
    WriteSum4<float, float>((float*)inout_shared, r79, r67, r74, r7);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = fmaf(r85, r8, r75 * r77);
    r75 = fmaf(r49, r84, r33 * r59);
    r7 = fmaf(r49, r58, r33 * r83);
    r49 = fmaf(r49, r8, r33 * r77);
    WriteSum4<float, float>((float*)inout_shared, r85, r75, r7, r49);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = fmaf(r59, r83, r84 * r58);
    r84 = fmaf(r84, r8, r59 * r77);
    r77 = fmaf(r83, r77, r58 * r8);
    WriteSum3<float, float>((float*)inout_shared, r49, r84, r77);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = r4 * r2;
    WriteSum2<float, float>((float*)inout_shared, r77, r3);
  };
  FlushSumShared<2, float>(out_principal_point_njtr,
                           0 * out_principal_point_njtr_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<float, float>((float*)inout_shared, r21, r21);
  };
  FlushSumShared<2, float>(out_principal_point_precond_diag,
                           0 * out_principal_point_precond_diag_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = fmaf(r26, r35, r47 * r1);
    r77 = r6 * r53;
    r84 = r27 * r29;
    r84 = r84 * r5;
    r49 = r27 * r26;
    r49 = fmaf(r56, r49, r55 * r84);
    r84 = r47 * r54;
    r49 = fmaf(r37, r84, r49);
    r49 = fmaf(r47, r81, r49);
    r77 = r77 * r0;
    r77 = r77 * r49;
    r21 = fmaf(r46, r77, r21);
    r77 = r4 * r47;
    r77 = r77 * r5;
    r77 = r77 * r55;
    r84 = r6 * r53;
    r84 = r84 * r5;
    r84 = r84 * r49;
    r84 = fmaf(r46, r84, r36 * r77);
    r84 = fmaf(r29, r35, r84);
    r77 = fmaf(r30, r35, r50 * r1);
    r49 = r6 * r53;
    r83 = r50 * r54;
    r8 = r27 * r30;
    r8 = fmaf(r56, r8, r37 * r83);
    r83 = r27 * r19;
    r83 = r83 * r5;
    r8 = fmaf(r55, r83, r8);
    r8 = fmaf(r50, r81, r8);
    r49 = r49 * r0;
    r49 = r49 * r8;
    r77 = fmaf(r46, r49, r77);
    r49 = r4 * r50;
    r49 = r49 * r5;
    r49 = r49 * r55;
    r49 = fmaf(r36, r49, r19 * r35);
    r83 = r6 * r53;
    r83 = r83 * r5;
    r83 = r83 * r8;
    r49 = fmaf(r46, r83, r49);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r21,
                                          r84,
                                          r77,
                                          r49);
    r1 = fmaf(r34, r35, r25 * r1);
    r83 = r6 * r53;
    r8 = r27 * r32;
    r8 = r8 * r5;
    r58 = r25 * r54;
    r58 = fmaf(r37, r58, r55 * r8);
    r8 = r27 * r34;
    r58 = fmaf(r56, r8, r58);
    r58 = fmaf(r25, r81, r58);
    r83 = r83 * r0;
    r83 = r83 * r58;
    r1 = fmaf(r46, r83, r1);
    r83 = r6 * r53;
    r83 = r83 * r5;
    r83 = r83 * r58;
    r35 = fmaf(r32, r35, r46 * r83);
    r83 = r4 * r25;
    r83 = r83 * r5;
    r83 = r83 * r55;
    r35 = fmaf(r36, r83, r35);
    WriteIdx2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r1, r35);
    r83 = r4 * r2;
    r83 = fmaf(r84, r3, r21 * r83);
    r36 = r4 * r2;
    r36 = fmaf(r49, r3, r77 * r36);
    r55 = r4 * r2;
    r3 = fmaf(r35, r3, r1 * r55);
    WriteSum3<float, float>((float*)inout_shared, r83, r36, r3);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = fmaf(r21, r21, r84 * r84);
    r36 = fmaf(r77, r77, r49 * r49);
    r83 = fmaf(r35, r35, r1 * r1);
    WriteSum3<float, float>((float*)inout_shared, r3, r36, r83);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = fmaf(r21, r77, r84 * r49);
    r21 = fmaf(r21, r1, r84 * r35);
    r35 = fmaf(r49, r35, r77 * r1);
    WriteSum3<float, float>((float*)inout_shared, r83, r21, r35);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void SimpleRadialSplitFixedFocalAndDistortionResJac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
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
    float* focal_and_distortion,
    unsigned int focal_and_distortion_num_alloc,
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
  SimpleRadialSplitFixedFocalAndDistortionResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
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
      focal_and_distortion,
      focal_and_distortion_num_alloc,
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