#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_opencv_split_fixed_focal_and_extra_fixed_point_res_jac.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedFocalAndExtraFixedPointResJacKernel(
        float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
        float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        float *principal_point, unsigned int principal_point_num_alloc,
        SharedIndex *principal_point_indices, float *pixel,
        unsigned int pixel_num_alloc, float *focal_and_extra,
        unsigned int focal_and_extra_num_alloc, float *point,
        unsigned int point_num_alloc, float *out_res,
        unsigned int out_res_num_alloc, float *out_pose_jac,
        unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float *const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float *const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        float *out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        float *const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        float *const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        float *const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92;
  LoadShared<2, float, float>(principal_point, 0 * principal_point_num_alloc,
                              principal_point_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float *)inout_shared,
                       principal_point_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(focal_and_extra,
                                         0 * focal_and_extra_num_alloc,
                                         global_thread_idx, r2, r3, r4, r5);
    ReadIdx2<1024, float, float, float2>(focal_and_extra,
                                         4 * focal_and_extra_num_alloc,
                                         global_thread_idx, r6, r7);
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r8, r9, r10);
    ReadIdx3<1024, float, float, float4>(point, 0 * point_num_alloc,
                                         global_thread_idx, r11, r12, r13);
    r14 = 2.00000000000000000e+00;
  };
  LoadShared<4, float, float>(pose, 0 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r15, r16, r17,
                       r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r19, r20, r21, r22);
    r23 = fmaf(r18, r19, r15 * r22);
    r24 = r16 * r21;
    r25 = -1.00000000000000000e+00;
    r23 = fmaf(r25, r24, r23);
    r23 = fmaf(r17, r20, r23);
    r24 = r14 * r23;
    r26 = r18 * r20;
    r27 = fmaf(r16, r22, r26);
    r28 = r15 * r21;
    r29 = r17 * r19;
    r27 = r27 + r28;
    r27 = fmaf(r25, r29, r27);
    r24 = r24 * r27;
    r30 = fmaf(r16, r19, r17 * r22);
    r31 = r15 * r20;
    r30 = fmaf(r25, r31, r30);
    r30 = fmaf(r18, r21, r30);
    r31 = r14 * r30;
    r32 = fmaf(r16, r20, r15 * r19);
    r32 = fmaf(r17, r21, r32);
    r32 = fmaf(r25, r32, r18 * r22);
    r31 = fmaf(r32, r31, r24);
    r31 = fmaf(r11, r31, r9);
  };
  LoadShared<3, float, float>(pose, 4 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r9, r33, r34);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r35 = r19 * r20;
    r35 = r35 * r14;
    r36 = r21 * r22;
    r36 = fmaf(r14, r36, r35);
    r37 = r19 * r19;
    r38 = -2.00000000000000000e+00;
    r37 = r37 * r38;
    r39 = 1.00000000000000000e+00;
    r40 = r21 * r21;
    r40 = fmaf(r38, r40, r39);
    r41 = r37 + r40;
    r42 = r20 * r21;
    r42 = r42 * r14;
    r43 = r19 * r22;
    r43 = fmaf(r38, r43, r42);
    r44 = r14 * r30;
    r44 = r44 * r27;
    r45 = r38 * r32;
    r46 = fmaf(r23, r45, r44);
    r47 = r30 * r30;
    r47 = r47 * r38;
    r48 = r39 + r47;
    r49 = r23 * r23;
    r49 = r49 * r38;
    r48 = r48 + r49;
    r31 = fmaf(r9, r36, r31);
    r31 = fmaf(r33, r41, r31);
    r31 = fmaf(r34, r43, r31);
    r31 = fmaf(r13, r46, r31);
    r31 = fmaf(r12, r48, r31);
    r48 = r31 * r31;
    r46 = 9.99999999999999955e-07;
    r50 = r14 * r30;
    r50 = r50 * r23;
    r51 = fmaf(r27, r45, r50);
    r51 = fmaf(r11, r51, r10);
    r10 = r19 * r21;
    r10 = r10 * r14;
    r52 = r20 * r22;
    r52 = fmaf(r38, r52, r10);
    r53 = r20 * r20;
    r53 = r53 * r38;
    r54 = r39 + r53;
    r54 = r54 + r37;
    r37 = r19 * r22;
    r37 = fmaf(r14, r37, r42);
    r42 = r14 * r23;
    r42 = fmaf(r32, r42, r44);
    r44 = r38 * r27;
    r44 = r44 * r27;
    r55 = r39 + r44;
    r55 = r55 + r49;
    r51 = fmaf(r9, r52, r51);
    r51 = fmaf(r34, r54, r51);
    r51 = fmaf(r33, r37, r51);
    r51 = fmaf(r12, r42, r51);
    r51 = fmaf(r13, r55, r51);
    r55 = copysign(1.0, r51);
    r55 = fmaf(r46, r55, r51);
    r46 = r55 * r55;
    r51 = 1.0 / r46;
    r48 = r48 * r51;
    r44 = r39 + r44;
    r44 = r44 + r47;
    r44 = fmaf(r11, r44, r8);
    r24 = fmaf(r30, r45, r24);
    r8 = r14 * r27;
    r8 = fmaf(r32, r8, r50);
    r50 = r20 * r22;
    r50 = fmaf(r14, r50, r10);
    r10 = r21 * r22;
    r10 = fmaf(r38, r10, r35);
    r40 = r53 + r40;
    r44 = fmaf(r12, r24, r44);
    r44 = fmaf(r13, r8, r44);
    r44 = fmaf(r34, r50, r44);
    r44 = fmaf(r33, r10, r44);
    r44 = fmaf(r9, r40, r44);
    r9 = 3.00000000000000000e+00;
    r33 = r44 * r9;
    r34 = r44 * r51;
    r33 = fmaf(r34, r33, r48);
    r8 = 1.0 / r55;
    r33 = fmaf(r44, r8, r7 * r33);
    r24 = r44 * r34;
    r48 = r24 + r48;
    r5 = r5 * r48;
    r48 = fmaf(r48, r5, r4 * r48);
    r53 = r48 * r8;
    r35 = r14 * r34;
    r47 = r6 * r35;
    r33 = fmaf(r44, r53, r33);
    r33 = fmaf(r31, r47, r33);
    r33 = fmaf(r2, r33, r0);
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r0, r42);
    r33 = fmaf(r0, r25, r33);
    r0 = r31 * r31;
    r0 = r0 * r9;
    r0 = fmaf(r51, r0, r24);
    r0 = fmaf(r31, r8, r6 * r0);
    r24 = r7 * r31;
    r0 = fmaf(r35, r24, r0);
    r0 = fmaf(r31, r53, r0);
    r0 = fmaf(r3, r0, r1);
    r0 = fmaf(r42, r25, r0);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r33, r0);
    r42 = 6.00000000000000000e+00;
    r1 = r14 * r32;
    r24 = r16 * r19;
    r49 = 5.00000000000000000e-01;
    r56 = r15 * r20;
    r57 = -5.00000000000000000e-01;
    r56 = fmaf(r57, r56, r49 * r24);
    r24 = r18 * r21;
    r56 = fmaf(r49, r24, r56);
    r58 = r22 * r49;
    r56 = fmaf(r17, r58, r56);
    r24 = r15 * r22;
    r59 = r18 * r19;
    r59 = fmaf(r57, r59, r57 * r24);
    r24 = r17 * r20;
    r59 = fmaf(r57, r24, r59);
    r60 = r16 * r21;
    r59 = fmaf(r49, r60, r59);
    r60 = r27 * r59;
    r1 = fmaf(r14, r60, r56 * r1);
    r24 = r14 * r23;
    r61 = r16 * r57;
    r62 = fmaf(r49, r29, r22 * r61);
    r62 = fmaf(r57, r26, r62);
    r62 = fmaf(r57, r28, r62);
    r63 = r14 * r30;
    r64 = r15 * r19;
    r65 = r17 * r21;
    r65 = fmaf(r57, r65, r57 * r64);
    r65 = fmaf(r18, r58, r65);
    r65 = fmaf(r20, r61, r65);
    r63 = r63 * r65;
    r24 = fmaf(r62, r24, r63);
    r1 = r1 + r24;
    r64 = r14 * r27;
    r64 = r64 * r65;
    r66 = r14 * r23;
    r66 = r66 * r56;
    r67 = r64 + r66;
    r68 = r30 * r38;
    r67 = fmaf(r59, r68, r67);
    r67 = fmaf(r62, r45, r67);
    r67 = fmaf(r12, r67, r13 * r1);
    r1 = r27 * r56;
    r68 = -4.00000000000000000e+00;
    r1 = r1 * r68;
    r69 = r30 * r62;
    r70 = r68 * r69;
    r71 = r1 + r70;
    r67 = fmaf(r11, r71, r67);
    r71 = r42 * r67;
    r72 = r44 * r44;
    r73 = r14 * r27;
    r73 = r73 * r62;
    r74 = r14 * r30;
    r74 = fmaf(r56, r74, r73);
    r75 = r14 * r23;
    r75 = r75 * r59;
    r76 = r14 * r32;
    r76 = r76 * r65;
    r77 = r75 + r76;
    r78 = r74 + r77;
    r56 = fmaf(r56, r45, r38 * r60);
    r56 = r56 + r24;
    r56 = fmaf(r11, r56, r12 * r78);
    r78 = r23 * r65;
    r78 = r78 * r68;
    r1 = r78 + r1;
    r56 = fmaf(r13, r1, r56);
    r1 = -6.00000000000000000e+00;
    r46 = r55 * r46;
    r46 = 1.0 / r46;
    r72 = r72 * r56;
    r72 = r72 * r1;
    r72 = fmaf(r46, r72, r34 * r71);
    r71 = r14 * r31;
    r55 = r23 * r38;
    r79 = r65 * r45;
    r55 = fmaf(r59, r55, r79);
    r55 = r55 + r74;
    r70 = r78 + r70;
    r70 = fmaf(r12, r70, r13 * r55);
    r55 = r14 * r32;
    r55 = fmaf(r62, r55, r66);
    r66 = r14 * r30;
    r66 = fmaf(r59, r66, r64);
    r55 = r55 + r66;
    r70 = fmaf(r11, r55, r70);
    r71 = r71 * r70;
    r55 = r38 * r31;
    r64 = r31 * r46;
    r55 = r55 * r56;
    r55 = fmaf(r64, r55, r51 * r71);
    r72 = r72 + r55;
    r71 = r25 * r48;
    r71 = r71 * r56;
    r71 = fmaf(r34, r71, r7 * r72);
    r72 = r6 * r56;
    r78 = r44 * r68;
    r78 = r78 * r64;
    r71 = fmaf(r78, r72, r71);
    r74 = r6 * r14;
    r74 = r74 * r31;
    r74 = r74 * r67;
    r71 = fmaf(r51, r74, r71);
    r80 = r25 * r56;
    r71 = fmaf(r34, r80, r71);
    r81 = r38 * r44;
    r81 = r81 * r44;
    r81 = r81 * r56;
    r81 = fmaf(r46, r81, r67 * r35);
    r55 = r55 + r81;
    r5 = r14 * r5;
    r55 = fmaf(r55, r5, r4 * r55);
    r82 = r44 * r55;
    r71 = fmaf(r8, r82, r71);
    r71 = fmaf(r70, r47, r71);
    r71 = fmaf(r67, r53, r71);
    r71 = fmaf(r67, r8, r71);
    r82 = r2 * r71;
    r80 = r31 * r70;
    r80 = r80 * r42;
    r74 = r31 * r1;
    r74 = r74 * r64;
    r80 = fmaf(r56, r74, r51 * r80);
    r80 = r80 + r81;
    r80 = fmaf(r70, r8, r6 * r80);
    r81 = r25 * r31;
    r81 = r81 * r56;
    r80 = fmaf(r51, r81, r80);
    r72 = r7 * r70;
    r80 = fmaf(r35, r72, r80);
    r83 = r31 * r55;
    r80 = fmaf(r8, r83, r80);
    r84 = r7 * r78;
    r85 = r7 * r14;
    r85 = r85 * r31;
    r85 = r85 * r67;
    r80 = fmaf(r51, r85, r80);
    r86 = r25 * r31;
    r86 = r86 * r48;
    r86 = r86 * r56;
    r80 = fmaf(r51, r86, r80);
    r80 = fmaf(r70, r53, r80);
    r80 = fmaf(r56, r84, r80);
    r86 = r3 * r80;
    r76 = r73 + r76;
    r73 = r14 * r30;
    r85 = r17 * r22;
    r83 = r15 * r20;
    r83 = fmaf(r49, r83, r57 * r85);
    r85 = r18 * r21;
    r83 = fmaf(r57, r85, r83);
    r83 = fmaf(r19, r61, r83);
    r73 = r73 * r83;
    r85 = r14 * r23;
    r72 = r18 * r19;
    r81 = r17 * r20;
    r81 = fmaf(r49, r81, r49 * r72);
    r81 = fmaf(r15, r58, r81);
    r81 = fmaf(r21, r61, r81);
    r85 = fmaf(r81, r85, r73);
    r76 = r76 + r85;
    r61 = r27 * r65;
    r61 = r61 * r68;
    r72 = r30 * r68;
    r72 = r72 * r81;
    r87 = r61 + r72;
    r87 = fmaf(r11, r87, r13 * r76);
    r76 = fmaf(r81, r45, r38 * r69);
    r88 = r14 * r23;
    r88 = r88 * r65;
    r89 = r14 * r27;
    r89 = fmaf(r83, r89, r88);
    r76 = r76 + r89;
    r87 = fmaf(r12, r76, r87);
    r76 = r42 * r87;
    r90 = r44 * r44;
    r91 = r38 * r27;
    r91 = fmaf(r62, r91, r79);
    r91 = r91 + r85;
    r85 = r14 * r27;
    r85 = r85 * r81;
    r92 = r14 * r32;
    r92 = fmaf(r83, r92, r85);
    r92 = r92 + r24;
    r92 = fmaf(r12, r92, r11 * r91);
    r91 = r23 * r83;
    r24 = r68 * r91;
    r61 = r61 + r24;
    r92 = fmaf(r13, r61, r92);
    r90 = r90 * r1;
    r90 = r90 * r92;
    r90 = fmaf(r46, r90, r34 * r76);
    r76 = r38 * r31;
    r76 = r76 * r92;
    r61 = r14 * r31;
    r85 = r63 + r85;
    r63 = r23 * r38;
    r85 = fmaf(r62, r63, r85);
    r85 = fmaf(r83, r45, r85);
    r63 = r14 * r32;
    r69 = fmaf(r14, r69, r81 * r63);
    r69 = r69 + r89;
    r69 = fmaf(r11, r69, r13 * r85);
    r24 = r72 + r24;
    r69 = fmaf(r12, r24, r69);
    r61 = r61 * r69;
    r61 = fmaf(r51, r61, r64 * r76);
    r90 = r90 + r61;
    r90 = fmaf(r87, r53, r7 * r90);
    r76 = r25 * r92;
    r90 = fmaf(r34, r76, r90);
    r24 = r6 * r92;
    r90 = fmaf(r78, r24, r90);
    r72 = r6 * r14;
    r72 = r72 * r31;
    r72 = r72 * r87;
    r90 = fmaf(r51, r72, r90);
    r85 = r25 * r48;
    r85 = r85 * r92;
    r90 = fmaf(r34, r85, r90);
    r63 = r38 * r44;
    r63 = r63 * r44;
    r63 = r63 * r92;
    r63 = fmaf(r46, r63, r87 * r35);
    r61 = r61 + r63;
    r61 = fmaf(r61, r5, r4 * r61);
    r81 = r44 * r61;
    r90 = fmaf(r8, r81, r90);
    r90 = fmaf(r87, r8, r90);
    r90 = fmaf(r69, r47, r90);
    r81 = r2 * r90;
    r85 = r31 * r42;
    r85 = r85 * r69;
    r85 = fmaf(r51, r85, r92 * r74);
    r85 = r85 + r63;
    r63 = r25 * r31;
    r63 = r63 * r48;
    r63 = r63 * r92;
    r63 = fmaf(r51, r63, r6 * r85);
    r85 = r31 * r61;
    r63 = fmaf(r8, r85, r63);
    r72 = r25 * r31;
    r72 = r72 * r92;
    r63 = fmaf(r51, r72, r63);
    r24 = r7 * r14;
    r24 = r24 * r31;
    r24 = r24 * r87;
    r63 = fmaf(r51, r24, r63);
    r76 = r7 * r69;
    r63 = fmaf(r35, r76, r63);
    r63 = fmaf(r69, r8, r63);
    r63 = fmaf(r92, r84, r63);
    r63 = fmaf(r69, r53, r63);
    r76 = r3 * r63;
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r82, r86,
        r81, r76);
    r76 = r44 * r44;
    r81 = r23 * r68;
    r29 = fmaf(r57, r29, r16 * r58);
    r29 = fmaf(r49, r26, r29);
    r29 = fmaf(r49, r28, r29);
    r81 = r81 * r29;
    r60 = r68 * r60;
    r28 = r81 + r60;
    r49 = r14 * r30;
    r49 = r49 * r29;
    r88 = r88 + r49;
    r26 = r38 * r27;
    r88 = fmaf(r83, r26, r88);
    r88 = fmaf(r59, r45, r88);
    r88 = fmaf(r11, r88, r13 * r28);
    r28 = r14 * r32;
    r28 = fmaf(r14, r91, r29 * r28);
    r28 = r28 + r66;
    r88 = fmaf(r12, r28, r88);
    r76 = r76 * r1;
    r76 = r76 * r88;
    r28 = r14 * r27;
    r28 = r28 * r29;
    r75 = r75 + r28;
    r26 = r30 * r38;
    r75 = fmaf(r83, r26, r75);
    r75 = r75 + r79;
    r65 = r30 * r65;
    r65 = r65 * r68;
    r60 = r65 + r60;
    r60 = fmaf(r11, r60, r12 * r75);
    r75 = r14 * r32;
    r75 = fmaf(r59, r75, r49);
    r75 = r75 + r89;
    r60 = fmaf(r13, r75, r60);
    r75 = r42 * r60;
    r75 = fmaf(r34, r75, r46 * r76);
    r76 = r38 * r31;
    r76 = r76 * r88;
    r89 = r14 * r31;
    r28 = r73 + r28;
    r28 = r28 + r77;
    r45 = fmaf(r29, r45, r38 * r91);
    r45 = r45 + r66;
    r45 = fmaf(r13, r45, r11 * r28);
    r81 = r65 + r81;
    r45 = fmaf(r12, r81, r45);
    r89 = r89 * r45;
    r89 = fmaf(r51, r89, r64 * r76);
    r75 = r75 + r89;
    r75 = fmaf(r60, r8, r7 * r75);
    r76 = r25 * r48;
    r76 = r76 * r88;
    r75 = fmaf(r34, r76, r75);
    r81 = r25 * r88;
    r75 = fmaf(r34, r81, r75);
    r12 = r6 * r14;
    r12 = r12 * r31;
    r12 = r12 * r60;
    r75 = fmaf(r51, r12, r75);
    r65 = r6 * r88;
    r75 = fmaf(r78, r65, r75);
    r13 = r38 * r44;
    r13 = r13 * r44;
    r13 = r13 * r88;
    r13 = fmaf(r60, r35, r46 * r13);
    r89 = r89 + r13;
    r89 = fmaf(r89, r5, r4 * r89);
    r28 = r44 * r89;
    r75 = fmaf(r8, r28, r75);
    r75 = fmaf(r45, r47, r75);
    r75 = fmaf(r60, r53, r75);
    r28 = r2 * r75;
    r65 = r31 * r42;
    r65 = r65 * r45;
    r65 = fmaf(r51, r65, r88 * r74);
    r65 = r65 + r13;
    r65 = fmaf(r45, r8, r6 * r65);
    r13 = r7 * r14;
    r13 = r13 * r31;
    r13 = r13 * r60;
    r65 = fmaf(r51, r13, r65);
    r12 = r31 * r89;
    r65 = fmaf(r8, r12, r65);
    r81 = r7 * r45;
    r65 = fmaf(r35, r81, r65);
    r76 = r25 * r31;
    r76 = r76 * r88;
    r65 = fmaf(r51, r76, r65);
    r11 = r25 * r31;
    r11 = r11 * r48;
    r11 = r11 * r88;
    r65 = fmaf(r51, r11, r65);
    r65 = fmaf(r88, r84, r65);
    r65 = fmaf(r45, r53, r65);
    r11 = r3 * r65;
    r76 = r52 * r44;
    r76 = r76 * r44;
    r76 = r76 * r1;
    r81 = r40 * r42;
    r81 = fmaf(r34, r81, r46 * r76);
    r76 = r38 * r52;
    r76 = r76 * r31;
    r12 = r14 * r36;
    r12 = r12 * r31;
    r12 = fmaf(r51, r12, r64 * r76);
    r81 = r81 + r12;
    r81 = fmaf(r40, r8, r7 * r81);
    r76 = r6 * r14;
    r76 = r76 * r40;
    r76 = r76 * r31;
    r81 = fmaf(r51, r76, r81);
    r13 = r6 * r52;
    r81 = fmaf(r78, r13, r81);
    r66 = r38 * r52;
    r66 = r66 * r44;
    r66 = r66 * r44;
    r66 = fmaf(r40, r35, r46 * r66);
    r12 = r12 + r66;
    r12 = fmaf(r12, r5, r4 * r12);
    r29 = r44 * r12;
    r81 = fmaf(r8, r29, r81);
    r91 = r25 * r52;
    r81 = fmaf(r34, r91, r81);
    r77 = r25 * r52;
    r77 = r77 * r48;
    r81 = fmaf(r34, r77, r81);
    r81 = fmaf(r40, r53, r81);
    r81 = fmaf(r36, r47, r81);
    r77 = r2 * r81;
    r91 = r36 * r31;
    r91 = r91 * r42;
    r91 = fmaf(r51, r91, r52 * r74);
    r91 = r91 + r66;
    r91 = fmaf(r36, r8, r6 * r91);
    r66 = r25 * r52;
    r66 = r66 * r31;
    r66 = r66 * r48;
    r91 = fmaf(r51, r66, r91);
    r29 = r7 * r14;
    r29 = r29 * r40;
    r29 = r29 * r31;
    r91 = fmaf(r51, r29, r91);
    r13 = r31 * r12;
    r91 = fmaf(r8, r13, r91);
    r76 = r25 * r52;
    r76 = r76 * r31;
    r91 = fmaf(r51, r76, r91);
    r73 = r7 * r36;
    r91 = fmaf(r35, r73, r91);
    r91 = fmaf(r52, r84, r91);
    r91 = fmaf(r36, r53, r91);
    r73 = r3 * r91;
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r28, r11,
        r77, r73);
    r73 = r37 * r44;
    r73 = r73 * r44;
    r73 = r73 * r1;
    r77 = r10 * r42;
    r77 = fmaf(r34, r77, r46 * r73);
    r73 = r38 * r37;
    r73 = r73 * r31;
    r11 = r14 * r41;
    r11 = r11 * r31;
    r11 = fmaf(r51, r11, r64 * r73);
    r77 = r77 + r11;
    r73 = r25 * r37;
    r73 = r73 * r48;
    r73 = fmaf(r34, r73, r7 * r77);
    r77 = r6 * r14;
    r77 = r77 * r10;
    r77 = r77 * r31;
    r73 = fmaf(r51, r77, r73);
    r28 = r38 * r37;
    r28 = r28 * r44;
    r28 = r28 * r44;
    r28 = fmaf(r10, r35, r46 * r28);
    r11 = r11 + r28;
    r11 = fmaf(r11, r5, r4 * r11);
    r76 = r44 * r11;
    r73 = fmaf(r8, r76, r73);
    r13 = r6 * r37;
    r73 = fmaf(r78, r13, r73);
    r29 = r25 * r37;
    r73 = fmaf(r34, r29, r73);
    r73 = fmaf(r10, r8, r73);
    r73 = fmaf(r41, r47, r73);
    r73 = fmaf(r10, r53, r73);
    r29 = r2 * r73;
    r13 = r41 * r31;
    r13 = r13 * r42;
    r13 = fmaf(r51, r13, r37 * r74);
    r13 = r13 + r28;
    r13 = fmaf(r41, r8, r6 * r13);
    r28 = r7 * r14;
    r28 = r28 * r10;
    r28 = r28 * r31;
    r13 = fmaf(r51, r28, r13);
    r76 = r7 * r41;
    r13 = fmaf(r35, r76, r13);
    r77 = r25 * r37;
    r77 = r77 * r31;
    r13 = fmaf(r51, r77, r13);
    r66 = r31 * r11;
    r13 = fmaf(r8, r66, r13);
    r49 = r25 * r37;
    r49 = r49 * r31;
    r49 = r49 * r48;
    r13 = fmaf(r51, r49, r13);
    r13 = fmaf(r41, r53, r13);
    r13 = fmaf(r37, r84, r13);
    r49 = r3 * r13;
    r66 = r50 * r42;
    r77 = r54 * r44;
    r77 = r77 * r44;
    r77 = r77 * r1;
    r77 = fmaf(r46, r77, r34 * r66);
    r66 = r14 * r43;
    r66 = r66 * r31;
    r1 = r38 * r54;
    r1 = r1 * r31;
    r1 = fmaf(r64, r1, r51 * r66);
    r77 = r77 + r1;
    r66 = r6 * r54;
    r66 = fmaf(r78, r66, r7 * r77);
    r77 = r38 * r54;
    r77 = r77 * r44;
    r77 = r77 * r44;
    r77 = fmaf(r46, r77, r50 * r35);
    r1 = r1 + r77;
    r5 = fmaf(r1, r5, r4 * r1);
    r1 = r44 * r5;
    r66 = fmaf(r8, r1, r66);
    r4 = r25 * r54;
    r4 = r4 * r48;
    r66 = fmaf(r34, r4, r66);
    r46 = r25 * r54;
    r66 = fmaf(r34, r46, r66);
    r34 = r6 * r14;
    r34 = r34 * r50;
    r34 = r34 * r31;
    r66 = fmaf(r51, r34, r66);
    r66 = fmaf(r50, r8, r66);
    r66 = fmaf(r43, r47, r66);
    r66 = fmaf(r50, r53, r66);
    r34 = r2 * r66;
    r46 = r43 * r31;
    r46 = r46 * r42;
    r74 = fmaf(r54, r74, r51 * r46);
    r74 = r74 + r77;
    r84 = fmaf(r54, r84, r6 * r74);
    r74 = r25 * r54;
    r74 = r74 * r31;
    r74 = r74 * r48;
    r84 = fmaf(r51, r74, r84);
    r77 = r7 * r43;
    r84 = fmaf(r35, r77, r84);
    r35 = r25 * r54;
    r35 = r35 * r31;
    r84 = fmaf(r51, r35, r84);
    r46 = r31 * r5;
    r84 = fmaf(r8, r46, r84);
    r4 = r7 * r14;
    r4 = r4 * r50;
    r4 = r4 * r31;
    r84 = fmaf(r51, r4, r84);
    r84 = fmaf(r43, r8, r84);
    r84 = fmaf(r43, r53, r84);
    r4 = r3 * r84;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx, r29, r49, r34, r4);
    r4 = r3 * r25;
    r4 = r4 * r0;
    r33 = r25 * r33;
    r34 = r2 * r33;
    r4 = fmaf(r71, r34, r80 * r4);
    r49 = r3 * r25;
    r49 = r49 * r0;
    r49 = fmaf(r90, r34, r63 * r49);
    r29 = r3 * r25;
    r29 = r29 * r0;
    r29 = fmaf(r75, r34, r65 * r29);
    r46 = r3 * r25;
    r46 = r46 * r0;
    r46 = fmaf(r81, r34, r91 * r46);
    WriteSum4<float, float>((float *)inout_shared, r4, r49, r29, r46);
  };
  FlushSumShared<4, float>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r3 * r25;
    r46 = r46 * r0;
    r46 = fmaf(r73, r34, r13 * r46);
    r29 = r3 * r25;
    r29 = r29 * r0;
    r34 = fmaf(r66, r34, r84 * r29);
    WriteSum2<float, float>((float *)inout_shared, r46, r34);
  };
  FlushSumShared<2, float>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r3 * r3;
    r46 = r80 * r34;
    r2 = r2 * r2;
    r29 = r71 * r2;
    r71 = fmaf(r71, r29, r80 * r46);
    r80 = r90 * r90;
    r49 = r63 * r63;
    r49 = fmaf(r34, r49, r2 * r80);
    r80 = r65 * r65;
    r4 = r75 * r75;
    r4 = fmaf(r2, r4, r34 * r80);
    r80 = r81 * r81;
    r35 = r91 * r91;
    r35 = fmaf(r34, r35, r2 * r80);
    WriteSum4<float, float>((float *)inout_shared, r71, r49, r4, r35);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r13 * r13;
    r4 = r73 * r73;
    r4 = fmaf(r2, r4, r34 * r35);
    r35 = r84 * r84;
    r49 = r66 * r66;
    r49 = fmaf(r2, r49, r34 * r35);
    WriteSum2<float, float>((float *)inout_shared, r4, r49);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = fmaf(r90, r29, r63 * r46);
    r4 = fmaf(r65, r46, r75 * r29);
    r35 = fmaf(r91, r46, r81 * r29);
    r71 = fmaf(r73, r29, r13 * r46);
    WriteSum4<float, float>((float *)inout_shared, r49, r4, r35, r71);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fmaf(r66, r29, r84 * r46);
    r46 = r90 * r75;
    r71 = r63 * r65;
    r71 = fmaf(r34, r71, r2 * r46);
    r46 = r63 * r91;
    r35 = r90 * r81;
    r35 = fmaf(r2, r35, r34 * r46);
    r46 = r63 * r13;
    r4 = r90 * r73;
    r4 = fmaf(r2, r4, r34 * r46);
    WriteSum4<float, float>((float *)inout_shared, r29, r71, r35, r4);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = r90 * r66;
    r35 = r63 * r84;
    r35 = fmaf(r34, r35, r2 * r4);
    r4 = r75 * r81;
    r71 = r65 * r91;
    r71 = fmaf(r34, r71, r2 * r4);
    r4 = r75 * r73;
    r29 = r65 * r13;
    r29 = fmaf(r34, r29, r2 * r4);
    r4 = r75 * r66;
    r46 = r65 * r84;
    r46 = fmaf(r34, r46, r2 * r4);
    WriteSum4<float, float>((float *)inout_shared, r35, r71, r29, r46);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r91 * r13;
    r29 = r81 * r73;
    r29 = fmaf(r2, r29, r34 * r46);
    r46 = r91 * r84;
    r71 = r81 * r66;
    r71 = fmaf(r2, r71, r34 * r46);
    r46 = r73 * r66;
    r35 = r13 * r84;
    r35 = fmaf(r34, r35, r2 * r46);
    WriteSum3<float, float>((float *)inout_shared, r29, r71, r35);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r25 * r0;
    WriteSum2<float, float>((float *)inout_shared, r33, r0);
  };
  FlushSumShared<2, float>(out_principal_point_njtr,
                           0 * out_principal_point_njtr_num_alloc,
                           principal_point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<float, float>((float *)inout_shared, r39, r39);
  };
  FlushSumShared<2, float>(out_principal_point_precond_diag,
                           0 * out_principal_point_precond_diag_num_alloc,
                           principal_point_indices_loc, (float *)inout_shared);
}

void OpencvSplitFixedFocalAndExtraFixedPointResJac(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *principal_point, unsigned int principal_point_num_alloc,
    SharedIndex *principal_point_indices, float *pixel,
    unsigned int pixel_num_alloc, float *focal_and_extra,
    unsigned int focal_and_extra_num_alloc, float *point,
    unsigned int point_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, float *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    float *out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    float *const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float *const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    float *const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvSplitFixedFocalAndExtraFixedPointResJacKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, sensor_from_rig,
      sensor_from_rig_num_alloc, principal_point, principal_point_num_alloc,
      principal_point_indices, pixel, pixel_num_alloc, focal_and_extra,
      focal_and_extra_num_alloc, point, point_num_alloc, out_res,
      out_res_num_alloc, out_pose_jac, out_pose_jac_num_alloc, out_pose_njtr,
      out_pose_njtr_num_alloc, out_pose_precond_diag,
      out_pose_precond_diag_num_alloc, out_pose_precond_tril,
      out_pose_precond_tril_num_alloc, out_principal_point_jac,
      out_principal_point_jac_num_alloc, out_principal_point_njtr,
      out_principal_point_njtr_num_alloc, out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar