#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_opencv_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJacKernel(
        float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
        float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        float *pixel, unsigned int pixel_num_alloc, float *focal_and_extra,
        unsigned int focal_and_extra_num_alloc, float *principal_point,
        unsigned int principal_point_num_alloc, float *point,
        unsigned int point_num_alloc, float *out_res,
        unsigned int out_res_num_alloc, float *const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float *const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float *const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
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
                                         global_thread_idx, r0, r1);
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
    r35 = r6 * r31;
    r47 = r14 * r34;
    r33 = fmaf(r47, r35, r33);
    r33 = fmaf(r44, r53, r33);
    r33 = fmaf(r2, r33, r0);
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r0, r35);
    r33 = fmaf(r0, r25, r33);
    r0 = r31 * r31;
    r0 = r0 * r9;
    r0 = fmaf(r51, r0, r24);
    r0 = fmaf(r31, r8, r6 * r0);
    r24 = r7 * r47;
    r0 = fmaf(r31, r24, r0);
    r0 = fmaf(r31, r53, r0);
    r0 = fmaf(r3, r0, r1);
    r0 = fmaf(r35, r25, r0);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r33, r0);
    r35 = r3 * r25;
    r1 = r23 * r38;
    r39 = r15 * r22;
    r42 = -5.00000000000000000e-01;
    r49 = r18 * r19;
    r49 = fmaf(r42, r49, r42 * r39);
    r39 = r17 * r20;
    r49 = fmaf(r42, r39, r49);
    r56 = r16 * r21;
    r57 = 5.00000000000000000e-01;
    r49 = fmaf(r57, r56, r49);
    r56 = r15 * r19;
    r39 = r17 * r21;
    r39 = fmaf(r42, r39, r42 * r56);
    r56 = r22 * r57;
    r58 = r16 * r42;
    r39 = fmaf(r18, r56, r39);
    r39 = fmaf(r20, r58, r39);
    r59 = r39 * r45;
    r1 = fmaf(r49, r1, r59);
    r60 = r14 * r27;
    r61 = fmaf(r57, r29, r22 * r58);
    r61 = fmaf(r42, r26, r61);
    r61 = fmaf(r42, r28, r61);
    r60 = r60 * r61;
    r62 = r14 * r30;
    r63 = r16 * r19;
    r64 = r15 * r20;
    r64 = fmaf(r42, r64, r57 * r63);
    r63 = r18 * r21;
    r64 = fmaf(r57, r63, r64);
    r64 = fmaf(r17, r56, r64);
    r62 = fmaf(r64, r62, r60);
    r1 = r1 + r62;
    r63 = r23 * r39;
    r65 = -4.00000000000000000e+00;
    r63 = r63 * r65;
    r66 = r30 * r61;
    r67 = r65 * r66;
    r68 = r63 + r67;
    r68 = fmaf(r12, r68, r13 * r1);
    r1 = r14 * r23;
    r1 = r1 * r64;
    r69 = r14 * r32;
    r69 = fmaf(r61, r69, r1);
    r70 = r14 * r27;
    r70 = r70 * r39;
    r71 = r14 * r30;
    r71 = fmaf(r49, r71, r70);
    r69 = r69 + r71;
    r68 = fmaf(r11, r69, r68);
    r69 = r31 * r68;
    r72 = 6.00000000000000000e+00;
    r69 = r69 * r72;
    r73 = r14 * r23;
    r73 = r73 * r49;
    r74 = r14 * r32;
    r74 = r74 * r39;
    r75 = r73 + r74;
    r62 = r62 + r75;
    r76 = r27 * r49;
    r77 = fmaf(r64, r45, r38 * r76);
    r78 = r14 * r23;
    r79 = r14 * r30;
    r79 = r79 * r39;
    r78 = fmaf(r61, r78, r79);
    r77 = r77 + r78;
    r77 = fmaf(r11, r77, r12 * r62);
    r62 = r27 * r64;
    r62 = r62 * r65;
    r63 = r63 + r62;
    r77 = fmaf(r13, r63, r77);
    r63 = -6.00000000000000000e+00;
    r80 = r31 * r63;
    r46 = r55 * r46;
    r46 = 1.0 / r46;
    r55 = r31 * r46;
    r80 = r80 * r55;
    r69 = fmaf(r77, r80, r51 * r69);
    r81 = r14 * r32;
    r81 = fmaf(r14, r76, r64 * r81);
    r81 = r81 + r78;
    r1 = r70 + r1;
    r70 = r30 * r38;
    r1 = fmaf(r49, r70, r1);
    r1 = fmaf(r61, r45, r1);
    r1 = fmaf(r12, r1, r13 * r81);
    r67 = r62 + r67;
    r1 = fmaf(r11, r67, r1);
    r67 = r38 * r44;
    r67 = r67 * r44;
    r67 = r67 * r77;
    r67 = fmaf(r46, r67, r1 * r47);
    r69 = r69 + r67;
    r69 = fmaf(r68, r8, r6 * r69);
    r62 = r25 * r31;
    r62 = r62 * r77;
    r69 = fmaf(r51, r62, r69);
    r81 = r14 * r31;
    r81 = r81 * r68;
    r70 = r38 * r31;
    r70 = r70 * r77;
    r70 = fmaf(r55, r70, r51 * r81);
    r67 = r67 + r70;
    r5 = r14 * r5;
    r67 = fmaf(r67, r5, r4 * r67);
    r81 = r31 * r67;
    r69 = fmaf(r8, r81, r69);
    r64 = r7 * r77;
    r82 = r44 * r65;
    r82 = r82 * r55;
    r69 = fmaf(r82, r64, r69);
    r83 = r7 * r14;
    r83 = r83 * r31;
    r83 = r83 * r1;
    r69 = fmaf(r51, r83, r69);
    r84 = r25 * r31;
    r84 = r84 * r48;
    r84 = r84 * r77;
    r69 = fmaf(r51, r84, r69);
    r69 = fmaf(r68, r24, r69);
    r69 = fmaf(r68, r53, r69);
    r35 = r35 * r0;
    r84 = r2 * r25;
    r84 = r84 * r33;
    r33 = r72 * r1;
    r83 = r44 * r44;
    r83 = r83 * r77;
    r83 = r83 * r63;
    r83 = fmaf(r46, r83, r34 * r33);
    r83 = r83 + r70;
    r70 = r25 * r48;
    r70 = r70 * r77;
    r70 = fmaf(r34, r70, r7 * r83);
    r83 = r6 * r68;
    r70 = fmaf(r47, r83, r70);
    r33 = r6 * r82;
    r64 = r6 * r14;
    r64 = r64 * r31;
    r64 = r64 * r1;
    r70 = fmaf(r51, r64, r70);
    r81 = r25 * r77;
    r70 = fmaf(r34, r81, r70);
    r62 = r44 * r67;
    r70 = fmaf(r8, r62, r70);
    r70 = fmaf(r1, r53, r70);
    r70 = fmaf(r77, r33, r70);
    r70 = fmaf(r1, r8, r70);
    r35 = fmaf(r70, r84, r69 * r35);
    r62 = r3 * r25;
    r81 = r38 * r27;
    r81 = fmaf(r61, r81, r59);
    r64 = r14 * r30;
    r83 = r17 * r22;
    r85 = r15 * r20;
    r85 = fmaf(r57, r85, r42 * r83);
    r83 = r18 * r21;
    r85 = fmaf(r42, r83, r85);
    r85 = fmaf(r19, r58, r85);
    r64 = r64 * r85;
    r83 = r14 * r23;
    r86 = r18 * r19;
    r87 = r17 * r20;
    r87 = fmaf(r57, r87, r57 * r86);
    r87 = fmaf(r15, r56, r87);
    r87 = fmaf(r21, r58, r87);
    r83 = fmaf(r87, r83, r64);
    r81 = r81 + r83;
    r58 = r14 * r27;
    r58 = r58 * r87;
    r86 = r14 * r32;
    r86 = fmaf(r85, r86, r58);
    r86 = r86 + r78;
    r86 = fmaf(r12, r86, r11 * r81);
    r81 = r27 * r39;
    r81 = r81 * r65;
    r78 = r23 * r85;
    r88 = r65 * r78;
    r89 = r81 + r88;
    r86 = fmaf(r13, r89, r86);
    r89 = r31 * r72;
    r58 = r79 + r58;
    r79 = r23 * r38;
    r58 = fmaf(r61, r79, r58);
    r58 = fmaf(r85, r45, r58);
    r79 = r14 * r32;
    r79 = fmaf(r14, r66, r87 * r79);
    r61 = r14 * r23;
    r61 = r61 * r39;
    r90 = r14 * r27;
    r90 = fmaf(r85, r90, r61);
    r79 = r79 + r90;
    r79 = fmaf(r11, r79, r13 * r58);
    r58 = r30 * r65;
    r58 = r58 * r87;
    r88 = r58 + r88;
    r79 = fmaf(r12, r88, r79);
    r89 = r89 * r79;
    r89 = fmaf(r51, r89, r86 * r80);
    r74 = r60 + r74;
    r74 = r74 + r83;
    r58 = r81 + r58;
    r58 = fmaf(r11, r58, r13 * r74);
    r87 = fmaf(r87, r45, r38 * r66);
    r87 = r87 + r90;
    r58 = fmaf(r12, r87, r58);
    r87 = r38 * r44;
    r87 = r87 * r44;
    r87 = r87 * r86;
    r87 = fmaf(r46, r87, r58 * r47);
    r89 = r89 + r87;
    r66 = r25 * r31;
    r66 = r66 * r48;
    r66 = r66 * r86;
    r66 = fmaf(r51, r66, r6 * r89);
    r89 = r38 * r31;
    r89 = r89 * r86;
    r74 = r14 * r31;
    r74 = r74 * r79;
    r74 = fmaf(r51, r74, r55 * r89);
    r87 = r87 + r74;
    r87 = fmaf(r87, r5, r4 * r87);
    r89 = r31 * r87;
    r66 = fmaf(r8, r89, r66);
    r81 = r25 * r31;
    r81 = r81 * r86;
    r66 = fmaf(r51, r81, r66);
    r83 = r7 * r86;
    r66 = fmaf(r82, r83, r66);
    r60 = r7 * r14;
    r60 = r60 * r31;
    r60 = r60 * r58;
    r66 = fmaf(r51, r60, r66);
    r66 = fmaf(r79, r8, r66);
    r66 = fmaf(r79, r24, r66);
    r66 = fmaf(r79, r53, r66);
    r62 = r62 * r0;
    r60 = r72 * r58;
    r83 = r44 * r44;
    r83 = r83 * r63;
    r83 = r83 * r86;
    r83 = fmaf(r46, r83, r34 * r60);
    r83 = r83 + r74;
    r83 = fmaf(r58, r53, r7 * r83);
    r74 = r25 * r86;
    r83 = fmaf(r34, r74, r83);
    r60 = r6 * r14;
    r60 = r60 * r31;
    r60 = r60 * r58;
    r83 = fmaf(r51, r60, r83);
    r81 = r6 * r79;
    r83 = fmaf(r47, r81, r83);
    r89 = r25 * r48;
    r89 = r89 * r86;
    r83 = fmaf(r34, r89, r83);
    r88 = r44 * r87;
    r83 = fmaf(r8, r88, r83);
    r83 = fmaf(r58, r8, r83);
    r83 = fmaf(r86, r33, r83);
    r62 = fmaf(r83, r84, r66 * r62);
    r88 = r3 * r25;
    r89 = r23 * r65;
    r29 = fmaf(r42, r29, r16 * r56);
    r29 = fmaf(r57, r26, r29);
    r29 = fmaf(r57, r28, r29);
    r89 = r89 * r29;
    r76 = r65 * r76;
    r28 = r89 + r76;
    r57 = r14 * r30;
    r57 = r57 * r29;
    r61 = r61 + r57;
    r26 = r38 * r27;
    r61 = fmaf(r85, r26, r61);
    r61 = fmaf(r49, r45, r61);
    r61 = fmaf(r11, r61, r13 * r28);
    r28 = r14 * r32;
    r28 = fmaf(r14, r78, r29 * r28);
    r28 = r28 + r71;
    r61 = fmaf(r12, r28, r61);
    r28 = r31 * r72;
    r26 = r14 * r27;
    r26 = r26 * r29;
    r64 = r64 + r26;
    r64 = r64 + r75;
    r45 = fmaf(r29, r45, r38 * r78);
    r45 = r45 + r71;
    r45 = fmaf(r13, r45, r11 * r64);
    r39 = r30 * r39;
    r39 = r39 * r65;
    r89 = r39 + r89;
    r45 = fmaf(r12, r89, r45);
    r28 = r28 * r45;
    r28 = fmaf(r51, r28, r61 * r80);
    r89 = r38 * r44;
    r89 = r89 * r44;
    r89 = r89 * r61;
    r26 = r73 + r26;
    r73 = r30 * r38;
    r26 = fmaf(r85, r73, r26);
    r26 = r26 + r59;
    r76 = r39 + r76;
    r76 = fmaf(r11, r76, r12 * r26);
    r11 = r14 * r32;
    r11 = fmaf(r49, r11, r57);
    r11 = r11 + r90;
    r76 = fmaf(r13, r11, r76);
    r89 = fmaf(r76, r47, r46 * r89);
    r28 = r28 + r89;
    r28 = fmaf(r45, r8, r6 * r28);
    r11 = r7 * r14;
    r11 = r11 * r31;
    r11 = r11 * r76;
    r28 = fmaf(r51, r11, r28);
    r13 = r7 * r61;
    r28 = fmaf(r82, r13, r28);
    r90 = r38 * r31;
    r90 = r90 * r61;
    r57 = r14 * r31;
    r57 = r57 * r45;
    r57 = fmaf(r51, r57, r55 * r90);
    r89 = r89 + r57;
    r89 = fmaf(r89, r5, r4 * r89);
    r90 = r31 * r89;
    r28 = fmaf(r8, r90, r28);
    r49 = r25 * r31;
    r49 = r49 * r61;
    r28 = fmaf(r51, r49, r28);
    r26 = r25 * r31;
    r26 = r26 * r48;
    r26 = r26 * r61;
    r28 = fmaf(r51, r26, r28);
    r28 = fmaf(r45, r24, r28);
    r28 = fmaf(r45, r53, r28);
    r88 = r88 * r0;
    r26 = r44 * r44;
    r26 = r26 * r63;
    r26 = r26 * r61;
    r49 = r72 * r76;
    r49 = fmaf(r34, r49, r46 * r26);
    r49 = r49 + r57;
    r49 = fmaf(r76, r8, r7 * r49);
    r57 = r25 * r48;
    r57 = r57 * r61;
    r49 = fmaf(r34, r57, r49);
    r26 = r25 * r61;
    r49 = fmaf(r34, r26, r49);
    r90 = r6 * r14;
    r90 = r90 * r31;
    r90 = r90 * r76;
    r49 = fmaf(r51, r90, r49);
    r13 = r6 * r45;
    r49 = fmaf(r47, r13, r49);
    r11 = r44 * r89;
    r49 = fmaf(r8, r11, r49);
    r49 = fmaf(r61, r33, r49);
    r49 = fmaf(r76, r53, r49);
    r88 = fmaf(r49, r84, r28 * r88);
    r11 = r3 * r25;
    r13 = r36 * r31;
    r13 = r13 * r72;
    r13 = fmaf(r51, r13, r52 * r80);
    r90 = r38 * r52;
    r90 = r90 * r44;
    r90 = r90 * r44;
    r90 = fmaf(r40, r47, r46 * r90);
    r13 = r13 + r90;
    r13 = fmaf(r36, r8, r6 * r13);
    r26 = r25 * r52;
    r26 = r26 * r31;
    r26 = r26 * r48;
    r13 = fmaf(r51, r26, r13);
    r57 = r7 * r14;
    r57 = r57 * r40;
    r57 = r57 * r31;
    r13 = fmaf(r51, r57, r13);
    r12 = r7 * r52;
    r13 = fmaf(r82, r12, r13);
    r39 = r38 * r52;
    r39 = r39 * r31;
    r59 = r14 * r36;
    r59 = r59 * r31;
    r59 = fmaf(r51, r59, r55 * r39);
    r90 = r90 + r59;
    r90 = fmaf(r90, r5, r4 * r90);
    r39 = r31 * r90;
    r13 = fmaf(r8, r39, r13);
    r73 = r25 * r52;
    r73 = r73 * r31;
    r13 = fmaf(r51, r73, r13);
    r13 = fmaf(r36, r53, r13);
    r13 = fmaf(r36, r24, r13);
    r11 = r11 * r0;
    r73 = r52 * r44;
    r73 = r73 * r44;
    r73 = r73 * r63;
    r39 = r40 * r72;
    r39 = fmaf(r34, r39, r46 * r73);
    r39 = r39 + r59;
    r39 = fmaf(r40, r8, r7 * r39);
    r59 = r6 * r14;
    r59 = r59 * r40;
    r59 = r59 * r31;
    r39 = fmaf(r51, r59, r39);
    r73 = r44 * r90;
    r39 = fmaf(r8, r73, r39);
    r12 = r25 * r52;
    r39 = fmaf(r34, r12, r39);
    r57 = r6 * r36;
    r39 = fmaf(r47, r57, r39);
    r26 = r25 * r52;
    r26 = r26 * r48;
    r39 = fmaf(r34, r26, r39);
    r39 = fmaf(r52, r33, r39);
    r39 = fmaf(r40, r53, r39);
    r11 = fmaf(r39, r84, r13 * r11);
    WriteSum4<float, float>((float *)inout_shared, r35, r62, r88, r11);
  };
  FlushSumShared<4, float>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = r3 * r25;
    r88 = r41 * r31;
    r88 = r88 * r72;
    r88 = fmaf(r51, r88, r37 * r80);
    r62 = r38 * r37;
    r62 = r62 * r44;
    r62 = r62 * r44;
    r62 = fmaf(r10, r47, r46 * r62);
    r88 = r88 + r62;
    r88 = fmaf(r41, r8, r6 * r88);
    r35 = r7 * r14;
    r35 = r35 * r10;
    r35 = r35 * r31;
    r88 = fmaf(r51, r35, r88);
    r26 = r7 * r37;
    r88 = fmaf(r82, r26, r88);
    r57 = r25 * r37;
    r57 = r57 * r31;
    r88 = fmaf(r51, r57, r88);
    r12 = r38 * r37;
    r12 = r12 * r31;
    r73 = r14 * r41;
    r73 = r73 * r31;
    r73 = fmaf(r51, r73, r55 * r12);
    r62 = r62 + r73;
    r62 = fmaf(r62, r5, r4 * r62);
    r12 = r31 * r62;
    r88 = fmaf(r8, r12, r88);
    r59 = r25 * r37;
    r59 = r59 * r31;
    r59 = r59 * r48;
    r88 = fmaf(r51, r59, r88);
    r88 = fmaf(r41, r24, r88);
    r88 = fmaf(r41, r53, r88);
    r11 = r11 * r0;
    r59 = r37 * r44;
    r59 = r59 * r44;
    r59 = r59 * r63;
    r12 = r10 * r72;
    r12 = fmaf(r34, r12, r46 * r59);
    r12 = r12 + r73;
    r73 = r25 * r37;
    r73 = r73 * r48;
    r73 = fmaf(r34, r73, r7 * r12);
    r12 = r6 * r14;
    r12 = r12 * r10;
    r12 = r12 * r31;
    r73 = fmaf(r51, r12, r73);
    r59 = r6 * r41;
    r73 = fmaf(r47, r59, r73);
    r57 = r44 * r62;
    r73 = fmaf(r8, r57, r73);
    r26 = r25 * r37;
    r73 = fmaf(r34, r26, r73);
    r73 = fmaf(r10, r8, r73);
    r73 = fmaf(r10, r53, r73);
    r73 = fmaf(r37, r33, r73);
    r11 = fmaf(r73, r84, r88 * r11);
    r26 = r3 * r25;
    r57 = r43 * r31;
    r57 = r57 * r72;
    r80 = fmaf(r54, r80, r51 * r57);
    r57 = r38 * r54;
    r57 = r57 * r44;
    r57 = r57 * r44;
    r57 = fmaf(r46, r57, r50 * r47);
    r80 = r80 + r57;
    r59 = r7 * r54;
    r59 = fmaf(r82, r59, r6 * r80);
    r80 = r25 * r54;
    r80 = r80 * r31;
    r80 = r80 * r48;
    r59 = fmaf(r51, r80, r59);
    r82 = r25 * r54;
    r82 = r82 * r31;
    r59 = fmaf(r51, r82, r59);
    r12 = r14 * r43;
    r12 = r12 * r31;
    r35 = r38 * r54;
    r35 = r35 * r31;
    r35 = fmaf(r55, r35, r51 * r12);
    r57 = r57 + r35;
    r5 = fmaf(r57, r5, r4 * r57);
    r57 = r31 * r5;
    r59 = fmaf(r8, r57, r59);
    r4 = r7 * r14;
    r4 = r4 * r50;
    r4 = r4 * r31;
    r59 = fmaf(r51, r4, r59);
    r59 = fmaf(r43, r8, r59);
    r59 = fmaf(r43, r53, r59);
    r59 = fmaf(r43, r24, r59);
    r26 = r26 * r0;
    r0 = r50 * r72;
    r4 = r54 * r44;
    r4 = r4 * r44;
    r4 = r4 * r63;
    r4 = fmaf(r46, r4, r34 * r0);
    r4 = r4 + r35;
    r33 = fmaf(r54, r33, r7 * r4);
    r4 = r6 * r43;
    r33 = fmaf(r47, r4, r33);
    r47 = r44 * r5;
    r33 = fmaf(r8, r47, r33);
    r35 = r25 * r54;
    r35 = r35 * r48;
    r33 = fmaf(r34, r35, r33);
    r0 = r25 * r54;
    r33 = fmaf(r34, r0, r33);
    r34 = r6 * r14;
    r34 = r34 * r50;
    r34 = r34 * r31;
    r33 = fmaf(r51, r34, r33);
    r33 = fmaf(r50, r8, r33);
    r33 = fmaf(r50, r53, r33);
    r84 = fmaf(r33, r84, r59 * r26);
    WriteSum2<float, float>((float *)inout_shared, r11, r84);
  };
  FlushSumShared<2, float>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r84 = r69 * r69;
    r11 = r3 * r3;
    r2 = r2 * r2;
    r26 = r70 * r70;
    r26 = fmaf(r2, r26, r11 * r84);
    r84 = r83 * r83;
    r34 = r66 * r66;
    r34 = fmaf(r11, r34, r2 * r84);
    r84 = r28 * r11;
    r0 = r49 * r2;
    r49 = fmaf(r49, r0, r28 * r84);
    r28 = r39 * r39;
    r35 = r13 * r13;
    r35 = fmaf(r11, r35, r2 * r28);
    WriteSum4<float, float>((float *)inout_shared, r26, r34, r49, r35);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r88 * r88;
    r49 = r73 * r73;
    r49 = fmaf(r2, r49, r11 * r35);
    r35 = r59 * r59;
    r34 = r33 * r33;
    r34 = fmaf(r2, r34, r11 * r35);
    WriteSum2<float, float>((float *)inout_shared, r49, r34);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r69 * r66;
    r49 = r70 * r83;
    r49 = fmaf(r2, r49, r11 * r34);
    r34 = fmaf(r69, r84, r70 * r0);
    r35 = r70 * r39;
    r26 = r69 * r13;
    r26 = fmaf(r11, r26, r2 * r35);
    r35 = r69 * r88;
    r28 = r70 * r73;
    r28 = fmaf(r2, r28, r11 * r35);
    WriteSum4<float, float>((float *)inout_shared, r49, r34, r26, r28);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = r69 * r59;
    r26 = r70 * r33;
    r26 = fmaf(r2, r26, r11 * r28);
    r28 = fmaf(r66, r84, r83 * r0);
    r34 = r66 * r13;
    r49 = r83 * r39;
    r49 = fmaf(r2, r49, r11 * r34);
    r34 = r66 * r88;
    r35 = r83 * r73;
    r35 = fmaf(r2, r35, r11 * r34);
    WriteSum4<float, float>((float *)inout_shared, r26, r28, r49, r35);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r83 * r33;
    r49 = r66 * r59;
    r49 = fmaf(r11, r49, r2 * r35);
    r35 = fmaf(r13, r84, r39 * r0);
    r28 = fmaf(r88, r84, r73 * r0);
    r84 = fmaf(r59, r84, r33 * r0);
    WriteSum4<float, float>((float *)inout_shared, r49, r35, r28, r84);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r84 = r13 * r88;
    r28 = r39 * r73;
    r28 = fmaf(r2, r28, r11 * r84);
    r84 = r13 * r59;
    r35 = r39 * r33;
    r35 = fmaf(r2, r35, r11 * r84);
    r84 = r73 * r33;
    r49 = r88 * r59;
    r49 = fmaf(r11, r49, r2 * r84);
    WriteSum3<float, float>((float *)inout_shared, r28, r35, r49);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
}

void OpencvSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJac(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *pixel, unsigned int pixel_num_alloc, float *focal_and_extra,
    unsigned int focal_and_extra_num_alloc, float *principal_point,
    unsigned int principal_point_num_alloc, float *point,
    unsigned int point_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, float *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJacKernel<<<
      n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, sensor_from_rig,
      sensor_from_rig_num_alloc, pixel, pixel_num_alloc, focal_and_extra,
      focal_and_extra_num_alloc, principal_point, principal_point_num_alloc,
      point, point_num_alloc, out_res, out_res_num_alloc, out_pose_njtr,
      out_pose_njtr_num_alloc, out_pose_precond_diag,
      out_pose_precond_diag_num_alloc, out_pose_precond_tril,
      out_pose_precond_tril_num_alloc, problem_size);
}

} // namespace caspar