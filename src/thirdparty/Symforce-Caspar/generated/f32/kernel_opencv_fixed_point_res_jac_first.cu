#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_opencv_fixed_point_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) OpencvFixedPointResJacFirstKernel(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    float *pixel, unsigned int pixel_num_alloc, float *point,
    unsigned int point_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *const out_rTr, float *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, float *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, float *out_calib_jac,
    unsigned int out_calib_jac_num_alloc, float *const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc, float *const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    float *const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102;
  LoadShared<4, float, float>(calib, 4 * calib_num_alloc, calib_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       calib_indices_loc[threadIdx.x].target, r0, r1, r2, r3);
  };
  __syncthreads();
  LoadShared<4, float, float>(calib, 0 * calib_num_alloc, calib_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       calib_indices_loc[threadIdx.x].target, r4, r5, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r8, r9, r10);
    ReadIdx3<1024, float, float, float4>(point, 0 * point_num_alloc,
                                         global_thread_idx, r11, r12, r13);
  };
  LoadShared<4, float, float>(pose, 0 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r14, r15, r16,
                       r17);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r18, r19, r20, r21);
    r22 = fmaf(r15, r18, r16 * r21);
    r23 = r14 * r19;
    r24 = -1.00000000000000000e+00;
    r22 = fmaf(r24, r23, r22);
    r22 = fmaf(r17, r20, r22);
    r23 = r22 * r22;
    r25 = -2.00000000000000000e+00;
    r23 = r23 * r25;
    r26 = 1.00000000000000000e+00;
    r27 = fmaf(r17, r19, r15 * r21);
    r28 = r14 * r20;
    r29 = r16 * r18;
    r27 = r27 + r28;
    r27 = fmaf(r24, r29, r27);
    r30 = r25 * r27;
    r30 = fmaf(r27, r30, r26);
    r31 = r23 + r30;
    r31 = fmaf(r11, r31, r8);
    r8 = 2.00000000000000000e+00;
    r32 = fmaf(r17, r18, r14 * r21);
    r33 = r15 * r20;
    r32 = fmaf(r24, r33, r32);
    r32 = fmaf(r16, r19, r32);
    r33 = r8 * r32;
    r33 = r33 * r27;
    r34 = fmaf(r15, r19, r14 * r18);
    r34 = fmaf(r16, r20, r34);
    r34 = fmaf(r24, r34, r17 * r21);
    r35 = r25 * r34;
    r36 = fmaf(r22, r35, r33);
    r37 = r8 * r22;
    r37 = r37 * r32;
    r38 = r8 * r27;
    r38 = fmaf(r34, r38, r37);
  };
  LoadShared<3, float, float>(pose, 4 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r39, r40, r41);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r42 = r18 * r20;
    r42 = r42 * r8;
    r43 = r19 * r21;
    r44 = fmaf(r8, r43, r42);
    r45 = r20 * r21;
    r46 = r18 * r19;
    r46 = r46 * r8;
    r45 = fmaf(r25, r45, r46);
    r47 = r19 * r19;
    r47 = r47 * r25;
    r48 = r26 + r47;
    r49 = r20 * r20;
    r49 = r49 * r25;
    r48 = r48 + r49;
    r31 = fmaf(r12, r36, r31);
    r31 = fmaf(r13, r38, r31);
    r31 = fmaf(r41, r44, r31);
    r31 = fmaf(r40, r45, r31);
    r31 = fmaf(r39, r48, r31);
    r38 = 3.00000000000000000e+00;
    r36 = r31 * r38;
    r50 = 9.99999999999999955e-07;
    r37 = fmaf(r27, r35, r37);
    r37 = fmaf(r11, r37, r10);
    r43 = fmaf(r25, r43, r42);
    r47 = r26 + r47;
    r42 = r18 * r18;
    r42 = r42 * r25;
    r47 = r47 + r42;
    r10 = r19 * r20;
    r10 = r10 * r8;
    r51 = r18 * r21;
    r51 = fmaf(r8, r51, r10);
    r52 = r8 * r22;
    r52 = r52 * r27;
    r53 = r8 * r32;
    r53 = fmaf(r34, r53, r52);
    r54 = r32 * r32;
    r54 = r54 * r25;
    r30 = r54 + r30;
    r37 = fmaf(r39, r43, r37);
    r37 = fmaf(r41, r47, r37);
    r37 = fmaf(r40, r51, r37);
    r37 = fmaf(r12, r53, r37);
    r37 = fmaf(r13, r30, r37);
    r30 = copysign(1.0, r37);
    r30 = fmaf(r50, r30, r37);
    r50 = r30 * r30;
    r37 = 1.0 / r50;
    r53 = r31 * r37;
    r55 = r8 * r22;
    r55 = fmaf(r34, r55, r33);
    r55 = fmaf(r11, r55, r9);
    r9 = r20 * r21;
    r9 = fmaf(r8, r9, r46);
    r49 = r26 + r49;
    r49 = r49 + r42;
    r42 = r18 * r21;
    r42 = fmaf(r25, r42, r10);
    r52 = fmaf(r32, r35, r52);
    r23 = r26 + r23;
    r23 = r23 + r54;
    r55 = fmaf(r39, r9, r55);
    r55 = fmaf(r40, r49, r55);
    r55 = fmaf(r41, r42, r55);
    r55 = fmaf(r13, r52, r55);
    r55 = fmaf(r12, r23, r55);
    r23 = r55 * r55;
    r23 = r23 * r37;
    r36 = fmaf(r53, r36, r23);
    r52 = 1.0 / r30;
    r41 = fmaf(r31, r52, r1 * r36);
    r40 = r31 * r53;
    r23 = r23 + r40;
    r39 = r23 * r23;
    r54 = fmaf(r7, r39, r6 * r23);
    r10 = r54 * r52;
    r46 = r0 * r53;
    r33 = r8 * r55;
    r41 = fmaf(r33, r46, r41);
    r41 = fmaf(r31, r10, r41);
    r46 = r4 * r41;
    r2 = r2 + r46;
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r56, r57);
    r2 = fmaf(r56, r24, r2);
    r56 = r38 * r55;
    r56 = r56 * r55;
    r56 = fmaf(r37, r56, r40);
    r40 = fmaf(r55, r52, r0 * r56);
    r58 = r1 * r53;
    r40 = fmaf(r33, r58, r40);
    r40 = fmaf(r55, r10, r40);
    r58 = r5 * r40;
    r3 = r3 + r58;
    r3 = fmaf(r57, r24, r3);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r2, r3);
    r57 = fmaf(r3, r3, r2 * r2);
  };
  SumStore<float>(out_rTr_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r57);
  if (global_thread_idx < problem_size) {
    r57 = r8 * r34;
    r59 = r16 * r21;
    r60 = 5.00000000000000000e-01;
    r61 = r15 * r18;
    r61 = fmaf(r60, r61, r60 * r59);
    r59 = r14 * r19;
    r62 = -5.00000000000000000e-01;
    r61 = fmaf(r62, r59, r61);
    r63 = r17 * r60;
    r61 = fmaf(r20, r63, r61);
    r59 = r14 * r21;
    r64 = r17 * r18;
    r64 = fmaf(r62, r64, r62 * r59);
    r59 = r16 * r19;
    r64 = fmaf(r62, r59, r64);
    r65 = r15 * r20;
    r64 = fmaf(r60, r65, r64);
    r65 = r27 * r64;
    r57 = fmaf(r8, r65, r61 * r57);
    r59 = r8 * r32;
    r66 = r17 * r19;
    r67 = r15 * r62;
    r66 = fmaf(r21, r67, r62 * r66);
    r66 = fmaf(r60, r29, r66);
    r66 = fmaf(r62, r28, r66);
    r68 = r8 * r22;
    r69 = r14 * r18;
    r70 = r16 * r20;
    r70 = fmaf(r62, r70, r62 * r69);
    r70 = fmaf(r21, r63, r70);
    r70 = fmaf(r19, r67, r70);
    r68 = r68 * r70;
    r59 = fmaf(r66, r59, r68);
    r57 = r57 + r59;
    r69 = r8 * r27;
    r69 = r69 * r70;
    r71 = r8 * r32;
    r71 = r71 * r61;
    r72 = r69 + r71;
    r73 = r22 * r25;
    r72 = fmaf(r64, r73, r72);
    r72 = fmaf(r66, r35, r72);
    r72 = fmaf(r12, r72, r13 * r57);
    r57 = r27 * r61;
    r73 = -4.00000000000000000e+00;
    r57 = r57 * r73;
    r74 = r22 * r66;
    r75 = r73 * r74;
    r76 = r57 + r75;
    r72 = fmaf(r11, r76, r72);
    r76 = 6.00000000000000000e+00;
    r77 = r72 * r76;
    r78 = r31 * r31;
    r79 = r8 * r27;
    r79 = r79 * r66;
    r80 = r8 * r22;
    r80 = fmaf(r61, r80, r79);
    r81 = r8 * r32;
    r81 = r81 * r64;
    r82 = r8 * r34;
    r82 = r82 * r70;
    r83 = r81 + r82;
    r84 = r80 + r83;
    r61 = fmaf(r61, r35, r25 * r65);
    r61 = r61 + r59;
    r61 = fmaf(r11, r61, r12 * r84);
    r84 = r32 * r70;
    r84 = r84 * r73;
    r57 = r57 + r84;
    r61 = fmaf(r13, r57, r61);
    r57 = -6.00000000000000000e+00;
    r85 = r61 * r57;
    r50 = r30 * r50;
    r86 = 1.0 / r50;
    r85 = r85 * r86;
    r78 = fmaf(r85, r78, r53 * r77);
    r77 = r32 * r25;
    r87 = r70 * r35;
    r77 = fmaf(r64, r77, r87);
    r77 = r77 + r80;
    r75 = r84 + r75;
    r75 = fmaf(r12, r75, r13 * r77);
    r77 = r8 * r34;
    r77 = fmaf(r66, r77, r71);
    r71 = r8 * r22;
    r71 = fmaf(r64, r71, r69);
    r77 = r77 + r71;
    r75 = fmaf(r11, r77, r75);
    r77 = r37 * r33;
    r69 = r25 * r55;
    r69 = r69 * r55;
    r69 = r69 * r61;
    r69 = fmaf(r86, r69, r75 * r77);
    r78 = r78 + r69;
    r78 = fmaf(r72, r10, r1 * r78);
    r84 = r0 * r72;
    r78 = fmaf(r77, r84, r78);
    r80 = r24 * r53;
    r88 = r54 * r80;
    r89 = r7 * r8;
    r90 = r8 * r72;
    r91 = r25 * r31;
    r91 = r91 * r31;
    r91 = r91 * r61;
    r91 = fmaf(r86, r91, r53 * r90);
    r69 = r69 + r91;
    r89 = r89 * r23;
    r69 = fmaf(r6, r69, r69 * r89);
    r89 = r31 * r69;
    r78 = fmaf(r52, r89, r78);
    r90 = r31 * r55;
    r90 = r90 * r73;
    r90 = r90 * r86;
    r92 = r0 * r90;
    r93 = r0 * r8;
    r93 = r93 * r75;
    r78 = fmaf(r53, r93, r78);
    r78 = fmaf(r61, r88, r78);
    r78 = fmaf(r61, r92, r78);
    r78 = fmaf(r61, r80, r78);
    r78 = fmaf(r72, r52, r78);
    r93 = r4 * r78;
    r89 = r55 * r75;
    r89 = r89 * r76;
    r84 = r55 * r55;
    r84 = fmaf(r85, r84, r37 * r89);
    r84 = r84 + r91;
    r91 = r1 * r77;
    r84 = fmaf(r72, r91, r0 * r84);
    r89 = r24 * r55;
    r89 = r89 * r61;
    r84 = fmaf(r37, r89, r84);
    r85 = r24 * r55;
    r85 = r85 * r54;
    r85 = r85 * r61;
    r84 = fmaf(r37, r85, r84);
    r94 = r55 * r69;
    r84 = fmaf(r52, r94, r84);
    r95 = r1 * r61;
    r84 = fmaf(r90, r95, r84);
    r96 = r1 * r8;
    r96 = r96 * r75;
    r84 = fmaf(r53, r96, r84);
    r84 = fmaf(r75, r10, r84);
    r84 = fmaf(r75, r52, r84);
    r96 = r5 * r84;
    r82 = r79 + r82;
    r79 = r8 * r22;
    r95 = r16 * r21;
    r94 = r14 * r19;
    r94 = fmaf(r60, r94, r62 * r95);
    r95 = r17 * r20;
    r94 = fmaf(r62, r95, r94);
    r94 = fmaf(r18, r67, r94);
    r79 = r79 * r94;
    r95 = r8 * r32;
    r85 = r14 * r21;
    r89 = r16 * r19;
    r89 = fmaf(r60, r89, r60 * r85);
    r89 = fmaf(r18, r63, r89);
    r89 = fmaf(r20, r67, r89);
    r95 = fmaf(r89, r95, r79);
    r82 = r82 + r95;
    r67 = r22 * r73;
    r67 = r67 * r89;
    r85 = r27 * r70;
    r85 = r85 * r73;
    r97 = r67 + r85;
    r97 = fmaf(r11, r97, r13 * r82);
    r82 = fmaf(r89, r35, r25 * r74);
    r98 = r8 * r32;
    r98 = r98 * r70;
    r99 = r8 * r27;
    r99 = fmaf(r94, r99, r98);
    r82 = r82 + r99;
    r97 = fmaf(r12, r82, r97);
    r82 = r76 * r97;
    r100 = r31 * r31;
    r101 = r25 * r27;
    r101 = fmaf(r66, r101, r87);
    r101 = r101 + r95;
    r95 = r8 * r27;
    r95 = r95 * r89;
    r102 = r8 * r34;
    r102 = fmaf(r94, r102, r95);
    r102 = r102 + r59;
    r102 = fmaf(r12, r102, r11 * r101);
    r101 = r32 * r94;
    r59 = r73 * r101;
    r85 = r85 + r59;
    r102 = fmaf(r13, r85, r102);
    r100 = r100 * r57;
    r100 = r100 * r102;
    r100 = fmaf(r86, r100, r53 * r82);
    r82 = r25 * r55;
    r82 = r82 * r55;
    r82 = r82 * r102;
    r85 = r32 * r25;
    r85 = fmaf(r66, r85, r68);
    r85 = r85 + r95;
    r85 = fmaf(r94, r35, r85);
    r95 = r8 * r34;
    r74 = fmaf(r8, r74, r89 * r95);
    r74 = r74 + r99;
    r74 = fmaf(r11, r74, r13 * r85);
    r59 = r67 + r59;
    r74 = fmaf(r12, r59, r74);
    r82 = fmaf(r74, r77, r86 * r82);
    r100 = r100 + r82;
    r59 = r0 * r8;
    r59 = r59 * r74;
    r59 = fmaf(r53, r59, r1 * r100);
    r100 = r8 * r97;
    r67 = r25 * r31;
    r67 = r67 * r31;
    r67 = r67 * r102;
    r67 = fmaf(r86, r67, r53 * r100);
    r82 = r82 + r67;
    r100 = r7 * r8;
    r100 = r100 * r23;
    r100 = fmaf(r82, r100, r6 * r82);
    r82 = r31 * r100;
    r59 = fmaf(r52, r82, r59);
    r85 = r0 * r97;
    r59 = fmaf(r77, r85, r59);
    r59 = fmaf(r102, r80, r59);
    r59 = fmaf(r97, r52, r59);
    r59 = fmaf(r102, r92, r59);
    r59 = fmaf(r97, r10, r59);
    r59 = fmaf(r102, r88, r59);
    r85 = r4 * r59;
    r82 = r55 * r55;
    r82 = r82 * r57;
    r82 = r82 * r102;
    r95 = r55 * r76;
    r95 = r95 * r74;
    r95 = fmaf(r37, r95, r86 * r82);
    r95 = r95 + r67;
    r67 = r1 * r8;
    r67 = r67 * r74;
    r67 = fmaf(r53, r67, r0 * r95);
    r95 = r24 * r55;
    r95 = r95 * r102;
    r67 = fmaf(r37, r95, r67);
    r82 = r1 * r102;
    r67 = fmaf(r90, r82, r67);
    r89 = r24 * r55;
    r89 = r89 * r54;
    r89 = r89 * r102;
    r67 = fmaf(r37, r89, r67);
    r68 = r55 * r100;
    r67 = fmaf(r52, r68, r67);
    r67 = fmaf(r74, r10, r67);
    r67 = fmaf(r74, r52, r67);
    r67 = fmaf(r97, r91, r67);
    r68 = r5 * r67;
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r93, r96,
        r85, r68);
    r68 = r31 * r31;
    r85 = r32 * r73;
    r96 = r15 * r21;
    r29 = fmaf(r62, r29, r60 * r96);
    r29 = fmaf(r19, r63, r29);
    r29 = fmaf(r60, r28, r29);
    r85 = r85 * r29;
    r65 = r73 * r65;
    r28 = r85 + r65;
    r60 = r8 * r22;
    r60 = r60 * r29;
    r98 = r98 + r60;
    r63 = r25 * r27;
    r98 = fmaf(r94, r63, r98);
    r98 = fmaf(r64, r35, r98);
    r98 = fmaf(r11, r98, r13 * r28);
    r28 = r8 * r34;
    r28 = fmaf(r8, r101, r29 * r28);
    r28 = r28 + r71;
    r98 = fmaf(r12, r28, r98);
    r68 = r68 * r57;
    r68 = r68 * r98;
    r28 = r8 * r27;
    r28 = r28 * r29;
    r81 = r81 + r28;
    r63 = r22 * r25;
    r81 = fmaf(r94, r63, r81);
    r81 = r81 + r87;
    r70 = r22 * r70;
    r70 = r70 * r73;
    r65 = r70 + r65;
    r65 = fmaf(r11, r65, r12 * r81);
    r81 = r8 * r34;
    r81 = fmaf(r64, r81, r60);
    r81 = r81 + r99;
    r65 = fmaf(r13, r81, r65);
    r81 = r76 * r65;
    r81 = fmaf(r53, r81, r86 * r68);
    r68 = r25 * r55;
    r68 = r68 * r55;
    r68 = r68 * r98;
    r28 = r79 + r28;
    r28 = r28 + r83;
    r35 = fmaf(r29, r35, r25 * r101);
    r35 = r35 + r71;
    r35 = fmaf(r13, r35, r11 * r28);
    r85 = r70 + r85;
    r35 = fmaf(r12, r85, r35);
    r68 = fmaf(r35, r77, r86 * r68);
    r81 = r81 + r68;
    r85 = r0 * r8;
    r85 = r85 * r35;
    r85 = fmaf(r53, r85, r1 * r81);
    r81 = r7 * r8;
    r12 = r25 * r31;
    r12 = r12 * r31;
    r12 = r12 * r98;
    r70 = r8 * r65;
    r70 = fmaf(r53, r70, r86 * r12);
    r68 = r68 + r70;
    r81 = r81 * r23;
    r68 = fmaf(r6, r68, r68 * r81);
    r81 = r31 * r68;
    r85 = fmaf(r52, r81, r85);
    r12 = r0 * r65;
    r85 = fmaf(r77, r12, r85);
    r85 = fmaf(r65, r52, r85);
    r85 = fmaf(r98, r80, r85);
    r85 = fmaf(r98, r92, r85);
    r85 = fmaf(r98, r88, r85);
    r85 = fmaf(r65, r10, r85);
    r12 = r4 * r85;
    r81 = r55 * r55;
    r81 = r81 * r57;
    r81 = r81 * r98;
    r13 = r55 * r76;
    r13 = r13 * r35;
    r13 = fmaf(r37, r13, r86 * r81);
    r13 = r13 + r70;
    r13 = fmaf(r35, r10, r0 * r13);
    r70 = r1 * r8;
    r70 = r70 * r35;
    r13 = fmaf(r53, r70, r13);
    r81 = r1 * r98;
    r13 = fmaf(r90, r81, r13);
    r28 = r55 * r68;
    r13 = fmaf(r52, r28, r13);
    r11 = r24 * r55;
    r11 = r11 * r54;
    r11 = r11 * r98;
    r13 = fmaf(r37, r11, r13);
    r71 = r24 * r55;
    r71 = r71 * r98;
    r13 = fmaf(r37, r71, r13);
    r13 = fmaf(r35, r52, r13);
    r13 = fmaf(r65, r91, r13);
    r71 = r5 * r13;
    r11 = r43 * r31;
    r11 = r11 * r31;
    r11 = r11 * r57;
    r28 = r48 * r76;
    r28 = fmaf(r53, r28, r86 * r11);
    r11 = r25 * r43;
    r11 = r11 * r55;
    r11 = r11 * r55;
    r11 = fmaf(r9, r77, r86 * r11);
    r28 = r28 + r11;
    r28 = fmaf(r48, r52, r1 * r28);
    r81 = r7 * r8;
    r35 = r25 * r43;
    r35 = r35 * r31;
    r35 = r35 * r31;
    r70 = r8 * r48;
    r70 = fmaf(r53, r70, r86 * r35);
    r11 = r11 + r70;
    r81 = r81 * r23;
    r11 = fmaf(r6, r11, r11 * r81);
    r81 = r31 * r11;
    r28 = fmaf(r52, r81, r28);
    r35 = r0 * r8;
    r35 = r35 * r9;
    r28 = fmaf(r53, r35, r28);
    r29 = r0 * r48;
    r28 = fmaf(r77, r29, r28);
    r28 = fmaf(r48, r10, r28);
    r28 = fmaf(r43, r80, r28);
    r28 = fmaf(r43, r88, r28);
    r28 = fmaf(r43, r92, r28);
    r29 = r4 * r28;
    r35 = r43 * r55;
    r35 = r35 * r55;
    r35 = r35 * r57;
    r81 = r9 * r55;
    r81 = r81 * r76;
    r81 = fmaf(r37, r81, r86 * r35);
    r81 = r81 + r70;
    r70 = r55 * r11;
    r70 = fmaf(r52, r70, r0 * r81);
    r81 = r24 * r43;
    r81 = r81 * r55;
    r81 = r81 * r54;
    r70 = fmaf(r37, r81, r70);
    r35 = r24 * r43;
    r35 = r35 * r55;
    r70 = fmaf(r37, r35, r70);
    r101 = r1 * r43;
    r70 = fmaf(r90, r101, r70);
    r83 = r1 * r8;
    r83 = r83 * r9;
    r70 = fmaf(r53, r83, r70);
    r70 = fmaf(r9, r52, r70);
    r70 = fmaf(r9, r10, r70);
    r70 = fmaf(r48, r91, r70);
    r83 = r5 * r70;
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r12, r71,
        r29, r83);
    r83 = r51 * r31;
    r83 = r83 * r31;
    r83 = r83 * r57;
    r29 = r45 * r76;
    r29 = fmaf(r53, r29, r86 * r83);
    r83 = r25 * r51;
    r83 = r83 * r55;
    r83 = r83 * r55;
    r83 = fmaf(r49, r77, r86 * r83);
    r29 = r29 + r83;
    r29 = fmaf(r45, r52, r1 * r29);
    r71 = r25 * r51;
    r71 = r71 * r31;
    r71 = r71 * r31;
    r12 = r8 * r45;
    r12 = fmaf(r53, r12, r86 * r71);
    r83 = r83 + r12;
    r71 = r7 * r8;
    r71 = r71 * r23;
    r71 = fmaf(r83, r71, r6 * r83);
    r83 = r31 * r71;
    r29 = fmaf(r52, r83, r29);
    r101 = r0 * r45;
    r29 = fmaf(r77, r101, r29);
    r35 = r0 * r8;
    r35 = r35 * r49;
    r29 = fmaf(r53, r35, r29);
    r29 = fmaf(r51, r92, r29);
    r29 = fmaf(r45, r10, r29);
    r29 = fmaf(r51, r80, r29);
    r29 = fmaf(r51, r88, r29);
    r35 = r4 * r29;
    r101 = r51 * r55;
    r101 = r101 * r55;
    r101 = r101 * r57;
    r83 = r49 * r55;
    r83 = r83 * r76;
    r83 = fmaf(r37, r83, r86 * r101);
    r83 = r83 + r12;
    r83 = fmaf(r49, r52, r0 * r83);
    r12 = r1 * r51;
    r83 = fmaf(r90, r12, r83);
    r101 = r55 * r71;
    r83 = fmaf(r52, r101, r83);
    r81 = r24 * r51;
    r81 = r81 * r55;
    r81 = r81 * r54;
    r83 = fmaf(r37, r81, r83);
    r79 = r1 * r8;
    r79 = r79 * r49;
    r83 = fmaf(r53, r79, r83);
    r99 = r24 * r51;
    r99 = r99 * r55;
    r83 = fmaf(r37, r99, r83);
    r83 = fmaf(r49, r10, r83);
    r83 = fmaf(r45, r91, r83);
    r99 = r5 * r83;
    r79 = r44 * r76;
    r81 = r47 * r31;
    r81 = r81 * r31;
    r81 = r81 * r57;
    r81 = fmaf(r86, r81, r53 * r79);
    r79 = r25 * r47;
    r79 = r79 * r55;
    r79 = r79 * r55;
    r79 = fmaf(r86, r79, r42 * r77);
    r81 = r81 + r79;
    r101 = r8 * r44;
    r12 = r25 * r47;
    r12 = r12 * r31;
    r12 = r12 * r31;
    r12 = fmaf(r86, r12, r53 * r101);
    r79 = r79 + r12;
    r101 = r7 * r8;
    r101 = r101 * r23;
    r101 = fmaf(r79, r101, r6 * r79);
    r79 = r31 * r101;
    r79 = fmaf(r52, r79, r1 * r81);
    r81 = r0 * r8;
    r81 = r81 * r42;
    r79 = fmaf(r53, r81, r79);
    r6 = r0 * r44;
    r79 = fmaf(r77, r6, r79);
    r79 = fmaf(r44, r52, r79);
    r79 = fmaf(r44, r10, r79);
    r79 = fmaf(r47, r92, r79);
    r79 = fmaf(r47, r88, r79);
    r79 = fmaf(r47, r80, r79);
    r6 = r4 * r79;
    r81 = r42 * r55;
    r81 = r81 * r76;
    r80 = r47 * r55;
    r80 = r80 * r55;
    r80 = r80 * r57;
    r80 = fmaf(r86, r80, r37 * r81);
    r80 = r80 + r12;
    r80 = fmaf(r42, r52, r0 * r80);
    r12 = r55 * r101;
    r80 = fmaf(r52, r12, r80);
    r81 = r1 * r47;
    r80 = fmaf(r90, r81, r80);
    r90 = r24 * r47;
    r90 = r90 * r55;
    r80 = fmaf(r37, r90, r80);
    r57 = r1 * r8;
    r57 = r57 * r42;
    r80 = fmaf(r53, r57, r80);
    r88 = r24 * r47;
    r88 = r88 * r55;
    r88 = r88 * r54;
    r80 = fmaf(r37, r88, r80);
    r80 = fmaf(r42, r10, r80);
    r80 = fmaf(r44, r91, r80);
    r88 = r5 * r80;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx, r35, r99, r6, r88);
    r88 = r5 * r24;
    r88 = r88 * r3;
    r6 = r4 * r24;
    r6 = r6 * r2;
    r6 = fmaf(r78, r6, r84 * r88);
    r88 = r4 * r24;
    r88 = r88 * r2;
    r99 = r5 * r24;
    r99 = r99 * r3;
    r99 = fmaf(r67, r99, r59 * r88);
    r88 = r5 * r24;
    r88 = r88 * r3;
    r35 = r4 * r24;
    r35 = r35 * r2;
    r35 = fmaf(r85, r35, r13 * r88);
    r88 = r4 * r24;
    r88 = r88 * r2;
    r91 = r5 * r24;
    r91 = r91 * r3;
    r91 = fmaf(r70, r91, r28 * r88);
    WriteSum4<float, float>((float *)inout_shared, r6, r99, r35, r91);
  };
  FlushSumShared<4, float>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r91 = r4 * r24;
    r91 = r91 * r2;
    r35 = r5 * r24;
    r35 = r35 * r3;
    r35 = fmaf(r83, r35, r29 * r91);
    r91 = r4 * r24;
    r91 = r91 * r2;
    r99 = r5 * r24;
    r99 = r99 * r3;
    r99 = fmaf(r80, r99, r79 * r91);
    WriteSum2<float, float>((float *)inout_shared, r35, r99);
  };
  FlushSumShared<2, float>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r99 = r5 * r5;
    r35 = r84 * r99;
    r91 = r4 * r4;
    r6 = r78 * r91;
    r78 = fmaf(r78, r6, r84 * r35);
    r84 = r59 * r59;
    r88 = r67 * r67;
    r88 = fmaf(r99, r88, r91 * r84);
    r84 = r13 * r13;
    r57 = r85 * r85;
    r57 = fmaf(r91, r57, r99 * r84);
    r84 = r70 * r70;
    r90 = r28 * r28;
    r90 = fmaf(r91, r90, r99 * r84);
    WriteSum4<float, float>((float *)inout_shared, r78, r88, r57, r90);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r90 = r83 * r83;
    r57 = r29 * r29;
    r57 = fmaf(r91, r57, r99 * r90);
    r90 = r79 * r79;
    r88 = r80 * r80;
    r88 = fmaf(r99, r88, r91 * r90);
    WriteSum2<float, float>((float *)inout_shared, r57, r88);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r88 = fmaf(r59, r6, r67 * r35);
    r57 = fmaf(r13, r35, r85 * r6);
    r90 = fmaf(r28, r6, r70 * r35);
    r78 = fmaf(r29, r6, r83 * r35);
    WriteSum4<float, float>((float *)inout_shared, r88, r57, r90, r78);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fmaf(r80, r35, r79 * r6);
    r6 = r59 * r85;
    r78 = r67 * r13;
    r78 = fmaf(r99, r78, r91 * r6);
    r6 = r67 * r70;
    r90 = r59 * r28;
    r90 = fmaf(r91, r90, r99 * r6);
    r6 = r59 * r29;
    r57 = r67 * r83;
    r57 = fmaf(r99, r57, r91 * r6);
    WriteSum4<float, float>((float *)inout_shared, r35, r78, r90, r57);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = r67 * r80;
    r90 = r59 * r79;
    r90 = fmaf(r91, r90, r99 * r57);
    r57 = r85 * r28;
    r78 = r13 * r70;
    r78 = fmaf(r99, r78, r91 * r57);
    r57 = r13 * r83;
    r35 = r85 * r29;
    r35 = fmaf(r91, r35, r99 * r57);
    r57 = r85 * r79;
    r6 = r13 * r80;
    r6 = fmaf(r99, r6, r91 * r57);
    WriteSum4<float, float>((float *)inout_shared, r90, r78, r35, r6);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r28 * r29;
    r35 = r70 * r83;
    r35 = fmaf(r99, r35, r91 * r6);
    r6 = r28 * r79;
    r78 = r70 * r80;
    r78 = fmaf(r99, r78, r91 * r6);
    r6 = r29 * r79;
    r90 = r83 * r80;
    r90 = fmaf(r99, r90, r91 * r6);
    WriteSum3<float, float>((float *)inout_shared, r35, r78, r90);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r90 = r4 * r31;
    r90 = r90 * r23;
    r90 = r90 * r52;
    r78 = r5 * r55;
    r78 = r78 * r23;
    r78 = r78 * r52;
    WriteIdx4<1024, float, float, float4>(
        out_calib_jac, 0 * out_calib_jac_num_alloc, global_thread_idx, r41, r40,
        r90, r78);
    r35 = r5 * r56;
    r6 = r4 * r31;
    r6 = r6 * r52;
    r6 = r6 * r39;
    r57 = r5 * r55;
    r57 = r57 * r52;
    r57 = r57 * r39;
    r88 = r4 * r53;
    r88 = r88 * r33;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          4 * out_calib_jac_num_alloc,
                                          global_thread_idx, r6, r57, r88, r35);
    r84 = r4 * r36;
    r10 = r5 * r53;
    r10 = r10 * r33;
    WriteIdx2<1024, float, float, float2>(out_calib_jac,
                                          8 * out_calib_jac_num_alloc,
                                          global_thread_idx, r84, r10);
    r81 = r24 * r41;
    r81 = r81 * r2;
    r12 = r24 * r40;
    r12 = r12 * r3;
    r54 = r5 * r24;
    r54 = r54 * r55;
    r54 = r54 * r23;
    r54 = r54 * r3;
    r92 = r4 * r24;
    r92 = r92 * r31;
    r92 = r92 * r23;
    r92 = r92 * r2;
    r92 = fmaf(r52, r92, r52 * r54);
    r54 = r5 * r24;
    r54 = r54 * r55;
    r54 = r54 * r3;
    r54 = r54 * r52;
    r77 = r4 * r24;
    r77 = r77 * r31;
    r77 = r77 * r2;
    r77 = r77 * r52;
    r77 = fmaf(r39, r77, r39 * r54);
    WriteSum4<float, float>((float *)inout_shared, r81, r12, r92, r77);
  };
  FlushSumShared<4, float>(out_calib_njtr, 0 * out_calib_njtr_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = r24 * r2;
    r92 = r24 * r3;
    r12 = r5 * r24;
    r12 = r12 * r56;
    r81 = r4 * r25;
    r81 = r81 * r55;
    r81 = r81 * r2;
    r81 = fmaf(r53, r81, r3 * r12);
    r12 = r4 * r24;
    r12 = r12 * r36;
    r54 = r5 * r25;
    r54 = r54 * r55;
    r54 = r54 * r3;
    r54 = fmaf(r53, r54, r2 * r12);
    WriteSum4<float, float>((float *)inout_shared, r81, r54, r77, r92);
  };
  FlushSumShared<4, float>(out_calib_njtr, 4 * out_calib_njtr_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r92 = r41 * r41;
    r77 = r40 * r40;
    r54 = r55 * r55;
    r54 = r54 * r37;
    r54 = r54 * r99;
    r81 = r31 * r53;
    r81 = r81 * r91;
    r81 = fmaf(r39, r81, r39 * r54);
    r54 = r39 * r39;
    r12 = r55 * r55;
    r12 = r12 * r37;
    r12 = r12 * r99;
    r37 = r31 * r53;
    r37 = r37 * r91;
    r54 = fmaf(r54, r37, r54 * r12);
    WriteSum4<float, float>((float *)inout_shared, r92, r77, r81, r54);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = r56 * r56;
    r81 = r31 * r31;
    r77 = 4.00000000000000000e+00;
    r50 = r30 * r50;
    r50 = 1.0 / r50;
    r81 = r81 * r55;
    r81 = r81 * r55;
    r81 = r81 * r77;
    r81 = r81 * r50;
    r54 = fmaf(r91, r81, r99 * r54);
    r50 = r36 * r36;
    r81 = fmaf(r99, r81, r91 * r50);
    WriteSum4<float, float>((float *)inout_shared, r54, r81, r26, r26);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           4 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = 0.00000000000000000e+00;
    r81 = r4 * r31;
    r81 = r81 * r23;
    r81 = r81 * r41;
    r81 = r81 * r52;
    r54 = r31 * r46;
    r50 = r52 * r39;
    r54 = r54 * r50;
    r77 = r53 * r33;
    r46 = r46 * r77;
    WriteSum4<float, float>((float *)inout_shared, r26, r81, r54, r46);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r4 * r36;
    r46 = r46 * r41;
    r54 = r5 * r55;
    r54 = r54 * r23;
    r54 = r54 * r40;
    r54 = r54 * r52;
    WriteSum4<float, float>((float *)inout_shared, r46, r41, r26, r54);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = r5 * r56;
    r54 = r54 * r40;
    r41 = r55 * r58;
    r41 = r41 * r50;
    r77 = r58 * r77;
    WriteSum4<float, float>((float *)inout_shared, r41, r54, r77, r26);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           8 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = r23 * r39;
    r37 = fmaf(r77, r37, r77 * r12);
    r77 = r31 * r31;
    r77 = r77 * r23;
    r77 = r77 * r86;
    r77 = r77 * r91;
    r12 = r55 * r23;
    r12 = r12 * r56;
    r12 = r12 * r52;
    r12 = fmaf(r99, r12, r33 * r77);
    r77 = r31 * r55;
    r77 = r77 * r23;
    r77 = r77 * r86;
    r77 = r77 * r99;
    r54 = r31 * r36;
    r54 = r54 * r23;
    r54 = r54 * r52;
    r54 = fmaf(r91, r54, r33 * r77);
    WriteSum4<float, float>((float *)inout_shared, r40, r37, r12, r54);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           12 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = r31 * r31;
    r54 = r54 * r86;
    r54 = r54 * r91;
    r54 = r54 * r33;
    r12 = r55 * r56;
    r12 = r12 * r52;
    r12 = r12 * r99;
    r12 = fmaf(r39, r12, r39 * r54);
    r54 = r31 * r55;
    r54 = r54 * r86;
    r54 = r54 * r99;
    r54 = r54 * r33;
    r86 = r31 * r36;
    r86 = r86 * r52;
    r86 = r86 * r91;
    r86 = fmaf(r39, r86, r39 * r54);
    WriteSum4<float, float>((float *)inout_shared, r90, r78, r12, r86);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           16 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r86 = r56 * r53;
    r86 = r86 * r99;
    r99 = r36 * r53;
    r99 = r99 * r91;
    r99 = fmaf(r33, r99, r33 * r86);
    WriteSum4<float, float>((float *)inout_shared, r6, r57, r99, r88);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           20 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float *)inout_shared, r35, r84, r10, r26);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           24 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void OpencvFixedPointResJacFirst(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    float *pixel, unsigned int pixel_num_alloc, float *point,
    unsigned int point_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *const out_rTr, float *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, float *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, float *out_calib_jac,
    unsigned int out_calib_jac_num_alloc, float *const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc, float *const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    float *const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvFixedPointResJacFirstKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, sensor_from_rig,
      sensor_from_rig_num_alloc, calib, calib_num_alloc, calib_indices, pixel,
      pixel_num_alloc, point, point_num_alloc, out_res, out_res_num_alloc,
      out_rTr, out_pose_jac, out_pose_jac_num_alloc, out_pose_njtr,
      out_pose_njtr_num_alloc, out_pose_precond_diag,
      out_pose_precond_diag_num_alloc, out_pose_precond_tril,
      out_pose_precond_tril_num_alloc, out_calib_jac, out_calib_jac_num_alloc,
      out_calib_njtr, out_calib_njtr_num_alloc, out_calib_precond_diag,
      out_calib_precond_diag_num_alloc, out_calib_precond_tril,
      out_calib_precond_tril_num_alloc, problem_size);
}

} // namespace caspar