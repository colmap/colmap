#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_opencv_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) OpencvResJacFirstKernel(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    float *point, unsigned int point_num_alloc, SharedIndex *point_indices,
    float *pixel, unsigned int pixel_num_alloc, float *out_res,
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
    unsigned int out_calib_precond_tril_num_alloc, float *out_point_jac,
    unsigned int out_point_jac_num_alloc, float *const out_point_njtr,
    unsigned int out_point_njtr_num_alloc, float *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {
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
      r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102, r103, r104,
      r105, r106, r107, r108, r109, r110;
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
  };
  LoadShared<3, float, float>(point, 0 * point_num_alloc, point_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       point_indices_loc[threadIdx.x].target, r11, r12, r13);
  };
  __syncthreads();
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
    r8 = fmaf(r11, r31, r8);
    r32 = 2.00000000000000000e+00;
    r33 = fmaf(r17, r18, r14 * r21);
    r34 = r15 * r20;
    r33 = fmaf(r24, r34, r33);
    r33 = fmaf(r16, r19, r33);
    r34 = r32 * r33;
    r34 = r34 * r27;
    r35 = fmaf(r15, r19, r14 * r18);
    r35 = fmaf(r16, r20, r35);
    r35 = fmaf(r24, r35, r17 * r21);
    r36 = r25 * r35;
    r37 = fmaf(r22, r36, r34);
    r38 = r32 * r22;
    r38 = r38 * r33;
    r39 = r32 * r27;
    r39 = fmaf(r35, r39, r38);
  };
  LoadShared<3, float, float>(pose, 4 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r40, r41, r42);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r43 = r18 * r20;
    r43 = r43 * r32;
    r44 = r19 * r21;
    r45 = fmaf(r32, r44, r43);
    r46 = r20 * r21;
    r47 = r18 * r19;
    r47 = r47 * r32;
    r46 = fmaf(r25, r46, r47);
    r48 = r19 * r19;
    r48 = r48 * r25;
    r49 = r26 + r48;
    r50 = r20 * r20;
    r50 = r50 * r25;
    r49 = r49 + r50;
    r8 = fmaf(r12, r37, r8);
    r8 = fmaf(r13, r39, r8);
    r8 = fmaf(r42, r45, r8);
    r8 = fmaf(r41, r46, r8);
    r8 = fmaf(r40, r49, r8);
    r51 = 3.00000000000000000e+00;
    r52 = r8 * r51;
    r53 = 9.99999999999999955e-07;
    r38 = fmaf(r27, r36, r38);
    r10 = fmaf(r11, r38, r10);
    r44 = fmaf(r25, r44, r43);
    r48 = r26 + r48;
    r43 = r18 * r18;
    r43 = r43 * r25;
    r48 = r48 + r43;
    r54 = r19 * r20;
    r54 = r54 * r32;
    r55 = r18 * r21;
    r55 = fmaf(r32, r55, r54);
    r56 = r32 * r22;
    r56 = r56 * r27;
    r57 = r32 * r33;
    r57 = fmaf(r35, r57, r56);
    r58 = r33 * r33;
    r58 = r58 * r25;
    r30 = r58 + r30;
    r10 = fmaf(r40, r44, r10);
    r10 = fmaf(r42, r48, r10);
    r10 = fmaf(r41, r55, r10);
    r10 = fmaf(r12, r57, r10);
    r10 = fmaf(r13, r30, r10);
    r59 = copysign(1.0, r10);
    r59 = fmaf(r53, r59, r10);
    r53 = r59 * r59;
    r10 = 1.0 / r53;
    r60 = r8 * r10;
    r61 = r32 * r22;
    r61 = fmaf(r35, r61, r34);
    r9 = fmaf(r11, r61, r9);
    r34 = r20 * r21;
    r34 = fmaf(r32, r34, r47);
    r50 = r26 + r50;
    r50 = r50 + r43;
    r43 = r18 * r21;
    r43 = fmaf(r25, r43, r54);
    r56 = fmaf(r33, r36, r56);
    r23 = r26 + r23;
    r23 = r23 + r58;
    r9 = fmaf(r40, r34, r9);
    r9 = fmaf(r41, r50, r9);
    r9 = fmaf(r42, r43, r9);
    r9 = fmaf(r13, r56, r9);
    r9 = fmaf(r12, r23, r9);
    r42 = r9 * r9;
    r42 = r42 * r10;
    r52 = fmaf(r60, r52, r42);
    r41 = 1.0 / r59;
    r40 = fmaf(r8, r41, r1 * r52);
    r58 = r8 * r60;
    r42 = r42 + r58;
    r54 = r42 * r42;
    r47 = fmaf(r7, r54, r6 * r42);
    r62 = r47 * r41;
    r63 = r32 * r60;
    r64 = r0 * r63;
    r40 = fmaf(r8, r62, r40);
    r40 = fmaf(r9, r64, r40);
    r2 = fmaf(r4, r40, r2);
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r65, r66);
    r2 = fmaf(r65, r24, r2);
    r65 = r51 * r9;
    r65 = r65 * r9;
    r65 = fmaf(r10, r65, r58);
    r58 = fmaf(r9, r41, r0 * r65);
    r67 = r1 * r9;
    r58 = fmaf(r63, r67, r58);
    r58 = fmaf(r9, r62, r58);
    r3 = fmaf(r5, r58, r3);
    r3 = fmaf(r66, r24, r3);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r2, r3);
    r66 = fmaf(r3, r3, r2 * r2);
  };
  SumStore<float>(out_rTr_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r66);
  if (global_thread_idx < problem_size) {
    r66 = r32 * r35;
    r67 = r16 * r21;
    r68 = 5.00000000000000000e-01;
    r69 = r15 * r18;
    r69 = fmaf(r68, r69, r68 * r67);
    r67 = r14 * r19;
    r70 = -5.00000000000000000e-01;
    r69 = fmaf(r70, r67, r69);
    r71 = r17 * r68;
    r69 = fmaf(r20, r71, r69);
    r67 = r14 * r21;
    r72 = r17 * r18;
    r72 = fmaf(r70, r72, r70 * r67);
    r67 = r16 * r19;
    r72 = fmaf(r70, r67, r72);
    r73 = r15 * r20;
    r72 = fmaf(r68, r73, r72);
    r73 = r27 * r72;
    r66 = fmaf(r32, r73, r69 * r66);
    r67 = r32 * r33;
    r74 = r17 * r19;
    r75 = r15 * r70;
    r74 = fmaf(r21, r75, r70 * r74);
    r74 = fmaf(r68, r29, r74);
    r74 = fmaf(r70, r28, r74);
    r76 = r32 * r22;
    r77 = r14 * r18;
    r78 = r16 * r20;
    r78 = fmaf(r70, r78, r70 * r77);
    r78 = fmaf(r21, r71, r78);
    r78 = fmaf(r19, r75, r78);
    r76 = r76 * r78;
    r67 = fmaf(r74, r67, r76);
    r66 = r66 + r67;
    r77 = r32 * r27;
    r77 = r77 * r78;
    r79 = r32 * r33;
    r79 = r79 * r69;
    r80 = r77 + r79;
    r81 = r22 * r25;
    r80 = fmaf(r72, r81, r80);
    r80 = fmaf(r74, r36, r80);
    r80 = fmaf(r12, r80, r13 * r66);
    r66 = r27 * r69;
    r81 = -4.00000000000000000e+00;
    r66 = r66 * r81;
    r82 = r22 * r74;
    r83 = r81 * r82;
    r84 = r66 + r83;
    r80 = fmaf(r11, r84, r80);
    r84 = 6.00000000000000000e+00;
    r85 = r80 * r84;
    r86 = r8 * r8;
    r87 = r32 * r27;
    r87 = r87 * r74;
    r88 = r32 * r22;
    r88 = fmaf(r69, r88, r87);
    r89 = r32 * r33;
    r89 = r89 * r72;
    r90 = r32 * r35;
    r90 = r90 * r78;
    r91 = r89 + r90;
    r92 = r88 + r91;
    r69 = fmaf(r69, r36, r25 * r73);
    r69 = r69 + r67;
    r69 = fmaf(r11, r69, r12 * r92);
    r92 = r33 * r78;
    r92 = r92 * r81;
    r66 = r66 + r92;
    r69 = fmaf(r13, r66, r69);
    r66 = -6.00000000000000000e+00;
    r53 = r59 * r53;
    r93 = 1.0 / r53;
    r86 = r86 * r69;
    r86 = r86 * r66;
    r86 = fmaf(r93, r86, r60 * r85);
    r85 = r32 * r9;
    r94 = r33 * r25;
    r95 = r78 * r36;
    r94 = fmaf(r72, r94, r95);
    r94 = r94 + r88;
    r83 = r92 + r83;
    r83 = fmaf(r12, r83, r13 * r94);
    r94 = r32 * r35;
    r94 = fmaf(r74, r94, r79);
    r79 = r32 * r22;
    r79 = fmaf(r72, r79, r77);
    r94 = r94 + r79;
    r83 = fmaf(r11, r94, r83);
    r85 = r85 * r83;
    r94 = r25 * r9;
    r77 = r9 * r93;
    r94 = r94 * r69;
    r94 = fmaf(r77, r94, r10 * r85);
    r86 = r86 + r94;
    r86 = fmaf(r80, r62, r1 * r86);
    r85 = r0 * r32;
    r85 = r85 * r9;
    r85 = r85 * r80;
    r86 = fmaf(r10, r85, r86);
    r92 = r24 * r47;
    r92 = r92 * r69;
    r86 = fmaf(r60, r92, r86);
    r88 = r7 * r32;
    r96 = r25 * r8;
    r96 = r96 * r8;
    r96 = r96 * r69;
    r96 = fmaf(r93, r96, r80 * r63);
    r94 = r94 + r96;
    r88 = r88 * r42;
    r94 = fmaf(r6, r94, r94 * r88);
    r88 = r8 * r94;
    r86 = fmaf(r41, r88, r86);
    r97 = r0 * r69;
    r98 = r8 * r77;
    r99 = r81 * r98;
    r86 = fmaf(r99, r97, r86);
    r100 = r24 * r69;
    r86 = fmaf(r60, r100, r86);
    r86 = fmaf(r83, r64, r86);
    r86 = fmaf(r80, r41, r86);
    r100 = r4 * r86;
    r97 = r9 * r83;
    r97 = r97 * r84;
    r88 = r9 * r66;
    r88 = r88 * r77;
    r97 = fmaf(r69, r88, r10 * r97);
    r97 = r97 + r96;
    r96 = r1 * r32;
    r96 = r96 * r9;
    r96 = r96 * r80;
    r96 = fmaf(r10, r96, r0 * r97);
    r97 = r24 * r9;
    r97 = r97 * r69;
    r96 = fmaf(r10, r97, r96);
    r92 = r24 * r9;
    r92 = r92 * r47;
    r92 = r92 * r69;
    r96 = fmaf(r10, r92, r96);
    r85 = r9 * r94;
    r96 = fmaf(r41, r85, r96);
    r101 = r1 * r99;
    r102 = r1 * r83;
    r96 = fmaf(r63, r102, r96);
    r96 = fmaf(r83, r62, r96);
    r96 = fmaf(r83, r41, r96);
    r96 = fmaf(r69, r101, r96);
    r102 = r5 * r96;
    r90 = r87 + r90;
    r87 = r32 * r22;
    r85 = r16 * r21;
    r92 = r14 * r19;
    r92 = fmaf(r68, r92, r70 * r85);
    r85 = r17 * r20;
    r92 = fmaf(r70, r85, r92);
    r92 = fmaf(r18, r75, r92);
    r87 = r87 * r92;
    r85 = r32 * r33;
    r97 = r14 * r21;
    r103 = r16 * r19;
    r103 = fmaf(r68, r103, r68 * r97);
    r103 = fmaf(r18, r71, r103);
    r103 = fmaf(r20, r75, r103);
    r85 = fmaf(r103, r85, r87);
    r90 = r90 + r85;
    r75 = r22 * r81;
    r75 = r75 * r103;
    r97 = r27 * r78;
    r97 = r97 * r81;
    r104 = r75 + r97;
    r104 = fmaf(r11, r104, r13 * r90);
    r90 = fmaf(r103, r36, r25 * r82);
    r105 = r32 * r33;
    r105 = r105 * r78;
    r106 = r32 * r27;
    r106 = fmaf(r92, r106, r105);
    r90 = r90 + r106;
    r104 = fmaf(r12, r90, r104);
    r90 = r84 * r104;
    r107 = r8 * r8;
    r108 = r25 * r27;
    r108 = fmaf(r74, r108, r95);
    r108 = r108 + r85;
    r85 = r32 * r27;
    r85 = r85 * r103;
    r109 = r32 * r35;
    r109 = fmaf(r92, r109, r85);
    r109 = r109 + r67;
    r109 = fmaf(r12, r109, r11 * r108);
    r108 = r33 * r92;
    r67 = r81 * r108;
    r97 = r97 + r67;
    r109 = fmaf(r13, r97, r109);
    r107 = r107 * r66;
    r107 = r107 * r109;
    r107 = fmaf(r93, r107, r60 * r90);
    r90 = r25 * r9;
    r90 = r90 * r109;
    r97 = r32 * r9;
    r110 = r33 * r25;
    r110 = fmaf(r74, r110, r76);
    r110 = r110 + r85;
    r110 = fmaf(r92, r36, r110);
    r85 = r32 * r35;
    r82 = fmaf(r32, r82, r103 * r85);
    r82 = r82 + r106;
    r82 = fmaf(r11, r82, r13 * r110);
    r67 = r75 + r67;
    r82 = fmaf(r12, r67, r82);
    r97 = r97 * r82;
    r97 = fmaf(r10, r97, r77 * r90);
    r107 = r107 + r97;
    r107 = fmaf(r82, r64, r1 * r107);
    r90 = r24 * r109;
    r107 = fmaf(r60, r90, r107);
    r67 = r25 * r8;
    r67 = r67 * r8;
    r67 = r67 * r109;
    r67 = fmaf(r93, r67, r104 * r63);
    r97 = r97 + r67;
    r75 = r7 * r32;
    r75 = r75 * r42;
    r75 = fmaf(r97, r75, r6 * r97);
    r97 = r8 * r75;
    r107 = fmaf(r41, r97, r107);
    r110 = r0 * r32;
    r110 = r110 * r9;
    r110 = r110 * r104;
    r107 = fmaf(r10, r110, r107);
    r85 = r0 * r109;
    r107 = fmaf(r99, r85, r107);
    r103 = r24 * r47;
    r103 = r103 * r109;
    r107 = fmaf(r60, r103, r107);
    r107 = fmaf(r104, r41, r107);
    r107 = fmaf(r104, r62, r107);
    r103 = r4 * r107;
    r85 = r9 * r84;
    r85 = r85 * r82;
    r85 = fmaf(r10, r85, r109 * r88);
    r85 = r85 + r67;
    r67 = r1 * r82;
    r67 = fmaf(r63, r67, r0 * r85);
    r85 = r24 * r9;
    r85 = r85 * r109;
    r67 = fmaf(r10, r85, r67);
    r110 = r1 * r32;
    r110 = r110 * r9;
    r110 = r110 * r104;
    r67 = fmaf(r10, r110, r67);
    r97 = r24 * r9;
    r97 = r97 * r47;
    r97 = r97 * r109;
    r67 = fmaf(r10, r97, r67);
    r90 = r9 * r75;
    r67 = fmaf(r41, r90, r67);
    r67 = fmaf(r82, r62, r67);
    r67 = fmaf(r82, r41, r67);
    r67 = fmaf(r109, r101, r67);
    r90 = r5 * r67;
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r100, r102,
        r103, r90);
    r90 = r8 * r8;
    r103 = r33 * r81;
    r102 = r15 * r21;
    r29 = fmaf(r70, r29, r68 * r102);
    r29 = fmaf(r19, r71, r29);
    r29 = fmaf(r68, r28, r29);
    r103 = r103 * r29;
    r73 = r81 * r73;
    r28 = r103 + r73;
    r68 = r32 * r22;
    r68 = r68 * r29;
    r105 = r105 + r68;
    r71 = r25 * r27;
    r105 = fmaf(r92, r71, r105);
    r105 = fmaf(r72, r36, r105);
    r105 = fmaf(r11, r105, r13 * r28);
    r28 = r32 * r35;
    r28 = fmaf(r32, r108, r29 * r28);
    r28 = r28 + r79;
    r105 = fmaf(r12, r28, r105);
    r90 = r90 * r66;
    r90 = r90 * r105;
    r28 = r32 * r27;
    r28 = r28 * r29;
    r89 = r89 + r28;
    r71 = r22 * r25;
    r89 = fmaf(r92, r71, r89);
    r89 = r89 + r95;
    r78 = r22 * r78;
    r78 = r78 * r81;
    r73 = r78 + r73;
    r73 = fmaf(r11, r73, r12 * r89);
    r89 = r32 * r35;
    r89 = fmaf(r72, r89, r68);
    r89 = r89 + r106;
    r73 = fmaf(r13, r89, r73);
    r89 = r84 * r73;
    r89 = fmaf(r60, r89, r93 * r90);
    r90 = r25 * r9;
    r90 = r90 * r105;
    r106 = r32 * r9;
    r28 = r87 + r28;
    r28 = r28 + r91;
    r36 = fmaf(r29, r36, r25 * r108);
    r36 = r36 + r79;
    r36 = fmaf(r13, r36, r11 * r28);
    r103 = r78 + r103;
    r36 = fmaf(r12, r103, r36);
    r106 = r106 * r36;
    r106 = fmaf(r10, r106, r77 * r90);
    r89 = r89 + r106;
    r89 = fmaf(r36, r64, r1 * r89);
    r90 = r7 * r32;
    r103 = r25 * r8;
    r103 = r103 * r8;
    r103 = r103 * r105;
    r103 = fmaf(r73, r63, r93 * r103);
    r106 = r106 + r103;
    r90 = r90 * r42;
    r106 = fmaf(r6, r106, r106 * r90);
    r90 = r8 * r106;
    r89 = fmaf(r41, r90, r89);
    r12 = r24 * r105;
    r89 = fmaf(r60, r12, r89);
    r78 = r0 * r105;
    r89 = fmaf(r99, r78, r89);
    r13 = r0 * r32;
    r13 = r13 * r9;
    r13 = r13 * r73;
    r89 = fmaf(r10, r13, r89);
    r28 = r24 * r47;
    r28 = r28 * r105;
    r89 = fmaf(r60, r28, r89);
    r89 = fmaf(r73, r41, r89);
    r89 = fmaf(r73, r62, r89);
    r28 = r4 * r89;
    r13 = r9 * r84;
    r13 = r13 * r36;
    r13 = fmaf(r10, r13, r105 * r88);
    r13 = r13 + r103;
    r13 = fmaf(r36, r62, r0 * r13);
    r103 = r1 * r36;
    r13 = fmaf(r63, r103, r13);
    r78 = r9 * r106;
    r13 = fmaf(r41, r78, r13);
    r12 = r24 * r9;
    r12 = r12 * r47;
    r12 = r12 * r105;
    r13 = fmaf(r10, r12, r13);
    r90 = r24 * r9;
    r90 = r90 * r105;
    r13 = fmaf(r10, r90, r13);
    r11 = r1 * r32;
    r11 = r11 * r9;
    r11 = r11 * r73;
    r13 = fmaf(r10, r11, r13);
    r13 = fmaf(r36, r41, r13);
    r13 = fmaf(r105, r101, r13);
    r11 = r5 * r13;
    r90 = r44 * r8;
    r90 = r90 * r8;
    r90 = r90 * r66;
    r12 = r49 * r84;
    r12 = fmaf(r60, r12, r93 * r90);
    r90 = r25 * r44;
    r90 = r90 * r9;
    r78 = r32 * r34;
    r78 = r78 * r9;
    r78 = fmaf(r10, r78, r77 * r90);
    r12 = r12 + r78;
    r12 = fmaf(r49, r41, r1 * r12);
    r90 = r24 * r44;
    r12 = fmaf(r60, r90, r12);
    r103 = r7 * r32;
    r79 = r25 * r44;
    r79 = r79 * r8;
    r79 = r79 * r8;
    r79 = fmaf(r49, r63, r93 * r79);
    r78 = r78 + r79;
    r103 = r103 * r42;
    r78 = fmaf(r6, r78, r78 * r103);
    r103 = r8 * r78;
    r12 = fmaf(r41, r103, r12);
    r29 = r24 * r44;
    r29 = r29 * r47;
    r12 = fmaf(r60, r29, r12);
    r108 = r0 * r44;
    r12 = fmaf(r99, r108, r12);
    r91 = r0 * r32;
    r91 = r91 * r49;
    r91 = r91 * r9;
    r12 = fmaf(r10, r91, r12);
    r12 = fmaf(r49, r62, r12);
    r12 = fmaf(r34, r64, r12);
    r91 = r4 * r12;
    r108 = r34 * r9;
    r108 = r108 * r84;
    r108 = fmaf(r10, r108, r44 * r88);
    r108 = r108 + r79;
    r79 = r9 * r78;
    r79 = fmaf(r41, r79, r0 * r108);
    r108 = r24 * r44;
    r108 = r108 * r9;
    r108 = r108 * r47;
    r79 = fmaf(r10, r108, r79);
    r29 = r24 * r44;
    r29 = r29 * r9;
    r79 = fmaf(r10, r29, r79);
    r103 = r1 * r34;
    r79 = fmaf(r63, r103, r79);
    r90 = r1 * r32;
    r90 = r90 * r49;
    r90 = r90 * r9;
    r79 = fmaf(r10, r90, r79);
    r79 = fmaf(r34, r41, r79);
    r79 = fmaf(r34, r62, r79);
    r79 = fmaf(r44, r101, r79);
    r90 = r5 * r79;
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r28, r11,
        r91, r90);
    r90 = r55 * r8;
    r90 = r90 * r8;
    r90 = r90 * r66;
    r91 = r46 * r84;
    r91 = fmaf(r60, r91, r93 * r90);
    r90 = r25 * r55;
    r90 = r90 * r9;
    r11 = r32 * r50;
    r11 = r11 * r9;
    r11 = fmaf(r10, r11, r77 * r90);
    r91 = r91 + r11;
    r91 = fmaf(r46, r41, r1 * r91);
    r90 = r0 * r55;
    r91 = fmaf(r99, r90, r91);
    r28 = r25 * r55;
    r28 = r28 * r8;
    r28 = r28 * r8;
    r28 = fmaf(r46, r63, r93 * r28);
    r11 = r11 + r28;
    r103 = r7 * r32;
    r103 = r103 * r42;
    r103 = fmaf(r11, r103, r6 * r11);
    r11 = r8 * r103;
    r91 = fmaf(r41, r11, r91);
    r29 = r0 * r32;
    r29 = r29 * r46;
    r29 = r29 * r9;
    r91 = fmaf(r10, r29, r91);
    r108 = r24 * r55;
    r91 = fmaf(r60, r108, r91);
    r87 = r24 * r55;
    r87 = r87 * r47;
    r91 = fmaf(r60, r87, r91);
    r91 = fmaf(r46, r62, r91);
    r91 = fmaf(r50, r64, r91);
    r87 = r4 * r91;
    r108 = r50 * r9;
    r108 = r108 * r84;
    r108 = fmaf(r10, r108, r55 * r88);
    r108 = r108 + r28;
    r108 = fmaf(r50, r41, r0 * r108);
    r28 = r9 * r103;
    r108 = fmaf(r41, r28, r108);
    r29 = r24 * r55;
    r29 = r29 * r9;
    r29 = r29 * r47;
    r108 = fmaf(r10, r29, r108);
    r11 = r1 * r32;
    r11 = r11 * r46;
    r11 = r11 * r9;
    r108 = fmaf(r10, r11, r108);
    r90 = r1 * r50;
    r108 = fmaf(r63, r90, r108);
    r68 = r24 * r55;
    r68 = r68 * r9;
    r108 = fmaf(r10, r68, r108);
    r108 = fmaf(r55, r101, r108);
    r108 = fmaf(r50, r62, r108);
    r68 = r5 * r108;
    r90 = r45 * r84;
    r11 = r48 * r8;
    r11 = r11 * r8;
    r11 = r11 * r66;
    r11 = fmaf(r93, r11, r60 * r90);
    r90 = r32 * r43;
    r90 = r90 * r9;
    r29 = r25 * r48;
    r29 = r29 * r9;
    r29 = fmaf(r77, r29, r10 * r90);
    r11 = r11 + r29;
    r90 = r25 * r48;
    r90 = r90 * r8;
    r90 = r90 * r8;
    r90 = fmaf(r93, r90, r45 * r63);
    r29 = r29 + r90;
    r28 = r7 * r32;
    r28 = r28 * r42;
    r28 = fmaf(r29, r28, r6 * r29);
    r29 = r8 * r28;
    r29 = fmaf(r41, r29, r1 * r11);
    r11 = r0 * r48;
    r29 = fmaf(r99, r11, r29);
    r72 = r24 * r48;
    r72 = r72 * r47;
    r29 = fmaf(r60, r72, r29);
    r81 = r24 * r48;
    r29 = fmaf(r60, r81, r29);
    r95 = r0 * r32;
    r95 = r95 * r45;
    r95 = r95 * r9;
    r29 = fmaf(r10, r95, r29);
    r29 = fmaf(r45, r41, r29);
    r29 = fmaf(r45, r62, r29);
    r29 = fmaf(r43, r64, r29);
    r95 = r4 * r29;
    r81 = r43 * r9;
    r81 = r81 * r84;
    r81 = fmaf(r48, r88, r10 * r81);
    r81 = r81 + r90;
    r81 = fmaf(r43, r41, r0 * r81);
    r90 = r9 * r28;
    r81 = fmaf(r41, r90, r81);
    r72 = r24 * r48;
    r72 = r72 * r9;
    r81 = fmaf(r10, r72, r81);
    r11 = r1 * r43;
    r81 = fmaf(r63, r11, r81);
    r71 = r1 * r32;
    r71 = r71 * r45;
    r71 = r71 * r9;
    r81 = fmaf(r10, r71, r81);
    r92 = r24 * r48;
    r92 = r92 * r9;
    r92 = r92 * r47;
    r81 = fmaf(r10, r92, r81);
    r81 = fmaf(r48, r101, r81);
    r81 = fmaf(r43, r62, r81);
    r92 = r5 * r81;
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r87, r68,
        r95, r92);
    r92 = r4 * r24;
    r92 = r92 * r2;
    r95 = r24 * r3;
    r68 = r5 * r95;
    r92 = fmaf(r96, r68, r86 * r92);
    r87 = r4 * r24;
    r87 = r87 * r2;
    r87 = fmaf(r67, r68, r107 * r87);
    r71 = r4 * r24;
    r71 = r71 * r2;
    r71 = fmaf(r13, r68, r89 * r71);
    r11 = r4 * r24;
    r11 = r11 * r2;
    r11 = fmaf(r79, r68, r12 * r11);
    WriteSum4<float, float>((float *)inout_shared, r92, r87, r71, r11);
  };
  FlushSumShared<4, float>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = r4 * r24;
    r11 = r11 * r2;
    r11 = fmaf(r108, r68, r91 * r11);
    r71 = r4 * r24;
    r71 = r71 * r2;
    r71 = fmaf(r81, r68, r29 * r71);
    WriteSum2<float, float>((float *)inout_shared, r11, r71);
  };
  FlushSumShared<2, float>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = r5 * r5;
    r11 = r96 * r71;
    r87 = r4 * r4;
    r92 = r86 * r87;
    r86 = fmaf(r86, r92, r96 * r11);
    r96 = r107 * r107;
    r72 = r67 * r67;
    r72 = fmaf(r71, r72, r87 * r96);
    r96 = r13 * r13;
    r90 = r89 * r89;
    r90 = fmaf(r87, r90, r71 * r96);
    r96 = r79 * r79;
    r70 = r12 * r12;
    r70 = fmaf(r87, r70, r71 * r96);
    WriteSum4<float, float>((float *)inout_shared, r86, r72, r90, r70);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = r108 * r108;
    r90 = r91 * r91;
    r90 = fmaf(r87, r90, r71 * r70);
    r70 = r29 * r29;
    r72 = r81 * r81;
    r72 = fmaf(r71, r72, r87 * r70);
    WriteSum2<float, float>((float *)inout_shared, r90, r72);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fmaf(r107, r92, r67 * r11);
    r90 = fmaf(r13, r11, r89 * r92);
    r70 = fmaf(r12, r92, r79 * r11);
    r86 = fmaf(r91, r92, r108 * r11);
    WriteSum4<float, float>((float *)inout_shared, r72, r90, r70, r86);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r81, r11, r29 * r92);
    r92 = r107 * r89;
    r86 = r67 * r13;
    r86 = fmaf(r71, r86, r87 * r92);
    r92 = r67 * r79;
    r70 = r107 * r12;
    r70 = fmaf(r87, r70, r71 * r92);
    r92 = r107 * r91;
    r90 = r67 * r108;
    r90 = fmaf(r71, r90, r87 * r92);
    WriteSum4<float, float>((float *)inout_shared, r11, r86, r70, r90);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r90 = r67 * r81;
    r70 = r107 * r29;
    r70 = fmaf(r87, r70, r71 * r90);
    r90 = r89 * r12;
    r86 = r13 * r79;
    r86 = fmaf(r71, r86, r87 * r90);
    r90 = r13 * r108;
    r11 = r89 * r91;
    r11 = fmaf(r87, r11, r71 * r90);
    r90 = r89 * r29;
    r92 = r13 * r81;
    r92 = fmaf(r71, r92, r87 * r90);
    WriteSum4<float, float>((float *)inout_shared, r70, r86, r11, r92);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r92 = r12 * r91;
    r11 = r79 * r108;
    r11 = fmaf(r71, r11, r87 * r92);
    r92 = r12 * r29;
    r86 = r79 * r81;
    r86 = fmaf(r71, r86, r87 * r92);
    r92 = r91 * r29;
    r70 = r108 * r81;
    r70 = fmaf(r71, r70, r87 * r92);
    WriteSum3<float, float>((float *)inout_shared, r11, r86, r70);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = r4 * r8;
    r70 = r70 * r42;
    r70 = r70 * r41;
    r86 = r5 * r9;
    r86 = r86 * r42;
    r86 = r86 * r41;
    WriteIdx4<1024, float, float, float4>(
        out_calib_jac, 0 * out_calib_jac_num_alloc, global_thread_idx, r40, r58,
        r70, r86);
    r11 = r4 * r41;
    r11 = r11 * r8;
    r11 = r11 * r54;
    r92 = r5 * r65;
    r90 = r5 * r9;
    r90 = r90 * r41;
    r90 = r90 * r54;
    r72 = r4 * r9;
    r72 = r72 * r63;
    WriteIdx4<1024, float, float, float4>(
        out_calib_jac, 4 * out_calib_jac_num_alloc, global_thread_idx, r11, r90,
        r72, r92);
    r96 = r4 * r52;
    r102 = r5 * r9;
    r102 = r102 * r63;
    WriteIdx2<1024, float, float, float2>(out_calib_jac,
                                          8 * out_calib_jac_num_alloc,
                                          global_thread_idx, r96, r102);
    r100 = r24 * r40;
    r100 = r100 * r2;
    r97 = r58 * r95;
    r110 = r9 * r42;
    r110 = r110 * r41;
    r85 = r4 * r24;
    r85 = r85 * r8;
    r85 = r85 * r42;
    r85 = r85 * r2;
    r85 = fmaf(r41, r85, r68 * r110);
    r110 = r9 * r41;
    r110 = r110 * r54;
    r76 = r24 * r2;
    r76 = fmaf(r11, r76, r68 * r110);
    WriteSum4<float, float>((float *)inout_shared, r100, r97, r85, r76);
  };
  FlushSumShared<4, float>(out_calib_njtr, 0 * out_calib_njtr_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = r24 * r2;
    r85 = r4 * r25;
    r85 = r85 * r9;
    r85 = r85 * r2;
    r85 = fmaf(r60, r85, r65 * r68);
    r97 = r4 * r24;
    r97 = r97 * r52;
    r100 = r5 * r25;
    r100 = r100 * r9;
    r100 = r100 * r3;
    r100 = fmaf(r60, r100, r2 * r97);
    WriteSum4<float, float>((float *)inout_shared, r85, r100, r76, r95);
  };
  FlushSumShared<4, float>(out_calib_njtr, 4 * out_calib_njtr_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r95 = r40 * r40;
    r76 = r58 * r58;
    r100 = r9 * r9;
    r100 = r100 * r10;
    r100 = r100 * r71;
    r85 = r60 * r87;
    r97 = r8 * r54;
    r85 = fmaf(r97, r85, r54 * r100);
    r100 = r9 * r9;
    r3 = r54 * r54;
    r100 = r100 * r10;
    r100 = r100 * r71;
    r110 = r60 * r87;
    r110 = r110 * r97;
    r110 = fmaf(r54, r110, r3 * r100);
    WriteSum4<float, float>((float *)inout_shared, r95, r76, r85, r110);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r110 = r65 * r65;
    r85 = r8 * r8;
    r76 = 4.00000000000000000e+00;
    r53 = r59 * r53;
    r53 = 1.0 / r53;
    r85 = r85 * r9;
    r85 = r85 * r9;
    r85 = r85 * r76;
    r85 = r85 * r53;
    r110 = fmaf(r87, r85, r71 * r110);
    r53 = r52 * r52;
    r85 = fmaf(r71, r85, r87 * r53);
    WriteSum4<float, float>((float *)inout_shared, r110, r85, r26, r26);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           4 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = 0.00000000000000000e+00;
    r85 = r4 * r8;
    r85 = r85 * r42;
    r85 = r85 * r40;
    r85 = r85 * r41;
    r110 = r40 * r11;
    r53 = r4 * r9;
    r53 = r53 * r40;
    r53 = r53 * r63;
    WriteSum4<float, float>((float *)inout_shared, r26, r85, r110, r53);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = r4 * r52;
    r53 = r53 * r40;
    r110 = r5 * r9;
    r110 = r110 * r42;
    r110 = r110 * r58;
    r110 = r110 * r41;
    WriteSum4<float, float>((float *)inout_shared, r53, r40, r26, r110);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r110 = r5 * r65;
    r110 = r110 * r58;
    r40 = r5 * r9;
    r40 = r40 * r58;
    r40 = r40 * r41;
    r40 = r40 * r54;
    r53 = r5 * r9;
    r53 = r53 * r58;
    r53 = r53 * r63;
    WriteSum4<float, float>((float *)inout_shared, r40, r110, r53, r26);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           8 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = r9 * r9;
    r53 = r53 * r42;
    r53 = r53 * r10;
    r53 = r53 * r71;
    r110 = r42 * r60;
    r110 = r110 * r87;
    r110 = fmaf(r97, r110, r54 * r53);
    r53 = r32 * r8;
    r53 = r53 * r42;
    r53 = r53 * r87;
    r40 = r9 * r42;
    r40 = r40 * r65;
    r40 = r40 * r41;
    r40 = fmaf(r71, r40, r98 * r53);
    r53 = r32 * r9;
    r53 = r53 * r42;
    r53 = r53 * r71;
    r85 = r8 * r52;
    r85 = r85 * r42;
    r85 = r85 * r41;
    r85 = fmaf(r87, r85, r98 * r53);
    WriteSum4<float, float>((float *)inout_shared, r58, r110, r40, r85);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           12 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = r32 * r87;
    r85 = r85 * r98;
    r40 = r9 * r65;
    r40 = r40 * r41;
    r40 = r40 * r71;
    r40 = fmaf(r54, r40, r97 * r85);
    r85 = r32 * r9;
    r85 = r85 * r71;
    r85 = r85 * r98;
    r98 = r52 * r87;
    r97 = r41 * r97;
    r98 = fmaf(r97, r98, r54 * r85);
    WriteSum4<float, float>((float *)inout_shared, r70, r86, r40, r98);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           16 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r98 = r9 * r65;
    r98 = r98 * r71;
    r40 = r9 * r52;
    r40 = r40 * r87;
    r40 = fmaf(r63, r40, r63 * r98);
    WriteSum4<float, float>((float *)inout_shared, r11, r90, r40, r72);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           20 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float *)inout_shared, r92, r96, r102, r26);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           24 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = r31 * r84;
    r102 = r38 * r8;
    r102 = r102 * r8;
    r102 = r102 * r66;
    r102 = fmaf(r93, r102, r60 * r26);
    r26 = r32 * r61;
    r26 = r26 * r9;
    r96 = r25 * r38;
    r96 = r96 * r9;
    r96 = fmaf(r77, r96, r10 * r26);
    r102 = r102 + r96;
    r102 = fmaf(r31, r41, r1 * r102);
    r26 = r0 * r38;
    r102 = fmaf(r99, r26, r102);
    r92 = r0 * r32;
    r92 = r92 * r31;
    r92 = r92 * r9;
    r102 = fmaf(r10, r92, r102);
    r72 = r24 * r38;
    r102 = fmaf(r60, r72, r102);
    r40 = r24 * r38;
    r40 = r40 * r47;
    r102 = fmaf(r60, r40, r102);
    r90 = r25 * r38;
    r90 = r90 * r8;
    r90 = r90 * r8;
    r90 = fmaf(r93, r90, r31 * r63);
    r96 = r96 + r90;
    r11 = r7 * r32;
    r11 = r11 * r42;
    r11 = fmaf(r96, r11, r6 * r96);
    r96 = r8 * r11;
    r102 = fmaf(r41, r96, r102);
    r102 = fmaf(r31, r62, r102);
    r102 = fmaf(r61, r64, r102);
    r96 = r4 * r102;
    r40 = r61 * r9;
    r40 = r40 * r84;
    r40 = fmaf(r38, r88, r10 * r40);
    r40 = r40 + r90;
    r40 = fmaf(r38, r101, r0 * r40);
    r90 = r24 * r38;
    r90 = r90 * r9;
    r90 = r90 * r47;
    r40 = fmaf(r10, r90, r40);
    r72 = r9 * r11;
    r40 = fmaf(r41, r72, r40);
    r92 = r1 * r32;
    r92 = r92 * r31;
    r92 = r92 * r9;
    r40 = fmaf(r10, r92, r40);
    r26 = r24 * r38;
    r26 = r26 * r9;
    r40 = fmaf(r10, r26, r40);
    r98 = r1 * r61;
    r40 = fmaf(r63, r98, r40);
    r40 = fmaf(r61, r41, r40);
    r40 = fmaf(r61, r62, r40);
    r98 = r5 * r40;
    r26 = r37 * r84;
    r92 = r57 * r8;
    r92 = r92 * r8;
    r92 = r92 * r66;
    r92 = fmaf(r93, r92, r60 * r26);
    r26 = r25 * r57;
    r26 = r26 * r9;
    r72 = r32 * r23;
    r72 = r72 * r9;
    r72 = fmaf(r10, r72, r77 * r26);
    r92 = r92 + r72;
    r92 = fmaf(r23, r64, r1 * r92);
    r26 = r25 * r57;
    r26 = r26 * r8;
    r26 = r26 * r8;
    r26 = fmaf(r93, r26, r37 * r63);
    r72 = r72 + r26;
    r90 = r7 * r32;
    r90 = r90 * r42;
    r90 = fmaf(r72, r90, r6 * r72);
    r72 = r8 * r90;
    r92 = fmaf(r41, r72, r92);
    r86 = r0 * r32;
    r86 = r86 * r37;
    r86 = r86 * r9;
    r92 = fmaf(r10, r86, r92);
    r70 = r0 * r57;
    r92 = fmaf(r99, r70, r92);
    r85 = r24 * r57;
    r85 = r85 * r47;
    r92 = fmaf(r60, r85, r92);
    r97 = r24 * r57;
    r92 = fmaf(r60, r97, r92);
    r92 = fmaf(r37, r41, r92);
    r92 = fmaf(r37, r62, r92);
    r97 = r4 * r92;
    r85 = r23 * r9;
    r85 = r85 * r84;
    r85 = fmaf(r10, r85, r57 * r88);
    r85 = r85 + r26;
    r26 = r24 * r57;
    r26 = r26 * r9;
    r26 = r26 * r47;
    r26 = fmaf(r10, r26, r0 * r85);
    r85 = r24 * r57;
    r85 = r85 * r9;
    r26 = fmaf(r10, r85, r26);
    r70 = r1 * r23;
    r26 = fmaf(r63, r70, r26);
    r86 = r1 * r32;
    r86 = r86 * r37;
    r86 = r86 * r9;
    r26 = fmaf(r10, r86, r26);
    r72 = r9 * r90;
    r26 = fmaf(r41, r72, r26);
    r26 = fmaf(r23, r41, r26);
    r26 = fmaf(r23, r62, r26);
    r26 = fmaf(r57, r101, r26);
    r72 = r5 * r26;
    WriteIdx4<1024, float, float, float4>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r96, r98,
        r97, r72);
    r72 = r30 * r8;
    r72 = r72 * r8;
    r72 = r72 * r66;
    r66 = r39 * r84;
    r66 = fmaf(r60, r66, r93 * r72);
    r72 = r32 * r56;
    r72 = r72 * r9;
    r97 = r25 * r30;
    r97 = r97 * r9;
    r97 = fmaf(r77, r97, r10 * r72);
    r66 = r66 + r97;
    r72 = r25 * r30;
    r72 = r72 * r8;
    r72 = r72 * r8;
    r72 = fmaf(r39, r63, r93 * r72);
    r97 = r97 + r72;
    r93 = r7 * r32;
    r93 = r93 * r42;
    r93 = fmaf(r97, r93, r6 * r97);
    r97 = r8 * r93;
    r97 = fmaf(r41, r97, r1 * r66);
    r66 = r24 * r30;
    r97 = fmaf(r60, r66, r97);
    r6 = r24 * r30;
    r6 = r6 * r47;
    r97 = fmaf(r60, r6, r97);
    r77 = r0 * r30;
    r97 = fmaf(r99, r77, r97);
    r99 = r0 * r32;
    r99 = r99 * r39;
    r99 = r99 * r9;
    r97 = fmaf(r10, r99, r97);
    r97 = fmaf(r56, r64, r97);
    r97 = fmaf(r39, r41, r97);
    r97 = fmaf(r39, r62, r97);
    r99 = r4 * r97;
    r77 = r56 * r9;
    r77 = r77 * r84;
    r88 = fmaf(r30, r88, r10 * r77);
    r88 = r88 + r72;
    r88 = fmaf(r56, r41, r0 * r88);
    r72 = r1 * r56;
    r88 = fmaf(r63, r72, r88);
    r63 = r24 * r30;
    r63 = r63 * r9;
    r88 = fmaf(r10, r63, r88);
    r77 = r24 * r30;
    r77 = r77 * r9;
    r77 = r77 * r47;
    r88 = fmaf(r10, r77, r88);
    r6 = r9 * r93;
    r88 = fmaf(r41, r6, r88);
    r66 = r1 * r32;
    r66 = r66 * r39;
    r66 = r66 * r9;
    r88 = fmaf(r10, r66, r88);
    r88 = fmaf(r56, r62, r88);
    r88 = fmaf(r30, r101, r88);
    r66 = r5 * r88;
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx, r99, r66);
    r66 = r4 * r24;
    r66 = r66 * r2;
    r66 = fmaf(r40, r68, r102 * r66);
    r99 = r4 * r24;
    r99 = r99 * r2;
    r99 = fmaf(r26, r68, r92 * r99);
    r6 = r4 * r24;
    r6 = r6 * r2;
    r68 = fmaf(r88, r68, r97 * r6);
    WriteSum3<float, float>((float *)inout_shared, r66, r99, r68);
  };
  FlushSumShared<3, float>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = r102 * r102;
    r99 = r40 * r40;
    r99 = fmaf(r71, r99, r87 * r68);
    r68 = r92 * r92;
    r66 = r26 * r26;
    r66 = fmaf(r71, r66, r87 * r68);
    r68 = r97 * r97;
    r6 = r88 * r88;
    r6 = fmaf(r71, r6, r87 * r68);
    WriteSum3<float, float>((float *)inout_shared, r99, r66, r6);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r40 * r26;
    r66 = r102 * r92;
    r66 = fmaf(r87, r66, r71 * r6);
    r6 = r102 * r97;
    r99 = r40 * r88;
    r99 = fmaf(r71, r99, r87 * r6);
    r6 = r92 * r97;
    r68 = r26 * r88;
    r68 = fmaf(r71, r68, r87 * r6);
    WriteSum3<float, float>((float *)inout_shared, r66, r99, r68);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void OpencvResJacFirst(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    float *point, unsigned int point_num_alloc, SharedIndex *point_indices,
    float *pixel, unsigned int pixel_num_alloc, float *out_res,
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
    unsigned int out_calib_precond_tril_num_alloc, float *out_point_jac,
    unsigned int out_point_jac_num_alloc, float *const out_point_njtr,
    unsigned int out_point_njtr_num_alloc, float *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvResJacFirstKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, sensor_from_rig,
      sensor_from_rig_num_alloc, calib, calib_num_alloc, calib_indices, point,
      point_num_alloc, point_indices, pixel, pixel_num_alloc, out_res,
      out_res_num_alloc, out_rTr, out_pose_jac, out_pose_jac_num_alloc,
      out_pose_njtr, out_pose_njtr_num_alloc, out_pose_precond_diag,
      out_pose_precond_diag_num_alloc, out_pose_precond_tril,
      out_pose_precond_tril_num_alloc, out_calib_jac, out_calib_jac_num_alloc,
      out_calib_njtr, out_calib_njtr_num_alloc, out_calib_precond_diag,
      out_calib_precond_diag_num_alloc, out_calib_precond_tril,
      out_calib_precond_tril_num_alloc, out_point_jac, out_point_jac_num_alloc,
      out_point_njtr, out_point_njtr_num_alloc, out_point_precond_diag,
      out_point_precond_diag_num_alloc, out_point_precond_tril,
      out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar