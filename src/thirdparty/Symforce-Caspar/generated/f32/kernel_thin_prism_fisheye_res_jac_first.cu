#include "kernel_thin_prism_fisheye_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) ThinPrismFisheyeResJacFirstKernel(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    float* out_calib_jac,
    unsigned int out_calib_jac_num_alloc,
    float* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    float* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    float* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
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
      r105, r106, r107, r108, r109, r110, r111, r112, r113, r114, r115, r116,
      r117, r118, r119, r120, r121, r122, r123, r124, r125, r126, r127, r128,
      r129, r130, r131, r132, r133, r134, r135, r136, r137, r138, r139, r140,
      r141, r142, r143, r144, r145, r146, r147, r148, r149, r150, r151, r152,
      r153, r154, r155;
  LoadShared<4, float, float>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r0,
                       r1,
                       r2,
                       r3);
  };
  __syncthreads();
  LoadShared<4, float, float>(
      calib, 4 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r4,
                       r5,
                       r6,
                       r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r8,
                                         r9,
                                         r10);
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r11,
                       r12,
                       r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r14 = 2.00000000000000000e+00;
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r15,
                       r16,
                       r17,
                       r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r19,
                                         r20,
                                         r21,
                                         r22);
    r23 = fmaf(r18, r19, r15 * r22);
    r24 = r16 * r21;
    r25 = -1.00000000000000000e+00;
    r23 = fmaf(r25, r24, r23);
    r23 = fmaf(r17, r20, r23);
    r24 = r14 * r23;
    r26 = r17 * r19;
    r26 = fmaf(r25, r26, r16 * r22);
    r26 = fmaf(r18, r20, r26);
    r26 = fmaf(r15, r21, r26);
    r24 = r24 * r26;
    r27 = r17 * r22;
    r28 = r16 * r19;
    r29 = r27 + r28;
    r30 = r18 * r21;
    r31 = r15 * r20;
    r29 = r29 + r30;
    r29 = fmaf(r25, r31, r29);
    r32 = r14 * r29;
    r33 = fmaf(r16, r20, r15 * r19);
    r33 = fmaf(r17, r21, r33);
    r33 = fmaf(r25, r33, r18 * r22);
    r32 = fmaf(r33, r32, r24);
    r9 = fmaf(r11, r32, r9);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r34,
                       r35,
                       r36);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r37 = r19 * r20;
    r37 = r37 * r14;
    r38 = r21 * r22;
    r38 = fmaf(r14, r38, r37);
    r39 = r21 * r21;
    r40 = -2.00000000000000000e+00;
    r39 = r39 * r40;
    r41 = 1.00000000000000000e+00;
    r42 = r19 * r19;
    r42 = fmaf(r40, r42, r41);
    r43 = r39 + r42;
    r44 = r20 * r21;
    r44 = r44 * r14;
    r45 = r19 * r22;
    r45 = fmaf(r40, r45, r44);
    r46 = r14 * r29;
    r46 = r46 * r26;
    r47 = r23 * r40;
    r47 = fmaf(r33, r47, r46);
    r48 = r23 * r23;
    r48 = r48 * r40;
    r49 = r41 + r48;
    r50 = r29 * r29;
    r50 = r50 * r40;
    r49 = r49 + r50;
    r9 = fmaf(r34, r38, r9);
    r9 = fmaf(r35, r43, r9);
    r9 = fmaf(r36, r45, r9);
    r9 = fmaf(r13, r47, r9);
    r9 = fmaf(r12, r49, r9);
    r51 = r9 * r9;
    r52 = r40 * r26;
    r52 = r52 * r26;
    r53 = r41 + r52;
    r53 = r53 + r50;
    r8 = fmaf(r11, r53, r8);
    r50 = r29 * r40;
    r50 = fmaf(r33, r50, r24);
    r24 = r14 * r29;
    r24 = r24 * r23;
    r54 = r14 * r26;
    r54 = fmaf(r33, r54, r24);
    r55 = r19 * r21;
    r55 = r55 * r14;
    r56 = r20 * r22;
    r56 = fmaf(r14, r56, r55);
    r57 = r21 * r22;
    r57 = fmaf(r40, r57, r37);
    r37 = r20 * r20;
    r37 = r37 * r40;
    r58 = r41 + r37;
    r58 = r58 + r39;
    r8 = fmaf(r12, r50, r8);
    r8 = fmaf(r13, r54, r8);
    r8 = fmaf(r36, r56, r8);
    r8 = fmaf(r35, r57, r8);
    r8 = fmaf(r34, r58, r8);
    r39 = r8 * r8;
    r59 = 9.99999999999999955e-07;
    r60 = r40 * r26;
    r60 = fmaf(r33, r60, r24);
    r10 = fmaf(r11, r60, r10);
    r24 = r20 * r22;
    r24 = fmaf(r40, r24, r55);
    r42 = r37 + r42;
    r37 = r19 * r22;
    r37 = fmaf(r14, r37, r44);
    r44 = r14 * r23;
    r44 = fmaf(r33, r44, r46);
    r52 = r41 + r52;
    r52 = r52 + r48;
    r10 = fmaf(r34, r24, r10);
    r10 = fmaf(r36, r42, r10);
    r10 = fmaf(r35, r37, r10);
    r10 = fmaf(r12, r44, r10);
    r10 = fmaf(r13, r52, r10);
    r35 = copysign(1.0, r10);
    r35 = fmaf(r59, r35, r10);
    r10 = r35 * r35;
    r36 = 1.0 / r10;
    r34 = r9 * r9;
    r34 = fmaf(r36, r34, r36 * r39);
    r39 = sqrtf(r34);
    r48 = atanf(r39);
    r46 = copysign(1.0, r39);
    r46 = fmaf(r59, r46, r39);
    r59 = r46 * r46;
    r39 = 1.0 / r59;
    r55 = r48 * r36;
    r61 = r39 * r55;
    r51 = r51 * r48;
    r51 = r51 * r61;
    r62 = 3.00000000000000000e+00;
    r63 = r8 * r62;
    r64 = r8 * r48;
    r63 = r63 * r61;
    r63 = fmaf(r64, r63, r51);
  };
  LoadShared<4, float, float>(
      calib, 8 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r65,
                       r66,
                       r67,
                       r68);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r69 = r8 * r61;
    r69 = r69 * r64;
    r51 = r51 + r69;
    r70 = fmaf(r67, r51, r7 * r63);
    r71 = r6 * r9;
    r72 = r14 * r61;
    r73 = r64 * r72;
    r70 = fmaf(r73, r71, r70);
    r74 = r51 * r51;
    r75 = fmaf(r5, r74, r4 * r51);
    r76 = r74 * r74;
    r77 = r51 * r74;
    r75 = fmaf(r66, r76, r75);
    r75 = fmaf(r65, r77, r75);
    r78 = 1.0 / r35;
    r79 = 1.0 / r46;
    r80 = r78 * r79;
    r81 = r75 * r80;
    r70 = fmaf(r64, r81, r70);
    r70 = fmaf(r80, r64, r70);
    r71 = r0 * r70;
    r2 = r2 + r71;
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r82, r83);
    r2 = fmaf(r82, r25, r2);
    r82 = r9 * r9;
    r82 = r82 * r48;
    r82 = r82 * r62;
    r82 = fmaf(r61, r82, r69);
    r69 = fmaf(r68, r51, r6 * r82);
    r84 = r7 * r73;
    r85 = r9 * r48;
    r69 = fmaf(r80, r85, r69);
    r86 = r9 * r48;
    r69 = fmaf(r81, r86, r69);
    r69 = fmaf(r9, r84, r69);
    r3 = fmaf(r1, r69, r3);
    r3 = fmaf(r83, r25, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r83 = fmaf(r3, r3, r2 * r2);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r83);
  if (global_thread_idx < problem_size) {
    r83 = r23 * r40;
    r86 = r15 * r22;
    r85 = -5.00000000000000000e-01;
    r87 = r18 * r19;
    r87 = fmaf(r85, r87, r85 * r86);
    r86 = r17 * r20;
    r87 = fmaf(r85, r86, r87);
    r88 = r16 * r21;
    r89 = 5.00000000000000000e-01;
    r87 = fmaf(r89, r88, r87);
    r88 = r18 * r22;
    r86 = r15 * r19;
    r86 = fmaf(r85, r86, r89 * r88);
    r88 = r16 * r20;
    r86 = fmaf(r85, r88, r86);
    r90 = r17 * r21;
    r86 = fmaf(r85, r90, r86);
    r90 = r33 * r86;
    r88 = r40 * r90;
    r83 = fmaf(r87, r83, r88);
    r91 = r14 * r26;
    r92 = r16 * r22;
    r93 = r17 * r19;
    r93 = fmaf(r89, r93, r85 * r92);
    r92 = r18 * r20;
    r93 = fmaf(r85, r92, r93);
    r94 = r15 * r21;
    r93 = fmaf(r85, r94, r93);
    r91 = r91 * r93;
    r94 = r14 * r29;
    r92 = fmaf(r89, r28, r89 * r27);
    r92 = fmaf(r85, r31, r92);
    r92 = fmaf(r89, r30, r92);
    r94 = fmaf(r92, r94, r91);
    r83 = r83 + r94;
    r95 = r23 * r86;
    r96 = -4.00000000000000000e+00;
    r95 = r95 * r96;
    r97 = r29 * r93;
    r98 = r96 * r97;
    r99 = r95 + r98;
    r99 = fmaf(r12, r99, r13 * r83);
    r83 = r14 * r23;
    r83 = r83 * r92;
    r100 = r14 * r33;
    r100 = fmaf(r93, r100, r83);
    r101 = r14 * r26;
    r101 = r101 * r86;
    r102 = r14 * r29;
    r102 = fmaf(r87, r102, r101);
    r100 = r100 + r102;
    r99 = fmaf(r11, r100, r99);
    r100 = r99 * r72;
    r103 = r9 * r48;
    r104 = r14 * r9;
    r104 = r104 * r99;
    r105 = r14 * r8;
    r106 = r14 * r33;
    r107 = r26 * r87;
    r106 = fmaf(r14, r107, r92 * r106);
    r108 = r14 * r23;
    r109 = r14 * r29;
    r109 = r109 * r86;
    r108 = fmaf(r93, r108, r109);
    r106 = r106 + r108;
    r83 = r101 + r83;
    r101 = r29 * r40;
    r83 = fmaf(r87, r101, r83);
    r110 = r40 * r33;
    r83 = fmaf(r93, r110, r83);
    r83 = fmaf(r12, r83, r13 * r106);
    r106 = r26 * r92;
    r106 = r106 * r96;
    r98 = r106 + r98;
    r83 = fmaf(r11, r98, r83);
    r105 = r105 * r83;
    r105 = fmaf(r36, r105, r36 * r104);
    r104 = r9 * r9;
    r98 = r14 * r23;
    r98 = r98 * r87;
    r90 = r14 * r90;
    r110 = r98 + r90;
    r94 = r94 + r110;
    r101 = r40 * r33;
    r101 = fmaf(r40, r107, r92 * r101);
    r101 = r101 + r108;
    r101 = fmaf(r11, r101, r12 * r94);
    r106 = r95 + r106;
    r101 = fmaf(r13, r106, r101);
    r10 = r35 * r10;
    r106 = 1.0 / r10;
    r95 = r40 * r106;
    r104 = r104 * r101;
    r105 = fmaf(r95, r104, r105);
    r94 = r8 * r8;
    r94 = r94 * r101;
    r105 = fmaf(r95, r94, r105);
    r94 = r41 + r34;
    r94 = 1.0 / r94;
    r34 = rsqrtf(r34);
    r104 = r9 * r34;
    r92 = r94 * r104;
    r111 = r105 * r92;
    r112 = r9 * r61;
    r111 = fmaf(r112, r111, r103 * r100);
    r100 = r25 * r105;
    r113 = r104 * r103;
    r59 = r46 * r59;
    r114 = 1.0 / r59;
    r115 = r114 * r55;
    r113 = r113 * r115;
    r111 = fmaf(r113, r100, r111);
    r116 = r9 * r9;
    r117 = r39 * r95;
    r118 = r48 * r48;
    r117 = r117 * r118;
    r116 = r116 * r101;
    r111 = fmaf(r117, r116, r111);
    r116 = r8 * r8;
    r116 = r116 * r105;
    r116 = r116 * r94;
    r116 = r116 * r34;
    r116 = fmaf(r61, r116, r83 * r73);
    r100 = r8 * r8;
    r100 = r100 * r101;
    r116 = fmaf(r117, r100, r116);
    r119 = r64 * r115;
    r120 = r8 * r34;
    r119 = r119 * r120;
    r121 = r105 * r119;
    r116 = fmaf(r25, r121, r116);
    r100 = r111 + r116;
    r122 = 6.00000000000000000e+00;
    r123 = r83 * r122;
    r123 = r123 * r61;
    r124 = r8 * r8;
    r124 = r124 * r62;
    r124 = r124 * r105;
    r124 = r124 * r94;
    r124 = r124 * r34;
    r124 = fmaf(r61, r124, r64 * r123);
    r123 = r8 * r8;
    r125 = r48 * r48;
    r126 = -6.00000000000000000e+00;
    r125 = r125 * r101;
    r125 = r125 * r126;
    r125 = r125 * r39;
    r125 = r125 * r106;
    r127 = -3.00000000000000000e+00;
    r124 = fmaf(r125, r123, r124);
    r124 = fmaf(r127, r121, r124);
    r124 = r124 + r111;
    r124 = fmaf(r7, r124, r67 * r100);
    r111 = r89 * r94;
    r111 = r111 * r81;
    r111 = r111 * r120;
    r120 = r6 * r99;
    r124 = fmaf(r73, r120, r124);
    r121 = r48 * r83;
    r124 = fmaf(r81, r121, r124);
    r128 = r8 * r48;
    r128 = r128 * r85;
    r128 = r128 * r105;
    r128 = r128 * r39;
    r128 = r128 * r78;
    r124 = fmaf(r34, r128, r124);
    r129 = r6 * r8;
    r129 = r129 * r105;
    r129 = r129 * r72;
    r124 = fmaf(r92, r129, r124);
    r130 = r48 * r83;
    r124 = fmaf(r80, r130, r124);
    r131 = r25 * r8;
    r131 = r131 * r101;
    r131 = r131 * r79;
    r124 = fmaf(r55, r131, r124);
    r132 = r5 * r14;
    r132 = r132 * r51;
    r132 = fmaf(r100, r132, r4 * r100);
    r65 = r65 * r62;
    r65 = r65 * r74;
    r133 = 4.00000000000000000e+00;
    r66 = r66 * r133;
    r66 = r66 * r77;
    r132 = fmaf(r100, r65, r132);
    r132 = fmaf(r100, r66, r132);
    r134 = r132 * r80;
    r124 = fmaf(r64, r134, r124);
    r135 = r6 * r83;
    r135 = r135 * r72;
    r124 = fmaf(r103, r135, r124);
    r136 = r6 * r8;
    r136 = r136 * r9;
    r136 = r136 * r48;
    r136 = r136 * r48;
    r136 = r136 * r96;
    r136 = r136 * r39;
    r136 = r136 * r106;
    r137 = r8 * r48;
    r137 = r137 * r75;
    r137 = r137 * r85;
    r137 = r137 * r105;
    r137 = r137 * r39;
    r137 = r137 * r78;
    r124 = fmaf(r34, r137, r124);
    r138 = r25 * r8;
    r138 = r138 * r75;
    r138 = r138 * r101;
    r138 = r138 * r79;
    r124 = fmaf(r55, r138, r124);
    r139 = r8 * r89;
    r139 = r139 * r105;
    r139 = r139 * r94;
    r139 = r139 * r34;
    r124 = fmaf(r80, r139, r124);
    r140 = r6 * r40;
    r140 = r140 * r105;
    r140 = r140 * r64;
    r140 = r140 * r104;
    r124 = fmaf(r115, r140, r124);
    r124 = fmaf(r105, r111, r124);
    r124 = fmaf(r101, r136, r124);
    r140 = r0 * r124;
    r139 = r9 * r48;
    r139 = r139 * r99;
    r139 = r139 * r122;
    r138 = r62 * r105;
    r138 = r138 * r92;
    r138 = fmaf(r112, r138, r61 * r139);
    r139 = r127 * r113;
    r137 = r9 * r9;
    r138 = fmaf(r105, r139, r138);
    r138 = fmaf(r125, r137, r138);
    r138 = r138 + r116;
    r138 = fmaf(r6, r138, r68 * r100);
    r100 = r48 * r85;
    r100 = r100 * r39;
    r100 = r100 * r78;
    r100 = r100 * r104;
    r116 = r75 * r100;
    r125 = r89 * r92;
    r135 = r80 * r125;
    r134 = r7 * r8;
    r134 = r134 * r105;
    r134 = r134 * r72;
    r138 = fmaf(r92, r134, r138);
    r131 = r105 * r81;
    r138 = fmaf(r125, r131, r138);
    r130 = r25 * r9;
    r130 = r130 * r75;
    r130 = r130 * r101;
    r130 = r130 * r79;
    r138 = fmaf(r55, r130, r138);
    r129 = r9 * r48;
    r129 = r129 * r132;
    r138 = fmaf(r80, r129, r138);
    r128 = r7 * r83;
    r128 = r128 * r72;
    r138 = fmaf(r103, r128, r138);
    r121 = r7 * r8;
    r121 = r121 * r9;
    r121 = r121 * r48;
    r121 = r121 * r48;
    r121 = r121 * r96;
    r121 = r121 * r101;
    r121 = r121 * r39;
    r138 = fmaf(r106, r121, r138);
    r120 = r25 * r9;
    r120 = r120 * r101;
    r120 = r120 * r79;
    r138 = fmaf(r55, r120, r138);
    r101 = r7 * r40;
    r101 = r101 * r105;
    r101 = r101 * r64;
    r101 = r101 * r104;
    r138 = fmaf(r115, r101, r138);
    r141 = r48 * r99;
    r138 = fmaf(r81, r141, r138);
    r142 = r48 * r99;
    r138 = fmaf(r80, r142, r138);
    r138 = fmaf(r105, r116, r138);
    r138 = fmaf(r105, r135, r138);
    r138 = fmaf(r99, r84, r138);
    r138 = fmaf(r105, r100, r138);
    r142 = r1 * r138;
    r141 = r8 * r8;
    r101 = r14 * r8;
    r90 = r91 + r90;
    r91 = r14 * r29;
    r28 = fmaf(r85, r28, r85 * r27);
    r28 = fmaf(r89, r31, r28);
    r28 = fmaf(r85, r30, r28);
    r91 = r91 * r28;
    r30 = r14 * r23;
    r31 = r15 * r22;
    r27 = r18 * r19;
    r27 = fmaf(r89, r27, r89 * r31);
    r31 = r17 * r20;
    r27 = fmaf(r89, r31, r27);
    r120 = r16 * r21;
    r27 = fmaf(r85, r120, r27);
    r30 = fmaf(r27, r30, r91);
    r90 = r90 + r30;
    r120 = r26 * r86;
    r120 = r120 * r96;
    r31 = r29 * r96;
    r31 = r31 * r27;
    r121 = r120 + r31;
    r121 = fmaf(r11, r121, r13 * r90);
    r90 = r40 * r33;
    r90 = fmaf(r40, r97, r27 * r90);
    r128 = r14 * r23;
    r128 = r128 * r86;
    r129 = r14 * r26;
    r129 = fmaf(r28, r129, r128);
    r90 = r90 + r129;
    r121 = fmaf(r12, r90, r121);
    r101 = r101 * r121;
    r90 = r8 * r8;
    r130 = r40 * r26;
    r130 = fmaf(r93, r130, r88);
    r130 = r130 + r30;
    r30 = r14 * r26;
    r30 = r30 * r27;
    r131 = r14 * r33;
    r131 = fmaf(r28, r131, r30);
    r131 = r131 + r108;
    r131 = fmaf(r12, r131, r11 * r130);
    r130 = r23 * r28;
    r108 = r96 * r130;
    r120 = r120 + r108;
    r131 = fmaf(r13, r120, r131);
    r90 = r90 * r131;
    r90 = fmaf(r95, r90, r36 * r101);
    r101 = r9 * r9;
    r101 = r101 * r131;
    r90 = fmaf(r95, r101, r90);
    r120 = r14 * r9;
    r30 = r109 + r30;
    r109 = r23 * r40;
    r30 = fmaf(r93, r109, r30);
    r93 = r40 * r33;
    r30 = fmaf(r28, r93, r30);
    r93 = r14 * r33;
    r97 = fmaf(r14, r97, r27 * r93);
    r97 = r97 + r129;
    r97 = fmaf(r11, r97, r13 * r30);
    r108 = r31 + r108;
    r97 = fmaf(r12, r108, r97);
    r120 = r120 * r97;
    r90 = fmaf(r36, r120, r90);
    r141 = r141 * r90;
    r141 = r141 * r94;
    r141 = r141 * r34;
    r120 = r25 * r90;
    r120 = fmaf(r119, r120, r61 * r141);
    r141 = r8 * r8;
    r141 = r141 * r131;
    r120 = fmaf(r117, r141, r120);
    r120 = fmaf(r121, r73, r120);
    r141 = r97 * r72;
    r101 = r25 * r90;
    r101 = fmaf(r113, r101, r103 * r141);
    r141 = r9 * r9;
    r141 = r141 * r131;
    r101 = fmaf(r117, r141, r101);
    r108 = r90 * r92;
    r101 = fmaf(r112, r108, r101);
    r108 = r120 + r101;
    r141 = r8 * r8;
    r141 = r141 * r62;
    r141 = r141 * r90;
    r141 = r141 * r94;
    r141 = r141 * r34;
    r31 = r127 * r90;
    r31 = fmaf(r119, r31, r61 * r141);
    r141 = r8 * r8;
    r141 = r141 * r48;
    r141 = r141 * r48;
    r141 = r141 * r126;
    r141 = r141 * r131;
    r141 = r141 * r39;
    r31 = fmaf(r106, r141, r31);
    r30 = r122 * r121;
    r30 = r30 * r61;
    r31 = fmaf(r64, r30, r31);
    r31 = r31 + r101;
    r31 = fmaf(r7, r31, r67 * r108);
    r101 = r6 * r121;
    r101 = r101 * r72;
    r31 = fmaf(r103, r101, r31);
    r30 = r8 * r48;
    r30 = r30 * r85;
    r30 = r30 * r90;
    r30 = r30 * r39;
    r30 = r30 * r78;
    r31 = fmaf(r34, r30, r31);
    r141 = r25 * r8;
    r141 = r141 * r131;
    r141 = r141 * r79;
    r31 = fmaf(r55, r141, r31);
    r93 = r8 * r48;
    r93 = r93 * r75;
    r93 = r93 * r85;
    r93 = r93 * r90;
    r93 = r93 * r39;
    r93 = r93 * r78;
    r31 = fmaf(r34, r93, r31);
    r27 = r48 * r121;
    r31 = fmaf(r81, r27, r31);
    r109 = r6 * r8;
    r109 = r109 * r90;
    r109 = r109 * r72;
    r31 = fmaf(r92, r109, r31);
    r134 = r25 * r8;
    r134 = r134 * r75;
    r134 = r134 * r131;
    r134 = r134 * r79;
    r31 = fmaf(r55, r134, r31);
    r143 = r5 * r14;
    r143 = r143 * r51;
    r143 = fmaf(r4, r108, r108 * r143);
    r143 = fmaf(r108, r66, r143);
    r143 = fmaf(r108, r65, r143);
    r144 = r143 * r80;
    r31 = fmaf(r64, r144, r31);
    r145 = r8 * r89;
    r145 = r145 * r90;
    r145 = r145 * r94;
    r145 = r145 * r34;
    r31 = fmaf(r80, r145, r31);
    r146 = r6 * r40;
    r146 = r146 * r90;
    r146 = r146 * r64;
    r146 = r146 * r104;
    r31 = fmaf(r115, r146, r31);
    r147 = r6 * r97;
    r31 = fmaf(r73, r147, r31);
    r148 = r48 * r121;
    r31 = fmaf(r80, r148, r31);
    r31 = fmaf(r131, r136, r31);
    r31 = fmaf(r90, r111, r31);
    r148 = r0 * r31;
    r147 = r9 * r48;
    r147 = r147 * r122;
    r147 = r147 * r97;
    r147 = fmaf(r90, r139, r61 * r147);
    r146 = r9 * r9;
    r146 = r146 * r48;
    r146 = r146 * r48;
    r146 = r146 * r126;
    r146 = r146 * r131;
    r146 = r146 * r39;
    r147 = fmaf(r106, r146, r147);
    r145 = r62 * r90;
    r145 = r145 * r92;
    r147 = fmaf(r112, r145, r147);
    r147 = r147 + r120;
    r147 = fmaf(r6, r147, r68 * r108);
    r108 = r25 * r9;
    r108 = r108 * r131;
    r108 = r108 * r79;
    r147 = fmaf(r55, r108, r147);
    r120 = r48 * r97;
    r147 = fmaf(r81, r120, r147);
    r145 = r7 * r121;
    r145 = r145 * r72;
    r147 = fmaf(r103, r145, r147);
    r146 = r9 * r48;
    r146 = r146 * r143;
    r147 = fmaf(r80, r146, r147);
    r144 = r90 * r81;
    r147 = fmaf(r125, r144, r147);
    r134 = r7 * r8;
    r134 = r134 * r90;
    r134 = r134 * r72;
    r147 = fmaf(r92, r134, r147);
    r109 = r25 * r9;
    r109 = r109 * r75;
    r109 = r109 * r131;
    r109 = r109 * r79;
    r147 = fmaf(r55, r109, r147);
    r27 = r7 * r8;
    r27 = r27 * r9;
    r27 = r27 * r48;
    r27 = r27 * r48;
    r27 = r27 * r96;
    r27 = r27 * r131;
    r27 = r27 * r39;
    r147 = fmaf(r106, r27, r147);
    r131 = r7 * r40;
    r131 = r131 * r90;
    r131 = r131 * r64;
    r131 = r131 * r104;
    r147 = fmaf(r115, r131, r147);
    r93 = r48 * r97;
    r147 = fmaf(r80, r93, r147);
    r147 = fmaf(r90, r116, r147);
    r147 = fmaf(r90, r135, r147);
    r147 = fmaf(r90, r100, r147);
    r147 = fmaf(r97, r84, r147);
    r93 = r1 * r147;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r140,
                                          r142,
                                          r148,
                                          r93);
    r93 = r8 * r8;
    r148 = r23 * r96;
    r142 = r16 * r22;
    r140 = r17 * r19;
    r140 = fmaf(r85, r140, r89 * r142);
    r142 = r18 * r20;
    r140 = fmaf(r89, r142, r140);
    r131 = r15 * r21;
    r140 = fmaf(r89, r131, r140);
    r148 = r148 * r140;
    r107 = r96 * r107;
    r131 = r148 + r107;
    r142 = r14 * r29;
    r142 = r142 * r140;
    r128 = r128 + r142;
    r27 = r40 * r26;
    r128 = fmaf(r28, r27, r128);
    r109 = r40 * r33;
    r128 = fmaf(r87, r109, r128);
    r128 = fmaf(r11, r128, r13 * r131);
    r131 = r14 * r33;
    r131 = fmaf(r14, r130, r140 * r131);
    r131 = r131 + r102;
    r128 = fmaf(r12, r131, r128);
    r93 = r93 * r128;
    r131 = r14 * r8;
    r109 = r14 * r26;
    r109 = r109 * r140;
    r98 = r98 + r109;
    r27 = r29 * r40;
    r98 = fmaf(r28, r27, r98);
    r98 = r98 + r88;
    r86 = r29 * r86;
    r86 = r86 * r96;
    r107 = r86 + r107;
    r107 = fmaf(r11, r107, r12 * r98);
    r98 = r14 * r33;
    r98 = fmaf(r87, r98, r142);
    r98 = r98 + r129;
    r107 = fmaf(r13, r98, r107);
    r131 = r131 * r107;
    r131 = fmaf(r36, r131, r95 * r93);
    r93 = r9 * r9;
    r93 = r93 * r128;
    r131 = fmaf(r95, r93, r131);
    r98 = r14 * r9;
    r109 = r91 + r109;
    r109 = r109 + r110;
    r110 = r40 * r33;
    r130 = fmaf(r40, r130, r140 * r110);
    r130 = r130 + r102;
    r130 = fmaf(r13, r130, r11 * r109);
    r148 = r86 + r148;
    r130 = fmaf(r12, r148, r130);
    r98 = r98 * r130;
    r131 = fmaf(r36, r98, r131);
    r98 = r131 * r92;
    r93 = r130 * r72;
    r93 = fmaf(r103, r93, r112 * r98);
    r98 = r9 * r9;
    r98 = r98 * r128;
    r93 = fmaf(r117, r98, r93);
    r148 = r25 * r131;
    r93 = fmaf(r113, r148, r93);
    r148 = r8 * r8;
    r148 = r148 * r128;
    r98 = r8 * r8;
    r98 = r98 * r131;
    r98 = r98 * r94;
    r98 = r98 * r34;
    r98 = fmaf(r61, r98, r117 * r148);
    r148 = r25 * r131;
    r98 = fmaf(r119, r148, r98);
    r98 = fmaf(r107, r73, r98);
    r148 = r93 + r98;
    r12 = r8 * r8;
    r12 = r12 * r48;
    r12 = r12 * r48;
    r12 = r12 * r126;
    r12 = r12 * r128;
    r12 = r12 * r39;
    r86 = r8 * r8;
    r86 = r86 * r62;
    r86 = r86 * r131;
    r86 = r86 * r94;
    r86 = r86 * r34;
    r86 = fmaf(r61, r86, r106 * r12);
    r12 = r127 * r131;
    r86 = fmaf(r119, r12, r86);
    r13 = r122 * r107;
    r13 = r13 * r61;
    r86 = fmaf(r64, r13, r86);
    r86 = r86 + r93;
    r86 = fmaf(r7, r86, r67 * r148);
    r93 = r6 * r130;
    r86 = fmaf(r73, r93, r86);
    r13 = r48 * r107;
    r86 = fmaf(r81, r13, r86);
    r12 = r8 * r89;
    r12 = r12 * r131;
    r12 = r12 * r94;
    r12 = r12 * r34;
    r86 = fmaf(r80, r12, r86);
    r109 = r5 * r14;
    r109 = r109 * r51;
    r109 = fmaf(r148, r109, r4 * r148);
    r109 = fmaf(r148, r65, r109);
    r109 = fmaf(r148, r66, r109);
    r11 = r109 * r80;
    r86 = fmaf(r64, r11, r86);
    r102 = r25 * r8;
    r102 = r102 * r128;
    r102 = r102 * r79;
    r86 = fmaf(r55, r102, r86);
    r110 = r6 * r8;
    r110 = r110 * r131;
    r110 = r110 * r72;
    r86 = fmaf(r92, r110, r86);
    r140 = r8 * r48;
    r140 = r140 * r75;
    r140 = r140 * r85;
    r140 = r140 * r131;
    r140 = r140 * r39;
    r140 = r140 * r78;
    r86 = fmaf(r34, r140, r86);
    r91 = r25 * r8;
    r91 = r91 * r75;
    r91 = r91 * r128;
    r91 = r91 * r79;
    r86 = fmaf(r55, r91, r86);
    r129 = r8 * r48;
    r129 = r129 * r85;
    r129 = r129 * r131;
    r129 = r129 * r39;
    r129 = r129 * r78;
    r86 = fmaf(r34, r129, r86);
    r142 = r6 * r40;
    r142 = r142 * r131;
    r142 = r142 * r64;
    r142 = r142 * r104;
    r86 = fmaf(r115, r142, r86);
    r87 = r6 * r107;
    r87 = r87 * r72;
    r86 = fmaf(r103, r87, r86);
    r88 = r48 * r107;
    r86 = fmaf(r80, r88, r86);
    r86 = fmaf(r128, r136, r86);
    r86 = fmaf(r131, r111, r86);
    r88 = r0 * r86;
    r87 = r62 * r131;
    r87 = r87 * r92;
    r142 = r9 * r48;
    r142 = r142 * r122;
    r142 = r142 * r130;
    r142 = fmaf(r61, r142, r112 * r87);
    r87 = r9 * r9;
    r87 = r87 * r48;
    r87 = r87 * r48;
    r87 = r87 * r126;
    r87 = r87 * r128;
    r87 = r87 * r39;
    r142 = fmaf(r106, r87, r142);
    r142 = fmaf(r131, r139, r142);
    r142 = r142 + r98;
    r142 = fmaf(r6, r142, r68 * r148);
    r148 = r9 * r48;
    r148 = r148 * r109;
    r142 = fmaf(r80, r148, r142);
    r98 = r48 * r130;
    r142 = fmaf(r81, r98, r142);
    r87 = r7 * r8;
    r87 = r87 * r131;
    r87 = r87 * r72;
    r142 = fmaf(r92, r87, r142);
    r129 = r7 * r8;
    r129 = r129 * r9;
    r129 = r129 * r48;
    r129 = r129 * r48;
    r129 = r129 * r96;
    r129 = r129 * r128;
    r129 = r129 * r39;
    r142 = fmaf(r106, r129, r142);
    r91 = r25 * r9;
    r91 = r91 * r75;
    r91 = r91 * r128;
    r91 = r91 * r79;
    r142 = fmaf(r55, r91, r142);
    r140 = r131 * r81;
    r142 = fmaf(r125, r140, r142);
    r110 = r25 * r9;
    r110 = r110 * r128;
    r110 = r110 * r79;
    r142 = fmaf(r55, r110, r142);
    r128 = r7 * r40;
    r128 = r128 * r131;
    r128 = r128 * r64;
    r128 = r128 * r104;
    r142 = fmaf(r115, r128, r142);
    r102 = r7 * r107;
    r102 = r102 * r72;
    r142 = fmaf(r103, r102, r142);
    r11 = r48 * r130;
    r142 = fmaf(r80, r11, r142);
    r142 = fmaf(r130, r84, r142);
    r142 = fmaf(r131, r100, r142);
    r142 = fmaf(r131, r135, r142);
    r142 = fmaf(r131, r116, r142);
    r11 = r1 * r142;
    r102 = r24 * r8;
    r102 = r102 * r8;
    r102 = r102 * r48;
    r102 = r102 * r48;
    r102 = r102 * r126;
    r102 = r102 * r39;
    r128 = r58 * r122;
    r128 = r128 * r61;
    r128 = fmaf(r64, r128, r106 * r102);
    r102 = r24 * r9;
    r102 = r102 * r9;
    r110 = r14 * r38;
    r110 = r110 * r9;
    r110 = fmaf(r36, r110, r95 * r102);
    r102 = r24 * r8;
    r102 = r102 * r8;
    r110 = fmaf(r95, r102, r110);
    r140 = r14 * r58;
    r140 = r140 * r8;
    r110 = fmaf(r36, r140, r110);
    r140 = r127 * r110;
    r128 = fmaf(r119, r140, r128);
    r102 = r8 * r8;
    r102 = r102 * r62;
    r102 = r102 * r110;
    r102 = r102 * r94;
    r102 = r102 * r34;
    r128 = fmaf(r61, r102, r128);
    r91 = r24 * r9;
    r91 = r91 * r9;
    r129 = r38 * r72;
    r129 = fmaf(r103, r129, r117 * r91);
    r91 = r25 * r110;
    r129 = fmaf(r113, r91, r129);
    r87 = r110 * r92;
    r129 = fmaf(r112, r87, r129);
    r128 = r128 + r129;
    r102 = r24 * r8;
    r102 = r102 * r8;
    r102 = fmaf(r58, r73, r117 * r102);
    r140 = r25 * r110;
    r102 = fmaf(r119, r140, r102);
    r87 = r8 * r8;
    r87 = r87 * r110;
    r87 = r87 * r94;
    r87 = r87 * r34;
    r102 = fmaf(r61, r87, r102);
    r129 = r129 + r102;
    r128 = fmaf(r67, r129, r7 * r128);
    r87 = r25 * r24;
    r87 = r87 * r8;
    r87 = r87 * r79;
    r128 = fmaf(r55, r87, r128);
    r140 = r8 * r48;
    r140 = r140 * r75;
    r140 = r140 * r85;
    r140 = r140 * r110;
    r140 = r140 * r39;
    r140 = r140 * r78;
    r128 = fmaf(r34, r140, r128);
    r91 = r25 * r24;
    r91 = r91 * r8;
    r91 = r91 * r75;
    r91 = r91 * r79;
    r128 = fmaf(r55, r91, r128);
    r98 = r58 * r48;
    r128 = fmaf(r80, r98, r128);
    r148 = r6 * r8;
    r148 = r148 * r110;
    r148 = r148 * r72;
    r128 = fmaf(r92, r148, r128);
    r12 = r6 * r58;
    r12 = r12 * r72;
    r128 = fmaf(r103, r12, r128);
    r13 = r58 * r48;
    r128 = fmaf(r81, r13, r128);
    r93 = r8 * r48;
    r93 = r93 * r85;
    r93 = r93 * r110;
    r93 = r93 * r39;
    r93 = r93 * r78;
    r128 = fmaf(r34, r93, r128);
    r27 = r8 * r89;
    r27 = r27 * r110;
    r27 = r27 * r94;
    r27 = r27 * r34;
    r128 = fmaf(r80, r27, r128);
    r28 = r5 * r14;
    r28 = r28 * r51;
    r28 = fmaf(r4, r129, r129 * r28);
    r28 = fmaf(r129, r66, r28);
    r28 = fmaf(r129, r65, r28);
    r134 = r28 * r80;
    r128 = fmaf(r64, r134, r128);
    r144 = r6 * r38;
    r128 = fmaf(r73, r144, r128);
    r146 = r6 * r40;
    r146 = r146 * r110;
    r146 = r146 * r64;
    r146 = r146 * r104;
    r128 = fmaf(r115, r146, r128);
    r128 = fmaf(r110, r111, r128);
    r128 = fmaf(r24, r136, r128);
    r146 = r0 * r128;
    r144 = r24 * r9;
    r144 = r144 * r9;
    r144 = r144 * r48;
    r144 = r144 * r48;
    r144 = r144 * r126;
    r144 = r144 * r39;
    r134 = r38 * r9;
    r134 = r134 * r48;
    r134 = r134 * r122;
    r134 = fmaf(r61, r134, r106 * r144);
    r144 = r62 * r110;
    r144 = r144 * r92;
    r134 = fmaf(r112, r144, r134);
    r134 = fmaf(r110, r139, r134);
    r134 = r134 + r102;
    r129 = fmaf(r68, r129, r6 * r134);
    r134 = r9 * r48;
    r134 = r134 * r28;
    r129 = fmaf(r80, r134, r129);
    r102 = r25 * r24;
    r102 = r102 * r9;
    r102 = r102 * r79;
    r129 = fmaf(r55, r102, r129);
    r144 = r110 * r81;
    r129 = fmaf(r125, r144, r129);
    r27 = r7 * r8;
    r27 = r27 * r110;
    r27 = r27 * r72;
    r129 = fmaf(r92, r27, r129);
    r93 = r38 * r48;
    r129 = fmaf(r80, r93, r129);
    r13 = r25 * r24;
    r13 = r13 * r9;
    r13 = r13 * r75;
    r13 = r13 * r79;
    r129 = fmaf(r55, r13, r129);
    r12 = r7 * r58;
    r12 = r12 * r72;
    r129 = fmaf(r103, r12, r129);
    r148 = r38 * r48;
    r129 = fmaf(r81, r148, r129);
    r98 = r7 * r24;
    r98 = r98 * r8;
    r98 = r98 * r9;
    r98 = r98 * r48;
    r98 = r98 * r48;
    r98 = r98 * r96;
    r98 = r98 * r39;
    r129 = fmaf(r106, r98, r129);
    r91 = r7 * r40;
    r91 = r91 * r110;
    r91 = r91 * r64;
    r91 = r91 * r104;
    r129 = fmaf(r115, r91, r129);
    r129 = fmaf(r110, r100, r129);
    r129 = fmaf(r110, r135, r129);
    r129 = fmaf(r110, r116, r129);
    r129 = fmaf(r38, r84, r129);
    r91 = r1 * r129;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r88,
                                          r11,
                                          r146,
                                          r91);
    r91 = r37 * r9;
    r91 = r91 * r9;
    r146 = r14 * r43;
    r146 = r146 * r9;
    r146 = fmaf(r36, r146, r95 * r91);
    r91 = r37 * r8;
    r91 = r91 * r8;
    r146 = fmaf(r95, r91, r146);
    r11 = r14 * r57;
    r11 = r11 * r8;
    r146 = fmaf(r36, r11, r146);
    r11 = r146 * r92;
    r91 = r37 * r117;
    r137 = fmaf(r91, r137, r112 * r11);
    r11 = r43 * r72;
    r137 = fmaf(r103, r11, r137);
    r88 = r25 * r146;
    r137 = fmaf(r113, r88, r137);
    r88 = r8 * r8;
    r88 = r88 * r146;
    r88 = r88 * r94;
    r88 = r88 * r34;
    r123 = fmaf(r91, r123, r61 * r88);
    r91 = r25 * r146;
    r123 = fmaf(r119, r91, r123);
    r123 = fmaf(r57, r73, r123);
    r91 = r137 + r123;
    r88 = r8 * r8;
    r88 = r88 * r62;
    r88 = r88 * r146;
    r88 = r88 * r94;
    r88 = r88 * r34;
    r11 = r37 * r8;
    r11 = r11 * r8;
    r11 = r11 * r48;
    r11 = r11 * r48;
    r11 = r11 * r126;
    r11 = r11 * r39;
    r11 = fmaf(r106, r11, r61 * r88);
    r88 = r57 * r122;
    r88 = r88 * r61;
    r11 = fmaf(r64, r88, r11);
    r98 = r127 * r146;
    r11 = fmaf(r119, r98, r11);
    r11 = r11 + r137;
    r11 = fmaf(r7, r11, r67 * r91);
    r137 = r6 * r57;
    r137 = r137 * r72;
    r11 = fmaf(r103, r137, r11);
    r98 = r57 * r48;
    r11 = fmaf(r81, r98, r11);
    r88 = r25 * r37;
    r88 = r88 * r8;
    r88 = r88 * r75;
    r88 = r88 * r79;
    r11 = fmaf(r55, r88, r11);
    r148 = r8 * r48;
    r148 = r148 * r85;
    r148 = r148 * r146;
    r148 = r148 * r39;
    r148 = r148 * r78;
    r11 = fmaf(r34, r148, r11);
    r12 = r5 * r14;
    r12 = r12 * r51;
    r12 = fmaf(r4, r91, r91 * r12);
    r12 = fmaf(r91, r66, r12);
    r12 = fmaf(r91, r65, r12);
    r13 = r12 * r80;
    r11 = fmaf(r64, r13, r11);
    r93 = r6 * r40;
    r93 = r93 * r146;
    r93 = r93 * r64;
    r93 = r93 * r104;
    r11 = fmaf(r115, r93, r11);
    r27 = r57 * r48;
    r11 = fmaf(r80, r27, r11);
    r144 = r8 * r89;
    r144 = r144 * r146;
    r144 = r144 * r94;
    r144 = r144 * r34;
    r11 = fmaf(r80, r144, r11);
    r102 = r6 * r43;
    r11 = fmaf(r73, r102, r11);
    r134 = r6 * r8;
    r134 = r134 * r146;
    r134 = r134 * r72;
    r11 = fmaf(r92, r134, r11);
    r140 = r25 * r37;
    r140 = r140 * r8;
    r140 = r140 * r79;
    r11 = fmaf(r55, r140, r11);
    r87 = r8 * r48;
    r87 = r87 * r75;
    r87 = r87 * r85;
    r87 = r87 * r146;
    r87 = r87 * r39;
    r87 = r87 * r78;
    r11 = fmaf(r34, r87, r11);
    r11 = fmaf(r146, r111, r11);
    r11 = fmaf(r37, r136, r11);
    r87 = r0 * r11;
    r140 = r62 * r146;
    r140 = r140 * r92;
    r134 = r37 * r9;
    r134 = r134 * r9;
    r134 = r134 * r48;
    r134 = r134 * r48;
    r134 = r134 * r126;
    r134 = r134 * r39;
    r134 = fmaf(r106, r134, r112 * r140);
    r140 = r43 * r9;
    r140 = r140 * r48;
    r140 = r140 * r122;
    r134 = fmaf(r61, r140, r134);
    r134 = fmaf(r146, r139, r134);
    r134 = r134 + r123;
    r134 = fmaf(r6, r134, r68 * r91);
    r91 = r25 * r37;
    r91 = r91 * r9;
    r91 = r91 * r79;
    r134 = fmaf(r55, r91, r134);
    r123 = r7 * r57;
    r123 = r123 * r72;
    r134 = fmaf(r103, r123, r134);
    r140 = r146 * r81;
    r134 = fmaf(r125, r140, r134);
    r102 = r7 * r40;
    r102 = r102 * r146;
    r102 = r102 * r64;
    r102 = r102 * r104;
    r134 = fmaf(r115, r102, r134);
    r144 = r9 * r48;
    r144 = r144 * r12;
    r134 = fmaf(r80, r144, r134);
    r27 = r7 * r8;
    r27 = r27 * r146;
    r27 = r27 * r72;
    r134 = fmaf(r92, r27, r134);
    r93 = r43 * r48;
    r134 = fmaf(r80, r93, r134);
    r13 = r25 * r37;
    r13 = r13 * r9;
    r13 = r13 * r75;
    r13 = r13 * r79;
    r134 = fmaf(r55, r13, r134);
    r148 = r43 * r48;
    r134 = fmaf(r81, r148, r134);
    r88 = r7 * r37;
    r88 = r88 * r8;
    r88 = r88 * r9;
    r88 = r88 * r48;
    r88 = r88 * r48;
    r88 = r88 * r96;
    r88 = r88 * r39;
    r134 = fmaf(r106, r88, r134);
    r134 = fmaf(r146, r135, r134);
    r134 = fmaf(r146, r116, r134);
    r134 = fmaf(r43, r84, r134);
    r134 = fmaf(r146, r100, r134);
    r88 = r1 * r134;
    r148 = r42 * r8;
    r148 = r148 * r8;
    r148 = r148 * r48;
    r148 = r148 * r48;
    r148 = r148 * r126;
    r148 = r148 * r39;
    r13 = r14 * r45;
    r13 = r13 * r9;
    r93 = r42 * r9;
    r93 = r93 * r9;
    r93 = fmaf(r95, r93, r36 * r13);
    r13 = r14 * r56;
    r13 = r13 * r8;
    r93 = fmaf(r36, r13, r93);
    r27 = r42 * r8;
    r27 = r27 * r8;
    r93 = fmaf(r95, r27, r93);
    r27 = r127 * r93;
    r27 = fmaf(r119, r27, r106 * r148);
    r148 = r56 * r122;
    r148 = r148 * r61;
    r27 = fmaf(r64, r148, r27);
    r13 = r8 * r8;
    r13 = r13 * r62;
    r13 = r13 * r93;
    r13 = r13 * r94;
    r13 = r13 * r34;
    r27 = fmaf(r61, r13, r27);
    r144 = r93 * r92;
    r102 = r45 * r72;
    r102 = fmaf(r103, r102, r112 * r144);
    r144 = r25 * r93;
    r102 = fmaf(r113, r144, r102);
    r140 = r42 * r9;
    r140 = r140 * r9;
    r102 = fmaf(r117, r140, r102);
    r27 = r27 + r102;
    r13 = r42 * r8;
    r13 = r13 * r8;
    r148 = r25 * r93;
    r148 = fmaf(r119, r148, r117 * r13);
    r13 = r8 * r8;
    r13 = r13 * r93;
    r13 = r13 * r94;
    r13 = r13 * r34;
    r148 = fmaf(r61, r13, r148);
    r148 = fmaf(r56, r73, r148);
    r102 = r102 + r148;
    r27 = fmaf(r67, r102, r7 * r27);
    r13 = r8 * r48;
    r13 = r13 * r75;
    r13 = r13 * r85;
    r13 = r13 * r93;
    r13 = r13 * r39;
    r13 = r13 * r78;
    r27 = fmaf(r34, r13, r27);
    r140 = r6 * r45;
    r27 = fmaf(r73, r140, r27);
    r144 = r5 * r14;
    r144 = r144 * r51;
    r144 = fmaf(r102, r144, r4 * r102);
    r144 = fmaf(r102, r66, r144);
    r144 = fmaf(r102, r65, r144);
    r123 = r144 * r80;
    r27 = fmaf(r64, r123, r27);
    r91 = r56 * r48;
    r27 = fmaf(r80, r91, r27);
    r98 = r25 * r42;
    r98 = r98 * r8;
    r98 = r98 * r79;
    r27 = fmaf(r55, r98, r27);
    r137 = r6 * r8;
    r137 = r137 * r93;
    r137 = r137 * r72;
    r27 = fmaf(r92, r137, r27);
    r145 = r8 * r48;
    r145 = r145 * r85;
    r145 = r145 * r93;
    r145 = r145 * r39;
    r145 = r145 * r78;
    r27 = fmaf(r34, r145, r27);
    r120 = r6 * r40;
    r120 = r120 * r93;
    r120 = r120 * r64;
    r120 = r120 * r104;
    r27 = fmaf(r115, r120, r27);
    r108 = r25 * r42;
    r108 = r108 * r8;
    r108 = r108 * r75;
    r108 = r108 * r79;
    r27 = fmaf(r55, r108, r27);
    r141 = r6 * r56;
    r141 = r141 * r72;
    r27 = fmaf(r103, r141, r27);
    r30 = r56 * r48;
    r27 = fmaf(r81, r30, r27);
    r101 = r8 * r89;
    r101 = r101 * r93;
    r101 = r101 * r94;
    r101 = r101 * r34;
    r27 = fmaf(r80, r101, r27);
    r27 = fmaf(r42, r136, r27);
    r27 = fmaf(r93, r111, r27);
    r101 = r0 * r27;
    r30 = r62 * r93;
    r30 = r30 * r92;
    r141 = r45 * r9;
    r141 = r141 * r48;
    r141 = r141 * r122;
    r141 = fmaf(r61, r141, r112 * r30);
    r30 = r42 * r9;
    r30 = r30 * r9;
    r30 = r30 * r48;
    r30 = r30 * r48;
    r30 = r30 * r126;
    r30 = r30 * r39;
    r141 = fmaf(r106, r30, r141);
    r141 = fmaf(r93, r139, r141);
    r141 = r141 + r148;
    r102 = fmaf(r68, r102, r6 * r141);
    r141 = r7 * r42;
    r141 = r141 * r8;
    r141 = r141 * r9;
    r141 = r141 * r48;
    r141 = r141 * r48;
    r141 = r141 * r96;
    r141 = r141 * r39;
    r102 = fmaf(r106, r141, r102);
    r148 = r93 * r81;
    r102 = fmaf(r125, r148, r102);
    r30 = r7 * r8;
    r30 = r30 * r93;
    r30 = r30 * r72;
    r102 = fmaf(r92, r30, r102);
    r108 = r45 * r48;
    r102 = fmaf(r81, r108, r102);
    r120 = r7 * r40;
    r120 = r120 * r93;
    r120 = r120 * r64;
    r120 = r120 * r104;
    r102 = fmaf(r115, r120, r102);
    r145 = r25 * r42;
    r145 = r145 * r9;
    r145 = r145 * r75;
    r145 = r145 * r79;
    r102 = fmaf(r55, r145, r102);
    r137 = r9 * r48;
    r137 = r137 * r144;
    r102 = fmaf(r80, r137, r102);
    r98 = r45 * r48;
    r102 = fmaf(r80, r98, r102);
    r91 = r25 * r42;
    r91 = r91 * r9;
    r91 = r91 * r79;
    r102 = fmaf(r55, r91, r102);
    r123 = r7 * r56;
    r123 = r123 * r72;
    r102 = fmaf(r103, r123, r102);
    r102 = fmaf(r45, r84, r102);
    r102 = fmaf(r93, r135, r102);
    r102 = fmaf(r93, r116, r102);
    r102 = fmaf(r93, r100, r102);
    r123 = r1 * r102;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r87,
                                          r88,
                                          r101,
                                          r123);
    r123 = r0 * r25;
    r123 = r123 * r2;
    r101 = r25 * r3;
    r88 = r1 * r101;
    r123 = fmaf(r138, r88, r124 * r123);
    r87 = r0 * r25;
    r87 = r87 * r2;
    r87 = fmaf(r147, r88, r31 * r87);
    r91 = r0 * r25;
    r91 = r91 * r2;
    r91 = fmaf(r142, r88, r86 * r91);
    r98 = r0 * r25;
    r98 = r98 * r2;
    r98 = fmaf(r129, r88, r128 * r98);
    WriteSum4<float, float>((float*)inout_shared, r123, r87, r91, r98);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r98 = r0 * r25;
    r98 = r98 * r2;
    r98 = fmaf(r134, r88, r11 * r98);
    r91 = r0 * r25;
    r91 = r91 * r2;
    r91 = fmaf(r102, r88, r27 * r91);
    WriteSum2<float, float>((float*)inout_shared, r98, r91);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r91 = r1 * r1;
    r98 = r138 * r138;
    r87 = r124 * r124;
    r123 = r0 * r0;
    r87 = fmaf(r123, r87, r91 * r98);
    r98 = r31 * r31;
    r137 = r147 * r147;
    r137 = fmaf(r91, r137, r123 * r98);
    r98 = r86 * r86;
    r145 = r142 * r142;
    r145 = fmaf(r91, r145, r123 * r98);
    r98 = r129 * r129;
    r120 = r128 * r128;
    r120 = fmaf(r123, r120, r91 * r98);
    WriteSum4<float, float>((float*)inout_shared, r87, r137, r145, r120);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r120 = r134 * r134;
    r145 = r11 * r11;
    r145 = fmaf(r123, r145, r91 * r120);
    r120 = r102 * r102;
    r137 = r27 * r27;
    r137 = fmaf(r123, r137, r91 * r120);
    WriteSum2<float, float>((float*)inout_shared, r145, r137);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r137 = r124 * r31;
    r145 = r138 * r147;
    r145 = fmaf(r91, r145, r123 * r137);
    r137 = r124 * r86;
    r120 = r138 * r142;
    r120 = fmaf(r91, r120, r123 * r137);
    r137 = r138 * r129;
    r87 = r124 * r128;
    r87 = fmaf(r123, r87, r91 * r137);
    r137 = r124 * r11;
    r98 = r138 * r134;
    r98 = fmaf(r91, r98, r123 * r137);
    WriteSum4<float, float>((float*)inout_shared, r145, r120, r87, r98);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r98 = r124 * r27;
    r87 = r138 * r102;
    r87 = fmaf(r91, r87, r123 * r98);
    r98 = r147 * r142;
    r120 = r31 * r86;
    r120 = fmaf(r123, r120, r91 * r98);
    r98 = r147 * r129;
    r145 = r31 * r128;
    r145 = fmaf(r123, r145, r91 * r98);
    r98 = r147 * r134;
    r137 = r31 * r11;
    r137 = fmaf(r123, r137, r91 * r98);
    WriteSum4<float, float>((float*)inout_shared, r87, r120, r145, r137);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r137 = r147 * r102;
    r145 = r31 * r27;
    r145 = fmaf(r123, r145, r91 * r137);
    r137 = r142 * r129;
    r120 = r86 * r128;
    r120 = fmaf(r123, r120, r91 * r137);
    r137 = r142 * r134;
    r87 = r86 * r11;
    r87 = fmaf(r123, r87, r91 * r137);
    r137 = r86 * r27;
    r98 = r142 * r102;
    r98 = fmaf(r91, r98, r123 * r137);
    WriteSum4<float, float>((float*)inout_shared, r145, r120, r87, r98);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r98 = r128 * r11;
    r87 = r129 * r134;
    r87 = fmaf(r91, r87, r123 * r98);
    r98 = r128 * r27;
    r120 = r129 * r102;
    r120 = fmaf(r91, r120, r123 * r98);
    r98 = r134 * r102;
    r145 = r11 * r27;
    r145 = fmaf(r123, r145, r91 * r98);
    WriteSum3<float, float>((float*)inout_shared, r87, r120, r145);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r145 = r0 * r51;
    r145 = r145 * r80;
    r145 = r145 * r64;
    r120 = r1 * r9;
    r120 = r120 * r48;
    r120 = r120 * r51;
    r120 = r120 * r80;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          0 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r70,
                                          r69,
                                          r145,
                                          r120);
    r87 = r1 * r82;
    r98 = r0 * r80;
    r98 = r98 * r64;
    r98 = r98 * r74;
    r137 = r1 * r80;
    r137 = r137 * r74;
    r137 = r137 * r103;
    r108 = r0 * r9;
    r108 = r108 * r73;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          4 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r98,
                                          r137,
                                          r108,
                                          r87);
    r30 = r0 * r63;
    r148 = r1 * r9;
    r148 = r148 * r73;
    r141 = r0 * r80;
    r141 = r141 * r64;
    r141 = r141 * r77;
    r140 = r1 * r80;
    r140 = r140 * r103;
    r140 = r140 * r77;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          8 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r30,
                                          r148,
                                          r141,
                                          r140);
    r13 = r0 * r51;
    r149 = r1 * r51;
    r150 = r0 * r80;
    r150 = r150 * r64;
    r150 = r150 * r76;
    r151 = r1 * r80;
    r151 = r151 * r103;
    r151 = r151 * r76;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          12 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r150,
                                          r151,
                                          r13,
                                          r149);
    r152 = r25 * r70;
    r152 = r152 * r2;
    r153 = r25 * r2;
    r154 = r69 * r101;
    WriteSum4<float, float>((float*)inout_shared, r152, r154, r153, r101);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r101 = r9 * r48;
    r101 = r101 * r51;
    r101 = r101 * r80;
    r153 = r0 * r25;
    r153 = r153 * r51;
    r153 = r153 * r2;
    r153 = r153 * r80;
    r153 = fmaf(r64, r153, r88 * r101);
    r101 = r80 * r74;
    r101 = r101 * r103;
    r154 = r0 * r25;
    r154 = r154 * r2;
    r154 = r154 * r80;
    r154 = r154 * r64;
    r154 = fmaf(r74, r154, r88 * r101);
    r101 = r0 * r40;
    r101 = r101 * r9;
    r101 = r101 * r2;
    r101 = r101 * r61;
    r101 = fmaf(r64, r101, r82 * r88);
    r152 = r0 * r25;
    r152 = r152 * r63;
    r155 = r1 * r40;
    r155 = r155 * r9;
    r155 = r155 * r3;
    r155 = r155 * r61;
    r155 = fmaf(r64, r155, r2 * r152);
    WriteSum4<float, float>((float*)inout_shared, r153, r154, r101, r155);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           4 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r155 = r0 * r25;
    r155 = r155 * r51;
    r155 = r155 * r2;
    r101 = r51 * r88;
    r154 = r80 * r103;
    r154 = r154 * r77;
    r153 = r0 * r25;
    r153 = r153 * r2;
    r153 = r153 * r80;
    r153 = r153 * r64;
    r153 = fmaf(r77, r153, r88 * r154);
    r154 = r80 * r103;
    r154 = r154 * r76;
    r152 = r0 * r25;
    r152 = r152 * r2;
    r152 = r152 * r80;
    r152 = r152 * r64;
    r152 = fmaf(r76, r152, r88 * r154);
    WriteSum4<float, float>((float*)inout_shared, r153, r152, r155, r101);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           8 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r101 = r70 * r70;
    r155 = r69 * r69;
    WriteSum4<float, float>((float*)inout_shared, r101, r155, r41, r41);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = r8 * r61;
    r41 = r41 * r64;
    r41 = r41 * r74;
    r155 = r74 * r103;
    r155 = r155 * r91;
    r155 = fmaf(r112, r155, r123 * r41);
    r41 = r8 * r61;
    r41 = r41 * r64;
    r41 = r41 * r123;
    r101 = r103 * r91;
    r101 = r101 * r76;
    r101 = fmaf(r112, r101, r76 * r41);
    r41 = r82 * r82;
    r152 = r8 * r8;
    r10 = r35 * r10;
    r10 = 1.0 / r10;
    r59 = r46 * r59;
    r59 = 1.0 / r59;
    r152 = r152 * r9;
    r152 = r152 * r9;
    r152 = r152 * r48;
    r152 = r152 * r48;
    r152 = r152 * r133;
    r152 = r152 * r10;
    r152 = r152 * r59;
    r152 = r152 * r118;
    r41 = fmaf(r123, r152, r91 * r41);
    r59 = r63 * r123;
    r152 = fmaf(r91, r152, r63 * r59);
    WriteSum4<float, float>((float*)inout_shared, r155, r101, r41, r152);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           4 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r152 = r74 * r123;
    r41 = r74 * r91;
    r155 = r77 * r77;
    r10 = r8 * r61;
    r10 = r10 * r64;
    r10 = r10 * r123;
    r133 = r103 * r91;
    r46 = r112 * r133;
    r35 = fmaf(r155, r46, r155 * r10);
    r153 = r76 * r76;
    r153 = fmaf(r46, r153, r10 * r153);
    WriteSum4<float, float>((float*)inout_shared, r35, r153, r152, r41);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           8 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = 0.00000000000000000e+00;
    r152 = r0 * r51;
    r152 = r152 * r70;
    r152 = r152 * r80;
    r152 = r152 * r64;
    WriteSum4<float, float>((float*)inout_shared, r41, r70, r41, r152);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = r0 * r63;
    r63 = r63 * r70;
    r152 = r0 * r70;
    r152 = r152 * r80;
    r152 = r152 * r64;
    r152 = r152 * r74;
    r153 = r9 * r73;
    r153 = r153 * r71;
    r154 = r0 * r70;
    r154 = r154 * r80;
    r154 = r154 * r64;
    r154 = r154 * r77;
    WriteSum4<float, float>((float*)inout_shared, r152, r153, r63, r154);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r154 = r0 * r51;
    r154 = r154 * r70;
    r70 = r80 * r64;
    r70 = r70 * r76;
    r70 = r70 * r71;
    WriteSum4<float, float>((float*)inout_shared, r70, r154, r41, r41);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           8 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r154 = r1 * r82;
    r154 = r154 * r69;
    r70 = r1 * r9;
    r70 = r70 * r48;
    r70 = r70 * r51;
    r70 = r70 * r69;
    r70 = r70 * r80;
    r71 = r1 * r69;
    r71 = r71 * r80;
    r71 = r71 * r74;
    r71 = r71 * r103;
    WriteSum4<float, float>((float*)inout_shared, r69, r70, r71, r154);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           12 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r154 = r1 * r9;
    r154 = r154 * r69;
    r154 = r154 * r73;
    r71 = r1 * r69;
    r71 = r71 * r80;
    r71 = r71 * r103;
    r71 = r71 * r77;
    r70 = r1 * r69;
    r70 = r70 * r80;
    r70 = r70 * r103;
    r70 = r70 * r76;
    WriteSum4<float, float>((float*)inout_shared, r154, r71, r70, r41);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           16 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = r1 * r51;
    r70 = r70 * r69;
    WriteSum4<float, float>((float*)inout_shared, r70, r41, r145, r98);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           20 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float*)inout_shared, r108, r30, r141, r150);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           24 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float*)inout_shared, r13, r41, r120, r137);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           28 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float*)inout_shared, r87, r148, r140, r151);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           32 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r151 = r8 * r61;
    r151 = r151 * r64;
    r151 = r151 * r77;
    r140 = r103 * r77;
    r140 = r140 * r91;
    r140 = fmaf(r112, r140, r123 * r151);
    r151 = r8 * r48;
    r148 = r14 * r8;
    r148 = r148 * r106;
    r148 = r148 * r114;
    r148 = r148 * r118;
    r118 = r9 * r148;
    r151 = r151 * r51;
    r151 = r151 * r123;
    r114 = r9 * r48;
    r114 = r114 * r51;
    r114 = r114 * r82;
    r114 = r114 * r80;
    r114 = fmaf(r91, r114, r118 * r151);
    WriteSum4<float, float>((float*)inout_shared, r41, r149, r140, r114);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           36 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r114 = r80 * r64;
    r114 = r114 * r74;
    r114 = r114 * r123;
    r140 = r80 * r64;
    r140 = r140 * r59;
    r149 = r9 * r48;
    r149 = r149 * r51;
    r149 = r149 * r91;
    r149 = fmaf(r118, r149, r51 * r140);
    r151 = r51 * r76;
    r87 = fmaf(r151, r46, r151 * r10);
    WriteSum4<float, float>((float*)inout_shared, r149, r101, r87, r114);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           40 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r114 = r80 * r74;
    r114 = r114 * r103;
    r114 = r114 * r91;
    r101 = r8 * r74;
    r101 = r101 * r103;
    r101 = r101 * r123;
    r149 = r82 * r80;
    r149 = r149 * r74;
    r149 = r149 * r103;
    r149 = fmaf(r91, r149, r148 * r101);
    r101 = r74 * r133;
    r101 = fmaf(r118, r101, r74 * r140);
    WriteSum4<float, float>((float*)inout_shared, r114, r149, r101, r87);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           44 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r87 = r80 * r64;
    r87 = r87 * r77;
    r87 = r87 * r123;
    r101 = r80 * r103;
    r101 = r101 * r77;
    r101 = r101 * r91;
    r149 = r9 * r73;
    r114 = r9 * r82;
    r114 = r114 * r91;
    r114 = fmaf(r73, r114, r59 * r149);
    WriteSum4<float, float>((float*)inout_shared, r35, r87, r101, r114);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           48 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r114 = r9 * r51;
    r114 = r114 * r123;
    r114 = r114 * r73;
    r101 = r51 * r82;
    r101 = r101 * r91;
    r87 = r8 * r103;
    r87 = r87 * r77;
    r87 = r87 * r123;
    r35 = r82 * r80;
    r35 = r35 * r103;
    r35 = r35 * r77;
    r35 = fmaf(r91, r35, r148 * r87);
    r87 = r8 * r103;
    r87 = r87 * r123;
    r87 = r87 * r76;
    r149 = r82 * r80;
    r149 = r149 * r103;
    r149 = r149 * r91;
    r149 = fmaf(r76, r149, r148 * r87);
    WriteSum4<float, float>((float*)inout_shared, r35, r149, r114, r101);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           52 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = r51 * r59;
    r101 = r9 * r51;
    r101 = r101 * r91;
    r101 = r101 * r73;
    r114 = r77 * r133;
    r114 = fmaf(r118, r114, r77 * r140);
    r149 = r76 * r133;
    r149 = fmaf(r118, r149, r76 * r140);
    WriteSum4<float, float>((float*)inout_shared, r114, r149, r59, r101);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           56 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r101 = r80 * r64;
    r101 = r101 * r123;
    r101 = r101 * r76;
    r59 = r80 * r103;
    r59 = r59 * r91;
    r59 = r59 * r76;
    r149 = r80 * r64;
    r149 = r149 * r123;
    r149 = r149 * r151;
    r155 = r51 * r155;
    r46 = fmaf(r155, r46, r155 * r10);
    WriteSum4<float, float>((float*)inout_shared, r46, r101, r59, r149);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           60 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r151 = r80 * r151;
    r151 = r151 * r133;
    WriteSum2<float, float>((float*)inout_shared, r151, r41);
  };
  FlushSumShared<2, float>(out_calib_precond_tril,
                           64 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = r14 * r32;
    r41 = r41 * r9;
    r151 = r14 * r53;
    r151 = r151 * r8;
    r151 = fmaf(r36, r151, r36 * r41);
    r41 = r60 * r8;
    r41 = r41 * r8;
    r151 = fmaf(r95, r41, r151);
    r149 = r60 * r9;
    r149 = r149 * r9;
    r151 = fmaf(r95, r149, r151);
    r149 = r25 * r151;
    r41 = r151 * r92;
    r41 = fmaf(r112, r41, r113 * r149);
    r149 = r60 * r9;
    r149 = r149 * r9;
    r41 = fmaf(r117, r149, r41);
    r59 = r32 * r72;
    r41 = fmaf(r103, r59, r41);
    r59 = r60 * r8;
    r59 = r59 * r8;
    r59 = fmaf(r117, r59, r53 * r73);
    r149 = r8 * r8;
    r149 = r149 * r151;
    r149 = r149 * r94;
    r149 = r149 * r34;
    r59 = fmaf(r61, r149, r59);
    r101 = r25 * r151;
    r59 = fmaf(r119, r101, r59);
    r101 = r41 + r59;
    r149 = r53 * r122;
    r149 = r149 * r61;
    r46 = r60 * r8;
    r46 = r46 * r8;
    r46 = r46 * r48;
    r46 = r46 * r48;
    r46 = r46 * r126;
    r46 = r46 * r39;
    r46 = fmaf(r106, r46, r64 * r149);
    r149 = r8 * r8;
    r149 = r149 * r62;
    r149 = r149 * r151;
    r149 = r149 * r94;
    r149 = r149 * r34;
    r46 = fmaf(r61, r149, r46);
    r155 = r127 * r151;
    r46 = fmaf(r119, r155, r46);
    r46 = r46 + r41;
    r46 = fmaf(r7, r46, r67 * r101);
    r41 = r8 * r48;
    r41 = r41 * r75;
    r41 = r41 * r85;
    r41 = r41 * r151;
    r41 = r41 * r39;
    r41 = r41 * r78;
    r46 = fmaf(r34, r41, r46);
    r155 = r6 * r53;
    r155 = r155 * r72;
    r46 = fmaf(r103, r155, r46);
    r149 = r25 * r60;
    r149 = r149 * r8;
    r149 = r149 * r79;
    r46 = fmaf(r55, r149, r46);
    r10 = r6 * r40;
    r10 = r10 * r151;
    r10 = r10 * r64;
    r10 = r10 * r104;
    r46 = fmaf(r115, r10, r46);
    r114 = r6 * r8;
    r114 = r114 * r151;
    r114 = r114 * r72;
    r46 = fmaf(r92, r114, r46);
    r140 = r8 * r48;
    r140 = r140 * r85;
    r140 = r140 * r151;
    r140 = r140 * r39;
    r140 = r140 * r78;
    r46 = fmaf(r34, r140, r46);
    r118 = r53 * r48;
    r46 = fmaf(r81, r118, r46);
    r35 = r53 * r48;
    r46 = fmaf(r80, r35, r46);
    r87 = r6 * r32;
    r46 = fmaf(r73, r87, r46);
    r148 = r25 * r60;
    r148 = r148 * r8;
    r148 = r148 * r75;
    r148 = r148 * r79;
    r46 = fmaf(r55, r148, r46);
    r137 = r5 * r14;
    r137 = r137 * r51;
    r137 = fmaf(r101, r137, r4 * r101);
    r137 = fmaf(r101, r66, r137);
    r137 = fmaf(r101, r65, r137);
    r120 = r137 * r80;
    r46 = fmaf(r64, r120, r46);
    r13 = r8 * r89;
    r13 = r13 * r151;
    r13 = r13 * r94;
    r13 = r13 * r34;
    r46 = fmaf(r80, r13, r46);
    r46 = fmaf(r151, r111, r46);
    r46 = fmaf(r60, r136, r46);
    r13 = r0 * r46;
    r120 = r62 * r151;
    r120 = r120 * r92;
    r120 = fmaf(r112, r120, r151 * r139);
    r148 = r60 * r9;
    r148 = r148 * r9;
    r148 = r148 * r48;
    r148 = r148 * r48;
    r148 = r148 * r126;
    r148 = r148 * r39;
    r120 = fmaf(r106, r148, r120);
    r87 = r32 * r9;
    r87 = r87 * r48;
    r87 = r87 * r122;
    r120 = fmaf(r61, r87, r120);
    r120 = r120 + r59;
    r120 = fmaf(r6, r120, r68 * r101);
    r101 = r151 * r81;
    r120 = fmaf(r125, r101, r120);
    r59 = r25 * r60;
    r59 = r59 * r9;
    r59 = r59 * r75;
    r59 = r59 * r79;
    r120 = fmaf(r55, r59, r120);
    r87 = r7 * r53;
    r87 = r87 * r72;
    r120 = fmaf(r103, r87, r120);
    r148 = r7 * r40;
    r148 = r148 * r151;
    r148 = r148 * r64;
    r148 = r148 * r104;
    r120 = fmaf(r115, r148, r120);
    r35 = r7 * r8;
    r35 = r35 * r151;
    r35 = r35 * r72;
    r120 = fmaf(r92, r35, r120);
    r118 = r32 * r48;
    r120 = fmaf(r81, r118, r120);
    r140 = r7 * r60;
    r140 = r140 * r8;
    r140 = r140 * r9;
    r140 = r140 * r48;
    r140 = r140 * r48;
    r140 = r140 * r96;
    r140 = r140 * r39;
    r120 = fmaf(r106, r140, r120);
    r114 = r25 * r60;
    r114 = r114 * r9;
    r114 = r114 * r79;
    r120 = fmaf(r55, r114, r120);
    r10 = r32 * r48;
    r120 = fmaf(r80, r10, r120);
    r149 = r9 * r48;
    r149 = r149 * r137;
    r120 = fmaf(r80, r149, r120);
    r120 = fmaf(r151, r100, r120);
    r120 = fmaf(r151, r135, r120);
    r120 = fmaf(r32, r84, r120);
    r120 = fmaf(r151, r116, r120);
    r149 = r1 * r120;
    r10 = r8 * r8;
    r114 = r44 * r9;
    r114 = r114 * r9;
    r140 = r14 * r50;
    r140 = r140 * r8;
    r140 = fmaf(r36, r140, r95 * r114);
    r114 = r14 * r49;
    r114 = r114 * r9;
    r140 = fmaf(r36, r114, r140);
    r118 = r44 * r8;
    r118 = r118 * r8;
    r140 = fmaf(r95, r118, r140);
    r10 = r10 * r62;
    r10 = r10 * r140;
    r10 = r10 * r94;
    r10 = r10 * r34;
    r118 = r44 * r8;
    r118 = r118 * r8;
    r118 = r118 * r48;
    r118 = r118 * r48;
    r118 = r118 * r126;
    r118 = r118 * r39;
    r118 = fmaf(r106, r118, r61 * r10);
    r10 = r127 * r140;
    r118 = fmaf(r119, r10, r118);
    r114 = r50 * r122;
    r114 = r114 * r61;
    r118 = fmaf(r64, r114, r118);
    r35 = r25 * r140;
    r148 = r49 * r72;
    r148 = fmaf(r103, r148, r113 * r35);
    r35 = r140 * r92;
    r148 = fmaf(r112, r35, r148);
    r87 = r44 * r9;
    r87 = r87 * r9;
    r148 = fmaf(r117, r87, r148);
    r118 = r118 + r148;
    r114 = r8 * r8;
    r114 = r114 * r140;
    r114 = r114 * r94;
    r114 = r114 * r34;
    r10 = r44 * r8;
    r10 = r10 * r8;
    r10 = fmaf(r117, r10, r61 * r114);
    r114 = r25 * r140;
    r10 = fmaf(r119, r114, r10);
    r10 = fmaf(r50, r73, r10);
    r148 = r148 + r10;
    r118 = fmaf(r67, r148, r7 * r118);
    r114 = r6 * r49;
    r118 = fmaf(r73, r114, r118);
    r87 = r8 * r48;
    r87 = r87 * r85;
    r87 = r87 * r140;
    r87 = r87 * r39;
    r87 = r87 * r78;
    r118 = fmaf(r34, r87, r118);
    r35 = r25 * r44;
    r35 = r35 * r8;
    r35 = r35 * r79;
    r118 = fmaf(r55, r35, r118);
    r59 = r25 * r44;
    r59 = r59 * r8;
    r59 = r59 * r75;
    r59 = r59 * r79;
    r118 = fmaf(r55, r59, r118);
    r101 = r50 * r48;
    r118 = fmaf(r81, r101, r118);
    r155 = r6 * r40;
    r155 = r155 * r140;
    r155 = r155 * r64;
    r155 = r155 * r104;
    r118 = fmaf(r115, r155, r118);
    r41 = r8 * r89;
    r41 = r41 * r140;
    r41 = r41 * r94;
    r41 = r41 * r34;
    r118 = fmaf(r80, r41, r118);
    r150 = r50 * r48;
    r118 = fmaf(r80, r150, r118);
    r141 = r8 * r48;
    r141 = r141 * r75;
    r141 = r141 * r85;
    r141 = r141 * r140;
    r141 = r141 * r39;
    r141 = r141 * r78;
    r118 = fmaf(r34, r141, r118);
    r30 = r5 * r14;
    r30 = r30 * r51;
    r30 = fmaf(r4, r148, r148 * r30);
    r30 = fmaf(r148, r66, r30);
    r30 = fmaf(r148, r65, r30);
    r108 = r30 * r80;
    r118 = fmaf(r64, r108, r118);
    r98 = r6 * r50;
    r98 = r98 * r72;
    r118 = fmaf(r103, r98, r118);
    r145 = r6 * r8;
    r145 = r145 * r140;
    r145 = r145 * r72;
    r118 = fmaf(r92, r145, r118);
    r118 = fmaf(r44, r136, r118);
    r118 = fmaf(r140, r111, r118);
    r145 = r0 * r118;
    r98 = r49 * r9;
    r98 = r98 * r48;
    r98 = r98 * r122;
    r98 = fmaf(r61, r98, r140 * r139);
    r108 = r62 * r140;
    r108 = r108 * r92;
    r98 = fmaf(r112, r108, r98);
    r141 = r44 * r9;
    r141 = r141 * r9;
    r141 = r141 * r48;
    r141 = r141 * r48;
    r141 = r141 * r126;
    r141 = r141 * r39;
    r98 = fmaf(r106, r141, r98);
    r98 = r98 + r10;
    r148 = fmaf(r68, r148, r6 * r98);
    r98 = r49 * r48;
    r148 = fmaf(r81, r98, r148);
    r10 = r49 * r48;
    r148 = fmaf(r80, r10, r148);
    r141 = r7 * r40;
    r141 = r141 * r140;
    r141 = r141 * r64;
    r141 = r141 * r104;
    r148 = fmaf(r115, r141, r148);
    r108 = r7 * r44;
    r108 = r108 * r8;
    r108 = r108 * r9;
    r108 = r108 * r48;
    r108 = r108 * r48;
    r108 = r108 * r96;
    r108 = r108 * r39;
    r148 = fmaf(r106, r108, r148);
    r150 = r9 * r48;
    r150 = r150 * r30;
    r148 = fmaf(r80, r150, r148);
    r41 = r140 * r81;
    r148 = fmaf(r125, r41, r148);
    r155 = r25 * r44;
    r155 = r155 * r9;
    r155 = r155 * r75;
    r155 = r155 * r79;
    r148 = fmaf(r55, r155, r148);
    r101 = r25 * r44;
    r101 = r101 * r9;
    r101 = r101 * r79;
    r148 = fmaf(r55, r101, r148);
    r59 = r7 * r50;
    r59 = r59 * r72;
    r148 = fmaf(r103, r59, r148);
    r35 = r7 * r8;
    r35 = r35 * r140;
    r35 = r35 * r72;
    r148 = fmaf(r92, r35, r148);
    r148 = fmaf(r49, r84, r148);
    r148 = fmaf(r140, r100, r148);
    r148 = fmaf(r140, r135, r148);
    r148 = fmaf(r140, r116, r148);
    r35 = r1 * r148;
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r13,
                                          r149,
                                          r145,
                                          r35);
    r35 = r14 * r47;
    r35 = r35 * r9;
    r145 = r52 * r9;
    r145 = r145 * r9;
    r145 = fmaf(r95, r145, r36 * r35);
    r35 = r52 * r8;
    r35 = r35 * r8;
    r145 = fmaf(r95, r35, r145);
    r95 = r14 * r54;
    r95 = r95 * r8;
    r145 = fmaf(r36, r95, r145);
    r95 = r25 * r145;
    r35 = r47 * r72;
    r35 = fmaf(r103, r35, r113 * r95);
    r95 = r145 * r92;
    r35 = fmaf(r112, r95, r35);
    r113 = r52 * r9;
    r113 = r113 * r9;
    r35 = fmaf(r117, r113, r35);
    r113 = r8 * r8;
    r113 = r113 * r145;
    r113 = r113 * r94;
    r113 = r113 * r34;
    r113 = fmaf(r61, r113, r54 * r73);
    r95 = r52 * r8;
    r95 = r95 * r8;
    r113 = fmaf(r117, r95, r113);
    r117 = r25 * r145;
    r113 = fmaf(r119, r117, r113);
    r117 = r35 + r113;
    r95 = r54 * r122;
    r95 = r95 * r61;
    r36 = r8 * r8;
    r36 = r36 * r62;
    r36 = r36 * r145;
    r36 = r36 * r94;
    r36 = r36 * r34;
    r36 = fmaf(r61, r36, r64 * r95);
    r95 = r52 * r8;
    r95 = r95 * r8;
    r95 = r95 * r48;
    r95 = r95 * r48;
    r95 = r95 * r126;
    r95 = r95 * r39;
    r36 = fmaf(r106, r95, r36);
    r149 = r127 * r145;
    r36 = fmaf(r119, r149, r36);
    r36 = r36 + r35;
    r36 = fmaf(r7, r36, r67 * r117);
    r67 = r5 * r14;
    r67 = r67 * r51;
    r4 = fmaf(r4, r117, r117 * r67);
    r4 = fmaf(r117, r66, r4);
    r4 = fmaf(r117, r65, r4);
    r65 = r4 * r80;
    r36 = fmaf(r64, r65, r36);
    r66 = r6 * r40;
    r66 = r66 * r145;
    r66 = r66 * r64;
    r66 = r66 * r104;
    r36 = fmaf(r115, r66, r36);
    r67 = r6 * r8;
    r67 = r67 * r145;
    r67 = r67 * r72;
    r36 = fmaf(r92, r67, r36);
    r51 = r25 * r52;
    r51 = r51 * r8;
    r51 = r51 * r79;
    r36 = fmaf(r55, r51, r36);
    r35 = r8 * r48;
    r35 = r35 * r85;
    r35 = r35 * r145;
    r35 = r35 * r39;
    r35 = r35 * r78;
    r36 = fmaf(r34, r35, r36);
    r149 = r25 * r52;
    r149 = r149 * r8;
    r149 = r149 * r75;
    r149 = r149 * r79;
    r36 = fmaf(r55, r149, r36);
    r95 = r54 * r48;
    r36 = fmaf(r80, r95, r36);
    r119 = r8 * r48;
    r119 = r119 * r75;
    r119 = r119 * r85;
    r119 = r119 * r145;
    r119 = r119 * r39;
    r119 = r119 * r78;
    r36 = fmaf(r34, r119, r36);
    r78 = r6 * r47;
    r36 = fmaf(r73, r78, r36);
    r85 = r54 * r48;
    r36 = fmaf(r81, r85, r36);
    r13 = r8 * r89;
    r13 = r13 * r145;
    r13 = r13 * r94;
    r13 = r13 * r34;
    r36 = fmaf(r80, r13, r36);
    r34 = r6 * r54;
    r34 = r34 * r72;
    r36 = fmaf(r103, r34, r36);
    r36 = fmaf(r52, r136, r36);
    r36 = fmaf(r145, r111, r36);
    r34 = r0 * r36;
    r13 = r47 * r9;
    r13 = r13 * r48;
    r13 = r13 * r122;
    r13 = fmaf(r61, r13, r145 * r139);
    r139 = r62 * r145;
    r139 = r139 * r92;
    r13 = fmaf(r112, r139, r13);
    r112 = r52 * r9;
    r112 = r112 * r9;
    r112 = r112 * r48;
    r112 = r112 * r48;
    r112 = r112 * r126;
    r112 = r112 * r39;
    r13 = fmaf(r106, r112, r13);
    r13 = r13 + r113;
    r13 = fmaf(r6, r13, r68 * r117);
    r117 = r145 * r81;
    r13 = fmaf(r125, r117, r13);
    r125 = r25 * r52;
    r125 = r125 * r9;
    r125 = r125 * r75;
    r125 = r125 * r79;
    r13 = fmaf(r55, r125, r13);
    r75 = r7 * r52;
    r75 = r75 * r8;
    r75 = r75 * r9;
    r75 = r75 * r48;
    r75 = r75 * r48;
    r75 = r75 * r96;
    r75 = r75 * r39;
    r13 = fmaf(r106, r75, r13);
    r106 = r47 * r48;
    r13 = fmaf(r80, r106, r13);
    r39 = r7 * r40;
    r39 = r39 * r145;
    r39 = r39 * r64;
    r39 = r39 * r104;
    r13 = fmaf(r115, r39, r13);
    r115 = r7 * r8;
    r115 = r115 * r145;
    r115 = r115 * r72;
    r13 = fmaf(r92, r115, r13);
    r104 = r9 * r48;
    r104 = r104 * r4;
    r13 = fmaf(r80, r104, r13);
    r64 = r47 * r48;
    r13 = fmaf(r81, r64, r13);
    r96 = r7 * r54;
    r96 = r96 * r72;
    r13 = fmaf(r103, r96, r13);
    r68 = r25 * r52;
    r68 = r68 * r9;
    r68 = r68 * r79;
    r13 = fmaf(r55, r68, r13);
    r13 = fmaf(r145, r100, r13);
    r13 = fmaf(r47, r84, r13);
    r13 = fmaf(r145, r116, r13);
    r13 = fmaf(r145, r135, r13);
    r135 = r1 * r13;
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r34,
                                          r135);
    r135 = r0 * r25;
    r135 = r135 * r2;
    r135 = fmaf(r120, r88, r46 * r135);
    r34 = r0 * r25;
    r34 = r34 * r2;
    r34 = fmaf(r148, r88, r118 * r34);
    r68 = r0 * r25;
    r68 = r68 * r2;
    r88 = fmaf(r13, r88, r36 * r68);
    WriteSum3<float, float>((float*)inout_shared, r135, r34, r88);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r88 = r120 * r120;
    r34 = r46 * r46;
    r34 = fmaf(r123, r34, r91 * r88);
    r88 = r118 * r118;
    r135 = r148 * r148;
    r135 = fmaf(r91, r135, r123 * r88);
    r88 = r13 * r13;
    r68 = r36 * r36;
    r68 = fmaf(r123, r68, r91 * r88);
    WriteSum3<float, float>((float*)inout_shared, r34, r135, r68);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = r120 * r148;
    r135 = r46 * r118;
    r135 = fmaf(r123, r135, r91 * r68);
    r68 = r120 * r13;
    r34 = r46 * r36;
    r34 = fmaf(r123, r34, r91 * r68);
    r68 = r148 * r13;
    r88 = r118 * r36;
    r88 = fmaf(r123, r88, r91 * r68);
    WriteSum3<float, float>((float*)inout_shared, r135, r34, r88);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void ThinPrismFisheyeResJacFirst(float* pose,
                                 unsigned int pose_num_alloc,
                                 SharedIndex* pose_indices,
                                 float* sensor_from_rig,
                                 unsigned int sensor_from_rig_num_alloc,
                                 float* calib,
                                 unsigned int calib_num_alloc,
                                 SharedIndex* calib_indices,
                                 float* point,
                                 unsigned int point_num_alloc,
                                 SharedIndex* point_indices,
                                 float* pixel,
                                 unsigned int pixel_num_alloc,
                                 float* out_res,
                                 unsigned int out_res_num_alloc,
                                 float* const out_rTr,
                                 float* out_pose_jac,
                                 unsigned int out_pose_jac_num_alloc,
                                 float* const out_pose_njtr,
                                 unsigned int out_pose_njtr_num_alloc,
                                 float* const out_pose_precond_diag,
                                 unsigned int out_pose_precond_diag_num_alloc,
                                 float* const out_pose_precond_tril,
                                 unsigned int out_pose_precond_tril_num_alloc,
                                 float* out_calib_jac,
                                 unsigned int out_calib_jac_num_alloc,
                                 float* const out_calib_njtr,
                                 unsigned int out_calib_njtr_num_alloc,
                                 float* const out_calib_precond_diag,
                                 unsigned int out_calib_precond_diag_num_alloc,
                                 float* const out_calib_precond_tril,
                                 unsigned int out_calib_precond_tril_num_alloc,
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
  ThinPrismFisheyeResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      calib,
      calib_num_alloc,
      calib_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      out_calib_jac,
      out_calib_jac_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
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