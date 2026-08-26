#include "kernel_thin_prism_fisheye_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeResJacKernel(float* pose,
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
    r119 = r25 * r105;
    r120 = r64 * r115;
    r121 = r8 * r34;
    r120 = r120 * r121;
    r116 = fmaf(r120, r119, r116);
    r119 = r111 + r116;
    r100 = 6.00000000000000000e+00;
    r122 = r83 * r100;
    r122 = r122 * r61;
    r123 = r8 * r8;
    r123 = r123 * r62;
    r123 = r123 * r105;
    r123 = r123 * r94;
    r123 = r123 * r34;
    r123 = fmaf(r61, r123, r64 * r122);
    r122 = r8 * r8;
    r124 = r48 * r48;
    r125 = -6.00000000000000000e+00;
    r124 = r124 * r101;
    r124 = r124 * r125;
    r124 = r124 * r39;
    r124 = r124 * r106;
    r126 = -3.00000000000000000e+00;
    r127 = r126 * r105;
    r123 = fmaf(r120, r127, r123);
    r123 = fmaf(r124, r122, r123);
    r123 = r123 + r111;
    r123 = fmaf(r7, r123, r67 * r119);
    r111 = r89 * r94;
    r111 = r111 * r81;
    r111 = r111 * r121;
    r121 = r6 * r99;
    r123 = fmaf(r73, r121, r123);
    r127 = r48 * r83;
    r123 = fmaf(r81, r127, r123);
    r128 = r8 * r48;
    r128 = r128 * r85;
    r128 = r128 * r105;
    r128 = r128 * r39;
    r128 = r128 * r78;
    r123 = fmaf(r34, r128, r123);
    r129 = r6 * r8;
    r129 = r129 * r105;
    r129 = r129 * r72;
    r123 = fmaf(r92, r129, r123);
    r130 = r48 * r83;
    r123 = fmaf(r80, r130, r123);
    r131 = r25 * r8;
    r131 = r131 * r101;
    r131 = r131 * r79;
    r123 = fmaf(r55, r131, r123);
    r132 = r5 * r14;
    r132 = r132 * r51;
    r132 = fmaf(r119, r132, r4 * r119);
    r65 = r65 * r62;
    r65 = r65 * r74;
    r133 = 4.00000000000000000e+00;
    r66 = r66 * r133;
    r66 = r66 * r77;
    r132 = fmaf(r119, r65, r132);
    r132 = fmaf(r119, r66, r132);
    r134 = r132 * r80;
    r123 = fmaf(r64, r134, r123);
    r135 = r6 * r83;
    r135 = r135 * r72;
    r123 = fmaf(r103, r135, r123);
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
    r123 = fmaf(r34, r137, r123);
    r138 = r25 * r8;
    r138 = r138 * r75;
    r138 = r138 * r101;
    r138 = r138 * r79;
    r123 = fmaf(r55, r138, r123);
    r139 = r8 * r89;
    r139 = r139 * r105;
    r139 = r139 * r94;
    r139 = r139 * r34;
    r123 = fmaf(r80, r139, r123);
    r140 = r6 * r40;
    r140 = r140 * r105;
    r140 = r140 * r64;
    r140 = r140 * r104;
    r123 = fmaf(r115, r140, r123);
    r123 = fmaf(r105, r111, r123);
    r123 = fmaf(r101, r136, r123);
    r140 = r0 * r123;
    r139 = r9 * r48;
    r139 = r139 * r99;
    r139 = r139 * r100;
    r138 = r62 * r105;
    r138 = r138 * r92;
    r138 = fmaf(r112, r138, r61 * r139);
    r139 = r126 * r113;
    r137 = r9 * r9;
    r138 = fmaf(r105, r139, r138);
    r138 = fmaf(r124, r137, r138);
    r138 = r138 + r116;
    r138 = fmaf(r6, r138, r68 * r119);
    r119 = r48 * r85;
    r119 = r119 * r39;
    r119 = r119 * r78;
    r119 = r119 * r104;
    r116 = r75 * r119;
    r124 = r89 * r92;
    r135 = r80 * r124;
    r134 = r7 * r8;
    r134 = r134 * r105;
    r134 = r134 * r72;
    r138 = fmaf(r92, r134, r138);
    r131 = r105 * r81;
    r138 = fmaf(r124, r131, r138);
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
    r127 = r7 * r8;
    r127 = r127 * r9;
    r127 = r127 * r48;
    r127 = r127 * r48;
    r127 = r127 * r96;
    r127 = r127 * r101;
    r127 = r127 * r39;
    r138 = fmaf(r106, r127, r138);
    r121 = r25 * r9;
    r121 = r121 * r101;
    r121 = r121 * r79;
    r138 = fmaf(r55, r121, r138);
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
    r138 = fmaf(r105, r119, r138);
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
    r121 = r16 * r21;
    r27 = fmaf(r85, r121, r27);
    r30 = fmaf(r27, r30, r91);
    r90 = r90 + r30;
    r121 = r26 * r86;
    r121 = r121 * r96;
    r31 = r29 * r96;
    r31 = r31 * r27;
    r127 = r121 + r31;
    r127 = fmaf(r11, r127, r13 * r90);
    r90 = r40 * r33;
    r90 = fmaf(r40, r97, r27 * r90);
    r128 = r14 * r23;
    r128 = r128 * r86;
    r129 = r14 * r26;
    r129 = fmaf(r28, r129, r128);
    r90 = r90 + r129;
    r127 = fmaf(r12, r90, r127);
    r101 = r101 * r127;
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
    r121 = r121 + r108;
    r131 = fmaf(r13, r121, r131);
    r90 = r90 * r131;
    r90 = fmaf(r95, r90, r36 * r101);
    r101 = r9 * r9;
    r101 = r101 * r131;
    r90 = fmaf(r95, r101, r90);
    r121 = r14 * r9;
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
    r121 = r121 * r97;
    r90 = fmaf(r36, r121, r90);
    r141 = r141 * r90;
    r141 = r141 * r94;
    r141 = r141 * r34;
    r121 = r25 * r90;
    r121 = fmaf(r120, r121, r61 * r141);
    r141 = r131 * r117;
    r121 = fmaf(r141, r122, r121);
    r121 = fmaf(r127, r73, r121);
    r122 = r97 * r72;
    r101 = r25 * r90;
    r101 = fmaf(r113, r101, r103 * r122);
    r122 = r90 * r92;
    r101 = fmaf(r112, r122, r101);
    r101 = fmaf(r141, r137, r101);
    r122 = r121 + r101;
    r137 = r8 * r8;
    r137 = r137 * r62;
    r137 = r137 * r90;
    r137 = r137 * r94;
    r137 = r137 * r34;
    r141 = r126 * r90;
    r141 = fmaf(r120, r141, r61 * r137);
    r137 = r8 * r8;
    r137 = r137 * r48;
    r137 = r137 * r48;
    r137 = r137 * r125;
    r137 = r137 * r131;
    r137 = r137 * r39;
    r141 = fmaf(r106, r137, r141);
    r108 = r100 * r127;
    r108 = r108 * r61;
    r141 = fmaf(r64, r108, r141);
    r141 = r141 + r101;
    r141 = fmaf(r7, r141, r67 * r122);
    r101 = r6 * r127;
    r101 = r101 * r72;
    r141 = fmaf(r103, r101, r141);
    r108 = r8 * r48;
    r108 = r108 * r85;
    r108 = r108 * r90;
    r108 = r108 * r39;
    r108 = r108 * r78;
    r141 = fmaf(r34, r108, r141);
    r137 = r25 * r8;
    r137 = r137 * r131;
    r137 = r137 * r79;
    r141 = fmaf(r55, r137, r141);
    r31 = r8 * r48;
    r31 = r31 * r75;
    r31 = r31 * r85;
    r31 = r31 * r90;
    r31 = r31 * r39;
    r31 = r31 * r78;
    r141 = fmaf(r34, r31, r141);
    r30 = r48 * r127;
    r141 = fmaf(r81, r30, r141);
    r93 = r6 * r8;
    r93 = r93 * r90;
    r93 = r93 * r72;
    r141 = fmaf(r92, r93, r141);
    r27 = r25 * r8;
    r27 = r27 * r75;
    r27 = r27 * r131;
    r27 = r27 * r79;
    r141 = fmaf(r55, r27, r141);
    r109 = r5 * r14;
    r109 = r109 * r51;
    r109 = fmaf(r4, r122, r122 * r109);
    r109 = fmaf(r122, r66, r109);
    r109 = fmaf(r122, r65, r109);
    r134 = r109 * r80;
    r141 = fmaf(r64, r134, r141);
    r143 = r8 * r89;
    r143 = r143 * r90;
    r143 = r143 * r94;
    r143 = r143 * r34;
    r141 = fmaf(r80, r143, r141);
    r144 = r6 * r40;
    r144 = r144 * r90;
    r144 = r144 * r64;
    r144 = r144 * r104;
    r141 = fmaf(r115, r144, r141);
    r145 = r6 * r97;
    r141 = fmaf(r73, r145, r141);
    r146 = r48 * r127;
    r141 = fmaf(r80, r146, r141);
    r141 = fmaf(r131, r136, r141);
    r141 = fmaf(r90, r111, r141);
    r146 = r0 * r141;
    r145 = r9 * r48;
    r145 = r145 * r100;
    r145 = r145 * r97;
    r145 = fmaf(r90, r139, r61 * r145);
    r144 = r9 * r9;
    r144 = r144 * r48;
    r144 = r144 * r48;
    r144 = r144 * r125;
    r144 = r144 * r131;
    r144 = r144 * r39;
    r145 = fmaf(r106, r144, r145);
    r143 = r62 * r90;
    r143 = r143 * r92;
    r145 = fmaf(r112, r143, r145);
    r145 = r145 + r121;
    r145 = fmaf(r6, r145, r68 * r122);
    r122 = r25 * r9;
    r122 = r122 * r131;
    r122 = r122 * r79;
    r145 = fmaf(r55, r122, r145);
    r121 = r48 * r97;
    r145 = fmaf(r81, r121, r145);
    r143 = r7 * r127;
    r143 = r143 * r72;
    r145 = fmaf(r103, r143, r145);
    r144 = r9 * r48;
    r144 = r144 * r109;
    r145 = fmaf(r80, r144, r145);
    r134 = r90 * r81;
    r145 = fmaf(r124, r134, r145);
    r27 = r7 * r8;
    r27 = r27 * r90;
    r27 = r27 * r72;
    r145 = fmaf(r92, r27, r145);
    r93 = r25 * r9;
    r93 = r93 * r75;
    r93 = r93 * r131;
    r93 = r93 * r79;
    r145 = fmaf(r55, r93, r145);
    r30 = r7 * r8;
    r30 = r30 * r9;
    r30 = r30 * r48;
    r30 = r30 * r48;
    r30 = r30 * r96;
    r30 = r30 * r131;
    r30 = r30 * r39;
    r145 = fmaf(r106, r30, r145);
    r131 = r7 * r40;
    r131 = r131 * r90;
    r131 = r131 * r64;
    r131 = r131 * r104;
    r145 = fmaf(r115, r131, r145);
    r31 = r48 * r97;
    r145 = fmaf(r80, r31, r145);
    r145 = fmaf(r90, r116, r145);
    r145 = fmaf(r90, r135, r145);
    r145 = fmaf(r90, r119, r145);
    r145 = fmaf(r97, r84, r145);
    r31 = r1 * r145;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r140,
                                          r142,
                                          r146,
                                          r31);
    r31 = r8 * r8;
    r146 = r23 * r96;
    r142 = r16 * r22;
    r140 = r17 * r19;
    r140 = fmaf(r85, r140, r89 * r142);
    r142 = r18 * r20;
    r140 = fmaf(r89, r142, r140);
    r131 = r15 * r21;
    r140 = fmaf(r89, r131, r140);
    r146 = r146 * r140;
    r107 = r96 * r107;
    r131 = r146 + r107;
    r142 = r14 * r29;
    r142 = r142 * r140;
    r128 = r128 + r142;
    r30 = r40 * r26;
    r128 = fmaf(r28, r30, r128);
    r93 = r40 * r33;
    r128 = fmaf(r87, r93, r128);
    r128 = fmaf(r11, r128, r13 * r131);
    r131 = r14 * r33;
    r131 = fmaf(r14, r130, r140 * r131);
    r131 = r131 + r102;
    r128 = fmaf(r12, r131, r128);
    r31 = r31 * r128;
    r131 = r14 * r8;
    r93 = r14 * r26;
    r93 = r93 * r140;
    r98 = r98 + r93;
    r30 = r29 * r40;
    r98 = fmaf(r28, r30, r98);
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
    r131 = fmaf(r36, r131, r95 * r31);
    r31 = r9 * r9;
    r31 = r31 * r128;
    r131 = fmaf(r95, r31, r131);
    r98 = r14 * r9;
    r93 = r91 + r93;
    r93 = r93 + r110;
    r110 = r40 * r33;
    r130 = fmaf(r40, r130, r140 * r110);
    r130 = r130 + r102;
    r130 = fmaf(r13, r130, r11 * r93);
    r146 = r86 + r146;
    r130 = fmaf(r12, r146, r130);
    r98 = r98 * r130;
    r131 = fmaf(r36, r98, r131);
    r98 = r131 * r92;
    r31 = r130 * r72;
    r31 = fmaf(r103, r31, r112 * r98);
    r98 = r9 * r9;
    r98 = r98 * r128;
    r31 = fmaf(r117, r98, r31);
    r146 = r25 * r131;
    r31 = fmaf(r113, r146, r31);
    r146 = r8 * r8;
    r146 = r146 * r128;
    r98 = r8 * r8;
    r98 = r98 * r131;
    r98 = r98 * r94;
    r98 = r98 * r34;
    r98 = fmaf(r61, r98, r117 * r146);
    r146 = r25 * r131;
    r98 = fmaf(r120, r146, r98);
    r98 = fmaf(r107, r73, r98);
    r146 = r31 + r98;
    r12 = r8 * r8;
    r12 = r12 * r48;
    r12 = r12 * r48;
    r12 = r12 * r125;
    r12 = r12 * r128;
    r12 = r12 * r39;
    r86 = r8 * r8;
    r86 = r86 * r62;
    r86 = r86 * r131;
    r86 = r86 * r94;
    r86 = r86 * r34;
    r86 = fmaf(r61, r86, r106 * r12);
    r12 = r126 * r131;
    r86 = fmaf(r120, r12, r86);
    r13 = r100 * r107;
    r13 = r13 * r61;
    r86 = fmaf(r64, r13, r86);
    r86 = r86 + r31;
    r86 = fmaf(r7, r86, r67 * r146);
    r31 = r6 * r130;
    r86 = fmaf(r73, r31, r86);
    r13 = r48 * r107;
    r86 = fmaf(r81, r13, r86);
    r12 = r8 * r89;
    r12 = r12 * r131;
    r12 = r12 * r94;
    r12 = r12 * r34;
    r86 = fmaf(r80, r12, r86);
    r93 = r5 * r14;
    r93 = r93 * r51;
    r93 = fmaf(r146, r93, r4 * r146);
    r93 = fmaf(r146, r65, r93);
    r93 = fmaf(r146, r66, r93);
    r11 = r93 * r80;
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
    r142 = r142 * r100;
    r142 = r142 * r130;
    r142 = fmaf(r61, r142, r112 * r87);
    r87 = r9 * r9;
    r87 = r87 * r48;
    r87 = r87 * r48;
    r87 = r87 * r125;
    r87 = r87 * r128;
    r87 = r87 * r39;
    r142 = fmaf(r106, r87, r142);
    r142 = fmaf(r131, r139, r142);
    r142 = r142 + r98;
    r142 = fmaf(r6, r142, r68 * r146);
    r146 = r9 * r48;
    r146 = r146 * r93;
    r142 = fmaf(r80, r146, r142);
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
    r142 = fmaf(r124, r140, r142);
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
    r142 = fmaf(r131, r119, r142);
    r142 = fmaf(r131, r135, r142);
    r142 = fmaf(r131, r116, r142);
    r11 = r1 * r142;
    r102 = r24 * r8;
    r102 = r102 * r8;
    r102 = r102 * r48;
    r102 = r102 * r48;
    r102 = r102 * r125;
    r102 = r102 * r39;
    r128 = r58 * r100;
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
    r140 = r126 * r110;
    r128 = fmaf(r120, r140, r128);
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
    r102 = fmaf(r120, r140, r102);
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
    r146 = r6 * r8;
    r146 = r146 * r110;
    r146 = r146 * r72;
    r128 = fmaf(r92, r146, r128);
    r12 = r6 * r58;
    r12 = r12 * r72;
    r128 = fmaf(r103, r12, r128);
    r13 = r58 * r48;
    r128 = fmaf(r81, r13, r128);
    r31 = r8 * r48;
    r31 = r31 * r85;
    r31 = r31 * r110;
    r31 = r31 * r39;
    r31 = r31 * r78;
    r128 = fmaf(r34, r31, r128);
    r30 = r8 * r89;
    r30 = r30 * r110;
    r30 = r30 * r94;
    r30 = r30 * r34;
    r128 = fmaf(r80, r30, r128);
    r28 = r5 * r14;
    r28 = r28 * r51;
    r28 = fmaf(r4, r129, r129 * r28);
    r28 = fmaf(r129, r66, r28);
    r28 = fmaf(r129, r65, r28);
    r27 = r28 * r80;
    r128 = fmaf(r64, r27, r128);
    r134 = r6 * r38;
    r128 = fmaf(r73, r134, r128);
    r144 = r6 * r40;
    r144 = r144 * r110;
    r144 = r144 * r64;
    r144 = r144 * r104;
    r128 = fmaf(r115, r144, r128);
    r128 = fmaf(r110, r111, r128);
    r128 = fmaf(r24, r136, r128);
    r144 = r0 * r128;
    r134 = r24 * r9;
    r134 = r134 * r9;
    r134 = r134 * r48;
    r134 = r134 * r48;
    r134 = r134 * r125;
    r134 = r134 * r39;
    r27 = r38 * r9;
    r27 = r27 * r48;
    r27 = r27 * r100;
    r27 = fmaf(r61, r27, r106 * r134);
    r134 = r62 * r110;
    r134 = r134 * r92;
    r27 = fmaf(r112, r134, r27);
    r27 = fmaf(r110, r139, r27);
    r27 = r27 + r102;
    r129 = fmaf(r68, r129, r6 * r27);
    r27 = r9 * r48;
    r27 = r27 * r28;
    r129 = fmaf(r80, r27, r129);
    r102 = r25 * r24;
    r102 = r102 * r9;
    r102 = r102 * r79;
    r129 = fmaf(r55, r102, r129);
    r134 = r110 * r81;
    r129 = fmaf(r124, r134, r129);
    r30 = r7 * r8;
    r30 = r30 * r110;
    r30 = r30 * r72;
    r129 = fmaf(r92, r30, r129);
    r31 = r38 * r48;
    r129 = fmaf(r80, r31, r129);
    r13 = r25 * r24;
    r13 = r13 * r9;
    r13 = r13 * r75;
    r13 = r13 * r79;
    r129 = fmaf(r55, r13, r129);
    r12 = r7 * r58;
    r12 = r12 * r72;
    r129 = fmaf(r103, r12, r129);
    r146 = r38 * r48;
    r129 = fmaf(r81, r146, r129);
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
    r129 = fmaf(r110, r119, r129);
    r129 = fmaf(r110, r135, r129);
    r129 = fmaf(r110, r116, r129);
    r129 = fmaf(r38, r84, r129);
    r91 = r1 * r129;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r88,
                                          r11,
                                          r144,
                                          r91);
    r91 = r37 * r9;
    r91 = r91 * r9;
    r144 = r14 * r43;
    r144 = r144 * r9;
    r144 = fmaf(r36, r144, r95 * r91);
    r91 = r37 * r8;
    r91 = r91 * r8;
    r144 = fmaf(r95, r91, r144);
    r11 = r14 * r57;
    r11 = r11 * r8;
    r144 = fmaf(r36, r11, r144);
    r11 = r144 * r92;
    r91 = r37 * r9;
    r91 = r91 * r9;
    r91 = fmaf(r117, r91, r112 * r11);
    r11 = r43 * r72;
    r91 = fmaf(r103, r11, r91);
    r88 = r25 * r144;
    r91 = fmaf(r113, r88, r91);
    r88 = r8 * r8;
    r88 = r88 * r144;
    r88 = r88 * r94;
    r88 = r88 * r34;
    r11 = r37 * r8;
    r11 = r11 * r8;
    r11 = fmaf(r117, r11, r61 * r88);
    r88 = r144 * r120;
    r11 = fmaf(r57, r73, r11);
    r11 = fmaf(r25, r88, r11);
    r98 = r91 + r11;
    r146 = r8 * r8;
    r146 = r146 * r62;
    r146 = r146 * r144;
    r146 = r146 * r94;
    r146 = r146 * r34;
    r12 = r37 * r8;
    r12 = r12 * r8;
    r12 = r12 * r48;
    r12 = r12 * r48;
    r12 = r12 * r125;
    r12 = r12 * r39;
    r12 = fmaf(r106, r12, r61 * r146);
    r146 = r57 * r100;
    r146 = r146 * r61;
    r12 = fmaf(r64, r146, r12);
    r12 = fmaf(r126, r88, r12);
    r12 = r12 + r91;
    r12 = fmaf(r7, r12, r67 * r98);
    r91 = r6 * r57;
    r91 = r91 * r72;
    r12 = fmaf(r103, r91, r12);
    r88 = r57 * r48;
    r12 = fmaf(r81, r88, r12);
    r146 = r25 * r37;
    r146 = r146 * r8;
    r146 = r146 * r75;
    r146 = r146 * r79;
    r12 = fmaf(r55, r146, r12);
    r13 = r8 * r48;
    r13 = r13 * r85;
    r13 = r13 * r144;
    r13 = r13 * r39;
    r13 = r13 * r78;
    r12 = fmaf(r34, r13, r12);
    r31 = r5 * r14;
    r31 = r31 * r51;
    r31 = fmaf(r4, r98, r98 * r31);
    r31 = fmaf(r98, r66, r31);
    r31 = fmaf(r98, r65, r31);
    r30 = r31 * r80;
    r12 = fmaf(r64, r30, r12);
    r134 = r6 * r40;
    r134 = r134 * r144;
    r134 = r134 * r64;
    r134 = r134 * r104;
    r12 = fmaf(r115, r134, r12);
    r102 = r57 * r48;
    r12 = fmaf(r80, r102, r12);
    r27 = r8 * r89;
    r27 = r27 * r144;
    r27 = r27 * r94;
    r27 = r27 * r34;
    r12 = fmaf(r80, r27, r12);
    r140 = r6 * r43;
    r12 = fmaf(r73, r140, r12);
    r87 = r6 * r8;
    r87 = r87 * r144;
    r87 = r87 * r72;
    r12 = fmaf(r92, r87, r12);
    r143 = r25 * r37;
    r143 = r143 * r8;
    r143 = r143 * r79;
    r12 = fmaf(r55, r143, r12);
    r121 = r8 * r48;
    r121 = r121 * r75;
    r121 = r121 * r85;
    r121 = r121 * r144;
    r121 = r121 * r39;
    r121 = r121 * r78;
    r12 = fmaf(r34, r121, r12);
    r12 = fmaf(r144, r111, r12);
    r12 = fmaf(r37, r136, r12);
    r121 = r0 * r12;
    r143 = r62 * r144;
    r143 = r143 * r92;
    r87 = r37 * r9;
    r87 = r87 * r9;
    r87 = r87 * r48;
    r87 = r87 * r48;
    r87 = r87 * r125;
    r87 = r87 * r39;
    r87 = fmaf(r106, r87, r112 * r143);
    r143 = r43 * r9;
    r143 = r143 * r48;
    r143 = r143 * r100;
    r87 = fmaf(r61, r143, r87);
    r87 = fmaf(r144, r139, r87);
    r87 = r87 + r11;
    r87 = fmaf(r6, r87, r68 * r98);
    r98 = r25 * r37;
    r98 = r98 * r9;
    r98 = r98 * r79;
    r87 = fmaf(r55, r98, r87);
    r11 = r7 * r57;
    r11 = r11 * r72;
    r87 = fmaf(r103, r11, r87);
    r143 = r144 * r81;
    r87 = fmaf(r124, r143, r87);
    r140 = r7 * r40;
    r140 = r140 * r144;
    r140 = r140 * r64;
    r140 = r140 * r104;
    r87 = fmaf(r115, r140, r87);
    r27 = r9 * r48;
    r27 = r27 * r31;
    r87 = fmaf(r80, r27, r87);
    r102 = r7 * r8;
    r102 = r102 * r144;
    r102 = r102 * r72;
    r87 = fmaf(r92, r102, r87);
    r134 = r43 * r48;
    r87 = fmaf(r80, r134, r87);
    r30 = r25 * r37;
    r30 = r30 * r9;
    r30 = r30 * r75;
    r30 = r30 * r79;
    r87 = fmaf(r55, r30, r87);
    r13 = r43 * r48;
    r87 = fmaf(r81, r13, r87);
    r146 = r7 * r37;
    r146 = r146 * r8;
    r146 = r146 * r9;
    r146 = r146 * r48;
    r146 = r146 * r48;
    r146 = r146 * r96;
    r146 = r146 * r39;
    r87 = fmaf(r106, r146, r87);
    r87 = fmaf(r144, r135, r87);
    r87 = fmaf(r144, r116, r87);
    r87 = fmaf(r43, r84, r87);
    r87 = fmaf(r144, r119, r87);
    r146 = r1 * r87;
    r13 = r42 * r8;
    r13 = r13 * r8;
    r13 = r13 * r48;
    r13 = r13 * r48;
    r13 = r13 * r125;
    r13 = r13 * r39;
    r30 = r14 * r45;
    r30 = r30 * r9;
    r134 = r42 * r9;
    r134 = r134 * r9;
    r134 = fmaf(r95, r134, r36 * r30);
    r30 = r14 * r56;
    r30 = r30 * r8;
    r134 = fmaf(r36, r30, r134);
    r102 = r42 * r8;
    r102 = r102 * r8;
    r134 = fmaf(r95, r102, r134);
    r102 = r126 * r134;
    r102 = fmaf(r120, r102, r106 * r13);
    r13 = r56 * r100;
    r13 = r13 * r61;
    r102 = fmaf(r64, r13, r102);
    r30 = r8 * r8;
    r30 = r30 * r62;
    r30 = r30 * r134;
    r30 = r30 * r94;
    r30 = r30 * r34;
    r102 = fmaf(r61, r30, r102);
    r27 = r134 * r92;
    r140 = r45 * r72;
    r140 = fmaf(r103, r140, r112 * r27);
    r27 = r25 * r134;
    r140 = fmaf(r113, r27, r140);
    r143 = r42 * r9;
    r143 = r143 * r9;
    r140 = fmaf(r117, r143, r140);
    r102 = r102 + r140;
    r30 = r42 * r8;
    r30 = r30 * r8;
    r13 = r25 * r134;
    r13 = fmaf(r120, r13, r117 * r30);
    r30 = r8 * r8;
    r30 = r30 * r134;
    r30 = r30 * r94;
    r30 = r30 * r34;
    r13 = fmaf(r61, r30, r13);
    r13 = fmaf(r56, r73, r13);
    r140 = r140 + r13;
    r102 = fmaf(r67, r140, r7 * r102);
    r30 = r8 * r48;
    r30 = r30 * r75;
    r30 = r30 * r85;
    r30 = r30 * r134;
    r30 = r30 * r39;
    r30 = r30 * r78;
    r102 = fmaf(r34, r30, r102);
    r143 = r6 * r45;
    r102 = fmaf(r73, r143, r102);
    r27 = r5 * r14;
    r27 = r27 * r51;
    r27 = fmaf(r140, r27, r4 * r140);
    r27 = fmaf(r140, r66, r27);
    r27 = fmaf(r140, r65, r27);
    r11 = r27 * r80;
    r102 = fmaf(r64, r11, r102);
    r98 = r56 * r48;
    r102 = fmaf(r80, r98, r102);
    r88 = r25 * r42;
    r88 = r88 * r8;
    r88 = r88 * r79;
    r102 = fmaf(r55, r88, r102);
    r91 = r6 * r8;
    r91 = r91 * r134;
    r91 = r91 * r72;
    r102 = fmaf(r92, r91, r102);
    r122 = r8 * r48;
    r122 = r122 * r85;
    r122 = r122 * r134;
    r122 = r122 * r39;
    r122 = r122 * r78;
    r102 = fmaf(r34, r122, r102);
    r137 = r6 * r40;
    r137 = r137 * r134;
    r137 = r137 * r64;
    r137 = r137 * r104;
    r102 = fmaf(r115, r137, r102);
    r108 = r25 * r42;
    r108 = r108 * r8;
    r108 = r108 * r75;
    r108 = r108 * r79;
    r102 = fmaf(r55, r108, r102);
    r101 = r6 * r56;
    r101 = r101 * r72;
    r102 = fmaf(r103, r101, r102);
    r147 = r56 * r48;
    r102 = fmaf(r81, r147, r102);
    r148 = r8 * r89;
    r148 = r148 * r134;
    r148 = r148 * r94;
    r148 = r148 * r34;
    r102 = fmaf(r80, r148, r102);
    r102 = fmaf(r42, r136, r102);
    r102 = fmaf(r134, r111, r102);
    r148 = r0 * r102;
    r147 = r62 * r134;
    r147 = r147 * r92;
    r101 = r45 * r9;
    r101 = r101 * r48;
    r101 = r101 * r100;
    r101 = fmaf(r61, r101, r112 * r147);
    r147 = r42 * r9;
    r147 = r147 * r9;
    r147 = r147 * r48;
    r147 = r147 * r48;
    r147 = r147 * r125;
    r147 = r147 * r39;
    r101 = fmaf(r106, r147, r101);
    r101 = fmaf(r134, r139, r101);
    r101 = r101 + r13;
    r140 = fmaf(r68, r140, r6 * r101);
    r101 = r7 * r42;
    r101 = r101 * r8;
    r101 = r101 * r9;
    r101 = r101 * r48;
    r101 = r101 * r48;
    r101 = r101 * r96;
    r101 = r101 * r39;
    r140 = fmaf(r106, r101, r140);
    r13 = r134 * r81;
    r140 = fmaf(r124, r13, r140);
    r147 = r7 * r8;
    r147 = r147 * r134;
    r147 = r147 * r72;
    r140 = fmaf(r92, r147, r140);
    r108 = r45 * r48;
    r140 = fmaf(r81, r108, r140);
    r137 = r7 * r40;
    r137 = r137 * r134;
    r137 = r137 * r64;
    r137 = r137 * r104;
    r140 = fmaf(r115, r137, r140);
    r122 = r25 * r42;
    r122 = r122 * r9;
    r122 = r122 * r75;
    r122 = r122 * r79;
    r140 = fmaf(r55, r122, r140);
    r91 = r9 * r48;
    r91 = r91 * r27;
    r140 = fmaf(r80, r91, r140);
    r88 = r45 * r48;
    r140 = fmaf(r80, r88, r140);
    r98 = r25 * r42;
    r98 = r98 * r9;
    r98 = r98 * r79;
    r140 = fmaf(r55, r98, r140);
    r11 = r7 * r56;
    r11 = r11 * r72;
    r140 = fmaf(r103, r11, r140);
    r140 = fmaf(r45, r84, r140);
    r140 = fmaf(r134, r135, r140);
    r140 = fmaf(r134, r116, r140);
    r140 = fmaf(r134, r119, r140);
    r11 = r1 * r140;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r121,
                                          r146,
                                          r148,
                                          r11);
    r11 = r0 * r25;
    r11 = r11 * r2;
    r148 = r25 * r3;
    r146 = r1 * r148;
    r11 = fmaf(r138, r146, r123 * r11);
    r121 = r0 * r25;
    r121 = r121 * r2;
    r121 = fmaf(r145, r146, r141 * r121);
    r98 = r0 * r25;
    r98 = r98 * r2;
    r98 = fmaf(r142, r146, r86 * r98);
    r88 = r0 * r25;
    r88 = r88 * r2;
    r88 = fmaf(r129, r146, r128 * r88);
    WriteSum4<float, float>((float*)inout_shared, r11, r121, r98, r88);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r88 = r0 * r25;
    r88 = r88 * r2;
    r88 = fmaf(r87, r146, r12 * r88);
    r98 = r0 * r25;
    r98 = r98 * r2;
    r98 = fmaf(r140, r146, r102 * r98);
    WriteSum2<float, float>((float*)inout_shared, r88, r98);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r98 = r1 * r1;
    r88 = r138 * r138;
    r121 = r123 * r123;
    r11 = r0 * r0;
    r121 = fmaf(r11, r121, r98 * r88);
    r88 = r141 * r141;
    r91 = r145 * r145;
    r91 = fmaf(r98, r91, r11 * r88);
    r88 = r86 * r86;
    r122 = r142 * r142;
    r122 = fmaf(r98, r122, r11 * r88);
    r88 = r129 * r129;
    r137 = r128 * r128;
    r137 = fmaf(r11, r137, r98 * r88);
    WriteSum4<float, float>((float*)inout_shared, r121, r91, r122, r137);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r137 = r87 * r87;
    r122 = r12 * r12;
    r122 = fmaf(r11, r122, r98 * r137);
    r137 = r140 * r140;
    r91 = r102 * r102;
    r91 = fmaf(r11, r91, r98 * r137);
    WriteSum2<float, float>((float*)inout_shared, r122, r91);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r91 = r123 * r141;
    r122 = r138 * r145;
    r122 = fmaf(r98, r122, r11 * r91);
    r91 = r123 * r86;
    r137 = r138 * r142;
    r137 = fmaf(r98, r137, r11 * r91);
    r91 = r138 * r129;
    r121 = r123 * r128;
    r121 = fmaf(r11, r121, r98 * r91);
    r91 = r123 * r12;
    r88 = r138 * r87;
    r88 = fmaf(r98, r88, r11 * r91);
    WriteSum4<float, float>((float*)inout_shared, r122, r137, r121, r88);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r88 = r123 * r102;
    r121 = r138 * r140;
    r121 = fmaf(r98, r121, r11 * r88);
    r88 = r145 * r142;
    r137 = r141 * r86;
    r137 = fmaf(r11, r137, r98 * r88);
    r88 = r145 * r129;
    r122 = r141 * r128;
    r122 = fmaf(r11, r122, r98 * r88);
    r88 = r145 * r87;
    r91 = r141 * r12;
    r91 = fmaf(r11, r91, r98 * r88);
    WriteSum4<float, float>((float*)inout_shared, r121, r137, r122, r91);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r91 = r145 * r140;
    r122 = r141 * r102;
    r122 = fmaf(r11, r122, r98 * r91);
    r91 = r142 * r129;
    r137 = r86 * r128;
    r137 = fmaf(r11, r137, r98 * r91);
    r91 = r142 * r87;
    r121 = r86 * r12;
    r121 = fmaf(r11, r121, r98 * r91);
    r91 = r86 * r102;
    r88 = r142 * r140;
    r88 = fmaf(r98, r88, r11 * r91);
    WriteSum4<float, float>((float*)inout_shared, r122, r137, r121, r88);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r88 = r128 * r12;
    r121 = r129 * r87;
    r121 = fmaf(r98, r121, r11 * r88);
    r88 = r128 * r102;
    r137 = r129 * r140;
    r137 = fmaf(r98, r137, r11 * r88);
    r88 = r87 * r140;
    r122 = r12 * r102;
    r122 = fmaf(r11, r122, r98 * r88);
    WriteSum3<float, float>((float*)inout_shared, r121, r137, r122);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r122 = r0 * r51;
    r122 = r122 * r80;
    r122 = r122 * r64;
    r137 = r1 * r9;
    r137 = r137 * r48;
    r137 = r137 * r51;
    r137 = r137 * r80;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          0 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r70,
                                          r69,
                                          r122,
                                          r137);
    r121 = r1 * r82;
    r88 = r0 * r80;
    r88 = r88 * r64;
    r88 = r88 * r74;
    r91 = r1 * r80;
    r91 = r91 * r74;
    r91 = r91 * r103;
    r108 = r0 * r9;
    r108 = r108 * r73;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          4 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r88,
                                          r91,
                                          r108,
                                          r121);
    r147 = r0 * r63;
    r13 = r1 * r9;
    r13 = r13 * r73;
    r101 = r0 * r80;
    r101 = r101 * r64;
    r101 = r101 * r77;
    r143 = r1 * r80;
    r143 = r143 * r103;
    r143 = r143 * r77;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          8 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r147,
                                          r13,
                                          r101,
                                          r143);
    r30 = r0 * r51;
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
                                          r30,
                                          r149);
    r152 = r25 * r70;
    r152 = r152 * r2;
    r153 = r25 * r2;
    r154 = r69 * r148;
    WriteSum4<float, float>((float*)inout_shared, r152, r154, r153, r148);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r148 = r9 * r48;
    r148 = r148 * r51;
    r148 = r148 * r80;
    r153 = r0 * r25;
    r153 = r153 * r51;
    r153 = r153 * r2;
    r153 = r153 * r80;
    r153 = fmaf(r64, r153, r146 * r148);
    r148 = r80 * r74;
    r148 = r148 * r103;
    r154 = r0 * r25;
    r154 = r154 * r2;
    r154 = r154 * r80;
    r154 = r154 * r64;
    r154 = fmaf(r74, r154, r146 * r148);
    r148 = r0 * r40;
    r148 = r148 * r9;
    r148 = r148 * r2;
    r148 = r148 * r61;
    r148 = fmaf(r64, r148, r82 * r146);
    r152 = r0 * r25;
    r152 = r152 * r63;
    r155 = r1 * r40;
    r155 = r155 * r9;
    r155 = r155 * r3;
    r155 = r155 * r61;
    r155 = fmaf(r64, r155, r2 * r152);
    WriteSum4<float, float>((float*)inout_shared, r153, r154, r148, r155);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           4 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r155 = r0 * r25;
    r155 = r155 * r51;
    r155 = r155 * r2;
    r148 = r51 * r146;
    r154 = r80 * r103;
    r154 = r154 * r77;
    r153 = r0 * r25;
    r153 = r153 * r2;
    r153 = r153 * r80;
    r153 = r153 * r64;
    r153 = fmaf(r77, r153, r146 * r154);
    r154 = r80 * r103;
    r154 = r154 * r76;
    r152 = r0 * r25;
    r152 = r152 * r2;
    r152 = r152 * r80;
    r152 = r152 * r64;
    r152 = fmaf(r76, r152, r146 * r154);
    WriteSum4<float, float>((float*)inout_shared, r153, r152, r155, r148);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           8 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r148 = r70 * r70;
    r155 = r69 * r69;
    WriteSum4<float, float>((float*)inout_shared, r148, r155, r41, r41);
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
    r155 = r155 * r98;
    r155 = fmaf(r112, r155, r11 * r41);
    r41 = r8 * r61;
    r41 = r41 * r64;
    r41 = r41 * r11;
    r148 = r103 * r98;
    r148 = r148 * r76;
    r148 = fmaf(r112, r148, r76 * r41);
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
    r41 = fmaf(r11, r152, r98 * r41);
    r59 = r63 * r11;
    r152 = fmaf(r98, r152, r63 * r59);
    WriteSum4<float, float>((float*)inout_shared, r155, r148, r41, r152);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           4 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r152 = r74 * r11;
    r41 = r74 * r98;
    r155 = r77 * r77;
    r10 = r8 * r61;
    r10 = r10 * r64;
    r10 = r10 * r11;
    r133 = r103 * r98;
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
    WriteSum4<float, float>((float*)inout_shared, r70, r41, r122, r88);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           20 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float*)inout_shared, r108, r147, r101, r150);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           24 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float*)inout_shared, r30, r41, r137, r91);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           28 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float*)inout_shared, r121, r13, r143, r151);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           32 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r151 = r8 * r61;
    r151 = r151 * r64;
    r151 = r151 * r77;
    r143 = r103 * r77;
    r143 = r143 * r98;
    r143 = fmaf(r112, r143, r11 * r151);
    r151 = r8 * r48;
    r13 = r14 * r8;
    r13 = r13 * r106;
    r13 = r13 * r114;
    r13 = r13 * r118;
    r118 = r9 * r13;
    r151 = r151 * r51;
    r151 = r151 * r11;
    r114 = r9 * r48;
    r114 = r114 * r51;
    r114 = r114 * r82;
    r114 = r114 * r80;
    r114 = fmaf(r98, r114, r118 * r151);
    WriteSum4<float, float>((float*)inout_shared, r41, r149, r143, r114);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           36 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r114 = r80 * r64;
    r114 = r114 * r74;
    r114 = r114 * r11;
    r143 = r80 * r64;
    r143 = r143 * r59;
    r149 = r9 * r48;
    r149 = r149 * r51;
    r149 = r149 * r98;
    r149 = fmaf(r118, r149, r51 * r143);
    r151 = r51 * r76;
    r121 = fmaf(r151, r46, r151 * r10);
    WriteSum4<float, float>((float*)inout_shared, r149, r148, r121, r114);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           40 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r114 = r80 * r74;
    r114 = r114 * r103;
    r114 = r114 * r98;
    r148 = r8 * r74;
    r148 = r148 * r103;
    r148 = r148 * r11;
    r149 = r82 * r80;
    r149 = r149 * r74;
    r149 = r149 * r103;
    r149 = fmaf(r98, r149, r13 * r148);
    r148 = r74 * r133;
    r148 = fmaf(r118, r148, r74 * r143);
    WriteSum4<float, float>((float*)inout_shared, r114, r149, r148, r121);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           44 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r121 = r80 * r64;
    r121 = r121 * r77;
    r121 = r121 * r11;
    r148 = r80 * r103;
    r148 = r148 * r77;
    r148 = r148 * r98;
    r149 = r9 * r73;
    r114 = r9 * r82;
    r114 = r114 * r98;
    r114 = fmaf(r73, r114, r59 * r149);
    WriteSum4<float, float>((float*)inout_shared, r35, r121, r148, r114);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           48 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r114 = r9 * r51;
    r114 = r114 * r11;
    r114 = r114 * r73;
    r148 = r51 * r82;
    r148 = r148 * r98;
    r121 = r8 * r103;
    r121 = r121 * r77;
    r121 = r121 * r11;
    r35 = r82 * r80;
    r35 = r35 * r103;
    r35 = r35 * r77;
    r35 = fmaf(r98, r35, r13 * r121);
    r121 = r8 * r103;
    r121 = r121 * r11;
    r121 = r121 * r76;
    r149 = r82 * r80;
    r149 = r149 * r103;
    r149 = r149 * r98;
    r149 = fmaf(r76, r149, r13 * r121);
    WriteSum4<float, float>((float*)inout_shared, r35, r149, r114, r148);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           52 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = r51 * r59;
    r148 = r9 * r51;
    r148 = r148 * r98;
    r148 = r148 * r73;
    r114 = r77 * r133;
    r114 = fmaf(r118, r114, r77 * r143);
    r149 = r76 * r133;
    r149 = fmaf(r118, r149, r76 * r143);
    WriteSum4<float, float>((float*)inout_shared, r114, r149, r59, r148);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           56 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r148 = r80 * r64;
    r148 = r148 * r11;
    r148 = r148 * r76;
    r59 = r80 * r103;
    r59 = r59 * r98;
    r59 = r59 * r76;
    r149 = r80 * r64;
    r149 = r149 * r11;
    r149 = r149 * r151;
    r155 = r51 * r155;
    r46 = fmaf(r155, r46, r155 * r10);
    WriteSum4<float, float>((float*)inout_shared, r46, r148, r59, r149);
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
    r148 = r25 * r151;
    r59 = fmaf(r120, r148, r59);
    r148 = r41 + r59;
    r149 = r53 * r100;
    r149 = r149 * r61;
    r46 = r60 * r8;
    r46 = r46 * r8;
    r46 = r46 * r48;
    r46 = r46 * r48;
    r46 = r46 * r125;
    r46 = r46 * r39;
    r46 = fmaf(r106, r46, r64 * r149);
    r149 = r8 * r8;
    r149 = r149 * r62;
    r149 = r149 * r151;
    r149 = r149 * r94;
    r149 = r149 * r34;
    r46 = fmaf(r61, r149, r46);
    r155 = r126 * r151;
    r46 = fmaf(r120, r155, r46);
    r46 = r46 + r41;
    r46 = fmaf(r7, r46, r67 * r148);
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
    r143 = r8 * r48;
    r143 = r143 * r85;
    r143 = r143 * r151;
    r143 = r143 * r39;
    r143 = r143 * r78;
    r46 = fmaf(r34, r143, r46);
    r118 = r53 * r48;
    r46 = fmaf(r81, r118, r46);
    r35 = r53 * r48;
    r46 = fmaf(r80, r35, r46);
    r121 = r6 * r32;
    r46 = fmaf(r73, r121, r46);
    r13 = r25 * r60;
    r13 = r13 * r8;
    r13 = r13 * r75;
    r13 = r13 * r79;
    r46 = fmaf(r55, r13, r46);
    r91 = r5 * r14;
    r91 = r91 * r51;
    r91 = fmaf(r148, r91, r4 * r148);
    r91 = fmaf(r148, r66, r91);
    r91 = fmaf(r148, r65, r91);
    r137 = r91 * r80;
    r46 = fmaf(r64, r137, r46);
    r30 = r8 * r89;
    r30 = r30 * r151;
    r30 = r30 * r94;
    r30 = r30 * r34;
    r46 = fmaf(r80, r30, r46);
    r46 = fmaf(r151, r111, r46);
    r46 = fmaf(r60, r136, r46);
    r30 = r0 * r46;
    r137 = r62 * r151;
    r137 = r137 * r92;
    r137 = fmaf(r112, r137, r151 * r139);
    r13 = r60 * r9;
    r13 = r13 * r9;
    r13 = r13 * r48;
    r13 = r13 * r48;
    r13 = r13 * r125;
    r13 = r13 * r39;
    r137 = fmaf(r106, r13, r137);
    r121 = r32 * r9;
    r121 = r121 * r48;
    r121 = r121 * r100;
    r137 = fmaf(r61, r121, r137);
    r137 = r137 + r59;
    r137 = fmaf(r6, r137, r68 * r148);
    r148 = r151 * r81;
    r137 = fmaf(r124, r148, r137);
    r59 = r25 * r60;
    r59 = r59 * r9;
    r59 = r59 * r75;
    r59 = r59 * r79;
    r137 = fmaf(r55, r59, r137);
    r121 = r7 * r53;
    r121 = r121 * r72;
    r137 = fmaf(r103, r121, r137);
    r13 = r7 * r40;
    r13 = r13 * r151;
    r13 = r13 * r64;
    r13 = r13 * r104;
    r137 = fmaf(r115, r13, r137);
    r35 = r7 * r8;
    r35 = r35 * r151;
    r35 = r35 * r72;
    r137 = fmaf(r92, r35, r137);
    r118 = r32 * r48;
    r137 = fmaf(r81, r118, r137);
    r143 = r7 * r60;
    r143 = r143 * r8;
    r143 = r143 * r9;
    r143 = r143 * r48;
    r143 = r143 * r48;
    r143 = r143 * r96;
    r143 = r143 * r39;
    r137 = fmaf(r106, r143, r137);
    r114 = r25 * r60;
    r114 = r114 * r9;
    r114 = r114 * r79;
    r137 = fmaf(r55, r114, r137);
    r10 = r32 * r48;
    r137 = fmaf(r80, r10, r137);
    r149 = r9 * r48;
    r149 = r149 * r91;
    r137 = fmaf(r80, r149, r137);
    r137 = fmaf(r151, r119, r137);
    r137 = fmaf(r151, r135, r137);
    r137 = fmaf(r32, r84, r137);
    r137 = fmaf(r151, r116, r137);
    r149 = r1 * r137;
    r10 = r8 * r8;
    r114 = r44 * r9;
    r114 = r114 * r9;
    r143 = r14 * r50;
    r143 = r143 * r8;
    r143 = fmaf(r36, r143, r95 * r114);
    r114 = r14 * r49;
    r114 = r114 * r9;
    r143 = fmaf(r36, r114, r143);
    r118 = r44 * r8;
    r118 = r118 * r8;
    r143 = fmaf(r95, r118, r143);
    r10 = r10 * r62;
    r10 = r10 * r143;
    r10 = r10 * r94;
    r10 = r10 * r34;
    r118 = r44 * r8;
    r118 = r118 * r8;
    r118 = r118 * r48;
    r118 = r118 * r48;
    r118 = r118 * r125;
    r118 = r118 * r39;
    r118 = fmaf(r106, r118, r61 * r10);
    r10 = r126 * r143;
    r118 = fmaf(r120, r10, r118);
    r114 = r50 * r100;
    r114 = r114 * r61;
    r118 = fmaf(r64, r114, r118);
    r35 = r25 * r143;
    r13 = r49 * r72;
    r13 = fmaf(r103, r13, r113 * r35);
    r35 = r143 * r92;
    r13 = fmaf(r112, r35, r13);
    r121 = r44 * r9;
    r121 = r121 * r9;
    r13 = fmaf(r117, r121, r13);
    r118 = r118 + r13;
    r114 = r8 * r8;
    r114 = r114 * r143;
    r114 = r114 * r94;
    r114 = r114 * r34;
    r10 = r44 * r8;
    r10 = r10 * r8;
    r10 = fmaf(r117, r10, r61 * r114);
    r114 = r25 * r143;
    r10 = fmaf(r120, r114, r10);
    r10 = fmaf(r50, r73, r10);
    r13 = r13 + r10;
    r118 = fmaf(r67, r13, r7 * r118);
    r114 = r6 * r49;
    r118 = fmaf(r73, r114, r118);
    r121 = r8 * r48;
    r121 = r121 * r85;
    r121 = r121 * r143;
    r121 = r121 * r39;
    r121 = r121 * r78;
    r118 = fmaf(r34, r121, r118);
    r35 = r25 * r44;
    r35 = r35 * r8;
    r35 = r35 * r79;
    r118 = fmaf(r55, r35, r118);
    r59 = r25 * r44;
    r59 = r59 * r8;
    r59 = r59 * r75;
    r59 = r59 * r79;
    r118 = fmaf(r55, r59, r118);
    r148 = r50 * r48;
    r118 = fmaf(r81, r148, r118);
    r155 = r6 * r40;
    r155 = r155 * r143;
    r155 = r155 * r64;
    r155 = r155 * r104;
    r118 = fmaf(r115, r155, r118);
    r41 = r8 * r89;
    r41 = r41 * r143;
    r41 = r41 * r94;
    r41 = r41 * r34;
    r118 = fmaf(r80, r41, r118);
    r150 = r50 * r48;
    r118 = fmaf(r80, r150, r118);
    r101 = r8 * r48;
    r101 = r101 * r75;
    r101 = r101 * r85;
    r101 = r101 * r143;
    r101 = r101 * r39;
    r101 = r101 * r78;
    r118 = fmaf(r34, r101, r118);
    r147 = r5 * r14;
    r147 = r147 * r51;
    r147 = fmaf(r4, r13, r13 * r147);
    r147 = fmaf(r13, r66, r147);
    r147 = fmaf(r13, r65, r147);
    r108 = r147 * r80;
    r118 = fmaf(r64, r108, r118);
    r88 = r6 * r50;
    r88 = r88 * r72;
    r118 = fmaf(r103, r88, r118);
    r122 = r6 * r8;
    r122 = r122 * r143;
    r122 = r122 * r72;
    r118 = fmaf(r92, r122, r118);
    r118 = fmaf(r44, r136, r118);
    r118 = fmaf(r143, r111, r118);
    r122 = r0 * r118;
    r88 = r49 * r9;
    r88 = r88 * r48;
    r88 = r88 * r100;
    r88 = fmaf(r61, r88, r143 * r139);
    r108 = r62 * r143;
    r108 = r108 * r92;
    r88 = fmaf(r112, r108, r88);
    r101 = r44 * r9;
    r101 = r101 * r9;
    r101 = r101 * r48;
    r101 = r101 * r48;
    r101 = r101 * r125;
    r101 = r101 * r39;
    r88 = fmaf(r106, r101, r88);
    r88 = r88 + r10;
    r13 = fmaf(r68, r13, r6 * r88);
    r88 = r49 * r48;
    r13 = fmaf(r81, r88, r13);
    r10 = r49 * r48;
    r13 = fmaf(r80, r10, r13);
    r101 = r7 * r40;
    r101 = r101 * r143;
    r101 = r101 * r64;
    r101 = r101 * r104;
    r13 = fmaf(r115, r101, r13);
    r108 = r7 * r44;
    r108 = r108 * r8;
    r108 = r108 * r9;
    r108 = r108 * r48;
    r108 = r108 * r48;
    r108 = r108 * r96;
    r108 = r108 * r39;
    r13 = fmaf(r106, r108, r13);
    r150 = r9 * r48;
    r150 = r150 * r147;
    r13 = fmaf(r80, r150, r13);
    r41 = r143 * r81;
    r13 = fmaf(r124, r41, r13);
    r155 = r25 * r44;
    r155 = r155 * r9;
    r155 = r155 * r75;
    r155 = r155 * r79;
    r13 = fmaf(r55, r155, r13);
    r148 = r25 * r44;
    r148 = r148 * r9;
    r148 = r148 * r79;
    r13 = fmaf(r55, r148, r13);
    r59 = r7 * r50;
    r59 = r59 * r72;
    r13 = fmaf(r103, r59, r13);
    r35 = r7 * r8;
    r35 = r35 * r143;
    r35 = r35 * r72;
    r13 = fmaf(r92, r35, r13);
    r13 = fmaf(r49, r84, r13);
    r13 = fmaf(r143, r119, r13);
    r13 = fmaf(r143, r135, r13);
    r13 = fmaf(r143, r116, r13);
    r35 = r1 * r13;
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r30,
                                          r149,
                                          r122,
                                          r35);
    r35 = r14 * r47;
    r35 = r35 * r9;
    r122 = r52 * r9;
    r122 = r122 * r9;
    r122 = fmaf(r95, r122, r36 * r35);
    r35 = r52 * r8;
    r35 = r35 * r8;
    r122 = fmaf(r95, r35, r122);
    r95 = r14 * r54;
    r95 = r95 * r8;
    r122 = fmaf(r36, r95, r122);
    r95 = r25 * r122;
    r35 = r47 * r72;
    r35 = fmaf(r103, r35, r113 * r95);
    r95 = r122 * r92;
    r35 = fmaf(r112, r95, r35);
    r113 = r52 * r9;
    r113 = r113 * r9;
    r35 = fmaf(r117, r113, r35);
    r113 = r8 * r8;
    r113 = r113 * r122;
    r113 = r113 * r94;
    r113 = r113 * r34;
    r113 = fmaf(r61, r113, r54 * r73);
    r95 = r52 * r8;
    r95 = r95 * r8;
    r113 = fmaf(r117, r95, r113);
    r117 = r25 * r122;
    r113 = fmaf(r120, r117, r113);
    r117 = r35 + r113;
    r95 = r54 * r100;
    r95 = r95 * r61;
    r36 = r8 * r8;
    r36 = r36 * r62;
    r36 = r36 * r122;
    r36 = r36 * r94;
    r36 = r36 * r34;
    r36 = fmaf(r61, r36, r64 * r95);
    r95 = r52 * r8;
    r95 = r95 * r8;
    r95 = r95 * r48;
    r95 = r95 * r48;
    r95 = r95 * r125;
    r95 = r95 * r39;
    r36 = fmaf(r106, r95, r36);
    r149 = r126 * r122;
    r36 = fmaf(r120, r149, r36);
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
    r66 = r66 * r122;
    r66 = r66 * r64;
    r66 = r66 * r104;
    r36 = fmaf(r115, r66, r36);
    r67 = r6 * r8;
    r67 = r67 * r122;
    r67 = r67 * r72;
    r36 = fmaf(r92, r67, r36);
    r51 = r25 * r52;
    r51 = r51 * r8;
    r51 = r51 * r79;
    r36 = fmaf(r55, r51, r36);
    r35 = r8 * r48;
    r35 = r35 * r85;
    r35 = r35 * r122;
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
    r120 = r8 * r48;
    r120 = r120 * r75;
    r120 = r120 * r85;
    r120 = r120 * r122;
    r120 = r120 * r39;
    r120 = r120 * r78;
    r36 = fmaf(r34, r120, r36);
    r78 = r6 * r47;
    r36 = fmaf(r73, r78, r36);
    r85 = r54 * r48;
    r36 = fmaf(r81, r85, r36);
    r30 = r8 * r89;
    r30 = r30 * r122;
    r30 = r30 * r94;
    r30 = r30 * r34;
    r36 = fmaf(r80, r30, r36);
    r34 = r6 * r54;
    r34 = r34 * r72;
    r36 = fmaf(r103, r34, r36);
    r36 = fmaf(r52, r136, r36);
    r36 = fmaf(r122, r111, r36);
    r34 = r0 * r36;
    r30 = r47 * r9;
    r30 = r30 * r48;
    r30 = r30 * r100;
    r30 = fmaf(r61, r30, r122 * r139);
    r139 = r62 * r122;
    r139 = r139 * r92;
    r30 = fmaf(r112, r139, r30);
    r112 = r52 * r9;
    r112 = r112 * r9;
    r112 = r112 * r48;
    r112 = r112 * r48;
    r112 = r112 * r125;
    r112 = r112 * r39;
    r30 = fmaf(r106, r112, r30);
    r30 = r30 + r113;
    r30 = fmaf(r6, r30, r68 * r117);
    r117 = r122 * r81;
    r30 = fmaf(r124, r117, r30);
    r124 = r25 * r52;
    r124 = r124 * r9;
    r124 = r124 * r75;
    r124 = r124 * r79;
    r30 = fmaf(r55, r124, r30);
    r75 = r7 * r52;
    r75 = r75 * r8;
    r75 = r75 * r9;
    r75 = r75 * r48;
    r75 = r75 * r48;
    r75 = r75 * r96;
    r75 = r75 * r39;
    r30 = fmaf(r106, r75, r30);
    r106 = r47 * r48;
    r30 = fmaf(r80, r106, r30);
    r39 = r7 * r40;
    r39 = r39 * r122;
    r39 = r39 * r64;
    r39 = r39 * r104;
    r30 = fmaf(r115, r39, r30);
    r115 = r7 * r8;
    r115 = r115 * r122;
    r115 = r115 * r72;
    r30 = fmaf(r92, r115, r30);
    r104 = r9 * r48;
    r104 = r104 * r4;
    r30 = fmaf(r80, r104, r30);
    r64 = r47 * r48;
    r30 = fmaf(r81, r64, r30);
    r96 = r7 * r54;
    r96 = r96 * r72;
    r30 = fmaf(r103, r96, r30);
    r68 = r25 * r52;
    r68 = r68 * r9;
    r68 = r68 * r79;
    r30 = fmaf(r55, r68, r30);
    r30 = fmaf(r122, r119, r30);
    r30 = fmaf(r47, r84, r30);
    r30 = fmaf(r122, r116, r30);
    r30 = fmaf(r122, r135, r30);
    r135 = r1 * r30;
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r34,
                                          r135);
    r135 = r0 * r25;
    r135 = r135 * r2;
    r135 = fmaf(r137, r146, r46 * r135);
    r34 = r0 * r25;
    r34 = r34 * r2;
    r34 = fmaf(r13, r146, r118 * r34);
    r68 = r0 * r25;
    r68 = r68 * r2;
    r146 = fmaf(r30, r146, r36 * r68);
    WriteSum3<float, float>((float*)inout_shared, r135, r34, r146);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r146 = r137 * r137;
    r34 = r46 * r46;
    r34 = fmaf(r11, r34, r98 * r146);
    r146 = r118 * r118;
    r135 = r13 * r13;
    r135 = fmaf(r98, r135, r11 * r146);
    r146 = r30 * r30;
    r68 = r36 * r36;
    r68 = fmaf(r11, r68, r98 * r146);
    WriteSum3<float, float>((float*)inout_shared, r34, r135, r68);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = r137 * r13;
    r135 = r46 * r118;
    r135 = fmaf(r11, r135, r98 * r68);
    r68 = r137 * r30;
    r34 = r46 * r36;
    r34 = fmaf(r11, r34, r98 * r68);
    r68 = r13 * r30;
    r146 = r118 * r36;
    r146 = fmaf(r11, r146, r98 * r68);
    WriteSum3<float, float>((float*)inout_shared, r135, r34, r146);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void ThinPrismFisheyeResJac(float* pose,
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
  ThinPrismFisheyeResJacKernel<<<n_blocks, 1024>>>(
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