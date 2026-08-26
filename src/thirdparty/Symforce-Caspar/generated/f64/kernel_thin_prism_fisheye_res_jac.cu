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
    ThinPrismFisheyeResJacKernel(double* pose,
                                 unsigned int pose_num_alloc,
                                 SharedIndex* pose_indices,
                                 double* sensor_from_rig,
                                 unsigned int sensor_from_rig_num_alloc,
                                 double* calib,
                                 unsigned int calib_num_alloc,
                                 SharedIndex* calib_indices,
                                 double* point,
                                 unsigned int point_num_alloc,
                                 SharedIndex* point_indices,
                                 double* pixel,
                                 unsigned int pixel_num_alloc,
                                 double* out_res,
                                 unsigned int out_res_num_alloc,
                                 double* out_pose_jac,
                                 unsigned int out_pose_jac_num_alloc,
                                 double* const out_pose_njtr,
                                 unsigned int out_pose_njtr_num_alloc,
                                 double* const out_pose_precond_diag,
                                 unsigned int out_pose_precond_diag_num_alloc,
                                 double* const out_pose_precond_tril,
                                 unsigned int out_pose_precond_tril_num_alloc,
                                 double* out_calib_jac,
                                 unsigned int out_calib_jac_num_alloc,
                                 double* const out_calib_njtr,
                                 unsigned int out_calib_njtr_num_alloc,
                                 double* const out_calib_precond_diag,
                                 unsigned int out_calib_precond_diag_num_alloc,
                                 double* const out_calib_precond_tril,
                                 unsigned int out_calib_precond_tril_num_alloc,
                                 double* out_point_jac,
                                 unsigned int out_point_jac_num_alloc,
                                 double* const out_point_njtr,
                                 unsigned int out_point_njtr_num_alloc,
                                 double* const out_point_precond_diag,
                                 unsigned int out_point_precond_diag_num_alloc,
                                 double* const out_point_precond_tril,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
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
      r153, r154, r155, r156;
  LoadShared<2, double, double>(
      calib, 2 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r2, r3);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      calib, 6 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r4, r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r6,
                                            r7);
  };
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r10, r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r12,
                                            r13);
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r14, r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r16,
                                            r17);
    r18 = fma(r15, r16, r10 * r13);
    r19 = r14 * r17;
    r20 = -1.00000000000000000e+00;
    r18 = fma(r20, r19, r18);
    r18 = fma(r11, r12, r18);
    r19 = r18 * r18;
    r21 = -2.00000000000000000e+00;
    r19 = r19 * r21;
    r22 = 1.00000000000000000e+00;
    r23 = r15 * r13;
    r24 = r11 * r17;
    r25 = r23 + r24;
    r26 = r14 * r12;
    r27 = r10 * r16;
    r25 = r25 + r26;
    r25 = fma(r20, r27, r25);
    r28 = r21 * r25;
    r28 = fma(r25, r28, r22);
    r29 = r19 + r28;
    r6 = fma(r8, r29, r6);
    r30 = 2.00000000000000000e+00;
    r31 = fma(r11, r16, r14 * r13);
    r32 = r15 * r12;
    r31 = fma(r20, r32, r31);
    r31 = fma(r10, r17, r31);
    r32 = r30 * r31;
    r32 = r32 * r25;
    r33 = r18 * r21;
    r34 = fma(r15, r17, r14 * r16);
    r34 = fma(r10, r12, r34);
    r34 = fma(r20, r34, r11 * r13);
    r33 = fma(r34, r33, r32);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r35);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r36 = r30 * r18;
    r36 = r36 * r31;
    r37 = r30 * r25;
    r37 = fma(r34, r37, r36);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r38);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r39 = r16 * r12;
    r39 = r39 * r30;
    r40 = r17 * r13;
    r40 = fma(r30, r40, r39);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r41, r42);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r43 = r12 * r13;
    r44 = r16 * r17;
    r44 = r44 * r30;
    r43 = fma(r21, r43, r44);
    r45 = r17 * r17;
    r45 = r45 * r21;
    r46 = r22 + r45;
    r47 = r12 * r12;
    r47 = r47 * r21;
    r46 = r46 + r47;
    r6 = fma(r9, r33, r6);
    r6 = fma(r35, r37, r6);
    r6 = fma(r38, r40, r6);
    r6 = fma(r42, r43, r6);
    r6 = fma(r41, r46, r6);
    r48 = r6 * r6;
    r49 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r50);
    r51 = r21 * r25;
    r51 = fma(r34, r51, r36);
    r50 = fma(r8, r51, r50);
    r36 = r17 * r13;
    r36 = fma(r21, r36, r39);
    r45 = r22 + r45;
    r39 = r16 * r16;
    r39 = r39 * r21;
    r45 = r45 + r39;
    r52 = r17 * r12;
    r52 = r52 * r30;
    r53 = r16 * r13;
    r53 = fma(r30, r53, r52);
    r54 = r30 * r18;
    r54 = r54 * r25;
    r55 = r30 * r31;
    r55 = fma(r34, r55, r54);
    r56 = r31 * r31;
    r56 = r56 * r21;
    r28 = r56 + r28;
    r50 = fma(r41, r36, r50);
    r50 = fma(r38, r45, r50);
    r50 = fma(r42, r53, r50);
    r50 = fma(r9, r55, r50);
    r50 = fma(r35, r28, r50);
    r57 = copysign(1.0, r50);
    r57 = fma(r49, r57, r50);
    r50 = r57 * r57;
    r58 = 1.0 / r50;
    r59 = r30 * r18;
    r59 = fma(r34, r59, r32);
    r7 = fma(r8, r59, r7);
    r32 = r12 * r13;
    r32 = fma(r30, r32, r44);
    r47 = r22 + r47;
    r47 = r47 + r39;
    r39 = r16 * r13;
    r39 = fma(r21, r39, r52);
    r52 = r31 * r21;
    r52 = fma(r34, r52, r54);
    r19 = r22 + r19;
    r19 = r19 + r56;
    r7 = fma(r41, r32, r7);
    r7 = fma(r42, r47, r7);
    r7 = fma(r38, r39, r7);
    r7 = fma(r35, r52, r7);
    r7 = fma(r9, r19, r7);
    r38 = r7 * r7;
    r38 = fma(r58, r38, r58 * r48);
    r48 = sqrt(r38);
    r42 = atan(r48);
    r41 = r42 * r58;
    r56 = r42 * r41;
    r54 = copysign(1.0, r48);
    r54 = fma(r49, r54, r48);
    r49 = r54 * r54;
    r48 = 1.0 / r49;
    r44 = r7 * r48;
    r60 = r7 * r44;
    r61 = r56 * r60;
    r62 = r6 * r6;
    r63 = 3.00000000000000000e+00;
    r62 = r62 * r63;
    r62 = r62 * r48;
    r62 = fma(r56, r62, r61);
  };
  LoadShared<2, double, double>(
      calib, 10 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r64, r65);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r66 = r6 * r6;
    r66 = r66 * r48;
    r66 = r66 * r56;
    r61 = r61 + r66;
    r67 = fma(r64, r61, r5 * r62);
  };
  LoadShared<2, double, double>(
      calib, 4 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r68, r69);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r70 = r61 * r61;
    r71 = fma(r69, r70, r68 * r61);
  };
  LoadShared<2, double, double>(
      calib, 8 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r72, r73);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r74 = r70 * r70;
    r75 = r61 * r70;
    r71 = fma(r73, r74, r71);
    r71 = fma(r72, r75, r71);
    r76 = 1.0 / r57;
    r77 = 1.0 / r54;
    r78 = r76 * r77;
    r79 = r42 * r78;
    r80 = r71 * r79;
    r81 = r4 * r6;
    r82 = r30 * r56;
    r81 = r81 * r44;
    r67 = fma(r82, r81, r67);
    r67 = fma(r6, r80, r67);
    r67 = fma(r6, r79, r67);
    r0 = fma(r2, r67, r0);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r81, r83);
    r0 = fma(r81, r20, r0);
    r81 = r63 * r56;
    r81 = fma(r60, r81, r66);
    r66 = fma(r65, r61, r4 * r81);
    r84 = r7 * r79;
    r85 = r5 * r6;
    r85 = r85 * r44;
    r66 = fma(r82, r85, r66);
    r66 = r66 + r84;
    r66 = fma(r7, r80, r66);
    r1 = fma(r3, r66, r1);
    r1 = fma(r83, r20, r1);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r0, r1);
    r83 = r30 * r6;
    r85 = r14 * r13;
    r86 = -5.00000000000000000e-01;
    r87 = r11 * r16;
    r87 = fma(r86, r87, r86 * r85);
    r85 = r10 * r17;
    r87 = fma(r86, r85, r87);
    r88 = r15 * r12;
    r89 = 5.00000000000000000e-01;
    r87 = fma(r89, r88, r87);
    r88 = r25 * r87;
    r85 = r10 * r13;
    r90 = r15 * r16;
    r90 = fma(r89, r90, r89 * r85);
    r85 = r14 * r17;
    r90 = fma(r86, r85, r90);
    r91 = r11 * r12;
    r90 = fma(r89, r91, r90);
    r91 = r34 * r90;
    r85 = fma(r30, r91, r30 * r88);
    r92 = r30 * r31;
    r93 = fma(r89, r27, r86 * r23);
    r93 = fma(r86, r24, r93);
    r93 = fma(r86, r26, r93);
    r94 = r30 * r18;
    r95 = r11 * r13;
    r96 = r14 * r16;
    r96 = fma(r86, r96, r89 * r95);
    r95 = r15 * r17;
    r96 = fma(r86, r95, r96);
    r97 = r10 * r12;
    r96 = fma(r86, r97, r96);
    r94 = r94 * r96;
    r92 = fma(r93, r92, r94);
    r85 = r85 + r92;
    r97 = r30 * r25;
    r97 = r97 * r96;
    r95 = r30 * r31;
    r95 = r95 * r90;
    r98 = r97 + r95;
    r99 = r18 * r21;
    r98 = fma(r87, r99, r98);
    r100 = r21 * r34;
    r98 = fma(r93, r100, r98);
    r98 = fma(r9, r98, r35 * r85);
    r85 = r25 * r90;
    r100 = -4.00000000000000000e+00;
    r85 = r85 * r100;
    r99 = r18 * r93;
    r101 = r100 * r99;
    r102 = r85 + r101;
    r98 = fma(r8, r102, r98);
    r83 = r83 * r98;
    r102 = r30 * r25;
    r102 = r102 * r93;
    r103 = r30 * r18;
    r103 = fma(r90, r103, r102);
    r90 = r30 * r31;
    r90 = r90 * r87;
    r104 = r30 * r34;
    r104 = r104 * r96;
    r105 = r90 + r104;
    r106 = r103 + r105;
    r91 = fma(r21, r91, r21 * r88);
    r91 = r91 + r92;
    r91 = fma(r8, r91, r9 * r106);
    r106 = r31 * r96;
    r106 = r106 * r100;
    r85 = r85 + r106;
    r91 = fma(r35, r85, r91);
    r50 = r57 * r50;
    r85 = 1.0 / r50;
    r107 = r21 * r85;
    r108 = r91 * r107;
    r109 = r6 * r6;
    r108 = fma(r109, r108, r58 * r83);
    r83 = r7 * r7;
    r83 = r83 * r91;
    r108 = fma(r107, r83, r108);
    r110 = r30 * r7;
    r111 = r31 * r21;
    r112 = r21 * r34;
    r112 = r112 * r96;
    r111 = fma(r87, r111, r112);
    r111 = r111 + r103;
    r101 = r106 + r101;
    r101 = fma(r9, r101, r35 * r111);
    r111 = r30 * r34;
    r111 = fma(r93, r111, r95);
    r95 = r30 * r18;
    r95 = fma(r87, r95, r97);
    r111 = r111 + r95;
    r101 = fma(r8, r111, r101);
    r110 = r110 * r101;
    r108 = fma(r58, r110, r108);
    r110 = r63 * r108;
    r83 = rsqrt(r38);
    r111 = r6 * r83;
    r38 = r22 + r38;
    r38 = 1.0 / r38;
    r97 = r38 * r41;
    r106 = r6 * r48;
    r110 = r110 * r111;
    r110 = r110 * r97;
    r103 = -3.00000000000000000e+00;
    r113 = r6 * r103;
    r49 = r54 * r49;
    r114 = 1.0 / r49;
    r115 = r114 * r56;
    r115 = r115 * r111;
    r113 = r113 * r108;
    r113 = fma(r115, r113, r106 * r110);
    r110 = r6 * r98;
    r116 = 6.00000000000000000e+00;
    r110 = r110 * r116;
    r110 = r110 * r48;
    r113 = fma(r56, r110, r113);
    r117 = r6 * r6;
    r118 = -6.00000000000000000e+00;
    r119 = r91 * r118;
    r119 = r119 * r85;
    r117 = r117 * r42;
    r117 = r117 * r42;
    r117 = r117 * r48;
    r113 = fma(r119, r117, r113);
    r120 = r91 * r60;
    r121 = r42 * r42;
    r122 = r107 * r121;
    r123 = r83 * r97;
    r123 = r123 * r60;
    r120 = fma(r108, r123, r122 * r120);
    r124 = r20 * r7;
    r124 = r124 * r7;
    r124 = r124 * r108;
    r124 = r124 * r83;
    r124 = r124 * r114;
    r120 = fma(r56, r124, r120);
    r125 = r101 * r44;
    r120 = fma(r82, r125, r120);
    r113 = r113 + r120;
    r117 = r108 * r111;
    r117 = r117 * r97;
    r110 = r20 * r6;
    r110 = r110 * r108;
    r110 = fma(r115, r110, r106 * r117);
    r117 = r82 * r106;
    r125 = r48 * r109;
    r125 = r125 * r122;
    r110 = fma(r98, r117, r110);
    r110 = fma(r91, r125, r110);
    r120 = r120 + r110;
    r113 = fma(r64, r120, r5 * r113);
    r124 = r42 * r86;
    r124 = r124 * r48;
    r124 = r124 * r76;
    r124 = r124 * r111;
    r126 = r71 * r124;
    r127 = r4 * r98;
    r127 = r127 * r44;
    r113 = fma(r82, r127, r113);
    r128 = r4 * r108;
    r129 = r30 * r44;
    r129 = r129 * r111;
    r129 = r129 * r97;
    r113 = fma(r129, r128, r113);
    r130 = r4 * r108;
    r131 = r21 * r7;
    r131 = r131 * r115;
    r113 = fma(r131, r130, r113);
    r132 = r89 * r108;
    r132 = r132 * r38;
    r132 = r132 * r78;
    r133 = r4 * r101;
    r113 = fma(r117, r133, r113);
    r134 = r69 * r30;
    r134 = r134 * r61;
    r134 = fma(r120, r134, r68 * r120);
    r135 = 4.00000000000000000e+00;
    r73 = r73 * r135;
    r73 = r73 * r75;
    r72 = r72 * r63;
    r72 = r72 * r70;
    r134 = fma(r120, r73, r134);
    r134 = fma(r120, r72, r134);
    r136 = r6 * r134;
    r113 = fma(r79, r136, r113);
    r137 = r4 * r6;
    r137 = r137 * r42;
    r137 = r137 * r42;
    r137 = r137 * r100;
    r137 = r137 * r85;
    r137 = r137 * r44;
    r138 = r20 * r6;
    r138 = r138 * r91;
    r138 = r138 * r77;
    r113 = fma(r41, r138, r113);
    r139 = r71 * r111;
    r113 = fma(r132, r139, r113);
    r140 = r20 * r6;
    r140 = r140 * r71;
    r140 = r140 * r91;
    r140 = r140 * r77;
    r113 = fma(r41, r140, r113);
    r113 = fma(r108, r126, r113);
    r113 = fma(r108, r124, r113);
    r113 = fma(r111, r132, r113);
    r113 = fma(r91, r137, r113);
    r113 = fma(r98, r79, r113);
    r113 = fma(r98, r80, r113);
    r140 = r2 * r113;
    r139 = r60 * r121;
    r138 = r63 * r108;
    r138 = fma(r123, r138, r119 * r139);
    r139 = r7 * r7;
    r139 = r139 * r103;
    r139 = r139 * r108;
    r139 = r139 * r83;
    r139 = r139 * r114;
    r138 = fma(r56, r139, r138);
    r119 = r101 * r116;
    r119 = r119 * r56;
    r138 = fma(r44, r119, r138);
    r138 = r138 + r110;
    r110 = r7 * r89;
    r110 = r110 * r76;
    r110 = r110 * r77;
    r110 = r110 * r83;
    r110 = r110 * r108;
    r110 = r110 * r38;
    r138 = fma(r4, r138, r110);
    r119 = r7 * r134;
    r138 = fma(r79, r119, r138);
    r139 = r20 * r7;
    r139 = r139 * r91;
    r139 = r139 * r77;
    r138 = fma(r41, r139, r138);
    r136 = r5 * r98;
    r136 = r136 * r44;
    r138 = fma(r82, r136, r138);
    r133 = r5 * r108;
    r138 = fma(r129, r133, r138);
    r132 = r5 * r108;
    r138 = fma(r131, r132, r138);
    r130 = r42 * r86;
    r130 = r130 * r108;
    r130 = r130 * r76;
    r130 = r130 * r83;
    r138 = fma(r44, r130, r138);
    r128 = r5 * r117;
    r127 = r5 * r6;
    r127 = r127 * r42;
    r127 = r127 * r42;
    r127 = r127 * r100;
    r127 = r127 * r91;
    r127 = r127 * r85;
    r138 = fma(r44, r127, r138);
    r141 = r42 * r71;
    r141 = r141 * r86;
    r141 = r141 * r108;
    r141 = r141 * r76;
    r141 = r141 * r83;
    r138 = fma(r44, r141, r138);
    r142 = r20 * r7;
    r142 = r142 * r71;
    r142 = r142 * r91;
    r142 = r142 * r77;
    r138 = fma(r41, r142, r138);
    r138 = fma(r65, r120, r138);
    r138 = fma(r101, r80, r138);
    r138 = fma(r71, r110, r138);
    r138 = fma(r101, r128, r138);
    r138 = fma(r101, r79, r138);
    r142 = r3 * r138;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             0 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r140,
                                             r142);
    r142 = r21 * r25;
    r142 = fma(r93, r142, r112);
    r140 = r30 * r18;
    r141 = r10 * r13;
    r127 = r15 * r16;
    r127 = fma(r86, r127, r86 * r141);
    r141 = r14 * r17;
    r127 = fma(r89, r141, r127);
    r110 = r11 * r12;
    r127 = fma(r86, r110, r127);
    r140 = r140 * r127;
    r110 = r30 * r31;
    r141 = r14 * r13;
    r130 = r11 * r16;
    r130 = fma(r89, r130, r89 * r141);
    r141 = r10 * r17;
    r130 = fma(r89, r141, r130);
    r132 = r15 * r12;
    r130 = fma(r86, r132, r130);
    r110 = fma(r130, r110, r140);
    r142 = r142 + r110;
    r132 = r30 * r25;
    r132 = r132 * r130;
    r141 = r30 * r34;
    r141 = fma(r127, r141, r132);
    r141 = r141 + r92;
    r141 = fma(r9, r141, r8 * r142);
    r142 = r25 * r96;
    r142 = r142 * r100;
    r92 = r31 * r127;
    r133 = r100 * r92;
    r136 = r142 + r133;
    r141 = fma(r35, r136, r141);
    r104 = r102 + r104;
    r104 = r104 + r110;
    r110 = r18 * r100;
    r110 = r110 * r130;
    r142 = r142 + r110;
    r142 = fma(r8, r142, r35 * r104);
    r104 = r21 * r34;
    r104 = fma(r21, r99, r130 * r104);
    r102 = r30 * r31;
    r102 = r102 * r96;
    r136 = r30 * r25;
    r136 = fma(r127, r136, r102);
    r104 = r104 + r136;
    r142 = fma(r9, r104, r142);
    r104 = fma(r142, r117, r141 * r125);
    r139 = r20 * r6;
    r119 = r30 * r6;
    r119 = r119 * r142;
    r120 = r30 * r7;
    r132 = r94 + r132;
    r94 = r31 * r21;
    r132 = fma(r93, r94, r132);
    r93 = r21 * r34;
    r132 = fma(r127, r93, r132);
    r93 = r30 * r34;
    r99 = fma(r30, r99, r130 * r93);
    r99 = r99 + r136;
    r99 = fma(r8, r99, r35 * r132);
    r133 = r110 + r133;
    r99 = fma(r9, r133, r99);
    r120 = r120 * r99;
    r120 = fma(r58, r120, r58 * r119);
    r119 = r7 * r7;
    r119 = r119 * r141;
    r120 = fma(r107, r119, r120);
    r133 = r141 * r107;
    r120 = fma(r109, r133, r120);
    r139 = r139 * r120;
    r104 = fma(r115, r139, r104);
    r133 = r120 * r111;
    r133 = r133 * r97;
    r104 = fma(r106, r133, r104);
    r133 = r99 * r44;
    r139 = r20 * r7;
    r139 = r139 * r7;
    r139 = r139 * r120;
    r139 = r139 * r83;
    r139 = r139 * r114;
    r139 = fma(r56, r139, r82 * r133);
    r133 = r141 * r60;
    r139 = fma(r122, r133, r139);
    r139 = fma(r120, r123, r139);
    r133 = r104 + r139;
    r119 = r6 * r6;
    r119 = r119 * r42;
    r119 = r119 * r42;
    r119 = r119 * r118;
    r119 = r119 * r141;
    r119 = r119 * r48;
    r110 = r6 * r116;
    r110 = r110 * r142;
    r110 = r110 * r48;
    r110 = fma(r56, r110, r85 * r119);
    r119 = r6 * r115;
    r132 = r103 * r120;
    r110 = fma(r132, r119, r110);
    r93 = r63 * r120;
    r93 = r93 * r111;
    r93 = r93 * r97;
    r110 = fma(r106, r93, r110);
    r110 = r110 + r139;
    r110 = fma(r5, r110, r64 * r133);
    r139 = r20 * r6;
    r139 = r139 * r141;
    r139 = r139 * r77;
    r110 = fma(r41, r139, r110);
    r93 = r89 * r120;
    r93 = r93 * r38;
    r93 = r93 * r78;
    r110 = fma(r111, r93, r110);
    r119 = r69 * r30;
    r119 = r119 * r61;
    r119 = fma(r133, r119, r68 * r133);
    r119 = fma(r133, r73, r119);
    r119 = fma(r133, r72, r119);
    r130 = r6 * r119;
    r110 = fma(r79, r130, r110);
    r94 = r4 * r99;
    r110 = fma(r117, r94, r110);
    r143 = r4 * r120;
    r110 = fma(r129, r143, r110);
    r144 = r4 * r142;
    r144 = r144 * r44;
    r110 = fma(r82, r144, r110);
    r145 = r4 * r120;
    r110 = fma(r131, r145, r110);
    r146 = r20 * r6;
    r146 = r146 * r71;
    r146 = r146 * r141;
    r146 = r146 * r77;
    r110 = fma(r41, r146, r110);
    r147 = r89 * r71;
    r147 = r147 * r120;
    r147 = r147 * r38;
    r147 = r147 * r78;
    r110 = fma(r111, r147, r110);
    r110 = fma(r142, r80, r110);
    r110 = fma(r141, r137, r110);
    r110 = fma(r142, r79, r110);
    r110 = fma(r120, r126, r110);
    r110 = fma(r120, r124, r110);
    r147 = r2 * r110;
    r146 = r116 * r99;
    r146 = r146 * r56;
    r145 = r7 * r7;
    r145 = r145 * r83;
    r145 = r145 * r114;
    r145 = r145 * r56;
    r145 = fma(r132, r145, r44 * r146);
    r146 = r118 * r141;
    r146 = r146 * r85;
    r146 = r146 * r60;
    r145 = fma(r121, r146, r145);
    r132 = r63 * r120;
    r145 = fma(r123, r132, r145);
    r145 = r145 + r104;
    r145 = fma(r4, r145, r65 * r133);
    r133 = r20 * r7;
    r133 = r133 * r71;
    r133 = r133 * r141;
    r133 = r133 * r77;
    r145 = fma(r41, r133, r145);
    r104 = r42 * r71;
    r104 = r104 * r86;
    r104 = r104 * r120;
    r104 = r104 * r76;
    r104 = r104 * r83;
    r145 = fma(r44, r104, r145);
    r132 = r42 * r86;
    r132 = r132 * r120;
    r132 = r132 * r76;
    r132 = r132 * r83;
    r145 = fma(r44, r132, r145);
    r146 = r5 * r120;
    r145 = fma(r129, r146, r145);
    r144 = r7 * r89;
    r144 = r144 * r120;
    r144 = r144 * r83;
    r144 = r144 * r38;
    r145 = fma(r78, r144, r145);
    r143 = r5 * r6;
    r143 = r143 * r42;
    r143 = r143 * r42;
    r143 = r143 * r100;
    r143 = r143 * r141;
    r143 = r143 * r85;
    r145 = fma(r44, r143, r145);
    r94 = r5 * r142;
    r94 = r94 * r44;
    r145 = fma(r82, r94, r145);
    r130 = r5 * r120;
    r145 = fma(r131, r130, r145);
    r93 = r7 * r119;
    r145 = fma(r79, r93, r145);
    r139 = r20 * r7;
    r139 = r139 * r141;
    r139 = r139 * r77;
    r145 = fma(r41, r139, r145);
    r148 = r7 * r89;
    r148 = r148 * r71;
    r148 = r148 * r120;
    r148 = r148 * r83;
    r148 = r148 * r38;
    r145 = fma(r78, r148, r145);
    r145 = fma(r99, r128, r145);
    r145 = fma(r99, r79, r145);
    r145 = fma(r99, r80, r145);
    r148 = r3 * r145;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             2 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r147,
                                             r148);
    r148 = r31 * r100;
    r27 = fma(r86, r27, r89 * r23);
    r27 = fma(r89, r24, r27);
    r27 = fma(r89, r26, r27);
    r148 = r148 * r27;
    r88 = r100 * r88;
    r26 = r148 + r88;
    r24 = r30 * r18;
    r24 = r24 * r27;
    r102 = r102 + r24;
    r23 = r21 * r25;
    r102 = fma(r127, r23, r102);
    r147 = r21 * r34;
    r102 = fma(r87, r147, r102);
    r102 = fma(r8, r102, r35 * r26);
    r26 = r30 * r34;
    r26 = fma(r30, r92, r27 * r26);
    r26 = r26 + r95;
    r102 = fma(r9, r26, r102);
    r26 = r102 * r60;
    r147 = r30 * r7;
    r23 = r30 * r25;
    r23 = r23 * r27;
    r140 = r140 + r23;
    r140 = r140 + r105;
    r105 = r21 * r34;
    r92 = fma(r21, r92, r27 * r105);
    r92 = r92 + r95;
    r92 = fma(r35, r92, r8 * r140);
    r96 = r18 * r96;
    r96 = r96 * r100;
    r148 = r148 + r96;
    r92 = fma(r9, r148, r92);
    r147 = r147 * r92;
    r148 = r102 * r107;
    r148 = fma(r109, r148, r58 * r147);
    r147 = r30 * r6;
    r112 = r90 + r112;
    r90 = r18 * r21;
    r112 = fma(r127, r90, r112);
    r112 = r112 + r23;
    r88 = r96 + r88;
    r88 = fma(r8, r88, r9 * r112);
    r8 = r30 * r34;
    r8 = fma(r87, r8, r24);
    r8 = r8 + r136;
    r88 = fma(r35, r8, r88);
    r147 = r147 * r88;
    r148 = fma(r58, r147, r148);
    r8 = r7 * r7;
    r8 = r8 * r102;
    r148 = fma(r107, r8, r148);
    r26 = fma(r148, r123, r122 * r26);
    r8 = r92 * r44;
    r26 = fma(r82, r8, r26);
    r147 = r20 * r7;
    r147 = r147 * r7;
    r147 = r147 * r148;
    r147 = r147 * r83;
    r147 = r147 * r114;
    r26 = fma(r56, r147, r26);
    r147 = r20 * r6;
    r147 = r147 * r148;
    r147 = fma(r88, r117, r115 * r147);
    r8 = r148 * r111;
    r8 = r8 * r97;
    r147 = fma(r106, r8, r147);
    r147 = fma(r102, r125, r147);
    r8 = r26 + r147;
    r35 = r6 * r103;
    r35 = r35 * r148;
    r136 = r6 * r116;
    r136 = r136 * r88;
    r136 = r136 * r48;
    r136 = fma(r56, r136, r115 * r35);
    r35 = r6 * r6;
    r35 = r35 * r42;
    r35 = r35 * r42;
    r35 = r35 * r118;
    r35 = r35 * r102;
    r35 = r35 * r48;
    r136 = fma(r85, r35, r136);
    r24 = r63 * r148;
    r24 = r24 * r111;
    r24 = r24 * r97;
    r136 = fma(r106, r24, r136);
    r136 = r136 + r26;
    r136 = fma(r5, r136, r64 * r8);
    r26 = r69 * r30;
    r26 = r26 * r61;
    r26 = fma(r8, r26, r68 * r8);
    r26 = fma(r8, r72, r26);
    r26 = fma(r8, r73, r26);
    r24 = r6 * r26;
    r136 = fma(r79, r24, r136);
    r35 = r4 * r92;
    r136 = fma(r117, r35, r136);
    r87 = r89 * r148;
    r87 = r87 * r38;
    r87 = r87 * r78;
    r136 = fma(r111, r87, r136);
    r112 = r4 * r148;
    r136 = fma(r129, r112, r136);
    r9 = r89 * r71;
    r9 = r9 * r148;
    r9 = r9 * r38;
    r9 = r9 * r78;
    r136 = fma(r111, r9, r136);
    r96 = r20 * r6;
    r96 = r96 * r102;
    r96 = r96 * r77;
    r136 = fma(r41, r96, r136);
    r90 = r4 * r88;
    r90 = r90 * r44;
    r136 = fma(r82, r90, r136);
    r23 = r20 * r6;
    r23 = r23 * r71;
    r23 = r23 * r102;
    r23 = r23 * r77;
    r136 = fma(r41, r23, r136);
    r127 = r4 * r148;
    r136 = fma(r131, r127, r136);
    r136 = fma(r88, r80, r136);
    r136 = fma(r148, r126, r136);
    r136 = fma(r102, r137, r136);
    r136 = fma(r88, r79, r136);
    r136 = fma(r148, r124, r136);
    r127 = r2 * r136;
    r23 = r118 * r102;
    r23 = r23 * r85;
    r23 = r23 * r60;
    r90 = r63 * r148;
    r90 = fma(r123, r90, r121 * r23);
    r23 = r116 * r92;
    r23 = r23 * r56;
    r90 = fma(r44, r23, r90);
    r96 = r7 * r7;
    r96 = r96 * r103;
    r96 = r96 * r148;
    r96 = r96 * r83;
    r96 = r96 * r114;
    r90 = fma(r56, r96, r90);
    r90 = r90 + r147;
    r90 = fma(r4, r90, r65 * r8);
    r8 = r20 * r7;
    r8 = r8 * r102;
    r8 = r8 * r77;
    r90 = fma(r41, r8, r90);
    r147 = r20 * r7;
    r147 = r147 * r71;
    r147 = r147 * r102;
    r147 = r147 * r77;
    r90 = fma(r41, r147, r90);
    r96 = r7 * r89;
    r96 = r96 * r148;
    r96 = r96 * r83;
    r96 = r96 * r38;
    r90 = fma(r78, r96, r90);
    r23 = r5 * r148;
    r90 = fma(r129, r23, r90);
    r9 = r42 * r71;
    r9 = r9 * r86;
    r9 = r9 * r148;
    r9 = r9 * r76;
    r9 = r9 * r83;
    r90 = fma(r44, r9, r90);
    r112 = r7 * r26;
    r90 = fma(r79, r112, r90);
    r87 = r42 * r86;
    r87 = r87 * r148;
    r87 = r87 * r76;
    r87 = r87 * r83;
    r90 = fma(r44, r87, r90);
    r35 = r7 * r89;
    r35 = r35 * r71;
    r35 = r35 * r148;
    r35 = r35 * r83;
    r35 = r35 * r38;
    r90 = fma(r78, r35, r90);
    r24 = r5 * r88;
    r24 = r24 * r44;
    r90 = fma(r82, r24, r90);
    r140 = r5 * r6;
    r140 = r140 * r42;
    r140 = r140 * r42;
    r140 = r140 * r100;
    r140 = r140 * r102;
    r140 = r140 * r85;
    r90 = fma(r44, r140, r90);
    r95 = r5 * r148;
    r90 = fma(r131, r95, r90);
    r90 = fma(r92, r128, r90);
    r90 = fma(r92, r79, r90);
    r90 = fma(r92, r80, r90);
    r95 = r3 * r90;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r127, r95);
    r95 = r46 * r6;
    r95 = r95 * r116;
    r95 = r95 * r48;
    r127 = r6 * r103;
    r140 = r30 * r32;
    r140 = r140 * r7;
    r24 = r36 * r7;
    r24 = r24 * r7;
    r24 = fma(r107, r24, r58 * r140);
    r140 = r36 * r107;
    r24 = fma(r109, r140, r24);
    r35 = r30 * r46;
    r35 = r35 * r6;
    r24 = fma(r58, r35, r24);
    r127 = r127 * r24;
    r127 = fma(r115, r127, r56 * r95);
    r95 = r63 * r24;
    r95 = r95 * r111;
    r95 = r95 * r97;
    r127 = fma(r106, r95, r127);
    r35 = r36 * r6;
    r35 = r35 * r6;
    r35 = r35 * r42;
    r35 = r35 * r42;
    r35 = r35 * r118;
    r35 = r35 * r48;
    r127 = fma(r85, r35, r127);
    r140 = r32 * r44;
    r87 = r20 * r7;
    r87 = r87 * r7;
    r87 = r87 * r24;
    r87 = r87 * r83;
    r87 = r87 * r114;
    r87 = fma(r56, r87, r82 * r140);
    r140 = r36 * r60;
    r87 = fma(r122, r140, r87);
    r87 = fma(r24, r123, r87);
    r127 = r127 + r87;
    r35 = r20 * r6;
    r35 = r35 * r24;
    r35 = fma(r115, r35, r46 * r117);
    r95 = r24 * r111;
    r95 = r95 * r97;
    r35 = fma(r106, r95, r35);
    r35 = fma(r36, r125, r35);
    r87 = r87 + r35;
    r127 = fma(r64, r87, r5 * r127);
    r95 = r89 * r24;
    r95 = r95 * r38;
    r95 = r95 * r78;
    r127 = fma(r111, r95, r127);
    r140 = r89 * r71;
    r140 = r140 * r24;
    r140 = r140 * r38;
    r140 = r140 * r78;
    r127 = fma(r111, r140, r127);
    r112 = r20 * r36;
    r112 = r112 * r6;
    r112 = r112 * r71;
    r112 = r112 * r77;
    r127 = fma(r41, r112, r127);
    r9 = r4 * r32;
    r127 = fma(r117, r9, r127);
    r23 = r69 * r30;
    r23 = r23 * r61;
    r23 = fma(r87, r23, r68 * r87);
    r23 = fma(r87, r72, r23);
    r23 = fma(r87, r73, r23);
    r96 = r6 * r23;
    r127 = fma(r79, r96, r127);
    r147 = r20 * r36;
    r147 = r147 * r6;
    r147 = r147 * r77;
    r127 = fma(r41, r147, r127);
    r8 = r4 * r24;
    r127 = fma(r129, r8, r127);
    r105 = r4 * r46;
    r105 = r105 * r44;
    r127 = fma(r82, r105, r127);
    r27 = r4 * r24;
    r127 = fma(r131, r27, r127);
    r127 = fma(r24, r126, r127);
    r127 = fma(r24, r124, r127);
    r127 = fma(r46, r80, r127);
    r127 = fma(r46, r79, r127);
    r127 = fma(r36, r137, r127);
    r27 = r2 * r127;
    r105 = r32 * r116;
    r105 = r105 * r56;
    r8 = r7 * r7;
    r8 = r8 * r103;
    r8 = r8 * r24;
    r8 = r8 * r83;
    r8 = r8 * r114;
    r8 = fma(r56, r8, r44 * r105);
    r105 = r36 * r118;
    r105 = r105 * r85;
    r105 = r105 * r60;
    r8 = fma(r121, r105, r8);
    r147 = r63 * r24;
    r8 = fma(r123, r147, r8);
    r8 = r8 + r35;
    r87 = fma(r65, r87, r4 * r8);
    r8 = r7 * r89;
    r8 = r8 * r24;
    r8 = r8 * r83;
    r8 = r8 * r38;
    r87 = fma(r78, r8, r87);
    r35 = r20 * r36;
    r35 = r35 * r7;
    r35 = r35 * r77;
    r87 = fma(r41, r35, r87);
    r147 = r7 * r89;
    r147 = r147 * r71;
    r147 = r147 * r24;
    r147 = r147 * r83;
    r147 = r147 * r38;
    r87 = fma(r78, r147, r87);
    r105 = r20 * r36;
    r105 = r105 * r7;
    r105 = r105 * r71;
    r105 = r105 * r77;
    r87 = fma(r41, r105, r87);
    r96 = r42 * r71;
    r96 = r96 * r86;
    r96 = r96 * r24;
    r96 = r96 * r76;
    r96 = r96 * r83;
    r87 = fma(r44, r96, r87);
    r9 = r5 * r24;
    r87 = fma(r129, r9, r87);
    r112 = r5 * r46;
    r112 = r112 * r44;
    r87 = fma(r82, r112, r87);
    r140 = r42 * r86;
    r140 = r140 * r24;
    r140 = r140 * r76;
    r140 = r140 * r83;
    r87 = fma(r44, r140, r87);
    r95 = r7 * r23;
    r87 = fma(r79, r95, r87);
    r139 = r5 * r24;
    r87 = fma(r131, r139, r87);
    r93 = r5 * r36;
    r93 = r93 * r6;
    r93 = r93 * r42;
    r93 = r93 * r42;
    r93 = r93 * r100;
    r93 = r93 * r85;
    r87 = fma(r44, r93, r87);
    r87 = fma(r32, r80, r87);
    r87 = fma(r32, r79, r87);
    r87 = fma(r32, r128, r87);
    r93 = r3 * r87;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r27, r93);
    r93 = r53 * r60;
    r27 = r47 * r44;
    r27 = fma(r82, r27, r122 * r93);
    r93 = r53 * r7;
    r93 = r93 * r7;
    r139 = r30 * r47;
    r139 = r139 * r7;
    r139 = fma(r58, r139, r107 * r93);
    r93 = r53 * r107;
    r139 = fma(r109, r93, r139);
    r95 = r30 * r43;
    r95 = r95 * r6;
    r139 = fma(r58, r95, r139);
    r95 = r20 * r7;
    r95 = r95 * r7;
    r95 = r95 * r139;
    r95 = r95 * r83;
    r95 = r95 * r114;
    r27 = fma(r56, r95, r27);
    r27 = fma(r139, r123, r27);
    r95 = r139 * r111;
    r95 = r95 * r97;
    r95 = fma(r53, r125, r106 * r95);
    r93 = r20 * r6;
    r93 = r93 * r139;
    r95 = fma(r115, r93, r95);
    r95 = fma(r43, r117, r95);
    r93 = r27 + r95;
    r140 = r63 * r139;
    r140 = r140 * r111;
    r140 = r140 * r97;
    r112 = r53 * r6;
    r112 = r112 * r6;
    r112 = r112 * r42;
    r112 = r112 * r42;
    r112 = r112 * r118;
    r112 = r112 * r48;
    r112 = fma(r85, r112, r106 * r140);
    r140 = r6 * r103;
    r140 = r140 * r139;
    r112 = fma(r115, r140, r112);
    r9 = r43 * r6;
    r9 = r9 * r116;
    r9 = r9 * r48;
    r112 = fma(r56, r9, r112);
    r112 = r112 + r27;
    r112 = fma(r5, r112, r64 * r93);
    r27 = r4 * r139;
    r112 = fma(r129, r27, r112);
    r9 = r20 * r53;
    r9 = r9 * r6;
    r9 = r9 * r77;
    r112 = fma(r41, r9, r112);
    r140 = r4 * r43;
    r140 = r140 * r44;
    r112 = fma(r82, r140, r112);
    r96 = r89 * r71;
    r96 = r96 * r139;
    r96 = r96 * r38;
    r96 = r96 * r78;
    r112 = fma(r111, r96, r112);
    r105 = r20 * r53;
    r105 = r105 * r6;
    r105 = r105 * r71;
    r105 = r105 * r77;
    r112 = fma(r41, r105, r112);
    r147 = r89 * r139;
    r147 = r147 * r38;
    r147 = r147 * r78;
    r112 = fma(r111, r147, r112);
    r35 = r4 * r47;
    r112 = fma(r117, r35, r112);
    r8 = r69 * r30;
    r8 = r8 * r61;
    r8 = fma(r68, r93, r93 * r8);
    r8 = fma(r93, r73, r8);
    r8 = fma(r93, r72, r8);
    r130 = r6 * r8;
    r112 = fma(r79, r130, r112);
    r94 = r4 * r139;
    r112 = fma(r131, r94, r112);
    r112 = fma(r43, r79, r112);
    r112 = fma(r139, r126, r112);
    r112 = fma(r43, r80, r112);
    r112 = fma(r139, r124, r112);
    r112 = fma(r53, r137, r112);
    r94 = r2 * r112;
    r130 = r53 * r118;
    r130 = r130 * r85;
    r130 = r130 * r60;
    r35 = r47 * r116;
    r35 = r35 * r56;
    r35 = fma(r44, r35, r121 * r130);
    r130 = r63 * r139;
    r35 = fma(r123, r130, r35);
    r147 = r7 * r7;
    r147 = r147 * r103;
    r147 = r147 * r139;
    r147 = r147 * r83;
    r147 = r147 * r114;
    r35 = fma(r56, r147, r35);
    r35 = r35 + r95;
    r35 = fma(r4, r35, r65 * r93);
    r93 = r20 * r53;
    r93 = r93 * r7;
    r93 = r93 * r77;
    r35 = fma(r41, r93, r35);
    r95 = r20 * r53;
    r95 = r95 * r7;
    r95 = r95 * r71;
    r95 = r95 * r77;
    r35 = fma(r41, r95, r35);
    r147 = r5 * r139;
    r35 = fma(r129, r147, r35);
    r130 = r7 * r89;
    r130 = r130 * r71;
    r130 = r130 * r139;
    r130 = r130 * r83;
    r130 = r130 * r38;
    r35 = fma(r78, r130, r35);
    r105 = r7 * r89;
    r105 = r105 * r139;
    r105 = r105 * r83;
    r105 = r105 * r38;
    r35 = fma(r78, r105, r35);
    r96 = r5 * r43;
    r96 = r96 * r44;
    r35 = fma(r82, r96, r35);
    r140 = r7 * r8;
    r35 = fma(r79, r140, r35);
    r9 = r42 * r86;
    r9 = r9 * r139;
    r9 = r9 * r76;
    r9 = r9 * r83;
    r35 = fma(r44, r9, r35);
    r27 = r5 * r139;
    r35 = fma(r131, r27, r35);
    r143 = r42 * r71;
    r143 = r143 * r86;
    r143 = r143 * r139;
    r143 = r143 * r76;
    r143 = r143 * r83;
    r35 = fma(r44, r143, r35);
    r144 = r5 * r53;
    r144 = r144 * r6;
    r144 = r144 * r42;
    r144 = r144 * r42;
    r144 = r144 * r100;
    r144 = r144 * r85;
    r35 = fma(r44, r144, r35);
    r35 = fma(r47, r80, r35);
    r35 = fma(r47, r128, r35);
    r35 = fma(r47, r79, r35);
    r144 = r3 * r35;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r94, r144);
    r144 = r20 * r6;
    r94 = r30 * r39;
    r94 = r94 * r7;
    r143 = r45 * r7;
    r143 = r143 * r7;
    r143 = fma(r107, r143, r58 * r94);
    r94 = r30 * r40;
    r94 = r94 * r6;
    r143 = fma(r58, r94, r143);
    r27 = r45 * r107;
    r143 = fma(r109, r27, r143);
    r144 = r144 * r143;
    r27 = r143 * r111;
    r27 = r27 * r97;
    r27 = fma(r106, r27, r115 * r144);
    r27 = fma(r40, r117, r27);
    r27 = fma(r45, r125, r27);
    r144 = r45 * r60;
    r94 = r39 * r44;
    r94 = fma(r82, r94, r122 * r144);
    r144 = r20 * r7;
    r144 = r144 * r7;
    r144 = r144 * r143;
    r144 = r144 * r83;
    r144 = r144 * r114;
    r94 = fma(r56, r144, r94);
    r94 = fma(r143, r123, r94);
    r144 = r27 + r94;
    r9 = r6 * r103;
    r9 = r9 * r143;
    r140 = r63 * r143;
    r140 = r140 * r111;
    r140 = r140 * r97;
    r140 = fma(r106, r140, r115 * r9);
    r9 = r40 * r6;
    r9 = r9 * r116;
    r9 = r9 * r48;
    r140 = fma(r56, r9, r140);
    r96 = r45 * r6;
    r96 = r96 * r6;
    r96 = r96 * r42;
    r96 = r96 * r42;
    r96 = r96 * r118;
    r96 = r96 * r48;
    r140 = fma(r85, r96, r140);
    r140 = r140 + r94;
    r140 = fma(r5, r140, r64 * r144);
    r94 = r4 * r39;
    r140 = fma(r117, r94, r140);
    r96 = r69 * r30;
    r96 = r96 * r61;
    r96 = fma(r68, r144, r144 * r96);
    r96 = fma(r144, r72, r96);
    r96 = fma(r144, r73, r96);
    r9 = r6 * r96;
    r140 = fma(r79, r9, r140);
    r105 = r20 * r45;
    r105 = r105 * r6;
    r105 = r105 * r71;
    r105 = r105 * r77;
    r140 = fma(r41, r105, r140);
    r130 = r143 * r131;
    r147 = r4 * r143;
    r140 = fma(r129, r147, r140);
    r95 = r89 * r143;
    r95 = r95 * r38;
    r95 = r95 * r78;
    r140 = fma(r111, r95, r140);
    r93 = r4 * r40;
    r93 = r93 * r44;
    r140 = fma(r82, r93, r140);
    r146 = r20 * r45;
    r146 = r146 * r6;
    r146 = r146 * r77;
    r140 = fma(r41, r146, r140);
    r132 = r89 * r71;
    r132 = r132 * r143;
    r132 = r132 * r38;
    r132 = r132 * r78;
    r140 = fma(r111, r132, r140);
    r140 = fma(r143, r126, r140);
    r140 = fma(r4, r130, r140);
    r140 = fma(r45, r137, r140);
    r140 = fma(r40, r80, r140);
    r140 = fma(r40, r79, r140);
    r140 = fma(r143, r124, r140);
    r132 = r2 * r140;
    r146 = r45 * r118;
    r146 = r146 * r85;
    r146 = r146 * r60;
    r93 = r39 * r116;
    r93 = r93 * r56;
    r93 = fma(r44, r93, r121 * r146);
    r146 = r63 * r143;
    r93 = fma(r123, r146, r93);
    r95 = r7 * r7;
    r95 = r95 * r103;
    r95 = r95 * r143;
    r95 = r95 * r83;
    r95 = r95 * r114;
    r93 = fma(r56, r95, r93);
    r93 = r93 + r27;
    r93 = fma(r4, r93, r65 * r144);
    r144 = r42 * r86;
    r144 = r144 * r143;
    r144 = r144 * r76;
    r144 = r144 * r83;
    r93 = fma(r44, r144, r93);
    r27 = r20 * r45;
    r27 = r27 * r7;
    r27 = r27 * r77;
    r93 = fma(r41, r27, r93);
    r95 = r42 * r71;
    r95 = r95 * r86;
    r95 = r95 * r143;
    r95 = r95 * r76;
    r95 = r95 * r83;
    r93 = fma(r44, r95, r93);
    r146 = r5 * r143;
    r93 = fma(r129, r146, r93);
    r147 = r5 * r45;
    r147 = r147 * r6;
    r147 = r147 * r42;
    r147 = r147 * r42;
    r147 = r147 * r100;
    r147 = r147 * r85;
    r93 = fma(r44, r147, r93);
    r105 = r7 * r89;
    r105 = r105 * r71;
    r105 = r105 * r143;
    r105 = r105 * r83;
    r105 = r105 * r38;
    r93 = fma(r78, r105, r93);
    r9 = r5 * r40;
    r9 = r9 * r44;
    r93 = fma(r82, r9, r93);
    r94 = r7 * r89;
    r94 = r94 * r143;
    r94 = r94 * r83;
    r94 = r94 * r38;
    r93 = fma(r78, r94, r93);
    r104 = r7 * r96;
    r93 = fma(r79, r104, r93);
    r133 = r20 * r45;
    r133 = r133 * r7;
    r133 = r133 * r71;
    r133 = r133 * r77;
    r93 = fma(r41, r133, r93);
    r93 = fma(r39, r128, r93);
    r93 = fma(r39, r79, r93);
    r93 = fma(r5, r130, r93);
    r93 = fma(r39, r80, r93);
    r133 = r3 * r93;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             10 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r132,
                                             r133);
    r133 = r3 * r20;
    r133 = r133 * r1;
    r132 = r20 * r0;
    r104 = r2 * r132;
    r133 = fma(r113, r104, r138 * r133);
    r94 = r3 * r20;
    r94 = r94 * r1;
    r94 = fma(r110, r104, r145 * r94);
    WriteSum2<double, double>((double*)inout_shared, r133, r94);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r94 = r3 * r20;
    r94 = r94 * r1;
    r94 = fma(r136, r104, r90 * r94);
    r133 = r3 * r20;
    r133 = r133 * r1;
    r133 = fma(r127, r104, r87 * r133);
    WriteSum2<double, double>((double*)inout_shared, r94, r133);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r133 = r3 * r20;
    r133 = r133 * r1;
    r133 = fma(r112, r104, r35 * r133);
    r94 = r3 * r20;
    r94 = r94 * r1;
    r94 = fma(r140, r104, r93 * r94);
    WriteSum2<double, double>((double*)inout_shared, r133, r94);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r94 = r113 * r113;
    r133 = r2 * r2;
    r9 = r138 * r138;
    r105 = r3 * r3;
    r9 = fma(r105, r9, r133 * r94);
    r94 = r110 * r110;
    r147 = r145 * r145;
    r147 = fma(r105, r147, r133 * r94);
    WriteSum2<double, double>((double*)inout_shared, r9, r147);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r147 = r90 * r90;
    r9 = r136 * r136;
    r9 = fma(r133, r9, r105 * r147);
    r147 = r87 * r87;
    r94 = r127 * r127;
    r94 = fma(r133, r94, r105 * r147);
    WriteSum2<double, double>((double*)inout_shared, r9, r94);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r94 = r35 * r35;
    r9 = r112 * r112;
    r9 = fma(r133, r9, r105 * r94);
    r94 = r140 * r140;
    r147 = r93 * r93;
    r147 = fma(r105, r147, r133 * r94);
    WriteSum2<double, double>((double*)inout_shared, r9, r147);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r147 = r113 * r110;
    r9 = r138 * r145;
    r9 = fma(r105, r9, r133 * r147);
    r147 = r138 * r90;
    r94 = r113 * r136;
    r94 = fma(r133, r94, r105 * r147);
    WriteSum2<double, double>((double*)inout_shared, r9, r94);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r94 = r113 * r127;
    r9 = r138 * r87;
    r9 = fma(r105, r9, r133 * r94);
    r94 = r138 * r35;
    r147 = r113 * r112;
    r147 = fma(r133, r147, r105 * r94);
    WriteSum2<double, double>((double*)inout_shared, r9, r147);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r147 = r113 * r140;
    r9 = r138 * r93;
    r9 = fma(r105, r9, r133 * r147);
    r147 = r145 * r90;
    r94 = r110 * r136;
    r94 = fma(r133, r94, r105 * r147);
    WriteSum2<double, double>((double*)inout_shared, r9, r94);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r94 = r145 * r87;
    r9 = r110 * r127;
    r9 = fma(r133, r9, r105 * r94);
    r94 = r110 * r112;
    r147 = r145 * r35;
    r147 = fma(r105, r147, r133 * r94);
    WriteSum2<double, double>((double*)inout_shared, r9, r147);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r147 = r110 * r140;
    r9 = r145 * r93;
    r9 = fma(r105, r9, r133 * r147);
    r147 = r90 * r87;
    r94 = r136 * r127;
    r94 = fma(r133, r94, r105 * r147);
    WriteSum2<double, double>((double*)inout_shared, r9, r94);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r94 = r90 * r35;
    r9 = r136 * r112;
    r9 = fma(r133, r9, r105 * r94);
    r94 = r90 * r93;
    r147 = r136 * r140;
    r147 = fma(r133, r147, r105 * r94);
    WriteSum2<double, double>((double*)inout_shared, r9, r147);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r147 = r87 * r35;
    r9 = r127 * r112;
    r9 = fma(r133, r9, r105 * r147);
    r147 = r87 * r93;
    r94 = r127 * r140;
    r94 = fma(r133, r94, r105 * r147);
    WriteSum2<double, double>((double*)inout_shared, r9, r94);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r94 = r35 * r93;
    r9 = r112 * r140;
    r9 = fma(r133, r9, r105 * r94);
    WriteSum1<double, double>((double*)inout_shared, r9);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             0 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r67,
                                             r66);
    r9 = r2 * r6;
    r9 = r9 * r61;
    r9 = r9 * r79;
    r94 = r3 * r7;
    r94 = r94 * r61;
    r94 = r94 * r79;
    WriteIdx2<1024, double, double, double2>(
        out_calib_jac, 2 * out_calib_jac_num_alloc, global_thread_idx, r9, r94);
    r147 = r2 * r6;
    r147 = r147 * r79;
    r147 = r147 * r70;
    r146 = r3 * r7;
    r146 = r146 * r79;
    r146 = r146 * r70;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             4 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r147,
                                             r146);
    r130 = r3 * r81;
    r95 = r2 * r6;
    r95 = r95 * r44;
    r95 = r95 * r82;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             6 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r95,
                                             r130);
    r27 = r2 * r62;
    r144 = r3 * r6;
    r144 = r144 * r44;
    r144 = r144 * r82;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             8 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r27,
                                             r144);
    r149 = r2 * r6;
    r149 = r149 * r79;
    r149 = r149 * r75;
    r150 = r3 * r7;
    r150 = r150 * r79;
    r150 = r150 * r75;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             10 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r149,
                                             r150);
    r151 = r2 * r6;
    r151 = r151 * r79;
    r151 = r151 * r74;
    r152 = r3 * r7;
    r152 = r152 * r79;
    r152 = r152 * r74;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             12 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r151,
                                             r152);
    r153 = r2 * r61;
    r154 = r3 * r61;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             14 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r153,
                                             r154);
    r155 = r20 * r66;
    r155 = r155 * r1;
    r156 = r67 * r132;
    WriteSum2<double, double>((double*)inout_shared, r156, r155);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r155 = r20 * r1;
    WriteSum2<double, double>((double*)inout_shared, r132, r155);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r155 = r6 * r61;
    r155 = r155 * r79;
    r132 = r3 * r20;
    r132 = r132 * r7;
    r132 = r132 * r61;
    r132 = r132 * r1;
    r132 = fma(r79, r132, r104 * r155);
    r155 = r3 * r20;
    r155 = r155 * r7;
    r155 = r155 * r1;
    r155 = r155 * r79;
    r156 = r6 * r79;
    r156 = r156 * r70;
    r156 = fma(r104, r156, r70 * r155);
    WriteSum2<double, double>((double*)inout_shared, r132, r156);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            4 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r156 = r3 * r20;
    r156 = r156 * r81;
    r132 = r2 * r21;
    r132 = r132 * r6;
    r132 = r132 * r0;
    r132 = r132 * r56;
    r132 = fma(r44, r132, r1 * r156);
    r156 = r3 * r21;
    r156 = r156 * r6;
    r156 = r156 * r1;
    r156 = r156 * r56;
    r156 = fma(r44, r156, r62 * r104);
    WriteSum2<double, double>((double*)inout_shared, r132, r156);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            6 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r156 = r3 * r20;
    r156 = r156 * r7;
    r156 = r156 * r1;
    r156 = r156 * r79;
    r132 = r6 * r79;
    r132 = r132 * r75;
    r132 = fma(r104, r132, r75 * r156);
    r156 = r3 * r20;
    r156 = r156 * r7;
    r156 = r156 * r1;
    r156 = r156 * r79;
    r0 = r6 * r79;
    r0 = r0 * r74;
    r0 = fma(r104, r0, r74 * r156);
    WriteSum2<double, double>((double*)inout_shared, r132, r0);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            8 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r3 * r20;
    r0 = r0 * r61;
    r0 = r0 * r1;
    r132 = r61 * r104;
    WriteSum2<double, double>((double*)inout_shared, r132, r0);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            10 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r67 * r67;
    r132 = r66 * r66;
    WriteSum2<double, double>((double*)inout_shared, r0, r132);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            0 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r22, r22);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            2 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = r56 * r70;
    r22 = r22 * r105;
    r132 = r6 * r56;
    r132 = r132 * r70;
    r132 = r132 * r133;
    r132 = fma(r106, r132, r60 * r22);
    r22 = r56 * r105;
    r22 = r22 * r60;
    r0 = r6 * r56;
    r0 = r0 * r133;
    r0 = r0 * r106;
    r0 = fma(r74, r0, r74 * r22);
    WriteSum2<double, double>((double*)inout_shared, r132, r0);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            4 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r132 = r7 * r7;
    r50 = r57 * r50;
    r50 = 1.0 / r50;
    r49 = r54 * r49;
    r49 = 1.0 / r49;
    r132 = r132 * r42;
    r132 = r132 * r42;
    r132 = r132 * r135;
    r132 = r132 * r50;
    r132 = r132 * r49;
    r132 = r132 * r121;
    r132 = r132 * r109;
    r49 = r81 * r105;
    r50 = fma(r81, r49, r133 * r132);
    r135 = r62 * r133;
    r132 = fma(r62, r135, r105 * r132);
    WriteSum2<double, double>((double*)inout_shared, r50, r132);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            6 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r132 = r56 * r105;
    r50 = r75 * r75;
    r132 = r132 * r60;
    r54 = r6 * r56;
    r54 = r54 * r133;
    r54 = r54 * r106;
    r132 = fma(r50, r54, r50 * r132);
    r57 = r74 * r74;
    r22 = r56 * r105;
    r22 = r22 * r60;
    r57 = fma(r54, r57, r57 * r22);
    WriteSum2<double, double>((double*)inout_shared, r132, r57);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            8 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = r70 * r133;
    r156 = r70 * r105;
    WriteSum2<double, double>((double*)inout_shared, r57, r156);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            10 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r156 = 0.00000000000000000e+00;
    WriteSum2<double, double>((double*)inout_shared, r156, r67);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = r2 * r6;
    r57 = r57 * r61;
    r57 = r57 * r67;
    r57 = r57 * r79;
    WriteSum2<double, double>((double*)inout_shared, r156, r57);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            2 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = r2 * r6;
    r57 = r57 * r67;
    r57 = r57 * r79;
    r57 = r57 * r70;
    r155 = r2 * r6;
    r155 = r155 * r67;
    r155 = r155 * r44;
    r155 = r155 * r82;
    WriteSum2<double, double>((double*)inout_shared, r57, r155);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = r2 * r62;
    r62 = r62 * r67;
    r155 = r2 * r6;
    r155 = r155 * r67;
    r155 = r155 * r79;
    r155 = r155 * r75;
    WriteSum2<double, double>((double*)inout_shared, r62, r155);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            6 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r155 = r2 * r61;
    r155 = r155 * r67;
    r62 = r2 * r6;
    r62 = r62 * r67;
    r62 = r62 * r79;
    r62 = r62 * r74;
    WriteSum2<double, double>((double*)inout_shared, r62, r155);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            8 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r155 = r3 * r7;
    r155 = r155 * r61;
    r155 = r155 * r66;
    r155 = r155 * r79;
    WriteSum2<double, double>((double*)inout_shared, r66, r155);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            12 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = r3 * r81;
    r81 = r81 * r66;
    r155 = r3 * r7;
    r155 = r155 * r66;
    r155 = r155 * r79;
    r155 = r155 * r70;
    WriteSum2<double, double>((double*)inout_shared, r155, r81);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            14 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = r3 * r6;
    r81 = r81 * r66;
    r81 = r81 * r44;
    r81 = r81 * r82;
    r155 = r3 * r7;
    r155 = r155 * r66;
    r155 = r155 * r79;
    r155 = r155 * r75;
    WriteSum2<double, double>((double*)inout_shared, r81, r155);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            16 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r155 = r3 * r7;
    r155 = r155 * r66;
    r155 = r155 * r79;
    r155 = r155 * r74;
    WriteSum2<double, double>((double*)inout_shared, r155, r156);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            18 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r155 = r3 * r61;
    r155 = r155 * r66;
    WriteSum2<double, double>((double*)inout_shared, r155, r156);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            20 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r9, r147);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            22 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r95, r27);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            24 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r149, r151);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            26 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r153, r156);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            28 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r94, r146);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            30 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r130, r144);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            32 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r150, r152);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            34 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r156, r154);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            36 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r154 = r56 * r75;
    r154 = r154 * r105;
    r152 = r6 * r56;
    r152 = r152 * r75;
    r152 = r152 * r133;
    r152 = fma(r106, r152, r60 * r154);
    r154 = r49 * r84;
    r150 = r30 * r7;
    r150 = r150 * r42;
    r150 = r150 * r61;
    r150 = r150 * r85;
    r150 = r150 * r114;
    r150 = r150 * r133;
    r150 = r150 * r121;
    r150 = fma(r109, r150, r61 * r154);
    WriteSum2<double, double>((double*)inout_shared, r152, r150);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            38 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r150 = r30 * r6;
    r150 = r150 * r7;
    r150 = r150 * r7;
    r150 = r150 * r42;
    r150 = r150 * r61;
    r150 = r150 * r85;
    r150 = r150 * r114;
    r150 = r150 * r105;
    r152 = r6 * r135;
    r144 = r79 * r152;
    r150 = fma(r61, r144, r121 * r150);
    WriteSum2<double, double>((double*)inout_shared, r150, r0);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            40 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r6 * r79;
    r0 = r0 * r70;
    r0 = r0 * r133;
    r150 = r56 * r105;
    r130 = r61 * r74;
    r150 = r150 * r60;
    r150 = fma(r130, r54, r130 * r150);
    WriteSum2<double, double>((double*)inout_shared, r150, r0);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            42 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r7 * r79;
    r0 = r0 * r70;
    r0 = r0 * r105;
    r146 = r30 * r7;
    r146 = r146 * r42;
    r146 = r146 * r85;
    r146 = r146 * r114;
    r146 = r146 * r70;
    r146 = r146 * r133;
    r146 = r146 * r121;
    r146 = fma(r109, r146, r70 * r154);
    WriteSum2<double, double>((double*)inout_shared, r0, r146);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            44 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r146 = r30 * r6;
    r146 = r146 * r7;
    r146 = r146 * r7;
    r146 = r146 * r42;
    r146 = r146 * r85;
    r146 = r146 * r114;
    r146 = r146 * r70;
    r146 = r146 * r105;
    r146 = fma(r70, r144, r121 * r146);
    WriteSum2<double, double>((double*)inout_shared, r146, r150);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            46 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r150 = r6 * r79;
    r150 = r150 * r75;
    r150 = r150 * r133;
    WriteSum2<double, double>((double*)inout_shared, r132, r150);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            48 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r150 = r7 * r79;
    r150 = r150 * r75;
    r150 = r150 * r105;
    r132 = r44 * r82;
    r146 = r6 * r49;
    r146 = fma(r132, r146, r152 * r132);
    WriteSum2<double, double>((double*)inout_shared, r150, r146);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            50 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r146 = r30 * r7;
    r146 = r146 * r42;
    r146 = r146 * r85;
    r146 = r146 * r114;
    r146 = r146 * r75;
    r146 = r146 * r133;
    r146 = r146 * r121;
    r146 = fma(r109, r146, r75 * r154);
    r150 = r30 * r7;
    r150 = r150 * r42;
    r150 = r150 * r85;
    r150 = r150 * r114;
    r150 = r150 * r133;
    r150 = r150 * r121;
    r150 = r150 * r74;
    r150 = fma(r109, r150, r74 * r154);
    WriteSum2<double, double>((double*)inout_shared, r146, r150);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            52 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r150 = r6 * r61;
    r150 = r150 * r44;
    r150 = r150 * r82;
    r150 = r150 * r133;
    r146 = r61 * r49;
    WriteSum2<double, double>((double*)inout_shared, r150, r146);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            54 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r146 = r30 * r6;
    r146 = r146 * r7;
    r146 = r146 * r7;
    r146 = r146 * r42;
    r146 = r146 * r85;
    r146 = r146 * r114;
    r146 = r146 * r75;
    r146 = r146 * r105;
    r146 = fma(r75, r144, r121 * r146);
    r150 = r30 * r6;
    r150 = r150 * r7;
    r150 = r150 * r7;
    r150 = r150 * r42;
    r150 = r150 * r85;
    r150 = r150 * r114;
    r150 = r150 * r105;
    r150 = r150 * r121;
    r144 = fma(r74, r144, r74 * r150);
    WriteSum2<double, double>((double*)inout_shared, r146, r144);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            56 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r135 = r61 * r135;
    r144 = r6 * r61;
    r144 = r144 * r44;
    r144 = r144 * r82;
    r144 = r144 * r105;
    WriteSum2<double, double>((double*)inout_shared, r135, r144);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            58 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r144 = r6 * r79;
    r144 = r144 * r133;
    r144 = r144 * r74;
    r50 = r61 * r50;
    r54 = fma(r50, r54, r50 * r22);
    WriteSum2<double, double>((double*)inout_shared, r54, r144);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            60 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r144 = r7 * r79;
    r144 = r144 * r105;
    r144 = r144 * r74;
    r74 = r6 * r79;
    r74 = r74 * r133;
    r74 = r74 * r130;
    WriteSum2<double, double>((double*)inout_shared, r144, r74);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            62 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r130 = r105 * r130;
    r130 = r130 * r84;
    WriteSum2<double, double>((double*)inout_shared, r130, r156);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            64 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r156 = r51 * r6;
    r156 = r156 * r6;
    r156 = r156 * r42;
    r156 = r156 * r42;
    r156 = r156 * r118;
    r156 = r156 * r48;
    r130 = r30 * r29;
    r130 = r130 * r6;
    r84 = r51 * r7;
    r84 = r84 * r7;
    r84 = fma(r107, r84, r58 * r130);
    r130 = r51 * r107;
    r84 = fma(r109, r130, r84);
    r74 = r30 * r59;
    r74 = r74 * r7;
    r84 = fma(r58, r74, r84);
    r74 = r63 * r84;
    r74 = r74 * r111;
    r74 = r74 * r97;
    r74 = fma(r106, r74, r85 * r156);
    r156 = r29 * r6;
    r156 = r156 * r116;
    r156 = r156 * r48;
    r74 = fma(r56, r156, r74);
    r130 = r6 * r103;
    r130 = r130 * r84;
    r74 = fma(r115, r130, r74);
    r144 = r51 * r60;
    r54 = r20 * r7;
    r54 = r54 * r7;
    r54 = r54 * r84;
    r54 = r54 * r83;
    r54 = r54 * r114;
    r54 = fma(r56, r54, r122 * r144);
    r144 = r59 * r44;
    r54 = fma(r82, r144, r54);
    r54 = fma(r84, r123, r54);
    r74 = r74 + r54;
    r130 = r84 * r111;
    r130 = r130 * r97;
    r130 = fma(r106, r130, r51 * r125);
    r156 = r20 * r6;
    r156 = r156 * r84;
    r130 = fma(r115, r156, r130);
    r130 = fma(r29, r117, r130);
    r54 = r54 + r130;
    r74 = fma(r64, r54, r5 * r74);
    r156 = r69 * r30;
    r156 = r156 * r61;
    r156 = fma(r54, r156, r68 * r54);
    r156 = fma(r54, r72, r156);
    r156 = fma(r54, r73, r156);
    r144 = r6 * r156;
    r74 = fma(r79, r144, r74);
    r50 = r4 * r84;
    r74 = fma(r129, r50, r74);
    r22 = r20 * r51;
    r22 = r22 * r6;
    r22 = r22 * r71;
    r22 = r22 * r77;
    r74 = fma(r41, r22, r74);
    r135 = r4 * r59;
    r74 = fma(r117, r135, r74);
    r146 = r89 * r84;
    r146 = r146 * r38;
    r146 = r146 * r78;
    r74 = fma(r111, r146, r74);
    r150 = r89 * r71;
    r150 = r150 * r84;
    r150 = r150 * r38;
    r150 = r150 * r78;
    r74 = fma(r111, r150, r74);
    r154 = r20 * r51;
    r154 = r154 * r6;
    r154 = r154 * r77;
    r74 = fma(r41, r154, r74);
    r132 = r4 * r29;
    r132 = r132 * r44;
    r74 = fma(r82, r132, r74);
    r152 = r4 * r84;
    r74 = fma(r131, r152, r74);
    r74 = fma(r51, r137, r74);
    r74 = fma(r29, r80, r74);
    r74 = fma(r29, r79, r74);
    r74 = fma(r84, r126, r74);
    r74 = fma(r84, r124, r74);
    r152 = r2 * r74;
    r132 = r51 * r118;
    r132 = r132 * r85;
    r132 = r132 * r60;
    r154 = r7 * r7;
    r154 = r154 * r103;
    r154 = r154 * r84;
    r154 = r154 * r83;
    r154 = r154 * r114;
    r154 = fma(r56, r154, r121 * r132);
    r132 = r63 * r84;
    r154 = fma(r123, r132, r154);
    r150 = r59 * r116;
    r150 = r150 * r56;
    r154 = fma(r44, r150, r154);
    r154 = r154 + r130;
    r54 = fma(r65, r54, r4 * r154);
    r154 = r7 * r89;
    r154 = r154 * r84;
    r154 = r154 * r83;
    r154 = r154 * r38;
    r54 = fma(r78, r154, r54);
    r130 = r7 * r89;
    r130 = r130 * r71;
    r130 = r130 * r84;
    r130 = r130 * r83;
    r130 = r130 * r38;
    r54 = fma(r78, r130, r54);
    r150 = r5 * r51;
    r150 = r150 * r6;
    r150 = r150 * r42;
    r150 = r150 * r42;
    r150 = r150 * r100;
    r150 = r150 * r85;
    r54 = fma(r44, r150, r54);
    r132 = r20 * r51;
    r132 = r132 * r7;
    r132 = r132 * r77;
    r54 = fma(r41, r132, r54);
    r146 = r5 * r84;
    r54 = fma(r129, r146, r54);
    r135 = r20 * r51;
    r135 = r135 * r7;
    r135 = r135 * r71;
    r135 = r135 * r77;
    r54 = fma(r41, r135, r54);
    r22 = r42 * r71;
    r22 = r22 * r86;
    r22 = r22 * r84;
    r22 = r22 * r76;
    r22 = r22 * r83;
    r54 = fma(r44, r22, r54);
    r50 = r42 * r86;
    r50 = r50 * r84;
    r50 = r50 * r76;
    r50 = r50 * r83;
    r54 = fma(r44, r50, r54);
    r144 = r7 * r156;
    r54 = fma(r79, r144, r54);
    r0 = r5 * r29;
    r0 = r0 * r44;
    r54 = fma(r82, r0, r54);
    r94 = r5 * r84;
    r54 = fma(r131, r94, r54);
    r54 = fma(r59, r80, r54);
    r54 = fma(r59, r128, r54);
    r54 = fma(r59, r79, r54);
    r94 = r3 * r54;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r152,
                                             r94);
    r94 = r33 * r6;
    r94 = r94 * r116;
    r94 = r94 * r48;
    r152 = r55 * r6;
    r152 = r152 * r6;
    r152 = r152 * r42;
    r152 = r152 * r42;
    r152 = r152 * r118;
    r152 = r152 * r48;
    r152 = fma(r85, r152, r56 * r94);
    r94 = r6 * r103;
    r0 = r55 * r107;
    r144 = r30 * r33;
    r144 = r144 * r6;
    r144 = fma(r58, r144, r109 * r0);
    r0 = r30 * r19;
    r0 = r0 * r7;
    r144 = fma(r58, r0, r144);
    r50 = r55 * r7;
    r50 = r50 * r7;
    r144 = fma(r107, r50, r144);
    r94 = r94 * r144;
    r152 = fma(r115, r94, r152);
    r50 = r63 * r144;
    r50 = r50 * r111;
    r50 = r50 * r97;
    r152 = fma(r106, r50, r152);
    r0 = r55 * r60;
    r0 = fma(r144, r123, r122 * r0);
    r22 = r19 * r44;
    r0 = fma(r82, r22, r0);
    r135 = r20 * r7;
    r135 = r135 * r7;
    r135 = r135 * r144;
    r135 = r135 * r83;
    r135 = r135 * r114;
    r0 = fma(r56, r135, r0);
    r152 = r152 + r0;
    r50 = fma(r55, r125, r33 * r117);
    r94 = r20 * r6;
    r94 = r94 * r144;
    r50 = fma(r115, r94, r50);
    r135 = r144 * r111;
    r135 = r135 * r97;
    r50 = fma(r106, r135, r50);
    r0 = r0 + r50;
    r152 = fma(r64, r0, r5 * r152);
    r135 = r89 * r144;
    r135 = r135 * r38;
    r135 = r135 * r78;
    r152 = fma(r111, r135, r152);
    r94 = r4 * r19;
    r152 = fma(r117, r94, r152);
    r22 = r144 * r129;
    r146 = r89 * r71;
    r146 = r146 * r144;
    r146 = r146 * r38;
    r146 = r146 * r78;
    r152 = fma(r111, r146, r152);
    r132 = r20 * r55;
    r132 = r132 * r6;
    r132 = r132 * r77;
    r152 = fma(r41, r132, r152);
    r150 = r4 * r144;
    r152 = fma(r131, r150, r152);
    r130 = r69 * r30;
    r130 = r130 * r61;
    r130 = fma(r0, r130, r68 * r0);
    r130 = fma(r0, r72, r130);
    r130 = fma(r0, r73, r130);
    r154 = r6 * r130;
    r152 = fma(r79, r154, r152);
    r153 = r20 * r55;
    r153 = r153 * r6;
    r153 = r153 * r71;
    r153 = r153 * r77;
    r152 = fma(r41, r153, r152);
    r151 = r4 * r33;
    r151 = r151 * r44;
    r152 = fma(r82, r151, r152);
    r152 = fma(r33, r80, r152);
    r152 = fma(r4, r22, r152);
    r152 = fma(r33, r79, r152);
    r152 = fma(r144, r126, r152);
    r152 = fma(r144, r124, r152);
    r152 = fma(r55, r137, r152);
    r151 = r2 * r152;
    r153 = r55 * r118;
    r153 = r153 * r85;
    r153 = r153 * r60;
    r154 = r63 * r144;
    r154 = fma(r123, r154, r121 * r153);
    r153 = r19 * r116;
    r153 = r153 * r56;
    r154 = fma(r44, r153, r154);
    r123 = r7 * r7;
    r123 = r123 * r103;
    r123 = r123 * r144;
    r123 = r123 * r83;
    r123 = r123 * r114;
    r154 = fma(r56, r123, r154);
    r154 = r154 + r50;
    r0 = fma(r65, r0, r4 * r154);
    r154 = r20 * r55;
    r154 = r154 * r7;
    r154 = r154 * r77;
    r0 = fma(r41, r154, r0);
    r50 = r42 * r71;
    r50 = r50 * r86;
    r50 = r50 * r144;
    r50 = r50 * r76;
    r50 = r50 * r83;
    r0 = fma(r44, r50, r0);
    r123 = r7 * r89;
    r123 = r123 * r71;
    r123 = r123 * r144;
    r123 = r123 * r83;
    r123 = r123 * r38;
    r0 = fma(r78, r123, r0);
    r153 = r7 * r89;
    r153 = r153 * r144;
    r153 = r153 * r83;
    r153 = r153 * r38;
    r0 = fma(r78, r153, r0);
    r150 = r20 * r55;
    r150 = r150 * r7;
    r150 = r150 * r71;
    r150 = r150 * r77;
    r0 = fma(r41, r150, r0);
    r132 = r5 * r144;
    r0 = fma(r131, r132, r0);
    r146 = r42 * r86;
    r146 = r146 * r144;
    r146 = r146 * r76;
    r146 = r146 * r83;
    r0 = fma(r44, r146, r0);
    r94 = r7 * r130;
    r0 = fma(r79, r94, r0);
    r135 = r5 * r55;
    r135 = r135 * r6;
    r135 = r135 * r42;
    r135 = r135 * r42;
    r135 = r135 * r100;
    r135 = r135 * r85;
    r0 = fma(r44, r135, r0);
    r149 = r5 * r33;
    r149 = r149 * r44;
    r0 = fma(r82, r149, r0);
    r0 = fma(r19, r128, r0);
    r0 = fma(r5, r22, r0);
    r0 = fma(r19, r80, r0);
    r0 = fma(r19, r79, r0);
    r149 = r3 * r0;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r151,
                                             r149);
    r149 = r28 * r6;
    r149 = r149 * r6;
    r149 = r149 * r42;
    r149 = r149 * r42;
    r149 = r149 * r118;
    r149 = r149 * r48;
    r151 = r6 * r103;
    r135 = r30 * r37;
    r135 = r135 * r6;
    r94 = r28 * r7;
    r94 = r94 * r7;
    r94 = fma(r107, r94, r58 * r135);
    r135 = r28 * r107;
    r94 = fma(r109, r135, r94);
    r109 = r30 * r52;
    r109 = r109 * r7;
    r94 = fma(r58, r109, r94);
    r151 = r151 * r94;
    r151 = fma(r115, r151, r85 * r149);
    r149 = r63 * r94;
    r149 = r149 * r111;
    r149 = r149 * r97;
    r151 = fma(r106, r149, r151);
    r109 = r37 * r6;
    r109 = r109 * r116;
    r109 = r109 * r48;
    r151 = fma(r56, r109, r151);
    r135 = r7 * r7;
    r135 = r58 * r135;
    r135 = r135 * r48;
    r135 = r135 * r42;
    r135 = r135 * r83;
    r135 = r135 * r38;
    r135 = r135 * r94;
    r48 = r52 * r44;
    r48 = fma(r82, r48, r135);
    r58 = r20 * r7;
    r58 = r58 * r7;
    r58 = r58 * r94;
    r58 = r58 * r83;
    r58 = r58 * r114;
    r48 = fma(r56, r58, r48);
    r146 = r28 * r60;
    r48 = fma(r122, r146, r48);
    r151 = r151 + r48;
    r109 = r20 * r6;
    r109 = r109 * r94;
    r109 = fma(r115, r109, r28 * r125);
    r125 = r94 * r111;
    r125 = r125 * r97;
    r109 = fma(r106, r125, r109);
    r109 = fma(r37, r117, r109);
    r48 = r48 + r109;
    r64 = fma(r64, r48, r5 * r151);
    r151 = r20 * r28;
    r151 = r151 * r6;
    r151 = r151 * r77;
    r64 = fma(r41, r151, r64);
    r125 = r89 * r94;
    r125 = r125 * r38;
    r125 = r125 * r78;
    r64 = fma(r111, r125, r64);
    r106 = r4 * r37;
    r106 = r106 * r44;
    r64 = fma(r82, r106, r64);
    r97 = r4 * r94;
    r64 = fma(r129, r97, r64);
    r149 = r20 * r28;
    r149 = r149 * r6;
    r149 = r149 * r71;
    r149 = r149 * r77;
    r64 = fma(r41, r149, r64);
    r146 = r4 * r52;
    r64 = fma(r117, r146, r64);
    r117 = r4 * r94;
    r64 = fma(r131, r117, r64);
    r58 = r69 * r30;
    r58 = r58 * r61;
    r68 = fma(r68, r48, r48 * r58);
    r68 = fma(r48, r72, r68);
    r68 = fma(r48, r73, r68);
    r73 = r6 * r68;
    r64 = fma(r79, r73, r64);
    r72 = r89 * r71;
    r72 = r72 * r94;
    r72 = r72 * r38;
    r72 = r72 * r78;
    r64 = fma(r111, r72, r64);
    r64 = fma(r94, r126, r64);
    r64 = fma(r94, r124, r64);
    r64 = fma(r28, r137, r64);
    r64 = fma(r37, r79, r64);
    r64 = fma(r37, r80, r64);
    r72 = r2 * r64;
    r73 = r52 * r116;
    r73 = r73 * r56;
    r117 = r7 * r7;
    r117 = r117 * r103;
    r117 = r117 * r94;
    r117 = r117 * r83;
    r117 = r117 * r114;
    r117 = fma(r56, r117, r44 * r73);
    r73 = r28 * r118;
    r73 = r73 * r85;
    r73 = r73 * r60;
    r117 = fma(r121, r73, r117);
    r117 = fma(r63, r135, r117);
    r117 = r117 + r109;
    r48 = fma(r65, r48, r4 * r117);
    r65 = r42 * r71;
    r65 = r65 * r86;
    r65 = r65 * r94;
    r65 = r65 * r76;
    r65 = r65 * r83;
    r48 = fma(r44, r65, r48);
    r117 = r5 * r28;
    r117 = r117 * r6;
    r117 = r117 * r42;
    r117 = r117 * r42;
    r117 = r117 * r100;
    r117 = r117 * r85;
    r48 = fma(r44, r117, r48);
    r85 = r42 * r86;
    r85 = r85 * r94;
    r85 = r85 * r76;
    r85 = r85 * r83;
    r48 = fma(r44, r85, r48);
    r76 = r7 * r89;
    r76 = r76 * r71;
    r76 = r76 * r94;
    r76 = r76 * r83;
    r76 = r76 * r38;
    r48 = fma(r78, r76, r48);
    r100 = r5 * r37;
    r100 = r100 * r44;
    r48 = fma(r82, r100, r48);
    r82 = r5 * r94;
    r48 = fma(r129, r82, r48);
    r129 = r20 * r28;
    r129 = r129 * r7;
    r129 = r129 * r71;
    r129 = r129 * r77;
    r48 = fma(r41, r129, r48);
    r109 = r7 * r68;
    r48 = fma(r79, r109, r48);
    r73 = r5 * r94;
    r48 = fma(r131, r73, r48);
    r131 = r7 * r89;
    r131 = r131 * r94;
    r131 = r131 * r83;
    r131 = r131 * r38;
    r48 = fma(r78, r131, r48);
    r78 = r20 * r28;
    r78 = r78 * r7;
    r78 = r78 * r77;
    r48 = fma(r41, r78, r48);
    r48 = fma(r52, r128, r48);
    r48 = fma(r52, r79, r48);
    r48 = fma(r52, r80, r48);
    r78 = r3 * r48;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r72,
                                             r78);
    r78 = r3 * r20;
    r78 = r78 * r1;
    r78 = fma(r74, r104, r54 * r78);
    r72 = r3 * r20;
    r72 = r72 * r1;
    r72 = fma(r152, r104, r0 * r72);
    WriteSum2<double, double>((double*)inout_shared, r78, r72);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r3 * r20;
    r72 = r72 * r1;
    r104 = fma(r64, r104, r48 * r72);
    WriteSum1<double, double>((double*)inout_shared, r104);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r104 = r74 * r74;
    r72 = r54 * r54;
    r72 = fma(r105, r72, r133 * r104);
    r104 = r0 * r0;
    r1 = r152 * r152;
    r1 = fma(r133, r1, r105 * r104);
    WriteSum2<double, double>((double*)inout_shared, r72, r1);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r48 * r48;
    r72 = r64 * r64;
    r72 = fma(r133, r72, r105 * r1);
    WriteSum1<double, double>((double*)inout_shared, r72);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r74 * r152;
    r1 = r54 * r0;
    r1 = fma(r105, r1, r133 * r72);
    r72 = r54 * r48;
    r104 = r74 * r64;
    r104 = fma(r133, r104, r105 * r72);
    WriteSum2<double, double>((double*)inout_shared, r1, r104);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r104 = r0 * r48;
    r1 = r152 * r64;
    r1 = fma(r133, r1, r105 * r104);
    WriteSum1<double, double>((double*)inout_shared, r1);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
}

void ThinPrismFisheyeResJac(double* pose,
                            unsigned int pose_num_alloc,
                            SharedIndex* pose_indices,
                            double* sensor_from_rig,
                            unsigned int sensor_from_rig_num_alloc,
                            double* calib,
                            unsigned int calib_num_alloc,
                            SharedIndex* calib_indices,
                            double* point,
                            unsigned int point_num_alloc,
                            SharedIndex* point_indices,
                            double* pixel,
                            unsigned int pixel_num_alloc,
                            double* out_res,
                            unsigned int out_res_num_alloc,
                            double* out_pose_jac,
                            unsigned int out_pose_jac_num_alloc,
                            double* const out_pose_njtr,
                            unsigned int out_pose_njtr_num_alloc,
                            double* const out_pose_precond_diag,
                            unsigned int out_pose_precond_diag_num_alloc,
                            double* const out_pose_precond_tril,
                            unsigned int out_pose_precond_tril_num_alloc,
                            double* out_calib_jac,
                            unsigned int out_calib_jac_num_alloc,
                            double* const out_calib_njtr,
                            unsigned int out_calib_njtr_num_alloc,
                            double* const out_calib_precond_diag,
                            unsigned int out_calib_precond_diag_num_alloc,
                            double* const out_calib_precond_tril,
                            unsigned int out_calib_precond_tril_num_alloc,
                            double* out_point_jac,
                            unsigned int out_point_jac_num_alloc,
                            double* const out_point_njtr,
                            unsigned int out_point_njtr_num_alloc,
                            double* const out_point_precond_diag,
                            unsigned int out_point_precond_diag_num_alloc,
                            double* const out_point_precond_tril,
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