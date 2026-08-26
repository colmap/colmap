#include "kernel_thin_prism_fisheye_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFixedPointResJacKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* point,
        unsigned int point_num_alloc,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102, r103, r104,
      r105, r106, r107, r108, r109, r110, r111, r112, r113, r114, r115, r116,
      r117, r118, r119, r120, r121, r122, r123, r124, r125, r126, r127, r128,
      r129, r130, r131, r132, r133, r134, r135, r136, r137, r138, r139, r140;
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
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9);
  };
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
    r29 = fma(r8, r29, r6);
    r6 = 2.00000000000000000e+00;
    r30 = fma(r11, r16, r14 * r13);
    r31 = r15 * r12;
    r30 = fma(r20, r31, r30);
    r30 = fma(r10, r17, r30);
    r31 = r6 * r30;
    r31 = r31 * r25;
    r32 = r18 * r21;
    r33 = fma(r15, r17, r14 * r16);
    r33 = fma(r10, r12, r33);
    r33 = fma(r20, r33, r11 * r13);
    r32 = fma(r33, r32, r31);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r34);
    r35 = r6 * r18;
    r35 = r35 * r30;
    r36 = r6 * r25;
    r36 = fma(r33, r36, r35);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r37);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r38 = r16 * r12;
    r38 = r38 * r6;
    r39 = r17 * r13;
    r39 = fma(r6, r39, r38);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r40, r41);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r42 = r12 * r13;
    r43 = r16 * r17;
    r43 = r43 * r6;
    r42 = fma(r21, r42, r43);
    r44 = r17 * r17;
    r44 = r44 * r21;
    r45 = r22 + r44;
    r46 = r12 * r12;
    r46 = r46 * r21;
    r45 = r45 + r46;
    r29 = fma(r9, r32, r29);
    r29 = fma(r34, r36, r29);
    r29 = fma(r37, r39, r29);
    r29 = fma(r41, r42, r29);
    r29 = fma(r40, r45, r29);
    r36 = r29 * r29;
    r32 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r47);
    r48 = r21 * r25;
    r48 = fma(r33, r48, r35);
    r48 = fma(r8, r48, r47);
    r47 = r17 * r13;
    r47 = fma(r21, r47, r38);
    r44 = r22 + r44;
    r38 = r16 * r16;
    r38 = r38 * r21;
    r44 = r44 + r38;
    r35 = r17 * r12;
    r35 = r35 * r6;
    r49 = r16 * r13;
    r49 = fma(r6, r49, r35);
    r50 = r6 * r18;
    r50 = r50 * r25;
    r51 = r6 * r30;
    r51 = fma(r33, r51, r50);
    r52 = r30 * r30;
    r52 = r52 * r21;
    r28 = r52 + r28;
    r48 = fma(r40, r47, r48);
    r48 = fma(r37, r44, r48);
    r48 = fma(r41, r49, r48);
    r48 = fma(r9, r51, r48);
    r48 = fma(r34, r28, r48);
    r28 = copysign(1.0, r48);
    r28 = fma(r32, r28, r48);
    r48 = r28 * r28;
    r51 = 1.0 / r48;
    r53 = r6 * r18;
    r53 = fma(r33, r53, r31);
    r53 = fma(r8, r53, r7);
    r7 = r12 * r13;
    r7 = fma(r6, r7, r43);
    r46 = r22 + r46;
    r46 = r46 + r38;
    r38 = r16 * r13;
    r38 = fma(r21, r38, r35);
    r35 = r30 * r21;
    r35 = fma(r33, r35, r50);
    r19 = r22 + r19;
    r19 = r19 + r52;
    r53 = fma(r40, r7, r53);
    r53 = fma(r41, r46, r53);
    r53 = fma(r37, r38, r53);
    r53 = fma(r34, r35, r53);
    r53 = fma(r9, r19, r53);
    r19 = r53 * r53;
    r19 = fma(r51, r19, r51 * r36);
    r36 = sqrt(r19);
    r35 = atan(r36);
    r37 = r53 * r35;
    r41 = copysign(1.0, r36);
    r41 = fma(r32, r41, r36);
    r32 = r41 * r41;
    r36 = 1.0 / r32;
    r40 = r51 * r36;
    r52 = r37 * r40;
    r50 = r53 * r52;
    r43 = r35 * r50;
    r31 = r29 * r35;
    r54 = 3.00000000000000000e+00;
    r55 = r29 * r35;
    r31 = r31 * r54;
    r31 = r31 * r40;
    r31 = fma(r55, r31, r43);
  };
  LoadShared<2, double, double>(
      calib, 10 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r56, r57);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r58 = r29 * r35;
    r58 = r58 * r40;
    r58 = r58 * r55;
    r43 = r43 + r58;
    r59 = fma(r56, r43, r5 * r31);
  };
  LoadShared<2, double, double>(
      calib, 4 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r60, r61);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r62 = r43 * r43;
    r63 = fma(r61, r62, r60 * r43);
  };
  LoadShared<2, double, double>(
      calib, 8 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r64, r65);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r66 = r62 * r62;
    r67 = r43 * r62;
    r63 = fma(r65, r66, r63);
    r63 = fma(r64, r67, r63);
    r68 = 1.0 / r28;
    r69 = 1.0 / r41;
    r70 = r68 * r69;
    r71 = r63 * r70;
    r72 = r4 * r6;
    r72 = r72 * r55;
    r59 = fma(r52, r72, r59);
    r59 = fma(r55, r71, r59);
    r59 = fma(r70, r55, r59);
    r0 = fma(r2, r59, r0);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r72, r73);
    r0 = fma(r72, r20, r0);
    r72 = r35 * r54;
    r72 = fma(r50, r72, r58);
    r58 = fma(r57, r43, r4 * r72);
    r74 = r5 * r6;
    r74 = r74 * r55;
    r58 = fma(r52, r74, r58);
    r58 = fma(r37, r71, r58);
    r58 = fma(r37, r70, r58);
    r1 = fma(r3, r58, r1);
    r1 = fma(r73, r20, r1);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r0, r1);
    r73 = r6 * r29;
    r74 = r14 * r13;
    r75 = -5.00000000000000000e-01;
    r76 = r11 * r16;
    r76 = fma(r75, r76, r75 * r74);
    r74 = r10 * r17;
    r76 = fma(r75, r74, r76);
    r77 = r15 * r12;
    r78 = 5.00000000000000000e-01;
    r76 = fma(r78, r77, r76);
    r77 = r25 * r76;
    r74 = r10 * r13;
    r79 = r15 * r16;
    r79 = fma(r78, r79, r78 * r74);
    r74 = r14 * r17;
    r79 = fma(r75, r74, r79);
    r80 = r11 * r12;
    r79 = fma(r78, r80, r79);
    r80 = r33 * r79;
    r74 = fma(r6, r80, r6 * r77);
    r81 = r6 * r30;
    r82 = fma(r78, r27, r75 * r23);
    r82 = fma(r75, r24, r82);
    r82 = fma(r75, r26, r82);
    r83 = r6 * r18;
    r84 = r11 * r13;
    r85 = r14 * r16;
    r85 = fma(r75, r85, r78 * r84);
    r84 = r15 * r17;
    r85 = fma(r75, r84, r85);
    r86 = r10 * r12;
    r85 = fma(r75, r86, r85);
    r83 = r83 * r85;
    r81 = fma(r82, r81, r83);
    r74 = r74 + r81;
    r86 = r6 * r25;
    r86 = r86 * r85;
    r84 = r6 * r30;
    r84 = r84 * r79;
    r87 = r86 + r84;
    r88 = r18 * r21;
    r87 = fma(r76, r88, r87);
    r89 = r21 * r33;
    r87 = fma(r82, r89, r87);
    r87 = fma(r9, r87, r34 * r74);
    r74 = r25 * r79;
    r89 = -4.00000000000000000e+00;
    r74 = r74 * r89;
    r88 = r18 * r82;
    r90 = r89 * r88;
    r91 = r74 + r90;
    r87 = fma(r8, r91, r87);
    r73 = r73 * r87;
    r91 = r6 * r25;
    r91 = r91 * r82;
    r92 = r6 * r18;
    r92 = fma(r79, r92, r91);
    r79 = r6 * r30;
    r79 = r79 * r76;
    r93 = r6 * r33;
    r93 = r93 * r85;
    r94 = r79 + r93;
    r95 = r92 + r94;
    r80 = fma(r21, r80, r21 * r77);
    r80 = r80 + r81;
    r80 = fma(r8, r80, r9 * r95);
    r95 = r30 * r85;
    r95 = r95 * r89;
    r74 = r74 + r95;
    r80 = fma(r34, r74, r80);
    r74 = r29 * r29;
    r48 = r28 * r48;
    r96 = 1.0 / r48;
    r97 = r21 * r96;
    r74 = r74 * r97;
    r73 = fma(r80, r74, r51 * r73);
    r98 = r53 * r53;
    r98 = r98 * r80;
    r73 = fma(r97, r98, r73);
    r99 = r6 * r53;
    r100 = r30 * r21;
    r101 = r21 * r33;
    r101 = r101 * r85;
    r100 = fma(r76, r100, r101);
    r100 = r100 + r92;
    r90 = r95 + r90;
    r90 = fma(r9, r90, r34 * r100);
    r100 = r6 * r33;
    r100 = fma(r82, r100, r84);
    r84 = r6 * r18;
    r84 = fma(r76, r84, r86);
    r100 = r100 + r84;
    r90 = fma(r8, r100, r90);
    r99 = r99 * r90;
    r73 = fma(r51, r99, r73);
    r99 = r54 * r73;
    r98 = r40 * r55;
    r100 = rsqrt(r19);
    r19 = r22 + r19;
    r19 = 1.0 / r19;
    r86 = r100 * r19;
    r95 = r29 * r86;
    r99 = r99 * r98;
    r92 = -3.00000000000000000e+00;
    r92 = r35 * r92;
    r32 = r41 * r32;
    r102 = 1.0 / r32;
    r92 = r92 * r51;
    r92 = r92 * r100;
    r92 = r92 * r102;
    r103 = r73 * r92;
    r104 = r35 * r74;
    r103 = fma(r104, r103, r95 * r99);
    r99 = r35 * r87;
    r105 = 6.00000000000000000e+00;
    r99 = r99 * r105;
    r99 = r99 * r40;
    r103 = fma(r55, r99, r103);
    r106 = r35 * r80;
    r107 = -6.00000000000000000e+00;
    r106 = r106 * r107;
    r106 = r106 * r36;
    r106 = r106 * r96;
    r108 = r29 * r106;
    r103 = fma(r55, r108, r103);
    r109 = r53 * r35;
    r109 = r109 * r80;
    r109 = r109 * r36;
    r109 = r109 * r37;
    r110 = r86 * r50;
    r109 = fma(r73, r110, r97 * r109);
    r111 = r20 * r53;
    r111 = r111 * r35;
    r111 = r111 * r73;
    r111 = r111 * r51;
    r111 = r111 * r100;
    r111 = r111 * r102;
    r109 = fma(r37, r111, r109);
    r112 = r90 * r52;
    r113 = r6 * r35;
    r109 = fma(r113, r112, r109);
    r103 = r103 + r109;
    r108 = r73 * r98;
    r99 = r20 * r29;
    r99 = r99 * r29;
    r99 = r99 * r35;
    r99 = r99 * r35;
    r99 = r99 * r73;
    r99 = r99 * r51;
    r99 = r99 * r100;
    r99 = fma(r102, r99, r95 * r108);
    r108 = r113 * r98;
    r112 = r35 * r36;
    r112 = r112 * r97;
    r112 = r104 * r112;
    r99 = fma(r87, r108, r99);
    r99 = fma(r80, r112, r99);
    r109 = r109 + r99;
    r103 = fma(r56, r109, r5 * r103);
    r111 = r75 * r73;
    r111 = r111 * r36;
    r111 = r111 * r68;
    r111 = r111 * r100;
    r114 = r63 * r111;
    r115 = r4 * r87;
    r115 = r115 * r52;
    r103 = fma(r113, r115, r103);
    r116 = r29 * r35;
    r103 = fma(r111, r116, r103);
    r117 = r4 * r73;
    r118 = r6 * r52;
    r118 = r118 * r95;
    r103 = fma(r118, r117, r103);
    r119 = r4 * r21;
    r119 = r119 * r73;
    r119 = r119 * r51;
    r119 = r119 * r100;
    r119 = r119 * r102;
    r119 = r119 * r37;
    r103 = fma(r55, r119, r103);
    r120 = r78 * r73;
    r120 = r120 * r70;
    r103 = fma(r95, r120, r103);
    r121 = r4 * r108;
    r122 = r61 * r6;
    r122 = r122 * r43;
    r122 = fma(r109, r122, r60 * r109);
    r123 = 4.00000000000000000e+00;
    r65 = r65 * r123;
    r65 = r65 * r67;
    r64 = r64 * r54;
    r64 = r64 * r62;
    r122 = fma(r109, r65, r122);
    r122 = fma(r109, r64, r122);
    r124 = r122 * r70;
    r103 = fma(r55, r124, r103);
    r125 = r4 * r80;
    r126 = r89 * r36;
    r126 = r126 * r96;
    r126 = r126 * r37;
    r126 = r126 * r55;
    r103 = fma(r126, r125, r103);
    r127 = r20 * r29;
    r127 = r127 * r35;
    r127 = r127 * r80;
    r127 = r127 * r51;
    r103 = fma(r69, r127, r103);
    r128 = r35 * r87;
    r103 = fma(r70, r128, r103);
    r129 = r78 * r71;
    r130 = r95 * r129;
    r131 = r20 * r29;
    r131 = r131 * r35;
    r131 = r131 * r63;
    r131 = r131 * r80;
    r131 = r131 * r51;
    r103 = fma(r69, r131, r103);
    r132 = r35 * r87;
    r103 = fma(r71, r132, r103);
    r103 = fma(r114, r55, r103);
    r103 = fma(r90, r121, r103);
    r103 = fma(r73, r130, r103);
    r132 = r2 * r103;
    r131 = r53 * r37;
    r128 = r54 * r73;
    r128 = fma(r110, r128, r106 * r131);
    r131 = r53 * r37;
    r131 = r131 * r92;
    r127 = r35 * r90;
    r127 = r127 * r105;
    r128 = fma(r52, r127, r128);
    r128 = fma(r73, r131, r128);
    r128 = r128 + r99;
    r109 = fma(r57, r109, r4 * r128);
    r128 = r35 * r90;
    r109 = fma(r71, r128, r109);
    r99 = r122 * r37;
    r109 = fma(r70, r99, r109);
    r127 = r20 * r80;
    r127 = r127 * r51;
    r127 = r127 * r69;
    r109 = fma(r37, r127, r109);
    r125 = r5 * r87;
    r125 = r125 * r52;
    r109 = fma(r113, r125, r109);
    r124 = r5 * r73;
    r109 = fma(r118, r124, r109);
    r120 = r5 * r21;
    r120 = r120 * r73;
    r120 = r120 * r51;
    r120 = r120 * r100;
    r120 = r120 * r102;
    r120 = r120 * r37;
    r109 = fma(r55, r120, r109);
    r119 = r53 * r78;
    r119 = r119 * r73;
    r119 = r119 * r70;
    r109 = fma(r86, r119, r109);
    r117 = r53 * r73;
    r117 = r117 * r86;
    r109 = fma(r129, r117, r109);
    r116 = r5 * r90;
    r109 = fma(r108, r116, r109);
    r115 = r5 * r126;
    r133 = r35 * r90;
    r109 = fma(r70, r133, r109);
    r134 = r20 * r63;
    r134 = r134 * r80;
    r134 = r134 * r51;
    r134 = r134 * r69;
    r109 = fma(r37, r134, r109);
    r109 = fma(r37, r111, r109);
    r109 = fma(r80, r115, r109);
    r109 = fma(r37, r114, r109);
    r134 = r3 * r109;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             0 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r132,
                                             r134);
    r134 = r21 * r25;
    r134 = fma(r82, r134, r101);
    r132 = r6 * r18;
    r133 = r10 * r13;
    r114 = r15 * r16;
    r114 = fma(r75, r114, r75 * r133);
    r133 = r14 * r17;
    r114 = fma(r78, r133, r114);
    r116 = r11 * r12;
    r114 = fma(r75, r116, r114);
    r132 = r132 * r114;
    r116 = r6 * r30;
    r133 = r14 * r13;
    r117 = r11 * r16;
    r117 = fma(r78, r117, r78 * r133);
    r133 = r10 * r17;
    r117 = fma(r78, r133, r117);
    r119 = r15 * r12;
    r117 = fma(r75, r119, r117);
    r116 = fma(r117, r116, r132);
    r134 = r134 + r116;
    r119 = r6 * r25;
    r119 = r119 * r117;
    r133 = r6 * r33;
    r133 = fma(r114, r133, r119);
    r133 = r133 + r81;
    r133 = fma(r9, r133, r8 * r134);
    r134 = r25 * r85;
    r134 = r134 * r89;
    r81 = r30 * r114;
    r111 = r89 * r81;
    r120 = r134 + r111;
    r133 = fma(r34, r120, r133);
    r93 = r91 + r93;
    r93 = r93 + r116;
    r116 = r18 * r89;
    r116 = r116 * r117;
    r134 = r134 + r116;
    r134 = fma(r8, r134, r34 * r93);
    r93 = r21 * r33;
    r93 = fma(r21, r88, r117 * r93);
    r91 = r6 * r30;
    r91 = r91 * r85;
    r120 = r6 * r25;
    r120 = fma(r114, r120, r91);
    r93 = r93 + r120;
    r134 = fma(r9, r93, r134);
    r93 = fma(r134, r108, r133 * r112);
    r124 = r20 * r29;
    r125 = r6 * r29;
    r125 = r125 * r134;
    r127 = r6 * r53;
    r119 = r83 + r119;
    r83 = r30 * r21;
    r119 = fma(r82, r83, r119);
    r82 = r21 * r33;
    r119 = fma(r114, r82, r119);
    r82 = r6 * r33;
    r88 = fma(r6, r88, r117 * r82);
    r88 = r88 + r120;
    r88 = fma(r8, r88, r34 * r119);
    r111 = r116 + r111;
    r88 = fma(r9, r111, r88);
    r127 = r127 * r88;
    r127 = fma(r51, r127, r51 * r125);
    r125 = r53 * r53;
    r125 = r125 * r133;
    r127 = fma(r97, r125, r127);
    r127 = fma(r133, r74, r127);
    r124 = r124 * r29;
    r124 = r124 * r35;
    r124 = r124 * r35;
    r124 = r124 * r127;
    r124 = r124 * r51;
    r124 = r124 * r100;
    r93 = fma(r102, r124, r93);
    r125 = r127 * r98;
    r93 = fma(r95, r125, r93);
    r125 = r88 * r52;
    r124 = r20 * r53;
    r124 = r124 * r35;
    r124 = r124 * r127;
    r124 = r124 * r51;
    r124 = r124 * r100;
    r124 = r124 * r102;
    r124 = fma(r37, r124, r113 * r125);
    r125 = r53 * r35;
    r125 = r125 * r133;
    r125 = r125 * r36;
    r125 = r125 * r37;
    r124 = fma(r97, r125, r124);
    r124 = fma(r127, r110, r124);
    r125 = r93 + r124;
    r111 = r29 * r29;
    r111 = r111 * r35;
    r111 = r111 * r35;
    r111 = r111 * r107;
    r111 = r111 * r133;
    r111 = r111 * r36;
    r116 = r35 * r105;
    r116 = r116 * r134;
    r116 = r116 * r40;
    r116 = fma(r55, r116, r96 * r111);
    r111 = r127 * r92;
    r116 = fma(r104, r111, r116);
    r119 = r54 * r127;
    r119 = r119 * r98;
    r116 = fma(r95, r119, r116);
    r116 = r116 + r124;
    r116 = fma(r5, r116, r56 * r125);
    r124 = r20 * r29;
    r124 = r124 * r35;
    r124 = r124 * r133;
    r124 = r124 * r51;
    r116 = fma(r69, r124, r116);
    r119 = r78 * r127;
    r119 = r119 * r70;
    r116 = fma(r95, r119, r116);
    r111 = r35 * r134;
    r116 = fma(r71, r111, r116);
    r82 = r61 * r6;
    r82 = r82 * r43;
    r82 = fma(r125, r82, r60 * r125);
    r82 = fma(r125, r65, r82);
    r82 = fma(r125, r64, r82);
    r117 = r82 * r70;
    r116 = fma(r55, r117, r116);
    r83 = r127 * r118;
    r99 = r4 * r133;
    r116 = fma(r126, r99, r116);
    r128 = r4 * r134;
    r128 = r128 * r52;
    r116 = fma(r113, r128, r116);
    r135 = r4 * r21;
    r135 = r135 * r127;
    r135 = r135 * r51;
    r135 = r135 * r100;
    r135 = r135 * r102;
    r135 = r135 * r37;
    r116 = fma(r55, r135, r116);
    r136 = r20 * r29;
    r136 = r136 * r35;
    r136 = r136 * r63;
    r136 = r136 * r133;
    r136 = r136 * r51;
    r116 = fma(r69, r136, r116);
    r137 = r35 * r134;
    r116 = fma(r70, r137, r116);
    r138 = r29 * r35;
    r138 = r138 * r63;
    r138 = r138 * r75;
    r138 = r138 * r127;
    r138 = r138 * r36;
    r138 = r138 * r68;
    r116 = fma(r100, r138, r116);
    r139 = r29 * r35;
    r139 = r139 * r75;
    r139 = r139 * r127;
    r139 = r139 * r36;
    r139 = r139 * r68;
    r116 = fma(r100, r139, r116);
    r116 = fma(r88, r121, r116);
    r116 = fma(r4, r83, r116);
    r116 = fma(r127, r130, r116);
    r139 = r2 * r116;
    r138 = r35 * r105;
    r138 = r138 * r88;
    r138 = fma(r127, r131, r52 * r138);
    r137 = r53 * r35;
    r137 = r137 * r107;
    r137 = r137 * r133;
    r137 = r137 * r36;
    r137 = r137 * r96;
    r138 = fma(r37, r137, r138);
    r136 = r54 * r127;
    r138 = fma(r110, r136, r138);
    r138 = r138 + r93;
    r138 = fma(r4, r138, r57 * r125);
    r125 = r20 * r63;
    r125 = r125 * r133;
    r125 = r125 * r51;
    r125 = r125 * r69;
    r138 = fma(r37, r125, r138);
    r93 = r63 * r75;
    r93 = r93 * r127;
    r93 = r93 * r36;
    r93 = r93 * r68;
    r93 = r93 * r100;
    r138 = fma(r37, r93, r138);
    r136 = r75 * r127;
    r136 = r136 * r36;
    r136 = r136 * r68;
    r136 = r136 * r100;
    r138 = fma(r37, r136, r138);
    r137 = r5 * r88;
    r138 = fma(r108, r137, r138);
    r135 = r53 * r78;
    r135 = r135 * r127;
    r135 = r135 * r70;
    r138 = fma(r86, r135, r138);
    r128 = r5 * r134;
    r128 = r128 * r52;
    r138 = fma(r113, r128, r138);
    r99 = r5 * r21;
    r99 = r99 * r127;
    r99 = r99 * r51;
    r99 = r99 * r100;
    r99 = r99 * r102;
    r99 = r99 * r37;
    r138 = fma(r55, r99, r138);
    r117 = r82 * r37;
    r138 = fma(r70, r117, r138);
    r111 = r20 * r133;
    r111 = r111 * r51;
    r111 = r111 * r69;
    r138 = fma(r37, r111, r138);
    r119 = r53 * r127;
    r119 = r119 * r86;
    r138 = fma(r129, r119, r138);
    r124 = r35 * r88;
    r138 = fma(r70, r124, r138);
    r140 = r35 * r88;
    r138 = fma(r71, r140, r138);
    r138 = fma(r5, r83, r138);
    r138 = fma(r133, r115, r138);
    r140 = r3 * r138;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             2 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r139,
                                             r140);
    r140 = r53 * r35;
    r139 = r30 * r89;
    r27 = fma(r75, r27, r78 * r23);
    r27 = fma(r78, r24, r27);
    r27 = fma(r78, r26, r27);
    r139 = r139 * r27;
    r77 = r89 * r77;
    r26 = r139 + r77;
    r24 = r6 * r18;
    r24 = r24 * r27;
    r91 = r91 + r24;
    r23 = r21 * r25;
    r91 = fma(r114, r23, r91);
    r124 = r21 * r33;
    r91 = fma(r76, r124, r91);
    r91 = fma(r8, r91, r34 * r26);
    r26 = r6 * r33;
    r26 = fma(r6, r81, r27 * r26);
    r26 = r26 + r84;
    r91 = fma(r9, r26, r91);
    r140 = r140 * r91;
    r140 = r140 * r36;
    r140 = r140 * r37;
    r26 = r6 * r53;
    r124 = r6 * r25;
    r124 = r124 * r27;
    r132 = r132 + r124;
    r132 = r132 + r94;
    r94 = r21 * r33;
    r81 = fma(r21, r81, r27 * r94);
    r81 = r81 + r84;
    r81 = fma(r34, r81, r8 * r132);
    r85 = r18 * r85;
    r85 = r85 * r89;
    r139 = r139 + r85;
    r81 = fma(r9, r139, r81);
    r26 = r26 * r81;
    r26 = fma(r91, r74, r51 * r26);
    r139 = r6 * r29;
    r101 = r79 + r101;
    r79 = r18 * r21;
    r101 = fma(r114, r79, r101);
    r101 = r101 + r124;
    r77 = r85 + r77;
    r77 = fma(r8, r77, r9 * r101);
    r8 = r6 * r33;
    r8 = fma(r76, r8, r24);
    r8 = r8 + r120;
    r77 = fma(r34, r8, r77);
    r139 = r139 * r77;
    r26 = fma(r51, r139, r26);
    r8 = r53 * r53;
    r8 = r8 * r91;
    r26 = fma(r97, r8, r26);
    r140 = fma(r26, r110, r97 * r140);
    r8 = r81 * r52;
    r140 = fma(r113, r8, r140);
    r139 = r20 * r53;
    r139 = r139 * r35;
    r139 = r139 * r26;
    r139 = r139 * r51;
    r139 = r139 * r100;
    r139 = r139 * r102;
    r140 = fma(r37, r139, r140);
    r139 = r20 * r29;
    r139 = r139 * r29;
    r139 = r139 * r35;
    r139 = r139 * r35;
    r139 = r139 * r26;
    r139 = r139 * r51;
    r139 = r139 * r100;
    r139 = fma(r77, r108, r102 * r139);
    r8 = r26 * r98;
    r139 = fma(r95, r8, r139);
    r139 = fma(r91, r112, r139);
    r8 = r140 + r139;
    r34 = r26 * r92;
    r120 = r35 * r105;
    r120 = r120 * r77;
    r120 = r120 * r40;
    r120 = fma(r55, r120, r104 * r34);
    r34 = r29 * r29;
    r34 = r34 * r35;
    r34 = r34 * r35;
    r34 = r34 * r107;
    r34 = r34 * r91;
    r34 = r34 * r36;
    r120 = fma(r96, r34, r120);
    r24 = r54 * r26;
    r24 = r24 * r98;
    r120 = fma(r95, r24, r120);
    r120 = r120 + r140;
    r120 = fma(r5, r120, r56 * r8);
    r140 = r61 * r6;
    r140 = r140 * r43;
    r140 = fma(r8, r140, r60 * r8);
    r140 = fma(r8, r64, r140);
    r140 = fma(r8, r65, r140);
    r24 = r140 * r70;
    r120 = fma(r55, r24, r120);
    r34 = r78 * r26;
    r34 = r34 * r70;
    r120 = fma(r95, r34, r120);
    r76 = r35 * r77;
    r120 = fma(r71, r76, r120);
    r101 = r4 * r26;
    r120 = fma(r118, r101, r120);
    r9 = r29 * r35;
    r9 = r9 * r63;
    r9 = r9 * r75;
    r9 = r9 * r26;
    r9 = r9 * r36;
    r9 = r9 * r68;
    r120 = fma(r100, r9, r120);
    r85 = r20 * r29;
    r85 = r85 * r35;
    r85 = r85 * r91;
    r85 = r85 * r51;
    r120 = fma(r69, r85, r120);
    r79 = r4 * r77;
    r79 = r79 * r52;
    r120 = fma(r113, r79, r120);
    r124 = r4 * r91;
    r120 = fma(r126, r124, r120);
    r114 = r35 * r77;
    r120 = fma(r70, r114, r120);
    r89 = r29 * r35;
    r89 = r89 * r75;
    r89 = r89 * r26;
    r89 = r89 * r36;
    r89 = r89 * r68;
    r120 = fma(r100, r89, r120);
    r132 = r20 * r29;
    r132 = r132 * r35;
    r132 = r132 * r63;
    r132 = r132 * r91;
    r132 = r132 * r51;
    r120 = fma(r69, r132, r120);
    r84 = r4 * r21;
    r84 = r84 * r26;
    r84 = r84 * r51;
    r84 = r84 * r100;
    r84 = r84 * r102;
    r84 = r84 * r37;
    r120 = fma(r55, r84, r120);
    r120 = fma(r81, r121, r120);
    r120 = fma(r26, r130, r120);
    r84 = r2 * r120;
    r132 = r53 * r35;
    r132 = r132 * r107;
    r132 = r132 * r91;
    r132 = r132 * r36;
    r132 = r132 * r96;
    r89 = r54 * r26;
    r89 = fma(r110, r89, r37 * r132);
    r132 = r35 * r105;
    r132 = r132 * r81;
    r89 = fma(r52, r132, r89);
    r89 = fma(r26, r131, r89);
    r89 = r89 + r139;
    r89 = fma(r4, r89, r57 * r8);
    r8 = r20 * r91;
    r8 = r8 * r51;
    r8 = r8 * r69;
    r89 = fma(r37, r8, r89);
    r139 = r5 * r81;
    r89 = fma(r108, r139, r89);
    r132 = r20 * r63;
    r132 = r132 * r91;
    r132 = r132 * r51;
    r132 = r132 * r69;
    r89 = fma(r37, r132, r89);
    r114 = r53 * r78;
    r114 = r114 * r26;
    r114 = r114 * r70;
    r89 = fma(r86, r114, r89);
    r124 = r35 * r81;
    r89 = fma(r70, r124, r89);
    r79 = r5 * r26;
    r89 = fma(r118, r79, r89);
    r85 = r63 * r75;
    r85 = r85 * r26;
    r85 = r85 * r36;
    r85 = r85 * r68;
    r85 = r85 * r100;
    r89 = fma(r37, r85, r89);
    r9 = r140 * r37;
    r89 = fma(r70, r9, r89);
    r101 = r75 * r26;
    r101 = r101 * r36;
    r101 = r101 * r68;
    r101 = r101 * r100;
    r89 = fma(r37, r101, r89);
    r76 = r53 * r26;
    r76 = r76 * r86;
    r89 = fma(r129, r76, r89);
    r34 = r5 * r77;
    r34 = r34 * r52;
    r89 = fma(r113, r34, r89);
    r24 = r35 * r81;
    r89 = fma(r71, r24, r89);
    r94 = r5 * r21;
    r94 = r94 * r26;
    r94 = r94 * r51;
    r94 = r94 * r100;
    r94 = r94 * r102;
    r94 = r94 * r37;
    r89 = fma(r55, r94, r89);
    r89 = fma(r91, r115, r89);
    r94 = r3 * r89;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r84, r94);
    r94 = r45 * r35;
    r94 = r94 * r105;
    r94 = r94 * r40;
    r84 = r6 * r7;
    r84 = r84 * r53;
    r24 = r47 * r53;
    r24 = r24 * r53;
    r24 = fma(r97, r24, r51 * r84);
    r84 = r6 * r45;
    r84 = r84 * r29;
    r24 = fma(r51, r84, r24);
    r24 = fma(r47, r74, r24);
    r84 = r24 * r92;
    r84 = fma(r104, r84, r55 * r94);
    r94 = r54 * r24;
    r94 = r94 * r98;
    r84 = fma(r95, r94, r84);
    r34 = r47 * r29;
    r34 = r34 * r29;
    r34 = r34 * r35;
    r34 = r34 * r35;
    r34 = r34 * r107;
    r34 = r34 * r36;
    r84 = fma(r96, r34, r84);
    r76 = r7 * r52;
    r101 = r20 * r53;
    r101 = r101 * r35;
    r101 = r101 * r24;
    r101 = r101 * r51;
    r101 = r101 * r100;
    r101 = r101 * r102;
    r101 = fma(r37, r101, r113 * r76);
    r76 = r47 * r53;
    r76 = r76 * r35;
    r76 = r76 * r36;
    r76 = r76 * r37;
    r101 = fma(r97, r76, r101);
    r101 = fma(r24, r110, r101);
    r84 = r84 + r101;
    r34 = r20 * r29;
    r34 = r34 * r29;
    r34 = r34 * r35;
    r34 = r34 * r35;
    r34 = r34 * r24;
    r34 = r34 * r51;
    r34 = r34 * r100;
    r34 = fma(r102, r34, r45 * r108);
    r94 = r24 * r98;
    r34 = fma(r95, r94, r34);
    r34 = fma(r47, r112, r34);
    r101 = r101 + r34;
    r84 = fma(r56, r101, r5 * r84);
    r94 = r29 * r35;
    r94 = r94 * r63;
    r94 = r94 * r75;
    r94 = r94 * r24;
    r94 = r94 * r36;
    r94 = r94 * r68;
    r84 = fma(r100, r94, r84);
    r76 = r78 * r24;
    r76 = r76 * r70;
    r84 = fma(r95, r76, r84);
    r9 = r29 * r35;
    r9 = r9 * r75;
    r9 = r9 * r24;
    r9 = r9 * r36;
    r9 = r9 * r68;
    r84 = fma(r100, r9, r84);
    r85 = r20 * r47;
    r85 = r85 * r29;
    r85 = r85 * r35;
    r85 = r85 * r63;
    r85 = r85 * r51;
    r84 = fma(r69, r85, r84);
    r79 = r61 * r6;
    r79 = r79 * r43;
    r79 = fma(r101, r79, r60 * r101);
    r79 = fma(r101, r64, r79);
    r79 = fma(r101, r65, r79);
    r124 = r79 * r70;
    r84 = fma(r55, r124, r84);
    r114 = r20 * r47;
    r114 = r114 * r29;
    r114 = r114 * r35;
    r114 = r114 * r51;
    r84 = fma(r69, r114, r84);
    r132 = r4 * r24;
    r84 = fma(r118, r132, r84);
    r139 = r45 * r35;
    r84 = fma(r71, r139, r84);
    r8 = r4 * r45;
    r8 = r8 * r52;
    r84 = fma(r113, r8, r84);
    r27 = r45 * r35;
    r84 = fma(r70, r27, r84);
    r23 = r4 * r21;
    r23 = r23 * r24;
    r23 = r23 * r51;
    r23 = r23 * r100;
    r23 = r23 * r102;
    r23 = r23 * r37;
    r84 = fma(r55, r23, r84);
    r119 = r4 * r47;
    r84 = fma(r126, r119, r84);
    r84 = fma(r24, r130, r84);
    r84 = fma(r7, r121, r84);
    r119 = r2 * r84;
    r23 = r7 * r35;
    r23 = r23 * r105;
    r23 = fma(r24, r131, r52 * r23);
    r27 = r47 * r53;
    r27 = r27 * r35;
    r27 = r27 * r107;
    r27 = r27 * r36;
    r27 = r27 * r96;
    r23 = fma(r37, r27, r23);
    r8 = r54 * r24;
    r23 = fma(r110, r8, r23);
    r23 = r23 + r34;
    r101 = fma(r57, r101, r4 * r23);
    r23 = r7 * r35;
    r101 = fma(r71, r23, r101);
    r34 = r53 * r78;
    r34 = r34 * r24;
    r34 = r34 * r70;
    r101 = fma(r86, r34, r101);
    r8 = r7 * r35;
    r101 = fma(r70, r8, r101);
    r27 = r20 * r47;
    r27 = r27 * r51;
    r27 = r27 * r69;
    r101 = fma(r37, r27, r101);
    r139 = r53 * r24;
    r139 = r139 * r86;
    r101 = fma(r129, r139, r101);
    r132 = r20 * r47;
    r132 = r132 * r63;
    r132 = r132 * r51;
    r132 = r132 * r69;
    r101 = fma(r37, r132, r101);
    r114 = r5 * r7;
    r101 = fma(r108, r114, r101);
    r124 = r63 * r75;
    r124 = r124 * r24;
    r124 = r124 * r36;
    r124 = r124 * r68;
    r124 = r124 * r100;
    r101 = fma(r37, r124, r101);
    r85 = r5 * r24;
    r101 = fma(r118, r85, r101);
    r9 = r5 * r45;
    r9 = r9 * r52;
    r101 = fma(r113, r9, r101);
    r76 = r75 * r24;
    r76 = r76 * r36;
    r76 = r76 * r68;
    r76 = r76 * r100;
    r101 = fma(r37, r76, r101);
    r94 = r79 * r37;
    r101 = fma(r70, r94, r101);
    r111 = r5 * r21;
    r111 = r111 * r24;
    r111 = r111 * r51;
    r111 = r111 * r100;
    r111 = r111 * r102;
    r111 = r111 * r37;
    r101 = fma(r55, r111, r101);
    r101 = fma(r47, r115, r101);
    r111 = r3 * r101;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             6 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r119,
                                             r111);
    r111 = r53 * r53;
    r111 = r51 * r111;
    r119 = r49 * r53;
    r119 = r119 * r53;
    r94 = r6 * r46;
    r94 = r94 * r53;
    r94 = fma(r51, r94, r97 * r119);
    r119 = r6 * r42;
    r119 = r119 * r29;
    r94 = fma(r51, r119, r94);
    r94 = fma(r49, r74, r94);
    r111 = r111 * r36;
    r111 = r111 * r35;
    r111 = r111 * r100;
    r111 = r111 * r19;
    r111 = r111 * r94;
    r19 = r49 * r53;
    r19 = r19 * r35;
    r19 = r19 * r36;
    r19 = r19 * r37;
    r19 = fma(r97, r19, r111);
    r119 = r46 * r52;
    r19 = fma(r113, r119, r19);
    r76 = r20 * r53;
    r76 = r76 * r35;
    r76 = r76 * r94;
    r76 = r76 * r51;
    r76 = r76 * r100;
    r76 = r76 * r102;
    r19 = fma(r37, r76, r19);
    r76 = r94 * r98;
    r76 = fma(r49, r112, r95 * r76);
    r119 = r20 * r29;
    r119 = r119 * r29;
    r119 = r119 * r35;
    r119 = r119 * r35;
    r119 = r119 * r94;
    r119 = r119 * r51;
    r119 = r119 * r100;
    r76 = fma(r102, r119, r76);
    r76 = fma(r42, r108, r76);
    r119 = r19 + r76;
    r9 = r54 * r94;
    r9 = r9 * r98;
    r85 = r49 * r29;
    r85 = r85 * r29;
    r85 = r85 * r35;
    r85 = r85 * r35;
    r85 = r85 * r107;
    r85 = r85 * r36;
    r85 = fma(r96, r85, r95 * r9);
    r9 = r94 * r92;
    r85 = fma(r104, r9, r85);
    r124 = r42 * r35;
    r124 = r124 * r105;
    r124 = r124 * r40;
    r85 = fma(r55, r124, r85);
    r85 = r85 + r19;
    r85 = fma(r5, r85, r56 * r119);
    r19 = r4 * r94;
    r85 = fma(r118, r19, r85);
    r124 = r42 * r35;
    r85 = fma(r70, r124, r85);
    r9 = r20 * r49;
    r9 = r9 * r29;
    r9 = r9 * r35;
    r9 = r9 * r51;
    r85 = fma(r69, r9, r85);
    r114 = r4 * r42;
    r114 = r114 * r52;
    r85 = fma(r113, r114, r85);
    r132 = r29 * r35;
    r132 = r132 * r63;
    r132 = r132 * r75;
    r132 = r132 * r94;
    r132 = r132 * r36;
    r132 = r132 * r68;
    r85 = fma(r100, r132, r85);
    r139 = r20 * r49;
    r139 = r139 * r29;
    r139 = r139 * r35;
    r139 = r139 * r63;
    r139 = r139 * r51;
    r85 = fma(r69, r139, r85);
    r27 = r78 * r94;
    r27 = r27 * r70;
    r85 = fma(r95, r27, r85);
    r8 = r42 * r35;
    r85 = fma(r71, r8, r85);
    r34 = r61 * r6;
    r34 = r34 * r43;
    r34 = fma(r60, r119, r119 * r34);
    r34 = fma(r119, r65, r34);
    r34 = fma(r119, r64, r34);
    r23 = r34 * r70;
    r85 = fma(r55, r23, r85);
    r117 = r29 * r35;
    r117 = r117 * r75;
    r117 = r117 * r94;
    r117 = r117 * r36;
    r117 = r117 * r68;
    r85 = fma(r100, r117, r85);
    r99 = r4 * r21;
    r99 = r99 * r94;
    r99 = r99 * r51;
    r99 = r99 * r100;
    r99 = r99 * r102;
    r99 = r99 * r37;
    r85 = fma(r55, r99, r85);
    r128 = r4 * r49;
    r85 = fma(r126, r128, r85);
    r85 = fma(r94, r130, r85);
    r85 = fma(r46, r121, r85);
    r128 = r2 * r85;
    r99 = r49 * r53;
    r99 = r99 * r35;
    r99 = r99 * r107;
    r99 = r99 * r36;
    r99 = r99 * r96;
    r117 = r46 * r35;
    r117 = r117 * r105;
    r117 = fma(r52, r117, r37 * r99);
    r117 = fma(r54, r111, r117);
    r117 = fma(r94, r131, r117);
    r117 = r117 + r76;
    r117 = fma(r4, r117, r57 * r119);
    r119 = r20 * r49;
    r119 = r119 * r51;
    r119 = r119 * r69;
    r117 = fma(r37, r119, r117);
    r76 = r20 * r49;
    r76 = r76 * r63;
    r76 = r76 * r51;
    r76 = r76 * r69;
    r117 = fma(r37, r76, r117);
    r111 = r5 * r94;
    r117 = fma(r118, r111, r117);
    r99 = r53 * r94;
    r99 = r99 * r86;
    r117 = fma(r129, r99, r117);
    r23 = r53 * r78;
    r23 = r23 * r94;
    r23 = r23 * r70;
    r117 = fma(r86, r23, r117);
    r8 = r5 * r42;
    r8 = r8 * r52;
    r117 = fma(r113, r8, r117);
    r27 = r46 * r35;
    r117 = fma(r71, r27, r117);
    r139 = r34 * r37;
    r117 = fma(r70, r139, r117);
    r132 = r75 * r94;
    r132 = r132 * r36;
    r132 = r132 * r68;
    r132 = r132 * r100;
    r117 = fma(r37, r132, r117);
    r114 = r5 * r46;
    r117 = fma(r108, r114, r117);
    r9 = r5 * r21;
    r9 = r9 * r94;
    r9 = r9 * r51;
    r9 = r9 * r100;
    r9 = r9 * r102;
    r9 = r9 * r37;
    r117 = fma(r55, r9, r117);
    r124 = r46 * r35;
    r117 = fma(r70, r124, r117);
    r19 = r63 * r75;
    r19 = r19 * r94;
    r19 = r19 * r36;
    r19 = r19 * r68;
    r19 = r19 * r100;
    r117 = fma(r37, r19, r117);
    r117 = fma(r49, r115, r117);
    r19 = r3 * r117;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r128, r19);
    r19 = r20 * r29;
    r128 = r6 * r38;
    r128 = r128 * r53;
    r124 = r44 * r53;
    r124 = r124 * r53;
    r124 = fma(r97, r124, r51 * r128);
    r128 = r6 * r39;
    r128 = r128 * r29;
    r124 = fma(r51, r128, r124);
    r124 = fma(r44, r74, r124);
    r19 = r19 * r29;
    r19 = r19 * r35;
    r19 = r19 * r35;
    r19 = r19 * r124;
    r19 = r19 * r51;
    r19 = r19 * r100;
    r74 = r124 * r98;
    r74 = fma(r95, r74, r102 * r19);
    r74 = fma(r39, r108, r74);
    r74 = fma(r44, r112, r74);
    r112 = r44 * r53;
    r112 = r112 * r35;
    r112 = r112 * r36;
    r112 = r112 * r37;
    r19 = r38 * r52;
    r19 = fma(r113, r19, r97 * r112);
    r112 = r20 * r53;
    r112 = r112 * r35;
    r112 = r112 * r124;
    r112 = r112 * r51;
    r112 = r112 * r100;
    r112 = r112 * r102;
    r19 = fma(r37, r112, r19);
    r19 = fma(r124, r110, r19);
    r112 = r74 + r19;
    r97 = r124 * r92;
    r128 = r54 * r124;
    r128 = r128 * r98;
    r128 = fma(r95, r128, r104 * r97);
    r97 = r39 * r35;
    r97 = r97 * r105;
    r97 = r97 * r40;
    r128 = fma(r55, r97, r128);
    r40 = r44 * r29;
    r40 = r40 * r29;
    r40 = r40 * r35;
    r40 = r40 * r35;
    r40 = r40 * r107;
    r40 = r40 * r36;
    r128 = fma(r96, r40, r128);
    r128 = r128 + r19;
    r128 = fma(r5, r128, r56 * r112);
    r56 = r61 * r6;
    r56 = r56 * r43;
    r60 = fma(r60, r112, r112 * r56);
    r60 = fma(r112, r64, r60);
    r60 = fma(r112, r65, r60);
    r65 = r60 * r70;
    r128 = fma(r55, r65, r128);
    r64 = r20 * r44;
    r64 = r64 * r29;
    r64 = r64 * r35;
    r64 = r64 * r63;
    r64 = r64 * r51;
    r128 = fma(r69, r64, r128);
    r56 = r29 * r35;
    r56 = r56 * r63;
    r56 = r56 * r75;
    r56 = r56 * r124;
    r56 = r56 * r36;
    r56 = r56 * r68;
    r128 = fma(r100, r56, r128);
    r19 = r4 * r21;
    r19 = r19 * r124;
    r19 = r19 * r51;
    r19 = r19 * r100;
    r19 = r19 * r102;
    r19 = r19 * r37;
    r128 = fma(r55, r19, r128);
    r40 = r4 * r124;
    r128 = fma(r118, r40, r128);
    r97 = r4 * r44;
    r128 = fma(r126, r97, r128);
    r126 = r78 * r124;
    r126 = r126 * r70;
    r128 = fma(r95, r126, r128);
    r95 = r39 * r35;
    r128 = fma(r71, r95, r128);
    r104 = r4 * r39;
    r104 = r104 * r52;
    r128 = fma(r113, r104, r128);
    r9 = r20 * r44;
    r9 = r9 * r29;
    r9 = r9 * r35;
    r9 = r9 * r51;
    r128 = fma(r69, r9, r128);
    r114 = r39 * r35;
    r128 = fma(r70, r114, r128);
    r132 = r29 * r35;
    r132 = r132 * r75;
    r132 = r132 * r124;
    r132 = r132 * r36;
    r132 = r132 * r68;
    r128 = fma(r100, r132, r128);
    r128 = fma(r38, r121, r128);
    r128 = fma(r124, r130, r128);
    r132 = r2 * r128;
    r130 = r44 * r53;
    r130 = r130 * r35;
    r130 = r130 * r107;
    r130 = r130 * r36;
    r130 = r130 * r96;
    r107 = r38 * r35;
    r107 = r107 * r105;
    r107 = fma(r52, r107, r37 * r130);
    r130 = r54 * r124;
    r107 = fma(r110, r130, r107);
    r107 = fma(r124, r131, r107);
    r107 = r107 + r74;
    r107 = fma(r4, r107, r57 * r112);
    r112 = r75 * r124;
    r112 = r112 * r36;
    r112 = r112 * r68;
    r112 = r112 * r100;
    r107 = fma(r37, r112, r107);
    r57 = r20 * r44;
    r57 = r57 * r51;
    r57 = r57 * r69;
    r107 = fma(r37, r57, r107);
    r74 = r5 * r38;
    r107 = fma(r108, r74, r107);
    r108 = r38 * r35;
    r107 = fma(r70, r108, r107);
    r131 = r63 * r75;
    r131 = r131 * r124;
    r131 = r131 * r36;
    r131 = r131 * r68;
    r131 = r131 * r100;
    r107 = fma(r37, r131, r107);
    r68 = r5 * r21;
    r68 = r68 * r124;
    r68 = r68 * r51;
    r68 = r68 * r100;
    r68 = r68 * r102;
    r68 = r68 * r37;
    r107 = fma(r55, r68, r107);
    r100 = r5 * r124;
    r107 = fma(r118, r100, r107);
    r118 = r53 * r124;
    r118 = r118 * r86;
    r107 = fma(r129, r118, r107);
    r129 = r38 * r35;
    r107 = fma(r71, r129, r107);
    r71 = r5 * r39;
    r71 = r71 * r52;
    r107 = fma(r113, r71, r107);
    r36 = r53 * r78;
    r36 = r36 * r124;
    r36 = r36 * r70;
    r107 = fma(r86, r36, r107);
    r86 = r60 * r37;
    r107 = fma(r70, r86, r107);
    r130 = r20 * r44;
    r130 = r130 * r63;
    r130 = r130 * r51;
    r130 = r130 * r69;
    r107 = fma(r37, r130, r107);
    r107 = fma(r44, r115, r107);
    r130 = r3 * r107;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             10 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r132,
                                             r130);
    r130 = r3 * r20;
    r130 = r130 * r1;
    r132 = r20 * r0;
    r86 = r2 * r132;
    r130 = fma(r103, r86, r109 * r130);
    r36 = r3 * r20;
    r36 = r36 * r1;
    r36 = fma(r116, r86, r138 * r36);
    WriteSum2<double, double>((double*)inout_shared, r130, r36);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = r3 * r20;
    r36 = r36 * r1;
    r36 = fma(r120, r86, r89 * r36);
    r130 = r3 * r20;
    r130 = r130 * r1;
    r130 = fma(r84, r86, r101 * r130);
    WriteSum2<double, double>((double*)inout_shared, r36, r130);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r130 = r3 * r20;
    r130 = r130 * r1;
    r130 = fma(r85, r86, r117 * r130);
    r36 = r3 * r20;
    r36 = r36 * r1;
    r36 = fma(r128, r86, r107 * r36);
    WriteSum2<double, double>((double*)inout_shared, r130, r36);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = r103 * r103;
    r130 = r2 * r2;
    r71 = r109 * r109;
    r129 = r3 * r3;
    r71 = fma(r129, r71, r130 * r36);
    r36 = r116 * r116;
    r118 = r138 * r138;
    r118 = fma(r129, r118, r130 * r36);
    WriteSum2<double, double>((double*)inout_shared, r71, r118);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r118 = r89 * r89;
    r71 = r120 * r120;
    r71 = fma(r130, r71, r129 * r118);
    r118 = r101 * r101;
    r36 = r84 * r84;
    r36 = fma(r130, r36, r129 * r118);
    WriteSum2<double, double>((double*)inout_shared, r71, r36);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = r117 * r117;
    r71 = r85 * r85;
    r71 = fma(r130, r71, r129 * r36);
    r36 = r128 * r128;
    r118 = r107 * r107;
    r118 = fma(r129, r118, r130 * r36);
    WriteSum2<double, double>((double*)inout_shared, r71, r118);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r118 = r103 * r116;
    r71 = r109 * r138;
    r71 = fma(r129, r71, r130 * r118);
    r118 = r109 * r89;
    r36 = r103 * r120;
    r36 = fma(r130, r36, r129 * r118);
    WriteSum2<double, double>((double*)inout_shared, r71, r36);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = r103 * r84;
    r71 = r109 * r101;
    r71 = fma(r129, r71, r130 * r36);
    r36 = r109 * r117;
    r118 = r103 * r85;
    r118 = fma(r130, r118, r129 * r36);
    WriteSum2<double, double>((double*)inout_shared, r71, r118);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r118 = r103 * r128;
    r71 = r109 * r107;
    r71 = fma(r129, r71, r130 * r118);
    r118 = r138 * r89;
    r36 = r116 * r120;
    r36 = fma(r130, r36, r129 * r118);
    WriteSum2<double, double>((double*)inout_shared, r71, r36);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = r138 * r101;
    r71 = r116 * r84;
    r71 = fma(r130, r71, r129 * r36);
    r36 = r116 * r85;
    r118 = r138 * r117;
    r118 = fma(r129, r118, r130 * r36);
    WriteSum2<double, double>((double*)inout_shared, r71, r118);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r118 = r116 * r128;
    r71 = r138 * r107;
    r71 = fma(r129, r71, r130 * r118);
    r118 = r89 * r101;
    r36 = r120 * r84;
    r36 = fma(r130, r36, r129 * r118);
    WriteSum2<double, double>((double*)inout_shared, r71, r36);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = r89 * r117;
    r71 = r120 * r85;
    r71 = fma(r130, r71, r129 * r36);
    r36 = r89 * r107;
    r118 = r120 * r128;
    r118 = fma(r130, r118, r129 * r36);
    WriteSum2<double, double>((double*)inout_shared, r71, r118);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r118 = r101 * r117;
    r71 = r84 * r85;
    r71 = fma(r130, r71, r129 * r118);
    r118 = r101 * r107;
    r36 = r84 * r128;
    r36 = fma(r130, r36, r129 * r118);
    WriteSum2<double, double>((double*)inout_shared, r71, r36);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = r117 * r107;
    r71 = r85 * r128;
    r71 = fma(r130, r71, r129 * r36);
    WriteSum1<double, double>((double*)inout_shared, r71);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             0 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r59,
                                             r58);
    r71 = r2 * r43;
    r71 = r71 * r70;
    r71 = r71 * r55;
    r36 = r3 * r43;
    r36 = r36 * r37;
    r36 = r36 * r70;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             2 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r71,
                                             r36);
    r118 = r2 * r70;
    r118 = r118 * r55;
    r118 = r118 * r62;
    r115 = r3 * r37;
    r115 = r115 * r70;
    r115 = r115 * r62;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             4 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r118,
                                             r115);
    r100 = r3 * r72;
    r68 = r2 * r6;
    r68 = r68 * r55;
    r68 = r68 * r52;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             6 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r68,
                                             r100);
    r131 = r2 * r31;
    r108 = r3 * r6;
    r108 = r108 * r55;
    r108 = r108 * r52;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             8 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r131,
                                             r108);
    r74 = r2 * r70;
    r74 = r74 * r55;
    r74 = r74 * r67;
    r57 = r3 * r37;
    r57 = r57 * r70;
    r57 = r57 * r67;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             10 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r74,
                                             r57);
    r112 = r2 * r70;
    r112 = r112 * r55;
    r112 = r112 * r66;
    r69 = r3 * r37;
    r69 = r69 * r70;
    r69 = r69 * r66;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             12 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r112,
                                             r69);
    r51 = r2 * r43;
    r110 = r3 * r43;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             14 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r51,
                                             r110);
    r114 = r20 * r58;
    r114 = r114 * r1;
    r9 = r59 * r132;
    WriteSum2<double, double>((double*)inout_shared, r9, r114);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r114 = r20 * r1;
    WriteSum2<double, double>((double*)inout_shared, r132, r114);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r114 = r43 * r70;
    r114 = r114 * r55;
    r132 = r3 * r20;
    r132 = r132 * r43;
    r132 = r132 * r1;
    r132 = r132 * r37;
    r132 = fma(r70, r132, r86 * r114);
    r114 = r3 * r20;
    r114 = r114 * r1;
    r114 = r114 * r37;
    r114 = r114 * r70;
    r9 = r70 * r55;
    r9 = r9 * r62;
    r9 = fma(r86, r9, r62 * r114);
    WriteSum2<double, double>((double*)inout_shared, r132, r9);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            4 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = r3 * r20;
    r9 = r9 * r72;
    r132 = r2 * r21;
    r132 = r132 * r0;
    r132 = r132 * r55;
    r132 = fma(r52, r132, r1 * r9);
    r9 = r3 * r21;
    r9 = r9 * r1;
    r9 = r9 * r55;
    r9 = fma(r52, r9, r31 * r86);
    WriteSum2<double, double>((double*)inout_shared, r132, r9);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            6 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = r3 * r20;
    r9 = r9 * r1;
    r9 = r9 * r37;
    r9 = r9 * r70;
    r132 = r70 * r55;
    r132 = r132 * r67;
    r132 = fma(r86, r132, r67 * r9);
    r9 = r3 * r20;
    r9 = r9 * r1;
    r9 = r9 * r37;
    r9 = r9 * r70;
    r0 = r70 * r55;
    r0 = r0 * r66;
    r0 = fma(r86, r0, r66 * r9);
    WriteSum2<double, double>((double*)inout_shared, r132, r0);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            8 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r3 * r20;
    r0 = r0 * r43;
    r0 = r0 * r1;
    r86 = r43 * r86;
    WriteSum2<double, double>((double*)inout_shared, r86, r0);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            10 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r59 * r59;
    r86 = r58 * r58;
    WriteSum2<double, double>((double*)inout_shared, r0, r86);
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
    r22 = r35 * r62;
    r22 = r22 * r129;
    r86 = r29 * r35;
    r86 = r86 * r62;
    r86 = r86 * r130;
    r86 = fma(r98, r86, r50 * r22);
    r22 = r35 * r129;
    r22 = r22 * r66;
    r0 = r29 * r35;
    r0 = r0 * r130;
    r0 = r0 * r98;
    r0 = fma(r66, r0, r50 * r22);
    WriteSum2<double, double>((double*)inout_shared, r86, r0);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            4 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r86 = r37 * r130;
    r22 = r29 * r53;
    r48 = r28 * r48;
    r48 = 1.0 / r48;
    r32 = r41 * r32;
    r32 = 1.0 / r32;
    r22 = r22 * r35;
    r22 = r22 * r35;
    r22 = r22 * r123;
    r22 = r22 * r48;
    r22 = r22 * r32;
    r22 = r22 * r55;
    r32 = r72 * r72;
    r32 = fma(r129, r32, r22 * r86);
    r86 = r37 * r129;
    r48 = r31 * r130;
    r22 = fma(r31, r48, r86 * r22);
    WriteSum2<double, double>((double*)inout_shared, r32, r22);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            6 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = r67 * r67;
    r32 = r35 * r22;
    r123 = r129 * r50;
    r41 = r35 * r22;
    r28 = r29 * r130;
    r28 = r28 * r98;
    r41 = fma(r28, r41, r123 * r32);
    r32 = r43 * r22;
    r32 = r35 * r32;
    r1 = r43 * r32;
    r1 = fma(r28, r1, r123 * r1);
    WriteSum2<double, double>((double*)inout_shared, r41, r1);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            8 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r62 * r130;
    r132 = r62 * r129;
    WriteSum2<double, double>((double*)inout_shared, r1, r132);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            10 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r132 = 0.00000000000000000e+00;
    WriteSum2<double, double>((double*)inout_shared, r132, r59);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r2 * r43;
    r1 = r1 * r59;
    r1 = r1 * r70;
    r1 = r1 * r55;
    WriteSum2<double, double>((double*)inout_shared, r132, r1);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            2 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r2 * r59;
    r1 = r1 * r70;
    r1 = r1 * r55;
    r1 = r1 * r62;
    r9 = r2 * r6;
    r9 = r9 * r59;
    r9 = r9 * r55;
    r9 = r9 * r52;
    WriteSum2<double, double>((double*)inout_shared, r1, r9);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r2 * r31;
    r31 = r31 * r59;
    r9 = r2 * r59;
    r9 = r9 * r70;
    r9 = r9 * r55;
    r9 = r9 * r67;
    WriteSum2<double, double>((double*)inout_shared, r31, r9);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            6 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = r2 * r43;
    r9 = r9 * r59;
    r59 = r2 * r59;
    r59 = r59 * r70;
    r59 = r59 * r55;
    r59 = r59 * r66;
    WriteSum2<double, double>((double*)inout_shared, r59, r9);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            8 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = r3 * r43;
    r9 = r9 * r58;
    r9 = r9 * r37;
    r9 = r9 * r70;
    WriteSum2<double, double>((double*)inout_shared, r58, r9);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            12 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = r3 * r72;
    r9 = r9 * r58;
    r59 = r3 * r58;
    r59 = r59 * r37;
    r59 = r59 * r70;
    r59 = r59 * r62;
    WriteSum2<double, double>((double*)inout_shared, r59, r9);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            14 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = r3 * r6;
    r9 = r9 * r58;
    r9 = r9 * r55;
    r9 = r9 * r52;
    r59 = r3 * r58;
    r59 = r59 * r37;
    r59 = r59 * r70;
    r59 = r59 * r67;
    WriteSum2<double, double>((double*)inout_shared, r9, r59);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            16 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = r3 * r58;
    r59 = r59 * r37;
    r59 = r59 * r70;
    r59 = r59 * r66;
    WriteSum2<double, double>((double*)inout_shared, r59, r132);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            18 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = r3 * r43;
    r59 = r59 * r58;
    WriteSum2<double, double>((double*)inout_shared, r59, r132);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            20 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r71, r118);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            22 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r68, r131);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            24 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r74, r112);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            26 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r51, r132);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            28 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r36, r115);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            30 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r100, r108);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            32 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r57, r69);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            34 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r132, r110);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            36 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r110 = r35 * r67;
    r110 = r110 * r129;
    r69 = r29 * r35;
    r69 = r69 * r67;
    r69 = r69 * r130;
    r69 = fma(r98, r69, r50 * r110);
    r110 = r70 * r86;
    r57 = r72 * r110;
    r108 = r29 * r43;
    r108 = r108 * r96;
    r108 = r108 * r102;
    r108 = r108 * r37;
    r108 = r108 * r55;
    r108 = r108 * r130;
    r108 = fma(r113, r108, r43 * r57);
    WriteSum2<double, double>((double*)inout_shared, r69, r108);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            38 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r108 = r53 * r43;
    r108 = r108 * r96;
    r108 = r108 * r102;
    r108 = r108 * r55;
    r108 = r108 * r113;
    r69 = r55 * r48;
    r100 = r70 * r69;
    r108 = fma(r43, r100, r86 * r108);
    WriteSum2<double, double>((double*)inout_shared, r108, r0);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            40 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r70 * r55;
    r0 = r0 * r62;
    r0 = r0 * r130;
    r108 = r35 * r129;
    r115 = r43 * r66;
    r108 = r108 * r50;
    r50 = r29 * r35;
    r50 = r50 * r130;
    r50 = r50 * r98;
    r50 = fma(r115, r50, r115 * r108);
    WriteSum2<double, double>((double*)inout_shared, r50, r0);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            42 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r62 * r110;
    r108 = r29 * r96;
    r108 = r108 * r102;
    r108 = r108 * r37;
    r108 = r108 * r55;
    r108 = r108 * r62;
    r108 = r108 * r130;
    r108 = fma(r113, r108, r62 * r57);
    WriteSum2<double, double>((double*)inout_shared, r0, r108);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            44 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r108 = r53 * r96;
    r108 = r108 * r102;
    r108 = r108 * r55;
    r108 = r108 * r62;
    r108 = r108 * r113;
    r108 = fma(r62, r100, r86 * r108);
    WriteSum2<double, double>((double*)inout_shared, r108, r50);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            46 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r70 * r55;
    r50 = r50 * r67;
    r50 = r50 * r130;
    WriteSum2<double, double>((double*)inout_shared, r41, r50);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            48 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r67 * r110;
    r41 = r6 * r52;
    r108 = r6 * r72;
    r108 = r108 * r55;
    r108 = r108 * r52;
    r108 = fma(r129, r108, r69 * r41);
    WriteSum2<double, double>((double*)inout_shared, r50, r108);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            50 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r108 = r29 * r96;
    r108 = r108 * r102;
    r108 = r108 * r37;
    r108 = r108 * r55;
    r108 = r108 * r67;
    r108 = r108 * r130;
    r108 = fma(r113, r108, r67 * r57);
    r50 = r29 * r96;
    r50 = r50 * r102;
    r50 = r50 * r37;
    r50 = r50 * r55;
    r50 = r50 * r130;
    r50 = r50 * r113;
    r50 = fma(r66, r50, r66 * r57);
    WriteSum2<double, double>((double*)inout_shared, r108, r50);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            52 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r6 * r43;
    r50 = r50 * r55;
    r50 = r50 * r52;
    r50 = r50 * r130;
    r108 = r43 * r72;
    r108 = r108 * r129;
    WriteSum2<double, double>((double*)inout_shared, r50, r108);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            54 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r108 = r53 * r96;
    r108 = r108 * r102;
    r108 = r108 * r55;
    r108 = r108 * r67;
    r108 = r108 * r113;
    r108 = fma(r67, r100, r86 * r108);
    r50 = r53 * r96;
    r50 = r50 * r102;
    r50 = r50 * r55;
    r50 = r50 * r113;
    r50 = r50 * r66;
    r100 = fma(r66, r100, r86 * r50);
    WriteSum2<double, double>((double*)inout_shared, r108, r100);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            56 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r43 * r48;
    r100 = r6 * r43;
    r100 = r100 * r55;
    r100 = r100 * r52;
    r100 = r100 * r129;
    WriteSum2<double, double>((double*)inout_shared, r48, r100);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            58 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r100 = r70 * r55;
    r100 = r100 * r130;
    r100 = r100 * r66;
    r28 = fma(r32, r28, r32 * r123);
    WriteSum2<double, double>((double*)inout_shared, r28, r100);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            60 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = r66 * r110;
    r100 = r70 * r55;
    r100 = r100 * r130;
    r100 = r100 * r115;
    WriteSum2<double, double>((double*)inout_shared, r66, r100);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            62 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r110 = r115 * r110;
    WriteSum2<double, double>((double*)inout_shared, r110, r132);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            64 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
}

void ThinPrismFisheyeFixedPointResJac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* point,
    unsigned int point_num_alloc,
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
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFixedPointResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
      point,
      point_num_alloc,
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
      problem_size);
}

}  // namespace caspar