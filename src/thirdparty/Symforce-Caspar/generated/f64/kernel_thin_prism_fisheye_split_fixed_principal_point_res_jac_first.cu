#include "kernel_thin_prism_fisheye_split_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPrincipalPointResJacFirstKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        SharedIndex* focal_and_extra_indices,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
        double* out_pose_jac,
        unsigned int out_pose_jac_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        double* out_focal_and_extra_jac,
        unsigned int out_focal_and_extra_jac_num_alloc,
        double* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        double* const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        double* const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
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

  __shared__ SharedIndex focal_and_extra_indices_loc[1024];
  focal_and_extra_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

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
      r141, r142, r143, r144, r145, r146, r147, r148;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(principal_point,
                                            0 * principal_point_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1);
  };
  LoadShared<2, double, double>(focal_and_extra,
                                0 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target,
                        r2,
                        r3);
  };
  __syncthreads();
  LoadShared<2, double, double>(focal_and_extra,
                                4 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target,
                        r4,
                        r5);
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
  if (global_thread_idx < problem_size) {
    r10 = 2.00000000000000000e+00;
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r11, r12);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r13,
                                            r14);
  };
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r15, r16);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r17,
                                            r18);
    r19 = fma(r16, r17, r11 * r14);
    r20 = r12 * r13;
    r21 = -1.00000000000000000e+00;
    r19 = fma(r21, r20, r19);
    r19 = fma(r15, r18, r19);
    r20 = r10 * r19;
    r22 = r12 * r14;
    r23 = r16 * r18;
    r24 = r22 + r23;
    r25 = r11 * r13;
    r26 = r15 * r17;
    r24 = r24 + r25;
    r24 = fma(r21, r26, r24);
    r20 = r20 * r24;
    r27 = fma(r12, r17, r15 * r14);
    r28 = r11 * r18;
    r27 = fma(r21, r28, r27);
    r27 = fma(r16, r13, r27);
    r28 = r10 * r27;
    r29 = fma(r12, r18, r11 * r17);
    r29 = fma(r15, r13, r29);
    r29 = fma(r21, r29, r16 * r14);
    r28 = fma(r29, r28, r20);
    r7 = fma(r8, r28, r7);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r30, r31);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r32 = r17 * r18;
    r32 = r32 * r10;
    r33 = r13 * r14;
    r33 = fma(r10, r33, r32);
    r34 = r17 * r17;
    r35 = -2.00000000000000000e+00;
    r34 = r34 * r35;
    r36 = 1.00000000000000000e+00;
    r37 = r13 * r13;
    r37 = fma(r35, r37, r36);
    r38 = r34 + r37;
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r39);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r40 = r18 * r13;
    r40 = r40 * r10;
    r41 = r17 * r14;
    r41 = fma(r35, r41, r40);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r42);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r43 = r10 * r27;
    r43 = r43 * r24;
    r44 = r19 * r35;
    r44 = fma(r29, r44, r43);
    r45 = r27 * r27;
    r45 = r45 * r35;
    r46 = r36 + r45;
    r47 = r19 * r19;
    r47 = r47 * r35;
    r46 = r46 + r47;
    r7 = fma(r30, r33, r7);
    r7 = fma(r31, r38, r7);
    r7 = fma(r39, r41, r7);
    r7 = fma(r42, r44, r7);
    r7 = fma(r9, r46, r7);
    r48 = r7 * r7;
    r49 = 1.00000000000000008e-15;
    r50 = r35 * r24;
    r50 = r50 * r24;
    r51 = r36 + r50;
    r51 = r51 + r45;
    r6 = fma(r8, r51, r6);
    r45 = r27 * r35;
    r45 = fma(r29, r45, r20);
    r20 = r10 * r27;
    r20 = r20 * r19;
    r52 = r10 * r24;
    r52 = fma(r29, r52, r20);
    r53 = r17 * r13;
    r53 = r53 * r10;
    r54 = r18 * r14;
    r54 = fma(r10, r54, r53);
    r55 = r13 * r14;
    r55 = fma(r35, r55, r32);
    r32 = r18 * r18;
    r32 = r32 * r35;
    r37 = r32 + r37;
    r6 = fma(r9, r45, r6);
    r6 = fma(r42, r52, r6);
    r6 = fma(r39, r54, r6);
    r6 = fma(r31, r55, r6);
    r6 = fma(r30, r37, r6);
    r56 = r6 * r6;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r57);
    r58 = r35 * r24;
    r58 = fma(r29, r58, r20);
    r57 = fma(r8, r58, r57);
    r20 = r18 * r14;
    r20 = fma(r35, r20, r53);
    r32 = r36 + r32;
    r32 = r32 + r34;
    r34 = r17 * r14;
    r34 = fma(r10, r34, r40);
    r40 = r10 * r19;
    r40 = fma(r29, r40, r43);
    r50 = r36 + r50;
    r50 = r50 + r47;
    r57 = fma(r30, r20, r57);
    r57 = fma(r39, r32, r57);
    r57 = fma(r31, r34, r57);
    r57 = fma(r9, r40, r57);
    r57 = fma(r42, r50, r57);
    r31 = copysign(1.0, r57);
    r31 = fma(r49, r31, r57);
    r57 = r31 * r31;
    r39 = 1.0 / r57;
    r30 = r7 * r7;
    r30 = fma(r39, r30, r39 * r56);
    r56 = sqrt(r30);
    r47 = copysign(1.0, r56);
    r47 = fma(r49, r47, r56);
    r49 = r47 * r47;
    r43 = 1.0 / r49;
    r56 = atan(r56);
    r53 = r56 * r39;
    r59 = r56 * r53;
    r48 = r48 * r43;
    r48 = r48 * r59;
    r60 = 3.00000000000000000e+00;
    r61 = r60 * r59;
    r62 = r6 * r43;
    r63 = r6 * r62;
    r61 = fma(r63, r61, r48);
  };
  LoadShared<2, double, double>(focal_and_extra,
                                8 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target,
                        r64,
                        r65);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r66 = r59 * r63;
    r48 = r48 + r66;
    r67 = fma(r64, r48, r5 * r61);
    r68 = 1.0 / r31;
    r69 = 1.0 / r47;
    r70 = r68 * r69;
    r71 = r56 * r70;
    r72 = r6 * r71;
    r73 = r4 * r7;
    r74 = r10 * r59;
    r73 = r73 * r62;
    r67 = fma(r74, r73, r67);
  };
  LoadShared<2, double, double>(focal_and_extra,
                                2 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target,
                        r75,
                        r76);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r77 = r48 * r48;
    r78 = fma(r76, r77, r75 * r48);
  };
  LoadShared<2, double, double>(focal_and_extra,
                                6 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target,
                        r79,
                        r80);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r81 = r77 * r77;
    r82 = r48 * r77;
    r78 = fma(r80, r81, r78);
    r78 = fma(r79, r82, r78);
    r83 = r78 * r71;
    r67 = r67 + r72;
    r67 = fma(r6, r83, r67);
    r0 = fma(r2, r67, r0);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r73, r84);
    r0 = fma(r73, r21, r0);
    r73 = r7 * r7;
    r73 = r73 * r60;
    r73 = r73 * r43;
    r73 = fma(r59, r73, r66);
    r66 = fma(r65, r48, r4 * r73);
    r85 = r5 * r7;
    r85 = r85 * r62;
    r66 = fma(r74, r85, r66);
    r66 = fma(r7, r83, r66);
    r66 = fma(r7, r71, r66);
    r1 = fma(r3, r66, r1);
    r1 = fma(r84, r21, r1);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r0, r1);
    r84 = fma(r1, r1, r0 * r0);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r84);
  if (global_thread_idx < problem_size) {
    r84 = 5.00000000000000000e-01;
    r85 = r6 * r84;
    r86 = rsqrt(r30);
    r87 = r10 * r6;
    r88 = r11 * r14;
    r89 = -5.00000000000000000e-01;
    r90 = r16 * r17;
    r90 = fma(r89, r90, r89 * r88);
    r88 = r15 * r18;
    r90 = fma(r89, r88, r90);
    r91 = r12 * r13;
    r90 = fma(r84, r91, r90);
    r91 = r24 * r90;
    r88 = r15 * r14;
    r92 = r12 * r17;
    r92 = fma(r84, r92, r84 * r88);
    r88 = r11 * r18;
    r92 = fma(r89, r88, r92);
    r93 = r16 * r13;
    r92 = fma(r84, r93, r92);
    r93 = r29 * r92;
    r88 = fma(r10, r93, r10 * r91);
    r94 = r10 * r19;
    r95 = fma(r84, r26, r89 * r22);
    r95 = fma(r89, r23, r95);
    r95 = fma(r89, r25, r95);
    r96 = r10 * r27;
    r97 = r16 * r14;
    r98 = r11 * r17;
    r98 = fma(r89, r98, r84 * r97);
    r97 = r12 * r18;
    r98 = fma(r89, r97, r98);
    r99 = r15 * r13;
    r98 = fma(r89, r99, r98);
    r96 = r96 * r98;
    r94 = fma(r95, r94, r96);
    r88 = r88 + r94;
    r99 = r10 * r24;
    r99 = r99 * r98;
    r97 = r10 * r19;
    r97 = r97 * r92;
    r100 = r99 + r97;
    r101 = r27 * r35;
    r100 = fma(r90, r101, r100);
    r102 = r35 * r29;
    r100 = fma(r95, r102, r100);
    r100 = fma(r9, r100, r42 * r88);
    r88 = r24 * r92;
    r102 = -4.00000000000000000e+00;
    r88 = r88 * r102;
    r101 = r27 * r95;
    r103 = r102 * r101;
    r104 = r88 + r103;
    r100 = fma(r8, r104, r100);
    r87 = r87 * r100;
    r104 = r6 * r6;
    r105 = r10 * r24;
    r105 = r105 * r95;
    r106 = r10 * r27;
    r106 = fma(r92, r106, r105);
    r92 = r10 * r19;
    r92 = r92 * r90;
    r107 = r10 * r29;
    r107 = r107 * r98;
    r108 = r92 + r107;
    r109 = r106 + r108;
    r93 = fma(r35, r93, r35 * r91);
    r93 = r93 + r94;
    r93 = fma(r8, r93, r9 * r109);
    r109 = r19 * r98;
    r109 = r109 * r102;
    r88 = r88 + r109;
    r93 = fma(r42, r88, r93);
    r57 = r31 * r57;
    r88 = 1.0 / r57;
    r110 = r35 * r88;
    r104 = r104 * r93;
    r104 = fma(r110, r104, r39 * r87);
    r87 = r93 * r110;
    r111 = r7 * r7;
    r104 = fma(r111, r87, r104);
    r112 = r10 * r7;
    r113 = r19 * r35;
    r114 = r35 * r29;
    r114 = r114 * r98;
    r113 = fma(r90, r113, r114);
    r113 = r113 + r106;
    r103 = r109 + r103;
    r103 = fma(r9, r103, r42 * r113);
    r113 = r10 * r29;
    r113 = fma(r95, r113, r97);
    r97 = r10 * r27;
    r97 = fma(r90, r97, r99);
    r113 = r113 + r97;
    r103 = fma(r8, r113, r103);
    r112 = r112 * r103;
    r104 = fma(r39, r112, r104);
    r30 = r36 + r30;
    r30 = 1.0 / r30;
    r85 = r85 * r68;
    r85 = r85 * r69;
    r85 = r85 * r86;
    r85 = r85 * r104;
    r85 = r85 * r30;
    r36 = r43 * r111;
    r112 = r56 * r56;
    r87 = r110 * r112;
    r36 = r36 * r87;
    r113 = r7 * r86;
    r99 = r104 * r113;
    r109 = r30 * r53;
    r106 = r7 * r43;
    r99 = r99 * r109;
    r99 = fma(r106, r99, r93 * r36);
    r115 = r21 * r7;
    r49 = r47 * r49;
    r116 = 1.0 / r49;
    r117 = r116 * r59;
    r117 = r117 * r113;
    r115 = r115 * r104;
    r99 = fma(r117, r115, r99);
    r118 = r74 * r106;
    r99 = fma(r103, r118, r99);
    r115 = r86 * r109;
    r115 = r115 * r63;
    r119 = r21 * r6;
    r119 = r119 * r6;
    r119 = r119 * r104;
    r119 = r119 * r86;
    r119 = r119 * r116;
    r119 = fma(r59, r119, r104 * r115);
    r120 = r100 * r62;
    r119 = fma(r74, r120, r119);
    r121 = r93 * r63;
    r119 = fma(r87, r121, r119);
    r121 = r99 + r119;
    r120 = fma(r64, r121, r85);
    r122 = r60 * r104;
    r123 = r6 * r6;
    r124 = -3.00000000000000000e+00;
    r123 = r123 * r124;
    r123 = r123 * r104;
    r123 = r123 * r86;
    r123 = r123 * r116;
    r123 = fma(r59, r123, r115 * r122);
    r122 = 6.00000000000000000e+00;
    r125 = r100 * r122;
    r125 = r125 * r59;
    r123 = fma(r62, r125, r123);
    r126 = r63 * r112;
    r127 = -6.00000000000000000e+00;
    r128 = r93 * r127;
    r128 = r128 * r88;
    r123 = fma(r128, r126, r123);
    r123 = r123 + r99;
    r99 = r4 * r100;
    r120 = fma(r118, r99, r120);
    r126 = r56 * r89;
    r126 = r126 * r104;
    r126 = r126 * r68;
    r126 = r126 * r86;
    r120 = fma(r62, r126, r120);
    r125 = r56 * r78;
    r125 = r125 * r89;
    r125 = r125 * r104;
    r125 = r125 * r68;
    r125 = r125 * r86;
    r120 = fma(r62, r125, r120);
    r129 = r21 * r6;
    r129 = r129 * r78;
    r129 = r129 * r93;
    r129 = r129 * r69;
    r120 = fma(r53, r129, r120);
    r130 = r4 * r104;
    r131 = r35 * r6;
    r131 = r131 * r117;
    r120 = fma(r131, r130, r120);
    r132 = r4 * r103;
    r132 = r132 * r62;
    r120 = fma(r74, r132, r120);
    r133 = r4 * r7;
    r133 = r133 * r56;
    r133 = r133 * r56;
    r133 = r133 * r102;
    r133 = r133 * r88;
    r133 = r133 * r62;
    r134 = r4 * r104;
    r135 = r10 * r62;
    r135 = r135 * r113;
    r135 = r135 * r109;
    r120 = fma(r135, r134, r120);
    r136 = r76 * r10;
    r136 = r136 * r48;
    r136 = fma(r75, r121, r121 * r136);
    r79 = r79 * r60;
    r79 = r79 * r77;
    r137 = 4.00000000000000000e+00;
    r80 = r80 * r137;
    r80 = r80 * r82;
    r136 = fma(r121, r79, r136);
    r136 = fma(r121, r80, r136);
    r138 = r6 * r136;
    r120 = fma(r71, r138, r120);
    r139 = r21 * r6;
    r139 = r139 * r93;
    r139 = r139 * r69;
    r120 = fma(r53, r139, r120);
    r120 = fma(r5, r123, r120);
    r120 = fma(r78, r85, r120);
    r120 = fma(r100, r83, r120);
    r120 = fma(r93, r133, r120);
    r120 = fma(r100, r71, r120);
    r139 = r2 * r120;
    r138 = r7 * r7;
    r138 = r138 * r56;
    r138 = r138 * r56;
    r138 = r138 * r43;
    r134 = r60 * r104;
    r134 = r134 * r113;
    r134 = r134 * r109;
    r134 = fma(r106, r134, r128 * r138);
    r138 = r7 * r124;
    r138 = r138 * r104;
    r134 = fma(r117, r138, r134);
    r128 = r7 * r103;
    r128 = r128 * r122;
    r128 = r128 * r43;
    r134 = fma(r59, r128, r134);
    r134 = r134 + r119;
    r134 = fma(r4, r134, r65 * r121);
    r121 = r5 * r118;
    r119 = r21 * r7;
    r119 = r119 * r93;
    r119 = r119 * r69;
    r134 = fma(r53, r119, r134);
    r128 = r56 * r89;
    r128 = r128 * r43;
    r128 = r128 * r68;
    r128 = r128 * r113;
    r138 = r78 * r128;
    r132 = r5 * r104;
    r134 = fma(r131, r132, r134);
    r130 = r84 * r104;
    r130 = r130 * r30;
    r130 = r130 * r70;
    r85 = r5 * r103;
    r85 = r85 * r62;
    r134 = fma(r74, r85, r134);
    r129 = r5 * r7;
    r129 = r129 * r56;
    r129 = r129 * r56;
    r129 = r129 * r102;
    r129 = r129 * r93;
    r129 = r129 * r88;
    r134 = fma(r62, r129, r134);
    r125 = r7 * r136;
    r134 = fma(r71, r125, r134);
    r126 = r5 * r104;
    r134 = fma(r135, r126, r134);
    r99 = r78 * r113;
    r134 = fma(r130, r99, r134);
    r123 = r21 * r7;
    r123 = r123 * r78;
    r123 = r123 * r93;
    r123 = r123 * r69;
    r134 = fma(r53, r123, r134);
    r134 = fma(r100, r121, r134);
    r134 = fma(r104, r138, r134);
    r134 = fma(r104, r128, r134);
    r134 = fma(r113, r130, r134);
    r134 = fma(r103, r83, r134);
    r134 = fma(r103, r71, r134);
    r123 = r3 * r134;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             0 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r139,
                                             r123);
    r123 = r35 * r24;
    r123 = fma(r95, r123, r114);
    r139 = r10 * r27;
    r99 = r15 * r14;
    r126 = r12 * r17;
    r126 = fma(r89, r126, r89 * r99);
    r99 = r11 * r18;
    r126 = fma(r84, r99, r126);
    r125 = r16 * r13;
    r126 = fma(r89, r125, r126);
    r139 = r139 * r126;
    r125 = r10 * r19;
    r99 = r11 * r14;
    r129 = r16 * r17;
    r129 = fma(r84, r129, r84 * r99);
    r99 = r15 * r18;
    r129 = fma(r84, r99, r129);
    r85 = r12 * r13;
    r129 = fma(r89, r85, r129);
    r125 = fma(r129, r125, r139);
    r123 = r123 + r125;
    r85 = r10 * r24;
    r85 = r85 * r129;
    r99 = r10 * r29;
    r99 = fma(r126, r99, r85);
    r99 = r99 + r94;
    r99 = fma(r9, r99, r8 * r123);
    r123 = r24 * r98;
    r123 = r123 * r102;
    r94 = r19 * r126;
    r130 = r102 * r94;
    r132 = r123 + r130;
    r99 = fma(r42, r132, r99);
    r132 = r127 * r99;
    r132 = r132 * r88;
    r132 = r132 * r63;
    r107 = r105 + r107;
    r107 = r107 + r125;
    r125 = r27 * r102;
    r125 = r125 * r129;
    r123 = r123 + r125;
    r123 = fma(r8, r123, r42 * r107);
    r107 = r35 * r29;
    r107 = fma(r35, r101, r129 * r107);
    r105 = r10 * r19;
    r105 = r105 * r98;
    r119 = r10 * r24;
    r119 = fma(r126, r119, r105);
    r107 = r107 + r119;
    r123 = fma(r9, r107, r123);
    r107 = r122 * r123;
    r107 = r107 * r59;
    r107 = fma(r62, r107, r112 * r132);
    r132 = r6 * r6;
    r140 = r10 * r6;
    r140 = r140 * r123;
    r141 = r10 * r7;
    r85 = r96 + r85;
    r96 = r19 * r35;
    r85 = fma(r95, r96, r85);
    r95 = r35 * r29;
    r85 = fma(r126, r95, r85);
    r95 = r10 * r29;
    r101 = fma(r10, r101, r129 * r95);
    r101 = r101 + r119;
    r101 = fma(r8, r101, r42 * r85);
    r130 = r125 + r130;
    r101 = fma(r9, r130, r101);
    r141 = r141 * r101;
    r141 = fma(r39, r141, r39 * r140);
    r140 = r99 * r110;
    r141 = fma(r111, r140, r141);
    r130 = r6 * r6;
    r130 = r130 * r99;
    r141 = fma(r110, r130, r141);
    r130 = r124 * r141;
    r132 = r132 * r86;
    r132 = r132 * r116;
    r132 = r132 * r59;
    r107 = fma(r130, r132, r107);
    r140 = r60 * r141;
    r107 = fma(r115, r140, r107);
    r125 = r21 * r7;
    r125 = r125 * r141;
    r125 = fma(r117, r125, r101 * r118);
    r85 = r141 * r113;
    r85 = r85 * r109;
    r125 = fma(r106, r85, r125);
    r125 = fma(r99, r36, r125);
    r107 = r107 + r125;
    r140 = r99 * r63;
    r132 = r123 * r62;
    r132 = fma(r74, r132, r87 * r140);
    r140 = r21 * r6;
    r140 = r140 * r6;
    r140 = r140 * r141;
    r140 = r140 * r86;
    r140 = r140 * r116;
    r132 = fma(r59, r140, r132);
    r132 = fma(r141, r115, r132);
    r125 = r125 + r132;
    r107 = fma(r64, r125, r5 * r107);
    r140 = r21 * r6;
    r140 = r140 * r99;
    r140 = r140 * r69;
    r107 = fma(r53, r140, r107);
    r85 = r6 * r84;
    r85 = r85 * r78;
    r85 = r85 * r141;
    r85 = r85 * r86;
    r85 = r85 * r30;
    r107 = fma(r70, r85, r107);
    r95 = r6 * r84;
    r95 = r95 * r141;
    r95 = r95 * r86;
    r95 = r95 * r30;
    r107 = fma(r70, r95, r107);
    r129 = r56 * r78;
    r129 = r129 * r89;
    r129 = r129 * r141;
    r129 = r129 * r68;
    r129 = r129 * r86;
    r107 = fma(r62, r129, r107);
    r96 = r76 * r10;
    r96 = r96 * r48;
    r96 = fma(r125, r96, r75 * r125);
    r96 = fma(r125, r80, r96);
    r96 = fma(r125, r79, r96);
    r142 = r6 * r96;
    r107 = fma(r71, r142, r107);
    r143 = r21 * r6;
    r143 = r143 * r78;
    r143 = r143 * r99;
    r143 = r143 * r69;
    r107 = fma(r53, r143, r107);
    r144 = r4 * r141;
    r107 = fma(r135, r144, r107);
    r145 = r4 * r141;
    r107 = fma(r131, r145, r107);
    r146 = r4 * r101;
    r146 = r146 * r62;
    r107 = fma(r74, r146, r107);
    r147 = r4 * r123;
    r107 = fma(r118, r147, r107);
    r148 = r56 * r89;
    r148 = r148 * r141;
    r148 = r148 * r68;
    r148 = r148 * r86;
    r107 = fma(r62, r148, r107);
    r107 = fma(r99, r133, r107);
    r107 = fma(r123, r71, r107);
    r107 = fma(r123, r83, r107);
    r148 = r2 * r107;
    r147 = r7 * r122;
    r147 = r147 * r101;
    r147 = r147 * r43;
    r146 = r7 * r117;
    r146 = fma(r130, r146, r59 * r147);
    r147 = r7 * r7;
    r147 = r147 * r56;
    r147 = r147 * r56;
    r147 = r147 * r127;
    r147 = r147 * r99;
    r147 = r147 * r43;
    r146 = fma(r88, r147, r146);
    r130 = r60 * r141;
    r130 = r130 * r113;
    r130 = r130 * r109;
    r146 = fma(r106, r130, r146);
    r146 = r146 + r132;
    r125 = fma(r65, r125, r4 * r146);
    r146 = r7 * r96;
    r125 = fma(r71, r146, r125);
    r132 = r84 * r141;
    r132 = r132 * r30;
    r132 = r132 * r70;
    r125 = fma(r113, r132, r125);
    r130 = r5 * r7;
    r130 = r130 * r56;
    r130 = r130 * r56;
    r130 = r130 * r102;
    r130 = r130 * r99;
    r130 = r130 * r88;
    r125 = fma(r62, r130, r125);
    r147 = r5 * r141;
    r125 = fma(r135, r147, r125);
    r145 = r21 * r7;
    r145 = r145 * r78;
    r145 = r145 * r99;
    r145 = r145 * r69;
    r125 = fma(r53, r145, r125);
    r144 = r5 * r141;
    r125 = fma(r131, r144, r125);
    r143 = r21 * r7;
    r143 = r143 * r99;
    r143 = r143 * r69;
    r125 = fma(r53, r143, r125);
    r142 = r5 * r101;
    r142 = r142 * r62;
    r125 = fma(r74, r142, r125);
    r129 = r84 * r78;
    r129 = r129 * r141;
    r129 = r129 * r30;
    r129 = r129 * r70;
    r125 = fma(r113, r129, r125);
    r125 = fma(r141, r128, r125);
    r125 = fma(r101, r83, r125);
    r125 = fma(r141, r138, r125);
    r125 = fma(r101, r71, r125);
    r125 = fma(r123, r121, r125);
    r129 = r3 * r125;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             2 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r148,
                                             r129);
    r129 = r19 * r102;
    r26 = fma(r89, r26, r84 * r22);
    r26 = fma(r84, r23, r26);
    r26 = fma(r84, r25, r26);
    r129 = r129 * r26;
    r91 = r102 * r91;
    r25 = r129 + r91;
    r23 = r10 * r27;
    r23 = r23 * r26;
    r105 = r105 + r23;
    r22 = r35 * r24;
    r105 = fma(r126, r22, r105);
    r148 = r35 * r29;
    r105 = fma(r90, r148, r105);
    r105 = fma(r8, r105, r42 * r25);
    r25 = r10 * r29;
    r25 = fma(r10, r94, r26 * r25);
    r25 = r25 + r97;
    r105 = fma(r9, r25, r105);
    r25 = r10 * r7;
    r148 = r10 * r24;
    r148 = r148 * r26;
    r139 = r139 + r148;
    r139 = r139 + r108;
    r108 = r35 * r29;
    r94 = fma(r35, r94, r26 * r108);
    r94 = r94 + r97;
    r94 = fma(r42, r94, r8 * r139);
    r98 = r27 * r98;
    r98 = r98 * r102;
    r129 = r129 + r98;
    r94 = fma(r9, r129, r94);
    r25 = r25 * r94;
    r129 = r6 * r6;
    r129 = r129 * r105;
    r129 = fma(r110, r129, r39 * r25);
    r25 = r10 * r6;
    r114 = r92 + r114;
    r92 = r27 * r35;
    r114 = fma(r126, r92, r114);
    r114 = r114 + r148;
    r91 = r98 + r91;
    r91 = fma(r8, r91, r9 * r114);
    r8 = r10 * r29;
    r8 = fma(r90, r8, r23);
    r8 = r8 + r119;
    r91 = fma(r42, r8, r91);
    r25 = r25 * r91;
    r129 = fma(r39, r25, r129);
    r8 = r105 * r110;
    r129 = fma(r111, r8, r129);
    r8 = r129 * r113;
    r8 = r8 * r109;
    r8 = fma(r106, r8, r105 * r36);
    r25 = r21 * r7;
    r25 = r25 * r129;
    r8 = fma(r117, r25, r8);
    r8 = fma(r94, r118, r8);
    r25 = r21 * r6;
    r25 = r25 * r6;
    r25 = r25 * r129;
    r25 = r25 * r86;
    r25 = r25 * r116;
    r42 = r91 * r62;
    r42 = fma(r74, r42, r59 * r25);
    r25 = r105 * r63;
    r42 = fma(r87, r25, r42);
    r42 = fma(r129, r115, r42);
    r25 = r8 + r42;
    r119 = r6 * r6;
    r119 = r119 * r124;
    r119 = r119 * r129;
    r119 = r119 * r86;
    r119 = r119 * r116;
    r23 = r122 * r91;
    r23 = r23 * r59;
    r23 = fma(r62, r23, r59 * r119);
    r119 = r127 * r105;
    r119 = r119 * r88;
    r119 = r119 * r63;
    r23 = fma(r112, r119, r23);
    r90 = r60 * r129;
    r23 = fma(r115, r90, r23);
    r23 = r23 + r8;
    r23 = fma(r5, r23, r64 * r25);
    r8 = r76 * r10;
    r8 = r8 * r48;
    r8 = fma(r25, r8, r75 * r25);
    r8 = fma(r25, r79, r8);
    r8 = fma(r25, r80, r8);
    r90 = r6 * r8;
    r23 = fma(r71, r90, r23);
    r119 = r4 * r91;
    r23 = fma(r118, r119, r23);
    r114 = r6 * r84;
    r114 = r114 * r129;
    r114 = r114 * r86;
    r114 = r114 * r30;
    r23 = fma(r70, r114, r23);
    r9 = r4 * r129;
    r23 = fma(r135, r9, r23);
    r98 = r21 * r6;
    r98 = r98 * r78;
    r98 = r98 * r105;
    r98 = r98 * r69;
    r23 = fma(r53, r98, r23);
    r92 = r6 * r84;
    r92 = r92 * r78;
    r92 = r92 * r129;
    r92 = r92 * r86;
    r92 = r92 * r30;
    r23 = fma(r70, r92, r23);
    r148 = r129 * r131;
    r126 = r21 * r6;
    r126 = r126 * r105;
    r126 = r126 * r69;
    r23 = fma(r53, r126, r23);
    r139 = r56 * r89;
    r139 = r139 * r129;
    r139 = r139 * r68;
    r139 = r139 * r86;
    r23 = fma(r62, r139, r23);
    r97 = r4 * r94;
    r97 = r97 * r62;
    r23 = fma(r74, r97, r23);
    r108 = r56 * r78;
    r108 = r108 * r89;
    r108 = r108 * r129;
    r108 = r108 * r68;
    r108 = r108 * r86;
    r23 = fma(r62, r108, r23);
    r23 = fma(r4, r148, r23);
    r23 = fma(r91, r83, r23);
    r23 = fma(r91, r71, r23);
    r23 = fma(r105, r133, r23);
    r108 = r2 * r23;
    r97 = r7 * r7;
    r97 = r97 * r56;
    r97 = r97 * r56;
    r97 = r97 * r127;
    r97 = r97 * r105;
    r97 = r97 * r43;
    r139 = r60 * r129;
    r139 = r139 * r113;
    r139 = r139 * r109;
    r139 = fma(r106, r139, r88 * r97);
    r97 = r7 * r122;
    r97 = r97 * r94;
    r97 = r97 * r43;
    r139 = fma(r59, r97, r139);
    r126 = r7 * r124;
    r126 = r126 * r129;
    r139 = fma(r117, r126, r139);
    r139 = r139 + r42;
    r139 = fma(r4, r139, r65 * r25);
    r25 = r21 * r7;
    r25 = r25 * r78;
    r25 = r25 * r105;
    r25 = r25 * r69;
    r139 = fma(r53, r25, r139);
    r42 = r21 * r7;
    r42 = r42 * r105;
    r42 = r42 * r69;
    r139 = fma(r53, r42, r139);
    r126 = r5 * r129;
    r139 = fma(r135, r126, r139);
    r97 = r84 * r129;
    r97 = r97 * r30;
    r97 = r97 * r70;
    r139 = fma(r113, r97, r139);
    r92 = r7 * r8;
    r139 = fma(r71, r92, r139);
    r98 = r84 * r78;
    r98 = r98 * r129;
    r98 = r98 * r30;
    r98 = r98 * r70;
    r139 = fma(r113, r98, r139);
    r9 = r5 * r7;
    r9 = r9 * r56;
    r9 = r9 * r56;
    r9 = r9 * r102;
    r9 = r9 * r105;
    r9 = r9 * r88;
    r139 = fma(r62, r9, r139);
    r114 = r5 * r94;
    r114 = r114 * r62;
    r139 = fma(r74, r114, r139);
    r139 = fma(r94, r83, r139);
    r139 = fma(r91, r121, r139);
    r139 = fma(r129, r138, r139);
    r139 = fma(r94, r71, r139);
    r139 = fma(r5, r148, r139);
    r139 = fma(r129, r128, r139);
    r114 = r3 * r139;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             4 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r108,
                                             r114);
    r114 = r37 * r122;
    r114 = r114 * r59;
    r108 = r6 * r6;
    r9 = r10 * r33;
    r9 = r9 * r7;
    r98 = r20 * r110;
    r98 = fma(r111, r98, r39 * r9);
    r9 = r20 * r6;
    r9 = r9 * r6;
    r98 = fma(r110, r9, r98);
    r92 = r10 * r37;
    r92 = r92 * r6;
    r98 = fma(r39, r92, r98);
    r108 = r108 * r124;
    r108 = r108 * r98;
    r108 = r108 * r86;
    r108 = r108 * r116;
    r108 = fma(r59, r108, r62 * r114);
    r114 = r60 * r98;
    r108 = fma(r115, r114, r108);
    r92 = r20 * r127;
    r92 = r92 * r88;
    r92 = r92 * r63;
    r108 = fma(r112, r92, r108);
    r9 = r21 * r7;
    r9 = r9 * r98;
    r9 = fma(r117, r9, r33 * r118);
    r148 = r98 * r113;
    r148 = r148 * r109;
    r9 = fma(r106, r148, r9);
    r9 = fma(r20, r36, r9);
    r108 = r108 + r9;
    r92 = r37 * r62;
    r114 = r21 * r6;
    r114 = r114 * r6;
    r114 = r114 * r98;
    r114 = r114 * r86;
    r114 = r114 * r116;
    r114 = fma(r59, r114, r74 * r92);
    r92 = r20 * r63;
    r114 = fma(r87, r92, r114);
    r114 = fma(r98, r115, r114);
    r9 = r9 + r114;
    r108 = fma(r64, r9, r5 * r108);
    r92 = r6 * r84;
    r92 = r92 * r78;
    r92 = r92 * r98;
    r92 = r92 * r86;
    r92 = r92 * r30;
    r108 = fma(r70, r92, r108);
    r148 = r4 * r98;
    r108 = fma(r135, r148, r108);
    r97 = r6 * r84;
    r97 = r97 * r98;
    r97 = r97 * r86;
    r97 = r97 * r30;
    r108 = fma(r70, r97, r108);
    r126 = r56 * r78;
    r126 = r126 * r89;
    r126 = r126 * r98;
    r126 = r126 * r68;
    r126 = r126 * r86;
    r108 = fma(r62, r126, r108);
    r42 = r56 * r89;
    r42 = r42 * r98;
    r42 = r42 * r68;
    r42 = r42 * r86;
    r108 = fma(r62, r42, r108);
    r25 = r4 * r98;
    r108 = fma(r131, r25, r108);
    r119 = r76 * r10;
    r119 = r119 * r48;
    r119 = fma(r75, r9, r9 * r119);
    r119 = fma(r9, r80, r119);
    r119 = fma(r9, r79, r119);
    r90 = r6 * r119;
    r108 = fma(r71, r90, r108);
    r26 = r21 * r20;
    r26 = r26 * r6;
    r26 = r26 * r78;
    r26 = r26 * r69;
    r108 = fma(r53, r26, r108);
    r22 = r21 * r20;
    r22 = r22 * r6;
    r22 = r22 * r69;
    r108 = fma(r53, r22, r108);
    r142 = r4 * r33;
    r142 = r142 * r62;
    r108 = fma(r74, r142, r108);
    r143 = r4 * r37;
    r108 = fma(r118, r143, r108);
    r108 = fma(r37, r83, r108);
    r108 = fma(r20, r133, r108);
    r108 = fma(r37, r71, r108);
    r143 = r2 * r108;
    r142 = r33 * r7;
    r142 = r142 * r122;
    r142 = r142 * r43;
    r22 = r7 * r124;
    r22 = r22 * r98;
    r22 = fma(r117, r22, r59 * r142);
    r142 = r20 * r7;
    r142 = r142 * r7;
    r142 = r142 * r56;
    r142 = r142 * r56;
    r142 = r142 * r127;
    r142 = r142 * r43;
    r22 = fma(r88, r142, r22);
    r26 = r60 * r98;
    r26 = r26 * r113;
    r26 = r26 * r109;
    r22 = fma(r106, r26, r22);
    r22 = r22 + r114;
    r9 = fma(r65, r9, r4 * r22);
    r22 = r84 * r98;
    r22 = r22 * r30;
    r22 = r22 * r70;
    r9 = fma(r113, r22, r9);
    r114 = r21 * r20;
    r114 = r114 * r7;
    r114 = r114 * r78;
    r114 = r114 * r69;
    r9 = fma(r53, r114, r9);
    r26 = r5 * r98;
    r9 = fma(r135, r26, r9);
    r142 = r21 * r20;
    r142 = r142 * r7;
    r142 = r142 * r69;
    r9 = fma(r53, r142, r9);
    r90 = r5 * r98;
    r9 = fma(r131, r90, r9);
    r25 = r5 * r20;
    r25 = r25 * r7;
    r25 = r25 * r56;
    r25 = r25 * r56;
    r25 = r25 * r102;
    r25 = r25 * r88;
    r9 = fma(r62, r25, r9);
    r42 = r7 * r119;
    r9 = fma(r71, r42, r9);
    r126 = r84 * r78;
    r126 = r126 * r98;
    r126 = r126 * r30;
    r126 = r126 * r70;
    r9 = fma(r113, r126, r9);
    r97 = r5 * r33;
    r97 = r97 * r62;
    r9 = fma(r74, r97, r9);
    r9 = fma(r33, r71, r9);
    r9 = fma(r98, r138, r9);
    r9 = fma(r33, r83, r9);
    r9 = fma(r98, r128, r9);
    r9 = fma(r37, r121, r9);
    r97 = r3 * r9;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r143, r97);
    r97 = r34 * r110;
    r143 = r10 * r38;
    r143 = r143 * r7;
    r143 = fma(r39, r143, r111 * r97);
    r97 = r34 * r6;
    r97 = r97 * r6;
    r143 = fma(r110, r97, r143);
    r126 = r10 * r55;
    r126 = r126 * r6;
    r143 = fma(r39, r126, r143);
    r126 = r60 * r143;
    r97 = r34 * r127;
    r97 = r97 * r88;
    r97 = r97 * r63;
    r97 = fma(r112, r97, r115 * r126);
    r126 = r6 * r6;
    r126 = r126 * r124;
    r126 = r126 * r143;
    r126 = r126 * r86;
    r126 = r126 * r116;
    r97 = fma(r59, r126, r97);
    r42 = r55 * r122;
    r42 = r42 * r59;
    r97 = fma(r62, r42, r97);
    r25 = fma(r38, r118, r34 * r36);
    r90 = r143 * r113;
    r90 = r90 * r109;
    r25 = fma(r106, r90, r25);
    r142 = r21 * r7;
    r142 = r142 * r143;
    r25 = fma(r117, r142, r25);
    r97 = r97 + r25;
    r42 = r34 * r63;
    r42 = fma(r87, r42, r143 * r115);
    r126 = r21 * r6;
    r126 = r126 * r6;
    r126 = r126 * r143;
    r126 = r126 * r86;
    r126 = r126 * r116;
    r42 = fma(r59, r126, r42);
    r142 = r55 * r62;
    r42 = fma(r74, r142, r42);
    r25 = r25 + r42;
    r97 = fma(r64, r25, r5 * r97);
    r142 = r4 * r38;
    r142 = r142 * r62;
    r97 = fma(r74, r142, r97);
    r126 = r143 * r135;
    r90 = r6 * r84;
    r90 = r90 * r78;
    r90 = r90 * r143;
    r90 = r90 * r86;
    r90 = r90 * r30;
    r97 = fma(r70, r90, r97);
    r26 = r21 * r34;
    r26 = r26 * r6;
    r26 = r26 * r69;
    r97 = fma(r53, r26, r97);
    r114 = r4 * r143;
    r97 = fma(r131, r114, r97);
    r22 = r6 * r84;
    r22 = r22 * r143;
    r22 = r22 * r86;
    r22 = r22 * r30;
    r97 = fma(r70, r22, r97);
    r148 = r56 * r89;
    r148 = r148 * r143;
    r148 = r148 * r68;
    r148 = r148 * r86;
    r97 = fma(r62, r148, r97);
    r92 = r76 * r10;
    r92 = r92 * r48;
    r92 = fma(r75, r25, r25 * r92);
    r92 = fma(r25, r79, r92);
    r92 = fma(r25, r80, r92);
    r144 = r6 * r92;
    r97 = fma(r71, r144, r97);
    r145 = r56 * r78;
    r145 = r145 * r89;
    r145 = r145 * r143;
    r145 = r145 * r68;
    r145 = r145 * r86;
    r97 = fma(r62, r145, r97);
    r147 = r21 * r34;
    r147 = r147 * r6;
    r147 = r147 * r78;
    r147 = r147 * r69;
    r97 = fma(r53, r147, r97);
    r130 = r4 * r55;
    r97 = fma(r118, r130, r97);
    r97 = fma(r4, r126, r97);
    r97 = fma(r55, r71, r97);
    r97 = fma(r34, r133, r97);
    r97 = fma(r55, r83, r97);
    r130 = r2 * r97;
    r147 = r34 * r7;
    r147 = r147 * r7;
    r147 = r147 * r56;
    r147 = r147 * r56;
    r147 = r147 * r127;
    r147 = r147 * r43;
    r145 = r38 * r7;
    r145 = r145 * r122;
    r145 = r145 * r43;
    r145 = fma(r59, r145, r88 * r147);
    r147 = r60 * r143;
    r147 = r147 * r113;
    r147 = r147 * r109;
    r145 = fma(r106, r147, r145);
    r144 = r7 * r124;
    r144 = r144 * r143;
    r145 = fma(r117, r144, r145);
    r145 = r145 + r42;
    r145 = fma(r4, r145, r65 * r25);
    r25 = r21 * r34;
    r25 = r25 * r7;
    r25 = r25 * r69;
    r145 = fma(r53, r25, r145);
    r42 = r5 * r38;
    r42 = r42 * r62;
    r145 = fma(r74, r42, r145);
    r144 = r84 * r143;
    r144 = r144 * r30;
    r144 = r144 * r70;
    r145 = fma(r113, r144, r145);
    r147 = r7 * r92;
    r145 = fma(r71, r147, r145);
    r148 = r5 * r34;
    r148 = r148 * r7;
    r148 = r148 * r56;
    r148 = r148 * r56;
    r148 = r148 * r102;
    r148 = r148 * r88;
    r145 = fma(r62, r148, r145);
    r22 = r5 * r143;
    r145 = fma(r131, r22, r145);
    r114 = r84 * r78;
    r114 = r114 * r143;
    r114 = r114 * r30;
    r114 = r114 * r70;
    r145 = fma(r113, r114, r145);
    r26 = r21 * r34;
    r26 = r26 * r7;
    r26 = r26 * r78;
    r26 = r26 * r69;
    r145 = fma(r53, r26, r145);
    r145 = fma(r38, r83, r145);
    r145 = fma(r5, r126, r145);
    r145 = fma(r143, r138, r145);
    r145 = fma(r143, r128, r145);
    r145 = fma(r38, r71, r145);
    r145 = fma(r55, r121, r145);
    r26 = r3 * r145;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r130, r26);
    r26 = r6 * r6;
    r130 = r10 * r41;
    r130 = r130 * r7;
    r114 = r32 * r110;
    r114 = fma(r111, r114, r39 * r130);
    r130 = r10 * r54;
    r130 = r130 * r6;
    r114 = fma(r39, r130, r114);
    r22 = r32 * r6;
    r22 = r22 * r6;
    r114 = fma(r110, r22, r114);
    r26 = r26 * r124;
    r26 = r26 * r114;
    r26 = r26 * r86;
    r26 = r26 * r116;
    r22 = r60 * r114;
    r22 = fma(r115, r22, r59 * r26);
    r26 = r54 * r122;
    r26 = r26 * r59;
    r22 = fma(r62, r26, r22);
    r130 = r32 * r127;
    r130 = r130 * r88;
    r130 = r130 * r63;
    r22 = fma(r112, r130, r22);
    r148 = fma(r41, r118, r32 * r36);
    r147 = r114 * r113;
    r147 = r147 * r109;
    r148 = fma(r106, r147, r148);
    r144 = r21 * r7;
    r144 = r144 * r114;
    r148 = fma(r117, r144, r148);
    r22 = r22 + r148;
    r130 = r21 * r6;
    r130 = r130 * r6;
    r130 = r130 * r114;
    r130 = r130 * r86;
    r130 = r130 * r116;
    r130 = fma(r114, r115, r59 * r130);
    r26 = r54 * r62;
    r130 = fma(r74, r26, r130);
    r144 = r32 * r63;
    r130 = fma(r87, r144, r130);
    r148 = r148 + r130;
    r22 = fma(r64, r148, r5 * r22);
    r144 = r4 * r54;
    r22 = fma(r118, r144, r22);
    r26 = r4 * r41;
    r26 = r26 * r62;
    r22 = fma(r74, r26, r22);
    r147 = r6 * r84;
    r147 = r147 * r78;
    r147 = r147 * r114;
    r147 = r147 * r86;
    r147 = r147 * r30;
    r22 = fma(r70, r147, r22);
    r126 = r6 * r84;
    r126 = r126 * r114;
    r126 = r126 * r86;
    r126 = r126 * r30;
    r22 = fma(r70, r126, r22);
    r42 = r76 * r10;
    r42 = r42 * r48;
    r42 = fma(r148, r42, r75 * r148);
    r42 = fma(r148, r80, r42);
    r42 = fma(r148, r79, r42);
    r25 = r6 * r42;
    r22 = fma(r71, r25, r22);
    r90 = r4 * r114;
    r22 = fma(r135, r90, r22);
    r142 = r21 * r32;
    r142 = r142 * r6;
    r142 = r142 * r69;
    r22 = fma(r53, r142, r22);
    r132 = r56 * r78;
    r132 = r132 * r89;
    r132 = r132 * r114;
    r132 = r132 * r68;
    r132 = r132 * r86;
    r22 = fma(r62, r132, r22);
    r146 = r4 * r114;
    r22 = fma(r131, r146, r22);
    r95 = r21 * r32;
    r95 = r95 * r6;
    r95 = r95 * r78;
    r95 = r95 * r69;
    r22 = fma(r53, r95, r22);
    r85 = r56 * r89;
    r85 = r85 * r114;
    r85 = r85 * r68;
    r85 = r85 * r86;
    r22 = fma(r62, r85, r22);
    r22 = fma(r54, r83, r22);
    r22 = fma(r32, r133, r22);
    r22 = fma(r54, r71, r22);
    r85 = r2 * r22;
    r95 = r32 * r7;
    r95 = r95 * r7;
    r95 = r95 * r56;
    r95 = r95 * r56;
    r95 = r95 * r127;
    r95 = r95 * r43;
    r146 = r41 * r7;
    r146 = r146 * r122;
    r146 = r146 * r43;
    r146 = fma(r59, r146, r88 * r95);
    r95 = r60 * r114;
    r95 = r95 * r113;
    r95 = r95 * r109;
    r146 = fma(r106, r95, r146);
    r132 = r7 * r124;
    r132 = r132 * r114;
    r146 = fma(r117, r132, r146);
    r146 = r146 + r130;
    r146 = fma(r4, r146, r65 * r148);
    r148 = r21 * r32;
    r148 = r148 * r7;
    r148 = r148 * r69;
    r146 = fma(r53, r148, r146);
    r130 = r21 * r32;
    r130 = r130 * r7;
    r130 = r130 * r78;
    r130 = r130 * r69;
    r146 = fma(r53, r130, r146);
    r132 = r84 * r78;
    r132 = r132 * r114;
    r132 = r132 * r30;
    r132 = r132 * r70;
    r146 = fma(r113, r132, r146);
    r95 = r5 * r32;
    r95 = r95 * r7;
    r95 = r95 * r56;
    r95 = r95 * r56;
    r95 = r95 * r102;
    r95 = r95 * r88;
    r146 = fma(r62, r95, r146);
    r142 = r5 * r41;
    r142 = r142 * r62;
    r146 = fma(r74, r142, r146);
    r90 = r5 * r114;
    r146 = fma(r135, r90, r146);
    r25 = r84 * r114;
    r25 = r25 * r30;
    r25 = r25 * r70;
    r146 = fma(r113, r25, r146);
    r126 = r7 * r42;
    r146 = fma(r71, r126, r146);
    r147 = r5 * r114;
    r146 = fma(r131, r147, r146);
    r146 = fma(r114, r128, r146);
    r146 = fma(r41, r71, r146);
    r146 = fma(r54, r121, r146);
    r146 = fma(r41, r83, r146);
    r146 = fma(r114, r138, r146);
    r147 = r3 * r146;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             10 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r85,
                                             r147);
    r147 = r3 * r21;
    r147 = r147 * r1;
    r85 = r21 * r0;
    r126 = r2 * r85;
    r147 = fma(r120, r126, r134 * r147);
    r25 = r3 * r21;
    r25 = r25 * r1;
    r25 = fma(r107, r126, r125 * r25);
    WriteSum2<double, double>((double*)inout_shared, r147, r25);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r3 * r21;
    r25 = r25 * r1;
    r25 = fma(r23, r126, r139 * r25);
    r147 = r3 * r21;
    r147 = r147 * r1;
    r147 = fma(r108, r126, r9 * r147);
    WriteSum2<double, double>((double*)inout_shared, r25, r147);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r147 = r3 * r21;
    r147 = r147 * r1;
    r147 = fma(r97, r126, r145 * r147);
    r25 = r3 * r21;
    r25 = r25 * r1;
    r25 = fma(r22, r126, r146 * r25);
    WriteSum2<double, double>((double*)inout_shared, r147, r25);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r120 * r120;
    r147 = r2 * r2;
    r90 = r134 * r134;
    r142 = r3 * r3;
    r90 = fma(r142, r90, r147 * r25);
    r25 = r107 * r107;
    r95 = r125 * r125;
    r95 = fma(r142, r95, r147 * r25);
    WriteSum2<double, double>((double*)inout_shared, r90, r95);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r95 = r23 * r23;
    r90 = r139 * r139;
    r90 = fma(r142, r90, r147 * r95);
    r95 = r9 * r9;
    r25 = r108 * r108;
    r25 = fma(r147, r25, r142 * r95);
    WriteSum2<double, double>((double*)inout_shared, r90, r25);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r97 * r97;
    r90 = r145 * r145;
    r90 = fma(r142, r90, r147 * r25);
    r25 = r146 * r146;
    r95 = r22 * r22;
    r95 = fma(r147, r95, r142 * r25);
    WriteSum2<double, double>((double*)inout_shared, r90, r95);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r95 = r134 * r125;
    r90 = r120 * r107;
    r90 = fma(r147, r90, r142 * r95);
    r95 = r120 * r23;
    r25 = r134 * r139;
    r25 = fma(r142, r25, r147 * r95);
    WriteSum2<double, double>((double*)inout_shared, r90, r25);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r120 * r108;
    r90 = r134 * r9;
    r90 = fma(r142, r90, r147 * r25);
    r25 = r120 * r97;
    r95 = r134 * r145;
    r95 = fma(r142, r95, r147 * r25);
    WriteSum2<double, double>((double*)inout_shared, r90, r95);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r95 = r134 * r146;
    r90 = r120 * r22;
    r90 = fma(r147, r90, r142 * r95);
    r95 = r125 * r139;
    r25 = r107 * r23;
    r25 = fma(r147, r25, r142 * r95);
    WriteSum2<double, double>((double*)inout_shared, r90, r25);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r107 * r108;
    r90 = r125 * r9;
    r90 = fma(r142, r90, r147 * r25);
    r25 = r107 * r97;
    r95 = r125 * r145;
    r95 = fma(r142, r95, r147 * r25);
    WriteSum2<double, double>((double*)inout_shared, r90, r95);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r95 = r125 * r146;
    r90 = r107 * r22;
    r90 = fma(r147, r90, r142 * r95);
    r95 = r139 * r9;
    r25 = r23 * r108;
    r25 = fma(r147, r25, r142 * r95);
    WriteSum2<double, double>((double*)inout_shared, r90, r25);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r23 * r97;
    r90 = r139 * r145;
    r90 = fma(r142, r90, r147 * r25);
    r25 = r139 * r146;
    r95 = r23 * r22;
    r95 = fma(r147, r95, r142 * r25);
    WriteSum2<double, double>((double*)inout_shared, r90, r95);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r95 = r108 * r97;
    r90 = r9 * r145;
    r90 = fma(r142, r90, r147 * r95);
    r95 = r9 * r146;
    r25 = r108 * r22;
    r25 = fma(r147, r25, r142 * r95);
    WriteSum2<double, double>((double*)inout_shared, r90, r25);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r145 * r146;
    r90 = r97 * r22;
    r90 = fma(r147, r90, r142 * r25);
    WriteSum1<double, double>((double*)inout_shared, r90);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        0 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r67,
        r66);
    r90 = r2 * r6;
    r90 = r90 * r48;
    r90 = r90 * r71;
    r25 = r3 * r7;
    r25 = r25 * r48;
    r25 = r25 * r71;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        2 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r90,
        r25);
    r25 = r2 * r6;
    r25 = r25 * r71;
    r25 = r25 * r77;
    r90 = r3 * r7;
    r90 = r90 * r71;
    r90 = r90 * r77;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        4 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r25,
        r90);
    r90 = r3 * r73;
    r25 = r2 * r7;
    r25 = r25 * r62;
    r25 = r25 * r74;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        6 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r25,
        r90);
    r90 = r2 * r61;
    r25 = r3 * r7;
    r25 = r25 * r62;
    r25 = r25 * r74;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        8 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r90,
        r25);
    r25 = r2 * r6;
    r25 = r25 * r71;
    r25 = r25 * r82;
    r90 = r3 * r7;
    r90 = r90 * r71;
    r90 = r90 * r82;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        10 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r25,
        r90);
    r90 = r2 * r6;
    r90 = r90 * r71;
    r90 = r90 * r81;
    r25 = r3 * r7;
    r25 = r25 * r71;
    r25 = r25 * r81;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        12 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r90,
        r25);
    r25 = r2 * r48;
    r90 = r3 * r48;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        14 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r25,
        r90);
    r90 = r21 * r66;
    r90 = r90 * r1;
    r85 = r67 * r85;
    WriteSum2<double, double>((double*)inout_shared, r85, r90);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            0 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r90 = r3 * r21;
    r90 = r90 * r7;
    r90 = r90 * r48;
    r90 = r90 * r1;
    r85 = r126 * r72;
    r90 = fma(r48, r85, r71 * r90);
    r25 = r3 * r21;
    r25 = r25 * r7;
    r25 = r25 * r1;
    r25 = r25 * r71;
    r25 = fma(r77, r85, r77 * r25);
    WriteSum2<double, double>((double*)inout_shared, r90, r25);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            2 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r3 * r21;
    r25 = r25 * r73;
    r90 = r2 * r35;
    r90 = r90 * r7;
    r90 = r90 * r0;
    r90 = r90 * r59;
    r90 = fma(r62, r90, r1 * r25);
    r25 = r3 * r35;
    r25 = r25 * r7;
    r25 = r25 * r1;
    r25 = r25 * r59;
    r25 = fma(r62, r25, r61 * r126);
    WriteSum2<double, double>((double*)inout_shared, r90, r25);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            4 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r3 * r21;
    r25 = r25 * r7;
    r25 = r25 * r1;
    r25 = r25 * r71;
    r25 = fma(r82, r85, r82 * r25);
    r90 = r3 * r21;
    r90 = r90 * r7;
    r90 = r90 * r1;
    r90 = r90 * r71;
    r85 = fma(r81, r85, r81 * r90);
    WriteSum2<double, double>((double*)inout_shared, r25, r85);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            6 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = r3 * r21;
    r85 = r85 * r48;
    r85 = r85 * r1;
    r25 = r48 * r126;
    WriteSum2<double, double>((double*)inout_shared, r25, r85);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            8 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = r67 * r67;
    r25 = r66 * r66;
    WriteSum2<double, double>((double*)inout_shared, r85, r25);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            0 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r59 * r77;
    r25 = r25 * r147;
    r85 = r7 * r59;
    r85 = r85 * r77;
    r85 = r85 * r142;
    r85 = fma(r106, r85, r63 * r25);
    r25 = r59 * r147;
    r25 = r25 * r63;
    r90 = r7 * r59;
    r90 = r90 * r142;
    r90 = r90 * r106;
    r90 = fma(r81, r90, r81 * r25);
    WriteSum2<double, double>((double*)inout_shared, r85, r90);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            2 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = r6 * r6;
    r57 = r31 * r57;
    r57 = 1.0 / r57;
    r49 = r47 * r49;
    r49 = 1.0 / r49;
    r85 = r85 * r56;
    r85 = r85 * r56;
    r85 = r85 * r137;
    r85 = r85 * r57;
    r85 = r85 * r49;
    r85 = r85 * r112;
    r85 = r85 * r111;
    r49 = r73 * r142;
    r57 = fma(r73, r49, r147 * r85);
    r137 = r61 * r147;
    r85 = fma(r61, r137, r142 * r85);
    WriteSum2<double, double>((double*)inout_shared, r57, r85);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            4 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = r59 * r147;
    r57 = r82 * r82;
    r85 = r85 * r63;
    r47 = r7 * r59;
    r47 = r47 * r142;
    r47 = r47 * r106;
    r85 = fma(r57, r47, r57 * r85);
    r31 = r81 * r81;
    r25 = r59 * r147;
    r25 = r25 * r63;
    r31 = fma(r47, r31, r31 * r25);
    WriteSum2<double, double>((double*)inout_shared, r85, r31);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            6 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r77 * r147;
    r0 = r77 * r142;
    WriteSum2<double, double>((double*)inout_shared, r31, r0);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            8 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
    r31 = r2 * r6;
    r31 = r31 * r48;
    r31 = r31 * r67;
    r31 = r31 * r71;
    WriteSum2<double, double>((double*)inout_shared, r0, r31);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            0 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r2 * r6;
    r31 = r31 * r67;
    r31 = r31 * r71;
    r31 = r31 * r77;
    r95 = r2 * r7;
    r95 = r95 * r67;
    r95 = r95 * r62;
    r95 = r95 * r74;
    WriteSum2<double, double>((double*)inout_shared, r31, r95);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            2 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = r2 * r61;
    r61 = r61 * r67;
    r95 = r2 * r6;
    r95 = r95 * r67;
    r95 = r95 * r71;
    r95 = r95 * r82;
    WriteSum2<double, double>((double*)inout_shared, r61, r95);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            4 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r95 = r2 * r48;
    r95 = r95 * r67;
    r61 = r2 * r6;
    r61 = r61 * r67;
    r61 = r61 * r71;
    r61 = r61 * r81;
    WriteSum2<double, double>((double*)inout_shared, r61, r95);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            6 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r95 = r3 * r7;
    r95 = r95 * r48;
    r95 = r95 * r66;
    r95 = r95 * r71;
    WriteSum2<double, double>((double*)inout_shared, r0, r95);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            8 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = r3 * r73;
    r73 = r73 * r66;
    r95 = r3 * r7;
    r95 = r95 * r66;
    r95 = r95 * r71;
    r95 = r95 * r77;
    WriteSum2<double, double>((double*)inout_shared, r95, r73);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            10 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = r3 * r7;
    r73 = r73 * r66;
    r73 = r73 * r62;
    r73 = r73 * r74;
    r95 = r3 * r7;
    r95 = r95 * r66;
    r95 = r95 * r71;
    r95 = r95 * r82;
    WriteSum2<double, double>((double*)inout_shared, r73, r95);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            12 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r95 = r3 * r7;
    r95 = r95 * r66;
    r95 = r95 * r71;
    r95 = r95 * r81;
    WriteSum2<double, double>((double*)inout_shared, r95, r0);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            14 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r3 * r48;
    r0 = r0 * r66;
    r66 = r59 * r82;
    r66 = r66 * r147;
    r95 = r7 * r59;
    r95 = r95 * r82;
    r95 = r95 * r142;
    r95 = fma(r106, r95, r63 * r66);
    WriteSum2<double, double>((double*)inout_shared, r0, r95);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            16 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r95 = r7 * r49;
    r0 = r71 * r95;
    r66 = r10 * r6;
    r66 = r66 * r6;
    r66 = r66 * r7;
    r66 = r66 * r56;
    r66 = r66 * r48;
    r66 = r66 * r88;
    r66 = r66 * r116;
    r66 = r66 * r147;
    r66 = fma(r112, r66, r48 * r0);
    r73 = r48 * r137;
    r61 = r10 * r6;
    r61 = r61 * r56;
    r61 = r61 * r48;
    r61 = r61 * r88;
    r61 = r61 * r116;
    r61 = r61 * r142;
    r61 = r61 * r112;
    r61 = fma(r111, r61, r72 * r73);
    WriteSum2<double, double>((double*)inout_shared, r66, r61);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            18 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = r59 * r147;
    r66 = r48 * r81;
    r61 = r61 * r63;
    r61 = fma(r66, r47, r66 * r61);
    WriteSum2<double, double>((double*)inout_shared, r90, r61);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            20 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r90 = r6 * r71;
    r90 = r90 * r77;
    r90 = r90 * r147;
    r73 = r7 * r71;
    r73 = r73 * r77;
    r73 = r73 * r142;
    WriteSum2<double, double>((double*)inout_shared, r90, r73);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            22 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = r10 * r6;
    r73 = r73 * r6;
    r73 = r73 * r7;
    r73 = r73 * r56;
    r73 = r73 * r88;
    r73 = r73 * r116;
    r73 = r73 * r77;
    r73 = r73 * r147;
    r73 = fma(r112, r73, r77 * r0);
    r90 = r77 * r137;
    r67 = r10 * r6;
    r67 = r67 * r56;
    r67 = r67 * r88;
    r67 = r67 * r116;
    r67 = r67 * r77;
    r67 = r67 * r142;
    r67 = r67 * r112;
    r67 = fma(r111, r67, r72 * r90);
    WriteSum2<double, double>((double*)inout_shared, r73, r67);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            24 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r61, r85);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            26 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = r6 * r71;
    r85 = r85 * r82;
    r85 = r85 * r147;
    r61 = r7 * r71;
    r61 = r61 * r82;
    r61 = r61 * r142;
    WriteSum2<double, double>((double*)inout_shared, r85, r61);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            28 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = r7 * r137;
    r85 = r62 * r74;
    r85 = fma(r95, r85, r85 * r61);
    r95 = r10 * r6;
    r95 = r95 * r6;
    r95 = r95 * r7;
    r95 = r95 * r56;
    r95 = r95 * r88;
    r95 = r95 * r116;
    r95 = r95 * r82;
    r95 = r95 * r147;
    r95 = fma(r112, r95, r82 * r0);
    WriteSum2<double, double>((double*)inout_shared, r85, r95);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            30 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r95 = r7 * r48;
    r95 = r95 * r62;
    r95 = r95 * r74;
    r95 = r95 * r147;
    r85 = r10 * r6;
    r85 = r85 * r6;
    r85 = r85 * r7;
    r85 = r85 * r56;
    r85 = r85 * r88;
    r85 = r85 * r116;
    r85 = r85 * r147;
    r85 = r85 * r112;
    r85 = fma(r81, r85, r81 * r0);
    WriteSum2<double, double>((double*)inout_shared, r85, r95);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            32 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r48 * r49;
    r95 = r82 * r137;
    r85 = r10 * r6;
    r85 = r85 * r56;
    r85 = r85 * r88;
    r85 = r85 * r116;
    r85 = r85 * r82;
    r85 = r85 * r142;
    r85 = r85 * r112;
    r85 = fma(r111, r85, r72 * r95);
    WriteSum2<double, double>((double*)inout_shared, r49, r85);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            34 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = r48 * r137;
    r49 = r81 * r137;
    r95 = r10 * r6;
    r95 = r95 * r56;
    r95 = r95 * r88;
    r95 = r95 * r116;
    r95 = r95 * r142;
    r95 = r95 * r112;
    r95 = r95 * r81;
    r95 = fma(r111, r95, r72 * r49);
    WriteSum2<double, double>((double*)inout_shared, r95, r85);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            36 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = r7 * r48;
    r85 = r85 * r62;
    r85 = r85 * r74;
    r85 = r85 * r142;
    r57 = r48 * r57;
    r47 = fma(r57, r47, r57 * r25);
    WriteSum2<double, double>((double*)inout_shared, r85, r47);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            38 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = r6 * r71;
    r47 = r47 * r147;
    r47 = r47 * r81;
    r85 = r7 * r71;
    r85 = r85 * r142;
    r85 = r85 * r81;
    WriteSum2<double, double>((double*)inout_shared, r47, r85);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            40 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = r147 * r66;
    r85 = r85 * r72;
    r72 = r7 * r71;
    r72 = r72 * r142;
    r72 = r72 * r66;
    WriteSum2<double, double>((double*)inout_shared, r85, r72);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            42 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r21 * r7;
    r85 = r10 * r51;
    r85 = r85 * r6;
    r66 = r58 * r110;
    r66 = fma(r111, r66, r39 * r85);
    r85 = r58 * r6;
    r85 = r85 * r6;
    r66 = fma(r110, r85, r66);
    r47 = r10 * r28;
    r47 = r47 * r7;
    r66 = fma(r39, r47, r66);
    r72 = r72 * r66;
    r72 = fma(r117, r72, r58 * r36);
    r47 = r66 * r113;
    r47 = r47 * r109;
    r72 = fma(r106, r47, r72);
    r72 = fma(r28, r118, r72);
    r47 = r58 * r63;
    r47 = fma(r66, r115, r87 * r47);
    r85 = r51 * r62;
    r47 = fma(r74, r85, r47);
    r57 = r21 * r6;
    r57 = r57 * r6;
    r57 = r57 * r66;
    r57 = r57 * r86;
    r57 = r57 * r116;
    r47 = fma(r59, r57, r47);
    r57 = r72 + r47;
    r85 = r58 * r127;
    r85 = r85 * r88;
    r85 = r85 * r63;
    r25 = r60 * r66;
    r25 = fma(r115, r25, r112 * r85);
    r85 = r51 * r122;
    r85 = r85 * r59;
    r25 = fma(r62, r85, r25);
    r95 = r6 * r6;
    r95 = r95 * r124;
    r95 = r95 * r66;
    r95 = r95 * r86;
    r95 = r95 * r116;
    r25 = fma(r59, r95, r25);
    r25 = r25 + r72;
    r25 = fma(r5, r25, r64 * r57);
    r72 = r6 * r84;
    r72 = r72 * r78;
    r72 = r72 * r66;
    r72 = r72 * r86;
    r72 = r72 * r30;
    r25 = fma(r70, r72, r25);
    r95 = r21 * r58;
    r95 = r95 * r6;
    r95 = r95 * r78;
    r95 = r95 * r69;
    r25 = fma(r53, r95, r25);
    r85 = r4 * r66;
    r25 = fma(r135, r85, r25);
    r49 = r4 * r66;
    r25 = fma(r131, r49, r25);
    r0 = r6 * r84;
    r0 = r0 * r66;
    r0 = r0 * r86;
    r0 = r0 * r30;
    r25 = fma(r70, r0, r25);
    r61 = r4 * r51;
    r25 = fma(r118, r61, r25);
    r67 = r21 * r58;
    r67 = r67 * r6;
    r67 = r67 * r69;
    r25 = fma(r53, r67, r25);
    r73 = r56 * r78;
    r73 = r73 * r89;
    r73 = r73 * r66;
    r73 = r73 * r68;
    r73 = r73 * r86;
    r25 = fma(r62, r73, r25);
    r90 = r76 * r10;
    r90 = r90 * r48;
    r90 = fma(r75, r57, r57 * r90);
    r90 = fma(r57, r80, r90);
    r90 = fma(r57, r79, r90);
    r31 = r6 * r90;
    r25 = fma(r71, r31, r25);
    r132 = r4 * r28;
    r132 = r132 * r62;
    r25 = fma(r74, r132, r25);
    r130 = r56 * r89;
    r130 = r130 * r66;
    r130 = r130 * r68;
    r130 = r130 * r86;
    r25 = fma(r62, r130, r25);
    r25 = fma(r51, r71, r25);
    r25 = fma(r58, r133, r25);
    r25 = fma(r51, r83, r25);
    r130 = r2 * r25;
    r132 = r58 * r7;
    r132 = r132 * r7;
    r132 = r132 * r56;
    r132 = r132 * r56;
    r132 = r132 * r127;
    r132 = r132 * r43;
    r31 = r7 * r124;
    r31 = r31 * r66;
    r31 = fma(r117, r31, r88 * r132);
    r132 = r60 * r66;
    r132 = r132 * r113;
    r132 = r132 * r109;
    r31 = fma(r106, r132, r31);
    r73 = r28 * r7;
    r73 = r73 * r122;
    r73 = r73 * r43;
    r31 = fma(r59, r73, r31);
    r31 = r31 + r47;
    r31 = fma(r4, r31, r65 * r57);
    r57 = r7 * r90;
    r31 = fma(r71, r57, r31);
    r47 = r84 * r66;
    r47 = r47 * r30;
    r47 = r47 * r70;
    r31 = fma(r113, r47, r31);
    r73 = r21 * r58;
    r73 = r73 * r7;
    r73 = r73 * r69;
    r31 = fma(r53, r73, r31);
    r132 = r5 * r66;
    r31 = fma(r135, r132, r31);
    r67 = r5 * r66;
    r31 = fma(r131, r67, r31);
    r61 = r84 * r78;
    r61 = r61 * r66;
    r61 = r61 * r30;
    r61 = r61 * r70;
    r31 = fma(r113, r61, r31);
    r0 = r21 * r58;
    r0 = r0 * r7;
    r0 = r0 * r78;
    r0 = r0 * r69;
    r31 = fma(r53, r0, r31);
    r49 = r5 * r58;
    r49 = r49 * r7;
    r49 = r49 * r56;
    r49 = r49 * r56;
    r49 = r49 * r102;
    r49 = r49 * r88;
    r31 = fma(r62, r49, r31);
    r85 = r5 * r28;
    r85 = r85 * r62;
    r31 = fma(r74, r85, r31);
    r31 = fma(r28, r83, r31);
    r31 = fma(r51, r121, r31);
    r31 = fma(r66, r128, r31);
    r31 = fma(r28, r71, r31);
    r31 = fma(r66, r138, r31);
    r85 = r3 * r31;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r130,
                                             r85);
    r85 = r45 * r122;
    r85 = r85 * r59;
    r130 = r40 * r127;
    r130 = r130 * r88;
    r130 = r130 * r63;
    r130 = fma(r112, r130, r62 * r85);
    r85 = r6 * r6;
    r49 = r40 * r6;
    r49 = r49 * r6;
    r0 = r10 * r45;
    r0 = r0 * r6;
    r0 = fma(r39, r0, r110 * r49);
    r49 = r10 * r46;
    r49 = r49 * r7;
    r0 = fma(r39, r49, r0);
    r61 = r40 * r110;
    r0 = fma(r111, r61, r0);
    r85 = r85 * r124;
    r85 = r85 * r0;
    r85 = r85 * r86;
    r85 = r85 * r116;
    r130 = fma(r59, r85, r130);
    r61 = r6 * r6;
    r61 = r39 * r61;
    r61 = r61 * r43;
    r61 = r61 * r56;
    r61 = r61 * r86;
    r61 = r61 * r30;
    r61 = r61 * r0;
    r49 = r0 * r113;
    r49 = r49 * r109;
    r49 = fma(r106, r49, r40 * r36);
    r67 = r21 * r7;
    r67 = r67 * r0;
    r49 = fma(r117, r67, r49);
    r49 = fma(r46, r118, r49);
    r130 = fma(r60, r61, r130);
    r130 = r130 + r49;
    r85 = r45 * r62;
    r85 = fma(r74, r85, r61);
    r61 = r40 * r63;
    r85 = fma(r87, r61, r85);
    r67 = r21 * r6;
    r67 = r67 * r6;
    r67 = r67 * r0;
    r67 = r67 * r86;
    r67 = r67 * r116;
    r85 = fma(r59, r67, r85);
    r49 = r49 + r85;
    r130 = fma(r64, r49, r5 * r130);
    r67 = r21 * r40;
    r67 = r67 * r6;
    r67 = r67 * r78;
    r67 = r67 * r69;
    r130 = fma(r53, r67, r130);
    r61 = r6 * r84;
    r61 = r61 * r0;
    r61 = r61 * r86;
    r61 = r61 * r30;
    r130 = fma(r70, r61, r130);
    r132 = r6 * r84;
    r132 = r132 * r78;
    r132 = r132 * r0;
    r132 = r132 * r86;
    r132 = r132 * r30;
    r130 = fma(r70, r132, r130);
    r73 = r76 * r10;
    r73 = r73 * r48;
    r73 = fma(r49, r73, r75 * r49);
    r73 = fma(r49, r79, r73);
    r73 = fma(r49, r80, r73);
    r47 = r6 * r73;
    r130 = fma(r71, r47, r130);
    r57 = r21 * r40;
    r57 = r57 * r6;
    r57 = r57 * r69;
    r130 = fma(r53, r57, r130);
    r95 = r4 * r46;
    r95 = r95 * r62;
    r130 = fma(r74, r95, r130);
    r72 = r4 * r45;
    r130 = fma(r118, r72, r130);
    r148 = r56 * r89;
    r148 = r148 * r0;
    r148 = r148 * r68;
    r148 = r148 * r86;
    r130 = fma(r62, r148, r130);
    r26 = r4 * r0;
    r130 = fma(r135, r26, r130);
    r144 = r4 * r0;
    r130 = fma(r131, r144, r130);
    r140 = r56 * r78;
    r140 = r140 * r89;
    r140 = r140 * r0;
    r140 = r140 * r68;
    r140 = r140 * r86;
    r130 = fma(r62, r140, r130);
    r130 = fma(r45, r83, r130);
    r130 = fma(r45, r71, r130);
    r130 = fma(r40, r133, r130);
    r140 = r2 * r130;
    r144 = r40 * r7;
    r144 = r144 * r7;
    r144 = r144 * r56;
    r144 = r144 * r56;
    r144 = r144 * r127;
    r144 = r144 * r43;
    r26 = r60 * r0;
    r26 = r26 * r113;
    r26 = r26 * r109;
    r26 = fma(r106, r26, r88 * r144);
    r144 = r46 * r7;
    r144 = r144 * r122;
    r144 = r144 * r43;
    r26 = fma(r59, r144, r26);
    r148 = r7 * r124;
    r148 = r148 * r0;
    r26 = fma(r117, r148, r26);
    r26 = r26 + r85;
    r49 = fma(r65, r49, r4 * r26);
    r26 = r21 * r40;
    r26 = r26 * r7;
    r26 = r26 * r69;
    r49 = fma(r53, r26, r49);
    r85 = r21 * r40;
    r85 = r85 * r7;
    r85 = r85 * r78;
    r85 = r85 * r69;
    r49 = fma(r53, r85, r49);
    r148 = r84 * r0;
    r148 = r148 * r30;
    r148 = r148 * r70;
    r49 = fma(r113, r148, r49);
    r144 = r5 * r46;
    r144 = r144 * r62;
    r49 = fma(r74, r144, r49);
    r72 = r5 * r0;
    r49 = fma(r135, r72, r49);
    r95 = r5 * r0;
    r49 = fma(r131, r95, r49);
    r57 = r5 * r40;
    r57 = r57 * r7;
    r57 = r57 * r56;
    r57 = r57 * r56;
    r57 = r57 * r102;
    r57 = r57 * r88;
    r49 = fma(r62, r57, r49);
    r47 = r84 * r78;
    r47 = r47 * r0;
    r47 = r47 * r30;
    r47 = r47 * r70;
    r49 = fma(r113, r47, r49);
    r132 = r7 * r73;
    r49 = fma(r71, r132, r49);
    r49 = fma(r46, r83, r49);
    r49 = fma(r0, r128, r49);
    r49 = fma(r45, r121, r49);
    r49 = fma(r46, r71, r49);
    r49 = fma(r0, r138, r49);
    r132 = r3 * r49;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r140,
                                             r132);
    r132 = r50 * r127;
    r132 = r132 * r88;
    r132 = r132 * r63;
    r140 = r6 * r6;
    r47 = r10 * r52;
    r47 = r47 * r6;
    r57 = r50 * r110;
    r57 = fma(r111, r57, r39 * r47);
    r47 = r50 * r6;
    r47 = r47 * r6;
    r57 = fma(r110, r47, r57);
    r111 = r10 * r44;
    r111 = r111 * r7;
    r57 = fma(r39, r111, r57);
    r140 = r140 * r124;
    r140 = r140 * r57;
    r140 = r140 * r86;
    r140 = r140 * r116;
    r140 = fma(r59, r140, r112 * r132);
    r132 = r60 * r57;
    r140 = fma(r115, r132, r140);
    r111 = r52 * r122;
    r111 = r111 * r59;
    r140 = fma(r62, r111, r140);
    r47 = r21 * r7;
    r47 = r47 * r57;
    r47 = fma(r117, r47, r44 * r118);
    r39 = r57 * r113;
    r39 = r39 * r109;
    r47 = fma(r106, r39, r47);
    r47 = fma(r50, r36, r47);
    r140 = r140 + r47;
    r111 = r50 * r63;
    r132 = r21 * r6;
    r132 = r132 * r6;
    r132 = r132 * r57;
    r132 = r132 * r86;
    r132 = r132 * r116;
    r132 = fma(r59, r132, r87 * r111);
    r111 = r52 * r62;
    r132 = fma(r74, r111, r132);
    r132 = fma(r57, r115, r132);
    r47 = r47 + r132;
    r64 = fma(r64, r47, r5 * r140);
    r140 = r21 * r50;
    r140 = r140 * r6;
    r140 = r140 * r69;
    r64 = fma(r53, r140, r64);
    r111 = r4 * r44;
    r111 = r111 * r62;
    r64 = fma(r74, r111, r64);
    r115 = r4 * r57;
    r64 = fma(r131, r115, r64);
    r87 = r56 * r89;
    r87 = r87 * r57;
    r87 = r87 * r68;
    r87 = r87 * r86;
    r64 = fma(r62, r87, r64);
    r116 = r6 * r84;
    r116 = r116 * r57;
    r116 = r116 * r86;
    r116 = r116 * r30;
    r64 = fma(r70, r116, r64);
    r36 = r6 * r84;
    r36 = r36 * r78;
    r36 = r36 * r57;
    r36 = r36 * r86;
    r36 = r36 * r30;
    r64 = fma(r70, r36, r64);
    r39 = r4 * r57;
    r64 = fma(r135, r39, r64);
    r95 = r4 * r52;
    r64 = fma(r118, r95, r64);
    r118 = r56 * r78;
    r118 = r118 * r89;
    r118 = r118 * r57;
    r118 = r118 * r68;
    r118 = r118 * r86;
    r64 = fma(r62, r118, r64);
    r86 = r21 * r50;
    r86 = r86 * r6;
    r86 = r86 * r78;
    r86 = r86 * r69;
    r64 = fma(r53, r86, r64);
    r68 = r76 * r10;
    r68 = r68 * r48;
    r68 = fma(r47, r68, r75 * r47);
    r68 = fma(r47, r80, r68);
    r68 = fma(r47, r79, r68);
    r79 = r6 * r68;
    r64 = fma(r71, r79, r64);
    r64 = fma(r52, r83, r64);
    r64 = fma(r50, r133, r64);
    r64 = fma(r52, r71, r64);
    r79 = r2 * r64;
    r86 = r44 * r7;
    r86 = r86 * r122;
    r86 = r86 * r43;
    r118 = r7 * r124;
    r118 = r118 * r57;
    r118 = fma(r117, r118, r59 * r86);
    r86 = r60 * r57;
    r86 = r86 * r113;
    r86 = r86 * r109;
    r118 = fma(r106, r86, r118);
    r106 = r50 * r7;
    r106 = r106 * r7;
    r106 = r106 * r56;
    r106 = r106 * r56;
    r106 = r106 * r127;
    r106 = r106 * r43;
    r118 = fma(r88, r106, r118);
    r118 = r118 + r132;
    r118 = fma(r4, r118, r65 * r47);
    r47 = r5 * r44;
    r47 = r47 * r62;
    r118 = fma(r74, r47, r118);
    r74 = r5 * r57;
    r118 = fma(r131, r74, r118);
    r131 = r5 * r57;
    r118 = fma(r135, r131, r118);
    r135 = r84 * r78;
    r135 = r135 * r57;
    r135 = r135 * r30;
    r135 = r135 * r70;
    r118 = fma(r113, r135, r118);
    r65 = r21 * r50;
    r65 = r65 * r7;
    r65 = r65 * r78;
    r65 = r65 * r69;
    r118 = fma(r53, r65, r118);
    r132 = r5 * r50;
    r132 = r132 * r7;
    r132 = r132 * r56;
    r132 = r132 * r56;
    r132 = r132 * r102;
    r132 = r132 * r88;
    r118 = fma(r62, r132, r118);
    r88 = r7 * r68;
    r118 = fma(r71, r88, r118);
    r102 = r84 * r57;
    r102 = r102 * r30;
    r102 = r102 * r70;
    r118 = fma(r113, r102, r118);
    r70 = r21 * r50;
    r70 = r70 * r7;
    r70 = r70 * r69;
    r118 = fma(r53, r70, r118);
    r118 = fma(r57, r128, r118);
    r118 = fma(r57, r138, r118);
    r118 = fma(r44, r83, r118);
    r118 = fma(r52, r121, r118);
    r118 = fma(r44, r71, r118);
    r70 = r3 * r118;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r79,
                                             r70);
    r70 = r3 * r21;
    r70 = r70 * r1;
    r70 = fma(r25, r126, r31 * r70);
    r79 = r3 * r21;
    r79 = r79 * r1;
    r79 = fma(r130, r126, r49 * r79);
    WriteSum2<double, double>((double*)inout_shared, r70, r79);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = r3 * r21;
    r79 = r79 * r1;
    r126 = fma(r64, r126, r118 * r79);
    WriteSum1<double, double>((double*)inout_shared, r126);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r31 * r31;
    r79 = r25 * r25;
    r79 = fma(r147, r79, r142 * r126);
    r126 = r49 * r49;
    r1 = r130 * r130;
    r1 = fma(r147, r1, r142 * r126);
    WriteSum2<double, double>((double*)inout_shared, r79, r1);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r118 * r118;
    r79 = r64 * r64;
    r79 = fma(r147, r79, r142 * r1);
    WriteSum1<double, double>((double*)inout_shared, r79);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = r31 * r49;
    r1 = r25 * r130;
    r1 = fma(r147, r1, r142 * r79);
    r79 = r31 * r118;
    r126 = r25 * r64;
    r126 = fma(r147, r126, r142 * r79);
    WriteSum2<double, double>((double*)inout_shared, r1, r126);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r49 * r118;
    r1 = r130 * r64;
    r1 = fma(r147, r1, r142 * r126);
    WriteSum1<double, double>((double*)inout_shared, r1);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void ThinPrismFisheyeSplitFixedPrincipalPointResJacFirst(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    double* out_focal_and_extra_jac,
    unsigned int out_focal_and_extra_jac_num_alloc,
    double* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    double* const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    double* const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
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
  ThinPrismFisheyeSplitFixedPrincipalPointResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
      focal_and_extra_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      principal_point,
      principal_point_num_alloc,
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
      out_focal_and_extra_jac,
      out_focal_and_extra_jac_num_alloc,
      out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc,
      out_focal_and_extra_precond_diag,
      out_focal_and_extra_precond_diag_num_alloc,
      out_focal_and_extra_precond_tril,
      out_focal_and_extra_precond_tril_num_alloc,
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