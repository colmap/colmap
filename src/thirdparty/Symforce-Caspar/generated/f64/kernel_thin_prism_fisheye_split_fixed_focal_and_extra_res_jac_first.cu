#include "kernel_thin_prism_fisheye_split_fixed_focal_and_extra_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedFocalAndExtraResJacFirstKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
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
        double* out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        double* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        double* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        double* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
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
      r129, r130, r131, r132, r133, r134, r135, r136, r137, r138, r139, r140;
  LoadShared<2, double, double>(principal_point,
                                0 * principal_point_num_alloc,
                                principal_point_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        principal_point_indices_loc[threadIdx.x].target,
                        r0,
                        r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            0 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r2,
                                            r3);
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            4 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r4,
                                            r5);
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
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            8 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r64,
                                            r65);
    r66 = r59 * r63;
    r48 = r48 + r66;
    r61 = fma(r64, r48, r5 * r61);
    r67 = r4 * r7;
    r68 = r10 * r59;
    r67 = r67 * r62;
    r61 = fma(r68, r67, r61);
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            2 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r69,
                                            r70);
    r71 = r48 * r48;
    r72 = fma(r70, r71, r69 * r48);
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            6 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r73,
                                            r74);
    r75 = r48 * r71;
    r74 = r74 * r75;
    r72 = fma(r48, r74, r72);
    r72 = fma(r73, r75, r72);
    r75 = 1.0 / r31;
    r76 = 1.0 / r47;
    r77 = r75 * r76;
    r78 = r56 * r77;
    r79 = r72 * r78;
    r61 = fma(r6, r79, r61);
    r61 = fma(r6, r78, r61);
    r61 = fma(r2, r61, r0);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r0, r67);
    r61 = fma(r0, r21, r61);
    r0 = r7 * r7;
    r0 = r0 * r60;
    r0 = r0 * r43;
    r0 = fma(r59, r0, r66);
    r0 = fma(r65, r48, r4 * r0);
    r66 = r5 * r7;
    r66 = r66 * r62;
    r0 = fma(r68, r66, r0);
    r0 = fma(r7, r79, r0);
    r0 = fma(r7, r78, r0);
    r0 = fma(r3, r0, r1);
    r0 = fma(r67, r21, r0);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r61, r0);
    r67 = fma(r0, r0, r61 * r61);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r67);
  if (global_thread_idx < problem_size) {
    r67 = r6 * r43;
    r1 = -5.00000000000000000e-01;
    r66 = rsqrt(r30);
    r80 = r10 * r6;
    r81 = r11 * r14;
    r82 = r16 * r17;
    r82 = fma(r1, r82, r1 * r81);
    r81 = r15 * r18;
    r82 = fma(r1, r81, r82);
    r83 = r12 * r13;
    r84 = 5.00000000000000000e-01;
    r82 = fma(r84, r83, r82);
    r83 = r24 * r82;
    r81 = r15 * r14;
    r85 = r12 * r17;
    r85 = fma(r84, r85, r84 * r81);
    r81 = r11 * r18;
    r85 = fma(r1, r81, r85);
    r86 = r16 * r13;
    r85 = fma(r84, r86, r85);
    r86 = r29 * r85;
    r81 = fma(r10, r86, r10 * r83);
    r87 = r10 * r19;
    r88 = fma(r84, r26, r1 * r22);
    r88 = fma(r1, r23, r88);
    r88 = fma(r1, r25, r88);
    r89 = r10 * r27;
    r90 = r16 * r14;
    r91 = r11 * r17;
    r91 = fma(r1, r91, r84 * r90);
    r90 = r12 * r18;
    r91 = fma(r1, r90, r91);
    r92 = r15 * r13;
    r91 = fma(r1, r92, r91);
    r89 = r89 * r91;
    r87 = fma(r88, r87, r89);
    r81 = r81 + r87;
    r92 = r10 * r24;
    r92 = r92 * r91;
    r90 = r10 * r19;
    r90 = r90 * r85;
    r93 = r92 + r90;
    r94 = r27 * r35;
    r93 = fma(r82, r94, r93);
    r95 = r35 * r29;
    r93 = fma(r88, r95, r93);
    r93 = fma(r9, r93, r42 * r81);
    r81 = r24 * r85;
    r95 = -4.00000000000000000e+00;
    r81 = r81 * r95;
    r94 = r27 * r88;
    r96 = r95 * r94;
    r97 = r81 + r96;
    r93 = fma(r8, r97, r93);
    r80 = r80 * r93;
    r97 = r6 * r6;
    r98 = r10 * r24;
    r98 = r98 * r88;
    r99 = r10 * r27;
    r99 = fma(r85, r99, r98);
    r85 = r10 * r19;
    r85 = r85 * r82;
    r100 = r10 * r29;
    r100 = r100 * r91;
    r101 = r85 + r100;
    r102 = r99 + r101;
    r86 = fma(r35, r86, r35 * r83);
    r86 = r86 + r87;
    r86 = fma(r8, r86, r9 * r102);
    r102 = r19 * r91;
    r102 = r102 * r95;
    r81 = r81 + r102;
    r86 = fma(r42, r81, r86);
    r57 = r31 * r57;
    r57 = 1.0 / r57;
    r31 = r35 * r57;
    r97 = r97 * r86;
    r97 = fma(r31, r97, r39 * r80);
    r80 = r7 * r7;
    r80 = r80 * r31;
    r81 = r10 * r7;
    r103 = r19 * r35;
    r104 = r35 * r29;
    r104 = r104 * r91;
    r103 = fma(r82, r103, r104);
    r103 = r103 + r99;
    r96 = r102 + r96;
    r96 = fma(r9, r96, r42 * r103);
    r103 = r10 * r29;
    r103 = fma(r88, r103, r90);
    r90 = r10 * r27;
    r90 = fma(r82, r90, r92);
    r103 = r103 + r90;
    r96 = fma(r8, r103, r96);
    r81 = r81 * r96;
    r97 = fma(r39, r81, r97);
    r97 = fma(r86, r80, r97);
    r67 = r67 * r56;
    r67 = r67 * r75;
    r67 = r67 * r1;
    r67 = r67 * r66;
    r67 = r67 * r97;
    r81 = r56 * r56;
    r103 = r43 * r81;
    r103 = r103 * r80;
    r92 = r7 * r66;
    r102 = r97 * r92;
    r30 = r36 + r30;
    r30 = 1.0 / r30;
    r99 = r30 * r53;
    r105 = r7 * r43;
    r102 = r102 * r99;
    r102 = fma(r105, r102, r86 * r103);
    r106 = r21 * r7;
    r49 = r47 * r49;
    r49 = 1.0 / r49;
    r47 = r49 * r59;
    r47 = r47 * r92;
    r106 = r106 * r97;
    r102 = fma(r47, r106, r102);
    r107 = r68 * r105;
    r102 = fma(r96, r107, r102);
    r106 = r97 * r66;
    r106 = r106 * r99;
    r108 = r21 * r6;
    r108 = r108 * r6;
    r108 = r108 * r97;
    r108 = r108 * r66;
    r108 = r108 * r49;
    r108 = fma(r59, r108, r63 * r106);
    r106 = r93 * r62;
    r108 = fma(r68, r106, r108);
    r109 = r86 * r31;
    r81 = r63 * r81;
    r108 = fma(r81, r109, r108);
    r109 = r102 + r108;
    r106 = fma(r64, r109, r67);
    r110 = r60 * r97;
    r110 = r110 * r66;
    r110 = r110 * r99;
    r111 = r6 * r6;
    r112 = -3.00000000000000000e+00;
    r111 = r111 * r112;
    r111 = r111 * r97;
    r111 = r111 * r66;
    r111 = r111 * r49;
    r111 = fma(r59, r111, r63 * r110);
    r110 = 6.00000000000000000e+00;
    r113 = r93 * r110;
    r113 = r113 * r59;
    r111 = fma(r62, r113, r111);
    r114 = -6.00000000000000000e+00;
    r115 = r114 * r57;
    r115 = r115 * r81;
    r111 = fma(r86, r115, r111);
    r111 = r111 + r102;
    r102 = r93 * r107;
    r113 = r21 * r6;
    r113 = r113 * r72;
    r113 = r113 * r86;
    r113 = r113 * r76;
    r106 = fma(r53, r113, r106);
    r116 = r6 * r84;
    r116 = r116 * r66;
    r116 = r116 * r30;
    r116 = r116 * r77;
    r117 = r72 * r116;
    r118 = r4 * r35;
    r118 = r118 * r6;
    r118 = r118 * r97;
    r106 = fma(r47, r118, r106);
    r119 = r4 * r96;
    r119 = r119 * r62;
    r106 = fma(r68, r119, r106);
    r120 = r4 * r7;
    r120 = r120 * r56;
    r120 = r120 * r56;
    r120 = r120 * r95;
    r120 = r120 * r57;
    r120 = r120 * r62;
    r121 = r4 * r97;
    r122 = r10 * r62;
    r122 = r122 * r92;
    r122 = r122 * r99;
    r106 = fma(r122, r121, r106);
    r123 = r70 * r10;
    r123 = r123 * r48;
    r123 = fma(r69, r109, r109 * r123);
    r73 = r73 * r60;
    r73 = r73 * r71;
    r71 = 4.00000000000000000e+00;
    r74 = r71 * r74;
    r123 = fma(r109, r73, r123);
    r123 = fma(r109, r74, r123);
    r71 = r6 * r123;
    r106 = fma(r78, r71, r106);
    r124 = r21 * r6;
    r124 = r124 * r86;
    r124 = r124 * r76;
    r106 = fma(r53, r124, r106);
    r106 = fma(r5, r111, r106);
    r106 = fma(r4, r102, r106);
    r106 = fma(r72, r67, r106);
    r106 = fma(r97, r117, r106);
    r106 = fma(r93, r79, r106);
    r106 = fma(r86, r120, r106);
    r106 = fma(r97, r116, r106);
    r106 = fma(r93, r78, r106);
    r124 = r2 * r106;
    r71 = r7 * r43;
    r71 = r71 * r56;
    r71 = r71 * r75;
    r71 = r71 * r1;
    r71 = r71 * r66;
    r71 = r71 * r97;
    r109 = fma(r65, r109, r71);
    r121 = r7 * r7;
    r121 = r121 * r56;
    r121 = r121 * r56;
    r121 = r121 * r86;
    r121 = r121 * r114;
    r121 = r121 * r43;
    r119 = r60 * r97;
    r119 = r119 * r92;
    r119 = r119 * r99;
    r119 = fma(r105, r119, r57 * r121);
    r121 = r7 * r112;
    r121 = r121 * r97;
    r119 = fma(r47, r121, r119);
    r118 = r7 * r96;
    r118 = r118 * r110;
    r118 = r118 * r43;
    r119 = fma(r59, r118, r119);
    r119 = r119 + r108;
    r108 = r21 * r7;
    r108 = r108 * r86;
    r108 = r108 * r76;
    r109 = fma(r53, r108, r109);
    r118 = r5 * r35;
    r118 = r118 * r6;
    r118 = r118 * r47;
    r121 = r84 * r97;
    r121 = r121 * r30;
    r121 = r121 * r92;
    r109 = fma(r77, r121, r109);
    r113 = r5 * r96;
    r113 = r113 * r62;
    r109 = fma(r68, r113, r109);
    r67 = r5 * r7;
    r67 = r67 * r56;
    r67 = r67 * r56;
    r67 = r67 * r95;
    r67 = r67 * r86;
    r67 = r67 * r57;
    r109 = fma(r62, r67, r109);
    r111 = r7 * r123;
    r109 = fma(r78, r111, r109);
    r125 = r5 * r97;
    r109 = fma(r122, r125, r109);
    r126 = r84 * r72;
    r126 = r126 * r97;
    r126 = r126 * r30;
    r126 = r126 * r92;
    r109 = fma(r77, r126, r109);
    r127 = r21 * r7;
    r127 = r127 * r72;
    r127 = r127 * r86;
    r127 = r127 * r76;
    r109 = fma(r53, r127, r109);
    r109 = fma(r4, r119, r109);
    r109 = fma(r5, r102, r109);
    r109 = fma(r72, r71, r109);
    r109 = fma(r97, r118, r109);
    r109 = fma(r96, r79, r109);
    r109 = fma(r96, r78, r109);
    r127 = r3 * r109;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             0 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r124,
                                             r127);
    r127 = r35 * r24;
    r127 = fma(r88, r127, r104);
    r124 = r10 * r27;
    r126 = r15 * r14;
    r125 = r12 * r17;
    r125 = fma(r1, r125, r1 * r126);
    r126 = r11 * r18;
    r125 = fma(r84, r126, r125);
    r111 = r16 * r13;
    r125 = fma(r1, r111, r125);
    r124 = r124 * r125;
    r111 = r10 * r19;
    r126 = r11 * r14;
    r67 = r16 * r17;
    r67 = fma(r84, r67, r84 * r126);
    r126 = r15 * r18;
    r67 = fma(r84, r126, r67);
    r113 = r12 * r13;
    r67 = fma(r1, r113, r67);
    r111 = fma(r67, r111, r124);
    r127 = r127 + r111;
    r113 = r10 * r24;
    r113 = r113 * r67;
    r126 = r10 * r29;
    r126 = fma(r125, r126, r113);
    r126 = r126 + r87;
    r126 = fma(r9, r126, r8 * r127);
    r127 = r24 * r91;
    r127 = r127 * r95;
    r87 = r19 * r125;
    r121 = r95 * r87;
    r71 = r127 + r121;
    r126 = fma(r42, r71, r126);
    r100 = r98 + r100;
    r100 = r100 + r111;
    r111 = r27 * r95;
    r111 = r111 * r67;
    r127 = r127 + r111;
    r127 = fma(r8, r127, r42 * r100);
    r100 = r35 * r29;
    r100 = fma(r35, r94, r67 * r100);
    r98 = r10 * r19;
    r98 = r98 * r91;
    r71 = r10 * r24;
    r71 = fma(r125, r71, r98);
    r100 = r100 + r71;
    r127 = fma(r9, r100, r127);
    r100 = r110 * r127;
    r100 = r100 * r59;
    r100 = fma(r62, r100, r126 * r115);
    r108 = r6 * r6;
    r102 = r10 * r6;
    r102 = r102 * r127;
    r119 = r10 * r7;
    r113 = r89 + r113;
    r89 = r19 * r35;
    r113 = fma(r88, r89, r113);
    r88 = r35 * r29;
    r113 = fma(r125, r88, r113);
    r88 = r10 * r29;
    r94 = fma(r10, r94, r67 * r88);
    r94 = r94 + r71;
    r94 = fma(r8, r94, r42 * r113);
    r121 = r111 + r121;
    r94 = fma(r9, r121, r94);
    r119 = r119 * r94;
    r119 = fma(r39, r119, r39 * r102);
    r102 = r6 * r6;
    r102 = r102 * r126;
    r119 = fma(r31, r102, r119);
    r119 = fma(r126, r80, r119);
    r102 = r112 * r119;
    r108 = r108 * r66;
    r108 = r108 * r49;
    r108 = r108 * r59;
    r100 = fma(r102, r108, r100);
    r121 = r60 * r119;
    r121 = r121 * r66;
    r121 = r121 * r99;
    r100 = fma(r63, r121, r100);
    r111 = r21 * r7;
    r111 = r111 * r119;
    r111 = fma(r47, r111, r94 * r107);
    r113 = r119 * r92;
    r113 = r113 * r99;
    r111 = fma(r105, r113, r111);
    r111 = fma(r126, r103, r111);
    r100 = r100 + r111;
    r121 = r126 * r31;
    r108 = r127 * r62;
    r108 = fma(r68, r108, r81 * r121);
    r121 = r21 * r6;
    r121 = r121 * r6;
    r121 = r121 * r119;
    r121 = r121 * r66;
    r121 = r121 * r49;
    r108 = fma(r59, r121, r108);
    r113 = r119 * r66;
    r113 = r113 * r99;
    r108 = fma(r63, r113, r108);
    r111 = r111 + r108;
    r100 = fma(r64, r111, r5 * r100);
    r113 = r21 * r6;
    r113 = r113 * r126;
    r113 = r113 * r76;
    r100 = fma(r53, r113, r100);
    r121 = r56 * r72;
    r121 = r121 * r1;
    r121 = r121 * r119;
    r121 = r121 * r75;
    r121 = r121 * r66;
    r100 = fma(r62, r121, r100);
    r88 = r70 * r10;
    r88 = r88 * r48;
    r88 = fma(r111, r88, r69 * r111);
    r88 = fma(r111, r74, r88);
    r88 = fma(r111, r73, r88);
    r67 = r6 * r88;
    r100 = fma(r78, r67, r100);
    r89 = r21 * r6;
    r89 = r89 * r72;
    r89 = r89 * r126;
    r89 = r89 * r76;
    r100 = fma(r53, r89, r100);
    r128 = r4 * r119;
    r100 = fma(r122, r128, r100);
    r129 = r4 * r35;
    r129 = r129 * r6;
    r129 = r129 * r119;
    r100 = fma(r47, r129, r100);
    r130 = r4 * r94;
    r130 = r130 * r62;
    r100 = fma(r68, r130, r100);
    r131 = r4 * r127;
    r100 = fma(r107, r131, r100);
    r132 = r56 * r1;
    r132 = r132 * r119;
    r132 = r132 * r75;
    r132 = r132 * r66;
    r100 = fma(r62, r132, r100);
    r100 = fma(r119, r117, r100);
    r100 = fma(r119, r116, r100);
    r100 = fma(r126, r120, r100);
    r100 = fma(r127, r78, r100);
    r100 = fma(r127, r79, r100);
    r132 = r2 * r100;
    r131 = r7 * r110;
    r131 = r131 * r94;
    r131 = r131 * r43;
    r130 = r7 * r47;
    r130 = fma(r102, r130, r59 * r131);
    r131 = r7 * r7;
    r131 = r131 * r56;
    r131 = r131 * r56;
    r131 = r131 * r114;
    r131 = r131 * r126;
    r131 = r131 * r43;
    r130 = fma(r57, r131, r130);
    r102 = r60 * r119;
    r102 = r102 * r92;
    r102 = r102 * r99;
    r130 = fma(r105, r102, r130);
    r130 = r130 + r108;
    r111 = fma(r65, r111, r4 * r130);
    r130 = r56 * r1;
    r130 = r130 * r119;
    r130 = r130 * r43;
    r130 = r130 * r75;
    r111 = fma(r92, r130, r111);
    r108 = r7 * r88;
    r111 = fma(r78, r108, r111);
    r102 = r84 * r119;
    r102 = r102 * r30;
    r102 = r102 * r92;
    r111 = fma(r77, r102, r111);
    r131 = r5 * r7;
    r131 = r131 * r56;
    r131 = r131 * r56;
    r131 = r131 * r95;
    r131 = r131 * r126;
    r131 = r131 * r57;
    r111 = fma(r62, r131, r111);
    r129 = r5 * r119;
    r111 = fma(r122, r129, r111);
    r128 = r21 * r7;
    r128 = r128 * r72;
    r128 = r128 * r126;
    r128 = r128 * r76;
    r111 = fma(r53, r128, r111);
    r89 = r56 * r72;
    r89 = r89 * r1;
    r89 = r89 * r119;
    r89 = r89 * r43;
    r89 = r89 * r75;
    r111 = fma(r92, r89, r111);
    r67 = r21 * r7;
    r67 = r67 * r126;
    r67 = r67 * r76;
    r111 = fma(r53, r67, r111);
    r121 = r5 * r94;
    r121 = r121 * r62;
    r111 = fma(r68, r121, r111);
    r113 = r84 * r72;
    r113 = r113 * r119;
    r113 = r113 * r30;
    r113 = r113 * r92;
    r111 = fma(r77, r113, r111);
    r133 = r5 * r127;
    r111 = fma(r107, r133, r111);
    r111 = fma(r94, r79, r111);
    r111 = fma(r119, r118, r111);
    r111 = fma(r94, r78, r111);
    r133 = r3 * r111;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             2 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r132,
                                             r133);
    r133 = r19 * r95;
    r26 = fma(r1, r26, r84 * r22);
    r26 = fma(r84, r23, r26);
    r26 = fma(r84, r25, r26);
    r133 = r133 * r26;
    r83 = r95 * r83;
    r25 = r133 + r83;
    r23 = r10 * r27;
    r23 = r23 * r26;
    r98 = r98 + r23;
    r22 = r35 * r24;
    r98 = fma(r125, r22, r98);
    r132 = r35 * r29;
    r98 = fma(r82, r132, r98);
    r98 = fma(r8, r98, r42 * r25);
    r25 = r10 * r29;
    r25 = fma(r10, r87, r26 * r25);
    r25 = r25 + r90;
    r98 = fma(r9, r25, r98);
    r25 = r10 * r7;
    r132 = r10 * r24;
    r132 = r132 * r26;
    r124 = r124 + r132;
    r124 = r124 + r101;
    r101 = r35 * r29;
    r87 = fma(r35, r87, r26 * r101);
    r87 = r87 + r90;
    r87 = fma(r42, r87, r8 * r124);
    r91 = r27 * r91;
    r91 = r91 * r95;
    r133 = r133 + r91;
    r87 = fma(r9, r133, r87);
    r25 = r25 * r87;
    r133 = r6 * r6;
    r133 = r133 * r98;
    r133 = fma(r31, r133, r39 * r25);
    r25 = r10 * r6;
    r104 = r85 + r104;
    r85 = r27 * r35;
    r104 = fma(r125, r85, r104);
    r104 = r104 + r132;
    r83 = r91 + r83;
    r83 = fma(r8, r83, r9 * r104);
    r8 = r10 * r29;
    r8 = fma(r82, r8, r23);
    r8 = r8 + r71;
    r83 = fma(r42, r8, r83);
    r25 = r25 * r83;
    r133 = fma(r39, r25, r133);
    r133 = fma(r98, r80, r133);
    r25 = r133 * r92;
    r25 = r25 * r99;
    r25 = fma(r105, r25, r98 * r103);
    r8 = r21 * r7;
    r8 = r8 * r133;
    r25 = fma(r47, r8, r25);
    r25 = fma(r87, r107, r25);
    r8 = r21 * r6;
    r8 = r8 * r6;
    r8 = r8 * r133;
    r8 = r8 * r66;
    r8 = r8 * r49;
    r42 = r83 * r62;
    r42 = fma(r68, r42, r59 * r8);
    r8 = r98 * r31;
    r42 = fma(r81, r8, r42);
    r71 = r133 * r66;
    r71 = r71 * r99;
    r42 = fma(r63, r71, r42);
    r71 = r25 + r42;
    r8 = r6 * r6;
    r8 = r8 * r112;
    r8 = r8 * r133;
    r8 = r8 * r66;
    r8 = r8 * r49;
    r23 = r110 * r83;
    r23 = r23 * r59;
    r23 = fma(r62, r23, r59 * r8);
    r8 = r60 * r133;
    r8 = r8 * r66;
    r8 = r8 * r99;
    r23 = fma(r63, r8, r23);
    r23 = fma(r98, r115, r23);
    r23 = r23 + r25;
    r23 = fma(r5, r23, r64 * r71);
    r25 = r70 * r10;
    r25 = r25 * r48;
    r25 = fma(r71, r25, r69 * r71);
    r25 = fma(r71, r73, r25);
    r25 = fma(r71, r74, r25);
    r8 = r6 * r25;
    r23 = fma(r78, r8, r23);
    r82 = r4 * r83;
    r23 = fma(r107, r82, r23);
    r104 = r4 * r133;
    r23 = fma(r122, r104, r23);
    r9 = r21 * r6;
    r9 = r9 * r72;
    r9 = r9 * r98;
    r9 = r9 * r76;
    r23 = fma(r53, r9, r23);
    r91 = r4 * r35;
    r91 = r91 * r6;
    r91 = r91 * r133;
    r23 = fma(r47, r91, r23);
    r85 = r21 * r6;
    r85 = r85 * r98;
    r85 = r85 * r76;
    r23 = fma(r53, r85, r23);
    r132 = r56 * r1;
    r132 = r132 * r133;
    r132 = r132 * r75;
    r132 = r132 * r66;
    r23 = fma(r62, r132, r23);
    r125 = r4 * r87;
    r125 = r125 * r62;
    r23 = fma(r68, r125, r23);
    r124 = r56 * r72;
    r124 = r124 * r1;
    r124 = r124 * r133;
    r124 = r124 * r75;
    r124 = r124 * r66;
    r23 = fma(r62, r124, r23);
    r23 = fma(r133, r116, r23);
    r23 = fma(r133, r117, r23);
    r23 = fma(r83, r79, r23);
    r23 = fma(r83, r78, r23);
    r23 = fma(r98, r120, r23);
    r124 = r2 * r23;
    r125 = r7 * r7;
    r125 = r125 * r56;
    r125 = r125 * r56;
    r125 = r125 * r114;
    r125 = r125 * r98;
    r125 = r125 * r43;
    r132 = r60 * r133;
    r132 = r132 * r92;
    r132 = r132 * r99;
    r132 = fma(r105, r132, r57 * r125);
    r125 = r7 * r110;
    r125 = r125 * r87;
    r125 = r125 * r43;
    r132 = fma(r59, r125, r132);
    r85 = r7 * r112;
    r85 = r85 * r133;
    r132 = fma(r47, r85, r132);
    r132 = r132 + r42;
    r132 = fma(r4, r132, r65 * r71);
    r71 = r21 * r7;
    r71 = r71 * r72;
    r71 = r71 * r98;
    r71 = r71 * r76;
    r132 = fma(r53, r71, r132);
    r42 = r21 * r7;
    r42 = r42 * r98;
    r42 = r42 * r76;
    r132 = fma(r53, r42, r132);
    r85 = r5 * r83;
    r132 = fma(r107, r85, r132);
    r125 = r56 * r72;
    r125 = r125 * r1;
    r125 = r125 * r133;
    r125 = r125 * r43;
    r125 = r125 * r75;
    r132 = fma(r92, r125, r132);
    r91 = r5 * r133;
    r132 = fma(r122, r91, r132);
    r9 = r84 * r133;
    r9 = r9 * r30;
    r9 = r9 * r92;
    r132 = fma(r77, r9, r132);
    r104 = r7 * r25;
    r132 = fma(r78, r104, r132);
    r82 = r56 * r1;
    r82 = r82 * r133;
    r82 = r82 * r43;
    r82 = r82 * r75;
    r132 = fma(r92, r82, r132);
    r8 = r84 * r72;
    r8 = r8 * r133;
    r8 = r8 * r30;
    r8 = r8 * r92;
    r132 = fma(r77, r8, r132);
    r90 = r5 * r7;
    r90 = r90 * r56;
    r90 = r90 * r56;
    r90 = r90 * r95;
    r90 = r90 * r98;
    r90 = r90 * r57;
    r132 = fma(r62, r90, r132);
    r101 = r5 * r87;
    r101 = r101 * r62;
    r132 = fma(r68, r101, r132);
    r132 = fma(r87, r79, r132);
    r132 = fma(r87, r78, r132);
    r132 = fma(r133, r118, r132);
    r101 = r3 * r132;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             4 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r124,
                                             r101);
    r101 = r37 * r110;
    r101 = r101 * r59;
    r124 = r6 * r6;
    r90 = r10 * r33;
    r90 = r90 * r7;
    r90 = fma(r20, r80, r39 * r90);
    r8 = r20 * r6;
    r8 = r8 * r6;
    r90 = fma(r31, r8, r90);
    r82 = r10 * r37;
    r82 = r82 * r6;
    r90 = fma(r39, r82, r90);
    r124 = r124 * r112;
    r124 = r124 * r90;
    r124 = r124 * r66;
    r124 = r124 * r49;
    r124 = fma(r59, r124, r62 * r101);
    r101 = r60 * r90;
    r101 = r101 * r66;
    r101 = r101 * r99;
    r124 = fma(r63, r101, r124);
    r82 = r21 * r7;
    r82 = r82 * r90;
    r82 = fma(r47, r82, r33 * r107);
    r8 = r90 * r92;
    r8 = r8 * r99;
    r82 = fma(r105, r8, r82);
    r82 = fma(r20, r103, r82);
    r124 = fma(r20, r115, r124);
    r124 = r124 + r82;
    r101 = r37 * r62;
    r8 = r21 * r6;
    r8 = r8 * r6;
    r8 = r8 * r90;
    r8 = r8 * r66;
    r8 = r8 * r49;
    r8 = fma(r59, r8, r68 * r101);
    r101 = r90 * r66;
    r101 = r101 * r99;
    r8 = fma(r63, r101, r8);
    r104 = r20 * r31;
    r8 = fma(r81, r104, r8);
    r82 = r82 + r8;
    r124 = fma(r64, r82, r5 * r124);
    r104 = r4 * r90;
    r124 = fma(r122, r104, r124);
    r101 = r56 * r72;
    r101 = r101 * r1;
    r101 = r101 * r90;
    r101 = r101 * r75;
    r101 = r101 * r66;
    r124 = fma(r62, r101, r124);
    r9 = r56 * r1;
    r9 = r9 * r90;
    r9 = r9 * r75;
    r9 = r9 * r66;
    r124 = fma(r62, r9, r124);
    r91 = r4 * r35;
    r91 = r91 * r6;
    r91 = r91 * r90;
    r124 = fma(r47, r91, r124);
    r125 = r70 * r10;
    r125 = r125 * r48;
    r125 = fma(r69, r82, r82 * r125);
    r125 = fma(r82, r74, r125);
    r125 = fma(r82, r73, r125);
    r85 = r6 * r125;
    r124 = fma(r78, r85, r124);
    r42 = r21 * r20;
    r42 = r42 * r6;
    r42 = r42 * r72;
    r42 = r42 * r76;
    r124 = fma(r53, r42, r124);
    r71 = r21 * r20;
    r71 = r71 * r6;
    r71 = r71 * r76;
    r124 = fma(r53, r71, r124);
    r26 = r4 * r33;
    r26 = r26 * r62;
    r124 = fma(r68, r26, r124);
    r22 = r4 * r37;
    r124 = fma(r107, r22, r124);
    r124 = fma(r37, r79, r124);
    r124 = fma(r90, r117, r124);
    r124 = fma(r90, r116, r124);
    r124 = fma(r20, r120, r124);
    r124 = fma(r37, r78, r124);
    r22 = r2 * r124;
    r26 = r33 * r7;
    r26 = r26 * r110;
    r26 = r26 * r43;
    r71 = r7 * r112;
    r71 = r71 * r90;
    r71 = fma(r47, r71, r59 * r26);
    r26 = r20 * r7;
    r26 = r26 * r7;
    r26 = r26 * r56;
    r26 = r26 * r56;
    r26 = r26 * r114;
    r26 = r26 * r43;
    r71 = fma(r57, r26, r71);
    r42 = r60 * r90;
    r42 = r42 * r92;
    r42 = r42 * r99;
    r71 = fma(r105, r42, r71);
    r71 = r71 + r8;
    r82 = fma(r65, r82, r4 * r71);
    r71 = r84 * r90;
    r71 = r71 * r30;
    r71 = r71 * r92;
    r82 = fma(r77, r71, r82);
    r8 = r56 * r72;
    r8 = r8 * r1;
    r8 = r8 * r90;
    r8 = r8 * r43;
    r8 = r8 * r75;
    r82 = fma(r92, r8, r82);
    r42 = r21 * r20;
    r42 = r42 * r7;
    r42 = r42 * r72;
    r42 = r42 * r76;
    r82 = fma(r53, r42, r82);
    r26 = r5 * r90;
    r82 = fma(r122, r26, r82);
    r85 = r21 * r20;
    r85 = r85 * r7;
    r85 = r85 * r76;
    r82 = fma(r53, r85, r82);
    r91 = r5 * r20;
    r91 = r91 * r7;
    r91 = r91 * r56;
    r91 = r91 * r56;
    r91 = r91 * r95;
    r91 = r91 * r57;
    r82 = fma(r62, r91, r82);
    r9 = r7 * r125;
    r82 = fma(r78, r9, r82);
    r101 = r84 * r72;
    r101 = r101 * r90;
    r101 = r101 * r30;
    r101 = r101 * r92;
    r82 = fma(r77, r101, r82);
    r104 = r5 * r33;
    r104 = r104 * r62;
    r82 = fma(r68, r104, r82);
    r113 = r56 * r1;
    r113 = r113 * r90;
    r113 = r113 * r43;
    r113 = r113 * r75;
    r82 = fma(r92, r113, r82);
    r121 = r5 * r37;
    r82 = fma(r107, r121, r82);
    r82 = fma(r33, r78, r82);
    r82 = fma(r90, r118, r82);
    r82 = fma(r33, r79, r82);
    r121 = r3 * r82;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r22, r121);
    r121 = r10 * r38;
    r121 = r121 * r7;
    r121 = fma(r39, r121, r34 * r80);
    r22 = r34 * r6;
    r22 = r22 * r6;
    r121 = fma(r31, r22, r121);
    r113 = r10 * r55;
    r113 = r113 * r6;
    r121 = fma(r39, r113, r121);
    r113 = r60 * r121;
    r113 = r113 * r66;
    r113 = r113 * r99;
    r113 = fma(r34, r115, r63 * r113);
    r22 = r6 * r6;
    r22 = r22 * r112;
    r22 = r22 * r121;
    r22 = r22 * r66;
    r22 = r22 * r49;
    r113 = fma(r59, r22, r113);
    r104 = r55 * r110;
    r104 = r104 * r59;
    r113 = fma(r62, r104, r113);
    r101 = fma(r38, r107, r34 * r103);
    r9 = r121 * r92;
    r9 = r9 * r99;
    r101 = fma(r105, r9, r101);
    r91 = r21 * r7;
    r91 = r91 * r121;
    r101 = fma(r47, r91, r101);
    r113 = r113 + r101;
    r104 = r121 * r66;
    r104 = r104 * r99;
    r22 = r34 * r31;
    r22 = fma(r81, r22, r63 * r104);
    r104 = r21 * r6;
    r104 = r104 * r6;
    r104 = r104 * r121;
    r104 = r104 * r66;
    r104 = r104 * r49;
    r22 = fma(r59, r104, r22);
    r91 = r55 * r62;
    r22 = fma(r68, r91, r22);
    r101 = r101 + r22;
    r113 = fma(r64, r101, r5 * r113);
    r91 = r4 * r38;
    r91 = r91 * r62;
    r113 = fma(r68, r91, r113);
    r104 = r4 * r121;
    r113 = fma(r122, r104, r113);
    r9 = r21 * r34;
    r9 = r9 * r6;
    r9 = r9 * r76;
    r113 = fma(r53, r9, r113);
    r85 = r4 * r35;
    r85 = r85 * r6;
    r85 = r85 * r121;
    r113 = fma(r47, r85, r113);
    r26 = r56 * r1;
    r26 = r26 * r121;
    r26 = r26 * r75;
    r26 = r26 * r66;
    r113 = fma(r62, r26, r113);
    r42 = r70 * r10;
    r42 = r42 * r48;
    r42 = fma(r69, r101, r101 * r42);
    r42 = fma(r101, r73, r42);
    r42 = fma(r101, r74, r42);
    r8 = r6 * r42;
    r113 = fma(r78, r8, r113);
    r71 = r56 * r72;
    r71 = r71 * r1;
    r71 = r71 * r121;
    r71 = r71 * r75;
    r71 = r71 * r66;
    r113 = fma(r62, r71, r113);
    r67 = r21 * r34;
    r67 = r67 * r6;
    r67 = r67 * r72;
    r67 = r67 * r76;
    r113 = fma(r53, r67, r113);
    r89 = r4 * r55;
    r113 = fma(r107, r89, r113);
    r113 = fma(r121, r117, r113);
    r113 = fma(r55, r78, r113);
    r113 = fma(r34, r120, r113);
    r113 = fma(r121, r116, r113);
    r113 = fma(r55, r79, r113);
    r89 = r2 * r113;
    r67 = r34 * r7;
    r67 = r67 * r7;
    r67 = r67 * r56;
    r67 = r67 * r56;
    r67 = r67 * r114;
    r67 = r67 * r43;
    r71 = r38 * r7;
    r71 = r71 * r110;
    r71 = r71 * r43;
    r71 = fma(r59, r71, r57 * r67);
    r67 = r60 * r121;
    r67 = r67 * r92;
    r67 = r67 * r99;
    r71 = fma(r105, r67, r71);
    r8 = r7 * r112;
    r8 = r8 * r121;
    r71 = fma(r47, r8, r71);
    r71 = r71 + r22;
    r71 = fma(r4, r71, r65 * r101);
    r101 = r21 * r34;
    r101 = r101 * r7;
    r101 = r101 * r76;
    r71 = fma(r53, r101, r71);
    r22 = r5 * r38;
    r22 = r22 * r62;
    r71 = fma(r68, r22, r71);
    r8 = r5 * r121;
    r71 = fma(r122, r8, r71);
    r67 = r84 * r121;
    r67 = r67 * r30;
    r67 = r67 * r92;
    r71 = fma(r77, r67, r71);
    r26 = r7 * r42;
    r71 = fma(r78, r26, r71);
    r85 = r56 * r72;
    r85 = r85 * r1;
    r85 = r85 * r121;
    r85 = r85 * r43;
    r85 = r85 * r75;
    r71 = fma(r92, r85, r71);
    r9 = r5 * r34;
    r9 = r9 * r7;
    r9 = r9 * r56;
    r9 = r9 * r56;
    r9 = r9 * r95;
    r9 = r9 * r57;
    r71 = fma(r62, r9, r71);
    r104 = r56 * r1;
    r104 = r104 * r121;
    r104 = r104 * r43;
    r104 = r104 * r75;
    r71 = fma(r92, r104, r71);
    r91 = r84 * r72;
    r91 = r91 * r121;
    r91 = r91 * r30;
    r91 = r91 * r92;
    r71 = fma(r77, r91, r71);
    r128 = r21 * r34;
    r128 = r128 * r7;
    r128 = r128 * r72;
    r128 = r128 * r76;
    r71 = fma(r53, r128, r71);
    r129 = r5 * r55;
    r71 = fma(r107, r129, r71);
    r71 = fma(r38, r79, r71);
    r71 = fma(r121, r118, r71);
    r71 = fma(r38, r78, r71);
    r129 = r3 * r71;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r89, r129);
    r129 = r6 * r6;
    r89 = r10 * r41;
    r89 = r89 * r7;
    r89 = fma(r32, r80, r39 * r89);
    r128 = r10 * r54;
    r128 = r128 * r6;
    r89 = fma(r39, r128, r89);
    r91 = r32 * r6;
    r91 = r91 * r6;
    r89 = fma(r31, r91, r89);
    r129 = r129 * r112;
    r129 = r129 * r89;
    r129 = r129 * r66;
    r129 = r129 * r49;
    r91 = r60 * r89;
    r91 = r91 * r66;
    r91 = r91 * r99;
    r91 = fma(r63, r91, r59 * r129);
    r129 = r54 * r110;
    r129 = r129 * r59;
    r91 = fma(r62, r129, r91);
    r128 = fma(r41, r107, r32 * r103);
    r104 = r89 * r92;
    r104 = r104 * r99;
    r128 = fma(r105, r104, r128);
    r9 = r21 * r7;
    r9 = r9 * r89;
    r128 = fma(r47, r9, r128);
    r91 = fma(r32, r115, r91);
    r91 = r91 + r128;
    r129 = r21 * r6;
    r129 = r129 * r6;
    r129 = r129 * r89;
    r129 = r129 * r66;
    r129 = r129 * r49;
    r9 = r89 * r66;
    r9 = r9 * r99;
    r9 = fma(r63, r9, r59 * r129);
    r129 = r54 * r62;
    r9 = fma(r68, r129, r9);
    r104 = r32 * r31;
    r9 = fma(r81, r104, r9);
    r128 = r128 + r9;
    r91 = fma(r64, r128, r5 * r91);
    r104 = r4 * r54;
    r91 = fma(r107, r104, r91);
    r129 = r4 * r41;
    r129 = r129 * r62;
    r91 = fma(r68, r129, r91);
    r85 = r70 * r10;
    r85 = r85 * r48;
    r85 = fma(r128, r85, r69 * r128);
    r85 = fma(r128, r74, r85);
    r85 = fma(r128, r73, r85);
    r26 = r6 * r85;
    r91 = fma(r78, r26, r91);
    r67 = r4 * r89;
    r91 = fma(r122, r67, r91);
    r8 = r21 * r32;
    r8 = r8 * r6;
    r8 = r8 * r76;
    r91 = fma(r53, r8, r91);
    r22 = r56 * r72;
    r22 = r22 * r1;
    r22 = r22 * r89;
    r22 = r22 * r75;
    r22 = r22 * r66;
    r91 = fma(r62, r22, r91);
    r101 = r4 * r35;
    r101 = r101 * r6;
    r101 = r101 * r89;
    r91 = fma(r47, r101, r91);
    r131 = r21 * r32;
    r131 = r131 * r6;
    r131 = r131 * r72;
    r131 = r131 * r76;
    r91 = fma(r53, r131, r91);
    r102 = r56 * r1;
    r102 = r102 * r89;
    r102 = r102 * r75;
    r102 = r102 * r66;
    r91 = fma(r62, r102, r91);
    r91 = fma(r54, r79, r91);
    r91 = fma(r32, r120, r91);
    r91 = fma(r89, r117, r91);
    r91 = fma(r89, r116, r91);
    r91 = fma(r54, r78, r91);
    r102 = r2 * r91;
    r131 = r32 * r7;
    r131 = r131 * r7;
    r131 = r131 * r56;
    r131 = r131 * r56;
    r131 = r131 * r114;
    r131 = r131 * r43;
    r101 = r41 * r7;
    r101 = r101 * r110;
    r101 = r101 * r43;
    r101 = fma(r59, r101, r57 * r131);
    r131 = r60 * r89;
    r131 = r131 * r92;
    r131 = r131 * r99;
    r101 = fma(r105, r131, r101);
    r22 = r7 * r112;
    r22 = r22 * r89;
    r101 = fma(r47, r22, r101);
    r101 = r101 + r9;
    r101 = fma(r4, r101, r65 * r128);
    r128 = r56 * r1;
    r128 = r128 * r89;
    r128 = r128 * r43;
    r128 = r128 * r75;
    r101 = fma(r92, r128, r101);
    r9 = r21 * r32;
    r9 = r9 * r7;
    r9 = r9 * r76;
    r101 = fma(r53, r9, r101);
    r22 = r5 * r54;
    r101 = fma(r107, r22, r101);
    r131 = r21 * r32;
    r131 = r131 * r7;
    r131 = r131 * r72;
    r131 = r131 * r76;
    r101 = fma(r53, r131, r101);
    r8 = r84 * r72;
    r8 = r8 * r89;
    r8 = r8 * r30;
    r8 = r8 * r92;
    r101 = fma(r77, r8, r101);
    r67 = r5 * r32;
    r67 = r67 * r7;
    r67 = r67 * r56;
    r67 = r67 * r56;
    r67 = r67 * r95;
    r67 = r67 * r57;
    r101 = fma(r62, r67, r101);
    r26 = r5 * r41;
    r26 = r26 * r62;
    r101 = fma(r68, r26, r101);
    r129 = r56 * r72;
    r129 = r129 * r1;
    r129 = r129 * r89;
    r129 = r129 * r43;
    r129 = r129 * r75;
    r101 = fma(r92, r129, r101);
    r104 = r5 * r89;
    r101 = fma(r122, r104, r101);
    r108 = r84 * r89;
    r108 = r108 * r30;
    r108 = r108 * r92;
    r101 = fma(r77, r108, r101);
    r130 = r7 * r85;
    r101 = fma(r78, r130, r101);
    r101 = fma(r41, r78, r101);
    r101 = fma(r41, r79, r101);
    r101 = fma(r89, r118, r101);
    r130 = r3 * r101;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             10 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r102,
                                             r130);
    r130 = r3 * r21;
    r130 = r130 * r0;
    r61 = r21 * r61;
    r102 = r2 * r61;
    r130 = fma(r106, r102, r109 * r130);
    r108 = r3 * r21;
    r108 = r108 * r0;
    r108 = fma(r100, r102, r111 * r108);
    WriteSum2<double, double>((double*)inout_shared, r130, r108);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r108 = r3 * r21;
    r108 = r108 * r0;
    r108 = fma(r23, r102, r132 * r108);
    r130 = r3 * r21;
    r130 = r130 * r0;
    r130 = fma(r124, r102, r82 * r130);
    WriteSum2<double, double>((double*)inout_shared, r108, r130);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r130 = r3 * r21;
    r130 = r130 * r0;
    r130 = fma(r113, r102, r71 * r130);
    r108 = r3 * r21;
    r108 = r108 * r0;
    r108 = fma(r91, r102, r101 * r108);
    WriteSum2<double, double>((double*)inout_shared, r130, r108);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r108 = r2 * r2;
    r130 = r106 * r108;
    r104 = r3 * r3;
    r129 = r109 * r104;
    r109 = fma(r109, r129, r106 * r130);
    r106 = r100 * r100;
    r26 = r111 * r111;
    r26 = fma(r104, r26, r108 * r106);
    WriteSum2<double, double>((double*)inout_shared, r109, r26);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = r23 * r23;
    r109 = r132 * r132;
    r109 = fma(r104, r109, r108 * r26);
    r26 = r82 * r82;
    r106 = r124 * r124;
    r106 = fma(r108, r106, r104 * r26);
    WriteSum2<double, double>((double*)inout_shared, r109, r106);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r106 = r113 * r113;
    r109 = r71 * r71;
    r109 = fma(r104, r109, r108 * r106);
    r106 = r101 * r101;
    r26 = r91 * r91;
    r26 = fma(r108, r26, r104 * r106);
    WriteSum2<double, double>((double*)inout_shared, r109, r26);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fma(r100, r130, r111 * r129);
    r109 = fma(r132, r129, r23 * r130);
    WriteSum2<double, double>((double*)inout_shared, r26, r109);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r109 = fma(r82, r129, r124 * r130);
    r26 = fma(r71, r129, r113 * r130);
    WriteSum2<double, double>((double*)inout_shared, r109, r26);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r130 = fma(r91, r130, r101 * r129);
    r129 = r111 * r132;
    r26 = r100 * r23;
    r26 = fma(r108, r26, r104 * r129);
    WriteSum2<double, double>((double*)inout_shared, r130, r26);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = r100 * r124;
    r130 = r111 * r82;
    r130 = fma(r104, r130, r108 * r26);
    r26 = r100 * r113;
    r129 = r111 * r71;
    r129 = fma(r104, r129, r108 * r26);
    WriteSum2<double, double>((double*)inout_shared, r130, r129);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r129 = r111 * r101;
    r130 = r100 * r91;
    r130 = fma(r108, r130, r104 * r129);
    r129 = r132 * r82;
    r26 = r23 * r124;
    r26 = fma(r108, r26, r104 * r129);
    WriteSum2<double, double>((double*)inout_shared, r130, r26);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = r23 * r113;
    r130 = r132 * r71;
    r130 = fma(r104, r130, r108 * r26);
    r26 = r132 * r101;
    r129 = r23 * r91;
    r129 = fma(r108, r129, r104 * r26);
    WriteSum2<double, double>((double*)inout_shared, r130, r129);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r129 = r124 * r113;
    r130 = r82 * r71;
    r130 = fma(r104, r130, r108 * r129);
    r129 = r82 * r101;
    r26 = r124 * r91;
    r26 = fma(r108, r26, r104 * r129);
    WriteSum2<double, double>((double*)inout_shared, r130, r26);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = r71 * r101;
    r130 = r113 * r91;
    r130 = fma(r108, r130, r104 * r26);
    WriteSum1<double, double>((double*)inout_shared, r130);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r130 = r21 * r0;
    WriteSum2<double, double>((double*)inout_shared, r61, r130);
  };
  FlushSumShared<2, double>(out_principal_point_njtr,
                            0 * out_principal_point_njtr_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r36, r36);
  };
  FlushSumShared<2, double>(out_principal_point_precond_diag,
                            0 * out_principal_point_precond_diag_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = r21 * r7;
    r130 = r10 * r51;
    r130 = r130 * r6;
    r130 = fma(r58, r80, r39 * r130);
    r61 = r58 * r6;
    r61 = r61 * r6;
    r130 = fma(r31, r61, r130);
    r26 = r10 * r28;
    r26 = r26 * r7;
    r130 = fma(r39, r26, r130);
    r36 = r36 * r130;
    r36 = fma(r47, r36, r58 * r103);
    r26 = r130 * r92;
    r26 = r26 * r99;
    r36 = fma(r105, r26, r36);
    r36 = fma(r28, r107, r36);
    r26 = r58 * r31;
    r61 = r130 * r66;
    r61 = r61 * r99;
    r61 = fma(r63, r61, r81 * r26);
    r26 = r51 * r62;
    r61 = fma(r68, r26, r61);
    r129 = r21 * r6;
    r129 = r129 * r6;
    r129 = r129 * r130;
    r129 = r129 * r66;
    r129 = r129 * r49;
    r61 = fma(r59, r129, r61);
    r129 = r36 + r61;
    r26 = r60 * r130;
    r26 = r26 * r66;
    r26 = r26 * r99;
    r26 = fma(r63, r26, r58 * r115);
    r109 = r51 * r110;
    r109 = r109 * r59;
    r26 = fma(r62, r109, r26);
    r106 = r6 * r6;
    r106 = r106 * r112;
    r106 = r106 * r130;
    r106 = r106 * r66;
    r106 = r106 * r49;
    r26 = fma(r59, r106, r26);
    r26 = r26 + r36;
    r26 = fma(r5, r26, r64 * r129);
    r36 = r21 * r58;
    r36 = r36 * r6;
    r36 = r36 * r72;
    r36 = r36 * r76;
    r26 = fma(r53, r36, r26);
    r106 = r130 * r122;
    r109 = r4 * r35;
    r109 = r109 * r6;
    r109 = r109 * r130;
    r26 = fma(r47, r109, r26);
    r67 = r4 * r51;
    r26 = fma(r107, r67, r26);
    r8 = r21 * r58;
    r8 = r8 * r6;
    r8 = r8 * r76;
    r26 = fma(r53, r8, r26);
    r131 = r56 * r72;
    r131 = r131 * r1;
    r131 = r131 * r130;
    r131 = r131 * r75;
    r131 = r131 * r66;
    r26 = fma(r62, r131, r26);
    r22 = r70 * r10;
    r22 = r22 * r48;
    r22 = fma(r69, r129, r129 * r22);
    r22 = fma(r129, r74, r22);
    r22 = fma(r129, r73, r22);
    r9 = r6 * r22;
    r26 = fma(r78, r9, r26);
    r128 = r4 * r28;
    r128 = r128 * r62;
    r26 = fma(r68, r128, r26);
    r134 = r56 * r1;
    r134 = r134 * r130;
    r134 = r134 * r75;
    r134 = r134 * r66;
    r26 = fma(r62, r134, r26);
    r26 = fma(r130, r117, r26);
    r26 = fma(r4, r106, r26);
    r26 = fma(r130, r116, r26);
    r26 = fma(r51, r78, r26);
    r26 = fma(r58, r120, r26);
    r26 = fma(r51, r79, r26);
    r134 = r2 * r26;
    r128 = r58 * r7;
    r128 = r128 * r7;
    r128 = r128 * r56;
    r128 = r128 * r56;
    r128 = r128 * r114;
    r128 = r128 * r43;
    r9 = r7 * r112;
    r9 = r9 * r130;
    r9 = fma(r47, r9, r57 * r128);
    r128 = r60 * r130;
    r128 = r128 * r92;
    r128 = r128 * r99;
    r9 = fma(r105, r128, r9);
    r131 = r28 * r7;
    r131 = r131 * r110;
    r131 = r131 * r43;
    r9 = fma(r59, r131, r9);
    r9 = r9 + r61;
    r9 = fma(r4, r9, r65 * r129);
    r129 = r7 * r22;
    r9 = fma(r78, r129, r9);
    r61 = r84 * r130;
    r61 = r61 * r30;
    r61 = r61 * r92;
    r9 = fma(r77, r61, r9);
    r131 = r21 * r58;
    r131 = r131 * r7;
    r131 = r131 * r76;
    r9 = fma(r53, r131, r9);
    r128 = r5 * r51;
    r9 = fma(r107, r128, r9);
    r8 = r56 * r1;
    r8 = r8 * r130;
    r8 = r8 * r43;
    r8 = r8 * r75;
    r9 = fma(r92, r8, r9);
    r67 = r84 * r72;
    r67 = r67 * r130;
    r67 = r67 * r30;
    r67 = r67 * r92;
    r9 = fma(r77, r67, r9);
    r109 = r56 * r72;
    r109 = r109 * r1;
    r109 = r109 * r130;
    r109 = r109 * r43;
    r109 = r109 * r75;
    r9 = fma(r92, r109, r9);
    r36 = r21 * r58;
    r36 = r36 * r7;
    r36 = r36 * r72;
    r36 = r36 * r76;
    r9 = fma(r53, r36, r9);
    r135 = r5 * r58;
    r135 = r135 * r7;
    r135 = r135 * r56;
    r135 = r135 * r56;
    r135 = r135 * r95;
    r135 = r135 * r57;
    r9 = fma(r62, r135, r9);
    r136 = r5 * r28;
    r136 = r136 * r62;
    r9 = fma(r68, r136, r9);
    r9 = fma(r28, r79, r9);
    r9 = fma(r5, r106, r9);
    r9 = fma(r130, r118, r9);
    r9 = fma(r28, r78, r9);
    r136 = r3 * r9;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r134,
                                             r136);
    r136 = r45 * r110;
    r136 = r136 * r59;
    r136 = fma(r40, r115, r62 * r136);
    r134 = r6 * r6;
    r135 = r40 * r6;
    r135 = r135 * r6;
    r36 = r10 * r45;
    r36 = r36 * r6;
    r36 = fma(r39, r36, r31 * r135);
    r135 = r10 * r46;
    r135 = r135 * r7;
    r36 = fma(r39, r135, r36);
    r36 = fma(r40, r80, r36);
    r134 = r134 * r112;
    r134 = r134 * r36;
    r134 = r134 * r66;
    r134 = r134 * r49;
    r136 = fma(r59, r134, r136);
    r135 = r60 * r36;
    r135 = r135 * r66;
    r135 = r135 * r99;
    r136 = fma(r63, r135, r136);
    r109 = r36 * r92;
    r109 = r109 * r99;
    r109 = fma(r105, r109, r40 * r103);
    r67 = r21 * r7;
    r67 = r67 * r36;
    r109 = fma(r47, r67, r109);
    r109 = fma(r46, r107, r109);
    r136 = r136 + r109;
    r135 = r45 * r62;
    r134 = r40 * r31;
    r134 = fma(r81, r134, r68 * r135);
    r135 = r21 * r6;
    r135 = r135 * r6;
    r135 = r135 * r36;
    r135 = r135 * r66;
    r135 = r135 * r49;
    r134 = fma(r59, r135, r134);
    r67 = r36 * r66;
    r67 = r67 * r99;
    r134 = fma(r63, r67, r134);
    r109 = r109 + r134;
    r136 = fma(r64, r109, r5 * r136);
    r67 = r21 * r40;
    r67 = r67 * r6;
    r67 = r67 * r72;
    r67 = r67 * r76;
    r136 = fma(r53, r67, r136);
    r135 = r70 * r10;
    r135 = r135 * r48;
    r135 = fma(r109, r135, r69 * r109);
    r135 = fma(r109, r73, r135);
    r135 = fma(r109, r74, r135);
    r8 = r6 * r135;
    r136 = fma(r78, r8, r136);
    r128 = r21 * r40;
    r128 = r128 * r6;
    r128 = r128 * r76;
    r136 = fma(r53, r128, r136);
    r106 = r4 * r46;
    r106 = r106 * r62;
    r136 = fma(r68, r106, r136);
    r131 = r4 * r45;
    r136 = fma(r107, r131, r136);
    r61 = r56 * r1;
    r61 = r61 * r36;
    r61 = r61 * r75;
    r61 = r61 * r66;
    r136 = fma(r62, r61, r136);
    r129 = r4 * r36;
    r136 = fma(r122, r129, r136);
    r137 = r4 * r35;
    r137 = r137 * r6;
    r137 = r137 * r36;
    r136 = fma(r47, r137, r136);
    r138 = r56 * r72;
    r138 = r138 * r1;
    r138 = r138 * r36;
    r138 = r138 * r75;
    r138 = r138 * r66;
    r136 = fma(r62, r138, r136);
    r136 = fma(r45, r79, r136);
    r136 = fma(r36, r116, r136);
    r136 = fma(r36, r117, r136);
    r136 = fma(r45, r78, r136);
    r136 = fma(r40, r120, r136);
    r138 = r2 * r136;
    r137 = r40 * r7;
    r137 = r137 * r7;
    r137 = r137 * r56;
    r137 = r137 * r56;
    r137 = r137 * r114;
    r137 = r137 * r43;
    r129 = r60 * r36;
    r129 = r129 * r92;
    r129 = r129 * r99;
    r129 = fma(r105, r129, r57 * r137);
    r137 = r46 * r7;
    r137 = r137 * r110;
    r137 = r137 * r43;
    r129 = fma(r59, r137, r129);
    r61 = r7 * r112;
    r61 = r61 * r36;
    r129 = fma(r47, r61, r129);
    r129 = r129 + r134;
    r109 = fma(r65, r109, r4 * r129);
    r129 = r21 * r40;
    r129 = r129 * r7;
    r129 = r129 * r76;
    r109 = fma(r53, r129, r109);
    r134 = r21 * r40;
    r134 = r134 * r7;
    r134 = r134 * r72;
    r134 = r134 * r76;
    r109 = fma(r53, r134, r109);
    r61 = r84 * r36;
    r61 = r61 * r30;
    r61 = r61 * r92;
    r109 = fma(r77, r61, r109);
    r137 = r5 * r46;
    r137 = r137 * r62;
    r109 = fma(r68, r137, r109);
    r131 = r56 * r1;
    r131 = r131 * r36;
    r131 = r131 * r43;
    r131 = r131 * r75;
    r109 = fma(r92, r131, r109);
    r106 = r5 * r45;
    r109 = fma(r107, r106, r109);
    r128 = r5 * r36;
    r109 = fma(r122, r128, r109);
    r8 = r5 * r40;
    r8 = r8 * r7;
    r8 = r8 * r56;
    r8 = r8 * r56;
    r8 = r8 * r95;
    r8 = r8 * r57;
    r109 = fma(r62, r8, r109);
    r67 = r84 * r72;
    r67 = r67 * r36;
    r67 = r67 * r30;
    r67 = r67 * r92;
    r109 = fma(r77, r67, r109);
    r139 = r7 * r135;
    r109 = fma(r78, r139, r109);
    r140 = r56 * r72;
    r140 = r140 * r1;
    r140 = r140 * r36;
    r140 = r140 * r43;
    r140 = r140 * r75;
    r109 = fma(r92, r140, r109);
    r109 = fma(r46, r79, r109);
    r109 = fma(r36, r118, r109);
    r109 = fma(r46, r78, r109);
    r140 = r3 * r109;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r138,
                                             r140);
    r140 = r6 * r6;
    r138 = r10 * r52;
    r138 = r138 * r6;
    r80 = fma(r50, r80, r39 * r138);
    r138 = r50 * r6;
    r138 = r138 * r6;
    r80 = fma(r31, r138, r80);
    r139 = r10 * r44;
    r139 = r139 * r7;
    r80 = fma(r39, r139, r80);
    r140 = r140 * r112;
    r140 = r140 * r80;
    r140 = r140 * r66;
    r140 = r140 * r49;
    r140 = fma(r59, r140, r50 * r115);
    r115 = r60 * r80;
    r115 = r115 * r66;
    r115 = r115 * r99;
    r140 = fma(r63, r115, r140);
    r139 = r52 * r110;
    r139 = r139 * r59;
    r140 = fma(r62, r139, r140);
    r138 = r21 * r7;
    r138 = r138 * r80;
    r138 = fma(r47, r138, r44 * r107);
    r39 = r80 * r92;
    r39 = r39 * r99;
    r138 = fma(r105, r39, r138);
    r138 = fma(r50, r103, r138);
    r140 = r140 + r138;
    r139 = r50 * r31;
    r115 = r21 * r6;
    r115 = r115 * r6;
    r115 = r115 * r80;
    r115 = r115 * r66;
    r115 = r115 * r49;
    r115 = fma(r59, r115, r81 * r139);
    r139 = r80 * r66;
    r139 = r139 * r99;
    r115 = fma(r63, r139, r115);
    r63 = r52 * r62;
    r115 = fma(r68, r63, r115);
    r138 = r138 + r115;
    r64 = fma(r64, r138, r5 * r140);
    r140 = r21 * r50;
    r140 = r140 * r6;
    r140 = r140 * r76;
    r64 = fma(r53, r140, r64);
    r63 = r4 * r44;
    r63 = r63 * r62;
    r64 = fma(r68, r63, r64);
    r139 = r4 * r35;
    r139 = r139 * r6;
    r139 = r139 * r80;
    r64 = fma(r47, r139, r64);
    r81 = r56 * r1;
    r81 = r81 * r80;
    r81 = r81 * r75;
    r81 = r81 * r66;
    r64 = fma(r62, r81, r64);
    r49 = r4 * r80;
    r64 = fma(r122, r49, r64);
    r103 = r4 * r52;
    r64 = fma(r107, r103, r64);
    r39 = r56 * r72;
    r39 = r39 * r1;
    r39 = r39 * r80;
    r39 = r39 * r75;
    r39 = r39 * r66;
    r64 = fma(r62, r39, r64);
    r67 = r21 * r50;
    r67 = r67 * r6;
    r67 = r67 * r72;
    r67 = r67 * r76;
    r64 = fma(r53, r67, r64);
    r8 = r70 * r10;
    r8 = r8 * r48;
    r8 = fma(r138, r8, r69 * r138);
    r8 = fma(r138, r74, r8);
    r8 = fma(r138, r73, r8);
    r73 = r6 * r8;
    r64 = fma(r78, r73, r64);
    r64 = fma(r52, r79, r64);
    r64 = fma(r80, r116, r64);
    r64 = fma(r80, r117, r64);
    r64 = fma(r50, r120, r64);
    r64 = fma(r52, r78, r64);
    r2 = r2 * r64;
    r73 = r44 * r7;
    r73 = r73 * r110;
    r73 = r73 * r43;
    r67 = r7 * r112;
    r67 = r67 * r80;
    r67 = fma(r47, r67, r59 * r73);
    r73 = r60 * r80;
    r73 = r73 * r92;
    r73 = r73 * r99;
    r67 = fma(r105, r73, r67);
    r105 = r50 * r7;
    r105 = r105 * r7;
    r105 = r105 * r56;
    r105 = r105 * r56;
    r105 = r105 * r114;
    r105 = r105 * r43;
    r67 = fma(r57, r105, r67);
    r67 = r67 + r115;
    r67 = fma(r4, r67, r65 * r138);
    r138 = r5 * r44;
    r138 = r138 * r62;
    r67 = fma(r68, r138, r67);
    r68 = r56 * r1;
    r68 = r68 * r80;
    r68 = r68 * r43;
    r68 = r68 * r75;
    r67 = fma(r92, r68, r67);
    r65 = r5 * r80;
    r67 = fma(r122, r65, r67);
    r122 = r56 * r72;
    r122 = r122 * r1;
    r122 = r122 * r80;
    r122 = r122 * r43;
    r122 = r122 * r75;
    r67 = fma(r92, r122, r67);
    r75 = r84 * r72;
    r75 = r75 * r80;
    r75 = r75 * r30;
    r75 = r75 * r92;
    r67 = fma(r77, r75, r67);
    r43 = r21 * r50;
    r43 = r43 * r7;
    r43 = r43 * r72;
    r43 = r43 * r76;
    r67 = fma(r53, r43, r67);
    r115 = r5 * r50;
    r115 = r115 * r7;
    r115 = r115 * r56;
    r115 = r115 * r56;
    r115 = r115 * r95;
    r115 = r115 * r57;
    r67 = fma(r62, r115, r67);
    r57 = r5 * r52;
    r67 = fma(r107, r57, r67);
    r107 = r7 * r8;
    r67 = fma(r78, r107, r67);
    r95 = r84 * r80;
    r95 = r95 * r30;
    r95 = r95 * r92;
    r67 = fma(r77, r95, r67);
    r77 = r21 * r50;
    r77 = r77 * r7;
    r77 = r77 * r76;
    r67 = fma(r53, r77, r67);
    r67 = fma(r80, r118, r67);
    r67 = fma(r44, r79, r67);
    r67 = fma(r44, r78, r67);
    r77 = r3 * r67;
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r2, r77);
    r77 = r3 * r21;
    r77 = r77 * r0;
    r77 = fma(r26, r102, r9 * r77);
    r2 = r3 * r21;
    r2 = r2 * r0;
    r2 = fma(r136, r102, r109 * r2);
    WriteSum2<double, double>((double*)inout_shared, r77, r2);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r3 * r21;
    r2 = r2 * r0;
    r102 = fma(r64, r102, r67 * r2);
    WriteSum1<double, double>((double*)inout_shared, r102);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r102 = r9 * r9;
    r2 = r26 * r26;
    r2 = fma(r108, r2, r104 * r102);
    r102 = r109 * r109;
    r0 = r136 * r136;
    r0 = fma(r108, r0, r104 * r102);
    WriteSum2<double, double>((double*)inout_shared, r2, r0);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r67 * r67;
    r2 = r64 * r64;
    r2 = fma(r108, r2, r104 * r0);
    WriteSum1<double, double>((double*)inout_shared, r2);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r9 * r109;
    r0 = r26 * r136;
    r0 = fma(r108, r0, r104 * r2);
    r2 = r9 * r67;
    r102 = r26 * r64;
    r102 = fma(r108, r102, r104 * r2);
    WriteSum2<double, double>((double*)inout_shared, r0, r102);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r102 = r109 * r67;
    r0 = r136 * r64;
    r0 = fma(r108, r0, r104 * r102);
    WriteSum1<double, double>((double*)inout_shared, r0);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void ThinPrismFisheyeSplitFixedFocalAndExtraResJacFirst(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
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
    double* out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    double* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    double* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    double* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
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
  ThinPrismFisheyeSplitFixedFocalAndExtraResJacFirstKernel<<<n_blocks, 1024>>>(
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
      focal_and_extra,
      focal_and_extra_num_alloc,
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