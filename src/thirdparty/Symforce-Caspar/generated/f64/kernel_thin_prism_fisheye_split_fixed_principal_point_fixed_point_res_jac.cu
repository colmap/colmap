#include "kernel_thin_prism_fisheye_split_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPrincipalPointFixedPointResJacKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        SharedIndex* focal_and_extra_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
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
        double* out_focal_and_extra_jac,
        unsigned int out_focal_and_extra_jac_num_alloc,
        double* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        double* const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        double* const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102, r103, r104,
      r105, r106, r107, r108, r109, r110, r111, r112, r113, r114, r115, r116,
      r117, r118, r119, r120, r121, r122, r123, r124, r125, r126, r127, r128,
      r129, r130, r131, r132, r133, r134, r135, r136, r137, r138;

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
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9);
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
    r28 = fma(r8, r28, r7);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r7, r30);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r31 = r17 * r18;
    r31 = r31 * r10;
    r32 = r13 * r14;
    r32 = fma(r10, r32, r31);
    r33 = r17 * r17;
    r34 = -2.00000000000000000e+00;
    r33 = r33 * r34;
    r35 = 1.00000000000000000e+00;
    r36 = r13 * r13;
    r36 = fma(r34, r36, r35);
    r37 = r33 + r36;
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r38);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r39 = r18 * r13;
    r39 = r39 * r10;
    r40 = r17 * r14;
    r40 = fma(r34, r40, r39);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r41);
    r42 = r10 * r27;
    r42 = r42 * r24;
    r43 = r19 * r34;
    r43 = fma(r29, r43, r42);
    r44 = r27 * r27;
    r44 = r44 * r34;
    r45 = r35 + r44;
    r46 = r19 * r19;
    r46 = r46 * r34;
    r45 = r45 + r46;
    r28 = fma(r7, r32, r28);
    r28 = fma(r30, r37, r28);
    r28 = fma(r38, r40, r28);
    r28 = fma(r41, r43, r28);
    r28 = fma(r9, r45, r28);
    r45 = r34 * r24;
    r45 = r45 * r24;
    r43 = r35 + r45;
    r43 = r43 + r44;
    r43 = fma(r8, r43, r6);
    r6 = r27 * r34;
    r6 = fma(r29, r6, r20);
    r20 = r10 * r27;
    r20 = r20 * r19;
    r44 = r10 * r24;
    r44 = fma(r29, r44, r20);
    r47 = r17 * r13;
    r47 = r47 * r10;
    r48 = r18 * r14;
    r48 = fma(r10, r48, r47);
    r49 = r13 * r14;
    r49 = fma(r34, r49, r31);
    r31 = r18 * r18;
    r31 = r31 * r34;
    r36 = r31 + r36;
    r43 = fma(r9, r6, r43);
    r43 = fma(r41, r44, r43);
    r43 = fma(r38, r48, r43);
    r43 = fma(r30, r49, r43);
    r43 = fma(r7, r36, r43);
    r44 = r43 * r43;
    r6 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r50);
    r51 = r34 * r24;
    r51 = fma(r29, r51, r20);
    r51 = fma(r8, r51, r50);
    r50 = r18 * r14;
    r50 = fma(r34, r50, r47);
    r31 = r35 + r31;
    r31 = r31 + r33;
    r33 = r17 * r14;
    r33 = fma(r10, r33, r39);
    r39 = r10 * r19;
    r39 = fma(r29, r39, r42);
    r45 = r35 + r45;
    r45 = r45 + r46;
    r51 = fma(r7, r50, r51);
    r51 = fma(r38, r31, r51);
    r51 = fma(r30, r33, r51);
    r51 = fma(r9, r39, r51);
    r51 = fma(r41, r45, r51);
    r45 = copysign(1.0, r51);
    r45 = fma(r6, r45, r51);
    r51 = r45 * r45;
    r39 = 1.0 / r51;
    r30 = r28 * r28;
    r30 = fma(r39, r30, r39 * r44);
    r44 = sqrt(r30);
    r38 = atan(r44);
    r7 = r28 * r38;
    r46 = copysign(1.0, r44);
    r46 = fma(r6, r46, r44);
    r6 = r46 * r46;
    r44 = 1.0 / r6;
    r42 = r39 * r44;
    r47 = r28 * r38;
    r7 = r7 * r42;
    r7 = r7 * r47;
    r20 = 3.00000000000000000e+00;
    r52 = r38 * r20;
    r53 = r43 * r38;
    r54 = r53 * r42;
    r55 = r43 * r54;
    r52 = fma(r55, r52, r7);
  };
  LoadShared<2, double, double>(focal_and_extra,
                                8 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target,
                        r56,
                        r57);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r58 = r38 * r55;
    r7 = r7 + r58;
    r59 = fma(r56, r7, r5 * r52);
    r60 = r4 * r10;
    r60 = r60 * r47;
    r59 = fma(r54, r60, r59);
  };
  LoadShared<2, double, double>(focal_and_extra,
                                2 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target,
                        r61,
                        r62);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r63 = r7 * r7;
    r64 = fma(r62, r63, r61 * r7);
  };
  LoadShared<2, double, double>(focal_and_extra,
                                6 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target,
                        r65,
                        r66);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r67 = r63 * r63;
    r68 = r7 * r63;
    r64 = fma(r66, r67, r64);
    r64 = fma(r65, r68, r64);
    r69 = 1.0 / r45;
    r70 = 1.0 / r46;
    r71 = r69 * r70;
    r72 = r64 * r71;
    r59 = fma(r53, r72, r59);
    r59 = fma(r53, r71, r59);
    r0 = fma(r2, r59, r0);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r60, r73);
    r0 = fma(r60, r21, r0);
    r60 = r28 * r38;
    r60 = r60 * r20;
    r60 = r60 * r42;
    r60 = fma(r47, r60, r58);
    r58 = fma(r57, r7, r4 * r60);
    r74 = r5 * r10;
    r74 = r74 * r47;
    r58 = fma(r54, r74, r58);
    r58 = fma(r47, r72, r58);
    r58 = fma(r71, r47, r58);
    r1 = fma(r3, r58, r1);
    r1 = fma(r73, r21, r1);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r0, r1);
    r73 = r10 * r24;
    r74 = -5.00000000000000000e-01;
    r75 = 5.00000000000000000e-01;
    r76 = fma(r75, r26, r74 * r22);
    r76 = fma(r74, r23, r76);
    r76 = fma(r74, r25, r76);
    r73 = r73 * r76;
    r77 = r10 * r27;
    r78 = r15 * r14;
    r79 = r12 * r17;
    r79 = fma(r75, r79, r75 * r78);
    r78 = r11 * r18;
    r79 = fma(r74, r78, r79);
    r80 = r16 * r13;
    r79 = fma(r75, r80, r79);
    r77 = fma(r79, r77, r73);
    r80 = r10 * r19;
    r78 = r11 * r14;
    r81 = r16 * r17;
    r81 = fma(r74, r81, r74 * r78);
    r78 = r15 * r18;
    r81 = fma(r74, r78, r81);
    r82 = r12 * r13;
    r81 = fma(r75, r82, r81);
    r80 = r80 * r81;
    r82 = r10 * r29;
    r78 = r16 * r14;
    r83 = r11 * r17;
    r83 = fma(r74, r83, r75 * r78);
    r78 = r12 * r18;
    r83 = fma(r74, r78, r83);
    r84 = r15 * r13;
    r83 = fma(r74, r84, r83);
    r82 = r82 * r83;
    r84 = r80 + r82;
    r78 = r77 + r84;
    r85 = r24 * r81;
    r86 = r29 * r79;
    r87 = fma(r34, r86, r34 * r85);
    r88 = r10 * r19;
    r89 = r10 * r27;
    r89 = r89 * r83;
    r88 = fma(r76, r88, r89);
    r87 = r87 + r88;
    r87 = fma(r8, r87, r9 * r78);
    r78 = r24 * r79;
    r90 = -4.00000000000000000e+00;
    r78 = r78 * r90;
    r91 = r19 * r83;
    r91 = r91 * r90;
    r92 = r78 + r91;
    r87 = fma(r41, r92, r87);
    r92 = r28 * r28;
    r51 = r45 * r51;
    r93 = 1.0 / r51;
    r94 = r34 * r93;
    r92 = r92 * r94;
    r95 = r38 * r92;
    r96 = r38 * r44;
    r96 = r96 * r94;
    r96 = r95 * r96;
    r97 = r10 * r43;
    r86 = fma(r10, r86, r10 * r85);
    r86 = r86 + r88;
    r98 = r10 * r24;
    r98 = r98 * r83;
    r99 = r10 * r19;
    r99 = r99 * r79;
    r79 = r98 + r99;
    r100 = r27 * r34;
    r79 = fma(r81, r100, r79);
    r101 = r34 * r29;
    r79 = fma(r76, r101, r79);
    r79 = fma(r9, r79, r41 * r86);
    r86 = r27 * r76;
    r101 = r90 * r86;
    r78 = r78 + r101;
    r79 = fma(r8, r78, r79);
    r97 = r97 * r79;
    r78 = r43 * r43;
    r78 = r78 * r87;
    r78 = fma(r94, r78, r39 * r97);
    r97 = r10 * r28;
    r100 = r19 * r34;
    r102 = r34 * r29;
    r102 = r102 * r83;
    r100 = fma(r81, r100, r102);
    r100 = r100 + r77;
    r101 = r91 + r101;
    r101 = fma(r9, r101, r41 * r100);
    r100 = r10 * r29;
    r100 = fma(r76, r100, r99);
    r99 = r10 * r27;
    r99 = fma(r81, r99, r98);
    r100 = r100 + r99;
    r101 = fma(r8, r100, r101);
    r97 = r97 * r101;
    r78 = fma(r39, r97, r78);
    r78 = fma(r87, r92, r78);
    r97 = r42 * r47;
    r100 = r78 * r97;
    r98 = rsqrt(r30);
    r30 = r35 + r30;
    r30 = 1.0 / r30;
    r35 = r98 * r30;
    r91 = r28 * r35;
    r100 = fma(r91, r100, r87 * r96);
    r77 = r21 * r28;
    r6 = r46 * r6;
    r103 = 1.0 / r6;
    r77 = r77 * r28;
    r77 = r77 * r38;
    r77 = r77 * r38;
    r77 = r77 * r78;
    r77 = r77 * r39;
    r77 = r77 * r98;
    r100 = fma(r103, r77, r100);
    r104 = r10 * r38;
    r105 = r104 * r97;
    r100 = fma(r101, r105, r100);
    r77 = r35 * r55;
    r106 = r21 * r43;
    r106 = r106 * r38;
    r106 = r106 * r78;
    r106 = r106 * r39;
    r106 = r106 * r98;
    r106 = r106 * r103;
    r106 = fma(r53, r106, r78 * r77);
    r107 = r79 * r54;
    r106 = fma(r104, r107, r106);
    r108 = r43 * r38;
    r108 = r108 * r87;
    r108 = r108 * r44;
    r108 = r108 * r53;
    r106 = fma(r94, r108, r106);
    r108 = r100 + r106;
    r107 = r20 * r78;
    r109 = r43 * r53;
    r110 = -3.00000000000000000e+00;
    r110 = r38 * r110;
    r110 = r110 * r39;
    r110 = r110 * r98;
    r110 = r110 * r103;
    r109 = r109 * r110;
    r107 = fma(r78, r109, r77 * r107);
    r111 = r38 * r79;
    r112 = 6.00000000000000000e+00;
    r111 = r111 * r112;
    r107 = fma(r54, r111, r107);
    r113 = r43 * r53;
    r114 = r38 * r87;
    r115 = -6.00000000000000000e+00;
    r114 = r114 * r115;
    r114 = r114 * r44;
    r114 = r114 * r93;
    r107 = fma(r114, r113, r107);
    r107 = r107 + r100;
    r107 = fma(r5, r107, r56 * r108);
    r100 = r4 * r105;
    r113 = r74 * r78;
    r113 = r113 * r44;
    r113 = r113 * r69;
    r113 = r113 * r98;
    r111 = r64 * r113;
    r116 = r21 * r64;
    r116 = r116 * r87;
    r116 = r116 * r39;
    r116 = r116 * r70;
    r107 = fma(r53, r116, r107);
    r117 = r43 * r78;
    r118 = r75 * r72;
    r117 = r117 * r35;
    r107 = fma(r118, r117, r107);
    r119 = r4 * r34;
    r119 = r119 * r78;
    r119 = r119 * r39;
    r119 = r119 * r98;
    r119 = r119 * r103;
    r119 = r119 * r53;
    r107 = fma(r47, r119, r107);
    r120 = r38 * r79;
    r107 = fma(r72, r120, r107);
    r121 = r4 * r101;
    r121 = r121 * r54;
    r107 = fma(r104, r121, r107);
    r122 = r4 * r87;
    r123 = r90 * r44;
    r123 = r123 * r93;
    r123 = r123 * r53;
    r123 = r123 * r47;
    r107 = fma(r123, r122, r107);
    r124 = r43 * r75;
    r124 = r124 * r78;
    r124 = r124 * r71;
    r107 = fma(r35, r124, r107);
    r125 = r4 * r78;
    r126 = r10 * r54;
    r126 = r126 * r91;
    r107 = fma(r126, r125, r107);
    r127 = r62 * r10;
    r127 = r127 * r7;
    r127 = fma(r61, r108, r108 * r127);
    r65 = r65 * r20;
    r65 = r65 * r63;
    r128 = 4.00000000000000000e+00;
    r66 = r66 * r128;
    r66 = r66 * r68;
    r127 = fma(r108, r65, r127);
    r127 = fma(r108, r66, r127);
    r129 = r127 * r53;
    r107 = fma(r71, r129, r107);
    r130 = r21 * r87;
    r130 = r130 * r39;
    r130 = r130 * r70;
    r107 = fma(r53, r130, r107);
    r131 = r38 * r79;
    r107 = fma(r71, r131, r107);
    r107 = fma(r79, r100, r107);
    r107 = fma(r53, r113, r107);
    r107 = fma(r53, r111, r107);
    r131 = r2 * r107;
    r130 = r28 * r28;
    r130 = r130 * r38;
    r129 = r20 * r78;
    r129 = r129 * r97;
    r129 = fma(r91, r129, r114 * r130);
    r130 = r78 * r110;
    r129 = fma(r95, r130, r129);
    r114 = r38 * r101;
    r114 = r114 * r112;
    r114 = r114 * r42;
    r129 = fma(r47, r114, r129);
    r129 = r129 + r106;
    r129 = fma(r4, r129, r57 * r108);
    r108 = r5 * r79;
    r129 = fma(r105, r108, r129);
    r106 = r21 * r28;
    r106 = r106 * r38;
    r106 = r106 * r87;
    r106 = r106 * r39;
    r129 = fma(r70, r106, r129);
    r114 = r5 * r34;
    r114 = r114 * r78;
    r114 = r114 * r39;
    r114 = r114 * r98;
    r114 = r114 * r103;
    r114 = r114 * r53;
    r129 = fma(r47, r114, r129);
    r130 = r75 * r78;
    r130 = r130 * r71;
    r129 = fma(r91, r130, r129);
    r125 = r5 * r101;
    r125 = r125 * r54;
    r129 = fma(r104, r125, r129);
    r124 = r5 * r123;
    r122 = r127 * r71;
    r129 = fma(r47, r122, r129);
    r121 = r5 * r78;
    r129 = fma(r126, r121, r129);
    r120 = r91 * r118;
    r119 = r38 * r101;
    r129 = fma(r72, r119, r129);
    r117 = r38 * r101;
    r129 = fma(r71, r117, r129);
    r116 = r21 * r28;
    r116 = r116 * r38;
    r116 = r116 * r64;
    r116 = r116 * r87;
    r116 = r116 * r39;
    r129 = fma(r70, r116, r129);
    r129 = fma(r47, r111, r129);
    r129 = fma(r113, r47, r129);
    r129 = fma(r87, r124, r129);
    r129 = fma(r78, r120, r129);
    r116 = r3 * r129;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             0 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r131,
                                             r116);
    r116 = r43 * r38;
    r131 = r34 * r24;
    r131 = fma(r76, r131, r102);
    r117 = r10 * r27;
    r119 = r15 * r14;
    r121 = r12 * r17;
    r121 = fma(r74, r121, r74 * r119);
    r119 = r11 * r18;
    r121 = fma(r75, r119, r121);
    r122 = r16 * r13;
    r121 = fma(r74, r122, r121);
    r117 = r117 * r121;
    r122 = r10 * r19;
    r119 = r11 * r14;
    r125 = r16 * r17;
    r125 = fma(r75, r125, r75 * r119);
    r119 = r15 * r18;
    r125 = fma(r75, r119, r125);
    r130 = r12 * r13;
    r125 = fma(r74, r130, r125);
    r122 = fma(r125, r122, r117);
    r131 = r131 + r122;
    r130 = r10 * r24;
    r130 = r130 * r125;
    r119 = r10 * r29;
    r119 = fma(r121, r119, r130);
    r119 = r119 + r88;
    r119 = fma(r9, r119, r8 * r131);
    r131 = r24 * r83;
    r131 = r131 * r90;
    r88 = r19 * r121;
    r114 = r90 * r88;
    r113 = r131 + r114;
    r119 = fma(r41, r113, r119);
    r116 = r116 * r115;
    r116 = r116 * r119;
    r116 = r116 * r44;
    r116 = r116 * r93;
    r113 = r38 * r112;
    r82 = r73 + r82;
    r82 = r82 + r122;
    r122 = r27 * r90;
    r122 = r122 * r125;
    r131 = r131 + r122;
    r131 = fma(r8, r131, r41 * r82);
    r82 = r34 * r29;
    r82 = fma(r34, r86, r125 * r82);
    r73 = r10 * r19;
    r73 = r73 * r83;
    r111 = r10 * r24;
    r111 = fma(r121, r111, r73);
    r82 = r82 + r111;
    r131 = fma(r9, r82, r131);
    r113 = r113 * r131;
    r113 = fma(r54, r113, r53 * r116);
    r116 = r10 * r43;
    r116 = r116 * r131;
    r82 = r10 * r28;
    r130 = r89 + r130;
    r89 = r19 * r34;
    r130 = fma(r76, r89, r130);
    r76 = r34 * r29;
    r130 = fma(r121, r76, r130);
    r76 = r10 * r29;
    r86 = fma(r10, r86, r125 * r76);
    r86 = r86 + r111;
    r86 = fma(r8, r86, r41 * r130);
    r114 = r122 + r114;
    r86 = fma(r9, r114, r86);
    r82 = r82 * r86;
    r82 = fma(r39, r82, r39 * r116);
    r116 = r43 * r43;
    r116 = r116 * r119;
    r82 = fma(r94, r116, r82);
    r82 = fma(r119, r92, r82);
    r116 = r20 * r82;
    r113 = fma(r77, r116, r113);
    r114 = r21 * r28;
    r114 = r114 * r28;
    r114 = r114 * r38;
    r114 = r114 * r38;
    r114 = r114 * r82;
    r114 = r114 * r39;
    r114 = r114 * r98;
    r114 = fma(r103, r114, r86 * r105);
    r122 = r82 * r97;
    r114 = fma(r91, r122, r114);
    r114 = fma(r119, r96, r114);
    r113 = fma(r82, r109, r113);
    r113 = r113 + r114;
    r116 = r43 * r38;
    r116 = r116 * r119;
    r116 = r116 * r44;
    r116 = r116 * r53;
    r122 = r131 * r54;
    r122 = fma(r104, r122, r94 * r116);
    r116 = r21 * r43;
    r116 = r116 * r38;
    r116 = r116 * r82;
    r116 = r116 * r39;
    r116 = r116 * r98;
    r116 = r116 * r103;
    r122 = fma(r53, r116, r122);
    r122 = fma(r82, r77, r122);
    r114 = r114 + r122;
    r113 = fma(r56, r114, r5 * r113);
    r116 = r21 * r119;
    r116 = r116 * r39;
    r116 = r116 * r70;
    r113 = fma(r53, r116, r113);
    r130 = r43 * r82;
    r130 = r130 * r35;
    r113 = fma(r118, r130, r113);
    r76 = r43 * r75;
    r76 = r76 * r82;
    r76 = r76 * r71;
    r113 = fma(r35, r76, r113);
    r125 = r64 * r74;
    r125 = r125 * r82;
    r125 = r125 * r44;
    r125 = r125 * r69;
    r125 = r125 * r98;
    r113 = fma(r53, r125, r113);
    r89 = r62 * r10;
    r89 = r89 * r7;
    r89 = fma(r114, r89, r61 * r114);
    r89 = fma(r114, r66, r89);
    r89 = fma(r114, r65, r89);
    r106 = r89 * r53;
    r113 = fma(r71, r106, r113);
    r108 = r21 * r64;
    r108 = r108 * r119;
    r108 = r108 * r39;
    r108 = r108 * r70;
    r113 = fma(r53, r108, r113);
    r132 = r4 * r119;
    r113 = fma(r123, r132, r113);
    r133 = r4 * r82;
    r113 = fma(r126, r133, r113);
    r134 = r38 * r131;
    r113 = fma(r71, r134, r113);
    r135 = r4 * r34;
    r135 = r135 * r82;
    r135 = r135 * r39;
    r135 = r135 * r98;
    r135 = r135 * r103;
    r135 = r135 * r53;
    r113 = fma(r47, r135, r113);
    r136 = r38 * r131;
    r113 = fma(r72, r136, r113);
    r137 = r4 * r86;
    r137 = r137 * r54;
    r113 = fma(r104, r137, r113);
    r138 = r74 * r82;
    r138 = r138 * r44;
    r138 = r138 * r69;
    r138 = r138 * r98;
    r113 = fma(r53, r138, r113);
    r113 = fma(r131, r100, r113);
    r138 = r2 * r113;
    r137 = r38 * r112;
    r137 = r137 * r86;
    r137 = r137 * r42;
    r136 = r82 * r110;
    r136 = fma(r95, r136, r47 * r137);
    r137 = r28 * r28;
    r137 = r137 * r38;
    r137 = r137 * r38;
    r137 = r137 * r115;
    r137 = r137 * r119;
    r137 = r137 * r44;
    r136 = fma(r93, r137, r136);
    r135 = r20 * r82;
    r135 = r135 * r97;
    r136 = fma(r91, r135, r136);
    r136 = r136 + r122;
    r114 = fma(r57, r114, r4 * r136);
    r136 = r28 * r38;
    r136 = r136 * r74;
    r136 = r136 * r82;
    r136 = r136 * r44;
    r136 = r136 * r69;
    r114 = fma(r98, r136, r114);
    r122 = r38 * r86;
    r114 = fma(r72, r122, r114);
    r135 = r89 * r71;
    r114 = fma(r47, r135, r114);
    r137 = r75 * r82;
    r137 = r137 * r71;
    r114 = fma(r91, r137, r114);
    r134 = r5 * r82;
    r114 = fma(r126, r134, r114);
    r133 = r21 * r28;
    r133 = r133 * r38;
    r133 = r133 * r64;
    r133 = r133 * r119;
    r133 = r133 * r39;
    r114 = fma(r70, r133, r114);
    r132 = r28 * r38;
    r132 = r132 * r64;
    r132 = r132 * r74;
    r132 = r132 * r82;
    r132 = r132 * r44;
    r132 = r132 * r69;
    r114 = fma(r98, r132, r114);
    r108 = r5 * r34;
    r108 = r108 * r82;
    r108 = r108 * r39;
    r108 = r108 * r98;
    r108 = r108 * r103;
    r108 = r108 * r53;
    r114 = fma(r47, r108, r114);
    r106 = r21 * r28;
    r106 = r106 * r38;
    r106 = r106 * r119;
    r106 = r106 * r39;
    r114 = fma(r70, r106, r114);
    r125 = r38 * r86;
    r114 = fma(r71, r125, r114);
    r76 = r5 * r86;
    r76 = r76 * r54;
    r114 = fma(r104, r76, r114);
    r130 = r5 * r131;
    r114 = fma(r105, r130, r114);
    r114 = fma(r119, r124, r114);
    r114 = fma(r82, r120, r114);
    r130 = r3 * r114;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             2 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r138,
                                             r130);
    r130 = r19 * r90;
    r26 = fma(r74, r26, r75 * r22);
    r26 = fma(r75, r23, r26);
    r26 = fma(r75, r25, r26);
    r130 = r130 * r26;
    r85 = r90 * r85;
    r25 = r130 + r85;
    r23 = r10 * r27;
    r23 = r23 * r26;
    r73 = r73 + r23;
    r22 = r34 * r24;
    r73 = fma(r121, r22, r73);
    r138 = r34 * r29;
    r73 = fma(r81, r138, r73);
    r73 = fma(r8, r73, r41 * r25);
    r25 = r10 * r29;
    r25 = fma(r10, r88, r26 * r25);
    r25 = r25 + r99;
    r73 = fma(r9, r25, r73);
    r25 = r10 * r28;
    r138 = r10 * r24;
    r138 = r138 * r26;
    r117 = r117 + r138;
    r117 = r117 + r84;
    r84 = r34 * r29;
    r88 = fma(r34, r88, r26 * r84);
    r88 = r88 + r99;
    r88 = fma(r41, r88, r8 * r117);
    r83 = r27 * r83;
    r83 = r83 * r90;
    r130 = r130 + r83;
    r88 = fma(r9, r130, r88);
    r25 = r25 * r88;
    r130 = r43 * r43;
    r130 = r130 * r73;
    r130 = fma(r94, r130, r39 * r25);
    r25 = r10 * r43;
    r102 = r80 + r102;
    r80 = r27 * r34;
    r102 = fma(r121, r80, r102);
    r102 = r102 + r138;
    r85 = r83 + r85;
    r85 = fma(r8, r85, r9 * r102);
    r8 = r10 * r29;
    r8 = fma(r81, r8, r23);
    r8 = r8 + r111;
    r85 = fma(r41, r8, r85);
    r25 = r25 * r85;
    r130 = fma(r39, r25, r130);
    r130 = fma(r73, r92, r130);
    r25 = r130 * r97;
    r25 = fma(r91, r25, r73 * r96);
    r8 = r21 * r28;
    r8 = r8 * r28;
    r8 = r8 * r38;
    r8 = r8 * r38;
    r8 = r8 * r130;
    r8 = r8 * r39;
    r8 = r8 * r98;
    r25 = fma(r103, r8, r25);
    r25 = fma(r88, r105, r25);
    r8 = r21 * r43;
    r8 = r8 * r38;
    r8 = r8 * r130;
    r8 = r8 * r39;
    r8 = r8 * r98;
    r8 = r8 * r103;
    r41 = r85 * r54;
    r41 = fma(r104, r41, r53 * r8);
    r8 = r43 * r38;
    r8 = r8 * r73;
    r8 = r8 * r44;
    r8 = r8 * r53;
    r41 = fma(r94, r8, r41);
    r41 = fma(r130, r77, r41);
    r8 = r25 + r41;
    r111 = r38 * r112;
    r111 = r111 * r85;
    r111 = fma(r54, r111, r130 * r109);
    r23 = r43 * r38;
    r23 = r23 * r115;
    r23 = r23 * r73;
    r23 = r23 * r44;
    r23 = r23 * r93;
    r111 = fma(r53, r23, r111);
    r81 = r20 * r130;
    r111 = fma(r77, r81, r111);
    r111 = r111 + r25;
    r111 = fma(r5, r111, r56 * r8);
    r25 = r62 * r10;
    r25 = r25 * r7;
    r25 = fma(r8, r25, r61 * r8);
    r25 = fma(r8, r65, r25);
    r25 = fma(r8, r66, r25);
    r81 = r25 * r53;
    r111 = fma(r71, r81, r111);
    r23 = r43 * r75;
    r23 = r23 * r130;
    r23 = r23 * r71;
    r111 = fma(r35, r23, r111);
    r102 = r4 * r130;
    r111 = fma(r126, r102, r111);
    r9 = r21 * r64;
    r9 = r9 * r73;
    r9 = r9 * r39;
    r9 = r9 * r70;
    r111 = fma(r53, r9, r111);
    r83 = r43 * r130;
    r83 = r83 * r35;
    r111 = fma(r118, r83, r111);
    r80 = r4 * r34;
    r80 = r80 * r130;
    r80 = r80 * r39;
    r80 = r80 * r98;
    r80 = r80 * r103;
    r80 = r80 * r53;
    r111 = fma(r47, r80, r111);
    r138 = r38 * r85;
    r111 = fma(r72, r138, r111);
    r121 = r21 * r73;
    r121 = r121 * r39;
    r121 = r121 * r70;
    r111 = fma(r53, r121, r111);
    r90 = r38 * r85;
    r111 = fma(r71, r90, r111);
    r117 = r74 * r130;
    r117 = r117 * r44;
    r117 = r117 * r69;
    r117 = r117 * r98;
    r111 = fma(r53, r117, r111);
    r99 = r4 * r73;
    r111 = fma(r123, r99, r111);
    r84 = r4 * r88;
    r84 = r84 * r54;
    r111 = fma(r104, r84, r111);
    r26 = r64 * r74;
    r26 = r26 * r130;
    r26 = r26 * r44;
    r26 = r26 * r69;
    r26 = r26 * r98;
    r111 = fma(r53, r26, r111);
    r111 = fma(r85, r100, r111);
    r26 = r2 * r111;
    r84 = r28 * r28;
    r84 = r84 * r38;
    r84 = r84 * r38;
    r84 = r84 * r115;
    r84 = r84 * r73;
    r84 = r84 * r44;
    r99 = r20 * r130;
    r99 = r99 * r97;
    r99 = fma(r91, r99, r93 * r84);
    r84 = r38 * r112;
    r84 = r84 * r88;
    r84 = r84 * r42;
    r99 = fma(r47, r84, r99);
    r117 = r130 * r110;
    r99 = fma(r95, r117, r99);
    r99 = r99 + r41;
    r99 = fma(r4, r99, r57 * r8);
    r8 = r21 * r28;
    r8 = r8 * r38;
    r8 = r8 * r64;
    r8 = r8 * r73;
    r8 = r8 * r39;
    r99 = fma(r70, r8, r99);
    r41 = r21 * r28;
    r41 = r41 * r38;
    r41 = r41 * r73;
    r41 = r41 * r39;
    r99 = fma(r70, r41, r99);
    r117 = r38 * r88;
    r99 = fma(r72, r117, r99);
    r84 = r5 * r85;
    r99 = fma(r105, r84, r99);
    r90 = r28 * r38;
    r90 = r90 * r64;
    r90 = r90 * r74;
    r90 = r90 * r130;
    r90 = r90 * r44;
    r90 = r90 * r69;
    r99 = fma(r98, r90, r99);
    r121 = r5 * r130;
    r99 = fma(r126, r121, r99);
    r138 = r75 * r130;
    r138 = r138 * r71;
    r99 = fma(r91, r138, r99);
    r80 = r38 * r88;
    r99 = fma(r71, r80, r99);
    r83 = r5 * r34;
    r83 = r83 * r130;
    r83 = r83 * r39;
    r83 = r83 * r98;
    r83 = r83 * r103;
    r83 = r83 * r53;
    r99 = fma(r47, r83, r99);
    r9 = r25 * r71;
    r99 = fma(r47, r9, r99);
    r102 = r28 * r38;
    r102 = r102 * r74;
    r102 = r102 * r130;
    r102 = r102 * r44;
    r102 = r102 * r69;
    r99 = fma(r98, r102, r99);
    r23 = r5 * r88;
    r23 = r23 * r54;
    r99 = fma(r104, r23, r99);
    r99 = fma(r130, r120, r99);
    r99 = fma(r73, r124, r99);
    r23 = r3 * r99;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r26, r23);
    r23 = r36 * r38;
    r23 = r23 * r112;
    r26 = r10 * r32;
    r26 = r26 * r28;
    r26 = fma(r50, r92, r39 * r26);
    r102 = r50 * r43;
    r102 = r102 * r43;
    r26 = fma(r94, r102, r26);
    r9 = r10 * r36;
    r9 = r9 * r43;
    r26 = fma(r39, r9, r26);
    r23 = fma(r26, r109, r54 * r23);
    r9 = r20 * r26;
    r23 = fma(r77, r9, r23);
    r102 = r50 * r43;
    r102 = r102 * r38;
    r102 = r102 * r115;
    r102 = r102 * r44;
    r102 = r102 * r93;
    r23 = fma(r53, r102, r23);
    r83 = r21 * r28;
    r83 = r83 * r28;
    r83 = r83 * r38;
    r83 = r83 * r38;
    r83 = r83 * r26;
    r83 = r83 * r39;
    r83 = r83 * r98;
    r83 = fma(r103, r83, r32 * r105);
    r80 = r26 * r97;
    r83 = fma(r91, r80, r83);
    r83 = fma(r50, r96, r83);
    r23 = r23 + r83;
    r102 = r36 * r54;
    r9 = r21 * r43;
    r9 = r9 * r38;
    r9 = r9 * r26;
    r9 = r9 * r39;
    r9 = r9 * r98;
    r9 = r9 * r103;
    r9 = fma(r53, r9, r104 * r102);
    r102 = r50 * r43;
    r102 = r102 * r38;
    r102 = r102 * r44;
    r102 = r102 * r53;
    r9 = fma(r94, r102, r9);
    r9 = fma(r26, r77, r9);
    r83 = r83 + r9;
    r23 = fma(r56, r83, r5 * r23);
    r102 = r36 * r38;
    r23 = fma(r72, r102, r23);
    r80 = r43 * r26;
    r80 = r80 * r35;
    r23 = fma(r118, r80, r23);
    r138 = r26 * r126;
    r121 = r43 * r75;
    r121 = r121 * r26;
    r121 = r121 * r71;
    r23 = fma(r35, r121, r23);
    r90 = r64 * r74;
    r90 = r90 * r26;
    r90 = r90 * r44;
    r90 = r90 * r69;
    r90 = r90 * r98;
    r23 = fma(r53, r90, r23);
    r84 = r74 * r26;
    r84 = r84 * r44;
    r84 = r84 * r69;
    r84 = r84 * r98;
    r23 = fma(r53, r84, r23);
    r117 = r4 * r34;
    r117 = r117 * r26;
    r117 = r117 * r39;
    r117 = r117 * r98;
    r117 = r117 * r103;
    r117 = r117 * r53;
    r23 = fma(r47, r117, r23);
    r41 = r4 * r50;
    r23 = fma(r123, r41, r23);
    r8 = r62 * r10;
    r8 = r8 * r7;
    r8 = fma(r61, r83, r83 * r8);
    r8 = fma(r83, r66, r8);
    r8 = fma(r83, r65, r8);
    r81 = r8 * r53;
    r23 = fma(r71, r81, r23);
    r22 = r21 * r50;
    r22 = r22 * r64;
    r22 = r22 * r39;
    r22 = r22 * r70;
    r23 = fma(r53, r22, r23);
    r76 = r21 * r50;
    r76 = r76 * r39;
    r76 = r76 * r70;
    r23 = fma(r53, r76, r23);
    r125 = r4 * r32;
    r125 = r125 * r54;
    r23 = fma(r104, r125, r23);
    r106 = r36 * r38;
    r23 = fma(r71, r106, r23);
    r23 = fma(r4, r138, r23);
    r23 = fma(r36, r100, r23);
    r106 = r2 * r23;
    r125 = r32 * r38;
    r125 = r125 * r112;
    r125 = r125 * r42;
    r76 = r26 * r110;
    r76 = fma(r95, r76, r47 * r125);
    r125 = r50 * r28;
    r125 = r125 * r28;
    r125 = r125 * r38;
    r125 = r125 * r38;
    r125 = r125 * r115;
    r125 = r125 * r44;
    r76 = fma(r93, r125, r76);
    r22 = r20 * r26;
    r22 = r22 * r97;
    r76 = fma(r91, r22, r76);
    r76 = r76 + r9;
    r83 = fma(r57, r83, r4 * r76);
    r76 = r75 * r26;
    r76 = r76 * r71;
    r83 = fma(r91, r76, r83);
    r9 = r32 * r38;
    r83 = fma(r71, r9, r83);
    r22 = r28 * r38;
    r22 = r22 * r64;
    r22 = r22 * r74;
    r22 = r22 * r26;
    r22 = r22 * r44;
    r22 = r22 * r69;
    r83 = fma(r98, r22, r83);
    r125 = r21 * r50;
    r125 = r125 * r28;
    r125 = r125 * r38;
    r125 = r125 * r64;
    r125 = r125 * r39;
    r83 = fma(r70, r125, r83);
    r81 = r21 * r50;
    r81 = r81 * r28;
    r81 = r81 * r38;
    r81 = r81 * r39;
    r83 = fma(r70, r81, r83);
    r41 = r5 * r34;
    r41 = r41 * r26;
    r41 = r41 * r39;
    r41 = r41 * r98;
    r41 = r41 * r103;
    r41 = r41 * r53;
    r83 = fma(r47, r41, r83);
    r117 = r8 * r71;
    r83 = fma(r47, r117, r83);
    r84 = r32 * r38;
    r83 = fma(r72, r84, r83);
    r90 = r5 * r32;
    r90 = r90 * r54;
    r83 = fma(r104, r90, r83);
    r121 = r28 * r38;
    r121 = r121 * r74;
    r121 = r121 * r26;
    r121 = r121 * r44;
    r121 = r121 * r69;
    r83 = fma(r98, r121, r83);
    r80 = r5 * r36;
    r83 = fma(r105, r80, r83);
    r83 = fma(r5, r138, r83);
    r83 = fma(r50, r124, r83);
    r83 = fma(r26, r120, r83);
    r80 = r3 * r83;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r106, r80);
    r80 = r10 * r37;
    r80 = r80 * r28;
    r80 = fma(r39, r80, r33 * r92);
    r106 = r33 * r43;
    r106 = r106 * r43;
    r80 = fma(r94, r106, r80);
    r121 = r10 * r49;
    r121 = r121 * r43;
    r80 = fma(r39, r121, r80);
    r121 = r20 * r80;
    r106 = r33 * r43;
    r106 = r106 * r38;
    r106 = r106 * r115;
    r106 = r106 * r44;
    r106 = r106 * r93;
    r106 = fma(r53, r106, r77 * r121);
    r121 = r49 * r38;
    r121 = r121 * r112;
    r106 = fma(r54, r121, r106);
    r90 = fma(r37, r105, r33 * r96);
    r84 = r80 * r97;
    r90 = fma(r91, r84, r90);
    r117 = r21 * r28;
    r117 = r117 * r28;
    r117 = r117 * r38;
    r117 = r117 * r38;
    r117 = r117 * r80;
    r117 = r117 * r39;
    r117 = r117 * r98;
    r90 = fma(r103, r117, r90);
    r106 = fma(r80, r109, r106);
    r106 = r106 + r90;
    r121 = r33 * r43;
    r121 = r121 * r38;
    r121 = r121 * r44;
    r121 = r121 * r53;
    r121 = fma(r94, r121, r80 * r77);
    r77 = r21 * r43;
    r77 = r77 * r38;
    r77 = r77 * r80;
    r77 = r77 * r39;
    r77 = r77 * r98;
    r77 = r77 * r103;
    r121 = fma(r53, r77, r121);
    r117 = r49 * r54;
    r121 = fma(r104, r117, r121);
    r90 = r90 + r121;
    r106 = fma(r56, r90, r5 * r106);
    r117 = r4 * r37;
    r117 = r117 * r54;
    r106 = fma(r104, r117, r106);
    r77 = r4 * r80;
    r106 = fma(r126, r77, r106);
    r84 = r43 * r80;
    r84 = r84 * r35;
    r106 = fma(r118, r84, r106);
    r41 = r49 * r38;
    r106 = fma(r71, r41, r106);
    r81 = r21 * r33;
    r81 = r81 * r39;
    r81 = r81 * r70;
    r106 = fma(r53, r81, r106);
    r138 = r4 * r33;
    r106 = fma(r123, r138, r106);
    r125 = r4 * r34;
    r125 = r125 * r80;
    r125 = r125 * r39;
    r125 = r125 * r98;
    r125 = r125 * r103;
    r125 = r125 * r53;
    r106 = fma(r47, r125, r106);
    r22 = r43 * r75;
    r22 = r22 * r80;
    r22 = r22 * r71;
    r106 = fma(r35, r22, r106);
    r9 = r49 * r38;
    r106 = fma(r72, r9, r106);
    r76 = r74 * r80;
    r76 = r76 * r44;
    r76 = r76 * r69;
    r76 = r76 * r98;
    r106 = fma(r53, r76, r106);
    r102 = r62 * r10;
    r102 = r102 * r7;
    r102 = fma(r61, r90, r90 * r102);
    r102 = fma(r90, r65, r102);
    r102 = fma(r90, r66, r102);
    r108 = r102 * r53;
    r106 = fma(r71, r108, r106);
    r132 = r64 * r74;
    r132 = r132 * r80;
    r132 = r132 * r44;
    r132 = r132 * r69;
    r132 = r132 * r98;
    r106 = fma(r53, r132, r106);
    r133 = r21 * r33;
    r133 = r133 * r64;
    r133 = r133 * r39;
    r133 = r133 * r70;
    r106 = fma(r53, r133, r106);
    r106 = fma(r49, r100, r106);
    r133 = r2 * r106;
    r132 = r33 * r28;
    r132 = r132 * r28;
    r132 = r132 * r38;
    r132 = r132 * r38;
    r132 = r132 * r115;
    r132 = r132 * r44;
    r108 = r37 * r38;
    r108 = r108 * r112;
    r108 = r108 * r42;
    r108 = fma(r47, r108, r93 * r132);
    r132 = r20 * r80;
    r132 = r132 * r97;
    r108 = fma(r91, r132, r108);
    r76 = r80 * r110;
    r108 = fma(r95, r76, r108);
    r108 = r108 + r121;
    r108 = fma(r4, r108, r57 * r90);
    r90 = r21 * r33;
    r90 = r90 * r28;
    r90 = r90 * r38;
    r90 = r90 * r39;
    r108 = fma(r70, r90, r108);
    r121 = r37 * r38;
    r108 = fma(r72, r121, r108);
    r76 = r5 * r37;
    r76 = r76 * r54;
    r108 = fma(r104, r76, r108);
    r132 = r5 * r80;
    r108 = fma(r126, r132, r108);
    r9 = r75 * r80;
    r9 = r9 * r71;
    r108 = fma(r91, r9, r108);
    r22 = r102 * r71;
    r108 = fma(r47, r22, r108);
    r125 = r28 * r38;
    r125 = r125 * r64;
    r125 = r125 * r74;
    r125 = r125 * r80;
    r125 = r125 * r44;
    r125 = r125 * r69;
    r108 = fma(r98, r125, r108);
    r138 = r5 * r34;
    r138 = r138 * r80;
    r138 = r138 * r39;
    r138 = r138 * r98;
    r138 = r138 * r103;
    r138 = r138 * r53;
    r108 = fma(r47, r138, r108);
    r81 = r28 * r38;
    r81 = r81 * r74;
    r81 = r81 * r80;
    r81 = r81 * r44;
    r81 = r81 * r69;
    r108 = fma(r98, r81, r108);
    r41 = r37 * r38;
    r108 = fma(r71, r41, r108);
    r84 = r21 * r33;
    r84 = r84 * r28;
    r84 = r84 * r38;
    r84 = r84 * r64;
    r84 = r84 * r39;
    r108 = fma(r70, r84, r108);
    r77 = r5 * r49;
    r108 = fma(r105, r77, r108);
    r108 = fma(r33, r124, r108);
    r108 = fma(r80, r120, r108);
    r77 = r3 * r108;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r133, r77);
    r77 = r10 * r40;
    r77 = r77 * r28;
    r92 = fma(r31, r92, r39 * r77);
    r77 = r10 * r48;
    r77 = r77 * r43;
    r92 = fma(r39, r77, r92);
    r133 = r31 * r43;
    r133 = r133 * r43;
    r92 = fma(r94, r133, r92);
    r133 = r43 * r43;
    r133 = r39 * r133;
    r133 = r133 * r44;
    r133 = r133 * r38;
    r133 = r133 * r98;
    r133 = r133 * r30;
    r133 = r133 * r92;
    r109 = fma(r20, r133, r92 * r109);
    r30 = r48 * r38;
    r30 = r30 * r112;
    r109 = fma(r54, r30, r109);
    r77 = r31 * r43;
    r77 = r77 * r38;
    r77 = r77 * r115;
    r77 = r77 * r44;
    r77 = r77 * r93;
    r109 = fma(r53, r77, r109);
    r96 = fma(r40, r105, r31 * r96);
    r84 = r92 * r97;
    r96 = fma(r91, r84, r96);
    r41 = r21 * r28;
    r41 = r41 * r28;
    r41 = r41 * r38;
    r41 = r41 * r38;
    r41 = r41 * r92;
    r41 = r41 * r39;
    r41 = r41 * r98;
    r96 = fma(r103, r41, r96);
    r109 = r109 + r96;
    r77 = r21 * r43;
    r77 = r77 * r38;
    r77 = r77 * r92;
    r77 = r77 * r39;
    r77 = r77 * r98;
    r77 = r77 * r103;
    r77 = fma(r53, r77, r133);
    r133 = r48 * r54;
    r77 = fma(r104, r133, r77);
    r30 = r31 * r43;
    r30 = r30 * r38;
    r30 = r30 * r44;
    r30 = r30 * r53;
    r77 = fma(r94, r30, r77);
    r96 = r96 + r77;
    r56 = fma(r56, r96, r5 * r109);
    r109 = r48 * r38;
    r56 = fma(r72, r109, r56);
    r30 = r4 * r31;
    r56 = fma(r123, r30, r56);
    r123 = r4 * r40;
    r123 = r123 * r54;
    r56 = fma(r104, r123, r56);
    r133 = r43 * r92;
    r133 = r133 * r35;
    r56 = fma(r118, r133, r56);
    r118 = r43 * r75;
    r118 = r118 * r92;
    r118 = r118 * r71;
    r56 = fma(r35, r118, r56);
    r35 = r62 * r10;
    r35 = r35 * r7;
    r35 = fma(r96, r35, r61 * r96);
    r35 = fma(r96, r66, r35);
    r35 = fma(r96, r65, r35);
    r65 = r35 * r53;
    r56 = fma(r71, r65, r56);
    r66 = r4 * r92;
    r56 = fma(r126, r66, r56);
    r61 = r21 * r31;
    r61 = r61 * r39;
    r61 = r61 * r70;
    r56 = fma(r53, r61, r56);
    r94 = r64 * r74;
    r94 = r94 * r92;
    r94 = r94 * r44;
    r94 = r94 * r69;
    r94 = r94 * r98;
    r56 = fma(r53, r94, r56);
    r41 = r4 * r34;
    r41 = r41 * r92;
    r41 = r41 * r39;
    r41 = r41 * r98;
    r41 = r41 * r103;
    r41 = r41 * r53;
    r56 = fma(r47, r41, r56);
    r84 = r48 * r38;
    r56 = fma(r71, r84, r56);
    r81 = r21 * r31;
    r81 = r81 * r64;
    r81 = r81 * r39;
    r81 = r81 * r70;
    r56 = fma(r53, r81, r56);
    r138 = r74 * r92;
    r138 = r138 * r44;
    r138 = r138 * r69;
    r138 = r138 * r98;
    r56 = fma(r53, r138, r56);
    r56 = fma(r48, r100, r56);
    r138 = r2 * r56;
    r81 = r31 * r28;
    r81 = r81 * r28;
    r81 = r81 * r38;
    r81 = r81 * r38;
    r81 = r81 * r115;
    r81 = r81 * r44;
    r115 = r40 * r38;
    r115 = r115 * r112;
    r115 = r115 * r42;
    r115 = fma(r47, r115, r93 * r81);
    r81 = r20 * r92;
    r81 = r81 * r97;
    r115 = fma(r91, r81, r115);
    r42 = r92 * r110;
    r115 = fma(r95, r42, r115);
    r115 = r115 + r77;
    r115 = fma(r4, r115, r57 * r96);
    r96 = r28 * r38;
    r96 = r96 * r74;
    r96 = r96 * r92;
    r96 = r96 * r44;
    r96 = r96 * r69;
    r115 = fma(r98, r96, r115);
    r57 = r21 * r31;
    r57 = r57 * r28;
    r57 = r57 * r38;
    r57 = r57 * r39;
    r115 = fma(r70, r57, r115);
    r77 = r40 * r38;
    r115 = fma(r71, r77, r115);
    r42 = r5 * r48;
    r115 = fma(r105, r42, r115);
    r105 = r21 * r31;
    r105 = r105 * r28;
    r105 = r105 * r38;
    r105 = r105 * r64;
    r105 = r105 * r39;
    r115 = fma(r70, r105, r115);
    r70 = r40 * r38;
    r115 = fma(r72, r70, r115);
    r72 = r5 * r40;
    r72 = r72 * r54;
    r115 = fma(r104, r72, r115);
    r81 = r28 * r38;
    r81 = r81 * r64;
    r81 = r81 * r74;
    r81 = r81 * r92;
    r81 = r81 * r44;
    r81 = r81 * r69;
    r115 = fma(r98, r81, r115);
    r69 = r5 * r92;
    r115 = fma(r126, r69, r115);
    r126 = r75 * r92;
    r126 = r126 * r71;
    r115 = fma(r91, r126, r115);
    r91 = r35 * r71;
    r115 = fma(r47, r91, r115);
    r44 = r5 * r34;
    r44 = r44 * r92;
    r44 = r44 * r39;
    r44 = r44 * r98;
    r44 = r44 * r103;
    r44 = r44 * r53;
    r115 = fma(r47, r44, r115);
    r115 = fma(r92, r120, r115);
    r115 = fma(r31, r124, r115);
    r44 = r3 * r115;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             10 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r138,
                                             r44);
    r44 = r3 * r21;
    r44 = r44 * r1;
    r138 = r21 * r0;
    r91 = r2 * r138;
    r44 = fma(r107, r91, r129 * r44);
    r126 = r3 * r21;
    r126 = r126 * r1;
    r126 = fma(r113, r91, r114 * r126);
    WriteSum2<double, double>((double*)inout_shared, r44, r126);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r3 * r21;
    r126 = r126 * r1;
    r126 = fma(r111, r91, r99 * r126);
    r44 = r3 * r21;
    r44 = r44 * r1;
    r44 = fma(r23, r91, r83 * r44);
    WriteSum2<double, double>((double*)inout_shared, r126, r44);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = r3 * r21;
    r44 = r44 * r1;
    r44 = fma(r106, r91, r108 * r44);
    r126 = r3 * r21;
    r126 = r126 * r1;
    r126 = fma(r56, r91, r115 * r126);
    WriteSum2<double, double>((double*)inout_shared, r44, r126);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r107 * r107;
    r44 = r2 * r2;
    r69 = r129 * r129;
    r81 = r3 * r3;
    r69 = fma(r81, r69, r44 * r126);
    r126 = r113 * r113;
    r72 = r114 * r114;
    r72 = fma(r81, r72, r44 * r126);
    WriteSum2<double, double>((double*)inout_shared, r69, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r111 * r111;
    r69 = r99 * r99;
    r69 = fma(r81, r69, r44 * r72);
    r72 = r83 * r83;
    r126 = r23 * r23;
    r126 = fma(r44, r126, r81 * r72);
    WriteSum2<double, double>((double*)inout_shared, r69, r126);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r106 * r106;
    r69 = r108 * r108;
    r69 = fma(r81, r69, r44 * r126);
    r126 = r115 * r115;
    r72 = r56 * r56;
    r72 = fma(r44, r72, r81 * r126);
    WriteSum2<double, double>((double*)inout_shared, r69, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r129 * r114;
    r69 = r107 * r113;
    r69 = fma(r44, r69, r81 * r72);
    r72 = r107 * r111;
    r126 = r129 * r99;
    r126 = fma(r81, r126, r44 * r72);
    WriteSum2<double, double>((double*)inout_shared, r69, r126);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r107 * r23;
    r69 = r129 * r83;
    r69 = fma(r81, r69, r44 * r126);
    r126 = r107 * r106;
    r72 = r129 * r108;
    r72 = fma(r81, r72, r44 * r126);
    WriteSum2<double, double>((double*)inout_shared, r69, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r129 * r115;
    r69 = r107 * r56;
    r69 = fma(r44, r69, r81 * r72);
    r72 = r114 * r99;
    r126 = r113 * r111;
    r126 = fma(r44, r126, r81 * r72);
    WriteSum2<double, double>((double*)inout_shared, r69, r126);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r113 * r23;
    r69 = r114 * r83;
    r69 = fma(r81, r69, r44 * r126);
    r126 = r113 * r106;
    r72 = r114 * r108;
    r72 = fma(r81, r72, r44 * r126);
    WriteSum2<double, double>((double*)inout_shared, r69, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r114 * r115;
    r69 = r113 * r56;
    r69 = fma(r44, r69, r81 * r72);
    r72 = r99 * r83;
    r126 = r111 * r23;
    r126 = fma(r44, r126, r81 * r72);
    WriteSum2<double, double>((double*)inout_shared, r69, r126);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r111 * r106;
    r69 = r99 * r108;
    r69 = fma(r81, r69, r44 * r126);
    r126 = r99 * r115;
    r72 = r111 * r56;
    r72 = fma(r44, r72, r81 * r126);
    WriteSum2<double, double>((double*)inout_shared, r69, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r23 * r106;
    r69 = r83 * r108;
    r69 = fma(r81, r69, r44 * r72);
    r72 = r83 * r115;
    r126 = r23 * r56;
    r126 = fma(r44, r126, r81 * r72);
    WriteSum2<double, double>((double*)inout_shared, r69, r126);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r108 * r115;
    r69 = r106 * r56;
    r69 = fma(r44, r69, r81 * r126);
    WriteSum1<double, double>((double*)inout_shared, r69);
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
        r59,
        r58);
    r69 = r2 * r7;
    r69 = r69 * r53;
    r69 = r69 * r71;
    r126 = r3 * r7;
    r126 = r126 * r71;
    r126 = r126 * r47;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        2 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r69,
        r126);
    r126 = r2 * r53;
    r126 = r126 * r71;
    r126 = r126 * r63;
    r69 = r3 * r71;
    r69 = r69 * r47;
    r69 = r69 * r63;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        4 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r126,
        r69);
    r69 = r3 * r60;
    r126 = r2 * r10;
    r126 = r126 * r47;
    r126 = r126 * r54;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        6 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r126,
        r69);
    r69 = r2 * r52;
    r126 = r3 * r10;
    r126 = r126 * r47;
    r126 = r126 * r54;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        8 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r69,
        r126);
    r126 = r2 * r53;
    r126 = r126 * r71;
    r126 = r126 * r68;
    r69 = r3 * r71;
    r69 = r69 * r47;
    r69 = r69 * r68;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        10 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r126,
        r69);
    r69 = r2 * r53;
    r69 = r69 * r71;
    r69 = r69 * r67;
    r126 = r3 * r71;
    r126 = r126 * r47;
    r126 = r126 * r67;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        12 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r69,
        r126);
    r126 = r2 * r7;
    r69 = r3 * r7;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        14 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r126,
        r69);
    r69 = r21 * r58;
    r69 = r69 * r1;
    r138 = r59 * r138;
    WriteSum2<double, double>((double*)inout_shared, r138, r69);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            0 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = r3 * r21;
    r69 = r69 * r7;
    r69 = r69 * r1;
    r69 = r69 * r71;
    r138 = r7 * r53;
    r138 = r138 * r71;
    r138 = fma(r91, r138, r47 * r69);
    r69 = r3 * r21;
    r69 = r69 * r1;
    r69 = r69 * r71;
    r69 = r69 * r47;
    r126 = r53 * r71;
    r126 = r126 * r63;
    r126 = fma(r91, r126, r63 * r69);
    WriteSum2<double, double>((double*)inout_shared, r138, r126);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            2 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r3 * r21;
    r126 = r126 * r60;
    r138 = r2 * r34;
    r138 = r138 * r0;
    r138 = r138 * r47;
    r138 = fma(r54, r138, r1 * r126);
    r126 = r3 * r34;
    r126 = r126 * r1;
    r126 = r126 * r47;
    r126 = fma(r54, r126, r52 * r91);
    WriteSum2<double, double>((double*)inout_shared, r138, r126);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            4 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r3 * r21;
    r126 = r126 * r1;
    r126 = r126 * r71;
    r126 = r126 * r47;
    r138 = r53 * r71;
    r138 = r138 * r68;
    r138 = fma(r91, r138, r68 * r126);
    r126 = r3 * r21;
    r126 = r126 * r1;
    r126 = r126 * r71;
    r126 = r126 * r47;
    r0 = r53 * r71;
    r0 = r0 * r67;
    r0 = fma(r91, r0, r67 * r126);
    WriteSum2<double, double>((double*)inout_shared, r138, r0);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            6 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r3 * r21;
    r0 = r0 * r7;
    r0 = r0 * r1;
    r91 = r7 * r91;
    WriteSum2<double, double>((double*)inout_shared, r91, r0);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            8 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r59 * r59;
    r91 = r58 * r58;
    WriteSum2<double, double>((double*)inout_shared, r0, r91);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            0 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r91 = r38 * r63;
    r91 = r91 * r44;
    r0 = r28 * r38;
    r0 = r0 * r63;
    r0 = r0 * r81;
    r0 = fma(r97, r0, r55 * r91);
    r91 = r38 * r44;
    r91 = r91 * r67;
    r1 = r28 * r38;
    r1 = r1 * r81;
    r1 = r1 * r97;
    r1 = fma(r67, r1, r55 * r91);
    WriteSum2<double, double>((double*)inout_shared, r0, r1);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            2 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r53 * r44;
    r91 = r43 * r28;
    r51 = r45 * r51;
    r51 = 1.0 / r51;
    r6 = r46 * r6;
    r6 = 1.0 / r6;
    r91 = r91 * r38;
    r91 = r91 * r38;
    r91 = r91 * r128;
    r91 = r91 * r51;
    r91 = r91 * r6;
    r91 = r91 * r47;
    r6 = r60 * r81;
    r51 = fma(r60, r6, r0 * r91);
    r128 = r53 * r81;
    r46 = r52 * r52;
    r46 = fma(r44, r46, r91 * r128);
    WriteSum2<double, double>((double*)inout_shared, r51, r46);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            4 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r68 * r68;
    r51 = r38 * r44;
    r51 = r51 * r55;
    r128 = r28 * r38;
    r128 = r128 * r81;
    r128 = r128 * r97;
    r128 = fma(r46, r128, r46 * r51);
    r91 = r67 * r67;
    r45 = r81 * r97;
    r45 = r45 * r47;
    r91 = fma(r91, r45, r51 * r91);
    WriteSum2<double, double>((double*)inout_shared, r128, r91);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            6 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r91 = r63 * r44;
    r138 = r63 * r81;
    WriteSum2<double, double>((double*)inout_shared, r91, r138);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            8 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r138 = 0.00000000000000000e+00;
    r91 = r2 * r7;
    r91 = r91 * r59;
    r91 = r91 * r53;
    r91 = r91 * r71;
    WriteSum2<double, double>((double*)inout_shared, r138, r91);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            0 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r91 = r2 * r59;
    r91 = r91 * r53;
    r91 = r91 * r71;
    r91 = r91 * r63;
    r126 = r2 * r10;
    r126 = r126 * r59;
    r126 = r126 * r47;
    r126 = r126 * r54;
    WriteSum2<double, double>((double*)inout_shared, r91, r126);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            2 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r2 * r52;
    r126 = r126 * r59;
    r91 = r2 * r59;
    r91 = r91 * r53;
    r91 = r91 * r71;
    r91 = r91 * r68;
    WriteSum2<double, double>((double*)inout_shared, r126, r91);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            4 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r91 = r2 * r7;
    r91 = r91 * r59;
    r59 = r2 * r59;
    r59 = r59 * r53;
    r59 = r59 * r71;
    r59 = r59 * r67;
    WriteSum2<double, double>((double*)inout_shared, r59, r91);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            6 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r91 = r3 * r7;
    r91 = r91 * r58;
    r91 = r91 * r71;
    r91 = r91 * r47;
    WriteSum2<double, double>((double*)inout_shared, r138, r91);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            8 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = r3 * r60;
    r60 = r60 * r58;
    r91 = r3 * r58;
    r91 = r91 * r71;
    r91 = r91 * r47;
    r91 = r91 * r63;
    WriteSum2<double, double>((double*)inout_shared, r91, r60);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            10 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = r3 * r10;
    r60 = r60 * r58;
    r60 = r60 * r47;
    r60 = r60 * r54;
    r91 = r3 * r58;
    r91 = r91 * r71;
    r91 = r91 * r47;
    r91 = r91 * r68;
    WriteSum2<double, double>((double*)inout_shared, r60, r91);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            12 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r91 = r3 * r58;
    r91 = r91 * r71;
    r91 = r91 * r47;
    r91 = r91 * r67;
    WriteSum2<double, double>((double*)inout_shared, r91, r138);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            14 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r138 = r3 * r7;
    r138 = r138 * r58;
    r58 = r38 * r68;
    r58 = r58 * r44;
    r91 = r28 * r38;
    r91 = r91 * r68;
    r91 = r91 * r81;
    r91 = fma(r97, r91, r55 * r58);
    WriteSum2<double, double>((double*)inout_shared, r138, r91);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            16 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r91 = r47 * r6;
    r138 = r71 * r91;
    r58 = r43 * r7;
    r58 = r58 * r93;
    r58 = r58 * r103;
    r58 = r58 * r47;
    r58 = r58 * r104;
    r58 = fma(r0, r58, r7 * r138);
    r60 = r71 * r0;
    r59 = r52 * r60;
    r126 = r28 * r7;
    r126 = r126 * r93;
    r126 = r126 * r103;
    r126 = r126 * r53;
    r126 = r126 * r47;
    r126 = r126 * r81;
    r126 = fma(r104, r126, r7 * r59);
    WriteSum2<double, double>((double*)inout_shared, r58, r126);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            18 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r38 * r44;
    r58 = r7 * r67;
    r126 = r126 * r55;
    r55 = r28 * r38;
    r55 = r55 * r81;
    r55 = r55 * r97;
    r55 = fma(r58, r55, r58 * r126);
    WriteSum2<double, double>((double*)inout_shared, r1, r55);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            20 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r63 * r60;
    r126 = r71 * r47;
    r126 = r126 * r63;
    r126 = r126 * r81;
    WriteSum2<double, double>((double*)inout_shared, r1, r126);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            22 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r43 * r93;
    r126 = r126 * r103;
    r126 = r126 * r47;
    r126 = r126 * r63;
    r126 = r126 * r104;
    r126 = fma(r0, r126, r63 * r138);
    r1 = r28 * r93;
    r1 = r1 * r103;
    r1 = r1 * r53;
    r1 = r1 * r47;
    r1 = r1 * r63;
    r1 = r1 * r81;
    r1 = fma(r104, r1, r63 * r59);
    WriteSum2<double, double>((double*)inout_shared, r126, r1);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            24 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r55, r128);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            26 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r128 = r68 * r60;
    r55 = r71 * r47;
    r55 = r55 * r68;
    r55 = r55 * r81;
    WriteSum2<double, double>((double*)inout_shared, r128, r55);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            28 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = r10 * r52;
    r55 = r55 * r47;
    r55 = r55 * r54;
    r128 = r10 * r54;
    r128 = fma(r91, r128, r44 * r55);
    r55 = r43 * r93;
    r55 = r55 * r103;
    r55 = r55 * r47;
    r55 = r55 * r68;
    r55 = r55 * r104;
    r55 = fma(r0, r55, r68 * r138);
    WriteSum2<double, double>((double*)inout_shared, r128, r55);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            30 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = r10 * r7;
    r55 = r55 * r47;
    r55 = r55 * r54;
    r55 = r55 * r44;
    r128 = r43 * r93;
    r128 = r128 * r103;
    r128 = r128 * r47;
    r128 = r128 * r104;
    r128 = r128 * r67;
    r128 = fma(r0, r128, r67 * r138);
    WriteSum2<double, double>((double*)inout_shared, r128, r55);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            32 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r7 * r6;
    r55 = r28 * r93;
    r55 = r55 * r103;
    r55 = r55 * r53;
    r55 = r55 * r47;
    r55 = r55 * r68;
    r55 = r55 * r81;
    r55 = fma(r104, r55, r68 * r59);
    WriteSum2<double, double>((double*)inout_shared, r6, r55);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            34 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = r52 * r7;
    r55 = r55 * r44;
    r6 = r28 * r93;
    r6 = r6 * r103;
    r6 = r6 * r53;
    r6 = r6 * r47;
    r6 = r6 * r81;
    r6 = r6 * r104;
    r6 = fma(r67, r6, r67 * r59);
    WriteSum2<double, double>((double*)inout_shared, r6, r55);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            36 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = r10 * r7;
    r55 = r55 * r47;
    r55 = r55 * r54;
    r55 = r55 * r81;
    r46 = r7 * r46;
    r45 = fma(r46, r45, r46 * r51);
    WriteSum2<double, double>((double*)inout_shared, r55, r45);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            38 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = r67 * r60;
    r55 = r71 * r47;
    r55 = r55 * r81;
    r55 = r55 * r67;
    WriteSum2<double, double>((double*)inout_shared, r45, r55);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            40 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = r58 * r60;
    r47 = r71 * r47;
    r47 = r47 * r81;
    r47 = r47 * r58;
    WriteSum2<double, double>((double*)inout_shared, r60, r47);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            42 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
}

void ThinPrismFisheyeSplitFixedPrincipalPointFixedPointResJac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
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
    double* out_focal_and_extra_jac,
    unsigned int out_focal_and_extra_jac_num_alloc,
    double* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    double* const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    double* const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeSplitFixedPrincipalPointFixedPointResJacKernel<<<n_blocks,
                                                                   1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
      focal_and_extra_indices,
      pixel,
      pixel_num_alloc,
      principal_point,
      principal_point_num_alloc,
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
      out_focal_and_extra_jac,
      out_focal_and_extra_jac_num_alloc,
      out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc,
      out_focal_and_extra_precond_diag,
      out_focal_and_extra_precond_diag_num_alloc,
      out_focal_and_extra_precond_tril,
      out_focal_and_extra_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar