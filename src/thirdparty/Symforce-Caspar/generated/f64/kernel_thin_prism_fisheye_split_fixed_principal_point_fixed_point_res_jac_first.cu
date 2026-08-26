#include "kernel_thin_prism_fisheye_split_fixed_principal_point_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPrincipalPointFixedPointResJacFirstKernel(
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
    r73 = fma(r1, r1, r0 * r0);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r73);
  if (global_thread_idx < problem_size) {
    r73 = r43 * r44;
    r74 = -5.00000000000000000e-01;
    r75 = rsqrt(r30);
    r76 = r10 * r43;
    r77 = r11 * r14;
    r78 = r16 * r17;
    r78 = fma(r74, r78, r74 * r77);
    r77 = r15 * r18;
    r78 = fma(r74, r77, r78);
    r79 = r12 * r13;
    r80 = 5.00000000000000000e-01;
    r78 = fma(r80, r79, r78);
    r79 = r24 * r78;
    r77 = r15 * r14;
    r81 = r12 * r17;
    r81 = fma(r80, r81, r80 * r77);
    r77 = r11 * r18;
    r81 = fma(r74, r77, r81);
    r82 = r16 * r13;
    r81 = fma(r80, r82, r81);
    r82 = r29 * r81;
    r77 = fma(r10, r82, r10 * r79);
    r83 = r10 * r19;
    r84 = fma(r80, r26, r74 * r22);
    r84 = fma(r74, r23, r84);
    r84 = fma(r74, r25, r84);
    r85 = r10 * r27;
    r86 = r16 * r14;
    r87 = r11 * r17;
    r87 = fma(r74, r87, r80 * r86);
    r86 = r12 * r18;
    r87 = fma(r74, r86, r87);
    r88 = r15 * r13;
    r87 = fma(r74, r88, r87);
    r85 = r85 * r87;
    r83 = fma(r84, r83, r85);
    r77 = r77 + r83;
    r88 = r10 * r24;
    r88 = r88 * r87;
    r86 = r10 * r19;
    r86 = r86 * r81;
    r89 = r88 + r86;
    r90 = r27 * r34;
    r89 = fma(r78, r90, r89);
    r91 = r34 * r29;
    r89 = fma(r84, r91, r89);
    r89 = fma(r9, r89, r41 * r77);
    r77 = r24 * r81;
    r91 = -4.00000000000000000e+00;
    r77 = r77 * r91;
    r90 = r27 * r84;
    r92 = r91 * r90;
    r93 = r77 + r92;
    r89 = fma(r8, r93, r89);
    r76 = r76 * r89;
    r93 = r43 * r43;
    r94 = r10 * r24;
    r94 = r94 * r84;
    r95 = r10 * r27;
    r95 = fma(r81, r95, r94);
    r81 = r10 * r19;
    r81 = r81 * r78;
    r96 = r10 * r29;
    r96 = r96 * r87;
    r97 = r81 + r96;
    r98 = r95 + r97;
    r82 = fma(r34, r82, r34 * r79);
    r82 = r82 + r83;
    r82 = fma(r8, r82, r9 * r98);
    r98 = r19 * r87;
    r98 = r98 * r91;
    r77 = r77 + r98;
    r82 = fma(r41, r77, r82);
    r51 = r45 * r51;
    r77 = 1.0 / r51;
    r99 = r34 * r77;
    r93 = r93 * r82;
    r93 = fma(r99, r93, r39 * r76);
    r76 = r28 * r28;
    r76 = r76 * r99;
    r100 = r10 * r28;
    r101 = r19 * r34;
    r102 = r34 * r29;
    r102 = r102 * r87;
    r101 = fma(r78, r101, r102);
    r101 = r101 + r95;
    r92 = r98 + r92;
    r92 = fma(r9, r92, r41 * r101);
    r101 = r10 * r29;
    r101 = fma(r84, r101, r86);
    r86 = r10 * r27;
    r86 = fma(r78, r86, r88);
    r101 = r101 + r86;
    r92 = fma(r8, r101, r92);
    r100 = r100 * r92;
    r93 = fma(r39, r100, r93);
    r93 = fma(r82, r76, r93);
    r73 = r73 * r38;
    r73 = r73 * r69;
    r73 = r73 * r74;
    r73 = r73 * r75;
    r73 = r73 * r93;
    r100 = r38 * r76;
    r101 = r38 * r44;
    r101 = r101 * r99;
    r101 = r100 * r101;
    r88 = r42 * r47;
    r98 = r93 * r88;
    r30 = r35 + r30;
    r30 = 1.0 / r30;
    r35 = r75 * r30;
    r95 = r28 * r35;
    r98 = fma(r95, r98, r82 * r101);
    r103 = r21 * r28;
    r6 = r46 * r6;
    r104 = 1.0 / r6;
    r103 = r103 * r28;
    r103 = r103 * r38;
    r103 = r103 * r38;
    r103 = r103 * r93;
    r103 = r103 * r39;
    r103 = r103 * r75;
    r98 = fma(r104, r103, r98);
    r105 = r10 * r38;
    r106 = r105 * r88;
    r98 = fma(r92, r106, r98);
    r103 = r35 * r55;
    r107 = r21 * r43;
    r107 = r107 * r38;
    r107 = r107 * r93;
    r107 = r107 * r39;
    r107 = r107 * r75;
    r107 = r107 * r104;
    r107 = fma(r53, r107, r93 * r103);
    r108 = r89 * r54;
    r107 = fma(r105, r108, r107);
    r109 = r43 * r38;
    r109 = r109 * r82;
    r109 = r109 * r44;
    r109 = r109 * r53;
    r107 = fma(r99, r109, r107);
    r109 = r98 + r107;
    r108 = fma(r56, r109, r73);
    r110 = r20 * r93;
    r111 = r43 * r53;
    r112 = -3.00000000000000000e+00;
    r112 = r38 * r112;
    r112 = r112 * r39;
    r112 = r112 * r75;
    r112 = r112 * r104;
    r111 = r111 * r112;
    r110 = fma(r93, r111, r103 * r110);
    r113 = r38 * r89;
    r114 = 6.00000000000000000e+00;
    r113 = r113 * r114;
    r110 = fma(r54, r113, r110);
    r115 = r43 * r53;
    r116 = r38 * r82;
    r117 = -6.00000000000000000e+00;
    r116 = r116 * r117;
    r116 = r116 * r44;
    r116 = r116 * r77;
    r110 = fma(r116, r115, r110);
    r110 = r110 + r98;
    r98 = r4 * r106;
    r115 = r21 * r64;
    r115 = r115 * r82;
    r115 = r115 * r39;
    r115 = r115 * r70;
    r108 = fma(r53, r115, r108);
    r113 = r43 * r93;
    r118 = r80 * r72;
    r113 = r113 * r35;
    r108 = fma(r118, r113, r108);
    r119 = r4 * r34;
    r119 = r119 * r93;
    r119 = r119 * r39;
    r119 = r119 * r75;
    r119 = r119 * r104;
    r119 = r119 * r53;
    r108 = fma(r47, r119, r108);
    r120 = r38 * r89;
    r108 = fma(r72, r120, r108);
    r121 = r4 * r92;
    r121 = r121 * r54;
    r108 = fma(r105, r121, r108);
    r122 = r4 * r82;
    r123 = r91 * r44;
    r123 = r123 * r77;
    r123 = r123 * r53;
    r123 = r123 * r47;
    r108 = fma(r123, r122, r108);
    r124 = r43 * r80;
    r124 = r124 * r93;
    r124 = r124 * r71;
    r108 = fma(r35, r124, r108);
    r125 = r4 * r93;
    r126 = r10 * r54;
    r126 = r126 * r95;
    r108 = fma(r126, r125, r108);
    r127 = r62 * r10;
    r127 = r127 * r7;
    r127 = fma(r61, r109, r109 * r127);
    r65 = r65 * r20;
    r65 = r65 * r63;
    r128 = 4.00000000000000000e+00;
    r66 = r66 * r128;
    r66 = r66 * r68;
    r127 = fma(r109, r65, r127);
    r127 = fma(r109, r66, r127);
    r129 = r127 * r53;
    r108 = fma(r71, r129, r108);
    r130 = r21 * r82;
    r130 = r130 * r39;
    r130 = r130 * r70;
    r108 = fma(r53, r130, r108);
    r131 = r38 * r89;
    r108 = fma(r71, r131, r108);
    r108 = fma(r5, r110, r108);
    r108 = fma(r89, r98, r108);
    r108 = fma(r64, r73, r108);
    r131 = r2 * r108;
    r130 = r28 * r28;
    r130 = r130 * r38;
    r129 = r20 * r93;
    r129 = r129 * r88;
    r129 = fma(r95, r129, r116 * r130);
    r130 = r93 * r112;
    r129 = fma(r100, r130, r129);
    r116 = r38 * r92;
    r116 = r116 * r114;
    r116 = r116 * r42;
    r129 = fma(r47, r116, r129);
    r129 = r129 + r107;
    r129 = fma(r4, r129, r57 * r109);
    r109 = r5 * r89;
    r129 = fma(r106, r109, r129);
    r107 = r21 * r28;
    r107 = r107 * r38;
    r107 = r107 * r82;
    r107 = r107 * r39;
    r129 = fma(r70, r107, r129);
    r116 = r28 * r38;
    r130 = r74 * r93;
    r130 = r130 * r44;
    r130 = r130 * r69;
    r130 = r130 * r75;
    r116 = r116 * r64;
    r129 = fma(r130, r116, r129);
    r125 = r28 * r38;
    r129 = fma(r130, r125, r129);
    r130 = r5 * r34;
    r130 = r130 * r93;
    r130 = r130 * r39;
    r130 = r130 * r75;
    r130 = r130 * r104;
    r130 = r130 * r53;
    r129 = fma(r47, r130, r129);
    r124 = r80 * r93;
    r124 = r124 * r71;
    r129 = fma(r95, r124, r129);
    r122 = r5 * r92;
    r122 = r122 * r54;
    r129 = fma(r105, r122, r129);
    r121 = r5 * r123;
    r120 = r127 * r71;
    r129 = fma(r47, r120, r129);
    r119 = r5 * r93;
    r129 = fma(r126, r119, r129);
    r113 = r95 * r118;
    r115 = r38 * r92;
    r129 = fma(r72, r115, r129);
    r73 = r38 * r92;
    r129 = fma(r71, r73, r129);
    r110 = r21 * r28;
    r110 = r110 * r38;
    r110 = r110 * r64;
    r110 = r110 * r82;
    r110 = r110 * r39;
    r129 = fma(r70, r110, r129);
    r129 = fma(r82, r121, r129);
    r129 = fma(r93, r113, r129);
    r110 = r3 * r129;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             0 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r131,
                                             r110);
    r110 = r43 * r38;
    r131 = r34 * r24;
    r131 = fma(r84, r131, r102);
    r73 = r10 * r27;
    r115 = r15 * r14;
    r119 = r12 * r17;
    r119 = fma(r74, r119, r74 * r115);
    r115 = r11 * r18;
    r119 = fma(r80, r115, r119);
    r120 = r16 * r13;
    r119 = fma(r74, r120, r119);
    r73 = r73 * r119;
    r120 = r10 * r19;
    r115 = r11 * r14;
    r122 = r16 * r17;
    r122 = fma(r80, r122, r80 * r115);
    r115 = r15 * r18;
    r122 = fma(r80, r115, r122);
    r124 = r12 * r13;
    r122 = fma(r74, r124, r122);
    r120 = fma(r122, r120, r73);
    r131 = r131 + r120;
    r124 = r10 * r24;
    r124 = r124 * r122;
    r115 = r10 * r29;
    r115 = fma(r119, r115, r124);
    r115 = r115 + r83;
    r115 = fma(r9, r115, r8 * r131);
    r131 = r24 * r87;
    r131 = r131 * r91;
    r83 = r19 * r119;
    r130 = r91 * r83;
    r125 = r131 + r130;
    r115 = fma(r41, r125, r115);
    r110 = r110 * r117;
    r110 = r110 * r115;
    r110 = r110 * r44;
    r110 = r110 * r77;
    r125 = r38 * r114;
    r96 = r94 + r96;
    r96 = r96 + r120;
    r120 = r27 * r91;
    r120 = r120 * r122;
    r131 = r131 + r120;
    r131 = fma(r8, r131, r41 * r96);
    r96 = r34 * r29;
    r96 = fma(r34, r90, r122 * r96);
    r94 = r10 * r19;
    r94 = r94 * r87;
    r116 = r10 * r24;
    r116 = fma(r119, r116, r94);
    r96 = r96 + r116;
    r131 = fma(r9, r96, r131);
    r125 = r125 * r131;
    r125 = fma(r54, r125, r53 * r110);
    r110 = r10 * r43;
    r110 = r110 * r131;
    r96 = r10 * r28;
    r124 = r85 + r124;
    r85 = r19 * r34;
    r124 = fma(r84, r85, r124);
    r84 = r34 * r29;
    r124 = fma(r119, r84, r124);
    r84 = r10 * r29;
    r90 = fma(r10, r90, r122 * r84);
    r90 = r90 + r116;
    r90 = fma(r8, r90, r41 * r124);
    r130 = r120 + r130;
    r90 = fma(r9, r130, r90);
    r96 = r96 * r90;
    r96 = fma(r39, r96, r39 * r110);
    r110 = r43 * r43;
    r110 = r110 * r115;
    r96 = fma(r99, r110, r96);
    r96 = fma(r115, r76, r96);
    r110 = r20 * r96;
    r125 = fma(r103, r110, r125);
    r130 = r21 * r28;
    r130 = r130 * r28;
    r130 = r130 * r38;
    r130 = r130 * r38;
    r130 = r130 * r96;
    r130 = r130 * r39;
    r130 = r130 * r75;
    r130 = fma(r104, r130, r90 * r106);
    r120 = r96 * r88;
    r130 = fma(r95, r120, r130);
    r130 = fma(r115, r101, r130);
    r125 = fma(r96, r111, r125);
    r125 = r125 + r130;
    r110 = r43 * r38;
    r110 = r110 * r115;
    r110 = r110 * r44;
    r110 = r110 * r53;
    r120 = r131 * r54;
    r120 = fma(r105, r120, r99 * r110);
    r110 = r21 * r43;
    r110 = r110 * r38;
    r110 = r110 * r96;
    r110 = r110 * r39;
    r110 = r110 * r75;
    r110 = r110 * r104;
    r120 = fma(r53, r110, r120);
    r120 = fma(r96, r103, r120);
    r130 = r130 + r120;
    r125 = fma(r56, r130, r5 * r125);
    r110 = r21 * r115;
    r110 = r110 * r39;
    r110 = r110 * r70;
    r125 = fma(r53, r110, r125);
    r124 = r43 * r96;
    r124 = r124 * r35;
    r125 = fma(r118, r124, r125);
    r84 = r43 * r80;
    r84 = r84 * r96;
    r84 = r84 * r71;
    r125 = fma(r35, r84, r125);
    r122 = r64 * r74;
    r122 = r122 * r96;
    r122 = r122 * r44;
    r122 = r122 * r69;
    r122 = r122 * r75;
    r125 = fma(r53, r122, r125);
    r85 = r62 * r10;
    r85 = r85 * r7;
    r85 = fma(r130, r85, r61 * r130);
    r85 = fma(r130, r66, r85);
    r85 = fma(r130, r65, r85);
    r107 = r85 * r53;
    r125 = fma(r71, r107, r125);
    r109 = r21 * r64;
    r109 = r109 * r115;
    r109 = r109 * r39;
    r109 = r109 * r70;
    r125 = fma(r53, r109, r125);
    r132 = r4 * r115;
    r125 = fma(r123, r132, r125);
    r133 = r96 * r126;
    r134 = r38 * r131;
    r125 = fma(r71, r134, r125);
    r135 = r4 * r34;
    r135 = r135 * r96;
    r135 = r135 * r39;
    r135 = r135 * r75;
    r135 = r135 * r104;
    r135 = r135 * r53;
    r125 = fma(r47, r135, r125);
    r136 = r38 * r131;
    r125 = fma(r72, r136, r125);
    r137 = r4 * r90;
    r137 = r137 * r54;
    r125 = fma(r105, r137, r125);
    r138 = r74 * r96;
    r138 = r138 * r44;
    r138 = r138 * r69;
    r138 = r138 * r75;
    r125 = fma(r53, r138, r125);
    r125 = fma(r4, r133, r125);
    r125 = fma(r131, r98, r125);
    r138 = r2 * r125;
    r137 = r38 * r114;
    r137 = r137 * r90;
    r137 = r137 * r42;
    r136 = r96 * r112;
    r136 = fma(r100, r136, r47 * r137);
    r137 = r28 * r28;
    r137 = r137 * r38;
    r137 = r137 * r38;
    r137 = r137 * r117;
    r137 = r137 * r115;
    r137 = r137 * r44;
    r136 = fma(r77, r137, r136);
    r135 = r20 * r96;
    r135 = r135 * r88;
    r136 = fma(r95, r135, r136);
    r136 = r136 + r120;
    r130 = fma(r57, r130, r4 * r136);
    r136 = r28 * r38;
    r136 = r136 * r74;
    r136 = r136 * r96;
    r136 = r136 * r44;
    r136 = r136 * r69;
    r130 = fma(r75, r136, r130);
    r120 = r38 * r90;
    r130 = fma(r72, r120, r130);
    r135 = r85 * r71;
    r130 = fma(r47, r135, r130);
    r137 = r80 * r96;
    r137 = r137 * r71;
    r130 = fma(r95, r137, r130);
    r134 = r21 * r28;
    r134 = r134 * r38;
    r134 = r134 * r64;
    r134 = r134 * r115;
    r134 = r134 * r39;
    r130 = fma(r70, r134, r130);
    r132 = r28 * r38;
    r132 = r132 * r64;
    r132 = r132 * r74;
    r132 = r132 * r96;
    r132 = r132 * r44;
    r132 = r132 * r69;
    r130 = fma(r75, r132, r130);
    r109 = r5 * r34;
    r109 = r109 * r96;
    r109 = r109 * r39;
    r109 = r109 * r75;
    r109 = r109 * r104;
    r109 = r109 * r53;
    r130 = fma(r47, r109, r130);
    r107 = r21 * r28;
    r107 = r107 * r38;
    r107 = r107 * r115;
    r107 = r107 * r39;
    r130 = fma(r70, r107, r130);
    r122 = r38 * r90;
    r130 = fma(r71, r122, r130);
    r84 = r5 * r90;
    r84 = r84 * r54;
    r130 = fma(r105, r84, r130);
    r124 = r5 * r131;
    r130 = fma(r106, r124, r130);
    r130 = fma(r115, r121, r130);
    r130 = fma(r5, r133, r130);
    r130 = fma(r96, r113, r130);
    r124 = r3 * r130;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             2 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r138,
                                             r124);
    r124 = r19 * r91;
    r26 = fma(r74, r26, r80 * r22);
    r26 = fma(r80, r23, r26);
    r26 = fma(r80, r25, r26);
    r124 = r124 * r26;
    r79 = r91 * r79;
    r25 = r124 + r79;
    r23 = r10 * r27;
    r23 = r23 * r26;
    r94 = r94 + r23;
    r22 = r34 * r24;
    r94 = fma(r119, r22, r94);
    r138 = r34 * r29;
    r94 = fma(r78, r138, r94);
    r94 = fma(r8, r94, r41 * r25);
    r25 = r10 * r29;
    r25 = fma(r10, r83, r26 * r25);
    r25 = r25 + r86;
    r94 = fma(r9, r25, r94);
    r25 = r10 * r28;
    r138 = r10 * r24;
    r138 = r138 * r26;
    r73 = r73 + r138;
    r73 = r73 + r97;
    r97 = r34 * r29;
    r83 = fma(r34, r83, r26 * r97);
    r83 = r83 + r86;
    r83 = fma(r41, r83, r8 * r73);
    r87 = r27 * r87;
    r87 = r87 * r91;
    r124 = r124 + r87;
    r83 = fma(r9, r124, r83);
    r25 = r25 * r83;
    r124 = r43 * r43;
    r124 = r124 * r94;
    r124 = fma(r99, r124, r39 * r25);
    r25 = r10 * r43;
    r102 = r81 + r102;
    r81 = r27 * r34;
    r102 = fma(r119, r81, r102);
    r102 = r102 + r138;
    r79 = r87 + r79;
    r79 = fma(r8, r79, r9 * r102);
    r8 = r10 * r29;
    r8 = fma(r78, r8, r23);
    r8 = r8 + r116;
    r79 = fma(r41, r8, r79);
    r25 = r25 * r79;
    r124 = fma(r39, r25, r124);
    r124 = fma(r94, r76, r124);
    r25 = r124 * r88;
    r25 = fma(r95, r25, r94 * r101);
    r8 = r21 * r28;
    r8 = r8 * r28;
    r8 = r8 * r38;
    r8 = r8 * r38;
    r8 = r8 * r124;
    r8 = r8 * r39;
    r8 = r8 * r75;
    r25 = fma(r104, r8, r25);
    r25 = fma(r83, r106, r25);
    r8 = r21 * r43;
    r8 = r8 * r38;
    r8 = r8 * r124;
    r8 = r8 * r39;
    r8 = r8 * r75;
    r8 = r8 * r104;
    r41 = r79 * r54;
    r41 = fma(r105, r41, r53 * r8);
    r8 = r43 * r38;
    r8 = r8 * r94;
    r8 = r8 * r44;
    r8 = r8 * r53;
    r41 = fma(r99, r8, r41);
    r41 = fma(r124, r103, r41);
    r8 = r25 + r41;
    r116 = r38 * r114;
    r116 = r116 * r79;
    r116 = fma(r54, r116, r124 * r111);
    r23 = r43 * r38;
    r23 = r23 * r117;
    r23 = r23 * r94;
    r23 = r23 * r44;
    r23 = r23 * r77;
    r116 = fma(r53, r23, r116);
    r78 = r20 * r124;
    r116 = fma(r103, r78, r116);
    r116 = r116 + r25;
    r116 = fma(r5, r116, r56 * r8);
    r25 = r62 * r10;
    r25 = r25 * r7;
    r25 = fma(r8, r25, r61 * r8);
    r25 = fma(r8, r65, r25);
    r25 = fma(r8, r66, r25);
    r78 = r25 * r53;
    r116 = fma(r71, r78, r116);
    r23 = r43 * r80;
    r23 = r23 * r124;
    r23 = r23 * r71;
    r116 = fma(r35, r23, r116);
    r102 = r4 * r124;
    r116 = fma(r126, r102, r116);
    r9 = r21 * r64;
    r9 = r9 * r94;
    r9 = r9 * r39;
    r9 = r9 * r70;
    r116 = fma(r53, r9, r116);
    r87 = r43 * r124;
    r87 = r87 * r35;
    r116 = fma(r118, r87, r116);
    r81 = r4 * r34;
    r81 = r81 * r124;
    r81 = r81 * r39;
    r81 = r81 * r75;
    r81 = r81 * r104;
    r81 = r81 * r53;
    r116 = fma(r47, r81, r116);
    r138 = r38 * r79;
    r116 = fma(r72, r138, r116);
    r119 = r21 * r94;
    r119 = r119 * r39;
    r119 = r119 * r70;
    r116 = fma(r53, r119, r116);
    r91 = r38 * r79;
    r116 = fma(r71, r91, r116);
    r73 = r74 * r124;
    r73 = r73 * r44;
    r73 = r73 * r69;
    r73 = r73 * r75;
    r116 = fma(r53, r73, r116);
    r86 = r4 * r94;
    r116 = fma(r123, r86, r116);
    r97 = r4 * r83;
    r97 = r97 * r54;
    r116 = fma(r105, r97, r116);
    r26 = r64 * r74;
    r26 = r26 * r124;
    r26 = r26 * r44;
    r26 = r26 * r69;
    r26 = r26 * r75;
    r116 = fma(r53, r26, r116);
    r116 = fma(r79, r98, r116);
    r26 = r2 * r116;
    r97 = r28 * r28;
    r97 = r97 * r38;
    r97 = r97 * r38;
    r97 = r97 * r117;
    r97 = r97 * r94;
    r97 = r97 * r44;
    r86 = r20 * r124;
    r86 = r86 * r88;
    r86 = fma(r95, r86, r77 * r97);
    r97 = r38 * r114;
    r97 = r97 * r83;
    r97 = r97 * r42;
    r86 = fma(r47, r97, r86);
    r73 = r124 * r112;
    r86 = fma(r100, r73, r86);
    r86 = r86 + r41;
    r86 = fma(r4, r86, r57 * r8);
    r8 = r21 * r28;
    r8 = r8 * r38;
    r8 = r8 * r64;
    r8 = r8 * r94;
    r8 = r8 * r39;
    r86 = fma(r70, r8, r86);
    r41 = r21 * r28;
    r41 = r41 * r38;
    r41 = r41 * r94;
    r41 = r41 * r39;
    r86 = fma(r70, r41, r86);
    r73 = r38 * r83;
    r86 = fma(r72, r73, r86);
    r97 = r5 * r79;
    r86 = fma(r106, r97, r86);
    r91 = r28 * r38;
    r91 = r91 * r64;
    r91 = r91 * r74;
    r91 = r91 * r124;
    r91 = r91 * r44;
    r91 = r91 * r69;
    r86 = fma(r75, r91, r86);
    r119 = r5 * r124;
    r86 = fma(r126, r119, r86);
    r138 = r80 * r124;
    r138 = r138 * r71;
    r86 = fma(r95, r138, r86);
    r81 = r38 * r83;
    r86 = fma(r71, r81, r86);
    r87 = r5 * r34;
    r87 = r87 * r124;
    r87 = r87 * r39;
    r87 = r87 * r75;
    r87 = r87 * r104;
    r87 = r87 * r53;
    r86 = fma(r47, r87, r86);
    r9 = r25 * r71;
    r86 = fma(r47, r9, r86);
    r102 = r28 * r38;
    r102 = r102 * r74;
    r102 = r102 * r124;
    r102 = r102 * r44;
    r102 = r102 * r69;
    r86 = fma(r75, r102, r86);
    r23 = r5 * r83;
    r23 = r23 * r54;
    r86 = fma(r105, r23, r86);
    r86 = fma(r124, r113, r86);
    r86 = fma(r94, r121, r86);
    r23 = r3 * r86;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r26, r23);
    r23 = r36 * r38;
    r23 = r23 * r114;
    r26 = r10 * r32;
    r26 = r26 * r28;
    r26 = fma(r50, r76, r39 * r26);
    r102 = r50 * r43;
    r102 = r102 * r43;
    r26 = fma(r99, r102, r26);
    r9 = r10 * r36;
    r9 = r9 * r43;
    r26 = fma(r39, r9, r26);
    r23 = fma(r26, r111, r54 * r23);
    r9 = r20 * r26;
    r23 = fma(r103, r9, r23);
    r102 = r50 * r43;
    r102 = r102 * r38;
    r102 = r102 * r117;
    r102 = r102 * r44;
    r102 = r102 * r77;
    r23 = fma(r53, r102, r23);
    r87 = r21 * r28;
    r87 = r87 * r28;
    r87 = r87 * r38;
    r87 = r87 * r38;
    r87 = r87 * r26;
    r87 = r87 * r39;
    r87 = r87 * r75;
    r87 = fma(r104, r87, r32 * r106);
    r81 = r26 * r88;
    r87 = fma(r95, r81, r87);
    r87 = fma(r50, r101, r87);
    r23 = r23 + r87;
    r102 = r36 * r54;
    r9 = r21 * r43;
    r9 = r9 * r38;
    r9 = r9 * r26;
    r9 = r9 * r39;
    r9 = r9 * r75;
    r9 = r9 * r104;
    r9 = fma(r53, r9, r105 * r102);
    r102 = r50 * r43;
    r102 = r102 * r38;
    r102 = r102 * r44;
    r102 = r102 * r53;
    r9 = fma(r99, r102, r9);
    r9 = fma(r26, r103, r9);
    r87 = r87 + r9;
    r23 = fma(r56, r87, r5 * r23);
    r102 = r36 * r38;
    r23 = fma(r72, r102, r23);
    r81 = r43 * r26;
    r81 = r81 * r35;
    r23 = fma(r118, r81, r23);
    r138 = r4 * r26;
    r23 = fma(r126, r138, r23);
    r119 = r43 * r80;
    r119 = r119 * r26;
    r119 = r119 * r71;
    r23 = fma(r35, r119, r23);
    r91 = r64 * r74;
    r91 = r91 * r26;
    r91 = r91 * r44;
    r91 = r91 * r69;
    r91 = r91 * r75;
    r23 = fma(r53, r91, r23);
    r97 = r74 * r26;
    r97 = r97 * r44;
    r97 = r97 * r69;
    r97 = r97 * r75;
    r23 = fma(r53, r97, r23);
    r73 = r4 * r34;
    r73 = r73 * r26;
    r73 = r73 * r39;
    r73 = r73 * r75;
    r73 = r73 * r104;
    r73 = r73 * r53;
    r23 = fma(r47, r73, r23);
    r41 = r4 * r50;
    r23 = fma(r123, r41, r23);
    r8 = r62 * r10;
    r8 = r8 * r7;
    r8 = fma(r61, r87, r87 * r8);
    r8 = fma(r87, r66, r8);
    r8 = fma(r87, r65, r8);
    r78 = r8 * r53;
    r23 = fma(r71, r78, r23);
    r22 = r21 * r50;
    r22 = r22 * r64;
    r22 = r22 * r39;
    r22 = r22 * r70;
    r23 = fma(r53, r22, r23);
    r84 = r21 * r50;
    r84 = r84 * r39;
    r84 = r84 * r70;
    r23 = fma(r53, r84, r23);
    r122 = r4 * r32;
    r122 = r122 * r54;
    r23 = fma(r105, r122, r23);
    r107 = r36 * r38;
    r23 = fma(r71, r107, r23);
    r23 = fma(r36, r98, r23);
    r107 = r2 * r23;
    r122 = r32 * r38;
    r122 = r122 * r114;
    r122 = r122 * r42;
    r84 = r26 * r112;
    r84 = fma(r100, r84, r47 * r122);
    r122 = r50 * r28;
    r122 = r122 * r28;
    r122 = r122 * r38;
    r122 = r122 * r38;
    r122 = r122 * r117;
    r122 = r122 * r44;
    r84 = fma(r77, r122, r84);
    r22 = r20 * r26;
    r22 = r22 * r88;
    r84 = fma(r95, r22, r84);
    r84 = r84 + r9;
    r87 = fma(r57, r87, r4 * r84);
    r84 = r80 * r26;
    r84 = r84 * r71;
    r87 = fma(r95, r84, r87);
    r9 = r32 * r38;
    r87 = fma(r71, r9, r87);
    r22 = r28 * r38;
    r22 = r22 * r64;
    r22 = r22 * r74;
    r22 = r22 * r26;
    r22 = r22 * r44;
    r22 = r22 * r69;
    r87 = fma(r75, r22, r87);
    r122 = r21 * r50;
    r122 = r122 * r28;
    r122 = r122 * r38;
    r122 = r122 * r64;
    r122 = r122 * r39;
    r87 = fma(r70, r122, r87);
    r78 = r5 * r26;
    r87 = fma(r126, r78, r87);
    r41 = r21 * r50;
    r41 = r41 * r28;
    r41 = r41 * r38;
    r41 = r41 * r39;
    r87 = fma(r70, r41, r87);
    r73 = r5 * r34;
    r73 = r73 * r26;
    r73 = r73 * r39;
    r73 = r73 * r75;
    r73 = r73 * r104;
    r73 = r73 * r53;
    r87 = fma(r47, r73, r87);
    r97 = r8 * r71;
    r87 = fma(r47, r97, r87);
    r91 = r32 * r38;
    r87 = fma(r72, r91, r87);
    r119 = r5 * r32;
    r119 = r119 * r54;
    r87 = fma(r105, r119, r87);
    r138 = r28 * r38;
    r138 = r138 * r74;
    r138 = r138 * r26;
    r138 = r138 * r44;
    r138 = r138 * r69;
    r87 = fma(r75, r138, r87);
    r81 = r5 * r36;
    r87 = fma(r106, r81, r87);
    r87 = fma(r50, r121, r87);
    r87 = fma(r26, r113, r87);
    r81 = r3 * r87;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r107, r81);
    r81 = r10 * r37;
    r81 = r81 * r28;
    r81 = fma(r39, r81, r33 * r76);
    r107 = r33 * r43;
    r107 = r107 * r43;
    r81 = fma(r99, r107, r81);
    r138 = r10 * r49;
    r138 = r138 * r43;
    r81 = fma(r39, r138, r81);
    r138 = r20 * r81;
    r107 = r33 * r43;
    r107 = r107 * r38;
    r107 = r107 * r117;
    r107 = r107 * r44;
    r107 = r107 * r77;
    r107 = fma(r53, r107, r103 * r138);
    r138 = r49 * r38;
    r138 = r138 * r114;
    r107 = fma(r54, r138, r107);
    r119 = fma(r37, r106, r33 * r101);
    r91 = r81 * r88;
    r119 = fma(r95, r91, r119);
    r97 = r21 * r28;
    r97 = r97 * r28;
    r97 = r97 * r38;
    r97 = r97 * r38;
    r97 = r97 * r81;
    r97 = r97 * r39;
    r97 = r97 * r75;
    r119 = fma(r104, r97, r119);
    r107 = fma(r81, r111, r107);
    r107 = r107 + r119;
    r138 = r33 * r43;
    r138 = r138 * r38;
    r138 = r138 * r44;
    r138 = r138 * r53;
    r138 = fma(r99, r138, r81 * r103);
    r103 = r21 * r43;
    r103 = r103 * r38;
    r103 = r103 * r81;
    r103 = r103 * r39;
    r103 = r103 * r75;
    r103 = r103 * r104;
    r138 = fma(r53, r103, r138);
    r97 = r49 * r54;
    r138 = fma(r105, r97, r138);
    r119 = r119 + r138;
    r107 = fma(r56, r119, r5 * r107);
    r97 = r4 * r37;
    r97 = r97 * r54;
    r107 = fma(r105, r97, r107);
    r103 = r4 * r81;
    r107 = fma(r126, r103, r107);
    r91 = r43 * r81;
    r91 = r91 * r35;
    r107 = fma(r118, r91, r107);
    r73 = r49 * r38;
    r107 = fma(r71, r73, r107);
    r41 = r21 * r33;
    r41 = r41 * r39;
    r41 = r41 * r70;
    r107 = fma(r53, r41, r107);
    r78 = r4 * r33;
    r107 = fma(r123, r78, r107);
    r122 = r4 * r34;
    r122 = r122 * r81;
    r122 = r122 * r39;
    r122 = r122 * r75;
    r122 = r122 * r104;
    r122 = r122 * r53;
    r107 = fma(r47, r122, r107);
    r22 = r43 * r80;
    r22 = r22 * r81;
    r22 = r22 * r71;
    r107 = fma(r35, r22, r107);
    r9 = r49 * r38;
    r107 = fma(r72, r9, r107);
    r84 = r74 * r81;
    r84 = r84 * r44;
    r84 = r84 * r69;
    r84 = r84 * r75;
    r107 = fma(r53, r84, r107);
    r102 = r62 * r10;
    r102 = r102 * r7;
    r102 = fma(r61, r119, r119 * r102);
    r102 = fma(r119, r65, r102);
    r102 = fma(r119, r66, r102);
    r109 = r102 * r53;
    r107 = fma(r71, r109, r107);
    r132 = r64 * r74;
    r132 = r132 * r81;
    r132 = r132 * r44;
    r132 = r132 * r69;
    r132 = r132 * r75;
    r107 = fma(r53, r132, r107);
    r134 = r21 * r33;
    r134 = r134 * r64;
    r134 = r134 * r39;
    r134 = r134 * r70;
    r107 = fma(r53, r134, r107);
    r107 = fma(r49, r98, r107);
    r134 = r2 * r107;
    r132 = r33 * r28;
    r132 = r132 * r28;
    r132 = r132 * r38;
    r132 = r132 * r38;
    r132 = r132 * r117;
    r132 = r132 * r44;
    r109 = r37 * r38;
    r109 = r109 * r114;
    r109 = r109 * r42;
    r109 = fma(r47, r109, r77 * r132);
    r132 = r20 * r81;
    r132 = r132 * r88;
    r109 = fma(r95, r132, r109);
    r84 = r81 * r112;
    r109 = fma(r100, r84, r109);
    r109 = r109 + r138;
    r109 = fma(r4, r109, r57 * r119);
    r119 = r21 * r33;
    r119 = r119 * r28;
    r119 = r119 * r38;
    r119 = r119 * r39;
    r109 = fma(r70, r119, r109);
    r138 = r37 * r38;
    r109 = fma(r72, r138, r109);
    r84 = r5 * r37;
    r84 = r84 * r54;
    r109 = fma(r105, r84, r109);
    r132 = r5 * r81;
    r109 = fma(r126, r132, r109);
    r9 = r80 * r81;
    r9 = r9 * r71;
    r109 = fma(r95, r9, r109);
    r22 = r102 * r71;
    r109 = fma(r47, r22, r109);
    r122 = r28 * r38;
    r122 = r122 * r64;
    r122 = r122 * r74;
    r122 = r122 * r81;
    r122 = r122 * r44;
    r122 = r122 * r69;
    r109 = fma(r75, r122, r109);
    r78 = r5 * r34;
    r78 = r78 * r81;
    r78 = r78 * r39;
    r78 = r78 * r75;
    r78 = r78 * r104;
    r78 = r78 * r53;
    r109 = fma(r47, r78, r109);
    r41 = r28 * r38;
    r41 = r41 * r74;
    r41 = r41 * r81;
    r41 = r41 * r44;
    r41 = r41 * r69;
    r109 = fma(r75, r41, r109);
    r73 = r37 * r38;
    r109 = fma(r71, r73, r109);
    r91 = r21 * r33;
    r91 = r91 * r28;
    r91 = r91 * r38;
    r91 = r91 * r64;
    r91 = r91 * r39;
    r109 = fma(r70, r91, r109);
    r103 = r5 * r49;
    r109 = fma(r106, r103, r109);
    r109 = fma(r33, r121, r109);
    r109 = fma(r81, r113, r109);
    r103 = r3 * r109;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             8 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r134,
                                             r103);
    r103 = r10 * r40;
    r103 = r103 * r28;
    r76 = fma(r31, r76, r39 * r103);
    r103 = r10 * r48;
    r103 = r103 * r43;
    r76 = fma(r39, r103, r76);
    r134 = r31 * r43;
    r134 = r134 * r43;
    r76 = fma(r99, r134, r76);
    r134 = r43 * r43;
    r134 = r39 * r134;
    r134 = r134 * r44;
    r134 = r134 * r38;
    r134 = r134 * r75;
    r134 = r134 * r30;
    r134 = r134 * r76;
    r111 = fma(r20, r134, r76 * r111);
    r30 = r48 * r38;
    r30 = r30 * r114;
    r111 = fma(r54, r30, r111);
    r103 = r31 * r43;
    r103 = r103 * r38;
    r103 = r103 * r117;
    r103 = r103 * r44;
    r103 = r103 * r77;
    r111 = fma(r53, r103, r111);
    r101 = fma(r40, r106, r31 * r101);
    r91 = r76 * r88;
    r101 = fma(r95, r91, r101);
    r73 = r21 * r28;
    r73 = r73 * r28;
    r73 = r73 * r38;
    r73 = r73 * r38;
    r73 = r73 * r76;
    r73 = r73 * r39;
    r73 = r73 * r75;
    r101 = fma(r104, r73, r101);
    r111 = r111 + r101;
    r103 = r21 * r43;
    r103 = r103 * r38;
    r103 = r103 * r76;
    r103 = r103 * r39;
    r103 = r103 * r75;
    r103 = r103 * r104;
    r103 = fma(r53, r103, r134);
    r134 = r48 * r54;
    r103 = fma(r105, r134, r103);
    r30 = r31 * r43;
    r30 = r30 * r38;
    r30 = r30 * r44;
    r30 = r30 * r53;
    r103 = fma(r99, r30, r103);
    r101 = r101 + r103;
    r56 = fma(r56, r101, r5 * r111);
    r111 = r48 * r38;
    r56 = fma(r72, r111, r56);
    r30 = r4 * r31;
    r56 = fma(r123, r30, r56);
    r123 = r4 * r40;
    r123 = r123 * r54;
    r56 = fma(r105, r123, r56);
    r134 = r43 * r76;
    r134 = r134 * r35;
    r56 = fma(r118, r134, r56);
    r118 = r43 * r80;
    r118 = r118 * r76;
    r118 = r118 * r71;
    r56 = fma(r35, r118, r56);
    r35 = r62 * r10;
    r35 = r35 * r7;
    r35 = fma(r101, r35, r61 * r101);
    r35 = fma(r101, r66, r35);
    r35 = fma(r101, r65, r35);
    r65 = r35 * r53;
    r56 = fma(r71, r65, r56);
    r66 = r4 * r76;
    r56 = fma(r126, r66, r56);
    r61 = r21 * r31;
    r61 = r61 * r39;
    r61 = r61 * r70;
    r56 = fma(r53, r61, r56);
    r99 = r64 * r74;
    r99 = r99 * r76;
    r99 = r99 * r44;
    r99 = r99 * r69;
    r99 = r99 * r75;
    r56 = fma(r53, r99, r56);
    r73 = r4 * r34;
    r73 = r73 * r76;
    r73 = r73 * r39;
    r73 = r73 * r75;
    r73 = r73 * r104;
    r73 = r73 * r53;
    r56 = fma(r47, r73, r56);
    r91 = r48 * r38;
    r56 = fma(r71, r91, r56);
    r41 = r21 * r31;
    r41 = r41 * r64;
    r41 = r41 * r39;
    r41 = r41 * r70;
    r56 = fma(r53, r41, r56);
    r78 = r74 * r76;
    r78 = r78 * r44;
    r78 = r78 * r69;
    r78 = r78 * r75;
    r56 = fma(r53, r78, r56);
    r56 = fma(r48, r98, r56);
    r78 = r2 * r56;
    r41 = r31 * r28;
    r41 = r41 * r28;
    r41 = r41 * r38;
    r41 = r41 * r38;
    r41 = r41 * r117;
    r41 = r41 * r44;
    r117 = r40 * r38;
    r117 = r117 * r114;
    r117 = r117 * r42;
    r117 = fma(r47, r117, r77 * r41);
    r41 = r20 * r76;
    r41 = r41 * r88;
    r117 = fma(r95, r41, r117);
    r42 = r76 * r112;
    r117 = fma(r100, r42, r117);
    r117 = r117 + r103;
    r117 = fma(r4, r117, r57 * r101);
    r101 = r28 * r38;
    r101 = r101 * r74;
    r101 = r101 * r76;
    r101 = r101 * r44;
    r101 = r101 * r69;
    r117 = fma(r75, r101, r117);
    r57 = r21 * r31;
    r57 = r57 * r28;
    r57 = r57 * r38;
    r57 = r57 * r39;
    r117 = fma(r70, r57, r117);
    r103 = r40 * r38;
    r117 = fma(r71, r103, r117);
    r42 = r5 * r48;
    r117 = fma(r106, r42, r117);
    r106 = r21 * r31;
    r106 = r106 * r28;
    r106 = r106 * r38;
    r106 = r106 * r64;
    r106 = r106 * r39;
    r117 = fma(r70, r106, r117);
    r70 = r40 * r38;
    r117 = fma(r72, r70, r117);
    r72 = r5 * r40;
    r72 = r72 * r54;
    r117 = fma(r105, r72, r117);
    r41 = r28 * r38;
    r41 = r41 * r64;
    r41 = r41 * r74;
    r41 = r41 * r76;
    r41 = r41 * r44;
    r41 = r41 * r69;
    r117 = fma(r75, r41, r117);
    r69 = r5 * r76;
    r117 = fma(r126, r69, r117);
    r126 = r80 * r76;
    r126 = r126 * r71;
    r117 = fma(r95, r126, r117);
    r95 = r35 * r71;
    r117 = fma(r47, r95, r117);
    r44 = r5 * r34;
    r44 = r44 * r76;
    r44 = r44 * r39;
    r44 = r44 * r75;
    r44 = r44 * r104;
    r44 = r44 * r53;
    r117 = fma(r47, r44, r117);
    r117 = fma(r76, r113, r117);
    r117 = fma(r31, r121, r117);
    r44 = r3 * r117;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r78, r44);
    r44 = r3 * r21;
    r44 = r44 * r1;
    r78 = r21 * r0;
    r95 = r2 * r78;
    r44 = fma(r108, r95, r129 * r44);
    r126 = r3 * r21;
    r126 = r126 * r1;
    r126 = fma(r125, r95, r130 * r126);
    WriteSum2<double, double>((double*)inout_shared, r44, r126);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r3 * r21;
    r126 = r126 * r1;
    r126 = fma(r116, r95, r86 * r126);
    r44 = r3 * r21;
    r44 = r44 * r1;
    r44 = fma(r23, r95, r87 * r44);
    WriteSum2<double, double>((double*)inout_shared, r126, r44);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = r3 * r21;
    r44 = r44 * r1;
    r44 = fma(r107, r95, r109 * r44);
    r126 = r3 * r21;
    r126 = r126 * r1;
    r126 = fma(r56, r95, r117 * r126);
    WriteSum2<double, double>((double*)inout_shared, r44, r126);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r108 * r108;
    r44 = r2 * r2;
    r69 = r129 * r129;
    r41 = r3 * r3;
    r69 = fma(r41, r69, r44 * r126);
    r126 = r125 * r125;
    r72 = r130 * r130;
    r72 = fma(r41, r72, r44 * r126);
    WriteSum2<double, double>((double*)inout_shared, r69, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r116 * r116;
    r69 = r86 * r86;
    r69 = fma(r41, r69, r44 * r72);
    r72 = r87 * r87;
    r126 = r23 * r23;
    r126 = fma(r44, r126, r41 * r72);
    WriteSum2<double, double>((double*)inout_shared, r69, r126);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r107 * r107;
    r69 = r109 * r109;
    r69 = fma(r41, r69, r44 * r126);
    r126 = r117 * r117;
    r72 = r56 * r56;
    r72 = fma(r44, r72, r41 * r126);
    WriteSum2<double, double>((double*)inout_shared, r69, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r129 * r130;
    r69 = r108 * r125;
    r69 = fma(r44, r69, r41 * r72);
    r72 = r108 * r116;
    r126 = r129 * r86;
    r126 = fma(r41, r126, r44 * r72);
    WriteSum2<double, double>((double*)inout_shared, r69, r126);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r108 * r23;
    r69 = r129 * r87;
    r69 = fma(r41, r69, r44 * r126);
    r126 = r108 * r107;
    r72 = r129 * r109;
    r72 = fma(r41, r72, r44 * r126);
    WriteSum2<double, double>((double*)inout_shared, r69, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r129 * r117;
    r69 = r108 * r56;
    r69 = fma(r44, r69, r41 * r72);
    r72 = r130 * r86;
    r126 = r125 * r116;
    r126 = fma(r44, r126, r41 * r72);
    WriteSum2<double, double>((double*)inout_shared, r69, r126);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r125 * r23;
    r69 = r130 * r87;
    r69 = fma(r41, r69, r44 * r126);
    r126 = r125 * r107;
    r72 = r130 * r109;
    r72 = fma(r41, r72, r44 * r126);
    WriteSum2<double, double>((double*)inout_shared, r69, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r130 * r117;
    r69 = r125 * r56;
    r69 = fma(r44, r69, r41 * r72);
    r72 = r86 * r87;
    r126 = r116 * r23;
    r126 = fma(r44, r126, r41 * r72);
    WriteSum2<double, double>((double*)inout_shared, r69, r126);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r116 * r107;
    r69 = r86 * r109;
    r69 = fma(r41, r69, r44 * r126);
    r126 = r86 * r117;
    r72 = r116 * r56;
    r72 = fma(r44, r72, r41 * r126);
    WriteSum2<double, double>((double*)inout_shared, r69, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r23 * r107;
    r69 = r87 * r109;
    r69 = fma(r41, r69, r44 * r72);
    r72 = r87 * r117;
    r126 = r23 * r56;
    r126 = fma(r44, r126, r41 * r72);
    WriteSum2<double, double>((double*)inout_shared, r69, r126);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r109 * r117;
    r69 = r107 * r56;
    r69 = fma(r44, r69, r41 * r126);
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
    r78 = r59 * r78;
    WriteSum2<double, double>((double*)inout_shared, r78, r69);
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
    r78 = r7 * r53;
    r78 = r78 * r71;
    r78 = fma(r95, r78, r47 * r69);
    r69 = r3 * r21;
    r69 = r69 * r1;
    r69 = r69 * r71;
    r69 = r69 * r47;
    r126 = r53 * r71;
    r126 = r126 * r63;
    r126 = fma(r95, r126, r63 * r69);
    WriteSum2<double, double>((double*)inout_shared, r78, r126);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            2 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r3 * r21;
    r126 = r126 * r60;
    r78 = r2 * r34;
    r78 = r78 * r0;
    r78 = r78 * r47;
    r78 = fma(r54, r78, r1 * r126);
    r126 = r3 * r34;
    r126 = r126 * r1;
    r126 = r126 * r47;
    r126 = fma(r54, r126, r52 * r95);
    WriteSum2<double, double>((double*)inout_shared, r78, r126);
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
    r78 = r53 * r71;
    r78 = r78 * r68;
    r78 = fma(r95, r78, r68 * r126);
    r126 = r3 * r21;
    r126 = r126 * r1;
    r126 = r126 * r71;
    r126 = r126 * r47;
    r0 = r53 * r71;
    r0 = r0 * r67;
    r0 = fma(r95, r0, r67 * r126);
    WriteSum2<double, double>((double*)inout_shared, r78, r0);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            6 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r3 * r21;
    r0 = r0 * r7;
    r0 = r0 * r1;
    r95 = r7 * r95;
    WriteSum2<double, double>((double*)inout_shared, r95, r0);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            8 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r59 * r59;
    r95 = r58 * r58;
    WriteSum2<double, double>((double*)inout_shared, r0, r95);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            0 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r95 = r38 * r63;
    r95 = r95 * r44;
    r0 = r28 * r38;
    r0 = r0 * r63;
    r0 = r0 * r41;
    r0 = fma(r88, r0, r55 * r95);
    r95 = r38 * r44;
    r95 = r95 * r67;
    r1 = r28 * r38;
    r1 = r1 * r41;
    r1 = r1 * r88;
    r1 = fma(r67, r1, r55 * r95);
    WriteSum2<double, double>((double*)inout_shared, r0, r1);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            2 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r53 * r44;
    r95 = r43 * r28;
    r51 = r45 * r51;
    r51 = 1.0 / r51;
    r6 = r46 * r6;
    r6 = 1.0 / r6;
    r95 = r95 * r38;
    r95 = r95 * r38;
    r95 = r95 * r128;
    r95 = r95 * r51;
    r95 = r95 * r6;
    r95 = r95 * r47;
    r6 = r60 * r41;
    r51 = fma(r60, r6, r0 * r95);
    r128 = r53 * r41;
    r46 = r52 * r52;
    r46 = fma(r44, r46, r95 * r128);
    WriteSum2<double, double>((double*)inout_shared, r51, r46);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            4 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r68 * r68;
    r51 = r38 * r46;
    r128 = r44 * r55;
    r95 = r38 * r46;
    r45 = r28 * r41;
    r45 = r45 * r88;
    r95 = fma(r45, r95, r128 * r51);
    r51 = r7 * r46;
    r51 = r38 * r51;
    r78 = r7 * r51;
    r78 = fma(r45, r78, r128 * r78);
    WriteSum2<double, double>((double*)inout_shared, r95, r78);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            6 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = r63 * r44;
    r126 = r63 * r41;
    WriteSum2<double, double>((double*)inout_shared, r78, r126);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            8 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = 0.00000000000000000e+00;
    r78 = r2 * r7;
    r78 = r78 * r59;
    r78 = r78 * r53;
    r78 = r78 * r71;
    WriteSum2<double, double>((double*)inout_shared, r126, r78);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            0 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = r2 * r59;
    r78 = r78 * r53;
    r78 = r78 * r71;
    r78 = r78 * r63;
    r69 = r2 * r10;
    r69 = r69 * r59;
    r69 = r69 * r47;
    r69 = r69 * r54;
    WriteSum2<double, double>((double*)inout_shared, r78, r69);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            2 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = r2 * r52;
    r69 = r69 * r59;
    r78 = r2 * r59;
    r78 = r78 * r53;
    r78 = r78 * r71;
    r78 = r78 * r68;
    WriteSum2<double, double>((double*)inout_shared, r69, r78);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            4 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = r2 * r7;
    r78 = r78 * r59;
    r59 = r2 * r59;
    r59 = r59 * r53;
    r59 = r59 * r71;
    r59 = r59 * r67;
    WriteSum2<double, double>((double*)inout_shared, r59, r78);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            6 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = r3 * r7;
    r78 = r78 * r58;
    r78 = r78 * r71;
    r78 = r78 * r47;
    WriteSum2<double, double>((double*)inout_shared, r126, r78);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            8 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = r3 * r60;
    r60 = r60 * r58;
    r78 = r3 * r58;
    r78 = r78 * r71;
    r78 = r78 * r47;
    r78 = r78 * r63;
    WriteSum2<double, double>((double*)inout_shared, r78, r60);
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
    r78 = r3 * r58;
    r78 = r78 * r71;
    r78 = r78 * r47;
    r78 = r78 * r68;
    WriteSum2<double, double>((double*)inout_shared, r60, r78);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            12 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = r3 * r58;
    r78 = r78 * r71;
    r78 = r78 * r47;
    r78 = r78 * r67;
    WriteSum2<double, double>((double*)inout_shared, r78, r126);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            14 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r126 = r3 * r7;
    r126 = r126 * r58;
    r58 = r38 * r68;
    r58 = r58 * r44;
    r78 = r28 * r38;
    r78 = r78 * r68;
    r78 = r78 * r41;
    r78 = fma(r88, r78, r55 * r58);
    WriteSum2<double, double>((double*)inout_shared, r126, r78);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            16 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = r47 * r6;
    r126 = r71 * r78;
    r58 = r43 * r7;
    r58 = r58 * r77;
    r58 = r58 * r104;
    r58 = r58 * r47;
    r58 = r58 * r105;
    r58 = fma(r0, r58, r7 * r126);
    r60 = r71 * r0;
    r59 = r52 * r60;
    r69 = r28 * r7;
    r69 = r69 * r77;
    r69 = r69 * r104;
    r69 = r69 * r53;
    r69 = r69 * r47;
    r69 = r69 * r41;
    r69 = fma(r105, r69, r7 * r59);
    WriteSum2<double, double>((double*)inout_shared, r58, r69);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            18 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = r38 * r44;
    r58 = r7 * r67;
    r69 = r69 * r55;
    r55 = r28 * r38;
    r55 = r55 * r41;
    r55 = r55 * r88;
    r55 = fma(r58, r55, r58 * r69);
    WriteSum2<double, double>((double*)inout_shared, r1, r55);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            20 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r63 * r60;
    r69 = r71 * r47;
    r69 = r69 * r63;
    r69 = r69 * r41;
    WriteSum2<double, double>((double*)inout_shared, r1, r69);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            22 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = r43 * r77;
    r69 = r69 * r104;
    r69 = r69 * r47;
    r69 = r69 * r63;
    r69 = r69 * r105;
    r69 = fma(r0, r69, r63 * r126);
    r1 = r28 * r77;
    r1 = r1 * r104;
    r1 = r1 * r53;
    r1 = r1 * r47;
    r1 = r1 * r63;
    r1 = r1 * r41;
    r1 = fma(r105, r1, r63 * r59);
    WriteSum2<double, double>((double*)inout_shared, r69, r1);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            24 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r55, r95);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            26 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r95 = r68 * r60;
    r55 = r71 * r47;
    r55 = r55 * r68;
    r55 = r55 * r41;
    WriteSum2<double, double>((double*)inout_shared, r95, r55);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            28 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = r10 * r52;
    r55 = r55 * r47;
    r55 = r55 * r54;
    r95 = r10 * r54;
    r95 = fma(r78, r95, r44 * r55);
    r55 = r43 * r77;
    r55 = r55 * r104;
    r55 = r55 * r47;
    r55 = r55 * r68;
    r55 = r55 * r105;
    r55 = fma(r0, r55, r68 * r126);
    WriteSum2<double, double>((double*)inout_shared, r95, r55);
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
    r95 = r43 * r77;
    r95 = r95 * r104;
    r95 = r95 * r47;
    r95 = r95 * r105;
    r95 = r95 * r67;
    r95 = fma(r0, r95, r67 * r126);
    WriteSum2<double, double>((double*)inout_shared, r95, r55);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            32 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r7 * r6;
    r55 = r28 * r77;
    r55 = r55 * r104;
    r55 = r55 * r53;
    r55 = r55 * r47;
    r55 = r55 * r68;
    r55 = r55 * r41;
    r55 = fma(r105, r55, r68 * r59);
    WriteSum2<double, double>((double*)inout_shared, r6, r55);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            34 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = r52 * r7;
    r55 = r55 * r44;
    r6 = r28 * r77;
    r6 = r6 * r104;
    r6 = r6 * r53;
    r6 = r6 * r47;
    r6 = r6 * r41;
    r6 = r6 * r105;
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
    r55 = r55 * r41;
    r45 = fma(r51, r45, r51 * r128);
    WriteSum2<double, double>((double*)inout_shared, r55, r45);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            38 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = r67 * r60;
    r55 = r71 * r47;
    r55 = r55 * r41;
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
    r47 = r47 * r41;
    r47 = r47 * r58;
    WriteSum2<double, double>((double*)inout_shared, r60, r47);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_tril,
                            42 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void ThinPrismFisheyeSplitFixedPrincipalPointFixedPointResJacFirst(
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
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeSplitFixedPrincipalPointFixedPointResJacFirstKernel<<<
      n_blocks,
      1024>>>(pose,
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
              problem_size);
}

}  // namespace caspar