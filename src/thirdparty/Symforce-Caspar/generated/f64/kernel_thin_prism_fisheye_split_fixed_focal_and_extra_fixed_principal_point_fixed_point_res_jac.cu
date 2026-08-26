#include "kernel_thin_prism_fisheye_split_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJacKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102, r103, r104,
      r105, r106, r107, r108, r109, r110, r111, r112, r113, r114, r115, r116,
      r117, r118, r119, r120, r121, r122, r123, r124, r125, r126;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(principal_point,
                                            0 * principal_point_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1);
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
    r45 = r28 * r28;
    r43 = 1.00000000000000008e-15;
    r47 = r34 * r24;
    r47 = r47 * r24;
    r48 = r35 + r47;
    r48 = r48 + r44;
    r48 = fma(r8, r48, r6);
    r6 = r27 * r34;
    r6 = fma(r29, r6, r20);
    r20 = r10 * r27;
    r20 = r20 * r19;
    r44 = r10 * r24;
    r44 = fma(r29, r44, r20);
    r49 = r17 * r13;
    r49 = r49 * r10;
    r50 = r18 * r14;
    r50 = fma(r10, r50, r49);
    r51 = r13 * r14;
    r51 = fma(r34, r51, r31);
    r31 = r18 * r18;
    r31 = r31 * r34;
    r36 = r31 + r36;
    r48 = fma(r9, r6, r48);
    r48 = fma(r41, r44, r48);
    r48 = fma(r38, r50, r48);
    r48 = fma(r30, r51, r48);
    r48 = fma(r7, r36, r48);
    r44 = r48 * r48;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r6);
    r52 = r34 * r24;
    r52 = fma(r29, r52, r20);
    r52 = fma(r8, r52, r6);
    r6 = r18 * r14;
    r6 = fma(r34, r6, r49);
    r31 = r35 + r31;
    r31 = r31 + r33;
    r33 = r17 * r14;
    r33 = fma(r10, r33, r39);
    r39 = r10 * r19;
    r39 = fma(r29, r39, r42);
    r47 = r35 + r47;
    r47 = r47 + r46;
    r52 = fma(r7, r6, r52);
    r52 = fma(r38, r31, r52);
    r52 = fma(r30, r33, r52);
    r52 = fma(r9, r39, r52);
    r52 = fma(r41, r47, r52);
    r47 = copysign(1.0, r52);
    r47 = fma(r43, r47, r52);
    r52 = r47 * r47;
    r39 = 1.0 / r52;
    r30 = r28 * r28;
    r30 = fma(r39, r30, r39 * r44);
    r44 = sqrt(r30);
    r38 = copysign(1.0, r44);
    r38 = fma(r43, r38, r44);
    r43 = r38 * r38;
    r7 = 1.0 / r43;
    r44 = atan(r44);
    r46 = r44 * r39;
    r42 = r44 * r46;
    r45 = r45 * r7;
    r45 = r45 * r42;
    r49 = 3.00000000000000000e+00;
    r20 = r49 * r42;
    r53 = r48 * r7;
    r54 = r48 * r53;
    r20 = fma(r54, r20, r45);
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            8 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r55,
                                            r56);
    r57 = r42 * r54;
    r45 = r45 + r57;
    r20 = fma(r55, r45, r5 * r20);
    r58 = r4 * r28;
    r59 = r10 * r42;
    r58 = r58 * r53;
    r20 = fma(r59, r58, r20);
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            2 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r60,
                                            r61);
    r62 = r45 * r45;
    r63 = fma(r61, r62, r60 * r45);
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            6 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r64,
                                            r65);
    r66 = r45 * r62;
    r65 = r65 * r66;
    r63 = fma(r45, r65, r63);
    r63 = fma(r64, r66, r63);
    r66 = 1.0 / r47;
    r67 = 1.0 / r38;
    r68 = r66 * r67;
    r69 = r44 * r68;
    r70 = r63 * r69;
    r20 = fma(r48, r70, r20);
    r20 = fma(r48, r69, r20);
    r20 = fma(r2, r20, r0);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r0, r58);
    r20 = fma(r0, r21, r20);
    r0 = r28 * r28;
    r0 = r0 * r49;
    r0 = r0 * r7;
    r0 = fma(r42, r0, r57);
    r0 = fma(r56, r45, r4 * r0);
    r57 = r5 * r28;
    r57 = r57 * r53;
    r0 = fma(r59, r57, r0);
    r0 = fma(r28, r70, r0);
    r0 = fma(r28, r69, r0);
    r0 = fma(r3, r0, r1);
    r0 = fma(r58, r21, r0);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r20, r0);
    r58 = r3 * r21;
    r1 = r28 * r7;
    r57 = -5.00000000000000000e-01;
    r71 = rsqrt(r30);
    r72 = r10 * r48;
    r73 = r11 * r14;
    r74 = r16 * r17;
    r74 = fma(r57, r74, r57 * r73);
    r73 = r15 * r18;
    r74 = fma(r57, r73, r74);
    r75 = r12 * r13;
    r76 = 5.00000000000000000e-01;
    r74 = fma(r76, r75, r74);
    r75 = r24 * r74;
    r73 = r15 * r14;
    r77 = r12 * r17;
    r77 = fma(r76, r77, r76 * r73);
    r73 = r11 * r18;
    r77 = fma(r57, r73, r77);
    r78 = r16 * r13;
    r77 = fma(r76, r78, r77);
    r78 = r29 * r77;
    r73 = fma(r10, r78, r10 * r75);
    r79 = r10 * r19;
    r80 = fma(r76, r26, r57 * r22);
    r80 = fma(r57, r23, r80);
    r80 = fma(r57, r25, r80);
    r81 = r10 * r27;
    r82 = r16 * r14;
    r83 = r11 * r17;
    r83 = fma(r57, r83, r76 * r82);
    r82 = r12 * r18;
    r83 = fma(r57, r82, r83);
    r84 = r15 * r13;
    r83 = fma(r57, r84, r83);
    r81 = r81 * r83;
    r79 = fma(r80, r79, r81);
    r73 = r73 + r79;
    r84 = r10 * r24;
    r84 = r84 * r83;
    r82 = r10 * r19;
    r82 = r82 * r77;
    r85 = r84 + r82;
    r86 = r27 * r34;
    r85 = fma(r74, r86, r85);
    r87 = r34 * r29;
    r85 = fma(r80, r87, r85);
    r85 = fma(r9, r85, r41 * r73);
    r73 = r24 * r77;
    r87 = -4.00000000000000000e+00;
    r73 = r73 * r87;
    r86 = r27 * r80;
    r88 = r87 * r86;
    r89 = r73 + r88;
    r85 = fma(r8, r89, r85);
    r72 = r72 * r85;
    r89 = r48 * r48;
    r90 = r10 * r24;
    r90 = r90 * r80;
    r91 = r10 * r27;
    r91 = fma(r77, r91, r90);
    r77 = r10 * r19;
    r77 = r77 * r74;
    r92 = r10 * r29;
    r92 = r92 * r83;
    r93 = r77 + r92;
    r94 = r91 + r93;
    r78 = fma(r34, r78, r34 * r75);
    r78 = r78 + r79;
    r78 = fma(r8, r78, r9 * r94);
    r94 = r19 * r83;
    r94 = r94 * r87;
    r73 = r73 + r94;
    r78 = fma(r41, r73, r78);
    r52 = r47 * r52;
    r52 = 1.0 / r52;
    r47 = r34 * r52;
    r89 = r89 * r78;
    r89 = fma(r47, r89, r39 * r72);
    r72 = r28 * r28;
    r72 = r72 * r47;
    r73 = r10 * r28;
    r95 = r19 * r34;
    r96 = r34 * r29;
    r96 = r96 * r83;
    r95 = fma(r74, r95, r96);
    r95 = r95 + r91;
    r88 = r94 + r88;
    r88 = fma(r9, r88, r41 * r95);
    r95 = r10 * r29;
    r95 = fma(r80, r95, r82);
    r82 = r10 * r27;
    r82 = fma(r74, r82, r84);
    r95 = r95 + r82;
    r88 = fma(r8, r95, r88);
    r73 = r73 * r88;
    r89 = fma(r39, r73, r89);
    r89 = fma(r78, r72, r89);
    r1 = r1 * r44;
    r1 = r1 * r66;
    r1 = r1 * r57;
    r1 = r1 * r71;
    r1 = r1 * r89;
    r73 = r44 * r44;
    r95 = r7 * r73;
    r95 = r95 * r72;
    r84 = r28 * r71;
    r94 = r89 * r84;
    r30 = r35 + r30;
    r30 = 1.0 / r30;
    r35 = r30 * r46;
    r91 = r28 * r7;
    r94 = r94 * r35;
    r94 = fma(r91, r94, r78 * r95);
    r97 = r21 * r28;
    r43 = r38 * r43;
    r43 = 1.0 / r43;
    r38 = r43 * r42;
    r38 = r38 * r84;
    r97 = r97 * r89;
    r94 = fma(r38, r97, r94);
    r98 = r59 * r91;
    r94 = fma(r88, r98, r94);
    r97 = r89 * r71;
    r97 = r97 * r35;
    r99 = r21 * r48;
    r99 = r99 * r48;
    r99 = r99 * r89;
    r99 = r99 * r71;
    r99 = r99 * r43;
    r99 = fma(r42, r99, r54 * r97);
    r97 = r85 * r53;
    r99 = fma(r59, r97, r99);
    r100 = r78 * r47;
    r73 = r54 * r73;
    r99 = fma(r73, r100, r99);
    r100 = r94 + r99;
    r97 = fma(r56, r100, r1);
    r101 = r28 * r28;
    r102 = -6.00000000000000000e+00;
    r101 = r101 * r44;
    r101 = r101 * r44;
    r101 = r101 * r78;
    r101 = r101 * r102;
    r101 = r101 * r7;
    r103 = r49 * r89;
    r103 = r103 * r84;
    r103 = r103 * r35;
    r103 = fma(r91, r103, r52 * r101);
    r101 = -3.00000000000000000e+00;
    r104 = r28 * r101;
    r104 = r104 * r89;
    r103 = fma(r38, r104, r103);
    r105 = r28 * r88;
    r106 = 6.00000000000000000e+00;
    r105 = r105 * r106;
    r105 = r105 * r7;
    r103 = fma(r42, r105, r103);
    r103 = r103 + r99;
    r99 = r85 * r98;
    r105 = r21 * r28;
    r105 = r105 * r78;
    r105 = r105 * r67;
    r97 = fma(r46, r105, r97);
    r104 = r5 * r34;
    r104 = r104 * r48;
    r104 = r104 * r38;
    r107 = r76 * r89;
    r107 = r107 * r30;
    r107 = r107 * r84;
    r97 = fma(r68, r107, r97);
    r108 = r5 * r88;
    r108 = r108 * r53;
    r97 = fma(r59, r108, r97);
    r109 = r5 * r28;
    r109 = r109 * r44;
    r109 = r109 * r44;
    r109 = r109 * r87;
    r109 = r109 * r78;
    r109 = r109 * r52;
    r97 = fma(r53, r109, r97);
    r110 = r61 * r10;
    r110 = r110 * r45;
    r110 = fma(r60, r100, r100 * r110);
    r64 = r64 * r49;
    r64 = r64 * r62;
    r62 = 4.00000000000000000e+00;
    r65 = r62 * r65;
    r110 = fma(r100, r64, r110);
    r110 = fma(r100, r65, r110);
    r62 = r28 * r110;
    r97 = fma(r69, r62, r97);
    r111 = r5 * r89;
    r112 = r10 * r53;
    r112 = r112 * r84;
    r112 = r112 * r35;
    r97 = fma(r112, r111, r97);
    r113 = r76 * r63;
    r113 = r113 * r89;
    r113 = r113 * r30;
    r113 = r113 * r84;
    r97 = fma(r68, r113, r97);
    r114 = r21 * r28;
    r114 = r114 * r63;
    r114 = r114 * r78;
    r114 = r114 * r67;
    r97 = fma(r46, r114, r97);
    r97 = fma(r4, r103, r97);
    r97 = fma(r5, r99, r97);
    r97 = fma(r63, r1, r97);
    r97 = fma(r89, r104, r97);
    r97 = fma(r88, r70, r97);
    r97 = fma(r88, r69, r97);
    r58 = r58 * r0;
    r114 = r2 * r21;
    r114 = r114 * r20;
    r20 = r48 * r7;
    r20 = r20 * r44;
    r20 = r20 * r66;
    r20 = r20 * r57;
    r20 = r20 * r71;
    r20 = r20 * r89;
    r100 = fma(r55, r100, r20);
    r113 = r49 * r89;
    r113 = r113 * r71;
    r113 = r113 * r35;
    r111 = r48 * r48;
    r111 = r111 * r101;
    r111 = r111 * r89;
    r111 = r111 * r71;
    r111 = r111 * r43;
    r111 = fma(r42, r111, r54 * r113);
    r113 = r85 * r106;
    r113 = r113 * r42;
    r111 = fma(r53, r113, r111);
    r62 = r102 * r52;
    r62 = r62 * r73;
    r111 = fma(r78, r62, r111);
    r111 = r111 + r94;
    r94 = r21 * r48;
    r94 = r94 * r63;
    r94 = r94 * r78;
    r94 = r94 * r67;
    r100 = fma(r46, r94, r100);
    r113 = r48 * r76;
    r113 = r113 * r71;
    r113 = r113 * r30;
    r113 = r113 * r68;
    r109 = r63 * r113;
    r108 = r4 * r34;
    r108 = r108 * r48;
    r108 = r108 * r89;
    r100 = fma(r38, r108, r100);
    r107 = r4 * r88;
    r107 = r107 * r53;
    r100 = fma(r59, r107, r100);
    r1 = r4 * r28;
    r1 = r1 * r44;
    r1 = r1 * r44;
    r1 = r1 * r87;
    r1 = r1 * r52;
    r1 = r1 * r53;
    r105 = r4 * r89;
    r100 = fma(r112, r105, r100);
    r103 = r48 * r110;
    r100 = fma(r69, r103, r100);
    r115 = r21 * r48;
    r115 = r115 * r78;
    r115 = r115 * r67;
    r100 = fma(r46, r115, r100);
    r100 = fma(r5, r111, r100);
    r100 = fma(r4, r99, r100);
    r100 = fma(r63, r20, r100);
    r100 = fma(r89, r109, r100);
    r100 = fma(r85, r70, r100);
    r100 = fma(r78, r1, r100);
    r100 = fma(r89, r113, r100);
    r100 = fma(r85, r69, r100);
    r58 = fma(r100, r114, r97 * r58);
    r115 = r3 * r21;
    r103 = r28 * r106;
    r105 = r10 * r24;
    r107 = r11 * r14;
    r108 = r16 * r17;
    r108 = fma(r76, r108, r76 * r107);
    r107 = r15 * r18;
    r108 = fma(r76, r107, r108);
    r94 = r12 * r13;
    r108 = fma(r57, r94, r108);
    r105 = r105 * r108;
    r81 = r81 + r105;
    r94 = r19 * r34;
    r81 = fma(r80, r94, r81);
    r107 = r34 * r29;
    r20 = r15 * r14;
    r99 = r12 * r17;
    r99 = fma(r57, r99, r57 * r20);
    r20 = r11 * r18;
    r99 = fma(r76, r20, r99);
    r111 = r16 * r13;
    r99 = fma(r57, r111, r99);
    r81 = fma(r99, r107, r81);
    r107 = r10 * r29;
    r107 = fma(r10, r86, r108 * r107);
    r94 = r10 * r19;
    r94 = r94 * r83;
    r111 = r10 * r24;
    r111 = fma(r99, r111, r94);
    r107 = r107 + r111;
    r107 = fma(r8, r107, r41 * r81);
    r81 = r27 * r87;
    r81 = r81 * r108;
    r20 = r19 * r99;
    r116 = r87 * r20;
    r117 = r81 + r116;
    r107 = fma(r9, r117, r107);
    r103 = r103 * r107;
    r103 = r103 * r7;
    r117 = r28 * r38;
    r118 = r10 * r48;
    r92 = r90 + r92;
    r90 = r10 * r27;
    r90 = r90 * r99;
    r119 = r10 * r19;
    r119 = fma(r108, r119, r90);
    r92 = r92 + r119;
    r120 = r24 * r83;
    r120 = r120 * r87;
    r81 = r120 + r81;
    r81 = fma(r8, r81, r41 * r92);
    r92 = r34 * r29;
    r86 = fma(r34, r86, r108 * r92);
    r86 = r86 + r111;
    r81 = fma(r9, r86, r81);
    r118 = r118 * r81;
    r86 = r10 * r28;
    r86 = r86 * r107;
    r86 = fma(r39, r86, r39 * r118);
    r118 = r34 * r24;
    r118 = fma(r80, r118, r96);
    r118 = r118 + r119;
    r119 = r10 * r29;
    r119 = fma(r99, r119, r105);
    r119 = r119 + r79;
    r119 = fma(r9, r119, r8 * r118);
    r116 = r120 + r116;
    r119 = fma(r41, r116, r119);
    r116 = r48 * r48;
    r116 = r116 * r119;
    r86 = fma(r47, r116, r86);
    r86 = fma(r119, r72, r86);
    r116 = r101 * r86;
    r117 = fma(r116, r117, r42 * r103);
    r103 = r28 * r28;
    r103 = r103 * r44;
    r103 = r103 * r44;
    r103 = r103 * r102;
    r103 = r103 * r119;
    r103 = r103 * r7;
    r117 = fma(r52, r103, r117);
    r120 = r49 * r86;
    r120 = r120 * r84;
    r120 = r120 * r35;
    r117 = fma(r91, r120, r117);
    r118 = r119 * r47;
    r79 = r81 * r53;
    r79 = fma(r59, r79, r73 * r118);
    r118 = r21 * r48;
    r118 = r118 * r48;
    r118 = r118 * r86;
    r118 = r118 * r71;
    r118 = r118 * r43;
    r79 = fma(r42, r118, r79);
    r105 = r86 * r71;
    r105 = r105 * r35;
    r79 = fma(r54, r105, r79);
    r117 = r117 + r79;
    r120 = r21 * r28;
    r120 = r120 * r86;
    r120 = fma(r38, r120, r107 * r98);
    r103 = r86 * r84;
    r103 = r103 * r35;
    r120 = fma(r91, r103, r120);
    r120 = fma(r119, r95, r120);
    r79 = r79 + r120;
    r117 = fma(r56, r79, r4 * r117);
    r103 = r44 * r57;
    r103 = r103 * r86;
    r103 = r103 * r7;
    r103 = r103 * r66;
    r117 = fma(r84, r103, r117);
    r105 = r61 * r10;
    r105 = r105 * r45;
    r105 = fma(r79, r105, r60 * r79);
    r105 = fma(r79, r65, r105);
    r105 = fma(r79, r64, r105);
    r118 = r28 * r105;
    r117 = fma(r69, r118, r117);
    r80 = r76 * r86;
    r80 = r80 * r30;
    r80 = r80 * r84;
    r117 = fma(r68, r80, r117);
    r92 = r5 * r28;
    r92 = r92 * r44;
    r92 = r92 * r44;
    r92 = r92 * r87;
    r92 = r92 * r119;
    r92 = r92 * r52;
    r117 = fma(r53, r92, r117);
    r108 = r5 * r86;
    r117 = fma(r112, r108, r117);
    r121 = r21 * r28;
    r121 = r121 * r63;
    r121 = r121 * r119;
    r121 = r121 * r67;
    r117 = fma(r46, r121, r117);
    r122 = r44 * r63;
    r122 = r122 * r57;
    r122 = r122 * r86;
    r122 = r122 * r7;
    r122 = r122 * r66;
    r117 = fma(r84, r122, r117);
    r123 = r21 * r28;
    r123 = r123 * r119;
    r123 = r123 * r67;
    r117 = fma(r46, r123, r117);
    r124 = r5 * r107;
    r124 = r124 * r53;
    r117 = fma(r59, r124, r117);
    r125 = r76 * r63;
    r125 = r125 * r86;
    r125 = r125 * r30;
    r125 = r125 * r84;
    r117 = fma(r68, r125, r117);
    r126 = r5 * r81;
    r117 = fma(r98, r126, r117);
    r117 = fma(r107, r70, r117);
    r117 = fma(r86, r104, r117);
    r117 = fma(r107, r69, r117);
    r115 = r115 * r0;
    r126 = r106 * r81;
    r126 = r126 * r42;
    r126 = fma(r53, r126, r119 * r62);
    r125 = r48 * r48;
    r125 = r125 * r71;
    r125 = r125 * r43;
    r125 = r125 * r42;
    r126 = fma(r116, r125, r126);
    r116 = r49 * r86;
    r116 = r116 * r71;
    r116 = r116 * r35;
    r126 = fma(r54, r116, r126);
    r126 = r126 + r120;
    r79 = fma(r55, r79, r5 * r126);
    r126 = r21 * r48;
    r126 = r126 * r119;
    r126 = r126 * r67;
    r79 = fma(r46, r126, r79);
    r120 = r44 * r63;
    r120 = r120 * r57;
    r120 = r120 * r86;
    r120 = r120 * r66;
    r120 = r120 * r71;
    r79 = fma(r53, r120, r79);
    r116 = r48 * r105;
    r79 = fma(r69, r116, r79);
    r125 = r21 * r48;
    r125 = r125 * r63;
    r125 = r125 * r119;
    r125 = r125 * r67;
    r79 = fma(r46, r125, r79);
    r124 = r4 * r86;
    r79 = fma(r112, r124, r79);
    r123 = r4 * r34;
    r123 = r123 * r48;
    r123 = r123 * r86;
    r79 = fma(r38, r123, r79);
    r122 = r4 * r107;
    r122 = r122 * r53;
    r79 = fma(r59, r122, r79);
    r121 = r4 * r81;
    r79 = fma(r98, r121, r79);
    r108 = r44 * r57;
    r108 = r108 * r86;
    r108 = r108 * r66;
    r108 = r108 * r71;
    r79 = fma(r53, r108, r79);
    r79 = fma(r86, r109, r79);
    r79 = fma(r86, r113, r79);
    r79 = fma(r119, r1, r79);
    r79 = fma(r81, r69, r79);
    r79 = fma(r81, r70, r79);
    r115 = fma(r79, r114, r117 * r115);
    WriteSum2<double, double>((double*)inout_shared, r58, r115);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r115 = r3 * r21;
    r58 = r19 * r87;
    r26 = fma(r57, r26, r76 * r22);
    r26 = fma(r76, r23, r26);
    r26 = fma(r76, r25, r26);
    r58 = r58 * r26;
    r75 = r87 * r75;
    r25 = r58 + r75;
    r23 = r10 * r27;
    r23 = r23 * r26;
    r94 = r94 + r23;
    r22 = r34 * r24;
    r94 = fma(r99, r22, r94);
    r108 = r34 * r29;
    r94 = fma(r74, r108, r94);
    r94 = fma(r8, r94, r41 * r25);
    r25 = r10 * r29;
    r25 = fma(r10, r20, r26 * r25);
    r25 = r25 + r82;
    r94 = fma(r9, r25, r94);
    r25 = r10 * r28;
    r108 = r10 * r24;
    r108 = r108 * r26;
    r90 = r90 + r108;
    r90 = r90 + r93;
    r93 = r34 * r29;
    r20 = fma(r34, r20, r26 * r93);
    r20 = r20 + r82;
    r20 = fma(r41, r20, r8 * r90);
    r83 = r27 * r83;
    r83 = r83 * r87;
    r58 = r58 + r83;
    r20 = fma(r9, r58, r20);
    r25 = r25 * r20;
    r58 = r48 * r48;
    r58 = r58 * r94;
    r58 = fma(r47, r58, r39 * r25);
    r25 = r10 * r48;
    r96 = r77 + r96;
    r77 = r27 * r34;
    r96 = fma(r99, r77, r96);
    r96 = r96 + r108;
    r75 = r83 + r75;
    r75 = fma(r8, r75, r9 * r96);
    r8 = r10 * r29;
    r8 = fma(r74, r8, r23);
    r8 = r8 + r111;
    r75 = fma(r41, r8, r75);
    r25 = r25 * r75;
    r58 = fma(r39, r25, r58);
    r58 = fma(r94, r72, r58);
    r25 = r58 * r84;
    r25 = r25 * r35;
    r25 = fma(r91, r25, r94 * r95);
    r8 = r21 * r28;
    r8 = r8 * r58;
    r25 = fma(r38, r8, r25);
    r25 = fma(r20, r98, r25);
    r8 = r21 * r48;
    r8 = r8 * r48;
    r8 = r8 * r58;
    r8 = r8 * r71;
    r8 = r8 * r43;
    r41 = r75 * r53;
    r41 = fma(r59, r41, r42 * r8);
    r8 = r94 * r47;
    r41 = fma(r73, r8, r41);
    r111 = r58 * r71;
    r111 = r111 * r35;
    r41 = fma(r54, r111, r41);
    r111 = r25 + r41;
    r8 = r28 * r28;
    r8 = r8 * r44;
    r8 = r8 * r44;
    r8 = r8 * r102;
    r8 = r8 * r94;
    r8 = r8 * r7;
    r23 = r49 * r58;
    r23 = r23 * r84;
    r23 = r23 * r35;
    r23 = fma(r91, r23, r52 * r8);
    r8 = r28 * r106;
    r8 = r8 * r20;
    r8 = r8 * r7;
    r23 = fma(r42, r8, r23);
    r74 = r28 * r101;
    r74 = r74 * r58;
    r23 = fma(r38, r74, r23);
    r23 = r23 + r41;
    r23 = fma(r4, r23, r56 * r111);
    r41 = r21 * r28;
    r41 = r41 * r63;
    r41 = r41 * r94;
    r41 = r41 * r67;
    r23 = fma(r46, r41, r23);
    r74 = r21 * r28;
    r74 = r74 * r94;
    r74 = r74 * r67;
    r23 = fma(r46, r74, r23);
    r8 = r5 * r75;
    r23 = fma(r98, r8, r23);
    r96 = r44 * r63;
    r96 = r96 * r57;
    r96 = r96 * r58;
    r96 = r96 * r7;
    r96 = r96 * r66;
    r23 = fma(r84, r96, r23);
    r9 = r5 * r58;
    r23 = fma(r112, r9, r23);
    r83 = r76 * r58;
    r83 = r83 * r30;
    r83 = r83 * r84;
    r23 = fma(r68, r83, r23);
    r77 = r61 * r10;
    r77 = r77 * r45;
    r77 = fma(r111, r77, r60 * r111);
    r77 = fma(r111, r64, r77);
    r77 = fma(r111, r65, r77);
    r108 = r28 * r77;
    r23 = fma(r69, r108, r23);
    r99 = r44 * r57;
    r99 = r99 * r58;
    r99 = r99 * r7;
    r99 = r99 * r66;
    r23 = fma(r84, r99, r23);
    r90 = r76 * r63;
    r90 = r90 * r58;
    r90 = r90 * r30;
    r90 = r90 * r84;
    r23 = fma(r68, r90, r23);
    r82 = r5 * r28;
    r82 = r82 * r44;
    r82 = r82 * r44;
    r82 = r82 * r87;
    r82 = r82 * r94;
    r82 = r82 * r52;
    r23 = fma(r53, r82, r23);
    r93 = r5 * r20;
    r93 = r93 * r53;
    r23 = fma(r59, r93, r23);
    r23 = fma(r20, r70, r23);
    r23 = fma(r20, r69, r23);
    r23 = fma(r58, r104, r23);
    r115 = r115 * r0;
    r93 = r48 * r48;
    r93 = r93 * r101;
    r93 = r93 * r58;
    r93 = r93 * r71;
    r93 = r93 * r43;
    r82 = r106 * r75;
    r82 = r82 * r42;
    r82 = fma(r53, r82, r42 * r93);
    r93 = r49 * r58;
    r93 = r93 * r71;
    r93 = r93 * r35;
    r82 = fma(r54, r93, r82);
    r82 = fma(r94, r62, r82);
    r82 = r82 + r25;
    r82 = fma(r5, r82, r55 * r111);
    r111 = r48 * r77;
    r82 = fma(r69, r111, r82);
    r25 = r4 * r75;
    r82 = fma(r98, r25, r82);
    r93 = r4 * r58;
    r82 = fma(r112, r93, r82);
    r90 = r21 * r48;
    r90 = r90 * r63;
    r90 = r90 * r94;
    r90 = r90 * r67;
    r82 = fma(r46, r90, r82);
    r99 = r4 * r34;
    r99 = r99 * r48;
    r99 = r99 * r58;
    r82 = fma(r38, r99, r82);
    r108 = r21 * r48;
    r108 = r108 * r94;
    r108 = r108 * r67;
    r82 = fma(r46, r108, r82);
    r83 = r44 * r57;
    r83 = r83 * r58;
    r83 = r83 * r66;
    r83 = r83 * r71;
    r82 = fma(r53, r83, r82);
    r9 = r4 * r20;
    r9 = r9 * r53;
    r82 = fma(r59, r9, r82);
    r96 = r44 * r63;
    r96 = r96 * r57;
    r96 = r96 * r58;
    r96 = r96 * r66;
    r96 = r96 * r71;
    r82 = fma(r53, r96, r82);
    r82 = fma(r58, r113, r82);
    r82 = fma(r58, r109, r82);
    r82 = fma(r75, r70, r82);
    r82 = fma(r75, r69, r82);
    r82 = fma(r94, r1, r82);
    r115 = fma(r82, r114, r23 * r115);
    r96 = r3 * r21;
    r9 = r32 * r28;
    r9 = r9 * r106;
    r9 = r9 * r7;
    r83 = r28 * r101;
    r108 = r10 * r32;
    r108 = r108 * r28;
    r108 = fma(r6, r72, r39 * r108);
    r99 = r6 * r48;
    r99 = r99 * r48;
    r108 = fma(r47, r99, r108);
    r90 = r10 * r36;
    r90 = r90 * r48;
    r108 = fma(r39, r90, r108);
    r83 = r83 * r108;
    r83 = fma(r38, r83, r42 * r9);
    r9 = r6 * r28;
    r9 = r9 * r28;
    r9 = r9 * r44;
    r9 = r9 * r44;
    r9 = r9 * r102;
    r9 = r9 * r7;
    r83 = fma(r52, r9, r83);
    r90 = r49 * r108;
    r90 = r90 * r84;
    r90 = r90 * r35;
    r83 = fma(r91, r90, r83);
    r99 = r36 * r53;
    r93 = r21 * r48;
    r93 = r93 * r48;
    r93 = r93 * r108;
    r93 = r93 * r71;
    r93 = r93 * r43;
    r93 = fma(r42, r93, r59 * r99);
    r99 = r108 * r71;
    r99 = r99 * r35;
    r93 = fma(r54, r99, r93);
    r25 = r6 * r47;
    r93 = fma(r73, r25, r93);
    r83 = r83 + r93;
    r90 = r21 * r28;
    r90 = r90 * r108;
    r90 = fma(r38, r90, r32 * r98);
    r9 = r108 * r84;
    r9 = r9 * r35;
    r90 = fma(r91, r9, r90);
    r90 = fma(r6, r95, r90);
    r93 = r93 + r90;
    r83 = fma(r56, r93, r4 * r83);
    r9 = r76 * r108;
    r9 = r9 * r30;
    r9 = r9 * r84;
    r83 = fma(r68, r9, r83);
    r25 = r44 * r63;
    r25 = r25 * r57;
    r25 = r25 * r108;
    r25 = r25 * r7;
    r25 = r25 * r66;
    r83 = fma(r84, r25, r83);
    r99 = r21 * r6;
    r99 = r99 * r28;
    r99 = r99 * r63;
    r99 = r99 * r67;
    r83 = fma(r46, r99, r83);
    r111 = r108 * r112;
    r8 = r21 * r6;
    r8 = r8 * r28;
    r8 = r8 * r67;
    r83 = fma(r46, r8, r83);
    r74 = r5 * r6;
    r74 = r74 * r28;
    r74 = r74 * r44;
    r74 = r74 * r44;
    r74 = r74 * r87;
    r74 = r74 * r52;
    r83 = fma(r53, r74, r83);
    r41 = r61 * r10;
    r41 = r41 * r45;
    r41 = fma(r60, r93, r93 * r41);
    r41 = fma(r93, r65, r41);
    r41 = fma(r93, r64, r41);
    r26 = r28 * r41;
    r83 = fma(r69, r26, r83);
    r22 = r76 * r63;
    r22 = r22 * r108;
    r22 = r22 * r30;
    r22 = r22 * r84;
    r83 = fma(r68, r22, r83);
    r121 = r5 * r32;
    r121 = r121 * r53;
    r83 = fma(r59, r121, r83);
    r122 = r44 * r57;
    r122 = r122 * r108;
    r122 = r122 * r7;
    r122 = r122 * r66;
    r83 = fma(r84, r122, r83);
    r123 = r5 * r36;
    r83 = fma(r98, r123, r83);
    r83 = fma(r32, r69, r83);
    r83 = fma(r5, r111, r83);
    r83 = fma(r108, r104, r83);
    r83 = fma(r32, r70, r83);
    r96 = r96 * r0;
    r123 = r36 * r106;
    r123 = r123 * r42;
    r122 = r48 * r48;
    r122 = r122 * r101;
    r122 = r122 * r108;
    r122 = r122 * r71;
    r122 = r122 * r43;
    r122 = fma(r42, r122, r53 * r123);
    r123 = r49 * r108;
    r123 = r123 * r71;
    r123 = r123 * r35;
    r122 = fma(r54, r123, r122);
    r122 = fma(r6, r62, r122);
    r122 = r122 + r90;
    r93 = fma(r55, r93, r5 * r122);
    r122 = r44 * r63;
    r122 = r122 * r57;
    r122 = r122 * r108;
    r122 = r122 * r66;
    r122 = r122 * r71;
    r93 = fma(r53, r122, r93);
    r90 = r44 * r57;
    r90 = r90 * r108;
    r90 = r90 * r66;
    r90 = r90 * r71;
    r93 = fma(r53, r90, r93);
    r123 = r4 * r34;
    r123 = r123 * r48;
    r123 = r123 * r108;
    r93 = fma(r38, r123, r93);
    r121 = r48 * r41;
    r93 = fma(r69, r121, r93);
    r22 = r21 * r6;
    r22 = r22 * r48;
    r22 = r22 * r63;
    r22 = r22 * r67;
    r93 = fma(r46, r22, r93);
    r26 = r21 * r6;
    r26 = r26 * r48;
    r26 = r26 * r67;
    r93 = fma(r46, r26, r93);
    r74 = r4 * r32;
    r74 = r74 * r53;
    r93 = fma(r59, r74, r93);
    r8 = r4 * r36;
    r93 = fma(r98, r8, r93);
    r93 = fma(r36, r70, r93);
    r93 = fma(r108, r109, r93);
    r93 = fma(r4, r111, r93);
    r93 = fma(r108, r113, r93);
    r93 = fma(r6, r1, r93);
    r93 = fma(r36, r69, r93);
    r96 = fma(r93, r114, r83 * r96);
    WriteSum2<double, double>((double*)inout_shared, r115, r96);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r96 = r3 * r21;
    r115 = fma(r37, r98, r33 * r95);
    r8 = r10 * r37;
    r8 = r8 * r28;
    r8 = fma(r39, r8, r33 * r72);
    r74 = r33 * r48;
    r74 = r74 * r48;
    r8 = fma(r47, r74, r8);
    r26 = r10 * r51;
    r26 = r26 * r48;
    r8 = fma(r39, r26, r8);
    r26 = r8 * r84;
    r26 = r26 * r35;
    r115 = fma(r91, r26, r115);
    r74 = r21 * r28;
    r74 = r74 * r8;
    r115 = fma(r38, r74, r115);
    r74 = r8 * r71;
    r74 = r74 * r35;
    r26 = r33 * r47;
    r26 = fma(r73, r26, r54 * r74);
    r74 = r21 * r48;
    r74 = r74 * r48;
    r74 = r74 * r8;
    r74 = r74 * r71;
    r74 = r74 * r43;
    r26 = fma(r42, r74, r26);
    r22 = r51 * r53;
    r26 = fma(r59, r22, r26);
    r22 = r115 + r26;
    r74 = r33 * r28;
    r74 = r74 * r28;
    r74 = r74 * r44;
    r74 = r74 * r44;
    r74 = r74 * r102;
    r74 = r74 * r7;
    r121 = r37 * r28;
    r121 = r121 * r106;
    r121 = r121 * r7;
    r121 = fma(r42, r121, r52 * r74);
    r74 = r49 * r8;
    r74 = r74 * r84;
    r74 = r74 * r35;
    r121 = fma(r91, r74, r121);
    r123 = r28 * r101;
    r123 = r123 * r8;
    r121 = fma(r38, r123, r121);
    r121 = r121 + r26;
    r121 = fma(r4, r121, r56 * r22);
    r26 = r21 * r33;
    r26 = r26 * r28;
    r26 = r26 * r67;
    r121 = fma(r46, r26, r121);
    r123 = r5 * r37;
    r123 = r123 * r53;
    r121 = fma(r59, r123, r121);
    r74 = r5 * r8;
    r121 = fma(r112, r74, r121);
    r90 = r76 * r8;
    r90 = r90 * r30;
    r90 = r90 * r84;
    r121 = fma(r68, r90, r121);
    r122 = r61 * r10;
    r122 = r122 * r45;
    r122 = fma(r60, r22, r22 * r122);
    r122 = fma(r22, r64, r122);
    r122 = fma(r22, r65, r122);
    r111 = r28 * r122;
    r121 = fma(r69, r111, r121);
    r99 = r44 * r63;
    r99 = r99 * r57;
    r99 = r99 * r8;
    r99 = r99 * r7;
    r99 = r99 * r66;
    r121 = fma(r84, r99, r121);
    r25 = r5 * r33;
    r25 = r25 * r28;
    r25 = r25 * r44;
    r25 = r25 * r44;
    r25 = r25 * r87;
    r25 = r25 * r52;
    r121 = fma(r53, r25, r121);
    r9 = r44 * r57;
    r9 = r9 * r8;
    r9 = r9 * r7;
    r9 = r9 * r66;
    r121 = fma(r84, r9, r121);
    r124 = r76 * r63;
    r124 = r124 * r8;
    r124 = r124 * r30;
    r124 = r124 * r84;
    r121 = fma(r68, r124, r121);
    r125 = r21 * r33;
    r125 = r125 * r28;
    r125 = r125 * r63;
    r125 = r125 * r67;
    r121 = fma(r46, r125, r121);
    r116 = r5 * r51;
    r121 = fma(r98, r116, r121);
    r121 = fma(r37, r70, r121);
    r121 = fma(r8, r104, r121);
    r121 = fma(r37, r69, r121);
    r96 = r96 * r0;
    r116 = r49 * r8;
    r116 = r116 * r71;
    r116 = r116 * r35;
    r116 = fma(r33, r62, r54 * r116);
    r125 = r48 * r48;
    r125 = r125 * r101;
    r125 = r125 * r8;
    r125 = r125 * r71;
    r125 = r125 * r43;
    r116 = fma(r42, r125, r116);
    r124 = r51 * r106;
    r124 = r124 * r42;
    r116 = fma(r53, r124, r116);
    r116 = r116 + r115;
    r22 = fma(r55, r22, r5 * r116);
    r116 = r4 * r37;
    r116 = r116 * r53;
    r22 = fma(r59, r116, r22);
    r115 = r4 * r8;
    r22 = fma(r112, r115, r22);
    r124 = r21 * r33;
    r124 = r124 * r48;
    r124 = r124 * r67;
    r22 = fma(r46, r124, r22);
    r125 = r4 * r34;
    r125 = r125 * r48;
    r125 = r125 * r8;
    r22 = fma(r38, r125, r22);
    r9 = r44 * r57;
    r9 = r9 * r8;
    r9 = r9 * r66;
    r9 = r9 * r71;
    r22 = fma(r53, r9, r22);
    r25 = r48 * r122;
    r22 = fma(r69, r25, r22);
    r99 = r44 * r63;
    r99 = r99 * r57;
    r99 = r99 * r8;
    r99 = r99 * r66;
    r99 = r99 * r71;
    r22 = fma(r53, r99, r22);
    r111 = r21 * r33;
    r111 = r111 * r48;
    r111 = r111 * r63;
    r111 = r111 * r67;
    r22 = fma(r46, r111, r22);
    r90 = r4 * r51;
    r22 = fma(r98, r90, r22);
    r22 = fma(r8, r109, r22);
    r22 = fma(r51, r69, r22);
    r22 = fma(r33, r1, r22);
    r22 = fma(r8, r113, r22);
    r22 = fma(r51, r70, r22);
    r96 = fma(r22, r114, r121 * r96);
    r90 = r3 * r21;
    r95 = fma(r40, r98, r31 * r95);
    r111 = r10 * r40;
    r111 = r111 * r28;
    r72 = fma(r31, r72, r39 * r111);
    r111 = r10 * r50;
    r111 = r111 * r48;
    r72 = fma(r39, r111, r72);
    r39 = r31 * r48;
    r39 = r39 * r48;
    r72 = fma(r47, r39, r72);
    r39 = r72 * r84;
    r39 = r39 * r35;
    r95 = fma(r91, r39, r95);
    r111 = r21 * r28;
    r111 = r111 * r72;
    r95 = fma(r38, r111, r95);
    r111 = r21 * r48;
    r111 = r111 * r48;
    r111 = r111 * r72;
    r111 = r111 * r71;
    r111 = r111 * r43;
    r39 = r72 * r71;
    r39 = r39 * r35;
    r39 = fma(r54, r39, r42 * r111);
    r111 = r50 * r53;
    r39 = fma(r59, r111, r39);
    r99 = r31 * r47;
    r39 = fma(r73, r99, r39);
    r99 = r95 + r39;
    r111 = r31 * r28;
    r111 = r111 * r28;
    r111 = r111 * r44;
    r111 = r111 * r44;
    r111 = r111 * r102;
    r111 = r111 * r7;
    r102 = r40 * r28;
    r102 = r102 * r106;
    r102 = r102 * r7;
    r102 = fma(r42, r102, r52 * r111);
    r111 = r49 * r72;
    r111 = r111 * r84;
    r111 = r111 * r35;
    r102 = fma(r91, r111, r102);
    r91 = r28 * r101;
    r91 = r91 * r72;
    r102 = fma(r38, r91, r102);
    r102 = r102 + r39;
    r102 = fma(r4, r102, r56 * r99);
    r56 = r44 * r57;
    r56 = r56 * r72;
    r56 = r56 * r7;
    r56 = r56 * r66;
    r102 = fma(r84, r56, r102);
    r39 = r21 * r31;
    r39 = r39 * r28;
    r39 = r39 * r67;
    r102 = fma(r46, r39, r102);
    r91 = r5 * r50;
    r102 = fma(r98, r91, r102);
    r111 = r21 * r31;
    r111 = r111 * r28;
    r111 = r111 * r63;
    r111 = r111 * r67;
    r102 = fma(r46, r111, r102);
    r73 = r76 * r63;
    r73 = r73 * r72;
    r73 = r73 * r30;
    r73 = r73 * r84;
    r102 = fma(r68, r73, r102);
    r25 = r5 * r31;
    r25 = r25 * r28;
    r25 = r25 * r44;
    r25 = r25 * r44;
    r25 = r25 * r87;
    r25 = r25 * r52;
    r102 = fma(r53, r25, r102);
    r52 = r5 * r40;
    r52 = r52 * r53;
    r102 = fma(r59, r52, r102);
    r87 = r44 * r63;
    r87 = r87 * r57;
    r87 = r87 * r72;
    r87 = r87 * r7;
    r87 = r87 * r66;
    r102 = fma(r84, r87, r102);
    r7 = r5 * r72;
    r102 = fma(r112, r7, r102);
    r9 = r76 * r72;
    r9 = r9 * r30;
    r9 = r9 * r84;
    r102 = fma(r68, r9, r102);
    r68 = r61 * r10;
    r68 = r68 * r45;
    r68 = fma(r99, r68, r60 * r99);
    r68 = fma(r99, r65, r68);
    r68 = fma(r99, r64, r68);
    r64 = r28 * r68;
    r102 = fma(r69, r64, r102);
    r102 = fma(r40, r69, r102);
    r102 = fma(r40, r70, r102);
    r102 = fma(r72, r104, r102);
    r90 = r90 * r0;
    r0 = r48 * r48;
    r0 = r0 * r101;
    r0 = r0 * r72;
    r0 = r0 * r71;
    r0 = r0 * r43;
    r43 = r49 * r72;
    r43 = r43 * r71;
    r43 = r43 * r35;
    r43 = fma(r54, r43, r42 * r0);
    r0 = r50 * r106;
    r0 = r0 * r42;
    r43 = fma(r53, r0, r43);
    r43 = fma(r31, r62, r43);
    r43 = r43 + r95;
    r99 = fma(r55, r99, r5 * r43);
    r55 = r4 * r50;
    r99 = fma(r98, r55, r99);
    r98 = r4 * r40;
    r98 = r98 * r53;
    r99 = fma(r59, r98, r99);
    r59 = r48 * r68;
    r99 = fma(r69, r59, r99);
    r43 = r4 * r72;
    r99 = fma(r112, r43, r99);
    r112 = r21 * r31;
    r112 = r112 * r48;
    r112 = r112 * r67;
    r99 = fma(r46, r112, r99);
    r95 = r44 * r63;
    r95 = r95 * r57;
    r95 = r95 * r72;
    r95 = r95 * r66;
    r95 = r95 * r71;
    r99 = fma(r53, r95, r99);
    r62 = r4 * r34;
    r62 = r62 * r48;
    r62 = r62 * r72;
    r99 = fma(r38, r62, r99);
    r0 = r21 * r31;
    r0 = r0 * r48;
    r0 = r0 * r63;
    r0 = r0 * r67;
    r99 = fma(r46, r0, r99);
    r46 = r44 * r57;
    r46 = r46 * r72;
    r46 = r46 * r66;
    r46 = r46 * r71;
    r99 = fma(r53, r46, r99);
    r99 = fma(r50, r70, r99);
    r99 = fma(r31, r1, r99);
    r99 = fma(r72, r109, r99);
    r99 = fma(r72, r113, r99);
    r99 = fma(r50, r69, r99);
    r114 = fma(r99, r114, r102 * r90);
    WriteSum2<double, double>((double*)inout_shared, r96, r114);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r2 * r2;
    r114 = r100 * r2;
    r96 = r3 * r3;
    r90 = r97 * r96;
    r97 = fma(r97, r90, r100 * r114);
    r100 = r79 * r79;
    r46 = r117 * r117;
    r46 = fma(r96, r46, r2 * r100);
    WriteSum2<double, double>((double*)inout_shared, r97, r46);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r82 * r82;
    r97 = r23 * r23;
    r97 = fma(r96, r97, r2 * r46);
    r46 = r83 * r83;
    r100 = r93 * r93;
    r100 = fma(r2, r100, r96 * r46);
    WriteSum2<double, double>((double*)inout_shared, r97, r100);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r100 = r22 * r22;
    r97 = r121 * r121;
    r97 = fma(r96, r97, r2 * r100);
    r100 = r102 * r102;
    r46 = r99 * r99;
    r46 = fma(r2, r46, r96 * r100);
    WriteSum2<double, double>((double*)inout_shared, r97, r46);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fma(r79, r114, r117 * r90);
    r97 = fma(r23, r90, r82 * r114);
    WriteSum2<double, double>((double*)inout_shared, r46, r97);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r97 = fma(r83, r90, r93 * r114);
    r46 = fma(r121, r90, r22 * r114);
    WriteSum2<double, double>((double*)inout_shared, r97, r46);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r114 = fma(r99, r114, r102 * r90);
    r90 = r117 * r23;
    r46 = r79 * r82;
    r46 = fma(r2, r46, r96 * r90);
    WriteSum2<double, double>((double*)inout_shared, r114, r46);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r79 * r93;
    r114 = r117 * r83;
    r114 = fma(r96, r114, r2 * r46);
    r46 = r79 * r22;
    r90 = r117 * r121;
    r90 = fma(r96, r90, r2 * r46);
    WriteSum2<double, double>((double*)inout_shared, r114, r90);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r90 = r117 * r102;
    r114 = r79 * r99;
    r114 = fma(r2, r114, r96 * r90);
    r90 = r23 * r83;
    r46 = r82 * r93;
    r46 = fma(r2, r46, r96 * r90);
    WriteSum2<double, double>((double*)inout_shared, r114, r46);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r82 * r22;
    r114 = r23 * r121;
    r114 = fma(r96, r114, r2 * r46);
    r46 = r23 * r102;
    r90 = r82 * r99;
    r90 = fma(r2, r90, r96 * r46);
    WriteSum2<double, double>((double*)inout_shared, r114, r90);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r90 = r93 * r22;
    r114 = r83 * r121;
    r114 = fma(r96, r114, r2 * r90);
    r90 = r83 * r102;
    r46 = r93 * r99;
    r46 = fma(r2, r46, r96 * r90);
    WriteSum2<double, double>((double*)inout_shared, r114, r46);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r121 * r102;
    r114 = r22 * r99;
    r114 = fma(r2, r114, r96 * r46);
    WriteSum1<double, double>((double*)inout_shared, r114);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
}

void ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeSplitFixedFocalAndExtraFixedPrincipalPointFixedPointResJacKernel<<<
      n_blocks,
      1024>>>(pose,
              pose_num_alloc,
              pose_indices,
              sensor_from_rig,
              sensor_from_rig_num_alloc,
              pixel,
              pixel_num_alloc,
              focal_and_extra,
              focal_and_extra_num_alloc,
              principal_point,
              principal_point_num_alloc,
              point,
              point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_pose_njtr,
              out_pose_njtr_num_alloc,
              out_pose_precond_diag,
              out_pose_precond_diag_num_alloc,
              out_pose_precond_tril,
              out_pose_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar