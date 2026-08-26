#include "kernel_thin_prism_fisheye_split_fixed_focal_and_extra_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedFocalAndExtraFixedPointResJacKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
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
        double* out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        double* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        double* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        double* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102, r103, r104,
      r105, r106, r107, r108, r109, r110, r111, r112, r113, r114, r115, r116,
      r117, r118, r119, r120, r121, r122, r123, r124;
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
    r58 = r48 * r7;
    r1 = -5.00000000000000000e-01;
    r57 = rsqrt(r30);
    r71 = r10 * r48;
    r72 = r11 * r14;
    r73 = r16 * r17;
    r73 = fma(r1, r73, r1 * r72);
    r72 = r15 * r18;
    r73 = fma(r1, r72, r73);
    r74 = r12 * r13;
    r75 = 5.00000000000000000e-01;
    r73 = fma(r75, r74, r73);
    r74 = r24 * r73;
    r72 = r15 * r14;
    r76 = r12 * r17;
    r76 = fma(r75, r76, r75 * r72);
    r72 = r11 * r18;
    r76 = fma(r1, r72, r76);
    r77 = r16 * r13;
    r76 = fma(r75, r77, r76);
    r77 = r29 * r76;
    r72 = fma(r10, r77, r10 * r74);
    r78 = r10 * r19;
    r79 = fma(r75, r26, r1 * r22);
    r79 = fma(r1, r23, r79);
    r79 = fma(r1, r25, r79);
    r80 = r10 * r27;
    r81 = r16 * r14;
    r82 = r11 * r17;
    r82 = fma(r1, r82, r75 * r81);
    r81 = r12 * r18;
    r82 = fma(r1, r81, r82);
    r83 = r15 * r13;
    r82 = fma(r1, r83, r82);
    r80 = r80 * r82;
    r78 = fma(r79, r78, r80);
    r72 = r72 + r78;
    r83 = r10 * r24;
    r83 = r83 * r82;
    r81 = r10 * r19;
    r81 = r81 * r76;
    r84 = r83 + r81;
    r85 = r27 * r34;
    r84 = fma(r73, r85, r84);
    r86 = r34 * r29;
    r84 = fma(r79, r86, r84);
    r84 = fma(r9, r84, r41 * r72);
    r72 = r24 * r76;
    r86 = -4.00000000000000000e+00;
    r72 = r72 * r86;
    r85 = r27 * r79;
    r87 = r86 * r85;
    r88 = r72 + r87;
    r84 = fma(r8, r88, r84);
    r71 = r71 * r84;
    r88 = r48 * r48;
    r89 = r10 * r24;
    r89 = r89 * r79;
    r90 = r10 * r27;
    r90 = fma(r76, r90, r89);
    r76 = r10 * r19;
    r76 = r76 * r73;
    r91 = r10 * r29;
    r91 = r91 * r82;
    r92 = r76 + r91;
    r93 = r90 + r92;
    r77 = fma(r34, r77, r34 * r74);
    r77 = r77 + r78;
    r77 = fma(r8, r77, r9 * r93);
    r93 = r19 * r82;
    r93 = r93 * r86;
    r72 = r72 + r93;
    r77 = fma(r41, r72, r77);
    r52 = r47 * r52;
    r52 = 1.0 / r52;
    r47 = r34 * r52;
    r88 = r88 * r77;
    r88 = fma(r47, r88, r39 * r71);
    r71 = r28 * r28;
    r71 = r71 * r47;
    r72 = r10 * r28;
    r94 = r19 * r34;
    r95 = r34 * r29;
    r95 = r95 * r82;
    r94 = fma(r73, r94, r95);
    r94 = r94 + r90;
    r87 = r93 + r87;
    r87 = fma(r9, r87, r41 * r94);
    r94 = r10 * r29;
    r94 = fma(r79, r94, r81);
    r81 = r10 * r27;
    r81 = fma(r73, r81, r83);
    r94 = r94 + r81;
    r87 = fma(r8, r94, r87);
    r72 = r72 * r87;
    r88 = fma(r39, r72, r88);
    r88 = fma(r77, r71, r88);
    r58 = r58 * r44;
    r58 = r58 * r66;
    r58 = r58 * r1;
    r58 = r58 * r57;
    r58 = r58 * r88;
    r72 = r44 * r44;
    r94 = r7 * r72;
    r94 = r94 * r71;
    r83 = r28 * r57;
    r93 = r88 * r83;
    r30 = r35 + r30;
    r30 = 1.0 / r30;
    r90 = r30 * r46;
    r96 = r28 * r7;
    r93 = r93 * r90;
    r93 = fma(r96, r93, r77 * r94);
    r97 = r21 * r28;
    r43 = r38 * r43;
    r43 = 1.0 / r43;
    r38 = r43 * r42;
    r38 = r38 * r83;
    r97 = r97 * r88;
    r93 = fma(r38, r97, r93);
    r98 = r59 * r96;
    r93 = fma(r87, r98, r93);
    r97 = r88 * r57;
    r97 = r97 * r90;
    r99 = r21 * r48;
    r99 = r99 * r48;
    r99 = r99 * r88;
    r99 = r99 * r57;
    r99 = r99 * r43;
    r99 = fma(r42, r99, r54 * r97);
    r97 = r84 * r53;
    r99 = fma(r59, r97, r99);
    r100 = r77 * r47;
    r72 = r54 * r72;
    r99 = fma(r72, r100, r99);
    r100 = r93 + r99;
    r97 = fma(r55, r100, r58);
    r101 = r49 * r88;
    r101 = r101 * r57;
    r101 = r101 * r90;
    r102 = r48 * r48;
    r103 = -3.00000000000000000e+00;
    r102 = r102 * r103;
    r102 = r102 * r88;
    r102 = r102 * r57;
    r102 = r102 * r43;
    r102 = fma(r42, r102, r54 * r101);
    r101 = 6.00000000000000000e+00;
    r104 = r84 * r101;
    r104 = r104 * r42;
    r102 = fma(r53, r104, r102);
    r105 = -6.00000000000000000e+00;
    r106 = r105 * r52;
    r106 = r106 * r72;
    r102 = fma(r77, r106, r102);
    r102 = r102 + r93;
    r93 = r84 * r98;
    r104 = r21 * r48;
    r104 = r104 * r63;
    r104 = r104 * r77;
    r104 = r104 * r67;
    r97 = fma(r46, r104, r97);
    r107 = r48 * r75;
    r107 = r107 * r57;
    r107 = r107 * r30;
    r107 = r107 * r68;
    r108 = r63 * r107;
    r109 = r4 * r34;
    r109 = r109 * r48;
    r109 = r109 * r88;
    r97 = fma(r38, r109, r97);
    r110 = r4 * r87;
    r110 = r110 * r53;
    r97 = fma(r59, r110, r97);
    r111 = r4 * r28;
    r111 = r111 * r44;
    r111 = r111 * r44;
    r111 = r111 * r86;
    r111 = r111 * r52;
    r111 = r111 * r53;
    r112 = r4 * r88;
    r113 = r10 * r53;
    r113 = r113 * r83;
    r113 = r113 * r90;
    r97 = fma(r113, r112, r97);
    r114 = r61 * r10;
    r114 = r114 * r45;
    r114 = fma(r60, r100, r100 * r114);
    r64 = r64 * r49;
    r64 = r64 * r62;
    r62 = 4.00000000000000000e+00;
    r65 = r62 * r65;
    r114 = fma(r100, r64, r114);
    r114 = fma(r100, r65, r114);
    r62 = r48 * r114;
    r97 = fma(r69, r62, r97);
    r115 = r21 * r48;
    r115 = r115 * r77;
    r115 = r115 * r67;
    r97 = fma(r46, r115, r97);
    r97 = fma(r5, r102, r97);
    r97 = fma(r4, r93, r97);
    r97 = fma(r63, r58, r97);
    r97 = fma(r88, r108, r97);
    r97 = fma(r84, r70, r97);
    r97 = fma(r77, r111, r97);
    r97 = fma(r88, r107, r97);
    r97 = fma(r84, r69, r97);
    r115 = r2 * r97;
    r62 = r28 * r7;
    r62 = r62 * r44;
    r62 = r62 * r66;
    r62 = r62 * r1;
    r62 = r62 * r57;
    r62 = r62 * r88;
    r100 = fma(r56, r100, r62);
    r112 = r28 * r28;
    r112 = r112 * r44;
    r112 = r112 * r44;
    r112 = r112 * r77;
    r112 = r112 * r105;
    r112 = r112 * r7;
    r110 = r49 * r88;
    r110 = r110 * r83;
    r110 = r110 * r90;
    r110 = fma(r96, r110, r52 * r112);
    r112 = r28 * r103;
    r112 = r112 * r88;
    r110 = fma(r38, r112, r110);
    r109 = r28 * r87;
    r109 = r109 * r101;
    r109 = r109 * r7;
    r110 = fma(r42, r109, r110);
    r110 = r110 + r99;
    r99 = r21 * r28;
    r99 = r99 * r77;
    r99 = r99 * r67;
    r100 = fma(r46, r99, r100);
    r109 = r5 * r34;
    r109 = r109 * r48;
    r109 = r109 * r38;
    r112 = r75 * r88;
    r112 = r112 * r30;
    r112 = r112 * r83;
    r100 = fma(r68, r112, r100);
    r104 = r5 * r87;
    r104 = r104 * r53;
    r100 = fma(r59, r104, r100);
    r58 = r5 * r28;
    r58 = r58 * r44;
    r58 = r58 * r44;
    r58 = r58 * r86;
    r58 = r58 * r77;
    r58 = r58 * r52;
    r100 = fma(r53, r58, r100);
    r102 = r28 * r114;
    r100 = fma(r69, r102, r100);
    r116 = r5 * r88;
    r100 = fma(r113, r116, r100);
    r117 = r75 * r63;
    r117 = r117 * r88;
    r117 = r117 * r30;
    r117 = r117 * r83;
    r100 = fma(r68, r117, r100);
    r118 = r21 * r28;
    r118 = r118 * r63;
    r118 = r118 * r77;
    r118 = r118 * r67;
    r100 = fma(r46, r118, r100);
    r100 = fma(r4, r110, r100);
    r100 = fma(r5, r93, r100);
    r100 = fma(r63, r62, r100);
    r100 = fma(r88, r109, r100);
    r100 = fma(r87, r70, r100);
    r100 = fma(r87, r69, r100);
    r118 = r3 * r100;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             0 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r115,
                                             r118);
    r118 = r34 * r24;
    r118 = fma(r79, r118, r95);
    r115 = r10 * r27;
    r117 = r15 * r14;
    r116 = r12 * r17;
    r116 = fma(r1, r116, r1 * r117);
    r117 = r11 * r18;
    r116 = fma(r75, r117, r116);
    r102 = r16 * r13;
    r116 = fma(r1, r102, r116);
    r115 = r115 * r116;
    r102 = r10 * r19;
    r117 = r11 * r14;
    r58 = r16 * r17;
    r58 = fma(r75, r58, r75 * r117);
    r117 = r15 * r18;
    r58 = fma(r75, r117, r58);
    r104 = r12 * r13;
    r58 = fma(r1, r104, r58);
    r102 = fma(r58, r102, r115);
    r118 = r118 + r102;
    r104 = r10 * r24;
    r104 = r104 * r58;
    r117 = r10 * r29;
    r117 = fma(r116, r117, r104);
    r117 = r117 + r78;
    r117 = fma(r9, r117, r8 * r118);
    r118 = r24 * r82;
    r118 = r118 * r86;
    r78 = r19 * r116;
    r112 = r86 * r78;
    r62 = r118 + r112;
    r117 = fma(r41, r62, r117);
    r91 = r89 + r91;
    r91 = r91 + r102;
    r102 = r27 * r86;
    r102 = r102 * r58;
    r118 = r118 + r102;
    r118 = fma(r8, r118, r41 * r91);
    r91 = r34 * r29;
    r91 = fma(r34, r85, r58 * r91);
    r89 = r10 * r19;
    r89 = r89 * r82;
    r62 = r10 * r24;
    r62 = fma(r116, r62, r89);
    r91 = r91 + r62;
    r118 = fma(r9, r91, r118);
    r91 = r101 * r118;
    r91 = r91 * r42;
    r91 = fma(r53, r91, r117 * r106);
    r99 = r48 * r48;
    r93 = r10 * r48;
    r93 = r93 * r118;
    r110 = r10 * r28;
    r104 = r80 + r104;
    r80 = r19 * r34;
    r104 = fma(r79, r80, r104);
    r79 = r34 * r29;
    r104 = fma(r116, r79, r104);
    r79 = r10 * r29;
    r85 = fma(r10, r85, r58 * r79);
    r85 = r85 + r62;
    r85 = fma(r8, r85, r41 * r104);
    r112 = r102 + r112;
    r85 = fma(r9, r112, r85);
    r110 = r110 * r85;
    r110 = fma(r39, r110, r39 * r93);
    r93 = r48 * r48;
    r93 = r93 * r117;
    r110 = fma(r47, r93, r110);
    r110 = fma(r117, r71, r110);
    r93 = r103 * r110;
    r99 = r99 * r57;
    r99 = r99 * r43;
    r99 = r99 * r42;
    r91 = fma(r93, r99, r91);
    r112 = r49 * r110;
    r112 = r112 * r57;
    r112 = r112 * r90;
    r91 = fma(r54, r112, r91);
    r102 = r21 * r28;
    r102 = r102 * r110;
    r102 = fma(r38, r102, r85 * r98);
    r104 = r110 * r83;
    r104 = r104 * r90;
    r102 = fma(r96, r104, r102);
    r102 = fma(r117, r94, r102);
    r91 = r91 + r102;
    r112 = r117 * r47;
    r99 = r118 * r53;
    r99 = fma(r59, r99, r72 * r112);
    r112 = r21 * r48;
    r112 = r112 * r48;
    r112 = r112 * r110;
    r112 = r112 * r57;
    r112 = r112 * r43;
    r99 = fma(r42, r112, r99);
    r104 = r110 * r57;
    r104 = r104 * r90;
    r99 = fma(r54, r104, r99);
    r102 = r102 + r99;
    r91 = fma(r55, r102, r5 * r91);
    r104 = r21 * r48;
    r104 = r104 * r117;
    r104 = r104 * r67;
    r91 = fma(r46, r104, r91);
    r112 = r44 * r63;
    r112 = r112 * r1;
    r112 = r112 * r110;
    r112 = r112 * r66;
    r112 = r112 * r57;
    r91 = fma(r53, r112, r91);
    r79 = r61 * r10;
    r79 = r79 * r45;
    r79 = fma(r102, r79, r60 * r102);
    r79 = fma(r102, r65, r79);
    r79 = fma(r102, r64, r79);
    r58 = r48 * r79;
    r91 = fma(r69, r58, r91);
    r80 = r21 * r48;
    r80 = r80 * r63;
    r80 = r80 * r117;
    r80 = r80 * r67;
    r91 = fma(r46, r80, r91);
    r119 = r4 * r110;
    r91 = fma(r113, r119, r91);
    r120 = r4 * r34;
    r120 = r120 * r48;
    r120 = r120 * r110;
    r91 = fma(r38, r120, r91);
    r121 = r4 * r85;
    r121 = r121 * r53;
    r91 = fma(r59, r121, r91);
    r122 = r4 * r118;
    r91 = fma(r98, r122, r91);
    r123 = r44 * r1;
    r123 = r123 * r110;
    r123 = r123 * r66;
    r123 = r123 * r57;
    r91 = fma(r53, r123, r91);
    r91 = fma(r110, r108, r91);
    r91 = fma(r110, r107, r91);
    r91 = fma(r117, r111, r91);
    r91 = fma(r118, r69, r91);
    r91 = fma(r118, r70, r91);
    r123 = r2 * r91;
    r122 = r28 * r101;
    r122 = r122 * r85;
    r122 = r122 * r7;
    r121 = r28 * r38;
    r121 = fma(r93, r121, r42 * r122);
    r122 = r28 * r28;
    r122 = r122 * r44;
    r122 = r122 * r44;
    r122 = r122 * r105;
    r122 = r122 * r117;
    r122 = r122 * r7;
    r121 = fma(r52, r122, r121);
    r93 = r49 * r110;
    r93 = r93 * r83;
    r93 = r93 * r90;
    r121 = fma(r96, r93, r121);
    r121 = r121 + r99;
    r102 = fma(r56, r102, r4 * r121);
    r121 = r44 * r1;
    r121 = r121 * r110;
    r121 = r121 * r7;
    r121 = r121 * r66;
    r102 = fma(r83, r121, r102);
    r99 = r28 * r79;
    r102 = fma(r69, r99, r102);
    r93 = r75 * r110;
    r93 = r93 * r30;
    r93 = r93 * r83;
    r102 = fma(r68, r93, r102);
    r122 = r5 * r28;
    r122 = r122 * r44;
    r122 = r122 * r44;
    r122 = r122 * r86;
    r122 = r122 * r117;
    r122 = r122 * r52;
    r102 = fma(r53, r122, r102);
    r120 = r5 * r110;
    r102 = fma(r113, r120, r102);
    r119 = r21 * r28;
    r119 = r119 * r63;
    r119 = r119 * r117;
    r119 = r119 * r67;
    r102 = fma(r46, r119, r102);
    r80 = r44 * r63;
    r80 = r80 * r1;
    r80 = r80 * r110;
    r80 = r80 * r7;
    r80 = r80 * r66;
    r102 = fma(r83, r80, r102);
    r58 = r21 * r28;
    r58 = r58 * r117;
    r58 = r58 * r67;
    r102 = fma(r46, r58, r102);
    r112 = r5 * r85;
    r112 = r112 * r53;
    r102 = fma(r59, r112, r102);
    r104 = r75 * r63;
    r104 = r104 * r110;
    r104 = r104 * r30;
    r104 = r104 * r83;
    r102 = fma(r68, r104, r102);
    r124 = r5 * r118;
    r102 = fma(r98, r124, r102);
    r102 = fma(r85, r70, r102);
    r102 = fma(r110, r109, r102);
    r102 = fma(r85, r69, r102);
    r124 = r3 * r102;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             2 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r123,
                                             r124);
    r124 = r19 * r86;
    r26 = fma(r1, r26, r75 * r22);
    r26 = fma(r75, r23, r26);
    r26 = fma(r75, r25, r26);
    r124 = r124 * r26;
    r74 = r86 * r74;
    r25 = r124 + r74;
    r23 = r10 * r27;
    r23 = r23 * r26;
    r89 = r89 + r23;
    r22 = r34 * r24;
    r89 = fma(r116, r22, r89);
    r123 = r34 * r29;
    r89 = fma(r73, r123, r89);
    r89 = fma(r8, r89, r41 * r25);
    r25 = r10 * r29;
    r25 = fma(r10, r78, r26 * r25);
    r25 = r25 + r81;
    r89 = fma(r9, r25, r89);
    r25 = r10 * r28;
    r123 = r10 * r24;
    r123 = r123 * r26;
    r115 = r115 + r123;
    r115 = r115 + r92;
    r92 = r34 * r29;
    r78 = fma(r34, r78, r26 * r92);
    r78 = r78 + r81;
    r78 = fma(r41, r78, r8 * r115);
    r82 = r27 * r82;
    r82 = r82 * r86;
    r124 = r124 + r82;
    r78 = fma(r9, r124, r78);
    r25 = r25 * r78;
    r124 = r48 * r48;
    r124 = r124 * r89;
    r124 = fma(r47, r124, r39 * r25);
    r25 = r10 * r48;
    r95 = r76 + r95;
    r76 = r27 * r34;
    r95 = fma(r116, r76, r95);
    r95 = r95 + r123;
    r74 = r82 + r74;
    r74 = fma(r8, r74, r9 * r95);
    r8 = r10 * r29;
    r8 = fma(r73, r8, r23);
    r8 = r8 + r62;
    r74 = fma(r41, r8, r74);
    r25 = r25 * r74;
    r124 = fma(r39, r25, r124);
    r124 = fma(r89, r71, r124);
    r25 = r124 * r83;
    r25 = r25 * r90;
    r25 = fma(r96, r25, r89 * r94);
    r8 = r21 * r28;
    r8 = r8 * r124;
    r25 = fma(r38, r8, r25);
    r25 = fma(r78, r98, r25);
    r8 = r21 * r48;
    r8 = r8 * r48;
    r8 = r8 * r124;
    r8 = r8 * r57;
    r8 = r8 * r43;
    r41 = r74 * r53;
    r41 = fma(r59, r41, r42 * r8);
    r8 = r89 * r47;
    r41 = fma(r72, r8, r41);
    r62 = r124 * r57;
    r62 = r62 * r90;
    r41 = fma(r54, r62, r41);
    r62 = r25 + r41;
    r8 = r48 * r48;
    r8 = r8 * r103;
    r8 = r8 * r124;
    r8 = r8 * r57;
    r8 = r8 * r43;
    r23 = r101 * r74;
    r23 = r23 * r42;
    r23 = fma(r53, r23, r42 * r8);
    r8 = r49 * r124;
    r8 = r8 * r57;
    r8 = r8 * r90;
    r23 = fma(r54, r8, r23);
    r23 = fma(r89, r106, r23);
    r23 = r23 + r25;
    r23 = fma(r5, r23, r55 * r62);
    r25 = r61 * r10;
    r25 = r25 * r45;
    r25 = fma(r62, r25, r60 * r62);
    r25 = fma(r62, r64, r25);
    r25 = fma(r62, r65, r25);
    r8 = r48 * r25;
    r23 = fma(r69, r8, r23);
    r73 = r4 * r74;
    r23 = fma(r98, r73, r23);
    r95 = r4 * r124;
    r23 = fma(r113, r95, r23);
    r9 = r21 * r48;
    r9 = r9 * r63;
    r9 = r9 * r89;
    r9 = r9 * r67;
    r23 = fma(r46, r9, r23);
    r82 = r4 * r34;
    r82 = r82 * r48;
    r82 = r82 * r124;
    r23 = fma(r38, r82, r23);
    r76 = r21 * r48;
    r76 = r76 * r89;
    r76 = r76 * r67;
    r23 = fma(r46, r76, r23);
    r123 = r44 * r1;
    r123 = r123 * r124;
    r123 = r123 * r66;
    r123 = r123 * r57;
    r23 = fma(r53, r123, r23);
    r116 = r4 * r78;
    r116 = r116 * r53;
    r23 = fma(r59, r116, r23);
    r115 = r44 * r63;
    r115 = r115 * r1;
    r115 = r115 * r124;
    r115 = r115 * r66;
    r115 = r115 * r57;
    r23 = fma(r53, r115, r23);
    r23 = fma(r124, r107, r23);
    r23 = fma(r124, r108, r23);
    r23 = fma(r74, r70, r23);
    r23 = fma(r74, r69, r23);
    r23 = fma(r89, r111, r23);
    r115 = r2 * r23;
    r116 = r28 * r28;
    r116 = r116 * r44;
    r116 = r116 * r44;
    r116 = r116 * r105;
    r116 = r116 * r89;
    r116 = r116 * r7;
    r123 = r49 * r124;
    r123 = r123 * r83;
    r123 = r123 * r90;
    r123 = fma(r96, r123, r52 * r116);
    r116 = r28 * r101;
    r116 = r116 * r78;
    r116 = r116 * r7;
    r123 = fma(r42, r116, r123);
    r76 = r28 * r103;
    r76 = r76 * r124;
    r123 = fma(r38, r76, r123);
    r123 = r123 + r41;
    r123 = fma(r4, r123, r56 * r62);
    r62 = r21 * r28;
    r62 = r62 * r63;
    r62 = r62 * r89;
    r62 = r62 * r67;
    r123 = fma(r46, r62, r123);
    r41 = r21 * r28;
    r41 = r41 * r89;
    r41 = r41 * r67;
    r123 = fma(r46, r41, r123);
    r76 = r5 * r74;
    r123 = fma(r98, r76, r123);
    r116 = r44 * r63;
    r116 = r116 * r1;
    r116 = r116 * r124;
    r116 = r116 * r7;
    r116 = r116 * r66;
    r123 = fma(r83, r116, r123);
    r82 = r5 * r124;
    r123 = fma(r113, r82, r123);
    r9 = r75 * r124;
    r9 = r9 * r30;
    r9 = r9 * r83;
    r123 = fma(r68, r9, r123);
    r95 = r28 * r25;
    r123 = fma(r69, r95, r123);
    r73 = r44 * r1;
    r73 = r73 * r124;
    r73 = r73 * r7;
    r73 = r73 * r66;
    r123 = fma(r83, r73, r123);
    r8 = r75 * r63;
    r8 = r8 * r124;
    r8 = r8 * r30;
    r8 = r8 * r83;
    r123 = fma(r68, r8, r123);
    r81 = r5 * r28;
    r81 = r81 * r44;
    r81 = r81 * r44;
    r81 = r81 * r86;
    r81 = r81 * r89;
    r81 = r81 * r52;
    r123 = fma(r53, r81, r123);
    r92 = r5 * r78;
    r92 = r92 * r53;
    r123 = fma(r59, r92, r123);
    r123 = fma(r78, r70, r123);
    r123 = fma(r78, r69, r123);
    r123 = fma(r124, r109, r123);
    r92 = r3 * r123;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r115, r92);
    r92 = r36 * r101;
    r92 = r92 * r42;
    r115 = r48 * r48;
    r81 = r10 * r32;
    r81 = r81 * r28;
    r81 = fma(r6, r71, r39 * r81);
    r8 = r6 * r48;
    r8 = r8 * r48;
    r81 = fma(r47, r8, r81);
    r73 = r10 * r36;
    r73 = r73 * r48;
    r81 = fma(r39, r73, r81);
    r115 = r115 * r103;
    r115 = r115 * r81;
    r115 = r115 * r57;
    r115 = r115 * r43;
    r115 = fma(r42, r115, r53 * r92);
    r92 = r49 * r81;
    r92 = r92 * r57;
    r92 = r92 * r90;
    r115 = fma(r54, r92, r115);
    r73 = r21 * r28;
    r73 = r73 * r81;
    r73 = fma(r38, r73, r32 * r98);
    r8 = r81 * r83;
    r8 = r8 * r90;
    r73 = fma(r96, r8, r73);
    r73 = fma(r6, r94, r73);
    r115 = fma(r6, r106, r115);
    r115 = r115 + r73;
    r92 = r36 * r53;
    r8 = r21 * r48;
    r8 = r8 * r48;
    r8 = r8 * r81;
    r8 = r8 * r57;
    r8 = r8 * r43;
    r8 = fma(r42, r8, r59 * r92);
    r92 = r81 * r57;
    r92 = r92 * r90;
    r8 = fma(r54, r92, r8);
    r95 = r6 * r47;
    r8 = fma(r72, r95, r8);
    r73 = r73 + r8;
    r115 = fma(r55, r73, r5 * r115);
    r95 = r81 * r113;
    r92 = r44 * r63;
    r92 = r92 * r1;
    r92 = r92 * r81;
    r92 = r92 * r66;
    r92 = r92 * r57;
    r115 = fma(r53, r92, r115);
    r9 = r44 * r1;
    r9 = r9 * r81;
    r9 = r9 * r66;
    r9 = r9 * r57;
    r115 = fma(r53, r9, r115);
    r82 = r4 * r34;
    r82 = r82 * r48;
    r82 = r82 * r81;
    r115 = fma(r38, r82, r115);
    r116 = r61 * r10;
    r116 = r116 * r45;
    r116 = fma(r60, r73, r73 * r116);
    r116 = fma(r73, r65, r116);
    r116 = fma(r73, r64, r116);
    r76 = r48 * r116;
    r115 = fma(r69, r76, r115);
    r41 = r21 * r6;
    r41 = r41 * r48;
    r41 = r41 * r63;
    r41 = r41 * r67;
    r115 = fma(r46, r41, r115);
    r62 = r21 * r6;
    r62 = r62 * r48;
    r62 = r62 * r67;
    r115 = fma(r46, r62, r115);
    r26 = r4 * r32;
    r26 = r26 * r53;
    r115 = fma(r59, r26, r115);
    r22 = r4 * r36;
    r115 = fma(r98, r22, r115);
    r115 = fma(r36, r70, r115);
    r115 = fma(r81, r108, r115);
    r115 = fma(r4, r95, r115);
    r115 = fma(r81, r107, r115);
    r115 = fma(r6, r111, r115);
    r115 = fma(r36, r69, r115);
    r22 = r2 * r115;
    r26 = r32 * r28;
    r26 = r26 * r101;
    r26 = r26 * r7;
    r62 = r28 * r103;
    r62 = r62 * r81;
    r62 = fma(r38, r62, r42 * r26);
    r26 = r6 * r28;
    r26 = r26 * r28;
    r26 = r26 * r44;
    r26 = r26 * r44;
    r26 = r26 * r105;
    r26 = r26 * r7;
    r62 = fma(r52, r26, r62);
    r41 = r49 * r81;
    r41 = r41 * r83;
    r41 = r41 * r90;
    r62 = fma(r96, r41, r62);
    r62 = r62 + r8;
    r73 = fma(r56, r73, r4 * r62);
    r62 = r75 * r81;
    r62 = r62 * r30;
    r62 = r62 * r83;
    r73 = fma(r68, r62, r73);
    r8 = r44 * r63;
    r8 = r8 * r1;
    r8 = r8 * r81;
    r8 = r8 * r7;
    r8 = r8 * r66;
    r73 = fma(r83, r8, r73);
    r41 = r21 * r6;
    r41 = r41 * r28;
    r41 = r41 * r63;
    r41 = r41 * r67;
    r73 = fma(r46, r41, r73);
    r26 = r21 * r6;
    r26 = r26 * r28;
    r26 = r26 * r67;
    r73 = fma(r46, r26, r73);
    r76 = r5 * r6;
    r76 = r76 * r28;
    r76 = r76 * r44;
    r76 = r76 * r44;
    r76 = r76 * r86;
    r76 = r76 * r52;
    r73 = fma(r53, r76, r73);
    r82 = r28 * r116;
    r73 = fma(r69, r82, r73);
    r9 = r75 * r63;
    r9 = r9 * r81;
    r9 = r9 * r30;
    r9 = r9 * r83;
    r73 = fma(r68, r9, r73);
    r92 = r5 * r32;
    r92 = r92 * r53;
    r73 = fma(r59, r92, r73);
    r104 = r44 * r1;
    r104 = r104 * r81;
    r104 = r104 * r7;
    r104 = r104 * r66;
    r73 = fma(r83, r104, r73);
    r112 = r5 * r36;
    r73 = fma(r98, r112, r73);
    r73 = fma(r32, r69, r73);
    r73 = fma(r5, r95, r73);
    r73 = fma(r81, r109, r73);
    r73 = fma(r32, r70, r73);
    r112 = r3 * r73;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r22, r112);
    r112 = r10 * r37;
    r112 = r112 * r28;
    r112 = fma(r39, r112, r33 * r71);
    r22 = r33 * r48;
    r22 = r22 * r48;
    r112 = fma(r47, r22, r112);
    r104 = r10 * r51;
    r104 = r104 * r48;
    r112 = fma(r39, r104, r112);
    r104 = r49 * r112;
    r104 = r104 * r57;
    r104 = r104 * r90;
    r104 = fma(r33, r106, r54 * r104);
    r22 = r48 * r48;
    r22 = r22 * r103;
    r22 = r22 * r112;
    r22 = r22 * r57;
    r22 = r22 * r43;
    r104 = fma(r42, r22, r104);
    r92 = r51 * r101;
    r92 = r92 * r42;
    r104 = fma(r53, r92, r104);
    r9 = fma(r37, r98, r33 * r94);
    r82 = r112 * r83;
    r82 = r82 * r90;
    r9 = fma(r96, r82, r9);
    r76 = r21 * r28;
    r76 = r76 * r112;
    r9 = fma(r38, r76, r9);
    r104 = r104 + r9;
    r92 = r112 * r57;
    r92 = r92 * r90;
    r22 = r33 * r47;
    r22 = fma(r72, r22, r54 * r92);
    r92 = r21 * r48;
    r92 = r92 * r48;
    r92 = r92 * r112;
    r92 = r92 * r57;
    r92 = r92 * r43;
    r22 = fma(r42, r92, r22);
    r76 = r51 * r53;
    r22 = fma(r59, r76, r22);
    r9 = r9 + r22;
    r104 = fma(r55, r9, r5 * r104);
    r76 = r4 * r37;
    r76 = r76 * r53;
    r104 = fma(r59, r76, r104);
    r92 = r4 * r112;
    r104 = fma(r113, r92, r104);
    r82 = r21 * r33;
    r82 = r82 * r48;
    r82 = r82 * r67;
    r104 = fma(r46, r82, r104);
    r26 = r4 * r34;
    r26 = r26 * r48;
    r26 = r26 * r112;
    r104 = fma(r38, r26, r104);
    r95 = r44 * r1;
    r95 = r95 * r112;
    r95 = r95 * r66;
    r95 = r95 * r57;
    r104 = fma(r53, r95, r104);
    r41 = r61 * r10;
    r41 = r41 * r45;
    r41 = fma(r60, r9, r9 * r41);
    r41 = fma(r9, r64, r41);
    r41 = fma(r9, r65, r41);
    r8 = r48 * r41;
    r104 = fma(r69, r8, r104);
    r62 = r44 * r63;
    r62 = r62 * r1;
    r62 = r62 * r112;
    r62 = r62 * r66;
    r62 = r62 * r57;
    r104 = fma(r53, r62, r104);
    r58 = r21 * r33;
    r58 = r58 * r48;
    r58 = r58 * r63;
    r58 = r58 * r67;
    r104 = fma(r46, r58, r104);
    r80 = r4 * r51;
    r104 = fma(r98, r80, r104);
    r104 = fma(r112, r108, r104);
    r104 = fma(r51, r69, r104);
    r104 = fma(r33, r111, r104);
    r104 = fma(r112, r107, r104);
    r104 = fma(r51, r70, r104);
    r80 = r2 * r104;
    r58 = r33 * r28;
    r58 = r58 * r28;
    r58 = r58 * r44;
    r58 = r58 * r44;
    r58 = r58 * r105;
    r58 = r58 * r7;
    r62 = r37 * r28;
    r62 = r62 * r101;
    r62 = r62 * r7;
    r62 = fma(r42, r62, r52 * r58);
    r58 = r49 * r112;
    r58 = r58 * r83;
    r58 = r58 * r90;
    r62 = fma(r96, r58, r62);
    r8 = r28 * r103;
    r8 = r8 * r112;
    r62 = fma(r38, r8, r62);
    r62 = r62 + r22;
    r62 = fma(r4, r62, r56 * r9);
    r9 = r21 * r33;
    r9 = r9 * r28;
    r9 = r9 * r67;
    r62 = fma(r46, r9, r62);
    r22 = r5 * r37;
    r22 = r22 * r53;
    r62 = fma(r59, r22, r62);
    r8 = r5 * r112;
    r62 = fma(r113, r8, r62);
    r58 = r75 * r112;
    r58 = r58 * r30;
    r58 = r58 * r83;
    r62 = fma(r68, r58, r62);
    r95 = r28 * r41;
    r62 = fma(r69, r95, r62);
    r26 = r44 * r63;
    r26 = r26 * r1;
    r26 = r26 * r112;
    r26 = r26 * r7;
    r26 = r26 * r66;
    r62 = fma(r83, r26, r62);
    r82 = r5 * r33;
    r82 = r82 * r28;
    r82 = r82 * r44;
    r82 = r82 * r44;
    r82 = r82 * r86;
    r82 = r82 * r52;
    r62 = fma(r53, r82, r62);
    r92 = r44 * r1;
    r92 = r92 * r112;
    r92 = r92 * r7;
    r92 = r92 * r66;
    r62 = fma(r83, r92, r62);
    r76 = r75 * r63;
    r76 = r76 * r112;
    r76 = r76 * r30;
    r76 = r76 * r83;
    r62 = fma(r68, r76, r62);
    r119 = r21 * r33;
    r119 = r119 * r28;
    r119 = r119 * r63;
    r119 = r119 * r67;
    r62 = fma(r46, r119, r62);
    r120 = r5 * r51;
    r62 = fma(r98, r120, r62);
    r62 = fma(r37, r70, r62);
    r62 = fma(r112, r109, r62);
    r62 = fma(r37, r69, r62);
    r120 = r3 * r62;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r80, r120);
    r120 = r48 * r48;
    r80 = r10 * r40;
    r80 = r80 * r28;
    r71 = fma(r31, r71, r39 * r80);
    r80 = r10 * r50;
    r80 = r80 * r48;
    r71 = fma(r39, r80, r71);
    r39 = r31 * r48;
    r39 = r39 * r48;
    r71 = fma(r47, r39, r71);
    r120 = r120 * r103;
    r120 = r120 * r71;
    r120 = r120 * r57;
    r120 = r120 * r43;
    r39 = r49 * r71;
    r39 = r39 * r57;
    r39 = r39 * r90;
    r39 = fma(r54, r39, r42 * r120);
    r120 = r50 * r101;
    r120 = r120 * r42;
    r39 = fma(r53, r120, r39);
    r94 = fma(r40, r98, r31 * r94);
    r80 = r71 * r83;
    r80 = r80 * r90;
    r94 = fma(r96, r80, r94);
    r119 = r21 * r28;
    r119 = r119 * r71;
    r94 = fma(r38, r119, r94);
    r39 = fma(r31, r106, r39);
    r39 = r39 + r94;
    r106 = r21 * r48;
    r106 = r106 * r48;
    r106 = r106 * r71;
    r106 = r106 * r57;
    r106 = r106 * r43;
    r43 = r71 * r57;
    r43 = r43 * r90;
    r43 = fma(r54, r43, r42 * r106);
    r106 = r50 * r53;
    r43 = fma(r59, r106, r43);
    r54 = r31 * r47;
    r43 = fma(r72, r54, r43);
    r94 = r94 + r43;
    r55 = fma(r55, r94, r5 * r39);
    r39 = r4 * r50;
    r55 = fma(r98, r39, r55);
    r54 = r4 * r40;
    r54 = r54 * r53;
    r55 = fma(r59, r54, r55);
    r106 = r61 * r10;
    r106 = r106 * r45;
    r106 = fma(r94, r106, r60 * r94);
    r106 = fma(r94, r65, r106);
    r106 = fma(r94, r64, r106);
    r64 = r48 * r106;
    r55 = fma(r69, r64, r55);
    r65 = r4 * r71;
    r55 = fma(r113, r65, r55);
    r60 = r21 * r31;
    r60 = r60 * r48;
    r60 = r60 * r67;
    r55 = fma(r46, r60, r55);
    r45 = r44 * r63;
    r45 = r45 * r1;
    r45 = r45 * r71;
    r45 = r45 * r66;
    r45 = r45 * r57;
    r55 = fma(r53, r45, r55);
    r72 = r4 * r34;
    r72 = r72 * r48;
    r72 = r72 * r71;
    r55 = fma(r38, r72, r55);
    r120 = r21 * r31;
    r120 = r120 * r48;
    r120 = r120 * r63;
    r120 = r120 * r67;
    r55 = fma(r46, r120, r55);
    r119 = r44 * r1;
    r119 = r119 * r71;
    r119 = r119 * r66;
    r119 = r119 * r57;
    r55 = fma(r53, r119, r55);
    r55 = fma(r50, r70, r55);
    r55 = fma(r31, r111, r55);
    r55 = fma(r71, r108, r55);
    r55 = fma(r71, r107, r55);
    r55 = fma(r50, r69, r55);
    r119 = r2 * r55;
    r120 = r31 * r28;
    r120 = r120 * r28;
    r120 = r120 * r44;
    r120 = r120 * r44;
    r120 = r120 * r105;
    r120 = r120 * r7;
    r105 = r40 * r28;
    r105 = r105 * r101;
    r105 = r105 * r7;
    r105 = fma(r42, r105, r52 * r120);
    r120 = r49 * r71;
    r120 = r120 * r83;
    r120 = r120 * r90;
    r105 = fma(r96, r120, r105);
    r96 = r28 * r103;
    r96 = r96 * r71;
    r105 = fma(r38, r96, r105);
    r105 = r105 + r43;
    r105 = fma(r4, r105, r56 * r94);
    r94 = r44 * r1;
    r94 = r94 * r71;
    r94 = r94 * r7;
    r94 = r94 * r66;
    r105 = fma(r83, r94, r105);
    r56 = r21 * r31;
    r56 = r56 * r28;
    r56 = r56 * r67;
    r105 = fma(r46, r56, r105);
    r43 = r5 * r50;
    r105 = fma(r98, r43, r105);
    r98 = r21 * r31;
    r98 = r98 * r28;
    r98 = r98 * r63;
    r98 = r98 * r67;
    r105 = fma(r46, r98, r105);
    r46 = r75 * r63;
    r46 = r46 * r71;
    r46 = r46 * r30;
    r46 = r46 * r83;
    r105 = fma(r68, r46, r105);
    r67 = r5 * r31;
    r67 = r67 * r28;
    r67 = r67 * r44;
    r67 = r67 * r44;
    r67 = r67 * r86;
    r67 = r67 * r52;
    r105 = fma(r53, r67, r105);
    r52 = r5 * r40;
    r52 = r52 * r53;
    r105 = fma(r59, r52, r105);
    r59 = r44 * r63;
    r59 = r59 * r1;
    r59 = r59 * r71;
    r59 = r59 * r7;
    r59 = r59 * r66;
    r105 = fma(r83, r59, r105);
    r66 = r5 * r71;
    r105 = fma(r113, r66, r105);
    r113 = r75 * r71;
    r113 = r113 * r30;
    r113 = r113 * r83;
    r105 = fma(r68, r113, r105);
    r68 = r28 * r106;
    r105 = fma(r69, r68, r105);
    r105 = fma(r40, r69, r105);
    r105 = fma(r40, r70, r105);
    r105 = fma(r71, r109, r105);
    r109 = r3 * r105;
    WriteIdx2<1024, double, double, double2>(out_pose_jac,
                                             10 * out_pose_jac_num_alloc,
                                             global_thread_idx,
                                             r119,
                                             r109);
    r109 = r3 * r21;
    r109 = r109 * r0;
    r20 = r21 * r20;
    r119 = r2 * r20;
    r109 = fma(r97, r119, r100 * r109);
    r68 = r3 * r21;
    r68 = r68 * r0;
    r68 = fma(r91, r119, r102 * r68);
    WriteSum2<double, double>((double*)inout_shared, r109, r68);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = r3 * r21;
    r68 = r68 * r0;
    r68 = fma(r23, r119, r123 * r68);
    r109 = r3 * r21;
    r109 = r109 * r0;
    r109 = fma(r115, r119, r73 * r109);
    WriteSum2<double, double>((double*)inout_shared, r68, r109);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r109 = r3 * r21;
    r109 = r109 * r0;
    r109 = fma(r104, r119, r62 * r109);
    r68 = r3 * r21;
    r68 = r68 * r0;
    r119 = fma(r55, r119, r105 * r68);
    WriteSum2<double, double>((double*)inout_shared, r109, r119);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r2 * r2;
    r119 = r97 * r2;
    r109 = r3 * r3;
    r68 = r100 * r109;
    r100 = fma(r100, r68, r97 * r119);
    r97 = r91 * r91;
    r113 = r102 * r102;
    r113 = fma(r109, r113, r2 * r97);
    WriteSum2<double, double>((double*)inout_shared, r100, r113);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r113 = r23 * r23;
    r100 = r123 * r123;
    r100 = fma(r109, r100, r2 * r113);
    r113 = r73 * r73;
    r97 = r115 * r115;
    r97 = fma(r2, r97, r109 * r113);
    WriteSum2<double, double>((double*)inout_shared, r100, r97);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r97 = r104 * r104;
    r100 = r62 * r62;
    r100 = fma(r109, r100, r2 * r97);
    r97 = r105 * r105;
    r113 = r55 * r55;
    r113 = fma(r2, r113, r109 * r97);
    WriteSum2<double, double>((double*)inout_shared, r100, r113);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r113 = fma(r91, r119, r102 * r68);
    r100 = fma(r123, r68, r23 * r119);
    WriteSum2<double, double>((double*)inout_shared, r113, r100);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r100 = fma(r73, r68, r115 * r119);
    r113 = fma(r62, r68, r104 * r119);
    WriteSum2<double, double>((double*)inout_shared, r100, r113);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r119 = fma(r55, r119, r105 * r68);
    r68 = r102 * r123;
    r113 = r91 * r23;
    r113 = fma(r2, r113, r109 * r68);
    WriteSum2<double, double>((double*)inout_shared, r119, r113);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r113 = r91 * r115;
    r119 = r102 * r73;
    r119 = fma(r109, r119, r2 * r113);
    r113 = r91 * r104;
    r68 = r102 * r62;
    r68 = fma(r109, r68, r2 * r113);
    WriteSum2<double, double>((double*)inout_shared, r119, r68);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = r102 * r105;
    r119 = r91 * r55;
    r119 = fma(r2, r119, r109 * r68);
    r68 = r123 * r73;
    r113 = r23 * r115;
    r113 = fma(r2, r113, r109 * r68);
    WriteSum2<double, double>((double*)inout_shared, r119, r113);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r113 = r23 * r104;
    r119 = r123 * r62;
    r119 = fma(r109, r119, r2 * r113);
    r113 = r123 * r105;
    r68 = r23 * r55;
    r68 = fma(r2, r68, r109 * r113);
    WriteSum2<double, double>((double*)inout_shared, r119, r68);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = r115 * r104;
    r119 = r73 * r62;
    r119 = fma(r109, r119, r2 * r68);
    r68 = r73 * r105;
    r113 = r115 * r55;
    r113 = fma(r2, r113, r109 * r68);
    WriteSum2<double, double>((double*)inout_shared, r119, r113);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r113 = r62 * r105;
    r119 = r104 * r55;
    r119 = fma(r2, r119, r109 * r113);
    WriteSum1<double, double>((double*)inout_shared, r119);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r21 * r0;
    WriteSum2<double, double>((double*)inout_shared, r20, r0);
  };
  FlushSumShared<2, double>(out_principal_point_njtr,
                            0 * out_principal_point_njtr_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r35, r35);
  };
  FlushSumShared<2, double>(out_principal_point_precond_diag,
                            0 * out_principal_point_precond_diag_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
}

void ThinPrismFisheyeSplitFixedFocalAndExtraFixedPointResJac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
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
    double* out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    double* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    double* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    double* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeSplitFixedFocalAndExtraFixedPointResJacKernel<<<n_blocks,
                                                                  1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      pixel,
      pixel_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
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
      out_principal_point_jac,
      out_principal_point_jac_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar