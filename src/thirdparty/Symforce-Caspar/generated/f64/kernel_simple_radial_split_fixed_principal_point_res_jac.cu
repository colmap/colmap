#include "kernel_simple_radial_split_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPrincipalPointResJacKernel(
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(principal_point,
                                            0 * principal_point_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r5);
  };
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r10,
                                            r11);
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r12, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r14,
                                            r15);
    r16 = fma(r13, r14, r8 * r11);
    r17 = r12 * r15;
    r16 = fma(r4, r17, r16);
    r16 = fma(r9, r10, r16);
    r17 = r16 * r16;
    r18 = -2.00000000000000000e+00;
    r17 = r17 * r18;
    r19 = 1.00000000000000000e+00;
    r20 = fma(r9, r15, r13 * r11);
    r21 = r12 * r10;
    r22 = r8 * r14;
    r20 = r20 + r21;
    r20 = fma(r4, r22, r20);
    r23 = r18 * r20;
    r23 = fma(r20, r23, r19);
    r24 = r17 + r23;
    r0 = fma(r6, r24, r0);
    r25 = 2.00000000000000000e+00;
    r26 = fma(r9, r14, r12 * r11);
    r27 = r13 * r10;
    r26 = fma(r4, r27, r26);
    r26 = fma(r8, r15, r26);
    r27 = r25 * r26;
    r27 = r27 * r20;
    r28 = r16 * r18;
    r29 = fma(r13, r15, r12 * r14);
    r29 = fma(r8, r10, r29);
    r29 = fma(r4, r29, r9 * r11);
    r28 = fma(r29, r28, r27);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r30);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r31 = r25 * r16;
    r31 = r31 * r26;
    r32 = r25 * r29;
    r33 = fma(r20, r32, r31);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r34);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r35 = r14 * r10;
    r35 = r35 * r25;
    r36 = r15 * r11;
    r37 = fma(r25, r36, r35);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r38, r39);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r40 = r10 * r11;
    r41 = r14 * r15;
    r41 = r41 * r25;
    r40 = fma(r18, r40, r41);
    r42 = r15 * r15;
    r42 = r42 * r18;
    r43 = r19 + r42;
    r44 = r10 * r10;
    r44 = r44 * r18;
    r43 = r43 + r44;
    r0 = fma(r7, r28, r0);
    r0 = fma(r30, r33, r0);
    r0 = fma(r34, r37, r0);
    r0 = fma(r39, r40, r0);
    r0 = fma(r38, r43, r0);
  };
  LoadShared<2, double, double>(focal_and_extra,
                                0 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target,
                        r45,
                        r46);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r47 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r48);
    r49 = r18 * r20;
    r49 = fma(r29, r49, r31);
    r48 = fma(r6, r49, r48);
    r36 = fma(r18, r36, r35);
    r42 = r19 + r42;
    r35 = r14 * r14;
    r35 = r35 * r18;
    r42 = r42 + r35;
    r31 = r15 * r10;
    r31 = r31 * r25;
    r50 = r14 * r11;
    r50 = fma(r25, r50, r31);
    r51 = r25 * r16;
    r51 = r51 * r20;
    r52 = fma(r26, r32, r51);
    r53 = r26 * r26;
    r53 = r53 * r18;
    r23 = r53 + r23;
    r48 = fma(r38, r36, r48);
    r48 = fma(r34, r42, r48);
    r48 = fma(r39, r50, r48);
    r48 = fma(r7, r52, r48);
    r48 = fma(r30, r23, r48);
    r54 = copysign(1.0, r48);
    r54 = fma(r47, r54, r48);
    r47 = r54 * r54;
    r48 = 1.0 / r47;
    r55 = r0 * r48;
    r27 = fma(r16, r32, r27);
    r5 = fma(r6, r27, r5);
    r56 = r10 * r11;
    r56 = fma(r25, r56, r41);
    r44 = r19 + r44;
    r44 = r44 + r35;
    r35 = r14 * r11;
    r35 = fma(r18, r35, r31);
    r31 = r26 * r18;
    r31 = fma(r29, r31, r51);
    r17 = r19 + r17;
    r17 = r17 + r53;
    r5 = fma(r38, r56, r5);
    r5 = fma(r39, r44, r5);
    r5 = fma(r34, r35, r5);
    r5 = fma(r30, r31, r5);
    r5 = fma(r7, r17, r5);
    r34 = r5 * r5;
    r39 = fma(r48, r34, r0 * r55);
    r19 = fma(r46, r39, r19);
    r38 = r0 * r19;
    r53 = 1.0 / r54;
    r51 = r45 * r53;
    r2 = fma(r51, r38, r2);
    r3 = fma(r3, r4, r1);
    r1 = r5 * r19;
    r3 = fma(r51, r1, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = r12 * r11;
    r38 = -5.00000000000000000e-01;
    r41 = r9 * r14;
    r41 = fma(r38, r41, r38 * r1);
    r1 = r8 * r15;
    r41 = fma(r38, r1, r41);
    r57 = r13 * r10;
    r58 = 5.00000000000000000e-01;
    r41 = fma(r58, r57, r41);
    r57 = r20 * r41;
    r1 = r8 * r11;
    r59 = r13 * r14;
    r59 = fma(r58, r59, r58 * r1);
    r1 = r12 * r15;
    r59 = fma(r38, r1, r59);
    r60 = r9 * r58;
    r59 = fma(r10, r60, r59);
    r1 = fma(r59, r32, r25 * r57);
    r61 = r25 * r26;
    r62 = r9 * r15;
    r63 = r13 * r38;
    r62 = fma(r11, r63, r38 * r62);
    r62 = fma(r58, r22, r62);
    r62 = fma(r38, r21, r62);
    r64 = r25 * r16;
    r65 = r12 * r14;
    r66 = r8 * r10;
    r66 = fma(r38, r66, r38 * r65);
    r66 = fma(r11, r60, r66);
    r66 = fma(r15, r63, r66);
    r64 = r64 * r66;
    r61 = fma(r62, r61, r64);
    r1 = r1 + r61;
    r65 = r25 * r20;
    r65 = r65 * r66;
    r67 = r25 * r26;
    r67 = r67 * r59;
    r68 = r65 + r67;
    r69 = r16 * r18;
    r68 = fma(r41, r69, r68);
    r70 = r18 * r29;
    r68 = fma(r62, r70, r68);
    r68 = fma(r7, r68, r30 * r1);
    r1 = r20 * r59;
    r70 = -4.00000000000000000e+00;
    r1 = r1 * r70;
    r69 = r16 * r62;
    r71 = r70 * r69;
    r72 = r1 + r71;
    r68 = fma(r6, r72, r68);
    r72 = r25 * r68;
    r73 = r25 * r20;
    r73 = r73 * r62;
    r74 = r25 * r16;
    r74 = fma(r59, r74, r73);
    r75 = r25 * r26;
    r75 = r75 * r41;
    r76 = r66 * r32;
    r77 = r75 + r76;
    r78 = r74 + r77;
    r79 = r18 * r29;
    r79 = fma(r18, r57, r59 * r79);
    r79 = r79 + r61;
    r79 = fma(r6, r79, r7 * r78);
    r78 = r26 * r70;
    r59 = r66 * r78;
    r1 = r1 + r59;
    r79 = fma(r30, r1, r79);
    r1 = r0 * r0;
    r47 = r54 * r47;
    r47 = 1.0 / r47;
    r47 = r18 * r47;
    r1 = r1 * r47;
    r72 = fma(r79, r1, r55 * r72);
    r54 = r79 * r47;
    r72 = fma(r34, r54, r72);
    r80 = r25 * r5;
    r81 = r26 * r18;
    r82 = r18 * r29;
    r82 = r82 * r66;
    r81 = fma(r41, r81, r82);
    r81 = r81 + r74;
    r59 = r71 + r59;
    r59 = fma(r7, r59, r30 * r81);
    r67 = fma(r62, r32, r67);
    r81 = r25 * r16;
    r81 = fma(r41, r81, r65);
    r67 = r67 + r81;
    r59 = fma(r6, r67, r59);
    r80 = r80 * r59;
    r72 = fma(r48, r80, r72);
    r46 = r46 * r51;
    r72 = r72 * r46;
    r80 = r19 * r68;
    r80 = fma(r51, r80, r0 * r72);
    r54 = r79 * r55;
    r67 = r4 * r19;
    r65 = r45 * r67;
    r80 = fma(r65, r54, r80);
    r54 = r19 * r59;
    r54 = fma(r51, r54, r5 * r72);
    r72 = r5 * r48;
    r72 = r72 * r65;
    r54 = fma(r79, r72, r54);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r80, r54);
    r76 = r73 + r76;
    r73 = r25 * r16;
    r71 = r8 * r11;
    r74 = r12 * r15;
    r74 = fma(r58, r74, r38 * r71);
    r71 = r9 * r10;
    r74 = fma(r38, r71, r74);
    r74 = fma(r14, r63, r74);
    r73 = r73 * r74;
    r71 = r25 * r26;
    r83 = r12 * r11;
    r84 = r8 * r15;
    r84 = fma(r58, r84, r58 * r83);
    r84 = fma(r14, r60, r84);
    r84 = fma(r10, r63, r84);
    r71 = fma(r84, r71, r73);
    r76 = r76 + r71;
    r63 = r20 * r66;
    r63 = r63 * r70;
    r83 = r16 * r70;
    r83 = r83 * r84;
    r85 = r63 + r83;
    r85 = fma(r6, r85, r30 * r76);
    r76 = r18 * r29;
    r76 = fma(r18, r69, r84 * r76);
    r86 = r25 * r26;
    r86 = r86 * r66;
    r87 = r25 * r20;
    r87 = fma(r74, r87, r86);
    r76 = r76 + r87;
    r85 = fma(r7, r76, r85);
    r76 = r25 * r85;
    r88 = r25 * r5;
    r89 = r26 * r18;
    r89 = fma(r62, r89, r64);
    r64 = r25 * r20;
    r64 = r64 * r84;
    r90 = r18 * r29;
    r89 = fma(r74, r90, r89);
    r89 = r89 + r64;
    r84 = fma(r84, r32, r25 * r69);
    r84 = r84 + r87;
    r84 = fma(r6, r84, r30 * r89);
    r89 = r74 * r78;
    r83 = r83 + r89;
    r84 = fma(r7, r83, r84);
    r88 = r88 * r84;
    r88 = fma(r48, r88, r55 * r76);
    r76 = r18 * r20;
    r76 = fma(r62, r76, r82);
    r76 = r76 + r71;
    r64 = fma(r74, r32, r64);
    r64 = r64 + r61;
    r64 = fma(r7, r64, r6 * r76);
    r89 = r63 + r89;
    r64 = fma(r30, r89, r64);
    r89 = r64 * r47;
    r88 = fma(r34, r89, r88);
    r88 = fma(r64, r1, r88);
    r89 = r0 * r88;
    r63 = r64 * r55;
    r63 = fma(r65, r63, r46 * r89);
    r89 = r19 * r85;
    r63 = fma(r51, r89, r63);
    r89 = r19 * r84;
    r76 = r5 * r88;
    r76 = fma(r46, r76, r51 * r89);
    r76 = fma(r64, r72, r76);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r63, r76);
    r89 = r13 * r11;
    r22 = fma(r38, r22, r58 * r89);
    r22 = fma(r15, r60, r22);
    r22 = fma(r58, r21, r22);
    r78 = r22 * r78;
    r57 = r70 * r57;
    r21 = r78 + r57;
    r58 = r25 * r16;
    r58 = r58 * r22;
    r86 = r86 + r58;
    r60 = r18 * r20;
    r86 = fma(r74, r60, r86);
    r38 = r18 * r29;
    r86 = fma(r41, r38, r86);
    r86 = fma(r6, r86, r30 * r21);
    r21 = r25 * r26;
    r21 = fma(r22, r32, r74 * r21);
    r21 = r21 + r81;
    r86 = fma(r7, r21, r86);
    r21 = r86 * r55;
    r82 = r75 + r82;
    r75 = r25 * r20;
    r75 = r75 * r22;
    r38 = r16 * r18;
    r82 = fma(r74, r38, r82);
    r82 = r82 + r75;
    r66 = r16 * r66;
    r66 = r66 * r70;
    r57 = r66 + r57;
    r57 = fma(r6, r57, r7 * r82);
    r32 = fma(r41, r32, r58);
    r32 = r32 + r87;
    r57 = fma(r30, r32, r57);
    r32 = r19 * r57;
    r32 = fma(r51, r32, r65 * r21);
    r21 = r25 * r5;
    r75 = r73 + r75;
    r75 = r75 + r77;
    r77 = r26 * r18;
    r73 = r18 * r29;
    r73 = fma(r22, r73, r74 * r77);
    r73 = r73 + r81;
    r73 = fma(r30, r73, r6 * r75);
    r78 = r66 + r78;
    r73 = fma(r7, r78, r73);
    r21 = r21 * r73;
    r21 = fma(r86, r1, r48 * r21);
    r78 = r25 * r57;
    r21 = fma(r55, r78, r21);
    r7 = r86 * r47;
    r21 = fma(r34, r7, r21);
    r7 = r0 * r21;
    r32 = fma(r46, r7, r32);
    r7 = r5 * r21;
    r7 = fma(r86, r72, r46 * r7);
    r78 = r19 * r73;
    r7 = fma(r51, r78, r7);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r32, r7);
    r78 = r36 * r55;
    r66 = r25 * r56;
    r66 = r66 * r5;
    r30 = r36 * r47;
    r30 = fma(r34, r30, r48 * r66);
    r66 = r25 * r43;
    r30 = fma(r55, r66, r30);
    r30 = fma(r36, r1, r30);
    r66 = r0 * r30;
    r66 = fma(r46, r66, r65 * r78);
    r78 = r43 * r19;
    r66 = fma(r51, r78, r66);
    r78 = r56 * r19;
    r75 = r5 * r30;
    r75 = fma(r46, r75, r51 * r78);
    r75 = fma(r36, r72, r75);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r66, r75);
    r78 = r40 * r19;
    r6 = r50 * r55;
    r6 = fma(r65, r6, r51 * r78);
    r78 = r50 * r47;
    r81 = r25 * r44;
    r81 = r81 * r5;
    r81 = fma(r48, r81, r34 * r78);
    r78 = r25 * r40;
    r81 = fma(r55, r78, r81);
    r81 = fma(r50, r1, r81);
    r78 = r0 * r81;
    r6 = fma(r46, r78, r6);
    r78 = r5 * r81;
    r77 = r44 * r19;
    r77 = fma(r51, r77, r46 * r78);
    r77 = fma(r50, r72, r77);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r6, r77);
    r78 = r25 * r35;
    r78 = r78 * r5;
    r22 = r42 * r47;
    r22 = fma(r34, r22, r48 * r78);
    r78 = r25 * r37;
    r22 = fma(r55, r78, r22);
    r22 = fma(r42, r1, r22);
    r78 = r0 * r22;
    r74 = r37 * r19;
    r74 = fma(r51, r74, r46 * r78);
    r78 = r42 * r55;
    r74 = fma(r65, r78, r74);
    r78 = r5 * r22;
    r78 = fma(r46, r78, r42 * r72);
    r87 = r35 * r19;
    r78 = fma(r51, r87, r78);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r74, r78);
    r87 = r4 * r3;
    r41 = r4 * r2;
    r41 = fma(r80, r41, r54 * r87);
    r87 = r4 * r2;
    r58 = r4 * r3;
    r58 = fma(r76, r58, r63 * r87);
    WriteSum2<double, double>((double*)inout_shared, r41, r58);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = r4 * r3;
    r41 = r4 * r2;
    r41 = fma(r32, r41, r7 * r58);
    r58 = r4 * r2;
    r87 = r4 * r3;
    r87 = fma(r75, r87, r66 * r58);
    WriteSum2<double, double>((double*)inout_shared, r41, r87);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r87 = r4 * r3;
    r41 = r4 * r2;
    r41 = fma(r6, r41, r77 * r87);
    r87 = r4 * r2;
    r58 = r4 * r3;
    r58 = fma(r78, r58, r74 * r87);
    WriteSum2<double, double>((double*)inout_shared, r41, r58);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = fma(r80, r80, r54 * r54);
    r41 = fma(r76, r76, r63 * r63);
    WriteSum2<double, double>((double*)inout_shared, r58, r41);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r32, r32, r7 * r7);
    r58 = fma(r66, r66, r75 * r75);
    WriteSum2<double, double>((double*)inout_shared, r41, r58);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = fma(r6, r6, r77 * r77);
    r41 = fma(r74, r74, r78 * r78);
    WriteSum2<double, double>((double*)inout_shared, r58, r41);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r80, r63, r54 * r76);
    r58 = fma(r54, r7, r80 * r32);
    WriteSum2<double, double>((double*)inout_shared, r41, r58);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = fma(r54, r75, r80 * r66);
    r41 = fma(r80, r6, r54 * r77);
    WriteSum2<double, double>((double*)inout_shared, r58, r41);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = fma(r80, r74, r54 * r78);
    r54 = fma(r76, r7, r63 * r32);
    WriteSum2<double, double>((double*)inout_shared, r80, r54);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = fma(r76, r75, r63 * r66);
    r80 = fma(r76, r77, r63 * r6);
    WriteSum2<double, double>((double*)inout_shared, r54, r80);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = fma(r76, r78, r63 * r74);
    r63 = fma(r32, r66, r7 * r75);
    WriteSum2<double, double>((double*)inout_shared, r76, r63);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = fma(r32, r6, r7 * r77);
    r7 = fma(r7, r78, r32 * r74);
    WriteSum2<double, double>((double*)inout_shared, r63, r7);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = fma(r75, r77, r66 * r6);
    r75 = fma(r75, r78, r66 * r74);
    WriteSum2<double, double>((double*)inout_shared, r7, r75);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = fma(r77, r78, r6 * r74);
    WriteSum1<double, double>((double*)inout_shared, r78);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = r0 * r19;
    r78 = r78 * r53;
    r77 = r5 * r19;
    r77 = r77 * r53;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        0 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r78,
        r77);
    r77 = r0 * r39;
    r77 = r77 * r51;
    r78 = r5 * r39;
    r78 = r78 * r51;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        2 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r77,
        r78);
    r78 = r0 * r2;
    r78 = r78 * r53;
    r77 = r5 * r3;
    r77 = r77 * r53;
    r77 = fma(r67, r77, r67 * r78);
    r78 = r4 * r0;
    r78 = r78 * r39;
    r78 = r78 * r2;
    r67 = r4 * r5;
    r67 = r67 * r39;
    r67 = r67 * r3;
    r67 = fma(r51, r67, r51 * r78);
    WriteSum2<double, double>((double*)inout_shared, r77, r67);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            0 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = r0 * r19;
    r67 = r67 * r19;
    r77 = r19 * r19;
    r77 = r77 * r48;
    r77 = fma(r34, r77, r55 * r67);
    r67 = r45 * r45;
    r78 = r39 * r39;
    r67 = r67 * r0;
    r67 = r67 * r55;
    r53 = r45 * r78;
    r74 = r45 * r48;
    r74 = r74 * r34;
    r53 = fma(r74, r53, r78 * r67);
    WriteSum2<double, double>((double*)inout_shared, r77, r53);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            0 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = r45 * r0;
    r53 = r53 * r39;
    r53 = r53 * r19;
    r77 = r39 * r19;
    r77 = fma(r74, r77, r55 * r53);
    WriteSum1<double, double>((double*)inout_shared, r77);
  };
  FlushSumShared<1, double>(out_focal_and_extra_precond_tril,
                            0 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = r49 * r55;
    r53 = r24 * r19;
    r53 = fma(r51, r53, r65 * r77);
    r77 = r25 * r24;
    r74 = r49 * r47;
    r74 = fma(r34, r74, r55 * r77);
    r77 = r25 * r27;
    r77 = r77 * r5;
    r74 = fma(r48, r77, r74);
    r74 = fma(r49, r1, r74);
    r77 = r0 * r74;
    r53 = fma(r46, r77, r53);
    r77 = r27 * r19;
    r77 = fma(r51, r77, r49 * r72);
    r67 = r5 * r74;
    r77 = fma(r46, r67, r77);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r53,
                                             r77);
    r67 = r52 * r55;
    r6 = r28 * r19;
    r6 = fma(r51, r6, r65 * r67);
    r67 = r25 * r28;
    r67 = fma(r55, r67, r52 * r1);
    r75 = r25 * r17;
    r75 = r75 * r5;
    r67 = fma(r48, r75, r67);
    r7 = r52 * r47;
    r67 = fma(r34, r7, r67);
    r7 = r0 * r67;
    r6 = fma(r46, r7, r6);
    r7 = r17 * r19;
    r7 = fma(r52, r72, r51 * r7);
    r75 = r5 * r67;
    r7 = fma(r46, r75, r7);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 2 * out_point_jac_num_alloc, global_thread_idx, r6, r7);
    r75 = r23 * r55;
    r66 = r25 * r33;
    r63 = r23 * r47;
    r63 = fma(r34, r63, r55 * r66);
    r66 = r25 * r31;
    r66 = r66 * r5;
    r63 = fma(r48, r66, r63);
    r63 = fma(r23, r1, r63);
    r66 = r0 * r63;
    r66 = fma(r46, r66, r65 * r75);
    r75 = r33 * r19;
    r66 = fma(r51, r75, r66);
    r75 = r5 * r63;
    r72 = fma(r23, r72, r46 * r75);
    r75 = r31 * r19;
    r72 = fma(r51, r75, r72);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r66,
                                             r72);
    r75 = r4 * r2;
    r51 = r4 * r3;
    r51 = fma(r77, r51, r53 * r75);
    r75 = r4 * r3;
    r46 = r4 * r2;
    r46 = fma(r6, r46, r7 * r75);
    WriteSum2<double, double>((double*)inout_shared, r51, r46);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r4 * r3;
    r51 = r4 * r2;
    r51 = fma(r66, r51, r72 * r46);
    WriteSum1<double, double>((double*)inout_shared, r51);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = fma(r77, r77, r53 * r53);
    r46 = fma(r7, r7, r6 * r6);
    WriteSum2<double, double>((double*)inout_shared, r51, r46);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fma(r72, r72, r66 * r66);
    WriteSum1<double, double>((double*)inout_shared, r46);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fma(r77, r7, r53 * r6);
    r53 = fma(r53, r66, r77 * r72);
    WriteSum2<double, double>((double*)inout_shared, r46, r53);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = fma(r6, r66, r7 * r72);
    WriteSum1<double, double>((double*)inout_shared, r66);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
}

void SimpleRadialSplitFixedPrincipalPointResJac(
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
  SimpleRadialSplitFixedPrincipalPointResJacKernel<<<n_blocks, 1024>>>(
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