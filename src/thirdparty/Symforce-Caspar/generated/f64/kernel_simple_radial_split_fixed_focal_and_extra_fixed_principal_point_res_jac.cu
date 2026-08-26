#include "kernel_simple_radial_split_fixed_focal_and_extra_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointResJacKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
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
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87;

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
    r45 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r46);
    r47 = r18 * r20;
    r47 = fma(r29, r47, r31);
    r46 = fma(r6, r47, r46);
    r36 = fma(r18, r36, r35);
    r42 = r19 + r42;
    r35 = r14 * r14;
    r35 = r35 * r18;
    r42 = r42 + r35;
    r31 = r15 * r10;
    r31 = r31 * r25;
    r48 = r14 * r11;
    r48 = fma(r25, r48, r31);
    r49 = r25 * r16;
    r49 = r49 * r20;
    r50 = fma(r26, r32, r49);
    r51 = r26 * r26;
    r51 = r51 * r18;
    r23 = r51 + r23;
    r46 = fma(r38, r36, r46);
    r46 = fma(r34, r42, r46);
    r46 = fma(r39, r48, r46);
    r46 = fma(r7, r50, r46);
    r46 = fma(r30, r23, r46);
    r52 = copysign(1.0, r46);
    r52 = fma(r45, r52, r46);
    r45 = 1.0 / r52;
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            0 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r46,
                                            r53);
    r54 = r52 * r52;
    r55 = 1.0 / r54;
    r56 = r0 * r55;
    r27 = fma(r16, r32, r27);
    r5 = fma(r6, r27, r5);
    r57 = r10 * r11;
    r57 = fma(r25, r57, r41);
    r44 = r19 + r44;
    r44 = r44 + r35;
    r35 = r14 * r11;
    r35 = fma(r18, r35, r31);
    r31 = r26 * r18;
    r31 = fma(r29, r31, r49);
    r17 = r19 + r17;
    r17 = r17 + r51;
    r5 = fma(r38, r57, r5);
    r5 = fma(r39, r44, r5);
    r5 = fma(r34, r35, r5);
    r5 = fma(r30, r31, r5);
    r5 = fma(r7, r17, r5);
    r34 = r5 * r5;
    r39 = fma(r55, r34, r0 * r56);
    r39 = fma(r53, r39, r19);
    r39 = r46 * r39;
    r19 = r45 * r39;
    r2 = fma(r0, r19, r2);
    r3 = fma(r3, r4, r1);
    r3 = fma(r5, r19, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = r46 * r53;
    r38 = r12 * r11;
    r51 = -5.00000000000000000e-01;
    r49 = r9 * r14;
    r49 = fma(r51, r49, r51 * r38);
    r38 = r8 * r15;
    r49 = fma(r51, r38, r49);
    r41 = r13 * r10;
    r58 = 5.00000000000000000e-01;
    r49 = fma(r58, r41, r49);
    r41 = r20 * r49;
    r38 = r8 * r11;
    r59 = r13 * r14;
    r59 = fma(r58, r59, r58 * r38);
    r38 = r12 * r15;
    r59 = fma(r51, r38, r59);
    r60 = r9 * r58;
    r59 = fma(r10, r60, r59);
    r38 = fma(r59, r32, r25 * r41);
    r61 = r25 * r26;
    r62 = r9 * r15;
    r63 = r13 * r51;
    r62 = fma(r11, r63, r51 * r62);
    r62 = fma(r58, r22, r62);
    r62 = fma(r51, r21, r62);
    r64 = r25 * r16;
    r65 = r12 * r14;
    r66 = r8 * r10;
    r66 = fma(r51, r66, r51 * r65);
    r66 = fma(r11, r60, r66);
    r66 = fma(r15, r63, r66);
    r64 = r64 * r66;
    r61 = fma(r62, r61, r64);
    r38 = r38 + r61;
    r65 = r25 * r20;
    r65 = r65 * r66;
    r67 = r25 * r26;
    r67 = r67 * r59;
    r68 = r65 + r67;
    r69 = r16 * r18;
    r68 = fma(r49, r69, r68);
    r70 = r18 * r29;
    r68 = fma(r62, r70, r68);
    r68 = fma(r7, r68, r30 * r38);
    r38 = r20 * r59;
    r70 = -4.00000000000000000e+00;
    r38 = r38 * r70;
    r69 = r16 * r62;
    r71 = r70 * r69;
    r72 = r38 + r71;
    r68 = fma(r6, r72, r68);
    r72 = r25 * r68;
    r73 = r25 * r20;
    r73 = r73 * r62;
    r74 = r25 * r16;
    r74 = fma(r59, r74, r73);
    r75 = r25 * r26;
    r75 = r75 * r49;
    r76 = r66 * r32;
    r77 = r75 + r76;
    r78 = r74 + r77;
    r79 = r18 * r29;
    r79 = fma(r18, r41, r59 * r79);
    r79 = r79 + r61;
    r79 = fma(r6, r79, r7 * r78);
    r78 = r26 * r70;
    r59 = r66 * r78;
    r38 = r38 + r59;
    r79 = fma(r30, r38, r79);
    r38 = r0 * r0;
    r54 = r52 * r54;
    r54 = 1.0 / r54;
    r54 = r18 * r54;
    r38 = r38 * r54;
    r72 = fma(r79, r38, r56 * r72);
    r52 = r79 * r54;
    r72 = fma(r34, r52, r72);
    r80 = r25 * r5;
    r81 = r26 * r18;
    r82 = r18 * r29;
    r82 = r82 * r66;
    r81 = fma(r49, r81, r82);
    r81 = r81 + r74;
    r59 = r71 + r59;
    r59 = fma(r7, r59, r30 * r81);
    r67 = fma(r62, r32, r67);
    r81 = r25 * r16;
    r81 = fma(r49, r81, r65);
    r67 = r67 + r81;
    r59 = fma(r6, r67, r59);
    r80 = r80 * r59;
    r72 = fma(r55, r80, r72);
    r1 = r1 * r72;
    r1 = r1 * r45;
    r72 = fma(r68, r19, r0 * r1);
    r80 = r4 * r39;
    r80 = r80 * r56;
    r72 = fma(r79, r80, r72);
    r59 = fma(r59, r19, r5 * r1);
    r1 = r4 * r5;
    r1 = r1 * r79;
    r1 = r1 * r55;
    r59 = fma(r39, r1, r59);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r72, r59);
    r1 = r46 * r53;
    r76 = r73 + r76;
    r73 = r25 * r16;
    r52 = r8 * r11;
    r67 = r12 * r15;
    r67 = fma(r58, r67, r51 * r52);
    r52 = r9 * r10;
    r67 = fma(r51, r52, r67);
    r67 = fma(r14, r63, r67);
    r73 = r73 * r67;
    r52 = r25 * r26;
    r65 = r12 * r11;
    r71 = r8 * r15;
    r71 = fma(r58, r71, r58 * r65);
    r71 = fma(r14, r60, r71);
    r71 = fma(r10, r63, r71);
    r52 = fma(r71, r52, r73);
    r76 = r76 + r52;
    r63 = r20 * r66;
    r63 = r63 * r70;
    r65 = r16 * r70;
    r65 = r65 * r71;
    r74 = r63 + r65;
    r74 = fma(r6, r74, r30 * r76);
    r76 = r18 * r29;
    r76 = fma(r18, r69, r71 * r76);
    r83 = r25 * r26;
    r83 = r83 * r66;
    r84 = r25 * r20;
    r84 = fma(r67, r84, r83);
    r76 = r76 + r84;
    r74 = fma(r7, r76, r74);
    r76 = r25 * r74;
    r85 = r25 * r5;
    r86 = r26 * r18;
    r86 = fma(r62, r86, r64);
    r64 = r25 * r20;
    r64 = r64 * r71;
    r87 = r18 * r29;
    r86 = fma(r67, r87, r86);
    r86 = r86 + r64;
    r71 = fma(r71, r32, r25 * r69);
    r71 = r71 + r84;
    r71 = fma(r6, r71, r30 * r86);
    r86 = r67 * r78;
    r65 = r65 + r86;
    r71 = fma(r7, r65, r71);
    r85 = r85 * r71;
    r85 = fma(r55, r85, r56 * r76);
    r76 = r18 * r20;
    r76 = fma(r62, r76, r82);
    r76 = r76 + r52;
    r64 = fma(r67, r32, r64);
    r64 = r64 + r61;
    r64 = fma(r7, r64, r6 * r76);
    r86 = r63 + r86;
    r64 = fma(r30, r86, r64);
    r86 = r64 * r54;
    r85 = fma(r34, r86, r85);
    r85 = fma(r64, r38, r85);
    r1 = r1 * r0;
    r1 = r1 * r85;
    r1 = fma(r64, r80, r45 * r1);
    r1 = fma(r74, r19, r1);
    r86 = r46 * r53;
    r86 = r86 * r5;
    r86 = r86 * r85;
    r86 = fma(r45, r86, r71 * r19);
    r71 = r4 * r5;
    r71 = r71 * r64;
    r71 = r71 * r55;
    r86 = fma(r39, r71, r86);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r1, r86);
    r71 = r13 * r11;
    r22 = fma(r51, r22, r58 * r71);
    r22 = fma(r15, r60, r22);
    r22 = fma(r58, r21, r22);
    r78 = r22 * r78;
    r41 = r70 * r41;
    r21 = r78 + r41;
    r58 = r25 * r16;
    r58 = r58 * r22;
    r83 = r83 + r58;
    r60 = r18 * r20;
    r83 = fma(r67, r60, r83);
    r51 = r18 * r29;
    r83 = fma(r49, r51, r83);
    r83 = fma(r6, r83, r30 * r21);
    r21 = r25 * r26;
    r21 = fma(r22, r32, r67 * r21);
    r21 = r21 + r81;
    r83 = fma(r7, r21, r83);
    r82 = r75 + r82;
    r75 = r25 * r20;
    r75 = r75 * r22;
    r21 = r16 * r18;
    r82 = fma(r67, r21, r82);
    r82 = r82 + r75;
    r66 = r16 * r66;
    r66 = r66 * r70;
    r41 = r66 + r41;
    r41 = fma(r6, r41, r7 * r82);
    r32 = fma(r49, r32, r58);
    r32 = r32 + r84;
    r41 = fma(r30, r32, r41);
    r32 = fma(r41, r19, r83 * r80);
    r84 = r46 * r53;
    r49 = r25 * r5;
    r75 = r73 + r75;
    r75 = r75 + r77;
    r77 = r26 * r18;
    r73 = r18 * r29;
    r73 = fma(r22, r73, r67 * r77);
    r73 = r73 + r81;
    r73 = fma(r30, r73, r6 * r75);
    r78 = r66 + r78;
    r73 = fma(r7, r78, r73);
    r49 = r49 * r73;
    r49 = fma(r83, r38, r55 * r49);
    r78 = r25 * r41;
    r49 = fma(r56, r78, r49);
    r7 = r83 * r54;
    r49 = fma(r34, r7, r49);
    r84 = r84 * r0;
    r84 = r84 * r49;
    r32 = fma(r45, r84, r32);
    r84 = r46 * r53;
    r84 = r84 * r5;
    r84 = r84 * r49;
    r49 = r4 * r5;
    r49 = r49 * r83;
    r49 = r49 * r55;
    r49 = fma(r39, r49, r45 * r84);
    r49 = fma(r73, r19, r49);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r32, r49);
    r73 = r46 * r53;
    r84 = r25 * r57;
    r84 = r84 * r5;
    r7 = r36 * r54;
    r7 = fma(r34, r7, r55 * r84);
    r84 = r25 * r43;
    r7 = fma(r56, r84, r7);
    r7 = fma(r36, r38, r7);
    r73 = r73 * r0;
    r73 = r73 * r7;
    r73 = fma(r45, r73, r36 * r80);
    r73 = fma(r43, r19, r73);
    r84 = r46 * r53;
    r84 = r84 * r5;
    r84 = r84 * r7;
    r84 = fma(r45, r84, r57 * r19);
    r7 = r4 * r36;
    r7 = r7 * r5;
    r7 = r7 * r55;
    r84 = fma(r39, r7, r84);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r73, r84);
    r7 = fma(r48, r80, r40 * r19);
    r78 = r46 * r53;
    r66 = r48 * r54;
    r30 = r25 * r44;
    r30 = r30 * r5;
    r30 = fma(r55, r30, r34 * r66);
    r66 = r25 * r40;
    r30 = fma(r56, r66, r30);
    r30 = fma(r48, r38, r30);
    r78 = r78 * r0;
    r78 = r78 * r30;
    r7 = fma(r45, r78, r7);
    r78 = r46 * r53;
    r78 = r78 * r5;
    r78 = r78 * r30;
    r78 = fma(r44, r19, r45 * r78);
    r30 = r4 * r48;
    r30 = r30 * r5;
    r30 = r30 * r55;
    r78 = fma(r39, r30, r78);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r7, r78);
    r30 = r46 * r53;
    r66 = r25 * r35;
    r66 = r66 * r5;
    r75 = r42 * r54;
    r75 = fma(r34, r75, r55 * r66);
    r66 = r25 * r37;
    r75 = fma(r56, r66, r75);
    r75 = fma(r42, r38, r75);
    r30 = r30 * r0;
    r30 = r30 * r75;
    r30 = fma(r37, r19, r45 * r30);
    r30 = fma(r42, r80, r30);
    r66 = r4 * r42;
    r66 = r66 * r5;
    r66 = r66 * r55;
    r6 = r46 * r53;
    r6 = r6 * r5;
    r6 = r6 * r75;
    r6 = fma(r45, r6, r39 * r66);
    r6 = fma(r35, r19, r6);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r30, r6);
    r66 = r4 * r3;
    r75 = r4 * r2;
    r75 = fma(r72, r75, r59 * r66);
    r66 = r4 * r2;
    r81 = r4 * r3;
    r81 = fma(r86, r81, r1 * r66);
    WriteSum2<double, double>((double*)inout_shared, r75, r81);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = r4 * r3;
    r75 = r4 * r2;
    r75 = fma(r32, r75, r49 * r81);
    r81 = r4 * r2;
    r66 = r4 * r3;
    r66 = fma(r84, r66, r73 * r81);
    WriteSum2<double, double>((double*)inout_shared, r75, r66);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = r4 * r3;
    r75 = r4 * r2;
    r75 = fma(r7, r75, r78 * r66);
    r66 = r4 * r2;
    r81 = r4 * r3;
    r81 = fma(r6, r81, r30 * r66);
    WriteSum2<double, double>((double*)inout_shared, r75, r81);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = fma(r72, r72, r59 * r59);
    r75 = fma(r86, r86, r1 * r1);
    WriteSum2<double, double>((double*)inout_shared, r81, r75);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = fma(r32, r32, r49 * r49);
    r81 = fma(r73, r73, r84 * r84);
    WriteSum2<double, double>((double*)inout_shared, r75, r81);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = fma(r7, r7, r78 * r78);
    r75 = fma(r30, r30, r6 * r6);
    WriteSum2<double, double>((double*)inout_shared, r81, r75);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = fma(r72, r1, r59 * r86);
    r81 = fma(r59, r49, r72 * r32);
    WriteSum2<double, double>((double*)inout_shared, r75, r81);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = fma(r59, r84, r72 * r73);
    r75 = fma(r72, r7, r59 * r78);
    WriteSum2<double, double>((double*)inout_shared, r81, r75);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fma(r72, r30, r59 * r6);
    r59 = fma(r86, r49, r1 * r32);
    WriteSum2<double, double>((double*)inout_shared, r72, r59);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = fma(r86, r84, r1 * r73);
    r72 = fma(r86, r78, r1 * r7);
    WriteSum2<double, double>((double*)inout_shared, r59, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r86 = fma(r86, r6, r1 * r30);
    r1 = fma(r32, r73, r49 * r84);
    WriteSum2<double, double>((double*)inout_shared, r86, r1);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fma(r32, r7, r49 * r78);
    r49 = fma(r49, r6, r32 * r30);
    WriteSum2<double, double>((double*)inout_shared, r1, r49);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = fma(r84, r78, r73 * r7);
    r84 = fma(r84, r6, r73 * r30);
    WriteSum2<double, double>((double*)inout_shared, r49, r84);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r78, r6, r7 * r30);
    WriteSum1<double, double>((double*)inout_shared, r6);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r24, r19, r47 * r80);
    r78 = r46 * r53;
    r30 = r25 * r24;
    r7 = r47 * r54;
    r7 = fma(r34, r7, r56 * r30);
    r30 = r25 * r27;
    r30 = r30 * r5;
    r7 = fma(r55, r30, r7);
    r7 = fma(r47, r38, r7);
    r78 = r78 * r0;
    r78 = r78 * r7;
    r6 = fma(r45, r78, r6);
    r78 = r4 * r47;
    r78 = r78 * r5;
    r78 = r78 * r55;
    r78 = fma(r27, r19, r39 * r78);
    r30 = r46 * r53;
    r30 = r30 * r5;
    r30 = r30 * r7;
    r78 = fma(r45, r30, r78);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r6, r78);
    r30 = fma(r28, r19, r50 * r80);
    r7 = r46 * r53;
    r84 = r25 * r28;
    r84 = fma(r56, r84, r50 * r38);
    r49 = r25 * r17;
    r49 = r49 * r5;
    r84 = fma(r55, r49, r84);
    r73 = r50 * r54;
    r84 = fma(r34, r73, r84);
    r7 = r7 * r0;
    r7 = r7 * r84;
    r30 = fma(r45, r7, r30);
    r7 = r4 * r50;
    r7 = r7 * r5;
    r7 = r7 * r55;
    r7 = fma(r39, r7, r17 * r19);
    r73 = r46 * r53;
    r73 = r73 * r5;
    r73 = r73 * r84;
    r7 = fma(r45, r73, r7);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 2 * out_point_jac_num_alloc, global_thread_idx, r30, r7);
    r73 = r46 * r53;
    r84 = r25 * r33;
    r49 = r23 * r54;
    r49 = fma(r34, r49, r56 * r84);
    r84 = r25 * r31;
    r84 = r84 * r5;
    r49 = fma(r55, r84, r49);
    r49 = fma(r23, r38, r49);
    r73 = r73 * r0;
    r73 = r73 * r49;
    r73 = fma(r45, r73, r23 * r80);
    r73 = fma(r33, r19, r73);
    r80 = r46 * r53;
    r80 = r80 * r5;
    r80 = r80 * r49;
    r49 = r4 * r23;
    r49 = r49 * r5;
    r49 = r49 * r55;
    r49 = fma(r39, r49, r45 * r80);
    r49 = fma(r31, r19, r49);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r73,
                                             r49);
    r19 = r4 * r2;
    r80 = r4 * r3;
    r80 = fma(r78, r80, r6 * r19);
    r19 = r4 * r3;
    r39 = r4 * r2;
    r39 = fma(r30, r39, r7 * r19);
    WriteSum2<double, double>((double*)inout_shared, r80, r39);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r4 * r3;
    r80 = r4 * r2;
    r80 = fma(r73, r80, r49 * r39);
    WriteSum1<double, double>((double*)inout_shared, r80);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = fma(r78, r78, r6 * r6);
    r39 = fma(r7, r7, r30 * r30);
    WriteSum2<double, double>((double*)inout_shared, r80, r39);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r49, r49, r73 * r73);
    WriteSum1<double, double>((double*)inout_shared, r39);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r78, r7, r6 * r30);
    r6 = fma(r6, r73, r78 * r49);
    WriteSum2<double, double>((double*)inout_shared, r39, r6);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r30, r73, r7 * r49);
    WriteSum1<double, double>((double*)inout_shared, r73);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
}

void SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointResJac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
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
  SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointResJacKernel<<<n_blocks,
                                                                       1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
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