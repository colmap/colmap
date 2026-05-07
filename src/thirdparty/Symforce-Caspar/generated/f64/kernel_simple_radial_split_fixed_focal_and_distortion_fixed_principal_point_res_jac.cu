#include "kernel_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointResJacKernel(
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
        double* focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc,
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
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86;

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
    ReadIdx2<1024, double, double, double2>(focal_and_distortion,
                                            0 * focal_and_distortion_num_alloc,
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
    r1 = r12 * r11;
    r38 = -5.00000000000000000e-01;
    r51 = r9 * r14;
    r51 = fma(r38, r51, r38 * r1);
    r1 = r8 * r15;
    r51 = fma(r38, r1, r51);
    r49 = r13 * r10;
    r41 = 5.00000000000000000e-01;
    r51 = fma(r41, r49, r51);
    r49 = r20 * r51;
    r1 = r8 * r11;
    r58 = r13 * r14;
    r58 = fma(r41, r58, r41 * r1);
    r1 = r12 * r15;
    r58 = fma(r38, r1, r58);
    r59 = r9 * r41;
    r58 = fma(r10, r59, r58);
    r1 = fma(r58, r32, r25 * r49);
    r60 = r25 * r26;
    r61 = r9 * r15;
    r62 = r13 * r38;
    r61 = fma(r11, r62, r38 * r61);
    r61 = fma(r41, r22, r61);
    r61 = fma(r38, r21, r61);
    r63 = r25 * r16;
    r64 = r12 * r14;
    r65 = r8 * r10;
    r65 = fma(r38, r65, r38 * r64);
    r65 = fma(r11, r59, r65);
    r65 = fma(r15, r62, r65);
    r63 = r63 * r65;
    r60 = fma(r61, r60, r63);
    r1 = r1 + r60;
    r64 = r25 * r20;
    r64 = r64 * r65;
    r66 = r25 * r26;
    r66 = r66 * r58;
    r67 = r64 + r66;
    r68 = r16 * r18;
    r67 = fma(r51, r68, r67);
    r69 = r18 * r29;
    r67 = fma(r61, r69, r67);
    r67 = fma(r7, r67, r30 * r1);
    r1 = r20 * r58;
    r69 = -4.00000000000000000e+00;
    r1 = r1 * r69;
    r68 = r16 * r61;
    r70 = r69 * r68;
    r71 = r1 + r70;
    r67 = fma(r6, r71, r67);
    r71 = r25 * r20;
    r71 = r71 * r61;
    r72 = r25 * r16;
    r72 = fma(r58, r72, r71);
    r73 = r25 * r26;
    r73 = r73 * r51;
    r74 = r65 * r32;
    r75 = r73 + r74;
    r76 = r72 + r75;
    r77 = r18 * r29;
    r77 = fma(r18, r49, r58 * r77);
    r77 = r77 + r60;
    r77 = fma(r6, r77, r7 * r76);
    r76 = r26 * r69;
    r58 = r65 * r76;
    r1 = r1 + r58;
    r77 = fma(r30, r1, r77);
    r1 = r4 * r39;
    r1 = r1 * r56;
    r78 = fma(r77, r1, r67 * r19);
    r79 = r46 * r53;
    r80 = r25 * r67;
    r81 = r0 * r0;
    r54 = r52 * r54;
    r54 = 1.0 / r54;
    r54 = r18 * r54;
    r81 = r81 * r54;
    r80 = fma(r77, r81, r56 * r80);
    r52 = r77 * r54;
    r80 = fma(r34, r52, r80);
    r82 = r25 * r5;
    r83 = r26 * r18;
    r84 = r18 * r29;
    r84 = r84 * r65;
    r83 = fma(r51, r83, r84);
    r83 = r83 + r72;
    r58 = r70 + r58;
    r58 = fma(r7, r58, r30 * r83);
    r66 = fma(r61, r32, r66);
    r83 = r25 * r16;
    r83 = fma(r51, r83, r64);
    r66 = r66 + r83;
    r58 = fma(r6, r66, r58);
    r82 = r82 * r58;
    r80 = fma(r55, r82, r80);
    r79 = r79 * r80;
    r79 = r79 * r45;
    r78 = fma(r0, r79, r78);
    r80 = r4 * r5;
    r80 = r80 * r77;
    r80 = r80 * r55;
    r79 = fma(r5, r79, r39 * r80);
    r79 = fma(r58, r19, r79);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r78, r79);
    r58 = r18 * r20;
    r58 = fma(r61, r58, r84);
    r80 = r25 * r16;
    r82 = r8 * r11;
    r52 = r12 * r15;
    r52 = fma(r41, r52, r38 * r82);
    r82 = r9 * r10;
    r52 = fma(r38, r82, r52);
    r52 = fma(r14, r62, r52);
    r80 = r80 * r52;
    r82 = r25 * r26;
    r66 = r12 * r11;
    r64 = r8 * r15;
    r64 = fma(r41, r64, r41 * r66);
    r64 = fma(r14, r59, r64);
    r64 = fma(r10, r62, r64);
    r82 = fma(r64, r82, r80);
    r58 = r58 + r82;
    r62 = r25 * r20;
    r62 = r62 * r64;
    r66 = fma(r52, r32, r62);
    r66 = r66 + r60;
    r66 = fma(r7, r66, r6 * r58);
    r58 = r20 * r65;
    r58 = r58 * r69;
    r60 = r52 * r76;
    r70 = r58 + r60;
    r66 = fma(r30, r70, r66);
    r74 = r71 + r74;
    r74 = r74 + r82;
    r82 = r16 * r69;
    r82 = r82 * r64;
    r58 = r58 + r82;
    r58 = fma(r6, r58, r30 * r74);
    r74 = r18 * r29;
    r74 = fma(r18, r68, r64 * r74);
    r71 = r25 * r26;
    r71 = r71 * r65;
    r70 = r25 * r20;
    r70 = fma(r52, r70, r71);
    r74 = r74 + r70;
    r58 = fma(r7, r74, r58);
    r74 = fma(r58, r19, r66 * r1);
    r72 = r46 * r53;
    r85 = r25 * r58;
    r86 = r25 * r5;
    r62 = r63 + r62;
    r63 = r26 * r18;
    r62 = fma(r61, r63, r62);
    r61 = r18 * r29;
    r62 = fma(r52, r61, r62);
    r64 = fma(r64, r32, r25 * r68);
    r64 = r64 + r70;
    r64 = fma(r6, r64, r30 * r62);
    r60 = r82 + r60;
    r64 = fma(r7, r60, r64);
    r86 = r86 * r64;
    r86 = fma(r55, r86, r56 * r85);
    r85 = r66 * r54;
    r86 = fma(r34, r85, r86);
    r86 = fma(r66, r81, r86);
    r72 = r72 * r0;
    r72 = r72 * r86;
    r74 = fma(r45, r72, r74);
    r72 = r46 * r53;
    r72 = r72 * r5;
    r72 = r72 * r86;
    r64 = fma(r64, r19, r45 * r72);
    r72 = r4 * r5;
    r72 = r72 * r66;
    r72 = r72 * r55;
    r64 = fma(r39, r72, r64);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r74, r64);
    r84 = r73 + r84;
    r73 = r25 * r20;
    r72 = r13 * r11;
    r22 = fma(r38, r22, r41 * r72);
    r22 = fma(r15, r59, r22);
    r22 = fma(r41, r21, r22);
    r73 = r73 * r22;
    r21 = r16 * r18;
    r84 = fma(r52, r21, r84);
    r84 = r84 + r73;
    r65 = r16 * r65;
    r65 = r65 * r69;
    r49 = r69 * r49;
    r69 = r65 + r49;
    r69 = fma(r6, r69, r7 * r84);
    r84 = r25 * r16;
    r84 = r84 * r22;
    r21 = fma(r51, r32, r84);
    r21 = r21 + r70;
    r69 = fma(r30, r21, r69);
    r76 = r22 * r76;
    r49 = r49 + r76;
    r84 = r71 + r84;
    r71 = r18 * r20;
    r84 = fma(r52, r71, r84);
    r21 = r18 * r29;
    r84 = fma(r51, r21, r84);
    r84 = fma(r6, r84, r30 * r49);
    r49 = r25 * r26;
    r32 = fma(r22, r32, r52 * r49);
    r32 = r32 + r83;
    r84 = fma(r7, r32, r84);
    r32 = fma(r84, r1, r69 * r19);
    r49 = r46 * r53;
    r21 = r25 * r5;
    r73 = r80 + r73;
    r73 = r73 + r75;
    r75 = r26 * r18;
    r80 = r18 * r29;
    r80 = fma(r22, r80, r52 * r75);
    r80 = r80 + r83;
    r80 = fma(r30, r80, r6 * r73);
    r76 = r65 + r76;
    r80 = fma(r7, r76, r80);
    r21 = r21 * r80;
    r21 = fma(r84, r81, r55 * r21);
    r76 = r25 * r69;
    r21 = fma(r56, r76, r21);
    r7 = r84 * r54;
    r21 = fma(r34, r7, r21);
    r49 = r49 * r0;
    r49 = r49 * r21;
    r32 = fma(r45, r49, r32);
    r49 = r4 * r5;
    r49 = r49 * r84;
    r49 = r49 * r55;
    r80 = fma(r80, r19, r39 * r49);
    r49 = r46 * r53;
    r49 = r49 * r5;
    r49 = r49 * r21;
    r80 = fma(r45, r49, r80);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r32, r80);
    r49 = fma(r36, r1, r43 * r19);
    r21 = r46 * r53;
    r7 = r25 * r57;
    r7 = r7 * r5;
    r76 = r36 * r54;
    r76 = fma(r34, r76, r55 * r7);
    r7 = r25 * r43;
    r76 = fma(r56, r7, r76);
    r76 = fma(r36, r81, r76);
    r21 = r21 * r0;
    r21 = r21 * r76;
    r49 = fma(r45, r21, r49);
    r21 = r4 * r36;
    r21 = r21 * r5;
    r21 = r21 * r55;
    r7 = r46 * r53;
    r7 = r7 * r5;
    r7 = r7 * r76;
    r7 = fma(r45, r7, r39 * r21);
    r7 = fma(r57, r19, r7);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r49, r7);
    r21 = r46 * r53;
    r76 = r48 * r54;
    r65 = r25 * r44;
    r65 = r65 * r5;
    r65 = fma(r55, r65, r34 * r76);
    r76 = r25 * r40;
    r65 = fma(r56, r76, r65);
    r65 = fma(r48, r81, r65);
    r21 = r21 * r0;
    r21 = r21 * r65;
    r21 = fma(r48, r1, r45 * r21);
    r21 = fma(r40, r19, r21);
    r76 = r4 * r48;
    r76 = r76 * r5;
    r76 = r76 * r55;
    r76 = fma(r39, r76, r44 * r19);
    r30 = r46 * r53;
    r30 = r30 * r5;
    r30 = r30 * r65;
    r76 = fma(r45, r30, r76);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r21, r76);
    r30 = r46 * r53;
    r65 = r25 * r35;
    r65 = r65 * r5;
    r73 = r42 * r54;
    r73 = fma(r34, r73, r55 * r65);
    r65 = r25 * r37;
    r73 = fma(r56, r65, r73);
    r73 = fma(r42, r81, r73);
    r30 = r30 * r0;
    r30 = r30 * r73;
    r30 = fma(r37, r19, r45 * r30);
    r30 = fma(r42, r1, r30);
    r65 = r4 * r42;
    r65 = r65 * r5;
    r65 = r65 * r55;
    r6 = r46 * r53;
    r6 = r6 * r5;
    r6 = r6 * r73;
    r6 = fma(r45, r6, r39 * r65);
    r6 = fma(r35, r19, r6);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r30, r6);
    r65 = r4 * r2;
    r73 = r4 * r3;
    r73 = fma(r79, r73, r78 * r65);
    r65 = r4 * r2;
    r83 = r4 * r3;
    r83 = fma(r64, r83, r74 * r65);
    WriteSum2<double, double>((double*)inout_shared, r73, r83);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = r4 * r3;
    r73 = r4 * r2;
    r73 = fma(r32, r73, r80 * r83);
    r83 = r4 * r2;
    r65 = r4 * r3;
    r65 = fma(r7, r65, r49 * r83);
    WriteSum2<double, double>((double*)inout_shared, r73, r65);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = r4 * r3;
    r73 = r4 * r2;
    r73 = fma(r21, r73, r76 * r65);
    r65 = r4 * r2;
    r83 = r4 * r3;
    r83 = fma(r6, r83, r30 * r65);
    WriteSum2<double, double>((double*)inout_shared, r73, r83);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = fma(r78, r78, r79 * r79);
    r73 = fma(r74, r74, r64 * r64);
    WriteSum2<double, double>((double*)inout_shared, r83, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r32, r32, r80 * r80);
    r83 = fma(r49, r49, r7 * r7);
    WriteSum2<double, double>((double*)inout_shared, r73, r83);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = fma(r76, r76, r21 * r21);
    r73 = fma(r6, r6, r30 * r30);
    WriteSum2<double, double>((double*)inout_shared, r83, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r79, r64, r78 * r74);
    r83 = fma(r79, r80, r78 * r32);
    WriteSum2<double, double>((double*)inout_shared, r73, r83);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = fma(r79, r7, r78 * r49);
    r73 = fma(r78, r21, r79 * r76);
    WriteSum2<double, double>((double*)inout_shared, r83, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = fma(r78, r30, r79 * r6);
    r79 = fma(r74, r32, r64 * r80);
    WriteSum2<double, double>((double*)inout_shared, r78, r79);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = fma(r74, r49, r64 * r7);
    r78 = fma(r74, r21, r64 * r76);
    WriteSum2<double, double>((double*)inout_shared, r79, r78);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fma(r74, r30, r64 * r6);
    r64 = fma(r32, r49, r80 * r7);
    WriteSum2<double, double>((double*)inout_shared, r74, r64);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = fma(r80, r76, r32 * r21);
    r32 = fma(r32, r30, r80 * r6);
    WriteSum2<double, double>((double*)inout_shared, r64, r32);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r49, r21, r7 * r76);
    r7 = fma(r7, r6, r49 * r30);
    WriteSum2<double, double>((double*)inout_shared, r32, r7);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r76, r6, r21 * r30);
    WriteSum1<double, double>((double*)inout_shared, r6);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r46 * r53;
    r76 = r25 * r24;
    r30 = r47 * r54;
    r30 = fma(r34, r30, r56 * r76);
    r76 = r25 * r27;
    r76 = r76 * r5;
    r30 = fma(r55, r76, r30);
    r30 = fma(r47, r81, r30);
    r6 = r6 * r0;
    r6 = r6 * r30;
    r6 = fma(r24, r19, r45 * r6);
    r6 = fma(r47, r1, r6);
    r76 = r46 * r53;
    r76 = r76 * r5;
    r76 = r76 * r30;
    r76 = fma(r27, r19, r45 * r76);
    r30 = r4 * r47;
    r30 = r30 * r5;
    r30 = r30 * r55;
    r76 = fma(r39, r30, r76);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r6, r76);
    r30 = r46 * r53;
    r21 = r25 * r28;
    r21 = fma(r56, r21, r50 * r81);
    r7 = r25 * r17;
    r7 = r7 * r5;
    r21 = fma(r55, r7, r21);
    r32 = r50 * r54;
    r21 = fma(r34, r32, r21);
    r30 = r30 * r0;
    r30 = r30 * r21;
    r30 = fma(r50, r1, r45 * r30);
    r30 = fma(r28, r19, r30);
    r32 = r4 * r50;
    r32 = r32 * r5;
    r32 = r32 * r55;
    r7 = r46 * r53;
    r7 = r7 * r5;
    r7 = r7 * r21;
    r7 = fma(r45, r7, r39 * r32);
    r7 = fma(r17, r19, r7);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 2 * out_point_jac_num_alloc, global_thread_idx, r30, r7);
    r1 = fma(r33, r19, r23 * r1);
    r32 = r46 * r53;
    r21 = r25 * r33;
    r49 = r23 * r54;
    r49 = fma(r34, r49, r56 * r21);
    r21 = r25 * r31;
    r21 = r21 * r5;
    r49 = fma(r55, r21, r49);
    r49 = fma(r23, r81, r49);
    r32 = r32 * r0;
    r32 = r32 * r49;
    r1 = fma(r45, r32, r1);
    r32 = r46 * r53;
    r32 = r32 * r5;
    r32 = r32 * r49;
    r19 = fma(r31, r19, r45 * r32);
    r32 = r4 * r23;
    r32 = r32 * r5;
    r32 = r32 * r55;
    r19 = fma(r39, r32, r19);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r1, r19);
    r32 = r4 * r2;
    r39 = r4 * r3;
    r39 = fma(r76, r39, r6 * r32);
    r32 = r4 * r3;
    r55 = r4 * r2;
    r55 = fma(r30, r55, r7 * r32);
    WriteSum2<double, double>((double*)inout_shared, r39, r55);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = r4 * r2;
    r39 = r4 * r3;
    r39 = fma(r19, r39, r1 * r55);
    WriteSum1<double, double>((double*)inout_shared, r39);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r6, r6, r76 * r76);
    r55 = fma(r30, r30, r7 * r7);
    WriteSum2<double, double>((double*)inout_shared, r39, r55);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fma(r19, r19, r1 * r1);
    WriteSum1<double, double>((double*)inout_shared, r55);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fma(r6, r30, r76 * r7);
    r6 = fma(r6, r1, r76 * r19);
    WriteSum2<double, double>((double*)inout_shared, r55, r6);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fma(r30, r1, r7 * r19);
    WriteSum1<double, double>((double*)inout_shared, r1);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
}

void SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointResJac(
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
    double* focal_and_distortion,
    unsigned int focal_and_distortion_num_alloc,
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
  SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointResJacKernel<<<
      n_blocks,
      1024>>>(pose,
              pose_num_alloc,
              pose_indices,
              sensor_from_rig,
              sensor_from_rig_num_alloc,
              point,
              point_num_alloc,
              point_indices,
              pixel,
              pixel_num_alloc,
              focal_and_distortion,
              focal_and_distortion_num_alloc,
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