#include "kernel_simple_radial_split_fixed_focal_and_distortion_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndDistortionResJacKernel(
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
        double* focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87;
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
    r38 = r45 * r39;
    r2 = fma(r0, r38, r2);
    r3 = fma(r3, r4, r1);
    r3 = fma(r5, r38, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = r12 * r11;
    r51 = -5.00000000000000000e-01;
    r49 = r9 * r14;
    r49 = fma(r51, r49, r51 * r1);
    r1 = r8 * r15;
    r49 = fma(r51, r1, r49);
    r41 = r13 * r10;
    r58 = 5.00000000000000000e-01;
    r49 = fma(r58, r41, r49);
    r41 = r20 * r49;
    r1 = r8 * r11;
    r59 = r13 * r14;
    r59 = fma(r58, r59, r58 * r1);
    r1 = r12 * r15;
    r59 = fma(r51, r1, r59);
    r60 = r9 * r58;
    r59 = fma(r10, r60, r59);
    r1 = fma(r59, r32, r25 * r41);
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
    r1 = r1 + r61;
    r65 = r25 * r20;
    r65 = r65 * r66;
    r67 = r25 * r26;
    r67 = r67 * r59;
    r68 = r65 + r67;
    r69 = r16 * r18;
    r68 = fma(r49, r69, r68);
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
    r72 = r25 * r20;
    r72 = r72 * r62;
    r73 = r25 * r16;
    r73 = fma(r59, r73, r72);
    r74 = r25 * r26;
    r74 = r74 * r49;
    r75 = r66 * r32;
    r76 = r74 + r75;
    r77 = r73 + r76;
    r78 = r18 * r29;
    r78 = fma(r18, r41, r59 * r78);
    r78 = r78 + r61;
    r78 = fma(r6, r78, r7 * r77);
    r77 = r26 * r70;
    r59 = r66 * r77;
    r1 = r1 + r59;
    r78 = fma(r30, r1, r78);
    r1 = r4 * r39;
    r1 = r1 * r56;
    r79 = fma(r78, r1, r68 * r38);
    r80 = r46 * r53;
    r81 = r25 * r68;
    r82 = r0 * r0;
    r54 = r52 * r54;
    r54 = 1.0 / r54;
    r54 = r18 * r54;
    r82 = r82 * r54;
    r81 = fma(r78, r82, r56 * r81);
    r52 = r78 * r54;
    r81 = fma(r34, r52, r81);
    r83 = r25 * r5;
    r84 = r26 * r18;
    r85 = r18 * r29;
    r85 = r85 * r66;
    r84 = fma(r49, r84, r85);
    r84 = r84 + r73;
    r59 = r71 + r59;
    r59 = fma(r7, r59, r30 * r84);
    r67 = fma(r62, r32, r67);
    r84 = r25 * r16;
    r84 = fma(r49, r84, r65);
    r67 = r67 + r84;
    r59 = fma(r6, r67, r59);
    r83 = r83 * r59;
    r81 = fma(r55, r83, r81);
    r80 = r80 * r81;
    r80 = r80 * r45;
    r79 = fma(r0, r80, r79);
    r81 = r4 * r5;
    r81 = r81 * r78;
    r81 = r81 * r55;
    r80 = fma(r5, r80, r39 * r81);
    r80 = fma(r59, r38, r80);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r79, r80);
    r59 = r18 * r20;
    r59 = fma(r62, r59, r85);
    r81 = r25 * r16;
    r83 = r8 * r11;
    r52 = r12 * r15;
    r52 = fma(r58, r52, r51 * r83);
    r83 = r9 * r10;
    r52 = fma(r51, r83, r52);
    r52 = fma(r14, r63, r52);
    r81 = r81 * r52;
    r83 = r25 * r26;
    r67 = r12 * r11;
    r65 = r8 * r15;
    r65 = fma(r58, r65, r58 * r67);
    r65 = fma(r14, r60, r65);
    r65 = fma(r10, r63, r65);
    r83 = fma(r65, r83, r81);
    r59 = r59 + r83;
    r63 = r25 * r20;
    r63 = r63 * r65;
    r67 = fma(r52, r32, r63);
    r67 = r67 + r61;
    r67 = fma(r7, r67, r6 * r59);
    r59 = r20 * r66;
    r59 = r59 * r70;
    r61 = r52 * r77;
    r71 = r59 + r61;
    r67 = fma(r30, r71, r67);
    r75 = r72 + r75;
    r75 = r75 + r83;
    r83 = r16 * r70;
    r83 = r83 * r65;
    r59 = r59 + r83;
    r59 = fma(r6, r59, r30 * r75);
    r75 = r18 * r29;
    r75 = fma(r18, r69, r65 * r75);
    r72 = r25 * r26;
    r72 = r72 * r66;
    r71 = r25 * r20;
    r71 = fma(r52, r71, r72);
    r75 = r75 + r71;
    r59 = fma(r7, r75, r59);
    r75 = fma(r59, r38, r67 * r1);
    r73 = r46 * r53;
    r86 = r25 * r59;
    r87 = r25 * r5;
    r63 = r64 + r63;
    r64 = r26 * r18;
    r63 = fma(r62, r64, r63);
    r62 = r18 * r29;
    r63 = fma(r52, r62, r63);
    r65 = fma(r65, r32, r25 * r69);
    r65 = r65 + r71;
    r65 = fma(r6, r65, r30 * r63);
    r61 = r83 + r61;
    r65 = fma(r7, r61, r65);
    r87 = r87 * r65;
    r87 = fma(r55, r87, r56 * r86);
    r86 = r67 * r54;
    r87 = fma(r34, r86, r87);
    r87 = fma(r67, r82, r87);
    r73 = r73 * r0;
    r73 = r73 * r87;
    r75 = fma(r45, r73, r75);
    r73 = r46 * r53;
    r73 = r73 * r5;
    r73 = r73 * r87;
    r65 = fma(r65, r38, r45 * r73);
    r73 = r4 * r5;
    r73 = r73 * r67;
    r73 = r73 * r55;
    r65 = fma(r39, r73, r65);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r75, r65);
    r85 = r74 + r85;
    r74 = r25 * r20;
    r73 = r13 * r11;
    r22 = fma(r51, r22, r58 * r73);
    r22 = fma(r15, r60, r22);
    r22 = fma(r58, r21, r22);
    r74 = r74 * r22;
    r21 = r16 * r18;
    r85 = fma(r52, r21, r85);
    r85 = r85 + r74;
    r66 = r16 * r66;
    r66 = r66 * r70;
    r41 = r70 * r41;
    r70 = r66 + r41;
    r70 = fma(r6, r70, r7 * r85);
    r85 = r25 * r16;
    r85 = r85 * r22;
    r21 = fma(r49, r32, r85);
    r21 = r21 + r71;
    r70 = fma(r30, r21, r70);
    r77 = r22 * r77;
    r41 = r41 + r77;
    r85 = r72 + r85;
    r72 = r18 * r20;
    r85 = fma(r52, r72, r85);
    r21 = r18 * r29;
    r85 = fma(r49, r21, r85);
    r85 = fma(r6, r85, r30 * r41);
    r41 = r25 * r26;
    r32 = fma(r22, r32, r52 * r41);
    r32 = r32 + r84;
    r85 = fma(r7, r32, r85);
    r32 = fma(r85, r1, r70 * r38);
    r41 = r46 * r53;
    r21 = r25 * r5;
    r74 = r81 + r74;
    r74 = r74 + r76;
    r76 = r26 * r18;
    r81 = r18 * r29;
    r81 = fma(r22, r81, r52 * r76);
    r81 = r81 + r84;
    r81 = fma(r30, r81, r6 * r74);
    r77 = r66 + r77;
    r81 = fma(r7, r77, r81);
    r21 = r21 * r81;
    r21 = fma(r85, r82, r55 * r21);
    r77 = r25 * r70;
    r21 = fma(r56, r77, r21);
    r7 = r85 * r54;
    r21 = fma(r34, r7, r21);
    r41 = r41 * r0;
    r41 = r41 * r21;
    r32 = fma(r45, r41, r32);
    r41 = r4 * r5;
    r41 = r41 * r85;
    r41 = r41 * r55;
    r81 = fma(r81, r38, r39 * r41);
    r41 = r46 * r53;
    r41 = r41 * r5;
    r41 = r41 * r21;
    r81 = fma(r45, r41, r81);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r32, r81);
    r41 = fma(r36, r1, r43 * r38);
    r21 = r46 * r53;
    r7 = r25 * r57;
    r7 = r7 * r5;
    r77 = r36 * r54;
    r77 = fma(r34, r77, r55 * r7);
    r7 = r25 * r43;
    r77 = fma(r56, r7, r77);
    r77 = fma(r36, r82, r77);
    r21 = r21 * r0;
    r21 = r21 * r77;
    r41 = fma(r45, r21, r41);
    r21 = r4 * r36;
    r21 = r21 * r5;
    r21 = r21 * r55;
    r7 = r46 * r53;
    r7 = r7 * r5;
    r7 = r7 * r77;
    r7 = fma(r45, r7, r39 * r21);
    r7 = fma(r57, r38, r7);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r41, r7);
    r21 = r46 * r53;
    r77 = r48 * r54;
    r66 = r25 * r44;
    r66 = r66 * r5;
    r66 = fma(r55, r66, r34 * r77);
    r77 = r25 * r40;
    r66 = fma(r56, r77, r66);
    r66 = fma(r48, r82, r66);
    r21 = r21 * r0;
    r21 = r21 * r66;
    r21 = fma(r48, r1, r45 * r21);
    r21 = fma(r40, r38, r21);
    r77 = r4 * r48;
    r77 = r77 * r5;
    r77 = r77 * r55;
    r77 = fma(r39, r77, r44 * r38);
    r30 = r46 * r53;
    r30 = r30 * r5;
    r30 = r30 * r66;
    r77 = fma(r45, r30, r77);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r21, r77);
    r30 = r46 * r53;
    r66 = r25 * r35;
    r66 = r66 * r5;
    r74 = r42 * r54;
    r74 = fma(r34, r74, r55 * r66);
    r66 = r25 * r37;
    r74 = fma(r56, r66, r74);
    r74 = fma(r42, r82, r74);
    r30 = r30 * r0;
    r30 = r30 * r74;
    r30 = fma(r37, r38, r45 * r30);
    r30 = fma(r42, r1, r30);
    r66 = r4 * r42;
    r66 = r66 * r5;
    r66 = r66 * r55;
    r6 = r46 * r53;
    r6 = r6 * r5;
    r6 = r6 * r74;
    r6 = fma(r45, r6, r39 * r66);
    r6 = fma(r35, r38, r6);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r30, r6);
    r66 = r4 * r3;
    r2 = r4 * r2;
    r66 = fma(r79, r2, r80 * r66);
    r74 = r4 * r3;
    r74 = fma(r75, r2, r65 * r74);
    WriteSum2<double, double>((double*)inout_shared, r66, r74);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = r4 * r3;
    r74 = fma(r32, r2, r81 * r74);
    r66 = r4 * r3;
    r66 = fma(r41, r2, r7 * r66);
    WriteSum2<double, double>((double*)inout_shared, r74, r66);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = r4 * r3;
    r66 = fma(r21, r2, r77 * r66);
    r74 = r4 * r3;
    r74 = fma(r30, r2, r6 * r74);
    WriteSum2<double, double>((double*)inout_shared, r66, r74);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fma(r79, r79, r80 * r80);
    r66 = fma(r75, r75, r65 * r65);
    WriteSum2<double, double>((double*)inout_shared, r74, r66);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = fma(r32, r32, r81 * r81);
    r74 = fma(r41, r41, r7 * r7);
    WriteSum2<double, double>((double*)inout_shared, r66, r74);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fma(r77, r77, r21 * r21);
    r66 = fma(r6, r6, r30 * r30);
    WriteSum2<double, double>((double*)inout_shared, r74, r66);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = fma(r80, r65, r79 * r75);
    r74 = fma(r80, r81, r79 * r32);
    WriteSum2<double, double>((double*)inout_shared, r66, r74);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fma(r80, r7, r79 * r41);
    r66 = fma(r79, r21, r80 * r77);
    WriteSum2<double, double>((double*)inout_shared, r74, r66);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = fma(r79, r30, r80 * r6);
    r80 = fma(r75, r32, r65 * r81);
    WriteSum2<double, double>((double*)inout_shared, r79, r80);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = fma(r75, r41, r65 * r7);
    r79 = fma(r75, r21, r65 * r77);
    WriteSum2<double, double>((double*)inout_shared, r80, r79);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = fma(r75, r30, r65 * r6);
    r65 = fma(r32, r41, r81 * r7);
    WriteSum2<double, double>((double*)inout_shared, r75, r65);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = fma(r81, r77, r32 * r21);
    r32 = fma(r32, r30, r81 * r6);
    WriteSum2<double, double>((double*)inout_shared, r65, r32);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r41, r21, r7 * r77);
    r7 = fma(r7, r6, r41 * r30);
    WriteSum2<double, double>((double*)inout_shared, r32, r7);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r77, r6, r21 * r30);
    WriteSum1<double, double>((double*)inout_shared, r6);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r4 * r3;
    WriteSum2<double, double>((double*)inout_shared, r2, r6);
  };
  FlushSumShared<2, double>(out_principal_point_njtr,
                            0 * out_principal_point_njtr_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r19, r19);
  };
  FlushSumShared<2, double>(out_principal_point_precond_diag,
                            0 * out_principal_point_precond_diag_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = r46 * r53;
    r6 = r25 * r24;
    r77 = r47 * r54;
    r77 = fma(r34, r77, r56 * r6);
    r6 = r25 * r27;
    r6 = r6 * r5;
    r77 = fma(r55, r6, r77);
    r77 = fma(r47, r82, r77);
    r19 = r19 * r0;
    r19 = r19 * r77;
    r19 = fma(r24, r38, r45 * r19);
    r19 = fma(r47, r1, r19);
    r6 = r46 * r53;
    r6 = r6 * r5;
    r6 = r6 * r77;
    r6 = fma(r27, r38, r45 * r6);
    r77 = r4 * r47;
    r77 = r77 * r5;
    r77 = r77 * r55;
    r6 = fma(r39, r77, r6);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r19, r6);
    r77 = r46 * r53;
    r30 = r25 * r28;
    r30 = fma(r56, r30, r50 * r82);
    r21 = r25 * r17;
    r21 = r21 * r5;
    r30 = fma(r55, r21, r30);
    r7 = r50 * r54;
    r30 = fma(r34, r7, r30);
    r77 = r77 * r0;
    r77 = r77 * r30;
    r77 = fma(r50, r1, r45 * r77);
    r77 = fma(r28, r38, r77);
    r7 = r4 * r50;
    r7 = r7 * r5;
    r7 = r7 * r55;
    r21 = r46 * r53;
    r21 = r21 * r5;
    r21 = r21 * r30;
    r21 = fma(r45, r21, r39 * r7);
    r21 = fma(r17, r38, r21);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r77,
                                             r21);
    r1 = fma(r33, r38, r23 * r1);
    r7 = r46 * r53;
    r30 = r25 * r33;
    r32 = r23 * r54;
    r32 = fma(r34, r32, r56 * r30);
    r30 = r25 * r31;
    r30 = r30 * r5;
    r32 = fma(r55, r30, r32);
    r32 = fma(r23, r82, r32);
    r7 = r7 * r0;
    r7 = r7 * r32;
    r1 = fma(r45, r7, r1);
    r7 = r46 * r53;
    r7 = r7 * r5;
    r7 = r7 * r32;
    r38 = fma(r31, r38, r45 * r7);
    r7 = r4 * r23;
    r7 = r7 * r5;
    r7 = r7 * r55;
    r38 = fma(r39, r7, r38);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r1, r38);
    r7 = r4 * r3;
    r7 = fma(r19, r2, r6 * r7);
    r39 = r4 * r3;
    r39 = fma(r77, r2, r21 * r39);
    WriteSum2<double, double>((double*)inout_shared, r7, r39);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r4 * r3;
    r2 = fma(r1, r2, r38 * r39);
    WriteSum1<double, double>((double*)inout_shared, r2);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = fma(r19, r19, r6 * r6);
    r39 = fma(r77, r77, r21 * r21);
    WriteSum2<double, double>((double*)inout_shared, r2, r39);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r38, r38, r1 * r1);
    WriteSum1<double, double>((double*)inout_shared, r39);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r19, r77, r6 * r21);
    r19 = fma(r19, r1, r6 * r38);
    WriteSum2<double, double>((double*)inout_shared, r39, r19);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fma(r77, r1, r21 * r38);
    WriteSum1<double, double>((double*)inout_shared, r1);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
}

void SimpleRadialSplitFixedFocalAndDistortionResJac(
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
    double* focal_and_distortion,
    unsigned int focal_and_distortion_num_alloc,
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
  SimpleRadialSplitFixedFocalAndDistortionResJacKernel<<<n_blocks, 1024>>>(
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
      focal_and_distortion,
      focal_and_distortion_num_alloc,
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