#include "kernel_simple_radial_split_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPrincipalPointResJacFirstKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc,
        SharedIndex* focal_and_distortion_indices,
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
        double* out_focal_and_distortion_jac,
        unsigned int out_focal_and_distortion_jac_num_alloc,
        double* const out_focal_and_distortion_njtr,
        unsigned int out_focal_and_distortion_njtr_num_alloc,
        double* const out_focal_and_distortion_precond_diag,
        unsigned int out_focal_and_distortion_precond_diag_num_alloc,
        double* const out_focal_and_distortion_precond_tril,
        unsigned int out_focal_and_distortion_precond_tril_num_alloc,
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

  __shared__ SharedIndex focal_and_distortion_indices_loc[1024];
  focal_and_distortion_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_distortion_indices[global_thread_idx]
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
  LoadShared<2, double, double>(focal_and_distortion,
                                0 * focal_and_distortion_num_alloc,
                                focal_and_distortion_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_distortion_indices_loc[threadIdx.x].target,
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
    r1 = fma(r2, r2, r3 * r3);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r1);
  if (global_thread_idx < problem_size) {
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
    r72 = r19 * r68;
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
    r1 = r79 * r55;
    r80 = r4 * r19;
    r81 = r45 * r80;
    r1 = fma(r81, r1, r51 * r72);
    r72 = r25 * r68;
    r82 = r0 * r0;
    r47 = r54 * r47;
    r47 = 1.0 / r47;
    r47 = r18 * r47;
    r82 = r82 * r47;
    r72 = fma(r79, r82, r55 * r72);
    r54 = r79 * r47;
    r72 = fma(r34, r54, r72);
    r83 = r25 * r5;
    r84 = r26 * r18;
    r85 = r18 * r29;
    r85 = r85 * r66;
    r84 = fma(r41, r84, r85);
    r84 = r84 + r74;
    r59 = r71 + r59;
    r59 = fma(r7, r59, r30 * r84);
    r67 = fma(r62, r32, r67);
    r84 = r25 * r16;
    r84 = fma(r41, r84, r65);
    r67 = r67 + r84;
    r59 = fma(r6, r67, r59);
    r83 = r83 * r59;
    r72 = fma(r48, r83, r72);
    r83 = r0 * r72;
    r46 = r46 * r51;
    r1 = fma(r46, r83, r1);
    r83 = r5 * r48;
    r83 = r83 * r81;
    r54 = r5 * r72;
    r54 = fma(r46, r54, r79 * r83);
    r67 = r19 * r59;
    r54 = fma(r51, r67, r54);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r1, r54);
    r67 = r18 * r20;
    r67 = fma(r62, r67, r85);
    r65 = r25 * r16;
    r71 = r8 * r11;
    r74 = r12 * r15;
    r74 = fma(r58, r74, r38 * r71);
    r71 = r9 * r10;
    r74 = fma(r38, r71, r74);
    r74 = fma(r14, r63, r74);
    r65 = r65 * r74;
    r71 = r25 * r26;
    r86 = r12 * r11;
    r87 = r8 * r15;
    r87 = fma(r58, r87, r58 * r86);
    r87 = fma(r14, r60, r87);
    r87 = fma(r10, r63, r87);
    r71 = fma(r87, r71, r65);
    r67 = r67 + r71;
    r63 = r25 * r20;
    r63 = r63 * r87;
    r86 = fma(r74, r32, r63);
    r86 = r86 + r61;
    r86 = fma(r7, r86, r6 * r67);
    r67 = r20 * r66;
    r67 = r67 * r70;
    r61 = r74 * r78;
    r88 = r67 + r61;
    r86 = fma(r30, r88, r86);
    r88 = r86 * r55;
    r76 = r73 + r76;
    r76 = r76 + r71;
    r71 = r16 * r70;
    r71 = r71 * r87;
    r67 = r67 + r71;
    r67 = fma(r6, r67, r30 * r76);
    r76 = r18 * r29;
    r76 = fma(r18, r69, r87 * r76);
    r73 = r25 * r26;
    r73 = r73 * r66;
    r89 = r25 * r20;
    r89 = fma(r74, r89, r73);
    r76 = r76 + r89;
    r67 = fma(r7, r76, r67);
    r76 = r19 * r67;
    r76 = fma(r51, r76, r81 * r88);
    r88 = r25 * r67;
    r90 = r25 * r5;
    r63 = r64 + r63;
    r64 = r26 * r18;
    r63 = fma(r62, r64, r63);
    r62 = r18 * r29;
    r63 = fma(r74, r62, r63);
    r87 = fma(r87, r32, r25 * r69);
    r87 = r87 + r89;
    r87 = fma(r6, r87, r30 * r63);
    r61 = r71 + r61;
    r87 = fma(r7, r61, r87);
    r90 = r90 * r87;
    r90 = fma(r48, r90, r55 * r88);
    r88 = r86 * r47;
    r90 = fma(r34, r88, r90);
    r90 = fma(r86, r82, r90);
    r88 = r0 * r90;
    r76 = fma(r46, r88, r76);
    r88 = r5 * r90;
    r61 = r19 * r87;
    r61 = fma(r51, r61, r46 * r88);
    r61 = fma(r86, r83, r61);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r76, r61);
    r85 = r75 + r85;
    r75 = r25 * r20;
    r88 = r13 * r11;
    r22 = fma(r38, r22, r58 * r88);
    r22 = fma(r15, r60, r22);
    r22 = fma(r58, r21, r22);
    r75 = r75 * r22;
    r21 = r16 * r18;
    r85 = fma(r74, r21, r85);
    r85 = r85 + r75;
    r66 = r16 * r66;
    r66 = r66 * r70;
    r57 = r70 * r57;
    r70 = r66 + r57;
    r70 = fma(r6, r70, r7 * r85);
    r85 = r25 * r16;
    r85 = r85 * r22;
    r21 = fma(r41, r32, r85);
    r21 = r21 + r89;
    r70 = fma(r30, r21, r70);
    r21 = r19 * r70;
    r78 = r22 * r78;
    r57 = r57 + r78;
    r85 = r73 + r85;
    r73 = r18 * r20;
    r85 = fma(r74, r73, r85);
    r89 = r18 * r29;
    r85 = fma(r41, r89, r85);
    r85 = fma(r6, r85, r30 * r57);
    r57 = r25 * r26;
    r32 = fma(r22, r32, r74 * r57);
    r32 = r32 + r84;
    r85 = fma(r7, r32, r85);
    r32 = r85 * r55;
    r32 = fma(r81, r32, r51 * r21);
    r21 = r25 * r5;
    r75 = r65 + r75;
    r75 = r75 + r77;
    r77 = r26 * r18;
    r65 = r18 * r29;
    r65 = fma(r22, r65, r74 * r77);
    r65 = r65 + r84;
    r65 = fma(r30, r65, r6 * r75);
    r78 = r66 + r78;
    r65 = fma(r7, r78, r65);
    r21 = r21 * r65;
    r21 = fma(r85, r82, r48 * r21);
    r78 = r25 * r70;
    r21 = fma(r55, r78, r21);
    r7 = r85 * r47;
    r21 = fma(r34, r7, r21);
    r7 = r0 * r21;
    r32 = fma(r46, r7, r32);
    r7 = r19 * r65;
    r7 = fma(r51, r7, r85 * r83);
    r78 = r5 * r21;
    r7 = fma(r46, r78, r7);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r32, r7);
    r78 = r43 * r19;
    r66 = r36 * r55;
    r66 = fma(r81, r66, r51 * r78);
    r78 = r25 * r56;
    r78 = r78 * r5;
    r30 = r36 * r47;
    r30 = fma(r34, r30, r48 * r78);
    r78 = r25 * r43;
    r30 = fma(r55, r78, r30);
    r30 = fma(r36, r82, r30);
    r78 = r0 * r30;
    r66 = fma(r46, r78, r66);
    r78 = r5 * r30;
    r78 = fma(r46, r78, r36 * r83);
    r75 = r56 * r19;
    r78 = fma(r51, r75, r78);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r66, r78);
    r75 = r50 * r47;
    r6 = r25 * r44;
    r6 = r6 * r5;
    r6 = fma(r48, r6, r34 * r75);
    r75 = r25 * r40;
    r6 = fma(r55, r75, r6);
    r6 = fma(r50, r82, r6);
    r75 = r0 * r6;
    r84 = r50 * r55;
    r84 = fma(r81, r84, r46 * r75);
    r75 = r40 * r19;
    r84 = fma(r51, r75, r84);
    r75 = r44 * r19;
    r75 = fma(r50, r83, r51 * r75);
    r77 = r5 * r6;
    r75 = fma(r46, r77, r75);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r84, r75);
    r77 = r25 * r35;
    r77 = r77 * r5;
    r22 = r42 * r47;
    r22 = fma(r34, r22, r48 * r77);
    r77 = r25 * r37;
    r22 = fma(r55, r77, r22);
    r22 = fma(r42, r82, r22);
    r77 = r0 * r22;
    r74 = r37 * r19;
    r74 = fma(r51, r74, r46 * r77);
    r77 = r42 * r55;
    r74 = fma(r81, r77, r74);
    r77 = r5 * r22;
    r77 = fma(r46, r77, r42 * r83);
    r57 = r35 * r19;
    r77 = fma(r51, r57, r77);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r74, r77);
    r57 = r4 * r2;
    r89 = r4 * r3;
    r89 = fma(r54, r89, r1 * r57);
    r57 = r4 * r2;
    r73 = r4 * r3;
    r73 = fma(r61, r73, r76 * r57);
    WriteSum2<double, double>((double*)inout_shared, r89, r73);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = r4 * r3;
    r89 = r4 * r2;
    r89 = fma(r32, r89, r7 * r73);
    r73 = r4 * r2;
    r57 = r4 * r3;
    r57 = fma(r78, r57, r66 * r73);
    WriteSum2<double, double>((double*)inout_shared, r89, r57);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = r4 * r3;
    r89 = r4 * r2;
    r89 = fma(r84, r89, r75 * r57);
    r57 = r4 * r2;
    r73 = r4 * r3;
    r73 = fma(r77, r73, r74 * r57);
    WriteSum2<double, double>((double*)inout_shared, r89, r73);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r1, r1, r54 * r54);
    r89 = fma(r76, r76, r61 * r61);
    WriteSum2<double, double>((double*)inout_shared, r73, r89);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r89 = fma(r32, r32, r7 * r7);
    r73 = fma(r66, r66, r78 * r78);
    WriteSum2<double, double>((double*)inout_shared, r89, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r75, r75, r84 * r84);
    r89 = fma(r77, r77, r74 * r74);
    WriteSum2<double, double>((double*)inout_shared, r73, r89);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r89 = fma(r54, r61, r1 * r76);
    r73 = fma(r54, r7, r1 * r32);
    WriteSum2<double, double>((double*)inout_shared, r89, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r54, r78, r1 * r66);
    r89 = fma(r1, r84, r54 * r75);
    WriteSum2<double, double>((double*)inout_shared, r73, r89);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fma(r1, r74, r54 * r77);
    r54 = fma(r76, r32, r61 * r7);
    WriteSum2<double, double>((double*)inout_shared, r1, r54);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = fma(r76, r66, r61 * r78);
    r1 = fma(r76, r84, r61 * r75);
    WriteSum2<double, double>((double*)inout_shared, r54, r1);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = fma(r76, r74, r61 * r77);
    r61 = fma(r32, r66, r7 * r78);
    WriteSum2<double, double>((double*)inout_shared, r76, r61);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fma(r7, r75, r32 * r84);
    r32 = fma(r32, r74, r7 * r77);
    WriteSum2<double, double>((double*)inout_shared, r61, r32);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r66, r84, r78 * r75);
    r78 = fma(r78, r77, r66 * r74);
    WriteSum2<double, double>((double*)inout_shared, r32, r78);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = fma(r75, r77, r84 * r74);
    WriteSum1<double, double>((double*)inout_shared, r77);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = r0 * r19;
    r77 = r77 * r53;
    r75 = r5 * r19;
    r75 = r75 * r53;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_distortion_jac,
        0 * out_focal_and_distortion_jac_num_alloc,
        global_thread_idx,
        r77,
        r75);
    r75 = r0 * r39;
    r75 = r75 * r51;
    r77 = r5 * r39;
    r77 = r77 * r51;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_distortion_jac,
        2 * out_focal_and_distortion_jac_num_alloc,
        global_thread_idx,
        r75,
        r77);
    r77 = r5 * r3;
    r77 = r77 * r53;
    r75 = r0 * r2;
    r75 = r75 * r53;
    r75 = fma(r80, r75, r80 * r77);
    r77 = r4 * r0;
    r77 = r77 * r39;
    r77 = r77 * r2;
    r80 = r4 * r5;
    r80 = r80 * r39;
    r80 = r80 * r3;
    r80 = fma(r51, r80, r51 * r77);
    WriteSum2<double, double>((double*)inout_shared, r75, r80);
  };
  FlushSumShared<2, double>(out_focal_and_distortion_njtr,
                            0 * out_focal_and_distortion_njtr_num_alloc,
                            focal_and_distortion_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = r0 * r19;
    r80 = r80 * r19;
    r75 = r19 * r19;
    r75 = r75 * r48;
    r75 = fma(r34, r75, r55 * r80);
    r80 = r39 * r39;
    r77 = r45 * r80;
    r53 = r45 * r48;
    r53 = r53 * r34;
    r74 = r45 * r45;
    r74 = r74 * r0;
    r74 = r74 * r55;
    r74 = fma(r80, r74, r53 * r77);
    WriteSum2<double, double>((double*)inout_shared, r75, r74);
  };
  FlushSumShared<2, double>(out_focal_and_distortion_precond_diag,
                            0 * out_focal_and_distortion_precond_diag_num_alloc,
                            focal_and_distortion_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = r39 * r19;
    r75 = r45 * r0;
    r75 = r75 * r39;
    r75 = r75 * r19;
    r75 = fma(r55, r75, r53 * r74);
    WriteSum1<double, double>((double*)inout_shared, r75);
  };
  FlushSumShared<1, double>(out_focal_and_distortion_precond_tril,
                            0 * out_focal_and_distortion_precond_tril_num_alloc,
                            focal_and_distortion_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = r25 * r24;
    r74 = r49 * r47;
    r74 = fma(r34, r74, r55 * r75);
    r75 = r25 * r27;
    r75 = r75 * r5;
    r74 = fma(r48, r75, r74);
    r74 = fma(r49, r82, r74);
    r74 = r74 * r46;
    r75 = r24 * r19;
    r75 = fma(r51, r75, r0 * r74);
    r53 = r49 * r55;
    r75 = fma(r81, r53, r75);
    r53 = r27 * r19;
    r53 = fma(r51, r53, r5 * r74);
    r53 = fma(r49, r83, r53);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r75,
                                             r53);
    r74 = r25 * r28;
    r74 = fma(r55, r74, r52 * r82);
    r77 = r25 * r17;
    r77 = r77 * r5;
    r74 = fma(r48, r77, r74);
    r84 = r52 * r47;
    r74 = fma(r34, r84, r74);
    r84 = r0 * r74;
    r77 = r52 * r55;
    r77 = fma(r81, r77, r46 * r84);
    r84 = r28 * r19;
    r77 = fma(r51, r84, r77);
    r84 = r5 * r74;
    r84 = fma(r46, r84, r52 * r83);
    r78 = r17 * r19;
    r84 = fma(r51, r78, r84);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r77,
                                             r84);
    r78 = r23 * r55;
    r32 = r33 * r19;
    r32 = fma(r51, r32, r81 * r78);
    r78 = r25 * r33;
    r81 = r23 * r47;
    r81 = fma(r34, r81, r55 * r78);
    r78 = r25 * r31;
    r78 = r78 * r5;
    r81 = fma(r48, r78, r81);
    r81 = fma(r23, r82, r81);
    r78 = r0 * r81;
    r32 = fma(r46, r78, r32);
    r78 = r5 * r81;
    r82 = r31 * r19;
    r82 = fma(r51, r82, r46 * r78);
    r82 = fma(r23, r83, r82);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r32,
                                             r82);
    r83 = r4 * r2;
    r78 = r4 * r3;
    r78 = fma(r53, r78, r75 * r83);
    r83 = r4 * r3;
    r51 = r4 * r2;
    r51 = fma(r77, r51, r84 * r83);
    WriteSum2<double, double>((double*)inout_shared, r78, r51);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = r4 * r2;
    r78 = r4 * r3;
    r78 = fma(r82, r78, r32 * r51);
    WriteSum1<double, double>((double*)inout_shared, r78);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = fma(r75, r75, r53 * r53);
    r51 = fma(r77, r77, r84 * r84);
    WriteSum2<double, double>((double*)inout_shared, r78, r51);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = fma(r82, r82, r32 * r32);
    WriteSum1<double, double>((double*)inout_shared, r51);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = fma(r75, r77, r53 * r84);
    r75 = fma(r75, r32, r53 * r82);
    WriteSum2<double, double>((double*)inout_shared, r51, r75);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r77, r32, r84 * r82);
    WriteSum1<double, double>((double*)inout_shared, r32);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedPrincipalPointResJacFirst(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* focal_and_distortion,
    unsigned int focal_and_distortion_num_alloc,
    SharedIndex* focal_and_distortion_indices,
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
    double* out_focal_and_distortion_jac,
    unsigned int out_focal_and_distortion_jac_num_alloc,
    double* const out_focal_and_distortion_njtr,
    unsigned int out_focal_and_distortion_njtr_num_alloc,
    double* const out_focal_and_distortion_precond_diag,
    unsigned int out_focal_and_distortion_precond_diag_num_alloc,
    double* const out_focal_and_distortion_precond_tril,
    unsigned int out_focal_and_distortion_precond_tril_num_alloc,
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
  SimpleRadialSplitFixedPrincipalPointResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      focal_and_distortion,
      focal_and_distortion_num_alloc,
      focal_and_distortion_indices,
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
      out_focal_and_distortion_jac,
      out_focal_and_distortion_jac_num_alloc,
      out_focal_and_distortion_njtr,
      out_focal_and_distortion_njtr_num_alloc,
      out_focal_and_distortion_precond_diag,
      out_focal_and_distortion_precond_diag_num_alloc,
      out_focal_and_distortion_precond_tril,
      out_focal_and_distortion_precond_tril_num_alloc,
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