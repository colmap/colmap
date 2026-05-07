#include "kernel_simple_radial_split_fixed_principal_point_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPrincipalPointFixedPointResJacFirstKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        double* focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc,
        SharedIndex* focal_and_distortion_indices,
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
        double* out_focal_and_distortion_jac,
        unsigned int out_focal_and_distortion_jac_num_alloc,
        double* const out_focal_and_distortion_njtr,
        unsigned int out_focal_and_distortion_njtr_num_alloc,
        double* const out_focal_and_distortion_precond_diag,
        unsigned int out_focal_and_distortion_precond_diag_num_alloc,
        double* const out_focal_and_distortion_precond_tril,
        unsigned int out_focal_and_distortion_precond_tril_num_alloc,
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

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80;

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
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r6, r7);
  };
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
    r24 = fma(r6, r24, r0);
    r0 = 2.00000000000000000e+00;
    r25 = fma(r9, r14, r12 * r11);
    r26 = r13 * r10;
    r25 = fma(r4, r26, r25);
    r25 = fma(r8, r15, r25);
    r26 = r0 * r25;
    r26 = r26 * r20;
    r27 = r16 * r18;
    r28 = fma(r13, r15, r12 * r14);
    r28 = fma(r8, r10, r28);
    r28 = fma(r4, r28, r9 * r11);
    r27 = fma(r28, r27, r26);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r29);
    r30 = r0 * r16;
    r30 = r30 * r25;
    r31 = r0 * r28;
    r32 = fma(r20, r31, r30);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r33);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r34 = r14 * r10;
    r34 = r34 * r0;
    r35 = r15 * r11;
    r36 = fma(r0, r35, r34);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r37, r38);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r39 = r10 * r11;
    r40 = r14 * r15;
    r40 = r40 * r0;
    r39 = fma(r18, r39, r40);
    r41 = r15 * r15;
    r41 = r41 * r18;
    r42 = r19 + r41;
    r43 = r10 * r10;
    r43 = r43 * r18;
    r42 = r42 + r43;
    r24 = fma(r7, r27, r24);
    r24 = fma(r29, r32, r24);
    r24 = fma(r33, r36, r24);
    r24 = fma(r38, r39, r24);
    r24 = fma(r37, r42, r24);
  };
  LoadShared<2, double, double>(focal_and_distortion,
                                0 * focal_and_distortion_num_alloc,
                                focal_and_distortion_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_distortion_indices_loc[threadIdx.x].target,
                        r32,
                        r27);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r44 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r45);
    r46 = r18 * r20;
    r46 = fma(r28, r46, r30);
    r46 = fma(r6, r46, r45);
    r35 = fma(r18, r35, r34);
    r41 = r19 + r41;
    r34 = r14 * r14;
    r34 = r34 * r18;
    r41 = r41 + r34;
    r45 = r15 * r10;
    r45 = r45 * r0;
    r30 = r14 * r11;
    r30 = fma(r0, r30, r45);
    r47 = r0 * r16;
    r47 = r47 * r20;
    r48 = fma(r25, r31, r47);
    r49 = r25 * r25;
    r49 = r49 * r18;
    r23 = r49 + r23;
    r46 = fma(r37, r35, r46);
    r46 = fma(r33, r41, r46);
    r46 = fma(r38, r30, r46);
    r46 = fma(r7, r48, r46);
    r46 = fma(r29, r23, r46);
    r23 = copysign(1.0, r46);
    r23 = fma(r44, r23, r46);
    r44 = r23 * r23;
    r46 = 1.0 / r44;
    r48 = r24 * r46;
    r26 = fma(r16, r31, r26);
    r26 = fma(r6, r26, r5);
    r5 = r10 * r11;
    r5 = fma(r0, r5, r40);
    r43 = r19 + r43;
    r43 = r43 + r34;
    r34 = r14 * r11;
    r34 = fma(r18, r34, r45);
    r45 = r25 * r18;
    r45 = fma(r28, r45, r47);
    r17 = r19 + r17;
    r17 = r17 + r49;
    r26 = fma(r37, r5, r26);
    r26 = fma(r38, r43, r26);
    r26 = fma(r33, r34, r26);
    r26 = fma(r29, r45, r26);
    r26 = fma(r7, r17, r26);
    r17 = r26 * r26;
    r45 = fma(r46, r17, r24 * r48);
    r19 = fma(r27, r45, r19);
    r33 = r24 * r19;
    r38 = 1.0 / r23;
    r37 = r32 * r38;
    r2 = fma(r37, r33, r2);
    r3 = fma(r3, r4, r1);
    r1 = r26 * r19;
    r3 = fma(r37, r1, r3);
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
    r33 = -5.00000000000000000e-01;
    r49 = r9 * r14;
    r49 = fma(r33, r49, r33 * r1);
    r1 = r8 * r15;
    r49 = fma(r33, r1, r49);
    r47 = r13 * r10;
    r40 = 5.00000000000000000e-01;
    r49 = fma(r40, r47, r49);
    r47 = r20 * r49;
    r1 = r8 * r11;
    r50 = r13 * r14;
    r50 = fma(r40, r50, r40 * r1);
    r1 = r12 * r15;
    r50 = fma(r33, r1, r50);
    r51 = r9 * r40;
    r50 = fma(r10, r51, r50);
    r1 = fma(r50, r31, r0 * r47);
    r52 = r0 * r25;
    r53 = r9 * r15;
    r54 = r13 * r33;
    r53 = fma(r11, r54, r33 * r53);
    r53 = fma(r40, r22, r53);
    r53 = fma(r33, r21, r53);
    r55 = r0 * r16;
    r56 = r12 * r14;
    r57 = r8 * r10;
    r57 = fma(r33, r57, r33 * r56);
    r57 = fma(r11, r51, r57);
    r57 = fma(r15, r54, r57);
    r55 = r55 * r57;
    r52 = fma(r53, r52, r55);
    r1 = r1 + r52;
    r56 = r0 * r20;
    r56 = r56 * r57;
    r58 = r0 * r25;
    r58 = r58 * r50;
    r59 = r56 + r58;
    r60 = r16 * r18;
    r59 = fma(r49, r60, r59);
    r61 = r18 * r28;
    r59 = fma(r53, r61, r59);
    r59 = fma(r7, r59, r29 * r1);
    r1 = r20 * r50;
    r61 = -4.00000000000000000e+00;
    r1 = r1 * r61;
    r60 = r16 * r53;
    r62 = r61 * r60;
    r63 = r1 + r62;
    r59 = fma(r6, r63, r59);
    r63 = r19 * r59;
    r64 = r0 * r20;
    r64 = r64 * r53;
    r65 = r0 * r16;
    r65 = fma(r50, r65, r64);
    r66 = r0 * r25;
    r66 = r66 * r49;
    r67 = r57 * r31;
    r68 = r66 + r67;
    r69 = r65 + r68;
    r70 = r18 * r28;
    r70 = fma(r18, r47, r50 * r70);
    r70 = r70 + r52;
    r70 = fma(r6, r70, r7 * r69);
    r69 = r25 * r61;
    r50 = r57 * r69;
    r1 = r1 + r50;
    r70 = fma(r29, r1, r70);
    r1 = r70 * r48;
    r71 = r4 * r19;
    r72 = r32 * r71;
    r1 = fma(r72, r1, r37 * r63);
    r63 = r0 * r59;
    r73 = r24 * r24;
    r44 = r23 * r44;
    r44 = 1.0 / r44;
    r44 = r18 * r44;
    r73 = r73 * r44;
    r63 = fma(r70, r73, r48 * r63);
    r23 = r70 * r44;
    r63 = fma(r17, r23, r63);
    r74 = r0 * r26;
    r75 = r25 * r18;
    r76 = r18 * r28;
    r76 = r76 * r57;
    r75 = fma(r49, r75, r76);
    r75 = r75 + r65;
    r50 = r62 + r50;
    r50 = fma(r7, r50, r29 * r75);
    r58 = fma(r53, r31, r58);
    r75 = r0 * r16;
    r75 = fma(r49, r75, r56);
    r58 = r58 + r75;
    r50 = fma(r6, r58, r50);
    r74 = r74 * r50;
    r63 = fma(r46, r74, r63);
    r27 = r27 * r37;
    r63 = r63 * r27;
    r1 = fma(r24, r63, r1);
    r74 = r26 * r46;
    r74 = r74 * r72;
    r63 = fma(r26, r63, r70 * r74);
    r23 = r19 * r50;
    r63 = fma(r37, r23, r63);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r1, r63);
    r23 = r18 * r20;
    r23 = fma(r53, r23, r76);
    r58 = r0 * r16;
    r56 = r8 * r11;
    r62 = r12 * r15;
    r62 = fma(r40, r62, r33 * r56);
    r56 = r9 * r10;
    r62 = fma(r33, r56, r62);
    r62 = fma(r14, r54, r62);
    r58 = r58 * r62;
    r56 = r0 * r25;
    r65 = r12 * r11;
    r77 = r8 * r15;
    r77 = fma(r40, r77, r40 * r65);
    r77 = fma(r14, r51, r77);
    r77 = fma(r10, r54, r77);
    r56 = fma(r77, r56, r58);
    r23 = r23 + r56;
    r54 = r0 * r20;
    r54 = r54 * r77;
    r65 = fma(r62, r31, r54);
    r65 = r65 + r52;
    r65 = fma(r7, r65, r6 * r23);
    r23 = r20 * r57;
    r23 = r23 * r61;
    r52 = r62 * r69;
    r78 = r23 + r52;
    r65 = fma(r29, r78, r65);
    r78 = r65 * r48;
    r67 = r64 + r67;
    r67 = r67 + r56;
    r56 = r16 * r61;
    r56 = r56 * r77;
    r23 = r23 + r56;
    r23 = fma(r6, r23, r29 * r67);
    r67 = r18 * r28;
    r67 = fma(r18, r60, r77 * r67);
    r64 = r0 * r25;
    r64 = r64 * r57;
    r79 = r0 * r20;
    r79 = fma(r62, r79, r64);
    r67 = r67 + r79;
    r23 = fma(r7, r67, r23);
    r67 = r19 * r23;
    r67 = fma(r37, r67, r72 * r78);
    r78 = r0 * r23;
    r80 = r0 * r26;
    r54 = r55 + r54;
    r55 = r25 * r18;
    r54 = fma(r53, r55, r54);
    r53 = r18 * r28;
    r54 = fma(r62, r53, r54);
    r77 = fma(r77, r31, r0 * r60);
    r77 = r77 + r79;
    r77 = fma(r6, r77, r29 * r54);
    r52 = r56 + r52;
    r77 = fma(r7, r52, r77);
    r80 = r80 * r77;
    r80 = fma(r46, r80, r48 * r78);
    r78 = r65 * r44;
    r80 = fma(r17, r78, r80);
    r80 = fma(r65, r73, r80);
    r78 = r24 * r80;
    r67 = fma(r27, r78, r67);
    r78 = r26 * r80;
    r52 = r19 * r77;
    r52 = fma(r37, r52, r27 * r78);
    r52 = fma(r65, r74, r52);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r67, r52);
    r76 = r66 + r76;
    r66 = r0 * r20;
    r78 = r13 * r11;
    r22 = fma(r33, r22, r40 * r78);
    r22 = fma(r15, r51, r22);
    r22 = fma(r40, r21, r22);
    r66 = r66 * r22;
    r21 = r16 * r18;
    r76 = fma(r62, r21, r76);
    r76 = r76 + r66;
    r57 = r16 * r57;
    r57 = r57 * r61;
    r47 = r61 * r47;
    r61 = r57 + r47;
    r61 = fma(r6, r61, r7 * r76);
    r76 = r0 * r16;
    r76 = r76 * r22;
    r21 = fma(r49, r31, r76);
    r21 = r21 + r79;
    r61 = fma(r29, r21, r61);
    r21 = r19 * r61;
    r69 = r22 * r69;
    r47 = r47 + r69;
    r76 = r64 + r76;
    r64 = r18 * r20;
    r76 = fma(r62, r64, r76);
    r79 = r18 * r28;
    r76 = fma(r49, r79, r76);
    r76 = fma(r6, r76, r29 * r47);
    r47 = r0 * r25;
    r31 = fma(r22, r31, r62 * r47);
    r31 = r31 + r75;
    r76 = fma(r7, r31, r76);
    r31 = r76 * r48;
    r31 = fma(r72, r31, r37 * r21);
    r21 = r0 * r26;
    r66 = r58 + r66;
    r66 = r66 + r68;
    r68 = r25 * r18;
    r58 = r18 * r28;
    r58 = fma(r22, r58, r62 * r68);
    r58 = r58 + r75;
    r58 = fma(r29, r58, r6 * r66);
    r69 = r57 + r69;
    r58 = fma(r7, r69, r58);
    r21 = r21 * r58;
    r21 = fma(r76, r73, r46 * r21);
    r69 = r0 * r61;
    r21 = fma(r48, r69, r21);
    r7 = r76 * r44;
    r21 = fma(r17, r7, r21);
    r7 = r24 * r21;
    r31 = fma(r27, r7, r31);
    r7 = r19 * r58;
    r7 = fma(r37, r7, r76 * r74);
    r69 = r26 * r21;
    r7 = fma(r27, r69, r7);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r31, r7);
    r69 = r42 * r19;
    r57 = r35 * r48;
    r57 = fma(r72, r57, r37 * r69);
    r69 = r0 * r5;
    r69 = r69 * r26;
    r29 = r35 * r44;
    r29 = fma(r17, r29, r46 * r69);
    r69 = r0 * r42;
    r29 = fma(r48, r69, r29);
    r29 = fma(r35, r73, r29);
    r69 = r24 * r29;
    r57 = fma(r27, r69, r57);
    r69 = r26 * r29;
    r69 = fma(r27, r69, r35 * r74);
    r66 = r5 * r19;
    r69 = fma(r37, r66, r69);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r57, r69);
    r66 = r30 * r44;
    r6 = r0 * r43;
    r6 = r6 * r26;
    r6 = fma(r46, r6, r17 * r66);
    r66 = r0 * r39;
    r6 = fma(r48, r66, r6);
    r6 = fma(r30, r73, r6);
    r66 = r24 * r6;
    r75 = r30 * r48;
    r75 = fma(r72, r75, r27 * r66);
    r66 = r39 * r19;
    r75 = fma(r37, r66, r75);
    r66 = r43 * r19;
    r66 = fma(r30, r74, r37 * r66);
    r68 = r26 * r6;
    r66 = fma(r27, r68, r66);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r75, r66);
    r68 = r0 * r34;
    r68 = r68 * r26;
    r22 = r41 * r44;
    r22 = fma(r17, r22, r46 * r68);
    r68 = r0 * r36;
    r22 = fma(r48, r68, r22);
    r22 = fma(r41, r73, r22);
    r73 = r24 * r22;
    r68 = r36 * r19;
    r68 = fma(r37, r68, r27 * r73);
    r73 = r41 * r48;
    r68 = fma(r72, r73, r68);
    r73 = r26 * r22;
    r73 = fma(r27, r73, r41 * r74);
    r74 = r34 * r19;
    r73 = fma(r37, r74, r73);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r68, r73);
    r74 = r4 * r2;
    r27 = r4 * r3;
    r27 = fma(r63, r27, r1 * r74);
    r74 = r4 * r2;
    r72 = r4 * r3;
    r72 = fma(r52, r72, r67 * r74);
    WriteSum2<double, double>((double*)inout_shared, r27, r72);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = r4 * r3;
    r27 = r4 * r2;
    r27 = fma(r31, r27, r7 * r72);
    r72 = r4 * r2;
    r74 = r4 * r3;
    r74 = fma(r69, r74, r57 * r72);
    WriteSum2<double, double>((double*)inout_shared, r27, r74);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = r4 * r3;
    r27 = r4 * r2;
    r27 = fma(r75, r27, r66 * r74);
    r74 = r4 * r2;
    r72 = r4 * r3;
    r72 = fma(r73, r72, r68 * r74);
    WriteSum2<double, double>((double*)inout_shared, r27, r72);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fma(r1, r1, r63 * r63);
    r27 = fma(r67, r67, r52 * r52);
    WriteSum2<double, double>((double*)inout_shared, r72, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r31, r31, r7 * r7);
    r72 = fma(r57, r57, r69 * r69);
    WriteSum2<double, double>((double*)inout_shared, r27, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fma(r66, r66, r75 * r75);
    r27 = fma(r73, r73, r68 * r68);
    WriteSum2<double, double>((double*)inout_shared, r72, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r63, r52, r1 * r67);
    r72 = fma(r63, r7, r1 * r31);
    WriteSum2<double, double>((double*)inout_shared, r27, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fma(r63, r69, r1 * r57);
    r27 = fma(r1, r75, r63 * r66);
    WriteSum2<double, double>((double*)inout_shared, r72, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fma(r1, r68, r63 * r73);
    r63 = fma(r67, r31, r52 * r7);
    WriteSum2<double, double>((double*)inout_shared, r1, r63);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = fma(r67, r57, r52 * r69);
    r1 = fma(r67, r75, r52 * r66);
    WriteSum2<double, double>((double*)inout_shared, r63, r1);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = fma(r67, r68, r52 * r73);
    r52 = fma(r31, r57, r7 * r69);
    WriteSum2<double, double>((double*)inout_shared, r67, r52);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fma(r7, r66, r31 * r75);
    r31 = fma(r31, r68, r7 * r73);
    WriteSum2<double, double>((double*)inout_shared, r52, r31);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fma(r57, r75, r69 * r66);
    r69 = fma(r69, r73, r57 * r68);
    WriteSum2<double, double>((double*)inout_shared, r31, r69);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r66, r73, r75 * r68);
    WriteSum1<double, double>((double*)inout_shared, r73);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = r24 * r19;
    r73 = r73 * r38;
    r66 = r26 * r19;
    r66 = r66 * r38;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_distortion_jac,
        0 * out_focal_and_distortion_jac_num_alloc,
        global_thread_idx,
        r73,
        r66);
    r66 = r24 * r45;
    r66 = r66 * r37;
    r73 = r26 * r45;
    r73 = r73 * r37;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_distortion_jac,
        2 * out_focal_and_distortion_jac_num_alloc,
        global_thread_idx,
        r66,
        r73);
    r73 = r26 * r3;
    r73 = r73 * r38;
    r66 = r24 * r2;
    r66 = r66 * r38;
    r66 = fma(r71, r66, r71 * r73);
    r73 = r4 * r24;
    r73 = r73 * r45;
    r73 = r73 * r2;
    r71 = r4 * r26;
    r71 = r71 * r45;
    r71 = r71 * r3;
    r71 = fma(r37, r71, r37 * r73);
    WriteSum2<double, double>((double*)inout_shared, r66, r71);
  };
  FlushSumShared<2, double>(out_focal_and_distortion_njtr,
                            0 * out_focal_and_distortion_njtr_num_alloc,
                            focal_and_distortion_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = r24 * r19;
    r71 = r71 * r19;
    r66 = r19 * r19;
    r66 = r66 * r46;
    r66 = fma(r17, r66, r48 * r71);
    r71 = r45 * r45;
    r73 = r32 * r71;
    r46 = r32 * r46;
    r46 = r46 * r17;
    r17 = r32 * r32;
    r17 = r17 * r24;
    r17 = r17 * r48;
    r17 = fma(r71, r17, r46 * r73);
    WriteSum2<double, double>((double*)inout_shared, r66, r17);
  };
  FlushSumShared<2, double>(out_focal_and_distortion_precond_diag,
                            0 * out_focal_and_distortion_precond_diag_num_alloc,
                            focal_and_distortion_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = r45 * r19;
    r66 = r32 * r24;
    r66 = r66 * r45;
    r66 = r66 * r19;
    r66 = fma(r48, r66, r46 * r17);
    WriteSum1<double, double>((double*)inout_shared, r66);
  };
  FlushSumShared<1, double>(out_focal_and_distortion_precond_tril,
                            0 * out_focal_and_distortion_precond_tril_num_alloc,
                            focal_and_distortion_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedPrincipalPointFixedPointResJacFirst(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* focal_and_distortion,
    unsigned int focal_and_distortion_num_alloc,
    SharedIndex* focal_and_distortion_indices,
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
    double* out_focal_and_distortion_jac,
    unsigned int out_focal_and_distortion_jac_num_alloc,
    double* const out_focal_and_distortion_njtr,
    unsigned int out_focal_and_distortion_njtr_num_alloc,
    double* const out_focal_and_distortion_precond_diag,
    unsigned int out_focal_and_distortion_precond_diag_num_alloc,
    double* const out_focal_and_distortion_precond_tril,
    unsigned int out_focal_and_distortion_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedPrincipalPointFixedPointResJacFirstKernel<<<n_blocks,
                                                                    1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      focal_and_distortion,
      focal_and_distortion_num_alloc,
      focal_and_distortion_indices,
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
      out_focal_and_distortion_jac,
      out_focal_and_distortion_jac_num_alloc,
      out_focal_and_distortion_njtr,
      out_focal_and_distortion_njtr_num_alloc,
      out_focal_and_distortion_precond_diag,
      out_focal_and_distortion_precond_diag_num_alloc,
      out_focal_and_distortion_precond_tril,
      out_focal_and_distortion_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar