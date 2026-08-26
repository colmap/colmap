#include "kernel_simple_radial_split_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPrincipalPointFixedPointResJacKernel(
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
      r76, r77, r78, r79, r80, r81;

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
  LoadShared<2, double, double>(focal_and_extra,
                                0 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target,
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
    r63 = r0 * r59;
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
    r1 = r24 * r24;
    r44 = r23 * r44;
    r44 = 1.0 / r44;
    r44 = r18 * r44;
    r1 = r1 * r44;
    r63 = fma(r70, r1, r48 * r63);
    r23 = r70 * r44;
    r63 = fma(r17, r23, r63);
    r71 = r0 * r26;
    r72 = r25 * r18;
    r73 = r18 * r28;
    r73 = r73 * r57;
    r72 = fma(r49, r72, r73);
    r72 = r72 + r65;
    r50 = r62 + r50;
    r50 = fma(r7, r50, r29 * r72);
    r58 = fma(r53, r31, r58);
    r72 = r0 * r16;
    r72 = fma(r49, r72, r56);
    r58 = r58 + r72;
    r50 = fma(r6, r58, r50);
    r71 = r71 * r50;
    r63 = fma(r46, r71, r63);
    r27 = r27 * r37;
    r63 = r63 * r27;
    r71 = r19 * r59;
    r71 = fma(r37, r71, r24 * r63);
    r23 = r70 * r48;
    r58 = r4 * r19;
    r56 = r32 * r58;
    r71 = fma(r56, r23, r71);
    r23 = r19 * r50;
    r23 = fma(r37, r23, r26 * r63);
    r63 = r26 * r46;
    r63 = r63 * r56;
    r23 = fma(r70, r63, r23);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r71, r23);
    r67 = r64 + r67;
    r64 = r0 * r16;
    r62 = r8 * r11;
    r65 = r12 * r15;
    r65 = fma(r40, r65, r33 * r62);
    r62 = r9 * r10;
    r65 = fma(r33, r62, r65);
    r65 = fma(r14, r54, r65);
    r64 = r64 * r65;
    r62 = r0 * r25;
    r74 = r12 * r11;
    r75 = r8 * r15;
    r75 = fma(r40, r75, r40 * r74);
    r75 = fma(r14, r51, r75);
    r75 = fma(r10, r54, r75);
    r62 = fma(r75, r62, r64);
    r67 = r67 + r62;
    r54 = r20 * r57;
    r54 = r54 * r61;
    r74 = r16 * r61;
    r74 = r74 * r75;
    r76 = r54 + r74;
    r76 = fma(r6, r76, r29 * r67);
    r67 = r18 * r28;
    r67 = fma(r18, r60, r75 * r67);
    r77 = r0 * r25;
    r77 = r77 * r57;
    r78 = r0 * r20;
    r78 = fma(r65, r78, r77);
    r67 = r67 + r78;
    r76 = fma(r7, r67, r76);
    r67 = r0 * r76;
    r79 = r0 * r26;
    r80 = r25 * r18;
    r80 = fma(r53, r80, r55);
    r55 = r0 * r20;
    r55 = r55 * r75;
    r81 = r18 * r28;
    r80 = fma(r65, r81, r80);
    r80 = r80 + r55;
    r75 = fma(r75, r31, r0 * r60);
    r75 = r75 + r78;
    r75 = fma(r6, r75, r29 * r80);
    r80 = r65 * r69;
    r74 = r74 + r80;
    r75 = fma(r7, r74, r75);
    r79 = r79 * r75;
    r79 = fma(r46, r79, r48 * r67);
    r67 = r18 * r20;
    r67 = fma(r53, r67, r73);
    r67 = r67 + r62;
    r55 = fma(r65, r31, r55);
    r55 = r55 + r52;
    r55 = fma(r7, r55, r6 * r67);
    r80 = r54 + r80;
    r55 = fma(r29, r80, r55);
    r80 = r55 * r44;
    r79 = fma(r17, r80, r79);
    r79 = fma(r55, r1, r79);
    r80 = r24 * r79;
    r54 = r55 * r48;
    r54 = fma(r56, r54, r27 * r80);
    r80 = r19 * r76;
    r54 = fma(r37, r80, r54);
    r80 = r19 * r75;
    r67 = r26 * r79;
    r67 = fma(r27, r67, r37 * r80);
    r67 = fma(r55, r63, r67);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r54, r67);
    r80 = r13 * r11;
    r22 = fma(r33, r22, r40 * r80);
    r22 = fma(r15, r51, r22);
    r22 = fma(r40, r21, r22);
    r69 = r22 * r69;
    r47 = r61 * r47;
    r21 = r69 + r47;
    r40 = r0 * r16;
    r40 = r40 * r22;
    r77 = r77 + r40;
    r51 = r18 * r20;
    r77 = fma(r65, r51, r77);
    r33 = r18 * r28;
    r77 = fma(r49, r33, r77);
    r77 = fma(r6, r77, r29 * r21);
    r21 = r0 * r25;
    r21 = fma(r22, r31, r65 * r21);
    r21 = r21 + r72;
    r77 = fma(r7, r21, r77);
    r21 = r77 * r48;
    r73 = r66 + r73;
    r66 = r0 * r20;
    r66 = r66 * r22;
    r33 = r16 * r18;
    r73 = fma(r65, r33, r73);
    r73 = r73 + r66;
    r57 = r16 * r57;
    r57 = r57 * r61;
    r47 = r57 + r47;
    r47 = fma(r6, r47, r7 * r73);
    r31 = fma(r49, r31, r40);
    r31 = r31 + r78;
    r47 = fma(r29, r31, r47);
    r31 = r19 * r47;
    r31 = fma(r37, r31, r56 * r21);
    r21 = r0 * r26;
    r66 = r64 + r66;
    r66 = r66 + r68;
    r68 = r25 * r18;
    r64 = r18 * r28;
    r64 = fma(r22, r64, r65 * r68);
    r64 = r64 + r72;
    r64 = fma(r29, r64, r6 * r66);
    r69 = r57 + r69;
    r64 = fma(r7, r69, r64);
    r21 = r21 * r64;
    r21 = fma(r77, r1, r46 * r21);
    r69 = r0 * r47;
    r21 = fma(r48, r69, r21);
    r7 = r77 * r44;
    r21 = fma(r17, r7, r21);
    r7 = r24 * r21;
    r31 = fma(r27, r7, r31);
    r7 = r26 * r21;
    r7 = fma(r77, r63, r27 * r7);
    r69 = r19 * r64;
    r7 = fma(r37, r69, r7);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r31, r7);
    r69 = r35 * r48;
    r57 = r0 * r5;
    r57 = r57 * r26;
    r29 = r35 * r44;
    r29 = fma(r17, r29, r46 * r57);
    r57 = r0 * r42;
    r29 = fma(r48, r57, r29);
    r29 = fma(r35, r1, r29);
    r57 = r24 * r29;
    r57 = fma(r27, r57, r56 * r69);
    r69 = r42 * r19;
    r57 = fma(r37, r69, r57);
    r69 = r5 * r19;
    r66 = r26 * r29;
    r66 = fma(r27, r66, r37 * r69);
    r66 = fma(r35, r63, r66);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r57, r66);
    r69 = r39 * r19;
    r6 = r30 * r48;
    r6 = fma(r56, r6, r37 * r69);
    r69 = r30 * r44;
    r72 = r0 * r43;
    r72 = r72 * r26;
    r72 = fma(r46, r72, r17 * r69);
    r69 = r0 * r39;
    r72 = fma(r48, r69, r72);
    r72 = fma(r30, r1, r72);
    r69 = r24 * r72;
    r6 = fma(r27, r69, r6);
    r69 = r26 * r72;
    r68 = r43 * r19;
    r68 = fma(r37, r68, r27 * r69);
    r68 = fma(r30, r63, r68);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r6, r68);
    r69 = r0 * r34;
    r69 = r69 * r26;
    r22 = r41 * r44;
    r22 = fma(r17, r22, r46 * r69);
    r69 = r0 * r36;
    r22 = fma(r48, r69, r22);
    r22 = fma(r41, r1, r22);
    r1 = r24 * r22;
    r69 = r36 * r19;
    r69 = fma(r37, r69, r27 * r1);
    r1 = r41 * r48;
    r69 = fma(r56, r1, r69);
    r1 = r26 * r22;
    r1 = fma(r27, r1, r41 * r63);
    r63 = r34 * r19;
    r1 = fma(r37, r63, r1);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r69, r1);
    r63 = r4 * r3;
    r27 = r4 * r2;
    r27 = fma(r71, r27, r23 * r63);
    r63 = r4 * r2;
    r56 = r4 * r3;
    r56 = fma(r67, r56, r54 * r63);
    WriteSum2<double, double>((double*)inout_shared, r27, r56);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = r4 * r3;
    r27 = r4 * r2;
    r27 = fma(r31, r27, r7 * r56);
    r56 = r4 * r2;
    r63 = r4 * r3;
    r63 = fma(r66, r63, r57 * r56);
    WriteSum2<double, double>((double*)inout_shared, r27, r63);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = r4 * r3;
    r27 = r4 * r2;
    r27 = fma(r6, r27, r68 * r63);
    r63 = r4 * r2;
    r56 = r4 * r3;
    r56 = fma(r1, r56, r69 * r63);
    WriteSum2<double, double>((double*)inout_shared, r27, r56);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = fma(r71, r71, r23 * r23);
    r27 = fma(r67, r67, r54 * r54);
    WriteSum2<double, double>((double*)inout_shared, r56, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r31, r31, r7 * r7);
    r56 = fma(r57, r57, r66 * r66);
    WriteSum2<double, double>((double*)inout_shared, r27, r56);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = fma(r6, r6, r68 * r68);
    r27 = fma(r69, r69, r1 * r1);
    WriteSum2<double, double>((double*)inout_shared, r56, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r71, r54, r23 * r67);
    r56 = fma(r23, r7, r71 * r31);
    WriteSum2<double, double>((double*)inout_shared, r27, r56);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = fma(r23, r66, r71 * r57);
    r27 = fma(r71, r6, r23 * r68);
    WriteSum2<double, double>((double*)inout_shared, r56, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = fma(r71, r69, r23 * r1);
    r23 = fma(r67, r7, r54 * r31);
    WriteSum2<double, double>((double*)inout_shared, r71, r23);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r67, r66, r54 * r57);
    r71 = fma(r67, r68, r54 * r6);
    WriteSum2<double, double>((double*)inout_shared, r23, r71);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = fma(r67, r1, r54 * r69);
    r54 = fma(r31, r57, r7 * r66);
    WriteSum2<double, double>((double*)inout_shared, r67, r54);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = fma(r31, r6, r7 * r68);
    r7 = fma(r7, r1, r31 * r69);
    WriteSum2<double, double>((double*)inout_shared, r54, r7);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = fma(r66, r68, r57 * r6);
    r66 = fma(r66, r1, r57 * r69);
    WriteSum2<double, double>((double*)inout_shared, r7, r66);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fma(r68, r1, r6 * r69);
    WriteSum1<double, double>((double*)inout_shared, r1);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r24 * r19;
    r1 = r1 * r38;
    r68 = r26 * r19;
    r68 = r68 * r38;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        0 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r1,
        r68);
    r68 = r24 * r45;
    r68 = r68 * r37;
    r1 = r26 * r45;
    r1 = r1 * r37;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        2 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r68,
        r1);
    r1 = r24 * r2;
    r1 = r1 * r38;
    r68 = r26 * r3;
    r68 = r68 * r38;
    r68 = fma(r58, r68, r58 * r1);
    r1 = r4 * r24;
    r1 = r1 * r45;
    r1 = r1 * r2;
    r58 = r4 * r26;
    r58 = r58 * r45;
    r58 = r58 * r3;
    r58 = fma(r37, r58, r37 * r1);
    WriteSum2<double, double>((double*)inout_shared, r68, r58);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            0 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = r24 * r19;
    r58 = r58 * r19;
    r68 = r19 * r19;
    r68 = r68 * r46;
    r68 = fma(r17, r68, r48 * r58);
    r58 = r32 * r32;
    r1 = r45 * r45;
    r58 = r58 * r24;
    r58 = r58 * r48;
    r37 = r32 * r1;
    r46 = r32 * r46;
    r46 = r46 * r17;
    r37 = fma(r46, r37, r1 * r58);
    WriteSum2<double, double>((double*)inout_shared, r68, r37);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            0 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = r32 * r24;
    r37 = r37 * r45;
    r37 = r37 * r19;
    r68 = r45 * r19;
    r68 = fma(r46, r68, r48 * r37);
    WriteSum1<double, double>((double*)inout_shared, r68);
  };
  FlushSumShared<1, double>(out_focal_and_extra_precond_tril,
                            0 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
}

void SimpleRadialSplitFixedPrincipalPointFixedPointResJac(
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
  SimpleRadialSplitFixedPrincipalPointFixedPointResJacKernel<<<n_blocks,
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