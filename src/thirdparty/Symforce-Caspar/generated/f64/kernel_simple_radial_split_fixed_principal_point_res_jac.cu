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
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92;

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
    r20 = r9 * r15;
    r21 = fma(r13, r11, r20);
    r22 = r12 * r10;
    r23 = r8 * r14;
    r21 = r21 + r22;
    r21 = fma(r4, r23, r21);
    r24 = r18 * r21;
    r24 = fma(r21, r24, r19);
    r25 = r17 + r24;
    r0 = fma(r6, r25, r0);
    r26 = 2.00000000000000000e+00;
    r27 = fma(r9, r14, r12 * r11);
    r28 = r13 * r10;
    r27 = fma(r4, r28, r27);
    r27 = fma(r8, r15, r27);
    r28 = r26 * r27;
    r28 = r28 * r21;
    r29 = r16 * r18;
    r30 = fma(r13, r15, r12 * r14);
    r30 = fma(r8, r10, r30);
    r30 = fma(r4, r30, r9 * r11);
    r29 = fma(r30, r29, r28);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r31);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r32 = r26 * r16;
    r32 = r32 * r27;
    r33 = r26 * r30;
    r34 = fma(r21, r33, r32);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r35);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r36 = r14 * r10;
    r36 = r36 * r26;
    r37 = r15 * r11;
    r37 = fma(r26, r37, r36);
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
    r41 = r41 * r26;
    r40 = fma(r18, r40, r41);
    r42 = r15 * r15;
    r42 = r42 * r18;
    r43 = r19 + r42;
    r44 = r10 * r10;
    r44 = r44 * r18;
    r43 = r43 + r44;
    r0 = fma(r7, r29, r0);
    r0 = fma(r31, r34, r0);
    r0 = fma(r35, r37, r0);
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
    r49 = r18 * r21;
    r49 = fma(r30, r49, r32);
    r48 = fma(r6, r49, r48);
    r32 = r15 * r11;
    r32 = fma(r18, r32, r36);
    r42 = r19 + r42;
    r36 = r14 * r14;
    r36 = r36 * r18;
    r42 = r42 + r36;
    r50 = r15 * r10;
    r50 = r50 * r26;
    r51 = r14 * r11;
    r51 = fma(r26, r51, r50);
    r52 = r26 * r16;
    r52 = r52 * r21;
    r53 = fma(r27, r33, r52);
    r54 = r27 * r27;
    r54 = r54 * r18;
    r24 = r54 + r24;
    r48 = fma(r38, r32, r48);
    r48 = fma(r35, r42, r48);
    r48 = fma(r39, r51, r48);
    r48 = fma(r7, r53, r48);
    r48 = fma(r31, r24, r48);
    r55 = copysign(1.0, r48);
    r55 = fma(r47, r55, r48);
    r47 = r55 * r55;
    r48 = 1.0 / r47;
    r56 = r0 * r48;
    r28 = fma(r16, r33, r28);
    r5 = fma(r6, r28, r5);
    r57 = r10 * r11;
    r57 = fma(r26, r57, r41);
    r44 = r19 + r44;
    r44 = r44 + r36;
    r36 = r14 * r11;
    r36 = fma(r18, r36, r50);
    r50 = r27 * r18;
    r50 = fma(r30, r50, r52);
    r17 = r19 + r17;
    r17 = r17 + r54;
    r5 = fma(r38, r57, r5);
    r5 = fma(r39, r44, r5);
    r5 = fma(r35, r36, r5);
    r5 = fma(r31, r50, r5);
    r5 = fma(r7, r17, r5);
    r35 = r5 * r5;
    r39 = fma(r48, r35, r0 * r56);
    r19 = fma(r46, r39, r19);
    r38 = r0 * r19;
    r54 = 1.0 / r55;
    r52 = r45 * r54;
    r2 = fma(r52, r38, r2);
    r3 = fma(r3, r4, r1);
    r1 = r5 * r19;
    r3 = fma(r52, r1, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = r26 * r21;
    r38 = -5.00000000000000000e-01;
    r41 = r13 * r38;
    r58 = 5.00000000000000000e-01;
    r59 = fma(r58, r23, r11 * r41);
    r59 = fma(r38, r20, r59);
    r59 = fma(r38, r22, r59);
    r1 = r1 * r59;
    r60 = r26 * r16;
    r61 = r13 * r14;
    r62 = r12 * r15;
    r62 = fma(r38, r62, r58 * r61);
    r61 = r9 * r10;
    r62 = fma(r58, r61, r62);
    r63 = r11 * r58;
    r62 = fma(r8, r63, r62);
    r60 = fma(r62, r60, r1);
    r61 = r26 * r27;
    r64 = r12 * r11;
    r65 = r9 * r14;
    r65 = fma(r38, r65, r38 * r64);
    r64 = r8 * r15;
    r65 = fma(r38, r64, r65);
    r66 = r13 * r10;
    r65 = fma(r58, r66, r65);
    r61 = r61 * r65;
    r66 = r12 * r14;
    r64 = r8 * r10;
    r64 = fma(r38, r64, r38 * r66);
    r64 = fma(r9, r63, r64);
    r64 = fma(r15, r41, r64);
    r66 = r64 * r33;
    r67 = r61 + r66;
    r68 = r60 + r67;
    r69 = r18 * r30;
    r70 = r21 * r65;
    r69 = fma(r18, r70, r62 * r69);
    r71 = r26 * r27;
    r72 = r26 * r16;
    r72 = r72 * r64;
    r71 = fma(r59, r71, r72);
    r69 = r69 + r71;
    r69 = fma(r6, r69, r7 * r68);
    r68 = r21 * r62;
    r73 = -4.00000000000000000e+00;
    r68 = r68 * r73;
    r74 = r27 * r73;
    r75 = r64 * r74;
    r76 = r68 + r75;
    r69 = fma(r31, r76, r69);
    r76 = r69 * r56;
    r77 = r4 * r19;
    r78 = r45 * r77;
    r79 = fma(r62, r33, r26 * r70);
    r79 = r79 + r71;
    r80 = r26 * r21;
    r80 = r80 * r64;
    r81 = r26 * r27;
    r81 = r81 * r62;
    r62 = r80 + r81;
    r82 = r16 * r18;
    r62 = fma(r65, r82, r62);
    r83 = r18 * r30;
    r62 = fma(r59, r83, r62);
    r62 = fma(r7, r62, r31 * r79);
    r79 = r16 * r59;
    r83 = r73 * r79;
    r68 = r68 + r83;
    r62 = fma(r6, r68, r62);
    r68 = r19 * r62;
    r68 = fma(r52, r68, r78 * r76);
    r76 = r26 * r62;
    r82 = r0 * r0;
    r47 = r55 * r47;
    r47 = 1.0 / r47;
    r47 = r18 * r47;
    r82 = r82 * r47;
    r76 = fma(r69, r82, r56 * r76);
    r55 = r69 * r47;
    r76 = fma(r35, r55, r76);
    r84 = r26 * r5;
    r85 = r27 * r18;
    r86 = r18 * r30;
    r86 = r86 * r64;
    r85 = fma(r65, r85, r86);
    r85 = r85 + r60;
    r83 = r75 + r83;
    r83 = fma(r7, r83, r31 * r85);
    r81 = fma(r59, r33, r81);
    r85 = r26 * r16;
    r85 = fma(r65, r85, r80);
    r81 = r81 + r85;
    r83 = fma(r6, r81, r83);
    r84 = r84 * r83;
    r76 = fma(r48, r84, r76);
    r46 = r46 * r52;
    r76 = r76 * r46;
    r68 = fma(r0, r76, r68);
    r84 = r19 * r83;
    r84 = fma(r52, r84, r5 * r76);
    r76 = r5 * r48;
    r76 = r76 * r78;
    r84 = fma(r69, r76, r84);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r68, r84);
    r66 = r1 + r66;
    r1 = r26 * r16;
    r55 = r8 * r11;
    r81 = r12 * r15;
    r81 = fma(r58, r81, r38 * r55);
    r55 = r9 * r10;
    r81 = fma(r38, r55, r81);
    r81 = fma(r14, r41, r81);
    r1 = r1 * r81;
    r55 = r26 * r27;
    r80 = r9 * r14;
    r75 = r8 * r15;
    r75 = fma(r58, r75, r58 * r80);
    r75 = fma(r12, r63, r75);
    r75 = fma(r10, r41, r75);
    r55 = fma(r75, r55, r1);
    r66 = r66 + r55;
    r41 = r21 * r64;
    r41 = r41 * r73;
    r80 = r16 * r73;
    r80 = r80 * r75;
    r60 = r41 + r80;
    r60 = fma(r6, r60, r31 * r66);
    r66 = r18 * r30;
    r66 = fma(r18, r79, r75 * r66);
    r87 = r26 * r27;
    r87 = r87 * r64;
    r88 = r26 * r21;
    r88 = fma(r81, r88, r87);
    r66 = r66 + r88;
    r60 = fma(r7, r66, r60);
    r66 = r19 * r60;
    r89 = r26 * r60;
    r90 = r26 * r5;
    r91 = r27 * r18;
    r91 = fma(r59, r91, r72);
    r72 = r26 * r21;
    r72 = r72 * r75;
    r92 = r18 * r30;
    r91 = fma(r81, r92, r91);
    r91 = r91 + r72;
    r75 = fma(r75, r33, r26 * r79);
    r75 = r75 + r88;
    r75 = fma(r6, r75, r31 * r91);
    r91 = r81 * r74;
    r80 = r80 + r91;
    r75 = fma(r7, r80, r75);
    r90 = r90 * r75;
    r90 = fma(r48, r90, r56 * r89);
    r89 = r18 * r21;
    r89 = fma(r59, r89, r86);
    r89 = r89 + r55;
    r72 = fma(r81, r33, r72);
    r72 = r72 + r71;
    r72 = fma(r7, r72, r6 * r89);
    r91 = r41 + r91;
    r72 = fma(r31, r91, r72);
    r91 = r72 * r47;
    r90 = fma(r35, r91, r90);
    r90 = fma(r72, r82, r90);
    r91 = r0 * r90;
    r91 = fma(r46, r91, r52 * r66);
    r66 = r72 * r56;
    r91 = fma(r78, r66, r91);
    r66 = r19 * r75;
    r41 = r5 * r90;
    r41 = fma(r46, r41, r52 * r66);
    r41 = fma(r72, r76, r41);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r91, r41);
    r66 = r26 * r5;
    r89 = r26 * r21;
    r23 = fma(r38, r23, r13 * r63);
    r23 = fma(r58, r20, r23);
    r23 = fma(r58, r22, r23);
    r89 = r89 * r23;
    r1 = r1 + r89;
    r1 = r1 + r67;
    r67 = r27 * r18;
    r22 = r18 * r30;
    r22 = fma(r23, r22, r81 * r67);
    r22 = r22 + r85;
    r22 = fma(r31, r22, r6 * r1);
    r64 = r16 * r64;
    r64 = r64 * r73;
    r74 = r23 * r74;
    r1 = r64 + r74;
    r22 = fma(r7, r1, r22);
    r66 = r66 * r22;
    r70 = r73 * r70;
    r74 = r74 + r70;
    r73 = r26 * r16;
    r73 = r73 * r23;
    r87 = r87 + r73;
    r1 = r18 * r21;
    r87 = fma(r81, r1, r87);
    r67 = r18 * r30;
    r87 = fma(r65, r67, r87);
    r87 = fma(r6, r87, r31 * r74);
    r74 = r26 * r27;
    r23 = fma(r23, r33, r81 * r74);
    r23 = r23 + r85;
    r87 = fma(r7, r23, r87);
    r66 = fma(r87, r82, r48 * r66);
    r86 = r61 + r86;
    r61 = r16 * r18;
    r86 = fma(r81, r61, r86);
    r86 = r86 + r89;
    r70 = r64 + r70;
    r70 = fma(r6, r70, r7 * r86);
    r33 = fma(r65, r33, r73);
    r33 = r33 + r88;
    r70 = fma(r31, r33, r70);
    r33 = r26 * r70;
    r66 = fma(r56, r33, r66);
    r31 = r87 * r47;
    r66 = fma(r35, r31, r66);
    r31 = r0 * r66;
    r33 = r19 * r70;
    r33 = fma(r52, r33, r46 * r31);
    r31 = r87 * r56;
    r33 = fma(r78, r31, r33);
    r31 = r5 * r66;
    r31 = fma(r87, r76, r46 * r31);
    r88 = r19 * r22;
    r31 = fma(r52, r88, r31);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r33, r31);
    r88 = r43 * r19;
    r65 = r26 * r57;
    r65 = r65 * r5;
    r73 = r32 * r47;
    r73 = fma(r35, r73, r48 * r65);
    r65 = r26 * r43;
    r73 = fma(r56, r65, r73);
    r73 = fma(r32, r82, r73);
    r65 = r0 * r73;
    r65 = fma(r46, r65, r52 * r88);
    r88 = r32 * r56;
    r65 = fma(r78, r88, r65);
    r88 = r57 * r19;
    r88 = fma(r52, r88, r32 * r76);
    r6 = r5 * r73;
    r88 = fma(r46, r6, r88);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r65, r88);
    r6 = r51 * r56;
    r86 = r40 * r19;
    r86 = fma(r52, r86, r78 * r6);
    r6 = r51 * r47;
    r7 = r26 * r44;
    r7 = r7 * r5;
    r7 = fma(r48, r7, r35 * r6);
    r6 = r26 * r40;
    r7 = fma(r56, r6, r7);
    r7 = fma(r51, r82, r7);
    r6 = r0 * r7;
    r86 = fma(r46, r6, r86);
    r6 = r44 * r19;
    r6 = fma(r52, r6, r51 * r76);
    r64 = r5 * r7;
    r6 = fma(r46, r64, r6);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r86, r6);
    r64 = r26 * r36;
    r64 = r64 * r5;
    r61 = r42 * r47;
    r61 = fma(r35, r61, r48 * r64);
    r64 = r26 * r37;
    r61 = fma(r56, r64, r61);
    r61 = fma(r42, r82, r61);
    r64 = r0 * r61;
    r89 = r37 * r19;
    r89 = fma(r52, r89, r46 * r64);
    r64 = r42 * r56;
    r89 = fma(r78, r64, r89);
    r64 = r36 * r19;
    r81 = r5 * r61;
    r81 = fma(r46, r81, r52 * r64);
    r81 = fma(r42, r76, r81);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r89, r81);
    r64 = r4 * r2;
    r23 = r4 * r3;
    r23 = fma(r84, r23, r68 * r64);
    r64 = r4 * r2;
    r85 = r4 * r3;
    r85 = fma(r41, r85, r91 * r64);
    WriteSum2<double, double>((double*)inout_shared, r23, r85);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = r4 * r3;
    r23 = r4 * r2;
    r23 = fma(r33, r23, r31 * r85);
    r85 = r4 * r3;
    r64 = r4 * r2;
    r64 = fma(r65, r64, r88 * r85);
    WriteSum2<double, double>((double*)inout_shared, r23, r64);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = r4 * r2;
    r23 = r4 * r3;
    r23 = fma(r6, r23, r86 * r64);
    r64 = r4 * r2;
    r85 = r4 * r3;
    r85 = fma(r81, r85, r89 * r64);
    WriteSum2<double, double>((double*)inout_shared, r23, r85);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = fma(r68, r68, r84 * r84);
    r23 = fma(r91, r91, r41 * r41);
    WriteSum2<double, double>((double*)inout_shared, r85, r23);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r33, r33, r31 * r31);
    r85 = fma(r65, r65, r88 * r88);
    WriteSum2<double, double>((double*)inout_shared, r23, r85);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = fma(r6, r6, r86 * r86);
    r23 = fma(r89, r89, r81 * r81);
    WriteSum2<double, double>((double*)inout_shared, r85, r23);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r68, r91, r84 * r41);
    r85 = fma(r68, r33, r84 * r31);
    WriteSum2<double, double>((double*)inout_shared, r23, r85);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r85 = fma(r68, r65, r84 * r88);
    r23 = fma(r84, r6, r68 * r86);
    WriteSum2<double, double>((double*)inout_shared, r85, r23);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = fma(r68, r89, r84 * r81);
    r84 = fma(r41, r31, r91 * r33);
    WriteSum2<double, double>((double*)inout_shared, r68, r84);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r84 = fma(r41, r88, r91 * r65);
    r68 = fma(r41, r6, r91 * r86);
    WriteSum2<double, double>((double*)inout_shared, r84, r68);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r41, r81, r91 * r89);
    r91 = fma(r33, r65, r31 * r88);
    WriteSum2<double, double>((double*)inout_shared, r41, r91);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r91 = fma(r33, r86, r31 * r6);
    r31 = fma(r31, r81, r33 * r89);
    WriteSum2<double, double>((double*)inout_shared, r91, r31);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fma(r65, r86, r88 * r6);
    r88 = fma(r88, r81, r65 * r89);
    WriteSum2<double, double>((double*)inout_shared, r31, r88);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = fma(r6, r81, r86 * r89);
    WriteSum1<double, double>((double*)inout_shared, r81);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = r0 * r19;
    r81 = r81 * r54;
    r6 = r5 * r19;
    r6 = r6 * r54;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        0 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r81,
        r6);
    r6 = r0 * r39;
    r6 = r6 * r52;
    r81 = r5 * r39;
    r81 = r81 * r52;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        2 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r6,
        r81);
    r81 = r0 * r2;
    r81 = r81 * r54;
    r6 = r5 * r3;
    r6 = r6 * r54;
    r6 = fma(r77, r6, r77 * r81);
    r81 = r4 * r5;
    r81 = r81 * r39;
    r81 = r81 * r3;
    r77 = r4 * r0;
    r77 = r77 * r39;
    r77 = r77 * r2;
    r77 = fma(r52, r77, r52 * r81);
    WriteSum2<double, double>((double*)inout_shared, r6, r77);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            0 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = r0 * r19;
    r77 = r77 * r19;
    r6 = r19 * r19;
    r6 = r6 * r48;
    r6 = fma(r35, r6, r56 * r77);
    r77 = r39 * r39;
    r81 = r45 * r77;
    r54 = r45 * r48;
    r54 = r54 * r35;
    r89 = r45 * r45;
    r89 = r89 * r0;
    r89 = r89 * r56;
    r89 = fma(r77, r89, r54 * r81);
    WriteSum2<double, double>((double*)inout_shared, r6, r89);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            0 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r89 = r45 * r0;
    r89 = r89 * r39;
    r89 = r89 * r19;
    r6 = r39 * r19;
    r6 = fma(r54, r6, r56 * r89);
    WriteSum1<double, double>((double*)inout_shared, r6);
  };
  FlushSumShared<1, double>(out_focal_and_extra_precond_tril,
                            0 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r26 * r25;
    r89 = r49 * r47;
    r89 = fma(r35, r89, r56 * r6);
    r6 = r26 * r28;
    r6 = r6 * r5;
    r89 = fma(r48, r6, r89);
    r89 = fma(r49, r82, r89);
    r6 = r0 * r89;
    r54 = r49 * r56;
    r54 = fma(r78, r54, r46 * r6);
    r6 = r25 * r19;
    r54 = fma(r52, r6, r54);
    r6 = r5 * r89;
    r6 = fma(r46, r6, r49 * r76);
    r81 = r28 * r19;
    r6 = fma(r52, r81, r6);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r54, r6);
    r81 = r26 * r29;
    r81 = fma(r56, r81, r53 * r82);
    r86 = r26 * r17;
    r86 = r86 * r5;
    r81 = fma(r48, r86, r81);
    r88 = r53 * r47;
    r81 = fma(r35, r88, r81);
    r88 = r0 * r81;
    r86 = r29 * r19;
    r86 = fma(r52, r86, r46 * r88);
    r88 = r53 * r56;
    r86 = fma(r78, r88, r86);
    r88 = r5 * r81;
    r88 = fma(r53, r76, r46 * r88);
    r31 = r17 * r19;
    r88 = fma(r52, r31, r88);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r86,
                                             r88);
    r31 = r26 * r34;
    r65 = r24 * r47;
    r65 = fma(r35, r65, r56 * r31);
    r31 = r26 * r50;
    r31 = r31 * r5;
    r65 = fma(r48, r31, r65);
    r65 = fma(r24, r82, r65);
    r31 = r0 * r65;
    r82 = r34 * r19;
    r82 = fma(r52, r82, r46 * r31);
    r31 = r24 * r56;
    r82 = fma(r78, r31, r82);
    r31 = r5 * r65;
    r76 = fma(r24, r76, r46 * r31);
    r31 = r50 * r19;
    r76 = fma(r52, r31, r76);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r82,
                                             r76);
    r31 = r4 * r2;
    r52 = r4 * r3;
    r52 = fma(r6, r52, r54 * r31);
    r31 = r4 * r3;
    r46 = r4 * r2;
    r46 = fma(r86, r46, r88 * r31);
    WriteSum2<double, double>((double*)inout_shared, r52, r46);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r4 * r2;
    r52 = r4 * r3;
    r52 = fma(r76, r52, r82 * r46);
    WriteSum1<double, double>((double*)inout_shared, r52);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fma(r6, r6, r54 * r54);
    r46 = fma(r88, r88, r86 * r86);
    WriteSum2<double, double>((double*)inout_shared, r52, r46);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fma(r82, r82, r76 * r76);
    WriteSum1<double, double>((double*)inout_shared, r46);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fma(r6, r88, r54 * r86);
    r6 = fma(r6, r76, r54 * r82);
    WriteSum2<double, double>((double*)inout_shared, r46, r6);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r82 = fma(r86, r82, r88 * r76);
    WriteSum1<double, double>((double*)inout_shared, r82);
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