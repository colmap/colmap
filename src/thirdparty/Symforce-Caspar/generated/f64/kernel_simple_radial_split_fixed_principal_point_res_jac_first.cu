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
        double* const out_rTr,
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

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93;

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
    r1 = fma(r3, r3, r2 * r2);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r1);
  if (global_thread_idx < problem_size) {
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
    r84 = r0 * r76;
    r46 = r46 * r52;
    r68 = fma(r46, r84, r68);
    r84 = r5 * r76;
    r55 = r19 * r83;
    r55 = fma(r52, r55, r46 * r84);
    r84 = r5 * r48;
    r84 = r84 * r78;
    r55 = fma(r69, r84, r55);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r68, r55);
    r66 = r1 + r66;
    r1 = r26 * r16;
    r81 = r8 * r11;
    r80 = r12 * r15;
    r80 = fma(r58, r80, r38 * r81);
    r81 = r9 * r10;
    r80 = fma(r38, r81, r80);
    r80 = fma(r14, r41, r80);
    r1 = r1 * r80;
    r81 = r26 * r27;
    r75 = r9 * r14;
    r60 = r8 * r15;
    r60 = fma(r58, r60, r58 * r75);
    r60 = fma(r12, r63, r60);
    r60 = fma(r10, r41, r60);
    r81 = fma(r60, r81, r1);
    r66 = r66 + r81;
    r41 = r21 * r64;
    r41 = r41 * r73;
    r75 = r16 * r73;
    r75 = r75 * r60;
    r87 = r41 + r75;
    r87 = fma(r6, r87, r31 * r66);
    r66 = r18 * r30;
    r66 = fma(r18, r79, r60 * r66);
    r88 = r26 * r27;
    r88 = r88 * r64;
    r89 = r26 * r21;
    r89 = fma(r80, r89, r88);
    r66 = r66 + r89;
    r87 = fma(r7, r66, r87);
    r66 = r19 * r87;
    r90 = r26 * r87;
    r91 = r26 * r5;
    r92 = r27 * r18;
    r92 = fma(r59, r92, r72);
    r72 = r26 * r21;
    r72 = r72 * r60;
    r93 = r18 * r30;
    r92 = fma(r80, r93, r92);
    r92 = r92 + r72;
    r60 = fma(r60, r33, r26 * r79);
    r60 = r60 + r89;
    r60 = fma(r6, r60, r31 * r92);
    r92 = r80 * r74;
    r75 = r75 + r92;
    r60 = fma(r7, r75, r60);
    r91 = r91 * r60;
    r91 = fma(r48, r91, r56 * r90);
    r90 = r18 * r21;
    r90 = fma(r59, r90, r86);
    r90 = r90 + r81;
    r72 = fma(r80, r33, r72);
    r72 = r72 + r71;
    r72 = fma(r7, r72, r6 * r90);
    r92 = r41 + r92;
    r72 = fma(r31, r92, r72);
    r92 = r72 * r47;
    r91 = fma(r35, r92, r91);
    r91 = fma(r72, r82, r91);
    r92 = r0 * r91;
    r92 = fma(r46, r92, r52 * r66);
    r66 = r72 * r56;
    r92 = fma(r78, r66, r92);
    r66 = r19 * r60;
    r41 = r5 * r91;
    r41 = fma(r46, r41, r52 * r66);
    r41 = fma(r72, r84, r41);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r92, r41);
    r66 = r26 * r5;
    r90 = r26 * r21;
    r23 = fma(r38, r23, r13 * r63);
    r23 = fma(r58, r20, r23);
    r23 = fma(r58, r22, r23);
    r90 = r90 * r23;
    r1 = r1 + r90;
    r1 = r1 + r67;
    r67 = r27 * r18;
    r22 = r18 * r30;
    r22 = fma(r23, r22, r80 * r67);
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
    r88 = r88 + r73;
    r1 = r18 * r21;
    r88 = fma(r80, r1, r88);
    r67 = r18 * r30;
    r88 = fma(r65, r67, r88);
    r88 = fma(r6, r88, r31 * r74);
    r74 = r26 * r27;
    r23 = fma(r23, r33, r80 * r74);
    r23 = r23 + r85;
    r88 = fma(r7, r23, r88);
    r66 = fma(r88, r82, r48 * r66);
    r86 = r61 + r86;
    r61 = r16 * r18;
    r86 = fma(r80, r61, r86);
    r86 = r86 + r90;
    r70 = r64 + r70;
    r70 = fma(r6, r70, r7 * r86);
    r33 = fma(r65, r33, r73);
    r33 = r33 + r89;
    r70 = fma(r31, r33, r70);
    r33 = r26 * r70;
    r66 = fma(r56, r33, r66);
    r31 = r88 * r47;
    r66 = fma(r35, r31, r66);
    r31 = r0 * r66;
    r33 = r19 * r70;
    r33 = fma(r52, r33, r46 * r31);
    r31 = r88 * r56;
    r33 = fma(r78, r31, r33);
    r31 = r5 * r66;
    r31 = fma(r88, r84, r46 * r31);
    r89 = r19 * r22;
    r31 = fma(r52, r89, r31);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r33, r31);
    r89 = r43 * r19;
    r65 = r26 * r57;
    r65 = r65 * r5;
    r73 = r32 * r47;
    r73 = fma(r35, r73, r48 * r65);
    r65 = r26 * r43;
    r73 = fma(r56, r65, r73);
    r73 = fma(r32, r82, r73);
    r73 = r73 * r46;
    r89 = fma(r0, r73, r52 * r89);
    r65 = r32 * r56;
    r89 = fma(r78, r65, r89);
    r65 = r57 * r19;
    r65 = fma(r52, r65, r32 * r84);
    r65 = fma(r5, r73, r65);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r89, r65);
    r73 = r51 * r56;
    r6 = r40 * r19;
    r6 = fma(r52, r6, r78 * r73);
    r73 = r51 * r47;
    r86 = r26 * r44;
    r86 = r86 * r5;
    r86 = fma(r48, r86, r35 * r73);
    r73 = r26 * r40;
    r86 = fma(r56, r73, r86);
    r86 = fma(r51, r82, r86);
    r73 = r0 * r86;
    r6 = fma(r46, r73, r6);
    r73 = r44 * r19;
    r73 = fma(r52, r73, r51 * r84);
    r7 = r5 * r86;
    r73 = fma(r46, r7, r73);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r6, r73);
    r7 = r26 * r36;
    r7 = r7 * r5;
    r64 = r42 * r47;
    r64 = fma(r35, r64, r48 * r7);
    r7 = r26 * r37;
    r64 = fma(r56, r7, r64);
    r64 = fma(r42, r82, r64);
    r7 = r0 * r64;
    r61 = r37 * r19;
    r61 = fma(r52, r61, r46 * r7);
    r7 = r42 * r56;
    r61 = fma(r78, r7, r61);
    r7 = r36 * r19;
    r90 = r5 * r64;
    r90 = fma(r46, r90, r52 * r7);
    r90 = fma(r42, r84, r90);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r61, r90);
    r7 = r4 * r2;
    r80 = r4 * r3;
    r80 = fma(r55, r80, r68 * r7);
    r7 = r4 * r2;
    r23 = r4 * r3;
    r23 = fma(r41, r23, r92 * r7);
    WriteSum2<double, double>((double*)inout_shared, r80, r23);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = r4 * r3;
    r80 = r4 * r2;
    r80 = fma(r33, r80, r31 * r23);
    r23 = r4 * r3;
    r7 = r4 * r2;
    r7 = fma(r89, r7, r65 * r23);
    WriteSum2<double, double>((double*)inout_shared, r80, r7);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r4 * r2;
    r80 = r4 * r3;
    r80 = fma(r73, r80, r6 * r7);
    r7 = r4 * r2;
    r23 = r4 * r3;
    r23 = fma(r90, r23, r61 * r7);
    WriteSum2<double, double>((double*)inout_shared, r80, r23);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r68, r68, r55 * r55);
    r80 = fma(r92, r92, r41 * r41);
    WriteSum2<double, double>((double*)inout_shared, r23, r80);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = fma(r33, r33, r31 * r31);
    r23 = fma(r89, r89, r65 * r65);
    WriteSum2<double, double>((double*)inout_shared, r80, r23);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r73, r73, r6 * r6);
    r80 = fma(r61, r61, r90 * r90);
    WriteSum2<double, double>((double*)inout_shared, r23, r80);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r80 = fma(r68, r92, r55 * r41);
    r23 = fma(r68, r33, r55 * r31);
    WriteSum2<double, double>((double*)inout_shared, r80, r23);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r68, r89, r55 * r65);
    r80 = fma(r55, r73, r68 * r6);
    WriteSum2<double, double>((double*)inout_shared, r23, r80);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = fma(r68, r61, r55 * r90);
    r55 = fma(r41, r31, r92 * r33);
    WriteSum2<double, double>((double*)inout_shared, r68, r55);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fma(r41, r65, r92 * r89);
    r68 = fma(r41, r73, r92 * r6);
    WriteSum2<double, double>((double*)inout_shared, r55, r68);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r41, r90, r92 * r61);
    r92 = fma(r33, r89, r31 * r65);
    WriteSum2<double, double>((double*)inout_shared, r41, r92);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r92 = fma(r33, r6, r31 * r73);
    r31 = fma(r31, r90, r33 * r61);
    WriteSum2<double, double>((double*)inout_shared, r92, r31);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fma(r89, r6, r65 * r73);
    r65 = fma(r65, r90, r89 * r61);
    WriteSum2<double, double>((double*)inout_shared, r31, r65);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r90 = fma(r73, r90, r6 * r61);
    WriteSum1<double, double>((double*)inout_shared, r90);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r90 = r0 * r19;
    r90 = r90 * r54;
    r73 = r5 * r19;
    r73 = r73 * r54;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        0 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r90,
        r73);
    r73 = r0 * r39;
    r73 = r73 * r52;
    r90 = r5 * r39;
    r90 = r90 * r52;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac,
        2 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx,
        r73,
        r90);
    r90 = r0 * r2;
    r90 = r90 * r54;
    r73 = r5 * r3;
    r73 = r73 * r54;
    r73 = fma(r77, r73, r77 * r90);
    r90 = r4 * r5;
    r90 = r90 * r39;
    r90 = r90 * r3;
    r77 = r4 * r0;
    r77 = r77 * r39;
    r77 = r77 * r2;
    r77 = fma(r52, r77, r52 * r90);
    WriteSum2<double, double>((double*)inout_shared, r73, r77);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            0 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = r0 * r19;
    r77 = r77 * r19;
    r73 = r19 * r19;
    r73 = r73 * r48;
    r73 = fma(r35, r73, r56 * r77);
    r77 = r39 * r39;
    r90 = r45 * r77;
    r54 = r45 * r48;
    r54 = r54 * r35;
    r61 = r45 * r45;
    r61 = r61 * r0;
    r61 = r61 * r56;
    r61 = fma(r77, r61, r54 * r90);
    WriteSum2<double, double>((double*)inout_shared, r73, r61);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            0 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = r45 * r0;
    r61 = r61 * r39;
    r61 = r61 * r19;
    r73 = r39 * r19;
    r73 = fma(r54, r73, r56 * r61);
    WriteSum1<double, double>((double*)inout_shared, r73);
  };
  FlushSumShared<1, double>(out_focal_and_extra_precond_tril,
                            0 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = r26 * r25;
    r61 = r49 * r47;
    r61 = fma(r35, r61, r56 * r73);
    r73 = r26 * r28;
    r73 = r73 * r5;
    r61 = fma(r48, r73, r61);
    r61 = fma(r49, r82, r61);
    r73 = r0 * r61;
    r54 = r49 * r56;
    r54 = fma(r78, r54, r46 * r73);
    r73 = r25 * r19;
    r54 = fma(r52, r73, r54);
    r73 = r5 * r61;
    r73 = fma(r46, r73, r49 * r84);
    r90 = r28 * r19;
    r73 = fma(r52, r90, r73);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r54,
                                             r73);
    r90 = r26 * r29;
    r90 = fma(r56, r90, r53 * r82);
    r6 = r26 * r17;
    r6 = r6 * r5;
    r90 = fma(r48, r6, r90);
    r65 = r53 * r47;
    r90 = fma(r35, r65, r90);
    r65 = r0 * r90;
    r6 = r29 * r19;
    r6 = fma(r52, r6, r46 * r65);
    r65 = r53 * r56;
    r6 = fma(r78, r65, r6);
    r65 = r5 * r90;
    r65 = fma(r53, r84, r46 * r65);
    r31 = r17 * r19;
    r65 = fma(r52, r31, r65);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 2 * out_point_jac_num_alloc, global_thread_idx, r6, r65);
    r31 = r26 * r34;
    r89 = r24 * r47;
    r89 = fma(r35, r89, r56 * r31);
    r31 = r26 * r50;
    r31 = r31 * r5;
    r89 = fma(r48, r31, r89);
    r89 = fma(r24, r82, r89);
    r31 = r0 * r89;
    r82 = r34 * r19;
    r82 = fma(r52, r82, r46 * r31);
    r31 = r24 * r56;
    r82 = fma(r78, r31, r82);
    r31 = r5 * r89;
    r84 = fma(r24, r84, r46 * r31);
    r31 = r50 * r19;
    r84 = fma(r52, r31, r84);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r82,
                                             r84);
    r31 = r4 * r2;
    r52 = r4 * r3;
    r52 = fma(r73, r52, r54 * r31);
    r31 = r4 * r3;
    r46 = r4 * r2;
    r46 = fma(r6, r46, r65 * r31);
    WriteSum2<double, double>((double*)inout_shared, r52, r46);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r4 * r2;
    r52 = r4 * r3;
    r52 = fma(r84, r52, r82 * r46);
    WriteSum1<double, double>((double*)inout_shared, r52);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fma(r73, r73, r54 * r54);
    r46 = fma(r65, r65, r6 * r6);
    WriteSum2<double, double>((double*)inout_shared, r52, r46);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fma(r82, r82, r84 * r84);
    WriteSum1<double, double>((double*)inout_shared, r46);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fma(r73, r65, r54 * r6);
    r73 = fma(r73, r84, r54 * r82);
    WriteSum2<double, double>((double*)inout_shared, r46, r73);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r82 = fma(r6, r82, r65 * r84);
    WriteSum1<double, double>((double*)inout_shared, r82);
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
    double* const out_rTr,
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
  SimpleRadialSplitFixedPrincipalPointResJacFirstKernel<<<n_blocks, 1024>>>(
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
      out_rTr,
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