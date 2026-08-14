#include "kernel_simple_radial_split_fixed_focal_and_extra_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndExtraResJacFirstKernel(
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
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
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

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89;
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
    r45 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r46);
    r47 = r18 * r21;
    r47 = fma(r30, r47, r32);
    r46 = fma(r6, r47, r46);
    r32 = r15 * r11;
    r32 = fma(r18, r32, r36);
    r42 = r19 + r42;
    r36 = r14 * r14;
    r36 = r36 * r18;
    r42 = r42 + r36;
    r48 = r15 * r10;
    r48 = r48 * r26;
    r49 = r14 * r11;
    r49 = fma(r26, r49, r48);
    r50 = r26 * r16;
    r50 = r50 * r21;
    r51 = fma(r27, r33, r50);
    r52 = r27 * r27;
    r52 = r52 * r18;
    r24 = r52 + r24;
    r46 = fma(r38, r32, r46);
    r46 = fma(r35, r42, r46);
    r46 = fma(r39, r49, r46);
    r46 = fma(r7, r51, r46);
    r46 = fma(r31, r24, r46);
    r53 = copysign(1.0, r46);
    r53 = fma(r45, r53, r46);
    r45 = 1.0 / r53;
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            0 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r46,
                                            r54);
    r55 = r53 * r53;
    r56 = 1.0 / r55;
    r57 = r0 * r56;
    r28 = fma(r16, r33, r28);
    r5 = fma(r6, r28, r5);
    r58 = r10 * r11;
    r58 = fma(r26, r58, r41);
    r44 = r19 + r44;
    r44 = r44 + r36;
    r36 = r14 * r11;
    r36 = fma(r18, r36, r48);
    r48 = r27 * r18;
    r48 = fma(r30, r48, r50);
    r17 = r19 + r17;
    r17 = r17 + r52;
    r5 = fma(r38, r58, r5);
    r5 = fma(r39, r44, r5);
    r5 = fma(r35, r36, r5);
    r5 = fma(r31, r48, r5);
    r5 = fma(r7, r17, r5);
    r35 = r5 * r5;
    r39 = fma(r56, r35, r0 * r57);
    r39 = fma(r54, r39, r19);
    r39 = r46 * r39;
    r38 = r45 * r39;
    r2 = fma(r0, r38, r2);
    r3 = fma(r3, r4, r1);
    r3 = fma(r5, r38, r3);
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
    r52 = -5.00000000000000000e-01;
    r50 = r13 * r52;
    r41 = 5.00000000000000000e-01;
    r59 = fma(r41, r23, r11 * r50);
    r59 = fma(r52, r20, r59);
    r59 = fma(r52, r22, r59);
    r1 = r1 * r59;
    r60 = r26 * r16;
    r61 = r13 * r14;
    r62 = r12 * r15;
    r62 = fma(r52, r62, r41 * r61);
    r61 = r9 * r10;
    r62 = fma(r41, r61, r62);
    r63 = r11 * r41;
    r62 = fma(r8, r63, r62);
    r60 = fma(r62, r60, r1);
    r61 = r26 * r27;
    r64 = r12 * r11;
    r65 = r9 * r14;
    r65 = fma(r52, r65, r52 * r64);
    r64 = r8 * r15;
    r65 = fma(r52, r64, r65);
    r66 = r13 * r10;
    r65 = fma(r41, r66, r65);
    r61 = r61 * r65;
    r66 = r12 * r14;
    r64 = r8 * r10;
    r64 = fma(r52, r64, r52 * r66);
    r64 = fma(r9, r63, r64);
    r64 = fma(r15, r50, r64);
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
    r76 = r4 * r39;
    r76 = r76 * r57;
    r77 = fma(r62, r33, r26 * r70);
    r77 = r77 + r71;
    r78 = r26 * r21;
    r78 = r78 * r64;
    r79 = r26 * r27;
    r79 = r79 * r62;
    r62 = r78 + r79;
    r80 = r16 * r18;
    r62 = fma(r65, r80, r62);
    r81 = r18 * r30;
    r62 = fma(r59, r81, r62);
    r62 = fma(r7, r62, r31 * r77);
    r77 = r16 * r59;
    r81 = r73 * r77;
    r68 = r68 + r81;
    r62 = fma(r6, r68, r62);
    r68 = fma(r62, r38, r69 * r76);
    r80 = r46 * r54;
    r82 = r26 * r62;
    r83 = r0 * r0;
    r55 = r53 * r55;
    r55 = 1.0 / r55;
    r55 = r18 * r55;
    r83 = r83 * r55;
    r82 = fma(r69, r83, r57 * r82);
    r53 = r69 * r55;
    r82 = fma(r35, r53, r82);
    r84 = r26 * r5;
    r85 = r27 * r18;
    r86 = r18 * r30;
    r86 = r86 * r64;
    r85 = fma(r65, r85, r86);
    r85 = r85 + r60;
    r81 = r75 + r81;
    r81 = fma(r7, r81, r31 * r85);
    r79 = fma(r59, r33, r79);
    r85 = r26 * r16;
    r85 = fma(r65, r85, r78);
    r79 = r79 + r85;
    r81 = fma(r6, r79, r81);
    r84 = r84 * r81;
    r82 = fma(r56, r84, r82);
    r80 = r80 * r82;
    r80 = r80 * r45;
    r68 = fma(r0, r80, r68);
    r81 = fma(r81, r38, r5 * r80);
    r80 = r4 * r5;
    r80 = r80 * r69;
    r80 = r80 * r56;
    r81 = fma(r39, r80, r81);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r68, r81);
    r66 = r1 + r66;
    r1 = r26 * r16;
    r80 = r8 * r11;
    r82 = r12 * r15;
    r82 = fma(r41, r82, r52 * r80);
    r80 = r9 * r10;
    r82 = fma(r52, r80, r82);
    r82 = fma(r14, r50, r82);
    r1 = r1 * r82;
    r80 = r26 * r27;
    r84 = r9 * r14;
    r53 = r8 * r15;
    r53 = fma(r41, r53, r41 * r84);
    r53 = fma(r12, r63, r53);
    r53 = fma(r10, r50, r53);
    r80 = fma(r53, r80, r1);
    r66 = r66 + r80;
    r50 = r21 * r64;
    r50 = r50 * r73;
    r84 = r16 * r73;
    r84 = r84 * r53;
    r79 = r50 + r84;
    r79 = fma(r6, r79, r31 * r66);
    r66 = r18 * r30;
    r66 = fma(r18, r77, r53 * r66);
    r78 = r26 * r27;
    r78 = r78 * r64;
    r75 = r26 * r21;
    r75 = fma(r82, r75, r78);
    r66 = r66 + r75;
    r79 = fma(r7, r66, r79);
    r66 = r46 * r54;
    r60 = r26 * r79;
    r87 = r26 * r5;
    r88 = r27 * r18;
    r88 = fma(r59, r88, r72);
    r72 = r26 * r21;
    r72 = r72 * r53;
    r89 = r18 * r30;
    r88 = fma(r82, r89, r88);
    r88 = r88 + r72;
    r53 = fma(r53, r33, r26 * r77);
    r53 = r53 + r75;
    r53 = fma(r6, r53, r31 * r88);
    r88 = r82 * r74;
    r84 = r84 + r88;
    r53 = fma(r7, r84, r53);
    r87 = r87 * r53;
    r87 = fma(r56, r87, r57 * r60);
    r60 = r18 * r21;
    r60 = fma(r59, r60, r86);
    r60 = r60 + r80;
    r72 = fma(r82, r33, r72);
    r72 = r72 + r71;
    r72 = fma(r7, r72, r6 * r60);
    r88 = r50 + r88;
    r72 = fma(r31, r88, r72);
    r88 = r72 * r55;
    r87 = fma(r35, r88, r87);
    r87 = fma(r72, r83, r87);
    r66 = r66 * r0;
    r66 = r66 * r87;
    r66 = fma(r45, r66, r79 * r38);
    r66 = fma(r72, r76, r66);
    r88 = r46 * r54;
    r88 = r88 * r5;
    r88 = r88 * r87;
    r88 = fma(r45, r88, r53 * r38);
    r53 = r4 * r5;
    r53 = r53 * r72;
    r53 = r53 * r56;
    r88 = fma(r39, r53, r88);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r66, r88);
    r53 = r46 * r54;
    r87 = r26 * r5;
    r50 = r26 * r21;
    r23 = fma(r52, r23, r13 * r63);
    r23 = fma(r41, r20, r23);
    r23 = fma(r41, r22, r23);
    r50 = r50 * r23;
    r1 = r1 + r50;
    r1 = r1 + r67;
    r67 = r27 * r18;
    r22 = r18 * r30;
    r22 = fma(r23, r22, r82 * r67);
    r22 = r22 + r85;
    r22 = fma(r31, r22, r6 * r1);
    r64 = r16 * r64;
    r64 = r64 * r73;
    r74 = r23 * r74;
    r1 = r64 + r74;
    r22 = fma(r7, r1, r22);
    r87 = r87 * r22;
    r70 = r73 * r70;
    r74 = r74 + r70;
    r73 = r26 * r16;
    r73 = r73 * r23;
    r78 = r78 + r73;
    r1 = r18 * r21;
    r78 = fma(r82, r1, r78);
    r67 = r18 * r30;
    r78 = fma(r65, r67, r78);
    r78 = fma(r6, r78, r31 * r74);
    r74 = r26 * r27;
    r23 = fma(r23, r33, r82 * r74);
    r23 = r23 + r85;
    r78 = fma(r7, r23, r78);
    r87 = fma(r78, r83, r56 * r87);
    r86 = r61 + r86;
    r61 = r16 * r18;
    r86 = fma(r82, r61, r86);
    r86 = r86 + r50;
    r70 = r64 + r70;
    r70 = fma(r6, r70, r7 * r86);
    r33 = fma(r65, r33, r73);
    r33 = r33 + r75;
    r70 = fma(r31, r33, r70);
    r33 = r26 * r70;
    r87 = fma(r57, r33, r87);
    r31 = r78 * r55;
    r87 = fma(r35, r31, r87);
    r53 = r53 * r0;
    r53 = r53 * r87;
    r53 = fma(r70, r38, r45 * r53);
    r53 = fma(r78, r76, r53);
    r31 = r46 * r54;
    r31 = r31 * r5;
    r31 = r31 * r87;
    r87 = r4 * r5;
    r87 = r87 * r78;
    r87 = r87 * r56;
    r87 = fma(r39, r87, r45 * r31);
    r87 = fma(r22, r38, r87);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r53, r87);
    r22 = r46 * r54;
    r31 = r26 * r58;
    r31 = r31 * r5;
    r33 = r32 * r55;
    r33 = fma(r35, r33, r56 * r31);
    r31 = r26 * r43;
    r33 = fma(r57, r31, r33);
    r33 = fma(r32, r83, r33);
    r22 = r22 * r0;
    r22 = r22 * r33;
    r22 = fma(r45, r22, r43 * r38);
    r22 = fma(r32, r76, r22);
    r31 = r4 * r32;
    r31 = r31 * r5;
    r31 = r31 * r56;
    r31 = fma(r58, r38, r39 * r31);
    r75 = r46 * r54;
    r75 = r75 * r5;
    r75 = r75 * r33;
    r31 = fma(r45, r75, r31);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r22, r31);
    r75 = fma(r40, r38, r49 * r76);
    r33 = r46 * r54;
    r65 = r49 * r55;
    r73 = r26 * r44;
    r73 = r73 * r5;
    r73 = fma(r56, r73, r35 * r65);
    r65 = r26 * r40;
    r73 = fma(r57, r65, r73);
    r73 = fma(r49, r83, r73);
    r33 = r33 * r0;
    r33 = r33 * r73;
    r75 = fma(r45, r33, r75);
    r33 = r4 * r49;
    r33 = r33 * r5;
    r33 = r33 * r56;
    r33 = fma(r44, r38, r39 * r33);
    r65 = r46 * r54;
    r65 = r65 * r5;
    r65 = r65 * r73;
    r33 = fma(r45, r65, r33);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r75, r33);
    r65 = r46 * r54;
    r73 = r26 * r36;
    r73 = r73 * r5;
    r6 = r42 * r55;
    r6 = fma(r35, r6, r56 * r73);
    r73 = r26 * r37;
    r6 = fma(r57, r73, r6);
    r6 = fma(r42, r83, r6);
    r65 = r65 * r0;
    r65 = r65 * r6;
    r65 = fma(r37, r38, r45 * r65);
    r65 = fma(r42, r76, r65);
    r73 = r46 * r54;
    r73 = r73 * r5;
    r73 = r73 * r6;
    r73 = fma(r45, r73, r36 * r38);
    r6 = r4 * r42;
    r6 = r6 * r5;
    r6 = r6 * r56;
    r73 = fma(r39, r6, r73);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r65, r73);
    r6 = r4 * r3;
    r2 = r4 * r2;
    r6 = fma(r68, r2, r81 * r6);
    r86 = r4 * r3;
    r86 = fma(r66, r2, r88 * r86);
    WriteSum2<double, double>((double*)inout_shared, r6, r86);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r86 = r4 * r3;
    r86 = fma(r53, r2, r87 * r86);
    r6 = r4 * r3;
    r6 = fma(r22, r2, r31 * r6);
    WriteSum2<double, double>((double*)inout_shared, r86, r6);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = r4 * r3;
    r6 = fma(r75, r2, r33 * r6);
    r86 = r4 * r3;
    r86 = fma(r65, r2, r73 * r86);
    WriteSum2<double, double>((double*)inout_shared, r6, r86);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r86 = fma(r68, r68, r81 * r81);
    r6 = fma(r66, r66, r88 * r88);
    WriteSum2<double, double>((double*)inout_shared, r86, r6);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r53, r53, r87 * r87);
    r86 = fma(r22, r22, r31 * r31);
    WriteSum2<double, double>((double*)inout_shared, r6, r86);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r86 = fma(r33, r33, r75 * r75);
    r6 = fma(r65, r65, r73 * r73);
    WriteSum2<double, double>((double*)inout_shared, r86, r6);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r68, r66, r81 * r88);
    r86 = fma(r68, r53, r81 * r87);
    WriteSum2<double, double>((double*)inout_shared, r6, r86);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r86 = fma(r68, r22, r81 * r31);
    r6 = fma(r81, r33, r68 * r75);
    WriteSum2<double, double>((double*)inout_shared, r86, r6);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = fma(r68, r65, r81 * r73);
    r81 = fma(r88, r87, r66 * r53);
    WriteSum2<double, double>((double*)inout_shared, r68, r81);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r81 = fma(r88, r31, r66 * r22);
    r68 = fma(r88, r33, r66 * r75);
    WriteSum2<double, double>((double*)inout_shared, r81, r68);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r88 = fma(r88, r73, r66 * r65);
    r66 = fma(r53, r22, r87 * r31);
    WriteSum2<double, double>((double*)inout_shared, r88, r66);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = fma(r53, r75, r87 * r33);
    r87 = fma(r87, r73, r53 * r65);
    WriteSum2<double, double>((double*)inout_shared, r66, r87);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r87 = fma(r22, r75, r31 * r33);
    r31 = fma(r31, r73, r22 * r65);
    WriteSum2<double, double>((double*)inout_shared, r87, r31);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r33, r73, r75 * r65);
    WriteSum1<double, double>((double*)inout_shared, r73);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = r4 * r3;
    WriteSum2<double, double>((double*)inout_shared, r2, r73);
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
    r19 = r46 * r54;
    r73 = r26 * r25;
    r33 = r47 * r55;
    r33 = fma(r35, r33, r57 * r73);
    r73 = r26 * r28;
    r73 = r73 * r5;
    r33 = fma(r56, r73, r33);
    r33 = fma(r47, r83, r33);
    r19 = r19 * r0;
    r19 = r19 * r33;
    r19 = fma(r47, r76, r45 * r19);
    r19 = fma(r25, r38, r19);
    r73 = r4 * r47;
    r73 = r73 * r5;
    r73 = r73 * r56;
    r65 = r46 * r54;
    r65 = r65 * r5;
    r65 = r65 * r33;
    r65 = fma(r45, r65, r39 * r73);
    r65 = fma(r28, r38, r65);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r19,
                                             r65);
    r73 = r46 * r54;
    r33 = r26 * r29;
    r33 = fma(r57, r33, r51 * r83);
    r75 = r26 * r17;
    r75 = r75 * r5;
    r33 = fma(r56, r75, r33);
    r31 = r51 * r55;
    r33 = fma(r35, r31, r33);
    r73 = r73 * r0;
    r73 = r73 * r33;
    r73 = fma(r29, r38, r45 * r73);
    r73 = fma(r51, r76, r73);
    r31 = r46 * r54;
    r31 = r31 * r5;
    r31 = r31 * r33;
    r33 = r4 * r51;
    r33 = r33 * r5;
    r33 = r33 * r56;
    r33 = fma(r39, r33, r45 * r31);
    r33 = fma(r17, r38, r33);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r73,
                                             r33);
    r31 = r46 * r54;
    r75 = r26 * r34;
    r87 = r24 * r55;
    r87 = fma(r35, r87, r57 * r75);
    r75 = r26 * r48;
    r75 = r75 * r5;
    r87 = fma(r56, r75, r87);
    r87 = fma(r24, r83, r87);
    r31 = r31 * r0;
    r31 = r31 * r87;
    r31 = fma(r34, r38, r45 * r31);
    r31 = fma(r24, r76, r31);
    r76 = r46 * r54;
    r76 = r76 * r5;
    r76 = r76 * r87;
    r87 = r4 * r24;
    r87 = r87 * r5;
    r87 = r87 * r56;
    r87 = fma(r39, r87, r45 * r76);
    r87 = fma(r48, r38, r87);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r31,
                                             r87);
    r38 = r4 * r3;
    r38 = fma(r19, r2, r65 * r38);
    r76 = r4 * r3;
    r76 = fma(r73, r2, r33 * r76);
    WriteSum2<double, double>((double*)inout_shared, r38, r76);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = r4 * r3;
    r2 = fma(r31, r2, r87 * r76);
    WriteSum1<double, double>((double*)inout_shared, r2);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = fma(r65, r65, r19 * r19);
    r76 = fma(r33, r33, r73 * r73);
    WriteSum2<double, double>((double*)inout_shared, r2, r76);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = fma(r31, r31, r87 * r87);
    WriteSum1<double, double>((double*)inout_shared, r76);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = fma(r65, r33, r19 * r73);
    r65 = fma(r65, r87, r19 * r31);
    WriteSum2<double, double>((double*)inout_shared, r76, r65);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fma(r73, r31, r33 * r87);
    WriteSum1<double, double>((double*)inout_shared, r31);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedFocalAndExtraResJacFirst(
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
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
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
  SimpleRadialSplitFixedFocalAndExtraResJacFirstKernel<<<n_blocks, 1024>>>(
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
      focal_and_extra,
      focal_and_extra_num_alloc,
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