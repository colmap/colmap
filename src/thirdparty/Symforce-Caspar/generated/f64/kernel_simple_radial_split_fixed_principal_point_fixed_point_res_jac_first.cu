#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_split_fixed_principal_point_fixed_point_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPrincipalPointFixedPointResJacFirstKernel(
        double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
        double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        double *focal_and_extra, unsigned int focal_and_extra_num_alloc,
        SharedIndex *focal_and_extra_indices, double *pixel,
        unsigned int pixel_num_alloc, double *principal_point,
        unsigned int principal_point_num_alloc, double *point,
        unsigned int point_num_alloc, double *out_res,
        unsigned int out_res_num_alloc, double *const out_rTr,
        double *out_pose_jac, unsigned int out_pose_jac_num_alloc,
        double *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
        double *const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double *const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        double *out_focal_and_extra_jac,
        unsigned int out_focal_and_extra_jac_num_alloc,
        double *const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        double *const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        double *const out_focal_and_extra_precond_tril,
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

  __shared__ double out_rTr_local[1];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0,
         r9 = 0, r10 = 0, r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0, r16 = 0,
         r17 = 0, r18 = 0, r19 = 0, r20 = 0, r21 = 0, r22 = 0, r23 = 0, r24 = 0,
         r25 = 0, r26 = 0, r27 = 0, r28 = 0, r29 = 0, r30 = 0, r31 = 0, r32 = 0,
         r33 = 0, r34 = 0, r35 = 0, r36 = 0, r37 = 0, r38 = 0, r39 = 0, r40 = 0,
         r41 = 0, r42 = 0, r43 = 0, r44 = 0, r45 = 0, r46 = 0, r47 = 0, r48 = 0,
         r49 = 0, r50 = 0, r51 = 0, r52 = 0, r53 = 0, r54 = 0, r55 = 0, r56 = 0,
         r57 = 0, r58 = 0, r59 = 0, r60 = 0, r61 = 0, r62 = 0, r63 = 0, r64 = 0,
         r65 = 0, r66 = 0, r67 = 0, r68 = 0, r69 = 0, r70 = 0, r71 = 0, r72 = 0,
         r73 = 0, r74 = 0, r75 = 0, r76 = 0, r77 = 0, r78 = 0, r79 = 0, r80 = 0,
         r81 = 0, r82 = 0, r83 = 0, r84 = 0, r85 = 0, r86 = 0, r87 = 0, r88 = 0,
         r89 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(principal_point,
                                            0 * principal_point_num_alloc,
                                            global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(pixel, 0 * pixel_num_alloc,
                                            global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r5 = fma(r2, r4, r0);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r6, r7);
    ReadIdx2<1024, double, double, double2>(point, 0 * point_num_alloc,
                                            global_thread_idx, r8, r9);
    r10 = -2.00000000000000000e+00;
  };
  LoadShared<2, double, double>(pose, 2 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r11, r12);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r13, r14);
  };
  LoadShared<2, double, double>(pose, 0 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r15, r16);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r17, r18);
    r19 = fma(r16, r17, r11 * r14);
    r20 = r15 * r18;
    r19 = fma(r4, r20, r19);
    r19 = fma(r12, r13, r19);
    r20 = r19 * r19;
    r20 = r10 * r20;
    r21 = 1.00000000000000000e+00;
    r22 = r11 * r17;
    r22 = fma(r4, r22, r16 * r14);
    r22 = fma(r12, r18, r22);
    r22 = fma(r15, r13, r22);
    r23 = r22 * r22;
    r23 = fma(r10, r23, r21);
    r24 = r20 + r23;
    r24 = fma(r8, r24, r6);
    r25 = 2.00000000000000000e+00;
    r26 = fma(r12, r17, r15 * r14);
    r27 = r16 * r13;
    r26 = fma(r4, r27, r26);
    r26 = fma(r11, r18, r26);
    r27 = r25 * r26;
    r28 = r22 * r27;
    r29 = fma(r16, r18, r15 * r17);
    r29 = fma(r11, r13, r29);
    r29 = fma(r4, r29, r12 * r14);
    r30 = r10 * r29;
    r31 = fma(r19, r30, r28);
    ReadIdx1<1024, double, double, double>(point, 2 * point_num_alloc,
                                           global_thread_idx, r32);
    r33 = r25 * r22;
    r34 = r19 * r27;
    r33 = fma(r29, r33, r34);
  };
  LoadShared<1, double, double>(pose, 6 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r35);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r36 = r17 * r13;
    r36 = r36 * r25;
    r37 = r18 * r14;
    r38 = fma(r25, r37, r36);
  };
  LoadShared<2, double, double>(pose, 4 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r39, r40);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r41 = r13 * r14;
    r42 = r17 * r18;
    r42 = r42 * r25;
    r41 = fma(r10, r41, r42);
    r43 = r18 * r18;
    r43 = r43 * r10;
    r44 = r21 + r43;
    r45 = r13 * r13;
    r45 = r10 * r45;
    r44 = r44 + r45;
    r24 = fma(r9, r31, r24);
    r24 = fma(r32, r33, r24);
    r24 = fma(r35, r38, r24);
    r24 = fma(r40, r41, r24);
    r24 = fma(r39, r44, r24);
  };
  LoadShared<2, double, double>(focal_and_extra, 0 * focal_and_extra_num_alloc,
                                focal_and_extra_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        focal_and_extra_indices_loc[threadIdx.x].target, r44,
                        r41);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r38 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r33);
    r34 = fma(r22, r30, r34);
    r34 = fma(r8, r34, r33);
    r37 = fma(r10, r37, r36);
    r43 = r21 + r43;
    r36 = r17 * r17;
    r36 = r10 * r36;
    r43 = r43 + r36;
    r31 = r18 * r13;
    r31 = r31 * r25;
    r46 = r17 * r14;
    r46 = fma(r25, r46, r31);
    r47 = r25 * r19;
    r47 = r47 * r22;
    r27 = fma(r29, r27, r47);
    r48 = r26 * r26;
    r48 = r48 * r10;
    r23 = r48 + r23;
    r34 = fma(r39, r37, r34);
    r34 = fma(r35, r43, r34);
    r34 = fma(r40, r46, r34);
    r34 = fma(r9, r27, r34);
    r34 = fma(r32, r23, r34);
    r23 = copysign(1.0, r34);
    r23 = fma(r38, r23, r34);
    r38 = r23 * r23;
    r38 = 1.0 / r38;
    r34 = r24 * r24;
    r27 = r25 * r19;
    r27 = fma(r29, r27, r28);
    r27 = fma(r8, r27, r7);
    r28 = r13 * r14;
    r28 = fma(r25, r28, r42);
    r45 = r21 + r45;
    r45 = r45 + r36;
    r36 = r17 * r14;
    r36 = fma(r10, r36, r31);
    r30 = fma(r26, r30, r47);
    r20 = r21 + r20;
    r20 = r20 + r48;
    r27 = fma(r39, r28, r27);
    r27 = fma(r40, r45, r27);
    r27 = fma(r35, r36, r27);
    r27 = fma(r32, r30, r27);
    r27 = fma(r9, r20, r27);
    r20 = r27 * r27;
    r20 = fma(r38, r20, r38 * r34);
    r20 = fma(r41, r20, r21);
    r20 = r44 * r20;
    r23 = 1.0 / r23;
    r20 = r20 * r23;
    r5 = fma(r24, r20, r5);
    r4 = fma(r3, r4, r1);
    r4 = fma(r27, r20, r4);
    WriteIdx2<1024, double, double, double2>(out_res, 0 * out_res_num_alloc,
                                             global_thread_idx, r5, r4);
    r4 = fma(r4, r4, r5 * r5);
  };
  SumStore<double>(out_rTr_local, (double *)inout_shared, 0,
                   global_thread_idx < problem_size, r4);
  if (global_thread_idx < problem_size) {
    r4 = fma(r16, r17, r11 * r14);
    r5 = r15 * r18;
    r20 = -1.00000000000000000e+00;
    r4 = fma(r20, r5, r4);
    r4 = fma(r12, r13, r4);
    r5 = r4 * r4;
    r27 = -2.00000000000000000e+00;
    r5 = r5 * r27;
    r24 = 1.00000000000000000e+00;
    r23 = fma(r12, r18, r16 * r14);
    r21 = r15 * r13;
    r38 = r11 * r17;
    r23 = r23 + r21;
    r23 = fma(r20, r38, r23);
    r34 = r27 * r23;
    r34 = fma(r23, r34, r24);
    r30 = r5 + r34;
    r30 = fma(r8, r30, r6);
    r36 = 2.00000000000000000e+00;
    r45 = fma(r12, r17, r15 * r14);
    r28 = r16 * r13;
    r45 = fma(r20, r28, r45);
    r45 = fma(r11, r18, r45);
    r28 = r36 * r45;
    r28 = r28 * r23;
    r48 = r4 * r27;
    r26 = fma(r16, r18, r15 * r17);
    r26 = fma(r11, r13, r26);
    r26 = fma(r20, r26, r12 * r14);
    r48 = fma(r26, r48, r28);
    r47 = r36 * r4;
    r47 = r47 * r45;
    r31 = r36 * r26;
    r10 = fma(r23, r31, r47);
    r42 = r17 * r13;
    r42 = r42 * r36;
    r29 = r18 * r14;
    r46 = fma(r36, r29, r42);
    r43 = r13 * r14;
    r37 = r17 * r18;
    r37 = r37 * r36;
    r43 = fma(r27, r43, r37);
    r49 = r18 * r18;
    r49 = r49 * r27;
    r50 = r24 + r49;
    r51 = r13 * r13;
    r51 = r51 * r27;
    r50 = r50 + r51;
    r30 = fma(r9, r48, r30);
    r30 = fma(r32, r10, r30);
    r30 = fma(r35, r46, r30);
    r30 = fma(r40, r43, r30);
    r30 = fma(r39, r50, r30);
    r10 = r15 * r14;
    r48 = -5.00000000000000000e-01;
    r52 = r12 * r17;
    r52 = fma(r48, r52, r48 * r10);
    r10 = r11 * r18;
    r52 = fma(r48, r10, r52);
    r53 = r16 * r13;
    r54 = 5.00000000000000000e-01;
    r52 = fma(r54, r53, r52);
    r53 = r23 * r52;
    r10 = r11 * r14;
    r55 = r16 * r17;
    r55 = fma(r54, r55, r54 * r10);
    r10 = r15 * r18;
    r55 = fma(r48, r10, r55);
    r56 = r12 * r54;
    r55 = fma(r13, r56, r55);
    r10 = fma(r55, r31, r36 * r53);
    r57 = r36 * r45;
    r58 = r12 * r18;
    r59 = r16 * r48;
    r58 = fma(r14, r59, r48 * r58);
    r58 = fma(r54, r38, r58);
    r58 = fma(r48, r21, r58);
    r60 = r36 * r4;
    r61 = r15 * r17;
    r62 = r11 * r13;
    r62 = fma(r48, r62, r48 * r61);
    r62 = fma(r14, r56, r62);
    r62 = fma(r18, r59, r62);
    r60 = r60 * r62;
    r57 = fma(r58, r57, r60);
    r10 = r10 + r57;
    r61 = r36 * r23;
    r61 = r61 * r62;
    r63 = r36 * r45;
    r63 = r63 * r55;
    r64 = r61 + r63;
    r65 = r4 * r27;
    r64 = fma(r52, r65, r64);
    r66 = r27 * r26;
    r64 = fma(r58, r66, r64);
    r64 = fma(r9, r64, r32 * r10);
    r10 = r23 * r55;
    r66 = -4.00000000000000000e+00;
    r10 = r10 * r66;
    r65 = r4 * r58;
    r67 = r66 * r65;
    r68 = r10 + r67;
    r64 = fma(r8, r68, r64);
    r68 = r36 * r64;
    r69 = 1.00000000000000008e-15;
    r70 = r27 * r23;
    r70 = fma(r26, r70, r47);
    r70 = fma(r8, r70, r33);
    r29 = fma(r27, r29, r42);
    r49 = r24 + r49;
    r42 = r17 * r17;
    r42 = r42 * r27;
    r49 = r49 + r42;
    r47 = r18 * r13;
    r47 = r47 * r36;
    r71 = r17 * r14;
    r71 = fma(r36, r71, r47);
    r72 = r36 * r4;
    r72 = r72 * r23;
    r73 = fma(r45, r31, r72);
    r74 = r45 * r45;
    r74 = r74 * r27;
    r34 = r74 + r34;
    r70 = fma(r39, r29, r70);
    r70 = fma(r35, r49, r70);
    r70 = fma(r40, r71, r70);
    r70 = fma(r9, r73, r70);
    r70 = fma(r32, r34, r70);
    r34 = copysign(1.0, r70);
    r34 = fma(r69, r34, r70);
    r69 = r34 * r34;
    r70 = 1.0 / r69;
    r73 = r30 * r70;
    r75 = r36 * r23;
    r75 = r75 * r58;
    r76 = r36 * r4;
    r76 = fma(r55, r76, r75);
    r77 = r36 * r45;
    r77 = r77 * r52;
    r78 = r62 * r31;
    r79 = r77 + r78;
    r80 = r76 + r79;
    r81 = r27 * r26;
    r81 = fma(r27, r53, r55 * r81);
    r81 = r81 + r57;
    r81 = fma(r8, r81, r9 * r80);
    r80 = r45 * r66;
    r55 = r62 * r80;
    r10 = r10 + r55;
    r81 = fma(r32, r10, r81);
    r10 = r30 * r30;
    r69 = r34 * r69;
    r69 = 1.0 / r69;
    r69 = r27 * r69;
    r10 = r10 * r69;
    r68 = fma(r81, r10, r73 * r68);
    r82 = r81 * r69;
    r28 = fma(r4, r31, r28);
    r28 = fma(r8, r28, r7);
    r83 = r13 * r14;
    r83 = fma(r36, r83, r37);
    r51 = r24 + r51;
    r51 = r51 + r42;
    r42 = r17 * r14;
    r42 = fma(r27, r42, r47);
    r47 = r45 * r27;
    r47 = fma(r26, r47, r72);
    r5 = r24 + r5;
    r5 = r5 + r74;
    r28 = fma(r39, r83, r28);
    r28 = fma(r40, r51, r28);
    r28 = fma(r35, r42, r28);
    r28 = fma(r32, r47, r28);
    r28 = fma(r9, r5, r28);
    r5 = r28 * r28;
    r68 = fma(r5, r82, r68);
    r47 = r36 * r28;
    r74 = r45 * r27;
    r72 = r27 * r26;
    r72 = r72 * r62;
    r74 = fma(r52, r74, r72);
    r74 = r74 + r76;
    r55 = r67 + r55;
    r55 = fma(r9, r55, r32 * r74);
    r63 = fma(r58, r31, r63);
    r74 = r36 * r4;
    r74 = fma(r52, r74, r61);
    r63 = r63 + r74;
    r55 = fma(r8, r63, r55);
    r47 = r47 * r55;
    r68 = fma(r70, r47, r68);
    r68 = r41 * r68;
    r34 = 1.0 / r34;
    r34 = r44 * r34;
    r68 = r68 * r34;
    r47 = fma(r70, r5, r30 * r73);
    r47 = fma(r41, r47, r24);
    r24 = r47 * r34;
    r82 = fma(r64, r24, r30 * r68);
    r63 = r44 * r20;
    r63 = r63 * r47;
    r63 = r63 * r73;
    r82 = fma(r81, r63, r82);
    r55 = fma(r55, r24, r28 * r68);
    r68 = r44 * r20;
    r68 = r68 * r81;
    r68 = r68 * r28;
    r68 = r68 * r47;
    r55 = fma(r70, r68, r55);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r82, r55);
    r68 = r41 * r30;
    r78 = r75 + r78;
    r75 = r36 * r4;
    r61 = r11 * r14;
    r67 = r15 * r18;
    r67 = fma(r54, r67, r48 * r61);
    r61 = r12 * r13;
    r67 = fma(r48, r61, r67);
    r67 = fma(r17, r59, r67);
    r75 = r75 * r67;
    r61 = r36 * r45;
    r76 = r15 * r14;
    r37 = r11 * r18;
    r37 = fma(r54, r37, r54 * r76);
    r37 = fma(r17, r56, r37);
    r37 = fma(r13, r59, r37);
    r61 = fma(r37, r61, r75);
    r78 = r78 + r61;
    r59 = r23 * r62;
    r59 = r59 * r66;
    r76 = r4 * r66;
    r76 = r76 * r37;
    r84 = r59 + r76;
    r84 = fma(r8, r84, r32 * r78);
    r78 = r27 * r26;
    r78 = fma(r27, r65, r37 * r78);
    r85 = r36 * r45;
    r85 = r85 * r62;
    r86 = r36 * r23;
    r86 = fma(r67, r86, r85);
    r78 = r78 + r86;
    r84 = fma(r9, r78, r84);
    r78 = r36 * r84;
    r87 = r36 * r28;
    r88 = r45 * r27;
    r88 = fma(r58, r88, r60);
    r60 = r36 * r23;
    r60 = r60 * r37;
    r89 = r27 * r26;
    r88 = fma(r67, r89, r88);
    r88 = r88 + r60;
    r37 = fma(r37, r31, r36 * r65);
    r37 = r37 + r86;
    r37 = fma(r8, r37, r32 * r88);
    r88 = r67 * r80;
    r76 = r76 + r88;
    r37 = fma(r9, r76, r37);
    r87 = r87 * r37;
    r87 = fma(r70, r87, r73 * r78);
    r78 = r27 * r23;
    r78 = fma(r58, r78, r72);
    r78 = r78 + r61;
    r60 = fma(r67, r31, r60);
    r60 = r60 + r57;
    r60 = fma(r9, r60, r8 * r78);
    r88 = r59 + r88;
    r60 = fma(r32, r88, r60);
    r88 = r60 * r69;
    r87 = fma(r5, r88, r87);
    r87 = fma(r60, r10, r87);
    r68 = r68 * r87;
    r68 = fma(r60, r63, r34 * r68);
    r68 = fma(r84, r24, r68);
    r88 = r41 * r28;
    r88 = r88 * r87;
    r88 = fma(r34, r88, r37 * r24);
    r37 = r44 * r20;
    r37 = r37 * r28;
    r37 = r37 * r47;
    r37 = r37 * r60;
    r88 = fma(r70, r37, r88);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r68, r88);
    r37 = r16 * r14;
    r38 = fma(r48, r38, r54 * r37);
    r38 = fma(r18, r56, r38);
    r38 = fma(r54, r21, r38);
    r80 = r38 * r80;
    r53 = r66 * r53;
    r21 = r80 + r53;
    r54 = r36 * r4;
    r54 = r54 * r38;
    r85 = r85 + r54;
    r56 = r27 * r23;
    r85 = fma(r67, r56, r85);
    r48 = r27 * r26;
    r85 = fma(r52, r48, r85);
    r85 = fma(r8, r85, r32 * r21);
    r21 = r36 * r45;
    r21 = fma(r38, r31, r67 * r21);
    r21 = r21 + r74;
    r85 = fma(r9, r21, r85);
    r72 = r77 + r72;
    r77 = r36 * r23;
    r77 = r77 * r38;
    r21 = r4 * r27;
    r72 = fma(r67, r21, r72);
    r72 = r72 + r77;
    r62 = r4 * r62;
    r62 = r62 * r66;
    r53 = r62 + r53;
    r53 = fma(r8, r53, r9 * r72);
    r31 = fma(r52, r31, r54);
    r31 = r31 + r86;
    r53 = fma(r32, r31, r53);
    r31 = fma(r53, r24, r85 * r63);
    r86 = r41 * r30;
    r52 = r36 * r28;
    r77 = r75 + r77;
    r77 = r77 + r79;
    r79 = r45 * r27;
    r75 = r27 * r26;
    r75 = fma(r38, r75, r67 * r79);
    r75 = r75 + r74;
    r75 = fma(r32, r75, r8 * r77);
    r80 = r62 + r80;
    r75 = fma(r9, r80, r75);
    r52 = r52 * r75;
    r52 = fma(r85, r10, r70 * r52);
    r80 = r36 * r53;
    r52 = fma(r73, r80, r52);
    r62 = r85 * r69;
    r52 = fma(r5, r62, r52);
    r86 = r86 * r52;
    r31 = fma(r34, r86, r31);
    r86 = r41 * r28;
    r86 = r86 * r52;
    r52 = r44 * r20;
    r52 = r52 * r28;
    r52 = r52 * r47;
    r52 = r52 * r85;
    r52 = fma(r70, r52, r34 * r86);
    r52 = fma(r75, r24, r52);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r31, r52);
    r75 = r41 * r30;
    r86 = r36 * r83;
    r86 = r86 * r28;
    r62 = r29 * r69;
    r62 = fma(r5, r62, r70 * r86);
    r86 = r36 * r50;
    r62 = fma(r73, r86, r62);
    r62 = fma(r29, r10, r62);
    r75 = r75 * r62;
    r75 = fma(r34, r75, r29 * r63);
    r75 = fma(r50, r24, r75);
    r86 = r41 * r28;
    r86 = r86 * r62;
    r86 = fma(r34, r86, r83 * r24);
    r62 = r44 * r20;
    r62 = r62 * r29;
    r62 = r62 * r28;
    r62 = r62 * r47;
    r86 = fma(r70, r62, r86);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r75, r86);
    r62 = fma(r71, r63, r43 * r24);
    r80 = r41 * r30;
    r77 = r71 * r69;
    r74 = r36 * r51;
    r74 = r74 * r28;
    r74 = fma(r70, r74, r5 * r77);
    r77 = r36 * r43;
    r74 = fma(r73, r77, r74);
    r74 = fma(r71, r10, r74);
    r80 = r80 * r74;
    r62 = fma(r34, r80, r62);
    r80 = r41 * r28;
    r80 = r80 * r74;
    r80 = fma(r51, r24, r34 * r80);
    r74 = r44 * r20;
    r74 = r74 * r71;
    r74 = r74 * r28;
    r74 = r74 * r47;
    r80 = fma(r70, r74, r80);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r62, r80);
    r74 = r41 * r30;
    r77 = r36 * r42;
    r77 = r77 * r28;
    r79 = r49 * r69;
    r79 = fma(r5, r79, r70 * r77);
    r77 = r36 * r46;
    r79 = fma(r73, r77, r79);
    r79 = fma(r49, r10, r79);
    r74 = r74 * r79;
    r74 = fma(r46, r24, r34 * r74);
    r74 = fma(r49, r63, r74);
    r63 = r44 * r20;
    r63 = r63 * r49;
    r63 = r63 * r28;
    r63 = r63 * r47;
    r47 = r41 * r28;
    r47 = r47 * r79;
    r47 = fma(r34, r47, r70 * r63);
    r47 = fma(r42, r24, r47);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r74, r47);
    r63 = r20 * r55;
    r34 = fma(r3, r20, r1);
    r34 = fma(r28, r24, r34);
    r70 = r20 * r82;
    r79 = fma(r2, r20, r0);
    r79 = fma(r30, r24, r79);
    r70 = fma(r79, r70, r34 * r63);
    r63 = r20 * r68;
    r24 = r20 * r88;
    r24 = fma(r34, r24, r79 * r63);
    WriteSum2<double, double>((double *)inout_shared, r70, r24);
  };
  FlushSumShared<2, double>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = r20 * r52;
    r70 = r20 * r31;
    r70 = fma(r79, r70, r34 * r24);
    r24 = r20 * r75;
    r63 = r20 * r86;
    r63 = fma(r34, r63, r79 * r24);
    WriteSum2<double, double>((double *)inout_shared, r70, r63);
  };
  FlushSumShared<2, double>(out_pose_njtr, 2 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = r20 * r80;
    r70 = r20 * r62;
    r70 = fma(r79, r70, r34 * r63);
    r63 = r20 * r74;
    r24 = r20 * r47;
    r24 = fma(r34, r24, r79 * r63);
    WriteSum2<double, double>((double *)inout_shared, r70, r24);
  };
  FlushSumShared<2, double>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r82, r82, r55 * r55);
    r70 = fma(r88, r88, r68 * r68);
    WriteSum2<double, double>((double *)inout_shared, r24, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r31, r31, r52 * r52);
    r24 = fma(r75, r75, r86 * r86);
    WriteSum2<double, double>((double *)inout_shared, r70, r24);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r62, r62, r80 * r80);
    r70 = fma(r74, r74, r47 * r47);
    WriteSum2<double, double>((double *)inout_shared, r24, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r82, r68, r55 * r88);
    r24 = fma(r55, r52, r82 * r31);
    WriteSum2<double, double>((double *)inout_shared, r70, r24);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r55, r86, r82 * r75);
    r70 = fma(r82, r62, r55 * r80);
    WriteSum2<double, double>((double *)inout_shared, r24, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r82, r74, r55 * r47);
    r24 = fma(r88, r52, r68 * r31);
    WriteSum2<double, double>((double *)inout_shared, r70, r24);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r88, r86, r68 * r75);
    r70 = fma(r88, r80, r68 * r62);
    WriteSum2<double, double>((double *)inout_shared, r24, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r88, r47, r68 * r74);
    r24 = fma(r31, r75, r52 * r86);
    WriteSum2<double, double>((double *)inout_shared, r70, r24);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r31, r62, r52 * r80);
    r70 = fma(r52, r47, r31 * r74);
    WriteSum2<double, double>((double *)inout_shared, r24, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r86, r80, r75 * r62);
    r24 = fma(r86, r47, r75 * r74);
    WriteSum2<double, double>((double *)inout_shared, r70, r24);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r80, r47, r62 * r74);
    WriteSum1<double, double>((double *)inout_shared, r24);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = -2.00000000000000000e+00;
    r70 = fma(r16, r17, r11 * r14);
    r63 = r15 * r18;
    r34 = -1.00000000000000000e+00;
    r70 = fma(r34, r63, r70);
    r70 = fma(r12, r13, r70);
    r63 = r70 * r70;
    r63 = r24 * r63;
    r79 = 1.00000000000000000e+00;
    r10 = r11 * r17;
    r10 = fma(r34, r10, r16 * r14);
    r10 = fma(r12, r18, r10);
    r10 = fma(r15, r13, r10);
    r77 = r10 * r10;
    r77 = fma(r24, r77, r79);
    r73 = r63 + r77;
    r73 = fma(r8, r73, r6);
    r6 = 2.00000000000000000e+00;
    r5 = fma(r12, r17, r15 * r14);
    r38 = r16 * r13;
    r5 = fma(r34, r38, r5);
    r5 = fma(r11, r18, r5);
    r38 = r6 * r5;
    r67 = r10 * r38;
    r54 = fma(r16, r18, r15 * r17);
    r54 = fma(r11, r13, r54);
    r54 = fma(r34, r54, r12 * r14);
    r72 = r24 * r54;
    r66 = fma(r70, r72, r67);
    r21 = r6 * r10;
    r48 = r70 * r38;
    r21 = fma(r54, r21, r48);
    r56 = r17 * r13;
    r56 = r56 * r6;
    r37 = r18 * r14;
    r87 = fma(r6, r37, r56);
    r59 = r13 * r14;
    r78 = r17 * r18;
    r78 = r78 * r6;
    r59 = fma(r24, r59, r78);
    r57 = r18 * r18;
    r57 = r57 * r24;
    r61 = r79 + r57;
    r58 = r13 * r13;
    r58 = r24 * r58;
    r61 = r61 + r58;
    r73 = fma(r9, r66, r73);
    r73 = fma(r32, r21, r73);
    r73 = fma(r35, r87, r73);
    r73 = fma(r40, r59, r73);
    r73 = fma(r39, r61, r73);
    r61 = 1.00000000000000008e-15;
    r48 = fma(r10, r72, r48);
    r48 = fma(r8, r48, r33);
    r37 = fma(r24, r37, r56);
    r57 = r79 + r57;
    r56 = r17 * r17;
    r56 = r24 * r56;
    r57 = r57 + r56;
    r33 = r18 * r13;
    r33 = r33 * r6;
    r59 = r17 * r14;
    r59 = fma(r6, r59, r33);
    r87 = r6 * r70;
    r87 = r87 * r10;
    r38 = fma(r54, r38, r87);
    r21 = r5 * r5;
    r21 = r21 * r24;
    r77 = r21 + r77;
    r48 = fma(r39, r37, r48);
    r48 = fma(r35, r57, r48);
    r48 = fma(r40, r59, r48);
    r48 = fma(r9, r38, r48);
    r48 = fma(r32, r77, r48);
    r77 = copysign(1.0, r48);
    r77 = fma(r61, r77, r48);
    r61 = r77 * r77;
    r61 = 1.0 / r61;
    r48 = r73 * r73;
    r38 = r6 * r70;
    r38 = fma(r54, r38, r67);
    r38 = fma(r8, r38, r7);
    r8 = r13 * r14;
    r8 = fma(r6, r8, r78);
    r58 = r79 + r58;
    r58 = r58 + r56;
    r56 = r17 * r14;
    r56 = fma(r24, r56, r33);
    r72 = fma(r5, r72, r87);
    r63 = r79 + r63;
    r63 = r63 + r21;
    r38 = fma(r39, r8, r38);
    r38 = fma(r40, r58, r38);
    r38 = fma(r35, r56, r38);
    r38 = fma(r32, r72, r38);
    r38 = fma(r9, r63, r38);
    r63 = r38 * r38;
    r9 = fma(r61, r63, r61 * r48);
    r79 = fma(r41, r9, r79);
    r77 = 1.0 / r77;
    r72 = r79 * r77;
    r32 = r73 * r72;
    r56 = r38 * r72;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac, 0 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx, r32, r56);
    r56 = r73 * r77;
    r32 = r44 * r9;
    r56 = r56 * r32;
    r35 = r38 * r77;
    r35 = r35 * r32;
    WriteIdx2<1024, double, double, double2>(
        out_focal_and_extra_jac, 2 * out_focal_and_extra_jac_num_alloc,
        global_thread_idx, r56, r35);
    r35 = r34 * r73;
    r2 = fma(r2, r34, r0);
    r0 = r44 * r73;
    r2 = fma(r72, r0, r2);
    r35 = r35 * r2;
    r2 = r34 * r38;
    r3 = fma(r3, r34, r1);
    r1 = r44 * r38;
    r3 = fma(r72, r1, r3);
    r2 = r2 * r3;
    r2 = fma(r72, r2, r72 * r35);
    r72 = r77 * r32;
    r1 = r34 * r38;
    r1 = r1 * r3;
    r1 = r1 * r77;
    r1 = fma(r32, r1, r35 * r72);
    WriteSum2<double, double>((double *)inout_shared, r2, r1);
  };
  FlushSumShared<2, double>(
      out_focal_and_extra_njtr, 0 * out_focal_and_extra_njtr_num_alloc,
      focal_and_extra_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r79 * r79;
    r1 = r1 * r61;
    r1 = fma(r63, r1, r48 * r1);
    r9 = r44 * r9;
    r61 = r61 * r32;
    r9 = r9 * r61;
    r9 = fma(r63, r9, r48 * r9);
    WriteSum2<double, double>((double *)inout_shared, r1, r9);
  };
  FlushSumShared<2, double>(out_focal_and_extra_precond_diag,
                            0 * out_focal_and_extra_precond_diag_num_alloc,
                            focal_and_extra_indices_loc,
                            (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = r79 * r48;
    r1 = r79 * r63;
    r1 = fma(r61, r1, r61 * r9);
    WriteSum1<double, double>((double *)inout_shared, r1);
  };
  FlushSumShared<1, double>(out_focal_and_extra_precond_tril,
                            0 * out_focal_and_extra_precond_tril_num_alloc,
                            focal_and_extra_indices_loc,
                            (double *)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedPrincipalPointFixedPointResJacFirst(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    double *focal_and_extra, unsigned int focal_and_extra_num_alloc,
    SharedIndex *focal_and_extra_indices, double *pixel,
    unsigned int pixel_num_alloc, double *principal_point,
    unsigned int principal_point_num_alloc, double *point,
    unsigned int point_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *const out_rTr, double *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, double *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, double *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    double *out_focal_and_extra_jac,
    unsigned int out_focal_and_extra_jac_num_alloc,
    double *const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    double *const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    double *const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
    size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedPrincipalPointFixedPointResJacFirstKernel<<<n_blocks,
                                                                    1024>>>(
      pose, pose_num_alloc, pose_indices, sensor_from_rig,
      sensor_from_rig_num_alloc, focal_and_extra, focal_and_extra_num_alloc,
      focal_and_extra_indices, pixel, pixel_num_alloc, principal_point,
      principal_point_num_alloc, point, point_num_alloc, out_res,
      out_res_num_alloc, out_rTr, out_pose_jac, out_pose_jac_num_alloc,
      out_pose_njtr, out_pose_njtr_num_alloc, out_pose_precond_diag,
      out_pose_precond_diag_num_alloc, out_pose_precond_tril,
      out_pose_precond_tril_num_alloc, out_focal_and_extra_jac,
      out_focal_and_extra_jac_num_alloc, out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc, out_focal_and_extra_precond_diag,
      out_focal_and_extra_precond_diag_num_alloc,
      out_focal_and_extra_precond_tril,
      out_focal_and_extra_precond_tril_num_alloc, problem_size);
}

} // namespace caspar