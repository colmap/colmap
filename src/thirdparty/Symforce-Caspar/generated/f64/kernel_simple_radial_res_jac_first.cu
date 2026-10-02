#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SimpleRadialResJacFirstKernel(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    double *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    double *point, unsigned int point_num_alloc, SharedIndex *point_indices,
    double *pixel, unsigned int pixel_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *const out_rTr, double *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, double *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, double *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, double *out_calib_jac,
    unsigned int out_calib_jac_num_alloc, double *const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc, double *const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    double *const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc, double *out_point_jac,
    unsigned int out_point_jac_num_alloc, double *const out_point_njtr,
    unsigned int out_point_njtr_num_alloc, double *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
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
         r89 = 0, r90 = 0, r91 = 0, r92 = 0, r93 = 0, r94 = 0, r95 = 0, r96 = 0,
         r97 = 0, r98 = 0, r99 = 0, r100 = 0;
  LoadShared<2, double, double>(calib, 2 * calib_num_alloc, calib_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        calib_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(pixel, 0 * pixel_num_alloc,
                                            global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r5 = fma(r2, r4, r0);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r6, r7);
  };
  LoadShared<2, double, double>(point, 0 * point_num_alloc, point_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        point_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
  };
  LoadShared<1, double, double>(point, 2 * point_num_alloc, point_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double *)inout_shared,
                        point_indices_loc[threadIdx.x].target, r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
  LoadShared<2, double, double>(calib, 0 * calib_num_alloc, calib_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        calib_indices_loc[threadIdx.x].target, r44, r41);
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
    r5 = fma(r5, r5, r4 * r4);
  };
  SumStore<double>(out_rTr_local, (double *)inout_shared, 0,
                   global_thread_idx < problem_size, r5);
  if (global_thread_idx < problem_size) {
    r5 = 2.00000000000000000e+00;
    r4 = r11 * r17;
    r20 = -1.00000000000000000e+00;
    r4 = fma(r20, r4, r16 * r14);
    r4 = fma(r12, r18, r4);
    r4 = fma(r15, r13, r4);
    r27 = r15 * r14;
    r24 = -5.00000000000000000e-01;
    r23 = r12 * r17;
    r23 = fma(r24, r23, r24 * r27);
    r27 = r11 * r18;
    r23 = fma(r24, r27, r23);
    r21 = r16 * r13;
    r38 = 5.00000000000000000e-01;
    r23 = fma(r38, r21, r23);
    r21 = r4 * r23;
    r27 = fma(r16, r18, r15 * r17);
    r27 = fma(r11, r13, r27);
    r27 = fma(r20, r27, r12 * r14);
    r34 = r5 * r27;
    r30 = r16 * r17;
    r36 = r11 * r14;
    r30 = fma(r38, r36, r38 * r30);
    r45 = r15 * r18;
    r28 = r12 * r38;
    r30 = fma(r24, r45, r30);
    r30 = fma(r13, r28, r30);
    r48 = fma(r30, r34, r5 * r21);
    r26 = fma(r12, r17, r15 * r14);
    r47 = r16 * r13;
    r26 = fma(r20, r47, r26);
    r26 = fma(r11, r18, r26);
    r47 = r5 * r26;
    r31 = r11 * r17;
    r10 = r12 * r18;
    r10 = fma(r24, r10, r38 * r31);
    r31 = r15 * r13;
    r10 = fma(r24, r31, r10);
    r42 = r16 * r24;
    r10 = fma(r14, r42, r10);
    r31 = fma(r16, r17, r36);
    r31 = fma(r12, r13, r31);
    r31 = fma(r20, r45, r31);
    r29 = r5 * r31;
    r46 = r15 * r17;
    r43 = r11 * r13;
    r43 = fma(r24, r43, r24 * r46);
    r43 = fma(r14, r28, r43);
    r43 = fma(r18, r42, r43);
    r29 = r29 * r43;
    r47 = fma(r10, r47, r29);
    r48 = r48 + r47;
    r46 = r5 * r4;
    r46 = r46 * r43;
    r37 = r5 * r26;
    r37 = r37 * r30;
    r49 = r46 + r37;
    r50 = -2.00000000000000000e+00;
    r51 = r31 * r50;
    r49 = fma(r23, r51, r49);
    r52 = r50 * r27;
    r49 = fma(r10, r52, r49);
    r49 = fma(r9, r49, r32 * r48);
    r48 = r4 * r30;
    r52 = -4.00000000000000000e+00;
    r48 = r48 * r52;
    r51 = r31 * r10;
    r53 = r52 * r51;
    r54 = r48 + r53;
    r49 = fma(r8, r54, r49);
    r54 = 1.00000000000000008e-15;
    r55 = r5 * r31;
    r55 = r55 * r26;
    r56 = r50 * r4;
    r56 = fma(r27, r56, r55);
    r56 = fma(r8, r56, r33);
    r57 = r17 * r13;
    r57 = r57 * r5;
    r58 = r18 * r14;
    r58 = fma(r50, r58, r57);
    r59 = r18 * r18;
    r59 = r59 * r50;
    r60 = 1.00000000000000000e+00;
    r61 = r17 * r17;
    r61 = fma(r50, r61, r60);
    r62 = r59 + r61;
    r63 = r18 * r13;
    r63 = r63 * r5;
    r64 = r17 * r14;
    r64 = fma(r5, r64, r63);
    r65 = r5 * r31;
    r65 = r65 * r4;
    r66 = fma(r26, r34, r65);
    r67 = r50 * r4;
    r67 = r67 * r4;
    r68 = r60 + r67;
    r69 = r26 * r26;
    r69 = r69 * r50;
    r68 = r68 + r69;
    r56 = fma(r39, r58, r56);
    r56 = fma(r35, r62, r56);
    r56 = fma(r40, r64, r56);
    r56 = fma(r9, r66, r56);
    r56 = fma(r32, r68, r56);
    r68 = copysign(1.0, r56);
    r68 = fma(r54, r68, r56);
    r54 = 1.0 / r68;
    r67 = r60 + r67;
    r56 = r31 * r31;
    r56 = r56 * r50;
    r67 = r67 + r56;
    r67 = fma(r8, r67, r6);
    r66 = r5 * r26;
    r66 = r66 * r4;
    r70 = r31 * r50;
    r70 = fma(r27, r70, r66);
    r55 = fma(r4, r34, r55);
    r71 = r18 * r14;
    r71 = fma(r5, r71, r57);
    r57 = r13 * r14;
    r72 = r17 * r18;
    r72 = r72 * r5;
    r57 = fma(r50, r57, r72);
    r59 = r60 + r59;
    r73 = r13 * r13;
    r73 = r50 * r73;
    r59 = r59 + r73;
    r67 = fma(r9, r70, r67);
    r67 = fma(r32, r55, r67);
    r67 = fma(r35, r71, r67);
    r67 = fma(r40, r57, r67);
    r67 = fma(r39, r59, r67);
    r55 = r68 * r68;
    r70 = 1.0 / r55;
    r74 = r67 * r70;
    r66 = fma(r31, r34, r66);
    r66 = fma(r8, r66, r7);
    r75 = r13 * r14;
    r75 = fma(r5, r75, r72);
    r61 = r73 + r61;
    r73 = r17 * r14;
    r73 = fma(r50, r73, r63);
    r63 = r26 * r50;
    r63 = fma(r27, r63, r65);
    r69 = r60 + r69;
    r69 = r69 + r56;
    r66 = fma(r39, r75, r66);
    r66 = fma(r40, r61, r66);
    r66 = fma(r35, r73, r66);
    r66 = fma(r32, r63, r66);
    r66 = fma(r9, r69, r66);
    r69 = r66 * r66;
    r63 = fma(r70, r69, r67 * r74);
    r63 = fma(r41, r63, r60);
    r63 = r44 * r63;
    r60 = r54 * r63;
    r56 = r44 * r41;
    r65 = r5 * r49;
    r72 = r5 * r4;
    r72 = r72 * r10;
    r76 = r5 * r31;
    r76 = fma(r30, r76, r72);
    r77 = r5 * r26;
    r77 = r77 * r23;
    r78 = r43 * r34;
    r79 = r77 + r78;
    r80 = r76 + r79;
    r81 = r50 * r27;
    r81 = fma(r50, r21, r30 * r81);
    r81 = r81 + r47;
    r81 = fma(r8, r81, r9 * r80);
    r80 = r26 * r52;
    r30 = r43 * r80;
    r48 = r48 + r30;
    r81 = fma(r32, r48, r81);
    r48 = r67 * r67;
    r55 = r68 * r55;
    r55 = 1.0 / r55;
    r55 = r50 * r55;
    r48 = r48 * r55;
    r65 = fma(r81, r48, r74 * r65);
    r68 = r81 * r55;
    r65 = fma(r69, r68, r65);
    r82 = r5 * r66;
    r83 = r26 * r50;
    r84 = r50 * r27;
    r84 = r84 * r43;
    r83 = fma(r23, r83, r84);
    r83 = r83 + r76;
    r30 = r53 + r30;
    r30 = fma(r9, r30, r32 * r83);
    r37 = fma(r10, r34, r37);
    r83 = r5 * r31;
    r83 = fma(r23, r83, r46);
    r37 = r37 + r83;
    r30 = fma(r8, r37, r30);
    r82 = r82 * r30;
    r65 = fma(r70, r82, r65);
    r56 = r56 * r65;
    r56 = r56 * r54;
    r65 = fma(r67, r56, r49 * r60);
    r82 = r20 * r63;
    r82 = r82 * r74;
    r65 = fma(r81, r82, r65);
    r68 = r20 * r66;
    r68 = r68 * r81;
    r68 = r68 * r70;
    r68 = fma(r63, r68, r66 * r56);
    r68 = fma(r30, r60, r68);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r65, r68);
    r78 = r72 + r78;
    r72 = r5 * r31;
    r30 = r12 * r13;
    r36 = fma(r24, r36, r24 * r30);
    r36 = fma(r17, r42, r36);
    r36 = fma(r38, r45, r36);
    r72 = r72 * r36;
    r45 = r5 * r26;
    r30 = r15 * r14;
    r56 = r11 * r18;
    r56 = fma(r38, r56, r38 * r30);
    r56 = fma(r17, r28, r56);
    r56 = fma(r13, r42, r56);
    r45 = fma(r56, r45, r72);
    r78 = r78 + r45;
    r42 = r4 * r43;
    r42 = r42 * r52;
    r30 = r31 * r52;
    r30 = r30 * r56;
    r37 = r42 + r30;
    r37 = fma(r8, r37, r32 * r78);
    r78 = r50 * r27;
    r78 = fma(r50, r51, r56 * r78);
    r46 = r5 * r26;
    r46 = r46 * r43;
    r53 = r5 * r4;
    r53 = fma(r36, r53, r46);
    r78 = r78 + r53;
    r37 = fma(r9, r78, r37);
    r78 = r50 * r4;
    r78 = fma(r10, r78, r84);
    r78 = r78 + r45;
    r45 = r5 * r4;
    r45 = r45 * r56;
    r76 = fma(r36, r34, r45);
    r76 = r76 + r47;
    r76 = fma(r9, r76, r8 * r78);
    r78 = r36 * r80;
    r42 = r42 + r78;
    r76 = fma(r32, r42, r76);
    r42 = fma(r76, r82, r37 * r60);
    r47 = r44 * r41;
    r85 = r5 * r37;
    r86 = r5 * r66;
    r45 = r29 + r45;
    r29 = r26 * r50;
    r45 = fma(r10, r29, r45);
    r10 = r50 * r27;
    r45 = fma(r36, r10, r45);
    r56 = fma(r56, r34, r5 * r51);
    r56 = r56 + r53;
    r56 = fma(r8, r56, r32 * r45);
    r78 = r30 + r78;
    r56 = fma(r9, r78, r56);
    r86 = r86 * r56;
    r86 = fma(r70, r86, r74 * r85);
    r85 = r76 * r55;
    r86 = fma(r69, r85, r86);
    r86 = fma(r76, r48, r86);
    r47 = r47 * r67;
    r47 = r47 * r86;
    r42 = fma(r54, r47, r42);
    r47 = r44 * r41;
    r47 = r47 * r66;
    r47 = r47 * r86;
    r86 = r20 * r66;
    r86 = r86 * r76;
    r86 = r86 * r70;
    r86 = fma(r63, r86, r54 * r47);
    r86 = fma(r56, r60, r86);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r42, r86);
    r56 = r16 * r14;
    r47 = r11 * r17;
    r47 = fma(r24, r47, r38 * r56);
    r56 = r15 * r13;
    r47 = fma(r38, r56, r47);
    r47 = fma(r18, r28, r47);
    r80 = r47 * r80;
    r21 = r52 * r21;
    r28 = r80 + r21;
    r56 = r5 * r31;
    r56 = r56 * r47;
    r46 = r46 + r56;
    r38 = r50 * r4;
    r46 = fma(r36, r38, r46);
    r24 = r50 * r27;
    r46 = fma(r23, r24, r46);
    r46 = fma(r8, r46, r32 * r28);
    r28 = r5 * r26;
    r28 = fma(r47, r34, r36 * r28);
    r28 = r28 + r83;
    r46 = fma(r9, r28, r46);
    r84 = r77 + r84;
    r77 = r5 * r4;
    r77 = r77 * r47;
    r28 = r31 * r50;
    r84 = fma(r36, r28, r84);
    r84 = r84 + r77;
    r43 = r31 * r43;
    r43 = r43 * r52;
    r21 = r43 + r21;
    r21 = fma(r8, r21, r9 * r84);
    r34 = fma(r23, r34, r56);
    r34 = r34 + r53;
    r21 = fma(r32, r34, r21);
    r34 = fma(r21, r60, r46 * r82);
    r53 = r44 * r41;
    r23 = r5 * r66;
    r77 = r72 + r77;
    r77 = r77 + r79;
    r79 = r26 * r50;
    r72 = r50 * r27;
    r72 = fma(r47, r72, r36 * r79);
    r72 = r72 + r83;
    r72 = fma(r32, r72, r8 * r77);
    r80 = r43 + r80;
    r72 = fma(r9, r80, r72);
    r23 = r23 * r72;
    r23 = fma(r46, r48, r70 * r23);
    r80 = r5 * r21;
    r23 = fma(r74, r80, r23);
    r43 = r46 * r55;
    r23 = fma(r69, r43, r23);
    r53 = r53 * r67;
    r53 = r53 * r23;
    r34 = fma(r54, r53, r34);
    r53 = r20 * r66;
    r53 = r53 * r46;
    r53 = r53 * r70;
    r53 = fma(r63, r53, r72 * r60);
    r72 = r44 * r41;
    r72 = r72 * r66;
    r72 = r72 * r23;
    r53 = fma(r54, r72, r53);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r34, r53);
    r72 = r44 * r41;
    r23 = r5 * r75;
    r23 = r23 * r66;
    r43 = r58 * r55;
    r43 = fma(r69, r43, r70 * r23);
    r23 = r5 * r59;
    r43 = fma(r74, r23, r43);
    r43 = fma(r58, r48, r43);
    r72 = r72 * r67;
    r72 = r72 * r43;
    r72 = fma(r59, r60, r54 * r72);
    r72 = fma(r58, r82, r72);
    r23 = r44 * r41;
    r23 = r23 * r66;
    r23 = r23 * r43;
    r23 = fma(r54, r23, r75 * r60);
    r43 = r20 * r58;
    r43 = r43 * r66;
    r43 = r43 * r70;
    r23 = fma(r63, r43, r23);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r72, r23);
    r43 = r44 * r41;
    r80 = r64 * r55;
    r77 = r5 * r61;
    r77 = r77 * r66;
    r77 = fma(r70, r77, r69 * r80);
    r80 = r5 * r57;
    r77 = fma(r74, r80, r77);
    r77 = fma(r64, r48, r77);
    r43 = r43 * r67;
    r43 = r43 * r77;
    r43 = fma(r54, r43, r64 * r82);
    r43 = fma(r57, r60, r43);
    r80 = r20 * r64;
    r80 = r80 * r66;
    r80 = r80 * r70;
    r83 = r44 * r41;
    r83 = r83 * r66;
    r83 = r83 * r77;
    r83 = fma(r54, r83, r63 * r80);
    r83 = fma(r61, r60, r83);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r43, r83);
    r80 = r44 * r41;
    r77 = r5 * r73;
    r77 = r77 * r66;
    r79 = r62 * r55;
    r79 = fma(r69, r79, r70 * r77);
    r77 = r5 * r71;
    r79 = fma(r74, r77, r79);
    r79 = fma(r62, r48, r79);
    r80 = r80 * r67;
    r80 = r80 * r79;
    r82 = fma(r62, r82, r54 * r80);
    r82 = fma(r71, r60, r82);
    r80 = r44 * r41;
    r80 = r80 * r66;
    r80 = r80 * r79;
    r80 = fma(r73, r60, r54 * r80);
    r54 = r20 * r62;
    r54 = r54 * r66;
    r54 = r54 * r70;
    r80 = fma(r63, r54, r80);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r82, r80);
    r54 = r20 * r68;
    r63 = fma(r3, r20, r1);
    r63 = fma(r66, r60, r63);
    r70 = r20 * r65;
    r79 = fma(r2, r20, r0);
    r79 = fma(r67, r60, r79);
    r70 = fma(r79, r70, r63 * r54);
    r54 = r20 * r86;
    r60 = r20 * r42;
    r60 = fma(r79, r60, r63 * r54);
    WriteSum2<double, double>((double *)inout_shared, r70, r60);
  };
  FlushSumShared<2, double>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = r20 * r53;
    r70 = r20 * r34;
    r70 = fma(r79, r70, r63 * r60);
    r60 = r20 * r23;
    r54 = r20 * r72;
    r54 = fma(r79, r54, r63 * r60);
    WriteSum2<double, double>((double *)inout_shared, r70, r54);
  };
  FlushSumShared<2, double>(out_pose_njtr, 2 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = r20 * r83;
    r70 = r20 * r43;
    r70 = fma(r79, r70, r63 * r54);
    r54 = r20 * r80;
    r60 = r20 * r82;
    r60 = fma(r79, r60, r63 * r54);
    WriteSum2<double, double>((double *)inout_shared, r70, r60);
  };
  FlushSumShared<2, double>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = fma(r68, r68, r65 * r65);
    r70 = fma(r42, r42, r86 * r86);
    WriteSum2<double, double>((double *)inout_shared, r60, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r53, r53, r34 * r34);
    r60 = fma(r72, r72, r23 * r23);
    WriteSum2<double, double>((double *)inout_shared, r70, r60);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = fma(r83, r83, r43 * r43);
    r70 = fma(r82, r82, r80 * r80);
    WriteSum2<double, double>((double *)inout_shared, r60, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r65, r42, r68 * r86);
    r60 = fma(r68, r53, r65 * r34);
    WriteSum2<double, double>((double *)inout_shared, r70, r60);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = fma(r65, r72, r68 * r23);
    r70 = fma(r65, r43, r68 * r83);
    WriteSum2<double, double>((double *)inout_shared, r60, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r68, r80, r65 * r82);
    r60 = fma(r86, r53, r42 * r34);
    WriteSum2<double, double>((double *)inout_shared, r70, r60);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = fma(r86, r23, r42 * r72);
    r70 = fma(r86, r83, r42 * r43);
    WriteSum2<double, double>((double *)inout_shared, r60, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r86, r80, r42 * r82);
    r60 = fma(r53, r23, r34 * r72);
    WriteSum2<double, double>((double *)inout_shared, r70, r60);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = fma(r53, r83, r34 * r43);
    r70 = fma(r53, r80, r34 * r82);
    WriteSum2<double, double>((double *)inout_shared, r60, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r23, r83, r72 * r43);
    r60 = fma(r23, r80, r72 * r82);
    WriteSum2<double, double>((double *)inout_shared, r70, r60);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = fma(r43, r82, r83 * r80);
    WriteSum1<double, double>((double *)inout_shared, r60);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = -2.00000000000000000e+00;
    r70 = fma(r16, r17, r11 * r14);
    r54 = r15 * r18;
    r79 = -1.00000000000000000e+00;
    r70 = fma(r79, r54, r70);
    r70 = fma(r12, r13, r70);
    r54 = r70 * r70;
    r54 = r60 * r54;
    r63 = 1.00000000000000000e+00;
    r67 = r11 * r17;
    r67 = fma(r79, r67, r16 * r14);
    r67 = fma(r12, r18, r67);
    r67 = fma(r15, r13, r67);
    r48 = r67 * r67;
    r48 = fma(r60, r48, r63);
    r77 = r54 + r48;
    r77 = fma(r8, r77, r6);
    r74 = 2.00000000000000000e+00;
    r69 = fma(r12, r17, r15 * r14);
    r47 = r16 * r13;
    r69 = fma(r79, r47, r69);
    r69 = fma(r11, r18, r69);
    r47 = r74 * r69;
    r36 = r67 * r47;
    r56 = fma(r16, r18, r15 * r17);
    r56 = fma(r11, r13, r56);
    r56 = fma(r79, r56, r12 * r14);
    r84 = r60 * r56;
    r52 = fma(r70, r84, r36);
    r28 = r74 * r67;
    r24 = r70 * r47;
    r28 = fma(r56, r28, r24);
    r38 = r17 * r13;
    r38 = r38 * r74;
    r85 = r18 * r14;
    r78 = fma(r74, r85, r38);
    r30 = r13 * r14;
    r45 = r17 * r18;
    r45 = r45 * r74;
    r30 = fma(r60, r30, r45);
    r51 = r18 * r18;
    r51 = r51 * r60;
    r10 = r63 + r51;
    r29 = r13 * r13;
    r29 = r60 * r29;
    r10 = r10 + r29;
    r77 = fma(r9, r52, r77);
    r77 = fma(r32, r28, r77);
    r77 = fma(r35, r78, r77);
    r77 = fma(r40, r30, r77);
    r77 = fma(r39, r10, r77);
    r10 = 1.00000000000000008e-15;
    r24 = fma(r67, r84, r24);
    r24 = fma(r8, r24, r33);
    r85 = fma(r60, r85, r38);
    r51 = r63 + r51;
    r38 = r17 * r17;
    r38 = r60 * r38;
    r51 = r51 + r38;
    r30 = r18 * r13;
    r30 = r30 * r74;
    r78 = r17 * r14;
    r78 = fma(r74, r78, r30);
    r28 = r74 * r70;
    r28 = r28 * r67;
    r47 = fma(r56, r47, r28);
    r52 = r69 * r69;
    r52 = r52 * r60;
    r48 = r52 + r48;
    r24 = fma(r39, r85, r24);
    r24 = fma(r35, r51, r24);
    r24 = fma(r40, r78, r24);
    r24 = fma(r9, r47, r24);
    r24 = fma(r32, r48, r24);
    r48 = copysign(1.0, r24);
    r48 = fma(r10, r48, r24);
    r10 = r48 * r48;
    r10 = 1.0 / r10;
    r24 = r77 * r77;
    r47 = r74 * r70;
    r47 = fma(r56, r47, r36);
    r47 = fma(r8, r47, r7);
    r36 = r13 * r14;
    r36 = fma(r74, r36, r45);
    r29 = r63 + r29;
    r29 = r29 + r38;
    r38 = r17 * r14;
    r38 = fma(r60, r38, r30);
    r84 = fma(r69, r84, r28);
    r54 = r63 + r54;
    r54 = r54 + r52;
    r47 = fma(r39, r36, r47);
    r47 = fma(r40, r29, r47);
    r47 = fma(r35, r38, r47);
    r47 = fma(r32, r84, r47);
    r47 = fma(r9, r54, r47);
    r54 = r47 * r47;
    r84 = fma(r10, r54, r10 * r24);
    r38 = fma(r41, r84, r63);
    r48 = 1.0 / r48;
    r29 = r38 * r48;
    r36 = r77 * r29;
    r52 = r47 * r29;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             0 * out_calib_jac_num_alloc,
                                             global_thread_idx, r36, r52);
    r69 = r77 * r48;
    r28 = r44 * r84;
    r69 = r69 * r28;
    r30 = r47 * r48;
    r30 = r30 * r28;
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             2 * out_calib_jac_num_alloc,
                                             global_thread_idx, r69, r30);
    r60 = fma(r3, r79, r1);
    r45 = r44 * r47;
    r60 = fma(r29, r45, r60);
    r60 = r79 * r60;
    r45 = r47 * r60;
    r56 = r79 * r77;
    r78 = fma(r2, r79, r0);
    r51 = r44 * r77;
    r78 = fma(r29, r51, r78);
    r56 = r56 * r78;
    r56 = fma(r29, r56, r29 * r45);
    r29 = r79 * r77;
    r29 = r29 * r78;
    r29 = r29 * r48;
    r51 = r48 * r28;
    r51 = fma(r45, r51, r28 * r29);
    WriteSum2<double, double>((double *)inout_shared, r56, r51);
  };
  FlushSumShared<2, double>(out_calib_njtr, 0 * out_calib_njtr_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = r79 * r78;
    WriteSum2<double, double>((double *)inout_shared, r78, r60);
  };
  FlushSumShared<2, double>(out_calib_njtr, 2 * out_calib_njtr_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = r38 * r38;
    r60 = r60 * r10;
    r60 = fma(r54, r60, r24 * r60);
    r84 = r44 * r84;
    r10 = r10 * r28;
    r84 = r84 * r10;
    r84 = fma(r24, r84, r54 * r84);
    WriteSum2<double, double>((double *)inout_shared, r60, r84);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            0 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double *)inout_shared, r63, r63);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            2 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = r38 * r24;
    r84 = r38 * r54;
    r84 = fma(r10, r84, r10 * r63);
    WriteSum2<double, double>((double *)inout_shared, r84, r36);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double *)inout_shared, r52, r69);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            2 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = 0.00000000000000000e+00;
    WriteSum2<double, double>((double *)inout_shared, r30, r69);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = -2.00000000000000000e+00;
    r30 = fma(r16, r17, r11 * r14);
    r52 = r15 * r18;
    r36 = -1.00000000000000000e+00;
    r30 = fma(r36, r52, r30);
    r30 = fma(r12, r13, r30);
    r52 = r30 * r30;
    r52 = r69 * r52;
    r84 = 1.00000000000000000e+00;
    r63 = r11 * r17;
    r63 = fma(r36, r63, r16 * r14);
    r63 = fma(r12, r18, r63);
    r63 = fma(r15, r13, r63);
    r10 = r63 * r63;
    r10 = fma(r69, r10, r84);
    r60 = r52 + r10;
    r6 = fma(r8, r60, r6);
    r78 = r30 * r69;
    r51 = fma(r16, r18, r15 * r17);
    r51 = fma(r11, r13, r51);
    r51 = fma(r36, r51, r12 * r14);
    r56 = 2.00000000000000000e+00;
    r29 = fma(r12, r17, r15 * r14);
    r45 = r16 * r13;
    r29 = fma(r36, r45, r29);
    r29 = fma(r11, r18, r29);
    r45 = r56 * r29;
    r85 = r63 * r45;
    r78 = fma(r51, r78, r85);
    r87 = r56 * r63;
    r88 = r30 * r45;
    r87 = fma(r51, r87, r88);
    r89 = r17 * r13;
    r89 = r89 * r56;
    r90 = r18 * r14;
    r91 = fma(r56, r90, r89);
    r92 = r13 * r14;
    r93 = r17 * r18;
    r93 = r93 * r56;
    r92 = fma(r69, r92, r93);
    r94 = r18 * r18;
    r94 = r94 * r69;
    r95 = r84 + r94;
    r96 = r13 * r13;
    r96 = r69 * r96;
    r95 = r95 + r96;
    r6 = fma(r9, r78, r6);
    r6 = fma(r32, r87, r6);
    r6 = fma(r35, r91, r6);
    r6 = fma(r40, r92, r6);
    r6 = fma(r39, r95, r6);
    r95 = r56 * r60;
    r92 = 1.00000000000000008e-15;
    r91 = r69 * r63;
    r91 = fma(r51, r91, r88);
    r33 = fma(r8, r91, r33);
    r90 = fma(r69, r90, r89);
    r94 = r84 + r94;
    r89 = r17 * r17;
    r89 = r69 * r89;
    r94 = r94 + r89;
    r88 = r18 * r13;
    r88 = r88 * r56;
    r97 = r17 * r14;
    r97 = fma(r56, r97, r88);
    r98 = r56 * r30;
    r98 = r98 * r63;
    r45 = fma(r51, r45, r98);
    r99 = r29 * r29;
    r99 = r99 * r69;
    r10 = r99 + r10;
    r33 = fma(r39, r90, r33);
    r33 = fma(r35, r94, r33);
    r33 = fma(r40, r97, r33);
    r33 = fma(r9, r45, r33);
    r33 = fma(r32, r10, r33);
    r97 = copysign(1.0, r33);
    r97 = fma(r92, r97, r33);
    r92 = r97 * r97;
    r33 = 1.0 / r92;
    r94 = r6 * r33;
    r92 = r97 * r92;
    r92 = 1.0 / r92;
    r92 = r69 * r92;
    r90 = r91 * r92;
    r100 = r56 * r30;
    r100 = fma(r51, r100, r85);
    r8 = fma(r8, r100, r7);
    r7 = r13 * r14;
    r7 = fma(r56, r7, r93);
    r96 = r84 + r96;
    r96 = r96 + r89;
    r89 = r17 * r14;
    r89 = fma(r69, r89, r88);
    r88 = r29 * r69;
    r88 = fma(r51, r88, r98);
    r52 = r84 + r52;
    r52 = r52 + r99;
    r8 = fma(r39, r7, r8);
    r8 = fma(r40, r96, r8);
    r8 = fma(r35, r89, r8);
    r8 = fma(r32, r88, r8);
    r8 = fma(r9, r52, r8);
    r9 = r8 * r8;
    r90 = fma(r9, r90, r94 * r95);
    r95 = r6 * r6;
    r95 = r95 * r92;
    r32 = r56 * r100;
    r32 = r32 * r8;
    r90 = fma(r33, r32, r90);
    r90 = fma(r91, r95, r90);
    r90 = r41 * r90;
    r97 = 1.0 / r97;
    r97 = r44 * r97;
    r90 = r90 * r97;
    r32 = r44 * r36;
    r89 = fma(r33, r9, r6 * r94);
    r89 = fma(r41, r89, r84);
    r32 = r32 * r89;
    r32 = r32 * r94;
    r84 = fma(r91, r32, r6 * r90);
    r35 = r89 * r97;
    r84 = fma(r60, r35, r84);
    r96 = r44 * r36;
    r96 = r96 * r91;
    r96 = r96 * r8;
    r96 = r96 * r89;
    r96 = fma(r33, r96, r100 * r35);
    r96 = fma(r8, r90, r96);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx, r84, r96);
    r90 = fma(r45, r32, r78 * r35);
    r40 = r41 * r6;
    r7 = r56 * r78;
    r7 = fma(r94, r7, r45 * r95);
    r39 = r56 * r52;
    r39 = r39 * r8;
    r7 = fma(r33, r39, r7);
    r99 = r45 * r92;
    r7 = fma(r9, r99, r7);
    r40 = r40 * r7;
    r90 = fma(r97, r40, r90);
    r40 = r41 * r8;
    r40 = r40 * r7;
    r7 = r44 * r36;
    r7 = r7 * r45;
    r7 = r7 * r8;
    r7 = r7 * r89;
    r7 = fma(r33, r7, r97 * r40);
    r7 = fma(r52, r35, r7);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 2 * out_point_jac_num_alloc, global_thread_idx, r90, r7);
    r40 = r41 * r6;
    r99 = r56 * r87;
    r39 = r10 * r92;
    r39 = fma(r9, r39, r94 * r99);
    r99 = r56 * r88;
    r99 = r99 * r8;
    r39 = fma(r33, r99, r39);
    r39 = fma(r10, r95, r39);
    r40 = r40 * r39;
    r32 = fma(r10, r32, r97 * r40);
    r32 = fma(r87, r35, r32);
    r40 = r41 * r8;
    r40 = r40 * r39;
    r40 = fma(r88, r35, r97 * r40);
    r97 = r44 * r36;
    r97 = r97 * r10;
    r97 = r97 * r8;
    r97 = r97 * r89;
    r40 = fma(r33, r97, r40);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx, r32, r40);
    r97 = r36 * r84;
    r2 = fma(r2, r36, r0);
    r2 = fma(r6, r35, r2);
    r0 = r36 * r96;
    r3 = fma(r3, r36, r1);
    r3 = fma(r8, r35, r3);
    r0 = fma(r3, r0, r2 * r97);
    r97 = r36 * r7;
    r35 = r36 * r90;
    r35 = fma(r2, r35, r3 * r97);
    WriteSum2<double, double>((double *)inout_shared, r0, r35);
  };
  FlushSumShared<2, double>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r36 * r40;
    r0 = r36 * r32;
    r0 = fma(r2, r0, r3 * r35);
    WriteSum1<double, double>((double *)inout_shared, r0);
  };
  FlushSumShared<1, double>(out_point_njtr, 2 * out_point_njtr_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = fma(r96, r96, r84 * r84);
    r35 = fma(r7, r7, r90 * r90);
    WriteSum2<double, double>((double *)inout_shared, r0, r35);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fma(r32, r32, r40 * r40);
    WriteSum1<double, double>((double *)inout_shared, r35);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fma(r96, r7, r84 * r90);
    r0 = fma(r84, r32, r96 * r40);
    WriteSum2<double, double>((double *)inout_shared, r35, r0);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = fma(r90, r32, r7 * r40);
    WriteSum1<double, double>((double *)inout_shared, r0);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc, (double *)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialResJacFirst(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    double *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    double *point, unsigned int point_num_alloc, SharedIndex *point_indices,
    double *pixel, unsigned int pixel_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *const out_rTr, double *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, double *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, double *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, double *out_calib_jac,
    unsigned int out_calib_jac_num_alloc, double *const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc, double *const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    double *const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc, double *out_point_jac,
    unsigned int out_point_jac_num_alloc, double *const out_point_njtr,
    unsigned int out_point_njtr_num_alloc, double *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialResJacFirstKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, sensor_from_rig,
      sensor_from_rig_num_alloc, calib, calib_num_alloc, calib_indices, point,
      point_num_alloc, point_indices, pixel, pixel_num_alloc, out_res,
      out_res_num_alloc, out_rTr, out_pose_jac, out_pose_jac_num_alloc,
      out_pose_njtr, out_pose_njtr_num_alloc, out_pose_precond_diag,
      out_pose_precond_diag_num_alloc, out_pose_precond_tril,
      out_pose_precond_tril_num_alloc, out_calib_jac, out_calib_jac_num_alloc,
      out_calib_njtr, out_calib_njtr_num_alloc, out_calib_precond_diag,
      out_calib_precond_diag_num_alloc, out_calib_precond_tril,
      out_calib_precond_tril_num_alloc, out_point_jac, out_point_jac_num_alloc,
      out_point_njtr, out_point_njtr_num_alloc, out_point_precond_diag,
      out_point_precond_diag_num_alloc, out_point_precond_tril,
      out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar