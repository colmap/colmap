#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedFocalFixedPrincipalPointFixedPointResJacFirstKernel(
        double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
        double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        double *pixel, unsigned int pixel_num_alloc, double *focal,
        unsigned int focal_num_alloc, double *principal_point,
        unsigned int principal_point_num_alloc, double *point,
        unsigned int point_num_alloc, double *out_res,
        unsigned int out_res_num_alloc, double *const out_rTr,
        double *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
        double *const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double *const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
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
         r73 = 0, r74 = 0, r75 = 0, r76 = 0, r77 = 0, r78 = 0, r79 = 0, r80 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(principal_point,
                                            0 * principal_point_num_alloc,
                                            global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(pixel, 0 * pixel_num_alloc,
                                            global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r5 = fma(r2, r4, r0);
    ReadIdx2<1024, double, double, double2>(focal, 0 * focal_num_alloc,
                                            global_thread_idx, r6, r7);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r8, r9);
    ReadIdx2<1024, double, double, double2>(point, 0 * point_num_alloc,
                                            global_thread_idx, r10, r11);
    r12 = -2.00000000000000000e+00;
  };
  LoadShared<2, double, double>(pose, 2 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r13, r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r15, r16);
  };
  LoadShared<2, double, double>(pose, 0 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r17, r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx, r19, r20);
    r21 = fma(r18, r19, r13 * r16);
    r22 = r17 * r20;
    r21 = fma(r4, r22, r21);
    r21 = fma(r14, r15, r21);
    r22 = r21 * r21;
    r22 = r12 * r22;
    r23 = 1.00000000000000000e+00;
    r24 = r13 * r19;
    r24 = fma(r4, r24, r18 * r16);
    r24 = fma(r14, r20, r24);
    r24 = fma(r17, r15, r24);
    r25 = r24 * r24;
    r25 = fma(r12, r25, r23);
    r26 = r22 + r25;
    r26 = fma(r10, r26, r8);
    r27 = 2.00000000000000000e+00;
    r28 = fma(r14, r19, r17 * r16);
    r29 = r18 * r15;
    r28 = fma(r4, r29, r28);
    r28 = fma(r13, r20, r28);
    r29 = r27 * r28;
    r30 = r24 * r29;
    r31 = fma(r18, r20, r17 * r19);
    r31 = fma(r13, r15, r31);
    r31 = fma(r4, r31, r14 * r16);
    r32 = r12 * r31;
    r33 = fma(r21, r32, r30);
    ReadIdx1<1024, double, double, double>(point, 2 * point_num_alloc,
                                           global_thread_idx, r34);
    r35 = r27 * r24;
    r36 = r21 * r29;
    r35 = fma(r31, r35, r36);
  };
  LoadShared<1, double, double>(pose, 6 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r37);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r38 = r19 * r15;
    r38 = r38 * r27;
    r39 = r20 * r16;
    r40 = fma(r27, r39, r38);
  };
  LoadShared<2, double, double>(pose, 4 * pose_num_alloc, pose_indices_loc,
                                (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double *)inout_shared,
                        pose_indices_loc[threadIdx.x].target, r41, r42);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r43 = r15 * r16;
    r44 = r19 * r20;
    r44 = r44 * r27;
    r43 = fma(r12, r43, r44);
    r45 = r20 * r20;
    r45 = r45 * r12;
    r46 = r23 + r45;
    r47 = r15 * r15;
    r47 = r12 * r47;
    r46 = r46 + r47;
    r26 = fma(r11, r33, r26);
    r26 = fma(r34, r35, r26);
    r26 = fma(r37, r40, r26);
    r26 = fma(r42, r43, r26);
    r26 = fma(r41, r46, r26);
    r46 = r6 * r26;
    r43 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r40);
    r36 = fma(r24, r32, r36);
    r36 = fma(r10, r36, r40);
    r39 = fma(r12, r39, r38);
    r45 = r23 + r45;
    r38 = r19 * r19;
    r38 = r12 * r38;
    r45 = r45 + r38;
    r35 = r20 * r15;
    r35 = r35 * r27;
    r33 = r19 * r16;
    r33 = fma(r27, r33, r35);
    r48 = r27 * r21;
    r48 = r48 * r24;
    r29 = fma(r31, r29, r48);
    r49 = r28 * r28;
    r49 = r49 * r12;
    r25 = r49 + r25;
    r36 = fma(r41, r39, r36);
    r36 = fma(r37, r45, r36);
    r36 = fma(r42, r33, r36);
    r36 = fma(r11, r29, r36);
    r36 = fma(r34, r25, r36);
    r25 = copysign(1.0, r36);
    r25 = fma(r43, r25, r36);
    r25 = 1.0 / r25;
    r5 = fma(r25, r46, r5);
    r4 = fma(r3, r4, r1);
    r46 = r27 * r21;
    r46 = fma(r31, r46, r30);
    r46 = fma(r10, r46, r9);
    r30 = r15 * r16;
    r30 = fma(r27, r30, r44);
    r47 = r23 + r47;
    r47 = r47 + r38;
    r38 = r19 * r16;
    r38 = fma(r12, r38, r35);
    r32 = fma(r28, r32, r48);
    r22 = r23 + r22;
    r22 = r22 + r49;
    r46 = fma(r41, r30, r46);
    r46 = fma(r42, r47, r46);
    r46 = fma(r37, r38, r46);
    r46 = fma(r34, r32, r46);
    r46 = fma(r11, r22, r46);
    r22 = r7 * r46;
    r4 = fma(r25, r22, r4);
    WriteIdx2<1024, double, double, double2>(out_res, 0 * out_res_num_alloc,
                                             global_thread_idx, r5, r4);
    r4 = fma(r4, r4, r5 * r5);
  };
  SumStore<double>(out_rTr_local, (double *)inout_shared, 0,
                   global_thread_idx < problem_size, r4);
  if (global_thread_idx < problem_size) {
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
    r0 = 1.00000000000000008e-15;
    r5 = fma(r18, r19, r13 * r16);
    r22 = r17 * r20;
    r5 = fma(r4, r22, r5);
    r5 = fma(r14, r15, r5);
    r22 = 2.00000000000000000e+00;
    r25 = fma(r14, r19, r17 * r16);
    r32 = r18 * r15;
    r25 = fma(r4, r32, r25);
    r25 = fma(r13, r20, r25);
    r32 = r22 * r25;
    r38 = r5 * r32;
    r47 = -2.00000000000000000e+00;
    r30 = fma(r18, r20, r17 * r19);
    r30 = fma(r13, r15, r30);
    r30 = fma(r4, r30, r14 * r16);
    r49 = r47 * r30;
    r23 = r14 * r20;
    r28 = fma(r18, r16, r23);
    r48 = r17 * r15;
    r35 = r13 * r19;
    r28 = r28 + r48;
    r28 = fma(r4, r35, r28);
    r12 = fma(r28, r49, r38);
    r12 = fma(r10, r12, r40);
    r40 = r19 * r15;
    r40 = r40 * r22;
    r44 = r20 * r16;
    r44 = fma(r47, r44, r40);
    r31 = r19 * r19;
    r31 = r31 * r47;
    r43 = 1.00000000000000000e+00;
    r36 = r20 * r20;
    r36 = fma(r47, r36, r43);
    r29 = r31 + r36;
    r33 = r20 * r15;
    r33 = r33 * r22;
    r45 = r19 * r16;
    r45 = fma(r22, r45, r33);
    r39 = r22 * r5;
    r39 = r39 * r28;
    r50 = fma(r30, r32, r39);
    r51 = r47 * r28;
    r51 = r51 * r28;
    r52 = r43 + r51;
    r53 = r25 * r25;
    r53 = r53 * r47;
    r52 = r52 + r53;
    r12 = fma(r41, r44, r12);
    r12 = fma(r37, r29, r12);
    r12 = fma(r42, r45, r12);
    r12 = fma(r11, r50, r12);
    r12 = fma(r34, r52, r12);
    r52 = copysign(1.0, r12);
    r52 = fma(r0, r52, r12);
    r0 = 1.0 / r52;
    r51 = r43 + r51;
    r12 = r5 * r5;
    r12 = r12 * r47;
    r51 = r51 + r12;
    r51 = fma(r10, r51, r8);
    r8 = r28 * r32;
    r50 = fma(r5, r49, r8);
    r54 = r22 * r28;
    r54 = fma(r30, r54, r38);
    r38 = r20 * r16;
    r38 = fma(r22, r38, r40);
    r40 = r15 * r16;
    r55 = r19 * r20;
    r55 = r55 * r22;
    r40 = fma(r47, r40, r55);
    r56 = r15 * r15;
    r56 = r56 * r47;
    r36 = r56 + r36;
    r51 = fma(r11, r50, r51);
    r51 = fma(r34, r54, r51);
    r51 = fma(r37, r38, r51);
    r51 = fma(r42, r40, r51);
    r51 = fma(r41, r36, r51);
    r51 = r6 * r51;
    r2 = fma(r0, r51, r2);
    r54 = r4 * r2;
    r50 = r22 * r28;
    r57 = -5.00000000000000000e-01;
    r58 = r18 * r57;
    r59 = 5.00000000000000000e-01;
    r60 = fma(r59, r35, r16 * r58);
    r60 = fma(r57, r23, r60);
    r60 = fma(r57, r48, r60);
    r50 = r50 * r60;
    r61 = r22 * r5;
    r62 = r18 * r19;
    r63 = r17 * r20;
    r63 = fma(r57, r63, r59 * r62);
    r62 = r14 * r15;
    r63 = fma(r59, r62, r63);
    r64 = r16 * r59;
    r63 = fma(r13, r64, r63);
    r61 = fma(r63, r61, r50);
    r62 = r22 * r30;
    r65 = r17 * r19;
    r66 = r13 * r15;
    r66 = fma(r57, r66, r57 * r65);
    r66 = fma(r14, r64, r66);
    r66 = fma(r20, r58, r66);
    r62 = r62 * r66;
    r65 = r17 * r16;
    r67 = r14 * r19;
    r67 = fma(r57, r67, r57 * r65);
    r65 = r13 * r20;
    r67 = fma(r57, r65, r67);
    r68 = r18 * r15;
    r67 = fma(r59, r68, r67);
    r68 = r67 * r32;
    r65 = r62 + r68;
    r69 = r61 + r65;
    r70 = r28 * r67;
    r71 = fma(r63, r49, r47 * r70);
    r72 = r22 * r5;
    r72 = r72 * r66;
    r73 = fma(r60, r32, r72);
    r71 = r71 + r73;
    r71 = fma(r10, r71, r11 * r69);
    r69 = r28 * r63;
    r74 = -4.00000000000000000e+00;
    r69 = r69 * r74;
    r75 = r66 * r74;
    r76 = r25 * r75;
    r77 = r69 + r76;
    r71 = fma(r34, r77, r71);
    r52 = r52 * r52;
    r52 = 1.0 / r52;
    r52 = r4 * r52;
    r51 = r52 * r51;
    r77 = r22 * r30;
    r77 = fma(r22, r70, r63 * r77);
    r77 = r77 + r73;
    r78 = r22 * r28;
    r78 = r78 * r66;
    r79 = r5 * r47;
    r79 = fma(r67, r79, r78);
    r63 = r63 * r32;
    r79 = r79 + r63;
    r79 = fma(r60, r49, r79);
    r79 = fma(r11, r79, r34 * r77);
    r77 = r5 * r60;
    r80 = r74 * r77;
    r69 = r69 + r80;
    r79 = fma(r10, r69, r79);
    r69 = r6 * r79;
    r69 = fma(r0, r69, r71 * r51);
    r3 = fma(r3, r4, r1);
    r1 = r22 * r5;
    r1 = fma(r30, r1, r8);
    r1 = fma(r10, r1, r9);
    r9 = r15 * r16;
    r9 = fma(r22, r9, r55);
    r56 = r43 + r56;
    r56 = r56 + r31;
    r31 = r19 * r16;
    r31 = fma(r47, r31, r33);
    r39 = fma(r25, r49, r39);
    r12 = r43 + r12;
    r12 = r12 + r53;
    r1 = fma(r41, r9, r1);
    r1 = fma(r42, r56, r1);
    r1 = fma(r37, r31, r1);
    r1 = fma(r34, r39, r1);
    r1 = fma(r11, r12, r1);
    r1 = r7 * r1;
    r3 = fma(r0, r1, r3);
    r12 = r4 * r3;
    r39 = r25 * r47;
    r37 = r66 * r49;
    r39 = fma(r67, r39, r37);
    r39 = r39 + r61;
    r80 = r76 + r80;
    r80 = fma(r11, r80, r34 * r39);
    r39 = r22 * r30;
    r39 = fma(r60, r39, r63);
    r63 = r22 * r5;
    r63 = fma(r67, r63, r78);
    r39 = r39 + r63;
    r80 = fma(r10, r39, r80);
    r39 = r7 * r80;
    r78 = r71 * r52;
    r78 = fma(r1, r78, r0 * r39);
    r12 = fma(r78, r12, r69 * r54);
    r54 = r4 * r3;
    r39 = r47 * r28;
    r39 = fma(r60, r39, r37);
    r76 = r22 * r5;
    r61 = r13 * r16;
    r42 = r17 * r20;
    r42 = fma(r59, r42, r57 * r61);
    r61 = r14 * r15;
    r42 = fma(r57, r61, r42);
    r42 = fma(r19, r58, r42);
    r76 = r76 * r42;
    r61 = r14 * r19;
    r41 = r13 * r20;
    r41 = fma(r59, r41, r59 * r61);
    r41 = fma(r17, r64, r41);
    r41 = fma(r15, r58, r41);
    r58 = fma(r41, r32, r76);
    r39 = r39 + r58;
    r61 = r22 * r28;
    r61 = r61 * r41;
    r53 = r22 * r30;
    r53 = fma(r42, r53, r61);
    r53 = r53 + r73;
    r53 = fma(r11, r53, r10 * r39);
    r39 = r25 * r74;
    r39 = r39 * r42;
    r73 = r28 * r75;
    r43 = r39 + r73;
    r53 = fma(r34, r43, r53);
    r43 = r53 * r52;
    r61 = r72 + r61;
    r72 = r25 * r47;
    r61 = fma(r60, r72, r61);
    r61 = fma(r42, r49, r61);
    r72 = r22 * r30;
    r72 = fma(r22, r77, r41 * r72);
    r60 = r22 * r28;
    r66 = r66 * r32;
    r60 = fma(r42, r60, r66);
    r72 = r72 + r60;
    r72 = fma(r10, r72, r34 * r61);
    r61 = r5 * r74;
    r61 = r61 * r41;
    r39 = r39 + r61;
    r72 = fma(r11, r39, r72);
    r39 = r7 * r72;
    r39 = fma(r0, r39, r1 * r43);
    r43 = r4 * r2;
    r62 = r50 + r62;
    r62 = r62 + r58;
    r73 = r61 + r73;
    r73 = fma(r10, r73, r34 * r62);
    r41 = fma(r41, r49, r47 * r77);
    r41 = r41 + r60;
    r73 = fma(r11, r41, r73);
    r41 = r6 * r73;
    r41 = fma(r0, r41, r53 * r51);
    r43 = fma(r41, r43, r39 * r54);
    WriteSum2<double, double>((double *)inout_shared, r12, r43);
  };
  FlushSumShared<2, double>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r4 * r3;
    r12 = r25 * r74;
    r35 = fma(r57, r35, r18 * r64);
    r35 = fma(r59, r23, r35);
    r35 = fma(r59, r48, r35);
    r12 = r12 * r35;
    r70 = r74 * r70;
    r74 = r12 + r70;
    r48 = r22 * r5;
    r48 = r48 * r35;
    r59 = r47 * r28;
    r59 = fma(r42, r59, r48);
    r59 = r59 + r66;
    r59 = fma(r67, r49, r59);
    r59 = fma(r10, r59, r34 * r74);
    r74 = r22 * r30;
    r32 = fma(r42, r32, r35 * r74);
    r32 = r32 + r63;
    r59 = fma(r11, r32, r59);
    r32 = r59 * r52;
    r74 = r22 * r28;
    r74 = r74 * r35;
    r76 = r76 + r74;
    r76 = r76 + r65;
    r65 = r25 * r47;
    r49 = fma(r35, r49, r42 * r65);
    r49 = r49 + r63;
    r49 = fma(r34, r49, r10 * r76);
    r75 = r5 * r75;
    r12 = r12 + r75;
    r49 = fma(r11, r12, r49);
    r12 = r7 * r49;
    r12 = fma(r0, r12, r1 * r32);
    r32 = r4 * r2;
    r76 = r5 * r47;
    r76 = fma(r42, r76, r74);
    r76 = r76 + r68;
    r76 = r76 + r37;
    r75 = r70 + r75;
    r75 = fma(r10, r75, r11 * r76);
    r10 = r22 * r30;
    r10 = fma(r67, r10, r48);
    r10 = r10 + r60;
    r75 = fma(r34, r10, r75);
    r10 = r6 * r75;
    r10 = fma(r0, r10, r59 * r51);
    r32 = fma(r10, r32, r12 * r43);
    r43 = r4 * r2;
    r34 = r6 * r36;
    r34 = fma(r44, r51, r0 * r34);
    r60 = r4 * r3;
    r48 = r44 * r52;
    r67 = r7 * r9;
    r67 = fma(r0, r67, r1 * r48);
    r60 = fma(r67, r60, r34 * r43);
    WriteSum2<double, double>((double *)inout_shared, r32, r60);
  };
  FlushSumShared<2, double>(out_pose_njtr, 2 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = r4 * r3;
    r32 = r45 * r52;
    r43 = r7 * r56;
    r43 = fma(r0, r43, r1 * r32);
    r32 = r4 * r2;
    r48 = r6 * r40;
    r48 = fma(r0, r48, r45 * r51);
    r32 = fma(r48, r32, r43 * r60);
    r60 = r4 * r2;
    r76 = r6 * r38;
    r76 = fma(r0, r76, r29 * r51);
    r51 = r4 * r3;
    r11 = r7 * r31;
    r70 = r29 * r52;
    r70 = fma(r1, r70, r0 * r11);
    r51 = fma(r70, r51, r76 * r60);
    WriteSum2<double, double>((double *)inout_shared, r32, r51);
  };
  FlushSumShared<2, double>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = fma(r69, r69, r78 * r78);
    r32 = fma(r39, r39, r41 * r41);
    WriteSum2<double, double>((double *)inout_shared, r51, r32);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r10, r10, r12 * r12);
    r51 = fma(r34, r34, r67 * r67);
    WriteSum2<double, double>((double *)inout_shared, r32, r51);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = fma(r43, r43, r48 * r48);
    r32 = fma(r76, r76, r70 * r70);
    WriteSum2<double, double>((double *)inout_shared, r51, r32);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r69, r41, r78 * r39);
    r51 = fma(r78, r12, r69 * r10);
    WriteSum2<double, double>((double *)inout_shared, r32, r51);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = fma(r78, r67, r69 * r34);
    r32 = fma(r78, r43, r69 * r48);
    WriteSum2<double, double>((double *)inout_shared, r51, r32);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = fma(r69, r76, r78 * r70);
    r78 = fma(r39, r12, r41 * r10);
    WriteSum2<double, double>((double *)inout_shared, r69, r78);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = fma(r41, r34, r39 * r67);
    r69 = fma(r39, r43, r41 * r48);
    WriteSum2<double, double>((double *)inout_shared, r78, r69);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r41, r76, r39 * r70);
    r39 = fma(r10, r34, r12 * r67);
    WriteSum2<double, double>((double *)inout_shared, r41, r39);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r10, r48, r12 * r43);
    r10 = fma(r10, r76, r12 * r70);
    WriteSum2<double, double>((double *)inout_shared, r39, r10);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fma(r67, r43, r34 * r48);
    r34 = fma(r34, r76, r67 * r70);
    WriteSum2<double, double>((double *)inout_shared, r10, r34);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r43, r70, r48 * r76);
    WriteSum1<double, double>((double *)inout_shared, r70);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc, (double *)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedFocalFixedPrincipalPointFixedPointResJacFirst(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    double *pixel, unsigned int pixel_num_alloc, double *focal,
    unsigned int focal_num_alloc, double *principal_point,
    unsigned int principal_point_num_alloc, double *point,
    unsigned int point_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *const out_rTr,
    double *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
    double *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedFocalFixedPrincipalPointFixedPointResJacFirstKernel<<<
      n_blocks, 1024>>>(pose, pose_num_alloc, pose_indices, sensor_from_rig,
                        sensor_from_rig_num_alloc, pixel, pixel_num_alloc,
                        focal, focal_num_alloc, principal_point,
                        principal_point_num_alloc, point, point_num_alloc,
                        out_res, out_res_num_alloc, out_rTr, out_pose_njtr,
                        out_pose_njtr_num_alloc, out_pose_precond_diag,
                        out_pose_precond_diag_num_alloc, out_pose_precond_tril,
                        out_pose_precond_tril_num_alloc, problem_size);
}

} // namespace caspar