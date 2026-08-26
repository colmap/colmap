#include "kernel_thin_prism_fisheye_split_fixed_pose_fixed_focal_and_extra_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraResJacFirstKernel(
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
        double* pose,
        unsigned int pose_num_alloc,
        double* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
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
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95;
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
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            0 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r2,
                                            r3);
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            4 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r4,
                                            r5);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r6,
                                            r7);
  };
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = 2.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            2 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r11,
                                            r12);
    ReadIdx2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r13, r14);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            0 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r15,
                                            r16);
    ReadIdx2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r17, r18);
    r19 = fma(r15, r18, r12 * r13);
    r20 = r11 * r14;
    r21 = -1.00000000000000000e+00;
    r19 = fma(r21, r20, r19);
    r19 = fma(r16, r17, r19);
    r20 = r10 * r19;
    r22 = r15 * r17;
    r22 = fma(r21, r22, r12 * r14);
    r22 = fma(r16, r18, r22);
    r22 = fma(r11, r13, r22);
    r20 = r20 * r22;
    r23 = fma(r15, r14, r12 * r17);
    r24 = r16 * r13;
    r23 = fma(r21, r24, r23);
    r23 = fma(r11, r18, r23);
    r24 = r10 * r23;
    r25 = fma(r16, r14, r15 * r13);
    r25 = fma(r11, r17, r25);
    r25 = fma(r21, r25, r12 * r18);
    r24 = fma(r25, r24, r20);
    r7 = fma(r8, r24, r7);
    ReadIdx2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r18, r26);
    r27 = r15 * r16;
    r27 = r27 * r10;
    r28 = r11 * r12;
    r28 = fma(r10, r28, r27);
    r29 = -2.00000000000000000e+00;
    r30 = r15 * r15;
    r30 = r29 * r30;
    r31 = 1.00000000000000000e+00;
    r32 = r11 * r11;
    r32 = fma(r29, r32, r31);
    r33 = r30 + r32;
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r34);
    r35 = r16 * r11;
    r35 = r35 * r10;
    r36 = r15 * r12;
    r36 = fma(r29, r36, r35);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r37);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r38 = r10 * r23;
    r38 = r38 * r22;
    r39 = r19 * r25;
    r40 = fma(r29, r39, r38);
    r41 = r23 * r23;
    r41 = r29 * r41;
    r42 = r31 + r41;
    r43 = r19 * r19;
    r43 = r43 * r29;
    r42 = r42 + r43;
    r7 = fma(r18, r28, r7);
    r7 = fma(r26, r33, r7);
    r7 = fma(r34, r36, r7);
    r7 = fma(r37, r40, r7);
    r7 = fma(r9, r42, r7);
    r36 = r7 * r7;
    r33 = 1.00000000000000008e-15;
    r28 = r22 * r22;
    r28 = r29 * r28;
    r44 = r31 + r28;
    r44 = r44 + r41;
    r6 = fma(r8, r44, r6);
    r41 = r23 * r29;
    r41 = fma(r25, r41, r20);
    r20 = r10 * r23;
    r20 = r20 * r19;
    r19 = r10 * r22;
    r19 = fma(r25, r19, r20);
    r45 = r15 * r11;
    r45 = r45 * r10;
    r46 = r16 * r12;
    r47 = fma(r10, r46, r45);
    r48 = r11 * r12;
    r48 = fma(r29, r48, r27);
    r27 = r16 * r16;
    r27 = r27 * r29;
    r32 = r27 + r32;
    r6 = fma(r9, r41, r6);
    r6 = fma(r37, r19, r6);
    r6 = fma(r34, r47, r6);
    r6 = fma(r26, r48, r6);
    r6 = fma(r18, r32, r6);
    r32 = r6 * r6;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r48);
    r47 = r29 * r22;
    r47 = fma(r25, r47, r20);
    r8 = fma(r8, r47, r48);
    r46 = fma(r29, r46, r45);
    r27 = r31 + r27;
    r27 = r27 + r30;
    r30 = r15 * r12;
    r30 = fma(r10, r30, r35);
    r39 = fma(r10, r39, r38);
    r28 = r31 + r28;
    r28 = r28 + r43;
    r8 = fma(r18, r46, r8);
    r8 = fma(r34, r27, r8);
    r8 = fma(r26, r30, r8);
    r8 = fma(r9, r39, r8);
    r8 = fma(r37, r28, r8);
    r37 = copysign(1.0, r8);
    r37 = fma(r33, r37, r8);
    r8 = r37 * r37;
    r9 = 1.0 / r8;
    r30 = r7 * r7;
    r30 = fma(r9, r30, r9 * r32);
    r32 = sqrt(r30);
    r26 = copysign(1.0, r32);
    r26 = fma(r33, r26, r32);
    r33 = r26 * r26;
    r27 = 1.0 / r33;
    r32 = atan(r32);
    r34 = r32 * r9;
    r46 = r32 * r34;
    r36 = r36 * r27;
    r36 = r36 * r46;
    r18 = 3.00000000000000000e+00;
    r43 = r18 * r46;
    r38 = r6 * r27;
    r35 = r6 * r38;
    r43 = fma(r35, r43, r36);
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            8 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r45,
                                            r48);
    r20 = r46 * r35;
    r36 = r36 + r20;
    r43 = fma(r45, r36, r5 * r43);
    r25 = r4 * r7;
    r49 = r10 * r46;
    r25 = r25 * r38;
    r43 = fma(r49, r25, r43);
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            2 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r50,
                                            r51);
    r52 = r36 * r36;
    r53 = fma(r51, r52, r50 * r36);
    ReadIdx2<1024, double, double, double2>(focal_and_extra,
                                            6 * focal_and_extra_num_alloc,
                                            global_thread_idx,
                                            r54,
                                            r55);
    r56 = r36 * r52;
    r55 = r55 * r56;
    r53 = fma(r36, r55, r53);
    r53 = fma(r54, r56, r53);
    r56 = 1.0 / r37;
    r57 = 1.0 / r26;
    r58 = r56 * r57;
    r59 = r32 * r58;
    r60 = r53 * r59;
    r43 = fma(r6, r60, r43);
    r43 = fma(r6, r59, r43);
    r43 = fma(r2, r43, r0);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r0, r25);
    r43 = fma(r0, r21, r43);
    r0 = r7 * r7;
    r0 = r0 * r18;
    r0 = r0 * r27;
    r0 = fma(r46, r0, r20);
    r0 = fma(r48, r36, r4 * r0);
    r20 = r5 * r7;
    r20 = r20 * r38;
    r0 = fma(r49, r20, r0);
    r0 = fma(r7, r60, r0);
    r0 = fma(r7, r59, r0);
    r0 = fma(r3, r0, r1);
    r0 = fma(r25, r21, r0);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r43, r0);
    r25 = fma(r0, r0, r43 * r43);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r25);
  if (global_thread_idx < problem_size) {
    r25 = r21 * r43;
    r1 = r21 * r0;
    WriteSum2<double, double>((double*)inout_shared, r25, r1);
  };
  FlushSumShared<2, double>(out_principal_point_njtr,
                            0 * out_principal_point_njtr_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r31, r31);
  };
  FlushSumShared<2, double>(out_principal_point_precond_diag,
                            0 * out_principal_point_precond_diag_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r32 * r32;
    r25 = r27 * r1;
    r20 = r7 * r7;
    r8 = r37 * r8;
    r8 = 1.0 / r8;
    r37 = r29 * r8;
    r20 = r20 * r37;
    r25 = r25 * r20;
    r61 = r21 * r7;
    r62 = r10 * r44;
    r62 = r62 * r6;
    r62 = fma(r47, r20, r9 * r62);
    r63 = r47 * r6;
    r63 = r63 * r6;
    r62 = fma(r37, r63, r62);
    r64 = r10 * r24;
    r64 = r64 * r7;
    r62 = fma(r9, r64, r62);
    r33 = r26 * r33;
    r33 = 1.0 / r33;
    r26 = r33 * r46;
    r64 = rsqrt(r30);
    r63 = r7 * r64;
    r26 = r26 * r63;
    r61 = r61 * r62;
    r61 = fma(r26, r61, r47 * r25);
    r65 = r62 * r63;
    r30 = r31 + r30;
    r30 = 1.0 / r30;
    r31 = r30 * r34;
    r66 = r7 * r27;
    r65 = r65 * r31;
    r61 = fma(r66, r65, r61);
    r67 = r49 * r66;
    r61 = fma(r24, r67, r61);
    r65 = r47 * r37;
    r1 = r35 * r1;
    r68 = r62 * r64;
    r68 = r68 * r31;
    r68 = fma(r35, r68, r1 * r65);
    r65 = r44 * r38;
    r68 = fma(r49, r65, r68);
    r69 = r21 * r6;
    r69 = r69 * r6;
    r69 = r69 * r62;
    r69 = r69 * r64;
    r69 = r69 * r33;
    r68 = fma(r46, r69, r68);
    r69 = r61 + r68;
    r65 = -6.00000000000000000e+00;
    r70 = r65 * r8;
    r70 = r70 * r1;
    r71 = r18 * r62;
    r71 = r71 * r64;
    r71 = r71 * r31;
    r71 = fma(r35, r71, r47 * r70);
    r72 = 6.00000000000000000e+00;
    r73 = r44 * r72;
    r73 = r73 * r46;
    r71 = fma(r38, r73, r71);
    r74 = r6 * r6;
    r75 = -3.00000000000000000e+00;
    r74 = r74 * r75;
    r74 = r74 * r62;
    r74 = r74 * r64;
    r74 = r74 * r33;
    r71 = fma(r46, r74, r71);
    r71 = r71 + r61;
    r71 = fma(r5, r71, r45 * r69);
    r61 = r6 * r27;
    r74 = -5.00000000000000000e-01;
    r61 = r61 * r32;
    r61 = r61 * r56;
    r61 = r61 * r74;
    r61 = r61 * r64;
    r61 = r61 * r62;
    r73 = 5.00000000000000000e-01;
    r76 = r6 * r73;
    r76 = r76 * r30;
    r76 = r76 * r64;
    r76 = r76 * r58;
    r77 = r53 * r76;
    r78 = r6 * r53;
    r79 = r21 * r47;
    r79 = r79 * r57;
    r79 = r79 * r34;
    r71 = fma(r79, r78, r71);
    r80 = r4 * r62;
    r81 = r10 * r38;
    r81 = r81 * r63;
    r81 = r81 * r31;
    r71 = fma(r81, r80, r71);
    r82 = r4 * r29;
    r82 = r82 * r6;
    r82 = r82 * r62;
    r71 = fma(r26, r82, r71);
    r83 = r4 * r44;
    r71 = fma(r67, r83, r71);
    r84 = r51 * r10;
    r84 = r84 * r36;
    r84 = fma(r50, r69, r69 * r84);
    r85 = 4.00000000000000000e+00;
    r55 = r85 * r55;
    r54 = r54 * r18;
    r54 = r54 * r52;
    r84 = fma(r69, r55, r84);
    r84 = fma(r69, r54, r84);
    r52 = r6 * r84;
    r71 = fma(r59, r52, r71);
    r85 = r4 * r7;
    r86 = -4.00000000000000000e+00;
    r85 = r85 * r32;
    r85 = r85 * r32;
    r85 = r85 * r86;
    r85 = r85 * r8;
    r85 = r85 * r38;
    r87 = r4 * r24;
    r87 = r87 * r38;
    r71 = fma(r49, r87, r71);
    r71 = r71 + r61;
    r71 = fma(r62, r77, r71);
    r71 = fma(r62, r76, r71);
    r71 = fma(r6, r79, r71);
    r71 = fma(r53, r61, r71);
    r71 = fma(r44, r59, r71);
    r71 = fma(r47, r85, r71);
    r71 = fma(r44, r60, r71);
    r87 = r2 * r71;
    r52 = r47 * r7;
    r52 = r52 * r7;
    r52 = r52 * r32;
    r52 = r52 * r32;
    r52 = r52 * r65;
    r52 = r52 * r27;
    r61 = r7 * r75;
    r61 = r61 * r62;
    r61 = fma(r26, r61, r8 * r52);
    r52 = r18 * r62;
    r52 = r52 * r63;
    r52 = r52 * r31;
    r61 = fma(r66, r52, r61);
    r83 = r24 * r7;
    r83 = r83 * r72;
    r83 = r83 * r27;
    r61 = fma(r46, r83, r61);
    r61 = r61 + r68;
    r61 = fma(r4, r61, r48 * r69);
    r69 = r7 * r84;
    r61 = fma(r59, r69, r61);
    r68 = r73 * r62;
    r68 = r68 * r30;
    r68 = r68 * r63;
    r61 = fma(r58, r68, r61);
    r83 = r5 * r62;
    r61 = fma(r81, r83, r61);
    r52 = r5 * r29;
    r52 = r52 * r6;
    r52 = r52 * r62;
    r61 = fma(r26, r52, r61);
    r82 = r5 * r67;
    r80 = r27 * r63;
    r78 = r32 * r74;
    r78 = r78 * r62;
    r78 = r78 * r56;
    r61 = fma(r78, r80, r61);
    r88 = r73 * r53;
    r88 = r88 * r62;
    r88 = r88 * r30;
    r88 = r88 * r63;
    r61 = fma(r58, r88, r61);
    r89 = r53 * r27;
    r89 = r89 * r63;
    r61 = fma(r78, r89, r61);
    r78 = r7 * r53;
    r61 = fma(r79, r78, r61);
    r90 = r5 * r47;
    r90 = r90 * r7;
    r90 = r90 * r32;
    r90 = r90 * r32;
    r90 = r90 * r86;
    r90 = r90 * r8;
    r61 = fma(r38, r90, r61);
    r91 = r5 * r24;
    r91 = r91 * r38;
    r61 = fma(r49, r91, r61);
    r61 = fma(r24, r60, r61);
    r61 = fma(r7, r79, r61);
    r61 = fma(r44, r82, r61);
    r61 = fma(r24, r59, r61);
    r91 = r3 * r61;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r87,
                                             r91);
    r91 = r41 * r72;
    r91 = r91 * r46;
    r91 = fma(r39, r70, r38 * r91);
    r87 = r6 * r6;
    r90 = r39 * r6;
    r90 = r90 * r6;
    r78 = r10 * r41;
    r78 = r78 * r6;
    r78 = fma(r9, r78, r37 * r90);
    r90 = r10 * r42;
    r90 = r90 * r7;
    r78 = fma(r9, r90, r78);
    r78 = fma(r39, r20, r78);
    r90 = r75 * r78;
    r87 = r87 * r64;
    r87 = r87 * r33;
    r87 = r87 * r46;
    r91 = fma(r90, r87, r91);
    r89 = r18 * r78;
    r89 = r89 * r64;
    r89 = r89 * r31;
    r91 = fma(r35, r89, r91);
    r88 = r78 * r63;
    r88 = r88 * r31;
    r88 = fma(r66, r88, r39 * r25);
    r80 = r21 * r7;
    r80 = r80 * r78;
    r88 = fma(r26, r80, r88);
    r88 = fma(r42, r67, r88);
    r91 = r91 + r88;
    r89 = r41 * r38;
    r87 = r39 * r37;
    r87 = fma(r1, r87, r49 * r89);
    r89 = r21 * r6;
    r89 = r89 * r6;
    r89 = r89 * r78;
    r89 = r89 * r64;
    r89 = r89 * r33;
    r87 = fma(r46, r89, r87);
    r80 = r78 * r64;
    r80 = r80 * r31;
    r87 = fma(r35, r80, r87);
    r88 = r88 + r87;
    r91 = fma(r45, r88, r5 * r91);
    r80 = r21 * r39;
    r80 = r80 * r6;
    r80 = r80 * r53;
    r80 = r80 * r57;
    r91 = fma(r34, r80, r91);
    r89 = r51 * r10;
    r89 = r89 * r36;
    r89 = fma(r88, r89, r50 * r88);
    r89 = fma(r88, r54, r89);
    r89 = fma(r88, r55, r89);
    r52 = r6 * r89;
    r91 = fma(r59, r52, r91);
    r83 = r21 * r39;
    r83 = r83 * r6;
    r83 = r83 * r57;
    r91 = fma(r34, r83, r91);
    r79 = r4 * r42;
    r79 = r79 * r38;
    r91 = fma(r49, r79, r91);
    r68 = r4 * r41;
    r91 = fma(r67, r68, r91);
    r69 = r32 * r74;
    r69 = r69 * r78;
    r69 = r69 * r56;
    r69 = r69 * r64;
    r91 = fma(r38, r69, r91);
    r92 = r4 * r78;
    r91 = fma(r81, r92, r91);
    r93 = r4 * r29;
    r93 = r93 * r6;
    r93 = r93 * r78;
    r91 = fma(r26, r93, r91);
    r94 = r32 * r53;
    r94 = r94 * r74;
    r94 = r94 * r78;
    r94 = r94 * r56;
    r94 = r94 * r64;
    r91 = fma(r38, r94, r91);
    r91 = fma(r41, r60, r91);
    r91 = fma(r78, r76, r91);
    r91 = fma(r78, r77, r91);
    r91 = fma(r41, r59, r91);
    r91 = fma(r39, r85, r91);
    r94 = r2 * r91;
    r93 = r39 * r7;
    r93 = r93 * r7;
    r93 = r93 * r32;
    r93 = r93 * r32;
    r93 = r93 * r65;
    r93 = r93 * r27;
    r92 = r18 * r78;
    r92 = r92 * r63;
    r92 = r92 * r31;
    r92 = fma(r66, r92, r8 * r93);
    r93 = r42 * r7;
    r93 = r93 * r72;
    r93 = r93 * r27;
    r92 = fma(r46, r93, r92);
    r69 = r7 * r26;
    r92 = fma(r90, r69, r92);
    r92 = r92 + r87;
    r88 = fma(r48, r88, r4 * r92);
    r92 = r21 * r39;
    r92 = r92 * r7;
    r92 = r92 * r57;
    r88 = fma(r34, r92, r88);
    r87 = r21 * r39;
    r87 = r87 * r7;
    r87 = r87 * r53;
    r87 = r87 * r57;
    r88 = fma(r34, r87, r88);
    r69 = r73 * r78;
    r69 = r69 * r30;
    r69 = r69 * r63;
    r88 = fma(r58, r69, r88);
    r93 = r5 * r42;
    r93 = r93 * r38;
    r88 = fma(r49, r93, r88);
    r90 = r32 * r74;
    r90 = r90 * r78;
    r90 = r90 * r27;
    r90 = r90 * r56;
    r88 = fma(r63, r90, r88);
    r68 = r5 * r78;
    r88 = fma(r81, r68, r88);
    r79 = r5 * r29;
    r79 = r79 * r6;
    r79 = r79 * r78;
    r88 = fma(r26, r79, r88);
    r83 = r5 * r39;
    r83 = r83 * r7;
    r83 = r83 * r32;
    r83 = r83 * r32;
    r83 = r83 * r86;
    r83 = r83 * r8;
    r88 = fma(r38, r83, r88);
    r52 = r73 * r53;
    r52 = r52 * r78;
    r52 = r52 * r30;
    r52 = r52 * r63;
    r88 = fma(r58, r52, r88);
    r80 = r7 * r89;
    r88 = fma(r59, r80, r88);
    r95 = r32 * r53;
    r95 = r95 * r74;
    r95 = r95 * r78;
    r95 = r95 * r27;
    r95 = r95 * r56;
    r88 = fma(r63, r95, r88);
    r88 = fma(r42, r60, r88);
    r88 = fma(r41, r82, r88);
    r88 = fma(r42, r59, r88);
    r95 = r3 * r88;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r94,
                                             r95);
    r95 = r6 * r6;
    r94 = r10 * r19;
    r94 = r94 * r6;
    r20 = fma(r28, r20, r9 * r94);
    r94 = r28 * r6;
    r94 = r94 * r6;
    r20 = fma(r37, r94, r20);
    r80 = r10 * r40;
    r80 = r80 * r7;
    r20 = fma(r9, r80, r20);
    r95 = r95 * r75;
    r95 = r95 * r20;
    r95 = r95 * r64;
    r95 = r95 * r33;
    r95 = fma(r46, r95, r28 * r70);
    r70 = r18 * r20;
    r70 = r70 * r64;
    r70 = r70 * r31;
    r95 = fma(r35, r70, r95);
    r80 = r19 * r72;
    r80 = r80 * r46;
    r95 = fma(r38, r80, r95);
    r94 = r21 * r7;
    r9 = r20 * r26;
    r94 = fma(r9, r94, r40 * r67);
    r52 = r20 * r63;
    r52 = r52 * r31;
    r94 = fma(r66, r52, r94);
    r94 = fma(r28, r25, r94);
    r95 = r95 + r94;
    r80 = r28 * r37;
    r70 = r21 * r6;
    r70 = r70 * r6;
    r70 = r70 * r20;
    r70 = r70 * r64;
    r70 = r70 * r33;
    r70 = fma(r46, r70, r1 * r80);
    r80 = r20 * r64;
    r80 = r80 * r31;
    r70 = fma(r35, r80, r70);
    r35 = r19 * r38;
    r70 = fma(r49, r35, r70);
    r94 = r94 + r70;
    r45 = fma(r45, r94, r5 * r95);
    r95 = r21 * r28;
    r95 = r95 * r6;
    r95 = r95 * r57;
    r45 = fma(r34, r95, r45);
    r35 = r4 * r40;
    r35 = r35 * r38;
    r45 = fma(r49, r35, r45);
    r80 = r29 * r6;
    r80 = r80 * r9;
    r1 = r32 * r74;
    r1 = r1 * r20;
    r1 = r1 * r56;
    r1 = r1 * r64;
    r45 = fma(r38, r1, r45);
    r33 = r4 * r20;
    r45 = fma(r81, r33, r45);
    r25 = r4 * r19;
    r45 = fma(r67, r25, r45);
    r67 = r32 * r53;
    r67 = r67 * r74;
    r67 = r67 * r20;
    r67 = r67 * r56;
    r67 = r67 * r64;
    r45 = fma(r38, r67, r45);
    r52 = r21 * r28;
    r52 = r52 * r6;
    r52 = r52 * r53;
    r52 = r52 * r57;
    r45 = fma(r34, r52, r45);
    r83 = r51 * r10;
    r83 = r83 * r36;
    r83 = fma(r94, r83, r50 * r94);
    r83 = fma(r94, r55, r83);
    r83 = fma(r94, r54, r83);
    r54 = r6 * r83;
    r45 = fma(r59, r54, r45);
    r45 = fma(r19, r60, r45);
    r45 = fma(r4, r80, r45);
    r45 = fma(r20, r76, r45);
    r45 = fma(r20, r77, r45);
    r45 = fma(r28, r85, r45);
    r45 = fma(r19, r59, r45);
    r54 = r2 * r45;
    r52 = r40 * r7;
    r52 = r52 * r72;
    r52 = r52 * r27;
    r67 = r7 * r75;
    r67 = fma(r9, r67, r46 * r52);
    r52 = r18 * r20;
    r52 = r52 * r63;
    r52 = r52 * r31;
    r67 = fma(r66, r52, r67);
    r66 = r28 * r7;
    r66 = r66 * r7;
    r66 = r66 * r32;
    r66 = r66 * r32;
    r66 = r66 * r65;
    r66 = r66 * r27;
    r67 = fma(r8, r66, r67);
    r67 = r67 + r70;
    r67 = fma(r4, r67, r48 * r94);
    r94 = r5 * r40;
    r94 = r94 * r38;
    r67 = fma(r49, r94, r67);
    r49 = r32 * r74;
    r49 = r49 * r20;
    r49 = r49 * r27;
    r49 = r49 * r56;
    r67 = fma(r63, r49, r67);
    r48 = r5 * r20;
    r67 = fma(r81, r48, r67);
    r81 = r32 * r53;
    r81 = r81 * r74;
    r81 = r81 * r20;
    r81 = r81 * r27;
    r81 = r81 * r56;
    r67 = fma(r63, r81, r67);
    r56 = r73 * r53;
    r56 = r56 * r20;
    r56 = r56 * r30;
    r56 = r56 * r63;
    r67 = fma(r58, r56, r67);
    r70 = r21 * r28;
    r70 = r70 * r7;
    r70 = r70 * r53;
    r70 = r70 * r57;
    r67 = fma(r34, r70, r67);
    r66 = r5 * r28;
    r66 = r66 * r7;
    r66 = r66 * r32;
    r66 = r66 * r32;
    r66 = r66 * r86;
    r66 = r66 * r8;
    r67 = fma(r38, r66, r67);
    r8 = r7 * r83;
    r67 = fma(r59, r8, r67);
    r86 = r73 * r20;
    r86 = r86 * r30;
    r86 = r86 * r63;
    r67 = fma(r58, r86, r67);
    r58 = r21 * r28;
    r58 = r58 * r7;
    r58 = r58 * r57;
    r67 = fma(r34, r58, r67);
    r67 = fma(r5, r80, r67);
    r67 = fma(r40, r60, r67);
    r67 = fma(r19, r82, r67);
    r67 = fma(r40, r59, r67);
    r58 = r3 * r67;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r54,
                                             r58);
    r58 = r3 * r21;
    r58 = r58 * r0;
    r54 = r2 * r21;
    r54 = r54 * r43;
    r54 = fma(r71, r54, r61 * r58);
    r58 = r2 * r21;
    r58 = r58 * r43;
    r86 = r3 * r21;
    r86 = r86 * r0;
    r86 = fma(r88, r86, r91 * r58);
    WriteSum2<double, double>((double*)inout_shared, r54, r86);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r86 = r3 * r21;
    r86 = r86 * r0;
    r0 = r2 * r21;
    r0 = r0 * r43;
    r0 = fma(r45, r0, r67 * r86);
    WriteSum1<double, double>((double*)inout_shared, r0);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r61 * r61;
    r86 = r3 * r3;
    r43 = r2 * r2;
    r54 = r71 * r43;
    r71 = fma(r71, r54, r86 * r0);
    r0 = r88 * r86;
    r58 = r91 * r91;
    r58 = fma(r43, r58, r88 * r0);
    WriteSum2<double, double>((double*)inout_shared, r71, r58);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = r67 * r67;
    r71 = r45 * r45;
    r71 = fma(r43, r71, r86 * r58);
    WriteSum1<double, double>((double*)inout_shared, r71);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = fma(r91, r54, r61 * r0);
    r58 = r61 * r67;
    r54 = fma(r45, r54, r86 * r58);
    WriteSum2<double, double>((double*)inout_shared, r71, r54);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = r91 * r45;
    r54 = fma(r43, r54, r67 * r0);
    WriteSum1<double, double>((double*)inout_shared, r54);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraResJacFirst(
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
    double* pose,
    unsigned int pose_num_alloc,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
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
  ThinPrismFisheyeSplitFixedPoseFixedFocalAndExtraResJacFirstKernel<<<n_blocks,
                                                                      1024>>>(
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
      pose,
      pose_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
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