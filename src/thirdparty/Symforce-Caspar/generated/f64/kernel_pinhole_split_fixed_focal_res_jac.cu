#include "kernel_pinhole_split_fixed_focal_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) PinholeSplitFixedFocalResJacKernel(
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
    double* focal,
    unsigned int focal_num_alloc,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80;
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
    r0 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        sensor_from_rig, 6 * sensor_from_rig_num_alloc, global_thread_idx, r5);
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
    r17 = 2.00000000000000000e+00;
    r18 = fma(r9, r14, r12 * r11);
    r19 = r13 * r10;
    r18 = fma(r4, r19, r18);
    r18 = fma(r8, r15, r18);
    r19 = r17 * r18;
    r20 = r16 * r19;
    r21 = -2.00000000000000000e+00;
    r22 = fma(r13, r15, r12 * r14);
    r22 = fma(r8, r10, r22);
    r22 = fma(r4, r22, r9 * r11);
    r23 = r21 * r22;
    r24 = r9 * r15;
    r25 = fma(r13, r11, r24);
    r26 = r12 * r10;
    r27 = r8 * r14;
    r25 = r25 + r26;
    r25 = fma(r4, r27, r25);
    r28 = fma(r25, r23, r20);
    r5 = fma(r6, r28, r5);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r29, r30);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r31 = r14 * r10;
    r31 = r31 * r17;
    r32 = r15 * r11;
    r32 = fma(r21, r32, r31);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r33);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r34 = r14 * r14;
    r34 = r34 * r21;
    r35 = 1.00000000000000000e+00;
    r36 = r15 * r15;
    r36 = fma(r21, r36, r35);
    r37 = r34 + r36;
    r38 = r15 * r10;
    r38 = r38 * r17;
    r39 = r14 * r11;
    r39 = fma(r17, r39, r38);
    r40 = r17 * r16;
    r40 = r40 * r25;
    r41 = fma(r22, r19, r40);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r42);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r43 = r21 * r25;
    r43 = r43 * r25;
    r44 = r35 + r43;
    r45 = r18 * r18;
    r45 = r45 * r21;
    r44 = r44 + r45;
    r5 = fma(r29, r32, r5);
    r5 = fma(r33, r37, r5);
    r5 = fma(r30, r39, r5);
    r5 = fma(r7, r41, r5);
    r5 = fma(r42, r44, r5);
    r46 = copysign(1.0, r5);
    r46 = fma(r0, r46, r5);
    r0 = 1.0 / r46;
    ReadIdx2<1024, double, double, double2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r5, r47);
    ReadIdx2<1024, double, double, double2>(sensor_from_rig,
                                            4 * sensor_from_rig_num_alloc,
                                            global_thread_idx,
                                            r48,
                                            r49);
    r43 = r35 + r43;
    r50 = r16 * r16;
    r50 = r50 * r21;
    r43 = r43 + r50;
    r48 = fma(r6, r43, r48);
    r51 = r25 * r19;
    r52 = fma(r16, r23, r51);
    r53 = r17 * r25;
    r53 = fma(r22, r53, r20);
    r20 = r15 * r11;
    r20 = fma(r17, r20, r31);
    r31 = r10 * r11;
    r54 = r14 * r15;
    r54 = r54 * r17;
    r31 = fma(r21, r31, r54);
    r55 = r10 * r10;
    r55 = r55 * r21;
    r36 = r55 + r36;
    r48 = fma(r7, r52, r48);
    r48 = fma(r42, r53, r48);
    r48 = fma(r33, r20, r48);
    r48 = fma(r30, r31, r48);
    r48 = fma(r29, r36, r48);
    r48 = r5 * r48;
    r2 = fma(r0, r48, r2);
    r3 = fma(r3, r4, r1);
    r1 = r17 * r16;
    r1 = fma(r22, r1, r51);
    r49 = fma(r6, r1, r49);
    r51 = r10 * r11;
    r51 = fma(r17, r51, r54);
    r55 = r35 + r55;
    r55 = r55 + r34;
    r34 = r14 * r11;
    r34 = fma(r21, r34, r38);
    r40 = fma(r18, r23, r40);
    r50 = r35 + r50;
    r50 = r50 + r45;
    r49 = fma(r29, r51, r49);
    r49 = fma(r30, r55, r49);
    r49 = fma(r33, r34, r49);
    r49 = fma(r42, r40, r49);
    r49 = fma(r7, r50, r49);
    r49 = r47 * r49;
    r3 = fma(r0, r49, r3);
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r33 = r17 * r25;
    r30 = -5.00000000000000000e-01;
    r29 = r13 * r30;
    r45 = 5.00000000000000000e-01;
    r38 = fma(r45, r27, r11 * r29);
    r38 = fma(r30, r24, r38);
    r38 = fma(r30, r26, r38);
    r33 = r33 * r38;
    r54 = r17 * r16;
    r56 = r13 * r14;
    r57 = r12 * r15;
    r57 = fma(r30, r57, r45 * r56);
    r56 = r9 * r10;
    r57 = fma(r45, r56, r57);
    r58 = r11 * r45;
    r57 = fma(r8, r58, r57);
    r54 = fma(r57, r54, r33);
    r56 = r17 * r22;
    r59 = r12 * r14;
    r60 = r8 * r10;
    r60 = fma(r30, r60, r30 * r59);
    r60 = fma(r9, r58, r60);
    r60 = fma(r15, r29, r60);
    r56 = r56 * r60;
    r59 = r12 * r11;
    r61 = r9 * r14;
    r61 = fma(r30, r61, r30 * r59);
    r59 = r8 * r15;
    r61 = fma(r30, r59, r61);
    r62 = r13 * r10;
    r61 = fma(r45, r62, r61);
    r62 = r61 * r19;
    r59 = r56 + r62;
    r63 = r54 + r59;
    r64 = r25 * r61;
    r65 = fma(r57, r23, r21 * r64);
    r66 = r17 * r16;
    r66 = r66 * r60;
    r67 = fma(r38, r19, r66);
    r65 = r65 + r67;
    r65 = fma(r6, r65, r7 * r63);
    r63 = r25 * r57;
    r68 = -4.00000000000000000e+00;
    r63 = r63 * r68;
    r69 = r60 * r68;
    r70 = r18 * r69;
    r71 = r63 + r70;
    r65 = fma(r42, r71, r65);
    r46 = r46 * r46;
    r46 = 1.0 / r46;
    r46 = r4 * r46;
    r48 = r46 * r48;
    r71 = r17 * r22;
    r71 = fma(r17, r64, r57 * r71);
    r71 = r71 + r67;
    r72 = r17 * r25;
    r72 = r72 * r60;
    r73 = r16 * r21;
    r73 = fma(r61, r73, r72);
    r57 = r57 * r19;
    r73 = r73 + r57;
    r73 = fma(r38, r23, r73);
    r73 = fma(r7, r73, r42 * r71);
    r71 = r16 * r38;
    r74 = r68 * r71;
    r63 = r63 + r74;
    r73 = fma(r6, r63, r73);
    r63 = r5 * r73;
    r63 = fma(r0, r63, r65 * r48);
    r75 = r18 * r21;
    r76 = r60 * r23;
    r75 = fma(r61, r75, r76);
    r75 = r75 + r54;
    r74 = r70 + r74;
    r74 = fma(r7, r74, r42 * r75);
    r75 = r17 * r22;
    r75 = fma(r38, r75, r57);
    r57 = r17 * r16;
    r57 = fma(r61, r57, r72);
    r75 = r75 + r57;
    r74 = fma(r6, r75, r74);
    r75 = r47 * r74;
    r72 = r65 * r46;
    r72 = fma(r49, r72, r0 * r75);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r63, r72);
    r75 = r21 * r25;
    r75 = fma(r38, r75, r76);
    r70 = r17 * r16;
    r54 = r8 * r11;
    r77 = r12 * r15;
    r77 = fma(r45, r77, r30 * r54);
    r54 = r9 * r10;
    r77 = fma(r30, r54, r77);
    r77 = fma(r14, r29, r77);
    r70 = r70 * r77;
    r54 = r9 * r14;
    r78 = r8 * r15;
    r78 = fma(r45, r78, r45 * r54);
    r78 = fma(r12, r58, r78);
    r78 = fma(r10, r29, r78);
    r29 = fma(r78, r19, r70);
    r75 = r75 + r29;
    r54 = r17 * r25;
    r54 = r54 * r78;
    r79 = r17 * r22;
    r79 = fma(r77, r79, r54);
    r79 = r79 + r67;
    r79 = fma(r7, r79, r6 * r75);
    r75 = r18 * r68;
    r75 = r75 * r77;
    r67 = r25 * r69;
    r80 = r75 + r67;
    r79 = fma(r42, r80, r79);
    r56 = r33 + r56;
    r56 = r56 + r29;
    r29 = r16 * r68;
    r29 = r29 * r78;
    r67 = r29 + r67;
    r67 = fma(r6, r67, r42 * r56);
    r56 = fma(r78, r23, r21 * r71);
    r33 = r17 * r25;
    r60 = r60 * r19;
    r33 = fma(r77, r33, r60);
    r56 = r56 + r33;
    r67 = fma(r7, r56, r67);
    r56 = r5 * r67;
    r56 = fma(r0, r56, r79 * r48);
    r80 = r79 * r46;
    r54 = r66 + r54;
    r66 = r18 * r21;
    r54 = fma(r38, r66, r54);
    r54 = fma(r77, r23, r54);
    r66 = r17 * r22;
    r71 = fma(r17, r71, r78 * r66);
    r71 = r71 + r33;
    r71 = fma(r6, r71, r42 * r54);
    r29 = r75 + r29;
    r71 = fma(r7, r29, r71);
    r29 = r47 * r71;
    r29 = fma(r0, r29, r49 * r80);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r56, r29);
    r80 = r18 * r68;
    r27 = fma(r30, r27, r13 * r58);
    r27 = fma(r45, r24, r27);
    r27 = fma(r45, r26, r27);
    r80 = r80 * r27;
    r64 = r68 * r64;
    r68 = r80 + r64;
    r26 = r17 * r16;
    r26 = r26 * r27;
    r45 = r21 * r25;
    r45 = fma(r77, r45, r26);
    r45 = r45 + r60;
    r45 = fma(r61, r23, r45);
    r45 = fma(r6, r45, r42 * r68);
    r68 = r17 * r22;
    r19 = fma(r77, r19, r27 * r68);
    r19 = r19 + r57;
    r45 = fma(r7, r19, r45);
    r19 = r17 * r25;
    r19 = r19 * r27;
    r68 = r16 * r21;
    r68 = fma(r77, r68, r19);
    r68 = r68 + r62;
    r68 = r68 + r76;
    r69 = r16 * r69;
    r64 = r64 + r69;
    r64 = fma(r6, r64, r7 * r68);
    r68 = r17 * r22;
    r68 = fma(r61, r68, r26);
    r68 = r68 + r33;
    r64 = fma(r42, r68, r64);
    r68 = r5 * r64;
    r68 = fma(r0, r68, r45 * r48);
    r19 = r70 + r19;
    r19 = r19 + r59;
    r59 = r18 * r21;
    r23 = fma(r27, r23, r77 * r59);
    r23 = r23 + r57;
    r23 = fma(r42, r23, r6 * r19);
    r69 = r80 + r69;
    r23 = fma(r7, r69, r23);
    r69 = r47 * r23;
    r7 = r45 * r46;
    r7 = fma(r49, r7, r0 * r69);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r68, r7);
    r69 = r5 * r36;
    r69 = fma(r32, r48, r0 * r69);
    r80 = r32 * r46;
    r42 = r47 * r51;
    r42 = fma(r0, r42, r49 * r80);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r69, r42);
    r80 = r5 * r31;
    r80 = fma(r0, r80, r39 * r48);
    r19 = r39 * r46;
    r6 = r47 * r55;
    r6 = fma(r0, r6, r49 * r19);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r80, r6);
    r19 = r5 * r20;
    r19 = fma(r0, r19, r37 * r48);
    r57 = r47 * r34;
    r27 = r37 * r46;
    r27 = fma(r49, r27, r0 * r57);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r19, r27);
    r57 = r4 * r2;
    r59 = r4 * r3;
    r59 = fma(r72, r59, r63 * r57);
    r57 = r4 * r3;
    r77 = r4 * r2;
    r77 = fma(r56, r77, r29 * r57);
    WriteSum2<double, double>((double*)inout_shared, r59, r77);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = r4 * r2;
    r59 = r4 * r3;
    r59 = fma(r7, r59, r68 * r77);
    r77 = r4 * r3;
    r57 = r4 * r2;
    r57 = fma(r69, r57, r42 * r77);
    WriteSum2<double, double>((double*)inout_shared, r59, r57);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = r4 * r3;
    r59 = r4 * r2;
    r59 = fma(r80, r59, r6 * r57);
    r57 = r4 * r2;
    r77 = r4 * r3;
    r77 = fma(r27, r77, r19 * r57);
    WriteSum2<double, double>((double*)inout_shared, r59, r77);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = fma(r72, r72, r63 * r63);
    r59 = fma(r56, r56, r29 * r29);
    WriteSum2<double, double>((double*)inout_shared, r77, r59);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = fma(r68, r68, r7 * r7);
    r77 = fma(r69, r69, r42 * r42);
    WriteSum2<double, double>((double*)inout_shared, r59, r77);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = fma(r6, r6, r80 * r80);
    r59 = fma(r27, r27, r19 * r19);
    WriteSum2<double, double>((double*)inout_shared, r77, r59);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = fma(r63, r56, r72 * r29);
    r77 = fma(r63, r68, r72 * r7);
    WriteSum2<double, double>((double*)inout_shared, r59, r77);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = fma(r72, r42, r63 * r69);
    r59 = fma(r63, r80, r72 * r6);
    WriteSum2<double, double>((double*)inout_shared, r77, r59);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fma(r72, r27, r63 * r19);
    r63 = fma(r29, r7, r56 * r68);
    WriteSum2<double, double>((double*)inout_shared, r72, r63);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = fma(r56, r69, r29 * r42);
    r72 = fma(r56, r80, r29 * r6);
    WriteSum2<double, double>((double*)inout_shared, r63, r72);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = fma(r56, r19, r29 * r27);
    r29 = fma(r7, r42, r68 * r69);
    WriteSum2<double, double>((double*)inout_shared, r56, r29);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fma(r68, r80, r7 * r6);
    r7 = fma(r7, r27, r68 * r19);
    WriteSum2<double, double>((double*)inout_shared, r29, r7);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = fma(r69, r80, r42 * r6);
    r42 = fma(r42, r27, r69 * r19);
    WriteSum2<double, double>((double*)inout_shared, r7, r42);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r6, r27, r80 * r19);
    WriteSum1<double, double>((double*)inout_shared, r27);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = r4 * r2;
    r6 = r4 * r3;
    WriteSum2<double, double>((double*)inout_shared, r27, r6);
  };
  FlushSumShared<2, double>(out_principal_point_njtr,
                            0 * out_principal_point_njtr_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r35, r35);
  };
  FlushSumShared<2, double>(out_principal_point_precond_diag,
                            0 * out_principal_point_precond_diag_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r5 * r43;
    r35 = fma(r0, r35, r28 * r48);
    r6 = r47 * r1;
    r27 = r28 * r46;
    r27 = fma(r49, r27, r0 * r6);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r35,
                                             r27);
    r6 = r5 * r52;
    r6 = fma(r0, r6, r41 * r48);
    r19 = r47 * r50;
    r80 = r41 * r46;
    r80 = fma(r49, r80, r0 * r19);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 2 * out_point_jac_num_alloc, global_thread_idx, r6, r80);
    r19 = r5 * r53;
    r19 = fma(r0, r19, r44 * r48);
    r48 = r44 * r46;
    r42 = r47 * r40;
    r42 = fma(r0, r42, r49 * r48);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r19,
                                             r42);
    r48 = r4 * r3;
    r0 = r4 * r2;
    r0 = fma(r35, r0, r27 * r48);
    r48 = r4 * r3;
    r49 = r4 * r2;
    r49 = fma(r6, r49, r80 * r48);
    WriteSum2<double, double>((double*)inout_shared, r0, r49);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r4 * r2;
    r0 = r4 * r3;
    r0 = fma(r42, r0, r19 * r49);
    WriteSum1<double, double>((double*)inout_shared, r0);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = fma(r35, r35, r27 * r27);
    r49 = fma(r6, r6, r80 * r80);
    WriteSum2<double, double>((double*)inout_shared, r0, r49);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = fma(r19, r19, r42 * r42);
    WriteSum1<double, double>((double*)inout_shared, r49);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = fma(r27, r80, r35 * r6);
    r27 = fma(r27, r42, r35 * r19);
    WriteSum2<double, double>((double*)inout_shared, r49, r27);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = fma(r80, r42, r6 * r19);
    WriteSum1<double, double>((double*)inout_shared, r42);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
}

void PinholeSplitFixedFocalResJac(
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
    double* focal,
    unsigned int focal_num_alloc,
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
  PinholeSplitFixedFocalResJacKernel<<<n_blocks, 1024>>>(
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
      focal,
      focal_num_alloc,
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