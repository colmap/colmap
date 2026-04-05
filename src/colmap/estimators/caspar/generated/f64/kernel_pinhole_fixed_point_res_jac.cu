#include "kernel_pinhole_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) pinhole_fixed_point_res_jac_kernel(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
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
    double* out_calib_jac,
    unsigned int out_calib_jac_num_alloc,
    double* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    double* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    double* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
    size_t problem_size) {
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46;
  load_shared<2, double, double>(
      calib, 2 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r0);
    r0 = 1.00000000000000008e-15;
  };
  load_shared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r6, r7);
  };
  load_shared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  load_shared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r10, r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = r9 * r10;
    r13 = 2.00000000000000000e+00;
    r12 = r12 * r13;
    r14 = r8 * r13;
    r15 = r11 * r14;
    r16 = r12 + r15;
    r5 = fma(r7, r16, r5);
    r17 = r9 * r11;
    r18 = -2.00000000000000000e+00;
    r17 = r17 * r18;
    r19 = r10 * r14;
    r20 = r17 + r19;
    read_idx_1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r21);
    r22 = r8 * r8;
    r22 = r22 * r18;
    r23 = 1.00000000000000000e+00;
    r24 = r9 * r9;
    r25 = fma(r18, r24, r23);
    r26 = r22 + r25;
    r5 = fma(r6, r20, r5);
    r5 = fma(r21, r26, r5);
    r26 = copysign(1.0, r5);
    r26 = fma(r0, r26, r5);
    r0 = 1.0 / r26;
  };
  load_shared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r5, r27);
  };
  __syncthreads();
  load_shared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r28, r29);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r30 = r10 * r18;
    r31 = r11 * r30;
    r14 = r9 * r14;
    r32 = r31 + r14;
    r28 = fma(r7, r32, r28);
    r33 = r9 * r11;
    r33 = r33 * r13;
    r19 = r33 + r19;
    r34 = r10 * r30;
    r25 = r34 + r25;
    r28 = fma(r21, r19, r28);
    r28 = fma(r6, r25, r28);
    r25 = r5 * r28;
    r2 = fma(r0, r25, r2);
    r3 = fma(r3, r4, r1);
    r1 = r10 * r11;
    r1 = r1 * r13;
    r14 = r1 + r14;
    r29 = fma(r6, r14, r29);
    r13 = r8 * r11;
    r13 = r13 * r18;
    r12 = r12 + r13;
    r34 = r23 + r34;
    r34 = r34 + r22;
    r29 = fma(r21, r12, r29);
    r29 = fma(r7, r34, r29);
    r34 = r27 * r29;
    r3 = fma(r0, r34, r3);
    write_idx_2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r22 = r8 * r9;
    r22 = r22 * r18;
    r1 = r1 + r22;
    r19 = fma(r7, r19, r21 * r1);
    r1 = r5 * r19;
    r9 = r9 * r30;
    r13 = r13 + r9;
    r18 = r10 * r10;
    r35 = r4 * r24;
    r36 = r18 + r35;
    r11 = r11 * r11;
    r37 = r8 * r8;
    r37 = r37 * r4;
    r38 = r11 + r37;
    r39 = r36 + r38;
    r39 = fma(r7, r39, r21 * r13);
    r13 = r5 * r28;
    r40 = r26 * r26;
    r41 = 1.0 / r40;
    r13 = r13 * r4;
    r13 = r13 * r41;
    r1 = fma(r39, r13, r0 * r1);
    r42 = r4 * r41;
    r43 = r39 * r42;
    r44 = r8 * r8;
    r45 = r4 * r11;
    r46 = r44 + r45;
    r36 = r36 + r46;
    r36 = fma(r21, r36, r7 * r12);
    r12 = r27 * r36;
    r12 = fma(r0, r12, r34 * r43);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r1, r12);
    r10 = r10 * r10;
    r10 = r10 * r4;
    r43 = r24 + r10;
    r46 = r46 + r43;
    r46 = fma(r6, r46, r21 * r20);
    r30 = r8 * r30;
    r17 = r17 + r30;
    r11 = r44 + r11;
    r11 = r11 + r35;
    r11 = r11 + r10;
    r11 = fma(r21, r11, r6 * r17);
    r17 = r5 * r11;
    r17 = fma(r0, r17, r46 * r13);
    r9 = r15 + r9;
    r9 = fma(r6, r9, r21 * r14);
    r14 = r27 * r9;
    r21 = r46 * r42;
    r21 = fma(r34, r21, r0 * r14);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r17, r21);
    r18 = r24 + r18;
    r18 = r18 + r37;
    r18 = r18 + r45;
    r18 = fma(r7, r18, r6 * r32);
    r32 = r5 * r18;
    r30 = r33 + r30;
    r30 = fma(r7, r30, r6 * r16);
    r32 = fma(r30, r13, r0 * r32);
    r31 = r22 + r31;
    r43 = r38 + r43;
    r43 = fma(r6, r43, r7 * r31);
    r6 = r27 * r43;
    r31 = r30 * r42;
    r31 = fma(r34, r31, r0 * r6);
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r32, r31);
    r6 = r5 * r0;
    r7 = r27 * r0;
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r6, r7);
    r7 = r42 * r34;
    write_idx_2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r13, r7);
    r7 = r4 * r2;
    r6 = r4 * r3;
    r6 = fma(r12, r6, r1 * r7);
    r7 = r4 * r2;
    r38 = r4 * r3;
    r38 = fma(r21, r38, r17 * r7);
    write_sum_2<double, double>((double*)inout_shared, r6, r38);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              0 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = r4 * r2;
    r6 = r4 * r3;
    r6 = fma(r31, r6, r32 * r38);
    r38 = r4 * r2;
    r38 = r38 * r0;
    r7 = r5 * r38;
    write_sum_2<double, double>((double*)inout_shared, r6, r7);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              2 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r27 * r4;
    r7 = r7 * r3;
    r7 = r7 * r0;
    r6 = r3 * r41;
    r22 = r2 * r41;
    r22 = fma(r25, r22, r34 * r6);
    write_sum_2<double, double>((double*)inout_shared, r7, r22);
  };
  flush_sum_shared<2, double>(out_pose_njtr,
                              4 * out_pose_njtr_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = fma(r1, r1, r12 * r12);
    r7 = fma(r21, r21, r17 * r17);
    write_sum_2<double, double>((double*)inout_shared, r22, r7);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              0 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r5 * r5;
    r7 = r7 * r41;
    r22 = fma(r31, r31, r32 * r32);
    write_sum_2<double, double>((double*)inout_shared, r22, r7);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              2 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r27 * r27;
    r7 = r7 * r41;
    r40 = r26 * r40;
    r26 = r26 * r40;
    r26 = 1.0 / r26;
    r22 = r29 * r26;
    r6 = r27 * r34;
    r16 = r5 * r28;
    r16 = r16 * r26;
    r16 = fma(r25, r16, r6 * r22);
    write_sum_2<double, double>((double*)inout_shared, r7, r16);
  };
  flush_sum_shared<2, double>(out_pose_precond_diag,
                              4 * out_pose_precond_diag_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = fma(r12, r21, r1 * r17);
    r7 = fma(r1, r32, r12 * r31);
    write_sum_2<double, double>((double*)inout_shared, r16, r7);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              0 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r5 * r1;
    r7 = r7 * r0;
    r16 = r27 * r12;
    r16 = r16 * r0;
    write_sum_2<double, double>((double*)inout_shared, r7, r16);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              2 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = fma(r17, r32, r21 * r31);
    r7 = r12 * r42;
    r1 = fma(r1, r13, r34 * r7);
    write_sum_2<double, double>((double*)inout_shared, r1, r16);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              4 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = r5 * r17;
    r16 = r16 * r0;
    r1 = r27 * r21;
    r1 = r1 * r0;
    write_sum_2<double, double>((double*)inout_shared, r16, r1);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              6 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r5 * r32;
    r1 = r1 * r0;
    r16 = r21 * r42;
    r16 = fma(r34, r16, r17 * r13);
    write_sum_2<double, double>((double*)inout_shared, r16, r1);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              8 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r27 * r31;
    r1 = r1 * r0;
    r16 = r31 * r42;
    r13 = fma(r32, r13, r34 * r16);
    write_sum_2<double, double>((double*)inout_shared, r1, r13);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              10 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = 0.00000000000000000e+00;
    r1 = r5 * r4;
    r40 = 1.0 / r40;
    r1 = r1 * r40;
    r1 = r1 * r25;
    write_sum_2<double, double>((double*)inout_shared, r13, r1);
  };
  flush_sum_shared<2, double>(out_pose_precond_tril,
                              12 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r4 * r40;
    r40 = r40 * r6;
    write_sum_1<double, double>((double*)inout_shared, r40);
  };
  flush_sum_shared<1, double>(out_pose_precond_tril,
                              14 * out_pose_precond_tril_num_alloc,
                              pose_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r28 * r0;
    r6 = r29 * r0;
    write_idx_2<1024, double, double, double2>(
        out_calib_jac, 0 * out_calib_jac_num_alloc, global_thread_idx, r40, r6);
    r38 = r28 * r38;
    r1 = r4 * r29;
    r1 = r1 * r3;
    r1 = r1 * r0;
    write_sum_2<double, double>((double*)inout_shared, r38, r1);
  };
  flush_sum_shared<2, double>(out_calib_njtr,
                              0 * out_calib_njtr_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r4 * r2;
    r38 = r4 * r3;
    write_sum_2<double, double>((double*)inout_shared, r1, r38);
  };
  flush_sum_shared<2, double>(out_calib_njtr,
                              2 * out_calib_njtr_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = r28 * r28;
    r38 = r38 * r41;
    r1 = r29 * r29;
    r1 = r1 * r41;
    write_sum_2<double, double>((double*)inout_shared, r38, r1);
  };
  flush_sum_shared<2, double>(out_calib_precond_diag,
                              0 * out_calib_precond_diag_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<double, double>((double*)inout_shared, r23, r23);
  };
  flush_sum_shared<2, double>(out_calib_precond_diag,
                              2 * out_calib_precond_diag_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<double, double>((double*)inout_shared, r13, r40);
  };
  flush_sum_shared<2, double>(out_calib_precond_tril,
                              0 * out_calib_precond_tril_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    write_sum_2<double, double>((double*)inout_shared, r6, r13);
  };
  flush_sum_shared<2, double>(out_calib_precond_tril,
                              4 * out_calib_precond_tril_num_alloc,
                              calib_indices_loc,
                              (double*)inout_shared);
}

void pinhole_fixed_point_res_jac(double* pose,
                                 unsigned int pose_num_alloc,
                                 SharedIndex* pose_indices,
                                 double* calib,
                                 unsigned int calib_num_alloc,
                                 SharedIndex* calib_indices,
                                 double* pixel,
                                 unsigned int pixel_num_alloc,
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
                                 double* out_calib_jac,
                                 unsigned int out_calib_jac_num_alloc,
                                 double* const out_calib_njtr,
                                 unsigned int out_calib_njtr_num_alloc,
                                 double* const out_calib_precond_diag,
                                 unsigned int out_calib_precond_diag_num_alloc,
                                 double* const out_calib_precond_tril,
                                 unsigned int out_calib_precond_tril_num_alloc,
                                 size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_fixed_point_res_jac_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
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
      out_calib_jac,
      out_calib_jac_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar