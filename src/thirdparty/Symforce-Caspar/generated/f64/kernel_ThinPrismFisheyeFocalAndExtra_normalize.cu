#include "kernel_ThinPrismFisheyeFocalAndExtra_normalize.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFocalAndExtraNormalizeKernel(
        double* precond_diag,
        unsigned int precond_diag_num_alloc,
        double* precond_tril,
        unsigned int precond_tril_num_alloc,
        double* njtr,
        unsigned int njtr_num_alloc,
        const double* const diag,
        double* out_normalized,
        unsigned int out_normalized_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 2 * precond_tril_num_alloc, global_thread_idx, r0, r1);
    r2 = -1.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 34 * precond_tril_num_alloc, global_thread_idx, r3, r4);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 16 * precond_tril_num_alloc, global_thread_idx, r5, r6);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 10 * precond_tril_num_alloc, global_thread_idx, r7, r8);
  };
  LoadUnique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = 1.00000000000000008e-15;
    r10 = r9 * r10;
    ReadIdx2<1024, double, double, double2>(
        precond_diag, 0 * precond_diag_num_alloc, global_thread_idx, r11, r12);
    r13 = 1.00000000000000000e+00;
    r13 = r9 + r13;
    r12 = fma(r12, r13, r10);
    r12 = 1.0 / r12;
    r9 = r8 * r12;
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 24 * precond_tril_num_alloc, global_thread_idx, r14, r15);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 18 * precond_tril_num_alloc, global_thread_idx, r16, r17);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 0 * precond_tril_num_alloc, global_thread_idx, r18, r19);
    r18 = r19 * r1;
    r11 = fma(r11, r13, r10);
    r11 = 1.0 / r11;
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 8 * precond_tril_num_alloc, global_thread_idx, r20, r21);
    r18 = fma(r21, r9, r11 * r18);
    r18 = fma(r2, r18, r16);
    r16 = r21 * r21;
    r20 = r19 * r19;
    r20 = fma(r11, r20, r12 * r16);
    r20 = fma(r2, r20, r10);
    ReadIdx2<1024, double, double, double2>(
        precond_diag, 2 * precond_diag_num_alloc, global_thread_idx, r16, r22);
    r20 = fma(r16, r13, r20);
    r20 = 1.0 / r20;
    r16 = r18 * r20;
    r23 = r19 * r0;
    r24 = r21 * r7;
    r24 = fma(r12, r24, r11 * r23);
    r24 = fma(r2, r24, r6);
    r6 = fma(r7, r9, r24 * r16);
    r23 = r0 * r1;
    r6 = fma(r11, r23, r6);
    r6 = fma(r2, r6, r14);
    r14 = r0 * r0;
    r23 = r7 * r7;
    r23 = fma(r12, r23, r11 * r14);
    r14 = r24 * r24;
    r23 = fma(r20, r14, r23);
    r23 = fma(r2, r23, r10);
    r23 = fma(r22, r13, r23);
    r23 = 1.0 / r23;
    r22 = r6 * r23;
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 28 * precond_tril_num_alloc, global_thread_idx, r14, r25);
    r26 = r7 * r5;
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 22 * precond_tril_num_alloc, global_thread_idx, r27, r28);
    r29 = r21 * r5;
    r29 = r29 * r2;
    r29 = fma(r12, r29, r28);
    r28 = r24 * r29;
    r28 = fma(r20, r28, r12 * r26);
    r28 = fma(r2, r28, r25);
    r25 = fma(r28, r22, r5 * r9);
    r25 = fma(r29, r16, r25);
    r25 = fma(r2, r25, r3);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 32 * precond_tril_num_alloc, global_thread_idx, r3, r26);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 6 * precond_tril_num_alloc, global_thread_idx, r30, r31);
    r32 = r19 * r31;
    r33 = r2 * r11;
    r32 = fma(r33, r32, r27);
    r27 = r1 * r31;
    r27 = fma(r11, r27, r32 * r16);
    r34 = r0 * r31;
    r35 = r24 * r32;
    r35 = fma(r20, r35, r11 * r34);
    r35 = fma(r2, r35, r14);
    r27 = fma(r35, r22, r27);
    r27 = fma(r2, r27, r26);
    r26 = r25 * r27;
    ReadIdx2<1024, double, double, double2>(
        precond_diag, 4 * precond_diag_num_alloc, global_thread_idx, r14, r34);
    r14 = fma(r14, r13, r10);
    r36 = r1 * r1;
    r6 = fma(r6, r22, r11 * r36);
    r6 = fma(r8, r9, r6);
    r6 = fma(r18, r16, r6);
    r14 = fma(r2, r6, r14);
    r14 = 1.0 / r14;
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 40 * precond_tril_num_alloc, global_thread_idx, r6, r18);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 30 * precond_tril_num_alloc, global_thread_idx, r8, r36);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 12 * precond_tril_num_alloc, global_thread_idx, r37, r38);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 4 * precond_tril_num_alloc, global_thread_idx, r39, r40);
    r41 = r1 * r40;
    r41 = fma(r11, r41, r38 * r9);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 20 * precond_tril_num_alloc, global_thread_idx, r42, r43);
    r44 = r19 * r40;
    r45 = r21 * r38;
    r45 = fma(r12, r45, r11 * r44);
    r45 = fma(r2, r45, r42);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 26 * precond_tril_num_alloc, global_thread_idx, r42, r44);
    r46 = r7 * r38;
    r47 = r0 * r40;
    r47 = fma(r11, r47, r12 * r46);
    r46 = r24 * r45;
    r47 = fma(r20, r46, r47);
    r47 = fma(r2, r47, r42);
    r41 = fma(r45, r16, r41);
    r41 = fma(r47, r22, r41);
    r41 = fma(r2, r41, r36);
    r36 = r41 * r14;
    r42 = r40 * r31;
    r42 = fma(r11, r42, r27 * r36);
    r46 = r47 * r35;
    r42 = fma(r23, r46, r42);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 36 * precond_tril_num_alloc, global_thread_idx, r48, r49);
    r50 = r39 * r31;
    r51 = r19 * r39;
    r52 = r21 * r37;
    r52 = fma(r12, r52, r11 * r51);
    r52 = fma(r2, r52, r17);
    r17 = r52 * r32;
    r17 = fma(r20, r17, r11 * r50);
    r50 = r7 * r37;
    r51 = r0 * r39;
    r51 = fma(r11, r51, r12 * r50);
    r50 = r24 * r52;
    r51 = fma(r20, r50, r51);
    r51 = fma(r2, r51, r15);
    r15 = r51 * r35;
    r17 = fma(r23, r15, r17);
    r50 = fma(r37, r9, r51 * r22);
    r53 = r1 * r39;
    r50 = fma(r11, r53, r50);
    r50 = fma(r52, r16, r50);
    r50 = fma(r2, r50, r8);
    r8 = r50 * r27;
    r17 = fma(r14, r8, r17);
    r17 = fma(r2, r17, r49);
    r49 = r45 * r52;
    r8 = r37 * r38;
    r8 = fma(r12, r8, r20 * r49);
    r49 = r39 * r40;
    r8 = fma(r11, r49, r8);
    r15 = r47 * r51;
    r8 = fma(r23, r15, r8);
    r8 = fma(r50, r36, r8);
    r8 = fma(r2, r8, r4);
    r34 = fma(r34, r13, r10);
    r4 = r50 * r50;
    r15 = r52 * r52;
    r15 = fma(r20, r15, r14 * r4);
    r4 = r51 * r51;
    r49 = r39 * r39;
    r53 = r37 * r37;
    r15 = fma(r23, r4, r15);
    r15 = fma(r11, r49, r15);
    r15 = fma(r12, r53, r15);
    r34 = fma(r2, r15, r34);
    r34 = 1.0 / r34;
    r15 = r8 * r34;
    r53 = r45 * r32;
    r42 = fma(r20, r53, r42);
    r42 = fma(r17, r15, r42);
    r42 = fma(r2, r42, r6);
    ReadIdx2<1024, double, double, double2>(
        precond_diag, 6 * precond_diag_num_alloc, global_thread_idx, r6, r53);
    r6 = fma(r6, r13, r10);
    r46 = r47 * r47;
    r49 = r45 * r45;
    r49 = fma(r20, r49, r23 * r46);
    r46 = r40 * r40;
    r4 = r38 * r38;
    r49 = fma(r11, r46, r49);
    r49 = fma(r12, r4, r49);
    r49 = fma(r8, r15, r49);
    r49 = fma(r41, r36, r49);
    r6 = fma(r2, r49, r6);
    r6 = 1.0 / r6;
    r49 = r42 * r6;
    r41 = r38 * r5;
    r41 = fma(r12, r41, r25 * r36);
    r8 = r45 * r29;
    r41 = fma(r20, r8, r41);
    r4 = r47 * r28;
    r41 = fma(r23, r4, r41);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 38 * precond_tril_num_alloc, global_thread_idx, r46, r54);
    r55 = r37 * r5;
    r56 = r28 * r51;
    r56 = fma(r23, r56, r12 * r55);
    r55 = r29 * r52;
    r56 = fma(r20, r55, r56);
    r57 = r50 * r25;
    r56 = fma(r14, r57, r56);
    r56 = fma(r2, r56, r46);
    r41 = fma(r56, r15, r41);
    r41 = fma(r2, r41, r18);
    r26 = fma(r41, r49, r14 * r26);
    r18 = r28 * r35;
    r26 = fma(r23, r18, r26);
    r4 = r56 * r17;
    r26 = fma(r34, r4, r26);
    r8 = r29 * r32;
    r26 = fma(r20, r8, r26);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 42 * precond_tril_num_alloc, global_thread_idx, r46, r57);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 14 * precond_tril_num_alloc, global_thread_idx, r55, r58);
    r58 = r7 * r55;
    r59 = r0 * r30;
    r59 = fma(r11, r59, r12 * r58);
    r58 = r19 * r30;
    r60 = r21 * r55;
    r60 = fma(r12, r60, r11 * r58);
    r60 = fma(r2, r60, r43);
    r43 = r24 * r60;
    r59 = fma(r20, r43, r59);
    r59 = fma(r2, r59, r44);
    r44 = r35 * r59;
    r43 = r30 * r31;
    r43 = fma(r11, r43, r23 * r44);
    r44 = r1 * r30;
    r44 = fma(r11, r44, r55 * r9);
    r44 = fma(r60, r16, r44);
    r44 = fma(r59, r22, r44);
    r44 = fma(r2, r44, r3);
    r3 = r38 * r55;
    r3 = fma(r12, r3, r44 * r36);
    r58 = r40 * r30;
    r3 = fma(r11, r58, r3);
    r61 = r47 * r59;
    r3 = fma(r23, r61, r3);
    r62 = r45 * r60;
    r3 = fma(r20, r62, r3);
    r63 = r52 * r60;
    r64 = r50 * r44;
    r64 = fma(r14, r64, r20 * r63);
    r63 = r37 * r55;
    r64 = fma(r12, r63, r64);
    r65 = r39 * r30;
    r64 = fma(r11, r65, r64);
    r66 = r51 * r59;
    r64 = fma(r23, r66, r64);
    r64 = fma(r2, r64, r48);
    r3 = fma(r64, r15, r3);
    r3 = fma(r2, r3, r54);
    r54 = r32 * r60;
    r43 = fma(r20, r54, r43);
    r62 = r27 * r44;
    r43 = fma(r14, r62, r43);
    r61 = r17 * r64;
    r43 = fma(r34, r61, r43);
    r43 = fma(r3, r49, r43);
    r43 = fma(r2, r43, r46);
    r53 = fma(r53, r13, r10);
    r46 = r44 * r44;
    r61 = r60 * r60;
    r61 = fma(r20, r61, r14 * r46);
    r46 = r30 * r30;
    r62 = r55 * r55;
    r54 = r3 * r3;
    r58 = r59 * r59;
    r48 = r64 * r64;
    r61 = fma(r11, r46, r61);
    r61 = fma(r12, r62, r61);
    r61 = fma(r6, r54, r61);
    r61 = fma(r23, r58, r61);
    r61 = fma(r34, r48, r61);
    r53 = fma(r2, r61, r53);
    r53 = 1.0 / r53;
    r61 = r43 * r53;
    r48 = r55 * r5;
    r58 = r56 * r64;
    r58 = fma(r34, r58, r12 * r48);
    r48 = r28 * r59;
    r58 = fma(r23, r48, r58);
    r54 = r25 * r44;
    r58 = fma(r14, r54, r58);
    r62 = r3 * r41;
    r58 = fma(r6, r62, r58);
    r46 = r29 * r60;
    r58 = fma(r20, r46, r58);
    r58 = fma(r2, r58, r57);
    r26 = fma(r58, r61, r26);
    ReadIdx2<1024, double, double, double2>(
        precond_diag, 8 * precond_diag_num_alloc, global_thread_idx, r8, r4);
    r8 = fma(r8, r13, r10);
    r18 = r35 * r35;
    r18 = fma(r23, r18, r42 * r49);
    r42 = r27 * r27;
    r57 = r32 * r32;
    r46 = r31 * r31;
    r62 = r17 * r17;
    r18 = fma(r43, r61, r18);
    r18 = fma(r14, r42, r18);
    r18 = fma(r20, r57, r18);
    r18 = fma(r11, r46, r18);
    r18 = fma(r34, r62, r18);
    r8 = fma(r2, r18, r8);
    r8 = 1.0 / r8;
    r18 = r26 * r8;
    r13 = fma(r4, r13, r10);
    r4 = r28 * r28;
    r10 = r56 * r56;
    r10 = fma(r34, r10, r23 * r4);
    r4 = r5 * r5;
    r62 = r41 * r41;
    r46 = r29 * r29;
    r57 = r58 * r58;
    r42 = r25 * r25;
    r10 = fma(r12, r4, r10);
    r10 = fma(r26, r18, r10);
    r10 = fma(r6, r62, r10);
    r10 = fma(r20, r46, r10);
    r10 = fma(r53, r57, r10);
    r10 = fma(r14, r42, r10);
    r13 = fma(r2, r10, r13);
    r13 = 1.0 / r13;
    ReadIdx2<1024, double, double, double2>(
        njtr, 8 * njtr_num_alloc, global_thread_idx, r10, r42);
    ReadIdx2<1024, double, double, double2>(
        njtr, 0 * njtr_num_alloc, global_thread_idx, r57, r46);
    r62 = r5 * r46;
    r62 = r62 * r2;
    r62 = fma(r12, r62, r42);
    r42 = r2 * r29;
    ReadIdx2<1024, double, double, double2>(
        njtr, 2 * njtr_num_alloc, global_thread_idx, r26, r4);
    r43 = r57 * r33;
    r26 = fma(r19, r43, r26);
    r54 = r21 * r46;
    r54 = r54 * r2;
    r26 = fma(r12, r54, r26);
    r42 = r42 * r26;
    r62 = fma(r20, r42, r62);
    r54 = r2 * r17;
    ReadIdx2<1024, double, double, double2>(
        njtr, 4 * njtr_num_alloc, global_thread_idx, r48, r66);
    r65 = r37 * r46;
    r65 = r65 * r2;
    r65 = fma(r12, r65, r66);
    r66 = r2 * r52;
    r66 = r66 * r26;
    r65 = fma(r20, r66, r65);
    r63 = r2 * r51;
    r67 = r7 * r46;
    r67 = r67 * r2;
    r67 = fma(r12, r67, r4);
    r4 = r2 * r24;
    r4 = r4 * r26;
    r67 = fma(r20, r4, r67);
    r67 = fma(r0, r43, r67);
    r63 = r63 * r67;
    r65 = fma(r23, r63, r65);
    r4 = r2 * r50;
    r68 = r46 * r2;
    r68 = fma(r9, r68, r48);
    r48 = r2 * r67;
    r68 = fma(r22, r48, r68);
    r69 = r2 * r26;
    r68 = fma(r16, r69, r68);
    r68 = fma(r1, r43, r68);
    r4 = r4 * r68;
    r65 = fma(r14, r4, r65);
    r65 = fma(r39, r43, r65);
    r54 = r54 * r65;
    r54 = fma(r34, r54, r10);
    r10 = r2 * r35;
    r10 = r10 * r67;
    r54 = fma(r23, r10, r54);
    ReadIdx2<1024, double, double, double2>(
        njtr, 6 * njtr_num_alloc, global_thread_idx, r4, r63);
    r66 = r38 * r46;
    r66 = r66 * r2;
    r66 = fma(r12, r66, r4);
    r4 = r2 * r45;
    r4 = r4 * r26;
    r66 = fma(r20, r4, r66);
    r15 = r2 * r15;
    r69 = r2 * r68;
    r66 = fma(r36, r69, r66);
    r48 = r2 * r47;
    r48 = r48 * r67;
    r66 = fma(r23, r48, r66);
    r66 = fma(r65, r15, r66);
    r66 = fma(r40, r43, r66);
    r48 = r2 * r66;
    r54 = fma(r49, r48, r54);
    r69 = r2 * r27;
    r69 = r69 * r68;
    r54 = fma(r14, r69, r54);
    r4 = r55 * r46;
    r4 = r4 * r2;
    r4 = fma(r12, r4, r63);
    r63 = r2 * r59;
    r63 = r63 * r67;
    r4 = fma(r23, r63, r4);
    r70 = r2 * r3;
    r70 = r70 * r66;
    r4 = fma(r6, r70, r4);
    r71 = r2 * r64;
    r71 = r71 * r65;
    r4 = fma(r34, r71, r4);
    r72 = r2 * r60;
    r72 = r72 * r26;
    r4 = fma(r20, r72, r4);
    r73 = r2 * r44;
    r73 = r73 * r68;
    r4 = fma(r14, r73, r4);
    r4 = fma(r30, r43, r4);
    r73 = r2 * r4;
    r54 = fma(r61, r73, r54);
    r72 = r2 * r32;
    r72 = r72 * r26;
    r54 = fma(r20, r72, r54);
    r54 = fma(r31, r43, r54);
    r43 = r2 * r56;
    r43 = r43 * r65;
    r62 = fma(r34, r43, r62);
    r72 = r2 * r28;
    r72 = r72 * r67;
    r62 = fma(r23, r72, r62);
    r73 = r2 * r41;
    r73 = r73 * r66;
    r62 = fma(r6, r73, r62);
    r69 = r2 * r58;
    r69 = r69 * r4;
    r62 = fma(r53, r69, r62);
    r48 = r2 * r25;
    r48 = r48 * r68;
    r62 = fma(r14, r48, r62);
    r62 = fma(r54, r18, r62);
    r62 = r13 * r62;
    r8 = fma(r54, r8, r62 * r18);
    r54 = r2 * r8;
    r54 = fma(r66, r6, r49 * r54);
    r49 = r41 * r6;
    r18 = r2 * r62;
    r54 = fma(r18, r49, r54);
    r13 = r2 * r3;
    r48 = r2 * r8;
    r48 = fma(r61, r48, r4 * r53);
    r61 = r58 * r53;
    r48 = fma(r18, r61, r48);
    r13 = r13 * r48;
    r54 = fma(r6, r13, r54);
    r13 = r2 * r54;
    r49 = r25 * r14;
    r49 = fma(r18, r49, r36 * r13);
    r13 = r2 * r27;
    r13 = r13 * r8;
    r49 = fma(r14, r13, r49);
    r36 = r2 * r50;
    r61 = r2 * r17;
    r61 = r61 * r8;
    r69 = r2 * r64;
    r69 = r69 * r48;
    r69 = fma(r34, r69, r34 * r61);
    r61 = r56 * r34;
    r69 = fma(r18, r61, r69);
    r69 = fma(r54, r15, r69);
    r69 = fma(r65, r34, r69);
    r36 = r36 * r69;
    r49 = fma(r14, r36, r49);
    r61 = r2 * r44;
    r61 = r61 * r48;
    r49 = fma(r14, r61, r49);
    r49 = fma(r68, r14, r49);
    r61 = r1 * r49;
    r36 = r39 * r69;
    r36 = fma(r33, r36, r33 * r61);
    r61 = r40 * r54;
    r36 = fma(r33, r61, r36);
    r13 = r2 * r59;
    r13 = r13 * r48;
    r65 = r2 * r51;
    r65 = r65 * r69;
    r65 = fma(r23, r65, r23 * r13);
    r13 = r28 * r23;
    r65 = fma(r18, r13, r65);
    r15 = r2 * r47;
    r15 = r15 * r54;
    r65 = fma(r23, r15, r65);
    r73 = r2 * r49;
    r65 = fma(r22, r73, r65);
    r22 = r2 * r35;
    r22 = r22 * r8;
    r65 = fma(r23, r22, r65);
    r65 = fma(r67, r23, r65);
    r22 = r0 * r65;
    r36 = fma(r33, r22, r36);
    r73 = r30 * r48;
    r36 = fma(r33, r73, r36);
    r15 = r2 * r24;
    r15 = r15 * r65;
    r15 = fma(r26, r20, r20 * r15);
    r13 = r2 * r49;
    r15 = fma(r16, r13, r15);
    r16 = r2 * r52;
    r16 = r16 * r69;
    r15 = fma(r20, r16, r15);
    r72 = r2 * r60;
    r72 = r72 * r48;
    r15 = fma(r20, r72, r15);
    r43 = r2 * r45;
    r43 = r43 * r54;
    r15 = fma(r20, r43, r15);
    r42 = r2 * r32;
    r42 = r42 * r8;
    r15 = fma(r20, r42, r15);
    r10 = r29 * r20;
    r15 = fma(r18, r10, r15);
    r10 = r19 * r15;
    r36 = fma(r33, r10, r36);
    r42 = r31 * r8;
    r36 = fma(r33, r42, r36);
    r36 = fma(r57, r11, r36);
    r42 = r37 * r2;
    r42 = r42 * r69;
    r42 = fma(r46, r12, r12 * r42);
    r10 = r55 * r2;
    r10 = r10 * r48;
    r42 = fma(r12, r10, r42);
    r11 = r38 * r2;
    r11 = r11 * r54;
    r42 = fma(r12, r11, r42);
    r57 = r21 * r2;
    r57 = r57 * r15;
    r42 = fma(r12, r57, r42);
    r73 = r5 * r12;
    r42 = fma(r18, r73, r42);
    r18 = r7 * r2;
    r18 = r18 * r65;
    r42 = fma(r12, r18, r42);
    r22 = r2 * r49;
    r42 = fma(r9, r22, r42);
    WriteIdx2<1024, double, double, double2>(out_normalized,
                                             0 * out_normalized_num_alloc,
                                             global_thread_idx,
                                             r36,
                                             r42);
    WriteIdx2<1024, double, double, double2>(out_normalized,
                                             2 * out_normalized_num_alloc,
                                             global_thread_idx,
                                             r15,
                                             r65);
    WriteIdx2<1024, double, double, double2>(out_normalized,
                                             4 * out_normalized_num_alloc,
                                             global_thread_idx,
                                             r49,
                                             r69);
    WriteIdx2<1024, double, double, double2>(out_normalized,
                                             6 * out_normalized_num_alloc,
                                             global_thread_idx,
                                             r54,
                                             r48);
    WriteIdx2<1024, double, double, double2>(out_normalized,
                                             8 * out_normalized_num_alloc,
                                             global_thread_idx,
                                             r8,
                                             r62);
  };
}

void ThinPrismFisheyeFocalAndExtraNormalize(
    double* precond_diag,
    unsigned int precond_diag_num_alloc,
    double* precond_tril,
    unsigned int precond_tril_num_alloc,
    double* njtr,
    unsigned int njtr_num_alloc,
    const double* const diag,
    double* out_normalized,
    unsigned int out_normalized_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFocalAndExtraNormalizeKernel<<<n_blocks, 1024>>>(
      precond_diag,
      precond_diag_num_alloc,
      precond_tril,
      precond_tril_num_alloc,
      njtr,
      njtr_num_alloc,
      diag,
      out_normalized,
      out_normalized_num_alloc,
      problem_size);
}

}  // namespace caspar