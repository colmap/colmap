#include "kernel_ThinPrismFisheyeCalib_normalize.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeCalibNormalizeKernel(double* precond_diag,
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
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97, r98, r99, r100, r101;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 8 * precond_tril_num_alloc, global_thread_idx, r0, r1);
    r2 = -1.00000000000000000e+00;
    r3 = r0 * r2;
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 64 * precond_tril_num_alloc, global_thread_idx, r4, r5);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 18 * precond_tril_num_alloc, global_thread_idx, r5, r6);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 20 * precond_tril_num_alloc, global_thread_idx, r6, r7);
    r7 = r5 * r6;
    ReadIdx2<1024, double, double, double2>(
        precond_diag, 0 * precond_diag_num_alloc, global_thread_idx, r8, r9);
  };
  LoadUnique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r11 = 1.00000000000000000e+00;
    r11 = r10 + r11;
    r12 = 1.00000000000000008e-15;
    r12 = r10 * r12;
    r9 = fma(r9, r11, r12);
    r9 = 1.0 / r9;
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 42 * precond_tril_num_alloc, global_thread_idx, r10, r13);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 22 * precond_tril_num_alloc, global_thread_idx, r14, r15);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 2 * precond_tril_num_alloc, global_thread_idx, r16, r17);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 0 * precond_tril_num_alloc, global_thread_idx, r16, r18);
    r16 = r18 * r2;
    r8 = fma(r8, r11, r12);
    r8 = 1.0 / r8;
    r16 = r16 * r8;
    r14 = fma(r17, r16, r14);
    ReadIdx2<1024, double, double, double2>(
        precond_diag, 2 * precond_diag_num_alloc, global_thread_idx, r19, r20);
    r19 = fma(r19, r11, r12);
    r19 = fma(r18, r16, r19);
    r19 = 1.0 / r19;
    r18 = r14 * r19;
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 26 * precond_tril_num_alloc, global_thread_idx, r21, r22);
    r22 = fma(r0, r16, r22);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 12 * precond_tril_num_alloc, global_thread_idx, r23, r24);
    r25 = r24 * r5;
    r25 = fma(r9, r25, r22 * r18);
    r26 = r17 * r0;
    r25 = fma(r8, r26, r25);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 30 * precond_tril_num_alloc, global_thread_idx, r27, r28);
    r29 = r2 * r9;
    r30 = r23 * r29;
    r27 = fma(r24, r30, r27);
    r20 = fma(r20, r11, r12);
    r20 = fma(r23, r30, r20);
    r20 = 1.0 / r20;
    r23 = r27 * r20;
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 34 * precond_tril_num_alloc, global_thread_idx, r31, r32);
    r32 = fma(r5, r30, r32);
    r25 = fma(r32, r23, r25);
    r25 = fma(r2, r25, r10);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 44 * precond_tril_num_alloc, global_thread_idx, r10, r26);
    r33 = r24 * r6;
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 36 * precond_tril_num_alloc, global_thread_idx, r34, r35);
    r35 = fma(r6, r30, r35);
    r33 = fma(r35, r23, r9 * r33);
    r33 = fma(r2, r33, r10);
    ReadIdx2<1024, double, double, double2>(
        precond_diag, 4 * precond_diag_num_alloc, global_thread_idx, r10, r34);
    r10 = fma(r10, r11, r12);
    r36 = r17 * r17;
    r36 = fma(r8, r36, r14 * r18);
    r14 = r24 * r24;
    r36 = fma(r9, r14, r36);
    r36 = fma(r27, r23, r36);
    r10 = fma(r2, r36, r10);
    r10 = 1.0 / r10;
    r36 = r33 * r10;
    r7 = fma(r25, r36, r9 * r7);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 56 * precond_tril_num_alloc, global_thread_idx, r27, r14);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 16 * precond_tril_num_alloc, global_thread_idx, r37, r38);
    r39 = r37 * r5;
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 6 * precond_tril_num_alloc, global_thread_idx, r40, r41);
    r42 = r40 * r0;
    r42 = fma(r8, r42, r9 * r39);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 40 * precond_tril_num_alloc, global_thread_idx, r39, r43);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 24 * precond_tril_num_alloc, global_thread_idx, r44, r45);
    r45 = fma(r40, r16, r45);
    r46 = r24 * r37;
    r46 = fma(r9, r46, r45 * r18);
    r47 = r17 * r40;
    r46 = fma(r8, r47, r46);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 32 * precond_tril_num_alloc, global_thread_idx, r48, r49);
    r49 = fma(r37, r30, r49);
    r46 = fma(r49, r23, r46);
    r46 = fma(r2, r46, r39);
    r39 = r46 * r25;
    r42 = fma(r10, r39, r42);
    r47 = r45 * r22;
    r42 = fma(r19, r47, r42);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 46 * precond_tril_num_alloc, global_thread_idx, r50, r51);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 4 * precond_tril_num_alloc, global_thread_idx, r52, r53);
    r15 = fma(r52, r16, r15);
    r54 = r15 * r45;
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 14 * precond_tril_num_alloc, global_thread_idx, r55, r56);
    r57 = r55 * r37;
    r57 = fma(r9, r57, r19 * r54);
    r54 = r52 * r40;
    r57 = fma(r8, r54, r57);
    r28 = fma(r55, r30, r28);
    r58 = r28 * r49;
    r57 = fma(r20, r58, r57);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 38 * precond_tril_num_alloc, global_thread_idx, r59, r60);
    r61 = r24 * r55;
    r61 = fma(r9, r61, r15 * r18);
    r62 = r17 * r52;
    r61 = fma(r8, r62, r61);
    r61 = fma(r28, r23, r61);
    r61 = fma(r2, r61, r59);
    r59 = r61 * r46;
    r57 = fma(r10, r59, r57);
    r57 = fma(r2, r57, r50);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 48 * precond_tril_num_alloc, global_thread_idx, r50, r59);
    r58 = r15 * r22;
    r54 = r55 * r5;
    r54 = fma(r9, r54, r19 * r58);
    r58 = r52 * r0;
    r54 = fma(r8, r58, r54);
    r62 = r28 * r32;
    r54 = fma(r20, r62, r54);
    r63 = r61 * r25;
    r54 = fma(r10, r63, r54);
    r54 = fma(r2, r54, r50);
    r50 = r57 * r54;
    r34 = fma(r34, r11, r12);
    r63 = r15 * r15;
    r62 = r28 * r28;
    r62 = fma(r20, r62, r19 * r63);
    r63 = r52 * r52;
    r58 = r55 * r55;
    r64 = r61 * r61;
    r62 = fma(r8, r63, r62);
    r62 = fma(r9, r58, r62);
    r62 = fma(r10, r64, r62);
    r34 = fma(r2, r62, r34);
    r34 = 1.0 / r34;
    r42 = fma(r34, r50, r42);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 52 * precond_tril_num_alloc, global_thread_idx, r62, r64);
    r58 = r56 * r5;
    r63 = r53 * r0;
    r63 = fma(r8, r63, r9 * r58);
    r44 = fma(r53, r16, r44);
    r58 = r24 * r56;
    r58 = fma(r9, r58, r44 * r18);
    r65 = r17 * r53;
    r58 = fma(r8, r65, r58);
    r48 = fma(r56, r30, r48);
    r58 = fma(r48, r23, r58);
    r58 = fma(r2, r58, r60);
    r60 = r58 * r25;
    r63 = fma(r10, r60, r63);
    r65 = r15 * r44;
    r66 = r55 * r56;
    r66 = fma(r9, r66, r19 * r65);
    r65 = r52 * r53;
    r66 = fma(r8, r65, r66);
    r67 = r61 * r58;
    r66 = fma(r10, r67, r66);
    r68 = r28 * r48;
    r66 = fma(r20, r68, r66);
    r66 = fma(r2, r66, r26);
    r26 = r66 * r34;
    r68 = r48 * r32;
    r63 = fma(r20, r68, r63);
    r67 = r44 * r22;
    r63 = fma(r19, r67, r63);
    r63 = fma(r54, r26, r63);
    r63 = fma(r2, r63, r64);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 50 * precond_tril_num_alloc, global_thread_idx, r64, r67);
    r68 = r56 * r37;
    r60 = r53 * r40;
    r60 = fma(r8, r60, r9 * r68);
    r68 = r58 * r46;
    r60 = fma(r10, r68, r60);
    r65 = r44 * r45;
    r60 = fma(r19, r65, r60);
    r69 = r48 * r49;
    r60 = fma(r20, r69, r60);
    r60 = fma(r57, r26, r60);
    r60 = fma(r2, r60, r67);
    ReadIdx2<1024, double, double, double2>(
        precond_diag, 6 * precond_diag_num_alloc, global_thread_idx, r67, r69);
    r67 = fma(r67, r11, r12);
    r65 = r44 * r44;
    r68 = r58 * r58;
    r68 = fma(r10, r68, r19 * r65);
    r65 = r53 * r53;
    r70 = r56 * r56;
    r71 = r48 * r48;
    r68 = fma(r8, r65, r68);
    r68 = fma(r9, r70, r68);
    r68 = fma(r20, r71, r68);
    r68 = fma(r66, r26, r68);
    r67 = fma(r2, r68, r67);
    r67 = 1.0 / r67;
    r68 = r60 * r67;
    r66 = r49 * r32;
    r42 = fma(r20, r66, r42);
    r42 = fma(r63, r68, r42);
    r42 = fma(r2, r42, r14);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 58 * precond_tril_num_alloc, global_thread_idx, r14, r66);
    r50 = r37 * r6;
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 54 * precond_tril_num_alloc, global_thread_idx, r47, r39);
    r71 = r56 * r6;
    r71 = fma(r58, r36, r9 * r71);
    r70 = r55 * r6;
    r70 = fma(r61, r36, r9 * r70);
    r65 = r28 * r35;
    r70 = fma(r20, r65, r70);
    r70 = fma(r2, r70, r64);
    r64 = r48 * r35;
    r71 = fma(r20, r64, r71);
    r71 = fma(r70, r26, r71);
    r71 = fma(r2, r71, r39);
    r50 = fma(r71, r68, r9 * r50);
    r39 = r57 * r70;
    r50 = fma(r34, r39, r50);
    r64 = r49 * r35;
    r50 = fma(r20, r64, r50);
    r50 = fma(r46, r36, r50);
    r50 = fma(r2, r50, r66);
    r69 = fma(r69, r11, r12);
    r66 = r57 * r57;
    r64 = r45 * r45;
    r64 = fma(r19, r64, r34 * r66);
    r66 = r40 * r40;
    r39 = r37 * r37;
    r65 = r46 * r46;
    r72 = r49 * r49;
    r64 = fma(r8, r66, r64);
    r64 = fma(r9, r39, r64);
    r64 = fma(r60, r68, r64);
    r64 = fma(r10, r65, r64);
    r64 = fma(r20, r72, r64);
    r69 = fma(r2, r64, r69);
    r69 = 1.0 / r69;
    r64 = r50 * r69;
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 60 * precond_tril_num_alloc, global_thread_idx, r72, r65);
    r21 = fma(r41, r16, r21);
    r60 = r24 * r38;
    r60 = fma(r9, r60, r21 * r18);
    r39 = r17 * r41;
    r60 = fma(r8, r39, r60);
    r31 = fma(r38, r30, r31);
    r60 = fma(r31, r23, r60);
    r60 = fma(r2, r60, r43);
    r43 = r60 * r46;
    r39 = r37 * r38;
    r39 = fma(r9, r39, r10 * r43);
    r43 = r40 * r41;
    r39 = fma(r8, r43, r39);
    r66 = r15 * r21;
    r73 = r55 * r38;
    r73 = fma(r9, r73, r19 * r66);
    r66 = r52 * r41;
    r73 = fma(r8, r66, r73);
    r74 = r60 * r61;
    r73 = fma(r10, r74, r73);
    r75 = r31 * r28;
    r73 = fma(r20, r75, r73);
    r73 = fma(r2, r73, r51);
    r51 = r73 * r57;
    r39 = fma(r34, r51, r39);
    r75 = r31 * r49;
    r39 = fma(r20, r75, r39);
    r74 = r21 * r45;
    r39 = fma(r19, r74, r39);
    r66 = r56 * r38;
    r76 = r53 * r41;
    r76 = fma(r8, r76, r9 * r66);
    r66 = r60 * r58;
    r76 = fma(r10, r66, r76);
    r77 = r31 * r48;
    r76 = fma(r20, r77, r76);
    r78 = r21 * r44;
    r76 = fma(r19, r78, r76);
    r76 = fma(r73, r26, r76);
    r76 = fma(r2, r76, r62);
    r39 = fma(r76, r68, r39);
    r39 = fma(r2, r39, r27);
    r27 = r39 * r42;
    r74 = r38 * r5;
    r74 = fma(r9, r74, r69 * r27);
    r27 = r41 * r0;
    r74 = fma(r8, r27, r74);
    r75 = r60 * r25;
    r74 = fma(r10, r75, r74);
    r51 = r76 * r63;
    r74 = fma(r67, r51, r74);
    r43 = r31 * r32;
    r74 = fma(r20, r43, r74);
    r62 = r73 * r54;
    r74 = fma(r34, r62, r74);
    r78 = r21 * r22;
    r74 = fma(r19, r78, r74);
    r74 = fma(r2, r74, r72);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 62 * precond_tril_num_alloc, global_thread_idx, r72, r78);
    r62 = r38 * r6;
    r43 = r73 * r70;
    r43 = fma(r34, r43, r9 * r62);
    r62 = r76 * r71;
    r43 = fma(r67, r62, r43);
    r51 = r31 * r35;
    r43 = fma(r20, r51, r43);
    r43 = fma(r60, r36, r43);
    r43 = fma(r39, r64, r43);
    r43 = fma(r2, r43, r72);
    ReadIdx2<1024, double, double, double2>(
        precond_diag, 8 * precond_diag_num_alloc, global_thread_idx, r72, r51);
    r72 = fma(r72, r11, r12);
    r62 = r21 * r21;
    r75 = r41 * r41;
    r75 = fma(r8, r75, r19 * r62);
    r62 = r38 * r38;
    r27 = r76 * r76;
    r77 = r73 * r73;
    r66 = r60 * r60;
    r79 = r31 * r31;
    r80 = r39 * r39;
    r75 = fma(r9, r62, r75);
    r75 = fma(r67, r27, r75);
    r75 = fma(r34, r77, r75);
    r75 = fma(r10, r66, r75);
    r75 = fma(r20, r79, r75);
    r75 = fma(r69, r80, r75);
    r72 = fma(r2, r75, r72);
    r72 = 1.0 / r72;
    r75 = r43 * r72;
    r80 = r70 * r54;
    r7 = fma(r34, r80, r7);
    r79 = r35 * r32;
    r7 = fma(r20, r79, r7);
    r66 = r71 * r63;
    r7 = fma(r67, r66, r7);
    r7 = fma(r42, r64, r7);
    r7 = fma(r74, r75, r7);
    r7 = fma(r2, r7, r4);
    r51 = fma(r51, r11, r12);
    r4 = r74 * r74;
    r66 = r22 * r22;
    r66 = fma(r19, r66, r72 * r4);
    r4 = r54 * r54;
    r79 = r0 * r0;
    r80 = r5 * r5;
    r77 = r63 * r63;
    r27 = r32 * r32;
    r62 = r25 * r25;
    r81 = r42 * r42;
    r66 = fma(r34, r4, r66);
    r66 = fma(r8, r79, r66);
    r66 = fma(r9, r80, r66);
    r66 = fma(r67, r77, r66);
    r66 = fma(r20, r27, r66);
    r66 = fma(r10, r62, r66);
    r66 = fma(r69, r81, r66);
    r51 = fma(r2, r66, r51);
    r51 = 1.0 / r51;
    r66 = r7 * r51;
    ReadIdx2<1024, double, double, double2>(
        njtr, 10 * njtr_num_alloc, global_thread_idx, r81, r62);
    ReadIdx2<1024, double, double, double2>(
        njtr, 8 * njtr_num_alloc, global_thread_idx, r27, r77);
    ReadIdx2<1024, double, double, double2>(
        njtr, 0 * njtr_num_alloc, global_thread_idx, r80, r79);
    r4 = r38 * r79;
    r4 = fma(r29, r4, r27);
    r27 = r2 * r76;
    ReadIdx2<1024, double, double, double2>(
        njtr, 6 * njtr_num_alloc, global_thread_idx, r82, r83);
    r84 = r56 * r79;
    r84 = fma(r29, r84, r82);
    ReadIdx2<1024, double, double, double2>(
        njtr, 4 * njtr_num_alloc, global_thread_idx, r82, r85);
    r86 = r55 * r79;
    r86 = fma(r29, r86, r85);
    r85 = r2 * r28;
    ReadIdx2<1024, double, double, double2>(
        njtr, 2 * njtr_num_alloc, global_thread_idx, r87, r88);
    r88 = fma(r79, r30, r88);
    r85 = r85 * r88;
    r86 = fma(r20, r85, r86);
    r89 = r2 * r15;
    r87 = fma(r80, r16, r87);
    r89 = r89 * r87;
    r86 = fma(r19, r89, r86);
    r90 = r2 * r61;
    r91 = r24 * r79;
    r91 = fma(r29, r91, r82);
    r82 = r2 * r88;
    r91 = fma(r23, r82, r91);
    r92 = r2 * r87;
    r91 = fma(r18, r92, r91);
    r93 = r17 * r80;
    r93 = r93 * r2;
    r91 = fma(r8, r93, r91);
    r90 = r90 * r91;
    r86 = fma(r10, r90, r86);
    r93 = r52 * r80;
    r93 = r93 * r2;
    r86 = fma(r8, r93, r86);
    r93 = r2 * r86;
    r84 = fma(r26, r93, r84);
    r90 = r2 * r48;
    r90 = r90 * r88;
    r84 = fma(r20, r90, r84);
    r89 = r53 * r80;
    r89 = r89 * r2;
    r84 = fma(r8, r89, r84);
    r85 = r2 * r58;
    r85 = r85 * r91;
    r84 = fma(r10, r85, r84);
    r92 = r2 * r44;
    r92 = r92 * r87;
    r84 = fma(r19, r92, r84);
    r27 = r27 * r84;
    r4 = fma(r67, r27, r4);
    r92 = r2 * r31;
    r92 = r92 * r88;
    r4 = fma(r20, r92, r4);
    r85 = r2 * r73;
    r85 = r85 * r86;
    r4 = fma(r34, r85, r4);
    r89 = r41 * r80;
    r89 = r89 * r2;
    r4 = fma(r8, r89, r4);
    r90 = r2 * r21;
    r90 = r90 * r87;
    r4 = fma(r19, r90, r4);
    r93 = r2 * r39;
    r82 = r37 * r79;
    r82 = fma(r29, r82, r83);
    r83 = r2 * r91;
    r83 = r83 * r46;
    r82 = fma(r10, r83, r82);
    r94 = r2 * r88;
    r94 = r94 * r49;
    r82 = fma(r20, r94, r82);
    r95 = r40 * r80;
    r95 = r95 * r2;
    r82 = fma(r8, r95, r82);
    r96 = r2 * r87;
    r96 = r96 * r45;
    r82 = fma(r19, r96, r82);
    r97 = r2 * r84;
    r82 = fma(r68, r97, r82);
    r98 = r2 * r86;
    r98 = r98 * r57;
    r82 = fma(r34, r98, r82);
    r93 = r93 * r82;
    r4 = fma(r69, r93, r4);
    r98 = r2 * r60;
    r98 = r98 * r91;
    r4 = fma(r10, r98, r4);
    r98 = r2 * r4;
    r98 = fma(r75, r98, r62);
    r62 = r6 * r79;
    r98 = fma(r29, r62, r98);
    r93 = r2 * r86;
    r93 = r93 * r70;
    r98 = fma(r34, r93, r98);
    r90 = r40 * r1;
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 28 * precond_tril_num_alloc, global_thread_idx, r89, r85);
    r89 = fma(r1, r16, r89);
    r85 = r15 * r89;
    r92 = r52 * r1;
    r92 = fma(r8, r92, r19 * r85);
    r85 = r17 * r1;
    r85 = fma(r8, r85, r89 * r18);
    r85 = fma(r2, r85, r13);
    r13 = r61 * r85;
    r92 = fma(r10, r13, r92);
    r92 = fma(r2, r92, r59);
    r59 = r57 * r92;
    r59 = fma(r34, r59, r8 * r90);
    r90 = r53 * r1;
    r90 = fma(r92, r26, r8 * r90);
    r13 = r44 * r89;
    r90 = fma(r19, r13, r90);
    r27 = r58 * r85;
    r90 = fma(r10, r27, r90);
    r90 = fma(r2, r90, r47);
    r47 = r46 * r85;
    r59 = fma(r10, r47, r59);
    r27 = r45 * r89;
    r59 = fma(r19, r27, r59);
    r59 = fma(r90, r68, r59);
    r59 = fma(r2, r59, r14);
    r14 = r0 * r1;
    r27 = r59 * r42;
    r27 = fma(r69, r27, r8 * r14);
    r14 = r90 * r63;
    r27 = fma(r67, r14, r27);
    r47 = r92 * r54;
    r27 = fma(r34, r47, r27);
    r13 = r89 * r22;
    r27 = fma(r19, r13, r27);
    r97 = r85 * r25;
    r27 = fma(r10, r97, r27);
    r96 = r73 * r92;
    r95 = r60 * r85;
    r95 = fma(r10, r95, r34 * r96);
    r96 = r41 * r1;
    r95 = fma(r8, r96, r95);
    r94 = r39 * r59;
    r95 = fma(r69, r94, r95);
    r83 = r21 * r89;
    r95 = fma(r19, r83, r95);
    r99 = r76 * r90;
    r95 = fma(r67, r99, r95);
    r95 = fma(r2, r95, r65);
    r65 = r95 * r74;
    r27 = fma(r72, r65, r27);
    r27 = fma(r2, r27, r78);
    r78 = r27 * r51;
    r65 = fma(r7, r78, r59 * r64);
    r97 = r70 * r92;
    r65 = fma(r34, r97, r65);
    r13 = r71 * r90;
    r65 = fma(r67, r13, r65);
    r65 = fma(r85, r36, r65);
    r65 = fma(r95, r75, r65);
    ReadIdx2<1024, double, double, double2>(
        precond_diag, 10 * precond_diag_num_alloc, global_thread_idx, r13, r97);
    r13 = fma(r13, r11, r12);
    r47 = r92 * r92;
    r14 = r89 * r89;
    r14 = fma(r19, r14, r34 * r47);
    r47 = r1 * r1;
    r99 = r85 * r85;
    r83 = r95 * r95;
    r94 = r90 * r90;
    r96 = r59 * r59;
    r14 = fma(r8, r47, r14);
    r14 = fma(r10, r99, r14);
    r14 = fma(r27, r78, r14);
    r14 = fma(r72, r83, r14);
    r14 = fma(r67, r94, r14);
    r14 = fma(r69, r96, r14);
    r13 = fma(r2, r14, r13);
    r13 = 1.0 / r13;
    r14 = r65 * r13;
    r96 = r2 * r4;
    r96 = r96 * r95;
    r96 = fma(r72, r96, r81);
    r78 = r2 * r78;
    r81 = r5 * r79;
    r81 = fma(r29, r81, r77);
    r77 = r2 * r4;
    r77 = r77 * r74;
    r81 = fma(r72, r77, r81);
    r94 = r2 * r88;
    r94 = r94 * r32;
    r81 = fma(r20, r94, r81);
    r83 = r2 * r82;
    r83 = r83 * r42;
    r81 = fma(r69, r83, r81);
    r27 = r0 * r80;
    r27 = r27 * r2;
    r81 = fma(r8, r27, r81);
    r99 = r2 * r84;
    r99 = r99 * r63;
    r81 = fma(r67, r99, r81);
    r47 = r2 * r87;
    r47 = r47 * r22;
    r81 = fma(r19, r47, r81);
    r100 = r2 * r91;
    r100 = r100 * r25;
    r81 = fma(r10, r100, r81);
    r101 = r2 * r86;
    r101 = r101 * r54;
    r81 = fma(r34, r101, r81);
    r101 = r2 * r84;
    r101 = r101 * r90;
    r96 = fma(r67, r101, r96);
    r100 = r2 * r87;
    r100 = r100 * r89;
    r96 = fma(r19, r100, r96);
    r47 = r2 * r91;
    r47 = r47 * r85;
    r96 = fma(r10, r47, r96);
    r99 = r2 * r86;
    r99 = r99 * r92;
    r96 = fma(r34, r99, r96);
    r27 = r1 * r80;
    r27 = r27 * r2;
    r96 = fma(r8, r27, r96);
    r83 = r2 * r82;
    r83 = r83 * r59;
    r96 = fma(r69, r83, r96);
    r96 = fma(r81, r78, r96);
    r83 = r2 * r82;
    r98 = fma(r64, r83, r98);
    r27 = r2 * r7;
    r27 = r27 * r81;
    r98 = fma(r51, r27, r98);
    r99 = r2 * r88;
    r99 = r99 * r35;
    r98 = fma(r20, r99, r98);
    r47 = r2 * r84;
    r47 = r47 * r71;
    r98 = fma(r67, r47, r98);
    r100 = r2 * r91;
    r98 = fma(r36, r100, r98);
    r98 = fma(r96, r14, r98);
    r11 = fma(r97, r11, r12);
    r97 = r71 * r71;
    r97 = fma(r67, r97, r50 * r64);
    r50 = r7 * r7;
    r12 = r6 * r6;
    r100 = r70 * r70;
    r47 = r35 * r35;
    r97 = fma(r51, r50, r97);
    r97 = fma(r9, r12, r97);
    r97 = fma(r33, r36, r97);
    r97 = fma(r34, r100, r97);
    r97 = fma(r20, r47, r97);
    r97 = fma(r43, r75, r97);
    r97 = fma(r65, r14, r97);
    r11 = fma(r2, r97, r11);
    r11 = 1.0 / r11;
    r11 = r98 * r11;
    r98 = r2 * r11;
    r81 = fma(r81, r51, r98 * r66);
    r14 = fma(r11, r14, r96 * r13);
    r81 = fma(r14, r78, r81);
    r3 = r3 * r81;
    r78 = r40 * r2;
    r13 = r2 * r42;
    r13 = r13 * r81;
    r13 = fma(r82, r69, r69 * r13);
    r96 = r2 * r59;
    r96 = r96 * r14;
    r13 = fma(r69, r96, r13);
    r66 = r2 * r39;
    r75 = fma(r98, r75, r4 * r72);
    r97 = r2 * r74;
    r97 = r97 * r81;
    r75 = fma(r72, r97, r75);
    r65 = r2 * r95;
    r65 = r65 * r14;
    r75 = fma(r72, r65, r75);
    r66 = r66 * r75;
    r13 = fma(r69, r66, r13);
    r13 = fma(r98, r64, r13);
    r78 = r78 * r13;
    r78 = fma(r8, r78, r8 * r3);
    r3 = r41 * r2;
    r3 = r3 * r75;
    r78 = fma(r8, r3, r78);
    r66 = r53 * r2;
    r64 = r2 * r63;
    r64 = r64 * r81;
    r96 = r2 * r76;
    r96 = r96 * r75;
    r96 = fma(r67, r96, r67 * r64);
    r64 = r2 * r90;
    r64 = r64 * r14;
    r96 = fma(r67, r64, r96);
    r69 = r2 * r13;
    r96 = fma(r68, r69, r96);
    r68 = r71 * r67;
    r96 = fma(r98, r68, r96);
    r96 = fma(r84, r67, r96);
    r66 = r66 * r96;
    r78 = fma(r8, r66, r78);
    r68 = r2 * r21;
    r68 = r68 * r75;
    r69 = r2 * r89;
    r69 = r69 * r14;
    r69 = fma(r19, r69, r19 * r68);
    r68 = r2 * r25;
    r68 = r68 * r81;
    r64 = r2 * r58;
    r64 = r64 * r96;
    r64 = fma(r10, r64, r10 * r68);
    r68 = r2 * r46;
    r68 = r68 * r13;
    r64 = fma(r10, r68, r64);
    r65 = r2 * r60;
    r65 = r65 * r75;
    r64 = fma(r10, r65, r64);
    r97 = r2 * r61;
    r72 = r2 * r92;
    r72 = r72 * r14;
    r43 = r2 * r57;
    r43 = r43 * r13;
    r43 = fma(r34, r43, r34 * r72);
    r72 = r2 * r96;
    r43 = fma(r26, r72, r43);
    r26 = r2 * r54;
    r26 = r26 * r81;
    r43 = fma(r34, r26, r43);
    r47 = r70 * r34;
    r43 = fma(r98, r47, r43);
    r100 = r2 * r73;
    r100 = r100 * r75;
    r43 = fma(r34, r100, r43);
    r43 = fma(r86, r34, r43);
    r97 = r97 * r43;
    r64 = fma(r10, r97, r64);
    r100 = r2 * r85;
    r100 = r100 * r14;
    r64 = fma(r10, r100, r64);
    r64 = fma(r36, r98, r64);
    r64 = fma(r91, r10, r64);
    r10 = r2 * r64;
    r69 = fma(r18, r10, r69);
    r18 = r2 * r15;
    r18 = r18 * r43;
    r69 = fma(r19, r18, r69);
    r100 = r2 * r45;
    r100 = r100 * r13;
    r69 = fma(r19, r100, r69);
    r97 = r2 * r44;
    r97 = r97 * r96;
    r69 = fma(r19, r97, r69);
    r65 = r2 * r22;
    r65 = r65 * r81;
    r69 = fma(r19, r65, r69);
    r69 = fma(r87, r19, r69);
    r65 = r17 * r2;
    r65 = r65 * r64;
    r78 = fma(r8, r65, r78);
    r97 = r1 * r2;
    r97 = r97 * r14;
    r78 = fma(r8, r97, r78);
    r100 = r52 * r2;
    r100 = r100 * r43;
    r78 = fma(r8, r100, r78);
    r78 = fma(r69, r16, r78);
    r78 = fma(r80, r8, r78);
    r100 = r35 * r20;
    r97 = r2 * r31;
    r97 = r97 * r75;
    r97 = fma(r20, r97, r98 * r100);
    r100 = r2 * r28;
    r100 = r100 * r43;
    r97 = fma(r20, r100, r97);
    r98 = r2 * r32;
    r98 = r98 * r81;
    r97 = fma(r20, r98, r97);
    r65 = r2 * r64;
    r97 = fma(r23, r65, r97);
    r23 = r2 * r49;
    r23 = r23 * r13;
    r97 = fma(r20, r23, r97);
    r8 = r2 * r48;
    r8 = r8 * r96;
    r97 = fma(r20, r8, r97);
    r97 = fma(r88, r20, r97);
    r8 = r6 * r29;
    r8 = fma(r11, r8, r97 * r30);
    r30 = r5 * r81;
    r8 = fma(r29, r30, r8);
    r23 = r37 * r13;
    r8 = fma(r29, r23, r8);
    r65 = r24 * r64;
    r8 = fma(r29, r65, r8);
    r98 = r55 * r43;
    r8 = fma(r29, r98, r8);
    r100 = r56 * r96;
    r8 = fma(r29, r100, r8);
    r16 = r38 * r75;
    r8 = fma(r29, r16, r8);
    r8 = fma(r79, r9, r8);
    WriteIdx2<1024, double, double, double2>(out_normalized,
                                             0 * out_normalized_num_alloc,
                                             global_thread_idx,
                                             r78,
                                             r8);
    WriteIdx2<1024, double, double, double2>(out_normalized,
                                             2 * out_normalized_num_alloc,
                                             global_thread_idx,
                                             r69,
                                             r97);
    WriteIdx2<1024, double, double, double2>(out_normalized,
                                             4 * out_normalized_num_alloc,
                                             global_thread_idx,
                                             r64,
                                             r43);
    WriteIdx2<1024, double, double, double2>(out_normalized,
                                             6 * out_normalized_num_alloc,
                                             global_thread_idx,
                                             r96,
                                             r13);
    WriteIdx2<1024, double, double, double2>(out_normalized,
                                             8 * out_normalized_num_alloc,
                                             global_thread_idx,
                                             r75,
                                             r81);
    WriteIdx2<1024, double, double, double2>(out_normalized,
                                             10 * out_normalized_num_alloc,
                                             global_thread_idx,
                                             r14,
                                             r11);
  };
}

void ThinPrismFisheyeCalibNormalize(double* precond_diag,
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
  ThinPrismFisheyeCalibNormalizeKernel<<<n_blocks, 1024>>>(
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