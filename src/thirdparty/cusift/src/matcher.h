#pragma once
#include <vector>

struct Match { int i, j; float dist2; };

// Metric space in which the ratio test is evaluated. Top-2 SELECTION is
// unaffected (cosine, unit-vector L2 and angle are monotone in one another, so
// the argmax is identical), and a max-distance THRESHOLD converts exactly
// between the spaces. Only the RATIO test genuinely differs, because a ratio of
// angles is not a ratio of distances.
//
//   kL2      - ratio of unit-vector L2 distances. What COLMAP's CPU matcher
//              does (sift.cc, FindBestMatchesOneWayIndex) and what VLFeat's
//              vl_ubcmatch does. Cheapest: d^2 = 2-2*cos is one FMA on the
//              value the GEMM already produced.
//   kAngular - ratio of angles, via acos. Behaviourally what SiftGPU does. Costs
//              a transcendental, but only in the O(n) reduction epilogue, never
//              in the O(n1*n2) inner loop.
//
// LICENSING: kAngular replicates SiftGPU's observable BEHAVIOUR only. No
// SiftGPU code is copied or adapted here -- that would defeat the entire point
// of this library, which exists to be a commercially-usable replacement for
// SiftGPU's non-commercial licence. This implementation works from true
// normalised cosines produced by cuBLAS (SiftGPU instead scales raw integer dot
// products by a hardcoded reciprocal), clamps both ends of the domain, and
// evaluates the test in a host-side epilogue over separately-reduced top-2
// arrays. The shared element is the mathematical idea -- compare a ratio of
// angles, and test both directions -- not any expression of it.
//
// NOTE this also changes the UNITS of any max-distance threshold applied by the
// caller: radians under kAngular, unit-vector L2 distance under kL2.
enum GemmDistanceType { kGemmDistanceL2 = 0, kGemmDistanceAngular = 1 };

// h_desc1/h_desc2: row-major (n x 128) float, L2-normalized.
// ratio: Lowe ratio on distances (e.g. 0.8; SiftMatchGPU default distmax
//        semantics differ — see bench notes). mutual: cross-check.
// Returns kernel time in ms; matches appended to `out`.
float MatchDescriptors(const float* h_desc1, int n1, const float* h_desc2, int n2,
                       float ratio, bool mutual, std::vector<Match>& out);

// GEMM-class matcher (cuBLAS SGEMM + tiled top-2 reductions). Same semantics
// as MatchDescriptors; ~SiftMatchGPU speed class.
//
// reverseRatio (only meaningful when `mutual` is set) selects how strong the
// cross-check is:
//   true  - also apply the ratio test in the reverse (column) direction, which
//           is what COLMAP's CPU matcher and SiftGPU both do. Costs nothing
//           extra on the device: the column runner-up falls out of the same
//           reduction that already computes the column best.
//   false - require only mutual-best-index agreement (the original behaviour of
//           this function). Cheaper on paper but materially weaker on
//           repetitive structure; see ColTop2's comment in matcher_gemm.cu.
// `out[k].dist2` is the squared L2 distance in the INPUT descriptors' own scale
// (so ~512^2-scaled for COLMAP descriptors, or unit-scale for unit vectors),
// regardless of distanceType. Divide by sqDescriptorNorm for unit scale.
// maxDistance is the absolute threshold on the best match, in the units the
// chosen distanceType implies: RADIANS under kAngular, and descriptor-scale
// distance under kL2 (i.e. the same units the input descriptors are in).
// sqDescriptorNorm is the squared norm the descriptors are nominally scaled to;
// kAngular divides the raw inner product by it to form the pseudo-cosine, which
// is how both of COLMAP's references do it (a FIXED constant, not per-row
// norms). Pass 512*512 for COLMAP-scale descriptors, or 1.0 for unit vectors.
float MatchDescriptorsGemm(const float* h_desc1, int n1, const float* h_desc2, int n2,
                           float ratio, bool mutual, std::vector<Match>& out,
                           bool reverseRatio = true,
                           int distanceType = kGemmDistanceAngular,
                           float maxDistance = 3.4028235e38f,
                           float sqDescriptorNorm = 512.0f * 512.0f);

// Same matcher, but taking descriptors in their native uint8 form. Prefer this
// whenever the caller already holds uint8 descriptors (COLMAP always does):
// the host->device transfer carries 128 B per descriptor instead of 512 B, and
// the widening runs on the device. Identical results to the float entry point --
// the two share one implementation after the upload.
//
// The uint8 device buffer this fills is laid out row-major with 128 contiguous
// bytes per descriptor (lda = 128, a multiple of 4), which is what a
// cublasGemmEx CUDA_R_8I -> CUDA_R_32I path needs, so that variant can be added
// behind a flag without re-plumbing the upload.
// Sizes the thread-local device buffers up front, for `maxNumDescriptors` per
// side. Optional but worth calling once per matching thread when an upper bound
// is known: the buffers are otherwise grown on demand, and every growth does a
// batch of cudaFree/cudaMalloc. cudaFree synchronises the WHOLE DEVICE, so in a
// pipelined host (COLMAP runs verification and I/O alongside matching) that
// stalls unrelated queued work -- a cost an isolated benchmark cannot see.
void ReserveGemmMatcher(int maxNumDescriptors);

float MatchDescriptorsGemmU8(const unsigned char* h_desc1, int n1,
                             const unsigned char* h_desc2, int n2,
                             float ratio, bool mutual, std::vector<Match>& out,
                             bool reverseRatio = true,
                             int distanceType = kGemmDistanceAngular,
                             float maxDistance = 3.4028235e38f,
                             float sqDescriptorNorm = 512.0f * 512.0f,
                             // fused=true selects the single-pass fused
                             // kernel (matcher_fused.cu), which never
                             // materialises the similarity matrix. Results
                             // must be bit-identical to the GEMM path; that
                             // equivalence is the oracle for both, and also
                             // a tripwire if cuBLAS ever routes SGEMM
                             // through TF32 tensor cores.
                             bool fused = false);
