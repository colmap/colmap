// SPDX-License-Identifier: MIT
// Copyright (c) 2017 Mårten Björkman (CudaSift; see ../LICENSE.CudaSift)
// Copyright (c) 2026 Yonatan Simson (see ../LICENSE)

//********************************************************//
// CUDA SIFT extractor by Mårten Björkman aka Celebrandil //
//********************************************************//  

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <array>
#include <vector>
#include <map>
#include <mutex>
#include <string>
#include "cudautils.h"

#include "cudaImage.h"
#include "cudaSift.h"
#include "cudaSiftD.h"

#include "cudaSiftD.cu"










   






 


//======= VLFeat-schedule pyramid + descriptor (clean-room from          ====//
//======= colmap/src/thirdparty/VLFeat/sift.c, BSD; not SiftGPU)          ====//
// Builds a scale-space pyramid referenced to VLFeat's own absolute-sigma
// schedule (not CudaSift's internal blur bookkeeping), so octave/level
// assignment follows VLFeat exactly.
//
// Formulas verified directly against colmap/src/thirdparty/VLFeat/sift.c:
//   vl_sift_new (874-935):                sigman=0.5, sigma0=1.6*sigmak,
//                                          sigmak=2^(1/S), s_min=-1, s_max=S+1
//   vl_sift_process_first_octave (977-1062): base-level blur-up, per-level
//                                          incremental blur dsigma0*sigmak^s
//   vl_sift_process_next_octave (1097-1136): pure 2x decimation of the
//                                          previous octave's level s_min+S,
//                                          NO extra blur needed (sa==sb by
//                                          construction -- verified exactly)
//   copy_and_downsample (836-851):        pure decimation, no anti-alias blur
//                                          (NOT the same as cusift's own
//                                          ScaleDown, which bakes one in)
//   update_gradient (1447-1484):          central differences, one-sided at
//                                          borders, angle wrapped to [0,2pi)

// Gaussian blur with a radius adapted to sigma -- see cudaSiftD.cu's
// GaussianBlurHorizontalKernel/GaussianBlurVerticalKernel comment for why a
// fixed radius-4 blur under-blurs the VLFeat sigma schedule.
// Memoized slot in the constant kernel bank for `sigma`. Keyed on the
// exact float bit pattern, so a hit reuses bit-identical weights; a miss
// computes them and appends. ~8 distinct sigmas per process, so all
// misses happen on the first image and every later blur is upload-free.
//
// One slot table per device, shared by every thread on it and guarded by a
// mutex. d_AdaptiveGaussKernel is a per-device __constant__ symbol, so the
// table must be per device (a slot recorded on device 0 was never uploaded to
// device 1) and must not be per thread: two threads on one device would each
// start at offset 0 and overwrite each other's weights whenever their sigma
// sequences differ, e.g. with different first_octave settings.
//
// A slot is recorded only after its upload has been issued, under the lock, so
// a thread can never be handed an offset whose weights are not on their way.
// The upload and every blur kernel run on the legacy default stream, which is
// shared by all host threads on a device, so the kernels that read a slot are
// ordered after the upload that wrote it.
struct GaussKernelBank {
  std::vector<std::pair<uint32_t, int>> slots;   // (sigma bits, offset)
  int used = 0;
};

static int GaussKernelSlot(float sigma, int radius)
{
  int dev = 0;
  safeCall(cudaGetDevice(&dev));
  static std::mutex mutex;
  static std::map<int, GaussKernelBank> banks;
  std::lock_guard<std::mutex> lock(mutex);
  GaussKernelBank &bank = banks[dev];
  std::vector<std::pair<uint32_t, int>> &slots = bank.slots;
  int &used = bank.used;

  uint32_t key;
  std::memcpy(&key, &sigma, sizeof(key));
  for (const auto &s : slots)
    if (s.first == key) return s.second;

  int n = 2 * radius + 1;
  if (used + n > GAUSS_BANK_FLOATS) {
    throw std::runtime_error(
        "cusift: Gaussian kernel bank exhausted (" + std::to_string(slots.size()) +
        " slots, " + std::to_string(used) + " floats)");
  }
  std::vector<float> kernel(n);
  float sum = 0.0f;
  for (int j = -radius; j <= radius; j++) {
    kernel[j + radius] = std::exp(-(float)(j * j) / (2.0f * sigma * sigma));
    sum += kernel[j + radius];
  }
  for (auto &v : kernel) v /= sum;
  safeCall(cudaMemcpyToSymbol(d_AdaptiveGaussKernel, kernel.data(),
                              kernel.size() * sizeof(float), used * sizeof(float)));
  int offset = used;
  used += n;
  slots.emplace_back(key, offset);
  return offset;
}

static void AdaptiveGaussianBlur(CudaImage &dst, CudaImage &src, float sigma)
{
  int radius = std::min((int)std::ceil(4.0f * sigma), MAX_GAUSS_RADIUS);
  radius = std::max(radius, 1);
  int kOffset = GaussKernelSlot(sigma, radius);

  CudaImage tmp;
  tmp.Allocate(src.width, src.height, iAlignUp(src.width, 128), false, NULL);
  dim3 threads(16, 16);
  dim3 blocks(iDivUp(src.width, 16), iDivUp(src.height, 16));
  GaussianBlurHorizontalKernel<<<blocks, threads>>>(tmp.d_data, src.d_data,
      src.width, tmp.pitch, src.pitch, src.height, radius, kOffset);
  checkMsg("GaussianBlurHorizontalKernel() execution failed\n");
  GaussianBlurVerticalKernel<<<blocks, threads>>>(dst.d_data, tmp.d_data,
      dst.width, dst.pitch, tmp.pitch, dst.height, radius, kOffset);
  checkMsg("GaussianBlurVerticalKernel() execution failed\n");
}

static void Decimate2x(CudaImage &dst, CudaImage &src)
{
  dim3 threads(16, 16);
  dim3 blocks(iDivUp(dst.width, 16), iDivUp(dst.height, 16));
  PureDecimate2xKernel<<<blocks, threads>>>(dst.d_data, src.d_data,
      dst.width, dst.pitch, dst.height, src.pitch);
  checkMsg("PureDecimate2xKernel() execution failed\n");
}

// Test-only: 2x upsample matching VLFeat's copy_and_upsample_rows (see
// Upsample2xHorizontalKernel/Upsample2xVerticalKernel comment). Used solely
// to test the o_min=-1 hypothesis (see plan) -- NOT wired into any
// production path.
static void Upsample2x(CudaImage &dst, CudaImage &src)
{
  CudaImage tmp;
  tmp.Allocate(2*src.width, src.height, iAlignUp(2*src.width, 128), false, NULL);
  dim3 threadsH(16, 16);
  dim3 blocksH(iDivUp(2*src.width, 16), iDivUp(src.height, 16));
  Upsample2xHorizontalKernel<<<blocksH, threadsH>>>(tmp.d_data, src.d_data,
      src.width, tmp.pitch, src.pitch, src.height);
  checkMsg("Upsample2xHorizontalKernel() execution failed\n");
  dim3 threadsV(16, 16);
  dim3 blocksV(iDivUp(dst.width, 16), iDivUp(2*src.height, 16));
  Upsample2xVerticalKernel<<<blocksV, threadsV>>>(dst.d_data, tmp.d_data,
      dst.width, dst.pitch, tmp.pitch, src.height);
  checkMsg("Upsample2xVerticalKernel() execution failed\n");
}

// Debug-only: poison a gradient level slice with signaling NaN -- see
// FillPlaneKernel's comment. If a Debug build still passes the accuracy tests
// with every skipped plane poisoned, nothing reads them.
static void PoisonGradientLevel(CudaImage &combined, int levelIdx, int w, int h, int p)
{
  uint32_t snanBits = 0x7FA00000u;   // signaling NaN
  float snan;
  std::memcpy(&snan, &snanBits, sizeof(snan));
  dim3 threads(16, 16);
  dim3 blocks(iDivUp(w, 16), iDivUp(h, 16));
  FillPlaneKernel<<<blocks, threads>>>(combined.d_data + (size_t)levelIdx * h * p,
                                       w, p, h, snan);
  checkMsg("FillPlaneKernel() execution failed\n");
}

static void ComputeGradientPlaneFor(CudaImage &level, CudaImage &gradMag, CudaImage &gradAng)
{
  dim3 threads(16, 16);
  dim3 blocks(iDivUp(level.width, 16), iDivUp(level.height, 16));
  ComputeGradientPlaneKernel<<<blocks, threads>>>(level.d_data, level.width, level.pitch, level.height,
      gradMag.d_data, gradAng.d_data);
  checkMsg("ComputeGradientPlaneKernel() execution failed\n");
}

// One VLFeat octave: nLevels = numScales+3 Gaussian levels (s = s_min..s_max,
// s_min=-1) stored as contiguous slices of one buffer each for the Gaussian
// image, gradient magnitude, and gradient angle (slice stride = height*pitch).
struct VLFeatOctaveBuffers {
  int width, height, pitch;
  int nLevels;
  CudaImage gauss;    // nLevels slices, self-allocated as one block below
  CudaImage gradMag;  // see ComputeGradientPlaneKernel for why these are
  CudaImage gradAng;  // NOT interleaved into one float2 plane
};

// Non-owning per-level view into a combined (nLevels-stacked) buffer -- slice
// i occupies rows [i*h, (i+1)*h) of the combined image.
static CudaImage LevelView(CudaImage &combined, int levelIdx, int w, int h, int p)
{
  CudaImage v;
  v.Allocate(w, h, p, false, combined.d_data + (size_t)levelIdx * h * p);
  return v;
}

// Gradient planes are computed only for the level indices a consumer can
// actually address. Keypoint levels are `is` in [0, numScales-1] and every
// consumer indexes gradients as `is - s_min` (VLFeatKp::level), so indices
// 1..numScales are the whole reachable set: the s_min and the two top levels exist in the Gaussian pyramid for DoG
// purposes but can never host a keypoint. That is 3 of 6 planes for
// numScales=3 -- half the gradient bandwidth of the pyramid stage, and at octave
// -1 those are 33 MB planes.
//
// The skipped planes are left UNINITIALIZED rather than zeroed (zeroing would
// spend the bandwidth this saves); the index invariant above is what keeps them
// unreachable.
static void BuildVLFeatOctave(VLFeatOctaveBuffers &oct, CudaImage *prevTopLevel /*NULL for octave 0*/,
                              CudaImage *img01 /*only used for octave 0*/, int numScales)
{
  const int s_min = -1, s_max = numScales + 1;
  const int nLevels = s_max - s_min + 1;  // = numScales+3
  const float sigmak = std::pow(2.0f, 1.0f / numScales);
  const float sigma0 = 1.6f * sigmak;
  const float dsigma0 = sigma0 * std::sqrt(1.0f - 1.0f / (sigmak * sigmak));

  int w, h;
  if (prevTopLevel) { w = prevTopLevel->width / 2; h = prevTopLevel->height / 2; }
  else              { w = img01->width;            h = img01->height; }

  // NOTE: CudaImage::Allocate with devMem=NULL self-allocates via
  // cudaMallocPitch, which OVERWRITES `pitch` with whatever CUDA actually
  // returns -- the requested value passed in is discarded. Must read back
  // the true pitch after allocation and use THAT everywhere below, not the
  // iAlignUp guess, or level-slice addressing silently corrupts.
  oct.gauss.Allocate(w, h * nLevels, iAlignUp(w, 128), false, NULL);
  oct.gradMag.Allocate(w, h * nLevels, iAlignUp(w, 128), false, NULL);
  oct.gradAng.Allocate(w, h * nLevels, iAlignUp(w, 128), false, NULL);
  int p = oct.gauss.pitch;
  // gradMag/gradAng were allocated with the identical (w, h*nLevels) request,
  // so cudaMallocPitch is expected to return the same pitch deterministically.
  if (oct.gradMag.pitch != p || oct.gradAng.pitch != p) {
    throw std::runtime_error("cusift: unexpected pitch mismatch in BuildVLFeatOctave");
  }
  oct.width = w; oct.height = h; oct.pitch = p; oct.nLevels = nLevels;

  CudaImage lvl0 = LevelView(oct.gauss, 0, w, h, p);
  if (prevTopLevel) {
    Decimate2x(lvl0, *prevTopLevel);
    // sa == sb by construction at octave transitions (verified against
    // vl_sift_process_next_octave: sigma0*sigmak^s_min == sigma0*sigmak^(s_best-S)
    // when s_best=s_min+numScales) -- no extra blur needed here.
  } else {
    // AdaptiveGaussianBlur takes explicit separate src/dst pitches, so no
    // same-pitch assumption between img01 and lvl0 is needed here.
    float sa = sigma0 * std::pow(sigmak, (float)s_min);
    float sb = 0.5f;  // sigman, o_min=0 (non-upsampled)
    if (sa > sb) AdaptiveGaussianBlur(lvl0, *img01, std::sqrt(sa * sa - sb * sb));
    else         AdaptiveGaussianBlur(lvl0, *img01, 0.001f);
  }
  for (int s = s_min + 1; s <= s_max; s++) {
    CudaImage prev = LevelView(oct.gauss, s - s_min - 1, w, h, p);
    CudaImage cur = LevelView(oct.gauss, s - s_min, w, h, p);
    float sd = dsigma0 * std::pow(sigmak, (float)s);
    AdaptiveGaussianBlur(cur, prev, std::max(sd, 0.001f));
  }
  const int gradLo = 1;          // see the note above BuildVLFeatOctave
  const int gradHi = numScales;
  for (int i = gradLo; i <= gradHi; i++) {
    CudaImage lvl = LevelView(oct.gauss, i, w, h, p);
    CudaImage gm = LevelView(oct.gradMag, i, w, h, p);
    CudaImage ga = LevelView(oct.gradAng, i, w, h, p);
    ComputeGradientPlaneFor(lvl, gm, ga);
  }
#if !defined(NDEBUG) || defined(CUSIFT_POISON_SKIPPED_GRADIENTS)
  // Debug builds, or an opt-in Release build used to TEST the invariant:
  //   cmake -B build-poison -DCMAKE_BUILD_TYPE=Release \
  //         -DCMAKE_CUDA_FLAGS=-DCUSIFT_POISON_SKIPPED_GRADIENTS
  // keeps --use_fast_math while poisoning every skipped plane. If the accuracy
  // tests still pass, nothing reads them.
  for (int i = 0; i < nLevels; i++) {
    if (i >= gradLo && i <= gradHi) continue;
    PoisonGradientLevel(oct.gradMag, i, w, h, p);
    PoisonGradientLevel(oct.gradAng, i, w, h, p);
  }
#endif
}

// Test-only: build octave -1 (VLFeat's real o_min=-1 default, sift.c:1017-1029)
// by 2x-upsampling img01 then blurring with sb=sigman*2^(-o_min)=1.0 (NOT the
// o_min=0 base blur's sb=0.5 -- verified sift.c:1046, sb = sigman*pow(2,-o_min)).
static void BuildUpsampledOctaveMinus1(VLFeatOctaveBuffers &oct, CudaImage &img01, int numScales)
{
  const int s_min = -1, s_max = numScales + 1;
  const int nLevels = s_max - s_min + 1;
  const float sigmak = std::pow(2.0f, 1.0f / numScales);
  const float sigma0 = 1.6f * sigmak;
  const float dsigma0 = sigma0 * std::sqrt(1.0f - 1.0f / (sigmak * sigmak));

  int w = img01.width * 2, h = img01.height * 2;
  oct.gauss.Allocate(w, h * nLevels, iAlignUp(w, 128), false, NULL);
  oct.gradMag.Allocate(w, h * nLevels, iAlignUp(w, 128), false, NULL);
  oct.gradAng.Allocate(w, h * nLevels, iAlignUp(w, 128), false, NULL);
  int p = oct.gauss.pitch;
  if (oct.gradMag.pitch != p || oct.gradAng.pitch != p) {
    throw std::runtime_error("cusift: unexpected pitch mismatch in BuildUpsampledOctaveMinus1");
  }
  oct.width = w; oct.height = h; oct.pitch = p; oct.nLevels = nLevels;

  CudaImage img2x;
  img2x.Allocate(w, h, iAlignUp(w, 128), false, NULL);
  Upsample2x(img2x, img01);

  CudaImage lvl0 = LevelView(oct.gauss, 0, w, h, p);
  float sa = sigma0 * std::pow(sigmak, (float)s_min);
  float sb = 1.0f;  // sigman(0.5) * 2^(-o_min) = 0.5*2 = 1.0 for o_min=-1
  if (sa > sb) AdaptiveGaussianBlur(lvl0, img2x, std::sqrt(sa * sa - sb * sb));
  else         AdaptiveGaussianBlur(lvl0, img2x, 0.001f);
  for (int s = s_min + 1; s <= s_max; s++) {
    CudaImage prev = LevelView(oct.gauss, s - s_min - 1, w, h, p);
    CudaImage cur = LevelView(oct.gauss, s - s_min, w, h, p);
    float sd = dsigma0 * std::pow(sigmak, (float)s);
    AdaptiveGaussianBlur(cur, prev, std::max(sd, 0.001f));
  }
  const int gradLo = 1;          // see the note above BuildVLFeatOctave
  const int gradHi = numScales;
  for (int i = gradLo; i <= gradHi; i++) {
    CudaImage lvl = LevelView(oct.gauss, i, w, h, p);
    CudaImage gm = LevelView(oct.gradMag, i, w, h, p);
    CudaImage ga = LevelView(oct.gradAng, i, w, h, p);
    ComputeGradientPlaneFor(lvl, gm, ga);
  }
#if !defined(NDEBUG) || defined(CUSIFT_POISON_SKIPPED_GRADIENTS)
  // Debug builds, or an opt-in Release build used to TEST the invariant:
  //   cmake -B build-poison -DCMAKE_BUILD_TYPE=Release \
  //         -DCMAKE_CUDA_FLAGS=-DCUSIFT_POISON_SKIPPED_GRADIENTS
  // keeps --use_fast_math while poisoning every skipped plane. If the accuracy
  // tests still pass, nothing reads them.
  for (int i = 0; i < nLevels; i++) {
    if (i >= gradLo && i <= gradHi) continue;
    PoisonGradientLevel(oct.gradMag, i, w, h, p);
    PoisonGradientLevel(oct.gradAng, i, w, h, p);
  }
#endif
}

// Detection. DetectedKeypoint (declared in cudaSift.h) mirrors
// VlSiftKeypoint's own fields exactly (o, is = coarse detection level, ix/iy
// = coarse pixel, x/y/sigma = refined absolute-pixel-unit position/scale,
// s = continuous refined level). `is` is carried as-is from the coarse scan
// through refinement (VLFeat never changes it during Newton iteration, only
// x/y can relocate -- verified directly against sift.c:1267-1427) so it's
// available as a first-class field for orientation/descriptor, never
// re-derived from sigma.

// Every thread_local holder below frees its memory in its destructor, which runs
// when the owning extraction thread exits: COLMAP starts new extraction threads
// for every extraction run, so without it each run leaked these buffers. Errors
// are ignored there, because at process exit the CUDA runtime may already be
// shutting down and a failed free cannot be acted on.
// Packed scratch for the VLFeat pipeline's kernel I/O (see VLFeatKp in
// cudaSift.h): grow-only, with pinned host memory, reused across octaves and
// images, thread_local so each extraction thread has its own. 20 B/row for
// keypoints and 512 B/row for descriptors.
struct PackedScratch {
  VLFeatKp *h_kps = nullptr, *d_kps = nullptr;
  VLFeatOri *h_ori = nullptr, *d_ori = nullptr;
  float *h_desc = nullptr, *d_desc = nullptr;
  int cap = 0;
  PackedScratch() = default;
  PackedScratch(const PackedScratch &) = delete;
  PackedScratch &operator=(const PackedScratch &) = delete;
  ~PackedScratch() { Release(/*check=*/false); }
  // Frees every buffer and nulls it, so a throw while regrowing leaves an
  // empty scratch (cap 0) rather than freed pointers behind an old cap.
  void Release(bool check) {
    cap = 0;
    FreeHost(&h_kps, check); FreeDevice(&d_kps, check);
    FreeHost(&h_ori, check); FreeDevice(&d_ori, check);
    FreeHost(&h_desc, check); FreeDevice(&d_desc, check);
  }

 private:
  template <typename T> static void FreeHost(T **p, bool check) {
    T *const old = *p;
    *p = nullptr;
    if (!old) return;
    const cudaError_t err = cudaFreeHost(old);
    if (check) safeCall(err);
  }
  template <typename T> static void FreeDevice(T **p, bool check) {
    T *const old = *p;
    *p = nullptr;
    if (!old) return;
    const cudaError_t err = cusiftFree(old);
    if (check) safeCall(err);
  }
};

static PackedScratch &Packed(int n, bool wantDesc)
{
  static thread_local PackedScratch p;
  if (n > p.cap) {
    p.Release(/*check=*/true);
    safeCall(cudaMallocHost((void **)&p.h_kps, (size_t)n * sizeof(VLFeatKp)));
    safeCall(cusiftMallocAsync((void **)&p.d_kps, (size_t)n * sizeof(VLFeatKp)));
    safeCall(cudaMallocHost((void **)&p.h_ori, (size_t)n * sizeof(VLFeatOri)));
    safeCall(cusiftMallocAsync((void **)&p.d_ori, (size_t)n * sizeof(VLFeatOri)));
    safeCall(cudaMallocHost((void **)&p.h_desc, (size_t)n * 128 * sizeof(float)));
    safeCall(cusiftMallocAsync((void **)&p.d_desc, (size_t)n * 128 * sizeof(float)));
    p.cap = n;
  }
  (void)wantDesc;
  return p;
}


// Candidates and refined keypoints are appended on the GPU through atomic
// counters, so their order depends on thread scheduling and changes from run
// to run. Sort each octave's keypoints into a fixed order: by integer location
// (level, row, column), the order VLFeat's own raster scan produces, with the
// refined values as a tie-break for the rare candidates that converge to the
// same location. Every later stage processes keypoints in this order, which
// makes the extractor's output order reproducible.
static void SortOctaveKeypoints(std::vector<DetectedKeypoint> &kps, size_t begin)
{
  std::sort(kps.begin() + begin, kps.end(),
            [](const DetectedKeypoint &a, const DetectedKeypoint &b) {
              if (a.is != b.is) return a.is < b.is;
              if (a.iy != b.iy) return a.iy < b.iy;
              if (a.ix != b.ix) return a.ix < b.ix;
              if (a.s != b.s) return a.s < b.s;
              if (a.y != b.y) return a.y < b.y;
              return a.x < b.x;
            });
}

// Full detection pipeline for one octave: DoG cube (GPU) -> coarse extrema
// scan (GPU) -> serial Newton refinement + edge test (CPU, per-candidate,
// matching VLFeat's own CPU-serial design -- see FindDogExtremaKernel's
// comment for why this split, not the refinement, runs on the GPU).
static void DetectOctaveVLFeat(VLFeatOctaveBuffers &oct, int vo,
                               float peakThresh, float edgeThresh, int numScales,
                               std::vector<DetectedKeypoint> &out)
{
  const size_t outBegin = out.size();
  const int s_min = -1, s_max = numScales + 1;
  int w = oct.width, h = oct.height, p = oct.pitch;
  int nDogLevels = oct.nLevels - 1;

  CudaImage dogCombined;
  dogCombined.Allocate(w, h*nDogLevels, iAlignUp(w, 128), false, NULL);
  int dp = dogCombined.pitch;
  {
  for (int i = 0; i < nDogLevels; i++) {
    CudaImage a = LevelView(oct.gauss, i, w, h, p);
    CudaImage b = LevelView(oct.gauss, i+1, w, h, p);
    CudaImage d = LevelView(dogCombined, i, w, h, dp);
    dim3 threads(16, 16);
    dim3 blocks(iDivUp(w, 16), iDivUp(h, 16));
    ComputeDogPlaneKernel<<<blocks, threads>>>(a.d_data, b.d_data, w, p, h, d.d_data);
    checkMsg("ComputeDogPlaneKernel() execution failed\n");
  }
  }
  const int maxCandidates = 200000;
  int *d_CandX, *d_CandY, *d_CandS; unsigned int *d_CandCount;
  safeCall(cusiftMallocAsync(&d_CandX, maxCandidates*sizeof(int)));
  safeCall(cusiftMallocAsync(&d_CandY, maxCandidates*sizeof(int)));
  safeCall(cusiftMallocAsync(&d_CandS, maxCandidates*sizeof(int)));
  safeCall(cusiftMallocAsync(&d_CandCount, sizeof(unsigned int)));
  safeCall(cudaMemset(d_CandCount, 0, sizeof(unsigned int)));

  unsigned int candCount = 0;
  {
  dim3 threads(16, 16, 1);
  dim3 blocks(iDivUp(w-2, 16), iDivUp(h-2, 16), std::max(nDogLevels-2, 1));
  FindDogExtremaKernel<<<blocks, threads>>>(dogCombined.d_data, w, dp, h, nDogLevels,
      peakThresh, d_CandX, d_CandY, d_CandS, maxCandidates, d_CandCount);
  checkMsg("FindDogExtremaKernel() execution failed\n");

  safeCall(cudaMemcpy(&candCount, d_CandCount, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  candCount = std::min(candCount, (unsigned int)maxCandidates);
  }

  const float sigmak = std::pow(2.0f, 1.0f/numScales);
  const float sigma0 = 1.6f*sigmak;

  {
    // Refine every candidate on the GPU, straight out of the pitched DoG
    // buffer, and bring back only the survivors.
    std::vector<DetectedKeypoint> accepted;
    if (candCount > 0) {
      DetectedKeypoint *d_Kp = nullptr; unsigned int *d_KpCount = nullptr;
      safeCall(cusiftMallocAsync(&d_Kp, (size_t)candCount*sizeof(DetectedKeypoint)));
      safeCall(cusiftMallocAsync(&d_KpCount, sizeof(unsigned int)));
      safeCall(cudaMemset(d_KpCount, 0, sizeof(unsigned int)));
      int threadsR = 128, blocksR = iDivUp((int)candCount, threadsR);
      RefineKeypointsKernel<<<blocksR, threadsR>>>(dogCombined.d_data, w, h, dp, h*dp,
          nDogLevels, s_min, s_max, d_CandX, d_CandY, d_CandS, candCount,
          peakThresh, edgeThresh, d_Kp, d_KpCount);
      checkMsg("RefineKeypointsKernel() execution failed\n");
      unsigned int nAcc = 0;
      safeCall(cudaMemcpy(&nAcc, d_KpCount, sizeof(unsigned int), cudaMemcpyDeviceToHost));
      accepted.resize(nAcc);
      if (nAcc)
        safeCall(cudaMemcpy(accepted.data(), d_Kp, (size_t)nAcc*sizeof(DetectedKeypoint),
                            cudaMemcpyDeviceToHost));
      safeCall(cusiftFree(d_Kp)); safeCall(cusiftFree(d_KpCount));
    }
    safeCall(cusiftFree(d_CandX)); safeCall(cusiftFree(d_CandY));
    safeCall(cusiftFree(d_CandS)); safeCall(cusiftFree(d_CandCount));
    out.reserve(out.size() + accepted.size());
    for (DetectedKeypoint kp : accepted) {
      kp.o = vo;
      kp.sigma = sigma0 * std::pow(sigmak, kp.s);
      out.push_back(kp);
    }
    SortOctaveKeypoints(out, outBegin);
  }
}



// Full pipeline: detect -> orient -> descriptor, one pass per octave, with
// is/x/y/sigma threaded natively throughout. Each keypoint is expanded into up to
// `maxNumOrientations` rows (COLMAP default 2), one per surviving angle -- mirrors COLMAP's own
// `SiftCPUFeatureExtractor::Extract` orientation-expansion loop
// (sift.cc:210-286), not something invented here. Output x/y/sigma are
// ABSOLUTE image coordinates, orientation in RADIANS (VLFeat convention),
// descriptor already UBC-ordered + L2/clamp/L2-normalized. No +0.5 applied --
// that's the output/DB-boundary's job, same rule as detection.
void ExtractSiftFullVLFeat(CudaImage &img, int numOctaves, float peakThresh, float edgeThresh,
                           int maxNumOrientations, int firstOctave, int maxNumFeatures,
                           std::vector<float> &outX, std::vector<float> &outY,
                           std::vector<float> &outSigma, std::vector<float> &outOrientation,
                           std::vector<std::array<float, 128>> &outDescriptors)
{
  const int numScales = NUM_SCALES;
  // firstOctave (VLFeat's o_min) must be 0 or -1. COLMAP's shipping default
  // is -1 (SiftExtractionOptions::first_octave, sift.h:46), i.e. the image is
  // 2x-upsampled and an extra finer octave -1 is scanned -- which roughly
  // doubles the keypoint count vs o_min=0, so it is required for parity with
  // a default-configured COLMAP.
  if (firstOctave != 0 && firstOctave != -1) {
    throw std::invalid_argument("cusift: firstOctave must be 0 or -1, got " +
                                std::to_string(firstOctave));
  }
  CudaImage img01;
  img01.Allocate(img.width, img.height, iAlignUp(img.width, 128), false, NULL);
  {
    dim3 threads(16, 16);
    dim3 blocks(iDivUp(img.width, 16), iDivUp(img.height, 16));
    ScaleImageKernel<<<blocks, threads>>>(img01.d_data, img.d_data,
        img.width, img01.pitch, img.pitch, img.height, 1.0f/255.0f);
    checkMsg("ScaleImageKernel() execution failed\n");
  }

  outX.clear(); outY.clear(); outSigma.clear(); outOrientation.clear(); outDescriptors.clear();
  std::vector<VLFeatOctaveBuffers> octaves(numOctaves);
  // Per-octave staging: detection+orientation runs for every octave first,
  // then max_num_features decides which whole DOG levels survive, and only
  // then are descriptors computed (the octave gradient buffers stay alive in
  // `octaves` for the whole function, so the second pass can still use them).
  std::vector<std::vector<int>> levelKpCount(numOctaves, std::vector<int>(numScales, 0));
  std::vector<std::vector<char>> keepLevel(numOctaves, std::vector<char>(numScales, 1));
  std::vector<std::vector<VLFeatKp>> pendingRows(numOctaves);
  std::vector<std::vector<int>> pendingLevel(numOctaves);
  std::vector<float> pendingXper(numOctaves, 1.0f);
  CudaImage prevTopLevel;
  for (int oi = 0; oi < numOctaves; oi++) {
    int vo = firstOctave + oi;   // TRUE VLFeat octave index (may be -1)
    {
      if (oi == 0) {
        if (vo == -1) BuildUpsampledOctaveMinus1(octaves[oi], img01, numScales);
        else          BuildVLFeatOctave(octaves[oi], nullptr, &img01, numScales);
      } else {
        VLFeatOctaveBuffers &prev = octaves[oi-1];
        prevTopLevel = LevelView(prev.gauss, numScales, prev.width, prev.height, prev.pitch);
        BuildVLFeatOctave(octaves[oi], &prevTopLevel, nullptr, numScales);
      }
    }
    // xper = 2^o, so octave -1 (upsampled) maps back with 0.5 -- positions
    // and sigmas in the upsampled frame are HALVED to original image units.
    float xper = std::pow(2.0f, (float)vo);
    VLFeatOctaveBuffers &oct = octaves[oi];

    std::vector<DetectedKeypoint> localKps;
    DetectOctaveVLFeat(oct, vo, peakThresh, edgeThresh, numScales, localKps);
    int n = (int)localKps.size();
    if (n == 0) continue;

    // Orientation pass. Packed I/O: 20 B/row in, 20 B/row out, with an
    // explicit keypoint count.
    PackedScratch &pk = Packed(n, false);
    for (int i = 0; i < n; i++) {
      pk.h_kps[i].x = localKps[i].x;
      pk.h_kps[i].y = localKps[i].y;
      pk.h_kps[i].sigma = localKps[i].sigma;
      pk.h_kps[i].ori = 0.0f;
      pk.h_kps[i].level = localKps[i].is - (-1);
    }
    safeCall(cudaMemcpy(pk.d_kps, pk.h_kps, sizeof(VLFeatKp)*n, cudaMemcpyHostToDevice));

    int levelStride = oct.height * oct.pitch;
    {
      dim3 blocks(512), threads(256);
      ComputeOrientationsVLFeat<<<blocks, threads>>>(
          oct.gradMag.d_data, oct.gradAng.d_data,
          oct.width, oct.pitch, oct.height, levelStride,
          pk.d_kps, pk.d_ori, n);
      checkMsg("ComputeOrientationsVLFeat() execution failed\n");
    }
    {
      safeCall(cudaMemcpy(pk.h_ori, pk.d_ori, sizeof(VLFeatOri)*n, cudaMemcpyDeviceToHost));
    }

    // Expand: one descriptor-kernel input row per surviving angle, capped at
    // maxNumOrientations, matching COLMAP's own per-DOG-level expansion.
    // Rows are VLFeatKp (20 B), reserved up front; angles stay in RADIANS.
    std::vector<VLFeatKp> &expanded = pendingRows[oi];
    std::vector<int> &expandedLevel = pendingLevel[oi];
    expanded.reserve((size_t)n * maxNumOrientations);
    expandedLevel.reserve((size_t)n * maxNumOrientations);
    for (int i = 0; i < n; i++) {
      const VLFeatOri &op = pk.h_ori[i];
      int count = std::min(op.count, maxNumOrientations);
      for (int a = 0; a < count; a++) {
        VLFeatKp p;
        p.x = localKps[i].x;
        p.y = localKps[i].y;
        p.sigma = localKps[i].sigma;
        p.ori = op.angles[a];
        p.level = localKps[i].is - (-1);
        expanded.push_back(p);
        expandedLevel.push_back(localKps[i].is);
      }
    }
    // Per-(octave,level) pre-orientation keypoint counts, for COLMAP's
    // max_num_features enforcement applied after all octaves (below).
    for (int i = 0; i < n; i++) levelKpCount[oi][localKps[i].is]++;
    pendingXper[oi] = xper;
  }

  // COLMAP's max_num_features enforcement is LEVEL-GRANULAR, not per-keypoint:
  // it walks DOG levels from the coarsest backwards, accumulating keypoint
  // counts, and as soon as the running total exceeds max_num_features it drops
  // every FINER level entirely (sift.cc:292-315, `first_level_to_keep`). It is
  // NOT a "top-N strongest/largest" selection -- reproducing it as one would
  // give a different keypoint set. Levels are ordered ascending-sigma by
  // (octave, level), which is exactly the flat order below since
  // sigma = sigma0*2^(s/S)*2^o is monotonic in that ordering.
  int firstKeepFlat = 0;
  {
    std::vector<std::pair<int,int>> flat;   // (octaveIdx, level), ascending sigma
    for (int oi = 0; oi < numOctaves; oi++)
      for (int lev = 0; lev < numScales; lev++) flat.push_back({oi, lev});
    long long acc = 0;
    for (int k = (int)flat.size() - 1; k >= 0; k--) {
      acc += levelKpCount[flat[k].first][flat[k].second];
      if (acc > maxNumFeatures) { firstKeepFlat = k; break; }
    }
    for (int k = 0; k < (int)flat.size(); k++)
      keepLevel[flat[k].first][flat[k].second] = (k >= firstKeepFlat) ? 1 : 0;
  }

  for (int oi = 0; oi < numOctaves; oi++) {
    if (pendingRows[oi].empty()) continue;
    VLFeatOctaveBuffers &oct = octaves[oi];
    float xper = pendingXper[oi];
    int levelStride = oct.height * oct.pitch;
    // Count first, then copy the surviving rows straight into the descriptor
    // input buffer -- an intermediate `expanded` vector here was a second full
    // copy of the same data.
    int m = 0;
    for (size_t i = 0; i < pendingRows[oi].size(); i++)
      if (keepLevel[oi][pendingLevel[oi][i]]) m++;
    if (m == 0) continue;

    PackedScratch &pk = Packed(m, true);
    for (size_t i = 0, k = 0; i < pendingRows[oi].size(); i++)
      if (keepLevel[oi][pendingLevel[oi][i]]) pk.h_kps[k++] = pendingRows[oi][i];
    safeCall(cudaMemcpy(pk.d_kps, pk.h_kps, sizeof(VLFeatKp)*m, cudaMemcpyHostToDevice));

    {
      // ExtractSiftDescriptorsVLFeat indexes threads via a FLAT 1D
      // threadIdx.x (tid/nthreads=blockDim.x) -- a 2D block here would give
      // 8 redundant physical threads per distinct tid, each atomicAdd-ing
      // the same contribution into the same descriptor bin (silently
      // inflating every histogram entry ~8x). Must match the verified-
      // working call elsewhere in this file (dim3 threads(256), 1D).
      dim3 blocks(512), threads(256);
      ExtractSiftDescriptorsVLFeat<<<blocks, threads>>>(
          oct.gradMag.d_data, oct.gradAng.d_data,
          oct.width, oct.pitch, oct.height, levelStride,
          pk.d_kps, pk.d_desc, m, xper, 3.0f /*magnif*/);
      checkMsg("ExtractSiftDescriptorsVLFeat() execution failed\n");
    }
    {
      safeCall(cudaMemcpy(pk.h_kps, pk.d_kps, sizeof(VLFeatKp)*m, cudaMemcpyDeviceToHost));
      safeCall(cudaMemcpy(pk.h_desc, pk.d_desc, sizeof(float)*128*m, cudaMemcpyDeviceToHost));
    }
    for (int i = 0; i < m; i++) {
      outX.push_back(pk.h_kps[i].x);
      outY.push_back(pk.h_kps[i].y);
      outSigma.push_back(pk.h_kps[i].sigma);
      outOrientation.push_back(pk.h_kps[i].ori);   // radians throughout now
      std::array<float, 128> desc;
      std::copy(pk.h_desc + (size_t)i*128, pk.h_desc + (size_t)i*128 + 128, desc.begin());
      outDescriptors.push_back(desc);
    }
  }
}






