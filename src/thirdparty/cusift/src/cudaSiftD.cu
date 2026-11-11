// SPDX-License-Identifier: MIT AND BSD-2-Clause
// Copyright (c) 2017 Mårten Björkman (CudaSift; see ../LICENSE.CudaSift)
// Copyright (c) 2026 Yonatan Simson (see ../LICENSE)
// Portions from VLFeat (BSD-2-Clause; see ../LICENSE.VLFeat)

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#include "cudautils.h"
#include "cudaSiftD.h"
#include "cudaSift.h"
#include "fast_expn.h"

///////////////////////////////////////////////////////////////////////////////
// Kernel configuration
///////////////////////////////////////////////////////////////////////////////

//======= VLFeat-schedule pyramid kernels (ported from               =======//
//======= colmap/src/thirdparty/VLFeat/sift.c, BSD-2-Clause; no SiftGPU) ===//

// Pure 2x decimation (no anti-alias blur) -- matches VLFeat's
// copy_and_downsample exactly (sift.c:836-851): dst(y,x) = src(2y,2x).
// VLFeat's octave transition assumes a blur-free decimation because the
// correct blur is already accounted for by the per-level sigma schedule.
__global__ void PureDecimate2xKernel(float *d_Result, float *d_Data,
                                     int dstWidth, int dstPitch, int dstHeight,
                                     int srcPitch)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x>=dstWidth || y>=dstHeight) return;
  d_Result[y*dstPitch + x] = d_Data[(2*y)*srcPitch + 2*x];
}

// Exact replica of vl_sift_detect's refinement step (sift.c:1267-1427):
// up to 5 Newton iterations with pivoted 3x3 Gauss elimination, relocation
// when |b[0]| or |b[1]| > 0.6, final threshold+edge test on the refined
// position.
//
// Runs on the device (RefineKeypointsKernel below), straight out of the
// pitched DoG buffer. `yo`/`so` are explicit row/plane strides (yo=pitch,
// so=height*pitch).
__device__ static bool RefineOneKeypoint(const float *dog,
                              int width, int height, int yo, size_t so,
                              int nDogLevels, int s_min, int s_max,
                              int ix0, int iy0, int is0,
                              double peakThresh, double edgeThresh,
                              DetectedKeypoint &out)
{
  const int xo = 1;
  int x = ix0, y = iy0;
  const int s = is0;  // level never changes during refinement (verified)
  double Dx=0, Dy=0, Ds=0, Dxx=0, Dyy=0, Dxy=0, Dxs=0, Dys=0;
  double A[9], b[3] = {0,0,0};
  int dx = 0, dy = 0;

  auto at = [&](int ddx, int ddy, int dds) -> double {
    return dog[(size_t)(s - s_min + dds)*so + (size_t)(y+ddy)*yo + (x+ddx)*xo];
  };

  for (int iter = 0; iter < 5; iter++) {
    x += dx; y += dy;
    if (x < 1 || x > width-2 || y < 1 || y > height-2) return false;

    Dx = 0.5*(at(1,0,0) - at(-1,0,0));
    Dy = 0.5*(at(0,1,0) - at(0,-1,0));
    Ds = 0.5*(at(0,0,1) - at(0,0,-1));
    double Dss_ = at(0,0,1) + at(0,0,-1) - 2.0*at(0,0,0);
    Dxx = at(1,0,0) + at(-1,0,0) - 2.0*at(0,0,0);
    Dyy = at(0,1,0) + at(0,-1,0) - 2.0*at(0,0,0);
    Dxy = 0.25*(at(1,1,0) + at(-1,-1,0) - at(-1,1,0) - at(1,-1,0));
    Dxs = 0.25*(at(1,0,1) + at(-1,0,-1) - at(-1,0,1) - at(1,0,-1));
    Dys = 0.25*(at(0,1,1) + at(0,-1,-1) - at(0,-1,1) - at(0,1,-1));

    A[0]=Dxx; A[4]=Dyy; A[8]=Dss_;
    A[1]=A[3]=Dxy; A[2]=A[6]=Dxs; A[5]=A[7]=Dys;
    b[0]=-Dx; b[1]=-Dy; b[2]=-Ds;

    for (int j = 0; j < 3; j++) {
      double maxabsa = 0, maxa = 0; int maxi = -1;
      for (int i = j; i < 3; i++) {
        double a = A[i+j*3], absa = fabs(a);
        if (absa > maxabsa) { maxa = a; maxabsa = absa; maxi = i; }
      }
      if (maxabsa < 1e-10) { b[0]=b[1]=b[2]=0; break; }
      int i = maxi;
      for (int jj = j; jj < 3; jj++) {
        double tmp = A[i+jj*3]; A[i+jj*3] = A[j+jj*3]; A[j+jj*3] = tmp;
        A[j+jj*3] /= maxa;
      }
      double tmp = b[j]; b[j] = b[i]; b[i] = tmp;
      b[j] /= maxa;
      for (int ii = j+1; ii < 3; ii++) {
        double xf = A[ii+j*3];
        for (int jj = j; jj < 3; jj++) A[ii+jj*3] -= xf*A[j+jj*3];
        b[ii] -= xf*b[j];
      }
    }
    for (int i = 2; i > 0; i--) {
      double xf = b[i];
      for (int ii = i-1; ii >= 0; ii--) b[ii] -= xf*A[ii+i*3];
    }

    dx = (b[0] >  0.6 && x < width-2)  ?  1 : 0;
    dx += (b[0] < -0.6 && x > 1)       ? -1 : 0;
    dy = (b[1] >  0.6 && y < height-2) ?  1 : 0;
    dy += (b[1] < -0.6 && y > 1)       ? -1 : 0;
    if (dx == 0 && dy == 0) break;
  }

  double val = at(0,0,0) + 0.5*(Dx*b[0] + Dy*b[1] + Ds*b[2]);
  double score = (Dxx+Dyy)*(Dxx+Dyy) / (Dxx*Dyy - Dxy*Dxy);
  double xn = x + b[0], yn = y + b[1], sn = s + b[2];

  bool good = fabs(val) > peakThresh
      && score < (edgeThresh+1)*(edgeThresh+1)/edgeThresh && score >= 0
      && fabs(b[0]) < 1.5 && fabs(b[1]) < 1.5 && fabs(b[2]) < 1.5
      && xn >= 0 && xn <= width-1 && yn >= 0 && yn <= height-1
      && sn >= s_min && sn <= s_max;
  if (!good) return false;

  out.ix = x; out.iy = y; out.is = s; out.s = (float)sn;
  out.x = (float)xn; out.y = (float)yn;
  return true;
}

// One thread per coarse candidate. Accepted keypoints are appended with an
// atomic counter, so only the survivors cross PCIe. The append order is
// nondeterministic; DetectOctaveVLFeat sorts each octave's keypoints afterwards
// (SortOctaveKeypoints).
__global__ void RefineKeypointsKernel(const float *dog, int width, int height,
                                      int yo, size_t so, int nDogLevels,
                                      int s_min, int s_max,
                                      const int *candX, const int *candY, const int *candS,
                                      unsigned int candCount,
                                      double peakThresh, double edgeThresh,
                                      DetectedKeypoint *out, unsigned int *outCount)
{
  unsigned int c = blockIdx.x*blockDim.x + threadIdx.x;
  if (c >= candCount) return;
  DetectedKeypoint kp;
  if (!RefineOneKeypoint(dog, width, height, yo, so, nDogLevels, s_min, s_max,
                         candX[c], candY[c], candS[c], peakThresh, edgeThresh, kp))
    return;
  out[atomicAdd(outCount, 1u)] = kp;
}

// Adaptive-radius separable Gaussian blur -- VLFeat truncates its smoothing
// kernels at radius=ceil(4*sigma), so the radius follows sigma (up to ~13 for
// the largest per-level increment). Both passes take EXPLICIT separate src/dst
// pitches -- no same-pitch assumption between independently allocated buffers.
// The sigma schedule is image-independent and only
// ~8 distinct sigmas ever occur (two base blurs + the per-level increments,
// which repeat identically in every octave), so weights live in a BANK of
// kernels uploaded once per process instead of one synchronous
// cudaMemcpyToSymbol per blur (24 per image). Each blur is handed the base
// offset of its own kernel.
#define MAX_GAUSS_RADIUS 64
#define GAUSS_BANK_FLOATS 4096
__constant__ float d_AdaptiveGaussKernel[GAUSS_BANK_FLOATS];

__global__ void GaussianBlurHorizontalKernel(float * __restrict__ d_Result,
                                             const float * __restrict__ d_Data,
                                             int width, int dstPitch, int srcPitch,
                                             int height, int radius, int kOffset)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x>=width || y>=height) return;
  float sum = 0.0f;
  for (int j=-radius;j<=radius;j++) {
    int xx = min(max(x+j,0), width-1);
    sum += d_AdaptiveGaussKernel[kOffset+j+radius] * d_Data[y*srcPitch+xx];
  }
  d_Result[y*dstPitch+x] = sum;
}

__global__ void GaussianBlurVerticalKernel(float * __restrict__ d_Result,
                                           const float * __restrict__ d_Data,
                                           int width, int dstPitch, int srcPitch,
                                           int height, int radius, int kOffset)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x>=width || y>=height) return;
  float sum = 0.0f;
  for (int j=-radius;j<=radius;j++) {
    int yy = min(max(y+j,0), height-1);
    sum += d_AdaptiveGaussKernel[kOffset+j+radius] * d_Data[yy*srcPitch+x];
  }
  d_Result[y*dstPitch+x] = sum;
}

// 2x linear upsample, one dimension at a time -- matches VLFeat's
// copy_and_upsample_rows exactly (sift.c:738-758): even output samples copy
// the input directly, odd samples are the average of the two neighboring
// input samples, with the last input sample repeated at the far edge (no
// extrapolation). Applying this once horizontally then once vertically
// reproduces VLFeat's actual o_min<0 upsampling (which does it via two
// transposed passes -- mathematically identical result, done here without
// the transpose since CUDA addressing doesn't need it).
__global__ void Upsample2xHorizontalKernel(float *d_Result, const float *d_Data,
                                           int srcWidth, int dstPitch, int srcPitch, int height)
{
  int xh = blockIdx.x*blockDim.x + threadIdx.x;  // output column, 0..2*srcWidth-1
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int dstWidth = 2*srcWidth;
  if (xh>=dstWidth || y>=height) return;
  int x0 = xh/2;
  if ((xh & 1) == 0) {
    d_Result[y*dstPitch+xh] = d_Data[y*srcPitch+x0];
  } else {
    int x1 = min(x0+1, srcWidth-1);
    d_Result[y*dstPitch+xh] = 0.5f*(d_Data[y*srcPitch+x0] + d_Data[y*srcPitch+x1]);
  }
}

__global__ void Upsample2xVerticalKernel(float *d_Result, const float *d_Data,
                                         int width, int dstPitch, int srcPitch, int srcHeight)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int yh = blockIdx.y*blockDim.y + threadIdx.y;  // output row, 0..2*srcHeight-1
  int dstHeight = 2*srcHeight;
  if (x>=width || yh>=dstHeight) return;
  int y0 = yh/2;
  if ((yh & 1) == 0) {
    d_Result[yh*dstPitch+x] = d_Data[y0*srcPitch+x];
  } else {
    int y1 = min(y0+1, srcHeight-1);
    d_Result[yh*dstPitch+x] = 0.5f*(d_Data[y0*srcPitch+x] + d_Data[y1*srcPitch+x]);
  }
}

// Elementwise scale (used to normalize [0,255] pixel input to [0,1] for the
// VLFeat-schedule pyramid, as COLMAP does before calling VLFeat).
__global__ void ScaleImageKernel(float *d_Result, float *d_Data,
                                 int width, int dstPitch, int srcPitch, int height, float factor)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x>=width || y>=height) return;
  d_Result[y*dstPitch+x] = d_Data[y*srcPitch+x] * factor;
}

// Gradient magnitude+angle plane for one Gaussian level. Central differences,
// one-sided at borders -- matches VLFeat's update_gradient exactly
// (sift.c:1447-1484). Angle wrapped to [0, 2*pi).
// VLFeat's update_gradient does NOT use exact sqrt/atan2 -- it uses
// vl_fast_sqrt_f (mathop.h:479-497, the classic Carmack/Tarolli fast inverse
// sqrt: magic constant 0x5f3759df + TWO Newton iterations) and
// vl_fast_atan2_f (mathop.h:406-423, a polynomial approximation, max error
// ~0.0061 rad). Replicated verbatim from VLFeat, which is BSD 2-clause --
// see LICENSE.VLFeat, whose notice must ship with any redistribution of
// this file -- using exact sqrtf/atan2f here would silently diverge
// from VLFeat's actual gradient field by up to ~0.3 deg per sample, exactly
// the kind of systematic, position-dependent bias that flips borderline
// 0.8*max orientation peaks.
__device__ float vl_fast_resqrt_f_cuda(float x)
{
  union { float f; int i; } u;
  float xhalf = 0.5f * x;
  u.f = x;
  u.i = 0x5f3759df - (u.i >> 1);
  u.f = u.f * (1.5f - xhalf*u.f*u.f);
  u.f = u.f * (1.5f - xhalf*u.f*u.f);
  return u.f;
}

__device__ float vl_fast_sqrt_f_cuda(float x)
{
  return (x < 1e-8f) ? 0.0f : x * vl_fast_resqrt_f_cuda(x);
}

// M_PI is not portable: MSVC only defines it under _USE_MATH_DEFINES, and that
// cannot be set here because cudaSiftH.cu includes <cmath> before this file.
// Same value as M_PI, so the float results below are unchanged.
constexpr double kCusiftPi = 3.14159265358979323846;

__device__ float vl_fast_atan2_f_cuda(float y, float x)
{
  const float c3 = 0.1821f;
  const float c1 = 0.9675f;
  float abs_y = fabsf(y) + 1.19209290e-07f;  // VL_EPSILON_F
  float angle, r;
  if (x >= 0.0f) {
    r = (x - abs_y) / (x + abs_y);
    angle = (float)(kCusiftPi / 4.0);
  } else {
    r = (x + abs_y) / (abs_y - x);
    angle = (float)(3.0 * kCusiftPi / 4.0);
  }
  angle += (c3*r*r - c1) * r;
  return (y < 0.0f) ? -angle : angle;
}

// Gradient magnitude and angle go to two separate planes. An interleaved float2
// plane fetches the same bytes per sample but doubles the row length, which
// hurts 2D-window locality in the orientation/descriptor kernels.
__global__ void ComputeGradientPlaneKernel(const float *d_Level, int width, int pitch, int height,
                                           float *d_GradMag, float *d_GradAng)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x>=width || y>=height) return;
  int xm = max(x-1,0), xp = min(x+1,width-1);
  int ym = max(y-1,0), yp = min(y+1,height-1);
  float l = d_Level[y*pitch+xm], r = d_Level[y*pitch+xp];
  float u = d_Level[ym*pitch+x], d = d_Level[yp*pitch+x];
  float gx = (x==0 || x==width-1) ? (r-l) : 0.5f*(r-l);
  float gy = (y==0 || y==height-1) ? (d-u) : 0.5f*(d-u);
  float mag = vl_fast_sqrt_f_cuda(gx*gx+gy*gy);
  float ang = vl_fast_atan2_f_cuda(gy,gx);
  const float twoPi = 6.283185307179586f;
  ang = ang + twoPi;
  ang = ang - twoPi*floorf(ang/twoPi);  // vl_mod_2pi_f(ang + 2*pi) equivalent
  d_GradMag[y*pitch+x] = mag;
  d_GradAng[y*pitch+x] = ang;
}

// VLFeat-parity orientation kernel. One block per candidate
// keypoint (x,y,scale only -- orientation not yet known). Mirrors
// vl_sift_calc_keypoint_orientations exactly (sift.c:1559-1699): 36-bin
// histogram, Gaussian window sigmaw=1.5*scale, radius W=max(floor(3*sigmaw),1),
// circular cutoff r^2<W^2+0.6, BILINEAR (soft) binning across 2 adjacent bins
// -- VL_SIFT_BILINEAR_ORIENTATIONS is unconditionally #define'd at
// sift.c:669, not gated behind any COLMAP option), 6-pass 3-tap box smoothing,
// peaks where h0>0.8*max && local-max, parabolic sub-bin interpolation, up
// to 4 angles returned (VLFeat's own cap; COLMAP's max_num_orientations=2
// cap is applied on the host afterward, matching sift.cc:250-251).
// Ported from that BSD source, with Lowe's papers; no SiftGPU code
// referenced.
//
// I/O is PACKED (VLFeatKp in, VLFeatOri out, 20 B each; see cudaSift.h), with
// an explicit `numKps` per octave batch. Output angles are radians in VLFeat's
// convention (NOT degrees) and count is 0-4.
// Deterministic histogram accumulation. Many threads add into the same
// histogram bin, and float addition is not associative, so float atomicAdd
// gives results that depend on thread scheduling and change from run to run.
// Every contribution here is non-negative, so accumulate in 64-bit fixed
// point (units of 2^-32) instead: integer addition gives the same total in any
// order. The result is deterministic, with at most one fixed-point unit of
// rounding per contribution; the total is rounded once more when converted
// back to float. The totals stay far below the 2^32 range limit.
__device__ __forceinline__ unsigned long long HistToFixed(float v)
{
  return __float2ull_rn(v * 4294967296.0f);
}

__device__ __forceinline__ float HistFromFixed(unsigned long long v)
{
  return __ull2float_rn(v) * (1.0f / 4294967296.0f);
}

__global__ void ComputeOrientationsVLFeat(
    const float * __restrict__ d_GradMag, const float * __restrict__ d_GradAng,
    int width, int pitch, int height, size_t levelStrideElems,
    const VLFeatKp * __restrict__ d_kps, VLFeatOri * __restrict__ d_out, int numKps)
{
  __shared__ unsigned long long hist[36];
  __shared__ float histSmooth[36];
  __shared__ double expnTable[CUSIFT_EXPN_SZ + 1];
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const float twoPi = 6.283185307179586f;
  LoadExpnTable(expnTable);

  for (int bx = blockIdx.x; bx < numKps; bx += gridDim.x) {
    for (int i = tid; i < 36; i += nthreads) hist[i] = 0ull;
    __syncthreads();

    float x = d_kps[bx].x;
    float y = d_kps[bx].y;
    float sigma = d_kps[bx].sigma;
    int level = d_kps[bx].level;

    float sigmaw = 1.5f * sigma;
    int W = max((int)floorf(3.0f * sigmaw), 1);

    int xi = (int)(x + 0.5f);
    int yi = (int)(y + 0.5f);

    const float *gmag = d_GradMag + level * levelStrideElems;
    const float *gang = d_GradAng + level * levelStrideElems;

    int dyMin = max(-W, -yi), dyMax = min(W, height - 1 - yi);
    int dxMin = max(-W, -xi), dxMax = min(W, width - 1 - xi);
    int nxRange = dxMax - dxMin + 1;
    int nyRange = dyMax - dyMin + 1;
    int total = (nxRange > 0 && nyRange > 0) ? nxRange * nyRange : 0;

    for (int idx = tid; idx < total; idx += nthreads) {
      int ys = dyMin + idx / nxRange;
      int xs = dxMin + idx % nxRange;

      float dx = (float)(xi + xs) - x;
      float dy = (float)(yi + ys) - y;
      float r2 = dx*dx + dy*dy;
      if (r2 >= (float)W*W + 0.6f) continue;

      int px = xi + xs, py = yi + ys;
      float mod = gmag[py*pitch + px];
      float ang = gang[py*pitch + px];
      float wgt =
          (float)FastExpn(expnTable, (double)r2 / (2.0 * sigmaw * sigmaw));

      // Bilinear (soft) binning across 2 adjacent bins -- VLFeat's actual
      // code path (VL_SIFT_BILINEAR_ORIENTATIONS is unconditionally #define'd
      // at sift.c:669). Matches sift.c:1636-1644 exactly.
      float fbin = 36.0f * ang / twoPi;
      int bin = (int)floorf(fbin - 0.5f);
      float rbin = fbin - (float)bin - 0.5f;
      int bin0 = ((bin % 36) + 36) % 36;
      int bin1 = ((bin + 1) % 36 + 36) % 36;
      atomicAdd(hist + bin0, HistToFixed((1.0f - rbin) * mod * wgt));
      atomicAdd(hist + bin1, HistToFixed(rbin * mod * wgt));
    }
    __syncthreads();

    // 6-pass 3-tap box smoothing + peak detection: sequential, single thread
    // (36 elements x 6 passes is trivial next to the accumulation above).
    if (tid == 0) {
      for (int i = 0; i < 36; i++) histSmooth[i] = HistFromFixed(hist[i]);
      for (int iter = 0; iter < 6; iter++) {
        float prev = histSmooth[35];
        float first = histSmooth[0];
        int i;
        for (i = 0; i < 35; i++) {
          float newh = (prev + histSmooth[i] + histSmooth[(i+1) % 36]) / 3.0f;
          prev = histSmooth[i];
          histSmooth[i] = newh;
        }
        histSmooth[i] = (prev + histSmooth[i] + first) / 3.0f;
      }

      float maxh = 0.0f;
      for (int i = 0; i < 36; i++) maxh = fmaxf(maxh, histSmooth[i]);

      int nangles = 0;
      float angles[kMaxVLFeatOrientations];
      for (int i = 0; i < 36 && nangles < kMaxVLFeatOrientations; i++) {
        float h0 = histSmooth[i];
        float hm = histSmooth[(i - 1 + 36) % 36];
        float hp = histSmooth[(i + 1) % 36];
        if (h0 > 0.8f*maxh && h0 > hm && h0 > hp) {
          float di = -0.5f * (hp - hm) / (hp + hm - 2.0f*h0);
          float th = twoPi * ((float)i + di + 0.5f) / 36.0f;
          angles[nangles++] = th;
        }
      }
      for (int k = 0; k < kMaxVLFeatOrientations; k++) d_out[bx].angles[k] = (nangles > k) ? angles[k] : 0.0f;
      d_out[bx].count = nangles;
    }
    __syncthreads();
  }
}

// VLFeat-parity descriptor kernel. One block per keypoint; window
// size is data-dependent (grows with the keypoint's own sigma), so threads
// grid-stride over the (2W+1)x(2W+1) pixel window. Samples read from the
// PRECOMPUTED per-level gradient buffer for this octave, indexed by the
// keypoint's own assigned level (VLFeatKp::level) -- not a shared per-octave
// texture. Mirrors vl_sift_calc_keypoint_descriptor (sift.c:1923-2100):
// magnif=3.0, NBP=4, NBO=8, window sigma=NBP/2, centered 2x2x2 trilinear
// binning, L2->clamp(0.2)->L2 normalization. Ported from that BSD source,
// with Lowe's papers; no SiftGPU code referenced.
//
// Launch with a 1D block (blockDim.y == blockDim.z == 1) whose size is a power
// of two from 128 to 1024: threads are indexed by threadIdx.x alone, so a 2D
// block would add every contribution several times; threads 0..127 write the
// 128 bins; and the norm reductions halve blockDim.x. The host launches 256
// threads; `partial` holds the CUDA maximum of 1024.
#define VLDESC_NBP 4
#define VLDESC_NBO 8
__global__ void ExtractSiftDescriptorsVLFeat(
    const float * __restrict__ d_GradMag,   // octave-combined, level-stacked planes
    const float * __restrict__ d_GradAng,
    int width, int pitch, int height, size_t levelStrideElems,
    VLFeatKp * __restrict__ d_kps, float * __restrict__ d_desc, int numKps,
    float subsampling, float magnif)
{
  __shared__ unsigned long long buffer[VLDESC_NBP*VLDESC_NBP*VLDESC_NBO];  // 128
  __shared__ float partial[1024];
  __shared__ double expnTable[CUSIFT_EXPN_SZ + 1];
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const float twoPi = 6.283185307179586f;
  LoadExpnTable(expnTable);

  for (int bx = blockIdx.x; bx < numKps; bx += gridDim.x) {
    for (int i = tid; i < 128; i += nthreads) buffer[i] = 0ull;
    __syncthreads();

    float x = d_kps[bx].x;
    float y = d_kps[bx].y;
    float sigma = d_kps[bx].sigma;
    int level = d_kps[bx].level;
    float angle0 = d_kps[bx].ori;   // already radians in VLFeatKp

    float SBP = magnif*sigma + 1e-10f;
    int W = (int)floorf(sqrtf(2.0f)*SBP*(VLDESC_NBP+1)/2.0f + 0.5f);

    int xi = (int)(x + 0.5f);
    int yi = (int)(y + 0.5f);
    float st0 = sinf(angle0), ct0 = cosf(angle0);

    const float *gmag = d_GradMag + level*levelStrideElems;
    const float *gang = d_GradAng + level*levelStrideElems;

    int dyMin = max(-W, 1-yi), dyMax = min(W, height-yi-2);
    int dxMin = max(-W, 1-xi), dxMax = min(W, width-xi-2);
    int nxRange = dxMax - dxMin + 1;
    int nyRange = dyMax - dyMin + 1;
    int total = (nxRange>0 && nyRange>0) ? nxRange*nyRange : 0;

    for (int idx = tid; idx < total; idx += nthreads) {
      int dyi = dyMin + idx/nxRange;
      int dxi = dxMin + idx%nxRange;

      int px = xi+dxi, py = yi+dyi;
      float mod = gmag[py*pitch+px];
      float angle = gang[py*pitch+px];
      float theta = angle - angle0;
      theta = theta - twoPi*floorf(theta/twoPi);  // wrap to [0,2pi)

      float dx = (float)(xi+dxi) - x;
      float dy = (float)(yi+dyi) - y;
      float nx = ( ct0*dx + st0*dy) / SBP;
      float ny = (-st0*dx + ct0*dy) / SBP;
      float nt = VLDESC_NBO * theta / twoPi;

      float wsigma = VLDESC_NBP / 2.0f;
      float win = (float)FastExpn(
          expnTable, (nx*nx + ny*ny) / (2.0 * wsigma * wsigma));

      int binx = (int)floorf(nx - 0.5f);
      int biny = (int)floorf(ny - 0.5f);
      int bint = (int)floorf(nt);
      float rbinx = nx - (binx + 0.5f);
      float rbiny = ny - (biny + 0.5f);
      float rbint = nt - bint;

      for (int dbinx = 0; dbinx < 2; dbinx++) {
        for (int dbiny = 0; dbiny < 2; dbiny++) {
          for (int dbint = 0; dbint < 2; dbint++) {
            int bx_ = binx + dbinx, by_ = biny + dbiny;
            if (bx_ >= -(VLDESC_NBP/2) && bx_ < (VLDESC_NBP/2) &&
                by_ >= -(VLDESC_NBP/2) && by_ < (VLDESC_NBP/2)) {
              float weight = win * mod
                  * fabsf(1.0f - dbinx - rbinx)
                  * fabsf(1.0f - dbiny - rbiny)
                  * fabsf(1.0f - dbint - rbint);
              int bt_ = (bint + dbint) % VLDESC_NBO;
              if (bt_ < 0) bt_ += VLDESC_NBO;
              // Write directly in COLMAP's "UBC" orientation-bin order (the
              // convention SiftGPU/Lowe binaries and existing COLMAP DBs
              // use), not VLFeat's native order -- matches
              // TransformVLFeatToUBCFeatureDescriptors exactly
              // (colmap/src/colmap/feature/sift.cc:118-131): q={0,7,6,5,4,3,2,1},
              // i.e. bin 0 fixed, bins 1..7 reversed, per spatial cell.
              int bt_ubc = (VLDESC_NBO - bt_) % VLDESC_NBO;
              int arrIdx = (by_ + VLDESC_NBP/2)*(VLDESC_NBO*VLDESC_NBP)
                         + (bx_ + VLDESC_NBP/2)*VLDESC_NBO + bt_ubc;
              atomicAdd(buffer + arrIdx, HistToFixed(weight));
            }
          }
        }
      }
    }
    __syncthreads();

    // Normalize: L2 -> clamp(0.2) -> L2, as VLFeat does.
    float val = (tid < 128) ? HistFromFixed(buffer[tid]) : 0.0f;
    partial[tid] = (tid < 128) ? val*val : 0.0f;
    __syncthreads();
    for (int s = nthreads/2; s > 0; s >>= 1) { if (tid<s) partial[tid] += partial[tid+s]; __syncthreads(); }
    float norm1 = sqrtf(partial[0]) + 1e-12f;
    float clamped = (tid < 128) ? fminf(val/norm1, 0.2f) : 0.0f;
    __syncthreads();
    partial[tid] = (tid < 128) ? clamped*clamped : 0.0f;
    __syncthreads();
    for (int s = nthreads/2; s > 0; s >>= 1) { if (tid<s) partial[tid] += partial[tid+s]; __syncthreads(); }
    float norm2 = sqrtf(partial[0]) + 1e-12f;
    if (tid < 128) d_desc[(size_t)bx*128 + tid] = clamped / norm2;
    __syncthreads();

    if (tid == 0) {
      d_kps[bx].x *= subsampling;
      d_kps[bx].y *= subsampling;
      d_kps[bx].sigma *= subsampling;
    }
    __syncthreads();
  }
}

//======= Detection (ported from colmap/src/thirdparty/VLFeat/            =//
//======= sift.c, BSD-2-Clause; no SiftGPU code)                           =//

// DoG plane = gaussB - gaussA, matches vl_sift_detect's own DoG computation
// (sift.c:1176-1185) exactly -- plain pointwise difference of two adjacent
// Gaussian levels, no separate blur.
__global__ void ComputeDogPlaneKernel(const float *d_GaussA, const float *d_GaussB,
                                      int width, int pitch, int height, float *d_Dog)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x>=width || y>=height) return;
  int idx = y*pitch+x;
  d_Dog[idx] = d_GaussB[idx] - d_GaussA[idx];
}

// Coarse (pre-refinement) 26-neighbor DoG extrema scan -- matches
// vl_sift_detect's CHECK_NEIGHBORS macro exactly (sift.c:1199-1232):
// threshold |v| >= 0.8*peak_thresh, strict extremum over all 26 neighbors in
// the 3x3x3 cube (x,y,s). Only the interior is scanned (x,y in [1,w-2],
// s-plane-index in [1, nDogLevels-2]) -- same border margin as VLFeat.
// Candidates are appended (ix, iy, s) to a flat output buffer via an atomic
// counter; s here is the TRUE VLFeat level value (dogPlaneIdx - 1, since
// s_min=-1 and dogPlaneIdx = s - s_min). Refinement (Newton, relocation,
// edge test) runs afterwards in RefineKeypointsKernel, one thread per
// candidate. Every extremum is counted, even past maxCandidates, so the
// caller can detect an overflow and rescan.
__global__ void FindDogExtremaKernel(const float *d_Dog, int width, int pitch, int height,
                                     int nDogLevels, float coarseThresh,
                                     int *d_CandX, int *d_CandY, int *d_CandS,
                                     int maxCandidates, unsigned int *d_CandCount)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x + 1;
  int y = blockIdx.y*blockDim.y + threadIdx.y + 1;
  int sIdx = blockIdx.z + 1;  // dog-plane index, interior range [1, nDogLevels-2]
  if (x >= width-1 || y >= height-1 || sIdx >= nDogLevels-1) return;

  // Plane stride, in size_t: a plane of an upsampled large image can exceed
  // the int range once multiplied by the plane index.
  const size_t so = (size_t)pitch*height;
  const float *pt = d_Dog + sIdx*so + (size_t)y*pitch + x;
  float v = *pt;
  const float th = coarseThresh;  // float(0.8 * peakThresh), VLFeat's threshold

  bool isMax = v >= th;
  bool isMin = v <= -th;
  if (!isMax && !isMin) return;

  // All 26 combinations of (dx,dy,ds) in {-1,0,1}^3 except (0,0,0), in
  // lexicographic order.
  const int dxs[26] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1};
  const int dys[26] = {-1,-1,-1,0,0,0,1,1,1,-1,-1,-1,0,0,1,1,1,-1,-1,-1,0,0,0,1,1,1};
  const int dss[26] = {-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1};
  #pragma unroll
  for (int k = 0; k < 26; k++) {
    float nv = *(pt + dxs[k] + dys[k]*pitch + dss[k]*(ptrdiff_t)so);
    if (isMax && v <= nv) { isMax = false; }
    if (isMin && v >= nv) { isMin = false; }
  }
  if (!isMax && !isMin) return;

  unsigned int idx = atomicAdd(d_CandCount, 1);
  if ((int)idx < maxCandidates) {
    d_CandX[idx] = x;
    d_CandY[idx] = y;
    d_CandS[idx] = sIdx - 1;  // true VLFeat level s = dogPlaneIdx - 1 (s_min=-1)
  }
}

