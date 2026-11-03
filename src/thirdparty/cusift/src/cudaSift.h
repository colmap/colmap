#ifndef CUDASIFT_H
#define CUDASIFT_H

#include "cudaImage.h"
#include <array>
#include <vector>

typedef struct {
  float xpos;
  float ypos;   
  float scale;
  float sharpness;
  float edgeness;
  float orientation;
  float score;
  float ambiguity;
  int match;
  float match_xpos;
  float match_ypos;
  float match_error;
  float subsampling;
  float empty[3];
  float data[128];
} SiftPoint;

typedef struct {
  int numPts;         // Number of available Sift points
  int maxPts;         // Number of allocated Sift points
#ifdef MANAGEDMEM
  SiftPoint *m_data;  // Managed data
#else
  SiftPoint *h_data;  // Host (CPU) data
  SiftPoint *d_data;  // Device (GPU) data
#endif
} SiftData;

// Packed kernel I/O for the VLFeat pipeline (docs/PERFORMANCE.md "packed kernel I/O
// struct"). `SiftPoint` is 576 B and the VLFeat kernels touch six of its fields;
// moving whole SiftPoints in and out of every octave pass cost ~9 ms/image in
// pure struct traffic. These are what the kernels actually read and write:
//   orientation: reads x, y, sigma, level      -> writes VLFeatOri
//   descriptor:  reads x, y, sigma, ori, level -> writes 128 floats, rescales x/y/sigma
// The public SiftData/SiftPoint API below is UNCHANGED -- the harness entry
// points convert at the boundary (a memcpy over 20 B/row, which does not give
// back the saving).
struct VLFeatKp {          // 20 B, vs 576 B for a SiftPoint
  float x, y, sigma, ori;  // ori in RADIANS here (the kernel's own convention)
  int level;               // gradient-plane index, = is - s_min
};

struct VLFeatOri {         // orientation-kernel output, 20 B
  float angles[4];         // radians, VLFeat convention
  int count;               // 0..4
};

void InitCuda(int devNum = 0);
float *AllocSiftTempMemory(int width, int height, int numOctaves, bool scaleUp = false);
void FreeSiftTempMemory(float *memoryTmp);
void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale = 0.0f, bool scaleUp = false, float *tempMemory = 0);
// Descriptor-isolation harness (Part 2a): siftData.h_data[0..numPts) must be
// pre-filled with externally supplied keypoints (xpos/ypos/scale = absolute
// image-pixel values, orientation = degrees). Skips detection/orientation
// entirely and computes descriptors only. See cudaSiftH.cu for details.
void ExtractSiftDescriptorsOnly(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur);
// Part 2c: computes VLFeat-parity orientation(s) for (x,y,scale)-only
// candidates. outAngles/outCount indexed by original candidate order;
// angles in RADIANS (VLFeat's own convention). See cudaSiftH.cu for details.
void ComputeOrientationsOnly(SiftData &siftData, CudaImage &img, int numOctaves,
                             std::vector<std::array<float, 4>> &outAngles,
                             std::vector<int> &outCount);
// Test-only: same as ComputeOrientationsOnly but builds an extra upsampled
// "octave -1" first, matching COLMAP's real first_octave=-1 default (see
// cudaSiftH.cu comment). Not for production use.
void ComputeOrientationsOnlyOm1(SiftData &siftData, CudaImage &img, int numOctaves,
                                std::vector<std::array<float, 4>> &outAngles,
                                std::vector<int> &outCount);
// Debug only: dump cusift's own CUDA-computed pyramid for one octave to raw
// float32 files, for direct ground-truth comparison against vlfeat_dump's
// output. See cudaSiftH.cu for details.
void DumpPyramidDebug(CudaImage &img, int numOctaves, int dumpOctave, const char *prefix);
// Part 2d: VLFeat-faithful detection (coarse extrema scan + iterative Newton
// refinement + edge test, both on GPU; the refinement body is one
// __host__ __device__ function whose CPU instantiation still serves
// --debug-dump-prefix and is bit-identical to the GPU one -- see cudaSiftD.cu's
// RefineOneKeypoint and docs/perf_tier1_rtx4060.md).
struct DetectedKeypoint {
  int o, is, ix, iy;
  float x, y, sigma, s;
};
void DetectKeypointsVLFeat(CudaImage &img, int numOctaves, float peakThresh, float edgeThresh,
                           std::vector<DetectedKeypoint> &out, const char *debugDumpPrefix = nullptr);
// Part 2d integration: detect + orient in one pass, no external-injection
// harness, no AssignOctaveLevel inversion. See cudaSiftH.cu for details.
void DetectAndOrientVLFeat(CudaImage &img, int numOctaves, float peakThresh, float edgeThresh,
                          std::vector<DetectedKeypoint> &outKps,
                          std::vector<std::array<float, 4>> &outAngles,
                          std::vector<int> &outCounts);
// Full pipeline: detect + orient + descriptor, multi-orientation expanded.
// See cudaSiftH.cu for details.
// firstOctave = VLFeat o_min: 0, or -1 for COLMAP's default 2x-upsampled
// extra fine octave (roughly doubles keypoint count).
void ExtractSiftFullVLFeat(CudaImage &img, int numOctaves, float peakThresh, float edgeThresh,
                           int maxNumOrientations, int firstOctave, int maxNumFeatures,
                           std::vector<float> &outX, std::vector<float> &outY,
                           std::vector<float> &outSigma, std::vector<float> &outOrientation,
                           std::vector<std::array<float, 128>> &outDescriptors);
void InitSiftData(SiftData &data, int num = 1024, bool host = false, bool dev = true);
void FreeSiftData(SiftData &data);
void PrintSiftData(SiftData &data);
double MatchSiftData(SiftData &data1, SiftData &data2);
double FindHomography(SiftData &data,  float *homography, int *numMatches, int numLoops = 1000, float minScore = 0.85f, float maxAmbiguity = 0.95f, float thresh = 5.0f);

#endif
