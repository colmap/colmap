#ifndef CUDASIFTH_H
#define CUDASIFTH_H

#include "cudautils.h"
#include "cudaImage.h"

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

int ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub);
void ExtractSiftOctave(SiftData &siftData, CudaImage &img, int octave, float thresh, float lowestScale, float subsampling, float *memoryTmp);
double ScaleDown(CudaImage &res, CudaImage &src, float variance);
double ScaleUp(CudaImage &res, CudaImage &src);
double ComputeOrientations(cudaTextureObject_t texObj, CudaImage &src, SiftData &siftData, int octave);
double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave);
double OrientAndExtract(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave);
double RescalePositions(SiftData &siftData, float scale);
double LowPass(CudaImage &res, CudaImage &src, float scale);
void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel);
double LaplaceMulti(cudaTextureObject_t texObj, CudaImage &baseImage, CudaImage *results, int octave);
double FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave);

// Descriptor-isolation harness (Part 2a): fill siftData.h_data[0..numPts) with
// externally supplied keypoints (xpos/ypos/scale = absolute image-pixel
// values, orientation = degrees), then call this to compute descriptors only
// -- detection and orientation are skipped entirely. See cudaSiftH.cu for the
// coordinate-frame details. Does not support scaleUp/--upsample.
void ExtractSiftDescriptorsOnly(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur);

#endif
