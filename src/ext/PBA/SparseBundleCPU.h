////////////////////////////////////////////////////////////////////////////
//  File:       SparseBundleCPU.h
//  Author:       Changchang Wu (ccwu@cs.washington.edu)
//  Description :   interface of the CPU-version of multi-core bundle adjustment
//
//  Copyright (c) 2011  Changchang Wu (ccwu@cs.washington.edu)
//    and the University of Washington at Seattle
//
//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation; either
//  Version 3 of the License, or (at your option) any later version.
//
//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  General Public License for more details.
//
////////////////////////////////////////////////////////////////////////////////

#if !defined(SPARSE_BUNDLE_CPU_H)
#define SPARSE_BUNDLE_CPU_H

// BYTE-ALIGNMENT for data allocation (16 required for SSE, 32 required for AVX)
// PREVIOUS version uses only SSE. The new version will include AVX.
// SO the alignment is increased from 16 to 32
#define VECTOR_ALIGNMENT 32
#define FLOAT_ALIGN 8
#define VECTOR_ALIGNMENT_MASK (VECTOR_ALIGNMENT - 1)
#define ALIGN_PTR(p) \
  ((((size_t)p) + VECTOR_ALIGNMENT_MASK) & (~VECTOR_ALIGNMENT_MASK))

namespace pba {

template <class Float>
class avec {
  bool _owner;
  Float* _data;
  Float* _last;
  size_t _size;
  size_t _capacity;

 public:
  static Float* allocate(size_t count) {
    size_t size = count * sizeof(Float);
#ifdef _MSC_VER
    Float* p = (Float*)_aligned_malloc(size, VECTOR_ALIGNMENT);
    if (p == NULL) throw std::bad_alloc();
    return p;
#else
    char* p = (char*)malloc(size + VECTOR_ALIGNMENT + 4);
    if (p == NULL) throw std::bad_alloc();
    char* p1 = p + 1;
    char* p2 =
        (char*)ALIGN_PTR(p1);  //(char*) (((((size_t)p1) + 15) >> 4) << 4);
    char* p3 = (p2 - 1);
    p3[0] = (p2 - p);
    return (Float*)p2;
#endif
  }
  static void deallocate(void* p) {
#ifdef _MSC_VER
    _aligned_free(p);
#else
    char* p3 = ((char*)p) - 1;
    free(((char*)p) - p3[0]);
#endif
  }

 public:
  avec() {
    _owner = true;
    _last = _data = NULL;
    _size = _capacity = 0;
  }
  avec(size_t count) {
    _data = allocate(count);
    _size = _capacity = count;
    _last = _data + count;
    _owner = true;
  }
  ~avec() {
    if (_data && _owner) deallocate(_data);
  }

  inline void resize(size_t newcount) {
    if (!_owner) {
      _data = _last = NULL;
      _capacity = _size = 0;
      _owner = true;
    }
    if (newcount <= _capacity) {
      _size = newcount;
      _last = _data + newcount;
    } else {
      if (_data && _owner) deallocate(_data);
      _data = allocate(newcount);
      _size = _capacity = newcount;
      _last = _data + newcount;
    }
  }

  inline void set(Float* data, size_t count) {
    if (_data && _owner) deallocate(_data);
    _data = data;
    _owner = false;
    _size = count;
    _last = _data + _size;
    _capacity = count;
  }
  inline void swap(avec<Float>& next) {
    bool _owner_bak = _owner;
    Float* _data_bak = _data;
    Float* _last_bak = _last;
    size_t _size_bak = _size;
    size_t _capa_bak = _capacity;

    _owner = next._owner;
    _data = next._data;
    _last = next._last;
    _size = next._size;
    _capacity = next._capacity;

    next._owner = _owner_bak;
    next._data = _data_bak;
    next._last = _last_bak;
    next._size = _size_bak;
    next._capacity = _capa_bak;
  }

  inline operator Float*() { return _size ? _data : NULL; }
  inline operator Float* const() const { return _data; }
  inline Float* begin() { return _size ? _data : NULL; }
  inline Float* data() { return _size ? _data : NULL; }
  inline Float* end() { return _last; }
  inline const Float* begin() const { return _size ? _data : NULL; }
  inline const Float* end() const { return _last; }
  inline size_t size() const { return _size; }
  inline size_t IsValid() const { return _size; }
  void SaveToFile(const char* name);
};

template <class Float>
class SparseBundleCPU : public ParallelBA, public ConfigBA {
 public:
  SparseBundleCPU(const int num_threads);

  typedef avec<Float> VectorF;
  typedef vector<int> VectorI;
  typedef float float_t;

 protected:  // cpu data
  int _num_camera;
  int _num_point;
  int _num_imgpt;
  CameraT* _camera_data;
  float* _point_data;

  ////////////////////////////////
  const float* _imgpt_data;
  const int* _camera_idx;
  const int* _point_idx;
  const int* _focal_mask;

  ///////////sumed square error
  float _projection_sse;

 protected:  // cuda data
  VectorF _cuCameraData;
  VectorF _cuCameraDataEX;
  VectorF _cuPointData;
  VectorF _cuPointDataEX;
  VectorF _cuMeasurements;
  VectorF _cuImageProj;
  VectorF _cuJacobianCamera;
  VectorF _cuJacobianPoint;
  VectorF _cuJacobianCameraT;
  VectorI _cuProjectionMap;
  VectorI _cuPointMeasurementMap;
  VectorI _cuCameraMeasurementMap;
  VectorI _cuCameraMeasurementList;
  VectorI _cuCameraMeasurementListT;

  //////////////////////////
  VectorF _cuBlockPC;
  VectorF _cuVectorSJ;

  /// LM normal    equation
  VectorF _cuVectorJtE;
  VectorF _cuVectorJJ;
  VectorF _cuVectorJX;
  VectorF _cuVectorXK;
  VectorF _cuVectorPK;
  VectorF _cuVectorZK;
  VectorF _cuVectorRK;

  //////////////////////////////////
 protected:
  int _num_imgpt_q;
  float _weight_q;
  VectorI _cuCameraQList;
  VectorI _cuCameraQMap;
  VectorF _cuCameraQMapW;
  VectorF _cuCameraQListW;

 protected:
  bool ProcessIndexCameraQ(vector<int>& qmap, vector<int>& qlist);
  void ProcessWeightCameraQ(vector<int>& cpnum, vector<int>& qmap, Float* qmapw,
                            Float* qlistw);

 protected:  // internal functions
  int ValidateInputData();
  int InitializeBundle();
  int GetParameterLength();
  void BundleAdjustment();
  void NormalizeData();
  void TransferDataToHost();
  ;
  void DenormalizeData();
  void NormalizeDataF();
  void NormalizeDataD();
  bool InitializeStorageForSFM();
  bool InitializeStorageForCG();

  void SaveBundleRecord(int iter, float res, float damping, float& g_norm,
                        float& g_inf);

 protected:
  void PrepareJacobianNormalization();
  void EvaluateJacobians();
  void ComputeJtE(VectorF& E, VectorF& JtE, int mode = 0);
  void ComputeJX(VectorF& X, VectorF& JX, int mode = 0);
  void ComputeDiagonal(VectorF& JJI);
  void ComputeBlockPC(float lambda, bool dampd);
  void ApplyBlockPC(VectorF& v, VectorF& pv, int mode = 0);
  float UpdateCameraPoint(VectorF& dx, VectorF& cuImageTempProj);
  float EvaluateProjection(VectorF& cam, VectorF& point, VectorF& proj);
  float EvaluateProjectionX(VectorF& cam, VectorF& point, VectorF& proj);
  float SaveUpdatedSystem(float residual_reduction, float dx_sqnorm,
                          float damping);
  float EvaluateDeltaNorm();
  int SolveNormalEquationPCGB(float lambda);
  int SolveNormalEquationPCGX(float lambda);
  int SolveNormalEquation(float lambda);
  void NonlinearOptimizeLM();
  void AdjustBundleAdjsutmentMode();
  void RunProfileSteps();
  void RunTestIterationLM(bool reduced);
  void DumpCooJacobian();

 private:
  static int FindProcessorCoreNum();

 public:
  virtual void AbortBundleAdjustment() { __abort_flag = true; }
  virtual int GetCurrentIteration() { return __current_iteration; }
  virtual void SetNextTimeBudget(int seconds) {
    __bundle_time_budget = seconds;
  }
  virtual void SetNextBundleMode(BundleModeT mode) {
    __bundle_mode_next = mode;
  }
  virtual void SetFixedIntrinsics(bool fixed) { __fixed_intrinsics = fixed; }
  virtual void EnableRadialDistortion(DistortionT type) {
    __use_radial_distortion = type;
  }
  virtual void ParseParam(int narg, char** argv) {
    ConfigBA::ParseParam(narg, argv);
  }
  virtual ConfigBA* GetInternalConfig() { return this; }

 public:
  SparseBundleCPU();
  virtual void SetCameraData(size_t ncam, CameraT* cams);
  virtual void SetPointData(size_t npoint, Point3D* pts);
  virtual void SetProjection(size_t nproj, const Point2D* imgpts,
                             const int* point_idx, const int* cam_idx);
  virtual void SetFocalMask(const int* fmask, float weight);
  virtual float GetMeanSquaredError();
  virtual int RunBundleAdjustment();
};

ParallelBA* NewSparseBundleCPU(bool dp, const int num_threads);

}  // namespace pba

#endif
