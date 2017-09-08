////////////////////////////////////////////////////////////////////////////
//  File:       ConfigBA.h
//  Author:       Changchang Wu (ccwu@cs.washington.edu)
//  Description :   configuration object class
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

#ifndef CONFIG_BA_H
#define CONFIG_BA_H
#include <vector>

namespace pba {

class ConfigBA {
 protected:
  enum {
    TIMER_OVERALL = 0,
    TIMER_OPTIMIZATION,
    TIMER_GPU_ALLOCATION,
    TIMER_GPU_UPLOAD,
    TIMER_PREPROCESSING,
    TIMER_GPU_DOWNLOAD,
    TIMER_CG_ITERATION,
    TIMER_LM_ITERATION,
    TIMER_FUNCTION_JJ,
    TIMER_FUNCTION_PJ,
    TIMER_FUNCTION_DD,
    TIMER_FUNCTION_JX,
    TIMER_FUNCTION_JTE,
    TIMER_FUNCTION_BC,
    TIMER_FUNCTION_MP,
    TIMER_FUNCTION_UP,
    TIMER_PROFILE_STEP,
    NUM_TIMER,
    FUNC_JX = 0,
    FUNC_JX_,
    FUNC_JTEC_JCT,
    FUNC_JTEC_JCO,
    FUNC_JTEP,
    FUNC_JTE_,
    FUNC_JJ_JCO_JCT_JP,
    FUNC_JJ_JCO_JP,
    FUNC_JJ_JCT_JP,
    FUNC_JJ_JP,
    FUNC_PJ,
    FUNC_BCC_JCT,
    FUNC_BCC_JCO,
    FUNC_BCP,
    FUNC_MPC,
    FUNC_MPP,
    FUNC_VS,
    FUNC_VV,
    NUM_FUNC
  };
  class TimerBA {
    ConfigBA* _config;
    int _timer;

   public:
    TimerBA(ConfigBA* config, int timer) {
      (_config = config)->BundleTimerStart(_timer = timer);
    }
    TimerBA(ConfigBA* config, int timer, bool) {
      (_config = config)->BundleTimerSwitch(_timer = timer);
    }
    ~TimerBA() { _config->BundleTimerSwitch(_timer); }
  };
  friend class TimerBA;

 public:
  //////////////////////////////
  int __lm_max_iteration;      //(default 50)
  int __cg_max_iteration;      //(default 100)
  int __cg_min_iteration;      //(default 10)
  int __cg_recalculate_freq;   //(default 0)
  bool __accurate_gain_ratio;  //(default true) accurate gain ratio for
                               //approximate solutions

  //////////////////////////////
  float __lm_delta_threshold;     //(default 1e-6)|dx|_2, I use absolute (not
                                  //relative) change
  float __lm_gradient_threshold;  //(default 1e-10)|Jt * e|_inf
  float __lm_mse_threshold;  //(default 0.25) quit if MSE is equal to or smaller
                             //than this
  float __lm_initial_damp;   //(default 0.001)initial damping factor
  float __lm_minimum_damp;   //(default 1e-10)minimum damping factor
  float __lm_maximum_damp;
  float __cg_norm_threshold;  //(default 0.1)terminate CG if norm ratio is less
                              //than threshold
  float __cg_norm_guard;      //(default 1.0)abort cg when norm increases to
  int __pba_experimental;
  bool __cg_schur_complement;

  //////////////////////////////
  bool __lm_check_gradient;  //(default false) check g_inf for convergence
  float __lm_damping_auto_switch;
  bool __lm_use_diagonal_damp;  //(default true)use (Jt*J + lambda * diag(Jt*J))
                                //= Jt * e
  //            or  use (Jt*J + lambda * I) = Jt * e
  bool __fixed_intrinsics;      //(default false) set true for calibrated camera
                                //system
  int __use_radial_distortion;  //(default 0, 1 for projection distortion, 2 for
                                //measurement distortion)
  bool __reset_initial_distortion;  //(default false) reset the initial
                                    //distortio to 0

  ////////////////////////////
  int __verbose_level;  //(default 2) how many messages to print out
  bool __abort_flag;    //(default false)abort the bundle adjustment loop if set
                        //true
  bool __verbose_cg_iteration;   //(default false)print out details of Conjugate
                                 //Gradients
  bool __verbose_function_time;  //(default false)print timing of some key
                                 //functions
  bool __save_gradient_norm;  //(default false)save |Jt * e|_2 of each iteration
  bool __verbose_allocation;  //(default false)whether print out allocation
                              //details
  bool __verbose_sse;         //(default false) show mse or sse

  ///////////////////////////////////
  bool __jc_store_transpose;  //(default true) whether store transpose of JC
  bool __no_jacobian_store;   //(default false) whether use memory saving mode
  bool __jc_store_original;   //(default true) whether store original JC

  ///////////////////////////////////
  bool __jacobian_normalize;  //(default true) scaling the jacobians according
                              //to initial jacobians
  bool __focal_normalize;     //(default true) data normalization
  bool __depth_normalize;     //(default true) data normalization
  bool __depth_degeneracy_fix;
  float __data_normalize_median;
  float __depth_check_epsilon;
  /////////////////////////////

 protected:
  bool __multiply_jx_usenoj;  // for debug purpose
 protected:
  /////////////////////////////
  int __selected_device;
  int __cpu_data_precision;
  int __bundle_time_budget;
  int __bundle_mode_next;
  int __bundle_current_mode;
  //////////////////////////////
  float __initial_mse;
  float __final_mse;
  float __final_mse_x;
  float __focal_scaling;
  float __depth_scaling;
  int __current_device;
  int __current_iteration;
  int __num_cg_iteration;
  int __num_lm_success;
  int __num_lm_iteration;
  int __num_projection_eval;
  int __num_jacobian_eval;
  int __num_camera_modified;
  int __num_point_behind;
  int __pba_return_code;
  int __recent_cg_status;
  int __profile_pba;
  bool __cpu_thread_profile;
  bool __debug_pba;
  bool __warmup_device;
  size_t __memory_usage;
  /////////////////////////////////////
  bool __matlab_format_stat;
  char* __stat_filename;
  const char* __driver_output;
  std::vector<float> __bundle_records;
  double __timer_record[NUM_TIMER];
  int __num_cpu_thread_all;
  int __num_cpu_thread[NUM_FUNC];

 protected:
  ConfigBA();
  ///////////////////////////////
  void ResetTemporarySetting();
  void ResetBundleStatistics();
  void PrintBundleStatistics();
  void SaveBundleStatistics(int ncam, int npt, int nproj);
  ///////////////////////////////////////
  void BundleTimerStart(int timer);
  void BundleTimerSwitch(int timer);
  float BundleTimerGet(int timer);
  void BundleTimerSwap(int timer1, int timer2);
  float BundleTimerGetNow(int timer = TIMER_OPTIMIZATION);
  /////////////////////////////////
  void SaveBundleRecord(int iter, float res, float damping, float gn, float gi);
  bool IsTimeBudgetAvailable();
  double MyClock();

 public:
  void ParseParam(int argc, char** argv);

 public:
  // the following are to be called after finishing BA
  const char* GetOutputParam() { return __driver_output; }
  float GetInitialMSE() { return __initial_mse; }
  float GetFinalMSE() { return __final_mse; }
  double GetBundleTiming(int timer = TIMER_OVERALL) {
    return __timer_record[timer];
  }
  int GetIterationsLM() { return __num_lm_iteration; }
  int GetIterationsCG() { return __num_cg_iteration; }
  int GetCurrentDevice() { return __current_device; }
  int GetBundleReturnCode() { return __pba_return_code; }
  int GetActiveDevice() { return __selected_device; }
};

}  // namespace pba

#endif
