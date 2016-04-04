////////////////////////////////////////////////////////////////////////////
//  File:           ConfigBA.cpp
//  Author:         Changchang Wu
//  Description :   implementation of the configuration object class
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

#include <string.h>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <iomanip>
#include <fstream>
#include <string>

using std::cout;
using std::ofstream;
using std::string;

#ifndef _WIN32
#include <sys/time.h>
#endif

#include "ConfigBA.h"

#ifdef _MSC_VER
#define strcpy strcpy_s
#define sprintf sprintf_s
#endif

namespace pba {

ConfigBA::ConfigBA() {
  __lm_max_iteration = 50;
  __lm_initial_damp = 1e-3f;
  __lm_minimum_damp = 1e-10f;
  __lm_maximum_damp = 1e+5f;
  __lm_delta_threshold = 1e-6f;
  __lm_gradient_threshold = 1e-10f;
  __lm_mse_threshold = 0.25f;
  __lm_use_diagonal_damp = true;
  __lm_check_gradient = false;
  __lm_damping_auto_switch = 0;
  __bundle_time_budget = 0;
  __bundle_mode_next = 0;
  __bundle_current_mode = 0;

  ////////////////////////////
  __cg_max_iteration = 100;
  __cg_min_iteration = 10;
  __cg_recalculate_freq = 0;
  __cg_norm_threshold = 0.1f;
  __cg_norm_guard = 1.0f;
  __pba_experimental = 0;
  __cg_schur_complement = 0;

  ////////////////////////////
  __fixed_intrinsics = false;
  __use_radial_distortion = 0;
  __reset_initial_distortion = false;

  //////////////////////////////
  __verbose_level = 2;
  __verbose_cg_iteration = false;
  __verbose_function_time = false;
  __verbose_allocation = false;
  __verbose_sse = false;
  __save_gradient_norm = false;
  __stat_filename = NULL;
  __matlab_format_stat = true;

  /////////////////////////////
  __jc_store_transpose = true;
  __jc_store_original = true;
  __no_jacobian_store = false;

  __focal_normalize = true;
  __depth_normalize = true;
  __depth_degeneracy_fix = true;
  __jacobian_normalize = true;
  __data_normalize_median = 0.5f;
  __depth_check_epsilon = 0.01f;

  ////////////////////////////
  __multiply_jx_usenoj = true;

  ////////////////////////////
  __accurate_gain_ratio = true;
  ////////////////////////////
  __cpu_data_precision = 0;
  __current_device = -1;
  __selected_device = -1;
  __memory_usage = 0;
  __current_iteration = 0;
  __num_cpu_thread_all = 0;

  ///////////////////////
  __debug_pba = false;
  __profile_pba = 0;
  __cpu_thread_profile = false;
  __warmup_device = false;

  ///////////////////////
  __driver_output = NULL;

  //////////////////////////
  ResetBundleStatistics();
}

void ConfigBA::ResetBundleStatistics() {
  __abort_flag = false;
  __num_lm_success = 0;
  __num_lm_iteration = 0;
  __num_cg_iteration = 0;
  __num_projection_eval = 0;
  __num_jacobian_eval = 0;
  __num_camera_modified = 0;
  __num_point_behind = 0;
  __initial_mse = 0;
  __final_mse = 0;
  __final_mse_x = 0;
  __focal_scaling = 1.0f;
  __depth_scaling = 1.0f;
  __pba_return_code = 0;
  __current_iteration = 0;
  __warmup_device = false;
  __bundle_current_mode = __bundle_mode_next;
  for (int i = 0; i < NUM_TIMER; ++i) __timer_record[i] = 0;
  __bundle_records.resize(0);
  if (__num_cpu_thread_all) {
    std::cout << "WARNING: set all thread number to " << __num_cpu_thread_all
              << '\n';
    for (int i = 0; i < NUM_FUNC; ++i)
      __num_cpu_thread[i] = __num_cpu_thread_all;
  }
}

void ConfigBA::ResetTemporarySetting() {
  __reset_initial_distortion = false;
  __bundle_time_budget = 0;
  __bundle_mode_next = 0;
  __bundle_current_mode = 0;
  __stat_filename = NULL;
  if (__lm_damping_auto_switch > 0 && !__lm_use_diagonal_damp)
    __lm_use_diagonal_damp = true;
}

void ConfigBA::SaveBundleStatistics(int ncam, int npt, int nproj) {
  if (__profile_pba) return;
  if (__stat_filename && __bundle_records.size() > 0) {
    char filenamebuf[1024];
    char* ret = strchr(__stat_filename, '\r');
    if (ret) ret[0] = 0;
    char* dot = strrchr(__stat_filename, '.');
    if (dot && strchr(dot, '/') == NULL && strchr(dot, '\\') == NULL)
      strcpy(filenamebuf, __stat_filename);  // if filename has extension, use
                                             // it
    else
      sprintf(filenamebuf, "%s%s%s%s%s%s%s%s%s.%s", __stat_filename,
              __cpu_data_precision == 0 ? "_gpu" : "_cpu",
              __cpu_data_precision == sizeof(double) ? "d" : "",
              __cg_schur_complement ? "_schur" : "\0",
              __lm_use_diagonal_damp
                  ? "\0"
                  : (__lm_damping_auto_switch > 0 ? "_ad" : "_id"),
              __use_radial_distortion == -1
                  ? "_md"
                  : (__use_radial_distortion ? "_pd" : "\0"),
              __jacobian_normalize ? "\0" : "_nojn",
              __focal_normalize || __depth_normalize ? "\0" : "_nodn",
              __depth_degeneracy_fix ? "\0" : "_nodf",
              __matlab_format_stat ? "m" : "log");

    ///////////////////////////////////////////////////////
    ofstream out(filenamebuf);
    out << std::left;

    float overhead =
        (BundleTimerGet(TIMER_OVERALL) - BundleTimerGet(TIMER_OPTIMIZATION));
    if (__matlab_format_stat)
      out << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
          << "ncam = " << ncam << "; npt = " << npt << "; nproj = " << nproj
          << ";\n"
          << "%% overhead = " << overhead << ";\n"
          << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
          << "%% " << std::setw(10) << __num_lm_iteration
          << "\t linear systems solved;\n"
          << "%% " << std::setw(10) << __num_cg_iteration
          << "\t conjugated gradient steps;\n"
          << "%% " << std::setw(10) << BundleTimerGet(TIMER_OVERALL)
          << "\t seconds used overall;\n"
          << "%% " << std::setw(10) << BundleTimerGet(TIMER_PREPROCESSING)
          << "\t seconds on pre-processing;\n"
          << "%% " << std::setw(10)
          << BundleTimerGet(TIMER_GPU_UPLOAD) +
                 BundleTimerGet(TIMER_GPU_ALLOCATION)
          << "\t seconds on upload;\n"
          << "%% " << std::setw(10) << BundleTimerGet(TIMER_OPTIMIZATION)
          << "\t seconds on optimization;\n"
          << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
          << (__cpu_data_precision == 0 ? "gpustat" : "cpustat")
          << (__cpu_data_precision == sizeof(double) ? "_db" : "")
          << (__jacobian_normalize ? "" : "_nojn")
          << (__depth_degeneracy_fix ? "" : "_nodf")
          << (__cg_schur_complement ? "_schur" : "") << " = [\n";

    for (size_t i = 0; i < __bundle_records.size(); ++i)
      out << std::setw((i % 7 > 2) ? ((i % 7 > 4 && !__save_gradient_norm &&
                                       !__lm_check_gradient)
                                          ? 0
                                          : 12)
                                   : 5)
          << (__bundle_records[i] + (i == 1 ? overhead : 0))
          << (i % 7 == 6 ? '\n' : '\t');

    if (__matlab_format_stat) out << "];\n\n";

    if (__verbose_level)
      std::cout << "\n---------------------------------------\n" << filenamebuf;
  }
}

#define REPORT_FUNCTION_TIME(FID)                                         \
  std::setw(5) << (((int)(BundleTimerGet(FID) * 100 + 50)) * 0.01) << "(" \
               << std::setw(2)                                            \
               << 0.1f * ((int)(1000 * BundleTimerGet(FID) /              \
                                BundleTimerGet(TIMER_OPTIMIZATION)))      \
               << "%)"

void ConfigBA::PrintBundleStatistics() {
  if (__profile_pba) return;

  if (__verbose_level)
    std::cout << "\n---------------------------------------\n" << std::setw(10)
              << __num_lm_success << "\t successful iterations;\n"
              << std::setw(10) << __num_lm_iteration
              << "\t linear systems solved;\n" << std::setw(10)
              << __num_cg_iteration << "\t conjugated gradient steps;\n"
              << std::setw(10) << BundleTimerGet(TIMER_OVERALL)
              << "\t seconds used overall;\n" << std::setw(10)
              << BundleTimerGet(TIMER_GPU_ALLOCATION)
              << "\t seconds on allocation;\n" << std::setw(10)
              << BundleTimerGet(TIMER_PREPROCESSING)
              << "\t seconds on pre-processing;\n" << std::setw(10)
              << BundleTimerGet(TIMER_GPU_UPLOAD) << "\t seconds on upload;\n"
              << std::setw(10) << BundleTimerGet(TIMER_OPTIMIZATION)
              << "\t seconds on optimization;\n";
  if (__verbose_level && __cpu_data_precision)
    std::cout << REPORT_FUNCTION_TIME(TIMER_FUNCTION_JJ)
              << "\t seconds on jacobians;\n"
              << REPORT_FUNCTION_TIME(TIMER_FUNCTION_PJ)
              << "\t seconds on projections;\n"
              << REPORT_FUNCTION_TIME(TIMER_FUNCTION_JX)
              << "\t seconds on JX;\n"
              << REPORT_FUNCTION_TIME(TIMER_FUNCTION_JTE)
              << "\t seconds on JtE;\n"
              << REPORT_FUNCTION_TIME(TIMER_FUNCTION_BC)
              << "\t seconds to compute preconditioner;\n"
              << REPORT_FUNCTION_TIME(TIMER_FUNCTION_MP)
              << "\t seconds to apply preconditioner;\n"
              << REPORT_FUNCTION_TIME(TIMER_FUNCTION_UP)
              << "\t seconds to update parameters;\n";
  if (__verbose_level)
    std::cout << "---------------------------------------\n"
              << "mse = " << __initial_mse << " -> " << __final_mse << ""
              << "  (" << __final_mse_x
              << (__use_radial_distortion == -1 ? 'D' : 'U') << ")\n"
              << "---------------------------------------\n";
}

double ConfigBA::MyClock() {
#ifdef _WIN32
  return clock() / double(CLOCKS_PER_SEC);
#else
  static int started = 0;
  static struct timeval tstart;
  if (started == 0) {
    gettimeofday(&tstart, NULL);
    started = 1;
    return 0;
  } else {
    struct timeval now;
    gettimeofday(&now, NULL);
    return ((now.tv_usec - tstart.tv_usec) / 1000000.0 +
            (now.tv_sec - tstart.tv_sec));
  }
#endif
}

void ConfigBA::BundleTimerStart(int timer) {
  __timer_record[timer] = MyClock();
}

void ConfigBA::BundleTimerSwitch(int timer) {
  __timer_record[timer] = MyClock() - __timer_record[timer];
}

void ConfigBA::BundleTimerSwap(int timer1, int timer2) {
  BundleTimerSwitch(timer1);
  BundleTimerSwitch(timer2);
}

float ConfigBA::BundleTimerGet(int timer) {
  return float(__timer_record[timer]);
}

float ConfigBA::BundleTimerGetNow(int timer) {
  return 0.01f * ((int)(100 * (MyClock() - __timer_record[timer])));
}

bool ConfigBA::IsTimeBudgetAvailable() {
  if (__bundle_time_budget <= 0) return true;
  return BundleTimerGetNow(TIMER_OVERALL) < __bundle_time_budget;
}

void ConfigBA::SaveBundleRecord(int iter, float res, float damping, float gn,
                                float gi) {
  __bundle_records.push_back(float(iter));
  __bundle_records.push_back(BundleTimerGetNow());
  __bundle_records.push_back(float(__num_cg_iteration));
  __bundle_records.push_back(res);
  __bundle_records.push_back(damping);
  __bundle_records.push_back(gn);
  __bundle_records.push_back(gi);
}

void ConfigBA::ParseParam(int argc, char** argv) {
#define CHAR1_TO_INT(x) ((x >= 'A' && x <= 'Z') ? x + 32 : x)
#define CHAR2_TO_INT(str, i) \
  (str[i] ? CHAR1_TO_INT(str[i]) + (CHAR1_TO_INT(str[i + 1]) << 8) : 0)
#define CHAR3_TO_INT(str, i) \
  (str[i] ? CHAR1_TO_INT(str[i]) + (CHAR2_TO_INT(str, i + 1) << 8) : 0)
#define STRING_TO_INT(str) (CHAR1_TO_INT(str[0]) + (CHAR3_TO_INT(str, 1) << 8))

#ifdef _MSC_VER
// charizing is microsoft only
#define MAKEINT1(a) (#@ a)
#define sscanf sscanf_s
#else
#define mychar0 '0'
#define mychar1 '1'
#define mychar2 '2'
#define mychar3 '3'
#define mychara 'a'
#define mycharb 'b'
#define mycharc 'c'
#define mychard 'd'
#define mychare 'e'
#define mycharf 'f'
#define mycharg 'g'
#define mycharh 'h'
#define mychari 'i'
#define mycharj 'j'
#define mychark 'k'
#define mycharl 'l'
#define mycharm 'm'
#define mycharn 'n'
#define mycharo 'o'
#define mycharp 'p'
#define mycharq 'q'
#define mycharr 'r'
#define mychars 's'
#define mychart 't'
#define mycharu 'u'
#define mycharv 'v'
#define mycharw 'w'
#define mycharx 'x'
#define mychary 'y'
#define mycharz 'z'
#define MAKEINT1(a) (mychar##a)
#endif
#define MAKEINT2(a, b) (MAKEINT1(a) + (MAKEINT1(b) << 8))
#define MAKEINT3(a, b, c) (MAKEINT1(a) + (MAKEINT2(b, c) << 8))
#define MAKEINT4(a, b, c, d) (MAKEINT1(a) + (MAKEINT3(b, c, d) << 8))

  char *arg, *param, *opt;
  int opti, argi;
  float argf;
  for (int i = 0; i < argc; i++) {
    arg = argv[i];
    if (arg == NULL || arg[0] != '-' || !arg[1]) continue;
    opt = arg + 1;
    opti = STRING_TO_INT(opt);
    param = argv[i + 1];

    ////////////////////////////////
    switch (opti) {
      case MAKEINT3(l, m, i):
        if (i + 1 < argc && sscanf(param, "%d", &argi) && argi > 0)
          __lm_max_iteration = argi;
        break;
      case MAKEINT3(l, m, d):
        if (i + 1 < argc && sscanf(param, "%f", &argf) && argf >= 0)
          __lm_delta_threshold = argf;
        break;
      case MAKEINT3(l, m, e):
        if (i + 1 < argc && sscanf(param, "%f", &argf) && argf >= 0)
          __lm_mse_threshold = argf;
        break;
      case MAKEINT3(l, m, g):
        if (i + 1 < argc && sscanf(param, "%f", &argf) && argf > 0)
          __lm_gradient_threshold = argf;
        break;
      case MAKEINT4(d, a, m, p):
        if (i + 1 < argc && sscanf(param, "%f", &argf) && argf > 0)
          __lm_initial_damp = argf;
        break;
      case MAKEINT4(d, m, i, n):
        if (i + 1 < argc && sscanf(param, "%f", &argf) && argf > 0)
          __lm_minimum_damp = argf;
        break;
      case MAKEINT4(d, m, a, x):
        if (i + 1 < argc && sscanf(param, "%f", &argf) && argf > 0)
          __lm_maximum_damp = argf;
        break;
      case MAKEINT3(c, g, i):
        if (i + 1 < argc && sscanf(param, "%d", &argi) && argi > 0)
          __cg_max_iteration = argi;
        break;
      case MAKEINT4(c, g, i, m):
        if (i + 1 < argc && sscanf(param, "%d", &argi) && argi > 0)
          __cg_min_iteration = argi;
        break;
      case MAKEINT3(c, g, n):
        if (i + 1 < argc && sscanf(param, "%f", &argf) && argf > 0)
          __cg_norm_threshold = argf;
        break;
      case MAKEINT3(c, g, g):
        if (i + 1 < argc && sscanf(param, "%f", &argf) && argf > 0)
          __cg_norm_guard = argf;
        break;
      case MAKEINT4(c, g, r, f):
        if (i + 1 < argc && sscanf(param, "%d", &argi) && argi > 0)
          __cg_recalculate_freq = argi;
        break;
      case MAKEINT1(v):
        if (i + 1 < argc && sscanf(param, "%d", &argi) && argi >= 0)
          __verbose_level = argi;
        break;
      case MAKEINT4(d, e, v, i):
        if (i + 1 < argc && sscanf(param, "%d", &argi) && argi >= 0)
          __selected_device = argi;
        break;
      case MAKEINT4(b, u, d, g):
        if (i + 1 < argc && sscanf(param, "%d", &argi) && argi >= 0)
          __bundle_time_budget = argi;
        break;
      case MAKEINT3(e, x, p):
        if (i + 1 < argc && sscanf(param, "%d", &argi) && argi >= 0)
          __pba_experimental = argi;
        break;
      case MAKEINT4(t, n, u, m):
        if (i + 1 < argc && sscanf(param, "%d", &argi) && argi > 0)
          __num_cpu_thread_all = argi;
        break;
      case MAKEINT4(p, r, o, f):
        __profile_pba = (i + 1 < argc && sscanf(param, "%d", &argi))
                            ? std::max(10, argi)
                            : 100;
        break;
      case MAKEINT4(t, p, r, o):
        __cpu_thread_profile = true;
        break;
      case MAKEINT4(c, a, l, i):
        __fixed_intrinsics = true;
        break;
      case MAKEINT4(s, c, h, u):
      case MAKEINT4(s, s, o, r):
        __cg_schur_complement = true;
        break;
      case MAKEINT2(m, d):
      case MAKEINT4(r, a, d, i):
        __use_radial_distortion = -1;
        break;
      case MAKEINT2(p, d):
        __use_radial_distortion = 1;
        break;
      case MAKEINT3(r, 0, 0):
        __reset_initial_distortion = true;
        break;
      case MAKEINT4(v, a, r, i):
        __fixed_intrinsics = false;
        break;
      case MAKEINT4(n, a, c, c):
        __accurate_gain_ratio = false;
        break;
      case MAKEINT4(v, c, g, i):
        __verbose_cg_iteration = true;
        break;
      case MAKEINT4(v, f, u, n):
        __verbose_function_time = true;
        break;
      case MAKEINT4(v, a, l, l):
        __verbose_allocation = true;
        break;
      case MAKEINT4(v, s, s, e):
        __verbose_sse = true;
        break;
      case MAKEINT4(s, v, g, n):
        __save_gradient_norm = true;
        break;
      case MAKEINT2(i, d):
        __lm_use_diagonal_damp = false;
        break;
      case MAKEINT3(d, a, s):
        if (i + 1 < argc && sscanf(param, "%f", &argf) && argf > 0)
          __lm_damping_auto_switch = std::max(argf, 0.1f);
        else
          __lm_damping_auto_switch = 2.0f;
        break;
      case MAKEINT4(c, h, k, g):
        __lm_check_gradient = true;
        break;
      case MAKEINT4(n, o, j, n):
        __jacobian_normalize = false;
        break;
      case MAKEINT2(n, j):
        __no_jacobian_store = true;
      case MAKEINT3(n, j, c):
        __jc_store_transpose = false;
        __jc_store_original = false;
        break;
      case MAKEINT4(n, j, c, o):
        __jc_store_original = false;
        break;
      case MAKEINT4(n, j, c, t):
        __jc_store_transpose = false;
        break;
      case MAKEINT3(j, x, j):
        __multiply_jx_usenoj = false;
        break;
      case MAKEINT4(j, x, n, j):
        __multiply_jx_usenoj = true;
        break;
      case MAKEINT4(n, o, d, n):
        __depth_normalize = false;
        __focal_normalize = false;
        break;
      case MAKEINT4(n, o, d, f):
        __depth_degeneracy_fix = false;
        break;
      case MAKEINT4(n, o, r, m):
        if (i + 1 < argc && sscanf(param, "%f", &argf) && argf > 0)
          __data_normalize_median = argf;
        break;
      case MAKEINT3(d, c, e):
        if (i + 1 < argc && sscanf(param, "%f", &argf) && argf > 0 &&
            argf <= 0.01)
          __depth_check_epsilon = argf;
        break;
      case MAKEINT4(d, e, b, u):
        __debug_pba = true;
        break;
      case MAKEINT4(e, v, a, l):
        __lm_max_iteration = 100;
        __warmup_device = true;
      case MAKEINT4(s, t, a, t):
        __stat_filename = (i + 1 < argc && param[0] != '-') ? param : NULL;
        break;
      case MAKEINT3(o, u, t):
        __driver_output = (i + 1 < argc && param[0] != '-') ? param : NULL;
        break;
      case MAKEINT4(w, a, r, m):
        __warmup_device = true;
        break;
      case MAKEINT4(m, o, t, i):
        __bundle_mode_next = 1;
        break;
      case MAKEINT4(s, t, r, u):
        __bundle_mode_next = 2;
        break;
    }
  }
}

}  // namespace pba
