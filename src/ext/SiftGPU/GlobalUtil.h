////////////////////////////////////////////////////////////////////////////
//	File:		GlobalUtil.h
//	Author:		Changchang Wu
//	Description : 
//		GlobalParam:	Global parameters
//		ClockTimer:		Timer 
//		GlobalUtil:		Global Function wrapper
//
//	Copyright (c) 2007 University of North Carolina at Chapel Hill
//	All Rights Reserved
//
//	Permission to use, copy, modify and distribute this software and its
//	documentation for educational, research and non-profit purposes, without
//	fee, and without a written agreement is hereby granted, provided that the
//	above copyright notice and the following paragraph appear in all copies.
//
//	The University of North Carolina at Chapel Hill make no representations
//	about the suitability of this software for any purpose. It is provided
//	'as is' without express or implied warranty. 
//
//	Please send BUG REPORTS to ccwu@cs.unc.edu
//
////////////////////////////////////////////////////////////////////////////


#ifndef _GLOBAL_UTILITY_H
#define _GLOBAL_UTILITY_H


//wrapper for some shader function
//class ProgramGPU;
class LiteWindow;

class GlobalParam
{
public:
	static GLuint	_texTarget;
	static GLuint   _iTexFormat;
	static int		_texMaxDim;
	static int		_texMaxDimGL; 
    static int      _texMinDim;
	static int		_MemCapGPU;
	static int		_FitMemoryCap;
	static int		_verbose;
	static int		_timingS;
	static int		_timingO;
	static int		_timingL;
	static int		_usePackedTex;
	static int		_IsNvidia;
	static int		_KeepShaderLoop;
	static int		_UseCUDA;
    static int      _UseOpenCL;
	static int		_UseDynamicIndexing; 
	static int		_debug;
	static int		_MaxFilterWidth;
	static float	_FilterWidthFactor;
	static float    _OrientationWindowFactor;
	static float	_DescriptorWindowFactor; 
	static int		_MaxOrientation;
	static int      _OrientationPack2;
	static int		_ListGenGPU;
	static int		_ListGenSkipGPU;
	static int		_SupportNVFloat;
	static int		_SupportTextureRG;
	static int		_FullSupported;
	static float	_MaxFeaturePercent;
	static int		_MaxLevelFeatureNum;
	static int		_DescriptorPPR; 
	static int		_DescriptorPPT; //pixel per texture for one descriptor
	static int		_FeatureTexBlock;
	static int		_NarrowFeatureTex; //implemented but no performance improvement
	static int		_SubpixelLocalization;
	static int		_ProcessOBO; //not implemented yet
    static int      _TruncateMethod;
	static int		_PreciseBorder; //implemented
	static int		_UseSiftGPUEX;
	static int		_ForceTightPyramid;
	static int		_octave_min_default;
	static int		_octave_num_default;
	static int		_InitPyramidWidth;
	static int		_InitPyramidHeight;
	static int		_PreProcessOnCPU;
	static int		_GoodOpenGL;
	static int		_FixedOrientation;
	static int		_LoweOrigin;
	static int		_ExitAfterSIFT; 
	static int		_NormalizedSIFT;
	static int		_BinarySIFT;
	static int		_KeepExtremumSign;
	static int		_FeatureCountThreshold;
    static int      _KeyPointListForceLevel0;
	static int		_DarknessAdaption;

	//for compatability with old version:
	static float	_OrientationExtraFactor;
	static float	_OrientationGaussianFactor;
	static float    _MulitiOrientationThreshold;

	////////////////////////////////////////
	static int				_WindowInitX;
	static int				_WindowInitY;
	static const char*		_WindowDisplay;
    static int              _DeviceIndex; 
};


class ClockTimer
{
private:
	char _current_event[256];
	int  _time_start;
	int  _time_stop;
public:
	static int	  ClockMS();
	static double CLOCK();
	static void	  InitHighResolution();
	void StopTimer(int verb = 1);
	void StartTimer(const char * event, int verb=0);
	float  GetElapsedTime();
};

class GlobalUtil:public GlobalParam
{
    static ClockTimer _globalTimer;                             
public:
	inline static double CLOCK()				{	return ClockTimer::CLOCK();			}
	inline static void StopTimer()				{	_globalTimer.StopTimer(_timingS);			}
	inline static void StartTimer(const char * event)	{	_globalTimer.StartTimer(event, _timingO);	}
	inline static float GetElapsedTime()		{	return _globalTimer.GetElapsedTime();		}

	static void FitViewPort(int width, int height);
	static void SetTextureParameter();
	static void SetTextureParameterUS();
#ifdef _DEBUG
	static void CheckErrorsGL(const char* location = NULL);
#else
	static void inline CheckErrorsGL(const char* location = NULL){};
#endif
	static bool CheckFramebufferStatus();
	//initialize Opengl parameters
	static void SelectDisplay();
	static void InitGLParam(int NotTargetGL = 0);
	static void SetGLParam();
	static int  CreateWindowEZ();
	static void CleanupOpenGL();
    static void SetDeviceParam(int argc, char** argv);
    static int  CreateWindowEZ(LiteWindow* window);
};


#if defined(_MSC_VER) && _MSC_VER == 1200
#define max(a,b)    (((a) > (b)) ? (a) : (b))
#define min(a,b)    (((a) < (b)) ? (a) : (b))
#endif

#endif

