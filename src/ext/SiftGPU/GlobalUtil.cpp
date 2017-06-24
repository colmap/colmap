////////////////////////////////////////////////////////////////////////////
//	File:		GlobalUtil.cpp
//	Author:		Changchang Wu
//	Description : Global Utility class for SiftGPU
//
//
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
#include <string.h>
#include <iostream>
using std::cout;

#include "GL/glew.h"
#include "GlobalUtil.h"

#if defined(_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <stdio.h>
#endif

#include "LiteWindow.h"

//
int GlobalParam::		_verbose =  1;
int	GlobalParam::       _timingS = 1;  //print out information of each step
int	GlobalParam::       _timingO = 0;  //print out information of each octave
int	GlobalParam::       _timingL = 0;	//print out information of each level
GLuint GlobalParam::	_texTarget = GL_TEXTURE_RECTANGLE_ARB; //only this one is supported
GLuint GlobalParam::	_iTexFormat =GL_RGBA32F_ARB;	//or GL_RGBA16F_ARB
int	GlobalParam::		_debug = 0;		//enable debug code?
int	GlobalParam::		_usePackedTex = 1;//packed implementation
int	GlobalParam::		_UseCUDA = 0;
int GlobalParam::       _UseOpenCL = 0;
int GlobalParam::		_MaxFilterWidth = -1;	//maximum filter width, use when GPU is not good enough
float GlobalParam::     _FilterWidthFactor	= 4.0f;	//the filter size will be _FilterWidthFactor*sigma*2+1
float GlobalParam::     _DescriptorWindowFactor = 3.0f; //descriptor sampling window factor
int GlobalParam::		_SubpixelLocalization = 1; //sub-pixel and sub-scale localization
int	GlobalParam::       _MaxOrientation = 2;	//whether we find multiple orientations for each feature
int	GlobalParam::       _OrientationPack2 = 0;  //use one float to store two orientations
float GlobalParam::		_MaxFeaturePercent = 0.005f;//at most 0.005 of all pixels
int	GlobalParam::		_MaxLevelFeatureNum = 4096; //maximum number of features of a level
int GlobalParam::		_FeatureTexBlock = 4; //feature texture storagte alignment
int	GlobalParam::		_NarrowFeatureTex = 0;

//if _ForceTightPyramid is not 0, pyramid will be reallocated to fit the size of input images.
//otherwise, pyramid can be reused for smaller input images.
int GlobalParam::		_ForceTightPyramid = 0;

//use gpu or cpu to generate feature list ...gpu is a little bit faster
int GlobalParam::		_ListGenGPU =	1;
int	GlobalParam::       _ListGenSkipGPU = 6;  //how many levels are skipped on gpu
int GlobalParam::		_PreProcessOnCPU = 1; //convert rgb 2 intensity on gpu, down sample on GPU

//hardware parameter,   automatically retrieved
int GlobalParam::		_texMaxDim = 3200;	//Maximum working size for SiftGPU, 3200 for packed
int	GlobalParam::		_texMaxDimGL = 4096;        //GPU texture limit
int GlobalParam::       _texMinDim = 16; //
int	GlobalParam::		_MemCapGPU = 0;
int GlobalParam::		_FitMemoryCap = 0;
int	GlobalParam::		_IsNvidia = 0;				//GPU vendor
int GlobalParam::		_KeepShaderLoop = 0;

//you can't change the following 2 values
//all other versions of code are now dropped
int GlobalParam::       _DescriptorPPR = 8;
int	GlobalParam::		_DescriptorPPT = 16;

//whether orientation/descriptor is supported by hardware
int GlobalParam::		_SupportNVFloat = 0;
int GlobalParam::       _SupportTextureRG = 0;
int	GlobalParam::		_UseDynamicIndexing = 0;
int GlobalParam::		_FullSupported = 1;

//when SiftGPUEX is not used, display VBO generation is skipped
int GlobalParam::		_UseSiftGPUEX = 0;
int GlobalParam::		_InitPyramidWidth=0;
int GlobalParam::		_InitPyramidHeight=0;
int	GlobalParam::		_octave_min_default=0;
int	GlobalParam::		_octave_num_default=-1;


//////////////////////////////////////////////////////////////////
int	GlobalParam::		_GoodOpenGL = -1;      //indicates OpenGl initialization status
int	GlobalParam::		_FixedOrientation = 0; //upright
int	GlobalParam::		_LoweOrigin = 0;       //(0, 0) to be at the top-left corner.
int	GlobalParam::       _NormalizedSIFT = 1;   //normalize descriptor
int GlobalParam::       _BinarySIFT = 0;       //saving binary format
int	GlobalParam::		_ExitAfterSIFT = 0;    //exif after saving result
int	GlobalParam::		_KeepExtremumSign = 0; // if 1, scales of dog-minimum will be multiplied by -1
///
int GlobalParam::       _KeyPointListForceLevel0 = 0;
int GlobalParam::		_DarknessAdaption = 0;
int	GlobalParam::		_ProcessOBO = 0;
int GlobalParam::       _TruncateMethod = 0;
int	GlobalParam::		_PreciseBorder = 1;

// parameter changing for better matching with Lowe's SIFT
float GlobalParam::		_OrientationWindowFactor = 2.0f;	// 1.0(-v292), 2(v293-),
float GlobalParam::		_OrientationGaussianFactor = 1.5f;	// 4.5(-v292), 1.5(v293-)
float GlobalParam::     _MulitiOrientationThreshold = 0.8f;
///
int GlobalParam::       _FeatureCountThreshold = -1;

///////////////////////////////////////////////
int	GlobalParam::			_WindowInitX = -1;
int GlobalParam::			_WindowInitY = -1;
int GlobalParam::           _DeviceIndex = 0;
const char * GlobalParam::	_WindowDisplay = NULL;



/////////////////
////
ClockTimer GlobalUtil::	_globalTimer;


#ifdef _DEBUG
void GlobalUtil::CheckErrorsGL(const char* location)
{
	GLuint errnum;
	const char *errstr;
	while (errnum = glGetError())
	{
		errstr = (const char *)(gluErrorString(errnum));
		if(errstr) {
			std::cerr << errstr;
		}
		else {
			std::cerr  << "Error " << errnum;
		}

		if(location) std::cerr  << " at " << location;
		std::cerr  << "\n";
	}
	return;
}

#endif

void GlobalUtil::CleanupOpenGL()
{
	glActiveTexture(GL_TEXTURE0);
}

void GlobalUtil::SetDeviceParam(int argc, char** argv)
{
    if(GlobalParam::_GoodOpenGL!= -1) return;

    #define CHAR1_TO_INT(x)         ((x >= 'A' && x <= 'Z') ? x + 32 : x)
    #define CHAR2_TO_INT(str, i)    (str[i] ? CHAR1_TO_INT(str[i]) + (CHAR1_TO_INT(str[i+1]) << 8) : 0)
    #define CHAR3_TO_INT(str, i)    (str[i] ? CHAR1_TO_INT(str[i]) + (CHAR2_TO_INT(str, i + 1) << 8) : 0)
    #define STRING_TO_INT(str)      (CHAR1_TO_INT(str[0]) +  (CHAR3_TO_INT(str, 1) << 8))

	char* arg, * opt;
	for(int i = 0; i< argc; i++)
	{
		arg = argv[i];
		if(arg == NULL || arg[0] != '-')continue;
		opt = arg+1;

        ////////////////////////////////
        switch( STRING_TO_INT(opt))
        {
        case 'w' + ('i' << 8) + ('n' << 16) + ('p' << 24):
            if(_GoodOpenGL != 2 && i + 1 < argc)
            {
                int x =0, y=0;
                if(sscanf(argv[++i], "%dx%d", &x, &y) == 2)
                {
                    GlobalParam::_WindowInitX = x;
                    GlobalParam::_WindowInitY = y;
                }
            }
            break;
        case 'd' + ('i' << 8) + ('s' << 16) + ('p' << 24):
            if(_GoodOpenGL != 2 && i + 1 < argc)
            {
                GlobalParam::_WindowDisplay = argv[++i];
            }
            break;
        case 'c' + ('u' << 8) + ('d' << 16) + ('a' << 24):
            if(i + 1 < argc)
            {
               int device =  0;
               scanf(argv[++i], "%d", &device) ;
               GlobalParam::_DeviceIndex = device;
            }
            break;
        default:
            break;
        }
    }
}

void GlobalUtil::SetTextureParameter()
{

	glTexParameteri (_texTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri (_texTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(_texTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(_texTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
}

//if image need to be up sampled ..use this one

void GlobalUtil::SetTextureParameterUS()
{

	glTexParameteri (_texTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri (_texTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(_texTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(_texTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
}


void GlobalUtil::FitViewPort(int width, int height)
{
	GLint port[4];
	glGetIntegerv(GL_VIEWPORT, port);
	if(port[2] !=width || port[3] !=height)
	{
		glViewport(0, 0, width, height);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0, width, 0, height,  0, 1);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
	}
}


bool GlobalUtil::CheckFramebufferStatus() {
    GLenum status;
    status=(GLenum)glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
    switch(status) {
        case GL_FRAMEBUFFER_COMPLETE_EXT:
            return true;
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
            std::cerr<<("Framebuffer incomplete,incomplete attachment\n");
            return false;
        case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
            std::cerr<<("Unsupported framebuffer format\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
            std::cerr<<("Framebuffer incomplete,missing attachment\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
            std::cerr<<("Framebuffer incomplete,attached images must have same dimensions\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
             std::cerr<<("Framebuffer incomplete,attached images must have same format\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
            std::cerr<<("Framebuffer incomplete,missing draw buffer\n");
            return false;
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
            std::cerr<<("Framebuffer incomplete,missing read buffer\n");
            return false;
    }
	return false;
}


int ClockTimer::ClockMS()
{
	return 0;
}

double ClockTimer::CLOCK()
{
	return 0;
}

void ClockTimer::InitHighResolution()
{
}

void ClockTimer::StartTimer(const char* event, int verb)
{

}

void ClockTimer::StopTimer(int verb)
{

}

float ClockTimer::GetElapsedTime()
{
	return 0;
}

void GlobalUtil::SetGLParam()
{
    if(GlobalUtil::_UseCUDA) return;
    else if(GlobalUtil::_UseOpenCL) return;
	glEnable(GlobalUtil::_texTarget);
	glActiveTexture(GL_TEXTURE0);
}

void GlobalUtil::InitGLParam(int NotTargetGL)
{
    //IF the OpenGL context passed the check
    if(GlobalUtil::_GoodOpenGL == 2) return;
    //IF the OpenGl context failed the check
    if(GlobalUtil::_GoodOpenGL == 0) return;
    //IF se use CUDA or OpenCL
    if(NotTargetGL && !GlobalUtil::_UseSiftGPUEX)
    {
        GlobalUtil::_GoodOpenGL = 1;
    }else
    {
        //first time in this function
        glewInit();

	    GlobalUtil::_GoodOpenGL = 2;

	    const char * vendor = (const char * )glGetString(GL_VENDOR);
	    if(vendor)
	    {
		    GlobalUtil::_IsNvidia  = (strstr(vendor, "NVIDIA") !=NULL ? 1 : 0);

			// Let nVidia compiler to take care of the unrolling.
			if (GlobalUtil::_IsNvidia) 		GlobalUtil::_KeepShaderLoop = 1;

#ifndef WIN32
			else if(!strstr(vendor, "ATI") )
			{
				// For non-nVidia non-ATI cards...simply assume it is Mesa
				// Keep the original shader loop, because some of the unrolled
				// loopes are too large, and it may take too much time to compile
				GlobalUtil::_KeepShaderLoop = 1;
			}
#endif

			if(GlobalUtil::_IsNvidia && glewGetExtension("GL_NVX_gpu_memory_info"))
			{
				glGetIntegerv(0x9049/*GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX*/, &_MemCapGPU);
				_MemCapGPU /= (1024);
			  if(GlobalUtil::_verbose) std::cout << "[GPU VENDOR]:\t" << vendor << ' ' <<_MemCapGPU << "MB\n";
			}else if(strstr(vendor, "ATI") && glewGetExtension("GL_ATI_meminfo"))
			{
				int info[4]; 	glGetIntegerv(0x87FC/*GL_TEXTURE_FREE_MEMORY_ATI*/, info);
				_MemCapGPU = info[0] / (1024);
			    if(GlobalUtil::_verbose) std::cout << "[GPU VENDOR]:\t" << vendor << ' ' <<_MemCapGPU << "MB\n";
			}else
			{
				if(GlobalUtil::_verbose) std::cout << "[GPU VENDOR]:\t" << vendor << "\n";
			}

	    }
	    if(GlobalUtil::_IsNvidia == 0 )GlobalUtil::_UseCUDA = 0;

	    if (glewGetExtension("GL_ARB_fragment_shader")    != GL_TRUE ||
		    glewGetExtension("GL_ARB_shader_objects")       != GL_TRUE ||
		    glewGetExtension("GL_ARB_shading_language_100") != GL_TRUE)
	    {
		    std::cerr << "Shader not supported by your hardware!\n";
		    GlobalUtil::_GoodOpenGL = 0;
	    }

	    if (glewGetExtension("GL_EXT_framebuffer_object") != GL_TRUE)
	    {
		    std::cerr << "Framebuffer object not supported!\n";
		    GlobalUtil::_GoodOpenGL = 0;
	    }

	    if(glewGetExtension("GL_ARB_texture_rectangle")==GL_TRUE)
	    {
	        GLint value;
		    GlobalUtil::_texTarget =  GL_TEXTURE_RECTANGLE_ARB;
		    glGetIntegerv(GL_MAX_RECTANGLE_TEXTURE_SIZE_EXT, &value);
		    GlobalUtil::_texMaxDimGL = value;
		    if(GlobalUtil::_verbose) std::cout << "TEXTURE:\t" << GlobalUtil::_texMaxDimGL << "\n";

		    if(GlobalUtil::_texMaxDim == 0 || GlobalUtil::_texMaxDim > GlobalUtil::_texMaxDimGL)
		    {
			    GlobalUtil::_texMaxDim = GlobalUtil::_texMaxDimGL;
		    }
		    glEnable(GlobalUtil::_texTarget);
	    }else
	    {
		    std::cerr << "GL_ARB_texture_rectangle not supported!\n";
		    GlobalUtil::_GoodOpenGL = 0;
	    }

	    GlobalUtil::_SupportNVFloat = glewGetExtension("GL_NV_float_buffer");
	    GlobalUtil::_SupportTextureRG = glewGetExtension("GL_ARB_texture_rg");


	    glShadeModel(GL_FLAT);
	    glPolygonMode(GL_FRONT, GL_FILL);

	    GlobalUtil::SetTextureParameter();

    }
}

void GlobalUtil::SelectDisplay()
{
#ifdef WIN32
	if(_WindowDisplay == NULL) return;

	HDC hdc = CreateDC(_WindowDisplay, _WindowDisplay, NULL, NULL);
	_WindowDisplay = NULL;
	if(hdc == NULL)
	{
		std::cout << "ERROR: invalid dispaly specified\n";
		return;
	}

	PIXELFORMATDESCRIPTOR pfd =
	{
		sizeof(PIXELFORMATDESCRIPTOR),1,
		PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL|PFD_DOUBLEBUFFER,
		PFD_TYPE_RGBA,24,0, 0, 0, 0, 0, 0,0,0,0,0, 0, 0, 0,16,0,0,
		PFD_MAIN_PLANE,0,0, 0, 0
	};
	ChoosePixelFormat(hdc, &pfd);
#endif
}

int GlobalUtil::CreateWindowEZ(LiteWindow* window)
{
	if(window == NULL) return 0;
    if(!window->IsValid())window->Create(_WindowInitX, _WindowInitY, _WindowDisplay);
    if(window->IsValid())
    {
        window->MakeCurrent();
        return 1;
    }
    else
    {
        std::cerr << "Unable to create OpenGL Context!\n";
		std::cerr << "For nVidia cards, you can try change to CUDA mode in this case\n";
        return 0;
    }
}

int GlobalUtil::CreateWindowEZ()
{
	static LiteWindow window;
    return CreateWindowEZ(&window);
}

int CreateLiteWindow(LiteWindow* window)
{
    return GlobalUtil::CreateWindowEZ(window);
}
