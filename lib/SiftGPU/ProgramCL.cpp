//////////////////////////////////////////////////////////////////////////////
//	File:		ProgramCL.cpp
//	Author:		Changchang Wu
//	Description :	implementation of CL related class.
//		            class ProgramCL			A simple wrapper of Cg programs
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

#if defined(CL_SIFTGPU_ENABLED) 

#include <CL/opencl.h>
#include <GL/glew.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <strstream>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <string.h>
using namespace std;

#include "GlobalUtil.h"
#include "CLTexImage.h"
#include "ProgramCL.h"
#include "SiftGPU.h"


#if  defined(_WIN32) 
	#pragma comment (lib, "OpenCL.lib")
#endif

#ifndef _INC_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
	#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif 

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

ProgramCL::ProgramCL()
{
	_program = NULL;
    _kernel = NULL;
    _valid = 0;
}

ProgramCL::~ProgramCL()
{
    if(_kernel)  clReleaseKernel(_kernel);
    if(_program) clReleaseProgram(_program);
}

ProgramCL::ProgramCL(const char* name, const char * code, cl_context context, cl_device_id device) : _valid(1)
{
    const char * src[1] = {code};     cl_int status;

    _program = clCreateProgramWithSource(context, 1, src, NULL, &status);
    if(status != CL_SUCCESS) _valid = 0;

    status = clBuildProgram(_program, 0, NULL, 
        GlobalUtil::_debug ? 
        "-cl-fast-relaxed-math -cl-single-precision-constant -cl-nv-verbose" : 
        "-cl-fast-relaxed-math -cl-single-precision-constant", NULL, NULL);

    if(status != CL_SUCCESS) {PrintBuildLog(device, 1); _valid = 0;}
    else if(GlobalUtil::_debug) PrintBuildLog(device, 0); 

    _kernel = clCreateKernel(_program, name, &status); 
    if(status != CL_SUCCESS) _valid = 0;
}

void ProgramCL::PrintBuildLog(cl_device_id device, int all)
{
    char buffer[10240] = "\0"; 
    cl_int status = clGetProgramBuildInfo(
        _program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
    if(all )  
    {
        std::cerr << buffer  << endl; 
    }else
    {
        const char * pos = strstr(buffer, "ptxas");
        if(pos) std::cerr << pos << endl; 
    }
}

///////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////PACKED VERSION?///////////////////////////////////

ProgramBagCL::ProgramBagCL()
{
    ////////////////////////////////////
    _context = NULL;   _queue = NULL;
    s_gray = s_sampling = NULL;
    s_packup = s_zero_pass = NULL;
    s_gray_pack = s_unpack = NULL;
    s_sampling_u = NULL;
    s_dog_pass   = NULL;
    s_grad_pass  = NULL;
    s_grad_pass2 = NULL;
    s_unpack_dog = NULL;
    s_unpack_grd = NULL;
    s_unpack_key = NULL;
    s_keypoint = NULL;
    f_gaussian_skip0 = NULL;
    f_gaussian_skip1 = NULL;
    f_gaussian_step = 0;

    ////////////////////////////////
	GlobalUtil::StartTimer("Initialize OpenCL");
    if(!InitializeContext()) return;
    GlobalUtil::StopTimer();

}



ProgramBagCL::~ProgramBagCL()
{
    if(s_gray)      delete s_gray;
	if(s_sampling)  delete s_sampling;
	if(s_zero_pass) delete s_zero_pass;
    if(s_packup)    delete s_packup;
    if(s_unpack)    delete s_unpack;
    if(s_gray_pack) delete s_gray_pack;
    if(s_sampling_u)  delete s_sampling_u;
    if(s_dog_pass)  delete s_dog_pass;
    if(s_grad_pass)  delete s_grad_pass;
    if(s_grad_pass2)  delete s_grad_pass2;
    if(s_unpack_dog) delete s_unpack_dog;
    if(s_unpack_grd) delete s_unpack_grd;
    if(s_unpack_key) delete s_unpack_key;
    if(s_keypoint)   delete s_keypoint;

    if(f_gaussian_skip1) delete f_gaussian_skip1;

    for(unsigned int i = 0; i < f_gaussian_skip0_v.size(); i++)
    {
	    if(f_gaussian_skip0_v[i]) delete f_gaussian_skip0_v[i];
    }
    if(f_gaussian_step && _gaussian_step_num > 0) 
    {
	    for(int i = 0; i< _gaussian_step_num; i++)
	    {
		    delete f_gaussian_step[i];
	    }
	    delete[] f_gaussian_step;
    }

    //////////////////////////////////////
    if(_context) clReleaseContext(_context);
    if(_queue) clReleaseCommandQueue(_queue);
}

bool ProgramBagCL::InitializeContext()
{
    cl_uint num_platform, num_device;
    cl_int status;
    // Get OpenCL platform count
    status = clGetPlatformIDs (0, NULL, &num_platform);
    if (status != CL_SUCCESS || num_platform == 0) return false; 

    cl_platform_id platforms[16];
    if(num_platform > 16 ) num_platform = 16;
    status = clGetPlatformIDs (num_platform, platforms, NULL);
    _platform = platforms[0];

    ///////////////////////////////
    status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_device);
    if(status != CL_SUCCESS || num_device == 0) return false;

    // Create the device list
    cl_device_id* devices = new cl_device_id [num_device];
    status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_GPU, num_device, devices, NULL);
    _device = (status == CL_SUCCESS? devices[0] : 0);  delete[] devices;   
    if(status != CL_SUCCESS)  return false;  


    if(GlobalUtil::_verbose)
    {
        cl_device_mem_cache_type is_gcache; 
        clGetDeviceInfo(_device, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(is_gcache), &is_gcache, NULL);
        if(is_gcache == CL_NONE) std::cout << "No cache for global memory\n";
        //else if(is_gcache == CL_READ_ONLY_CACHE) std::cout << "Read only cache for global memory\n";
        //else std::cout << "Read/Write cache for global memory\n";
    }

    //context;
    if(GlobalUtil::_UseSiftGPUEX)
    {
        cl_context_properties prop[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)_platform,
        CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
        CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),  0 };
        _context = clCreateContext(prop, 1, &_device, NULL, NULL, &status);    
        if(status != CL_SUCCESS) return false;
    }else
    {
        _context = clCreateContext(0, 1, &_device, NULL, NULL, &status);    
        if(status != CL_SUCCESS) return false;
    }

    //command queue
    _queue = clCreateCommandQueue(_context, _device, 0, &status);
    return status == CL_SUCCESS;
}

void ProgramBagCL::InitProgramBag(SiftParam&param)
{
	GlobalUtil::StartTimer("Load Programs");
    LoadFixedShaders();
    LoadDynamicShaders(param);
    if(GlobalUtil::_UseSiftGPUEX) LoadDisplayShaders();
    GlobalUtil::StopTimer();
}


void ProgramBagCL::UnloadProgram()
{

}

void ProgramBagCL::FinishCL()
{
    clFinish(_queue);
}

void ProgramBagCL::LoadFixedShaders()
{

 
    s_gray = new ProgramCL( "gray", 
        "__kernel void gray(__read_only  image2d_t input, __write_only image2d_t output) {\n"
        "sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
        "int2 coord = (int2)(get_global_id(0),  get_global_id(1));\n"
        "float4 weight = (float4)(0.299, 0.587, 0.114, 0.0);\n"
        "float intensity = dot(weight, read_imagef(input,sampler, coord ));\n"
	    "float4 result= (float4)(intensity, intensity, intensity, 1.0);\n"
        "write_imagef(output, coord, result); }", _context, _device	);


	s_sampling = new ProgramCL("sampling",
        "__kernel void sampling(__read_only  image2d_t input, __write_only image2d_t output,\n"
        "                   int width, int height) {\n"
        "sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
        "int x = get_global_id(0), y =  get_global_id(1); \n"
        "if( x >= width || y >= height) return;\n"
        "int xa = x + x,   ya = y + y; \n"
        "int xb = xa + 1,  yb = ya + 1; \n"
        "float v1 = read_imagef(input, sampler, (int2) (xa, ya)).x; \n"
        "float v2 = read_imagef(input, sampler, (int2) (xb, ya)).x; \n"
        "float v3 = read_imagef(input, sampler, (int2) (xa, yb)).x; \n"
        "float v4 = read_imagef(input, sampler, (int2) (xb, yb)).x; \n"
        "float4 result = (float4) (v1, v2, v3, v4);"
        "write_imagef(output, (int2) (x, y), result); }"  , _context, _device);

	s_sampling_k = new ProgramCL("sampling_k",
        "__kernel void sampling_k(__read_only  image2d_t input, __write_only image2d_t output, "
        "                   int width, int height,\n"
        "                   int step,  int halfstep) {\n"
        "sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
        "int x = get_global_id(0), y =  get_global_id(1); \n"
        "if( x >= width || y >= height) return;\n"
        "int xa = x * step,   ya = y *step; \n"
        "int xb = xa + halfstep,  yb = ya + halfstep; \n"
        "float v1 = read_imagef(input, sampler, (int2) (xa, ya)).x; \n"
        "float v2 = read_imagef(input, sampler, (int2) (xb, ya)).x; \n"
        "float v3 = read_imagef(input, sampler, (int2) (xa, yb)).x; \n"
        "float v4 = read_imagef(input, sampler, (int2) (xb, yb)).x; \n"
        "float4 result = (float4) (v1, v2, v3, v4);"
        "write_imagef(output, (int2) (x, y), result); }"  , _context, _device);


	s_sampling_u = new ProgramCL("sampling_u",
        "__kernel void sampling_u(__read_only  image2d_t input, \n"
        "                   __write_only image2d_t output,\n"
        "                   int width, int height,\n"
        "                   float step, float halfstep) {\n"
        "sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;\n"
        "int x = get_global_id(0), y =  get_global_id(1); \n"
        "if( x >= width || y >= height) return;\n"
        "float xa = x * step,       ya = y *step; \n"
        "float xb = xa + halfstep,  yb = ya + halfstep; \n"
        "float v1 = read_imagef(input, sampler, (float2) (xa, ya)).x; \n"
        "float v2 = read_imagef(input, sampler, (float2) (xb, ya)).x; \n"
        "float v3 = read_imagef(input, sampler, (float2) (xa, yb)).x; \n"
        "float v4 = read_imagef(input, sampler, (float2) (xb, yb)).x; \n"
        "float4 result = (float4) (v1, v2, v3, v4);"
        "write_imagef(output, (int2) (x, y), result); }"  , _context, _device);


    s_zero_pass = new ProgramCL("zero_pass",
        "__kernel void zero_pass(__write_only image2d_t output){\n"
        "int2 coord = (int2)(get_global_id(0),  get_global_id(1));\n"
        "write_imagef(output, coord, (float4)(0.0));}", _context, _device);

    s_packup = new ProgramCL("packup",
        "__kernel void packup(__global float* input, __write_only image2d_t output,\n"
        "                   int twidth, int theight, int width){\n"
        "int2 coord = (int2)(get_global_id(0),  get_global_id(1));\n"
        "if(coord.x >= twidth || coord.y >= theight) return;\n"
        "int index0 = (coord.y + coord.y) * width; \n"
        "int index1 = index0 + coord.x;\n"
        "int x2 = min(width -1, coord.x); \n"
        "float v1 = input[index1 + coord.x], v2 = input[index1 + x2]; \n"
        "int index2 = index1 + width; \n"
        "float v3 = input[index2 + coord.x], v4 = input[index2 + x2]; \n "
        "write_imagef(output, coord, (float4) (v1, v2, v3, v4));}", _context, _device);

   s_dog_pass = new ProgramCL("dog_pass",
        "__kernel void dog_pass(__read_only image2d_t tex,  __read_only image2d_t texp,\n"
        "                   __write_only image2d_t dog, int width, int height) {\n"
        "sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |\n"
        "                    CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n" 
        "int2 coord = (int2)(get_global_id(0), get_global_id(1)); \n"
        "if( coord.x >= width || coord.y >= height) return;\n"
        "float4 cc = read_imagef(tex , sampler, coord); \n"
        "float4 cp = read_imagef(texp, sampler, coord);\n"
        "write_imagef(dog, coord, cc - cp); }\n", _context, _device); 

   s_grad_pass = new ProgramCL("grad_pass",
        "__kernel void grad_pass(__read_only image2d_t tex,  __read_only image2d_t texp,\n"
        "                   __write_only image2d_t dog, int width, int height,\n"
        "                    __write_only image2d_t grad, __write_only image2d_t rot) {\n"
        "sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |\n"
        "                    CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n" 
        "int x = get_global_id(0), y =  get_global_id(1); \n"
        "if( x >= width || y >= height) return;\n"
        "int2 coord = (int2) (x, y);\n"
        "float4 cc = read_imagef(tex , sampler, coord); \n"
        "float4 cp = read_imagef(texp, sampler, coord);\n"
        "float2 cl = read_imagef(tex, sampler, (int2)(x - 1, y)).yw;\n"
        "float2 cr = read_imagef(tex, sampler, (int2)(x + 1, y)).xz;\n"
	    "float2 cd = read_imagef(tex, sampler, (int2)(x, y - 1)).zw;\n"
        "float2 cu = read_imagef(tex, sampler, (int2)(x, y + 1)).xy;\n"
        "write_imagef(dog, coord, cc - cp); \n"
        "float4 dx = (float4)(cc.y - cl.x,  cr.x - cc.x, cc.w - cl.y, cr.y - cc.z);\n"
        "float4 dy = (float4)(cc.zw - cd.xy, cu.xy - cc.xy);\n"
        "write_imagef(grad, coord, 0.5 * sqrt(dx*dx + dy * dy));\n"
        "write_imagef(rot, coord, atan2(dy, dx + (float4) (FLT_MIN)));}\n", _context, _device); 

    s_grad_pass2 = new ProgramCL("grad_pass2",
        "#define BLOCK_DIMX 32\n"
        "#define BLOCK_DIMY 14\n"
        "#define BLOCK_SIZE (BLOCK_DIMX * BLOCK_DIMY)\n"
        "__kernel void grad_pass2(__read_only image2d_t tex,  __read_only image2d_t texp,\n"
        "                   __write_only image2d_t dog, int width, int height,\n"
        "                   __write_only image2d_t grd, __write_only image2d_t rot){\n"//,  __local float* block) {\n"
        "__local float block[BLOCK_SIZE * 4]; \n"
        "sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |\n"
        "                    CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n" 
        "int2 coord = (int2) (  get_global_id(0) - get_group_id(0) * 2 - 1, \n"
        "                       get_global_id(1) - get_group_id(1) * 2- 1); \n"
        "int idx =  mad24(get_local_id(1), BLOCK_DIMX, get_local_id(0));\n"
        "float4 cc = read_imagef(tex, sampler, coord);\n"
        "block[idx                 ] = cc.x;\n"
        "block[idx + BLOCK_SIZE    ] = cc.y;\n"
        "block[idx + BLOCK_SIZE * 2] = cc.z;\n"
        "block[idx + BLOCK_SIZE * 3] = cc.w;\n"
        "barrier(CLK_LOCAL_MEM_FENCE);\n"
        "if( get_local_id(0) == 0 || get_local_id(0) == BLOCK_DIMX - 1) return;\n"
        "if( get_local_id(1) == 0 || get_local_id(1) == BLOCK_DIMY - 1) return;\n"
        "if( coord.x >= width) return; \n"
        "if( coord.y >= height) return;\n"
        "float4 cp = read_imagef(texp, sampler, coord);\n"
        "float4 dx = (float4)(  cc.y - block[idx - 1 + BLOCK_SIZE], \n"
        "                       block[idx + 1] - cc.x, \n"
        "                       cc.w - block[idx - 1 + 3 * BLOCK_SIZE], \n"
        "                       block[idx + 1 + 2 * BLOCK_SIZE] - cc.z);\n"
        "float4 dy = (float4)(  cc.z - block[idx - BLOCK_DIMX + 2 * BLOCK_SIZE], \n"
        "                       cc.w - block[idx - BLOCK_DIMX + 3 * BLOCK_SIZE],"
        //"                       cc.zw - block[idx - BLOCK_DIMX].zw, \n"
        "                       block[idx + BLOCK_DIMX] - cc.x,\n "
        "                       block[idx + BLOCK_DIMX + BLOCK_SIZE] - cc.y);\n"
        //"                       block[idx + BLOCK_DIMX].xy - cc.xy);\n"
        "write_imagef(dog, coord, cc - cp); \n"
        "write_imagef(grd, coord, 0.5 * sqrt(dx*dx + dy * dy));\n"
        "write_imagef(rot, coord, atan2(dy, dx + (float4) (FLT_MIN)));}\n", _context, _device); 
}

void ProgramBagCL::LoadDynamicShaders(SiftParam& param)
{
    LoadKeypointShader();
    LoadGenListShader(param._dog_level_num, 0);
    CreateGaussianFilters(param);
}


void ProgramBagCL::SelectInitialSmoothingFilter(int octave_min, SiftParam&param)
{
    float sigma = param.GetInitialSmoothSigma(octave_min);
    if(sigma == 0) 
    {
       f_gaussian_skip0 = NULL; 
    }else
    {
	    for(unsigned int i = 0; i < f_gaussian_skip0_v.size(); i++)
	    {
		    if(f_gaussian_skip0_v[i]->_id == octave_min)
		    {
			    f_gaussian_skip0 = f_gaussian_skip0_v[i];
			    return ;
		    }
	    }
	    FilterCL * filter = CreateGaussianFilter(sigma); 
	    filter->_id = octave_min;
	    f_gaussian_skip0_v.push_back(filter);
	    f_gaussian_skip0 = filter; 
    }

}

void ProgramBagCL::CreateGaussianFilters(SiftParam&param)
{
	if(param._sigma_skip0>0.0f) 
	{
		f_gaussian_skip0 = CreateGaussianFilter(param._sigma_skip0);
		f_gaussian_skip0->_id = GlobalUtil::_octave_min_default; 
		f_gaussian_skip0_v.push_back(f_gaussian_skip0);
	}
	if(param._sigma_skip1>0.0f) 
	{
		f_gaussian_skip1 = CreateGaussianFilter(param._sigma_skip1);
	}

	f_gaussian_step = new FilterCL*[param._sigma_num];
	for(int i = 0; i< param._sigma_num; i++)
	{
		f_gaussian_step[i] =  CreateGaussianFilter(param._sigma[i]);
	}
    _gaussian_step_num = param._sigma_num;
}


FilterCL* ProgramBagCL::CreateGaussianFilter(float sigma)
{
	//pixel inside 3*sigma box
	int sz = int( ceil( GlobalUtil::_FilterWidthFactor * sigma -0.5) ) ;//
	int width = 2*sz + 1;

	//filter size truncation
	if(GlobalUtil::_MaxFilterWidth >0 && width > GlobalUtil::_MaxFilterWidth)
	{
		std::cout<<"Filter size truncated from "<<width<<" to "<<GlobalUtil::_MaxFilterWidth<<endl;
		sz = GlobalUtil::_MaxFilterWidth>>1;
		width = 2 * sz + 1;
	}

	int i;
	float * kernel = new float[width];
	float   rv = 1.0f/(sigma*sigma);
	float   v, ksum =0; 

	// pre-compute filter
	for( i = -sz ; i <= sz ; ++i) 
	{
		kernel[i+sz] =  v = exp(-0.5f * i * i *rv) ;
		ksum += v;
	}

	//normalize the kernel
	rv = 1.0f / ksum;
	for(i = 0; i< width ;i++) kernel[i]*=rv;

    FilterCL * filter = CreateFilter(kernel, width);
    delete [] kernel;
    if(GlobalUtil::_verbose && GlobalUtil::_timingL) std::cout<<"Filter: sigma = "<<sigma<<", size = "<<width<<"x"<<width<<endl;
    return filter;
}

FilterCL*  ProgramBagCL::CreateFilter(float kernel[], int width)
{
    FilterCL * filter = new FilterCL;
    filter->s_shader_h = CreateFilterH(kernel, width); 
    filter->s_shader_v = CreateFilterV(kernel, width);
    filter->_size = width; 
    filter->_id  = 0;
    return filter;
}

ProgramCL* ProgramBagCL::CreateFilterH(float kernel[], int width)
{
	int halfwidth  = width >>1;
	float * pf = kernel + halfwidth;
	int nhpixel = (halfwidth+1)>>1;	//how many neighbour pixels need to be looked up
	int npixel  = (nhpixel<<1)+1;//
	float weight[3];

    ////////////////////////////
	char buffer[10240];
	ostrstream out(buffer, 10240);
	out<<setprecision(8);


    //CL_DEVICE_IMAGE2D_MAX_WIDTH
	out<<
          "const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;"
          "__kernel void filter_h(__read_only  image2d_t input, \n"
          "          __write_only image2d_t output, int width_, int height_) {\n"
          "int x = get_global_id(0);\n"
          "int y = get_global_id(1); \n"
          "if( x > width_ || y > height_) return; \n"
          "float4 pc; int2 coord; \n"
          "float4 result = (float4)(0.0);\n";
    for(int i = 0 ; i < npixel ; i++)
	{
		out<<"coord = (int2)(x + ("<< (i - nhpixel) << "), y);\n";
		out<<"pc= read_imagef(input, sampler, coord);\n";
		if(GlobalUtil::_PreciseBorder)	
        out<<"if(coord.x < 0) pc = pc.xxzz; else if (coord.x > width_) pc = pc.yyww; \n";
		//for each sub-pixel j  in center, the weight of sub-pixel k 
		int xw = (i - nhpixel)*2;
		for(int j = 0; j < 3; j++)
		{
			int xwn = xw  + j  -1;
			weight[j] = xwn < -halfwidth || xwn > halfwidth? 0 : pf[xwn];
		}
		if(weight[1] == 0.0)
		{
			out<<"result += (float4)("<<weight[2]<<","<<weight[0]<<","<<weight[2]<<","<<weight[0]<<") * pc.yxwz;\n";
		}
		else
		{
			out<<"result += (float4)("<<weight[1]<<", "<<weight[0]<<", "<<weight[1]<<", "<<weight[0]<<") * pc.xxzz;\n";
			out<<"result += (float4)("<<weight[2]<<", "<<weight[1]<<", "<<weight[2]<<", "<<weight[1]<<") * pc.yyww;\n";
		}	
	}
    out << "write_imagef(output, (int2)(x, y), result); }\n" << '\0';
	return new ProgramCL("filter_h", buffer, _context, _device); 
}



ProgramCL* ProgramBagCL::CreateFilterV(float kernel[], int width)
{

	int halfwidth  = width >>1;
	float * pf = kernel + halfwidth;
	int nhpixel = (halfwidth+1)>>1;	//how many neighbour pixels need to be looked up
	int npixel  = (nhpixel<<1)+1;//
	float weight[3];

    ////////////////////////////
	char buffer[10240];
	ostrstream out(buffer, 10240);
	out<<setprecision(8);


    //CL_DEVICE_IMAGE2D_MAX_WIDTH
	out<< 
          "const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;"
          "__kernel void filter_v(__read_only  image2d_t input, \n"
          "          __write_only image2d_t output, int width_, int height_) {\n"
          "int x = get_global_id(0);\n"
          "int y = get_global_id(1); \n"
          "if( x > width_ || y >= height_) return; \n"
          "float4 pc; int2 coord; \n"
          "float4 result = (float4)(0.0);\n";
    for(int i = 0 ; i < npixel ; i++)
	{
		out<<"coord = (int2)(x, y + ("<< (i - nhpixel) << "));\n";
		out<<"pc= read_imagef(input, sampler, coord);\n";
		if(GlobalUtil::_PreciseBorder)	
        out<<"if(coord.y < 0) pc = pc.xyxy; else if (coord.y > height_) pc = pc.zwzw; \n";
		//for each sub-pixel j  in center, the weight of sub-pixel k 
		int xw = (i - nhpixel)*2;
		for(int j = 0; j < 3; j++)
		{
			int xwn = xw  + j  -1;
			weight[j] = xwn < -halfwidth || xwn > halfwidth? 0 : pf[xwn];
		}
		if(weight[1] == 0.0)
		{
			out<<"result += (float4)("<<weight[2]<<","<<weight[2]<<","<<weight[0]<<","<<weight[0]<<") * pc.zwxy;\n";
		}
		else
		{
			out<<"result += (float4)("<<weight[1]<<", "<<weight[1]<<", "<<weight[0]<<", "<<weight[0]<<") * pc.xyxy;\n";
			out<<"result += (float4)("<<weight[2]<<", "<<weight[2]<<", "<<weight[1]<<", "<<weight[1]<<") * pc.zwzw;\n";
		}	
	}
    out << "write_imagef(output, (int2)(x, y), result); }\n" << '\0';
	return new ProgramCL("filter_v", buffer, _context, _device); 

}

void ProgramBagCL::FilterImage(FilterCL* filter, CLTexImage *dst, CLTexImage *src, CLTexImage*tmp)
{
    cl_kernel kernelh = filter->s_shader_h->_kernel;
    cl_kernel kernelv = filter->s_shader_v->_kernel;
    //////////////////////////////////////////////////////////////////

    cl_int status, w = dst->GetImgWidth(), h = dst->GetImgHeight();
    cl_int w_ = w - 1, h_ = h - 1; 

    size_t dim0 = 16, dim1 = 16;
    size_t gsz[2] = {(w + dim0 - 1) / dim0 * dim0, (h + dim1 - 1) / dim1 * dim1}, lsz[2] = {dim0, dim1};

    clSetKernelArg(kernelh, 0, sizeof(cl_mem), &src->_clData);
    clSetKernelArg(kernelh, 1, sizeof(cl_mem), &tmp->_clData);
    clSetKernelArg(kernelh, 2, sizeof(cl_int), &w_);
    clSetKernelArg(kernelh, 3, sizeof(cl_int), &h_);
    status = clEnqueueNDRangeKernel(_queue, kernelh, 2, NULL, gsz, lsz, 0, NULL, NULL);
    CheckErrorCL(status, "ProgramBagCL::FilterImageH");
    if(status != CL_SUCCESS) return;

    clSetKernelArg(kernelv, 0, sizeof(cl_mem), &tmp->_clData);
    clSetKernelArg(kernelv, 1, sizeof(cl_mem), &dst->_clData);
    clSetKernelArg(kernelv, 2, sizeof(cl_int), &w_);
    clSetKernelArg(kernelv, 3, sizeof(cl_int), &h_);
    size_t gsz2[2] = {(w + dim1 - 1) / dim1 * dim1, (h + dim0 - 1) / dim0 * dim0}, lsz2[2] = {dim1, dim0};
    status = clEnqueueNDRangeKernel(_queue, kernelv, 2, NULL, gsz2, lsz2, 0, NULL, NULL);
    CheckErrorCL(status, "ProgramBagCL::FilterImageV");
    //clReleaseEvent(event);
}

void ProgramBagCL::SampleImageU(CLTexImage *dst, CLTexImage *src, int log_scale)
{
    cl_kernel  kernel= s_sampling_u->_kernel; 
    float scale  = 1.0f / (1 << log_scale);
    float offset = scale * 0.5f; 
    cl_int w = dst->GetImgWidth(), h = dst->GetImgHeight();
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &(src->_clData));
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &(dst->_clData));
    clSetKernelArg(kernel, 2, sizeof(cl_int), &(w));
    clSetKernelArg(kernel, 3, sizeof(cl_int), &(h));
    clSetKernelArg(kernel, 4, sizeof(cl_float), &(scale));
    clSetKernelArg(kernel, 5, sizeof(cl_float), &(offset));

    size_t dim0 = 16, dim1 = 16;
    //while( w * h / dim0 / dim1 < 8 && dim1 > 1) dim1 /= 2; 
    size_t gsz[2] = {(w + dim0 - 1) / dim0 * dim0, (h + dim1 - 1) / dim1 * dim1}, lsz[2] = {dim0, dim1};
    cl_int status = clEnqueueNDRangeKernel(_queue, kernel, 2, NULL, gsz, lsz, 0, NULL, NULL);
    CheckErrorCL(status, "ProgramBagCL::SampleImageU");
}

void ProgramBagCL::SampleImageD(CLTexImage *dst, CLTexImage *src, int log_scale)
{
    cl_kernel  kernel; 
    cl_int w = dst->GetImgWidth(), h = dst->GetImgHeight(); 
    if(log_scale == 1)
    {
        kernel = s_sampling->_kernel;
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &(src->_clData));
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &(dst->_clData));
        clSetKernelArg(kernel, 2, sizeof(cl_int), &(w));
        clSetKernelArg(kernel, 3, sizeof(cl_int), &(h));
    }else
    {
        cl_int fullstep = (1 << log_scale);
        cl_int halfstep = fullstep >> 1;
        kernel = s_sampling_k->_kernel;
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &(src->_clData));
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &(dst->_clData));
        clSetKernelArg(kernel, 2, sizeof(cl_int), &(w));
        clSetKernelArg(kernel, 3, sizeof(cl_int), &(h));
        clSetKernelArg(kernel, 4, sizeof(cl_int), &(fullstep));
        clSetKernelArg(kernel, 5, sizeof(cl_int), &(halfstep));
    }
    size_t dim0 = 128, dim1 = 1;
    //while( w * h / dim0 / dim1 < 8 && dim1 > 1) dim1 /= 2; 
    size_t gsz[2] = {(w + dim0 - 1) / dim0 * dim0, (h + dim1 - 1) / dim1 * dim1}, lsz[2] = {dim0, dim1};
    cl_int status = clEnqueueNDRangeKernel(_queue, kernel, 2, NULL, gsz, lsz, 0, NULL, NULL);
    CheckErrorCL(status, "ProgramBagCL::SampleImageD");
}

void ProgramBagCL::FilterInitialImage(CLTexImage* tex, CLTexImage* buf)
{
    if(f_gaussian_skip0) FilterImage(f_gaussian_skip0, tex, tex, buf);
}

void ProgramBagCL::FilterSampledImage(CLTexImage* tex, CLTexImage* buf)
{
    if(f_gaussian_skip1) FilterImage(f_gaussian_skip1, tex, tex, buf);
}

void ProgramBagCL::ComputeDOG(CLTexImage*tex, CLTexImage* texp, CLTexImage* dog, CLTexImage* grad, CLTexImage* rot)
{
    int margin = 0, use_gm2 = 1; 
    bool both_grad_dog = rot->_clData && grad->_clData;
    cl_int w = tex->GetImgWidth(), h = tex->GetImgHeight();
    cl_kernel  kernel ; size_t dim0, dim1; 
    if(!both_grad_dog)  {kernel = s_dog_pass->_kernel; dim0 = 16; dim1 = 12; }
    else if(use_gm2)    {kernel = s_grad_pass2->_kernel; dim0 = 32; dim1 = 14; margin = 2; }
    else                {kernel = s_grad_pass->_kernel; dim0 = 16; dim1 = 20; }
    size_t gsz[2] = {   (w + dim0 - 1 - margin) / (dim0 - margin) * dim0,
                        (h + dim1 - 1 - margin) / (dim1 - margin) * dim1};
    size_t lsz[2] = {dim0, dim1};
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &(tex->_clData));
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &(texp->_clData));
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &(dog->_clData));
    clSetKernelArg(kernel, 3, sizeof(cl_int), &(w));
    clSetKernelArg(kernel, 4, sizeof(cl_int), &(h)); 
    if(both_grad_dog)
    {
        clSetKernelArg(kernel, 5, sizeof(cl_mem), &(grad->_clData));
        clSetKernelArg(kernel, 6, sizeof(cl_mem), &(rot->_clData));
    }
    ///////////////////////////////////////////////////////
    cl_int status = clEnqueueNDRangeKernel(_queue, kernel, 2, NULL, gsz, lsz, 0, NULL, NULL);
    CheckErrorCL(status, "ProgramBagCL::ComputeDOG");
}


void ProgramBagCL::ComputeKEY(CLTexImage*dog, CLTexImage* key, float Tdog, float Tedge)
{
    cl_kernel  kernel = s_keypoint->_kernel; 
    cl_int w = key->GetImgWidth(), h = key->GetImgHeight();
	float threshold0 = Tdog* (GlobalUtil::_SubpixelLocalization?0.8f:1.0f);
	float threshold1 = Tdog;
	float threshold2 = (Tedge+1)*(Tedge+1)/Tedge;
	
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &(dog->_clData));
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &((dog + 1)->_clData));
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &((dog - 1)->_clData));
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &(key->_clData));
    clSetKernelArg(kernel, 4, sizeof(cl_float), &(threshold0));
    clSetKernelArg(kernel, 5, sizeof(cl_float), &(threshold1));
    clSetKernelArg(kernel, 6, sizeof(cl_float), &(threshold2));
    clSetKernelArg(kernel, 7, sizeof(cl_int), &(w));
    clSetKernelArg(kernel, 8, sizeof(cl_int), &(h)); 

    size_t dim0 = 8, dim1 = 8;
    //if( w * h / dim0 / dim1 < 16) dim1 /= 2; 
    size_t gsz[2] = {(w + dim0 - 1) / dim0 * dim0, (h + dim1 - 1) / dim1 * dim1}, lsz[2] = {dim0, dim1};
    cl_int status = clEnqueueNDRangeKernel(_queue, kernel, 2, NULL, gsz, lsz, 0, NULL, NULL);
    CheckErrorCL(status, "ProgramBagCL::ComputeKEY");
}

void ProgramBagCL::UnpackImage(CLTexImage*src, CLTexImage* dst)
{
    cl_kernel  kernel = s_unpack->_kernel; 
    cl_int w = dst->GetImgWidth(), h = dst->GetImgHeight();
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &(src->_clData));
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &(dst->_clData));
    clSetKernelArg(kernel, 2, sizeof(cl_int), &(w));
    clSetKernelArg(kernel, 3, sizeof(cl_int), &(h));
    const size_t dim0 = 16, dim1 = 16;
    size_t gsz[2] = {(w + dim0 - 1) / dim0 * dim0, (h + dim1 - 1) / dim1 * dim1}, lsz[2] = {dim0, dim1};
    cl_int status = clEnqueueNDRangeKernel(_queue, kernel, 2, NULL, gsz, lsz, 0, NULL, NULL);

    CheckErrorCL(status, "ProgramBagCL::UnpackImage");
    FinishCL();

}

void ProgramBagCL::UnpackImageDOG(CLTexImage*src, CLTexImage* dst)
{
    if(s_unpack_dog == NULL) return; 
    cl_kernel  kernel = s_unpack_dog->_kernel; 
    cl_int w = dst->GetImgWidth(), h = dst->GetImgHeight();
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &(src->_clData));
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &(dst->_clData));
    clSetKernelArg(kernel, 2, sizeof(cl_int), &(w));
    clSetKernelArg(kernel, 3, sizeof(cl_int), &(h));
    const size_t dim0 = 16, dim1 = 16;
    size_t gsz[2] = {(w + dim0 - 1) / dim0 * dim0, (h + dim1 - 1) / dim1 * dim1}, lsz[2] = {dim0, dim1};
    cl_int status = clEnqueueNDRangeKernel(_queue, kernel, 2, NULL, gsz, lsz, 0, NULL, NULL);

    CheckErrorCL(status, "ProgramBagCL::UnpackImage");
    FinishCL();
}

void ProgramBagCL::UnpackImageGRD(CLTexImage*src, CLTexImage* dst)
{
    if(s_unpack_grd == NULL) return; 
    cl_kernel  kernel = s_unpack_grd->_kernel; 
    cl_int w = dst->GetImgWidth(), h = dst->GetImgHeight();
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &(src->_clData));
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &(dst->_clData));
    clSetKernelArg(kernel, 2, sizeof(cl_int), &(w));
    clSetKernelArg(kernel, 3, sizeof(cl_int), &(h));
    const size_t dim0 = 16, dim1 = 16;
    size_t gsz[2] = {(w + dim0 - 1) / dim0 * dim0, (h + dim1 - 1) / dim1 * dim1}, lsz[2] = {dim0, dim1};
    cl_int status = clEnqueueNDRangeKernel(_queue, kernel, 2, NULL, gsz, lsz, 0, NULL, NULL);

    CheckErrorCL(status, "ProgramBagCL::UnpackImage");
    FinishCL();
}
void ProgramBagCL::UnpackImageKEY(CLTexImage*src, CLTexImage* dog, CLTexImage* dst)
{
    if(s_unpack_key == NULL) return;
    cl_kernel  kernel = s_unpack_key->_kernel; 
    cl_int w = dst->GetImgWidth(), h = dst->GetImgHeight();
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &(dog->_clData));
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &(src->_clData));
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &(dst->_clData));
    clSetKernelArg(kernel, 3, sizeof(cl_int), &(w));
    clSetKernelArg(kernel, 4, sizeof(cl_int), &(h));
    const size_t dim0 = 16, dim1 = 16;
    size_t gsz[2] = {(w + dim0 - 1) / dim0 * dim0, (h + dim1 - 1) / dim1 * dim1}, lsz[2] = {dim0, dim1};
    cl_int status = clEnqueueNDRangeKernel(_queue, kernel, 2, NULL, gsz, lsz, 0, NULL, NULL);

    CheckErrorCL(status, "ProgramBagCL::UnpackImageKEY");
    FinishCL();
}
void ProgramBagCL::LoadDescriptorShader()
{
	GlobalUtil::_DescriptorPPT = 16;
	LoadDescriptorShaderF2();
}

void ProgramBagCL::LoadDescriptorShaderF2()
{

}

void ProgramBagCL::LoadOrientationShader(void)
{

}

void ProgramBagCL::LoadGenListShader(int ndoglev,int nlev)
{

}

void ProgramBagCL::LoadKeypointShader()
{
    int i;    char buffer[20240];
	ostrstream out(buffer, 20240);
	streampos pos;

	//tex(X)(Y)
	//X: (CLR) (CENTER 0, LEFT -1, RIGHT +1)  
	//Y: (CDU) (CENTER 0, DOWN -1, UP    +1) 
	out<<
    "__kernel void keypoint(__read_only image2d_t tex, __read_only image2d_t texU,\n"
    "           __read_only image2d_t texD, __write_only image2d_t texK,\n"
    "          float THRESHOLD0, float THRESHOLD1, \n"
    "          float THRESHOLD2, int width, int height)\n"
	"{\n"
    "   sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | \n"
    "         CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;"
    "   int x = get_global_id(0), y = get_global_id(1);\n"
    "   if(x  >= width || y >= height) return; \n"
    "   int  xp = x - 1, xn = x + 1;\n"
    "   int  yp = y - 1, yn = y + 1;\n"
    "   int2 coord0 = (int2) (x, y); \n"
    "   int2 coord1 = (int2) (xp, y); \n"
    "   int2 coord2 = (int2) (xn, y); \n"
    "   int2 coord3 = (int2) (x, yp); \n"
    "   int2 coord4 = (int2) (x, yn); \n"
    "   int2 coord5 = (int2) (xp, yp); \n"
    "   int2 coord6 = (int2) (xp, yn); \n"
    "   int2 coord7 = (int2) (xn, yp); \n"
    "   int2 coord8 = (int2) (xn, yn); \n"
    "	float4 ccc = read_imagef(tex, sampler,coord0);\n"
	"	float4 clc = read_imagef(tex, sampler,coord1);\n"
	"	float4 crc = read_imagef(tex, sampler,coord2);\n"
	"	float4 ccd = read_imagef(tex, sampler,coord3);\n"
	"	float4 ccu = read_imagef(tex, sampler,coord4);\n"
	"	float4 cld = read_imagef(tex, sampler,coord5);\n"
	"	float4 clu = read_imagef(tex, sampler,coord6);\n"
	"	float4 crd = read_imagef(tex, sampler,coord7);\n"
	"	float4 cru = read_imagef(tex, sampler,coord8);\n"
    "	float4   cc = ccc;\n"
	"	float4  v1[4], v2[4];\n"
	"	v1[0] = (float4)(clc.y, ccc.y, ccd.z, ccc.z);\n"
	"	v1[1] = (float4)(ccc.x, crc.x, ccd.w, ccc.w);\n"
	"	v1[2] = (float4)(clc.w, ccc.w, ccc.x, ccu.x);\n"
	"	v1[3] = (float4)(ccc.z, crc.z, ccc.y, ccu.y);\n"
	"	v2[0] = (float4)(cld.w, clc.w, ccd.w, ccc.w);\n"
	"	v2[1] = (float4)(ccd.z, ccc.z, crd.z, crc.z);\n"
	"	v2[2] = (float4)(clc.y, clu.y, ccc.y, ccu.y);\n"
	"	v2[3] = (float4)(ccc.x, ccu.x, crc.x, cru.x);\n"
    "   float4 key4 = (float4)(0); \n";
	//test against 8 neighbours
	//use variable to identify type of extremum
	//1.0 for local maximum and -1.0 for minimum
    for(i = 0; i < 4; ++i)
	out<<
    "   if(cc.s"<<i<<" > THRESHOLD0){ \n"
    "           if(all(isgreater((float4)(cc.s"<<i<<"), max(v1["<<i<<"], v2["<<i<<"]))))key4.s"<<i<<" = 1.0;\n"
    "	}else if(cc.s"<<i<<" < -THRESHOLD0){ \n"
    "           if(all(isless((float4)(cc.s"<<i<<"), min(v1["<<i<<"], v2["<<i<<"]))))key4.s"<<i<<" = -1.0;\n"
    "   }";

	out<<
    "   if(x ==0) {key4.x =  key4.z= 0; }\n"
    "   else if(x + 1 == width) {key4.y =  key4.w = 0;}\n"
    "   if(y ==0) {key4.x =   key4.y = 0; }\n"
    "   else if(y + 1 == height) {key4.z = key4.w = 0;}\n"
    "   float4 ak = fabs(key4); \n"
    "   float keysum = ak.x + ak.y + ak.z + ak.w; \n"
	"	float4 result = (float4)(0.0);\n"
	"	if(keysum == 1.0) {\n"
	"	float fxx[4], fyy[4], fxy[4], fx[4], fy[4];\n";
	
    //do edge supression first.. 
	//vector v1 is < (-1, 0), (1, 0), (0,-1), (0, 1)>
	//vector v2 is < (-1,-1), (-1,1), (1,-1), (1, 1)>
    for(i = 0; i < 4; ++i)
	out <<
	"	if(key4.s"<<i<<" != 0)\n"
	"	{\n"
	"		float4 D2 = v1["<<i<<"].xyzw - cc.s"<<i<<";\n"
	"		float2 D4 = v2["<<i<<"].xw - v2["<<i<<"].yz;\n"
	"		float2 D5 = 0.5*(v1["<<i<<"].yw-v1["<<i<<"].xz); \n"
	"		fx["<<i<<"] = D5.x;	fy["<<i<<"] = D5.y ;\n"
	"		fxx["<<i<<"] = D2.x + D2.y;\n"
	"		fyy["<<i<<"] = D2.z + D2.w;\n"
	"		fxy["<<i<<"] = 0.25*(D4.x + D4.y);\n"
	"		float fxx_plus_fyy = fxx["<<i<<"] + fyy["<<i<<"];\n"
	"		float score_up = fxx_plus_fyy*fxx_plus_fyy; \n"
	"		float score_down = (fxx["<<i<<"]*fyy["<<i<<"] - fxy["<<i<<"]*fxy["<<i<<"]);\n"
	"		if( score_down <= 0 || score_up > THRESHOLD2 * score_down)keysum = 0;\n"
	"	}\n";

    out << 
	"	if(keysum == 1) {\n";
	////////////////////////////////////////////////
	//read 9 pixels of upper/lower level
	out<<
	"	float4  v4[4], v5[4], v6[4];\n"
	"	ccc = read_imagef(texU, sampler,coord0);\n"
	"	clc = read_imagef(texU, sampler,coord1);\n"
	"	crc = read_imagef(texU, sampler,coord2);\n"
	"	ccd = read_imagef(texU, sampler,coord3);\n"
	"	ccu = read_imagef(texU, sampler,coord4);\n"
	"	cld = read_imagef(texU, sampler,coord5);\n"
	"	clu = read_imagef(texU, sampler,coord6);\n"
	"	crd = read_imagef(texU, sampler,coord7);\n"
	"	cru = read_imagef(texU, sampler,coord8);\n"
    "	float4 cu = ccc;\n"
	"	v4[0] = (float4)(clc.y, ccc.y, ccd.z, ccc.z);\n"
	"	v4[1] = (float4)(ccc.x, crc.x, ccd.w, ccc.w);\n"
	"	v4[2] = (float4)(clc.w, ccc.w, ccc.x, ccu.x);\n"
	"	v4[3] = (float4)(ccc.z, crc.z, ccc.y, ccu.y);\n"
	"	v6[0] = (float4)(cld.w, clc.w, ccd.w, ccc.w);\n"
	"	v6[1] = (float4)(ccd.z, ccc.z, crd.z, crc.z);\n"
	"	v6[2] = (float4)(clc.y, clu.y, ccc.y, ccu.y);\n"
	"	v6[3] = (float4)(ccc.x, ccu.x, crc.x, cru.x);\n";

    for(i = 0; i < 4; ++i)
	out <<
	"	if(key4.s"<<i<<" == 1.0)\n"
	"	{\n"
	"		if(cc.s"<<i<<" < cu.s"<<i<<" || \n"
    "           any(isless((float4)(cc.s"<<i<<"), max(v4["<<i<<"], v6["<<i<<"]))))keysum = 0; \n"
	"	}else if(key4.s"<<i<<" == -1.0)\n"
	"	{\n"
	"		if(cc.s"<<i<<" > cu.s"<<i<<" || \n"
    "           any(isgreater((float4)(cc.s"<<i<<"), min(v4["<<i<<"], v6["<<i<<"]))) )keysum = 0; \n"
	"	}\n";

    out <<
	"	if(keysum == 1.0) { \n";
    out <<
	"	ccc = read_imagef(texD, sampler,coord0);\n"
	"	clc = read_imagef(texD, sampler,coord1);\n"
	"	crc = read_imagef(texD, sampler,coord2);\n"
	"	ccd = read_imagef(texD, sampler,coord3);\n"
	"	ccu = read_imagef(texD, sampler,coord4);\n"
	"	cld = read_imagef(texD, sampler,coord5);\n"
	"	clu = read_imagef(texD, sampler,coord6);\n"
	"	crd = read_imagef(texD, sampler,coord7);\n"
	"	cru = read_imagef(texD, sampler,coord8);\n"
    "	float4 cd = ccc;\n"
	"	v5[0] = (float4)(clc.y, ccc.y, ccd.z, ccc.z);\n"
	"	v5[1] = (float4)(ccc.x, crc.x, ccd.w, ccc.w);\n"
	"	v5[2] = (float4)(clc.w, ccc.w, ccc.x, ccu.x);\n"
	"	v5[3] = (float4)(ccc.z, crc.z, ccc.y, ccu.y);\n"
	"	v6[0] = (float4)(cld.w, clc.w, ccd.w, ccc.w);\n"
	"	v6[1] = (float4)(ccd.z, ccc.z, crd.z, crc.z);\n"
	"	v6[2] = (float4)(clc.y, clu.y, ccc.y, ccu.y);\n"
	"	v6[3] = (float4)(ccc.x, ccu.x, crc.x, cru.x);\n";
    for(i = 0; i < 4; ++i)
    out <<
	"	if(key4.s"<<i<<" == 1.0)\n"
	"	{\n"
	"		if(cc.s"<<i<<" < cd.s"<<i<<" ||\n"
    "           any(isless((float4)(cc.s"<<i<<"), max(v5["<<i<<"], v6["<<i<<"]))))keysum = 0; \n"
	"	}else if(key4.s"<<i<<" == -1.0)\n"
	"	{\n"
	"		if(cc.s"<<i<<" > cd.s"<<i<<" ||\n"
    "           any(isgreater((float4)(cc.s"<<i<<"), min(v5["<<i<<"], v6["<<i<<"]))))keysum = 0; \n"
	"	}\n";

    out << 
	"	if(keysum==1.0) {\n";
	//////////////////////////////////////////////////////////////////////
	if(GlobalUtil::_SubpixelLocalization)
    {
	    out <<
	    "	float4 offset = (float4)(0); \n";
        for(i = 1; i < 4; ++i)
        out <<
	    "	if(key4.s"<<i<<" != 0) \n"
	    "	{\n"
	    "		cu.s0 = cu.s"<<i<<";	cd.s0 = cd.s"<<i<<";	cc.s0 = cc.s"<<i<<";	\n"
	    "		v4[0] = v4["<<i<<"];	v5[0] = v5["<<i<<"];						\n"
	    "		fxy[0] = fxy["<<i<<"];	fxx[0] = fxx["<<i<<"];	fyy[0] = fyy["<<i<<"];	\n"
	    "		fx[0] = fx["<<i<<"];	fy[0] = fy["<<i<<"];						\n"
	    "	}\n";

        out <<	
	    "	float fs = 0.5*( cu.s0 - cd.s0 );				\n"
	    "	float fss = cu.s0 + cd.s0 - cc.s0 - cc.s0;\n"
	    "	float fxs = 0.25 * (v4[0].y + v5[0].x - v4[0].x - v5[0].y);\n"
	    "	float fys = 0.25 * (v4[0].w + v5[0].z - v4[0].z - v5[0].w);\n"
	    "	float4 A0, A1, A2 ;			\n"
	    "	A0 = (float4)(fxx[0], fxy[0], fxs, -fx[0]);	\n"
	    "	A1 = (float4)(fxy[0], fyy[0], fys, -fy[0]);	\n"
	    "	A2 = (float4)(fxs, fys, fss, -fs);	\n"
        "	float4 x3 = fabs((float4)(fxx[0], fxy[0], fxs, 0));		\n"
	    "	float maxa = max(max(x3.x, x3.y), x3.z);	\n"
	    "	if(maxa >= 1e-10 ) \n"
	    "	{												\n"
	    "		if(x3.y ==maxa )							\n"
	    "		{											\n"
	    "			float4 TEMP = A1; A1 = A0; A0 = TEMP;	\n"
	    "		}else if( x3.z == maxa )					\n"
	    "		{											\n"
	    "			float4 TEMP = A2; A2 = A0; A0 = TEMP;	\n"
	    "		}											\n"
	    "		A0 /= A0.x;									\n"
	    "		A1 -= A1.x * A0;							\n"
	    "		A2 -= A2.x * A0;							\n"
        "		float2 x2 = fabs((float2)(A1.y, A2.y));		\n"
	    "		if( x2.y > x2.x )							\n"
	    "		{											\n"
	    "			float4 TEMP = A2.yzwx;					\n"
	    "			A2.yzw = A1.yzw;						\n"
	    "			A1.yzw = TEMP.xyz;							\n"
	    "			x2.x = x2.y;							\n"
	    "		}											\n"
	    "		if(x2.x >= 1e-10) {								\n"
	    "			A1.yzw /= A1.y;								\n"
	    "			A2.yzw -= A2.y * A1.yzw;					\n"
	    "			if(fabs(A2.z) >= 1e-10) {\n"
	    "				offset.z = A2.w /A2.z;				    \n"
	    "				offset.y = A1.w - offset.z*A1.z;			    \n"
	    "				offset.x = A0.w - offset.z*A0.z - offset.y*A0.y;	\n"
        "				if(fabs(cc.s0 + 0.5*dot((float4)(fx[0], fy[0], fs, 0), offset ))<=THRESHOLD1\n"
        "                   || any( isgreater(fabs(offset), (float4)(1.0)))) key4 = (float4)(0.0);\n"
	    "			}\n"
	    "		}\n"
	    "	}\n"
	    <<"\n"
        "	float keyv = dot(key4, (float4)(1.0, 2.0, 3.0, 4.0));\n"
	    "	result = (float4)(keyv,  offset.xyz);\n"
	    "	}}}}\n"
        "   write_imagef(texK, coord0, result);\n "
	    "}\n"	<<'\0';
    }
	else 
    {
        out << "\n"
        "	float keyv = dot(key4, (float4)(1.0, 2.0, 3.0, 4.0));\n"
        "	result =  (float4)(keyv, 0, 0, 0);\n"
        "	}}}}\n"
        "   write_imagef(texK, coord0, result);\n "
        "}\n"	<<'\0';
    }

    s_keypoint = new ProgramCL("keypoint", buffer, _context, _device);
}

void ProgramBagCL::LoadDisplayShaders()
{
    //"uniform sampler2DRect tex; void main(){\n"
    //"vec4 pc = texture2DRect(tex, gl_TexCoord[0].xy);	bvec2 ff = lessThan(fract(gl_TexCoord[0].xy), vec2(0.5));\n"
    //"float v = ff.y?(ff.x? pc.r : pc.g):(ff.x?pc.b:pc.a); gl_FragColor = vec4(vec3(v), 1.0);}");
	s_unpack = new ProgramCL("main", 
    "__kernel void main(__read_only  image2d_t input, __write_only image2d_t output,\n"
    "                   int width, int height) {\n"
    "sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
    "int x = get_global_id(0), y =  get_global_id(1); \n"
    "if(x >= width || y >= height) return;\n"
    "int xx = x / 2, yy = y / 2; \n"
    "float4 vv = read_imagef(input, sampler, (int2) (xx, yy)); \n"
    "float v1 = (x & 1 ? vv.w : vv.z); \n"
    "float v2 = (x & 1 ? vv.y : vv.x);\n"
    "float v = y & 1 ? v1 : v2;\n"
    "float4 result = (float4) (v, v, v, 1);"
    "write_imagef(output, (int2) (x, y), result); }"  , _context, _device);

	s_unpack_dog = new ProgramCL("main", 
    "__kernel void main(__read_only  image2d_t input, __write_only image2d_t output,\n"
    "                   int width, int height) {\n"
    "sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
    "int x = get_global_id(0), y =  get_global_id(1); \n"
    "if(x >= width || y >= height) return;\n"
    "int xx = x / 2, yy = y / 2; \n"
    "float4 vv = read_imagef(input, sampler, (int2) (xx, yy)); \n"
    "float v1 = (x & 1 ? vv.w : vv.z); \n"
    "float v2 = (x & 1 ? vv.y : vv.x);\n"
    "float v0 = y & 1 ? v1 : v2;\n"
    "float v = 0.5 + 20.0 * v0;\n "
    "float4 result = (float4) (v, v, v, 1);"
    "write_imagef(output, (int2) (x, y), result); }"  , _context, _device);
	
    s_unpack_grd = new ProgramCL("main", 
    "__kernel void main(__read_only  image2d_t input, __write_only image2d_t output,\n"
    "                   int width, int height) {\n"
    "sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
    "int x = get_global_id(0), y =  get_global_id(1); \n"
    "if(x >= width || y >= height) return;\n"
    "int xx = x / 2, yy = y / 2; \n"
    "float4 vv = read_imagef(input, sampler, (int2) (xx, yy)); \n"
    "float v1 = (x & 1 ? vv.w : vv.z); \n"
    "float v2 = (x & 1 ? vv.y : vv.x);\n"
    "float v0 = y & 1 ? v1 : v2;\n"
    "float v = 5.0 * v0;\n "
    "float4 result = (float4) (v, v, v, 1);"
    "write_imagef(output, (int2) (x, y), result); }"  , _context, _device);

	s_unpack_key = new ProgramCL("main", 
    "__kernel void main(__read_only  image2d_t dog,\n"
    "                   __read_only image2d_t key,\n"
    "                   __write_only image2d_t output,\n"
    "                   int width, int height) {\n"
    "sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
    "int x = get_global_id(0), y =  get_global_id(1); \n"
    "if(x >= width || y >= height) return;\n"
    "int xx = x / 2, yy = y / 2; \n"
    "float4 kk = read_imagef(key, sampler, (int2) (xx, yy));\n"
    "int4 cc = isequal(fabs(kk.xxxx), (float4)(1.0, 2.0, 3.0, 4.0));\n"
    "int k1 = (x & 1 ? cc.w : cc.z); \n"
    "int k2 = (x & 1 ? cc.y : cc.x);\n"
    "int k0 = y & 1 ? k1 : k2;\n"
    "float4 result;\n"
    "if(k0 != 0){\n"
    "   //result = kk.x > 0 ? ((float4)(1.0, 0, 0, 1.0)) : ((float4) (0.0, 1.0, 0.0, 1.0)); \n"
    "   result = kk.x < 0 ? ((float4)(0, 1.0, 0, 1.0)) : ((float4) (1.0, 0.0,  0.0, 1.0)); \n"
    "}else{"
        "float4 vv = read_imagef(dog, sampler, (int2) (xx, yy));\n"
        "float v1 = (x & 1 ? vv.w : vv.z); \n"
        "float v2 = (x & 1 ? vv.y : vv.x);\n"
        "float v0 = y & 1 ? v1 : v2;\n"
        "float v = 0.5 + 20.0 * v0;\n "
        "result = (float4) (v, v, v, 1);"
    "}\n"
    "write_imagef(output, (int2) (x, y), result); }"  , _context, _device);
}


void ProgramBagCL::SetMarginCopyParam(int xmax, int ymax)
{

}

void ProgramBagCL::SetGradPassParam(int texP)
{

}

void ProgramBagCL::SetGenListEndParam(int ktex)
{

}

void ProgramBagCL::SetDogTexParam(int texU, int texD)
{

}

void ProgramBagCL::SetGenListInitParam(int w, int h)
{
	float bbox[4] = {(w -1.0f) * 0.5f +0.25f, (w-1.0f) * 0.5f - 0.25f,  (h - 1.0f) * 0.5f + 0.25f, (h-1.0f) * 0.5f - 0.25f};

}


void ProgramBagCL::SetGenListStartParam(float width, int tex0)
{

}



void ProgramBagCL::SetGenListStepParam(int tex, int tex0)
{

}

void ProgramBagCL::SetGenVBOParam(float width, float fwidth, float size)
{

}

void ProgramBagCL::SetSimpleOrientationInput(int oTex, float sigma, float sigma_step)
{

}


void ProgramBagCL::SetFeatureOrientationParam(int gtex, int width, int height, float sigma, int otex, float step)
{


}

void ProgramBagCL::SetFeatureDescirptorParam(int gtex, int otex, float dwidth, float fwidth,  float width, float height, float sigma)
{

}



const char* ProgramBagCL::GetErrorString(cl_int error)
{
    static const char* errorString[] = {
        "CL_SUCCESS",
        "CL_DEVICE_NOT_FOUND",
        "CL_DEVICE_NOT_AVAILABLE",
        "CL_COMPILER_NOT_AVAILABLE",
        "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        "CL_OUT_OF_RESOURCES",
        "CL_OUT_OF_HOST_MEMORY",
        "CL_PROFILING_INFO_NOT_AVAILABLE",
        "CL_MEM_COPY_OVERLAP",
        "CL_IMAGE_FORMAT_MISMATCH",
        "CL_IMAGE_FORMAT_NOT_SUPPORTED",
        "CL_BUILD_PROGRAM_FAILURE",
        "CL_MAP_FAILURE",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "CL_INVALID_VALUE",
        "CL_INVALID_DEVICE_TYPE",
        "CL_INVALID_PLATFORM",
        "CL_INVALID_DEVICE",
        "CL_INVALID_CONTEXT",
        "CL_INVALID_QUEUE_PROPERTIES",
        "CL_INVALID_COMMAND_QUEUE",
        "CL_INVALID_HOST_PTR",
        "CL_INVALID_MEM_OBJECT",
        "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
        "CL_INVALID_IMAGE_SIZE",
        "CL_INVALID_SAMPLER",
        "CL_INVALID_BINARY",
        "CL_INVALID_BUILD_OPTIONS",
        "CL_INVALID_PROGRAM",
        "CL_INVALID_PROGRAM_EXECUTABLE",
        "CL_INVALID_KERNEL_NAME",
        "CL_INVALID_KERNEL_DEFINITION",
        "CL_INVALID_KERNEL",
        "CL_INVALID_ARG_INDEX",
        "CL_INVALID_ARG_VALUE",
        "CL_INVALID_ARG_SIZE",
        "CL_INVALID_KERNEL_ARGS",
        "CL_INVALID_WORK_DIMENSION",
        "CL_INVALID_WORK_GROUP_SIZE",
        "CL_INVALID_WORK_ITEM_SIZE",
        "CL_INVALID_GLOBAL_OFFSET",
        "CL_INVALID_EVENT_WAIT_LIST",
        "CL_INVALID_EVENT",
        "CL_INVALID_OPERATION",
        "CL_INVALID_GL_OBJECT",
        "CL_INVALID_BUFFER_SIZE",
        "CL_INVALID_MIP_LEVEL",
        "CL_INVALID_GLOBAL_WORK_SIZE",
    };

    const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

    const int index = -error;

    return (index >= 0 && index < errorCount) ? errorString[index] : "";
}

bool ProgramBagCL::CheckErrorCL(cl_int error, const char* location)
{
    if(error == CL_SUCCESS) return true;
	const char *errstr = GetErrorString(error);
	if(errstr && errstr[0]) std::cerr << errstr; 
	else std::cerr  << "Error " << error;
	if(location) std::cerr  << " at " << location;		
	std::cerr  << "\n";
    exit(0);
    return false;

}


////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

void ProgramBagCLN::LoadFixedShaders()
{
   	s_sampling = new ProgramCL("sampling",
        "const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
        "__kernel void sampling(__read_only  image2d_t input, __write_only image2d_t output, "
        "                   int width, int height) {\n"
        "int2 coord = (int2)(get_global_id(0), get_global_id(1)); \n"
        "if( coord.x >= width || coord.y >= height) return;\n"
        "write_imagef(output, coord, read_imagef(input, sampler, coord << 1)); }"  , _context, _device); 
    
    s_sampling_k = new ProgramCL("sampling_k",
        "const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
        "__kernel void sampling_k(__read_only  image2d_t input, __write_only image2d_t output, "
        "                   int width, int height, int step) {\n"
        "int x = get_global_id(0), y =  get_global_id(1); \n"
        "if( x >= width || y >= height) return;\n"
        "int xa = x * step,   ya = y *step; \n"
        "float4 v1 = read_imagef(input, sampler, (int2) (xa, ya)); \n"
        "write_imagef(output, (int2) (x, y), v1); }"  , _context, _device);


	s_sampling_u = new ProgramCL("sampling_u",
        "const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;\n"
        "__kernel void sampling_u(__read_only  image2d_t input, \n"
        "                   __write_only image2d_t output,\n"
        "                   int width, int height, float step) {\n"
        "int x = get_global_id(0), y =  get_global_id(1); \n"
        "if( x >= width || y >= height) return;\n"
        "float xa = x * step,       ya = y *step; \n"
        "float v1 = read_imagef(input, sampler, (float2) (xa, ya)).x; \n"
        "write_imagef(output, (int2) (x, y), (float4)(v1)); }"  , _context, _device);

   s_dog_pass = new ProgramCL("dog_pass",
        "const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
        "__kernel void dog_pass(__read_only image2d_t tex,  __read_only image2d_t texp,\n"
        "                   __write_only image2d_t dog, int width, int height) {\n"
        "int2 coord = (int2)(get_global_id(0), get_global_id(1)); \n"
        "if( coord.x >= width || coord.y >= height) return;\n"
        "float cc = read_imagef(tex , sampler, coord).x; \n"
        "float cp = read_imagef(texp, sampler, coord).x;\n"
        "write_imagef(dog, coord, (float4)(cc - cp)); }\n", _context, _device); 

   s_grad_pass = new ProgramCL("grad_pass",
        "const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
        "__kernel void grad_pass(__read_only image2d_t tex,  __read_only image2d_t texp,\n"
        "                   __write_only image2d_t dog, int width, int height, \n"
        "                   __write_only image2d_t grad, __write_only image2d_t rot) {\n"
        "int x = get_global_id(0), y =  get_global_id(1); \n"
        "if( x >= width || y >= height) return;\n"
        "int2 coord = (int2) (x, y);\n"
        "float cl = read_imagef(tex, sampler, (int2)(x - 1, y)).x;\n"
        "float cc = read_imagef(tex , sampler, coord).x; \n"
        "float cr = read_imagef(tex, sampler, (int2)(x + 1, y)).x;\n"
        "float cp = read_imagef(texp, sampler, coord).x;\n"
        "write_imagef(dog, coord, (float4)(cc - cp)); \n"
	    "float cd = read_imagef(tex, sampler, (int2)(x, y - 1)).x;\n"
        "float cu = read_imagef(tex, sampler, (int2)(x, y + 1)).x;\n"
        "float dx = cr - cl, dy = cu - cd; \n"
	    "float gg = 0.5 * sqrt(dx*dx + dy * dy);\n"
        "write_imagef(grad, coord, (float4)(gg));\n"
        "float oo = atan2(dy, dx + FLT_MIN);\n"
        "write_imagef(rot, coord, (float4)(oo));}\n", _context, _device); 

   s_grad_pass2 = new ProgramCL("grad_pass2",
        "#define BLOCK_DIMX 32\n"
        "#define BLOCK_DIMY 14\n"
        "#define BLOCK_SIZE (BLOCK_DIMX * BLOCK_DIMY)\n"
        "__kernel void grad_pass2(__read_only image2d_t tex,  __read_only image2d_t texp,\n"
        "                   __write_only image2d_t dog, int width, int height,\n"
        "                   __write_only image2d_t grd, __write_only image2d_t rot){\n"//,  __local float* block) {\n"
        "__local float block[BLOCK_SIZE]; \n"
        "sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |\n"
        "                    CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n" 
        "int2 coord = (int2) (  get_global_id(0) - get_group_id(0) * 2 - 1, \n"
        "                       get_global_id(1) - get_group_id(1) * 2 - 1); \n"
        "int idx =  mad24(get_local_id(1), BLOCK_DIMX, get_local_id(0));\n"
        "float cc = read_imagef(tex, sampler, coord).x;\n"
        "block[idx] = cc;\n"
        "barrier(CLK_LOCAL_MEM_FENCE);\n"
        "if( get_local_id(0) == 0 || get_local_id(0) == BLOCK_DIMX - 1) return;\n"
        "if( get_local_id(1) == 0 || get_local_id(1) == BLOCK_DIMY - 1) return;\n"
        "if( coord.x >= width) return; \n"
        "if( coord.y >= height) return;\n"
        "float cp = read_imagef(texp, sampler, coord).x;\n"
        "float dx = block[idx + 1] - block[idx - 1];\n"
        "float dy = block[idx + BLOCK_DIMX ] - block[idx - BLOCK_DIMX];\n"
        "write_imagef(dog, coord, (float4)(cc - cp)); \n"
        "write_imagef(grd, coord, (float4)(0.5 * sqrt(dx*dx + dy * dy)));\n"
        "write_imagef(rot, coord, (float4)(atan2(dy, dx + FLT_MIN)));}\n", _context, _device); 
}

void ProgramBagCLN::LoadDisplayShaders()
{
	s_unpack = new ProgramCL("main", 
    "const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
    "__kernel void main(__read_only  image2d_t input, __write_only image2d_t output,\n"
    "                   int width, int height) {\n"
    "int x = get_global_id(0), y =  get_global_id(1); \n"
    "if(x >= width || y >= height) return;\n"
    "float v = read_imagef(input, sampler, (int2) (x, y)).x; \n"
    "float4 result = (float4) (v, v, v, 1);"
    "write_imagef(output, (int2) (x, y), result); }"  , _context, _device);

    s_unpack_grd = new ProgramCL("main", 
    "const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |\n"
    "           CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
    "__kernel void main(__read_only  image2d_t input, __write_only image2d_t output,\n"
    "                   int width, int height) {\n"
    "int x = get_global_id(0), y =  get_global_id(1); \n"
    "if(x >= width || y >= height) return;\n"
    "float v0 = read_imagef(input, sampler, (int2) (x, y)).x; \n"
    "float v = 5.0 * v0;  float4 result = (float4) (v, v, v, 1);"
    "write_imagef(output, (int2) (x, y), result); }"  , _context, _device);

	s_unpack_dog = new ProgramCL("main", 
    "const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
    "__kernel void main(__read_only  image2d_t input, __write_only image2d_t output,\n"
    "                   int width, int height) {\n"
    "int x = get_global_id(0), y =  get_global_id(1); \n"
    "if(x >= width || y >= height) return;\n"
    "float v0 = read_imagef(input, sampler, (int2) (x, y)).x; \n"
    "float v = 0.5 + 20.0 * v0; float4 result = (float4) (v, v, v, 1);"
    "write_imagef(output, (int2) (x, y), result); }"  , _context, _device);
}

ProgramCL* ProgramBagCLN::CreateFilterH(float kernel[], int width)
{
    ////////////////////////////
	char buffer[10240];
	ostrstream out(buffer, 10240);
    out <<  "#define KERNEL_WIDTH " << width << "\n"
        <<  "#define KERNEL_HALF_WIDTH " << (width / 2) << "\n" 
            "#define BLOCK_WIDTH 128\n"
            "#define BLOCK_HEIGHT 1\n"
            "#define CACHE_WIDTH (BLOCK_WIDTH + KERNEL_WIDTH - 1)\n"
            "#define CACHE_WIDTH_ALIGNED ((CACHE_WIDTH + 15) / 16 * 16)\n"
            "#define CACHE_COUNT (2 + (CACHE_WIDTH - 2) / BLOCK_WIDTH)\n"
            "const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
            "__kernel void filter_h(__read_only  image2d_t input, \n"
            "          __write_only image2d_t output, int width_, int height_, \n"
            "          __constant float* weight) {\n"
            "__local float data[CACHE_WIDTH]; \n"
            "int x = get_global_id(0), y = get_global_id(1);\n"
            "#pragma unroll\n"
	        "for(int j = 0; j < CACHE_COUNT; ++j)\n"
	        "{\n"
		    "    if(get_local_id(0) + j * BLOCK_WIDTH < CACHE_WIDTH)\n"
		    "    {\n"
			"        int fetch_index = min(x + j * BLOCK_WIDTH - KERNEL_HALF_WIDTH, width_);\n"
            "        data[get_local_id(0) + j * BLOCK_WIDTH] = read_imagef(input, sampler, (int2)(fetch_index, y)).x;\n"
		    "    }\n"
	        "}\n"
            "barrier(CLK_LOCAL_MEM_FENCE); \n"
            "if( x > width_ || y > height_) return; \n"
            "float result = 0; \n"
            "#pragma unroll\n"
            "for(int i = 0; i < KERNEL_WIDTH; ++i)\n"
            "{\n"
            "   result += data[get_local_id(0) + i] * weight[i];\n"
            "}\n"
         << "write_imagef(output, (int2)(x, y), (float4)(result)); }\n" << '\0';
	return new ProgramCL("filter_h", buffer, _context, _device); 
}



ProgramCL* ProgramBagCLN::CreateFilterV(float kernel[], int width)
{
    ////////////////////////////
	char buffer[10240];
	ostrstream out(buffer, 10240);
    out <<  "#define KERNEL_WIDTH " << width << "\n"
        <<  "#define KERNEL_HALF_WIDTH " << (width / 2) << "\n" 
            "#define BLOCK_WIDTH 128\n"
            "#define CACHE_WIDTH (BLOCK_WIDTH + KERNEL_WIDTH - 1)\n"
            "#define CACHE_WIDTH_ALIGNED ((CACHE_WIDTH + 15) / 16 * 16)\n"
            "#define CACHE_COUNT (2 + (CACHE_WIDTH - 2) / BLOCK_WIDTH)\n"
            "const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | \n"
            "           CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
            "__kernel void filter_v(__read_only  image2d_t input, \n"
            "          __write_only image2d_t output, int width_, int height_, \n"
            "          __constant float* weight) {\n"
            "__local float data[CACHE_WIDTH]; \n"
            "int x = get_global_id(0), y = get_global_id(1);\n"
            "#pragma unroll\n"
	        "for(int j = 0; j < CACHE_COUNT; ++j)\n"
	        "{\n"
		    "    if(get_local_id(1) + j * BLOCK_WIDTH  < CACHE_WIDTH)\n"
		    "    {\n"
			"        int fetch_index = min(y + j * BLOCK_WIDTH - KERNEL_HALF_WIDTH, height_);\n"
            "        data[get_local_id(1) + j * BLOCK_WIDTH ] = read_imagef(input, sampler, (int2)(x, fetch_index)).x;\n"
		    "    }\n"
	        "}\n"
            "barrier(CLK_LOCAL_MEM_FENCE); \n"
            "if( x > width_ || y > height_) return; \n"
            "float result = 0; \n"
            "#pragma unroll\n"
            "for(int i = 0; i < KERNEL_WIDTH; ++i)\n"
            "{\n"
            "   result += data[get_local_id(1) + i] * weight[i];\n"
            "}\n"
         << "write_imagef(output, (int2)(x, y), (float4)(result)); }\n" << '\0';
	
	return new ProgramCL("filter_v", buffer, _context, _device); 
}

FilterCL*  ProgramBagCLN::CreateFilter(float kernel[], int width)
{
    FilterCL * filter = new FilterCL;
    filter->s_shader_h = CreateFilterH(kernel, width); 
    filter->s_shader_v = CreateFilterV(kernel, width);
    filter->_weight = new CLTexImage(_context, _queue);
    filter->_weight->InitBufferTex(width, 1, 1);
    filter->_weight->CopyFromHost(kernel);
    filter->_size = width; 
    return filter;
}


void ProgramBagCLN::FilterImage(FilterCL* filter, CLTexImage *dst, CLTexImage *src, CLTexImage*tmp)
{
    cl_kernel kernelh = filter->s_shader_h->_kernel;
    cl_kernel kernelv = filter->s_shader_v->_kernel;
    //////////////////////////////////////////////////////////////////

    cl_int status, w = dst->GetImgWidth(), h = dst->GetImgHeight();
    cl_mem weight = (cl_mem) filter->_weight->_clData;
    cl_int w_ = w - 1, h_ = h - 1; 


    clSetKernelArg(kernelh, 0, sizeof(cl_mem), &src->_clData);
    clSetKernelArg(kernelh, 1, sizeof(cl_mem), &tmp->_clData);
    clSetKernelArg(kernelh, 2, sizeof(cl_int), &w_);
    clSetKernelArg(kernelh, 3, sizeof(cl_int), &h_);
    clSetKernelArg(kernelh, 4, sizeof(cl_mem), &weight);

    size_t dim00 = 128, dim01 = 1;
    size_t gsz1[2] = {(w + dim00 - 1) / dim00 * dim00, (h + dim01 - 1) / dim01 * dim01}, lsz1[2] = {dim00, dim01};
    status = clEnqueueNDRangeKernel(_queue, kernelh, 2, NULL, gsz1, lsz1, 0, NULL, NULL);
    CheckErrorCL(status, "ProgramBagCLN::FilterImageH");
    if(status != CL_SUCCESS) return;


    clSetKernelArg(kernelv, 0, sizeof(cl_mem), &tmp->_clData);
    clSetKernelArg(kernelv, 1, sizeof(cl_mem), &dst->_clData);
    clSetKernelArg(kernelv, 2, sizeof(cl_int), &w_);
    clSetKernelArg(kernelv, 3, sizeof(cl_int), &h_);
    clSetKernelArg(kernelv, 4, sizeof(cl_mem), &weight); 

    size_t dim10 = 1, dim11 = 128;
    size_t gsz2[2] = {(w + dim10 - 1) / dim10 * dim10, (h + dim11 - 1) / dim11 * dim11}, lsz2[2] = {dim10, dim11};
    status = clEnqueueNDRangeKernel(_queue, kernelv, 2, NULL, gsz2, lsz2, 0, NULL, NULL);
    CheckErrorCL(status, "ProgramBagCLN::FilterImageV");
    //clReleaseEvent(event);
}

void ProgramBagCLN::SampleImageD(CLTexImage *dst, CLTexImage *src, int log_scale)
{
    cl_kernel  kernel; 
    cl_int w = dst->GetImgWidth(), h = dst->GetImgHeight(); 

    cl_int fullstep = (1 << log_scale);
    kernel = log_scale == 1? s_sampling->_kernel : s_sampling_k->_kernel;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &(src->_clData));
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &(dst->_clData));
    clSetKernelArg(kernel, 2, sizeof(cl_int), &(w));
    clSetKernelArg(kernel, 3, sizeof(cl_int), &(h));
    if(log_scale > 1) clSetKernelArg(kernel, 4, sizeof(cl_int), &(fullstep));

    size_t dim0 = 128, dim1 = 1;
    //while( w * h / dim0 / dim1 < 8 && dim1 > 1) dim1 /= 2; 
    size_t gsz[2] = {(w + dim0 - 1) / dim0 * dim0, (h + dim1 - 1) / dim1 * dim1}, lsz[2] = {dim0, dim1};
    cl_int status = clEnqueueNDRangeKernel(_queue, kernel, 2, NULL, gsz, lsz, 0, NULL, NULL);
    CheckErrorCL(status, "ProgramBagCLN::SampleImageD");
}


#endif

