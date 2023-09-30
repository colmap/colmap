////////////////////////////////////////////////////////////////////////////
//	File:		ProgramCL.h
//	Author:		Changchang Wu
//	Description :	interface for the ProgramCL classes.
//		ProgramCL:		Cg programs
//		ShaderBagCG:	All Cg shaders for Sift in a bag
//		FilterCL:		Cg Gaussian Filters
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

#ifndef _PROGRAM_CL_H
#define _PROGRAM_CL_H

#include "ProgramGPU.h"

class ProgramCL: public ProgramGPU
{
	cl_program		  _program; 
	cl_kernel		  _kernel;
	int				  _valid;
public:
	int IsValidProgram(){return _program && _valid;}
	ProgramCL(const char* name, const char * code, cl_context contex, cl_device_id device);
	ProgramCL();
    void PrintBuildLog(cl_device_id device, int all);
	virtual ~ProgramCL();
    virtual int UseProgram(){return 1;}
    virtual void * GetProgramID() {return _kernel;}
    friend class ProgramBagCL;
    friend class ProgramBagCLN;
};

class  CLTexImage;
class FilterCL
{
public:
	ProgramCL*  s_shader_h;
	ProgramCL*  s_shader_v;
	int			 _size;
	int			 _id; 
    CLTexImage * _weight;
public:
    FilterCL() : s_shader_h(NULL), s_shader_v(NULL), _size(0), _id(0), _weight(NULL) {}
    ~FilterCL() {if(s_shader_h) delete s_shader_h; if(s_shader_v) delete s_shader_v; if(_weight) delete _weight; }
};

class  SiftParam;

class ProgramBagCL
{
protected:
    cl_platform_id      _platform;
    cl_device_id        _device;
    cl_context          _context;
    cl_command_queue    _queue;
protected:
    ProgramCL  * s_gray;
	ProgramCL  * s_sampling;
    ProgramCL  * s_sampling_k;
    ProgramCL  * s_sampling_u;
	ProgramCL  * s_zero_pass;
    ProgramCL  * s_packup;
    ProgramCL  * s_unpack;
    ProgramCL  * s_unpack_dog;
    ProgramCL  * s_unpack_grd;
    ProgramCL  * s_unpack_key;
    ProgramCL  * s_dog_pass;
    ProgramCL  * s_grad_pass;
    ProgramCL  * s_grad_pass2;
    ProgramCL  * s_gray_pack;
    ProgramCL  * s_keypoint;
public:
	FilterCL  *         f_gaussian_skip0;
	vector<FilterCL*>   f_gaussian_skip0_v;
	FilterCL  *         f_gaussian_skip1;
	FilterCL  **        f_gaussian_step;
    int                     _gaussian_step_num;
public:
	ProgramBagCL();
    bool InitializeContext();
	virtual ~ProgramBagCL();
    void FinishCL();
    cl_context          GetContextCL() {return _context;}
    cl_command_queue    GetCommandQueue() {return _queue;}
    static const char* GetErrorString(cl_int error);
    static bool  CheckErrorCL(cl_int error, const char* location = NULL);
public:
    FilterCL * CreateGaussianFilter(float sigma);
    void CreateGaussianFilters(SiftParam&param);
    void SelectInitialSmoothingFilter(int octave_min, SiftParam&param);
    void FilterInitialImage(CLTexImage* tex, CLTexImage* buf);
    void FilterSampledImage(CLTexImage* tex, CLTexImage* buf);
    void UnpackImage(CLTexImage*src, CLTexImage* dst); 
    void UnpackImageDOG(CLTexImage*src, CLTexImage* dst); 
    void UnpackImageGRD(CLTexImage*src, CLTexImage* dst); 
    void UnpackImageKEY(CLTexImage*src, CLTexImage* dog, CLTexImage* dst); 
    void ComputeDOG(CLTexImage*tex, CLTexImage* texp, CLTexImage* dog, CLTexImage* grad, CLTexImage* rot);
    void ComputeKEY(CLTexImage*dog, CLTexImage* key, float Tdog, float Tedge);
public:
	virtual void SampleImageU(CLTexImage *dst, CLTexImage *src, int log_scale);
	virtual void SampleImageD(CLTexImage *dst, CLTexImage *src, int log_scale = 1); 
    virtual void FilterImage(FilterCL* filter, CLTexImage *dst, CLTexImage *src, CLTexImage*tmp);
    virtual ProgramCL* CreateFilterH(float kernel[], int width);
    virtual ProgramCL* CreateFilterV(float kernel[], int width);
    virtual FilterCL*  CreateFilter(float kernel[], int width);
public:
    virtual void InitProgramBag(SiftParam&param);
	virtual void LoadDescriptorShader();
	virtual void LoadDescriptorShaderF2();
	virtual void LoadOrientationShader();
	virtual void LoadGenListShader(int ndoglev, int nlev);
	virtual void UnloadProgram() ;
	virtual void LoadKeypointShader();
	virtual void LoadFixedShaders();
	virtual void LoadDisplayShaders();
    virtual void LoadDynamicShaders(SiftParam& param);
public:
	//parameters
	virtual void SetGradPassParam(int texP);
	virtual void SetGenListEndParam(int ktex);
	virtual void SetGenListStartParam(float width, int tex0);
	virtual void SetGenListInitParam(int w, int h);
	virtual void SetMarginCopyParam(int xmax, int ymax);
	virtual void SetDogTexParam(int texU, int texD);
	virtual void SetGenListStepParam(int tex, int tex0);
	virtual void SetGenVBOParam( float width, float fwidth, float size);
	virtual void SetFeatureDescirptorParam(int gtex, int otex, float dwidth, float fwidth, float width, float height, float sigma);
	virtual void SetFeatureOrientationParam(int gtex, int width, int height, float sigma, int stex, float step);
	virtual void SetSimpleOrientationInput(int oTex, float sigma, float sigma_step);

};

class CLTexImage ;
class ProgramBagCLN: public ProgramBagCL
{
public:
	virtual void SampleImageD(CLTexImage *dst, CLTexImage *src, int log_scale = 1); 
    virtual FilterCL*  CreateFilter(float kernel[], int width);
    virtual ProgramCL* CreateFilterH(float kernel[], int width);
    virtual ProgramCL* CreateFilterV(float kernel[], int width);
    virtual void FilterImage(FilterCL* filter, CLTexImage *dst, CLTexImage *src, CLTexImage*tmp);
    virtual void LoadFixedShaders();
	virtual void LoadDisplayShaders();
};
#endif
#endif

