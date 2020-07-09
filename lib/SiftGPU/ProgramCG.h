////////////////////////////////////////////////////////////////////////////
//	File:		ProgramCG.h
//	Author:		Changchang Wu
//	Description :	interface for the ProgramCG classes.
//		ProgramCG:		Cg programs
//		ShaderBagCG:	All Cg shaders for Sift in a bag
//		FilterGLCG:		Cg Gaussian Filters
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


#if defined(CG_SIFTGPU_ENABLED) 

#ifndef _PROGRAM_CG_H
#define _PROGRAM_CG_H

#include "ProgramGPU.h"
class FragmentProgram;
#include "Cg/cgGL.h"

class ProgramCG:public ProgramGPU  
{
	CGprogram		  _programID; 
	CGprofile		  _profile;
	int				  _valid;
public:
	static CGcontext _Context;
	static CGprofile _FProfile;	
public:
	operator CGprogram (){return _programID;}
	CGprogram GetProgramID(){return _programID;}
	int UseProgram();
	int IsValidProgram(){return _programID && _valid;}
	static void  ErrorCallback();
	static void InitContext();
	static void DestroyContext();
	ProgramCG(const char * code, const char** cg_compile_args= NULL, CGprofile profile = ProgramCG::_FProfile);
	ProgramCG();
	virtual ~ProgramCG();

};

class ShaderBagCG:public ShaderBag
{
	CGparameter _param_dog_texu;
	CGparameter	_param_dog_texd;
	CGparameter _param_genlist_start_tex0;
	CGparameter _param_ftex_width;
	CGparameter _param_genlist_step_tex;
	CGparameter _param_genlist_step_tex0;
	CGparameter _param_genvbo_size;
	CGparameter _param_orientation_gtex;
	CGparameter _param_orientation_stex;
	CGparameter _param_orientation_size;
	CGparameter _param_descriptor_gtex;
	CGparameter _param_descriptor_size;
	CGparameter _param_descriptor_dsize;
	CGparameter _param_margin_copy_truncate;
	CGparameter _param_genlist_init_bbox;
public:
	virtual void LoadDescriptorShader();
	void	LoadDescriptorShaderF2();
	static void  WriteOrientationCodeToStream(ostream& out);
	virtual void SetGenListInitParam(int w, int h);
	virtual void SetMarginCopyParam(int xmax, int ymax);
	virtual void SetFeatureOrientationParam(int gtex, int width, int height, float sigma, int stex = 0, float step = 1.0f);
	virtual void SetFeatureDescirptorParam(int gtex, int otex, float dwidth, float fwidth, float width, float height, float sigma);
	virtual void SetSimpleOrientationInput(int oTex, float sigma, float sigma_step);
	void LoadOrientationShader();
	virtual void SetGenListStartParam(float width, int tex0);
	static ProgramCG* LoadGenListStepShader(int start, int step);
	static ProgramCG* LoadGenListStepShaderV2(int start, int step);
	void LoadGenListShader(int ndoglev,  int nlev);
	virtual void UnloadProgram();
	virtual void SetDogTexParam(int texU, int texD);
	virtual void SetGenListStepParam(int tex, int tex0);
	virtual void SetGenVBOParam( float width, float fwidth,  float size);
	virtual void LoadFixedShaders();
	virtual void LoadDisplayShaders();
	virtual void LoadKeypointShader(float threshold, float edgeThreshold);
	virtual int  LoadKeypointShaderMR(float threshold, float edgeThreshold);
	ShaderBagCG();
	virtual ~ShaderBagCG(){}
};


class FilterGLCG : public FilterProgram
{
private:
	ProgramGPU* CreateFilterH(float kernel[], float offset[], int width);
	ProgramGPU* CreateFilterV(float kernel[], float offset[], int height);
	//packed version 
	ProgramGPU* CreateFilterHPK(float kernel[], float offset[], int width);
	ProgramGPU* CreateFilterVPK(float kernel[], float offset[], int height);
};

class ShaderBagPKCG:public ShaderBag
{
private:
	CGparameter _param_dog_texu;
	CGparameter	_param_dog_texd;
	CGparameter _param_margin_copy_truncate;
	CGparameter _param_grad_pass_texp;
	CGparameter _param_genlist_init_bbox;
	CGparameter _param_genlist_start_tex0;
	CGparameter _param_ftex_width;
	CGparameter _param_genlist_step_tex;
	CGparameter _param_genlist_step_tex0;
	CGparameter _param_genlist_end_ktex;
	CGparameter _param_genvbo_size;
	CGparameter _param_orientation_gtex;
	CGparameter _param_orientation_otex;
	CGparameter _param_orientation_size;
	CGparameter	_param_descriptor_gtex; 
	CGparameter	_param_descriptor_otex;
	CGparameter	_param_descriptor_size; 
	CGparameter	_param_descriptor_dsize;

public:
	ShaderBagPKCG();
	virtual ~ShaderBagPKCG(){}
	virtual void LoadDescriptorShader();
	virtual void LoadDescriptorShaderF2();
	virtual void LoadOrientationShader();
	virtual void LoadGenListShader(int ndoglev, int nlev);
	virtual void LoadGenListShaderV2(int ndoglev, int nlev);
	virtual void UnloadProgram() ;
	virtual void LoadKeypointShader(float threshold, float edgeTrheshold);
	virtual void LoadFixedShaders();
	virtual void LoadDisplayShaders();
	virtual void SetGradPassParam(int texP);
	virtual void SetGenListEndParam(int ktex);
public:
	//parameters
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
#endif
#endif

