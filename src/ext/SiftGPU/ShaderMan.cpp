////////////////////////////////////////////////////////////////////////////
//	File:		ShaderMan.cpp
//	Author:		Changchang Wu
//	Description :	implementation of the ShaderMan class.
//				A Shader Manager that calls different implementation of shaders
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


#include "GL/glew.h"
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <math.h>
using std::vector;
using std::ostream;
using std::endl;


#include "ProgramGLSL.h"
#include "GlobalUtil.h"
#include "GLTexImage.h"
#include "SiftGPU.h"
#include "ShaderMan.h"

///
ShaderBag   * ShaderMan::s_bag = NULL;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

void ShaderMan::InitShaderMan(SiftParam&param)
{
	if(s_bag) return;

	if(GlobalUtil::_usePackedTex )	s_bag = new ShaderBagPKSL;
	else		                	s_bag =new ShaderBagGLSL;	

    GlobalUtil::StartTimer("Load Programs");
	s_bag->LoadFixedShaders();
    s_bag->LoadDynamicShaders(param);
	if(GlobalUtil::_UseSiftGPUEX) 	s_bag->LoadDisplayShaders();
    GlobalUtil::StopTimer();

	GlobalUtil::CheckErrorsGL("InitShaderMan");
}


void ShaderMan::DestroyShaders()
{
	if(s_bag) delete s_bag;
    s_bag   =   NULL; 
}

void ShaderMan::UnloadProgram()
{
	if(s_bag) s_bag->UnloadProgram();
}

void ShaderMan::FilterImage(FilterProgram* filter, GLTexImage *dst, GLTexImage *src, GLTexImage*tmp)
{
    if(filter == NULL) return; 

    //////////////////////////////
    src->FillMargin(filter->_size, 0);

	//output parameter
	if(tmp) tmp->AttachToFBO(0);
	else dst->AttachToFBO(0);


	//input parameter
	src->BindTex();
	dst->FitTexViewPort();

	//horizontal filter
	filter->s_shader_h->UseProgram();
	dst->DrawQuad();

	//parameters
	if(tmp)
	{
		// fill margin for out-of-boundary lookup
		tmp->DetachFBO(0);
		tmp->AttachToFBO(0);
		tmp->FillMargin(0, filter->_size);
		tmp->DetachFBO(0);
		dst->AttachToFBO(0);
		tmp->BindTex();
	}
	else
	{
		glFinish();
		// fill margin for out-of-boundary lookup
		dst->FillMargin(0, filter->_size);
		dst->BindTex();
	}

	//vertical filter
	filter->s_shader_v->UseProgram();
	dst->DrawQuad();


	//clean up
	dst->UnbindTex();
	dst->DetachFBO(0);

	//
	ShaderMan::UnloadProgram();
}


void ShaderMan::FilterInitialImage(GLTexImage* tex, GLTexImage* buf)
{
    if(s_bag->f_gaussian_skip0) FilterImage(s_bag->f_gaussian_skip0, tex, tex, buf);
}

void ShaderMan::FilterSampledImage(GLTexImage* tex, GLTexImage* buf)
{
    if(s_bag->f_gaussian_skip1) FilterImage(s_bag->f_gaussian_skip1, tex, tex, buf);
}

void ShaderMan::TextureCopy(GLTexImage*dst, GLTexImage*src)
{

	dst->AttachToFBO(0);

	src->BindTex();

	dst->FitTexViewPort();

	dst->DrawQuad();

	dst->UnbindTex();
//	ShaderMan::UnloadProgram();
	dst->DetachFBO(0);
	return;
}
void ShaderMan::TextureDownSample(GLTexImage *dst, GLTexImage *src, int scale)
{
	//output parameter
	
	dst->AttachToFBO(0);

	//input parameter
	src->BindTex();

	//
	dst->FitTexViewPort();

	s_bag->s_sampling->UseProgram();

	dst->DrawQuadDS(scale);
	src->UnbindTex();

	UnloadProgram();

	dst->DetachFBO(0); 
}

void ShaderMan::TextureUpSample(GLTexImage *dst, GLTexImage *src, int scale)
{

	//output parameter
	dst->AttachToFBO(0);
	//input parameter
	src->BindTex();

	dst->FitTexViewPort();

	GlobalUtil::SetTextureParameterUS();

	if(GlobalUtil::_usePackedTex)
	{
		s_bag->s_sampling->UseProgram();
	}

	dst->DrawQuadUS(scale);
	src->UnbindTex();

	UnloadProgram();

	dst->DetachFBO(0);

	GlobalUtil::SetTextureParameter();
}



void ShaderMan::UseShaderDisplayGaussian()
{
	if(s_bag && s_bag->s_display_gaussian) s_bag->s_display_gaussian->UseProgram();
}

void ShaderMan::UseShaderDisplayDOG()
{
	if(s_bag && s_bag->s_display_dog) s_bag->s_display_dog->UseProgram();
}



void ShaderMan::UseShaderRGB2Gray()
{
	if(s_bag && s_bag->s_gray)s_bag->s_gray->UseProgram();
}


void ShaderMan::UseShaderDisplayGrad()
{
	if(s_bag && s_bag->s_display_grad) s_bag->s_display_grad->UseProgram();
}


void ShaderMan::UseShaderDisplayKeypoints()
{
	if(s_bag && s_bag->s_display_keys) s_bag->s_display_keys->UseProgram();
}





void ShaderMan::UseShaderGradientPass(int texP)
{
	s_bag->s_grad_pass->UseProgram();
	s_bag->SetGradPassParam(texP);	
}


void ShaderMan::UseShaderKeypoint(int texU, int texD)
{
	s_bag->s_keypoint->UseProgram();
	s_bag->SetDogTexParam(texU, texD);
}



void ShaderMan::UseShaderGenListInit(int w, int h, int tight)
{
	if(tight)
	{
		s_bag->s_genlist_init_tight->UseProgram();
	}else
	{
		s_bag->s_genlist_init_ex->UseProgram();
		s_bag->SetGenListInitParam(w, h);
	}
}

void ShaderMan::UseShaderGenListHisto()
{
	s_bag->s_genlist_histo->UseProgram();

}




void ShaderMan::UseShaderGenListStart(float fw, int tex0)
{
	s_bag->s_genlist_start->UseProgram();
	s_bag->SetGenListStartParam(fw, tex0);
}

void ShaderMan::UseShaderGenListStep(int tex, int tex0)
{
	s_bag->s_genlist_step->UseProgram();
	s_bag->SetGenListStepParam( tex,  tex0);
}

void ShaderMan::UseShaderGenListEnd(int ktex)
{
	s_bag->s_genlist_end->UseProgram();
	s_bag->SetGenListEndParam(ktex);
}

void ShaderMan::UseShaderDebug()
{
	if(s_bag->s_debug)	s_bag->s_debug->UseProgram();
}

void ShaderMan::UseShaderZeroPass()
{
	if(s_bag->s_zero_pass) s_bag->s_zero_pass->UseProgram();
}

void ShaderMan::UseShaderGenVBO( float width, float fwidth, float size)
{
	s_bag->s_vertex_list->UseProgram();
	s_bag->SetGenVBOParam(width, fwidth, size);
}
void ShaderMan::UseShaderMarginCopy(int xmax, int ymax)
{
	s_bag->s_margin_copy->UseProgram();
	s_bag->SetMarginCopyParam(xmax, ymax);
	
}
void ShaderMan::UseShaderCopyKeypoint()
{
	s_bag->s_copy_key->UseProgram();
}

void ShaderMan::UseShaderSimpleOrientation(int oTex, float sigma, float sigma_step)
{
	s_bag->s_orientation->UseProgram();
	s_bag->SetSimpleOrientationInput(oTex, sigma, sigma_step);
}



void ShaderMan::UseShaderOrientation(int gtex, int width, int height, float sigma, int auxtex, float step, int keypoint_list)
{
	s_bag->s_orientation->UseProgram();

	//changes in v345. 
	//set sigma to 0 to identify keypoit list mode
	//set sigma to negative to identify fixed_orientation
	if(keypoint_list) sigma = 0.0f;
	else if(GlobalUtil::_FixedOrientation) sigma = - sigma;

	s_bag->SetFeatureOrientationParam(gtex, width, height, sigma, auxtex, step);
}

void ShaderMan::UseShaderDescriptor(int gtex, int otex, int dwidth, int fwidth,  int width, int height, float sigma)
{
	s_bag->s_descriptor_fp->UseProgram();
	s_bag->SetFeatureDescirptorParam(gtex, otex, (float)dwidth,  (float)fwidth, (float)width, (float)height, sigma);
}

void ShaderMan::SelectInitialSmoothingFilter(int octave_min, SiftParam&param)
{
    s_bag->SelectInitialSmoothingFilter(octave_min, param);
}
