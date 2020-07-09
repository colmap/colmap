////////////////////////////////////////////////////////////////////////////
//	File:		ShaderMan.h
//	Author:		Changchang Wu
//	Description : interface for the ShaderMan class.
//			This is a class that manages all the shaders for SIFT
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



#ifndef _SIFT_SHADER_MAN_H
#define _SIFT_SHADER_MAN_H


#include "ProgramGPU.h"
#include "ProgramGLSL.h"
///////////////////////////////////////////////////////////////////
//class ShaderMan
//description:	pure static class
//				wrapper of shaders from different GPU languages
///////////////////////////////////////////////////////////////////
class SiftParam;
class FilterGLSL;

class ShaderMan
{
public:
	static ShaderBag*   s_bag;
public:
	static void SelectInitialSmoothingFilter(int octave_min, SiftParam&param); 
	static void UseShaderMarginCopy(int xmax, int ymax);
	static void UseShaderOrientation(int gtex, int width, int height, float sigma, int auxtex, float step, int keypoint_list);
	static void UseShaderDescriptor(int gtex, int otex, int dwidth, int fwidth, int width, int height, float sigma);
	static void UseShaderSimpleOrientation(int oTex, float sigma, float sigma_step);
	static void UseShaderCopyKeypoint();
	static void UseShaderGenVBO( float width, float fwidth,  float size);
	static void UseShaderDebug();
	static void UseShaderZeroPass();
	static void UseShaderGenListStart(float fw, int tex0);
	static void UseShaderGenListStep(int tex, int tex0);
	static void UseShaderGenListEnd(int ktex);
	static void UseShaderGenListHisto();
	static void UseShaderGenListInit(int w, int h, int tight = 1);
	static void UseShaderKeypoint(int texU, int texD);
	static void UseShaderGradientPass(int texP = 0);
	static void UseShaderDisplayKeypoints();
	static void UseShaderDisplayGrad();
	static void UseShaderRGB2Gray();
	static void UseShaderDisplayDOG();
	static void UseShaderDisplayGaussian();
    ///////////////////////////////////////////
    static void FilterInitialImage(GLTexImage* tex, GLTexImage* buf);
    static void FilterSampledImage(GLTexImage* tex, GLTexImage* buf);
    static void FilterImage(FilterProgram* filter, GLTexImage *dst, GLTexImage *src, GLTexImage*tmp);
	static void TextureCopy(GLTexImage*dst, GLTexImage*src);
	static void TextureDownSample(GLTexImage* dst, GLTexImage*src, int scale = 2);
	static void TextureUpSample(GLTexImage* dst, GLTexImage*src, int scale);
	///////////////////////////////////////////////
	static void InitShaderMan(SiftParam&param);
	static void DestroyShaders();
	static int  HaveShaderMan(){return s_bag != NULL;}
	static void UnloadProgram();
};

#endif 
