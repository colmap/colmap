////////////////////////////////////////////////////////////////////////////
//	File:		PyramidGL.cpp
//	Author:		Changchang Wu
//	Description : implementation of PyramidGL/PyramidNaive/PyramidPackdc .
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
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <fstream>
#include <math.h>
#include <string.h>
using namespace std;

#include "GlobalUtil.h"
#include "GLTexImage.h"
#include "SiftGPU.h"
#include "ShaderMan.h"
#include "SiftPyramid.h"
#include "ProgramGLSL.h"
#include "PyramidGL.h"
#include "FrameBufferObject.h"

#ifdef USE_SSE_FOR_SIFTGPU
#ifndef __SSE__
#error Compiling SSE functions but SSE is not supported by the compiler.
#endif
#include <xmmintrin.h>
#endif


#define USE_TIMING()		double t, t0, tt;
#define OCTAVE_START()		if(GlobalUtil::_timingO){	t = t0 = CLOCK();	cout<<"#"<<i+_down_sample_factor<<"\t";	}
#define LEVEL_FINISH()		if(GlobalUtil::_timingL){	glFinish();	tt = CLOCK();cout<<(tt-t)<<"\t";	t = CLOCK();}
#define OCTAVE_FINISH()		if(GlobalUtil::_timingO)cout<<"|\t"<<(CLOCK()-t0)<<endl;


//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
PyramidNaive::PyramidNaive(SiftParam& sp): PyramidGL(sp)
{
	_texPyramid = NULL;
	_auxPyramid = NULL;
}

PyramidNaive::~PyramidNaive()
{
	DestroyPyramidData();
}

//align must be 2^i
void PyramidGL::	GetAlignedStorageSize(int num, int align,  int &fw, int &fh)
{
	if(num <=0)
	{
		fw = fh = 0;
	}else if(num < align*align)
	{
		fw = align;
		fh = (int)ceil(double(num) / fw);
	}else if(GlobalUtil::_NarrowFeatureTex)
	{
		double dn = double(num);
		int nb = (int) ceil(dn/GlobalUtil::_texMaxDim/align);	
		fw = align * nb;	
		fh = (int)ceil(dn /fw);/**/
	}else
	{
		double dn = double(num);
		int nb = (int) ceil(dn/GlobalUtil::_texMaxDim/align);
		fh = align * nb;
		if(nb <=1)
		{
			fw = (int)ceil(dn / fh);
			//align this dimension to blocksize
			fw = ((int) ceil(double(fw) /align))*align;
		}else
		{
			fw = GlobalUtil::_texMaxDim;
		}

	}


}

void PyramidGL::GetTextureStorageSize(int num, int &fw, int& fh)
{
	if(num <=0)
	{
		fw = fh = 0;
	}else if(num <= GlobalUtil::_FeatureTexBlock)
	{
		fw = num;
		fh = 1;
	}else if(GlobalUtil::_NarrowFeatureTex)
	{
		double dn = double(num);
		int nb = (int) ceil(dn/GlobalUtil::_texMaxDim/GlobalUtil::_FeatureTexBlock);	
		fw = GlobalUtil::_FeatureTexBlock * nb;	
		fh = (int)ceil(dn /fw);/**/
	}else
	{
		double dn = double(num);
		int nb = (int) ceil(dn/GlobalUtil::_texMaxDim/GlobalUtil::_FeatureTexBlock);
		fh = GlobalUtil::_FeatureTexBlock * nb;
		if(nb <=1)
		{
			fw = (int)ceil(dn / fh);

			//align this dimension to blocksize

			//
			if( fw < fh)
			{
				int temp = fh;
				fh = fw;
				fw = temp;
			}
		}else
		{
			fw = GlobalUtil::_texMaxDim;
		}
	}
}

void PyramidNaive::DestroyPyramidData()
{
	if(_texPyramid)
	{
		delete [] _texPyramid;
		_texPyramid = NULL;
	}
	if(_auxPyramid)
	{
		delete [] _auxPyramid;  
		_auxPyramid = NULL;
	}
}
PyramidGL::PyramidGL(SiftParam &sp):SiftPyramid(sp)
{
	_featureTex = NULL;
	_orientationTex = NULL;
	_descriptorTex = NULL;
	_histoPyramidTex = NULL;	
    //////////////////////////
    InitializeContext();
}

PyramidGL::~PyramidGL()
{
	DestroyPerLevelData();
	DestroySharedData();
	ShaderMan::DestroyShaders();
}

void PyramidGL::InitializeContext()
{
    GlobalUtil::InitGLParam(0);
    if(!GlobalUtil::_GoodOpenGL) return;

    //////////////////////////////////////////////
	ShaderMan::InitShaderMan(param);
}

void PyramidGL::DestroyPerLevelData()
{
	//integers vector to store the feature numbers.
	if(_levelFeatureNum)
	{
		delete [] _levelFeatureNum;
		_levelFeatureNum = NULL;
	}
	//texture used to store features
	if(	_featureTex)
	{
		delete [] _featureTex;
		_featureTex =	NULL;
	}
	//texture used for multi-orientation 
	if(_orientationTex)
	{
		delete [] _orientationTex;
		_orientationTex = NULL;
	}
	int no = _octave_num* param._dog_level_num;

	//two sets of vbos used to display the features
	if(_featureDisplayVBO)
	{
		glDeleteBuffers(no, _featureDisplayVBO);
		delete [] _featureDisplayVBO;
		_featureDisplayVBO = NULL;
	}
	if( _featurePointVBO)
	{
		glDeleteBuffers(no, _featurePointVBO);
		delete [] _featurePointVBO;
		_featurePointVBO = NULL;
	}

}

void PyramidGL::DestroySharedData()
{
	//histogram reduction
	if(_histoPyramidTex)
	{
		delete[]	_histoPyramidTex;
		_hpLevelNum = 0;
		_histoPyramidTex = NULL;
	}

	//descriptor storage shared by all levels
	if(_descriptorTex)
	{
		delete [] _descriptorTex;
		_descriptorTex = NULL;
	}
	//cpu reduction buffer.
	if(_histo_buffer)
	{
		delete[] _histo_buffer;
		_histo_buffer = 0;
	}
}

void PyramidNaive::FitHistogramPyramid()
{
	GLTexImage * tex, *htex;
	int hist_level_num = _hpLevelNum - _pyramid_octave_first; 

	tex = GetBaseLevel(_octave_min , DATA_KEYPOINT) + 2;
	htex = _histoPyramidTex + hist_level_num - 1;
	int w = tex->GetImgWidth() >> 1;
	int h = tex->GetImgHeight() >> 1;

	for(int k = 0; k <hist_level_num -1; k++, htex--)
	{
		if(htex->GetImgHeight()!= h || htex->GetImgWidth() != w)
		{	
			htex->SetImageSize(w, h);
			htex->ZeroHistoMargin();
		}

		w = (w + 1)>>1; h = (h + 1) >> 1;
	}
}

void PyramidNaive::FitPyramid(int w, int h)
{
	//(w, h) <= (_pyramid_width, _pyramid_height);

	_pyramid_octave_first = 0;
	//
	_octave_num  = GlobalUtil::_octave_num_default;

	int _octave_num_max = GetRequiredOctaveNum(min(w, h));

	if(_octave_num < 1 || _octave_num > _octave_num_max) 
	{
		_octave_num = _octave_num_max;
	}


	int pw = _pyramid_width>>1, ph = _pyramid_height>>1;
	while(_pyramid_octave_first + _octave_num < _pyramid_octave_num &&  
		pw >= w && ph >= h)
	{
		_pyramid_octave_first++;
		pw >>= 1;
		ph >>= 1;
	}

	for(int i = 0; i < _octave_num; i++)
	{
		GLTexImage * tex = GetBaseLevel(i + _octave_min);
		GLTexImage * aux = GetBaseLevel(i + _octave_min, DATA_KEYPOINT);
		for(int j = param._level_min; j <= param._level_max; j++, tex++, aux++)
		{
			tex->SetImageSize(w, h);
			aux->SetImageSize(w, h);
		}
		w>>=1;
		h>>=1;
	}
}
void PyramidNaive::InitPyramid(int w, int h, int ds)
{
	int wp, hp, toobig = 0;
	if(ds == 0)
	{
		_down_sample_factor = 0;
		if(GlobalUtil::_octave_min_default>=0)
		{
			wp = w >> GlobalUtil::_octave_min_default;
			hp = h >> GlobalUtil::_octave_min_default;
		}else 
		{
			wp = w << (-GlobalUtil::_octave_min_default);
			hp = h << (-GlobalUtil::_octave_min_default);
		}
		_octave_min = _octave_min_default;
	}else
	{
		//must use 0 as _octave_min; 
		_octave_min = 0;
		_down_sample_factor = ds;
		w >>= ds;
		h >>= ds;
		wp = w;
		hp = h; 

	}

	while(wp > GlobalUtil::_texMaxDim || hp > GlobalUtil::_texMaxDim)
	{
		_octave_min ++;
		wp >>= 1;
		hp >>= 1;
		toobig = 1;
	}

	while(GlobalUtil::_MemCapGPU > 0 && GlobalUtil::_FitMemoryCap  && (wp >_pyramid_width || hp > _pyramid_height) &&
		max(max(wp, hp), max(_pyramid_width, _pyramid_height)) >  1024 * sqrt(GlobalUtil::_MemCapGPU / 140.0))
	{
		_octave_min ++;
		wp >>= 1;
		hp >>= 1;
		toobig = 2;
	}

	if(toobig && GlobalUtil::_verbose)
	{
		std::cout<<(toobig == 2 ? "[**SKIP OCTAVES**]:\tExceeding Memory Cap (-nomc)\n" :
					"[**SKIP OCTAVES**]:\tReaching the dimension limit (-maxd)!\n");
	}

	if( wp == _pyramid_width && hp == _pyramid_height && _allocated )
	{
		FitPyramid(wp, hp);
	}else if(GlobalUtil::_ForceTightPyramid || _allocated ==0)
	{
		ResizePyramid(wp, hp);
	}
	else if( wp > _pyramid_width || hp > _pyramid_height )
	{
		ResizePyramid(max(wp, _pyramid_width), max(hp, _pyramid_height));
		if(wp < _pyramid_width || hp < _pyramid_height)  FitPyramid(wp, hp);
	}
	else
	{
		//try use the pyramid allocated for large image on small input images
		FitPyramid(wp, hp);
	}

	//select the initial smoothing filter according to the new _octave_min
	ShaderMan::SelectInitialSmoothingFilter(_octave_min + _down_sample_factor, param);
}

void PyramidNaive::ResizePyramid( int w,  int h)
{
	//
	unsigned int totalkb = 0;
	int _octave_num_new, input_sz;
	int i, j;
	GLTexImage * tex, *aux;
	//

	if(_pyramid_width == w && _pyramid_height == h && _allocated) return;

	if(w > GlobalUtil::_texMaxDim || h > GlobalUtil::_texMaxDim) return ;

	if(GlobalUtil::_verbose && GlobalUtil::_timingS) std::cout<<"[Allocate Pyramid]:\t" <<w<<"x"<<h<<endl;
	//first octave does not change
	_pyramid_octave_first = 0;

	
	//compute # of octaves

	input_sz = min(w,h) ;


	_pyramid_width =  w;
	_pyramid_height =  h;

	//reset to preset parameters
	_octave_num_new  = GlobalUtil::_octave_num_default;

	if(_octave_num_new < 1) _octave_num_new = GetRequiredOctaveNum(input_sz)  ;

	if(_pyramid_octave_num != _octave_num_new)
	{
		//destroy the original pyramid if the # of octave changes
		if(_octave_num >0)
		{
			DestroyPerLevelData();
			DestroyPyramidData();
		}
		_pyramid_octave_num = _octave_num_new;
	}

	_octave_num = _pyramid_octave_num;

	int noct = _octave_num;
	int nlev = param._level_num;

	//	//initialize the pyramid
	if(_texPyramid==NULL)	_texPyramid = new GLTexImage[ noct* nlev ];
	if(_auxPyramid==NULL)	_auxPyramid = new GLTexImage[ noct* nlev ];


	tex = GetBaseLevel(_octave_min, DATA_GAUSSIAN);
	aux = GetBaseLevel(_octave_min, DATA_KEYPOINT);
	for(i = 0; i< noct; i++)
	{
		totalkb += (nlev * w * h * 16 / 1024);
		for( j = 0; j< nlev; j++, tex++)
		{
			tex->InitTexture(w, h);
			//tex->AttachToFBO(0);
		}
		//several auxilary textures are not actually required
		totalkb += ((nlev - 3) * w * h * 16 /1024);
		for( j = 0; j< nlev ; j++, aux++)
		{
			if(j < 2) continue;
			if(j >= nlev - 1) continue;
			aux->InitTexture(w, h, 0);
			//aux->AttachToFBO(0);
		}

		w>>=1;
		h>>=1;
	}

	totalkb += ResizeFeatureStorage();


	//
	_allocated = 1;

	if(GlobalUtil::_verbose && GlobalUtil::_timingS) std::cout<<"[Allocate Pyramid]:\t" <<(totalkb/1024)<<"MB\n";

}


int PyramidGL::ResizeFeatureStorage()
{
	int totalkb = 0;
	if(_levelFeatureNum==NULL)	_levelFeatureNum = new int[_octave_num * param._dog_level_num];
	std::fill(_levelFeatureNum, _levelFeatureNum+_octave_num * param._dog_level_num, 0); 

	int wmax = GetBaseLevel(_octave_min)->GetDrawWidth();
	int hmax = GetBaseLevel(_octave_min)->GetDrawHeight();
	int w ,h, i;

	//use a fbo to initialize textures..
	FrameBufferObject fbo;
	
	//
	if(_histo_buffer == NULL) _histo_buffer = new float[((size_t)1) << (2 + 2 * GlobalUtil::_ListGenSkipGPU)];
	//histogram for feature detection

	int num = (int)ceil(log(double(max(wmax, hmax)))/log(2.0));

	if( _hpLevelNum != num)
	{
		_hpLevelNum = num;
		if(GlobalUtil::_ListGenGPU)
		{
			if(_histoPyramidTex ) delete [] _histoPyramidTex;
			_histoPyramidTex = new GLTexImage[_hpLevelNum];
			w = h = 1 ;
			for(i = 0; i < _hpLevelNum; i++)
			{
				_histoPyramidTex[i].InitTexture(w, h, 0);
				_histoPyramidTex[i].AttachToFBO(0);
				w<<=1;
				h<<=1;
			}
		}
	}

	// (4 ^ (_hpLevelNum) -1 / 3) pixels
	if(GlobalUtil::_ListGenGPU) totalkb += (((1 << (2 * _hpLevelNum)) -1) / 3 * 16 / 1024);



	//initialize the feature texture

	int idx = 0, n = _octave_num * param._dog_level_num;
	if(_featureTex==NULL)	_featureTex = new GLTexImage[n];
	if(GlobalUtil::_MaxOrientation >1 && GlobalUtil::_OrientationPack2==0)
	{
		if(_orientationTex== NULL)		_orientationTex = new GLTexImage[n];
	}


	for(i = 0; i < _octave_num; i++)
	{
		GLTexImage * tex = GetBaseLevel(i+_octave_min);
		int fmax = int(tex->GetImgWidth()*tex->GetImgHeight()*GlobalUtil::_MaxFeaturePercent);
		int fw, fh;
		//
		if(fmax > GlobalUtil::_MaxLevelFeatureNum) fmax = GlobalUtil::_MaxLevelFeatureNum;
		else if(fmax < 32) fmax = 32;	//give it at least a space of 32 feature

		GetTextureStorageSize(fmax, fw, fh);
		
		for(int j = 0; j < param._dog_level_num; j++, idx++)
		{

			_featureTex[idx].InitTexture(fw, fh, 0);
			_featureTex[idx].AttachToFBO(0);
			//
			if(_orientationTex)
			{
				_orientationTex[idx].InitTexture(fw, fh, 0);
				_orientationTex[idx].AttachToFBO(0);
			}
		}
		totalkb += fw * fh * 16 * param._dog_level_num * (_orientationTex? 2 : 1) /1024;
	}


	//this just need be initialized once
	if(_descriptorTex==NULL)
	{
		//initialize feature texture pyramid
		wmax = _featureTex->GetImgWidth();
		hmax = _featureTex->GetImgHeight();

		int nf, ns;
		if(GlobalUtil::_DescriptorPPT)
		{
			//32*4 = 128. 
			nf = 32 / GlobalUtil::_DescriptorPPT;	// how many textures we need
			ns = max(4, GlobalUtil::_DescriptorPPT);		    // how many point in one texture for one descriptor
		}else
		{
			//at least one, resue for visualization and other work
			nf = 1; ns = 4;
		}
		//
		_alignment = ns;
		//
		_descriptorTex = new GLTexImage[nf];

		int fw, fh;
		GetAlignedStorageSize(hmax*wmax* max(ns, 10), _alignment, fw, fh);

		if(fh < hmax ) fh = hmax;
		if(fw < wmax ) fw = wmax;

		totalkb += ( fw * fh * nf * 16 /1024);
		for(i =0; i < nf; i++)
		{
			_descriptorTex[i].InitTexture(fw, fh);
		}
	}else
	{
		int nf = GlobalUtil::_DescriptorPPT? 32 / GlobalUtil::_DescriptorPPT: 1;
		totalkb += nf * _descriptorTex[0].GetTexWidth() * _descriptorTex[0].GetTexHeight() * 16 /1024;
	}
	return totalkb;
}


void PyramidNaive::BuildPyramid(GLTexInput *input)
{
	USE_TIMING();
	GLTexPacked * tex;
	FilterProgram** filter;
	FrameBufferObject fbo;

	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	input->FitTexViewPort();

	for (int i = _octave_min; i < _octave_min + _octave_num; i++)
	{

		tex = (GLTexPacked*)GetBaseLevel(i);
		filter = ShaderMan::s_bag->f_gaussian_step;

		OCTAVE_START();

		if( i == _octave_min )
		{
			if(i < 0)	TextureUpSample(tex, input, 1<<(-i)	);			
			else        TextureDownSample(tex, input, 1<<i);
	        ShaderMan::FilterInitialImage(tex, NULL);
		}else
		{
			TextureDownSample(tex, GetLevelTexture(i-1, param._level_ds)); 
            ShaderMan::FilterSampledImage(tex, NULL); 
        }
		LEVEL_FINISH();

		for(int j = param._level_min + 1; j <=  param._level_max ; j++, tex++, filter++)
		{
			// filtering
            ShaderMan::FilterImage(*filter, tex+1, tex, NULL);
			LEVEL_FINISH();
		}
		OCTAVE_FINISH();

	}
	if(GlobalUtil::_timingS)	glFinish();
	UnloadProgram();
}






GLTexImage*  PyramidNaive::GetLevelTexture(int octave, int level, int dataName)
{
	if(octave <_octave_min || octave > _octave_min + _octave_num) return NULL;
	switch(dataName)
	{
		case DATA_GAUSSIAN:
		case DATA_DOG:
		case DATA_GRAD:
		case DATA_ROT:
			return _texPyramid+ (_pyramid_octave_first + octave - _octave_min) * param._level_num + (level - param._level_min);
		case DATA_KEYPOINT:
			return _auxPyramid + (_pyramid_octave_first + octave - _octave_min) * param._level_num + (level - param._level_min);
		default:
			return NULL;
	}
}

GLTexImage*  PyramidNaive::GetLevelTexture(int octave, int level)
{
	return _texPyramid+ (_pyramid_octave_first + octave - _octave_min) * param._level_num 
		+ (level - param._level_min);
}

//in the packed implementation
// DATA_GAUSSIAN, DATA_DOG, DATA_GAD will be stored in different textures.
GLTexImage*  PyramidNaive::GetBaseLevel(int octave, int dataName)
{
	if(octave <_octave_min || octave > _octave_min + _octave_num) return NULL;
	switch(dataName)
	{
		case DATA_GAUSSIAN:
		case DATA_DOG:
		case DATA_GRAD:
		case DATA_ROT:
			return _texPyramid+ (_pyramid_octave_first + octave - _octave_min) * param._level_num;
		case DATA_KEYPOINT:
			return _auxPyramid + (_pyramid_octave_first + octave - _octave_min) * param._level_num;
		default:
			return NULL;
	}
}









void PyramidNaive::ComputeGradient()
{

	int i, j;
	double  ts, t1;
	GLTexImage * tex;
	FrameBufferObject fbo;


	if(GlobalUtil::_timingS && GlobalUtil::_verbose)ts = CLOCK();
	
	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

	for ( i = _octave_min; i < _octave_min + _octave_num; i++)
	{
		for( j = param._level_min + 1 ; j < param._level_max ; j++)
		{
			tex = GetLevelTexture(i, j);
			tex->FitTexViewPort();
			tex->AttachToFBO(0);
			tex->BindTex();
			ShaderMan::UseShaderGradientPass();
			tex->DrawQuadMT4();
		}
	}		

	if(GlobalUtil::_timingS && GlobalUtil::_verbose)
	{
		glFinish();
		t1 = CLOCK();	
		std::cout<<"<Compute Gradient>\t"<<(t1-ts)<<"\n";
	}

	UnloadProgram();
	GLTexImage::UnbindMultiTex(3);
	fbo.UnattachTex(GL_COLOR_ATTACHMENT1_EXT);
}


//keypoint detection with subpixel localization
void PyramidNaive::DetectKeypointsEX()
{
	int i, j;
	double t0, t, ts, t1, t2;
	GLTexImage * tex, *aux;
	FrameBufferObject fbo;

	if(GlobalUtil::_timingS && GlobalUtil::_verbose)ts = CLOCK();
	
	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	//extra gradient data required for visualization
	int gradient_only_levels[2] = {param._level_min +1, param._level_max};
	int n_gradient_only_level = GlobalUtil::_UseSiftGPUEX ? 2 : 1;
	for ( i = _octave_min; i < _octave_min + _octave_num; i++)
	{
		for( j =0; j < n_gradient_only_level ; j++)
		{
			tex = GetLevelTexture(i, gradient_only_levels[j]);
			tex->FitTexViewPort();
			tex->AttachToFBO(0);
			tex->BindTex();
			ShaderMan::UseShaderGradientPass();
			tex->DrawQuadMT4();
		}
	}		

	if(GlobalUtil::_timingS && GlobalUtil::_verbose)
	{
		glFinish();
		t1 = CLOCK();
	}

	GLenum buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT };
	glDrawBuffers(2, buffers);
	for ( i = _octave_min; i < _octave_min + _octave_num; i++)
	{
		if(GlobalUtil::_timingO)
		{
			t0 = CLOCK();
			std::cout<<"#"<<(i + _down_sample_factor)<<"\t";
		}
		tex = GetBaseLevel(i) + 2;
		aux = GetBaseLevel(i, DATA_KEYPOINT) +2;
		aux->FitTexViewPort();

		for( j = param._level_min + 2; j <  param._level_max ; j++, aux++, tex++)
		{
			if(GlobalUtil::_timingL)t = CLOCK();		
			tex->AttachToFBO(0);
			aux->AttachToFBO(1);
			glActiveTexture(GL_TEXTURE0);
			tex->BindTex();
			glActiveTexture(GL_TEXTURE1);
			(tex+1)->BindTex();
			glActiveTexture(GL_TEXTURE2);
			(tex-1)->BindTex();
			ShaderMan::UseShaderKeypoint((tex+1)->GetTexID(), (tex-1)->GetTexID());
			aux->DrawQuadMT8();
	
			if(GlobalUtil::_timingL)
			{
				glFinish();
				std::cout<<(CLOCK()-t)<<"\t";
			}
			tex->DetachFBO(0);
			aux->DetachFBO(1);
		}
		if(GlobalUtil::_timingO)
		{
			std::cout<<"|\t"<<(CLOCK()-t0)<<"\n";
		}
	}

	if(GlobalUtil::_timingS)
	{
		glFinish();
		t2 = CLOCK();
		if(GlobalUtil::_verbose) 
			std::cout	<<"<Get Keypoints ..  >\t"<<(t2-t1)<<"\n"
						<<"<Extra Gradient..  >\t"<<(t1-ts)<<"\n";
	}
	UnloadProgram();
	GLTexImage::UnbindMultiTex(3);
	fbo.UnattachTex(GL_COLOR_ATTACHMENT1_EXT);


}

void PyramidNaive::GenerateFeatureList(int i, int j)
{
	int hist_level_num = _hpLevelNum - _pyramid_octave_first; 
	int hist_skip_gpu = GlobalUtil::_ListGenSkipGPU; 
    int idx = i * param._dog_level_num + j;
    GLTexImage* htex, *ftex, *tex;
	tex = GetBaseLevel(_octave_min + i, DATA_KEYPOINT) + 2 + j;
    ftex = _featureTex + idx;
	htex = _histoPyramidTex + hist_level_num - 1 - i;

	///
	glActiveTexture(GL_TEXTURE0);
	tex->BindTex();
	htex->AttachToFBO(0);
	int tight = ((htex->GetImgWidth() * 2 == tex->GetImgWidth() -1 || tex->GetTexWidth() == tex->GetImgWidth()) &&
				 (htex->GetImgHeight() *2 == tex->GetImgHeight()-1 || tex->GetTexHeight() == tex->GetImgHeight()));
	ShaderMan::UseShaderGenListInit(tex->GetImgWidth(), tex->GetImgHeight(), tight);
	htex->FitTexViewPort();
	//this uses the fact that no feature is on the edge.
	htex->DrawQuadReduction();

	//reduction..
	htex--;

	//this part might have problems on several GPUS
	//because the output of one pass is the input of the next pass
	//need to call glFinish to make it right
	//but too much glFinish makes it slow
	for(int k = 0; k <hist_level_num - i - 1 - hist_skip_gpu; k++, htex--)
	{
		htex->AttachToFBO(0);
		htex->FitTexViewPort();
		(htex+1)->BindTex();
		ShaderMan::UseShaderGenListHisto();
		htex->DrawQuadReduction();					
	}

	//
	if(hist_skip_gpu == 0)
	{	
		//read back one pixel
		float fn[4], fcount;
		glReadPixels(0, 0, 1, 1, GL_RGBA , GL_FLOAT, fn);
		fcount = (fn[0] + fn[1] + fn[2] + fn[3]);
		if(fcount < 1) fcount = 0;


		_levelFeatureNum[ idx] = (int)(fcount);
		SetLevelFeatureNum(idx, (int)fcount);
		_featureNum += int(fcount);

		//
		if(fcount < 1.0) return;
	

		///generate the feature texture

		htex=  _histoPyramidTex;

		htex->BindTex();

		//first pass
		ftex->AttachToFBO(0);
		if(GlobalUtil::_MaxOrientation>1)
		{
			//this is very important...
			ftex->FitRealTexViewPort();
			glClear(GL_COLOR_BUFFER_BIT);
			glFinish();
		}else
		{
			ftex->FitTexViewPort();
            //glFinish();
		}


		ShaderMan::UseShaderGenListStart((float)ftex->GetImgWidth(), htex->GetTexID());

		ftex->DrawQuad();
		//make sure it finishes before the next step
		ftex->DetachFBO(0);

		//pass on each pyramid level
		htex++;
	}else
	{

		int tw = htex[1].GetDrawWidth(), th = htex[1].GetDrawHeight();
		int fc = 0;
		glReadPixels(0, 0, tw, th, GL_RGBA , GL_FLOAT, _histo_buffer);	
		_keypoint_buffer.resize(0);
		for(int y = 0, pos = 0; y < th; y++)
		{
			for(int x= 0; x < tw; x++)
			{
				for(int c = 0; c < 4; c++, pos++)
				{
					int ss =  (int) _histo_buffer[pos]; 
					if(ss == 0) continue;
					float ft[4] = {2 * x + (c%2? 1.5f:  0.5f), 2 * y + (c>=2? 1.5f: 0.5f), 0, 1 };
					for(int t = 0; t < ss; t++)
					{
						ft[2] = (float) t; 
						_keypoint_buffer.insert(_keypoint_buffer.end(), ft, ft+4);
					}
					fc += (int)ss; 
				}
			}
		}
		_levelFeatureNum[ idx] = fc;
		SetLevelFeatureNum(idx, fc);
		if(fc == 0)  return;
        _featureNum += fc;
		/////////////////////
		ftex->AttachToFBO(0);
		if(GlobalUtil::_MaxOrientation>1)
		{
			ftex->FitRealTexViewPort();
			glClear(GL_COLOR_BUFFER_BIT);
			glFlush();
		}else
		{					
			ftex->FitTexViewPort();
			glFlush();
		}
		_keypoint_buffer.resize(ftex->GetDrawWidth() * ftex->GetDrawHeight()*4, 0);
		///////////
		glActiveTexture(GL_TEXTURE0);
		ftex->BindTex();
		glTexSubImage2D(GlobalUtil::_texTarget, 0, 0, 0, ftex->GetDrawWidth(),
			ftex->GetDrawHeight(), GL_RGBA, GL_FLOAT, &_keypoint_buffer[0]);
		htex += 2;
	}

	for(int lev = 1 + hist_skip_gpu; lev < hist_level_num  - i; lev++, htex++)
	{

		glActiveTexture(GL_TEXTURE0);
		ftex->BindTex();
		ftex->AttachToFBO(0);
		glActiveTexture(GL_TEXTURE1);
		htex->BindTex();
		ShaderMan::UseShaderGenListStep(ftex->GetTexID(), htex->GetTexID());
		ftex->DrawQuad();
		ftex->DetachFBO(0);	
	}
	GLTexImage::UnbindMultiTex(2);

}

//generate feature list on GPU
void PyramidNaive::GenerateFeatureList()
{
	//generate the histogram0pyramid
	FrameBufferObject fbo;
	glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	double t1, t2; 
	int ocount, reverse = (GlobalUtil::_TruncateMethod == 1);
	_featureNum = 0;

	FitHistogramPyramid();

	//for(int i = 0, idx = 0; i < _octave_num; i++)
    FOR_EACH_OCTAVE(i, reverse)
	{
		//output
		if(GlobalUtil::_timingO)
		{
            t1= CLOCK();
			ocount = 0;
			std::cout<<"#"<<i+_octave_min + _down_sample_factor<<":\t";
		}
		//for(int j = 0; j < param._dog_level_num; j++, idx++)
        FOR_EACH_LEVEL(j, reverse)
        {

            if(GlobalUtil::_TruncateMethod && GlobalUtil::_FeatureCountThreshold > 0
				&& _featureNum > GlobalUtil::_FeatureCountThreshold) 
			{
				_levelFeatureNum[i * param._dog_level_num + j] = 0;
				continue;
			}else
			{
				GenerateFeatureList(i, j); 
				if(GlobalUtil::_timingO)	
				{
					int idx = i * param._dog_level_num + j;
					std::cout<< _levelFeatureNum[idx] <<"\t";
					ocount += _levelFeatureNum[idx];
				}
			}
		}
		if(GlobalUtil::_timingO)
		{	
			t2 = CLOCK(); 
			std::cout << "| \t" << int(ocount) << " :\t(" << (t2 - t1) << ")\n";
		}
	}
	if(GlobalUtil::_timingS)glFinish();
	if(GlobalUtil::_verbose)
	{
		std::cout<<"#Features:\t"<<_featureNum<<"\n";
	}
}


void PyramidGL::GenerateFeatureDisplayVBO()
{
	//use a big VBO to save all the SIFT box vertices
	int w, h, esize; GLint bsize;
	int nvbo = _octave_num * param._dog_level_num;
	//initialize the vbos
	if(_featureDisplayVBO==NULL)
	{
		_featureDisplayVBO = new GLuint[nvbo];
		glGenBuffers( nvbo, _featureDisplayVBO );
    }
    if(_featurePointVBO == NULL)
    {
		_featurePointVBO = new GLuint[nvbo];
		glGenBuffers(nvbo, _featurePointVBO);
	}

	FrameBufferObject fbo;
	glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	glActiveTexture(GL_TEXTURE0);
	//
	GLTexImage & tempTex = *_descriptorTex;
	//
	for(int i = 0, idx = 0; i < _octave_num; i++)
	{
		for(int j = 0; j < param._dog_level_num; j ++, idx++)
		{
			GLTexImage * ftex  = _featureTex + idx;
			if(_levelFeatureNum[idx]<=0)continue;

			//copy the texture into vbo
			fbo.BindFBO();
			tempTex.AttachToFBO(0);

			ftex->BindTex();
			ftex->FitTexViewPort();
			ShaderMan::UseShaderCopyKeypoint();
			ftex->DrawQuad();

			glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB,  _featurePointVBO[ idx]);
			glGetBufferParameteriv(GL_PIXEL_PACK_BUFFER_ARB, GL_BUFFER_SIZE, &bsize);
			esize = ftex->GetImgHeight() * ftex->GetImgWidth()*sizeof(float) *4;

			//increase size when necessary
			if(bsize < esize)
			{
				glBufferData(GL_PIXEL_PACK_BUFFER_ARB, esize*3/2 ,	NULL, GL_STATIC_DRAW_ARB);
				glGetBufferParameteriv(GL_PIXEL_PACK_BUFFER_ARB, GL_BUFFER_SIZE, &bsize);
			}

			//read back if we have enough buffer
			if(bsize >= esize) glReadPixels(0, 0, ftex->GetImgWidth(), ftex->GetImgHeight(), GL_RGBA, GL_FLOAT, 0);
			else  glBufferData(GL_PIXEL_PACK_BUFFER_ARB, 0,	NULL, GL_STATIC_DRAW_ARB);
			glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);


			//box display vbo
			int count = _levelFeatureNum[idx]* 10;
			GetAlignedStorageSize(count, _alignment, w, h);
			w = (int)ceil(double(count)/ h);

			//input
			fbo.BindFBO();
			ftex->BindTex();

			//output
			tempTex.AttachToFBO(0);
			GlobalUtil::FitViewPort(w, h);
			//shader
			ShaderMan::UseShaderGenVBO(	(float)ftex->GetImgWidth(),  (float) w, 
				param.GetLevelSigma(j + param._level_min + 1));
			GLTexImage::DrawQuad(0,  (float)w, 0, (float)h);
		
			//
			glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, _featureDisplayVBO[ idx]);
			glGetBufferParameteriv(GL_PIXEL_PACK_BUFFER_ARB, GL_BUFFER_SIZE, &bsize);
			esize = w*h * sizeof(float)*4;
			//increase size when necessary
			if(bsize < esize)
			{
				glBufferData(GL_PIXEL_PACK_BUFFER_ARB, esize*3/2,	NULL, GL_STATIC_DRAW_ARB);
				glGetBufferParameteriv(GL_PIXEL_PACK_BUFFER_ARB, GL_BUFFER_SIZE, &bsize);
			}
			
			//read back if we have enough buffer	
			if(bsize >= esize)	glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, 0);
			else glBufferData(GL_PIXEL_PACK_BUFFER_ARB, 0,	NULL, GL_STATIC_DRAW_ARB);
			glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);



			
		}
	}
	glReadBuffer(GL_NONE);
	glFinish();

}





void PyramidNaive::GetFeatureOrientations()
{
	GLTexImage * gtex;
	GLTexImage * stex = NULL;
	GLTexImage * ftex = _featureTex;
	GLTexImage * otex = _orientationTex;
	int sid = 0; 
	int * count	 = _levelFeatureNum;
	float sigma, sigma_step = powf(2.0f, 1.0f/param._dog_level_num);
	FrameBufferObject fbo;
	if(_orientationTex)
	{
		GLenum buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT };
		glDrawBuffers(2, buffers);
	}else
	{
		glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	}
	for(int i = 0; i < _octave_num; i++)
	{
		gtex = GetLevelTexture(i+_octave_min, param._level_min + 1);
		if(GlobalUtil::_SubpixelLocalization || GlobalUtil::_KeepExtremumSign)
			stex = GetBaseLevel(i+_octave_min, DATA_KEYPOINT) + 2;

		for(int j = 0; j < param._dog_level_num; j++, ftex++, otex++, count++, gtex++, stex++)
		{
			if(*count<=0)continue;

			sigma = param.GetLevelSigma(j+param._level_min+1);

			//
			ftex->FitTexViewPort();

			glActiveTexture(GL_TEXTURE0);
			ftex->BindTex();
			glActiveTexture(GL_TEXTURE1);
			gtex->BindTex();
			//
			ftex->AttachToFBO(0);
			if(_orientationTex)		otex->AttachToFBO(1);
			if(!_existing_keypoints && (GlobalUtil::_SubpixelLocalization|| GlobalUtil::_KeepExtremumSign))
			{
				glActiveTexture(GL_TEXTURE2);
				stex->BindTex();
				sid = * stex;
			}
			ShaderMan::UseShaderOrientation(gtex->GetTexID(),
				gtex->GetImgWidth(), gtex->GetImgHeight(),
				sigma, sid, sigma_step, _existing_keypoints);
			ftex->DrawQuad();
	//		glFinish();
			
		}
	}

	GLTexImage::UnbindMultiTex(3);
	if(GlobalUtil::_timingS)glFinish();

	if(_orientationTex)	fbo.UnattachTex(GL_COLOR_ATTACHMENT1_EXT);

}



//to compare with GPU feature list generation
void PyramidNaive::GenerateFeatureListCPU()
{

	FrameBufferObject fbo;
	_featureNum = 0;
	GLTexImage * tex = GetBaseLevel(_octave_min);
	float * mem = new float [tex->GetTexWidth()*tex->GetTexHeight()];
	vector<float> list;
	int idx = 0;
	for(int i = 0; i < _octave_num; i++)
	{
		for(int j = 0; j < param._dog_level_num; j++, idx++)
		{
			tex = GetBaseLevel(_octave_min + i, DATA_KEYPOINT) + j + 2;
			tex->BindTex();
			glGetTexImage(GlobalUtil::_texTarget, 0, GL_RED, GL_FLOAT, mem);
			//tex->AttachToFBO(0);
			//tex->FitTexViewPort();
			//glReadPixels(0, 0, tex->GetTexWidth(), tex->GetTexHeight(), GL_RED, GL_FLOAT, mem);
			//
			//make a list of 
			list.resize(0);
			float * p = mem;
			int fcount = 0 ;
			for(int k = 0; k < tex->GetTexHeight(); k++)
			{
				for( int m = 0; m < tex->GetTexWidth(); m ++, p++)
				{
					if(*p==0)continue;
					if(m ==0 || k ==0 || k >= tex->GetImgHeight() -1 || m >= tex->GetImgWidth() -1 ) continue;
					list.push_back(m+0.5f);
					list.push_back(k+0.5f);
					list.push_back(0);
					list.push_back(1);
					fcount ++;


				}
			}
			if(fcount==0)continue;


			
			GLTexImage * ftex = _featureTex+idx;
			_levelFeatureNum[idx] = (fcount);
			SetLevelFeatureNum(idx, fcount);

			_featureNum += (fcount);


			int fw = ftex->GetImgWidth();
			int fh = ftex->GetImgHeight();

			list.resize(4*fh*fw);

			ftex->BindTex();
			ftex->AttachToFBO(0);
	//		glTexImage2D(GlobalUtil::_texTarget, 0, GlobalUtil::_iTexFormat, fw, fh, 0, GL_BGRA, GL_FLOAT, &list[0]);
			glTexSubImage2D(GlobalUtil::_texTarget, 0, 0, 0, fw, fh, GL_RGBA, GL_FLOAT, &list[0]);
			//
		}
	}
	GLTexImage::UnbindTex();
	delete[] mem;
	if(GlobalUtil::_verbose)
	{
		std::cout<<"#Features:\t"<<_featureNum<<"\n";
	}
}

#define FEATURELIST_USE_PBO

void PyramidGL::ReshapeFeatureListCPU()
{
	//make a compact feature list, each with only one orientation
	//download orientations and the featue list
	//reshape it and upload it

	FrameBufferObject fbo;
	int i, szmax =0, sz;
	int n = param._dog_level_num*_octave_num;
	for( i = 0; i < n; i++)
	{
		sz = _featureTex[i].GetImgWidth() * _featureTex[i].GetImgHeight();
		if(sz > szmax ) szmax = sz;
	}
	float * buffer = new float[szmax*24];
	float * buffer1 = buffer;
	float * buffer2 = buffer + szmax*4;
	float * buffer3 = buffer + szmax*8;

	glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

#ifdef FEATURELIST_USE_PBO
    GLuint ListUploadPBO;
    glGenBuffers(1, &ListUploadPBO); 
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB,  ListUploadPBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, szmax * 8 * sizeof(float),	NULL, GL_STREAM_DRAW);
#endif

	_featureNum = 0;

#ifdef NO_DUPLICATE_DOWNLOAD
	const double twopi = 2.0*3.14159265358979323846;
	_keypoint_buffer.resize(0);
	float os = _octave_min>=0? float(1<<_octave_min): 1.0f/(1<<(-_octave_min));
	if(_down_sample_factor>0) os *= float(1<<_down_sample_factor); 
	float offset = GlobalUtil::_LoweOrigin? 0 : 0.5f;
#endif

	for(i = 0; i < n; i++)
	{
		if(_levelFeatureNum[i]==0)continue;

		_featureTex[i].AttachToFBO(0);
		_featureTex[i].FitTexViewPort();
		glReadPixels(0, 0, _featureTex[i].GetImgWidth(), _featureTex[i].GetImgHeight(),GL_RGBA, GL_FLOAT, buffer1);
		
		int fcount =0, ocount;
		float * src = buffer1;
		float * orientation  = buffer2;
		float * des = buffer3;
		if(GlobalUtil::_OrientationPack2 == 0)
		{	
			//read back orientations from another texture
			_orientationTex[i].AttachToFBO(0);
			glReadPixels(0, 0, _orientationTex[i].GetImgWidth(), _orientationTex[i].GetImgHeight(),GL_RGBA, GL_FLOAT, buffer2);
			//make the feature list
			for(int j = 0; j < _levelFeatureNum[i]; j++, src+=4, orientation+=4)
			{
				if(_existing_keypoints) 
				{
					des[0] = src[0];
					des[1] = src[1];
					des[2] = orientation[0];
					des[3] = src[3];			
					fcount++;
					des += 4;
				}else
				{
					ocount = (int)src[2];
					for(int k = 0 ; k < ocount; k++, des+=4)
					{
						des[0] = src[0];
						des[1] = src[1];
						des[2] = orientation[k];
						des[3] = src[3];			
						fcount++;
					}
				}
			}
		}else
		{
			_featureTex[i].DetachFBO(0);
			const static double factor  = 2.0*3.14159265358979323846/65535.0;
			for(int j = 0; j < _levelFeatureNum[i]; j++, src+=4)
			{
				unsigned short * orientations = (unsigned short*) (&src[2]);
				if(_existing_keypoints) 
				{
					des[0] = src[0];
					des[1] = src[1];
					des[2] = float( factor* orientations[0]);
					des[3] = src[3];			
					fcount++;
					des += 4;
				}else
				{
					if(orientations[0] != 65535)
					{
						des[0] = src[0];
						des[1] = src[1];
						des[2] = float( factor* orientations[0]);
						des[3] = src[3];			
						fcount++;
						des += 4;

						if(orientations[1] != 65535)
						{
							des[0] = src[0];
							des[1] = src[1];
							des[2] = float(factor* orientations[1]);
							des[3] = src[3];			
							fcount++;
							des += 4;
						}
					}
				}
			}
		}

		if (fcount == 0){	_levelFeatureNum[i] = 0; continue;	}

		//texture size --------------
		SetLevelFeatureNum(i, fcount);
		int nfw = _featureTex[i].GetImgWidth();
		int nfh = _featureTex[i].GetImgHeight();
		int sz = nfh * nfw;
		if(sz > fcount) memset(des, 0, sizeof(float) * (sz - fcount) * 4);

#ifndef FEATURELIST_USE_PBO
		_featureTex[i].BindTex();
		glTexSubImage2D(GlobalUtil::_texTarget, 0, 0, 0, nfw, nfh, GL_RGBA, GL_FLOAT, buffer3);
		_featureTex[i].UnbindTex();
#else
        float* mem = (float*) glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB,  GL_WRITE_ONLY);
        memcpy(mem, buffer3,  sz * 4 * sizeof(float) );
        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
		_featureTex[i].BindTex();
		glTexSubImage2D(GlobalUtil::_texTarget, 0, 0, 0, nfw, nfh, GL_RGBA, GL_FLOAT, 0);
		_featureTex[i].UnbindTex();
#endif

#ifdef NO_DUPLICATE_DOWNLOAD
		if(fcount > 0)
		{
			float oss = os * (1 << (i / param._dog_level_num));
			_keypoint_buffer.resize((_featureNum + fcount) * 4);
			float* ds = &_keypoint_buffer[_featureNum * 4];
			float* fs = buffer3;
			for(int k = 0;  k < fcount; k++, ds+=4, fs+=4)
			{
				ds[0] = oss*(fs[0]-0.5f) + offset;	//x
				ds[1] = oss*(fs[1]-0.5f) + offset;	//y
				ds[3] = (float)fmod(twopi-fs[2], twopi);	//orientation, mirrored
				ds[2] = oss*fs[3];  //scale
			}
		}
#endif
		_levelFeatureNum[i] = fcount;
		_featureNum += fcount;
	}

	delete[] buffer;
	if(GlobalUtil::_verbose)
	{
		std::cout<<"#Features MO:\t"<<_featureNum<<endl;
	}
    ///////////////////////////////////
#ifdef FEATURELIST_USE_PBO
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB,  0);
    glDeleteBuffers(1, &ListUploadPBO);
#endif
}



inline void PyramidGL::SetLevelFeatureNum(int idx, int fcount)
{
	int fw, fh;
	GLTexImage * ftex = _featureTex + idx;
	//set feature texture size. normally fh will be one
	GetTextureStorageSize(fcount, fw, fh);
	if(fcount >  ftex->GetTexWidth()*ftex->GetTexHeight())
	{
		ftex->InitTexture(fw, fh, 0);
		if(_orientationTex)			_orientationTex[idx].InitTexture(fw, fh, 0);

	}
	if(GlobalUtil::_NarrowFeatureTex)
		fh = fcount ==0? 0:(int)ceil(double(fcount)/fw);
	else
		fw = fcount ==0? 0:(int)ceil(double(fcount)/fh);
	ftex->SetImageSize(fw, fh);
	if(_orientationTex)		_orientationTex[idx].SetImageSize(fw, fh);
}

void PyramidGL::CleanUpAfterSIFT()
{
	GLTexImage::UnbindMultiTex(3);
	ShaderMan::UnloadProgram();
	FrameBufferObject::DeleteGlobalFBO();
	GlobalUtil::CleanupOpenGL();
}

void PyramidNaive::GetSimplifiedOrientation()
{
	//
	int idx = 0;
//	int n = _octave_num  * param._dog_level_num;
	float sigma, sigma_step = powf(2.0f, 1.0f/param._dog_level_num); 
	GLTexImage * ftex = _featureTex;

	FrameBufferObject fbo;
	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	for(int i = 0; i < _octave_num; i++)
	{
		GLTexImage *gtex = GetLevelTexture(i+_octave_min, 2+param._level_min);
		for(int j = 0; j < param._dog_level_num; j++, ftex++,  gtex++, idx ++)
		{
			if(_levelFeatureNum[idx]<=0)continue;
			sigma = param.GetLevelSigma(j+param._level_min+1);

			//
			ftex->AttachToFBO(0);
			ftex->FitTexViewPort();

			glActiveTexture(GL_TEXTURE0);
			ftex->BindTex();
			glActiveTexture(GL_TEXTURE1);
			gtex->BindTex();

			ShaderMan::UseShaderSimpleOrientation(gtex->GetTexID(), sigma, sigma_step);
			ftex->DrawQuad();
		}
	}

	GLTexImage::UnbindMultiTex(2);

}


#ifdef USE_SSE_FOR_SIFTGPU
	static inline float dotproduct_128d(float * p)
	{
		float z = 0.0f;
		__m128 sse =_mm_load_ss(&z);
		float* pf = (float*) (&sse);
		for( int i = 0; i < 32; i++, p+=4)
		{
			__m128 ps = _mm_loadu_ps(p);
			sse = _mm_add_ps(sse,  _mm_mul_ps(ps, ps));
		}
		return pf[0] + pf[1] + pf[2] + pf[3];

	}
	static inline void multiply_and_truncate_128d(float* p, float m)
	{
		float z = 0.2f;
		__m128 t = _mm_load_ps1(&z);
		__m128 r = _mm_load_ps1(&m);
		for(int i = 0; i < 32; i++, p+=4)
		{
			__m128 ps = _mm_loadu_ps(p);
			_mm_storeu_ps(p, _mm_min_ps(_mm_mul_ps(ps, r), t));
		}
	}
	static inline void multiply_128d(float* p, float m)
	{
		__m128 r = _mm_load_ps1(&m);
		for(int i = 0; i < 32; i++, p+=4)
		{
			__m128 ps = _mm_loadu_ps(p);
			_mm_storeu_ps(p, _mm_mul_ps(ps, r));
		}
	}
#endif


inline void PyramidGL::NormalizeDescriptor(int num, float*pd)
{

#ifdef USE_SSE_FOR_SIFTGPU
	for(int k = 0; k < num; k++, pd +=128)
	{
		float sq;
		//normalize and truncate to .2
		sq = dotproduct_128d(pd);		sq = 1.0f / sqrtf(sq);
		multiply_and_truncate_128d(pd, sq);

		//renormalize
		sq = dotproduct_128d(pd);		sq = 1.0f / sqrtf(sq);
		multiply_128d(pd, sq);
	}
#else
	//descriptor normalization runs on cpu for OpenGL implemenations
	for(int k = 0; k < num; k++, pd +=128)
	{
		int v;
		float* ppd, sq = 0;
		//int v;
		//normalize
		ppd = pd;
		for(v = 0 ; v < 128; v++, ppd++)	sq += (*ppd)*(*ppd);
		sq = 1.0f / sqrtf(sq);
		//truncate to .2
		ppd = pd;
		for(v = 0; v < 128; v ++, ppd++)	*ppd = min(*ppd*sq, 0.2f);

		//renormalize
		ppd = pd; sq = 0;
		for(v = 0; v < 128; v++, ppd++)	sq += (*ppd)*(*ppd);
		sq = 1.0f / sqrtf(sq);

		ppd = pd;
		for(v = 0; v < 128; v ++, ppd++)	*ppd = *ppd*sq;
	}

#endif
}

inline void PyramidGL::InterlaceDescriptorF2(int w, int h, float* buf, float* pd, int step)
{
	/*
	if(GlobalUtil::_DescriptorPPR == 8)
	{
		const int dstep = w * 128;
		float* pp1 = buf;
		float* pp2 = buf + step;

		for(int u = 0; u < h ; u++, pd+=dstep)
		{
			int v; 
			float* ppd = pd;
			for(v= 0; v < w; v++)
			{
				for(int t = 0; t < 8; t++)
				{
					*ppd++ = *pp1++;*ppd++ = *pp1++;*ppd++ = *pp1++;*ppd++ = *pp1++;
					*ppd++ = *pp2++;*ppd++ = *pp2++;*ppd++ = *pp2++;*ppd++ = *pp2++;
				}
				ppd += 64;
			}
			ppd = pd + 64;
			for(v= 0; v < w; v++)
			{
				for(int t = 0; t < 8; t++)
				{
					*ppd++ = *pp1++;*ppd++ = *pp1++;*ppd++ = *pp1++;*ppd++ = *pp1++;
					*ppd++ = *pp2++;*ppd++ = *pp2++;*ppd++ = *pp2++;*ppd++ = *pp2++;
				}
				ppd += 64;
			}
		}

	}else */
	if(GlobalUtil::_DescriptorPPR == 8)
	{
		//interlace
		for(int k = 0; k < 2; k++)
		{
			float* pp = buf + k * step;
			float* ppd = pd + k * 4;
			for(int u = 0; u < h ; u++)
			{
				int v; 
				for(v= 0; v < w; v++)
				{
					for(int t = 0; t < 8; t++)
					{
						ppd[0] = pp[0];
						ppd[1] = pp[1];
						ppd[2] = pp[2];
						ppd[3] = pp[3];
						ppd += 8;
						pp+= 4;
					}
					ppd += 64;
				}
				ppd += ( 64 - 128 * w );
				for(v= 0; v < w; v++)
				{
					for(int t = 0; t < 8; t++)
					{
						ppd[0] = pp[0];
						ppd[1] = pp[1];
						ppd[2] = pp[2];
						ppd[3] = pp[3];

						ppd += 8;
						pp+= 4;
					}
					ppd += 64;
				}
				ppd -=64;
			}
		}
	}else if(GlobalUtil::_DescriptorPPR == 4)
	{

	}



}
void PyramidGL::GetFeatureDescriptors()
{
	//descriptors...
	float sigma;
	int idx, i, j, k, w, h;
	int ndf = 32 / GlobalUtil::_DescriptorPPT; //number of textures
	int block_width = GlobalUtil::_DescriptorPPR;
	int block_height = GlobalUtil::_DescriptorPPT/GlobalUtil::_DescriptorPPR;
	float* pd =  &_descriptor_buffer[0], * pbuf  = NULL;
	vector<float>read_buffer, descriptor_buffer2;

	//use another buffer, if we need to re-order the descriptors
	if(_keypoint_index.size() > 0)
	{
		descriptor_buffer2.resize(_descriptor_buffer.size());
		pd = &descriptor_buffer2[0];
	}
	FrameBufferObject fbo;

	GLTexImage * gtex, *otex, * ftex;
	GLenum buffers[8] = { 
		GL_COLOR_ATTACHMENT0_EXT,		GL_COLOR_ATTACHMENT1_EXT ,
		GL_COLOR_ATTACHMENT2_EXT,		GL_COLOR_ATTACHMENT3_EXT ,
		GL_COLOR_ATTACHMENT4_EXT,		GL_COLOR_ATTACHMENT5_EXT ,
		GL_COLOR_ATTACHMENT6_EXT,		GL_COLOR_ATTACHMENT7_EXT ,
	};

	glDrawBuffers(ndf, buffers);
	glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);


	for( i = 0, idx = 0, ftex = _featureTex; i < _octave_num; i++)
	{
		gtex = GetBaseLevel(i + _octave_min, DATA_GRAD) + 1;
		otex = GetBaseLevel(i + _octave_min, DATA_ROT)  + 1;
		for( j = 0; j < param._dog_level_num; j++, ftex++, idx++, gtex++, otex++)
		{
			if(_levelFeatureNum[idx]==0)continue;

            sigma = IsUsingRectDescription()?  0 : param.GetLevelSigma(j+param._level_min+1);
			int count = _levelFeatureNum[idx] * block_width;
			GetAlignedStorageSize(count, block_width, w, h);
			h = ((int)ceil(double(count) / w)) * block_height;

			//not enought space for holding the descriptor data
			if(w > _descriptorTex[0].GetTexWidth() || h > _descriptorTex[0].GetTexHeight())
			{
				for(k = 0; k < ndf; k++)_descriptorTex[k].InitTexture(w, h);
			}
			for(k = 0; k < ndf; k++)	_descriptorTex[k].AttachToFBO(k);
			GlobalUtil::FitViewPort(w, h);
			glActiveTexture(GL_TEXTURE0);
			ftex->BindTex();
			glActiveTexture(GL_TEXTURE1);
			gtex->BindTex();
			if(otex!=gtex)
			{
				glActiveTexture(GL_TEXTURE2);
				otex->BindTex();
			}

			ShaderMan::UseShaderDescriptor(gtex->GetTexID(), otex->GetTexID(), 
				w, ftex->GetImgWidth(), gtex->GetImgWidth(), gtex->GetImgHeight(), sigma);
			GLTexImage::DrawQuad(0, (float)w, 0, (float)h);

			 //read back float format descriptors and do normalization on CPU
			int step = w*h*4;
			if((unsigned int)step*ndf > read_buffer.size())
			{
				read_buffer.resize(ndf*step);
			}
			pbuf = &read_buffer[0];
			
			//read back
			for(k = 0; k < ndf; k++, pbuf+=step)
			{
				glReadBuffer(GL_COLOR_ATTACHMENT0_EXT + k);
				if(GlobalUtil::_IsNvidia ||  w * h <=  16384) //were
				{
					glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, pbuf);
				}else
				{
					int hstep = 16384 / w; 
					for(int kk = 0; kk < h; kk += hstep)
						glReadPixels(0, kk, w, min(hstep, h - kk), GL_RGBA, GL_FLOAT, pbuf + w * kk * 4);
				}
			}
	
			//the following two steps run on cpu, so better cpu better speed
			//and release version can be a lot faster than debug version
			//interlace data on the two texture to get the descriptor
			InterlaceDescriptorF2(w / block_width, h / block_height, &read_buffer[0], pd, step);
			
			//need to do normalization
			//the new version uses SSE to speed up this part
			if(GlobalUtil::_NormalizedSIFT) NormalizeDescriptor(_levelFeatureNum[idx], pd);

			pd += 128*_levelFeatureNum[idx];
			glReadBuffer(GL_NONE);
		}
	}


	//finally, put the descriptor back to their original order for existing keypoint list.
	if(_keypoint_index.size() > 0)
	{
		for(i = 0; i < _featureNum; ++i)
		{
			int index = _keypoint_index[i];
			memcpy(&_descriptor_buffer[index*128], &descriptor_buffer2[i*128], 128 * sizeof(float));
		}
	}

	////////////////////////
	GLTexImage::UnbindMultiTex(3); 
	glDrawBuffer(GL_NONE);
	ShaderMan::UnloadProgram();
	if(GlobalUtil::_timingS)glFinish();
	for(i = 0; i < ndf; i++) fbo.UnattachTex(GL_COLOR_ATTACHMENT0_EXT +i);

}


void PyramidGL::DownloadKeypoints()
{
	const double twopi = 2.0*3.14159265358979323846;
	int idx = 0;
	float * buffer = &_keypoint_buffer[0];
	vector<float> keypoint_buffer2;
	//use a different keypoint buffer when processing with an exisint features list
	//without orientation information. 
	if(_keypoint_index.size() > 0)
	{
		keypoint_buffer2.resize(_keypoint_buffer.size());
		buffer = &keypoint_buffer2[0];
	}
	float * p = buffer, *ps, sigma;
	GLTexImage * ftex = _featureTex;
	FrameBufferObject fbo;
	ftex->FitRealTexViewPort();
	/////////////////////
	float os = _octave_min>=0? float(1<<_octave_min): 1.0f/(1<<(-_octave_min));
	if(_down_sample_factor>0) os *= float(1<<_down_sample_factor); 
	float offset = GlobalUtil::_LoweOrigin? 0 : 0.5f;
	/////////////////////
	for(int i = 0; i < _octave_num; i++, os *= 2.0f)
	{
		
		for(int j = 0; j  < param._dog_level_num; j++, idx++, ftex++)
		{

			if(_levelFeatureNum[idx]>0)
			{	
				ftex->AttachToFBO(0);
				glReadPixels(0, 0, ftex->GetImgWidth(), ftex->GetImgHeight(),GL_RGBA, GL_FLOAT, p);
				ps = p;
				for(int k = 0;  k < _levelFeatureNum[idx]; k++, ps+=4)
				{
					ps[0] = os*(ps[0]-0.5f) + offset;	//x
					ps[1] = os*(ps[1]-0.5f) + offset;	//y
					sigma = os*ps[3]; 
					ps[3] = (float)fmod(twopi-ps[2], twopi);	//orientation, mirrored
					ps[2] = sigma;  //scale
				}
				p+= 4* _levelFeatureNum[idx];
			}
		}
	}

	//put the feature into their original order

	if(_keypoint_index.size() > 0)
	{
		for(int i = 0; i < _featureNum; ++i)
		{
			int index = _keypoint_index[i];
			memcpy(&_keypoint_buffer[index*4], &keypoint_buffer2[i*4], 4 * sizeof(float));
		}
	}
}


void PyramidGL::GenerateFeatureListTex()
{
	//generate feature list texture from existing keypoints
	//do feature sorting in the same time?

	FrameBufferObject fbo;
	vector<float> list;
	int idx = 0;
	const double twopi = 2.0*3.14159265358979323846;
	float sigma_half_step = powf(2.0f, 0.5f / param._dog_level_num);
	float octave_sigma = _octave_min>=0? float(1<<_octave_min): 1.0f/(1<<(-_octave_min));
	float offset = GlobalUtil::_LoweOrigin? 0 : 0.5f; 
	if(_down_sample_factor>0) octave_sigma *= float(1<<_down_sample_factor); 

    
    std::fill(_levelFeatureNum, _levelFeatureNum + _octave_num * param._dog_level_num, 0);

	_keypoint_index.resize(0); // should already be 0
	for(int i = 0; i < _octave_num; i++, octave_sigma*= 2.0f)
	{
		for(int j = 0; j < param._dog_level_num; j++, idx++)
		{
			list.resize(0);
			float level_sigma = param.GetLevelSigma(j + param._level_min + 1) * octave_sigma;
			float sigma_min = level_sigma / sigma_half_step;
			float sigma_max = level_sigma * sigma_half_step;
			int fcount = 0 ;
			for(int k = 0; k < _featureNum; k++)
			{
				float * key = &_keypoint_buffer[k*4];
                float sigmak = key[2]; 

                //////////////////////////////////////
                if(IsUsingRectDescription()) sigmak = min(key[2], key[3]) / 12.0f; 

                if( (sigmak >= sigma_min && sigmak < sigma_max)
					||(sigmak < sigma_min && i ==0 && j == 0)
					||(sigmak > sigma_max && j == param._dog_level_num - 1&& 
                            (i == _octave_num -1  || GlobalUtil::_KeyPointListForceLevel0)))
				{
					//add this keypoint to the list
					list.push_back((key[0] - offset) / octave_sigma + 0.5f);
					list.push_back((key[1] - offset) / octave_sigma + 0.5f);
                    if(IsUsingRectDescription())
                    {
                        list.push_back(key[2] / octave_sigma);
                        list.push_back(key[3] / octave_sigma);
                    }else
                    {
					    list.push_back((float)fmod(twopi-key[3], twopi));
					    list.push_back(key[2] / octave_sigma);
                    }
					fcount ++;
					//save the index of keypoints
					_keypoint_index.push_back(k);
				}
			}

			_levelFeatureNum[idx] = fcount;
			if(fcount==0)continue;
			GLTexImage * ftex = _featureTex+idx;

			SetLevelFeatureNum(idx, fcount);

			int fw = ftex->GetImgWidth();
			int fh = ftex->GetImgHeight();

			list.resize(4*fh*fw);

			ftex->BindTex();
			ftex->AttachToFBO(0);
			glTexSubImage2D(GlobalUtil::_texTarget, 0, 0, 0, fw, fh, GL_RGBA, GL_FLOAT, &list[0]);

            if( fcount == _featureNum) _keypoint_index.resize(0);
		}
        if( GlobalUtil::_KeyPointListForceLevel0 ) break;
	}
	GLTexImage::UnbindTex();
	if(GlobalUtil::_verbose)
	{
		std::cout<<"#Features:\t"<<_featureNum<<"\n";
	}
}



PyramidPacked::PyramidPacked(SiftParam& sp): PyramidGL(sp)
{
	_allPyramid = NULL;
}

PyramidPacked::~PyramidPacked()
{
	DestroyPyramidData();
}


//build the gaussian pyrmaid

void PyramidPacked::BuildPyramid(GLTexInput * input)
{
	//
	USE_TIMING();
	GLTexImage * tex, *tmp;
	FilterProgram ** filter;
	FrameBufferObject fbo;

	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	input->FitTexViewPort();

	for (int i = _octave_min; i < _octave_min + _octave_num; i++)
	{
		tex = GetBaseLevel(i);
		tmp = GetBaseLevel(i, DATA_DOG) + 2; //use this as a temperory texture

		
		filter = ShaderMan::s_bag->f_gaussian_step;

		OCTAVE_START();

		if( i == _octave_min )
		{
			if(i < 0)	TextureUpSample(tex, input, 1<<(-i-1));			
			else	    TextureDownSample(tex, input, 1<<(i+1));
            ShaderMan::FilterInitialImage(tex, tmp); 
		}else
		{
			TextureDownSample(tex, GetLevelTexture(i-1, param._level_ds)); 
			ShaderMan::FilterSampledImage(tex, tmp); 
		}
		LEVEL_FINISH();		

		for(int j = param._level_min + 1; j <=  param._level_max ; j++, tex++, filter++)
		{
			// filtering
            ShaderMan::FilterImage(*filter, tex+1, tex, tmp);
			LEVEL_FINISH();
		}

		OCTAVE_FINISH();

	}
	if(GlobalUtil::_timingS)	glFinish();
	UnloadProgram();	
}

void PyramidPacked::ComputeGradient()
{
	
	//first pass, compute dog, gradient, orientation
	GLenum buffers[4] = { 
		GL_COLOR_ATTACHMENT0_EXT,		GL_COLOR_ATTACHMENT1_EXT ,
		GL_COLOR_ATTACHMENT2_EXT,		GL_COLOR_ATTACHMENT3_EXT
	};

	int i, j;
	double ts, t1;
	FrameBufferObject fbo;

	if(GlobalUtil::_timingS && GlobalUtil::_verbose)ts = CLOCK();

	for(i = _octave_min; i < _octave_min + _octave_num; i++)
	{
		GLTexImage * gus = GetBaseLevel(i) +  1;
		GLTexImage * dog = GetBaseLevel(i, DATA_DOG) +  1;
		GLTexImage * grd = GetBaseLevel(i, DATA_GRAD) +  1;
		GLTexImage * rot = GetBaseLevel(i, DATA_ROT) +  1;
		glDrawBuffers(3, buffers);
		gus->FitTexViewPort();
		//compute the gradient
		for(j = 0; j <  param._dog_level_num ; j++, gus++, dog++, grd++, rot++)
		{
			//gradient, dog, orientation
			glActiveTexture(GL_TEXTURE0);
			gus->BindTex();
			glActiveTexture(GL_TEXTURE1);
			(gus-1)->BindTex();
			//output
			dog->AttachToFBO(0);
			grd->AttachToFBO(1);
			rot->AttachToFBO(2);
			ShaderMan::UseShaderGradientPass((gus-1)->GetTexID());
			//compute
			dog->DrawQuadMT4();
		}
	}
	if(GlobalUtil::_timingS)
	{
		glFinish();
		if(GlobalUtil::_verbose)
		{
			t1 = CLOCK();
			std::cout	<<"<Gradient, DOG  >\t"<<(t1-ts)<<"\n";
		}
	}
	GLTexImage::DetachFBO(1);
	GLTexImage::DetachFBO(2);
	UnloadProgram();
	GLTexImage::UnbindMultiTex(3);
	fbo.UnattachTex(GL_COLOR_ATTACHMENT1_EXT);
}

void PyramidPacked::DetectKeypointsEX()
{

	//first pass, compute dog, gradient, orientation
	GLenum buffers[4] = { 
		GL_COLOR_ATTACHMENT0_EXT,		GL_COLOR_ATTACHMENT1_EXT ,
		GL_COLOR_ATTACHMENT2_EXT,		GL_COLOR_ATTACHMENT3_EXT
	};

	int i, j;
	double t0, t, ts, t1, t2;
	FrameBufferObject fbo;

	if(GlobalUtil::_timingS && GlobalUtil::_verbose)ts = CLOCK();

	for(i = _octave_min; i < _octave_min + _octave_num; i++)
	{
		GLTexImage * gus = GetBaseLevel(i) + 1;
		GLTexImage * dog = GetBaseLevel(i, DATA_DOG) + 1;
		GLTexImage * grd = GetBaseLevel(i, DATA_GRAD) + 1;
		GLTexImage * rot = GetBaseLevel(i, DATA_ROT) + 1;
		glDrawBuffers(3, buffers);
		gus->FitTexViewPort();
		//compute the gradient
		for(j = param._level_min +1; j <=  param._level_max ; j++, gus++, dog++, grd++, rot++)
		{
			//gradient, dog, orientation
			glActiveTexture(GL_TEXTURE0);
			gus->BindTex();
			glActiveTexture(GL_TEXTURE1);
			(gus-1)->BindTex();
			//output
			dog->AttachToFBO(0);
			grd->AttachToFBO(1);
			rot->AttachToFBO(2);
			ShaderMan::UseShaderGradientPass((gus-1)->GetTexID());
			//compute
			dog->DrawQuadMT4();
		}
	}
	if(GlobalUtil::_timingS && GlobalUtil::_verbose)
	{
		glFinish();
		t1 = CLOCK();
	}
	GLTexImage::DetachFBO(1);
	GLTexImage::DetachFBO(2);
	//glDrawBuffers(1, buffers);
	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	

	GlobalUtil::CheckErrorsGL();

	for ( i = _octave_min; i < _octave_min + _octave_num; i++)
	{
		if(GlobalUtil::_timingO)
		{
			t0 = CLOCK();
			std::cout<<"#"<<(i + _down_sample_factor)<<"\t";
		}
		GLTexImage * dog = GetBaseLevel(i, DATA_DOG) + 2;
		GLTexImage * key = GetBaseLevel(i, DATA_KEYPOINT) +2;
		GLTexImage * gus = GetBaseLevel(i) +  2;
		key->FitTexViewPort();

		for( j = param._level_min +2; j <  param._level_max ; j++, dog++, key++, gus++)
		{
			if(GlobalUtil::_timingL)t = CLOCK();		
			key->AttachToFBO(0);
			glActiveTexture(GL_TEXTURE0);
			dog->BindTex();
			glActiveTexture(GL_TEXTURE1);
			(dog+1)->BindTex();
			glActiveTexture(GL_TEXTURE2);
			(dog-1)->BindTex();
			if(GlobalUtil::_DarknessAdaption)
			{
				glActiveTexture(GL_TEXTURE3);
				gus->BindTex();
			}
			ShaderMan::UseShaderKeypoint((dog+1)->GetTexID(), (dog-1)->GetTexID());
			key->DrawQuadMT8();
			if(GlobalUtil::_timingL)
			{
				glFinish();
				std::cout<<(CLOCK()-t)<<"\t";
			}
		}
		if(GlobalUtil::_timingO)
		{
			glFinish();
			std::cout<<"|\t"<<(CLOCK()-t0)<<"\n";
		}
	}

	if(GlobalUtil::_timingS)
	{
		glFinish();
		if(GlobalUtil::_verbose) 
		{	
			t2 = CLOCK();
			std::cout	<<"<Gradient, DOG  >\t"<<(t1-ts)<<"\n"
						<<"<Get Keypoints  >\t"<<(t2-t1)<<"\n";
		}
						
	}
	UnloadProgram();
	GLTexImage::UnbindMultiTex(3);
	fbo.UnattachTex(GL_COLOR_ATTACHMENT1_EXT);
}


void PyramidPacked::GenerateFeatureList(int i, int j)
{
	float fcount = 0.0f; 
	int hist_skip_gpu = GlobalUtil::_ListGenSkipGPU;
    int idx = i * param._dog_level_num + j; 
    int hist_level_num = _hpLevelNum - _pyramid_octave_first;
	GLTexImage * htex, * ftex, * tex;
	htex = _histoPyramidTex + hist_level_num - 1 - i;
	ftex = _featureTex + idx;
	tex = GetBaseLevel(_octave_min + i, DATA_KEYPOINT) + 2 + j;


	//fill zero to an extra row/col if the height/width is odd
	glActiveTexture(GL_TEXTURE0);
	tex->BindTex();
	htex->AttachToFBO(0);
	int tight = (htex->GetImgWidth() * 4 == tex->GetImgWidth() -1 && htex->GetImgHeight() *4 == tex->GetImgHeight()-1 );
	ShaderMan::UseShaderGenListInit(tex->GetImgWidth(), tex->GetImgHeight(), tight);
	htex->FitTexViewPort();
	//this uses the fact that no feature is on the edge.
	htex->DrawQuadReduction();
	//reduction..
	htex--;
	
	//this part might have problems on several GPUS
	//because the output of one pass is the input of the next pass
	//may require glFinish to make it right, but too much glFinish makes it slow
	for(int k = 0; k <hist_level_num - i-1 - hist_skip_gpu; k++, htex--)
	{
		htex->AttachToFBO(0);
		htex->FitTexViewPort();
		(htex+1)->BindTex();
		ShaderMan::UseShaderGenListHisto();
		htex->DrawQuadReduction();		
	}

	if(hist_skip_gpu == 0)
	{		
		//read back one pixel
		float fn[4];
		glReadPixels(0, 0, 1, 1, GL_RGBA , GL_FLOAT, fn);
		fcount = (fn[0] + fn[1] + fn[2] + fn[3]);
		if(fcount < 1) fcount = 0;

		_levelFeatureNum[ idx] = (int)(fcount);
		SetLevelFeatureNum(idx, (int)fcount);

		//save  number of features
		_featureNum += int(fcount);

		//
		if(fcount < 1.0) 				return;;


		///generate the feature texture
		htex=  _histoPyramidTex;

		htex->BindTex();

		//first pass
		ftex->AttachToFBO(0);
		if(GlobalUtil::_MaxOrientation>1)
		{
			//this is very important...
			ftex->FitRealTexViewPort();
			glClear(GL_COLOR_BUFFER_BIT);
			glFinish();
		}else
		{	
			ftex->FitTexViewPort();
			//glFinish();
		}


		ShaderMan::UseShaderGenListStart((float)ftex->GetImgWidth(), htex->GetTexID());

		ftex->DrawQuad();
		//make sure it finishes before the next step
		ftex->DetachFBO(0);
		//pass on each pyramid level
		htex++;
	}else
	{

		int tw = htex[1].GetDrawWidth(), th = htex[1].GetDrawHeight();
		int fc = 0;
		glReadPixels(0, 0, tw, th, GL_RGBA , GL_FLOAT, _histo_buffer);	
		_keypoint_buffer.resize(0);
		for(int y = 0, pos = 0; y < th; y++)
		{
			for(int x= 0; x < tw; x++)
			{
				for(int c = 0; c < 4; c++, pos++)
				{
					int ss =  (int) _histo_buffer[pos]; 
					if(ss == 0) continue;
					float ft[4] = {2 * x + (c%2? 1.5f:  0.5f), 2 * y + (c>=2? 1.5f: 0.5f), 0, 1 };
					for(int t = 0; t < ss; t++)
					{
						ft[2] = (float) t; 
						_keypoint_buffer.insert(_keypoint_buffer.end(), ft, ft+4);
					}
					fc += (int)ss; 
				}
			}
		}
		_levelFeatureNum[ idx] = fc;
		SetLevelFeatureNum(idx, fc);
		if(fc == 0)  return; 

		fcount = (float) fc; 	
		_featureNum += fc;
		/////////////////////
		ftex->AttachToFBO(0);
		if(GlobalUtil::_MaxOrientation>1)
		{
			ftex->FitRealTexViewPort();
			glClear(GL_COLOR_BUFFER_BIT);
			glFlush();
		}else
		{					
			ftex->FitTexViewPort();
            glFlush();
		}
		_keypoint_buffer.resize(ftex->GetDrawWidth() * ftex->GetDrawHeight()*4, 0);
		///////////
		glActiveTexture(GL_TEXTURE0);
		ftex->BindTex();
		glTexSubImage2D(GlobalUtil::_texTarget, 0, 0, 0, ftex->GetDrawWidth(),
			ftex->GetDrawHeight(), GL_RGBA, GL_FLOAT, &_keypoint_buffer[0]);
		htex += 2; 
	}

	for(int lev = 1 + hist_skip_gpu; lev < hist_level_num  - i; lev++, htex++)
	{
		glActiveTexture(GL_TEXTURE0);
		ftex->BindTex();
		ftex->AttachToFBO(0);
		glActiveTexture(GL_TEXTURE1);
		htex->BindTex();
		ShaderMan::UseShaderGenListStep(ftex->GetTexID(), htex->GetTexID());
		ftex->DrawQuad();
		ftex->DetachFBO(0);	
	}

	ftex->AttachToFBO(0);
	glActiveTexture(GL_TEXTURE1);
	tex->BindTex();
	ShaderMan::UseShaderGenListEnd(tex->GetTexID());
	ftex->DrawQuad();
	GLTexImage::UnbindMultiTex(2);

}

void PyramidPacked::GenerateFeatureList()
{
	//generate the histogram pyramid
	FrameBufferObject fbo;
	glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	double t1, t2; 
	int ocount= 0, reverse = (GlobalUtil::_TruncateMethod == 1);
	_featureNum = 0;

	FitHistogramPyramid();
	//for(int i = 0, idx = 0; i < _octave_num; i++)
    FOR_EACH_OCTAVE(i, reverse)
	{
		if(GlobalUtil::_timingO)
		{
            t1= CLOCK();
			ocount = 0;
			std::cout<<"#"<<i+_octave_min + _down_sample_factor<<":\t";
		}
		//for(int j = 0; j < param._dog_level_num; j++, idx++)
        FOR_EACH_LEVEL(j, reverse)
		{
            if(GlobalUtil::_TruncateMethod && GlobalUtil::_FeatureCountThreshold > 0
				&& _featureNum > GlobalUtil::_FeatureCountThreshold) 
			{
				_levelFeatureNum[i * param._dog_level_num + j] = 0;
				continue;
			}
            
            GenerateFeatureList(i, j); 

			if(GlobalUtil::_timingO)
			{
                int idx = i * param._dog_level_num + j;
                ocount += _levelFeatureNum[idx];
				std::cout<< _levelFeatureNum[idx] <<"\t";
			}
		}
		if(GlobalUtil::_timingO)
		{	
            t2 = CLOCK();
			std::cout << "| \t" << int(ocount) << " :\t(" << (t2 - t1) << ")\n";
		}
	}
	if(GlobalUtil::_timingS)glFinish();
	if(GlobalUtil::_verbose)
	{
		std::cout<<"#Features:\t"<<_featureNum<<"\n";
	}

}

void PyramidPacked::GenerateFeatureListCPU()
{
	FrameBufferObject fbo;
	_featureNum = 0;
	GLTexImage * tex = GetBaseLevel(_octave_min);
	float * mem = new float [tex->GetTexWidth()*tex->GetTexHeight()*4];
	vector<float> list;
	int idx = 0;
	for(int i = 0; i < _octave_num; i++)
	{
		for(int j = 0; j < param._dog_level_num; j++, idx++)
		{
			tex = GetBaseLevel(_octave_min + i, DATA_KEYPOINT) + j + 2;
			tex->BindTex();
			glGetTexImage(GlobalUtil::_texTarget, 0, GL_RGBA, GL_FLOAT, mem);
			//tex->AttachToFBO(0);
			//tex->FitTexViewPort();
			//glReadPixels(0, 0, tex->GetTexWidth(), tex->GetTexHeight(), GL_RED, GL_FLOAT, mem);
			//
			//make a list of 
			list.resize(0);
			float *pl = mem;
			int fcount = 0 ;
			for(int k = 0; k < tex->GetDrawHeight(); k++)
			{
				float * p = pl; 
				pl += tex->GetTexWidth() * 4;
				for( int m = 0; m < tex->GetDrawWidth(); m ++, p+=4)
				{
				//	if(m ==0 || k ==0 || k == tex->GetDrawHeight() -1 || m == tex->GetDrawWidth() -1) continue;
				//	if(*p == 0) continue;
					int t = ((int) fabs(p[0])) - 1;
					if(t < 0) continue;
					int xx = m + m + ( (t %2)? 1 : 0);
					int yy = k + k + ( (t <2)? 0 : 1);
					if(xx ==0 || yy == 0) continue;
					if(xx >= tex->GetImgWidth() - 1 || yy >= tex->GetImgHeight() - 1)continue;
					list.push_back(xx + 0.5f + p[1]);
					list.push_back(yy + 0.5f + p[2]);
					list.push_back(GlobalUtil::_KeepExtremumSign && p[0] < 0 ? -1.0f : 1.0f);
					list.push_back(p[3]);
					fcount ++;
				}
			}
			if(fcount==0)continue;

			if(GlobalUtil::_timingL) std::cout<<fcount<<".";
			
			GLTexImage * ftex = _featureTex+idx;
			_levelFeatureNum[idx] = (fcount);
			SetLevelFeatureNum(idx, fcount);

			_featureNum += (fcount);


			int fw = ftex->GetImgWidth();
			int fh = ftex->GetImgHeight();

			list.resize(4*fh*fw);

			ftex->BindTex();
			ftex->AttachToFBO(0);
	//		glTexImage2D(GlobalUtil::_texTarget, 0, GlobalUtil::_iTexFormat, fw, fh, 0, GL_BGRA, GL_FLOAT, &list[0]);
			glTexSubImage2D(GlobalUtil::_texTarget, 0, 0, 0, fw, fh, GL_RGBA, GL_FLOAT, &list[0]);
			//
		}
	}
	GLTexImage::UnbindTex();
	delete[] mem;
	if(GlobalUtil::_verbose)
	{
		std::cout<<"#Features:\t"<<_featureNum<<"\n";
	}
}



void PyramidPacked::GetFeatureOrientations()
{
	GLTexImage * gtex, * otex;
	GLTexImage * ftex = _featureTex;
	GLTexImage * fotex = _orientationTex; 
	int * count	 = _levelFeatureNum;
	float sigma, sigma_step = powf(2.0f, 1.0f/param._dog_level_num);


	FrameBufferObject fbo;
	if(_orientationTex)
	{
		GLenum buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT };
		glDrawBuffers(2, buffers);
	}else
	{
		glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	}

	for(int i = 0; i < _octave_num; i++)
	{
		gtex = GetBaseLevel(i+_octave_min, DATA_GRAD) + 1;
		otex = GetBaseLevel(i+_octave_min, DATA_ROT) + 1;

		for(int j = 0; j < param._dog_level_num; j++, ftex++, otex++, count++, gtex++, fotex++)
		{
			if(*count<=0)continue;

			sigma = param.GetLevelSigma(j+param._level_min+1);


			ftex->FitTexViewPort();

			glActiveTexture(GL_TEXTURE0);
			ftex->BindTex();
			glActiveTexture(GL_TEXTURE1);
			gtex->BindTex();
			glActiveTexture(GL_TEXTURE2);
			otex->BindTex();
			//
			ftex->AttachToFBO(0);
			if(_orientationTex)		fotex->AttachToFBO(1);

			GlobalUtil::CheckFramebufferStatus();

			ShaderMan::UseShaderOrientation(gtex->GetTexID(),
				gtex->GetImgWidth(), gtex->GetImgHeight(),
				sigma, otex->GetTexID(), sigma_step, _existing_keypoints);
			ftex->DrawQuad();
		}
	}

	GLTexImage::UnbindMultiTex(3);
	if(GlobalUtil::_timingS)glFinish();
	if(_orientationTex)	fbo.UnattachTex(GL_COLOR_ATTACHMENT1_EXT);

}


void PyramidPacked::GetSimplifiedOrientation()
{
	//
	int idx = 0;
//	int n = _octave_num  * param._dog_level_num;
	float sigma, sigma_step = powf(2.0f, 1.0f/param._dog_level_num); 
	GLTexImage * ftex = _featureTex;

	FrameBufferObject fbo;
	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	for(int i = 0; i < _octave_num; i++)
	{
		GLTexImage *otex = GetBaseLevel(i + _octave_min, DATA_ROT) + 2;
		for(int j = 0; j < param._dog_level_num; j++, ftex++,  otex++, idx ++)
		{
			if(_levelFeatureNum[idx]<=0)continue;
			sigma = param.GetLevelSigma(j+param._level_min+1);
			//
			ftex->AttachToFBO(0);
			ftex->FitTexViewPort();

			glActiveTexture(GL_TEXTURE0);
			ftex->BindTex();
			glActiveTexture(GL_TEXTURE1);
			otex->BindTex();

			ShaderMan::UseShaderSimpleOrientation(otex->GetTexID(), sigma, sigma_step);
			ftex->DrawQuad();
		}
	}
	GLTexImage::UnbindMultiTex(2);
}

void PyramidPacked::InitPyramid(int w, int h, int ds)
{
	int wp, hp, toobig = 0;
	if(ds == 0)
	{
		_down_sample_factor = 0;
		if(GlobalUtil::_octave_min_default>=0)
		{
			wp = w >> GlobalUtil::_octave_min_default;
			hp = h >> GlobalUtil::_octave_min_default;
		}else 
		{
			wp = w << (-GlobalUtil::_octave_min_default);
			hp = h << (-GlobalUtil::_octave_min_default);
		}
		_octave_min = _octave_min_default;
	}else
	{
		//must use 0 as _octave_min; 
		_octave_min = 0;
		_down_sample_factor = ds;
		w >>= ds;
		h >>= ds;
		wp = w;
		hp = h; 
	}

	while(wp > GlobalUtil::_texMaxDim  || hp > GlobalUtil::_texMaxDim )
	{
		_octave_min ++;
		wp >>= 1;
		hp >>= 1;
		toobig = 1;
	}

	while(GlobalUtil::_MemCapGPU > 0 && GlobalUtil::_FitMemoryCap && (wp >_pyramid_width || hp > _pyramid_height) &&
		max(max(wp, hp), max(_pyramid_width, _pyramid_height)) >  1024 * sqrt(GlobalUtil::_MemCapGPU / 96.0) )
	{
		_octave_min ++;
		wp >>= 1;
		hp >>= 1;
		toobig = 2;
	}

	if(toobig && GlobalUtil::_verbose)
	{
		std::cout<<(toobig == 2 ? "[**SKIP OCTAVES**]:\tExceeding Memory Cap (-nomc)\n" :
					"[**SKIP OCTAVES**]:\tReaching the dimension limit (-maxd)!\n");
	}

	if( wp == _pyramid_width && hp == _pyramid_height && _allocated )
	{
		FitPyramid(wp, hp);
	}else if(GlobalUtil::_ForceTightPyramid || _allocated ==0)
	{
		ResizePyramid(wp, hp);
	}
	else if( wp > _pyramid_width || hp > _pyramid_height )
	{
		ResizePyramid(max(wp, _pyramid_width), max(hp, _pyramid_height));
		if(wp < _pyramid_width || hp < _pyramid_height)  FitPyramid(wp, hp);
	}
	else
	{
		//try use the pyramid allocated for large image on small input images
		FitPyramid(wp, hp);
	}

	//select the initial smoothing filter according to the new _octave_min
	ShaderMan::SelectInitialSmoothingFilter(_octave_min + _down_sample_factor, param);
}



void PyramidPacked::FitPyramid(int w, int h)
{
	//(w, h) <= (_pyramid_width, _pyramid_height);

	_pyramid_octave_first = 0;
	//
	_octave_num  = GlobalUtil::_octave_num_default;

	int _octave_num_max = GetRequiredOctaveNum(min(w, h));

	if(_octave_num < 1 || _octave_num > _octave_num_max) 
	{
		_octave_num = _octave_num_max;
	}


	int pw = _pyramid_width>>1, ph = _pyramid_height>>1;
	while(_pyramid_octave_first + _octave_num < _pyramid_octave_num &&  
		pw >= w && ph >= h)
	{
		_pyramid_octave_first++;
		pw >>= 1;
		ph >>= 1;
	}

	for(int i = 0; i < _octave_num; i++)
	{
		GLTexImage * tex = GetBaseLevel(i + _octave_min);
		GLTexImage * dog = GetBaseLevel(i + _octave_min, DATA_DOG);
		GLTexImage * grd = GetBaseLevel(i + _octave_min, DATA_GRAD);
		GLTexImage * rot = GetBaseLevel(i + _octave_min, DATA_ROT);
		GLTexImage * key = GetBaseLevel(i + _octave_min, DATA_KEYPOINT);
		for(int j = param._level_min; j <= param._level_max; j++, tex++, dog++, grd++, rot++, key++)
		{
			tex->SetImageSize(w, h);
			if(j == param._level_min) continue;
			dog->SetImageSize(w, h);
			grd->SetImageSize(w, h);
			rot->SetImageSize(w, h);
			if(j == param._level_min + 1 || j == param._level_max) continue;
			key->SetImageSize(w, h);
		}
		w>>=1;
		h>>=1;
	}
}


void PyramidPacked::ResizePyramid( int w,  int h)
{
	//
	unsigned int totalkb = 0;
	int _octave_num_new, input_sz, i, j;
	//

	if(_pyramid_width == w && _pyramid_height == h && _allocated) return;

	if(w > GlobalUtil::_texMaxDim || h > GlobalUtil::_texMaxDim) return ;

	if(GlobalUtil::_verbose && GlobalUtil::_timingS) std::cout<<"[Allocate Pyramid]:\t" <<w<<"x"<<h<<endl;
	//first octave does not change
	_pyramid_octave_first = 0;

	
	//compute # of octaves

	input_sz = min(w,h) ;
	_pyramid_width =  w;
	_pyramid_height =  h;

	//reset to preset parameters
	_octave_num_new  = GlobalUtil::_octave_num_default;

	if(_octave_num_new < 1) _octave_num_new = GetRequiredOctaveNum(input_sz) ;


	if(_pyramid_octave_num != _octave_num_new)
	{
		//destroy the original pyramid if the # of octave changes
		if(_octave_num >0) 
		{
			DestroyPerLevelData();
			DestroyPyramidData();
		}
		_pyramid_octave_num = _octave_num_new;
	}

	_octave_num = _pyramid_octave_num;

	int noct = _octave_num;
	int nlev = param._level_num;

	//	//initialize the pyramid
	if(_allPyramid==NULL)	_allPyramid = new GLTexPacked[ noct* nlev * DATA_NUM];


	GLTexPacked * gus = (GLTexPacked *) GetBaseLevel(_octave_min, DATA_GAUSSIAN);
	GLTexPacked * dog = (GLTexPacked *) GetBaseLevel(_octave_min, DATA_DOG);
	GLTexPacked * grd = (GLTexPacked *) GetBaseLevel(_octave_min, DATA_GRAD);
	GLTexPacked * rot = (GLTexPacked *) GetBaseLevel(_octave_min, DATA_ROT);
	GLTexPacked * key = (GLTexPacked *) GetBaseLevel(_octave_min, DATA_KEYPOINT);


	////////////there could be "out of memory" happening during the allocation

	for(i = 0; i< noct; i++)
	{
		for( j = 0; j< nlev; j++, gus++, dog++, grd++, rot++, key++)
		{
			gus->InitTexture(w, h);
			if(j==0)continue;
			dog->InitTexture(w, h);
			grd->InitTexture(w, h, 0);
			rot->InitTexture(w, h);
			if(j<=1 || j >=nlev -1) continue;
			key->InitTexture(w, h, 0);
		}
		int tsz = (gus -1)->GetTexPixelCount() * 16;
		totalkb += ((nlev *5 -6)* tsz / 1024);
		//several auxilary textures are not actually required
		w>>=1;
		h>>=1;
	}

	totalkb += ResizeFeatureStorage();

	_allocated = 1;

	if(GlobalUtil::_verbose && GlobalUtil::_timingS) std::cout<<"[Allocate Pyramid]:\t" <<(totalkb/1024)<<"MB\n";

}

void PyramidPacked::DestroyPyramidData()
{
	if(_allPyramid)
	{
		delete [] _allPyramid;
		_allPyramid = NULL;
	}
}


GLTexImage*  PyramidPacked::GetLevelTexture(int octave, int level, int dataName)
{
	return _allPyramid+ (_pyramid_octave_first + octave - _octave_min) * param._level_num 
		+ param._level_num * _pyramid_octave_num * dataName
		+ (level - param._level_min);

}

GLTexImage*  PyramidPacked::GetLevelTexture(int octave, int level)
{
	return _allPyramid+ (_pyramid_octave_first + octave - _octave_min) * param._level_num 
		+ (level - param._level_min);
}

//in the packed implementation( still in progress)
// DATA_GAUSSIAN, DATA_DOG, DATA_GAD will be stored in different textures.

GLTexImage*  PyramidPacked::GetBaseLevel(int octave, int dataName)
{
	if(octave <_octave_min || octave > _octave_min + _octave_num) return NULL;
	int offset = (_pyramid_octave_first + octave - _octave_min) * param._level_num;
	int num = param._level_num * _pyramid_octave_num;
	return _allPyramid + num *dataName + offset;
}


void PyramidPacked::FitHistogramPyramid()
{
	GLTexImage * tex, *htex;
	int hist_level_num = _hpLevelNum - _pyramid_octave_first; 

	tex = GetBaseLevel(_octave_min , DATA_KEYPOINT) + 2;
	htex = _histoPyramidTex + hist_level_num - 1;
	int w = (tex->GetImgWidth() + 2) >> 2;
	int h = (tex->GetImgHeight() + 2)>> 2;


	//4n+1 -> n; 4n+2,2, 3 -> n+1
	for(int k = 0; k <hist_level_num -1; k++, htex--)
	{
		if(htex->GetImgHeight()!= h || htex->GetImgWidth() != w)
		{	
			htex->SetImageSize(w, h);
			htex->ZeroHistoMargin();
		}

		w = (w + 1)>>1; h = (h + 1) >> 1;
	}
}

