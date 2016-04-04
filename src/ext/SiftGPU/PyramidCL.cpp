////////////////////////////////////////////////////////////////////////////
//	File:		PyramidCL.cpp
//	Author:		Changchang Wu
//	Description : implementation of the PyramidCL class.
//				OpenCL-based implementation of SiftPyramid
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


#include "GL/glew.h"
#include <CL/OpenCL.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
using namespace std;

#include "GlobalUtil.h"
#include "GLTexImage.h"
#include "CLTexImage.h" 
#include "SiftGPU.h"
#include "SiftPyramid.h"
#include "ProgramCL.h"
#include "PyramidCL.h"


#define USE_TIMING()		double t, t0, tt;
#define OCTAVE_START()		if(GlobalUtil::_timingO){	t = t0 = CLOCK();	cout<<"#"<<i+_down_sample_factor<<"\t";	}
#define LEVEL_FINISH()		if(GlobalUtil::_timingL){	_OpenCL->FinishCL();	tt = CLOCK();cout<<(tt-t)<<"\t";	t = CLOCK();}
#define OCTAVE_FINISH()		if(GlobalUtil::_timingO)cout<<"|\t"<<(CLOCK()-t0)<<endl;


PyramidCL::PyramidCL(SiftParam& sp) : SiftPyramid(sp)
{
	_allPyramid = NULL;
	_histoPyramidTex = NULL;
	_featureTex = NULL;
	_descriptorTex = NULL;
	_orientationTex = NULL;
    _bufferTEX = NULL;
    if(GlobalUtil::_usePackedTex)    _OpenCL = new ProgramBagCL();
    else                             _OpenCL = new ProgramBagCLN();
    _OpenCL->InitProgramBag(sp);
	_inputTex = new CLTexImage( _OpenCL->GetContextCL(),
                                _OpenCL->GetCommandQueue());
    /////////////////////////
    InitializeContext();
}

PyramidCL::~PyramidCL()
{
	DestroyPerLevelData();
	DestroySharedData();
	DestroyPyramidData();
    if(_OpenCL)   delete _OpenCL;
	if(_inputTex) delete _inputTex;
    if(_bufferTEX) delete _bufferTEX;
}

void PyramidCL::InitializeContext()
{
    GlobalUtil::InitGLParam(1);
}

void PyramidCL::InitPyramid(int w, int h, int ds)
{
	int wp, hp, toobig = 0;
	if(ds == 0)
	{
		_down_sample_factor = 0;
		if(GlobalUtil::_octave_min_default>=0)
		{
			wp = w >> _octave_min_default;
			hp = h >> _octave_min_default;
		}else
		{
			//can't upsample by more than 8
			_octave_min_default = max(-3, _octave_min_default);
			//
			wp = w << (-_octave_min_default);
			hp = h << (-_octave_min_default);
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
	if(toobig && GlobalUtil::_verbose && _octave_min > 0)
	{
		std::cout<< "**************************************************************\n"
					"Image larger than allowed dimension, data will be downsampled!\n"
					"use -maxd to change the settings\n"
					"***************************************************************\n";
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

    _OpenCL->SelectInitialSmoothingFilter(_octave_min + _down_sample_factor, param);
}

void PyramidCL::ResizePyramid(int w, int h)
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
    int texNum = noct* nlev * DATA_NUM;

	//	//initialize the pyramid
	if(_allPyramid==NULL)
    {
        _allPyramid = new CLTexImage[ texNum];
        cl_context       context  = _OpenCL->GetContextCL();
        cl_command_queue queue    = _OpenCL->GetCommandQueue();
        for(i = 0; i < texNum; ++i)   _allPyramid[i].SetContext(context, queue);
    }



	CLTexImage * gus =  GetBaseLevel(_octave_min, DATA_GAUSSIAN);
	CLTexImage * dog =  GetBaseLevel(_octave_min, DATA_DOG);
	CLTexImage * grd =  GetBaseLevel(_octave_min, DATA_GRAD);
    CLTexImage * rot =  GetBaseLevel(_octave_min, DATA_ROT);
	CLTexImage * key =  GetBaseLevel(_octave_min, DATA_KEYPOINT);

	////////////there could be "out of memory" happening during the allocation



	for(i = 0; i< noct; i++)
	{
		for( j = 0; j< nlev; j++, gus++, dog++, grd++, rot++, key++)
		{
            gus->InitPackedTex(w, h, GlobalUtil::_usePackedTex);
			if(j==0)continue;
			dog->InitPackedTex(w, h, GlobalUtil::_usePackedTex);
            if(j < 1 + param._dog_level_num)
			{
			    grd->InitPackedTex(w, h, GlobalUtil::_usePackedTex);
			    rot->InitPackedTex(w, h, GlobalUtil::_usePackedTex);
            }
			if(j > 1 && j < nlev -1) key->InitPackedTex(w, h, GlobalUtil::_usePackedTex);
		}
        ////////////////////////////////////////
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

void PyramidCL::FitPyramid(int w, int h)
{
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

	//////////////////
	for(int i = 0; i < _octave_num; i++)
	{
		CLTexImage * tex = GetBaseLevel(i + _octave_min);
		CLTexImage * dog = GetBaseLevel(i + _octave_min, DATA_DOG);
		CLTexImage * grd = GetBaseLevel(i + _octave_min, DATA_GRAD);
		CLTexImage * rot = GetBaseLevel(i + _octave_min, DATA_ROT);
		CLTexImage * key = GetBaseLevel(i + _octave_min, DATA_KEYPOINT);
		for(int j = param._level_min; j <= param._level_max; j++, tex++, dog++, grd++, rot++, key++)
		{
			tex->SetPackedSize(w, h, GlobalUtil::_usePackedTex);
			if(j == param._level_min) continue;
			dog->SetPackedSize(w, h, GlobalUtil::_usePackedTex);
            if(j < param._level_max - 1)
            {
			    grd->SetPackedSize(w, h, GlobalUtil::_usePackedTex);
			    rot->SetPackedSize(w, h, GlobalUtil::_usePackedTex);
            }
			if(j > param._level_min + 1 &&  j < param._level_max) key->SetPackedSize(w, h, GlobalUtil::_usePackedTex);
		}
		w>>=1;
		h>>=1;
	}
}


void PyramidCL::SetLevelFeatureNum(int idx, int fcount)
{
    _featureTex[idx].InitBufferTex(fcount, 1, 4);
	_levelFeatureNum[idx] = fcount;
}

int PyramidCL::ResizeFeatureStorage()
{
	int totalkb = 0;
	if(_levelFeatureNum==NULL)	_levelFeatureNum = new int[_octave_num * param._dog_level_num];
	std::fill(_levelFeatureNum, _levelFeatureNum+_octave_num * param._dog_level_num, 0); 

    cl_context       context  = _OpenCL->GetContextCL();
    cl_command_queue queue    = _OpenCL->GetCommandQueue();
	int wmax = GetBaseLevel(_octave_min)->GetImgWidth() * 2;
	int hmax = GetBaseLevel(_octave_min)->GetImgHeight() * 2;
	int whmax = max(wmax, hmax);
	int w,  i;

	//
	int num = (int)ceil(log(double(whmax))/log(4.0));

	if( _hpLevelNum != num)
	{
		_hpLevelNum = num;
		if(_histoPyramidTex ) delete [] _histoPyramidTex;
		_histoPyramidTex = new CLTexImage[_hpLevelNum];
        for(i = 0; i < _hpLevelNum; ++i) _histoPyramidTex[i].SetContext(context, queue);
	}

	for(i = 0, w = 1; i < _hpLevelNum; i++)
	{
        _histoPyramidTex[i].InitBufferTex(w, whmax, 4);
		w<<=2;
	}

	// (4 ^ (_hpLevelNum) -1 / 3) pixels
	totalkb += (((1 << (2 * _hpLevelNum)) -1) / 3 * 16 / 1024);

	//initialize the feature texture
	int idx = 0, n = _octave_num * param._dog_level_num;
	if(_featureTex==NULL)	
    {
        _featureTex = new CLTexImage[n];
        for(i = 0; i <n; ++i) _featureTex[i].SetContext(context, queue);
    }
	if(GlobalUtil::_MaxOrientation >1 && GlobalUtil::_OrientationPack2==0 && _orientationTex== NULL)	
    {
        _orientationTex = new CLTexImage[n];
        for(i = 0; i < n; ++i) _orientationTex[i].SetContext(context, queue);
    }


	for(i = 0; i < _octave_num; i++)
	{
		CLTexImage * tex = GetBaseLevel(i+_octave_min);
		int fmax = int(4 * tex->GetTexWidth() * tex->GetTexHeight()*GlobalUtil::_MaxFeaturePercent);
		//
		if(fmax > GlobalUtil::_MaxLevelFeatureNum) fmax = GlobalUtil::_MaxLevelFeatureNum;
		else if(fmax < 32) fmax = 32;	//give it at least a space of 32 feature

		for(int j = 0; j < param._dog_level_num; j++, idx++)
		{
			_featureTex[idx].InitBufferTex(fmax, 1, 4);
			totalkb += fmax * 16 /1024;
			//
			if(GlobalUtil::_MaxOrientation>1 && GlobalUtil::_OrientationPack2 == 0)
			{
				_orientationTex[idx].InitBufferTex(fmax, 1, 4);
				totalkb += fmax * 16 /1024;
			}
		}
	}

	//this just need be initialized once
	if(_descriptorTex==NULL)
	{
		//initialize feature texture pyramid
		int fmax = _featureTex->GetImgWidth();
		_descriptorTex = new CLTexImage(context, queue);
		totalkb += ( fmax /2);
		_descriptorTex->InitBufferTex(fmax *128, 1, 1);
	}else
	{
		totalkb +=  _descriptorTex->GetDataSize()/1024;
	}
	return totalkb;
}

void PyramidCL::GetFeatureDescriptors() 
{
	//descriptors...
	/*float* pd =  &_descriptor_buffer[0];
	vector<float> descriptor_buffer2;

	//use another buffer if we need to re-order the descriptors
	if(_keypoint_index.size() > 0)
	{
		descriptor_buffer2.resize(_descriptor_buffer.size());
		pd = &descriptor_buffer2[0];
	}

	CLTexImage * got, * ftex= _featureTex;
	for(int i = 0, idx = 0; i < _octave_num; i++)
	{
		got = GetBaseLevel(i + _octave_min, DATA_GRAD) + 1;
		for(int j = 0; j < param._dog_level_num; j++, ftex++, idx++, got++)
		{
			if(_levelFeatureNum[idx]==0) continue;
			ProgramCL::ComputeDescriptor(ftex, got, _descriptorTex);//process
			_descriptorTex->CopyToHost(pd); //readback descriptor
			pd += 128*_levelFeatureNum[idx];
		}
	}

	if(GlobalUtil::_timingS) _OpenCL->FinishCL();

	if(_keypoint_index.size() > 0)
	{
	    //put the descriptor back to the original order for keypoint list.
		for(int i = 0; i < _featureNum; ++i)
		{
			int index = _keypoint_index[i];
			memcpy(&_descriptor_buffer[index*128], &descriptor_buffer2[i*128], 128 * sizeof(float));
		}
	}*/ 
}

void PyramidCL::GenerateFeatureListTex() 
{

	vector<float> list;
	int idx = 0;
	const double twopi = 2.0*3.14159265358979323846;
	float sigma_half_step = powf(2.0f, 0.5f / param._dog_level_num);
	float octave_sigma = _octave_min>=0? float(1<<_octave_min): 1.0f/(1<<(-_octave_min));
	float offset = GlobalUtil::_LoweOrigin? 0 : 0.5f; 
	if(_down_sample_factor>0) octave_sigma *= float(1<<_down_sample_factor); 

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
				if(   (key[2] >= sigma_min && key[2] < sigma_max)
					||(key[2] < sigma_min && i ==0 && j == 0)
					||(key[2] > sigma_max && i == _octave_num -1 && j == param._dog_level_num - 1))
				{
					//add this keypoint to the list
					list.push_back((key[0] - offset) / octave_sigma + 0.5f);
					list.push_back((key[1] - offset) / octave_sigma + 0.5f);
					list.push_back(key[2] / octave_sigma);
					list.push_back((float)fmod(twopi-key[3], twopi));
					fcount ++;
					//save the index of keypoints
					_keypoint_index.push_back(k);
				}

			}

			_levelFeatureNum[idx] = fcount;
			if(fcount==0)continue;
			CLTexImage * ftex = _featureTex+idx;

			SetLevelFeatureNum(idx, fcount);
			ftex->CopyFromHost(&list[0]);
		}
	}

	if(GlobalUtil::_verbose)
	{
		std::cout<<"#Features:\t"<<_featureNum<<"\n";
	}

}

void PyramidCL::ReshapeFeatureListCPU() 
{
	int i, szmax =0, sz;
	int n = param._dog_level_num*_octave_num;
	for( i = 0; i < n; i++) 
	{
		sz = _levelFeatureNum[i];
		if(sz > szmax ) szmax = sz;
	}
	float * buffer = new float[szmax*16];
	float * buffer1 = buffer;
	float * buffer2 = buffer + szmax*4;



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

		_featureTex[i].CopyToHost(buffer1);
		
		int fcount =0;
		float * src = buffer1;
		float * des = buffer2;
		const static double factor  = 2.0*3.14159265358979323846/65535.0;
		for(int j = 0; j < _levelFeatureNum[i]; j++, src+=4)
		{
			unsigned short * orientations = (unsigned short*) (&src[3]);
			if(orientations[0] != 65535)
			{
				des[0] = src[0];
				des[1] = src[1];
				des[2] = src[2];
				des[3] = float( factor* orientations[0]);
				fcount++;
				des += 4;
				if(orientations[1] != 65535 && orientations[1] != orientations[0])
				{
					des[0] = src[0];
					des[1] = src[1];
					des[2] = src[2];
					des[3] = float(factor* orientations[1]);	
					fcount++;
					des += 4;
				}
			}
		}
		//texture size
		SetLevelFeatureNum(i, fcount);
		_featureTex[i].CopyFromHost(buffer2);
		
		if(fcount == 0) continue;

#ifdef NO_DUPLICATE_DOWNLOAD
		float oss = os * (1 << (i / param._dog_level_num));
		_keypoint_buffer.resize((_featureNum + fcount) * 4);
		float* ds = &_keypoint_buffer[_featureNum * 4];
		float* fs = buffer2;
		for(int k = 0;  k < fcount; k++, ds+=4, fs+=4)
		{
			ds[0] = oss*(fs[0]-0.5f) + offset;	//x
			ds[1] = oss*(fs[1]-0.5f) + offset;	//y
			ds[2] = oss*fs[2];  //scale
			ds[3] = (float)fmod(twopi-fs[3], twopi);	//orientation, mirrored
		}
#endif
		_featureNum += fcount;
	}
	delete[] buffer;
	if(GlobalUtil::_verbose)
	{
		std::cout<<"#Features MO:\t"<<_featureNum<<endl;
	}
}

void PyramidCL::GenerateFeatureDisplayVBO() 
{
	//it is weried that this part is very slow.
	//use a big VBO to save all the SIFT box vertices
	/*int nvbo = _octave_num * param._dog_level_num;
	if(_featureDisplayVBO==NULL)
	{
		//initialize the vbos
		_featureDisplayVBO = new GLuint[nvbo];
		_featurePointVBO = new GLuint[nvbo];
		glGenBuffers(nvbo, _featureDisplayVBO);	
		glGenBuffers(nvbo, _featurePointVBO);
	}
	for(int i = 0; i < nvbo; i++)
	{
		if(_levelFeatureNum[i]<=0)continue;
		CLTexImage * ftex  = _featureTex + i;
		CLTexImage texPBO1( _levelFeatureNum[i]* 10, 1, 4, _featureDisplayVBO[i]);
		CLTexImage texPBO2(_levelFeatureNum[i], 1, 4, _featurePointVBO[i]);
		_OpenCL->DisplayKeyBox(ftex, &texPBO1);
		_OpenCL->DisplayKeyPoint(ftex, &texPBO2);	
	}*/
}

void PyramidCL::DestroySharedData() 
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
		delete _descriptorTex;
		_descriptorTex = NULL;
	}
	//cpu reduction buffer.
	if(_histo_buffer)
	{
		delete[] _histo_buffer;
		_histo_buffer = 0;
	}
}

void PyramidCL::DestroyPerLevelData() 
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

void PyramidCL::DestroyPyramidData()
{
	if(_allPyramid)
	{
		delete [] _allPyramid;
		_allPyramid = NULL;
	}
}

void PyramidCL::DownloadKeypoints() 
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
	float * p = buffer, *ps;
	CLTexImage * ftex = _featureTex;
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
				ftex->CopyToHost(ps = p);
				for(int k = 0;  k < _levelFeatureNum[idx]; k++, ps+=4)
				{
					ps[0] = os*(ps[0]-0.5f) + offset;	//x
					ps[1] = os*(ps[1]-0.5f) + offset;	//y
					ps[2] = os*ps[2]; 
					ps[3] = (float)fmod(twopi-ps[3], twopi);	//orientation, mirrored
				}
				p+= 4* _levelFeatureNum[idx];
			}
		}
	}

	//put the feature into their original order for existing keypoint 
	if(_keypoint_index.size() > 0)
	{
		for(int i = 0; i < _featureNum; ++i)
		{
			int index = _keypoint_index[i];
			memcpy(&_keypoint_buffer[index*4], &keypoint_buffer2[i*4], 4 * sizeof(float));
		}
	}
}

void PyramidCL::GenerateFeatureListCPU()
{
	//no cpu version provided
	GenerateFeatureList();
}

void PyramidCL::GenerateFeatureList(int i, int j, int reduction_count, vector<int>& hbuffer)
{
    /*int fcount = 0, idx = i * param._dog_level_num  + j;
	int hist_level_num = _hpLevelNum - _pyramid_octave_first /2; 
	int ii, k, len; 

	CLTexImage * htex, * ftex, * tex, *got;
	ftex = _featureTex + idx;
	htex = _histoPyramidTex + hist_level_num -1;
	tex = GetBaseLevel(_octave_min + i, DATA_KEYPOINT) + 2 + j;
	got = GetBaseLevel(_octave_min + i, DATA_GRAD) + 2 + j;

	_OpenCL->InitHistogram(tex, htex);

	for(k = 0; k < reduction_count - 1; k++, htex--)
	{
		ProgramCL::ReduceHistogram(htex, htex -1);	
	}
	
	//htex has the row reduction result
	len = htex->GetImgHeight() * 4;
	hbuffer.resize(len);
	_OpenCL->FinishCL();
	htex->CopyToHost(&hbuffer[0]);
	//
	for(ii = 0; ii < len; ++ii)		fcount += hbuffer[ii];
	SetLevelFeatureNum(idx, fcount);
	
    //build the feature list
	if(fcount > 0)
	{
		_featureNum += fcount;
		_keypoint_buffer.resize(fcount * 4);
		//vector<int> ikbuf(fcount*4);
		int* ibuf = (int*) (&_keypoint_buffer[0]);

		for(ii = 0; ii < len; ++ii)
		{
			int x = ii%4, y = ii / 4;
			for(int jj = 0 ; jj < hbuffer[ii]; ++jj, ibuf+=4)
			{
				ibuf[0] = x; ibuf[1] = y; ibuf[2] = jj; ibuf[3] = 0;
			}
		}
		_featureTex[idx].CopyFromHost(&_keypoint_buffer[0]);
	
		////////////////////////////////////////////
		ProgramCL::GenerateList(_featureTex + idx, ++htex);
		for(k = 2; k < reduction_count; k++)
		{
			ProgramCL::GenerateList(_featureTex + idx, ++htex);
		}
	}*/
}

void PyramidCL::GenerateFeatureList()
{
	/*double t1, t2; 
	int ocount = 0, reduction_count;
    int reverse = (GlobalUtil::_TruncateMethod == 1);

	vector<int> hbuffer;
	_featureNum = 0;

	//for(int i = 0, idx = 0; i < _octave_num; i++)
    FOR_EACH_OCTAVE(i, reverse)
	{
        CLTexImage* tex = GetBaseLevel(_octave_min + i, DATA_KEYPOINT) + 2;
		reduction_count = FitHistogramPyramid(tex);

		if(GlobalUtil::_timingO)
		{
			t1 = CLOCK(); 
			ocount = 0;
			std::cout<<"#"<<i+_octave_min + _down_sample_factor<<":\t";
		}
		//for(int j = 0; j < param._dog_level_num; j++, idx++)
        FOR_EACH_LEVEL(j, reverse)
		{
            if(GlobalUtil::_TruncateMethod && GlobalUtil::_FeatureCountThreshold > 0 && _featureNum > GlobalUtil::_FeatureCountThreshold) continue;

	        GenerateFeatureList(i, j, reduction_count, hbuffer);

			/////////////////////////////
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
	/////
	CopyGradientTex();
	/////
	if(GlobalUtil::_timingS)_OpenCL->FinishCL();

	if(GlobalUtil::_verbose)
	{
		std::cout<<"#Features:\t"<<_featureNum<<"\n";
	}*/
}

GLTexImage* PyramidCL::GetLevelTexture(int octave, int level)
{
	return GetLevelTexture(octave, level, DATA_GAUSSIAN);
}

GLTexImage* PyramidCL::ConvertTexCL2GL(CLTexImage* tex, int dataName)
{
   
    if(_bufferTEX == NULL) _bufferTEX = new GLTexImage;

    ///////////////////////////////////////////
    int ratio = GlobalUtil::_usePackedTex ? 2 : 1; 
    int width  = tex->GetImgWidth() * ratio;
    int height = tex->GetImgHeight() * ratio; 
    int tw = max(width,  _bufferTEX->GetTexWidth());
    int th = max(height, _bufferTEX->GetTexHeight());
    _bufferTEX->InitTexture(tw, th, 1, GL_RGBA);
    _bufferTEX->SetImageSize(width, height); 

    //////////////////////////////////
    CLTexImage texCL(_OpenCL->GetContextCL(), _OpenCL->GetCommandQueue()); 
    texCL.InitTextureGL(*_bufferTEX, width, height, 4); 

	switch(dataName)
	{
	case DATA_GAUSSIAN: _OpenCL->UnpackImage(tex, &texCL); break;
	case DATA_DOG:_OpenCL->UnpackImageDOG(tex, &texCL); break;
	case DATA_GRAD:_OpenCL->UnpackImageGRD(tex, &texCL); break;
	case DATA_KEYPOINT:_OpenCL->UnpackImageKEY(tex,
        tex - param._level_num * _pyramid_octave_num, &texCL);break;
	default:
			break;
	}


	return _bufferTEX;
}

GLTexImage* PyramidCL::GetLevelTexture(int octave, int level, int dataName) 
{
	CLTexImage* tex = GetBaseLevel(octave, dataName) + (level - param._level_min);
	return ConvertTexCL2GL(tex, dataName);
}

void PyramidCL::ConvertInputToCL(GLTexInput* input, CLTexImage* output)
{
	int ws = input->GetImgWidth(), hs = input->GetImgHeight();
	//copy the input image to pixel buffer object
    if(input->_pixel_data)
    {
        output->InitTexture(ws, hs, 1);
        output->CopyFromHost(input->_pixel_data); 
    }else /*if(input->_rgb_converted && input->CopyToPBO(_bufferPBO, ws, hs, GL_LUMINANCE))
    {
		output->InitTexture(ws, hs, 1);
        output->CopyFromPBO(ws, hs, _bufferPBO); 
    }else if(input->CopyToPBO(_bufferPBO, ws, hs))
	{
		CLTexImage texPBO(ws, hs, 4, _bufferPBO);
		output->InitTexture(ws, hs, 1);
		ProgramCL::ReduceToSingleChannel(output, &texPBO, !input->_rgb_converted);
	}else*/
	{
		std::cerr<< "Unable To Convert Intput\n";
	}
}

void PyramidCL::BuildPyramid(GLTexInput * input)
{

	USE_TIMING();

	int i, j;
	
	for ( i = _octave_min; i < _octave_min + _octave_num; i++)
	{

		CLTexImage *tex = GetBaseLevel(i);
		CLTexImage *buf = GetBaseLevel(i, DATA_DOG) +2;
        FilterCL ** filter  = _OpenCL->f_gaussian_step; 
		j = param._level_min + 1;

		OCTAVE_START();

		if( i == _octave_min )
		{	
            if(GlobalUtil::_usePackedTex)
            {
			    ConvertInputToCL(input, _inputTex);
			    if(i < 0)	_OpenCL->SampleImageU(tex, _inputTex, -i- 1);			
			    else		_OpenCL->SampleImageD(tex, _inputTex, i + 1);
            }else
            {
                if(i == 0) ConvertInputToCL(input, tex);
                else
                {
                    ConvertInputToCL(input, _inputTex);
                    if(i < 0) _OpenCL->SampleImageU(tex, _inputTex, -i);
                    else      _OpenCL->SampleImageD(tex, _inputTex,  i);
                }
            }
            _OpenCL->FilterInitialImage(tex, buf);   
		}else
		{
			_OpenCL->SampleImageD(tex, GetBaseLevel(i - 1) + param._level_ds - param._level_min); 
            _OpenCL->FilterSampledImage(tex, buf);
		}
		LEVEL_FINISH();
		for( ; j <=  param._level_max ; j++, tex++, filter++)
		{
			// filtering
			_OpenCL->FilterImage(*filter, tex + 1, tex, buf);
			LEVEL_FINISH();
		}
		OCTAVE_FINISH();
	}
	if(GlobalUtil::_timingS) _OpenCL->FinishCL();
}

void PyramidCL::DetectKeypointsEX()
{
	int i, j;
	double t0, t, ts, t1, t2;
    
	if(GlobalUtil::_timingS && GlobalUtil::_verbose) ts = CLOCK();

	for(i = _octave_min; i < _octave_min + _octave_num; i++)
	{
		CLTexImage * gus = GetBaseLevel(i) + 1;
		CLTexImage * dog = GetBaseLevel(i, DATA_DOG) + 1;
		CLTexImage * grd = GetBaseLevel(i, DATA_GRAD) + 1;
        CLTexImage * rot = GetBaseLevel(i, DATA_ROT) + 1;
		//compute the gradient
		for(j = param._level_min +1; j <=  param._level_max ; j++, gus++, dog++, grd++, rot++)
		{
			//input: gus and gus -1
			//output: gradient, dog, orientation
			_OpenCL->ComputeDOG(gus, gus - 1, dog, grd, rot);
		}
	}
	if(GlobalUtil::_timingS && GlobalUtil::_verbose)
	{
		_OpenCL->FinishCL();
		t1 = CLOCK();
	}
    //if(GlobalUtil::_timingS) _OpenCL->FinishCL();
    //if(!GlobalUtil::_usePackedTex) return; //not finished
    //return; 

	for ( i = _octave_min; i < _octave_min + _octave_num; i++)
	{
		if(GlobalUtil::_timingO)
		{
			t0 = CLOCK();
			std::cout<<"#"<<(i + _down_sample_factor)<<"\t";
		}
		CLTexImage * dog = GetBaseLevel(i, DATA_DOG) + 2;
		CLTexImage * key = GetBaseLevel(i, DATA_KEYPOINT) +2;


		for( j = param._level_min +2; j <  param._level_max ; j++, dog++, key++)
		{
			if(GlobalUtil::_timingL)t = CLOCK();
			//input, dog, dog + 1, dog -1
			//output, key
			_OpenCL->ComputeKEY(dog, key, param._dog_threshold, param._edge_threshold);
			if(GlobalUtil::_timingL)
			{
				std::cout<<(CLOCK()-t)<<"\t";
			}
		}
		if(GlobalUtil::_timingO)
		{
			std::cout<<"|\t"<<(CLOCK()-t0)<<"\n";
		}
	}

	if(GlobalUtil::_timingS)
	{
		_OpenCL->FinishCL();
		if(GlobalUtil::_verbose) 
		{	
			t2 = CLOCK();
			std::cout	<<"<Gradient, DOG  >\t"<<(t1-ts)<<"\n"
						<<"<Get Keypoints  >\t"<<(t2-t1)<<"\n";
		}				
	}
}

void PyramidCL::CopyGradientTex()
{
	/*double ts, t1;

	if(GlobalUtil::_timingS && GlobalUtil::_verbose)ts = CLOCK();

	for(int i = 0, idx = 0; i < _octave_num; i++)
	{
		CLTexImage * got = GetBaseLevel(i + _octave_min, DATA_GRAD) +  1;
		//compute the gradient
		for(int j = 0; j <  param._dog_level_num ; j++, got++, idx++)
		{
			if(_levelFeatureNum[idx] > 0)	got->CopyToTexture2D();
		}
	}
	if(GlobalUtil::_timingS)
	{
		ProgramCL::FinishCLDA();
		if(GlobalUtil::_verbose)
		{
			t1 = CLOCK();
			std::cout	<<"<Copy Grad/Orientation>\t"<<(t1-ts)<<"\n";
		}
	}*/
}

void PyramidCL::ComputeGradient() 
{

	/*int i, j;
	double ts, t1;

	if(GlobalUtil::_timingS && GlobalUtil::_verbose)ts = CLOCK();

	for(i = _octave_min; i < _octave_min + _octave_num; i++)
	{
		CLTexImage * gus = GetBaseLevel(i) +  1;
		CLTexImage * dog = GetBaseLevel(i, DATA_DOG) +  1;
		CLTexImage * got = GetBaseLevel(i, DATA_GRAD) +  1;

		//compute the gradient
		for(j = 0; j <  param._dog_level_num ; j++, gus++, dog++, got++)
		{
			ProgramCL::ComputeDOG(gus, dog, got);
		}
	}
	if(GlobalUtil::_timingS)
	{
		ProgramCL::FinishCLDA();
		if(GlobalUtil::_verbose)
		{
			t1 = CLOCK();
			std::cout	<<"<Gradient, DOG  >\t"<<(t1-ts)<<"\n";
		}
	}*/
}

int PyramidCL::FitHistogramPyramid(CLTexImage* tex)
{
	CLTexImage *htex;
	int hist_level_num = _hpLevelNum - _pyramid_octave_first / 2; 
	htex = _histoPyramidTex + hist_level_num - 1;
	int w = (tex->GetImgWidth() + 2) >> 2;
	int h = tex->GetImgHeight();
	int count = 0; 
	for(int k = 0; k < hist_level_num; k++, htex--)
	{
		//htex->SetImageSize(w, h);	
		htex->InitTexture(w, h, 4); 
		++count;
		if(w == 1)
            break;
		w = (w + 3)>>2; 
	}
	return count;
}

void PyramidCL::GetFeatureOrientations() 
{

/*
	CLTexImage * ftex = _featureTex;
	int * count	 = _levelFeatureNum;
	float sigma, sigma_step = powf(2.0f, 1.0f/param._dog_level_num);

	for(int i = 0; i < _octave_num; i++)
	{
		CLTexImage* got = GetBaseLevel(i + _octave_min, DATA_GRAD) + 1;
		CLTexImage* key = GetBaseLevel(i + _octave_min, DATA_KEYPOINT) + 2;

		for(int j = 0; j < param._dog_level_num; j++, ftex++, count++, got++, key++)
		{
			if(*count<=0)continue;

			//if(ftex->GetImgWidth() < *count) ftex->InitTexture(*count, 1, 4);

			sigma = param.GetLevelSigma(j+param._level_min+1);

			ProgramCL::ComputeOrientation(ftex, got, key, sigma, sigma_step, _existing_keypoints);		
		}
	}

	if(GlobalUtil::_timingS)ProgramCL::FinishCL();
	*/


}

void PyramidCL::GetSimplifiedOrientation() 
{
	//no simplified orientation
	GetFeatureOrientations();
}

CLTexImage* PyramidCL::GetBaseLevel(int octave, int dataName)
{
	if(octave <_octave_min || octave > _octave_min + _octave_num) return NULL;
	int offset = (_pyramid_octave_first + octave - _octave_min) * param._level_num;
	int num = param._level_num * _pyramid_octave_num;
	return _allPyramid + num * dataName + offset;
}

#endif

