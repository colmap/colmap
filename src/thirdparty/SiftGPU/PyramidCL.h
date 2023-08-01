////////////////////////////////////////////////////////////////////////////
//	File:		PyramidCL.h
//	Author:		Changchang Wu
//	Description : interface for the PyramdCL
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



#ifndef _PYRAMID_CL_H
#define _PYRAMID_CL_H
#if defined(CL_SIFTGPU_ENABLED)

class CLTexImage;
class SiftPyramid;
class ProgramBagCL;
class PyramidCL: public SiftPyramid
{
	CLTexImage* 	_inputTex;
	CLTexImage* 	_allPyramid;
	CLTexImage* 	_histoPyramidTex;
	CLTexImage* 	_featureTex;
	CLTexImage* 	_descriptorTex;
	CLTexImage* 	_orientationTex;
    ProgramBagCL*   _OpenCL;
    GLTexImage*     _bufferTEX;
public:
	virtual void GetFeatureDescriptors();
	virtual void GenerateFeatureListTex();
	virtual void ReshapeFeatureListCPU();
	virtual void GenerateFeatureDisplayVBO();
	virtual void DestroySharedData();
	virtual void DestroyPerLevelData();
	virtual void DestroyPyramidData();
	virtual void DownloadKeypoints();
	virtual void GenerateFeatureListCPU();
	virtual void GenerateFeatureList();
	virtual GLTexImage* GetLevelTexture(int octave, int level);
	virtual GLTexImage* GetLevelTexture(int octave, int level, int dataName);
	virtual void BuildPyramid(GLTexInput * input);
	virtual void DetectKeypointsEX();
	virtual void ComputeGradient();
	virtual void GetFeatureOrientations();
	virtual void GetSimplifiedOrientation();
	virtual void InitPyramid(int w, int h, int ds = 0);
	virtual void ResizePyramid(int w, int h);
	
	//////////
	void CopyGradientTex();
	void FitPyramid(int w, int h);

    void InitializeContext();
	int ResizeFeatureStorage();
	int FitHistogramPyramid(CLTexImage* tex);
	void SetLevelFeatureNum(int idx, int fcount);
	void ConvertInputToCL(GLTexInput* input, CLTexImage* output);
	GLTexImage* ConvertTexCL2GL(CLTexImage* tex, int dataName);
	CLTexImage* GetBaseLevel(int octave, int dataName = DATA_GAUSSIAN);
private:
	void GenerateFeatureList(int i, int j, int reduction_count, vector<int>& hbuffer);
public:
	PyramidCL(SiftParam& sp);
	virtual ~PyramidCL();
};


#endif
#endif

