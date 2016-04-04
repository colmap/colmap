////////////////////////////////////////////////////////////////////////////
//	File:		SiftMatch.h
//	Author:		Changchang Wu
//	Description :	interface for the SiftMatchGL
////
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


#ifndef GPU_SIFT_MATCH_H
#define GPU_SIFT_MATCH_H
class GLTexImage;
class ProgramGPU;

class SiftMatchGL:public SiftMatchGPU
{
	typedef GLint ParameterGL;
private:
	//tex storage
	GLTexImage _texLoc[2];
	GLTexImage _texDes[2];
	GLTexImage _texDot;
	GLTexImage _texMatch[2];

	//programs
	ProgramGPU * s_multiply;
	ProgramGPU * s_guided_mult;
	ProgramGPU * s_col_max;
	ProgramGPU * s_row_max;

	//matching parameters
	ParameterGL _param_multiply_tex1;
	ParameterGL _param_multiply_tex2;
	ParameterGL _param_multiply_size;
	ParameterGL _param_rowmax_param;
	ParameterGL _param_colmax_param;

	///guided matching
	ParameterGL _param_guided_mult_tex1;
	ParameterGL _param_guided_mult_tex2;
	ParameterGL _param_guided_mult_texl1;
	ParameterGL _param_guided_mult_texl2;
	ParameterGL _param_guided_mult_h;
	ParameterGL _param_guided_mult_f;
	ParameterGL _param_guided_mult_param;
	//
	int _max_sift; 
	int _num_sift[2];
	int _id_sift[2];
	int _have_loc[2];

	//gpu parameter
	int _sift_per_stripe;
	int _sift_num_stripe;
	int	_sift_per_row;
	int	_pixel_per_sift;
	int _initialized;
	//
	vector<float> sift_buffer; 
private:
	void AllocateSiftMatch();
	void LoadSiftMatchShadersGLSL();
	int  GetBestMatch(int max_match, int match_buffer[][2], float distmax, float ratiomax, int mbm);
public:
	SiftMatchGL(int max_sift, int use_glsl);
	virtual ~SiftMatchGL();
public:
	void InitSiftMatch();
	void SetMaxSift(int max_sift);
	void SetDescriptors(int index, int num, const unsigned char * descriptor, int id = -1);
	void SetDescriptors(int index, int num, const float * descriptor, int id = -1);
	void SetFeautreLocation(int index, const float* locatoins, int gap);
	int  GetSiftMatch(int max_match, int match_buffer[][2], float distmax, float ratiomax, int mbm);
	int  GetGuidedSiftMatch(int max_match, int match_buffer[][2], float H[3][3],  float F[3][3], 
		float distmax, float ratiomax, float hdistmax,float fdistmax, int mbm);
};


#endif

