////////////////////////////////////////////////////////////////////////////
//	File:		SiftMatchCU.h
//	Author:		Changchang Wu
//	Description :	interface for the SiftMatchCU
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



#ifndef CU_SIFT_MATCH_H
#define CU_SIFT_MATCH_H
#if defined(SIFTGPU_CUDA_ENABLED)

class CuTexImage;
class SiftMatchCU:public SiftMatchGPU
{
private:
	//tex storage
	CuTexImage _texLoc[2];
	CuTexImage _texDes[2];
	CuTexImage _texDot;
	CuTexImage _texMatch[2];
	CuTexImage _texCRT;

	//programs
	//
	int _num_sift[2];
	int _id_sift[2];
	int _have_loc[2];

	//gpu parameter
	int _initialized;
	vector<int> sift_buffer;
private:
	int  GetBestMatch(int max_match, uint32_t match_buffer[][2], float distmax, float ratiomax, int mbm);
public:
	SiftMatchCU(int max_sift);
	virtual ~SiftMatchCU(){};
	void InitSiftMatch();
  bool Allocate(int max_sift, int mbm) override;
	void SetMaxSift(int max_sift) override;
	void SetDescriptors(int index, int num, const unsigned char * descriptor, int id = -1);
	void SetDescriptors(int index, int num, const float * descriptor, int id = -1);
	void SetFeautreLocation(int index, const float* locatoins, int gap);
	int  GetSiftMatch(int max_match, uint32_t match_buffer[][2], float distmax, float ratiomax, int mbm);
	int  GetGuidedSiftMatch(int max_match, uint32_t match_buffer[][2], float* H, float* F,
									 float distmax, float ratiomax, float hdistmax, float fdistmax, int mbm);
    //////////////////////////////
    static int  CheckCudaDevice(int device);
};

#endif
#endif

