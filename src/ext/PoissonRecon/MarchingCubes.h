/*
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#ifndef MARCHING_CUBES_INCLUDED
#define MARCHING_CUBES_INCLUDED
#include <vector>
#include "Geometry.h"

#define NEW_ORDERING 1

class Square
{
public:
	const static unsigned int CORNERS=4 , EDGES=4 , FACES=1;
	static int  CornerIndex			(int x,int y);
	static int  AntipodalCornerIndex(int idx);
	static void FactorCornerIndex	(int idx,int& x,int& y);
	static int  EdgeIndex			(int orientation,int i);
	static void FactorEdgeIndex		(int idx,int& orientation,int& i);

	static int  ReflectCornerIndex	(int idx,int edgeIndex);
	static int  ReflectEdgeIndex	(int idx,int edgeIndex);

	static void EdgeCorners(int idx,int& c1,int &c2);
};

class Cube{
public:
	const static unsigned int CORNERS=8 , EDGES=12 , FACES=6;

	static int  CornerIndex			( int x , int y , int z );
	static void FactorCornerIndex	( int idx , int& x , int& y , int& z );
	static int  EdgeIndex			( int orientation , int i , int j );
	static void FactorEdgeIndex		( int idx , int& orientation , int& i , int &j);
	static int  FaceIndex			( int dir , int offSet );
	static int  FaceIndex			( int x , int y , int z );
	static void FactorFaceIndex		( int idx , int& x , int &y , int& z );
	static void FactorFaceIndex		( int idx , int& dir , int& offSet );

	static int  AntipodalCornerIndex	( int idx );
	static int  FaceReflectCornerIndex	( int idx , int faceIndex );
	static int  FaceReflectEdgeIndex	( int idx , int faceIndex );
	static int	FaceReflectFaceIndex	( int idx , int faceIndex );
	static int	EdgeReflectCornerIndex	( int idx , int edgeIndex );
	static int	EdgeReflectEdgeIndex	( int edgeIndex );

	static int  FaceAdjacentToEdges	( int eIndex1 , int eIndex2 );
	static void FacesAdjacentToEdge	( int eIndex , int& f1Index , int& f2Index );

	static void EdgeCorners( int idx , int& c1 , int &c2 );
	static void FaceCorners( int idx , int& c1 , int &c2 , int& c3 , int& c4 );

	static bool IsEdgeCorner( int cIndex , int e );
	static bool IsFaceCorner( int cIndex , int f );
};

class MarchingSquares
{
	static double Interpolate(double v1,double v2);
	static void SetVertex(int e,const double values[Square::CORNERS],double iso);
public:
	const static unsigned int MAX_EDGES=2;
	static const int edgeMask[1<<Square::CORNERS];
	static const int edges[1<<Square::CORNERS][2*MAX_EDGES+1];
	static double vertexList[Square::EDGES][2];
#if NEW_ORDERING
	static const int cornerMap[Square::CORNERS];
#endif // NEW_ORDERING

	static unsigned char GetIndex( const float  values[Square::CORNERS] , float  iso );
	static unsigned char GetIndex( const double values[Square::CORNERS] , double iso );
	static bool IsAmbiguous( const double v[Square::CORNERS] , double isoValue );
	static bool IsAmbiguous( unsigned char idx );
	static bool HasRoots( unsigned char mcIndex );
#if NEW_ORDERING
	static bool HasEdgeRoots( unsigned char mcIndex , int edgeIndex );
#endif // NEW_ORDERING
	static int AddEdges( const double v[Square::CORNERS] , double isoValue , Edge* edges );
	static int AddEdgeIndices( const double v[Square::CORNERS] , double isoValue , int* edges);
	static int AddEdgeIndices( unsigned char mcIndex , int* edges);
};

class MarchingCubes
{
	static void SetVertex(int e,const double values[Cube::CORNERS],double iso);
	static unsigned char GetFaceIndex( const double values[Cube::CORNERS] , double iso , int faceIndex );

	static void SetVertex(int e,const float values[Cube::CORNERS],float iso);
	static unsigned char GetFaceIndex( const float values[Cube::CORNERS] , float iso , int faceIndex );

public:
	static unsigned char GetFaceIndex( unsigned char mcIndex , int faceIndex );
	static double Interpolate(double v1,double v2);
	static float Interpolate(float v1,float v2);
	const static unsigned int MAX_TRIANGLES=5;
	static const int edgeMask[1<<Cube::CORNERS];
	static const int triangles[1<<Cube::CORNERS][3*MAX_TRIANGLES+1];
	static const int cornerMap[Cube::CORNERS];
	static double vertexList[Cube::EDGES][3];

	static int AddTriangleIndices(int mcIndex,int* triangles);

	static unsigned char GetIndex( const double values[Cube::CORNERS] , double iso );
	static bool IsAmbiguous( const double v[Cube::CORNERS] , double isoValue , int faceIndex );
	static bool HasRoots( const double v[Cube::CORNERS] , double isoValue );
	static bool HasRoots( const double v[Cube::CORNERS] , double isoValue , int faceIndex );
	static int AddTriangles( const double v[Cube::CORNERS] , double isoValue , Triangle* triangles );
	static int AddTriangleIndices( const double v[Cube::CORNERS] , double isoValue , int* triangles );

	static unsigned char GetIndex( const float values[Cube::CORNERS] , float iso );
	static bool IsAmbiguous( const float v[Cube::CORNERS] , float isoValue , int faceIndex );
	static bool HasRoots( const float v[Cube::CORNERS] , float isoValue );
	static bool HasRoots( const float v[Cube::CORNERS] , float isoValue , int faceIndex );
	static int AddTriangles( const float v[Cube::CORNERS] , float isoValue , Triangle* triangles );
	static int AddTriangleIndices( const float v[Cube::CORNERS] , float isoValue , int* triangles );

	static bool IsAmbiguous( unsigned char mcIndex , int faceIndex );
	static bool HasRoots( unsigned char mcIndex );
	static bool HasFaceRoots( unsigned char mcIndex , int faceIndex );
	static bool HasEdgeRoots( unsigned char mcIndex , int edgeIndex );
};
#endif //MARCHING_CUBES_INCLUDED
