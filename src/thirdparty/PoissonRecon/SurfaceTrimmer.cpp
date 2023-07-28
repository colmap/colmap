/*
Copyright (c) 2013, Michael Kazhdan
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

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP
#include <algorithm>
#include "CmdLineParser.h"
#include "Geometry.h"
#include "Ply.h"
#include "MAT.h"
#include "MyTime.h"

cmdLineString In( "in" ) , Out( "out" );
cmdLineInt Smooth( "smooth" , 5 );
cmdLineFloat Trim( "trim" ) , IslandAreaRatio( "aRatio" , 0.001f );
cmdLineReadable PolygonMesh( "polygonMesh" );

cmdLineReadable* params[] =
{
	&In , &Out , &Trim , &PolygonMesh , &Smooth , &IslandAreaRatio
};

void ShowUsage( char* ex )
{
	printf( "Usage: %s\n" , ex );
	printf( "\t --%s <input polygon mesh>\n" , In.name );
	printf( "\t[--%s <ouput polygon mesh>]\n" , Out.name );
	printf( "\t[--%s <smoothing iterations>=%d]\n" , Smooth.name , Smooth.value );
	printf( "\t[--%s <trimming value>]\n" , Trim.name );
	printf( "\t[--%s <relative area of islands>=%f]\n" , IslandAreaRatio.name , IslandAreaRatio.value );
	printf( "\t[--%s]\n" , PolygonMesh.name );
}

long long EdgeKey( int key1 , int key2 )
{
	if( key1<key2 ) return ( ( (long long)key1 )<<32 ) | ( (long long)key2 );
	else            return ( ( (long long)key2 )<<32 ) | ( (long long)key1 );
}

template< class Real , class Vertex >
Vertex InterpolateVertices( const Vertex& v1 , const Vertex& v2 , Real value )
{
	typename Vertex::Wrapper _v1(v1) , _v2(v2);
	if( _v1.value==_v2.value ) return Vertex( (_v1+_v2)/Real(2.) );

	Real dx = ( _v1.value-value ) / ( _v1.value-_v2.value );
	return Vertex( _v1*(1.f-dx) + _v2*dx );
}
template< class Real , class Vertex >
void SmoothValues( std::vector< Vertex >& vertices , const std::vector< std::vector< int > >& polygons )
{
	std::vector< int > count( vertices.size() );
	std::vector< Real > sums( vertices.size() , 0 );
	for( size_t i=0 ; i<polygons.size() ; i++ )
	{
		int sz = int(polygons[i].size());
		for( int j=0 ; j<sz ; j++ )
		{
			int j1 = j , j2 = (j+1)%sz;
			int v1 = polygons[i][j1] , v2 = polygons[i][j2];
			count[v1]++ , count[v2]++;
			sums[v1] += vertices[v2].value , sums[v2] += vertices[v1].value;
		}
	}
	for( size_t i=0 ; i<vertices.size() ; i++ ) vertices[i].value = ( sums[i] + vertices[i].value ) / ( count[i] + 1 );
}
template< class Real , class Vertex >
void SplitPolygon
	(
	const std::vector< int >& polygon ,
	std::vector< Vertex >& vertices ,
	std::vector< std::vector< int > >* ltPolygons , std::vector< std::vector< int > >* gtPolygons ,
	std::vector< bool >* ltFlags , std::vector< bool >* gtFlags ,
	hash_map< long long , int >& vertexTable ,
	Real trimValue
	)
{
	int sz = int( polygon.size() );
	std::vector< bool > gt( sz );
	int gtCount = 0;
	for( int j=0 ; j<sz ; j++ )
	{
		gt[j] = ( vertices[ polygon[j] ].value>trimValue );
		if( gt[j] ) gtCount++;
	}
	if     ( gtCount==sz ){ if( gtPolygons ) gtPolygons->push_back( polygon ) ; if( gtFlags ) gtFlags->push_back( false ); }
	else if( gtCount==0  ){ if( ltPolygons ) ltPolygons->push_back( polygon ) ; if( ltFlags ) ltFlags->push_back( false ); }
	else
	{
		int start;
		for( start=0 ; start<sz ; start++ ) if( gt[start] && !gt[(start+sz-1)%sz] ) break;

		bool gtFlag = true;
		std::vector< int > poly;

		// Add the initial vertex
		{
			int j1 = (start+int(sz)-1)%sz , j2 = start;
			int v1 = polygon[j1] , v2 = polygon[j2];
			int vIdx;
			hash_map< long long , int >::iterator iter = vertexTable.find( EdgeKey( v1 , v2 ) );
			if( iter==vertexTable.end() )
			{
				vertexTable[ EdgeKey( v1 , v2 ) ] = vIdx = int( vertices.size() );
				vertices.push_back( InterpolateVertices( vertices[v1] , vertices[v2] , trimValue ) );
			}
			else vIdx = iter->second;
			poly.push_back( vIdx );
		}

		for( int _j=0  ; _j<=sz ; _j++ )
		{
			int j1 = (_j+start+sz-1)%sz , j2 = (_j+start)%sz;
			int v1 = polygon[j1] , v2 = polygon[j2];
			if( gt[j2]==gtFlag ) poly.push_back( v2 );
			else
			{
				int vIdx;
				hash_map< long long , int >::iterator iter = vertexTable.find( EdgeKey( v1 , v2 ) );
				if( iter==vertexTable.end() )
				{
					vertexTable[ EdgeKey( v1 , v2 ) ] = vIdx = int( vertices.size() );
					vertices.push_back( InterpolateVertices( vertices[v1] , vertices[v2] , trimValue ) );
				}
				else vIdx = iter->second;
				poly.push_back( vIdx );
				if( gtFlag ){ if( gtPolygons ) gtPolygons->push_back( poly ) ; if( ltFlags ) ltFlags->push_back( true ); }
				else        { if( ltPolygons ) ltPolygons->push_back( poly ) ; if( gtFlags ) gtFlags->push_back( true ); }
				poly.clear() , poly.push_back( vIdx ) , poly.push_back( v2 );
				gtFlag = !gtFlag;
			}
		}
	}
}
template< class Real , class Vertex >
void Triangulate( const std::vector< Vertex >& vertices , const std::vector< std::vector< int > >& polygons , std::vector< std::vector< int > >& triangles )
{
	triangles.clear();
	for( size_t i=0 ; i<polygons.size() ; i++ )
		if( polygons.size()>3 )
		{
			MinimalAreaTriangulation< Real > mat;
			std::vector< Point3D< Real > > _vertices( polygons[i].size() );
			std::vector< TriangleIndex > _triangles;
			for( int j=0 ; j<int( polygons[i].size() ) ; j++ ) _vertices[j] = vertices[ polygons[i][j] ].point;
			mat.GetTriangulation( _vertices , _triangles );

			// Add the triangles to the mesh
			size_t idx = triangles.size();
			triangles.resize( idx+_triangles.size() );
			for( int j=0 ; j<int(_triangles.size()) ; j++ )
			{
				triangles[idx+j].resize(3);
				for( int k=0 ; k<3 ; k++ ) triangles[idx+j][k] = polygons[i][ _triangles[j].idx[k] ];
			}
		}
		else if( polygons[i].size()==3 ) triangles.push_back( polygons[i] );
}
template< class Real , class Vertex >
double PolygonArea( const std::vector< Vertex >& vertices , const std::vector< int >& polygon )
{
	if( polygon.size()<3 ) return 0.;
	else if( polygon.size()==3 ) return TriangleArea( vertices[polygon[0]].point , vertices[polygon[1]].point , vertices[polygon[2]].point );
	else
	{
		Point3D< Real > center;
		for( size_t i=0 ; i<polygon.size() ; i++ ) center += vertices[ polygon[i] ].point;
		center /= Real( polygon.size() );
		double area = 0;
		for( size_t i=0 ; i<polygon.size() ; i++ ) area += TriangleArea( center , vertices[ polygon[i] ].point , vertices[ polygon[ (i+1)%polygon.size() ] ].point );
		return area;
	}
}

template< class Vertex >
void RemoveHangingVertices( std::vector< Vertex >& vertices , std::vector< std::vector< int > >& polygons )
{
	hash_map< int , int > vMap;
	std::vector< bool > vertexFlags( vertices.size() , false );
	for( size_t i=0 ; i<polygons.size() ; i++ ) for( size_t j=0 ; j<polygons[i].size() ; j++ ) vertexFlags[ polygons[i][j] ] = true;
	int vCount = 0;
	for( int i=0 ; i<int(vertices.size()) ; i++ ) if( vertexFlags[i] ) vMap[i] = vCount++;
	for( size_t i=0 ; i<polygons.size() ; i++ ) for( size_t j=0 ; j<polygons[i].size() ; j++ ) polygons[i][j] = vMap[ polygons[i][j] ];

	std::vector< Vertex > _vertices( vCount );
	for( int i=0 ; i<int(vertices.size()) ; i++ ) if( vertexFlags[i] ) _vertices[ vMap[i] ] = vertices[i];
	vertices = _vertices;
}
void SetConnectedComponents( const std::vector< std::vector< int > >& polygons , std::vector< std::vector< int > >& components )
{
	std::vector< int > polygonRoots( polygons.size() );
	for( size_t i=0 ; i<polygons.size() ; i++ ) polygonRoots[i] = int(i);
	hash_map< long long , int > edgeTable;
	for( size_t i=0 ; i<polygons.size() ; i++ )
	{
		int sz = int( polygons[i].size() );
		for( int j=0 ; j<sz ; j++ )
		{
			int j1 = j , j2 = (j+1)%sz;
			int v1 = polygons[i][j1] , v2 = polygons[i][j2];
			long long eKey = EdgeKey( v1 , v2 );
			hash_map< long long , int >::iterator iter = edgeTable.find( eKey );
			if( iter==edgeTable.end() ) edgeTable[ eKey ] = int(i);
			else
			{
				int p = iter->second;
				while( polygonRoots[p]!=p )
				{
					int temp = polygonRoots[p];
					polygonRoots[p] = int(i);
					p = temp;
				}
				polygonRoots[p] = int(i);
			}
		}
	}
	for( size_t i=0 ; i<polygonRoots.size() ; i++ )
	{
		int p = int(i);
		while( polygonRoots[p]!=p ) p = polygonRoots[p];
		int root = p;
		p = int(i);
		while( polygonRoots[p]!=p )
		{
			int temp = polygonRoots[p];
			polygonRoots[p] = root;
			p = temp;
		}
	}
	int cCount = 0;
	hash_map< int , int > vMap;
	for( int i= 0 ; i<int(polygonRoots.size()) ; i++ ) if( polygonRoots[i]==i ) vMap[i] = cCount++;
	components.resize( cCount );
	for( int i=0 ; i<int(polygonRoots.size()) ; i++ ) components[ vMap[ polygonRoots[i] ] ].push_back(i);
}
template< class Real >
inline Point3D< Real > CrossProduct( Point3D< Real > p1 , Point3D< Real > p2 ){ return Point3D< Real >( p1[1]*p2[2]-p1[2]*p2[1] , p1[2]*p2[0]-p1[0]*p2[2] , p1[0]*p1[1]-p1[1]*p2[0] ); }
template< class Real >
double TriangleArea( Point3D< Real > v1 , Point3D< Real > v2 , Point3D< Real > v3 )
{
	Point3D< Real > n = CrossProduct( v2-v1 , v3-v1 );
	return sqrt( n[0]*n[0] + n[1]*n[1] + n[2]*n[2] ) / 2.;
}
template< class Vertex >
int Execute( void )
{
	float min , max;
	int paramNum = sizeof(params)/sizeof(cmdLineReadable*);
	std::vector< Vertex > vertices;
	std::vector< std::vector< int > > polygons;

	int ft , commentNum = paramNum+2;
	char** comments;
	PlyReadPolygons( In.value , vertices , polygons , Vertex::ReadProperties , Vertex::ReadComponents , ft , &comments , &commentNum );
	for( int i=0 ; i<Smooth.value ; i++ ) SmoothValues< float , Vertex >( vertices , polygons );
	min = max = vertices[0].value;
	for( size_t i=0 ; i<vertices.size() ; i++ ) min = std::min< float >( min , vertices[i].value ) , max = std::max< float >( max , vertices[i].value );
	printf( "Value Range: [%f,%f]\n" , min , max );


	hash_map< long long , int > vertexTable;
	std::vector< std::vector< int > > ltPolygons , gtPolygons;
	std::vector< bool > ltFlags , gtFlags;

	for( int i=0 ; i<paramNum+2 ; i++ ) comments[i+commentNum]=new char[1024];
	sprintf( comments[commentNum++] , "Running Surface Trimmer (V5)" );
	if(              In.set ) sprintf(comments[commentNum++],"\t--%s %s" , In.name , In.value );
	if(             Out.set ) sprintf(comments[commentNum++],"\t--%s %s" , Out.name , Out.value );
	if(            Trim.set ) sprintf(comments[commentNum++],"\t--%s %f" , Trim.name , Trim.value );
	if(          Smooth.set ) sprintf(comments[commentNum++],"\t--%s %d" , Smooth.name , Smooth.value );
	if( IslandAreaRatio.set ) sprintf(comments[commentNum++],"\t--%s %f" , IslandAreaRatio.name , IslandAreaRatio.value );
	if(     PolygonMesh.set ) sprintf(comments[commentNum++],"\t--%s" , PolygonMesh.name );

	double t=Time();
	for( size_t i=0 ; i<polygons.size() ; i++ ) SplitPolygon( polygons[i] , vertices , &ltPolygons , &gtPolygons , &ltFlags , &gtFlags , vertexTable , Trim.value );
	if( IslandAreaRatio.value>0 )
	{
		std::vector< std::vector< int > > _ltPolygons , _gtPolygons;
		std::vector< std::vector< int > > ltComponents , gtComponents;
		SetConnectedComponents( ltPolygons , ltComponents );
		SetConnectedComponents( gtPolygons , gtComponents );
		std::vector< double > ltAreas( ltComponents.size() , 0. ) , gtAreas( gtComponents.size() , 0. );
		std::vector< bool > ltComponentFlags( ltComponents.size() , false ) , gtComponentFlags( gtComponents.size() , false );
		double area = 0.;
		for( size_t i=0 ; i<ltComponents.size() ; i++ )
		{
			for( size_t j=0 ; j<ltComponents[i].size() ; j++ )
			{
				ltAreas[i] += PolygonArea< float , Vertex >( vertices , ltPolygons[ ltComponents[i][j] ] );
				ltComponentFlags[i] = ( ltComponentFlags[i] || ltFlags[ ltComponents[i][j] ] );
			}
			area += ltAreas[i];
		}
		for( size_t i=0 ; i<gtComponents.size() ; i++ )
		{
			for( size_t j=0 ; j<gtComponents[i].size() ; j++ )
			{
				gtAreas[i] += PolygonArea< float , Vertex >( vertices , gtPolygons[ gtComponents[i][j] ] );
				gtComponentFlags[i] = ( gtComponentFlags[i] || gtFlags[ gtComponents[i][j] ] );
			}
			area += gtAreas[i];
		}
		for( size_t i=0 ; i<ltComponents.size() ; i++ )
		{
			if( ltAreas[i]<area*IslandAreaRatio.value && ltComponentFlags[i] ) for( size_t j=0 ; j<ltComponents[i].size() ; j++ ) _gtPolygons.push_back( ltPolygons[ ltComponents[i][j] ] );
			else                                                               for( size_t j=0 ; j<ltComponents[i].size() ; j++ ) _ltPolygons.push_back( ltPolygons[ ltComponents[i][j] ] );
		}
		for( size_t i=0 ; i<gtComponents.size() ; i++ )
		{
			if( gtAreas[i]<area*IslandAreaRatio.value && gtComponentFlags[i] ) for( size_t j=0 ; j<gtComponents[i].size() ; j++ ) _ltPolygons.push_back( gtPolygons[ gtComponents[i][j] ] );
			else                                                               for( size_t j=0 ; j<gtComponents[i].size() ; j++ ) _gtPolygons.push_back( gtPolygons[ gtComponents[i][j] ] );
		}
		ltPolygons = _ltPolygons , gtPolygons = _gtPolygons;
	}
	if( !PolygonMesh.set )
	{
		{
			std::vector< std::vector< int > > polys = ltPolygons;
			Triangulate< float , Vertex >( vertices , ltPolygons , polys ) , ltPolygons = polys;
		}
		{
			std::vector< std::vector< int > > polys = gtPolygons;
			Triangulate< float  , Vertex >( vertices , gtPolygons , polys ) , gtPolygons = polys;
		}
	}

	RemoveHangingVertices( vertices , gtPolygons );
	sprintf( comments[commentNum++] , "#Trimmed In: %9.1f (s)" , Time()-t );
	if( Out.set ) PlyWritePolygons( Out.value , vertices , gtPolygons , Vertex::WriteProperties , Vertex::WriteComponents , ft , comments , commentNum );

	return EXIT_SUCCESS;
}
int SurfaceTrimmer( int argc , char* argv[] )
{
	int paramNum = sizeof(params)/sizeof(cmdLineReadable*);
	cmdLineParse( argc-1 , &argv[1] , paramNum , params , 0 );

	if( !In.set || !Trim.set )
	{
		ShowUsage( argv[0] );
		return EXIT_FAILURE;
	}
	bool readFlags[ PlyColorAndValueVertex< float >::ReadComponents ];
	if( !PlyReadHeader( In.value , PlyColorAndValueVertex< float >::ReadProperties , PlyColorAndValueVertex< float >::ReadComponents , readFlags ) ) fprintf( stderr , "[ERROR] Failed to read ply header: %s\n" , In.value ) , exit( 0 );

	bool hasValue = readFlags[3];
	bool hasColor = ( readFlags[4] || readFlags[7] ) && ( readFlags[5] || readFlags[8] ) && ( readFlags[6] || readFlags[9] );

	if( !hasValue ) fprintf( stderr , "[ERROR] Ply file does not contain values\n" ) , exit( 0 );
	if( hasColor ) return Execute< PlyColorAndValueVertex< float > >();
	else           return Execute< PlyValueVertex< float > >();
}
