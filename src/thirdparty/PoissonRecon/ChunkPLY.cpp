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

#include "PreProcessor.h"

#define NEW_CHUNKS
#define DISABLE_PARALLELIZATION

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <limits>
#include <sstream>
#include <unordered_map>
#include "MyMiscellany.h"
#include "CmdLineParser.h"
#include "Geometry.h"
#include "Ply.h"
#include "DataStream.h"
#include "VertexFactory.h"
#include "DataStream.imp.h"

using namespace PoissonRecon;

CmdLineParameters< char* > In( "in" );
CmdLineParameter< char* > Out( "out" );
CmdLineParameter< float > Width( "width" , -1.f ) , PadRadius( "radius" , 0.f );
CmdLineParameterArray< float , 6 > BoundingBox( "bBox" );
CmdLineParameters< Point< float , 4 > > HalfSpaces( "halfSpaces" );
CmdLineReadable ASCII( "ascii" ) , Verbose( "verbose" ) , NoNormals( "noNormals" ) , Colors( "colors" ) , Values( "values" );

CmdLineReadable* params[] = { &In , &Out , &Width , &PadRadius , &ASCII , &Verbose , &BoundingBox , &NoNormals , &Colors , &Values , &HalfSpaces , NULL };

void ShowUsage( char* ex )
{
	printf( "Usage: %s\n" , ex );
	printf( "\t --%s <input geometry file count, geometry file #1, geometry file #2, ... >\n" , In.name );
	printf( "\t[--%s <output ply file name/header>]\n" , Out.name );
	printf( "\t[--%s <chunk width>=%f]\n" , Width.name , Width.value );
	printf( "\t[--%s <padding radius (as a fraction of the width)>=%f]\n" , PadRadius.name , PadRadius.value );
	printf( "\t[--%s <minx miny minz maxx maxy maxz>]\n" , BoundingBox.name );
	printf( "\t[--%s <half-space num, {x1,y1,z1,o1}, ..., {xn,yn,zn,on}>]\n" , HalfSpaces.name );
	printf( "\t[--%s]\n" , NoNormals.name );
	printf( "\t[--%s]\n" , Colors.name );
	printf( "\t[--%s]\n" , Values.name );
	printf( "\t[--%s]\n" , ASCII.name );
	printf( "\t[--%s]\n" , Verbose.name );
}

void PrintBoundingBox( Point< float , 3 > min , Point< float , 3 > max )
{
	printf( "[" );
	for( unsigned int d=0 ; d<3 ; d++ ) printf( " %f" , min[d] );
	printf( " ] [" );
	for( unsigned int d=0 ; d<3 ; d++ ) printf( " %f" , max[d] );
	printf( " ] ->" );
	for( unsigned int d=0 ; d<3 ; d++ ) printf( " %.2e" , max[d]-min[d] );
	printf( " ]" );
}

template< typename Real , unsigned int Dim , typename VertexDataFactory >
using FullVertexFactory = VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > , VertexDataFactory >;
template< typename Real , unsigned int Dim , typename VertexDataFactory >
using VertexType = typename FullVertexFactory< Real , Dim , VertexDataFactory >::VertexType;


template< typename VertexDataFactory >
void Read( char *const *fileNames , unsigned int fileNum , VertexDataFactory vertexDataFactory , std::vector< VertexType< float , 3 , VertexDataFactory > > &vertices , std::vector< std::vector< long long > > &polygons , int &ft , std::vector< std::string > &comments )
{
	typedef float Real;
	static constexpr unsigned int Dim = 3;
	FullVertexFactory< Real , Dim , VertexDataFactory > vertexFactory( VertexFactory::PositionFactory< Real , Dim >() , vertexDataFactory );

	for( unsigned int i=0 ; i<fileNum ; i++ )
	{
		std::vector< VertexType< Real , Dim , VertexDataFactory > > _vertices;
		std::vector< std::vector< long long > > _polygons;
		char *ext = GetFileExtension( fileNames[i] );

		if( !strcasecmp( ext , "ply" ) ) PLY::ReadPolygons( fileNames[i] , vertexFactory , _vertices , _polygons , ft , comments );
		else
		{
			InputDataStream< VertexType< Real , Dim , VertexDataFactory > > *pointStream;
			if( !strcasecmp( ext , "bnpts" ) ) pointStream = new BinaryInputDataStream< FullVertexFactory< Real , Dim , VertexDataFactory > >( fileNames[i] , vertexFactory );
			else                               pointStream = new  ASCIIInputDataStream< FullVertexFactory< Real , Dim , VertexDataFactory > >( fileNames[i] , vertexFactory );
			size_t count = 0;
			VertexType< Real , Dim , VertexDataFactory > v = vertexFactory();
			while( pointStream->read( v ) ) count++;
			pointStream->reset();
			_vertices.resize( count , v );
			comments.resize( 0 );
			ft = PLY_BINARY_NATIVE;

			count = 0;
			while( pointStream->read( _vertices[count++] ) );
			delete pointStream;
		}
		delete[] ext;

		long long vOffset = (long long)vertices.size();

		vertices.reserve( vertices.size() + _vertices.size() );
		for( int i=0 ; i<_vertices.size() ; i++ ) vertices.push_back( _vertices[i] );

		polygons.reserve( polygons.size() + _polygons.size() );
		for( int i=0 ; i<_polygons.size() ; i++ )
		{
			for( int j=0 ; j<_polygons[i].size() ; j++ ) _polygons[i][j] += vOffset;
			polygons.push_back( _polygons[i] );
		}
	}
}

template< typename VertexDataFactory >
void WritePoints( const char *fileName , int ft , VertexDataFactory vertexDataFactory , const std::vector< VertexType< float , 3 , VertexDataFactory > > &vertices , const std::vector< std::string > &comments )
{
	typedef float Real;
	static constexpr unsigned int Dim = 3;
	FullVertexFactory< Real , Dim , VertexDataFactory > vertexFactory( VertexFactory::PositionFactory< Real , Dim >() , vertexDataFactory );
	char *ext = GetFileExtension( fileName );

	OutputDataStream< VertexType< Real , Dim , VertexDataFactory > > *pointStream;

	if     ( !strcasecmp( ext , "ply"   ) ) pointStream = new    PLYOutputDataStream< FullVertexFactory< Real , Dim , VertexDataFactory > >( fileName , vertexFactory , vertices.size() , ft );
	else if( !strcasecmp( ext , "bnpts" ) ) pointStream = new BinaryOutputDataStream< FullVertexFactory< Real , Dim , VertexDataFactory > >( fileName , vertexFactory );
	else                                    pointStream = new  ASCIIOutputDataStream< FullVertexFactory< Real , Dim , VertexDataFactory > >( fileName , vertexFactory );
	for( size_t i=0 ; i<vertices.size() ; i++ ) pointStream->write( vertices[i] );
	delete pointStream;

	delete[] ext;
}


template< typename VertexDataFactory >
void WriteMesh( const char *fileName , int ft , VertexDataFactory vertexDataFactory , const std::vector< VertexType< float , 3 , VertexDataFactory > > &vertices , const std::vector< std::vector< long long > > &polygons , const std::vector< std::string > &comments )
{
	typedef float Real;
	static constexpr unsigned int Dim = 3;
	FullVertexFactory< Real , Dim , VertexDataFactory > vertexFactory( VertexFactory::PositionFactory< Real , Dim >() , vertexDataFactory );

	char *ext = GetFileExtension( fileName );
	if( strcasecmp( ext , "ply" ) ) MK_THROW( "Can only output mesh to .ply file" );
	delete[] ext;

	if( vertices.size()>std::numeric_limits< int >::max() )
	{
		if( vertices.size()>std::numeric_limits< unsigned int >::max() ) MK_THROW( "more vertices than can be indexed by an unsigned int: %llu" , (unsigned long long)vertices.size() );
		MK_WARN( "more vertices than can be indexed by an int, using unsigned int instead: %llu" , (unsigned long long)vertices.size() );
		std::vector< std::vector< unsigned int > > outPolygons;
		outPolygons.resize( polygons.size() );
		for( size_t i=0 ; i<polygons.size() ; i++ )
		{
			outPolygons[i].resize( polygons[i].size() );
			for( int j=0 ; j<polygons[i].size() ; j++ ) outPolygons[i][j] = (unsigned int)polygons[i][j];
		}
		PLY::WritePolygons( fileName , vertexFactory , vertices , outPolygons , ft , comments );
	}
	else
	{
		std::vector< std::vector< int > > outPolygons;
		outPolygons.resize( polygons.size() );
		for( size_t i=0 ; i<polygons.size() ; i++ )
		{
			outPolygons[i].resize( polygons[i].size() );
			for( int j=0 ; j<polygons[i].size() ; j++ ) outPolygons[i][j] = (int)polygons[i][j];
		}
		PLY::WritePolygons( fileName , vertexFactory , vertices , outPolygons , ft , comments );
	}
}

template< typename VertexDataFactory >
void GetBoundingBox( const std::vector< VertexType< float , 3 , VertexDataFactory > > &vertices , Point< float , 3 > &min , Point< float , 3 > &max )
{
	min = max = vertices[0].template get<0>();
	for( size_t i=0 ; i<vertices.size() ; i++ ) for( unsigned int d=0 ; d<3 ; d++ )
	{
		min[d] = std::min< float >( min[d] , vertices[i].template get<0>()[d] );
		max[d] = std::max< float >( max[d] , vertices[i].template get<0>()[d] );
	}
}

template< typename VertexDataFactory >
void GetSubPoints( const std::vector< VertexType< float , 3 , VertexDataFactory > > &vertices , Point< float , 3 > min , Point< float , 3 > max , std::vector< VertexType< float , 3 , VertexDataFactory > > &subVertices )
{
	subVertices.resize( 0 );

	for( size_t i=0 ; i<vertices.size() ; i++ )
	{
		bool inside = true;
		for( unsigned int d=0 ; d<3 ; d++ ) if( vertices[i].template get<0>()[d]<min[d] || vertices[i].template get<0>()[d]>=max[d] ) inside = false;
		if( inside ) subVertices.push_back( vertices[i] );
	}
}

template< typename VertexDataFactory >
void GetSubVertices( const std::vector< VertexType< float , 3 , VertexDataFactory > > &vertices , std::vector< std::vector< long long > > &polygons , std::vector< VertexType< float , 3 , VertexDataFactory > > &subVertices )
{
	subVertices.resize( 0 );
	long long count = 0;
	std::unordered_map< long long , long long > vMap;
	for( size_t i=0 ; i<polygons.size() ; i++ ) for( size_t j=0 ; j<polygons[i].size() ; j++ )
	{
		auto iter = vMap.find( polygons[i][j] );
		if( iter==vMap.end() ) vMap[ polygons[i][j] ] = count++;
	}

	subVertices.resize( vMap.size() );

	for( size_t i=0 ; i<polygons.size() ; i++ ) for( size_t j=0 ; j<polygons[i].size() ; j++ )
	{
		long long oldIdx = polygons[i][j];
		long long newIdx = vMap[ oldIdx ];
		subVertices[ newIdx ] = vertices[ oldIdx ];
		polygons[i][j] = newIdx;
	}
}

template< typename VertexDataFactory >
void GetSubMesh( const std::vector< VertexType< float , 3 , VertexDataFactory > > &vertices , const std::vector< std::vector< long long > > &polygons , Point< float , 3 > min , Point< float , 3 > max , std::vector< VertexType< float , 3 , VertexDataFactory > > &subVertices , std::vector< std::vector< long long > > &subPolygons )
{
	subPolygons.resize( 0 );

	for( size_t i=0 ; i<polygons.size() ; i++ )
	{
		Point< float , 3 > center;
		for( size_t j=0 ; j<polygons[i].size() ; j++ ) center += vertices[ polygons[i][j] ].template get<0>();
		center /= (float)polygons[i].size();
		bool inside = true;
		for( unsigned int d=0 ; d<3 ; d++ ) if( center[d]<min[d] || center[d]>=max[d] ) inside = false;
		if( inside ) subPolygons.push_back( polygons[i] );
	}

	GetSubVertices( vertices , subPolygons , subVertices );
}

template< typename VertexDataFactory >
void Execute( VertexDataFactory vertexDataFactory )
{
	typedef float Real;
	static constexpr unsigned int Dim = 3;
	Timer timer;
	double t = Time();
	std::vector< VertexType< Real , Dim , VertexDataFactory > > vertices;
	std::vector< std::vector< long long > > polygons;

	int ft;
	std::vector< std::string > comments;
	Read( In.values , In.count , vertexDataFactory , vertices , polygons , ft , comments );
	printf( "Vertices / Polygons: %llu / %llu\n" , (unsigned long long)vertices.size() , (unsigned long long)polygons.size() );
	printf( "Time (Wall/CPU): %.2f / %.2f\n" , timer.wallTime() , timer.cpuTime() );
	printf( "Peak Memory (MB): %d\n" , MemoryInfo::PeakMemoryUsageMB() );

	Point< float , 3 > min , max;
	GetBoundingBox< VertexDataFactory >( vertices , min , max );
	PrintBoundingBox( min , max ) ; printf( "\n" );

	float width = Width.value;


	if( BoundingBox.set || HalfSpaces.set )
	{
		Point< float , 3 > min , max;

		if( BoundingBox.set )
		{
			min = Point< float , 3 >( BoundingBox.values[0] , BoundingBox.values[1] , BoundingBox.values[2] );
			max = Point< float , 3 >( BoundingBox.values[3] , BoundingBox.values[4] , BoundingBox.values[5] );
		}

		auto Inside = [&]( Point< float , 3 > p )
			{
				bool inside = true;
				if( BoundingBox.set ) inside &= p[0]>=min[0] && p[0]<max[0] && p[1]>=min[1] && p[1]<max[1] && p[2]>=min[2] && p[2]<max[2];
				if( inside && HalfSpaces.set )
				{
					Point< float , 4 > _p( p[0] , p[1] , p[2] , 1.f );
					for( unsigned int i=0 ; i<(unsigned int)HalfSpaces.count ; i++ ) inside &= Point< float , 4 >::Dot( _p , HalfSpaces.values[i] )<=0;
				}
				return inside;
			};

		if( polygons.size() )
		{
			std::vector< std::vector< long long > > _polygons;

			Timer timer;
#ifdef NEW_CHUNKS
			{
				size_t polygonCount = 0;
				for( size_t i=0 ; i<polygons.size() ; i++ )
				{
					Point< float , 3 > center;
					for( int j=0 ; j<polygons[i].size() ; j++ ) center += vertices[ polygons[i][j] ].template get<0>();
					center /= polygons[i].size();
					if( Inside( center ) ) polygonCount++;
				}
				_polygons.reserve( polygonCount );
			}
#endif // NEW_CHUNKS
			for( size_t i=0 ; i<polygons.size() ; i++ )
			{
				Point< float , 3 > center;
				for( int j=0 ; j<polygons[i].size() ; j++ ) center += vertices[ polygons[i][j] ].template get<0>();
				center /= polygons[i].size();
				if( Inside( center ) ) _polygons.push_back( polygons[i] );
			}
			printf( "\tChunked polygons:\n" );
			printf( "\t\tTime (Wall/CPU): %.2f / %.2f\n" , timer.wallTime() , timer.cpuTime() );
			printf( "\t\tPeak Memory (MB): %d\n" , MemoryInfo::PeakMemoryUsageMB() );

			if( Out.set )
				if( _polygons.size() )
				{
					std::vector< VertexType< Real , Dim , VertexDataFactory > > _vertices;
					GetSubVertices< VertexDataFactory >( vertices , _polygons , _vertices );

					if( Verbose.set )
					{
						printf( "\t\t%s\n" , Out.value );
						printf( "\t\t\tVertices / Polygons: %llu / %llu\n" , (unsigned long long)_vertices.size() , (unsigned long long)_polygons.size() );
					}

					WriteMesh( Out.value , ASCII.set ? PLY_ASCII : ft , vertexDataFactory , _vertices , _polygons , comments );
				}
				else MK_WARN( "no polygons in bounding box" );
		}
		else
		{
			std::vector< VertexType< Real , Dim , VertexDataFactory > > _vertices;

			Timer timer;
#ifdef NEW_CHUNKS
			{
				size_t vertexCount = 0;
				for( size_t i=0 ; i<vertices.size() ; i++ ) if( Inside( vertices[i].template get<0>() ) ) vertexCount++;
				_vertices.reserve( vertexCount );
			}
#endif // NEW_CHUNKS

			for( size_t i=0 ; i<vertices.size() ; i++ ) if( Inside( vertices[i].template get<0>() ) ) _vertices.push_back( vertices[i] );
			printf( "\tChunked vertices:\n" );
			printf( "\t\tTime (Wall/CPU): %.2f / %.2f\n" , timer.wallTime() , timer.cpuTime() );
			printf( "\t\tPeak Memory (MB): %d\n" , MemoryInfo::PeakMemoryUsageMB() );

			if( Out.set )
				if( _vertices.size() )
				{
					if( Verbose.set )
					{
						printf( "\t\t%s\n" , Out.value );
						printf( "\t\t\tPoints: %llu\n" , (unsigned long long)_vertices.size() );
					}

					WritePoints( Out.value , ASCII.set ? PLY_ASCII : ft , vertexDataFactory , _vertices , comments );
				}
				else MK_WARN( "no vertices in bounding box" );
		}
	}
	else if( width>0 )
	{
		float radius = PadRadius.value * width;
		size_t vCount=0 , pCount=0;
		for( unsigned int d=0 ; d<3 ; d++ ) min[d] -= radius , max[d] += radius;
		for( unsigned int d=0 ; d<3 ; d++ ) min[d] -= width/10000.f , max[d] += width/10000.f;
		int begin[] = { (int)floor( min[0]/width ) , (int)floor( min[1]/width ) , (int)floor( min[2]/width ) };
		int end  [] = { (int)ceil ( max[0]/width ) , (int)ceil ( max[1]/width ) , (int)ceil ( max[2]/width ) };
		int size [] = { end[0]-begin[0]+1 , end[1]-begin[1]+1 , end[2]-begin[2]+1 };
		struct Range{ int begin[3] , end[3]; };
		auto SetRange = [&]( Point< float , 3 > p , Range &range )
		{
			for( int d=0 ; d<3 ; d++ )
			{
				range.begin[d] = (int)floor( (p[d]-radius)/width ) , range.end[d] = (int)ceil( (p[d]+radius)/width );
				if( range.begin[d]==range.end[d] ) range.end[d]++;
			}
		};
		auto Index1D = [&]( int x , int y , int z )
		{
			x -= begin[0] , y -= begin[1] , z -= begin[2];
			return x + y*size[0] + z*size[0]*size[1];
		};
		auto Index3D = [&]( int idx , int &x , int &y , int &z )
		{
			x = idx % size[0];
			idx /= size[0];
			y = idx % size[1];
			idx /= size[1];
			z = idx % size[2];
			x += begin[0] , y += begin[1] , z += begin[2];
		};

		if( polygons.size() )
		{
			std::vector< std::vector< std::vector< long long > > > _polygons( size[0]*size[1]*size[2] );
			Range range;

			Timer timer;
#ifdef NEW_CHUNKS
			{
				std::vector< size_t > polygonCounts( size[0]*size[1]*size[2] , 0 );
				for( size_t i=0 ; i<polygons.size() ; i++ )
				{
					Point< float , 3 > center;
					for( int j=0 ; j<polygons[i].size() ; j++ ) center += vertices[ polygons[i][j] ].template get<0>();
					center /= polygons[i].size();
					SetRange( center , range );
					for( int x=range.begin[0] ; x<range.end[0] ; x++ ) for( int y=range.begin[1] ; y<range.end[1] ; y++ ) for( int z=range.begin[2] ; z<range.end[2] ; z++ )
						polygonCounts[ Index1D(x,y,z) ]++;
				}
				for( size_t i=0 ; i<polygonCounts.size() ; i++ ) _polygons[i].reserve( polygonCounts[i] );
			}
#endif // NEW_CHUNKS
			for( size_t i=0 ; i<polygons.size() ; i++ )
			{
				Point< float , 3 > center;
				for( int j=0 ; j<polygons[i].size() ; j++ ) center += vertices[ polygons[i][j] ].template get<0>();
				center /= polygons[i].size();
				SetRange( center , range );
				for( int x=range.begin[0] ; x<range.end[0] ; x++ ) for( int y=range.begin[1] ; y<range.end[1] ; y++ ) for( int z=range.begin[2] ; z<range.end[2] ; z++ )
					_polygons[ Index1D(x,y,z) ].push_back( polygons[i] );
			}
			printf( "\tChunked polygons:\n" );
			printf( "\t\tTime (Wall/CPU): %.2f / %.2f\n" , timer.wallTime() , timer.cpuTime() );
			printf( "\t\tPeak Memory (MB): %d\n" , MemoryInfo::PeakMemoryUsageMB() );

			if( Out.set )
			{
#ifdef DISABLE_PARALLELIZATION
#else // !DISABLE_PARALLELIZATION
#pragma omp parallel for
#endif // DISABLE_PARALLELIZATION
				for( int i=0 ; i<_polygons.size() ; i++ ) if( _polygons[i].size() )
				{
					std::vector< VertexType< Real , Dim , VertexDataFactory > > _vertices;
					GetSubVertices< VertexDataFactory >( vertices , _polygons[i] , _vertices );
					std::stringstream stream;
					int x , y , z;
					Index3D( i , x , y , z );

					stream << Out.value << "." << x << "." << y << "." << z << ".ply";


					Point< float , 3 > min , max;
					min = Point< float , 3 >( x+0 , y+0 , z+0 ) * width;
					max = Point< float , 3 >( x+1 , y+1 , z+1 ) * width;
					if( Verbose.set )
					{
						static std::mutex mutex;
						std::lock_guard< std::mutex > lock( mutex );
						printf( "\t\t%s\n" , stream.str().c_str() );
						printf( "\t\t\tVertices / Polygons: %llu / %llu\n" , (unsigned long long)_vertices.size() , (unsigned long long)_polygons[i].size() );
						printf( "\t\t\t" ) ; PrintBoundingBox( min , max ) ; printf( "\n" );
					}

					WriteMesh( stream.str().c_str() , ASCII.set ? PLY_ASCII : ft , vertexDataFactory , _vertices , _polygons[i] , comments );
					vCount += _vertices.size() , pCount += _polygons[i].size();
				}
			}
		}
		else
		{
			std::vector< std::vector< VertexType< Real , Dim , VertexDataFactory > > > _vertices( size[0]*size[1]*size[2] );
			Range range;

			Timer timer;
#ifdef NEW_CHUNKS
			{
				std::vector< size_t > vertexCounts( size[0]*size[1]*size[2] , 0 );
				for( size_t i=0 ; i<vertices.size() ; i++ )
				{
					SetRange( vertices[i].template get<0>() , range );
					for( int x=range.begin[0] ; x<range.end[0] ; x++ ) for( int y=range.begin[1] ; y<range.end[1] ; y++ ) for( int z=range.begin[2] ; z<range.end[2] ; z++ )
						vertexCounts[ Index1D(x,y,z) ]++;
				}
				for( size_t i=0 ; i<vertexCounts.size() ; i++ ) _vertices[i].reserve( vertexCounts[i] );
			}
#endif // NEW_CHUNKS

			for( size_t i=0 ; i<vertices.size() ; i++ )
			{
				SetRange( vertices[i].template get<0>() , range );
				for( int x=range.begin[0] ; x<range.end[0] ; x++ ) for( int y=range.begin[1] ; y<range.end[1] ; y++ ) for( int z=range.begin[2] ; z<range.end[2] ; z++ )
					_vertices[ Index1D(x,y,z) ].push_back( vertices[i] );
			}
			printf( "\tChunked vertices:\n" );
			printf( "\t\tTime (Wall/CPU): %.2f / %.2f\n" , timer.wallTime() , timer.cpuTime() );
			printf( "\t\tPeak Memory (MB): %d\n" , MemoryInfo::PeakMemoryUsageMB() );

			if( Out.set )
			{
#ifdef DISABLE_PARALLELIZATION
#else // !DISABLE_PARALLELIZATION
#pragma omp parallel for
#endif  // DISABLE_PARALLELIZATION
				for( int i=0 ; i<_vertices.size() ; i++ ) if( _vertices[i].size() )
				{
					std::stringstream stream;
					int x , y , z;
					Index3D( i , x , y , z );
					stream << Out.value << "." << x << "." << y << "." << z << ".ply";
					Point< float , 3 > min , max;
					min = Point< float , 3 >( x+0 , y+0 , z+0 ) * width;
					max = Point< float , 3 >( x+1 , y+1 , z+1 ) * width;

					if( Verbose.set )
					{
						static std::mutex mutex;
						std::lock_guard< std::mutex > lock( mutex );
						printf( "\t\t%s\n" , stream.str().c_str() );
						printf( "\t\t\tPoints: %llu\n" , (unsigned long long)_vertices[i].size() );
						printf( "\t\t\t" ) ; PrintBoundingBox( min , max ) ; printf( "\n" );
					}

					WritePoints( stream.str().c_str() , ASCII.set ? PLY_ASCII : ft , vertexDataFactory , _vertices[i] , comments );
					vCount += _vertices[i].size();
				}
			}
		}
		if( !radius )
		{
			if( polygons.size() )
			{
				if( pCount!=polygons.size() ) MK_WARN( "polygon counts don't match: " , polygons.size() , " != " , pCount );
			}
			else
			{
				if( vCount!=vertices.size() ) MK_WARN( "vertex counts don't match:" , vertices.size() , " != " , vCount );
			}
		}
	}
	else
	{
		if( Out.set )
		{
			if( polygons.size() ) WriteMesh( Out.value , ASCII.set ? PLY_ASCII : ft , vertexDataFactory , vertices , polygons , comments );
			else WritePoints( Out.value , ASCII.set ? PLY_ASCII : ft , vertexDataFactory , vertices , comments );
		}
	}
}
int main( int argc , char* argv[] )
{
	typedef float Real;
	static constexpr unsigned int Dim = 3;

	CmdLineParse( argc-1 , &argv[1] , params );
#ifdef ARRAY_DEBUG
	MK_WARN( "Array debugging enabled" );
#endif // ARRAY_DEBUG

	if( !In.set )
	{
		ShowUsage( argv[0] );
		return EXIT_FAILURE;
	}
	Timer timer;

	bool isPly;
	for( int i=0 ; i<In.count ; i++ )
	{
		char *ext = GetFileExtension( In.values[i] );
		bool _isPly = strcasecmp( ext , "ply" )==0;
		if( !i ) isPly = _isPly;
		else if( isPly!=_isPly ) MK_THROW( "All files must be of the same type" );
		delete[] ext;
	}
	if( isPly )
	{
		VertexFactory::PositionFactory< Real , Dim > factory;
		bool *readFlags = new bool[ factory.plyReadNum() ];
		VertexFactory::DynamicFactory< Real > *remainingProperties = NULL;
		for( int i=0 ; i<In.count ; i++ )
		{
			std::vector< PlyProperty > unprocessedProperties;
			PLY::ReadVertexHeader( In.values[i] , factory , readFlags , unprocessedProperties );
			if( !factory.plyValidReadProperties( readFlags ) ) MK_THROW( "Ply file does not contain positions" );
			VertexFactory::DynamicFactory< Real > _remainingProperties( unprocessedProperties );
			if( !i ) remainingProperties = new VertexFactory::DynamicFactory< Real >( _remainingProperties );
			else if( (*remainingProperties)!=(_remainingProperties) ) MK_THROW( "Remaining properties differ" );
		}
		delete[] readFlags;
		if( !remainingProperties || !remainingProperties->size() ) Execute( VertexFactory::EmptyFactory< Real >() );
		else Execute( *remainingProperties );
		delete remainingProperties;
	}
	else
	{
		if( Values.set )
			if( Colors.set )
				if( !NoNormals.set ) Execute( VertexFactory::Factory< Real , VertexFactory::ValueFactory< Real > , VertexFactory::NormalFactory< Real , Dim > , VertexFactory::RGBColorFactory< Real > >() );
				else                 Execute( VertexFactory::Factory< Real , VertexFactory::ValueFactory< Real > ,                                              VertexFactory::RGBColorFactory< Real > >() );
			else
				if( !NoNormals.set ) Execute( VertexFactory::Factory< Real , VertexFactory::ValueFactory< Real > , VertexFactory::NormalFactory< Real , Dim >                                     >() );
				else                 Execute(                                VertexFactory::ValueFactory< Real >                                                                                 () );
		else
			if( Colors.set )
				if( !NoNormals.set ) Execute( VertexFactory::Factory< Real ,                                     VertexFactory::NormalFactory< Real , Dim > , VertexFactory::RGBColorFactory< Real > >() );
				else                 Execute(                                                                                                                 VertexFactory::RGBColorFactory< Real >  () );
			else
				if( !NoNormals.set ) Execute(                                                                  VertexFactory::NormalFactory< Real , Dim >                                      () );
				else                 Execute( VertexFactory::EmptyFactory< Real >() );
	}

	printf( "Time (Wall/CPU): %.2f / %.2f\n" , timer.wallTime() , timer.cpuTime() );
	printf( "Peak Memory (MB): %d\n" , MemoryInfo::PeakMemoryUsageMB() );

	return EXIT_SUCCESS;
}
