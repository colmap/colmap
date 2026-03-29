/*
Copyright (c) 2024, Michael Kazhdan
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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <algorithm>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include "Ply.h"
#include "CmdLineParser.h"
#include "Geometry.h"
#include "VertexFactory.h"
#include "MyMiscellany.h"
#include "DataStream.imp.h"

using namespace PoissonRecon;

CmdLineParameter< std::string >
	In( "in" ) ,
	Out( "out" );

CmdLineParameter< unsigned int >
	Res( "res" , 12 ) ,
	PointsToKeep( "keep" );

CmdLineParameter< float >
	Scale( "scale" , 0.005f ) ,
	Fraction( "fraction" , 1.f );

CmdLineReadable
	Verbose( "verbose" );

CmdLineParameter< float >
	LengthToRadiusExponent( "lExp" , 0.66f );

CmdLineReadable* params[] =
{
	&In ,
	&Out ,
	&Scale ,
	&Res ,
	&PointsToKeep ,
	&Fraction ,
	&LengthToRadiusExponent ,
	&Verbose ,
	NULL
};


void ShowUsage( char* ex )
{
	printf( "Usage: %s\n" , ex );
	printf( "\t --%s <input mesh>\n" , In.name );
	printf( "\t --%s <ouBtput mesh>\n" , Out.name );
	printf( "\t[--%s <disk size>=%f]\n" , Scale.name , Scale.value );
	printf( "\t[--%s <disk resolution>=%d]\n" , Res.name , Res.value );
	printf( "\t[--%s <points to keep>]\n" , PointsToKeep.name );
	printf( "\t[--%s <fraction to keep>=%f]\n" , Fraction.name , Fraction.value );
	printf( "\t[--%s <length to radius exponent>=%f]\n" , LengthToRadiusExponent.name , LengthToRadiusExponent.value );
	printf( "\t[--%s]\n" , Verbose.name );
}

int main( int argc , char* argv[] )
{
	auto SampleRadius = []( float length , float radius ){ return radius * (float)pow(length,0.66); };

	auto NormalColor = []( Point< float , 3 > n )
		{
			Point< float , 3 > color =
				(
				( ( n[0]<0 ) ? Point< float , 3 >(     0 , -n[0] , -n[0] ) : Point< float , 3 >( n[0] ,    0 ,    0 ) ) +
				( ( n[1]<0 ) ? Point< float , 3 >( -n[1] ,     0 , -n[1] ) : Point< float , 3 >(    0 , n[1] ,    0 ) ) +
				( ( n[2]<0 ) ? Point< float , 3 >( -n[2] , -n[2] ,     0 ) : Point< float , 3 >(    0 ,    0 , n[2] ) )
				) * 255;
			color[0] = std::min< float >( 255.f , color[0] );
			color[1] = std::min< float >( 255.f , color[1] );
			color[2] = std::min< float >( 255.f , color[2] );
			return color;
		};

	CmdLineParse( argc-1 , &argv[1] , params );
	if( !In.set || !Out.set )
	{
		ShowUsage( argv [0] );
		return EXIT_FAILURE;
	}

	if( Verbose.set )
	{
		std::cout << "***********************************************************" << std::endl;
		std::cout << "***********************************************************" << std::endl;
		std::cout << "** Running Points to Disks (Version " << ADAPTIVE_SOLVERS_VERSION << ") **" << std::endl;
		std::cout << "***********************************************************" << std::endl;
		std::cout << "***********************************************************" << std::endl;
	}

	if( PointsToKeep.set && Fraction.set )
	{
		MK_WARN( "One of --" , PointsToKeep.name , " and --" , Fraction.name , " should be set. Using --" , PointsToKeep.name );
		Fraction.set = false;
	}

	using InFactory = VertexFactory::Factory< float , VertexFactory::PositionFactory< float , 3 > , VertexFactory::NormalFactory< float , 3 > , VertexFactory::RGBColorFactory< float > >;
	using InVertex = typename InFactory::VertexType;
	std::vector< InVertex > vertices;

	bool hasNormals = true;
	bool hasColors = false;
	int fileType = PLY_BINARY_NATIVE;
	{
		std::vector< std::vector< int > > polygons;
		char *ext = GetFileExtension( In.value.c_str() );

		if( !strcasecmp( ext , "bnpts" ) )
		{
			using Factory = VertexFactory::Factory< float , VertexFactory::PositionFactory< float , 3 > , VertexFactory::NormalFactory< float , 3 > >;
			using Vertex = typename Factory::VertexType;
			Factory factory;
			BinaryInputDataStream< Factory > pointStream( In.value.c_str() , factory );
			Vertex v;
			while( pointStream.read( v ) )
			{
				InVertex _v;
				_v.template get<0>() = v.template get<0>();
				_v.template get<1>() = v.template get<1>();
				vertices.push_back( _v );
			}
		}
		else if( !strcasecmp( ext , "ply"   ) )
		{
			InFactory inFactory;
			bool *readFlags = new bool[ inFactory.plyReadNum() ];
			int file_type;
			std::vector< std::string > comments;
			PLY::ReadPolygons( In.value , inFactory , vertices , polygons , file_type , comments , readFlags );
			hasNormals = readFlags[3] && readFlags[4] && readFlags[5];
			hasColors = ( readFlags[6] && readFlags[7] && readFlags[8] ) || ( readFlags[9] && readFlags[10] && readFlags[11] );
			delete[] readFlags;
		}
		else
		{
			using Factory = VertexFactory::Factory< float , VertexFactory::PositionFactory< float , 3 > , VertexFactory::NormalFactory< float , 3 > >;
			using Vertex = typename Factory::VertexType;
			Factory factory;
			ASCIIInputDataStream< Factory > pointStream( In.value.c_str() , factory );
			Vertex v;
			while( pointStream.read( v ) )
			{
				InVertex _v;
				_v.template get<0>() = v.template get<0>();
				_v.template get<1>() = v.template get<1>();
				vertices.push_back( _v );
			}
		}
		delete[] ext;
	}
	if( Verbose.set ) std::cout << "Input points: " << vertices.size() << std::endl;
	if( PointsToKeep.set && PointsToKeep.value>vertices.size() )
	{
		MK_WARN( "--" , PointsToKeep.name , " value exceeds number of points: " , PointsToKeep.value , " > " , vertices.size() );
		PointsToKeep.value = (unsigned int)vertices.size();
	}

	if( !hasNormals ) MK_THROW( "Input is not oriented" );
	
	if( PointsToKeep.set )
	{
		std::vector< InVertex > _vertices( PointsToKeep.value );
		for( unsigned int i=0 ; i<PointsToKeep.value ; i++ )
		{
			int idx = (int)( ( i * vertices.size() ) / PointsToKeep.value );
			_vertices[i] = vertices[idx];
		}
		vertices = _vertices;
	}
	else if( Fraction.set )
	{
		std::vector< InVertex > _vertices;
		for( unsigned int i=0 ; i<vertices.size() ; i++ ) if( Random< float >()<=Fraction.value ) _vertices.push_back( vertices[i] );
		vertices = _vertices;
	}

	Point< float , 3 > min , max;
	min = max = vertices[0].template get<0>();
	for( int i=0 ; i<vertices.size() ; i++ ) for( int j=0 ; j<3 ; j++ )
	{
		if( vertices[i].template get<0>()[j]<min[j] ) min[j] = vertices[i].template get<0>()[j];
		if( vertices[i].template get<0>()[j]>max[j] ) max[j] = vertices[i].template get<0>()[j];
	}
	float radius = (float)sqrt( Point< float , 3 >::SquareNorm( max-min ) ) * Scale.value;
	if( Verbose.set ) std::cout << "Scale -> Radius: " << Scale.value << " -> " << radius << std::endl;

	{
		std::vector< InVertex > verts;
		std::vector< std::vector< int > > polygons;

		verts.reserve( vertices.size() * Res.value );
		polygons.reserve( vertices.size() );

		auto ProcessPoint = [&]( Point< float , 3 > p , Point< float , 3 > n , Point< float , 3 > c )
			{
				unsigned int vIdx = (unsigned int)verts.size();
				Point< float , 3 > v1 , v2;
				double l = sqrt( Point< float , 3 >::SquareNorm( n ) );
				if( !l ) return;
				n /= (float)l;

				float radiusScale = (float)pow( l , LengthToRadiusExponent.value );
				v1 = Point< float , 3 >( -n[2] , 0 , n[0] );
				if( Point< float , 3 >::SquareNorm( v1 )<0.0000001 ) v1 = Point< float , 3 >( 1 , 0 , 0 );
				v2 = Point< float , 3 >::CrossProduct( n , v1 );
				v1 /= sqrt( Point< float , 3 >::SquareNorm( v1 ) );
				v2 /= sqrt( Point< float , 3 >::SquareNorm( v2 ) );

				InVertex v;
				v.template get<2>() = hasColors ? c : NormalColor( n );
				v.template get<1>() = n;

				for( unsigned int j=0 ; j<Res.value ; j++ )
				{
					double theta = 2.0 * M_PI * double(j) / Res.value;
					v.template get<0>() = p + ( v1 * (float)cos(theta) + v2 * (float)sin(theta) ) * radius * radiusScale;
					verts.push_back( v );
				}
				std::vector< int > polygon( Res.value );
				for( unsigned int j=0 ; j<Res.value ; j++ ) polygon[j] = vIdx + j;
				polygons.push_back( polygon );
			};

		for( unsigned int i=0 ; i<vertices.size() ; i++ ) ProcessPoint( vertices[i].template get<0>() , vertices[i].template get<1>() , vertices[i].template get<2>() );

		InFactory inFactory;
		std::vector< std::string > comments;
		PLY::WritePolygons( Out.value , inFactory , verts , polygons, fileType , comments );
	}
	return EXIT_SUCCESS;
}