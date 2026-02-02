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

CmdLineParameters< std::string >
	ConfidenceNames( "cNames" );

CmdLineParameters< float >
	ConfidenceExponents( "cExps" );
	

CmdLineReadable
	Verbose( "verbose" );

CmdLineReadable* params[] =
{
	&In ,
	&Out ,
	&ConfidenceNames , 
	&ConfidenceExponents ,
	&Verbose ,
	NULL
};


void ShowUsage( char* ex )
{
	printf( "Usage: %s\n" , ex );
	printf( "\t --%s <input points>\n" , In.name );
	printf( "\t --%s <output points>\n" , Out.name );
	printf( "\t --%s <confidence counts, confidence name1, confidence name2...>\n" , ConfidenceNames.name );
	printf( "\t --%s <confidence counts, confidence exp1, confidence exp2...>\n" , ConfidenceExponents.name );
	printf( "\t[--%s]\n" , Verbose.name );
}

template< typename Real , unsigned int Dim , typename Factory >
void Execute( Factory factory )
{
	auto Scale = [&]( Point< Real > scales )
		{
			Real s = (Real)1.;
			for( unsigned int d=0 ; d<scales.dim() ; d++ ) s *= (Real)pow( scales[d] , ConfidenceExponents.values[d] );
			return s;
		};

	size_t count;
	PLYInputDataStream< Factory > inPointStream( In.value.c_str() , factory , count );
	PLYOutputDataStream< Factory > outPointStream( Out.value.c_str() , factory , count );

	if( Verbose.set ) std::cout << "Points: " << count << std::endl;

	typename Factory::VertexType v = factory();
	while( inPointStream.read( v ) )
	{
		v.template get<1>() *= Scale( v.template get<2>() );
		outPointStream.write( v );
	}
};

template< typename Real , unsigned int Dim >
void Execute( void )
{
	std::vector< std::pair< std::string , VertexFactory::TypeOnDisk > > namesAndTypesOnDisk( ConfidenceNames.count );
	for( unsigned int i=0 ; i<ConfidenceNames.count ; i++ )
	{
		namesAndTypesOnDisk[i].first = ConfidenceNames.values[i];
		namesAndTypesOnDisk[i].second = VertexFactory::template GetTypeOnDisk< Real >();
	}

	VertexFactory::PositionFactory< Real , Dim > positionFactory;
	VertexFactory::NormalFactory< Real , Dim > normalFactory;
	VertexFactory::DynamicFactory< Real > scaleFactory( namesAndTypesOnDisk );

	typedef VertexFactory::Factory< Real , typename VertexFactory::PositionFactory< Real , Dim > , typename VertexFactory::NormalFactory< Real , Dim > , VertexFactory::DynamicFactory< Real > > Factory;
	Factory factory( positionFactory , normalFactory , scaleFactory );

	bool *readFlags = new bool[ factory.plyReadNum() ];
	std::vector< PlyProperty > unprocessedProperties;

	PLY::ReadVertexHeader( In.value , factory , readFlags , unprocessedProperties );
	if( !factory.template plyValidReadProperties<0>( readFlags ) ) MK_THROW( "Ply file does not contain positions" );
	if( !factory.template plyValidReadProperties<1>( readFlags ) ) MK_THROW( "Ply file does not contain normals" );
	if( !factory.template plyValidReadProperties<2>( readFlags ) ) MK_THROW( "Ply file does not contain scales" );
	delete[] readFlags;

	if( Verbose.set && unprocessedProperties.size() )
	{
		std::cout << "Unprocessed properties:" << std::endl;
		for( unsigned int i=0 ; i<unprocessedProperties.size() ; i++ ) std::cout << "\t" << unprocessedProperties[i] << std::endl;
	}

	if( unprocessedProperties.size() )
	{
		using _Factory = VertexFactory::Factory< Real , typename VertexFactory::PositionFactory< Real , Dim > , typename VertexFactory::NormalFactory< Real , Dim > , VertexFactory::DynamicFactory< Real > , VertexFactory::DynamicFactory< Real > >;
		Execute< Real , Dim >( _Factory( positionFactory , normalFactory , scaleFactory , VertexFactory::DynamicFactory< Real >( unprocessedProperties ) ) );
	}
	else
	{
		using _Factory = VertexFactory::Factory< Real , typename VertexFactory::PositionFactory< Real , Dim > , typename VertexFactory::NormalFactory< Real , Dim > , VertexFactory::DynamicFactory< Real > >;
		Execute< Real , Dim >( _Factory( positionFactory , normalFactory , scaleFactory ) );
	}
}

int main( int argc , char* argv[] )
{
	CmdLineParse( argc-1 , &argv[1] , params );
	if( !In.set || !Out.set || !ConfidenceNames.set || !ConfidenceExponents.set )
	{
		ShowUsage( argv [0] );
		return EXIT_FAILURE;
	}

	if( ConfidenceNames.count!=ConfidenceExponents.count )
		MK_THROW( "Number of confidence names and exponents does not match: " , ConfidenceNames.count , " != " , ConfidenceExponents.count );

	if( Verbose.set )
	{
		std::cout << "*******************************************" << std::endl;
		std::cout << "*******************************************" << std::endl;
		std::cout << "** Running Scale Normals (Version " << ADAPTIVE_SOLVERS_VERSION << ") **" << std::endl;
		std::cout << "*******************************************" << std::endl;
		std::cout << "*******************************************" << std::endl;
	}

	Execute< float , 3 >();


	return EXIT_SUCCESS;
}