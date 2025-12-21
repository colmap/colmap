/*
Copyright (c) 2023, Michael Kazhdan
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

#define BIG_DATA
#include "PreProcessor.h"

#include "Socket.h"
#include "PoissonReconClientServer.h"
#include "PointPartitionClientServer.h"
#include "MergePlyClientServer.h"

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "MyMiscellany.h"
#include "Reconstructors.h"
#include "CmdLineParser.h"

#define DEFAULT_DIMENSION 3

using namespace PoissonRecon;

CmdLineParameter< std::string >
	Address( "address" , "127.0.0.1" );

CmdLineParameter< int >
	MaxMemoryGB( "maxMemory" , 0 ) ,
	ParallelType( "parallel" , 0 ) ,
	ScheduleType( "schedule" , (int)ThreadPool::Schedule ) ,
	ThreadChunkSize( "chunkSize" , (int)ThreadPool::ChunkSize ) ,
	MultiClient( "multi" , 1 ) ,
	Port( "port" , 0 ) ,
	PeakMemorySampleMS( "sampleMS" , 10 );

CmdLineReadable
	Pause( "pause" );


CmdLineReadable* params[] =
{
	&Port , &MultiClient , &Address ,
	&MaxMemoryGB , &ParallelType , &ScheduleType , &ThreadChunkSize ,
	&Pause ,
	&PeakMemorySampleMS ,
	NULL
};

void ShowUsage( char* ex )
{
	printf( "Usage: %s\n" , ex );
	printf( "\t --%s <server port>\n" , Port.name );
	printf( "\t[--%s <multiplicity of serial sub-clients>=%d]\n" , MultiClient.name , MultiClient.value );
	printf( "\t[--%s <server connection address>=%s]\n" , Address.name , Address.value.c_str() );
	printf( "\t[--%s <parallel type>=%d]\n" , ParallelType.name , ParallelType.value );
	for( size_t i=0 ; i<ThreadPool::ParallelNames.size() ; i++ ) printf( "\t\t%d] %s\n" , (int)i , ThreadPool::ParallelNames[i].c_str() );
	printf( "\t[--%s <schedue type>=%d]\n" , ScheduleType.name , ScheduleType.value );
	for( size_t i=0 ; i<ThreadPool::ScheduleNames.size() ; i++ ) printf( "\t\t%d] %s\n" , (int)i , ThreadPool::ScheduleNames[i].c_str() );
	printf( "\t[--%s <thread chunk size>=%d]\n" , ThreadChunkSize.name , ThreadChunkSize.value );
	printf( "\t[--%s <peak memory sampling rate (ms)>=%d]\n" , PeakMemorySampleMS.name , PeakMemorySampleMS.value );
	printf( "\t[--%s <maximum memory (in GB)>=%d]\n" , MaxMemoryGB.name , MaxMemoryGB.value );
	printf( "\t[--%s]\n" , Pause.name );
}

template< typename Real , unsigned int Dim >
void Partition( std::vector< Socket > &serverSockets )
{
	PointPartitionClientServer::RunClients< Real , Dim >( serverSockets );
	unsigned int peakMem = MemoryInfo::PeakMemoryUsageMB();
	for( unsigned int i=0 ; i<serverSockets.size() ; i++ ) SocketStream( serverSockets[i] ).write( peakMem );
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
void Reconstruct( std::vector< Socket > &serverSockets )
{
	PoissonReconClientServer::RunClient< Real , Dim , BType , Degree >( serverSockets , PeakMemorySampleMS.value );
	unsigned int peakMem = MemoryInfo::PeakMemoryUsageMB();
	for( unsigned int i=0 ; i<serverSockets.size() ; i++ ) SocketStream( serverSockets[i] ).write( peakMem );
}

template< typename Real , unsigned int Dim >
void Merge( std::vector< Socket > &serverSockets )
{
	MergePlyClientServer::RunClients< Real , Dim >( serverSockets , PeakMemorySampleMS.value );
	unsigned int peakMem = MemoryInfo::PeakMemoryUsageMB();
	for( unsigned int i=0 ; i<serverSockets.size() ; i++ ) SocketStream( serverSockets[i] ).write( peakMem );
}

#ifdef FAST_COMPILE
#else // !FAST_COMPILE
template< typename Real , unsigned int Dim , BoundaryType BType >
void Reconstruct( unsigned int degree , std::vector< Socket > &serverSockets )
{
	switch( degree )
	{
		case 1: return Reconstruct< Real , Dim , BType , 1 >( serverSockets );
		case 2: return Reconstruct< Real , Dim , BType , 2 >( serverSockets );
		default: MK_THROW( "Only B-Splines of degree 1 - 2 are supported" );
	}
}

template< typename Real , unsigned int Dim >
void Reconstruct( BoundaryType bType , unsigned int degree , std::vector< Socket > &serverSockets )
{
	switch( bType )
	{
		case BOUNDARY_FREE:      return Reconstruct< Real , Dim , BOUNDARY_FREE      >( degree , serverSockets );
		case BOUNDARY_NEUMANN:   return Reconstruct< Real , Dim , BOUNDARY_NEUMANN   >( degree , serverSockets );
		case BOUNDARY_DIRICHLET: return Reconstruct< Real , Dim , BOUNDARY_DIRICHLET >( degree , serverSockets );
		default: MK_THROW( "Not a valid boundary type: " , bType );
	}
}

template< typename Real , unsigned int Dim >
void Reconstruct( std::vector< Socket > &serverSockets )
{
	BoundaryType bType;
	unsigned int degree;
	for( unsigned int i=0 ; i<serverSockets.size() ; i++ )
	{
		if( !SocketStream( serverSockets[i] ).read( bType ) ) MK_THROW( "Failed to read boundary-type" );
		if( !SocketStream( serverSockets[i] ).read( degree ) ) MK_THROW( "Failed to read degree" );
	}
	Reconstruct< Real , Dim >( bType , degree , serverSockets );
}

#endif // FAST_COMPILE

int main( int argc , char* argv[] )
{
#ifdef ARRAY_DEBUG
	MK_WARN( "Array debugging enabled" );
#endif // ARRAY_DEBUG
#ifdef USE_DOUBLE
	typedef double Real;
#else // !USE_DOUBLE
	typedef float  Real;
#endif // USE_DOUBLE
	static const unsigned int Dim = DEFAULT_DIMENSION;

	Timer timer;
	CmdLineParse( argc-1 , &argv[1] , params );

	if( !Port.set )
	{
		ShowUsage( argv[0] );
		return 0;
	}

	if( MaxMemoryGB.value>0 ) SetPeakMemoryMB( MaxMemoryGB.value<<10 );
	ThreadPool::ChunkSize = ThreadChunkSize.value;
	ThreadPool::Schedule = (ThreadPool::ScheduleType)ScheduleType.value;
	ThreadPool::ParallelizationType= (ThreadPool::ParallelType)ParallelType.value;

	std::vector< Socket > serverSockets( MultiClient.value , NULL );
	for( unsigned int i=0 ; i<(unsigned int)MultiClient.value ; i++ ) serverSockets[i] = GetConnectSocket( Address.value.c_str() , Port.value , SOCKET_CONNECT_WAIT , false );

	{
		Partition< Real , Dim >( serverSockets );

#ifdef FAST_COMPILE
		Reconstruct< Real , Dim , Reconstructor::Poisson::DefaultFEMBoundary , Reconstructor::Poisson::DefaultFEMDegree >( serverSockets );
#else // !FAST_COMPILE
		Reconstruct< Real , Dim >( serverSockets );
#endif // FAST_COMPILE

		if constexpr( Dim==3 ) Merge< Real , Dim >( serverSockets );

		for( unsigned int i=0 ; i<serverSockets.size() ; i++ ) CloseSocket( serverSockets[i] );
	}

	if( Pause.set )
	{
		std::cout << "Hit [ENTER] to terminate" << std::endl;
		std::string foo;
		std::getline( std::cin , foo );
	}

	return EXIT_SUCCESS;
}
