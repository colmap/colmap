/*
Copyright (c) 2017, Michael Kazhdan
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
#ifndef MULTI_THREADING_INCLUDED
#define MULTI_THREADING_INCLUDED

#include <thread>
#include <vector>
#include <atomic>
#include <functional>
#include <future>
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

namespace PoissonRecon
{
	struct ThreadPool
	{
		enum ParallelType
		{
#ifdef _OPENMP
			OPEN_MP ,
#endif // _OPENMP
			ASYNC ,
			NONE
		};
		static const std::vector< std::string > ParallelNames;

		enum ScheduleType
		{
			STATIC ,
			DYNAMIC
		};
		static const std::vector< std::string > ScheduleNames;

		static unsigned int NumThreads( void ){ return _NumThreads; }
		static ParallelType ParallelizationType;
		static size_t ChunkSize;
		static ScheduleType Schedule;

		template< typename Function , typename ... Functions >
		static void ParallelSections( const Function &function , const Functions & ... functions )
		{
			std::vector< std::future< void > > futures;
			if constexpr( sizeof ... (Functions) )
			{
				futures.reserve( sizeof...(Functions) );
				_ParallelSections( futures , functions... );
			}
			function();
			for( unsigned int i=0 ; i<futures.size() ; i++ ) futures[i].get();
		}

		template< typename Function , typename ... Functions >
		static void ParallelSections( const Function &&function , const Functions && ... functions )
		{
			std::vector< std::future< void > > futures;
			if constexpr( sizeof ... (Functions) )
			{
				futures.reserve( sizeof...(Functions) );
				_ParallelSections( futures , std::move(functions)... );
			}
			function();
			for( unsigned int i=0 ; i<futures.size() ; i++ ) futures[i].get();
		}

		static void ParallelFor( size_t begin , size_t end , const std::function< void ( unsigned int , size_t ) > &iterationFunction , unsigned int numThreads=_NumThreads , ParallelType pType=ParallelizationType , ScheduleType schedule=Schedule , size_t chunkSize=ChunkSize )
		{
			if( begin>=end ) return;
			size_t range = end - begin;
			size_t chunks = ( range + chunkSize - 1 ) / chunkSize;
			std::atomic< size_t > index;
			index.store( 0 );

			// If the computation is serial, go ahead and run it
			if( pType==ParallelType::NONE || numThreads<=1 )
			{
				for( size_t i=begin ; i<end ; i++ ) iterationFunction( 0 , i );
				return;
			}

			// If the chunkSize is too large to satisfy all the threads, lower it
			if( range<=chunkSize*(numThreads-1) )
			{
				chunkSize = ( range + numThreads - 1 ) / numThreads;
				chunks = numThreads = (unsigned int)( ( range + chunkSize - 1 ) / chunkSize );
			}

			std::function< void (unsigned int , size_t ) > _ChunkFunction = [ &iterationFunction , begin , end , chunkSize ]( unsigned int thread , size_t chunk )
				{
					const size_t _begin = begin + chunkSize*chunk;
					const size_t _end = std::min< size_t >( end , _begin+chunkSize );
					for( size_t i=_begin ; i<_end ; i++ ) iterationFunction( thread , i );
				};
			std::function< void (unsigned int ) > _StaticThreadFunction = [ &_ChunkFunction , chunks , numThreads ]( unsigned int thread )
				{
					for( size_t chunk=thread ; chunk<chunks ; chunk+=numThreads ) _ChunkFunction( thread , chunk );
				};

			std::function< void (unsigned int ) > _DynamicThreadFunction = [ &_ChunkFunction , chunks , &index ]( unsigned int thread )
				{
					size_t chunk;
					while( ( chunk=index.fetch_add(1) )<chunks ) _ChunkFunction( thread , chunk );
				};

			std::function< void (unsigned int ) > ThreadFunction;
			if     ( schedule==ScheduleType::STATIC  ) ThreadFunction = _StaticThreadFunction;
			else if( schedule==ScheduleType::DYNAMIC ) ThreadFunction = _DynamicThreadFunction;

			if( false ){}
#ifdef _OPENMP
			else if( pType==ParallelType::OPEN_MP )
			{
				if( schedule==ScheduleType::STATIC )
#pragma omp parallel for num_threads( numThreads ) schedule( static , 1 )
					for( int c=0 ; c<chunks ; c++ ) _ChunkFunction( omp_get_thread_num() , c );
				else if( schedule==ScheduleType::DYNAMIC )
#pragma omp parallel for num_threads( numThreads ) schedule( dynamic , 1 )
					for( int c=0 ; c<chunks ; c++ ) _ChunkFunction( omp_get_thread_num() , c );
			}
#endif // _OPENMP
			else if( pType==ParallelType::ASYNC )
			{
				static std::vector< std::future< void > > futures;
				futures.resize( numThreads-1 );
				for( unsigned int t=1 ; t<numThreads ; t++ ) futures[t-1] = std::async( std::launch::async , ThreadFunction , t );
				ThreadFunction( 0 );
				for( unsigned int t=1 ; t<numThreads ; t++ ) futures[t-1].get();
			}
		}

	private:
		static unsigned int _NumThreads;

		template< typename Function , typename ... Functions >
		static void _ParallelSections( std::vector< std::future< void > > &futures , const Function &function , const Functions & ... functions )
		{
			futures.push_back( std::async( std::launch::async , function ) );
			if constexpr( sizeof...(Functions) ) _ParallelSections( futures , functions... );
		}

		template< typename Function , typename ... Functions >
		static void _ParallelSections( std::vector< std::future< void > > &futures , const Function &&function , const Functions && ... functions )
		{
			futures.push_back( std::async( std::launch::async , function ) );
			if constexpr( sizeof...(Functions) ) _ParallelSections( futures , std::move(functions)... );
		}
	};

	inline ThreadPool::ParallelType ThreadPool::ParallelizationType = ThreadPool::ParallelType::NONE;
	inline unsigned int ThreadPool::_NumThreads = std::thread::hardware_concurrency();
	inline ThreadPool::ScheduleType ThreadPool::Schedule = ThreadPool::DYNAMIC;
	inline size_t ThreadPool::ChunkSize = 128;

	const inline std::vector< std::string > ThreadPool::ParallelNames =
	{
#ifdef _OPENMP
		"open mp" ,
#endif // _OPENMP
		"async" ,
		"none"
	};
	const inline std::vector< std::string > ThreadPool::ScheduleNames = { "static" , "dynamic" };
}
#endif // MULTI_THREADING_INCLUDED
