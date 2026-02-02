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

template< unsigned int WindowDimension , unsigned int IterationDimensions , unsigned int CurrentIteration >
struct _Loop
{
	static_assert( IterationDimensions<=WindowDimension , "[ERROR] Iteration dimensions cannot excceed window dimension" );
	static_assert( CurrentIteration>0 , "[ERROR] Current iteration cannot be zero" );
	static_assert( CurrentIteration<=IterationDimensions , "[ERROR] Current iteration cannot exceed iteration dimensions" );

protected:
	static const int CurrentDimension = CurrentIteration + WindowDimension - IterationDimensions;
	friend struct Loop< WindowDimension , IterationDimensions >;
	friend struct _Loop< WindowDimension , IterationDimensions , CurrentIteration+1 >;


	///////////////////////////////
	// Single-threaded execution //
	///////////////////////////////
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( int begin , int end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin ; i<end ; i++ )
		{
			updateState( WindowDimension - CurrentDimension , i );
			if constexpr( CurrentIteration==1 ) function( w[i] ... );
			else _Loop< WindowDimension , IterationDimensions , CurrentIteration-1 >::Run( begin , end , updateState , function , w[i] ... );
		}
	}

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( const int* begin , const int* end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin[0] ; i<end[0] ; i++ )
		{
			updateState( WindowDimension - CurrentDimension , i );
			if constexpr( CurrentIteration==1 ) function( w[i] ... );
			else _Loop< WindowDimension , IterationDimensions , CurrentIteration-1 >::Run( begin+1 , end+1 , updateState , function , w[i] ... );
		}
	}

	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void Run( ParameterPack::UIntPack< Begin ... > begin , ParameterPack::UIntPack< End ... > end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=ParameterPack::UIntPack< Begin ... >::First ; i<ParameterPack::UIntPack< End ... >::First ; i++ )
		{
			updateState( WindowDimension - CurrentDimension , i );
			if constexpr( CurrentIteration==1 ) function( w[i] ... );
			else _Loop< WindowDimension , IterationDimensions , CurrentIteration-1 >::Run( typename ParameterPack::UIntPack< Begin ... >::Rest() , typename ParameterPack::UIntPack< End ... >::Rest() , updateState , function , w[i] ... );
		}
	}


	//////////////////////////////
	// Multi-threaded execution //
	//////////////////////////////
	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( int begin , int end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		ThreadPool::ParallelFor
		(
			begin , end ,
			[&]( unsigned int thread , size_t i )
			{
				updateState( thread , WindowDimension - CurrentDimension , i );
				if constexpr( CurrentIteration==1 ) function( thread , w[i] ... );
				else _Loop< WindowDimension , IterationDimensions , CurrentIteration-1 >::RunThread( begin , end , thread , updateState , function , w[i] ... );
			}
		);
	}

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( const int* begin , const int* end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		ThreadPool::ParallelFor
		(
			begin[0] , end[0] ,
			[&]( unsigned int thread , size_t i )
			{
				updateState( thread , WindowDimension - CurrentDimension , i );
				if constexpr( CurrentIteration==1 ) function( thread , w[i] ... );
				else _Loop< WindowDimension , IterationDimensions , CurrentIteration-1 >::RunThread( begin+1 , end+1 , thread , updateState , function , w[i] ... );
			}
		);
	}

	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunParallel( ParameterPack::UIntPack< Begin ... > begin , ParameterPack::UIntPack< End ... > end , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		ThreadPool::ParallelFor
		(
			ParameterPack::UIntPack< Begin ... >::First , ParameterPack::UIntPack< End ... >::First ,
			[&]( unsigned int thread , size_t i )
			{
				updateState( thread , WindowDimension - CurrentDimension , i );
				if constexpr( CurrentIteration==1 ) function( thread , w[i] ... );
				else _Loop< WindowDimension , IterationDimensions , CurrentIteration-1 >::RunThread( typename ParameterPack::UIntPack< Begin ... >::Rest() , typename ParameterPack::UIntPack< End ... >::Rest() , thread , updateState , function , w[i] ... );
			}
		);
	}


	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunThread( int begin , int end , unsigned int thread , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin ; i<end ; i++ )
		{
			updateState( thread , WindowDimension - CurrentDimension , i );
			if constexpr( CurrentIteration==1 ) function( thread , w[i] ... );
			else _Loop< WindowDimension , IterationDimensions , CurrentIteration-1 >::RunThread( begin , end , thread , updateState , function , w[i] ... );
		}
	}

	template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunThread( const int* begin , const int* end , unsigned int thread , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=begin[0] ; i<end[0] ; i++ )
		{
			updateState( thread , WindowDimension - CurrentDimension , i );
			if constexpr( CurrentIteration==1 ) function( thread , w[i] ... );
			else _Loop< WindowDimension , IterationDimensions , CurrentIteration-1 >::RunThread( begin+1 , end+1 , thread , updateState , function , w[i] ... );
		}
	}

	template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
	static void RunThread( ParameterPack::UIntPack< Begin ... > begin , ParameterPack::UIntPack< End ... > end , unsigned int thread , UpdateFunction& updateState , ProcessFunction& function , Windows ... w )
	{
		for( int i=ParameterPack::UIntPack< Begin ... >::First ; i<ParameterPack::UIntPack< End ... >::First ; i++ )
		{
			updateState( thread , WindowDimension - CurrentDimension , i );
			if constexpr( CurrentIteration==1 ) function( thread , w[i] ... );
			else _Loop< WindowDimension , IterationDimensions , CurrentIteration-1 >::RunThread( typename ParameterPack::UIntPack< Begin ... >::Rest() , typename ParameterPack::UIntPack< End ... >::Rest() , thread , updateState , function , w[i] ... );
		}
	}
};