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

#ifndef DATA_STREAM_INCLUDED
#define DATA_STREAM_INCLUDED

#include <mutex>
#include <vector>
#include <atomic>
#include "Geometry.h"

namespace PoissonRecon
{
	// The input/output stream types
	template< typename ... Data > struct  InputDataStream;
	template< typename ... Data > struct OutputDataStream;

	// Input/output streams, internally represented by an std::vector< Input/OutputDataStream< Data ... > >
	template< typename ... Data > struct  MultiInputDataStream;
	template< typename ... Data > struct MultiOutputDataStream;

	// Class for de-interleaving output
	template< typename ... Data > using                    OutputIndexedDataStream =      OutputDataStream< size_t , Data ... >;
	template< typename ... Data > using               MultiOutputIndexedDataStream = MultiOutputDataStream< size_t , Data ... >;
	template< typename ... Data > struct DeInterleavedMultiOutputIndexedDataStream;

	// Classes for re-interleaving input
	template< typename ... Data > using                  InputIndexedDataStream =      InputDataStream< size_t , Data ... >;
	template< typename ... Data > using             MultiInputIndexedDataStream = MultiInputDataStream< size_t , Data ... >;
	template< typename ... Data > struct InterleavedMultiInputIndexedDataStream;

	// Classes for converting one stream type to another
	template< typename IDataTuple , typename XDataTuple > struct  InputDataStreamConverter;
	template< typename IDataTuple , typename XDataTuple > struct OutputDataStreamConverter;

	template< typename ... Data >
	struct TupleConverter
	{
		static void FromTuple( const std::tuple< Data ... > &dTuple , Data& ... d );
		static void ToTuple( std::tuple< Data ... > &dTuple , const Data& ... d );
	protected:
		template< unsigned int I , typename _Data , typename ... _Datas >
		static void _FromTuple( const std::tuple< Data ... > &dTuple , _Data &d , _Datas& ... ds );
		template< unsigned int I , typename _Data , typename ... _Datas >
		static void _ToTuple( std::tuple< Data ... > &dTuple , const _Data &d , const _Datas& ... ds );
	};

	template< typename Real , typename ... Data >
	struct DirectSumConverter
	{
		static void FromDirectSum( const DirectSum< Real ,  Data ... > &dSum , Data& ... d );
		static void ToDirectSum( DirectSum< Real , Data ... > &dSum , const Data& ... d );
	protected:
		template< unsigned int I , typename _Data , typename ... _Datas >
		static void _FromDirectSum( const DirectSum<Real , Data ... > &dSum , _Data &d , _Datas& ... ds );
		template< unsigned int I , typename _Data , typename ... _Datas >
		static void _ToDirectSum( DirectSum< Real , Data ... > &dSum , const _Data &d , const _Datas& ... ds );
	};

	////////////////////////
	// Input data streams //
	////////////////////////

	// An input stream containing "Data" types
	// Supporting:
	// -- Resetting the stream to the start
	// -- Trying to read the next element from the stream
	template< typename ... Data >
	struct InputDataStream
	{
		friend struct MultiInputDataStream< Data...  >;

		virtual ~InputDataStream( void ){}

		// Reset to the start of the stream
		virtual void reset( void ) = 0;

		// Read the next datum from the stream
		virtual bool read(                       Data& ... d ) = 0;
		virtual bool read( unsigned int thread , Data& ... d );

		bool read(                       std::tuple< Data ... > &data );
		bool read( unsigned int thread , std::tuple< Data ... > &data );

		template< typename Real >
		bool read(                       DirectSum< Real , Data ... > &data );
		template< typename Real >
		bool read( unsigned int thread , DirectSum< Real , Data ... > &data );

	protected:
		std::mutex _insertionMutex;

		template< typename ... _Data >
		bool _read( std::tuple< Data ... > &data , _Data& ... _data );
		template< typename ... _Data >
		bool _read( unsigned int thread , std::tuple< Data ... > &data , _Data& ... _data );
		template< typename Real , typename ... _Data >
		bool _read( DirectSum< Real , Data ... > &data , _Data& ... _data );
		template< typename Real , typename ... _Data >
		bool _read( unsigned int thread , DirectSum< Real , Data ... > &data , _Data& ... _data );
	};


	/////////////////////////
	// Output data streams //
	/////////////////////////
	// 
	// An output stream containing "Data" types
	// Supporting:
	// -- Writing the next element to the stream
	// -- Pushing the next element to the stream (and getting the count of elements pushed so far)
	template< typename ... Data >
	struct OutputDataStream
	{
		friend struct MultiOutputDataStream< Data ... >;

		OutputDataStream( void ){}
		virtual ~OutputDataStream( void ){}

		virtual size_t size( void ) const = 0;

		virtual size_t write(                       const Data& ... d ) = 0;
		virtual size_t write( unsigned int thread , const Data& ... d );

		size_t write(                       const std::tuple< Data ... > &data );
		size_t write( unsigned int thread , const std::tuple< Data ... > &data );

		template< typename Real > size_t write(                       const DirectSum< Real , Data ... > &data );
		template< typename Real > size_t write( unsigned int thread , const DirectSum< Real , Data ... > &data );

	protected:
		std::mutex _insertionMutex;

		template< typename ... _Data > size_t _write(                       const std::tuple< Data ... > &data , const _Data& ... _data );
		template< typename ... _Data > size_t _write( unsigned int thread , const std::tuple< Data ... > &data , const _Data& ... _data );

		template< typename Real , typename ... _Data > size_t _write(                       const DirectSum< Real , Data ... > &data , const _Data& ... _data );
		template< typename Real , typename ... _Data > size_t _write( unsigned int thread , const DirectSum< Real , Data ... > &data , const _Data& ... _data );
	};

	////////////////////////////////
	// Multiple input data stream //
	////////////////////////////////
	template< typename ...Data >
	struct MultiInputDataStream : public InputDataStream< Data ... >
	{
		using InputDataStream< Data ... >::read;

		MultiInputDataStream( InputDataStream< Data ... > **streams , size_t N );
		MultiInputDataStream( const std::vector< InputDataStream< Data ... > * > &streams );

		void reset( void );

		unsigned int numStreams( void ) const;

		bool read( unsigned int t , Data& ... d );
		bool read(                  Data& ... d );

	protected:
		std::vector< InputDataStream< Data ... > * > _streams;
		unsigned int _current;
	};


	/////////////////////////////////
	// Multiple output data stream //
	/////////////////////////////////
	template< typename ... Data >
	struct MultiOutputDataStream : public OutputDataStream< Data ... >
	{
		using OutputDataStream< Data ... >::write;

		MultiOutputDataStream( OutputDataStream< Data ... > **streams , size_t N );
		MultiOutputDataStream( const std::vector< OutputDataStream< Data ... > * > &streams );

		unsigned int numStreams( void ) const;

		size_t size( void ) const { return _size; }

		size_t write(                  const Data& ... d );
		size_t write( unsigned int t , const Data& ... d );

	protected:
		std::vector< OutputDataStream< Data ... > * > _streams;
		std::atomic< size_t > _size;
	};

	/////////////////////////////////
	// Input data stream converter //
	/////////////////////////////////
	template< typename ... IData , typename ... XData >
	struct InputDataStreamConverter< std::tuple< IData ... > , std::tuple< XData ... > > : public InputDataStream< XData ... >
	{
		using InputDataStream< XData ... >::read;

		InputDataStreamConverter( InputDataStream< IData ... > &stream , std::function< std::tuple< XData ... > ( const std::tuple< IData ... >& ) > converter , IData ... zero );
		void reset( void );
		bool read(                       XData& ... xData );
		bool read( unsigned int thread , XData& ... xData );

	protected:
		InputDataStream< IData ... > &_stream;
		std::function< std::tuple< XData ... > ( const std::tuple< IData ... >& ) > _converter;
		std::tuple< IData ... > _iZero;
	};

	//////////////////////////////////
	// Output data stream converter //
	//////////////////////////////////
	template< typename ... IData , typename ... XData >
	struct OutputDataStreamConverter< std::tuple< IData ... > , std::tuple< XData ... > > : public OutputDataStream< XData ... >
	{
		using OutputDataStream< XData ... >::write;

		OutputDataStreamConverter( OutputDataStream< IData ... > &stream , std::function< std::tuple< IData ... > ( const std::tuple< XData ... >& ) > converter );

		size_t size( void ) const { return _stream.size(); }

		size_t write(                       const XData& ... xData );
		size_t write( unsigned int thread , const XData& ... xData );
	protected:
		OutputDataStream< IData ... > &_stream;
		std::function< std::tuple< IData ... > ( const std::tuple< XData ... >& ) > _converter;
	};


	////////////////////////////////////////////////
	// De-interleaved multiple output data stream //
	////////////////////////////////////////////////
	template< typename ... Data >
	struct DeInterleavedMultiOutputIndexedDataStream : public OutputDataStream< Data ... >
	{
		using OutputDataStream< Data ... >::write;

		DeInterleavedMultiOutputIndexedDataStream( OutputIndexedDataStream< Data ... > **streams , size_t N );
		DeInterleavedMultiOutputIndexedDataStream( const std::vector< OutputIndexedDataStream<  Data ... > * > &streams );

		unsigned int numStreams( void ) const;

		size_t size( void ) const { return _size; }

		size_t write(                       const Data& ... d );
		size_t write( unsigned int thread , const Data& ... d );

	protected:
		MultiOutputIndexedDataStream< Data ... > _multiStream;
		std::atomic< size_t > _size;
	};


	////////////////////////////////////////////
	// Interleaved multiple input data stream //
	////////////////////////////////////////////

	template< typename ... Data >
	struct InterleavedMultiInputIndexedDataStream : public InputDataStream< Data ... >
	{
		using InputDataStream< Data... >::read;

		InterleavedMultiInputIndexedDataStream( InputIndexedDataStream< Data ... > **streams , size_t N );
		InterleavedMultiInputIndexedDataStream( const std::vector< InputIndexedDataStream<  Data ... > * > &streams );
		void reset( void );
		bool read( Data& ... d );

	protected:

		struct _NextValue
		{
			std::tuple< size_t , Data ... > data;
			unsigned int streamIndex;
			bool validData;

			// Returns true if v1>v2 so that it's a min-heap
			static bool Compare( const _NextValue &v1 , const _NextValue &v2 );
		};

		MultiInputIndexedDataStream< Data... > _multiStream;
		std::vector< _NextValue > _nextValues;
		bool _firstTime;

		void _init( const std::tuple< Data ... > &data );
	};
#include "DataStream.inl"
}

#endif // DATA_STREAM_INCLUDED
