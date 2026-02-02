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

#ifndef POINT_PARTITION_INCLUDED
#define POINT_PARTITION_INCLUDED

#include <string>
#include <sstream>
#include <filesystem>
#include "Streams.h"
#include "MyMiscellany.h"
#include "Ply.h"
#include "DataStream.h"
#include "PointExtent.h"

namespace PoissonRecon
{
	static const unsigned int BUFFER_IO = 1<<14;			// Buffer the points before reading/writing


	namespace PointPartition
	{
		template< typename Real , unsigned int Dim >
		struct PointSetInfo
		{
			std::string header;
			unsigned filesPerDir;
			XForm< Real , Dim+1 > modelToUnitCube;
			std::vector< size_t > pointsPerSlab;
			std::vector< PlyProperty > auxiliaryProperties;

			PointSetInfo( void );
			PointSetInfo( unsigned int slabs );
			PointSetInfo( BinaryStream &stream );

			void write( BinaryStream &stream ) const;
		};


		struct Partition
		{
			Partition( void );
			Partition( unsigned int dCount , const std::vector< size_t > &slabSizes );

#ifdef ADAPTIVE_PADDING
			void optimize( bool useMax );
			std::pair< unsigned int , unsigned int > range( unsigned int i ) const;
			size_t size( unsigned int i ) const;
			void printDistribution( void ) const;
			double l2Energy( void ) const;
			double maxEnergy( void ) const;
#else // !ADAPTIVE_PADDING
			void optimize( bool useMax , unsigned int padSize );
			std::pair< unsigned int , unsigned int > range( unsigned int i , unsigned int padSize ) const;
			size_t size( unsigned int i , unsigned int padSize ) const;
			void printDistribution( unsigned int padSize ) const;
			double l2Energy( unsigned int padSize ) const;
			double maxEnergy( unsigned int padSize ) const;
#endif // ADAPTIVE_PADDING
			size_t size( void ) const;
			unsigned int partitions( void ) const;
			unsigned int slabs( void ) const;

		protected:
			std::vector< unsigned int > _starts;
			std::vector< size_t  > _slabSizes;
		};

		long ReadPLYProperties( FILE *fp , std::vector< PlyProperty > &properties );
		long ReadPLYProperties( const char *fileName , std::vector< PlyProperty > &properties );
		long WritePLYProperties( FILE *fp , const std::vector< PlyProperty > &properties );
		long WritePLYProperties( const char *fileName , const std::vector< PlyProperty > &properties );

		template< typename InputFactory >
		struct BufferedBinaryInputDataStream : public InputDataStream< typename InputFactory::VertexType >
		{

			typedef typename InputFactory::VertexType Data;
			BufferedBinaryInputDataStream( const char *fileName , const InputFactory &factory , size_t bufferSize );
			~BufferedBinaryInputDataStream( void );
			void reset( void );
			bool read( Data &d );
		protected:
			size_t _bufferSize , _current , _inBuffer , _elementSize;
			Pointer( char ) _buffer;
			FILE *_fp;
			const InputFactory &_factory;
			long _inset;
		};

		template< typename OutputFactory >
		struct BufferedBinaryOutputDataStream : public OutputDataStream< typename OutputFactory::VertexType >
		{
			typedef typename OutputFactory::VertexType Data;
			BufferedBinaryOutputDataStream( const char *fileName , const OutputFactory &factory , size_t bufferSize );
			~BufferedBinaryOutputDataStream( void );
			void reset( void );
			size_t write( const Data &d );
			size_t size( void ) const;
		protected:
			size_t _bufferSize , _current , _elementSize;
			Pointer( char ) _buffer;
			FILE *_fp;
			const OutputFactory &_factory;
			long _inset;
		};

		void RemovePointSlabDirs( std::string dir );
		static void CreatePointSlabDirs( std::string dir , unsigned int count , unsigned int filesPerDir );

		std::string FileDir( std::string dir , std::string header );
		std::string FileDir( std::string dir , std::string header , unsigned int clientIndex );

		std::string FileName( std::string dir ,                                                 unsigned int slab , unsigned int slabs , unsigned int filesPerDir );
		std::string FileName( std::string dir , std::string header ,                            unsigned int slab , unsigned int slabs , unsigned int filesPerDir );
		std::string FileName( std::string dir , std::string header , unsigned int clientIndex , unsigned int slab , unsigned int slabs , unsigned int filesPerDir );

		std::string PointSetInfoName( std::string dir , std::string header );

#include "PointPartition.inl"
	}
}

#endif // POINT_PARTITION_INCLUDED