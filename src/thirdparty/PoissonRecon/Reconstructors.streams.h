/*
Copyright (c) 2022, Michael Kazhdan and Matthew Bolitho
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

#ifndef RECONSTRUCTORS_STREAMS_INCLUDED
#define RECONSTRUCTORS_STREAMS_INCLUDED

#include "FEMTree.h"
#include "MyExceptions.h"
#include "Array.h"

namespace PoissonRecon
{
	namespace Reconstructor
	{
		template< typename Real , unsigned int Dim > using Position = Point< Real , Dim >;
		template< typename Real , unsigned int Dim > using Normal   = Point< Real , Dim >;
		template< typename Real , unsigned int Dim > using Gradient = Point< Real , Dim >;
		template< typename Real > using Weight = Real;
		template< typename Real > using Value = Real;

		////////////////////////
		// Input stream types //
		////////////////////////

		template< typename Real , unsigned int Dim , typename ... Data > using InputSampleStream = InputDataStream< Point< Real , Dim > , Data ... >;
		template< typename Real , unsigned int Dim , typename ... Data > struct TransformedInputSampleStream;

		template< typename Real , unsigned int Dim , typename ... Data > using InputOrientedSampleStream = InputDataStream< Position< Real , Dim > , Normal< Real , Dim > , Data ... >;
		template< typename Real , unsigned int Dim , typename ... Data > struct TransformedInputOrientedSampleStream;

		template< typename Real , unsigned int Dim , typename ... Data > using InputValuedSampleStream = InputSampleStream< Real , Dim , Value< Real > , Data ... >;
		template< typename Real , unsigned int Dim , typename ... Data > using TransformedInputValuedSampleStream = TransformedInputSampleStream< Real , Dim , Value< Real > , Data ... >;


		/////////////////////////
		// Output stream types //
		/////////////////////////

		template< typename Real , unsigned int Dim , typename ... Data > using OutputLevelSetVertexStream = OutputDataStream< Position< Real , Dim > , Gradient< Real , Dim > , Weight< Real > , Data ... >;
		template< typename Real , unsigned int Dim , typename ... Data > struct TransformedOutputLevelSetVertexStream;

		//////////////////
		// Face streams //
		//////////////////
		template< unsigned int FaceDim , typename T=node_index_type > using Face = std::conditional_t< FaceDim==2 , std::vector< T > , std::conditional_t< FaceDim==1 , std::pair< T , T > , void * > >;
		template< unsigned int FaceDim > using  InputFaceStream =  InputDataStream< Face< FaceDim > >;
		template< unsigned int FaceDim > using OutputFaceStream = OutputDataStream< Face< FaceDim > >;

		//////////////////////////////////////////////
		// Information about the output vertex type //
		//////////////////////////////////////////////
		template< typename Real , unsigned int Dim , bool HasGradients , bool HasDensity , typename ... AxuDataFactories > struct OutputVertexInfo;

		//////////////////////////////////////////
		// Transformed Input Sample Data Stream //
		//////////////////////////////////////////
		template< typename Real , unsigned int Dim , typename ... Data >
		struct TransformedInputSampleStream : public InputSampleStream< Real , Dim , Data ... >
		{
			// A constructor initialized with the transformation to be applied to the samples, and a sample stream
			TransformedInputSampleStream( XForm< Real , Dim+1 > xForm , InputSampleStream< Real , Dim , Data ... > &stream ) : _stream(stream) , _xForm(xForm) {}

			// Functionality to reset the stream to the start
			void reset( void ){ _stream.reset(); }

			// Functionality to extract the next position/normal pair.
			// The method returns true if there was another point in the stream to read, and false otherwise
			bool read( Position< Real , Dim > &p , Data& ... d )
			{
				if( !_stream.read( p , d... ) ) return false;
				p = _xForm * p;
				return true;
			}
			bool read( unsigned int thread , Position< Real , Dim > &p , Data& ... d )
			{
				if( !_stream.read( thread , p , d... ) ) return false;
				p = _xForm * p;
				return true;
			}

		protected:
			// A reference to the underlying stream
			InputSampleStream< Real , Dim , Data ... > &_stream;

			// The affine transformation to be applied to the positions
			XForm< Real , Dim+1 > _xForm;
		};

		//////////////////////////////////////////////////
		// Transformed Input Oriente Sample Data Stream //
		//////////////////////////////////////////////////
		template< typename Real , unsigned int Dim , typename ... Data >
		struct TransformedInputOrientedSampleStream : public InputOrientedSampleStream< Real , Dim , Data ... >
		{
			// A constructor initialized with the transformation to be applied to the samples, and a sample stream
			TransformedInputOrientedSampleStream( XForm< Real , Dim+1 > xForm , InputOrientedSampleStream< Real , Dim , Data ... > &stream ) : _stream(stream) , _positionXForm(xForm)
			{
				_normalXForm = XForm< Real , Dim > ( xForm ).inverse().transpose() * (Real)pow( fabs( xForm.determinant() ) , 1./Dim );
			}

			// Functionality to reset the stream to the start
			void reset( void ){ _stream.reset(); }

			// Functionality to extract the next position/normal pair.
			// The method returns true if there was another point in the stream to read, and false otherwise
			bool read( Position< Real , Dim > &p , Normal< Real , Dim > &n , Data& ... d )
			{
				if( !_stream.read( p , n , d... ) ) return false;
				p = _positionXForm * p , n = _normalXForm * n;
				return true;
			}
			bool read( unsigned int thread , Position< Real , Dim > &p , Normal< Real , Dim > &n , Data& ... d )
			{
				if( !_stream.read( thread , p , n , d... ) ) return false;
				p = _positionXForm * p , n = _normalXForm * n;
				return true;
			}

		protected:
			// A reference to the underlying stream
			InputOrientedSampleStream< Real , Dim , Data ... > &_stream;

			// The affine transformation to be applied to the positions
			XForm< Real , Dim+1 > _positionXForm;

			// The linear transformation to be applied to the normals
			XForm< Real , Dim > _normalXForm;
		};

		//////////////////////////////////////
		// Transformed Output Vertex Stream //
		//////////////////////////////////////
		template< typename Real , unsigned int Dim , typename ... Data >
		struct TransformedOutputLevelSetVertexStream : public OutputLevelSetVertexStream< Real , Dim , Data ... >
		{
			// A constructor initialized with the transformation to be applied to the samples, and a sample stream
			TransformedOutputLevelSetVertexStream( XForm< Real , Dim+1 > xForm , OutputLevelSetVertexStream< Real , Dim , Data ... > &stream ) : _stream(stream) , _positionXForm(xForm)
			{
				_gradientXForm = XForm< Real , Dim > ( xForm ).inverse().transpose() * (Real)pow( xForm.determinant() , 1./Dim );
			}

			// Need to write the union to ensure that the counter gets set
			size_t write(                       const Position< Real , Dim > &p , const Gradient< Real , Dim > &g , const Weight< Real > &w , const Data& ... d ){ return _stream.write(          _positionXForm * p , _gradientXForm * g , w , d... ); }
			size_t write( unsigned int thread , const Position< Real , Dim > &p , const Gradient< Real , Dim > &g , const Weight< Real > &w , const Data& ... d ){ return _stream.write( thread , _positionXForm * p , _gradientXForm * g , w , d... ); }
			size_t size( void ) const { return _stream.size(); }

		protected:
			// A reference to the underlying stream
			OutputLevelSetVertexStream< Real , Dim , Data ... > &_stream;

			// The affine transformation to be applied to the positions
			XForm< Real , Dim+1 > _positionXForm;

			// The linear transformation to be applied to the normals
			XForm< Real , Dim > _gradientXForm;
		};


		//////////////////////////////////
		// File-backed streaming memory //
		//////////////////////////////////
		class FileBackedReadWriteStream
		{
		public:
			struct FileDescription
			{
				FILE *fp;

				FileDescription( FILE *fp ) : fp(fp) , _closeFile(false)
				{
					if( !this->fp )
					{
						this->fp = std::tmpfile();
						_closeFile = true;
						if( !this->fp ) MK_THROW( "Failed to open temporary file" );
					}
				}
				~FileDescription( void ){ if( _closeFile ) fclose(fp); }
			protected:
#ifdef SHOW_WARNINGS
#pragma message( "[WARNING] Probably can let the system handle closing the file" )
#endif // SHOW_WARNINGS
				bool _closeFile;
			};

			FileBackedReadWriteStream( FILE *fp ) : _fd(fp) {}
			bool write( ConstPointer(char) data , size_t size ){ return fwrite( data , sizeof(char) , size , _fd.fp )==size; }
			bool read( Pointer(char) data , size_t size ){ return fread( data , sizeof(char) , size , _fd.fp )==size; }
			void reset( void ){ fseek( _fd.fp , 0 , SEEK_SET ); }
		protected:
			FileDescription _fd;
		};


		//////////////////////////////////////////////////////////////////////////////
		// Output and the input face stream, backed either by a file or by a vector //
		//////////////////////////////////////////////////////////////////////////////
		// [WARNING] These assume that the stream starts as write-only and after the reset method is invoked, the stream becomes read-only.

		template< unsigned int FaceDim , bool InCore , bool Parallel >
		struct OutputInputFaceStream
			: public OutputFaceStream< FaceDim >
			, public  InputFaceStream< FaceDim >
		{
			// The streams for communicating the information
			InputFaceStream < FaceDim > * inStream;
			OutputFaceStream< FaceDim > *outStream;

			void reset( void ){ inStream->reset(); }
			bool read(                  Face< FaceDim > &f ){ return inStream->read(  f); }
			bool read( unsigned int t , Face< FaceDim > &f ){ return inStream->read(t,f); }
			size_t write(                  const Face< FaceDim > &f ){ return outStream->write(  f); }
			size_t write( unsigned int t , const Face< FaceDim > &f ){ return outStream->write(t,f); }
			size_t size( void ) const { return outStream->size(); }

			OutputInputFaceStream( void )
			{
				size_t sz = ThreadPool::NumThreads();

				_backingVectors.resize( sz , nullptr );

				_backingFiles.resize( sz , nullptr );

				_inStreams.resize( sz , nullptr );
				_outStreams.resize( sz , nullptr );

				if constexpr( InCore )
				{
					if constexpr( Parallel )
					{
						for( unsigned int i=0 ; i<sz ; i++ )
						{
							_backingVectors[i] = new std::vector< Face< FaceDim > >();
							_inStreams[i] = new VectorBackedInputDataStream< Face< FaceDim > >( *_backingVectors[i] );
							_outStreams[i] = new VectorBackedOutputDataStream< Face< FaceDim > >( *_backingVectors[i] );
						}
						inStream = new MultiInputDataStream< Face< FaceDim > >( _inStreams );
						outStream = new MultiOutputDataStream< Face< FaceDim > >( _outStreams );
					}
					else
					{
						_backingVector = new std::vector< Face< FaceDim > >();
						inStream = new VectorBackedInputDataStream< Face< FaceDim > >( *_backingVector );
						outStream = new VectorBackedOutputDataStream< Face< FaceDim > >( *_backingVector );
					}
				}
				else
				{
					if constexpr( Parallel )
					{
						for( unsigned int i=0 ; i<sz ; i++ )
						{
							_backingFiles[i] = new FileBackedReadWriteStream::FileDescription( NULL );
							_inStreams[i] = new FileBackedInputDataStream< Face< FaceDim > >( _backingFiles[i]->fp );
							_outStreams[i] = new FileBackedOutputDataStream< Face< FaceDim > >( _backingFiles[i]->fp );
						}
						inStream = new MultiInputDataStream< Face< FaceDim > >( _inStreams );
						outStream = new MultiOutputDataStream< Face< FaceDim > >( _outStreams );
					}
					else
					{
						_backingFile = new FileBackedReadWriteStream::FileDescription( NULL );
						inStream = new FileBackedInputDataStream< Face< FaceDim > >( _backingFile->fp );
						outStream = new FileBackedOutputDataStream< Face< FaceDim > >( _backingFile->fp );
					}
				}
			}

			~OutputInputFaceStream( void )
			{
				size_t sz = ThreadPool::NumThreads();

				delete _backingVector;
				delete _backingFile;

				for( unsigned int i=0 ; i<sz ; i++ )
				{
					delete _backingVectors[i];
					delete _backingFiles[i];
					delete  _inStreams[i];
					delete _outStreams[i];
				}

				delete  inStream;
				delete outStream;
			}
		protected:
			std::vector< Face< FaceDim > > *_backingVector = nullptr;
			FileBackedReadWriteStream::FileDescription *_backingFile = nullptr;
			std::vector< std::vector< Face< FaceDim > > * > _backingVectors;
			std::vector< FileBackedReadWriteStream::FileDescription * > _backingFiles;
			std::vector<  InputDataStream< Face< FaceDim > > * >  _inStreams;
			std::vector< OutputDataStream< Face< FaceDim > > * > _outStreams;
		};


		template< typename Real , unsigned int Dim , typename Factory , bool InCore , bool Parallel , typename ... Data >
		struct OutputInputFactoryTypeStream
			: public OutputDataStream< Position< Real , Dim > , Gradient< Real , Dim > , Weight< Real > , Data ... >
			, public InputDataStream< typename Factory::VertexType >
		{
			using Vertex = typename Factory::VertexType;

			// The streams for communicating the information
			using OutputStreamType = OutputDataStream< Vertex >;
			using  InputStreamType =  InputDataStream< Vertex >;
			OutputStreamType *outStream;
			InputStreamType  * inStream;

			void reset( void ){ inStream->reset(); }
			size_t write( const Position< Real , Dim > &p , const Gradient< Real , Dim > &g , const Weight< Real > &w , const Data& ... d )
			{
				Vertex v = _converter( p , g , w , d... );
				return outStream->write( v );
			}
			size_t write( unsigned int thread , const Position< Real , Dim > &p , const Gradient< Real , Dim > &g , const Weight< Real > &w , const Data& ... d )
			{
				Vertex v = _converter( p , g , w , d... );
				return outStream->write( thread , v );
			}
			bool read(                       Vertex &v ){ return inStream->read(          v ); }
			bool read( unsigned int thread , Vertex &v ){ return inStream->read( thread , v ); }

			size_t size( void ) const { return outStream->size(); }

			OutputInputFactoryTypeStream( Factory &factory , std::function< Vertex ( const Position< Real , Dim > & , const Gradient< Real , Dim > & , const Weight< Real > & , const Data & ...  ) > converter )
				: _converter( converter )
			{
				size_t sz = ThreadPool::NumThreads();

				_backingVectors.resize( sz , nullptr );

				_backingFiles.resize( sz , nullptr );

				_inStreams.resize( sz , nullptr );
				_outStreams.resize( sz , nullptr );

				if constexpr( Parallel )
				{
					for( unsigned int i=0 ; i<sz ; i++ )
					{
						if constexpr( InCore )
						{
							_backingVectors[i] = new std::vector< std::pair< size_t , Vertex > >();
							_outStreams[i] = new VectorBackedOutputIndexedDataStream< Vertex >( *_backingVectors[i] );
							_inStreams[i] = new VectorBackedInputIndexedDataStream< Vertex >( *_backingVectors[i] );
						}
						else
						{
							_backingFiles[i] = new FileBackedReadWriteStream::FileDescription( NULL );
							_outStreams[i] = new FileBackedOutputIndexedFactoryTypeStream< Factory >( _backingFiles[i]->fp , factory  );
							_inStreams[i] = new FileBackedInputIndexedFactoryTypeStream< Factory >( _backingFiles[i]->fp , factory );
						}
					}
					outStream = new DeInterleavedMultiOutputIndexedDataStream< Vertex >( _outStreams );
					inStream = new InterleavedMultiInputIndexedDataStream< Vertex >( _inStreams );
				}
				else
				{
					if constexpr( InCore )
					{
						_backingVector = new std::vector< Vertex >();
						outStream = new VectorBackedOutputDataStream< Vertex >( *_backingVector );
						inStream = new VectorBackedInputDataStream< Vertex >( *_backingVector );
					}
					else
					{
						_backingFile = new FileBackedReadWriteStream::FileDescription( NULL );
						outStream = new FileBackedOutputFactoryTypeStream< Factory >( _backingFile->fp , factory );
						inStream = new FileBackedInputFactoryTypeStream< Factory >( _backingFile->fp , factory );
					}
				}
			}

			~OutputInputFactoryTypeStream( void )
			{
				size_t sz = ThreadPool::NumThreads();

				delete _backingVector;
				delete _backingFile;

				for( unsigned int i=0 ; i<sz ; i++ )
				{
					delete _backingVectors[i];
					delete _backingFiles[i];
					delete  _inStreams[i];
					delete _outStreams[i];
				}

				delete  inStream;
				delete outStream;
				delete _inMultiStream;
			}
		protected:
			std::function< Vertex ( const Position< Real , Dim > & , const Gradient< Real , Dim > & , const Weight< Real > & , const Data & ...  ) > _converter;
			std::vector< Vertex > *_backingVector = nullptr;
			FileBackedReadWriteStream::FileDescription *_backingFile = nullptr;
			std::vector< std::vector< std::conditional_t< Parallel , std::pair< size_t , Vertex > , Vertex > > * >_backingVectors;
			std::vector< FileBackedReadWriteStream::FileDescription * > _backingFiles;
			std::vector< OutputIndexedDataStream< Vertex > * > _outStreams;
			std::vector<  InputIndexedDataStream< Vertex > * >  _inStreams;
			MultiInputIndexedDataStream< Vertex > *_inMultiStream = nullptr;
		};

		template< typename Real , unsigned int Dim , bool HasGradients , bool HasDensity >
		struct OutputVertexInfo< Real , Dim , HasGradients , HasDensity >
		{
			using Factory =
				typename std::conditional_t
				<
				HasGradients ,
				typename std::conditional_t
				<
				HasDensity ,
				VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > , VertexFactory::NormalFactory< Real , Dim > , VertexFactory::ValueFactory< Real > > ,
				VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > , VertexFactory::NormalFactory< Real , Dim > >
				> ,
				typename std::conditional_t
				<
				HasDensity ,
				VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > , VertexFactory::ValueFactory< Real > > ,
				VertexFactory::PositionFactory< Real , Dim >
				>
				>;
			using Vertex = typename Factory::VertexType;

			static Factory GetFactory( void ){ return Factory(); }

			static Vertex Convert( const Position< Real , Dim > &p , const Gradient< Real , Dim > &g , const Weight< Real > & w )
			{
				Vertex v;
				if constexpr( !HasGradients && !HasDensity ) v = p;
				else
				{
					v.template get<0>() = p;
					if constexpr( HasGradients )
					{
						if constexpr( HasDensity ) v.template get<1>() = g , v.template get<2>() = w;
						else                       v.template get<1>() = g;
					}
					else
					{
						if constexpr( HasDensity ) v.template get<1>() = w;
					}
				}
				return v;
			}
		};

		template< typename Real , unsigned int Dim , bool HasGradients , bool HasDensity , typename AuxDataFactory >
		struct OutputVertexInfo< Real , Dim , HasGradients , HasDensity , AuxDataFactory >
		{
			using Factory =
				typename std::conditional
				<
				HasGradients ,
				typename std::conditional
				<
				HasDensity ,
				VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > , VertexFactory::NormalFactory< Real , Dim > , VertexFactory::ValueFactory< Real > , AuxDataFactory > ,
				VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > , VertexFactory::NormalFactory< Real , Dim > , AuxDataFactory >
				>::type ,
				typename std::conditional
				<
				HasDensity ,
				VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > , VertexFactory::ValueFactory< Real > , AuxDataFactory > ,
				VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > , AuxDataFactory >
				>::type
				>::type;
			using AuxData = typename AuxDataFactory::VertexType;

			using _Vertex = DirectSum< Real , Point< Real , Dim > , Point< Real , Dim > , Real , typename AuxDataFactory::VertexType >;
			using Vertex = typename Factory::VertexType;

			static Factory GetFactory( AuxDataFactory auxDataFactory )
			{
				if constexpr( HasGradients )
				{
					if constexpr( HasDensity ) return Factory( VertexFactory::PositionFactory< Real , Dim >() , VertexFactory::NormalFactory< Real , Dim >() , VertexFactory::ValueFactory< Real >() , auxDataFactory );
					else                       return Factory( VertexFactory::PositionFactory< Real , Dim >() , VertexFactory::NormalFactory< Real , Dim >() ,                                         auxDataFactory );
				}
				else
				{
					if constexpr( HasDensity ) return Factory( VertexFactory::PositionFactory< Real , Dim >() ,                                                VertexFactory::ValueFactory< Real >() , auxDataFactory );
					else                       return Factory( VertexFactory::PositionFactory< Real , Dim >() ,                                                                                        auxDataFactory );
				}
			}

			static Vertex Convert( const Position< Real , Dim > &p , const Gradient< Real , Dim > &g , const Weight< Real > & w , const AuxData &data )
			{
				Vertex v;
				v.template get<0>() = p;
				if constexpr( HasGradients )
				{
					if constexpr( HasDensity ) v.template get<1>() = g , v.template get<2>() = w , v.template get<3>() = data;
					else                       v.template get<1>() = g , v.template get<2>() = data;
				}
				else
				{
					if constexpr( HasDensity ) v.template get<1>() = w , v.template get<2>() = data;
					else                       v.template get<1>() = data;
				}
				return v;
			}
		};
	}
}

#endif // RECONSTRUCTORS_STREAMS_INCLUDED