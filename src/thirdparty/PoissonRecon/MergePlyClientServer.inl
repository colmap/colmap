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

enum _IndexType{ INT , U_INT , LONG_LONG };

inline std::string _FullFileName( std::string dir , std::string fileName )
{
	if( dir.back()!=FileSeparator ) dir += std::string(1,FileSeparator);
	return dir + fileName;
}

inline void _Copy( FILE *target , FILE *source , size_t sz , size_t bufferSize=1<<20 )
{
	Pointer( unsigned char ) buffer = NewPointer< unsigned char >( bufferSize );
	if( sz==-1 )
	{
		size_t ioBytes;
		while( ( ioBytes=fread( buffer , sizeof(unsigned char) , bufferSize , source ) ) ) fwrite( buffer , sizeof(unsigned char) , ioBytes , target );
	}
	else
	{
		while( sz )
		{
			size_t ioBytes = std::min< size_t >( bufferSize , sz );
			if( ioBytes!=fread( buffer , sizeof(unsigned char) , ioBytes , source ) ) MK_THROW( "Failed to read from source: " , ioBytes );
			if( ioBytes!=fwrite( buffer , sizeof(unsigned char) , ioBytes , target ) ) MK_THROW( "Failed to write to target: " , ioBytes );
			sz -= ioBytes;
		}
	}
	DeletePointer( buffer );
}

template< typename Factory >
size_t SizeOnDisk( const Factory &factory )
{
	size_t sizeOnDisk = 0;
	if constexpr( Factory::IsStaticallyAllocated() ) for( unsigned int i=0 ; i<factory.plyWriteNum() ; i++ ) sizeOnDisk += ply_type_size[ factory.plyStaticWriteProperty(i).external_type ];
	else                                             for( unsigned int i=0 ; i<factory.plyWriteNum() ; i++ ) sizeOnDisk += ply_type_size[ factory.plyWriteProperty(i).external_type ];
	return sizeOnDisk;
}

template< typename Index , typename Factory >
void _OffsetPolygons( const Factory &factory , std::string in , std::string out , size_t offset , Profiler &profiler )
{

	int ft;
	std::vector< std::string > comments;

	std::vector< std::tuple< std::string , size_t , std::vector< PlyProperty > > > _elems;
	PlyFile *inPly = PLY::ReadHeader( in , ft , _elems );

	size_t sizeOnDisk = SizeOnDisk( factory );

	// Skip the vertices
	{
#if defined( _WIN32 ) || defined( _WIN64 )
		_fseeki64( inPly->fp , sizeOnDisk * std::get<1>( _elems[0] ) , SEEK_CUR );
#else // !_WIN32 && !_WIN64
		fseek( inPly->fp , sizeOnDisk * std::get<1>( _elems[0] ) , SEEK_CUR );
#endif // _WIN32 || _WIN64
	}

	std::vector< std::tuple< std::string , size_t , std::vector< PlyProperty > > > elems(1);
	std::get<0>( elems[0] ) = std::string( "face" );
	std::get<1>( elems[0] ) = std::get<1>( _elems[1] );
	std::get<2>( elems[0] ).resize(1);
	std::get<2>( elems[0] )[0] = PLY::Face< Index >::Properties[0];

	PlyFile *outPly = PLY::WriteHeader( out , ft , elems , comments );

	// Read and offset the polygons
	{
		int maxIndices = 64;
		Index *faceIndices = (Index*)malloc( sizeof(Index) * maxIndices );

		auto ReadPolygon = [&]( FILE *fp )
		{
			int n;
			if( fread( &n , sizeof(int) , 1 , fp )!=1 ) MK_THROW( "Failed to read polygon size" );
			if( n>maxIndices )
			{
				maxIndices = n;
				faceIndices = (Index*)realloc( faceIndices , sizeof(Index) * maxIndices );
			}
			if( fread( faceIndices , sizeof(Index) , n , fp )!=n ) MK_THROW( "Failed to read polygon indices" );
			return n;
		};

		auto WritePolygon = [&]( FILE *fp , int n )
		{
			fwrite( &n , sizeof(int) , 1 , fp );
			fwrite( faceIndices , sizeof(Index) , n , fp );
		};

		for( unsigned int j=0 ; j<std::get<2>( _elems[1] ).size() ; j++ ) inPly->get_property( std::get<0>( _elems[1] ) , &std::get<2>( _elems[1] )[j] );
		outPly->put_element_setup( std::string( "face" ) );

		for( size_t i=0 ; i<std::get<1>( _elems[1] ) ; i++ )
		{
			int n = ReadPolygon( inPly->fp );
			for( int j=0 ; j<n ; j++ ) faceIndices[j] += (Index)offset;
			WritePolygon( outPly->fp , n );
		}

		profiler.update();

		delete[] faceIndices;
	}
	delete inPly;
	delete outPly;
}


////////////
// Server //
////////////

template< typename Real , unsigned int Dim , typename Factory >
void _RunServer
(
	std::string inDir , 
	std::string tempDir ,
	std::string header ,
	std::string out ,
	std::vector< Socket > &clientSockets ,
	const std::vector< unsigned int > &sharedVertexCounts ,
	ClientMergePlyInfo clientMergePlyInfo ,
	const Factory &factory ,
	unsigned int sampleMS ,
	std::function< std::vector< std::string > (unsigned int) > commentFunctor
)
{
	Profiler profiler(sampleMS);
	typedef typename Factory::VertexType Vertex;

	auto ClientFile = []( std::string dir , std::string header , unsigned int idx )
	{
		std::stringstream ss;
		ss << _FullFileName( dir , header ) << "." << idx << ".ply";
		return ss.str();
	};

	auto InFile = [&]( unsigned int idx ){ return ClientFile( inDir , header , idx ); };
	auto TempPolygonFile = [&]( unsigned int idx ){ return ClientFile( tempDir , header + std::string( ".polygons" ) , idx ); };

	profiler.reset();

	// Get the number of vertices and polygons per file
	std::vector< size_t > vNum( sharedVertexCounts.size()+1 ) , fNum( sharedVertexCounts.size()+1 ) , offsets( sharedVertexCounts.size()+1 );
	{
		for( unsigned int i=0 ; i<=sharedVertexCounts.size() ; i++ )
		{
			std::string inFile = InFile(i);
			int ft;
			std::vector< std::tuple< std::string , size_t , std::vector< PlyProperty > > > elems;
			PlyFile *ply = PLY::ReadHeader( inFile , ft , elems );
			delete ply;
			bool foundVertices = false , foundFaces = false;
			for( unsigned int j=0 ; j<elems.size() ; j++ )
				if     ( std::get<0>( elems[j] )==std::string( "vertex" ) ) foundVertices = true , vNum[i] = std::get<1>( elems[j] );
				else if( std::get<0>( elems[j] )==std::string( "face"   ) ) foundFaces    = true , fNum[i] = std::get<1>( elems[j] );
			if( !foundVertices ) MK_THROW( "Could not find vertices" );
			if( !foundFaces ) MK_THROW( "Could not find faces" );
			profiler.update();
		}
		offsets[0] = 0;
		for( unsigned int i=1 ; i<=sharedVertexCounts.size() ; i++ ) offsets[i] = clientMergePlyInfo.keepSeparate ? 0 : offsets[i-1] + vNum[i-1] - sharedVertexCounts[i-1];
	}

	// Strip out the polygons (though perhaps we should just have the reconstruction code do this).
#ifdef SHOW_WARNINGS
	MK_WARN( "Should split the mesh during reconstruction" );
#endif // SHOW_WARNINGS

	_IndexType idxType;
	{
		size_t maxV;
		if( clientMergePlyInfo.keepSeparate )
		{
			maxV = 0;
			for( unsigned int i=0 ; i<vNum.size() ; i++ ) maxV = std::max< size_t >( maxV , vNum[i] );
		}
		else maxV = offsets.back()+vNum.back();

		if( maxV>std::numeric_limits< int >::max() )
		{
			if( maxV>std::numeric_limits< unsigned int >::max () ) idxType = LONG_LONG;
			else idxType = U_INT;
		}
		else idxType = INT;
	}

	if( clientMergePlyInfo.verbose ) std::cout << "Got mesh info: " << profiler(true) << std::endl;


	for( unsigned int i=0 ; i<clientSockets.size() ; i++ )
	{
		SocketStream clientStream( clientSockets[i] );
		clientStream.write( InFile(i) );
		clientStream.write( TempPolygonFile(i) );
		clientStream.write( idxType );
		clientStream.write( offsets[i] );

	}
	for( unsigned int i=0 ; i<clientSockets.size() ; i++ )
	{
		unsigned char done;
		SocketStream clientStream( clientSockets[i] );
		clientStream.read( done );
	}

	profiler.reset();
	std::vector< std::tuple< std::string , size_t , std::vector< PlyProperty > > > elems(2);
	{
		std::get<0>( elems[0] ) = std::string( "vertex" );
		std::get<1>( elems[0] ) = 0;
		for( unsigned int i=0 ; i<vNum.size() ; i++ ) std::get<1>( elems[0] ) += vNum[i];
		for( unsigned int i=0 ; i<sharedVertexCounts.size() ; i++ ) std::get<1>( elems[0] ) -= sharedVertexCounts[i];
		std::get<2>( elems[0] ).resize( factory.plyWriteNum() );
		for( unsigned int i=0 ; i<factory.plyWriteNum() ; i++ )
			if constexpr( Factory::IsStaticallyAllocated() ) std::get<2>( elems[0] )[i] = factory.plyStaticWriteProperty(i);
			else                                             std::get<2>( elems[0] )[i] = factory.plyWriteProperty(i);

		std::get<0>( elems[1] ) = std::string( "face" );
		std::get<1>( elems[1] ) = 0;
		for( unsigned int i=0 ; i<fNum.size() ; i++ ) std::get<1>( elems[1] ) += fNum[i];
		std::get<2>( elems[1] ).resize(1);
		switch( idxType )
		{
			case INT:       std::get<2>( elems[1] )[0] = PLY::Face<          int >::Properties[0] ; break;
			case U_INT:     std::get<2>( elems[1] )[0] = PLY::Face< unsigned int >::Properties[0] ; break;
			case LONG_LONG: std::get<2>( elems[1] )[0] = PLY::Face<    long long >::Properties[0] ; break;
			default: MK_THROW( "Unrecognized output type" );
		}
	}

	std::vector< PlyFile * > outPly( clientMergePlyInfo.keepSeparate ? sharedVertexCounts.size()+1 : 1 , NULL );
	if( clientMergePlyInfo.keepSeparate )
		for( unsigned int i=0 ; i<=sharedVertexCounts.size() ; i++ )
		{
			std::string fileName = out + std::string( "." ) + std::to_string(i) + std::string( ".ply" );
			std::get<1>( elems[0] ) = vNum[i];
			std::get<1>( elems[1] ) = fNum[i];
			std::pair< unsigned , unsigned int > shared;
			shared.first  = i>0                         ? sharedVertexCounts[i-1] : 0;
			shared.second = i<sharedVertexCounts.size() ? sharedVertexCounts[i  ] : 0;
			std::vector< std::string > _comments = commentFunctor( i );
			std::vector< std::string > comments( 1+_comments.size() );
			comments[0] = std::string( "Shared: " ) + std::to_string( shared.first ) + std::string( "|" ) + std::to_string( shared.second );
			for( unsigned int j=0 ; j<_comments.size() ; j++ ) comments[j+1] = _comments[j];
			outPly[i] = PLY::WriteHeader( fileName , PLY_BINARY_NATIVE , elems , comments );
		}
	else
	{
		std::vector< std::string > comments = commentFunctor(-1);
		outPly[0] = PLY::WriteHeader( out , PLY_BINARY_NATIVE , elems , comments );
	}

	// Write out the (merged) vertices
	{
		Pointer( char ) vBuffer = NewPointer< char >( factory.bufferSize() );

		for( unsigned int i=0 ; i<outPly.size() ; i++ ) outPly[i]->put_element_setup( std::string( "vertex" ) );

		size_t sizeOnDisk = SizeOnDisk( factory );

		std::vector< Vertex > sharedVertices;
		for( unsigned int i=0 ; i<=sharedVertexCounts.size() ; i++ )
		{
			std::vector< std::tuple< std::string , size_t , std::vector< PlyProperty > > > _elems;
			int ft;
			PlyFile *inPly = PLY::ReadHeader( InFile(i) , ft , _elems );

			for( unsigned int j=0 ; j<std::get<2>( elems[0] ).size() ; j++ ) inPly->get_property( std::get<0>( elems[0] ) , &std::get<2>( elems[0] )[j] );

			// Merge the start vertices with the end vertices from the previous client
			Vertex v = factory();
			for( unsigned int j=0 ; j<sharedVertices.size() ; j++ )
			{
				if constexpr( Factory::IsStaticallyAllocated() )
				{
					inPly->get_element( &v );
					sharedVertices[j] = ( sharedVertices[j] + v ) / (Real)2.;
					if( clientMergePlyInfo.keepSeparate )
					{
						// Write the merged vertices into the end of the previous
						if( i ) outPly[i-1]->put_element( (void*)&sharedVertices[j] );
						// Write the merged vertices into the beginning of the current
						outPly[i]->put_element( (void*)&sharedVertices[j] );
					}
					else outPly[0]->put_element( (void*)&sharedVertices[j] );
				}
				else
				{
					inPly->get_element( PointerAddress( vBuffer ) );
					factory.fromBuffer( vBuffer , v );
					sharedVertices[j] = ( sharedVertices[j] + v ) / (Real)2.;
					factory.toBuffer( sharedVertices[j] , vBuffer );
					if( clientMergePlyInfo.keepSeparate )
					{
						// Write the merged vertices into the end of the previous
						if( i ) outPly[i-1]->put_element( PointerAddress( vBuffer ) );
						// Write the merged vertices into the beginning of the current
						outPly[i]->put_element( PointerAddress( vBuffer ) );
					}
					else outPly[0]->put_element( PointerAddress( vBuffer ) );
				}
			}

			// Copy the interior vertices
			{
				size_t vNum = std::get<1>( _elems[0] ) - sharedVertices.size();
				if( i<sharedVertexCounts.size() ) vNum -= sharedVertexCounts[i];
				_Copy( outPly[ clientMergePlyInfo.keepSeparate ? i : 0 ]->fp , inPly->fp , vNum*sizeOnDisk , clientMergePlyInfo.bufferSize );
			}

			// Buffer the end vertices
			if( i<sharedVertexCounts.size() )
			{
				sharedVertices.resize( sharedVertexCounts[i] , factory() );

				for( unsigned int j=0 ; j<sharedVertices.size() ; j++ )
				{
					if constexpr( Factory::IsStaticallyAllocated() ) inPly->get_element( (void*)&sharedVertices[j] );
					else
					{
						inPly->get_element( PointerAddress( vBuffer ) );
						factory.fromBuffer( vBuffer , sharedVertices[j] );
					}
				}
			}
			else sharedVertices.resize(0);
			profiler.update();

			delete inPly;
		}
		DeletePointer( vBuffer );
		if( clientMergePlyInfo.verbose ) std::cout << "Merged vertices: " << profiler(true) << std::endl;
	}

	profiler.reset();
	// Write out the polygons
	{
		for( unsigned int i=0 ; i<outPly.size() ; i++ )	outPly[i]->put_element_setup( std::string( "face" ) );

		for( unsigned int i=0 ; i<=sharedVertexCounts.size() ; i++ )
		{
			std::vector< std::tuple< std::string , size_t , std::vector< PlyProperty > > > _elems;
			int ft;
			PlyFile *inPly = PLY::ReadHeader( TempPolygonFile(i) , ft , _elems );

			_Copy( outPly[ clientMergePlyInfo.keepSeparate ? i : 0 ]->fp , inPly->fp , -1 , clientMergePlyInfo.bufferSize );
			profiler.update();
			delete inPly;
		}
		if( clientMergePlyInfo.verbose ) std::cout << "Merged polygons: " << profiler(true) << std::endl;
	}
	for( unsigned int i=0 ; i<outPly.size() ; i++ ) delete outPly[i];


	for( unsigned int i=0 ; i<=sharedVertexCounts.size() ; i++ )
	{
		std::string fileName = TempPolygonFile(i);
		std::remove( fileName.c_str() );
	}
}

template< typename Real , unsigned int Dim >
void RunServer
(
	std::string inDir , 
	std::string tempDir ,
	std::string header ,
	std::string out ,
	std::vector< Socket > &clientSockets ,
	const std::vector< unsigned int > &sharedVertexCounts ,
	ClientMergePlyInfo clientMergePlyInfo ,
	unsigned int sampleMS ,
	std::function< std::vector< std::string > (unsigned int) > commentFunctor
)
{
	if( clientSockets.size()!=sharedVertexCounts.size()+1 ) MK_THROW( "Socket num and shared vertex count don't match: " , clientSockets.size() , " / " , sharedVertexCounts.size() );

	for( unsigned int i=0 ; i<clientSockets.size() ; i++ )
	{
		SocketStream clientStream( clientSockets[i] );
		clientMergePlyInfo.write( clientStream );
	}

	if( clientMergePlyInfo.auxProperties.size() )
	{
		typedef VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > ,  VertexFactory::DynamicFactory< Real > > Factory;
		VertexFactory::PositionFactory< Real , Dim > vFactory;
		VertexFactory::DynamicFactory< Real > dFactory( clientMergePlyInfo.auxProperties );
		Factory factory( vFactory , dFactory );
		_RunServer< Real , Dim >( inDir , tempDir , header , out , clientSockets , sharedVertexCounts , clientMergePlyInfo , factory , sampleMS , commentFunctor );
	}
	else
	{
		typedef VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > > Factory;
		Factory factory;
		_RunServer< Real , Dim >( inDir , tempDir , header , out , clientSockets , sharedVertexCounts , clientMergePlyInfo , factory , sampleMS , commentFunctor );
	}

}

////////////
// Client //
////////////
template< typename Real , unsigned int Dim , typename Factory >
void _RunClients
(
	ClientMergePlyInfo clientMergePlyInfo ,
	const Factory &factory ,
	std::vector< Socket > &serverSockets ,
	unsigned int sampleMS
)
{
	Profiler profiler( sampleMS );

	for( unsigned int c=0 ; c<serverSockets.size() ; c++ )
	{
		SocketStream socketStream( serverSockets[c] );
		std::string in , out;
		_IndexType idxType;
		size_t offset;
		socketStream.read( in );
		socketStream.read( out );
		socketStream.read( idxType );
		socketStream.read( offset );

		switch( idxType )
		{
			case INT:       _OffsetPolygons<          int >( factory , in , out , offset , profiler ) ; break;
			case U_INT:     _OffsetPolygons< unsigned int >( factory , in , out , offset , profiler ) ; break;
			case LONG_LONG: _OffsetPolygons<    long long >( factory , in , out , offset , profiler ) ; break;
			default: MK_THROW( "Unrecognized output index type" );
		}
		char done = 1;
		socketStream.write( done );
	}
	if( clientMergePlyInfo.verbose ) std::cout << "Offset polygons: " << profiler(true) << std::endl;
}

template< typename Real , unsigned int Dim >
void RunClients( std::vector< Socket > &serverSockets , unsigned int sampleMS )
{

	ClientMergePlyInfo clientMergePlyInfo;

	for( unsigned int i=0 ; i<serverSockets.size() ; i++ )
	{
		SocketStream serverStream( serverSockets[i] );
		clientMergePlyInfo = ClientMergePlyInfo( serverStream );
	}

	if( clientMergePlyInfo.auxProperties.size() )
	{
		typedef VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > ,  VertexFactory::DynamicFactory< Real > > Factory;
		VertexFactory::PositionFactory< Real , Dim > vFactory;
		VertexFactory::DynamicFactory< Real > dFactory( clientMergePlyInfo.auxProperties );
		Factory factory( vFactory , dFactory );
		_RunClients< Real , Dim >( clientMergePlyInfo , factory , serverSockets , sampleMS );
	}
	else
	{
		typedef VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > > Factory;
		Factory factory;
		_RunClients< Real , Dim >( clientMergePlyInfo , factory , serverSockets , sampleMS );
	}
}

/////////////////////////
// ClientPartitionInfo //
/////////////////////////
ClientMergePlyInfo::ClientMergePlyInfo( void ){}

ClientMergePlyInfo::ClientMergePlyInfo( BinaryStream &stream )
{
	auto ReadBool = [&]( bool &b )
	{
		char _b;
		if( !stream.read( _b ) ) return false;
		b = _b!=0;
		return true;
	};

	if( !stream.read( bufferSize ) ) MK_THROW( "Failed to read buffer size" );
	{
		size_t sz;
		if( !stream.read( sz ) ) MK_THROW( "Failed to read number of auxiliary properties" );
		auxProperties.resize(sz);
		for( size_t i=0 ; i<sz ; i++ ) auxProperties[i].read( stream );
	}
	if( !ReadBool( keepSeparate ) ) MK_THROW( "Failed to read keep-separate flag" );
	if( !ReadBool( verbose ) ) MK_THROW( "Failed to read verbose flag" );
}

void ClientMergePlyInfo::write( BinaryStream &stream ) const
{
	auto WriteBool = [&]( bool b )
	{
		char _b = b ? 1 : 0;
		stream.write( _b );
	};

	stream.write( bufferSize );
	{
		size_t sz = auxProperties.size();
		stream.write( sz );
		for( size_t j=0 ; j<sz ; j++ ) auxProperties[j].write( stream );
	}
	WriteBool( keepSeparate );
	WriteBool( verbose );
}
