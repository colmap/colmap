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

template< typename Real , unsigned int Dim >
size_t _SampleCount( std::string in , std::vector< PlyProperty > &auxProperties )
{
	char *ext = GetFileExtension( in.c_str() );
	if( strcasecmp( ext , "ply" ) ) MK_THROW( "Only .ply files supported: "  , in );
	delete[] ext;

	size_t vNum;
	typedef VertexFactory::Factory< Real , typename VertexFactory::PositionFactory< Real , Dim > , typename VertexFactory::NormalFactory< Real , Dim > > Factory;
	Factory factory;
	bool *readFlags = new bool[ factory.plyReadNum() ];
	int fileType = PLY::ReadVertexHeader( in , factory , readFlags , auxProperties , vNum );
	if( fileType==PLY_ASCII ) MK_THROW( "Point set must be in binary format" );
	if( !factory.template plyValidReadProperties<0>( readFlags ) ) MK_THROW( "Ply file does not contain positions" );
	if( !factory.template plyValidReadProperties<1>( readFlags ) ) MK_THROW( "Ply file does not contain normals" );
	delete[] readFlags;
	return vNum;
}

template< typename Factory , typename VertexFunctor /* = std::function< void ( Factory::VertexType ) > */ >
void _ProcessPLY( std::string in , std::pair< size_t , size_t > range , const Factory &factory , VertexFunctor vf )
{
	using Vertex = typename Factory::VertexType;
	std::vector< std::string > elist = { std::string( "vertex" ) };

	float version;
	int file_type;

	PlyFile *ply = PlyFile::Read( in , elist , file_type , version );
	if( !ply ) MK_THROW( "Could not create ply file for reading: " , in );
	if( file_type==PLY_ASCII ) MK_THROW( "Only binary file type supported" );

	size_t vCount;
	std::vector< PlyProperty > plist = ply->get_element_description( std::string( "vertex" ) , vCount );
	if( !plist.size() ) MK_THROW( "Could not read element properties: vertex" );
	if( range.second==-1 ) range.second = vCount;
	if( range.first>=range.second ) MK_THROW( "Bad Range: [ " , range.first , " , " , range.second , " )" );
	if( range.second>vCount )
	{
		MK_WARN( "Max range too large, resetting" );
		range.second = vCount;
	}

	size_t leftToReadCount = range.second - range.first;
	size_t vSize = 0;
	if constexpr( Factory::IsStaticallyAllocated() ) vSize = sizeof( Vertex );
	else                                             vSize = factory.bufferSize();

	for( unsigned int i=0 ; i<factory.plyReadNum() ; i++)
	{
		PlyProperty prop;
		if constexpr( Factory::IsStaticallyAllocated() ) prop = factory.plyStaticReadProperty(i);
		else                                             prop = factory.plyReadProperty(i);
		ply->get_property( std::string( "vertex" ) , &prop );
	}

	size_t sizeOnDisk = 0;
	for( unsigned int i=0 ; i<plist.size() ; i++ ) sizeOnDisk += ply_type_size[ plist[i].external_type ];

#if defined( _WIN32 ) || defined( _WIN64 )
	_fseeki64( ply->fp , sizeOnDisk * range.first , SEEK_CUR );
#else // !_WIN32 && !_WIN64
	fseek( ply->fp , sizeOnDisk * range.first , SEEK_CUR );
#endif // _WIN32 || _WIN64

	Vertex vertex = factory();
	Pointer( char ) buffer = NewPointer< char >( factory.bufferSize() );
	for( size_t i=range.first ; i<range.second ; i++ )
	{
		if constexpr( Factory::IsStaticallyAllocated() ) ply->get_element( (void *)&vertex );
		else
		{
			ply->get_element( PointerAddress( buffer ) );
			factory.fromBuffer( buffer , vertex );
		}
		vf( vertex );
	}
	DeletePointer( buffer );

	delete ply;
}

template< typename Real , unsigned int Dim , typename Factory >
void _MergeSlabs( std::string inDir , std::string outDir , std::string header , unsigned int clientCount , std::pair< unsigned int , unsigned int > slabRange , unsigned int slabs , unsigned int filesPerDir , const Factory &factory , size_t bufferSize )
{
	using Vertex = typename Factory::VertexType;

	Vertex v = factory();

	for( unsigned int s=slabRange.first ; s<slabRange.second ; s++ )
	{
		std::string outFileName = PointPartition::FileName( outDir , header , s , slabs , filesPerDir );
		PointPartition::BufferedBinaryOutputDataStream< Factory > outStream( outFileName.c_str() , factory , bufferSize );
		for( unsigned int c=0 ; c<clientCount ; c++ )
		{
			std::string inFileName = PointPartition::FileName( inDir , header , c , s , slabs , filesPerDir );
			PointPartition::BufferedBinaryInputDataStream< Factory > inStream( inFileName.c_str() , factory , bufferSize );
			while( inStream.read( v ) ) outStream.write( v );
		}
	}
}

template< typename Real , unsigned int Dim , typename Factory >
std::vector< size_t > _PartitionIntoSlabs( std::string in , std::string dir , std::string header , unsigned int clientIndex , std::pair< size_t , size_t > range , unsigned int slabs , unsigned int filesPerDir , XForm< Real , Dim+1 > xForm , const Factory &factory , size_t bufferSize )
{
	using Vertex = typename Factory::VertexType;
	using _XForm = typename Factory::Transform;

	_XForm _xForm(xForm);
	std::vector< size_t > slabSizes( slabs , 0 );

	using OutputPointStream = OutputDataStream< Vertex >;

	std::vector< OutputPointStream * > outStreams( slabs );
	for( unsigned int s=0 ; s<slabs ; s++ )
	{
		std::string fileName = PointPartition::FileName( dir , header , clientIndex , s , slabs , filesPerDir );
		outStreams[s] = new PointPartition::BufferedBinaryOutputDataStream< Factory >( fileName.c_str() , factory , bufferSize );
	}

	size_t outOfRangeCount = 0;
	auto vertexFunctor = [&]( Vertex v )
	{
		_xForm.inPlace( v );
		Point< Real , Dim > p = v.template get<0>();
		int slab = (int)floor( p[Dim-1] * slabs );
		if( slab>=0 && slab<(int)slabs )
		{
			outStreams[slab]->write( v );
			slabSizes[slab]++;
		}
		else outOfRangeCount++;
	};
	_ProcessPLY( in , range , factory , vertexFunctor );
	for( unsigned int i=0 ; i<slabs ; i++ ) delete outStreams[i];
	if( outOfRangeCount ) MK_WARN( "Out of range count: " , outOfRangeCount );
	return slabSizes;
}

template< typename Real , unsigned int Dim , typename Factory , bool ExtendedAxes=true >
PointExtent::Extent< Real , Dim , ExtendedAxes > _GetExtent( std::string in , std::pair< size_t , size_t > range , const Factory &factory )
{
	using Vertex = typename Factory::VertexType;
	PointExtent::Extent< Real , Dim , ExtendedAxes > extent;
	_ProcessPLY( in , range , factory , [&]( const Vertex &vertex ){ extent.add( vertex.template get<0>() ); } );
	return extent;
}

template< typename Real , unsigned int Dim >
std::vector< PlyProperty > _GetUnprocessedProperties( std::string in )
{
	char *ext = GetFileExtension( in.c_str() );
	if( strcasecmp( ext , "ply" ) ) MK_THROW( "Expected .ply file" );
	delete[] ext;

	std::vector< PlyProperty > unprocessedProperties;
	{
		typedef VertexFactory::Factory< Real , typename VertexFactory::PositionFactory< Real , Dim > , typename VertexFactory::NormalFactory< Real , Dim > > Factory;
		Factory factory;
		bool *readFlags = new bool[ factory.plyReadNum() ];
		PLY::ReadVertexHeader( in , factory , readFlags , unprocessedProperties );
		if( !factory.template plyValidReadProperties<0>( readFlags ) ) MK_THROW( "Ply file does not contain positions" );
		if( !factory.template plyValidReadProperties<1>( readFlags ) ) MK_THROW( "Ply file does not contain normals" );
		delete[] readFlags;
	}
	return unprocessedProperties;
}

////////////
// Server //
////////////

template< typename Real , unsigned int Dim >
std::pair< PointPartition::PointSetInfo< Real , Dim > , PointPartition::Partition > RunServer
(
	std::vector< Socket > &clientSockets ,
	ClientPartitionInfo< Real > clientPartitionInfo ,
	bool loadBalance
)
{
	Timer timer;
	clientPartitionInfo.clientCount = ( unsigned int )clientSockets.size();
	for( unsigned int c=0 ; c<clientSockets.size() ; c++ )
	{
		SocketStream clientSocketStream( clientSockets[c] );
		clientPartitionInfo.write( clientSocketStream );
	}
	for( unsigned int c=0 ; c<clientSockets.size() ; c++ ) SocketStream( clientSockets[c] ).write( c );

	PointPartition::PointSetInfo< Real , Dim > pointSetInfo( clientPartitionInfo.slabs );
	pointSetInfo.header = clientPartitionInfo.outHeader;
	pointSetInfo.filesPerDir = clientPartitionInfo.filesPerDir;

	// Create the directory for the slabs 
	PointPartition::CreatePointSlabDirs( PointPartition::FileDir( clientPartitionInfo.outDir , clientPartitionInfo.outHeader ) , clientPartitionInfo.slabs , clientPartitionInfo.filesPerDir );

	/////////////
	// Phase 1 //
	/////////////
	// Get the number of samples
	size_t sampleCount = _SampleCount< Real , Dim >( clientPartitionInfo.in , pointSetInfo.auxiliaryProperties );
	if( clientPartitionInfo.verbose ) std::cout << "Samples: " << sampleCount << std::endl;

	// Send the partitions to the clients
	for( unsigned int c=0 ; c<clientSockets.size() ; c++ )
	{
		size_t start = ( sampleCount * c )/clientSockets.size();
		size_t end = ( sampleCount * (c+1) )/clientSockets.size();
		if( clientPartitionInfo.verbose ) std::cout << "[ " << start << " , "  << end << " )" << std::endl;
		SocketStream( clientSockets[c] ).write( std::make_pair( start , end ) );
	}

	/////////////
	// Phase 2 //
	/////////////
	// Merge the clients' extents and get the direction of maximal extent
	PointExtent::Extent< Real , Dim > e;
	for( unsigned int c=0 ; c<clientSockets.size() ; c++ )
	{
		PointExtent::Extent< Real , Dim > _e;
		SocketStream( clientSockets[c] ).read( _e );
		e = e + _e;
	}
	pointSetInfo.modelToUnitCube = PointExtent::GetXForm( e , clientPartitionInfo.scale , clientPartitionInfo.sliceDir );

	// Send the transformation to the clients
	for( unsigned int c=0 ; c<clientSockets.size() ; c++ ) SocketStream( clientSockets[c] ).write( pointSetInfo.modelToUnitCube );

	/////////////
	// Phase 3 //
	/////////////
	pointSetInfo.pointsPerSlab.resize( clientPartitionInfo.slabs , 0 );
	for( unsigned int c=0 ; c<clientSockets.size() ; c++ )
	{
		std::vector< size_t > slabSizes;
		SocketStream( clientSockets[c] ).read( slabSizes );
		if( slabSizes.size()!=clientPartitionInfo.slabs ) MK_THROW( "Unexpected number of slabs: " , slabSizes.size() , " != " , clientPartitionInfo.slabs );
		for( unsigned int i=0 ; i<clientPartitionInfo.slabs ; i++ ) pointSetInfo.pointsPerSlab[i] += slabSizes[i];
	}
	if( clientPartitionInfo.verbose )
	{
		std::cout << "Partitions:" << std::endl;
		for( unsigned int i=0 ; i<pointSetInfo.pointsPerSlab.size() ; i++ ) std::cout << "\t" << i << "] " << pointSetInfo.pointsPerSlab[i] << std::endl;
	}

	// Compute the assignment of slabs to clients
	PointPartition::Partition pointPartition( (unsigned int)clientSockets.size() , pointSetInfo.pointsPerSlab );
	if( loadBalance ) pointPartition.optimize( true );
	for( unsigned int c=0 ; c<clientSockets.size() ; c++ )
	{
		std::pair< unsigned int , unsigned int > range = pointPartition.range( c );
		if( clientPartitionInfo.verbose ) std::cout << "Range[ " << c << " ]: [ " << range.first << " , "  << range.second << " )" << std::endl;
		SocketStream( clientSockets[c] ).write( range );
	}

	// Check that the clients are done
	for( unsigned int c=0 ; c<clientSockets.size() ; c++ )
	{
		unsigned int done;
		SocketStream( clientSockets[c] ).read( done );
	}

	// Let the clients know that everybody else is done so they can clean up the temporary files
	for( unsigned int c=0 ; c<clientSockets.size() ; c++ )
	{
		unsigned int done = 1;
		SocketStream( clientSockets[c] ).write( done );
	}

	return std::make_pair( pointSetInfo , pointPartition );
}


////////////
// Client //
////////////

template< typename Real , unsigned int Dim , typename Factory >
void _RunClients
(
	ClientPartitionInfo< Real > clientPartitionInfo ,
	const Factory &factory ,
	std::vector< Socket > &serverSockets
)
{
	std::vector< unsigned int > clientIndices( serverSockets.size() );

	for( unsigned int c=0 ; c<serverSockets.size() ; c++ ) SocketStream( serverSockets[c] ).read( clientIndices[c] );

	int maxFiles = 2*clientPartitionInfo.slabs;
#ifdef _WIN32
	if( _setmaxstdio( maxFiles )!=maxFiles ) MK_THROW( "Could not set max file handles: " , maxFiles );
#else // !_WIN32
	struct rlimit rl;
	getrlimit( RLIMIT_NOFILE , &rl ); 
	rl.rlim_cur = maxFiles+3; 
	setrlimit( RLIMIT_NOFILE , &rl );
#endif // _WIN32

	for( unsigned int i=0 ; i<serverSockets.size() ; i++ ) PointPartition::CreatePointSlabDirs( PointPartition::FileDir( clientPartitionInfo.tempDir , clientPartitionInfo.outHeader , clientIndices[i] ) , clientPartitionInfo.slabs , clientPartitionInfo.filesPerDir );

	/////////////
	// Phase 1 //
	/////////////
	std::vector< std::pair< size_t , size_t > > ranges( serverSockets.size() );
	for( unsigned int i=0 ; i<serverSockets.size() ; i++ )
	{
		// Get the client's range
		SocketStream( serverSockets[i] ).read( ranges[i] );
		if( clientPartitionInfo.verbose ) std::cout << "Got range: " << MemoryInfo::PeakMemoryUsageMB() << " (MB)" << std::endl;
		// Get the extent and send to the client
		PointExtent::Extent< Real , Dim > e = _GetExtent< Real , Dim >( clientPartitionInfo.in , ranges[i] , factory );
		SocketStream( serverSockets[i] ).write( e );
		if( clientPartitionInfo.verbose ) std::cout << "Sent extent: " << MemoryInfo::PeakMemoryUsageMB() << " (MB)" << std::endl;
	}

	/////////////
	// Phase 2 //
	/////////////
	for( unsigned int i=0 ; i<serverSockets.size() ; i++ )
	{
		XForm< Real , Dim+1 > xForm;
		SocketStream( serverSockets[i] ).read( xForm );
		if( clientPartitionInfo.verbose ) std::cout << "Got transform: " << MemoryInfo::PeakMemoryUsageMB() << " (MB)" << std::endl;

		PointPartition::CreatePointSlabDirs( PointPartition::FileDir( clientPartitionInfo.tempDir , clientPartitionInfo.outHeader , clientIndices[i] ) , clientPartitionInfo.slabs , clientPartitionInfo.filesPerDir );
		std::vector< size_t > slabSizes = _PartitionIntoSlabs< Real , Dim >( clientPartitionInfo.in , clientPartitionInfo.tempDir , clientPartitionInfo.outHeader , clientIndices[i] , ranges[i] , clientPartitionInfo.slabs , clientPartitionInfo.filesPerDir , xForm , factory , clientPartitionInfo.bufferSize );
		SocketStream( serverSockets[i] ).write( slabSizes );
		if( clientPartitionInfo.verbose ) std::cout << "Wrote slab sizes: " << MemoryInfo::PeakMemoryUsageMB() << " (MB)" << std::endl;
	}

	/////////////
	// Phase 3 //
	/////////////
	for( unsigned int i=0 ; i<serverSockets.size() ; i++ )
	{
		std::pair< unsigned int , unsigned int > slabRange;
		SocketStream( serverSockets[i] ).read( slabRange );
		if( clientPartitionInfo.verbose ) std::cout << "Slab range: [ " << slabRange.first << " , " << slabRange.second << " )" << std::endl;
		_MergeSlabs< Real , Dim >( clientPartitionInfo.tempDir , clientPartitionInfo.outDir , clientPartitionInfo.outHeader , clientPartitionInfo.clientCount , slabRange , clientPartitionInfo.slabs , clientPartitionInfo.filesPerDir , factory , clientPartitionInfo.bufferSize );
		if( clientPartitionInfo.verbose ) std::cout << "Merged slabs: "  << MemoryInfo::PeakMemoryUsageMB() << " (MB)" << std::endl;

		// Notify the server that you're done
		unsigned int done = 1;
		SocketStream( serverSockets[i] ).write( done );
	}

	for( unsigned int i=0 ; i<serverSockets.size() ; i++ )
	{
		// Confirm that all the other clients are done and clean up
		unsigned int done = 1;
		SocketStream( serverSockets[i] ).read( done );

		PointPartition::RemovePointSlabDirs( PointPartition::FileDir( clientPartitionInfo.tempDir , clientPartitionInfo.outHeader , clientIndices[i] ) );
	}
}

template< typename Real , unsigned int Dim >
void RunClients( std::vector< Socket > &serverSockets )
{
	ClientPartitionInfo< Real > clientPartitionInfo;
	for( unsigned int i=0 ; i<serverSockets.size() ; i++ )
	{
		SocketStream serverSocketStream( serverSockets[i] );
		clientPartitionInfo = ClientPartitionInfo< Real >( serverSocketStream );
	}
	std::vector< PlyProperty > unprocessedProperties = _GetUnprocessedProperties< Real , Dim >( clientPartitionInfo.in );

	if( unprocessedProperties.size() )
	{
		typedef VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > , VertexFactory::NormalFactory< Real , Dim > , VertexFactory::DynamicFactory< Real > > Factory;
		VertexFactory::PositionFactory< Real , Dim > vFactory;
		VertexFactory::NormalFactory< Real , Dim > nFactory;
		VertexFactory::DynamicFactory< Real > dFactory( unprocessedProperties );
		Factory factory( vFactory , nFactory , dFactory );
		_RunClients< Real , Dim >( clientPartitionInfo , factory , serverSockets );
	}
	else
	{
		typedef VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > , VertexFactory::NormalFactory< Real , Dim > > Factory;
		Factory factory;
		_RunClients< Real , Dim >( clientPartitionInfo , factory , serverSockets );
	}
}

/////////////////////////
// ClientPartitionInfo //
/////////////////////////
template< typename Real >
ClientPartitionInfo< Real >::ClientPartitionInfo( void ) : scale((Real)1.1) , sliceDir(-1) , verbose(false) , slabs(0) , filesPerDir(-1) , bufferSize(BUFFER_IO) , clientCount(0) {}

template< typename Real >
ClientPartitionInfo< Real >::ClientPartitionInfo( BinaryStream &stream )
{
	auto ReadBool = [&]( bool &b )
	{
		char _b;
		if( !stream.read( _b ) ) return false;
		b = _b!=0;
		return true;
	};
	if( !stream.read( in ) ) MK_THROW( "Failed to read in" );
	if( !stream.read( tempDir ) ) MK_THROW( "Failed to read temp dir" );
	if( !stream.read( outDir ) ) MK_THROW( "Failed to read out dir" );
	if( !stream.read( outHeader ) ) MK_THROW( "Failed to read out header" );
	if( !stream.read( slabs ) ) MK_THROW( "Failed to read slabs" );
	if( !stream.read( filesPerDir ) ) MK_THROW( "Failed to read files per dir" );
	if( !stream.read( bufferSize ) ) MK_THROW( "Failed to read buffer size" );
	if( !stream.read( scale ) ) MK_THROW( "Failed to read scale" );
	if( !stream.read( clientCount ) ) MK_THROW( "Failed to read client count" );
	if( !stream.read( sliceDir ) ) MK_THROW( "Failed to read slice direction" );
	if( !ReadBool( verbose ) ) MK_THROW( "Failed to read verbose flag" );
}

template< typename Real >
void ClientPartitionInfo< Real >::write( BinaryStream &stream ) const
{
	auto WriteBool = [&]( bool b )
	{
		char _b = b ? 1 : 0;
		stream.write( _b );
	};
	stream.write( in );
	stream.write( tempDir );
	stream.write( outDir );
	stream.write( outHeader );
	stream.write( slabs );
	stream.write( filesPerDir );
	stream.write( bufferSize );
	stream.write( scale );
	stream.write( clientCount );
	stream.write( sliceDir );
	WriteBool( verbose );
}
