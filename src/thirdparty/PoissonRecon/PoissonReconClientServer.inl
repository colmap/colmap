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

template< typename Real >
using AuxDataType = typename VertexFactory::DynamicFactory< Real >::VertexType;

template< typename Real >
struct AuxDataTypeSerializer : public Serializer< AuxDataType< Real > >
{
	AuxDataTypeSerializer( const std::vector< PlyProperty > &properties ) : _factory( properties ){}

	size_t size( void ) const { return _factory.bufferSize(); }

	void serialize( const AuxDataType< Real > &data , Pointer( char )buffer ) const
	{
		_factory.toBuffer( data , buffer );
	}
	void deserialize( ConstPointer( char ) buffer , AuxDataType< Real > &data ) const
	{
		data = _factory();
		_factory.fromBuffer( buffer , data );
	}
protected:
	VertexFactory::DynamicFactory< Real > _factory;
};

template< typename Real >
struct ProjectiveAuxDataTypeSerializer : public Serializer< ProjectiveData< AuxDataType< Real > , Real > >
{
	using Data = ProjectiveData< AuxDataType< Real > , Real >;

	ProjectiveAuxDataTypeSerializer( const std::vector< PlyProperty > &properties ) : _factory( properties ){}

	size_t size( void ) const { return sizeof( Real ) + _factory.bufferSize(); }

	void serialize( const Data &data , Pointer( char )buffer ) const
	{
		memcpy( buffer , &data.weight , sizeof(Real) );
		_factory.toBuffer( data.data , buffer+sizeof(Real) );
	}
	void deserialize( ConstPointer( char ) buffer , Data &data ) const
	{
		data.data = _factory();
		memcpy( &data.weight , buffer , sizeof(Real) );
		_factory.fromBuffer( buffer+sizeof(Real) , data.data );
	}
protected:
	VertexFactory::DynamicFactory< Real > _factory;
};

template< typename Real , unsigned int Dim >
using SampleDataType = DirectSum< Real , typename VertexFactory::NormalFactory< Real , Dim >::VertexType , typename VertexFactory::DynamicFactory< Real >::VertexType >;

template< typename Real , unsigned int Dim >
struct SampleDataTypeSerializer : public Serializer< SampleDataType< Real , Dim > >
{
	using Normal = typename VertexFactory::NormalFactory< Real , Dim >::VertexType;
	using AuxData = typename VertexFactory::DynamicFactory< Real >::VertexType;
	using Data = DirectSum< Real , Normal , AuxData >;

	SampleDataTypeSerializer( const std::vector< PlyProperty > &properties ) : _factory( properties ){}

	size_t size( void ) const { return sizeof(Normal) + _factory.bufferSize(); }

	void serialize( const Data &data , Pointer( char )buffer ) const
	{
		memcpy( buffer , &data.template get<0>() , sizeof(Normal) );
		_factory.toBuffer( data.template get<1>() , buffer+sizeof(Normal) );
	}
	void deserialize( ConstPointer( char ) buffer , Data &data ) const
	{
		Point< Real > p = _factory();
		memcpy( &data.template get<0>() , buffer , sizeof(Normal) );
		_factory.fromBuffer( buffer+sizeof(Normal) , p );
		data.template get<1>() = p;
	}
protected:
	VertexFactory::DynamicFactory< Real > _factory;
};

template< typename Real , unsigned int Dim >
struct ProjectiveSampleDataTypeSerializer : public Serializer< ProjectiveData< SampleDataType< Real , Dim > , Real > >
{
	using Normal = typename VertexFactory::NormalFactory< Real , Dim >::VertexType;
	using AuxData = typename VertexFactory::DynamicFactory< Real >::VertexType;
	using Data = ProjectiveData< DirectSum< Real , Normal , AuxData > , Real >;

	ProjectiveSampleDataTypeSerializer( const std::vector< PlyProperty > &properties ) : _factory( properties ){}

	size_t size( void ) const { return sizeof( Real ) + sizeof(Normal) + _factory.bufferSize(); }

	void serialize( const Data &data , Pointer( char )buffer ) const
	{
		memcpy( buffer , &data.weight , sizeof(Real) );
		memcpy( buffer + sizeof(Real) , &data.data.template get<0>() , sizeof(Normal) );
		_factory.toBuffer( data.data.template get<1>() , buffer+sizeof(Real)+sizeof(Normal) );
	}
	void deserialize( ConstPointer( char ) buffer , Data &data ) const
	{
		Point< Real > p = _factory();
		memcpy( &data.weight , buffer , sizeof(Real) );
		memcpy( &data.data.template get<0>() , buffer+sizeof(Real) , sizeof(Normal) );
		_factory.fromBuffer( buffer+sizeof(Real)+sizeof(Normal) , p );
		data.data.template get<1>() = p;
	}
protected:
	VertexFactory::DynamicFactory< Real > _factory;
};

std::string SendDataString( unsigned int phase , size_t ioBytes )
{
	std::stringstream ss;
	ss << "[SEND    " << phase << "] ";
	size_t sz;
	std::string type;
	if     ( ioBytes<(1<<10) ) sz = (ioBytes>> 0) , type = "  B";
	else if( ioBytes<(1<<20) ) sz = (ioBytes>>10) , type = "KiB";
	else if( ioBytes<(1<<30) ) sz = (ioBytes>>20) , type = "MiB";
	else                       sz = (ioBytes>>30) , type = "GiB";
	if     ( sz<10   ) ss << "   ";
	else if( sz<100  ) ss << "  ";
	else if( sz<1000 ) ss << " ";
	ss << sz << " " << type << ": ";
	return ss.str();
}
std::string ReceiveDataString( unsigned int phase , size_t ioBytes )
{
	std::stringstream ss;
	ss << "[RECEIVE " << phase << "] ";
	size_t sz;
	std::string type;
	if     ( ioBytes<(1<<10) ) sz = (ioBytes>> 0) , type = "  B";
	else if( ioBytes<(1<<20) ) sz = (ioBytes>>10) , type = "KiB";
	else if( ioBytes<(1<<30) ) sz = (ioBytes>>20) , type = "MiB";
	else                       sz = (ioBytes>>30) , type = "GiB";
	if     ( sz<10   ) ss << "   ";
	else if( sz<100  ) ss << "  ";
	else if( sz<1000 ) ss << " ";
	ss << sz << " " << type << ": ";
	return ss.str();
}

template< unsigned int Dim >
void TreeAddressesToIndices( Pointer( RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > ) nodes , size_t nodeCount )
{
	typedef RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > FEMTreeNode;

	for( size_t i=0 ; i<nodeCount ; i++ )
	{
		if( nodes[i].parent ) nodes[i].parent = (FEMTreeNode*)( nodes[i].parent - PointerAddress( nodes ) );
		else                  nodes[i].parent = (FEMTreeNode*)-1;
		if( nodes[i].children ) nodes[i].children = (FEMTreeNode*)( nodes[i].children - PointerAddress( nodes ) );
		else                    nodes[i].children = (FEMTreeNode*)-1;
	}
}

template< unsigned int Dim >
void TreeIndicesToAddresses( Pointer( RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > ) nodes , size_t nodeCount )
{
	typedef RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > FEMTreeNode;

	for( size_t i=0 ; i<nodeCount ; i++ )
	{
		if( (size_t)nodes[i].parent!=-1 ) nodes[i].parent = PointerAddress( nodes ) + (size_t)nodes[i].parent;
		else                              nodes[i].parent = NULL;
		if( (size_t)nodes[i].children!=-1 ) nodes[i].children = PointerAddress( nodes ) + (size_t)nodes[i].children;
		else                                nodes[i].children = NULL;
	}
}

template< bool ReadFromFile >
struct ClientServerStream : BinaryStream
{
	template< typename Real , unsigned int Dim >
	static void Reset( unsigned int idx , const ClientReconstructionInfo< Real , Dim > &clientReconInfo )
	{
		auto CleanUp =[]( std::string fileName )
		{
			std::ifstream fs;
			fs.open( fileName , std::ios::in | std::ios::binary );
			if( fs.is_open() )
			{
				fs.close();
				remove( fileName.c_str() );
			}
		};
		{
			CleanUp( clientReconInfo.sharedFile( idx , ClientReconstructionInfo< Real , Dim >::ShareType::BACK ) );
			CleanUp( clientReconInfo.sharedFile( idx , ClientReconstructionInfo< Real , Dim >::ShareType::CENTER ) );
			CleanUp( clientReconInfo.sharedFile( idx , ClientReconstructionInfo< Real , Dim >::ShareType::FRONT ) );
		}
	}

	ClientServerStream( ClientServerStream &&css ) : _socket( std::move( css )._socket ) , _fs( std::move( css )._fs ) , _fileName( std::move( css )._fileName ) {}

	template< typename Real , unsigned int Dim >
	ClientServerStream( SocketStream &socket , unsigned int idx , const ClientReconstructionInfo< Real , Dim > &clientReconInfo , typename ClientReconstructionInfo< Real , Dim >::ShareType shareType = ClientReconstructionInfo< Real , Dim >::CENTER , unsigned int maxTries=-1 )
		: _socket(socket)
	{
		_fileName = clientReconInfo.sharedFile( idx , shareType );
		if constexpr( ReadFromFile )
		{
			// Check that the file exists and is the right size
			auto validFile = []( std::string fileName , size_t sz )
			{
				std::ifstream fs;
				fs.open( fileName , std::ios::in | std::ios::binary );
				if( !fs.is_open() ) return false;

				fs.seekg ( 0 , fs.end );
				if( fs.tellg()!=sz )
				{
					std::cout << fs.tellg() << " / " << sz << std::endl;
					return false;
				}
				else return true;
			};
			size_t sz;
			_socket.read( sz );
			unsigned int tries=0;
			while( !validFile( _fileName , sz ) && tries<maxTries )
			{
				std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
				tries++;
			}
			_fs.open( _fileName , std::ios::in | std::ios::binary );
			if( !_fs.is_open() ) MK_THROW( "Failed to open file for reading: " , _fileName );
		}
		else
		{
			// Check that the file does not exist
			auto validFile = []( std::string fileName )
			{
				std::ifstream fs;
				fs.open( fileName , std::ios::in | std::ios::binary );
				return !fs.is_open();
			};
			unsigned int tries=0;
			while( !validFile( _fileName ) && tries<maxTries )
			{
				std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
				tries++;
			}
			if( !validFile( _fileName ) ) MK_THROW( "File exists: " , _fileName , " , " , tries , " / " , maxTries );

			_fs.open( _fileName , std::ios::out | std::ios::binary );
			if( !_fs.is_open() ) MK_THROW( "Failed to open file for writing: " , _fileName );
		}
	}
	~ClientServerStream( void )
	{
		_fs.close();
		if( ReadFromFile ) std::remove( _fileName.c_str() );
		else _socket.write( ioBytes );
	}

protected:
	std::conditional_t< ReadFromFile , std::ifstream , std::ofstream > _fs;
	SocketStream &_socket;
	std::string _fileName;

	bool _read( Pointer( unsigned char ) ptr , size_t sz )
	{
		if constexpr( ReadFromFile ) return (bool)_fs.read ( ( char * )PointerAddress( ptr ) , sz );
		else return false;
	}
	bool _write( ConstPointer( unsigned char ) ptr , size_t sz )
	{
		if constexpr( !ReadFromFile ) return (bool)_fs.write( ( const char * )PointerAddress( ptr ) , sz );
		else return false;
	}
};

struct PhaseInfo
{
	double processTime , readTime , writeTime;
	size_t readBytes , writeBytes;

	PhaseInfo( void ) : processTime(0) , readTime(0) , writeTime(0) , readBytes(0) , writeBytes(0) {}

	PhaseInfo &operator += ( const PhaseInfo &pi )
	{
		processTime += pi.processTime;
		readTime    += pi.readTime;
		readBytes   += pi.readBytes;
		writeBytes  += pi.writeBytes;
		return *this;
	}
};

//////////////////////////////
// ClientReconstructionInfo //
//////////////////////////////
template< typename Real , unsigned int Dim >
ClientReconstructionInfo< Real , Dim >::ClientReconstructionInfo( void )
{
	distributionDepth = 0;
	baseDepth = 5;
	iters = 8;
	confidence = false;
	samplesPerNode = (Real)1.5;
	dataX = (Real)32.;
	density = false;
	linearFit = false;
	mergeType = MergeType::TOPOLOGY_AND_FUNCTION;
	bufferSize = BUFFER_IO;
	filesPerDir = -1;
	outputSolution = false;
	targetValue = (Real)0.5;
	gridCoordinates = false;
}

template< typename Real , unsigned int Dim >
ClientReconstructionInfo< Real , Dim >::ClientReconstructionInfo( BinaryStream &stream )
{
	auto ReadBool = [&]( bool &b )
	{
		char _b;
		if( !stream.read( _b ) ) return false;
		b = _b!=0;
		return true;
	};
	if( !stream.read( inDir ) ) MK_THROW( "Failed to read in dir" );
	if( !stream.read( tempDir ) ) MK_THROW( "Failed to read temp dir" );
	if( !stream.read( outDir ) ) MK_THROW( "Failed to read out dir" );
	if( !stream.read( header ) ) MK_THROW( "Failed to read header" );
	if( !stream.read( bufferSize ) ) MK_THROW( "Failed to read buffer size" );
	if( !stream.read( filesPerDir ) ) MK_THROW( "Failed to read files per dir" );
	if( !stream.read( reconstructionDepth ) ) MK_THROW( "Failed to read reconstruction depth" );
	if( !stream.read( sharedDepth ) ) MK_THROW( "Failed to read shared depth" );
	if( !stream.read( distributionDepth ) ) MK_THROW( "Failed to read distribution depth" );
	if( !stream.read( baseDepth ) ) MK_THROW( "Failed to read base depth" );
	if( !stream.read( kernelDepth ) ) MK_THROW( "Failed to read kernel depth" );
	if( !stream.read( solveDepth ) ) MK_THROW( "Failed to read solveDepth depth" );
	if( !stream.read( iters ) ) MK_THROW( "Failed to read iters" );
	if( !stream.read( cgSolverAccuracy ) ) MK_THROW( "Failed to read CG-solver-accuracy" );
	if( !stream.read( targetValue ) ) MK_THROW( "Failed to read target-value" );
	if( !stream.read( pointWeight ) ) MK_THROW( "Failed to read point-weight" );
	if( !stream.read( samplesPerNode ) ) MK_THROW( "Failed to read samples-per-node" );
	if( !stream.read( dataX ) ) MK_THROW( "Failed to read data-multiplier" );
	if( !stream.read( padSize ) ) MK_THROW( "Failed to read padSize" );
	if( !stream.read( verbose ) ) MK_THROW( "Failed to read verbose" );
	if( !stream.read( mergeType ) ) MK_THROW( "Failed to read merge-type" );
	if( !ReadBool( density ) ) MK_THROW( "Failed to read density flag" );
	if( !ReadBool( linearFit ) ) MK_THROW( "Failed to read linear-fit flag" );
	if( !ReadBool( outputSolution ) ) MK_THROW( "Failed to read output-solution flag" );
	if( !ReadBool( gridCoordinates ) ) MK_THROW( "Failed to read grid-coordinates flag" );
	if( !ReadBool( ouputVoxelGrid ) ) MK_THROW( "Failed to read output-voxel-grid flag" );
	if( !ReadBool( confidence ) ) MK_THROW( "Failed to read confidence flag" );
	{
		size_t sz;
		if( !stream.read( sz ) ) MK_THROW( "Failed to read number of auxiliary properties" );
		auxProperties.resize(sz);
		for( size_t i=0 ; i<sz ; i++ ) auxProperties[i].read( stream );
	}
}

template< typename Real , unsigned int Dim >
void ClientReconstructionInfo< Real , Dim >::write( BinaryStream &stream ) const
{
	auto WriteBool = [&]( bool b )
	{
		char _b = b ? 1 : 0;
		stream.write( _b );
	};
	stream.write( inDir );
	stream.write( tempDir );
	stream.write( outDir );
	stream.write( header );
	stream.write( bufferSize );
	stream.write( filesPerDir );
	stream.write( reconstructionDepth );
	stream.write( sharedDepth );
	stream.write( distributionDepth );
	stream.write( baseDepth );
	stream.write( kernelDepth );
	stream.write( solveDepth );
	stream.write( iters );
	stream.write( cgSolverAccuracy );
	stream.write( targetValue );
	stream.write( pointWeight );
	stream.write( samplesPerNode );
	stream.write( dataX );
	stream.write( padSize );
	stream.write( verbose );
	stream.write( mergeType );
	WriteBool( density );
	WriteBool( linearFit );
	WriteBool( outputSolution );
	WriteBool( gridCoordinates );
	WriteBool( ouputVoxelGrid );
	WriteBool( confidence );
	{
		size_t sz = auxProperties.size();
		stream.write( sz );
		for( size_t j=0 ; j<sz ; j++ ) auxProperties[j].write( stream );
	}
}

template< typename Real , unsigned int Dim >
std::string ClientReconstructionInfo< Real , Dim >::sharedFile( unsigned int idx , ShareType shareType ) const
{
	std::stringstream sStream;
	switch( shareType )
	{
	case BACK:   sStream << PointPartition::FileDir( tempDir , header ) << "." << idx << ".back.shared"  ; break;
	case CENTER: sStream << PointPartition::FileDir( tempDir , header ) << "." << idx << ".shared"       ; break;
	case FRONT:  sStream << PointPartition::FileDir( tempDir , header ) << "." << idx << ".front.shared" ; break;
	default: MK_THROW( "Unrecognized share type: " , shareType );
	}
	return sStream.str();
}

