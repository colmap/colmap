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

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
struct Client
{
	typedef typename FEMTree< Dim , Real >::template DensityEstimator< Reconstructor::WeightDegree > DensityEstimator;
	typedef typename FEMTree< Dim , Real >::template ApproximatePointInterpolationInfo< Real , 0 , Reconstructor::Poisson::ConstraintDual< Dim , Real > , Reconstructor::Poisson::SystemDual< Dim , Real > > ApproximatePointInterpolationInfo;
	typedef IsotropicUIntPack< Dim , FEMDegreeAndBType< Degree , BType >::Signature > Sigs;
	typedef IsotropicUIntPack< Dim , Degree > Degrees;
	typedef IsotropicUIntPack< Dim , FEMDegreeAndBType< Reconstructor::Poisson::NormalDegree , DerivativeBoundary< BType , 1 >::BType >::Signature > NormalSigs;
	static const unsigned int DataSig = FEMDegreeAndBType< Reconstructor::DataDegree , BOUNDARY_FREE >::Signature;
	typedef VertexFactory::DynamicFactory< Real > AuxDataFactory;
	typedef typename AuxDataFactory::VertexType AuxData;
	typedef VertexFactory::Factory< Real , VertexFactory::PositionFactory< Real , Dim > , VertexFactory::Factory< Real , VertexFactory::NormalFactory< Real , Dim > , AuxDataFactory > > InputSampleFactory;
	typedef VertexFactory::Factory< Real , VertexFactory::NormalFactory< Real , Dim > , AuxDataFactory > InputSampleDataFactory;
	typedef DirectSum< Real , Point< Real , Dim > , typename AuxDataFactory::VertexType > InputSampleDataType;
	typedef DirectSum< Real , Point< Real , Dim > , InputSampleDataType > InputSampleType;
	typedef InputDataStream< InputSampleType > InputPointStream;
	typedef RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > FEMTreeNode;
	using BoundaryData = std::conditional_t< Dim==3 , typename LevelSetExtractor< Real , 2 , Point< Real , 2 > >::TreeSliceValuesAndVertexPositions , char >;

	~Client( void );

	static std::function< int ( Point< Real , Dim > ) > PointDepthFunctor( Real begin , Real end , unsigned int padSize , unsigned int minDepth , unsigned int maxDepth );
protected:
	unsigned int _index;
	SocketStream _serverSocket;
	std::pair< unsigned int , unsigned int > _range;
	XForm< Real , Dim+1 > _modelToUnitCube;
	FEMTree< Dim , Real > _tree;

	DensityEstimator *_density;																							// Phases [1,7]
	SparseNodeData< Point< Real , Dim > , NormalSigs > *_normalInfo , *_paddedNormalInfo;								// Phases [1,3]
	std::vector< typename FEMTree< Dim , Real >::PointSample >       _samples;											// Phases [1,5]
	std::vector< typename FEMTree< Dim , Real >::PointSample > _paddedSamples;											// Phases [1,3]
	std::vector< InputSampleDataType > _sampleData , _paddedSampleData;													// Phases [1,3]
	DenseNodeData< Real , Sigs > _constraints;																			// Phases [3,5]
	DenseNodeData< Real , Sigs > _solution;																				// Phases [5,7]
	ApproximatePointInterpolationInfo *_iInfo;																			// Phases [3,5]
	SparseNodeData< ProjectiveData< AuxData , Real > , IsotropicUIntPack< Dim , DataSig > > _auxDataField;				// Phases [3,7]

	static std::pair< unsigned int , unsigned int > _PaddedRange( std::pair< unsigned int , unsigned int > range , unsigned int depth , unsigned int padSize );

	struct _State3
	{
		_State3( void ) : subNodes( NullPointer( FEMTreeNode ) ) {}
		~_State3( void ){ DeletePointer( subNodes ); }
		Pointer( FEMTreeNode ) subNodes;
		size_t subNodeCount;
		DenseNodeData< Real , Sigs > constraints;
		ApproximatePointInterpolationInfo iInfo;
		SparseNodeData< ProjectiveData< AuxData , Real > , IsotropicUIntPack< Dim , DataSig > > auxDataField;
	};
	struct _State5
	{
		using Data = ProjectiveData< AuxData , Real >;

		_State5( void ) : auxDataField(NULL) , subNodes( NullPointer( FEMTreeNode ) ) {}
		~_State5( void )
		{
			DeletePointer( subNodes );
			delete auxDataField;
		}

		struct BoundaryInfo
		{
			using SliceSigs = typename Sigs::Transpose::Rest::Transpose;
			FEMTree< Dim-1 , Real > *tree;
			DenseNodeData< Real , SliceSigs > solution , dSolution;
			BoundaryInfo( void ) : tree(NULL) {}
			~BoundaryInfo( void ){ delete tree; }
		};

		Pointer( FEMTreeNode ) subNodes;
		DenseNodeData< Real , Sigs > solution;
		SparseNodeData< Data , IsotropicUIntPack< Dim , DataSig > > *auxDataField;
		std::pair< BoundaryInfo , BoundaryInfo > boundaryInfo;
	};
	struct _State7
	{
		BoundaryData *backBoundary , *frontBoundary;
		std::vector< std::vector< Real > > backDValues , frontDValues;
		Real isoValue;

		_State7( void ) : backBoundary(NULL) , frontBoundary(NULL) , isoValue((Real)0.5){}
		~_State7( void )
		{
			delete  backBoundary;
			delete frontBoundary;
		}
	};

	PhaseInfo _phase1( const ClientReconstructionInfo< Real , Dim > &clientReconInfo, Profiler &profiler );
	PhaseInfo _phase3( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , _State3 &state3 , Profiler &profiler );
	PhaseInfo _phase5( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , Profiler &profiler );
	PhaseInfo _phase7( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , Profiler &profiler );

	size_t _receive1( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , Profiler &profiler );
	void _process1( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , ProjectiveData< Point< Real , 2 > , Real > &pointDepthAndWeight , std::pair< size_t , size_t > &nodeCounts , Profiler &profiler );
	size_t _send1( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , ProjectiveData< Real , Real > pointWeight , std::pair< size_t , size_t > nodeCounts , Profiler &profiler );

	size_t _receive3( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , ProjectiveData< Real , Real > &cumulativePointWeight , Profiler &profiler );
	void _process3( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , ProjectiveData< Real , Real > cumulativePointWeight , _State3 &state3 , Profiler &profiler );
	size_t _send3( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , const _State3 &state3 , Profiler &profiler );

	size_t _receive5( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , _State5 &state5 , Profiler &profiler );
	std::pair< double , double > _process5( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , _State5 &state5 , Profiler &profiler );
	size_t _send5( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , const _State5 &state5 , Profiler &profiler );

	size_t _receive7( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , _State7 &state7 , Profiler &profiler );
	void _process7( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , _State7 &state7 , Profiler &profiler );

	Client( void );
	Client( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , BinaryStream &stream , unsigned int phase );
	void _write( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , BinaryStream &stream , unsigned int phase ) const;

	template< typename _Real , unsigned int _Dim , BoundaryType _BType , unsigned int _Degree >
	friend void RunClient( std::vector< Socket > &serverSockets , unsigned int sampleMS );

	template< bool HasGradients , bool HasDensity >
	void _writeMeshWithData( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , _State7 &state7 , XForm< Real , Dim+1 > unitCubeToModel );
};

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
void RunClient( std::vector< Socket > &serverSockets , unsigned int sampleMS )
{
	std::vector< SocketStream > serverSocketStreams( serverSockets.size() );
	for( unsigned int i=0 ; i<serverSocketStreams.size() ; i++ ) serverSocketStreams[i] = SocketStream( serverSockets[i] );
	Profiler profiler( sampleMS );
	std::vector< FileStream > cacheFiles;
	ClientReconstructionInfo< Real , Dim > clientReconInfo;
	std::vector< unsigned int > clientIndices( serverSocketStreams.size() );

	for( unsigned int i=0 ; i<serverSocketStreams.size() ; i++ ) clientReconInfo = ClientReconstructionInfo< Real , Dim >( serverSocketStreams[i] );
	for( unsigned int i=0 ; i<serverSocketStreams.size() ; i++ ) serverSocketStreams[i].read( clientIndices[i] );

	if( serverSocketStreams.size()>1 ) for( unsigned int idx=0 ; idx<serverSocketStreams.size() ; idx++ ) cacheFiles.emplace_back( std::tmpfile() );

	Client< Real , Dim , BType , Degree > *client = NULL;
	if( serverSocketStreams.size()==1 )
	{
		client = new Client< Real , Dim , BType , Degree >();
		client->_serverSocket = serverSocketStreams[0];
		client->_index        = clientIndices[0];
	}

	// Phase 1
	{
		PhaseInfo phaseInfo;
		double cacheTime = 0;
		size_t cacheBytes = 0;

		profiler.reset();
		for( unsigned int i=0 ; i<serverSocketStreams.size() ; i++ )
		{
			if( serverSocketStreams.size()>1 )
			{
				client = new Client< Real , Dim , BType , Degree >();
				client->_serverSocket = serverSocketStreams[i];
				client->_index        = clientIndices[i];
			}

			phaseInfo += client->_phase1( clientReconInfo , profiler );

			if( serverSocketStreams.size()>1 )
			{
				Timer timer;

				cacheFiles[i].reset();
				cacheFiles[i].ioBytes = 0;
				client->_write( clientReconInfo , cacheFiles[i] , 1 );
				cacheBytes += cacheFiles[i].ioBytes;
				delete client;
				client = NULL;

				cacheTime += timer.wallTime();
			}
		}
		if( clientReconInfo.verbose>0 )
		{
			StreamFloatPrecision sfp( std::cout , 1 );
			std::cout << ReceiveDataString( 1 , phaseInfo.readBytes ) << phaseInfo.readTime << " (s)" << std::endl;
			std::cout << "[PROCESS 1]         : " << phaseInfo.processTime << " (s), " << profiler(false) << std::endl;
			std::cout << SendDataString( 1 , phaseInfo.writeBytes ) << phaseInfo.writeTime << " (s)" << std::endl;
			if( serverSocketStreams.size()>1 ) std::cout << "[CACHE   1]         : " << cacheTime << " (s) , " << (cacheBytes>>20) << " (MB)" << std::endl;
		}
	}

	// Phase 3
	{
		PhaseInfo phaseInfo;
		double cacheTime = 0;
		size_t cacheBytes = 0;

		profiler.reset();
		for( unsigned int i=0 ; i<serverSocketStreams.size() ; i++ )
		{
			typename Client< Real , Dim , BType , Degree >::_State3 state3;

			if( serverSocketStreams.size()>1 )
			{
				Timer timer;
				cacheFiles[i].ioBytes = 0;
				cacheFiles[i].reset();
				client = new Client< Real , Dim , BType , Degree >( clientReconInfo , cacheFiles[i] , 3 );
				cacheTime += timer.wallTime();
				cacheBytes += cacheFiles[i].ioBytes;
				client->_serverSocket = serverSocketStreams[i];
			}

			phaseInfo += client->_phase3( clientReconInfo , state3 , profiler );

			if( serverSocketStreams.size()>1 )
			{
				Timer timer;
				cacheFiles[i].ioBytes = 0;
				cacheFiles[i].reset();
				client->_write( clientReconInfo , cacheFiles[i] , 3 );
				delete client;
				client = NULL;
				cacheTime += timer.wallTime();
				cacheBytes += cacheFiles[i].ioBytes;
			}
		}

		if( clientReconInfo.verbose>0 )
		{
			StreamFloatPrecision sfp( std::cout , 1 );
			std::cout << ReceiveDataString( 3 , phaseInfo.readBytes ) << phaseInfo.readTime << " (s)" << std::endl;
			std::cout << "[PROCESS 3]         : " << phaseInfo.processTime << " (s), " << profiler(false) << std::endl;
			std::cout << SendDataString( 3 , phaseInfo.writeBytes ) << phaseInfo.writeTime << " (s)" << std::endl;
			if( serverSocketStreams.size()>1 ) std::cout << "[CACHE   3]         : " << cacheTime << " (s) , " << (cacheBytes>>20) << " (MB)" << std::endl;
		}
	}

	// Phase 5
	{
		PhaseInfo phaseInfo;
		double cacheTime = 0;
		size_t cacheBytes = 0;

		profiler.reset();
		for( unsigned int i=0 ; i<serverSocketStreams.size() ; i++ )
		{
			if( serverSocketStreams.size()>1 )
			{
				Timer timer;
				cacheFiles[i].ioBytes = 0;
				cacheFiles[i].reset();
				client = new Client< Real , Dim , BType , Degree >( clientReconInfo , cacheFiles[i] , 5 );
				cacheTime += timer.wallTime();
				cacheBytes += cacheFiles[i].ioBytes;
				client->_serverSocket = serverSocketStreams[i];
			}

			phaseInfo += client->_phase5( clientReconInfo , profiler );

			if( serverSocketStreams.size()>1 )
			{
				Timer timer;
				cacheFiles[i].ioBytes = 0;
				cacheFiles[i].reset();
				client->_write( clientReconInfo , cacheFiles[i] , 5 );
				delete client;
				client = NULL;
				cacheTime += timer.wallTime();
				cacheBytes += cacheFiles[i].ioBytes;
			}
		}

		if( clientReconInfo.verbose>0 )
		{
			StreamFloatPrecision sfp( std::cout , 1 );
			std::cout << ReceiveDataString( 5 , phaseInfo.readBytes ) << phaseInfo.readTime << " (s)" << std::endl;
			std::cout << "[PROCESS 5]         : " << phaseInfo.processTime << " (s), " << profiler(false) << std::endl;
			std::cout << SendDataString( 5 , phaseInfo.writeBytes ) << phaseInfo.writeTime << " (s)" << std::endl;
			if( serverSocketStreams.size()>1 ) std::cout << "[CACHE   5]         : " << cacheTime << " (s) , " << (cacheBytes>>20) << " (MB)" << std::endl;
		}
	}
	// Phase 7
	{
		PhaseInfo phaseInfo;
		double cacheTime = 0;
		size_t cacheBytes = 0;

		profiler.reset();
		for( unsigned int i=0 ; i<serverSocketStreams.size() ; i++ )
		{
			if( serverSocketStreams.size()>1 )
			{
				Timer timer;
				cacheFiles[i].ioBytes = 0;
				cacheFiles[i].reset();
				client = new Client< Real , Dim , BType , Degree >( clientReconInfo , cacheFiles[i] , 7 );
				cacheTime += timer.wallTime();
				cacheBytes += cacheFiles[i].ioBytes;
				client->_serverSocket = serverSocketStreams[i];
			}

			phaseInfo += client->_phase7( clientReconInfo , profiler );

			if( serverSocketStreams.size()>1 )
			{
				delete client;
				client = NULL;
			}
		}

		if( clientReconInfo.verbose>0 )
		{
			StreamFloatPrecision sfp( std::cout , 1 );
			std::cout << ReceiveDataString( 7 , phaseInfo.readBytes ) << phaseInfo.readTime << " (s)" << std::endl;
			std::cout << "[PROCESS 7]         : " << phaseInfo.processTime << " (s), " << profiler(false) << std::endl;
			if( serverSocketStreams.size()>1 ) std::cout << "[CACHE   7]         : " << cacheTime << " (s) , " << (cacheBytes>>20) << " (MB)" << std::endl;
		}
	}
	if( serverSocketStreams.size()==1 ) delete client;
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
Client< Real , Dim , BType , Degree >::Client( void )
	: _serverSocket( _INVALID_SOCKET_ ) , _tree( MEMORY_ALLOCATOR_BLOCK_SIZE ) , _density(NULL) , _normalInfo(NULL) , _paddedNormalInfo(NULL) , _iInfo(NULL) , _index(-1)
{
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
Client< Real , Dim , BType , Degree >::~Client( void )
{
	if( _density ) delete _density;
	if( _normalInfo ) delete _normalInfo;
	if( _paddedNormalInfo ) delete _paddedNormalInfo;
	if( _iInfo ) delete _iInfo;
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
std::pair< unsigned int , unsigned int > Client< Real , Dim , BType , Degree >::_PaddedRange( std::pair< unsigned int , unsigned int > range , unsigned int depth , unsigned int padSize )
{
	std::pair< unsigned int , unsigned int > paddedRange;
	if( range.first<=padSize ) paddedRange.first = 0;
	else paddedRange.first = range.first - padSize;
	if( range.second+padSize>(1u<<depth) ) paddedRange.second = (1<<depth);
	else paddedRange.second = range.second + padSize;
	return paddedRange;
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
std::function< int ( Point< Real , Dim > ) > Client< Real , Dim , BType , Degree >::PointDepthFunctor( Real begin , Real end , unsigned int padSize , unsigned int minDepth , unsigned int maxDepth )
{
	// Using a padding size of two should do the trick. And it does, but only if we don't use adaptive
	const double Log2 = log(2.);
	return [begin,end,padSize,minDepth,maxDepth,Log2]( Point< Real , Dim > p )
	{
		// For interior points, add to the full depth
		if( p[Dim-1]>=begin && p[Dim-1]<=end ) return (int)maxDepth;
		else if( p[Dim-1]<begin )
		{
			// Solve for the largest d s.t.:
			//		p[Dim-1] >= b-padSize/(1<<d)
			//		b-p[Dim-1] <= padSize/(1<<d)
			//		(b-p[Dim-1])/padSize <= 1/(1<<d)
			//		padSize/(b-p[Dim-1]) >= (1<<d)
			//		log_2( padSize/(b-p[Dim-1]) ) >= d
			// =>	d = floor( log_2( padSize/(b-p[Dim-1]) ) )
			return std::max< int >( std::min< int >( (int)floor( log( padSize/( begin - p[Dim-1] ) ) / Log2 ) , (int)maxDepth ) , (int)minDepth );
		}
		else if( p[Dim-1]>end )
		{
			// Solve for the largest d s.t.:
			//		p[Dim-1] <= e+padSize/(1<<d)
			//		p[Dim-1]-e <= padSize/(1<<d)
			//		(p[Dim-1]-d)/padSize <= 1/(1<<d)
			//		padSize/(p[Dim-1]-e) >= (1<<d)
			//		log_2( padSize/(p[Dim-1]-e) ) >= d
			// =>	d = floor( log_2( padSize/(p[Dim-1]-e) ) )
			return std::max< int >( std::min< int >( (int)floor( log( padSize/( p[Dim-1] - end ) ) / Log2 ) , (int)maxDepth ) , (int)minDepth );
		}
		else return -1;
	};
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
PhaseInfo Client< Real , Dim , BType , Degree >::_phase1( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , Profiler &profiler )
{
	PhaseInfo phaseInfo;
	ProjectiveData< Point< Real , 2 > , Real > pointDepthAndWeight;
	std::pair< size_t , size_t > nodeCounts;

	{
		Timer timer;
		phaseInfo.readBytes += _receive1( clientReconInfo , profiler );
		phaseInfo.readTime += timer.wallTime();
	}
	if( clientReconInfo.verbose>1 ) std::cout << "Range[" << _index << "]: [ " << _range.first << " , " << _range.second << " ) +/- " << clientReconInfo.padSize << std::endl;

	{
		Timer timer;
		_process1( clientReconInfo , pointDepthAndWeight , nodeCounts , profiler );
		phaseInfo.processTime += timer.wallTime();
	}

	ProjectiveData< Real , Real > pointWeight;
	pointWeight.data = pointDepthAndWeight.data[1];
	pointWeight.weight = pointDepthAndWeight.weight;
	{
		Timer timer;
		phaseInfo.writeBytes += _send1( clientReconInfo , pointWeight , nodeCounts , profiler );
		phaseInfo.writeTime += timer.wallTime();
	}
	return phaseInfo;
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
size_t Client< Real , Dim , BType , Degree >::_receive1( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , Profiler &profiler )
{
	ClientServerStream< true > serverStream( _serverSocket , _index , clientReconInfo );
	serverStream.ioBytes = 0;

	if( !serverStream.read( _modelToUnitCube ) ) MK_THROW( "Failed to read model-to-unit-cube transform" );
	serverStream.read( _range );
	profiler.update();

	return serverStream.ioBytes;
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
size_t Client< Real , Dim , BType , Degree >::_send1( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , ProjectiveData< Real , Real > pointWeight , std::pair< size_t , size_t > nodeCounts , Profiler &profiler )
{
	ClientServerStream< false > serverStream( _serverSocket , _index , clientReconInfo );
	serverStream.ioBytes = 0;
	serverStream.write( pointWeight );
	serverStream.write( nodeCounts );
	profiler.update();
	return serverStream.ioBytes;
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
void Client< Real , Dim , BType , Degree >::_process1( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , ProjectiveData< Point< Real , 2 > , Real > &pointDepthAndWeight , std::pair< size_t , size_t > &nodeCounts , Profiler &profiler )
{
	std::pair< unsigned int , unsigned int > paddedRange = _PaddedRange( _range  , clientReconInfo.sharedDepth , clientReconInfo.padSize );
	AuxDataFactory auxDataFactory( clientReconInfo.auxProperties );
	InputSampleFactory inputSampleFactory( VertexFactory::PositionFactory< Real , Dim >() , InputSampleDataFactory( VertexFactory::NormalFactory< Real , Dim >() , auxDataFactory ) );
	InputSampleDataFactory inputSampleDataFactory( VertexFactory::NormalFactory< Real , Dim >() , auxDataFactory );

	size_t pointCount = 0 , paddedPointCount = 0;
	ProjectiveData< Point< Real , 2 > , Real > paddedPointDepthAndWeight;
	paddedPointDepthAndWeight.data = Point< Real , 2 >();
	paddedPointDepthAndWeight.weight = 0;

	unsigned int beginIndex=_range.first , endIndex = _range.second;
	unsigned int beginPaddedIndex = paddedRange.first , endPaddedIndex = paddedRange.second;

	pointDepthAndWeight.data = Point< Real , 2 >();
	pointDepthAndWeight.weight = 0;

	FEMTreeInitializer< Dim , Real >::Initialize( _tree.spaceRoot() , clientReconInfo.baseDepth , []( int , int[] ){ return true; } , _tree.nodeAllocators[0] , _tree.initializer() );


#ifdef ADAPTIVE_PADDING
	std::function< int ( Point< Real , Dim > ) > pointDepthFunctor = PointDepthFunctor( (Real)_range.first/(1<<clientReconInfo.sharedDepth) , (Real)_range.second/(1<<clientReconInfo.sharedDepth) , clientReconInfo.padSize , clientReconInfo.kernelDepth , clientReconInfo.reconstructionDepth );
#endif // ADAPTIVE_PADDING
	// Read in the samples (and color data)
	{
		Timer timer;
		auto ProcessData = [&]( const Point< Real , Dim > &p , InputSampleDataType &d )
			{
				Real l = (Real)Length( d.template get<0>() );
				if( !l || !std::isfinite( l ) ) return (Real)-1.;
				d.template get<0>() /= l;
				return clientReconInfo.confidence ? l : (Real)1.;
			};

		std::vector< InputDataStream< typename InputSampleFactory::VertexType > * > pointStreams( endPaddedIndex - beginPaddedIndex , NULL );
		auto PointStreamFunctor = [&]( unsigned int idx )
		{
			std::string fileName = PointPartition::FileName( clientReconInfo.header , idx , 1<<clientReconInfo.sharedDepth , clientReconInfo.filesPerDir );

			fileName = PointPartition::FileDir( clientReconInfo.inDir , fileName );
			pointStreams[idx-beginPaddedIndex] = new PointPartition::BufferedBinaryInputDataStream< InputSampleFactory >( fileName.c_str() , inputSampleFactory , clientReconInfo.bufferSize );
		};
		{
			std::vector< std::thread > pointStreamThreads;
			pointStreamThreads.reserve( endPaddedIndex - beginPaddedIndex );
			for( unsigned int i=beginPaddedIndex ; i<endPaddedIndex ; i++ ) pointStreamThreads.emplace_back( PointStreamFunctor , i );
			for( unsigned int i=0 ; i<( endPaddedIndex - beginPaddedIndex ) ; i++ ) pointStreamThreads[i].join();
		}

		using MultiPointStream = MultiInputDataStream< typename InputSampleFactory::VertexType >;

		using ExternalType = std::tuple< Point< Real , Dim > , InputSampleDataType >;
		using InternalType = std::tuple< typename InputSampleFactory::VertexType >;
		auto converter = []( const InternalType &iType )
			{
				ExternalType xType;
				std::get< 0 >( xType )                   = std::get< 0 >( iType ).template get<0>();
				std::get< 1 >( xType ).template get<0>() = std::get< 0 >( iType ).template get<1>().template get<0>();
				std::get< 1 >( xType ).template get<1>() = std::get< 0 >( iType ).template get<1>().template get<1>();
				return xType;
			};

		auto ProcessInteriorPointSlabs = [&]( typename FEMTreeInitializer< Dim , Real >::StreamInitializationData &sid , unsigned int start , unsigned int end )
		{
			if( start==end ) return;
			MultiPointStream pointStream( &pointStreams[start-beginPaddedIndex] , end - start );
			typename InputSampleDataFactory::VertexType zeroData = inputSampleDataFactory();
			InputDataStreamConverter< InternalType , ExternalType > _pointStream( pointStream , converter , inputSampleFactory() );
			pointCount += FEMTreeInitializer< Dim , Real >::template Initialize< InputSampleDataType >( sid , _tree.spaceRoot() , _pointStream , zeroData , clientReconInfo.reconstructionDepth , _samples , _sampleData , _tree.nodeAllocators[0] , _tree.initializer() , ProcessData );
			profiler.update();
		};
		auto ProcessPadPointSlabs = [&]( typename FEMTreeInitializer< Dim , Real >::StreamInitializationData &sid , unsigned int start , unsigned int end )
		{
			if( start==end ) return;
			MultiPointStream pointStream( &pointStreams[start-beginPaddedIndex] , end - start );
			typename InputSampleDataFactory::VertexType zeroData = inputSampleDataFactory();
			InputDataStreamConverter< InternalType , ExternalType > _pointStream( pointStream , converter , inputSampleFactory() );
#ifdef ADAPTIVE_PADDING
			paddedPointCount += FEMTreeInitializer< Dim , Real >::template Initialize< InputSampleDataType >( sid , _tree.spaceRoot() , _pointStream , zeroData , clientReconInfo.reconstructionDepth , pointDepthFunctor , _paddedSamples , _paddedSampleData , _tree.nodeAllocators[0] , _tree.initializer() , ProcessData );
#else // !ADAPTIVE_PADDING
			paddedPointCount += FEMTreeInitializer< Dim , Real >::template Initialize< InputSampleDataType >( sid , _tree.spaceRoot() , _pointStream , zeroData , clientReconInfo.reconstructionDepth , _paddedSamples , _paddedSampleData , _tree.nodeAllocators[0] , _tree.initializer() , ProcessData );
#endif // ADAPTIVE_PADDING
			profiler.update();
		};

		auto ProcessPointSlab = [&]( typename FEMTreeInitializer< Dim , Real >::StreamInitializationData &sid , unsigned int idx )
		{
			InputDataStream< typename InputSampleFactory::VertexType > &pointStream = *pointStreams[idx-beginPaddedIndex];
				
			if( idx>=beginIndex && idx<endIndex )
			{
				typename InputSampleDataFactory::VertexType zeroData = inputSampleDataFactory();
				InputDataStreamConverter< InternalType , ExternalType > _pointStream( pointStream , converter , inputSampleFactory() );
				pointCount += FEMTreeInitializer< Dim , Real >::template Initialize< InputSampleDataType >( sid , _tree.spaceRoot() , _pointStream , zeroData , clientReconInfo.reconstructionDepth , _samples , _sampleData , _tree.nodeAllocators[0] , _tree.initializer() , ProcessData );
			}
			else
			{
				typename InputSampleDataFactory::VertexType zeroData = inputSampleDataFactory();
				InputDataStreamConverter< InternalType , ExternalType > _pointStream( pointStream , converter , inputSampleFactory() );
#ifdef ADAPTIVE_PADDING
				paddedPointCount += FEMTreeInitializer< Dim , Real >::template Initialize< InputSampleDataType >( sid , _tree.spaceRoot() , _pointStream , zeroData , clientReconInfo.reconstructionDepth , pointDepthFunctor , _paddedSamples , _paddedSampleData , _tree.nodeAllocators[0] , _tree.initializer() , ProcessData );
#else // !ADAPTIVE_PADDING
				paddedPointCount += FEMTreeInitializer< Dim , Real >::template Initialize< InputSampleDataType >( sid , _tree.spaceRoot() , _pointStream , zeroData , clientReconInfo.reconstructionDepth , _paddedSamples , _paddedSampleData , _tree.nodeAllocators[0] , _tree.initializer() , ProcessData );
#endif // ADAPTIVE_PADDING
			}
			profiler.update();
		};

		{
			typename FEMTreeInitializer< Dim , Real >::StreamInitializationData sid;
			ProcessInteriorPointSlabs( sid , beginIndex , endIndex );
		}
		nodeCounts.first = _tree.spaceRoot().nodes();

		{
			typename FEMTreeInitializer< Dim , Real >::StreamInitializationData sid;
			ProcessPadPointSlabs( sid , beginPaddedIndex , beginIndex );
			ProcessPadPointSlabs( sid , endIndex , endPaddedIndex );
		}
		for( unsigned int i=0 ; i<pointStreams.size() ; i++ ) delete pointStreams[i];



		nodeCounts.second = _tree.spaceRoot().nodes();

		if( clientReconInfo.verbose>1 ) std::cout << "Input Points / Padded Input Points / Samples / Padded Samples: " << pointCount << " / " << paddedPointCount << " / " << _samples.size() << " / " << _paddedSamples.size() << std::endl;
		if( clientReconInfo.verbose>1 ) std::cout << "#          Read input into tree: " << timer << std::endl;
	}

	if( clientReconInfo.verbose>1 ) std::cout << "Nodes [Initialized]: " << _tree.allNodes() << std::endl;

	// Get the kernel density estimator
	{
		Timer timer;
		_density = _tree.template setDensityEstimator< 1 , Reconstructor::WeightDegree >( _samples , clientReconInfo.kernelDepth , clientReconInfo.samplesPerNode );
#ifdef ADAPTIVE_PADDING
		_tree.template updateDensityEstimator< 1 , Reconstructor::WeightDegree >( *_density , _paddedSamples , 0 , clientReconInfo.kernelDepth , pointDepthFunctor );
#else // !ADAPTIVE_PADDING
		_tree.template updateDensityEstimator< 1 , Reconstructor::WeightDegree >( *_density , _paddedSamples , 0 , clientReconInfo.kernelDepth );
#endif // ADAPTIVE_PADDING
		profiler.update();
		if( clientReconInfo.verbose>1 ) std::cout << "#            Got kernel density: " << timer << std::endl;
	}

	if( clientReconInfo.verbose>1 ) std::cout << "Nodes [Density Estimator]: " << _tree.allNodes() << std::endl;

	// Transform the Hermite samples into a vector field
	{
		Timer timer;
		_normalInfo = new SparseNodeData< Point< Real , Dim > , NormalSigs >();
		_paddedNormalInfo = new SparseNodeData< Point< Real , Dim > , NormalSigs >();

		std::function< bool ( InputSampleDataType , Point< Real , Dim >& ) > ConversionFunction = []( InputSampleDataType in , Point< Real , Dim > &out )
		{
			Point< Real , Dim > n = in.template get<0>();
			Real l = (Real)Length( n );
			// It is possible that the samples have non-zero normals but there are two co-located samples with negative normals...
			if( !l ) return false;
			out = n / l;
			return true;
		};

		{
			*_normalInfo = _tree.setInterpolatedDataField( Point< Real , Dim >() , NormalSigs() , _samples , _sampleData , _density , clientReconInfo.sharedDepth , clientReconInfo.reconstructionDepth , (Real)0 , pointDepthAndWeight , ConversionFunction );
			if( clientReconInfo.verbose>1 ) std::cout << "Nodes [Interior Data Field " << _normalInfo->size() << " / " << _normalInfo->reserved() << "]: " << _tree.allNodes() << std::endl;
#ifdef ADAPTIVE_PADDING
			*_paddedNormalInfo = _tree.setInterpolatedDataField( Point< Real , Dim >() , NormalSigs() , _paddedSamples , _paddedSampleData , _density , clientReconInfo.sharedDepth , clientReconInfo.reconstructionDepth , pointDepthFunctor , (Real)0 , paddedPointDepthAndWeight , ConversionFunction );
#else // !ADAPTIVE_PADDING
			*_paddedNormalInfo = _tree.setInterpolatedDataField( Point< Real , Dim >() , NormalSigs() , _paddedSamples , _paddedSampleData , _density , clientReconInfo.sharedDepth , clientReconInfo.reconstructionDepth , (Real)0 , paddedPointDepthAndWeight , ConversionFunction );
#endif // ADAPTIVE_PADDING
		}

		ThreadPool::ParallelFor( 0 , _normalInfo->size() , [&]( unsigned int , size_t i ){ (*_normalInfo)[i] *= (Real)-1.; } );
		ThreadPool::ParallelFor( 0 , _paddedNormalInfo->size() , [&]( unsigned int , size_t i ){ (*_paddedNormalInfo)[i] *= (Real)-1.; } );
		profiler.update();
		if( clientReconInfo.verbose>1 )
		{
			std::cout << "#              Got normal field: " << timer << std::endl;
			std::cout << "Point Depth / Point Weight / Padded Point Depth / Padded Point Weight / Estimated Measure /  Estimated Padded Measure: " << pointDepthAndWeight.value()[0] << " / " << pointDepthAndWeight.value()[1] << " / " << paddedPointDepthAndWeight.value()[0] << " / " << paddedPointDepthAndWeight.value()[1] << " / " << pointCount* pointDepthAndWeight.value()[1] << " / " << paddedPointCount* paddedPointDepthAndWeight.value()[1] << std::endl;
		}
	}
	if( clientReconInfo.verbose>1 ) std::cout << "Nodes [Padded Data Field " << _paddedNormalInfo->size() << " / " << _paddedNormalInfo->reserved() << "]: " << _tree.allNodes() << std::endl;

	_tree.resetNodeIndices( 0 , std::make_tuple( _density , _normalInfo , _paddedNormalInfo ) );

	if( clientReconInfo.verbose>1 ) std::cout << "Memory Usage: " << float( MemoryInfo::Usage() )/(1<<20) << " MB" << std::endl;
	profiler.update();
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
PhaseInfo Client< Real , Dim , BType , Degree >::_phase3( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , _State3 &state3 , Profiler &profiler )
{
	PhaseInfo phaseInfo;
	ProjectiveData< Real , Real > cumulativePointWeight;

	{
		Timer timer;
		phaseInfo.readBytes += _receive3( clientReconInfo , cumulativePointWeight , profiler );
		phaseInfo.readTime += timer.wallTime();
	}

	{
		Timer timer;
		_process3( clientReconInfo , cumulativePointWeight , state3 , profiler );
		phaseInfo.processTime += timer.wallTime();
	}

	{
		Timer timer;
		phaseInfo.writeBytes += _send3( clientReconInfo , state3 , profiler );
		phaseInfo.writeTime += timer.wallTime();
	}

	return phaseInfo;
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
size_t Client< Real , Dim , BType , Degree >::_receive3( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , ProjectiveData< Real , Real > &cumulativePointWeight , Profiler &profiler )
{
	ClientServerStream< true > serverStream( _serverSocket , _index , clientReconInfo );
	serverStream.ioBytes = 0;
	if( !serverStream.read( cumulativePointWeight ) ) MK_THROW( "Could not read cumulative point weight" );
	profiler.update();
	return serverStream.ioBytes;
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
size_t Client< Real , Dim , BType , Degree >::_send3( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , const _State3 &state3 , Profiler &profiler )
{
	size_t ioBytes = 0;
	{
		ClientServerStream< false > serverStream( _serverSocket , _index , clientReconInfo , ClientReconstructionInfo< Real , Dim >::BACK );
		serverStream.ioBytes = 0;

		serverStream.write( state3.subNodeCount );
		TreeAddressesToIndices( state3.subNodes , state3.subNodeCount );
		serverStream.write( state3.subNodes , state3.subNodeCount );
		ioBytes += serverStream.ioBytes;
		profiler.update();
	}
	{
		AuxDataFactory auxDataFactory( clientReconInfo.auxProperties );
		bool needAuxData = clientReconInfo.dataX>0 && auxDataFactory.bufferSize();

		ClientServerStream< false > serverStream( _serverSocket , _index , clientReconInfo , ClientReconstructionInfo< Real , Dim >::FRONT );
		serverStream.ioBytes = 0;

		state3.constraints.write( serverStream );
		state3.iInfo.write( serverStream );

		if( needAuxData )
		{
			if constexpr( !AuxDataFactory::IsStaticallyAllocated() )
			{
				ProjectiveAuxDataTypeSerializer< Real > serializer( clientReconInfo.auxProperties );
				state3.auxDataField.write( serverStream , serializer );
			}
			else state3.auxDataField.write( serverStream );
		}
		ioBytes += serverStream.ioBytes;
		profiler.update();
	}
	return ioBytes;
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
void Client< Real , Dim , BType , Degree >::_process3( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , ProjectiveData< Real , Real > cumulativePointWeight , _State3 &state3 , Profiler &profiler )
{
	std::pair< unsigned int , unsigned int > paddedRange = _PaddedRange( _range  , clientReconInfo.sharedDepth , clientReconInfo.padSize );
	AuxDataFactory auxDataFactory( clientReconInfo.auxProperties );

	InputSampleFactory inputSampleFactory( VertexFactory::PositionFactory< Real , Dim >() , InputSampleDataFactory( VertexFactory::NormalFactory< Real , Dim >() , auxDataFactory ) );
	InputSampleDataFactory inputSampleDataFactory( VertexFactory::NormalFactory< Real , Dim >() , auxDataFactory );

	bool needAuxData = clientReconInfo.dataX>0 && auxDataFactory.bufferSize();

	Real targetValue = clientReconInfo.targetValue;

	unsigned int beginIndex = _range.first , endIndex = _range.second;
	unsigned int beginPaddedIndex = paddedRange.first , endPaddedIndex = paddedRange.second;

	ApproximatePointInterpolationInfo *iInfo = NULL;
	ApproximatePointInterpolationInfo *paddedIInfo = NULL;

	// Add the interpolation constraints
	{
		Timer timer;
		iInfo       = FEMTree< Dim , Real >::template InitializeApproximatePointInterpolationInfo< Real , 0 > ( _tree ,       _samples , Reconstructor::Poisson::ConstraintDual< Dim , Real >( targetValue , clientReconInfo.pointWeight * cumulativePointWeight.value() ) , Reconstructor::Poisson::SystemDual< Dim , Real >( clientReconInfo.pointWeight * cumulativePointWeight.value() ) , true , clientReconInfo.reconstructionDepth , 1 );
		paddedIInfo = FEMTree< Dim , Real >::template InitializeApproximatePointInterpolationInfo< Real , 0 > ( _tree , _paddedSamples , Reconstructor::Poisson::ConstraintDual< Dim , Real >( targetValue , clientReconInfo.pointWeight * cumulativePointWeight.value() ) , Reconstructor::Poisson::SystemDual< Dim , Real >( clientReconInfo.pointWeight * cumulativePointWeight.value() ) , true , clientReconInfo.reconstructionDepth , 1 );
		profiler.update();

		if( clientReconInfo.verbose>1 ) std::cout << "# Set interpolation constraints: " << timer << std::endl;
	}

	// Trim the tree and prepare for multigrid
	{
		Timer timer;

		constexpr int MaxDegree = Reconstructor::Poisson::NormalDegree > Degrees::Max() ? Reconstructor::Poisson::NormalDegree : Degrees::Max();
		typename FEMTree< Dim , Real >::template HasNormalDataFunctor< NormalSigs > hasNormalDataFunctor( *_normalInfo );
		typename FEMTree< Dim , Real >::template HasNormalDataFunctor< NormalSigs > hasPaddedNormalDataFunctor( *_paddedNormalInfo );
		auto hasDataFunctor = [&]( const FEMTreeNode *node ){ return hasNormalDataFunctor( node ) || hasPaddedNormalDataFunctor( node ); };

		const int StartOffset = BSplineSupportSizes< MaxDegree >::SupportStart;
		const int   EndOffset = BSplineSupportSizes< MaxDegree >::SupportEnd+1;
		auto addNodeFunctor = [&]( int d , const int off[Dim] )
		{
			if( d<0 ) return true;
			else if( d>(int)clientReconInfo.baseDepth ) return false;
			else
			{
				int start = ( off[Dim-1] + StartOffset )<<(clientReconInfo.sharedDepth-d);
				int end   = ( off[Dim-1] +   EndOffset )<<(clientReconInfo.sharedDepth-d);
				return start<(int)endPaddedIndex && end>(int)beginPaddedIndex;
			}
		};
		_tree.template finalizeForMultigrid< MaxDegree , Degrees::Max() >( clientReconInfo.baseDepth , addNodeFunctor , hasDataFunctor , std::make_tuple( iInfo , paddedIInfo ) , std::make_tuple( _normalInfo , _paddedNormalInfo , _density ) );
		profiler.update();
		if( clientReconInfo.verbose>1 )
		{
			std::cout << "All Nodes / Active Nodes / Ghost Nodes: " << _tree.allNodes() << " / " << _tree.activeNodes() << " / " << _tree.ghostNodes() << std::endl;
			std::cout << "#                Finalized tree: " << timer << std::endl;
		}
	}

	// Compute the FEM constraints for the points interior to the slab
	{
		_constraints = _tree.initDenseNodeData( Sigs() );

		// Add Poisson constraints
		{
			Timer timer;
			typename FEMIntegrator::template Constraint< Sigs , IsotropicUIntPack< Dim , 1 > , NormalSigs , IsotropicUIntPack< Dim , 0 > , Dim > F;
			unsigned int derivatives2[Dim];
			for( int d=0 ; d<Dim ; d++ ) derivatives2[d] = 0;
			typedef IsotropicUIntPack< Dim , 1 > Derivatives1;
			typedef IsotropicUIntPack< Dim , 0 > Derivatives2;
			for( int d=0 ; d<Dim ; d++ )
			{
				unsigned int derivatives1[Dim];
				for( int dd=0 ; dd<Dim ; dd++ ) derivatives1[dd] = dd==d ? 1 : 0;
				F.weights[d][ TensorDerivatives< Derivatives1 >::Index( derivatives1 ) ][ TensorDerivatives< Derivatives2 >::Index( derivatives2 ) ] = 1;
			}
			_tree.addFEMConstraints( F , *_normalInfo , _constraints , clientReconInfo.reconstructionDepth );
			if( clientReconInfo.verbose>1 ) std::cout << "#  Set interior FEM constraints: " << timer << std::endl;
		}
		profiler.update();

		// Free up the normal info
		delete _normalInfo , _normalInfo = NULL;

		{
			Timer timer;
			_tree.addInterpolationConstraints( _constraints , clientReconInfo.reconstructionDepth , std::make_tuple( iInfo ) );
			profiler.update();
			if( clientReconInfo.verbose>1 ) std::cout << "#Set interior point constraints: " << timer << std::endl;
		}
	}
	if( needAuxData ) _auxDataField = _tree.template setExtrapolatedDataField< DataSig , false , Reconstructor::WeightDegree , AuxData >( auxDataFactory() , _samples.size() , [&]( size_t i ) -> const typename FEMTree< Dim , Real >::PointSample & { return _samples[i]; } , [&]( size_t i ) -> const AuxData & { return _sampleData[i].template get<1>(); } , (DensityEstimator*)NULL );

	// Get the shared tree, constraints, interpolation info, and data-field
	{
		auto keepNodeFunctor = [&]( const FEMTreeNode *node )
		{
			int d , off[Dim];
			_tree.depthAndOffset( node , d , off );

			return d<=(int)clientReconInfo.sharedDepth;
		};
		state3.subNodes = _tree.tree().serializeSubTree( keepNodeFunctor , state3.subNodeCount );
		state3.constraints = _tree.trimToDepth( _constraints , clientReconInfo.sharedDepth );
		state3.iInfo = _tree.trimToDepth( *iInfo , clientReconInfo.sharedDepth );
		if( needAuxData ) state3.auxDataField = _tree.trimToDepth( _auxDataField , clientReconInfo.sharedDepth );
	}

	if( needAuxData )
	{
		_tree.template updateExtrapolatedDataField< DataSig , false >( auxDataFactory() , _auxDataField , _paddedSamples.size() , [&]( size_t i ) -> const typename FEMTree< Dim , Real >::PointSample & { return _paddedSamples[i]; } , [&]( size_t i ) -> const AuxData & { return _paddedSampleData[i].template get<1>(); } , (DensityEstimator*)NULL );
		auto nodeFunctor = [&]( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > *n )
		{
			ProjectiveData< AuxData , Real >* clr = _auxDataField( n );
			if( clr ) (*clr) *= (Real)pow( clientReconInfo.dataX , _tree.depth( n ) );
		};
		_tree.tree().processNodes( nodeFunctor );
	}

	// Add the FEM constraints for the points in the padding region
	{
		Timer timer;
		// Add Poisson constraints
		{
			typename FEMIntegrator::template Constraint< Sigs , IsotropicUIntPack< Dim , 1 > , NormalSigs , IsotropicUIntPack< Dim , 0 > , Dim > F;
			unsigned int derivatives2[Dim];
			for( int d=0 ; d<Dim ; d++ ) derivatives2[d] = 0;
			typedef IsotropicUIntPack< Dim , 1 > Derivatives1;
			typedef IsotropicUIntPack< Dim , 0 > Derivatives2;
			for( int d=0 ; d<Dim ; d++ )
			{
				unsigned int derivatives1[Dim];
				for( int dd=0 ; dd<Dim ; dd++ ) derivatives1[dd] = dd==d ? 1 : 0;
				F.weights[d][ TensorDerivatives< Derivatives1 >::Index( derivatives1 ) ][ TensorDerivatives< Derivatives2 >::Index( derivatives2 ) ] = 1;
			}
			_tree.addFEMConstraints( F , *_paddedNormalInfo , _constraints , clientReconInfo.reconstructionDepth );
		}
		if( clientReconInfo.verbose>1 ) std::cout << "#    Set padded FEM constraints: " << timer << std::endl;

		profiler.update();

		// Free up the normal info
		delete _paddedNormalInfo , _paddedNormalInfo = NULL;

		{
			Timer timer;
			_tree.addInterpolationConstraints( _constraints , clientReconInfo.reconstructionDepth , std::make_tuple( paddedIInfo ) );
			profiler.update();
			if( clientReconInfo.verbose>1 ) std::cout << "#  Set padded point constraints: " << timer << std::endl;
		}
	}

	// Merge the interpolation information
	{
		Timer timer;
		using InterpolationData = typename ApproximatePointInterpolationInfo::Data;
		auto  preMergeFunctor = []( InterpolationData data ){ data.position *= data.weight ; return data; };
		auto postMergeFunctor = []( InterpolationData data ){ data.position /= data.weight ; return data; };

		_iInfo = new ApproximatePointInterpolationInfo( Reconstructor::Poisson::ConstraintDual< Dim , Real >( targetValue , clientReconInfo.pointWeight * cumulativePointWeight.value() ) , Reconstructor::Poisson::SystemDual< Dim , Real >( clientReconInfo.pointWeight * cumulativePointWeight.value() ) , true );
		_iInfo->iData.reserve( _tree.nodesSize() );

		_iInfo->iData.merge( iInfo->iData , preMergeFunctor );
		_iInfo->iData.merge( paddedIInfo->iData , preMergeFunctor );

		for( unsigned int i=0 ; i<_iInfo->iData.size() ; i++ ) _iInfo->iData[i] = postMergeFunctor( _iInfo->iData[i] );

		if( clientReconInfo.verbose>1 ) std::cout << "#         Merged interpolation: " << timer << std::endl;
	}
	if( clientReconInfo.verbose>1 ) std::cout << "Memory Usage: " << float( MemoryInfo::Usage() )/(1<<20) << " MB" << std::endl;
	profiler.update();
	delete iInfo;
	delete paddedIInfo;
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
PhaseInfo Client< Real , Dim , BType , Degree >::_phase5( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , Profiler &profiler )
{
	PhaseInfo phaseInfo;
	std::pair< double , double > isoInfo;

	typename Client< Real , Dim , BType , Degree >::_State5 state5;
	{
		Timer timer;
		phaseInfo.readBytes += _receive5( clientReconInfo , state5 , profiler );
		phaseInfo.readTime += timer.wallTime();
	}

	{
		Timer timer;
		isoInfo = _process5( clientReconInfo , state5 , profiler );
		phaseInfo.processTime += timer.wallTime();
	}

	{
		Timer timer;
		_serverSocket.write( isoInfo );
		phaseInfo.writeBytes += _send5( clientReconInfo , state5 , profiler );
		phaseInfo.writeTime += timer.wallTime();
	}
	return phaseInfo;
}


template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
size_t Client< Real , Dim , BType , Degree >::_receive5( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , _State5 &state5 , Profiler &profiler )
{
	AuxDataFactory auxDataFactory( clientReconInfo.auxProperties );
	bool needAuxData = clientReconInfo.dataX>0 && auxDataFactory.bufferSize();

	ClientServerStream< true > serverStream( _serverSocket , _index , clientReconInfo );
	serverStream.ioBytes = 0;
	size_t subNodeCount;
	serverStream.read( subNodeCount );
	state5.subNodes = NewPointer< FEMTreeNode >( subNodeCount );
	serverStream.read( state5.subNodes , subNodeCount );
	TreeIndicesToAddresses( state5.subNodes , subNodeCount );
	state5.solution.read( serverStream );

	if( needAuxData )
	{
		using Data = typename _State5::Data;
		Data defaultValue;

		if constexpr( !AuxDataFactory::IsStaticallyAllocated() )
		{
			ProjectiveAuxDataTypeSerializer< Real > serializer( clientReconInfo.auxProperties );
			state5.auxDataField = new SparseNodeData< ProjectiveData< AuxData , Real > , IsotropicUIntPack< Dim , DataSig > >( serverStream , serializer );
		}
		else state5.auxDataField = new SparseNodeData< ProjectiveData< AuxData , Real > , IsotropicUIntPack< Dim , DataSig > >( serverStream );
	}
	profiler.update();

	return serverStream.ioBytes;
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
size_t Client< Real , Dim , BType , Degree >::_send5( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , const _State5 &state5 , Profiler &profiler )
{
	size_t ioBytes = 0;

	auto Send = [&]( const typename Client::_State5::BoundaryInfo &bInfo , typename ClientReconstructionInfo< Real , Dim >::ShareType st )
	{
		ClientServerStream< false > serverStream( _serverSocket , _index , clientReconInfo , st );
		serverStream.ioBytes=0;

		XForm< Real , Dim > sliceModelToUnitCube;
		{
			for( unsigned int i=0 ; i<Dim-1 ; i++ )
			{
				for( unsigned int j=0 ; j<Dim-1 ; j++ ) sliceModelToUnitCube(i,j) = _modelToUnitCube(i,j);
				sliceModelToUnitCube(Dim-1,i) = _modelToUnitCube(Dim,i);
			}
			sliceModelToUnitCube(Dim-1,Dim-1) = 1;
		}

		if( bInfo.tree )
		{
			bInfo.tree->write( serverStream , true );
			serverStream.write( sliceModelToUnitCube );
			bInfo.solution.write( serverStream );
			bInfo.dSolution.write( serverStream );
		}
		profiler.update();
		return serverStream.ioBytes;
	};
	if( _range.first!=0 )                                 ioBytes += Send( state5.boundaryInfo.first  , ClientReconstructionInfo< Real , Dim >::BACK  );
	if( _range.second!=(1<<clientReconInfo.sharedDepth) ) ioBytes += Send( state5.boundaryInfo.second , ClientReconstructionInfo< Real , Dim >::FRONT );

	return ioBytes;
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
std::pair< double , double > Client< Real , Dim , BType , Degree >::_process5( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , _State5 &state5 , Profiler &profiler )
{
	AuxDataFactory auxDataFactory( clientReconInfo.auxProperties );
	bool needAuxData = clientReconInfo.dataX>0 && auxDataFactory.bufferSize();

	// Refine the client's tree so that it contains the server's subtree
	{
		Timer timer;
		// Refine client nodes as needed
		std::function< void ( const FEMTreeNode * , FEMTreeNode * ) > ProcessIndices = [&]( const FEMTreeNode *serverNode , FEMTreeNode *clientNode )
		{
			if( serverNode->children )
			{
				if( !clientNode->children ) clientNode->template initChildren< false >( _tree.nodeAllocators[0] , _tree.initializer() );
				for( int c=0 ; c<(1<<Dim) ; c++ ) ProcessIndices( serverNode->children+c , clientNode->children+c );
			}
		};
		ProcessIndices( &state5.subNodes[0] , const_cast< FEMTreeNode * >( &_tree.tree() ) );
		// Re-finalize and re-index
		auto addNodeFunctor = [&]( int d , const int off[Dim] ){ return d<=(int)clientReconInfo.baseDepth; };
		auto hasDataFunctor = []( const FEMTreeNode * ){ return true; };

		constexpr int MaxDegree = Reconstructor::Poisson::NormalDegree > Degrees::Max() ? Reconstructor::Poisson::NormalDegree : Degrees::Max();
		SparseNodeData< ProjectiveData< AuxData , Real > , IsotropicUIntPack< Dim , DataSig > > *auxDataField = needAuxData ? &_auxDataField : NULL;
		DenseNodeData< Real , Sigs > *constraints = &_constraints;

		// [WARNING] This assumes that nothing needs to be done with the Dirichlet flags
		_tree.setSortedTreeNodes( std::make_tuple( _iInfo ) , std::make_tuple( _density , constraints , auxDataField ) );
		profiler.update();

		if( clientReconInfo.verbose>1 ) std::cout << "#  Updated client tree: " << timer << std::endl;
	}

	{
		Timer timer;
		_solution = _tree.initDenseNodeData( Sigs() );
		std::vector< node_index_type > clientToServer( _tree.nodesSize() , -1 );
		{
			std::function< void ( const FEMTreeNode * , const FEMTreeNode * ) > ProcessIndices = [&]( const FEMTreeNode *serverNode , const FEMTreeNode *clientNode )
			{
				if( clientNode->nodeData.nodeIndex!=-1 )
				{
					if( clientNode->nodeData.nodeIndex>=(node_index_type)clientToServer.size() ) MK_THROW( "More client nodes than server nodes" );
					clientToServer[ clientNode->nodeData.nodeIndex ] = serverNode->nodeData.nodeIndex;
				}
				if( _tree.depth( clientNode )<(int)clientReconInfo.sharedDepth && clientNode->children )
				{
					if( serverNode->children ) for( int c=0 ; c<(1<<Dim) ; c++ ) ProcessIndices( serverNode->children+c , clientNode->children+c );
				}
			};
			ProcessIndices( &state5.subNodes[0] , &_tree.tree() );
		}
		profiler.update();
		if( clientReconInfo.verbose>1 ) std::cout << "# Set client-to-server: " << timer << std::endl;

		if( needAuxData )
		{
			Timer timer;
			using Data = ProjectiveData< AuxData , Real >;
			Data defaultValue = auxDataFactory();

			// Clearing the low freqeuncy
			auto nodeFunctor = [&]( const FEMTreeNode *n )
			{
				if( n->nodeData.nodeIndex!=-1 )
				{
					node_index_type idx = state5.auxDataField->index( clientToServer[ n->nodeData.nodeIndex ] );
					if( idx!=-1 ) _auxDataField[n] = (*state5.auxDataField)[idx];
				}
				return _tree.depth( n )<(int)clientReconInfo.sharedDepth;
			};
			_tree.tree().processNodes( nodeFunctor );
			profiler.update();
			if( clientReconInfo.verbose>1 ) std::cout << "#      Copied aux data: " << profiler << std::endl;
		}

		// (Over-)write coarse solution
		for( unsigned int i=0 ; i<_solution.size() ; i++ ) if( clientToServer[i]!=-1 ) _solution[i] = state5.solution[ clientToServer[i] ];
		profiler.update();
		if( clientReconInfo.verbose>1 ) std::cout << "#      Copied solution: " << timer << std::endl;
	}

	// Solve the linear system
	{
		Timer timer;
		typename FEMTree< Dim , Real >::SolverInfo sInfo;
		sInfo.cgDepth = 0 , sInfo.cascadic = true , sInfo.vCycles = 1 , sInfo.iters = clientReconInfo.iters , sInfo.cgAccuracy = 0 , sInfo.verbose = clientReconInfo.verbose>1 , sInfo.showResidual = clientReconInfo.verbose>2 , sInfo.showGlobalResidual = SHOW_GLOBAL_RESIDUAL_NONE , sInfo.sliceBlockSize = 1;
		sInfo.baseVCycles = 0 , sInfo.clearSolution = false;
		typename FEMIntegrator::template System< Sigs , IsotropicUIntPack< Dim , 1 > > F( { 0. , 1. } );
		_tree.solveSystem( Sigs() , F , _constraints , _solution , []( Real v , Real w ){ return v*w; } , clientReconInfo.sharedDepth+1 , clientReconInfo.solveDepth , sInfo , std::make_tuple( _iInfo ) );

		profiler.update();
		if( _iInfo ) delete _iInfo , _iInfo = NULL;

		if( clientReconInfo.verbose>1 ) std::cout << "# Linear system solved: " << timer << std::endl;
	}

	std::pair< double , double > isoInfo(0.,0.);
	// Compute the average iso-value
	{
		Timer timer;
		typename FEMTree< Dim , Real >::template MultiThreadedEvaluator< Sigs , 0 > evaluator( &_tree , _solution );
		std::vector< double > valueSums( ThreadPool::NumThreads() , 0 ) , weightSums( ThreadPool::NumThreads() , 0 );
		ThreadPool::ParallelFor( 0 , _samples.size() , [&]( unsigned int thread , size_t j )
			{
				ProjectiveData< Point< Real , Dim > , Real > &sample = _samples[j].sample;
				Real w = sample.weight;
				if( w>0 ) weightSums[thread] += w , valueSums[thread] += evaluator.values( sample.data / sample.weight , thread , _samples[j].node )[0] * w;
			}
		);
		for( size_t t=0 ; t<valueSums.size() ; t++ ) isoInfo.first += valueSums[t] , isoInfo.second += weightSums[t];
		profiler.update();
		if( clientReconInfo.verbose>1 ) std::cout << "#        Got iso-value: " << timer << std::endl;
	}

	// Set the boundary information
	if constexpr( Dim==3 )
	{
		Timer timer;
		using SliceSigs = typename Sigs::Transpose::Rest::Transpose;
		static const unsigned int CrossSig = Sigs::Transpose::First;

		auto SetBoundary = [&]( unsigned int index , typename _State5::BoundaryInfo &boundaryInfo )
		{
			bool needsPadding = !clientReconInfo.linearFit && FEMSignature< CrossSig >::Degree==1;
			if( needsPadding ) boundaryInfo.tree = FEMTree< Dim-1 , Real >::template Slice< FEMSignature< CrossSig >::Degree , 1 >( _tree , clientReconInfo.sharedDepth , index , true , MEMORY_ALLOCATOR_BLOCK_SIZE );
			else               boundaryInfo.tree = FEMTree< Dim-1 , Real >::template Slice< FEMSignature< CrossSig >::Degree , 0 >( _tree , clientReconInfo.sharedDepth , index , true , MEMORY_ALLOCATOR_BLOCK_SIZE );
			boundaryInfo.solution = boundaryInfo.tree->initDenseNodeData( SliceSigs() );
			boundaryInfo.dSolution = boundaryInfo.tree->initDenseNodeData( SliceSigs() );
			boundaryInfo.tree->template slice< 0 , CrossSig >( _tree , 0 , _solution , boundaryInfo.solution , clientReconInfo.sharedDepth , index );
			if( needsPadding ) boundaryInfo.tree->template slice< 1 , CrossSig >( _tree , 1 , _solution , boundaryInfo.dSolution , clientReconInfo.sharedDepth , index );
			else               boundaryInfo.tree->template slice< 0 , CrossSig >( _tree , 1 , _solution , boundaryInfo.dSolution , clientReconInfo.sharedDepth , index );
		};
		if( _range.first !=0                                ) SetBoundary( _range.first  , state5.boundaryInfo.first  );
		if( _range.second!=(1<<clientReconInfo.sharedDepth) ) SetBoundary( _range.second , state5.boundaryInfo.second );
		profiler.update();
		if( clientReconInfo.verbose>1 ) std::cout << "#    Set boundary info: " << timer << std::endl;
	}

	if( clientReconInfo.outputSolution )
	{
		std::string outFileName = std::string( "solution." ) + std::to_string(_index) + std::string( ".tree" );
		if( clientReconInfo.outDir.length() ) outFileName = PointPartition::FileDir( clientReconInfo.outDir , outFileName );

		FILE* fp = fopen( outFileName.c_str() , "wb" );
		if( !fp ) MK_THROW( "Failed to open file for writing: " , outFileName );
		FileStream fs(fp);
		FEMTree< Dim , Real >::WriteParameter( fs );
		DenseNodeData< Real , Sigs >::WriteSignatures( fs );
		XForm< Real , Dim+1 > voxelToUnitCube = XForm< Real , Dim+1 >::Identity();
		_tree.write( fs , false );
		fs.write( voxelToUnitCube );
		_solution.write( fs );
		fclose( fp );
	}

	return isoInfo;
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
PhaseInfo Client< Real , Dim , BType , Degree >::_phase7( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , Profiler &profiler )
{
	PhaseInfo phaseInfo;

	typename Client< Real , Dim , BType , Degree >::_State7 state7;
	if constexpr( Dim==3 ) if( clientReconInfo.mergeType!=ClientReconstructionInfo< Real , Dim >::MergeType::NONE )
	{
		Timer timer;
		phaseInfo.readBytes += _receive7( clientReconInfo , state7 , profiler );
		phaseInfo.readTime += timer.wallTime();
	}

	{
		Timer timer;
		_process7( clientReconInfo , state7 , profiler );
		phaseInfo.processTime += timer.wallTime();
	}
	return phaseInfo;
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
size_t Client< Real , Dim , BType , Degree >::_receive7( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , _State7 &state7 , Profiler &profiler )
{
	size_t ioBytes = 0;
	auto Receive = [&]( BoundaryData *&boundary , std::vector< std::vector< Real > > &dValues , typename ClientReconstructionInfo< Real , Dim >::ShareType st )
	{
		ClientServerStream< true > serverStream( _serverSocket , _index , clientReconInfo , st );
		serverStream.ioBytes = 0;
		serverStream.read( state7.isoValue );
		if( clientReconInfo.mergeType!=ClientReconstructionInfo< Real , Dim >::MergeType::NONE )
		{
			XForm< Real , Dim > modelToUnitCube;
			boundary = new BoundaryData( serverStream , modelToUnitCube , MEMORY_ALLOCATOR_BLOCK_SIZE );
			if( !clientReconInfo.linearFit )
			{
				dValues.resize( boundary->sliceValues.size() );
				for( unsigned int i=0 ; i<dValues.size() ; i++ ) serverStream.read( dValues[i] );
			}
		}
		profiler.update();
		return serverStream.ioBytes;
	};

	if( _range.first!=0 )                                 ioBytes += Receive( state7.backBoundary  , state7.backDValues  , ClientReconstructionInfo< Real , Dim >::BACK );
	if( _range.second!=(1<<clientReconInfo.sharedDepth) ) ioBytes += Receive( state7.frontBoundary , state7.frontDValues , ClientReconstructionInfo< Real , Dim >::FRONT );

	return ioBytes;
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
template< bool HasGradients , bool HasDensity >
void Client< Real , Dim , BType , Degree >::_writeMeshWithData( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , _State7 &state7 , XForm< Real , Dim+1 > unitCubeToModel )
{
	Timer timer;
	std::string tempHeader( "PR_" );
	using AuxData = typename AuxDataFactory::VertexType;
	AuxDataFactory auxDataFactory( clientReconInfo.auxProperties );
	if( clientReconInfo.tempDir.length() ) tempHeader = PointPartition::FileDir( clientReconInfo.tempDir , tempHeader );

	SparseNodeData< ProjectiveData< AuxData , Real > , IsotropicUIntPack< Dim , DataSig > > *auxDataField = NULL;
	if( _auxDataField.size() ) auxDataField = &_auxDataField;

	std::string outFileName;
	{
		std::stringstream ss;
		ss << clientReconInfo.header << "." << _index << ".ply";
		outFileName = ss.str();
	}

	if( clientReconInfo.outDir.length() ) outFileName = PointPartition::FileDir( clientReconInfo.outDir , outFileName );


	// A description of the output vertex information
	using VInfo = Reconstructor::OutputVertexInfo< Real , Dim , HasGradients , HasDensity , AuxDataFactory >;

	// A factory generating the output vertices
	using Factory = typename VInfo::Factory;
	Factory factory = VInfo::GetFactory( auxDataFactory );

	// A backing stream for the vertices
	Reconstructor::OutputInputFactoryTypeStream< Real , Dim , Factory , false , true , AuxData > vertexStream( factory , VInfo::Convert );
	Reconstructor::OutputInputFaceStream< Dim-1 , false , true > faceStream;

	typename LevelSetExtractor< Real , Dim , AuxData >::Stats stats;

	{
		// The transformed stream
		Reconstructor::TransformedOutputLevelSetVertexStream< Real , Dim , AuxData > __vertexStream( unitCubeToModel , vertexStream );

		// Extract the mesh
		stats = LevelSetExtractor< Real , Dim , AuxData >::template Extract< Reconstructor::WeightDegree , DataSig >
			(
				Sigs() ,
				UIntPack< Reconstructor::WeightDegree >() ,
				UIntPack< DataSig >() ,
				_tree ,
				clientReconInfo.reconstructionDepth ,
				_density ,
				auxDataField ,
				_solution ,
				state7.isoValue ,
				clientReconInfo.sharedDepth ,
				_range.first ,
				_range.second ,
				__vertexStream ,
				faceStream ,
				auxDataFactory() ,
				!clientReconInfo.linearFit ,
				HasGradients , false , true , false ,
				state7.backBoundary ,
				state7.frontBoundary ,
				state7.backDValues ,
				state7.frontDValues ,
				clientReconInfo.mergeType==ClientReconstructionInfo< Real , Dim >::MergeType::TOPOLOGY_AND_FUNCTION
			);
	}

	if( clientReconInfo.verbose>1 )
	{
		std::cout << "Vertices / Faces: " << vertexStream.size() << " / " << faceStream.size() << std::endl;
		std::cout << stats.toString() << std::endl;
		std::cout << "#            Got faces: " << timer << std::endl;
	}

	// Write the mesh to a .ply file
	std::vector< std::string > noComments;
	vertexStream.reset();
	PLY::Write< Factory , node_index_type , Real , Dim >( outFileName.c_str() , factory , vertexStream.size() , faceStream.size() , vertexStream , faceStream , PLY_BINARY_NATIVE , noComments );
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
void Client< Real , Dim , BType , Degree >::_process7( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , _State7 &state7 , Profiler &profiler )
{
	static const bool HasAuxData = !std::is_same< AuxDataFactory , VertexFactory::EmptyFactory< Real > >::value;
	AuxDataFactory auxDataFactory( clientReconInfo.auxProperties );

	// Extract the mesh
	if constexpr( Dim==3 )
	{
		XForm< Real , Dim+1 > unitCubeToModel;
		if( clientReconInfo.gridCoordinates )
		{
			unitCubeToModel = XForm< Real , Dim+1 >::Identity();
			unsigned int res = 1<<clientReconInfo.reconstructionDepth;
			for( unsigned int d=0 ; d<Dim ; d++ ) unitCubeToModel(d,d) = (Real)res;
		}
		else unitCubeToModel = _modelToUnitCube.inverse();
		if( clientReconInfo.density ) _writeMeshWithData< false , true  >( clientReconInfo , state7 , unitCubeToModel );
		else                          _writeMeshWithData< false , false >( clientReconInfo , state7 , unitCubeToModel );
	}

	if( clientReconInfo.ouputVoxelGrid )
	{
		std::string outFileName;
		{
			std::stringstream ss;
			ss << clientReconInfo.header << "." << _index << ".grid";
			outFileName = ss.str();
		}

		if( clientReconInfo.outDir.length() ) outFileName = PointPartition::FileDir( clientReconInfo.outDir , outFileName );

		unsigned int begin[Dim] , end[Dim] , res[Dim];
		for( unsigned int d=0 ; d<Dim-1 ; d++ ) begin[d] = 0 , end[d] = 1<<clientReconInfo.reconstructionDepth;
		begin[Dim-1] = _range.first<<( clientReconInfo.reconstructionDepth - clientReconInfo.sharedDepth ) , end[Dim-1] = _range.second<<( clientReconInfo.reconstructionDepth - clientReconInfo.sharedDepth );

		Pointer( Real ) values = _tree.template regularGridEvaluate< true >( _solution , begin , end , res );

		XForm< Real , Dim+1 > voxelToUnitCube = XForm< Real , Dim+1 >::Identity();
		for( int d=0 ; d<Dim ; d++ ) voxelToUnitCube( d , d ) = (Real)( 1. / res[d] ) , voxelToUnitCube( Dim , d ) = (Real)( 0.5 / res[d] );
		XForm< Real , Dim+1 > unitCubeToModel = _modelToUnitCube.inverse();
		RegularGrid< Real , Dim >::Write( outFileName , res , values , unitCubeToModel * voxelToUnitCube );

		DeletePointer( values );
	}

	profiler.update();
}


template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
Client< Real , Dim , BType , Degree >::Client( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , BinaryStream &stream , unsigned int phase )
	: _serverSocket( _INVALID_SOCKET_ ) , _tree( stream , MEMORY_ALLOCATOR_BLOCK_SIZE ) , _density(NULL) , _normalInfo(NULL) , _paddedNormalInfo(NULL) , _iInfo(NULL) , _index(-1)
{
	stream.read( _modelToUnitCube );

	AuxDataFactory auxDataFactory( clientReconInfo.auxProperties );
	bool needAuxData = clientReconInfo.dataX>0 && auxDataFactory.bufferSize();

	if( phase!=3 && phase!=5 && phase!=7 ) MK_THROW( "Only phases 3, 5, and 7 supported: " , phase );

	stream.read( _index );
	stream.read( _range );

	_density = new DensityEstimator( stream );

	auto readSamples = [&]( BinaryStream &stream , std::vector< typename FEMTree< Dim , Real >::PointSample > &samples )
	{
		using SampleType = ProjectiveData< Point< Real , Dim > , Real >;
		struct T{ SampleType sample; node_index_type index; };

		{
			std::vector< FEMTreeNode * > nodes( _tree.spaceRoot().nodes() , NULL );
			size_t idx = 0;
			_tree.spaceRoot().processNodes( [&]( FEMTreeNode *node ){ nodes[idx++] = node; } );
if( idx!=nodes.size() ) MK_THROW( "uhoh" );

			std::vector< T > _samples;
			if( !stream.read( _samples ) ) MK_THROW( "Failed to read samples" );
			samples.resize( _samples.size() );
			// Convert indices to node pointers
			for( size_t i=0 ; i<samples.size() ; i++ ) samples[i].sample = _samples[i].sample , samples[i].node = nodes[ _samples[i].index ];
		}
	};

	if( phase==3 )
	{
		std::vector< FEMTreeNode * > nodes;
		auto readSamplesAndData = [&]( BinaryStream &stream , std::vector< typename FEMTree< Dim , Real >::PointSample > &samples , std::vector< InputSampleDataType > &sampleData )
		{
			using SampleType = ProjectiveData< Point< Real , Dim > , Real >;
			struct T
			{
				SampleType sample;
				node_index_type index;
			};
			{
				std::vector< T > _samples;
				if( !stream.read( _samples ) ) MK_THROW( "Failed to read samples" );
				size_t sz = _samples.size();
				SampleDataTypeSerializer< Real , Dim > serializer( clientReconInfo.auxProperties );
				samples.resize( sz );
				sampleData.resize( sz );
				size_t serializedSize = serializer.size();
				{
					Pointer( char ) buffer = NewPointer< char >( sz * serializedSize );
					if( !stream.read( buffer , sz*serializedSize ) ) MK_THROW( "Failed to read sample data" );
					for( unsigned int i=0 ; i<sz ; i++ ) serializer.deserialize( buffer+i*serializedSize , sampleData[i] );
					DeletePointer( buffer );
				}
				// Convert indices to node pointers
				for( size_t i=0 ; i<samples.size() ; i++ ) samples[i].sample = _samples[i].sample , samples[i].node = nodes[ _samples[i].index ];
			}
		};

		// Get the mapping from indices to node pointers
		{
			FEMTreeNode *root = &_tree.spaceRoot();
			while( root->parent ) root = root->parent;
			nodes.reserve( _tree.tree().nodes() );
			root->processNodes( [&]( FEMTreeNode *node ){ nodes.push_back( node ); } );
		}
		_normalInfo = new SparseNodeData< Point< Real , Dim > , NormalSigs >( stream );
		_paddedNormalInfo = new SparseNodeData< Point< Real , Dim > , NormalSigs >( stream );
		readSamplesAndData( stream , _samples , _sampleData );
		readSamplesAndData( stream , _paddedSamples , _paddedSampleData );
	}
	else if( phase==5 )
	{
		_constraints.read( stream );

		_iInfo = new ApproximatePointInterpolationInfo( stream );

		readSamples( stream , _samples );

		if( needAuxData )
		{
			if constexpr( !AuxDataFactory::IsStaticallyAllocated() )
			{
				ProjectiveAuxDataTypeSerializer< Real > serializer( clientReconInfo.auxProperties );
				_auxDataField.read( stream , serializer );
			}
			else _auxDataField.read( stream );
		}
	}
	else if( phase==7 )
	{
		_solution.read( stream );

		bool needAuxData = clientReconInfo.dataX>0 && auxDataFactory.bufferSize();
		if( needAuxData )
		{
			if constexpr( !AuxDataFactory::IsStaticallyAllocated() )
			{
				ProjectiveAuxDataTypeSerializer< Real > serializer( clientReconInfo.auxProperties );
				_auxDataField.read( stream , serializer );
			}
			else _auxDataField.read( stream );
		}
	}
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
void Client< Real , Dim , BType , Degree >::_write( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , BinaryStream &stream , unsigned int phase ) const
{
	AuxDataFactory auxDataFactory( clientReconInfo.auxProperties );
	bool needAuxData = clientReconInfo.dataX>0 && auxDataFactory.bufferSize();

	if( phase!=1 && phase!=3 && phase!=5 ) MK_THROW( "Only phases 1, 3, and 5 supported: " , phase );

	_tree.write( stream , false );
	stream.write( _modelToUnitCube );

	stream.write( _index );
	stream.write( _range );

	_density->write( stream );

	auto writeSamples = [&]( BinaryStream &stream , const std::vector< typename FEMTree< Dim , Real >::PointSample > &samples )
	{
		// [NOTE] The samples may be assigned to ghost nodes, so we can't just use the node index
		using SampleType = ProjectiveData< Point< Real , Dim > , Real >;
		struct T { SampleType sample ; node_index_type index; };

		std::vector< node_index_type > oldIndices( _tree.spaceRoot().nodes() , -1 );
		std::vector< T > _samples( samples.size() );

		// Grab the old indices and over-write
		{
			size_t idx = 0;
			const_cast< FEMTree< Dim , Real > & >( _tree ).spaceRoot().processNodes( [&]( FEMTreeNode *node ){ oldIndices[idx] = node->nodeData.nodeIndex ; node->nodeData.nodeIndex = idx++; } );
		}

		for( size_t i=0 ; i<samples.size() ; i++ ) _samples[i].sample = samples[i].sample , _samples[i].index = samples[i].node->nodeData.nodeIndex;
		stream.write( _samples );

		// Set the node indices back
		{
			size_t idx = 0;
			const_cast< FEMTree< Dim , Real > & >( _tree ).spaceRoot().processNodes( [&]( FEMTreeNode *node ){ node->nodeData.nodeIndex = oldIndices[ idx++ ]; } );
		}
	};

	if( phase==1 )
	{
		auto writeSamplesAndData = [&]( BinaryStream &stream , const std::vector< typename FEMTree< Dim , Real >::PointSample > &samples , const std::vector< InputSampleDataType > &sampleData )
		{
			using SampleType = ProjectiveData< Point< Real , Dim > , Real >;
			struct T
			{
				SampleType sample;
				node_index_type index;
			};
			std::vector< T > _samples( samples.size() );
			for( size_t i=0 ; i<samples.size() ; i++ ) _samples[i].sample = samples[i].sample , _samples[i].index = samples[i].node->nodeData.nodeIndex;
			stream.write( _samples );

			SampleDataTypeSerializer< Real , Dim > serializer( clientReconInfo.auxProperties );
			size_t serializedSize = serializer.size();
			{
				Pointer( char ) buffer = NewPointer< char >( samples.size() * serializedSize );
				for( unsigned int i=0 ; i<samples.size() ; i++ ) serializer.serialize( sampleData[i] , buffer+i*serializedSize );
				stream.write( buffer , samples.size()*serializedSize );
				DeletePointer( buffer );
			}
		};
		_normalInfo->write( stream );
		_paddedNormalInfo->write( stream );
		writeSamplesAndData( stream , _samples , _sampleData );
		writeSamplesAndData( stream , _paddedSamples , _paddedSampleData );
	}
	else if( phase==3 )
	{
		_constraints.write( stream );

		_iInfo->write( stream );

		writeSamples( stream , _samples );

		if( needAuxData )
		{
			if constexpr( !AuxDataFactory::IsStaticallyAllocated() )
			{
				ProjectiveAuxDataTypeSerializer< Real > serializer( clientReconInfo.auxProperties );
				_auxDataField.write( stream , serializer );
			}
			else _auxDataField.write( stream );
		}
	}
	else if( phase==5 )
	{
		_solution.write( stream );

		if( _auxDataField.size() )
		{
			if constexpr( !AuxDataFactory::IsStaticallyAllocated() )
			{
				ProjectiveAuxDataTypeSerializer< Real > serializer( clientReconInfo.auxProperties );
				_auxDataField.write( stream , serializer );
			}
			else _auxDataField.write( stream );			
		}
	}
}
