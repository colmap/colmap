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
struct Server
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

protected:
	std::vector< SocketStream > _clientSockets;
	PointPartition::PointSetInfo< Real , Dim > _pointSetInfo;
	PointPartition::Partition _pointPartition;

	struct _State4
	{
		FEMTree< Dim , Real > tree;
		DenseNodeData< Real , Sigs > constraints , solution;
		ApproximatePointInterpolationInfo *iInfo;
		SparseNodeData< ProjectiveData< AuxData , Real > , IsotropicUIntPack< Dim , DataSig > > *auxDataField;

		_State4( void ) : tree( MEMORY_ALLOCATOR_BLOCK_SIZE ) , iInfo(NULL) , auxDataField(NULL) {}
		~_State4( void ){ delete iInfo ; delete auxDataField; }
	};
	struct _State6
	{
		using SliceSigs = typename Sigs::Transpose::Rest::Transpose;
		using Vertex = typename VertexFactory::PositionFactory< Real , Dim-1 >::VertexType;
		FEMTree< Dim-1 , Real > *sliceTree;
		XForm< Real , Dim > xForm;
		DenseNodeData< Real , SliceSigs > solution , dSolution;
		std::vector< std::conditional_t< Dim==3 , typename LevelSetExtractor< Real , 2 >::SliceValues , char > > sliceValues , dSliceValues;
		std::vector< Point< Real , Dim-1 > > vertices;

		_State6( void ) : sliceTree(NULL) {}
		~_State6( void ){ delete sliceTree; }
	};

	PhaseInfo _phase0( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , Profiler &profiler );
	PhaseInfo _phase2( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , ProjectiveData< Real , Real > &cumulativePointWeight , Profiler &profiler );
	PhaseInfo _phase4( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , ProjectiveData< Real , Real > cumulativePointWeight , unsigned int baseVCycles , Real &isoValue , Profiler &profiler );
	PhaseInfo _phase6( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , Real isoValue , bool showDiscontinuity , bool outputBoundarySlices , std::vector< unsigned int > &sharedVertexCounts , Profiler &profiler );

	template< typename _Real , unsigned int _Dim , BoundaryType _BType , unsigned int _Degree >
	friend std::vector< unsigned int > RunServer( PointPartition::PointSetInfo< _Real , _Dim > , PointPartition::Partition , std::vector< Socket > , ClientReconstructionInfo< _Real , _Dim > , unsigned int , unsigned int , bool , bool );
};


template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
std::vector< unsigned int > RunServer
(
	PointPartition::PointSetInfo< Real , Dim > pointSetInfo ,
	PointPartition::Partition pointPartition ,
	std::vector< Socket > clientSockets ,
	ClientReconstructionInfo< Real , Dim > clientReconInfo , 
	unsigned int baseVCycles , 
	unsigned int sampleMS ,
	bool showDiscontinuity ,
	bool outputBoundarySlices
)
{
	std::vector< unsigned int > sharedVertexCounts;
	Server< Real , Dim , BType , Degree > server;
	Profiler profiler( sampleMS );

	clientReconInfo.auxProperties = pointSetInfo.auxiliaryProperties;

	// Initialization
	{
		server._pointSetInfo = pointSetInfo;
		server._pointPartition = pointPartition;
		server._clientSockets.resize( clientSockets.size() );

		for( unsigned int i=0 ; i<clientSockets.size() ; i++ ) server._clientSockets[i] = SocketStream( clientSockets[i] );

		for( unsigned int i=0 ; i<clientSockets.size() ; i++ ) clientReconInfo.write( server._clientSockets[i] );
		for( unsigned int i=0 ; i<clientSockets.size() ; i++ ) server._clientSockets[i].write( i );
	}

	if( clientReconInfo.verbose>1 )
	{
		for( unsigned int i=0 ; i<clientSockets.size() ; i++ )
		{
#ifdef ADAPTIVE_PADDING
			std::pair< unsigned int , unsigned int > range = pointPartition.range( i );
			if( range.first<clientReconInfo.padSize ) range.first = 0;
			else range.first -= clientReconInfo.padSize;
			if( range.second+clientReconInfo.padSize>pointPartition.slabs() ) range.second = pointPartition.slabs();
			else range.second += clientReconInfo.padSize;
#else // !ADAPTIVE_PADDING
			std::pair< unsigned int , unsigned int > range = pointPartition.range( i , clientReconInfo.padSize );
#endif // ADAPTIVE_PADDING
			std::cout << "Range[ " << i << " ]: [ " << range.first << " , "  << range.second << " )" << std::endl;
		}
		std::cout << "Boundary-type = " << BoundaryNames[BType] << " ; Degree = " << Degree << std::endl;
	}

	ProjectiveData< Real , Real > cumulativePointWeight;
	Real isoValue;

	// [PHASE 0] Send the client initial information
	{
		profiler.reset();
		PhaseInfo phaseInfo = server._phase0( clientReconInfo , profiler );
		if( clientReconInfo.verbose>0 )
		{
			StreamFloatPrecision sfp( std::cout , 1 );
			std::cout << SendDataString( 0 , phaseInfo.writeBytes ) << phaseInfo.writeTime << " (s)" << std::endl;
		}
	}

	// [PHASE 2] Accumulate/Send the point weight
	{
		profiler.reset();
		PhaseInfo phaseInfo = server._phase2( clientReconInfo , cumulativePointWeight , profiler );
		if( clientReconInfo.verbose>0 )
		{
			StreamFloatPrecision sfp( std::cout , 1 );
			std::cout << ReceiveDataString( 2 , phaseInfo.readBytes ) << phaseInfo.readTime << " (s)" << std::endl;
			std::cout << "[PROCESS 2]         : " << phaseInfo.processTime << " (s), " << profiler(false) << std::endl;
			std::cout << SendDataString( 2 , phaseInfo.writeBytes ) << phaseInfo.writeTime << " (s)" << std::endl;
		}
	}

	// [PHASE 4] Accumulate/solve/send coarse system
	{
		profiler.reset();
		PhaseInfo phaseInfo = server._phase4( clientReconInfo , cumulativePointWeight , baseVCycles , isoValue , profiler );
		if( clientReconInfo.verbose>0 )
		{
			StreamFloatPrecision sfp( std::cout , 1 );
			std::cout << ReceiveDataString( 4 , phaseInfo.readBytes ) << phaseInfo.readTime << " (s)" << std::endl;
			std::cout << "[PROCESS 4]         : " << phaseInfo.processTime << " (s), " << profiler(false) << std::endl;
			std::cout << SendDataString( 4 , phaseInfo.writeBytes ) << phaseInfo.writeTime << " (s)" << std::endl;
		}
	}

	// [PHASE 6] Accumulate/merge/send boundary tree info
	{
		profiler.reset();
		PhaseInfo phaseInfo = server._phase6( clientReconInfo , isoValue , showDiscontinuity , outputBoundarySlices , sharedVertexCounts , profiler );
		if( clientReconInfo.verbose>0 )
		{
			StreamFloatPrecision sfp( std::cout , 1 );
			std::cout << ReceiveDataString( 6 , phaseInfo.readBytes ) << phaseInfo.readTime << " (s)" << std::endl;
			std::cout << "[PROCESS 6]         : " << phaseInfo.processTime << " (s), " << profiler(false) << std::endl;
			std::cout << SendDataString( 6 , phaseInfo.writeBytes ) << phaseInfo.writeTime << " (s)" << std::endl;
		}
	}
	return sharedVertexCounts;
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
PhaseInfo Server< Real , Dim , BType , Degree >::_phase0( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , Profiler &profiler )
{
	PhaseInfo phaseInfo;

	Timer timer;
	for( unsigned int i=0 ; i<_clientSockets.size() ; i++ )
	{
		ClientServerStream< false > clientStream( _clientSockets[i] , i , clientReconInfo );
		clientStream.ioBytes = 0;

		// Send the information about the point-set
		clientStream.write( _pointSetInfo.modelToUnitCube );

		// Send the client's range
#ifdef ADAPTIVE_PADDING
		std::pair< unsigned int , unsigned int > range = _pointPartition.range( i );
#else // !ADAPTIVE_PADDING
		std::pair< unsigned int , unsigned int > range = _pointPartition.range( i , 0 );
#endif // ADAPTIVE_PADDING
		clientStream.write( range );

		phaseInfo.writeBytes += clientStream.ioBytes;
	}
	phaseInfo.writeTime += timer.wallTime();
	profiler.update();
	return phaseInfo;
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
PhaseInfo Server< Real , Dim , BType , Degree >::_phase2( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , ProjectiveData< Real , Real > &cumulativePointWeight , Profiler &profiler )
{
	PhaseInfo phaseInfo;
	std::pair< size_t , size_t > cumulativeNodeCounts;


	cumulativeNodeCounts = std::pair< size_t , size_t >(0,0);
	cumulativePointWeight = ProjectiveData< Real , Real >( (Real)0. );

	for( unsigned int i=0 ; i<_clientSockets.size() ; i++ )
	{
		Timer timer;
		ProjectiveData< Real , Real > pointWeight;
		std::pair< size_t , size_t > nodeCounts;
		ClientServerStream< true > clientStream( _clientSockets[i] , i , clientReconInfo );
		clientStream.ioBytes = 0;
		clientStream.read( pointWeight );
		clientStream.read( nodeCounts );
		phaseInfo.readBytes += clientStream.ioBytes;
		phaseInfo.readTime += timer.wallTime();

		timer = Timer();
		cumulativePointWeight += pointWeight;
		cumulativeNodeCounts.first += nodeCounts.first;
		cumulativeNodeCounts.second += nodeCounts.second;
		phaseInfo.processTime += timer.wallTime();
	}

	Timer timer;
	for( unsigned int i=0 ; i<_clientSockets.size() ; i++ )
	{
		ClientServerStream< false > clientStream( _clientSockets[i] , i , clientReconInfo );
		clientStream.ioBytes = 0;
		clientStream.write( cumulativePointWeight );
		phaseInfo.writeBytes += clientStream.ioBytes;
	}
	phaseInfo.writeTime = timer.wallTime();

	if( clientReconInfo.verbose>1 ) std::cout << "Redundancy: " << cumulativeNodeCounts.second << " / " << cumulativeNodeCounts.first << " = " << (double)cumulativeNodeCounts.second/cumulativeNodeCounts.first << std::endl;

	profiler.update();

	return phaseInfo;
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
PhaseInfo Server< Real , Dim , BType , Degree >::_phase4( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , ProjectiveData< Real , Real > cumulativePointWeight , unsigned int baseVCycles , Real &isoValue , Profiler &profiler )
{
	constexpr int MaxDegree = Reconstructor::Poisson::NormalDegree > Degrees::Max() ? Reconstructor::Poisson::NormalDegree : Degrees::Max();
	const int StartOffset = BSplineSupportSizes< MaxDegree >::SupportStart;
	const int   EndOffset = BSplineSupportSizes< MaxDegree >::SupportEnd+1;
	PhaseInfo phaseInfo;

	_State4 state4;

	std::vector< std::vector< node_index_type > > clientsToServer( _clientSockets.size() );

	// Initialize the tree
	{
		Timer timer;
		FEMTreeInitializer< Dim , Real >::Initialize( state4.tree.spaceRoot() , clientReconInfo.baseDepth , []( int , int[] ){ return true; } , state4.tree.nodeAllocators[0] , state4.tree.initializer() );

		auto hasDataFunctor = []( const FEMTreeNode * ){ return false; };

		// Set the root of the tree so we can copy constraints into it
		auto addNodeFunctor = [&]( int d , const int off[Dim] ){ return d<=(int)clientReconInfo.baseDepth; };
		state4.tree.template finalizeForMultigrid< MaxDegree , Degrees::Max() >( clientReconInfo.baseDepth , addNodeFunctor , hasDataFunctor , std::make_tuple() , std::make_tuple() );
		phaseInfo.processTime += timer.wallTime();
		profiler.update();
	}

	// Get/merge client data
	for( unsigned int i=0 ; i<_clientSockets.size() ; i++ )
	{
		Timer timer;

		std::vector< node_index_type > &clientToServer = clientsToServer[i];
		ClientServerStream< true > clientStream( _clientSockets[i] , i , clientReconInfo , ClientReconstructionInfo< Real , Dim >::BACK );
		clientStream.ioBytes = 0;

		// The number of nodes in the client's coarse sub-tree
		size_t subNodeCount;
		clientStream.read( subNodeCount );

		// The nodes in the client's coarse sub-tree
		Pointer( FEMTreeNode ) subNodes = NewPointer< FEMTreeNode >( subNodeCount );
		clientStream.read( subNodes , subNodeCount );

		phaseInfo.readBytes += clientStream.ioBytes;
		phaseInfo.readTime += timer.wallTime();


		timer = Timer();
		// Transform the nodes indices back to pointers
		TreeIndicesToAddresses( subNodes , subNodeCount );
		clientToServer.resize( subNodeCount , -1 );

		// Find the map from client to index to server index and add client nodes as necessary
		std::function< void ( FEMTreeNode * , const FEMTreeNode * ) > ProcessIndices = [&]( FEMTreeNode *serverNode , const FEMTreeNode *clientNode )
		{
			if( clientNode->nodeData.nodeIndex!=-1 ) clientToServer[ clientNode->nodeData.nodeIndex ] = serverNode->nodeData.nodeIndex;
			if( clientNode->children )
			{
				if( !serverNode->children ) serverNode->template initChildren< false >( state4.tree.nodeAllocators[0] , state4.tree.initializer() );
				for( int c=0 ; c<(1<<Dim) ; c++ ) ProcessIndices( serverNode->children+c , clientNode->children+c );
			}
		};
		ProcessIndices( const_cast< FEMTreeNode * >( &state4.tree.tree() ) , &subNodes[0] );
		profiler.update();
		DeletePointer( subNodes );
		phaseInfo.processTime += timer.wallTime();
	}

	{
		Timer timer;
		constexpr int MaxDegree = Reconstructor::Poisson::NormalDegree > Degrees::Max() ? Reconstructor::Poisson::NormalDegree : Degrees::Max();
		auto hasDataFunctor = []( const FEMTreeNode * ){ return true; };
		auto addNodeFunctor = [&]( int d , const int off[Dim] ){ return d<=(int)clientReconInfo.baseDepth; };
		std::vector< node_index_type > newToOld = state4.tree.template finalizeForMultigrid< MaxDegree , Degrees::Max() >( clientReconInfo.baseDepth , addNodeFunctor , hasDataFunctor , std::make_tuple() , std::make_tuple() );
		node_index_type idx = newToOld[0];
		for( unsigned int i=0 ; i<newToOld.size() ; i++ ) idx = std::max< node_index_type >( idx , newToOld[i] );
		idx++;
		std::vector< node_index_type > oldToNew( idx , -1 );
		for( unsigned int i=0 ; i<newToOld.size() ; i++ ) if( newToOld[i]!=-1 ) oldToNew[ newToOld[i] ] = i;
		for( unsigned int i=0 ; i<clientsToServer.size() ; i++ ) for( unsigned int j=0 ; j<clientsToServer[i].size() ; j++ ) clientsToServer[i][j] = oldToNew[ clientsToServer[i][j] ];
		phaseInfo.processTime += timer.wallTime();
		profiler.update();
	}

	Timer timer;
	AuxDataFactory auxDataFactory( _pointSetInfo.auxiliaryProperties );
	bool needAuxData = clientReconInfo.dataX>0 && auxDataFactory.bufferSize();

	state4.constraints = state4.tree.initDenseNodeData( Sigs() );

	{
		Real targetValue = clientReconInfo.targetValue;
		state4.iInfo = new ApproximatePointInterpolationInfo( Reconstructor::Poisson::ConstraintDual< Dim , Real >( targetValue , clientReconInfo.pointWeight * cumulativePointWeight.value() ) , Reconstructor::Poisson::SystemDual< Dim , Real >( clientReconInfo.pointWeight * cumulativePointWeight.value() ) , true );
		state4.iInfo->iData.reserve( state4.tree.nodesSize() );
	}

	using InterpolationData = typename ApproximatePointInterpolationInfo::Data;
	auto  preMergeFunctor = []( InterpolationData data ){ data.position *= data.weight ; return data; };
	auto postMergeFunctor = []( InterpolationData data ){ data.position /= data.weight ; return data; };

	if( needAuxData )
	{
		state4.auxDataField = new SparseNodeData< ProjectiveData< AuxData , Real > , IsotropicUIntPack< Dim , DataSig > >();
		state4.auxDataField->reserve( state4.tree.nodesSize() );
	}
	phaseInfo.processTime += timer.wallTime();
	profiler.update();

	for( unsigned int i=0 ; i<_clientSockets.size() ; i++ )
	{
		std::vector< node_index_type > &clientToServer = clientsToServer[i];
		ClientServerStream< true > clientStream( _clientSockets[i] , i , clientReconInfo , ClientReconstructionInfo< Real , Dim >::FRONT );
		clientStream.ioBytes = 0;

		// The constraints
		{
			Timer timer;
			DenseNodeData< Real , Sigs > _constraints( clientStream );
			phaseInfo.readTime += timer.wallTime();

			timer = Timer();
			for( unsigned int i=0 ; i<_constraints.size() ; i++ )
				if( clientToServer[i]==-1 || clientToServer[i]>=(node_index_type)state4.constraints.size() ){ MK_WARN_ONCE( "Unmatched client node(s): " , clientToServer[i] ); }
				else state4.constraints[ clientToServer[i] ] += _constraints[i];
			phaseInfo.processTime += timer.wallTime();
		}

		// The points
		{
			Timer timer;
			ApproximatePointInterpolationInfo _iInfo( clientStream );
			phaseInfo.readTime += timer.wallTime();

			timer = Timer();
			state4.iInfo->iData.mergeFromTarget( _iInfo.iData , [&]( unsigned int idx ){ return clientToServer[idx]; } , preMergeFunctor );
			phaseInfo.processTime += timer.wallTime();
		}
		// The data field
		if( needAuxData )
		{
			Timer timer;
			SparseNodeData< ProjectiveData< AuxData , Real > , IsotropicUIntPack< Dim , DataSig > > *_auxDataField;
			if constexpr( !AuxDataFactory::IsStaticallyAllocated() )
			{
				ProjectiveAuxDataTypeSerializer< Real > serializer( clientReconInfo.auxProperties );
				_auxDataField = new SparseNodeData< ProjectiveData< AuxData , Real > , IsotropicUIntPack< Dim , DataSig > >( clientStream , serializer );
			}
			else _auxDataField = new SparseNodeData< ProjectiveData< AuxData , Real > , IsotropicUIntPack< Dim , DataSig > >( clientStream );
			phaseInfo.readTime += timer.wallTime();

			timer = Timer();
			state4.auxDataField->mergeFromTarget( *_auxDataField , [&]( unsigned int idx ){ return clientToServer[idx]; } );
			phaseInfo.processTime += timer.wallTime();

			profiler.update();
			delete _auxDataField;
		}
		else profiler.update();
		phaseInfo.readBytes += clientStream.ioBytes;
	}

	if( needAuxData )
	{
		Timer timer;
		auto nodeFunctor = [&]( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >* n )
		{
			ProjectiveData< AuxData , Real >* clr = state4.auxDataField->operator()( n );
			if( clr ) (*clr) *= (Real)pow( clientReconInfo.dataX , state4.tree.depth( n ) );
		};
		state4.tree.tree().processNodes( nodeFunctor );
		phaseInfo.processTime += timer.wallTime();
	}
	{
		Timer timer;
		for( unsigned int i=0 ; i<state4.iInfo->iData.size() ; i++ ) state4.iInfo->iData[i] = postMergeFunctor( state4.iInfo->iData[i] );
		phaseInfo.processTime += timer.wallTime();
	}

	{
		Timer timer;

		typename FEMTree< Dim , Real >::SolverInfo sInfo;
		sInfo.cgDepth = 0 , sInfo.cascadic = true , sInfo.vCycles = 1 , sInfo.iters = clientReconInfo.iters , sInfo.cgAccuracy = clientReconInfo.cgSolverAccuracy , sInfo.verbose = clientReconInfo.verbose>1 , sInfo.showResidual = clientReconInfo.verbose>2 , sInfo.showGlobalResidual = SHOW_GLOBAL_RESIDUAL_NONE , sInfo.sliceBlockSize = 1;
		sInfo.baseVCycles = baseVCycles;
		typename FEMIntegrator::template System< Sigs , IsotropicUIntPack< Dim , 1 > > F( { 0. , 1. } );
		state4.solution = state4.tree.solveSystem( Sigs() , F , state4.constraints , clientReconInfo.baseDepth , std::min< unsigned int >( clientReconInfo.sharedDepth , clientReconInfo.solveDepth ) , sInfo , std::make_tuple( state4.iInfo ) );
		phaseInfo.processTime += timer.wallTime();

		profiler.update();
	}

	for( unsigned int i=0 ; i<_clientSockets.size() ; i++ )
	{
		Timer timer;
		std::pair< unsigned int , unsigned int > range;
#ifdef ADAPTIVE_PADDING
		// Compute the range of slices @ clientReconInfo.sharedDepth that can contribute to the client
		{
			range = _pointPartition.range( i );
			if( range.first<clientReconInfo.padSize ) range.first = 0;
			else range.first -= clientReconInfo.padSize;
			if( range.second+clientReconInfo.padSize>_pointPartition.slabs() ) range.second = _pointPartition.slabs();
			else range.second += clientReconInfo.padSize;
		}
#else // !ADAPTIVE_PADDING
		range = _pointPartition.range( i , clientReconInfo.padSize );
#endif // ADAPTIVE_PADDING

		ClientServerStream< false > clientStream( _clientSockets[i] , i , clientReconInfo );
		clientStream.ioBytes = 0;
		auto keepNodeFunctor = [&]( const FEMTreeNode *node )
		{
			int d , off[Dim];
			state4.tree.depthAndOffset( node , d , off );

			// Pre-tree nodes need to be shared and nodes finer than @ clientReconInfo.sharedDepth don't need to be
			if( d<0 ) return true;
			else if( d>(int)clientReconInfo.sharedDepth )
			{
				MK_WARN( "Why does the client have fine nodes?" );
				return false;
			}
			else
			{
				// Compute the space of the range of the support @ clientReconInfo.sharedDepth
				int start = ( off[Dim-1] + StartOffset )<<(clientReconInfo.sharedDepth-d);
				int end   = ( off[Dim-1] +   EndOffset )<<(clientReconInfo.sharedDepth-d);

				// Keep the node if its support fall within the client's range
				return start<(int)range.second && end>(int)range.first;
			}
		};
		size_t subNodeCount;
		Pointer( FEMTreeNode ) subNodes = state4.tree.tree().serializeSubTree( keepNodeFunctor , subNodeCount );

		node_index_type count = 0;
		subNodes[0].processNodes( [&]( const FEMTreeNode *node ){ if( node->nodeData.nodeIndex!=-1 ) count++; } );
		std::vector< node_index_type > newToOld( count );
		count = 0;
		for( size_t i=0 ; i<subNodeCount ; i++ ) if( subNodes[i].nodeData.nodeIndex!=-1 )
		{
			newToOld[ count ] = subNodes[i].nodeData.nodeIndex;
			subNodes[i].nodeData.nodeIndex = count++;
		}

		TreeAddressesToIndices( subNodes , subNodeCount );
		phaseInfo.processTime += timer.wallTime();

		timer = Timer();
		clientStream.write( subNodeCount );
		clientStream.write( subNodes , subNodeCount );
		phaseInfo.writeTime += timer.wallTime();

		timer = Timer();
		DenseNodeData< Real , Sigs > _solution( count );
		for( unsigned int i=0 ; i<newToOld.size() ; i++ ) _solution[i] = state4.solution[ newToOld[i] ];
		phaseInfo.processTime += timer.wallTime();

		timer = Timer();
		_solution.write( clientStream );
		phaseInfo.writeTime += timer.wallTime();

		if( needAuxData )
		{
			Timer timer;
			SparseNodeData< ProjectiveData< AuxData , Real > , IsotropicUIntPack< Dim , DataSig > > _auxDataField;
			_auxDataField.reserve( count );
			for( size_t i=0 ; i<subNodeCount ; i++ ) if( subNodes[i].nodeData.nodeIndex!=-1 )
			{
				const FEMTreeNode *node = state4.tree.node( newToOld[ subNodes[i].nodeData.nodeIndex ] );
				ProjectiveData< AuxData , Real > *data = state4.auxDataField->operator()( node );
				if( data ) _auxDataField[ subNodes+i ] = *data;
			}
			phaseInfo.processTime += timer.wallTime();

			timer = Timer();
			if constexpr( !AuxDataFactory::IsStaticallyAllocated() )
			{
				ProjectiveAuxDataTypeSerializer< Real > serializer( clientReconInfo.auxProperties );
				_auxDataField.write( clientStream , serializer );
			}
			else _auxDataField.write( clientStream );
			phaseInfo.writeTime += timer.wallTime();
		}
		profiler.update();
		DeletePointer( subNodes );
		phaseInfo.writeBytes += clientStream.ioBytes;
	}

	{
		std::pair< double , double > isoInfo(0.,0.);
		for( unsigned int i=0 ; i<_clientSockets.size() ; i++ )
		{
			std::pair< double , double > _isoInfo(0.,0.);
			_clientSockets[i].read( _isoInfo );
			isoInfo.first += _isoInfo.first;
			isoInfo.second += _isoInfo.second;
		}
		if( clientReconInfo.verbose>1 ) std::cout << "Iso-value: " << ( isoInfo.first / isoInfo.second ) << std::endl;
		isoValue = (Real)( isoInfo.first / isoInfo.second );
	}

	if( clientReconInfo.outputSolution )
	{
		std::string outFileName = std::string( "solution.tree" );
		if( clientReconInfo.outDir.length() ) outFileName = PointPartition::FileDir( clientReconInfo.outDir , outFileName );

		FILE* fp = fopen( outFileName.c_str() , "wb" );
		if( !fp ) MK_THROW( "Failed to open file for writing: " , outFileName );
		FileStream fs(fp);
		FEMTree< Dim , Real >::WriteParameter( fs );
		DenseNodeData< Real , Sigs >::WriteSignatures( fs );
		XForm< Real , Dim+1 > voxelToUnitCube = XForm< Real , Dim+1 >::Identity();
		state4.tree.write( fs , false );
		fs.write( voxelToUnitCube );
		state4.solution.write( fs );
		fclose( fp );
	}

	return phaseInfo;
}

template< typename Real , unsigned int Dim , BoundaryType BType , unsigned int Degree >
PhaseInfo Server< Real , Dim , BType , Degree >::_phase6( const ClientReconstructionInfo< Real , Dim > &clientReconInfo , Real isoValue , bool showDiscontinuity , bool outputBoundarySlices , std::vector< unsigned int > &sharedVertexCounts , Profiler &profiler )
{
	using SliceSigs = typename _State6::SliceSigs;
	using Vertex = typename _State6::Vertex;
	PhaseInfo phaseInfo;

	sharedVertexCounts.resize( _clientSockets.size()-1 );

	if constexpr( Dim==3 ) for( unsigned int i=0 ; i<_clientSockets.size()-1 ; i++ )
	{
		_State6 state6;

		auto ReadSlice = [&]( BinaryStream &clientStream , DenseNodeData< Real , SliceSigs > &solution , DenseNodeData< Real , SliceSigs > &dSolution , XForm< Real , Dim > &modelToUnitCube )
		{
			FEMTree< Dim-1 , Real > *sliceTree = new FEMTree< Dim-1 , Real >( clientStream , MEMORY_ALLOCATOR_BLOCK_SIZE );
			clientStream.read( modelToUnitCube );
			solution.read( clientStream );
			if( !clientReconInfo.linearFit ) dSolution.read( clientStream );
			return sliceTree;
		};


		{
			ClientServerStream< true > clientStream0( _clientSockets[i+0] , i+0 , clientReconInfo , ClientReconstructionInfo< Real , Dim >::FRONT );
			ClientServerStream< true > clientStream1( _clientSockets[i+1] , i+1 , clientReconInfo , ClientReconstructionInfo< Real , Dim >::BACK  );
			DenseNodeData< Real , SliceSigs > backSolution , frontSolution , backDSolution , frontDSolution;
			FEMTree< Dim-1 , Real > *backSliceTree , *frontSliceTree;
			XForm< Real , Dim > backXForm , frontXForm;

			Timer timer;
			backSliceTree  = ReadSlice( clientStream0 ,  backSolution ,  backDSolution , backXForm );
			frontSliceTree = ReadSlice( clientStream1 , frontSolution , frontDSolution , frontXForm );
			phaseInfo.readTime += timer.wallTime();

			if( showDiscontinuity )
			{
				double discontinuityTime = 0;
				size_t discontinuityCount = 0;
				double l1 = 0 , l2 = 0 , d2 = 0 , dInf = 0;

				double t = Time();

				int res1 = 0 , res2 = 0;
				Pointer( Real ) values1 =  backSliceTree->template regularGridEvaluate< true >(  backSolution , res1 , -1 , false );
				Pointer( Real ) values2 = frontSliceTree->template regularGridEvaluate< true >( frontSolution , res2 , -1 , false );
				if( res1!=res2 ) MK_THROW( "Different resolutions: " , res1 , " != " , res2 );
				size_t count = 1;
				for( unsigned int d=0 ; d<(Dim-1) ; d++ ) count *= (unsigned int)res1;
				discontinuityCount += count;
				for( unsigned int i=0 ; i<count ; i++ ) l1 += values1[i] * values1[i] , l2 += values2[i] * values2[i] , d2 += ( values1[i] - values2[i] ) * ( values1[i] - values2[i] ) , dInf = std::max< double >( dInf , fabs( values1[i]-values2[i] ) );
				DeletePointer( values1 );
				DeletePointer( values2 );

				discontinuityTime += Time()-t;

				if( discontinuityCount )
				{
					d2 /= discontinuityCount;
					l1 /= discontinuityCount;
					l2 /= discontinuityCount;
					std::cout << "Discontinuity L2 / L-infinity: " << sqrt(d2) << " / " << dInf << " [ " << sqrt(l1) << " , " << sqrt(l2) << " ] " << discontinuityTime << " (s)" << std::endl;
				}
				profiler.update();
			}

			timer = Timer();
			state6.sliceTree = FEMTree< Dim-1 , Real >::Merge( *backSliceTree , *frontSliceTree , MEMORY_ALLOCATOR_BLOCK_SIZE );
			state6.solution = state6.sliceTree->initDenseNodeData( SliceSigs() );
			state6.sliceTree->merge( * backSliceTree ,  backSolution , state6.solution );
			state6.sliceTree->merge( *frontSliceTree , frontSolution , state6.solution );
			for( unsigned int j=0 ; j<state6.solution.size() ; j++ ) state6.solution[j] /= (Real)2.;
			if( !clientReconInfo.linearFit )
			{
				state6.dSolution = state6.sliceTree->initDenseNodeData( SliceSigs() );
				state6.sliceTree->merge( * backSliceTree ,  backDSolution , state6.dSolution );
				state6.sliceTree->merge( *frontSliceTree , frontDSolution , state6.dSolution );
				for( unsigned int j=0 ; j<state6.dSolution.size() ; j++ ) state6.dSolution[j] /= (Real)2.;
			}
			state6.xForm = ( backXForm + frontXForm )/(Real)2.;
			phaseInfo.processTime += timer.wallTime();

			profiler.update();

			if( outputBoundarySlices )
			{
				auto WriteBoundary = [&]( std::string fileName , const FEMTree< Dim-1 , Real > *sliceTree , const DenseNodeData< Real , SliceSigs > &solution )
				{
					FILE* fp = fopen( fileName.c_str() , "wb" );
					if( !fp ) MK_THROW( "Failed to open file for writing: " , fileName );
					FileStream fs(fp);
					FEMTree< Dim-1 , Real >::WriteParameter( fs );
					DenseNodeData< Real , SliceSigs >::WriteSignatures( fs );
					sliceTree->write( fs , false );
					fs.write( state6.xForm );
					solution.write( fs );
					fclose( fp );
				};
				{
					std::stringstream ss;
					ss << "back." << i << ".tree";
					WriteBoundary( ss.str() , backSliceTree , backSolution );
				}
				{
					std::stringstream ss;
					ss << "front." << i << ".tree";
					WriteBoundary( ss.str() , frontSliceTree , frontSolution );
				}
				{
					std::stringstream ss;
					ss << "merged." << i << ".tree";
					WriteBoundary( ss.str() , state6.sliceTree , state6.solution );
				}
			}
			delete backSliceTree;
			delete frontSliceTree;

			phaseInfo.readBytes += clientStream0.ioBytes + clientStream1.ioBytes;
		}

		{
			Timer timer;
			using Factory = VertexFactory::EmptyFactory< Real >;
			using Data = typename Factory::VertexType;
			Factory factory;
			const typename FEMTree< Dim-1 , Real >::template DensityEstimator< Reconstructor::WeightDegree > *density=NULL;
			const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim-1 , DataSig > > *data=NULL;
			{
				VectorBackedOutputDataStream< Point< Real , Dim-1 > > _vertexStream( state6.vertices );
				using ExternalType = std::tuple< Point< Real , Dim-1 > , Point< Real , Dim-1 > , Real >;
				using InternalType = std::tuple< Point< Real , Dim-1 > >;
				auto converter = []( const ExternalType &xType )
					{
						InternalType iType;
						std::get< 0 >( iType ) = std::get< 0 >( xType );
						return iType;
					};
				OutputDataStreamConverter< InternalType , ExternalType > __vertexStream( _vertexStream , converter );

				LevelSetExtractor< Real , Dim-1 >::SetSliceValues( SliceSigs() , UIntPack< Reconstructor::WeightDegree >() , *state6.sliceTree , clientReconInfo.reconstructionDepth , density , state6.solution , isoValue , __vertexStream , !clientReconInfo.linearFit , false , state6.sliceValues , LevelSetExtractor< Real , Dim-1 , Vertex >::SetIsoEdgesFlag() );
				if( !clientReconInfo.linearFit ) LevelSetExtractor< Real , Dim-1 >::SetSliceValues( SliceSigs() , UIntPack< Reconstructor::WeightDegree >() , *state6.sliceTree , clientReconInfo.reconstructionDepth , density , state6.dSolution , isoValue , __vertexStream , false , false , state6.dSliceValues , LevelSetExtractor< Real , Dim-1 , Vertex >::SetCornerValuesFlag() );
			}
			sharedVertexCounts[i] = (unsigned int)state6.vertices.size();
			phaseInfo.processTime += timer.wallTime();
		}
		profiler.update();

		{
			Timer timer;
			ClientServerStream< false > clientStream0( _clientSockets[i+0] , i+0 , clientReconInfo , ClientReconstructionInfo< Real , Dim >::FRONT );
			ClientServerStream< false > clientStream1( _clientSockets[i+1] , i+1 , clientReconInfo , ClientReconstructionInfo< Real , Dim >::BACK  );
			clientStream0.ioBytes = 0;
			clientStream1.ioBytes = 0;
			clientStream0.write( isoValue );
			clientStream1.write( isoValue );
			if( clientReconInfo.mergeType!=ClientReconstructionInfo< Real , Dim >::MergeType::NONE )
			{
				LevelSetExtractor< Real , Dim-1 >::TreeSliceValuesAndVertexPositions::Write( clientStream0 , state6.sliceTree , state6.xForm , state6.sliceValues , state6.vertices , true );
				if( !clientReconInfo.linearFit )
				{
					for( unsigned int i=0 ; i<state6.dSliceValues.size() ; i++ )
					{
						clientStream0.write( state6.dSliceValues[i].cellIndices.counts[0] );
						clientStream0.write( state6.dSliceValues[i].cornerValues , state6.dSliceValues[i].cellIndices.counts[0] );
					}
				}
				LevelSetExtractor< Real , Dim-1 >::TreeSliceValuesAndVertexPositions::Write( clientStream1 , state6.sliceTree , state6.xForm , state6.sliceValues , state6.vertices , true );
				if( !clientReconInfo.linearFit )
				{
					for( unsigned int i=0 ; i<state6.dSliceValues.size() ; i++ )
					{
						clientStream1.write( state6.dSliceValues[i].cellIndices.counts[0] );
						clientStream1.write( state6.dSliceValues[i].cornerValues , state6.dSliceValues[i].cellIndices.counts[0] );
					}
				}
			}
			phaseInfo.writeBytes += clientStream0.ioBytes + clientStream1.ioBytes;
			phaseInfo.writeTime += timer.wallTime();
		}
		profiler.update();
	}

	return phaseInfo;
}
