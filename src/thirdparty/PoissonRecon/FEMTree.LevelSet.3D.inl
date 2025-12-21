/*
Copyright (c) 2022, Michael Kazhdan
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

// Specialized level-set surface extraction
template< bool HasData , typename Real , typename Data >
struct _LevelSetExtractor< HasData , Real , 3 , Data >
{
	static const unsigned int Dim = 3;
	// Store the position, the (interpolated) gradient, the weight, and possibly data
	typedef std::conditional_t
		<
			HasData ,
			DirectSum< Real , Point< Real , Dim > , Point< Real , Dim > , Real , Data > ,
			DirectSum< Real , Point< Real , Dim > , Point< Real , Dim > , Real >
		> Vertex;

	using OutputVertexStream = std::conditional_t
		<
			HasData ,
			OutputDataStream< Point< Real , Dim > , Point< Real , Dim > , Real , Data > ,
			OutputDataStream< Point< Real , Dim > , Point< Real , Dim > , Real >
		>;

protected:
	static std::atomic< size_t > _BadRootCount;

public:
	typedef typename FEMTree< Dim , Real >::LocalDepth LocalDepth;
	typedef typename FEMTree< Dim , Real >::LocalOffset LocalOffset;
	typedef typename FEMTree< Dim , Real >::ConstOneRingNeighborKey ConstOneRingNeighborKey;
	typedef typename FEMTree< Dim , Real >::ConstOneRingNeighbors ConstOneRingNeighbors;
	typedef RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > TreeNode;
	template< unsigned int WeightDegree > using DensityEstimator = typename FEMTree< Dim , Real >::template DensityEstimator< WeightDegree >;
	template< typename FEMSigPack , unsigned int PointD > using _Evaluator = typename FEMTree< Dim , Real >::template _Evaluator< FEMSigPack , PointD >;

	using Key = LevelSetExtraction::Key< Dim >;
	using IsoEdge = LevelSetExtraction::IsoEdge< Dim >;
	template< unsigned int D , unsigned int ... Ks >
	using HyperCubeTables = LevelSetExtraction::HyperCubeTables< D , Ks ... >;

	////////////////
	// FaceEdges //
	////////////////
	struct FaceEdges
	{
		IsoEdge edges[2];
		int count;
		FaceEdges( void ) : count(-1){}

		friend FaceEdges SetAtomic( volatile FaceEdges & value , FaceEdges newValue )
		{
			FaceEdges oldEdge;
			oldEdge.edges[0] = SetAtomic( value.edges[0] , newValue.edges[0] );
			oldEdge.edges[1] = SetAtomic( value.edges[1] , newValue.edges[1] );
			oldEdge.count = SetAtomic( value.count , newValue.count );
			return oldEdge;
		}
	};

	///////////////
	// SliceData //
	///////////////
	class SliceData
	{
	public:
		template< unsigned int Indices >
		struct  _Indices
		{
			node_index_type idx[Indices];
			_Indices( void ){ for( unsigned int i=0 ; i<Indices ; i++ ) idx[i] = -1; }
			node_index_type& operator[] ( int i ) { return idx[i]; }
			const node_index_type& operator[] ( int i ) const { return idx[i]; }
		};
		// The four corner indices associated with a square face
		using SquareCornerIndices = _Indices< HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() >;
		// The four edge indices associated with a square face
		using SquareEdgeIndices   = _Indices< HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() >;
		// The one face index associated with a square face
		using SquareFaceIndices   = _Indices< HyperCube::Cube< Dim-1 >::template ElementNum< 2 >() >;
	};

	/////////////////
	// SliceValues //
	/////////////////
	struct SliceValues
	{
		struct Scratch
		{
			using FKeyValues = std::vector< std::vector< std::pair< Key , std::vector< IsoEdge > > > >;
			using EKeyValues = std::vector< std::vector< std::pair< Key , std::pair< node_index_type , Vertex > > > >;
			using VKeyValues = std::vector< std::vector< std::pair< Key , Key > > >;

			FKeyValues fKeyValues;
			EKeyValues eKeyValues;
			VKeyValues vKeyValues;

#ifdef SANITIZED_PR
			Pointer( std::atomic< char > ) cSet;
			Pointer( std::atomic< char > ) eSet;
			Pointer( std::atomic< char > ) fSet;
#else // !SANITIZED_PR
			Pointer( char ) cSet;
			Pointer( char ) eSet;
			Pointer( char ) fSet;
#endif // SANITIZED_PR

			Scratch( void )
			{
				vKeyValues.resize( ThreadPool::NumThreads() );
				eKeyValues.resize( ThreadPool::NumThreads() );
				fKeyValues.resize( ThreadPool::NumThreads() );
#ifdef SANITIZED_PR
				cSet = NullPointer( std::atomic< char > );
				eSet = NullPointer( std::atomic< char > );
				fSet = NullPointer( std::atomic< char > );
#else // !SANITIZED_PR
				cSet = NullPointer( char );
				eSet = NullPointer( char );
				fSet = NullPointer( char );
#endif // SANITIZED_PR
			}

			~Scratch( void )
			{
#ifdef SANITIZED_PR
				DeletePointer( cSet );
				DeletePointer( eSet );
				DeletePointer( fSet );
#else // !SANITIZED_PR
				FreePointer( cSet );
				FreePointer( eSet );
				FreePointer( fSet );
#endif // SANITIZED_PR
			}

			void reset( const LevelSetExtraction::SliceCellIndexData< Dim > &cellIndices )
			{
				for( size_t i=0 ; i<vKeyValues.size() ; i++ ) vKeyValues[i].clear();
				for( size_t i=0 ; i<eKeyValues.size() ; i++ ) eKeyValues[i].clear();
				for( size_t i=0 ; i<fKeyValues.size() ; i++ ) fKeyValues[i].clear();
#ifdef SANITIZED_PR
				DeletePointer( cSet );
				DeletePointer( eSet );
				DeletePointer( fSet );
				if( cellIndices.counts[0] )
				{
					cSet = NewPointer< std::atomic< char > >( cellIndices.counts[0] );
					for( unsigned int i=0 ; i<cellIndices.counts[0] ; i++ ) cSet[i] = 0;
				}
				if( cellIndices.counts[1] )
				{
					eSet = NewPointer< std::atomic< char > >( cellIndices.counts[1] );
					for( unsigned int i=0 ; i<cellIndices.counts[1] ; i++ ) eSet[i] = 0;
				}
				if( cellIndices.counts[2] )
				{
					fSet = NewPointer< std::atomic< char > >( cellIndices.counts[2] );
					for( unsigned int i=0 ; i<cellIndices.counts[2] ; i++ ) fSet[i] = 0;
				}
#else // !SANITIZED_PR
				FreePointer( cSet );
				FreePointer( eSet );
				FreePointer( fSet );
				if( cellIndices.counts[0] )
				{
					cSet = AllocPointer< char >( cellIndices.counts[0] );
					memset( cSet , 0 , sizeof( char ) * cellIndices.counts[0] );
				}
				if( cellIndices.counts[1] )
				{
					eSet = AllocPointer< char >( cellIndices.counts[1] );
					memset( eSet , 0 , sizeof( char ) * cellIndices.counts[1] );
				}
				if( cellIndices.counts[2] )
				{
					fSet = AllocPointer< char >( cellIndices.counts[2] );
					memset( fSet , 0 , sizeof( char ) * cellIndices.counts[2] );
				}
#endif // SANITIZED_PR
			}
		};

		LevelSetExtraction::SliceCellIndexData< Dim > cellIndices;
		Pointer( Real ) cornerValues ; Pointer( Point< Real , Dim > ) cornerGradients;
		Pointer( Key ) edgeKeys;
		Pointer( FaceEdges ) faceEdges;
		Pointer( char ) mcIndices;
		LevelSetExtraction::KeyMap< Dim , std::vector< IsoEdge > > faceEdgeMap;
		LevelSetExtraction::KeyMap< Dim , std::pair< node_index_type , Vertex > > edgeVertexMap;
		LevelSetExtraction::KeyMap< Dim , Key > vertexPairMap;

		SliceValues( void )
		{
			_slice = -1;
			_oldCCount = _oldECount = _oldFCount = 0;
			_oldNCount = 0;
			cornerValues = NullPointer( Real ) ; cornerGradients = NullPointer( Point< Real , Dim > );
			edgeKeys = NullPointer( Key );
			faceEdges = NullPointer( FaceEdges );
			mcIndices = NullPointer( char );
		}
		~SliceValues( void )
		{
			_oldCCount = _oldECount = _oldFCount = 0;
			_oldNCount = 0;
			FreePointer( cornerValues ) ; FreePointer( cornerGradients );
			DeletePointer( edgeKeys );
			DeletePointer( faceEdges );
			FreePointer( mcIndices );
		}

		void setFromScratch( typename Scratch::VKeyValues &scratch )
		{
			for( unsigned int t=0 ; t<scratch.size() ; t++ )
			{
				for( int i=0 ; i<scratch[t].size() ; i++ )
				{
					vertexPairMap[ scratch[t][i].first ] = scratch[t][i].second;
					vertexPairMap[ scratch[t][i].second ] = scratch[t][i].first;
				}
				scratch[t].clear();
			}
		}

		void setFromScratch( typename Scratch::EKeyValues &scratch )
		{
			for( unsigned int t=0 ; t<scratch.size() ; t++ )
			{
				for( int i=0 ; i<scratch[t].size() ; i++ ) edgeVertexMap[ scratch[t][i].first ] = scratch[t][i].second;
				scratch[t].clear();
			}
		}

		void setFromScratch( typename Scratch::FKeyValues &scratch )
		{
			for( unsigned int t=0 ; t<scratch.size() ; t++ )
			{
				for( int i=0 ; i<scratch[t].size() ; i++ )
				{
					std::vector< IsoEdge > &faceEdges = faceEdgeMap[ scratch[t][i].first ];
					faceEdges.insert( faceEdges.end() , scratch[t][i].second.begin() , scratch[t][i].second.end() );
				}
				scratch[t].clear();
			}
		}

		unsigned int slice( void ) const { return _slice; }
		void reset( unsigned int slice , bool computeGradients )
		{
			_slice = slice;
			faceEdgeMap.clear() , edgeVertexMap.clear() , vertexPairMap.clear();

			if( _oldNCount<(node_index_type)cellIndices.size() )
			{
				_oldNCount = (node_index_type)cellIndices.size();
				FreePointer( mcIndices );
				if( _oldNCount>0 ) mcIndices = AllocPointer< char >( _oldNCount );
			}
			if( _oldCCount<(node_index_type)cellIndices.counts[0] )
			{
				_oldCCount = (node_index_type)cellIndices.counts[0];
				FreePointer( cornerValues ) ; FreePointer( cornerGradients );
				if( cellIndices.counts[0]>0 )
				{
					cornerValues = AllocPointer< Real >( _oldCCount );
					if( computeGradients ) cornerGradients = AllocPointer< Point< Real , Dim > >( _oldCCount );
				}
			}
			if( _oldECount<(node_index_type)cellIndices.counts[1] )
			{
				_oldECount = (node_index_type)cellIndices.counts[1];
				DeletePointer( edgeKeys );
				edgeKeys = NewPointer< Key >( _oldECount );
			}
			if( _oldFCount<(node_index_type)cellIndices.counts[2] )
			{
				_oldFCount = (node_index_type)cellIndices.counts[2];
				DeletePointer( faceEdges );
				faceEdges = NewPointer< FaceEdges >( _oldFCount );
			}
		}

		template< typename FaceIndexFunctor /* = std::function< LevelSetExtraction::Key< Dim > ( const TreeNode * , typename HyperCube::Cube< Dim >::template Element< 2 > ) */ >
		void addIsoEdges( const LevelSetExtraction::KeyGenerator< Dim > &keyGenerator , FaceIndexFunctor &faceIndexFunctor , const FEMTree< Dim , Real > &tree , const Scratch &scratch , const TreeNode *leaf , unsigned int sliceIndex , std::vector< IsoEdge > &edges , bool isOriented ) const
		{
			auto _FaceIndex = [&]( const TreeNode *node )
			{
				int depth , offset[Dim];
				tree.depthAndOffset( node , depth , offset );
				typename HyperCube::Cube< Dim >::template Element< 2 > f;
				if     ( offset[Dim-1]+0==sliceIndex ) f = typename HyperCube::Cube< Dim >::template Element< 2 >( HyperCube::BACK  , 0 );
				else if( offset[Dim-1]+1==sliceIndex ) f = typename HyperCube::Cube< Dim >::template Element< 2 >( HyperCube::FRONT , 0 );
				else MK_THROW( "Node/slice-index mismatch: " , offset[Dim-1] , " <-> " , sliceIndex );
				return faceIndexFunctor( node , f );
			};

			int flip = isOriented ? 0 : 1;

			node_index_type fIdx = cellIndices.template indices<2>(leaf->nodeData.nodeIndex)[0];
			if( scratch.fSet[fIdx] )
			{
				const FaceEdges &fe = faceEdges[ fIdx ];
				for( int i=0 ; i<fe.count ; i++ ) edges.push_back( IsoEdge( fe.edges[i][flip] , fe.edges[i][1-flip] ) );
			}
			else
			{
				Key key = _FaceIndex( leaf );
				typename LevelSetExtraction::KeyMap< Dim , std::vector< IsoEdge > >::const_iterator iter = faceEdgeMap.find( key );
				if( iter!=faceEdgeMap.end() )
				{
					const std::vector< IsoEdge >& _edges = iter->second;
					for( size_t i=0 ; i<_edges.size() ; i++ ) edges.push_back( IsoEdge( _edges[i][flip] , _edges[i][1-flip] ) );
				}
				else
				{
					LocalDepth d ; LocalOffset off;
					tree.depthAndOffset( leaf , d , off );
					// [WARNING] Is this right? If the face isn't set, wouldn't it inherit?
					MK_WARN( "Invalid face: [" , off[0] , " " , off[1] , " " , off[2] , " @ " , d , " | " , sliceIndex , " : " , leaf->nodeData.nodeIndex , " ( " , keyGenerator.to_string(key) , " | " , key.to_string() , " )"  );
				}
			}
		}

		bool setVertexPair( Key current , Key &pair ) const
		{
			typename LevelSetExtraction::KeyMap< Dim , Key >::const_iterator iter;

			if( (iter=vertexPairMap.find(current))!=vertexPairMap.end() )
			{
				pair = iter->second;
				return true;
			}
			else return false;
		}

		bool setEdgeVertex( Key key , std::pair< node_index_type , Vertex > &edgeVertex ) const
		{
			typename LevelSetExtraction::KeyMap< Dim , std::pair< node_index_type , Vertex > >::const_iterator iter;
			if( ( iter=edgeVertexMap.find( key ) )!=edgeVertexMap.end() )
			{
				edgeVertex = iter->second;
				return true;
			}
			else return false;
		}
	protected:
		node_index_type _oldCCount , _oldECount , _oldFCount;
		node_index_type _oldNCount;
		unsigned int _slice;
	};

	//////////////////
	// XSliceValues //
	//////////////////
	struct XSliceValues
	{
		struct Scratch
		{
			using FKeyValues = std::vector< std::vector< std::pair< Key , std::vector< IsoEdge > > > >;
			using EKeyValues = std::vector< std::vector< std::pair< Key , std::pair< node_index_type , Vertex > > > >;
			using VKeyValues = std::vector< std::vector< std::pair< Key , Key > > >;

			FKeyValues fKeyValues;
			EKeyValues eKeyValues;
			VKeyValues vKeyValues;

#ifdef SANITIZED_PR
			Pointer( std::atomic< char > ) eSet;
			Pointer( std::atomic< char > ) fSet;
#else // !SANITIZED_PR
			Pointer( char ) eSet;
			Pointer( char ) fSet;
#endif // SANITIZED_PR

			Scratch( void )
			{
				vKeyValues.resize( ThreadPool::NumThreads() );
				eKeyValues.resize( ThreadPool::NumThreads() );
				fKeyValues.resize( ThreadPool::NumThreads() );
#ifdef SANITIZED_PR
				eSet = NullPointer( std::atomic< char > );
				fSet = NullPointer( std::atomic< char > );
#else // !SANITIZED_PR
				eSet = NullPointer( char );
				fSet = NullPointer( char );
#endif // SANITIZED_PR
			}

			~Scratch( void )
			{
#ifdef SANITIZED_PR
				DeletePointer( eSet );
				DeletePointer( fSet );
#else // !SANITIZED_PR
				FreePointer( eSet );
				FreePointer( fSet );
#endif // SANITIZED_PR
			}

			void reset( const LevelSetExtraction::SlabCellIndexData< Dim > &cellIndices )
			{
				for( size_t i=0 ; i<vKeyValues.size() ; i++ ) vKeyValues[i].clear();
				for( size_t i=0 ; i<eKeyValues.size() ; i++ ) eKeyValues[i].clear();
				for( size_t i=0 ; i<fKeyValues.size() ; i++ ) fKeyValues[i].clear();
#ifdef SANITIZED_PR
				DeletePointer( eSet );
				DeletePointer( fSet );
				if( cellIndices.counts[0] )
				{
					eSet = NewPointer< std::atomic< char > >( cellIndices.counts[0] );
					for( unsigned int i=0 ; i<cellIndices.counts[0] ; i++ ) eSet[i] = 0;
				}
				if( cellIndices.counts[1] )
				{
					fSet = NewPointer< std::atomic< char > >( cellIndices.counts[1] );
					for( unsigned int i=0 ; i<cellIndices.counts[1] ; i++ ) fSet[i] = 0;
				}
#else // !SANITIZED_PR
				FreePointer( eSet );
				FreePointer( fSet );
				if( cellIndices.counts[0] )
				{
					eSet = AllocPointer< char >( cellIndices.counts[0] );
					memset( eSet , 0 , sizeof( char ) * cellIndices.counts[0] );
				}
				if( cellIndices.counts[1] )
				{
					fSet = AllocPointer< char >( cellIndices.counts[1] );
					memset( fSet , 0 , sizeof( char ) * cellIndices.counts[1] );
				}
#endif // SANITIZED_PR
			}
		};

		LevelSetExtraction::SlabCellIndexData< Dim > cellIndices;
		Pointer( Key ) edgeKeys;
		Pointer( FaceEdges ) faceEdges;
		LevelSetExtraction::KeyMap< Dim , std::vector< IsoEdge > > faceEdgeMap;
		LevelSetExtraction::KeyMap< Dim , std::pair< node_index_type , Vertex > > edgeVertexMap;
		LevelSetExtraction::KeyMap< Dim , Key > vertexPairMap;

		XSliceValues( void )
		{
			_slab = -1;
			_oldECount = _oldFCount = 0;
			edgeKeys = NullPointer( Key );
			faceEdges = NullPointer( FaceEdges );
		}

		~XSliceValues( void )
		{
			_oldECount = _oldFCount = 0;
			DeletePointer( edgeKeys );
			DeletePointer( faceEdges );
		}

		void setFromScratch( typename Scratch::VKeyValues &scratch )
		{
			for( unsigned int t=0 ; t<scratch.size() ; t++ )
			{
				for( int i=0 ; i<scratch[t].size() ; i++ )
				{
					vertexPairMap[ scratch[t][i].first ] = scratch[t][i].second;
					vertexPairMap[ scratch[t][i].second ] = scratch[t][i].first;
				}
				scratch[t].clear();
			}
		}

		void setFromScratch( typename Scratch::EKeyValues &scratch )
		{
			for( unsigned int t=0 ; t<scratch.size() ; t++ )
			{
				for( int i=0 ; i<scratch[t].size() ; i++ ) edgeVertexMap[ scratch[t][i].first ] = scratch[t][i].second;
				scratch[t].clear();
			}
		}

		void setFromScratch( typename Scratch::FKeyValues &scratch )
		{
			for( unsigned int t=0 ; t<scratch.size() ; t++ )
			{
				for( int i=0 ; i<scratch[t].size() ; i++ )
				{
					auto iter = faceEdgeMap.find( scratch[t][i].first );
					if( iter==faceEdgeMap.end() ) faceEdgeMap[ scratch[t][i].first ] = scratch[t][i].second;
					else for( int j=0 ; j<scratch[t][i].second.size() ; j++ ) iter->second.push_back( scratch[t][i].second[j] );
				}
				scratch[t].clear();
			}
		}

		unsigned int slab( void ) const { return _slab; }
		void reset( unsigned int slab )
		{
			_slab = slab;
			faceEdgeMap.clear() , edgeVertexMap.clear() , vertexPairMap.clear();

			if( _oldECount<(node_index_type)cellIndices.counts[0] )
			{
				_oldECount = (node_index_type)cellIndices.counts[0];
				DeletePointer( edgeKeys );
				edgeKeys = NewPointer< Key >( _oldECount );
			}
			if( _oldFCount<(node_index_type)cellIndices.counts[1] )
			{
				_oldFCount = (node_index_type)cellIndices.counts[1];
				DeletePointer( faceEdges );
				faceEdges = NewPointer< FaceEdges >( _oldFCount );
			}
		}

	protected:
		node_index_type _oldECount , _oldFCount;
		unsigned int _slab;
	};

	////////////////
	// SlabValues //
	////////////////
	struct SlabValues
	{
	protected:
		XSliceValues _xSliceValues[2];
		SliceValues _sliceValues[2];
		typename  SliceValues::Scratch  _sliceScratch[2];
		typename XSliceValues::Scratch _xSliceScratch[2];
	public:
		      SliceValues& sliceValues( int idx ){ return _sliceValues[idx&1]; }
		const SliceValues& sliceValues( int idx ) const { return _sliceValues[idx&1]; }
		      XSliceValues& xSliceValues( int idx ){ return _xSliceValues[idx&1]; }
		const XSliceValues& xSliceValues( int idx ) const { return _xSliceValues[idx&1]; }
		      typename  SliceValues::Scratch &sliceScratch ( int idx )       { return  _sliceScratch[idx&1]; }
		const typename  SliceValues::Scratch &sliceScratch ( int idx ) const { return  _sliceScratch[idx&1]; }
		      typename XSliceValues::Scratch &xSliceScratch( int idx )       { return _xSliceScratch[idx&1]; }
		const typename XSliceValues::Scratch &xSliceScratch( int idx ) const { return _xSliceScratch[idx&1]; }

		bool validSlice( int slice ) const { return _sliceValues[slice&1].slice()==slice; }
		bool validXSlice( int slab ) const { return _xSliceValues[slab&1].slab()==slab; }
	};

	static std::vector< std::pair< node_index_type , node_index_type > > SetIncidence( const FEMTree< Dim , Real > &tree , const FEMTree< Dim-1 , Real > &sliceTree , LocalDepth fullDepth , unsigned int sliceAtMaxDepth , unsigned int maxDepth )
	{
		using SliceTreeNode    = typename LevelSetExtractor< Real , Dim-1 , Point< Real , Dim-1 > >::TreeNode;
		using SliceLocalDepth  = typename FEMTree< Dim-1 , Real >::LocalDepth;
		using SliceLocalOffset = typename FEMTree< Dim-1 , Real >::LocalOffset;

		std::vector< std::pair< node_index_type , node_index_type > > incidence( sliceTree.nodesSize() , std::pair< node_index_type , node_index_type >( -1 , -1 ) );

		// Recursively set the incidence
		std::function< void ( const TreeNode * , const SliceTreeNode * ) > SetIncidenceFunctor = [&]( const TreeNode *node , const SliceTreeNode *sliceNode )
		{
			LocalDepth depth ; LocalOffset offset;
			SliceLocalDepth sliceDepth ; SliceLocalOffset sliceOffset;
			tree.depthAndOffset( node , depth , offset );
			sliceTree.depthAndOffset( sliceNode , sliceDepth , sliceOffset );
			if( depth!=sliceDepth ) MK_THROW( "Depths do not match: " , depth , " != " , sliceDepth );
			for( unsigned int i=0 ; i<Dim-1 ; i++ ) if( offset[i]!=sliceOffset[i] ) MK_THROW( "Offsets do not match[ " , i , "]: " , offset[i] , " != " , sliceOffset[i] );

			unsigned int beginAtMaxDepth = ( offset[Dim-1] + 0 )<<( maxDepth - depth );
			unsigned int   endAtMaxDepth = ( offset[Dim-1] + 1 )<<( maxDepth - depth );
			unsigned int   midAtMaxDepth = ( beginAtMaxDepth + endAtMaxDepth ) / 2;

			if( node->nodeData.nodeIndex==-1 ) return;
			else if( sliceNode->nodeData.nodeIndex==-1 ) MK_THROW( "Expected valid slice node" );

			if( sliceAtMaxDepth<beginAtMaxDepth || sliceAtMaxDepth>endAtMaxDepth ) MK_THROW( "Bad slice: " , sliceAtMaxDepth , " in [ " , beginAtMaxDepth , " , " , endAtMaxDepth , " ]" );
			if( depth>=fullDepth )
			{
				// Set the incidence
				if     ( sliceAtMaxDepth==beginAtMaxDepth ) incidence[ sliceNode->nodeData.nodeIndex ].second = node->nodeData.nodeIndex;
				else if( sliceAtMaxDepth==endAtMaxDepth   ) incidence[ sliceNode->nodeData.nodeIndex ].first  = node->nodeData.nodeIndex;
				else
				{
					incidence[ sliceNode->nodeData.nodeIndex ].second = node->nodeData.nodeIndex;
					incidence[ sliceNode->nodeData.nodeIndex ].first  = node->nodeData.nodeIndex;
				}
			}

			if( !GetGhostFlag( node->children ) )
			{
				if( !sliceNode->children )
				{
					int d , off[Dim] , _d , _off[Dim-1];
					Point< int , Dim > p;
					Point< int , Dim-1 > _p;
					for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = off[d];
					for( unsigned int d=0 ; d<Dim-1 ; d++ ) _p[d] = _off[d];
					tree.depthAndOffset( node , d , off );
					sliceTree.depthAndOffset( sliceNode , _d , _off );
					MK_THROW( "Expected slice children: " , p , " @ " , d , " <-> " , _p , " @ " , _d , " : ",  node->nodeData.nodeIndex , " <-> " , sliceNode->nodeData.nodeIndex );
				}
				if( sliceAtMaxDepth<=midAtMaxDepth ) for( int c=0 ; c<(1<<(Dim-1)) ; c++ ) SetIncidenceFunctor( node->children+(c             ) , sliceNode->children + c );
				if( sliceAtMaxDepth>=midAtMaxDepth ) for( int c=0 ; c<(1<<(Dim-1)) ; c++ ) SetIncidenceFunctor( node->children+(c|(1<<(Dim-1))) , sliceNode->children + c );
			}
		};
		SetIncidenceFunctor( &tree.spaceRoot() , &sliceTree.spaceRoot() );
		return incidence;
	}

	template< unsigned int WeightDegree , unsigned int DataSig , typename VertexStream , typename SliceFunctor /* = std::function< SliceValues & ( unsigned int ) > */ , typename ScratchFunctor /* = std::function< typename SliceValues::Scratch & ( unsigned int ) */ >
	static void CopyIsoStructure
	(
		const LevelSetExtraction::KeyGenerator< Dim > &keyGenerator ,
		const typename LevelSetExtractor< Real , Dim-1 , Point< Real , Dim-1 > >::TreeSliceValuesAndVertexPositions &boundaryInfo ,
		const FEMTree< Dim , Real > &tree ,
		LocalDepth fullDepth ,
		unsigned int sliceAtMaxDepth ,
		unsigned int maxDepth ,
		SliceFunctor sliceFunctor ,
		ScratchFunctor scratchFunctor ,
		const std::vector< std::pair< node_index_type , node_index_type > > &incidence ,
		VertexStream &vertexStream ,
		bool gradientNormals ,
		typename FEMIntegrator::template PointEvaluator< IsotropicUIntPack< Dim , DataSig > , ZeroUIntPack< Dim > > *pointEvaluator ,
		const DensityEstimator< WeightDegree >* densityWeights ,
		const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > >* data ,
		const Data &zeroData
	)
	{
		using SliceTreeNode    = typename LevelSetExtractor< Real , Dim-1 , Point< Real , Dim-1 > >::TreeNode;
		using SliceSliceValues = typename LevelSetExtractor< Real , Dim-1 , Point< Real , Dim-1 > >::SliceValues;
		using SliceKey         = LevelSetExtraction::Key< Dim-1 >;
		using SliceIsoEdge     = LevelSetExtraction::IsoEdge< Dim-1 >;

		auto PromoteIsoVertex = [&]( SliceKey key )
		{
			return keyGenerator( maxDepth , sliceAtMaxDepth , key );
		};

		auto PromoteIsoEdge = [&]( SliceIsoEdge edge )
		{
			IsoEdge pEdge;
			for( unsigned int i=0 ; i<2 ; i++ ) pEdge[i] = keyGenerator( maxDepth , sliceAtMaxDepth , edge[i] );
			return pEdge;
		};

		std::vector< node_index_type > vertexIndices( boundaryInfo.vertexPositions.size() );
		std::vector< Vertex > vertices( boundaryInfo.vertexPositions.size() );
		// Add the iso-vertices
		{
			static const unsigned int DataDegree = FEMSignature< DataSig >::Degree;

			// A functor returning the finest leaf node containing the iso-vertex
			std::function< const TreeNode * ( const TreeNode * , Key ) > FinestLeaf = [&]( const TreeNode *node , Key key )
			{
				const TreeNode *candidate = node;
				if( node->children )
				{
					LocalDepth depth ; LocalOffset offset;
					int start[Dim] , mid[Dim] , end[Dim];
					tree.depthAndOffset( node , depth , offset );
					for( unsigned int d=0 ; d<Dim ; d++ )
					{
						start[d] = (offset[d]+0)<<(maxDepth+2-depth);
						end  [d] = (offset[d]+1)<<(maxDepth+2-depth);
						mid  [d] = (start[d]+end[d])>>1;
					}
					for( unsigned int c=0 ; c<(1<<Dim) ; c++ )
					{
						bool containsKey = true;
						for( unsigned int d=0 ; d<Dim ; d++ )
							if( c&(1<<d) ) containsKey &= ((int)key[d]>=mid  [d] && (int)key[d]<=end[d]);
							else           containsKey &= ((int)key[d]>=start[d] && (int)key[d]<=mid[d]);
						if( containsKey )
						{
							const TreeNode *_candidate = FinestLeaf( node->children+c , key );
							if( tree.depth(_candidate)>tree.depth(candidate) ) candidate = _candidate;
						}
					}
				}
				return candidate;
			};

			std::vector< Key > keys;
			{
				std::vector< LevelSetExtraction::Key< Dim-1 > > _keys = boundaryInfo.vertexKeys();
				keys.resize( _keys.size() );
				for( unsigned int i=0 ; i<_keys.size() ; i++ ) keys[i] = PromoteIsoVertex( _keys[i] );
			}

			for( unsigned int i=0 ; i<boundaryInfo.vertexPositions.size(); i++ )
			{
				ConstPointSupportKey< IsotropicUIntPack< Dim , WeightDegree > > weightKey;
				ConstPointSupportKey< IsotropicUIntPack< Dim , DataDegree > > dataKey;
				Point< Real , Dim > p;
				for( unsigned int d=0 ; d<Dim-1 ; d++ ) p[d] = boundaryInfo.vertexPositions[i][d];
				p[Dim-1] = (Real)sliceAtMaxDepth/(Real)(1<<maxDepth);
#ifdef SHOW_WARNINGS
#pragma message( "[WARNING] Not setting normal gradients" )
#endif // SHOW_WARNINGS
				Real depth = (Real)1.;
				Data dataValue;
				const TreeNode *node = NULL;
				if( densityWeights || data ) node = FinestLeaf( &tree.spaceRoot() , keys[i] );
				if( densityWeights )
				{
					weightKey.set( node->depth() );
					weightKey.getNeighbors( node );
					Real weight;
					tree._getSampleDepthAndWeight( *densityWeights , node , p , weightKey , depth , weight );
				}
				if constexpr( HasData ) if( data )
				{
					dataKey.set( node->depth() );
					dataKey.getNeighbors( node );
					Point< Real , Dim > start;
					Real width;
					tree.startAndWidth( node , start , width );
					if( DataDegree==0 ) 
					{
						Point< Real , Dim > center( start[0] + width/2 , start[1] + width/2 , start[2] + width/2 );
						ProjectiveData< Data , Real > pValue( zeroData );
						tree.template _addEvaluation< ProjectiveData< Data , Real > , SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > , 0 >( *data , center , *pointEvaluator , dataKey , pValue );
						dataValue = pValue.weight ? pValue.value() : zeroData;
					}
					else
					{
						ProjectiveData< Data , Real > pValue( zeroData );
						tree.template _addEvaluation< ProjectiveData< Data , Real > , SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > , 0 >( *data , p , *pointEvaluator , dataKey , pValue );
						dataValue = pValue.weight ? pValue.value() : zeroData;
					}
				}
				vertices[i].template get<0>() = p;
				vertices[i].template get<1>() = Point< Real , Dim >();
				vertices[i].template get<2>() = depth;
				if constexpr( HasData ) vertices[i].template get<3>() = dataValue;
				vertexIndices[i] = (node_index_type)vertexStream.write( 0 , vertices[i] );
			}
		}

		// Copy over the edge/face -> key tables
		for( unsigned int depth=fullDepth ; depth<=std::min< unsigned int >( tree._maxDepth , boundaryInfo.sliceTree.depth() ) ; depth++ )
		{
			SliceValues &sValues = sliceFunctor( depth );
			typename SliceValues::Scratch &sScratch = scratchFunctor( depth );
			const SliceSliceValues &ssValues = boundaryInfo.sliceValues[depth];

			auto CopyEdgeAndFaceInfo = [&]( node_index_type sliceIndex , node_index_type index )
			{
				if( index==-1 ) return;
				const TreeNode *node = tree._sNodes.treeNodes[ index ];
				const SliceTreeNode *sliceNode = boundaryInfo.sliceTree._sNodes.treeNodes[ sliceIndex ];

				const typename LevelSetExtraction::SliceCellIndexData< Dim >::template CellIndices<1> &eIndices = sValues.cellIndices.template indices<1>( node );
				const typename LevelSetExtraction::FullCellIndexData< Dim-1 >::template CellIndices<1> &sliceEIndices = ssValues.cellIndices.template indices< 1 >( sliceNode );
				for( typename HyperCube::Cube< Dim-1 >::template Element< 1 > _e ; _e<HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() ; _e++ )
				{
					node_index_type eIndex = eIndices[_e.index];
					node_index_type sliceEIndex = sliceEIndices[_e.index];
					if( ssValues.edgeKeys[sliceEIndex].idx[0]!=-1 )
					{
						sValues.edgeKeys[eIndex] = keyGenerator( maxDepth , sliceAtMaxDepth , ssValues.edgeKeys[sliceEIndex] );
						sScratch.eSet[eIndex] = 1;
					}
					else sScratch.eSet[eIndex] = 0;
				}

				const typename LevelSetExtraction::SliceCellIndexData< Dim >::template CellIndices<2> &fIndices = sValues.cellIndices.template indices<2>( node );
				const typename LevelSetExtraction::FullCellIndexData< Dim-1 >::template CellIndices< 2 > &sliceFIndices = ssValues.cellIndices.template indices<2>( sliceNode );
				for( typename HyperCube::Cube< Dim-1 >::template Element< 2 > _f ; _f<HyperCube::Cube< Dim-1 >::template ElementNum< 2 >() ; _f++ )
				{
					node_index_type fIndex = fIndices[_f.index];
					node_index_type sliceFIndex = sliceFIndices[_f.index];
					sValues.faceEdges[fIndex].count = ssValues.faceEdges[sliceFIndex].count;
					for( int c=0 ; c<ssValues.faceEdges[sliceFIndex].count ; c++ ) sValues.faceEdges[fIndex].edges[c] = PromoteIsoEdge( ssValues.faceEdges[sliceFIndex].edges[c] );
					sScratch.fSet[fIndex] = sValues.faceEdges[fIndex].count!=-1;
				}
			};
			for( node_index_type i=boundaryInfo.sliceTree.nodesBegin(depth) ; i<boundaryInfo.sliceTree.nodesEnd(depth) ; i++ ) CopyEdgeAndFaceInfo( i , incidence[i].first ) , CopyEdgeAndFaceInfo( i , incidence[i].second );
		}

		// Copy over the iso-information
		for( unsigned int depth=fullDepth ; depth<=std::min< unsigned int >( tree._maxDepth , boundaryInfo.sliceTree.depth() ) ; depth++ )
		{
			SliceValues &sValues = sliceFunctor( depth );
			const SliceSliceValues &ssValues = boundaryInfo.sliceValues[depth];

			// Copy the face->edge map
			for( auto iter=ssValues.faceEdgeMap.cbegin() ; iter!=ssValues.faceEdgeMap.cend() ; iter++ )
			{
				Key key = keyGenerator( maxDepth , sliceAtMaxDepth , iter->first );
				std::vector< IsoEdge > edges( iter->second.size() );
				for( unsigned int i=0 ; i<iter->second.size() ; i++ ) edges[i] = PromoteIsoEdge( iter->second[i] );
				sValues.faceEdgeMap[key] = edges;
			}

			// Copy the edge-vertex map
			for( auto iter=ssValues.edgeVertexMap.cbegin() ; iter!=ssValues.edgeVertexMap.cend() ; iter++ )
			{
				Key key = keyGenerator( maxDepth , sliceAtMaxDepth , iter->first );
				node_index_type idx = iter->second;
				sValues.edgeVertexMap[key] = std::pair< node_index_type , Vertex >( vertexIndices[idx] , vertices[idx] );
			}

			// Copy the vertex-vertex map
			for( auto iter=ssValues.vertexPairMap.cbegin() ; iter!=ssValues.vertexPairMap.cend() ; iter++ )
			{
				Key key = keyGenerator( maxDepth , sliceAtMaxDepth , iter->first );
				sValues.vertexPairMap[key] = keyGenerator( maxDepth , sliceAtMaxDepth , iter->second );
			}
		}
	}

	static void OverwriteCornerValues( const typename LevelSetExtractor< Real , Dim-1 , Point< Real , Dim-1 > >::TreeSliceValuesAndVertexPositions &boundaryInfo , const std::vector< Real > &dValues , const FEMTree< Dim , Real > &tree , LocalDepth depth , unsigned int sliceAtMaxDepth , unsigned int maxDepth , bool isBack , std::vector< SlabValues > &slabValues , const std::vector< std::pair< node_index_type , node_index_type > > &incidence )
	{
		using SliceTreeNode    = typename LevelSetExtractor< Real , Dim-1 , Point< Real , Dim-1 > >::TreeNode;
		using SliceSliceValues = typename LevelSetExtractor< Real , Dim-1 , Point< Real , Dim-1 > >::SliceValues;

		unsigned int slice = sliceAtMaxDepth>>( maxDepth - depth );
		if( !isBack && sliceAtMaxDepth!=( slice<<(maxDepth-depth ) ) ) slice++;
		if( !slabValues[depth].validSlice( slice ) ) MK_THROW( "Invalid slice: " , slice , " @ " , depth , " : " , slabValues[depth].sliceValues(slice).slice() );
		SliceValues &sValues = slabValues[depth].sliceValues( slice );
		const SliceSliceValues &ssValues = boundaryInfo.sliceValues[depth];

		if( sValues.cornerGradients && !ssValues.cornerGradients ) MK_THROW( "Epxected slice gradients" );

		auto CopyCornerInfo = [&]( node_index_type sliceIndex , node_index_type index )
		{

			if( index==-1 ) return;
			const TreeNode *node = tree._sNodes.treeNodes[ index ];
			const SliceTreeNode *sliceNode = boundaryInfo.sliceTree._sNodes.treeNodes[ sliceIndex ];

			const typename LevelSetExtraction::SliceCellIndexData< Dim >::template CellIndices< 0 > &cIndices = sValues.cellIndices.template indices<0>( node );
			const typename LevelSetExtraction::FullCellIndexData< Dim-1 >::template CellIndices< 0 > &sliceCIndices = ssValues.cellIndices.template indices< 0 >( sliceNode );
			for( typename HyperCube::Cube< Dim-1 >::template Element< 0 > _c ; _c<HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() ; _c++ )
			{
				node_index_type vIndex = cIndices[_c.index];
				node_index_type sliceVIndex = sliceCIndices[_c.index];
				sValues.cornerValues[vIndex] = ssValues.cornerValues[sliceVIndex];
				if( sValues.cornerGradients )
				{
					for( unsigned int i=0 ; i<Dim-1 ; i++ ) sValues.cornerGradients[vIndex][i] = ssValues.cornerGradients[sliceVIndex][i];
					sValues.cornerGradients[vIndex][Dim-1] = dValues[sliceVIndex];
				}
			}
		};
		for( node_index_type i=boundaryInfo.sliceTree.nodesBegin(depth) ; i<boundaryInfo.sliceTree.nodesEnd(depth) ; i++ ) CopyCornerInfo( i , incidence[i].first ) , CopyCornerInfo( i , incidence[i].second );
	}

	template< unsigned int ... FEMSigs >
	static void SetSliceCornerValuesAndMCIndices( const FEMTree< Dim , Real >& tree , ConstPointer( Real ) coefficients , ConstPointer( Real ) coarseCoefficients , Real isoValue , LocalDepth depth , LocalDepth fullDepth , int slice ,         std::vector< SlabValues >& slabValues , const _Evaluator< UIntPack< FEMSigs ... > , 1 >& evaluator )
	{
		if( slice>0          ) SetSliceCornerValuesAndMCIndices< FEMSigs ... >( tree , coefficients , coarseCoefficients , isoValue , depth , fullDepth , slice , HyperCube::FRONT , slabValues , evaluator );
		if( slice<(1<<depth) ) SetSliceCornerValuesAndMCIndices< FEMSigs ... >( tree , coefficients , coarseCoefficients , isoValue , depth , fullDepth , slice , HyperCube::BACK  , slabValues , evaluator );
	}

	template< unsigned int ... FEMSigs >
	static void SetSliceCornerValuesAndMCIndices( const FEMTree< Dim , Real >& tree , ConstPointer( Real ) coefficients , ConstPointer( Real ) coarseCoefficients , Real isoValue , LocalDepth depth , LocalDepth fullDepth , int slice , HyperCube::Direction zDir , std::vector< SlabValues >& slabValues , const _Evaluator< UIntPack< FEMSigs ... > , 1 >& evaluator )
	{
		static const unsigned int FEMDegrees[] = { FEMSignature< FEMSigs >::Degree ... };
		SliceValues& sValues = slabValues[depth].sliceValues( slice );
		typename SliceValues::Scratch &sScratch = slabValues[depth].sliceScratch( slice );
		bool useBoundaryEvaluation = false;
		for( int d=0 ; d<Dim ; d++ ) if( FEMDegrees[d]==0 || ( FEMDegrees[d]==1 && sValues.cornerGradients ) ) useBoundaryEvaluation = true;
		std::vector< ConstPointSupportKey< UIntPack< FEMSignature< FEMSigs >::Degree ... > > > neighborKeys( ThreadPool::NumThreads() );
		std::vector< ConstCornerSupportKey< UIntPack< FEMSignature< FEMSigs >::Degree ... > > > bNeighborKeys( ThreadPool::NumThreads() );
		if( useBoundaryEvaluation ) for( size_t i=0 ; i<neighborKeys.size() ; i++ ) bNeighborKeys[i].set( tree._localToGlobal( depth ) );
		else                        for( size_t i=0 ; i<neighborKeys.size() ; i++ )  neighborKeys[i].set( tree._localToGlobal( depth ) );
		ThreadPool::ParallelFor( tree._sNodesBegin(depth,slice-(zDir==HyperCube::BACK ? 0 : 1)) , tree._sNodesEnd(depth,slice-(zDir==HyperCube::BACK ? 0 : 1)) , [&]( unsigned int thread , size_t i )
		{
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) )
			{
				Real squareValues[ HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() ];
				ConstPointSupportKey< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& neighborKey = neighborKeys[ thread ];
				ConstCornerSupportKey< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& bNeighborKey = bNeighborKeys[ thread ];
				TreeNode* leaf = tree._sNodes.treeNodes[i];
				if( !IsActiveNode< Dim >( leaf->children ) )
				{
					const typename LevelSetExtraction::SliceCellIndexData< Dim >::template CellIndices<0> &cIndices = sValues.cellIndices.template indices<0>( leaf );

					bool isInterior = tree._isInteriorlySupported( UIntPack< FEMSignature< FEMSigs >::Degree ... >() , leaf->parent );
					if( useBoundaryEvaluation ) bNeighborKey.getNeighbors( leaf );
					else                         neighborKey.getNeighbors( leaf );

					for( typename HyperCube::Cube< Dim-1 >::template Element< 0 > _c ; _c<HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() ; _c++ )
					{
						typename HyperCube::Cube< Dim >::template Element< 0 > c( zDir , _c.index );
#ifdef SANITIZED_PR
						node_index_type vIndex = ReadAtomic( cIndices[_c.index] );
#else // !SANITIZED_PR
						node_index_type vIndex = cIndices[_c.index];
#endif // SANITIZED_PR
						if( !sScratch.cSet[vIndex] )
						{
							if( sValues.cornerGradients )
							{
								CumulativeDerivativeValues< Real , Dim , 1 > p;
								if( useBoundaryEvaluation ) p = tree.template _getCornerValues< Real , 1 >( bNeighborKey , leaf , c.index , coefficients , coarseCoefficients , evaluator , tree._maxDepth , isInterior );
								else                        p = tree.template _getCornerValues< Real , 1 >(  neighborKey , leaf , c.index , coefficients , coarseCoefficients , evaluator , tree._maxDepth , isInterior );
#ifdef SANITIZED_PR
								SetAtomic( sValues.cornerValues[vIndex] , p[0] );
								SetAtomic( sValues.cornerGradients[vIndex] , Point< Real , Dim >( p[1] , p[2] , p[3] ) );
#else // !SANITIZED_PR
								sValues.cornerValues[vIndex] = p[0] , sValues.cornerGradients[vIndex] = Point< Real , Dim >( p[1] , p[2] , p[3] );
#endif // SANITIZED_PR
							}
							else
							{
#ifdef SANITIZED_PR
								if( useBoundaryEvaluation ) SetAtomic( sValues.cornerValues[vIndex] , tree.template _getCornerValues< Real , 0 >( bNeighborKey , leaf , c.index , coefficients , coarseCoefficients , evaluator , tree._maxDepth , isInterior )[0] );
								else                        SetAtomic( sValues.cornerValues[vIndex] , tree.template _getCornerValues< Real , 0 >(  neighborKey , leaf , c.index , coefficients , coarseCoefficients , evaluator , tree._maxDepth , isInterior )[0] );
#else // !SANITIZED_PR
								if( useBoundaryEvaluation ) sValues.cornerValues[vIndex] = tree.template _getCornerValues< Real , 0 >( bNeighborKey , leaf , c.index , coefficients , coarseCoefficients , evaluator , tree._maxDepth , isInterior )[0];
								else                        sValues.cornerValues[vIndex] = tree.template _getCornerValues< Real , 0 >(  neighborKey , leaf , c.index , coefficients , coarseCoefficients , evaluator , tree._maxDepth , isInterior )[0];
#endif // SANITIZED_PR
							}
							sScratch.cSet[vIndex] = 1;
						}
#ifdef SANITIZED_PR
						squareValues[_c.index] = ReadAtomic( sValues.cornerValues[ vIndex ] );
#else // !SANITIZED_PR
						squareValues[_c.index] = sValues.cornerValues[ vIndex ];
#endif // SANITIZED_PR
						TreeNode* node = leaf;
						LocalDepth _depth = depth;
						int _slice = slice;
						while( _depth>fullDepth && tree._isValidSpaceNode( node->parent ) && (node-node->parent->children)==c.index )
						{
							node = node->parent , _depth-- , _slice >>= 1;
							SliceValues &_sValues = slabValues[_depth].sliceValues( _slice );
							typename SliceValues::Scratch &_sScratch = slabValues[_depth].sliceScratch( _slice );
							const typename LevelSetExtraction::SliceCellIndexData< Dim >::template CellIndices<0> &_cIndices = _sValues.cellIndices.template indices<0>( node );
							node_index_type _vIndex = _cIndices[_c.index];
#ifdef SANITIZED_PR
							SetAtomic( _sValues.cornerValues[_vIndex] , ReadAtomic( sValues.cornerValues[vIndex] ) );
							if( _sValues.cornerGradients ) SetAtomic( _sValues.cornerGradients[_vIndex] , ReadAtomic( sValues.cornerGradients[vIndex] ) );
#else // !SANITIZED_PR
							_sValues.cornerValues[_vIndex] = sValues.cornerValues[vIndex];
							if( _sValues.cornerGradients ) _sValues.cornerGradients[_vIndex] = sValues.cornerGradients[vIndex];
#endif // SANITIZED_PR
							_sScratch.cSet[_vIndex] = 1;
						}
					}
					sValues.mcIndices[ i - sValues.cellIndices.nodeOffset ] = HyperCube::Cube< Dim-1 >::MCIndex( squareValues , isoValue );
				}
			}
		}
		);
	}

	static void SetMCIndices( const FEMTree< Dim , Real >& tree , Real isoValue , LocalDepth depth , LocalDepth fullDepth , int slice , std::vector< SlabValues >& slabValues )
	{
		if( slice>0          ) SetMCIndices( tree , isoValue , depth , fullDepth , slice , HyperCube::FRONT , slabValues );
		if( slice<(1<<depth) ) SetMCIndices( tree , isoValue , depth , fullDepth , slice , HyperCube::BACK  , slabValues );
	}

	static void SetMCIndices( const FEMTree< Dim , Real >& tree , Real isoValue , LocalDepth depth , LocalDepth fullDepth , int slice , HyperCube::Direction zDir , std::vector< SlabValues >& slabValues )
	{
		SliceValues& sValues = slabValues[depth].sliceValues( slice );
		bool useBoundaryEvaluation = false;
		ThreadPool::ParallelFor( tree._sNodesBegin(depth,slice-(zDir==HyperCube::BACK ? 0 : 1)) , tree._sNodesEnd(depth,slice-(zDir==HyperCube::BACK ? 0 : 1)) , [&]( unsigned int thread , size_t i )
			{
				Real squareValues[ HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() ];
				TreeNode* leaf = tree._sNodes.treeNodes[i];

				if( tree._isValidSpaceNode( leaf ) && !IsActiveNode< Dim >( leaf->children ) )
				{
					const typename LevelSetExtraction::SliceCellIndexData< Dim >::template CellIndices<0> &cIndices = sValues.cellIndices.template indices<0>( leaf );

					for( typename HyperCube::Cube< Dim-1 >::template Element< 0 > _c ; _c<HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() ; _c++ )
					{
						typename HyperCube::Cube< Dim >::template Element< 0 > c( zDir , _c.index );
						node_index_type vIndex = cIndices[_c.index];
						squareValues[_c.index] = sValues.cornerValues[ vIndex ];
						sValues.mcIndices[ i - sValues.cellIndices.nodeOffset ] = HyperCube::Cube< Dim-1 >::MCIndex( squareValues , isoValue );
					}
				}
			}
		);
	}

	template< unsigned int WeightDegree , unsigned int DataSig , typename VertexStream >
	static void SetSliceIsoVertices( const LevelSetExtraction::KeyGenerator< Dim > &keyGenerator , const FEMTree< Dim , Real >& tree , bool nonLinearFit , bool gradientNormals , typename FEMIntegrator::template PointEvaluator< IsotropicUIntPack< Dim , DataSig > , ZeroUIntPack< Dim > >* pointEvaluator , const DensityEstimator< WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > >* data , Real isoValue , LocalDepth depth , LocalDepth fullDepth , int slice , VertexStream &vertexStream , std::vector< SlabValues >& slabValues , const Data &zeroData )
	{
		if( slice>0          ) SetSliceIsoVertices< WeightDegree , DataSig >( keyGenerator , tree , nonLinearFit , gradientNormals , pointEvaluator , densityWeights , data , isoValue , depth , fullDepth , slice , HyperCube::FRONT , vertexStream , slabValues , zeroData );
		if( slice<(1<<depth) ) SetSliceIsoVertices< WeightDegree , DataSig >( keyGenerator , tree , nonLinearFit , gradientNormals , pointEvaluator , densityWeights , data , isoValue , depth , fullDepth , slice , HyperCube::BACK  , vertexStream , slabValues , zeroData );
	}

	template< unsigned int WeightDegree , unsigned int DataSig , typename VertexStream >
	static void SetSliceIsoVertices( const LevelSetExtraction::KeyGenerator< Dim > &keyGenerator , const FEMTree< Dim , Real >& tree , bool nonLinearFit , bool gradientNormals , typename FEMIntegrator::template PointEvaluator< IsotropicUIntPack< Dim , DataSig > , ZeroUIntPack< Dim > >* pointEvaluator , const DensityEstimator< WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > >* data , Real isoValue , LocalDepth depth , LocalDepth fullDepth , int slice , HyperCube::Direction zDir , VertexStream &vertexStream , std::vector< SlabValues >& slabValues , const Data &zeroData )
	{
		auto _EdgeIndex = [&]( const TreeNode *node , typename HyperCube::Cube< Dim >::template Element< 1 > e )
		{
			int depth , offset[Dim];
			tree.depthAndOffset( node , depth , offset );
			return keyGenerator( depth , offset , e );
		};

		static const unsigned int DataDegree = FEMSignature< DataSig >::Degree;
		SliceValues &sValues = slabValues[depth].sliceValues( slice );
		typename SliceValues::Scratch &sScratch = slabValues[depth].sliceScratch( slice );
		// [WARNING] In the case Degree=2, these two keys are the same, so we don't have to maintain them separately.
		std::vector< ConstOneRingNeighborKey > neighborKeys( ThreadPool::NumThreads() );
		std::vector< ConstPointSupportKey< IsotropicUIntPack< Dim , WeightDegree > > > weightKeys( ThreadPool::NumThreads() );
		std::vector< ConstPointSupportKey< IsotropicUIntPack< Dim , DataDegree > > > dataKeys( ThreadPool::NumThreads() );
		for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( tree._localToGlobal( depth ) ) , weightKeys[i].set( tree._localToGlobal( depth ) ) , dataKeys[i].set( tree._localToGlobal( depth ) );
		ThreadPool::ParallelFor( tree._sNodesBegin(depth,slice-(zDir==HyperCube::BACK ? 0 : 1)) , tree._sNodesEnd(depth,slice-(zDir==HyperCube::BACK ? 0 : 1)) , [&]( unsigned int thread , size_t i )
		{
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) )
			{
				ConstOneRingNeighborKey& neighborKey =  neighborKeys[ thread ];
				ConstPointSupportKey< IsotropicUIntPack< Dim , WeightDegree > >& weightKey = weightKeys[ thread ];
				ConstPointSupportKey< IsotropicUIntPack< Dim , DataDegree > >& dataKey = dataKeys[ thread ];
				TreeNode* leaf = tree._sNodes.treeNodes[i];
				if( !IsActiveNode< Dim >( leaf->children ) )
				{
					node_index_type idx = (node_index_type)i - sValues.cellIndices.nodeOffset;
					const typename LevelSetExtraction::SliceCellIndexData< Dim >::template CellIndices<1> &eIndices = sValues.cellIndices.template indices<1>( leaf );
					if( HyperCube::Cube< Dim-1 >::HasMCRoots( sValues.mcIndices[idx] ) )
					{
						neighborKey.getNeighbors( leaf );
						if( densityWeights ) weightKey.getNeighbors( leaf );
						if constexpr( HasData ) if( data ) dataKey.getNeighbors( leaf );

						for( typename HyperCube::Cube< Dim-1 >::template Element< 1 > _e ; _e<HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() ; _e++ )
							if( HyperCube::Cube< 1 >::HasMCRoots( HyperCube::Cube< Dim-1 >::ElementMCIndex( _e , sValues.mcIndices[idx] ) ) )
							{
								typename HyperCube::Cube< Dim >::template Element< 1 > e( zDir , _e.index );
								node_index_type vIndex = eIndices[_e.index];
#ifdef SANITIZED_PR
								std::atomic< char > &edgeSet = sScratch.eSet[vIndex];
#else // !SANITIZED_PR
								volatile char &edgeSet = sScratch.eSet[vIndex];
#endif // SANITIZED_PR
								if( !edgeSet )
								{
									Vertex vertex;
									Key key = _EdgeIndex( leaf , e );
									GetIsoVertex< WeightDegree , DataSig >( tree , nonLinearFit , gradientNormals , pointEvaluator , densityWeights , data , isoValue , weightKey , dataKey , leaf , _e , zDir , sValues , vertex , zeroData );
									bool stillOwner = false;
									std::pair< node_index_type , Vertex > hashed_vertex;

									{
										char desired = 1 , expected = 0;
#ifdef SANITIZED_PR
										if( edgeSet.compare_exchange_weak( expected , desired ) )
#else // !SANITIZED_PR
										if( SetAtomic( edgeSet , desired , expected ) )
#endif // SANITIZED_PR
										{
											hashed_vertex = std::pair< node_index_type , Vertex >( (node_index_type)vertexStream.write( thread , vertex ) , vertex );
											stillOwner = true;
										}
									}

									if( stillOwner )
									{
										sValues.edgeKeys[ vIndex ] = key;
										sScratch.eKeyValues[ thread ].push_back( std::pair< Key , std::pair< node_index_type , Vertex > >( key , hashed_vertex ) );
										// We only need to pass the iso-vertex down if the edge it lies on is adjacent to a coarser leaf
										auto IsNeeded = [&]( unsigned int depth )
										{
											bool isNeeded = false;
											typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 1 > my_ic = HyperCubeTables< Dim , 1 >::IncidentCube[e.index];
											for( typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 1 > ic ; ic<HyperCube::Cube< Dim >::template IncidentCubeNum< 1 >() ; ic++ ) if( ic!=my_ic )
											{
												unsigned int xx = HyperCubeTables< Dim , 1 >::CellOffset[e.index][ic.index];
												isNeeded |= !tree._isValidSpaceNode( neighborKey.neighbors[ tree._localToGlobal( depth ) ].neighbors.data[xx] );
											}
											return isNeeded;
										};
										if( IsNeeded( depth ) )
										{
											const typename HyperCube::Cube< Dim >::template Element< Dim-1 > *f = HyperCubeTables< Dim , 1 , Dim-1 >::OverlapElements[e.index];
											for( int k=0 ; k<HyperCubeTables< Dim , 1 , Dim-1 >::OverlapElementNum ; k++ )
											{
												TreeNode* node = leaf;
												LocalDepth _depth = depth;
												int _slice = slice;
												bool _cross = false;
												while( tree._isValidSpaceNode( node->parent ) && HyperCubeTables< Dim , Dim-1 , 0 >::Overlap[f[k].index][(unsigned int)(node-node->parent->children) ] )
												{
													if( _slice&1 ) _cross = true;
													node = node->parent , _depth-- , _slice >>= 1;
													SliceValues &_sValues = slabValues[_depth].sliceValues( _slice );
													typename SliceValues::Scratch &_sScratch = slabValues[_depth].sliceScratch( _slice );

													if( _depth>=fullDepth && !_cross )
													{
														const typename LevelSetExtraction::SliceCellIndexData< Dim >::template CellIndices<1> &_eIndices = _sValues.cellIndices.template indices<1>( node );
														node_index_type _vIndex = _eIndices[_e.index];
														_sScratch.eSet[_vIndex] = 1;
													}

													if( _cross )
													{
														XSliceValues& _xValues = slabValues[_depth].xSliceValues( _slice );
														typename XSliceValues::Scratch &_xScratch = slabValues[_depth].xSliceScratch( _slice );
														_xScratch.eKeyValues[ thread ].push_back( std::pair< Key , std::pair< node_index_type , Vertex > >( key , hashed_vertex ) );
													}
													else
													{
														SliceValues& _sValues = slabValues[_depth].sliceValues( _slice );
														typename SliceValues::Scratch &_sScratch = slabValues[_depth].sliceScratch( _slice );
														_sScratch.eKeyValues[ thread ].push_back( std::pair< Key , std::pair< node_index_type , Vertex > >( key , hashed_vertex ) );
													}
													if( !IsNeeded( _depth ) ) break;
												}
											}
										}
									}
								}
							}
					}
				}
			}
		}
		);
	}

	////////////////////
	// Iso-Extraction //
	////////////////////
	template< unsigned int WeightDegree , unsigned int DataSig , typename VertexStream >
	static void SetXSliceIsoVertices( const LevelSetExtraction::KeyGenerator< Dim >  &keyGenerator , const FEMTree< Dim , Real >& tree , bool nonLinearFit , bool gradientNormals , typename FEMIntegrator::template PointEvaluator< IsotropicUIntPack< Dim , DataSig > , ZeroUIntPack< Dim > >* pointEvaluator , const DensityEstimator< WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > >* data , Real isoValue , LocalDepth depth , LocalDepth fullDepth , int slab , Real bCoordinate , Real fCoordinate , VertexStream &vertexStream , std::vector< SlabValues >& slabValues , const Data &zeroData )
	{
		auto _EdgeIndex = [&]( const TreeNode *node , typename HyperCube::Cube< Dim >::template Element< 1 > e )
		{
			int depth , offset[Dim];
			tree.depthAndOffset( node , depth , offset );
			return keyGenerator( depth , offset , e );
		};

		static const unsigned int DataDegree = FEMSignature< DataSig >::Degree;
		SliceValues& bValues = slabValues[depth].sliceValues ( slab   );
		SliceValues& fValues = slabValues[depth].sliceValues ( slab+1 );
		XSliceValues& xValues = slabValues[depth].xSliceValues( slab   );
		typename  SliceValues::Scratch &bScratch = slabValues[depth]. sliceScratch( slab   );
		typename  SliceValues::Scratch &fScratch = slabValues[depth]. sliceScratch( slab+1 );
		typename XSliceValues::Scratch &xScratch = slabValues[depth].xSliceScratch( slab   );

		// [WARNING] In the case Degree=2, these two keys are the same, so we don't have to maintain them separately.
		std::vector< ConstOneRingNeighborKey > neighborKeys( ThreadPool::NumThreads() );
		std::vector< ConstPointSupportKey< IsotropicUIntPack< Dim , WeightDegree > > > weightKeys( ThreadPool::NumThreads() );
		std::vector< ConstPointSupportKey< IsotropicUIntPack< Dim , DataDegree > > > dataKeys( ThreadPool::NumThreads() );
		for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( tree._localToGlobal( depth ) ) , weightKeys[i].set( tree._localToGlobal( depth ) ) , dataKeys[i].set( tree._localToGlobal( depth ) );
		ThreadPool::ParallelFor( tree._sNodesBegin(depth,slab) , tree._sNodesEnd(depth,slab) , [&]( unsigned int thread , size_t i )
		{
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) )
			{
				ConstOneRingNeighborKey& neighborKey =  neighborKeys[ thread ];
				ConstPointSupportKey< IsotropicUIntPack< Dim , WeightDegree > >& weightKey = weightKeys[ thread ];
				ConstPointSupportKey< IsotropicUIntPack< Dim , DataDegree > >& dataKey = dataKeys[ thread ];
				TreeNode* leaf = tree._sNodes.treeNodes[i];
				if( !IsActiveNode< Dim >( leaf->children ) )
				{
					unsigned char mcIndex = ( bValues.mcIndices[ i - bValues.cellIndices.nodeOffset ] ) | ( fValues.mcIndices[ i - fValues.cellIndices.nodeOffset ] )<<4;
					const typename LevelSetExtraction::SlabCellIndexData< Dim >::template CellIndices<0> &eIndices = xValues.cellIndices.template indices<0>( leaf );
					if( HyperCube::Cube< Dim >::HasMCRoots( mcIndex ) )
					{
						neighborKey.getNeighbors( leaf );
						if( densityWeights ) weightKey.getNeighbors( leaf );
						if constexpr( HasData ) if( data ) dataKey.getNeighbors( leaf );
						for( typename HyperCube::Cube< Dim-1 >::template Element< 0 > _c ; _c<HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() ; _c++ )
						{
							typename HyperCube::Cube< Dim >::template Element< 1 > e( HyperCube::CROSS , _c.index );
							unsigned int _mcIndex = HyperCube::Cube< Dim >::ElementMCIndex( e , mcIndex );
							if( HyperCube::Cube< 1 >::HasMCRoots( _mcIndex ) )
							{
								node_index_type vIndex = eIndices[_c.index];
#ifdef SANITIZED_PR
								std::atomic< char > &edgeSet = xScratch.eSet[vIndex];
#else // !SANITIZED_PR
								volatile char &edgeSet = xScratch.eSet[vIndex];
#endif // SANITIZED_PR
								if( !edgeSet )
								{
									Vertex vertex;
									Key key = _EdgeIndex( leaf , e.index );
									GetIsoVertex< WeightDegree , DataSig >( tree , nonLinearFit , gradientNormals , pointEvaluator , densityWeights , data , isoValue , weightKey , dataKey , leaf , _c , bCoordinate , fCoordinate , bValues , fValues , vertex , zeroData );
									bool stillOwner = false;
									std::pair< node_index_type , Vertex > hashed_vertex;

									{
										char desired = 1 , expected = 0;
#ifdef SANITIZED_PR
										if( edgeSet.compare_exchange_weak( expected , desired ) )
#else // !SANITIZED_PR
										if( SetAtomic( edgeSet , desired , expected ) )
#endif // SANITIZED_PR
										{
											hashed_vertex = std::pair< node_index_type , Vertex >( (node_index_type)vertexStream.write( thread , vertex ) , vertex );
											stillOwner = true;
										}
									}

									if( stillOwner )
									{
										xValues.edgeKeys[ vIndex ] = key;
										xScratch.eKeyValues[ thread ].push_back( std::pair< Key , std::pair< node_index_type , Vertex > >( key , hashed_vertex ) );

										// We only need to pass the iso-vertex down if the edge it lies on is adjacent to a coarser leaf
										auto IsNeeded = [&]( unsigned int depth )
										{
											bool isNeeded = false;
											typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 1 > my_ic = HyperCubeTables< Dim , 1 >::IncidentCube[e.index];
											for( typename HyperCube::Cube< Dim >::template IncidentCubeIndex< 1 > ic ; ic<HyperCube::Cube< Dim >::template IncidentCubeNum< 1 >() ; ic++ ) if( ic!=my_ic )
											{
												unsigned int xx = HyperCubeTables< Dim , 1 >::CellOffset[e.index][ic.index];
												isNeeded |= !tree._isValidSpaceNode( neighborKey.neighbors[ tree._localToGlobal( depth ) ].neighbors.data[xx] );
											}
											return isNeeded;
										};
										if( IsNeeded( depth ) )
										{
											const typename HyperCube::Cube< Dim >::template Element< Dim-1 > *f = HyperCubeTables< Dim , 1 , Dim-1 >::OverlapElements[e.index];
											for( int k=0 ; k<2 ; k++ )
											{
												TreeNode* node = leaf;
												LocalDepth _depth = depth;
												int _slab = slab;
												// As long as we are still in the tree and the parent is also adjacent to the node
												while( tree._isValidSpaceNode( node->parent ) && HyperCubeTables< Dim , Dim-1 , 0 >::Overlap[f[k].index][(unsigned int)(node-node->parent->children) ] )
												{
													node = node->parent , _depth-- , _slab >>= 1;
													XSliceValues& _xValues = slabValues[_depth].xSliceValues( _slab );
													typename XSliceValues::Scratch &_xScratch = slabValues[_depth].xSliceScratch( _slab );
													_xScratch.eKeyValues[ thread ].push_back( std::pair< Key , std::pair< node_index_type , Vertex > >( key , hashed_vertex ) );

													if( _depth>=fullDepth )
													{
														const typename LevelSetExtraction::SlabCellIndexData< Dim >::template CellIndices<0> &_eIndices = _xValues.cellIndices.template indices<0>( node );
														node_index_type _vIndex = _eIndices[_c.index];
														_xScratch.eSet[_vIndex] = 1;
													}

													if( !IsNeeded( _depth ) ) break;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		);
	}

	static void CopyFinerSliceIsoEdgeKeys( const FEMTree< Dim , Real >& tree , LocalDepth depth , LocalDepth fullDepth , int slice , std::vector< SlabValues >& slabValues )
	{
		if( slice>0          ) CopyFinerSliceIsoEdgeKeys( tree , depth , fullDepth , slice , HyperCube::FRONT , slabValues );
		if( slice<(1<<depth) ) CopyFinerSliceIsoEdgeKeys( tree , depth , fullDepth , slice , HyperCube::BACK  , slabValues );
	}

	static void CopyFinerSliceIsoEdgeKeys( const FEMTree< Dim , Real >& tree , LocalDepth depth , LocalDepth fullDepth , int slice , HyperCube::Direction zDir , std::vector< SlabValues >& slabValues )
	{
		SliceValues& pSliceValues = slabValues[depth  ].sliceValues(slice   );
		SliceValues& cSliceValues = slabValues[depth+1].sliceValues(slice<<1);
		typename SliceValues::Scratch &pSliceScratch = slabValues[depth  ].sliceScratch(slice   );
		typename SliceValues::Scratch &cSliceScratch = slabValues[depth+1].sliceScratch(slice<<1);
		LevelSetExtraction::SliceCellIndexData< Dim > &pCellIndices = pSliceValues.cellIndices;
		LevelSetExtraction::SliceCellIndexData< Dim > &cCellIndices = cSliceValues.cellIndices;
		ThreadPool::ParallelFor( tree._sNodesBegin(depth,slice-(zDir==HyperCube::BACK ? 0 : 1)) , tree._sNodesEnd(depth,slice-(zDir==HyperCube::BACK ? 0 : 1)) , [&]( unsigned int thread , size_t i )
		{
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) ) if( IsActiveNode< Dim >( tree._sNodes.treeNodes[i]->children ) )
			{
				typename LevelSetExtraction::SliceCellIndexData< Dim >::template CellIndices<1> &pIndices = pCellIndices.template indices<1>( (node_index_type)i );
				// Copy the edges that overlap the coarser edges
				for( typename HyperCube::Cube< Dim-1 >::template Element< 1 > _e ; _e<HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() ; _e++ )
				{
					node_index_type pIndex = pIndices[_e.index];
					{
						typename HyperCube::Cube< Dim >::template Element< 1 > e( zDir , _e.index );
						const typename HyperCube::Cube< Dim >::template Element< 0 > *c = HyperCubeTables< Dim , 1 , 0 >::OverlapElements[e.index];
						// [SANITY CHECK]
						//						if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i]->children + c[0].index )!=tree._isValidSpaceNode( tree._sNodes.treeNodes[i]->children + c[1].index ) ) MK_THROW( "Finer edges should both be valid or invalid" );
						if( !tree._isValidSpaceNode( tree._sNodes.treeNodes[i]->children + c[0].index ) || !tree._isValidSpaceNode( tree._sNodes.treeNodes[i]->children + c[1].index ) ) continue;

						node_index_type cIndex1 = cCellIndices.template indices<1>( tree._sNodes.treeNodes[i]->children + c[0].index )[_e.index];
						node_index_type cIndex2 = cCellIndices.template indices<1>( tree._sNodes.treeNodes[i]->children + c[1].index )[_e.index];
						if( cSliceScratch.eSet[cIndex1] != cSliceScratch.eSet[cIndex2] )
						{
							Key key;
							if( cSliceScratch.eSet[cIndex1] ) key = cSliceValues.edgeKeys[cIndex1];
							else                              key = cSliceValues.edgeKeys[cIndex2];
#ifdef SANITIZED_PR
							SetAtomic( pSliceValues.edgeKeys[pIndex] , key );
#else // !SANITIZED_PR
							pSliceValues.edgeKeys[pIndex] = key;
#endif // SANITIZED_PR
							pSliceScratch.eSet[pIndex] = 1;
						}
						else if( cSliceScratch.eSet[cIndex1] && cSliceScratch.eSet[cIndex2] )
						{
							Key key1 = cSliceValues.edgeKeys[cIndex1] , key2 = cSliceValues.edgeKeys[cIndex2];
							pSliceScratch.vKeyValues[ thread ].push_back( std::pair< Key , Key >( key1 , key2 ) );

							const TreeNode* node = tree._sNodes.treeNodes[i];
							LocalDepth _depth = depth;
							int _slice = slice;
							while( _depth>fullDepth && tree._isValidSpaceNode( node->parent ) && HyperCubeTables< Dim , 1 , 0 >::Overlap[e.index][(unsigned int)(node-node->parent->children) ] )
							{
								node = node->parent , _depth-- , _slice >>= 1;
								SliceValues& _pSliceValues = slabValues[_depth].sliceValues(_slice);
								typename SliceValues::Scratch &_pSliceScratch = slabValues[_depth].sliceScratch(_slice);
								_pSliceScratch.vKeyValues[ thread ].push_back( std::pair< Key , Key >( key1 , key2 ) );
							}
						}
					}
				}
			}
		}
		);
	}

	static void CopyFinerXSliceIsoEdgeKeys( const FEMTree< Dim , Real >& tree , LocalDepth depth , LocalDepth fullDepth , int slab , std::vector< SlabValues>& slabValues )
	{
		XSliceValues& pSliceValues  = slabValues[depth  ].xSliceValues(slab);
		XSliceValues& cSliceValues0 = slabValues[depth+1].xSliceValues( (slab<<1)|0 );
		XSliceValues& cSliceValues1 = slabValues[depth+1].xSliceValues( (slab<<1)|1 );
		typename XSliceValues::Scratch &pSliceScratch  = slabValues[depth  ].xSliceScratch(slab);
		typename XSliceValues::Scratch &cSliceScratch0 = slabValues[depth+1].xSliceScratch( (slab<<1)|0 );
		typename XSliceValues::Scratch &cSliceScratch1 = slabValues[depth+1].xSliceScratch( (slab<<1)|1 );
		LevelSetExtraction::SlabCellIndexData< Dim > &pCellIndices  = pSliceValues. cellIndices;
		LevelSetExtraction::SlabCellIndexData< Dim > &cCellIndices0 = cSliceValues0.cellIndices;
		LevelSetExtraction::SlabCellIndexData< Dim > &cCellIndices1 = cSliceValues1.cellIndices;

		bool has0 = cSliceValues0.slab()==((slab<<1)|0);
		bool has1 = cSliceValues1.slab()==((slab<<1)|1);

		ThreadPool::ParallelFor( tree._sNodesBegin(depth,slab) , tree._sNodesEnd(depth,slab) , [&]( unsigned int thread , size_t i )
		{
			// If the node is not a leaf, inherit iso-edges from children
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) &&  IsActiveNode< Dim >( tree._sNodes.treeNodes[i]->children ) )
			{
				// Get the mapping from node + local face-corner -> global corner
				typename LevelSetExtraction::SlabCellIndexData< Dim >::template CellIndices<0> &pIndices = pCellIndices.template indices<0>( (node_index_type)i );
				for( typename HyperCube::Cube< Dim-1 >::template Element< 0 > _c ; _c<HyperCube::Cube< Dim-1 >::template ElementNum< 0 >() ; _c++ )
				{
					// Transform the face-corner index to a cross-edge index
					typename HyperCube::Cube< Dim >::template Element< 1 > e( HyperCube::CROSS , _c.index );
					node_index_type pIndex = pIndices[ _c.index ];

					{
						typename HyperCube::Cube< Dim >::template Element< 0 > c0( HyperCube::BACK , _c.index ) , c1( HyperCube::FRONT , _c.index );

						// [SANITY CHECK]
						//					if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i]->children + c0 )!=tree._isValidSpaceNode( tree._sNodes.treeNodes[i]->children + c1 ) ) MK_THROW( "Finer edges should both be valid or invalid" );
						if( !tree._isValidSpaceNode( tree._sNodes.treeNodes[i]->children + c0.index ) || !tree._isValidSpaceNode( tree._sNodes.treeNodes[i]->children + c1.index ) ) continue;

						node_index_type cIndex0 , cIndex1;
						if( has0 ) cIndex0 = cCellIndices0.template indices<0>( tree._sNodes.treeNodes[i]->children + c0.index )[_c.index];
						if( has1 ) cIndex1 = cCellIndices1.template indices<0>( tree._sNodes.treeNodes[i]->children + c1.index )[_c.index];
						char eSet0 = has0 && cSliceScratch0.eSet[cIndex0] , eSet1 = has1 && cSliceScratch1.eSet[cIndex1];

						// If there's one zero-crossing along the edge
						if( eSet0 != eSet1 )
						{
							Key key;
							if( eSet0 ) key = cSliceValues0.edgeKeys[cIndex0]; //, vPair = cSliceValues0.edgeVertexMap.find( key )->second;
							else        key = cSliceValues1.edgeKeys[cIndex1]; //, vPair = cSliceValues1.edgeVertexMap.find( key )->second;
#ifdef SANITIZED_PR
							SetAtomic( pSliceValues.edgeKeys[ pIndex ] , key );
#else // !SANITIZED_PR
							pSliceValues.edgeKeys[ pIndex ] = key;
#endif // SANITIZED_PR
							pSliceScratch.eSet[ pIndex ] = 1;
						}
						// If there's are two zero-crossings along the edge
						else if( eSet0 && eSet1 )
						{
							Key key0 = cSliceValues0.edgeKeys[cIndex0] , key1 = cSliceValues1.edgeKeys[cIndex1];
							pSliceScratch.vKeyValues[ thread ].push_back( std::pair< Key , Key >( key0 , key1 ) );
							const TreeNode* node = tree._sNodes.treeNodes[i];
							LocalDepth _depth = depth;
							int _slab = slab;
							while( _depth>fullDepth && tree._isValidSpaceNode( node->parent ) && HyperCubeTables< Dim , 1 , 0 >::Overlap[e.index][(unsigned int)(node-node->parent->children) ] )
							{
								node = node->parent , _depth-- , _slab>>= 1;
								SliceValues& _pSliceValues = slabValues[_depth].sliceValues(_slab);
								typename SliceValues::Scratch &_pSliceScratch = slabValues[_depth].sliceScratch(_slab);
								_pSliceScratch.vKeyValues[ thread ].push_back( std::pair< Key , Key >( key0 , key1 ) );
							}
						}
					}
				}
			}
		}
		);
	}

	static void SetSliceIsoEdges( const LevelSetExtraction::KeyGenerator< Dim > &keyGenerator , const FEMTree< Dim , Real >& tree , LocalDepth depth , int slice , std::vector< SlabValues >& slabValues )
	{
		if( slice>0          ) SetSliceIsoEdges( keyGenerator , tree , depth , slice , HyperCube::FRONT , slabValues );
		if( slice<(1<<depth) ) SetSliceIsoEdges( keyGenerator , tree , depth , slice , HyperCube::BACK  , slabValues );
	}

	static void SetSliceIsoEdges( const LevelSetExtraction::KeyGenerator< Dim > &keyGenerator , const FEMTree< Dim , Real >& tree , LocalDepth depth , int slice , HyperCube::Direction zDir , std::vector< SlabValues >& slabValues )
	{
		auto _FaceIndex = [&]( const TreeNode *node , typename HyperCube::Cube< Dim >::template Element< 2 > f )
		{
			int depth , offset[Dim];
			tree.depthAndOffset( node , depth , offset );
			return keyGenerator( depth , offset , f );
		};

		SliceValues& sValues = slabValues[depth].sliceValues( slice );
		typename SliceValues::Scratch &sScratch = slabValues[depth].sliceScratch( slice );
		std::vector< ConstOneRingNeighborKey > neighborKeys( ThreadPool::NumThreads() );
		for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( tree._localToGlobal( depth ) );
		ThreadPool::ParallelFor( tree._sNodesBegin(depth, slice-(zDir==HyperCube::BACK ? 0 : 1)) , tree._sNodesEnd(depth,slice-(zDir==HyperCube::BACK ? 0 : 1)) , [&]( unsigned int thread , size_t i )
		{
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) )
			{
				int isoEdges[ 2 * HyperCube::MarchingSquares::MAX_EDGES ];
				ConstOneRingNeighborKey& neighborKey = neighborKeys[ thread ];
				TreeNode* leaf = tree._sNodes.treeNodes[i];
				if( !IsActiveNode< Dim >( leaf->children ) )
				{
					node_index_type idx = (node_index_type)i - sValues.cellIndices.nodeOffset;
					const typename LevelSetExtraction::SliceCellIndexData< Dim >::template CellIndices<1> &eIndices = sValues.cellIndices.template indices<1>( leaf );
					const typename LevelSetExtraction::SliceCellIndexData< Dim >::template CellIndices<2> &fIndices = sValues.cellIndices.template indices<2>( leaf );
					unsigned char mcIndex = sValues.mcIndices[idx];
					if( !sScratch.fSet[ fIndices[0] ] )
					{
						neighborKey.getNeighbors( leaf );
						unsigned int xx = WindowIndex< IsotropicUIntPack< Dim , 3 > , IsotropicUIntPack< Dim , 1 > >::Index + (zDir==HyperCube::BACK ? -1 : 1);
						if( !IsActiveNode< Dim >( neighborKey.neighbors[ tree._localToGlobal( depth ) ].neighbors.data[xx] ) || !IsActiveNode< Dim >( neighborKey.neighbors[ tree._localToGlobal( depth ) ].neighbors.data[xx]->children ) )
						{
							FaceEdges fe;
							fe.count = HyperCube::MarchingSquares::AddEdgeIndices( mcIndex , isoEdges );
							for( int j=0 ; j<fe.count ; j++ ) for( int k=0 ; k<2 ; k++ )
							{
								if( !sScratch.eSet[ eIndices[ isoEdges[2*j+k] ] ] ) MK_THROW( "Edge not set: " , slice-(zDir==HyperCube::BACK ? 0 : 1) , " / " , 1<<depth );
								fe.edges[j][k] = sValues.edgeKeys[ eIndices[ isoEdges[2*j+k] ] ];
							}
							sScratch.fSet[ fIndices[0] ] = 1;
							sValues.faceEdges[ fIndices[0] ] = fe;

							TreeNode* node = leaf;
							LocalDepth _depth = depth;
							int _slice = slice;
							typename HyperCube::Cube< Dim >::template Element< Dim-1 > f( zDir , 0 );
							std::vector< IsoEdge > edges;
							edges.resize( fe.count );
							for( int j=0 ; j<fe.count ; j++ ) edges[j] = fe.edges[j];
							while( tree._isValidSpaceNode( node->parent ) && HyperCubeTables< Dim , 2 , 0 >::Overlap[f.index][(unsigned int)(node-node->parent->children) ] )
							{
								node = node->parent , _depth-- , _slice >>= 1;
								if( IsActiveNode< Dim >( neighborKey.neighbors[ tree._localToGlobal( _depth ) ].neighbors.data[xx] ) && IsActiveNode< Dim >( neighborKey.neighbors[ tree._localToGlobal( _depth ) ].neighbors.data[xx]->children ) ) break;
								Key key = _FaceIndex( node , f );
								SliceValues& _sValues = slabValues[_depth].sliceValues( _slice );
								typename SliceValues::Scratch &_sScratch = slabValues[_depth].sliceScratch( _slice );
								_sScratch.fKeyValues[ thread ].push_back( std::pair< Key , std::vector< IsoEdge > >( key , edges ) );
							}
						}
					}
				}
			}
		}
		);
	}

	static void SetXSliceIsoEdges( const LevelSetExtraction::KeyGenerator< Dim > &keyGenerator , const FEMTree< Dim , Real >& tree , LocalDepth depth , int slab , std::vector< SlabValues >& slabValues )
	{
		auto _FaceIndex = [&]( const TreeNode *node , typename HyperCube::Cube< Dim >::template Element< 2 > f )
		{
			int depth , offset[Dim];
			tree.depthAndOffset( node , depth , offset );
			return keyGenerator( depth , offset , f );
		};

		SliceValues& bValues = slabValues[depth].sliceValues ( slab   );
		SliceValues& fValues = slabValues[depth].sliceValues ( slab+1 );
		XSliceValues& xValues = slabValues[depth].xSliceValues( slab   );
		typename  SliceValues::Scratch &bScratch = slabValues[depth]. sliceScratch( slab   );
		typename  SliceValues::Scratch &fScratch = slabValues[depth]. sliceScratch( slab+1 );
		typename XSliceValues::Scratch &xScratch = slabValues[depth].xSliceScratch( slab   );

		std::vector< ConstOneRingNeighborKey > neighborKeys( ThreadPool::NumThreads() );
		for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( tree._localToGlobal( depth ) );
		ThreadPool::ParallelFor( tree._sNodesBegin(depth,slab) , tree._sNodesEnd(depth,slab) , [&]( unsigned int thread , size_t i )
		{
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) )
			{
				int isoEdges[ 2 * HyperCube::MarchingSquares::MAX_EDGES ];
				ConstOneRingNeighborKey& neighborKey = neighborKeys[ thread ];
				TreeNode* leaf = tree._sNodes.treeNodes[i];
				if( !IsActiveNode< Dim >( leaf->children ) )
				{
					const typename LevelSetExtraction::SlabCellIndexData< Dim >::template CellIndices<0> &cIndices = xValues.cellIndices.template indices<0>( leaf );
					const typename LevelSetExtraction::SlabCellIndexData< Dim >::template CellIndices<1> &eIndices = xValues.cellIndices.template indices<1>( leaf );
					unsigned char mcIndex = ( bValues.mcIndices[ i - bValues.cellIndices.nodeOffset ] ) | ( fValues.mcIndices[ i - fValues.cellIndices.nodeOffset ]<<4 );
					{
						neighborKey.getNeighbors( leaf );
						// Iterate over the edges on the back
						for( typename HyperCube::Cube< Dim-1 >::template Element< 1 > _e ; _e<HyperCube::Cube< Dim-1 >::template ElementNum< 1 >() ; _e++ )
						{
							typename HyperCube::Cube< Dim >::template Element< 2 > f( HyperCube::CROSS , _e.index );
							unsigned char _mcIndex = HyperCube::Cube< Dim >::template ElementMCIndex< 2 >( f , mcIndex );

							unsigned int xx = HyperCubeTables< Dim , 2 >::CellOffsetAntipodal[f.index];
							if(	!xScratch.fSet[ eIndices[_e.index] ] && ( !IsActiveNode< Dim >( neighborKey.neighbors[ tree._localToGlobal( depth ) ].neighbors.data[xx] ) || !IsActiveNode< Dim >( neighborKey.neighbors[ tree._localToGlobal( depth ) ].neighbors.data[xx]->children ) ) )
							{
								FaceEdges fe;
								fe.count = HyperCube::MarchingSquares::AddEdgeIndices( _mcIndex , isoEdges );
								for( int j=0 ; j<fe.count ; j++ ) for( int k=0 ; k<2 ; k++ )
								{
									typename HyperCube::Cube< Dim >::template Element< 1 > e( f , typename HyperCube::Cube< Dim-1 >::template Element< 1 >( isoEdges[2*j+k] ) );
									HyperCube::Direction dir ; unsigned int coIndex;
									e.factor( dir , coIndex );
									if( dir==HyperCube::CROSS ) // Cross-edge
									{
										node_index_type idx = cIndices[ coIndex ];
										if( !xScratch.eSet[ idx ] ) MK_THROW( "Edge not set: " , slab , " / " , 1<<depth );
										fe.edges[j][k] = xValues.edgeKeys[ idx ];
									}
									else
									{
										const SliceValues& sValues = dir==HyperCube::BACK ? bValues : fValues;
										const typename SliceValues::Scratch &sScratch = dir==HyperCube::BACK ? bScratch : fScratch;
										node_index_type idx = sValues.cellIndices.template indices<1>((node_index_type)i)[ coIndex ];
										if( !sScratch.eSet[ idx ] ) MK_THROW( "Edge not set: " , slab , " / " , 1<<depth );
										fe.edges[j][k] = sValues.edgeKeys[ idx ];
									}
								}
								xScratch.fSet[ eIndices[_e.index] ] = 1;
#ifdef SANITIZED_PR
								SetAtomic( xValues.faceEdges[ eIndices[_e.index] ] , fe );
#else // !SANITIZED_PR
								xValues.faceEdges[ eIndices[_e.index] ] = fe;
#endif // SANITIZED_PR

								TreeNode* node = leaf;
								LocalDepth _depth = depth;
								int _slab = slab;
								std::vector< IsoEdge > edges;
								edges.resize( fe.count );
								for( int j=0 ; j<fe.count ; j++ ) edges[j] = fe.edges[j];
								while( tree._isValidSpaceNode( node->parent ) && HyperCubeTables< Dim , 2 , 0 >::Overlap[f.index][(unsigned int)(node-node->parent->children) ] )
								{
									node = node->parent , _depth-- , _slab >>= 1;
									if( IsActiveNode< Dim >( neighborKey.neighbors[ tree._localToGlobal( _depth ) ].neighbors.data[xx] ) && IsActiveNode< Dim >( neighborKey.neighbors[ tree._localToGlobal( _depth ) ].neighbors.data[xx]->children ) ) break;
									Key key = _FaceIndex( node , f );
									XSliceValues& _xValues = slabValues[_depth].xSliceValues( _slab );
									typename XSliceValues::Scratch &_xScratch = slabValues[_depth].xSliceScratch( _slab );
									_xScratch.fKeyValues[ thread ].push_back( std::pair< Key , std::vector< IsoEdge > >( key , edges ) );
								}
							}
						}
					}
				}
			}
		} );
	}

	template< typename VertexStream , typename FaceIndexFunctor /* = std::function< LevelSetExtraction::Key< Dim > ( const TreeNode * , typename HyperCube::Cube< Dim >::template Element< 2 > ) */ >
	static void SetLevelSet( const LevelSetExtraction::KeyGenerator< Dim > &keyGenerator , FaceIndexFunctor faceIndexFunctor , const FEMTree< Dim , Real >& tree , LocalDepth depth , int offset , const SliceValues& bValues , const SliceValues& fValues , const XSliceValues& xValues , const typename SliceValues::Scratch &bScratch , const typename SliceValues::Scratch &fScratch , const typename XSliceValues::Scratch &xScratch , VertexStream &vertexStream , OutputDataStream< std::vector< node_index_type > > &polygonStream , bool polygonMesh , bool addBarycenter , bool flipOrientation )
	{
		std::vector< std::pair< node_index_type , Vertex > > polygon;
		std::vector< std::vector< IsoEdge > > edgess( ThreadPool::NumThreads() );
		ThreadPool::ParallelFor( tree._sNodesBegin(depth,offset) , tree._sNodesEnd(depth,offset) , [&]( unsigned int thread , size_t i )
		{
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) )
			{
				std::vector< IsoEdge >& edges = edgess[ thread ];
				TreeNode* leaf = tree._sNodes.treeNodes[i];
				int res = 1<<depth;
				LocalDepth d ; LocalOffset off;
				tree._localDepthAndOffset( leaf , d , off );
				bool inBounds = off[0]>=0 && off[0]<res && off[1]>=0 && off[1]<res && off[2]>=0 && off[2]<res;
				if( inBounds && !IsActiveNode< Dim >( leaf->children ) )
				{
					edges.clear();
					{
						// Gather the edges from the faces (with the correct orientation)
						for( typename HyperCube::Cube< Dim >::template Element< Dim-1 > f ; f<HyperCube::Cube< Dim >::template ElementNum< Dim-1 >() ; f++ )
						{
							bool isOriented = HyperCube::Cube< Dim >::IsOriented( f );
							int flip = isOriented ? 0 : 1;
							HyperCube::Direction fDir = f.direction();

							if     ( fDir==HyperCube::BACK  ) bValues.addIsoEdges( keyGenerator , faceIndexFunctor , tree , bScratch , leaf , offset+0 , edges , isOriented );
							else if( fDir==HyperCube::FRONT ) fValues.addIsoEdges( keyGenerator , faceIndexFunctor , tree , fScratch , leaf , offset+1 , edges , isOriented );
							else
							{
								node_index_type fIdx = xValues.cellIndices.template indices<1>((node_index_type)i)[ f.coIndex() ];
								if( xScratch.fSet[fIdx] )
								{
									const FaceEdges& fe = xValues.faceEdges[ fIdx ];
									for( int j=0 ; j<fe.count ; j++ ) edges.push_back( IsoEdge( fe.edges[j][flip] , fe.edges[j][1-flip] ) );
								}
								else
								{
									Key key = faceIndexFunctor( leaf , f );
									typename LevelSetExtraction::KeyMap< Dim , std::vector< IsoEdge > >::const_iterator iter = xValues.faceEdgeMap.find(key);
									if( iter!=xValues.faceEdgeMap.end() )
									{
										const std::vector< IsoEdge >& _edges = iter->second;
										for( size_t j=0 ; j<_edges.size() ; j++ ) edges.push_back( IsoEdge( _edges[j][flip] , _edges[j][1-flip] ) );
									}
									else MK_THROW( "Invalid faces: " , i , "  " ,  fDir==HyperCube::BACK ? "back" : ( fDir==HyperCube::FRONT ? "front" : ( fDir==HyperCube::CROSS ? "cross" : "unknown" ) ) );
								}
							}
						}
						// Get the edge loops
						std::vector< std::vector< Key > > loops;
						while( edges.size() )
						{
							loops.resize( loops.size()+1 );
							IsoEdge edge = edges.back();
							edges.pop_back();
							Key start = edge[0] , current = edge[1];
							while( current!=start )
							{
								int idx;
								for( idx=0 ; idx<(int)edges.size() ; idx++ ) if( edges[idx][0]==current ) break;
								if( idx==edges.size() )
								{
									typename LevelSetExtraction::KeyMap< Dim , Key >::const_iterator iter;
									Key pair;
									if     ( bValues.setVertexPair(current,pair) ) loops.back().push_back( current ) , current = pair;
									else if( fValues.setVertexPair(current,pair) ) loops.back().push_back( current ) , current = pair;
									else if( (iter=xValues.vertexPairMap.find(current))!=xValues.vertexPairMap.end() ) loops.back().push_back( current ) , current = iter->second;
									else MK_THROW( "Failed to close loop for node[" , i , "]: [" , off[0] , " " , off[1] , " " , off[2] , " @ " , d , "] | " , keyGenerator.to_string( current ) , " -- " , keyGenerator.to_string( start ) , " | " , current.to_string() , " -- " , start.to_string() );
								}
								else
								{
									loops.back().push_back( current );
									current = edges[idx][1];
									edges[idx] = edges.back() , edges.pop_back();
								}
							}
							loops.back().push_back( start );
						}
						// Add the loops to the mesh
						for( size_t j=0 ; j<loops.size() ; j++ )
						{
							std::vector< std::pair< node_index_type , Vertex > > polygon( loops[j].size() );
							for( size_t k=0 ; k<loops[j].size() ; k++ )
							{
								Key key = loops[j][k];
								typename LevelSetExtraction::KeyMap< Dim , std::pair< node_index_type , Vertex > >::const_iterator iter;
								size_t kk = flipOrientation ? loops[j].size()-1-k : k;
								if     ( bValues.setEdgeVertex( key , polygon[kk] ) );
								else if( fValues.setEdgeVertex( key , polygon[kk] ) );
								else if( ( iter=xValues.edgeVertexMap.find( key ) )!=xValues.edgeVertexMap.end() ) polygon[kk] = iter->second;
								else MK_THROW( "Couldn't find vertex in edge map: " , off[0] , " , " , off[1] , " , " , off[2] , " @ " , depth , " : " , keyGenerator.to_string( key ) , " | " , key.to_string() );
							}
							AddIsoPolygons( thread , vertexStream , polygonStream , polygon , polygonMesh , addBarycenter );
						}
					}
				}
			}
		} );
	}

	template< unsigned int WeightDegree , unsigned int DataSig >
	static bool GetIsoVertex
	(
		const FEMTree< Dim , Real >& tree ,
		bool nonLinearFit ,
		bool gradientNormals ,
		typename FEMIntegrator::template PointEvaluator< IsotropicUIntPack< Dim , DataSig > , ZeroUIntPack< Dim > > *pointEvaluator ,
		const DensityEstimator< WeightDegree > *densityWeights ,
		const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > *data ,
		Real isoValue ,
		ConstPointSupportKey< IsotropicUIntPack< Dim , WeightDegree > >& weightKey ,
		ConstPointSupportKey< IsotropicUIntPack< Dim , FEMSignature< DataSig >::Degree > > &dataKey ,
		const TreeNode *node ,
		typename HyperCube::template Cube< Dim-1 >::template Element< 1 > _e ,
		HyperCube::Direction zDir ,
		const SliceValues& sValues ,
		Vertex& vertex ,
		const Data &zeroData
	)
	{
		static const unsigned int DataDegree = FEMSignature< DataSig >::Degree;
		Point< Real , Dim > position , gradient;
		int c0 , c1;
		const typename HyperCube::Cube< Dim-1 >::template Element< 0 > *_c = HyperCubeTables< Dim-1 , 1 , 0 >::OverlapElements[_e.index];
		c0 = _c[0].index , c1 = _c[1].index;

		const typename LevelSetExtraction::SliceCellIndexData< Dim >::template CellIndices<0> &idx = sValues.cellIndices.template indices<0>( node );
		Real x0 = sValues.cornerValues[idx[c0]] , x1 = sValues.cornerValues[idx[c1]];
		Point< Real , 3 > dx0 , dx1;
		if( gradientNormals ) dx0 = sValues.cornerGradients[idx[c0]] , dx1 = sValues.cornerGradients[idx[c1]];
		Point< Real , Dim > s;
		Real start , width;
		tree._startAndWidth( node , s , width );
		int o;
		{
			const HyperCube::Direction* dirs = HyperCubeTables< Dim-1 , 1 >::Directions[ _e.index ];
			for( int d=0 ; d<Dim-1 ; d++ ) if( dirs[d]==HyperCube::CROSS )
			{
				o = d;
				start = s[d];
				for( int dd=1 ; dd<Dim-1 ; dd++ ) position[(d+dd)%(Dim-1)] = s[(d+dd)%(Dim-1)] + width * ( dirs[(d+dd)%(Dim-1)]==HyperCube::BACK ? 0 : 1 );
			}
		}
		position[ Dim-1 ] = s[Dim-1] + width * ( zDir==HyperCube::BACK ? 0 : 1 );

		double averageRoot;
		bool rootFound = false;
		if( nonLinearFit )
		{
			double dx0 = sValues.cornerGradients[idx[c0]][o] * width , dx1 = sValues.cornerGradients[idx[c1]][o] * width;

			// The scaling will turn the Hermite Spline into a quadratic
			double scl = (x1-x0) / ( (dx1+dx0 ) / 2 );
			dx0 *= scl , dx1 *= scl;

			// Hermite Spline
			Polynomial< 2 > P;
			P.coefficients[0] = x0;
			P.coefficients[1] = dx0;
			P.coefficients[2] = 3*(x1-x0)-dx1-2*dx0;

			double roots[2];
			int rCount = 0 , rootCount = P.getSolutions( isoValue , roots , 0 );
			averageRoot = 0;
			for( int i=0 ; i<rootCount ; i++ ) if( roots[i]>=0 && roots[i]<=1 ) averageRoot += roots[i] , rCount++;
			if( rCount ) rootFound = true;
			averageRoot /= rCount;
		}
		if( !rootFound )
		{
			// We have a linear function L, with L(0) = x0 and L(1) = x1
			// => L(t) = x0 + t * (x1-x0)
			// => L(t) = isoValue <=> t = ( isoValue - x0 ) / ( x1 - x0 )
			if( x0==x1 ) MK_THROW( "Not a zero-crossing root: " , x0 , " " , x1 );
			averageRoot = ( isoValue - x0 ) / ( x1 - x0 );
		}
		if( averageRoot<=0 || averageRoot>=1 )
		{
			_BadRootCount++;
			if( averageRoot<0 ) averageRoot = 0;
			if( averageRoot>1 ) averageRoot = 1;
		}
		position[o] = Real( start + width*averageRoot );
		gradient = dx0 * (Real)( 1.-averageRoot ) + dx1 * (Real)averageRoot;
		Real depth = (Real)1.;
		Data dataValue;
		if( densityWeights )
		{
			Real weight;
			tree._getSampleDepthAndWeight( *densityWeights , node , position , weightKey , depth , weight );
		}
		if constexpr( HasData ) if( data )
		{
			if( DataDegree==0 ) 
			{
				Point< Real , 3 > center( s[0] + width/2 , s[1] + width/2 , s[2] + width/2 );
				ProjectiveData< Data , Real > pValue( zeroData );
				tree.template _addEvaluation< ProjectiveData< Data , Real > , SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > , 0 >( *data , center , *pointEvaluator , dataKey , pValue );
				dataValue = pValue.weight ? pValue.value() : zeroData;
			}
			else
			{
				ProjectiveData< Data , Real > pValue( zeroData );
				tree.template _addEvaluation< ProjectiveData< Data , Real > , SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > , 0 >( *data , position , *pointEvaluator , dataKey , pValue );
				dataValue = pValue.weight ? pValue.value() : zeroData;
			}
		}
		vertex.template get<0>() = position;
		vertex.template get<1>() = gradient;
		vertex.template get<2>() = depth;
		if constexpr( HasData ) vertex.template get<3>() = dataValue;
		return true;
	}

	template< unsigned int WeightDegree , unsigned int DataSig >
	static bool GetIsoVertex
	(
		const FEMTree< Dim , Real > &tree ,
		bool nonLinearFit ,
		bool gradientNormals ,
		typename FEMIntegrator::template PointEvaluator< IsotropicUIntPack< Dim , DataSig > , ZeroUIntPack< Dim > > *pointEvaluator ,
		const DensityEstimator< WeightDegree > *densityWeights ,
		const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > *data ,
		Real isoValue ,
		ConstPointSupportKey< IsotropicUIntPack< Dim , WeightDegree > > &weightKey ,
		ConstPointSupportKey< IsotropicUIntPack< Dim , FEMSignature< DataSig >::Degree > > &dataKey ,
		const TreeNode *node ,
		typename HyperCube::template Cube< Dim-1 >::template Element< 0 > _c ,
		Real bCoordinate ,
		Real fCoordinate ,
		const SliceValues &bValues ,
		const SliceValues &fValues ,
		Vertex &vertex ,
		const Data &zeroData
	)
	{
		static const unsigned int DataDegree = FEMSignature< DataSig >::Degree;
		Point< Real , Dim > position , gradient;

		const typename LevelSetExtraction::SliceCellIndexData< Dim >::template CellIndices<0> &idx0 = bValues.cellIndices.template indices<0>( node );
		const typename LevelSetExtraction::SliceCellIndexData< Dim >::template CellIndices<0> &idx1 = fValues.cellIndices.template indices<0>( node );
		Real x0 = bValues.cornerValues[ idx0[_c.index] ] , x1 = fValues.cornerValues[ idx1[_c.index] ];
		Point< Real , 3 > dx0 , dx1;
		if( gradientNormals ) dx0 = bValues.cornerGradients[ idx0[_c.index] ] , dx1 = fValues.cornerGradients[ idx1[_c.index] ];
		Point< Real , Dim > s;
		Real w;
		tree._startAndWidth( node , s , w );
		int x , y;
		{
			const HyperCube::Direction* xx = HyperCubeTables< Dim-1 , 0 >::Directions[ _c.index ];
			x = xx[0]==HyperCube::BACK ? 0 : 1 , y = xx[1]==HyperCube::BACK ? 0 : 1;
		}

		position[0] = s[0] + w*x;
		position[1] = s[1] + w*y;

		double averageRoot;
		bool rootFound = false;

		if( nonLinearFit )
		{
			double dx0 = bValues.cornerGradients[ idx0[_c.index] ][2] * (fCoordinate-bCoordinate) , dx1 = fValues.cornerGradients[ idx1[_c.index] ][2] * (fCoordinate-bCoordinate);
			// The scaling will turn the Hermite Spline into a quadratic
			double scl = (x1-x0) / ( (dx1+dx0 ) / 2 );
			dx0 *= scl , dx1 *= scl;

			// Hermite Spline
			Polynomial< 2 > P;
			P.coefficients[0] = x0;
			P.coefficients[1] = dx0;
			P.coefficients[2] = 3*(x1-x0)-dx1-2*dx0;

			double roots[2];
			int rCount = 0 , rootCount = P.getSolutions( isoValue , roots , 0 );
			averageRoot = 0;
			for( int i=0 ; i<rootCount ; i++ ) if( roots[i]>=0 && roots[i]<=1 ) averageRoot += roots[i] , rCount++;
			if( rCount ) rootFound = true;
			averageRoot /= rCount;
		}
		if( !rootFound )
		{
			// We have a linear function L, with L(0) = x0 and L(1) = x1
			// => L(t) = x0 + t * (x1-x0)
			// => L(t) = isoValue <=> t = ( isoValue - x0 ) / ( x1 - x0 )
			if( x0==x1 ) MK_THROW( "Not a zero-crossing root: " , x0 , " " , x1 );
			averageRoot = ( isoValue - x0 ) / ( x1 - x0 );
		}
		if( averageRoot<=0 || averageRoot>=1 )
		{
			_BadRootCount++;
			if( averageRoot<0 ) averageRoot = 0;
			if( averageRoot>1 ) averageRoot = 1;
		}
		position[2] = Real( bCoordinate + (fCoordinate-bCoordinate)*averageRoot );
		gradient = dx0 * (Real)( 1.-averageRoot ) + dx1 * (Real)averageRoot;
		Real depth = (Real)1.;
		Data dataValue;
		if( densityWeights )
		{
			Real weight;
			tree._getSampleDepthAndWeight( *densityWeights , node , position , weightKey , depth , weight );
		}
		if constexpr( HasData ) if( data )
		{
			if( DataDegree==0 ) 
			{
				Point< Real , 3 > center( s[0] + w/2 , s[1] + w/2 , (bCoordinate+fCoordinate)/2 );
				ProjectiveData< Data , Real > pValue( zeroData );
				tree.template _addEvaluation< ProjectiveData< Data , Real > , SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > , 0 >( *data , center , *pointEvaluator , dataKey , pValue );
				dataValue = pValue.weight ? pValue.value() : zeroData;
			}
			else
			{
				ProjectiveData< Data , Real > pValue( zeroData );
				tree.template _addEvaluation< ProjectiveData< Data , Real > , SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > , 0 >( *data , position , *pointEvaluator , dataKey , pValue );
				dataValue = pValue.weight ? pValue.value() : zeroData;
			}
		}
		vertex.template get<0>() = position;
		vertex.template get<1>() = gradient;
		vertex.template get<2>() = depth;
		if constexpr( HasData ) vertex.template get<3>() = dataValue;
		return true;
	}

	template< typename VertexStream >
	static unsigned int AddIsoPolygons( unsigned int thread , VertexStream &vertexStream , OutputDataStream< std::vector< node_index_type > > &polygonStream , std::vector< std::pair< node_index_type , Vertex > >& polygon , bool polygonMesh , bool addBarycenter )
	{
		if( polygonMesh )
		{
			std::vector< node_index_type > vertices( polygon.size() );
			for( unsigned int i=0 ; i<polygon.size() ; i++ ) vertices[i] = polygon[polygon.size()-1-i].first;
			polygonStream.write( thread , vertices );
			return 1;
		}
		if( polygon.size()>3 )
		{
			bool isCoplanar = false;
			std::vector< node_index_type > triangle( 3 );

			if( addBarycenter )
				for( unsigned int i=0 ; i<polygon.size() ; i++ ) for( unsigned int j=0 ; j<i ; j++ )
					if( (i+1)%polygon.size()!=j && (j+1)%polygon.size()!=i )
					{
						Vertex v1 = polygon[i].second , v2 = polygon[j].second;
						for( int k=0 ; k<3 ; k++ ) if( v1.template get<0>()[k]==v2.template get<0>()[k] ) isCoplanar = true;
					}
			if( isCoplanar )
			{
				Vertex c;
				c *= 0;
				for( unsigned int i=0 ; i<polygon.size() ; i++ ) c += polygon[i].second;
				c /= ( typename Vertex::Real )polygon.size();

				node_index_type cIdx = (node_index_type)vertexStream.write( thread , c );

				for( unsigned i=0 ; i<polygon.size() ; i++ )
				{
					triangle[0] = polygon[ i                  ].first;
					triangle[1] = cIdx;
					triangle[2] = polygon[(i+1)%polygon.size()].first;
					polygonStream.write( thread , triangle );
				}
				return (unsigned int)polygon.size();
			}
			else
			{
				std::vector< Point< Real , Dim > > vertices( polygon.size() );
				for( unsigned int i=0 ; i<polygon.size() ; i++ ) vertices[i] = polygon[i].second.template get<0>();
				std::vector< TriangleIndex< node_index_type > > triangles = MinimalAreaTriangulation< node_index_type , Real , Dim >( ( ConstPointer( Point< Real , Dim > ) )GetPointer( vertices ) , (node_index_type)vertices.size() );
				if( triangles.size()!=polygon.size()-2 ) MK_THROW( "Minimal area triangulation failed:" , triangles.size() , " != " , polygon.size()-2 );
				for( unsigned int i=0 ; i<triangles.size() ; i++ )
				{
					for( int j=0 ; j<3 ; j++ ) triangle[2-j] = polygon[ triangles[i].idx[j] ].first;
					polygonStream.write( thread , triangle );
				}
			}
		}
		else if( polygon.size()==3 )
		{
			std::vector< node_index_type > vertices( 3 );
			for( int i=0 ; i<3 ; i++ ) vertices[2-i] = polygon[i].first;
			polygonStream.write( thread , vertices );
		}
		return (unsigned int)polygon.size()-2;
	}

	struct Stats
	{
		double cornersTime , verticesTime , edgesTime , surfaceTime;
		double setTableTime;
		Stats( void ) : cornersTime(0) , verticesTime(0) , edgesTime(0) , surfaceTime(0) , setTableTime(0) {;}
		std::string toString( void ) const
		{
			std::stringstream stream;
			stream << "Corners / Vertices / Edges / Surface / Set Table: ";
			stream << std::fixed << std::setprecision(1) << cornersTime << " / " << verticesTime << " / " << edgesTime << " / " << surfaceTime << " / " << setTableTime;
			stream << " (s)";
			return stream.str();
		}
	};


	template< unsigned int WeightDegree , unsigned int DataSig , typename OutputVertexStream , unsigned int ... FEMSigs >
	static Stats Extract
	(
		UIntPack< FEMSigs ... > ,
		UIntPack< WeightDegree > ,
		UIntPack< DataSig > ,
		const FEMTree< Dim , Real > &tree ,
		int maxKeyDepth ,
		const DensityEstimator< WeightDegree > *densityWeights ,
		const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > *data ,
		const DenseNodeData< Real , UIntPack< FEMSigs ... > > &coefficients ,
		Real isoValue ,
		unsigned int slabDepth ,
		unsigned int slabStart ,
		unsigned int slabEnd ,
		OutputVertexStream &vertexStream ,
		OutputDataStream< std::vector< node_index_type > > &polygonStream ,
		const Data &zeroData ,
		bool nonLinearFit ,
		bool gradientNormals ,
		bool addBarycenter ,
		bool polygonMesh ,
		bool flipOrientation ,
		const typename LevelSetExtractor< Real , Dim-1 , Point< Real , Dim-1 > >::TreeSliceValuesAndVertexPositions * backBoundary ,
		const typename LevelSetExtractor< Real , Dim-1 , Point< Real , Dim-1 > >::TreeSliceValuesAndVertexPositions *frontBoundary ,
		const std::vector< std::vector< Real > > &backDValues ,
		const std::vector< std::vector< Real > > &frontDValues ,
		bool copyTopology
	)
	{
		if( maxKeyDepth<tree._maxDepth ) MK_THROW( "Max key depth has to be at least tree depth: " , tree._maxDepth , " <= " , maxKeyDepth );
		if( slabStart>=slabEnd ) MK_THROW( "Slab start cannot excceed slab end: " , slabStart , " < " , slabEnd );
		if( slabEnd>(1u<<slabDepth) ) MK_THROW( "Slab end cannot exceed slab num: " , slabEnd , " <= " , 1<<slabDepth );

		LevelSetExtraction::KeyGenerator< Dim > keyGenerator( maxKeyDepth );
		LocalOffset start , end;
		for( unsigned int d=0 ; d<Dim-1 ; d++ ) start[d] = 0 , end[d] = (1<<slabDepth);
		start[Dim-1] = slabStart  , end[Dim-1] = slabEnd;
		LocalDepth fullDepth = tree.getFullDepth( UIntPack< FEMSignature< FEMSigs >::Degree ... >() , slabDepth , start , end );
		unsigned int maxDepth = std::max< unsigned int >( tree._maxDepth , slabDepth );
		unsigned int slabStartAtMaxDepth = slabStart << ( maxDepth - slabDepth );
		unsigned int slabEndAtMaxDepth = slabEnd << ( maxDepth - slabDepth );
#ifdef SHOW_WARNINGS
		if( slabDepth>(unsigned int)fullDepth && ( ( slabStart!=0 && !backBoundary ) || ( slabEnd+1!=1<<(slabDepth) && !frontBoundary ) ) ) MK_WARN( "Slab depth exceeds full depth, reconstruction may not be water-tight: " , slabDepth , " <= " , fullDepth , " [ " , slabStart , " , " , slabEnd , " )" );
#endif // SHOW_WARNINGS

		_BadRootCount = 0u;
		Stats stats;
		static_assert( sizeof...(FEMSigs)==Dim , "[ERROR] Number of signatures should match dimension" );
		tree._setFEM1ValidityFlags( UIntPack< FEMSigs ... >() );
		static const unsigned int DataDegree = FEMSignature< DataSig >::Degree;
		static const int FEMDegrees[] = { FEMSignature< FEMSigs >::Degree ... };
		for( int d=0 ; d<Dim ; d++ ) if( FEMDegrees[d]==0 && ( nonLinearFit || gradientNormals ) ) MK_THROW( "Constant B-Splines do not support gradient estimation" );

		LevelSetExtraction::SetHyperCubeTables< Dim >();
		LevelSetExtraction::SetHyperCubeTables< Dim-1 >();

		typename FEMIntegrator::template PointEvaluator< IsotropicUIntPack< Dim , DataSig > , ZeroUIntPack< Dim > >* pointEvaluator = NULL;
		if constexpr( HasData ) if( data ) pointEvaluator = new typename FEMIntegrator::template PointEvaluator< IsotropicUIntPack< Dim , DataSig > , ZeroUIntPack< Dim > >( tree._maxDepth );
		DenseNodeData< Real , UIntPack< FEMSigs ... > > coarseCoefficients( tree._sNodesEnd( tree._maxDepth-1 ) );
		memset( coarseCoefficients() , 0 , sizeof(Real)*tree._sNodesEnd( tree._maxDepth-1 ) );
		ThreadPool::ParallelFor( tree._sNodesBegin(0) , tree._sNodesEnd( tree._maxDepth-1 ) , [&]( unsigned int, size_t i ){ coarseCoefficients[i] = coefficients[i]; } );
		typename FEMIntegrator::template RestrictionProlongation< UIntPack< FEMSigs ... > > rp;
		for( LocalDepth d=1 ; d<tree._maxDepth ; d++ ) tree._upSample( UIntPack< FEMSigs ... >() , rp , d , ( ConstPointer(Real) )coarseCoefficients()+tree._sNodesBegin(d-1) , coarseCoefficients()+tree._sNodesBegin(d) );

		std::vector< _Evaluator< UIntPack< FEMSigs ... > , 1 > > evaluators( tree._maxDepth+1 );
		for( LocalDepth d=0 ; d<=tree._maxDepth ; d++ ) evaluators[d].set( tree._maxDepth );

		std::vector< SlabValues > slabValues( tree._maxDepth+1 );
		std::vector< std::pair< node_index_type , node_index_type > > backIncidence , frontIncidence;
		if(  backBoundary )  backIncidence = SetIncidence( tree ,  backBoundary->sliceTree , fullDepth , slabStartAtMaxDepth , maxDepth );
		if( frontBoundary ) frontIncidence = SetIncidence( tree , frontBoundary->sliceTree , fullDepth , slabEndAtMaxDepth , maxDepth );

		enum BoundarySliceType { BACK , FRONT , NEITHER };

		auto BoundarySlice = [&]( unsigned int sliceAtMaxDepth )
		{
			if     (  backBoundary && sliceAtMaxDepth==slabStartAtMaxDepth ) return BoundarySliceType::BACK;
			else if( frontBoundary && sliceAtMaxDepth==  slabEndAtMaxDepth ) return BoundarySliceType::FRONT;
			else return BoundarySliceType::NEITHER;
		};


		auto SetCoarseSlice = [&]( unsigned int sliceAtMaxDepth , unsigned int coarseDepth , unsigned int &coarseSlice )
		{
			BoundarySliceType bst = BoundarySlice( sliceAtMaxDepth );
			unsigned dOff = maxDepth - coarseDepth;
			coarseSlice = sliceAtMaxDepth>>dOff;
			if( sliceAtMaxDepth!=( coarseSlice<<dOff ) )
			{
				if     ( bst==BoundarySliceType::BACK ) ;
				else if( bst==BoundarySliceType::FRONT ) coarseSlice++;
				else return false;
			}
			return true;
		};

		auto InteriorSlab = [&]( unsigned int d , unsigned int o )
		{
			unsigned int startAtMaxDepth = (o+0)<<(maxDepth-d);
			unsigned int   endAtMaxDepth = (o+1)<<(maxDepth-d);
			if(   endAtMaxDepth<=slabStartAtMaxDepth ) return false;
			if( startAtMaxDepth>=  slabEndAtMaxDepth ) return false;
			if( startAtMaxDepth<slabStartAtMaxDepth && ! backBoundary ) return false;
			if(   endAtMaxDepth>  slabEndAtMaxDepth && !frontBoundary ) return false;
			return true;
		};

		auto SetSlabBounds = [&]( unsigned int d , unsigned int o , Real &bCoordinate , Real &fCoordinate )
		{
			unsigned int start = std::max< unsigned int >( (o+0)<<(maxDepth-d) , slabStartAtMaxDepth );
			unsigned int   end = std::min< unsigned int >( (o+1)<<(maxDepth-d) ,   slabEndAtMaxDepth );
			bCoordinate = start/(Real)(1<<maxDepth);
			fCoordinate =   end/(Real)(1<<maxDepth);
		};

		// Initializes the indexing tables
		auto InitSlice = [&]( unsigned int sliceAtMaxDepth )
		{

			BoundarySliceType bst = BoundarySlice( sliceAtMaxDepth );

			for( LocalDepth d=maxDepth ; d>=fullDepth ; d-- )
			{
				unsigned int slice;
				if( !SetCoarseSlice( sliceAtMaxDepth , d , slice ) ) break;
				if( d<=tree._maxDepth )
				{
					double t = Time();
					slabValues[d].sliceValues(slice).cellIndices.set( tree._sNodes , tree._localToGlobal( d ) , slice + tree._localInset( d ) );
					stats.setTableTime += Time()-t;

					slabValues[d].sliceValues(slice).reset( slice , nonLinearFit || gradientNormals );

					slabValues[d].sliceScratch(slice).reset( slabValues[d].sliceValues(slice).cellIndices );
				}
			}
		};

		auto InitSlab = [&]( unsigned int slabAtMaxDepth , bool first )
		{
			unsigned int slab = slabAtMaxDepth;
			for( LocalDepth d=maxDepth ; d>=fullDepth ; d-- , slab>>=1 )
			{
				if( d<=tree._maxDepth )
				{
					double t = Time();
					slabValues[d].xSliceValues(slab).cellIndices.set( tree._sNodes , tree._localToGlobal( d ) , slab + tree._localInset( d ) );
					stats.setTableTime += Time()-t;

					slabValues[d].xSliceValues(slab).reset(slab);

					slabValues[d].xSliceScratch(slab).reset( slabValues[d].xSliceValues(slab).cellIndices );
				}
				if( (slab&1) && !first ) break;
			}
		};

		auto FinalizeSlice = [&]( unsigned int sliceAtMaxDepth )
		{

			bool useBoundary = ( ( sliceAtMaxDepth==slabStartAtMaxDepth && backBoundary ) || ( sliceAtMaxDepth==slabEndAtMaxDepth && frontBoundary ) ) && copyTopology;
			if( !useBoundary )
			{
				LocalDepth d ; unsigned int o;
				for( d=maxDepth , o=sliceAtMaxDepth ; d>=fullDepth ; d-- , o>>=1 )
				{
					if( d<=tree._maxDepth )
					{
						ThreadPool::ParallelSections
						(
							[ &slabValues , d , o ]( void ){ slabValues[d].sliceValues(o).setFromScratch( slabValues[d].sliceScratch(o).vKeyValues ); } ,
							[ &slabValues , d , o ]( void ){ slabValues[d].sliceValues(o).setFromScratch( slabValues[d].sliceScratch(o).eKeyValues ); } ,
							[ &slabValues , d , o ]( void ){ slabValues[d].sliceValues(o).setFromScratch( slabValues[d].sliceScratch(o).fKeyValues ); }
						);
					}
					if( o&1 ) break;
				}
			}
		};

		auto FinalizeSlab = [&]( unsigned int slabAtMaxDepth )
		{
			LocalDepth d ; unsigned int o;
			bool boundary = (slabAtMaxDepth+1)==slabEndAtMaxDepth && frontBoundary;
			for( d=maxDepth , o=slabAtMaxDepth ; d>=fullDepth ; d-- , o>>=1 )
			{
				if( d<=tree._maxDepth )
				{
					ThreadPool::ParallelSections
					(
						[ &slabValues , d , o ]( void ){ slabValues[d].xSliceValues(o).setFromScratch( slabValues[d].xSliceScratch(o).vKeyValues ); } ,
						[ &slabValues , d , o ]( void ){ slabValues[d].xSliceValues(o).setFromScratch( slabValues[d].xSliceScratch(o).eKeyValues ); } ,
						[ &slabValues , d , o ]( void ){ slabValues[d].xSliceValues(o).setFromScratch( slabValues[d].xSliceScratch(o).fKeyValues ); }
					);
				}
				if( !(o&1) && !boundary ) break;
			}
		};

		auto SetSliceValues = [&]( unsigned int sliceAtMaxDepth )
		{

			LocalDepth d ; unsigned int o;
			const typename LevelSetExtractor< Real , Dim-1 , Point< Real , Dim-1 > >::TreeSliceValuesAndVertexPositions *boundary = NULL;
			const std::vector< std::pair< node_index_type , node_index_type > > *incidence = NULL;
			const std::vector< std::vector< Real > > *dValues = NULL;
			if     ( sliceAtMaxDepth==slabStartAtMaxDepth ) boundary =  backBoundary , incidence =  &backIncidence , dValues = & backDValues;
			else if( sliceAtMaxDepth==  slabEndAtMaxDepth ) boundary = frontBoundary , incidence = &frontIncidence , dValues = &frontDValues;

			if( boundary && copyTopology )
			{
				double t = Time();
				for( LocalDepth d=maxDepth ; d>=fullDepth ; d-- ) if( d<=boundary->sliceTree.depth() && d<=tree._maxDepth )
				{
					unsigned int slice;
					if( !SetCoarseSlice( sliceAtMaxDepth , d , slice ) ) MK_THROW( "Could not set coarse slice" );
					OverwriteCornerValues( *boundary , (*dValues)[d] , tree , d , sliceAtMaxDepth , maxDepth , sliceAtMaxDepth==slabStartAtMaxDepth , slabValues , *incidence );
					SetMCIndices( tree , isoValue , d , fullDepth , slice , slabValues );
				}
				stats.cornersTime += Time()-t , t = Time();
			}
			else
			{
				double t;

				t = Time();
				for( d=maxDepth , o=sliceAtMaxDepth ; d>=fullDepth ; d-- , o>>=1 )
				{
					if( d<=tree._maxDepth )
					{
						if( boundary )
						{
							if( d<=boundary->sliceTree.depth() )
							{
								OverwriteCornerValues( *boundary , (*dValues)[d] , tree , d , sliceAtMaxDepth , maxDepth , sliceAtMaxDepth==slabStartAtMaxDepth , slabValues , backIncidence );
								SetMCIndices( tree , isoValue , d , fullDepth , o , slabValues );
							}
						}
						else SetSliceCornerValuesAndMCIndices< FEMSigs ... >( tree , coefficients() , coarseCoefficients() , isoValue , d , fullDepth , o , slabValues , evaluators[d] );				
					}
					if( o&1 ) break;
				}
				stats.cornersTime += Time()-t;
			}
		};

		auto SetSliceIso = [&]( unsigned int sliceAtMaxDepth )
		{
			LocalDepth d ; unsigned int o;

			const typename LevelSetExtractor< Real , Dim-1 , Point< Real , Dim-1 > >::TreeSliceValuesAndVertexPositions *boundary = NULL;
			const std::vector< std::pair< node_index_type , node_index_type > > *incidence = NULL;
			const std::vector< std::vector< Real > > *dValues = NULL;
			if     ( sliceAtMaxDepth==slabStartAtMaxDepth ) boundary =  backBoundary , incidence =  &backIncidence , dValues = & backDValues;
			else if( sliceAtMaxDepth==  slabEndAtMaxDepth ) boundary = frontBoundary , incidence = &frontIncidence , dValues = &frontDValues;

			if( boundary && copyTopology )
			{
				auto sliceFunctor = [&]( unsigned int depth ) -> SliceValues &
				{
					unsigned int slice;
					if( !SetCoarseSlice( sliceAtMaxDepth , depth , slice ) ) MK_THROW( "Could not set coarse slice" );
					return slabValues[depth].sliceValues( slice );
				};
				auto scratchFunctor = [&]( unsigned int depth ) -> typename SliceValues::Scratch &
				{
					unsigned int slice;
					if( !SetCoarseSlice( sliceAtMaxDepth , depth , slice ) ) MK_THROW( "Could not set coarse slice" );
					return slabValues[depth].sliceScratch( slice );
				};
				CopyIsoStructure< WeightDegree , DataSig >( keyGenerator , *boundary , tree , fullDepth , sliceAtMaxDepth , maxDepth , sliceFunctor , scratchFunctor , *incidence , vertexStream , gradientNormals , pointEvaluator , densityWeights , data , zeroData );
			}
			else
			{
				double t;
				// Set the iso-vertices
				t = Time();
				for( d=maxDepth , o=sliceAtMaxDepth ; d>=fullDepth ; d-- , o>>=1 )
				{
					if( d<=tree._maxDepth )
					{
						SetSliceIsoVertices< WeightDegree , DataSig >( keyGenerator , tree , nonLinearFit , gradientNormals , pointEvaluator , densityWeights , data , isoValue , d , fullDepth , o , vertexStream , slabValues , zeroData );
					}
					if( o&1 ) break;
				}
				stats.verticesTime += Time()-t;

				// Set the iso-edges
				t = Time();
				for( d=maxDepth , o=sliceAtMaxDepth ; d>=fullDepth ; d-- , o>>=1 )
				{
					if( d<=tree._maxDepth )
					{
						if( d<tree._maxDepth ) CopyFinerSliceIsoEdgeKeys( tree , d , fullDepth , o , slabValues );
						SetSliceIsoEdges( keyGenerator , tree , d , o , slabValues );
					}
					if( o&1 ) break;
				}
				stats.edgesTime += Time()-t;
			}
		};


		auto SetSlabIsoVertices = [&]( unsigned int slabAtMaxDepth )
		{
			LocalDepth d ; unsigned int o;
			double t = Time();

			bool boundary = (slabAtMaxDepth+1)==slabEndAtMaxDepth && frontBoundary;

			// Set the iso-edges
			for( d=maxDepth , o=slabAtMaxDepth ; d>=fullDepth ; d-- , o>>=1 )
			{
				if( d<=tree._maxDepth )
				{
					if( InteriorSlab( d , o ) ) 
					{
						// Set the iso-vertices
						Real bCoordinate , fCoordinate;
						SetSlabBounds( d , o , bCoordinate , fCoordinate );
						SetXSliceIsoVertices< WeightDegree , DataSig >( keyGenerator , tree , nonLinearFit , gradientNormals , pointEvaluator , densityWeights , data , isoValue , d , std::max< unsigned int >( slabDepth , fullDepth ) , o , bCoordinate , fCoordinate , vertexStream , slabValues , zeroData );
					}
				}

				if( !(o&1) && !boundary ) break;
			}
			stats.edgesTime += Time()-t;
		};

		auto SetSlabIsoEdges = [&]( unsigned int slabAtMaxDepth )
		{
			LocalDepth d ; unsigned int o;
			double t = Time();

			bool boundary = (slabAtMaxDepth+1)==slabEndAtMaxDepth && frontBoundary;

			// Set the iso-edges
			for( d=maxDepth , o=slabAtMaxDepth ; d>=fullDepth ; d-- , o>>=1 )
			{
				if( d<=tree._maxDepth )
				{
					if( InteriorSlab( d , o ) ) 
					{
						// Copy the edges from finer
						if( d<tree._maxDepth )
						{
							CopyFinerXSliceIsoEdgeKeys( tree , d , fullDepth , o , slabValues );
						}
						SetXSliceIsoEdges( keyGenerator , tree , d , o , slabValues );
					}

				}
				if( !(o&1) && !boundary ) break;
			}
			stats.edgesTime += Time()-t;
		};


		auto IsoSurface = [&]( unsigned int slabAtMaxDepth )
		{
			double t = Time();
			bool boundary = (slabAtMaxDepth+1)==slabEndAtMaxDepth && frontBoundary;
			LocalDepth d ; int o;
			for( d=maxDepth , o=slabAtMaxDepth ; d>=fullDepth ; d-- , o>>=1 )
			{
				if( d<=tree._maxDepth )
				{
					auto faceIndexFunctor = [&]( const TreeNode *node , typename HyperCube::Cube< Dim >::template Element< 2 > f )
					{
						HyperCube::Direction dir ; unsigned int coIndex;
						f.factor( dir , coIndex );
						int depth , offset[Dim];
						tree.depthAndOffset( node , depth , offset );
						LevelSetExtraction::Key< Dim > key = keyGenerator( depth , offset , f );
						if     ( dir==HyperCube::BACK  && ( (((unsigned int)offset[Dim-1]+0)<<(maxDepth-depth))<slabStartAtMaxDepth ) ) key[Dim-1] = keyGenerator.cornerIndex( maxDepth , slabStartAtMaxDepth );
						else if( dir==HyperCube::FRONT && ( (((unsigned int)offset[Dim-1]+1)<<(maxDepth-depth))>  slabEndAtMaxDepth ) ) key[Dim-1] = keyGenerator.cornerIndex( maxDepth , slabEndAtMaxDepth );
						return key;
					};
					if( InteriorSlab( d , o ) ) SetLevelSet( keyGenerator , faceIndexFunctor , tree , d , o , slabValues[d].sliceValues(o) , slabValues[d].sliceValues(o+1) , slabValues[d].xSliceValues(o) , slabValues[d].sliceScratch(o) , slabValues[d].sliceScratch(o+1) , slabValues[d].xSliceScratch(o) , vertexStream , polygonStream , polygonMesh , addBarycenter , flipOrientation );
				}
				if( !(o&1) && !boundary ) break;
			}
			stats.surfaceTime += Time()-t;
		};

		InitSlice( slabStartAtMaxDepth );
		InitSlab( slabStartAtMaxDepth , true );	// This needs to be done in case the slice wants to push iso-vertices down to the slab
		SetSliceValues( slabStartAtMaxDepth );
		SetSliceIso( slabStartAtMaxDepth );
		FinalizeSlice( slabStartAtMaxDepth );

		// Iterate over the slabs at the finest level
		for( unsigned int slab=slabStartAtMaxDepth ; slab<slabEndAtMaxDepth ; slab++)
		{
			// Need to init the slab before setting the slice (in case things need to be pushed down from the slice to the slab)
			InitSlice( slab+1 );
			if( slab!=slabStartAtMaxDepth ) InitSlab( slab , false );

			// Need to set the front slice corner values before computing iso-vertices
			SetSliceValues( slab+1 );

			// Compute the iso-vertices first, so that slice's iso-vertices are set last
			SetSlabIsoVertices( slab );
			SetSliceIso( slab+1 );

			// Now compute the iso-edges
			SetSlabIsoEdges( slab );

			FinalizeSlice( slab+1 );
			FinalizeSlab( slab );

			IsoSurface( slab );
		}

		if( pointEvaluator ) delete pointEvaluator;
		size_t badRootCount = _BadRootCount;
		if( badRootCount!=0 ) MK_WARN( "bad average roots: " , badRootCount );
		return stats;
	}
};

template< bool HasData , typename Real , typename Data > std::atomic< size_t > _LevelSetExtractor< HasData , Real , 3 , Data >::_BadRootCount;

template< typename Real >
struct LevelSetExtractor< Real , 3 >
{
	static const unsigned int Dim = 3;
	static const bool HasData = false;
	typedef unsigned char Data;
	typedef typename _LevelSetExtractor< HasData , Real , Dim , Data >::Stats Stats;
	typedef typename _LevelSetExtractor< HasData , Real , Dim , Data >::Vertex Vertex;
	typedef typename _LevelSetExtractor< HasData , Real , Dim , Data >::TreeNode TreeNode;
	using OutputVertexStream = typename _LevelSetExtractor< HasData , Real , Dim , Data >::OutputVertexStream;
	template< unsigned int WeightDegree > using DensityEstimator = typename _LevelSetExtractor< HasData , Real , Dim , Data >::template DensityEstimator< WeightDegree >;

	template< unsigned int WeightDegree , unsigned int ... FEMSigs >
	static Stats Extract( UIntPack< FEMSigs ... > , UIntPack< WeightDegree > , const FEMTree< Dim , Real >& tree , const DensityEstimator< WeightDegree > *densityWeights , const DenseNodeData< Real , UIntPack< FEMSigs ... > > &coefficients , Real isoValue , OutputVertexStream &vertexStream , OutputDataStream< std::vector< node_index_type > > &polygonStream , bool nonLinearFit , bool outputGradients , bool addBarycenter , bool polygonMesh , bool flipOrientation )
	{
		return Extract( UIntPack< FEMSigs ... >() , UIntPack< WeightDegree >() , tree , densityWeights , coefficients , isoValue , 0 , 0 , 1 , vertexStream , polygonStream , nonLinearFit , outputGradients , addBarycenter , polygonMesh , flipOrientation );
	}

	template< unsigned int WeightDegree , unsigned int ... FEMSigs >
	static Stats Extract( UIntPack< FEMSigs ... > , UIntPack< WeightDegree > , const FEMTree< Dim , Real >& tree , const DensityEstimator< WeightDegree >* densityWeights , const DenseNodeData< Real , UIntPack< FEMSigs ... > >& coefficients , Real isoValue , unsigned int slabDepth , unsigned int slabStart , unsigned int slabEnd , OutputVertexStream &vertexStream , OutputDataStream< std::vector< node_index_type > > &polygonStream , bool nonLinearFit , bool outputGradients , bool addBarycenter , bool polygonMesh , bool flipOrientation )
	{
		std::vector< std::vector< Real > > dValues;
		return Extract( UIntPack< FEMSigs ... >() , UIntPack< WeightDegree >() , tree , tree._maxDepth , densityWeights , coefficients , isoValue , slabDepth , slabStart , slabEnd , vertexStream , polygonStream , nonLinearFit , outputGradients , addBarycenter , polygonMesh , flipOrientation , NULL , NULL , dValues , dValues , false );
	}

	template< unsigned int WeightDegree , unsigned int ... FEMSigs >
	static Stats Extract( UIntPack< FEMSigs ... > , UIntPack< WeightDegree > , const FEMTree< Dim , Real >& tree , int maxKeyDepth , const DensityEstimator< WeightDegree >* densityWeights , const DenseNodeData< Real , UIntPack< FEMSigs ... > >& coefficients , Real isoValue , unsigned int slabDepth , unsigned int slabStart , unsigned int slabEnd , OutputVertexStream &vertexStream , OutputDataStream< std::vector< node_index_type > > &polygonStream , bool nonLinearFit , bool outputGradients , bool addBarycenter , bool polygonMesh , bool flipOrientation , const typename LevelSetExtractor< Real , Dim-1 , Point< Real , Dim-1 > >::TreeSliceValuesAndVertexPositions *backBoundary , const typename LevelSetExtractor< Real , Dim-1 , Point< Real , Dim-1 > >::TreeSliceValuesAndVertexPositions *frontBoundary , const std::vector< std::vector< Real > > &backDValues , const std::vector< std::vector< Real > > &frontDValues , bool copyTopology )
	{
		static const unsigned int DataSig = FEMDegreeAndBType< 0 , BOUNDARY_FREE >::Signature;
		const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > *data = NULL;
		Data zeroData = 0;
		return _LevelSetExtractor< HasData , Real , Dim , Data >::template Extract< WeightDegree , DataSig , OutputVertexStream , FEMSigs ... >( UIntPack< FEMSigs ... >() , UIntPack< WeightDegree >() , UIntPack< DataSig >() , tree , maxKeyDepth , densityWeights , data , coefficients , isoValue , slabDepth , slabStart , slabEnd , vertexStream , polygonStream , zeroData , nonLinearFit , outputGradients , addBarycenter , polygonMesh , flipOrientation , backBoundary , frontBoundary , backDValues , frontDValues , copyTopology );
	}
};

template< typename Real , typename Data >
struct LevelSetExtractor< Real , 3 , Data >
{
	static const unsigned int Dim = 3;
	static const bool HasData = true;
	typedef typename _LevelSetExtractor< HasData , Real , Dim , Data >::Stats Stats;
	typedef typename _LevelSetExtractor< HasData , Real , Dim , Data >::Vertex Vertex;
	typedef typename _LevelSetExtractor< HasData , Real , Dim , Data >::TreeNode TreeNode;
	using OutputVertexStream = typename _LevelSetExtractor< HasData , Real , Dim , Data >::OutputVertexStream;
	template< unsigned int WeightDegree > using DensityEstimator = typename _LevelSetExtractor< HasData , Real , Dim , Data >::template DensityEstimator< WeightDegree >;

	template< unsigned int WeightDegree , unsigned int DataSig , unsigned int ... FEMSigs >
	static Stats Extract( UIntPack< FEMSigs ... > , UIntPack< WeightDegree > , UIntPack< DataSig > , const FEMTree< Dim , Real >& tree , const DensityEstimator< WeightDegree > *densityWeights , const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > *data , const DenseNodeData< Real , UIntPack< FEMSigs ... > > &coefficients , Real isoValue , OutputVertexStream &vertexStream , OutputDataStream< std::vector< node_index_type > > &polygonStream , const Data &zeroData , bool nonLinearFit , bool outputGradients , bool addBarycenter , bool polygonMesh , bool flipOrientation )
	{
		return Extract( UIntPack< FEMSigs ... >() , UIntPack< WeightDegree >() , UIntPack< DataSig >() , tree , densityWeights , data , coefficients , isoValue , 0 , 0 , 1 , vertexStream , polygonStream , zeroData , nonLinearFit , outputGradients , addBarycenter , polygonMesh , flipOrientation );
	}

	template< unsigned int WeightDegree , unsigned int DataSig , unsigned int ... FEMSigs >
	static Stats Extract( UIntPack< FEMSigs ... > , UIntPack< WeightDegree > , UIntPack< DataSig > , const FEMTree< Dim , Real >& tree , const DensityEstimator< WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > >* data , const DenseNodeData< Real , UIntPack< FEMSigs ... > >& coefficients , Real isoValue , unsigned int slabDepth , unsigned int slabStart , unsigned int slabEnd , OutputVertexStream &vertexStream , OutputDataStream< std::vector< node_index_type > > &polygonStream , const Data &zeroData , bool nonLinearFit , bool outputGradients , bool addBarycenter , bool polygonMesh , bool flipOrientation )
	{
		std::vector< std::vector< Real >  > dValues;
		return Extract( UIntPack< FEMSigs ... >() , UIntPack< WeightDegree >() , UIntPack< DataSig >() , tree , tree._maxDepth , densityWeights , data , coefficients , isoValue , slabDepth , slabStart , slabEnd , vertexStream , polygonStream , zeroData , nonLinearFit , outputGradients , addBarycenter , polygonMesh , flipOrientation , NULL , NULL , dValues , dValues , false );
	}

	template< unsigned int WeightDegree , unsigned int DataSig , unsigned int ... FEMSigs >
	static Stats Extract( UIntPack< FEMSigs ... > , UIntPack< WeightDegree > , UIntPack< DataSig > , const FEMTree< Dim , Real >& tree , int maxKeyDepth , const DensityEstimator< WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > >* data , const DenseNodeData< Real , UIntPack< FEMSigs ... > >& coefficients , Real isoValue , unsigned int slabDepth , unsigned int slabStart , unsigned int slabEnd , OutputVertexStream &vertexStream , OutputDataStream< std::vector< node_index_type > > &polygonStream , const Data &zeroData , bool nonLinearFit , bool outputGradients , bool addBarycenter , bool polygonMesh , bool flipOrientation , const typename LevelSetExtractor< Real , Dim-1 , Point< Real , Dim-1 > >::TreeSliceValuesAndVertexPositions *backBoundary , const typename LevelSetExtractor< Real , Dim-1 , Point< Real , Dim-1 > >::TreeSliceValuesAndVertexPositions *frontBoundary , const std::vector< std::vector< Real > > &backDValues , const std::vector< std::vector< Real > > &frontDValues , bool copyTopology )
	{
		return _LevelSetExtractor< HasData , Real , Dim , Data >::template Extract< WeightDegree , DataSig , OutputVertexStream , FEMSigs ... >( UIntPack< FEMSigs ... >() , UIntPack< WeightDegree >() , UIntPack< DataSig >() , tree , maxKeyDepth , densityWeights , data , coefficients , isoValue , slabDepth , slabStart , slabEnd , vertexStream , polygonStream , zeroData , nonLinearFit , outputGradients , addBarycenter , polygonMesh , flipOrientation , backBoundary , frontBoundary , backDValues , frontDValues , copyTopology );
	}
};
