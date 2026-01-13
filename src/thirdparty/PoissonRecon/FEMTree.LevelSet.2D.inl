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

// Specialized level-set curve extraction
template< bool HasData , typename Real , typename Data >
struct _LevelSetExtractor< HasData , Real , 2 , Data >
{
	static const unsigned int Dim = 2;
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

	using LocalDepth = typename FEMTree< Dim , Real >::LocalDepth;
	using LocalOffset = typename FEMTree< Dim , Real >::LocalOffset;
	using ConstOneRingNeighborKey = typename FEMTree< Dim , Real >::ConstOneRingNeighborKey;
	using ConstOneRingNeighbors = typename FEMTree< Dim , Real >::ConstOneRingNeighbors;
	using TreeNode = RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >;
	template< unsigned int WeightDegree > using DensityEstimator = typename FEMTree< Dim , Real >::template DensityEstimator< WeightDegree >;
	template< typename FEMSigPack , unsigned int PointD > using _Evaluator = typename FEMTree< Dim , Real >::template _Evaluator< FEMSigPack , PointD >;

	using Key = LevelSetExtraction::Key< Dim >;
	using IsoEdge = LevelSetExtraction::IsoEdge< Dim >;
	template< unsigned int D , unsigned int ... Ks >
	using HyperCubeTables = typename LevelSetExtraction::template HyperCubeTables< D , Ks ... >;

	////////////////
	// FaceEdges //
	////////////////
	struct FaceEdges
	{
		IsoEdge edges[2];
		int count;
		FaceEdges( void ) : count(-1){}
	};

	/////////////////
	// SliceValues //
	/////////////////
	struct SliceValues
	{
		struct Scratch
		{
			using FKeyValues = std::vector< std::vector< std::pair< Key , std::vector< IsoEdge > > > >;
			using EKeyValues = std::vector< std::vector< std::pair< Key , node_index_type > > >;
			using VKeyValues = std::vector< std::vector< std::pair< Key , Key > > >;

			FKeyValues fKeyValues;
			EKeyValues eKeyValues;
			VKeyValues vKeyValues;

#ifdef SANITIZED_PR
			Pointer( std::atomic< char > ) cSet;
			Pointer( std::atomic< char > ) eSet;
#else // !SANITIZED_PR
			Pointer( char ) cSet;
			Pointer( char ) eSet;
#endif // SANITIZED_PR

			Scratch( void )
			{
				vKeyValues.resize( ThreadPool::NumThreads() );
				eKeyValues.resize( ThreadPool::NumThreads() );
				fKeyValues.resize( ThreadPool::NumThreads() );
#ifdef SANITIZED_PR
				cSet = NullPointer( std::atomic< char > );
				eSet = NullPointer( std::atomic< char > );
#else // !SANITIZED_PR
				cSet = NullPointer( char );
				eSet = NullPointer( char );
#endif // SANITIZED_PR
			}

			~Scratch( void )
			{
#ifdef SANITIZED_PR
				DeletePointer( cSet );
				DeletePointer( eSet );
#else // !SANITIZED_PR
				FreePointer( cSet );
				FreePointer( eSet );
#endif // SANITIZED_PR
			}

			void reset( const LevelSetExtraction::FullCellIndexData< Dim > &cellIndices )
			{
				for( size_t i=0 ; i<vKeyValues.size() ; i++ ) vKeyValues[i].clear();
				for( size_t i=0 ; i<eKeyValues.size() ; i++ ) eKeyValues[i].clear();
				for( size_t i=0 ; i<fKeyValues.size() ; i++ ) fKeyValues[i].clear();
#ifdef SANITIZED_PR
				DeletePointer( cSet );
				DeletePointer( eSet );
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
#else // !SANITIZED_PR
				FreePointer( cSet );
				FreePointer( eSet );
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
#endif // SANITIZED_PR
			}
		};

		// A data-structure for taking the a node index and the local index of a cell within the node and returning the global index for that cell
		LevelSetExtraction::FullCellIndexData< Dim > cellIndices;

		// A table taking a node index and returning the associated marching squares index
		Pointer( char ) mcIndices;

		// Tables taking a corner index and returning the value (and possibly the gradient) at the corresponding location
		Pointer( Real ) cornerValues ; Pointer( Point< Real , Dim > ) cornerGradients;

		// A table taking an edge index and returning the associated key
		Pointer( Key ) edgeKeys;

		// A table taking a face index and returning the associated key
		Pointer( FaceEdges ) faceEdges;

		// A map taking a key for a face and returning the iso-edges within that face
		LevelSetExtraction::KeyMap< Dim , std::vector< IsoEdge > > faceEdgeMap;
		// A map taking a key for an edge and returning the index of the iso-vertex along that edge
		LevelSetExtraction::KeyMap< Dim , node_index_type > edgeVertexMap;
		// A map linking an iso-vertex to its pair
		LevelSetExtraction::KeyMap< Dim , Key > vertexPairMap;

		SliceValues( void )
		{
			cornerValues = NullPointer( Real ) ; cornerGradients = NullPointer( Point< Real , Dim > );
			edgeKeys = NullPointer( Key );
			faceEdges = NullPointer( FaceEdges );
			mcIndices = NullPointer( char );
		}

		~SliceValues( void ){ clear(); }

		void clear( void )
		{
			cellIndices.clear();
			FreePointer( cornerValues ) ; FreePointer( cornerGradients );
			DeletePointer( edgeKeys );
			DeletePointer( faceEdges );
			FreePointer( mcIndices );
		}

		void read( BinaryStream &stream )
		{
			clear();

			cellIndices.read( stream );

			if( cellIndices.size() )
			{
				mcIndices = AllocPointer< char >( cellIndices.size() );
				if( !stream.read( mcIndices , cellIndices.size() ) ) MK_THROW( "Failed to read mc indices" );
			}
			if( cellIndices.counts[0] )
			{
				cornerValues = AllocPointer< Real >( cellIndices.counts[0] );
				if( !stream.read( cornerValues , cellIndices.counts[0] ) ) MK_THROW( "Failed to read corner values" );
				char hasCornerGradients;
				if( !stream.read( hasCornerGradients ) ) MK_THROW( "Could not read corner gradient state" );
				if( hasCornerGradients )
				{
					cornerGradients = AllocPointer< Point< Real , Dim > >( cellIndices.counts[0] );
					if( !stream.read( cornerGradients , cellIndices.counts[0] ) ) MK_THROW( "Could not read corner gradients" );
				}
			}

			if( cellIndices.counts[1] )
			{
				edgeKeys = NewPointer< Key >( cellIndices.counts[1] );
				if( !stream.read( edgeKeys , cellIndices.counts[1]) ) MK_THROW( "Could not read edge keys" );
			}

			if( cellIndices.counts[2] )
			{
				faceEdges = NewPointer< FaceEdges >( cellIndices.counts[2] );
				if( !stream.read( faceEdges , cellIndices.counts[2] ) ) MK_THROW( "Could not read face edges" );
			}

			auto ReadIsoEdgeVector = [&]( BinaryStream &stream , std::vector< IsoEdge > &edges )
			{
				size_t sz;
				if( !stream.read( sz ) ) MK_THROW( "Could not read iso-edge vector size" );
				edges.resize( sz );
				if( sz && !stream.read( GetPointer( edges ) , sz ) ) MK_THROW( "Could not read iso-edges" );
			};
			{
				size_t sz;
				if( !stream.read( sz ) ) MK_THROW( "Could not read face-edge-map size" );
				for( unsigned int i=0 ; i<sz ; i++ )
				{
					Key key;
					if( !stream.read( key ) ) MK_THROW( "Could not read face-edge-map key" );
					ReadIsoEdgeVector( stream , faceEdgeMap[key] );
				}
			}
			{
				size_t sz;
				if( !stream.read( sz ) ) MK_THROW( "Could not read edge-vertex-map size" );
				for( unsigned int i=0 ; i<sz ; i++ )
				{
					Key key;
					if( !stream.read( key ) ) MK_THROW( "Could not read edge-vertex-map key" );
					if( !stream.read( edgeVertexMap[key] ) ) MK_THROW( "Could not read edge-vertex-map value" );
				}
			}
			{
				size_t sz;
				if( !stream.read( sz ) ) MK_THROW( "Could not read vertex-pair-map size" );
				for( unsigned int i=0 ; i<sz ; i++ )
				{
					Key key;
					if( !stream.read( key ) ) MK_THROW( "Could not read vertex-pair-map key" );
					if( !stream.read( vertexPairMap[key] ) ) MK_THROW( "Could not read vertex-pair-map value" );
				}
			}
		}

		void write( BinaryStream &stream , bool serialize ) const
		{
			cellIndices.write( stream );

			if( cellIndices.size() ) stream.write( mcIndices , cellIndices.size() );
			if( cellIndices.counts[0] )
			{
				stream.write( cornerValues , cellIndices.counts[0] );
				char hasCornerGradients = cornerGradients==NullPointer( char ) ? 0 : 1;
				stream.write( hasCornerGradients );
				if( hasCornerGradients ) stream.write( cornerGradients , cellIndices.counts[0] );
			}
			if( cellIndices.counts[1] ) stream.write(  edgeKeys , cellIndices.counts[1] );
			if( cellIndices.counts[2] ) stream.write( faceEdges , cellIndices.counts[2] );

			auto WriteIsoEdgeVector = [&]( BinaryStream &stream , const std::vector< IsoEdge > &edges )
			{
				size_t sz = edges.size();
				stream.write( sz );
				if( sz ) stream.write( GetPointer( edges ) , sz );
			};

			auto SerializeFaceEdgeMap = [&]( size_t &sz )
			{
				using map = LevelSetExtraction::KeyMap< Dim , std::vector< IsoEdge > >;

				sz = 0;
				for( typename map::const_iterator iter=faceEdgeMap.begin() ; iter!=faceEdgeMap.end() ; iter++ )
					sz += sizeof( typename LevelSetExtraction::Key< Dim > ) + sizeof( size_t ) + sizeof( IsoEdge ) * iter->second.size();

				Pointer( char ) buffer = NewPointer< char >( sz );
				Pointer( char ) _buffer = buffer;
				for( typename map::const_iterator iter=faceEdgeMap.begin() ; iter!=faceEdgeMap.end() ; iter++ )
				{
					memcpy( _buffer , &iter->first , sizeof( typename LevelSetExtraction::Key< Dim > ) ) ; _buffer += sizeof( typename LevelSetExtraction::Key< Dim > );
					size_t num = iter->second.size();
					memcpy( _buffer , &num , sizeof( size_t ) ) ; _buffer += sizeof( size_t );
					if( num ) memcpy( _buffer , &iter->second[0] , sizeof(IsoEdge) * iter->second.size() ) ; _buffer += sizeof(IsoEdge) * iter->second.size();
				}
				return buffer;
			};

			{
				using map = LevelSetExtraction::KeyMap< Dim , std::vector< IsoEdge > >;
				if( serialize )
				{
					size_t sz = faceEdgeMap.size();
					stream.write( sz );
					if( sz )
					{
						Pointer( char ) buffer = SerializeFaceEdgeMap( sz );
						stream.write( buffer , sz );
						DeletePointer( buffer );
					}
				}
				else
				{
					size_t sz = faceEdgeMap.size();
					stream.write( sz );
					for( typename map::const_iterator iter=faceEdgeMap.begin() ; iter!=faceEdgeMap.end() ; iter++ )
					{
						stream.write( iter->first );
						WriteIsoEdgeVector( stream , iter->second );
					}
				}
			}

			{
				using map = LevelSetExtraction::KeyMap< Dim , node_index_type >;
				if( serialize )
				{
					size_t sz = edgeVertexMap.size();
					stream.write( sz );
					if( sz )
					{
						size_t elementSize = sizeof( typename LevelSetExtraction::Key< Dim > ) + sizeof( node_index_type );
						Pointer( char ) buffer = NewPointer< char >( edgeVertexMap.size() * elementSize );
						{
							Pointer( char ) _buffer = buffer;
							for( typename map::const_iterator iter=edgeVertexMap.begin() ; iter!=edgeVertexMap.end() ; iter++ )
							{
								memcpy( _buffer , &iter->first , sizeof( typename LevelSetExtraction::Key< Dim >  ) ) ; _buffer += sizeof( typename LevelSetExtraction::Key< Dim > );
								memcpy( _buffer , &iter->second , sizeof( node_index_type ) ) ; _buffer += sizeof( node_index_type );
							}
						}
						stream.write( buffer , edgeVertexMap.size() * elementSize );
						DeletePointer( buffer );
					}
				}
				else
				{
					size_t sz = edgeVertexMap.size();
					stream.write( sz );
					for( typename map::const_iterator iter=edgeVertexMap.begin() ; iter!=edgeVertexMap.end() ; iter++ )
					{
						stream.write( iter->first );
						stream.write( iter->second );
					}
				}
			}

			{
				using map = LevelSetExtraction::KeyMap< Dim , Key >;
				if( serialize )
				{
					std::vector< std::pair< typename LevelSetExtraction::Key< Dim > , Key > > pairs;
					pairs.reserve( vertexPairMap.size() );
					for( typename map::const_iterator iter=vertexPairMap.begin() ; iter!=vertexPairMap.end() ; iter++ )
						pairs.push_back( std::pair< typename LevelSetExtraction::Key< Dim > , Key >( iter->first , iter->second ) );
					stream.write( pairs );
				}
				else
				{
					size_t sz = vertexPairMap.size();
					stream.write( sz );
					for( typename map::const_iterator iter=vertexPairMap.begin() ; iter!=vertexPairMap.end() ; iter++ )
					{
						stream.write( iter->first );
						stream.write( iter->second );
					}
				}
			}
		}

		void setFromScratch( typename Scratch::EKeyValues &scratch )
		{
			for( node_index_type i=0 ; i<(node_index_type)scratch.size() ; i++ )
			{
				for( int j=0 ; j<scratch[i].size() ; j++ ) edgeVertexMap[ scratch[i][j].first ] = scratch[i][j].second;
				scratch[i].clear();
			}
		}

		void setFromScratch( typename Scratch::VKeyValues &scratch )
		{
			for( node_index_type i=0 ; i<(node_index_type)scratch.size() ; i++ )
			{
				for( int j=0 ; j<scratch[i].size() ; j++ )
				{
					// Note that both vertices are added as keys
					vertexPairMap[ scratch[i][j].first ] = scratch[i][j].second;
					vertexPairMap[ scratch[i][j].second ] = scratch[i][j].first;
				}
				scratch[i].clear();
			}
		}

		void setFromScratch( typename Scratch::FKeyValues &scratch )
		{
			for( node_index_type i=0 ; i<(node_index_type)scratch.size() ; i++ )
			{
				for( int j=0 ; j<scratch[i].size() ; j++ )
				{
					std::vector< IsoEdge > &faceEdges = faceEdgeMap[ scratch[i][j].first ];
					faceEdges.insert( faceEdges.end() , scratch[i][j].second.begin() , scratch[i][j].second.end() );
				}
				scratch[i].clear();
			}
		}

		void reset( bool computeGradients )
		{
			faceEdgeMap.clear() , edgeVertexMap.clear() , vertexPairMap.clear();

			FreePointer( mcIndices );
			if( cellIndices.size()>0 ) mcIndices = AllocPointer< char >( cellIndices.size() );

			FreePointer( cornerValues ) ; FreePointer( cornerGradients );
			if( cellIndices.counts[0]>0 )
			{
				cornerValues = AllocPointer< Real >( cellIndices.counts[0] );
				if( computeGradients ) cornerGradients = AllocPointer< Point< Real , Dim > >( cellIndices.counts[0] );
			}

			DeletePointer( edgeKeys );
			edgeKeys = NewPointer< Key >( cellIndices.counts[1] );

			DeletePointer( faceEdges );
			faceEdges = NewPointer< FaceEdges >( cellIndices.counts[2] );

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
	};

	struct TreeSliceValuesAndVertexPositions
	{
		const FEMTree< Dim , Real > sliceTree;
		std::vector< SliceValues > sliceValues;
		// [WARNING] Should really be using Vertex instead of Point< Real , Dim >
		std::vector< Point< Real , Dim > > vertexPositions;

		TreeSliceValuesAndVertexPositions( void ) : sliceTree(NULL){}
		TreeSliceValuesAndVertexPositions( BinaryStream &stream  , XForm< Real , Dim+1 > &xForm , size_t blockSize ) : sliceTree( stream , blockSize )
		{
			stream.read( xForm );

			size_t sz;
			if( !stream.read( sz ) ) MK_THROW( "Could not read slice count" );
			sliceValues.resize( sz );
			for( unsigned int i=0 ; i<sliceValues.size() ; i++ ) sliceValues[i].read( stream );
			if( !stream.read( sz ) ) MK_THROW( "Could not read iso-vertex count" );
			if( sz )
			{
				vertexPositions.resize( sz );
				if( sz && !stream.read( GetPointer( vertexPositions ) , sz ) ) MK_THROW( "Could not read iso-vertex positions" );
			}
		}

		std::vector< Key > vertexKeys( void ) const
		{
			std::vector< Key > keys( vertexPositions.size() );
			for( unsigned int i=0 ; i<sliceValues.size() ; i++ )
				for( typename LevelSetExtraction::KeyMap< Dim , node_index_type >::const_iterator iter=sliceValues[i].edgeVertexMap.cbegin() ; iter!=sliceValues[i].edgeVertexMap.cend() ; iter++ )
				{
					if( iter->second>=(node_index_type)vertexPositions.size() ) MK_THROW( "Unexpected vertex index: " , iter->second , " <= " , vertexPositions.size() );
					keys[iter->second] = iter->first;
				}
			return keys;
		};

		static void Write( BinaryStream &stream , const FEMTree< Dim , Real > *sliceTree , XForm< Real , Dim+1 > xForm , const std::vector< SliceValues > &sliceValues , const std::vector< Point< Real , Dim > > &vertices , bool serialize )
		{
			sliceTree->write( stream , serialize );
			stream.write( xForm );

			size_t sz = sliceValues.size();
			stream.write( sz );
			for( unsigned int i=0 ; i<sliceValues.size() ; i++ ) sliceValues[i].write( stream , serialize );

			stream.write( vertices );
		}

		void write( BinaryStream &stream , XForm< Real , Dim+1 > xForm , bool serialize ) const
		{
			Write( stream , &sliceTree , xForm , sliceValues , vertexPositions , serialize );
		}
	};


	template< unsigned int ... FEMSigs >
	static void SetSCornerValues( const FEMTree< Dim , Real >& tree , ConstPointer( Real ) coefficients , ConstPointer( Real ) coarseCoefficients , Real isoValue , LocalDepth depth , LocalDepth fullDepth , std::vector< SliceValues >& sliceValues , std::vector< typename SliceValues::Scratch > &scratchValues , const _Evaluator< UIntPack< FEMSigs ... > , 1 >& evaluator )
	{
		static const unsigned int FEMDegrees[] = { FEMSignature< FEMSigs >::Degree ... };
		SliceValues& sValues = sliceValues[depth];
#ifdef SANITIZED_PR
		Pointer( std::atomic< char > ) cornerSet = scratchValues[depth].cSet;
#else // !SANITIZED_PR
		Pointer( char ) cornerSet = scratchValues[depth].cSet;
#endif // SANITIZED_PR
		bool useBoundaryEvaluation = false;
		for( int d=0 ; d<Dim ; d++ ) if( FEMDegrees[d]==0 || ( FEMDegrees[d]==1 && sValues.cornerGradients ) ) useBoundaryEvaluation = true;
		std::vector< ConstPointSupportKey< UIntPack< FEMSignature< FEMSigs >::Degree ... > > > neighborKeys( ThreadPool::NumThreads() );
		std::vector< ConstCornerSupportKey< UIntPack< FEMSignature< FEMSigs >::Degree ... > > > bNeighborKeys( ThreadPool::NumThreads() );
		if( useBoundaryEvaluation ) for( size_t i=0 ; i<neighborKeys.size() ; i++ ) bNeighborKeys[i].set( tree._localToGlobal( depth ) );
		else                        for( size_t i=0 ; i<neighborKeys.size() ; i++ )  neighborKeys[i].set( tree._localToGlobal( depth ) );
		ThreadPool::ParallelFor( tree._sNodesBegin(depth) , tree._sNodesEnd(depth) , [&]( unsigned int thread , size_t i )
		{
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) )
			{
				Real squareValues[ HyperCube::Cube< Dim >::template ElementNum< 0 >() ];
				ConstPointSupportKey< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& neighborKey = neighborKeys[ thread ];
				ConstCornerSupportKey< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& bNeighborKey = bNeighborKeys[ thread ];
				TreeNode* leaf = tree._sNodes.treeNodes[i];
				// Iterate over all leaf nodes
				if( !IsActiveNode< Dim >( leaf->children ) )
				{
					const typename LevelSetExtraction::FullCellIndexData< Dim >::template CellIndices<0> &cIndices = sValues.cellIndices.template indices<0>( leaf );

					bool isInterior = tree._isInteriorlySupported( UIntPack< FEMSignature< FEMSigs >::Degree ... >() , leaf->parent );
					if( useBoundaryEvaluation ) bNeighborKey.getNeighbors( leaf );
					else                         neighborKey.getNeighbors( leaf );

					// Iterate over the corners of the cell
					for( typename HyperCube::Cube< Dim >::template Element< 0 > c ; c<HyperCube::Cube< Dim >::template ElementNum< 0 >() ; c++ )
					{
						// Grab the global corner index, and if its value hasn't been set yet get the values (and gradients) and store them
						node_index_type vIndex = cIndices[c.index];
						if( !cornerSet[vIndex] )
						{
							if( sValues.cornerGradients )
							{
								CumulativeDerivativeValues< Real , Dim , 1 > p;
								if( useBoundaryEvaluation ) p = tree.template _getCornerValues< Real , 1 >( bNeighborKey , leaf , c.index , coefficients , coarseCoefficients , evaluator , tree._maxDepth , isInterior );
								else                        p = tree.template _getCornerValues< Real , 1 >(  neighborKey , leaf , c.index , coefficients , coarseCoefficients , evaluator , tree._maxDepth , isInterior );
								sValues.cornerValues[vIndex] = p[0] , sValues.cornerGradients[vIndex] = Point< Real , Dim >( p[1] , p[2] );
							}
							else
							{
								if( useBoundaryEvaluation ) sValues.cornerValues[vIndex] = tree.template _getCornerValues< Real , 0 >( bNeighborKey , leaf , c.index , coefficients , coarseCoefficients , evaluator , tree._maxDepth , isInterior )[0];
								else                        sValues.cornerValues[vIndex] = tree.template _getCornerValues< Real , 0 >(  neighborKey , leaf , c.index , coefficients , coarseCoefficients , evaluator , tree._maxDepth , isInterior )[0];
							}
							cornerSet[vIndex] = 1;
						}
						squareValues[c.index] = sValues.cornerValues[ vIndex ];

						// Copy from the finer depth to the coarser depth
						TreeNode* node = leaf;
						LocalDepth _depth = depth;
						// Iterate to the parents as long as they contain the corner
						while( _depth>fullDepth && tree._isValidSpaceNode( node->parent ) && (node-node->parent->children)==c.index )
						{
							node = node->parent , _depth--;
							SliceValues& _sValues = sliceValues[_depth];
#ifdef SANITIZED_PR
							Pointer( std::atomic< char > ) _cornerSet = scratchValues[_depth].cSet;
#else // !SANITIZED_PR
							Pointer( char ) _cornerSet = scratchValues[_depth].cSet;
#endif // SANITIZED_PR
							const typename LevelSetExtraction::FullCellIndexData< Dim >::template CellIndices<0> &_cIndices = _sValues.cellIndices.template indices<0>( node );
							node_index_type _vIndex = _cIndices[c.index];
							_sValues.cornerValues[_vIndex] = sValues.cornerValues[vIndex];
							if( _sValues.cornerGradients ) _sValues.cornerGradients[_vIndex] = sValues.cornerGradients[vIndex];
							_cornerSet[_vIndex] = 1;
						}
					}

					// Set the marching squares index for the face
					sValues.mcIndices[ i - sValues.cellIndices.nodeOffset ] = HyperCube::Cube< Dim >::MCIndex( squareValues , isoValue );
				}
			}
		}
		);
	}


	template< unsigned int WeightDegree , unsigned int DataSig , typename VertexStream >
	static void SetIsoVertices
	(
		const LevelSetExtraction::KeyGenerator< Dim > &keyGenerator ,
		const FEMTree< Dim , Real >& tree ,
		bool nonLinearFit ,
		bool outputGradients ,
		typename FEMIntegrator::template PointEvaluator< IsotropicUIntPack< Dim , DataSig > , ZeroUIntPack< Dim > > *pointEvaluator ,
		const DensityEstimator< WeightDegree > *densityWeights ,
		const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > >* data ,
		Real isoValue ,
		LocalDepth depth ,
		LocalDepth fullDepth ,
		VertexStream &vertexStream ,
		std::vector< SliceValues > &sliceValues ,
		std::vector< typename SliceValues::Scratch > &scratchValues ,
		const Data &zeroData
	)
	{
		auto _EdgeIndex = [&]( const TreeNode *node , typename HyperCube::Cube< Dim >::template Element< 1 > e )
		{
			int depth , offset[Dim];
			tree.depthAndOffset( node , depth , offset );
			return keyGenerator( depth , offset , e );
		};
		static const unsigned int DataDegree = FEMSignature< DataSig >::Degree;

		SliceValues &sValues = sliceValues[depth];
		typename SliceValues::Scratch &scValues = scratchValues[depth];
		// [WARNING] In the case Degree=2, these two keys are the same, so we don't have to maintain them separately.
		std::vector< ConstOneRingNeighborKey > neighborKeys( ThreadPool::NumThreads() );
		std::vector< ConstPointSupportKey< IsotropicUIntPack< Dim , WeightDegree > > > weightKeys( ThreadPool::NumThreads() );
		std::vector< ConstPointSupportKey< IsotropicUIntPack< Dim , DataDegree > > > dataKeys( ThreadPool::NumThreads() );
		for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( tree._localToGlobal( depth ) ) , weightKeys[i].set( tree._localToGlobal( depth ) ) , dataKeys[i].set( tree._localToGlobal( depth ) );

		ThreadPool::ParallelFor( tree._sNodesBegin(depth) , tree._sNodesEnd(depth) , [&]( unsigned int thread , size_t i )
		{
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) )
			{
				ConstOneRingNeighborKey& neighborKey = neighborKeys[ thread ];
				ConstPointSupportKey< IsotropicUIntPack< Dim , WeightDegree > >& weightKey = weightKeys[ thread ];
				ConstPointSupportKey< IsotropicUIntPack< Dim , DataDegree > >& dataKey = dataKeys[ thread ];
				TreeNode* leaf = tree._sNodes.treeNodes[i];

				// Iterate over all leaf nodes in the tree
				if( !IsActiveNode< Dim >( leaf->children ) )
				{
					node_index_type idx = (node_index_type)i - sValues.cellIndices.nodeOffset;
					const typename LevelSetExtraction::FullCellIndexData< Dim >::template CellIndices<1> eIndices = sValues.cellIndices.template indices<1>( leaf );

					// Check if the face has zero-crossings
					if( HyperCube::Cube< Dim >::HasMCRoots( sValues.mcIndices[idx] ) )
					{
						neighborKey.getNeighbors( leaf );
						if( densityWeights ) weightKey.getNeighbors( leaf );
						if constexpr( HasData ) if( data ) dataKey.getNeighbors( leaf );

						// Check if the individual edges have zero-crossings
						for( typename HyperCube::Cube< Dim >::template Element< 1 > e ; e<HyperCube::Cube< Dim >::template ElementNum< 1 >() ; e++ )
							if( HyperCube::Cube< 1 >::HasMCRoots( HyperCube::Cube< Dim >::ElementMCIndex( e , sValues.mcIndices[idx] ) ) )
							{
								node_index_type vIndex = eIndices[e.index];
#ifdef SANITIZED_PR
								std::atomic< char > &edgeSet = scValues.eSet[vIndex];
#else // !SANITIZED_PR
								volatile char &edgeSet = scValues.eSet[vIndex];
#endif // SANITIZED_PR
								// If the edge hasn't been set already (e.g. either by another thread or from a finer resolution)
								if( !edgeSet )
								{
									Vertex vertex;
									Key key = _EdgeIndex( leaf , e );
									GetIsoVertex< WeightDegree , DataSig >( tree , nonLinearFit , outputGradients , pointEvaluator , densityWeights , data , isoValue , weightKey , dataKey , leaf , e , sValues , vertex , zeroData );
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

									if( stillOwner )	// If this edge is the one generating the iso-vertex
									{
										sValues.edgeKeys[ vIndex ] = key;
										scValues.eKeyValues[ thread ].push_back( std::pair< Key , node_index_type >( key , hashed_vertex.first ) );

										// We only need to pass the iso-vertex down if the edge it lies on is adjacent to a coarser leaf
										const typename HyperCube::Cube< Dim >::template Element< Dim > *f = HyperCubeTables< Dim , 1 , Dim >::OverlapElements[e.index];
										// Note that this is a trivial loop of size 1
										for( int k=0 ; k<HyperCubeTables< Dim , 1 , Dim >::OverlapElementNum ; k++ )
										{
											TreeNode *node = leaf;
											LocalDepth _depth = depth;
											while( _depth>fullDepth && tree._isValidSpaceNode( node->parent ) && HyperCubeTables< Dim , Dim , 0 >::Overlap[f[k].index][(unsigned int)(node-node->parent->children) ] ) 
											{
												node = node->parent , _depth--;
												typename SliceValues::Scratch &_scValues = scratchValues[_depth];
												_scValues.eKeyValues[ thread ].push_back( std::pair< Key , node_index_type >( key , hashed_vertex.first ) );
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

	static void CopyFinerSliceIsoEdgeKeys( const FEMTree< Dim , Real >& tree , LocalDepth depth , LocalDepth fullDepth , std::vector< SliceValues >& sliceValues , std::vector< typename SliceValues::Scratch > &scratchValues )
	{
		SliceValues& pSliceValues = sliceValues[depth  ];
		SliceValues& cSliceValues = sliceValues[depth+1];
		typename SliceValues::Scratch &pScratchSliceValues = scratchValues[depth  ];
		typename SliceValues::Scratch &cScratchSliceValues = scratchValues[depth+1];
		LevelSetExtraction::FullCellIndexData< Dim > &pCellIndices = pSliceValues.cellIndices;
		LevelSetExtraction::FullCellIndexData< Dim > &cCellIndices = cSliceValues.cellIndices;
		ThreadPool::ParallelFor( tree._sNodesBegin(depth) , tree._sNodesEnd(depth) , [&]( unsigned int thread , size_t i )
		{
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) ) if( IsActiveNode< Dim >( tree._sNodes.treeNodes[i]->children ) )
			{
				// Get the mapping from local edge indices to global edge indices
				typename LevelSetExtraction::FullCellIndexData< Dim >::template CellIndices<1> &pIndices = pCellIndices.template indices<1>( (node_index_type)i );
				// Copy the edges that overlap the coarser edges
				for( typename HyperCube::Cube< Dim >::template Element< 1 > e ; e<HyperCube::Cube< Dim >::template ElementNum< 1 >() ; e++ )
				{
					// The global (coarse) edge index
					node_index_type pIndex = pIndices[e.index];
					if( !pScratchSliceValues.eSet[ pIndex ] )
					{
						// The corner indices incident on the edeg
						const typename HyperCube::Cube< Dim >::template Element< 0 > *c = HyperCubeTables< Dim , 1 , 0 >::OverlapElements[e.index];
						// [SANITY CHECK]
						//						if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i]->children + c[0].index )!=tree._isValidSpaceNode( tree._sNodes.treeNodes[i]->children + c[1].index ) ) MK_THROW( "Finer edges should both be valid or invalid" );
						// Can only copy edge information from the finer nodes incident on the edge if they are valid (note since we refine in broods, we can't have one child in and the other out)
						if( !tree._isValidSpaceNode( tree._sNodes.treeNodes[i]->children + c[0].index ) || !tree._isValidSpaceNode( tree._sNodes.treeNodes[i]->children + c[1].index ) ) continue;


						// The global (fine) edge indices
						node_index_type cIndex1 = cCellIndices.template indices<1>( tree._sNodes.treeNodes[i]->children + c[0].index )[e.index];
						node_index_type cIndex2 = cCellIndices.template indices<1>( tree._sNodes.treeNodes[i]->children + c[1].index )[e.index];

						// If only one of the finer edges has a zero-crossing, then the coarse edge will as well
						if( cScratchSliceValues.eSet[cIndex1] != cScratchSliceValues.eSet[cIndex2] )
						{
							Key key;
							if( cScratchSliceValues.eSet[cIndex1] ) key = cSliceValues.edgeKeys[cIndex1];
							else                                    key = cSliceValues.edgeKeys[cIndex2];
							pSliceValues.edgeKeys[pIndex] = key;
							pScratchSliceValues.eSet[pIndex] = 1;
						}
						// If both of the finer edges have a zero-crossing, those will form a pair (but the coarser edge will not have a zero crossing)
						else if( cScratchSliceValues.eSet[cIndex1] && cScratchSliceValues.eSet[cIndex2] )
						{
							Key key1 = cSliceValues.edgeKeys[cIndex1] , key2 = cSliceValues.edgeKeys[cIndex2];
							pScratchSliceValues.vKeyValues[ thread ].push_back( std::pair< Key , Key >( key1 , key2 ) );

							const TreeNode* node = tree._sNodes.treeNodes[i];
							LocalDepth _depth = depth;
							while( _depth>fullDepth && tree._isValidSpaceNode( node->parent ) && HyperCubeTables< Dim , 1 , 0 >::Overlap[e.index][(unsigned int)(node-node->parent->children) ] )
							{
								node = node->parent , _depth--;
								typename SliceValues::Scratch &_pScratchSliceValues = scratchValues[_depth];
								_pScratchSliceValues.vKeyValues[ thread ].push_back( std::pair< Key , Key >( key1 , key2 ) );
							}
						}
					}
				}
			}
		}
		);
	}


	static void SetIsoEdges( const LevelSetExtraction::KeyGenerator< Dim > &keyGenerator , const FEMTree< Dim , Real >& tree , LocalDepth depth , LocalDepth fullDepth ,  std::vector< SliceValues >& sliceValues , std::vector< typename SliceValues::Scratch > &scratchValues )
	{
		auto _FaceIndex = [&]( const TreeNode *node , typename HyperCube::Cube< Dim >::template Element< 2 > f )
		{
			int depth , offset[Dim];
			tree.depthAndOffset( node , depth , offset );
			return keyGenerator( depth , offset , f );
		};

		SliceValues& sValues = sliceValues[depth];
#ifdef SANITIZED_PR
		Pointer( std::atomic< char > ) edgeSet = scratchValues[depth].eSet;
#else // !SANITIZED_PR
		Pointer( char ) edgeSet = scratchValues[depth].eSet;
#endif // SANITIZED_PR
		std::vector< ConstOneRingNeighborKey > neighborKeys( ThreadPool::NumThreads() );
		for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( tree._localToGlobal( depth ) );
		ThreadPool::ParallelFor( tree._sNodesBegin(depth) , tree._sNodesEnd(depth) , [&]( unsigned int thread , size_t i )
		{
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) )
			{
				int isoEdges[ 2 * HyperCube::MarchingSquares::MAX_EDGES ];
				ConstOneRingNeighborKey& neighborKey = neighborKeys[ thread ];
				TreeNode* leaf = tree._sNodes.treeNodes[i];
				// Process the face if it is on a leaf node
				if( !IsActiveNode< Dim >( leaf->children ) )
				{
					const typename LevelSetExtraction::FullCellIndexData< Dim >::template CellIndices<1> &eIndices = sValues.cellIndices.template indices<1>( leaf );
					const typename LevelSetExtraction::FullCellIndexData< Dim >::template CellIndices<2> &fIndices = sValues.cellIndices.template indices<2>( leaf );
					unsigned char mcIndex = sValues.mcIndices[ (node_index_type)i - sValues.cellIndices.nodeOffset ];

					// [NOTE] We do not add the vector of iso-edges at the depth where they are generate (as there can be at most two).
					//        We only add them to the coarser resolutions.
					// Calculate the iso-edges for the face
					FaceEdges fe;
					fe.count = HyperCube::MarchingSquares::AddEdgeIndices( mcIndex , isoEdges );
					for( int j=0 ; j<fe.count ; j++ ) for( int k=0 ; k<2 ; k++ )
					{
						if( !edgeSet[ eIndices[ isoEdges[2*j+k] ] ] ) MK_THROW( "Edge not set: " , 1<<depth );
						fe.edges[j][k] = sValues.edgeKeys[ eIndices[ isoEdges[2*j+k] ] ];
					}
					sValues.faceEdges[ fIndices[0] ] = fe;
					// Pass the information to the parents
					// [NOTE] For the 3D case we ony pass up to the parents if the face-adjacent neighbor does not exists.
					//        That could be bad if we want to use the information to extract from a sub-tree
					TreeNode *node = leaf;
					LocalDepth _depth = depth;
					typename HyperCube::Cube< Dim >::template Element< Dim > f( 0u );

					// Add the edges to the vector of iso-edges associated with a face
					std::vector< IsoEdge > edges( fe.count );
					for( int j=0 ; j<fe.count ; j++ ) edges[j] = fe.edges[j];
					// Add the edges to all ancestors
					while( _depth>fullDepth && tree._isValidSpaceNode( node->parent ) && HyperCubeTables< Dim , 2 , 0 >::Overlap[f.index][(unsigned int)(node-node->parent->children) ] )
					{
						node = node->parent , _depth--;
						Key key = _FaceIndex( node , f );
						typename SliceValues::Scratch &_scValues = scratchValues[_depth];
						_scValues.fKeyValues[ thread ].push_back( std::pair< Key , std::vector< IsoEdge > >( key , edges ) );
					}
				}
			}
		}
		);
	}

	static void SetLevelSet( const LevelSetExtraction::KeyGenerator< Dim > &keyGenerator , const FEMTree< Dim , Real >& tree , LocalDepth depth , const SliceValues &sValues , OutputDataStream< std::pair< node_index_type , node_index_type > >& edgeStream , bool flipOrientation )
	{
		auto _FaceIndex = [&]( const TreeNode *node , typename HyperCube::Cube< Dim >::template Element< 2 > f )
		{
			int depth , offset[Dim];
			tree.depthAndOffset( node , depth , offset );
			return keyGenerator( depth , offset , f );
		};

		auto AddEdge = [&]( unsigned int thread , IsoEdge e )
		{
			typename LevelSetExtraction::KeyMap< Dim , node_index_type >::const_iterator iter;
			node_index_type idx1 , idx2;
			if( ( iter=sValues.edgeVertexMap.find( e[0] ) )!=sValues.edgeVertexMap.end() ) idx1 = iter->second;
			else MK_THROW( "Couldn't find vertex in edge map" );
			if( ( iter=sValues.edgeVertexMap.find( e[1] ) )!=sValues.edgeVertexMap.end() ) idx2 = iter->second;
			else MK_THROW( "Couldn't find vertex in edge map" );
			if( flipOrientation ) edgeStream.write( thread , std::make_pair( idx2 , idx1 ) );
			else                  edgeStream.write( thread , std::make_pair( idx1 , idx2 ) );
		};

		ThreadPool::ParallelFor( tree._sNodesBegin(depth) , tree._sNodesEnd(depth) , [&]( unsigned int thread , size_t i )
		{
			if( tree._isValidSpaceNode( tree._sNodes.treeNodes[i] ) )
			{
				TreeNode *leaf = tree._sNodes.treeNodes[i];
				int res = 1<<depth;
				LocalDepth d ; LocalOffset off;
				tree._localDepthAndOffset( leaf , d , off );
				bool inBounds = off[0]>=0 && off[0]<res && off[1]>=0 && off[1]<res;

				if( inBounds && !IsActiveNode< Dim >( leaf->children ) )
				{
					// Gather the edges from the faces
					for( typename HyperCube::Cube< Dim >::template Element< 2 > f ; f<HyperCube::Cube< Dim >::template ElementNum< 2 >() ; f++ )
					{
						node_index_type fIdx = sValues.cellIndices.template indices<2>((node_index_type)i)[0];
						{
							const FaceEdges& fe = sValues.faceEdges[ fIdx ];
							for( int j=0 ; j<fe.count ; j++ ) AddEdge( thread , IsoEdge( fe.edges[j][0] , fe.edges[j][1] ) );
						}
					}
				}
			}
		}
		);
	}


	template< unsigned int WeightDegree , unsigned int DataSig >
	static bool GetIsoVertex( const FEMTree< Dim , Real >& tree , bool nonLinearFit , bool outputGradients , typename FEMIntegrator::template PointEvaluator< IsotropicUIntPack< Dim , DataSig > , ZeroUIntPack< Dim > >* pointEvaluator , const DensityEstimator< WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > >* data , Real isoValue , ConstPointSupportKey< IsotropicUIntPack< Dim , WeightDegree > >& weightKey , ConstPointSupportKey< IsotropicUIntPack< Dim , FEMSignature< DataSig >::Degree > >& dataKey , const TreeNode* node , typename HyperCube::template Cube< Dim >::template Element< 1 > e , const SliceValues& sValues , Vertex& vertex , const Data &zeroData )
	{
		static const unsigned int DataDegree = FEMSignature< DataSig >::Degree;
		Point< Real , Dim > position , gradient;
		int c0 , c1;
		const typename HyperCube::Cube< Dim >::template Element< 0 > *c = HyperCubeTables< Dim , 1 , 0 >::OverlapElements[e.index];
		c0 = c[0].index , c1 = c[1].index;

		const typename LevelSetExtraction::FullCellIndexData< Dim >::template CellIndices<0> &idx = sValues.cellIndices.template indices<0>( node );
		Real x0 = sValues.cornerValues[idx[c0]] , x1 = sValues.cornerValues[idx[c1]];
		Point< Real , Dim > dx0 , dx1;
		if( outputGradients ) dx0 = sValues.cornerGradients[idx[c0]] , dx1 = sValues.cornerGradients[idx[c1]];
		Point< Real , Dim > s;
		Real start , width;
		tree._startAndWidth( node , s , width );
		int o;
		{
			const HyperCube::Direction* dirs = HyperCubeTables< Dim , 1 >::Directions[ e.index ];
			for( int d=0 ; d<Dim ; d++ ) if( dirs[d]==HyperCube::CROSS )
			{
				o = d;
				start = s[d];
				for( int dd=1 ; dd<Dim ; dd++ ) position[(d+dd)%Dim] = s[(d+dd)%Dim] + width * ( dirs[(d+dd)%Dim]==HyperCube::BACK ? 0 : 1 );
			}
		}

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
				Point< Real , 2 > center( s[0] + width/2 , s[1] + width/2 );
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

	struct Stats
	{
		double cornersTime , verticesTime , edgesTime , curveTime;
		double copyFinerTime , setTableTime;
		Stats( void ) : cornersTime(0) , verticesTime(0) , edgesTime(0) , curveTime(0), copyFinerTime(0) , setTableTime(0) {;}
		std::string toString( void ) const
		{
			std::stringstream stream;
			stream << "Corners / Vertices / Edges / Curve / Set Table / Copy Finer: ";
			stream << std::fixed << std::setprecision(1) << cornersTime << " / " << verticesTime << " / " << edgesTime << " / " << curveTime << " / " << setTableTime << " / " << copyFinerTime;
			stream << " (s)";
			return stream.str();
		}
	};

protected:
	enum _SetFlag
	{
		CORNER_VALUES = 1,
		ISO_VERTICES = 2,
		ISO_EDGES = 4
	};

public:
	static int SetCornerValuesFlag( void ){ return _SetFlag::CORNER_VALUES; }
	static int SetIsoVerticesFlag ( void ){ return _SetFlag::CORNER_VALUES | _SetFlag::ISO_VERTICES; }
	static int SetIsoEdgesFlag    ( void ){ return _SetFlag::CORNER_VALUES | _SetFlag::ISO_VERTICES | _SetFlag::ISO_EDGES; }

	template< unsigned int WeightDegree , unsigned int DataSig , typename VertexStream , unsigned int ... FEMSigs >
	static Stats SetSliceValues( UIntPack< FEMSigs ... > , UIntPack< WeightDegree > , UIntPack< DataSig > , const FEMTree< Dim , Real > &tree , int maxKeyDepth , const DensityEstimator< WeightDegree > *densityWeights , const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > *data , const DenseNodeData< Real , UIntPack< FEMSigs ... > > &coefficients , Real isoValue , VertexStream &vertexStream , const Data &zeroData , bool nonLinearFit , bool outputGradients , std::vector< SliceValues > &sliceValues , int setFlag )
	{
		if( maxKeyDepth<tree._maxDepth ) MK_THROW( "Max key depth has to be at least tree depth: " , tree._maxDepth , " <= " , maxKeyDepth );
		LevelSetExtraction::KeyGenerator< Dim > keyGenerator( maxKeyDepth );
		unsigned int slabStart = 0 , slabEnd = 1<<tree._maxDepth;
		LocalDepth fullDepth = tree.getFullDepth( UIntPack< FEMSignature< FEMSigs >::Degree ... >() );

		_BadRootCount = 0u;
		Stats stats;

		static_assert( sizeof...(FEMSigs)==Dim , "[ERROR] Number of signatures should match dimension" );
		tree._setFEM1ValidityFlags( UIntPack< FEMSigs ... >() );
		static const int FEMDegrees[] = { FEMSignature< FEMSigs >::Degree ... };
		for( int d=0 ; d<Dim ; d++ ) if( FEMDegrees[d]==0 && nonLinearFit ) MK_THROW( "Constant B-Splines do not support gradient estimation" );

		LevelSetExtraction::SetHyperCubeTables< Dim >();

		typename FEMIntegrator::template PointEvaluator< IsotropicUIntPack< Dim , DataSig > , ZeroUIntPack< Dim > >* pointEvaluator = NULL;
		if constexpr( HasData ) if( data ) pointEvaluator = new typename FEMIntegrator::template PointEvaluator< IsotropicUIntPack< Dim , DataSig > , ZeroUIntPack< Dim > >( tree._maxDepth );
		DenseNodeData< Real , UIntPack< FEMSigs ... > > coarseCoefficients( tree._sNodesEnd( tree._maxDepth-1 ) );
		memset( coarseCoefficients() , 0 , sizeof(Real)*tree._sNodesEnd( tree._maxDepth-1 ) );
		ThreadPool::ParallelFor( tree._sNodesBegin(0) , tree._sNodesEnd( tree._maxDepth-1 ) , [&]( unsigned int, size_t i ){ coarseCoefficients[i] = coefficients[i]; } );
		typename FEMIntegrator::template RestrictionProlongation< UIntPack< FEMSigs ... > > rp;
		for( LocalDepth d=1 ; d<tree._maxDepth ; d++ ) tree._upSample( UIntPack< FEMSigs ... >() , rp , d , ( ConstPointer(Real) )coarseCoefficients()+tree._sNodesBegin(d-1) , coarseCoefficients()+tree._sNodesBegin(d) );

		std::vector< _Evaluator< UIntPack< FEMSigs ... > , 1 > > evaluators( tree._maxDepth+1 );
		for( LocalDepth d=0 ; d<=tree._maxDepth ; d++ ) evaluators[d].set( tree._maxDepth );

		sliceValues.resize( tree._maxDepth+1 );
		std::vector< typename SliceValues::Scratch > scratchValues( tree._maxDepth+1 );

		// Initialize the slice
		for( LocalDepth d=tree._maxDepth ; d>=fullDepth ; d-- )
		{
			double t = Time();
			sliceValues[d].cellIndices.set( tree._sNodes , tree._localToGlobal( d ) );
			stats.setTableTime += Time()-t;
			sliceValues[d].reset( nonLinearFit || outputGradients );
			scratchValues[d].reset( sliceValues[d].cellIndices );
		}

		for( LocalDepth d=tree._maxDepth; d>=fullDepth ; d-- )
		{
			// Copy edges from finer
			if( setFlag & _SetFlag::ISO_EDGES )
			{
				double t = Time();
				if( d<tree._maxDepth ) CopyFinerSliceIsoEdgeKeys( tree , d , fullDepth , sliceValues , scratchValues );
				stats.copyFinerTime += Time()-t;
			}

			if( setFlag & _SetFlag::CORNER_VALUES )
			{
				double t = Time();
				SetSCornerValues< FEMSigs ... >( tree , coefficients() , coarseCoefficients() , isoValue , d , fullDepth , sliceValues , scratchValues , evaluators[d] );
				stats.cornersTime += Time()-t;
			}

			if( setFlag & _SetFlag::ISO_VERTICES )
			{
				double t = Time();
				SetIsoVertices< WeightDegree , DataSig >( keyGenerator , tree , nonLinearFit , outputGradients , pointEvaluator , densityWeights , data , isoValue , d , fullDepth , vertexStream , sliceValues , scratchValues , zeroData );
				stats.verticesTime += Time()-t;
			}

			if( setFlag & _SetFlag::ISO_EDGES )
			{
				double t = Time();
				SetIsoEdges( keyGenerator , tree , d , fullDepth , sliceValues , scratchValues );
				stats.edgesTime += Time()-t , t = Time();
			}
		}

		for( LocalDepth d=tree._maxDepth ; d>=fullDepth ; d-- )
		{
			ThreadPool::ParallelSections
			(
				[ &sliceValues , &scratchValues , d ]( void ){ sliceValues[d].setFromScratch( scratchValues[d].vKeyValues ); } ,
				[ &sliceValues , &scratchValues , d ]( void ){ sliceValues[d].setFromScratch( scratchValues[d].eKeyValues ); } ,
				[ &sliceValues , &scratchValues , d ]( void ){ sliceValues[d].setFromScratch( scratchValues[d].fKeyValues ); }
			);
			if( ( setFlag & _SetFlag::ISO_VERTICES) && d<tree._maxDepth ) for( auto iter=sliceValues[d+1].vertexPairMap.begin() ; iter!=sliceValues[d+1].vertexPairMap.end() ; iter++ ) sliceValues[d].vertexPairMap[ iter->first ] = iter->second;
		}

		size_t badRootCount = _BadRootCount;
		if( badRootCount!=0 ) MK_WARN( "bad average roots: " , badRootCount );
		return stats;
	}

	static void SetLevelSets( const LevelSetExtraction::KeyGenerator< Dim > &keyGenerator , const FEMTree< Dim , Real >& tree , LocalDepth fullDepth , const std::vector< SliceValues > &sliceValues , OutputDataStream< std::pair< node_index_type , node_index_type > >& edgeStream , bool flipOrientation )
	{
		for( LocalDepth d=tree._maxDepth ; d>=fullDepth ; d-- ) SetLevelSet( keyGenerator , tree , d , sliceValues[d] , edgeStream , flipOrientation );
	}

	template< unsigned int WeightDegree , unsigned int DataSig , typename VertexStream , unsigned int ... FEMSigs >
	static Stats Extract( UIntPack< FEMSigs ... > , UIntPack< WeightDegree > , UIntPack< DataSig > , const FEMTree< Dim , Real >& tree , const DensityEstimator< WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > >* data , const DenseNodeData< Real , UIntPack< FEMSigs ... > >& coefficients , Real isoValue , VertexStream &vertexStream , OutputDataStream< std::pair< node_index_type , node_index_type > > &edgeStream , const Data &zeroData , bool nonLinearFit , bool outputGradients , bool flipOrientation )
	{
		std::vector< SliceValues > sliceValues;

		Stats stats = SetSliceValues( UIntPack< FEMSigs ... >() , UIntPack< WeightDegree >() , UIntPack< DataSig >() , tree , tree.maxDepth() , densityWeights , data , coefficients , isoValue , vertexStream , zeroData , nonLinearFit , outputGradients , sliceValues , SetIsoEdgesFlag() );
		LevelSetExtraction::KeyGenerator< Dim > keyGenerator( tree.maxDepth() );
		{
			double t = Time();
			SetLevelSets( keyGenerator , tree , tree.getFullDepth( UIntPack< FEMSignature< FEMSigs >::Degree ... >() ) , sliceValues , edgeStream , flipOrientation );
			stats.curveTime += Time()-t;
		}
		return stats;
	}
};

template< bool HasData , typename Real , typename Data > std::atomic< size_t > _LevelSetExtractor< HasData , Real , 2 , Data >::_BadRootCount;

template< typename Real >
struct LevelSetExtractor< Real , 2 >
{
	static const unsigned int Dim = 2;
	static const bool HasData = false;
	typedef unsigned char Data;
	typedef typename _LevelSetExtractor< HasData , Real , Dim , Data >::Stats Stats;
	typedef typename _LevelSetExtractor< HasData , Real , Dim , Data >::Vertex Vertex;
	typedef typename _LevelSetExtractor< HasData , Real , Dim , Data >::SliceValues SliceValues;
	typedef typename _LevelSetExtractor< HasData , Real , Dim , Data >::TreeSliceValuesAndVertexPositions TreeSliceValuesAndVertexPositions;
	typedef typename _LevelSetExtractor< HasData , Real , Dim , Data >::TreeNode TreeNode;
	using OutputVertexStream = typename _LevelSetExtractor< HasData , Real , Dim , Data >::OutputVertexStream;
	template< unsigned int WeightDegree > using DensityEstimator = typename _LevelSetExtractor< HasData , Real , Dim , Data >::template DensityEstimator< WeightDegree >;
	static int SetCornerValuesFlag( void ){ return _LevelSetExtractor< HasData , Real , Dim , Data >::SetCornerValuesFlag(); }
	static int SetIsoVerticesFlag ( void ){ return _LevelSetExtractor< HasData , Real , Dim , Data >::SetIsoVerticesFlag (); }
	static int SetIsoEdgesFlag    ( void ){ return _LevelSetExtractor< HasData , Real , Dim , Data >::SetIsoEdgesFlag    (); }

	template< unsigned int WeightDegree , unsigned int ... FEMSigs >
	static Stats Extract( UIntPack< FEMSigs ... > , UIntPack< WeightDegree > , const FEMTree< Dim , Real >& tree , const DensityEstimator< WeightDegree >* densityWeights , const DenseNodeData< Real , UIntPack< FEMSigs ... > >& coefficients , Real isoValue , OutputVertexStream &vertexStream , OutputDataStream< std::pair< node_index_type , node_index_type > > &edgeStream , bool nonLinearFit , bool outputGradients , bool flipOrientation )
	{
		Data zeroData = 0;
		static const unsigned int DataSig = FEMDegreeAndBType< 0 , BOUNDARY_FREE >::Signature;
		const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > *data = NULL;
		return _LevelSetExtractor< HasData , Real , Dim , Data >::template Extract< WeightDegree , DataSig , OutputVertexStream , FEMSigs ... >( UIntPack< FEMSigs ... >() , UIntPack< WeightDegree >() , UIntPack< DataSig >() , tree , densityWeights , data , coefficients , isoValue , vertexStream , edgeStream , zeroData , nonLinearFit , outputGradients , flipOrientation );
	}

	template< unsigned int WeightDegree , unsigned int ... FEMSigs >
	static Stats SetSliceValues( UIntPack< FEMSigs ... > , UIntPack< WeightDegree > , const FEMTree< Dim , Real > &tree , int maxKeyDepth , const DensityEstimator< WeightDegree > *densityWeights , const DenseNodeData< Real , UIntPack< FEMSigs ... > > &coefficients , Real isoValue , OutputVertexStream &vertexStream , bool nonLinearFit , bool outputGradients , std::vector< SliceValues > &sliceValues , int setFlag )
	{
		Data zeroData = 0;
		static const unsigned int DataSig = FEMDegreeAndBType< 0 , BOUNDARY_FREE >::Signature;
		const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > *data = NULL;
		return _LevelSetExtractor< HasData , Real , Dim , Data >::template SetSliceValues< WeightDegree , DataSig , OutputVertexStream , FEMSigs ... >( UIntPack< FEMSigs ... >() , UIntPack< WeightDegree >() , UIntPack< DataSig >() , tree , maxKeyDepth , densityWeights , data , coefficients , isoValue , vertexStream , zeroData , nonLinearFit , outputGradients , sliceValues , setFlag );
	}
};

template< typename Real , typename Data >
struct LevelSetExtractor< Real , 2 , Data >
{
	static const unsigned int Dim = 2;
	static const bool HasData = true;
	typedef typename _LevelSetExtractor< HasData , Real , Dim , Data >::Stats Stats;
	typedef typename _LevelSetExtractor< HasData , Real , Dim , Data >::Vertex Vertex;
	typedef typename _LevelSetExtractor< HasData , Real , Dim , Data >::SliceValues SliceValues;
	typedef typename _LevelSetExtractor< HasData , Real , Dim , Data >::TreeSliceValuesAndVertexPositions TreeSliceValuesAndVertexPositions;
	typedef typename _LevelSetExtractor< HasData , Real , Dim , Data >::TreeNode TreeNode;
	using OutputVertexStream = typename _LevelSetExtractor< HasData , Real , Dim , Data >::OutputVertexStream;
	template< unsigned int WeightDegree > using DensityEstimator = typename _LevelSetExtractor< HasData , Real , Dim , Data >::template DensityEstimator< WeightDegree >;
	static int SetCornerValuesFlag( void ){ return _LevelSetExtractor< HasData , Real , Dim , Data >::SetCornerValuesFlag(); }
	static int SetIsoVerticesFlag ( void ){ return _LevelSetExtractor< HasData , Real , Dim , Data >::SetIsoVerticesFlag (); }
	static int SetIsoEdgesFlag    ( void ){ return _LevelSetExtractor< HasData , Real , Dim , Data >::SetIsoEdgesFlag    (); }

	template< unsigned int WeightDegree , unsigned int DataSig , unsigned int ... FEMSigs >
	static Stats Extract( UIntPack< FEMSigs ... > , UIntPack< WeightDegree > , UIntPack< DataSig > , const FEMTree< Dim , Real >& tree , const DensityEstimator< WeightDegree > *densityWeights , const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > *data , const DenseNodeData< Real , UIntPack< FEMSigs ... > > &coefficients , Real isoValue , OutputVertexStream &vertexStream , OutputDataStream< std::pair< node_index_type , node_index_type > > &edgeStream , const Data &zeroData , bool nonLinearFit , bool outputGradients , bool flipOrientation )
	{
		return _LevelSetExtractor< HasData , Real , Dim , Data >::template Extract< WeightDegree , DataSig , OutputVertexStream , FEMSigs ... >( UIntPack< FEMSigs ... >() , UIntPack< WeightDegree >() , UIntPack< DataSig >() , tree , densityWeights , data , coefficients , isoValue , vertexStream , edgeStream , zeroData , nonLinearFit , outputGradients , flipOrientation );
	}

	template< unsigned int WeightDegree , unsigned int DataSig , unsigned int ... FEMSigs >
	static Stats SetSliceValues( UIntPack< FEMSigs ... > , UIntPack< WeightDegree > , UIntPack< DataSig > , const FEMTree< Dim , Real > &tree , int maxKeyDepth , const DensityEstimator< WeightDegree > *densityWeights , const SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > *data , const DenseNodeData< Real , UIntPack< FEMSigs ... > > &coefficients , Real isoValue , OutputVertexStream &vertexStream , const Data &zeroData , bool nonLinearFit , bool outputGradients , std::vector< SliceValues > &sliceValues , int setFlag )
	{
		return _LevelSetExtractor< HasData , Real , Dim , Data >::template SetSliceValues< WeightDegree , DataSig , OutputVertexStream , FEMSigs ... >( UIntPack< FEMSigs ... >() , UIntPack< WeightDegree >() , UIntPack< DataSig >() , tree , maxKeyDepth , densityWeights , data , coefficients , isoValue , vertexStream , zeroData , nonLinearFit , outputGradients , sliceValues , setFlag );
	}
};
