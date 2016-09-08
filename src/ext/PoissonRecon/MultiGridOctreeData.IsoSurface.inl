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

#include "Octree.h"
#include "MyTime.h"
#include "MemoryUsage.h"
#include "MAT.h"

template< class Real >
template< class Vertex >
Octree< Real >::SliceValues< Vertex >::SliceValues( void )
{
	_oldCCount = _oldECount = _oldFCount = _oldNCount = 0;
	cornerValues = NullPointer( Real ) ; cornerGradients = NullPointer( Point3D< Real > ) ; cornerSet = NullPointer( char );
	edgeKeys = NullPointer( long long ) ; edgeSet = NullPointer( char );
	faceEdges = NullPointer( FaceEdges ) ; faceSet = NullPointer( char );
	mcIndices = NullPointer( char );
}
template< class Real >
template< class Vertex >
Octree< Real >::SliceValues< Vertex >::~SliceValues( void )
{
	_oldCCount = _oldECount = _oldFCount = _oldNCount = 0;
	FreePointer( cornerValues ) ; FreePointer( cornerGradients ) ; FreePointer( cornerSet );
	FreePointer( edgeKeys ) ; FreePointer( edgeSet );
	FreePointer( faceEdges ) ; FreePointer( faceSet );
	FreePointer( mcIndices );
}
template< class Real >
template< class Vertex >
void Octree< Real >::SliceValues< Vertex >::reset( bool nonLinearFit )
{
	faceEdgeMap.clear() , edgeVertexMap.clear() , vertexPairMap.clear();

	if( _oldNCount<sliceData.nodeCount )
	{
		_oldNCount = sliceData.nodeCount;
		FreePointer( mcIndices );
		if( sliceData.nodeCount>0 ) mcIndices = AllocPointer< char >( _oldNCount );
	}
	if( _oldCCount<sliceData.cCount )
	{
		_oldCCount = sliceData.cCount;
		FreePointer( cornerValues ) ; FreePointer( cornerGradients ) ; FreePointer( cornerSet );
		if( sliceData.cCount>0 )
		{
			cornerValues = AllocPointer< Real >( _oldCCount );
			if( nonLinearFit ) cornerGradients = AllocPointer< Point3D< Real > >( _oldCCount );
			cornerSet = AllocPointer< char >( _oldCCount );
		}
	}
	if( _oldECount<sliceData.eCount )
	{
		_oldECount = sliceData.eCount;
		FreePointer( edgeKeys ) ; FreePointer( edgeSet );
		edgeKeys = AllocPointer< long long >( _oldECount );
		edgeSet = AllocPointer< char >( _oldECount );
	}
	if( _oldFCount<sliceData.fCount )
	{
		_oldFCount = sliceData.fCount;
		FreePointer( faceEdges ) ; FreePointer( faceSet );
		faceEdges = AllocPointer< FaceEdges >( _oldFCount );
		faceSet = AllocPointer< char >( _oldFCount );
	}
	
	if( sliceData.cCount>0 ) memset( cornerSet , 0 , sizeof( char ) * sliceData.cCount );
	if( sliceData.eCount>0 ) memset(   edgeSet , 0 , sizeof( char ) * sliceData.eCount );
	if( sliceData.fCount>0 ) memset(   faceSet , 0 , sizeof( char ) * sliceData.fCount );
}
template< class Real >
template< class Vertex >
Octree< Real >::XSliceValues< Vertex >::XSliceValues( void )
{
	_oldECount = _oldFCount = 0;
	edgeKeys = NullPointer( long long ) ; edgeSet = NullPointer( char );
	faceEdges = NullPointer( FaceEdges ) ; faceSet = NullPointer( char );
}
template< class Real >
template< class Vertex >
Octree< Real >::XSliceValues< Vertex >::~XSliceValues( void )
{
	_oldECount = _oldFCount = 0;
	FreePointer( edgeKeys ) ; FreePointer( edgeSet );
	FreePointer( faceEdges ) ; FreePointer( faceSet );
}
template< class Real >
template< class Vertex >
void Octree< Real >::XSliceValues< Vertex >::reset( void )
{
	faceEdgeMap.clear() , edgeVertexMap.clear() , vertexPairMap.clear();

	if( _oldECount<xSliceData.eCount )
	{
		_oldECount = xSliceData.eCount;
		FreePointer( edgeKeys ) ; FreePointer( edgeSet );
		edgeKeys = AllocPointer< long long >( _oldECount );
		edgeSet = AllocPointer< char >( _oldECount );
	}
	if( _oldFCount<xSliceData.fCount )
	{
		_oldFCount = xSliceData.fCount;
		FreePointer( faceEdges ) ; FreePointer( faceSet );
		faceEdges = AllocPointer< FaceEdges >( _oldFCount );
		faceSet = AllocPointer< char >( _oldFCount );
	}
	if( xSliceData.eCount>0 ) memset( edgeSet , 0 , sizeof( char ) * xSliceData.eCount );
	if( xSliceData.fCount>0 ) memset( faceSet , 0 , sizeof( char ) * xSliceData.fCount );
}

template< class Real >
template< int FEMDegree , int WeightDegree , int ColorDegree , class Vertex >
void Octree< Real >::GetMCIsoSurface( const SparseNodeData< Real , WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Point3D< Real > > , ColorDegree >* colorData , const DenseNodeData< Real , FEMDegree >& solution , Real isoValue , CoredMeshData< Vertex >& mesh , bool nonLinearFit , bool addBarycenter , bool polygonMesh )
{
	int maxDepth = _tree.maxDepth();
	if( FEMDegree==1 && nonLinearFit ) fprintf( stderr , "[WARNING] First order B-Splines do not support non-linear interpolation\n" ) , nonLinearFit = false;

	BSplineData< ColorDegree >* colorBSData = NULL;
	if( colorData )
	{
		colorBSData = new BSplineData< ColorDegree >();
		colorBSData->set( maxDepth , _dirichlet );
	}

	DenseNodeData< Real , FEMDegree > coarseSolution( _sNodes.end( maxDepth-1 ) );
	memset( coarseSolution.data , 0 , sizeof(Real)*_sNodes.end( maxDepth-1 ) );
#pragma omp parallel for num_threads( threads )
	for( int i=_sNodes.begin(_minDepth) ; i<_sNodes.end(maxDepth-1) ; i++ ) coarseSolution[i] = solution[i];
	for( int d=_minDepth+1 ; d<maxDepth ; d++ ) _UpSample( d , coarseSolution );
	MemoryUsage();

	std::vector< _Evaluator< FEMDegree > > evaluators( maxDepth+1 );
	for( int d=_minDepth ; d<=maxDepth ; d++ ) evaluators[d].set( d-1 , _dirichlet );
	int vertexOffset = 0;
	std::vector< SlabValues< Vertex > > slabValues( maxDepth+1 );

	// Initialize the back slice
	for( int d=maxDepth ; d>=_minDepth ; d-- )
	{
		_sNodes.setSliceTableData ( slabValues[d]. sliceValues(0). sliceData , d , 0 , threads );
		_sNodes.setSliceTableData ( slabValues[d]. sliceValues(1). sliceData , d , 1 , threads );
		_sNodes.setXSliceTableData( slabValues[d].xSliceValues(0).xSliceData , d , 0 , threads );
		slabValues[d].sliceValues (0).reset( nonLinearFit );
		slabValues[d].sliceValues (1).reset( nonLinearFit );
		slabValues[d].xSliceValues(0).reset( );
	}
	for( int d=maxDepth ; d>=_minDepth ; d-- )
	{
		// Copy edges from finer
		if( d<maxDepth ) CopyFinerSliceIsoEdgeKeys( d , 0 , slabValues , threads );
		SetSliceIsoCorners( solution , coarseSolution , isoValue , d , 0 , slabValues , evaluators[d] , threads );
		SetSliceIsoVertices< WeightDegree , ColorDegree >( colorBSData , densityWeights , colorData , isoValue , d , 0 , vertexOffset , mesh , slabValues , threads );
		SetSliceIsoEdges( d , 0 , slabValues , threads );
	}
	// Iterate over the slices at the finest level
	for( int slice=0 ; slice<( 1<<(maxDepth-1) ) ; slice++ )
	{
		// Process at all depths that that contain this slice
		for( int d=maxDepth , o=slice+1 ; d>=_minDepth ; d-- , o>>=1 )
		{
			// Copy edges from finer (required to ensure we correctly track edge cancellations)
			if( d<maxDepth )
			{
				CopyFinerSliceIsoEdgeKeys( d , o , slabValues , threads );
				CopyFinerXSliceIsoEdgeKeys( d , o-1 , slabValues , threads );
			}

			// Set the slice values/vertices
			SetSliceIsoCorners( solution , coarseSolution , isoValue , d , o , slabValues , evaluators[d] , threads );
			SetSliceIsoVertices< WeightDegree , ColorDegree >( colorBSData , densityWeights , colorData , isoValue , d , o , vertexOffset , mesh , slabValues , threads );
			SetSliceIsoEdges( d , o , slabValues , threads );

			// Set the cross-slice edges
			SetXSliceIsoVertices< WeightDegree , ColorDegree >( colorBSData , densityWeights , colorData , isoValue , d , o-1 , vertexOffset , mesh , slabValues , threads );
			SetXSliceIsoEdges( d , o-1 , slabValues , threads );

			// Add the triangles
			SetIsoSurface( d , o-1 , slabValues[d].sliceValues(o-1) , slabValues[d].sliceValues(o) , slabValues[d].xSliceValues(o-1) , mesh , polygonMesh , addBarycenter , vertexOffset , threads );

			if( o&1 ) break;
		}
		for( int d=maxDepth , o=slice+1 ; d>=_minDepth ; d-- , o>>=1 )
		{
			// Initialize for the next pass
			if( o<(1<<d) )
			{
				_sNodes.setSliceTableData( slabValues[d].sliceValues(o+1).sliceData , d , o+1 , threads );
				_sNodes.setXSliceTableData( slabValues[d].xSliceValues(o).xSliceData , d , o , threads );
				slabValues[d].sliceValues(o+1).reset( nonLinearFit );
				slabValues[d].xSliceValues(o).reset();
			}
			if( o&1 ) break;
		}
	}
	MemoryUsage();
	if( colorBSData ) delete colorBSData;
	coarseSolution.resize( 0 );
}

template< class Real >
template< int FEMDegree , int NormalDegree >
Real Octree< Real >::GetIsoValue( const DenseNodeData< Real , FEMDegree >& solution , const SparseNodeData< Real , NormalDegree >& nodeWeights )
{
	Real isoValue=0 , weightSum=0;
	int maxDepth = _tree.maxDepth();

	Pointer( Real ) nodeValues = AllocPointer< Real >( _sNodes.end(maxDepth) );
	memset( nodeValues , 0 , sizeof(Real) * _sNodes.end(maxDepth) );
	DenseNodeData< Real , FEMDegree > metSolution( _sNodes.end( maxDepth-1 ) );
	memset( metSolution.data , 0 , sizeof(Real)*_sNodes.end( maxDepth-1 ) );
#pragma omp parallel for num_threads( threads )
	for( int i=_sNodes.begin(_minDepth) ; i<_sNodes.end(maxDepth-1) ; i++ ) metSolution[i] = solution[i];
	for( int d=_minDepth+1 ; d<maxDepth ; d++ ) _UpSample( d , metSolution );
	for( int d=maxDepth ; d>=_minDepth ; d-- )
	{
		_Evaluator< FEMDegree > evaluator;
		evaluator.set( d-1 , _dirichlet );
		std::vector< ConstPointSupportKey< FEMDegree > > neighborKeys( std::max< int >( 1 , threads ) );
		for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( d );
#pragma omp parallel for num_threads( threads ) reduction( + : isoValue , weightSum )
		for( int i=_sNodes.begin(d) ; i<_sNodes.end(d) ; i++ ) if( _IsValidNode< 0 >( _sNodes.treeNodes[i] ) )
		{
			ConstPointSupportKey< FEMDegree >& neighborKey = neighborKeys[ omp_get_thread_num() ];
			TreeOctNode* node = _sNodes.treeNodes[i];
			Real value = Real(0);
			if( node->children )
			{
				if( NormalDegree&1 ) value = nodeValues[ node->children->nodeData.nodeIndex ];
				else for( int c=0 ; c<Cube::CORNERS ; c++ ) value += nodeValues[ node->children[c].nodeData.nodeIndex ] / Cube::CORNERS;
			}
			else if( nodeWeights.index( _sNodes.treeNodes[i] )>=0 )
			{
				neighborKey.getNeighbors( node );
				int c=0 , x , y , z;
				if( node->parent ) c = int( node - node->parent->children );
				Cube::FactorCornerIndex( c , x , y , z );

				// Since evaluation requires parent indices, we need to check that the node's parent is interiorly supported
				bool isInterior = _IsInteriorlySupported< FEMDegree >( node->parent );

				if( NormalDegree&1 ) value = _getCornerValue( neighborKey , node , 0 , solution , metSolution , evaluator , isInterior );
				else                 value = _getCenterValue( neighborKey , node ,     solution , metSolution , evaluator , isInterior );
			}
			nodeValues[i] = value;
			int idx = nodeWeights.index( _sNodes.treeNodes[i] );
			if( idx!=-1 )
			{
				Real w = nodeWeights.data[ idx ];
				if( w!=0 ) isoValue += value * w , weightSum += w;
			}
		}
	}
	metSolution.resize( 0 );
	FreePointer( nodeValues );

	return isoValue / weightSum;
}

template< class Real >
template< class Vertex , int FEMDegree >
void Octree< Real >::SetSliceIsoCorners( const DenseNodeData< Real , FEMDegree >& solution , const DenseNodeData< Real , FEMDegree >& coarseSolution , Real isoValue , int depth , int slice , std::vector< SlabValues< Vertex > >& slabValues , const _Evaluator< FEMDegree >& evaluator , int threads )
{
	if( slice>0          ) SetSliceIsoCorners( solution , coarseSolution , isoValue , depth , slice , 1 , slabValues , evaluator , threads );
	if( slice<(1<<depth) ) SetSliceIsoCorners( solution , coarseSolution , isoValue , depth , slice , 0 , slabValues , evaluator , threads );
}
template< class Real >
template< class Vertex , int FEMDegree >
void Octree< Real >::SetSliceIsoCorners( const DenseNodeData< Real , FEMDegree >& solution , const DenseNodeData< Real , FEMDegree >& coarseSolution , Real isoValue , int depth , int slice , int z , std::vector< SlabValues< Vertex > >& slabValues , const struct _Evaluator< FEMDegree >& evaluator , int threads )
{
	typename Octree::template SliceValues< Vertex >& sValues = slabValues[depth].sliceValues( slice );
	std::vector< ConstPointSupportKey< FEMDegree > > neighborKeys( std::max< int >( 1 , threads ) );
	for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( depth );
#pragma omp parallel for num_threads( threads )
	for( int i=_sNodes.begin(depth,slice-z) ; i<_sNodes.end(depth,slice-z) ; i++ ) if( _IsValidNode< 0 >( _sNodes.treeNodes[i] ) )
	{
		Real squareValues[ Square::CORNERS ];
		ConstPointSupportKey< FEMDegree >& neighborKey = neighborKeys[ omp_get_thread_num() ];
		TreeOctNode* leaf = _sNodes.treeNodes[i];
		if( !leaf->children )
		{
			const typename SortedTreeNodes::SquareCornerIndices& cIndices = sValues.sliceData.cornerIndices( leaf );

			bool isInterior = _IsInteriorlySupported< FEMDegree >( leaf->parent );
			neighborKey.getNeighbors( leaf );

			for( int x=0 ; x<2 ; x++ ) for( int y=0 ; y<2 ; y++ )
			{
				int cc = Cube::CornerIndex( x , y , z );
				int fc = Square::CornerIndex( x , y );
				int vIndex = cIndices[fc];
				if( !sValues.cornerSet[vIndex] )
				{
					if( sValues.cornerGradients )
					{
						std::pair< Real , Point3D< Real > > p = _getCornerValueAndGradient( neighborKey , leaf , cc , solution , coarseSolution , evaluator , isInterior );
						sValues.cornerValues[vIndex] = p.first , sValues.cornerGradients[vIndex] = p.second;
					}
					else sValues.cornerValues[vIndex] = _getCornerValue( neighborKey , leaf , cc , solution , coarseSolution , evaluator , isInterior );
					sValues.cornerSet[vIndex] = 1;
				}
				squareValues[fc] = sValues.cornerValues[ vIndex ];
				TreeOctNode* node = leaf;
				int _depth = depth , _slice = slice;
				while( _IsValidNode< 0 >( node->parent ) && (node-node->parent->children)==cc )
				{
					node = node->parent , _depth-- , _slice >>= 1;
					typename Octree::template SliceValues< Vertex >& _sValues = slabValues[_depth].sliceValues( _slice );
					const typename SortedTreeNodes::SquareCornerIndices& _cIndices = _sValues.sliceData.cornerIndices( node );
					int _vIndex = _cIndices[fc];
					_sValues.cornerValues[_vIndex] = sValues.cornerValues[vIndex];
					if( _sValues.cornerGradients ) _sValues.cornerGradients[_vIndex] = sValues.cornerGradients[vIndex];
					_sValues.cornerSet[_vIndex] = 1;
				}
			}
			sValues.mcIndices[ i - sValues.sliceData.nodeOffset ] = MarchingSquares::GetIndex( squareValues , isoValue );
		}
	}
}

template< class Real >
template< int WeightDegree , int ColorDegree , class Vertex >
void Octree< Real >::SetSliceIsoVertices( const BSplineData< ColorDegree >* colorBSData , const SparseNodeData< Real , WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Point3D< Real > > , ColorDegree >* colorData , Real isoValue , int depth , int slice , int& vOffset , CoredMeshData< Vertex >& mesh , std::vector< SlabValues< Vertex > >& slabValues , int threads )
{
	if( slice>0          ) SetSliceIsoVertices< WeightDegree , ColorDegree >( colorBSData , densityWeights , colorData , isoValue , depth , slice , 1 , vOffset , mesh , slabValues , threads );
	if( slice<(1<<depth) ) SetSliceIsoVertices< WeightDegree , ColorDegree >( colorBSData , densityWeights , colorData , isoValue , depth , slice , 0 , vOffset , mesh , slabValues , threads );
}
template< class Real >
template< int WeightDegree , int ColorDegree , class Vertex >
void Octree< Real >::SetSliceIsoVertices( const BSplineData< ColorDegree >* colorBSData , const SparseNodeData< Real , WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Point3D< Real > > , ColorDegree >* colorData , Real isoValue , int depth , int slice , int z , int& vOffset , CoredMeshData< Vertex >& mesh , std::vector< SlabValues< Vertex > >& slabValues , int threads )
{
	typename Octree::template SliceValues< Vertex >& sValues = slabValues[depth].sliceValues( slice );
	// [WARNING] In the case Degree=2, these two keys are the same, so we don't have to maintain them separately.
	std::vector< ConstAdjacenctNodeKey > neighborKeys( std::max< int >( 1 , threads ) );
	std::vector< ConstPointSupportKey< WeightDegree > > weightKeys( std::max< int >( 1 , threads ) );
	std::vector< ConstPointSupportKey< ColorDegree > > colorKeys( std::max< int >( 1 , threads ) );
	for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( depth ) , weightKeys[i].set( depth ) , colorKeys[i].set( depth );
#pragma omp parallel for num_threads( threads )
	for( int i=_sNodes.begin(depth,slice-z) ; i<_sNodes.end(depth,slice-z) ; i++ ) if( _IsValidNode< 0 >( _sNodes.treeNodes[i] ) )
	{
		ConstAdjacenctNodeKey& neighborKey =  neighborKeys[ omp_get_thread_num() ];
		ConstPointSupportKey< WeightDegree >& weightKey = weightKeys[ omp_get_thread_num() ];
		ConstPointSupportKey< ColorDegree >& colorKey = colorKeys[ omp_get_thread_num() ];
		TreeOctNode* leaf = _sNodes.treeNodes[i];
		if( !leaf->children )
		{
			int idx = i - sValues.sliceData.nodeOffset;
			const typename SortedTreeNodes::SquareEdgeIndices& eIndices = sValues.sliceData.edgeIndices( leaf );
			if( MarchingSquares::HasRoots( sValues.mcIndices[idx] ) )
			{
				neighborKey.getNeighbors( leaf );
				if( densityWeights ) weightKey.getNeighbors( leaf );
				if( colorData ) colorKey.getNeighbors( leaf );
				for( int e=0 ; e<Square::EDGES ; e++ )
					if( MarchingSquares::HasEdgeRoots( sValues.mcIndices[idx] , e ) )
					{
						int vIndex = eIndices[e];
						if( !sValues.edgeSet[vIndex] )
						{
							Vertex vertex;
							int o , y;
							Square::FactorEdgeIndex( e , o , y );
							long long key = VertexData::EdgeIndex( leaf , Cube::EdgeIndex( o , y , z ) , _sNodes.levels() );
							GetIsoVertex( colorBSData , densityWeights , colorData , isoValue , weightKey , colorKey , leaf , e , z , sValues , vertex );
							vertex.point = vertex.point * _scale + _center;
							bool stillOwner = false;
							std::pair< int , Vertex > hashed_vertex;
#pragma omp critical (add_point_access)
							{
								if( !sValues.edgeSet[vIndex] )
								{
									mesh.addOutOfCorePoint( vertex );
									sValues.edgeSet[ vIndex ] = 1;
									sValues.edgeKeys[ vIndex ] = key;
									sValues.edgeVertexMap[key] = hashed_vertex = std::pair< int , Vertex >( vOffset , vertex );
									vOffset++;
									stillOwner = true;
								}
							}
							if( stillOwner )
							{
								// We only need to pass the iso-vertex down if the edge it lies on is adjacent to a coarser leaf
								bool isNeeded;
								switch( o )
								{
								case 0: isNeeded = ( !_IsValidNode< 0 >( neighborKey.neighbors[depth].neighbors[1][2*y][1] ) || !_IsValidNode< 0 >( neighborKey.neighbors[depth].neighbors[1][2*y][2*z] ) || !_IsValidNode< 0 >( neighborKey.neighbors[depth].neighbors[1][1][2*z] ) ) ; break;
								case 1: isNeeded = ( !_IsValidNode< 0 >( neighborKey.neighbors[depth].neighbors[2*y][1][1] ) || !_IsValidNode< 0 >( neighborKey.neighbors[depth].neighbors[2*y][1][2*z] ) || !_IsValidNode< 0 >( neighborKey.neighbors[depth].neighbors[1][1][2*z] ) ) ; break;
								}
								if( isNeeded )
								{
									int f[2];
									Cube::FacesAdjacentToEdge( Cube::EdgeIndex( o , y , z ) , f[0] , f[1] );
									for( int k=0 ; k<2 ; k++ )
									{
										TreeOctNode* node = leaf;
										int _depth = depth , _slice = slice;
										bool _isNeeded = isNeeded;
										while( _isNeeded && node->parent && Cube::IsFaceCorner( (int)(node-node->parent->children) , f[k] ) )
										{
											node = node->parent , _depth-- , _slice >>= 1;
											typename Octree::template SliceValues< Vertex >& _sValues = slabValues[_depth].sliceValues( _slice );
#pragma omp critical (add_coarser_point_access)
											_sValues.edgeVertexMap[key] = hashed_vertex;
											switch( o )
											{
												case 0: _isNeeded = ( !_IsValidNode< 0 >( neighborKey.neighbors[_depth].neighbors[1][2*y][1] ) || !_IsValidNode< 0 >( neighborKey.neighbors[_depth].neighbors[1][2*y][2*z] ) || !_IsValidNode< 0 >( neighborKey.neighbors[_depth].neighbors[1][1][2*z] ) ) ; break;
												case 1: _isNeeded = ( !_IsValidNode< 0 >( neighborKey.neighbors[_depth].neighbors[2*y][1][1] ) || !_IsValidNode< 0 >( neighborKey.neighbors[_depth].neighbors[2*y][1][2*z] ) || !_IsValidNode< 0 >( neighborKey.neighbors[_depth].neighbors[1][1][2*z] ) ) ; break;
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
template< class Real >
template< int WeightDegree , int ColorDegree , class Vertex >
void Octree< Real >::SetXSliceIsoVertices( const BSplineData< ColorDegree >* colorBSData , const SparseNodeData< Real , WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Point3D< Real > > , ColorDegree >* colorData , Real isoValue , int depth , int slab , int& vOffset , CoredMeshData< Vertex >& mesh , std::vector< SlabValues< Vertex > >& slabValues , int threads )
{
	typename Octree::template  SliceValues< Vertex >& bValues = slabValues[depth].sliceValues ( slab   );
	typename Octree::template  SliceValues< Vertex >& fValues = slabValues[depth].sliceValues ( slab+1 );
	typename Octree::template XSliceValues< Vertex >& xValues = slabValues[depth].xSliceValues( slab   );

	// [WARNING] In the case Degree=2, these two keys are the same, so we don't have to maintain them separately.
	std::vector< ConstAdjacenctNodeKey > neighborKeys( std::max< int >( 1 , threads ) );
	std::vector< ConstPointSupportKey< WeightDegree > > weightKeys( std::max< int >( 1 , threads ) );
	std::vector< ConstPointSupportKey< ColorDegree > > colorKeys( std::max< int >( 1 , threads ) );
	for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( depth ) , weightKeys[i].set( depth ) , colorKeys[i].set( depth );
#pragma omp parallel for num_threads( threads )
	for( int i=_sNodes.begin(depth,slab) ; i<_sNodes.end(depth,slab) ; i++ ) if( _IsValidNode< 0 >( _sNodes.treeNodes[i] ) )
	{
		ConstAdjacenctNodeKey& neighborKey =  neighborKeys[ omp_get_thread_num() ];
		ConstPointSupportKey< WeightDegree >& weightKey = weightKeys[ omp_get_thread_num() ];
		ConstPointSupportKey< ColorDegree >& colorKey = colorKeys[ omp_get_thread_num() ];
		TreeOctNode* leaf = _sNodes.treeNodes[i];
		if( !leaf->children )
		{
			unsigned char mcIndex = ( bValues.mcIndices[ i - bValues.sliceData.nodeOffset ] ) | ( fValues.mcIndices[ i - fValues.sliceData.nodeOffset ] )<<4;
			const typename SortedTreeNodes::SquareCornerIndices& eIndices = xValues.xSliceData.edgeIndices( leaf );
			if( MarchingCubes::HasRoots( mcIndex ) )
			{
				neighborKey.getNeighbors( leaf );
				if( densityWeights ) weightKey.getNeighbors( leaf );
				if( colorData ) colorKey.getNeighbors( leaf );
				for( int x=0 ; x<2 ; x++ ) for( int y=0 ; y<2 ; y++ )
				{
					int c = Square::CornerIndex( x , y );
					int e = Cube::EdgeIndex( 2 , x , y );
					if( MarchingCubes::HasEdgeRoots( mcIndex , e ) )
					{
						int vIndex = eIndices[c];
						if( !xValues.edgeSet[vIndex] )
						{
							Vertex vertex;
							long long key = VertexData::EdgeIndex( leaf , e , _sNodes.levels() );
							GetIsoVertex( colorBSData , densityWeights , colorData , isoValue , weightKey , colorKey , leaf , c , bValues , fValues , vertex );
							vertex.point = vertex.point * _scale + _center;
							bool stillOwner = false;
							std::pair< int , Vertex > hashed_vertex;
#pragma omp critical (add_x_point_access)
							{
								if( !xValues.edgeSet[vIndex] )
								{
									mesh.addOutOfCorePoint( vertex );
									xValues.edgeSet[ vIndex ] = 1;
									xValues.edgeKeys[ vIndex ] = key;
									xValues.edgeVertexMap[key] = hashed_vertex = std::pair< int , Vertex >( vOffset , vertex );
									stillOwner = true;
									vOffset++;
								}
							}
							if( stillOwner )
							{
								// We only need to pass the iso-vertex down if the edge it lies on is adjacent to a coarser leaf
								bool isNeeded = ( !_IsValidNode< 0 >( neighborKey.neighbors[depth].neighbors[2*x][1][1] ) || !_IsValidNode< 0 >( neighborKey.neighbors[depth].neighbors[2*x][2*y][1] ) || !_IsValidNode< 0 >( neighborKey.neighbors[depth].neighbors[1][2*y][1] ) );
								if( isNeeded )
								{
									int f[2];
									Cube::FacesAdjacentToEdge( e , f[0] , f[1] );
									for( int k=0 ; k<2 ; k++ )
									{
										TreeOctNode* node = leaf;
										int _depth = depth , _slab = slab;
										bool _isNeeded = isNeeded;
										while( _isNeeded && node->parent && Cube::IsFaceCorner( (int)(node-node->parent->children) , f[k] ) )
										{
											node = node->parent , _depth-- , _slab >>= 1;
											typename Octree::template XSliceValues< Vertex >& _xValues = slabValues[_depth].xSliceValues( _slab );
#pragma omp critical (add_x_coarser_point_access)
											_xValues.edgeVertexMap[key] = hashed_vertex;
											_isNeeded = ( !_IsValidNode< 0 >( neighborKey.neighbors[_depth].neighbors[2*x][1][1] ) || !_IsValidNode< 0 >( neighborKey.neighbors[_depth].neighbors[2*x][2*y][1] ) || !_IsValidNode< 0 >( neighborKey.neighbors[_depth].neighbors[1][2*y][1] ) );
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
template< class Real >
template< class Vertex >
void Octree< Real >::CopyFinerSliceIsoEdgeKeys( int depth , int slice , std::vector< SlabValues< Vertex > >& slabValues , int threads )
{
	if( slice>0          ) CopyFinerSliceIsoEdgeKeys( depth , slice , 1 , slabValues , threads );
	if( slice<(1<<depth) ) CopyFinerSliceIsoEdgeKeys( depth , slice , 0 , slabValues , threads );
}
template< class Real >
template< class Vertex >
void Octree< Real >::CopyFinerSliceIsoEdgeKeys( int depth , int slice , int z , std::vector< SlabValues< Vertex > >& slabValues , int threads )
{
	SliceValues< Vertex >& pSliceValues = slabValues[depth  ].sliceValues(slice   );
	SliceValues< Vertex >& cSliceValues = slabValues[depth+1].sliceValues(slice<<1);
	typename SortedTreeNodes::SliceTableData& pSliceData = pSliceValues.sliceData;
	typename SortedTreeNodes::SliceTableData& cSliceData = cSliceValues.sliceData;
#pragma omp parallel for num_threads( threads )
	for( int i=_sNodes.begin(depth,slice-z) ; i<_sNodes.end(depth,slice-z) ; i++ ) if( _IsValidNode< 0 >( _sNodes.treeNodes[i] ) )
		if( _sNodes.treeNodes[i]->children )
		{
			typename SortedTreeNodes::SquareEdgeIndices& pIndices = pSliceData.edgeIndices( i );
			// Copy the edges that overlap the coarser edges
			for( int orientation=0 ; orientation<2 ; orientation++ ) for( int y=0 ; y<2 ; y++ )
			{
				int fe = Square::EdgeIndex( orientation , y );
				int pIndex = pIndices[fe];
				if( !pSliceValues.edgeSet[ pIndex ] )
				{
					int ce = Cube::EdgeIndex( orientation , y , z );
					int c1 , c2;
					switch( orientation )
					{
					case 0: c1 = Cube::CornerIndex( 0 , y , z ) , c2 = Cube::CornerIndex( 1 , y , z ) ; break;
					case 1: c1 = Cube::CornerIndex( y , 0 , z ) , c2 = Cube::CornerIndex( y , 1 , z ) ; break;
					}
					// [SANITY CHECK]
//					if( _IsValidNode< 0 >( _sNodes.treeNodes[i]->children + c1 )!=_IsValidNode< 0 >( _sNodes.treeNodes[i]->children + c2 ) ) fprintf( stderr , "[WARNING] Finer edges should both be valid or invalid\n" ) , exit( 0 );
					if( !_IsValidNode< 0 >( _sNodes.treeNodes[i]->children + c1 ) || !_IsValidNode< 0 >( _sNodes.treeNodes[i]->children + c2 ) ) continue;

					int cIndex1 = cSliceData.edgeIndices( _sNodes.treeNodes[i]->children + c1 )[fe];
					int cIndex2 = cSliceData.edgeIndices( _sNodes.treeNodes[i]->children + c2 )[fe];
					if( cSliceValues.edgeSet[cIndex1] != cSliceValues.edgeSet[cIndex2] )
					{
						long long key;
						if( cSliceValues.edgeSet[cIndex1] ) key = cSliceValues.edgeKeys[cIndex1];
						else                                key = cSliceValues.edgeKeys[cIndex2];
						std::pair< int , Vertex > vPair = cSliceValues.edgeVertexMap.find( key )->second;
#pragma omp critical ( copy_finer_edge_keys )
						pSliceValues.edgeVertexMap[key] = vPair;
						pSliceValues.edgeKeys[pIndex] = key;
						pSliceValues.edgeSet[pIndex] = 1;
					}
					else if( cSliceValues.edgeSet[cIndex1] && cSliceValues.edgeSet[cIndex2] )
					{
						long long key1 = cSliceValues.edgeKeys[cIndex1] , key2 = cSliceValues.edgeKeys[cIndex2];
#pragma omp critical ( set_edge_pairs )
						pSliceValues.vertexPairMap[ key1 ] = key2 ,	pSliceValues.vertexPairMap[ key2 ] = key1;

						const TreeOctNode* node = _sNodes.treeNodes[i];
						int _depth = depth , _slice = slice;
						while( node->parent && Cube::IsEdgeCorner( (int)( node - node->parent->children ) , ce ) )
						{
							node = node->parent , _depth-- , _slice >>= 1;
							SliceValues< Vertex >& _pSliceValues = slabValues[_depth].sliceValues(_slice);
#pragma omp critical ( set_edge_pairs )
							_pSliceValues.vertexPairMap[ key1 ] = key2 , _pSliceValues.vertexPairMap[ key2 ] = key1;
						}
					}
				}
			}
		}
}
template< class Real >
template< class Vertex >
void Octree< Real >::CopyFinerXSliceIsoEdgeKeys( int depth , int slab , std::vector< SlabValues< Vertex > >& slabValues , int threads )
{
	XSliceValues< Vertex >& pSliceValues  = slabValues[depth  ].xSliceValues(slab);
	XSliceValues< Vertex >& cSliceValues0 = slabValues[depth+1].xSliceValues( (slab<<1)|0 );
	XSliceValues< Vertex >& cSliceValues1 = slabValues[depth+1].xSliceValues( (slab<<1)|1 );
	typename SortedTreeNodes::XSliceTableData& pSliceData  = pSliceValues.xSliceData;
	typename SortedTreeNodes::XSliceTableData& cSliceData0 = cSliceValues0.xSliceData;
	typename SortedTreeNodes::XSliceTableData& cSliceData1 = cSliceValues1.xSliceData;
#pragma omp parallel for num_threads( threads )
	for( int i=_sNodes.begin(depth,slab) ; i<_sNodes.end(depth,slab) ; i++ ) if( _IsValidNode< 0 >( _sNodes.treeNodes[i] ) )
		if( _sNodes.treeNodes[i]->children )
		{
			typename SortedTreeNodes::SquareCornerIndices& pIndices = pSliceData.edgeIndices( i );
			for( int x=0 ; x<2 ; x++ ) for( int y=0 ; y<2 ; y++ )
			{
				int fc = Square::CornerIndex( x , y );
				int pIndex = pIndices[fc];
				if( !pSliceValues.edgeSet[pIndex] )
				{
					int c0 = Cube::CornerIndex( x , y , 0 ) , c1 = Cube::CornerIndex( x , y , 1 );

					// [SANITY CHECK]
//					if( _IsValidNode< 0 >( _sNodes.treeNodes[i]->children + c0 )!=_IsValidNode< 0 >( _sNodes.treeNodes[i]->children + c1 ) ) fprintf( stderr , "[ERROR] Finer edges should both be valid or invalid\n" ) , exit( 0 );
					if( !_IsValidNode< 0 >( _sNodes.treeNodes[i]->children + c0 ) || !_IsValidNode< 0 >( _sNodes.treeNodes[i]->children + c1 ) ) continue;

					int cIndex0 = cSliceData0.edgeIndices( _sNodes.treeNodes[i]->children + c0 )[fc];
					int cIndex1 = cSliceData1.edgeIndices( _sNodes.treeNodes[i]->children + c1 )[fc];
					if( cSliceValues0.edgeSet[cIndex0] != cSliceValues1.edgeSet[cIndex1] )
					{
						long long key;
						std::pair< int , Vertex > vPair;
						if( cSliceValues0.edgeSet[cIndex0] ) key = cSliceValues0.edgeKeys[cIndex0] , vPair = cSliceValues0.edgeVertexMap.find( key )->second;
						else                                 key = cSliceValues1.edgeKeys[cIndex1] , vPair = cSliceValues1.edgeVertexMap.find( key )->second;
#pragma omp critical ( copy_finer_x_edge_keys )
						pSliceValues.edgeVertexMap[key] = vPair;
						pSliceValues.edgeKeys[ pIndex ] = key;
						pSliceValues.edgeSet[ pIndex ] = 1;
					}
					else if( cSliceValues0.edgeSet[cIndex0] && cSliceValues1.edgeSet[cIndex1] )
					{
						long long key0 = cSliceValues0.edgeKeys[cIndex0] , key1 = cSliceValues1.edgeKeys[cIndex1];
#pragma omp critical ( set_x_edge_pairs )
						pSliceValues.vertexPairMap[ key0 ] = key1 , pSliceValues.vertexPairMap[ key1 ] = key0;
						const TreeOctNode* node = _sNodes.treeNodes[i];
						int _depth = depth , _slab = slab , ce = Cube::CornerIndex( 2 , x , y );
						while( node->parent && Cube::IsEdgeCorner( (int)( node - node->parent->children ) , ce ) )
						{
							node = node->parent , _depth-- , _slab>>= 1;
							SliceValues< Vertex >& _pSliceValues = slabValues[_depth].sliceValues(_slab);
#pragma omp critical ( set_x_edge_pairs )
							_pSliceValues.vertexPairMap[ key0 ] = key1 , _pSliceValues.vertexPairMap[ key1 ] = key0;
						}
					}
				}
			}
		}
}
template< class Real >
template< class Vertex >
void Octree< Real >::SetSliceIsoEdges( int depth , int slice , std::vector< SlabValues< Vertex > >& slabValues , int threads )
{
	if( slice>0          ) SetSliceIsoEdges( depth , slice , 1 , slabValues , threads );
	if( slice<(1<<depth) ) SetSliceIsoEdges( depth , slice , 0 , slabValues , threads );
}
template< class Real >
template< class Vertex >
void Octree< Real >::SetSliceIsoEdges( int depth , int slice , int z , std::vector< SlabValues< Vertex > >& slabValues , int threads )
{
	typename Octree::template SliceValues< Vertex >& sValues = slabValues[depth].sliceValues( slice );
	std::vector< ConstAdjacenctNodeKey > neighborKeys( std::max< int >( 1 , threads ) );
	for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( depth );
#pragma omp parallel for num_threads( threads )
	for( int i=_sNodes.begin(depth,slice-z) ; i<_sNodes.end(depth,slice-z) ; i++ ) if( _IsValidNode< 0 >( _sNodes.treeNodes[i] ) )
	{
		int isoEdges[ 2 * MarchingSquares::MAX_EDGES ];
		ConstAdjacenctNodeKey& neighborKey = neighborKeys[ omp_get_thread_num() ];
		TreeOctNode* leaf = _sNodes.treeNodes[i];
		if( !leaf->children )
		{
			int idx = i - sValues.sliceData.nodeOffset;
			const typename SortedTreeNodes::SquareEdgeIndices& eIndices = sValues.sliceData.edgeIndices( leaf );
			const typename SortedTreeNodes::SquareFaceIndices& fIndices = sValues.sliceData.faceIndices( leaf );
			unsigned char mcIndex = sValues.mcIndices[idx];
			if( !sValues.faceSet[ fIndices[0] ] )
			{
				neighborKey.getNeighbors( leaf );
				if( !neighborKey.neighbors[depth].neighbors[1][1][2*z] || !neighborKey.neighbors[depth].neighbors[1][1][2*z]->children )
				{
					FaceEdges fe;
					fe.count = MarchingSquares::AddEdgeIndices( mcIndex , isoEdges );
					for( int j=0 ; j<fe.count ; j++ ) for( int k=0 ; k<2 ; k++ )
					{
						if( !sValues.edgeSet[ eIndices[ isoEdges[2*j+k] ] ] ) fprintf( stderr , "[ERROR] Edge not set 1: %d / %d\n" , slice , 1<<depth ) , exit( 0 );
						fe.edges[j][k] = sValues.edgeKeys[ eIndices[ isoEdges[2*j+k] ] ];
					}
					sValues.faceSet[ fIndices[0] ] = 1;
					sValues.faceEdges[ fIndices[0] ] = fe;

					TreeOctNode* node = leaf;
					int _depth = depth , _slice = slice , f = Cube::FaceIndex( 2 , z );
					std::vector< IsoEdge > edges;
					edges.resize( fe.count );
					for( int j=0 ; j<fe.count ; j++ ) edges[j] = fe.edges[j];
					while( node->parent && Cube::IsFaceCorner( (int)(node-node->parent->children) , f ) )
					{
						node = node->parent , _depth-- , _slice >>= 1;
						if( neighborKey.neighbors[_depth].neighbors[1][1][2*z] && neighborKey.neighbors[_depth].neighbors[1][1][2*z]->children ) break;
						long long key = VertexData::FaceIndex( node , f , _sNodes.levels() );
#pragma omp critical( add_iso_edge_access )
						{
							typename Octree::template SliceValues< Vertex >& _sValues = slabValues[_depth].sliceValues( _slice );
							typename hash_map< long long , std::vector< IsoEdge > >::iterator iter = _sValues.faceEdgeMap.find(key);
							if( iter==_sValues.faceEdgeMap.end() ) _sValues.faceEdgeMap[key] = edges;
							else for( int j=0 ; j<fe.count ; j++ ) iter->second.push_back( fe.edges[j] );
						}
					}
				}
			}
		}
	}
}
template< class Real >
template< class Vertex >
void Octree< Real >::SetXSliceIsoEdges( int depth , int slab , std::vector< SlabValues< Vertex > >& slabValues , int threads )
{
	typename Octree::template  SliceValues< Vertex >& bValues = slabValues[depth].sliceValues ( slab   );
	typename Octree::template  SliceValues< Vertex >& fValues = slabValues[depth].sliceValues ( slab+1 );
	typename Octree::template XSliceValues< Vertex >& xValues = slabValues[depth].xSliceValues( slab   );

	std::vector< ConstAdjacenctNodeKey > neighborKeys( std::max< int >( 1 , threads ) );
	for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( depth );
#pragma omp parallel for num_threads( threads )
	for( int i=_sNodes.begin(depth,slab) ; i<_sNodes.end(depth,slab) ; i++ ) if( _IsValidNode< 0 >( _sNodes.treeNodes[i] ) )
	{
		int isoEdges[ 2 * MarchingSquares::MAX_EDGES ];
		ConstAdjacenctNodeKey& neighborKey = neighborKeys[ omp_get_thread_num() ];
		TreeOctNode* leaf = _sNodes.treeNodes[i];
		if( !leaf->children )
		{
			const typename SortedTreeNodes::SquareCornerIndices& cIndices = xValues.xSliceData.edgeIndices( leaf );
			const typename SortedTreeNodes::SquareEdgeIndices& eIndices = xValues.xSliceData.faceIndices( leaf );
			unsigned char mcIndex = ( bValues.mcIndices[ i - bValues.sliceData.nodeOffset ] ) | ( fValues.mcIndices[ i - fValues.sliceData.nodeOffset ]<<4 );
			{
				neighborKey.getNeighbors( leaf );
				for( int o=0 ; o<2 ; o++ ) for( int x=0 ; x<2 ; x++ )
				{
					int e = Square::EdgeIndex( o , x );
					int f = Cube::FaceIndex( 1-o , x );
					unsigned char _mcIndex = MarchingCubes::GetFaceIndex( mcIndex , f );
					int xx = o==1 ? 2*x : 1 , yy = o==0 ? 2*x : 1 , zz = 1;
					if(	!xValues.faceSet[ eIndices[e] ] && ( !neighborKey.neighbors[depth].neighbors[xx][yy][zz] || !neighborKey.neighbors[depth].neighbors[xx][yy][zz]->children ) )
					{
						FaceEdges fe;
						fe.count = MarchingSquares::AddEdgeIndices( _mcIndex , isoEdges );
						for( int j=0 ; j<fe.count ; j++ ) for( int k=0 ; k<2 ; k++ )
						{
							int _o , _x;
							Square::FactorEdgeIndex( isoEdges[2*j+k] , _o , _x );
							if( _o==1 ) // Cross-edge
							{
								int idx = o==0 ? cIndices[ Square::CornerIndex(_x,x) ] : cIndices[ Square::CornerIndex(x,_x) ];
								if( !xValues.edgeSet[ idx ] ) fprintf( stderr , "[ERROR] Edge not set 3: %d / %d\n" , slab , 1<<depth ) , exit( 0 );
								fe.edges[j][k] = xValues.edgeKeys[ idx ];
							}
							else
							{
								const typename Octree::template SliceValues< Vertex >& sValues = (_x==0) ? bValues : fValues;
								int idx = sValues.sliceData.edgeIndices(i)[ Square::EdgeIndex(o,x) ];
								if( !sValues.edgeSet[ idx ] ) fprintf( stderr , "[ERROR] Edge not set 5: %d / %d\n" , slab , 1<<depth ) , exit( 0 );
								fe.edges[j][k] = sValues.edgeKeys[ idx ];
							}
						}
						xValues.faceSet[ eIndices[e] ] = 1;
						xValues.faceEdges[ eIndices[e] ] = fe;

						TreeOctNode* node = leaf;
						int _depth = depth , _slab = slab;
						std::vector< IsoEdge > edges;
						edges.resize( fe.count );
						for( int j=0 ; j<fe.count ; j++ ) edges[j] = fe.edges[j];
						while( node->parent && Cube::IsFaceCorner( (int)(node-node->parent->children) , f ) )
						{
							node = node->parent , _depth-- , _slab >>= 1;
							if( neighborKey.neighbors[_depth].neighbors[xx][yy][zz] && neighborKey.neighbors[_depth].neighbors[xx][yy][zz]->children ) break;
							long long key = VertexData::FaceIndex( node , f , _sNodes.levels() );
#pragma omp critical( add_x_iso_edge_access )
							{
								typename Octree::template XSliceValues< Vertex >& _xValues = slabValues[_depth].xSliceValues( _slab );
								typename hash_map< long long , std::vector< IsoEdge > >::iterator iter = _xValues.faceEdgeMap.find(key);
								if( iter==_xValues.faceEdgeMap.end() ) _xValues.faceEdgeMap[key] = edges;
								else for( int j=0 ; j<fe.count ; j++ ) iter->second.push_back( fe.edges[j] );
							}
						}
					}
				}
			}
		}
	}
}
template< class Real >
template< class Vertex >
void Octree< Real >::SetIsoSurface( int depth , int offset , const SliceValues< Vertex >& bValues , const SliceValues< Vertex >& fValues , const XSliceValues< Vertex >& xValues , CoredMeshData< Vertex >& mesh , bool polygonMesh , bool addBarycenter , int& vOffset , int threads )
{
	std::vector< std::pair< int , Vertex > > polygon;
	std::vector< std::vector< IsoEdge > > edgess( std::max< int >( 1 , threads ) );
#pragma omp parallel for num_threads( threads )
	for( int i=_sNodes.begin(depth,offset) ; i<_sNodes.end(depth,offset) ; i++ ) if( _IsValidNode< 0 >( _sNodes.treeNodes[i] ) )
	{
		std::vector< IsoEdge >& edges = edgess[ omp_get_thread_num() ];
		TreeOctNode* leaf = _sNodes.treeNodes[i];
		int d , off[3];
		leaf->depthAndOffset( d , off );
		int res = _Resolution( depth );
		bool inBounds = off[0]<res && off[1]<res && off[2]<res;
		if( inBounds&& !leaf->children )
		{
			edges.clear();
			unsigned char mcIndex = ( bValues.mcIndices[ i - bValues.sliceData.nodeOffset ] ) | ( fValues.mcIndices[ i - fValues.sliceData.nodeOffset ]<<4 );
			// [WARNING] Just because the node looks empty doesn't mean it doesn't get eges from finer neighbors
			{
				// Gather the edges from the faces (with the correct orientation)
				for( int f=0 ; f<Cube::FACES ; f++ )
				{
					int d , o;
					Cube::FactorFaceIndex( f , d , o );
					int flip = d==1 ? 1 : 0; // To account for the fact that the section in y flips the orientation
					if( o ) flip = 1-flip;
					flip = 1-flip; // To get the right orientation
					if( d==2 )
					{
						const SliceValues< Vertex >& sValues = (o==0) ? bValues : fValues;
						int fIdx = sValues.sliceData.faceIndices(i)[0];
						if( sValues.faceSet[fIdx] )
						{
							const FaceEdges& fe = sValues.faceEdges[ fIdx ];
							for( int j=0 ; j<fe.count ; j++ ) edges.push_back( IsoEdge( fe.edges[j][flip] , fe.edges[j][1-flip] ) );
						}
						else
						{
							long long key = VertexData::FaceIndex( leaf , f , _sNodes.levels() );
							typename hash_map< long long , std::vector< IsoEdge > >::const_iterator iter = sValues.faceEdgeMap.find( key );
							if( iter!=sValues.faceEdgeMap.end() )
							{
								const std::vector< IsoEdge >& _edges = iter->second;
								for( size_t j=0 ; j<_edges.size() ; j++ ) edges.push_back( IsoEdge( _edges[j][flip] , _edges[j][1-flip] ) );
							}
							else fprintf( stderr , "[ERROR] Invalid faces: %d  %d %d\n" , i , d , o ) , exit( 0 );
						}
					}
					else
					{
						int fIdx = xValues.xSliceData.faceIndices(i)[ Square::EdgeIndex( 1-d , o ) ];
						if( xValues.faceSet[fIdx] )
						{
							const FaceEdges& fe = xValues.faceEdges[ fIdx ];
							for( int j=0 ; j<fe.count ; j++ ) edges.push_back( IsoEdge( fe.edges[j][flip] , fe.edges[j][1-flip] ) );
						}
						else
						{
							long long key = VertexData::FaceIndex( leaf , f , _sNodes.levels() );
							typename hash_map< long long , std::vector< IsoEdge > >::const_iterator iter = xValues.faceEdgeMap.find( key );
							if( iter!=xValues.faceEdgeMap.end() )
							{
								const std::vector< IsoEdge >& _edges = iter->second;
								for( size_t j=0 ; j<_edges.size() ; j++ ) edges.push_back( IsoEdge( _edges[j][flip] , _edges[j][1-flip] ) );
							}
							else fprintf( stderr , "[ERROR] Invalid faces: %d  %d %d\n" , i , d , o ) , exit( 0 );
						}
					}
				}
				// Get the edge loops
				std::vector< std::vector< long long  > > loops;
				while( edges.size() )
				{
					loops.resize( loops.size()+1 );
					IsoEdge edge = edges.back();
					edges.pop_back();
					long long start = edge[0] , current = edge[1];
					while( current!=start )
					{
						int idx;
						for( idx=0 ; idx<(int)edges.size() ; idx++ ) if( edges[idx][0]==current ) break;
						if( idx==edges.size() )
						{
							typename hash_map< long long , long long >::const_iterator iter;
							if     ( (iter=bValues.vertexPairMap.find(current))!=bValues.vertexPairMap.end() ) loops.back().push_back( current ) , current = iter->second;
							else if( (iter=fValues.vertexPairMap.find(current))!=fValues.vertexPairMap.end() ) loops.back().push_back( current ) , current = iter->second;
							else if( (iter=xValues.vertexPairMap.find(current))!=xValues.vertexPairMap.end() ) loops.back().push_back( current ) , current = iter->second;
							else
							{
								int d , off[3];
								leaf->depthAndOffset( d , off );
								fprintf( stderr , "[ERROR] Failed to close loop [%d: %d %d %d] | (%d): %lld\n" , d-1 , off[0] , off[1] , off[2] , i , current );
								exit( 0 );
							}
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
					std::vector< std::pair< int , Vertex > > polygon( loops[j].size() );
					for( size_t k=0 ; k<loops[j].size() ; k++ )
					{
						long long key = loops[j][k];
						typename hash_map< long long , std::pair< int , Vertex > >::const_iterator iter;
						if     ( ( iter=bValues.edgeVertexMap.find( key ) )!=bValues.edgeVertexMap.end() ) polygon[k] = iter->second;
						else if( ( iter=fValues.edgeVertexMap.find( key ) )!=fValues.edgeVertexMap.end() ) polygon[k] = iter->second;
						else if( ( iter=xValues.edgeVertexMap.find( key ) )!=xValues.edgeVertexMap.end() ) polygon[k] = iter->second;
						else fprintf( stderr , "[ERROR] Couldn't find vertex in edge map\n" ) , exit( 0 );
					}
					AddIsoPolygons( mesh , polygon , polygonMesh , addBarycenter , vOffset );
				}
			}
		}
	}
}
template< class Real > void SetColor( Point3D< Real >& color , unsigned char c[3] ){ for( int i=0 ; i<3 ; i++ ) c[i] = (unsigned char)std::max< int >( 0 , std::min< int >( 255 , (int)( color[i]+0.5 ) ) ); }

template< class Real > void SetIsoVertex(              PlyVertex< float  >& vertex , Point3D< Real > color , Real value ){ ; }
template< class Real > void SetIsoVertex(         PlyColorVertex< float  >& vertex , Point3D< Real > color , Real value ){ SetColor( color , vertex.color ); }
template< class Real > void SetIsoVertex(         PlyValueVertex< float  >& vertex , Point3D< Real > color , Real value ){                                    vertex.value = float(value); }
template< class Real > void SetIsoVertex( PlyColorAndValueVertex< float  >& vertex , Point3D< Real > color , Real value ){ SetColor( color , vertex.color ) , vertex.value = float(value); }
template< class Real > void SetIsoVertex(              PlyVertex< double >& vertex , Point3D< Real > color , Real value ){ ; }
template< class Real > void SetIsoVertex(         PlyColorVertex< double >& vertex , Point3D< Real > color , Real value ){ SetColor( color , vertex.color ); }
template< class Real > void SetIsoVertex(         PlyValueVertex< double >& vertex , Point3D< Real > color , Real value ){                                    vertex.value = double(value); }
template< class Real > void SetIsoVertex( PlyColorAndValueVertex< double >& vertex , Point3D< Real > color , Real value ){ SetColor( color , vertex.color ) , vertex.value = double(value); }

template< class Real >
template< int WeightDegree , int ColorDegree , class Vertex >
bool Octree< Real >::GetIsoVertex( const BSplineData< ColorDegree >* colorBSData , const SparseNodeData< Real , WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Point3D< Real > > , ColorDegree >* colorData , Real isoValue , ConstPointSupportKey< WeightDegree >& weightKey , ConstPointSupportKey< ColorDegree >& colorKey , const TreeOctNode* node , int edgeIndex , int z , const SliceValues< Vertex >& sValues , Vertex& vertex )
{
	Point3D< Real > position;
	int c0 , c1;
	Square::EdgeCorners( edgeIndex , c0 , c1 );

	bool nonLinearFit = sValues.cornerGradients!=NullPointer( Point3D< Real > );
	const typename SortedTreeNodes::SquareCornerIndices& idx = sValues.sliceData.cornerIndices( node );
	Real x0 = sValues.cornerValues[idx[c0]] , x1 = sValues.cornerValues[idx[c1]];
	Point3D< Real > s;
	Real start , width;
	_StartAndWidth( node , s , width );
	int o , y;
	Square::FactorEdgeIndex( edgeIndex , o , y );
	start = s[o];
	switch( o )
	{
	case 0:
		position[1] = s[1] + width*y;
		position[2] = s[2] + width*z;
		break;
	case 1:
		position[0] = s[0] + width*y;
		position[2] = s[2] + width*z;
		break;
	}

	double averageRoot;
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
		int rCount = 0 , rootCount = P.getSolutions( isoValue , roots , EPSILON );
		averageRoot = 0;
		for( int i=0 ; i<rootCount ; i++ ) if( roots[i]>=0 && roots[i]<=1 ) averageRoot += roots[i] , rCount++;
		averageRoot /= rCount;
	}
	else
	{
		// We have a linear function L, with L(0) = x0 and L(1) = x1
		// => L(t) = x0 + t * (x1-x0)
		// => L(t) = isoValue <=> t = ( isoValue - x0 ) / ( x1 - x0 )
		if( x0==x1 ) fprintf( stderr , "[ERROR] Not a zero-crossing root: %g %g\n" , x0 , x1 ) , exit( 0 );
		averageRoot = ( isoValue - x0 ) / ( x1 - x0 );
	}
	if( averageRoot<0 || averageRoot>1 )
	{
		fprintf( stderr , "[WARNING] Bad average root: %f\n" , averageRoot );
		fprintf( stderr , "\t(%f %f) (%f)\n" , x0 , x1 , isoValue );
		if( averageRoot<0 ) averageRoot = 0;
		if( averageRoot>1 ) averageRoot = 1;
	}
	position[o] = Real( start + width*averageRoot );
	vertex.point = position;
	Point3D< Real > color;
	Real depth(0);
	if( densityWeights )
	{
		Real weight;
		const TreeOctNode* temp = node;
		while( _Depth( temp )>_splatDepth ) temp=temp->parent;
		_GetSampleDepthAndWeight( *densityWeights , temp , position , weightKey , depth , weight );
	}
	if( colorData ) color = Point3D< Real >( _Evaluate( *colorData , position , *colorBSData , colorKey ) );
	SetIsoVertex( vertex , color , depth );
	return true;
}
template< class Real >
template< int WeightDegree , int ColorDegree , class Vertex >
bool Octree< Real >::GetIsoVertex( const BSplineData< ColorDegree >* colorBSData , const SparseNodeData< Real , WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Point3D< Real > > , ColorDegree >* colorData , Real isoValue , ConstPointSupportKey< WeightDegree >& weightKey , ConstPointSupportKey< ColorDegree >& colorKey , const TreeOctNode* node , int cornerIndex , const SliceValues< Vertex >& bValues , const SliceValues< Vertex >& fValues , Vertex& vertex )
{
	Point3D< Real > position;

	bool nonLinearFit = bValues.cornerGradients!=NullPointer( Point3D< Real > ) && fValues.cornerGradients!=NullPointer( Point3D< Real > );
	const typename SortedTreeNodes::SquareCornerIndices& idx0 = bValues.sliceData.cornerIndices( node );
	const typename SortedTreeNodes::SquareCornerIndices& idx1 = fValues.sliceData.cornerIndices( node );
	Real x0 = bValues.cornerValues[ idx0[cornerIndex] ] , x1 = fValues.cornerValues[ idx1[cornerIndex] ];
	Point3D< Real > s;
	Real start , width;
	_StartAndWidth( node , s , width );
	start = s[2];
	int x , y;
	Square::FactorCornerIndex( cornerIndex , x , y );


	position[0] = s[0] + width*x;
	position[1] = s[1] + width*y;

	double averageRoot;

	if( nonLinearFit )
	{
		double dx0 = bValues.cornerGradients[ idx0[cornerIndex] ][2] * width , dx1 = fValues.cornerGradients[ idx1[cornerIndex] ][2] * width;
		// The scaling will turn the Hermite Spline into a quadratic
		double scl = (x1-x0) / ( (dx1+dx0 ) / 2 );
		dx0 *= scl , dx1 *= scl;

		// Hermite Spline
		Polynomial< 2 > P;
		P.coefficients[0] = x0;
		P.coefficients[1] = dx0;
		P.coefficients[2] = 3*(x1-x0)-dx1-2*dx0;

		double roots[2];
		int rCount = 0 , rootCount = P.getSolutions( isoValue , roots , EPSILON );
		averageRoot = 0;
		for( int i=0 ; i<rootCount ; i++ ) if( roots[i]>=0 && roots[i]<=1 ) averageRoot += roots[i] , rCount++;
		averageRoot /= rCount;
	}
	else
	{
		// We have a linear function L, with L(0) = x0 and L(1) = x1
		// => L(t) = x0 + t * (x1-x0)
		// => L(t) = isoValue <=> t = ( isoValue - x0 ) / ( x1 - x0 )
		if( x0==x1 ) fprintf( stderr , "[ERROR] Not a zero-crossing root: %g %g\n" , x0 , x1 ) , exit( 0 );
		averageRoot = ( isoValue - x0 ) / ( x1 - x0 );
	}
	if( averageRoot<0 || averageRoot>1 )
	{
		fprintf( stderr , "[WARNING] Bad average root: %f\n" , averageRoot );
		fprintf( stderr , "\t(%f %f) (%f)\n" , x0 , x1 , isoValue );
		if( averageRoot<0 ) averageRoot = 0;
		if( averageRoot>1 ) averageRoot = 1;
	}
	position[2] = Real( start + width*averageRoot );
	vertex.point = position;
	Point3D< Real > color;
	Real depth(0);
	if( densityWeights )
	{
		Real weight;
		const TreeOctNode* temp = node;
		while( _Depth( temp )>_splatDepth ) temp=temp->parent;
		_GetSampleDepthAndWeight( *densityWeights , temp , position , weightKey , depth , weight );
	}
	if( colorData ) color = Point3D< Real >( _Evaluate( *colorData , position , *colorBSData , colorKey ) );
	SetIsoVertex( vertex , color , depth );
	return true;
}

template< class Real >
template< class Vertex >
int Octree< Real >::AddIsoPolygons( CoredMeshData< Vertex >& mesh , std::vector< std::pair< int , Vertex > >& polygon , bool polygonMesh , bool addBarycenter , int& vOffset )
{
	if( polygonMesh )
	{
		std::vector< int > vertices( polygon.size() );
		for( int i=0 ; i<(int)polygon.size() ; i++ ) vertices[i] = polygon[polygon.size()-1-i].first;
		mesh.addPolygon_s( vertices );
		return 1;
	}
	if( polygon.size()>3 )
	{
		bool isCoplanar = false;
		std::vector< int > triangle( 3 );

		if( addBarycenter )
			for( int i=0 ; i<(int)polygon.size() ; i++ )
				for( int j=0 ; j<i ; j++ )
					if( (i+1)%polygon.size()!=j && (j+1)%polygon.size()!=i )
					{
						Vertex v1 = polygon[i].second , v2 = polygon[j].second;
						for( int k=0 ; k<3 ; k++ ) if( v1.point[k]==v2.point[k] ) isCoplanar = true;
					}
		if( isCoplanar )
		{
			Vertex c;
			typename Vertex::Wrapper _c;
			_c *= 0;
			for( int i=0 ; i<(int)polygon.size() ; i++ ) _c += typename Vertex::Wrapper( polygon[i].second );
			_c /= Real( polygon.size() );
			c = Vertex( _c );
			int cIdx;
#pragma omp critical (add_barycenter_point_access)
			{
				cIdx = mesh.addOutOfCorePoint( c );
				vOffset++;
			}
			for( int i=0 ; i<(int)polygon.size() ; i++ )
			{
				triangle[0] = polygon[ i                  ].first;
				triangle[1] = cIdx;
				triangle[2] = polygon[(i+1)%polygon.size()].first;
				mesh.addPolygon_s( triangle );
			}
			return (int)polygon.size();
		}
		else
		{
			MinimalAreaTriangulation< Real > MAT;
			std::vector< Point3D< Real > > vertices;
			std::vector< TriangleIndex > triangles;
			vertices.resize( polygon.size() );
			// Add the points
			for( int i=0 ; i<(int)polygon.size() ; i++ ) vertices[i] = polygon[i].second.point;
			MAT.GetTriangulation( vertices , triangles );
			for( int i=0 ; i<(int)triangles.size() ; i++ )
			{
				for( int j=0 ; j<3 ; j++ ) triangle[2-j] = polygon[ triangles[i].idx[j] ].first;
				mesh.addPolygon_s( triangle );
			}
		}
	}
	else if( polygon.size()==3 )
	{
		std::vector< int > vertices( 3 );
		for( int i=0 ; i<3 ; i++ ) vertices[2-i] = polygon[i].first;
		mesh.addPolygon_s( vertices );
	}
	return (int)polygon.size()-2;
}
