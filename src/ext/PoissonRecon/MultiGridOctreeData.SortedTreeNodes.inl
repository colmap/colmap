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

/////////////////////
// SortedTreeNodes //
/////////////////////
SortedTreeNodes::SortedTreeNodes( void )
{
	_sliceStart = NullPointer( Pointer( int ) );
	treeNodes = NullPointer( TreeOctNode* );
	_levels = 0;
}
SortedTreeNodes::~SortedTreeNodes( void )
{
	if( _sliceStart ) for( int d=0 ; d<_levels ; d++ ) FreePointer( _sliceStart[d] );
	FreePointer( _sliceStart );
	DeletePointer( treeNodes );
}
void SortedTreeNodes::set( TreeOctNode& root , std::vector< int >* map )
{
	set( root );

	if( map )
	{
		map->resize( _sliceStart[_levels-1][(size_t)1<<(_levels-1)] );
		for( int i=0 ; i<_sliceStart[_levels-1][(size_t)1<<(_levels-1)] ; i++ ) (*map)[i] = treeNodes[i]->nodeData.nodeIndex;
	}
	for( int i=0 ; i<_sliceStart[_levels-1][(size_t)1<<(_levels-1)] ; i++ ) treeNodes[i]->nodeData.nodeIndex = i;
}
void SortedTreeNodes::set( TreeOctNode& root )
{
	_levels = root.maxDepth()+1;

	if( _sliceStart ) for( int d=0 ; d<_levels ; d++ ) FreePointer( _sliceStart[d] );
	FreePointer( _sliceStart );
	DeletePointer( treeNodes );

	_sliceStart = AllocPointer< Pointer( int ) >( _levels );
	for( int l=0 ; l<_levels ; l++ )
	{
		_sliceStart[l] = AllocPointer< int >( ((size_t)1<<l)+1 );
		memset( _sliceStart[l] , 0 , sizeof(int)*( ((size_t)1<<l)+1 ) );
	}

	// Count the number of nodes in each slice
	for( TreeOctNode* node = root.nextNode() ; node ; node = root.nextNode( node ) )
	{
		int d , off[3];
		node->depthAndOffset( d , off );
		_sliceStart[d][ off[2]+1 ]++;
	}

	// Get the start index for each slice
	{
		int levelOffset = 0;
		for( int l=0 ; l<_levels ; l++ )
		{
			_sliceStart[l][0] = levelOffset;
			for( int s=0 ; s<((size_t)1<<l); s++ ) _sliceStart[l][s+1] += _sliceStart[l][s];
			levelOffset = _sliceStart[l][(size_t)1<<l];
		}
	}
	// Allocate memory for the tree nodes
	treeNodes = NewPointer< TreeOctNode* >( _sliceStart[_levels-1][(size_t)1<<(_levels-1)] );

	// Add the tree nodes
	for( TreeOctNode* node=root.nextNode() ; node ; node=root.nextNode( node ) )
	{
		int d , off[3];
		node->depthAndOffset( d , off );
		treeNodes[ _sliceStart[d][ off[2] ]++ ] = node;
	}

	// Shift the slice offsets up since we incremented as we added
	for( int l=0 ; l<_levels ; l++ )
	{
		for( int s=(1<<l) ; s>0 ; s-- ) _sliceStart[l][s] = _sliceStart[l][s-1];
		_sliceStart[l][0] = l>0 ? _sliceStart[l-1][(size_t)1<<(l-1)] : 0;
	}
}
SortedTreeNodes::SquareCornerIndices& SortedTreeNodes::SliceTableData::cornerIndices( const TreeOctNode* node ) { return cTable[ node->nodeData.nodeIndex - nodeOffset ]; }
SortedTreeNodes::SquareCornerIndices& SortedTreeNodes::SliceTableData::cornerIndices( int idx ) { return cTable[ idx - nodeOffset ]; }
const SortedTreeNodes::SquareCornerIndices& SortedTreeNodes::SliceTableData::cornerIndices( const TreeOctNode* node ) const { return cTable[ node->nodeData.nodeIndex - nodeOffset ]; }
const SortedTreeNodes::SquareCornerIndices& SortedTreeNodes::SliceTableData::cornerIndices( int idx ) const { return cTable[ idx - nodeOffset ]; }
SortedTreeNodes::SquareEdgeIndices& SortedTreeNodes::SliceTableData::edgeIndices( const TreeOctNode* node ) { return eTable[ node->nodeData.nodeIndex - nodeOffset ]; }
SortedTreeNodes::SquareEdgeIndices& SortedTreeNodes::SliceTableData::edgeIndices( int idx ) { return eTable[ idx - nodeOffset ]; }
const SortedTreeNodes::SquareEdgeIndices& SortedTreeNodes::SliceTableData::edgeIndices( const TreeOctNode* node ) const { return eTable[ node->nodeData.nodeIndex - nodeOffset ]; }
const SortedTreeNodes::SquareEdgeIndices& SortedTreeNodes::SliceTableData::edgeIndices( int idx ) const { return eTable[ idx - nodeOffset ]; }
SortedTreeNodes::SquareFaceIndices& SortedTreeNodes::SliceTableData::faceIndices( const TreeOctNode* node ) { return fTable[ node->nodeData.nodeIndex - nodeOffset ]; }
SortedTreeNodes::SquareFaceIndices& SortedTreeNodes::SliceTableData::faceIndices( int idx ) { return fTable[ idx - nodeOffset ]; }
const SortedTreeNodes::SquareFaceIndices& SortedTreeNodes::SliceTableData::faceIndices( const TreeOctNode* node ) const { return fTable[ node->nodeData.nodeIndex - nodeOffset ]; }
const SortedTreeNodes::SquareFaceIndices& SortedTreeNodes::SliceTableData::faceIndices( int idx ) const { return fTable[ idx - nodeOffset ]; }
SortedTreeNodes::SquareCornerIndices& SortedTreeNodes::XSliceTableData::edgeIndices( const TreeOctNode* node ) { return eTable[ node->nodeData.nodeIndex - nodeOffset ]; }
SortedTreeNodes::SquareCornerIndices& SortedTreeNodes::XSliceTableData::edgeIndices( int idx ) { return eTable[ idx - nodeOffset ]; }
const SortedTreeNodes::SquareCornerIndices& SortedTreeNodes::XSliceTableData::edgeIndices( const TreeOctNode* node ) const { return eTable[ node->nodeData.nodeIndex - nodeOffset ]; }
const SortedTreeNodes::SquareCornerIndices& SortedTreeNodes::XSliceTableData::edgeIndices( int idx ) const { return eTable[ idx - nodeOffset ]; }
SortedTreeNodes::SquareEdgeIndices& SortedTreeNodes::XSliceTableData::faceIndices( const TreeOctNode* node ) { return fTable[ node->nodeData.nodeIndex - nodeOffset ]; }
SortedTreeNodes::SquareEdgeIndices& SortedTreeNodes::XSliceTableData::faceIndices( int idx ) { return fTable[ idx - nodeOffset ]; }
const SortedTreeNodes::SquareEdgeIndices& SortedTreeNodes::XSliceTableData::faceIndices( const TreeOctNode* node ) const { return fTable[ node->nodeData.nodeIndex - nodeOffset ]; }
const SortedTreeNodes::SquareEdgeIndices& SortedTreeNodes::XSliceTableData::faceIndices( int idx ) const { return fTable[ idx - nodeOffset ]; }

void SortedTreeNodes::setSliceTableData( SliceTableData& sData , int depth , int offset , int threads ) const
{
	// [NOTE] This is structure is purely for determining adjacency and is independent of the FEM degree
	typedef OctNode< TreeNodeData >::template ConstNeighborKey< 1 , 1 > ConstAdjacenctNodeKey;
	if( offset<0 || offset>((size_t)1<<depth) ) return;
	if( threads<=0 ) threads = 1;
	// The vector of per-depth node spans
	std::pair< int , int > span( _sliceStart[depth][ std::max< int >( 0 , offset-1 ) ] , _sliceStart[depth][ std::min< int >( (size_t)1<<depth , offset+1 ) ] );
	sData.nodeOffset = span.first;
	sData.nodeCount = span.second - span.first;

	DeletePointer( sData._cMap ) ; DeletePointer( sData._eMap ) ; DeletePointer( sData._fMap );
	DeletePointer( sData.cTable ) ; DeletePointer( sData.eTable ) ; DeletePointer( sData.fTable );
	if( sData.nodeCount )
	{
		sData._cMap = NewPointer< int >( sData.nodeCount * Square::CORNERS );
		sData._eMap = NewPointer< int >( sData.nodeCount * Square::EDGES );
		sData._fMap = NewPointer< int >( sData.nodeCount * Square::FACES );
		sData.cTable = NewPointer< typename SortedTreeNodes::SquareCornerIndices >( sData.nodeCount );
		sData.eTable = NewPointer< typename SortedTreeNodes::SquareCornerIndices >( sData.nodeCount );
		sData.fTable = NewPointer< typename SortedTreeNodes::SquareFaceIndices >( sData.nodeCount );
		memset( sData._cMap , 0 , sizeof(int) * sData.nodeCount * Square::CORNERS );
		memset( sData._eMap , 0 , sizeof(int) * sData.nodeCount * Square::EDGES );
		memset( sData._fMap , 0 , sizeof(int) * sData.nodeCount * Square::FACES );
	}
	std::vector< ConstAdjacenctNodeKey > neighborKeys( std::max< int >( 1 , threads ) );
	for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( depth );
#pragma omp parallel for num_threads( threads )
	for( int i=span.first ; i<span.second ; i++ )
	{
		ConstAdjacenctNodeKey& neighborKey = neighborKeys[ omp_get_thread_num() ];
		TreeOctNode* node = treeNodes[i];
		const TreeOctNode::ConstNeighbors< 3 >& neighbors = neighborKey.getNeighbors( node );
		int d , off[3];
		node->depthAndOffset( d , off );
		int z;
		if     ( off[2]==offset-1 ) z = 1;
		else if( off[2]==offset   ) z = 0;
		else fprintf( stderr , "[ERROR] Node out of bounds: %d %d\n" , offset , off[2] ) , exit( 0 );
		// Process the corners
		for( int x=0 ; x<2 ; x++ ) for( int y=0 ; y<2 ; y++ )
		{
			int c = Cube::CornerIndex( x , y , z );
			int fc = Square::CornerIndex( x , y );
			bool cornerOwner = true;
			int ac = Cube::AntipodalCornerIndex(c); // The index of the node relative to the corner
			for( int cc=0 ; cc<Cube::CORNERS ; cc++ ) // Iterate over the corner's cells
			{
				int xx , yy , zz;
				Cube::FactorCornerIndex( cc , xx , yy , zz );
				xx += x , yy += y , zz += z;
				if( neighbors.neighbors[xx][yy][zz] && cc<ac ){ cornerOwner = false ; break; }
			}
			if( cornerOwner )
			{
				int myCount = (i - sData.nodeOffset) * Square::CORNERS + fc;
				sData._cMap[ myCount ] = 1;
				for( int cc=0 ; cc<Cube::CORNERS ; cc++ )
				{
					int xx , yy , zz;
					Cube::FactorCornerIndex( cc , xx , yy , zz );
					int ac = Square::CornerIndex( 1-xx , 1-yy );
					xx += x , yy += y , zz += z;
					if( neighbors.neighbors[xx][yy][zz] ) sData.cornerIndices( neighbors.neighbors[xx][yy][zz] )[ac] = myCount;
				}
			}
		}
		// Process the edges
		for( int o=0 ; o<2 ; o++ ) for( int y=0 ; y<2 ; y++ )
		{
			int fe = Square::EdgeIndex( o , y );
			bool edgeOwner = true;

			int ac = Square::AntipodalCornerIndex( Square::CornerIndex( y , z ) );
			for( int cc=0 ; cc<Square::CORNERS ; cc++ )
			{
				int ii , jj , xx , yy , zz;
				Square::FactorCornerIndex( cc , ii , jj );
				ii += y , jj += z;
				switch( o )
				{
				case 0: yy = ii , zz = jj , xx = 1 ; break;
				case 1: xx = ii , zz = jj , yy = 1 ; break;
				}
				if( neighbors.neighbors[xx][yy][zz] && cc<ac ){ edgeOwner = false ; break; }
			}
			if( edgeOwner )
			{
				int myCount = ( i - sData.nodeOffset ) * Square::EDGES + fe;
				sData._eMap[ myCount ] = 1;
				// Set all edge indices
				for( int cc=0 ; cc<Square::CORNERS ; cc++ )
				{
					int ii , jj , aii , ajj , xx , yy , zz;
					Square::FactorCornerIndex( cc , ii , jj );
					Square::FactorCornerIndex( Square::AntipodalCornerIndex( cc ) , aii , ajj );
					ii += y , jj += z;
					switch( o )
					{
					case 0: yy = ii , zz = jj , xx = 1 ; break;
					case 1: xx = ii , zz = jj , yy = 1 ; break;
					}
					if( neighbors.neighbors[xx][yy][zz] ) sData.edgeIndices( neighbors.neighbors[xx][yy][zz] )[ Square::EdgeIndex( o , aii ) ] = myCount;
				}
			}
		}
		// Process the Faces
		{
			bool faceOwner = !( neighbors.neighbors[1][1][2*z] && !z );
			if( faceOwner )
			{
				int myCount = ( i - sData.nodeOffset ) * Square::FACES;
				sData._fMap[ myCount ] = 1;
				// Set the face indices
				sData.faceIndices( node )[0] = myCount;
				if( neighbors.neighbors[1][1][2*z] ) sData.faceIndices( neighbors.neighbors[1][1][2*z] )[0] = myCount;
			}
		}
	}
	int cCount = 0 , eCount = 0 , fCount = 0;

	for( size_t i=0 ; i<sData.nodeCount * Square::CORNERS ; i++ ) if( sData._cMap[i] ) sData._cMap[i] = cCount++;
	for( size_t i=0 ; i<sData.nodeCount * Square::EDGES   ; i++ ) if( sData._eMap[i] ) sData._eMap[i] = eCount++;
	for( size_t i=0 ; i<sData.nodeCount * Square::FACES   ; i++ ) if( sData._fMap[i] ) sData._fMap[i] = fCount++;
#pragma omp parallel for num_threads( threads )
	for( int i=0 ; i<sData.nodeCount ; i++ )
	{
		for( int j=0 ; j<Square::CORNERS ; j++ ) sData.cTable[i][j] = sData._cMap[ sData.cTable[i][j] ];
		for( int j=0 ; j<Square::EDGES   ; j++ ) sData.eTable[i][j] = sData._eMap[ sData.eTable[i][j] ];
		for( int j=0 ; j<Square::FACES   ; j++ ) sData.fTable[i][j] = sData._fMap[ sData.fTable[i][j] ];
	}

	sData.cCount = cCount , sData.eCount = eCount , sData.fCount = fCount;
}
void SortedTreeNodes::setXSliceTableData( XSliceTableData& sData , int depth , int offset , int threads ) const
{
	typedef OctNode< TreeNodeData >::template ConstNeighborKey< 1 , 1 > ConstAdjacenctNodeKey;
	if( offset<0 || offset>=((size_t)1<<depth) ) return;
	if( threads<=0 ) threads = 1;
	// The vector of per-depth node spans
	std::pair< int , int > span( _sliceStart[depth][offset] , _sliceStart[depth][offset+1] );
	sData.nodeOffset = span.first;
	sData.nodeCount = span.second - span.first;

	DeletePointer( sData._eMap ) ; DeletePointer( sData._fMap );
	DeletePointer( sData.eTable ) ; DeletePointer( sData.fTable );
	if( sData.nodeCount )
	{
		sData._eMap = NewPointer< int >( sData.nodeCount * Square::CORNERS );
		sData._fMap = NewPointer< int >( sData.nodeCount * Square::EDGES );
		sData.eTable = NewPointer< typename SortedTreeNodes::SquareCornerIndices >( sData.nodeCount );
		sData.fTable = NewPointer< typename SortedTreeNodes::SquareEdgeIndices >( sData.nodeCount );
		memset( sData._eMap , 0 , sizeof(int) * sData.nodeCount * Square::CORNERS );
		memset( sData._fMap , 0 , sizeof(int) * sData.nodeCount * Square::EDGES );
	}

	std::vector< ConstAdjacenctNodeKey > neighborKeys( std::max< int >( 1 , threads ) );
	for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( depth );
#pragma omp parallel for num_threads( threads )
	for( int i=span.first ; i<span.second ; i++ )
	{
		ConstAdjacenctNodeKey& neighborKey = neighborKeys[ omp_get_thread_num() ];
		TreeOctNode* node = treeNodes[i];
		const TreeOctNode::ConstNeighbors<3>& neighbors = neighborKey.getNeighbors( node );
		int d , off[3];
		node->depthAndOffset( d , off );
		// Process the edges
		int o=2;
		for( int x=0 ; x<2 ; x++ ) for( int y=0 ; y<2 ; y++ )
		{
			int fc = Square::CornerIndex( x , y );
			bool edgeOwner = true;

			int ac = Square::AntipodalCornerIndex( Square::CornerIndex( x , y ) );
			for( int cc=0 ; cc<Square::CORNERS ; cc++ )
			{
				int ii , jj , xx , yy , zz;
				Square::FactorCornerIndex( cc , ii , jj );
				ii += x , jj += y;
				xx = ii , yy = jj , zz = 1;
				if( neighbors.neighbors[xx][yy][zz] && cc<ac ){ edgeOwner = false ; break; }
			}
			if( edgeOwner )
			{
				int myCount = ( i - sData.nodeOffset ) * Square::CORNERS + fc;
				sData._eMap[ myCount ] = 1;

				// Set all edge indices
				for( int cc=0 ; cc<Square::CORNERS ; cc++ )
				{
					int ii , jj , aii , ajj , xx , yy , zz;
					Square::FactorCornerIndex( cc , ii , jj );
					Square::FactorCornerIndex( Square::AntipodalCornerIndex( cc ) , aii , ajj );
					ii += x , jj += y;
					xx = ii , yy = jj , zz = 1;
					if( neighbors.neighbors[xx][yy][zz] ) sData.edgeIndices( neighbors.neighbors[xx][yy][zz] )[ Square::CornerIndex( aii , ajj ) ] = myCount;
				}
			}
		}
		// Process the faces
		for( int o=0 ; o<2 ; o++ ) for( int y=0 ; y<2 ; y++ )
		{
			bool faceOwner;
			if( o==0 ) faceOwner = !( neighbors.neighbors[1][2*y][1] && !y );
			else       faceOwner = !( neighbors.neighbors[2*y][1][1] && !y );
			if( faceOwner )
			{
				int fe = Square::EdgeIndex( o , y );
				int ae = Square::EdgeIndex( o , 1-y );
				int myCount = ( i - sData.nodeOffset ) * Square::EDGES + fe;
				sData._fMap[ myCount ] = 1;
				// Set the face indices
				sData.faceIndices( node )[fe] = myCount;
				if( o==0 && neighbors.neighbors[1][2*y][1] ) sData.faceIndices( neighbors.neighbors[1][2*y][1] )[ae] = myCount;
				if( o==1 && neighbors.neighbors[2*y][1][1] ) sData.faceIndices( neighbors.neighbors[2*y][1][1] )[ae] = myCount;
			}
		}
	}
	int eCount = 0 , fCount = 0;

	for( size_t i=0 ; i<sData.nodeCount * Square::CORNERS ; i++ ) if( sData._eMap[i] ) sData._eMap[i] = eCount++;
	for( size_t i=0 ; i<sData.nodeCount * Square::EDGES   ; i++ ) if( sData._fMap[i] ) sData._fMap[i] = fCount++;
#pragma omp parallel for num_threads( threads )
	for( int i=0 ; i<sData.nodeCount ; i++ )
	{
		for( int j=0 ; j<Square::CORNERS ; j++ ) sData.eTable[i][j] = sData._eMap[ sData.eTable[i][j] ];
		for( int j=0 ; j<Square::EDGES   ; j++ ) sData.fTable[i][j] = sData._fMap[ sData.fTable[i][j] ];
	}

	sData.eCount = eCount , sData.fCount = fCount;
}
