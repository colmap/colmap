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
// RegularTreeNode //
/////////////////////
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::RegularTreeNode( void )
{
	parent = children = NULL;
	_depth = 0;
	memset( _offset , 0 , sizeof(_offset ) );
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename Initializer >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::RegularTreeNode( Initializer &initializer ) : RegularTreeNode() { initializer( *this ); }

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::cleanChildren( bool deleteChildren )
{
	if( children )
	{
		for( int c=0 ; c<(1<<Dim) ; c++ ) children[c].cleanChildren( deleteChildren );
		if( deleteChildren ) delete[] children;
	}
	children = NULL;
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::~RegularTreeNode(void)
{
#ifdef SHOW_WARNINGS
#pragma message( "[WARNING] Deallocation of children is your responsibility" )
#endif // SHOW_WARNINGS
	parent = children = NULL;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename Initializer >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NewBrood( Allocator< RegularTreeNode >* nodeAllocator , Initializer &initializer )
{
	RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* brood;
	if( nodeAllocator ) brood = PointerAddress( nodeAllocator->newElements( 1<<Dim ) );
	else                brood = new RegularTreeNode[ 1<<Dim ];
	for( int idx=0 ; idx<(1<<Dim) ; idx++ )
	{
		initializer( brood[idx] );
		brood[idx]._depth = 0;
		for( int d=0 ; d<Dim ; d++ ) brood[idx]._offset[d] = (idx>>d) & 1;
	}
	return brood;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ResetDepthAndOffset( RegularTreeNode* root , int depth , int offset[Dim] )
{
	root->_depth = depth;
	for( unsigned int d=0 ; d<Dim ; d++ ) root->_offset[d] = offset[d];
	if( root->children )
	{
		int _offset[Dim];
		for( unsigned int d=0 ; d<Dim ; d++ ) _offset[d] = offset[d]<<1;
		for( unsigned int c=0 ; c<(1<<Dim) ; c++ )
		{
			for( unsigned int d=0 ; d<Dim ; d++ ) _offset[d] = ( offset[d]<<1 ) | ( (c>>d) & 1 );
			ResetDepthAndOffset( root->children + c , depth+1 , _offset );
		}
	}
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename Initializer >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::setFullDepth( int maxDepth , Allocator< RegularTreeNode >* nodeAllocator , Initializer &initializer )
{
	if( maxDepth>0 )
	{
		if( !children ) initChildren< false >( nodeAllocator , initializer );
		for( int i=0 ; i<(1<<Dim) ; i++ ) children[i].setFullDepth( maxDepth-1 , nodeAllocator , initializer );
	}
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename PruneChildrenFunctor >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::pruneChildren( PruneChildrenFunctor pruneFunctor , bool deleteChildren )
{
	if( children )
	{
		if( pruneFunctor( this ) ) cleanChildren( deleteChildren );
		else for( int i=0 ; i<(1<<Dim) ; i++ ) children[i].pruneChildren( pruneFunctor , deleteChildren );
	}
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename Initializer >
bool RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::_initChildren( Allocator< RegularTreeNode >* nodeAllocator , Initializer &initializer )
{
	if( nodeAllocator ) children = PointerAddress( nodeAllocator->newElements( 1<<Dim ) );
	else
	{
		if( children ) delete[] children;
		children = new RegularTreeNode[ 1<<Dim ];
	}
	if( !children ) MK_THROW( "Failed to initialize children" );
	for( int idx=0 ; idx<(1<<Dim) ; idx++ )
	{
		children[idx].parent = this;
		children[idx].children = NULL;
		initializer( children[idx] );
		children[idx]._depth = _depth+1;
		for( int d=0 ; d<Dim ; d++ ) children[idx]._offset[d] = (_offset[d]<<1) | ( (idx>>d) & 1 );
	}
	return true;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename Initializer >
bool RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::_initChildren_s( Allocator< RegularTreeNode >* nodeAllocator , Initializer &initializer )
{
	RegularTreeNode * volatile & children = this->children;
	RegularTreeNode *_children;

	// Allocate the children
	if( nodeAllocator ) _children = PointerAddress( nodeAllocator->newElements( 1<<Dim ) );
	else                _children = new RegularTreeNode[ 1<<Dim ];
	if( !_children ) MK_THROW( "Failed to initialize children" );
	for( int idx=0 ; idx<(1<<Dim) ; idx++ )
	{
		_children[idx].parent = this;
		_children[idx].children = NULL;
		_children[idx]._depth = _depth+1;
		for( int d=0 ; d<Dim ; d++ ) _children[idx]._offset[d] = (_offset[d]<<1) | ( (idx>>d) & 1 );
		// [WARNING] We are assuming that it's OK to initialize nodes that may not be used.
		for( int idx=0 ; idx<(1<<Dim) ; idx++ ) initializer( _children[idx] );
	}

	// If we are the first to set the child, initialize
	if( SetAtomic( children , _children , (RegularTreeNode *)NULL ) ) return true;
	// Otherwise clean up
	else
	{
		if( nodeAllocator ) ;
		else delete[] _children;
		return false;
	}
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< class MergeFunctor >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::merge( RegularTreeNode* node , MergeFunctor& f )
{
	if( node )
	{
		nodeData = f( nodeData , node->nodeData );
		if( children && node->children ) for( int c=0 ; c<(1<<Dim) ; c++ ) children[c].merge( node->children[c] , f );
		else if( node->children )
		{
			children = node->children;
			for( int c=0 ; c<(1<<Dim) ; c++ ) children[c].parent = this;
			node->children = NULL;
		}
	}
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
inline typename RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::DepthAndOffset RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::depthAndOffset( void ) const
{
	DepthAndOffset doff;
	doff.depth = _depth;
	for( int d=0 ; d<Dim ; d++ ) doff.offset[d] = _offset[d];
	return doff;
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
inline void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::depthAndOffset( int& depth , int offset[Dim] ) const
{
	depth = _depth;
	for( int d=0 ; d<Dim ; d++ ) offset[d] = _offset[d];
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
inline void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::centerIndex( int index[Dim] ) const
{
	for( int i=0 ; i<Dim ; i++ ) index[i] = BinaryNode::CenterIndex( _depth , _offset[i] );
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
inline int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::depth( void ) const { return _depth; }

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< class Real >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::centerAndWidth( Point< Real , Dim >& center , Real& width ) const
{
	width = Real( 1.0 / (1<<_depth) );
	for( int d=0 ; d<Dim ; d++ ) center[d] = Real( 0.5+_offset[d] ) * width;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< class Real >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::startAndWidth( Point< Real , Dim >& start , Real& width ) const
{
	width = Real( 1.0 / (1<<_depth) );
	for( int d=0 ; d<Dim ; d++ ) start[d] = Real( _offset[d] ) * width;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< class Real >
bool RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::isInside( Point< Real , Dim > p ) const
{
	Point< Real , Dim > c ; Real w;
	centerAndWidth( c , w ) , w /= 2;
	for( int d=0 ; d<Dim ; d++ ) if( p[d]<=(c[d]-w) || p[d]>(c[d]+w) ) return false;
	return true;
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::maxDepth(void) const
{
	if( !children ) return 0;
	else
	{
		int c , d;
		for( int i=0 ; i<(1<<Dim) ; i++ )
		{
			d = children[i].maxDepth();
			if( !i || d>c ) c=d;
		}
		return c+1;
	}
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
size_t RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::nodes( void ) const
{
	if( !children ) return 1;
	else
	{
		size_t c=0;
		for( int i=0 ; i<(1<<Dim) ; i++ ) c += children[i].nodes();
		return c+1;
	}
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
size_t RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::leaves( void ) const
{
	if( !children ) return 1;
	else
	{
		size_t c=0;
		for( int i=0 ; i<(1<<Dim) ; i++ ) c += children[i].leaves();
		return c;
	}
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
size_t RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::maxDepthLeaves( int maxDepth ) const
{
	if( depth()>maxDepth ) return 0;
	if( !children ) return 1;
	else
	{
		size_t c=0;
		for( int i=0 ; i<(1<<Dim) ; i++ ) c += children[i].maxDepthLeaves(maxDepth);
		return c;
	}
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
const RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::root( void ) const
{
	const RegularTreeNode* temp = this;
	while( temp->parent ) temp = temp->parent;
	return temp;
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename NodeFunctor /* = std::function< bool/void ( RegularTreeNode * ) > */ >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::processNodes( NodeFunctor nodeFunctor )
{
	if constexpr( std::is_same< bool , typename std::invoke_result< NodeFunctor , RegularTreeNode * >::type >::value )
	{
		if( nodeFunctor( this ) && children )
			for( int c=0 ; c<(1<<Dim) ; c++ )
				if( nodeFunctor( children + c ) && children[c].children )
					children[c]._processChildNodes( nodeFunctor );
	}
	else
	{
		nodeFunctor( this );
		if( children ) for( int c=0 ; c<(1<<Dim) ; c++ )
		{
			nodeFunctor( children + c );
			if( children[c].children ) children[c]._processChildNodes( nodeFunctor );
		}
	}
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename NodeFunctor /* = std::function< bool/void ( const RegularTreeNode * ) > */ >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::processNodes( NodeFunctor nodeFunctor ) const
{
	if constexpr( std::is_same< bool , typename std::invoke_result< NodeFunctor , RegularTreeNode * >::type >::value )
	{
		if( nodeFunctor( this ) && children )
			for( int c=0 ; c<(1<<Dim) ; c++ )
				if( nodeFunctor( children + c ) && children[c].children )
					children[c]._processChildNodes( nodeFunctor );
	}
	else
	{
		nodeFunctor( this );
		if( children ) for( int c=0 ; c<(1<<Dim) ; c++ )
		{
			nodeFunctor( children + c );
			if( children[c].children ) children[c]._processChildNodes( nodeFunctor );
		}
	}
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename NodeFunctor /* = std::function< void ( RegularTreeNode * ) > */ >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::processLeaves( NodeFunctor nodeFunctor )
{
	if( children )
	{
		for( int c=0 ; c<(1<<Dim) ; c++ )
			if( children[c].children ) children[c]._processChildLeaves( nodeFunctor );
			else nodeFunctor( children+c );
	}
	else nodeFunctor( this );
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename NodeFunctor /* = std::function< void ( const RegularTreeNode * ) > */ >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::processLeaves( NodeFunctor nodeFunctor ) const
{
	if( children )
	{
		for( int c=0 ; c<(1<<Dim) ; c++ )
			if( children[c].children ) children[c]._processChildLeaves( nodeFunctor );
			else nodeFunctor( children+c );
	}
	else nodeFunctor( this );
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename NodeFunctor /* = std::function< bool/void ( RegularTreeNode * ) > */ >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::_processChildNodes( NodeFunctor &nodeFunctor )
{
	if constexpr( std::is_same< bool , typename std::invoke_result< NodeFunctor , RegularTreeNode * >::type >::value )
	{
		for( int c=0 ; c<(1<<Dim) ; c++ )
			if( nodeFunctor( children + c ) && children[c].children )
				children[c]._processChildNodes( nodeFunctor );
	}
	else
	{
		for( int c=0 ; c<(1<<Dim) ; c++ )
		{
			nodeFunctor( children + c );
			if( children[c].children ) children[c]._processChildNodes( nodeFunctor );
		}
	}
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename NodeFunctor /* = std::function< bool/void ( const RegularTreeNode * ) > */ >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::_processChildNodes( NodeFunctor &nodeFunctor ) const
{
	if constexpr( std::is_same< bool , typename std::invoke_result< NodeFunctor , RegularTreeNode * >::type >::value )
	{
		for( int c=0 ; c<(1<<Dim) ; c++ )
			if( nodeFunctor( children + c ) && children[c].children )
				children[c]._processChildNodes( nodeFunctor );
	}
	else
	{
		for( int c=0 ; c<(1<<Dim) ; c++ )
		{
			nodeFunctor( children + c );
			if( children[c].children ) children[c]._processChildNodes( nodeFunctor );
		}
	}
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename NodeFunctor /* = std::function< void ( RegularTreeNode * ) > */ >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::_processChildLeaves( NodeFunctor &nodeFunctor )
{
	for( int c=0 ; c<(1<<Dim) ; c++ )
		if( children[c].children ) children[c]._processChildLeaves( nodeFunctor );
		else nodeFunctor( children+c );
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename NodeFunctor /* = std::function< void ( const RegularTreeNode * ) > */ >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::_processChildLeaves( NodeFunctor &nodeFunctor ) const
{
	for( int c=0 ; c<(1<<Dim) ; c++ )
		if( children[c].children ) children[c]._processChildLeaves( nodeFunctor );
		else nodeFunctor( children+c );
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::printRange(void) const
{
	Point< float , Dim > center;
	float width;
	centerAndWidth( center , width );
	for( int d=0 ; d<Dim ; d++ )
	{
		printf( "[%f,%f]" , center[d]-width/2 , center[d]+width/2 );
		if( d<Dim-1 ) printf( " x " );
		else printf("\n");
	}
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< class Real >
int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ChildIndex( const Point< Real , Dim >& center , const Point< Real , Dim >& p )
{
	int cIndex=0;
	for( int d=0 ; d<Dim ; d++ ) if( p[d]>center[d] ) cIndex |= (1<<d);
	return cIndex;
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename KeepNodeFunctor >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::copySubTree( RegularTreeNode< Dim , NodeData , DepthAndOffsetType > &subTree , const KeepNodeFunctor &keepNodeFunctor , Allocator< RegularTreeNode > *nodeAllocator ) const
{
	bool copyChildren = false;
	if( children ) for( unsigned int c=0 ; c<(1<<Dim) ; c++ ) copyChildren |= keepNodeFunctor( children+c );
	if( copyChildren )
	{
		if( nodeAllocator ) subTree.children = PointerAddress( nodeAllocator->newElements( 1<<Dim ) );
		else                subTree.children = new RegularTreeNode[ 1<<Dim ];
		for( unsigned int c=0 ; c<(1<<Dim) ; c++ )
		{
			subTree.children[c] = children[c];
			subTree.children[c].children = NULL;
			subTree.children[c].parent = &subTree;
			children[c].copySubTree( subTree.children[c] , keepNodeFunctor );
		}
	}
}

// KeepNodeFunctor looks like std::function< bool ( const RegularTreeNode * ) >
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename KeepNodeFunctor >
Pointer( RegularTreeNode< Dim , NodeData , DepthAndOffsetType > ) RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::serializeSubTree( const KeepNodeFunctor &keepNodeFunctor , size_t &nodeCount ) const
{
	std::function< size_t ( const RegularTreeNode * ) > ChildNodeCount = [&]( const RegularTreeNode *node )
	{
		size_t count = 0;
		bool keepChildren = false;
		if( node->children ) for( unsigned int c=0 ; c<(1<<Dim) ; c++ ) keepChildren |= keepNodeFunctor( node->children+c );
		if( keepChildren )
		{
			count += 1<<Dim;
			for( unsigned int c=0 ; c<(1<<Dim) ; c++ ) count += ChildNodeCount( node->children+c );
		}
		return count;
	};

	nodeCount = 1 + ChildNodeCount( this );
	Pointer( RegularTreeNode ) nodes = NewPointer< RegularTreeNode >( nodeCount );

	std::function< Pointer( RegularTreeNode ) ( const RegularTreeNode * , RegularTreeNode & , Pointer( RegularTreeNode ) ) > SetChildNodes = [&]( const RegularTreeNode *node , RegularTreeNode &subNode , Pointer( RegularTreeNode ) buffer )
	{
		bool keepChildren = false;
		if( node->children ) for( unsigned int c=0 ; c<(1<<Dim) ; c++ ) keepChildren |= keepNodeFunctor( node->children+c );
		if( keepChildren )
		{
			subNode.children = PointerAddress( buffer );
			for( unsigned int c=0 ; c<(1<<Dim) ; c++ )
			{
				buffer[c] = node->children[c];
				buffer[c].parent = &subNode;
				buffer[c].children = NULL;
			}
			buffer += 1<<Dim;
			for( unsigned int c=0 ; c<(1<<Dim) ; c++ ) buffer = SetChildNodes( node->children+c , subNode.children[c] , buffer );
		}
		return buffer;
	};

	nodes[0] = *this;
	nodes[0].parent = NULL;
	nodes[0].children = NULL;

	SetChildNodes( this , nodes[0] , nodes+1 );

	return nodes;
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename WriteNodeFunctor >
bool RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::write( BinaryStream &stream , bool serialize , const WriteNodeFunctor &writeNodeFunctor ) const
{
	if( serialize )
	{
		size_t nodeCount;
		Pointer( RegularTreeNode ) nodes = serializeSubTree( writeNodeFunctor , nodeCount );
		stream.write( nodes , nodeCount );
		DeletePointer( nodes );
	}
	else
	{
		std::function< void ( const RegularTreeNode *node ) > WriteChildren = [&]( const RegularTreeNode *node )
		{
			bool writeChildren = false;
			if( node->children ) for( unsigned int c=0 ; c<(1<<Dim) ; c++ ) writeChildren |= writeNodeFunctor( node->children+c );
			if( writeChildren )
			{
				stream.write( GetPointer( node->children , 1<<Dim ) , 1<<Dim );
				for( unsigned int c=0 ; c<(1<<Dim) ; c++ ) WriteChildren( node->children+c );
			}
		};
		stream.write( *this );
		WriteChildren( this );
	}
	return true;
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< typename Initializer >
bool RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::read( BinaryStream &stream , Allocator< RegularTreeNode >* nodeAllocator , Initializer &initializer )
{
	std::function< void ( RegularTreeNode *node ) > ReadChildren = [&]( RegularTreeNode *node )
	{
		if( node->children )
		{
			node->children = NULL;
			node->initChildren< false >( nodeAllocator , initializer );
			if( !stream.read( GetPointer( node->children , 1<<Dim ) , 1<<Dim ) ) MK_THROW( "Failed to read children" );
			for( int i=0 ; i<(1<<Dim) ; i++ )
			{
				node->children[i].parent = node;
				ReadChildren( node->children+i );
			}
		}
	};
	if( !stream.read( *this ) ) MK_THROW( "Failed to read root" );
	ReadChildren( this );
	return true;
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::width( int maxDepth ) const
{
	int d=depth();
	return 1<<(maxDepth-d); 
}

////////////////////////////////
// RegularTreeNode::Neighbors //
////////////////////////////////
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... Widths >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::Neighbors< ParameterPack::UIntPack< Widths ... > >::Neighbors( void ){ static_assert( sizeof...(Widths)==Dim , "[ERROR] Window and tree dimensions don't match" ) ; clear(); }
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... Widths >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::Neighbors< ParameterPack::UIntPack< Widths ... > >::clear( void ){ for( unsigned int i=0 ; i<Window::Size< Widths... >() ; i++ ) neighbors.data[i] = NULL; }

/////////////////////////////////////
// RegularTreeNode::ConstNeighbors //
/////////////////////////////////////
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... Widths >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighbors< ParameterPack::UIntPack< Widths ... > >::ConstNeighbors( void ){ static_assert( sizeof...(Widths)==Dim , "[ERROR] Window and tree dimensions don't match" ) ; clear(); }
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... Widths >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighbors< ParameterPack::UIntPack< Widths ... > >::clear( void ){ for( unsigned int i=0 ; i<Window::Size< Widths... >() ; i++ ) neighbors.data[i] = NULL; }

//////////////////////////////////
// RegularTreeNode::NeighborKey //
//////////////////////////////////
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::NeighborKey( void ){ _depth=-1 , neighbors=NULL; }
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::NeighborKey( const NeighborKey& key )
{
	_depth = 0 , neighbors = NULL;
	set( key._depth );
	for( int d=0 ; d<=_depth ; d++ ) memcpy( &neighbors[d] , &key.neighbors[d] , sizeof( Neighbors< ParameterPack::UIntPack< ( LeftRadii + RightRadii + 1 ) ... > > ) );
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::~NeighborKey( void )
{
	if( neighbors ) delete[] neighbors;
	neighbors=NULL;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::set( int d )
{
	if( neighbors ) delete[] neighbors;
	neighbors = NULL;
	_depth = d;
	if( d<0 ) return;
	neighbors = new NeighborType[d+1];
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< bool CreateNodes , bool ThreadSafe , typename NodeInitializer , unsigned int ... _PLeftRadii , unsigned int ... _PRightRadii , unsigned int ... _CLeftRadii , unsigned int ... _CRightRadii >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::_NeighborsLoop( ParameterPack::UIntPack< _PLeftRadii ... > pLeftRadii , ParameterPack::UIntPack< _PRightRadii ... > pRightRadii , ParameterPack::UIntPack< _CLeftRadii ... > cLeftRadii , ParameterPack::UIntPack< _CRightRadii ... > cRightRadii , Window::ConstSlice< RegularTreeNode* , ( _PLeftRadii + _PRightRadii + 1 ) ... > pNeighbors , Window::Slice< RegularTreeNode* , ( _CLeftRadii + _CRightRadii + 1 ) ... > cNeighbors , int cIdx , Allocator< RegularTreeNode >* nodeAllocator , NodeInitializer &initializer )
{
	static_assert( Dim==sizeof ... ( _PLeftRadii ) && Dim==sizeof ... ( _PRightRadii ) && Dim==sizeof ... ( _CLeftRadii ) && Dim==sizeof ... ( _CRightRadii ) , "[ERROR] Dimensions don't match" );
	int c[Dim];
	for( int d=0 ; d<Dim ; d++ ) c[d] = ( cIdx>>d ) & 1;
	return _Run< CreateNodes , ThreadSafe , NodeInitializer , ParameterPack::UIntPack< _PLeftRadii ... > , ParameterPack::UIntPack< _PRightRadii ... > , ParameterPack::UIntPack< _CLeftRadii ... > , ParameterPack::UIntPack< _CRightRadii ... > >::Run( pNeighbors , cNeighbors , c , 0 , nodeAllocator , initializer );
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< bool CreateNodes , bool ThreadSafe , typename NodeInitializer , unsigned int ... _PLeftRadii , unsigned int ... _PRightRadii , unsigned int ... _CLeftRadii , unsigned int ... _CRightRadii >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::_NeighborsLoop( ParameterPack::UIntPack< _PLeftRadii ... > pLeftRadii , ParameterPack::UIntPack< _PRightRadii ... > pRightRadii , ParameterPack::UIntPack< _CLeftRadii ... > cLeftRadii , ParameterPack::UIntPack< _CRightRadii ... > cRightRadii , Window::Slice< RegularTreeNode* , ( _PLeftRadii + _PRightRadii + 1 ) ... > pNeighbors , Window::Slice< RegularTreeNode* , ( _CLeftRadii + _CRightRadii + 1 ) ... > cNeighbors , int cIdx , Allocator< RegularTreeNode >* nodeAllocator , NodeInitializer &initializer )
{
	return _NeighborsLoop< CreateNodes , ThreadSafe , NodeInitializer >( ParameterPack::UIntPack< _PLeftRadii ... >() , ParameterPack::UIntPack< _PRightRadii ... >() , ParameterPack::UIntPack< _CLeftRadii ... >() , ParameterPack::UIntPack< _CRightRadii ... >() , ( Window::ConstSlice< RegularTreeNode* , ( _PLeftRadii + _PRightRadii + 1 ) ... > )pNeighbors , cNeighbors , cIdx , nodeAllocator , initializer );
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< bool CreateNodes , bool ThreadSafe , typename NodeInitializer , unsigned int _PLeftRadius , unsigned int ... _PLeftRadii , unsigned int _PRightRadius , unsigned int ... _PRightRadii , unsigned int _CLeftRadius , unsigned int ... _CLeftRadii , unsigned int _CRightRadius , unsigned int ... _CRightRadii >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::_Run< CreateNodes , ThreadSafe , NodeInitializer , ParameterPack::UIntPack< _PLeftRadius , _PLeftRadii ... > , ParameterPack::UIntPack< _PRightRadius , _PRightRadii ... > , ParameterPack::UIntPack< _CLeftRadius , _CLeftRadii ... > , ParameterPack::UIntPack< _CRightRadius , _CRightRadii ... > >::Run( Window::ConstSlice< RegularTreeNode* , _PLeftRadius + _PRightRadius + 1 , ( _PLeftRadii + _PRightRadii + 1 ) ... > pNeighbors , Window::Slice< RegularTreeNode* , _CLeftRadius + _CRightRadius + 1 , ( _CLeftRadii + _CRightRadii + 1 ) ... > cNeighbors , int* c , int cornerIndex , Allocator< RegularTreeNode >* nodeAllocator , NodeInitializer &initializer )
{
	static const int D = sizeof ... ( _PLeftRadii ) + 1;
	unsigned int count=0;
	for( int i=-(int)_CLeftRadius ; i<=(int)_CRightRadius ; i++ )
	{
		int _i = (i+c[Dim-D]) + ( _CLeftRadius<<1 ) , pi = ( _i>>1 ) - _CLeftRadius + _PLeftRadius  , ci = i + _CLeftRadius;
		count += _Run< CreateNodes , ThreadSafe , NodeInitializer , ParameterPack::UIntPack< _PLeftRadii ... > , ParameterPack::UIntPack< _PRightRadii ... > , ParameterPack::UIntPack< _CLeftRadii ... > , ParameterPack::UIntPack< _CRightRadii ... > >::Run( pNeighbors[pi] , cNeighbors[ci] , c , cornerIndex | ( ( _i&1)<<(Dim-D) ) , nodeAllocator , initializer );
	}
	return count;
}


template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< bool CreateNodes , bool ThreadSafe , typename NodeInitializer , unsigned int _PLeftRadius , unsigned int _PRightRadius , unsigned int _CLeftRadius , unsigned int _CRightRadius >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::_Run< CreateNodes , ThreadSafe , NodeInitializer , ParameterPack::UIntPack< _PLeftRadius > , ParameterPack::UIntPack< _PRightRadius > , ParameterPack::UIntPack< _CLeftRadius > , ParameterPack::UIntPack< _CRightRadius > >::Run( Window::ConstSlice< RegularTreeNode* , _PLeftRadius+_PRightRadius+1 > pNeighbors , Window::Slice< RegularTreeNode* , _CLeftRadius+_CRightRadius+1 > cNeighbors , int* c , int cornerIndex , Allocator< RegularTreeNode >* nodeAllocator , NodeInitializer &initializer )
{
	static const int D = 1;
	unsigned int count=0;
	for( int i=-(int)_CLeftRadius ; i<=(int)_CRightRadius ; i++ )
	{
		int _i = (i+c[Dim-1]) + ( _CLeftRadius<<1 ) , pi = ( _i>>1 ) - _CLeftRadius + _PLeftRadius  , ci = i + _CLeftRadius;
		if( CreateNodes )
		{
			if( pNeighbors[pi] )
			{
#ifdef SANITIZED_PR
//#ifdef NEW_CODE
				RegularTreeNode * children = ReadAtomic( pNeighbors[pi]->children );
				if( !children )
				{
					pNeighbors[pi]->template initChildren< ThreadSafe >( nodeAllocator , initializer );
					children = ReadAtomic( pNeighbors[pi]->children );
				}
				cNeighbors[ci] = children + ( cornerIndex | ( ( _i&1)<<(Dim-1) ) );
#else // !SANITIZED_PR
				if( !pNeighbors[pi]->children ) pNeighbors[pi]->template initChildren< ThreadSafe >( nodeAllocator , initializer );
				cNeighbors[ci] = pNeighbors[pi]->children + ( cornerIndex | ( ( _i&1)<<(Dim-1) ) );
#endif // SANITIZED_PR
				count++;
			}
			else cNeighbors[ci] = NULL;
		}
		else
		{
			if( pNeighbors[pi] && pNeighbors[pi]->children ) cNeighbors[ci] = pNeighbors[pi]->children + ( cornerIndex | ( ( _i&1)<<(Dim-1) ) ) , count++;
			else cNeighbors[ci] = NULL;
		}
	}
	return count;
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< bool CreateNodes , bool ThreadSafe , typename NodeInitializer >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::getChildNeighbors( int cIdx , int d , NeighborType& cNeighbors , Allocator< RegularTreeNode >* nodeAllocator , NodeInitializer &initializer ) const
{
	NeighborType& pNeighbors = neighbors[d];
	// Check that we actually have a center node
	if( !pNeighbors.neighbors.data[ CenterIndex ] ) return 0;
	return _NeighborsLoop< CreateNodes , ThreadSafe >( ParameterPack::UIntPack< LeftRadii ... >() , ParameterPack::UIntPack< RightRadii ... >() , ParameterPack::UIntPack< LeftRadii ... >() , ParameterPack::UIntPack< RightRadii ... >() , pNeighbors.neighbors() , cNeighbors.neighbors() , cIdx , nodeAllocator , initializer );
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< bool CreateNodes , bool ThreadSafe , typename NodeInitializer , class Real >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::getChildNeighbors( Point< Real , Dim > p , int d , NeighborType& cNeighbors , Allocator< RegularTreeNode >* nodeAllocator , NodeInitializer &initializer ) const
{
	NeighborType& pNeighbors = neighbors[d];
	// Check that we actually have a center node
	if( !pNeighbors.neighbors.data[ CenterIndex ] ) return 0;
	Point< Real , Dim > c;
	Real w;
	pNeighbors.neighbors.data[ CenterIndex ]->centerAndWidth( c , w );
	return getChildNeighbors< CreateNodes , ThreadSafe >( CornerIndex( c , p ) , d , cNeighbors , nodeAllocator , initializer );
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< bool CreateNodes , bool ThreadSafe , typename NodeInitializer >
typename RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::template Neighbors< ParameterPack::UIntPack< ( LeftRadii + RightRadii + 1 ) ... > >& RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::getNeighbors( RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* node , Allocator< RegularTreeNode >* nodeAllocator , NodeInitializer &initializer )
{
	NeighborType& neighbors = this->neighbors[node->depth()];
	// This is required in case the neighbors have been constructed between the last call to getNeighbors and this one
	if( node==neighbors.neighbors.data[ CenterIndex ] )
	{
		bool reset = false;
		for( unsigned int i=0 ; i<Window::Size< ( LeftRadii+RightRadii+1 ) ... >() ; i++ ) if( !neighbors.neighbors.data[i] ) reset = true;
		if( reset ) neighbors.neighbors.data[ CenterIndex ] = NULL;
	}
	if( node!=neighbors.neighbors.data[ CenterIndex ] )
	{
		for( int d=node->depth()+1 ; d<=_depth && this->neighbors[d].neighbors.data[ CenterIndex ] ; d++ ) this->neighbors[d].neighbors.data[ CenterIndex ] = NULL;
		neighbors.clear();
		if( !node->parent ) neighbors.neighbors.data[ CenterIndex ] = node;
		else _NeighborsLoop< CreateNodes , ThreadSafe >( ParameterPack::UIntPack< LeftRadii ... >() , ParameterPack::UIntPack< RightRadii ... >() , ParameterPack::UIntPack< LeftRadii ... >() , ParameterPack::UIntPack< RightRadii ... >() , getNeighbors< CreateNodes , ThreadSafe >( node->parent , nodeAllocator , initializer ).neighbors() , neighbors.neighbors() , (int)( node - node->parent->children ) , nodeAllocator , initializer );
	}
	return neighbors;
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< bool CreateNodes , bool ThreadSafe , typename NodeInitializer , unsigned int ... _LeftRadii , unsigned int ... _RightRadii >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::getNeighbors( ParameterPack::UIntPack< _LeftRadii ... > , ParameterPack::UIntPack< _RightRadii ... > , RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* node , Neighbors< ParameterPack::UIntPack< ( _LeftRadii + _RightRadii + 1 ) ... > >& neighbors , Allocator< RegularTreeNode >* nodeAllocator , NodeInitializer &initializer )
{
	static const unsigned int _CenterIndex = Window::Index( ParameterPack::UIntPack< ( _LeftRadii + _RightRadii + 1 ) ... >() , ParameterPack::UIntPack< _LeftRadii ... >() );
	neighbors.clear();
	if( !node ) return;

	// [WARNING] This estimate of the required radius is somewhat conservative if the readius is odd (depending on where the node is relative to its parent)
	ParameterPack::UIntPack<  LeftRadii ... >  leftRadii;
	ParameterPack::UIntPack< RightRadii ... > rightRadii;
	ParameterPack::UIntPack< (  _LeftRadii+1 )/2 ... >  pLeftRadii;
	ParameterPack::UIntPack< ( _RightRadii+1 )/2 ... > pRightRadii;
	ParameterPack::UIntPack<  _LeftRadii ... >  cLeftRadii;
	ParameterPack::UIntPack< _RightRadii ... > cRightRadii;

	// If we are at the root of the tree, we are done
	if( !node->parent ) neighbors.neighbors.data[ _CenterIndex ] = node;
	// If we can get the data from the the key for the parent node, do that
	else if( pLeftRadii<=leftRadii && pRightRadii<=rightRadii )
	{
		getNeighbors< CreateNodes , ThreadSafe >( node->parent , nodeAllocator , initializer );
		const Neighbors< ParameterPack::UIntPack< ( LeftRadii + RightRadii + 1 ) ... > >& pNeighbors = this->neighbors[ node->depth()-1 ];
		_NeighborsLoop< CreateNodes , ThreadSafe >( leftRadii , rightRadii , cLeftRadii , cRightRadii , pNeighbors.neighbors() , neighbors.neighbors() , (int)( node - node->parent->children ) , nodeAllocator , initializer );
	}
	// Otherwise recurse
	else
	{
		Neighbors< ParameterPack::UIntPack< ( ( _LeftRadii+1 )/2  + ( _RightRadii+1 )/2 + 1 ) ... > > pNeighbors;
		getNeighbors< CreateNodes , ThreadSafe >( pLeftRadii , pRightRadii , node->parent , pNeighbors , nodeAllocator , initializer );
		_NeighborsLoop< CreateNodes , ThreadSafe >( pLeftRadii , pRightRadii , cLeftRadii , cRightRadii , pNeighbors.neighbors() , neighbors.neighbors() , (int)( node - node->parent->children ) , nodeAllocator , initializer );
	}
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< bool CreateNodes , bool ThreadSafe , typename NodeInitializer , unsigned int ... _LeftRadii , unsigned int ... _RightRadii >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::getNeighbors( ParameterPack::UIntPack< _LeftRadii ... > , ParameterPack::UIntPack< _RightRadii ... > , RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* node , Neighbors< ParameterPack::UIntPack< ( _LeftRadii + _RightRadii + 1 ) ... > >& pNeighbors , Neighbors< ParameterPack::UIntPack< ( _LeftRadii + _RightRadii + 1 ) ... > >& neighbors , Allocator< RegularTreeNode >* nodeAllocator , NodeInitializer &initializer )
{
	ParameterPack::UIntPack<  _LeftRadii ... >  leftRadii;
	ParameterPack::UIntPack< _RightRadii ... > rightRadii;
	if( !node->parent ) getNeighbors< CreateNodes , ThreadSafe >( leftRadii , rightRadii , node , neighbors , nodeAllocator , initializer );
	else
	{
		getNeighbors< CreateNodes , ThreadSafe >( leftRadii , rightRadii , node->parent , pNeighbors , nodeAllocator , initializer );
		_NeighborsLoop< CreateNodes , ThreadSafe >( leftRadii , rightRadii , leftRadii , rightRadii , pNeighbors.neighbors() , neighbors.neighbors() , (int)( node - node->parent->children ) , nodeAllocator , initializer );
	}
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::NeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::setLeafNeighbors( RegularTreeNode *node , Window::StaticWindow< RegularTreeNode * , ( LeftRadii + RightRadii + 1 ) ... > &leaves )
{
	// Suppose that we have a node at index I and we want the leaf nodes supported on the (possibly virtual) node K away
	// Case 1: The K-th neighbor exists
	// ---> Iterate over the leaf nodes of the sub-tree rooted at the K-th neighbor
	// Case 2: The K-th neighbor does not exist
	// ---> The index of the K-th neighbor is I+K
	// ---> The index of the parent is floor( I/2 )
	// ---> The index of the K-th neighbors parent is floor( (I+K)/2 )
	// ---> The parent of the K-th neighbor is the [ floor( (I+k)/2 ) - floor( I/2 ) ]-th neighbor of the parent

	static const unsigned int _LeftRadii[] = { LeftRadii ... };
	auto GetNeighborLeaf = [&]( unsigned int depth , const int index[Dim] , const int offset[Dim] )
	{
//		unsigned int _index[Dim] , _offset[Dim] , __offset[Dim];
		int _index[Dim] , _offset[Dim] , __offset[Dim];
		for( int dim=0 ; dim<Dim ; dim++ ) _index[dim] = index[dim] , _offset[dim] = offset[dim];
		for( int d=depth ; d>=0 ; d-- )
		{
			for( int dim=0 ; dim<Dim ; dim++ ) __offset[dim] = _offset[dim] + _LeftRadii[dim];
			if( neighbors[d].neighbors( __offset ) ) return neighbors[d].neighbors( __offset );
			else
			{
				for( int dim=0 ; dim<Dim ; dim++ )
				{
					_offset[dim] = ( ( _index[dim] + _offset[dim] ) >> 1 ) - ( _index[dim] >> 1 );
					_index[dim] >>= 1;
				}
			}
		}
		return ( RegularTreeNode * )NULL;
	};
	getNeighbors( node );
	int depth , index[Dim] , offset[Dim] , _offset[Dim];
	node->depthAndOffset( depth , index );

	Window::Loop< Dim >::Run
	(
		ParameterPack::IsotropicUIntPack< Dim , 0 >() , ParameterPack::UIntPack< ( LeftRadii + RightRadii + 1 ) ... >() ,
		[&]( int d , int i ){ offset[d] = i , _offset[d] = i-_LeftRadii[d]; } ,
		[&]( void ){ leaves( offset ) = GetNeighborLeaf( depth , index , _offset ); }
	);
}

///////////////////////////////////////
// RegularTreeNode::ConstNeighborKey //
///////////////////////////////////////
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::ConstNeighborKey( void ){ _depth=-1 , neighbors=NULL; }
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::ConstNeighborKey( const ConstNeighborKey& key )
{
	_depth = 0 , neighbors = NULL;
	set( key._depth );
	for( int d=0 ; d<=_depth ; d++ ) memcpy( &neighbors[d] , &key.neighbors[d] , sizeof( ConstNeighbors< ParameterPack::UIntPack< ( LeftRadii + RightRadii + 1 ) ... > > ) );
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::~ConstNeighborKey( void )
{
	if( neighbors ) delete[] neighbors;
	neighbors=NULL;
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
typename RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::template ConstNeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >& RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::operator = ( const ConstNeighborKey& key )
{
	set( key._depth );
	for( int d=0 ; d<=_depth ; d++ ) memcpy( &neighbors[d] , &key.neighbors[d] , sizeof( ConstNeighbors< ParameterPack::UIntPack< ( LeftRadii + RightRadii + 1 ) ... > > ) );
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::set( int d )
{
	if( neighbors ) delete[] neighbors;
	neighbors = NULL;
	_depth = d;
	if( d<0 ) return;
	neighbors = new NeighborType[d+1];
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< unsigned int ... _PLeftRadii , unsigned int ... _PRightRadii , unsigned int ... _CLeftRadii , unsigned int ... _CRightRadii >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::_NeighborsLoop( ParameterPack::UIntPack< _PLeftRadii ... > pLeftRadii , ParameterPack::UIntPack< _PRightRadii ... > pRightRadii , ParameterPack::UIntPack< _CLeftRadii ... > cLeftRadii , ParameterPack::UIntPack< _CRightRadii ... > cRightRadii , Window::ConstSlice< const RegularTreeNode * , ( _PLeftRadii + _PRightRadii + 1 ) ... > pNeighbors , Window::Slice< const RegularTreeNode * , ( _CLeftRadii + _CRightRadii + 1 ) ... > cNeighbors , int cIdx )
{
	static_assert( Dim==sizeof ... ( _PLeftRadii ) && Dim==sizeof ... ( _PRightRadii ) && Dim==sizeof ... ( _CLeftRadii ) && Dim==sizeof ... ( _CRightRadii ) , "[ERROR] Dimensions don't match" );
	int c[Dim];
	for( int d=0 ; d<Dim ; d++ ) c[d] = ( cIdx>>d ) & 1;
	return _Run< ParameterPack::UIntPack< _PLeftRadii ... > , ParameterPack::UIntPack< _PRightRadii ... > , ParameterPack::UIntPack< _CLeftRadii ... > , ParameterPack::UIntPack< _CRightRadii ... > >::Run( pNeighbors , cNeighbors , c , 0 );
}
template< unsigned int Dim , class NodeData , class DepthAndOffsetType >

template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< unsigned int ... _PLeftRadii , unsigned int ... _PRightRadii , unsigned int ... _CLeftRadii , unsigned int ... _CRightRadii >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::_NeighborsLoop( ParameterPack::UIntPack< _PLeftRadii ... > pLeftRadii , ParameterPack::UIntPack< _PRightRadii ... > pRightRadii , ParameterPack::UIntPack< _CLeftRadii ... > cLeftRadii , ParameterPack::UIntPack< _CRightRadii ... > cRightRadii , Window::Slice< const RegularTreeNode* , ( _PLeftRadii + _PRightRadii + 1 ) ... > pNeighbors , Window::Slice< const RegularTreeNode* , ( _CLeftRadii + _CRightRadii + 1 ) ... > cNeighbors , int cIdx )
{
	return _NeighborsLoop( ParameterPack::UIntPack< _PLeftRadii ... >() , ParameterPack::UIntPack< _PRightRadii ... >() , ParameterPack::UIntPack< _CLeftRadii ... >() , ParameterPack::UIntPack< _CRightRadii ... >() , ( Window::ConstSlice< const RegularTreeNode* , ( _PLeftRadii + _PRightRadii + 1 ) ... > )pNeighbors , cNeighbors , cIdx );
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< unsigned int _PLeftRadius , unsigned int ... _PLeftRadii , unsigned int _PRightRadius , unsigned int ... _PRightRadii , unsigned int _CLeftRadius , unsigned int ... _CLeftRadii , unsigned int _CRightRadius , unsigned int ... _CRightRadii >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::_Run< ParameterPack::UIntPack< _PLeftRadius , _PLeftRadii ... > , ParameterPack::UIntPack< _PRightRadius , _PRightRadii ... > , ParameterPack::UIntPack< _CLeftRadius , _CLeftRadii ... > , ParameterPack::UIntPack< _CRightRadius , _CRightRadii ... > >::Run( Window::ConstSlice< const RegularTreeNode* , _PLeftRadius + _PRightRadius + 1 , ( _PLeftRadii + _PRightRadii + 1 ) ... > pNeighbors , Window::Slice< const RegularTreeNode* ,  _CLeftRadius + _CRightRadius + 1 , ( _CLeftRadii + _CRightRadii + 1 ) ... > cNeighbors , int* c , int cornerIndex )
{
	static const int D = sizeof ... ( _PLeftRadii ) + 1;
	unsigned int count=0;
	for( int i=-(int)_CLeftRadius ; i<=(int)_CRightRadius ; i++ )
	{
		int _i = (i+c[Dim-D]) + ( _CLeftRadius<<1 ) , pi = ( _i>>1 ) - _CLeftRadius + _PLeftRadius  , ci = i + _CLeftRadius;
		count += _Run< ParameterPack::UIntPack< _PLeftRadii ... > , ParameterPack::UIntPack< _PRightRadii ... > , ParameterPack::UIntPack< _CLeftRadii ... > , ParameterPack::UIntPack<  _CRightRadii ... > >::Run( pNeighbors[pi] , cNeighbors[ci] , c , cornerIndex | ( ( _i&1)<<(Dim-D) ) );
	}
	return count;
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< unsigned int _PLeftRadius , unsigned int _PRightRadius , unsigned int _CLeftRadius , unsigned int _CRightRadius  >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::_Run< ParameterPack::UIntPack< _PLeftRadius > , ParameterPack::UIntPack< _PRightRadius > , ParameterPack::UIntPack< _CLeftRadius > , ParameterPack::UIntPack< _CRightRadius > >::Run( Window::ConstSlice< const RegularTreeNode* , _PLeftRadius+_PRightRadius+1 > pNeighbors , Window::Slice< const RegularTreeNode* , _CLeftRadius+_CRightRadius+1 > cNeighbors , int* c , int cornerIndex )
{
	static const int D = 1;
	unsigned int count=0;
	for( int i=-(int)_CLeftRadius ; i<=(int)_CRightRadius ; i++ )
	{
		int _i = (i+c[Dim-D]) + ( _CLeftRadius<<1 ) , pi = ( _i>>1 ) - _CLeftRadius + _PLeftRadius  , ci = i + _CLeftRadius;
		if( pNeighbors[pi] && pNeighbors[pi]->children ) cNeighbors[ci] = pNeighbors[pi]->children + ( cornerIndex | ( ( _i&1)<<(Dim-1) ) ) , count++;
		else cNeighbors[ci] = NULL;
	}
	return count;
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::getChildNeighbors( int cIdx , int d , ConstNeighbors< ParameterPack::UIntPack< ( LeftRadii + RightRadii + 1 ) ... > >& cNeighbors ) const
{
	const ConstNeighbors< ParameterPack::UIntPack< ( LeftRadii + RightRadii + 1 ) ... > >& pNeighbors = neighbors[d];
	// Check that we actually have a center node
	if( !pNeighbors.neighbors.data[ CenterIndex ] ) return 0;

	return _NeighborsLoop( ParameterPack::UIntPack< LeftRadii ... >() , ParameterPack::UIntPack< RightRadii ... >() , ParameterPack::UIntPack< LeftRadii ... >() , ParameterPack::UIntPack< RightRadii ... >() , pNeighbors.neighbors() , cNeighbors.neighbors() , cIdx );
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
typename RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::template ConstNeighbors< ParameterPack::UIntPack< ( LeftRadii + RightRadii + 1 ) ... > >& RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::getNeighbors( const RegularTreeNode* node )
{
	ConstNeighbors< ParameterPack::UIntPack< ( LeftRadii + RightRadii + 1 ) ... > >& neighbors = this->neighbors[ node->depth() ];
	if( node!=neighbors.neighbors.data[ CenterIndex ] )
	{
		for( int d=node->depth()+1 ; d<=_depth && this->neighbors[d].neighbors.data[ CenterIndex ] ; d++ ) this->neighbors[d].neighbors.data[ CenterIndex ] = NULL;
		neighbors.clear();
		if( !node->parent ) neighbors.neighbors.data[ CenterIndex ] = node;
		else _NeighborsLoop( ParameterPack::UIntPack< LeftRadii ... >() , ParameterPack::UIntPack< RightRadii ... >() , ParameterPack::UIntPack< LeftRadii ... >() , ParameterPack::UIntPack< RightRadii ... >() , getNeighbors( node->parent ).neighbors() , neighbors.neighbors() , (int)( node - node->parent->children ) );
	}
	return neighbors;
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< unsigned int ... _LeftRadii , unsigned int ... _RightRadii >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::getNeighbors( ParameterPack::UIntPack< _LeftRadii ... > , ParameterPack::UIntPack< _RightRadii ... > , const RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* node , ConstNeighbors< ParameterPack::UIntPack< ( _LeftRadii + _RightRadii + 1 ) ... > >& neighbors )
{
	static const unsigned int _CenterIndex = Window::Index< ( _LeftRadii + _RightRadii + 1 ) ... >::template I< _LeftRadii ... >();

	neighbors.clear();
	if( !node ) return;

	ParameterPack::UIntPack<  LeftRadii ... >  leftRadii;
	ParameterPack::UIntPack< RightRadii ... > rightRadii;
	ParameterPack::UIntPack< (  _LeftRadii+1 )/2 ... >  pLeftRadii;
	ParameterPack::UIntPack< ( _RightRadii+1 )/2 ... > pRightRadii;
	ParameterPack::UIntPack<  _LeftRadii ... >  cLeftRadii;
	ParameterPack::UIntPack< _RightRadii ... > cRightRadii;
	// If we are at the root of the tree, we are done
	if( !node->parent ) neighbors.neighbors.data[ _CenterIndex ] = node;
	// If we can get the data from the the key for the parent node, do that
	else if( pLeftRadii<=leftRadii && pRightRadii<=rightRadii )
	{
		getNeighbors( node->parent );
		const ConstNeighbors< ParameterPack::UIntPack< ( LeftRadii + RightRadii + 1 ) ... > >& pNeighbors = this->neighbors[ node->depth()-1 ];
		_NeighborsLoop( leftRadii , rightRadii , cLeftRadii , cRightRadii , pNeighbors.neighbors() , neighbors.neighbors() , (int)( node - node->parent->children ) );
	}
	// Otherwise recurse
	else
	{
		ConstNeighbors< ParameterPack::UIntPack< ( ( _LeftRadii+1 )/2  + ( _RightRadii+1 )/2 + 1 ) ... > > pNeighbors;
		getNeighbors( pLeftRadii , pRightRadii , node->parent , pNeighbors );
		_NeighborsLoop( pLeftRadii , pRightRadii , cLeftRadii , cRightRadii , pNeighbors.neighbors() , neighbors.neighbors() , (int)( node - node->parent->children ) );
	}
	return;
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< unsigned int ... _LeftRadii , unsigned int ... _RightRadii >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::getNeighbors( ParameterPack::UIntPack< _LeftRadii ... > , ParameterPack::UIntPack< _RightRadii ... > , const RegularTreeNode< Dim , NodeData , DepthAndOffsetType >* node , ConstNeighbors< ParameterPack::UIntPack< ( _LeftRadii + _RightRadii + 1 ) ... > >& pNeighbors , ConstNeighbors< ParameterPack::UIntPack< ( _LeftRadii + _RightRadii + 1 ) ... > >& neighbors )
{
	ParameterPack::UIntPack<  _LeftRadii ... >  leftRadii;
	ParameterPack::UIntPack< _RightRadii ... > rightRadii;
	if( !node->parent ) return getNeighbors( leftRadii , rightRadii , node , neighbors );
	else
	{
		 getNeighbors( leftRadii , rightRadii , node->parent , pNeighbors );
		_NeighborsLoop( leftRadii , rightRadii , leftRadii , rightRadii , pNeighbors.neighbors() , neighbors.neighbors() , (int)( node - node->parent->children ) );
	}
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
template< class Real >
unsigned int RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::getChildNeighbors( Point< Real , Dim > p , int d , ConstNeighbors< ParameterPack::UIntPack< ( LeftRadii + RightRadii + 1 ) ... > >& cNeighbors ) const
{
	ConstNeighbors< ParameterPack::UIntPack< ( LeftRadii + RightRadii + 1 ) ... > >& pNeighbors = neighbors[d];
	// Check that we actually have a center node
	if( !pNeighbors.neighbors.data[ CenterIndex ] ) return 0;
	Point< Real , Dim > c;
	Real w;
	pNeighbors.neighbors.data[ CenterIndex ]->centerAndWidth( c , w );
	int cIdx = 0;
	for( int d=0 ; d<Dim ; d++ ) if( p[d]>c[d] ) cIdx |= (1<<d);
	return getChildNeighbors( cIdx , d , cNeighbors );
}

template< unsigned int Dim , class NodeData , class DepthAndOffsetType >
template< unsigned int ... LeftRadii , unsigned int ... RightRadii >
void RegularTreeNode< Dim , NodeData , DepthAndOffsetType >::ConstNeighborKey< ParameterPack::UIntPack< LeftRadii ... > , ParameterPack::UIntPack< RightRadii ... > >::setLeafNeighbors( const RegularTreeNode *node , Window::StaticWindow< RegularTreeNode * , ( LeftRadii + RightRadii + 1 ) ... > &leaves )
{
	// Suppose that we have a node at index I and we want the leaf nodes supported on the (possibly virtual) node K away
	// Case 1: The K-th neighbor exists
	// ---> Iterate over the leaf nodes of the sub-tree rooted at the K-th neighbor
	// Case 2: The K-th neighbor does not exist
	// ---> The index of the K-th neighbor is I+K
	// ---> The index of the parent is floor( I/2 )
	// ---> The index of the K-th neighbors parent is floor( (I+K)/2 )
	// ---> The parent of the K-th neighbor is the [ floor( (I+k)/2 ) - floor( I/2 ) ]-th neighbor of the parent

	static const unsigned int _LeftRadii[] = { LeftRadii ... };
	auto GetNeighborLeaf = [&]( unsigned int depth , const int index[Dim] , const int offset[Dim] )
	{
		//		unsigned int _index[Dim] , _offset[Dim] , __offset[Dim];
		int _index[Dim] , _offset[Dim] , __offset[Dim];
		for( int dim=0 ; dim<Dim ; dim++ ) _index[dim] = index[dim] , _offset[dim] = offset[dim];
		for( int d=depth ; d>=0 ; d-- )
		{
			for( int dim=0 ; dim<Dim ; dim++ ) __offset[dim] = _offset[dim] + _LeftRadii[dim];
			if( neighbors[d].neighbors( __offset ) ) return neighbors[d].neighbors( __offset );
			else
			{
				for( int dim=0 ; dim<Dim ; dim++ )
				{
					_offset[dim] = ( ( _index[dim] + _offset[dim] ) >> 1 ) - ( _index[dim] >> 1 );
					_index[dim] >>= 1;
				}
			}
		}
		return ( RegularTreeNode * )NULL;
	};
	getNeighbors( node );
	int depth , index[Dim] , offset[Dim] , _offset[Dim];
	node->depthAndOffset( depth , index );

	Window::Loop< Dim >::Run
	(
		ParameterPack::IsotropicUIntPack< Dim , 0 >() , ParameterPack::UIntPack< ( LeftRadii + RightRadii + 1 ) ... >() ,
		[&]( int d , int i ){ offset[d] = i , _offset[d] = i-_LeftRadii[d]; } ,
		[&]( void ){ leaves( offset ) = GetNeighborLeaf( depth , index , _offset ); }
	);
}
