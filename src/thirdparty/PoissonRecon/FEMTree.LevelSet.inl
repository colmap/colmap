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

// Level-set extraction data
namespace LevelSetExtraction
{
	/////////
	// Key //
	/////////
	template< unsigned int Dim >
	struct Key
	{
	protected:
		template< unsigned int _Dim > friend struct KeyGenerator;
	public:
		unsigned int idx[Dim];

		Key( void ){ for( unsigned int d=0 ; d<Dim ; d++ ) idx[d] = -1; }

		unsigned int &operator[]( int i ){ return idx[i]; }
		const unsigned int &operator[]( int i ) const { return idx[i]; }

		bool operator == ( const Key &key ) const
		{
			for( unsigned int d=0 ; d<Dim ; d++ ) if( idx[d]!=key[d] ) return false;
			return true;
		}
		bool operator != ( const Key &key ) const { return !operator==( key ); }

		std::string to_string( void ) const
		{
			std::stringstream stream;
			stream << "(";
			for( unsigned int d=0 ; d<Dim ; d++ )
			{
				if( d ) stream << ",";
				stream << idx[d];
			}
			stream << ")";
			return stream.str();
		}

		friend std::ostream &operator << ( std::ostream &os , const Key &key ){ return os << key.to_string(); }

		friend Key SetAtomic( volatile Key & value , Key newValue )
		{
			Key oldValue;
			for( unsigned int d=0 ; d<Dim ; d++ ) oldValue.idx[d] = PoissonRecon::SetAtomic( value.idx[d] , newValue.idx[d] );
			return oldValue;
		}

#ifdef SHOW_WARNINGS
#pragma message( "[WARNING] Could we hash better?" )
#endif // SHOW_WARNINGS
		struct Hasher
		{
			size_t operator()( const Key &i ) const
			{
				size_t hash = 0;
				for( unsigned int d=0 ; d<Dim ; d++ ) hash ^= i.idx[d];
				return hash;
			}
		};
	};

	template< unsigned int Dim , typename Data > using KeyMap = std::unordered_map< Key< Dim > , Data , typename Key< Dim >::Hasher >;

	/////////////
	// IsoEdge //
	/////////////
	template< unsigned int Dim >
	struct IsoEdge
	{
		Key< Dim > vertices[2];
		IsoEdge( void ) {}
		IsoEdge( Key< Dim > v1 , Key< Dim > v2 ){ vertices[0] = v1 , vertices[1] = v2; }
		Key< Dim > &operator[]( int idx ){ return vertices[idx]; }
		const Key< Dim > &operator[]( int idx ) const { return vertices[idx]; }

		friend IsoEdge SetAtomic( volatile IsoEdge & value , IsoEdge newValue )
		{
			IsoEdge oldValue;
			oldValue.vertices[0] = SetAtomic( value.vertices[0] , newValue.vertices[0] );
			oldValue.vertices[1] = SetAtomic( value.vertices[1] , newValue.vertices[1] );
			return oldValue;
		}
	};

	/////////////////////
	// HyperCubeTables //
	/////////////////////
	template< unsigned int D , unsigned int ... Ks > struct HyperCubeTables{};

	template< unsigned int D , unsigned int K1 , unsigned int K2 >
	struct HyperCubeTables< D , K1 , K2 >
	{
		// The number of K1-dimensional elements in a D-dimensional cube
		static constexpr unsigned int ElementNum1 = HyperCube::Cube< D >::template ElementNum< K1 >();
		// The number of K2-dimensional elements in a D-dimensional cube
		static constexpr unsigned int ElementNum2 = HyperCube::Cube< D >::template ElementNum< K2 >();
		// The number of K2-dimensional elements overlapping the K1-dimensional element
		static constexpr unsigned int OverlapElementNum = HyperCube::Cube< D >::template OverlapElementNum< K1 , K2 >();
		// The list of K2-dimensional elements overlapping the K1-dimensional element
		static typename HyperCube::Cube< D >::template Element< K2 > OverlapElements[ ElementNum1 ][ OverlapElementNum ];
		// Indicates if the K2-dimensional element overlaps the K1-dimensional element
		static bool Overlap[ ElementNum1 ][ ElementNum2 ];

		static void SetTables( void )
		{
			for( typename HyperCube::Cube< D >::template Element< K1 > e ; e<HyperCube::Cube< D >::template ElementNum< K1 >() ; e++ )
			{
				for( typename HyperCube::Cube< D >::template Element< K2 > _e ; _e<HyperCube::Cube< D >::template ElementNum< K2 >() ; _e++ )
					Overlap[e.index][_e.index] = HyperCube::Cube< D >::Overlap( e , _e );
				HyperCube::Cube< D >::OverlapElements( e , OverlapElements[e.index] );
			}
			if( !K2 ) HyperCubeTables< D , K1 >::SetTables();
		}
	};

	template< unsigned int D , unsigned int K >
	struct HyperCubeTables< D , K >
	{
		static constexpr unsigned int IncidentCubeNum = HyperCube::Cube< D >::template IncidentCubeNum< K >();
		static constexpr unsigned int ElementNum = HyperCube::Cube< D >::template ElementNum< K >();
		static unsigned int CellOffset[ ElementNum ][ IncidentCubeNum ];
		static unsigned int IncidentElementCoIndex[ ElementNum ][ IncidentCubeNum ];
		static unsigned int IncidentElementIndex[ ElementNum ][ IncidentCubeNum ];
		static unsigned int CellOffsetAntipodal[ ElementNum ];
		static typename HyperCube::Cube< D >::template IncidentCubeIndex< K > IncidentCube[ ElementNum ];
		static typename HyperCube::Direction Directions[ ElementNum ][ D ];

		static void SetTables( void )
		{
			for( typename HyperCube::Cube< D >::template Element< K > e ; e<HyperCube::Cube< D >::template ElementNum< K >() ; e++ )
			{
				for( typename HyperCube::Cube< D >::template IncidentCubeIndex< K > i ; i<HyperCube::Cube< D >::template IncidentCubeNum< K >() ; i++ )
				{
					CellOffset[e.index][i.index] = HyperCube::Cube< D >::CellOffset( e , i );
					IncidentElementCoIndex[e.index][i.index] = HyperCube::Cube< D >::IncidentElement( e , i ).coIndex();
					IncidentElementIndex[e.index][i.index] = HyperCube::Cube< D >::IncidentElement( e , i ).index;
				}
				CellOffsetAntipodal[e.index] = HyperCube::Cube< D >::CellOffset( e , HyperCube::Cube< D >::IncidentCube( e ).antipodal() );
				IncidentCube[ e.index ] = HyperCube::Cube< D >::IncidentCube( e );
				e.directions( Directions[e.index] );
			}
		}
	};

	template< unsigned int D , unsigned int K1=D , unsigned int K2=D >
	static void SetHyperCubeTables( void )
	{
		if constexpr( K2!=0 )
		{
			HyperCubeTables< D , K1 , K2 >::SetTables();
			SetHyperCubeTables< D , K1 , K2-1 >();
		}
		else if constexpr( K1!=0 && K2==0 )
		{
			HyperCubeTables< D , K1 , K2 >::SetTables();
			SetHyperCubeTables< D , K1-1 , D >();
		}
		else if constexpr( D!=1 && K1==0 && K2==0 )
		{
			HyperCubeTables< D , K1 , K2 >::SetTables();
			SetHyperCubeTables< D-1 , D-1 , D-1 >();
		}
		else HyperCubeTables< D , K1 , K2 >::SetTables();
	}

	// A helper class for storing a static array
	template< unsigned int Indices >
	struct _Indices
	{
		node_index_type idx[Indices];
		_Indices( void ){ for( unsigned int i=0 ; i<Indices ; i++ ) idx[i] = -1; }
		node_index_type& operator[] ( int i ) { return idx[i]; }
		const node_index_type& operator[] ( int i ) const { return idx[i]; }
	};

	template< unsigned int Dim , unsigned int CellDim >
	struct _CellIndices
	{
		static const unsigned int ElementNum = HyperCube::Cube< Dim >::template ElementNum< CellDim >();
		using Type = _Indices< ElementNum >;
	};

	// A helper struct for storing arrays of different types of indices
	template< unsigned int Dim , typename T > struct _Tables{};
	template< unsigned int Dim , unsigned int ... CellDimensions > struct _Tables< Dim , std::integer_sequence< unsigned int , CellDimensions ... > >
	{
		typedef std::tuple< Pointer( typename _CellIndices< Dim , CellDimensions >::Type ) ... > Type;
	};

	// Temporary data used for tracking ownership of cells shared by different nodes
	template< unsigned int Dim , unsigned int MaxCellDim >
	struct _Scratch
	{
		Pointer( node_index_type ) maps[MaxCellDim+1];
		_Scratch( void ) : _reserved(0) { for( unsigned int i=0 ; i<=MaxCellDim ; i++ ) maps[i] = NullPointer( node_index_type ); }
		~_Scratch( void ){ clear(); }
		void clear( void ){ for( unsigned int i=0 ; i<=MaxCellDim ; i++ ) DeletePointer( maps[i] ); }
		void resize( size_t sz )
		{
			if( sz>_reserved )
			{
				clear();
				_allocate<0>( sz );
				_reserved = sz;
			}
			_zeroOut<0>( sz );
		}
	protected:

		template< unsigned int CellDim >
		void _allocate( size_t sz )
		{
			maps[ CellDim ] = NewPointer< node_index_type >( sz * HyperCube::Cube< Dim >::template ElementNum< CellDim >() );

			if constexpr( CellDim==MaxCellDim ) return;
			else _allocate< CellDim+1 >( sz );
		}

		template< unsigned int CellDim >
		void _zeroOut( size_t sz )
		{
			if( sz && maps[CellDim] ) memset( maps[CellDim] , 0 , sizeof(node_index_type) * sz * HyperCube::Cube< Dim >::template ElementNum< CellDim >() );
			else if( sz ) MK_THROW( "Traying to zero out null pointer" );

			if constexpr( CellDim==MaxCellDim ) return;
			else _zeroOut< CellDim+1 >( sz );
		}
		size_t _reserved;
	};

	///////////////////
	// CellIndexData //
	///////////////////
	// A data-struture that allows us to take a node together with a local cell index and returns the global cell's index
	template< unsigned int Dim , unsigned int _MaxCellDim >
	struct CellIndexData
	{
		static const unsigned int MaxCellDim = _MaxCellDim;
		using TreeNode = RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >;

		// Static arrays for holding specific dimension cell indices
		template< unsigned int CellDim > using CellIndices = typename _CellIndices< Dim , CellDim >::Type;

		// A tuple of pointers to cell indices of dimensions {0,...,MaxCellDim}
		using TablesType = typename _Tables< Dim , std::make_integer_sequence< unsigned int , MaxCellDim+1 > >::Type;

		// The tables holding the cell indices of different dimensions
		TablesType tables;

		// The count of the different number of cells
		size_t counts[MaxCellDim+1];

		CellIndexData( void ) : _size(0) , _capacity(0) { _init<0>(); }
		~CellIndexData( void ){ clear(); }

		void clear( void ){ _clear<0>(); }
		void resize( size_t sz )
		{
			if( sz>_capacity )
			{
				_clear<0>();
				_allocate<0>( sz );
				_capacity = sz;
			}
			else for( unsigned int d=0 ; d<=MaxCellDim ; d++ ) counts[d] = 0;
			_size = sz;
		}
		size_t size( void ) const { return _size; }

		void read( BinaryStream &stream )
		{
			if( !stream.read( _size ) ) MK_THROW( "Failed to read node count" );
			resize( _size );
			if( _size ) _read<0>( stream );
		}

		void write( BinaryStream &stream ) const
		{
			stream.write( _size );
			if( _size ) _write<0>( stream );
		}

	protected:
		size_t _size , _capacity;

		template< unsigned int CellDim >
		void _init( void )
		{
			counts[CellDim] = 0;
			std::get< CellDim >( tables ) = NullPointer( CellIndices< CellDim > );

			if constexpr( CellDim==MaxCellDim ) return;
			else _init< CellDim+1 >();
		}

		template< unsigned int CellDim >
		void _clear( void )
		{
			counts[CellDim] = 0;
			DeletePointer( std::get< CellDim >( tables ) );

			if constexpr( CellDim==MaxCellDim ) return;
			else _clear< CellDim+1 >();
		}

		template< unsigned int CellDim >
		void _allocate( size_t sz )
		{
			std::get< CellDim >( tables ) = NewPointer< CellIndices< CellDim > >( sz );

			if constexpr( CellDim==MaxCellDim ) return;
			else _allocate< CellDim+1 >( sz );
		}

		template< unsigned int CellDim >
		void _read( BinaryStream &stream )
		{
			if( !stream.read( counts[CellDim] ) ) MK_THROW( "Failed to read count at dimension: " , CellDim );
			if( !stream.read( std::get< CellDim >( tables ) , _size ) ) MK_THROW( "Failed to read table at dimension: " , CellDim );

			if constexpr( CellDim==MaxCellDim ) return;
			else _read< CellDim+1 >( stream );
		}

		template< unsigned int CellDim >
		void _write( BinaryStream &stream ) const
		{
			stream.write( counts[CellDim] );
			stream.write( std::get< CellDim >( tables ) , _size );

			if constexpr( CellDim==MaxCellDim ) return;
			else _write< CellDim+1 >( stream );
		}
	};

	///////////////////////
	// FullCellIndexData //
	///////////////////////
	// A data-struture that allows us to take a node together with a local cell index and returns the global cell's index
	// [WARNING] Do we want to go up to Dim or up to Dim-1?
	template< unsigned int Dim >
	struct FullCellIndexData : public CellIndexData< Dim , Dim >
	{
		static const unsigned int MaxCellDim = CellIndexData< Dim , Dim >::MaxCellDim;
		template< unsigned int CellDim > using CellIndices = typename CellIndexData< Dim , Dim >::template CellIndices< CellDim >;
		using TreeNode = RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >;
		using ConstOneRingNeighborKey = typename TreeNode::template ConstNeighborKey< IsotropicUIntPack< Dim , 1 > , IsotropicUIntPack< Dim , 1 > >;
		using ConstNeighbors = typename TreeNode::template ConstNeighbors< IsotropicUIntPack< Dim , 3 > >;
		using CellIndexData< Dim , Dim >::tables;
		using CellIndexData< Dim , Dim >::size;
		using CellIndexData< Dim , Dim >::resize;
		using CellIndexData< Dim , Dim >::counts;

		node_index_type nodeOffset;

		FullCellIndexData( void ) : nodeOffset(0) {}

		void read( BinaryStream &stream )
		{
			if( !stream.read( nodeOffset ) ) MK_THROW( "Failed to read node ofset" );
			CellIndexData< Dim , MaxCellDim >::read( stream );
		}

		void write( BinaryStream &stream ) const
		{
			stream.write( nodeOffset );
			CellIndexData< Dim , MaxCellDim >::write( stream );
		}

		void set( const SortedTreeNodes< Dim > &sNodes , int depth )
		{
			std::pair< node_index_type , node_index_type > span( sNodes.begin( depth ) , sNodes.end( depth ) );
			nodeOffset = (size_t)span.first;
			resize( (size_t)( span.second-span.first ) );
			_scratch.resize( size() );

			std::vector< ConstOneRingNeighborKey > neighborKeys( ThreadPool::NumThreads() );
			for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( depth );

			// Try and get at the nodes outside of the slab through the neighbor key
			ThreadPool::ParallelFor( sNodes.begin(depth) , sNodes.end(depth) , [&]( unsigned int thread , size_t i )
				{
					ConstOneRingNeighborKey& neighborKey = neighborKeys[ thread ];
					const TreeNode *node = sNodes.treeNodes[i];
					ConstNeighbors &neighbors = neighborKey.getNeighbors( node );
					_setProcess<0>( neighbors , _scratch.maps );
				}
			);

			_setCounts<0>( _scratch.maps );
			ThreadPool::ParallelFor( 0 , size() , [&]( unsigned int , size_t i ){ _setTables<0>( (unsigned int)i , _scratch.maps ); } );
		}

		// Maps from tree nodes (and their associated indices) to the associated indices for the cell indices
		template< unsigned int CellDim >       CellIndices< CellDim > &indices( const TreeNode *node      )       { return std::get< CellDim >( tables )[ node->nodeData.nodeIndex - nodeOffset ]; }
		template< unsigned int CellDim >       CellIndices< CellDim > &indices( node_index_type nodeIndex )       { return std::get< CellDim >( tables )[                nodeIndex - nodeOffset ]; }
		template< unsigned int CellDim > const CellIndices< CellDim > &indices( const TreeNode *node      ) const { return std::get< CellDim >( tables )[ node->nodeData.nodeIndex - nodeOffset ]; }
		template< unsigned int CellDim > const CellIndices< CellDim > &indices( node_index_type nodeIndex ) const { return std::get< CellDim >( tables )[                nodeIndex - nodeOffset ]; }

	protected:
		_Scratch< Dim , MaxCellDim > _scratch;

		template< unsigned int CellDim >
		void _setProcess( const ConstNeighbors& neighbors , Pointer( node_index_type ) maps[MaxCellDim+1] )
		{
			const TreeNode *node = neighbors.neighbors.data[ WindowIndex< IsotropicUIntPack< Dim , 3 > , IsotropicUIntPack< Dim , 1 > >::Index ];
			node_index_type i = node->nodeData.nodeIndex;

			// Iterate over the cells in the node
			for( typename HyperCube::Cube< Dim >::template Element< CellDim > c ; c<HyperCube::Cube< Dim >::template ElementNum< CellDim >() ; c++ )
			{
				bool owner = true;

				// The index of the node relative to the cell
				typename HyperCube::Cube< Dim >::template IncidentCubeIndex< CellDim > my_ic = HyperCubeTables< Dim , CellDim >::IncidentCube[c.index];

				// Iterate over the nodes adjacent to the cell
				for( typename HyperCube::Cube< Dim >::template IncidentCubeIndex< CellDim > ic ; ic<HyperCube::Cube< Dim >::template IncidentCubeNum< CellDim >() ; ic++ )
				{
					// Get the index of node relative to the cell neighbors
					unsigned int xx = HyperCubeTables< Dim , CellDim >::CellOffset[c.index][ic.index];
					// If the neighbor exists and comes before, they own the corner
					if( IsActiveNode< Dim >( neighbors.neighbors.data[xx] ) && ic<my_ic ){ owner = false ; break; }
				}
				if( owner )
				{
					node_index_type myCount = ( i - nodeOffset ) * HyperCube::Cube< Dim >::template ElementNum< CellDim >() + c.index;
					maps[CellDim][ myCount ] = 1;
					// Set the cell index for all nodes incident on the cell
					for( typename HyperCube::Cube< Dim >::template IncidentCubeIndex< CellDim > ic ; ic<HyperCube::Cube< Dim >::template IncidentCubeNum< CellDim >() ; ic++ )	// Iterate over the nodes adjacent to the cell
					{
						unsigned int xx = HyperCubeTables< Dim , CellDim >::CellOffset[c.index][ic.index];
						// If the neighbor exits, sets its corner
						if( IsActiveNode< Dim >( neighbors.neighbors.data[xx] ) ) this->template indices< CellDim >( neighbors.neighbors.data[xx] )[ HyperCubeTables< Dim , CellDim >::IncidentElementIndex[c.index][ic.index] ] = myCount;
					}
				}
			}

			if constexpr( CellDim==MaxCellDim ) return;
			else _setProcess< CellDim+1 >( neighbors , maps );
		}

		template< unsigned int CellDim >
		void _setCounts( Pointer( node_index_type ) maps[MaxCellDim+1] )
		{
			node_index_type count = 0;
			for( node_index_type i=0 ; i<(node_index_type)size() * (node_index_type)HyperCube::Cube< Dim >::template ElementNum< CellDim >() ; i++ )
				if( maps[CellDim][i] ) maps[CellDim][i] = count++;
			counts[ CellDim ] = count;

			if constexpr( CellDim==MaxCellDim ) return;
			else _setCounts< CellDim+1 >( maps );
		}

		template< unsigned int CellDim >
		void _setTables( unsigned int i , Pointer( node_index_type ) maps[MaxCellDim+1] )
		{
			for( unsigned int j=0 ; j<HyperCube::Cube< Dim >::template ElementNum< CellDim >() ; j++ )
				std::get< CellDim >( tables )[i][j] = maps[CellDim][ std::get< CellDim >( tables )[i][j] ];

			if constexpr( CellDim==MaxCellDim ) return;
			else _setTables< CellDim+1 >( i , maps );
		}
	};

	////////////////////////
	// SliceCellIndexData //
	////////////////////////
	// A data-struture that allows us to take a node adjacent to a slice together with a local cell index and returns the global cell's index
	template< unsigned int Dim >
	struct SliceCellIndexData : public CellIndexData< Dim-1 , Dim-1 >
	{
		static const unsigned int _Dim = Dim-1;
		static const unsigned int MaxCellDim = CellIndexData< _Dim , _Dim >::MaxCellDim;
		template< unsigned int CellDim > using CellIndices = typename CellIndexData< _Dim , _Dim >::template CellIndices< CellDim >;
		using TreeNode = RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >;
		using ConstOneRingNeighborKey = typename TreeNode::template ConstNeighborKey< IsotropicUIntPack< Dim , 1 > , IsotropicUIntPack< Dim , 1 > >;
		using ConstNeighbors = typename TreeNode::template ConstNeighbors< IsotropicUIntPack< Dim , 3 > >;
		using CellIndexData< _Dim , _Dim >::tables;
		using CellIndexData< _Dim , _Dim >::size;
		using CellIndexData< _Dim , _Dim >::resize;
		using CellIndexData< _Dim , _Dim >::counts;

		node_index_type nodeOffset;

		SliceCellIndexData( void ) : nodeOffset(0) {}

		void read( BinaryStream &stream )
		{
			if( !stream.read( nodeOffset ) ) MK_THROW( "Failed to read node ofset" );
			CellIndexData< _Dim , _Dim >::read( stream );
		}

		void write( BinaryStream &stream ) const
		{
			stream.write( nodeOffset );
			CellIndexData< _Dim , _Dim >::write( stream );
		}

		void set( const SortedTreeNodes< Dim > &sNodes , int depth , int slice )
		{
			std::pair< node_index_type , node_index_type > span( sNodes.begin( depth , slice-1 ) , sNodes.end( depth , slice ) );
			nodeOffset = (size_t)span.first;
			resize( (size_t)( span.second - span.first ) );
			_scratch.resize( size() );

			std::vector< ConstOneRingNeighborKey > neighborKeys( ThreadPool::NumThreads() );
			for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( depth );

			// Try and get at the nodes outside of the slab through the neighbor key
			ThreadPool::ParallelFor( sNodes.begin( depth , slice-1 ) , sNodes.end( depth , slice ) , [&]( unsigned int thread , size_t i )
				{
					ConstOneRingNeighborKey &neighborKey = neighborKeys[ thread ];
					const TreeNode *node = sNodes.treeNodes[i];
					ConstNeighbors &neighbors = neighborKey.getNeighbors( node );
					_setProcess<0>( neighbors , i<(size_t)sNodes.end( depth , slice-1 ) , _scratch.maps );
				}
			);

			_setCounts<0>( _scratch.maps );
			ThreadPool::ParallelFor( 0 , size() , [&]( unsigned int , size_t i ){ _setTables<0>( (unsigned int)i , _scratch.maps ); } );
		}

		// Maps from tree nodes (and their associated indices) to the associated indices for the cell indices
		template< unsigned int CellDim >       CellIndices< CellDim > &indices( const TreeNode *node      )       { return std::get< CellDim >( tables )[ node->nodeData.nodeIndex - nodeOffset ]; }
		template< unsigned int CellDim >       CellIndices< CellDim > &indices( node_index_type nodeIndex )       { return std::get< CellDim >( tables )[                nodeIndex - nodeOffset ]; }
		template< unsigned int CellDim > const CellIndices< CellDim > &indices( const TreeNode *node      ) const { return std::get< CellDim >( tables )[ node->nodeData.nodeIndex - nodeOffset ]; }
		template< unsigned int CellDim > const CellIndices< CellDim > &indices( node_index_type nodeIndex ) const { return std::get< CellDim >( tables )[                nodeIndex - nodeOffset ]; }

	protected:
		_Scratch< _Dim , MaxCellDim > _scratch;

		template< unsigned int CellDim >
		void _setProcess( const ConstNeighbors& neighbors , bool fromBehind , Pointer( node_index_type ) maps[MaxCellDim+1] )
		{
			HyperCube::Direction dir = fromBehind ? HyperCube::FRONT : HyperCube::BACK;
			const TreeNode *node = neighbors.neighbors.data[ WindowIndex< IsotropicUIntPack< Dim , 3 > , IsotropicUIntPack< Dim , 1 > >::Index ];
			node_index_type i = node->nodeData.nodeIndex;

			// Iterate over the cells in the face
			for( typename HyperCube::Cube< _Dim >::template Element< CellDim > _c ; _c<HyperCube::Cube< _Dim >::template ElementNum< CellDim >() ; _c++ )
			{
				// Get the corresponding index of the cell in the node
				typename HyperCube::Cube< Dim >::template Element< CellDim > c( dir , _c.index );

				bool owner = true;

				// The index of the node relative to the cell
				typename HyperCube::Cube< Dim >::template IncidentCubeIndex< CellDim > my_ic = HyperCubeTables< Dim , CellDim >::IncidentCube[c.index];

				// Iterate over the nodes adjacent to the cell
				for( typename HyperCube::Cube< Dim >::template IncidentCubeIndex< CellDim > ic ; ic<HyperCube::Cube< Dim >::template IncidentCubeNum< CellDim >() ; ic++ )
				{
					// Get the index of node relative to the cell neighbors
					unsigned int xx = HyperCubeTables< Dim , CellDim >::CellOffset[c.index][ic.index];
					// If the neighbor exists and comes before, they own the corner
					if( IsActiveNode< Dim >( neighbors.neighbors.data[xx] ) && ic<my_ic ){ owner = false ; break; }
				}
				if( owner )
				{
					node_index_type myCount = ( i - nodeOffset ) * HyperCube::Cube< _Dim >::template ElementNum< CellDim >() + _c.index;
					maps[CellDim][ myCount ] = 1;
					// Set the cell index for all nodes incident on the cell
					for( typename HyperCube::Cube< Dim >::template IncidentCubeIndex< CellDim > ic ; ic<HyperCube::Cube< Dim >::template IncidentCubeNum< CellDim >() ; ic++ )	// Iterate over the nodes adjacent to the cell
					{
						unsigned int xx = HyperCubeTables< Dim , CellDim >::CellOffset[c.index][ic.index];
						// If the neighbor exits, sets its cell
						if( IsActiveNode< Dim >( neighbors.neighbors.data[xx] ) ) this->template indices< CellDim >( neighbors.neighbors.data[xx] )[ HyperCubeTables< Dim , CellDim >::IncidentElementCoIndex[c.index][ic.index] ] = myCount;
					}
				}
			}

			if constexpr( CellDim==MaxCellDim ) return;
			else _setProcess< CellDim+1 >( neighbors , fromBehind , maps );
		}

		template< unsigned int CellDim >
		void _setCounts( Pointer( node_index_type ) maps[MaxCellDim+1] )
		{
			node_index_type count = 0;
			for( node_index_type i=0 ; i<(node_index_type)size() * (node_index_type)HyperCube::Cube< _Dim >::template ElementNum< CellDim >() ; i++ )
				if( maps[CellDim][i] ) maps[CellDim][i] = count++;
			counts[ CellDim ] = count;

			if constexpr( CellDim==MaxCellDim ) return;
			else _setCounts< CellDim+1 >( maps );
		}

		template< unsigned int CellDim >
		void _setTables( unsigned int i , Pointer( node_index_type ) maps[MaxCellDim+1] )
		{
			for( unsigned int j=0 ; j<HyperCube::Cube< _Dim >::template ElementNum< CellDim >() ; j++ )
				std::get< CellDim >( tables )[i][j] = maps[CellDim][ std::get< CellDim >( tables )[i][j] ];

			if constexpr( CellDim==MaxCellDim ) return;
			else _setTables< CellDim+1 >( i , maps );
		}
	};

	///////////////////////
	// SlabCellIndexData //
	///////////////////////
	// A data-struture that allows us to take a node adjacent to together with a local cell index and returns the global cell's index
	template< unsigned int Dim >
	struct SlabCellIndexData : public CellIndexData< Dim-1 , Dim-1 >
	{
		static const unsigned int _Dim = Dim-1;
		static const unsigned int _MaxCellDim = CellIndexData< _Dim , _Dim >::MaxCellDim;
		template< unsigned int _CellDim > using CellIndices = typename CellIndexData< _Dim , _Dim >::template CellIndices< _CellDim >;
		using TreeNode = RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >;
		using ConstOneRingNeighborKey = typename TreeNode::template ConstNeighborKey< IsotropicUIntPack< Dim , 1 > , IsotropicUIntPack< Dim , 1 > >;
		using ConstNeighbors = typename TreeNode::template ConstNeighbors< IsotropicUIntPack< Dim , 3 > >;
		using CellIndexData< _Dim , _Dim >::tables;
		using CellIndexData< _Dim , _Dim >::size;
		using CellIndexData< _Dim , _Dim >::resize;
		using CellIndexData< _Dim , _Dim >::counts;

		node_index_type nodeOffset;

		SlabCellIndexData( void ) : nodeOffset(0) {}

		void read( BinaryStream &stream )
		{
			if( !stream.read( nodeOffset ) ) MK_THROW( "Failed to read node ofset" );
			CellIndexData< _Dim , _Dim >::read( stream );
		}

		void write( BinaryStream &stream ) const
		{
			stream.write( nodeOffset );
			CellIndexData< _Dim , _Dim >::write( stream );
		}

		void set( const SortedTreeNodes< Dim > &sNodes , int depth , int slab )
		{
			std::pair< node_index_type , node_index_type > span( sNodes.begin( depth , slab ) , sNodes.end( depth , slab ) );
			nodeOffset = (size_t)span.first;
			resize( (size_t)( span.second - span.first ) );
			_scratch.resize( size() );

			std::vector< ConstOneRingNeighborKey > neighborKeys( ThreadPool::NumThreads() );
			for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( depth );

			// Try and get at the nodes outside of the slab through the neighbor key
			ThreadPool::ParallelFor( sNodes.begin( depth , slab ) , sNodes.end( depth , slab ) , [&]( unsigned int thread , size_t i )
				{
					ConstOneRingNeighborKey &neighborKey = neighborKeys[ thread ];
					const TreeNode *node = sNodes.treeNodes[i];
					ConstNeighbors &neighbors = neighborKey.getNeighbors( node );
					_setProcess<0>( neighbors , _scratch.maps );
				}
			);

			_setCounts<0>( _scratch.maps );
			ThreadPool::ParallelFor( 0 , size() , [&]( unsigned int , size_t i ){ _setTables<0>( (unsigned int)i , _scratch.maps ); } );
		}

		// Maps from tree nodes (and their associated indices) to the associated indices for the cell indices
		template< unsigned int CellDim >       CellIndices< CellDim > &indices( const TreeNode *node      )       { return std::get< CellDim >( tables )[ node->nodeData.nodeIndex - nodeOffset ]; }
		template< unsigned int CellDim >       CellIndices< CellDim > &indices( node_index_type nodeIndex )       { return std::get< CellDim >( tables )[                nodeIndex - nodeOffset ]; }
		template< unsigned int CellDim > const CellIndices< CellDim > &indices( const TreeNode *node      ) const { return std::get< CellDim >( tables )[ node->nodeData.nodeIndex - nodeOffset ]; }
		template< unsigned int CellDim > const CellIndices< CellDim > &indices( node_index_type nodeIndex ) const { return std::get< CellDim >( tables )[                nodeIndex - nodeOffset ]; }

	protected:
		_Scratch< _Dim , _MaxCellDim > _scratch;


		template< unsigned int _CellDim >
		void _setProcess( const ConstNeighbors& neighbors , Pointer( node_index_type ) maps[_MaxCellDim+1] )
		{
			static const unsigned int CellDim = _CellDim+1;
			const TreeNode *node = neighbors.neighbors.data[ WindowIndex< IsotropicUIntPack< Dim , 3 > , IsotropicUIntPack< Dim , 1 > >::Index ];
			node_index_type i = node->nodeData.nodeIndex;

			// Iterate over the cells in the face
			for( typename HyperCube::Cube< _Dim >::template Element< _CellDim > _c ; _c<HyperCube::Cube< _Dim >::template ElementNum< _CellDim >() ; _c++ )
			{
				// Get the index of the extruded cell
				typename HyperCube::Cube< Dim >::template Element< CellDim > c( HyperCube::CROSS , _c.index );
				bool owner = true;

				// The index of the node relative to the extruded cell
				typename HyperCube::Cube< Dim >::template IncidentCubeIndex< CellDim > my_ic = HyperCubeTables< Dim , CellDim >::IncidentCube[c.index];

				// Iterate over the nodes adjacent to the cell
				for( typename HyperCube::Cube< Dim >::template IncidentCubeIndex< CellDim > ic ; ic<HyperCube::Cube< Dim >::template IncidentCubeNum< CellDim >() ; ic++ )
				{
					// Get the index of node relative to the cell neighbors
					unsigned int xx = HyperCubeTables< Dim , CellDim >::CellOffset[c.index][ic.index];
					// If the neighbor exists and comes before, they own the corner
					if( IsActiveNode< Dim >( neighbors.neighbors.data[xx] ) && ic<my_ic ){ owner = false ; break; }
				}
				if( owner )
				{
					node_index_type myCount = ( i - nodeOffset ) * HyperCube::Cube< _Dim >::template ElementNum< _CellDim >() + _c.index;
					maps[_CellDim][ myCount ] = 1;
					// Set the cell index for all nodes incident on the cell
					for( typename HyperCube::Cube< Dim >::template IncidentCubeIndex< CellDim > ic ; ic<HyperCube::Cube< Dim >::template IncidentCubeNum< CellDim >() ; ic++ )	// Iterate over the nodes adjacent to the cell
					{
						unsigned int xx = HyperCubeTables< Dim , CellDim >::CellOffset[c.index][ic.index];
						// If the neighbor exits, sets its cell
						if( IsActiveNode< Dim >( neighbors.neighbors.data[xx] ) )
							this->template indices< _CellDim >( neighbors.neighbors.data[xx] )[ HyperCubeTables< Dim , CellDim >::IncidentElementCoIndex[c.index][ic.index] ] = myCount;
					}
				}
			}

			if constexpr( _CellDim==_MaxCellDim ) return;
			else _setProcess< _CellDim+1 >( neighbors , maps );
		}

		template< unsigned int _CellDim >
		void _setCounts( Pointer( node_index_type ) maps[_MaxCellDim+1] )
		{
			node_index_type count = 0;
			for( node_index_type i=0 ; i<(node_index_type)size() * (node_index_type)HyperCube::Cube< _Dim >::template ElementNum< _CellDim >() ; i++ )
				if( maps[_CellDim][i] ) maps[_CellDim][i] = count++;
			counts[ _CellDim ] = count;

			if constexpr( _CellDim==_MaxCellDim ) return;
			else _setCounts< _CellDim+1 >( maps );
		}

		template< unsigned int _CellDim >
		void _setTables( unsigned int i , Pointer( node_index_type ) maps[_MaxCellDim+1] )
		{
			for( unsigned int j=0 ; j<HyperCube::Cube< _Dim >::template ElementNum< _CellDim >() ; j++ )
				std::get< _CellDim >( tables )[i][j] = maps[_CellDim][ std::get< _CellDim >( tables )[i][j] ];

			if constexpr( _CellDim==_MaxCellDim ) return;
			else _setTables< _CellDim+1 >( i , maps );
		}
	};

	//////////////////
	// KeyGenerator //
	//////////////////
	template< unsigned int Dim >
	struct KeyGenerator
	{
		KeyGenerator( int maxDepth ) : _maxDepth(maxDepth){}

		template< unsigned int CellDim >
		Key< Dim > operator()( int depth , const int offset[Dim] , typename HyperCube::Cube< Dim >::template Element< CellDim > e ) const
		{
			static_assert( ( CellDim<=Dim ) , "[ERROR] Cell dimension cannot exceed total dimension" );
			Key< Dim > key;
			const HyperCube::Direction* x = HyperCubeTables< Dim , CellDim >::Directions[ e.index ];
			for( int dd=0 ; dd<Dim ; dd++ ) key[dd] = index( depth , offset[dd] , x[dd] );
			return key;
		}

		Key< Dim > operator()( int depth , int offset , Key< Dim-1 > key ) const
		{
			Key< Dim > pKey;
			if( depth>_maxDepth ) MK_THROW( "Depth cannot exceed max depth: " , depth , " <= " , _maxDepth );
			for( unsigned int d=0 ; d<Dim-1 ; d++ ) pKey[d] = key[d];
			pKey[Dim-1] = cornerIndex( depth , offset );
			return pKey;
		}

		// The corner indices are divisible by 4
		unsigned int cornerIndex( unsigned int depth , unsigned int offset ) const { return offset<<( _maxDepth+2-depth); }
		// The edge indices are odd
		unsigned int   edgeIndex( unsigned int depth , unsigned int offset ) const { return ( cornerIndex(depth,offset) + cornerIndex(depth,offset+1) )/2+1; }
		unsigned int       index( unsigned int depth , unsigned int offset , HyperCube::Direction dir ) const
		{
			return dir==HyperCube::CROSS ? edgeIndex( depth , offset ) : cornerIndex( depth , offset + ( dir==HyperCube::BACK ? 0 : 1 ) );
		}

		std::string to_string( Key< Dim > key ) const
		{
			std::stringstream stream;
			stream << "(";
			for( unsigned int d=0 ; d<Dim ; d++ )
			{
				if( d ) stream << ",";
				if( key[d]&1 )	// If it's odd, it will be an edge
				{
					unsigned int depth = _maxDepth;
					key[d]>>=1;
					while( !(key[d]&1) ) depth-- , key[d]>>=1;
					stream << "[" << (((key[d]-1)/2)<<(_maxDepth-depth)) << "," << (((key[d]+1)/2)<<(_maxDepth-depth)) << "]";
				}
				else stream << (key[d]>>2);
			}
			stream << ")/" << (1<<_maxDepth);
			return stream.str();
		}

		int maxDepth( void ) const { return _maxDepth; }

	protected:
		int _maxDepth;
	};

	template< unsigned int D , unsigned int K >
	unsigned int HyperCubeTables< D , K >::CellOffset[ ElementNum ][ IncidentCubeNum ];
	template< unsigned int D , unsigned int K >
	unsigned int HyperCubeTables< D , K >::IncidentElementCoIndex[ ElementNum ][ IncidentCubeNum ];
	template< unsigned int D , unsigned int K >
	unsigned int HyperCubeTables< D , K >::IncidentElementIndex[ ElementNum ][ IncidentCubeNum ];
	template< unsigned int D , unsigned int K >
	unsigned int HyperCubeTables< D , K >::CellOffsetAntipodal[ ElementNum ];
	template< unsigned int D , unsigned int K >
	typename HyperCube::Cube< D >::template IncidentCubeIndex < K > HyperCubeTables< D , K >::IncidentCube[ ElementNum ];
	template< unsigned int D , unsigned int K >
	typename HyperCube::Direction HyperCubeTables< D , K >::Directions[ ElementNum ][ D ];
	template< unsigned int D , unsigned int K1 , unsigned int K2 >
	typename HyperCube::Cube< D >::template Element< K2 > HyperCubeTables< D , K1 , K2 >::OverlapElements[ ElementNum1 ][ OverlapElementNum ];
	template< unsigned int D , unsigned int K1 , unsigned int K2 >
	bool HyperCubeTables< D , K1 , K2 >::Overlap[ ElementNum1 ][ ElementNum2 ];
}
