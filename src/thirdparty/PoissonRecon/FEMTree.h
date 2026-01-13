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

// -- [TODO] Make as many of the functions (related to the solver) const as possible.
// -- [TODO] Move the point interpolation constraint scaling by 1<<maxDepth
// -- [TODO] Add support for staggered-grid test functions
// -- [TODO] Store signatures with constraints/systems/restriction-prolongations
// -- [TODO] Make a virtual evaluation that only needs to know the degree
// -- [TODO] Modify (public) functions so that template parameters don't need to be passed when they are called
// -- [TODO] Confirm that whenever _isValidFEM*Node is called, the flags have already been set.
// -- [TODO] Make weight evaluation more efficient in _getSamplesPerNode by reducing the number of calls to getNeighbors
// -- [TODO] For point evaluation:
//        1. Have the evaluator store stencils for all depths [DONE]
//        2. When testing centers/corners, don't use generic evaluation
// -- [TODO] Support nested parallelism with thread pools
// -- [TODO] Make the node flags protected
// -- [TODO] Identify members that are only valid after finalization
// -- [TODO] Force the MaxDegree and finite-element degrees into the template parameters for the FEMTree so that root vs space-root are set up on construction

#ifndef FEM_TREE_INCLUDED
#define FEM_TREE_INCLUDED

#include <atomic>
#ifdef SANITIZED_PR
#include <shared_mutex>
#endif // SANITIZED_PR
#include "MyMiscellany.h"
#include "BSplineData.h"
#include "Geometry.h"
#include "DataStream.h"
#include "RegularTree.h"
#include "SparseMatrix.h"
#include "NestedVector.h"
#include "Rasterizer.h"
#include <limits>
#include <functional>
#include <string>
#include <tuple>
#include <functional>
#include <cmath>
#include <climits>
#include "MarchingCubes.h"
#include <sstream>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include "MAT.h"

namespace PoissonRecon
{

#ifdef BIG_DATA
	// The integer type used for indexing the nodes in the octree
	typedef long long node_index_type;
	// The integer type used for indexing the entries of the matrix
	typedef int matrix_index_type;
#else // !BIG_DATA
	typedef int node_index_type;
	typedef int matrix_index_type;
#endif // BIG_DATA
#ifdef USE_DEEP_TREE_NODES
	// The integer type used for storing the depth and offset within an octree node
	typedef unsigned int depth_and_offset_type;
#else // !USE_DEEP_TREE_NODES
	typedef unsigned short depth_and_offset_type;
#endif // USE_DEEP_TREE_NODES

	template< unsigned int Dim , class Real > class FEMTree;

	enum
	{
		SHOW_GLOBAL_RESIDUAL_NONE ,
		SHOW_GLOBAL_RESIDUAL_LAST ,
		SHOW_GLOBAL_RESIDUAL_ALL  ,
		SHOW_GLOBAL_RESIDUAL_COUNT
	};
	static const char* ShowGlobalResidualNames[] = { "show none" , "show last" , "show all" };

	class FEMTreeNodeData
	{
	public:
		enum
		{
			SPACE_FLAG               = 1 << 0 , // Part of the partition of the unit cube
			FEM_FLAG_1               = 1 << 1 ,	// Indexes a valid finite element
			FEM_FLAG_2               = 1 << 2 ,	// Indexes a valid finite element
			DIRICHLET_NODE_FLAG      = 1 << 3 ,	// The finite elements should evaluate to zero on this node
			DIRICHLET_ELEMENT_FLAG   = 1 << 4 ,	// Coefficient of this node should be locked to zero
			GEOMETRY_SUPPORTED_FLAG  = 1 << 5 ,	// The finite element defined by this node has support overlapping geometry constraints
			GHOST_FLAG               = 1 << 6 ,	// Children are pruned out
			SCRATCH_FLAG             = 1 << 7 ,
		};
		node_index_type nodeIndex;
#ifdef SANITIZED_PR
		mutable std::atomic< unsigned char > flags;
#else // !SANITIZED_PR
		mutable char flags;
#endif // SANITIZED_PR
		void setGhostFlag( bool f ) const { if( f ) flags |= GHOST_FLAG ; else flags &= ~GHOST_FLAG; }
		bool getGhostFlag( void ) const { return ( flags & GHOST_FLAG )!=0; }
		void setDirichletNodeFlag( bool f ) const { if( f ) flags |= DIRICHLET_NODE_FLAG ; else flags &= ~DIRICHLET_NODE_FLAG; }
		bool getDirichletNodeFlag( void ) const { return ( flags & DIRICHLET_NODE_FLAG )!=0; }
		void setDirichletElementFlag( bool f ) const { if( f ) flags |= DIRICHLET_ELEMENT_FLAG ; else flags &= ~DIRICHLET_ELEMENT_FLAG; }
		bool getDirichletElementFlag( void ) const { return ( flags & DIRICHLET_ELEMENT_FLAG )!=0; }
		void setScratchFlag( bool f ) const { if( f ) flags |= SCRATCH_FLAG ; else flags &= (unsigned char)~SCRATCH_FLAG; }
		bool getScratchFlag( void ) const { return ( flags & SCRATCH_FLAG )!=0; }
		inline FEMTreeNodeData( void );
		inline ~FEMTreeNodeData( void );
#ifdef SANITIZED_PR
		inline FEMTreeNodeData &operator = ( const FEMTreeNodeData &data );
#endif // SANITIZED_PR
	};

	template< unsigned int Dim >
	class SortedTreeNodes
	{
		typedef RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > TreeNode;
	protected:
		Pointer( Pointer( node_index_type ) ) _sliceStart;
		int _levels;

		void _set( TreeNode& root );
	public:
		Pointer( TreeNode* ) treeNodes;
		node_index_type begin( int depth ) const { return _sliceStart[depth][0]; }
		node_index_type   end( int depth ) const { return _sliceStart[depth][(size_t)1<<depth]; }
		node_index_type begin( int depth , int slice ) const { return _sliceStart[depth][ slice<0 ? 0 : ( slice>(1<<depth) ? (1<<depth) : slice ) ]; }
		node_index_type   end( int depth , int slice ) const { return begin( depth , slice+1 ); }
		size_t size( void ) const { return _levels ? _sliceStart[_levels-1][(size_t)1<<(_levels-1)] : 0; }
		size_t size( int depth ) const
		{
			if( depth<0 || depth>=_levels ) MK_THROW( "bad depth: 0 <= " , depth , " < " , _levels );
			return _sliceStart[depth][(size_t)1<<depth] - _sliceStart[depth][0];
		}
		size_t size( int depth , int slice ) const { return end( depth , slice ) - begin( depth , slice ); }
		int levels( void ) const { return _levels; }

		SortedTreeNodes( void );
		~SortedTreeNodes( void );
		// Resets the sorted tree nodes and sets map[i] to the index previously stored with the i-th node.
		void reset( TreeNode& root , std::vector< node_index_type > &map );
		void set( TreeNode& root );

		void write( BinaryStream &stream ) const;
		void read( BinaryStream &stream , TreeNode &root );
	};

	template< typename T > struct DotFunctor{};
	template< > struct DotFunctor< float >
	{
		double operator()( float  v1 , float  v2 ){ return v1*v2; }
		unsigned int dimension( void ) const { return 1; }
	};
	template< > struct DotFunctor< double >
	{
		double operator()( double v1 , double v2 ){ return v1*v2; }
		unsigned int dimension( void ) const { return 1; }
	};
	template< class Real , unsigned int Dim > struct DotFunctor< Point< Real , Dim > >
	{
		double operator()( Point< Real , Dim > v1 , Point< Real , Dim > v2 ){ return Point< Real , Dim >::Dot( v1 , v2 ); }
		unsigned int dimension( void ) const { return Dim; }
	};

	template< typename Pack > struct SupportKey{ };
	template< unsigned int ... Degrees >
	struct SupportKey< UIntPack< Degrees ... > > : public RegularTreeNode< sizeof...(Degrees) , FEMTreeNodeData , depth_and_offset_type >::template NeighborKey< UIntPack< (-BSplineSupportSizes< Degrees >::SupportStart) ... > , UIntPack< BSplineSupportSizes< Degrees >::SupportEnd ... > >
	{
		typedef UIntPack< (-BSplineSupportSizes< Degrees >::SupportStart ) ... > LeftRadii;
		typedef UIntPack< ( BSplineSupportSizes< Degrees >::SupportEnd   ) ... > RightRadii;
		typedef UIntPack< ( BSplineSupportSizes< Degrees >::SupportSize  ) ... > Sizes; 
	};

	template< typename Pack > struct ConstSupportKey{ };
	template< unsigned int ... Degrees >
	struct ConstSupportKey< UIntPack< Degrees ... > > : public RegularTreeNode< sizeof...(Degrees) , FEMTreeNodeData , depth_and_offset_type >::template ConstNeighborKey< UIntPack< (-BSplineSupportSizes< Degrees >::SupportStart ) ... > , UIntPack< BSplineSupportSizes< Degrees >::SupportEnd ... > >
	{
		typedef UIntPack< (-BSplineSupportSizes< Degrees >::SupportStart ) ... > LeftRadii;
		typedef UIntPack< ( BSplineSupportSizes< Degrees >::SupportEnd   ) ... > RightRadii;
		typedef UIntPack< ( BSplineSupportSizes< Degrees >::SupportSize  ) ... > Sizes; 
	};

	template< typename Pack > struct OverlapKey{ };
	template< unsigned int ... Degrees >
	struct OverlapKey< UIntPack< Degrees ... > > : public RegularTreeNode< sizeof...(Degrees) , FEMTreeNodeData , depth_and_offset_type >::template NeighborKey< UIntPack< (-BSplineOverlapSizes< Degrees , Degrees >::OverlapStart ) ... > , UIntPack< BSplineOverlapSizes< Degrees , Degrees >::OverlapEnd ... > >
	{
		typedef UIntPack< (-BSplineOverlapSizes< Degrees , Degrees >::OverlapStart ) ... > LeftRadii;
		typedef UIntPack< ( BSplineOverlapSizes< Degrees , Degrees >::OverlapEnd   ) ... > RightRadii;
		typedef UIntPack< ( BSplineOverlapSizes< Degrees , Degrees >::OverlapSize  ) ... > Sizes; 
	};

	template< typename Pack > struct ConstOverlapKey{ };
	template< unsigned int ... Degrees >
	struct ConstOverlapKey< UIntPack< Degrees ... > > : public RegularTreeNode< sizeof...(Degrees) , FEMTreeNodeData , depth_and_offset_type >::template ConstNeighborKey< UIntPack< (-BSplineOverlapSizes< Degrees , Degrees >::OverlapStart ) ... > , UIntPack< BSplineOverlapSizes< Degrees , Degrees >::OverlapEnd ... > >
	{
		typedef UIntPack< (-BSplineOverlapSizes< Degrees , Degrees >::OverlapStart ) ... > LeftRadii;
		typedef UIntPack< ( BSplineOverlapSizes< Degrees , Degrees >::OverlapEnd   ) ... > RightRadii;
		typedef UIntPack< ( BSplineOverlapSizes< Degrees , Degrees >::OverlapSize  ) ... > Sizes; 
	};

	template< typename Pack > struct PointSupportKey{ };
	template< unsigned int ... Degrees >
	struct PointSupportKey< UIntPack< Degrees ... > > : public RegularTreeNode< sizeof...(Degrees) , FEMTreeNodeData , depth_and_offset_type >::template NeighborKey< UIntPack< BSplineSupportSizes< Degrees >::SupportEnd ... > , UIntPack< (-BSplineSupportSizes< Degrees >::SupportStart ) ... > >
	{
		typedef UIntPack< ( BSplineSupportSizes< Degrees >::SupportEnd   ) ... > LeftRadii;
		typedef UIntPack< (-BSplineSupportSizes< Degrees >::SupportStart ) ... > RightRadii;
		typedef UIntPack< ( BSplineSupportSizes< Degrees >::SupportEnd - BSplineSupportSizes< Degrees >::SupportStart + 1 ) ... > Sizes; 
	};

	template< typename Pack > struct ConstPointSupportKey{ };
	template< unsigned int ... Degrees >
	struct ConstPointSupportKey< UIntPack< Degrees ... > > : public RegularTreeNode< sizeof...(Degrees) , FEMTreeNodeData , depth_and_offset_type >::template ConstNeighborKey< UIntPack< BSplineSupportSizes< Degrees >::SupportEnd ... > , UIntPack< (-BSplineSupportSizes< Degrees >::SupportStart ) ... > >
	{
		typedef UIntPack< ( BSplineSupportSizes< Degrees >::SupportEnd   ) ... > LeftRadii;
		typedef UIntPack< (-BSplineSupportSizes< Degrees >::SupportStart ) ... > RightRadii;
		typedef UIntPack< ( BSplineSupportSizes< Degrees >::SupportEnd - BSplineSupportSizes< Degrees >::SupportStart + 1 ) ... > Sizes; 
	};

	template< typename Pack > struct CornerSupportKey{ };
	template< unsigned int ... Degrees >
	struct CornerSupportKey< UIntPack< Degrees ... > > : public RegularTreeNode< sizeof...(Degrees) , FEMTreeNodeData , depth_and_offset_type >::template NeighborKey< UIntPack< BSplineSupportSizes< Degrees >::BCornerEnd ... > , UIntPack< ( -BSplineSupportSizes< Degrees >::BCornerStart + 1 ) ... > >
	{
		typedef UIntPack< ( BSplineSupportSizes< Degrees >::BCornerEnd       ) ... > LeftRadii;
		typedef UIntPack< (-BSplineSupportSizes< Degrees >::BCornerStart + 1 ) ... > RightRadii;
		typedef UIntPack< ( BSplineSupportSizes< Degrees >::BCornerSize  + 1 ) ... > Sizes; 
	};

	template< typename Pack > struct ConstCornerSupportKey{ };
	template< unsigned int ... Degrees >
	struct ConstCornerSupportKey< UIntPack< Degrees ... > > : public RegularTreeNode< sizeof...(Degrees) , FEMTreeNodeData , depth_and_offset_type >::template ConstNeighborKey< UIntPack< BSplineSupportSizes< Degrees >::BCornerEnd ... > , UIntPack< ( -BSplineSupportSizes< Degrees >::BCornerStart + 1 ) ... > >
	{
		typedef UIntPack< ( BSplineSupportSizes< Degrees >::BCornerEnd       ) ... > LeftRadii;
		typedef UIntPack< (-BSplineSupportSizes< Degrees >::BCornerStart + 1 ) ... > RightRadii;
		typedef UIntPack< ( BSplineSupportSizes< Degrees >::BCornerSize  + 1 ) ... > Sizes; 
	};


	template< class Data , typename Pack > struct _SparseOrDenseNodeData{};
	template< class Data , unsigned int ... FEMSigs >
	struct _SparseOrDenseNodeData< Data , UIntPack< FEMSigs ... > >
	{
		static const unsigned int Dim = sizeof ... ( FEMSigs );
		typedef UIntPack< FEMSigs ... > FEMSignatures;
		typedef Data data_type;

		// Methods for accessing as an array
		virtual size_t size( void ) const = 0;
		virtual const Data& operator[] ( size_t idx ) const = 0;
		virtual Data& operator[] ( size_t idx ) = 0;

		// Method for accessing (and inserting if necessary) using a node
		virtual Data& operator[]( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > *node ) = 0;
		// Methods for accessing using a node
		virtual Data* operator()( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > *node ) = 0;
		virtual const Data* operator()( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >* node ) const = 0;

		// Method for getting the actual index associated with a node
		virtual node_index_type index( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >* node ) const = 0;
	};

	template< class Data , typename Pack > struct SparseNodeData{};
	template< class Data , unsigned int ... FEMSigs >
	struct SparseNodeData< Data , UIntPack< FEMSigs ... > > : public _SparseOrDenseNodeData< Data , UIntPack< FEMSigs ... > >
	{
		static const unsigned int Dim = sizeof ... ( FEMSigs );

		static void WriteSignatures( BinaryStream &stream )
		{
			unsigned int dim = sizeof ... ( FEMSigs );
			stream.write( dim );
			unsigned int femSigs[] = { FEMSigs ... };
			stream.write( femSigs , dim );
		}
		void write( BinaryStream &stream , const Serializer< Data > &serializer ) const
		{
			_indices.write( stream );
			_data.write( stream , serializer );
		}
		void read( BinaryStream &stream , const Serializer< Data > &serializer )
		{
			_indices.read( stream );
			_data.read( stream , serializer );
		}
		void write( BinaryStream &stream ) const
		{
			_indices.write( stream );
			_data.write( stream );
		}
		void read( BinaryStream &stream )
		{
			_indices.read( stream );
			_data.read( stream );
		}
		SparseNodeData( void ){}
		SparseNodeData( BinaryStream &stream ){ read(stream); }
		SparseNodeData( BinaryStream &stream , const Serializer< Data > &serializer ){ read(stream,serializer); }
		// [WARNING] Default constructing the mutex
		SparseNodeData( const SparseNodeData &d ) : _indices(d._indices) , _data(d._data){};
		SparseNodeData( SparseNodeData &&d ) : _indices(d._indices) , _data(d._data){};
		// [WARNING] Not copying the mutex
		SparseNodeData &operator = ( const SparseNodeData &d ){ _indices = d._indices , _data = d._data ; return *this; }
		SparseNodeData &operator = ( SparseNodeData &&d ){ std::swap( _indices , d._indices ) , std::swap( _data , d._data ); return *this; }

		size_t size( void ) const { return _data.size(); }
		const Data& operator[] ( size_t idx ) const { return _data[idx]; }
		Data& operator[] ( size_t idx ) { return _data[idx]; }

		void reserve( size_t sz ){ if( sz>_indices.size() ) _indices.resize( sz , -1 ); }
		size_t reserved( void ) const { return _indices.size(); } 
		Data* operator()( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >* node ){ return ( node->nodeData.nodeIndex<0 || node->nodeData.nodeIndex>=(node_index_type)_indices.size() || _indices[ node->nodeData.nodeIndex ]==-1 ) ? NULL : &_data[ _indices[ node->nodeData.nodeIndex ] ]; }
		const Data* operator()( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >* node ) const { return ( node->nodeData.nodeIndex<0 || node->nodeData.nodeIndex>=(node_index_type)_indices.size() || _indices[ node->nodeData.nodeIndex ]==-1 ) ? NULL : &_data[ _indices[ node->nodeData.nodeIndex ] ]; }

		Data& operator[]( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >* node ){ return at( node ); }
		Data &at( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > *node , Data zero=Data() )
		{
			_indices.resize( node->nodeData.nodeIndex+1 , -1 );

			// If the node hasn't been allocated yet
#ifdef SANITIZED_PR
			volatile node_index_type *indexPtr = &_indices[ node->nodeData.nodeIndex ];
			node_index_type _index = ReadAtomic( *indexPtr );
#else // !SANITIZED_PR
			volatile node_index_type &_index = _indices[ node->nodeData.nodeIndex ];
#endif // SANITIZED_PR
			if( _index==-1 )
			{
				std::lock_guard< std::mutex > lock( _updateMutex );
#ifdef SANITIZED_PR
				_index = ReadAtomic( *indexPtr );
#endif // SANITIZED_PR
				if( _index==-1 )
				{
					size_t sz = _data.size();
					_data.resize( sz+1 , zero );
#ifdef SANITIZED_PR
					_index = (node_index_type)sz;
					SetAtomic( *indexPtr , _index );
#else // !SANITIZED_PR
					*(node_index_type*)&_index = (node_index_type)sz;
#endif // SANITIZED_PR
				}
			}
			return _data[ _index ];
		}
		node_index_type index( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > *node ) const
		{
			if( !node || node->nodeData.nodeIndex<0 || node->nodeData.nodeIndex>=(node_index_type)_indices.size() ) return -1;
			else return _indices[ node->nodeData.nodeIndex ];
		}

		node_index_type index( node_index_type idx ) const
		{
			if( idx<0 || idx>=(node_index_type)_indices.size() ) return -1;
			else return _indices[ idx ];
		}

		void merge( const SparseNodeData &data ){ return merge( data , []( const Data &data ){ return data; } ); }

		template< typename MergeFunctor >
		void merge( const SparseNodeData &data , const MergeFunctor &mergeFunctor )
		{
			size_t sz = _indices.size();
			node_index_type newDataCount = 0;
			for( unsigned int j=0 ; j<data._indices.size() ; j++ ) if( data._indices[j]!=-1 && data._indices[j]<(node_index_type)sz && _indices[j]==-1 ) newDataCount++;
			size_t oldSize = _data.size();
			_data.resize( oldSize + newDataCount );
			newDataCount = 0;
			for( unsigned int j=0 ; j<data._indices.size() ; j++ ) if( data._indices[j]!=-1 && data._indices[j]<(node_index_type)sz )
				if( _indices[j]==-1 )
				{
					_indices[j] = (node_index_type)oldSize + newDataCount;
					_data[ oldSize + newDataCount ] = mergeFunctor( data._data[ data._indices[j] ] );
					newDataCount++;
				}
				else _data[ _indices[j] ] += mergeFunctor( data._data[ data._indices[j] ] );
		}

		template< typename TargetToSourceFunctor >
		void mergeFromTarget( const SparseNodeData &data , const TargetToSourceFunctor &targetToSourceFunctor ){ return mergeFromTarget( data , targetToSourceFunctor , []( const Data &data ){ return data; } ); }

		template< typename TargetToSourceFunctor , typename MergeFunctor >
		void mergeFromTarget( const SparseNodeData &target , const TargetToSourceFunctor &targetToSourceFunctor , const MergeFunctor &mergeFunctor )
		{
			size_t sz = _indices.size();
			node_index_type newDataCount = 0;
			for( unsigned int j=0 ; j<target._indices.size() ; j++ ) if( target._indices[j]!=-1 && target._indices[j]<(node_index_type)sz && _indices[ targetToSourceFunctor(j) ]==-1 ) newDataCount++;
			size_t oldSize = _data.size();
			_data.resize( oldSize + newDataCount );
			newDataCount = 0;
			for( unsigned int j=0 ; j<target._indices.size() ; j++ ) if( target._indices[j]!=-1 && target._indices[j]<(node_index_type)sz )
			{
				node_index_type _j = targetToSourceFunctor( j );
				if( _indices[_j]==-1 )
				{
					_indices[_j] = (node_index_type)oldSize + newDataCount;
					_data[ oldSize + newDataCount ] = mergeFunctor( target._data[ target._indices[j] ] );
					newDataCount++;
				}
				else _data[ _indices[_j] ] += mergeFunctor( target._data[ target._indices[j] ] );
			}
		}

		template< typename SourceToTargetFunctor >
		void mergeToSource( const SparseNodeData &data , const SourceToTargetFunctor &sourceToTargetFunctor ){ return mergeToSource( data , sourceToTargetFunctor , []( const Data &data ){ return data; } ); }

		template< typename SourceToTargetFunctor , typename MergeFunctor >
		void mergeToSource( const SparseNodeData &target , const SourceToTargetFunctor &sourceToTargetFunctor , const MergeFunctor &mergeFunctor )
		{
			size_t _sz = target._indices.size();
			node_index_type newDataCount = 0;
			for( unsigned int j=0 ; j<_indices.size() ; j++ ) if( _indices[j]==-1 )
			{
				unsigned int _j = sourceToTargetFunctor( j );
				if( _j<(node_index_type)_sz && target._indices[_j]!=-1 ) newDataCount++;
			}
			size_t oldSize = _data.size();
			_data.resize( oldSize + newDataCount );
			newDataCount = 0;
			for( unsigned int j=0 ; j<_indices.size() ; j++ )
			{
				unsigned int _j = sourceToTargetFunctor( j );
				if( _j<(node_index_type)_sz && target._indices[_j]!=-1 )
					if( _indices[j]==-1 )
					{
						_indices[j] = (node_index_type)oldSize + newDataCount;
						_data[ oldSize + newDataCount ] = mergeFunctor( target._data[ target._indices[_j] ] );
						newDataCount++;
					}
					else _data[ _indices[j] ] += mergeFunctor( target._data[ target._indices[_j] ] );
			}
		}

	protected:
		template< unsigned int _Dim , class _Real > friend class FEMTree;

		// Map should be the size of the new number of entries and map[i] should give the old index of the i-th node
		void _remapIndices( ConstPointer( node_index_type )oldNodeIndices , size_t newNodeCount )
		{
			NestedVector< node_index_type , NESTED_VECTOR_LEVELS > newIndices;
			newIndices.resize( newNodeCount );
			for( node_index_type i=0 ; i<(node_index_type)newNodeCount ; i++ )
			{
				newIndices[i] = -1;
				if( oldNodeIndices[i]!=-1 && oldNodeIndices[i]<(node_index_type)_indices.size() ) newIndices[i] = _indices[ oldNodeIndices[i] ];
			}
			std::swap( _indices , newIndices );
		}

		SparseNodeData _trim( node_index_type endIndex ) const
		{
			size_t dataCount = 0;
			for( node_index_type i=0 ; i<endIndex ; i++ ) if( _indices[i]!=-1 ) dataCount++;
			SparseNodeData sparseNodeData;
			sparseNodeData._indices.resize( endIndex );
			sparseNodeData._data.resize( dataCount );
			node_index_type idx = 0;
			for( node_index_type i=0 ; i<endIndex ; i++ )
				if( _indices[i]!=-1 )
				{
					sparseNodeData._indices[i] = idx;
					sparseNodeData._data[idx] = _data[ _indices[i] ];
					idx++;
				}
				else sparseNodeData._indices[i] = -1;
			return sparseNodeData;
		}

		std::mutex _updateMutex;
		NestedVector< node_index_type , NESTED_VECTOR_LEVELS > _indices;
		NestedVector< Data , NESTED_VECTOR_LEVELS > _data;
	};

	template< class Data , typename Pack > struct DenseNodeData{};
	template< class Data , unsigned int ... FEMSigs >
	struct DenseNodeData< Data , UIntPack< FEMSigs ... > > : public _SparseOrDenseNodeData< Data , UIntPack< FEMSigs ... > >
	{
		static const unsigned int Dim = sizeof ... ( FEMSigs );
		DenseNodeData( void ) { _data = NullPointer( Data ) ; _sz = 0; }
		DenseNodeData( size_t sz ){ _sz = sz ; if( sz ) _data = NewPointer< Data >( sz ) ; else _data = NullPointer( Data ); }
		DenseNodeData( const DenseNodeData&  d ) : DenseNodeData() { _resize( d._sz ) ; if( _sz ) memcpy( _data , d._data , sizeof(Data) * _sz ); }
		DenseNodeData(       DenseNodeData&& d ){ _data = d._data , _sz = d._sz ; d._data = NullPointer( Data ) , d._sz = 0; }
		DenseNodeData& operator = ( const DenseNodeData&  d ){ _resize( d._sz ) ; if( _sz ) memcpy( _data , d._data , sizeof(Data) * _sz ) ; return *this; }
		DenseNodeData& operator = (       DenseNodeData&& d ){ size_t __sz = _sz ; Pointer( Data ) __data = _data ; _data = d._data , _sz = d._sz ; d._data = __data , d._sz = __sz ; return *this; }
		DenseNodeData( BinaryStream &stream ) : DenseNodeData() { read(stream); }
		~DenseNodeData( void ){ DeletePointer( _data ) ; _sz = 0; }
		void resize( size_t sz ){ DeletePointer( _data ) ; _sz = sz ; if( sz ) _data = NewPointer< Data >( sz ) ; else _data = NullPointer( Data ); }
		static void WriteSignatures( BinaryStream &stream )
		{
			unsigned int dim = sizeof ... ( FEMSigs );
			stream.write( dim );
			unsigned int femSigs[] = { FEMSigs ... };
			stream.write( GetPointer( femSigs , dim ) , dim );
		}
		void write( BinaryStream &stream ) const
		{
			stream.write( _sz );
			stream.write( _data , _sz );
		}
		void read( BinaryStream &stream )
		{
			if( !stream.read( _sz ) ) MK_THROW( "Failed to read size" );
			_data = NewPointer< Data >( _sz );
			if( !stream.read( _data , _sz ) ) MK_THROW( "failed to read data" );
		}

		Data& operator[] ( size_t idx ) { return _data[idx]; }
		const Data& operator[] ( size_t idx ) const { return _data[idx]; }
		size_t size( void ) const { return _sz; }
		Data& operator[]( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >* node ) { return _data[ node->nodeData.nodeIndex ]; }
		Data* operator()( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >* node ) { return ( node==NULL || node->nodeData.nodeIndex>=(node_index_type)_sz ) ? NULL : &_data[ node->nodeData.nodeIndex ]; }
		const Data* operator()( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >* node ) const { return ( node==NULL || node->nodeData.nodeIndex>=(node_index_type)_sz ) ? NULL : &_data[ node->nodeData.nodeIndex ]; }
		node_index_type index( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >* node ) const { return ( !node || node->nodeData.nodeIndex<0 || node->nodeData.nodeIndex>=(node_index_type)_sz ) ? -1 : node->nodeData.nodeIndex; }
		Pointer( Data ) operator()( void ) { return _data; }
		ConstPointer( Data ) operator()( void ) const { return ( ConstPointer( Data ) )_data; }
	protected:
		template< unsigned int _Dim , class _Real > friend class FEMTree;

		// Map should be the size of the new number of entries and map[i] should give the old index of the i-th node
		void _remapIndices( ConstPointer( node_index_type )oldNodeIndices , size_t newNodeCount )
		{
			Pointer( Data ) newData = NewPointer< Data >( newNodeCount );
			memset( newData , 0 , sizeof(Data)*newNodeCount );
			for( size_t i=0 ; i<newNodeCount ; i++ ) if( oldNodeIndices[i]>=0 && oldNodeIndices[i]<(node_index_type)_sz ) newData[i] = _data[ oldNodeIndices[i] ];
			DeletePointer( _data );
			_data = newData;
			_sz = newNodeCount;
		}

		DenseNodeData _trim( node_index_type endIndex ) const
		{
			DenseNodeData denseNodeData;
			denseNodeData._sz = endIndex;
			denseNodeData._data = NewPointer< Data >( endIndex );
			memcpy( denseNodeData._data , _data , sizeof(Data) * denseNodeData._sz );
			return denseNodeData;
		}

		size_t _sz;
		void _resize( size_t sz ){ DeletePointer( _data ) ; if( sz ) _data = NewPointer< Data >( sz ) ; else _data = NullPointer( Data ) ; _sz = sz; }
		Pointer( Data ) _data;
	};

	enum FEMTreeRealType
	{
		FEM_TREE_REAL_FLOAT ,
		FEM_TREE_REAL_DOUBLE ,
		FEM_TREE_REAL_COUNT
	};
	static const char *FEMTreeRealNames[] = { "float" , "double" };

	inline void ReadFEMTreeParameter( BinaryStream &stream , FEMTreeRealType& realType , unsigned int &dimension )
	{
		if( !stream.read( realType ) ) MK_THROW( "Failed to read real type" );
		if( !stream.read( dimension ) ) MK_THROW( "Failed to read dimension" );
	}

	inline unsigned int* ReadDenseNodeDataSignatures( BinaryStream &stream , unsigned int &dim )
	{
		if( !stream.read( dim ) ) MK_THROW( "Failed to read dimension" );
		unsigned int* femSigs = new unsigned int[dim];
		if( !stream.read( GetPointer( femSigs , dim ) , dim ) ) MK_THROW( "Failed to read signatures" );
		return femSigs;
	}

	// The Derivative method needs static members:
	//		Dim: the dimensionality of the space in which derivatives are evaluated
	//		Size: the total number of derivatives
	// and static methods:
	//		Index: takes the number of partials along each dimension and returns the index
	//		Factor: takes an index and sets the number of partials along each dimension

	template< typename T > struct TensorDerivatives{ };
	template< class Real , typename T > struct TensorDerivativeValues{ };

	// Specify the derivatives for each dimension separately
	template< unsigned int D , unsigned int ... Ds >
	struct TensorDerivatives< UIntPack< D , Ds ... > >
	{
		typedef TensorDerivatives< UIntPack< Ds ... > > _TensorDerivatives;
		static const unsigned int LastDerivative = UIntPack< D , Ds ... >::template Get< sizeof ... (Ds) >();
		static const unsigned int Dim = _TensorDerivatives::Dim + 1;
		static const unsigned int Size = _TensorDerivatives::Size * ( D+1 );
		static void Factor( unsigned int idx , unsigned int derivatives[Dim] ){ derivatives[0] = idx / _TensorDerivatives::Size ; _TensorDerivatives::Factor( idx % _TensorDerivatives::Size , derivatives+1 ); }
		static unsigned int Index( const unsigned int derivatives[Dim] ){ return _TensorDerivatives::Index( derivatives + 1 ) + _TensorDerivatives::Size * derivatives[0]; }
	};
	template< unsigned int D >
	struct TensorDerivatives< UIntPack< D > >
	{
		static const unsigned int LastDerivative = D;
		static const unsigned int Dim = 1;
		static const unsigned int Size = D+1;
		static void Factor( unsigned int idx , unsigned int derivatives[1] ){ derivatives[0] = idx; }
		static unsigned int Index( const unsigned int derivatives[1] ){ return derivatives[0]; }
	};
	template< class Real , unsigned int ... Ds > struct TensorDerivativeValues< Real , UIntPack< Ds ... > > : public Point< Real , TensorDerivatives< UIntPack< Ds ... > >::Size >{ };

	// Specify the sum of the derivatives
	template< unsigned int Dim , unsigned int D >
	struct CumulativeDerivatives
	{
		typedef CumulativeDerivatives< Dim , D-1 > _CumulativeDerivatives;
		static const unsigned int LastDerivative = D;
		static const unsigned int Size = _CumulativeDerivatives::Size * Dim + 1;
		static void Factor( unsigned int idx , unsigned int d[Dim] )
		{
			if( idx<_CumulativeDerivatives::Size ) return _CumulativeDerivatives::Factor( idx , d );
			else _Factor( idx - _CumulativeDerivatives::Size , d );
		}
		static unsigned int Index( const unsigned int derivatives[Dim] )
		{
			unsigned int dCount = 0;
			for( unsigned int d=0 ; d<Dim ; d++ ) dCount += derivatives[d];
			if( dCount>=D ) MK_THROW( "More derivatives than allowed" );
			else if( dCount<D ) return _CumulativeDerivatives::Index( derivatives );
			else                return _CumulativeDerivatives::Size + _Index( derivatives );
		}
	protected:
		static const unsigned int _Size = _CumulativeDerivatives::_Size * Dim;
		static void _Factor( unsigned int idx , unsigned int d[Dim] )
		{
			_CumulativeDerivatives::_Factor( idx % _CumulativeDerivatives::_Size , d );
			d[ idx / _CumulativeDerivatives::_Size ]++;
		}
		static unsigned int _Index( const unsigned int d[Dim] )
		{
			unsigned int _d[Dim];
			memcpy( _d , d , sizeof(_d) );
			for( unsigned int i=0 ; i<Dim ; i++ ) if( _d[i] )
			{
				_d[i]--;
				return _CumulativeDerivatives::Index( _d ) * Dim + i;
			}
			MK_THROW( "No derivatives specified" );
			return -1;
		}
		friend CumulativeDerivatives< Dim , D+1 >;
	};
	template< unsigned int Dim >
	struct CumulativeDerivatives< Dim , 0 >
	{
		static const unsigned int LastDerivative = 0;
		static const unsigned int Size = 1;
		static void Factor( unsigned int idx , unsigned int d[Dim] ){ memset( d , 0 , sizeof(unsigned int)*Dim ); }
		static unsigned int Index( const unsigned int derivatives[Dim] ){ return 0; }
	protected:
		static const unsigned int _Size = 1;
		static void _Factor( unsigned int idx , unsigned int d[Dim] ){ memset( d , 0 , sizeof(unsigned int)*Dim ); }
		friend CumulativeDerivatives< Dim , 1 >;
	};
	template< typename Real , unsigned int Dim , unsigned int D > using CumulativeDerivativeValues = Point< Real , CumulativeDerivatives< Dim , D >::Size >;


	template< unsigned int Dim , class Real , unsigned int D >
	CumulativeDerivativeValues< Real , Dim , D > Evaluate( const double dValues[Dim][D+1] )
	{
		CumulativeDerivativeValues< Real , Dim , D > v;
		unsigned int _d[Dim];
		for( unsigned int d=0 ; d<CumulativeDerivatives< Dim , D >::Size ; d++ )
		{
			CumulativeDerivatives< Dim , D >::Factor( d , _d );
			double value = dValues[0][ _d[0] ];
			for( unsigned int dd=1 ; dd<Dim ; dd++ ) value *= dValues[dd][ _d[dd] ];
			v[d] = (Real)value;
		}
		return v;
	}

	template< unsigned int Dim , class Real , typename T , unsigned int D >
	struct DualPointInfo
	{
		Point< Real , Dim > position;
		Real weight;
		CumulativeDerivativeValues< T , Dim , D > dualValues;
		DualPointInfo  operator +  ( const DualPointInfo& p ) const { return DualPointInfo( position + p.position , dualValues + p.dualValues , weight + p.weight ); }
		DualPointInfo& operator += ( const DualPointInfo& p ){ position += p.position ; weight += p.weight , dualValues += p.dualValues ; return *this; }
		DualPointInfo  operator *  ( Real s ) const { return DualPointInfo( position*s , weight*s , dualValues*s ); }
		DualPointInfo& operator *= ( Real s ){ position *= s , weight *= s , dualValues *= s ; return *this; }
		DualPointInfo  operator /  ( Real s ) const { return DualPointInfo( position/s , weight/s , dualValues/s ); }
		DualPointInfo& operator /= ( Real s ){ position /= s , weight /= s , dualValues /= s ; return *this; }
		DualPointInfo( void ) : weight(0) { }
		DualPointInfo( Point< Real , Dim > p , CumulativeDerivativeValues< T , Dim , D > c , Real w ) { position = p , dualValues = c , weight = w; }
	};
	template< unsigned int Dim , class Real , typename Data , typename T , unsigned int D >
	struct DualPointAndDataInfo
	{
		DualPointInfo< Dim , Real , T , D > pointInfo;
		Data data;
		DualPointAndDataInfo  operator +  ( const DualPointAndDataInfo& p ) const { return DualPointAndDataInfo( pointInfo + p.pointInfo , data + p.data ); }
		DualPointAndDataInfo  operator *  ( Real s )                        const { return DualPointAndDataInfo( pointInfo * s , data * s ); }
		DualPointAndDataInfo  operator /  ( Real s )                        const { return DualPointAndDataInfo( pointInfo / s , data / s ); }
		DualPointAndDataInfo& operator += ( const DualPointAndDataInfo& p ){ pointInfo += p.pointInfo ; data += p.data ; return *this; }
		DualPointAndDataInfo& operator *= ( Real s )                       { pointInfo *= s , data *= s ; return *this; }
		DualPointAndDataInfo& operator /= ( Real s )                       { pointInfo /= s , data /= s ; return *this; }
		DualPointAndDataInfo( void ){ }
		DualPointAndDataInfo( DualPointInfo< Dim , Real , T , D > p , Data d ) { pointInfo = p , data = d; }
	};
	template< unsigned int Dim , class Real , typename T , unsigned int D >
	struct DualPointInfoBrood
	{
		DualPointInfo< Dim , Real , T , D >& operator[]( size_t idx ){ return _dpInfo[idx]; }
		const DualPointInfo< Dim , Real , T , D >& operator[]( size_t idx ) const { return _dpInfo[idx]; }
		void finalize( void ){ _size = 0 ; for( unsigned int i=0 ; i<(1<<Dim) ; i++ ) if( _dpInfo[i].weight>0 ) _dpInfo[_size++] = _dpInfo[i]; }
		unsigned int size( void ) const { return _size; }

		DualPointInfoBrood  operator +  ( const DualPointInfoBrood& p ) const { DualPointInfoBrood d ; for( unsigned int i=0 ; i<(1<<Dim) ; i++ ) d._dpInfo[i] = _dpInfo[i] + p._dpInfo[i] ;  return d; }
		DualPointInfoBrood  operator *  ( Real s )                      const { DualPointInfoBrood d ; for( unsigned int i=0 ; i<(1<<Dim) ; i++ ) d._dpInfo[i] = _dpInfo[i] * s            ;  return d; }
		DualPointInfoBrood  operator /  ( Real s )                      const { DualPointInfoBrood d ; for( unsigned int i=0 ; i<(1<<Dim) ; i++ ) d._dpInfo[i] = _dpInfo[i] / s            ;  return d; }
		DualPointInfoBrood& operator += ( const DualPointInfoBrood& p ){ for( unsigned int i=0 ; i<(1<<Dim) ; i++ ) _dpInfo[i] += p._dpInfo[i] ; return *this; }
		DualPointInfoBrood& operator *= ( Real s )                     { for( unsigned int i=0 ; i<(1<<Dim) ; i++ ) _dpInfo[i] *= s            ; return *this; }
		DualPointInfoBrood& operator /= ( Real s )                     { for( unsigned int i=0 ; i<(1<<Dim) ; i++ ) _dpInfo[i] /= s            ; return *this; }
	protected:
		DualPointInfo< Dim , Real , T , D > _dpInfo[1<<Dim];
		unsigned int _size;
	};
	template< unsigned int Dim , class Real , typename Data , typename T , unsigned int D >
	struct DualPointAndDataInfoBrood
	{
		DualPointAndDataInfo< Dim , Real , Data , T , D >& operator[]( size_t idx ){ return _dpInfo[idx]; }
		const DualPointAndDataInfo< Dim , Real , Data , T , D >& operator[]( size_t idx ) const { return _dpInfo[idx]; }
		void finalize( void ){ _size = 0 ; for( unsigned int i=0 ; i<(1<<Dim) ; i++ ) if( _dpInfo[i].pointInfo.weight>0 ) _dpInfo[_size++] = _dpInfo[i]; }
		unsigned int size( void ) const { return _size; }

		DualPointAndDataInfoBrood  operator +  ( const DualPointAndDataInfoBrood& p ) const { DualPointAndDataInfoBrood d ; for( unsigned int i=0 ; i<(1<<Dim) ; i++ ) d._dpInfo[i] = _dpInfo[i] + p._dpInfo[i] ;  return d; }
		DualPointAndDataInfoBrood  operator *  ( Real s )                             const { DualPointAndDataInfoBrood d ; for( unsigned int i=0 ; i<(1<<Dim) ; i++ ) d._dpInfo[i] = _dpInfo[i] * s            ;  return d; }
		DualPointAndDataInfoBrood  operator /  ( Real s )                             const { DualPointAndDataInfoBrood d ; for( unsigned int i=0 ; i<(1<<Dim) ; i++ ) d._dpInfo[i] = _dpInfo[i] / s            ;  return d; }
		DualPointAndDataInfoBrood& operator += ( const DualPointAndDataInfoBrood& p ){ for( unsigned int i=0 ; i<(1<<Dim) ; i++ ) _dpInfo[i] += p._dpInfo[i] ; return *this; }
		DualPointAndDataInfoBrood& operator *= ( Real s )                            { for( unsigned int i=0 ; i<(1<<Dim) ; i++ ) _dpInfo[i] *= s ; return *this; }
		DualPointAndDataInfoBrood& operator /= ( Real s )                            { for( unsigned int i=0 ; i<(1<<Dim) ; i++ ) _dpInfo[i] /= s ; return *this; }
	protected:
		DualPointAndDataInfo< Dim , Real , Data , T , D > _dpInfo[1<<Dim];
		unsigned int _size;
	};


	////////////////////////////
	// The virtual integrator //
	////////////////////////////
	struct BaseFEMIntegrator
	{
		template< typename TDegreePack                                            > struct                  System{};
		template< typename TDegreePack                                            > struct RestrictionProlongation{};
		template< typename TDegreePack , typename CDegreePack , unsigned int CDim > struct              Constraint{};
		template< typename TDegreePack                                            > struct        SystemConstraint{};
		template< typename TDegreePack                                            > struct          PointEvaluator{};

	protected:
		template< unsigned int Degree , unsigned int ... Degrees >
		static typename std::enable_if< sizeof ... ( Degrees )==0 , bool >::type _IsSupported( UIntPack< Degree , Degrees ... > , unsigned int femDepth , const int femOffset[] , unsigned int spaceDepth , const int spaceOffset[] )
		{
			int femRes = 1<<femDepth , spaceRes = 1<<spaceDepth;
			int   femBegin = ( 0 +   femOffset[0] + BSplineSupportSizes< Degree >::SupportStart ) * spaceRes;
			int   femEnd   = ( 1 +   femOffset[0] + BSplineSupportSizes< Degree >::SupportEnd   ) * spaceRes; 
			int spaceBegin = ( 0 + spaceOffset[0] + BSplineSupportSizes< 0      >::SupportStart ) *   femRes;
			int spaceEnd   = ( 1 + spaceOffset[0] + BSplineSupportSizes< 0      >::SupportEnd   ) *   femRes;
			return spaceBegin<femEnd && spaceEnd>femBegin;
		}
		template< unsigned int Degree , unsigned int ... Degrees >
		static typename std::enable_if< sizeof ... ( Degrees )!=0 , bool >::type _IsSupported( UIntPack< Degree , Degrees ... > , unsigned int femDepth , const int femOffset[] , unsigned int spaceDepth , const int spaceOffset[] )
		{
			int femRes = 1<<femDepth , spaceRes = 1<<spaceDepth;
			int   femBegin = ( 0 +   femOffset[0] + BSplineSupportSizes< Degree >::SupportStart ) * spaceRes;
			int   femEnd   = ( 1 +   femOffset[0] + BSplineSupportSizes< Degree >::SupportEnd   ) * spaceRes; 
			int spaceBegin = ( 0 + spaceOffset[0] + BSplineSupportSizes< 0      >::SupportStart ) *   femRes;
			int spaceEnd   = ( 1 + spaceOffset[0] + BSplineSupportSizes< 0      >::SupportEnd   ) *   femRes;
			return ( spaceBegin<femEnd && spaceEnd>femBegin ) && _IsSupported( UIntPack< Degrees ... >() , femDepth , femOffset+1 , spaceDepth , spaceOffset+1 );
		}
		template< unsigned int Degree , unsigned int ... Degrees >
		static typename std::enable_if< sizeof ... ( Degrees )==0 , bool >::type _IsInteriorlySupported( UIntPack< Degree , Degrees ... > , unsigned int depth , const int off[] )
		{
			int begin , end;
			BSplineSupportSizes< Degree >::InteriorSupportedSpan( depth , begin , end );
			return off[0]>=begin && off[0]<end;
		}
		template< unsigned int Degree , unsigned int ... Degrees >
		static typename std::enable_if< sizeof ... ( Degrees )!=0 , bool >::type _IsInteriorlySupported( UIntPack< Degree , Degrees ... > , unsigned int depth , const int off[] )
		{
			int begin , end;
			BSplineSupportSizes< Degree >::InteriorSupportedSpan( depth , begin , end );
			return ( off[0]>=begin && off[0]<end ) && _IsInteriorlySupported( UIntPack< Degrees ... >() , depth , off+1 );
		}
		template< unsigned int Degree , unsigned int ... Degrees >
		static typename std::enable_if< sizeof ... ( Degrees )==0 , bool >::type _IsInteriorlySupported( UIntPack< Degree , Degrees ... > , unsigned int depth , const int off[] , const double begin[] , const double end[] )
		{
			int res = 1<<depth;
			double b = ( 0. + off[0] + BSplineSupportSizes< Degree >::SupportStart ) / res;
			double e = ( 1. + off[0] + BSplineSupportSizes< Degree >::SupportEnd   ) / res; 
			return b>=begin[0] && e<=end[0];
		}
		template< unsigned int Degree , unsigned int ... Degrees >
		static typename std::enable_if< sizeof ... ( Degrees )!=0 , bool >::type _IsInteriorlySupported( UIntPack< Degree , Degrees ... > , unsigned int depth , const int off[] , const double begin[] , const double end[] )
		{
			int res = 1<<depth;
			double b = ( 0. + off[0] + BSplineSupportSizes< Degree >::SupportStart ) / res;
			double e = ( 1. + off[0] + BSplineSupportSizes< Degree >::SupportEnd   ) / res; 
			return b>=begin[0] && e<=end[0] && _IsInteriorlySupported( UIntPack< Degrees ... >() , depth , off+1 , begin+1 , end+1 );
		}
		template< unsigned int Degree1 , unsigned int ... Degrees1 , unsigned int Degree2 , unsigned int ... Degrees2 >
		static typename std::enable_if< sizeof ... ( Degrees1 )==0 >::type _InteriorOverlappedSpan( UIntPack< Degree1 , Degrees1 ... > , UIntPack< Degree2 , Degrees2 ... > , int depth , int begin[] , int end[] )
		{
			BSplineIntegrationData< FEMDegreeAndBType< Degree1 , BOUNDARY_NEUMANN >::Signature , FEMDegreeAndBType< Degree2 , BOUNDARY_NEUMANN >::Signature >::InteriorOverlappedSpan( depth , begin[0] , end[0] );
		}
		template< unsigned int Degree1 , unsigned int ... Degrees1 , unsigned int Degree2 , unsigned int ... Degrees2 >
		static typename std::enable_if< sizeof ... ( Degrees1 )!=0 >::type _InteriorOverlappedSpan( UIntPack< Degree1 , Degrees1 ... > , UIntPack< Degree2 , Degrees2 ... > , int depth , int begin[] , int end[] )
		{
			BSplineIntegrationData< FEMDegreeAndBType< Degree1 , BOUNDARY_NEUMANN >::Signature , FEMDegreeAndBType< Degree2 , BOUNDARY_NEUMANN >::Signature >::InteriorOverlappedSpan( depth , begin[0] , end[0] );
			_InteriorOverlappedSpan( UIntPack< Degrees1 ... >() , UIntPack< Degrees2 ... >() , depth , begin+1 , end+1 );
		}
		template< unsigned int Degree1 , unsigned int ... Degrees1 , unsigned int Degree2 , unsigned int ... Degrees2 >
		static typename std::enable_if< sizeof ... ( Degrees1 )==0 , bool >::type _IsInteriorlyOverlapped( UIntPack< Degree1 , Degrees1 ... > , UIntPack< Degree2 , Degrees2 ... > , unsigned int depth , const int off[] )
		{
			int begin , end;
			BSplineIntegrationData< FEMDegreeAndBType< Degree1 , BOUNDARY_NEUMANN >::Signature , FEMDegreeAndBType< Degree2 , BOUNDARY_NEUMANN >::Signature >::InteriorOverlappedSpan( depth , begin , end );
			return off[0]>= begin && off[0]<end;
		}
		template< unsigned int Degree1 , unsigned int ... Degrees1 , unsigned int Degree2 , unsigned int ... Degrees2 >
		static typename std::enable_if< sizeof ... ( Degrees1 )!=0 , bool >::type _IsInteriorlyOverlapped( UIntPack< Degree1 , Degrees1 ... > , UIntPack< Degree2 , Degrees2 ... > , unsigned int depth , const int off[] )
		{
			int begin , end;
			BSplineIntegrationData< FEMDegreeAndBType< Degree1 , BOUNDARY_NEUMANN >::Signature , FEMDegreeAndBType< Degree2 , BOUNDARY_NEUMANN >::Signature >::InteriorOverlappedSpan( depth , begin , end );
			return ( off[0]>= begin && off[0]<end ) && _IsInteriorlyOverlapped( UIntPack< Degrees1 ... >() , UIntPack< Degrees2 ... >() , depth , off+1 );
		}
		template< unsigned int Degree1 , unsigned int ... Degrees1 , unsigned int Degree2 , unsigned int ... Degrees2 >
		static typename std::enable_if< sizeof ... ( Degrees1 )==0 >::type _ParentOverlapBounds( UIntPack< Degree1 , Degrees1 ... > , UIntPack< Degree2 , Degrees2 ... > , unsigned int depth , const int off[] , int start[] , int end[] )
		{
			const int OverlapStart = BSplineOverlapSizes< Degree1 , Degree2 >::OverlapStart;
			start[0] = BSplineOverlapSizes< Degree1 , Degree2 >::ParentOverlapStart[ off[0] & 1 ] - OverlapStart;
			end  [0] = BSplineOverlapSizes< Degree1 , Degree2 >::ParentOverlapEnd  [ off[0] & 1 ] - OverlapStart + 1;
		}
		template< unsigned int Degree1 , unsigned int ... Degrees1 , unsigned int Degree2 , unsigned int ... Degrees2 >
		static typename std::enable_if< sizeof ... ( Degrees1 )!=0 >::type _ParentOverlapBounds( UIntPack< Degree1 , Degrees1 ... > , UIntPack< Degree2 , Degrees2 ... > , unsigned int depth , const int off[] , int start[] , int end[] )
		{
			const int OverlapStart = BSplineOverlapSizes< Degree1 , Degree2 >::OverlapStart;
			start[0] = BSplineOverlapSizes< Degree1 , Degree2 >::ParentOverlapStart[ off[0] & 1 ] - OverlapStart;
			end  [0] = BSplineOverlapSizes< Degree1 , Degree2 >::ParentOverlapEnd  [ off[0] & 1 ] - OverlapStart + 1;
			_ParentOverlapBounds( UIntPack< Degrees1 ... >() , UIntPack< Degrees2 ... >() , depth , off+1 , start+1 , end+1 );
		}
		template< unsigned int Degree1 , unsigned int ... Degrees1 , unsigned int Degree2 , unsigned int ... Degrees2 >
		static typename std::enable_if< sizeof ... ( Degrees1 )==0 >::type _ParentOverlapBounds( UIntPack< Degree1 , Degrees1 ... > , UIntPack< Degree2 , Degrees2 ... > , int corner , int start[] , int end[] )
		{
			const int OverlapStart = BSplineOverlapSizes< Degree1 , Degree2 >::OverlapStart;
			start[0] = BSplineOverlapSizes< Degree1 , Degree2 >::ParentOverlapStart[ corner & 1 ] - OverlapStart;
			end  [0] = BSplineOverlapSizes< Degree1 , Degree2 >::ParentOverlapEnd  [ corner & 1 ] - OverlapStart + 1;
		}
		template< unsigned int Degree1 , unsigned int ... Degrees1 , unsigned int Degree2 , unsigned int ... Degrees2 >
		static typename std::enable_if< sizeof ... ( Degrees1 )!=0 >::type _ParentOverlapBounds( UIntPack< Degree1 , Degrees1 ... > , UIntPack< Degree2 , Degrees2 ... > , int corner , int start[] , int end[] )
		{
			const int OverlapStart = BSplineOverlapSizes< Degree1 , Degree2 >::OverlapStart;
			start[0] = BSplineOverlapSizes< Degree1 , Degree2 >::ParentOverlapStart[ corner & 1 ] - OverlapStart;
			end  [0] = BSplineOverlapSizes< Degree1 , Degree2 >::ParentOverlapEnd  [ corner & 1 ] - OverlapStart + 1;
			_ParentOverlapBounds( UIntPack< Degrees1 ... >() , UIntPack< Degrees2 ... >() , corner>>1 , start+1 , end+1 );
		}

	public:
		template< unsigned int ... Degrees >
		static bool IsSupported( UIntPack< Degrees ... > , int femDepth , const int femOffset[] , int spaceDepth , const int spaceOffset[] ){ return femDepth>=0 && spaceDepth>=0 && _IsSupported( UIntPack< Degrees ... >() , femDepth , femOffset , spaceDepth , spaceOffset ); }
		template< unsigned int ... Degrees >
		static bool IsInteriorlySupported( UIntPack< Degrees ... > , int depth , const int offset[] ){ return depth>=0 && _IsInteriorlySupported( UIntPack< Degrees ... >() , depth , offset ); }
		template< unsigned int ... Degrees >
		static bool IsInteriorlySupported( UIntPack< Degrees ... > , int depth , const int offset[] , const double begin[] , const double end[] ){ return depth>=0 && _IsInteriorlySupported( UIntPack< Degrees ... >() , depth , offset , begin , end ); }

		template< unsigned int ... Degrees1 , unsigned int ... Degrees2 >
		static void InteriorOverlappedSpan( UIntPack< Degrees1 ... > , UIntPack< Degrees2 ... > , int depth , int begin[] , int end[] )
		{
			static_assert( sizeof ... ( Degrees1 ) == sizeof ... ( Degrees2 ) , "[ERROR] Dimensions don't match" );
			_InteriorOverlappedSpan( UIntPack< Degrees1 ... >() , UIntPack< Degrees2 ... >() , depth , begin , end );
		}
		template< unsigned int ... Degrees1 , unsigned int ... Degrees2 >
		static bool IsInteriorlyOverlapped( UIntPack< Degrees1 ... > , UIntPack< Degrees2 ... > , int depth , const int offset[] )
		{
			static_assert( sizeof ... ( Degrees1 ) == sizeof ... ( Degrees2 ) , "[ERROR] Dimensions don't match" );
			return depth>=0 && _IsInteriorlyOverlapped( UIntPack< Degrees1 ... >() , UIntPack< Degrees2 ... >() , depth , offset );
		}

		template< unsigned int ... Degrees1 , unsigned int ... Degrees2 >
		static void ParentOverlapBounds( UIntPack< Degrees1 ... > , UIntPack< Degrees2 ... > , int depth , const int offset[] , int start[] , int end[] )
		{
			static_assert( sizeof ... ( Degrees1 ) == sizeof ... ( Degrees2 ) , "[ERROR] Dimensions don't match" );
			if( depth>0 ) _ParentOverlapBounds( UIntPack< Degrees1 ... >() , UIntPack< Degrees2 ... >() , depth , offset , start , end );
		}
		template< unsigned int ... Degrees1 , unsigned int ... Degrees2 >
		static void ParentOverlapBounds( UIntPack< Degrees1 ... > , UIntPack< Degrees2 ... > , int corner , int start[] , int end[] )
		{
			static_assert( sizeof ... ( Degrees1 ) == sizeof ... ( Degrees2 ) , "[ERROR] Dimensions don't match" );
			_ParentOverlapBounds( UIntPack< Degrees1 ... >() , UIntPack< Degrees2 ... >() , corner , start , end );
		}

		template< unsigned int Dim >
		struct PointEvaluatorState
		{
			virtual double value( const int offset[] , const unsigned int d[] ) const = 0;
			virtual double subValue( const int offset[] , const unsigned int d[] ) const = 0;
			template< class Real , typename DerivativeType >
			Point< Real , DerivativeType::Size > dValues( const int offset[] ) const
			{
				Point< Real , DerivativeType::Size > v;
				unsigned int _d[Dim];
				for( int d=0 ; d<DerivativeType::Size ; d++ )
				{
					DerivativeType::Factor( d , _d );
					v[d] = (Real)value( offset , _d );
				}
				return v;
			}
			template< class Real , typename DerivativeType >
			Point< Real , DerivativeType::LastDerivative+1 > partialDotDValues( Point< Real , DerivativeType::Size > v , const int offset[] ) const
			{
				Point< Real , DerivativeType::LastDerivative+1 > dot;
				unsigned int _d[Dim];
				for( int d=0 ; d<DerivativeType::Size ; d++ )
				{
					DerivativeType::Factor( d , _d );
					dot[ _d[Dim-1] ] += (Real)( subValue( offset , _d ) * v[d] );
				}
				return dot;
			}
		};

		template< unsigned int ... TDegrees >
		struct PointEvaluator< UIntPack< TDegrees ... > >
		{
			static const unsigned int Dim = sizeof ... ( TDegrees );
		};

		template< unsigned int ... TDegrees >
		struct RestrictionProlongation< UIntPack< TDegrees ... > >
		{
			virtual void init( void ){ }
			virtual double upSampleCoefficient( const int pOff[] , const int cOff[] ) const = 0;

			typedef DynamicWindow< double , UIntPack< ( - BSplineSupportSizes< TDegrees >::DownSample0Start + BSplineSupportSizes< TDegrees >::DownSample1End + 1 ) ... > > DownSampleStencil;
			struct   UpSampleStencil  : public DynamicWindow< double , UIntPack< BSplineSupportSizes< TDegrees >::UpSampleSize ... > > { };
			struct DownSampleStencils : public DynamicWindow< DownSampleStencil , IsotropicUIntPack< sizeof ... ( TDegrees ) , 2 > > { };

			void init( int highDepth ){ _highDepth = highDepth ; init(); }
			void setStencil (   UpSampleStencil & stencil  ) const;
			void setStencils( DownSampleStencils& stencils ) const;
			int highDepth( void ) const { return _highDepth; }

		protected:
			int _highDepth;
		};


		template< unsigned int ... TDegrees >
		struct System< UIntPack< TDegrees ... > >
		{
			virtual void init( void ){ }
			virtual double ccIntegrate( const int off1[] , const int off2[] ) const = 0;
			virtual double pcIntegrate( const int off1[] , const int off2[] ) const = 0;
			virtual bool vanishesOnConstants( void ) const { return false; }
			virtual RestrictionProlongation< UIntPack< TDegrees ... > >& restrictionProlongation( void ) = 0;

			struct CCStencil : public DynamicWindow< double , UIntPack< BSplineOverlapSizes< TDegrees , TDegrees >::OverlapSize ... > >{ };
#ifdef SHOW_WARNINGS
#pragma message ( "[WARNING] Why are the parent/child stencils so big?" )
#endif // SHOW_WARNINGS
			struct PCStencils : public DynamicWindow< CCStencil , IsotropicUIntPack< sizeof ... ( TDegrees ) , 2 > >{ };

			void init( int highDepth ){ _highDepth = highDepth ; init(); }
			template< bool IterateFirst > void setStencil ( CCStencil & stencil  ) const;
			template< bool IterateFirst > void setStencils( PCStencils& stencils ) const;
			int highDepth( void ) const { return _highDepth; }

		protected:
			int _highDepth;
		};

		template< unsigned int ... TDegrees , unsigned int ... CDegrees , unsigned int CDim >
		struct Constraint< UIntPack< TDegrees ... > , UIntPack< CDegrees ... > , CDim >
		{
			static_assert( sizeof...(TDegrees)==sizeof...(CDegrees) , "[ERROR] BaseFEMIntegrator::Constraint: Test and constraint dimensions don't match" );

			virtual void init( void ){ ; }
			virtual Point< double , CDim > ccIntegrate( const int off1[] , const int off2[] ) const = 0;
			virtual Point< double , CDim > pcIntegrate( const int off1[] , const int off2[] ) const = 0;
			virtual Point< double , CDim > cpIntegrate( const int off1[] , const int off2[] ) const = 0;
			virtual RestrictionProlongation< UIntPack< TDegrees ... > >& tRestrictionProlongation( void ) = 0;
			virtual RestrictionProlongation< UIntPack< CDegrees ... > >& cRestrictionProlongation( void ) = 0;

			struct CCStencil : public DynamicWindow< Point< double , CDim > , UIntPack< BSplineOverlapSizes< TDegrees , CDegrees >::OverlapSize ... > >{ };
#ifdef SHOW_WARNINGS
#pragma message ( "[WARNING] Why are the parent/child stencils so big?" )
#endif // SHOW_WARNINGS
			struct PCStencils : public DynamicWindow< CCStencil , IsotropicUIntPack< sizeof ... ( TDegrees ) , 2 > >{ };
			struct CPStencils : public DynamicWindow< CCStencil , IsotropicUIntPack< sizeof ... ( TDegrees ) , 2 > >{ };

			void init( int highDepth ){ _highDepth = highDepth ; init(); }
			template< bool IterateFirst > void setStencil ( CCStencil & stencil  ) const;
			template< bool IterateFirst > void setStencils( PCStencils& stencils ) const;
			template< bool IterateFirst > void setStencils( CPStencils& stencils ) const;
			int highDepth( void ) const { return _highDepth; }

		protected:
			int _highDepth;
		};

		template< unsigned int ... TDegrees >
		struct SystemConstraint< UIntPack< TDegrees ... > > :  public Constraint< UIntPack< TDegrees ... > , UIntPack< TDegrees ... > , 1 >
		{
			typedef  Constraint< UIntPack< TDegrees ... > , UIntPack< TDegrees ... > , 1 > Base;
			SystemConstraint( System< UIntPack< TDegrees ... > >& sys ) : _sys( sys ){;}
			void init( void ){ _sys.init( Base::highDepth() ) ; _sys.init(); }
			Point< double , 1 > ccIntegrate( const int off1[] , const int off2[] ) const{ return Point< double , 1 >( _sys.ccIntegrate( off1 , off2 ) ); }
			Point< double , 1 > pcIntegrate( const int off1[] , const int off2[] ) const{ return Point< double , 1 >( _sys.pcIntegrate( off1 , off2 ) ); }
			Point< double , 1 > cpIntegrate( const int off1[] , const int off2[] ) const{ return Point< double , 1 >( _sys.pcIntegrate( off2 , off1 ) ); }
			RestrictionProlongation< UIntPack< TDegrees ... > >& tRestrictionProlongation( void ){ return _sys.restrictionProlongation(); }
			RestrictionProlongation< UIntPack< TDegrees ... > >& cRestrictionProlongation( void ){ return _sys.restrictionProlongation(); }
		protected:
			System< UIntPack< TDegrees ... > >& _sys;
		};
	};

	/////////////////////////////////////////////////
	// An implementation of the virtual integrator //
	/////////////////////////////////////////////////
	struct FEMIntegrator
	{
	protected:
		template< unsigned int FEMSig , unsigned int ... FEMSigs >
		static typename std::enable_if< sizeof ... ( FEMSigs )==0 , bool >::type _IsValidFEMNode( UIntPack< FEMSig , FEMSigs ... > , unsigned int depth , const int offset[] )
		{
			return !BSplineEvaluationData< FEMSig >::OutOfBounds( depth , offset[0] );
		}
		template< unsigned int FEMSig , unsigned int ... FEMSigs >
		static typename std::enable_if< sizeof ... ( FEMSigs )!=0 , bool >::type _IsValidFEMNode( UIntPack< FEMSig , FEMSigs ... > , unsigned int depth , const int offset[] )
		{
			return !BSplineEvaluationData< FEMSig >::OutOfBounds( depth , offset[0] ) && _IsValidFEMNode( UIntPack< FEMSigs ... >() , depth , offset+1 );
		}
		template< unsigned int FEMSig , unsigned ... FEMSigs >
		static typename std::enable_if< sizeof ... ( FEMSigs )==0 , bool >::type _IsOutOfBounds( UIntPack< FEMSig , FEMSigs ... > , unsigned int depth , const int offset[] )
		{
			return BSplineEvaluationData< FEMSig >::OutOfBounds( depth , offset[0] );
		}
		template< unsigned int FEMSig , unsigned ... FEMSigs >
		static typename std::enable_if< sizeof ... ( FEMSigs )!=0 , bool >::type _IsOutOfBounds( UIntPack< FEMSig , FEMSigs ... > , unsigned int depth , const int offset[] )
		{
			return BSplineEvaluationData< FEMSig >::OutOfBounds( depth , offset[0] ) || _IsOutOfBounds( UIntPack< FEMSigs ... >() , depth , offset+1 );
		}
		template< unsigned int FEMSig , unsigned int ... FEMSigs >
		static typename std::enable_if< sizeof ... ( FEMSigs )==0 >::type _BSplineBegin( UIntPack< FEMSig , FEMSigs ... > , unsigned int depth , int begin[] )
		{
			begin[0] = BSplineEvaluationData< FEMSig >::Begin( depth );
		}
		template< unsigned int FEMSig , unsigned int ... FEMSigs >
		static typename std::enable_if< sizeof ... ( FEMSigs )!=0 >::type _BSplineBegin( UIntPack< FEMSig , FEMSigs ... > , unsigned int depth , int begin[] )
		{
			begin[0] = BSplineEvaluationData< FEMSig >::Begin( depth ) ; _BSplineBegin( UIntPack< FEMSigs ... >() , depth , begin+1 );
		}
		template< unsigned int FEMSig , unsigned int ... FEMSigs >
		static typename std::enable_if< sizeof ... ( FEMSigs )==0 >::type _BSplineEnd( UIntPack< FEMSig , FEMSigs ... > , unsigned int depth , int end[] )
		{
			end[0] = BSplineEvaluationData< FEMSig >::End( depth );
		}
		template< unsigned int FEMSig , unsigned int ... FEMSigs >
		static typename std::enable_if< sizeof ... ( FEMSigs )!=0 >::type _BSplineEnd( UIntPack< FEMSig , FEMSigs ... > , unsigned int depth , int end[] )
		{
			end[0] = BSplineEvaluationData< FEMSig >::End( depth ) ; _BSplineEnd( UIntPack< FEMSigs ... >() , depth , end+1 );
		}
		template< unsigned int FEMSig , unsigned int ... FEMSigs >
		static typename std::enable_if< sizeof ... ( FEMSigs )==0 , double >::type _Integral( UIntPack< FEMSig , FEMSigs ... > , unsigned int depth , const int offset[] , const double begin[] , const double end[] )
		{
			return BSplineEvaluationData< FEMSig >::Integral( depth , offset[0] , begin[0] , end[0] , 0 );
		}
		template< unsigned int FEMSig , unsigned int ... FEMSigs >
		static typename std::enable_if< sizeof ... ( FEMSigs )!=0 , double >::type _Integral( UIntPack< FEMSig , FEMSigs ... > , unsigned int depth , const int offset[] , const double begin[] , const double end[] )
		{
			return BSplineEvaluationData< FEMSig >::Integral( depth , offset[0] , begin[0] , end[0] , 0 ) * _Integral( UIntPack< FEMSigs ... >() , depth , offset+1 , begin+1 , end+1 );
		}
	public:
		template< unsigned int ... FEMSigs >
		static double Integral( UIntPack< FEMSigs ... > , int depth , const int offset[] , const double begin[] , const double end[] )
		{
			if( depth<0 ) return 0;
			else return _Integral( UIntPack< FEMSigs ... >() , depth , offset , begin , end );
		}
		template< unsigned int ... FEMSigs > static bool IsValidFEMNode( UIntPack< FEMSigs ... > , int depth , const int offset[] ){ return _IsValidFEMNode( UIntPack< FEMSigs ... >() , depth , offset ); }
		template< unsigned int ... FEMSigs > static bool IsOutOfBounds( UIntPack< FEMSigs ... > , int depth , const int offset[] ){ return depth<0 || _IsOutOfBounds( UIntPack< FEMSigs ... >() , depth , offset ); }
		template< unsigned int ... FEMSigs > static void BSplineBegin( UIntPack< FEMSigs ... > , int depth , int begin[] ){ if( depth>=0 ) _BSplineBegin( UIntPack< FEMSigs ... >() , depth , begin ); }
		template< unsigned int ... FEMSigs > static void BSplineEnd  ( UIntPack< FEMSigs ... > , int depth , int end  [] ){ if( depth>=0 ) _BSplineEnd  ( UIntPack< FEMSigs ... >() , depth , end   ); }

		template< typename TSignatures , typename TDerivatives                                                                    > struct                  System{};
		template< typename TSignatures , typename TDerivatives , typename CSignatures , typename CDerivatives , unsigned int CDim > struct              Constraint{};
		template< typename TSignatures , typename TDerivatives , typename CSignatures , typename CDerivatives                     > struct        ScalarConstraint{};
		template< typename TSignatures                                                                                            > struct RestrictionProlongation{};
		template< typename TSignatures , typename TDerivatives                                                                    > struct          PointEvaluator{};
		template< typename TSignatures , typename TDerivatives                                                                    > struct     PointEvaluatorState{};

		template< unsigned int ... TSignatures , unsigned int ... TDs >
		struct PointEvaluatorState< UIntPack< TSignatures ... > , UIntPack< TDs ... > > : public BaseFEMIntegrator::template PointEvaluatorState< sizeof ... ( TSignatures ) >
		{
			static_assert( sizeof...(TSignatures)==sizeof...(TDs) , "[ERROR] Degree and derivative dimensions don't match" );
			static_assert( ParameterPack::Comparison< UIntPack< FEMSignature< TSignatures >::Degree ... > , UIntPack< TDs ... > >::GreaterThanOrEqual , "[ERROR] PointEvaluatorState: More derivatives than degrees" );

			static const unsigned int Dim = sizeof...(TSignatures);

			double value   ( const int offset[] , const unsigned int derivatives[] ) const { return _value< Dim   >( offset , derivatives ); }
			double subValue( const int offset[] , const unsigned int derivatives[] ) const { return _value< Dim-1 >( offset , derivatives ); }
			// Bypassing the "auto" keyword 
			template< unsigned int _Dim >
			const double (*(values)( void ) const )[ UIntPack< TDs ... >::template Get< _Dim >()+1 ] { return std::template get< _Dim >( _oneDValues ).values; }
		protected:
			int _pointOffset[Dim];

			template< unsigned int Degree , unsigned int D > struct _OneDValues
			{
				double values[ BSplineSupportSizes< Degree >::SupportSize ][ D+1 ];
				double value( int dOff , unsigned int d ) const
				{
					if( dOff>=-BSplineSupportSizes< Degree >::SupportEnd && dOff<=-BSplineSupportSizes< Degree >::SupportStart && d<=D ) return values[ dOff+BSplineSupportSizes< Degree >::SupportEnd][d];
					else return 0;
				}
			};
			std::tuple< _OneDValues< FEMSignature< TSignatures >::Degree , TDs > ... > _oneDValues;
			template< unsigned int MaxDim=Dim , unsigned int I=0 > typename std::enable_if< I==MaxDim , double >::type _value( const int off[] , const unsigned int d[] ) const { return 1.; }
			template< unsigned int MaxDim=Dim , unsigned int I=0 > typename std::enable_if< I!=MaxDim , double >::type _value( const int off[] , const unsigned int d[] ) const { return std::get< I >( _oneDValues ).value( off[I]-_pointOffset[I] , d[I] ) * _value< MaxDim , I+1 >( off , d ); }
			template< typename T1 , typename T2 > friend struct PointEvaluator;
		};

		template< unsigned int ... TSignatures , unsigned int ... TDs >
		struct PointEvaluator< UIntPack< TSignatures ... > , UIntPack< TDs ... > > : public BaseFEMIntegrator::template PointEvaluator< UIntPack< FEMSignature< TSignatures >::Degree ... > >
		{
			static_assert( sizeof...(TSignatures)==sizeof...(TDs) , "[ERROR] PointEvaluator: Degree and derivative dimensions don't match" );
			static_assert( ParameterPack::Comparison< UIntPack< FEMSignature< TSignatures >::Degree ... > , UIntPack< TDs ... > >::GreaterThanOrEqual , "[ERROR] PointEvaluator: More derivatives than degrees" );

			static const unsigned int Dim = sizeof ... ( TSignatures );

			typedef typename BaseFEMIntegrator::template PointEvaluator< UIntPack< FEMSignature< TSignatures >::Degree ... > > Base;

			PointEvaluator( unsigned int maxDepth ) : _maxDepth( maxDepth ) { _init(); }
			template< unsigned int ... EDs >
			void initEvaluationState( Point< double , Dim > p , unsigned int depth , PointEvaluatorState< UIntPack< TSignatures ... > , UIntPack< EDs ... > >& state ) const
			{
				unsigned int res = 1<<depth;
				for( int d=0 ; d<Dim ; d++ ) state._pointOffset[d] = (int)( p[d] * res );
				initEvaluationState( p , depth , state._pointOffset , state );
			}
			template< unsigned int ... EDs >
			void initEvaluationState( Point< double , Dim > p , unsigned int depth , const int* offset , PointEvaluatorState< UIntPack< TSignatures ... > , UIntPack< EDs ... > >& state ) const
			{
				static_assert( ParameterPack::Comparison< UIntPack< TDs ... > , UIntPack< EDs ... > >::GreaterThanOrEqual , "[ERROR] PointEvaluator::init: More evaluation derivatives than stored derivatives" );
				for( int d=0 ; d<Dim ; d++ ) state._pointOffset[d] = (int)offset[d];
				_initEvaluationState( UIntPack< TSignatures ... >() , UIntPack< EDs ... >() , &p[0] , depth , state );
			}
		protected:
			unsigned int _maxDepth;
			std::tuple< BSplineData< TSignatures , TDs > ... > _bSplineData;
			template< unsigned int I=0 > typename std::enable_if< I==Dim >::type _init( void ){}
			template< unsigned int I=0 > typename std::enable_if< (I<Dim) >::type _init( void ){ std::get< I >( _bSplineData ).reset( _maxDepth ) ; _init< I+1 >( ); }

			template< unsigned int I , unsigned int TSig , unsigned int D , typename State >
			void _setEvaluationState( const double* p , unsigned int depth , State& state ) const
			{
				static const int       LeftSupportRadius = -BSplineSupportSizes< FEMSignature< TSig >::Degree >::SupportStart;
				static const int  LeftPointSupportRadius =  BSplineSupportSizes< FEMSignature< TSig >::Degree >::SupportEnd  ;
				static const int      RightSupportRadius =  BSplineSupportSizes< FEMSignature< TSig >::Degree >::SupportEnd  ;
				static const int RightPointSupportRadius = -BSplineSupportSizes< FEMSignature< TSig >::Degree >::SupportStart;
				for( int s=-LeftPointSupportRadius ; s<=RightPointSupportRadius ; s++ )
				{
					int pIdx = state._pointOffset[I];
					int fIdx = state._pointOffset[I]+s;
					double _p = p[I];
					const Polynomial< FEMSignature< TSig >::Degree >* components = std::get< I >( _bSplineData )[depth].polynomialsAndOffset( _p , pIdx , fIdx );
					for( int d=0 ; d<=D ; d++ ) std::get< I >( state._oneDValues ).values[ s+LeftPointSupportRadius ][d] = components[d]( _p );
				}
			}
			template< typename State , unsigned int TSig , unsigned int ... TSigs , unsigned int D , unsigned int ... Ds >
			typename std::enable_if< sizeof...(TSigs)==0 >::type _initEvaluationState( UIntPack< TSig , TSigs ... > , UIntPack< D , Ds ... > , const double* p , unsigned int depth , State& state ) const
			{
				_setEvaluationState< Dim-1 , TSig , D >( p , depth , state );
			}
			template< typename State , unsigned int TSig , unsigned int ... TSigs , unsigned int D , unsigned int ... Ds >
			typename std::enable_if< sizeof...(TSigs)!=0 >::type _initEvaluationState( UIntPack< TSig , TSigs ... > , UIntPack< D , Ds ... > , const double* p , unsigned int depth , State& state ) const
			{
				_setEvaluationState< Dim-1-sizeof...(TSigs) , TSig , D >( p , depth , state );
				_initEvaluationState( UIntPack< TSigs ... >() , UIntPack< Ds ... >() , p , depth , state );
			}
		};

		template< unsigned int ... TSignatures >
		struct RestrictionProlongation< UIntPack< TSignatures ... > > : public BaseFEMIntegrator::template RestrictionProlongation< UIntPack< FEMSignature< TSignatures >::Degree ... > >
		{
			static const unsigned int Dim = sizeof ... ( TSignatures );
			typedef typename BaseFEMIntegrator::template RestrictionProlongation< UIntPack< FEMSignature< TSignatures >::Degree ... > > Base;

			double upSampleCoefficient( const int pOff[] , const int cOff[] ) const { return _coefficient( pOff , cOff ); }
			void init( unsigned int depth ){ Base::init( depth ); }
			void init( void ){ _init( Base::highDepth() ); }

		protected:
			std::tuple< typename BSplineEvaluationData< TSignatures >::UpSampleEvaluator ... > _upSamplers;

			template< unsigned int D=0 > typename std::enable_if< D==Dim >::type _init( int highDepth ){ }
			template< unsigned int D=0 > typename std::enable_if< D< Dim >::type _init( int highDepth ){ std::get< D >( _upSamplers ).set( highDepth-1 ) ; _init< D+1 >( highDepth ); }
			template< unsigned int D=0 > typename std::enable_if< D==Dim , double >::type _coefficient( const int pOff[] , const int cOff[] ) const { return 1.; }
			template< unsigned int D=0 > typename std::enable_if< D< Dim , double >::type _coefficient( const int pOff[] , const int cOff[] ) const { return _coefficient< D+1 >( pOff , cOff ) * std::get< D >( _upSamplers ).value( pOff[D] , cOff[D] ); }
		};

		template< unsigned int ... TSignatures , unsigned int ... TDerivatives , unsigned int ... CSignatures , unsigned int ... CDerivatives , unsigned int CDim >
		struct Constraint< UIntPack< TSignatures ... > , UIntPack< TDerivatives ... > , UIntPack< CSignatures ... > , UIntPack< CDerivatives ... > , CDim > : public BaseFEMIntegrator::template Constraint< UIntPack< FEMSignature< TSignatures >::Degree ... > , UIntPack< FEMSignature< CSignatures >::Degree ... > , CDim >
		{
			static_assert( sizeof ... ( TSignatures ) == sizeof ... ( CSignatures ) , "[ERROR] Test signatures and contraint signatures must have the same dimension" );
			static_assert( sizeof ... ( TSignatures ) == sizeof ... ( TDerivatives ) , "[ERROR] Test signatures and derivatives must have the same dimension" );
			static_assert( sizeof ... ( CSignatures ) == sizeof ... ( CDerivatives ) , "[ERROR] Constraint signatures and derivatives must have the same dimension" );
			static_assert( ParameterPack::Comparison< UIntPack< FEMSignature< TSignatures >::Degree ... > , UIntPack< TDerivatives ... > >::GreaterThanOrEqual , "[ERROR] Test functions cannot have more derivatives than the degree" );
			static_assert( ParameterPack::Comparison< UIntPack< FEMSignature< CSignatures >::Degree ... > , UIntPack< CDerivatives ... > >::GreaterThanOrEqual , "[ERROR] Test functions cannot have more derivatives than the degree" );

			static const unsigned int Dim = sizeof ... ( TSignatures );
			typedef typename BaseFEMIntegrator::template Constraint< UIntPack< FEMSignature< TSignatures >::Degree ... > , UIntPack< FEMSignature< CSignatures >::Degree ... > , CDim > Base;

			static const unsigned int TDerivativeSize = TensorDerivatives< UIntPack< TDerivatives ... > >::Size;
			static const unsigned int CDerivativeSize = TensorDerivatives< UIntPack< CDerivatives ... > >::Size;
			static inline void TFactorDerivatives( unsigned int idx , unsigned int d[ Dim ] ){ TensorDerivatives< UIntPack< TDerivatives ... > >::Factor( idx , d ); }
			static inline void CFactorDerivatives( unsigned int idx , unsigned int d[ Dim ] ){ TensorDerivatives< UIntPack< CDerivatives ... > >::Factor( idx , d ); }
			static inline unsigned int TDerivativeIndex( const unsigned int d[ Dim ] ){ return TensorDerivatives< UIntPack< TDerivatives ... > >::Index( d ); }
			static inline unsigned int CDerivativeIndex( const unsigned int d[ Dim ] ){ return TensorDerivatives< UIntPack< CDerivatives ... > >::Index( d ); }
			Matrix< double , TDerivativeSize , CDerivativeSize > weights[CDim];

			Point< double , CDim > ccIntegrate( const int off1[] , const int off2[] ) const { return _integrate( INTEGRATE_CHILD_CHILD  , off1 , off2 ); }
			Point< double , CDim > pcIntegrate( const int off1[] , const int off2[] ) const { return _integrate( INTEGRATE_PARENT_CHILD , off1 , off2 ); }
			Point< double , CDim > cpIntegrate( const int off1[] , const int off2[] ) const { return _integrate( INTEGRATE_CHILD_PARENT , off1 , off2 ); }

			void init( unsigned int depth ){ Base::init( depth ); }
			void init( void )
			{

				_init( Base::highDepth() );
				_weightedIndices.resize(0);
				for( unsigned int d1=0 ; d1<TDerivativeSize ; d1++ ) for( unsigned int d2=0 ; d2<CDerivativeSize ; d2++ )
				{
					_WeightedIndices w(d1,d2);
					for( unsigned int c=0 ; c<CDim ; c++ ) if( weights[c](d1,d2)>0 ) w.indices.push_back( std::pair< unsigned int , double >( c , weights[c](d1,d2) ) );
					if( w.indices.size() ) _weightedIndices.push_back(w);
				}
			}
			typename BaseFEMIntegrator::template RestrictionProlongation< UIntPack< FEMSignature< TSignatures >::Degree ... > >& tRestrictionProlongation( void ){ return _tRestrictionProlongation; }
			typename BaseFEMIntegrator::template RestrictionProlongation< UIntPack< FEMSignature< CSignatures >::Degree ... > >& cRestrictionProlongation( void ){ return _cRestrictionProlongation; }
		protected:
			RestrictionProlongation< UIntPack< TSignatures ... > > _tRestrictionProlongation;
			RestrictionProlongation< UIntPack< CSignatures ... > > _cRestrictionProlongation;
			struct _WeightedIndices
			{
				_WeightedIndices( unsigned int _d1=0 , unsigned int _d2=0 ) : d1(_d1) , d2(_d2) { ; }
				unsigned int d1 , d2;
				std::vector< std::pair< unsigned int , double > > indices;
			};
			std::vector< _WeightedIndices > _weightedIndices;
			enum IntegrationType
			{
				INTEGRATE_CHILD_CHILD ,
				INTEGRATE_PARENT_CHILD ,
				INTEGRATE_CHILD_PARENT
			};

			template< unsigned int _TSig , unsigned int _TDerivatives , unsigned int _CSig , unsigned int _CDerivatives >
			struct _Integrators
			{
				typename BSplineIntegrationData< _TSig , _CSig >::FunctionIntegrator::template      Integrator< _TDerivatives , _CDerivatives > ccIntegrator;
				typename BSplineIntegrationData< _TSig , _CSig >::FunctionIntegrator::template ChildIntegrator< _TDerivatives , _CDerivatives > pcIntegrator;
				typename BSplineIntegrationData< _CSig , _TSig >::FunctionIntegrator::template ChildIntegrator< _CDerivatives , _TDerivatives > cpIntegrator;
			};
			std::tuple< _Integrators< TSignatures , TDerivatives , CSignatures , CDerivatives > ... > _integrators;

			template< unsigned int D=0 >
			typename std::enable_if< D==Dim >::type _init( int depth ){ ; }
			template< unsigned int D=0 >
			typename std::enable_if< D< Dim >::type _init( int depth )
			{
				std::get< D >( _integrators ).ccIntegrator.set( depth );
				if( depth ) std::get< D >( _integrators ).pcIntegrator.set( depth-1 ) , std::get< D >( _integrators ).cpIntegrator.set( depth-1 );
				_init< D+1 >( depth );
			}
			template< unsigned int D=0 >
			typename std::enable_if< D==Dim , double >::type _integral( IntegrationType iType , const int off1[] , const int off2[] , const unsigned int d1[] , const unsigned int d2[] ) const { return 1.; }
			template< unsigned int D=0 >
			typename std::enable_if< D< Dim , double >::type _integral( IntegrationType iType , const int off1[] , const int off2[] , const unsigned int d1[] , const unsigned int d2[] ) const
			{
				double remainingIntegral = _integral< D+1 >( iType , off1 , off2 , d1 , d2 );
				switch( iType )
				{
				case INTEGRATE_CHILD_CHILD:  return std::get< D >( _integrators ).ccIntegrator.dot( off1[D] , off2[D] , d1[D] , d2[D] ) * remainingIntegral;
				case INTEGRATE_PARENT_CHILD: return std::get< D >( _integrators ).pcIntegrator.dot( off1[D] , off2[D] , d1[D] , d2[D] ) * remainingIntegral;
				case INTEGRATE_CHILD_PARENT: return std::get< D >( _integrators ).cpIntegrator.dot( off2[D] , off1[D] , d2[D] , d1[D] ) * remainingIntegral;
				default: MK_THROW( "Undefined integration type" );
				}
				return 0;
			}
			Point< double , CDim > _integrate( IntegrationType iType , const int off1[] , const int off[] ) const;
		};

		template< unsigned int ... TSignatures , unsigned int ... TDerivatives , unsigned int ... CSignatures , unsigned int ... CDerivatives >
		struct ScalarConstraint< UIntPack< TSignatures ... > , UIntPack< TDerivatives ... > , UIntPack< CSignatures ... > , UIntPack< CDerivatives ... > > : public Constraint< UIntPack< TSignatures ... > , UIntPack< TDerivatives ... > , UIntPack< CSignatures ... > , UIntPack< CDerivatives ... > , 1 >
		{
			static const unsigned int Dim = sizeof ... ( TSignatures );
			typedef typename BaseFEMIntegrator::template Constraint<  UIntPack< FEMSignature< TSignatures >::Degree ... > , UIntPack< FEMSignature< CSignatures >::Degree ... > , 1 > Base;

			typedef Constraint< UIntPack< TSignatures ... > , UIntPack< TDerivatives ... > , UIntPack< CSignatures ... > , UIntPack< CDerivatives ... > , 1 > FullConstraint;
			using FullConstraint::weights;
			// [NOTE] We define the constructor using a recursive function call to take into account multiplicity (e.g. so that d^2/dxdy and d^2/dydx each contribute)
			ScalarConstraint( const std::initializer_list< double >& w )
			{
				std::function< void ( unsigned int[] , const double[] , unsigned int ) > SetDerivativeWeights = [&]( unsigned int derivatives[Dim] , const double w[] , unsigned int d )
					{
						unsigned int idx1 = FullConstraint::TDerivativeIndex( derivatives ) , idx2 = FullConstraint::CDerivativeIndex( derivatives );
						weights[0][idx1][idx2] += w[0];
						if( d>0 ) for( int dd=0 ; dd<Dim ; dd++ ){ derivatives[dd]++ ; SetDerivativeWeights( derivatives , w+1 , d-1 ) ; derivatives[dd]--; }
					};
				static const unsigned int DMax = std::min< unsigned int >( UIntPack< TDerivatives ... >::Min() , UIntPack< CDerivatives ... >::Min() );

				unsigned int derivatives[Dim];
				double _w[DMax+1];
				memset( _w , 0 , sizeof(_w) );
				{
					unsigned int dd=0;
					for( typename std::initializer_list< double >::const_iterator iter=w.begin() ; iter!=w.end() && dd<=DMax ; dd++ , iter++ ) _w[dd] = *iter;
				}
				for( int d=0 ; d<Dim ; d++ ) derivatives[d] = 0;
				if( w.size() ) SetDerivativeWeights( derivatives , _w , std::min< unsigned int >( DMax+1 , (unsigned int)w.size() )-1 );
			}
		};

		template< unsigned int ... TSignatures , unsigned int ... TDerivatives >
		struct System< UIntPack< TSignatures ... > , UIntPack< TDerivatives ... > > : public BaseFEMIntegrator::template System< UIntPack< FEMSignature< TSignatures >::Degree... > >
		{
			static_assert( sizeof ... ( TSignatures ) == sizeof ... ( TDerivatives ) , "[ERROR] Test signatures and derivatives must have the same dimension" );

			static const unsigned int Dim = sizeof ... ( TSignatures );
			typedef typename BaseFEMIntegrator::template System< UIntPack< FEMSignature< TSignatures >::Degree... > > Base;

			System( const std::initializer_list< double >& w ) : _sc( w ){ ; }
			void init( unsigned int depth ){ Base::init( depth ); }
			void init( void ){ ( (BaseFEMIntegrator::template Constraint< UIntPack< FEMSignature< TSignatures >::Degree ... > , UIntPack< FEMSignature< TSignatures >::Degree ... > , 1 >&)_sc ).init( BaseFEMIntegrator::template System< UIntPack< FEMSignature< TSignatures >::Degree... > >::_highDepth ); }
			double ccIntegrate( const int off1[] , const int off2[] ) const { return _sc.ccIntegrate( off1 , off2 )[0]; }
			double pcIntegrate( const int off1[] , const int off2[] ) const { return _sc.pcIntegrate( off1 , off2 )[0]; }
			bool vanishesOnConstants( void ) const { return _sc.weights[0][0][0]==0; }

			typename BaseFEMIntegrator::template RestrictionProlongation< UIntPack< FEMSignature< TSignatures >::Degree ... > >& restrictionProlongation( void ){ return _sc.tRestrictionProlongation(); }

		protected:
			ScalarConstraint< UIntPack< TSignatures ... > , UIntPack< TDerivatives ... >  , UIntPack< TSignatures ... > , UIntPack< TDerivatives ... > > _sc;
		};
	};

	//////////////////////////////////////////

	template< unsigned int Dim > inline void SetGhostFlag(       RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >* node , bool flag ){ if( node && node->parent ) node->parent->nodeData.setGhostFlag( flag ); }
	template< unsigned int Dim > inline bool GetGhostFlag( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >* node ){ return node==NULL || node->parent==NULL || node->parent->nodeData.getGhostFlag( ); }
	template< unsigned int Dim > inline bool IsActiveNode( const RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type >* node ){ return !GetGhostFlag( node ); }

	// A class representing an extractot with and without auxiliary data
	template< typename Real , unsigned int Dim , typename ... Params > struct LevelSetExtractor;

	// A helper class which consolidates the two extractors, templated over HasData
	template< bool HasData , typename Real , unsigned int Dim , typename Data > struct _LevelSetExtractor;

	template< unsigned int Dim , class Data >
	struct NodeSample
	{
		RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > *node;
		Data data;
	};
	template< unsigned int Dim , class Real >
	struct NodeAndPointSample
	{
		RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > *node;
		ProjectiveData< Point< Real , Dim > , Real > sample;
	};
	template< unsigned int Dim , class Real > using NodeSimplices = NodeSample< Dim , std::vector< std::pair< node_index_type , Simplex< Real , Dim , Dim-1 > > > >;


	template< typename T > struct WindowLoopData{ };

	template< unsigned int ... Sizes >
	struct WindowLoopData< UIntPack< Sizes ... > >
	{
		static const int Dim = sizeof ... ( Sizes );
		unsigned int size[1<<Dim];
		unsigned int indices[1<<Dim][ WindowSize< UIntPack< Sizes ... > >::Size ];
		template< typename BoundsFunction >
		WindowLoopData( const BoundsFunction &boundsFunction )
		{
			int start[Dim] , end[Dim];
			for( int c=0 ; c<(1<<Dim) ; c++ )
			{
				size[c] = 0;
				boundsFunction( c , start , end );
				unsigned int idx[Dim];
				WindowLoop< Dim >::Run
				(
					start , end ,
					[&]( int d , int i ){ idx[d] = i; } ,
					[&]( void ){ indices[c][ size[c]++ ] = Window::GetIndex< Sizes ... >( idx ); }
				);
			}
		}
	};

	template< class Real , unsigned int Dim >
	struct Atomic< Point< Real , Dim > >
	{
		using Value = Point< Real , Dim >;
		static void Add( volatile Value &a , const Value &b ){ for( unsigned int d=0 ; d<Dim ; d++ ) AddAtomic( a.coords[d] , b[d] ); }
	};

	template< class Real >
	struct Atomic< Point< Real > >
	{
		using Value = Point< Real >;
		static void Add( volatile Value &a , const Value &b )
		{
			if( a._dim !=b._dim ) MK_THROW( "Sizes don't match: " , a._dim , " != " , b._dim );
			for( unsigned int d=0 ; d<a._dim && d<b.dim() ; d++ ) AddAtomic( a._coords[d] , b[d] );
		}
	};

	template< typename Data , typename Real >
	struct Atomic< ProjectiveData< Data , Real > >
	{
		using Value = ProjectiveData< Data , Real >;
		static void Add( volatile Value &a , const Value &b ){ Atomic< Real >::Add( a.weight , b.weight ) ; Atomic< Data >::Add( a.data , b.data ); }
	};

	template< typename Real , typename FirstType , typename ... RestTypes >
	struct Atomic< DirectSum< Real , FirstType , RestTypes... > > 
	{
		using Value = DirectSum< Real , FirstType , RestTypes... >;
		static void Add( volatile Value &a , const Value &b )
		{
			Atomic< FirstType >::Add( a._first , b._first );
			Atomic< DirectSum< Real , RestTypes... > >::Add( a._rest , b._rest );
		}
	};

	template< typename Real >
	struct Atomic< DirectSum< Real > > 
	{
		using Value = DirectSum< Real >;
		static void Add( volatile Value &a , const Value &b ){}
	};

	template< class Real , unsigned int Dim >
	Point< Real , Dim > ReadAtomic( const volatile Point< Real , Dim > & a )
	{
		Point< Real , Dim > p;
		for( int d=0 ; d<Dim ; d++ ) p[d] = ReadAtomic( a.coords[d] );
		return p;
	}

	template< class Real , unsigned int Dim >
	Point< Real , Dim > SetAtomic( volatile Point< Real , Dim > & p , Point< Real , Dim > newP )
	{
		Point< Real , Dim > oldP;
		for( int d=0 ; d<Dim ; d++ ) oldP[d] = SetAtomic( p.coords[d] , newP[d] );
		return oldP;
	}

	template< class Data >
	bool IsZero( const Data& data ){ return false; }
	template< class Real , unsigned int Dim >
	bool IsZero( const Point< Real , Dim >& d )
	{
		bool zero = true;
		for( int i=0 ; i<Dim ; i++ ) zero &= (d[i]==0); 
		return zero;
	}
	inline bool IsZero( const float& f ){ return f==0.f; }
	inline bool IsZero( const double& f ){ return f==0.; }

	template< unsigned int Dim , class Real >
	class FEMTree
	{
	public:
		typedef RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > FEMTreeNode;
		template< unsigned int ... FEMSigs > using SystemMatrixType = SparseMatrix< Real , matrix_index_type , WindowSize< UIntPack< BSplineOverlapSizes< FEMSignature< FEMSigs >::Degree >::OverlapSize ... > >::Size >;
		std::vector< Allocator< FEMTreeNode > * > nodeAllocators;
	protected:
		template< unsigned int _Dim , class _Real > friend class FEMTree;
		struct SubTreeExtractor : public FEMTreeNode::SubTreeExtractor
		{
			SubTreeExtractor( FEMTreeNode &node , int &depthOffset ) : FEMTreeNode::SubTreeExtractor( node ) , _depthOffset( depthOffset )
			{
				_depthOffsetValue = _depthOffset;
				_depthOffset = 0;
			}
			SubTreeExtractor( FEMTreeNode *node , int &depthOffset ) : SubTreeExtractor( *node , depthOffset ){}
			~SubTreeExtractor( void ){ _depthOffset = _depthOffsetValue; }
		protected:
			int &_depthOffset , _depthOffsetValue;
		};

		// Don't need this first one
		template< typename _Real , unsigned int _Dim , typename ... _Params > friend struct LevelSetExtractor;
		template< bool _HasData , typename _Real , unsigned int _Dim , typename _Data > friend struct _LevelSetExtractor;
		std::atomic< node_index_type > _nodeCount;
		struct _NodeInitializer
		{
			FEMTree& femTree;
			_NodeInitializer( FEMTree& f ) : femTree(f){;}
			void operator() ( FEMTreeNode& node ){ node.nodeData.nodeIndex = femTree._nodeCount++; }
		};
		_NodeInitializer _nodeInitializer;
	public:
		typedef int LocalDepth;
		typedef int LocalOffset[Dim];

		node_index_type nodeCount( void ) const { return _nodeCount; }

		typedef NodeAndPointSample< Dim , Real > PointSample;

		typedef typename FEMTreeNode::template      NeighborKey< IsotropicUIntPack< Dim , 1 > , IsotropicUIntPack< Dim , 1 > >      OneRingNeighborKey;
		typedef typename FEMTreeNode::template ConstNeighborKey< IsotropicUIntPack< Dim , 1 > , IsotropicUIntPack< Dim , 1 > > ConstOneRingNeighborKey;
		typedef typename FEMTreeNode::template      Neighbors< IsotropicUIntPack< Dim , 3 > >      OneRingNeighbors;
		typedef typename FEMTreeNode::template ConstNeighbors< IsotropicUIntPack< Dim , 3 > > ConstOneRingNeighbors;

		template< typename FEMDegreePack >                        using BaseSystem          = typename BaseFEMIntegrator::template System< FEMDegreePack >;
		template< typename FEMSigPack , typename DerivativePack > using PointEvaluator      = typename     FEMIntegrator::template PointEvaluator< FEMSigPack , DerivativePack >;
		template< typename FEMSigPack , typename DerivativePack > using PointEvaluatorState = typename     FEMIntegrator::template PointEvaluatorState< FEMSigPack , DerivativePack >;	
		template< typename FEMDegreePack > using CCStencil  = typename BaseSystem< FEMDegreePack >::CCStencil;
		template< typename FEMDegreePack > using PCStencils = typename BaseSystem< FEMDegreePack >::PCStencils;

		template< unsigned int ... FEMSigs > bool isValidFEMNode( UIntPack< FEMSigs ... > , const FEMTreeNode* node ) const;
		bool isValidSpaceNode( const FEMTreeNode* node ) const;
		const FEMTreeNode* leaf( Point< Real , Dim > p ) const;
	protected:
		template< bool ThreadSafe >
		FEMTreeNode* _leaf( Allocator< FEMTreeNode > *nodeAllocator , Point< Real , Dim > p , LocalDepth maxDepth=-1 );
		template< bool ThreadSafe >
		FEMTreeNode* _leaf( Allocator< FEMTreeNode > *nodeAllocator , Point< Real , Dim > p , std::function< int ( Point< Real , Dim > ) > &pointDepthFunctor );
	public:

		// [NOTE] In the case that T != double, we require both operators() for computing the system dual
		template< typename T , unsigned int PointD >
		struct InterpolationInfo
		{
			virtual void range( const FEMTreeNode* node , size_t& begin , size_t& end ) const = 0;
			virtual Point< T , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx ) const = 0;
			virtual Point< T , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx , const Point< T , CumulativeDerivatives< Dim , PointD >::Size >& dValues ) const = 0;
			virtual Point< double , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx , const Point< double , CumulativeDerivatives< Dim , PointD >::Size >& dValues ) const = 0;
			virtual const DualPointInfo< Dim , Real , T , PointD >& operator[]( size_t pointIdx ) const = 0;
			virtual bool constrainsDCTerm( void ) const = 0;
			virtual ~InterpolationInfo( void ){}

			DualPointInfo< Dim , Real , T , PointD >& operator[]( size_t pointIndex ){ return const_cast< DualPointInfo< Dim , Real , T , PointD >& >( ( ( const InterpolationInfo* )this )->operator[]( pointIndex ) ); }
		protected:
			virtual void _remapIndices( ConstPointer( node_index_type )oldNodeIndices , size_t newNodeCount ) = 0;
			friend class FEMTree< Dim , Real >;
		};

		template< unsigned int PointD >
		struct InterpolationInfo< double , PointD >
		{
			virtual void range( const FEMTreeNode* node , size_t& begin , size_t& end ) const = 0;
			virtual Point< double , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx ) const = 0;
			virtual Point< double , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx , const Point< double , CumulativeDerivatives< Dim , PointD >::Size >& dValues ) const = 0;
			virtual const DualPointInfo< Dim , Real , double , PointD >& operator[]( size_t pointIdx ) const = 0;
			virtual bool constrainsDCTerm( void ) const = 0;
			virtual ~InterpolationInfo( void ){}

			DualPointInfo< Dim , Real , double , PointD >& operator[]( size_t pointIndex ){ return const_cast< DualPointInfo< Dim , Real , double , PointD >& >( ( ( const InterpolationInfo* )this )->operator[]( pointIndex ) ); }
		protected:
			virtual void _remapIndices( ConstPointer( node_index_type )oldNodeIndices , size_t newNodeCount ) = 0;
			friend class FEMTree< Dim , Real >;
		};

		template< typename T , unsigned int PointD , typename ConstraintDual , typename SystemDual >
		struct ApproximatePointInterpolationInfo : public InterpolationInfo< T , PointD >
		{
			using Data = DualPointInfo< Dim , Real , T , PointD >;
			void range( const FEMTreeNode *node , size_t &begin , size_t &end ) const
			{
				node_index_type idx = iData.index( node );
				if( idx==-1 ) begin = end = 0;
				else begin = idx , end = idx+1;
			}
			bool constrainsDCTerm( void ) const { return _constrainsDCTerm; }
			const DualPointInfo< Dim , Real , T , PointD >& operator[]( size_t pointIdx ) const { return iData[ pointIdx ]; }
			Point< T , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx ) const { return _constraintDual( iData[ pointIdx ].position ); }
			Point< T , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx , const Point< T , CumulativeDerivatives< Dim , PointD >::Size >& dValues ) const { return _systemDual( iData[ pointIdx ].position , dValues ); }
			Point< double , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx , const Point< double , CumulativeDerivatives< Dim , PointD >::Size >& dValues ) const { return _systemDual( iData[ pointIdx ].position , dValues ); }

			ApproximatePointInterpolationInfo( ConstraintDual constraintDual , SystemDual systemDual , bool constrainsDCTerm ) : _constraintDual( constraintDual ) , _systemDual( systemDual ) , _constrainsDCTerm( constrainsDCTerm ) { }

			void write( BinaryStream &stream ) const
			{
				unsigned char constrainsDCTerm = _constrainsDCTerm ? 1 : 0;
				stream.write( constrainsDCTerm );
				stream.write( _constraintDual );
				stream.write( _systemDual );
				iData.write( stream );
			}
			void read( BinaryStream &stream )
			{
				unsigned char constrainsDCTerm;
				if( !stream.read( constrainsDCTerm ) ) MK_THROW( "Failed to read constrainsDCTerm" );
				_constrainsDCTerm = constrainsDCTerm!=0;
				if( !stream.read( _constraintDual ) ) MK_THROW( "Failed to read _constraintDual" );
				if( !stream.read( _systemDual ) ) MK_THROW( "Failed to read _systemDual" );
				iData.read( stream );
			}
			ApproximatePointInterpolationInfo( BinaryStream &stream ){ read(stream); }
			ApproximatePointInterpolationInfo( void ){}

			SparseNodeData< DualPointInfo< Dim , Real , T , PointD > , ZeroUIntPack< Dim > > iData;

		protected:

			bool _constrainsDCTerm;
			ConstraintDual _constraintDual;
			SystemDual _systemDual;

			void _remapIndices( ConstPointer( node_index_type )oldNodeIndices , size_t newNodeCount ){ iData._remapIndices( oldNodeIndices , newNodeCount ); }

			friend class FEMTree< Dim , Real >;
		};

		template< unsigned int PointD , typename ConstraintDual , typename SystemDual >
		struct ApproximatePointInterpolationInfo< double , PointD , ConstraintDual , SystemDual > : public InterpolationInfo< double , PointD >
		{
			typedef double T;
			using Data = DualPointInfo< Dim , Real , T , PointD >;
			void range( const FEMTreeNode* node , size_t& begin , size_t& end ) const
			{
				node_index_type idx = iData.index( node );
				if( idx==-1 ) begin = end = 0;
				else begin = idx , end = idx+1;
			}
			bool constrainsDCTerm( void ) const { return _constrainsDCTerm; }
			const DualPointInfo< Dim , Real , T , PointD >& operator[]( size_t pointIdx ) const { return iData[ pointIdx ]; }
			Point< T , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx ) const { return _constraintDual( iData[ pointIdx ].position ); }
			Point< T , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx , const Point< T , CumulativeDerivatives< Dim , PointD >::Size >& dValues ) const { return _systemDual( iData[ pointIdx ].position , dValues ); }

			ApproximatePointInterpolationInfo( ConstraintDual constraintDual , SystemDual systemDual , bool constrainsDCTerm ) : _constraintDual( constraintDual ) , _systemDual( systemDual ) , _constrainsDCTerm( constrainsDCTerm ) { }

			void write( BinaryStream &stream ) const
			{
				unsigned char constrainsDCTerm = _constrainsDCTerm ? 1 : 0;
				stream.write( constrainsDCTerm );
				stream.write( _constraintDual );
				stream.write( _systemDual );
				iData.write( stream );
			}
			void read( BinaryStream &stream )
			{
				unsigned char constrainsDCTerm;
				if( !stream.read( constrainsDCTerm ) ) MK_THROW( "Failed to read constrainsDCTerm" );
				_constrainsDCTerm = constrainsDCTerm!=0;
				if( !stream.read( _constraintDual ) ) MK_THROW( "Failed to read _constraintDual" );
				if( !stream.read( _systemDual ) ) MK_THROW( "Failed to read _systemDual" );
				iData.read( stream );
			}
			ApproximatePointInterpolationInfo( BinaryStream &stream ){ read(stream); }
			ApproximatePointInterpolationInfo( void ){}

			SparseNodeData< DualPointInfo< Dim , Real , T , PointD > , ZeroUIntPack< Dim > > iData;

		protected:

			bool _constrainsDCTerm;
			ConstraintDual _constraintDual;
			SystemDual _systemDual;

			void _remapIndices( ConstPointer( node_index_type )oldNodeIndices , size_t newNodeCount ){ iData._remapIndices( oldNodeIndices , newNodeCount ); }

			friend class FEMTree< Dim , Real >;
		};

		template< typename T , typename Data , unsigned int PointD , typename ConstraintDual , typename SystemDual >
		struct ApproximatePointAndDataInterpolationInfo : public InterpolationInfo< T , PointD >
		{
			void range( const FEMTreeNode* node , size_t& begin , size_t& end ) const
			{
				node_index_type idx = iData.index( node );
				if( idx==-1 ) begin = end = 0;
				else begin = idx , end = idx+1;
			}
			bool constrainsDCTerm( void ) const { return _constrainsDCTerm; }
			const DualPointInfo< Dim , Real , T , PointD >& operator[]( size_t pointIdx ) const { return iData[ pointIdx ].pointInfo; }
			Point< T , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx ) const { return _constraintDual( iData[ pointIdx ].pointInfo.position , iData[ pointIdx ].data ); }
			Point< T , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx , const Point< T , CumulativeDerivatives< Dim , PointD >::Size >& dValues ) const { return _systemDual( iData[ pointIdx ].pointInfo.position , iData[ (int)pointIdx ].data , dValues ); }
			typename std::enable_if< !std::is_same< T , double >::value , Point< double , CumulativeDerivatives< Dim , PointD >::Size > >::type operator() ( size_t pointIdx , const Point< double , CumulativeDerivatives< Dim , PointD >::Size >& dValues ) const { return _systemDual( iData[ (int)pointIdx ].pointInfo.position , iData[ (int)pointIdx ].data , dValues ); }

			ApproximatePointAndDataInterpolationInfo( ConstraintDual constraintDual , SystemDual systemDual , bool constrainsDCTerm ) : _constraintDual( constraintDual ) , _systemDual( systemDual ) , _constrainsDCTerm( constrainsDCTerm ) { }

			void write( BinaryStream &stream ) const
			{
				unsigned char constrainsDCTerm = _constrainsDCTerm ? 1 : 0;
				stream.write( constrainsDCTerm );
				stream.write( _constraintDual );
				stream.write( _systemDual );
				iData.write( stream );
			}
			void read( BinaryStream &stream )
			{
				unsigned char constrainsDCTerm;
				if( !stream.read( constrainsDCTerm ) ) MK_THROW( "Failed to read constrainsDCTerm" );
				_constrainsDCTerm = constrainsDCTerm!=0;
				if( !stream.read( _constraintDual ) ) MK_THROW( "Failed to read _constraintDual" );
				if( !stream.read( _systemDual ) ) MK_THROW( "Failed to read _systemDual" );
				iData.read( stream );
			}
			ApproximatePointAndDataInterpolationInfo( BinaryStream &stream ){ read(stream); }

			SparseNodeData< DualPointAndDataInfo< Dim , Real , Data , T , PointD > , ZeroUIntPack< Dim > > iData;

		protected:

			bool _constrainsDCTerm;
			ConstraintDual _constraintDual;
			SystemDual _systemDual;

			void _remapIndices( ConstPointer( node_index_type )oldNodeIndices , size_t newNodeCount ){ iData._remapIndices( oldNodeIndices , newNodeCount ); }

			ApproximatePointAndDataInterpolationInfo( void ){}
			friend class FEMTree< Dim , Real >;
		};

		template< typename Data , unsigned int PointD , typename ConstraintDual , typename SystemDual >
		struct ApproximatePointAndDataInterpolationInfo< double , Data , PointD , ConstraintDual , SystemDual > : public InterpolationInfo< double , PointD >
		{
			typedef double T;
			void range( const FEMTreeNode* node , size_t& begin , size_t& end ) const
			{
				node_index_type idx = iData.index( node );
				if( idx==-1 ) begin = end = 0;
				else begin = idx , end = idx+1;
			}
			bool constrainsDCTerm( void ) const { return _constrainsDCTerm; }
			const DualPointInfo< Dim , Real , T , PointD >& operator[]( size_t pointIdx ) const { return iData[ pointIdx ].pointInfo; }
			Point< T , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx ) const { return _constraintDual( iData[ pointIdx ].pointInfo.position , iData[ pointIdx ].data ); }
			Point< T , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx , const Point< T , CumulativeDerivatives< Dim , PointD >::Size >& dValues ) const { return _systemDual( iData[ pointIdx ].pointInfo.position , iData[ pointIdx ].data , dValues ); }

			ApproximatePointAndDataInterpolationInfo( ConstraintDual constraintDual , SystemDual systemDual , bool constrainsDCTerm ) : _constraintDual( constraintDual ) , _systemDual( systemDual ) , _constrainsDCTerm( constrainsDCTerm ) { }

			void write( BinaryStream &stream ) const
			{
				unsigned char constrainsDCTerm = _constrainsDCTerm ? 1 : 0;
				stream.write( constrainsDCTerm );
				stream.write( _constraintDual );
				stream.write( _systemDual );
				iData.write( stream );
			}
			void read( BinaryStream &stream )
			{
				unsigned char constrainsDCTerm;
				if( !stream.read( constrainsDCTerm ) ) MK_THROW( "Failed to read constrainsDCTerm" );
				_constrainsDCTerm = constrainsDCTerm!=0;
				if( !stream.read( _constraintDual ) ) MK_THROW( "Failed to read _constraintDual" );
				if( !stream.read( _systemDual ) ) MK_THROW( "Failed to read _systemDual" );
				iData.read( stream );
			}

			ApproximatePointAndDataInterpolationInfo( BinaryStream &stream ){ read(stream); }

			SparseNodeData< DualPointAndDataInfo< Dim , Real , Data , T , PointD > , ZeroUIntPack< Dim > > iData;

		protected:

			bool _constrainsDCTerm;
			ConstraintDual _constraintDual;
			SystemDual _systemDual;

			void _remapIndices( ConstPointer( node_index_type )oldNodeIndices , size_t newNodeCount ){ iData._remapIndices( oldNodeIndices , newNodeCount ); }

			ApproximatePointAndDataInterpolationInfo( void ){}
			friend class FEMTree< Dim , Real >;
		};

		template< typename T , unsigned int PointD , typename ConstraintDual , typename SystemDual >
		struct ExactPointInterpolationInfo : public InterpolationInfo< T , PointD >
		{
			void range( const FEMTreeNode *node , size_t &begin , size_t &end ) const { begin = _sampleSpan[ node->nodeData.nodeIndex ].first , end = _sampleSpan[ node->nodeData.nodeIndex ].second; }
			bool constrainsDCTerm( void ) const { return _constrainsDCTerm; }
			const DualPointInfo< Dim , Real , T , PointD >& operator[]( size_t pointIdx ) const { return _iData[ pointIdx ]; }
			Point< T , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx ) const { return _constraintDual( _iData[ pointIdx ].position ); }
			Point< T , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx , const Point< T , CumulativeDerivatives< Dim , PointD >::Size >& dValues ) const { return _systemDual( _iData[ pointIdx ].position , dValues ); }
			typename std::enable_if< !std::is_same< T , double >::value , Point< double , CumulativeDerivatives< Dim , PointD >::Size > >::type operator() ( size_t pointIdx , const Point< double , CumulativeDerivatives< Dim , PointD >::Size >& dValues ) const { return _systemDual( _iData[ pointIdx ].position , dValues ); }

			ExactPointInterpolationInfo( ConstraintDual constraintDual , SystemDual systemDual , bool constrainsDCTerm ) : _constraintDual( constraintDual ) , _systemDual( systemDual ) , _constrainsDCTerm( constrainsDCTerm ) { }
		protected:
			void _init( const class FEMTree< Dim , Real >& tree , const std::vector< PointSample >& samples , bool noRescale );

			std::vector< std::pair< node_index_type , node_index_type > > _sampleSpan;
			std::vector< DualPointInfo< Dim , Real , T , PointD > > _iData;
			bool _constrainsDCTerm;
			ConstraintDual _constraintDual;
			SystemDual _systemDual;

			void _remapIndices( ConstPointer( node_index_type )oldNodeIndices , size_t newNodeCount )
			{
				std::vector< std::pair< node_index_type , node_index_type > > _newSampleSpan( newNodeCount );
				for( size_t i=0 ; i<newNodeCount ; i++ ) if( oldNodeIndices[i]!=-1 && oldNodeIndices[i]<(node_index_type)_sampleSpan.size() ) _newSampleSpan[i] = _sampleSpan[ oldNodeIndices[i] ];
				_sampleSpan = _newSampleSpan;
			}

			friend class FEMTree< Dim , Real >;
		};

		template< unsigned int PointD , typename ConstraintDual , typename SystemDual >
		struct ExactPointInterpolationInfo< double , PointD , ConstraintDual , SystemDual > : public InterpolationInfo< double , PointD >
		{
			typedef double T;
			void range( const FEMTreeNode* node , size_t& begin , size_t& end ) const { begin = _sampleSpan[ node->nodeData.nodeIndex ].first , end = _sampleSpan[ node->nodeData.nodeIndex ].second; }
			bool constrainsDCTerm( void ) const { return _constrainsDCTerm; }
			const DualPointInfo< Dim , Real , T , PointD >& operator[]( size_t pointIdx ) const { return _iData[ pointIdx ]; }
			Point< T , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx ) const { return _constraintDual( _iData[ pointIdx ].position ); }
			Point< T , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx , const Point< T , CumulativeDerivatives< Dim , PointD >::Size >& dValues ) const { return _systemDual( _iData[ pointIdx ].position , dValues ); }

			ExactPointInterpolationInfo( ConstraintDual constraintDual , SystemDual systemDual , bool constrainsDCTerm ) : _constraintDual( constraintDual ) , _systemDual( systemDual ) , _constrainsDCTerm( constrainsDCTerm ) { }
		protected:
			void _init( const class FEMTree< Dim , Real >& tree , const std::vector< PointSample >& samples , bool noRescale );

			std::vector< std::pair< node_index_type , node_index_type > > _sampleSpan;
			std::vector< DualPointInfo< Dim , Real , T , PointD > > _iData;
			bool _constrainsDCTerm;
			ConstraintDual _constraintDual;
			SystemDual _systemDual;

			void _remapIndices( ConstPointer( node_index_type )oldNodeIndices , size_t newNodeCount )
			{
				std::vector< std::pair< node_index_type , node_index_type > > _newSampleSpan( newNodeCount );
				for( size_t i=0 ; i<newNodeCount ; i++ ) if( oldNodeIndices[i]!=-1 && oldNodeIndices[i]<(node_index_type)_sampleSpan.size() ) _newSampleSpan[i] = _sampleSpan[ oldNodeIndices[i] ];
				_sampleSpan = _newSampleSpan;
			}

			friend class FEMTree< Dim , Real >;
		};

		template< typename T , typename Data , unsigned int PointD , typename ConstraintDual , typename SystemDual >
		struct _ExactPointAndDataInterpolationInfo : public InterpolationInfo< T , PointD >
		{
			_ExactPointAndDataInterpolationInfo( ConstraintDual constraintDual , SystemDual systemDual , bool constrainsDCTerm ) : _constraintDual( constraintDual ) , _systemDual( systemDual ) , _constrainsDCTerm( constrainsDCTerm ) { }

		protected:
			void _init( const class FEMTree< Dim , Real >& tree , const std::vector< PointSample >& samples , ConstPointer( Data ) sampleData , bool noRescale );

			std::vector< std::pair< node_index_type , node_index_type > > _sampleSpan;
			std::vector< DualPointAndDataInfo< Dim , Real , Data , T , PointD > > _iData;
			bool _constrainsDCTerm;
			ConstraintDual _constraintDual;
			SystemDual _systemDual;

			void _remapIndices( ConstPointer( node_index_type )oldNodeIndices , size_t newNodeCount )
			{
				std::vector< std::pair< node_index_type , node_index_type > > _newSampleSpan( newNodeCount );
				for( size_t i=0 ; i<newNodeCount ; i++ ) if( oldNodeIndices[i]!=-1 && oldNodeIndices[i]<(node_index_type)_sampleSpan.size() ) _newSampleSpan[i] = _sampleSpan[ oldNodeIndices[i]  ];
				_sampleSpan = _newSampleSpan;
			}

			friend class FEMTree< Dim , Real >;
		};

		template< typename T , typename Data , unsigned int PointD , typename ConstraintDual , typename SystemDual >
		struct ExactPointAndDataInterpolationInfo : public _ExactPointAndDataInterpolationInfo< T , Data , PointD , ConstraintDual , SystemDual >
		{
			using _ExactPointAndDataInterpolationInfo< T , Data , PointD , ConstraintDual , SystemDual >::_sampleSpan;
			using _ExactPointAndDataInterpolationInfo< T , Data , PointD , ConstraintDual , SystemDual >::_constrainsDCTerm;
			using _ExactPointAndDataInterpolationInfo< T , Data , PointD , ConstraintDual , SystemDual >::_iData;
			using _ExactPointAndDataInterpolationInfo< T , Data , PointD , ConstraintDual , SystemDual >::_constraintDual;
			using _ExactPointAndDataInterpolationInfo< T , Data , PointD , ConstraintDual , SystemDual >::_systemDual;
			void range( const FEMTreeNode* node , size_t& begin , size_t& end ) const { begin = _sampleSpan[ node->nodeData.nodeIndex ].first , end = _sampleSpan[ node->nodeData.nodeIndex ].second; }
			bool constrainsDCTerm( void ) const { return _constrainsDCTerm; }
			const DualPointInfo< Dim , Real , T , PointD >& operator[]( size_t pointIdx ) const { return _iData[ pointIdx ].pointInfo; }
			Point< T , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx ) const { return _constraintDual( _iData[ pointIdx ].pointInfo.position , _iData[ pointIdx ].data ); }
			Point< T , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx , const Point< T , CumulativeDerivatives< Dim , PointD >::Size >& dValues ) const { return _systemDual( _iData[ pointIdx ].pointInfo.position , _iData[ pointIdx ].data , dValues ); }
			Point< double , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx , const Point< double , CumulativeDerivatives< Dim , PointD >::Size >& dValues ) const { return _systemDual( _iData[ pointIdx ].pointInfo.position , _iData[ (int)pointIdx ].data , dValues ); }

			ExactPointAndDataInterpolationInfo( ConstraintDual constraintDual , SystemDual systemDual , bool constrainsDCTerm ) : _ExactPointAndDataInterpolationInfo< T , Data , PointD , ConstraintDual , SystemDual >( constraintDual , systemDual , constrainsDCTerm ) { }
		};

		template< typename Data , unsigned int PointD , typename ConstraintDual , typename SystemDual >
		struct ExactPointAndDataInterpolationInfo< double , Data , PointD , ConstraintDual , SystemDual > : public _ExactPointAndDataInterpolationInfo< double , Data , PointD , ConstraintDual , SystemDual >
		{
			using _ExactPointAndDataInterpolationInfo< double , Data , PointD , ConstraintDual , SystemDual >::_sampleSpan;
			using _ExactPointAndDataInterpolationInfo< double , Data , PointD , ConstraintDual , SystemDual >::_constrainsDCTerm;
			using _ExactPointAndDataInterpolationInfo< double , Data , PointD , ConstraintDual , SystemDual >::_iData;
			using _ExactPointAndDataInterpolationInfo< double , Data , PointD , ConstraintDual , SystemDual >::_constraintDual;
			using _ExactPointAndDataInterpolationInfo< double , Data , PointD , ConstraintDual , SystemDual >::_systemDual;
			void range( const FEMTreeNode* node , size_t& begin , size_t& end ) const { begin = _sampleSpan[ node->nodeData.nodeIndex ].first , end = _sampleSpan[ node->nodeData.nodeIndex ].second; }
			bool constrainsDCTerm( void ) const { return _constrainsDCTerm; }
			const DualPointInfo< Dim , Real , double , PointD >& operator[]( size_t pointIdx ) const { return _iData[ pointIdx ].pointInfo; }
			Point< double , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx ) const { return _constraintDual( _iData[ pointIdx ].pointInfo.position , _iData[ pointIdx ].data ); }
			Point< double , CumulativeDerivatives< Dim , PointD >::Size > operator() ( size_t pointIdx , const Point< double , CumulativeDerivatives< Dim , PointD >::Size >& dValues ) const { return _systemDual( _iData[ pointIdx ].pointInfo.position , _iData[ (int)pointIdx ].data , dValues ); }

			ExactPointAndDataInterpolationInfo( ConstraintDual constraintDual , SystemDual systemDual , bool constrainsDCTerm ) : _ExactPointAndDataInterpolationInfo< double , Data , PointD , ConstraintDual , SystemDual >( constraintDual , systemDual , constrainsDCTerm ) { }
		};

		template< typename T ,                 unsigned int PointD , typename ConstraintDual , typename SystemDual >
		static ApproximatePointInterpolationInfo< T ,            PointD , ConstraintDual , SystemDual >* InitializeApproximatePointInterpolationInfo( const class FEMTree< Dim , Real >& tree , const std::vector< PointSample >& samples , ConstraintDual constraintDual , SystemDual systemDual , bool constrainsDCTerm , LocalDepth maxDepth , int adaptiveExponent )
		{
			ApproximatePointInterpolationInfo< T , PointD , ConstraintDual , SystemDual >* a = new ApproximatePointInterpolationInfo< T , PointD , ConstraintDual , SystemDual >( constraintDual , systemDual , constrainsDCTerm );
			a->iData = tree._densifyInterpolationInfoAndSetDualConstraints< T , PointD >( samples , constraintDual , maxDepth , adaptiveExponent );
			return a;
		}
		template< typename T , typename Data , unsigned int PointD , typename ConstraintDual , typename SystemDual >
		static ApproximatePointAndDataInterpolationInfo< T , Data , PointD , ConstraintDual , SystemDual >* InitializeApproximatePointAndDataInterpolationInfo( const class FEMTree< Dim , Real >& tree , const std::vector< PointSample >& samples , ConstPointer( Data ) sampleData , ConstraintDual constraintDual , SystemDual systemDual , bool constrainsDCTerm , LocalDepth maxDepth , int adaptiveExponent )
		{
			ApproximatePointAndDataInterpolationInfo< T , Data , PointD , ConstraintDual , SystemDual >* a = new ApproximatePointAndDataInterpolationInfo< T , Data , PointD , ConstraintDual , SystemDual >( constraintDual , systemDual , constrainsDCTerm );
			a->iData = tree._densifyInterpolationInfoAndSetDualConstraints< T , Data , PointD >( samples , sampleData , constraintDual , maxDepth , adaptiveExponent );
			return a;
		}

		template< typename T , unsigned int PointD , typename ConstraintDual , typename SystemDual >
		static ExactPointInterpolationInfo< T , PointD , ConstraintDual , SystemDual >* InitializeExactPointInterpolationInfo( const class FEMTree< Dim , Real >& tree , const std::vector< PointSample >& samples , ConstraintDual constraintDual , SystemDual systemDual , bool constrainsDCTerm , bool noRescale )
		{
			ExactPointInterpolationInfo< T , PointD , ConstraintDual , SystemDual >* e = new ExactPointInterpolationInfo< T , PointD , ConstraintDual , SystemDual >( constraintDual , systemDual , constrainsDCTerm );
			e->_init( tree , samples , noRescale );
			return e;
		}
		template< typename T , typename Data , unsigned int PointD , typename ConstraintDual , typename SystemDual >
		static ExactPointAndDataInterpolationInfo< T , Data , PointD , ConstraintDual , SystemDual >* InitializeExactPointAndDataInterpolationInfo( const class FEMTree< Dim , Real >& tree , const std::vector< PointSample >& samples , ConstPointer( Data ) sampleData , ConstraintDual constraintDual , SystemDual systemDual , bool constrainsDCTerm , bool noRescale )
		{
			ExactPointAndDataInterpolationInfo< T , Data , PointD , ConstraintDual , SystemDual >* e = new ExactPointAndDataInterpolationInfo< T , Data , PointD , ConstraintDual , SystemDual >( constraintDual , systemDual , constrainsDCTerm );
			e->_init( tree , samples , sampleData , noRescale );
			return e;
		}

		template< typename T , unsigned int PointD , typename ConstraintDual , typename SystemDual > friend struct ExactPointInterpolationInfo;
		template< typename T , typename Data , unsigned int PointD , typename ConstraintDual , typename SystemDual > friend struct ExactPointAndDataInterpolationInfo;

		template< typename ... InterpolationInfos >
		static bool ConstrainsDCTerm( std::tuple< InterpolationInfos *... > interpolationInfos ){ return _ConstrainsDCTerm< 0 >( interpolationInfos ); }
	protected:
		template< unsigned int Idx , typename ... InterpolationInfos >
		static typename std::enable_if< ( Idx==sizeof...(InterpolationInfos) ) , bool >::type _ConstrainsDCTerm( std::tuple< InterpolationInfos *... > interpolationInfos ){ return false; }
		template< unsigned int Idx , typename ... InterpolationInfos >
		static typename std::enable_if< ( Idx <sizeof...(InterpolationInfos) ) , bool >::type _ConstrainsDCTerm( std::tuple< InterpolationInfos *... > interpolationInfos )
		{
			if(  std::get< Idx >( interpolationInfos ) ) return _ConstrainsDCTerm< Idx+1 >( interpolationInfos ) || std::get< Idx >( interpolationInfos )->constrainsDCTerm();
			else                                         return _ConstrainsDCTerm< Idx+1 >( interpolationInfos );
		}
	public:

#ifdef SHOW_WARNINGS
#pragma message( "[WARNING] This should not be isotropic" )
#endif // SHOW_WARNINGS
		template< unsigned int DensityDegree > struct DensityEstimator : public SparseNodeData< Real , IsotropicUIntPack< Dim , FEMDegreeAndBType< DensityDegree >::Signature > >
		{
			DensityEstimator( int kernelDepth , int coDimension , Real samplesPerNode ) : _kernelDepth( kernelDepth ) , _coDimension( coDimension ) , _samplesPerNode( samplesPerNode ){ }
			Real samplesPerNode( void ) const { return _samplesPerNode; }
			int coDimension( void ) const { return _coDimension; }
			int kernelDepth( void ) const { return _kernelDepth; }

			void write( BinaryStream &stream ) const
			{
				stream.write( _kernelDepth );
				stream.write( _coDimension );
				stream.write( _samplesPerNode );
				SparseNodeData< Real , IsotropicUIntPack< Dim , FEMDegreeAndBType< DensityDegree >::Signature > >::write( stream );
			}
			void read( BinaryStream &stream )
			{
				if( !stream.read( _kernelDepth ) ) MK_THROW( "Failed to read _kernelDepth" );
				if( !stream.read( _coDimension ) ) MK_THROW( "Failed to read _coDimension" );
				if( !stream.read( _samplesPerNode ) ) MK_THROW( "Failed to read _samplesPerNode" );
				SparseNodeData< Real , IsotropicUIntPack< Dim , FEMDegreeAndBType< DensityDegree >::Signature > >::read( stream );
			}
			DensityEstimator( BinaryStream &stream ){ read(stream); }

		protected:
			Real _samplesPerNode;
			int _kernelDepth , _coDimension;
		};

	protected:
		bool _isValidSpaceNode( const FEMTreeNode* node ) const { return !GetGhostFlag< Dim >( node ) && ( node->nodeData.flags & FEMTreeNodeData::SPACE_FLAG ); }
		bool _isValidFEM1Node ( const FEMTreeNode* node ) const { return !GetGhostFlag< Dim >( node ) && ( node->nodeData.flags & FEMTreeNodeData::FEM_FLAG_1 ); }
		bool _isValidFEM2Node ( const FEMTreeNode* node ) const { return !GetGhostFlag< Dim >( node ) && ( node->nodeData.flags & FEMTreeNodeData::FEM_FLAG_2 ); }

		FEMTreeNode _tree;
		mutable FEMTreeNode* _spaceRoot;
		SortedTreeNodes< Dim > _sNodes;
		LocalDepth _maxDepth;
		int _depthOffset;
		LocalDepth _baseDepth;
#ifdef USE_EXACT_PROLONGATION
		LocalDepth _exactDepth;
#endif // USE_EXACT_PROLONGATION
		mutable unsigned int _femSigs1[ Dim ];
		mutable unsigned int _femSigs2[ Dim ];
		void _init( void );

		static bool _InBounds( Point< Real , Dim > p );
		int _localToGlobal( LocalDepth d ) const { return d + _depthOffset; }
		LocalDepth _globalToLocal( int d ) const { return d - _depthOffset; }
		LocalDepth _localDepth( const FEMTreeNode* node ) const { return node->depth() - _depthOffset; }
		int _localInset( LocalDepth d ) const { return _depthOffset==0 ? 0 : 1<<( d + _depthOffset - 1 ); }
		void _localDepthAndOffset( const FEMTreeNode* node , LocalDepth& d , LocalOffset& off ) const
		{
			node->depthAndOffset( d , off );
			d -= _depthOffset;
			if( d<0 ) for( int d=0 ; d<Dim ; d++ ) off[d] = -1;
			else
			{
				int inset = _localInset( d );
				for( int d=0 ; d<Dim ; d++ ) off[d] -= inset;
			}
		}
		template< unsigned int FEMSig > static int _BSplineBegin( LocalDepth depth ){ return BSplineEvaluationData< FEMSig >::Begin( depth ); }
		template< unsigned int FEMSig > static int _BSplineEnd  ( LocalDepth depth ){ return BSplineEvaluationData< FEMSig >::End  ( depth ); }
		template< unsigned int ... FEMSigs >
		bool _outOfBounds( UIntPack< FEMSigs ... > , const FEMTreeNode* node ) const
		{
			if( !node ) return true;
			LocalDepth d ; LocalOffset off ; _localDepthAndOffset( node , d , off );
			return FEMIntegrator::IsOutOfBounds( UIntPack< FEMSigs ... >() , d , off );
		}
		node_index_type _sNodesBegin( LocalDepth d )             const { return _sNodes.begin( _localToGlobal( d ) ); }
		node_index_type _sNodesBegin( LocalDepth d , int slice ) const { return _sNodes.begin( _localToGlobal( d ) , slice + _localInset( d ) ); }
		node_index_type _sNodesBeginSlice( LocalDepth d )        const { return _localInset(d); }
		node_index_type _sNodesEnd( LocalDepth d )             const { return _sNodes.end  ( _localToGlobal( d ) ); }
		node_index_type _sNodesEnd( LocalDepth d , int slice ) const { return _sNodes.end  ( _localToGlobal( d ) , slice + _localInset( d ) ); }
		node_index_type _sNodesEndSlice( LocalDepth d ) const{ return ( 1<<_localToGlobal(d) ) - _localInset(d) - 1; }
		size_t _sNodesSize( LocalDepth d )             const { return _sNodes.size( _localToGlobal( d ) ); }
		size_t _sNodesSize( LocalDepth d , int slice ) const { return _sNodes.size( _localToGlobal( d ) , slice + _localInset( d ) ); }

		template< unsigned int ... FEMDegrees > static bool _IsSupported( UIntPack< FEMDegrees ... > , LocalDepth femDepth , const LocalOffset femOffset , LocalDepth spaceDepth , const LocalOffset spaceOffset){ return BaseFEMIntegrator::IsSupported( UIntPack< FEMDegrees ... >() , femDepth , femOffset , spaceDepth , spaceOffset ); }
		template< unsigned int ... FEMDegrees > bool _isSupported( UIntPack< FEMDegrees ... > , const FEMTreeNode *femNode , const FEMTreeNode *spaceNode ) const
		{
			if( !femNode || !spaceNode ) return false;
			LocalDepth femDepth , spaceDepth ; LocalOffset femOffset , spaceOffset;
			_localDepthAndOffset( femNode , femDepth , femOffset ) , _localDepthAndOffset( spaceNode , spaceDepth , spaceOffset );
			return _IsSupported( UIntPack< FEMDegrees ... >() , femDepth , femOffset , spaceDepth , spaceOffset );
		}

		template< unsigned int FEMDegree > static bool _IsInteriorlySupported( LocalDepth depth , const LocalOffset off )
		{
			if( depth>=0 )
			{
				int begin , end;
				BSplineSupportSizes< FEMDegree >::InteriorSupportedSpan( depth , begin , end );
				bool interior = true;
				for( int dd=0 ; dd<Dim ; dd++ ) interior &= off[dd]>=begin && off[dd]<end;
				return interior;
			}
			else return false;
		}

		template< unsigned int FEMDegree > bool _isInteriorlySupported( const FEMTreeNode* node ) const
		{
			if( !node ) return false;
			LocalDepth d ; LocalOffset off;
			_localDepthAndOffset( node , d , off );
			return _IsInteriorlySupported< FEMDegree >( d , off );
		}
		template< unsigned int ... FEMDegrees > static bool _IsInteriorlySupported( UIntPack< FEMDegrees ... > , LocalDepth depth , const LocalOffset off ){ return BaseFEMIntegrator::IsInteriorlySupported( UIntPack< FEMDegrees ... >() , depth , off ); }
		template< unsigned int ... FEMDegrees > bool _isInteriorlySupported( UIntPack< FEMDegrees ... > , const FEMTreeNode* node ) const
		{
			if( !node ) return false;
			LocalDepth d ; LocalOffset off ; _localDepthAndOffset( node , d , off );
			return _IsInteriorlySupported< FEMDegrees ... >( UIntPack< FEMDegrees ... >() , d , off );
		}
		template< unsigned int FEMDegree1 , unsigned int FEMDegree2 > static bool _IsInteriorlyOverlapped( LocalDepth depth , const LocalOffset off )
		{
			if( depth>=0 )
			{
				int begin , end;
				BSplineIntegrationData< FEMDegreeAndBType< FEMDegree1 , BOUNDARY_NEUMANN >::Signature , FEMDegreeAndBType< FEMDegree2 , BOUNDARY_NEUMANN >::Signature >::InteriorOverlappedSpan( depth , begin , end );
				bool interior = true;
				for( int dd=0 ; dd<Dim ; dd++ ) interior &= off[dd]>=begin && off[dd]<end;
				return interior;
			}
			else return false;
		}

		template< unsigned int FEMDegree1 , unsigned int FEMDegree2 > bool _isInteriorlyOverlapped( const FEMTreeNode* node ) const
		{
			if( !node ) return false;
			LocalDepth d ; LocalOffset off;
			_localDepthAndOffset( node , d , off );
			return _IsInteriorlyOverlapped< FEMDegree1 , FEMDegree2 >( d , off );
		}
		template< unsigned int ... FEMDegrees1 , unsigned int ... FEMDegrees2 > static bool _IsInteriorlyOverlapped( UIntPack< FEMDegrees1 ... > , UIntPack< FEMDegrees2 ... > , LocalDepth depth , const LocalOffset off ){ return BaseFEMIntegrator::IsInteriorlyOverlapped( UIntPack< FEMDegrees1 ... >() , UIntPack< FEMDegrees2 ... >() , depth , off ); }
		template< unsigned int ... FEMDegrees1 , unsigned int ... FEMDegrees2 > bool _isInteriorlyOverlapped( UIntPack< FEMDegrees1 ... > , UIntPack< FEMDegrees2 ... > , const FEMTreeNode* node ) const
		{
			if( !node ) return false;
			LocalDepth d ; LocalOffset off ; _localDepthAndOffset( node , d , off );
			return _IsInteriorlyOverlapped( UIntPack< FEMDegrees1 ... >() , UIntPack< FEMDegrees2 ... >() , d , off );
		}

		void _startAndWidth( const FEMTreeNode* node , Point< Real , Dim >& start , Real& width ) const
		{
			LocalDepth d ; LocalOffset off;
			_localDepthAndOffset( node , d , off );
			if( d>=0 ) width = Real( 1.0 / (1<<  d ) );
			else       width = Real( 1.0 * (1<<(-d)) );
			for( int dd=0 ; dd<Dim ; dd++ ) start[dd] = Real( off[dd] ) * width;
		}

		void _centerAndWidth( const FEMTreeNode* node , Point< Real , Dim >& center , Real& width ) const
		{
			int d , off[Dim];
			_localDepthAndOffset( node , d , off );
			width = Real( 1.0 / (1<<d) );
			for( int dd=0 ; dd<Dim ; dd++ ) center[dd] = Real( off[dd] + 0.5 ) * width;
		}

		int _childIndex( const FEMTreeNode* node , Point< Real , Dim > p ) const
		{
			Point< Real , Dim > c ; Real w;
			_centerAndWidth( node , c , w );
			int cIdx = 0;
			for( int d=0 ; d<Dim ; d++ ) if( p[d]>=c[d] ) cIdx |= (1<<d);
			return cIdx;
		}

		template< bool ThreadSafe , unsigned int ... Degrees > void _setFullDepth( UIntPack< Degrees ... > , Allocator< FEMTreeNode > *nodeAllocator , FEMTreeNode* node , LocalDepth depth );
		template< bool ThreadSafe , unsigned int ... Degrees > void _setFullDepth( UIntPack< Degrees ... > , Allocator< FEMTreeNode > *nodeAllocator , LocalDepth depth );
		template< unsigned int ... Degrees > LocalDepth _getFullDepth( UIntPack< Degrees ... > , const FEMTreeNode* node ) const;
		template< unsigned int ... Degrees > LocalDepth _getFullDepth( UIntPack< Degrees ... > , const LocalDepth depth , const LocalOffset begin , const LocalOffset end , const FEMTreeNode *node ) const;

		template< bool ThreadSafe , typename AddNodeFunctor , unsigned int ... Degrees > void _refine( UIntPack< Degrees ... > , Allocator< FEMTreeNode > *nodeAllocator , const AddNodeFunctor &addNodeFunctor , FEMTreeNode *node );
		template< bool ThreadSafe , typename AddNodeFunctor , unsigned int ... Degrees > void _refine( UIntPack< Degrees ... > , Allocator< FEMTreeNode > *nodeAllocator , const AddNodeFunctor &addNodeFunctor );

	public:
		template< unsigned int ... Degrees > LocalDepth getFullDepth( UIntPack< Degrees ... > ) const;
		template< unsigned int ... Degrees > LocalDepth getFullDepth( UIntPack< Degrees ... > , const LocalDepth depth , const LocalOffset begin , const LocalOffset end ) const;

		LocalDepth depth( const FEMTreeNode* node ) const { return _localDepth( node ); }
		void depthAndOffset( const FEMTreeNode* node , LocalDepth& depth , LocalOffset& offset ) const { _localDepthAndOffset( node , depth , offset ); }

		size_t nodesSize( void ) const { return _sNodes.size(); }
		node_index_type nodesBegin( LocalDepth d ) const { return _sNodes.begin( _localToGlobal( d ) ); }
		node_index_type nodesEnd  ( LocalDepth d ) const { return _sNodes.end  ( _localToGlobal( d ) ); }
		size_t nodesSize ( LocalDepth d ) const { return _sNodes.size ( _localToGlobal( d ) ); }
		node_index_type nodesBegin( LocalDepth d , int slice ) const { return _sNodes.begin( _localToGlobal( d ) , slice + _localInset( d ) ); }
		node_index_type nodesEnd  ( LocalDepth d , int slice ) const { return _sNodes.end  ( _localToGlobal( d ) , slice + _localInset( d ) ); }
		size_t nodesSize ( LocalDepth d , int slice ) const { return _sNodes.size ( _localToGlobal( d ) , slice + _localInset( d ) ); }
		const FEMTreeNode* node( node_index_type idx ) const { return _sNodes.treeNodes[idx]; }
		void centerAndWidth( node_index_type idx , Point< Real , Dim >& center , Real& width ) const { _centerAndWidth( _sNodes.treeNodes[idx] , center , width ); }
		void  startAndWidth( node_index_type idx , Point< Real , Dim >& center , Real& width ) const {  _startAndWidth( _sNodes.treeNodes[idx] , center , width ); }
		void centerAndWidth( const FEMTreeNode* node , Point< Real , Dim >& center , Real& width ) const { _centerAndWidth( node , center , width ); }
		void  startAndWidth( const FEMTreeNode* node , Point< Real , Dim >& start  , Real& width ) const {  _startAndWidth( node , start  , width ); }

	protected:
		/////////////////////////////////////
		// System construction code        //
		// MultiGridFEMTreeData.System.inl //
		/////////////////////////////////////
	public:
		template< unsigned int ... FEMSigs > void setMultiColorIndices( UIntPack< FEMSigs ... > , int depth , std::vector< std::vector< size_t > >& indices ) const;
	protected:
		template< unsigned int ... FEMSigs > void _setMultiColorIndices( UIntPack< FEMSigs ... > , node_index_type start , node_index_type end , std::vector< std::vector< size_t > >& indices ) const;

		struct _SolverStats
		{
			double constraintUpdateTime , systemTime , solveTime;
			double bNorm2 , inRNorm2 , outRNorm2;
		};

		// For some reason MSVC has trouble determining the template parameters when using:
		//		template< unsigned int Idx , unsigned int ... FEMSigs , typename ... InterpolationInfos >
		template< unsigned int Idx , typename UIntPackFEMSigs , typename StaticWindow , typename ConstNeighbors , typename PointEvaluator , typename ... InterpolationInfos >
		typename std::enable_if< ( Idx==sizeof...(InterpolationInfos) ) >::type _addPointValues( UIntPackFEMSigs , StaticWindow &pointValues , const ConstNeighbors &neighbors , const PointEvaluator &bsData , std::tuple< InterpolationInfos *... > interpolationInfos ) const {}
		template< unsigned int Idx , typename UIntPackFEMSigs , typename StaticWindow , typename ConstNeighbors , typename PointEvaluator , typename ... InterpolationInfos >
		typename std::enable_if< ( Idx <sizeof...(InterpolationInfos) ) >::type _addPointValues( UIntPackFEMSigs , StaticWindow &pointValues , const ConstNeighbors &neighbors , const PointEvaluator &bsData , std::tuple< InterpolationInfos *... > interpolationInfos ) const
		{
			_addPointValues( UIntPackFEMSigs() , pointValues , neighbors , bsData , std::get< Idx >( interpolationInfos ) );
			_addPointValues< Idx+1 >( UIntPackFEMSigs() , pointValues , neighbors , bsData , interpolationInfos );
		}
		template< unsigned int ... FEMSigs , typename T , unsigned int PointD >
		void _addPointValues( UIntPack< FEMSigs ... > , StaticWindow< Real , UIntPack< BSplineOverlapSizes< FEMSignature< FEMSigs >::Degree >::OverlapSize ... > >& pointValues , const typename FEMTreeNode::template ConstNeighbors< UIntPack< BSplineOverlapSizes< FEMSignature< FEMSigs >::Degree >::OverlapSize ... > >& neighbors , const PointEvaluator< UIntPack< FEMSigs ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >& bsData , const InterpolationInfo< T , PointD >* interpolationInfo ) const;

		template< unsigned int Idx , typename UIntPackFEMSigs , typename WindowSlice , typename ConstNeighbors , typename PointEvaluator , typename ... InterpolationInfos >
		typename std::enable_if< ( Idx==sizeof...(InterpolationInfos) ) >::type _addProlongedPointValues( UIntPackFEMSigs , WindowSlice pointValues , const ConstNeighbors &neighbors , const ConstNeighbors &pNeighbors , const PointEvaluator &bsData , std::tuple< InterpolationInfos *... > interpolationInfos ) const {}
		template< unsigned int Idx , typename UIntPackFEMSigs , typename WindowSlice , typename ConstNeighbors , typename PointEvaluator , typename ... InterpolationInfos >
		typename std::enable_if< ( Idx <sizeof...(InterpolationInfos) ) >::type _addProlongedPointValues( UIntPackFEMSigs , WindowSlice pointValues , const ConstNeighbors &neighbors , const ConstNeighbors &pNeighbors , const PointEvaluator&bsData , std::tuple< InterpolationInfos *... > interpolationInfos ) const
		{
			_addProlongedPointValues( UIntPackFEMSigs() , pointValues , neighbors , pNeighbors , bsData , std::get< Idx >( interpolationInfos ) );
			_addProlongedPointValues< Idx+1 >( UIntPackFEMSigs() , pointValues , neighbors , pNeighbors , bsData , interpolationInfos );
		}
		template< unsigned int ... FEMSigs , typename T , unsigned int PointD >
		void _addProlongedPointValues( UIntPack< FEMSigs ... > , WindowSlice< Real , UIntPack< BSplineOverlapSizes< FEMSignature< FEMSigs >::Degree >::OverlapSize ... > > pointValues , const typename FEMTreeNode::template ConstNeighbors< UIntPack< BSplineOverlapSizes< FEMSignature< FEMSigs >::Degree >::OverlapSize ... > >& neighbors , const typename FEMTreeNode::template ConstNeighbors< UIntPack< BSplineOverlapSizes< FEMSignature< FEMSigs >::Degree >::OverlapSize ... > >& pNeighbors , const PointEvaluator< UIntPack< FEMSigs ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >& bsData , const InterpolationInfo< T , PointD >* iInfo ) const;

		template< unsigned int Idx , typename PointEvaluator , typename T , typename ... InterpolationInfos >
		typename std::enable_if< ( Idx==sizeof...(InterpolationInfos) ) >::type _setPointValuesFromProlongedSolution( LocalDepth highDepth , const PointEvaluator &bsData , ConstPointer( T ) prolongedSolution , std::tuple< InterpolationInfos *... > interpolationInfos ) const {}
		template< unsigned int Idx , typename PointEvaluator , typename T , typename ... InterpolationInfos >
		typename std::enable_if< ( Idx <sizeof...(InterpolationInfos) ) >::type _setPointValuesFromProlongedSolution( LocalDepth highDepth , const PointEvaluator &bsData , ConstPointer( T ) prolongedSolution , std::tuple< InterpolationInfos *... > interpolationInfos ) const
		{
			_setPointValuesFromProlongedSolution( highDepth , bsData , prolongedSolution , std::get< Idx >( interpolationInfos ) );
			_setPointValuesFromProlongedSolution< Idx+1 >( highDepth , bsData , prolongedSolution , interpolationInfos );
		}
		template< unsigned int ... FEMSigs , typename T , unsigned int PointD >
		void _setPointValuesFromProlongedSolution( LocalDepth highDepth , const PointEvaluator< UIntPack< FEMSigs ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >& bsData , ConstPointer( T ) prolongedSolution , InterpolationInfo< T , PointD >* interpolationInfo ) const;

		template< unsigned int Idx , typename ConstNeighbors , typename T , typename PointEvaluator , typename ... InterpolationInfos >
		typename std::enable_if< ( Idx==sizeof...(InterpolationInfos) ) , T >::type _getInterpolationConstraintFromProlongedSolution( const ConstNeighbors &neighbors , const FEMTreeNode* node , ConstPointer( T ) prolongedSolution , const PointEvaluator &bsData , std::tuple< InterpolationInfos ... > interpolationInfos ) const { return T{}; }
		template< unsigned int Idx , typename ConstNeighbors , typename T , typename PointEvaluator , typename ... InterpolationInfos >
		typename std::enable_if< ( Idx <sizeof...(InterpolationInfos) ) , T >::type _getInterpolationConstraintFromProlongedSolution( const ConstNeighbors &neighbors , const FEMTreeNode* node , ConstPointer( T ) prolongedSolution , const PointEvaluator &bsData , std::tuple< InterpolationInfos ... > interpolationInfos ) const
		{
			return
				_getInterpolationConstraintFromProlongedSolution( neighbors , node , prolongedSolution , bsData , std::get< Idx >( interpolationInfos ) ) +
				_getInterpolationConstraintFromProlongedSolution< Idx+1 >( neighbors , node , prolongedSolution , bsData , interpolationInfos );
		}
		template< unsigned int ... FEMSigs , typename T , unsigned int PointD >
		T _getInterpolationConstraintFromProlongedSolution( const typename FEMTreeNode::template ConstNeighbors< UIntPack< BSplineOverlapSizes< FEMSignature< FEMSigs >::Degree >::OverlapSize ... > >& neighbors , const FEMTreeNode* node , ConstPointer( T ) prolongedSolution , const PointEvaluator< UIntPack< FEMSigs ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >& bsData , const InterpolationInfo< T , PointD >* iInfo ) const;

		template< unsigned int Idx , typename PointEvaluator , typename T , typename ... InterpolationInfos >
		typename std::enable_if< ( Idx==sizeof...(InterpolationInfos) ) >::type _updateRestrictedInterpolationConstraints( const PointEvaluator &bsData , LocalDepth highDepth , ConstPointer( T ) solution , Pointer( T ) cumulativeConstraints , std::tuple< InterpolationInfos ... > interpolationInfos ) const {}
		template< unsigned int Idx , typename PointEvaluator , typename T , typename ... InterpolationInfos >
		typename std::enable_if< ( Idx <sizeof...(InterpolationInfos) ) >::type _updateRestrictedInterpolationConstraints( const PointEvaluator &bsData , LocalDepth highDepth , ConstPointer( T ) solution , Pointer( T ) cumulativeConstraints , std::tuple< InterpolationInfos ... > interpolationInfos ) const
		{
			_updateRestrictedInterpolationConstraints( bsData , highDepth , solution , cumulativeConstraints , std::get< Idx >( interpolationInfos ) );
			_updateRestrictedInterpolationConstraints< Idx+1 >( bsData , highDepth , solution , cumulativeConstraints , interpolationInfos );
		}
		template< unsigned int ... FEMSigs , typename T , unsigned int PointD >
		void _updateRestrictedInterpolationConstraints( const PointEvaluator< UIntPack< FEMSigs ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >& bsData , LocalDepth highDepth , ConstPointer( T ) solution , Pointer( T ) cumulativeConstraints , const InterpolationInfo< T , PointD >* interpolationInfo ) const;

		template< unsigned int FEMDegree1 , unsigned int FEMDegree2 > static void _SetParentOverlapBounds( const FEMTreeNode* node , int start[Dim] , int end[Dim] );
		template< unsigned int FEMDegree1 , unsigned int FEMDegree2 > static void _SetParentOverlapBounds( int cIdx , int start[Dim] , int end[Dim] );
		template< unsigned int ... FEMDegrees1 , unsigned int ... FEMDegrees2 > static void _SetParentOverlapBounds( UIntPack< FEMDegrees1 ... > , UIntPack< FEMDegrees2 ... > , const FEMTreeNode* node , int start[Dim] , int end[Dim] )
		{
			if( node )
			{
				int d , off[Dim] ; node->depthAndOffset( d , off );
				BaseFEMIntegrator::ParentOverlapBounds( UIntPack< FEMDegrees1 ... >() , UIntPack< FEMDegrees2 ... >() , d , off , start , end );
			}
		}
		template< unsigned int ... FEMDegrees1 , unsigned int ... FEMDegrees2 > static void _SetParentOverlapBounds( UIntPack< FEMDegrees1 ... > , UIntPack< FEMDegrees2 ... > , int cIdx , int start[Dim] , int end[Dim] )
		{
			BaseFEMIntegrator::ParentOverlapBounds( UIntPack< FEMDegrees1 ... >() , UIntPack< FEMDegrees2 ... >() , cIdx , start , end );
		}

		template< unsigned int ... FEMSigs >
		int _getProlongedMatrixRowSize( const FEMTreeNode* node , const typename FEMTreeNode::template ConstNeighbors< UIntPack< BSplineOverlapSizes< FEMSignature< FEMSigs >::Degree >::OverlapSize ... > >& pNeighbors ) const;
		template< typename T , unsigned int ... FEMSigs , typename ... InterpolationInfos , typename MatrixType >
		T _setMatrixRowAndGetConstraintFromProlongation( UIntPack< FEMSigs ... > , const BaseSystem< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& F , const typename FEMTreeNode::template ConstNeighbors< UIntPack< BSplineOverlapSizes< FEMSignature< FEMSigs >::Degree >::OverlapSize ... > >& pNeighbors , const typename FEMTreeNode::template ConstNeighbors< UIntPack< BSplineOverlapSizes< FEMSignature< FEMSigs >::Degree >::OverlapSize ... > >& neighbors , size_t idx , MatrixType &M , node_index_type offset , const PCStencils< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& pcStencils , const CCStencil< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& ccStencil , const PointEvaluator< UIntPack< FEMSigs ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >& bsData , ConstPointer( T ) prolongedSolution , std::tuple< InterpolationInfos *... > interpolationInfos ) const;
		template< typename T , unsigned int ... FEMSigs , typename ... InterpolationInfos >
		int _setProlongedMatrixRow( const typename BaseFEMIntegrator::System< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& F , const typename FEMTreeNode::template ConstNeighbors< UIntPack< BSplineOverlapSizes< FEMSignature< FEMSigs >::Degree >::OverlapSize ... > >& neighbors , const typename FEMTreeNode::template ConstNeighbors< UIntPack< BSplineOverlapSizes< FEMSignature< FEMSigs >::Degree >::OverlapSize ... > >& pNeighbors , Pointer( MatrixEntry< Real , matrix_index_type > ) row , node_index_type offset , const DynamicWindow< double , UIntPack< BSplineOverlapSizes< FEMSignature< FEMSigs >::Degree >::OverlapSize ... > >& stencil , const PointEvaluator< UIntPack< FEMSigs ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >& bsData , std::tuple< InterpolationInfos *... > interpolationInfos ) const;

		// Updates the constraints @(depth) based on the solution coefficients @(depth-1)
		template< unsigned int ... FEMSigs , typename T , typename ... InterpolationInfos >
		T _getConstraintFromProlongedSolution( UIntPack< FEMSigs ... > , const BaseSystem< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& F , const typename FEMTreeNode::template ConstNeighbors< UIntPack< BSplineOverlapSizes< FEMSignature< FEMSigs >::Degree >::OverlapSize ... > >& neighbors , const typename FEMTreeNode::template ConstNeighbors< UIntPack< BSplineOverlapSizes< FEMSignature< FEMSigs >::Degree >::OverlapSize ... > >& pNeighbors , const FEMTreeNode* node , ConstPointer( T ) prolongedSolution , const DynamicWindow< double , UIntPack< BSplineOverlapSizes< FEMSignature< FEMSigs >::Degree >::OverlapSize ... > >& stencil , const PointEvaluator< UIntPack< FEMSigs ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >& bsData , std::tuple< InterpolationInfos *... > interpolationInfos ) const;

		template< unsigned int ... FEMSigs , typename T , typename TDotT , typename SORWeights , typename ... InterpolationInfos >
		int _solveFullSystemGS( UIntPack< FEMSigs ... > , const typename BaseFEMIntegrator::System< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& F , const PointEvaluator< UIntPack< FEMSigs ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >& bsData , LocalDepth depth , Pointer( T ) solution , ConstPointer( T ) prolongedSolution , ConstPointer( T ) constraints , TDotT Dot , int iters , bool coarseToFine , SORWeights sorWeights , _SolverStats& stats , bool computeNorms , std::tuple< InterpolationInfos *... > interpolationInfos ) const;
		template< unsigned int ... FEMSigs , typename T , typename TDotT , typename SORWeights , typename ... InterpolationInfos >
		int _solveSlicedSystemGS( UIntPack< FEMSigs ... > , const typename BaseFEMIntegrator::System< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& F , const PointEvaluator< UIntPack< FEMSigs ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >& bsData , LocalDepth depth , Pointer( T ) solution , ConstPointer( T ) prolongedSolution , ConstPointer( T ) constraints , TDotT Dot , int iters , bool coarseToFine , unsigned int sliceBlockSize , SORWeights sorWeights , _SolverStats& stats , bool computeNorms , std::tuple< InterpolationInfos *... > interpolationInfos ) const;
		template< unsigned int ... FEMSigs , typename T , typename TDotT , typename SORWeights , typename ... InterpolationInfos >
		int _solveSystemGS( UIntPack< FEMSigs ... > , bool sliced , const typename BaseFEMIntegrator::System< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& F , const PointEvaluator< UIntPack< FEMSigs ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >& bsData , LocalDepth depth , Pointer( T ) solution , ConstPointer( T ) prolongedSolution , ConstPointer( T ) constraints , TDotT Dot , int iters , bool coarseToFine , unsigned int sliceBlockSize , SORWeights sorWeights , _SolverStats& stats , bool computeNorms , std::tuple< InterpolationInfos *... > interpolationInfos ) const
		{
			if( sliced ) return _solveSlicedSystemGS( UIntPack< FEMSigs ... >() , F , bsData , depth , solution , prolongedSolution , constraints , Dot , iters , coarseToFine , sliceBlockSize , sorWeights , stats , computeNorms , interpolationInfos );
			else         return _solveFullSystemGS  ( UIntPack< FEMSigs ... >() , F , bsData , depth , solution , prolongedSolution , constraints , Dot , iters , coarseToFine ,                  sorWeights , stats , computeNorms , interpolationInfos );
		}
		template< unsigned int ... FEMSigs , typename T , typename TDotT , typename ... InterpolationInfos >
		int _solveSystemCG( UIntPack< FEMSigs ... > , const typename BaseFEMIntegrator::System< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& F , const PointEvaluator< UIntPack< FEMSigs ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >& bsData , LocalDepth depth , Pointer( T ) solution , ConstPointer( T ) prolongedSolution , ConstPointer( T ) constraints , TDotT Dot , int iters , bool coarseToFine , _SolverStats& stats , bool computeNorms , double cgAccuracy , std::tuple< InterpolationInfos *... > interpolationInfos ) const;
		template< unsigned int ... FEMSigs , typename T , typename TDotT , typename ... InterpolationInfos >
		void _solveRegularMG( UIntPack< FEMSigs ... > , typename BaseFEMIntegrator::System< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& F , const PointEvaluator< UIntPack< FEMSigs ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >& bsData , LocalDepth depth , Pointer( T ) solution , ConstPointer( T ) constraints , TDotT Dot , int vCycles , int iters , _SolverStats& stats , bool computeNorms , double cgAccuracy , std::tuple< InterpolationInfos *... > interpolationInfos ) const;

		// Updates the cumulative integral constraints @(depth-1) based on the change in solution coefficients @(depth)
		template< unsigned int ... FEMSigs , typename T >
		void _updateRestrictedIntegralConstraints( UIntPack< FEMSigs ... > , const typename BaseFEMIntegrator::System< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& F , LocalDepth highDepth , ConstPointer( T ) solution , Pointer( T ) cumulativeConstraints ) const;

		template< unsigned int PointD , typename T , unsigned int ... FEMSigs >
		CumulativeDerivativeValues< T , Dim , PointD > _coarserFunctionValues( UIntPack< FEMSigs ... > , Point< Real , Dim > p , const ConstPointSupportKey< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& neighborKey , const FEMTreeNode* node , const PointEvaluator< UIntPack< FEMSigs ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >& bsData , ConstPointer( T ) coefficients ) const;
		template< unsigned int PointD , typename T , unsigned int ... FEMSigs >
		CumulativeDerivativeValues< T , Dim , PointD >   _finerFunctionValues( UIntPack< FEMSigs ... > , Point< Real , Dim > p , const ConstPointSupportKey< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& neighborKey , const FEMTreeNode* node , const PointEvaluator< UIntPack< FEMSigs ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >& bsData , ConstPointer( T ) coefficients ) const;

		template< unsigned int ... FEMSigs , typename T , typename ... InterpolationInfos >
		int _getSliceMatrixAndProlongationConstraints( UIntPack< FEMSigs ... > , const BaseSystem< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& F , SystemMatrixType< FEMSigs ... > &matrix , Pointer( Real ) diagonalR , const PointEvaluator< UIntPack< FEMSigs ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >& bsData , LocalDepth depth , node_index_type nBegin , node_index_type nEnd , ConstPointer( T ) prolongedSolution , Pointer( T ) constraints , const CCStencil < UIntPack< FEMSignature< FEMSigs >::Degree ... > >& ccStencil , const PCStencils< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& pcStencils , std::tuple< InterpolationInfos *... > interpolationInfos ) const;

		// Down samples constraints @(depth) to constraints @(depth-1)
		template< class C , typename ArrayWrapper , unsigned ... Degrees , unsigned int ... FEMSigs > void _downSample( UIntPack< FEMSigs ... > , typename BaseFEMIntegrator::template RestrictionProlongation< UIntPack< Degrees ... > >& RP , LocalDepth highDepth , ArrayWrapper finerConstraints , Pointer( C ) coarserConstraints ) const;
		// Up samples coefficients @(depth-1) to coefficients @(depth)
		template< class C , typename ArrayWrapper , unsigned ... Degrees , unsigned int ... FEMSigs > void   _upSample( UIntPack< FEMSigs ... > , typename BaseFEMIntegrator::template RestrictionProlongation< UIntPack< Degrees ... > >& RP , LocalDepth highDepth , ArrayWrapper coarserCoefficients , Pointer( C ) finerCoefficients ) const;

		template< unsigned int ... FEMSigs , typename ValidNodeFunctor >
		SparseMatrix< Real , matrix_index_type > _downSampleMatrix( UIntPack< FEMSigs ... > , LocalDepth highDepth , ValidNodeFunctor validNodeFunctor ) const;
		template< unsigned int ... FEMSigs , typename ValidNodeFunctor >
		SparseMatrix< Real , matrix_index_type > _upSampleMatrix( UIntPack< FEMSigs ... > , LocalDepth highDepth , ValidNodeFunctor validNodeFunctor ) const;
		template< unsigned int ... FEMSigs , typename ValidNodeFunctor >
		SparseMatrix< Real , matrix_index_type > _restrictSystemMatrix( UIntPack< FEMSigs ... > , LocalDepth highDepth , const SparseMatrix< Real , matrix_index_type > &M , ValidNodeFunctor validNodeFunctor ) const;

		template< bool XMajor , class C , unsigned int ... FEMSigs > static void _RegularGridUpSample( UIntPack< FEMSigs ... > ,                                                                                           LocalDepth highDepth , ConstPointer( C ) lowCoefficients , Pointer( C ) highCoefficients );
		template< bool XMajor , class C , unsigned int ... FEMSigs > static void _RegularGridUpSample( UIntPack< FEMSigs ... > , const int lowBegin[] , const int lowEnd[] , const int highBegin[] , const int highEnd[] , LocalDepth highDepth , ConstPointer( C ) lowCoefficients , Pointer( C ) highCoefficients );
	public:
		template< class C , unsigned int ... FEMSigs > DenseNodeData< C , UIntPack< FEMSigs ... > > coarseCoefficients( const  DenseNodeData< C , UIntPack< FEMSigs ... > >& coefficients ) const;
		template< class C , unsigned int ... FEMSigs > DenseNodeData< C , UIntPack< FEMSigs ... > > coarseCoefficients( const SparseNodeData< C , UIntPack< FEMSigs ... > >& coefficients ) const;
		template< class C , unsigned int ... FEMSigs > DenseNodeData< C , UIntPack< FEMSigs ... > > denseCoefficients( const SparseNodeData< C , UIntPack< FEMSigs ... > >& coefficients ) const;

		void trimToDepth( LocalDepth coarseDepth );

		template< class C , unsigned int ... FEMSigs >  DenseNodeData< C , UIntPack< FEMSigs ... > > trimToDepth( const  DenseNodeData< C , UIntPack< FEMSigs ... > >& coefficients , LocalDepth coarseDepth ) const;
		template< class C , unsigned int ... FEMSigs > SparseNodeData< C , UIntPack< FEMSigs ... > > trimToDepth( const SparseNodeData< C , UIntPack< FEMSigs ... > >& coefficients , LocalDepth coarseDepth ) const;

		template< typename T , unsigned int PointD , typename ConstraintDual , typename SystemDual >
		ApproximatePointInterpolationInfo< T , PointD , ConstraintDual , SystemDual > trimToDepth( const ApproximatePointInterpolationInfo< T , PointD , ConstraintDual , SystemDual > &iInfo , LocalDepth coarseDepth ) const;
		template< typename T , typename Data , unsigned int PointD , typename ConstraintDual , typename SystemDual >
		ApproximatePointAndDataInterpolationInfo< T , Data , PointD , ConstraintDual , SystemDual > trimToDepth( const ApproximatePointAndDataInterpolationInfo< T , Data , PointD , ConstraintDual , SystemDual > &iInfo , LocalDepth coarseDepth ) const;

		// For each (valid) fem node, compute the ratio of the sum of active prolongation weights to the sum of total prolongation weights
		// If the prolongToChildren flag is set, then these weights are pushed to the children by computing the ratio of the prolongation of the above weights to the prolongation of unity weights 

		template< unsigned int ... FEMSigs > DenseNodeData< Real , UIntPack< FEMSigs ... > > supportWeights( UIntPack< FEMSigs ... > ) const;
		template< unsigned int ... FEMSigs > DenseNodeData< Real , UIntPack< FEMSigs ... > > prolongationWeights( UIntPack< FEMSigs ... > , bool prolongToChildren ) const;

	protected:

		//////////////////////////////////////////////
		// Code for splatting point-sample data     //
		// MultiGridFEMTreeData.WeightedSamples.inl //
		//////////////////////////////////////////////
		template< unsigned int CoDim , unsigned int Degree >
		Real _GetScaleValue( unsigned int res ) const;
		template< unsigned int CoDim , unsigned int Degree >
		Real _GetScaleValue( Point< Real , Dim > p ) const;
		template< bool ThreadSafe , unsigned int CoDim , unsigned int WeightDegree >
		void _addWeightContribution( Allocator< FEMTreeNode > *nodeAllocator , DensityEstimator< WeightDegree >& densityWeights , FEMTreeNode* node , Point< Real , Dim > position , PointSupportKey< IsotropicUIntPack< Dim , WeightDegree > >& weightKey , Real weight=Real(1.0) );
		template< unsigned int WeightDegree , class PointSupportKey >
		Real _getSamplesPerNode( const DensityEstimator< WeightDegree >& densityWeights , const FEMTreeNode* node , Point< Real , Dim > position , PointSupportKey& weightKey ) const;
		template< unsigned int WeightDegree , class WeightKey >
		void _getSampleDepthAndWeight( const DensityEstimator< WeightDegree >& densityWeights , const FEMTreeNode* node , Point< Real , Dim > position , WeightKey& weightKey , Real& depth , Real& weight ) const;
		template< unsigned int WeightDegree , class WeightKey >
		void _getSampleDepthAndWeight( const DensityEstimator< WeightDegree >& densityWeights ,                           Point< Real , Dim > position , WeightKey& weightKey , Real& depth , Real& weight ) const;

		template< bool CreateNodes , bool ThreadSafe ,                             class V , unsigned int ... DataSigs > void                   _splatPointData( V zero , Allocator< FEMTreeNode > *nodeAllocator ,                                                          FEMTreeNode* node   , Point< Real , Dim > point , V v , SparseNodeData< V , UIntPack< DataSigs ... > >& data ,                                                                         PointSupportKey< UIntPack< FEMSignature< DataSigs >::Degree ... > >& dataKey );
		template< bool CreateNodes , bool ThreadSafe , unsigned int WeightDegree , class V , unsigned int ... DataSigs > Point< Real , 2 >      _splatPointData( V zero , Allocator< FEMTreeNode > *nodeAllocator , const DensityEstimator< WeightDegree >& densityWeights , Real minDepthCutoff , Point< Real , Dim > point , V v , SparseNodeData< V , UIntPack< DataSigs ... > >& data , PointSupportKey< IsotropicUIntPack< Dim , WeightDegree > >& weightKey , PointSupportKey< UIntPack< FEMSignature< DataSigs >::Degree ... > >& dataKey , LocalDepth minDepth , LocalDepth maxDepth , int dim , Real depthBias );
		template< bool CreateNodes , bool ThreadSafe , unsigned int WeightDegree , class V , unsigned int ... DataSigs > Point< Real , 2 >      _splatPointData( V zero , Allocator< FEMTreeNode > *nodeAllocator , const DensityEstimator< WeightDegree >& densityWeights , Real minDepthCutoff , Point< Real , Dim > point , V v , SparseNodeData< V , UIntPack< DataSigs ... > >& data , PointSupportKey< IsotropicUIntPack< Dim , WeightDegree > >& weightKey , PointSupportKey< UIntPack< FEMSignature< DataSigs >::Degree ... > >& dataKey , LocalDepth minDepth , std::function< int ( Point< Real , Dim > ) > &pointDepthFunctor , int dim , Real depthBias );
		template< bool CreateNodes , bool ThreadSafe , unsigned int WeightDegree , class V , unsigned int ... DataSigs > Point< Real , 2 > _multiSplatPointData( V zero , Allocator< FEMTreeNode > *nodeAllocator , const DensityEstimator< WeightDegree >* densityWeights , FEMTreeNode* node   , Point< Real , Dim > point , V v , SparseNodeData< V , UIntPack< DataSigs ... > >& data , PointSupportKey< IsotropicUIntPack< Dim , WeightDegree > >& weightKey , PointSupportKey< UIntPack< FEMSignature< DataSigs >::Degree ... > >& dataKey , int dim );
		template<                                      unsigned int WeightDegree , class V , unsigned int ... DataSigs > Real       _nearestMultiSplatPointData( V zero ,                                           const DensityEstimator< WeightDegree >* densityWeights , FEMTreeNode* node   , Point< Real , Dim > point , V v , SparseNodeData< V , UIntPack< DataSigs ... > >& data , PointSupportKey< IsotropicUIntPack< Dim , WeightDegree > >& weightKey , int dim=Dim );
		template< class V , class Coefficients , unsigned int D , unsigned int ... DataSigs >
		void _addEvaluation( const Coefficients& coefficients , Point< Real , Dim > p ,                         const PointEvaluator< UIntPack< DataSigs ... > , IsotropicUIntPack< Dim , D > >& pointEvaluator , const ConstPointSupportKey< UIntPack< FEMSignature< DataSigs >::Degree ... > >& dataKey , V &value ) const;
		template< class V , class Coefficients , unsigned int D , unsigned int ... DataSigs >
		void _addEvaluation( const Coefficients& coefficients , Point< Real , Dim > p , LocalDepth pointDepth , const PointEvaluator< UIntPack< DataSigs ... > , IsotropicUIntPack< Dim , D > >& pointEvaluator , const ConstPointSupportKey< UIntPack< FEMSignature< DataSigs >::Degree ... > >& dataKey , V &value ) const;
		template< typename V , typename Coefficients , unsigned int D , typename AccumulationFunctor /*=std::function< void ( const V & , Real s ) > */ , unsigned int ... DataSigs >
		void _accumulate( const Coefficients& coefficients , Point< Real , Dim > p ,                         const PointEvaluator< UIntPack< DataSigs ... > , IsotropicUIntPack< Dim , D > >& pointEvaluator , const ConstPointSupportKey< UIntPack< FEMSignature< DataSigs >::Degree ... > >& dataKey , AccumulationFunctor &AF ) const;
		template< typename V , typename Coefficients , unsigned int D , typename AccumulationFunctor /*=std::function< void ( const V & , Real s ) > */ , unsigned int ... DataSigs >
		void _accumulate( const Coefficients& coefficients , Point< Real , Dim > p , LocalDepth pointDepth , const PointEvaluator< UIntPack< DataSigs ... > , IsotropicUIntPack< Dim , D > >& pointEvaluator , const ConstPointSupportKey< UIntPack< FEMSignature< DataSigs >::Degree ... > >& dataKey , AccumulationFunctor &AF ) const;

	public:

		template< bool XMajor , class V , unsigned int ... DataSigs > Pointer( V ) regularGridEvaluate( const DenseNodeData< V , UIntPack< DataSigs ... > >& coefficients , int& res , LocalDepth depth=-1 , bool primal=false ) const;
		template< bool XMajor , class V , unsigned int ... DataSigs > Pointer( V ) regularGridEvaluate( const DenseNodeData< V , UIntPack< DataSigs ... > >& coefficients , const unsigned int begin[Dim] , const unsigned int end[Dim] , unsigned int res[Dim] , bool primal=false ) const;
		template< bool XMajor , class V , unsigned int ... DataSigs > Pointer( V ) regularGridUpSample( const DenseNodeData< V , UIntPack< DataSigs ... > >& coefficients , LocalDepth depth=-1 ) const;
		template< bool XMajor , class V , unsigned int ... DataSigs > Pointer( V ) regularGridUpSample( const DenseNodeData< V , UIntPack< DataSigs ... > >& coefficients , const int begin[Dim] , const int end[Dim] , LocalDepth depth=-1 ) const;
		template< class V , unsigned int ... DataSigs > V average( const DenseNodeData< V , UIntPack< DataSigs ... > >& coefficients ) const;
		template< class V , unsigned int ... DataSigs > V average( const DenseNodeData< V , UIntPack< DataSigs ... > >& coefficients , const Real begin[Dim] , const Real end[Dim] ) const;
		template< typename T > struct HasNormalDataFunctor{};
		template< unsigned int ... NormalSigs >
		struct HasNormalDataFunctor< UIntPack< NormalSigs ... > >
		{
			const SparseNodeData< Point< Real , Dim > , UIntPack< NormalSigs ... > >& normalInfo;
			HasNormalDataFunctor( const SparseNodeData< Point< Real , Dim > , UIntPack< NormalSigs ... > >& ni ) : normalInfo( ni ){ ; }
			bool operator() ( const FEMTreeNode* node ) const
			{
				const Point< Real , Dim >* n = normalInfo( node );
				if( n )
				{
					const Point< Real , Dim >& normal = *n;
					for( int d=0 ; d<Dim ; d++ ) if( normal[d]!=0 ) return true;
				}
				if( node->children ) for( int c=0 ; c<(1<<Dim) ; c++ ) if( (*this)( node->children + c ) ) return true;
				return false;
			}
		};
		struct TrivialHasDataFunctor{ bool operator() ( const FEMTreeNode* node ) const { return true; } };
	protected:
		// [NOTE] The input/output for this method is pre-scaled by weight
		template< typename T > bool _setInterpolationInfoFromChildren( FEMTreeNode* node , SparseNodeData< T , IsotropicUIntPack< Dim , FEMTrivialSignature > >& iInfo ) const;
		template< typename T ,                 unsigned int PointD , typename ConstraintDual > SparseNodeData< DualPointInfo       < Dim , Real ,        T , PointD > , IsotropicUIntPack< Dim , FEMTrivialSignature > > _densifyInterpolationInfoAndSetDualConstraints( const std::vector< PointSample >& samples ,                                   ConstraintDual constraintDual , LocalDepth maxDepth , int adaptiveExponent ) const;
		template< typename T , typename Data , unsigned int PointD , typename ConstraintDual > SparseNodeData< DualPointAndDataInfo< Dim , Real , Data , T , PointD > , IsotropicUIntPack< Dim , FEMTrivialSignature > > _densifyInterpolationInfoAndSetDualConstraints( const std::vector< PointSample >& samples , ConstPointer( Data ) sampleData , ConstraintDual constraintDual , LocalDepth maxDepth , int adaptiveExponent ) const;
		template< typename T ,                 unsigned int PointD , typename ConstraintDual > void _densifyInterpolationInfoAndSetDualConstraints( SparseNodeData< DualPointInfo       < Dim , Real ,        T , PointD > , IsotropicUIntPack< Dim , FEMTrivialSignature > > &iInfo , const std::vector< PointSample >& samples ,                                   ConstraintDual constraintDual , LocalDepth maxDepth , int adaptiveExponent ) const;
		template< typename T , typename Data , unsigned int PointD , typename ConstraintDual > void _densifyInterpolationInfoAndSetDualConstraints( SparseNodeData< DualPointAndDataInfo< Dim , Real , Data , T , PointD > , IsotropicUIntPack< Dim , FEMTrivialSignature > > &iInfo , const std::vector< PointSample >& samples , ConstPointer( Data ) sampleData , ConstraintDual constraintDual , LocalDepth maxDepth , int adaptiveExponent ) const;
		template< typename T ,                 unsigned int PointD , typename ConstraintDual > SparseNodeData< DualPointInfoBrood       < Dim , Real ,        T , PointD > , IsotropicUIntPack< Dim , FEMTrivialSignature > > _densifyChildInterpolationInfoAndSetDualConstraints( const std::vector< PointSample >& samples ,                                   ConstraintDual constraintDual , bool noRescale ) const;
		template< typename T , typename Data , unsigned int PointD , typename ConstraintDual > SparseNodeData< DualPointAndDataInfoBrood< Dim , Real , Data , T , PointD > , IsotropicUIntPack< Dim , FEMTrivialSignature > > _densifyChildInterpolationInfoAndSetDualConstraints( const std::vector< PointSample >& samples , ConstPointer( Data ) sampleData , ConstraintDual constraintDual , bool noRescale ) const;

		void _setSpaceValidityFlags( void ) const;
		template< unsigned int ... FEMSigs1 > void _setFEM1ValidityFlags( UIntPack< FEMSigs1 ... > ) const;
		template< unsigned int ... FEMSigs2 > void _setFEM2ValidityFlags( UIntPack< FEMSigs2 ... > ) const;
		template< class HasDataFunctor > void _clipTree( const HasDataFunctor& f , LocalDepth fullDepth );

	public:

		template< unsigned int PointD , unsigned int ... FEMSigs > SparseNodeData< CumulativeDerivativeValues< Real , Dim , PointD > , IsotropicUIntPack< Dim , FEMTrivialSignature > > leafValues( const DenseNodeData< Real , UIntPack< FEMSigs ... > >& coefficients , int maxDepth=-1 ) const;

	protected:

		/////////////////////////////////////
		// Evaluation Methods              //
		// MultiGridFEMTreeData.Evaluation //
		/////////////////////////////////////
		static const unsigned int CHILDREN = 1<<Dim;
		template< typename Pack , unsigned int PointD > struct _Evaluator{ };
		template< unsigned int ... FEMSigs , unsigned int PointD >
		struct _Evaluator< UIntPack< FEMSigs ... > , PointD >
		{
			static_assert( Dim == sizeof...(FEMSigs) , "[ERROR] Number of signatures doesn't match dimension" );

			typedef DynamicWindow< CumulativeDerivativeValues< double , Dim , PointD > , UIntPack< BSplineSupportSizes< FEMSignature< FEMSigs >::Degree >::SupportSize ... > > CenterStencil;
			typedef DynamicWindow< CumulativeDerivativeValues< double , Dim , PointD > , UIntPack< BSplineSupportSizes< FEMSignature< FEMSigs >::Degree >::SupportSize ... > > CornerStencil;
			typedef DynamicWindow< CumulativeDerivativeValues< double , Dim , PointD > , UIntPack< ( BSplineSupportSizes< FEMSignature< FEMSigs >::Degree >::BCornerSize + 1 ) ... > > BCornerStencil;

			typedef std::tuple< typename BSplineEvaluationData< FEMSigs >::template      Evaluator< PointD > ... >      Evaluators;
			typedef std::tuple< typename BSplineEvaluationData< FEMSigs >::template ChildEvaluator< PointD > ... > ChildEvaluators;
			struct StencilData
			{
				CenterStencil ccCenterStencil , pcCenterStencils[CHILDREN];
				CornerStencil ccCornerStencil[CHILDREN] , pcCornerStencils[CHILDREN][CHILDREN];
				BCornerStencil ccBCornerStencil[CHILDREN] , pcBCornerStencils[CHILDREN][CHILDREN];
			};
			Pointer( StencilData ) stencilData;
			Pointer(      Evaluators )      evaluators;
			Pointer( ChildEvaluators ) childEvaluators;

			void set( LocalDepth depth );
			_Evaluator( void ){ _pointEvaluator = NULL ; stencilData = NullPointer( StencilData ) , evaluators = NullPointer( Evaluators ) , childEvaluators = NullPointer( ChildEvaluators ); }
			~_Evaluator( void ){ if( _pointEvaluator ) delete _pointEvaluator , _pointEvaluator = NULL ; if( stencilData ) DeletePointer( stencilData ) ; if( evaluators ) DeletePointer( evaluators ) ; if( childEvaluators ) DeletePointer( childEvaluators ); }
		protected:
			enum _CenterOffset{ CENTER=-1 , BACK=0 , FRONT=1 };
			template< unsigned int _PointD=PointD > CumulativeDerivativeValues< double , Dim , _PointD >       _values( unsigned int d , const int fIdx[Dim] , const int idx[Dim] , const _CenterOffset off[Dim] , bool parentChild ) const;
			template< unsigned int _PointD=PointD > CumulativeDerivativeValues< double , Dim , _PointD > _centerValues( unsigned int d , const int fIdx[Dim] , const int idx[Dim] ,                                bool parentChild ) const;
			template< unsigned int _PointD=PointD > CumulativeDerivativeValues< double , Dim , _PointD > _cornerValues( unsigned int d , const int fIdx[Dim] , const int idx[Dim] , int corner ,                   bool parentChild ) const;
			template< unsigned int _PointD=PointD , unsigned int I=0 > typename std::enable_if< I==Dim >::type _setDValues( unsigned int d , const int fIdx[] , const int cIdx[] , const _CenterOffset off[] , bool pc , double dValues[][_PointD+1] ) const{ }
			template< unsigned int _PointD=PointD , unsigned int I=0 > typename std::enable_if< (I<Dim) >::type _setDValues( unsigned int d , const int fIdx[] , const int cIdx[] , const _CenterOffset off[] , bool pc , double dValues[][_PointD+1] ) const
			{
				if( pc ) for( int dd=0 ; dd<=_PointD ; dd++ ) dValues[I][dd] = off[I]==CENTER ? std::get< I >( childEvaluators[d] ).centerValue( fIdx[I] , cIdx[I] , dd ) : std::get< I >( childEvaluators[d] ).cornerValue( fIdx[I] , cIdx[I]+off[I] , dd );
				else     for( int dd=0 ; dd<=_PointD ; dd++ ) dValues[I][dd] = off[I]==CENTER ? std::get< I >(      evaluators[d] ).centerValue( fIdx[I] , cIdx[I] , dd ) : std::get< I >(      evaluators[d] ).cornerValue( fIdx[I] , cIdx[I]+off[I] , dd );
				_setDValues< _PointD , I+1 >( d , fIdx , cIdx , off , pc , dValues );
			}

			template< unsigned int I=0 > typename std::enable_if< I==Dim >::type _setEvaluators( unsigned int maxDepth ){ }
			template< unsigned int I=0 > typename std::enable_if< (I<Dim) >::type _setEvaluators( unsigned int maxDepth )
			{
				static const unsigned int FEMSig = UIntPack< FEMSigs ... >::template Get< I >();
				for( unsigned int d=0 ; d<=maxDepth ; d++ ) BSplineEvaluationData< FEMSig >::     SetEvaluator( std::template get< I >(      evaluators[d] ) , d   );
				for( unsigned int d=1 ; d<=maxDepth ; d++ ) BSplineEvaluationData< FEMSig >::SetChildEvaluator( std::template get< I >( childEvaluators[d] ) , d-1 );
				_setEvaluators< I+1 >( maxDepth );
			}
			typename FEMIntegrator::template PointEvaluator< UIntPack< FEMSigs ... > , IsotropicUIntPack< Dim , PointD > >* _pointEvaluator;
			friend FEMTree;
		};

		template< class V , unsigned int _PointD , unsigned int ... FEMSigs , unsigned int PointD >
		CumulativeDerivativeValues< V , Dim , _PointD > _getCenterValues( const ConstPointSupportKey< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& neighborKey , const FEMTreeNode* node ,                         ConstPointer( V ) solution , ConstPointer( V ) coarseSolution , const _Evaluator< UIntPack< FEMSigs ... > , PointD >& evaluator , int maxDepth , bool isInterior ) const;
		template< class V , unsigned int _PointD , unsigned int ... FEMSigs , unsigned int PointD >
		CumulativeDerivativeValues< V , Dim , _PointD > _getCornerValues( const ConstPointSupportKey< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& neighborKey , const FEMTreeNode* node , int corner            , ConstPointer( V ) solution , ConstPointer( V ) coarseSolution , const _Evaluator< UIntPack< FEMSigs ... > , PointD >& evaluator , int maxDepth , bool isInterior ) const;
		template< class V , unsigned int _PointD , unsigned int ... FEMSigs , unsigned int PointD >
		CumulativeDerivativeValues< V , Dim , _PointD > _getValues      ( const ConstPointSupportKey< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& neighborKey , const FEMTreeNode* node , Point< Real , Dim > p , ConstPointer( V ) solution , ConstPointer( V ) coarseSolution , const _Evaluator< UIntPack< FEMSigs ... > , PointD >& evaluator , int maxDepth ) const;
		template< class V , unsigned int _PointD , unsigned int ... FEMSigs , unsigned int PointD >
		CumulativeDerivativeValues< V , Dim , _PointD > _getCornerValues( const ConstCornerSupportKey< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& neighborKey , const FEMTreeNode* node , int corner            , ConstPointer( V ) solution , ConstPointer( V ) coarseSolution , const _Evaluator< UIntPack< FEMSigs ... > , PointD >& evaluator , int maxDepth , bool isInterior ) const;
		template< unsigned int ... SupportSizes >
		struct CornerLoopData
		{
			typedef UIntPack< SupportSizes ... > _SupportSizes;
			//		static const unsigned int supportSizes[] = { SupportSizes ... };
			static const unsigned int supportSizes[];
			unsigned int ccSize[1<<Dim] , pcSize[1<<Dim][1<<Dim];
			unsigned int ccIndices[1<<Dim]        [ WindowSize< _SupportSizes >::Size ];
			unsigned int pcIndices[1<<Dim][1<<Dim][ WindowSize< _SupportSizes >::Size ];
			CornerLoopData( void )
			{
				int start[Dim] , end[Dim] , _start[Dim] , _end[Dim];
				for( int c=0 ; c<(1<<Dim) ; c++ )
				{
					ccSize[c] = 0;
					for( int dd=0 ; dd<Dim ; dd++ ) 
					{
						start[dd] = 0 , end[dd] = supportSizes[dd];
						if( (c>>dd) & 1 ) start[dd]++;
						else              end  [dd]--;
					}
					unsigned int idx[Dim];
					WindowLoop< Dim >::Run
					(
						start , end ,
						[&]( int d , int i ){ idx[d] = i; } ,
						[&]( void ){ ccIndices[c][ ccSize[c]++ ] = GetWindowIndex( _SupportSizes() , idx ); }
					);

					for( int _c=0 ; _c<(1<<Dim) ; _c++ )
					{
						pcSize[c][_c] = 0;
						for( int dd=0 ; dd<Dim ; dd++ ) 
						{
							if( ( (_c>>dd) & 1 ) != ( (c>>dd) & 1 ) ) _start[dd] = 0 , _end[dd] = supportSizes[dd];
							else _start[dd] = start[dd] , _end[dd] = end[dd];
						}

						unsigned int idx[Dim];
						WindowLoop< Dim >::Run
						(
							_start , _end ,
							[&]( int d , int i ){ idx[d] = i; } ,
							[&]( void ){ pcIndices[c][_c][ pcSize[c][_c]++ ] = GetWindowIndex( _SupportSizes() , idx ); }
						);
					}
				}
			}
		};
	public:
		template< typename Pack , unsigned int PointD , typename T > struct _MultiThreadedEvaluator{ };
		template< unsigned int ... FEMSigs , unsigned int PointD , typename T >
		struct _MultiThreadedEvaluator< UIntPack< FEMSigs ... > , PointD , T >
		{
			typedef UIntPack< FEMSigs ... > FEMSignatures;
			typedef UIntPack< FEMSignature< FEMSigs >::Degree ... > FEMDegrees;
			const FEMTree* _tree;
			int _threads;
			std::vector< ConstPointSupportKey< FEMDegrees > > _pointNeighborKeys;
			std::vector< ConstCornerSupportKey< FEMDegrees > > _cornerNeighborKeys;
			_Evaluator< FEMSignatures , PointD > _evaluator;
			const DenseNodeData< T , FEMSignatures >& _coefficients;
			DenseNodeData< T , FEMSignatures > _coarseCoefficients;
		public:
			_MultiThreadedEvaluator( const FEMTree* tree , const DenseNodeData< T , FEMSignatures >& coefficients , int threads=ThreadPool::NumThreads() );
			template< unsigned int _PointD=PointD > CumulativeDerivativeValues< T , Dim , _PointD > values( Point< Real , Dim > p , int thread=0 , const FEMTreeNode* node=NULL );
			template< unsigned int _PointD=PointD > CumulativeDerivativeValues< T , Dim , _PointD > centerValues( const FEMTreeNode* node , int thread=0 );
			template< unsigned int _PointD=PointD > CumulativeDerivativeValues< T , Dim , _PointD > cornerValues( const FEMTreeNode* node , int corner , int thread=0 );
		};
		template< typename Pack , unsigned int PointD , typename T=Real > using MultiThreadedEvaluator = _MultiThreadedEvaluator< Pack , PointD , T >;
		template< unsigned int DensityDegree >
		struct MultiThreadedWeightEvaluator
		{
			const FEMTree* _tree;
			int _threads;
			std::vector< ConstPointSupportKey< IsotropicUIntPack< Dim , DensityDegree > > > _neighborKeys;
			const DensityEstimator< DensityDegree >& _density;
		public:
			MultiThreadedWeightEvaluator( const FEMTree* tree , const DensityEstimator< DensityDegree >& density , int threads=ThreadPool::NumThreads() );
			Real weight( Point< Real , Dim > p , int thread=0 );
		};

		template< typename Pack , typename T > struct MultiThreadedSparseEvaluator{ };
		template< unsigned int ... FEMSigs , typename T >
		struct MultiThreadedSparseEvaluator< UIntPack< FEMSigs ... > , T >
		{
			typedef UIntPack< FEMSigs ... > FEMSignatures;
			typedef UIntPack< FEMSignature< FEMSigs >::Degree ... > FEMDegrees;
			const FEMTree* _tree;
			int _threads;
			std::vector< ConstPointSupportKey< FEMDegrees > > _pointNeighborKeys;
			typename FEMIntegrator::template PointEvaluator< UIntPack< FEMSigs ... > , ZeroUIntPack< Dim > > *_pointEvaluator;
			const SparseNodeData< T , FEMSignatures >& _coefficients;
		public:
			MultiThreadedSparseEvaluator( const FEMTree* tree , const SparseNodeData< T , FEMSignatures >& coefficients , int threads=ThreadPool::NumThreads() );
			~MultiThreadedSparseEvaluator( void ){ if( _pointEvaluator ) delete _pointEvaluator; }
			void addValue( Point< Real , Dim > p , T &t , int thread=0 , const FEMTreeNode* node=NULL );
			template< typename AccumulationFunctor/*=std::function< void ( const T & , Real s ) > */ >
			void accumulate( Point< Real , Dim > p , AccumulationFunctor &Accumulate , int thread=0 , const FEMTreeNode* node=NULL );
		};

	protected:

		template< unsigned int Idx , typename ... DenseOrSparseNodeData >
		typename std::enable_if< ( Idx==sizeof...(DenseOrSparseNodeData) ) >::type _reorderDenseOrSparseNodeData( ConstPointer( node_index_type ) map , size_t sz , std::tuple< DenseOrSparseNodeData* ... > data ){}
		template< unsigned int Idx , typename ... DenseOrSparseNodeData >
		typename std::enable_if< ( Idx <sizeof...(DenseOrSparseNodeData) ) >::type _reorderDenseOrSparseNodeData( ConstPointer( node_index_type ) map , size_t sz , std::tuple< DenseOrSparseNodeData* ... > data )
		{
			if( std::get<Idx>( data ) ) std::get<Idx>( data )->_remapIndices( map , sz );
			_reorderDenseOrSparseNodeData< Idx+1 >( map , sz , data );
		}

		template< unsigned int Idx , typename ... InterpolationInfos >
		typename std::enable_if< ( Idx==sizeof...(InterpolationInfos) ) >::type _reorderInterpolationInfo( ConstPointer( node_index_type ) map , size_t sz , std::tuple< InterpolationInfos* ... > interpolationInfos ){}
		template< unsigned int Idx , typename ... InterpolationInfos >
		typename std::enable_if< ( Idx <sizeof...(InterpolationInfos) ) >::type _reorderInterpolationInfo( ConstPointer( node_index_type ) map , size_t sz , std::tuple< InterpolationInfos* ... > interpolationInfos )
		{
			if( std::get<Idx>( interpolationInfos ) ) std::get<Idx>( interpolationInfos )->_remapIndices( map , sz );
			_reorderInterpolationInfo< Idx+1 >( map , sz , interpolationInfos );
		}

	public:

		FEMTree( size_t blockSize );
		FEMTree( BinaryStream &stream , size_t blockSize );
		template< unsigned int CrossDegree , unsigned int Pad >
		static FEMTree< Dim , Real > *Slice( const FEMTree< Dim+1 , Real > &tree , unsigned int sliceDepth , unsigned int sliceIndex , bool includeBounds , size_t blockSize );

		static FEMTree< Dim , Real > *Merge( const FEMTree< Dim , Real > &tree1 , const FEMTree< Dim , Real > &tree2 , size_t blockSize );
		~FEMTree( void )
		{
			_tree.cleanChildren( !nodeAllocators.size() );
			for( size_t i=0 ; i<nodeAllocators.size() ; i++ ) delete nodeAllocators[i];
		}
		void write( BinaryStream &stream , bool serialize ) const;
		static void WriteParameter( BinaryStream &stream )
		{
			FEMTreeRealType realType;
			if     ( typeid( Real )==typeid( float  ) ) realType=FEM_TREE_REAL_FLOAT;
			else if( typeid( Real )==typeid( double ) ) realType=FEM_TREE_REAL_DOUBLE;
			else MK_THROW( "Unrecognized real type" );
			stream.write( realType );
			int dim = Dim;
			stream.write( dim );
		}

		template< unsigned int LeftRadius , unsigned int RightRadius , bool CreateNodes , typename ProcessingNodeFunctor , typename ... DenseOrSparseNodeData , typename InitializeFunctor > void processNeighbors( ProcessingNodeFunctor processNodes ,     std::tuple< DenseOrSparseNodeData *... > data , InitializeFunctor initialize );
		template< unsigned int LeftRadius , unsigned int RightRadius , bool CreateNodes ,                                  typename ... DenseOrSparseNodeData , typename InitializeFunctor > void processNeighbors( FEMTreeNode** nodes , size_t nodeCount , std::tuple< DenseOrSparseNodeData *... > data , InitializeFunctor initialize );
		template< unsigned int Radius                                , bool CreateNodes , typename ProcessingNodeFunctor , typename ... DenseOrSparseNodeData , typename InitializeFunctor > void processNeighbors( ProcessingNodeFunctor processNodes     , std::tuple< DenseOrSparseNodeData *... > data , InitializeFunctor initialize ){ processNeighbors< Radius , Radius , CreateNodes >( processNodes , data , initialize ); }
		template< unsigned int Radius                                , bool CreateNodes ,                                  typename ... DenseOrSparseNodeData , typename InitializeFunctor > void processNeighbors( FEMTreeNode** nodes , size_t nodeCount , std::tuple< DenseOrSparseNodeData *... > data , InitializeFunctor initialize ){ processNeighbors< Radius , Radius , CreateNodes >( nodes , nodeCount , data , initialize ); }

		template< unsigned int LeftRadius , unsigned int RightRadius , bool CreateNodes , typename ProcessingNodeFunctor , typename ... DenseOrSparseNodeData >                              void processNeighbors( ProcessingNodeFunctor processNodes     , std::tuple< DenseOrSparseNodeData *... > data                                ){ return processNeighbors< LeftRadius , RightRadius , CreateNodes >( processNodes , data , []( const FEMTreeNode * ){} ); }
		template< unsigned int LeftRadius , unsigned int RightRadius , bool CreateNodes ,                                  typename ... DenseOrSparseNodeData >                              void processNeighbors( FEMTreeNode** nodes , size_t nodeCount , std::tuple< DenseOrSparseNodeData *... > data                                ){ return processNeighbors< LeftRadius , RightRadius , CreateNodes >( nodes , nodeCount , data , []( const FEMTreeNode * ){} ); }
		template< unsigned int Radius                                , bool CreateNodes , typename ProcessingNodeFunctor , typename ... DenseOrSparseNodeData >                              void processNeighbors( ProcessingNodeFunctor processNodes     , std::tuple< DenseOrSparseNodeData *... > data                                ){ return processNeighbors< Radius , CreateNodes >( processNodes , data , []( const FEMTreeNode * ){} ); }
		template< unsigned int Radius                                , bool CreateNodes ,                                  typename ... DenseOrSparseNodeData >                              void processNeighbors( FEMTreeNode** nodes , size_t nodeCount , std::tuple< DenseOrSparseNodeData *... > data                                ){ return processNeighbors< Radius , CreateNodes >( nodes , nodeCount , data , []( const FEMTreeNode * ){} ) ; }

		template< unsigned int LeftRadius , unsigned int RightRadius , typename IsProcessingNodeFunctor , typename ProcessingKernel > void processNeighboringLeaves( IsProcessingNodeFunctor isProcessingNode , ProcessingKernel kernel , bool processSubTree );
		template< unsigned int LeftRadius , unsigned int RightRadius                                    , typename ProcessingKernel > void processNeighboringLeaves( FEMTreeNode** nodes , size_t nodeCount   , ProcessingKernel kernel , bool processSubTree );
		template< unsigned int Radius                                , typename IsProcessingNodeFunctor , typename ProcessingKernel > void processNeighboringLeaves( IsProcessingNodeFunctor isProcessingNode , ProcessingKernel kernel , bool processSubTree ){ return processNeighboringLeaves< Radius , Radius >( isProcessingNode , kernel , processSubTree ); }
		template< unsigned int Radius                                                                   , typename ProcessingKernel > void processNeighboringLeaves( FEMTreeNode** nodes , size_t nodeCount   , ProcessingKernel kernel , bool processSubTree ){ return processNeighboringLeaves< Radius , Radius >( nodes , nodeCount , kernel , processSubTree ); }

		template< unsigned int CoDim , unsigned int DensityDegree >
		typename FEMTree::template DensityEstimator< DensityDegree > *setDensityEstimator( const std::vector< PointSample >& samples , LocalDepth splatDepth , Real samplesPerNode );
		template< unsigned int CoDim , unsigned int DensityDegree >
		void updateDensityEstimator( typename FEMTree::template DensityEstimator< DensityDegree > &density , const std::vector< PointSample >& samples , LocalDepth minSplatDepth , LocalDepth maxSplatDepth );
		template< unsigned int CoDim , unsigned int DensityDegree >
		typename FEMTree::template DensityEstimator< DensityDegree > *setDensityEstimator( const std::vector< PointSample >& samples , LocalDepth splatDepth , std::function< int ( Point< Real , Dim > ) > pointDepthFunctor , Real samplesPerNode );
		template< unsigned int CoDim , unsigned int DensityDegree >
		void updateDensityEstimator( typename FEMTree::template DensityEstimator< DensityDegree > &density , const std::vector< PointSample >& samples , LocalDepth minSplatDepth , LocalDepth maxSplatDepth , std::function< int ( Point< Real , Dim > ) > pointDepthFunctor );

		template< unsigned int ... DataSigs , unsigned int DensityDegree , class InData , class OutData >
		SparseNodeData< OutData , UIntPack< DataSigs ... > > setInterpolatedDataField( OutData zero , UIntPack< DataSigs ... > , const std::vector< PointSample >& samples , const std::vector< InData >& data , const DensityEstimator< DensityDegree >* density , LocalDepth minDepth , LocalDepth maxDepth , Real minDepthCutoff , ProjectiveData< Point< Real , 2 > , Real > &pointDepthAndWeight , std::function< bool ( InData , OutData & , Real & ) > ConversionAndBiasFunction );
#if defined(_WIN32) || defined(_WIN64)
		template< unsigned int ... DataSigs , unsigned int DensityDegree , class InData , class OutData >
		SparseNodeData< OutData , UIntPack< DataSigs ... > > setInterpolatedDataField( OutData zero , UIntPack< DataSigs ... > , const std::vector< PointSample >& samples , const std::vector< InData >& data , const DensityEstimator< DensityDegree >* density , LocalDepth minDepth , LocalDepth maxDepth , Real minDepthCutoff , ProjectiveData< Point< Real , 2 > , Real > &pointDepthAndWeight , std::function< bool ( InData , OutData & ) > ConversionFunction , std::function< Real ( InData ) > BiasFunction = []( InData ){ return 0.f; } );
#else // !_WIN32 && !_WIN64
		template< unsigned int ... DataSigs , unsigned int DensityDegree , class InData , class OutData >
		SparseNodeData< OutData , UIntPack< DataSigs ... > > setInterpolatedDataField( OutData zero , UIntPack< DataSigs ... > , const std::vector< PointSample >& samples , const std::vector< InData >& data , const DensityEstimator< DensityDegree >* density , LocalDepth minDepth , LocalDepth maxDepth , Real minDepthCutoff , ProjectiveData< Point< Real , 2 > , Real > &pointDepthAndWeight , std::function< bool ( InData , OutData & ) > ConversionFunction , std::function< Real ( InData ) > BiasFunction = []( InData ){ return (Real)0; } );
#endif // _WIN32 || _WIN64
		template< unsigned int ... DataSigs , unsigned int DensityDegree , class InData , class OutData >
		SparseNodeData< OutData , UIntPack< DataSigs ... > > setInterpolatedDataField( OutData zero , UIntPack< DataSigs ... > , const std::vector< PointSample >& samples , const std::vector< InData >& data , const DensityEstimator< DensityDegree >* density , LocalDepth minDepth , LocalDepth maxDepth , std::function< int ( Point< Real , Dim > ) > pointDepthFunctor , Real minDepthCutoff , ProjectiveData< Point< Real , 2 > , Real > &pointDepthAndWeight , std::function< bool ( InData , OutData & , Real & ) > ConversionAndBiasFunction );
#if defined(_WIN32) || defined(_WIN64)
		template< unsigned int ... DataSigs , unsigned int DensityDegree , class InData , class OutData >
		SparseNodeData< OutData , UIntPack< DataSigs ... > > setInterpolatedDataField( OutData zero , UIntPack< DataSigs ... > , const std::vector< PointSample >& samples , const std::vector< InData >& data , const DensityEstimator< DensityDegree >* density , LocalDepth minDepth , LocalDepth maxDepth , std::function< int ( Point< Real , Dim > ) > pointDepthFunctor , Real minDepthCutoff , ProjectiveData< Point< Real , 2 > , Real > &pointDepthAndWeight , std::function< bool ( InData , OutData & ) > ConversionFunction , std::function< Real ( InData ) > BiasFunction = []( InData ){ return 0.f; } );
#else // !_WIN32 && !_WIN64
		template< unsigned int ... DataSigs , unsigned int DensityDegree , class InData , class OutData >
		SparseNodeData< OutData , UIntPack< DataSigs ... > > setInterpolatedDataField( OutData zero , UIntPack< DataSigs ... > , const std::vector< PointSample >& samples , const std::vector< InData >& data , const DensityEstimator< DensityDegree >* density , LocalDepth minDepth , LocalDepth maxDepth , std::function< int ( Point< Real , Dim > ) > pointDepthFunctor , Real minDepthCutoff , ProjectiveData< Point< Real , 2 > , Real > &pointDepthAndWeight , std::function< bool ( InData , OutData & ) > ConversionFunction , std::function< Real ( InData ) > BiasFunction = []( InData ){ return (Real)0; } );
#endif // _WIN32 || _WIN64

		template< unsigned int DataSig , bool CreateNodes , unsigned int DensityDegree , class Data >
		SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > setExtrapolatedDataField( Data zero , const std::vector< PointSample >& samples , const std::vector< Data >& sampleData , const DensityEstimator< DensityDegree >* density , bool nearest=false );
		template< unsigned int DataSig , bool CreateNodes , unsigned int DensityDegree , class Data >
		void updateExtrapolatedDataField( Data zero , SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > &dataField , const std::vector< PointSample >& samples , const std::vector< Data >& sampleData , const DensityEstimator< DensityDegree >* density , bool nearest=false );
		template< unsigned int DataSig , bool CreateNodes , unsigned int DensityDegree , class Data , class SampleFunctor /* = std::function< const PointSample & (size_t) >*/ , class SampleDataFunctor /* = std::function< const Data & (size_t) > */ >
		SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > setExtrapolatedDataField( Data zero , size_t sampleNum , SampleFunctor sampleFunctor , SampleDataFunctor sampleDataFunctor , const DensityEstimator< DensityDegree >* density , bool nearest=false );
		template< unsigned int DataSig , bool CreateNodes , unsigned int DensityDegree , class Data , class SampleFunctor /* = std::function< const PointSample & (size_t) >*/ , class SampleDataFunctor /* = std::function< const Data & (size_t) > */ >
		void updateExtrapolatedDataField( Data zero , SparseNodeData< ProjectiveData< Data , Real > , IsotropicUIntPack< Dim , DataSig > > &dataField , size_t sampleNum , SampleFunctor sampleFunctor , SampleDataFunctor sampleDataFunctor , const DensityEstimator< DensityDegree >* density , bool nearest=false );

		template< unsigned int MaxDegree , unsigned int SystemDegree , typename AddNodeFunctor , typename HasDataFunctor , typename IsDirichletLeafFunctor , typename ... InterpolationInfos , typename ... DenseOrSparseNodeData > std::vector< node_index_type > finalizeForMultigridWithDirichlet( LocalDepth baseDepth , const AddNodeFunctor addNodeFunctor , const HasDataFunctor hasDataFunctor , const IsDirichletLeafFunctor isDirichletLeafFunctor , std::tuple< InterpolationInfos *... > interpolationInfos , std::tuple< DenseOrSparseNodeData *... > data );
		template< unsigned int MaxDegree , unsigned int SystemDegree , typename AddNodeFunctor , typename HasDataFunctor , typename IsDirichletLeafFunctor , typename ... InterpolationInfos                                      > std::vector< node_index_type > finalizeForMultigridWithDirichlet( LocalDepth baseDepth , const AddNodeFunctor addNodeFunctor , const HasDataFunctor hasDataFunctor , const IsDirichletLeafFunctor isDirichletLeafFunctor , std::tuple< InterpolationInfos *... > interpolationInfos ){ return finalizeForMultigridWithDirichlet< MaxDegree , SystemDegree , AddNodeFunctor , HasDataFunctor >( baseDepth , addNodeFunctor , hasDataFunctor , isDirichletLeafFunctor , interpolationInfos , std::make_tuple() ); }

		template< unsigned int MaxDegree , unsigned int SystemDegree , typename AddNodeFunctor , typename HasDataFunctor , typename ... InterpolationInfos , typename ... DenseOrSparseNodeData > std::vector< node_index_type > finalizeForMultigrid( LocalDepth baseDepth , const AddNodeFunctor addNodeFunctor , const HasDataFunctor hasDataFunctor , std::tuple< InterpolationInfos *... > interpolationInfos , std::tuple< DenseOrSparseNodeData *... > data );
		template< unsigned int MaxDegree , unsigned int SystemDegree , typename AddNodeFunctor , typename HasDataFunctor , typename ... InterpolationInfos                                      > std::vector< node_index_type > finalizeForMultigrid( LocalDepth baseDepth , const AddNodeFunctor addNodeFunctor , const HasDataFunctor hasDataFunctor , std::tuple< InterpolationInfos *... > interpolationInfos ){ return finalizeForMultigrid< MaxDegree , SystemDegree , AddNodeFunctor , HasDataFunctor >( baseDepth , addNodeFunctor , hasDataFunctor , interpolationInfos , std::make_tuple() ); }

	protected:
		template< bool HasDirichlet , unsigned int MaxDegree , unsigned int SystemDegree , typename AddNodeFunctor , typename HasDataFunctor , typename IsDirichletLeafFunctor , typename ... InterpolationInfos , typename ... DenseOrSparseNodeData >
		std::vector< node_index_type > _finalizeForMultigrid( LocalDepth baseDepth , const AddNodeFunctor addNodeFunctor , const HasDataFunctor hasDataFunctor , const IsDirichletLeafFunctor isDirichletLeafFunctor , std::tuple< InterpolationInfos *... > interpolationInfos , std::tuple< DenseOrSparseNodeData *... > data );
	public:
		template< typename ... InterpolationInfos , typename ... DenseOrSparseNodeData > std::vector< node_index_type > setSortedTreeNodes( std::tuple< InterpolationInfos *... > interpolationInfos , std::tuple< DenseOrSparseNodeData *... > data );
		template< typename ... InterpolationInfos                                      > std::vector< node_index_type > setSortedTreeNodes( std::tuple< InterpolationInfos *... > interpolationInfos ){ return setSortedTreeNodes( interpolationInfos , std::make_tuple() ); }

		template< class ... DenseOrSparseNodeData > void resetIndices( std::tuple< DenseOrSparseNodeData *... > data );

		template< typename PruneChildrenFunctor , typename ... InterpolationInfos , typename ... DenseOrSparseNodeData >
		void pruneChildren( const PruneChildrenFunctor pruneChildren , std::tuple< InterpolationInfos *... > interpolationInfos , std::tuple< DenseOrSparseNodeData *... > data );

		template< unsigned int ... FEMSigs > DenseNodeData< Real , UIntPack< FEMSigs ... > > initDenseNodeData( UIntPack< FEMSigs ... > ) const;
		template< class Data , unsigned int ... FEMSigs > DenseNodeData< Data , UIntPack< FEMSigs ... > > initDenseNodeData( UIntPack< FEMSigs ... > ) const;

		template< unsigned int Pad , unsigned int FEMSig , unsigned int ... FEMSigs , typename Data >
		void slice( const FEMTree< Dim+1 , Real > &tree , unsigned int d , const DenseNodeData< Data , UIntPack< FEMSigs ... , FEMSig > > &coefficients , DenseNodeData< Data , UIntPack< FEMSigs ... > > &sliceCoefficients , unsigned int sliceDepth , unsigned int sliceIndex ) const;

		template< unsigned int ... FEMSigs , typename Data >
		void merge( const FEMTree< Dim , Real > &tree , const DenseNodeData< Data , UIntPack< FEMSigs ... > > &coefficients , DenseNodeData< Data , UIntPack< FEMSigs ... > > &mergedCoefficients ) const;

		// Add multiple-dimensions -> one-dimension constraints
		template< typename T , unsigned int ... FEMDegrees , unsigned int ... FEMSigs , unsigned int ... CDegrees , unsigned int ... CSigs , unsigned int CDim >
		void addFEMConstraints( typename BaseFEMIntegrator::template Constraint< UIntPack< FEMDegrees ... > , UIntPack< CDegrees ... > , CDim >& F , const _SparseOrDenseNodeData< Point< T , CDim > , UIntPack< CSigs ... > >& coefficients , DenseNodeData< T , UIntPack< FEMSigs ... > >& constraints , LocalDepth maxDepth ) const
		{
			typedef SparseNodeData< Point< T , CDim > , UIntPack< CSigs ... > > SparseType;
			typedef  DenseNodeData< Point< T , CDim > , UIntPack< CSigs ... > >  DenseType;
			static_assert( sizeof...( FEMDegrees )==Dim && sizeof...( FEMSigs )==Dim && sizeof...( CDegrees )==Dim && sizeof...( CSigs )==Dim , "[ERROR] Dimensions don't match" );
			static_assert( ParameterPack::Comparison< UIntPack< FEMDegrees ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >::Equal , "[ERROR] FEM signature and degrees don't match" );
			static_assert( ParameterPack::Comparison< UIntPack<   CDegrees ... > , UIntPack< FEMSignature<   CSigs >::Degree ... > >::Equal , "[ERROR] Constraint signature and degrees don't match" );
			if     ( typeid(coefficients)==typeid(SparseType) ) return _addFEMConstraints< T >( UIntPack< FEMSigs ... >() , UIntPack< CSigs ... >() , F , static_cast< const SparseType& >( coefficients ) , constraints() , maxDepth );
			else if( typeid(coefficients)==typeid( DenseType) ) return _addFEMConstraints< T >( UIntPack< FEMSigs ... >() , UIntPack< CSigs ... >() , F , static_cast< const  DenseType& >( coefficients ) , constraints() , maxDepth );
			else                                                return _addFEMConstraints< T >( UIntPack< FEMSigs ... >() , UIntPack< CSigs ... >() , F ,                                   coefficients   , constraints() , maxDepth );
		}
		// Add one-dimensions -> one-dimension constraints (with distinct signatures)
		template< typename T , unsigned int ... FEMDegrees , unsigned int ... FEMSigs , unsigned int ... CDegrees , unsigned int ... CSigs >
		void addFEMConstraints( typename BaseFEMIntegrator::template Constraint< UIntPack< FEMDegrees ... > , UIntPack< CDegrees ... > , 1 >& F , const _SparseOrDenseNodeData< T , UIntPack< CSigs ... > >& coefficients , DenseNodeData< T , UIntPack< FEMSigs ... > >& constraints , LocalDepth maxDepth ) const
		{
			typedef SparseNodeData< T , UIntPack< CSigs ... > > SparseType;
			typedef  DenseNodeData< T , UIntPack< CSigs ... > >  DenseType;
			static_assert( sizeof...( FEMDegrees )==Dim && sizeof...( FEMSigs )==Dim && sizeof...( CDegrees )==Dim && sizeof...( CSigs )==Dim  , "[ERROR] Dimensions don't match" );
			static_assert( ParameterPack::Comparison< UIntPack< FEMDegrees ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >::Equal , "[ERROR] FEM signature and degrees don't match" );
			static_assert( ParameterPack::Comparison< UIntPack<   CDegrees ... > , UIntPack< FEMSignature<   CSigs >::Degree ... > >::Equal , "[ERROR] Constaint signature and degrees don't match" );
			if     ( typeid(coefficients)==typeid(SparseType) ) return _addFEMConstraints< T >( UIntPack< FEMSigs ... >() , UIntPack< CSigs ... >() , F , static_cast< const SparseType& >( coefficients ) , constraints() , maxDepth );
			else if( typeid(coefficients)==typeid( DenseType) ) return _addFEMConstraints< T >( UIntPack< FEMSigs ... >() , UIntPack< CSigs ... >() , F , static_cast< const  DenseType& >( coefficients ) , constraints() , maxDepth );
			else                                                return _addFEMConstraints< T >( UIntPack< FEMSigs ... >() , UIntPack< CSigs ... >() , F ,                                   coefficients   , constraints() , maxDepth );
		}
		// Add one-dimensions -> one-dimension constraints (with the same signatures)
		template< typename T , unsigned int ... FEMDegrees , unsigned int ... FEMSigs >
		void addFEMConstraints( typename BaseFEMIntegrator::template System< UIntPack< FEMDegrees ... > >& F , const _SparseOrDenseNodeData< T , UIntPack< FEMSigs ... > >& coefficients , DenseNodeData< T , UIntPack< FEMSigs ... > >& constraints , LocalDepth maxDepth ) const
		{
			typedef SparseNodeData< T , UIntPack< FEMSigs ... > > SparseType;
			typedef  DenseNodeData< T , UIntPack< FEMSigs ... > >  DenseType;
			static_assert( sizeof...( FEMDegrees )==Dim && sizeof...( FEMSigs )==Dim , "[ERROR] Dimensions don't match" );
			static_assert( ParameterPack::Comparison< UIntPack< FEMDegrees ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >::Equal , "[ERROR] FEM signatures and degrees don't match" );
			typename BaseFEMIntegrator::template SystemConstraint< UIntPack< FEMDegrees ... > > _F( F );
			if     ( typeid(coefficients)==typeid(SparseType) ) return _addFEMConstraints< T >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , _F , static_cast< const SparseType& >( coefficients ) , constraints() , maxDepth );
			else if( typeid(coefficients)==typeid( DenseType) ) return _addFEMConstraints< T >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , _F , static_cast< const  DenseType& >( coefficients ) , constraints() , maxDepth );
			else                                                return _addFEMConstraints< T >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , _F ,                                   coefficients   , constraints() , maxDepth );
		}

	protected:
		template< unsigned int MaxDegree > void _supportApproximateProlongation( void );
		template< unsigned int SystemDegree > void _markInexactInterpolationElements( void);
		template< unsigned int SystemDegree > void _addAndMarkExactInterpolationElements( void );
		template< unsigned int SystemDegree > void _markNonBaseDirichletElements( void );
		template< unsigned int SystemDegree > void _markBaseDirichletElements( void );

		template< unsigned int Idx , typename T , unsigned int ... FEMSigs , typename ... InterpolationInfos >
		typename std::enable_if< ( Idx==sizeof...(InterpolationInfos) ) >::type _addInterpolationConstraints( DenseNodeData< T , UIntPack< FEMSigs ... > >& constraints , LocalDepth maxDepth , std::tuple< InterpolationInfos *... > interpolationInfos ) const {}
		template< unsigned int Idx , typename T , unsigned int ... FEMSigs , typename ... InterpolationInfos >
		typename std::enable_if< ( Idx <sizeof...(InterpolationInfos) ) >::type _addInterpolationConstraints( DenseNodeData< T , UIntPack< FEMSigs ... > >& constraints , LocalDepth maxDepth , std::tuple< InterpolationInfos *... > interpolationInfos ) const
		{
			_addInterpolationConstraints( constraints , maxDepth , std::get< Idx >( interpolationInfos ) );
			_addInterpolationConstraints< Idx+1 >( constraints , maxDepth , interpolationInfos );
		}
		template< typename T , unsigned int ... FEMSigs , unsigned int PointD >
		void _addInterpolationConstraints( DenseNodeData< T , UIntPack< FEMSigs ... > >& constraints , LocalDepth maxDepth , const InterpolationInfo< T , PointD > *interpolationInfo ) const;
	public:
		template< typename T , unsigned int ... FEMSigs , typename ... InterpolationInfos >
		void addInterpolationConstraints( DenseNodeData< T , UIntPack< FEMSigs ... > >& constraints , LocalDepth maxDepth , std::tuple< InterpolationInfos *... > interpolationInfos ) const { return _addInterpolationConstraints< 0 >( constraints , maxDepth , interpolationInfos ); }

		// Real
		template< unsigned int ... FEMDegrees1 , unsigned int ... FEMSigs1 , unsigned int ... FEMDegrees2 , unsigned int ... FEMSigs2 >
		double dot( typename BaseFEMIntegrator::Constraint< UIntPack< FEMDegrees1 ... > , UIntPack< FEMDegrees2 ... > , 1 >& F , const _SparseOrDenseNodeData< Real , UIntPack< FEMSigs1 ... > >& coefficients1 , const _SparseOrDenseNodeData< Real , UIntPack< FEMSigs2 ... > >& coefficients2 ) const
		{
			typedef SparseNodeData< Real , UIntPack< FEMSigs1 ... > > SparseType1;
			typedef  DenseNodeData< Real , UIntPack< FEMSigs1 ... > >  DenseType1;
			typedef SparseNodeData< Real , UIntPack< FEMSigs2 ... > > SparseType2;
			typedef  DenseNodeData< Real , UIntPack< FEMSigs2 ... > >  DenseType2;
			static_assert( sizeof...( FEMDegrees1 )==Dim && sizeof...( FEMSigs1 )==Dim && sizeof...( FEMDegrees2 )==Dim && sizeof...( FEMSigs2 )==Dim  , "[ERROR] Dimensions don't match" );
			static_assert( ParameterPack::Comparison< UIntPack< FEMDegrees1 ... > , UIntPack< FEMSignature< FEMSigs1 >::Degree ... > >::Equal , "[ERROR] FEM signature and degrees don't match" );
			static_assert( ParameterPack::Comparison< UIntPack< FEMDegrees2 ... > , UIntPack< FEMSignature< FEMSigs2 >::Degree ... > >::Equal , "[ERROR] FEM signature and degrees don't match" );
			if     ( typeid(coefficients1)==typeid(SparseType1) && typeid(coefficients2)==typeid(SparseType2) ) return _dot< Real >( UIntPack< FEMSigs1 ... >() , UIntPack< FEMSigs2 ... >() , F , static_cast< const SparseType1& >( coefficients1 ) , static_cast< const SparseType2& >( coefficients2 ) , []( Real v ,  Real w ){ return v*w; } );
			else if( typeid(coefficients1)==typeid(SparseType1) && typeid(coefficients2)==typeid( DenseType2) ) return _dot< Real >( UIntPack< FEMSigs1 ... >() , UIntPack< FEMSigs2 ... >() , F , static_cast< const SparseType1& >( coefficients1 ) , static_cast< const  DenseType2& >( coefficients2 ) , []( Real v ,  Real w ){ return v*w; } );
			else if( typeid(coefficients1)==typeid( DenseType1) && typeid(coefficients2)==typeid( DenseType2) ) return _dot< Real >( UIntPack< FEMSigs1 ... >() , UIntPack< FEMSigs2 ... >() , F , static_cast< const  DenseType1& >( coefficients1 ) , static_cast< const  DenseType2& >( coefficients2 ) , []( Real v ,  Real w ){ return v*w; } );
			else if( typeid(coefficients1)==typeid( DenseType1) && typeid(coefficients2)==typeid(SparseType2) ) return _dot< Real >( UIntPack< FEMSigs1 ... >() , UIntPack< FEMSigs2 ... >() , F , static_cast< const  DenseType1& >( coefficients1 ) , static_cast< const SparseType2& >( coefficients2 ) , []( Real v ,  Real w ){ return v*w; } );
			else                                                                                                return _dot< Real >( UIntPack< FEMSigs1 ... >() , UIntPack< FEMSigs2 ... >() , F ,                                    coefficients1   ,                                    coefficients2   , []( Real v ,  Real w ){ return v*w; } );
		}
		template< unsigned int ... FEMDegrees , unsigned int ... FEMSigs >
		double dot( typename BaseFEMIntegrator::System< UIntPack< FEMDegrees ... > >& F , const _SparseOrDenseNodeData< Real , UIntPack< FEMSigs ... > >& coefficients1 , const _SparseOrDenseNodeData< Real , UIntPack< FEMSigs ... > >& coefficients2 ) const
		{
			typedef SparseNodeData< Real , UIntPack< FEMSigs ... > > SparseType;
			typedef  DenseNodeData< Real , UIntPack< FEMSigs ... > >  DenseType;
			static_assert( sizeof...( FEMDegrees )==Dim && sizeof...( FEMSigs )==Dim , "[ERROR] Dimensions don't match" );
			static_assert( ParameterPack::Comparison< UIntPack< FEMDegrees ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >::Equal , "[ERROR] FEM signatures and degrees don't match" );
			typename BaseFEMIntegrator::template SystemConstraint< UIntPack< FEMDegrees ... > > _F( F );
			if     ( typeid(coefficients1)==typeid(SparseType) && typeid(coefficients2)==typeid(SparseType) ) return _dot< Real >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , _F , static_cast< const SparseType& >( coefficients1 ) , static_cast< const SparseType& >( coefficients2 ) , []( Real v ,  Real w ){ return v*w; } );
			else if( typeid(coefficients1)==typeid(SparseType) && typeid(coefficients2)==typeid( DenseType) ) return _dot< Real >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , _F , static_cast< const SparseType& >( coefficients1 ) , static_cast< const  DenseType& >( coefficients2 ) , []( Real v ,  Real w ){ return v*w; } );
			else if( typeid(coefficients1)==typeid( DenseType) && typeid(coefficients2)==typeid( DenseType) ) return _dot< Real >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , _F , static_cast< const  DenseType& >( coefficients1 ) , static_cast< const  DenseType& >( coefficients2 ) , []( Real v ,  Real w ){ return v*w; } );
			else if( typeid(coefficients1)==typeid( DenseType) && typeid(coefficients2)==typeid(SparseType) ) return _dot< Real >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , _F , static_cast< const  DenseType& >( coefficients1 ) , static_cast< const SparseType& >( coefficients2 ) , []( Real v ,  Real w ){ return v*w; } );
			else                                                                                              return _dot< Real >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , _F ,                                   coefficients1   ,                                   coefficients2   , []( Real v ,  Real w ){ return v*w; } );
		}
		template< unsigned int ... FEMDegrees , unsigned int ... FEMSigs >
		double squareNorm( typename BaseFEMIntegrator::template System< UIntPack< FEMDegrees ... > >& F , const _SparseOrDenseNodeData< Real , UIntPack< FEMSigs ... > >& coefficients ) const
		{
			typedef SparseNodeData< Real , UIntPack< FEMSigs ... > > SparseType;
			typedef  DenseNodeData< Real , UIntPack< FEMSigs ... > >  DenseType;
			typename BaseFEMIntegrator::template SystemConstraint< UIntPack< FEMDegrees ... > > _F( F );
			if     ( typeid(coefficients)==typeid(SparseType) ) return _dot< Real >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , _F , static_cast< const SparseType& >( coefficients ) , static_cast< const SparseType& >( coefficients ) , []( Real v ,  Real w ){ return v*w; } );
			else if( typeid(coefficients)==typeid( DenseType) ) return _dot< Real >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , _F , static_cast< const  DenseType& >( coefficients ) , static_cast< const  DenseType& >( coefficients ) , []( Real v ,  Real w ){ return v*w; } );
			else                                                return _dot< Real >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , _F ,                                   coefficients   ,                                   coefficients   , []( Real v ,  Real w ){ return v*w; } );
		}

		template< unsigned int ... FEMSigs1 , unsigned int ... FEMSigs2 , typename ... InterpolationInfos >
		double interpolationDot( const DenseNodeData< Real , UIntPack< FEMSigs1 ... > >& coefficients1 , const DenseNodeData< Real , UIntPack< FEMSigs2 ... > >& coefficients2 , std::tuple< InterpolationInfos *... > interpolationInfos ) const
		{
			static_assert( sizeof...( FEMSigs1 )==Dim && sizeof...( FEMSigs2 )==Dim , "[ERROR] Dimensions don't match" );
			return _interpolationDot< 0 >( UIntPack< FEMSigs1 ... >() , UIntPack< FEMSigs2 ... >() , coefficients1 , coefficients2 , []( Real v ,  Real w ){ return v*w; } , interpolationInfos );
		}
		template< unsigned int ... FEMSigs , typename ... InterpolationInfos >
		double interpolationSquareNorm( const DenseNodeData< Real , UIntPack< FEMSigs ... > >& coefficients , std::tuple< InterpolationInfos *... > interpolationInfos ) const
		{
			static_assert( sizeof...( FEMSigs )==Dim , "[ERROR] Dimensions don't match" );
			return _interpolationDot< 0 , Real >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , coefficients , coefficients , []( Real v ,  Real w ){ return v*w; } , interpolationInfos );
		}

		// Generic
		template< typename T , typename TDotT , unsigned int ... FEMDegrees1 , unsigned int ... FEMSigs1 , unsigned int ... FEMDegrees2 , unsigned int ... FEMSigs2 >
		double dot( TDotT Dot , typename BaseFEMIntegrator::Constraint< UIntPack< FEMDegrees1 ... > , UIntPack< FEMDegrees2 ... > , 1 >& F , const _SparseOrDenseNodeData< T , UIntPack< FEMSigs1 ... > >& coefficients1 , const _SparseOrDenseNodeData< T , UIntPack< FEMSigs2 ... > >& coefficients2 ) const
		{
			typedef SparseNodeData< T , UIntPack< FEMSigs1 ... > > SparseType1;
			typedef  DenseNodeData< T , UIntPack< FEMSigs1 ... > >  DenseType1;
			typedef SparseNodeData< T , UIntPack< FEMSigs2 ... > > SparseType2;
			typedef  DenseNodeData< T , UIntPack< FEMSigs2 ... > >  DenseType2;
			static_assert( sizeof...( FEMDegrees1 )==Dim && sizeof...( FEMSigs1 )==Dim && sizeof...( FEMDegrees2 )==Dim && sizeof...( FEMSigs2 )==Dim  , "[ERROR] Dimensions don't match" );
			static_assert( ParameterPack::Comparison< UIntPack< FEMDegrees1 ... > , UIntPack< FEMSignature< FEMSigs1 >::Degree ... > >::Equal , "[ERROR] FEM signature and degrees don't match" );
			static_assert( ParameterPack::Comparison< UIntPack< FEMDegrees2 ... > , UIntPack< FEMSignature< FEMSigs2 >::Degree ... > >::Equal , "[ERROR] FEM signature and degrees don't match" );
			if     ( typeid(coefficients1)==typeid(SparseType1) && typeid(coefficients2)==typeid(SparseType2) ) return _dot< T >( UIntPack< FEMSigs1 ... >() , UIntPack< FEMSigs2 ... >() , F , static_cast< const SparseType1& >( coefficients1 ) , static_cast< const SparseType2& >( coefficients2 ) , Dot );
			else if( typeid(coefficients1)==typeid(SparseType1) && typeid(coefficients2)==typeid( DenseType2) ) return _dot< T >( UIntPack< FEMSigs1 ... >() , UIntPack< FEMSigs2 ... >() , F , static_cast< const SparseType1& >( coefficients1 ) , static_cast< const  DenseType2& >( coefficients2 ) , Dot );
			else if( typeid(coefficients1)==typeid( DenseType1) && typeid(coefficients2)==typeid( DenseType2) ) return _dot< T >( UIntPack< FEMSigs1 ... >() , UIntPack< FEMSigs2 ... >() , F , static_cast< const  DenseType1& >( coefficients1 ) , static_cast< const  DenseType2& >( coefficients2 ) , Dot );
			else if( typeid(coefficients1)==typeid( DenseType1) && typeid(coefficients2)==typeid(SparseType2) ) return _dot< T >( UIntPack< FEMSigs1 ... >() , UIntPack< FEMSigs2 ... >() , F , static_cast< const  DenseType1& >( coefficients1 ) , static_cast< const SparseType2& >( coefficients2 ) , Dot );
			else                                                                                                return _dot< T >( UIntPack< FEMSigs1 ... >() , UIntPack< FEMSigs2 ... >() , F ,                                    coefficients1   ,                                    coefficients2   , Dot );
		}
		template< typename T , typename TDotT , unsigned int ... FEMDegrees , unsigned int ... FEMSigs >
		double dot( TDotT Dot , typename BaseFEMIntegrator::System< UIntPack< FEMDegrees ... > >& F , const _SparseOrDenseNodeData< T , UIntPack< FEMSigs ... > >& coefficients1 , const _SparseOrDenseNodeData< T , UIntPack< FEMSigs ... > >& coefficients2 ) const
		{
			typedef SparseNodeData< T , UIntPack< FEMSigs ... > > SparseType;
			typedef  DenseNodeData< T , UIntPack< FEMSigs ... > >  DenseType;
			static_assert( sizeof...( FEMDegrees )==Dim && sizeof...( FEMSigs )==Dim , "[ERROR] Dimensions don't match" );
			static_assert( ParameterPack::Comparison< UIntPack< FEMDegrees ... > , UIntPack< FEMSignature< FEMSigs >::Degree ... > >::Equal , "[ERROR] FEM signatures and degrees don't match" );
			typename BaseFEMIntegrator::template SystemConstraint< UIntPack< FEMDegrees ... > > _F( F );
			if     ( typeid(coefficients1)==typeid(SparseType) && typeid(coefficients2)==typeid(SparseType) ) return _dot< T >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , _F , static_cast< const SparseType& >( coefficients1 ) , static_cast< const SparseType& >( coefficients2 ) , Dot );
			else if( typeid(coefficients1)==typeid(SparseType) && typeid(coefficients2)==typeid( DenseType) ) return _dot< T >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , _F , static_cast< const SparseType& >( coefficients1 ) , static_cast< const  DenseType& >( coefficients2 ) , Dot );
			else if( typeid(coefficients1)==typeid( DenseType) && typeid(coefficients2)==typeid( DenseType) ) return _dot< T >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , _F , static_cast< const  DenseType& >( coefficients1 ) , static_cast< const  DenseType& >( coefficients2 ) , Dot );
			else if( typeid(coefficients1)==typeid( DenseType) && typeid(coefficients2)==typeid(SparseType) ) return _dot< T >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , _F , static_cast< const  DenseType& >( coefficients1 ) , static_cast< const SparseType& >( coefficients2 ) , Dot );
			else                                                                                              return _dot< T >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , _F ,                                   coefficients1   ,                                   coefficients2   , Dot );
		}
		template< typename T , typename TDotT , unsigned int ... FEMDegrees , unsigned int ... FEMSigs >
		double squareNorm( TDotT Dot , typename BaseFEMIntegrator::template System< UIntPack< FEMDegrees ... > >& F , const _SparseOrDenseNodeData< T , UIntPack< FEMSigs ... > >& coefficients ) const
		{
			typedef SparseNodeData< T , UIntPack< FEMSigs ... > > SparseType;
			typedef  DenseNodeData< T , UIntPack< FEMSigs ... > >  DenseType;
			typename BaseFEMIntegrator::template SystemConstraint< UIntPack< FEMDegrees ... > > _F( F );
			if     ( typeid(coefficients)==typeid(SparseType) ) return _dot< T >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , _F , static_cast< const SparseType& >( coefficients ) , static_cast< const SparseType& >( coefficients ) , Dot );
			else if( typeid(coefficients)==typeid( DenseType) ) return _dot< T >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , _F , static_cast< const  DenseType& >( coefficients ) , static_cast< const  DenseType& >( coefficients ) , Dot );
			else                                                return _dot< T >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , _F ,                                   coefficients   ,                                   coefficients   , Dot );
		}

		template< typename T , typename TDotT , unsigned int ... FEMSigs1 , unsigned int ... FEMSigs2 , typename ... InterpolationInfos >
		double interpolationDot( TDotT Dot , const DenseNodeData< T , UIntPack< FEMSigs1 ... > >& coefficients1 , const DenseNodeData< T , UIntPack< FEMSigs2 ... > >& coefficients2 , std::tuple< InterpolationInfos *... > interpolationInfos ) const
		{
			static_assert( sizeof...( FEMSigs1 )==Dim && sizeof...( FEMSigs2 )==Dim , "[ERROR] Dimensions don't match" );
			return _interpolationDot< T >( UIntPack< FEMSigs1 ... >() , UIntPack< FEMSigs2 ... >() , coefficients1 , coefficients2 , Dot , interpolationInfos );
		}
		template< typename T , typename TDotT , unsigned int ... FEMSigs , typename ... InterpolationInfos >
		double interpolationSquareNorm( TDotT Dot , const DenseNodeData< T , UIntPack< FEMSigs ... > >& coefficients , std::tuple< InterpolationInfos *... > interpolationInfos ) const
		{
			static_assert( sizeof...( FEMSigs )==Dim , "[ERROR] Dimensions don't match" );
			return _interpolationDot< T >( UIntPack< FEMSigs ... >() , UIntPack< FEMSigs ... >() , coefficients , coefficients , Dot , interpolationInfos );
		}

		template< unsigned int ... FEMSigs , typename ... InterpolationInfos >
		SparseMatrix< Real , matrix_index_type > systemMatrix( UIntPack< FEMSigs ... > , typename BaseFEMIntegrator::System< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& F , LocalDepth depth , std::tuple< InterpolationInfos *... > interpolationInfos ) const;
		template< unsigned int ... FEMSigs , typename ... InterpolationInfos >
		SparseMatrix< Real , matrix_index_type > prolongedSystemMatrix( UIntPack< FEMSigs ... > , typename BaseFEMIntegrator::System< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& F , LocalDepth highDepth , std::tuple< InterpolationInfos *... > interpolationInfos ) const;
		template< unsigned int ... FEMSigs >
		SparseMatrix< Real , matrix_index_type > downSampleMatrix( UIntPack< FEMSigs ... > , LocalDepth highDepth ) const;
		template< unsigned int ... FEMSigs >
		SparseMatrix< Real , matrix_index_type > upSampleMatrix( UIntPack< FEMSigs ... > , LocalDepth highDepth ) const;
		template< unsigned int ... FEMSigs , typename ... InterpolationInfos >
		SparseMatrix< Real , matrix_index_type > fullSystemMatrix( UIntPack< FEMSigs ... > , typename BaseFEMIntegrator::System< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& F , LocalDepth depth , std::tuple< InterpolationInfos *... > interpolationInfos ) const;

		template< typename T , unsigned int ... FEMSigs >
		void pushToBaseDepth( DenseNodeData< T , UIntPack< FEMSigs ... > >& coefficients ) const;

		struct SolverInfo
		{
		protected:
			struct _IterFunction
			{
				_IterFunction( int i ) : _i0(i) , _type(0) {}
				_IterFunction( std::function< int (              int ) > iFunction ) : _i1(iFunction) , _type(1) {}
				_IterFunction( std::function< int (       bool , int ) > iFunction ) : _i2(iFunction) , _type(2) {}
				_IterFunction( std::function< int ( int , bool , int ) > iFunction ) : _i3(iFunction) , _type(3) {}
				_IterFunction& operator = ( int i ){ *this = _IterFunction(i) ; return *this; }
				_IterFunction& operator = ( std::function< int (              int ) > iFunction ){ *this = _IterFunction(iFunction) ; return *this; }
				_IterFunction& operator = ( std::function< int (       bool , int ) > iFunction ){ *this = _IterFunction(iFunction) ; return *this; }
				_IterFunction& operator = ( std::function< int ( int , bool , int ) > iFunction ){ *this = _IterFunction(iFunction) ; return *this; }

				int operator()( int vCycle , bool restriction , int depth ) const
				{
					switch( _type )
					{
					case 0: return _i0;
					case 1: return _i1( depth );
					case 2: return _i2( restriction , depth );
					case 3: return _i3( vCycle , restriction , depth );
					default: return 0;
					}
				}
			protected:
				int _i0;
				std::function< int ( int ) > _i1;
				std::function< int ( bool , int ) > _i2;
				std::function< int ( int i3 , bool , int ) > _i3;
				int _type;
			};
		public:
			// How to solve
			bool wCycle;
			LocalDepth cgDepth;
			bool cascadic;
			unsigned int sliceBlockSize;
			bool useSupportWeights , useProlongationSupportWeights;
			std::function< Real ( Real , Real ) > sorRestrictionFunction;
			std::function< Real ( Real , Real ) > sorProlongationFunction;
			_IterFunction iters;
			int vCycles;
			double cgAccuracy;
			bool clearSolution;
			int baseVCycles;
			// What to output
			bool verbose , showResidual;
			int showGlobalResidual;

			SolverInfo( void ) : cgDepth(0) , wCycle(false) , cascadic(true) , iters(1) , vCycles(1) , cgAccuracy(0.) , verbose(false) , showResidual(false) , showGlobalResidual(SHOW_GLOBAL_RESIDUAL_NONE) , sliceBlockSize(1) , sorRestrictionFunction( []( Real , Real ){ return (Real)1; } ) , sorProlongationFunction( []( Real , Real ){ return (Real)1; } ) , useSupportWeights( false ) , useProlongationSupportWeights( false ) , baseVCycles(1) , clearSolution(true) { }
		};
		// Solve the linear system
		// There are several depths playing into the solver:
		// 1. maxDepth: The maximum depth of the tree
		// 2. solveDepth: The depth up to which we can solve (solveDepth<=maxDepth)
		// 3. fullDepth: The depth up to which the octree is completely refined (fullDepth<=maxDepth)
		// 4. baseDepth: The depth up to which the system is defined through the regular prolongation operators (baseDepth<=fullDepth)
		template< unsigned int ... FEMSigs , typename T , typename TDotT , typename ... InterpolationInfos >
		void solveSystem( UIntPack< FEMSigs ... > , typename BaseFEMIntegrator::template System< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& F , const DenseNodeData< T , UIntPack< FEMSigs ... > >& constraints , DenseNodeData< T , UIntPack< FEMSigs ... > >& solution , TDotT Dot , LocalDepth minSolveDepth , LocalDepth maxSolveDepth , const SolverInfo& solverInfo , std::tuple< InterpolationInfos *... > interpolationInfos=std::make_tuple() ) const;
		template< unsigned int ... FEMSigs , typename T , typename TDotT , typename ... InterpolationInfos >
		DenseNodeData< T , UIntPack< FEMSigs ... > > solveSystem( UIntPack< FEMSigs ... > , typename BaseFEMIntegrator::template System< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& F , const DenseNodeData< T , UIntPack< FEMSigs ... > >& constraints , TDotT Dot , LocalDepth minSolveDepth , LocalDepth maxSolveDepth , const SolverInfo& solverInfo , std::tuple< InterpolationInfos *... > interpolationInfos=std::make_tuple() ) const;
		template< unsigned int ... FEMSigs , typename ... InterpolationInfos >
		void solveSystem( UIntPack< FEMSigs ... > , typename BaseFEMIntegrator::template System< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& F , const DenseNodeData< Real , UIntPack< FEMSigs ... > >& constraints , DenseNodeData< Real , UIntPack< FEMSigs ... > >& solution , LocalDepth minSolveDepth , LocalDepth maxSolveDepth , const SolverInfo& solverInfo , std::tuple< InterpolationInfos *... > interpolationInfos=std::make_tuple() ) const
		{
			return solveSystem< FEMSigs ... , Real >( UIntPack< FEMSigs ... >() , F , constraints , solution , []( Real v , Real w ){ return v*w; } , minSolveDepth , maxSolveDepth , solverInfo , interpolationInfos );
		}
		template< unsigned int ... FEMSigs , typename ... InterpolationInfos >
		DenseNodeData< Real , UIntPack< FEMSigs ... > > solveSystem( UIntPack< FEMSigs ... > , typename BaseFEMIntegrator::template System< UIntPack< FEMSignature< FEMSigs >::Degree ... > >& F , const DenseNodeData< Real , UIntPack< FEMSigs ... > >& constraints , LocalDepth minSolveDepth , LocalDepth maxSolveDepth , const SolverInfo& solverInfo , std::tuple< InterpolationInfos *... > interpolationInfos=std::make_tuple() ) const
		{
			return solveSystem( UIntPack< FEMSigs ... >() , F , constraints , []( Real v , Real w ){ return v*w; } , minSolveDepth , maxSolveDepth , solverInfo , interpolationInfos );
		}

		FEMTreeNode& spaceRoot( void ){ return *_spaceRoot; }
		const FEMTreeNode &spaceRoot( void ) const { return *_spaceRoot; }
		const FEMTreeNode& tree( void ) const { return _tree; }
		_NodeInitializer &initializer( void ){ return _nodeInitializer; }
		size_t leaves( void ) const { return _tree.leaves(); }
		size_t allNodes              ( void ) const { size_t count = 0 ; _tree.processNodes( [&]( const FEMTreeNode *  ){ count++; } ) ; return count; }
		size_t activeNodes           ( void ) const { size_t count = 0 ; _tree.processNodes( [&]( const FEMTreeNode *n ){ if( IsActiveNode< Dim >( n ) ) count++; } ) ; return count; }
		size_t ghostNodes            ( void ) const { size_t count = 0 ; _tree.processNodes( [&]( const FEMTreeNode *n ){ if( !IsActiveNode< Dim >( n ) ) count++; } ) ; return count; }
		size_t dirichletNodes        ( void ) const { size_t count = 0 ; _tree.processNodes( [&]( const FEMTreeNode *n ){ if( n->nodeData.getDirichletNodeFlag() ) count++; } ) ; return count; }
		size_t dirichletElements     ( void ) const { size_t count = 0 ; _tree.processNodes( [&]( const FEMTreeNode *n ){ if( n->nodeData.getDirichletElementFlag() ) count++; } ) ; return count; }
		inline size_t validSpaceNodes( void ) const { size_t count = 0 ; _tree.processNodes( [&]( const FEMTreeNode *n ){ if( isValidSpaceNode( n ) ) count++; } ) ; return count; }
		inline size_t validSpaceNodes( LocalDepth d ) const { size_t count = 0 ; _tree.process( [&]( const FEMTreeNode *n ){ if( _localDepth(n)==d && isValidSpaceNode( n ) ) count++; } ) ; return count; }
		template< unsigned int ... FEMSigs > size_t validFEMNodes        ( UIntPack< FEMSigs ... > )                const { size_t count = 0 ; _tree.processNodes( [&]( const FEMTreeNode *n ){ if( isValidFEMNode( UIntPack< FEMSigs ... >() , n ) ) count++; } ) ; return count; }
		template< unsigned int ... FEMSigs > size_t validFEMNodes        ( UIntPack< FEMSigs ... > , LocalDepth d ) const { size_t count = 0 ; _tree.processNodes( [&]( const FEMTreeNode *n ){ if( _localDepth(n)==d && isValidFEMNode( UIntPack< FEMSigs ... >() , n ) ) count++; } ) ; return count; }
		template< unsigned int ... FEMSigs > size_t validUnlockedFEMNodes( UIntPack< FEMSigs ... > )                const { size_t count = 0 ; _tree.processNodes( [&]( const FEMTreeNode *n ){ if( isValidFEMNode( UIntPack< FEMSigs ... >() , n ) && !n->nodeData.getDirichletSupportedFlag() ) count++; } ) ; return count; }
		template< unsigned int ... FEMSigs > size_t validUnlockedFEMNodes( UIntPack< FEMSigs ... > , LocalDepth d ) const { size_t count = 0 ; _tree.processNodes( [&]( const FEMTreeNode *n ){ if( _localDepth(n)==d && isValidFEMNode( UIntPack< FEMSigs ... >() , n ) && !n->nodeData.getDirichletElementFlag() ) count++; } ) ; return count; }
		LocalDepth depth( void ) const { return _spaceRoot->maxDepth(); }
		LocalDepth maxDepth( void ) const { return _maxDepth; }
		template< typename ... DenseOrSparseNodeData >
		node_index_type resetNodeIndices( char flagsToClear , std::tuple< DenseOrSparseNodeData *... > data )
		{
			char mask = ~flagsToClear;
			if( sizeof...( DenseOrSparseNodeData ) )
			{
				std::vector< node_index_type > map( _nodeCount );
				_nodeCount = 0;
				auto nodeFunctor = [&]( FEMTreeNode *node )
					{
						node_index_type idx = node->nodeData.nodeIndex;
						_nodeInitializer( *node );
						if( node->nodeData.nodeIndex!=-1 ) map[ node->nodeData.nodeIndex ] = idx;
						node->nodeData.flags &= mask;
					};
				_tree.processNodes( nodeFunctor );

				_reorderDenseOrSparseNodeData< 0 >( GetPointer( map ) , (size_t)_nodeCount , data );
			}
			else
			{
				_nodeCount = 0;
				auto nodeFunctor = [&]( FEMTreeNode *node )
					{
						_nodeInitializer( *node );
						node->nodeData.flags &= mask;
					};
				_tree.processNodes( nodeFunctor );
			}
			return _nodeCount;
		}
		node_index_type resetNodeIndices( char flagsToClear ){ return resetNodeIndices( flagsToClear , std::make_tuple() ); }

		std::vector< node_index_type > merge( FEMTree* tree );
	protected:
		template< class Real1 , unsigned int _Dim > static bool _IsZero( Point< Real1 , _Dim > p );
		template< class Real1 >                     static bool _IsZero( Real1 p );
		template< class SReal , class Data , unsigned int _Dim > static Data _StencilDot( Point< SReal , _Dim > p1 , Point< Data , _Dim > p2 );
		template< class SReal , class Data >                     static Data _StencilDot( Point< SReal , 1 >    p1 , Point< Data , 1 >    p2 );
		template< class SReal , class Data >                     static Data _StencilDot( SReal                 p1 , Point< Data , 1 >    p2 );
		template< class SReal , class Data >                     static Data _StencilDot( Point< SReal , 1 >    p1 , Data                 p2 );
		template< class SReal , class Data >                     static Data _StencilDot( SReal                 p1 , Data                 p2 );

		// We need the signatures to test if nodes are valid
		template< typename T , unsigned int ... FEMSigs , unsigned int ... CSigs , unsigned int ... FEMDegrees , unsigned int ... CDegrees , unsigned int CDim , class Coefficients >
		void _addFEMConstraints( UIntPack< FEMSigs ... > , UIntPack< CSigs ... > , typename BaseFEMIntegrator::Constraint< UIntPack< FEMDegrees ... > , UIntPack< CDegrees ... > , CDim >& F , const Coefficients& coefficients , Pointer( T ) constraints , LocalDepth maxDepth ) const;
		template< typename T , typename TDotT , unsigned int ... FEMSigs1 , unsigned int ... FEMSigs2 , unsigned int ... Degrees1 , unsigned int ... Degrees2 , class Coefficients1 , class Coefficients2 >
		double _dot( UIntPack< FEMSigs1 ... > , UIntPack< FEMSigs2 ... > , typename BaseFEMIntegrator::Constraint< UIntPack< Degrees1 ... > , UIntPack< Degrees2 ... > , 1 >& F , const Coefficients1& coefficients1 , const Coefficients2& coefficients2 , TDotT Dot ) const;
		template< typename T , typename TDotT , unsigned int ... FEMSigs1 , unsigned int ... FEMSigs2 , class Coefficients1 , class Coefficients2 , unsigned int PointD >
		double _interpolationDot( UIntPack< FEMSigs1 ... > , UIntPack< FEMSigs2 ... > , const Coefficients1& coefficients1 , const Coefficients2& coefficients2 , TDotT Dot , const InterpolationInfo< T , PointD >* iInfo ) const;
		template< unsigned int Idx , typename T , typename TDotT , unsigned int ... FEMSigs1 , unsigned int ... FEMSigs2 , class Coefficients1 , class Coefficients2 , typename ... InterpolationInfos >
		typename std::enable_if< ( Idx==sizeof...(InterpolationInfos) ) , double >::type _interpolationDot( UIntPack< FEMSigs1 ... > , UIntPack< FEMSigs2 ... > , const Coefficients1& coefficients1 , const Coefficients2& coefficients2 , TDotT Dot , std::tuple< InterpolationInfos *... > interpolationInfos ) const { return 0.; }
		template< unsigned int Idx , typename T , typename TDotT , unsigned int ... FEMSigs1 , unsigned int ... FEMSigs2 , class Coefficients1 , class Coefficients2 , typename ... InterpolationInfos >
		typename std::enable_if< ( Idx <sizeof...(InterpolationInfos) ) , double >::type _interpolationDot( UIntPack< FEMSigs1 ... > , UIntPack< FEMSigs2 ... > , const Coefficients1& coefficients1 , const Coefficients2& coefficients2 , TDotT Dot , std::tuple< InterpolationInfos *... > interpolationInfos ) const
		{
			return _interpolationDot< T >( UIntPack< FEMSigs1 ... >() , UIntPack< FEMSigs2 ... >() , coefficients1 , coefficients2 , Dot , std::get< Idx >( interpolationInfos ) ) + _interpolationDot< Idx+1 , T >( UIntPack< FEMSigs1 ... >() , UIntPack< FEMSigs2 ... >() , coefficients1 , coefficients2 , Dot , interpolationInfos );
		}
		template< typename T , typename TDotT , unsigned int ... FEMSigs1 , unsigned int ... FEMSigs2 , class Coefficients1 , class Coefficients2 > double _interpolationDot( UIntPack< FEMSigs1 ... > , UIntPack< FEMSigs2 ... > , const Coefficients1& coefficients1 , const Coefficients2& coefficients2 , TDotT Dot ) const{ return 0; }
	};


	template< unsigned int Dim , class Real >
	struct FEMTreeInitializer
	{
		typedef RegularTreeNode< Dim , FEMTreeNodeData , depth_and_offset_type > FEMTreeNode;
		typedef NodeAndPointSample< Dim , Real > PointSample;

		template< class Data >
		struct DerivativeStream
		{
			virtual void resolution( unsigned int res[] ) const = 0;
			virtual bool nextDerivative( unsigned int idx[] , unsigned int& dir , Data& dValue ) = 0;
		};

		// Initialize the tree using a refinement avatar
		static size_t Initialize( FEMTreeNode& root , int maxDepth , std::function< bool ( int , int[] ) > Refine , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer );

		// Initialize the tree using a point stream
		template< typename Data >
		struct InputPointStream
		{
			typedef Data DataType;
			typedef DirectSum< Real , Point< Real , Dim > , Data > PointAndDataType;
			typedef InputDataStream< PointAndDataType > StreamType;
			static       Point< Real , Dim > &GetPoint(       PointAndDataType &pd ){ return pd.template get<0>(); }
			static const Point< Real , Dim > &GetPoint( const PointAndDataType &pd ){ return pd.template get<0>(); }
			static       DataType &GetData(       PointAndDataType &pd ){ return pd.template get<1>(); }
			static const DataType &GetData( const PointAndDataType &pd ){ return pd.template get<1>(); }

			static void BoundingBox( StreamType &stream , Data d , Point< Real , Dim >& min , Point< Real , Dim >& max )
			{
				PointAndDataType p;
				p.template get<1>() = d;
				for( unsigned int d=0 ; d<Dim ; d++ ) min[d] = std::numeric_limits< Real >::infinity() , max[d] = -std::numeric_limits< Real >::infinity();
				while( stream.read( p ) ) for( unsigned int d=0 ; d<Dim ; d++ ) min[d] = std::min< Real >( min[d] , p.template get<0>()[d] ) , max[d] = std::max< Real >( max[d] , p.template get<0>()[d] );
				stream.reset();
			}
		};

		template< typename Data >
		struct OutputPointStream
		{
			typedef Data DataType;
			typedef DirectSum< Real , Point< Real , Dim > , Data > PointAndDataType;
			typedef OutputDataStream< PointAndDataType > StreamType;
			static       Point< Real , Dim > &GetPoint(       PointAndDataType &pd ){ return pd.template get<0>(); }
			static const Point< Real , Dim > &GetPoint( const PointAndDataType &pd ){ return pd.template get<0>(); }
			static       DataType &GetData(       PointAndDataType &pd ){ return pd.template get<1>(); }
			static const DataType &GetData( const PointAndDataType &pd ){ return pd.template get<1>(); }
		};

		struct StreamInitializationData
		{
			friend FEMTreeInitializer;
		protected:
			std::vector< node_index_type > _nodeToIndexMap;
		};

		template< typename IsValidFunctor/*=std::function< bool ( const Point< Real , Dim > & , AuxData &... ) >*/ , typename ProcessFunctor/*=std::function< bool ( FEMTreeNode & , const Point< Real , Dim > & , AuxData &... ) >*/ , typename ... AuxData >
		static size_t Initialize( FEMTreeNode &root , InputDataStream< Point< Real , Dim > , AuxData ... > &pointStream , AuxData ... zeroData , int maxDepth ,                                                                  Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer , IsValidFunctor IsValid , ProcessFunctor Process );
		template< typename IsValidFunctor/*=std::function< bool ( const Point< Real , Dim > & , AuxData &... ) >*/ , typename ProcessFunctor/*=std::function< bool ( FEMTreeNode & , const Point< Real , Dim > & , AuxData &... ) >*/ , typename ... AuxData >
		static size_t Initialize( FEMTreeNode &root , InputDataStream< Point< Real , Dim > , AuxData ... > &pointStream , AuxData ... zeroData , int maxDepth , std::function< int ( Point< Real , Dim > ) > pointDepthFunctor , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer , IsValidFunctor IsValid , ProcessFunctor Process );

		template< typename AuxData >
		static size_t Initialize( struct StreamInitializationData &sid , FEMTreeNode &root , InputDataStream< Point< Real , Dim > , AuxData > &pointStream , AuxData zeroData , int maxDepth ,                                                                  std::vector< PointSample >& samplePoints , std::vector< AuxData > &sampleData , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer , std::function< Real ( const Point< Real , Dim > & , AuxData & ) > ProcessData = []( const Point< Real , Dim > & , AuxData & ){ return (Real)1.; } );
		template< typename AuxData >
		static size_t Initialize( struct StreamInitializationData &sid , FEMTreeNode &root , InputDataStream< Point< Real , Dim > , AuxData > &pointStream , AuxData zeroData , int maxDepth , std::function< int ( Point< Real , Dim > ) > pointDepthFunctor , std::vector< PointSample >& samplePoints , std::vector< AuxData > &sampleData , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer , std::function< Real ( const Point< Real , Dim > & , AuxData & ) > ProcessData = []( const Point< Real , Dim > & , AuxData & ){ return (Real)1.; } );
		template< typename AuxData >
		static size_t Initialize( FEMTreeNode &root , InputDataStream< Point< Real , Dim > , AuxData > &pointStream , AuxData zeroData , int maxDepth ,                                                                  std::vector< PointSample >& samplePoints , std::vector< AuxData > &sampleData , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer , std::function< Real ( const Point< Real , Dim > & , AuxData & ) > ProcessData = []( const Point< Real , Dim > & , AuxData & ){ return (Real)1.; } );
		template< typename AuxData >
		static size_t Initialize( FEMTreeNode &root , InputDataStream< Point< Real , Dim > , AuxData > &pointStream , AuxData zeroData , int maxDepth , std::function< int ( Point< Real , Dim > ) > pointDepthFunctor , std::vector< PointSample >& samplePoints , std::vector< AuxData > &sampleData , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer , std::function< Real ( const Point< Real , Dim > & , AuxData & ) > ProcessData = []( const Point< Real , Dim > & , AuxData & ){ return (Real)1.; } );

		// Initialize the tree using simplices
		static void Initialize( FEMTreeNode& root , const std::vector< Point< Real , Dim > >& vertices , const std::vector< SimplexIndex< Dim-1 , node_index_type > >& simplices , int maxDepth , std::vector< PointSample >& samples , std::vector< Allocator< FEMTreeNode > * > &nodeAllocators , std::function< void ( FEMTreeNode& ) > NodeInitializer );
		static void Initialize( FEMTreeNode& root , const std::vector< Point< Real , Dim > >& vertices , const std::vector< SimplexIndex< Dim-1 , node_index_type > >& simplices , unsigned int regularGridDepth , unsigned int maxDepth , std::vector< NodeSimplices< Dim , Real > >& nodeSimplices , std::vector< Allocator< FEMTreeNode > * > &nodeAllocators , std::function< void ( FEMTreeNode& ) > NodeInitializer );

		struct GeometryNodeType
		{
			enum Type : char
			{
				UNKNOWN ,
				INTERIOR ,
				BOUNDARY ,
				EXTERIOR
			};
			Type type;
			GeometryNodeType( Type t=UNKNOWN ) : type( t ){}
			bool operator== ( GeometryNodeType gnt ) const { return type==gnt.type; }
			bool operator!= ( GeometryNodeType gnt ) const { return type!=gnt.type; }
			bool operator== ( Type type ) const { return this->type==type; }
			bool operator!= ( Type type ) const { return this->type!=type; }

			friend std::ostream &operator << ( std::ostream &os , const GeometryNodeType &type )
			{
				if     ( type.type==UNKNOWN  ) return os << "unknown";
				else if( type.type==INTERIOR ) return os << "interior";
				else if( type.type==BOUNDARY ) return os << "boundary";
				else if( type.type==EXTERIOR ) return os << "exterior";
				return os;
			}
		};

		template< unsigned int _Dim=Dim >
		static typename std::enable_if< _Dim!=1 , DenseNodeData< GeometryNodeType , IsotropicUIntPack< Dim , FEMTrivialSignature > > >::type GetGeometryNodeDesignators( FEMTreeNode *root , const std::vector< Point< Real , Dim > >& vertices , const std::vector< SimplexIndex< Dim-1 , node_index_type > >& simplices , unsigned int regularGridDepth , unsigned int maxDepth , std::vector< Allocator< FEMTreeNode > * > &nodeAllocators , std::function< void ( FEMTreeNode& ) > NodeInitializer );
		template< unsigned int _Dim=Dim >
		static typename std::enable_if< _Dim==1 , DenseNodeData< GeometryNodeType , IsotropicUIntPack< Dim , FEMTrivialSignature > > >::type GetGeometryNodeDesignators( FEMTreeNode *root , const std::vector< Point< Real , Dim > >& vertices , const std::vector< SimplexIndex< Dim-1 , node_index_type > >& simplices , unsigned int regularGridDepth , unsigned int maxDepth , std::vector< Allocator< FEMTreeNode > * > &nodeAllocators , std::function< void ( FEMTreeNode& ) > NodeInitializer );
		static void TestGeometryNodeDesignators( const FEMTreeNode *root , const DenseNodeData< GeometryNodeType , IsotropicUIntPack< Dim , FEMTrivialSignature > > &geometryNodeDesignators );
		static void PushGeometryNodeDesignatorsToFiner( const FEMTreeNode *root , DenseNodeData< GeometryNodeType , IsotropicUIntPack< Dim , FEMTrivialSignature > > &geometryNodeDesignators , unsigned int maxDepth=-1 );
		static void PullGeometryNodeDesignatorsFromFiner( const FEMTreeNode *root , DenseNodeData< GeometryNodeType , IsotropicUIntPack< Dim , FEMTrivialSignature > > &geometryNodeDesignators , unsigned int maxDepth=-1 );

		// Initialize the tree using weighted points
		static void Initialize( FEMTreeNode &root , const std::vector< ProjectiveData< Point< Real , Dim > , Real > > &points , int maxDepth , std::vector< PointSample >& samples , std::vector< Allocator< FEMTreeNode > * > &nodeAllocators , std::function< void ( FEMTreeNode& ) > NodeInitializer );
		static void Initialize( FEMTreeNode &root , const std::vector< ProjectiveData< Point< Real , Dim > , Real > > &points , int maxDepth , std::vector< PointSample >& samples );

		template< class Data , class _Data , bool Dual=true >
		static size_t Initialize( FEMTreeNode& root , ConstPointer( Data ) values , ConstPointer( int ) labels , int resolution[Dim] , std::vector< NodeSample< Dim , _Data > > derivatives[Dim] , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer , std::function< _Data ( const Data& ) > DataConverter = []( const Data& d ){ return (_Data)d; }	);
		template< bool Dual , class Data >
		static unsigned int Initialize( FEMTreeNode& root , DerivativeStream< Data >& dStream , Data zeroData , std::vector< NodeSample< Dim , Data > > derivatives[Dim] , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer );

	protected:
		static size_t _Initialize( FEMTreeNode &node , int maxDepth , std::function< bool ( int , int[] ) > Refine , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer );
		template< bool ThreadSafe > static size_t _AddSimplex( FEMTreeNode& root , Simplex< Real , Dim , Dim-1 >& s , int maxDepth , std::vector< PointSample >& samples , std::vector< node_index_type >* nodeToIndexMap , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer );
		template< bool ThreadSafe > static size_t _AddSimplex( FEMTreeNode* node , Simplex< Real , Dim , Dim-1 >& s , int maxDepth , std::vector< PointSample >& samples , std::vector< node_index_type >* nodeToIndexMap , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer );
		template< bool ThreadSafeAllocation , bool ThreadSafeSimplices > static size_t _AddSimplex( FEMTreeNode& root , node_index_type id , Simplex< Real , Dim , Dim-1 >& s , int maxDepth , std::vector< NodeSimplices< Dim , Real > >& simplices , std::vector< node_index_type >& nodeToIndexMap , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer );
		template< bool ThreadSafeAllocation , bool ThreadSafeSimplices > static size_t _AddSimplex( FEMTreeNode* node , node_index_type id , Simplex< Real , Dim , Dim-1 >& s , int maxDepth , std::vector< NodeSimplices< Dim , Real > >& simplices , std::vector< node_index_type >& nodeToIndexMap , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer );
		template< bool ThreadSafe > static size_t _AddSample( FEMTreeNode& root , ProjectiveData< Point< Real , Dim > , Real > s , int maxDepth , std::vector< PointSample >& samples , std::vector< node_index_type >* nodeToIndexMap , Allocator< FEMTreeNode >* nodeAllocator , std::function< void ( FEMTreeNode& ) > NodeInitializer );
		static size_t _AddSample( FEMTreeNode& root , ProjectiveData< Point< Real , Dim > , Real > s , int maxDepth , std::vector< PointSample >& samples , std::vector< node_index_type >* nodeToIndexMap );
		static size_t _AddSample( FEMTreeNode* node , ProjectiveData< Point< Real , Dim > , Real > s , int maxDepth , std::vector< PointSample >& samples , std::vector< node_index_type >* nodeToIndexMap );
		static DenseNodeData< GeometryNodeType , IsotropicUIntPack< Dim , FEMTrivialSignature > > _GetGeometryNodeDesignators( FEMTreeNode *root , const std::vector< Point< Real , Dim > >& vertices , const std::vector< SimplexIndex< Dim-1 , node_index_type > >& simplices , const std::vector< Point< Real , Dim > > &normals , unsigned int regularGridDepth , unsigned int maxDepth , std::vector< Allocator< FEMTreeNode > * > &nodeAllocators , std::function< void ( FEMTreeNode& ) > NodeInitializer );
	};
	template< unsigned int Dim , class Real >
	template< unsigned int ... SupportSizes >
	const unsigned int FEMTree< Dim , Real >::CornerLoopData< SupportSizes ... >::supportSizes[] = { SupportSizes ... };


#include "FEMTree.inl"
#include "FEMTree.SortedTreeNodes.inl"
#include "FEMTree.WeightedSamples.inl"
#include "FEMTree.System.inl"
#include "FEMTree.Evaluation.inl"
#include "FEMTree.LevelSet.inl"
#include "FEMTree.LevelSet.2D.inl"
#include "FEMTree.LevelSet.3D.inl"
#include "FEMTree.Initialize.inl"
}

#endif // FEM_TREE_INCLUDED
