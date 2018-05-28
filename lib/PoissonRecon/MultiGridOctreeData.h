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
// [COMMENTS]
// -- Throughout the code, should make a distinction between indices and offsets
// -- Make an instance of _Evaluate that samples the finite-elements correctly (specifically, to handle the boundaries)
// -- Make functions like depthAndOffset parity dependent (ideally all "depth"s should be relative to the B-Slpline resolution
// -- Make all points relative to the unit-cube, regardless of degree parity
// -- It's possible that for odd degrees, the iso-surfacing will fail because the leaves in the SortedTreeNodes do not form a partition of space
// -- [MAYBE] Treat normal field as a sum of delta functions, rather than a smoothed signal (again, so that high degrees aren't forced to generate smooth reconstructions)
// -- [MAYBE] Make the degree of the B-Spline with which the normals are splatted independent of the degree of the FEM system. (This way, higher degree systems aren't forced to generate smoother normal fields.)

// [TODO]
// -- Currently, the implementation assumes that the boundary constraints are the same for vector fields and scalar fields
// -- Fix up the ordering in the divergence evaluation

#ifndef MULTI_GRID_OCTREE_DATA_INCLUDED
#define MULTI_GRID_OCTREE_DATA_INCLUDED

#define NEW_CODE 1
#define NEW_NEW_CODE 0		// Enabling this ensures that all the nodes contained in the support of the normal field are in the tree

#define DATA_DEGREE 1		// The order of the B-Spline used to splat in data for color interpolation
#define WEIGHT_DEGREE 2		// The order of the B-Spline used to splat in the weights for density estimation
#define NORMAL_DEGREE 2		// The order of the B-Spline used to splat int the normals for constructing the Laplacian constraints

//#define MAX_MEMORY_GB 15
#define MAX_MEMORY_GB 0

#define GRADIENT_DOMAIN_SOLUTION 1	// Given the constraint vector-field V(p), there are two ways to solve for the coefficients, x, of the indicator function
									// with respect to the B-spline basis {B_i(p)}
									// 1] Find x minimizing:
									//			|| V(p) - \sum_i \nabla x_i B_i(p) ||^2
									//		which is solved by the system A_1x = b_1 where:
									//			A_1[i,j] = < \nabla B_i(p) , \nabla B_j(p) >
									//			b_1[i]   = < \nabla B_i(p) , V(p) >
									// 2] Formulate this as a Poisson equation:
									//			\sum_i x_i \Delta B_i(p) = \nabla \cdot V(p)
									//		which is solved by the system A_2x = b_2 where:
									//			A_2[i,j] = - < \Delta B_i(p) , B_j(p) >
									//			b_2[i]   = - < B_i(p) , \nabla \cdot V(p) >
									// Although the two system matrices should be the same (assuming that the B_i satisfy dirichlet/neumann boundary conditions)
									// the constraint vectors can differ when V does not satisfy the Neumann boundary conditions:
									//		A_1[i,j] = \int_R < \nabla B_i(p) , \nabla B_j(p) >
									//               = \int_R [ \nabla \cdot ( B_i(p) \nabla B_j(p) ) - B_i(p) \Delta B_j(p) ]
									//               = \int_dR < N(p) , B_i(p) \nabla B_j(p) > + A_2[i,j]
									// and the first integral is zero if either f_i is zero on the boundary dR or the derivative of B_i across the boundary is zero.
									// However, for the constraints we have:
									//		b_1(i)   = \int_R < \nabla B_i(p) , V(p) >
									//               = \int_R [ \nabla \cdot ( B_i(p) V(p) ) - B_i(p) \nabla \cdot V(p) ]
									//               = \int_dR < N(p) ,  B_i(p) V(p) > + b_2[i]
									// In particular, this implies that if the B_i satisfy the Neumann boundary conditions (rather than Dirichlet),
									// and V is not zero across the boundary, then the two constraints are different.
									// Forcing the < V(p) , N(p) > = 0 on the boundary, by killing off the component of the vector-field in the normal direction
									// (FORCE_NEUMANN_FIELD), makes the two systems equal, and the value of this flag should be immaterial.
									// Note that under interpretation 1, we have:
									//		\sum_i b_1(i) = < \nabla \sum_ i B_i(p) , V(p) > = 0
									// because the B_i's sum to one. However, in general, we could have
									//		\sum_i b_2(i) \neq 0.
									// This could cause trouble because the constant functions are in the kernel of the matrix A, so CG will misbehave if the constraint
									// has a non-zero DC term. (Again, forcing < V(p) , N(p) > = 0 along the boundary resolves this problem.)

#define FORCE_NEUMANN_FIELD 1		// This flag forces the normal component across the boundary of the integration domain to be zero.
									// This should be enabled if GRADIENT_DOMAIN_SOLUTION is not, so that CG doesn't run into trouble.

#if !FORCE_NEUMANN_FIELD
#pragma message( "[WARNING] Not zeroing out normal component on boundary" )
#endif // !FORCE_NEUMANN_FIELD

#include "Hash.h"
#include "BSplineData.h"
#include "PointStream.h"

#ifndef _OPENMP
int omp_get_num_procs( void ){ return 1; }
int omp_get_thread_num( void ){ return 0; }
#endif // _OPENMP

class TreeNodeData
{
public:
	static size_t NodeCount;
	int nodeIndex;
	char flags;

	TreeNodeData( void );
	~TreeNodeData( void );
};

class VertexData
{
	typedef OctNode< TreeNodeData > TreeOctNode;
public:
	static const int VERTEX_COORDINATE_SHIFT = ( sizeof( long long ) * 8 ) / 3;
	static long long   EdgeIndex( const TreeOctNode* node , int eIndex , int maxDepth , int index[DIMENSION] );
	static long long   EdgeIndex( const TreeOctNode* node , int eIndex , int maxDepth );
	static long long   FaceIndex( const TreeOctNode* node , int fIndex , int maxDepth,int index[DIMENSION] );
	static long long   FaceIndex( const TreeOctNode* node , int fIndex , int maxDepth );
	static long long CornerIndex( const TreeOctNode* node , int cIndex , int maxDepth , int index[DIMENSION] );
	static long long CornerIndex( const TreeOctNode* node , int cIndex , int maxDepth );
	static long long CenterIndex( const TreeOctNode* node , int maxDepth , int index[DIMENSION] );
	static long long CenterIndex( const TreeOctNode* node , int maxDepth );
	static long long CornerIndex( int depth , const int offSet[DIMENSION] , int cIndex , int maxDepth , int index[DIMENSION] );
	static long long CenterIndex( int depth , const int offSet[DIMENSION] , int maxDepth , int index[DIMENSION] );
	static long long CornerIndexKey( const int index[DIMENSION] );
};

// This class stores the octree nodes, sorted by depth and then by z-slice.
// To support primal representations, the initializer takes a function that
// determines if a node should be included/indexed in the sorted list.
class SortedTreeNodes
{
	typedef OctNode< TreeNodeData > TreeOctNode;
protected:
	Pointer( Pointer( int ) ) _sliceStart;
	int _levels;
public:
	Pointer( TreeOctNode* ) treeNodes;
	int begin( int depth ) const{ return _sliceStart[depth][0]; }
	int   end( int depth ) const{ return _sliceStart[depth][(size_t)1<<depth]; }
	int begin( int depth , int slice ) const{ return _sliceStart[depth][slice  ]  ; }
	int   end( int depth , int slice ) const{ if(depth<0||depth>=_levels||slice<0||slice>=(1<<depth)) printf( "uh oh\n" ) ; return _sliceStart[depth][slice+1]; }
	int size( void ) const { return _sliceStart[_levels-1][(size_t)1<<(_levels-1)]; }
	int size( int depth ) const { if(depth<0||depth>=_levels) printf( "uhoh\n" ); return _sliceStart[depth][(size_t)1<<depth] - _sliceStart[depth][0]; }
	int size( int depth , int slice ) const { return _sliceStart[depth][slice+1] - _sliceStart[depth][slice]; }
	int levels( void ) const { return _levels; }

	SortedTreeNodes( void );
	~SortedTreeNodes( void );
	void set( TreeOctNode& root , std::vector< int >* map );
	void set( TreeOctNode& root );

	template< int Indices >
	struct  _Indices
	{
		int idx[Indices];
		_Indices( void ){ memset( idx , -1 , sizeof( int ) * Indices ); }
		int& operator[] ( int i ) { return idx[i]; }
		const int& operator[] ( int i ) const { return idx[i]; }
	};
	typedef _Indices< Square::CORNERS > SquareCornerIndices;
	typedef _Indices< Square::EDGES > SquareEdgeIndices;
	typedef _Indices< Square::FACES > SquareFaceIndices;

	struct SliceTableData
	{
		Pointer( SquareCornerIndices ) cTable;
		Pointer( SquareEdgeIndices   ) eTable;
		Pointer( SquareFaceIndices   ) fTable;
		int cCount , eCount , fCount , nodeOffset , nodeCount;
		SliceTableData( void ){ fCount = eCount = cCount = 0 , cTable = NullPointer( SquareCornerIndices ) , eTable = NullPointer( SquareEdgeIndices ) , fTable = NullPointer( SquareFaceIndices ) , _cMap = _eMap = _fMap = NullPointer( int ); }
		~SliceTableData( void ){ clear(); }
		void clear( void ){ DeletePointer( cTable ) ; DeletePointer( eTable ) ; DeletePointer( fTable ) ; fCount = eCount = cCount = 0; }
		SquareCornerIndices& cornerIndices( const TreeOctNode* node );
		SquareCornerIndices& cornerIndices( int idx );
		const SquareCornerIndices& cornerIndices( const TreeOctNode* node ) const;
		const SquareCornerIndices& cornerIndices( int idx ) const;
		SquareEdgeIndices& edgeIndices( const TreeOctNode* node );
		SquareEdgeIndices& edgeIndices( int idx );
		const SquareEdgeIndices& edgeIndices( const TreeOctNode* node ) const;
		const SquareEdgeIndices& edgeIndices( int idx ) const;
		SquareFaceIndices& faceIndices( const TreeOctNode* node );
		SquareFaceIndices& faceIndices( int idx );
		const SquareFaceIndices& faceIndices( const TreeOctNode* node ) const;
		const SquareFaceIndices& faceIndices( int idx ) const;
	protected:
		Pointer( int ) _cMap;
		Pointer( int ) _eMap;
		Pointer( int ) _fMap;
		friend class SortedTreeNodes;
	};
	struct XSliceTableData
	{
		Pointer( SquareCornerIndices ) eTable;
		Pointer( SquareEdgeIndices ) fTable;
		int fCount , eCount , nodeOffset , nodeCount;
		XSliceTableData( void ){ fCount = eCount = 0 , eTable = NullPointer( SquareCornerIndices ) , fTable = NullPointer( SquareEdgeIndices ) , _eMap = _fMap = NullPointer( int ); }
		~XSliceTableData( void ){ clear(); }
		void clear( void ) { DeletePointer( fTable ) ; DeletePointer( eTable ) ; fCount = eCount = 0; }
		SquareCornerIndices& edgeIndices( const TreeOctNode* node );
		SquareCornerIndices& edgeIndices( int idx );
		const SquareCornerIndices& edgeIndices( const TreeOctNode* node ) const;
		const SquareCornerIndices& edgeIndices( int idx ) const;
		SquareEdgeIndices& faceIndices( const TreeOctNode* node );
		SquareEdgeIndices& faceIndices( int idx );
		const SquareEdgeIndices& faceIndices( const TreeOctNode* node ) const;
		const SquareEdgeIndices& faceIndices( int idx ) const;
	protected:
		Pointer( int ) _eMap;
		Pointer( int ) _fMap;
		friend class SortedTreeNodes;
	};
	void setSliceTableData (  SliceTableData& sData , int depth , int offset , int threads ) const;
	void setXSliceTableData( XSliceTableData& sData , int depth , int offset , int threads ) const;
};

template< int Degree >
struct PointSupportKey : public OctNode< TreeNodeData >::NeighborKey< BSplineEvaluationData< Degree >::SupportEnd , -BSplineEvaluationData< Degree >::SupportStart >
{
	static const int LeftRadius  =  BSplineEvaluationData< Degree >::SupportEnd;
	static const int RightRadius = -BSplineEvaluationData< Degree >::SupportStart;
	static const int Size = LeftRadius + RightRadius + 1;
};
template< int Degree >
struct ConstPointSupportKey : public OctNode< TreeNodeData >::ConstNeighborKey< BSplineEvaluationData< Degree >::SupportEnd , -BSplineEvaluationData< Degree >::SupportStart >
{
	static const int LeftRadius  =  BSplineEvaluationData< Degree >::SupportEnd;
	static const int RightRadius = -BSplineEvaluationData< Degree >::SupportStart;
	static const int Size = LeftRadius + RightRadius + 1;
};

template< class Real >
struct PointData
{
	Point3D< Real > position;
	Real weightedCoarserDValue;
	Real weight;
	PointData( Point3D< Real > p=Point3D< Real >() , Real w=0 ) { position = p , weight = w , weightedCoarserDValue = Real(0); }
};
template< class Data , int Degree >
struct SparseNodeData
{
	std::vector< int > indices;
	std::vector< Data > data;
	template< class TreeNodeData >
	int index( const OctNode< TreeNodeData >* node ) const { return ( !node || node->nodeData.nodeIndex<0 || node->nodeData.nodeIndex>=(int)indices.size() ) ? -1 : indices[ node->nodeData.nodeIndex ]; }
#if NEW_NEW_CODE
	int index( int nodeIndex ) const { return ( nodeIndex<0 || nodeIndex>=(int)indices.size() ) ? -1 : indices[ nodeIndex ]; }
#endif // NEW_NEW_CODE
	void resize( size_t sz ){ indices.resize( sz , -1 ); }
	void remapIndices( const std::vector< int >& map )
	{
		std::vector< int > temp = indices;
		indices.resize( map.size() );
		for( size_t i=0 ; i<map.size() ; i++ )
			if( map[i]<(int)temp.size() ) indices[i] = temp[ map[i] ];
			else                          indices[i] = -1;
	}
};
template< class Data , int Degree >
struct DenseNodeData
{
	Pointer( Data ) data;
	DenseNodeData( void ) { data = NullPointer( Data ); }
	DenseNodeData( size_t sz ){ if( sz ) data = NewPointer< Data >( sz ) ; else data = NullPointer( Data ); }
	void resize( size_t sz ){ DeletePointer( data ) ; if( sz ) data = NewPointer< Data >( sz ) ; else data = NullPointer( Data ); }
	Data& operator[] ( int idx ) { return data[idx]; }
	const Data& operator[] ( int idx ) const { return data[idx]; }
};

template< class C , int N > struct Stencil{ C values[N][N][N]; };

template< int Degree1 , int Degree2 >
class SystemCoefficients
{
	typedef typename BSplineIntegrationData< Degree1 , Degree2 >::FunctionIntegrator FunctionIntegrator;
	static const int OverlapSize  = BSplineIntegrationData< Degree1 , Degree2 >::OverlapSize;
	static const int OverlapStart = BSplineIntegrationData< Degree1 , Degree2 >::OverlapStart;
	static const int OverlapEnd   = BSplineIntegrationData< Degree1 , Degree2 >::OverlapEnd;
public:
	static double GetLaplacian  ( const typename FunctionIntegrator::     Integrator& integrator , const int off1[3] , const int off2[3] );
	static double GetLaplacian  ( const typename FunctionIntegrator::ChildIntegrator& integrator , const int off1[3] , const int off2[3] );
	static double GetDivergence1( const typename FunctionIntegrator::     Integrator& integrator , const int off1[3] , const int off2[3] , Point3D< double > normal1 );
	static double GetDivergence1( const typename FunctionIntegrator::ChildIntegrator& integrator , const int off1[3] , const int off2[3] , Point3D< double > normal1 );
	static double GetDivergence2( const typename FunctionIntegrator::     Integrator& integrator , const int off1[3] , const int off2[3] , Point3D< double > normal2 );
	static double GetDivergence2( const typename FunctionIntegrator::ChildIntegrator& integrator , const int off1[3] , const int off2[3] , Point3D< double > normal2 );
	static Point3D< double > GetDivergence1 ( const typename FunctionIntegrator::     Integrator& integrator , const int off1[3] , const int off2[3] );
	static Point3D< double > GetDivergence1 ( const typename FunctionIntegrator::ChildIntegrator& integrator , const int off1[3] , const int off2[3] );
	static Point3D< double > GetDivergence2 ( const typename FunctionIntegrator::     Integrator& integrator , const int off1[3] , const int off2[3] );
	static Point3D< double > GetDivergence2 ( const typename FunctionIntegrator::ChildIntegrator& integrator , const int off1[3] , const int off2[3] );
	static void SetCentralDivergenceStencil ( const typename FunctionIntegrator::     Integrator& integrator , Stencil< Point3D< double > , OverlapSize >& stencil , bool scatter );
	static void SetCentralDivergenceStencils( const typename FunctionIntegrator::ChildIntegrator& integrator , Stencil< Point3D< double > , OverlapSize > stencil[2][2][2] , bool scatter );
	static void SetCentralLaplacianStencil  ( const typename FunctionIntegrator::     Integrator& integrator , Stencil< double , OverlapSize >& stencil );
	static void SetCentralLaplacianStencils ( const typename FunctionIntegrator::ChildIntegrator& integrator , Stencil< double , OverlapSize > stencil[2][2][2] );
};

// Note that throughout this code, the "depth" parameter refers to the depth in the octree, not the corresponding depth
// of the B-Spline element
template< class Real >
class Octree
{
	typedef OctNode< TreeNodeData > TreeOctNode;
public:
	template< int FEMDegree > static void FunctionIndex( const TreeOctNode* node , int idx[3] );

	typedef typename TreeOctNode::     NeighborKey< 1 , 1 >      AdjacenctNodeKey;
	typedef typename TreeOctNode::ConstNeighborKey< 1 , 1 > ConstAdjacenctNodeKey;

	template< class V >
	struct ProjectiveData
	{
		V v;
		Real w;
		ProjectiveData( V vv=V(0) , Real ww=Real(0) ) : v(vv) , w(ww) { }
		operator V (){ return w!=0 ? v/w : v*w; }
		ProjectiveData& operator += ( const ProjectiveData& p ){ v += p.v , w += p.w ; return *this; }
		ProjectiveData& operator -= ( const ProjectiveData& p ){ v -= p.v , w -= p.w ; return *this; }
		ProjectiveData& operator *= ( Real s ){ v *= s , w *= s ; return *this; }
		ProjectiveData& operator /= ( Real s ){ v /= s , w /= s ; return *this; }
		ProjectiveData operator + ( const ProjectiveData& p ) const { return ProjectiveData( v+p.v , w+p.w ); }
		ProjectiveData operator - ( const ProjectiveData& p ) const { return ProjectiveData( v-p.v , w-p.w ); }
		ProjectiveData operator * ( Real s ) const { return ProjectiveData( v*s , w*s ); }
		ProjectiveData operator / ( Real s ) const { return ProjectiveData( v/s , w/s ); }
	};
	template< int FEMDegree > static bool IsValidNode( const TreeOctNode* node , bool dirichlet );
protected:
	template< int FEMDegree > bool _IsValidNode( const TreeOctNode* node ) const { return node && ( node->nodeData.flags & ( 1<<( FEMDegree&1 ) ) ) ; }

	TreeOctNode _tree;
	TreeOctNode* _spaceRoot;
	SortedTreeNodes _sNodes;
	int _splatDepth;
	int _maxDepth;
	int _minDepth;
	int _fullDepth;
	bool _constrainValues;
	bool _dirichlet;
	Real _scale;
	Point3D< Real > _center;
	int _multigridDegree;

	bool _InBounds( Point3D< Real > ) const;
	template< int FEMDegree > static int _Dimension( int depth ){ return BSplineData< FEMDegree >::Dimension( depth-1 ); }
	static int _Resolution( int depth ){ return 1<<(depth-1); }
	template< int FEMDegree > static bool _IsInteriorlySupported( int d , int x , int y , int z )
	{
		if( d-1>=0 )
		{
			int begin , end;
			BSplineEvaluationData< FEMDegree >::InteriorSupportedSpan( d-1 , begin , end );
			return ( x>=begin && x<end && y>=begin && y<end && z>=begin && z<end );
		}
		else return false;
	}
	template< int FEMDegree > static bool _IsInteriorlySupported( const TreeOctNode* node )
	{
		if( !node ) return false;
		int d , off[3];
		node->depthAndOffset( d , off );
		return _IsInteriorlySupported< FEMDegree >( d , off[0] , off[1] , off[2] );
	}
	template< int FEMDegree1 , int FEMDegree2 > static bool _IsInteriorlyOverlapped( int d , int x , int y , int z )
	{
		if( d-1>=0 )
		{
			int begin , end;
			BSplineIntegrationData< FEMDegree1 , FEMDegree2 >::InteriorOverlappedSpan( d-1 , begin , end );
			return ( x>=begin && x<end && y>=begin && y<end && z>=begin && z<end );
		}
		else return false;
	}
	template< int FEMDegree1 , int FEMDegree2 > static bool _IsInteriorlyOverlapped( const TreeOctNode* node )
	{
		if( !node ) return false;
		int d , off[3];
		node->depthAndOffset( d , off );
		return _IsInteriorlyOverlapped< FEMDegree1 , FEMDegree2 >( d , off[0] , off[1] , off[2] );
	}
	static void _DepthAndOffset( const TreeOctNode* node , int& d , int off[3] ){ node->depthAndOffset( d , off ) ; d -= 1; }
	static int  _Depth( const TreeOctNode* node ){ return node->depth()-1; }
	static void _StartAndWidth( const TreeOctNode* node , Point3D< Real >& start , Real& width )
	{
		int d , off[3];
		_DepthAndOffset( node , d , off );
		if( d>=0 ) width = Real( 1.0 / (1<<  d ) );
		else       width = Real( 1.0 * (1<<(-d)) );
		for( int dd=0 ; dd<DIMENSION ; dd++ ) start[dd] = Real( off[dd] ) * width;
	}
	static void _CenterAndWidth( const TreeOctNode* node , Point3D< Real >& center , Real& width )
	{
		int d , off[3];
		_DepthAndOffset( node , d , off );
		width = Real( 1.0 / (1<<d) );
		for( int dd=0 ; dd<DIMENSION ; dd++ ) center[dd] = Real( off[dd] + 0.5 ) * width;
	}
	template< int LeftRadius , int RightRadius >
	static typename TreeOctNode::ConstNeighbors< LeftRadius + RightRadius + 1 >& _Neighbors( TreeOctNode::ConstNeighborKey< LeftRadius , RightRadius >& key , int depth ){ return key.neighbors[ depth + 1 ]; }
	template< int LeftRadius , int RightRadius >
	static typename TreeOctNode::Neighbors< LeftRadius + RightRadius + 1 >& _Neighbors( TreeOctNode::NeighborKey< LeftRadius , RightRadius >& key , int depth ){ return key.neighbors[ depth + 1 ]; }
	template< int LeftRadius , int RightRadius >
	static const typename TreeOctNode::template Neighbors< LeftRadius + RightRadius + 1 >& _Neighbors( const typename TreeOctNode::template NeighborKey< LeftRadius , RightRadius >& key , int depth ){ return key.neighbors[ depth + 1 ]; }
	template< int LeftRadius , int RightRadius >
	static const typename TreeOctNode::template ConstNeighbors< LeftRadius + RightRadius + 1 >& _Neighbors( const typename TreeOctNode::template ConstNeighborKey< LeftRadius , RightRadius >& key , int depth ){ return key.neighbors[ depth + 1 ]; }

	static void _SetFullDepth( TreeOctNode* node , int depth );
	void _setFullDepth( int depth );

	////////////////////////////////////
	// System construction code       //
	// MultiGridOctreeData.System.inl //
	////////////////////////////////////
	template< int FEMDegree >
	void _setMultiColorIndices( int start , int end , std::vector< std::vector< int > >& indices ) const;
	template< int FEMDegree >
	int _SolveSystemGS( const BSplineData< FEMDegree >& bsData , SparseNodeData< PointData< Real > , 0 >& pointInfo , int depth , DenseNodeData< Real , FEMDegree >& solution , DenseNodeData< Real , FEMDegree >& constraints , DenseNodeData< Real , FEMDegree >& metSolutionConstraints , int iters , bool coarseToFine , bool showResidual=false , double* bNorm2=NULL , double* inRNorm2=NULL , double* outRNorm2=NULL , bool forceSilent=false );
	template< int FEMDegree >
	int _SolveSystemCG( const BSplineData< FEMDegree >& bsData , SparseNodeData< PointData< Real > , 0 >& pointInfo , int depth , DenseNodeData< Real , FEMDegree >& solution , DenseNodeData< Real , FEMDegree >& constraints , DenseNodeData< Real , FEMDegree >& metSolutionConstraints , int iters , bool coarseToFine , bool showResidual=false , double* bNorm2=NULL , double* inRNorm2=NULL , double* outRNorm2=NULL , double accuracy=0 );
	template< int FEMDegree >
	int _SetMatrixRow( const SparseNodeData< PointData< Real > , 0 >& pointInfo , const typename TreeOctNode::Neighbors< BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapSize >& neighbors , Pointer( MatrixEntry< Real > ) row , int offset , const typename BSplineIntegrationData< FEMDegree , FEMDegree >::FunctionIntegrator::Integrator& integrator , const Stencil< double , BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapSize >& stencil , const BSplineData< FEMDegree >& bsData ) const;
	template< int FEMDegree >
	int _GetMatrixRowSize( const typename TreeOctNode::Neighbors< BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapSize >& neighbors ) const;

	template< int FEMDegree1 , int FEMDegree2 > static void _SetParentOverlapBounds( const TreeOctNode* node , int& startX , int& endX , int& startY , int& endY , int& startZ , int& endZ );
	template< int FEMDegree >
	void _UpdateConstraintsFromCoarser( const SparseNodeData< PointData< Real > , 0 >& pointInfo , const typename TreeOctNode::Neighbors< BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapSize >& neighbors , const typename TreeOctNode::Neighbors< BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapSize >& pNeighbors , TreeOctNode* node , DenseNodeData< Real , FEMDegree >& constraints , const DenseNodeData< Real , FEMDegree >& metSolution , const typename BSplineIntegrationData< FEMDegree , FEMDegree >::FunctionIntegrator::ChildIntegrator& childIntegrator , const Stencil< double , BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapSize >& stencil , const BSplineData< FEMDegree >& bsData ) const;
	// Updates the constraints @(depth-1) based on the solution coefficients @(depth)
	template< int FEMDegree >
	void _UpdateConstraintsFromFiner( const typename BSplineIntegrationData< FEMDegree , FEMDegree >::FunctionIntegrator::ChildIntegrator& childIntegrator , const BSplineData< FEMDegree >& bsData , int highDepth , const DenseNodeData< Real , FEMDegree >& fineSolution , DenseNodeData< Real , FEMDegree >& coarseConstraints ) const;
	// Evaluate the points @(depth) using coefficients @(depth-1)
	template< int FEMDegree >
	void _SetPointValuesFromCoarser( SparseNodeData< PointData< Real > , 0 >& pointInfo , int highDepth , const BSplineData< FEMDegree >& bsData , const DenseNodeData< Real , FEMDegree >& upSampledCoefficients );
	// Evalutes the solution @(depth) at the points @(depth-1) and updates the met constraints @(depth-1)
	template< int FEMDegree >
	void _SetPointConstraintsFromFiner( const SparseNodeData< PointData< Real > , 0 >& pointInfo , int highDepth , const BSplineData< FEMDegree >& bsData , const DenseNodeData< Real , FEMDegree >& finerCoefficients , DenseNodeData< Real , FEMDegree >& metConstraints ) const;
	template< int FEMDegree >
	Real _CoarserFunctionValue( Point3D< Real > p , const PointSupportKey< FEMDegree >& neighborKey , const TreeOctNode* node , const BSplineData< FEMDegree >& bsData , const DenseNodeData< Real , FEMDegree >& upSampledCoefficients ) const;
	template< int FEMDegree >
	Real _FinerFunctionValue  ( Point3D< Real > p , const PointSupportKey< FEMDegree >& neighborKey , const TreeOctNode* node , const BSplineData< FEMDegree >& bsData , const DenseNodeData< Real , FEMDegree >& coefficients ) const;
	template< int FEMDegree >
	int _GetSliceMatrixAndUpdateConstraints( const SparseNodeData< PointData< Real > , 0 >& pointInfo , SparseMatrix< Real >& matrix , DenseNodeData< Real , FEMDegree >& constraints , typename BSplineIntegrationData< FEMDegree , FEMDegree >::FunctionIntegrator::Integrator& integrator , typename BSplineIntegrationData< FEMDegree , FEMDegree >::FunctionIntegrator::ChildIntegrator& childIntegrator , const BSplineData< FEMDegree >& bsData , int depth , int slice , const DenseNodeData< Real , FEMDegree >& metSolution , bool coarseToFine );
	template< int FEMDegree >
	int _GetMatrixAndUpdateConstraints( const SparseNodeData< PointData< Real > , 0 >& pointInfo , SparseMatrix< Real >& matrix , DenseNodeData< Real , FEMDegree >& constraints , typename BSplineIntegrationData< FEMDegree , FEMDegree >::FunctionIntegrator::Integrator& integrator , typename BSplineIntegrationData< FEMDegree , FEMDegree >::FunctionIntegrator::ChildIntegrator& childIntegrator , const BSplineData< FEMDegree >& bsData , int depth , const DenseNodeData< Real , FEMDegree >* metSolution , bool coarseToFine );

	// Down samples constraints @(depth) to constraints @(depth-1)
	template< class C , int FEMDegree > void _DownSample( int highDepth , DenseNodeData< C , FEMDegree >& constraints ) const;
	// Up samples coefficients @(depth-1) to coefficients @(depth)
	template< class C , int FEMDegree > void _UpSample( int highDepth , DenseNodeData< C , FEMDegree >& coefficients ) const;
	template< class C , int FEMDegree > static void _UpSample( int highDepth , ConstPointer( C ) lowCoefficients , Pointer( C ) highCoefficients , bool dirichlet , int threads );

	/////////////////////////////////////////////
	// Code for splatting point-sample data    //
	// MultiGridOctreeData.WeightedSamples.inl //
	/////////////////////////////////////////////
	template< int WeightDegree >
	void _AddWeightContribution( SparseNodeData< Real , WeightDegree >& densityWeights , TreeOctNode* node , Point3D< Real > position , PointSupportKey< WeightDegree >& weightKey , Real weight=Real(1.0) );
	template< int WeightDegree >
	Real _GetSamplesPerNode( const SparseNodeData< Real , WeightDegree >& densityWeights , const TreeOctNode* node , Point3D< Real > position , ConstPointSupportKey< WeightDegree >& weightKey ) const;
	template< int WeightDegree >
	Real _GetSamplesPerNode( const SparseNodeData< Real , WeightDegree >& densityWeights ,       TreeOctNode* node , Point3D< Real > position ,      PointSupportKey< WeightDegree >& weightKey );
	template< int WeightDegree >
	void _GetSampleDepthAndWeight( const SparseNodeData< Real , WeightDegree >& densityWeights , const TreeOctNode* node , Point3D< Real > position , ConstPointSupportKey< WeightDegree >& weightKey , Real& depth , Real& weight ) const;
	template< int WeightDegree >
	void _GetSampleDepthAndWeight( const SparseNodeData< Real , WeightDegree >& densityWeights ,       TreeOctNode* node , Point3D< Real > position ,      PointSupportKey< WeightDegree >& weightKey , Real& depth , Real& weight );
public:
	template< int WeightDegree >
	void _GetSampleDepthAndWeight( const SparseNodeData< Real , WeightDegree >& densityWeights , Point3D< Real > position ,      PointSupportKey< WeightDegree >& weightKey , Real& depth , Real& weight );
	template< int WeightDegree >
	void _GetSampleDepthAndWeight( const SparseNodeData< Real , WeightDegree >& densityWeights , Point3D< Real > position , ConstPointSupportKey< WeightDegree >& weightKey , Real& depth , Real& weight );
protected:
	template< int DataDegree , class V > void _SplatPointData( TreeOctNode* node , Point3D< Real > point , V v , SparseNodeData< V , DataDegree >& data , PointSupportKey< DataDegree >& dataKey );
	template< int WeightDegree , int DataDegree , class V > Real      _SplatPointData( const SparseNodeData< Real , WeightDegree >& densityWeights , Point3D< Real > point , V v , SparseNodeData< V , DataDegree >& data , PointSupportKey< WeightDegree >& weightKey , PointSupportKey< DataDegree >& dataKey , int minDepth , int maxDepth , int dim=DIMENSION );
	template< int WeightDegree , int DataDegree , class V > void _MultiSplatPointData( const SparseNodeData< Real , WeightDegree >* densityWeights , Point3D< Real > point , V v , SparseNodeData< V , DataDegree >& data , PointSupportKey< WeightDegree >& weightKey , PointSupportKey< DataDegree >& dataKey , int maxDepth , int dim=DIMENSION );
	template< class V , int DataDegree > V _Evaluate( const DenseNodeData< V , DataDegree >& coefficients , Point3D< Real > p , const BSplineData< DataDegree >& bsData , const ConstPointSupportKey< DataDegree >& neighborKey ) const;
	template< class V , int DataDegree > V _Evaluate( const SparseNodeData< V , DataDegree >& coefficients , Point3D< Real > p , const BSplineData< DataDegree >& bsData , const ConstPointSupportKey< DataDegree >& dataKey ) const;
public:
	template< class V , int DataDegree > V Evaluate( const  DenseNodeData< V , DataDegree >& coefficients , Point3D< Real > p , const BSplineData< DataDegree >& bsData ) const;
	template< class V , int DataDegree > V Evaluate( const SparseNodeData< V , DataDegree >& coefficients , Point3D< Real > p , const BSplineData< DataDegree >& bsData ) const;
	template< class V , int DataDegree > Pointer( V ) Evaluate( const DenseNodeData< V , DataDegree >& coefficients , int& res , Real isoValue=0.f , int depth=-1 , bool primal=false );

	template< int NormalDegree > int _HasNormals( TreeOctNode* node , const SparseNodeData< Point3D< Real > , NormalDegree >& normalInfo );
	void _MakeComplete( void );
	void _SetValidityFlags( void );
	template< int NormalDegree > void _ClipTree( const SparseNodeData< Point3D< Real > , NormalDegree >& normalInfo );

	////////////////////////////////////
	// Evaluation Methods             //
	// MultiGridOctreeData.Evaluation //
	////////////////////////////////////
	static const int CHILDREN = Cube::CORNERS;
	template< int FEMDegree >
	struct _Evaluator
	{
		typename BSplineEvaluationData< FEMDegree >::Evaluator evaluator;
		typename BSplineEvaluationData< FEMDegree >::ChildEvaluator childEvaluator;
		Stencil< double , BSplineEvaluationData< FEMDegree >::SupportSize > cellStencil;
		Stencil< double , BSplineEvaluationData< FEMDegree >::SupportSize > cellStencils  [CHILDREN];
		Stencil< double , BSplineEvaluationData< FEMDegree >::SupportSize > edgeStencil             [Cube::EDGES  ];
		Stencil< double , BSplineEvaluationData< FEMDegree >::SupportSize > edgeStencils  [CHILDREN][Cube::EDGES  ];
		Stencil< double , BSplineEvaluationData< FEMDegree >::SupportSize > faceStencil             [Cube::FACES  ];
		Stencil< double , BSplineEvaluationData< FEMDegree >::SupportSize > faceStencils  [CHILDREN][Cube::FACES  ];
		Stencil< double , BSplineEvaluationData< FEMDegree >::SupportSize > cornerStencil           [Cube::CORNERS];
		Stencil< double , BSplineEvaluationData< FEMDegree >::SupportSize > cornerStencils[CHILDREN][Cube::CORNERS];

		Stencil< Point3D< double > , BSplineEvaluationData< FEMDegree >::SupportSize > dCellStencil;
		Stencil< Point3D< double > , BSplineEvaluationData< FEMDegree >::SupportSize > dCellStencils  [CHILDREN];
		Stencil< Point3D< double > , BSplineEvaluationData< FEMDegree >::SupportSize > dEdgeStencil             [Cube::EDGES  ];
		Stencil< Point3D< double > , BSplineEvaluationData< FEMDegree >::SupportSize > dEdgeStencils  [CHILDREN][Cube::EDGES  ];
		Stencil< Point3D< double > , BSplineEvaluationData< FEMDegree >::SupportSize > dFaceStencil             [Cube::FACES  ];
		Stencil< Point3D< double > , BSplineEvaluationData< FEMDegree >::SupportSize > dFaceStencils  [CHILDREN][Cube::FACES  ];
		Stencil< Point3D< double > , BSplineEvaluationData< FEMDegree >::SupportSize > dCornerStencil           [Cube::CORNERS];
		Stencil< Point3D< double > , BSplineEvaluationData< FEMDegree >::SupportSize > dCornerStencils[CHILDREN][Cube::CORNERS];
		void set( int depth , bool dirichlet );
	};
	template< class V , int FEMDegree >
	V _getCenterValue( const ConstPointSupportKey< FEMDegree >& neighborKey , const TreeOctNode* node ,              const DenseNodeData< V , FEMDegree >& solution , const DenseNodeData< V , FEMDegree >& metSolution , const _Evaluator< FEMDegree >& evaluator , bool isInterior ) const;
	template< class V , int FEMDegree >
	V _getCornerValue( const ConstPointSupportKey< FEMDegree >& neighborKey , const TreeOctNode* node , int corner , const DenseNodeData< V , FEMDegree >& solution , const DenseNodeData< V , FEMDegree >& metSolution , const _Evaluator< FEMDegree >& evaluator , bool isInterior ) const;
	template< class V , int FEMDegree >
	V _getEdgeValue  ( const ConstPointSupportKey< FEMDegree >& neighborKey , const TreeOctNode* node , int edge   , const DenseNodeData< V , FEMDegree >& solution , const DenseNodeData< V , FEMDegree >& metSolution , const _Evaluator< FEMDegree >& evaluator , bool isInterior ) const;

	template< int FEMDegree >
	std::pair< Real , Point3D< Real > > _getCornerValueAndGradient( const ConstPointSupportKey< FEMDegree >& neighborKey , const TreeOctNode* node , int corner , const DenseNodeData< Real , FEMDegree >& solution , const DenseNodeData< Real , FEMDegree >& metSolution , const _Evaluator< FEMDegree >& evaluator , bool isInterior ) const;
	template< int FEMDegree >
	std::pair< Real , Point3D< Real > > _getEdgeValueAndGradient  ( const ConstPointSupportKey< FEMDegree >& neighborKey , const TreeOctNode* node , int edge   , const DenseNodeData< Real , FEMDegree >& solution , const DenseNodeData< Real , FEMDegree >& metSolution , const _Evaluator< FEMDegree >& evaluator , bool isInterior ) const;

	////////////////////////////////////////
	// Iso-Surfacing Methods              //
	// MultiGridOctreeData.IsoSurface.inl //
	////////////////////////////////////////
	struct IsoEdge
	{
		long long edges[2];
		IsoEdge( void ){ edges[0] = edges[1] = 0; }
		IsoEdge( long long v1 , long long v2 ){ edges[0] = v1 , edges[1] = v2; }
		long long& operator[]( int idx ){ return edges[idx]; }
		const long long& operator[]( int idx ) const { return edges[idx]; }
	};
	struct FaceEdges
	{
		IsoEdge edges[2];
		int count;
	};
	template< class Vertex >
	struct SliceValues
	{
		typename SortedTreeNodes::SliceTableData sliceData;
		Pointer( Real ) cornerValues ; Pointer( Point3D< Real > ) cornerGradients ; Pointer( char ) cornerSet;
		Pointer( long long ) edgeKeys ; Pointer( char ) edgeSet;
		Pointer( FaceEdges ) faceEdges ; Pointer( char ) faceSet;
		Pointer( char ) mcIndices;
		hash_map< long long , std::vector< IsoEdge > > faceEdgeMap;
		hash_map< long long , std::pair< int , Vertex > > edgeVertexMap;
		hash_map< long long , long long > vertexPairMap;

		SliceValues( void );
		~SliceValues( void );
		void reset( bool nonLinearFit );
	protected:
		int _oldCCount , _oldECount , _oldFCount , _oldNCount;
	};
	template< class Vertex >
	struct XSliceValues
	{
		typename SortedTreeNodes::XSliceTableData xSliceData;
		Pointer( long long ) edgeKeys ; Pointer( char ) edgeSet;
		Pointer( FaceEdges ) faceEdges ; Pointer( char ) faceSet;
		hash_map< long long , std::vector< IsoEdge > > faceEdgeMap;
		hash_map< long long , std::pair< int , Vertex > > edgeVertexMap;
		hash_map< long long , long long > vertexPairMap;

		XSliceValues( void );
		~XSliceValues( void );
		void reset( void );
	protected:
		int _oldECount , _oldFCount;
	};
	template< class Vertex >
	struct SlabValues
	{
		XSliceValues< Vertex > _xSliceValues[2];
		SliceValues< Vertex > _sliceValues[2];
		SliceValues< Vertex >& sliceValues( int idx ){ return _sliceValues[idx&1]; }
		const SliceValues< Vertex >& sliceValues( int idx ) const { return _sliceValues[idx&1]; }
		XSliceValues< Vertex >& xSliceValues( int idx ){ return _xSliceValues[idx&1]; }
		const XSliceValues< Vertex >& xSliceValues( int idx ) const { return _xSliceValues[idx&1]; }
	};
	template< class Vertex , int FEMDegree >
	void SetSliceIsoCorners( const DenseNodeData< Real , FEMDegree >& solution , const DenseNodeData< Real , FEMDegree >& coarseSolution , Real isoValue , int depth , int slice ,         std::vector< SlabValues< Vertex > >& sValues , const _Evaluator< FEMDegree >& evaluator , int threads );
	template< class Vertex , int FEMDegree >
	void SetSliceIsoCorners( const DenseNodeData< Real , FEMDegree >& solution , const DenseNodeData< Real , FEMDegree >& coarseSolution , Real isoValue , int depth , int slice , int z , std::vector< SlabValues< Vertex > >& sValues , const _Evaluator< FEMDegree >& evaluator , int threads );
	template< int WeightDegree , int ColorDegree , class Vertex >
	void SetSliceIsoVertices( const BSplineData< ColorDegree >* colorBSData , const SparseNodeData< Real , WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Point3D< Real > > , ColorDegree >* colorData , Real isoValue , int depth , int slice ,         int& vOffset , CoredMeshData< Vertex >& mesh , std::vector< SlabValues< Vertex > >& sValues , int threads );
	template< int WeightDegree , int ColorDegree , class Vertex >
	void SetSliceIsoVertices( const BSplineData< ColorDegree >* colorBSData , const SparseNodeData< Real , WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Point3D< Real > > , ColorDegree >* colorData , Real isoValue , int depth , int slice , int z , int& vOffset , CoredMeshData< Vertex >& mesh , std::vector< SlabValues< Vertex > >& sValues , int threads );
	template< int WeightDegree , int ColorDegree , class Vertex >
	void SetXSliceIsoVertices( const BSplineData< ColorDegree >* colorBSData , const SparseNodeData< Real , WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Point3D< Real > > , ColorDegree >* colorData , Real isoValue , int depth , int slab , int& vOffset , CoredMeshData< Vertex >& mesh , std::vector< SlabValues< Vertex > >& sValues , int threads );
	template< class Vertex >
	void CopyFinerSliceIsoEdgeKeys( int depth , int slice ,         std::vector< SlabValues< Vertex > >& sValues , int threads );
	template< class Vertex >
	void CopyFinerSliceIsoEdgeKeys( int depth , int slice , int z , std::vector< SlabValues< Vertex > >& sValues , int threads );
	template< class Vertex >
	void CopyFinerXSliceIsoEdgeKeys( int depth , int slab , std::vector< SlabValues< Vertex > >& sValues , int threads );
	template< class Vertex >
	void SetSliceIsoEdges( int depth , int slice ,         std::vector< SlabValues< Vertex > >& slabValues , int threads );
	template< class Vertex >
	void SetSliceIsoEdges( int depth , int slice , int z , std::vector< SlabValues< Vertex > >& slabValues , int threads );
	template< class Vertex >
	void SetXSliceIsoEdges( int depth , int slice , std::vector< SlabValues< Vertex > >& slabValues , int threads );

	template< class Vertex >
	void SetIsoSurface( int depth , int offset , const SliceValues< Vertex >& bValues , const SliceValues< Vertex >& fValues , const XSliceValues< Vertex >& xValues , CoredMeshData< Vertex >& mesh , bool polygonMesh , bool addBarycenter , int& vOffset , int threads );

	template< class Vertex >
	static int AddIsoPolygons( CoredMeshData< Vertex >& mesh , std::vector< std::pair< int , Vertex > >& polygon , bool polygonMesh , bool addBarycenter , int& vOffset );

	template< int WeightDegree , int ColorDegree , class Vertex >
	bool GetIsoVertex( const BSplineData< ColorDegree >* colorBSData , const SparseNodeData< Real , WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Point3D< Real > > , ColorDegree >* colorData , Real isoValue , ConstPointSupportKey< WeightDegree >& weightKey , ConstPointSupportKey< ColorDegree >& colorKey , const TreeOctNode* node , int edgeIndex , int z , const SliceValues< Vertex >& sValues , Vertex& vertex );
	template< int WeightDegree , int ColorDegree , class Vertex >
	bool GetIsoVertex( const BSplineData< ColorDegree >* colorBSData , const SparseNodeData< Real , WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Point3D< Real > > , ColorDegree >* colorData , Real isoValue , ConstPointSupportKey< WeightDegree >& weightKey , ConstPointSupportKey< ColorDegree >& colorKey , const TreeOctNode* node , int cornerIndex , const SliceValues< Vertex >& bValues , const SliceValues< Vertex >& fValues , Vertex& vertex );

public:
	static double maxMemoryUsage;
	int threads;

	static double MemoryUsage( void );
	Octree( void );

	// After calling set tree, the indices of the octree node will be stored by depth, and within depth they will be sorted by slice
	template< class PointReal , int NormalDegree , int WeightDegree , int DataDegree , class Data , class _Data >
	int SetTree( OrientedPointStream< PointReal >* pointStream , int minDepth , int maxDepth , int fullDepth , int splatDepth , Real samplesPerNode ,
		Real scaleFactor , bool useConfidence , bool useNormalWeight ,
		Real constraintWeight , int adaptiveExponent ,
		SparseNodeData< Real , WeightDegree >& densityWeights , SparseNodeData< PointData< Real > , 0 >& pointInfo , SparseNodeData< Point3D< Real > , NormalDegree >& normalInfo , SparseNodeData< Real , NormalDegree >& nodeWeights ,
		SparseNodeData< ProjectiveData< _Data > , DataDegree >* dataValues ,
		XForm4x4< Real >& xForm , bool dirichlet=false , bool makeComplete=false );

	template< int FEMDegree > void EnableMultigrid( std::vector< int >* map );

	template< int FEMDegree , int NormalDegree >
	DenseNodeData< Real , FEMDegree > SetLaplacianConstraints( const SparseNodeData< Point3D< Real > , NormalDegree >& normalInfo );
	template< int FEMDegree >
	DenseNodeData< Real , FEMDegree > SolveSystem( SparseNodeData< PointData< Real > , 0 >& pointInfo , DenseNodeData< Real , FEMDegree >& constraints , bool showResidual , int iters , int maxSolveDepth , int cgDepth=0 , double cgAccuracy=0 );

	template< int FEMDegree , int NormalDegree >
	Real GetIsoValue( const DenseNodeData< Real , FEMDegree >& solution , const SparseNodeData< Real , NormalDegree >& nodeWeights );
	template< int FEMDegree , int WeightDegree , int ColorDegree , class Vertex >
	void GetMCIsoSurface( const SparseNodeData< Real , WeightDegree >* densityWeights , const SparseNodeData< ProjectiveData< Point3D< Real > > , ColorDegree >* colorData , const DenseNodeData< Real , FEMDegree >& solution , Real isoValue , CoredMeshData< Vertex >& mesh , bool nonLinearFit=true , bool addBarycenter=false , bool polygonMesh=false );

	const TreeOctNode& tree( void ) const{ return _tree; }
	size_t leaves( void ) const { return _tree.leaves(); }
	size_t nodes( void ) const { return _tree.nodes(); }
};

template< class Real >
void Reset( void )
{
	TreeNodeData::NodeCount=0;
	Octree< Real >::maxMemoryUsage = 0;
}

#include "MultiGridOctreeData.inl"
#include "MultiGridOctreeData.SortedTreeNodes.inl"
#include "MultiGridOctreeData.WeightedSamples.inl"
#include "MultiGridOctreeData.System.inl"
#include "MultiGridOctreeData.IsoSurface.inl"
#include "MultiGridOctreeData.Evaluation.inl"
#endif // MULTI_GRID_OCTREE_DATA_INCLUDED
