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

template< int Degree1 , int Degree2 >
double SystemCoefficients< Degree1 , Degree2 >::GetLaplacian( const typename FunctionIntegrator::Integrator& integrator , const int off1[] , const int off2[] )
{
	double vv[] = { integrator.dot( off1[0] , off2[0] , false , false ) , integrator.dot( off1[1] , off2[1] , false , false ) , integrator.dot( off1[2] , off2[2] , false , false ) };
	double dd[] = { integrator.dot( off1[0] , off2[0] , true  , true  ) , integrator.dot( off1[1] , off2[1] , true  , true  ) , integrator.dot( off1[2] , off2[2] , true  , true  ) };
	return dd[0]*vv[1]*vv[2] + vv[0]*dd[1]*vv[2] + vv[0]*vv[1]*dd[2];
}
template< int Degree1 , int Degree2 >
double SystemCoefficients< Degree1 , Degree2 >::GetLaplacian( const typename FunctionIntegrator::ChildIntegrator& integrator , const int off1[] , const int off2[] )
{
	double vv[] = { integrator.dot( off1[0] , off2[0] , false , false ) , integrator.dot( off1[1] , off2[1] , false , false ) , integrator.dot( off1[2] , off2[2] , false , false ) };
	double dd[] = { integrator.dot( off1[0] , off2[0] , true  , true  ) , integrator.dot( off1[1] , off2[1] , true  , true  ) , integrator.dot( off1[2] , off2[2] , true  , true  ) };
	return dd[0]*vv[1]*vv[2] + vv[0]*dd[1]*vv[2] + vv[0]*vv[1]*dd[2];
}
template< int Degree1 , int Degree2 >
double SystemCoefficients< Degree1 , Degree2 >::GetDivergence1( const typename FunctionIntegrator::Integrator& integrator , const int off1[] , const int off2[] , Point3D< double > normal1 )
{
	return Point3D< double >::Dot( GetDivergence1( integrator , off1 , off2 ) , normal1 );
}
template< int Degree1 , int Degree2 >
double SystemCoefficients< Degree1 , Degree2 >::GetDivergence1( const typename FunctionIntegrator::ChildIntegrator& integrator , const int off1[] , const int off2[] , Point3D< double > normal1 )
{
	return Point3D< double >::Dot( GetDivergence1( integrator , off1 , off2 ) , normal1 );
}
template< int Degree1 , int Degree2 >
double SystemCoefficients< Degree1 , Degree2 >::GetDivergence2( const typename FunctionIntegrator::Integrator& integrator , const int off1[] , const int off2[] , Point3D< double > normal2 )
{
	return Point3D< double >::Dot( GetDivergence2( integrator , off1 , off2 ) , normal2 );
}
template< int Degree1 , int Degree2 >
double SystemCoefficients< Degree1 , Degree2 >::GetDivergence2( const typename FunctionIntegrator::ChildIntegrator& integrator , const int off1[] , const int off2[] , Point3D< double > normal2 )
{
	return Point3D< double >::Dot( GetDivergence2( integrator , off1 , off2 ) , normal2 );
}
template< int Degree1 , int Degree2 >
Point3D< double > SystemCoefficients< Degree1 , Degree2 >::GetDivergence1( const typename FunctionIntegrator::Integrator& integrator , const int off1[] , const int off2[] )
{
	double vv[] = { integrator.dot( off1[0] , off2[0] , false , false ) , integrator.dot( off1[1] , off2[1] , false , false ) , integrator.dot( off1[2] , off2[2] , false , false ) };
#if GRADIENT_DOMAIN_SOLUTION
	// Take the dot-product of the vector-field with the gradient of the basis function
	double vd[] = { integrator.dot( off1[0] , off2[0] , true , false ) , integrator.dot( off1[1] , off2[1] , true , false ) , integrator.dot( off1[2] , off2[2] , true , false ) };
	return  Point3D< double >( vd[0]*vv[1]*vv[2] , vv[0]*vd[1]*vv[2] , vv[0]*vv[1]*vd[2] );
#else // !GRADIENT_DOMAIN_SOLUTION
	// Take the dot-product of the divergence of the vector-field with the basis function
	double dv[] = { integrator.dot( off1[0] , off2[0] , false , true ) , integrator.dot( off1[1] , off2[1] , false , true ) , integrator.dot( off1[2] , off2[2] , false , true ) };
	return  -Point3D< double >( dv[0]*vv[1]*vv[2] , vv[0]*dv[1]*vv[2] , vv[0]*vv[1]*dv[2] );
#endif // GRADIENT_DOMAIN_SOLUTION
}
template< int Degree1 , int Degree2 >
Point3D< double > SystemCoefficients< Degree1 , Degree2 >::GetDivergence1( const typename FunctionIntegrator::ChildIntegrator& integrator , const int off1[] , const int off2[] )
{
	double vv[] = { integrator.dot( off1[0] , off2[0] , false , false ) , integrator.dot( off1[1] , off2[1] , false , false ) , integrator.dot( off1[2] , off2[2] , false , false ) };
#if GRADIENT_DOMAIN_SOLUTION
	// Take the dot-product of the vector-field with the gradient of the basis function
	double vd[] = { integrator.dot( off1[0] , off2[0] , true , false ) , integrator.dot( off1[1] , off2[1] , true , false ) , integrator.dot( off1[2] , off2[2] , true , false ) };
	return  Point3D< double >( vd[0]*vv[1]*vv[2] , vv[0]*vd[1]*vv[2] , vv[0]*vv[1]*vd[2] );
#else // !GRADIENT_DOMAIN_SOLUTION
	// Take the dot-product of the divergence of the vector-field with the basis function
	double dv[] = { integrator.dot( off1[0] , off2[0] , false , true ) , integrator.dot( off1[1] , off2[1] , false , true ) , integrator.dot( off1[2] , off2[2] , false , true ) };
	return  -Point3D< double >( dv[0]*vv[1]*vv[2] , vv[0]*dv[1]*vv[2] , vv[0]*vv[1]*dv[2] );
#endif // GRADIENT_DOMAIN_SOLUTION
}
template< int Degree1 , int Degree2 >
Point3D< double > SystemCoefficients< Degree1 , Degree2 >::GetDivergence2( const typename FunctionIntegrator::Integrator& integrator , const int off1[] , const int off2[] )
{
	double vv[] = { integrator.dot( off1[0] , off2[0] , false , false ) , integrator.dot( off1[1] , off2[1] , false , false ) , integrator.dot( off1[2] , off2[2] , false , false ) };
#if GRADIENT_DOMAIN_SOLUTION
	// Take the dot-product of the vector-field with the gradient of the basis function
	double dv[] = { integrator.dot( off1[0] , off2[0] , false , true ) , integrator.dot( off1[1] , off2[1] , false , true ) , integrator.dot( off1[2] , off2[2] , false , true ) };
	return  Point3D< double >( dv[0]*vv[1]*vv[2] , vv[0]*dv[1]*vv[2] , vv[0]*vv[1]*dv[2] );
#else // !GRADIENT_DOMAIN_SOLUTION
	// Take the dot-product of the divergence of the vector-field with the basis function
	double vd[] = { integrator.dot( off1[0] , off2[0] , true , false ) , integrator.dot( off1[1] , off2[1] , true , false ) , integrator.dot( off1[2] , off2[2] , true , false ) };
	return -Point3D< double >( vd[0]*vv[1]*vv[2] , vv[0]*vd[1]*vv[2] , vv[0]*vv[1]*vd[2] );
#endif // GRADIENT_DOMAIN_SOLUTION
}
template< int Degree1 , int Degree2 >
Point3D< double > SystemCoefficients< Degree1 , Degree2 >::GetDivergence2( const typename FunctionIntegrator::ChildIntegrator& integrator , const int off1[] , const int off2[] )
{
	double vv[] = { integrator.dot( off1[0] , off2[0] , false , false ) , integrator.dot( off1[1] , off2[1] , false , false ) , integrator.dot( off1[2] , off2[2] , false , false ) };
#if GRADIENT_DOMAIN_SOLUTION
	// Take the dot-product of the vector-field with the gradient of the basis function
	double dv[] = { integrator.dot( off1[0] , off2[0] , false , true ) , integrator.dot( off1[1] , off2[1] , false , true ) , integrator.dot( off1[2] , off2[2] , false , true ) };
	return  Point3D< double >( dv[0]*vv[1]*vv[2] , vv[0]*dv[1]*vv[2] , vv[0]*vv[1]*dv[2] );
#else // !GRADIENT_DOMAIN_SOLUTION
	// Take the dot-product of the divergence of the vector-field with the basis function
	double vd[] = { integrator.dot( off1[0] , off2[0] , true , false ) , integrator.dot( off1[1] , off2[1] , true , false ) , integrator.dot( off1[2] , off2[2] , true , false ) };
	return -Point3D< double >( vd[0]*vv[1]*vv[2] , vv[0]*vd[1]*vv[2] , vv[0]*vv[1]*vd[2] );
#endif // GRADIENT_DOMAIN_SOLUTION
}
// if( scatter ) normals come from the center node
// else          normals come from the neighbors
template< int Degree1 , int Degree2 >
void SystemCoefficients< Degree1 , Degree2 >::SetCentralDivergenceStencil( const typename FunctionIntegrator::Integrator& integrator , Stencil< Point3D< double > , OverlapSize >& stencil , bool scatter )
{
	int center = ( 1<<integrator.depth() )>>1;
	int offset[] = { center , center , center };
	for( int x=0 ; x<OverlapSize ; x++ ) for( int y=0 ; y<OverlapSize ; y++ ) for( int z=0 ; z<OverlapSize ; z++ )
	{
		int _offset[] = { x+center-OverlapEnd , y+center-OverlapEnd , z+center-OverlapEnd };
		stencil.values[x][y][z] = scatter ? GetDivergence1( integrator , _offset , offset ) : GetDivergence2( integrator , _offset , offset );
	}
}
template< int Degree1 , int Degree2 >
void SystemCoefficients< Degree1 , Degree2 >::SetCentralDivergenceStencils( const typename FunctionIntegrator::ChildIntegrator& integrator , Stencil< Point3D< double > , OverlapSize > stencils[2][2][2] , bool scatter )
{
	int center = ( 1<<integrator.childDepth() )>>1;
	for( int i=0 ; i<2 ; i++ ) for( int j=0 ; j<2 ; j++ ) for( int k=0 ; k<2 ; k++ )
	{
		int offset[] = { center+i , center+j , center+k };
		for( int x=0 ; x<OverlapSize ; x++ ) for( int y=0 ; y<OverlapSize ; y++ ) for( int z=0 ; z<OverlapSize ; z++ )
		{
			int _offset[] = { x+center/2-OverlapEnd , y+center/2-OverlapEnd , z+center/2-OverlapEnd };
			stencils[i][j][k].values[x][y][z] = scatter ? GetDivergence1( integrator , _offset , offset ) : GetDivergence2( integrator , _offset , offset );
		}
	}
}
template< int Degree1 , int Degree2 >
void SystemCoefficients< Degree1 , Degree2 >::SetCentralLaplacianStencil( const typename FunctionIntegrator::Integrator& integrator , Stencil< double , OverlapSize >& stencil )
{
	int center = ( 1<<integrator.depth() )>>1;
	int offset[] = { center , center , center };
	for( int x=0 ; x<OverlapSize ; x++ ) for( int y=0 ; y<OverlapSize ; y++ ) for( int z=0 ; z<OverlapSize ; z++ )
	{
		int _offset[] = { x+center-OverlapEnd , y+center-OverlapEnd , z+center-OverlapEnd };
		stencil.values[x][y][z] = GetLaplacian( integrator , _offset , offset );
	}
}
template< int Degree1 , int Degree2 >
void SystemCoefficients< Degree1 , Degree2 >::SetCentralLaplacianStencils( const typename FunctionIntegrator::ChildIntegrator& integrator , Stencil< double , OverlapSize > stencils[2][2][2] )
{
	int center = ( 1<<integrator.childDepth() )>>1;
	for( int i=0 ; i<2 ; i++ ) for( int j=0 ; j<2 ; j++ ) for( int k=0 ; k<2 ; k++ )
	{
		int offset[] = { center+i , center+j , center+k };
		for( int x=0 ; x<OverlapSize ; x++ ) for( int y=0 ; y<OverlapSize ; y++ ) for( int z=0 ; z<OverlapSize ; z++ )
		{
			int _offset[] = { x+center/2-OverlapEnd , y+center/2-OverlapEnd , z+center/2-OverlapEnd };
			stencils[i][j][k].values[x][y][z] = GetLaplacian( integrator , _offset , offset );
		}
	}
}

template< class Real >
template< int FEMDegree >
void Octree< Real >::_setMultiColorIndices( int start , int end , std::vector< std::vector< int > >& indices ) const
{
	static const int OverlapRadius = - BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapStart;

	const int modulus = OverlapRadius+1;
	indices.resize( modulus*modulus*modulus );
	int count[modulus*modulus*modulus];
	memset( count , 0 , sizeof(int)*modulus*modulus*modulus );
#pragma omp parallel for num_threads( threads )
	for( int i=start ; i<end ; i++ ) if( _IsValidNode< FEMDegree >( _sNodes.treeNodes[i] ) )
	{
		int d , off[3];
		_sNodes.treeNodes[i]->depthAndOffset( d , off );
		int idx = (modulus*modulus) * ( off[2]%modulus ) + modulus * ( off[1]%modulus ) + ( off[0]%modulus );
#pragma omp atomic
		count[idx]++;
	}

	for( int i=0 ; i<modulus*modulus*modulus ; i++ ) indices[i].reserve( count[i] ) , count[i]=0;

	for( int i=start ; i<end ; i++ ) if( _IsValidNode< FEMDegree >( _sNodes.treeNodes[i] ) )
	{
		int d , off[3];
		_sNodes.treeNodes[i]->depthAndOffset( d , off );
		int idx = (modulus*modulus) * ( off[2]%modulus ) + modulus * ( off[1]%modulus ) + ( off[0]%modulus );
		indices[idx].push_back( i - start );
	}
}

template< class Real >
template< class C , int FEMDegree >
void Octree< Real >::_DownSample( int highDepth , DenseNodeData< C , FEMDegree >& constraints ) const
{
	typedef typename TreeOctNode::NeighborKey< -BSplineEvaluationData< FEMDegree >::UpSampleStart , BSplineEvaluationData< FEMDegree >::UpSampleEnd > UpSampleKey;

	int lowDepth = highDepth-1;
	if( lowDepth<_minDepth ) return;

	typename BSplineEvaluationData< FEMDegree >::UpSampleEvaluator upSampleEvaluator;
	BSplineEvaluationData< FEMDegree >::SetUpSampleEvaluator( upSampleEvaluator , lowDepth-1 , _dirichlet );
	std::vector< UpSampleKey > neighborKeys( std::max< int >( 1 , threads ) );
	for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( lowDepth );

	Stencil< double , BSplineEvaluationData< FEMDegree >::UpSampleSize > upSampleStencil;
	int lowCenter = _Dimension< FEMDegree >(lowDepth)>>1;
	for( int i=0 ; i<BSplineEvaluationData< FEMDegree >::UpSampleSize ; i++ ) for( int j=0 ; j<BSplineEvaluationData< FEMDegree >::UpSampleSize ; j++ ) for( int k=0 ; k<BSplineEvaluationData< FEMDegree >::UpSampleSize ; k++ )
		upSampleStencil.values[i][j][k] =
			upSampleEvaluator.value( lowCenter , 2*lowCenter + i + BSplineEvaluationData< FEMDegree >::UpSampleStart ) *
			upSampleEvaluator.value( lowCenter , 2*lowCenter + j + BSplineEvaluationData< FEMDegree >::UpSampleStart ) *
			upSampleEvaluator.value( lowCenter , 2*lowCenter + k + BSplineEvaluationData< FEMDegree >::UpSampleStart );
	int dim = _Dimension< FEMDegree >(lowDepth);

	// Iterate over all (valid) parent nodes
#pragma omp parallel for num_threads( threads )
	for( int i=_sNodes.begin(lowDepth) ; i<_sNodes.end(lowDepth) ; i++ ) if( _IsValidNode< FEMDegree >( _sNodes.treeNodes[i] ) )
	{
		TreeOctNode* pNode = _sNodes.treeNodes[i];

		UpSampleKey& neighborKey = neighborKeys[ omp_get_thread_num() ];
		int d , off[3];
		pNode->depthAndOffset( d , off );

		neighborKey.template getNeighbors< false >( pNode );

		// Get the child neighbors
		typename TreeOctNode::Neighbors< BSplineEvaluationData< FEMDegree >::UpSampleSize > neighbors;
		neighborKey.template getChildNeighbors< false >( 0 , d , neighbors );

		C& coarseConstraint = constraints[i];

		// Want to make sure test if contained children are interior.
		// This is more conservative because we are test that overlapping children are interior
		bool isInterior = _IsInteriorlyOverlapped< FEMDegree , FEMDegree >( pNode );
		if( isInterior )
		{
			for( int ii=0 ; ii<BSplineEvaluationData< FEMDegree >::UpSampleSize ; ii++ ) for( int jj=0 ; jj<BSplineEvaluationData< FEMDegree >::UpSampleSize ; jj++ ) for( int kk=0 ; kk<BSplineEvaluationData< FEMDegree >::UpSampleSize ; kk++ )
			{
				const TreeOctNode* cNode = neighbors.neighbors[ii][jj][kk];
				if( cNode ) coarseConstraint += (C)( constraints[ cNode->nodeData.nodeIndex ] * upSampleStencil.values[ii][jj][kk] );
			}
		}
		else
		{
			double upSampleValues[3][ BSplineEvaluationData< FEMDegree >::UpSampleSize ];
			for( int ii=0 ; ii<BSplineEvaluationData< FEMDegree >::UpSampleSize ; ii++ )
			{
				upSampleValues[0][ii] = upSampleEvaluator.value( off[0] , 2*off[0] + ii + BSplineEvaluationData< FEMDegree >::UpSampleStart );
				upSampleValues[1][ii] = upSampleEvaluator.value( off[1] , 2*off[1] + ii + BSplineEvaluationData< FEMDegree >::UpSampleStart );
				upSampleValues[2][ii] = upSampleEvaluator.value( off[2] , 2*off[2] + ii + BSplineEvaluationData< FEMDegree >::UpSampleStart );
			}

			for( int ii=0 ; ii<BSplineEvaluationData< FEMDegree >::UpSampleSize ; ii++ ) for( int jj=0 ; jj<BSplineEvaluationData< FEMDegree >::UpSampleSize ; jj++ )
			{
				double dxy = upSampleValues[0][ii] * upSampleValues[1][jj];
				for( int kk=0 ; kk<BSplineEvaluationData< FEMDegree >::UpSampleSize ; kk++ )
				{
					const TreeOctNode* cNode = neighbors.neighbors[ii][jj][kk];
					if( _IsValidNode< FEMDegree >( cNode ) ) coarseConstraint += (C)( constraints[ cNode->nodeData.nodeIndex ] * dxy * upSampleValues[2][kk] );
				}
			}
		}
	}
}
template< class Real >
template< class C , int FEMDegree>
void Octree< Real >::_UpSample( int highDepth , DenseNodeData< C , FEMDegree >& coefficients ) const
{
	static const int  LeftDownSampleRadius = -( ( BSplineEvaluationData< FEMDegree >::DownSample0Start < BSplineEvaluationData< FEMDegree >::DownSample1Start ) ? BSplineEvaluationData< FEMDegree >::DownSample0Start : BSplineEvaluationData< FEMDegree >::DownSample1Start );
	static const int RightDownSampleRadius =  ( ( BSplineEvaluationData< FEMDegree >::DownSample0End   > BSplineEvaluationData< FEMDegree >::DownSample1End   ) ? BSplineEvaluationData< FEMDegree >::DownSample0End   : BSplineEvaluationData< FEMDegree >::DownSample1End   );
	typedef TreeOctNode::NeighborKey< LeftDownSampleRadius , RightDownSampleRadius > DownSampleKey;

	int lowDepth = highDepth-1;
	if( lowDepth<_minDepth ) return;

	typename BSplineEvaluationData< FEMDegree >::UpSampleEvaluator upSampleEvaluator;
	BSplineEvaluationData< FEMDegree >::SetUpSampleEvaluator( upSampleEvaluator , lowDepth-1 , _dirichlet );
	std::vector< DownSampleKey > neighborKeys( std::max< int >( 1 , threads ) );
	for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( lowDepth );
	
	static const int DownSampleSize = BSplineEvaluationData< FEMDegree >::DownSample0Size > BSplineEvaluationData< FEMDegree >::DownSample1Size ? BSplineEvaluationData< FEMDegree >::DownSample0Size : BSplineEvaluationData< FEMDegree >::DownSample1Size;
	Stencil< double , DownSampleSize > downSampleStencils[ Cube::CORNERS ];
	int lowCenter = _Dimension< FEMDegree >( lowDepth )>>1;
	for( int c=0 ; c<Cube::CORNERS ; c++ )
	{
		int cx , cy , cz;
		Cube::FactorCornerIndex( c , cx , cy , cz );
		for( int ii=0 ; ii<BSplineEvaluationData< FEMDegree >::DownSampleSize[cx] ; ii++ )
			for( int jj=0 ; jj<BSplineEvaluationData< FEMDegree >::DownSampleSize[cy] ; jj++ )
				for( int kk=0 ; kk<BSplineEvaluationData< FEMDegree >::DownSampleSize[cz] ; kk++ )
					downSampleStencils[c].values[ii][jj][kk] = 
						upSampleEvaluator.value( lowCenter + ii + BSplineEvaluationData< FEMDegree >::DownSampleStart[cx] , 2*lowCenter + cx ) *
						upSampleEvaluator.value( lowCenter + jj + BSplineEvaluationData< FEMDegree >::DownSampleStart[cy] , 2*lowCenter + cy ) *
						upSampleEvaluator.value( lowCenter + kk + BSplineEvaluationData< FEMDegree >::DownSampleStart[cz] , 2*lowCenter + cz ) ;
	}
	int dim = _Dimension< FEMDegree >( lowDepth );

	// For Dirichlet constraints, can't get to all children from parents because boundary nodes are invalid
#pragma omp parallel for num_threads( threads )
	for( int i=_sNodes.begin(highDepth) ; i<_sNodes.end(highDepth) ; i++ ) if( _IsValidNode< FEMDegree >( _sNodes.treeNodes[i] ) )
	{
		TreeOctNode *cNode = _sNodes.treeNodes[i] , *pNode = cNode->parent;
		int c = (int)( cNode-pNode->children );

		DownSampleKey& neighborKey = neighborKeys[ omp_get_thread_num() ];
		int d , off[3];
		pNode->depthAndOffset( d , off );
		typename TreeOctNode::Neighbors< LeftDownSampleRadius + RightDownSampleRadius + 1 >& neighbors = neighborKey.template getNeighbors< false >( pNode );

		// Want to make sure test if contained children are interior.
		// This is more conservative because we are test that overlapping children are interior
		bool isInterior = _IsInteriorlyOverlapped< FEMDegree , FEMDegree >( pNode );

		C& fineCoefficient = coefficients[ cNode->nodeData.nodeIndex ];

		int cx , cy , cz;
		Cube::FactorCornerIndex( c , cx , cy , cz );

		if( isInterior )
		{
			for( int ii=0 ; ii<BSplineEvaluationData< FEMDegree >::DownSampleSize[cx] ; ii++ ) for( int jj=0 ; jj<BSplineEvaluationData< FEMDegree >::DownSampleSize[cy] ; jj++ )
			{
				int _ii = ii + BSplineEvaluationData< FEMDegree >::DownSampleStart[cx] + LeftDownSampleRadius;
				int _jj = jj + BSplineEvaluationData< FEMDegree >::DownSampleStart[cy] + LeftDownSampleRadius;
				for( int kk=0 ; kk<BSplineEvaluationData< FEMDegree >::DownSampleSize[cz] ; kk++ )
				{
					int _kk = kk + BSplineEvaluationData< FEMDegree >::DownSampleStart[cz] + LeftDownSampleRadius;
					const TreeOctNode* _pNode = neighbors.neighbors[_ii][_jj][_kk];
					if( _pNode ) fineCoefficient += (C)( coefficients[ _pNode->nodeData.nodeIndex ] * downSampleStencils[c].values[ii][jj][kk] );
				}
			}
		}
		else
		{
			double downSampleValues[3][ BSplineEvaluationData< FEMDegree >::DownSample0Size > BSplineEvaluationData< FEMDegree >::DownSample1Size ? BSplineEvaluationData< FEMDegree >::DownSample0Size : BSplineEvaluationData< FEMDegree >::DownSample1Size ];

			for( int ii=0 ; ii<BSplineEvaluationData< FEMDegree >::DownSampleSize[cx] ; ii++ ) downSampleValues[0][ii] = upSampleEvaluator.value( off[0] + ii + BSplineEvaluationData< FEMDegree >::DownSampleStart[cx] , 2*off[0] + cx );
			for( int ii=0 ; ii<BSplineEvaluationData< FEMDegree >::DownSampleSize[cy] ; ii++ ) downSampleValues[1][ii] = upSampleEvaluator.value( off[1] + ii + BSplineEvaluationData< FEMDegree >::DownSampleStart[cy] , 2*off[1] + cy );
			for( int ii=0 ; ii<BSplineEvaluationData< FEMDegree >::DownSampleSize[cz] ; ii++ ) downSampleValues[2][ii] = upSampleEvaluator.value( off[2] + ii + BSplineEvaluationData< FEMDegree >::DownSampleStart[cz] , 2*off[2] + cz );

			for( int ii=0 ; ii<BSplineEvaluationData< FEMDegree >::DownSampleSize[cx] ; ii++ ) for( int jj=0 ; jj<BSplineEvaluationData< FEMDegree >::DownSampleSize[cy] ; jj++ )
			{
				double dxy = downSampleValues[0][ii] * downSampleValues[1][jj];
				int _ii = ii + BSplineEvaluationData< FEMDegree >::DownSampleStart[cx] + LeftDownSampleRadius;
				int _jj = jj + BSplineEvaluationData< FEMDegree >::DownSampleStart[cy] + LeftDownSampleRadius;
				for( int kk=0 ; kk<BSplineEvaluationData< FEMDegree >::DownSampleSize[cz] ; kk++ )
				{
					int _kk = kk + BSplineEvaluationData< FEMDegree >::DownSampleStart[cz] + LeftDownSampleRadius;
					const TreeOctNode* _pNode = neighbors.neighbors[_ii][_jj][_kk];
					if( _IsValidNode< FEMDegree >( _pNode ) ) fineCoefficient += (C)( coefficients[ _pNode->nodeData.nodeIndex ] * dxy * downSampleValues[2][kk] );
				}
			}
		}
	}
}

template< class Real >
template< class C , int FEMDegree >
void Octree< Real >::_UpSample( int highDepth , ConstPointer( C ) lowCoefficients , Pointer( C ) highCoefficients , bool dirichlet , int threads )
{
	static const int  LeftDownSampleRadius = -( ( BSplineEvaluationData< FEMDegree >::DownSample0Start < BSplineEvaluationData< FEMDegree >::DownSample1Start ) ? BSplineEvaluationData< FEMDegree >::DownSample0Start : BSplineEvaluationData< FEMDegree >::DownSample1Start );
	static const int RightDownSampleRadius =  ( ( BSplineEvaluationData< FEMDegree >::DownSample0End   > BSplineEvaluationData< FEMDegree >::DownSample1End   ) ? BSplineEvaluationData< FEMDegree >::DownSample0End   : BSplineEvaluationData< FEMDegree >::DownSample1End   );
	typedef TreeOctNode::NeighborKey< LeftDownSampleRadius , RightDownSampleRadius > DownSampleKey;

	int lowDepth = highDepth-1;
	if( lowDepth<1 ) return;

	typename BSplineEvaluationData< FEMDegree >::UpSampleEvaluator upSampleEvaluator;
	BSplineEvaluationData< FEMDegree >::SetUpSampleEvaluator( upSampleEvaluator , lowDepth-1 , dirichlet );
	std::vector< DownSampleKey > neighborKeys( std::max< int >( 1 , threads ) );

	static const int DownSampleSize = BSplineEvaluationData< FEMDegree >::DownSample0Size > BSplineEvaluationData< FEMDegree >::DownSample1Size ? BSplineEvaluationData< FEMDegree >::DownSample0Size : BSplineEvaluationData< FEMDegree >::DownSample1Size;
	Stencil< double , DownSampleSize > downSampleStencils[ Cube::CORNERS ];
	int lowCenter = _Dimension< FEMDegree >( lowDepth )>>1;
	for( int c=0 ; c<Cube::CORNERS ; c++ )
	{
		int cx , cy , cz;
		Cube::FactorCornerIndex( c , cx , cy , cz );
		for( int ii=0 ; ii<BSplineEvaluationData< FEMDegree >::DownSampleSize[cx] ; ii++ )
			for( int jj=0 ; jj<BSplineEvaluationData< FEMDegree >::DownSampleSize[cy] ; jj++ )
				for( int kk=0 ; kk<BSplineEvaluationData< FEMDegree >::DownSampleSize[cz] ; kk++ )
					downSampleStencils[c].values[ii][jj][kk] = 
						upSampleEvaluator.value( lowCenter + ii + BSplineEvaluationData< FEMDegree >::DownSampleStart[cx] , 2*lowCenter + cx ) *
						upSampleEvaluator.value( lowCenter + jj + BSplineEvaluationData< FEMDegree >::DownSampleStart[cy] , 2*lowCenter + cy ) *
						upSampleEvaluator.value( lowCenter + kk + BSplineEvaluationData< FEMDegree >::DownSampleStart[cz] , 2*lowCenter + cz ) ;
	}
	int lowDim = _Dimension< FEMDegree >( lowDepth ) , highDim = _Dimension< FEMDegree >( highDepth );

	// Iterate over all parent nodes
#pragma omp parallel for num_threads( threads )
	for( int k=0 ; k<lowDim ; k++ ) for( int j=0 ; j<lowDim ; j++ ) for( int i=0 ; i<lowDim ; i++ )
	{
		DownSampleKey& neighborKey = neighborKeys[ omp_get_thread_num() ];
		int off[] = { i , j , k } , lowIdx = i + j * lowDim  + k * lowDim * lowDim;

		// Want to make sure test if contained children are interior.
		// This is more conservative because we are test that overlapping children are interior
		bool isInterior = _IsInteriorlyOverlapped< FEMDegree , FEMDegree >( lowDepth , i , j , k );

		// Iterate over all the children of the parent
		for( int c=0 ; c<Cube::CORNERS ; c++ )
		{
			int cx , cy , cz;
			Cube::FactorCornerIndex( c , cx , cy , cz );

			// For odd degrees not all children are valid
			int ii = (i<<1)|cx , jj = (j<<1)|cy , kk = (k<<1)|cz;
			if( ii<0 || ii>=highDim || jj<0 || jj>=highDim || kk<0 || kk>=highDim ) continue;

			C& highCoefficient = highCoefficients[ ii + jj*highDim + kk*highDim*highDim ];

			if( isInterior )
			{
				for( int ii=0 ; ii<BSplineEvaluationData< FEMDegree >::DownSampleSize[cx] ; ii++ ) for( int jj=0 ; jj<BSplineEvaluationData< FEMDegree >::DownSampleSize[cy] ; jj++ )
				{
					int _i = i + ii + BSplineEvaluationData< FEMDegree >::DownSampleStart[cx];
					int _j = j + jj + BSplineEvaluationData< FEMDegree >::DownSampleStart[cy];
					for( int kk=0 ; kk<BSplineEvaluationData< FEMDegree >::DownSampleSize[cz] ; kk++ )
					{
						int _k = k + kk + BSplineEvaluationData< FEMDegree >::DownSampleStart[cz];
						highCoefficient += (C)( lowCoefficients[ _i + _j*lowDim  + _k*lowDim*lowDim ] * downSampleStencils[c].values[ii][jj][kk] );
					}
				}
			}
			else
			{
				double downSampleValues[3][ BSplineEvaluationData< FEMDegree >::DownSample0Size > BSplineEvaluationData< FEMDegree >::DownSample1Size ? BSplineEvaluationData< FEMDegree >::DownSample0Size : BSplineEvaluationData< FEMDegree >::DownSample1Size ];

				for( int ii=0 ; ii<BSplineEvaluationData< FEMDegree >::DownSampleSize[cx] ; ii++ ) downSampleValues[0][ii] = upSampleEvaluator.value( off[0] + ii + BSplineEvaluationData< FEMDegree >::DownSampleStart[cx] , 2*off[0] + cx );
				for( int ii=0 ; ii<BSplineEvaluationData< FEMDegree >::DownSampleSize[cy] ; ii++ ) downSampleValues[1][ii] = upSampleEvaluator.value( off[1] + ii + BSplineEvaluationData< FEMDegree >::DownSampleStart[cy] , 2*off[1] + cy );
				for( int ii=0 ; ii<BSplineEvaluationData< FEMDegree >::DownSampleSize[cz] ; ii++ ) downSampleValues[2][ii] = upSampleEvaluator.value( off[2] + ii + BSplineEvaluationData< FEMDegree >::DownSampleStart[cz] , 2*off[2] + cz );

				for( int ii=0 ; ii<BSplineEvaluationData< FEMDegree >::DownSampleSize[cx] ; ii++ ) for( int jj=0 ; jj<BSplineEvaluationData< FEMDegree >::DownSampleSize[cy] ; jj++ )
				{
					double dxy = downSampleValues[0][ii] * downSampleValues[1][jj];
					int _i = i + ii + BSplineEvaluationData< FEMDegree >::DownSampleStart[cx];
					int _j = j + jj + BSplineEvaluationData< FEMDegree >::DownSampleStart[cy];
					if( _i>=0 && _i<lowDim && _j>=0 && _j<lowDim )
						for( int kk=0 ; kk<BSplineEvaluationData< FEMDegree >::DownSampleSize[cz] ; kk++ )
						{
							int _k = k + kk + BSplineEvaluationData< FEMDegree >::DownSampleStart[cz];
							if( _k>=0 && _k<lowDim ) highCoefficient += (C)( lowCoefficients[ _i + _j*lowDim  + _k*lowDim*lowDim ] * dxy * downSampleValues[2][kk] );
					}
				}
			}
		}
	}
}
template< class Real >
template< int FEMDegree >
Real Octree< Real >::_CoarserFunctionValue( Point3D< Real > p , const PointSupportKey< FEMDegree >& neighborKey , const TreeOctNode* pointNode , const BSplineData< FEMDegree >& bsData , const DenseNodeData< Real , FEMDegree >& upSampledCoefficients ) const
{
	static const int SupportSize = BSplineEvaluationData< FEMDegree >::SupportSize;
	static const int  LeftSupportRadius = - BSplineEvaluationData< FEMDegree >::SupportStart;
	static const int RightSupportRadius =   BSplineEvaluationData< FEMDegree >::SupportEnd;
	static const int  LeftPointSupportRadius =   BSplineEvaluationData< FEMDegree >::SupportEnd;
	static const int RightPointSupportRadius = - BSplineEvaluationData< FEMDegree >::SupportStart;

	double pointValue = 0;
	int depth = pointNode->depth();
	if( depth<=_minDepth ) return Real(0.);

	// Iterate over all basis functions that overlap the point at the coarser resolution
	{
		const typename TreeOctNode::Neighbors< SupportSize >& neighbors = neighborKey.neighbors[depth-1];
		int _d , _off[3];
		pointNode->parent->depthAndOffset( _d , _off );
		int fStart , fEnd;
		BSplineData< FEMDegree >::FunctionSpan( _d-1 , fStart , fEnd );

		double pointValues[ DIMENSION ][SupportSize];
		memset( pointValues , 0 , sizeof(double) * DIMENSION * SupportSize );

		for( int dd=0 ; dd<DIMENSION ; dd++ ) for( int i=-LeftPointSupportRadius ; i<=RightPointSupportRadius ; i++ )
		{
			int fIdx = BSplineData< FEMDegree >::FunctionIndex( _d-1 , _off[dd]+i );
			if( fIdx>=fStart && fIdx<fEnd ) pointValues[dd][i+LeftPointSupportRadius] = bsData.baseBSplines[ fIdx ][LeftSupportRadius-i]( p[dd] );
		}

		for( int j=0 ; j<SupportSize ; j++ ) for( int k=0 ; k<SupportSize ; k++ )
		{
			double xyValue = pointValues[0][j] * pointValues[1][k];
			double _pointValue = 0;
			for( int l=0 ; l<SupportSize ; l++ )
			{
				const TreeOctNode* _node = neighbors.neighbors[j][k][l];
				if( _IsValidNode< FEMDegree >( _node ) ) _pointValue += pointValues[2][l] * double( upSampledCoefficients[_node->nodeData.nodeIndex] );
			}
			pointValue += _pointValue * xyValue;
		}
	}
	return Real( pointValue );
}

template< class Real >
template< int FEMDegree >
Real Octree< Real >::_FinerFunctionValue( Point3D< Real > p , const PointSupportKey< FEMDegree >& neighborKey , const TreeOctNode* pointNode , const BSplineData< FEMDegree >& bsData , const DenseNodeData< Real , FEMDegree >& finerCoefficients ) const
{
	typename TreeOctNode::Neighbors< BSplineEvaluationData< FEMDegree >::SupportSize > childNeighbors;
	static const int  LeftPointSupportRadius =  BSplineEvaluationData< FEMDegree >::SupportEnd;
	static const int RightPointSupportRadius = -BSplineEvaluationData< FEMDegree >::SupportStart;
	static const int  LeftSupportRadius = -BSplineEvaluationData< FEMDegree >::SupportStart;
	static const int RightSupportRadius =  BSplineEvaluationData< FEMDegree >::SupportEnd;

	double pointValue = 0;
	int depth = pointNode->depth();
	neighborKey.template getChildNeighbors< false >( p , depth , childNeighbors );
	for( int j=-LeftPointSupportRadius ; j<=RightPointSupportRadius ; j++ )
		for( int k=-LeftPointSupportRadius ; k<=RightPointSupportRadius ; k++ )
			for( int l=-LeftPointSupportRadius ; l<=RightPointSupportRadius ; l++ )
			{
				const TreeOctNode* _node = childNeighbors.neighbors[j+LeftPointSupportRadius][k+LeftPointSupportRadius][l+LeftPointSupportRadius];
				if( _IsValidNode< FEMDegree >( _node ) )
				{
					int fIdx[3];
					FunctionIndex< FEMDegree >( _node , fIdx );
					pointValue += 
						bsData.baseBSplines[ fIdx[0] ][LeftSupportRadius-j]( p[0] ) *
						bsData.baseBSplines[ fIdx[1] ][LeftSupportRadius-k]( p[1] ) *
						bsData.baseBSplines[ fIdx[2] ][LeftSupportRadius-l]( p[2] ) *
						double( finerCoefficients[ _node->nodeData.nodeIndex ] );
				}
			}
	return Real( pointValue );
}

template< class Real >
template< int FEMDegree >
void Octree< Real >::_SetPointValuesFromCoarser( SparseNodeData< PointData< Real > , 0 >& pointInfo , int highDepth , const BSplineData< FEMDegree >& bsData , const DenseNodeData< Real , FEMDegree >& upSampledCoefficients )
{
	int lowDepth = highDepth-1;
	if( lowDepth<_minDepth ) return;
	std::vector< PointData< Real > >& points = pointInfo.data;
	std::vector< PointSupportKey< FEMDegree > > neighborKeys( std::max< int >( 1 , threads ) );
	for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( lowDepth );

#pragma omp parallel for num_threads( threads )
	for( int i=_sNodes.begin(highDepth) ; i<_sNodes.end(highDepth) ; i++ ) if( _IsValidNode< FEMDegree >( _sNodes.treeNodes[i] ) )
	{
		PointSupportKey< FEMDegree >& neighborKey = neighborKeys[ omp_get_thread_num() ];
		int pIdx = pointInfo.index( _sNodes.treeNodes[i] );
		if( pIdx!=-1 )
		{
			neighborKey.template getNeighbors< false >( _sNodes.treeNodes[i]->parent );
			points[ pIdx ].weightedCoarserDValue = (Real)( _CoarserFunctionValue( points[pIdx].position , neighborKey , _sNodes.treeNodes[i] , bsData , upSampledCoefficients ) - 0.5 ) * points[pIdx].weight;
		}
	}
}

template< class Real >
template< int FEMDegree >
void Octree< Real >::_SetPointConstraintsFromFiner( const SparseNodeData< PointData< Real > , 0 >& pointInfo , int highDepth , const BSplineData< FEMDegree >& bsData , const DenseNodeData< Real , FEMDegree >& finerCoefficients , DenseNodeData< Real , FEMDegree >& coarserConstraints ) const
{
	static const int SupportSize = BSplineEvaluationData< FEMDegree >::SupportSize;
	static const int  LeftPointSupportRadius =  BSplineEvaluationData< FEMDegree >::SupportEnd;
	static const int RightPointSupportRadius = -BSplineEvaluationData< FEMDegree >::SupportStart;
	static const int  LeftSupportRadius = -BSplineEvaluationData< FEMDegree >::SupportStart;
	static const int RightSupportRadius =  BSplineEvaluationData< FEMDegree >::SupportEnd;

	const std::vector< PointData< Real > >& points = pointInfo.data;
	// Note: We can't iterate over the finer point nodes as the point weights might be
	// scaled incorrectly, due to the adaptive exponent. So instead, we will iterate
	// over the coarser nodes and evaluate the finer solution at the associated points.
	int  lowDepth = highDepth-1;
	if( lowDepth<_minDepth ) return;
	size_t start = _sNodes.begin(lowDepth) , end = _sNodes.end(lowDepth) , range = end-start;
	memset( coarserConstraints.data+start , 0 , sizeof( Real ) * range );
	std::vector< PointSupportKey< FEMDegree > > neighborKeys( std::max< int >( 1 , threads ) );
	for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( lowDepth );
#pragma omp parallel for num_threads( threads )
	for( int i=_sNodes.begin(lowDepth) ; i<_sNodes.end(lowDepth) ; i++ ) if( _IsValidNode< FEMDegree >( _sNodes.treeNodes[i] ) )
	{
		PointSupportKey< FEMDegree >& neighborKey = neighborKeys[ omp_get_thread_num() ];
		int pIdx = pointInfo.index( _sNodes.treeNodes[i] );
		if( pIdx!=-1 )
		{
			typename TreeOctNode::Neighbors< SupportSize >& neighbors = neighborKey.template getNeighbors< false >( _sNodes.treeNodes[i] );
			// Evaluate the solution @( depth ) at the current point @( depth-1 )
			{
				Real finerPointDValue = (Real)( _FinerFunctionValue( points[pIdx].position , neighborKey , _sNodes.treeNodes[i] , bsData , finerCoefficients ) - 0.5 ) * points[pIdx].weight;
				Point3D< Real > p = points[ pIdx ].position;
				// Update constraints for all nodes @( depth-1 ) that overlap the point
				int d , idx[3];
				neighbors.neighbors[LeftPointSupportRadius][LeftPointSupportRadius][LeftPointSupportRadius]->depthAndOffset( d, idx );
				// Set the (offset) index to the top-left-front corner of the 3x3x3 block of b-splines
				// overlapping the point.
				idx[0] = BinaryNode::CenterIndex( d , idx[0] );
				idx[1] = BinaryNode::CenterIndex( d , idx[1] );
				idx[2] = BinaryNode::CenterIndex( d , idx[2] );
				for( int x=-LeftPointSupportRadius ; x<=RightPointSupportRadius ; x++ )
					for( int y=-LeftPointSupportRadius ; y<=RightPointSupportRadius ; y++ )
						for( int z=-LeftPointSupportRadius ; z<=RightPointSupportRadius ; z++ )
							if( _IsValidNode< FEMDegree >( neighbors.neighbors[x+LeftPointSupportRadius][y+LeftPointSupportRadius][z+LeftPointSupportRadius] ) )
							{
#pragma omp atomic
								coarserConstraints[ neighbors.neighbors[x+LeftPointSupportRadius][y+LeftPointSupportRadius][z+LeftPointSupportRadius]->nodeData.nodeIndex - _sNodes.begin(lowDepth) ] +=
									Real(
									bsData.baseBSplines[idx[0]+x][LeftSupportRadius-x]( p[0] ) *
									bsData.baseBSplines[idx[1]+y][LeftSupportRadius-y]( p[1] ) *
									bsData.baseBSplines[idx[2]+z][LeftSupportRadius-z]( p[2] ) * 
									finerPointDValue
									);
							}
			}
		}
	}
}

template< class Real >
template< int FEMDegree >
int Octree< Real >::_SetMatrixRow( const SparseNodeData< PointData< Real > , 0 >& pointInfo , const typename TreeOctNode::Neighbors< BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapSize >& neighbors , Pointer( MatrixEntry< Real > ) row , int offset , const typename BSplineIntegrationData< FEMDegree , FEMDegree >::FunctionIntegrator::Integrator& integrator , const Stencil< double , BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapSize >& stencil , const BSplineData< FEMDegree >& bsData ) const
{
	static const int SupportSize = BSplineEvaluationData< FEMDegree >::SupportSize;
	static const int OverlapRadius = - BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapStart;
	static const int OverlapSize   =   BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapSize;
	static const int LeftSupportRadius  = -BSplineEvaluationData< FEMDegree >::SupportStart;
	static const int RightSupportRadius =  BSplineEvaluationData< FEMDegree >::SupportEnd;
	static const int LeftPointSupportRadius  = BSplineEvaluationData< FEMDegree >::SupportEnd;
	static const int RightPointSupportRadius = -BSplineEvaluationData< FEMDegree >::SupportStart;

	const std::vector< PointData< Real > >& points = pointInfo.data;
	bool hasYZPoints[SupportSize] , hasZPoints[SupportSize][SupportSize];
	Real diagonal = 0;
	// Given a node:
	// -- for each node in its support:
	// ---- if the supporting node contains a point:
	// ------ evaluate the x, y, and z B-splines of the nodes supporting the point
	// splineValues \in [-LeftSupportRadius,RightSupportRadius] x [-LeftSupportRadius,RightSupportRadius] x [-LeftSupportRadius,RightSupportRadius] x [0,Dimension) x [-LeftPointSupportRadius,RightPointSupportRadius]
	Real splineValues[SupportSize][SupportSize][SupportSize][DIMENSION][SupportSize];
	memset( splineValues , 0 , sizeof( Real ) * SupportSize * SupportSize * SupportSize * DIMENSION *SupportSize );

	int count = 0;
	const TreeOctNode* node = neighbors.neighbors[OverlapRadius][OverlapRadius][OverlapRadius];
	int d , off[3];
	node->depthAndOffset( d , off );
	int fStart , fEnd;
	BSplineData< FEMDegree >::FunctionSpan( d-1 , fStart , fEnd );
	bool isInterior = _IsInteriorlyOverlapped< FEMDegree , FEMDegree >( node );

	if( _constrainValues )
	{
		// Iterate over all neighboring nodes that may have a constraining point
		// -- For each one, compute the values of the spline functions supported on the point
		for( int j=0 ; j<SupportSize ; j++ )
		{
			hasYZPoints[j] = false;
			for( int k=0 ; k<SupportSize ; k++ ) hasZPoints[j][k] = false;
		}
		for( int j=-LeftSupportRadius , jj=0 ; j<=RightSupportRadius ; j++ , jj++ )
			for( int k=-LeftSupportRadius , kk=0 ; k<=RightSupportRadius ; k++ , kk++ )
				for( int l=-LeftSupportRadius , ll=0 ; l<=RightSupportRadius ; l++ , ll++ )
				{
					const TreeOctNode* _node = neighbors.neighbors[OverlapRadius+j][OverlapRadius+k][OverlapRadius+l];
					if( _IsValidNode< 0 >( _node ) && pointInfo.index( _node )!=-1 )
					{
						int pOff[] = { off[0]+j , off[1]+k , off[2]+l };
						hasYZPoints[jj] = hasZPoints[jj][kk] = true;
						const PointData< Real >& pData = points[ pointInfo.index( _node ) ];
						Real (*_splineValues)[SupportSize] = splineValues[jj][kk][ll];
						Real weight = pData.weight;
						Point3D< Real > p = pData.position;
						// Evaluate the point p at all the nodes whose functions have it in their support
						for( int s=-LeftPointSupportRadius ; s<=RightPointSupportRadius ; s++ ) for( int dd=0 ; dd<DIMENSION ; dd++ )
						{
							int fIdx = BSplineData< FEMDegree >::FunctionIndex( d-1 , pOff[dd]+s );
							if( fIdx>=fStart && fIdx<fEnd ) _splineValues[dd][ s+LeftPointSupportRadius ] = Real( bsData.baseBSplines[ fIdx ][ -s+LeftSupportRadius ]( p[dd] ) );
						}
						// The value of the function of the node that we started with
						Real value = _splineValues[0][-j+LeftPointSupportRadius] * _splineValues[1][-k+LeftPointSupportRadius] * _splineValues[2][-l+LeftPointSupportRadius];
						Real weightedValue = value * weight;
						diagonal += value * weightedValue;

						// Pre-multiply the x-coordinate values so that when we evaluate at one of the neighboring basis functions
						// we get the product of the values of the center base function and the base function of the neighboring node
						for( int s=0 ; s<SupportSize ; s++ ) _splineValues[0][s] *= weightedValue;
					}
				}
	}

	Real pointValues[OverlapSize][OverlapSize][OverlapSize];
	if( _constrainValues )
	{
		memset( pointValues , 0 , sizeof(Real) * OverlapSize * OverlapSize * OverlapSize );
		// Iterate over all supported neighbors that could have a point constraint	
		for( int i=-LeftSupportRadius ; i<=RightSupportRadius ; i++ ) if( hasYZPoints[i+LeftSupportRadius] )
			for( int j=-LeftSupportRadius ; j<=RightSupportRadius ; j++ ) if( hasZPoints[i+LeftSupportRadius][j+LeftSupportRadius] )
				for( int k=-LeftSupportRadius ; k<=RightSupportRadius ; k++ )
				{
					const TreeOctNode* _node = neighbors.neighbors[i+OverlapRadius][j+OverlapRadius][k+OverlapRadius];
					Real (*_splineValues)[SupportSize] = splineValues[i+LeftSupportRadius][j+LeftSupportRadius][k+LeftSupportRadius];
					if( _IsValidNode< 0 >( _node ) && pointInfo.index( _node )!=-1 )
						// Iterate over all neighbors whose support contains the point and accumulate the mutual integral
						for( int ii=-LeftPointSupportRadius ; ii<=RightPointSupportRadius ; ii++ )
							for( int jj=-LeftPointSupportRadius ; jj<=RightPointSupportRadius ; jj++ )
								for( int kk=-LeftPointSupportRadius ; kk<=RightPointSupportRadius ; kk++ )
								{
									TreeOctNode* _node = neighbors.neighbors[i+ii+OverlapRadius][j+jj+OverlapRadius][k+kk+OverlapRadius];
									if( _IsValidNode< FEMDegree >( _node ) )
										pointValues[i+ii+OverlapRadius][j+jj+OverlapRadius][k+kk+OverlapRadius] +=
											_splineValues[0][ii+LeftPointSupportRadius ] * _splineValues[1][jj+LeftPointSupportRadius ] * _splineValues[2][kk+LeftPointSupportRadius ];
								}
				}
	}
	pointValues[OverlapRadius][OverlapRadius][OverlapRadius] = diagonal;
	int nodeIndex = neighbors.neighbors[OverlapRadius][OverlapRadius][OverlapRadius]->nodeData.nodeIndex;
	if( isInterior ) // General case, so try to make fast
	{
		const TreeOctNode* const * _nodes = &neighbors.neighbors[0][0][0];
		const double* _stencil = &stencil.values[0][0][0];
		Real* _values = &pointValues[0][0][0];
		const static int CenterIndex = OverlapSize*OverlapSize*OverlapRadius + OverlapSize*OverlapRadius + OverlapRadius;
		if( _constrainValues ) for( int i=0 ; i<OverlapSize*OverlapSize*OverlapSize ; i++ ) _values[i] = Real( _stencil[i] + _values[i] );
		else                   for( int i=0 ; i<OverlapSize*OverlapSize*OverlapSize ; i++ ) _values[i] = Real( _stencil[i] );

		row[count++] = MatrixEntry< Real >( nodeIndex-offset , _values[CenterIndex] );
		for( int i=0 ; i<OverlapSize*OverlapSize*OverlapSize ; i++ ) if( i!=CenterIndex && _nodes[i] )
			row[count++] = MatrixEntry< Real >( _nodes[i]->nodeData.nodeIndex-offset , _values[i] );
	}
	else
	{
		int d , off[3];
		node->depthAndOffset( d , off );
		Real temp = Real( SystemCoefficients< FEMDegree , FEMDegree >::GetLaplacian( integrator , off , off ) );
		if( _constrainValues ) temp += pointValues[OverlapRadius][OverlapRadius][OverlapRadius];
		row[count++] = MatrixEntry< Real >( nodeIndex-offset , temp );
		for( int x=0 ; x<OverlapSize ; x++ ) for( int y=0 ; y<OverlapSize ; y++ ) for( int z=0 ; z<OverlapSize ; z++ )
			if( (x!=OverlapRadius || y!=OverlapRadius || z!=OverlapRadius) && _IsValidNode< FEMDegree >( neighbors.neighbors[x][y][z] ) )
			{
				const TreeOctNode* _node = neighbors.neighbors[x][y][z];
				int _d , _off[3];
				_node->depthAndOffset( _d , _off );
				Real temp = Real( SystemCoefficients< FEMDegree , FEMDegree >::GetLaplacian( integrator , _off , off ) );
				if( _constrainValues ) temp += pointValues[x][y][z];
				row[count++] = MatrixEntry< Real >( _node->nodeData.nodeIndex-offset , temp );
			}
	}
	return count;
}

template< class Real >
template< int FEMDegree >
int Octree< Real >::_GetMatrixAndUpdateConstraints( const SparseNodeData< PointData< Real > , 0 >& pointInfo , SparseMatrix< Real >& matrix , DenseNodeData< Real , FEMDegree >& constraints , typename BSplineIntegrationData< FEMDegree , FEMDegree >::FunctionIntegrator::Integrator& integrator , typename BSplineIntegrationData< FEMDegree , FEMDegree >::FunctionIntegrator::ChildIntegrator& childIntegrator , const BSplineData< FEMDegree >& bsData , int depth , const DenseNodeData< Real , FEMDegree >* metSolution , bool coarseToFine )
{
	static const int OverlapRadius = - BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapStart;
	static const int OverlapSize   =   BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapSize;

	size_t start = _sNodes.begin(depth) , end = _sNodes.end(depth) , range = end-start;
	Stencil< double , OverlapSize > stencil , stencils[2][2][2];
	SystemCoefficients< FEMDegree , FEMDegree >::SetCentralLaplacianStencil (      integrator , stencil  );
	SystemCoefficients< FEMDegree , FEMDegree >::SetCentralLaplacianStencils( childIntegrator , stencils );
	matrix.Resize( (int)range );
	std::vector< AdjacenctNodeKey > neighborKeys( std::max< int >( 1 , threads ) );
	for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( depth );
#pragma omp parallel for num_threads( threads )
	for( int i=0 ; i<(int)range ; i++ ) if( _IsValidNode< FEMDegree >( _sNodes.treeNodes[i+start] ) )
	{
		AdjacenctNodeKey& neighborKey = neighborKeys[ omp_get_thread_num() ];
		TreeOctNode* node = _sNodes.treeNodes[i+start];
		// Get the matrix row size
		typename TreeOctNode::Neighbors< OverlapSize > neighbors;
		neighborKey.template getNeighbors< false , OverlapRadius , OverlapRadius >( node , neighbors );
		int count = _GetMatrixRowSize< FEMDegree >( neighbors );

		// Allocate memory for the row
#pragma omp critical (matrix_set_row_size)
		matrix.SetRowSize( i , count );

		// Set the row entries
		matrix.rowSizes[i] = _SetMatrixRow( pointInfo , neighbors , matrix[i] , (int)start , integrator , stencil , bsData );
		if( depth>_minDepth )
		{
			// Offset the constraints using the solution from lower resolutions.
			int x , y , z , c;
			if( node->parent )
			{
				c = int( node - node->parent->children );
				Cube::FactorCornerIndex( c , x , y , z );
			}
			else x = y = z = 0;
			if( coarseToFine )
			{
				typename TreeOctNode::Neighbors< OverlapSize > pNeighbors;
				neighborKey.template getNeighbors< false , OverlapRadius , OverlapRadius >( node->parent , pNeighbors );
				_UpdateConstraintsFromCoarser( pointInfo , neighbors , pNeighbors , node , constraints , *metSolution , childIntegrator , stencils[x][y][z] , bsData );
			}
		}
	}
	return 1;
}

template< class Real >
template< int FEMDegree >
int Octree< Real >::_GetSliceMatrixAndUpdateConstraints( const SparseNodeData< PointData< Real > , 0 >& pointInfo , SparseMatrix< Real >& matrix , DenseNodeData< Real , FEMDegree >& constraints , typename BSplineIntegrationData< FEMDegree , FEMDegree >::FunctionIntegrator::Integrator& integrator , typename BSplineIntegrationData< FEMDegree , FEMDegree >::FunctionIntegrator::ChildIntegrator& childIntegrator , const BSplineData< FEMDegree >& bsData , int depth , int slice , const DenseNodeData< Real , FEMDegree >& metSolution , bool coarseToFine )
{
	static const int OverlapSize   =  BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapSize;
	static const int OverlapRadius = -BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapStart;

	int nStart = _sNodes.begin( depth , slice ) , nEnd = _sNodes.end( depth , slice );
	size_t range = nEnd-nStart;
	Stencil< double , OverlapSize > stencil , stencils[2][2][2];
	SystemCoefficients< FEMDegree , FEMDegree >::SetCentralLaplacianStencil (      integrator , stencil  );
	SystemCoefficients< FEMDegree , FEMDegree >::SetCentralLaplacianStencils( childIntegrator , stencils );

	matrix.Resize( (int)range );
	std::vector< AdjacenctNodeKey > neighborKeys( std::max< int >( 1 , threads ) );
	for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( depth );
#pragma omp parallel for num_threads( threads )
	for( int i=0 ; i<(int)range ; i++ ) if( _IsValidNode< FEMDegree >( _sNodes.treeNodes[i+nStart] ) )
	{
		AdjacenctNodeKey& neighborKey = neighborKeys[ omp_get_thread_num() ];
		TreeOctNode* node = _sNodes.treeNodes[i+nStart];
		// Get the matrix row size
		typename TreeOctNode::Neighbors< OverlapSize > neighbors;
		neighborKey.template getNeighbors< false , OverlapRadius , OverlapRadius >( node , neighbors );
		int count = _GetMatrixRowSize< FEMDegree >( neighbors );

		// Allocate memory for the row
#pragma omp critical (matrix_set_row_size)
		{
			matrix.SetRowSize( i , count );
		}

		// Set the row entries
		matrix.rowSizes[i] = _SetMatrixRow( pointInfo , neighbors , matrix[i] , _sNodes.begin(depth,slice) , integrator , stencil , bsData );


		if( depth>_minDepth )
		{
			// Offset the constraints using the solution from lower resolutions.
			int x , y , z , c;
			if( node->parent )
			{
				c = int( node - node->parent->children );
				Cube::FactorCornerIndex( c , x , y , z );
			}
			else x = y = z = 0;
			if( coarseToFine )
			{
				typename TreeOctNode::Neighbors< OverlapSize > pNeighbors;
				neighborKey.template getNeighbors< false, OverlapRadius , OverlapRadius >( node->parent , pNeighbors );
				_UpdateConstraintsFromCoarser( pointInfo , neighbors , pNeighbors , node , constraints , metSolution , childIntegrator , stencils[x][y][z] , bsData );
			}
		}
	}
	return 1;
}

template< class Real >
template< int FEMDegree >
int Octree< Real >::_SolveSystemGS( const BSplineData< FEMDegree >& bsData , SparseNodeData< PointData< Real > , 0 >& pointInfo , int depth , DenseNodeData< Real , FEMDegree >& solution , DenseNodeData< Real , FEMDegree >& constraints , DenseNodeData< Real , FEMDegree >& metSolutionConstraints , int iters , bool coarseToFine , bool showResidual , double* bNorm2 , double* inRNorm2 , double* outRNorm2 , bool forceSilent )
{
	const int OverlapRadius = -BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapStart;
	typename BSplineIntegrationData< FEMDegree , FEMDegree >::FunctionIntegrator::Integrator integrator;
	typename BSplineIntegrationData< FEMDegree , FEMDegree >::FunctionIntegrator::ChildIntegrator childIntegrator;
	BSplineIntegrationData< FEMDegree , FEMDegree >::SetIntegrator( integrator , depth-1 , _dirichlet , _dirichlet );
	if( depth>_minDepth ) BSplineIntegrationData< FEMDegree , FEMDegree >::SetChildIntegrator( childIntegrator , depth-2 , _dirichlet , _dirichlet );

	DenseNodeData< Real , FEMDegree > metSolution , metConstraints;
	if( coarseToFine ) metSolution    = metSolutionConstraints;	// This stores the up-sampled solution up to depth-2
	else               metConstraints = metSolutionConstraints; // This stores the down-sampled constraints up to depth

	double _maxMemoryUsage = maxMemoryUsage;
	maxMemoryUsage = 0;
	int slices = _Dimension< FEMDegree >(depth);
	double systemTime=0. , solveTime=0. , updateTime=0. ,  evaluateTime = 0.;

	if( coarseToFine )
	{
		if( depth>_minDepth )
		{
			// Up-sample the cumulative change in solution @(depth-2) into the cumulative change in solution @(depth-1)
			if( depth-2>=_minDepth ) _UpSample( depth-1 , metSolution );
			// Add in the change in solution @(depth-1)
#pragma omp parallel for num_threads( threads )
			for( int i=_sNodes.begin(depth-1) ; i<_sNodes.end(depth-1) ; i++ ) metSolution[i] += solution[i];
			// Evaluate the points @(depth) using the cumulative change in solution @(depth-1)
			if( _constrainValues )
			{
				evaluateTime = Time();
				_SetPointValuesFromCoarser( pointInfo , depth , bsData , metSolution );
				evaluateTime = Time() - evaluateTime;
			}
		}
	}
	else if( depth<_sNodes.levels()-1 )
		for( int i=_sNodes.begin(depth) ; i<_sNodes.end(depth) ; i++ ) constraints[i] -= metConstraints[i];
	double bNorm=0 , inRNorm=0 , outRNorm=0;
	if( depth>=_minDepth )
	{
		// Add padding space if we are computing residuals
		int frontOffset = ( showResidual || inRNorm2 ) ? OverlapRadius : 0;
		int backOffset = ( showResidual || outRNorm2 ) ? OverlapRadius : 0;
		// Set the number of in-memory slices required for a temporally blocked solver
		int solveSlices = std::min< int >( OverlapRadius*iters - (OverlapRadius-1) , slices ) , matrixSlices = std::max< int >( 1 , std::min< int >( solveSlices+frontOffset+backOffset , slices ) );
		// The list of matrices for each in-memory slices
		std::vector< SparseMatrix< Real > > _M( matrixSlices );
		// The list of multi-colored indices  for each in-memory slice
		std::vector< std::vector< std::vector< int > > > __mcIndices( std::max< int >( 0 , solveSlices ) );

		int dir = coarseToFine ? -1 : 1 , start = coarseToFine ? slices-1 : 0 , end = coarseToFine ? -1 : slices;
		for( int frontSlice=start-frontOffset*dir , backSlice = frontSlice-OverlapRadius*(iters-1)*dir ; backSlice!=end+backOffset*dir ; frontSlice+=dir , backSlice+=dir )
		{
			double t;
			if( frontSlice+frontOffset*dir>=0 && frontSlice+frontOffset*dir<slices )
			{
				int s = frontSlice+frontOffset*dir , _s = s % matrixSlices;
				t = Time();
				// Compute the system matrix
				ConstPointer( Real ) B = constraints.data + _sNodes.begin( depth , s );
				Pointer( Real ) X = solution.data + _sNodes.begin( depth , s );
				_GetSliceMatrixAndUpdateConstraints( pointInfo , _M[_s] , constraints , integrator , childIntegrator , bsData , depth , s , metSolution , coarseToFine );
				systemTime += Time()-t;
				Pointer( TreeOctNode* ) const nodes = _sNodes.treeNodes + _sNodes.begin(depth);
				// Compute residuals
				if( showResidual || inRNorm2 )
#pragma omp parallel for num_threads( threads ) reduction( + : bNorm , inRNorm )
					for( int j=0 ; j<_M[_s].rows ; j++ )
					{
						Real temp = Real(0);
						ConstPointer( MatrixEntry< Real > ) start = _M[_s][j];
						ConstPointer( MatrixEntry< Real > ) end = start + _M[_s].rowSizes[j];
						ConstPointer( MatrixEntry< Real > ) e;
						for( e=start ; e!=end ; e++ ) temp += X[ e->N ] * e->Value;
						bNorm += B[j]*B[j];
						inRNorm += (temp-B[j]) * (temp-B[j]);
					}
				else if( bNorm2 )
#pragma omp parallel for num_threads( threads ) reduction( + : bNorm )
					for( int j=0 ; j<_M[_s].rows ; j++ ) bNorm += B[j]*B[j];
			}
			t = Time();
			// Compute the multicolor indices
			if( iters && frontSlice>=0 && frontSlice<slices )
			{
				int s = frontSlice , _s = s % matrixSlices , __s = s % solveSlices;
				for( int i=0 ; i<int( __mcIndices[__s].size() ) ; i++ ) __mcIndices[__s][i].clear();
				_setMultiColorIndices< FEMDegree >( _sNodes.begin(depth,s) , _sNodes.end(depth,s) , __mcIndices[__s] );
			}
			// Advance through the in-memory slices, taking an appropriately sized stride
			for( int slice=frontSlice ; slice*dir>=backSlice*dir ; slice-=OverlapRadius*dir )
				if( slice>=0 && slice<slices )
				{
					int s = slice , _s = s % matrixSlices , __s = s % solveSlices;
					// Do the GS solver
					ConstPointer( Real ) B = constraints.data + _sNodes.begin( depth , s );
					Pointer( Real ) X = solution.data + _sNodes.begin( depth , s );
					SparseMatrix< Real >::SolveGS( __mcIndices[__s] , _M[_s] , B , X , !coarseToFine , threads );
				}
			solveTime += Time() - t;
			// Compute residuals
			if( (showResidual || outRNorm2) && backSlice-backOffset*dir>=0 && backSlice-backOffset*dir<slices )
			{
				int s = backSlice-backOffset*dir , _s = s % matrixSlices;
				ConstPointer( Real ) B = constraints.data + _sNodes.begin( depth , s );
				Pointer( Real ) X = solution.data + _sNodes.begin( depth , s );
#pragma omp parallel for num_threads( threads ) reduction( + : outRNorm )
				for( int j=0 ; j<_M[_s].rows ; j++ )
				{
					Real temp = Real(0);
					ConstPointer( MatrixEntry< Real > ) start = _M[_s][j];
					ConstPointer( MatrixEntry< Real > ) end = start + _M[_s].rowSizes[j];
					ConstPointer( MatrixEntry< Real > ) e;
					for( e=start ; e!=end ; e++ ) temp += X[ e->N ] * e->Value;
					outRNorm += (temp-B[j]) * (temp-B[j]);
				}
			}
		}
	}

	if( bNorm2 ) bNorm2[depth] = bNorm;
	if( inRNorm2 ) inRNorm2[depth] = inRNorm;
	if( outRNorm2 ) outRNorm2[depth] = outRNorm;
	if( showResidual && iters )
	{
		for( int i=0 ; i<depth ; i++ ) printf( "  " );
		printf( "GS: %.4e -> %.4e -> %.4e (%.2e) [%d]\n" , sqrt( bNorm ) , sqrt( inRNorm ) , sqrt( outRNorm ) , sqrt( outRNorm/bNorm ) , iters );
	}

	if( !coarseToFine && depth>_minDepth )
	{
		// Explicitly compute the restriction of the met solution onto the coarser nodes
		// and down-sample the previous accumulation
		{
			_UpdateConstraintsFromFiner( childIntegrator , bsData , depth , solution , metConstraints );
			if( _constrainValues ) _SetPointConstraintsFromFiner( pointInfo , depth , bsData , solution , metConstraints );
			if( depth<_sNodes.levels()-1 ) _DownSample( depth , metConstraints );
		}
	}
	MemoryUsage();
	if( !forceSilent ) DumpOutput( "\tEvaluated / Got / Solved in: %6.3f / %6.3f / %6.3f\t(%.3f MB)\n" , evaluateTime , systemTime , solveTime , float( maxMemoryUsage ) );
	maxMemoryUsage = std::max< double >( maxMemoryUsage , _maxMemoryUsage );

	return iters;
}

template< class Real >
template< int FEMDegree >
int Octree< Real >::_SolveSystemCG( const BSplineData< FEMDegree >& bsData , SparseNodeData< PointData< Real > , 0 >& pointInfo , int depth , DenseNodeData< Real , FEMDegree >& solution , DenseNodeData< Real , FEMDegree >& constraints , DenseNodeData< Real , FEMDegree >& metSolutionConstraints , int iters , bool coarseToFine , bool showResidual , double* bNorm2 , double* inRNorm2 , double* outRNorm2 , double accuracy )
{
	typename BSplineIntegrationData< FEMDegree , FEMDegree >::FunctionIntegrator::Integrator integrator;
	typename BSplineIntegrationData< FEMDegree , FEMDegree >::FunctionIntegrator::ChildIntegrator childIntegrator;
	BSplineIntegrationData< FEMDegree , FEMDegree >::SetIntegrator( integrator , depth-1 , _dirichlet , _dirichlet );
	if( depth>_minDepth ) BSplineIntegrationData< FEMDegree , FEMDegree >::SetChildIntegrator( childIntegrator , depth-2 , _dirichlet , _dirichlet );

	DenseNodeData< Real , FEMDegree > metSolution , metConstraints;
	if( coarseToFine ) metSolution    = metSolutionConstraints;	// This stores the up-sampled solution up to depth-2
	else               metConstraints = metSolutionConstraints; // This stores the down-sampled constraints up to depth
	double _maxMemoryUsage = maxMemoryUsage;
	maxMemoryUsage = 0;
	int iter = 0;
	Pointer( Real ) X = solution.data + _sNodes.begin( depth );
	Pointer( Real ) B = constraints.data + _sNodes.begin( depth );
	SparseMatrix< Real > M;
	double systemTime=0. , solveTime=0. , updateTime=0. ,  evaluateTime = 0.;

	if( coarseToFine )
	{
		if( depth>_minDepth )
		{
			// Up-sample the cumulative change in solution @(depth-2) into the cumulative change in solution @(depth-1)
			if( depth-2>=_minDepth ) _UpSample( depth-1 , metSolution );
			// Add in the change in solution @(depth-1)
#pragma omp parallel for num_threads( threads )
			for( int i=_sNodes.begin(depth-1) ; i<_sNodes.end(depth-1) ; i++ ) metSolution[i] += solution[i];
			// Evaluate the points @(depth) using the cumulative change in solution @(depth-1)
			if( _constrainValues )
			{
				evaluateTime = Time();
				_SetPointValuesFromCoarser( pointInfo , depth , bsData , metSolution );
				evaluateTime = Time() - evaluateTime;
			}
		}
	}
	else if( depth<_sNodes.levels()-1 )
		for( int i=_sNodes.begin(depth) ; i<_sNodes.end(depth) ; i++ ) constraints[i] -= metConstraints[i];

	// Get the system matrix (and adjust the right-hand-side based on the coarser solution if prolonging)
	systemTime = Time();
	_GetMatrixAndUpdateConstraints( pointInfo , M , constraints , integrator , childIntegrator , bsData , depth , coarseToFine ? &metSolution : NULL , coarseToFine );
	systemTime = Time()-systemTime;

	solveTime = Time();
	// Solve the linear system
	accuracy = Real( accuracy / 100000 ) * M.rows;
	int dim = _Dimension< FEMDegree >( depth );
	int nonZeroRows = 0;
	for( int i=0 ; i<M.rows ; i++ ) if( M.rowSizes[i] ) nonZeroRows++;
	bool addDCTerm = ( nonZeroRows==dim*dim*dim && !_constrainValues && !_dirichlet );
	double bNorm , inRNorm , outRNorm;
	if( showResidual || bNorm2 )
	{
		bNorm = 0;
#pragma omp parallel for num_threads( threads ) reduction( + : bNorm )
		for( int i=0 ; i<_sNodes.size( depth ) ; i++ ) bNorm += B[i] * B[i];
	}
	if( showResidual || inRNorm2 )
	{
		inRNorm = 0;
		Pointer( Real ) temp = AllocPointer< Real >( _sNodes.size(depth) );
		if( addDCTerm ) M.MultiplyAndAddAverage( ( ConstPointer( Real ) )X , temp , threads );
		else            M.Multiply( ( ConstPointer( Real ) )X , temp , threads );
#pragma omp parallel for num_threads( threads )
		for( int i=0 ; i<_sNodes.size(depth) ; i++ ) temp[i] -= B[i];
#pragma omp parallel for num_threads( threads ) reduction( + : inRNorm )
		for( int i=0 ; i<_sNodes.size(depth) ; i++ ) inRNorm += temp[i] * temp[i];
		FreePointer( temp );
	}

	iters = std::min< int >( nonZeroRows , iters );
	if( iters ) iter += SparseMatrix< Real >::SolveCG( M , ( ConstPointer( Real ) )B , iters , X , Real( accuracy ) , 0 , addDCTerm , false , threads );
	solveTime = Time()-solveTime;
	if( showResidual || outRNorm2 )
	{
		outRNorm = 0;
		Pointer( Real ) temp = AllocPointer< Real >( _sNodes.size(depth) );
		if( addDCTerm ) M.MultiplyAndAddAverage( ( ConstPointer( Real ) )X , temp , threads );
		else            M.Multiply( ( ConstPointer( Real ) )X , temp , threads );
#pragma omp parallel for num_threads( threads )
		for( int i=0 ; i<_sNodes.size(depth) ; i++ ) temp[i] -= B[i];
#pragma omp parallel for num_threads( threads ) reduction( + : outRNorm )
		for( int i=0 ; i<_sNodes.size(depth) ; i++ ) outRNorm += temp[i] * temp[i];
		FreePointer( temp );
	}
	if( bNorm2 ) bNorm2[depth] = bNorm * bNorm;
	if( inRNorm2 ) inRNorm2[depth] = inRNorm * inRNorm;
	if( outRNorm2 ) outRNorm2[depth] = outRNorm * outRNorm;
	if( showResidual && iters )
	{
		for( int i=0 ; i<depth ; i++ ) printf( "  " );
		printf( "CG: %.4e -> %.4e -> %.4e (%.2e) [%d]\n" , bNorm , inRNorm , outRNorm , outRNorm/bNorm , iter );
	}

	if( !coarseToFine && depth>_minDepth )
	{
		// Explicitly compute the restriction of the met solution onto the coarser nodes
		// and down-sample the previous accumulation
		{
			_UpdateConstraintsFromFiner( childIntegrator , bsData , depth , solution , metConstraints );
			if( _constrainValues ) _SetPointConstraintsFromFiner( pointInfo , depth , bsData , solution , metConstraints );
			if( depth<_sNodes.levels()-1 ) _DownSample( depth , metConstraints );
		}
	}

	MemoryUsage();
	DumpOutput( "\tEvaluated / Got / Solved in: %6.3f / %6.3f / %6.3f\t(%.3f MB)\n" , evaluateTime , systemTime , solveTime , float( maxMemoryUsage ) );
	maxMemoryUsage = std::max< double >( maxMemoryUsage , _maxMemoryUsage );
	return iter;
}

template< class Real >
template< int FEMDegree >
int Octree< Real >::_GetMatrixRowSize( const typename TreeOctNode::Neighbors< BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapSize >& neighbors ) const
{
	static const int OverlapSize   =   BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapSize;
	static const int OverlapRadius = - BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapStart;

	int count = 0;
	int nodeIndex = neighbors.neighbors[OverlapRadius][OverlapRadius][OverlapRadius]->nodeData.nodeIndex;
	const TreeOctNode* const * _nodes = &neighbors.neighbors[0][0][0];
	for( int i=0 ; i<OverlapSize*OverlapSize*OverlapSize ; i++ ) if( _IsValidNode< FEMDegree >( _nodes[i] ) ) count++;
	return count;
}


template< class Real >
template< int FEMDegree1 , int FEMDegree2 >
void Octree< Real >::_SetParentOverlapBounds( const TreeOctNode* node , int& startX , int& endX , int& startY , int& endY , int& startZ , int& endZ )
{
	const int OverlapStart = BSplineIntegrationData< FEMDegree1 , FEMDegree2 >::OverlapStart;

	if( node->parent )
	{
		int x , y , z , c = int( node - node->parent->children );
		Cube::FactorCornerIndex( c , x , y , z );
		startX = BSplineIntegrationData< FEMDegree1 , FEMDegree2 >::ParentOverlapStart[x]-OverlapStart , endX = BSplineIntegrationData< FEMDegree1 , FEMDegree2 >::ParentOverlapEnd[x]-OverlapStart+1;
		startY = BSplineIntegrationData< FEMDegree1 , FEMDegree2 >::ParentOverlapStart[y]-OverlapStart , endY = BSplineIntegrationData< FEMDegree1 , FEMDegree2 >::ParentOverlapEnd[y]-OverlapStart+1;
		startZ = BSplineIntegrationData< FEMDegree1 , FEMDegree2 >::ParentOverlapStart[z]-OverlapStart , endZ = BSplineIntegrationData< FEMDegree1 , FEMDegree2 >::ParentOverlapEnd[z]-OverlapStart+1;
	}
}

// It is assumed that at this point, the evaluationg of the current depth's points, using the coarser resolution solution
// has already happened
template< class Real >
template< int FEMDegree >
void Octree< Real >::_UpdateConstraintsFromCoarser( const SparseNodeData< PointData< Real > , 0 >& pointInfo , const typename TreeOctNode::Neighbors< BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapSize >& neighbors , const typename TreeOctNode::Neighbors< BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapSize >& pNeighbors , TreeOctNode* node , DenseNodeData< Real , FEMDegree >& constraints , const DenseNodeData< Real , FEMDegree >& metSolution , const typename BSplineIntegrationData< FEMDegree , FEMDegree >::FunctionIntegrator::ChildIntegrator& childIntegrator , const Stencil< double , BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapSize >& lapStencil , const BSplineData< FEMDegree >& bsData ) const
{
	static const int OverlapSize = BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapStart;
	static const int LeftSupportRadius  = -BSplineEvaluationData< FEMDegree >::SupportStart;
	static const int RightSupportRadius =  BSplineEvaluationData< FEMDegree >::SupportEnd;
	static const int OverlapRadius = - BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapStart;

	const std::vector< PointData< Real > >& points = pointInfo.data;
	if( node->depth()<=_minDepth ) return;
	// This is a conservative estimate as we only need to make sure that the parent nodes don't overlap the child (not the parent itself)
	bool isInterior = _IsInteriorlyOverlapped< FEMDegree , FEMDegree >( node->parent );
	int d , off[3];
	node->depthAndOffset( d , off );
	Real constraint = Real( 0 );
	// Offset the constraints using the solution from lower resolutions.
	int startX , endX , startY , endY , startZ , endZ;
	_SetParentOverlapBounds< FEMDegree , FEMDegree >( node , startX , endX , startY , endY , startZ , endZ );

	for( int x=startX ; x<endX ; x++ ) for( int y=startY ; y<endY ; y++ ) for( int z=startZ ; z<endZ ; z++ )
		if( _IsValidNode< FEMDegree >( pNeighbors.neighbors[x][y][z] ) )
		{
			const TreeOctNode* _node = pNeighbors.neighbors[x][y][z];
			Real _solution = metSolution[ _node->nodeData.nodeIndex ];
			{
				if( isInterior ) constraints[ node->nodeData.nodeIndex ] -= Real( lapStencil.values[x][y][z] * _solution );
				else
				{
					int _d , _off[3];
					_node->depthAndOffset( _d , _off );
					constraints[ node->nodeData.nodeIndex ] -= Real( SystemCoefficients< FEMDegree , FEMDegree >::GetLaplacian( childIntegrator , _off , off ) * _solution );
				}
			}
		}
	if( _constrainValues )
	{
		double constraint = 0;
		int fIdx[3];
		FunctionIndex< FEMDegree >( node , fIdx );
		// Evaluate the current node's basis function at adjacent points
		for( int x=-LeftSupportRadius ; x<=RightSupportRadius ; x++ ) for( int y=-LeftSupportRadius ; y<=RightSupportRadius ; y++ ) for( int z=-LeftSupportRadius ; z<=RightSupportRadius ; z++ )
		{
			const TreeOctNode* _node = neighbors.neighbors[x+OverlapRadius][y+OverlapRadius][z+OverlapRadius];
			if( _IsValidNode< 0 >( _node ) && pointInfo.index( _node )!=-1 )
			{
				const PointData< Real >& pData = points[ pointInfo.index( _node ) ];
				Point3D< Real > p = pData.position;
				constraint += 
					bsData.baseBSplines[ fIdx[0] ][x+LeftSupportRadius]( p[0] ) *
					bsData.baseBSplines[ fIdx[1] ][y+LeftSupportRadius]( p[1] ) *
					bsData.baseBSplines[ fIdx[2] ][z+LeftSupportRadius]( p[2] ) * 
					pData.weightedCoarserDValue;
			}
		}
		constraints[ node->nodeData.nodeIndex ] -= Real( constraint );
	}
}

// Given the solution @( depth ) add to the met constraints @( depth-1 )
template< class Real >
template< int FEMDegree >
void Octree< Real >::_UpdateConstraintsFromFiner( const typename BSplineIntegrationData< FEMDegree , FEMDegree >::FunctionIntegrator::ChildIntegrator& childIntegrator , const BSplineData< FEMDegree >& bsData , int depth , const DenseNodeData< Real , FEMDegree >& fineSolution , DenseNodeData< Real , FEMDegree >& coarseConstraints ) const
{
	static const int OverlapSize   =   BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapSize;
	static const int OverlapRadius = - BSplineIntegrationData< FEMDegree , FEMDegree >::OverlapStart;
	typedef typename TreeOctNode::NeighborKey< -BSplineEvaluationData< FEMDegree >::SupportStart , BSplineEvaluationData< FEMDegree >::SupportEnd >SupportKey;

	if( depth<=_minDepth ) return;
	// Get the stencil describing the Laplacian relating coefficients @(depth) with coefficients @(depth-1)
	Stencil< double , OverlapSize > stencils[2][2][2];
	SystemCoefficients< FEMDegree , FEMDegree >::SetCentralLaplacianStencils( childIntegrator , stencils );
	size_t start = _sNodes.begin(depth) , end = _sNodes.end(depth) , range = end-start;
	int lStart = _sNodes.begin(depth-1);
	memset( coarseConstraints.data + _sNodes.begin(depth-1) , 0 , sizeof(Real)*_sNodes.size(depth-1) );

	// Iterate over the nodes @( depth )
	std::vector< SupportKey > neighborKeys( std::max< int >( 1 , threads ) );
	for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( depth-1 );
#pragma omp parallel for num_threads( threads )
	for( int i=_sNodes.begin(depth) ; i<_sNodes.end(depth) ; i++ ) if( _IsValidNode< FEMDegree >( _sNodes.treeNodes[i] ) )
	{
		SupportKey& neighborKey = neighborKeys[ omp_get_thread_num() ];
		TreeOctNode* node = _sNodes.treeNodes[i];

		// Offset the coarser constraints using the solution from the current resolutions.
		int x , y , z , c;
		c = int( node - node->parent->children );
		Cube::FactorCornerIndex( c , x , y , z );
		{
			typename TreeOctNode::Neighbors< OverlapSize > pNeighbors;
			neighborKey.template getNeighbors< false , OverlapRadius , OverlapRadius >( node->parent , pNeighbors );
			const Stencil< double , OverlapSize >& lapStencil = stencils[x][y][z];

			bool isInterior = _IsInteriorlyOverlapped< FEMDegree , FEMDegree >( node->parent );
			int d , off[3];
			node->depthAndOffset( d , off );

			// Offset the constraints using the solution from finer resolutions.
			int startX , endX , startY , endY , startZ , endZ;
			_SetParentOverlapBounds< FEMDegree , FEMDegree >( node , startX , endX , startY  , endY , startZ , endZ );

			Real solution = fineSolution[ node->nodeData.nodeIndex ];
			for( int x=startX ; x<endX ; x++ ) for( int y=startY ; y<endY ; y++ ) for( int z=startZ ; z<endZ ; z++ )
				if( _IsValidNode< FEMDegree >( pNeighbors.neighbors[x][y][z] ) )
				{
					const TreeOctNode* _node = pNeighbors.neighbors[x][y][z];
					if( isInterior )
#pragma omp atomic
						coarseConstraints[ _node->nodeData.nodeIndex ] += Real( lapStencil.values[x][y][z] * solution );
					else
					{
						int _d , _off[3];
						_node->depthAndOffset( _d , _off );
#pragma omp atomic
						coarseConstraints[ _node->nodeData.nodeIndex ] += Real( SystemCoefficients< FEMDegree , FEMDegree >::GetLaplacian( childIntegrator , _off , off ) * solution );
					}
				}
		}
	}
}


template< class Real >
template< int FEMDegree >
DenseNodeData< Real , FEMDegree > Octree< Real >::SolveSystem( SparseNodeData< PointData< Real > , 0 >& pointInfo , DenseNodeData< Real , FEMDegree >& constraints , bool showResidual , int iters , int maxSolveDepth , int cgDepth , double accuracy )
{
	BSplineData< FEMDegree > bsData;
	bsData.set( maxSolveDepth , _dirichlet );

	maxSolveDepth++;
	int iter=0;
	iters = std::max< int >( 0 , iters );

	DenseNodeData< Real , FEMDegree > solution( _sNodes.size() );
	memset( solution.data , 0 , sizeof(Real)*_sNodes.size() );

	solution[0] = 0;

	DenseNodeData< Real , FEMDegree > metSolution( _sNodes.end( _sNodes.levels()-2 ) );
	memset( metSolution.data , 0 , sizeof(Real)*_sNodes.end( _sNodes.levels()-2 ) );
	for( int d=_minDepth ; d<_sNodes.levels() ; d++ )
	{
		DumpOutput( "Depth[%d/%d]: %d\n" , d-1 , _sNodes.levels()-2 , _sNodes.size( d ) );
		if( d==_minDepth ) _SolveSystemCG( bsData , pointInfo , d , solution , constraints , metSolution , _sNodes.size(_minDepth) , true , showResidual , NULL , NULL , NULL );
		else
		{
			if( d>cgDepth ) iter += _SolveSystemGS( bsData , pointInfo , d , solution , constraints , metSolution , d>maxSolveDepth ? 0 : iters , true , showResidual , NULL , NULL , NULL );
			else            iter += _SolveSystemCG( bsData , pointInfo , d , solution , constraints , metSolution , d>maxSolveDepth ? 0 : iters , true , showResidual , NULL , NULL , NULL , accuracy );
		}
	}
	metSolution.resize( 0 );
	return solution;
}

template< class Real >
template< int FEMDegree , int NormalDegree >
DenseNodeData< Real , FEMDegree > Octree< Real >::SetLaplacianConstraints( const SparseNodeData< Point3D< Real > , NormalDegree >& normalInfo )
{
	typedef typename TreeOctNode::NeighborKey< -BSplineEvaluationData< FEMDegree >::SupportStart , BSplineEvaluationData< FEMDegree >::SupportEnd > SupportKey;
	const int               OverlapSize   =  BSplineIntegrationData< NormalDegree , FEMDegree >::OverlapSize;
	const int  LeftNormalFEMOverlapRadius = -BSplineIntegrationData< NormalDegree , FEMDegree >::OverlapStart;
	const int RightNormalFEMOverlapRadius =  BSplineIntegrationData< NormalDegree , FEMDegree >::OverlapEnd;
	const int  LeftFEMNormalOverlapRadius = -BSplineIntegrationData< FEMDegree , NormalDegree >::OverlapStart;
	const int RightFEMNormalOverlapRadius =  BSplineIntegrationData< FEMDegree , NormalDegree >::OverlapEnd;

	// To set the Laplacian constraints, we iterate over the
	// splatted normals and compute the dot-product of the
	// divergence of the normal field with all the basis functions.
	// Within the same depth: set directly as a gather
	// Coarser depths 
	int maxDepth = _sNodes.levels()-1;
	DenseNodeData< Real , FEMDegree > constraints( _sNodes.size() ) , _constraints( _sNodes.end( maxDepth-1 ) );
	memset( constraints.data , 0 , sizeof(Real)*_sNodes.size() );
	memset( _constraints.data , 0 , sizeof(Real)*( _sNodes.end(maxDepth-1) ) );
	MemoryUsage();

	for( int d=maxDepth ; d>=_minDepth ; d-- )
	{
		int offset = d>0 ? _sNodes.begin(d-1) : 0;
		Stencil< Point3D< double > , OverlapSize > stencil , stencils[2][2][2];
		typename BSplineIntegrationData< NormalDegree , FEMDegree >::FunctionIntegrator::Integrator integrator;
		typename BSplineIntegrationData< FEMDegree , NormalDegree >::FunctionIntegrator::ChildIntegrator childIntegrator;
		BSplineIntegrationData< NormalDegree , FEMDegree >::SetIntegrator( integrator , d-1 , _dirichlet , _dirichlet );
		if( d>_minDepth ) BSplineIntegrationData< FEMDegree , NormalDegree >::SetChildIntegrator( childIntegrator , d-2 , _dirichlet , _dirichlet );
		SystemCoefficients< NormalDegree , FEMDegree >::SetCentralDivergenceStencil (      integrator , stencil  , false );
		SystemCoefficients< FEMDegree , NormalDegree >::SetCentralDivergenceStencils( childIntegrator , stencils , true  );

		std::vector< SupportKey > neighborKeys( std::max< int >( 1 , threads ) );
		for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( _maxDepth );

#pragma omp parallel for num_threads( threads )
		for( int i=_sNodes.begin(d) ; i<_sNodes.end(d) ; i++ )
		{
			SupportKey& neighborKey = neighborKeys[ omp_get_thread_num() ];
			TreeOctNode* node = _sNodes.treeNodes[i];
			int startX=0 , endX=OverlapSize , startY=0 , endY=OverlapSize , startZ=0 , endZ=OverlapSize;
			int depth = node->depth();
			typename TreeOctNode::Neighbors< OverlapSize > neighbors;
			neighborKey.template getNeighbors< false , LeftFEMNormalOverlapRadius , RightFEMNormalOverlapRadius >( node , neighbors );
			bool isInterior = _IsInteriorlyOverlapped< FEMDegree , NormalDegree >( node ) , isInterior2 = _IsInteriorlyOverlapped< NormalDegree , FEMDegree >( node->parent );

			int cx , cy , cz;
			if( d>_minDepth ) Cube::FactorCornerIndex( (int)( node-node->parent->children) , cx , cy ,cz );
			else cx = cy = cz = 0;
			Stencil< Point3D< double > , OverlapSize >& _stencil = stencils[cx][cy][cz];
			int d , off[3];
			node->depthAndOffset( d , off );
			// Set constraints from current depth
			// Gather the constraints from the vector-field at _node into the constraint stored with node
			if( _IsValidNode< FEMDegree >( node ) )
			{
				for( int x=startX ; x<endX ; x++ ) for( int y=startY ; y<endY ; y++ ) for( int z=startZ ; z<endZ ; z++ )
				{
					const TreeOctNode* _node = neighbors.neighbors[x][y][z];
					if( _IsValidNode< NormalDegree >( _node ) )
					{
						int _idx = normalInfo.index( _node );
						if( _idx>=0 ) 
							if( isInterior ) constraints[i] += Point3D< Real >::Dot( stencil.values[x][y][z] , normalInfo.data[ _idx ] );
							else
							{
								int _d , _off[3];
								_node->depthAndOffset( _d , _off );
								constraints[i] += Real( SystemCoefficients< NormalDegree , FEMDegree >::GetDivergence2( integrator , _off , off , normalInfo.data[ _idx ] ) );
							}
					}
				}
				_SetParentOverlapBounds< NormalDegree , FEMDegree >( node , startX , endX , startY , endY , startZ , endZ );
			}
			if( !_IsValidNode< NormalDegree >( node ) ) continue;
			int idx = normalInfo.index( node );
			if( idx<0 ) continue;
			const Point3D< Real >& normal = normalInfo.data[ idx ];
			if( normal[0]==0 && normal[1]==0 && normal[2]==0 ) continue;

			// Set the _constraints for the parents
			if( depth>_minDepth )
			{
				neighborKey.template getNeighbors< false , LeftNormalFEMOverlapRadius , RightNormalFEMOverlapRadius >( node->parent , neighbors );

				for( int x=startX ; x<endX ; x++ ) for( int y=startY ; y<endY ; y++ ) for( int z=startZ ; z<endZ ; z++ )
				{
					TreeOctNode* _node = neighbors.neighbors[x][y][z];
					if( _node && ( isInterior2 || _IsValidNode< FEMDegree >( _node ) ) )
					{
						TreeOctNode* _node = neighbors.neighbors[x][y][z];
						Real c;
						if( isInterior2 ) c = Point3D< Real >::Dot( _stencil.values[x][y][z] , normal );
						else
						{
							int _d , _off[3];
							_node->depthAndOffset( _d , _off );
							c = Real( SystemCoefficients< FEMDegree , NormalDegree >::GetDivergence1( childIntegrator , _off , off , normal ) );
						}
#pragma omp atomic
						_constraints[ _node->nodeData.nodeIndex ] += c;
					}
				}
			}
		}
		MemoryUsage();
	}

	// Fine-to-coarse down-sampling of constraints
	for( int d=maxDepth-1 ; d>_minDepth ; d-- ) _DownSample( d , _constraints );

	// Add the accumulated constraints from all finer depths
#pragma omp parallel for num_threads( threads )
	for( int i=0 ; i<_sNodes.end(maxDepth-1) ; i++ ) constraints[i] += _constraints[i];

	_constraints.resize( 0 );

	DenseNodeData< Point3D< Real > , NormalDegree > coefficients( _sNodes.end( maxDepth-1 ) );
	for( int d=maxDepth-1 ; d>=_minDepth ; d-- )
	{
#pragma omp parallel for num_threads( threads )
		for( int i=_sNodes.begin(d) ; i<_sNodes.end(d) ; i++ ) if( _IsValidNode< NormalDegree >( _sNodes.treeNodes[i] ) )
		{
			int idx = normalInfo.index( _sNodes.treeNodes[i] );
			if( idx<0 ) continue;
			coefficients[i] = normalInfo.data[ idx ];
		}
	}

	// Coarse-to-fine up-sampling of coefficients
	for( int d=_minDepth+1 ; d<maxDepth ; d++ ) _UpSample( d , coefficients );

	// Compute the contribution from all coarser depths
	for( int d=_minDepth ; d<=maxDepth ; d++ )
	{
		size_t start = _sNodes.begin(d) , end = _sNodes.end(d) , range = end - start;
		Stencil< Point3D< double > , OverlapSize > stencils[2][2][2];
		typename BSplineIntegrationData< NormalDegree , FEMDegree >::FunctionIntegrator::ChildIntegrator childIntegrator;
		if( d>_minDepth ) BSplineIntegrationData< NormalDegree , FEMDegree >::SetChildIntegrator( childIntegrator , d-2 , _dirichlet , _dirichlet );
		SystemCoefficients< NormalDegree , FEMDegree >::SetCentralDivergenceStencils( childIntegrator , stencils , false );
		std::vector< SupportKey > neighborKeys( std::max< int >( 1 , threads ) );
		for( size_t i=0 ; i<neighborKeys.size() ; i++ ) neighborKeys[i].set( maxDepth );
#pragma omp parallel for num_threads( threads )
		for( int i=_sNodes.begin(d) ; i<_sNodes.end(d) ; i++ ) if( _IsValidNode< FEMDegree >( _sNodes.treeNodes[i] ) )
		{
			SupportKey& neighborKey = neighborKeys[ omp_get_thread_num() ];
			TreeOctNode* node = _sNodes.treeNodes[i];
			int depth = node->depth();
			if( !depth ) continue;
			int startX , endX , startY , endY , startZ , endZ;
			_SetParentOverlapBounds< FEMDegree , NormalDegree >( node , startX , endX , startY , endY , startZ , endZ );
			typename TreeOctNode::Neighbors< OverlapSize > neighbors;
			neighborKey.template getNeighbors< false , LeftFEMNormalOverlapRadius , RightFEMNormalOverlapRadius >( node->parent , neighbors );

			bool isInterior = _IsInteriorlyOverlapped< FEMDegree , NormalDegree >( node->parent );
			int cx , cy , cz;
			if( d )
			{
				int c = int( node - node->parent->children );
				Cube::FactorCornerIndex( c , cx , cy , cz );
			}
			else cx = cy = cz = 0;
			Stencil< Point3D< double > , OverlapSize >& _stencil = stencils[cx][cy][cz];

			Real constraint = Real(0);
			int d , off[3];
			node->depthAndOffset( d , off );
			for( int x=startX ; x<endX ; x++ ) for( int y=startY ; y<endY ; y++ ) for( int z=startZ ; z<endZ ; z++ )
			{
				TreeOctNode* _node = neighbors.neighbors[x][y][z];
				if( _IsValidNode< NormalDegree >( _node ) )
				{
					int _i = _node->nodeData.nodeIndex;
					if( isInterior ) constraint += Point3D< Real >::Dot( coefficients[_i] , _stencil.values[x][y][z] );
					else
					{
						int _d , _off[3];
						_node->depthAndOffset( _d , _off );
						constraint += Real( SystemCoefficients< NormalDegree , FEMDegree >::GetDivergence2( childIntegrator , _off , off , coefficients[_i] ) );
					}
				}
			}
			constraints[ node->nodeData.nodeIndex ] += constraint;
		}
	}
	MemoryUsage();
	coefficients.resize( 0 );

	return constraints;
}
