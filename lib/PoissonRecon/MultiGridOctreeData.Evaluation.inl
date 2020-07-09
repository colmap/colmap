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

template< class Real >
template< int FEMDegree >
void Octree< Real >::_Evaluator< FEMDegree >::set( int depth , bool dirichlet )
{
	static const int  LeftPointSupportRadius =  BSplineEvaluationData< FEMDegree >::SupportEnd;
	static const int RightPointSupportRadius = -BSplineEvaluationData< FEMDegree >::SupportStart;

	BSplineEvaluationData< FEMDegree >::SetEvaluator( evaluator , depth , dirichlet );
	if( depth>0 ) BSplineEvaluationData< FEMDegree >::SetChildEvaluator( childEvaluator , depth-1 , dirichlet );
	int center = BSplineData< FEMDegree >::Dimension( depth )>>1;

	// First set the stencils for the current depth
	for( int x=-LeftPointSupportRadius ; x<=RightPointSupportRadius ; x++ ) for( int y=-LeftPointSupportRadius ; y<=RightPointSupportRadius ; y++ ) for( int z=-LeftPointSupportRadius ; z<=RightPointSupportRadius ; z++ )
	{
		int fIdx[] = { center+x , center+y , center+z };

		//// The cell stencil
		{
			double vv[3] , dv[3];
			for( int dd=0 ; dd<DIMENSION ; dd++ )
			{
				vv[dd] = evaluator.centerValue( fIdx[dd] , center , false );
				dv[dd] = evaluator.centerValue( fIdx[dd] , center , true  );
			}
			 cellStencil.values[x+LeftPointSupportRadius][y+LeftPointSupportRadius][z+LeftPointSupportRadius] = vv[0] * vv[1] * vv[2];
			dCellStencil.values[x+LeftPointSupportRadius][y+LeftPointSupportRadius][z+LeftPointSupportRadius] = Point3D< double >( dv[0] * vv[1] * vv[2] , vv[0] * dv[1] * vv[2] , vv[0] * vv[1] * dv[2] );
		}

		//// The face stencil
		for( int f=0 ; f<Cube::FACES ; f++ )
		{
			int dir , off;
			Cube::FactorFaceIndex( f , dir , off );
			double vv[3] , dv[3];
			switch( dir )
			{
			case 0:
				vv[0] = evaluator.cornerValue( fIdx[0] , center+off , false );
				vv[1] = evaluator.centerValue( fIdx[1] , center     , false );
				vv[2] = evaluator.centerValue( fIdx[2] , center     , false );
				dv[0] = evaluator.cornerValue( fIdx[0] , center+off , true  );
				dv[1] = evaluator.centerValue( fIdx[1] , center     , true  );
				dv[2] = evaluator.centerValue( fIdx[2] , center     , true  );
				break;
			case 1:
				vv[0] = evaluator.centerValue( fIdx[0] , center     , false );
				vv[1] = evaluator.cornerValue( fIdx[1] , center+off , false );
				vv[2] = evaluator.centerValue( fIdx[2] , center     , false );
				dv[0] = evaluator.centerValue( fIdx[0] , center     , true  );
				dv[1] = evaluator.cornerValue( fIdx[1] , center+off , true  );
				dv[2] = evaluator.centerValue( fIdx[2] , center     , true  );
				break;
			case 2:
				vv[0] = evaluator.centerValue( fIdx[0] , center     , false );
				vv[1] = evaluator.centerValue( fIdx[1] , center     , false );
				vv[2] = evaluator.cornerValue( fIdx[2] , center+off , false );
				dv[0] = evaluator.centerValue( fIdx[0] , center     , true  );
				dv[1] = evaluator.centerValue( fIdx[1] , center     , true  );
				dv[2] = evaluator.cornerValue( fIdx[2] , center+off , true  );
				break;
			}
			 faceStencil[f].values[x+LeftPointSupportRadius][y+LeftPointSupportRadius][z+LeftPointSupportRadius] = vv[0] * vv[1] * vv[2];
			dFaceStencil[f].values[x+LeftPointSupportRadius][y+LeftPointSupportRadius][z+LeftPointSupportRadius] = Point3D< double >( dv[0] * vv[1] * vv[2] , vv[0] * dv[1] * vv[2] , vv[0] * vv[1] * dv[2] );
		}

		//// The edge stencil
		for( int e=0 ; e<Cube::EDGES ; e++ )
		{
			int orientation , i1 , i2;
			Cube::FactorEdgeIndex( e , orientation , i1 , i2 );
			double vv[3] , dv[3];
			switch( orientation )
			{
			case 0:
				vv[0] = evaluator.centerValue( fIdx[0] , center    , false );
				vv[1] = evaluator.cornerValue( fIdx[1] , center+i1 , false );
				vv[2] = evaluator.cornerValue( fIdx[2] , center+i2 , false );
				dv[0] = evaluator.centerValue( fIdx[0] , center    , true  );
				dv[1] = evaluator.cornerValue( fIdx[1] , center+i1 , true  );
				dv[2] = evaluator.cornerValue( fIdx[2] , center+i2 , true  );
				break;
			case 1:
				vv[0] = evaluator.cornerValue( fIdx[0] , center+i1 , false );
				vv[1] = evaluator.centerValue( fIdx[1] , center    , false );
				vv[2] = evaluator.cornerValue( fIdx[2] , center+i2 , false );
				dv[0] = evaluator.cornerValue( fIdx[0] , center+i1 , true  );
				dv[1] = evaluator.centerValue( fIdx[1] , center    , true  );
				dv[2] = evaluator.cornerValue( fIdx[2] , center+i2 , true  );
				break;
			case 2:
				vv[0] = evaluator.cornerValue( fIdx[0] , center+i1 , false );
				vv[1] = evaluator.cornerValue( fIdx[1] , center+i2 , false );
				vv[2] = evaluator.centerValue( fIdx[2] , center    , false );
				dv[0] = evaluator.cornerValue( fIdx[0] , center+i1 , true  );
				dv[1] = evaluator.cornerValue( fIdx[1] , center+i2 , true  );
				dv[2] = evaluator.centerValue( fIdx[2] , center    , true  );
				break;
			}
			 edgeStencil[e].values[x+LeftPointSupportRadius][y+LeftPointSupportRadius][z+LeftPointSupportRadius] = vv[0] * vv[1] * vv[2];
			dEdgeStencil[e].values[x+LeftPointSupportRadius][y+LeftPointSupportRadius][z+LeftPointSupportRadius] = Point3D< double >( dv[0] * vv[1] * vv[2] , vv[0] * dv[1] * vv[2] , vv[0] * vv[1] * dv[2] );
		}

		//// The corner stencil
		for( int c=0 ; c<Cube::CORNERS ; c++ )
		{
			int cx , cy  ,cz;
			Cube::FactorCornerIndex( c , cx , cy , cz );
			double vv[3] , dv[3];
			vv[0] = evaluator.cornerValue( fIdx[0] , center+cx , false );
			vv[1] = evaluator.cornerValue( fIdx[1] , center+cy , false );
			vv[2] = evaluator.cornerValue( fIdx[2] , center+cz , false );
			dv[0] = evaluator.cornerValue( fIdx[0] , center+cx , true  );
			dv[1] = evaluator.cornerValue( fIdx[1] , center+cy , true  );
			dv[2] = evaluator.cornerValue( fIdx[2] , center+cz , true  );
			 cornerStencil[c].values[x+LeftPointSupportRadius][y+LeftPointSupportRadius][z+LeftPointSupportRadius] = vv[0] * vv[1] * vv[2];
			dCornerStencil[c].values[x+LeftPointSupportRadius][y+LeftPointSupportRadius][z+LeftPointSupportRadius] = Point3D< double >( dv[0] * vv[1] * vv[2] , vv[0] * dv[1] * vv[2] , vv[0] * vv[1] * dv[2] );
		}
	}

	// Now set the stencils for the parents
	for( int child=0 ; child<CHILDREN ; child++ )
	{
		int childX , childY , childZ;
		Cube::FactorCornerIndex( child , childX , childY , childZ );
		for( int x=-LeftPointSupportRadius ; x<=RightPointSupportRadius ; x++ ) for( int y=-LeftPointSupportRadius ; y<=RightPointSupportRadius ; y++ ) for( int z=-LeftPointSupportRadius ; z<=RightPointSupportRadius ; z++ )
		{
			int fIdx[] = { center/2+x , center/2+y , center/2+z };

			//// The cell stencil
			{
				double vv[3] , dv[3];
				vv[0] = childEvaluator.centerValue( fIdx[0] , center+childX , false );
				vv[1] = childEvaluator.centerValue( fIdx[1] , center+childY , false );
				vv[2] = childEvaluator.centerValue( fIdx[2] , center+childZ , false );
				dv[0] = childEvaluator.centerValue( fIdx[0] , center+childX , true  );
				dv[1] = childEvaluator.centerValue( fIdx[1] , center+childY , true  );
				dv[2] = childEvaluator.centerValue( fIdx[2] , center+childZ , true  );
				 cellStencils[child].values[x+LeftPointSupportRadius][y+LeftPointSupportRadius][z+LeftPointSupportRadius] = vv[0] * vv[1] * vv[2];
				dCellStencils[child].values[x+LeftPointSupportRadius][y+LeftPointSupportRadius][z+LeftPointSupportRadius] = Point3D< double >( dv[0] * vv[1] * vv[2] , vv[0] * dv[1] * vv[2] , vv[0] * vv[1] * dv[2] );
			}

			//// The face stencil
			for( int f=0 ; f<Cube::FACES ; f++ )
			{
				int dir , off;
				Cube::FactorFaceIndex( f , dir , off );
				double vv[3] , dv[3];
				switch( dir )
				{
				case 0:
					vv[0] = childEvaluator.cornerValue( fIdx[0] , center+childX+off , false );
					vv[1] = childEvaluator.centerValue( fIdx[1] , center+childY     , false );
					vv[2] = childEvaluator.centerValue( fIdx[2] , center+childZ     , false );
					dv[0] = childEvaluator.cornerValue( fIdx[0] , center+childX+off , true  );
					dv[1] = childEvaluator.centerValue( fIdx[1] , center+childY     , true  );
					dv[2] = childEvaluator.centerValue( fIdx[2] , center+childZ     , true  );
					break;
				case 1:
					vv[0] = childEvaluator.centerValue( fIdx[0] , center+childX     , false );
					vv[1] = childEvaluator.cornerValue( fIdx[1] , center+childY+off , false );
					vv[2] = childEvaluator.centerValue( fIdx[2] , center+childZ     , false );
					dv[0] = childEvaluator.centerValue( fIdx[0] , center+childX     , true  );
					dv[1] = childEvaluator.cornerValue( fIdx[1] , center+childY+off , true  );
					dv[2] = childEvaluator.centerValue( fIdx[2] , center+childZ     , true  );
					break;
				case 2:
					vv[0] = childEvaluator.centerValue( fIdx[0] , center+childX     , false );
					vv[1] = childEvaluator.centerValue( fIdx[1] , center+childY     , false );
					vv[2] = childEvaluator.cornerValue( fIdx[2] , center+childZ+off , false );
					dv[0] = childEvaluator.centerValue( fIdx[0] , center+childX     , true  );
					dv[1] = childEvaluator.centerValue( fIdx[1] , center+childY     , true  );
					dv[2] = childEvaluator.cornerValue( fIdx[2] , center+childZ+off , true  );
					break;
				}
				 faceStencils[child][f].values[x+LeftPointSupportRadius][y+LeftPointSupportRadius][z+LeftPointSupportRadius] = vv[0] * vv[1] * vv[2];
				dFaceStencils[child][f].values[x+LeftPointSupportRadius][y+LeftPointSupportRadius][z+LeftPointSupportRadius] = Point3D< double >( dv[0] * vv[1] * vv[2] , vv[0] * dv[1] * vv[2] , vv[0] * vv[1] * dv[2] );
			}
	
			//// The edge stencil
			for( int e=0 ; e<Cube::EDGES ; e++ )
			{
				int orientation , i1 , i2;
				Cube::FactorEdgeIndex( e , orientation , i1 , i2 );
				double vv[3] , dv[3];
				switch( orientation )
				{
				case 0:
					vv[0] = childEvaluator.centerValue( fIdx[0] , center+childX    , false );
					vv[1] = childEvaluator.cornerValue( fIdx[1] , center+childY+i1 , false );
					vv[2] = childEvaluator.cornerValue( fIdx[2] , center+childZ+i2 , false );
					dv[0] = childEvaluator.centerValue( fIdx[0] , center+childX    , true  );
					dv[1] = childEvaluator.cornerValue( fIdx[1] , center+childY+i1 , true  );
					dv[2] = childEvaluator.cornerValue( fIdx[2] , center+childZ+i2 , true  );
					break;
				case 1:
					vv[0] = childEvaluator.cornerValue( fIdx[0] , center+childX+i1 , false );
					vv[1] = childEvaluator.centerValue( fIdx[1] , center+childY    , false );
					vv[2] = childEvaluator.cornerValue( fIdx[2] , center+childZ+i2 , false );
					dv[0] = childEvaluator.cornerValue( fIdx[0] , center+childX+i1 , true  );
					dv[1] = childEvaluator.centerValue( fIdx[1] , center+childY    , true  );
					dv[2] = childEvaluator.cornerValue( fIdx[2] , center+childZ+i2 , true  );
					break;
				case 2:
					vv[0] = childEvaluator.cornerValue( fIdx[0] , center+childX+i1 , false );
					vv[1] = childEvaluator.cornerValue( fIdx[1] , center+childY+i2 , false );
					vv[2] = childEvaluator.centerValue( fIdx[2] , center+childZ    , false );
					dv[0] = childEvaluator.cornerValue( fIdx[0] , center+childX+i1 , true  );
					dv[1] = childEvaluator.cornerValue( fIdx[1] , center+childY+i2 , true  );
					dv[2] = childEvaluator.centerValue( fIdx[2] , center+childZ    , true  );
					break;
				}
				 edgeStencils[child][e].values[x+LeftPointSupportRadius][y+LeftPointSupportRadius][z+LeftPointSupportRadius] = vv[0] * vv[1] * vv[2];
				dEdgeStencils[child][e].values[x+LeftPointSupportRadius][y+LeftPointSupportRadius][z+LeftPointSupportRadius] = Point3D< double >( dv[0] * vv[1] * vv[2] , vv[0] * dv[1] * vv[2] , vv[0] * vv[1] * dv[2] );
			}
	
			//// The corner stencil
			for( int c=0 ; c<Cube::CORNERS ; c++ )
			{
				int cx , cy  ,cz;
				Cube::FactorCornerIndex( c , cx , cy , cz );
				double vv[3] , dv[3];
				vv[0] = childEvaluator.cornerValue( fIdx[0] , center+childX+cx , false );
				vv[1] = childEvaluator.cornerValue( fIdx[1] , center+childY+cy , false );
				vv[2] = childEvaluator.cornerValue( fIdx[2] , center+childZ+cz , false );
				dv[0] = childEvaluator.cornerValue( fIdx[0] , center+childX+cx , true  );
				dv[1] = childEvaluator.cornerValue( fIdx[1] , center+childY+cy , true  );
				dv[2] = childEvaluator.cornerValue( fIdx[2] , center+childZ+cz , true  );
				 cornerStencils[child][c].values[x+LeftPointSupportRadius][y+LeftPointSupportRadius][z+LeftPointSupportRadius] = vv[0] * vv[1] * vv[2];
				dCornerStencils[child][c].values[x+LeftPointSupportRadius][y+LeftPointSupportRadius][z+LeftPointSupportRadius] = Point3D< double >( dv[0] * vv[1] * vv[2] , vv[0] * dv[1] * vv[2] , vv[0] * vv[1] * dv[2] );
			}
		}
	}
}
template< class Real >
template< class V , int FEMDegree >
V Octree< Real >::_getCenterValue( const ConstPointSupportKey< FEMDegree >& neighborKey , const TreeOctNode* node , const DenseNodeData< V , FEMDegree >& solution , const DenseNodeData< V , FEMDegree >& metSolution , const _Evaluator< FEMDegree >& evaluator , bool isInterior ) const
{
	static const int SupportSize = BSplineEvaluationData< FEMDegree >::SupportSize;
	static const int  LeftPointSupportRadius =   BSplineEvaluationData< FEMDegree >::SupportEnd;
	static const int RightPointSupportRadius = - BSplineEvaluationData< FEMDegree >::SupportStart;

	if( node->children ) fprintf( stderr , "[WARNING] getCenterValue assumes leaf node\n" );
	V value(0);
	int d = _Depth( node );

	if( isInterior )
	{
		const typename TreeOctNode::ConstNeighbors< SupportSize >& neighbors = _Neighbors< LeftPointSupportRadius , RightPointSupportRadius >( neighborKey , d );
		for( int i=0 ; i<SupportSize ; i++ ) for( int j=0 ; j<SupportSize ; j++ ) for( int k=0 ; k<SupportSize ; k++ )
		{
			const TreeOctNode* n = neighbors.neighbors[i][j][k];
			if( n ) value += solution[ n->nodeData.nodeIndex ] * Real( evaluator.cellStencil.values[i][j][k] );
		}
		if( d>_minDepth-1 )
		{
			int _corner = int( node - node->parent->children );
			const typename TreeOctNode::ConstNeighbors< SupportSize >& neighbors = _Neighbors< LeftPointSupportRadius , RightPointSupportRadius >( neighborKey , d-1 );
			for( int i=0 ; i<SupportSize ; i++ ) for( int j=0 ; j<SupportSize ; j++ ) for( int k=0 ; k<SupportSize ; k++ )
			{
				const TreeOctNode* n = neighbors.neighbors[i][j][k];
				if( n ) value += metSolution[n->nodeData.nodeIndex] * Real( evaluator.cellStencils[_corner].values[i][j][k] );
			}
		}
	}
	else
	{
		int cIdx[3];
		_DepthAndOffset( node , d , cIdx );
		const typename TreeOctNode::ConstNeighbors< SupportSize >& neighbors = _Neighbors< LeftPointSupportRadius , RightPointSupportRadius >( neighborKey , d );

		for( int i=0 ; i<SupportSize ; i++ ) for( int j=0 ; j<SupportSize ; j++ ) for( int k=0 ; k<SupportSize ; k++ )
		{
			const TreeOctNode* n = neighbors.neighbors[i][j][k];

			if( _IsValidNode< FEMDegree >( n ) )
			{
				int _d , fIdx[3];
				_DepthAndOffset( n , _d , fIdx );
				value +=
					solution[ n->nodeData.nodeIndex ] *
					Real(
						evaluator.evaluator.centerValue( fIdx[0] , cIdx[0] , false ) *
						evaluator.evaluator.centerValue( fIdx[1] , cIdx[1] , false ) *
						evaluator.evaluator.centerValue( fIdx[2] , cIdx[2] , false )
					);
			}
		}
		if( d>_minDepth-1 )
		{
			const typename TreeOctNode::ConstNeighbors< SupportSize >& neighbors = _Neighbors< LeftPointSupportRadius , RightPointSupportRadius >( neighborKey , d-1 );
			for( int i=0 ; i<SupportSize ; i++ ) for( int j=0 ; j<SupportSize ; j++ ) for( int k=0 ; k<SupportSize ; k++ )
			{
				const TreeOctNode* n = neighbors.neighbors[i][j][k];
				if( _IsValidNode< FEMDegree >( n ) )
				{
					int _d , fIdx[3];
					_DepthAndOffset( n , _d , fIdx );
					value +=
						metSolution[ n->nodeData.nodeIndex ] *
						Real(
							evaluator.childEvaluator.centerValue( fIdx[0] , cIdx[0] , false ) *
							evaluator.childEvaluator.centerValue( fIdx[1] , cIdx[1] , false ) *
							evaluator.childEvaluator.centerValue( fIdx[2] , cIdx[2] , false )
						);
				}
			}
		}
	}
	return value;
}
template< class Real >
template< class V , int FEMDegree >
V Octree< Real >::_getEdgeValue( const ConstPointSupportKey< FEMDegree >& neighborKey , const TreeOctNode* node , int edge , const DenseNodeData< V , FEMDegree >& solution , const DenseNodeData< V , FEMDegree >& metSolution , const _Evaluator< FEMDegree >& evaluator , bool isInterior ) const
{
	static const int SupportSize = BSplineEvaluationData< FEMDegree >::SupportSize;
	static const int  LeftPointSupportRadius =  BSplineEvaluationData< FEMDegree >::SupportEnd;
	static const int RightPointSupportRadius = -BSplineEvaluationData< FEMDegree >::SupportStart;
	V value(0);
	int d , cIdx[3];
	_DepthAndOffset( node , d , cIdx );
	int startX = 0 , endX = SupportSize , startY = 0 , endY = SupportSize , startZ = 0 , endZ = SupportSize;
	int orientation , i1 , i2;
	Cube::FactorEdgeIndex( edge , orientation , i1 , i2 );
	switch( orientation )
	{
		case 0:
			cIdx[1] += i1 , cIdx[2] += i2;
			if( i1 ) startY++ ; else endY--;
			if( i2 ) startZ++ ; else endZ--;
			break;
		case 1:
			cIdx[0] += i1 , cIdx[2] += i2;
			if( i1 ) startX++ ; else endX--;
			if( i2 ) startZ++ ; else endZ--;
			break;
		case 2:
			cIdx[0] += i1 , cIdx[1] += i2;
			if( i1 ) startX++ ; else endX--;
			if( i2 ) startY++ ; else endY--;
			break;
	}

	{
		const typename TreeOctNode::ConstNeighbors< SupportSize >& neighbors = _Neighbors< LeftPointSupportRadius , RightPointSupportRadius >( neighborKey , d );
		for( int x=startX ; x<endX ; x++ ) for( int y=startY ; y<endY ; y++ ) for( int z=startZ ; z<endZ ; z++ )
		{
			const TreeOctNode* _node = neighbors.neighbors[x][y][z];
			if( _IsValidNode< FEMDegree >( _node ) )
			{
				if( isInterior ) value += solution[ _node->nodeData.nodeIndex ] * evaluator.edgeStencil[edge].values[x][y][z];
				else
				{
					int _d , fIdx[3];
					_DepthAndOffset( _node , _d , fIdx );
					switch( orientation )
					{
						case 0:
							value +=
								solution[ _node->nodeData.nodeIndex ] *
									Real(
									evaluator.evaluator.centerValue( fIdx[0] , cIdx[0] , false ) *
									evaluator.evaluator.cornerValue( fIdx[1] , cIdx[1] , false ) *
									evaluator.evaluator.cornerValue( fIdx[2] , cIdx[2] , false )
								);
							break;
						case 1:
							value +=
								solution[ _node->nodeData.nodeIndex ] *
								Real(
									evaluator.evaluator.cornerValue( fIdx[0] , cIdx[0] , false ) *
									evaluator.evaluator.centerValue( fIdx[1] , cIdx[1] , false ) *
									evaluator.evaluator.cornerValue( fIdx[2] , cIdx[2] , false )
								);
							break;
						case 2:
							value +=
								solution[ _node->nodeData.nodeIndex ] *
								Real(
									evaluator.evaluator.cornerValue( fIdx[0] , cIdx[0] , false ) *
									evaluator.evaluator.cornerValue( fIdx[1] , cIdx[1] , false ) *
									evaluator.evaluator.centerValue( fIdx[2] , cIdx[2] , false )
								);
						break;
					}
				}
			}
		}
	}
	if( d>_minDepth-1 )
	{
		int _corner = int( node - node->parent->children );
		int _cx , _cy , _cz;
		Cube::FactorCornerIndex( _corner , _cx , _cy , _cz );
		// If the corner/child indices don't match, then the sample position is in the interior of the
		// coarser cell and so the full support resolution should be used.
		switch( orientation )
		{
		case 0:
			if( _cy!=i1 ) startY = 0 , endY = SupportSize;
			if( _cz!=i2 ) startZ = 0 , endZ = SupportSize;
			break;
		case 1:
			if( _cx!=i1 ) startX = 0 , endX = SupportSize;
			if( _cz!=i2 ) startZ = 0 , endZ = SupportSize;
			break;
		case 2:
			if( _cx!=i1 ) startX = 0 , endX = SupportSize;
			if( _cy!=i2 ) startY = 0 , endY = SupportSize;
			break;
		}
		const typename TreeOctNode::ConstNeighbors< SupportSize >& neighbors = _Neighbors< LeftPointSupportRadius , RightPointSupportRadius >( neighborKey , d-1 );
		for( int x=startX ; x<endX ; x++ ) for( int y=startY ; y<endY ; y++ ) for( int z=startZ ; z<endZ ; z++ )
		{
			const TreeOctNode* _node = neighbors.neighbors[x][y][z];
			if( _IsValidNode< FEMDegree >( _node ) )
			{
				if( isInterior ) value += metSolution[ _node->nodeData.nodeIndex ] * evaluator.edgeStencils[_corner][edge].values[x][y][z];
				else
				{
					int _d , fIdx[3];
					_DepthAndOffset( _node , _d , fIdx );
					switch( orientation )
					{
						case 0:
							value +=
								metSolution[ _node->nodeData.nodeIndex ] *
								Real(
									evaluator.childEvaluator.centerValue( fIdx[0] , cIdx[0] , false ) *
									evaluator.childEvaluator.cornerValue( fIdx[1] , cIdx[1] , false ) *
									evaluator.childEvaluator.cornerValue( fIdx[2] , cIdx[2] , false )
								);
							break;
						case 1:
							value +=
								metSolution[ _node->nodeData.nodeIndex ] *
								Real(
									evaluator.childEvaluator.cornerValue( fIdx[0] , cIdx[0] , false ) *
									evaluator.childEvaluator.centerValue( fIdx[1] , cIdx[1] , false ) *
									evaluator.childEvaluator.cornerValue( fIdx[2] , cIdx[2] , false )
								);
							break;
						case 2:
							value +=
								metSolution[ _node->nodeData.nodeIndex ] *
								Real(
									evaluator.childEvaluator.cornerValue( fIdx[0] , cIdx[0] , false ) *
									evaluator.childEvaluator.cornerValue( fIdx[1] , cIdx[1] , false ) *
									evaluator.childEvaluator.centerValue( fIdx[2] , cIdx[2] , false )
								);
							break;
					}
				}
			}
		}
	}
	return Real( value );
}
template< class Real >
template< int FEMDegree >
std::pair< Real , Point3D< Real > > Octree< Real >::_getEdgeValueAndGradient( const ConstPointSupportKey< FEMDegree >& neighborKey , const TreeOctNode* node , int edge , const DenseNodeData< Real , FEMDegree >& solution , const DenseNodeData< Real , FEMDegree >& metSolution , const _Evaluator< FEMDegree >& evaluator , bool isInterior ) const
{
	static const int SupportSize = BSplineEvaluationData< FEMDegree >::SupportSize;
	static const int  LeftPointSupportRadius =  BSplineEvaluationData< FEMDegree >::SupportEnd;
	static const int RightPointSupportRadius = -BSplineEvaluationData< FEMDegree >::SupportStart;
	double value = 0;
	Point3D< double > gradient;
	int d , cIdx[3];
	_DepthAndOffset( node , d , cIdx );

	int startX = 0 , endX = SupportSize , startY = 0 , endY = SupportSize , startZ = 0 , endZ = SupportSize;
	int orientation , i1 , i2;
	Cube::FactorEdgeIndex( edge , orientation , i1 , i2 );
	switch( orientation )
	{
		case 0:
			cIdx[1] += i1 , cIdx[2] += i2;
			if( i1 ) startY++ ; else endY--;
			if( i2 ) startZ++ ; else endZ--;
			break;
		case 1:
			cIdx[0] += i1 , cIdx[2] += i2;
			if( i1 ) startX++ ; else endX--;
			if( i2 ) startZ++ ; else endZ--;
			break;
		case 2:
			cIdx[0] += i1 , cIdx[1] += i2;
			if( i1 ) startX++ ; else endX--;
			if( i2 ) startY++ ; else endY--;
			break;
	}
	{
		const typename TreeOctNode::ConstNeighbors< SupportSize >& neighbors = _Neighbors< LeftPointSupportRadius , RightPointSupportRadius >( neighborKey , d );
		for( int x=startX ; x<endX ; x++ ) for( int y=startY ; y<endY ; y++ ) for( int z=startZ ; z<endZ ; z++ )
		{
			const TreeOctNode* _node = neighbors.neighbors[x][y][z];
			if( _IsValidNode< FEMDegree >( _node ) )
			{
				if( isInterior )
				{
					value    += evaluator. edgeStencil[edge].values[x][y][z] * solution[ _node->nodeData.nodeIndex ];
					gradient += evaluator.dEdgeStencil[edge].values[x][y][z] * solution[ _node->nodeData.nodeIndex ];
				}
				else
				{
					int _d , fIdx[3];
					_DepthAndOffset( _node , _d , fIdx );

					double vv[3] , dv[3];
					switch( orientation )
					{
						case 0:
							vv[0] = evaluator.evaluator.centerValue( fIdx[0] , cIdx[0] , false );
							vv[1] = evaluator.evaluator.cornerValue( fIdx[1] , cIdx[1] , false );
							vv[2] = evaluator.evaluator.cornerValue( fIdx[2] , cIdx[2] , false );
							dv[0] = evaluator.evaluator.centerValue( fIdx[0] , cIdx[0] , true  );
							dv[1] = evaluator.evaluator.cornerValue( fIdx[1] , cIdx[1] , true  );
							dv[2] = evaluator.evaluator.cornerValue( fIdx[2] , cIdx[2] , true  );
							break;
						case 1:
							vv[0] = evaluator.evaluator.cornerValue( fIdx[0] , cIdx[0] , false );
							vv[1] = evaluator.evaluator.centerValue( fIdx[1] , cIdx[1] , false );
							vv[2] = evaluator.evaluator.cornerValue( fIdx[2] , cIdx[2] , false );
							dv[0] = evaluator.evaluator.cornerValue( fIdx[0] , cIdx[0] , true  );
							dv[1] = evaluator.evaluator.centerValue( fIdx[1] , cIdx[1] , true  );
							dv[2] = evaluator.evaluator.cornerValue( fIdx[2] , cIdx[2] , true  );
							break;
						case 2:
							vv[0] = evaluator.evaluator.cornerValue( fIdx[0] , cIdx[0] , false );
							vv[1] = evaluator.evaluator.cornerValue( fIdx[1] , cIdx[1] , false );
							vv[2] = evaluator.evaluator.centerValue( fIdx[2] , cIdx[2] , false );
							dv[0] = evaluator.evaluator.cornerValue( fIdx[0] , cIdx[0] , true  );
							dv[1] = evaluator.evaluator.cornerValue( fIdx[1] , cIdx[1] , true  );
							dv[2] = evaluator.evaluator.centerValue( fIdx[2] , cIdx[2] , true  );
							break;
					}
					value += solution[ _node->nodeData.nodeIndex ] * vv[0] * vv[1] * vv[2];
					gradient += Point3D< double >( dv[0]*vv[1]*vv[2] , vv[0]*dv[1]*vv[2] , vv[0]*vv[1]*dv[2] ) * solution[ _node->nodeData.nodeIndex ];
				}
			}
		}
	}
	if( d>_minDepth-1 )
	{
		int _corner = int( node - node->parent->children );
		int _cx , _cy , _cz;
		Cube::FactorCornerIndex( _corner , _cx , _cy , _cz );
		// If the corner/child indices don't match, then the sample position is in the interior of the
		// coarser cell and so the full support resolution should be used.
		switch( orientation )
		{
		case 0:
			if( _cy!=i1 ) startY = 0 , endY = SupportSize;
			if( _cz!=i2 ) startZ = 0 , endZ = SupportSize;
			break;
		case 1:
			if( _cx!=i1 ) startX = 0 , endX = SupportSize;
			if( _cz!=i2 ) startZ = 0 , endZ = SupportSize;
			break;
		case 2:
			if( _cx!=i1 ) startX = 0 , endX = SupportSize;
			if( _cy!=i2 ) startY = 0 , endY = SupportSize;
			break;
		}
		const typename TreeOctNode::ConstNeighbors< SupportSize >& neighbors = _Neighbors< LeftPointSupportRadius , RightPointSupportRadius >( neighborKey , d-1 );
		for( int x=startX ; x<endX ; x++ ) for( int y=startY ; y<endY ; y++ ) for( int z=startZ ; z<endZ ; z++ )
		{
			const TreeOctNode* _node = neighbors.neighbors[x][y][z];
			if( _IsValidNode< FEMDegree >( _node ) )
			{
				if( isInterior )
				{
					value    += evaluator. edgeStencils[_corner][edge].values[x][y][z] * metSolution[ _node->nodeData.nodeIndex ];
					gradient += evaluator.dEdgeStencils[_corner][edge].values[x][y][z] * metSolution[ _node->nodeData.nodeIndex ];
				}
				else
				{
					int _d , fIdx[3];
					_DepthAndOffset( _node , _d , fIdx );
					double vv[3] , dv[3];
					switch( orientation )
					{
						case 0:
							vv[0] = evaluator.childEvaluator.centerValue( fIdx[0] , cIdx[0] , false );
							vv[1] = evaluator.childEvaluator.cornerValue( fIdx[1] , cIdx[1] , false );
							vv[2] = evaluator.childEvaluator.cornerValue( fIdx[2] , cIdx[2] , false );
							dv[0] = evaluator.childEvaluator.centerValue( fIdx[0] , cIdx[0] , true  );
							dv[1] = evaluator.childEvaluator.cornerValue( fIdx[1] , cIdx[1] , true  );
							dv[2] = evaluator.childEvaluator.cornerValue( fIdx[2] , cIdx[2] , true  );
							break;
						case 1:
							vv[0] = evaluator.childEvaluator.cornerValue( fIdx[0] , cIdx[0] , false );
							vv[1] = evaluator.childEvaluator.centerValue( fIdx[1] , cIdx[1] , false );
							vv[2] = evaluator.childEvaluator.cornerValue( fIdx[2] , cIdx[2] , false );
							dv[0] = evaluator.childEvaluator.cornerValue( fIdx[0] , cIdx[0] , true  );
							dv[1] = evaluator.childEvaluator.centerValue( fIdx[1] , cIdx[1] , true  );
							dv[2] = evaluator.childEvaluator.cornerValue( fIdx[2] , cIdx[2] , true  );
							break;
						case 2:
							vv[0] = evaluator.childEvaluator.cornerValue( fIdx[0] , cIdx[0] , false );
							vv[1] = evaluator.childEvaluator.cornerValue( fIdx[1] , cIdx[1] , false );
							vv[2] = evaluator.childEvaluator.centerValue( fIdx[2] , cIdx[2] , false );
							dv[0] = evaluator.childEvaluator.cornerValue( fIdx[0] , cIdx[0] , true  );
							dv[1] = evaluator.childEvaluator.cornerValue( fIdx[1] , cIdx[1] , true  );
							dv[2] = evaluator.childEvaluator.centerValue( fIdx[2] , cIdx[2] , true  );
							break;
					}
					value += metSolution[ _node->nodeData.nodeIndex ] * vv[0] * vv[1] * vv[2];
					gradient += Point3D< double >( dv[0]*vv[1]*vv[2] , vv[0]*dv[1]*vv[2] , vv[0]*vv[1]*dv[2] ) * metSolution[ _node->nodeData.nodeIndex ];
				}
			}
		}
	}
	return std::pair< Real , Point3D< Real > >( Real( value ) , Point3D< Real >( gradient ) );
}

template< class Real >
template< class V , int FEMDegree >
V Octree< Real >::_getCornerValue( const ConstPointSupportKey< FEMDegree >& neighborKey , const TreeOctNode* node , int corner , const DenseNodeData< V , FEMDegree >& solution , const DenseNodeData< V , FEMDegree >& metSolution , const _Evaluator< FEMDegree >& evaluator , bool isInterior ) const
{
	static const int SupportSize = BSplineEvaluationData< FEMDegree >::SupportSize;
	static const int  LeftPointSupportRadius =   BSplineEvaluationData< FEMDegree >::SupportEnd;
	static const int RightPointSupportRadius = - BSplineEvaluationData< FEMDegree >::SupportStart;

	V value(0);
	int d , cIdx[3];
	_DepthAndOffset( node , d , cIdx );

	int cx , cy , cz;
	int startX = 0 , endX = SupportSize , startY = 0 , endY = SupportSize , startZ = 0 , endZ = SupportSize;
	Cube::FactorCornerIndex( corner , cx , cy , cz );
	cIdx[0] += cx , cIdx[1] += cy , cIdx[2] += cz;
	{
		const typename TreeOctNode::ConstNeighbors< SupportSize >& neighbors = _Neighbors< LeftPointSupportRadius , RightPointSupportRadius >( neighborKey , d );
		if( cx==0 ) endX--;
		else      startX++;
		if( cy==0 ) endY--;
		else      startY++;
		if( cz==0 ) endZ--;
		else      startZ++;
		if( isInterior )
			for( int x=startX ; x<endX ; x++ ) for( int y=startY ; y<endY ; y++ ) for( int z=startZ ; z<endZ ; z++ )
			{
				const TreeOctNode* _node=neighbors.neighbors[x][y][z];
				if( _node ) value += solution[ _node->nodeData.nodeIndex ] * Real( evaluator.cornerStencil[corner].values[x][y][z] );
			}
		else
			for( int x=startX ; x<endX ; x++ ) for( int y=startY ; y<endY ; y++ ) for( int z=startZ ; z<endZ ; z++ )
			{
				const TreeOctNode* _node = neighbors.neighbors[x][y][z];
				if( _IsValidNode< FEMDegree >( _node ) )
				{
					int _d , fIdx[3];
					_DepthAndOffset( _node , _d , fIdx );
					value +=
						solution[ _node->nodeData.nodeIndex ] *
						Real(
							evaluator.evaluator.cornerValue( fIdx[0] , cIdx[0] , false ) *
							evaluator.evaluator.cornerValue( fIdx[1] , cIdx[1] , false ) *
							evaluator.evaluator.cornerValue( fIdx[2] , cIdx[2] , false )
						);
				}
			}
	}
	if( d>_minDepth-1 )
	{
		int _corner = int( node - node->parent->children );
		int _cx , _cy , _cz;
		Cube::FactorCornerIndex( _corner , _cx , _cy , _cz );
		// If the corner/child indices don't match, then the sample position is in the interior of the
		// coarser cell and so the full support resolution should be used.
		if( cx!=_cx ) startX = 0 , endX = SupportSize;
		if( cy!=_cy ) startY = 0 , endY = SupportSize;
		if( cz!=_cz ) startZ = 0 , endZ = SupportSize;
		const typename TreeOctNode::ConstNeighbors< SupportSize >& neighbors = _Neighbors< LeftPointSupportRadius , RightPointSupportRadius >( neighborKey , d-1 );
		if( isInterior )
			for( int x=startX ; x<endX ; x++ ) for( int y=startY ; y<endY ; y++ ) for( int z=startZ ; z<endZ ; z++ )
			{
				const TreeOctNode* _node=neighbors.neighbors[x][y][z];
				if( _node ) value += metSolution[ _node->nodeData.nodeIndex ] * Real( evaluator.cornerStencils[_corner][corner].values[x][y][z] );
			}
		else
			for( int x=startX ; x<endX ; x++ ) for( int y=startY ; y<endY ; y++ ) for( int z=startZ ; z<endZ ; z++ )
			{
				const TreeOctNode* _node = neighbors.neighbors[x][y][z];
				if( _IsValidNode< FEMDegree >( _node ) )
				{
					int _d , fIdx[3];
					_DepthAndOffset( _node , _d , fIdx );
					value +=
						metSolution[ _node->nodeData.nodeIndex ] *
						Real(
							evaluator.childEvaluator.cornerValue( fIdx[0] , cIdx[0] , false ) *
							evaluator.childEvaluator.cornerValue( fIdx[1] , cIdx[1] , false ) *
							evaluator.childEvaluator.cornerValue( fIdx[2] , cIdx[2] , false )
						);
				}
			}
	}
	return Real( value );
}
template< class Real >
template< int FEMDegree >
std::pair< Real , Point3D< Real > > Octree< Real >::_getCornerValueAndGradient( const ConstPointSupportKey< FEMDegree >& neighborKey , const TreeOctNode* node , int corner , const DenseNodeData< Real , FEMDegree >& solution , const DenseNodeData< Real , FEMDegree >& metSolution , const _Evaluator< FEMDegree >& evaluator , bool isInterior ) const
{
	static const int SupportSize = BSplineEvaluationData< FEMDegree >::SupportSize;
	static const int  LeftPointSupportRadius =   BSplineEvaluationData< FEMDegree >::SupportEnd;
	static const int RightPointSupportRadius = - BSplineEvaluationData< FEMDegree >::SupportStart;

	double value = 0;
	Point3D< double > gradient;
	int d , cIdx[3];
	_DepthAndOffset( node , d , cIdx );

	int cx , cy , cz;
	int startX = 0 , endX = SupportSize , startY = 0 , endY = SupportSize , startZ = 0 , endZ = SupportSize;
	Cube::FactorCornerIndex( corner , cx , cy , cz );
	cIdx[0] += cx , cIdx[1] += cy , cIdx[2] += cz;
	{
		if( cx==0 ) endX--;
		else      startX++;
		if( cy==0 ) endY--;
		else      startY++;
		if( cz==0 ) endZ--;
		else      startZ++;
		const typename TreeOctNode::ConstNeighbors< SupportSize >& neighbors = _Neighbors< LeftPointSupportRadius , RightPointSupportRadius >( neighborKey , d );
		if( isInterior )
			for( int x=startX ; x<endX ; x++ ) for( int y=startY ; y<endY ; y++ ) for( int z=startZ ; z<endZ ; z++ )
			{
				const TreeOctNode* _node=neighbors.neighbors[x][y][z];
				if( _node ) value += solution[ _node->nodeData.nodeIndex ] * evaluator.cornerStencil[corner].values[x][y][z] , gradient += evaluator.dCornerStencil[corner].values[x][y][z] * solution[ _node->nodeData.nodeIndex ];
			}
		else
			for( int x=startX ; x<endX ; x++ ) for( int y=startY ; y<endY ; y++ ) for( int z=startZ ; z<endZ ; z++ )
			{
				const TreeOctNode* _node = neighbors.neighbors[x][y][z];
				if( _IsValidNode< FEMDegree >( _node ) )
				{
					int _d , fIdx[3];
					_DepthAndOffset( _node , _d , fIdx );
					double v [] = { evaluator.evaluator.cornerValue( fIdx[0] , cIdx[0] , false ) , evaluator.evaluator.cornerValue( fIdx[1] , cIdx[1] , false ) , evaluator.evaluator.cornerValue( fIdx[2] , cIdx[2] , false ) };
					double dv[] = { evaluator.evaluator.cornerValue( fIdx[0] , cIdx[0] , true  ) , evaluator.evaluator.cornerValue( fIdx[1] , cIdx[1] , true  ) , evaluator.evaluator.cornerValue( fIdx[2] , cIdx[2] , true  ) };
					value += solution[ _node->nodeData.nodeIndex ] * v[0] * v[1] * v[2];
					gradient += Point3D< double >( dv[0]*v[1]*v[2] , v[0]*dv[1]*v[2] , v[0]*v[1]*dv[2] ) * solution[ _node->nodeData.nodeIndex ];
				}
			}
	}
	if( d>_minDepth-1 )
	{
		int _corner = int( node - node->parent->children );
		int _cx , _cy , _cz;
		Cube::FactorCornerIndex( _corner , _cx , _cy , _cz );
		if( cx!=_cx ) startX = 0 , endX = SupportSize;
		if( cy!=_cy ) startY = 0 , endY = SupportSize;
		if( cz!=_cz ) startZ = 0 , endZ = SupportSize;
		const typename TreeOctNode::ConstNeighbors< SupportSize >& neighbors = _Neighbors< LeftPointSupportRadius , RightPointSupportRadius >( neighborKey , d-1 );
		if( isInterior )
			for( int x=startX ; x<endX ; x++ ) for( int y=startY ; y<endY ; y++ ) for( int z=startZ ; z<endZ ; z++ )
			{
				const TreeOctNode* _node=neighbors.neighbors[x][y][z];
				if( _node ) value += metSolution[ _node->nodeData.nodeIndex ] * evaluator.cornerStencils[_corner][corner].values[x][y][z] , gradient += evaluator.dCornerStencils[_corner][corner].values[x][y][z] * metSolution[ _node->nodeData.nodeIndex ];
			}
		else
			for( int x=startX ; x<endX ; x++ ) for( int y=startY ; y<endY ; y++ ) for( int z=startZ ; z<endZ ; z++ )
			{
				const TreeOctNode* _node = neighbors.neighbors[x][y][z];
				if( _IsValidNode< FEMDegree >( _node ) )
				{
					int _d , fIdx[3];
					_DepthAndOffset( _node , _d , fIdx );
					double v [] = { evaluator.childEvaluator.cornerValue( fIdx[0] , cIdx[0] , false ) , evaluator.childEvaluator.cornerValue( fIdx[1] , cIdx[1] , false ) , evaluator.childEvaluator.cornerValue( fIdx[2] , cIdx[2] , false ) };
					double dv[] = { evaluator.childEvaluator.cornerValue( fIdx[0] , cIdx[0] , true  ) , evaluator.childEvaluator.cornerValue( fIdx[1] , cIdx[1] , true  ) , evaluator.childEvaluator.cornerValue( fIdx[2] , cIdx[2] , true  ) };
					value += metSolution[ _node->nodeData.nodeIndex ] * v[0] * v[1] * v[2];
					gradient += Point3D< double >( dv[0]*v[1]*v[2] , v[0]*dv[1]*v[2] , v[0]*v[1]*dv[2] ) * metSolution[ _node->nodeData.nodeIndex ];
				}
			}
	}
	return std::pair< Real , Point3D< Real > >( Real( value ) , Point3D< Real >( gradient ) );
}
