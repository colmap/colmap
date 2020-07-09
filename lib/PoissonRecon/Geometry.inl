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

#include <stdio.h>

template<class Real>
Real Random(void){return Real(rand())/RAND_MAX;}

template<class Real>
Point3D<Real> RandomBallPoint(void){
	Point3D<Real> p;
	while(1){
		p.coords[0]=Real(1.0-2.0*Random<Real>());
		p.coords[1]=Real(1.0-2.0*Random<Real>());
		p.coords[2]=Real(1.0-2.0*Random<Real>());
		double l=SquareLength(p);
		if(l<=1){return p;}
	}
}
template<class Real>
Point3D<Real> RandomSpherePoint(void){
	Point3D<Real> p=RandomBallPoint<Real>();
	Real l=Real(Length(p));
	p.coords[0]/=l;
	p.coords[1]/=l;
	p.coords[2]/=l;
	return p;
}

template<class Real>
double SquareLength(const Point3D<Real>& p){return p.coords[0]*p.coords[0]+p.coords[1]*p.coords[1]+p.coords[2]*p.coords[2];}

template<class Real>
double Length(const Point3D<Real>& p){return sqrt(SquareLength(p));}

template<class Real>
double SquareDistance(const Point3D<Real>& p1,const Point3D<Real>& p2){
	return (p1.coords[0]-p2.coords[0])*(p1.coords[0]-p2.coords[0])+(p1.coords[1]-p2.coords[1])*(p1.coords[1]-p2.coords[1])+(p1.coords[2]-p2.coords[2])*(p1.coords[2]-p2.coords[2]);
}

template<class Real>
double Distance(const Point3D<Real>& p1,const Point3D<Real>& p2){return sqrt(SquareDistance(p1,p2));}

template <class Real>
void CrossProduct(const Point3D<Real>& p1,const Point3D<Real>& p2,Point3D<Real>& p){
	p.coords[0]= p1.coords[1]*p2.coords[2]-p1.coords[2]*p2.coords[1];
	p.coords[1]=-p1.coords[0]*p2.coords[2]+p1.coords[2]*p2.coords[0];
	p.coords[2]= p1.coords[0]*p2.coords[1]-p1.coords[1]*p2.coords[0];
}
template<class Real>
void EdgeCollapse(const Real& edgeRatio,std::vector<TriangleIndex>& triangles,std::vector< Point3D<Real> >& positions,std::vector< Point3D<Real> >* normals){
	int i,j,*remapTable,*pointCount,idx[3];
	Point3D<Real> p[3],q[2],c;
	double d[3],a;
	double Ratio=12.0/sqrt(3.0);	// (Sum of Squares Length / Area) for and equilateral triangle

	remapTable=new int[positions.size()];
	pointCount=new int[positions.size()];
	for(i=0;i<int(positions.size());i++){
		remapTable[i]=i;
		pointCount[i]=1;
	}
	for(i=int(triangles.size()-1);i>=0;i--){
		for(j=0;j<3;j++){
			idx[j]=triangles[i].idx[j];
			while(remapTable[idx[j]]<idx[j]){idx[j]=remapTable[idx[j]];}
		}
		if(idx[0]==idx[1] || idx[0]==idx[2] || idx[1]==idx[2]){
			triangles[i]=triangles[triangles.size()-1];
			triangles.pop_back();
			continue;
		}
		for(j=0;j<3;j++){
			p[j].coords[0]=positions[idx[j]].coords[0]/pointCount[idx[j]];
			p[j].coords[1]=positions[idx[j]].coords[1]/pointCount[idx[j]];
			p[j].coords[2]=positions[idx[j]].coords[2]/pointCount[idx[j]];
		}
		for(j=0;j<3;j++){
			q[0].coords[j]=p[1].coords[j]-p[0].coords[j];
			q[1].coords[j]=p[2].coords[j]-p[0].coords[j];
			d[j]=SquareDistance(p[j],p[(j+1)%3]);
		}
		CrossProduct(q[0],q[1],c);
		a=Length(c)/2;

		if((d[0]+d[1]+d[2])*edgeRatio > a*Ratio){
			// Find the smallest edge
			j=0;
			if(d[1]<d[j]){j=1;}
			if(d[2]<d[j]){j=2;}

			int idx1,idx2;
			if(idx[j]<idx[(j+1)%3]){
				idx1=idx[j];
				idx2=idx[(j+1)%3];
			}
			else{
				idx2=idx[j];
				idx1=idx[(j+1)%3];
			}
			positions[idx1].coords[0]+=positions[idx2].coords[0];
			positions[idx1].coords[1]+=positions[idx2].coords[1];
			positions[idx1].coords[2]+=positions[idx2].coords[2];
			if(normals){
				(*normals)[idx1].coords[0]+=(*normals)[idx2].coords[0];
				(*normals)[idx1].coords[1]+=(*normals)[idx2].coords[1];
				(*normals)[idx1].coords[2]+=(*normals)[idx2].coords[2];
			}
			pointCount[idx1]+=pointCount[idx2];
			remapTable[idx2]=idx1;
			triangles[i]=triangles[triangles.size()-1];
			triangles.pop_back();
		}
	}
	int pCount=0;
	for(i=0;i<int(positions.size());i++){
		for(j=0;j<3;j++){positions[i].coords[j]/=pointCount[i];}
		if(normals){
			Real l=Real(Length((*normals)[i]));
			for(j=0;j<3;j++){(*normals)[i].coords[j]/=l;}
		}
		if(remapTable[i]==i){ // If vertex i is being used
			positions[pCount]=positions[i];
			if(normals){(*normals)[pCount]=(*normals)[i];}
			pointCount[i]=pCount;
			pCount++;
		}
	}
	positions.resize(pCount);
	for(i=int(triangles.size()-1);i>=0;i--){
		for(j=0;j<3;j++){
			idx[j]=triangles[i].idx[j];
			while(remapTable[idx[j]]<idx[j]){idx[j]=remapTable[idx[j]];}
			triangles[i].idx[j]=pointCount[idx[j]];
		}
		if(idx[0]==idx[1] || idx[0]==idx[2] || idx[1]==idx[2]){
			triangles[i]=triangles[triangles.size()-1];
			triangles.pop_back();
		}
	}

	delete[] pointCount;
	delete[] remapTable;
}
template<class Real>
void TriangleCollapse(const Real& edgeRatio,std::vector<TriangleIndex>& triangles,std::vector< Point3D<Real> >& positions,std::vector< Point3D<Real> >* normals){
	int i,j,*remapTable,*pointCount,idx[3];
	Point3D<Real> p[3],q[2],c;
	double d[3],a;
	double Ratio=12.0/sqrt(3.0);	// (Sum of Squares Length / Area) for and equilateral triangle

	remapTable=new int[positions.size()];
	pointCount=new int[positions.size()];
	for(i=0;i<int(positions.size());i++){
		remapTable[i]=i;
		pointCount[i]=1;
	}
	for(i=int(triangles.size()-1);i>=0;i--){
		for(j=0;j<3;j++){
			idx[j]=triangles[i].idx[j];
			while(remapTable[idx[j]]<idx[j]){idx[j]=remapTable[idx[j]];}
		}
		if(idx[0]==idx[1] || idx[0]==idx[2] || idx[1]==idx[2]){
			triangles[i]=triangles[triangles.size()-1];
			triangles.pop_back();
			continue;
		}
		for(j=0;j<3;j++){
			p[j].coords[0]=positions[idx[j]].coords[0]/pointCount[idx[j]];
			p[j].coords[1]=positions[idx[j]].coords[1]/pointCount[idx[j]];
			p[j].coords[2]=positions[idx[j]].coords[2]/pointCount[idx[j]];
		}
		for(j=0;j<3;j++){
			q[0].coords[j]=p[1].coords[j]-p[0].coords[j];
			q[1].coords[j]=p[2].coords[j]-p[0].coords[j];
			d[j]=SquareDistance(p[j],p[(j+1)%3]);
		}
		CrossProduct(q[0],q[1],c);
		a=Length(c)/2;

		if((d[0]+d[1]+d[2])*edgeRatio > a*Ratio){
			// Find the smallest edge
			j=0;
			if(d[1]<d[j]){j=1;}
			if(d[2]<d[j]){j=2;}

			int idx1,idx2,idx3;
			if(idx[0]<idx[1]){
				if(idx[0]<idx[2]){
					idx1=idx[0];
					idx2=idx[2];
					idx3=idx[1];
				}
				else{
					idx1=idx[2];
					idx2=idx[0];
					idx3=idx[1];
				}
			}
			else{
				if(idx[1]<idx[2]){
					idx1=idx[1];
					idx2=idx[2];
					idx3=idx[0];
				}
				else{
					idx1=idx[2];
					idx2=idx[1];
					idx3=idx[0];
				}
			}
			positions[idx1].coords[0]+=positions[idx2].coords[0]+positions[idx3].coords[0];
			positions[idx1].coords[1]+=positions[idx2].coords[1]+positions[idx3].coords[1];
			positions[idx1].coords[2]+=positions[idx2].coords[2]+positions[idx3].coords[2];
			if(normals){
				(*normals)[idx1].coords[0]+=(*normals)[idx2].coords[0]+(*normals)[idx3].coords[0];
				(*normals)[idx1].coords[1]+=(*normals)[idx2].coords[1]+(*normals)[idx3].coords[1];
				(*normals)[idx1].coords[2]+=(*normals)[idx2].coords[2]+(*normals)[idx3].coords[2];
			}
			pointCount[idx1]+=pointCount[idx2]+pointCount[idx3];
			remapTable[idx2]=idx1;
			remapTable[idx3]=idx1;
			triangles[i]=triangles[triangles.size()-1];
			triangles.pop_back();
		}
	}
	int pCount=0;
	for(i=0;i<int(positions.size());i++){
		for(j=0;j<3;j++){positions[i].coords[j]/=pointCount[i];}
		if(normals){
			Real l=Real(Length((*normals)[i]));
			for(j=0;j<3;j++){(*normals)[i].coords[j]/=l;}
		}
		if(remapTable[i]==i){ // If vertex i is being used
			positions[pCount]=positions[i];
			if(normals){(*normals)[pCount]=(*normals)[i];}
			pointCount[i]=pCount;
			pCount++;
		}
	}
	positions.resize(pCount);
	for(i=int(triangles.size()-1);i>=0;i--){
		for(j=0;j<3;j++){
			idx[j]=triangles[i].idx[j];
			while(remapTable[idx[j]]<idx[j]){idx[j]=remapTable[idx[j]];}
			triangles[i].idx[j]=pointCount[idx[j]];
		}
		if(idx[0]==idx[1] || idx[0]==idx[2] || idx[1]==idx[2]){
			triangles[i]=triangles[triangles.size()-1];
			triangles.pop_back();
		}
	}
	delete[] pointCount;
	delete[] remapTable;
}

///////////////////
// Triangulation //
///////////////////
template<class Real>
long long Triangulation<Real>::EdgeIndex( int p1 , int p2 )
{
	if(p1>p2)	{return ((long long)(p1)<<32) | ((long long)(p2));}
	else		{return ((long long)(p2)<<32) | ((long long)(p1));}
}

template<class Real>
int Triangulation<Real>::factor(int tIndex,int& p1,int& p2,int & p3){
	if(triangles[tIndex].eIndex[0]<0 || triangles[tIndex].eIndex[1]<0 || triangles[tIndex].eIndex[2]<0){return 0;}
	if(edges[triangles[tIndex].eIndex[0]].tIndex[0]==tIndex){p1=edges[triangles[tIndex].eIndex[0]].pIndex[0];}
	else													{p1=edges[triangles[tIndex].eIndex[0]].pIndex[1];}
	if(edges[triangles[tIndex].eIndex[1]].tIndex[0]==tIndex){p2=edges[triangles[tIndex].eIndex[1]].pIndex[0];}
	else													{p2=edges[triangles[tIndex].eIndex[1]].pIndex[1];}
	if(edges[triangles[tIndex].eIndex[2]].tIndex[0]==tIndex){p3=edges[triangles[tIndex].eIndex[2]].pIndex[0];}
	else													{p3=edges[triangles[tIndex].eIndex[2]].pIndex[1];}
	return 1;
}
template<class Real>
double Triangulation<Real>::area(int p1,int p2,int p3){
	Point3D<Real> q1,q2,q;
	for(int i=0;i<3;i++){
		q1.coords[i]=points[p2].coords[i]-points[p1].coords[i];
		q2.coords[i]=points[p3].coords[i]-points[p1].coords[i];
	}
	CrossProduct(q1,q2,q);
	return Length(q);
}
template<class Real>
double Triangulation<Real>::area(int tIndex){
	int p1,p2,p3;
	factor(tIndex,p1,p2,p3);
	return area(p1,p2,p3);
}
template<class Real>
double Triangulation<Real>::area(void){
	double a=0;
	for(int i=0;i<int(triangles.size());i++){a+=area(i);}
	return a;
}
template<class Real>
int Triangulation<Real>::addTriangle(int p1,int p2,int p3){
	hash_map<long long,int>::iterator iter;
	int tIdx,eIdx,p[3];
	p[0]=p1;
	p[1]=p2;
	p[2]=p3;
	triangles.push_back(TriangulationTriangle());
	tIdx=int(triangles.size())-1;

	for(int i=0;i<3;i++)
	{
		long long e = EdgeIndex(p[i],p[(i+1)%3]);
		iter=edgeMap.find(e);
		if(iter==edgeMap.end())
		{
			TriangulationEdge edge;
			edge.pIndex[0]=p[i];
			edge.pIndex[1]=p[(i+1)%3];
			edges.push_back(edge);
			eIdx=int(edges.size())-1;
			edgeMap[e]=eIdx;
			edges[eIdx].tIndex[0]=tIdx;
		}
		else{
			eIdx=edgeMap[e];
			if(edges[eIdx].pIndex[0]==p[i]){
				if(edges[eIdx].tIndex[0]<0){edges[eIdx].tIndex[0]=tIdx;}
				else{printf("Edge Triangle in use 1\n");return 0;}
			}
			else{
				if(edges[eIdx].tIndex[1]<0){edges[eIdx].tIndex[1]=tIdx;}
				else{printf("Edge Triangle in use 2\n");return 0;}
			}

		}
		triangles[tIdx].eIndex[i]=eIdx;
	}
	return tIdx;
}
template<class Real>
int Triangulation<Real>::flipMinimize(int eIndex){
	double oldArea,newArea;
	int oldP[3],oldQ[3],newP[3],newQ[3];
	TriangulationEdge newEdge;

	if(edges[eIndex].tIndex[0]<0 || edges[eIndex].tIndex[1]<0){return 0;}

	if(!factor(edges[eIndex].tIndex[0],oldP[0],oldP[1],oldP[2])){return 0;}
	if(!factor(edges[eIndex].tIndex[1],oldQ[0],oldQ[1],oldQ[2])){return 0;}

	oldArea=area(oldP[0],oldP[1],oldP[2])+area(oldQ[0],oldQ[1],oldQ[2]);
	int idxP,idxQ;
	for(idxP=0;idxP<3;idxP++){
		int i;
		for(i=0;i<3;i++){if(oldP[idxP]==oldQ[i]){break;}}
		if(i==3){break;}
	}
	for(idxQ=0;idxQ<3;idxQ++){
		int i;
		for(i=0;i<3;i++){if(oldP[i]==oldQ[idxQ]){break;}}
		if(i==3){break;}
	}
	if(idxP==3 || idxQ==3){return 0;}
	newP[0]=oldP[idxP];
	newP[1]=oldP[(idxP+1)%3];
	newP[2]=oldQ[idxQ];
	newQ[0]=oldQ[idxQ];
	newQ[1]=oldP[(idxP+2)%3];
	newQ[2]=oldP[idxP];

	newArea=area(newP[0],newP[1],newP[2])+area(newQ[0],newQ[1],newQ[2]);
	if(oldArea<=newArea){return 0;}

	// Remove the entry in the hash_table for the old edge
	edgeMap.erase(EdgeIndex(edges[eIndex].pIndex[0],edges[eIndex].pIndex[1]));
	// Set the new edge so that the zero-side is newQ
	edges[eIndex].pIndex[0]=newP[0];
	edges[eIndex].pIndex[1]=newQ[0];
	// Insert the entry into the hash_table for the new edge
	edgeMap[EdgeIndex(newP[0],newQ[0])]=eIndex;
	// Update the triangle information
	for(int i=0;i<3;i++){
		int idx;
		idx=edgeMap[EdgeIndex(newQ[i],newQ[(i+1)%3])];
		triangles[edges[eIndex].tIndex[0]].eIndex[i]=idx;
		if(idx!=eIndex){
			if(edges[idx].tIndex[0]==edges[eIndex].tIndex[1]){edges[idx].tIndex[0]=edges[eIndex].tIndex[0];}
			if(edges[idx].tIndex[1]==edges[eIndex].tIndex[1]){edges[idx].tIndex[1]=edges[eIndex].tIndex[0];}
		}

		idx=edgeMap[EdgeIndex(newP[i],newP[(i+1)%3])];
		triangles[edges[eIndex].tIndex[1]].eIndex[i]=idx;
		if(idx!=eIndex){
			if(edges[idx].tIndex[0]==edges[eIndex].tIndex[0]){edges[idx].tIndex[0]=edges[eIndex].tIndex[1];}
			if(edges[idx].tIndex[1]==edges[eIndex].tIndex[0]){edges[idx].tIndex[1]=edges[eIndex].tIndex[1];}
		}
	}
	return 1;
}
/////////////////////////
// CoredVectorMeshData //
/////////////////////////
template< class Vertex >
CoredVectorMeshData< Vertex >::CoredVectorMeshData( void ) { oocPointIndex = polygonIndex = 0; }
template< class Vertex >
void CoredVectorMeshData< Vertex >::resetIterator ( void ) { oocPointIndex = polygonIndex = 0; }
template< class Vertex >
int CoredVectorMeshData< Vertex >::addOutOfCorePoint( const Vertex& p )
{
	oocPoints.push_back(p);
	return int(oocPoints.size())-1;
}
template< class Vertex >
int CoredVectorMeshData< Vertex >::addOutOfCorePoint_s( const Vertex& p )
{
	size_t sz;
#pragma omp critical (CoredVectorMeshData_addOutOfCorePoint_s )
	{
		sz = oocPoints.size();
		oocPoints.push_back(p);
	}
	return (int)sz;
}
template< class Vertex >
int CoredVectorMeshData< Vertex >::addPolygon_s( const std::vector< int >& polygon )
{
	size_t sz;
#pragma omp critical (CoredVectorMeshData_addPolygon_s)
	{
		sz = polygon.size();
		polygons.push_back( polygon );
	}
	return (int)sz;
}
template< class Vertex >
int CoredVectorMeshData< Vertex >::addPolygon_s( const std::vector< CoredVertexIndex >& vertices )
{
	std::vector< int > polygon( vertices.size() );
	for( int i=0 ; i<(int)vertices.size() ; i++ ) 
		if( vertices[i].inCore ) polygon[i] =  vertices[i].idx;
		else                     polygon[i] = -vertices[i].idx-1;
	return addPolygon_s( polygon );
}
template< class Vertex >
int CoredVectorMeshData< Vertex >::nextOutOfCorePoint( Vertex& p )
{
	if( oocPointIndex<int(oocPoints.size()) )
	{
		p=oocPoints[oocPointIndex++];
		return 1;
	}
	else{return 0;}
}
template< class Vertex >
int CoredVectorMeshData< Vertex >::nextPolygon( std::vector< CoredVertexIndex >& vertices )
{
	if( polygonIndex<int( polygons.size() ) )
	{
		std::vector< int >& polygon = polygons[ polygonIndex++ ];
		vertices.resize( polygon.size() );
		for( int i=0 ; i<int(polygon.size()) ; i++ )
			if( polygon[i]<0 ) vertices[i].idx = -polygon[i]-1 , vertices[i].inCore = false;
			else               vertices[i].idx =  polygon[i]   , vertices[i].inCore = true;
		return 1;
	}
	else return 0;
}
template< class Vertex >
int CoredVectorMeshData< Vertex >::outOfCorePointCount(void){return int(oocPoints.size());}
template< class Vertex >
int CoredVectorMeshData< Vertex >::polygonCount( void ) { return int( polygons.size() ); }

///////////////////////
// CoredFileMeshData //
///////////////////////
template< class Vertex >
CoredFileMeshData< Vertex >::CoredFileMeshData( void )
{
	oocPoints = polygons = 0;
	
	oocPointFile = new BufferedReadWriteFile();
	polygonFile = new BufferedReadWriteFile();
}
template< class Vertex >
CoredFileMeshData< Vertex >::~CoredFileMeshData( void )
{
	delete oocPointFile;
	delete polygonFile;
}
template< class Vertex >
void CoredFileMeshData< Vertex >::resetIterator ( void )
{
	oocPointFile->reset();
	polygonFile->reset();
}
template< class Vertex >
int CoredFileMeshData< Vertex >::addOutOfCorePoint( const Vertex& p )
{
	oocPointFile->write( &p , sizeof( Vertex ) );
	oocPoints++;
	return oocPoints-1;
}
template< class Vertex >
int CoredFileMeshData< Vertex >::addOutOfCorePoint_s( const Vertex& p )
{
	int sz;
#pragma omp critical (CoredFileMeshData_addOutOfCorePoint_s)
	{
		sz = oocPoints;
		oocPointFile->write( &p , sizeof( Vertex ) );
		oocPoints++;
	}
	return sz;
}
template< class Vertex >
int CoredFileMeshData< Vertex >::addPolygon_s( const std::vector< int >& vertices )
{
	int sz , vSize = (int)vertices.size();
#pragma omp critical (CoredFileMeshData_addPolygon_s )
	{
		sz = polygons;
		polygonFile->write( &vSize , sizeof(int) );
		polygonFile->write( &vertices[0] , sizeof(int) * vSize );
		polygons++;
	}
	return sz;
}
template< class Vertex >
int CoredFileMeshData< Vertex >::addPolygon_s( const std::vector< CoredVertexIndex >& vertices )
{
	std::vector< int > polygon( vertices.size() );
	for( int i=0 ; i<(int)vertices.size() ; i++ ) 
		if( vertices[i].inCore ) polygon[i] =  vertices[i].idx;
		else                     polygon[i] = -vertices[i].idx-1;
	return addPolygon_s( polygon );
}
template< class Vertex >
int CoredFileMeshData< Vertex >::nextOutOfCorePoint( Vertex& p )
{
	if( oocPointFile->read( &p , sizeof( Vertex ) ) ) return 1;
	else return 0;
}
template< class Vertex >
int CoredFileMeshData< Vertex >::nextPolygon( std::vector< CoredVertexIndex >& vertices )
{
	int pSize;
	if( polygonFile->read( &pSize , sizeof(int) ) )
	{
		std::vector< int > polygon( pSize );
		if( polygonFile->read( &polygon[0] , sizeof(int)*pSize ) )
		{
			vertices.resize( pSize );
			for( int i=0 ; i<int(polygon.size()) ; i++ )
				if( polygon[i]<0 ) vertices[i].idx = -polygon[i]-1 , vertices[i].inCore = false;
				else               vertices[i].idx =  polygon[i]   , vertices[i].inCore = true;
			return 1;
		}
		return 0;
	}
	else return 0;
}
template< class Vertex >
int CoredFileMeshData< Vertex >::outOfCorePointCount( void ){ return oocPoints; }
template< class Vertex >
int CoredFileMeshData< Vertex >::polygonCount( void ) { return polygons; }
