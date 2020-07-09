/*
Copyright (c) 2007, Michael Kazhdan
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
//////////////////////////////
// MinimalAreaTriangulation //
//////////////////////////////
template <class Real>
MinimalAreaTriangulation<Real>::MinimalAreaTriangulation(void)
{
	bestTriangulation=NULL;
	midPoint=NULL;
}
template <class Real>
MinimalAreaTriangulation<Real>::~MinimalAreaTriangulation(void)
{
	if(bestTriangulation)
		delete[] bestTriangulation;
	bestTriangulation=NULL;
	if(midPoint)
		delete[] midPoint;
	midPoint=NULL;
}
template <class Real>
void MinimalAreaTriangulation<Real>::GetTriangulation(const std::vector<Point3D<Real> >& vertices,std::vector<TriangleIndex>& triangles)
{
	if(vertices.size()==3)
	{
		triangles.resize(1);
		triangles[0].idx[0]=0;
		triangles[0].idx[1]=1;
		triangles[0].idx[2]=2;
		return;
	}
	else if(vertices.size()==4)
	{
		TriangleIndex tIndex[2][2];
		Real area[2];

		area[0]=area[1]=0;
		triangles.resize(2);

		tIndex[0][0].idx[0]=0;
		tIndex[0][0].idx[1]=1;
		tIndex[0][0].idx[2]=2;
		tIndex[0][1].idx[0]=2;
		tIndex[0][1].idx[1]=3;
		tIndex[0][1].idx[2]=0;

		tIndex[1][0].idx[0]=0;
		tIndex[1][0].idx[1]=1;
		tIndex[1][0].idx[2]=3;
		tIndex[1][1].idx[0]=3;
		tIndex[1][1].idx[1]=1;
		tIndex[1][1].idx[2]=2;

		Point3D<Real> n,p1,p2;
		for(int i=0;i<2;i++)
			for(int j=0;j<2;j++)
			{
				p1=vertices[tIndex[i][j].idx[1]]-vertices[tIndex[i][j].idx[0]];
				p2=vertices[tIndex[i][j].idx[2]]-vertices[tIndex[i][j].idx[0]];
				CrossProduct(p1,p2,n);
				area[i] += Real( Length(n) );
			}
		if(area[0]>area[1])
		{
			triangles[0]=tIndex[1][0];
			triangles[1]=tIndex[1][1];
		}
		else
		{
			triangles[0]=tIndex[0][0];
			triangles[1]=tIndex[0][1];
		}
		return;
	}
	if(bestTriangulation)
		delete[] bestTriangulation;
	if(midPoint)
		delete[] midPoint;
	bestTriangulation=NULL;
	midPoint=NULL;
	size_t eCount=vertices.size();
	bestTriangulation=new Real[eCount*eCount];
	midPoint=new int[eCount*eCount];
	for(size_t i=0;i<eCount*eCount;i++)
		bestTriangulation[i]=-1;
	memset(midPoint,-1,sizeof(int)*eCount*eCount);
	GetArea(0,1,vertices);
	triangles.clear();
	GetTriangulation(0,1,vertices,triangles);
}
template <class Real>
Real MinimalAreaTriangulation<Real>::GetArea(const std::vector<Point3D<Real> >& vertices)
{
	if(bestTriangulation)
		delete[] bestTriangulation;
	if(midPoint)
		delete[] midPoint;
	bestTriangulation=NULL;
	midPoint=NULL;
	int eCount=vertices.size();
	bestTriangulation=new double[eCount*eCount];
	midPoint=new int[eCount*eCount];
	for(int i=0;i<eCount*eCount;i++)
		bestTriangulation[i]=-1;
	memset(midPoint,-1,sizeof(int)*eCount*eCount);
	return GetArea(0,1,vertices);
}
template<class Real>
void MinimalAreaTriangulation<Real>::GetTriangulation(const size_t& i,const size_t& j,const std::vector<Point3D<Real> >& vertices,std::vector<TriangleIndex>& triangles)
{
	TriangleIndex tIndex;
	size_t eCount=vertices.size();
	size_t ii=i;
	if(i<j)
		ii+=eCount;
	if(j+1>=ii)
		return;
	ii=midPoint[i*eCount+j];
	if(ii>=0)
	{
		tIndex.idx[0] = int( i );
		tIndex.idx[1] = int( j );
		tIndex.idx[2] = int( ii );
		triangles.push_back(tIndex);
		GetTriangulation(i,ii,vertices,triangles);
		GetTriangulation(ii,j,vertices,triangles);
	}
}

template<class Real>
Real MinimalAreaTriangulation<Real>::GetArea(const size_t& i,const size_t& j,const std::vector<Point3D<Real> >& vertices)
{
	Real a=FLT_MAX,temp;
	size_t eCount=vertices.size();
	size_t idx=i*eCount+j;
	size_t ii=i;
	if(i<j)
		ii+=eCount;
	if(j+1>=ii)
	{
		bestTriangulation[idx]=0;
		return 0;
	}
	if(midPoint[idx]!=-1)
		return bestTriangulation[idx];
	int mid=-1;
	for(size_t r=j+1;r<ii;r++)
	{
		size_t rr=r%eCount;
		size_t idx1=i*eCount+rr,idx2=rr*eCount+j;
		Point3D<Real> p,p1,p2;
		p1=vertices[i]-vertices[rr];
		p2=vertices[j]-vertices[rr];
		CrossProduct(p1,p2,p);
		temp = Real( Length(p) );
		if(bestTriangulation[idx1]>=0)
		{
			temp+=bestTriangulation[idx1];
			if(temp>a)
				continue;
			if(bestTriangulation[idx2]>0)
				temp+=bestTriangulation[idx2];
			else
				temp+=GetArea(rr,j,vertices);
		}
		else
		{
			if(bestTriangulation[idx2]>=0)
				temp+=bestTriangulation[idx2];
			else
				temp+=GetArea(rr,j,vertices);
			if(temp>a)
				continue;
			temp+=GetArea(i,rr,vertices);
		}

		if(temp<a)
		{
			a=temp;
			mid=int(rr);
		}
	}
	bestTriangulation[idx]=a;
	midPoint[idx]=mid;

	return a;
}
