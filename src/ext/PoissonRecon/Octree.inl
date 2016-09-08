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

#include <stdlib.h>
#include <math.h>
#include <algorithm>

/////////////
// OctNode //
/////////////
template< class NodeData > const int OctNode< NodeData >::DepthShift=5;
template< class NodeData > const int OctNode< NodeData >::OffsetShift = ( sizeof(long long)*8 - DepthShift ) / 3;
template< class NodeData > const int OctNode< NodeData >::DepthMask=(1<<DepthShift)-1;
template< class NodeData > const int OctNode< NodeData >::OffsetMask=(1<<OffsetShift)-1;
template< class NodeData > const int OctNode< NodeData >::OffsetShift1=DepthShift;
template< class NodeData > const int OctNode< NodeData >::OffsetShift2=OffsetShift1+OffsetShift;
template< class NodeData > const int OctNode< NodeData >::OffsetShift3=OffsetShift2+OffsetShift;

template< class NodeData > int OctNode< NodeData >::UseAlloc=0;
template< class NodeData > Allocator<OctNode< NodeData > > OctNode< NodeData >::NodeAllocator;

template< class NodeData >
void OctNode< NodeData >::SetAllocator(int blockSize)
{
	if(blockSize>0)
	{
		UseAlloc=1;
		NodeAllocator.set(blockSize);
	}
	else{UseAlloc=0;}
}
template< class NodeData >
int OctNode< NodeData >::UseAllocator(void){return UseAlloc;}

template< class NodeData >
OctNode< NodeData >::OctNode(void){
	parent=children=NULL;
	_depthAndOffset = 0;
}

template< class NodeData >
OctNode< NodeData >::~OctNode(void){
	if(!UseAlloc){if(children){delete[] children;}}
	parent=children=NULL;
}
template< class NodeData >
void OctNode< NodeData >::setFullDepth( int maxDepth )
{
	if( maxDepth )
	{
		if( !children ) initChildren();
		for( int i=0 ; i<8 ; i++ ) children[i].setFullDepth( maxDepth-1 );
	}
}

template< class NodeData >
int OctNode< NodeData >::initChildren( void )
{
	if( UseAlloc ) children=NodeAllocator.newElements(8);
	else
	{
		if( children ) delete[] children;
		children = NULL;
		children = new OctNode[Cube::CORNERS];
	}
	if( !children )
	{
		fprintf(stderr,"Failed to initialize children in OctNode::initChildren\n");
		exit(0);
		return 0;
	}
	int d , off[3];
	depthAndOffset( d , off );
	for( int i=0 ; i<2 ; i++ ) for( int j=0 ; j<2 ; j++ ) for( int k=0 ; k<2 ; k++ )
	{
		int idx=Cube::CornerIndex(i,j,k);
		children[idx].parent = this;
		children[idx].children = NULL;
		int off2[3];
		off2[0] = (off[0]<<1)+i;
		off2[1] = (off[1]<<1)+j;
		off2[2] = (off[2]<<1)+k;
		children[idx]._depthAndOffset = Index( d+1 , off2 );
	}
	return 1;
}
template< class NodeData >
inline void OctNode< NodeData >::Index(int depth,const int offset[3],short& d,short off[3]){
	d=short(depth);
	off[0]=short((1<<depth)+offset[0]-1);
	off[1]=short((1<<depth)+offset[1]-1);
	off[2]=short((1<<depth)+offset[2]-1);
}

template< class NodeData >
inline void OctNode< NodeData >::depthAndOffset( int& depth , int offset[DIMENSION] ) const
{
	depth = int( _depthAndOffset & DepthMask );
	offset[0] = int( (_depthAndOffset>>OffsetShift1) & OffsetMask );
	offset[1] = int( (_depthAndOffset>>OffsetShift2) & OffsetMask );
	offset[2] = int( (_depthAndOffset>>OffsetShift3) & OffsetMask );
}
template< class NodeData >
inline void OctNode< NodeData >::centerIndex( int index[DIMENSION] ) const
{
	int d , off[DIMENSION];
	depthAndOffset( d , off );
	for( int i=0 ; i<DIMENSION ; i++ ) index[i] = BinaryNode::CenterIndex( d , off[i] );
}
template< class NodeData >
inline unsigned long long OctNode< NodeData >::Index( int depth , const int offset[3] )
{
	unsigned long long idx=0;
	idx |= ( ( (unsigned long long)(depth    ) ) & DepthMask  );
	idx |= ( ( (unsigned long long)(offset[0]) ) & OffsetMask ) << OffsetShift1;
	idx |= ( ( (unsigned long long)(offset[1]) ) & OffsetMask ) << OffsetShift2;
	idx |= ( ( (unsigned long long)(offset[2]) ) & OffsetMask ) << OffsetShift3;
	return idx;
}
template< class NodeData >
inline int OctNode< NodeData >::depth( void ) const {return int( _depthAndOffset & DepthMask );}
template< class NodeData >
inline void OctNode< NodeData >::DepthAndOffset(const long long& index,int& depth,int offset[3]){
	depth=int(index&DepthMask);
	offset[0]=(int((index>>OffsetShift1)&OffsetMask)+1)&(~(1<<depth));
	offset[1]=(int((index>>OffsetShift2)&OffsetMask)+1)&(~(1<<depth));
	offset[2]=(int((index>>OffsetShift3)&OffsetMask)+1)&(~(1<<depth));
}
template< class NodeData >
inline int OctNode< NodeData >::Depth(const long long& index){return int(index&DepthMask);}
template< class NodeData >
template< class Real >
void OctNode< NodeData >::centerAndWidth( Point3D<Real>& center , Real& width ) const
{
	int depth , offset[3];
	depthAndOffset( depth , offset );
	width = Real( 1.0 / (1<<depth) );
	for( int dim=0 ; dim<DIMENSION ; dim++ ) center.coords[dim] = Real( 0.5+offset[dim] ) * width;
}
template< class NodeData >
template< class Real >
void OctNode< NodeData >::startAndWidth( Point3D<Real>& start , Real& width ) const
{
	int depth , offset[3];
	depthAndOffset( depth , offset );
	width = Real( 1.0 / (1<<depth) );
	for( int dim=0 ; dim<DIMENSION ; dim++ ) start.coords[dim] = Real( offset[dim] ) * width;
}
template< class NodeData >
template< class Real >
bool OctNode< NodeData >::isInside( Point3D< Real > p ) const
{
	Point3D< Real > c;
	Real w;
	centerAndWidth( c , w );
	w /= 2;
	return (c[0]-w)<p[0] && p[0]<=(c[0]+w) && (c[1]-w)<p[1] && p[1]<=(c[1]+w) && (c[2]-w)<p[2] && p[2]<=(c[2]+w);
}
template< class NodeData >
template< class Real >
inline void OctNode< NodeData >::CenterAndWidth(const long long& index,Point3D<Real>& center,Real& width){
	int depth,offset[3];
	depth=index&DepthMask;
	offset[0]=(int((index>>OffsetShift1)&OffsetMask)+1)&(~(1<<depth));
	offset[1]=(int((index>>OffsetShift2)&OffsetMask)+1)&(~(1<<depth));
	offset[2]=(int((index>>OffsetShift3)&OffsetMask)+1)&(~(1<<depth));
	width=Real(1.0/(1<<depth));
	for(int dim=0;dim<DIMENSION;dim++){center.coords[dim]=Real(0.5+offset[dim])*width;}
}
template< class NodeData >
template< class Real >
inline void OctNode< NodeData >::StartAndWidth( const long long& index , Point3D< Real >& start , Real& width )
{
	int depth,offset[3];
	depth = index&DepthMask;
	offset[0] = (int((index>>OffsetShift1)&OffsetMask)+1)&(~(1<<depth));
	offset[1] = (int((index>>OffsetShift2)&OffsetMask)+1)&(~(1<<depth));
	offset[2] = (int((index>>OffsetShift3)&OffsetMask)+1)&(~(1<<depth));
	width = Real(1.0/(1<<depth));
	for( int dim=0 ; dim<DIMENSION ; dim++ ) start.coords[dim] = Real(offset[dim])*width;
}

template< class NodeData >
int OctNode< NodeData >::maxDepth(void) const{
	if(!children){return 0;}
	else{
		int c,d;
		for(int i=0;i<Cube::CORNERS;i++){
			d=children[i].maxDepth();
			if(!i || d>c){c=d;}
		}
		return c+1;
	}
}
template< class NodeData >
size_t OctNode< NodeData >::nodes( void ) const
{
	if( !children ) return 1;
	else
	{
		size_t c=0;
		for( int i=0 ; i<Cube::CORNERS ; i++ ) c += children[i].nodes();
		return c+1;
	}
}
template< class NodeData >
size_t OctNode< NodeData >::leaves( void ) const
{
	if( !children ) return 1;
	else
	{
		size_t c=0;
		for( int i=0 ; i<Cube::CORNERS ; i++ ) c += children[i].leaves();
		return c;
	}
}
template< class NodeData >
size_t OctNode< NodeData >::maxDepthLeaves( int maxDepth ) const
{
	if( depth()>maxDepth ) return 0;
	if( !children ) return 1;
	else
	{
		size_t c=0;
		for( int i=0 ; i<Cube::CORNERS ; i++ ) c += children[i].maxDepthLeaves(maxDepth);
		return c;
	}
}
template< class NodeData >
const OctNode< NodeData >* OctNode< NodeData >::root(void) const{
	const OctNode* temp=this;
	while(temp->parent){temp=temp->parent;}
	return temp;
}


template< class NodeData >
const OctNode< NodeData >* OctNode< NodeData >::nextBranch( const OctNode* current ) const
{
	if( !current->parent || current==this ) return NULL;
	if(current-current->parent->children==Cube::CORNERS-1) return nextBranch( current->parent );
	else return current+1;
}
template< class NodeData >
OctNode< NodeData >* OctNode< NodeData >::nextBranch(OctNode* current){
	if(!current->parent || current==this){return NULL;}
	if(current-current->parent->children==Cube::CORNERS-1){return nextBranch(current->parent);}
	else{return current+1;}
}
template< class NodeData >
const OctNode< NodeData >* OctNode< NodeData >::prevBranch( const OctNode* current ) const
{
	if( !current->parent || current==this ) return NULL;
	if( current-current->parent->children==0 ) return prevBranch( current->parent );
	else return current-1;
}
template< class NodeData >
OctNode< NodeData >* OctNode< NodeData >::prevBranch( OctNode* current )
{
	if( !current->parent || current==this ) return NULL;
	if( current-current->parent->children==0 ) return prevBranch( current->parent );
	else return current-1;
}
template< class NodeData >
const OctNode< NodeData >* OctNode< NodeData >::nextLeaf(const OctNode* current) const{
	if(!current){
		const OctNode< NodeData >* temp=this;
		while(temp->children){temp=&temp->children[0];}
		return temp;
	}
	if(current->children){return current->nextLeaf();}
	const OctNode* temp=nextBranch(current);
	if(!temp){return NULL;}
	else{return temp->nextLeaf();}
}
template< class NodeData >
OctNode< NodeData >* OctNode< NodeData >::nextLeaf(OctNode* current){
	if(!current){
		OctNode< NodeData >* temp=this;
		while(temp->children){temp=&temp->children[0];}
		return temp;
	}
	if(current->children){return current->nextLeaf();}
	OctNode* temp=nextBranch(current);
	if(!temp){return NULL;}
	else{return temp->nextLeaf();}
}

template< class NodeData >
const OctNode< NodeData >* OctNode< NodeData >::nextNode( const OctNode* current ) const
{
	if( !current ) return this;
	else if( current->children ) return &current->children[0];
	else return nextBranch(current);
}
template< class NodeData >
OctNode< NodeData >* OctNode< NodeData >::nextNode( OctNode* current )
{
	if( !current ) return this;
	else if( current->children ) return &current->children[0];
	else return nextBranch( current );
}

template< class NodeData >
void OctNode< NodeData >::printRange(void) const
{
	Point3D< float > center;
	float width;
	centerAndWidth(center,width);
	for(int dim=0;dim<DIMENSION;dim++){
		printf("%[%f,%f]",center.coords[dim]-width/2,center.coords[dim]+width/2);
		if(dim<DIMENSION-1){printf("x");}
		else printf("\n");
	}
}

template< class NodeData >
template< class Real >
int OctNode< NodeData >::CornerIndex(const Point3D<Real>& center,const Point3D<Real>& p){
	int cIndex=0;
	if(p.coords[0]>center.coords[0]){cIndex|=1;}
	if(p.coords[1]>center.coords[1]){cIndex|=2;}
	if(p.coords[2]>center.coords[2]){cIndex|=4;}
	return cIndex;
}

template< class NodeData >
OctNode< NodeData >* OctNode< NodeData >::faceNeighbor(int faceIndex,int forceChildren){return __faceNeighbor(faceIndex>>1,faceIndex&1,forceChildren);}
template< class NodeData >
const OctNode< NodeData >* OctNode< NodeData >::faceNeighbor(int faceIndex) const {return __faceNeighbor(faceIndex>>1,faceIndex&1);}
template< class NodeData >
OctNode< NodeData >* OctNode< NodeData >::__faceNeighbor(int dir,int off,int forceChildren){
	if(!parent){return NULL;}
	int pIndex=int(this-parent->children);
	pIndex^=(1<<dir);
	if((pIndex & (1<<dir))==(off<<dir)){return &parent->children[pIndex];}
	else{
		OctNode* temp=parent->__faceNeighbor(dir,off,forceChildren);
		if(!temp){return NULL;}
		if(!temp->children){
			if(forceChildren){temp->initChildren();}
			else{return temp;}
		}
		return &temp->children[pIndex];
	}
}
template< class NodeData >
const OctNode< NodeData >* OctNode< NodeData >::__faceNeighbor(int dir,int off) const {
	if(!parent){return NULL;}
	int pIndex=int(this-parent->children);
	pIndex^=(1<<dir);
	if((pIndex & (1<<dir))==(off<<dir)){return &parent->children[pIndex];}
	else{
		const OctNode* temp=parent->__faceNeighbor(dir,off);
		if(!temp || !temp->children){return temp;}
		else{return &temp->children[pIndex];}
	}
}

template< class NodeData >
OctNode< NodeData >* OctNode< NodeData >::edgeNeighbor(int edgeIndex,int forceChildren){
	int idx[2],o,i[2];
	Cube::FactorEdgeIndex(edgeIndex,o,i[0],i[1]);
	switch(o){
		case 0:	idx[0]=1;	idx[1]=2;	break;
		case 1:	idx[0]=0;	idx[1]=2;	break;
		case 2:	idx[0]=0;	idx[1]=1;	break;
	};
	return __edgeNeighbor(o,i,idx,forceChildren);
}
template< class NodeData >
const OctNode< NodeData >* OctNode< NodeData >::edgeNeighbor(int edgeIndex) const {
	int idx[2],o,i[2];
	Cube::FactorEdgeIndex(edgeIndex,o,i[0],i[1]);
	switch(o){
		case 0:	idx[0]=1;	idx[1]=2;	break;
		case 1:	idx[0]=0;	idx[1]=2;	break;
		case 2:	idx[0]=0;	idx[1]=1;	break;
	};
	return __edgeNeighbor(o,i,idx);
}
template< class NodeData >
const OctNode< NodeData >* OctNode< NodeData >::__edgeNeighbor(int o,const int i[2],const int idx[2]) const{
	if(!parent){return NULL;}
	int pIndex=int(this-parent->children);
	int aIndex,x[DIMENSION];

	Cube::FactorCornerIndex(pIndex,x[0],x[1],x[2]);
	aIndex=(~((i[0] ^ x[idx[0]]) | ((i[1] ^ x[idx[1]])<<1))) & 3;
	pIndex^=(7 ^ (1<<o));
	if(aIndex==1)	{	// I can get the neighbor from the parent's face adjacent neighbor
		const OctNode* temp=parent->__faceNeighbor(idx[0],i[0]);
		if(!temp || !temp->children){return NULL;}
		else{return &temp->children[pIndex];}
	}
	else if(aIndex==2)	{	// I can get the neighbor from the parent's face adjacent neighbor
		const OctNode* temp=parent->__faceNeighbor(idx[1],i[1]);
		if(!temp || !temp->children){return NULL;}
		else{return &temp->children[pIndex];}
	}
	else if(aIndex==0)	{	// I can get the neighbor from the parent
		return &parent->children[pIndex];
	}
	else if(aIndex==3)	{	// I can get the neighbor from the parent's edge adjacent neighbor
		const OctNode* temp=parent->__edgeNeighbor(o,i,idx);
		if(!temp || !temp->children){return temp;}
		else{return &temp->children[pIndex];}
	}
	else{return NULL;}
}
template< class NodeData >
OctNode< NodeData >* OctNode< NodeData >::__edgeNeighbor(int o,const int i[2],const int idx[2],int forceChildren){
	if(!parent){return NULL;}
	int pIndex=int(this-parent->children);
	int aIndex,x[DIMENSION];

	Cube::FactorCornerIndex(pIndex,x[0],x[1],x[2]);
	aIndex=(~((i[0] ^ x[idx[0]]) | ((i[1] ^ x[idx[1]])<<1))) & 3;
	pIndex^=(7 ^ (1<<o));
	if(aIndex==1)	{	// I can get the neighbor from the parent's face adjacent neighbor
		OctNode* temp=parent->__faceNeighbor(idx[0],i[0],0);
		if(!temp || !temp->children){return NULL;}
		else{return &temp->children[pIndex];}
	}
	else if(aIndex==2)	{	// I can get the neighbor from the parent's face adjacent neighbor
		OctNode* temp=parent->__faceNeighbor(idx[1],i[1],0);
		if(!temp || !temp->children){return NULL;}
		else{return &temp->children[pIndex];}
	}
	else if(aIndex==0)	{	// I can get the neighbor from the parent
		return &parent->children[pIndex];
	}
	else if(aIndex==3)	{	// I can get the neighbor from the parent's edge adjacent neighbor
		OctNode* temp=parent->__edgeNeighbor(o,i,idx,forceChildren);
		if(!temp){return NULL;}
		if(!temp->children){
			if(forceChildren){temp->initChildren();}
			else{return temp;}
		}
		return &temp->children[pIndex];
	}
	else{return NULL;}
}

template< class NodeData >
const OctNode< NodeData >* OctNode< NodeData >::cornerNeighbor(int cornerIndex) const {
	int pIndex,aIndex=0;
	if(!parent){return NULL;}

	pIndex=int(this-parent->children);
	aIndex=(cornerIndex ^ pIndex);	// The disagreement bits
	pIndex=(~pIndex)&7;				// The antipodal point
	if(aIndex==7){					// Agree on no bits
		return &parent->children[pIndex];
	}
	else if(aIndex==0){				// Agree on all bits
		const OctNode* temp=((const OctNode*)parent)->cornerNeighbor(cornerIndex);
		if(!temp || !temp->children){return temp;}
		else{return &temp->children[pIndex];}
	}
	else if(aIndex==6){				// Agree on face 0
		const OctNode* temp=((const OctNode*)parent)->__faceNeighbor(0,cornerIndex & 1);
		if(!temp || !temp->children){return NULL;}
		else{return & temp->children[pIndex];}
	}
	else if(aIndex==5){				// Agree on face 1
		const OctNode* temp=((const OctNode*)parent)->__faceNeighbor(1,(cornerIndex & 2)>>1);
		if(!temp || !temp->children){return NULL;}
		else{return & temp->children[pIndex];}
	}
	else if(aIndex==3){				// Agree on face 2
		const OctNode* temp=((const OctNode*)parent)->__faceNeighbor(2,(cornerIndex & 4)>>2);
		if(!temp || !temp->children){return NULL;}
		else{return & temp->children[pIndex];}
	}
	else if(aIndex==4){				// Agree on edge 2
		const OctNode* temp=((const OctNode*)parent)->edgeNeighbor(8 | (cornerIndex & 1) | (cornerIndex & 2) );
		if(!temp || !temp->children){return NULL;}
		else{return & temp->children[pIndex];}
	}
	else if(aIndex==2){				// Agree on edge 1
		const OctNode* temp=((const OctNode*)parent)->edgeNeighbor(4 | (cornerIndex & 1) | ((cornerIndex & 4)>>1) );
		if(!temp || !temp->children){return NULL;}
		else{return & temp->children[pIndex];}
	}
	else if(aIndex==1){				// Agree on edge 0
		const OctNode* temp=((const OctNode*)parent)->edgeNeighbor(((cornerIndex & 2) | (cornerIndex & 4))>>1 );
		if(!temp || !temp->children){return NULL;}
		else{return & temp->children[pIndex];}
	}
	else{return NULL;}
}
template< class NodeData >
OctNode< NodeData >* OctNode< NodeData >::cornerNeighbor(int cornerIndex,int forceChildren){
	int pIndex,aIndex=0;
	if(!parent){return NULL;}

	pIndex=int(this-parent->children);
	aIndex=(cornerIndex ^ pIndex);	// The disagreement bits
	pIndex=(~pIndex)&7;				// The antipodal point
	if(aIndex==7){					// Agree on no bits
		return &parent->children[pIndex];
	}
	else if(aIndex==0){				// Agree on all bits
		OctNode* temp=((OctNode*)parent)->cornerNeighbor(cornerIndex,forceChildren);
		if(!temp){return NULL;}
		if(!temp->children){
			if(forceChildren){temp->initChildren();}
			else{return temp;}
		}
		return &temp->children[pIndex];
	}
	else if(aIndex==6){				// Agree on face 0
		OctNode* temp=((OctNode*)parent)->__faceNeighbor(0,cornerIndex & 1,0);
		if(!temp || !temp->children){return NULL;}
		else{return & temp->children[pIndex];}
	}
	else if(aIndex==5){				// Agree on face 1
		OctNode* temp=((OctNode*)parent)->__faceNeighbor(1,(cornerIndex & 2)>>1,0);
		if(!temp || !temp->children){return NULL;}
		else{return & temp->children[pIndex];}
	}
	else if(aIndex==3){				// Agree on face 2
		OctNode* temp=((OctNode*)parent)->__faceNeighbor(2,(cornerIndex & 4)>>2,0);
		if(!temp || !temp->children){return NULL;}
		else{return & temp->children[pIndex];}
	}
	else if(aIndex==4){				// Agree on edge 2
		OctNode* temp=((OctNode*)parent)->edgeNeighbor(8 | (cornerIndex & 1) | (cornerIndex & 2) );
		if(!temp || !temp->children){return NULL;}
		else{return & temp->children[pIndex];}
	}
	else if(aIndex==2){				// Agree on edge 1
		OctNode* temp=((OctNode*)parent)->edgeNeighbor(4 | (cornerIndex & 1) | ((cornerIndex & 4)>>1) );
		if(!temp || !temp->children){return NULL;}
		else{return & temp->children[pIndex];}
	}
	else if(aIndex==1){				// Agree on edge 0
		OctNode* temp=((OctNode*)parent)->edgeNeighbor(((cornerIndex & 2) | (cornerIndex & 4))>>1 );
		if(!temp || !temp->children){return NULL;}
		else{return & temp->children[pIndex];}
	}
	else{return NULL;}
}

////////////////////////
// OctNode::Neighbors //
////////////////////////
template< class NodeData >
template< unsigned int Width >
OctNode< NodeData >::Neighbors< Width >::Neighbors( void ){ clear(); }
template< class NodeData >
template< unsigned int Width >
void OctNode< NodeData >::Neighbors< Width >::clear( void ){ for( int i=0 ; i<Width ; i++ ) for( int j=0 ; j<Width ; j++ ) for( int k=0 ; k<Width ; k++ ) neighbors[i][j][k]=NULL; }

/////////////////////////////
// OctNode::ConstNeighbors //
/////////////////////////////
template< class NodeData >
template< unsigned int Width >
OctNode< NodeData >::ConstNeighbors< Width >::ConstNeighbors( void ){ clear(); }
template< class NodeData >
template< unsigned int Width >
void OctNode< NodeData >::ConstNeighbors< Width >::clear( void ){ for( int i=0 ; i<Width ; i++ ) for( int j=0 ; j<Width ; j++ ) for( int k=0 ; k<Width ; k++ ) neighbors[i][j][k]=NULL; }

//////////////////////////
// OctNode::NeighborKey //
//////////////////////////
template< class NodeData >
template< unsigned int LeftRadius , unsigned int RightRadius >
OctNode< NodeData >::NeighborKey< LeftRadius , RightRadius >::NeighborKey( void ){ _depth=-1 , neighbors=NULL; }
template< class NodeData >
template< unsigned int LeftRadius , unsigned int RightRadius >
OctNode< NodeData >::NeighborKey< LeftRadius , RightRadius >::NeighborKey( const NeighborKey& nKey )
{
	_depth = 0 , neighbors = NULL;
	set( nKey._depth );
	for( int d=0 ; d<=_depth ; d++ ) memcpy( &neighbors[d] , &nKey.neighbors[d] , sizeof( Neighbors< Width > ) );
}
template< class NodeData >
template< unsigned int LeftRadius , unsigned int RightRadius >
OctNode< NodeData >::NeighborKey< LeftRadius , RightRadius >::~NeighborKey( void )
{
	if( neighbors ) delete[] neighbors;
	neighbors = NULL;
}

template< class NodeData >
template< unsigned int LeftRadius , unsigned int RightRadius >
void OctNode< NodeData >::NeighborKey< LeftRadius , RightRadius >::set( int d )
{
	if( neighbors ) delete[] neighbors;
	neighbors = NULL;
	_depth = d;
	if( d<0 ) return;
	neighbors = new Neighbors< Width >[d+1];
}
template< class NodeData >
template< unsigned int LeftRadius , unsigned int RightRadius >
template< bool CreateNodes >
bool OctNode< NodeData >::NeighborKey< LeftRadius , RightRadius >::getChildNeighbors( int cIdx , int d , Neighbors< Width >& cNeighbors ) const
{
	Neighbors< Width >& pNeighbors = neighbors[d];
	// Check that we actuall have a center node
	if( !pNeighbors.neighbors[LeftRadius][LeftRadius][LeftRadius] ) return false;
	
	// Get the indices of the child node that would contain the point (and its antipode)
	int cx , cy , cz;
	Cube::FactorCornerIndex( cIdx , cx , cy , cz );


	// Iterate over the finer neighbors and set them (if you can)
	// Here:
	// (x,y,z) give the position of the finer nodes relative to the center,
	// (_x,_y,_z) give a positive global position, up to an even offset, and
	// (px-LeftRadius,py-LeftRadius,pz-LeftRadius) give the positions of their parents relative to the parent of the center
	for( int z=-(int)LeftRadius ; z<=(int)RightRadius ; z++ )
	{
		int _z = (z+cz) + (LeftRadius<<1) , pz = ( _z>>1 ) , zz = z+LeftRadius;
		for( int y=-(int)LeftRadius ; y<=(int)RightRadius ; y++ )
		{
			int _y = (y+cy) + (LeftRadius<<1) , py = ( _y>>1 ) , yy = y+LeftRadius;

			int cornerIndex = ( (_z&1)<<2 ) | ( (_y&1)<<1 );
			for( int x=-(int)LeftRadius ; x<=(int)RightRadius ; x++ )
			{
				int _x = (x+cx) + (LeftRadius<<1) , px = ( _x>>1 ) , xx = x+LeftRadius;

				if( CreateNodes )
				{
					if( pNeighbors.neighbors[px][py][pz] )
					{
						if( !pNeighbors.neighbors[px][py][pz]->children ) pNeighbors.neighbors[px][py][pz]->initChildren();
						cNeighbors.neighbors[xx][yy][zz] = pNeighbors.neighbors[px][py][pz]->children + ( cornerIndex | (_x&1) );
					}
					else cNeighbors.neighbors[xx][yy][zz] = NULL;
				}
				else
				{
					if( pNeighbors.neighbors[px][py][pz] && pNeighbors.neighbors[px][py][pz]->children )
						cNeighbors.neighbors[xx][yy][zz] = pNeighbors.neighbors[px][py][pz]->children + ( cornerIndex | (_x&1) );
					else cNeighbors.neighbors[xx][yy][zz] = NULL;
				}
			}
		}
	}
	return true;
}
template< class NodeData >
template< unsigned int LeftRadius , unsigned int RightRadius >
template< bool CreateNodes , class Real >
bool OctNode< NodeData >::NeighborKey< LeftRadius , RightRadius >::getChildNeighbors( Point3D< Real > p , int d , Neighbors< Width >& cNeighbors ) const
{
	Neighbors< Width >& pNeighbors = neighbors[d];
	// Check that we actuall have a center node
	if( !pNeighbors.neighbors[LeftRadius][LeftRadius][LeftRadius] ) return false;
	Point3D< Real > c;
	Real w;
	pNeighbors.neighbors[LeftRadius][LeftRadius][LeftRadius]->centerAndWidth( c , w );
	return getChildNeighbors< CreateNodes >( CornerIndex( c , p ) , d , cNeighbors );
}

template< class NodeData >
template< unsigned int LeftRadius , unsigned int RightRadius >
template< bool CreateNodes >
typename OctNode< NodeData >::template Neighbors< LeftRadius+RightRadius+1 >& OctNode< NodeData >::NeighborKey< LeftRadius , RightRadius >::getNeighbors( OctNode< NodeData >* node )
{
	Neighbors< Width >& neighbors = this->neighbors[ node->depth() ];
	if( node==neighbors.neighbors[LeftRadius][LeftRadius][LeftRadius] )
	{
		bool reset = false;
		for( int i=0 ; i<Width ; i++ ) for( int j=0 ; j<Width ; j++ ) for( int k=0 ; k<Width ; k++ ) if( !neighbors.neighbors[i][j][k] ) reset = true;
		if( reset ) neighbors.neighbors[LeftRadius][LeftRadius][LeftRadius] = NULL;
	}
	if( node!=neighbors.neighbors[LeftRadius][LeftRadius][LeftRadius] )
	{
		neighbors.clear();

		if( !node->parent ) neighbors.neighbors[LeftRadius][LeftRadius][LeftRadius] = node;
		else
		{
			Neighbors< Width >& pNeighbors = getNeighbors< CreateNodes >( node->parent );


			// Get the indices of the child node that would contain the point (and its antipode)
			int cx , cy , cz;
			Cube::FactorCornerIndex( (int)( node - node->parent->children ) , cx , cy , cz );


			// Iterate over the finer neighbors and set them (if you can)
			// Here:
			// (x,y,z) give the position of the finer nodes relative to the center,
			// (_x,_y,_z) give a positive global position, up to an even offset, and
			// (px-LeftRadius,py-LeftRadius,pz-LeftRadius) give the positions of their parents relative to the parent of the center
			for( int z=-(int)LeftRadius ; z<=(int)RightRadius ; z++ )
			{
				int _z = (z+cz) + (LeftRadius<<1) , pz = ( _z>>1 ) , zz = z+LeftRadius;
				for( int y=-(int)LeftRadius ; y<=(int)RightRadius ; y++ )
				{
					int _y = (y+cy) + (LeftRadius<<1) , py = ( _y>>1 ) , yy = y+LeftRadius;

					int cornerIndex = ( (_z&1)<<2 ) | ( (_y&1)<<1 );
					for( int x=-(int)LeftRadius ; x<=(int)RightRadius ; x++ )
					{
						int _x = (x+cx) + (LeftRadius<<1) , px = ( _x>>1 ) , xx = x+LeftRadius;
						if( CreateNodes )
						{
							if( pNeighbors.neighbors[px][py][pz] )
							{
								if( !pNeighbors.neighbors[px][py][pz]->children ) pNeighbors.neighbors[px][py][pz]->initChildren();
								neighbors.neighbors[xx][yy][zz] = pNeighbors.neighbors[px][py][pz]->children + ( cornerIndex | (_x&1) );
							}
							else neighbors.neighbors[xx][yy][zz] = NULL;
						}
						else
						{
							if( pNeighbors.neighbors[px][py][pz] && pNeighbors.neighbors[px][py][pz]->children )
								neighbors.neighbors[xx][yy][zz] = pNeighbors.neighbors[px][py][pz]->children + ( cornerIndex | (_x&1) );
							else neighbors.neighbors[xx][yy][zz] = NULL;
						}
					}
				}
			}
		}
	}
	return neighbors;
}
template< class NodeData >
template< unsigned int LeftRadius , unsigned int RightRadius >
template< bool CreateNodes , unsigned int _LeftRadius , unsigned int _RightRadius >
void OctNode< NodeData >::NeighborKey< LeftRadius , RightRadius >::getNeighbors( OctNode< NodeData >* node , Neighbors< _LeftRadius + _RightRadius + 1 >& neighbors )
{
	neighbors.clear();
	if( !node ) return;

	// [WARNING] This estimate of the required radius is somewhat conservative if the radius is odd (depending on where the node is relative to its parent)
	const unsigned int _PLeftRadius = (_LeftRadius+1)/2 , _PRightRadius = (_RightRadius+1)/2;
	// If we are at the root of the tree, we are done
	if( !node->parent ) neighbors.neighbors[_LeftRadius][_LeftRadius][_LeftRadius] = node;
	// If we can get the data from the the key for the parent node, do that
	else if( _PLeftRadius<=LeftRadius && _PRightRadius<=RightRadius )
	{
		getNeighbors< CreateNodes >( node->parent );
		const Neighbors< LeftRadius + RightRadius + 1 >& pNeighbors = this->neighbors[ node->depth()-1 ];
		// Get the indices of the child node that would contain the point (and its antipode)
		int cx , cy , cz;
		Cube::FactorCornerIndex( (int)( node - node->parent->children ) , cx , cy , cz );


		// Iterate over the finer neighbors
		// Here:
		// (x,y,z) give the position of the finer nodes relative to the center,
		// (_x,_y,_z) give a positive global position, up to an even offset, and
		// (px-LeftRadius,py-LeftRadius,pz-LeftRadius) give the positions of their parents relative to the parent of the center
		for( int z=-(int)_LeftRadius ; z<=(int)_RightRadius ; z++ )
		{
			int _z = (z+cz) + (_LeftRadius<<1) , pz = ( _z>>1 ) - _LeftRadius + LeftRadius , zz = z + _LeftRadius;
			for( int y=-(int)_LeftRadius ; y<=(int)_RightRadius ; y++ )
			{
				int _y = (y+cy) + (_LeftRadius<<1) , py = ( _y>>1 ) - _LeftRadius + LeftRadius , yy = y + _LeftRadius;

				int cornerIndex = ( (_z&1)<<2 ) | ( (_y&1)<<1 );
				for( int x=-(int)_LeftRadius ; x<=(int)_RightRadius ; x++ )
				{
					int _x = (x+cx) + (_LeftRadius<<1) , px = ( _x>>1 ) - _LeftRadius + LeftRadius , xx = x + _LeftRadius;
					if( CreateNodes )
					{
						if( pNeighbors.neighbors[px][py][pz] )
						{
							if( !pNeighbors.neighbors[px][py][pz]->children ) pNeighbors.neighbors[px][py][pz]->initChildren();
							neighbors.neighbors[xx][yy][zz] = pNeighbors.neighbors[px][py][pz]->children + ( cornerIndex | (_x&1) );
						}
						else neighbors.neighbors[xx][yy][zz] = NULL;
					}
					else
					{
						if( pNeighbors.neighbors[px][py][pz] && pNeighbors.neighbors[px][py][pz]->children )
							neighbors.neighbors[xx][yy][zz] = pNeighbors.neighbors[px][py][pz]->children + ( cornerIndex | (_x&1) );
						else neighbors.neighbors[xx][yy][zz] = NULL;
					}
				}
			}
		}
	}
	// Otherwise recurse
	else
	{
		Neighbors< _PLeftRadius + _PRightRadius + 1 > pNeighbors;
		getNeighbors< CreateNodes , _PLeftRadius , _PRightRadius >( node->parent , pNeighbors );

		// Get the indices of the child node that would contain the point (and its antipode)
		int cx , cy , cz;
		Cube::FactorCornerIndex( (int)( node - node->parent->children ) , cx , cy , cz );


		// Iterate over the finer neighbors
		// Here:
		// (x,y,z) give the position of the finer nodes relative to the center,
		// (_x,_y,_z) give a positive global position, up to an even offset, and
		// (px-LeftRadius,py-LeftRadius,pz-LeftRadius) give the positions of their parents relative to the parent of the center
		for( int z=-(int)_LeftRadius ; z<=(int)_RightRadius ; z++ )
		{
			int _z = (z+cz) + (_LeftRadius<<1) , pz = ( _z>>1 ) - _LeftRadius + _PLeftRadius , zz = z + _LeftRadius;
			for( int y=-(int)_LeftRadius ; y<=(int)_RightRadius ; y++ )
			{
				int _y = (y+cy) + (_LeftRadius<<1) , py = ( _y>>1 ) - _LeftRadius + _PLeftRadius , yy = y + _LeftRadius;

				int cornerIndex = ( (_z&1)<<2 ) | ( (_y&1)<<1 );
				for( int x=-(int)_LeftRadius ; x<=(int)_RightRadius ; x++ )
				{
					int _x = (x+cx) + (_LeftRadius<<1) , px = ( _x>>1 ) - _LeftRadius + _PLeftRadius , xx = x + _LeftRadius;
					if( CreateNodes )
					{
						if( pNeighbors.neighbors[px][py][pz] )
						{
							if( !pNeighbors.neighbors[px][py][pz]->children ) pNeighbors.neighbors[px][py][pz]->initChildren();
							neighbors.neighbors[xx][yy][zz] = pNeighbors.neighbors[px][py][pz]->children + ( cornerIndex | (_x&1) );
						}
						else neighbors.neighbors[xx][yy][zz] = NULL;
					}
					else
					{
						if( pNeighbors.neighbors[px][py][pz] && pNeighbors.neighbors[px][py][pz]->children )
							neighbors.neighbors[xx][yy][zz] = pNeighbors.neighbors[px][py][pz]->children + ( cornerIndex | (_x&1) );
						else neighbors.neighbors[xx][yy][zz] = NULL;
					}
				}
			}
		}
	}
}

///////////////////////////////
// OctNode::ConstNeighborKey //
///////////////////////////////
template< class NodeData >
template< unsigned int LeftRadius , unsigned int RightRadius >
OctNode< NodeData >::ConstNeighborKey< LeftRadius , RightRadius >::ConstNeighborKey( void ){ _depth=-1 , neighbors=NULL; }
template< class NodeData >
template< unsigned int LeftRadius , unsigned int RightRadius >
OctNode< NodeData >::ConstNeighborKey< LeftRadius , RightRadius >::ConstNeighborKey( const ConstNeighborKey& key )
{
	_depth = 0 , neighbors = NULL;
	set( key._depth );
	for( int d=0 ; d<=_depth ; d++ ) memcpy( &neighbors[d] , &key.neighbors[d] , sizeof( ConstNeighbors< Width > ) );
}
template< class NodeData >
template< unsigned int LeftRadius , unsigned int RightRadius >
OctNode< NodeData >::ConstNeighborKey< LeftRadius , RightRadius >::~ConstNeighborKey( void )
{
	if( neighbors ) delete[] neighbors;
	neighbors=NULL;
}

template< class NodeData >
template< unsigned int LeftRadius , unsigned int RightRadius >
void OctNode< NodeData >::ConstNeighborKey< LeftRadius , RightRadius >::set( int d )
{
	if( neighbors ) delete[] neighbors;
	neighbors = NULL;
	_depth = d;
	if( d<0 ) return;
	neighbors = new ConstNeighbors< Width >[d+1];
}
template< class NodeData >
template< unsigned int LeftRadius , unsigned int RightRadius >
typename OctNode< NodeData >::template ConstNeighbors< LeftRadius+RightRadius+1 >& OctNode< NodeData >::ConstNeighborKey< LeftRadius , RightRadius >::getNeighbors( const OctNode< NodeData >* node )
{
	ConstNeighbors< Width >& neighbors = this->neighbors[ node->depth() ];
	if( node!=neighbors.neighbors[LeftRadius][LeftRadius][LeftRadius])
	{
		neighbors.clear();

		if( !node->parent ) neighbors.neighbors[LeftRadius][LeftRadius][LeftRadius] = node;
		else
		{
			ConstNeighbors< Width >& pNeighbors = getNeighbors( node->parent );

			// Get the indices of the child node that would contain the point (and its antipode)
			int cx , cy , cz;
			Cube::FactorCornerIndex( (int)( node - node->parent->children ) , cx , cy , cz );


			// Iterate over the finer neighbors and set them (if you can)
			// Here:
			// (x,y,z) give the position of the finer nodes relative to the center,
			// (_x,_y,_z) give a positive global position, up to an even offset, and
			// (px-LeftRadius,py-LeftRadius,pz-LeftRadius) give the positions of their parents relative to the parent of the center
			for( int z=-(int)LeftRadius ; z<=(int)RightRadius ; z++ )
			{
				int _z = (z+cz) + (LeftRadius<<1) , pz = ( _z>>1 ) , zz = z+LeftRadius;
				for( int y=-(int)LeftRadius ; y<=(int)RightRadius ; y++ )
				{
					int _y = (y+cy) + (LeftRadius<<1) , py = ( _y>>1 ) , yy = y+LeftRadius;

					int cornerIndex = ( (_z&1)<<2 ) | ( (_y&1)<<1 );
					for( int x=-(int)LeftRadius ; x<=(int)RightRadius ; x++ )
					{
						int _x = (x+cx) + (LeftRadius<<1) , px = ( _x>>1 ) , xx = x+LeftRadius;
						if( pNeighbors.neighbors[px][py][pz] && pNeighbors.neighbors[px][py][pz]->children )
							neighbors.neighbors[xx][yy][zz] = pNeighbors.neighbors[px][py][pz]->children + ( cornerIndex | (_x&1) );
						else
							neighbors.neighbors[xx][yy][zz] = NULL;
					}
				}
			}
		}
	}
	return neighbors;
}
template< class NodeData >
template< unsigned int LeftRadius , unsigned int RightRadius >
template< unsigned int _LeftRadius , unsigned int _RightRadius >
void OctNode< NodeData >::ConstNeighborKey< LeftRadius , RightRadius >::getNeighbors( const OctNode< NodeData >* node , ConstNeighbors< _LeftRadius+_RightRadius+1 >& neighbors )
{
	neighbors.clear();
	if( !node ) return;

	// [WARNING] This estimate of the required radius is somewhat conservative if the readius is odd (depending on where the node is relative to its parent)
	const unsigned int _PLeftRadius = (_LeftRadius+1)/2 , _PRightRadius = (_RightRadius+1)/2;
	// If we are at the root of the tree, we are done
	if( !node->parent ) neighbors.neighbors[_LeftRadius][_LeftRadius][_LeftRadius] = node;
	// If we can get the data from the the key for the parent node, do that
	else if( _PLeftRadius<=LeftRadius && _PRightRadius<=RightRadius )
	{
		getNeighbors( node->parent );
		const ConstNeighbors< LeftRadius + RightRadius + 1 >& pNeighbors = this->neighbors[ node->depth()-1 ];
		// Get the indices of the child node that would contain the point (and its antipode)
		int cx , cy , cz;
		Cube::FactorCornerIndex( (int)( node - node->parent->children ) , cx , cy , cz );


		// Iterate over the finer neighbors
		// Here:
		// (x,y,z) give the position of the finer nodes relative to the center,
		// (_x,_y,_z) give a positive global position, up to an even offset, and
		// (px-LeftRadius,py-LeftRadius,pz-LeftRadius) give the positions of their parents relative to the parent of the center
		for( int z=-(int)_LeftRadius ; z<=(int)_RightRadius ; z++ )
		{
			int _z = (z+cz) + (_LeftRadius<<1) , pz = ( _z>>1 ) - _LeftRadius + LeftRadius , zz = z + _LeftRadius;
			for( int y=-(int)_LeftRadius ; y<=(int)_RightRadius ; y++ )
			{
				int _y = (y+cy) + (_LeftRadius<<1) , py = ( _y>>1 ) - _LeftRadius + LeftRadius , yy = y + _LeftRadius;

				int cornerIndex = ( (_z&1)<<2 ) | ( (_y&1)<<1 );
				for( int x=-(int)_LeftRadius ; x<=(int)_RightRadius ; x++ )
				{
					int _x = (x+cx) + (_LeftRadius<<1) , px = ( _x>>1 ) - _LeftRadius + LeftRadius , xx = x + _LeftRadius;
					if( pNeighbors.neighbors[px][py][pz] && pNeighbors.neighbors[px][py][pz]->children ) 
						neighbors.neighbors[xx][yy][zz] = pNeighbors.neighbors[px][py][pz]->children + ( cornerIndex | (_x&1) );
					else
						neighbors.neighbors[xx][yy][zz] = NULL;
				}
			}
		}
	}
	// Otherwise recurse
	else
	{
		ConstNeighbors< _PLeftRadius + _PRightRadius + 1 > pNeighbors;
		getNeighbors< _PLeftRadius , _PRightRadius >( node->parent , pNeighbors );

		// Get the indices of the child node that would contain the point (and its antipode)
		int cx , cy , cz;
		Cube::FactorCornerIndex( (int)( node - node->parent->children ) , cx , cy , cz );


		// Iterate over the finer neighbors
		// Here:
		// (x,y,z) give the position of the finer nodes relative to the center,
		// (_x,_y,_z) give a positive global position, up to an even offset, and
		// (px-LeftRadius,py-LeftRadius,pz-LeftRadius) give the positions of their parents relative to the parent of the center
		for( int z=-(int)_LeftRadius ; z<=(int)_RightRadius ; z++ )
		{
			int _z = (z+cz) + (_LeftRadius<<1) , pz = ( _z>>1 ) - _LeftRadius + _PLeftRadius , zz = z + _LeftRadius;
			for( int y=-(int)_LeftRadius ; y<=(int)_RightRadius ; y++ )
			{
				int _y = (y+cy) + (_LeftRadius<<1) , py = ( _y>>1 ) - _LeftRadius + _PLeftRadius , yy = y + _LeftRadius;

				int cornerIndex = ( (_z&1)<<2 ) | ( (_y&1)<<1 );
				for( int x=-(int)_LeftRadius ; x<=(int)_RightRadius ; x++ )
				{
					int _x = (x+cx) + (_LeftRadius<<1) , px = ( _x>>1 ) - _LeftRadius + _PLeftRadius , xx = x + _LeftRadius;

					if( pNeighbors.neighbors[px][py][pz] && pNeighbors.neighbors[px][py][pz]->children )
						neighbors.neighbors[xx][yy][zz] = pNeighbors.neighbors[px][py][pz]->children + ( cornerIndex | (_x&1) );
					else
						neighbors.neighbors[xx][yy][zz] = NULL;
				}
			}
		}
	}
	return;
}

template< class NodeData >
int OctNode< NodeData >::write(const char* fileName) const{
	FILE* fp=fopen(fileName,"wb");
	if(!fp){return 0;}
	int ret=write(fp);
	fclose(fp);
	return ret;
}
template< class NodeData >
int OctNode< NodeData >::write(FILE* fp) const{
	fwrite(this,sizeof(OctNode< NodeData >),1,fp);
	if(children){for(int i=0;i<Cube::CORNERS;i++){children[i].write(fp);}}
	return 1;
}
template< class NodeData >
int OctNode< NodeData >::read(const char* fileName){
	FILE* fp=fopen(fileName,"rb");
	if(!fp){return 0;}
	int ret=read(fp);
	fclose(fp);
	return ret;
}
template< class NodeData >
int OctNode< NodeData >::read(FILE* fp){
	fread(this,sizeof(OctNode< NodeData >),1,fp);
	parent=NULL;
	if(children){
		children=NULL;
		initChildren();
		for(int i=0;i<Cube::CORNERS;i++){
			children[i].read(fp);
			children[i].parent=this;
		}
	}
	return 1;
}
template< class NodeData >
int OctNode< NodeData >::width(int maxDepth) const {
	int d=depth();
	return 1<<(maxDepth-d); 
}
template< class NodeData >
void OctNode< NodeData >::centerIndex(int maxDepth,int index[DIMENSION]) const
{
	int d,o[3];
	depthAndOffset(d,o);
	for(int i=0;i<DIMENSION;i++) index[i]=BinaryNode::CornerIndex( maxDepth , d+1 , o[i]<<1 , 1 );
}
