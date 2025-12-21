/*
Copyright (c) 2013, Michael Kazhdan
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

#include "PreProcessor.h"

#define DEFAULT_DIMENSION 3

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <algorithm>
#include <unordered_set>
#include <list>
#include "FEMTree.h"
#include "MyMiscellany.h"
#include "CmdLineParser.h"
#include "MAT.h"
#include "Geometry.h"
#include "Ply.h"
#include "VertexFactory.h"

using namespace PoissonRecon;

namespace {

CmdLineParameter< char* >
	In( "in" ) ,
	Out( "out" );
CmdLineParameter< float >
	Trim( "trim" ) ,
	IslandAreaRatio( "aRatio" , 0.001f );
CmdLineReadable
	PolygonMesh( "polygonMesh" ) ,
	Long( "long" ) ,
	ASCII( "ascii" ) ,
	RemoveIslands( "removeIslands" ) ,
	Debug( "debug" ) ,
	Verbose( "verbose" );


CmdLineReadable* params[] =
{
	&In , &Out , &Trim , &PolygonMesh , &IslandAreaRatio , &Verbose , &Long , &ASCII , &RemoveIslands , &Debug ,
	NULL
};

void ShowUsage( char* ex )
{
	printf( "Usage: %s\n" , ex );
	printf( "\t --%s <input polygon mesh>\n" , In.name );
	printf( "\t --%s <trimming value>\n" , Trim.name );
	printf( "\t[--%s <ouput polygon mesh>]\n" , Out.name );
	printf( "\t[--%s <relative area of islands>=%f]\n" , IslandAreaRatio.name , IslandAreaRatio.value );
	printf( "\t[--%s]\n" , RemoveIslands.name );
	printf( "\t[--%s]\n" , Debug.name );
	printf( "\t[--%s]\n" , PolygonMesh.name );
	printf( "\t[--%s]\n" , Long.name );
	printf( "\t[--%s]\n" , ASCII.name );
	printf( "\t[--%s]\n" , Verbose.name );
}

template< typename Index >
struct ComponentGraph
{
	struct Node
	{
		double area;
		std::vector< Node * > neighbors;
		std::list< Index > polygonIndices;

		Node( void ) : area(0) {}

		void merge( void )
		{
			auto PopBack =[&]( std::vector< Node * > &nodes , size_t idx )
			{
				nodes[idx] = nodes.back();
				nodes.pop_back();
			};

			if( !neighbors.size() ) MK_THROW( "No neighbors" );

			// Remove the node from the neighbors of the neighbors
			for( unsigned int i=0 ; i<neighbors.size() ; i++ ) for( int j=(int)neighbors[i]->neighbors.size()-1 ; j>=0 ; j-- ) if( neighbors[i]->neighbors[j]==this )
				PopBack( neighbors[i]->neighbors , j );

			// Merge the node into its first neighbor
			Node *first = neighbors[0];
			first->area += area;
			first->polygonIndices.splice( first->polygonIndices.end() , polygonIndices );

			// Merge the remaining neighbors into the first neighbor
			for( unsigned int i=1 ; i<neighbors.size() ; i++ )
			{
				first->area += neighbors[i]->area;
				first->polygonIndices.splice( first->polygonIndices.end() , neighbors[i]->polygonIndices );
				for( unsigned int j=0 ; j<neighbors[i]->neighbors.size() ; j++ )
				{
					bool foundNeighbor = false;
					for( int k=(int)neighbors[i]->neighbors[j]->neighbors.size()-1 ; k>=0 ; k-- )
						if( neighbors[i]->neighbors[j]->neighbors[k]==neighbors[i] ) PopBack( neighbors[i]->neighbors[j]->neighbors , k );

					for( unsigned int k=0 ; k<first->neighbors.size() ; k++ ) foundNeighbor |= neighbors[i]->neighbors[j]==first->neighbors[k];
					if( !foundNeighbor )
					{
						first->neighbors.push_back( neighbors[i]->neighbors[j] );
						neighbors[i]->neighbors[j]->neighbors.push_back( first );
					}
				}

				neighbors[i]->area = 0;
				neighbors[i]->neighbors.clear();
			}
			// Clean up the node
			polygonIndices.clear();
			area = 0;
		}
	};

	static void SanityCheck( size_t count , std::function< const Node * ( size_t ) > nodeFunction )
	{
		std::unordered_map< const Node * , int > flags;
		for( unsigned int i=0 ; i<count ; i++ ) flags[ nodeFunction(i) ] = 0;
		for( auto iter=flags.begin() ; iter!=flags.end() ; iter++ ) if( !iter->second ) _PropagateFlag( flags , iter->first , 1 );

		for( auto iter=flags.begin() ; iter!=flags.end() ; iter++ ) for( unsigned int j=0 ; j<iter->first->neighbors.size() ; j++ )
		{
			if( iter->second==flags[ iter->first->neighbors[j] ] ) MK_THROW( "Not a bipartite graph" );
			bool foundSelf = false;
			for( unsigned int k=0 ; k<iter->first->neighbors[j]->neighbors.size() ; k++ ) if( iter->first->neighbors[j]->neighbors[k]==iter->first ) foundSelf = true;
			if( !foundSelf ) MK_THROW( "Asymmetric graph" );
		}
	}

protected:
	static void _PropagateFlag( std::unordered_map< const Node * , int > &flags , const Node *n , int flag )
	{
		if( flags[n] ) return;
		else
		{
			flags[n] = flag;
			for( unsigned int i=0 ; i<n->neighbors.size() ; i++ ) _PropagateFlag( flags , n->neighbors[i] , -flag );
		}

	}

};

template< typename Real , unsigned int Dim , typename ... AuxData >
using ValuedPointData = DirectSum< Real , Point< Real , Dim > , Real , AuxData ... >;

template< typename Index >
size_t BoostHash( Index i1 , Index i2 )
{
	size_t hash = (size_t)i1 + 0x9e3779b9;
	hash ^= (size_t)i2 + 0x9e3779b9 + (hash<<6) + (hash>>2);
	return hash;
}

template< typename Index >
struct EdgeKey
{
	Index key1 , key2;
	EdgeKey( Index k1=0 , Index k2=0 )
	{
		if( k1<k2 ) key1 = k1 , key2 = k2;
		else        key1 = k2 , key2 = k1;
	}
	bool operator == ( const EdgeKey &key ) const  { return key1==key.key1 && key2==key.key2; }
	struct Hasher{ size_t operator()( const EdgeKey &key ) const { return BoostHash(key.key1,key.key2); } };
};

template< typename Index >
struct HalfEdgeKey
{
	Index key1 , key2;
	HalfEdgeKey( Index k1=0 , Index k2=0 ) : key1(k1) , key2(k2) {}
	HalfEdgeKey opposite( void ) const { return HalfEdgeKey( key2 , key1 ); }
	bool operator == ( const HalfEdgeKey &key ) const  { return key1==key.key1 && key2==key.key2; }
	struct Hasher{ size_t operator()( const HalfEdgeKey &key ) const { return BoostHash(key.key1,key.key2); } };
};

template< typename Real , unsigned int Dim ,  typename ... AuxData >
ValuedPointData< Real , Dim , AuxData ... > InterpolateVertices( const ValuedPointData< Real , Dim , AuxData ... >& v1 , const ValuedPointData< Real , Dim , AuxData ... >& v2 , Real value )
{
	if( v1.template get<1>()==v2.template get<1>() ) return (v1+v2)/Real(2.);
	Real dx = ( v1.template get<1>()-value ) / ( v1.template get<1>()-v2.template get<1>() );
	return v1 * (Real)(1.-dx) + v2*dx;
}

template< typename Real , unsigned int Dim , typename Index , typename ... AuxData >
void SplitPolygon
(
	const std::vector< Index >& polygon ,
	std::vector< ValuedPointData< Real , Dim , AuxData ... > >& vertices ,
	std::vector< std::vector< Index > >* ltPolygons , std::vector< std::vector< Index > >* gtPolygons ,
	std::vector< bool >* ltFlags , std::vector< bool >* gtFlags ,
	std::unordered_map< EdgeKey< Index > , Index , typename EdgeKey< Index >::Hasher >& vertexTable,
	Real trimValue
)
{
	int sz = int( polygon.size() );
	std::vector< bool > gt( sz );
	int gtCount = 0;
	for( int j=0 ; j<sz ; j++ )
	{
		gt[j] = ( vertices[ polygon[j] ].template get<1>()>trimValue );
		if( gt[j] ) gtCount++;
	}
	if     ( gtCount==sz ){ if( gtPolygons ) gtPolygons->push_back( polygon ) ; if( gtFlags ) gtFlags->push_back( false ); }
	else if( gtCount==0  ){ if( ltPolygons ) ltPolygons->push_back( polygon ) ; if( ltFlags ) ltFlags->push_back( false ); }
	else
	{
		int start;
		for( start=0 ; start<sz ; start++ ) if( gt[start] && !gt[(start+sz-1)%sz] ) break;

		bool gtFlag = true;
		std::vector< Index > poly;

		// Add the initial vertex
		{
			int j1 = (start+int(sz)-1)%sz , j2 = start;
			Index v1 = polygon[j1] , v2 = polygon[j2] , vIdx;
			typename std::unordered_map< EdgeKey< Index > , Index , typename EdgeKey< Index >::Hasher >::iterator iter = vertexTable.find( EdgeKey< Index >(v1,v2) );
			if( iter==vertexTable.end() )
			{
				vertexTable[ EdgeKey< Index >(v1,v2) ] = vIdx = (Index)vertices.size();
				vertices.push_back( InterpolateVertices( vertices[v1] , vertices[v2] , trimValue ) );
			}
			else vIdx = iter->second;
			poly.push_back( vIdx );
		}

		for( int _j=0  ; _j<=sz ; _j++ )
		{
			int j1 = (_j+start+sz-1)%sz , j2 = (_j+start)%sz;
			Index v1 = polygon[j1] , v2 = polygon[j2];
			if( gt[j2]==gtFlag ) poly.push_back( v2 );
			else
			{
				Index vIdx;
				typename std::unordered_map< EdgeKey< Index > , Index , typename EdgeKey< Index >::Hasher >::iterator iter = vertexTable.find( EdgeKey< Index >(v1,v2) );
				if( iter==vertexTable.end() )
				{
					vertexTable[ EdgeKey< Index >(v1,v2) ] = vIdx = (Index)vertices.size();
					vertices.push_back( InterpolateVertices( vertices[v1] , vertices[v2] , trimValue ) );
				}
				else vIdx = iter->second;
				poly.push_back( vIdx );
				if( gtFlag ){ if( gtPolygons ) gtPolygons->push_back( poly ) ; if( ltFlags ) ltFlags->push_back( true ); }
				else        { if( ltPolygons ) ltPolygons->push_back( poly ) ; if( gtFlags ) gtFlags->push_back( true ); }
				poly.clear() , poly.push_back( vIdx ) , poly.push_back( v2 );
				gtFlag = !gtFlag;
			}
		}
	}
}

template< class Real , unsigned int Dim , typename Index , class Vertex >
void Triangulate( const std::vector< Vertex >& vertices , const std::vector< std::vector< Index > >& polygons , std::vector< std::vector< Index > >& triangles )
{
	triangles.clear();
	for( size_t i=0 ; i<polygons.size() ; i++ )
		if( polygons[i].size()>3 )
		{
			std::vector< Point< Real , Dim > > _vertices( polygons[i].size() );
			for( int j=0 ; j<int( polygons[i].size() ) ; j++ ) _vertices[j] = vertices[ polygons[i][j] ].template get<0>();
			std::vector< TriangleIndex< Index > > _triangles = MinimalAreaTriangulation< Index , Real , Dim >( ( ConstPointer( Point< Real , Dim > ) )GetPointer( _vertices ) , _vertices.size() );

			// Add the triangles to the mesh
			size_t idx = triangles.size();
			triangles.resize( idx+_triangles.size() );
			for( int j=0 ; j<int(_triangles.size()) ; j++ )
			{
				triangles[idx+j].resize(3);
				for( int k=0 ; k<3 ; k++ ) triangles[idx+j][k] = polygons[i][ _triangles[j].idx[k] ];
			}
		}
		else if( polygons[i].size()==3 ) triangles.push_back( polygons[i] );
}

template< class Real , unsigned int Dim , typename Index , class Vertex >
double PolygonArea( const std::vector< Vertex >& vertices , const std::vector< Index >& polygon )
{
	auto Area =[]( Point< Real , Dim > v1 , Point< Real , Dim > v2 , Point< Real , Dim > v3 )
	{
		Point< Real , Dim > v[] = { v2-v1 , v3-v1 };
		XForm< Real , 2 > Mass;
		for( int i=0 ; i<2 ; i++ ) for( int j=0 ; j<2 ; j++ ) Mass(i,j) = Point< Real , Dim >::Dot( v[i] , v[j] );
		double det = Mass.determinant();
		if( det<0 ) return (Real)0;
		else return (Real)( sqrt( Mass.determinant() ) / 2. );
	};

	if( polygon.size()<3 ) return 0.;
	else if( polygon.size()==3 ) return Area( vertices[polygon[0]].template get<0>() , vertices[polygon[1]].template get<0>() , vertices[polygon[2]].template get<0>() );
	else
	{
		Point< Real , DEFAULT_DIMENSION > center;
		for( size_t i=0 ; i<polygon.size() ; i++ ) center += vertices[ polygon[i] ].template get<0>();
		center /= Real( polygon.size() );
		double area = 0;
		for( size_t i=0 ; i<polygon.size() ; i++ ) area += Area( center , vertices[ polygon[i] ].template get<0>() , vertices[ polygon[ (i+1)%polygon.size() ] ].template get<0>() );
		return area;
	}
}

template< typename Index , class Vertex >
void RemoveHangingVertices( std::vector< Vertex >& vertices , std::vector< std::vector< Index > >& polygons )
{
	std::unordered_map< Index, Index > vMap;
	std::vector< bool > vertexFlags( vertices.size() , false );
	for( size_t i=0 ; i<polygons.size() ; i++ ) for( size_t j=0 ; j<polygons[i].size() ; j++ ) vertexFlags[ polygons[i][j] ] = true;
	Index vCount = 0;
	for( Index i=0 ; i<(Index)vertices.size() ; i++ ) if( vertexFlags[i] ) vMap[i] = vCount++;
	for( size_t i=0 ; i<polygons.size() ; i++ ) for( size_t j=0 ; j<polygons[i].size() ; j++ ) polygons[i][j] = vMap[ polygons[i][j] ];

	std::vector< Vertex > _vertices( vCount );
	for( Index i=0 ; i<(Index)vertices.size() ; i++ ) if( vertexFlags[i] ) _vertices[ vMap[i] ] = vertices[i];
	vertices = _vertices;
}

template< typename Index >
void SetConnectedComponents( const std::vector< std::vector< Index > >& polygons , std::vector< std::vector< Index > >& components )
{
	std::vector< Index > polygonRoots( polygons.size() );
	for( size_t i=0 ; i<polygons.size() ; i++ ) polygonRoots[i] = (Index)i;
	std::unordered_map< EdgeKey< Index > , Index , typename EdgeKey< Index >::Hasher > edgeTable;
	for( size_t i=0 ; i<polygons.size() ; i++ )
	{
		int sz = int( polygons[i].size() );
		for( int j=0 ; j<sz ; j++ )
		{
			int j1 = j , j2 = (j+1)%sz;
			Index v1 = polygons[i][j1] , v2 = polygons[i][j2];
			EdgeKey< Index > eKey = EdgeKey< Index >(v1,v2);
			typename std::unordered_map< EdgeKey< Index > , Index , typename EdgeKey< Index >::Hasher >::iterator iter = edgeTable.find(eKey);
			if( iter==edgeTable.end() ) edgeTable[ eKey ] = (Index)i;
			else
			{
				Index p = iter->second;
				while( polygonRoots[p]!=p )
				{
					Index temp = polygonRoots[p];
					polygonRoots[p] = (Index)i;
					p = temp;
				}
				polygonRoots[p] = (Index)i;
			}
		}
	}
	for( size_t i=0 ; i<polygonRoots.size() ; i++ )
	{
		Index p = (Index)i;
		while( polygonRoots[p]!=p ) p = polygonRoots[p];
		Index root = p;
		p = (Index)i;
		while( polygonRoots[p]!=p )
		{
			Index temp = polygonRoots[p];
			polygonRoots[p] = root;
			p = temp;
		}
	}
	int cCount = 0;
	std::unordered_map< Index , Index > vMap;
	for( Index i=0 ; i<(Index)polygonRoots.size() ; i++ ) if( polygonRoots[i]==i ) vMap[i] = cCount++;
	components.resize( cCount );
	for( Index i=0 ; i<(Index)polygonRoots.size() ; i++ ) components[ vMap[ polygonRoots[i] ] ].push_back(i);
}

template< typename Real , unsigned int Dim , typename Index , typename ... AuxDataFactories >
int Execute( AuxDataFactories ... auxDataFactories )
{
	typedef VertexFactory::Factory< Real , typename VertexFactory::PositionFactory< Real , Dim > , typename VertexFactory::ValueFactory< Real > , AuxDataFactories ... > Factory;
	typedef typename Factory::VertexType Vertex;
	typename VertexFactory::PositionFactory< Real , Dim > pFactory;
	typename VertexFactory::ValueFactory< Real > vFactory;
	Factory factory( pFactory , vFactory , auxDataFactories ... );
	Real min , max;

	std::vector< Vertex > vertices;
	std::vector< std::vector< Index > > polygons;

	int ft;
	std::vector< std::string > comments;
	PLY::ReadPolygons< Factory , Index >( In.value , factory , vertices , polygons , ft , comments );

	min = max = vertices[0].template get<1>();
	for( size_t i=0 ; i<vertices.size() ; i++ ) min = std::min< Real >( min , vertices[i].template get<1>() ) , max = std::max< Real >( max , vertices[i].template get<1>() );

	std::unordered_map< EdgeKey< Index > , Index , typename EdgeKey< Index >::Hasher > vertexTable;
	std::vector< std::vector< Index > > ltPolygons , gtPolygons;
	std::vector< bool > ltFlags , gtFlags;

	if( Verbose.set )
	{
		std::cout << "*********************************************" << std::endl;
		std::cout << "*********************************************" << std::endl;
		std::cout << "** Running Surface Trimmer (Version " << ADAPTIVE_SOLVERS_VERSION << ") **" << std::endl;
		std::cout << "*********************************************" << std::endl;
		std::cout << "*********************************************" << std::endl;
	}
	char str[1024];
	for( int i=0 ; params[i] ; i++ )
		if( params[i]->set )
		{
			params[i]->writeValue( str );
			if( Verbose.set )
			{
				if( strlen( str ) ) std::cout << "\t--" << params[i]->name << " " << str << std::endl;
				else                std::cout << "\t--" << params[i]->name << std::endl;
			}
		}
	if( Verbose.set ) printf( "Value Range: [%f,%f]\n" , min , max );

	double t=Time();
	for( size_t i=0 ; i<polygons.size() ; i++ ) SplitPolygon( polygons[i] , vertices , &ltPolygons , &gtPolygons , &ltFlags , &gtFlags , vertexTable , Trim.value );

	if( IslandAreaRatio.value>0 )
	{
		std::vector< std::vector< Index > > _polygons , _components;
		size_t gtComponentStart;
		{
			std::vector< std::vector< Index > > ltComponents , gtComponents;
			SetConnectedComponents( ltPolygons , ltComponents );
			SetConnectedComponents( gtPolygons , gtComponents );
			gtComponentStart = ltComponents.size();
			for( unsigned int i=0 ; i<gtComponents.size() ; i++ ) for( unsigned int j=0 ; j<gtComponents[i].size() ; j++ ) gtComponents[i][j] += (Index)ltPolygons.size();

			_polygons.reserve( ltPolygons.size() + gtPolygons.size() );
			_components.reserve( ltComponents.size() + gtComponents.size() );
			_polygons.insert( _polygons.end() , ltPolygons.begin() , ltPolygons.end() );
			_polygons.insert( _polygons.end() , gtPolygons.begin() , gtPolygons.end() );
			_components.insert( _components.end() , ltComponents.begin() , ltComponents.end() );
			_components.insert( _components.end() , gtComponents.begin() , gtComponents.end() );
		}
		std::vector< typename ComponentGraph< Index >::Node > nodes( _components.size() );

		// Set the polygons within each component and compute areas
		for( unsigned int i=0 ; i<_components.size() ; i++ )
		{
			nodes[i].polygonIndices.insert( nodes[i].polygonIndices.end() , _components[i].begin() , _components[i].end() );
			for( auto iter=nodes[i].polygonIndices.begin() ; iter!=nodes[i].polygonIndices.end() ; iter++ ) nodes[i].area += PolygonArea< Real , Dim , Index , Vertex >( vertices , _polygons[ *iter ] );
		}

		// Compute the connectivity

		// A map identifying half-edges along the boundaries of components and associating them with the component
		std::unordered_map< HalfEdgeKey< Index > , Index , typename HalfEdgeKey< Index >::Hasher > componentBoundaryHalfEdges;
		for( unsigned int i=0 ; i<_components.size() ; i++ )
		{
			// All the half-edges for a given component
			std::unordered_set< HalfEdgeKey< Index > , typename HalfEdgeKey< Index >::Hasher > componentHalfEdges;
			for( unsigned int j=0 ; j<_components[i].size() ; j++ )
			{
				const std::vector< Index > &poly = _polygons[ _components[i][j] ];
				for( unsigned int k=0 ; k<poly.size() ; k++ )
				{
					Index v1 = poly[k] , v2 = poly[ (k+1)%poly.size() ];
					HalfEdgeKey< Index > eKey = HalfEdgeKey< Index >(v1,v2);
					componentHalfEdges.insert( eKey );
				}
			}
			for( auto iter=componentHalfEdges.begin() ; iter!=componentHalfEdges.end() ; iter++ )
			{
				HalfEdgeKey< Index > key = *iter;
				HalfEdgeKey< Index > _key = key.opposite();
				if( componentHalfEdges.find( _key )==componentHalfEdges.end() ) componentBoundaryHalfEdges[ key ] = (Index)i;
			}
		}

		// A set identify the dual edges of the component graph
		std::unordered_set< EdgeKey< Index > , typename EdgeKey< Index >::Hasher > componentEdges;
		for( auto iter=componentBoundaryHalfEdges.begin() ; iter!=componentBoundaryHalfEdges.end() ; iter++ )
		{
			HalfEdgeKey< Index > key = iter->first;
			HalfEdgeKey< Index > _key = key.opposite();
			auto _iter = componentBoundaryHalfEdges.find( _key );
			if( _iter!=componentBoundaryHalfEdges.end() ) componentEdges.insert( EdgeKey< Index >( iter->second , _iter->second ) );
		}
		for( auto iter=componentEdges.begin() ; iter!=componentEdges.end() ; iter++ )
		{
			nodes[ iter->key1 ].neighbors.push_back( &nodes[ iter->key2 ] );
			nodes[ iter->key2 ].neighbors.push_back( &nodes[ iter->key1 ] );
		}
		if( Debug.set ) ComponentGraph< Index >::SanityCheck( nodes.size() , [&]( size_t i ){ return &nodes[i]; } );

		auto ComponentCount = [&]( void )
		{
			unsigned int count = 0;
			for( unsigned int i=0 ; i<nodes.size() ; i++ ) if( nodes[i].polygonIndices.size() ) count++;
			return count;
		};

		double area = 0;
		for( unsigned int i=0 ; i<nodes.size() ; i++ ) area += nodes[i].area;

		bool done = false;
		while( !done )
		{
			done = true;
			unsigned int idx = -1;
			for( unsigned int i=0 ; i<nodes.size() ; i++ ) if( nodes[i].polygonIndices.size() && nodes[i].neighbors.size() ) if( idx==-1 || nodes[i].area<nodes[idx].area ) idx = i;
			if( idx!=-1 && nodes[idx].area<area*IslandAreaRatio.value )
			{
				nodes[idx].merge();
				done = false;
				if( Debug.set ) ComponentGraph< Index >::SanityCheck( nodes.size() , [&]( size_t i ){ return &nodes[i]; } );
			}
		}

		ltPolygons.clear() , gtPolygons.clear();

		for( unsigned int i=0 ; i<gtComponentStart ; i++ )
			if( !nodes[i].neighbors.size() && nodes[i].area<area*IslandAreaRatio.value && RemoveIslands.set ) ; // small island
			else for( auto iter=nodes[i].polygonIndices.begin() ; iter!=nodes[i].polygonIndices.end() ; iter++ ) ltPolygons.push_back( _polygons[ *iter ] );
		for( unsigned int i=(unsigned int)gtComponentStart ; i<nodes.size() ; i++ )
			if( !nodes[i].neighbors.size() && nodes[i].area<area*IslandAreaRatio.value && RemoveIslands.set ) ; // small island
			else for( auto iter=nodes[i].polygonIndices.begin() ; iter!=nodes[i].polygonIndices.end() ; iter++ ) gtPolygons.push_back( _polygons[ *iter ] );
	}

	if( !PolygonMesh.set )
	{
		{
			std::vector< std::vector< Index > > polys = ltPolygons;
			Triangulate< Real , Dim , Index , Vertex >( vertices , ltPolygons , polys ) , ltPolygons = polys;
		}
		{
			std::vector< std::vector< Index > > polys = gtPolygons;
			Triangulate< Real , Dim , Index , Vertex >( vertices , gtPolygons , polys ) , gtPolygons = polys;
		}
	}

	RemoveHangingVertices( vertices , gtPolygons );
	char comment[1024];
	sprintf( comment , "#Trimmed In: %9.1f (s)" , Time()-t );
	if( Out.set ) PLY::WritePolygons( Out.value , factory , vertices , gtPolygons , ASCII.set ? PLY_ASCII : ft , comments );

	return EXIT_SUCCESS;
}

}  // namespace

int RunSurfaceTrimmer( int argc , char* argv[] )
{
	CmdLineParse( argc-1 , &argv[1] , params );

	if( !In.set || !Trim.set )
	{
		ShowUsage( argv[0] );
		return EXIT_FAILURE;
	}
	typedef float Real;
	static constexpr unsigned int Dim = DEFAULT_DIMENSION;
	typedef VertexFactory::Factory< Real , typename VertexFactory::PositionFactory< Real , Dim > , typename VertexFactory::ValueFactory< Real > > Factory;
	Factory factory;
	bool *readFlags = new bool[ factory.plyReadNum() ];
	std::vector< PlyProperty > unprocessedProperties;
	size_t vNum;
	PLY::ReadVertexHeader( In.value , factory , readFlags , unprocessedProperties , vNum );
	if( vNum>std::numeric_limits< int >::max() )
	{
		if( !Long.set ) MK_WARN( "Number of vertices not supported by 32-bit indexing. Switching to 64-bit indexing" );
		Long.set = true;
	}
	if( !factory.template plyValidReadProperties<0>( readFlags ) ) MK_THROW( "Ply file does not contain positions" );
	if( !factory.template plyValidReadProperties<1>( readFlags ) ) MK_THROW( "Ply file does not contain values" );
	delete[] readFlags;

	if( Long.set ) return Execute< Real , Dim , long long >( VertexFactory::DynamicFactory< Real >( unprocessedProperties ) );
	else           return Execute< Real , Dim , int       >( VertexFactory::DynamicFactory< Real >( unprocessedProperties ) );

}
