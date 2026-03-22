/*

Header for PLY polygon files.

- Greg Turk, March 1994

A PLY file contains a single polygonal _object_.

An object is composed of lists of _elements_.  Typical elements are
vertices, faces, edges and materials.

Each type of element for a given object has one or more _properties_
associated with the element type.  For instance, a vertex element may
have as properties three floating-point values x,y,z and three unsigned
chars for red, green and blue.

---------------------------------------------------------------

Copyright (c) 2020, Georgia Institute of Technology
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer. Redistributions in binary form
must reproduce the above copyright notice, this list of conditions and the
following disclaimer in the documentation and/or other materials provided with
the distribution. 

Neither the name of the Georgia Institute of Technology nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef PLY_INCLUDED
#define PLY_INCLUDED

#include <vector>
#include <string>
#include <functional>
#include <math.h>
#include "PlyFile.h"
#include "Geometry.h"
#include "DataStream.h"
#include "MyMiscellany.h"
#include "Array.h"

namespace PoissonRecon
{
	namespace PLY
	{
		// Converts from C-type to PLY type
		template< class Scalar > int Type( void );

		// Converts from C-type to PLY name
		template< typename Integer > struct Traits{ static const std::string name; };

		// A structure representing a face
		template< typename Index >
		struct Edge
		{
			Index v1 , v2;
			static const PlyProperty Properties[];
		};

		// A structure representing a face
		template< typename Index , bool UseCharIndex=false >
		struct Face
		{
			unsigned int nr_vertices;
			Index *vertices;

			static const PlyProperty Properties[];
		};

		inline int DefaultFileType( void );

		// PLY read header functionality

		// Get the properties (and return the file type)
		inline int ReadVertexHeader( std::string fileName , std::vector< PlyProperty > &properties );

		// Test which properties are represented by elements of the vertex factory (and return the file type)
		template< typename VertexFactory >
		int ReadVertexHeader( std::string fileName , const VertexFactory &vFactory , bool *readFlags );

		// Test which properties are represented by elements of the vertex factory and add the others to the property list (and return the file type)
		template< typename VertexFactory >
		int ReadVertexHeader( std::string fileName , const VertexFactory &vFactory , bool *readFlags , std::vector< PlyProperty > &unprocessedProperties );

		// PLY write mesh functionality
		template< typename VertexFactory , typename Index , class Real , int Dim , typename OutputIndex=int , bool UseCharIndex=false >
		void Write( std::string fileName , const VertexFactory &vFactory , size_t vertexNum , size_t polygonNum , InputDataStream< typename VertexFactory::VertexType > &vertexStream , InputDataStream< std::vector< Index > > &polygonStream , int file_type , const std::vector< std::string >& comments );

		template< typename VertexFactory , typename Index , class Real , int Dim , typename OutputIndex=int >
		void Write( std::string fileName , const VertexFactory &vFactory , size_t vertexNum , size_t edgeNum , InputDataStream< typename VertexFactory::VertexType > &vertexStream , InputDataStream< std::pair< Index , Index > > &edgeStream , int file_type , const std::vector< std::string >& comments );

		template< typename VertexFactory , typename Index , bool UseCharIndex=false >
		void WritePolygons( std::string fileName , const VertexFactory &vFactory , const std::vector< typename VertexFactory::VertexType > &vertices , const std::vector< std::vector< Index > > &polygons , int file_type , const std::vector< std::string > &comments );

		// PLY read mesh functionality
		template< typename VertexFactory , typename Index >
		void ReadPolygons( std::string fileName , const VertexFactory &vFactory , std::vector< typename VertexFactory::VertexType > &vertices , std::vector< std::vector< Index > >& polygons , int &file_type , std::vector< std::string > &comments , bool* readFlags=NULL );

		template< typename VertexFactory , typename Index >
		void ReadEdges( std::string fileName , const VertexFactory &vFactory , std::vector< typename VertexFactory::VertexType > &vertices , std::vector< std::pair< Index , Index > >& edges , int &file_type , std::vector< std::string > &comments , bool* readFlags=NULL );
	}
#include "Ply.inl"
}

#endif // PLY_INCLUDED
