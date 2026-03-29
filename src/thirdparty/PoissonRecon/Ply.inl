/*
Copyright (c) 2020, Michael Kazhdan
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

namespace PLY
{
	inline int DefaultFileType( void ){ return PLY_ASCII; }

	template<> inline int Type<          int  >( void ){ return PLY_INT   ; }
	template<> inline int Type< unsigned int  >( void ){ return PLY_UINT  ; }
	template<> inline int Type<          char >( void ){ return PLY_CHAR  ; }
	template<> inline int Type< unsigned char >( void ){ return PLY_UCHAR ; }
	template<> inline int Type<        float  >( void ){ return PLY_FLOAT ; }
	template<> inline int Type<        double >( void ){ return PLY_DOUBLE; }
	template< class Scalar > inline int Type( void )
	{
		MK_THROW( "Unrecognized scalar type: " , typeid(Scalar).name() );
		return -1;
	}

	template<> const inline std::string Traits<          int >::name="int";
	template<> const inline std::string Traits< unsigned int >::name="unsigned int";
	template<> const inline std::string Traits<          long >::name="long";
	template<> const inline std::string Traits< unsigned long >::name="unsigned long";
	template<> const inline std::string Traits<          long long >::name="long long";
	template<> const inline std::string Traits< unsigned long long >::name="unsigned long long";

	template<> const inline PlyProperty Edge<          int       >::Properties[] = { PlyProperty( "v1" , PLY_INT       , PLY_INT       , offsetof( Edge , v1 ) ) , PlyProperty( "v2" , PLY_INT       , PLY_INT       , offsetof( Edge , v2 ) ) };
	template<> const inline PlyProperty Edge< unsigned int       >::Properties[] = { PlyProperty( "v1" , PLY_UINT      , PLY_UINT      , offsetof( Edge , v1 ) ) , PlyProperty( "v2" , PLY_UINT      , PLY_UINT      , offsetof( Edge , v2 ) ) };
	template<> const inline PlyProperty Edge<          long long >::Properties[] = { PlyProperty( "v1" , PLY_LONGLONG  , PLY_LONGLONG  , offsetof( Edge , v1 ) ) , PlyProperty( "v2" , PLY_LONGLONG  , PLY_LONGLONG  , offsetof( Edge , v2 ) ) };
	template<> const inline PlyProperty Edge< unsigned long long >::Properties[] = { PlyProperty( "v1" , PLY_ULONGLONG , PLY_ULONGLONG , offsetof( Edge , v1 ) ) , PlyProperty( "v2" , PLY_ULONGLONG , PLY_ULONGLONG , offsetof( Edge , v2 ) ) };

	template<> const inline PlyProperty Face<          int       , false >::Properties[] = { PlyProperty( "vertex_indices" , PLY_INT       , PLY_INT       , offsetof( Face , vertices ) , 1 , PLY_INT , PLY_INT , offsetof( Face , nr_vertices ) ) };
	template<> const inline PlyProperty Face< unsigned int       , false >::Properties[] = { PlyProperty( "vertex_indices" , PLY_UINT      , PLY_UINT      , offsetof( Face , vertices ) , 1 , PLY_INT , PLY_INT , offsetof( Face , nr_vertices ) ) };
	template<> const inline PlyProperty Face<          long long , false >::Properties[] = { PlyProperty( "vertex_indices" , PLY_LONGLONG  , PLY_LONGLONG  , offsetof( Face , vertices ) , 1 , PLY_INT , PLY_INT , offsetof( Face , nr_vertices ) ) };
	template<> const inline PlyProperty Face< unsigned long long , false >::Properties[] = { PlyProperty( "vertex_indices" , PLY_ULONGLONG , PLY_ULONGLONG , offsetof( Face , vertices ) , 1 , PLY_INT , PLY_INT , offsetof( Face , nr_vertices ) ) };
	template<> const inline PlyProperty Face<          int       , true  >::Properties[] = { PlyProperty( "vertex_indices" , PLY_INT       , PLY_INT       , offsetof( Face , vertices ) , 1 , PLY_CHAR , PLY_CHAR , offsetof( Face , nr_vertices ) ) };
	template<> const inline PlyProperty Face< unsigned int       , true  >::Properties[] = { PlyProperty( "vertex_indices" , PLY_UINT      , PLY_UINT      , offsetof( Face , vertices ) , 1 , PLY_CHAR , PLY_CHAR , offsetof( Face , nr_vertices ) ) };
	template<> const inline PlyProperty Face<          long long , true  >::Properties[] = { PlyProperty( "vertex_indices" , PLY_LONGLONG  , PLY_LONGLONG  , offsetof( Face , vertices ) , 1 , PLY_CHAR , PLY_CHAR , offsetof( Face , nr_vertices ) ) };
	template<> const inline PlyProperty Face< unsigned long long , true  >::Properties[] = { PlyProperty( "vertex_indices" , PLY_ULONGLONG , PLY_ULONGLONG , offsetof( Face , vertices ) , 1 , PLY_CHAR , PLY_CHAR , offsetof( Face , nr_vertices ) ) };

	// Read
	inline PlyFile *ReadHeader( std::string fileName , int &fileType , std::vector< std::tuple< std::string , size_t , std::vector< PlyProperty > > > &elems , std::vector< std::string > &comments )
	{
		std::vector< std::string > elist;
		float version;

		PlyFile *ply = PlyFile::Read( fileName , elist , fileType , version );
		if( !ply ) MK_THROW( "Could not open ply file for reading: " , fileName );

		elems.resize( elist.size() );
		for( unsigned int i=0 ; i<elist.size() ; i++ )
		{
			std::get<0>( elems[i] ) = elist[i];
			std::get<2>( elems[i] ) = ply->get_element_description( std::get<0>( elems[i] ) , std::get<1>( elems[i] ) );
		}

		comments.resize( ply->comments.size() );
		for( int i=0 ; i<ply->comments.size() ; i++ ) comments[i] = ply->comments[i];

		return ply;
	}

	inline PlyFile *WriteHeader( std::string fileName , int fileType , const std::vector< std::tuple< std::string , size_t , std::vector< PlyProperty > > > &elems , const std::vector< std::string > &comments )
	{
		PlyFile *ply = NULL;
		{
			float version;
			std::vector< std::string > elist( elems.size() );
			for( unsigned int i=0 ; i<elems.size() ; i++ ) elist[i] = std::get<0>( elems[i] );
			ply = PlyFile::Write( fileName , elist , fileType , version );
		}
		if( !ply ) MK_THROW( "Could not open ply for writing: " , fileName );
		for( unsigned int i=0 ; i<elems.size() ; i++ )
		{
			ply->element_count( std::get<0>( elems[i] ) , std::get<1>( elems[i] ) );
			const std::vector< PlyProperty > &props = std::get<2>( elems[i] );
			for( unsigned int j=0 ; j<props.size() ; j++ ) ply->describe_property( std::get<0>( elems[i] ) , &props[j] );
		}

		for( int i=0 ; i<comments.size() ; i++ ) ply->put_comment( comments[i] );
		ply->header_complete();

		return ply;
	}

	inline PlyFile *ReadHeader( std::string fileName , int &fileType , std::vector< std::tuple< std::string , size_t , std::vector< PlyProperty > > > &elems )
	{
		std::vector< std::string > comments;
		return ReadHeader( fileName , fileType , elems , comments );
	}

	inline PlyFile *WriteHeader( std::string fileName , int fileType , const std::vector< std::tuple< std::string , size_t , std::vector< PlyProperty > > > &elems )
	{
		std::vector< std::string > comments;
		return WriteHeader( fileName , fileType , elems , comments );
	}

	template< typename VertexFactory >
	inline int ReadVertexHeader( std::string fileName , const VertexFactory &vFactory , bool *readFlags )
	{
		int fileType;
		std::vector< std::string > elist;
		float version;

		PlyFile *ply = PlyFile::Read( fileName , elist , fileType , version );
		if( !ply ) MK_THROW( "could not create read ply file: " , fileName );

		for( int i=0 ; i<(int)elist.size() ; i++ ) if( elist[i]=="vertex" ) for( unsigned int j=0 ; j<vFactory.plyReadNum() ; j++ )
		{
			PlyProperty prop;
			if constexpr( VertexFactory::IsStaticallyAllocated() ) prop = vFactory.plyStaticReadProperty(j);
			else                                                   prop = vFactory.plyReadProperty(j);
			readFlags[j] = ( ply->get_property( elist[i] , &prop )!=0 );
		}

		delete ply;
		return fileType;
	}

	template< typename VertexFactory >
	inline int ReadVertexHeader( std::string fileName , const VertexFactory &vFactory , bool *readFlags , std::vector< PlyProperty > &unprocessedProperties , size_t &vNum )
	{
		int fileType;
		std::vector< std::string > elist;
		float version;

		PlyFile *ply = PlyFile::Read( fileName, elist, fileType, version );
		if( !ply ) MK_THROW( "Failed to open ply file for reading: " , fileName );

		std::vector< PlyProperty > plist = ply->get_element_description( "vertex" , vNum );
		if( !plist.size() ) MK_THROW( "Failed to get element description: vertex" );
		for( unsigned int i=0 ; i<vFactory.plyReadNum() ; i++ ) readFlags[i] = false;

		for( int i=0 ; i<plist.size() ; i++ )
		{
			bool found = false;
			for( unsigned int j=0 ; j<vFactory.plyReadNum() ; j++ )
			{
				PlyProperty prop;
				if constexpr( VertexFactory::IsStaticallyAllocated() ) prop = vFactory.plyStaticReadProperty(j);
				else                                                   prop = vFactory.plyReadProperty(j);
				if( prop.name==plist[i].name ) found = readFlags[j] = true;
			}
			if( !found ) unprocessedProperties.push_back( plist[i] );
		}
		delete ply;
		return fileType;
	}
	template< typename VertexFactory >
	inline int ReadVertexHeader( std::string fileName , const VertexFactory &vFactory , bool *readFlags , std::vector< PlyProperty > &unprocessedProperties )
	{
		size_t vNum;
		return ReadVertexHeader( fileName , vFactory , readFlags , unprocessedProperties , vNum );
	}

	inline int ReadVertexHeader( std::string fileName , std::vector< PlyProperty > &properties , size_t &vNum )
	{
		int fileType;
		std::vector< std::string > elist;
		float version;

		PlyFile *ply = PlyFile::Read( fileName, elist, fileType, version );
		if( !ply ) MK_THROW( "Failed to open ply file for reading: " , fileName );

		std::vector< PlyProperty > plist = ply->get_element_description( "vertex" , vNum );
		for( int i=0 ; i<plist.size() ; i++ ) properties.push_back( plist[i] );
		delete ply;
		return fileType;
	}

	inline int ReadVertexHeader( std::string fileName , std::vector< PlyProperty > &properties )
	{
		size_t vNum;
		return ReadVertexHeader( fileName , properties , vNum );
	}

	template< typename VertexFactory , typename Index >
	void ReadEdges( std::string fileName , const VertexFactory &vFactory , std::vector< typename VertexFactory::VertexType > &vertices , std::vector< std::pair< Index , Index > > &edges , int &file_type , std::vector< std::string > &comments , bool *readFlags )
	{
		std::vector< std::string > elist;
		float version;

		PlyFile *ply = PlyFile::Read( fileName , elist , file_type , version );
		if( !ply ) MK_THROW( "Could not create ply file for reading: " , fileName );

		comments.reserve( comments.size() + ply->comments.size() );
		for( int i=0 ; i<ply->comments.size() ; i++ ) comments.push_back( ply->comments[i] );

		for( int i=0 ; i<elist.size() ; i++ )
		{
			std::string &elem_name = elist[i];
			size_t num_elems;
			std::vector< PlyProperty > plist = ply->get_element_description( elem_name , num_elems );
			if( !plist.size() ) MK_THROW( "Could not read element properties: " , elem_name );
			if( elem_name=="vertex" )
			{
				for( unsigned int i=0 ; i<vFactory.plyReadNum() ; i++ )
				{
					PlyProperty prop;
					if constexpr( VertexFactory::IsStaticallyAllocated() ) prop = vFactory.plyStaticReadProperty(i);
					else                                                   prop = vFactory.plyReadProperty(i);
					int hasProperty = ply->get_property( elem_name , &prop );
					if( readFlags ) readFlags[i] = (hasProperty!=0);
				}
				vertices.resize( num_elems , vFactory() );
				Pointer( char ) buffer = NewPointer< char >( vFactory.bufferSize() );
				for( size_t j=0 ; j<num_elems ; j++ )
				{
					if constexpr( VertexFactory::IsStaticallyAllocated() ) ply->get_element( (void*)&vertices[j] );
					else
					{
						ply->get_element( PointerAddress( buffer ) );
						vFactory.fromBuffer( buffer , vertices[j] );
					}
				}
				DeletePointer( buffer );
			}
			else if( elem_name=="edge" )
			{
				ply->get_property( elem_name , &Edge< Index >::Properties[0] );
				ply->get_property( elem_name , &Edge< Index >::Properties[1] );
				edges.resize( num_elems );
				for( size_t j=0 ; j<num_elems ; j++ )
				{
					Edge< Index > ply_edge;
					ply->get_element( (void *)&ply_edge );
					edges[j].first  = ply_edge.v1;
					edges[j].second = ply_edge.v2;
				}  // for, read edges
			}  // if face
			else ply->get_other_element( elem_name , num_elems );
		}  // for each type of element

		delete ply;
	}

	template< typename VertexFactory , typename Index >
	void ReadPolygons( std::string fileName , const VertexFactory &vFactory , std::vector< typename VertexFactory::VertexType > &vertices , std::vector< std::vector< Index > > &polygons , int &file_type , std::vector< std::string > &comments , bool *readFlags )
	{
		std::vector< std::string > elist;
		float version;

		PlyFile *ply = PlyFile::Read( fileName , elist , file_type , version );
		if( !ply ) MK_THROW( "Could not create ply file for reading: " , fileName );

		comments.reserve( comments.size() + ply->comments.size() );
		for( int i=0 ; i<ply->comments.size() ; i++ ) comments.push_back( ply->comments[i] );

		for( int i=0 ; i<elist.size() ; i++ )
		{
			std::string &elem_name = elist[i];
			size_t num_elems;
			std::vector< PlyProperty > plist = ply->get_element_description( elem_name , num_elems );
			if( !plist.size() ) MK_THROW( "Could not read element properties: " , elem_name );
			if( elem_name=="vertex" )
			{
				for( unsigned int i=0 ; i<vFactory.plyReadNum() ; i++)
				{
					PlyProperty prop;
					if constexpr( VertexFactory::IsStaticallyAllocated() ) prop = vFactory.plyStaticReadProperty(i);
					else                                                   prop = vFactory.plyReadProperty(i);
					int hasProperty = ply->get_property( elem_name , &prop );
					if( readFlags ) readFlags[i] = (hasProperty!=0);
				}
				vertices.resize( num_elems , vFactory() );
				Pointer( char ) buffer = NewPointer< char >( vFactory.bufferSize() );
				for( size_t j=0 ; j<num_elems ; j++ )
				{
					if constexpr( VertexFactory::IsStaticallyAllocated() ) ply->get_element( (void*)&vertices[j] );
					else
					{
						ply->get_element( PointerAddress( buffer ) );
						vFactory.fromBuffer( buffer , vertices[j] );
					}
				}
				DeletePointer( buffer );
			}
			else if( elem_name=="face" )
			{
				ply->get_property( elem_name , Face< Index >::Properties );
				polygons.resize( num_elems );
				for( size_t j=0 ; j<num_elems ; j++ )
				{
					Face< Index > ply_face;
					ply->get_element( (void *)&ply_face );
					polygons[j].resize( ply_face.nr_vertices );
					for( unsigned int k=0 ; k<ply_face.nr_vertices ; k++ ) polygons[j][k] = ply_face.vertices[k];
					free( ply_face.vertices );
				}  // for, read faces
			}  // if face
			else ply->get_other_element( elem_name , num_elems );
		}  // for each type of element

		delete ply;
	}

	template< typename VertexFactory , typename Index , bool UseCharIndex >
	void WritePolygons( std::string fileName , const VertexFactory &vFactory , const std::vector< typename VertexFactory::VertexType > &vertices , const std::vector< std::vector< Index > > &polygons , int file_type , const std::vector< std::string > &comments )
	{
		size_t nr_vertices = vertices.size();
		size_t nr_faces = polygons.size();
		float version;
		std::vector< std::string > elem_names = { std::string( "vertex" ) , std::string( "face" ) };
		PlyFile *ply = PlyFile::Write( fileName , elem_names , file_type , version );
		if( !ply ) MK_THROW( "Could not create ply file for writing: " , fileName );

		//
		// describe vertex and face properties
		//
		ply->element_count( "vertex", nr_vertices );
		for( unsigned int i=0 ; i<vFactory.plyWriteNum() ; i++ )
		{
			PlyProperty prop;
			if constexpr( VertexFactory::IsStaticallyAllocated() ) prop = vFactory.plyStaticWriteProperty(i);
			else                                                   prop = vFactory.plyWriteProperty(i);
			ply->describe_property( "vertex" , &prop );
		}
		ply->element_count( "face" , nr_faces );
		ply->describe_property( "face" , Face< Index , UseCharIndex >::Properties );

		// Write in the comments
		for( int i=0 ; i<comments.size() ; i++ ) ply->put_comment( comments[i] );
		ply->header_complete();

		// write vertices
		ply->put_element_setup( elem_names[0] );

		Pointer( char ) buffer = NewPointer< char >( vFactory.bufferSize() );
		for( size_t j=0 ; j<vertices.size() ; j++ )
		{
			if constexpr( VertexFactory::IsStaticallyAllocated() ) ply->put_element( (void *)&vertices[j] );
			else
			{
				vFactory.toBuffer( vertices[j] , buffer );
				ply->put_element( PointerAddress( buffer ) );
			}
		}
		DeletePointer( buffer );

		// write faces
		Face< Index > ply_face;
		int maxFaceVerts=3;
		ply_face.nr_vertices = 3;
		ply_face.vertices = new Index[3];

		ply->put_element_setup( elem_names[1] );
		for( size_t i=0 ; i<nr_faces ; i++ )
		{
			if( (int)polygons[i].size()>maxFaceVerts )
			{
				delete[] ply_face.vertices;
				maxFaceVerts = (int)polygons[i].size();
				ply_face.vertices=new Index[ maxFaceVerts ];
			}
			ply_face.nr_vertices = (int)polygons[i].size();
			for( size_t j=0 ; j<ply_face.nr_vertices ; j++ ) ply_face.vertices[j] = polygons[i][j];
			ply->put_element( (void *)&ply_face );
		}

		delete[] ply_face.vertices;
		delete ply;
	}

	template< typename VertexFactory , typename Index , class Real , int Dim , typename OutputIndex , bool UseCharIndex >
	void Write( std::string fileName , const VertexFactory &vFactory , size_t vertexNum , size_t polygonNum , InputDataStream< typename VertexFactory::VertexType > &vertexStream , InputDataStream< std::vector< Index > > &polygonStream , int file_type , const std::vector< std::string > &comments )
	{
		if( vertexNum>(size_t)std::numeric_limits< OutputIndex >::max() )
		{
			if( std::is_same< Index , OutputIndex >::value ) MK_THROW( "more vertices than can be represented using " , Traits< Index >::name );
			MK_WARN( "more vertices than can be represented using " , Traits< OutputIndex >::name , " using " , Traits< Index >::name , " instead" );
			return Write< VertexFactory , Index , Real , Dim , Index >( fileName , vFactory , vertexNum , polygonNum , vertexStream , polygonStream , file_type , comments );
		}
		float version;
		std::vector< std::string > elem_names = { std::string( "vertex" ) , std::string( "face" ) };
		PlyFile *ply = PlyFile::Write( fileName , elem_names , file_type , version );
		if( !ply ) MK_THROW( "Could not create ply file for writing: " , fileName );

		vertexStream.reset();
		polygonStream.reset();

		//
		// describe vertex and face properties
		//
		ply->element_count( "vertex" , vertexNum );
		for( unsigned int i=0 ; i<vFactory.plyWriteNum() ; i++ )
		{
			PlyProperty prop;
			if constexpr( VertexFactory::IsStaticallyAllocated() ) prop = vFactory.plyStaticWriteProperty(i);
			else                                                   prop = vFactory.plyWriteProperty(i);
			ply->describe_property( "vertex" , &prop );
		}
		ply->element_count( "face" , polygonNum );
		ply->describe_property( "face" , Face< OutputIndex , UseCharIndex >::Properties );

		// Write in the comments
		for( int i=0 ; i<comments.size() ; i++ ) ply->put_comment( comments[i] );
		ply->header_complete();

		// write vertices
		ply->put_element_setup( "vertex" );
		if constexpr( VertexFactory::IsStaticallyAllocated() )
		{
			for( size_t i=0; i<vertexNum ; i++ )
			{
				typename VertexFactory::VertexType vertex = vFactory();
				if( !vertexStream.read( vertex ) ) MK_THROW( "Failed to read vertex " , i , " / " , vertexNum );
				ply->put_element( (void *)&vertex );
			}
		}
		else
		{
			Pointer( char ) buffer = NewPointer< char >( vFactory.bufferSize() );
			for( size_t i=0; i<vertexNum ; i++ )
			{
				typename VertexFactory::VertexType vertex = vFactory();
				if( !vertexStream.read( vertex ) ) MK_THROW( "Failed to read vertex " , i , " / " , vertexNum );
				vFactory.toBuffer( vertex , buffer );
				ply->put_element( PointerAddress( buffer ) );
			}
			DeletePointer( buffer );
		}

		// write faces
		std::vector< Index > polygon;
		ply->put_element_setup( "face" );
		for( size_t i=0 ; i<polygonNum ; i++ )
		{
			//
			// create and fill a struct that the ply code can handle
			//
			Face< OutputIndex > ply_face;
			if( !polygonStream.read( polygon ) ) MK_THROW( "Failed to read polygon " , i , " / " , polygonNum ); 
			ply_face.nr_vertices = int( polygon.size() );
			ply_face.vertices = new OutputIndex[ polygon.size() ];
			for( int j=0 ; j<int(polygon.size()) ; j++ ) ply_face.vertices[j] = (OutputIndex)polygon[j];
			ply->put_element( (void *)&ply_face );
			delete[] ply_face.vertices;
		}  // for, write faces


		delete ply;
	}

	template< typename VertexFactory , typename Index , class Real , int Dim , typename OutputIndex >
	void Write( std::string fileName , const VertexFactory &vFactory , size_t vertexNum , size_t edgeNum , InputDataStream< typename VertexFactory::VertexType > &vertexStream , InputDataStream< std::pair< Index , Index > > &edgeStream , int file_type , const std::vector< std::string > &comments )
	{
		if( vertexNum>(size_t)std::numeric_limits< OutputIndex >::max() )
		{
			if( std::is_same< Index , OutputIndex >::value ) MK_THROW( "more vertices than can be represented using " , Traits< Index >::name );
			MK_WARN( "more vertices than can be represented using " , Traits< OutputIndex >::name , " using " , Traits< Index >::name , " instead" );
			return Write< VertexFactory , Index , Real , Dim , Index >( fileName , vFactory , vertexNum , edgeNum , vertexStream , edgeStream , file_type , comments );
		}
		float version;
		std::vector< std::string > elem_names = { std::string( "vertex" ) , std::string( "edge" ) };
		PlyFile *ply = PlyFile::Write( fileName , elem_names , file_type , version );
		if( !ply ) MK_THROW( "Could not create ply file for writing: " , fileName );

		vertexStream.reset();
		edgeStream.reset();

		//
		// describe vertex and face properties
		//
		ply->element_count( "vertex" , vertexNum );
		for( unsigned int i=0 ; i<vFactory.plyWriteNum() ; i++ )
		{
			PlyProperty prop;
			if constexpr( VertexFactory::IsStaticallyAllocated() ) prop = vFactory.plyStaticWriteProperty(i);
			else                                                   prop = vFactory.plyWriteProperty(i);
			ply->describe_property( "vertex" , &prop );
		}
		ply->element_count( "edge" , edgeNum );
		ply->describe_property( "edge" , &Edge< OutputIndex >::Properties[0] );
		ply->describe_property( "edge" , &Edge< OutputIndex >::Properties[1] );

		// Write in the comments
		for( int i=0 ; i<comments.size() ; i++ ) ply->put_comment( comments[i] );
		ply->header_complete();

		// write vertices
		ply->put_element_setup( "vertex" );
		if constexpr( VertexFactory::IsStaticallyAllocated() )
		{
			for( size_t i=0; i<vertexNum ; i++ )
			{
				typename VertexFactory::VertexType vertex = vFactory();
				if( !vertexStream.read( vertex ) ) MK_THROW( "Failed to read vertex " , i , " / " , vertexNum );
				ply->put_element( (void *)&vertex );
			}
		}
		else
		{
			Pointer( char ) buffer = NewPointer< char >( vFactory.bufferSize() );
			for( size_t i=0; i<vertexNum ; i++ )
			{
				typename VertexFactory::VertexType vertex = vFactory();
				if( !vertexStream.read( vertex ) ) MK_THROW( "Failed to read vertex " , i , " / " , vertexNum );
				vFactory.toBuffer( vertex , buffer );
				ply->put_element( PointerAddress( buffer ) );
			}
			DeletePointer( buffer );
		}

		// write edges
		std::pair< Index , Index > edge;
		ply->put_element_setup( "edge" );
		for( size_t i=0 ; i<edgeNum ; i++ )
		{
			//
			// create and fill a struct that the ply code can handle
			//
			Edge< OutputIndex > ply_edge;
			if( !edgeStream.read( edge ) ) MK_THROW( "Failed to read edge " , i , " / " , edgeNum ); 
			ply_edge.v1 = (OutputIndex)edge.first;
			ply_edge.v2 = (OutputIndex)edge.second;
			ply->put_element( (void *)&ply_edge );
		}  // for, write edges

		delete ply;
	}
}
