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

#ifndef PLY_FILE_INCLUDED
#define PLY_FILE_INCLUDED

#include <string>
#include <vector>

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include "MyMiscellany.h"
#include "Streams.h"

namespace PoissonRecon
{

#define PLY_ASCII         1      /* ascii PLY file */
#define PLY_BINARY_BE     2      /* binary PLY file, big endian */
#define PLY_BINARY_LE     3      /* binary PLY file, little endian */
#define PLY_BINARY_NATIVE 4      /* binary PLY file, same endianness as current architecture */

#define PLY_OKAY    0           /* ply routine worked okay */
#define PLY_ERROR  -1           /* error in ply routine */

	/* scalar data types supported by PLY format */

#define PLY_START_TYPE 0
#define PLY_CHAR       1
#define PLY_SHORT      2
#define PLY_INT        3
#define PLY_LONGLONG   4
#define PLY_UCHAR      5
#define PLY_USHORT     6
#define PLY_UINT       7
#define PLY_ULONGLONG  8
#define PLY_FLOAT      9
#define PLY_DOUBLE     10
#define PLY_INT_8      11
#define PLY_UINT_8     12
#define PLY_INT_16     13
#define PLY_UINT_16    14
#define PLY_INT_32     15
#define PLY_UINT_32    16
#define PLY_INT_64     17
#define PLY_UINT_64    18
#define PLY_FLOAT_32   19
#define PLY_FLOAT_64   20

#define PLY_END_TYPE   21

#define  PLY_SCALAR  0
#define  PLY_LIST    1

#define PLY_STRIP_COMMENT_HEADER 0

	const std::string PlyTypes[]
	{
		"start type" ,
		"char" ,
		"short" ,
		"int" ,
		"long long " ,
		"unsigned char" ,
		"unsigned short" ,
		"unsigned int" ,
		"unsigned long long" ,
		"float" ,
		"double" ,
		"int8" ,
		"unsigned int8" ,
		"int16" ,
		"unsigned int16" ,
		"int32" ,
		"unsigned int32" ,
		"int64" ,
		"unsigned int64" ,
		"float32" ,
		"float64"
	};

	/* description of a property */
	struct PlyProperty
	{
		std::string name;                     /* property name */
		int external_type;                    /* file's data type */
		int internal_type;                    /* program's data type */
		int offset;                           /* offset bytes of prop in a struct */

		int is_list;                          /* 1 = list, 0 = scalar */
		int count_external;                   /* file's count type */
		int count_internal;                   /* program's count type */
		int count_offset;                     /* offset byte for list count */

		PlyProperty( const std::string &n , int et , int it , int o , int il=0 , int ce=0 , int ci=0 , int co=0 ) : name(n) , external_type(et) , internal_type(it) , offset(o) , is_list(il) , count_external(ce) , count_internal(ci) , count_offset(co){ }
		PlyProperty( const std::string &n ) : PlyProperty( n , 0 , 0 , 0 , 0 , 0 , 0 , 0 ){ }
		PlyProperty( void ) : external_type(0) , internal_type(0) , offset(0) , is_list(0) , count_external(0) , count_internal(0) , count_offset(0){ }

		void write( BinaryStream &stream ) const
		{
			stream.write( name );
			stream.write( external_type );
			stream.write( offset );
			stream.write( is_list );
			stream.write( count_external );
			stream.write( count_internal );
			stream.write( count_offset );
		}
		void read( BinaryStream &stream )
		{
			if( !stream.read( name ) ) MK_THROW( "Failed to read name" );
			if( !stream.read( external_type ) ) MK_THROW( "Failed to read external_type" );
			if( !stream.read( offset ) ) MK_THROW( "Failed to read offset" );
			if( !stream.read( is_list ) ) MK_THROW( "Failed to read is_list" );
			if( !stream.read( count_external ) ) MK_THROW( "Failed to read count_external" );
			if( !stream.read( count_internal ) ) MK_THROW( "Failed to read count_internal" );
			if( !stream.read( count_offset ) ) MK_THROW( "Failed to read count_offset" );
		}
	};

	inline std::ostream &operator << ( std::ostream &os , PlyProperty p )
	{
		if( p.is_list ) return os << "{ " << p.name << " , " << PlyTypes[ p.count_external ] << " -> " << PlyTypes[ p.count_internal ] << " , " << PlyTypes[ p.external_type ] << " -> " << PlyTypes[ p.internal_type ] << " , " << p.offset << " }";
		else            return os << "{ " << p.name << " , " <<                                                                                    PlyTypes[ p.external_type ] << " -> " << PlyTypes[ p.internal_type ] << " , " << p.offset << " }";
	}

	struct PlyStoredProperty
	{
		PlyProperty prop ; char store;
		PlyStoredProperty( void ){ }
		PlyStoredProperty( const PlyProperty &p , char s ) : prop(p) , store(s){ }
	};

	/* description of an element */
	struct PlyElement
	{
		std::string name;             /* element name */
		size_t num;                   /* number of elements in this object */
		int size;                     /* size of element (bytes) or -1 if variable */
		std::vector< PlyStoredProperty > props; /* list of properties in the file */
		int other_offset;             /* offset to un-asked-for props, or -1 if none*/
		int other_size;               /* size of other_props structure */
		inline PlyProperty *find_property( const std::string &prop_name , int &index );
	};

	/* describes other properties in an element */
	struct PlyOtherProp
	{
		std::string name;                   /* element name */
		int size;                           /* size of other_props */
		std::vector< PlyProperty > props;   /* list of properties in other_props */
	};

	/* storing other_props for an other element */
	struct OtherData
	{
		void *other_props;
		OtherData( void ) : other_props(NULL){ }
		~OtherData( void ){ if( other_props ) free( other_props ); }
	};

	/* data for one "other" element */
	struct OtherElem
	{
		std::string elem_name;                /* names of other elements */
		std::vector< OtherData > other_data;  /* actual property data for the elements */
		PlyOtherProp other_props;             /* description of the property data */
	};

	/* "other" elements, not interpreted by user */
	struct PlyOtherElems
	{
		std::vector< OtherElem > other_list; /* list of data for other elements */
	};

	/* description of PLY file */
	struct PlyFile
	{
		FILE *fp;                            /* file pointer */
		int file_type;                       /* ascii or binary */
		float version;                       /* version number of file */
		std::vector< PlyElement > elems;     /* list of elements of object */
		std::vector< std::string > comments; /* list of comments */
		std::vector< std::string > obj_info; /* list of object info items */
		PlyElement *which_elem;              /* which element we're currently writing */
		PlyOtherElems *other_elems;         /* "other" elements from a PLY file */

		static inline PlyFile *Write( const std::string & , const std::vector< std::string > & , int   , float & );
		static inline PlyFile *Read ( const std::string & ,       std::vector< std::string > & , int & , float & );

		PlyFile( FILE *f ) : fp(f) , other_elems(NULL) , version(1.) { }
		~PlyFile( void ){ if( fp ) fclose(fp) ; if(other_elems) delete other_elems; }

		inline void describe_element ( const std::string & , size_t , int , const PlyProperty * );
		inline void describe_property( const std::string & , const PlyProperty * );
		inline void describe_other_elements( PlyOtherElems * );
		inline PlyElement *find_element( const std::string & );
		inline void element_count( const std::string & , size_t );
		inline void header_complete( void );
		inline void put_element_setup( const std::string & );
		inline void put_element ( void * );
		inline void put_comment ( const std::string & );
		inline void put_obj_info( const std::string & );
		inline void put_other_elements( void );
		inline void add_element ( const std::vector< std::string > & );
		inline void add_property( const std::vector< std::string > & );
		inline void add_comment ( const std::string & );
		inline void add_obj_info( const std::string & );

		inline std::vector< PlyProperty > get_element_description( const std::string & , size_t & );
		inline void get_element_setup( const std::string & , int , PlyProperty * );
		inline int get_property( const std::string & , const PlyProperty * );
		inline void describe_other_properties( const PlyOtherProp & , int );
		inline bool set_other_properties( const std::string & , int , PlyOtherProp & );
		inline void get_element( void * );
		inline std::vector< std::string > &get_comments( void );
		inline std::vector< std::string > &get_obj_info( void );
		inline void get_info( float & , int & );
		inline PlyOtherElems *get_other_element( std::string & , size_t );
	protected:
		inline void _ascii_get_element ( void * );
		inline void _binary_get_element( void * );
		static inline PlyFile *_Write( FILE * , const std::vector< std::string > & , int );
		static inline PlyFile *_Read ( FILE * ,       std::vector< std::string > & );
	};

#include "PlyFile.inl"
}
#endif // PLY_FILE_INCLUDED
