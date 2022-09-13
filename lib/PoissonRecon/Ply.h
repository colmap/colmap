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
	  
	   Copyright (c) 1994 The Board of Trustees of The Leland Stanford
	   Junior University.  All rights reserved.   
	   
		Permission to use, copy, modify and distribute this software and its   
		documentation for any purpose is hereby granted without fee, provided   
		that the above copyright notice and this permission notice appear in   
		all copies of this software and that you do not sell the software.   
		
		 THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,   
		 EXPRESS, IMPLIED OR OTHERWISE, INCLUDING WITHOUT LIMITATION, ANY   
		 WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.   
		 
*/

#ifndef __PLY_H__
#define __PLY_H__

#define USE_PLY_WRAPPER 1

#ifndef WIN32
#define _strdup strdup
#endif

#ifdef __cplusplus
extern "C" {
#endif
	
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
    
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
#define PLY_UCHAR      4
#define PLY_USHORT     5
#define PLY_UINT       6
#define PLY_FLOAT      7
#define PLY_DOUBLE     8
#define PLY_INT_8      9
#define PLY_UINT_8     10
#define PLY_INT_16     11
#define PLY_UINT_16    12
#define PLY_INT_32     13
#define PLY_UINT_32    14
#define PLY_FLOAT_32   15
#define PLY_FLOAT_64   16
	
#define PLY_END_TYPE   17
	
#define  PLY_SCALAR  0
#define  PLY_LIST    1
	
#define PLY_STRIP_COMMENT_HEADER 0

typedef struct PlyProperty {    /* description of a property */
	
	char *name;                           /* property name */
	int external_type;                    /* file's data type */
	int internal_type;                    /* program's data type */
	int offset;                           /* offset bytes of prop in a struct */
	
	int is_list;                          /* 1 = list, 0 = scalar */
	int count_external;                   /* file's count type */
	int count_internal;                   /* program's count type */
	int count_offset;                     /* offset byte for list count */
	
} PlyProperty;

typedef struct PlyElement {     /* description of an element */
	char *name;                   /* element name */
	int num;                      /* number of elements in this object */
	int size;                     /* size of element (bytes) or -1 if variable */
	int nprops;                   /* number of properties for this element */
	PlyProperty **props;          /* list of properties in the file */
	char *store_prop;             /* flags: property wanted by user? */
	int other_offset;             /* offset to un-asked-for props, or -1 if none*/
	int other_size;               /* size of other_props structure */
} PlyElement;

typedef struct PlyOtherProp {   /* describes other properties in an element */
	char *name;                   /* element name */
	int size;                     /* size of other_props */
	int nprops;                   /* number of properties in other_props */
	PlyProperty **props;          /* list of properties in other_props */
} PlyOtherProp;

typedef struct OtherData { /* for storing other_props for an other element */
	void *other_props;
} OtherData;

typedef struct OtherElem {     /* data for one "other" element */
	char *elem_name;             /* names of other elements */
	int elem_count;              /* count of instances of each element */
	OtherData **other_data;      /* actual property data for the elements */
	PlyOtherProp *other_props;   /* description of the property data */
} OtherElem;

typedef struct PlyOtherElems {  /* "other" elements, not interpreted by user */
	int num_elems;                /* number of other elements */
	OtherElem *other_list;        /* list of data for other elements */
} PlyOtherElems;

typedef struct PlyFile {        /* description of PLY file */
	FILE *fp;                     /* file pointer */
	int file_type;                /* ascii or binary */
	float version;                /* version number of file */
	int nelems;                   /* number of elements of object */
	PlyElement **elems;           /* list of elements */
	int num_comments;             /* number of comments */
	char **comments;              /* list of comments */
	int num_obj_info;             /* number of items of object information */
	char **obj_info;              /* list of object info items */
	PlyElement *which_elem;       /* which element we're currently writing */
	PlyOtherElems *other_elems;   /* "other" elements from a PLY file */
} PlyFile;
	
	/* memory allocation */
extern char *my_alloc();
#define myalloc(mem_size) my_alloc((mem_size), __LINE__, __FILE__)

#ifndef ALLOCN
#define REALLOCN(PTR,TYPE,OLD_N,NEW_N)							\
{										\
	if ((OLD_N) == 0)                                           		\
{   ALLOCN((PTR),TYPE,(NEW_N));}                            		\
	else									\
{								    		\
	(PTR) = (TYPE *)realloc((PTR),(NEW_N)*sizeof(TYPE));			\
	if (((PTR) == NULL) && ((NEW_N) != 0))					\
{									\
	fprintf(stderr, "Memory reallocation failed on line %d in %s\n", 	\
	__LINE__, __FILE__);                             		\
	fprintf(stderr, "  tried to reallocate %d->%d\n",       		\
	(OLD_N), (NEW_N));                              		\
	exit(-1);								\
}									\
	if ((NEW_N)>(OLD_N))							\
	memset((char *)(PTR)+(OLD_N)*sizeof(TYPE), 0,			\
	((NEW_N)-(OLD_N))*sizeof(TYPE));				\
}										\
}

#define  ALLOCN(PTR,TYPE,N) 					\
{ (PTR) = (TYPE *) calloc(((unsigned)(N)),sizeof(TYPE));\
	if ((PTR) == NULL) {    				\
	fprintf(stderr, "Memory allocation failed on line %d in %s\n", \
	__LINE__, __FILE__);                           \
	exit(-1);                                             \
	}							\
}


#define FREE(PTR)  { free((PTR)); (PTR) = NULL; }
#endif


/*** delcaration of routines ***/

extern PlyFile *ply_write(FILE *, int, const char **, int);
extern PlyFile *ply_open_for_writing(char *, int, const char **, int, float *);
extern void ply_describe_element(PlyFile *, char *, int, int, PlyProperty *);
extern void ply_describe_property(PlyFile *, const char *, PlyProperty *);
extern void ply_element_count(PlyFile *, const char *, int);
extern void ply_header_complete(PlyFile *);
extern void ply_put_element_setup(PlyFile *, const char *);
extern void ply_put_element(PlyFile *, void *);
extern void ply_put_comment(PlyFile *, char *);
extern void ply_put_obj_info(PlyFile *, char *);
extern PlyFile *ply_read(FILE *, int *, char ***);
extern PlyFile *ply_open_for_reading( char *, int *, char ***, int *, float *);
extern PlyProperty **ply_get_element_description(PlyFile *, char *, int*, int*);
extern void ply_get_element_setup( PlyFile *, char *, int, PlyProperty *);
extern int ply_get_property(PlyFile *, char *, PlyProperty *);
extern PlyOtherProp *ply_get_other_properties(PlyFile *, char *, int);
extern void ply_get_element(PlyFile *, void *);
extern char **ply_get_comments(PlyFile *, int *);
extern char **ply_get_obj_info(PlyFile *, int *);
extern void ply_close(PlyFile *);
extern void ply_get_info(PlyFile *, float *, int *);
extern PlyOtherElems *ply_get_other_element (PlyFile *, char *, int);
extern void ply_describe_other_elements ( PlyFile *, PlyOtherElems *);
extern void ply_put_other_elements (PlyFile *);
extern void ply_free_other_elements (PlyOtherElems *);
extern void ply_describe_other_properties(PlyFile *, PlyOtherProp *, int);

extern int equal_strings(const char *, const char *);

#ifdef __cplusplus
}
#endif
#include "Geometry.h"
#include <vector>

template< class Real > int PLYType( void );
template<> inline int PLYType< int           >( void ){ return PLY_INT   ; }
template<> inline int PLYType<          char >( void ){ return PLY_CHAR  ; }
template<> inline int PLYType< unsigned char >( void ){ return PLY_UCHAR ; }
template<> inline int PLYType<        float  >( void ){ return PLY_FLOAT ; }
template<> inline int PLYType<        double >( void ){ return PLY_DOUBLE; }
template< class Real > inline int PLYType( void ){ fprintf( stderr , "[ERROR] Unrecognized type\n" ) , exit( 0 ); }

typedef struct PlyFace
{
	unsigned char nr_vertices;
	int *vertices;
	int segment;
} PlyFace;
static PlyProperty face_props[] =
{
	{ _strdup( "vertex_indices" ) , PLY_INT , PLY_INT , offsetof( PlyFace , vertices ) , 1 , PLY_UCHAR, PLY_UCHAR , offsetof(PlyFace,nr_vertices) },
};


///////////////////
// PlyVertexType //
///////////////////

// The "Wrapper" class indicates the class to cast to/from in order to support linear operations.
template< class Real >
class PlyVertex
{
public:
	typedef PlyVertex Wrapper;

	const static int ReadComponents=3;
	const static int WriteComponents=3;
	static PlyProperty ReadProperties[];
	static PlyProperty WriteProperties[];

	Point3D< Real > point;

	PlyVertex( void ) { ; }
	PlyVertex( Point3D< Real > p ) { point=p; }
	PlyVertex operator + ( PlyVertex p ) const { return PlyVertex( point+p.point ); }
	PlyVertex operator - ( PlyVertex p ) const { return PlyVertex( point-p.point ); }
	template< class _Real > PlyVertex operator * ( _Real s ) const { return PlyVertex( point*s ); }
	template< class _Real > PlyVertex operator / ( _Real s ) const { return PlyVertex( point/s ); }
	PlyVertex& operator += ( PlyVertex p ) { point += p.point ; return *this; }
	PlyVertex& operator -= ( PlyVertex p ) { point -= p.point ; return *this; }
	template< class _Real > PlyVertex& operator *= ( _Real s ) { point *= s ; return *this; }
	template< class _Real > PlyVertex& operator /= ( _Real s ) { point /= s ; return *this; }
};
template< class Real , class _Real > PlyVertex< Real > operator * ( XForm4x4< _Real > xForm , PlyVertex< Real > v ) { return PlyVertex< Real >( xForm * v.point ); }
template< class Real > PlyProperty PlyVertex< Real >::ReadProperties[]=
{
	{ _strdup( "x" ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyVertex , point.coords[0] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "y" ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyVertex , point.coords[1] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "z" ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyVertex , point.coords[2] ) ) , 0 , 0 , 0 , 0 }
};
template< class Real > PlyProperty PlyVertex< Real >::WriteProperties[]=
{
	{ _strdup( "x" ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyVertex , point.coords[0] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "y" ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyVertex , point.coords[1] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "z" ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyVertex , point.coords[2] ) ) , 0 , 0 , 0 , 0 }
};
template< class Real >
class PlyValueVertex
{
public:
	typedef PlyValueVertex Wrapper;

	const static int ReadComponents=4;
	const static int WriteComponents=4;
	static PlyProperty ReadProperties[];
	static PlyProperty WriteProperties[];

	Point3D<Real> point;
	Real value;

	PlyValueVertex( void ) : value( Real(0) ) { ; }
	PlyValueVertex( Point3D< Real > p , Real v ) : point(p) , value(v) { ; }
	PlyValueVertex operator + ( PlyValueVertex p ) const { return PlyValueVertex( point+p.point , value+p.value ); }
	PlyValueVertex operator - ( PlyValueVertex p ) const { return PlyValueVertex( point-p.value , value-p.value ); }
	template< class _Real > PlyValueVertex operator * ( _Real s ) const { return PlyValueVertex( point*s , Real(value*s) ); }
	template< class _Real > PlyValueVertex operator / ( _Real s ) const { return PlyValueVertex( point/s , Real(value/s) ); }
	PlyValueVertex& operator += ( PlyValueVertex p ) { point += p.point , value += p.value ; return *this; }
	PlyValueVertex& operator -= ( PlyValueVertex p ) { point -= p.point , value -= p.value ; return *this; }
	template< class _Real > PlyValueVertex& operator *= ( _Real s ) { point *= s , value *= Real(s) ; return *this; }
	template< class _Real > PlyValueVertex& operator /= ( _Real s ) { point /= s , value /= Real(s) ; return *this; }
};
template< class Real , class _Real > PlyValueVertex< Real > operator * ( XForm4x4< _Real > xForm , PlyValueVertex< Real > v ) { return PlyValueVertex< Real >( xForm * v.point , v.value ); }
template< class Real > PlyProperty PlyValueVertex< Real >::ReadProperties[]=
{
	{ _strdup( "x"     ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyValueVertex , point.coords[0] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "y"     ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyValueVertex , point.coords[1] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "z"     ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyValueVertex , point.coords[2] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "value" ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyValueVertex , value           ) ) , 0 , 0 , 0 , 0 }
};
template< class Real > PlyProperty PlyValueVertex< Real >::WriteProperties[]=
{
	{ _strdup( "x"     ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyValueVertex , point.coords[0] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "y"     ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyValueVertex , point.coords[1] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "z"     ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyValueVertex , point.coords[2] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "value" ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyValueVertex , value           ) ) , 0 , 0 , 0 , 0 }
};
template< class Real >
class PlyOrientedVertex
{
public:
	typedef PlyOrientedVertex Wrapper;

	const static int ReadComponents=6;
	const static int WriteComponents=6;
	static PlyProperty ReadProperties[];
	static PlyProperty WriteProperties[];

	Point3D<Real> point , normal;

	PlyOrientedVertex( void ) { ; }
	PlyOrientedVertex( Point3D< Real > p , Point3D< Real > n ) : point(p) , normal(n) { ; }
  	PlyOrientedVertex operator + ( PlyOrientedVertex p ) const { return PlyOrientedVertex( point+p.point , normal+p.normal ); }
	PlyOrientedVertex operator - ( PlyOrientedVertex p ) const { return PlyOrientedVertex( point-p.value , normal-p.normal ); }
	template< class _Real > PlyOrientedVertex operator * ( _Real s ) const { return PlyOrientedVertex( point*s , normal*s ); }
	template< class _Real > PlyOrientedVertex operator / ( _Real s ) const { return PlyOrientedVertex( point/s , normal/s ); }
	PlyOrientedVertex& operator += ( PlyOrientedVertex p ) { point += p.point , normal += p.normal ; return *this; }
	PlyOrientedVertex& operator -= ( PlyOrientedVertex p ) { point -= p.point , normal -= p.normal ; return *this; }
	template< class _Real > PlyOrientedVertex& operator *= ( _Real s ) { point *= s , normal *= s ; return *this; }
	template< class _Real > PlyOrientedVertex& operator /= ( _Real s ) { point /= s , normal /= s ; return *this; }
};
template< class Real , class _Real > PlyOrientedVertex< Real > operator * ( XForm4x4< _Real > xForm , PlyOrientedVertex< Real > v ) { return PlyOrientedVertex< Real >( xForm * v.point , xForm.inverse().transpose() * v.normal ); }
template< class Real > PlyProperty PlyOrientedVertex< Real >::ReadProperties[]=
{
	{ _strdup( "x"  ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyOrientedVertex ,  point.coords[0] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "y"  ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyOrientedVertex ,  point.coords[1] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "z"  ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyOrientedVertex ,  point.coords[2] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "nx" ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyOrientedVertex , normal.coords[0] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "ny" ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyOrientedVertex , normal.coords[1] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "nz" ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyOrientedVertex , normal.coords[2] ) ) , 0 , 0 , 0 , 0 }
};
template< class Real > PlyProperty PlyOrientedVertex< Real >::WriteProperties[]=
{
	{ _strdup( "x"  ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyOrientedVertex ,  point.coords[0] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "y"  ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyOrientedVertex ,  point.coords[1] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "z"  ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyOrientedVertex ,  point.coords[2] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "nx" ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyOrientedVertex , normal.coords[0] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "ny" ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyOrientedVertex , normal.coords[1] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "nz" ) , PLYType< Real >() , PLYType< Real >() , int( offsetof( PlyOrientedVertex , normal.coords[2] ) ) , 0 , 0 , 0 , 0 }
};
template< class Real >
class PlyColorVertex
{
public:
	struct _PlyColorVertex
	{
		Point3D< Real > point , color;
		_PlyColorVertex( void ) { ; }
		_PlyColorVertex( Point3D< Real > p , Point3D< Real > c ) : point(p) , color(c) { ; }
		_PlyColorVertex( PlyColorVertex< Real > p ){ point = p.point ; for( int c=0 ; c<3 ; c++ ) color[c] = (Real) p.color[c]; }
		operator PlyColorVertex< Real > ()
		{
			PlyColorVertex< Real > p;
			p.point = point;
			for( int c=0 ; c<3 ; c++ ) p.color[c] = (unsigned char)std::max< int >( 0 , std::min< int >( 255 , (int)( color[c]+0.5 ) ) );
			return p;
		}

	  	_PlyColorVertex operator + ( _PlyColorVertex p ) const { return _PlyColorVertex( point+p.point , color+p.color ); }
		_PlyColorVertex operator - ( _PlyColorVertex p ) const { return _PlyColorVertex( point-p.value , color-p.color ); }
		template< class _Real > _PlyColorVertex operator * ( _Real s ) const { return _PlyColorVertex( point*s , color*s ); }
		template< class _Real > _PlyColorVertex operator / ( _Real s ) const { return _PlyColorVertex( point/s , color/s ); }
		_PlyColorVertex& operator += ( _PlyColorVertex p ) { point += p.point , color += p.color ; return *this; }
		_PlyColorVertex& operator -= ( _PlyColorVertex p ) { point -= p.point , color -= p.color ; return *this; }
		template< class _Real > _PlyColorVertex& operator *= ( _Real s ) { point *= s , color *= s ; return *this; }
		template< class _Real > _PlyColorVertex& operator /= ( _Real s ) { point /= s , color /= s ; return *this; }
	};

	typedef _PlyColorVertex Wrapper;

	const static int ReadComponents=9;
	const static int WriteComponents=6;
	static PlyProperty ReadProperties[];
	static PlyProperty WriteProperties[];

	Point3D< Real > point;
	unsigned char color[3];

	operator Point3D< Real >& (){ return point; }
	operator const Point3D< Real >& () const { return point; }
	PlyColorVertex( void ) { point.coords[0] = point.coords[1] = point.coords[2] = 0 , color[0] = color[1] = color[2] = 0; }
	PlyColorVertex( const Point3D<Real>& p ) { point=p; }
	PlyColorVertex( const Point3D< Real >& p , const unsigned char c[3] ) { point = p , color[0] = c[0] , color[1] = c[1] , color[2] = c[2]; }
};
template< class Real , class _Real > PlyColorVertex< Real > operator * ( XForm4x4< _Real > xForm , PlyColorVertex< Real > v ) { return PlyColorVertex< Real >( xForm * v.point , v.color ); }

template< class Real > PlyProperty PlyColorVertex< Real >::ReadProperties[]=
{
	{ _strdup( "x"     ) , PLYType<          Real >() , PLYType<          Real >(), int( offsetof( PlyColorVertex , point.coords[0] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "y"     ) , PLYType<          Real >() , PLYType<          Real >(), int( offsetof( PlyColorVertex , point.coords[1] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "z"     ) , PLYType<          Real >() , PLYType<          Real >(), int( offsetof( PlyColorVertex , point.coords[2] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "red"   ) , PLYType< unsigned char >() , PLYType< unsigned char >(), int( offsetof( PlyColorVertex ,        color[0] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "green" ) , PLYType< unsigned char >() , PLYType< unsigned char >(), int( offsetof( PlyColorVertex ,        color[1] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "blue"  ) , PLYType< unsigned char >() , PLYType< unsigned char >(), int( offsetof( PlyColorVertex ,        color[2] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "r"     ) , PLYType< unsigned char >() , PLYType< unsigned char >(), int( offsetof( PlyColorVertex ,        color[0] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "g"     ) , PLYType< unsigned char >() , PLYType< unsigned char >(), int( offsetof( PlyColorVertex ,        color[1] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "b"     ) , PLYType< unsigned char >() , PLYType< unsigned char >(), int( offsetof( PlyColorVertex ,        color[2] ) ) , 0 , 0 , 0 , 0 }
};
template< class Real > PlyProperty PlyColorVertex< Real >::WriteProperties[]=
{
	{ _strdup( "x"     ) , PLYType<          Real >() , PLYType<          Real >(), int( offsetof( PlyColorVertex , point.coords[0] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "y"     ) , PLYType<          Real >() , PLYType<          Real >(), int( offsetof( PlyColorVertex , point.coords[1] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "z"     ) , PLYType<          Real >() , PLYType<          Real >(), int( offsetof( PlyColorVertex , point.coords[2] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "red"   ) , PLYType< unsigned char >() , PLYType< unsigned char >(), int( offsetof( PlyColorVertex ,        color[0] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "green" ) , PLYType< unsigned char >() , PLYType< unsigned char >(), int( offsetof( PlyColorVertex ,        color[1] ) ) , 0 , 0 , 0 , 0 },
	{ _strdup( "blue"  ) , PLYType< unsigned char >() , PLYType< unsigned char >(), int( offsetof( PlyColorVertex ,        color[2] ) ) , 0 , 0 , 0 , 0 }
};
template< class Real >
class PlyColorAndValueVertex
{
public:
	struct _PlyColorAndValueVertex
	{
		Point3D< Real > point , color;
		Real value;
		_PlyColorAndValueVertex( void ) : value(0) { ; }
		_PlyColorAndValueVertex( Point3D< Real > p , Point3D< Real > c , Real v ) : point(p) , color(c) , value(v) { ; }
		_PlyColorAndValueVertex( PlyColorAndValueVertex< Real > p ){ point = p.point ; for( int c=0 ; c<3 ; c++ ) color[c] = (Real) p.color[c] ; value = p.value; }
		operator PlyColorAndValueVertex< Real > ()
		{
			PlyColorAndValueVertex< Real > p;
			p.point = point;
			for( int c=0 ; c<3 ; c++ ) p.color[c] = (unsigned char)std::max< int >( 0 , std::min< int >( 255 , (int)( color[c]+0.5 ) ) );
			p.value = value;
			return p;
		}

	  	_PlyColorAndValueVertex operator + ( _PlyColorAndValueVertex p ) const { return _PlyColorAndValueVertex( point+p.point , color+p.color , value+p.value ); }
		_PlyColorAndValueVertex operator - ( _PlyColorAndValueVertex p ) const { return _PlyColorAndValueVertex( point-p.value , color-p.color , value+p.value ); }
		template< class _Real > _PlyColorAndValueVertex operator * ( _Real s ) const { return _PlyColorAndValueVertex( point*s , color*s , value*s ); }
		template< class _Real > _PlyColorAndValueVertex operator / ( _Real s ) const { return _PlyColorAndValueVertex( point/s , color/s , value/s ); }
		_PlyColorAndValueVertex& operator += ( _PlyColorAndValueVertex p ) { point += p.point , color += p.color , value += p.value ; return *this; }
		_PlyColorAndValueVertex& operator -= ( _PlyColorAndValueVertex p ) { point -= p.point , color -= p.color , value -= p.value ; return *this; }
		template< class _Real > _PlyColorAndValueVertex& operator *= ( _Real s ) { point *= s , color *= s , value *= (Real)s ; return *this; }
		template< class _Real > _PlyColorAndValueVertex& operator /= ( _Real s ) { point /= s , color /= s , value /= (Real)s ; return *this; }
	};

	typedef _PlyColorAndValueVertex Wrapper;

	const static int ReadComponents=10;
	const static int WriteComponents=7;
	static PlyProperty ReadProperties[];
	static PlyProperty WriteProperties[];

	Point3D< Real > point;
	unsigned char color[3];
	Real value;

	operator Point3D< Real >& (){ return point; }
	operator const Point3D< Real >& () const { return point; }
	PlyColorAndValueVertex( void ) { point.coords[0] = point.coords[1] = point.coords[2] = (Real)0 , color[0] = color[1] = color[2] = 0 , value = (Real)0; }
	PlyColorAndValueVertex( const Point3D< Real >& p ) { point=p; }
	PlyColorAndValueVertex( const Point3D< Real >& p , const unsigned char c[3] , Real v) { point = p , color[0] = c[0] , color[1] = c[1] , color[2] = c[2] , value = v; }
};
template< class Real , class _Real > PlyColorAndValueVertex< Real > operator * ( XForm4x4< _Real > xForm , PlyColorAndValueVertex< Real > v ) { return PlyColorAndValueVertex< Real >( xForm * v.point , v.color , v.value ); }
template< class Real > PlyProperty PlyColorAndValueVertex< Real >::ReadProperties[]=
{
	{ _strdup( "x"     ) , PLYType<          Real >() , PLYType<          Real >() , int( offsetof( PlyColorAndValueVertex , point.coords[0] ) ) , 0 , 0 , 0 , 0 } ,
	{ _strdup( "y"     ) , PLYType<          Real >() , PLYType<          Real >() , int( offsetof( PlyColorAndValueVertex , point.coords[1] ) ) , 0 , 0 , 0 , 0 } ,
	{ _strdup( "z"     ) , PLYType<          Real >() , PLYType<          Real >() , int( offsetof( PlyColorAndValueVertex , point.coords[2] ) ) , 0 , 0 , 0 , 0 } ,
	{ _strdup( "value" ) , PLYType<          Real >() , PLYType<          Real >() , int( offsetof( PlyColorAndValueVertex ,        value    ) ) , 0 , 0 , 0 , 0 } ,
	{ _strdup( "red"   ) , PLYType< unsigned char >() , PLYType< unsigned char >() , int( offsetof( PlyColorAndValueVertex ,        color[0] ) ) , 0 , 0 , 0 , 0 } ,
	{ _strdup( "green" ) , PLYType< unsigned char >() , PLYType< unsigned char >() , int( offsetof( PlyColorAndValueVertex ,        color[1] ) ) , 0 , 0 , 0 , 0 } ,
	{ _strdup( "blue"  ) , PLYType< unsigned char >() , PLYType< unsigned char >() , int( offsetof( PlyColorAndValueVertex ,        color[2] ) ) , 0 , 0 , 0 , 0 } ,
	{ _strdup( "r"     ) , PLYType< unsigned char >() , PLYType< unsigned char >() , int( offsetof( PlyColorAndValueVertex ,        color[0] ) ) , 0 , 0 , 0 , 0 } ,
	{ _strdup( "g"     ) , PLYType< unsigned char >() , PLYType< unsigned char >() , int( offsetof( PlyColorAndValueVertex ,        color[1] ) ) , 0 , 0 , 0 , 0 } ,
	{ _strdup( "b"     ) , PLYType< unsigned char >() , PLYType< unsigned char >() , int( offsetof( PlyColorAndValueVertex ,        color[2] ) ) , 0 , 0 , 0 , 0 }
};
template< class Real > PlyProperty PlyColorAndValueVertex< Real >::WriteProperties[]=
{
	{ _strdup( "x"     ) , PLYType<          Real >() , PLYType<          Real >() , int( offsetof( PlyColorAndValueVertex , point.coords[0] ) ) , 0 , 0 , 0 , 0 } ,
	{ _strdup( "y"     ) , PLYType<          Real >() , PLYType<          Real >() , int( offsetof( PlyColorAndValueVertex , point.coords[1] ) ) , 0 , 0 , 0 , 0 } ,
	{ _strdup( "z"     ) , PLYType<          Real >() , PLYType<          Real >() , int( offsetof( PlyColorAndValueVertex , point.coords[2] ) ) , 0 , 0 , 0 , 0 } ,
	{ _strdup( "value" ) , PLYType<          Real >() , PLYType<          Real >() , int( offsetof( PlyColorAndValueVertex ,        value    ) ) , 0 , 0 , 0 , 0 } ,
	{ _strdup( "red"   ) , PLYType< unsigned char >() , PLYType< unsigned char >() , int( offsetof( PlyColorAndValueVertex ,        color[0] ) ) , 0 , 0 , 0 , 0 } ,
	{ _strdup( "green" ) , PLYType< unsigned char >() , PLYType< unsigned char >() , int( offsetof( PlyColorAndValueVertex ,        color[1] ) ) , 0 , 0 , 0 , 0 } ,
	{ _strdup( "blue"  ) , PLYType< unsigned char >() , PLYType< unsigned char >() , int( offsetof( PlyColorAndValueVertex ,        color[2] ) ) , 0 , 0 , 0 , 0 }
};

template< class Vertex , class Real >
int PlyWritePolygons( char* fileName , CoredMeshData< Vertex >*  mesh , int file_type , const Point3D< float >& translate , float scale , char** comments=NULL , int commentNum=0 , XForm4x4< Real > xForm=XForm4x4< Real >::Identity() );

template< class Vertex , class Real >
int PlyWritePolygons( char* fileName , CoredMeshData< Vertex >*  mesh , int file_type , char** comments=NULL , int commentNum=0 , XForm4x4< Real > xForm=XForm4x4< Real >::Identity() );

inline bool PlyReadHeader( char* fileName , PlyProperty* properties , int propertyNum , bool* readFlags , int& file_type )
{
	int nr_elems;
	char **elist;
	float version;
	PlyFile* ply;
	char* elem_name;
	int num_elems;
	int nr_props;
	PlyProperty** plist;

	ply = ply_open_for_reading( fileName , &nr_elems , &elist , &file_type , &version );
	if( !ply ) return false;

	for( int i=0 ; i<nr_elems ; i++ )
	{
		elem_name = elist[i];
		plist = ply_get_element_description( ply , elem_name , &num_elems , &nr_props );
		if( !plist )
		{
			for( int i=0 ; i<nr_elems ; i++ )
			{
				free( ply->elems[i]->name );
				free( ply->elems[i]->store_prop );
				for( int j=0 ; j<ply->elems[i]->nprops ; j++ )
				{
					free( ply->elems[i]->props[j]->name );
					free( ply->elems[i]->props[j] );
				}
				free( ply->elems[i]->props );
			}
			for( int i=0 ; i<nr_elems ; i++ ) free( ply->elems[i] );
			free( ply->elems );
			for( int i=0 ; i<ply->num_comments ; i++ ) free( ply->comments[i] );
			free( ply->comments );
			for( int i=0 ; i<ply->num_obj_info ; i++ ) free( ply->obj_info[i] );
			free( ply->obj_info );
			ply_free_other_elements( ply->other_elems );
			
			for( int i=0 ; i<nr_elems ; i++ ) free( elist[i] );
			free( elist );
			ply_close( ply );
			return 0;
		}		
		if( equal_strings( "vertex" , elem_name ) )
			for( int i=0 ; i<propertyNum ; i++ )
				if( readFlags ) readFlags[i] = ply_get_property( ply , elem_name , &properties[i] )!=0;

		for( int j=0 ; j<nr_props ; j++ )
		{
			free( plist[j]->name );
			free( plist[j] );
		}
		free( plist );
	}  // for each type of element
	
	for( int i=0 ; i<nr_elems ; i++ )
	{
		free( ply->elems[i]->name );
		free( ply->elems[i]->store_prop );
		for( int j=0 ; j<ply->elems[i]->nprops ; j++ )
		{
			free( ply->elems[i]->props[j]->name );
			free( ply->elems[i]->props[j] );
		}
		if( ply->elems[i]->props && ply->elems[i]->nprops ) free(ply->elems[i]->props);
	}
	for( int i=0 ; i<nr_elems ; i++ ) free(ply->elems[i]);
	free( ply->elems) ;
	for( int i=0 ; i<ply->num_comments ; i++ ) free( ply->comments[i] );
	free( ply->comments );
	for( int i=0 ; i<ply->num_obj_info ; i++ ) free( ply->obj_info[i] );
	free( ply->obj_info );
	ply_free_other_elements(ply->other_elems);
	
	
	for( int i=0 ; i<nr_elems ; i++ ) free( elist[i] );
	free( elist );
	ply_close( ply );
	return true;
}
inline bool PlyReadHeader( char* fileName , PlyProperty* properties , int propertyNum , bool* readFlags )
{
	int file_type;
	return PlyReadHeader( fileName , properties , propertyNum , readFlags , file_type );
}


template<class Vertex>
int PlyReadPolygons(char* fileName,
					std::vector<Vertex>& vertices,std::vector<std::vector<int> >& polygons,
					PlyProperty* properties,int propertyNum,
					int& file_type,
					char*** comments=NULL,int* commentNum=NULL , bool* readFlags=NULL );

template<class Vertex>
int PlyWritePolygons(char* fileName,
					 const std::vector<Vertex>& vertices,const std::vector<std::vector<int> >& polygons,
					 PlyProperty* properties,int propertyNum,
					 int file_type,
					 char** comments=NULL,const int& commentNum=0);

template<class Vertex>
int PlyWritePolygons(char* fileName,
					 const std::vector<Vertex>& vertices , const std::vector< std::vector< int > >& polygons,
					 PlyProperty* properties,int propertyNum,
					 int file_type,
					 char** comments,const int& commentNum)
{
	int nr_vertices=int(vertices.size());
	int nr_faces=int(polygons.size());
	float version;
	const char *elem_names[] = { "vertex" , "face" };
	PlyFile *ply = ply_open_for_writing( fileName , 2 , elem_names , file_type , &version );
	if (!ply){return 0;}
	
	//
	// describe vertex and face properties
	//
	ply_element_count(ply, "vertex", nr_vertices);
	for(int i=0;i<propertyNum;i++)
		ply_describe_property(ply, "vertex", &properties[i]);
	
	ply_element_count(ply, "face", nr_faces);
	ply_describe_property(ply, "face", &face_props[0]);
	
	// Write in the comments
	if(comments && commentNum)
		for(int i=0;i<commentNum;i++)
			ply_put_comment(ply,comments[i]);

	ply_header_complete(ply);
	
	// write vertices
	ply_put_element_setup(ply, "vertex");
	for (int i=0; i < int(vertices.size()); i++)
		ply_put_element(ply, (void *) &vertices[i]);

	// write faces
	PlyFace ply_face;
	int maxFaceVerts=3;
	ply_face.nr_vertices = 3;
	ply_face.vertices = new int[3];

	ply_put_element_setup(ply, "face");
	for (int i=0; i < nr_faces; i++)
	{
		if(int(polygons[i].size())>maxFaceVerts)
		{
			delete[] ply_face.vertices;
			maxFaceVerts=int(polygons[i].size());
			ply_face.vertices=new int[maxFaceVerts];
		}
		ply_face.nr_vertices=int(polygons[i].size());
		for(int j=0;j<ply_face.nr_vertices;j++)
			ply_face.vertices[j]=polygons[i][j];
		ply_put_element(ply, (void *) &ply_face);
	}

	delete[] ply_face.vertices;
	ply_close(ply);
	return 1;
}
template<class Vertex>
int PlyReadPolygons(char* fileName,
					std::vector<Vertex>& vertices , std::vector<std::vector<int> >& polygons ,
					 PlyProperty* properties , int propertyNum ,
					int& file_type ,
					char*** comments , int* commentNum , bool* readFlags )
{
	int nr_elems;
	char **elist;
	float version;
	int i,j,k;
	PlyFile* ply;
	char* elem_name;
	int num_elems;
	int nr_props;
	PlyProperty** plist;
	PlyFace ply_face;

	ply = ply_open_for_reading(fileName, &nr_elems, &elist, &file_type, &version);
	if(!ply) return 0;

	if(comments)
	{
		(*comments)=new char*[*commentNum+ply->num_comments];
		for(int i=0;i<ply->num_comments;i++)
			(*comments)[i]=_strdup(ply->comments[i]);
		*commentNum=ply->num_comments;
	}

	for (i=0; i < nr_elems; i++) {
		elem_name = elist[i];
		plist = ply_get_element_description(ply, elem_name, &num_elems, &nr_props);
		if(!plist)
		{
			for(i=0;i<nr_elems;i++){
				free(ply->elems[i]->name);
				free(ply->elems[i]->store_prop);
				for(j=0;j<ply->elems[i]->nprops;j++){
					free(ply->elems[i]->props[j]->name);
					free(ply->elems[i]->props[j]);
				}
				free(ply->elems[i]->props);
			}
			for(i=0;i<nr_elems;i++){free(ply->elems[i]);}
			free(ply->elems);
			for(i=0;i<ply->num_comments;i++){free(ply->comments[i]);}
			free(ply->comments);
			for(i=0;i<ply->num_obj_info;i++){free(ply->obj_info[i]);}
			free(ply->obj_info);
			ply_free_other_elements (ply->other_elems);
			
			for(i=0;i<nr_elems;i++){free(elist[i]);}
			free(elist);
			ply_close(ply);
			return 0;
		}		
		if (equal_strings("vertex", elem_name))
		{
			for( int i=0 ; i<propertyNum ; i++)
			{
				int hasProperty = ply_get_property(ply,elem_name,&properties[i]);
				if( readFlags ) readFlags[i] = (hasProperty!=0);
			}
			vertices.resize(num_elems);
			for (j=0; j < num_elems; j++)	ply_get_element (ply, (void *) &vertices[j]);
		}
		else if (equal_strings("face", elem_name))
		{
			ply_get_property (ply, elem_name, &face_props[0]);
			polygons.resize(num_elems);
			for (j=0; j < num_elems; j++)
			{
				ply_get_element (ply, (void *) &ply_face);
				polygons[j].resize(ply_face.nr_vertices);
				for(k=0;k<ply_face.nr_vertices;k++)	polygons[j][k]=ply_face.vertices[k];
				delete[] ply_face.vertices;
			}  // for, read faces
		}  // if face
		else{ply_get_other_element (ply, elem_name, num_elems);}

		for(j=0;j<nr_props;j++){
			free(plist[j]->name);
			free(plist[j]);
		}
		free(plist);
	}  // for each type of element
	
	for(i=0;i<nr_elems;i++){
		free(ply->elems[i]->name);
		free(ply->elems[i]->store_prop);
		for(j=0;j<ply->elems[i]->nprops;j++){
			free(ply->elems[i]->props[j]->name);
			free(ply->elems[i]->props[j]);
		}
		if(ply->elems[i]->props && ply->elems[i]->nprops){free(ply->elems[i]->props);}
	}
	for(i=0;i<nr_elems;i++){free(ply->elems[i]);}
	free(ply->elems);
	for(i=0;i<ply->num_comments;i++){free(ply->comments[i]);}
	free(ply->comments);
	for(i=0;i<ply->num_obj_info;i++){free(ply->obj_info[i]);}
	free(ply->obj_info);
	ply_free_other_elements (ply->other_elems);
	
	
	for(i=0;i<nr_elems;i++){free(elist[i]);}
	free(elist);
	ply_close(ply);
	return 1;
}

template< class Vertex , class Real >
int PlyWritePolygons( char* fileName , CoredMeshData< Vertex >* mesh , int file_type , const Point3D<float>& translate , float scale , char** comments , int commentNum , XForm4x4< Real > xForm )
{
	int i;
	int nr_vertices=int(mesh->outOfCorePointCount()+mesh->inCorePoints.size());
	int nr_faces=mesh->polygonCount();
	float version;
	const char *elem_names[] = { "vertex" , "face" };
	PlyFile *ply = ply_open_for_writing( fileName , 2 , elem_names , file_type , &version );
	if( !ply ) return 0;

	mesh->resetIterator();
	
	//
	// describe vertex and face properties
	//
	ply_element_count( ply , "vertex" , nr_vertices );
	for( int i=0 ; i<Vertex::Components ; i++ ) ply_describe_property( ply , "vertex" , &Vertex::Properties[i] );
	
	ply_element_count( ply , "face" , nr_faces );
	ply_describe_property( ply , "face" , &face_props[0] );
	
	// Write in the comments
	for( i=0 ; i<commentNum ; i++ ) ply_put_comment( ply , comments[i] );

	ply_header_complete( ply );
	
	// write vertices
	ply_put_element_setup( ply , "vertex" );
	for( i=0 ; i<int( mesh->inCorePoints.size() ) ; i++ )
	{
		Vertex vertex = xForm * ( mesh->inCorePoints[i] * scale + translate );
		ply_put_element(ply, (void *) &vertex);
	}
	for( i=0; i<mesh->outOfCorePointCount() ; i++ )
	{
		Vertex vertex;
		mesh->nextOutOfCorePoint( vertex );
		vertex = xForm * ( vertex * scale +translate );
		ply_put_element(ply, (void *) &vertex);		
	}  // for, write vertices
	
	// write faces
	std::vector< CoredVertexIndex > polygon;
	ply_put_element_setup( ply , "face" );
	for( i=0 ; i<nr_faces ; i++ )
	{
		//
		// create and fill a struct that the ply code can handle
		//
		PlyFace ply_face;
		mesh->nextPolygon( polygon );
		ply_face.nr_vertices = int( polygon.size() );
		ply_face.vertices = new int[ polygon.size() ];
		for( int i=0 ; i<int(polygon.size()) ; i++ )
			if( polygon[i].inCore ) ply_face.vertices[i] = polygon[i].idx;
			else                    ply_face.vertices[i] = polygon[i].idx + int( mesh->inCorePoints.size() );
		ply_put_element( ply, (void *) &ply_face );
		delete[] ply_face.vertices;
	}  // for, write faces
	
	ply_close( ply );
	return 1;
}
template< class Vertex , class Real >
int PlyWritePolygons( char* fileName , CoredMeshData< Vertex >* mesh , int file_type , char** comments , int commentNum , XForm4x4< Real > xForm )
{
	int i;
	int nr_vertices=int(mesh->outOfCorePointCount()+mesh->inCorePoints.size());
	int nr_faces=mesh->polygonCount();
	float version;
	const char *elem_names[] = { "vertex" , "face" };
	PlyFile *ply = ply_open_for_writing( fileName , 2 , elem_names , file_type , &version );
	if( !ply ) return 0;

	mesh->resetIterator();
	
	//
	// describe vertex and face properties
	//
	ply_element_count( ply , "vertex" , nr_vertices );
	for( int i=0 ; i<Vertex::WriteComponents ; i++ ) ply_describe_property( ply , "vertex" , &Vertex::WriteProperties[i] );
	
	ply_element_count( ply , "face" , nr_faces );
	ply_describe_property( ply , "face" , &face_props[0] );
	
	// Write in the comments
	for( i=0 ; i<commentNum ; i++ ) ply_put_comment( ply , comments[i] );

	ply_header_complete( ply );
	
	// write vertices
	ply_put_element_setup( ply , "vertex" );
	for( i=0 ; i<int( mesh->inCorePoints.size() ) ; i++ )
	{
		Vertex vertex = xForm * mesh->inCorePoints[i];
		ply_put_element(ply, (void *) &vertex);
	}
	for( i=0; i<mesh->outOfCorePointCount() ; i++ )
	{
		Vertex vertex;
		mesh->nextOutOfCorePoint( vertex );
		vertex = xForm * ( vertex );
		ply_put_element(ply, (void *) &vertex);		
	}  // for, write vertices
	
	// write faces
	std::vector< CoredVertexIndex > polygon;
	ply_put_element_setup( ply , "face" );
	for( i=0 ; i<nr_faces ; i++ )
	{
		//
		// create and fill a struct that the ply code can handle
		//
		PlyFace ply_face;
		mesh->nextPolygon( polygon );
		ply_face.nr_vertices = int( polygon.size() );
		ply_face.vertices = new int[ polygon.size() ];
		for( int i=0 ; i<int(polygon.size()) ; i++ )
			if( polygon[i].inCore ) ply_face.vertices[i] = polygon[i].idx;
			else                    ply_face.vertices[i] = polygon[i].idx + int( mesh->inCorePoints.size() );
		ply_put_element( ply, (void *) &ply_face );
		delete[] ply_face.vertices;
	}  // for, write faces
	
	ply_close( ply );
	return 1;
}
inline int PlyDefaultFileType(void){return PLY_ASCII;}

#endif /* !__PLY_H__ */
