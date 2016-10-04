/** @file host.h
 ** @brief Host
 ** @author Andrea Vedaldi
 ** @sa @ref portability
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_HOST_H
#define VL_HOST_H

/** ------------------------------------------------------------------
 ** @name Configuration options
 ** @{ */

#if defined(__DOXYGEN__)
#define VL_DISABLE_THREADS
#define VL_DISABLE_SSE2
#define VL_DISABLE_OPENMP
#endif

/** @} */

/** ------------------------------------------------------------------
 ** @name Defining functions
 ** @{ */

#if defined(__DOXYGEN__)
#define VL_EXPORT
#define VL_INLINE
#endif

/** @} */

/** ------------------------------------------------------------------
 ** @name C preprocessor helper macros
 ** @{ */

/** @brief Convert the argument to a string
 ** @param x value to be stringified.
 **
 ** This macro stringifies the argument @a x by means of the
 ** <code>#</code> prerpocessor operator.
 **
 ** The standard C preprocessor does not prescan arguments which are
 ** stringified, so
 **
 ** @code
 ** #define A B
 ** char const * str = VL_STRINGIFY(A) ;
 ** @endcode
 **
 ** initializes <code>str</code> with a pointer to the string
 ** <code>"A"</code>, which mihgt be unexpected. To fix this issue,
 ** you can use ::VL_XSTRINGIFY.
 **
 ** @sa ::VL_XSTRINGIFY
 **/

#define VL_STRINGIFY(x) # x

/** @brief Expand and then convert the argument to a string
 ** @param x value to be macro-expanded and converted.
 **
 ** This macro macro-expands the argument @a x and stringifies the
 ** result of the expansion. For instance
 **
 ** @code
 ** #define A B
 ** char const * str = VL_STRINGIFY(A) ;
 ** @endcode
 **
 ** initializes <code>str</code> with a pointer to the string
 ** <code>"B"</code>.
 **
 ** @sa ::VL_STRINGIFY
 **/

#define VL_XSTRINGIFY(x) VL_STRINGIFY(x)

/** @brief Concatenate two arguments into a lexical unit
 ** @param x first argument to be concatenated.
 ** @param y second argument to be concatenated.
 **
 ** This macro concatenates its arguments into a single lexical unit
 ** by means of the <code>##</code> preprocessor operator. Notice that
 ** arguments concatenated by <code>##</code> are not pre-expanded by
 ** the C preprocessor. To macro-expand the arguments and then
 ** concatenate them,use ::VL_XCAT.
 **
 ** @see ::VL_XCAT
 **/

#define VL_CAT(x,y) x ## y

/** @brief Expand and then concatenate two arguments into a lexical unit
 ** @param x first argument to be concatenated.
 ** @param y second argument to be concatenated.
 **
 ** This macro is the same as ::VL_CAT, except that the arguments are
 ** macro expanded before being concatenated.
 **
 ** @see ::VL_CAT
 **/

#define VL_XCAT(x,y) VL_CAT(x,y)

/** @brief Expand and then concatenate three arguments into a lexical unit
 ** @param x first argument to be concatenated.
 ** @param y second argument to be concatenated.
 ** @param z third argument to be concatenated.
 **
 ** This macro is the same as ::VL_XCAT, except that it has three arguments.
 **
 ** @see ::VL_XCAT
 **/

#define VL_XCAT3(x,y,z) VL_XCAT(VL_XCAT(x,y),z)

/** @brief Expand and then concatenate four arguments into a lexical unit
 ** @param x first argument to be concatenated.
 ** @param y second argument to be concatenated.
 ** @param z third argument to be concatenated.
 ** @param u fourth argument to be concatenated.
 **
 ** This macro is the same as ::VL_XCAT, except that it has four arguments.
 **
 ** @see ::VL_XCAT
 **/

#define VL_XCAT4(x,y,z,u) VL_XCAT(VL_XCAT3(x,y,z),u)

/** @brief Expand and then concatenate five arguments into a lexical unit
 ** @param x first argument to be concatenated.
 ** @param y second argument to be concatenated.
 ** @param z third argument to be concatenated.
 ** @param u fourth argument to be concatenated.
 ** @param v fifth argument to be concatenated.
 **
 ** This macro is the same as ::VL_XCAT, except that it has five arguments.
 **
 ** @see ::VL_XCAT
 **/

#define VL_XCAT5(x,y,z,u,v) VL_XCAT(VL_XCAT4(x,y,z,u),v)

/** @brief Convert a boolean to "yes" or "no" strings
 ** @param x boolean to convert.
 **
 ** A pointer to either the string "yes" (if @a x is true)
 ** or the string "no".
 **
 ** @par Example
 ** @code
 ** VL_PRINTF("Is x true? %s.", VL_YESNO(x))
 ** @endcode
 **/

#define VL_YESNO(x) ((x)?"yes":"no")

/** @} */

/*
 The following macros identify the host OS, architecture and compiler.
 They are derived from http://predef.sourceforge.net/
 */

/** @name Identifying the host operating system
 ** @{ */
#if defined(linux)     || \
    defined(__linux)   || \
    defined(__linux__) || \
    defined(__DOXYGEN__)
#define VL_OS_LINUX 1
#endif

#if (defined(__APPLE__) & defined(__MACH__)) || \
     defined(__DOXYGEN__)
#define VL_OS_MACOSX 1
#endif

#if defined(__WIN32__) || \
    defined(_WIN32)    || \
    defined(__DOXYGEN__)
#define VL_OS_WIN 1
#endif

#if defined(_WIN64) || \
    defined(__DOXYGEN__)
#define VL_OS_WIN64 1
#endif
/** @} */

/** @name Identifying the host threading library
 ** @{ */
#if defined(VL_OS_MACOSX) || defined(VL_OS_LINUX) || \
defined(__DOXYGEN__)
#define VL_THREADS_POSIX 1
#endif

#if defined(VL_OS_WIN) || defined(VL_OS_WIN64) || \
defined(__DOXYGEN__)
#define VL_THREADS_WIN 1
#endif
/** @} */

/** @name Identifying the host compiler
 ** @{ */
#if defined(__GNUC__) || defined(__DOXYGEN__)
# if defined(__GNUC_PATCHLEVEL__)
#  define VL_COMPILER_GNUC (__GNUC__ * 10000 \
+ __GNUC_MINOR__ * 100 \
+ __GNUC_PATCHLEVEL__)
# else
#  define VL_COMPILER_GNUC (__GNUC__ * 10000 \
+ __GNUC_MINOR__ * 100)
# endif
#endif

#if defined(_MSC_VER) || defined(__DOXYGEN__)
#define VL_COMPILER_MSC _MSC_VER
#endif

#if defined(__LCC__) || defined(__DOXYGEN__)
#warning "LCC support is experimental!"
#define VL_COMPILER_LCC 1
#endif

/** @} */

/** @name Identifying the host CPU architecture
 ** @{ */
#if defined(i386)     || \
    defined(__i386__) || \
    defined(__DOXYGEN__)
#define VL_ARCH_IX86 300
#elif defined(__i486__)
#define VL_ARCH_IX86 400
#elif defined(__i586__)
#define VL_ARCH_IX86 500
#elif defined(__i686__)
#define VL_ARCH_IX86 600
#elif defined(_M_IX86)
#define VL_ARCH_IX86 _M_IX86
#endif

#if defined(_M_X64)    || \
    defined(__amd64__) || \
    defined(__amd64)   || \
    defined(__x86_64)  || \
    defined(__x86_64)
#define VL_ARCH_X64
#endif

#if defined(__ia64__)   || \
    defined(_IA64)      || \
    defined(__IA64)     || \
    defined(__ia64)     || \
    defined(_M_IA64)    || \
    defined(__DOXYGEN__)
#define VL_ARCH_IA64
#endif
/** @} */

/** @name Identifying the host data model
 ** @{ */
#if defined(__LLP64__) || \
    defined(__LLP64)   || \
    defined(__LLP64)   || \
    (defined(VL_COMPILER_MSC) & defined(VL_OS_WIN64)) || \
    (defined(VL_COMPILER_LCC) & defined(VL_OS_WIN64)) || \
    defined(__DOXYGEN__)
#define VL_COMPILER_LLP64
#endif

#if defined(__LP64__) || \
    defined(__LP64)   || \
    defined(__LP64)   || \
    (defined(VL_OS_MACOSX) & defined(VL_ARCH_IA64)) || \
    defined(__DOXYGEN__)
#define VL_COMPILER_LP64
#endif

#if (!defined(VL_COMPILER_LLP64) & !defined(VL_COMPILER_LP64)) || \
    defined(__DOXYGEN__)
#define VL_COMPILER_ILP32
#endif
/** @} */

/** @name Identifying the host endianness
 ** @{ */
#if defined(__LITTLE_ENDIAN__) || \
    defined(VL_ARCH_IX86)      || \
    defined(VL_ARCH_IA64)      || \
    defined(VL_ARCH_X64)       || \
    defined(__DOXYGEN__)
#define VL_ARCH_LITTLE_ENDIAN
#endif

#if  defined(__DOXYGEN__) || \
    !defined(VL_ARCH_LITTLE_ENDIAN)
#define VL_ARCH_BIG_ENDIAN
#endif
/** @} */

#if defined(VL_COMPILER_MSC) & ! defined(__DOXYGEN__)
#  define VL_UNUSED
#  define VL_INLINE static __inline
#if defined _MSC_VER && _MSC_VER < 1900
#  define snprintf _snprintf
#endif
#  define isnan _isnan
#  ifdef VL_BUILD_DLL
#    ifdef __cplusplus
#      define VL_EXPORT extern "C" __declspec(dllexport)
#    else
#      define VL_EXPORT extern __declspec(dllexport)
#    endif
#  else
#    ifdef __cplusplus
#      define VL_EXPORT extern "C"
#    else
#      define VL_EXPORT extern
#    endif
#  endif
#endif

#if defined(VL_COMPILER_LCC) & ! defined(__DOXYGEN__)
#  define VL_UNUSED
#  define VL_INLINE static __inline
#  define snprintf _snprintf
#  define isnan _isnan
VL_INLINE float fabsf(float x) { return (float) fabs((double) x) ; }
#  ifdef VL_BUILD_DLL
#    define VL_EXPORT extern __declspec(dllexport)
#  else
#    define VL_EXPORT extern
#  endif
#endif

#if defined(VL_COMPILER_GNUC) & ! defined(__DOXYGEN__)
#  define VL_UNUSED __attribute__((unused))
#  define VL_INLINE static __inline__
#  ifdef VL_BUILD_DLL
#    ifdef __cplusplus
#      define VL_EXPORT __attribute__((visibility ("default"))) extern "C"
#    else
#      define VL_EXPORT __attribute__((visibility ("default"))) extern
#    endif
#  else
#    ifdef __cplusplus
#      define VL_EXPORT extern "C"
#    else
#      define VL_EXPORT extern
#    endif
#  endif
#endif

VL_EXPORT char * vl_static_configuration_to_string_copy () ;

/** ------------------------------------------------------------------
 ** @name Atomic data types
 ** @{
 **/

#define VL_TRUE 1   /**< @brief @c true (1) constant */
#define VL_FALSE 0  /**< @brief @c false (0) constant */

#if defined(VL_COMPILER_LP64) || defined(VL_COMPILER_LLP64)
typedef long long           vl_int64 ;   /**< @brief Signed 64-bit integer. */
typedef int                 vl_int32 ;   /**< @brief Signed 32-bit integer. */
typedef short               vl_int16 ;   /**< @brief Signed 16-bit integer. */
typedef char                vl_int8  ;   /**< @brief Signed  8-bit integer. */

typedef long long unsigned  vl_uint64 ;  /**< @brief Unsigned 64-bit integer. */
typedef int       unsigned  vl_uint32 ;  /**< @brief Unsigned 32-bit integer. */
typedef short     unsigned  vl_uint16 ;  /**< @brief Unsigned 16-bit integer. */
typedef char      unsigned  vl_uint8 ;   /**< @brief Unsigned  8-bit integer. */

typedef int                 vl_int ;     /**< @brief Same as @c int. */
typedef unsigned int        vl_uint ;    /**< @brief Same as <code>unsigned int</code>. */

typedef int                 vl_bool ;    /**< @brief Boolean. */
typedef vl_int64            vl_intptr ;  /**< @brief Integer holding a pointer. */
typedef vl_uint64           vl_uintptr ; /**< @brief Unsigned integer holding a pointer. */
typedef vl_uint64           vl_size ;    /**< @brief Unsigned integer holding the size of a memory block. */
typedef vl_int64            vl_index ;   /**< @brief Signed version of ::vl_size and ::vl_uindex */
typedef vl_uint64           vl_uindex ;  /**< @brief Same as ::vl_size */
#endif

#if defined(VL_COMPILER_ILP32)

#ifdef VL_COMPILER_MSC
typedef __int64             vl_int64 ;
#else
typedef long long           vl_int64 ;
#endif

typedef int                 vl_int32 ;
typedef short               vl_int16 ;
typedef char                vl_int8  ;

#ifdef VL_COMPILER_MSC
typedef __int64   unsigned  vl_uint64 ;
#else
typedef long long unsigned  vl_uint64 ;
#endif
typedef int       unsigned  vl_uint32 ;
typedef short     unsigned  vl_uint16 ;
typedef char      unsigned  vl_uint8 ;

typedef int                 vl_int ;
typedef unsigned int        vl_uint ;

typedef int                 vl_bool ;
typedef vl_int32            vl_intptr ;
typedef vl_uint32           vl_uintptr ;
typedef vl_uint32           vl_size ;
typedef vl_int32            vl_index ;
typedef vl_uint32           vl_uindex ;
#endif
/** @} */

/** @name Creating integer constants
 ** @{ */
#if defined(VL_COMPILER_LP64) || defined(__DOXYGEN__)
#define VL_INT8_C(x) x
#define VL_INT16_C(x) x
#define VL_INT32_C(x) x
#define VL_INT64_C(x) x ## L

#define VL_UINT8_C(x) x
#define VL_UINT16_C(x) x
#define VL_UINT32_C(x) x ## U
#define VL_UINT64_C(x) x ## UL
#endif

#if (defined(VL_COMPILER_LLP64) || defined(VL_COMPILER_ILP32)) \
    & !defined(__DOXYGEN__)
#define VL_INT8_C(x) x
#define VL_INT16_C(x) x
#define VL_INT32_C(x) x
#define VL_INT64_C(x) x ## LL

#define VL_UINT8_C(x) x
#define VL_UINT16_C(x) x
#define VL_UINT32_C(x) x ## U
#define VL_UINT64_C(x) x ## ULL
#endif
/** @} */

/** ------------------------------------------------------------------
 ** @name Printing the atomic data types
 ** @{ */

/* Lengths only: */

/** @def VL_FL_INT64
 ** @brief @c printf length flag for ::vl_int64 and ::vl_uint64.
 **/

/** @def VL_FL_INT32
 ** @brief @c printf length flag for ::vl_int32 and ::vl_uint32.
 **/

/** @def VL_FL_INT16
 ** @brief @c printf length flag for ::vl_int16 and ::vl_uint16.
 **/

/** @def VL_FL_INT8
 ** @brief @c printf length flag for ::vl_int8 and ::vl_uint8.
 **/

/** @def VL_FL_INDEX
 ** @brief @c printf length flag for ::vl_index and ::vl_uindex
 **/

#ifdef VL_COMPILER_MSC
#define VL_FL_INT64  "I64"
#else
#define VL_FL_INT64  "ll"
#endif
#define VL_FL_INT32  ""
#define VL_FL_INT16  "h"
#define VL_FL_INT8   "hh"

#if defined(VL_COMPILER_LP64) || defined(VL_COMPILER_LLP64)
#define VL_FL_INDEX VL_FL_INT64
#endif

#if defined(VL_COMPILER_ILP32)
#define VL_FL_INDEX VL_FL_INT32
#endif

/* Formats (but not conversions!): */

/** @def VL_FMT_SIZE
 ** @brief @c printf flag for ::vl_size
 **/

/** @def VL_FMT_INDEX
 ** @brief @c printf flag for ::vl_index
 **/

/** @def VL_FMT_UINDEX
 ** @brief @c printf flag for ::vl_uindex
 **/

/** @def VL_FMT_INTPTR
 ** @brief @c printf flag for ::vl_intptr
 **/

/** @def VL_FMT_UINTPTR
 ** @brief @c printf flag for ::vl_uintptr
 **/

#define VL_FMT_INDEX   VL_FL_INDEX "d"
#define VL_FMT_INTPTR  VL_FMT_INDEX
#define VL_FMT_UINDEX  VL_FL_INDEX "u"
#define VL_FMT_SIZE    VL_FMT_UINDEX
#define VL_FMT_UINTPTR VL_FMT_UINDEX

/** @} */

/** ------------------------------------------------------------------
 ** @name Atomic data types limits
 ** @{ */

/** @brief Largest integer (math constant) */
#define VL_BIG_INT  0x7FFFFFFFL

/** @brief Smallest integer (math constant) */
#define VL_SMALL_INT  (- VL_BIG_INT - 1)

/** @brief Largest unsigned integer (math constant) */
#define VL_BIG_UINT 0xFFFFFFFFUL

/** @} */

/** ------------------------------------------------------------------
 ** @name Endianness detection and conversion
 ** @{
 **/
VL_INLINE void vl_swap_host_big_endianness_8 (void *dst, void* src) ;
VL_INLINE void vl_swap_host_big_endianness_4 (void *dst, void* src) ;
VL_INLINE void vl_swap_host_big_endianness_2 (void *dst, void* src) ;
/** @} */

/** ------------------------------------------------------------------
 ** @name Obtaining host info at run time
 ** @{ */

typedef struct _VlX86CpuInfo
{
  union {
    char string [0x20] ;
    vl_uint32 words [0x20 / 4] ;
  } vendor ;
  vl_bool hasAVX ;
  vl_bool hasSSE42 ;
  vl_bool hasSSE41 ;
  vl_bool hasSSE3 ;
  vl_bool hasSSE2 ;
  vl_bool hasSSE ;
  vl_bool hasMMX ;
} VlX86CpuInfo ;

void _vl_x86cpu_info_init (VlX86CpuInfo *self) ;
char * _vl_x86cpu_info_to_string_copy (VlX86CpuInfo const *self) ;

/** @} */

/** ------------------------------------------------------------------
 ** @brief Host <-> big endian transformation for 8-bytes value
 **
 ** @param dst destination 8-byte buffer.
 ** @param src source 8-byte bufffer.
 ** @see @ref host-arch-endianness.
 **/

VL_INLINE void
vl_swap_host_big_endianness_8 (void *dst, void* src)
{
  char *dst_ = (char*) dst ;
  char *src_ = (char*) src ;
#if defined(VL_ARCH_BIG_ENDIAN)
    dst_ [0] = src_ [0] ;
    dst_ [1] = src_ [1] ;
    dst_ [2] = src_ [2] ;
    dst_ [3] = src_ [3] ;
    dst_ [4] = src_ [4] ;
    dst_ [5] = src_ [5] ;
    dst_ [6] = src_ [6] ;
    dst_ [7] = src_ [7] ;
#else
    dst_ [0] = src_ [7] ;
    dst_ [1] = src_ [6] ;
    dst_ [2] = src_ [5] ;
    dst_ [3] = src_ [4] ;
    dst_ [4] = src_ [3] ;
    dst_ [5] = src_ [2] ;
    dst_ [6] = src_ [1] ;
    dst_ [7] = src_ [0] ;
#endif
}

/** ------------------------------------------------------------------
 ** @brief Host <-> big endian transformation for 4-bytes value
 **
 ** @param dst destination 4-byte buffer.
 ** @param src source 4-byte bufffer.
 ** @sa @ref host-arch-endianness.
 **/

VL_INLINE void
vl_swap_host_big_endianness_4 (void *dst, void* src)
{
  char *dst_ = (char*) dst ;
  char *src_ = (char*) src ;
#if defined(VL_ARCH_BIG_ENDIAN)
    dst_ [0] = src_ [0] ;
    dst_ [1] = src_ [1] ;
    dst_ [2] = src_ [2] ;
    dst_ [3] = src_ [3] ;
#else
    dst_ [0] = src_ [3] ;
    dst_ [1] = src_ [2] ;
    dst_ [2] = src_ [1] ;
    dst_ [3] = src_ [0] ;
#endif
}

/** ------------------------------------------------------------------
 ** @brief Host <-> big endian transformation for 2-bytes value
 **
 ** @param dst destination 2-byte buffer.
 ** @param src source 2-byte bufffer.
 ** @see @ref host-arch-endianness.
 **/

VL_INLINE void
vl_swap_host_big_endianness_2 (void *dst, void* src)
{
  char *dst_ = (char*) dst ;
  char *src_ = (char*) src ;
#if defined(VL_ARCH_BIG_ENDIAN)
    dst_ [0] = src_ [0] ;
    dst_ [1] = src_ [1] ;
#else
    dst_ [0] = src_ [1] ;
    dst_ [1] = src_ [0] ;
#endif
}

/* VL_HOST_H */
#endif
