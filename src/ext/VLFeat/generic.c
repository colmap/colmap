/** @file generic.c
 ** @brief Generic - Definition
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
Copyright (C) 2013 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@mainpage Vision Lab Features Library (VLFeat)
@version __VLFEAT_VERSION__
@author The VLFeat Team
@par Copyright &copy; 2012-14 The VLFeat Authors
@par Copyright &copy; 2007-11 Andrea Vedaldi and Brian Fulkerson
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The VLFeat C library implements common computer
vision algorithms, with a special focus on visual features, as used
in state-of-the-art object recognition and image
matching applications.

VLFeat strives to be clutter-free, simple, portable, and well documented.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section main-contents Contents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

- **Visual feature detectors and descriptors**
  - @subpage sift
  - @subpage dsift
  - @subpage mser
  - @subpage covdet
  - @subpage scalespace
  - @subpage hog
  - @subpage fisher
  - @subpage vlad
  - @subpage liop
  - @subpage lbp

- **Clustering and indexing**
  - @subpage kmeans
  - @subpage ikmeans.h  "Integer K-means (IKM)"
  - @subpage hikmeans.h "Hierarchical Integer K-means (HIKM)"
  - @subpage gmm
  - @subpage aib
  - @subpage kdtree

- **Segmentation**
  - @subpage slic
  - @subpage quickshift

- **Statistical methods**
  - @subpage aib
  - @subpage homkermap
  - @subpage svm

- **Utilities**
  - @subpage random
  - @subpage mathop
  - @subpage stringop.h  "String operations"
  - @subpage imopv.h     "Image operations"
  - @subpage pgm.h       "PGM image format"
  - @subpage heap-def.h  "Generic heap object (priority queue)"
  - @subpage rodrigues.h "Rodrigues formula"
  - @subpage mexutils.h  "MATLAB MEX helper functions"
  - @subpage getopt_long.h "Drop-in @c getopt_long replacement"

- **General information**
  - @subpage conventions
  - @subpage generic
  - @subpage portability
  - @ref resources
  - @subpage objects
  - @ref threads
  - @subpage matlab
  - @subpage metaprogram

- @subpage dev
- @subpage glossary
**/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page resources Memory and resource handling
@author Andrea Vedaldi
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Some VLFeat functions return pointers to memory blocks or
objects. Only ::vl_malloc, ::vl_calloc, ::vl_realloc and functions
whose name contains either the keywords @c new or @c copy transfer the
ownership of the memory block or object to the caller. The caller must
dispose explicitly of all the resources it owns (by calling ::vl_free
for a memory block, or the appropriate deletion function for an
object).

The memory allocation functions can be customized by
::vl_set_alloc_func (which sets the implementations of ::vl_malloc,
::vl_realloc, ::vl_calloc and ::vl_free). Remapping the memory
allocation functions can be done only if there are no currently
allocated VLFeat memory blocks or objects -- thus typically at the
very beginning of a program. The memory allocation functions are a
global property, shared by all threads.

VLFeat uses three rules that simplify handling exceptions when used in
combination which certain environment such as MATLAB.

- The library allocates local memory only through the re-programmable
  ::vl_malloc, ::vl_calloc, and ::vl_realloc functions.

- The only resource referenced by VLFeat objects is memory (for
  instance, it is illegal for an object to reference an open file).
  Other resources such as files or threads may be allocated within a
  VLFeat function call, but they are all released before the function
  ends, or their ownership is directly transferred to the caller.

- The global library state is an exception. It cannot reference any
  local object created by the caller and uses the standard C memory
  allocation functions.

In this way, the VLFeat local state can be reset at any time simply by
disposing of all the memory allocated by the library so far. The
latter can be done easily by mapping the memory allocation functions
to implementations that track the memory blocks allocated, and then
disposing of all such blocks. Since the global state does not
reference any local object nor uses the remapped memory functions, it
is unaffected by such an operation; conversely, since no VLFeat object
references anything but memory, this guarantees that all allocated
resources are properly disposed (avoiding leaking resource). This is
used extensively in the design of MATLAB MEX files (see @ref
matlab).
**/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page conventions Conventions
@author Andrea Vedaldi
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

This page summarizes some of the conventions used by the library.

@section conventions-storage Matrix and image storage conventions

If not otherwise specified, matrices in VLFeat are stored in memory in
<em>column major</em> order. Given a matrix $[A_{ij}] \in \real^{m
\times n}$, this amounts of enumerating the elements one column per
time: $A_{11}, A_{21}, \dots, A_{m1}, A_{12}, \dots, A_{mn}$. This
convention is compatible with Fortran, MATLAB, and popular numerical
libraries.

Matrices are often used in the library to pack a number data vectors
$\bx_1,\dots,\bx_n \in \real^m$ of equal dimension together. These are
normally stored as the columns of the matrix:

\[
X = \begin{bmatrix} \bx_1, \dots, \bx_n \end{bmatrix},
\qquad
X \in \real_{m\times n}
\]

In this manner, consecutive elements of each data vector $\bx_i$ is
stored in consecutive memory locations, improving memory access
locality in most algorithms.

Images $I(x,y)$ are stored instead in <em>row-major</em> order,
i.e. one row after the other. Note that an image can be naturally
identified as a matrix $I_{yx}$, where the vertical coordinate $y$
indexes the rows and the horizontal coordinate $x$ the columns. The
image convention amounts to storing this matrix in row-major rather
than column-major order, which is in conflict with the rule given
above. The reason for this choice is that most image processing and
graphical libraries assume this convention; it is, however,
<em>not</em> the same as MATLAB's.

**/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page objects Objects
@author Andrea Vedaldi
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Many VLFeat algorithms are available in the form of *objects*. The C
language, used by VLFeat, does not support objects explicitly. Here an
object is intended a C structure along with a number of functions (the
object member functions or methods) operating on it. Ideally, the
object data structure is kept opaque to the user, for example by
defining it in the @c .c implementation files which are not accessible
to the library user.

Object names are capitalized and start with the <code>Vl</code> prefix
(for example @c VlExampleObject). Object methods are lowercase and
start with the <code>vl_<object_name>_</code> suffix
(e.g. @c vl_example_object_new).

<!-- ------------------------------------------------------------  -->
@section objects-lifecycle Object lifecycle
<!-- ------------------------------------------------------------  -->

Conceptually, an object undergoes four phases during its lifecycle:
allocation, initialization, finalization, and deallocation:

- **Allocation.** The memory to hold the object structure is allocated.
  This is usually done by calling a memory allocation function such as
  ::vl_calloc to reserve an object of the required size @c
  sizeof(VlExampleObject). Alternatively, the object can simply by
  allocated on the stack by declaring a local variable of type
  VlExampleObject.
- **Initialization.** The object is initialized by assigning a value to
  its data members and potentially allocating a number of resources,
  including other objects or memory buffers. Initialization is
  done by methods containing the @c init keyword, e.g.  @c
  vl_example_object_init. Several such methods may be provided.
- **Finalization.** Initialization is undone by finalization, whose main
  purpose is to release any resource allocated and still owned by the
  object. Finalization is done by the @c vl_example_object_finalize
  method.
- **Deallocation.** The memory holding the object structure is
  disposed of, for example by calling ::vl_free or automatically when
  the corresponding local variable is popped from the stack.

In practice, most VlFeat object are supposed to be created on the
heap. To this end, allocation/initialization and
finalization/deallocation are combined into two operations:

- **Creating a new object.** This allocates a new object on the heap
  and initializes it, combining allocation and initialization in a
  single operation. It is done by methods containing the @c new keyword,
  e.g. @c vl_example_object_new.
- **Deleting an object.** This disposes of an object created by a @c
  new method, combining finalization and deallocation, for example
  @c vl_example_object_delete.

<!-- ------------------------------------------------------------  -->
@section objects-getters-setters Getters and setters
<!-- ------------------------------------------------------------  -->

Most objects contain a number of methods to get (getters) and set
(setters) properties. These should contain the @c get and @c set
keywords in their name, for example

@code
double x = vl_example_object_get_property () ;
vl_example_object_set_property(x) ;
@endcode
**/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page matlab MATLAB integration
@author Andrea Vedaldi
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The VLFeat C library is designed to integrate seamlessly with MATLAB.
Binary compatibility is simplified by the use of the C language
(rather than C++). In addition, the library design follows certain
restrictions that make it compatible with the MATLAB MEX interface.

The main issue in calling a library function from a MATLAB MEX
function is that MATLAB can abort the execution of the MEX function
at any point, either due to an error, or directly upon a user request
(Ctrl-C) (empirically, however, a MEX function seems to be
incorruptible only during the invocation of certain functions of the
MEX API such as @c mexErrMsgTxt).

When a MEX function is interrupted, resources (memory blocks or
objects) whose ownership was transferred from VLFeat to the MEX
function may be leaked. Notice that interrupting a MEX function would
similarly leak any memory block allocated within the MEX function. To
solve this issue, MATLAB provides his own memory manager (@c
mxMalloc, @c mxRealloc, ...). When a MEX file is interrupted or ends,
all memory blocks allocated by using one of such functions are
released, preventing leakage.

In order to integrate VLFeat with this model in the most seamless
way, VLFeat memory allocation functions (::vl_malloc, ::vl_realloc,
::vl_calloc) are mapped to the corresponding MEX memory allocation
functions. Such functions automatically dispose of all the memory
allocated by a MEX function when the function ends (even because of
an exception). Because of the restrictions of the library design
illustrated in @ref resources, this operation is safe and
correctly dispose of VLFeat local state. As a consequence, it is
possible to call @c mexErrMsgTxt at any point in the MEX function
without worrying about leaking resources.

This however comes at the price of some limitations. Beyond the
restrictions illustrated in @ref resources, here we note that no
VLFeat local resource (memory blocks or objects) can persist across
MEX file invocations. This implies that any result produced by a
VLFeat MEX function must be converted back to a MATLAB object such as
a vector or a structure. In particular, there is no direct way of
creating an object within a MEX file, returning it to MATLAB, and
passing it again to another MEX file.
**/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page metaprogram Preprocessor metaprogramming
@author Andrea Vedaldi
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Part of VLFeat code uses a simple form of preprocessor metaprogramming.
This technique is used, similarly to C++ templates, to instantiate
multiple version of a given algorithm for different data types
(e.g. @c float and @c double).

In most cases preprocessor metaprogramming is invisible to the library
user, as it is used only internally.
**/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page glossary Glossary
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

 - <b>Column-major.</b> A <em>M x N </em> matrix <em>A</em> is
 stacked with column-major order as the sequence \f$(A_{11}, A_{21},
 \dots, A_{12}, \dots)\f$. More in general, when stacking a multi
 dimensional array this indicates that the first index is the one
 varying most quickly, with the other followed in the natural order.
 - <b>Opaque structure.</b> A structure is opaque if the user is not supposed
 to access its member directly, but through appropriate interface functions.
 Opaque structures are commonly used to define objects.
 - <b>Row-major.</b> A <em>M x N </em> matrix <em>A</em> is
 stacked with row-major order as the sequence \f$(A_{11}, A_{12},
 \dots, A_{21}, \dots)\f$. More in general, when stacking a multi
 dimensional array this indicates that the last index is the one
 varying most quickly, with the other followed in reverse order.
 - <b>Feature frame.</b> A <em>feature frame</em> is the geometrical
 description of a visual features. For instance, the frame of
 a @ref sift.h "SIFT feature" is oriented disk and the frame of
 @ref mser.h "MSER feature" is either a compact and connected set or
 a disk.
 - <b>Feature descriptor.</b> A <em>feature descriptor</em> is a quantity
 (usually a vector) which describes compactly the appearance of an
 image region (usually corresponding to a feature frame).
**/

/**

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page dev Developing the library
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

This page contains information useful to the developer of VLFeat.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section dev-copy Copyright
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

A short copyright notice is added at the beginning of each file. For
example:

<pre>
Copyright (C) 2013 Milan Sulc
Copyright (C) 2012 Daniele Perrone.
Copyright (C) 2011-13 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
</pre>

The copyright of each file is assigned to the authors of the file.
Every author making a substantial contribution to a file should
note its copyright by adding a line to the copyright list with the year
of the modification. Year ranges are acceptable. Lines are never
deleted, only appended, or potentially modified to list
more years.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section dev-style Coding style
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

<ul>

<li><b>Look at existing code before you start.</b> The general rule
is: try to match the style of the existing code as much as
possible.</li>

<li><b>No white spaces at the end of lines.</b> White spaces introduce
invisible changes in the code that are however picked up by control
version systems such as Git.</li>

<li><b>Descriptive variable names.</b> Most variable names start with
a lower case letter and are capitalized, e.g., @c numElements. Only
the following abbreviations are considered acceptable: @c num. The @c
dimension of a vector is the number of elements it contains (for other
objects that could be a @c size, a @c length, or a @c
numElements). For multi-dimensional arrays, @c dimensions could
indicate the array with each of the @c numDimensions dimensions.</li>

<li><b>Short variable names.</b> For indexes in short for loops it is
fine to use short index names such as @c i, @c j, and @c k. For example:
<pre>
for (i = 0 ; i < numEntries ; ++i) values[i] ++ ;
</pre>
is considered acceptable.</li>

<li><b>Function arguments.</b> VLFeat functions that operate on an
object (member functions) should be passed the object address as first
argument; this argument should be called @c self. For example:
<pre>
   void vl_object_do_something(VlObject *self) ;
</pre>
Multi-dimensional arrays should be specified first by their address,
and then by their dimensions. For example
<pre>
  void vl_use_array (float * array, vl_size numColumns, vl_size numRows) ; // good
  void vl_use_array (vl_size numColumns, vl_size numRows, float * array) ; // bad
</pre>
Arguments that are used as outputs should be specified first (closer to
the left-hand side of an expression). For example
<pre>
 void vl_sum_numbers (float * output, float input1, float input2) ; // good
 void vl_sum_numbers (float input1, float input2, float * output) ; // bad
</pre>
These rules can be combined. For example
<pre>
 void vl_object_sum_to_array (VlObject * self, float * outArray,
        vl_size numColumns, vl_size numRows, float * inArray) ; // good
</pre>
Note that in this case no dimension for @c inArray is specified as it
is assumed that @c numColumns and @c numRows are the dimensions of
both arrays.
</li>
</ul>

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection dev-style-matlab MATLAB coding style
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

<ul>
<li><b>Help messages.</b> Each @c .m file should include a standard
help comment block (accessible from MATLAB @c help() command).
The first line of the block has a space, the name of the function,
4 spaces, and a brief command description. The body of the help
message is indented with 4 spaces. For example
@code
% VL_FUNCTION    An example function
%    VL_FUNCTION() does nothing.
@endcode
The content HELP message itself should follow MATLAB default style.
For example, rather than giving a list of formal input and output
arguments as often done, one simply shows how to use the function, explaining
along the way the different ways the function can be called and
the format of the parameters.
</li>
</ul>

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section dev-doc Documenting the code
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

The VLFeat C library code contains its own in documentation <a
href='http://www.stack.nl/~dimitri/doxygen/'>Doxygen</a> format. The
documentation consists in generic pages, such as the @ref index
"index" and the page you are reading, and documentations for each
library module, usually corresponding to a certain header file.

- **Inline comments.** Inline Doxygen comments are discouraged except
  in the documentation of data members of structures. They start with
  a capital letter and end with a period. For example:
  @code
  struct VlExampleStructure {
    int aMember ; /\*\*< A useful data member.
  }
  @endcode

- **Brief comments.** Brief Doxygen comments starts by a capital
  and end with a period. The documentation of all functions start
  with a brief comment.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection devl-doc-modules Documenting the library modules
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

A library module groups a number of data types and functions that
implement a certain functionality of VLFeat. The documentation of a
library module is generally organized as follows:

1. A page introducing the module and including a getting started
   section (3.g. @ref svm-starting) containing a short tutorial to
   quickly familiarize the user with the module (e.g. @ref svm).
2. One or more pages of detailed technical background discussing the
   algorithms implemented. These sections are used not just as part of
   the C API, but also as documentation for other APIs such as MATLAB
   (e.g. @ref svm-fundamentals).
3. One or more pages with the structure and function documentation
   (e.g. @ref svm.h).

More in detail, consider a module called <em>Example Module</em>. Then one would
typically have:

<ul>
<li>A header or declaration file @c example-module.h. Such a file has an
heading of the type:

@verbinclude example-module-doc.h

This comment block contains a file directive, causing the file to be
included in the documentation, a brief directive, specifying a short
description of what the file is, and a list of authors. A
(non-Doxygen) comment block with a short the copyright notice follows.
The brief directive should include a <code>@@ref</code> directive to point
to the main documentation page describing the module, if there is one.
</li>

<li> An implementation or definition file @c example-module.c. This file
has an heading of the type:

@verbinclude example-module-doc.c

This is similar to the declaration file, except for the content of the
brief comment.
</li>
</ul>

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection devl-doc-functions Documenting functions
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection devl-doc-structures Documenting structures
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection devl-doc-structures Documenting objects
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

As seen in @ref objects, VLFeat treats certain structures with
an object-like semantics. Usually, a module defines exactly one such
objects. In this case, the object member functions should be grouped
(by using Doxygen grouping functionality) as

- **Construct and destroy** for the @c vl_object_new, @c
    vl_object_delete and similar member functions.
- **Set parameters** for setter functions.
- **Retrieve parameters and data** for getter functions.
- **Process data** for functions processing data.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@subsection devl-doc-bib Bibliographic references
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Since version 0.9.14, the VLFeat C library documentation makes use of
a proper bibliographic reference in BibTeX format (see the file @c
docsrc/vlfeat.bib). Doxygen uses this file when it sees instances of
the <code>@@cite{xyz}</code> command.  Here @c xyz is a BibTeX
key. For example, @c vlfeat.bib file contains the entry:

<pre>
@@inproceedings{martin97the-det-curve,
	Author = {A. Martin and G. Doddington and T. Kamm and M. Ordowski and M. Przybocki},
	Booktitle = {Proc. Conf. on Speech Communication and Technology},
	Title = {The {DET} curve in assessment of detection task performance},
	Year = {1997}}
</pre>

For example, the Doxygen directive
<code>@@cite{martin97the-det-curve}</code> generates the output
@cite{martin97the-det-curve}, which is a link to the corresponding
entry in the bibliography.

**/

/**

@file generic.h

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page generic General support functionalities
@author Andrea Vedaldi
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

VLFeat contains several support functionalities addressing the C
preprocessors, using multiple threads (including parallel computations),
handling errors, allocating memory, etc. These are described in
the following pages:

- @subpage resources
- @subpage threads
- @subpage misc
**/

/**

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page misc Preprocssor, library state, etc.
@author Andrea Vedaldi
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section misc-preproc C preprocessor helpers
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

VLFeat provides a few C preprocessor macros of general
utility. These include stringification (::VL_STRINGIFY,
::VL_XSTRINGIFY) and concatenation (::VL_CAT, ::VL_XCAT) of symbols.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section misc-state VLFeat state and configuration parameters
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

VLFeat has some global configuration parameters that can
changed. Changing the configuration is thread unsave
(@ref threads). Use ::vl_set_simd_enabled to toggle the use of
a SIMD unit (Intel SSE code), ::vl_set_alloc_func to change
the memory allocation functions, and ::vl_set_printf_func
to change the logging function.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section misc-error Error handling
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

Some VLFeat functions signal errors in a way similar to the
standard C library. In case of error, a VLFeat function
may return an error code directly,
or an invalid result (for instance a negative file descriptor or a null
pointer). Then ::vl_get_last_error and ::vl_get_last_error_message can be used
to retrieve further details about the error (these functions should be
used right after an error has occurred, before any other VLFeat call).

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section misc-memory Memory allocation
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

VLFeat uses the ::vl_malloc, ::vl_realloc, ::vl_calloc and ::vl_free
functions to allocate memory. Normally these functions are mapped to
the underlying standard C library implementations. However
::vl_set_alloc_func can be used to map them to other
implementations.  For instance, in MATALB MEX files these functions
are mapped to the MATLAB equivalent which has a garbage collection
mechanism to cope with interruptions during execution.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section misc-logging Logging
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

VLFeat uses the macros ::VL_PRINT and ::VL_PRINTF to print progress
or debug informations. These functions are normally mapped to the @c
printf function of the underlying standard C library. However
::vl_set_printf_func can be used to map it to a different
implementation. For instance, in MATLAB MEX files this function is
mapped to @c mexPrintf. Setting the function to @c NULL disables
logging.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section misc-time Measuring time
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

VLFeat provides ::vl_tic and ::vl_toc as an easy way of measuring
elapsed time.

**/

/**
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@page threads Threading
@tableofcontents
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

VLFeat supports for threaded computations can be used to take advantage
of multi-core architectures. Threading support includes:

- Supporting using VLFeat functions and objects from multiple threads
  simultaneously. This is discussed in @ref threads-multiple.
- Using multiple cores to accelerate computations. This is
  discussed in @ref threads-parallel.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section threads-multiple Using VLFeat from multiple threads
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

VLFeat can be used from multiple threads simultaneously if proper
rules are followed.

- <b>A VLFeat object instance is accessed only from one thread at any
  given time.</b> Functions operating on objects (member functions)
  are conditionally thread safe: the same function may be called
  simultaneously from multiple threads provided that it operates on
  different, independent objects. However, modifying the same object
  from multiple threads (using the same or different member functions)
  is possible only from one thread at any given time, and should
  therefore be synchronized. Certain VLFeat objects may contain
  features specific to simplify multi-threaded operations
  (e.g. ::VlKDForest).
- <b>Thread-safe global functions are used.</b> These include
  thread-specific operations such as retrieving the last error by
  ::vl_get_last_error and obtaining the thread-specific random number
  generator instance by ::vl_get_rand. In these cases, the functions
  operate on thread-specific data that VLFeat creates and
  maintains. Note in particular that each thread has an independent
  default random number generator (as returned by
  ::vl_get_rand). VLFeat objects that involve using random numbers
  will typically use the random number generator of the thread
  currently accessing the object (although an object-specific
  generator can be often be specified instead).
- <b>Any other global function is considered non-thread safe and is
  accessed exclusively by one thread at a time.</b> A small number of
  operations are non-reentrant <em>and</em> affect all threads
  simultaneously. These are restricted to changing certain global
  configuration parameters, such as the memory allocation functions by
  ::vl_set_alloc_func. These operations are <em>not</em> thread safe
  and are preferably executed before multiple threads start to operate
  with the library.

<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->
@section threads-parallel Parallel computations
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  -->

VLFeat uses OpenMP to implement parallel computations. Generally, this
means that multiple cores are uses appropriately and transparently,
provided that other multi-threaded parts of the application use OpenMP
and that VLFeat and the application link to the same OpenMP library.
If finer control is required, read on.

VLFeat functions avoids affecting OpenMP global state, including the
desired number of computational threads, in order to minimize side
effects to the linked application (e.g. MATLAB). Instead, VLFeat
duplicates a few OpenMP control parameters when needed (this approach
is similar to the method used by other libraries such as Intel MKL).

The maximum number of threads available to the application can be
obtained by ::vl_get_thread_limit (for OpenMP version 3.0 and
greater). This limit is controlled by the OpenMP library (the function
is a wrapper around @c omp_get_thread_limit), which in turn may
determined that based on the number of computational cores or the
value of the @c OMP_THREAD_LIMIT variable when the program is
launched. This value is an upper bound on the number of computation
threads that can be used at any time.

The maximum number of computational thread that VLFeat should use is
set by ::vl_set_num_threads() and retrieved by ::vl_get_max_threads().
This number is a target value as well as an upper bound to the number
of threads used by VLFeat. This value is stored in the VLFeat private
state and is not necessarily equal to the corresponding OpenMP state
variable retrieved by calling @c omp_get_max_threads(). @c
vl_set_num_threads(1) disables the use of multiple threads and @c
vl_set_num_threads(0) uses the value returned by the OpenMP call @c
omp_get_max_threads(). The latter value is controlled, for example, by
calling @c omp_set_num_threads() in the application. Note that:

- @c vl_set_num_threads(0) determines the number of treads using @c
  omp_get_max_threads() *when it is called*. Subsequent calls to @c
  omp_set_num_threads() will therefore *not* affect the number of
  threads used by VLFeat.
- @c vl_set_num_threads(vl_get_thread_limit()) causes VLFeat use all
  the available threads, regardless on the number of threads set
  within the application by calls to @c omp_set_num_threads().
- OpenMP may still dynamically decide to use a smaller number of
  threads in any specific parallel computation.

@sa http://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_userguide_win/GUID-C2295BC8-DD22-466B-94C9-5FAA79D4F56D.htm
 http://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_userguide_win/index.htm#GUID-DEEF0363-2B34-4BAB-87FA-A75DBE842040.htm
 http://software.intel.com/sites/products/documentation/hpc/mkl/lin/MKL_UG_managing_performance/Using_Additional_Threading_Control.htm

**/

#include "generic.h"

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>

#if defined(VL_OS_WIN)
#include <Windows.h>
#endif

#if ! defined(VL_DISABLE_THREADS) && defined(VL_THREADS_POSIX)
#include <pthread.h>
#endif

#if defined(VL_OS_MACOSX) || defined(VL_OS_LINUX)
#include <unistd.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

/* ---------------------------------------------------------------- */
/*                                         Global and thread states */
/* ---------------------------------------------------------------- */

/* Thread state */
typedef struct _VlThreadState
{
  /* errors */
  int lastError ;
  char lastErrorMessage [VL_ERR_MSG_LEN] ;

  /* random number generator */
  VlRand rand ;

  /* time */
#if defined(VL_OS_WIN)
  LARGE_INTEGER ticFreq ;
  LARGE_INTEGER ticMark ;
#else
  clock_t ticMark ;
#endif
} VlThreadState ;

/* Gobal state */
typedef struct _VlState
{
  /* The thread state uses either a mutex (POSIX)
    or a critical section (Win) */
#if defined(VL_DISABLE_THREADS)
  VlThreadState * threadState ;
#else
#if defined(VL_THREADS_POSIX)
  pthread_key_t threadKey ;
  pthread_mutex_t mutex ;
  pthread_t mutexOwner ;
  pthread_cond_t mutexCondition ;
  size_t mutexCount ;
#elif defined(VL_THREADS_WIN)
  DWORD tlsIndex ;
  CRITICAL_SECTION mutex ;
#endif
#endif /* VL_DISABLE_THREADS */

  /* Configurable functions */
  int   (*printf_func)  (char const * format, ...) ;
  void *(*malloc_func)  (size_t) ;
  void *(*realloc_func) (void*,size_t) ;
  void *(*calloc_func)  (size_t, size_t) ;
  void  (*free_func)    (void*) ;

#if defined(VL_ARCH_IX86) || defined(VL_ARCH_X64) || defined(VL_ARCH_IA64)
  VlX86CpuInfo cpuInfo ;
#endif
  vl_size numCPUs ;
  vl_bool simdEnabled ;
  vl_size numThreads ;
} VlState ;

/* Global state instance */
VlState _vl_state ;

/* ----------------------------------------------------------------- */
VL_INLINE VlState * vl_get_state () ;
VL_INLINE VlThreadState * vl_get_thread_specific_state () ;
static void vl_lock_state (void) ;
static void vl_unlock_state (void) ;
static VlThreadState * vl_thread_specific_state_new (void) ;
static void vl_thread_specific_state_delete (VlThreadState * self) ;

/** @brief Get VLFeat version string
 ** @return the library version string.
 **/

char const *
vl_get_version_string ()
{
  return VL_VERSION_STRING ;
}

/** @brief Get VLFeat configuration string.
 ** @return a new configuration string.
 **
 ** The function returns a new string containing a human readable
 ** description of the library configuration.
 **/

char *
vl_configuration_to_string_copy ()
{
  char * string = 0 ;
  int length = 0 ;
  char * staticString = vl_static_configuration_to_string_copy() ;
  char * cpuString =
#if defined(VL_ARCH_IX86) || defined(VL_ARCH_X64) || defined(VL_ARCH_IA64)
  _vl_x86cpu_info_to_string_copy(&vl_get_state()->cpuInfo) ;
#else
  "Generic CPU" ;
#endif
#if defined(DEBUG)
  int const debug = 1 ;
#else
  int const debug = 0 ;
#endif

  while (string == 0) {
    if (length > 0) {
      string = vl_malloc(sizeof(char) * length) ;
      if (string == NULL) break ;
    }
    length = snprintf(string, length,
                      "VLFeat version %s\n"
                      "    Static config: %s\n"
                      "    %" VL_FMT_SIZE " CPU(s): %s\n"
#if defined(_OPENMP)
                      "    OpenMP: max threads: %d (library: %" VL_FMT_SIZE ")\n"
#endif
                      "    Debug: %s\n",
                      vl_get_version_string (),
                      staticString,
                      vl_get_num_cpus(), cpuString,
#if defined(_OPENMP)
                      omp_get_max_threads(), vl_get_max_threads(),
#endif
                      VL_YESNO(debug)) ;
    length += 1 ;
  }

  if (staticString) vl_free(staticString) ;
  if (cpuString) vl_free(cpuString) ;
  return string ;
}

/** @internal @brief A printf that does not do anything */
static int
do_nothing_printf (char const* format VL_UNUSED, ...)
{
  return 0 ;
}

/** @internal
 ** @brief Lock VLFeat state
 **
 ** The function locks VLFeat global state mutex.
 **
 ** The mutex is recursive: locking multiple times from the same thread
 ** is a valid operations, but requires an equivalent number
 ** of calls to ::vl_unlock_state.
 **
 ** @sa ::vl_unlock_state
 **/

static void
vl_lock_state (void)
{
#if ! defined(VL_DISABLE_THREADS)
#if   defined(VL_THREADS_POSIX)
  VlState * state = vl_get_state () ;
  pthread_t thisThread = pthread_self () ;
  pthread_mutex_lock (&state->mutex) ;
  if (state->mutexCount >= 1 &&
      pthread_equal (state->mutexOwner, thisThread)) {
    state->mutexCount ++ ;
  } else {
    while (state->mutexCount >= 1) {
      pthread_cond_wait (&state->mutexCondition, &state->mutex) ;
    }
    state->mutexOwner = thisThread ;
    state->mutexCount = 1 ;
  }
  pthread_mutex_unlock (&state->mutex) ;
#elif defined(VL_THREADS_WIN)
  EnterCriticalSection (&vl_get_state()->mutex) ;
#endif
#endif
}

/** @internal
 ** @brief Unlock VLFeat state
 **
 ** The function unlocks VLFeat global state mutex.
 **
 ** @sa ::vl_lock_state
 **/

static void
vl_unlock_state (void)
{
#if ! defined(VL_DISABLE_THREADS)
#if   defined(VL_THREADS_POSIX)
  VlState * state = vl_get_state () ;
  pthread_mutex_lock (&state->mutex) ;
  -- state->mutexCount ;
  if (state->mutexCount == 0) {
    pthread_cond_signal (&state->mutexCondition) ;
  }
  pthread_mutex_unlock (&state->mutex) ;
#elif defined(VL_THREADS_WIN)
  LeaveCriticalSection (&vl_get_state()->mutex) ;
#endif
#endif
}

/** @internal
 ** @brief Return VLFeat global state
 **
 ** The function returns a pointer to VLFeat global state.
 **
 ** @return pointer to the global state structure.
 **/

VL_INLINE VlState *
vl_get_state (void)
{
  return &_vl_state ;
}

/** @internal@brief Get VLFeat thread state
 ** @return pointer to the thread state structure.
 **
 ** The function returns a pointer to VLFeat thread state.
 **/

VL_INLINE VlThreadState *
vl_get_thread_specific_state (void)
{
#ifdef VL_DISABLE_THREADS
  return vl_get_state()->threadState ;
#else
  VlState * state ;
  VlThreadState * threadState ;

  vl_lock_state() ;
  state = vl_get_state() ;

#if defined(VL_THREADS_POSIX)
  threadState = (VlThreadState *) pthread_getspecific(state->threadKey) ;
#elif defined(VL_THREADS_WIN)
  threadState = (VlThreadState *) TlsGetValue(state->tlsIndex) ;
#endif

  if (! threadState) {
    threadState = vl_thread_specific_state_new () ;
  }

#if defined(VL_THREADS_POSIX)
  pthread_setspecific(state->threadKey, threadState) ;
#elif defined(VL_THREADS_WIN)
  TlsSetValue(state->tlsIndex, threadState) ;
#endif

  vl_unlock_state() ;
  return threadState ;
#endif
}

/* ---------------------------------------------------------------- */
/** @brief Get the number of CPU cores of the host
 ** @return number of CPU cores.
 **/

vl_size
vl_get_num_cpus (void)
{
  return vl_get_state()->numCPUs ;
}

/** @fn ::vl_set_simd_enabled(vl_bool)
 ** @brief Toggle usage of SIMD instructions
 ** @param x @c true if SIMD instructions are used.
 **
 ** Notice that SIMD instructions are used only if the CPU model
 ** supports them. Note also that data alignment may restrict the use
 ** of such instructions.
 **
 ** @see ::vl_cpu_has_sse2(), ::vl_cpu_has_sse3(), etc.
 **/

void
vl_set_simd_enabled (vl_bool x)
{
  vl_get_state()->simdEnabled = x ;
}

/** @brief Are SIMD instructons enabled?
 ** @return @c true if SIMD instructions are enabled.
 **/

vl_bool
vl_get_simd_enabled (void)
{
  return vl_get_state()->simdEnabled ;
}

/** @brief Check for AVX instruction set
 ** @return @c true if AVX is present.
 **/

vl_bool
vl_cpu_has_avx (void)
{
#if defined(VL_ARCH_IX86) || defined(VL_ARCH_X64) || defined(VL_ARCH_IA64)
  return vl_get_state()->cpuInfo.hasAVX ;
#else
  return VL_FALSE ;
#endif
}

/** @brief Check for SSE3 instruction set
 ** @return @c true if SSE3 is present.
 **/

vl_bool
vl_cpu_has_sse3 (void)
{
#if defined(VL_ARCH_IX86) || defined(VL_ARCH_X64) || defined(VL_ARCH_IA64)
  return vl_get_state()->cpuInfo.hasSSE3 ;
#else
  return VL_FALSE ;
#endif
}

/** @brief Check for SSE2 instruction set
 ** @return @c true if SSE2 is present.
 **/

vl_bool
vl_cpu_has_sse2 (void)
{
#if defined(VL_ARCH_IX86) || defined(VL_ARCH_X64) || defined(VL_ARCH_IA64)
  return vl_get_state()->cpuInfo.hasSSE2 ;
#else
  return VL_FALSE ;
#endif
}

/* ---------------------------------------------------------------- */

/** @brief Get the number of computational threads available to the application
 ** @return number of threads.
 **
 ** This function wraps the OpenMP function @c
 ** omp_get_thread_limit(). If VLFeat was compiled without OpenMP
 ** support, this function returns 1. If VLFeat was compiled with
 ** OpenMP prior to version 3.0 (2008/05), it returns 0.
 **
 ** @sa @ref threads-parallel
 **/

vl_size
vl_get_thread_limit (void)
{
#if defined(_OPENMP)
#if _OPENMP >= 200805
  /* OpenMP version >= 3.0 */
  return omp_get_thread_limit() ;
#else
  return 0 ;
#endif
#else
  return 1 ;
#endif
}

/** @brief Get the maximum number of computational threads used by VLFeat.
 ** @return number of threads.
 **
 ** This function returns the maximum number of thread used by
 ** VLFeat. VLFeat will try to use this number of computational
 ** threads and never exceed it.
 **
 ** This is similar to the OpenMP function @c omp_get_max_threads();
 ** however, it reads a parameter private to VLFeat which is
 ** independent of the value used by the OpenMP library.
 **
 ** If VLFeat was compiled without OpenMP support, this function
 ** returns 1.
 **
 ** @sa vl_set_num_threads(), @ref threads-parallel
 **/

vl_size
vl_get_max_threads (void)
{
#if defined(_OPENMP)
  return vl_get_state()->numThreads ;
#else
  return 1 ;
#endif
}

/** @brief Set the maximum number of threads used by VLFeat.
 ** @param numThreads number of threads to use.
 **
 ** This function sets the maximum number of computational threads
 ** that will be used by VLFeat. VLFeat may in practice use fewer
 ** threads (for example because @a numThreads is larger than the
 ** number of computational cores in the host, or because the number
 ** of threads exceeds the limit available to the application).
 **
 ** If @c numThreads is set to 0, then VLFeat sets the number of
 ** threads to the OpenMP current maximum, obtained by calling @c
 ** omp_get_max_threads().
 **
 ** This function is similar to @c omp_set_num_threads() but changes a
 ** parameter internal to VLFeat rather than affecting OpenMP global
 ** state.
 **
 ** If VLFeat was compiled without, this function does nothing.
 **
 ** @sa vl_get_max_threads(), @ref threads-parallel
 **/

#if defined(_OPENMP)
void
vl_set_num_threads (vl_size numThreads)
{
  if (numThreads == 0) {
    numThreads = omp_get_max_threads() ;
  }
  vl_get_state()->numThreads = numThreads ;
}
#else
void
vl_set_num_threads (vl_size numThreads VL_UNUSED) { }
#endif

/* ---------------------------------------------------------------- */
/** @brief Set last VLFeat error
 ** @param error error code.
 ** @param errorMessage error message format string.
 ** @param ... format string arguments.
 ** @return error code.
 **
 ** The function sets the code and optionally the error message
 ** of the last encountered error. @a errorMessage is the message
 ** format. It uses the @c printf convention and is followed by
 ** the format arguments. The maximum length of the error message is
 ** given by ::VL_ERR_MSG_LEN (longer messages are truncated).
 **
 ** Passing @c NULL as @a errorMessage
 ** sets the error message to the empty string.
 **/

int
vl_set_last_error (int error, char const * errorMessage, ...)
{
  VlThreadState * state = vl_get_thread_specific_state() ;
  va_list args;
  va_start(args, errorMessage) ;
  if (errorMessage) {
#ifdef VL_COMPILER_LCC
    vsprintf(state->lastErrorMessage, errorMessage, args) ;
#else
    vsnprintf(state->lastErrorMessage,
              sizeof(state->lastErrorMessage)/sizeof(char),
              errorMessage, args) ;
#endif
  } else {
    state->lastErrorMessage[0] = 0 ;
  }
  state->lastError = error ;
  va_end(args) ;
  return error ;
}

/** @brief Get the code of the last error
 ** @return error code.
 ** @sa ::vl_get_last_error_message.
 **/

int
vl_get_last_error (void) {
  return vl_get_thread_specific_state()->lastError ;
}

/** @brief Get the last error message
 ** @return pointer to the error message.
 ** @sa ::vl_get_last_error.
 **/

char const *
vl_get_last_error_message (void)
{
  return vl_get_thread_specific_state()->lastErrorMessage ;
}

/* ---------------------------------------------------------------- */
/** @brief Set memory allocation functions
 ** @param malloc_func  pointer to @c malloc.
 ** @param realloc_func pointer to @c realloc.
 ** @param calloc_func  pointer to @c calloc.
 ** @param free_func    pointer to @c free.
 **/

void
vl_set_alloc_func (void *(*malloc_func)  (size_t),
                   void *(*realloc_func) (void*, size_t),
                   void *(*calloc_func)  (size_t, size_t),
                   void  (*free_func)    (void*))
{
  VlState * state ;
  vl_lock_state () ;
  state = vl_get_state() ;
  state->malloc_func  = malloc_func ;
  state->realloc_func = realloc_func ;
  state->calloc_func  = calloc_func ;
  state->free_func    = free_func ;
  vl_unlock_state () ;
}

/** @brief Allocate a memory block
 ** @param n size in bytes of the new block.
 ** @return pointer to the allocated block.
 **
 ** This function allocates a memory block of the specified size.
 ** The synopsis is the same as the POSIX @c malloc function.
 **/

void *
vl_malloc (size_t n)
{
  return (vl_get_state()->malloc_func)(n) ;
  //return (memalign)(32,n) ;
}


/** @brief Reallocate a memory block
 ** @param ptr pointer to a memory block previously allocated.
 ** @param n size in bytes of the new block.
 ** @return pointer to the new block.
 **
 ** This function reallocates a memory block to change its size.
 ** The synopsis is the same as the POSIX @c realloc function.
 **/

void *
vl_realloc (void* ptr, size_t n)
{
  return (vl_get_state()->realloc_func)(ptr, n) ;
}

/** @brief Free and clear a memory block
 ** @param n number of items to allocate.
 ** @param size size in bytes of an item.
 ** @return pointer to the new block.
 **
 ** This function allocates and clears a memory block.
 ** The synopsis is the same as the POSIX @c calloc function.
 **/

void *
vl_calloc (size_t n, size_t size)
{
  return (vl_get_state()->calloc_func)(n, size) ;
}

/** @brief Free a memory block
 ** @param ptr pointer to the memory block.
 **
 ** This function frees a memory block allocated by ::vl_malloc,
 ** ::vl_calloc, or ::vl_realloc. The synopsis is the same as the POSIX
 ** @c malloc function.
 **/

void
vl_free (void *ptr)
{
  (vl_get_state()->free_func)(ptr) ;
}

/* ---------------------------------------------------------------- */

/** @brief Set the printf function
 ** @param printf_func pointer to a @c printf implementation.
 ** Set @c print_func to NULL to disable printf.
 **/

void
vl_set_printf_func (printf_func_t printf_func)
{
  vl_get_state()->printf_func = printf_func ? printf_func : do_nothing_printf ;
}

/** @brief Get the printf function
 ** @return printf_func pointer to the @c printf implementation.
 ** @sa ::vl_set_printf_func.
 **/

printf_func_t
vl_get_printf_func (void) {
  return vl_get_state()->printf_func ;
}

/* ---------------------------------------------------------------- */
/** @brief Get processor time
 ** @return processor time in seconds.
 ** @sa ::vl_tic, ::vl_toc
 **/

double
vl_get_cpu_time ()
{
  #ifdef VL_OS_WIN
  VlThreadState * threadState = vl_get_thread_specific_state() ;
  LARGE_INTEGER mark ;
  QueryPerformanceCounter (&mark) ;
  return (double)mark.QuadPart / (double)threadState->ticFreq.QuadPart ;
#else
  return (double)clock() / (double)CLOCKS_PER_SEC ;
#endif
}

/** @brief Reset processor time reference
 ** The function resets VLFeat TIC/TOC time reference. There is one
 ** such reference per thread.
 ** @sa ::vl_get_cpu_time, ::vl_toc.
 **/

void
vl_tic (void)
{
  VlThreadState * threadState = vl_get_thread_specific_state() ;
#ifdef VL_OS_WIN
  QueryPerformanceCounter (&threadState->ticMark) ;
#else
  threadState->ticMark = clock() ;
#endif
}

/** @brief Get elapsed time since tic
 ** @return elapsed time in seconds.
 **
 ** The function
 ** returns the processor time elapsed since ::vl_tic was called last.
 **
 ** @remark In multi-threaded applications, there is an independent
 ** timer for each execution thread.
 **
 ** @remark On UNIX, this function uses the @c clock() system call.
 ** On Windows, it uses the @c QueryPerformanceCounter() system call,
 ** which is more accurate than @c clock() on this platform.
 **/

double
vl_toc (void)
{
  VlThreadState * threadState = vl_get_thread_specific_state() ;
#ifdef VL_OS_WIN
  LARGE_INTEGER tocMark ;
  QueryPerformanceCounter(&tocMark) ;
  return (double) (tocMark.QuadPart - threadState->ticMark.QuadPart) /
    threadState->ticFreq.QuadPart ;
#else
  return (double) (clock() - threadState->ticMark) / CLOCKS_PER_SEC ;
#endif
}

/* ---------------------------------------------------------------- */
/** @brief Get the default random number generator.
 ** @return random number generator.
 **
 ** The function returns a pointer to the default
 ** random number generator.
 ** There is one such generator per thread.
 **/

VL_EXPORT VlRand *
vl_get_rand (void)
{
  return &vl_get_thread_specific_state()->rand ;
}

/* ---------------------------------------------------------------- */
/*                    Library construction and destruction routines */
/*  --------------------------------------------------------------- */

/** @internal@brief Construct a new thread state object
 ** @return new state structure.
 **/

static VlThreadState *
vl_thread_specific_state_new (void)
{
  VlThreadState * self ;
#if defined(DEBUG)
  printf("VLFeat DEBUG: thread constructor begins.\n") ;
#endif
  self = malloc(sizeof(VlThreadState)) ;
  self->lastError = 0 ;
  self->lastErrorMessage[0] = 0 ;
#if defined(VL_OS_WIN)
  QueryPerformanceFrequency (&self->ticFreq) ;
  self->ticMark.QuadPart = 0 ;
#else
  self->ticMark = 0 ;
#endif
  vl_rand_init (&self->rand) ;

  return self ;
}

/** @internal@brief Delete a thread state structure
 ** @param self thread state object.
 **/

static void
vl_thread_specific_state_delete (VlThreadState * self)
{
#if defined(DEBUG)
  printf("VLFeat DEBUG: thread destructor begins.\n") ;
#endif
  free (self) ;
}
/* ---------------------------------------------------------------- */
/*                                        DLL entry and exit points */
/* ---------------------------------------------------------------- */
/* A constructor and a destructor must be called to initialize or dispose of VLFeat
 * state when the DLL is loaded or unloaded. This is obtained
 * in different ways depending on the operating system.
 */

#if (defined(VL_OS_LINUX) || defined(VL_OS_MACOSX)) && defined(VL_COMPILER_GNUC)
static void vl_constructor () __attribute__ ((constructor)) ;
static void vl_destructor () __attribute__ ((destructor))  ;
#endif

#if defined(VL_OS_WIN)
static void vl_constructor () ;
static void vl_destructor () ;

BOOL WINAPI DllMain(
    HINSTANCE hinstDLL,  // handle to DLL module
    DWORD fdwReason,     // reason for calling function
    LPVOID lpReserved )  // reserved
{
  VlState * state ;
  VlThreadState * threadState ;
  switch (fdwReason) {
    case DLL_PROCESS_ATTACH:
      /* Initialize once for each new process */
      vl_constructor () ;
      break ;

    case DLL_THREAD_ATTACH:
      /* Do thread-specific initialization */
      break ;

    case DLL_THREAD_DETACH:
      /* Do thread-specific cleanup */
#if ! defined(VL_DISABLE_THREADS) && defined(VL_THREADS_WIN)
      state = vl_get_state() ;
      threadState = (VlThreadState*) TlsGetValue(state->tlsIndex) ;
      if (threadState) {
        vl_thread_specific_state_delete (threadState) ;
      }
#endif
      break;

    case DLL_PROCESS_DETACH:
      /* Perform any necessary cleanup */
      vl_destructor () ;
      break;
    }
    return TRUE ; /* Successful DLL_PROCESS_ATTACH */
}
#endif /* VL_OS_WIN */

/* ---------------------------------------------------------------- */
/*                               Library constructor and destructor */
/* ---------------------------------------------------------------- */

/** @internal @brief Initialize VLFeat state */
static void
vl_constructor (void)
{
  VlState * state ;
#if defined(DEBUG)
  printf("VLFeat DEBUG: constructor begins.\n") ;
#endif

  state = vl_get_state() ;

#if ! defined(VL_DISABLE_THREADS)
#if defined(DEBUG)
  printf("VLFeat DEBUG: constructing thread specific state.\n") ;
#endif
#if defined(VL_THREADS_POSIX)
  {
    typedef void (*destructorType)(void * );
    pthread_key_create (&state->threadKey,
                        (destructorType)
                          vl_thread_specific_state_delete) ;
    pthread_mutex_init (&state->mutex, NULL) ;
    pthread_cond_init (&state->mutexCondition, NULL) ;
  }
#elif defined(VL_THREADS_WIN)
  InitializeCriticalSection (&state->mutex) ;
  state->tlsIndex = TlsAlloc () ;
#endif
#else

/* threading support disabled */
#if defined(DEBUG)
  printf("VLFeat DEBUG: constructing the generic thread state instance (threading support disabled).\n") ;
#endif
  vl_get_state()->threadState = vl_thread_specific_state_new() ;
#endif

  state->malloc_func  = malloc ;
  state->realloc_func = realloc ;
  state->calloc_func  = calloc ;
  state->free_func    = free ;
  state->printf_func  = printf ;

  /* on x86 platforms read the CPUID register */
#if defined(VL_ARCH_IX86) || defined(VL_ARCH_X64) || defined(VL_ARCH_IA64)
  _vl_x86cpu_info_init (&state->cpuInfo) ;
#endif

  /* get the number of CPUs */
#if defined(VL_OS_WIN)
  {
    SYSTEM_INFO info;
    GetSystemInfo (&info) ;
    state->numCPUs = info.dwNumberOfProcessors ;
  }
#elif defined(VL_OS_MACOSX) || defined(VL_OS_LINUX)
  state->numCPUs = sysconf(_SC_NPROCESSORS_ONLN) ;
#else
  state->numCPUs = 1 ;
#endif
  state->simdEnabled = VL_TRUE ;

  /* get the number of (OpenMP) threads used by the library */
#if defined(_OPENMP)
  state->numThreads = omp_get_max_threads() ;
#else
  state->numThreads = 1 ;
#endif

#if defined(DEBUG)
  printf("VLFeat DEBUG: constructor ends.\n") ;
#endif
}

/** @internal @brief Destruct VLFeat */
static void
vl_destructor ()
{
  VlState * state ;
#if defined(DEBUG)
  printf("VLFeat DEBUG: destructor begins.\n") ;
#endif

  state = vl_get_state() ;

#if ! defined(VL_DISABLE_THREADS)
#if defined(DEBUG)
  printf("VLFeat DEBUG: destroying a thread specific state instance.\n") ;
#endif
#if   defined(VL_THREADS_POSIX)
  {
    /* Delete the thread state of this thread as the
       destructor is not called by pthread_key_delete or after
       the key is deleted. When the library
       is unloaded, this thread should also be the last one
       using the library, so this is fine.
     */
    VlThreadState * threadState =
       pthread_getspecific(state->threadKey) ;
    if (threadState) {
      vl_thread_specific_state_delete (threadState) ;
      pthread_setspecific(state->threadKey, NULL) ;
    }
  }
  pthread_cond_destroy (&state->mutexCondition) ;
  pthread_mutex_destroy (&state->mutex) ;
  pthread_key_delete (state->threadKey) ;
#elif defined(VL_THREADS_WIN)
 {
    /* Delete the thread state of this thread as the
       destructor is not called by pthread_key_delete or after
       the key is deleted. When the library
       is unloaded, this thread should also be the last one
       using the library, so this is fine.
     */
    VlThreadState * threadState =
       TlsGetValue(state->tlsIndex) ;
    if (threadState) {
      vl_thread_specific_state_delete (threadState) ;
      TlsSetValue(state->tlsIndex, NULL) ;
    }
  }
  TlsFree (state->tlsIndex) ;
  DeleteCriticalSection (&state->mutex) ;
#endif
#else
#if defined(DEBUG)
  printf("VLFeat DEBUG: destroying the generic thread state instance (threading support disabled).\n") ;
#endif
  vl_thread_specific_state_delete(vl_get_state()->threadState) ;
#endif

#if defined(DEBUG)
  printf("VLFeat DEBUG: destructor ends.\n") ;
#endif
}
