if(IS_GNU OR IS_CLANG)
    if(NOT HAS_AVX_EXTENSION)
        add_definitions(-DVL_DISABLE_AVX)
    endif()

    if(NOT HAS_SSE2_EXTENSION)
        add_definitions(-DVL_DISABLE_SSE2)
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SSE_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS}")
else()
    add_definitions(-DVL_DISABLE_AVX)
    add_definitions(-DVL_DISABLE_SSE2)
endif()

if(NOT OPENMP_ENABLED OR NOT OPENMP_FOUND)
    add_definitions(-DVL_DISABLE_OPENMP)
endif()

set(VLFEAT_SOURCE_FILES
    aib.c
    aib.h
    array.c
    array.h
    covdet.c
    covdet.h
    dsift.c
    dsift.h
    fisher.c
    fisher.h
    float.h
    generic.c
    generic.h
    getopt_long.c
    getopt_long.h
    gmm.c
    gmm.h
    heap-def.h
    hikmeans.c
    hikmeans.h
    hog.c
    hog.h
    homkermap.c
    homkermap.h
    host.c
    host.h
    ikmeans.c
    ikmeans.h
    ikmeans_elkan.tc
    ikmeans_init.tc
    ikmeans_lloyd.tc
    imopv.c
    imopv.h
    imopv_sse2.c
    imopv_sse2.h
    kdtree.c
    kdtree.h
    kmeans.c
    kmeans.h
    lbp.c
    lbp.h
    liop.c
    liop.h
    mathop.c
    mathop.h
    mathop_avx.c
    mathop_avx.h
    mathop_sse2.c
    mathop_sse2.h
    mser.c
    mser.h
    pgm.c
    pgm.h
    qsort-def.h
    quickshift.c
    quickshift.h
    random.c
    random.h
    rodrigues.c
    rodrigues.h
    scalespace.c
    scalespace.h
    shuffle-def.h
    sift.c
    sift.h
    slic.c
    slic.h
    stringop.c
    stringop.h
    svm.c
    svm.h
    svmdataset.c
    svmdataset.h
    vlad.c
    vlad.h)

COLMAP_ADD_LIBRARY(vlfeat ${VLFEAT_SOURCE_FILES})
