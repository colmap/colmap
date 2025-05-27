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

if(SIMD_ENABLED AND IS_X86)
    if(MSVC)
        # https://github.com/vlfeat/vlfeat/commit/4f0098fd47e9
        add_compile_definitions(VL_DISABLE_AVX)
    else()
        set(AVX_SOURCES
            mathop_avx.c
            mathop_avx.h)
    endif()

    set(SSE2_SOURCES
        imopv_sse2.c
        imopv_sse2.h
        mathop_sse2.c
        mathop_sse2.h)

    list(APPEND VLFEAT_SOURCE_FILES ${AVX_SOURCES} ${SSE2_SOURCES})

    if(MSVC)
        set_source_files_properties(${AVX_SOURCES}
            PROPERTIES COMPILE_FLAGS "/arch:AVX")
        set_source_files_properties(${SSE2_SOURCES}
            PROPERTIES COMPILE_FLAGS "/D__SSE2__")
    else()
        set_source_files_properties(${AVX_SOURCES}
            PROPERTIES COMPILE_FLAGS "-mavx")
        set_source_files_properties(${SSE2_SOURCES}
            PROPERTIES COMPILE_FLAGS "-msse2")
    endif()
else()
    add_compile_definitions(VL_DISABLE_AVX)
    add_compile_definitions(VL_DISABLE_SSE2)
endif()

if(NOT OPENMP_ENABLED OR NOT OPENMP_FOUND)
    add_compile_definitions(VL_DISABLE_OPENMP)
endif()

COLMAP_ADD_LIBRARY(
    NAME colmap_vlfeat
    SRCS ${VLFEAT_SOURCE_FILES}
)
if(OPENMP_FOUND)
    target_link_libraries(colmap_vlfeat PRIVATE OpenMP::OpenMP_C)
endif()
