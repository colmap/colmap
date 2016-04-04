set(VocabLib_SOURCE_FILES
    defines.h
    kmeans.cpp
    kmeans.h
    kmeans_kd.cpp
    kmeans_kd.h
    qsort.c
    qsort.h
    util.c
    util.h
    VocabFlatNode.cpp
    VocabTree.cpp
    VocabTree.h
    VocabTreeBuild.cpp
    VocabTreeIO.cpp
    VocabTreeUtil.cpp
)

COLMAP_ADD_LIBRARY(vocab_lib ${VocabLib_SOURCE_FILES})
