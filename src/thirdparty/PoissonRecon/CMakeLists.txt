if(NOT IS_MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -ffast-math -std=c++11")
endif()

add_definitions("-DRELEASE")

COLMAP_ADD_LIBRARY(
    NAME colmap_poisson_recon
    SRCS
        Allocator.h
        Array.h
        Array.inl
        BSplineData.h
        BSplineData.inl
        BinaryNode.h
        CmdLineParser.h CmdLineParser.cpp
        CmdLineParser.inl
        Factor.h Factor.cpp
        FunctionData.h
        FunctionData.inl
        Geometry.h Geometry.cpp
        Geometry.inl
        Hash.h
        MAT.h
        MAT.inl
        MarchingCubes.h MarchingCubes.cpp
        MemoryUsage.h
        MultiGridOctreeData.Evaluation.inl
        MultiGridOctreeData.IsoSurface.inl
        MultiGridOctreeData.SortedTreeNodes.inl
        MultiGridOctreeData.System.inl
        MultiGridOctreeData.WeightedSamples.inl
        MultiGridOctreeData.h
        MultiGridOctreeData.inl
        MyTime.h
        Octree.h
        Octree.inl
        PPolynomial.h
        PPolynomial.inl
        Ply.h
        PointStream.h PlyFile.cpp
        PointStream.inl
        PoissonRecon.h PoissonRecon.cpp
        Polynomial.h
        Polynomial.inl
        SparseMatrix.h
        SparseMatrix.inl
        SurfaceTrimmer.h SurfaceTrimmer.cpp
)
if(OPENMP_FOUND)
    target_link_libraries(colmap_poisson_recon PRIVATE OpenMP::OpenMP_CXX)
endif()
