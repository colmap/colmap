set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DRELEASE")

if(NOT IS_MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -ffast-math")
endif()

COLMAP_ADD_LIBRARY(poisson_recon
    Allocator.h
    Array.h
    Array.inl
    BSplineData.h
    BSplineData.inl
    BinaryNode.h
    CmdLineParser.cpp
    CmdLineParser.h
    CmdLineParser.inl
    Factor.cpp
    Factor.h
    FunctionData.h
    FunctionData.inl
    Geometry.cpp
    Geometry.h
    Geometry.inl
    Hash.h
    MAT.h
    MAT.inl
    MarchingCubes.cpp
    MarchingCubes.h
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
    PlyFile.cpp
    PointStream.h
    PointStream.inl
    PoissonRecon.cpp
    PoissonRecon.h
    Polynomial.h
    Polynomial.inl
    SparseMatrix.h
    SparseMatrix.inl
    SurfaceTrimmer.cpp
    SurfaceTrimmer.h
)
