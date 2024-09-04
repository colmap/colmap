


incremental_mapper
```c++
// IncrementalPipeline src/colmap/sfm/incremental_mapper.h
IncrementalPipeline::Reconstruct 
    -> IncrementalPipeline::ReconstructSubModel 
        -> IncrementalMapper::BeginReconstruction
        -> IncrementalPipeline::InitializeReconstruction
            ->IncrementalMapper::FindInitialImagePair
            ->IncrementalMapper::EstimateInitialTwoViewGeometry
            ->IncrementalMapper::RegisterInitialImagePair
            ->IncrementalMapper::AdjustGlobalBundle
            ->IncrementalMapper::FilterPoints
            ->IncrementalMapper::FilterImages
        -> IncrementalMapper::FindNextImages
        -> IncrementalMapper::RegisterNextImage
        -> IncrementalMapper::TriangulateImage
            ->IncrementalTriangulator::TriangulateImage
        -> IncrementalMapper::IterativeLocalRefinement
        -> IncrementalPipeline::CheckRunGlobalRefinement
        -> IncrementalPipeline::IterativeGlobalRefinement
```

