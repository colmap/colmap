
COLMAP_VERSION = "3.11.0.dev0"


# incremental_mapper
```c++
// IncrementalPipeline src/colmap/sfm/incremental_mapper.h
IncrementalPipeline::Reconstruct() 
    -> IncrementalPipeline::ReconstructSubModel() 
        -> IncrementalMapper::BeginReconstruction()
        -> IncrementalPipeline::InitializeReconstruction()
            ->IncrementalMapper::FindInitialImagePair()
            ->IncrementalMapper::EstimateInitialTwoViewGeometry()
            ->IncrementalMapper::RegisterInitialImagePair()
            ->IncrementalMapper::AdjustGlobalBundle()
            ->IncrementalMapper::FilterPoints()
            ->IncrementalMapper::FilterImages()
        -> IncrementalMapper::FindNextImages()
        -> IncrementalMapper::RegisterNextImage()
        -> IncrementalMapper::TriangulateImage()
            ->IncrementalTriangulator::TriangulateImage()
        -> IncrementalMapper::IterativeLocalRefinement()
        -> IncrementalPipeline::CheckRunGlobalRefinement()
        -> IncrementalPipeline::IterativeGlobalRefinement()
```

# matcher
```c++
// src/colmap/exe/feature.cc
|-> RunFeatureExtractor() 
|-> RunFeatureImporter() 
|-> RunExhaustiveMatcher() 
|-> RunMatchesImporter() 
|-> RunSequentialMatcher() 
|-> RunSpatialMatcher() 
|-> RunTransitiveMatcher() 
|-> RunVocabTreeMatcher() 
    -> Create matcher
        // src/colmap/controllers/feature_matching.cc
        |-> CreateExhaustiveFeatureMatcher()   -- ExhaustivePairGenerator::
        |-> CreateSpatialFeatureMatcher()      -- SpatialPairGenerator::
        |-> CreateSequentialFeatureMatcher()   -- SequentialPairGenerator::
        |-> CreateTransitiveFeatureMatcher()   -- TransitivePairGenerator::
        |-> CreateImagePairsFeatureMatcher()   -- ImportedPairGenerator::
        |-> CreateVocabTreeFeatureMatcher()    -- VocabTreePairGenerator::
        |-> CreateFeaturePairsFeatureMatcher() -- FeaturePairsFeatureMatcher::
    -> matcher->Start()
        // src/colmap/controllers/feature_matching.cc
        -> GenericFeatureMatcher::Run() // call DerivedPairGenerator 
            // src/colmap/controllers/feature_matching_utils.cc
            -> FeatureMatcherController::Setup()
            // src/colmap/feature/matcher.h
            -> FeatureMatcherCache::Setup()
            // src/colmap/feature/pairing.cc
            -> DerivedPairGenerator::Next()
            -> FeatureMatcherController::Match()
                // src/colmap/controllers/feature_matching_utils.cc
                -> FeatureMatcherWorker::Run() 
                    // src/colmap/feature/matcher.h
                    // src/colmap/feature/sift.cc
                    // FeatureMatcher, SiftGPUFeatureMatcher, SiftCPUFeatureMatcher
                    |-> FeatureMatcher::Match() | 
                        -> ComputeSiftDistanceMatrix() 
                        -> FindBestMatchesBruteForce()
                        -> FindBestMatchesIndex()
                    |-> FeatureMatcher::MatchGuided() 
                        -> ComputeSiftDistanceMatrix() 
                        -> FindBestMatchesIndex()
    -> matcher->Wait()
```

# Thread
```c++
Thread:: // src/colmap/util/threading.cc
* Thread::Start()
    -> Thread::Wait()
    -> Thread::RunFunc // std::thread
        -> Thread::Run()
* Thread::Stop()
```