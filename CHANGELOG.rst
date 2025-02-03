Changelog
=========

------------------------
COLMAP 3.11 (11/28/2024)
------------------------

New Features
------------
* New pose prior based incremental mapper that can leverage absolute pose priors from e.g. GPS measurements.
* New bundle adjustment covariance estimation functionality. Significantly faster and more robust than Ceres.
* API documentation with auto-generated stubs for pycolmap.
* Use PoseLib's minimal solvers for faster performance and improved robustness.
* Experimental support for CUDA-based bundle adjustment through Ceres (disabled by default).
* Support for reading 16-bit PNG grayscale images.
* New RAD_TAN_THIN_PRISM_FISHEYE camera model in support of Meta's Project Aria devices.
* Replace numerical with analytical Jacobian in image undistortion for better convergence.
* Many more performance optimizations and other improvements. See full list of changes below.

Bug Fixes
---------
* Fixed non-deterministic behavior of CUDA SIFT feature extractor. Broken since 3.10 release.
* Fixed orientation detection of covariant/affine SIFT feature extractor. Broken since initial release.
* Fixed point triangulator crashing due to bug in observation manager. Broken since 3.10 release.
* Fixed sequential feature matcher overlap missing the farthest image. Broken since initial release.
* Fixed rare deadlock during matching due to concurrent database access. Broken since 3.10 release.
* Fixed little/big endian detection. Broken since 3.1 release.
* For other bug fixes, see full list of changes below.

Breaking Changes
----------------
* Dropped official support for Ubuntu 18.04, Visual Studio 2019.
* Upgrade to C++17 standard in C++ and C++14 in CUDA source code.
* New ``pose_priors`` table in database in support of pose prior based mapper.
* PyCOLMAP API:

  * ``align_reconstrution_to_locations`` is renamed to ``align_reconstruction_to_locations`` (typo).
  * ``pycomap.cost_functions`` becomes a module and should be explicitly imported as ``import pycolmap.cost_functions``.
  * Replaced ``Image.registered`` by ``Image.{has_pose,reset_pose}``.
  * Replaced ``Image.{get_valid_point2D_ids,get_valid_points2D}`` by ``Image.{get_observation_point2D_idxs,get_observation_points2D}``.
  * Replaced ``Track.{append,remove}`` by ``Track.{add_element,delete_element}``.
  * ``AbsolutePoseErrorCost`` becomes ``AbsolutePosePriorCost``.
  * ``MetricRelativePoseErrorCost`` becomes ``RelativePosePriorCost``.
  * The signature of ``ReprojErrorCost`` and related cost functions was changed: arguments are reordered, the detection uncertainty is now a 2x2 covariance matrix.
  * ``BundleAdjuster`` becomes virtual and should be created with ``pycolmap.create_default_bundle_adjuster()``.
  * ``absolute_pose_estimation`` becomes ``estimate_and_refine_absolute_pose``.
  * ``pose_refinement`` becomes ``refine_absolute_pose``.
  * ``essential_matrix_estimation`` becomes ``estimate_essential_matrix``.
  * ``fundamental_matrix_estimation`` becomes ``estimate_fundamental_matrix``.
  * ``rig_absolute_pose_estimation`` becomes ``estimate_and_refine_generalized_absolute_pose``.
  * ``homography_matrix_estimation`` becomes ``estimate_homography_matrix``.
  * ``squared_sampson_error`` becomes ``compute_squared_sampson_error``.
  * ``homography_decomposition`` becomes ``pose_from_homography_matrix``.
  * ``Rigid3d.essential_matrix`` becomes ``pycolmap.essential_matrix_from_pose``.

Full Change List (sorted temporally)
------------------------------------
* Updates for pycolmap by @ahojnnes in https://github.com/colmap/colmap/pull/2672
* Trigger CI on release/* branches by @ahojnnes in https://github.com/colmap/colmap/pull/2673
* Use consistent versioning scheme between C++/Python by @ahojnnes in https://github.com/colmap/colmap/pull/2674
* Add cost function for 3D alignment (with covariance) by @B1ueber2y in https://github.com/colmap/colmap/pull/2621
* Numpy 2 compatibility by @sarlinpe in https://github.com/colmap/colmap/pull/2682
* Add fix for specifying the correct pycolmap CMake python development … by @fulkast in https://github.com/colmap/colmap/pull/2683
* Remove non existant flags of model_aligner from docs by @TamirCohen in https://github.com/colmap/colmap/pull/2696
* Reset CMAKE_MODULE_PATH to previous value by @mvieth in https://github.com/colmap/colmap/pull/2699
* Robustify nchoosek against overflow by @ahojnnes in https://github.com/colmap/colmap/pull/2706
* Observation manager needs to check if image_id exists before query operations by @bo-rc in https://github.com/colmap/colmap/pull/2704
* Remove pose prior from database.py:add_image by @sarlinpe in https://github.com/colmap/colmap/pull/2707
* Fix: sequential matcher overlap number should be inclusive by @flm8620 in https://github.com/colmap/colmap/pull/2701
* Fix table mangled by clang-format by @sweber1 in https://github.com/colmap/colmap/pull/2710
* Write out options to ini in full precision, relax bundle adjuster convergence by @ahojnnes in https://github.com/colmap/colmap/pull/2713
* Tests for pairing library in feature matching by @ahojnnes in https://github.com/colmap/colmap/pull/2711
* Rename IncrementalMapperOptions to IncrementalPipelineOptions by @B1ueber2y in https://github.com/colmap/colmap/pull/2708
* Add support for CUDA sparse BA solver by @ahojnnes in https://github.com/colmap/colmap/pull/2717
* Rename HierarchicalMapperController to HierarchicalPipeline by @ahojnnes in https://github.com/colmap/colmap/pull/2718
* Make VisualIndex::Quantize const to improve readability by @IshitaTakeshi in https://github.com/colmap/colmap/pull/2723
* Fix CUDA_ENABLED macro in new bundle adjustment code by @drkoller in https://github.com/colmap/colmap/pull/2725
* Automatically generate stub files by @sarlinpe in https://github.com/colmap/colmap/pull/2721
* Add CUDA-based dense BA solver by @ahojnnes in https://github.com/colmap/colmap/pull/2732
* Improved and simplified caching in feature matching by @ahojnnes in https://github.com/colmap/colmap/pull/2731
* Fix colmap namespace in the macro support of logging. by @B1ueber2y in https://github.com/colmap/colmap/pull/2733
* Add callbacks by move by @ahojnnes in https://github.com/colmap/colmap/pull/2734
* Implement transitive matcher with pair generator + tests by @ahojnnes in https://github.com/colmap/colmap/pull/2735
* Provide reasonable defaults for some estimator options by @sarlinpe in https://github.com/colmap/colmap/pull/2745
* Fix mismatched Delaunay meshing options by @sarlinpe in https://github.com/colmap/colmap/pull/2748
* PyCOLMAP API documentation by @sarlinpe in https://github.com/colmap/colmap/pull/2749
* Improved pycolmap coverage and docs by @sarlinpe in https://github.com/colmap/colmap/pull/2752
* Follow-up fixes in pycolmap by @sarlinpe in https://github.com/colmap/colmap/pull/2755
* Report errors in import_images by @sarlinpe in https://github.com/colmap/colmap/pull/2750
* Further simplification of feature matcher code by @ahojnnes in https://github.com/colmap/colmap/pull/2744
* Add missing ClearModifiedPoints3D by @sarlinpe in https://github.com/colmap/colmap/pull/2761
* Store shared camera ptr for reconstruction images by @ahojnnes in https://github.com/colmap/colmap/pull/2762
* Avoid unnecessary copy of queue in IncrementalTriangulator::Complete() by @ahojnnes in https://github.com/colmap/colmap/pull/2764
* Branch prediction for THROW_CHECK_NOTNULL by @ahojnnes in https://github.com/colmap/colmap/pull/2765
* Use shared camera pointer in more places by @ahojnnes in https://github.com/colmap/colmap/pull/2763
* Support switching camera directly with camera pointer by @B1ueber2y in https://github.com/colmap/colmap/pull/2767
* Add test for MergeReconstructions by @B1ueber2y in https://github.com/colmap/colmap/pull/2766
* Fix little/big endian detection by @ahojnnes in https://github.com/colmap/colmap/pull/2768
* Fix options for CUDA sparse BA solver by @whuaegeanse in https://github.com/colmap/colmap/pull/2758
* Rename SupperMeasurer::Compare for improved readability by @ahojnnes in https://github.com/colmap/colmap/pull/2774
* Improvements for install docs by @ahojnnes in https://github.com/colmap/colmap/pull/2773
* fixed typo of align_reconstrution_to_locations to align_reconstructio… by @TamirCohen in https://github.com/colmap/colmap/pull/2776
* Fix missing camera ptr for Reconstruction.DeleteAllPoints2DAndPoints3D() by @B1ueber2y in https://github.com/colmap/colmap/pull/2779
* Rename remaining proj_matrix instances to cam_from_world by @ahojnnes in https://github.com/colmap/colmap/pull/2780
* Relative pose decomposition uses Rigid3d by @ahojnnes in https://github.com/colmap/colmap/pull/2781
* Minor renaming on pycolmap point2d and point3d filenames by @B1ueber2y in https://github.com/colmap/colmap/pull/2784
* Add validity check for pixel coordinate in the Fisheye camera. Fix tests.  by @B1ueber2y in https://github.com/colmap/colmap/pull/2790
* Use branch prediction in PRNG functions by @ahojnnes in https://github.com/colmap/colmap/pull/2796
* Implementation of Aria Fisheye camera model by @nushakrishnan in https://github.com/colmap/colmap/pull/2786
* Upgrade to C++ 17 by @B1ueber2y in https://github.com/colmap/colmap/pull/2801
* Pose Prior based Incremental Mapper by @ferreram in https://github.com/colmap/colmap/pull/2660
* Expose UpdatePoint3DErrors to pycolmap by @theartful in https://github.com/colmap/colmap/pull/2805
* Switch to the Ruff Python formatter by @sarlinpe in https://github.com/colmap/colmap/pull/2803
* Add mixed Python-C++ PyCOLMAP package by @sarlinpe in https://github.com/colmap/colmap/pull/2747
* Enable Ruff linter for Python by @sarlinpe in https://github.com/colmap/colmap/pull/2806
* Use C++17 structured bindings in some places by @ahojnnes in https://github.com/colmap/colmap/pull/2808
* Add RAD_TAN_THIN_PRISM_FISHEYE to camera docs by @ahojnnes in https://github.com/colmap/colmap/pull/2810
* Customized cost functions should be functors instead by @B1ueber2y in https://github.com/colmap/colmap/pull/2811
* Install and use newer clang-format from pypi by @ahojnnes in https://github.com/colmap/colmap/pull/2812
* Return a reference in Reconstruction.image/camera/point3D by @sarlinpe in https://github.com/colmap/colmap/pull/2814
* Add test for PositionPriorErrorCostFunctor. by @ferreram in https://github.com/colmap/colmap/pull/2815
* Replace boost/filesystem with standard library by @ahojnnes in https://github.com/colmap/colmap/pull/2809
* Fix selection of BA solver type when there is no cuda by @ahojnnes in https://github.com/colmap/colmap/pull/2822
* More informative exception if invalid access of image/camera/point3D by @sarlinpe in https://github.com/colmap/colmap/pull/2825
* Use minimal solvers from poselib by @ahojnnes in https://github.com/colmap/colmap/pull/2288
* Disable -march=native flags in poselib by @ahojnnes in https://github.com/colmap/colmap/pull/2828
* Make ``Image::cam_from_world_`` optional by @sarlinpe in https://github.com/colmap/colmap/pull/2824
* Remove warning in configure step by @sarlinpe in https://github.com/colmap/colmap/pull/2830
* Fix coordinate notation in EstimateAbsolutePose by @ahojnnes in https://github.com/colmap/colmap/pull/2833
* Return success status in low-level triangulation functions by @ahojnnes in https://github.com/colmap/colmap/pull/2834
* Pin mypy version for tests by @ahojnnes in https://github.com/colmap/colmap/pull/2849
* Suppress CMP0167 warning for FindBoost under CMake 3.30 or newer by @ahojnnes in https://github.com/colmap/colmap/pull/2853
* Reconstruction reader/writer tests and scene class repr by @ahojnnes in https://github.com/colmap/colmap/pull/2842
* Select CUDA device when bundle adjustment uses GPU by @ahojnnes in https://github.com/colmap/colmap/pull/2846
* Fix copying behaviors of Reconstruction regarding camera pointers by @B1ueber2y in https://github.com/colmap/colmap/pull/2841
* Use the C++ string representation for Python dataclass objects by @sarlinpe in https://github.com/colmap/colmap/pull/2855
* Various improvements for pycolmap bindings by @ahojnnes in https://github.com/colmap/colmap/pull/2854
* Use analytical Jacobian in IterativeUndistortion. Add trust region by @B1ueber2y in https://github.com/colmap/colmap/pull/2857
* Improve the conditioning of covariance estimation by @B1ueber2y in https://github.com/colmap/colmap/pull/2860
* Avoid unnecessary copy of RANSAC inlier masks by @ahojnnes in https://github.com/colmap/colmap/pull/2863
* Various improvements for cost functors by @ahojnnes in https://github.com/colmap/colmap/pull/2867
* Rename ``*_mapper`` to ``*_pipeline`` files by @ahojnnes in https://github.com/colmap/colmap/pull/2870
* Update the manylinux CI to GCC 10 by @sarlinpe in https://github.com/colmap/colmap/pull/2873
* Fix rare deadlock during matching due to concurrent database access by @ahojnnes in https://github.com/colmap/colmap/pull/2876
* Add new and missing options to automatic reconstructor by @ahojnnes in https://github.com/colmap/colmap/pull/2877
* Shared auto diff cost function creation by @ahojnnes in https://github.com/colmap/colmap/pull/2878
* Enable model alignment to reference model by @ahojnnes in https://github.com/colmap/colmap/pull/2879
* Add covariance weighted cost functor by @ahojnnes in https://github.com/colmap/colmap/pull/2880
* Fix unused variable warnings under MSVC by @ahojnnes in https://github.com/colmap/colmap/pull/2884
* Skip all but latest Python version in PR builds by @ahojnnes in https://github.com/colmap/colmap/pull/2881
* [doc] Fix path to example in README.md by @kielnino in https://github.com/colmap/colmap/pull/2886
* Update Github actions versions by @ahojnnes in https://github.com/colmap/colmap/pull/2887
* [doc] Fix typo for gui menu item by @kielnino in https://github.com/colmap/colmap/pull/2885
* Fix input type for automatic stereo fusion on extreme quality setting by @ahojnnes in https://github.com/colmap/colmap/pull/2893
* Make target with all sources optional by @HernandoR in https://github.com/colmap/colmap/pull/2889
* Gracefully handle missing image pose in viewer by @ahojnnes in https://github.com/colmap/colmap/pull/2894
* Update to latest vcpkg release 2024.10.21 by @ahojnnes in https://github.com/colmap/colmap/pull/2908
* Fix conversion from CUDA texture references to objects in SIFT feature extraction by @ahojnnes in https://github.com/colmap/colmap/pull/2911
* Modernized bundle adjustment interface by @ahojnnes in https://github.com/colmap/colmap/pull/2896
* Add missing unit tests for reconstruction alignment functions by @ahojnnes in https://github.com/colmap/colmap/pull/2913
* Do not test EstimateManhattanWorldFrame if LSD is disabled by @sarlinpe in https://github.com/colmap/colmap/pull/2920
* Custom macro for enum to string support by @B1ueber2y in https://github.com/colmap/colmap/pull/2918
* Bind the estimation of Sim3d by @sarlinpe in https://github.com/colmap/colmap/pull/2903
* Initialize glog in custom gmock main function by @ahojnnes in https://github.com/colmap/colmap/pull/2916
* Update ccache for faster windows CI builds by @ahojnnes in https://github.com/colmap/colmap/pull/2922
* Fixes for Windows ARM64 support by @ahojnnes in https://github.com/colmap/colmap/pull/2921
* Move geometry implementation of ``__repr__``, ``__eq__`` overloads to C++ side by @ahojnnes in https://github.com/colmap/colmap/pull/2915
* Consistent interface and various improvements for pycolmap/estimators by @ahojnnes in https://github.com/colmap/colmap/pull/2923
* Exclude DetectLineSegments if LSD is disabled by @sarlinpe in https://github.com/colmap/colmap/pull/2927
* Enable reading 16bit/channel (png) images to grayscale by @Ediolot in https://github.com/colmap/colmap/pull/2924
* Cleanup of remaining pycolmap interfaces by @ahojnnes in https://github.com/colmap/colmap/pull/2925
* Fix affine SIFT feature orientation detection by @ahojnnes in https://github.com/colmap/colmap/pull/2929
* Improvements to deprecated pycolmap members by @sarlinpe in https://github.com/colmap/colmap/pull/2932
* Fix pkgconf installation in Mac CI by @ahojnnes in https://github.com/colmap/colmap/pull/2936
* Make sphinx show the pycolmap constructors by @sarlinpe in https://github.com/colmap/colmap/pull/2935
* Bind synthetic dataset functionality in pycolmap by @ahojnnes in https://github.com/colmap/colmap/pull/2938
* Cleaner import of C++ symbols by @sarlinpe in https://github.com/colmap/colmap/pull/2933
* Fix pycolmap breakage for Python 3.8 by @sarlinpe in https://github.com/colmap/colmap/pull/2941
* Remove legacy boost test macro by @ahojnnes in https://github.com/colmap/colmap/pull/2940
* Drop support for VS 2019 CI checks by @ahojnnes in https://github.com/colmap/colmap/pull/2943
* Fix CI cache thrashing by inconsistent vcpkg binary caching by @ahojnnes in https://github.com/colmap/colmap/pull/2942
* Introduce gmock Eigen matrix matchers by @ahojnnes in https://github.com/colmap/colmap/pull/2939
* Prevent double initialization of glog for <=0.5 by @sarlinpe in https://github.com/colmap/colmap/pull/2945
* Fixes and refactoring for bundle adjustment covariance estimation by @ahojnnes in https://github.com/colmap/colmap/pull/2788
* Fix duplicate library warnings in linking stage by @ahojnnes in https://github.com/colmap/colmap/pull/2871
* Add test for Python mapping pipeline by @ahojnnes in https://github.com/colmap/colmap/pull/2946
* Add helper script for incremental pycolmap build by @ahojnnes in https://github.com/colmap/colmap/pull/2947
* Fix and consistently define Qt window flags by @ahojnnes in https://github.com/colmap/colmap/pull/2949
* Cross platform usage of monospace font by @ahojnnes in https://github.com/colmap/colmap/pull/2950
* Update to latest pybind11 version by @ahojnnes in https://github.com/colmap/colmap/pull/2952
* Update install instructions for Mac using homebrew by @ahojnnes in https://github.com/colmap/colmap/pull/2953

------------------------
COLMAP 3.10 (07/23/2024)
------------------------
* Add missing "include <memory>" needed for unique_ptr by @Tobias-Fischer in https://github.com/colmap/colmap/pull/2338
* Support decoding multi-byte characters in Python script by @jot-jt in https://github.com/colmap/colmap/pull/2344
* Split Dockerfile in two stages: builder and runtime. by @pablospe in https://github.com/colmap/colmap/pull/2347
* Dockerfile improvements by @pablospe in https://github.com/colmap/colmap/pull/2356
* Update VCPKG commit in Windows CI by @sarlinpe in https://github.com/colmap/colmap/pull/2365
* Simplify the creation of reprojection error cost functions by @sarlinpe in https://github.com/colmap/colmap/pull/2364
* Migrate pycolmap by @sarlinpe in https://github.com/colmap/colmap/pull/2367
* Rename master -> main in pycolmap CI by @sarlinpe in https://github.com/colmap/colmap/pull/2370
* Bind SetPRNGSeed by @sarlinpe in https://github.com/colmap/colmap/pull/2369
* Encapsulate freeimage usage from pycolmap in colmap bitmap by @ahojnnes in https://github.com/colmap/colmap/pull/2372
* Re-generate version info on git changes by @ahojnnes in https://github.com/colmap/colmap/pull/2373
* Consolidate colmap/pycolmap readmes, updated acknowledgements, etc. by @ahojnnes in https://github.com/colmap/colmap/pull/2374
* Fix crashing pycolmap CI on Windows by @sarlinpe in https://github.com/colmap/colmap/pull/2383
* Add costs for pose graph optimization by @sarlinpe in https://github.com/colmap/colmap/pull/2378
* Switch to exception checks - v2 by @sarlinpe in https://github.com/colmap/colmap/pull/2376
* Cleanup checks in pycolmap by @sarlinpe in https://github.com/colmap/colmap/pull/2388
* Add RigReprojErrorConstantRigCostFunction by @sarlinpe in https://github.com/colmap/colmap/pull/2377
* Add cost functions to pycolmap by @sarlinpe in https://github.com/colmap/colmap/pull/2393
* Fix warning C4722 by @whuaegeanse in https://github.com/colmap/colmap/pull/2391
* Move reconstruction IO utils to a new file by @sarlinpe in https://github.com/colmap/colmap/pull/2399
* Acquire the GIL before returning None by @sarlinpe in https://github.com/colmap/colmap/pull/2400
* Disentangle the controller from threading and integrate the new logic into IncrementalMapperController by @B1ueber2y in https://github.com/colmap/colmap/pull/2392
* Simplify the low-level triangulation API by @sarlinpe in https://github.com/colmap/colmap/pull/2402
* Initialize glog in pycolmap only if not already done by @sarlinpe in https://github.com/colmap/colmap/pull/2405
* Adapt all the controllers to inherit from BaseController rather than Thread (except for feature extraction and matching) by @B1ueber2y in https://github.com/colmap/colmap/pull/2406
* Update path to models.h in database docs by @diffner in https://github.com/colmap/colmap/pull/2412
* Migrate Ubuntu CI pipelines from ADO to Github by @ahojnnes in https://github.com/colmap/colmap/pull/2411
* Build wheels for Python 3.12 by @sarlinpe in https://github.com/colmap/colmap/pull/2416
* Migrate MacOS CI pipeline from ADO to Github by @ahojnnes in https://github.com/colmap/colmap/pull/2418
* Improve bindings of Database by @sarlinpe in https://github.com/colmap/colmap/pull/2413
* Migrate Windows CI pipeline from ADO to Github by @ahojnnes in https://github.com/colmap/colmap/pull/2419
* Reduce logging during incremental mapping by @sarlinpe in https://github.com/colmap/colmap/pull/2420
* Migrate Docker CI from ADO to Github, remove ADO pipelines by @ahojnnes in https://github.com/colmap/colmap/pull/2422
* Simplify IncrementalMapperController by @sarlinpe in https://github.com/colmap/colmap/pull/2421
* Fix for glog 0.7.0 by @sarlinpe in https://github.com/colmap/colmap/pull/2428
* Fix typo by @whuaegeanse in https://github.com/colmap/colmap/pull/2430
* Fix RunMapper by @whuaegeanse in https://github.com/colmap/colmap/pull/2431
* Do triangulation in the IncrementalMapperController by @sarlinpe in https://github.com/colmap/colmap/pull/2429
* Only push a new Docker image on release by @sarlinpe in https://github.com/colmap/colmap/pull/2436
* model aligner with type "custom" does not update reconstruction by @lpanaf in https://github.com/colmap/colmap/pull/2433
* Define vcpkg manifest by @ahojnnes in https://github.com/colmap/colmap/pull/2426
* Fix ordering of keyword arguments in pycolmap.rig_absolute_pose_estimation by @sarlinpe in https://github.com/colmap/colmap/pull/2440
* Reduce the build time of pycolmap by @sarlinpe in https://github.com/colmap/colmap/pull/2443
* Improve bindings of CorrespondenceGraph by @sarlinpe in https://github.com/colmap/colmap/pull/2476
* Bind Reconstruction::{SetUp,ImagePairStats} by @sarlinpe in https://github.com/colmap/colmap/pull/2477
* Add bindings for substeps of incremental mapper with a python example by @B1ueber2y in https://github.com/colmap/colmap/pull/2478
* Debug crashing VCPKG-based CI builds by @sarlinpe in https://github.com/colmap/colmap/pull/2508
* Upgrade to pybind11 v2.12. Fix bind_map and reconstruction.points3D by @B1ueber2y in https://github.com/colmap/colmap/pull/2502
* Minor fix on logging for the pycolmap customized runner by @B1ueber2y in https://github.com/colmap/colmap/pull/2503
* Fix missing public link deps, break circular feature-scene dependency by @ahojnnes in https://github.com/colmap/colmap/pull/2497
* Avoid duplicate image allocation during undistortion by @fseegraeber in https://github.com/colmap/colmap/pull/2520
* Fix reconstruction.points3D by @B1ueber2y in https://github.com/colmap/colmap/pull/2523
* Fix 'std::out_of_range' error when using hierarchical_mapper by @GrayMask in https://github.com/colmap/colmap/pull/2526
* Fix binding for std::vector<Point2D> by @sarlinpe in https://github.com/colmap/colmap/pull/2533
* Include pybind eigen header by @tmnku in https://github.com/colmap/colmap/pull/2510
* Fix pycolmap python pipeline for multiple models by @B1ueber2y in https://github.com/colmap/colmap/pull/2531
* make two view geometry writable by @tmnku in https://github.com/colmap/colmap/pull/2540
* Customized python interface for bundle adjustment by @B1ueber2y in https://github.com/colmap/colmap/pull/2509
* Fix typos by @MaximSmolskiy in https://github.com/colmap/colmap/pull/2553
* Implicitly convert iterator to ListPoint2D by @sarlinpe in https://github.com/colmap/colmap/pull/2558
* Fix model_cropper not resetting image.num_points3D of cropped_rec by @ArneSchulzTUBS in https://github.com/colmap/colmap/pull/2557
* Split pair generation and matching by @sarlinpe in https://github.com/colmap/colmap/pull/2573
* Add ObservationManager by @sarlinpe in https://github.com/colmap/colmap/pull/2575
* Log info about created feature extractor/matcher types by @ahojnnes in https://github.com/colmap/colmap/pull/2579
* LSD: making the AGPL dependency optional by @zap150 in https://github.com/colmap/colmap/pull/2578
* Disable LSD when building pycolmap wheels by @sarlinpe in https://github.com/colmap/colmap/pull/2580
* Synthesize full two-view geometry and raw matches by @ahojnnes in https://github.com/colmap/colmap/pull/2595
* Support Adjoint matrix computation for Rigid3d by @B1ueber2y in https://github.com/colmap/colmap/pull/2598
* Fix cost functions for pose graph optimization by @B1ueber2y in https://github.com/colmap/colmap/pull/2601
* Fix python bundle adjustment example with pyceres by @B1ueber2y in https://github.com/colmap/colmap/pull/2606
* Faster homography estimator by @ahojnnes in https://github.com/colmap/colmap/pull/2603
* Add function to find real cubic polynomial roots by @ahojnnes in https://github.com/colmap/colmap/pull/2609
* Align with the convention of ceres doc on SqrtInformation. by @B1ueber2y in https://github.com/colmap/colmap/pull/2611
* Faster 7-point fundamental matrix estimator by @ahojnnes in https://github.com/colmap/colmap/pull/2612
* Faster 8-point fundamental matrix estimator by @ahojnnes in https://github.com/colmap/colmap/pull/2613
* Covariance estimation for bundle adjustment with Schur elimination by @B1ueber2y in https://github.com/colmap/colmap/pull/2610
* Mac OS improvements by @BSVogler in https://github.com/colmap/colmap/pull/2622
* Update cibuildwheel to 2.19.2 by @ahojnnes in https://github.com/colmap/colmap/pull/2632
* Faster essential matrix estimators by @ahojnnes in https://github.com/colmap/colmap/pull/2618
* Remove CamFromWorldPrior and create LocationPrior by @sarlinpe in https://github.com/colmap/colmap/pull/2620
* Add option to disable uninstall target, restore CI pipeline by @ahojnnes in https://github.com/colmap/colmap/pull/2634
* Faster covariance computation for small blocks by @B1ueber2y in https://github.com/colmap/colmap/pull/2633
* Fix optimal point algorithm by @morrishelle in https://github.com/colmap/colmap/pull/2640
* Add shell script helper for profiling by @ahojnnes in https://github.com/colmap/colmap/pull/2635
* Declare PosePrior::IsValid as const by @ahojnnes in https://github.com/colmap/colmap/pull/2653
* Add CI build for Windows CUDA by @ahojnnes in https://github.com/colmap/colmap/pull/2651
* Publish windows binaries from CI by @ahojnnes in https://github.com/colmap/colmap/pull/2663

-------------------------
COLMAP 3.9.1 (01/08/2024)
-------------------------
* Version 3.9 changelog by @ahojnnes in https://github.com/colmap/colmap/pull/2325
* Fully encapsulate freeimage in bitmap library (#2332) by @ahojnnes in https://github.com/colmap/colmap/pull/2334

-----------------------
COLMAP 3.9 (01/06/2024)
-----------------------
* clang format all code and require clang-format-14 by @ahojnnes in https://github.com/colmap/colmap/pull/1785
* Fix compilation for vcpkg windows build by @ahojnnes in https://github.com/colmap/colmap/pull/1791
* Increment version number to 3.9 by @ahojnnes in https://github.com/colmap/colmap/pull/1794
* Remove unnecessary /arch:sse2 flag for MSVC by @ahojnnes in https://github.com/colmap/colmap/pull/1798
* Updated faq.rst by @CGCooke in https://github.com/colmap/colmap/pull/1801
* Fixed mistake in code comment for OpenCV Fisheye camera by @CGCooke in https://github.com/colmap/colmap/pull/1802
* Replace deprecated cudaThreadSynchronize with cudaDeviceSynchronize by @ahojnnes in https://github.com/colmap/colmap/pull/1806
* Replace deprecated Cuda texture references with texture objects by @ahojnnes in https://github.com/colmap/colmap/pull/1809
* Remove unused SIFT GPU cuda texture reference by @ahojnnes in https://github.com/colmap/colmap/pull/1823
* Upgrade SiftGPU to use CUDA texture objects by @ahojnnes in https://github.com/colmap/colmap/pull/1838
* Remove PBA as bundle adjustment backend to support CUDA 12+ by @ahojnnes in https://github.com/colmap/colmap/pull/1840
* Replace deprecated CUDA sature function call by @ahojnnes in https://github.com/colmap/colmap/pull/1841
* Avoid unnecessary mallocs during sampling by @ahojnnes in https://github.com/colmap/colmap/pull/1842
* Cleaned up docker readme and scripts by @ahojnnes in https://github.com/colmap/colmap/pull/1852
* add "Shared intrinsics per sub-folder" checkbox to automatic reconstruction window by @kenshi84 in https://github.com/colmap/colmap/pull/1853
* Update vcpkg by @ahojnnes in https://github.com/colmap/colmap/pull/1925
* Log the name of the file that causes Mat::Read() to checkfail by @SomeAlphabetGuy in https://github.com/colmap/colmap/pull/1923
* check Z_index correctly in ReadPly by @countywest in https://github.com/colmap/colmap/pull/1896
* Don't re-open files when reading and writing matrices by @SomeAlphabetGuy in https://github.com/colmap/colmap/pull/1926
* Update vcpkg to latest commit by @ahojnnes in https://github.com/colmap/colmap/pull/1948
* Remove unnecessary custom Eigen aligned allocator macros by @ahojnnes in https://github.com/colmap/colmap/pull/1947
* Prefix internal sources/includes with colmap by @ahojnnes in https://github.com/colmap/colmap/pull/1949
* Simplify clang-format config and sort includes by @ahojnnes in https://github.com/colmap/colmap/pull/1950
* Handle possible overflow in median function by @ahojnnes in https://github.com/colmap/colmap/pull/1951
* Run ASan pipeline under Ubuntu 22.04 by @ahojnnes in https://github.com/colmap/colmap/pull/1952
* Fix Ceres version test by @drkoller in https://github.com/colmap/colmap/pull/1954
* Fix deprecation warning for Qt font metrics width by @ahojnnes in https://github.com/colmap/colmap/pull/1958
* Setup clang-tidy and enable perf warnings by @ahojnnes in https://github.com/colmap/colmap/pull/1959
* VCPKG binary caching for windows CI by @ahojnnes in https://github.com/colmap/colmap/pull/1957
* Cosmetics for VS dev shell script by @ahojnnes in https://github.com/colmap/colmap/pull/1965
* Enable clang-tidy concurrency checks by @ahojnnes in https://github.com/colmap/colmap/pull/1967
* [Bug] fix finding shared points3D in FindLocalBundle by @wesleyliwei in https://github.com/colmap/colmap/pull/1963
* Enable compiler caching in CI by @ahojnnes in https://github.com/colmap/colmap/pull/1972
* Set number of features for different quality levels by @ahojnnes in https://github.com/colmap/colmap/pull/1975
* Specify parameter name using inline comment by @ahojnnes in https://github.com/colmap/colmap/pull/1976
* Fix Windows CCache by @ahojnnes in https://github.com/colmap/colmap/pull/1977
* Add e2e tests in CI pipeline using ETH3D datasets by @ahojnnes in https://github.com/colmap/colmap/pull/1397
* [feature] print verbose information for model analyzer by @wesleyliwei in https://github.com/colmap/colmap/pull/1978
* Add a missing include to compile with gcc13 by @EstebanDugueperoux2 in https://github.com/colmap/colmap/pull/1984
* Speed up snapshot construct in RigBundleAdjuster by @wesleyliwei in https://github.com/colmap/colmap/pull/1988
* Update outdated docker cuda image tag by @ahojnnes in https://github.com/colmap/colmap/pull/1992
* Add boulders ETH3D dataset to CI E2E tests by @ahojnnes in https://github.com/colmap/colmap/pull/1991
* Update executable paths in documentation by @ahojnnes in https://github.com/colmap/colmap/pull/1993
* Avoid unnecessary copy in ExtractTopScaleFeatures by @ahojnnes in https://github.com/colmap/colmap/pull/1994
* Move related code under new image library folder by @ahojnnes in https://github.com/colmap/colmap/pull/1995
* Move related code under new camera folder by @ahojnnes in https://github.com/colmap/colmap/pull/1996
* Added a virtual destructor to Sampler by @SomeAlphabetGuy in https://github.com/colmap/colmap/pull/2000
* Add a few more clang-tidy checks by @ahojnnes in https://github.com/colmap/colmap/pull/2001
* Move related code to new geometry module by @ahojnnes in https://github.com/colmap/colmap/pull/2006
* Use #pragma once as include guard by @ahojnnes in https://github.com/colmap/colmap/pull/2007
* Add bugprone-* clang-tidy checks by @ahojnnes in https://github.com/colmap/colmap/pull/2010
* Avoid const params in declarations by @ahojnnes in https://github.com/colmap/colmap/pull/2011
* Set and require C++14 by @ahojnnes in https://github.com/colmap/colmap/pull/2012
* Cleanup math functions that are now part of eigen/stdlib by @ahojnnes in https://github.com/colmap/colmap/pull/2013
* Add clang-analyzer checks by @ahojnnes in https://github.com/colmap/colmap/pull/2014
* Replace CMake provided find_package scripts and modern CMake targets by @ahojnnes in https://github.com/colmap/colmap/pull/2016
* Switch from Boost unit tests to Gtest by @ahojnnes in https://github.com/colmap/colmap/pull/2017
* Fix ccache restore keys in pipeline caching by @ahojnnes in https://github.com/colmap/colmap/pull/2018
* Add missing cacheHitVar to fix ccache by @ahojnnes in https://github.com/colmap/colmap/pull/2020
* Add missing Boost::graph import by @sarlinpe in https://github.com/colmap/colmap/pull/2021
* Compressed/flattened correspondence graph for faster triangulation / less memory by @ahojnnes in https://github.com/colmap/colmap/pull/2019
* Fix window ccache key by @ahojnnes in https://github.com/colmap/colmap/pull/2024
* Consistently use shared_ptr for shared pointers for SFM objects by @ahojnnes in https://github.com/colmap/colmap/pull/2023
* Remove check on Qt version by @sarlinpe in https://github.com/colmap/colmap/pull/2022
* Synthetics for E2E incremental mapper tests by @ahojnnes in https://github.com/colmap/colmap/pull/2025
* New math module by @ahojnnes in https://github.com/colmap/colmap/pull/2028
* Simplify similarity transform and more tests by @ahojnnes in https://github.com/colmap/colmap/pull/2030
* Extract reconstruction alignment functions into new file by @ahojnnes in https://github.com/colmap/colmap/pull/2032
* Add E2E hierarchical mapper tests by @ahojnnes in https://github.com/colmap/colmap/pull/2033
* Rename SimilarityTransform3 to Sim3d by @ahojnnes in https://github.com/colmap/colmap/pull/2034
* Add Rigid3d transform class by @ahojnnes in https://github.com/colmap/colmap/pull/2035
* Consolidate and simplify Rigid3d and Sim3d by @ahojnnes in https://github.com/colmap/colmap/pull/2037
* Some small improvements/cleanup for rigid3d/sim3d usage by @ahojnnes in https://github.com/colmap/colmap/pull/2041
* CamFromWorld replaces qvec/tvec by @ahojnnes in https://github.com/colmap/colmap/pull/2039
* Retry download of ETH3D datasets by @ahojnnes in https://github.com/colmap/colmap/pull/2043
* WorldToImage becomes CamToImg by @ahojnnes in https://github.com/colmap/colmap/pull/2044
* Camera models operate on camera rays by @ahojnnes in https://github.com/colmap/colmap/pull/2045
* Ignore directory .vs by @whuaegeanse in https://github.com/colmap/colmap/pull/2046
* Use the reference of Rigid3d to reduce memory consumption by @whuaegeanse in https://github.com/colmap/colmap/pull/2047
* Inline point to image projection by @ahojnnes in https://github.com/colmap/colmap/pull/2050
* Point2D becomes simpler pure data struct by @ahojnnes in https://github.com/colmap/colmap/pull/2051
* Use Eigen math for estimator utils by @ahojnnes in https://github.com/colmap/colmap/pull/2052
* Move cost functions under geometry module and rename by @ahojnnes in https://github.com/colmap/colmap/pull/2053
* Bundle adjuster is an estimator by @ahojnnes in https://github.com/colmap/colmap/pull/2054
* Remaining base targets move to new scene module by @ahojnnes in https://github.com/colmap/colmap/pull/2055
* Vote and verify improvements/speedup by @ahojnnes in https://github.com/colmap/colmap/pull/2056
* Generate version info in .cc file to reduce number of recompilations by @ahojnnes in https://github.com/colmap/colmap/pull/2057
* Option manager moves to controllers to disentangle circular deps by @ahojnnes in https://github.com/colmap/colmap/pull/2058
* Granular CMake modules and build targets by @ahojnnes in https://github.com/colmap/colmap/pull/2059
* Fix docker build by @ahojnnes in https://github.com/colmap/colmap/pull/2069
* Remove warnings about duplicated marco NOMINMAX by @whuaegeanse in https://github.com/colmap/colmap/pull/2067
* lib folder becomes thirdparty folder by @ahojnnes in https://github.com/colmap/colmap/pull/2068
* Remove unnecessary checks in image pair conversion by @ahojnnes in https://github.com/colmap/colmap/pull/2074
* Replace flaky ETH3D terrace with courtyard dataset by @ahojnnes in https://github.com/colmap/colmap/pull/2075
* Synthesize chained match graph for more mapper tests by @ahojnnes in https://github.com/colmap/colmap/pull/2076
* Introduce abstract feature extractor by @ahojnnes in https://github.com/colmap/colmap/pull/2077
* Avoid unnecessary data copies in feature conversion utils by @ahojnnes in https://github.com/colmap/colmap/pull/2078
* Abstract feature matcher by @ahojnnes in https://github.com/colmap/colmap/pull/2082
* Encapsulate feature matching controller/worker implementations by @ahojnnes in https://github.com/colmap/colmap/pull/2085
* Some cosmetics for util/feature types by @ahojnnes in https://github.com/colmap/colmap/pull/2084
* Use std:: when cmath included by @whuaegeanse in https://github.com/colmap/colmap/pull/2081
* Encapsulate feature extraction controller/worker implementations by @ahojnnes in https://github.com/colmap/colmap/pull/2086
* Reenable VS2022 CI pipeline by @ahojnnes in https://github.com/colmap/colmap/pull/1689
* Consistent transform convention for CenterAndNormalizeImagePoints by @ahojnnes in https://github.com/colmap/colmap/pull/2092
* Retire Mac 11 CI build by @ahojnnes in https://github.com/colmap/colmap/pull/2094
* Add ReprojErrorConstantPoint3DCostFunction to speed up the RefineAbsolutePose function by @whuaegeanse in https://github.com/colmap/colmap/pull/2089
* Numeric differentiation of camera model using partial piv LU by @ahojnnes in https://github.com/colmap/colmap/pull/2100
* cmake: add testing.cc to colmap_util only if TESTS_ENABLED=ON by @NeroBurner in https://github.com/colmap/colmap/pull/2102
* Set CUDA_STANDARD to 14 by @ahojnnes in https://github.com/colmap/colmap/pull/2108
* Transform back to existing images positions after mapper processing if set fixed by @ferreram in https://github.com/colmap/colmap/pull/2095
* Update documentation with new branch policy by @ahojnnes in https://github.com/colmap/colmap/pull/2110
* Update CMake find dependencies for vcpkg by @ahojnnes in https://github.com/colmap/colmap/pull/2116
* Decouple SIFT match from two view geometry options by @ahojnnes in https://github.com/colmap/colmap/pull/2118
* Fix docker build by @vnmsklnk in https://github.com/colmap/colmap/pull/2122
* Trigger build pipeline on main branch by @ahojnnes in https://github.com/colmap/colmap/pull/2123
* Update Linux install documentation with new branch policy by @joshuaoreilly in https://github.com/colmap/colmap/pull/2126
* Fix link in camera model documentation by @CFretter in https://github.com/colmap/colmap/pull/2152
* [Bugfix] Fix GUI_ENABLED=OFF and skip SiftGPU if no GUI and no CUDA by @sarlinpe in https://github.com/colmap/colmap/pull/2151
* [Bugfix] Properly handle CGAL_ENABLED by @sarlinpe in https://github.com/colmap/colmap/pull/2149
* Refinement of intrinsics in the point_triangulator by @tsattler in https://github.com/colmap/colmap/pull/2144
* Bugfix in handling COLMAP_GPU_ENABLED by @sarlinpe in https://github.com/colmap/colmap/pull/2163
* Expose exe as libs by @sarlinpe in https://github.com/colmap/colmap/pull/2165
* Add Sim3d::FromMatrix by @sarlinpe in https://github.com/colmap/colmap/pull/2147
* Check code format in CI by @ahojnnes in https://github.com/colmap/colmap/pull/2171
* Clean up dependencies by @sarlinpe in https://github.com/colmap/colmap/pull/2173
* Move tests into anonymous namespaces by @ahojnnes in https://github.com/colmap/colmap/pull/2175
* Fix glew/qopengl conflict warning by @ahojnnes in https://github.com/colmap/colmap/pull/2176
* Update documentation with new link to GitHub discussions by @ahojnnes in https://github.com/colmap/colmap/pull/2177
* Restore GLEW include by @sarlinpe in https://github.com/colmap/colmap/pull/2178
* Align reconstructions via shared 3D points by @sarlinpe in https://github.com/colmap/colmap/pull/2169
* Add clang-tidy-cachein CI by @ahojnnes in https://github.com/colmap/colmap/pull/2182
* Disable GUI build in one CI config by @ahojnnes in https://github.com/colmap/colmap/pull/2181
* Show verbose ccache stats by @ahojnnes in https://github.com/colmap/colmap/pull/2183
* Add EstimateGeneralizedAbsolutePose by @sarlinpe in https://github.com/colmap/colmap/pull/2174
* Fix bug in ReconstructionManagerWidget::Update by @whuaegeanse in https://github.com/colmap/colmap/pull/2186
* Fix missing retrieval dependency by @ahojnnes in https://github.com/colmap/colmap/pull/2189
* Removing clustering_options and mapper_options in Hierarchical Mapper Controller by @Serenitysmk in https://github.com/colmap/colmap/pull/2193
* Publish docker image to docker hub by @ahojnnes in https://github.com/colmap/colmap/pull/2195
* Fix Cuda architecture in docker build by @ahojnnes in https://github.com/colmap/colmap/pull/2196
* Fix all-major cuda arch missing in CMake < 3.23 by @ahojnnes in https://github.com/colmap/colmap/pull/2197
* Update triangulation.cc by @RayShark0605 in https://github.com/colmap/colmap/pull/2205
* Update author and acknowledgements by @ahojnnes in https://github.com/colmap/colmap/pull/2207
* Code formatting for Python by @ahojnnes in https://github.com/colmap/colmap/pull/2208
* Retire outdated build script by @ahojnnes in https://github.com/colmap/colmap/pull/2217
* Remove mention of deprecated build script by @sarlinpe in https://github.com/colmap/colmap/pull/2220
* Improve word spelling by @zchrissirhcz in https://github.com/colmap/colmap/pull/2235
* Stack allocate camera param idx arrays by @ahojnnes in https://github.com/colmap/colmap/pull/2234
* fix: typo in colmap/src/colmap/ui/project_widget.cc by @varundhand in https://github.com/colmap/colmap/pull/2241
* Update reconstruction.cc by @RayShark0605 in https://github.com/colmap/colmap/pull/2238
* Update to Docker CUDA 12.2.2 by @ahojnnes in https://github.com/colmap/colmap/pull/2244
* Stop setting C++ standard flags manually by @AdrianBunk in https://github.com/colmap/colmap/pull/2251
* Setting clear_points to true per default in point_triangulator by @tsattler in https://github.com/colmap/colmap/pull/2252
* Update cameras.rst to fix link to code by @tsattler in https://github.com/colmap/colmap/pull/2246
* Fix matching of imported features without descriptors by @ahojnnes in https://github.com/colmap/colmap/pull/2269
* Consistent versioning between documentation and code by @ahojnnes in https://github.com/colmap/colmap/pull/2275
* Reduce mallocs for RANSAC estimator models by @ahojnnes in https://github.com/colmap/colmap/pull/2283
* Migrate to glog logging by @ahojnnes in https://github.com/colmap/colmap/pull/2172
* Turn Point3D into simple data struct by @ahojnnes in https://github.com/colmap/colmap/pull/2285
* Camera becomes simple data struct by @ahojnnes in https://github.com/colmap/colmap/pull/2286
* Recover custom Eigen std::vector allocator for Eigen <3.4 support by @ahojnnes in https://github.com/colmap/colmap/pull/2293
* Replace result_of with invoke_result_t by @sarlinpe in https://github.com/colmap/colmap/pull/2300
* Allow getters FocalLength{X,Y} for isotropic models by @sarlinpe in https://github.com/colmap/colmap/pull/2301
* Add missing Boost targets and cleanup includes by @sarlinpe in https://github.com/colmap/colmap/pull/2304
* Expose IncrementalMapperOptions::{mapper,triangulation} by @sarlinpe in https://github.com/colmap/colmap/pull/2308
* Update install instructions for Mac by @Dawars in https://github.com/colmap/colmap/pull/2310
* Remove unused ceres reference in doc by @ahojnnes in https://github.com/colmap/colmap/pull/2315
* Fix typo by @whuaegeanse in https://github.com/colmap/colmap/pull/2317
* Stable version 3.9 release by @ahojnnes in https://github.com/colmap/colmap/pull/2319

-----------------------
COLMAP 3.8 (01/31/2023)
-----------------------
* Updating geo-registration doc. by @ferreram in https://github.com/colmap/colmap/pull/1410
* Adding user-specified option for reconstructing purely planar scene. … by @ferreram in https://github.com/colmap/colmap/pull/1408
* Only apply sqlite vacuum command when elements are deleted from the database. by @ferreram in https://github.com/colmap/colmap/pull/1414
* Replace Graclus with Metis dependency by @ahojnnes in https://github.com/colmap/colmap/pull/1422
* Update ceres download URL in build script by @whuaegeanse in https://github.com/colmap/colmap/pull/1430
* Fix type errors when building colmap with build.py in windows by @whuaegeanse in https://github.com/colmap/colmap/pull/1440
* Fix bug in the computation of the statistics Global/Local BA by @whuaegeanse in https://github.com/colmap/colmap/pull/1449
* Add RefineGeneralizedAbsolutePose and covariance estimation by @Skydes in https://github.com/colmap/colmap/pull/1464
* Update docker image definition by @ahojnnes in https://github.com/colmap/colmap/pull/1478
* Upgrade deprecated ceres parameterizations to manifolds by @ahojnnes in https://github.com/colmap/colmap/pull/1477
* Use masks for stereo fusion on automatic reconstruction by @ibrarmalik in https://github.com/colmap/colmap/pull/1488
* fix random seed set failed from external interface by @WZG3661 in https://github.com/colmap/colmap/pull/1498
* Replace deprecated Eigen nonZeros() call for most recent Eigen versions. by @nackjaylor in https://github.com/colmap/colmap/pull/1494
* Fix ceres-solver folder name by @f-fl0 in https://github.com/colmap/colmap/pull/1501
* Improved convergence criterion for XYZ to ELL conversion by @ahojnnes in https://github.com/colmap/colmap/pull/1505
* Fix bug in the function SetPtr of Bitmap by @whuaegeanse in https://github.com/colmap/colmap/pull/1525
* Avoid the calling of copy constructor/assignment by @whuaegeanse in https://github.com/colmap/colmap/pull/1524
* Avoid calling copy constructors of  FeatureKeypoints and FeatureDescriptors by @whuaegeanse in https://github.com/colmap/colmap/pull/1540
* Initialize freeimage if statically linked by @ahojnnes in https://github.com/colmap/colmap/pull/1549
* Avoid hard crash if Jacobian matrix is rank deficient by @mihaidusmanu in https://github.com/colmap/colmap/pull/1557
* visualize_model.py: added FULL_OPENCV model by @soeroesg in https://github.com/colmap/colmap/pull/1552
* Update vcpkg version to fix CI pipeline by @ahojnnes in https://github.com/colmap/colmap/pull/1568
* Replace deprecated Mac OS 10.15 with Mac OS 12 build in CI by @ahojnnes in https://github.com/colmap/colmap/pull/1569
* Fix inconsistent between the actual executed image reader option and the saved project.ini file by @XuChengHUST in https://github.com/colmap/colmap/pull/1564
* checkout the expected version of ceres solver by @scott-vsi in https://github.com/colmap/colmap/pull/1576
* use default qt5 brew install directory #1573 by @catapulta in https://github.com/colmap/colmap/pull/1574
* Fix image undistortion with nested image folders by @ahojnnes in https://github.com/colmap/colmap/pull/1606
* Fix source file permissions by @ahojnnes in https://github.com/colmap/colmap/pull/1607
* Fixed the collection of arguments in colmap.bat by @tdegraaff in https://github.com/colmap/colmap/pull/1121
* Add OpenMP to COLMAP_EXTERNAL_LIBRARIES if enabled by @logchan in https://github.com/colmap/colmap/pull/1632
* Fix output tile reconstructions are the same as the input reconstruction in `RunModelSplitter` (#1513) by @Serenitysmk in https://github.com/colmap/colmap/pull/1531
* add `libmetis-dev` to solve `METIS_INCLUDE_DIRS`. by @FavorMylikes in https://github.com/colmap/colmap/pull/1672
* Update install.rst by @tomer-grin in https://github.com/colmap/colmap/pull/1671
* Update freeimage links. by @Yulv-git in https://github.com/colmap/colmap/pull/1675
* fix small typo by @skal65535 in https://github.com/colmap/colmap/pull/1668
* Update build.py with new glew link by @aghand0ur in https://github.com/colmap/colmap/pull/1658
* Add use_cache in fusion options GUI by @hrflr in https://github.com/colmap/colmap/pull/1655
* Add CI pipeline for Ubuntu 22.04 by @ahojnnes in https://github.com/colmap/colmap/pull/1688
* Avoid unnecessary copies of data by @ahojnnes in https://github.com/colmap/colmap/pull/1691
* Reduce memory allocations in correspondence graph search by @ahojnnes in https://github.com/colmap/colmap/pull/1692
* Use FindCUDAToolkit when available. by @hanseuljun in https://github.com/colmap/colmap/pull/1693
* Fixed a crash due to inconsistent undistortion by @SomeAlphabetGuy in https://github.com/colmap/colmap/pull/1698
* Add CUDA Ubuntu 22.04 CI build by @ahojnnes in https://github.com/colmap/colmap/pull/1705
* Delete the redundancy install of libmetis-dev by @thomas-graphopti in https://github.com/colmap/colmap/pull/1721
* Fix broken loading of image masks on macOS by @buesma in https://github.com/colmap/colmap/pull/1639
* Update install instructions with latest hints and known issues by @ahojnnes in https://github.com/colmap/colmap/pull/1736
* Modernize smart pointer initialization, fix alloc/dealloc mismatch by @ahojnnes in https://github.com/colmap/colmap/pull/1737
* Fix typo in cli.rst by @ojhernandez in https://github.com/colmap/colmap/pull/1747
* Fix inconsistent image resizing between CPU/GPU implementations of SIFT by @Yzhbuaa in https://github.com/colmap/colmap/pull/1642
* Reduce number of SIFT test features to make tests run under WSL by @ahojnnes in https://github.com/colmap/colmap/pull/1748
* Tag documentation version with dev by @ahojnnes in https://github.com/colmap/colmap/pull/1749
* Update copyright to 2023 by @ahojnnes in https://github.com/colmap/colmap/pull/1750
* Fix max image dimension for positive first_octave by @ahojnnes in https://github.com/colmap/colmap/pull/1751
* Fix SIFT GPU match creation by @ahojnnes in https://github.com/colmap/colmap/pull/1757
* Fix SIFT tests for OpenGL by @ahojnnes in https://github.com/colmap/colmap/pull/1762
* Suppress CUDA stack size warning for ptxas by @ahojnnes in https://github.com/colmap/colmap/pull/1770
* Simplify CUDA CMake configuration by @ahojnnes in https://github.com/colmap/colmap/pull/1776
* Fixes for CUDA compilation by @ahojnnes in https://github.com/colmap/colmap/pull/1777
* Improvements to dockerfile and build pipeline by @ahojnnes in https://github.com/colmap/colmap/pull/1778
* Explicitly require CMAKE_CUDA_ARCHITECTURES to be defined by @ahojnnes in https://github.com/colmap/colmap/pull/1781
* Depend on system installed FLANN by @ahojnnes in https://github.com/colmap/colmap/pull/1782
* Option to store relative pose between two cameras in database by @yanxke in https://github.com/colmap/colmap/pull/1774
* Depend on system installed SQLite3 by @ahojnnes in https://github.com/colmap/colmap/pull/1783

-----------------------
COLMAP 3.7 (01/26/2022)
-----------------------
* Allow to save fused point cloud in colmap format when using command line by @boitumeloruf in https://github.com/colmap/colmap/pull/799
* Fix typos in image.h by @Pascal-So in https://github.com/colmap/colmap/pull/936
* Fix for EPnP estimator by @vlarsson in https://github.com/colmap/colmap/pull/943
* Visualize models using Python in Open3D by @ahojnnes in https://github.com/colmap/colmap/pull/948
* Update tutorial.rst by @ignacio-rocco in https://github.com/colmap/colmap/pull/953
* 8 point algorithm internal contraint fix by @mihaidusmanu in https://github.com/colmap/colmap/pull/982
* Python script for writing depth/normal arrays by @SBCV in https://github.com/colmap/colmap/pull/957
* BuildImageModel: use std::vector instead of numbered arguments by @Pascal-So in https://github.com/colmap/colmap/pull/949
* Fix bugs of sift feature matching by @whuaegeanse in https://github.com/colmap/colmap/pull/985
* script for modifying fused results by @SBCV in https://github.com/colmap/colmap/pull/984
* fix camera model query by @Pascal-So in https://github.com/colmap/colmap/pull/997
* fixed small bug in visualize_model.py by @sniklaus in https://github.com/colmap/colmap/pull/1007
* Update .travis.yml by @srinivas32 in https://github.com/colmap/colmap/pull/989
* Ensure DecomposeHomographyMatrix() always returns rotations by @daithimaco in https://github.com/colmap/colmap/pull/1040
* Remove deprecated qt foreach by @UncleGene in https://github.com/colmap/colmap/pull/1039
* Fix AMD/Windows GUI visualization bug by @drkoller in https://github.com/colmap/colmap/pull/1079
* include colmap_cuda in COLMAP_LIBRARIES when compiled with cuda by @ClementPinard in https://github.com/colmap/colmap/pull/1084
* Fix runtime crash when sparsesuite is missing from ceres by @anmatako in https://github.com/colmap/colmap/pull/1115
* Store relative poses in two_view_geometry table by @Ahmed-Salama in https://github.com/colmap/colmap/pull/1103
* search src images for patch_match from all set, not only referenced subset by @DaniilSNikulin in https://github.com/colmap/colmap/pull/1038
* Replace Travis CI with Azure Pipelines for Linux/Mac builds by @ahojnnes in https://github.com/colmap/colmap/pull/1119
* Allow ReadPly to handle double precision files by @anmatako in https://github.com/colmap/colmap/pull/1131
* Update GPSTransform calculations to improve accuracy by @anmatako in https://github.com/colmap/colmap/pull/1132
* Add scale template flag in SimilarityTransform3::Estimate by @anmatako in https://github.com/colmap/colmap/pull/1133
* Add CopyFile utility that can copy or hard/soft-link files by @anmatako in https://github.com/colmap/colmap/pull/1134
* Expose BA options in IncrementalMapper by @anmatako in https://github.com/colmap/colmap/pull/1139
* Allow configurable paths for mvs::Model by @anmatako in https://github.com/colmap/colmap/pull/1141
* Change ReconstructionMaanger to write larger recons first by @anmatako in https://github.com/colmap/colmap/pull/1137
* Setup Azure pipelines for Windows build by @ahojnnes in https://github.com/colmap/colmap/pull/1150
* Add fixed extrinsics in rig config by @anmatako in https://github.com/colmap/colmap/pull/1144
* Allow custom config and missing dependencies for patch-match by @anmatako in https://github.com/colmap/colmap/pull/1142
* Update print statements for Python 3 compatibility by @UncleGene in https://github.com/colmap/colmap/pull/1126
* Allow cleanup of SQLite tables using new database_cleaner command by @anmatako in https://github.com/colmap/colmap/pull/1136
* Extend SceneClustering to support non-hierarchical (flat) clusters by @anmatako in https://github.com/colmap/colmap/pull/1140
* Support more formats in model_converter by @anmatako in https://github.com/colmap/colmap/pull/1147
* Fix Mac 10.15 build due to changed Qt5 path by @ahojnnes in https://github.com/colmap/colmap/pull/1157
* Fix bug in ReadCameraRigConfig when reading extrinsics by @anmatako in https://github.com/colmap/colmap/pull/1158
* Add utility to compare poses between two sparse models by @ahojnnes in https://github.com/colmap/colmap/pull/1159
* Modularize executable main functions into separate sources by @ahojnnes in https://github.com/colmap/colmap/pull/1160
* Fix unnecessary copies in for range loops by @ahojnnes in https://github.com/colmap/colmap/pull/1162
* Add script to clang-format all source code by @ahojnnes in https://github.com/colmap/colmap/pull/1163
* Add back new options and formats for model_converter by @anmatako in https://github.com/colmap/colmap/pull/1164
* ImageReder new option and bug fix in GPS priors by @anmatako in https://github.com/colmap/colmap/pull/1146
* Parallelize stereo fusion; needs pre-loading of entire workspace by @anmatako in https://github.com/colmap/colmap/pull/1148
* Refactoring and new functionality in Reconstruction class by @anmatako in https://github.com/colmap/colmap/pull/1169
* Add new functionality in image_undistorter by @anmatako in https://github.com/colmap/colmap/pull/1168
* Add new CMake option to disable GUI by @anmatako in https://github.com/colmap/colmap/pull/1165
* Fix the memory leak caused by not releasing the memory of the PRNG at the end of the thread by @whuaegeanse in https://github.com/colmap/colmap/pull/1170
* Fix fusion segfault bug by @anmatako in https://github.com/colmap/colmap/pull/1176
* Update SiftGPU to use floorf for floats by @anmatako in https://github.com/colmap/colmap/pull/1182
* fix typo in extraction.cc by @iuk in https://github.com/colmap/colmap/pull/1191
* Improvements to NVM, Cam, Recon3D, and Bundler exporters by @drkoller in https://github.com/colmap/colmap/pull/1187
* Update model_aligner functionality by @anmatako in https://github.com/colmap/colmap/pull/1177
* Add new model_cropper and model_splitter commands by @anmatako in https://github.com/colmap/colmap/pull/1179
* use type point2D_t instead of image_t by @iuk in https://github.com/colmap/colmap/pull/1199
* Fix radial distortion in Cam format exporter by @drkoller in https://github.com/colmap/colmap/pull/1196
* Add new model_transformer command by @anmatako in https://github.com/colmap/colmap/pull/1178
* Fix error of using urllib to download eigen from gitlab by @whuaegeanse in https://github.com/colmap/colmap/pull/1194
* Multi-line string fix in Python model script by @mihaidusmanu in https://github.com/colmap/colmap/pull/1217
* added visibility_sigma to CLI input options for delaunay_mesher. by @Matstah in https://github.com/colmap/colmap/pull/1236
* Backwards compatibility of model_aligner by @tsattler in https://github.com/colmap/colmap/pull/1240
* [update undistortion] update dumped commands by @hiakru in https://github.com/colmap/colmap/pull/1276
* Compute reprojection error in generalized absolute solver by @Skydes in https://github.com/colmap/colmap/pull/1257
* Modifying scripts/python/flickr_downloader.py to create files with correct extensions by @snavely in https://github.com/colmap/colmap/pull/1275
* revise Dockerfile and readme. by @MasahiroOgawa in https://github.com/colmap/colmap/pull/1281
* Update to latest vcpkg version by @ahojnnes in https://github.com/colmap/colmap/pull/1319
* Fix compiler warnings reported by GCC by @ahojnnes in https://github.com/colmap/colmap/pull/1317
* Auto-rotate JPEG images based on EXIF orientation by @ahojnnes in https://github.com/colmap/colmap/pull/1318
* Upgrade vcpkg to fix CI build issues by @ahojnnes in https://github.com/colmap/colmap/pull/1331
* Added descriptor normalization argument to feature_extractor. by @mihaidusmanu in https://github.com/colmap/colmap/pull/1332
* Fix memory leak in the function of StringAppendV by @whuaegeanse in https://github.com/colmap/colmap/pull/1337
* Add CUDA_SAFE_CALL to cudaGetDeviceCount. by @chpatrick in https://github.com/colmap/colmap/pull/1334
* Add missing include in case CUDA/GUI is not available by @ahojnnes in https://github.com/colmap/colmap/pull/1329
* Fix wrong WGS84 model and test cases in GPSTransform by @Freeverc in https://github.com/colmap/colmap/pull/1333
* Fixes bug in sprt.cc: num_inliers was not set. by @rmbrualla in https://github.com/colmap/colmap/pull/1360
* Prevent a divide by zero corner case. by @rmbrualla in https://github.com/colmap/colmap/pull/1361
* Adds missing header. by @rmbrualla in https://github.com/colmap/colmap/pull/1362
* Require Qt in COLMAPConfig only if GUI is enabled by @Skydes in https://github.com/colmap/colmap/pull/1365
* Keep precision in the process of storing in text. by @whuaegeanse in https://github.com/colmap/colmap/pull/1363
* Expose exe internals by @Skydes in https://github.com/colmap/colmap/pull/1366
* Fix inliers matches extraction in EstimateUncalibrated function. by @ferreram in https://github.com/colmap/colmap/pull/1369
* Expose exe internals - fix by @Skydes in https://github.com/colmap/colmap/pull/1368
* Remove deprecated Mac OSX 10.14 image in ADO pipeline by @ahojnnes in https://github.com/colmap/colmap/pull/1383
* Add Mac OSX 11 ADO pipeline job by @ahojnnes in https://github.com/colmap/colmap/pull/1384
* Fix warnings for latest compiler/libraries by @ahojnnes in https://github.com/colmap/colmap/pull/1382
* Fix clang compiler warnings by @ahojnnes in https://github.com/colmap/colmap/pull/1387
* Add Address Sanitizer options and fix reported issues by @ahojnnes in https://github.com/colmap/colmap/pull/1390
* User/joschonb/asan cleanup by @ahojnnes in https://github.com/colmap/colmap/pull/1391
* Add ADO pipeline for Visual Studio 2022 by @ahojnnes in https://github.com/colmap/colmap/pull/1392
* Add ccache option by @ahojnnes in https://github.com/colmap/colmap/pull/1395
* Update ModelAligner to handle GPS and custom coords. and more by @ferreram in https://github.com/colmap/colmap/pull/1371

-----------------------
COLMAP 3.6 (07/24/2020)
-----------------------
* Improved robustness and faster incremental reconstruction process
* Add ``image_deleter`` command to remove images from sparse model
* Add ``image_filter`` command to filter bad registrations from sparse model
* Add ``point_filtering`` command to filter sparse model point clouds
* Add ``database_merger`` command to merge two databases, which is
  useful to parallelize matching across different machines
* Add ``image_undistorter_standalone`` to enable undistorting images
  without a pre-existing full sparse model
* Improved undistortion for fisheye cameras and FOV camera model
* Support for masking input images in feature extraction stage
* Improved HiDPI support in GUI for high-resolution monitors
* Import sparse model when launching GUI from CLI
* Faster CPU-based matching using approximate NN search
* Support for bundle adjustment with fixed extrinsics
* Support for fixing existing images when continuing reconstruction
* Camera model colors in viewer can be customized
* Support for latest GPU architectures in CUDA build
* Support for writing sparse models in Python scripts
* Scripts for building and running COLMAP in Docker
* Many more bug fixes and improvements to code and documentation

-----------------------
COLMAP 3.5 (08/22/2018)
-----------------------
* COLMAP is now released under the BSD license instead of the GPL
* COLMAP is now installed as a library, whose headers can be included and
  libraries linked against from other C/C++ code
* Add hierarchical mapper for parallelized reconstruction or large scenes
* Add sparse and dense Delaunay meshing algorithms, which reconstruct a
  watertight surface using a graph cut on the Delaunay triangulation of the
  reconstructed sparse or dense point cloud
* Improved robustness when merging different models
* Improved pre-trained vocabulary trees available for download
* Add COLMAP as a software entry under Linux desktop systems
* Add support to compile COLMAP on ARM platforms
* Add example Python script to read/write COLMAP database
* Add region of interest (ROI) cropping in image undistortion
* Several import bug fixes for spatial verification in image retrieval
* Add more extensive continuous integration across more compilation scenarios
* Many more bug fixes and improvements to code and documentation

-----------------------
COLMAP 3.4 (01/29/2018)
-----------------------
* Unified command-line interface: The functionality of previous executables have
  been merged into the ``src/exe/colmap.cc`` executable. The GUI can now be
  started using the command ``colmap gui`` and other commands are available
  as ``colmap [command]``. For example, the feature extractor is now available
  as ``colmap feature_extractor [args]`` while all command-line arguments stay
  the same as before. This should result in much faster project compile times
  and smaller disk space usage of the program. More details about the new
  interface are documented at https://colmap.github.io/cli.html
* More complete depth and normal maps with larger patch sizes
* Faster dense stereo computation by skipping rows/columns in patch match,
  improved random sampling in patch match, and faster bilateral NCC
* Better high DPI screen support for the graphical user interface
* Improved model viewer under Windows, which now requires Qt 5.4
* Save computed two-view geometries in database
* Images (keypoint/matches visualization, depth and normal maps) can now be
  saved from the graphical user interface
* Support for PMVS format without sparse bundler file
* Faster covariant feature detection
* Many more bug fixes and improvements

-----------------------
COLMAP 3.3 (11/21/2017)
-----------------------
* Add DSP (Domain Size Pooling) SIFT implementation. DSP-SIFT outperforms
  standard SIFT in most cases, as shown in "Comparative Evaluation of
  Hand-Crafted and Learned Local Features", Schoenberger et al., CVPR 2017
* Improved parameters dense reconstruction of smaller models
* Improved compile times due to various code optimizations
* Add option to specify camera model in automatic reconstruction
* Add new model orientation alignment based on upright image assumption
* Improved numerical stability for generalized absolute pose solver
* Support for image range specification in PMVS dense reconstruction format
* Support for older Python versions in automatic build script
* Fix OpenCV Fisheye camera model to exactly match OpenCV specifications

---------------------
COLMAP 3.2 (9/2/2017)
---------------------
* Fully automatic cross-platform build script (Windows, Mac, Linux)
* Add multi-GPU feature extraction if multiple CUDA devices are available
* Configurable dimension and data type for vocabulary tree implementation
* Add new sequential matching mode for image sequences with high frame-rate
* Add generalized relative pose solver for multi-camera systems
* Add sparse least absolute deviation solver
* Add CPU/GPU options to automatic reconstruction tool
* Add continuous integration system under Windows, Mac, Linux through Github
* Many more bug fixes and improvements

----------------------
COLMAP 3.1 (6/15/2017)
----------------------
* Add fast spatial verification to image retrieval module
* Add binary file format for sparse models by default. Old text format still
  fully compatible and possible conversion in GUI and CLI
* Add cross-platform little endian binary file reading and writing
* Faster and less memory hungry stereo fusion by computing consistency on demand
  and possible limitation of image size in fusion
* Simpler geometric stereo processing interface.
  Now geometric stereo output can be computed using a single pass
* Faster and multi-architecture CUDA compilation
* Add medium quality option in automatic reconstructor
* Many more bug fixes and improvements

----------------------
COLMAP 3.0 (5/22/2017)
----------------------
* Add automatic end-to-end reconstruction tool that automatically performs
  sparse and dense reconstruction on a given set of images
* Add multi-GPU dense stereo if multiple CUDA devices are available
* Add multi-GPU feature matching if multiple CUDA devices are available
* Add Manhattan-world / gravity alignment using line detection
* Add CUDA-based feature extraction useful for usage on clusters
* Add CPU-based feature matching for machines without GPU
* Add new THIN_PRISM_FISHEYE camera model with tangential/radial correction
* Add binary to triangulate existing/empty sparse reconstruction
* Add binary to print summary statistics about sparse reconstruction
* Add transitive feature matching to transitively complete match graph
* Improved scalability of dense reconstruction by using caching
* More stable GPU-based feature matching with informative warnings
* Faster vocabulary tree matching using dynamic scheduling in FLANN
* Faster spatial feature matching using linear index instead of kd-tree
* More stable camera undistortion using numerical Newton iteration
* Improved option parsing with some backwards incompatible option renaming
* Faster compile times by optimizing includes and CUDA flags
* More stable view selection for small baseline scenario in dense reconstruction
* Many more bug fixes and improvements

----------------------
COLMAP 2.1 (12/7/2016)
----------------------
* Support to only index and match specific images in vocabulary tree matching
* Support to perform image retrieval using vocabulary tree
* Several bug fixes and improvements for multi-view stereo module
* Improved Structure-from-Motion initialization strategy
* Support to only reconstruct the scene using specific images in the database
* Add support to merge two models using overlapping registered images
* Add support to geo-register/align models using known camera locations
* Support to only extract specific images in feature extraction module
* Support for snapshot model export during reconstruction
* Skip already undistorted images if they exist in output directory
* Support to limit the number of features in image retrieval for improved speed
* Miscellaneous bug fixes and improvements

---------------------
COLMAP 2.0 (9/8/2016)
---------------------
* Implementation of dense reconstruction pipeline
* Improved feature matching performance
* New bundle adjuster for rigidly mounted multi-camera systems
* New generalized absolute pose solver for multi-camera systems
* New executable to extract colors from all images
* Boost can now be linked in shared and static mode
* Various bug fixes and performance improvements

----------------------
COLMAP 1.1 (5/19/2016)
----------------------
* Implementation of state-of-the-art image retrieval system using Hamming
  embedding for vocabulary tree matching. This should lead to much improved
  matching results as compared to the previous implementation.
* Guided matching as an optional functionality.
* New demo datasets for download.
* Automatically switch to PBA if supported by the project.
* Implementation of EPNP solver for local pose optimization in RANSAC.
* Add option to extract upright SIFT features.
* Saving JPEGs in superb quality by default in export.
* Add option to clear matches and inlier matches in the project.
* New fisheye camera models, including the FOV camera model used by Google
  Project Tango (Thomas Schoeps).
* Extended documentation based on user feedback.
* Fixed typo in documentation (Thomas Schoeps).

---------------------
COLMAP 1.0 (4/4/2016)
---------------------
* Initial release of COLMAP.
