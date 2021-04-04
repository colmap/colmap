.. _cli:

Command-line Interface
======================

The command-line interface provides access to all of COLMAP's functionality for
automated scripting. Each core functionality is implemented as a command to the
``colmap`` executable. Run ``colmap -h`` to list the available commands (or
``COLMAP.bat -h`` under Windows). Note that if you run COLMAP from the CMake
build folder, the executable is located at ``./src/exe/colmap``. To start the
graphical user interface, run ``colmap gui``.

Example
-------

Assuming you stored the images of your project in the following structure::

    /path/to/project/...
    +── images
    │   +── image1.jpg
    │   +── image2.jpg
    │   +── ...
    │   +── imageN.jpg

The command for the automatic reconstruction tool would be::

    # The project folder must contain a folder "images" with all the images.
    $ DATASET_PATH=/path/to/project

    $ colmap automatic_reconstructor \
        --workspace_path $DATASET_PATH \
        --image_path $DATASET_PATH/images

Note that any command lists all available options using the ``-h,--help``
command-line argument. In case you need more control over the individual
parameters of the reconstruction process, you can execute the following sequence
of commands as an alternative to the automatic reconstruction command::

    # The project folder must contain a folder "images" with all the images.
    $ DATASET_PATH=/path/to/dataset

    $ colmap feature_extractor \
       --database_path $DATASET_PATH/database.db \
       --image_path $DATASET_PATH/images

    $ colmap exhaustive_matcher \
       --database_path $DATASET_PATH/database.db

    $ mkdir $DATASET_PATH/sparse

    $ colmap mapper \
        --database_path $DATASET_PATH/database.db \
        --image_path $DATASET_PATH/images \
        --output_path $DATASET_PATH/sparse

    $ mkdir $DATASET_PATH/dense

    $ colmap image_undistorter \
        --image_path $DATASET_PATH/images \
        --input_path $DATASET_PATH/sparse/0 \
        --output_path $DATASET_PATH/dense \
        --output_type COLMAP \
        --max_image_size 2000

    $ colmap patch_match_stereo \
        --workspace_path $DATASET_PATH/dense \
        --workspace_format COLMAP \
        --PatchMatchStereo.geom_consistency true

    $ colmap stereo_fusion \
        --workspace_path $DATASET_PATH/dense \
        --workspace_format COLMAP \
        --input_type geometric \
        --output_path $DATASET_PATH/dense/fused.ply

    $ colmap poisson_mesher \
        --input_path $DATASET_PATH/dense/fused.ply \
        --output_path $DATASET_PATH/dense/meshed-poisson.ply

    $ colmap delaunay_mesher \
        --input_path $DATASET_PATH/dense \
        --output_path $DATASET_PATH/dense/meshed-delaunay.ply

If you want to run COLMAP on a computer without an attached display (e.g.,
cluster or cloud service), COLMAP automatically switches to use CUDA if
supported by your system. If no CUDA enabled device is available, you can
manually select to use CPU-based feature extraction and matching by setting the
``--SiftExtraction.use_gpu 0`` and ``--SiftMatching.use_gpu 0`` options.

Help
----

The available commands can be listed using the command::

    $ colmap help

        Usage:
          colmap [command] [options]

        Documentation:
          https://colmap.github.io/

        Example usage:
          colmap help [ -h, --help ]
          colmap gui
          colmap gui -h [ --help ]
          colmap automatic_reconstructor -h [ --help ]
          colmap automatic_reconstructor --image_path IMAGES --workspace_path WORKSPACE
          colmap feature_extractor --image_path IMAGES --database_path DATABASE
          colmap exhaustive_matcher --database_path DATABASE
          colmap mapper --image_path IMAGES --database_path DATABASE --output_path MODEL
          ...

        Available commands:
          help
          gui
          automatic_reconstructor
          bundle_adjuster
          color_extractor
          database_creator
          delaunay_mesher
          exhaustive_matcher
          feature_extractor
          feature_importer
          image_deleter
          image_rectifier
          image_registrator
          image_undistorter
          mapper
          matches_importer
          model_aligner
          model_analyzer
          model_converter
          model_merger
          model_orientation_aligner
          patch_match_stereo
          point_triangulator
          poisson_mesher
          rig_bundle_adjuster
          sequential_matcher
          spatial_matcher
          stereo_fusion
          transitive_matcher
          vocab_tree_builder
          vocab_tree_matcher
          vocab_tree_retriever

And each command has a ``-h,--help`` command-line argument to show the usage and
the available options, e.g.::

    $ colmap feature_extractor -h

        Options can either be specified via command-line or by defining
        them in a .ini project file passed to `--project_path`.

          -h [ --help ]
          --project_path arg
          --database_path arg
          --image_path arg
          --image_list_path arg
          --ImageReader.camera_model arg (=SIMPLE_RADIAL)
          --ImageReader.single_camera arg (=0)
          --ImageReader.camera_params arg
          --ImageReader.default_focal_length_factor arg (=1.2)
          --SiftExtraction.num_threads arg (=-1)
          --SiftExtraction.use_gpu arg (=1)
          --SiftExtraction.gpu_index arg (=-1)
          --SiftExtraction.max_image_size arg (=3200)
          --SiftExtraction.max_num_features arg (=8192)
          --SiftExtraction.first_octave arg (=-1)
          --SiftExtraction.num_octaves arg (=4)
          --SiftExtraction.octave_resolution arg (=3)
          --SiftExtraction.peak_threshold arg (=0.0066666666666666671)
          --SiftExtraction.edge_threshold arg (=10)
          --SiftExtraction.estimate_affine_shape arg (=0)
          --SiftExtraction.max_num_orientations arg (=2)
          --SiftExtraction.upright arg (=0)
          --SiftExtraction.domain_size_pooling arg (=0)
          --SiftExtraction.dsp_min_scale arg (=0.16666666666666666)
          --SiftExtraction.dsp_max_scale arg (=3)
          --SiftExtraction.dsp_num_scales arg (=10)


The available options can either be provided directly from the command-line or
through a `.ini` file provided to ``--project_path``.


Commands
--------

The following list briefly documents the functionality of each command, that is
available as ``colmap [command]``:

- ``gui``: The graphical user interface, see
  :ref:`Graphical User Interface <gui>` for more information.

- ``automatic_reconstruction``: Automatically reconstruct sparse and dense model
  for a set of input images.

- ``project_generator``: Generate project files at different quality settings.

- ``feature_extractor``, ``feature_importer``: Perform feature extraction or
  import features for a set of images.

- ``exhaustive_matcher``, ``vocab_tree_matcher``, ``sequential_matcher``,
  ``spatial_matcher``, ``transitive_matcher``, ``matches_importer``:
  Perform feature matching after performing feature extraction.

- ``mapper``: Sparse 3D reconstruction / mapping of the dataset using SfM after
  performing feature extraction and matching.

- ``hierarchical_mapper``: Sparse 3D reconstruction / mapping of the dataset
  using hierarchical SfM after performing feature extraction and matching.
  This parallelizes the reconstruction process by partitioning the scene into
  overlapping submodels and then reconstructing each submodel independently.
  Finally, the overlapping submodels are merged into a single reconstruction.
  It is recommended to run a few rounds of point triangulation and bundle
  adjustment after this step.

- ``image_undistorter``: Undistort images and/or export them for MVS or to
  external dense reconstruction software, such as CMVS/PMVS.

- ``image_rectifier``: Stereo rectify cameras and undistort images for stereo
  disparity estimation.

- ``image_filterer``: Filter images from a sparse reconstruction.

- ``image_deleter``: Delete specific images from a sparse reconstruction.

- ``patch_match_stereo``: Dense 3D reconstruction / mapping using MVS after
  running the ``image_undistorter`` to initialize the workspace.

- ``stereo_fusion``: Fusion of ``patch_match_stereo`` results into to a colored
  point cloud.

- ``poisson_mesher``: Meshing of the fused point cloud using Poisson
  surface reconstruction.

- ``delaunay_mesher``: Meshing of the reconstructed sparse or dense point cloud
  using a graph cut on the Delaunay triangulation and visibility voting.

- ``image_registrator``: Register new images in the database against an existing
  model, e.g., when extracting features and matching newly added images in a
  database after running ``mapper``. Note that no bundle adjustment or
  triangulation is performed.

- ``point_triangulator``: Triangulate all observations of registered images in
  an existing model using the feature matches in a database.

- ``point_filtering``: Filter sparse points in model by enforcing criteria,
  such as minimum track length, maximum reprojection error, etc.

- ``bundle_adjuster``: Run global bundle adjustment on a reconstructed scene,
  e.g., when a refinement of the intrinsics is needed or
  after running the ``image_registrator``.

- ``database_creator``: Create an empty COLMAP SQLite database with the
  necessary database schema information.

- ``database_merger``: Merge two databases into a new database. Note that the
  cameras will not be merged and that the unique camera and image identifiers
  might change during the merging process.

- ``model_analyzer``: Print statistics about reconstructions.

- ``model_aligner``: Align/geo-register model to coordinate system of given
  camera centers.

- ``model_orientation_aligner``: Align the coordinate axis of a model using a
  Manhattan world assumption.

- ``model_converter``: Convert the COLMAP export format to another format,
  such as PLY or NVM.

- ``model_cropper``: Crop model to specific bounding box described in GPS or
  model coordinate system.

- ``model_splitter``: Divide model in rectangular sub-models specified from
  file containing bounding box coordinates, or max extent of sub-model, or
  number of subdivisions in each dimension.

- ``model_merger``: Attempt to merge two disconnected reconstructions,
  if they have common registered images.

- ``color_extractor``: Extract mean colors for all 3D points of a model.

- ``vocab_tree_builder``: Create a vocabulary tree from a database with
  extracted images. This is an offline procedure and can be run once, while the
  same vocabulary tree can be reused for other datasets. Note that, as a rule of
  thumb, you should use at least 10-100 times more features than visual words.
  Pre-trained trees can be downloaded from https://demuc.de/colmap/.
  This is useful if you want to build a custom tree with a different trade-off
  in terms of precision/recall vs. speed.

- ``vocab_tree_retriever``: Perform vocabulary tree based image retrieval.


Visualization
-------------

If you want to quickly visualize the outputs of the sparse or dense
reconstruction pipelines, COLMAP offers you the following possibilities:

- The sparse point cloud obtained with the ``mapper`` can be visualized via the
  COLMAP GUI by importing the following files: choose ``File > Import Model``
  and select the folder where the three files, ``cameras.txt``, ``images.txt``,
  and ``points3d.txt`` are located.

- The dense point cloud obtained with the ``stereo_fusion`` can be visualized
  via the COLMAP GUI by importing ``fused.ply``: choose
  ``File > Import Model from...`` and then select the file ``fused.ply``.

- The dense mesh model ``meshed-*.ply`` obtained with the ``poisson_mesher`` or
  the ``delaunay_mesher`` can currently not be visualized with COLMAP, instead
  you can use an external viewer, such as Meshlab.
