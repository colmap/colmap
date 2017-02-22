.. _cli:

Command-line Interface
======================

The command-line interface provides access to most of COLMAP's functionality for
automated scripting. Each core functionality uses a different executable inside
the ``./src/exe/*`` folder.

Example
-------

Assume you stored the images of your project in the following folder structure::

    /path/to/project/...
    +── images
    │   +── image1.jpg
    │   +── image2.jpg
    │   +── ...
    │   +── imageN.jpg

Then, an exemplary sequence of commands to reconstruct the scene would be::

    # The project folder must contain a folder "images" with all the images.
    PROJECT_PATH=/path/to/project

    $ ./src/exe/feature_extractor \
       --General.database_path $PROJECT_PATH/database.db \
       --General.image_path $PROJECT_PATH/images

    $ ./src/exe/exhaustive_matcher \
       --General.database_path $PROJECT_PATH/database.db

    $ mkdir $PROJECT_PATH/sparse

    $ ./src/exe/mapper \
        --General.database_path $PROJECT_PATH/database.db \
        --General.image_path $PROJECT_PATH/images \
        --export_path $PROJECT_PATH/sparse

    $ mkdir $PROJECT_PATH/dense

    $ ./src/exe/image_undistorter \
        --General.image_path $PROJECT_PATH/images \
        --input_path $PROJECT_PATH/sparse/0 \
        --output_path $PROJECT_PATH/dense \
        --output_type COLMAP \
        --max_image_size 2000

    $ ./src/exe/dense_mapper \
        --workspace_path $PROJECT_PATH/dense \
        --workspace_format COLMAP \
        --DenseMapperOptions.max_image_size 0 \
        --DenseMapperOptions.patch_match_filter false \
        --DenseMapperOptions.patch_match_geom_consistency false \
        --DenseMapperOptions.patch_match_num_iterations 4

    $ ./exe/dense_mapper \
        --workspace_path $PROJECT_PATH/dense \
        --workspace_format COLMAP \
        --DenseMapperOptions.max_image_size 0 \
        --DenseMapperOptions.patch_match_filter true \
        --DenseMapperOptions.patch_match_geom_consistency true

    $ ./exe/dense_fuser \
        --workspace_path $PROJECT_PATH/dense \
        --workspace_format COLMAP \
        --input_type geometric \
        --output_path $PROJECT_PATH/dense/point-cloud.ply

    $ ./src/exe/dense_mesher \
        --input_path $PROJECT_PATH/dense/point-cloud.ply \
        --output_path $PROJECT_PATH/dense/mesh.ply

If you want to run COLMAP on a computer (e.g., cluster or cloud service) without
an attached display, you should run the ``feature_extractor`` with the option
``--use_gpu false`` and the ``*_matcher`` with ``--gpu_index 0`` if a CUDA
device is available or with ``--use_gpu false`` for CPU-based matching.

Help
----

All executables have a "-h,--help" command-line argument to show the usage and
the available options, e.g.::

    $ ./src/exe/feature_extractor -h

      -h [ --help ]                     Configuration can either be specified
                                        via command_line or by defining the
                                        parameters in a .ini project_file (see
                                        `--project_path`).
      --project_path arg
      --General.debug_log_to_stderr arg (=0)
      --General.debug_log_level arg (=2)
      --General.database_path arg
      --General.image_path arg
      --ExtractionOptions.camera_model arg (=SIMPLE_RADIAL)
      --ExtractionOptions.single_camera arg (=0)
      --ExtractionOptions.camera_params arg
      --ExtractionOptions.default_focal_length_factor arg (=1.2)
      --ExtractionOptions.sift_options_max_image_size arg (=3200)
      --ExtractionOptions.sift_options_max_num_features arg (=8192)
      --ExtractionOptions.sift_options_first_octave arg (=-1)
      --ExtractionOptions.sift_options_octave_resolution arg (=3)
      --ExtractionOptions.sift_options_peak_threshold arg (=0.0066666666666666671)
      --ExtractionOptions.sift_options_edge_threshold arg (=10)
      --ExtractionOptions.sift_options_max_num_orientations arg (=2)
      --ExtractionOptions.sift_options_upright arg (=0)
      --ExtractionOptions.cpu_options_batch_size_factor arg (=3)
      --ExtractionOptions.cpu_options_num_threads arg (=-1)
      --use_gpu arg (=1)


The available options can either be provided directly from the command-line or
through a `.ini` file provided to ``--project_path``.


Executables
-----------

- ``colmap``: The graphical user interface, see
  :ref:`Graphical User Interface <gui>` for more information.

- ``feature_extractor``, ``feature_importer``: Perform feature extraction or
  import features for a set of images.

- ``exhaustive_matcher``, ``vocab_tree_matcher``, ``sequential_matcher``,
  ``spatial_matcher``, ``matches_importer``: Perform feature matching after
  performing feature extraction.

- ``mapper``: Sparse 3D reconstruction / mapping of the dataset using SfM after
  performing feature extraction and matching.

- ``image_undistorter``: Undistort images and/or export them for MVS or to
  external dense reconstruction software, such as CMVS/PMVS.

- ``image_rectifier``: Stereo rectify cameras and undistort images for stereo
  disparity estimation.

- ``dense_mapper``: Dense 3D reconstruction / mapping using MVS after running
  the ``image_undistorter`` to initialize the workspace.

- ``dense_fuser``: Fusion of MVS depth and normal maps to a colored point cloud.

- ``dense_mesher``: Meshing of the fused point cloud using Poisson surface
  reconstruction.

- ``image_registrator``: Register new images in the database against an existing
  model, e.g., when extracting features and matching newly added images in a
  database after running ``mapper``. Note that no bundle adjustment or
  triangulation is performed.

- ``bundle_adjuster``: Run global bundle adjustment on a reconstructed scene,
  e.g., when a refinement of the intrinsics is needed or
  after running the ``image_registrator``.

- ``database_creator``: Create an empty COLMAP SQLite database with the
  necessary database schema information.

- ``model_aligner``: Align/geo-register model to coordinate system of given
  camera centers.

- ``model_orientation_aligner``: Align the coordinate axis of a model using a
  Manhattan world assumption.

- ``model_converter``: Convert the COLMAP export format to another format,
  such as PLY or NVM.

- ``model_merger``: Attempt to merge two disconnected reconstructions,
  if they have common registered images.

- ``color_extractor``: Extract mean colors for all 3D points of a model.

- ``vocab_tree_builder``: Create a vocabulary tree from a database with
  extracted images. This is an offline procedure and can be run once, while the
  same vocabulary tree can be reused for other datasets. Note that, as a rule of
  thumb, you should use at least 10-100 times more features than visual words.
  Pre-trained trees can be downloaded from
  http://people.inf.ethz.ch/jschoenb/colmap/.
  This is useful if you want to build a custom tree with a different trade-off
  in terms of precision/recall vs. speed.
