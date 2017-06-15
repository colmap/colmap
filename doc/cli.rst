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

The command for the automatic reconstruction tools would be::

    # The project folder must contain a folder "images" with all the images.
    $ PROJECT_PATH=/path/to/project

    $ ./src/exe/automatic_reconstructor \
        --workspace_path $PROJECT_PATH \
        --image_path $PROJECT_PATH/images

Note that any executable lists all available options using the command-line
argument ``--help``. As an alternative to the automatic reconstruction tool in
case you need more control over the parameters of the individual reconstruction
steps, an exemplary sequence of commands to reconstruct the scene would be::

    # The project folder must contain a folder "images" with all the images.
    $ PROJECT_PATH=/path/to/project

    $ ./src/exe/feature_extractor \
       --database_path $PROJECT_PATH/database.db \
       --image_path $PROJECT_PATH/images

    $ ./src/exe/exhaustive_matcher \
       --database_path $PROJECT_PATH/database.db

    $ mkdir $PROJECT_PATH/sparse

    $ ./src/exe/mapper \
        --database_path $PROJECT_PATH/database.db \
        --image_path $PROJECT_PATH/images \
        --export_path $PROJECT_PATH/sparse

    $ mkdir $PROJECT_PATH/dense

    $ ./src/exe/image_undistorter \
        --image_path $PROJECT_PATH/images \
        --input_path $PROJECT_PATH/sparse/0 \
        --output_path $PROJECT_PATH/dense \
        --output_type COLMAP \
        --max_image_size 2000

    $ ./exe/dense_stereo \
        --workspace_path $PROJECT_PATH/dense \
        --workspace_format COLMAP \
        --DenseStereo.geom_consistency true

    $ ./exe/dense_fuser \
        --workspace_path $PROJECT_PATH/dense \
        --workspace_format COLMAP \
        --input_type geometric \
        --output_path $PROJECT_PATH/dense/fused.ply

    $ ./src/exe/dense_mesher \
        --input_path $PROJECT_PATH/dense/fused.ply \
        --output_path $PROJECT_PATH/dense/meshed.ply

If you want to run COLMAP on a computer (e.g., cluster or cloud service) without
an attached display, you should run the ``feature_extractor`` and set the
``--SiftGPUExtraction.index 0`` explicitly if a CUDA device is available or with
the option ``--use_gpu false``. Then, you should run the ``*_matcher`` with
``--SiftMatching.use_gpu true`` if a CUDA device is available or with ``--SiftMatching.use_gpu false`` for
CPU-based feature matching.

Help
----

All executables have a "-h,--help" command-line argument to show the usage and
the available options, e.g.::

    $ ./src/exe/feature_extractor -h

        Options can either be specified via command-line or by defining
        them in a .ini project file passed to `--project_path`.

          -h [ --help ]
          --project_path arg
          --log_to_stderr arg (=0)
          --log_level arg (=2)
          --database_path arg
          --image_path arg
          --use_gpu arg (=1)
          --image_list_path arg
          --ImageReader.camera_model arg (=SIMPLE_RADIAL)
          --ImageReader.single_camera arg (=0)
          --ImageReader.camera_params arg
          --ImageReader.default_focal_length_factor arg (=1.2)
          --SiftExtraction.max_image_size arg (=3200)
          --SiftExtraction.max_num_features arg (=8192)
          --SiftExtraction.first_octave arg (=-1)
          --SiftExtraction.octave_resolution arg (=3)
          --SiftExtraction.peak_threshold arg (=0.0066666666666666671)
          --SiftExtraction.edge_threshold arg (=10)
          --SiftExtraction.max_num_orientations arg (=2)
          --SiftExtraction.upright arg (=0)
          --SiftCPUExtraction.batch_size_factor arg (=3)
          --SiftCPUExtraction.num_threads arg (=-1)
          --SiftGPUExtraction.index arg (=-1)


The available options can either be provided directly from the command-line or
through a `.ini` file provided to ``--project_path``.


Executables
-----------

- ``colmap``: The graphical user interface, see
  :ref:`Graphical User Interface <gui>` for more information.

- ``automatic_reconstruction``: Automatically reconstruct sparse and dense model
  for a set of input images.

- ``feature_extractor``, ``feature_importer``: Perform feature extraction or
  import features for a set of images.

- ``exhaustive_matcher``, ``vocab_tree_matcher``, ``sequential_matcher``,
  ``spatial_matcher``, ``transitive_matcher``, ``matches_importer``:
  Perform feature matching after performing feature extraction.

- ``mapper``: Sparse 3D reconstruction / mapping of the dataset using SfM after
  performing feature extraction and matching.

- ``image_undistorter``: Undistort images and/or export them for MVS or to
  external dense reconstruction software, such as CMVS/PMVS.

- ``image_rectifier``: Stereo rectify cameras and undistort images for stereo
  disparity estimation.

- ``dense_stereo``: Dense 3D reconstruction / mapping using MVS after running
  the ``image_undistorter`` to initialize the workspace.

- ``dense_fuser``: Fusion of MVS depth and normal maps to a colored point cloud.

- ``dense_mesher``: Meshing of the fused point cloud using Poisson surface
  reconstruction.

- ``image_registrator``: Register new images in the database against an existing
  model, e.g., when extracting features and matching newly added images in a
  database after running ``mapper``. Note that no bundle adjustment or
  triangulation is performed.

- ``point_triangulator``: Triangulate all observations of registered images in
  an existing model using the feature matches in a database.

- ``bundle_adjuster``: Run global bundle adjustment on a reconstructed scene,
  e.g., when a refinement of the intrinsics is needed or
  after running the ``image_registrator``.

- ``database_creator``: Create an empty COLMAP SQLite database with the
  necessary database schema information.

- ``model_analyzer``: Print statistics about reconstructions.

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
  Pre-trained trees can be downloaded from https://demuc.de/colmap/.
  This is useful if you want to build a custom tree with a different trade-off
  in terms of precision/recall vs. speed.


Visualization
-------------

If you want to quickly visualize the outputs of the sparse or dense
reconstruction pipelines, COLMAP offers you the following possibilities:

- The sparse point cloud obtained with the ``mapper`` can be visualized via the
  COLMAP GUI by importing the following files: choose ``File > Import Model``
  and select the folder where the three files, ``cameras.txt``,``images.txt``,
  and ``points3d.txt`` are located.

- The dense point cloud obtained with the ``dense_fuser`` can be visualized via
  the COLMAP GUI by importing ``fused.ply``: choose
  ``File > Import Model from...`` and then select the file ``fused.ply``.

- The dense mesh model ``meshed.ply`` obtained with the ``dense_mesher`` can
  currently not be visualized with COLMAP, instead you can, e.g., use Meshlab.
