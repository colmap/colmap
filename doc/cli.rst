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

    ./src/exe/feature_extractor \
        --General.database_path /path/to/project/database.db \
        --General.image_path /path/to/project/images

    ./src/exe/exhaustive_matcher \
        --General.database_path /path/to/project/database.db

    mkdir /path/to/project/results

    ./src/exe/mapper \
        --General.database_path /path/to/project/database.db \
        --General.image_path /path/to/project/images \
        --export_path /path/to/project/results


Executables
-----------

- ``colmap``: The graphical user interface, see
  :ref:`Graphical User Interface <gui>` for more information.

- ``feature_extractor``, ``feature_importer``: Perform feature extraction or
  import features for a set of images.

- ``exhaustive_matcher``, ``vocab_tree_matcher``, ``sequential_matcher``,
  ``spatial_matcher``, ``matches_importer``: Perform feature matching after
  performing feature extraction.

- ``mapper``: 3D reconstruction / mapping of the dataset after performing
  feature extraction and matching.

- ``image_undistorter``: Undistort images and/or export them to external dense
  reconstruction software, such as CMVS/PMVS.

- ``image_registrator``: Register new images in the database against an existing
  model, e.g., when extracting features and matching newly added images in a
  database after running ``mapper``. Note that no bundle adjustment or
  triangulation is performed.

- ``bundle_adjuster``: Run global bundle adjustment on a reconstructed scene,
  e.g., when a refinement of the intrinsics is needed or
  after running the ``image_registrator``.

- ``database_creator``: Create an empty COLMAP SQLite database with the
  necessary database schema information.

- ``model_converter``: Convert the COLMAP export format to another format,
  such as PLY or NVM.

- ``vocab_tree_builder``: Create a vocabulary tree from a database with
  extracted images. This is an offline procedure and can be run once, while the
  same vocabulary tree can be reused for other datasets. Note that, as a rule of
  thumb, you should use at least 10-100 times more features than visual words.
  Pre-trained trees can be downloaded from
  http://people.inf.ethz.ch/jschoenb/colmap/.
  This is useful if you want to build a custom tree with a different trade-off
  in terms of precision/recall vs. speed.
