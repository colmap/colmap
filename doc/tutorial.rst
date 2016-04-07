.. _tutorial:

Tutorial
========

Structure-from-Motion
---------------------

Structure-from-Motion is the process of reconstructing 3D structure from its
projections into a series of images. The input is a set of overlapping images of
the same object, taken from different viewpoints. The output is a 3-D
reconstruction of the object, and the reconstructed intrinsic and extrinsic
camera parameters of all images. Typically, Structure-from-Motion systems divide
this process into three stages:

    1) Feature detection and extraction
    2) Feature matching and geometric verification
    3) Structure and motion reconstruction

COLMAP reflects these stages in different modules, that can be combined
depending on the application. More information on Structure-from-Motion in
general and the algorithms in COLMAP can be found in [schoenberger16]_.

If you have control over the picture capture process, please follow these
guidelines for optimal reconstruction results:

* Capture images with **good texture**. Avoid completely texture-less images
  (e.g., a white wall or empty desk). If the scene does not contain enough
  texture itself, you could place additional background objects, such as
  posters, etc.

* Capture images at **similar illumination** conditions. Avoid high dynamic
  range scenes (e.g., pictures against the sun with shadows or pictures
  through doors/windows). Avoid specularities on shiny surfaces.

* Capture images with **high visual overlap**. Make sure that each object is
  seen in at least 3 images -- the more images the better.

* Capture images from **different viewpoints**. Do not take images from the
  same location by only rotating the camera, e.g., make a few steps after each
  shot. At the same time, try to have enough images from a relatively similar
  viewpoint.


Preface
-------

COLMAP requires only few steps to do a standard reconstruction for a general
user. For more experienced users, the program exposes many different parameters,
only some of which are intuitive to a general user. The program should work
without the need to modify any parameters. The defaults are chosen as a tradeoff
between reconstruction robustness/quality and speed. If in doubt what setting to
choose, stick to the defaults. The source code contains more documentation about
all parameters.

COLMAP is research software and in rare cases it may exit ungracefully if some
constraints are not fulfilled. In this case, the program prints a traceback to
stdout. To see this traceback or more debug information, it is recommended to
run the executables (including the GUI) from the command-line, where you can
define various levels of logging verbosity.


Terminology
-----------

The term **camera** is associated with the physical object of a camera using the
same zoom-factor and lens. A camera defines the intrinsic projection model in
COLMAP. A single camera can take multiple images with the same resolution,
intrinsic parameters, and distortion characteristics. The term **image** is
associated with a bitmap file, e.g., a JPEG or PNG file on disk. COLMAP detects
**keypoints** in each each image whose appearance is described by numerical
**descriptors**. Pure appearance-based correspondences between
keypoints/descriptors are defined by **matches**, while **inlier matches** are
geometrically verified and used for the reconstruction procedure.


Data Structure
--------------

COLMAP assumes that all input images are in one input directory with potentially
nested sub-directories. It recursively considers all images stored in this
directory, and it supports various different image formats (see `FreeImage
<http://freeimage.sourceforge.net/documentation.html>`_). Other files are
automatically ignored. If high performance is a requirement, then you should
separate any files that are not images. Images are identified uniquely by their
relative file path. For lather processing, such as image undistortion or dense
reconstruction, the relative folder structure should be preserved. COLMAP does
not modify the input images or directory and all extracted data is stored in a
single, self-contained SQLite database file (see :doc:`database`).

The first step is to create a new project by choosing ``File > New project``. In
this dialog, you must select where to store the database and the folder that
contains the images. For convenience, you can save the entire project settings
to a configuration file by choosing ``File > Save project``. The project
configuration stores the absolute path information of the database and image
folder in addition to any other parameter settings. If you decide to move the
database or image folder, you must change the paths accordingly by creating a
new project. Alternatively, the resulting `.ini` configuration file can be
directly modified in a text editor of your choice. To reopen an existing
project, you can simply open the configuration file by choosing ``File > Open
project``. Note that all COLMAP executables can be started from the command-line
by either specifying individual settings as command-line arguments or by
providing the path to the project configuration file.

An example folder structure could look like this::

    /path/to/project/...
    +── images
    │   +── image1.jpg
    │   +── image2.jpg
    │   +── ...
    │   +── imageN.jpg
    +── database.db
    +── project.ini

In this example, you would select `/path/to/project/images` as the image folder
path and `/path/to/project/database.db` as the database file path.


Feature Detection and Extraction
--------------------------------

In the first step, feature detection/extraction finds sparse feature points in
the image and describes their appearance using a numerical descriptor. COLMAP
imports images and performs feature detection/extraction in one step in order to
only load images from disk once.

Next, choose ``Processing > Extract features``. In this dialog, you must first
decide on the employed intrinsic camera model. You can either automatically
extract focal length information from the embedded EXIF information or manually
specify intrinsic parameters, e.g., as obtained in a lab calibration. If an
image has partial EXIF information, COLMAP tries to find the missing camera
specifications in a large database of camera models automatically. If all your
images were captured by the same physical camera with identical zoom factor, it
is recommended to share intrinsics between all images. Note that the program
will exit ungracefully if the same camera model is shared among all images but
not all images have the same size or EXIF focal length. If you have several
groups of images that share the same intrinsic camera parameters, you can easily
modify the camera models at a later point (see :ref:`Database Management
<database-management>`). If in doubt what to choose in this step, simply stick
to the default parameters.

You can either detect and extract new features from the images or import
existing features from text files. COLMAP extracts [lowe04]_ features either on
the GPU or the CPU. The GPU version requires an attached display, while the CPU
version is recommended for use on a server. In general, the GPU version is
favorable as it has a customized feature detection mode that often produces
higher quality features in the case of high contrast images. If you import
existing features, every image must have a text file next to it (e.g.,
`/path/to/image1.jpg` and `/path/to/image1.jpg.txt`) in the following format::

    NUM_FEATURES 128
    X Y SCALE ORIENTATION D_1 D_2 D_3 ... D_128
    ...
    X Y SCALE ORIENTATION D_1 D_2 D_3 ... D_128

where `X, Y, SCALE, ORIENTATION` are floating point numbers and `D_1...D_128`
values in the range `0...255`. The file should have `NUM_FEATURES` lines with
one line per feature. Note that by convention the upper left corner of an image
has coordinate `(0, 0)` and the center of the upper left most pixel has
coordinate `(0.5, 0.5)`. If you must  import features for large image
collections, it is much more efficient to directly access the database with your
favorite scripting language (see :ref:`Database Format <database-format>`).

If you are done setting all options, choose ``Extract`` and wait for the
extraction to finish or cancel. If you cancel during the extraction process, the
next time you start extracting images for the same project, COLMAP automatically
continues where it left off. This also allows you to add images to an existing
project/reconstruction. In this case, be sure to verify the camera parameters
when using shared intrinsics.

All extracted data will be stored in the database file and can be
reviewed/managed in the database management tool (see :ref:`Database Management
<database-management>`) or, for experts, directly modified using SQLite (see
:ref:`Database Format <database-format>`).


Feature Matching and Geometric Verification
-------------------------------------------

In the second step, feature matching and geometric verification finds
correspondences between the feature points in different images.

Please, choose ``Processing > Match features`` and select on of the provided
matching modes, that are intended for different input scenarios:

* **Exhaustive Matching**: If the number of images in your dataset is
  relatively low (up to several hundreds), this matching mode should be fast
  enough and leads to the best reconstruction results. Here, every image is
  matched against every other image, while the block size determines how many
  images are loaded from disk into memory at the same time. The preemptive
  option filters image pairs that are not likely to match [wu13]_.
  Typically though, vocabulary tree matching yields better results than
  preemptive matching [schoenberger15]_.

* **Sequential Matching**: This mode is useful if the images are acquired in
  sequential order, e.g., by a video camera. In this case, consecutive frames
  have visual overlap and there is no need to match all image pairs
  exhaustively. Instead, consecutively captured images are matched against
  each other. This matching mode has built-in loop detection based on a
  vocabulary tree, where every N-th image (`loop_detection_period`) is matched
  against its visually most similar images (`loop_detection_num_images`). Note
  that image file names must be ordered sequentially (e.g., `image0001.jpg`,
  `image0002.jpg`, etc.). You can verify the correct order in the database
  management tool (see :ref:`Database Format <database-format>`). Note that
  loop detection requires a pre-trained vocabulary tree, that can be downloaded
  from the project website.

* **Vocabulary Tree Matching**: In this matching mode, every image is matched
  against its visual nearest neighbors using a vocabulary tree. This is the
  recommended matching mode for large image collections (several thousands).
  This requires a pre-trained vocabulary tree, that can be downloaded from
  the project website.

* **Spatial Matching**: This matching mode matches every image against its
  spatial nearest neighbors. Spatial locations can be manually set in the
  database management. By default, COLMAP also extracts GPS information from
  EXIF and uses it for spatial nearest neighbor search. If accurate prior
  location information is available, this is the recommended matching mode.

* **Custom Matching**: This mode allows to specify individual image pairs for
  matching or to import individual feature matches. To specify image pairs, you
  have to provide a text file with one image pair per line::

    image1.jpg image2.jpg
    image1.jpg image3.jpg
    ...

  where `image1.jpg` is the relative path in the image folder. You have two
  options to import individual feature matches. Either raw feature matches,
  which are not geometrically verified or already geometrically verified feature
  matches. In both cases, the expected format is::

    image1.jpg image2.jpg
    0 1
    1 2
    3 4
    <empty-line>
    image1.jpg image3.jpg
    0 1
    1 2
    3 4
    4 5
    <empty-line>
    ...

  where `image1.jpg` is the relative path in the image folder and the pairs of
  numbers are zero-based feature indices in the respective images. If you must
  import many matches for large image collections, it is more efficient to
  directly access the database with a scripting language of your choice.

If you are done setting all options, choose ``Match`` and wait for the matching
to finish or cancel in between. Note that this step can take a significant
amount of time depending on the number of images, the number of features per
image, and the chosen matching mode. Expected times for exhaustive matching are
from a few minutes for tens of images to a few hours for hundreds of images to
days or weeks for thousands of images. If you cancel the matching process or
import new images after matching, COLMAP only matches image pairs that have not
been matched previously. The overhead of skipping already matched image pairs is
low. This also enables to match additional images imported after an initial
matching and it enables to combine different matching modes for the same
dataset.

All extracted data will be stored in the database file and can be
reviewed/managed in the database management tool (see :ref:`Database Management
<database-management>`) or, for experts, directly modified using SQLite (see
:ref:`Database Format <database-format>`).

Note that feature matching requires a GPU and that the display performance of
your computer might degrade significantly during the matching process. If your
system has multiple CUDA-enabled GPUs, you can select specific GPUs with the
`gpu_index` option.


Structure and motion reconstruction
-----------------------------------

After producing the scene graph in the previous two steps, you can start the
incremental reconstruction process by choosing ``Reconstruction > Start``.
COLMAP first loads all extracted data from the database into memory and seeds
the reconstruction from an initial image pair. Then, the scene is incrementally
extended by registering new images and triangulating new points. The results are
visualized in "real-time" during this reconstruction process. COLMAP attempts to
reconstruct multiple models if not all images are registered into the same
model. The different models can be selected from the drop-down menu in the
toolbar. If all your images use the `SIMPLE_RADIAL` camera model (default)
without shared intrinsics, you can use PBA [wu11]_ for fast bundle adjustment,
which can be activated in the reconstruction options under the bundle adjustment
section (`use_pba=true`).

Ideally, the reconstruction works fine and all images are registered. If this is
not the case, it is recommended to:

* Perform additional matching. For best results, use exhaustive matching,
  increase the number of nearest neighbors in vocabulary tree matching, or
  increase the overlap in sequential matching, etc.

* Manually choose an initial image pair, if COLMAP fails to initialize. Choose
  ``Reconstruction > Reconstruction options > Init`` and set images from the
  database management tool that have enough matches from different viewpoints.


Exporting results
-----------------

COLMAP provides several export options for further processing. For full
flexibility, it is recommended to export the reconstruction in COLMAP's data
format by choosing ``File > Export`` to export the currently viewed model or
``File > Export all models`` to export all reconstructed models. The model is
exported in the selected folder using separate text files for the reconstructed
cameras, images, and points. When exporting in COLMAP's data format, you can re-
import the reconstruction for later visualization, image undistortion, or to
continue an existing reconstruction from where it left off. To import a model,
choose ``File > Import`` and select the export folder path. Alternatively, you
can also export the model in various other formats, such as Bundler, VisualSfM
[#f1]_, PLY, or VRML by choosing ``File > Export as...``. COLMAP can visualize
plain PLY point cloud files with RGB information by choosing ``File > Import
From...``.

For post-processing by dense reconstruction software, such as CMVS/PMVS
[furukawa10]_ or CMP-MVS [jancosek11]_, please choose ``Extras > Undistort
images`` and select the appropriate format. To run CMVS and PMVS2, please run
the following commands::

    ./path/to/cmvs /path/to/undistortion/folder/pmvs/
    ./path/to/genOption /path/to/undistortion/folder/pmvs/
    ./path/to/pmvs2 /path/to/undistortion/folder/pmvs/

where `/path/to/undistortion/folder` is the folder selected in the undistortion
dialog. Make sure not to forget the trailing slash in
`/path/to/undistortion/folder/pmvs/` in the above command-line arguments.


.. _database-management:

Database Management
-------------------

You can review and manage the import cameras, images, and feature matches in the
database management tool. Choose ``Processing > Manage database``. In the
opening dialog, you can see the list of imported images and cameras. You can
view the features and matches for each image by clicking ``Show image`` and
``Show matches``. Note that any changes to the database are only effective after
clicking ``Save``.

To share intrinsic camera parameters between arbitrary groups of images, select
a single or multiple images, choose ``Set camera`` and set the `camera_id`,
which corresponds to the unique `camera_id` column in the cameras table. You can
also add new cameras with specific parameters. By setting the
`prior_focal_length` flag to 0 or 1, you can give a hint whether the
reconstruction algorithm should trust the focal length value. In case of a prior
lab calibration, you want to set this value to 1. Without prior knowledge about
the focal length, it is recommended to set this value to `1.25 *
max(width_in_px, height_in_px)`.

The database management tool has only limited functionality and, for full
control over the data, you must directly modify the SQLite database (see
:ref:`Database Format <database-format>`). By accessing the database directly,
you can use COLMAP only for feature extraction and matching or you can import
your own features and matches to only use COLMAP's incremental reconstruction
algorithm.

.. rubric:: Footnotes

.. [#f1] VisualSfM's projection model applies the distortion to the measurements
    and COLMAP to the projection, hence the exported NVM file is not fully
    compatible with VisualSfM.
