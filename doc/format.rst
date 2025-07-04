.. _output-format:

Output Format
=============

==================
Binary File Format
==================

Note that all binary data is stored using little endian byte ordering. All x86
processors are little endian and thus no special care has to be taken when
reading COLMAP binary data on most platforms. The data can be most conveniently
parsed using the C++ reconstruction API under ``src/colmap/scene/reconstruction_io.h``
or using the Python API provided by pycolmap.


=======================
Indices and Identifiers
=======================

Any variable name ending with ``*_idx`` should be considered as an ordered,
contiguous zero-based index. In general, any variable name ending with ``*_id``
should be considered as an unordered, non-contiguous identifier.

For example, the unique identifiers of cameras (``CAMERA_ID``), images
(``IMAGE_ID``), and 3D points (``POINT3D_ID``) are unordered and are most likely not
contiguous. This also means that the maximum ``POINT3D_ID`` does not necessarily
correspond to the number 3D points, since some ``POINT3D_ID``'s are missing due to
filtering during the reconstruction, etc.


=====================
Sparse Reconstruction
=====================

By default, COLMAP uses a binary file format (machine-readable, fast) for
storing sparse models. In addition, COLMAP provides the option to store the
sparse models as text files (human-readable, slow). In both cases, the
information is split into multiples files for the information about ``rigs``,
``cameras``, ``frames``, ``images``, and ``points``. Any directory containing these
files constitutes a sparse model. The binary files have the file extension
``.bin`` and the text files the file extension ``.txt``. Note that when loading a
model from a directory which contains both binary and text files, COLMAP prefers
the binary format.

Note that older versions of COLMAP had no rig support and thus the ``rigs`` and
``frames`` files may be missing. The reconstruction I/O routines in COLMAP are
fully backwards compatible in that models without these files can be read and
trivial rigs and frames will be automatically initialized. Furthermore, newer
output reconstructions' ``cameras`` and ``images`` files are fully compatible with
old outputs.

To export the currently selected model in the GUI, choose ``File > Export
model``. To export all reconstructed models in the current dataset, choose
``File > Export all``. The selected folder then contains the three files, and
for convenience, the current project configuration for importing the model to
COLMAP. To import the exported models, e.g., for visualization or to resume the
reconstruction, choose ``File > Import model`` and select the folder containing
the ``cameras``, ``images``, and ``points3D`` files.

To convert between the binary and text format in the GUI, you can load the model
using ``File > Import model`` and then export the model in the desired output
format using ``File > Export model`` (binary) or ``File > Export model as text``
(text). In addition, you can export sparse models to other formats, such as
VisualSfM's NVM, Bundler files, PLY, VRML, etc., using ``File > Export as...``.
To convert between various formats from the CLI, use the ``model_converter``
executable.

There are two source files to conveniently read the sparse reconstructions using
Python (``scripts/python/read_write_model.py`` supporting binary and text) and Matlab
(``scripts/matlab/read_model.m`` supporting text).


-----------
Text Format
-----------

COLMAP exports the following three text files for every reconstructed model:
``rigs.txt``, ``cameras.txt``, ``frames.txt``, ``images.txt``, and ``points3D.txt``.
Comments start with a leading "#" character and are ignored. The first comment
lines briefly describe the format of the text files, as described in more
detailed on this page.


rigs.txt
-----------

This file contains the configured rigs and sensors, e.g.::

    # Rig calib list with one line of data per calib:
    #   RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, SENSORS[] as (SENSOR_TYPE, SENSOR_ID, HAS_POSE, [QW, QX, QY, QZ, TX, TY, TZ])
    # Number of rigs: 1
    1 2 CAMERA 1 CAMERA 2 1 -0.9999701516465348 -0.0011120266840749639 -0.0075347911527510894 0.0012985125893421306 -0.19316906391350164 0.00085222218993398979 0.0070758955539026785
    2 1 CAMERA 3

Here, the dataset contains two rigs: the first rig has two cameras and the second
one has 1 camera.


cameras.txt
-----------

This file contains the intrinsic parameters of all reconstructed cameras in the
dataset using one line per camera, e.g.::

    # Camera list with one line of data per camera:
    #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    # Number of cameras: 3
    1 SIMPLE_PINHOLE 3072 2304 2559.81 1536 1152
    2 PINHOLE 3072 2304 2560.56 2560.56 1536 1152
    3 SIMPLE_RADIAL 3072 2304 2559.69 1536 1152 -0.0218531

Here, the dataset contains 3 cameras based using different distortion models
with the same sensor dimensions (width: 3072, height: 2304). The length of
parameters is variable and depends on the camera model. For the first camera,
there are 3 parameters with a single focal length of 2559.81 pixels and a
principal point at pixel location ``(1536, 1152)``. The intrinsic parameters of a
camera can be shared by multiple images, which refer to cameras using the unique
identifier ``CAMERA_ID``.


frames.txt
----------

This file contains the frames, where a frame defines a specific
instance of a rig with all or a subset of sensors exposed at the same time, e.g.::

    # Frame list with one line of data per frame:
    #   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], NUM_DATA_IDS, DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)
    # Number of frames: 151
    1 1 0.99801363919752195 0.040985139360073107 0.041890917712361225 -0.023111584553400576 -5.2666546897987896 -0.17120007823690631 0.12300519697527648 2 CAMERA 1 1 CAMERA 2 2
    2 2 0.99816472047267968 0.037605501383281774 0.043101511724657163 -0.019881568259519072 -5.1956060695789192 -0.20794508616745555 0.14967533910764824 1 CAMERA 3 3

Here, the dataset contains two frames, where frame 1 is an instance of rig 1 and
frame 2 an instance of rig 2. 


images.txt
----------

This file contains the pose and keypoints of all reconstructed images in the
dataset using two lines per image, e.g.::

    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    # Number of images: 2, mean observations per image: 2
    1 0.851773 0.0165051 0.503764 -0.142941 -0.737434 1.02973 3.74354 1 P1180141.JPG
    2362.39 248.498 58396 1784.7 268.254 59027 1784.7 268.254 -1
    2 0.851773 0.0165051 0.503764 -0.142941 -0.737434 1.02973 3.74354 1 P1180142.JPG
    1190.83 663.957 23056 1258.77 640.354 59070

Here, the first two lines define the information of the first image, and so on.
The reconstructed pose of an image is specified as the projection from world to
the camera coordinate system of an image using a quaternion ``(QW, QX, QY, QZ)``
and a translation vector ``(TX, TY, TZ)``. The quaternion is defined using the
Hamilton convention, which is, for example, also used by the Eigen library. The
coordinates of the projection/camera center are given by ``-R^t * T``, where
``R^t`` is the inverse/transpose of the 3x3 rotation matrix composed from the
quaternion and ``T`` is the translation vector. The local camera coordinate
system of an image is defined in a way that the X axis points to the right, the
Y axis to the bottom, and the Z axis to the front as seen from the image.

Both images in the example above use the same camera model and share intrinsics
(``CAMERA_ID = 1``). The image name is relative to the selected base image folder
of the project. The first image has 3 keypoints and the second image has 2
keypoints, while the location of the keypoints is specified in pixel
coordinates. Both images observe 2 3D points and note that the last keypoint of
the first image does not observe a 3D point in the reconstruction as the 3D
point identifier is -1.


points3D.txt
------------

This file contains the information of all reconstructed 3D points in the
dataset using one line per point, e.g.::

    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    # Number of points: 3, mean track length: 3.3334
    63390 1.67241 0.292931 0.609726 115 121 122 1.33927 16 6542 15 7345 6 6714 14 7227
    63376 2.01848 0.108877 -0.0260841 102 209 250 1.73449 16 6519 15 7322 14 7212 8 3991
    63371 1.71102 0.28566 0.53475 245 251 249 0.612829 118 4140 117 4473

Here, there are three reconstructed 3D points, where ``POINT2D_IDX`` defines the
zero-based index of the keypoint in the ``images.txt`` file. The error is given in
pixels of reprojection error and is only updated after global bundle adjustment.


====================
Dense Reconstruction
====================

COLMAP uses the following workspace folder structure::

    +── images
    │   +── image1.jpg
    │   +── image2.jpg
    │   +── ...
    +── sparse
    │   +── cameras.txt
    │   +── images.txt
    │   +── points3D.txt
    +── stereo
    │   +── consistency_graphs
    │   │   +── image1.jpg.photometric.bin
    │   │   +── image2.jpg.photometric.bin
    │   │   +── ...
    │   +── depth_maps
    │   │   +── image1.jpg.photometric.bin
    │   │   +── image2.jpg.photometric.bin
    │   │   +── ...
    │   +── normal_maps
    │   │   +── image1.jpg.photometric.bin
    │   │   +── image2.jpg.photometric.bin
    │   │   +── ...
    │   +── patch-match.cfg
    │   +── fusion.cfg
    +── fused.ply
    +── meshed-poisson.ply
    +── meshed-delaunay.ply
    +── run-colmap-geometric.sh
    +── run-colmap-photometric.sh

Here, the ``images`` folder contains the undistorted images, the ``sparse`` folder
contains the sparse reconstruction with undistorted cameras, the ``stereo`` folder
contains the stereo reconstruction results, ``point-cloud.ply`` and ``mesh.ply`` are
the results of the fusion and meshing procedure, and ``run-colmap-geometric.sh``
and ``run-colmap-photometric.sh`` contain example command-line usage to perform
the dense reconstruction.


---------------------
Depth and Normal Maps
---------------------

The depth maps are stored as mixed text and binary files. The text header
defines the dimensions of the image in the format ``with&height&channels&``
followed by row-major ``float32`` binary data. For depth maps ``channels=1`` and
for normal maps ``channels=3``. The depth and normal maps can be conveniently
read with Python using the functions in ``scripts/python/read_dense.py`` and
with Matlab using the functions in ``scripts/matlab/read_depth_map.m`` and
``scripts/matlab/read_normal_map.m``.


------------------
Consistency Graphs
------------------

The consistency graph defines, for all pixels in an image, the source images a
pixel is consistent with. The graph is stored as a mixed text and binary file,
while the text part is equivalent to the depth and normal maps and the binary
part is a continuous list of ``int32`` values in the format
``<row><col><N><image_idx1>...<image_idxN>``. Here, ``(row, col)``  defines the
location of the pixel in the image followed by a list of ``N`` image indices.
The indices are specified w.r.t. the ordering in the ``images.txt`` file.
