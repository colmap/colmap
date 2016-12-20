Output Format
=============

---------------------
Sparse Reconstruction
---------------------

COLMAP exports three text files for every reconstructed model: `cameras.txt`,
`images.txt`, and `points3D.txt`. Comments start with a leading "#" character.

To export the currently selected model in the GUI, choose ``File > Export
model``. To export all reconstructed models in the current dataset, choose
``File > Export all models``. The selected folder then contains the three files,
and for convenience, the current project configuration for importing the model
to COLMAP. To import the exported models, e.g. for visualization or to resume
the reconstruction, choose ``File > Import model`` and select the folder
containing the `cameras.txt`, `images.txt`, and `points3D.txt` files.

Note that the unique identifiers of cameras (`CAMERA_ID`), images (`IMAGE_ID`),
and 3D points (`POINT3D_ID`) are unordered and are most likely not contiguous.

There are two source files to conveniently read the sparse reconstructions using
Python (``scripts/python/read_model.py``) and Matlab
(``scripts/matlab/read_model.m``).


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

Here, the dataset contains 3 cameras based using different distortion models and
they have the same sensor dimensions (width: 3072, height: 2304). The length of
parameters is variable and depends on the camera model. For the first camera,
there are 3 parameters with a single focal length of 2559.81 pixels and a
principal point at pixel location `(1536, 1152)`. The intrinsic parameters of a
camera can be shared by multiple images, which refer to cameras using the unique
identifier `CAMERA_ID`.


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

Here, there first two lines define the information of the first image, and so
on. The reconstructed pose of an image is specified as the projection from world
to image coordinate system using a quaternion (QW, QX, QY, QZ) and a translation
vector (TX, TY, TZ). Both images use the same camera model and share intrinsics
(`CAMERA_ID = 1`). The image name is relative to the selected base image folder
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

Here, there are three reconstructed 3D points, where `POINT2D_IDX` defines the
zero-based index of the keypoint in the `images.txt` file. The error is given in
pixels of reprojection error and is only updated after global bundle adjustment.


--------------------
Dense Reconstruction
--------------------

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
    +── dense-reconstruction.sh
    +── fused.ply
    +── meshed.ply

Here, the `images` folder contains the undistorted images, the `sparse` folder
contains the sparse reconstruction with undistorted cameras, the `stereo` folder
contains the stereo reconstruction results, `fused.ply` and `meshed.ply` are the
results of the fusion and meshing procedure, and `dense-reconstruction.sh`
contains example command-line usage to perform the dense reconstruction.


Depth and Normal Maps
---------------------

The depth maps are stored as mixed text and binary files. The text header
defines the dimensions of the image in the format ``with&height&channels&``
followed by row-major `float32` binary data. For depth maps ``channels=1`` and
for normal maps ``channels=3``. The depth and normal maps can be conveniently
read with Matlab using the functions in ``scripts/matlab/read_depth_map.m`` and
``scripts/matlab/read_normal_map.m``.


Consistency Graphs
------------------

The consistency graph defines, for all pixels in an image, the source images a
pixel is consistent with. the graph is stored as a continuous list of `int32`
values in the format ``<row><col><N><image_idx1>...<image_idxN>``, where ``(row,
col)``  defines the location of the pixel in the image followed by a list of
``N`` image indices. The indices are specified w.r.t. the ordering in the
``images.txt`` file.
