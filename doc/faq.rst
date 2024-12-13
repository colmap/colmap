Frequently Asked Questions
==========================

Adjusting the options for different reconstruction scenarios and output quality
-------------------------------------------------------------------------------

COLMAP provides many options that can be tuned for different reconstruction
scenarios and to trade off accuracy and completeness versus efficiency. The
default options are set to for medium to high quality reconstruction of
unstructured input data. There are several presets for different scenarios and
quality levels, which can be set in the GUI as ``Extras > Set options for ...``.
To use these presets from the command-line, you can save the current set of
options as ``File > Save project`` after choosing the presets. The resulting
project file can be opened with a text editor to view the different options.
Alternatively, you can generate the project file also from the command-line
by running ``colmap project_generator``.


Extending COLMAP
----------------

If you need to simply analyze the produced sparse or dense reconstructions from
COLMAP, you can load the sparse models in Python and Matlab using the provided
scripts in ``scripts/python`` and ``scripts/matlab``.

If you want to write a C/C++ executable that builds on top of COLMAP, there are
two possible approaches. First, the COLMAP headers and library are installed
to the ``CMAKE_INSTALL_PREFIX`` by default. Compiling against COLMAP as a
library is described :ref:`here <installation-library>`. Alternatively, you can
start from the ``src/tools/example.cc`` code template and implement the desired
functionality directly as a new binary within COLMAP.


.. _faq-share-intrinsics:

Share intrinsics
----------------

COLMAP supports shared intrinsics for arbitrary groups of images and camera
models. Images share the same intrinsics, if they refer to the same camera, as
specified by the `camera_id` property in the database. You can add new cameras
and set shared intrinsics in the database management tool. Please, refer to
:ref:`Database Management <database-management>` for more information.


.. _faq-fix-intrinsics:

Fix intrinsics
--------------

By default, COLMAP tries to refine the intrinsic camera parameters (except
principal point) automatically during the reconstruction. Usually, if there are
enough images in the dataset and you share the intrinsics between multiple
images, the estimated intrinsic camera parameters in SfM should be better than
parameters manually obtained with a calibration pattern.

However, sometimes COLMAP's self-calibration routine might converge in
degenerate parameters, especially in case of the more complex camera models with
many distortion parameters. If you know the calibration parameters a priori, you
can fix different parameter groups during the reconstruction. Choose
``Reconstruction > Reconstruction options > Bundle Adj. > refine_*`` and check
which parameter group to refine or to keep constant. Even if you keep the
parameters constant during the reconstruction, you can refine the parameters in
a final global bundle adjustment by setting ``Reconstruction > Bundle adj.
options > refine_*`` and then running ``Reconstruction > Bundle adjustment``.


Principal point refinement
--------------------------

By default, COLMAP keeps the principal point constant during the reconstruction,
as principal point estimation is an ill-posed problem in general. Once all
images are reconstructed, the problem is most often constrained enough that you
can try to refine the principal point in global bundle adjustment, especially
when sharing intrinsic parameters between multiple images. Please, refer to
:ref:`Fix intrinsics <faq-fix-intrinsics>` for more information.


Increase number of matches / sparse 3D points
---------------------------------------------

To increase the number of matches, you should use the more discriminative
DSP-SIFT features instead of plain SIFT and also estimate the affine feature
shape using the options: ``--SiftExtraction.estimate_affine_shape=true`` and
``--SiftExtraction.domain_size_pooling=true``. In addition, you should enable
guided feature matching using: ``--SiftMatching.guided_matching=true``.

By default, COLMAP ignores two-view feature tracks in triangulation, resulting
in fewer 3D points than possible. Triangulation of two-view tracks can in rare
cases improve the stability of sparse image collections by providing additional
constraints in bundle adjustment. To also triangulate two-view tracks, unselect
the option ``Reconstruction > Reconstruction options > Triangulation >
ignore_two_view_tracks``. If your images are taken from far distance with
respect to the scene, you can try to reduce the minimum triangulation angle.


Reconstruct sparse/dense model from known camera poses
------------------------------------------------------

If the camera poses are known and you want to reconstruct a sparse or dense
model of the scene, you must first manually construct a sparse model by creating
a ``cameras.txt``, ``points3D.txt``, and ``images.txt`` under a new folder::

    +── path/to/manually/created/sparse/model
    │   +── cameras.txt
    │   +── images.txt
    │   +── points3D.txt

The ``points3D.txt`` file should be empty while every other line in the ``images.txt``
should also be empty, since the sparse features are computed, as described below. You can
refer to :ref:`this article <output-format>` for more information about the structure of
a sparse model.

Example of images.txt::

    1 0.695104 0.718385 -0.024566 0.012285 -0.046895 0.005253 -0.199664 1 image0001.png
    # Make sure every other line is left empty
    2 0.696445 0.717090 -0.023185 0.014441 -0.041213 0.001928 -0.134851 2 image0002.png

    3 0.697457 0.715925 -0.025383 0.018967 -0.054056 0.008579 -0.378221 1 image0003.png

    4 0.698777 0.714625 -0.023996 0.021129 -0.048184 0.004529 -0.313427 2 image0004.png

Each image above must have the same ``image_id`` (first column) as in the database (next step). 
This database can be inspected either in the GUI (under ``Database management > Processing``),
or, one can create a reconstruction with colmap and later export  it as text in order to see
the images.txt file it creates.

To reconstruct a sparse map, you first have to recompute features from the
images of the known camera poses as follows::

    colmap feature_extractor \
        --database_path $PROJECT_PATH/database.db \
        --image_path $PROJECT_PATH/images

If your known camera intrinsics have large distortion coefficients, you should
now manually copy the parameters from your ``cameras.txt`` to the database, such
that the matcher can leverage the intrinsics. Modifying the database is possible
in many ways, but an easy option is to use the provided
``scripts/python/database.py`` script. Otherwise, you can skip this step and
simply continue as follows::

    colmap exhaustive_matcher \ # or alternatively any other matcher
        --database_path $PROJECT_PATH/database.db

    colmap point_triangulator \
        --database_path $PROJECT_PATH/database.db \
        --image_path $PROJECT_PATH/images
        --input_path path/to/manually/created/sparse/model \
        --output_path path/to/triangulated/sparse/model

Note that the sparse reconstruction step is not necessary in order to compute
a dense model from known camera poses. Assuming you computed a sparse model
from the known camera poses, you can compute a dense model as follows::

    colmap image_undistorter \
        --image_path $PROJECT_PATH/images \
        --input_path path/to/triangulated/sparse/model \
        --output_path path/to/dense/workspace

    colmap patch_match_stereo \
        --workspace_path path/to/dense/workspace

    colmap stereo_fusion \
        --workspace_path path/to/dense/workspace \
        --output_path path/to/dense/workspace/fused.ply

Alternatively, you can also produce a dense model without a sparse model as::

    colmap image_undistorter \
        --image_path $PROJECT_PATH/images \
        --input_path path/to/manually/created/sparse/model \
        --output_path path/to/dense/workspace

Since the sparse point cloud is used to automatically select neighboring images
during the dense stereo stage, you have to manually specify the source images,
as described :ref:`here <faq-dense-manual-source>`. The dense stereo stage
now also requires a manual specification of the depth range::

    colmap patch_match_stereo \
        --workspace_path path/to/dense/workspace \
        --PatchMatchStereo.depth_min $MIN_DEPTH \
        --PatchMatchStereo.depth_max $MAX_DEPTH

    colmap stereo_fusion \
        --workspace_path path/to/dense/workspace \
        --output_path path/to/dense/workspace/fused.ply


.. _faq-merge-models:

Merge disconnected models
-------------------------

Sometimes COLMAP fails to reconstruct all images into the same model and hence
produces multiple sub-models. If those sub-models have common registered images,
they can be merged into a single model as post-processing step::

    colmap model_merger \
        --input_path1 /path/to/sub-model1 \
        --input_path2 /path/to/sub-model2 \
        --output_path /path/to/merged-model

To improve the quality of the alignment between the two sub-models, it is
recommended to run another global bundle adjustment after the merge::

    colmap bundle_adjuster \
        --input_path /path/to/merged-model \
        --output_path /path/to/refined-merged-model


Geo-registration
----------------

Geo-registration of models is possible by providing the 3D locations for the
camera centers of a subset or all registered images. The 3D similarity
transformation between the reconstructed model and the target coordinate frame
of the geo-registration is determined from these correspondences.

The geo-registered 3D coordinates can either be extracted from the database 
(tvec_prior field) or from a user specified text file. 
For text-files, the geo-registered 3D coordinates of the camera centers for 
images must be specified with the following format::

    image_name1.jpg X1 Y1 Z1
    image_name2.jpg X2 Y2 Z2
    image_name3.jpg X3 Y3 Z3
    ...

The coordinates can be either GPS-based (lat/lon/alt) or cartesian-based (x/y/z).
In case of GPS coordinates, a conversion will be performed to turn those into
cartesian coordinates.  The conversion can be done from GPS to ECEF
(Earth-Centered-Earth-Fixed) or to ENU (East-North-Up) coordinates. If ENU coordinates
are used, the first image GPS coordinates will define the origin of the ENU frame.
It is also possible to use ECEF coordinates for alignment and then rotate the aligned 
reconstruction into the ENU plane.

Note that at least 3 images must be specified to estimate a 3D similarity
transformation. Then, the model can be geo-registered using::

    colmap model_aligner \
        --input_path /path/to/model \
        --output_path /path/to/geo-registered-model \
        --ref_images_path /path/to/text-file (or --database_path /path/to/database.db) \
        --ref_is_gps 1 \
        --alignment_type ecef \
        --alignment_max_error 3.0 (where 3.0 is the error threshold to be used in RANSAC)

A 3D similarity transformation will be estimated with a RANSAC estimator to be robust to potential outliers
in the data. It is required to provide the error threshold to be used in the RANSAC estimator.

Manhattan world alignment
-------------------------

COLMAP has functionality to align the coordinate axes of a reconstruction using
a Manhattan world assumption, i.e. COLMAP can automatically determine the
gravity axis and the major horizontal axis of the Manhattan world through
vanishing point detection in the images. Please, refer to the
``model_orientation_aligner`` for more details.


Mask image regions
------------------

COLMAP supports masking of keypoints during feature extraction two different ways:

1. Passing ``mask_path`` to a folder with image masks. For a given image, the corresponding
mask must have the same sub-path below this root as the image has below
``image_path``. The filename must be equal, aside from the added extension
``.png``. For example, for an image ``image_path/abc/012.jpg``, the mask would
be ``mask_path/abc/012.jpg.png``.

2. Passing ``camera_mask_path`` to a single mask image. This single mask is applied to all images.

In both cases no features will be extracted in regions,
where the mask image is black (pixel intensity value 0 in grayscale).



Register/localize new images into an existing reconstruction
------------------------------------------------------------

If you have an existing reconstruction of images and want to register/localize
new images within this reconstruction, you can follow these steps::

    colmap feature_extractor \
        --database_path $PROJECT_PATH/database.db \
        --image_path $PROJECT_PATH/images \
        --image_list_path /path/to/image-list.txt

    colmap vocab_tree_matcher \
        --database_path $PROJECT_PATH/database.db \
        --VocabTreeMatching.match_list_path /path/to/image-list.txt

    colmap image_registrator \
        --database_path $PROJECT_PATH/database.db \
        --input_path /path/to/existing-model \
        --output_path /path/to/model-with-new-images

    colmap bundle_adjuster \
        --input_path /path/to/model-with-new-images \
        --output_path /path/to/model-with-new-images

Note that this first extracts features for the new images, then matches them to
the existing images in the database, and finally registers them into the model.
The image list text file contains a list of images to extract and match,
specified as one image file name per line. The bundle adjustment is optional.

If you need a more accurate image registration with triangulation, then you
should restart or continue the reconstruction process rather than just
registering the images to the model. Instead of running the
``image_registrator``, you should run the ``mapper`` to continue the
reconstruction process from the existing model::

    colmap mapper \
        --database_path $PROJECT_PATH/database.db \
        --image_path $PROJECT_PATH/images \
        --input_path /path/to/existing-model \
        --output_path /path/to/model-with-new-images

Or, alternatively, you can start the reconstruction from scratch::

    colmap mapper \
        --database_path $PROJECT_PATH/database.db \
        --image_path $PROJECT_PATH/images \
        --output_path /path/to/model-with-new-images

Note that dense reconstruction must be re-run from scratch after running the
``mapper`` or the ``bundle_adjuster``, as the coordinate frame of the model can
change during these steps.


Available functionality without GPU/CUDA
----------------------------------------

If you do not have a CUDA-enabled GPU but some other GPU, you can use all COLMAP
functionality except the dense reconstruction part. However, you can use
external dense reconstruction software as an alternative, as described in the
:ref:`Tutorial <dense-reconstruction>`. If you have a GPU with low compute power
or you want to execute COLMAP on a machine without an attached display and
without CUDA support, you can run all steps on the CPU by specifying the
appropriate options (e.g., ``--SiftExtraction.use_gpu=false`` for the feature
extraction step). But note that this might result in a significant slow-down of
the reconstruction pipeline. Please, also note that feature extraction on the
CPU can consume excessive RAM for large images in the default settings, which
might require manually reducing the maximum image size using
``--SiftExtraction.max_image_size`` and/or setting
``--SiftExtraction.first_octave 0`` or by manually limiting the number of
threads using ``--SiftExtraction.num_threads``.


Multi-GPU support in feature extraction/matching
------------------------------------------------

You can run feature extraction/matching on multiple GPUs by specifying multiple
indices for CUDA-enabled GPUs, e.g., ``--SiftExtraction.gpu_index=0,1,2,3`` and
``--SiftMatching.gpu_index=0,1,2,3`` runs the feature extraction/matching on 4
GPUs in parallel. Note that you can only run one thread per GPU and this
typically also gives the best performance. By default, COLMAP runs one feature
extraction/matching thread per CUDA-enabled GPU and this usually gives the best
performance as compared to running multiple threads on the same GPU.


Feature matching fails due to illegal memory access
---------------------------------------------------

If you encounter the following error message::

    MultiplyDescriptor: an illegal memory access was encountered

or the following:

    ERROR: Feature matching failed. This probably caused by insufficient GPU
           memory. Consider reducing the maximum number of features.

during feature matching, your GPU runs out of memory. Try decreasing the option
``--SiftMatching.max_num_matches`` until the error disappears. Note that this
might lead to inferior feature matching results, since the lower-scale input
features will be clamped in order to fit them into GPU memory. Alternatively,
you could change to CPU-based feature matching, but this can become very slow,
or better you buy a GPU with more memory.

The maximum required GPU memory can be approximately estimated using the
following formula: ``4 * num_matches * num_matches + 4 * num_matches * 256``.
For example, if you set ``--SiftMatching.max_num_matches 10000``, the maximum
required GPU memory will be around 400MB, which are only allocated if one of
your images actually has that many features.


Trading off completeness and accuracy in dense reconstruction
-------------------------------------------------------------

If the dense point cloud contains too many outliers and too much noise, try to
increase the value of option ``--StereoFusion.min_num_pixels``.

If the reconstructed dense surface mesh model using Poisson reconstruction
contains no surface or there are too many outlier surfaces, you should reduce
the value of option ``--PoissonMeshing.trim`` to decrease the surface are and
vice versa to increase it. Also consider to try the reduce the outliers or
increase the completeness in the fusion stage, as described above.

If the reconstructed dense surface mesh model using Delaunay reconstruction
contains too noisy or incomplete surfaces, you should increase the
``--DenaunayMeshing.quality_regularization`` parameter to obtain a smoother
surface. If the resolution of the mesh is too coarse, you should reduce the
``--DelaunayMeshing.max_proj_dist`` option to a lower value.


Improving dense reconstruction results for weakly textured surfaces
-------------------------------------------------------------------

For scenes with weakly textured surfaces it can help to have a high resolution
of the input images (``--PatchMatchStereo.max_image_size``) and a large patch window
radius (``--PatchMatchStereo.window_radius``). You may also want to reduce the
filtering threshold for the photometric consistency cost
(``--PatchMatchStereo.filter_min_ncc``).


Surface mesh reconstruction
---------------------------

COLMAP supports two types of surface reconstruction algorithms. Poisson surface
reconstruction [kazhdan2013]_ and graph-cut based surface extraction from a
Delaunay triangulation. Poisson surface reconstruction typically requires an
almost outlier-free input point cloud and it often produces bad surfaces in the
presence of outliers or large holes in the input data. The Delaunay
triangulation based meshing algorithm is more robust to outliers and in general
more scalable to large datasets than the Poisson algorithm, but it usually
produces less smooth surfaces. Furthermore, the Delaunay based meshing can be
applied to sparse and dense reconstruction results. To increase the smoothness
of the surface as a post-processing step, you could use Laplacian smoothing, as
e.g. implemented in Meshlab.

Note that the two algorithms can also be combined by first running the Delaunay
meshing to robustly filter outliers from the sparse or dense point cloud and
then, in the second step, performing Poisson surface reconstruction to obtain a
smooth surface.


Speedup dense reconstruction
----------------------------

The dense reconstruction can be speeded up in multiple ways:

- Put more GPUs in your system as the dense reconstruction can make use of
  multiple GPUs during the stereo reconstruction step. Put more RAM into your
  system and increase the ``--PatchMatchStereo.cache_size``,
  ``--StereoFusion.cache_size`` to the largest possible value in order to
  speed up the dense fusion step.

- Do not perform geometric dense stereo reconstruction
  ``--PatchMatchStereo.geom_consistency false``. Make sure to also enable
  ``--PatchMatchStereo.filter true`` in this case.

- Reduce the ``--PatchMatchStereo.max_image_size``, ``--StereoFusion.max_image_size``
  values to perform dense reconstruction on a maximum image resolution.

- Reduce the number of source images per reference image to be considered, as
  described :ref:`here <faq-dense-memory>`.

- Increase the patch windows step ``--PatchMatchStereo.window_step`` to 2.

- Reduce the patch window radius ``--PatchMatchStereo.window_radius``.

- Reduce the number of patch match iterations ``--PatchMatchStereo.num_iterations``.

- Reduce the number of sampled views ``--PatchMatchStereo.num_samples``.

- To speedup the dense stereo and fusion step for very large reconstructions,
  you can use CMVS to partition your scene into multiple clusters and to prune
  redundant images, as described :ref:`here <faq-dense-memory>`.

Note that apart from upgrading your hardware, the proposed changes might degrade
the quality of the dense reconstruction results. When canceling the stereo
reconstruction process and restarting it later, the previous progress is not
lost and any already processed views will be skipped.


.. _faq-dense-memory:

Reduce memory usage during dense reconstruction
-----------------------------------------------

If you run out of GPU memory during patch match stereo, you can either reduce
the maximum image size by setting the option ``--PatchMatchStereo.max_image_size`` or
reduce the number of source images in the ``stereo/patch-match.cfg`` file from
e.g. ``__auto__, 30`` to ``__auto__, 10``. Note that enabling the
``geom_consistency`` option increases the required GPU memory.

If you run out of CPU memory during stereo or fusion, you can reduce the
``--PatchMatchStereo.cache_size`` or ``--StereoFusion.cache_size`` specified in
gigabytes or you can reduce ``--PatchMatchStereo.max_image_size`` or
``--StereoFusion.max_image_size``. Note that a too low value might lead to very
slow processing and heavy load on the hard disk.

For large-scale reconstructions of several thousands of images, you should
consider splitting your sparse reconstruction into more manageable clusters of
images using e.g. CMVS [furukawa10]_. In addition, CMVS allows to prune
redundant images observing the same scene elements. Note that, for this use
case, COLMAP's dense reconstruction pipeline also supports the PMVS/CMVS folder
structure when executed from the command-line. Please, refer to the workspace
folder for example shell scripts. Note that the example shell scripts for
PMVS/CMVS are only generated, if the output type is set to PMVS. Since CMVS
produces highly overlapping clusters, it is recommended to increase the default
value of 100 images per cluster to as high as possible according to your
available system resources and speed requirements. To change the number of
images using CMVS, you must modify the shell scripts accordingly. For example,
``cmvs pmvs/ 500`` to limit each cluster to 500 images. If you want to use CMVS
to prune redundant images but not to cluster the scene, you can simply set this
number to a very large value.


.. _faq-dense-manual-source:

Manual specification of source images during dense reconstruction
-----------------------------------------------------------------

You can change the number of source images in the ``stereo/patch-match.cfg``
file from e.g. ``__auto__, 30`` to ``__auto__, 10``. This selects the images
with the most visual overlap automatically as source images. You can also use
all other images as source images, by specifying ``__all__``. Alternatively, you
can manually specify images with their name, for example::

    image1.jpg
    image2.jpg, image3.jpg
    image2.jpg
    image1.jpg, image3.jpg
    image3.jpg
    image1.jpg, image2.jpg

Here, ``image2.jpg`` and ``image3.jpg`` are used as source images for
``image1.jpg``, etc.


Multi-GPU support in dense reconstruction
-----------------------------------------

You can run dense reconstruction on multiple GPUs by specifying multiple indices
for CUDA-enabled GPUs, e.g., ``--PatchMatchStereo.gpu_index=0,1,2,3`` runs the dense
reconstruction on 4 GPUs in parallel. You can also run multiple dense
reconstruction threads on the same GPU by specifying the same GPU index twice,
e.g., ``--PatchMatchStereo.gpu_index=0,0,1,1,2,3``. By default, COLMAP runs one
dense reconstruction thread per CUDA-enabled GPU.


.. _faq-dense-timeout:

Fix GPU freezes and timeouts during dense reconstruction
--------------------------------------------------------

The stereo reconstruction pipeline runs on the GPU using CUDA and puts the GPU
under heavy load. You might experience a display freeze or even a program crash
during the reconstruction. As a solution to this problem, you could use a
secondary GPU in your system, that is not connected to your display by setting
the GPU indices explicitly (usually index 0 corresponds to the card that the
display is attached to). Alternatively, you can increase the GPU timeouts of
your system, as detailed in the following.

By default, the Windows operating system detects response problems from the GPU,
and recovers to a functional desktop by resetting the card and aborting the
stereo reconstruction process. The solution is to increase the so-called
"Timeout Detection & Recovery" (TDR) delay to a larger value. Please, refer to
the `NVIDIA Nsight documentation <https://goo.gl/UWKVs6>`_ or to the `Microsoft
documentation <http://www.microsoft.com/whdc/device/display/wddm_timeout.mspx>`_
on how to increase the delay time under Windows. You can increase the delay
using the following Windows Registry entries::

    [HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers]
    "TdrLevel"=dword:00000001
    "TdrDelay"=dword:00000120

To set the registry entries, execute the following commands using administrator
privileges (e.g., in ``cmd.exe`` or ``powershell.exe``)::

    reg add HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers /v TdrLevel /t REG_DWORD /d 00000001
    reg add HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers /v TdrDelay /t REG_DWORD /d 00000120

and restart your machine afterwards to make the changes effective.

The X window system under Linux/Unix has a similar feature and detects response
problems of the GPU. The easiest solution to avoid timeout problems under the X
window system is to shut it down and run the stereo reconstruction from the
command-line. Under Ubuntu, you could first stop X using::

    sudo service lightdm stop

And then run the dense reconstruction code from the command-line::

    colmap patch_match_stereo ...

Finally, you can restart your desktop environment with the following command::

    sudo service lightdm start

If the dense reconstruction still crashes after these changes, the reason is
probably insufficient GPU memory, as discussed in a separate item in this list.
