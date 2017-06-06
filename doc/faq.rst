Frequently Asked Questions
==========================

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


Increase number of sparse 3D points
-----------------------------------

By default, COLMAP ignores two-view feature tracks in triangulation, resulting
in fewer 3D points than possible. Triangulation of two-view tracks can in rare
cases improve the stability of sparse image collections by providing additional
constraints in bundle adjustment. To also triangulate two-view tracks, unselect
the option ``Reconstruction > Reconstruction options > Triangulation >
ignore_two_view_tracks``. If your images are taken from far distance with
respect to the scene, you can try to reduce the minimum triangulation angle.


.. _faq-merge-models:

Merge disconnected models
-------------------------

Sometimes COLMAP fails to reconstruct all images into the same model and hence
produces multiple sub-models. If those sub-models have common registered images,
they can be merged into a single model as post-processing step::

    ./src/exe/model_merger \
        --input_path1 /path/to/sub-model1 \
        --input_path2 /path/to/sub-model2 \
        --output_path /path/to/merged-model

To improve the quality of the alignment between the two sub-models, it is
recommended to run another global bundle adjustment after the merge::

    ./src/exe/bundle_adjuster \
        --input_path /path/to/merged-model \
        --output_path /path/to/refined-merged-model


Geo-registration
----------------

Geo-registration of models is possible by providing the 3D locations for the
camera centers of a subset or all registered images. The 3D similarity
transformation between the reconstructed model and the target coordinate frame
of the geo-registration is determined from these correspondences.

The geo-registered 3D coordinates of the camera centers for images must be
specified in a text-file with the following format::

    image_name1.jpg X1 Y1 Z1
    image_name2.jpg X2 Y2 Z2
    image_name3.jpg X3 Y3 Z3
    ...

Note that at least 3 images must be specified to estimate a 3D similarity
transformation. Then, the model can be geo-registered using::

    ./src/exe/model_aligner \
        --input_path /path/to/model \
        --output_path /path/to/geo-registered-model \
        --ref_images_path /path/to/text-file


Manhattan world alignment
-------------------------

COLMAP has functionality to align the coordinate axes of a reconstruction using
a Manhattan world assumption, i.e. COLMAP can automatically determine the
gravity axis and the major horizontal axis of the Manhattan world through
vanishing point detection in the images. Please, refer to the
``model_orientation_aligner`` for more details.


Register/localize new images into an existing reconstruction
------------------------------------------------------------

If you have an existing reconstruction of images and want to register/localize
new images within this reconstruction, you can follow these steps::

    ./src/exe/feature_extractor \
        --database_path $PROJECT_PATH/database.db \
        --image_path $PROJECT_PATH/images \
        --image_list_path /path/to/image-list.txt

    ./src/exe/vocab_tree_matcher \
        --database_path $PROJECT_PATH/database.db \
        --VocabTreeMatching.vocab_tree_path /path/to/vocab-tree.bin \
        --VocabTreeMatching.match_list_path /path/to/image-list.txt

    ./src/exe/image_registrator \
        --database_path $PROJECT_PATH/database.db \
        --image_path $PROJECT_PATH/images \
        --import_path /path/to/existing-model \
        --export_path /path/to/model-with-new-images

Note that this first extracts features for the new images, then matches them to
the existing images in the database, and finally registers them into the model.
The image list text file contains a list of images to extract and match,
specified as one image file name per line.


Multi-GPU support in feature matching
-------------------------------------

You can run feature matching on multiple GPUs by specifying multiple indices for
CUDA-enabled GPUs, e.g., ``--SiftMatching.gpu_index=0,1,2,3`` runs the feature
matching on 4 GPUs in parallel. You can also run multiple feature matching
threads on the same GPU by specifying the same GPU index twice, e.g.,
``--SiftMatching.gpu_index=0,0,1,1,2,3``. By default, COLMAP runs one feature
matching thread per CUDA-enabled GPU.


Feature matching fails due to illegal memory access
---------------------------------------------------

If you encounter the following error message::

    MultiplyDescriptor: an illegal memory access was encountered

during feature matching, your GPU runs out of memory. Try decreasing the option
``--SiftMatching.max_num_matches`` until the error disappears. Note that this
might lead to inferior feature matching results, since the lower-scale input
features will be clamped in order to fit them into GPU memory. Alternatively,
you could change to CPU-based feature matching, but this can become very slow,
or you use a GPU with more memory.


Trading off completeness and accuracy in dense reconstruction
-------------------------------------------------------------

If the dense point cloud contains too many outliers and too much noise, try to
increase the value of option ``--DenseFusion.min_num_pixels``.

If the reconstructed dense surface mesh model contains no surface or there are
too many outlier surfaces, you should reduce the value of option
``--DenseMeshing.trim`` to decrease the surface are and vice versa to increase
it. Also consider to try the reduce the outliers or increase the completeness in
the fusion stage, as described above.


.. _faq-dense-memory:

Reduce memory usage during dense reconstruction
-----------------------------------------------

If you run out of GPU memory during patch match stereo, you can either reduce
the maximum image size by setting the option ``--DenseStereo.max_image_size`` or
reduce the number of source images in the ``stereo/patch-match.cfg`` file from
e.g. ``__auto__, 30`` to ``__auto__, 10``. Note that enabling the
``geom_consistency`` option increases the required GPU memory.

If you run out of CPU memory during stereo or fusion, you can reduce the
``--DenseStereo.cache_size`` or ``--DenseFusion.cache_size`` specified in
gigabytes or you can reduce ``--DenseStereo.max_image_size`` or
``--DenseFusion.max_image_size``. Note that a too low value might lead to very
slow processing and heavy load on the hard disk.

For large-scale reconstructions of several thousands of images, you should
consider splitting your sparse reconstruction into more manageable clusters of
images using e.g. CMVS [furukawa10]_. Note that, for this use case, COLMAP's
dense reconstruction pipeline also supports the PMVS/CMVS folder structure when
executed from the command-line. Please, refer to the workspace folder for
example shell scripts. To change the number of images using CMVS, you must
modify the shell scripts accordingly. For example, ``cmvs pmvs/ 500`` to limit
each cluster to 500 images. Since CMVS produces highly overlapping clusters, it
is recommended to increase the default value of 100 images to as high as
possible according to your available system resources.


Manual specification of source images during dense reconstruction
-----------------------------------------------------------------

You can change the number of source images in the ``stereo/patch-match.cfg``
file from e.g. ``__auto__, 30`` to ``__auto__, 10``. This selects the images
with the most visual overlap automatically as source images. Alternatively, you
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
for CUDA-enabled GPUs, e.g., ``--DenseStereo.gpu_index=0,1,2,3`` runs the dense
reconstruction on 4 GPUs in parallel. You can also run multiple dense
reconstruction threads on the same GPU by specifying the same GPU index twice,
e.g., ``--SiftMatching.gpu_index=0,0,1,1,2,3``. By default, COLMAP runs one
dense reconstruction thread per CUDA-enabled GPU.


.. _faq-dense-timeout:

Fix GPU freezes and timeouts during dense reconstruction
--------------------------------------------------------

The stereo reconstruction pipeline runs on the GPU using CUDA and puts the GPU
under heavy load. You might experience a display freeze or even a program crash
during the reconstruction. As a solution to this problem, you could use a
secondary GPU in your system, that is not connected to your display.
Alternatively, you can increase the GPU timeouts of your system, as detailed in
the following.

By default, the Windows operating system detects response problems from the GPU,
and recovers to a functional desktop by resetting the card and aborting the
stereo reconstruction process. The solution is to increase the so-called
"Timeout Detection & Recovery" (TDR) delay to a larger value. Please, refer to
the `NVIDIA Nsight documentation <https://goo.gl/UWKVs6>`_ or to the `Microsoft
documentation <http://www.microsoft.com/whdc/device/display/wddm_timeout.mspx>`_
on how to increase the delay time under Windows.

The X window system under Linux/Unix has a similar feature and detects response
problems of the GPU. The easiest solution to avoid timeout problems under the X
window system is to shut it down and run the stereo reconstruction from the
command-line. Under Ubuntu, you could first stop X using::

    sudo service lightdm stop

And then run the dense reconstruction code from the command-line::

    ./src/exe/dense_stereo ...

Finally, you can restart your desktop environment with the following command::

    sudo service lightdm start
