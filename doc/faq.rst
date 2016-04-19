Frequently Asked Questions
==========================

Camera models
-------------

COLMAP implements different camera models of varying complexity. If no intrinsic
parameters are known a priori, it is generally best to use the simplest camera
model that is complex enough to model the distortion effects:

- *SimplePinholeCameraModel*, *PinholeCameraModel*: Use these camera models, if
  your images are undistorted a priori. Note that even in the case of
  undistorted images, COLMAP could try to improve the intrinsics with a more
  complex camera model.
- *SimpleRadialCameraModel*, *RadialCameraModel*: This should be the camera
  model of choice, if the intrinsics are unknown and every image has a different
  camera calibration, e.g., in the case of Internet photos. Both models are
  simplified versions of the *OpenCVCameraModel* only modeling radial distortion
  effects with one and two parameters, respectively.
- *OpenCVCameraModel*, *FullOpenCVCameraModel*: Use these camera models, if you
  know the calibration parameters a priori. You can also try to let COLMAP
  estimate the parameters, if you share the intrinsics for multiple images. Note
  that the automatic estimation of parameters will most likely fail, if every
  image has a separate set of intrinsic parameters.
- *OpenCVFisheyeCameraModel*: Use this camera model for fisheye lenses and note
  that all other models are not really capable of modeling the distortion
  effects of fisheye lenses.

You can inspect the estimated intrinsic parameters by double-clicking specific
images in the model viewer or by exporting the model and opening the
`cameras.txt` file.

To achieve optimal reconstruction results, you might have to try different
camera models for your problem. Generally, when the reconstruction fails and the
estimated focal length values / distortion coefficients are grossly wrong, it is
a sign of using a too complex camera model. Contrary, if COLMAP uses many
iterative local and global bundle adjustments, it is a sign of using a too
simple camera model that is not able to fully model the distortion effects.

You can also share intrinsics between multiple
images to obtain more reliable results
(see :ref:`Share intrinsic camera parameters <faq-share-intrinsics>`) or you can
fix the intrinsic parameters during the reconstruction
(see :ref:`Fix intrinsic camera parameters <faq-fix-intrinsics>`).


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
