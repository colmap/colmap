Camera Models
=============

COLMAP implements different camera models of varying complexity. If no intrinsic
parameters are known a priori, it is generally best to use the simplest camera
model that is complex enough to model the distortion effects:

- ``SIMPLE_PINHOLE``, ``PINHOLE``: Use these camera models,
  if your images are undistorted a priori. These use one and two focal length
  parameters, respectively. Note that even in the case of undistorted images, 
  COLMAP could try to improve the intrinsics with a more complex camera model.
- ``SIMPLE_RADIAL``, ``RADIAL``: This should be the camera model of choice,
  if the intrinsics are unknown and every image has a different camera
  calibration, e.g., in the case of Internet photos. Both models are simplified
  versions of the ``OPENCV`` model only modeling radial distortion
  effects with one and two parameters, respectively.
- ``OPENCV``, ``FULL_OPENCV``: Use these camera models, if
  you know the calibration parameters a priori. You can also try to let COLMAP
  estimate the parameters, if you share the intrinsics for multiple images. Note
  that the automatic estimation of parameters will most likely fail, if every
  image has a separate set of intrinsic parameters.
- ``SIMPLE_RADIAL_FISHEYE``, ``RADIAL_FISHEYE``, ``OPENCV_FISHEYE``, ``FOV``,
  ``THIN_PRISM_FISHEYE``: Use these camera models for fisheye lenses
  and note that all other models are not really capable of modeling the
  distortion effects of fisheye lenses. The ``FOV`` model is used by
  Google Project Tango (make sure to not initialize `omega` to zero).

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

Please, refer to the camera models header file for information on the parameters
of the different camera models:
https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h
