Camera Models
=============

COLMAP implements different camera models of varying complexity. If no intrinsic
parameters are known a priori, it is generally best to use the simplest camera
model that is complex enough to model the distortion effects:

- ``SIMPLE_PINHOLE``, ``PINHOLE``: Use these camera models, if your images are
  undistorted a priori. These use one and two focal length parameters,
  respectively. Note that even in the case of undistorted images, COLMAP could
  try to improve the intrinsics with a more complex camera model.
- ``SIMPLE_RADIAL``, ``RADIAL``: This should be the camera model of choice, if the
  intrinsics are unknown and every image has a different camera calibration,
  e.g., in the case of Internet photos. Both models are simplified versions of
  the ``OPENCV`` model only modeling radial distortion effects with one and two
  parameters, respectively.
- ``OPENCV``, ``FULL_OPENCV``: Use these camera models, if you know the calibration
  parameters a priori. You can also try to let COLMAP estimate the parameters,
  if you share the intrinsics for multiple images. Note that the automatic
  estimation of parameters will most likely fail, if every image has a separate
  set of intrinsic parameters.
- ``SIMPLE_RADIAL_FISHEYE``, ``RADIAL_FISHEYE``, ``OPENCV_FISHEYE``, ``FOV``,
  ``THIN_PRISM_FISHEYE``, ``RAD_TAN_THIN_PRISM_FISHEYE``: Use these camera models
  for fisheye lenses and note that all other models are not really capable of
  modeling the distortion effects of fisheye lenses. The ``FOV`` model is used by
  Google Project Tango (make sure to not initialize ``omega`` to zero).
- ``SIMPLE_FISHEYE``, ``FISHEYE``: Use these camera models for fisheye
  lenses with equidistant projection where distortion can be ignored
  or has been pre-corrected. These models use the equidistant projection
  (theta = atan(r)) without any distortion parameters. ``SIMPLE_FISHEYE``
  has a single focal length (f), while ``FISHEYE`` has two (fx, fy).
- ``SIMPLE_DIVISION``, ``DIVISION``: Use these camera models, if you know the
  calibration parameters a priori. Similar to ``SIMPLE_RADIAL`` and ``RADIAL``
  models, they can model simple radial distortion effects. The two models
  have first-order local equivalence for small distortions.
- ``EUCM``: Use this camera model for wide-angle fisheye cameras and catadioptric
  systems. It represents radial distortion using two
  parameters in addition to the standard pinhole parameters.

You can inspect the estimated intrinsic parameters by double-clicking specific
images in the model viewer or by exporting the model and opening the
``cameras.txt`` file.

Projection
----------

All perspective camera models map a 3D point in the camera coordinate system to
a 2D pixel coordinate in three steps: perspective division, distortion, and the
intrinsic transform (focal length and principal point). COLMAP uses a
corner-based pixel convention, in which the center of the top-left pixel is at
``(0.5, 0.5)`` (see :doc:`database`).

Taking ``SIMPLE_RADIAL`` (parameter list ``f, cx, cy, k``) as a worked example, a
point :math:`(X, Y, Z)` in the camera frame, which looks down the positive
:math:`Z` axis, is projected as follows:

1. Perspective division onto the normalized image plane:

   .. math::

       u = X / Z, \qquad v = Y / Z

2. Radial distortion with :math:`r^2 = u^2 + v^2`:

   .. math::

       u' = u \, (1 + k \, r^2), \qquad v' = v \, (1 + k \, r^2)

3. Focal length and principal point, giving the pixel coordinate:

   .. math::

       x = f \, u' + c_x, \qquad y = f \, v' + c_y

The inverse mapping (pixel to normalized camera ray) subtracts the principal
point, divides by the focal length, and then removes the distortion iteratively.

All other perspective models share this three-step structure and differ only in
the number of focal length parameters (a single shared ``f`` or separate ``fx``,
``fy``) and in the distortion function, e.g. ``RADIAL`` adds a second radial term
``k2`` and ``OPENCV`` adds tangential terms ``p1, p2``. The fisheye models
instead replace the perspective division with an equidistant projection. The
exact parameter list of every model is given by its ``params_info`` string and
defined in the camera models header:
https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h

Configuration
-------------

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
