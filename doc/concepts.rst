.. _concepts:

Key Concepts
============

Starting from COLMAP 3.12, the concepts of rigs and frames have been introduced
to enable a principled modeling of multi-sensor platforms as well as 360° panorama
images. These concepts provide a structured framework to organize sensors and
their measurements, enabling more flexible calibration and fusion of diverse
data types (e.g., see :ref:`rig-support`).

These additions are backward-compatible and do not affect the traditional, default usage
of COLMAP for single-camera capture setups.


.. _sensors:

Sensors and Measurements
------------------------

A **sensor** is a device that captures data about the environment, producing
measurements at specific timestamps. The most common sensor type is the camera,
which captures images as its measurements. Other examples include IMUs
(Inertial Measurement Units), which record acceleration and angular velocity,
and GNSS receivers, which provide absolute position data. 

Currently, COLMAP supports only cameras and their image measurements, though the
sensor concept is designed to extend to other types such as IMUs and GNSS for
future support of multi-modal data fusion.


.. _rigs:

Rigs
----

A **rig** models a platform composed of multiple sensors with fixed relative poses,
enabling synchronized and consistent multi-sensor data collection. Examples
include stereo camera setups, headworn AR/VR devices, and autonomous driving
sensor suites. It can also be virtual — for example, a rig modeling multiple
virtual cameras arranged to capture overlapping views used to create seamless
360° panoramic images.

In COLMAP, each sensor must be uniquely associated with exactly one rig. Each rig
has a single reference sensor that defines its origin. For example, in a stereo
camera rig, one camera is designated as the reference sensor with an identity
`sensor_from_rig` pose, while the second camera’s pose is defined relative to
this reference. In a single-camera setup, the camera itself serves as the sole
reference sensor for its rig.


.. _frames:

Frames
------

A **frame** represents a rig captured at a single timestamp, containing measurements
from one or more sensors within that rig. For example, if a rig consists of
three sensors, a frame may include measurements from all three sensors, or only
a subset, depending on availability. This concept allows association of multi-sensor 
data at specific points in time.

For instance, in a stereo camera rig recording video, each frame corresponds to a
set of two images—one from each camera—captured at the same moment.


