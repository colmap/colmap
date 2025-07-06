.. _rig-support:

Rig Support
===========

COLMAP has native support for modeling sensor rigs during the reconstruction
process. The sensors in a rig are assumed to have fixed relative poses between
each other with one reference sensor defining the origin of the rig. A frame
defines a specific instance of the rig with all or a subset of sensors exposed
at the same time. For example, in a stereo camera rig, one camera would be
defined as the reference sensor and have an identity ``sensor_from_rig`` pose,
whereas the second camera would be posed relative to the reference camera. Each
frame would then usually be composed of two images as the measurements of both
of the cameras at the same time.

Workflow
--------

By default, when running the standard reconstruction pipeline, each camera is
modeled with a separate rig and thus each frame contains only a single image. To
model rigs, the recommended workflow is to organize images by rigs and cameras
in a folder structure as follows (ensure that images corresponding to the same
frame have identical filenames across all folders)::

    rig1/
        camera1/
            image0001.jpg
            image0002.jpg
            ...
        camera2/
            image0001.jpg # same frame as camera1/image0001.jpg
            image0002.jpg # same frame as camera1/image0002.jpg
            ...
        ...
    rig2/
        camera1/
            ...
        ...
    ...

As a next step, we would first extract features using::

    colmap feature_extractor \
        --image_path $DATASET_PATH/images \
        --database_path $DATASET_PATH/database.db \
        --ImageReader.single_camera_per_folder 1

By default, the resulting database now contains a separate rig for each camera
and a separate frame for each image. As such, we must adjust the relationships
in the database with the desired rig configuration. This is done using::

    colmap rig_configurator \
        --database_path $DATASET_PATH/database.db \
        --rig_config_path $DATASET_PATH/rig_config.json

where the ``rig_config.json`` could look as follows, if the relative sensor poses
in the rig are known a priori::

    [
      {
        "cameras": [
          {
            "image_prefix": "rig1/camera1/",
            "ref_sensor": true
          },
          {
            "image_prefix": "rig1/camera2/",
            "cam_from_rig_rotation": [
                0.7071067811865475,
                0.0,
                0.7071067811865476,
                0.0
            ],
            "cam_from_rig_translation": [
                0,
                0,
                0
            ]
          }
        ]
      },
      {
        "cameras": [
          {
            "image_prefix": "rig2/camera1/",
            "ref_sensor": true
          },
          ...
        ]
      },
      ...
    ]

Notice that this modifies the rig and frame configuration in the database, which
contains the full specification of rigs that we later feed as an input to
downstream processing steps.

With known calibrated camera parameters, each camera can optionally also have
specified ``camera_model_name`` and ``camera_params`` fields.

For more fine-grain configuration of rigs and frames, the most convenient option
is to manually configure the database using pycolmap by either using the
``apply_rig_config`` function or by individually adding the desired rig and frame
objects to the reconstruction for the most flexibility.

Next, we run standard feature matching. Note that it is important to configure
the rigs before sequential feature matching, as images in consecutive frames will
be automatically matched against each other.

Finally, we can reconstruct the scene using the standard ``mapper`` command with
the option of keeping the relative poses in the rig fixed using
``--Mapper.ba_refine_sensor_from_rig 0``.

Unknown rig sensor poses
------------------------

If the relative poses of sensors in the rig are not known a priori and we only
know that a specific set of sensors are rigidly mounted and exposed at the same
time, one can attempt the following two-step reconstruction approach. Before
starting, ensure to organize your images as detailed above and perform feature
extraction with the ``--ImageReader.single_camera_per_folder 1`` option.

Next, reconstruct the scene without rig constraints by modeling each camera as
its own rig (the default behavior of COLMAP without further configuration). Note
that this can be a partial reconstruction from a subset of the full set of input
images. The only requirement is that each camera must have at least one
registered image in the same frame with a registered image of the reference
camera. If the reconstruction was successful and the relative poses between
registered images look roughly correct, we can proceed with the next step.

The ``rig_configurator`` can also work without ``cam_from_rig_*`` transformations.
By providing an existing (partial) reconstruction of the scene, it can compute
the average relative rig sensor poses from all registered images::

    colmap rig_configurator \
        --database_path $DATASET_PATH/database.db \
        --input_path $DATASET_PATH/sparse-model-without-rigs-and-frames \
        --rig_config_path $DATASET_PATH/rig_config.json \
        [ --output_path $DATASET_PATH/sparse-model-with-rigs-and-frames ]

The provided ``rig_config.json`` must simply omit the respective
``cam_from_rig_rotation`` and ``cam_from_rig_translation`` fields.

Now, we can either run rig bundle adjustment on the (optional) output
reconstruction with configured rigs and frames::

    colmap bundle_adjuster \
        --input_path $DATASET_PATH/sparse-model-with-rigs-and-frames \
        --output_path $DATASET_PATH/bundled-sparse-model-with-rigs-and-frames

or alternatively start the reconstruction process from scratch with rig
constraints, which may lead to more accurate reconstruction results::

    colmap mapper
        --image_path $DATASET_PATH/images \
        --database_path $DATASET_PATH/database.db \
        --output_path $DATASET_PATH/sparse-model-with-rigs-and-frames


Example
-------

The following example shows an end-to-end example for how to reconstruct one of
the ETH3D rig datasets using COLMAP's rig support::

    wget https://www.eth3d.net/data/terrains_rig_undistorted.7z
    7zz x terrains_rig_undistorted.7z

    colmap feature_extractor \
        --database_path terrains/database.db \
        --image_path terrains/images \
        --ImageReader.single_camera_per_folder 1

The ETH3D dataset conveniently comes with a groundtruth COLMAP reconstruction
that we use to configure the sensor rig poses as well as camera models using::

    colmap rig_configurator \
        --database_path terrains/database.db \
        --rig_config_path terrains/rig_config.json \
        --input_path terrains/rig_calibration_undistorted

with the ``rig_config.json``::

    [
        {
            "cameras": [
                {
                    "image_prefix": "images_rig_cam4_undistorted/",
                    "ref_sensor": true
                },
                {
                    "image_prefix": "images_rig_cam5_undistorted/"
                },
                {
                    "image_prefix": "images_rig_cam6_undistorted/"
                },
                {
                    "image_prefix": "images_rig_cam7_undistorted/"
                }
            ]
        }
    ]

Notice that we do not specify the sensor poses, because we used an existing
reconstruction (in this case, the groundtruth but it can also be a
reconstruction without rig constraints, as explained in the previous section) to
automatically infer the average rig extrinsics and camera parameters.

Next, we sequentially match the frames, since they were captured as a video::

    colmap sequential_matcher --database_path terrains/database.db

Finally, we reconstruct the scene using the mapper while keeping the groundtruth
sensor rig poses and camera parameters fixed::

    mkdir -p terrains/sparse
    colmap mapper \
        --database_path terrains/database.db \
        --Mapper.ba_refine_sensor_from_rig 0 \
        --Mapper.ba_refine_focal_length 0 \
        --Mapper.ba_refine_extra_params 0 \
        --image_path terrains/images \
        --output_path terrains/sparse


Reconstruction from 360° spherical images
-----------------------------------------

COLMAP can handle collections of 360° panoramas by rendering virtual pinhole
images (similar to a cubemap) and treating them as a camera rig. Since the rig
extrinsics and camera intrinsics are known, the reconstruction process is more
robust. We provide an example Python script to reconstruct a 360° collection::

    python python/examples/panorama_sfm.py \
        --input_image_path image_directory \
        --output_path output_directory

