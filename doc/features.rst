.. _features:

Feature Extraction and Matching
===============================

COLMAP supports multiple feature extraction and matching algorithms. This page
describes how to switch between them using the command-line interface or the
graphical user interface.


Feature Extractor Types
-----------------------

The following feature extractor types are available:

- ``SIFT``: Scale-Invariant Feature Transform (default). The classic and most
  widely tested feature extractor. Produces 128-dimensional uint8 descriptors.

- ``ALIKED``: A Lighter Keypoint and Descriptor Extractor. A learned feature
  extractor that produces floating-point descriptors. Requires ONNX support to
  be enabled at build time (``-DONNX_ENABLED=ON``).

- ``LOMA``: A learned feature extractor, introduced in LoMa: Local Feature
  Matching Revisited, ECCV26, that uses the DeDoDe architecture. Two descriptor
  variants are available: ``LOMA_B`` (256-dim, frozen DINOv2 features combined
  with trained convolutional features) and ``LOMA_B128`` (128-dim, more
  lightweight). Requires ONNX support to be enabled at build time
  (``-DONNX_ENABLED=ON``).

To select a feature extractor type via the command-line::

    $ colmap feature_extractor \
        --database_path $DATASET_PATH/database.db \
        --image_path $DATASET_PATH/images \
        --FeatureExtraction.type ALIKED_N16ROT \
        --AlikedExtraction.max_num_features 2048

For SIFT (the default), you can omit the type or explicitly set it::

    $ colmap feature_extractor \
        --database_path $DATASET_PATH/database.db \
        --image_path $DATASET_PATH/images \
        --FeatureExtraction.type SIFT \
        --SiftExtraction.max_num_features 8192

In the GUI, open ``Processing > Feature extraction`` and select the desired
tab (SIFT, ALIKED, LoMa, etc.) before clicking Extract.


Feature Matcher Types
---------------------

The following feature matcher types are available:

- ``SIFT_BRUTEFORCE``: Brute-force matching optimized for SIFT descriptors
  (default). Uses L2 distance with ratio test.

- ``ALIKED_BRUTEFORCE``: Brute-force matching for ALIKED descriptors. Uses
  cosine similarity. Requires ONNX support to be enabled at build time.

- ``SIFT_LIGHTGLUE``: Neural network-based matching using the LightGlue model
  for SIFT descriptors. This typically produces more matches and higher inlier
  ratios than brute-force matching, especially for challenging image pairs with
  large viewpoint or illumination changes. Requires ONNX support to be enabled
  at build time.

- ``ALIKED_LIGHTGLUE``: Neural network-based matching using the LightGlue model
  for ALIKED descriptors. Requires ONNX support to be enabled at build time.

- ``LOMA_BRUTEFORCE``: Brute-force matching for either of the LoMa descriptors.
  Uses cosine similarity. Requires ONNX support to be enabled at build time.

- ``LOMA_B``: Dedicated neural network matcher for the 256-dim ``LOMA_B``
  descriptor, comparable in size to LightGlue. Requires ONNX support to be
  enabled at build time.

- ``LOMA_R``: Same size as ``LOMA_B``, but trained with rotation augmentation
  for better robustness to rotated image pairs. Also matches the 256-dim
  ``LOMA_B`` descriptor. Requires ONNX support to be enabled at build time.

- ``LOMA_L``, ``LOMA_G``: Larger dedicated matchers for the 256-dim
  ``LOMA_B`` descriptor, in increasing order of model size and matching
  quality. Requires ONNX support to be enabled at build time.

- ``LOMA_B128``: Dedicated matcher for the lightweight 128-dim ``LOMA_B128``
  descriptor. Requires ONNX support to be enabled at build time.

- ``FLEXGLUE``: Neural network matcher operating on dense image features
  (FlexGlue). It works with the keypoints of any feature extractor and ignores
  the descriptors, which are not required. Since FlexGlue computes dense
  features from the image pixels, the images are read during matching from
  ``--FlexGlueMatching.image_path`` (required). The matcher commands have no
  global ``--image_path``, so ``FlexGlueMatching.image_path`` must also be set
  in project files. FlexGlue also predicts whether two images are covisible at
  all, and ``--FlexGlueMatching.min_covisibility`` (default 0.5, 0 disables it)
  rejects all matches of pairs with a lower covisibility score, which suppresses
  look-alike pairs of repeated structures. Guided matching is not supported.
  Requires ONNX support to be enabled at build time; a GPU is strongly
  recommended.

To select a feature matcher type via the command-line::

    $ colmap exhaustive_matcher \
        --database_path $DATASET_PATH/database.db \
        --FeatureMatching.type ALIKED_BRUTEFORCE \
        --AlikedMatching.brute_force_min_cossim 0.85

For SIFT matching (the default)::

    $ colmap exhaustive_matcher \
        --database_path $DATASET_PATH/database.db \
        --FeatureMatching.type SIFT_BRUTEFORCE \
        --SiftMatching.max_ratio 0.8

In the GUI, open ``Processing > Feature matching``, select any matching tab
(Exhaustive, Sequential, etc.), and choose the matcher type from the "Type"
dropdown in the shared options section.


Compatible Extractor and Matcher Types
--------------------------------------

The feature extractor and matcher types should be compatible:

- Use ``SIFT`` extraction with ``SIFT_BRUTEFORCE`` or ``SIFT_LIGHTGLUE`` matching.
- Use ``ALIKED_*`` extraction with ``ALIKED_BRUTEFORCE`` or ``ALIKED_LIGHTGLUE`` matching.
- Use ``LOMA_B`` extraction with ``LOMA_BRUTEFORCE``, ``LOMA_B``, ``LOMA_R``,
  ``LOMA_L``, or ``LOMA_G`` matching.
- Use ``LOMA_B128`` extraction with ``LOMA_BRUTEFORCE`` or ``LOMA_B128``
  matching.

- ``FLEXGLUE`` matching works with any extraction type, as it only uses the
  keypoint locations.

Mixing incompatible types (e.g., SIFT features with ALIKED matcher, or
``LOMA_B128`` features with the ``LOMA_B`` matcher) will result in a runtime
error. Do not mix different feature extractor types (e.g., SIFT and ALIKED)
in the same database.


ALIKED Model Variants
---------------------

ALIKED requires an ONNX model file. Several model variants are available with
different trade-offs between speed and accuracy:

- ``aliked-n16rot``: Faster and trained for some viewpoint invariance. 128-dim descriptors.
- ``aliked-n32``: More expensive but not explicitly trained for viewpoint invariance, 128-dim descriptors.

Specify the model path using ``--AlikedExtraction.*_model_path``. If the path is
a URL, COLMAP will automatically download and cache the model. You can download
different ALIKED models from the release page at https://github.com/colmap/colmap/releases/


LoMa Model Variants
--------------------

LoMa, similar to ALIKED, uses a detect+describe+match framework. The descriptor
comes in two variants:

- ``LOMA_B``: 256-dim descriptor combining frozen DINOv2 features with trained
  convolutional features.
- ``LOMA_B128``: A more lightweight 128-dim descriptor.

Each descriptor variant has its own dedicated matcher(s) with
different trade-offs between speed and accuracy:

- ``LOMA_B`` descriptors can be matched with ``LOMA_B`` (smallest, same size as
  LightGlue), ``LOMA_R`` (same size as ``LOMA_B`` but trained with rotation
  augmentation), ``LOMA_L``, or ``LOMA_G`` (progressively larger, slower, and
  more accurate), as well as ``LOMA_BRUTEFORCE``.
- ``LOMA_B128`` descriptors can only be matched with ``LOMA_B128`` or
  ``LOMA_BRUTEFORCE``. Fastest option but least accurate out of the LoMa
  matchers.

By default, all LoMa model weights are downloaded automatically and cached
locally the first time they are used, so no manual setup is required to get
started. To use a different model, specify its path using
``--LomaExtraction.*_model_path`` for extraction and
``--LomaMatching.*_model_path`` for matching. As with ALIKED, if the path is a
URL, COLMAP will automatically download and cache it. Extraction and
matching also support an opt-in ``use_bf16`` mode
(``--LomaExtraction.use_bf16`` / ``--LomaMatching.use_bf16``) for faster
inference on CUDA GPUs with compute capability 8.0 (Ampere) or newer. On other
devices, the fp32 models are used.


FlexGlue
--------

FlexGlue encodes each image into a dense stride-8 feature map with a
DINOv3-ConvNeXt backbone (at a fixed 512x512 resolution), refines the feature
maps of both images jointly with a transformer, and predicts the
correspondence of every keypoint in the other image. Keypoints are then matched
by mutual nearest neighbors between the keypoints and the predicted
correspondences. Example usage::

    $ colmap exhaustive_matcher \
        --database_path $DATASET_PATH/database.db \
        --FeatureMatching.type FLEXGLUE \
        --FlexGlueMatching.image_path $DATASET_PATH/images

With ``min_covisibility > 0``, the default models also stop early on pairs that
are confidently not covisible: after layers 2, 4, 6, and 8 of the 9 transformer
layers, a pair is rejected without running the remaining layers if its
covisibility score is below ``min_covisibility * (1 - early_stop_decay)^(9 -
layer)``. Lower values of ``--FlexGlueMatching.early_stop_decay`` stop earlier,
and 1 disables early stopping. On the 22569 exhaustive pairs of ETH3D DSLR, the
default models with the default ``early_stop_decay 0.5`` and
``min_covisibility 0.5`` stop early on 35% of the pairs, and only 6 of the
10288 pairs that are covisible according to the ground truth are rejected early
although they would pass ``min_covisibility`` after the last layer.

The dense features of the most recently used images are cached in memory
(``--FlexGlueMatching.feature_cache_size``, ~4MB per image). On CUDA GPUs with
compute capability 7.0 or newer, a variant of the matcher whose
transformer layers run in float16 is used by default
(``--FlexGlueMatching.use_fp16``), which is faster than the float32 matcher
with practically identical matches; the float32 matcher is used on the CPU, on
older GPUs, or if the float16 variant fails to initialize. On the CPU, a
single matching thread runs the models with ``--FeatureMatching.num_threads``
threads, rather than one copy of the models per thread. The
``automatic_reconstructor`` provides the ``--feature flexglue`` preset, which
extracts the same DaD keypoints as the LoMa presets and matches them with
FlexGlue. The models are downloaded automatically, and custom models can
be specified with ``--FlexGlueMatching.descriptor_model_path``,
``--FlexGlueMatching.matcher_model_path``, and
``--FlexGlueMatching.matcher_model_path_fp16``. The models are exported with
``onnx/export.py`` from https://github.com/cvg/FlexGlue. Custom matchers must
be exported with early stopping, e.g., ``--early_stop_layers 2,4,6,8``, since
the matcher requires the ``early_stop_decay`` input.


Iterative Runs
---------------------

When re-running matching on an existing database, image pairs that already have
a two-view geometry are skipped, so changing ``--FeatureMatching.type`` on such
a database leaves previously matched pairs untouched. To force re-matching, first
clear the existing results with ``database_cleaner``::

    $ colmap database_cleaner \
        --database_path $DATASET_PATH/database.db \
        --type two_view_geometries \
        --type matches

When re-extracting with a different ``--FeatureExtraction.type``, use
``--type features`` to additionally clear the extracted keypoints and descriptors.
