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
tab (SIFT, ALIKED, etc.) before clicking Extract.


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

To select a feature matcher type via the command-line::

    $ colmap exhaustive_matcher \
        --database_path $DATASET_PATH/database.db \
        --FeatureMatching.type ALIKED_BRUTEFORCE \
        --AlikedMatching.min_cossim 0.85

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

Mixing incompatible types (e.g., SIFT features with ALIKED matcher) will
result in a runtime error. Do not mix different feature extractor types
(e.g., SIFT and ALIKED) in the same database.


ALIKED Model Variants
---------------------

ALIKED requires an ONNX model file. Several model variants are available with
different trade-offs between speed and accuracy:

- ``aliked-n16rot``: Faster and trained for some viewpoint invariance. 128-dim descriptors.
- ``aliked-n32``: More expensive but not explicitly trained for viewpoint invariance, 128-dim descriptors.

Specify the model path using ``--AlikedExtraction.*_model_path``. If the path is
a URL, COLMAP will automatically download and cache the model. You can download
different ALIKED models from the release page at https://github.com/colmap/colmap/releases/
