# Python bindings for COLMAP

PyCOLMAP exposes to Python most capabilities of the
[COLMAP](https://colmap.github.io/) Structure-from-Motion (SfM) and Multi-View
Stereo (MVS) pipeline.

## Installation

Pre-built wheels for Linux, macOS, and Windows can be installed using pip:
```bash
pip install pycolmap
```

The wheels are automatically built and pushed to
[PyPI](https://pypi.org/project/pycolmap/) at each release. They are currently
not built with CUDA support, which requires building from source.

<details>
<summary>[Building PyCOLMAP from source - click to expand]</summary>

1. Install COLMAP from source following [the official guide](https://colmap.github.io/install.html).

3. Build PyCOLMAP:
  - On Linux and macOS:
```bash
cd pycolmap
python -m pip install .
```
  - On Windows, after installing COLMAP [via VCPKG](https://colmap.github.io/install.html#id3), run in powershell:
```powershell
cd pycolmap
python -m pip install . `
    --cmake.define.CMAKE_TOOLCHAIN_FILE="$VCPKG_INSTALLATION_ROOT/scripts/buildsystems/vcpkg.cmake" `
    --cmake.define.VCPKG_TARGET_TRIPLET="x64-windows"
```

</details>

## Reconstruction pipeline

PyCOLMAP provides bindings for multiple steps of the standard reconstruction pipeline:

- extracting and matching SIFT features
- importing an image folder into a COLMAP database
- inferring the camera parameters from the EXIF metadata of an image file
- running two-view geometric verification of matches on a COLMAP database
- triangulating points into an existing COLMAP model
- running incremental reconstruction from a COLMAP database
- dense reconstruction with multi-view stereo

Sparse & Dense reconstruction from a folder of images can be performed with:
```python
output_path: pathlib.Path
image_dir: pathlib.Path

output_path.mkdir()
mvs_path = output_path / "mvs"
database_path = output_path / "database.db"

pycolmap.extract_features(database_path, image_dir)
pycolmap.match_exhaustive(database_path)
maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
maps[0].write(output_path)
# dense reconstruction
pycolmap.undistort_images(mvs_path, output_path, image_dir)
pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)
```

PyCOLMAP can leverage the GPU for feature extraction, matching, and multi-view
stereo if COLMAP was compiled with CUDA support. Similarly, PyCOLMAP can run
Delaunay Triangulation if COLMAP was compiled with CGAL support. This requires
to build the package from source and is not available with the PyPI wheels.

All of the above steps are easily configurable with python dicts which are
recursively merged into their respective defaults, for example:
```python
pycolmap.extract_features(database_path, image_dir, sift_options={"max_num_features": 512})
# equivalent to
ops = pycolmap.SiftExtractionOptions()
ops.max_num_features = 512
pycolmap.extract_features(database_path, image_dir, sift_options=ops)
```

To list available options and their default parameters:

```python
help(pycolmap.SiftExtractionOptions)
```

For another example of usage, see [`example.py`](./examples/example.py) or
[`hloc/reconstruction.py`](https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/reconstruction.py).

## Reconstruction object

We can load and manipulate an existing COLMAP 3D reconstruction:

```python
import pycolmap
reconstruction = pycolmap.Reconstruction("path/to/reconstruction/dir")
print(reconstruction.summary())

for image_id, image in reconstruction.images.items():
    print(image_id, image)

for point3D_id, point3D in reconstruction.points3D.items():
    print(point3D_id, point3D)

for camera_id, camera in reconstruction.cameras.items():
    print(camera_id, camera)

reconstruction.write("path/to/reconstruction/dir/")
```

The object API mirrors the COLMAP C++ library. The bindings support many other operations, for example:

- projecting a 3D point into an image with arbitrary camera model:
```python
uv = camera.img_from_cam(image.cam_from_world * point3D.xyz)
```

- aligning two 3D reconstructions by their camera poses:
```python
rec2_from_rec1 = pycolmap.align_reconstructions_via_reprojections(reconstruction1, reconstrution2)
reconstruction1.transform(rec2_from_rec1)
print(rec2_from_rec1.scale, rec2_from_rec1.rotation, rec2_from_rec1.translation)
```

- exporting reconstructions to text, PLY, or other formats:
```python
reconstruction.write_text("path/to/new/reconstruction/dir/")  # text format
reconstruction.export_PLY("rec.ply")  # PLY format
```

## Estimators

We provide robust RANSAC-based estimators for absolute camera pose
(single-camera and multi-camera-rig), essential matrix, fundamental matrix,
homography, and two-view relative pose for calibrated cameras.

All RANSAC and estimation parameters are exposed as objects that behave
similarly as Python dataclasses. The RANSAC options are described in
[`colmap/optim/ransac.h`](https://github.com/colmap/colmap/blob/main/src/colmap/optim/ransac.h#L43-L72)
and their default values are:

```python
ransac_options = pycolmap.RANSACOptions(
    max_error=4.0,  # for example the reprojection error in pixels
    min_inlier_ratio=0.01,
    confidence=0.9999,
    min_num_trials=1000,
    max_num_trials=100000,
)
```

### Absolute pose estimation

For instance, to estimate the absolute pose of a query camera given 2D-3D correspondences:
```python
# Parameters:
# - points2D: Nx2 array; pixel coordinates
# - points3D: Nx3 array; world coordinates
# - camera: pycolmap.Camera
# Optional parameters:
# - estimation_options: dict or pycolmap.AbsolutePoseEstimationOptions
# - refinement_options: dict or pycolmap.AbsolutePoseRefinementOptions
answer = pycolmap.absolute_pose_estimation(points2D, points3D, camera)
# Returns: dictionary of estimation outputs or None if failure
```

2D and 3D points are passed as Numpy arrays or lists. The options are defined in
[`estimators/absolute_pose.cc`](./pycolmap/estimators/absolute_pose.h#L100-L122)
and can be passed as regular (nested) Python dictionaries:

```python
pycolmap.absolute_pose_estimation(
    points2D, points3D, camera,
    estimation_options=dict(ransac=dict(max_error=12.0)),
    refinement_options=dict(refine_focal_length=True),
)
```

### Absolute Pose Refinement

```python
# Parameters:
# - cam_from_world: pycolmap.Rigid3d, initial pose
# - points2D: Nx2 array; pixel coordinates
# - points3D: Nx3 array; world coordinates
# - inlier_mask: array of N bool; inlier_mask[i] is true if correpondence i is an inlier
# - camera: pycolmap.Camera
# Optional parameters:
# - refinement_options: dict or pycolmap.AbsolutePoseRefinementOptions
answer = pycolmap.pose_refinement(cam_from_world, points2D, points3D, inlier_mask, camera)
# Returns: dictionary of refinement outputs or None if failure
```

### Essential matrix estimation

```python
# Parameters:
# - points1: Nx2 array; 2D pixel coordinates in image 1
# - points2: Nx2 array; 2D pixel coordinates in image 2
# - camera1: pycolmap.Camera of image 1
# - camera2: pycolmap.Camera of image 2
# Optional parameters:
# - options: dict or pycolmap.RANSACOptions (default inlier threshold is 4px)
answer = pycolmap.essential_matrix_estimation(points1, points2, camera1, camera2)
# Returns: dictionary of estimation outputs or None if failure
```

### Fundamental matrix estimation

```python
answer = pycolmap.fundamental_matrix_estimation(
    points1,
    points2,
    [options],       # optional dict or pycolmap.RANSACOptions
)
```

### Homography estimation

```python
answer = pycolmap.homography_matrix_estimation(
    points1,
    points2,
    [options],       # optional dict or pycolmap.RANSACOptions
)
```

### Two-view geometry estimation

COLMAP can also estimate a relative pose between two calibrated cameras by
estimating both E and H and accounting for the degeneracies of each model.

```python
# Parameters:
# - camera1: pycolmap.Camera of image 1
# - points1: Nx2 array; 2D pixel coordinates in image 1
# - camera2: pycolmap.Camera of image 2
# - points2: Nx2 array; 2D pixel coordinates in image 2
# Optional parameters:
# - matches: Nx2 integer array; correspondences across images
# - options: dict or pycolmap.TwoViewGeometryOptions
answer = pycolmap.estimate_calibrated_two_view_geometry(camera1, points1, camera2, points2)
# Returns: pycolmap.TwoViewGeometry
```

The `TwoViewGeometryOptions` control how each model is selected. The output
structure contains the geometric model, inlier matches, the relative pose (if
`options.compute_relative_pose=True`), and the type of camera configuration,
which is an instance of the enum `pycolmap.TwoViewGeometryConfiguration`.

### Camera argument

Some estimators expect a COLMAP camera object, which can be created as follows:

```python
camera = pycolmap.Camera(
    model=camera_model_name_or_id,
    width=width,
    height=height,
    params=params,
)
```

The different camera models and their extra parameters are defined in
[`colmap/src/colmap/sensor/models.h`](https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h).
For example for a pinhole camera:

```python
camera = pycolmap.Camera(
    model='SIMPLE_PINHOLE',
    width=width,
    height=height,
    params=[focal_length, cx, cy],
)
```

Alternatively, we can also pass a camera dictionary:

```python
camera_dict = {
    'model': COLMAP_CAMERA_MODEL_NAME_OR_ID,
    'width': IMAGE_WIDTH,
    'height': IMAGE_HEIGHT,
    'params': EXTRA_CAMERA_PARAMETERS_LIST
}
```


## SIFT feature extraction

```python
import numpy as np
import pycolmap
from PIL import Image, ImageOps

# Input should be grayscale image with range [0, 1].
img = Image.open('image.jpg').convert('RGB')
img = ImageOps.grayscale(img)
img = np.array(img).astype(np.float) / 255.

# Optional parameters:
# - options: dict or pycolmap.SiftExtractionOptions
# - device: default pycolmap.Device.auto uses the GPU if available
sift = pycolmap.Sift()

# Parameters:
# - image: HxW float array
keypoints, descriptors = sift.extract(img)
# Returns:
# - keypoints: Nx4 array; format: x (j), y (i), scale, orientation
# - descriptors: Nx128 array; L2-normalized descriptors
```
