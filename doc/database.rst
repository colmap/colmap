.. _database-format:

Database Format
===============

COLMAP stores all extracted information in a single SQLite database file. The
database can be accessed with the database management toolkit in the COLMAP GUI,
the provided C++ database API (see ``src/base/database.h``), or with a scripting
language of your choice (see ``scripts/python/database.py``).

The database contains the following tables:

- cameras
- images
- keypoints
- descriptors
- matches
- two_view_geometries

To initialize an empty SQLite database file with the required schema, you can
either create a new project in the GUI or execute `src/colmap/exe/database_create.cc`.

Cameras and Images
------------------

The relation between cameras and images is 1-to-N. This has important
implications for Structure-from-Motion, since one camera shares the same
intrinsic parameters (focal length, principal point, distortion, etc.), while
every image has separate extrinsic parameters (orientation and location).

The intrinsic parameters of cameras are stored as contiguous binary blobs in
`float64`, ordered as specified in ``src/base/camera_models.h``. COLMAP only
uses cameras that are referenced by images, all other cameras are ignored.

The ``name`` column in the images table is the unique relative path in the image
folder. As such, the database file and image folder can be moved to different
locations, as long as the relative folder structure is preserved.

When manually inserting images and cameras into the database, make sure
that all identifiers are positive and non-zero, i.e. ``image_id > 0``
and ``camera_id > 0``.


Keypoints and Descriptors
-------------------------

The detected keypoints are stored as row-major `float32` binary blobs, where the
first two columns are the X and Y locations in the image, respectively. COLMAP
uses the convention that the upper left image corner has coordinate `(0, 0)` and
the center of the upper left most pixel has coordinate `(0.5, 0.5)`. If the
keypoints have 4 columns, then the feature geometry is a similarity and the
third column is the scale and the fourth column the orientation of the feature
(according to SIFT conventions). If the keypoints have 6 columns, then the
feature geometry is an affinity and the last 4 columns encode its affine shape
(see ``src/feature/types.h`` for details).

The extracted descriptors are stored as row-major `uint8` binary blobs, where
each row describes the feature appearance of the corresponding entry in the
keypoints table. Note that COLMAP only supports 128-D descriptors for now, i.e.
the `cols` column must be 128.

In both tables, the `rows` table specifies the number of detected features per
image, while `rows=0` means that an image has no features. For feature matching
and geometric verification, every image must have a corresponding keypoints and
descriptors entry. Note that only vocabulary tree matching with fast spatial
verification requires meaningful values for the local feature geometry, i.e.,
only X and Y must be provided and the other keypoint columns can be set to zero.
The rest of the reconstruction pipeline only uses the keypoint locations.


Matches
-------

Feature matching stores its output in the `matches` table and geometric
verification in the `two_view_geometries` table. COLMAP only uses the data in
`two_view_geometries` for reconstruction. Every entry in the two tables stores
the feature matches between two unique images, where the `pair_id` is the
row-major, linear index in the upper-triangular match matrix, generated as
follows::

    def image_ids_to_pair_id(image_id1, image_id2):
        if image_id1 > image_id2:
            return 2147483647 * image_id2 + image_id1
        else:
            return 2147483647 * image_id1 + image_id2

and image identifiers can be uniquely determined from the `pair_id` as::

    def pair_id_to_image_ids(pair_id):
        image_id2 = pair_id % 2147483647
        image_id1 = (pair_id - image_id2) / 2147483647
        return image_id1, image_id2

The `pair_id` enables efficient database queries, as the matches tables may
contain several hundred millions of entries. This scheme limits the maximum
number of images in a database to 2147483647 (maximum value of signed 32-bit
integers), i.e. `image_id` must be smaller than 2147483647.

The binary blobs in the matches tables are row-major `uint32` matrices, where
the left column are zero-based indices into the features of `image_id1` and the
second column into the features of `image_id2`. The column `cols` must be 2 and
the `rows` column specifies the number of feature matches.

The F, E, H blobs in the `two_view_geometries` table are stored as 3x3 matrices
in row-major `float64` format. The meaning of the `config` values are documented
in the `src/estimators/two_view_geometry.h` source file.
