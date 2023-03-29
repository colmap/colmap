# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

# This script is based on an original implementation by True Price.

import sys
import sqlite3
import numpy as np


IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)


    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.full(4, np.NaN), prior_t=np.full(3, np.NaN), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3),
                              qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                              tvec=np.zeros(3), config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        qvec = np.asarray(qvec, dtype=np.float64)
        tvec = np.asarray(tvec, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H),
             array_to_blob(qvec), array_to_blob(tvec)))


def example_usage():
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    args = parser.parse_args()

    if os.path.exists(args.database_path):
        print("ERROR: database path already exists -- will not modify it.")
        return

    # Open the database.

    db = COLMAPDatabase.connect(args.database_path)

    # For convenience, try creating all the tables upfront.

    db.create_tables()

    # Create dummy cameras.

    model1, width1, height1, params1 = \
        0, 1024, 768, np.array((1024., 512., 384.))
    model2, width2, height2, params2 = \
        2, 1024, 768, np.array((1024., 512., 384., 0.1))

    camera_id1 = db.add_camera(model1, width1, height1, params1)
    camera_id2 = db.add_camera(model2, width2, height2, params2)

    # Create dummy images.

    image_id1 = db.add_image("image1.png", camera_id1)
    image_id2 = db.add_image("image2.png", camera_id1)
    image_id3 = db.add_image("image3.png", camera_id2)
    image_id4 = db.add_image("image4.png", camera_id2)

    # Create dummy keypoints.
    #
    # Note that COLMAP supports:
    #      - 2D keypoints: (x, y)
    #      - 4D keypoints: (x, y, theta, scale)
    #      - 6D affine keypoints: (x, y, a_11, a_12, a_21, a_22)

    num_keypoints = 1000
    keypoints1 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints2 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints3 = np.random.rand(num_keypoints, 2) * (width2, height2)
    keypoints4 = np.random.rand(num_keypoints, 2) * (width2, height2)

    db.add_keypoints(image_id1, keypoints1)
    db.add_keypoints(image_id2, keypoints2)
    db.add_keypoints(image_id3, keypoints3)
    db.add_keypoints(image_id4, keypoints4)

    # Create dummy matches.

    M = 50
    matches12 = np.random.randint(num_keypoints, size=(M, 2))
    matches23 = np.random.randint(num_keypoints, size=(M, 2))
    matches34 = np.random.randint(num_keypoints, size=(M, 2))

    db.add_matches(image_id1, image_id2, matches12)
    db.add_matches(image_id2, image_id3, matches23)
    db.add_matches(image_id3, image_id4, matches34)

    # Commit the data to the file.

    db.commit()

    # Read and check cameras.

    rows = db.execute("SELECT * FROM cameras")

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id1
    assert model == model1 and width == width1 and height == height1
    assert np.allclose(params, params1)

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id2
    assert model == model2 and width == width2 and height == height2
    assert np.allclose(params, params2)

    # Read and check keypoints.

    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM keypoints"))

    assert np.allclose(keypoints[image_id1], keypoints1)
    assert np.allclose(keypoints[image_id2], keypoints2)
    assert np.allclose(keypoints[image_id3], keypoints3)
    assert np.allclose(keypoints[image_id4], keypoints4)

    # Read and check matches.

    pair_ids = [image_ids_to_pair_id(*pair) for pair in
                ((image_id1, image_id2),
                 (image_id2, image_id3),
                 (image_id3, image_id4))]

    matches = dict(
        (pair_id_to_image_ids(pair_id),
         blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
    )

    assert np.all(matches[(image_id1, image_id2)] == matches12)
    assert np.all(matches[(image_id2, image_id3)] == matches23)
    assert np.all(matches[(image_id3, image_id4)] == matches34)

    # Clean up.

    db.close()

    if os.path.exists(args.database_path):
        os.remove(args.database_path)


def get_keypoints(cursor, image_id):
    cursor.execute("SELECT * FROM keypoints WHERE image_id = ?;", (image_id,))
    image_idx, n_rows, n_columns, raw_data = cursor.fetchone()
    kypnts = np.frombuffer(raw_data, dtype=np.float32).reshape(n_rows, n_columns).copy()
    kypnts = kypnts[:,0:2]
    return kypnts

def normalize_points(pts):
    # Normalize image coordinates to have zero mean and unit variance
    pts_norm = (pts - np.mean(pts, axis=0)) / np.std(pts, axis=0)
    return pts_norm

def read_KRT():
    intrinsics = {}
    extrinsics = {}

    with open('/home/gaini/capstone/dataset/KRT') as f:
        lines = f.readlines()

        num = len(lines)
        i = 0
        while i < num:
            camera_id = int(lines[i])
            a = lines[i+1].split(" ")
            b = lines[i+2].split(" ")

            aa = a[2]
            bb = b[2]
            K = [[0, 0, 0], [0, 0, 0], [0, 0, 1]]
            K[0][0] = a[0]
            K[1][1] = b[1]
            K[0][2] = aa[:-2]
            K[1][2] = bb[:-2]
            intrinsics[camera_id] = np.array(K).astype('float64')


            extr = np.zeros((3, 4))
            extr1 = lines[i + 5].split(" ")
            extr2 = lines[i + 6].split(" ")
            extr3 = lines[i + 7].split(" ")
            extr[0] = extr1
            extr[1] = extr2
            extr[2] = extr3
            extrinsics[camera_id] = extr.astype('float64')

            i += 9

    return intrinsics, extrinsics

def compute_fundamental_matrix(pts1, pts2):
    # Normalize image coordinates
    pts1_norm = normalize_points(pts1)
    pts2_norm = normalize_points(pts2)

    # Construct the A matrix
    A = np.zeros((len(pts1_norm), 9))
    for i in range(len(pts1_norm)):
        u1, v1 = pts1_norm[i]
        u2, v2 = pts2_norm[i]
        A[i] = [u1*u2, v1*u2, u2, u1*v2, v1*v2, v2, u1, v1, 1]

    # Solve for the null space of A using SVD
    _, _, Vt = np.linalg.svd(A)
    f = Vt[-1].reshape(3, 3)

    # Enforce rank 2 constraint on the fundamental matrix
    Uf, Sf, Vf = np.linalg.svd(f)
    Sf[-1] = 0
    Sf = np.diag(Sf)
    f_rank2 = Uf @ Sf @ Vf

    # Denormalize fundamental matrix
    T1 = np.array([[1/np.std(pts1[:,0]), 0, -np.mean(pts1[:,0])/np.std(pts1[:,0])],
                   [0, 1/np.std(pts1[:,1]), -np.mean(pts1[:,1])/np.std(pts1[:,1])],
                   [0, 0, 1]])
    T2 = np.array([[1/np.std(pts2[:,0]), 0, -np.mean(pts2[:,0])/np.std(pts2[:,0])],
                   [0, 1/np.std(pts2[:,1]), -np.mean(pts2[:,1])/np.std(pts2[:,1])],
                   [0, 0, 1]])
    f_denorm = T2.T @ f_rank2 @ T1

    return f_denorm

def calc_l2(prediction, groundtruth):
    R_prediction = []
    R_groundtruth = []
    t_prediction = []
    t_groundtruth = []

    for id in prediction:
        R_prediction.append(prediction[id][:, :3])
        R_groundtruth.append(groundtruth[id][:, :3])
        t_prediction.append(prediction[id][:, 3:])
        t_groundtruth.append(groundtruth[id][:, 3:])

    R_prediction = np.array(R_prediction)
    R_groundtruth = np.array(R_groundtruth)

    t_prediction = np.array(t_prediction)
    t_groundtruth = np.array(t_groundtruth)

    l2_norm_R = np.sum(np.power((R_prediction - R_groundtruth), 2))
    l2_norm_t = np.sum(np.power((t_prediction - t_groundtruth), 2))

    print("l2 R", l2_norm_R) # l2 R 2.2580556793279634
    print("l2 t", l2_norm_t) # l2 t 7660005.091985998

    return l2_norm_R, l2_norm_t

def get_matching_points():
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    parser.add_argument("--outdir", default="./calculated_extrinsics.txt")
    args = parser.parse_args()

    connection = sqlite3.connect(args.database_path)
    cursor = connection.cursor()

    list_image_ids = []
    img_ids_to_names_dict = {}
    cursor.execute(
        'SELECT image_id, name, cameras.width, cameras.height FROM images LEFT JOIN cameras ON images.camera_id == cameras.camera_id;')
    for row in cursor:
        image_idx, name, width, height = row
        list_image_ids.append(image_idx)
        img_ids_to_names_dict[image_idx] = name


    num_image_ids = len(list_image_ids)

    cursor.execute('SELECT pair_id, rows, cols, data FROM two_view_geometries;')
    all_matches = {}
    for row in cursor:
        pair_id = row[0]
        rows = row[1]
        cols = row[2]
        raw_data = row[3]
        if (rows < 5):
            continue

        matches = np.frombuffer(raw_data, dtype=np.uint32).reshape(rows, cols)

        if matches.shape[0] < 5:
            continue

        all_matches[pair_id] = matches
    intrinsics, extrinsics = read_KRT()

    extrinsics_matrices = dict()

    for key in all_matches:
        pair_id = key
        matches = all_matches[key]
        id1, id2 = pair_id_to_image_ids(pair_id)

        image_name1 = img_ids_to_names_dict[id1]
        image_name2 = img_ids_to_names_dict[id2]

        reference_camera1 = image_name1.split('.')[0].split("_")[0] == "400029"
        reference_camera2 = image_name2.split('.')[0].split("_")[0] == "400029"

        if reference_camera2 or reference_camera1:

            if reference_camera2:
                id1, id2 = id2, id1

            keys1 = get_keypoints(cursor, id1)
            keys2 = get_keypoints(cursor, id2)

            match_positions = np.empty([matches.shape[0], 4])
            for i in range(0, matches.shape[0]):
                match_positions[i, :] = np.array(
                    [keys1[matches[i, int(reference_camera2)]][0], keys1[matches[i, int(reference_camera2)]][1],
                     keys2[matches[i, int(reference_camera1)]][0], keys2[matches[i, int(reference_camera1)]][1]])
            F = compute_fundamental_matrix(match_positions[:, :2], match_positions[:, 2:4])

            id1 = image_name1.split('.')[0].split("_")[0]
            id2 = image_name2.split('.')[0].split("_")[0]


            E = np.dot(intrinsics[int(id1)].T, np.dot(F, intrinsics[int(id2)]))

            U, s, Vt = np.linalg.svd(E)

            W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            R1 = np.dot(U, np.dot(W, Vt))
            R2 = np.dot(U, np.dot(W.T, Vt))

            # Choose the rotation matrix that is closest to the identity matrix
            if np.linalg.det(R1) < 0:
                R1 = -R1
            if np.linalg.det(R2) < 0:
                R2 = -R2
            if np.linalg.norm(np.eye(3) - R1) < np.linalg.norm(np.eye(3) - R2):
                R = R1
            else:
                R = R2

            # Compute translation vector t
            t = U[:, 2]

            extrinsic_matrix = np.hstack((R, t.reshape(3, 1)))

            cam = int(image_name1.split('.')[0].split("_")[0])
            if reference_camera1:
                cam = int(image_name2.split('.')[0].split("_")[0])

            extrinsics_matrices[cam] = extrinsic_matrix

    cursor.close()
    connection.close()

    with open(args.outdir, 'w') as f:
        for id in extrinsics_matrices:
            f.write(str(id)+'\n')
            f.write(np.array2string(extrinsics_matrices[id]))
            f.write('\n')

    calc_l2(extrinsics_matrices, extrinsics)

if __name__ == "__main__":
    # example_usage()
    # python database.py --database_path /home/gaini/capstone/manual_extr_calculation/database.db --outdir ./predictions.txt
    get_matching_points()
