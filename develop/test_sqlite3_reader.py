from scripts.python.database import *
import pdb
import os
import cv2

def cameras(databsae):
    cameras_rows = databsae.execute("SELECT * FROM cameras")
    is_empty = False
    while not is_empty:
        row  = cameras_rows.fetchone()
        if row is None:
            is_empty = True
            continue
        camera_id, model, width, height, params, prior = row
        print(camera_id, model, width, height, blob_to_array(params, np.float64), prior)



def keypoints(databsae):
    keypoints_rows = databsae.execute("SELECT * FROM keypoints")
    is_empty = False
    while not is_empty:
        row  = keypoints_rows.fetchone()
        if row is None:
            is_empty = True
            continue
        image_id, rows, cols, data = row
        print("image_id", image_id, rows, cols, blob_to_array(data, np.float32, (-1, cols)).shape)
        # print(blob_to_array(data, np.float32, (-1, 2)))
        features = blob_to_array(data, np.float32, (-1, 2))
        # for feature in features:
        #     print(int(feature[0]), int(feature[1]))

        pdb.set_trace()


def matchs(databsae):
    matches_rows = databsae.execute("SELECT * FROM matches")
    is_empty = False
    while not is_empty:
        matches_row  = matches_rows.fetchone()
        if matches_row is None:
            is_empty = True
            continue
        pair_id, rows, cols, data = matches_row
        if rows == 0:
            continue
        print("pair_id", pair_id_to_image_ids(pair_id),
            "rows", rows, "cols", cols)
        # print(blob_to_array(data, np.uint32, (-1, cols)).shape)
        # print(blob_to_array(data, np.uint32, (-1, cols)))


def two_view_geometries(databsae):
    two_view_geometries_rows = databsae.execute("SELECT * FROM two_view_geometries")
    is_empty = False
    while not is_empty:
        two_view_geometries_row  = two_view_geometries_rows.fetchone()
        if two_view_geometries_row is None:
            is_empty = True
            continue
        pair_id, rows, cols, data, config, F, E, H, qvec, tvec = two_view_geometries_row
        if rows == 0:
            continue
        print("two_view_geometries", pair_id_to_image_ids(pair_id),
            "rows", rows, "cols", cols, blob_to_array(data, np.uint32, (-1, cols)).shape)
        # print("F", blob_to_array(F,dtype=np.float64,shape=(-1, 3)))
        # print("E", blob_to_array(E,dtype=np.float64,shape=(-1, 3)))
        # print("H", blob_to_array(H,dtype=np.float64,shape=(-1, 3)))
        print(config)


base_path = "./test_data/"
database_path = os.path.join(base_path, "database.db")
databsae = sqlite3.connect(database_path, factory=COLMAPDatabase)

# print(" - - - -  cameras - - - - - - -")
# cameras(databsae)

# print(" - - - -  keypoints - - - - - - -")
# keypoints(databsae)

# print(" - - - -  matchs - - - - - - -")
# matchs(databsae)


# print(" - - - -  two_view_geometries - - - - - - -")
# two_view_geometries(databsae)


keypoints_map = dict(
    (image_id, blob_to_array(data, np.float32, (-1, 2)))
    for image_id, data in databsae.execute("SELECT image_id, data FROM keypoints")
)
images_map = dict(
    (image_id, name)
    for image_id, name in databsae.execute("SELECT image_id, name FROM images")
)

matchs_map = dict(
    (pair_id_to_image_ids(pair_id), blob_to_array(data, np.uint32, (-1, 2)))
    for pair_id, rows, data in databsae.execute("SELECT pair_id, rows, data FROM matches") if rows!= 0
)


print(images_map.keys())
print(keypoints_map.keys())

for idx in keypoints_map.keys():
    images_path = os.path.join(base_path, "images",images_map[idx])
    image = cv2.imread(images_path)
    image_with_keypoints = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for keypoint in keypoints_map[idx]:
        center = (int(keypoint[0]), int(keypoint[1]))
        cv2.circle(image_with_keypoints, center, 2, (1,0,0))

    cv2.imshow('SIFT Keypoints', image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(images_path, len(keypoints_map[idx]))