import cv2
import pycolmap
import os
import numpy as np

def show_features( image, keypoints):
    #feature ['keypoints', 'scores', 'descriptors', 'tile_idx', 'image_size']
    import copy
    image_copy = copy.deepcopy(image)
    # image_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for pair in keypoints:
        u = pair[0].astype(np.int64)
        v = pair[1].astype(np.int64)

        cv2.circle(image_copy, (int(u), int(v)), 4, (int(0), int(1), int(0)), -1)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(image_copy)
    # plt.show()
    return image_copy

base_path = "./test_data/"
database_path = os.path.join(base_path, "database.db")

database = pycolmap.Database(database_path)
all_images = database.read_all_images()
images_id = []
for img in all_images:
    print(img.image_id, img.camera_id, img.name)
    images_id.append(img.image_id)

for src_image_id in images_id:
    # descriptors = database.read_descriptors(src_image_id)
    # keypoints = database.read_keypoints(src_image_id)
    # if len(descriptors) == len(keypoints):
    #     print(len(descriptors), len(keypoints))
    for tgt_image_id in images_id:
        src_keypoints = database.read_keypoints(src_image_id)
        tgt_keypoints = database.read_keypoints(tgt_image_id)
        matches = database.read_matches(src_image_id, tgt_image_id)
        src_img = cv2.imread(os.path.join(base_path, "images", str(database.read_image(src_image_id).name)))
        tgt_img = cv2.imread(os.path.join(base_path, "images", str(database.read_image(tgt_image_id).name)))

        print(len(matches), len(src_keypoints), len(tgt_keypoints))
        print(np.array(matches))
        matches_cv = []
        for idx in range(len(matches)):
            if idx % 20 != 0:
                continue
            idx_pair = matches[idx]
            if not (idx_pair[0] < len(src_keypoints) and idx_pair[1] < len(tgt_keypoints)):
                continue
            dmatch = cv2.DMatch(idx_pair[0], idx_pair[1], 0, 0.0)

            matches_cv.append(dmatch)
        # print(len(matches))
        # if len(matches) > 0:
            
        #     max_x = np.array(matches)[:,0].max()
        #     max_y = np.array(matches)[:,1].max()
        #     print(max_x,max_y )

        src_keypoints_cv = [cv2.KeyPoint(x=p[0], y=p[1], size=1) for p in src_keypoints]
        tgt_keypoints_cv = [cv2.KeyPoint(x=p[0], y=p[1], size=1) for p in tgt_keypoints]
        img_matches = cv2.drawMatches(src_img, src_keypoints_cv, tgt_img, tgt_keypoints_cv, matches_cv, None, 
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                              matchesThickness=1)
        cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Matches', 2160, 2160)
        cv2.imshow('Matches', img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()