from database import COLMAPDatabase
import os
import numpy as np
import cv2

def get_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
    return keypoints, descriptors

def get_matches(features1, features2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(features1, features2, k=2)
    match_idx1, match_idx2 = [], []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            match_idx1.append(m.queryIdx)
            match_idx2.append(m.trainIdx)
        
    matches = np.array([match_idx1, match_idx2]).T
    return matches



def create_database(database_path):
    # Create a new database
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()


    # Add cameras
    # Specify the camera parameters
    model1, width1, height1, params1 = (
        0,
        1024,
        768,
        np.array((1024.0, 512.0, 384.0)),
    )
    camera_id1 = db.add_camera(model1, width1, height1, params1)
    print('Camera ID:', camera_id1)

    
    # Here specify the path to the images
    image1_path = 'test/1.png'
    image2_path = 'test/2.png'
    image3_path = 'test/4.png'

    # Generate matches
    kp1,des1 = get_features(image1_path)
    kp2,des2 = get_features(image2_path)
    kp3,des3 = get_features(image3_path)

    matches12 = get_matches(des1, des2)
    matches23 = get_matches(des2, des3)
    matches13 = get_matches(des1, des3)
    


    # Add Images
    db.add_image(image1_path.split('/')[-1], camera_id1)
    db.add_image(image2_path.split('/')[-1], camera_id1)
    db.add_image(image3_path.split('/')[-1], camera_id1)

    # Add two view geometry
    db.add_two_view_geometry(1, 2, matches12)
    db.add_two_view_geometry(2, 3, matches23)
    db.add_two_view_geometry(1, 3, matches13)
    # No need for: db.add_two_view_geometry(2, 1, matches)


    # Add keypoints
    db.add_keypoints(1,kp1)
    db.add_keypoints(2,kp2)
    db.add_keypoints(3,kp3)

    # Add descriptors [Optional]
    # db.add_descriptors(1, features1[1])
    # db.add_descriptors(2, features2[1])

    # Add matches
    db.add_matches( 1, 2, matches12)
    db.add_matches( 2, 3, matches23)
    db.add_matches( 1, 3, matches13)

    # Commit the data to the file
    db.commit()
    db.close()

    return db


if __name__ == '__main__':
    # Create a new database
    database_path = 'test/database.db'
    if os.path.exists(database_path):
        exit('Database already exists')

    db = create_database(database_path)
    print('Database created at:', database_path)



