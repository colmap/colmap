import numpy as np
from read_model import read_model


def compare_cameras(cameras_1, cameras_2):
    assert len(cameras_1) == len(cameras_2)
    for cam_id_1, cam_id_2 in zip(cameras_1, cameras_2):
        cam_1 = cameras_1[cam_id_1]
        cam_2 = cameras_2[cam_id_2]
        assert cam_1.id == cam_2.id
        assert cam_1.width == cam_2.width
        assert cam_1.height == cam_2.height
        assert np.allclose(cam_1.params, cam_2.params)


def compare_images(imgs_1, imgs_2):
    assert len(imgs_1) == len(imgs_2)
    for img_1_id, img_2_id in zip(imgs_1, imgs_2):
        img_1 = imgs_1[img_1_id]
        img_2 = imgs_2[img_2_id]
        assert img_1.id == img_2.id
        assert np.allclose(img_1.qvec, img_2.qvec)
        assert np.allclose(img_1.tvec, img_2.tvec)
        assert img_1.camera_id == img_2.camera_id
        assert img_1.name == img_2.name
        assert np.allclose(img_1.xys, img_2.xys)
        assert np.array_equal(img_1.point3D_ids, img_2.point3D_ids)


def compare_points(points3D_1, points3D_2):
    for point_id_1, point_id_2 in zip(points3D_1, points3D_2):
        point_1 = points3D_1[point_id_1]
        point_2 = points3D_2[point_id_2]
        assert point_1.id == point_2.id
        assert np.allclose(point_1.xyz, point_1.xyz)
        assert np.array_equal(point_1.rgb, point_1.rgb)
        assert np.allclose(point_1.error, point_2.error)
        assert np.array_equal(point_1.image_ids, point_1.image_ids)
        assert np.array_equal(point_1.point2D_idxs, point_1.point2D_idxs)


def main():
    import sys
    if len(sys.argv) != 3:
        print("Usage: python read_model.py path/to/model/folder/txt path/to/model/folder/bin")
        return

    print('Comparing text and binary output.')
    path_to_model_txt_folder = sys.argv[1]
    path_to_model_bin_folder = sys.argv[2]
    cameras_txt, images_txt, points3D_txt = read_model(path_to_model_txt_folder, ext='.txt')
    cameras_bin, images_bin, points3D_bin = read_model(path_to_model_bin_folder, ext='.bin')
    compare_cameras(cameras_txt, cameras_bin)
    compare_images(images_txt, images_bin)
    compare_points(points3D_txt, points3D_bin)
    print('Text and binary output are equal.')

if __name__ == '__main__':
    main()
