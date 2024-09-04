from scripts.python.read_write_model import read_model

cameras_bin = "./test_data/sparse/0/cameras.bin"
images_bin = "./test_data/sparse/0/images.bin"
points3D_bin = "./test_data/sparse/0/cpoints3D.bin"

path = "./test_data/sparse/0/"
cameras, images, points3D = read_model(path)

print("cameras keys: ", cameras.keys())
for cameras_key in cameras.keys():
    print(cameras[cameras_key])


print("images keys: ", images.keys())
for images_key in images.keys():
    # print(images[images_key])
    print(images[images_key].name)

# # print(points3D.keys())
# for points3D_key in points3D.keys():
#     # print(points3D_key[points3D_key])
#     print(points3D[points3D_key].id)
#     print(points3D[points3D_key].image_ids)
