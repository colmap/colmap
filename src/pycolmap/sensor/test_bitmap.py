import pycolmap
import numpy as np

b = pycolmap.Bitmap.from_array(np.zeros((100, 100, 3), np.uint8))
print(b)
c = pycolmap.Bitmap.read("/Users/jsch/Downloads/IMG_0006.JPG", True)
print(c)
d = c.to_array()
print(d)
e = pycolmap.Bitmap.from_array(d)
f = e.to_array()
print(d == f)

# b.from_array(np.asarray(b))
# print(b)
# print(np.asarray(pycolmap.Bitmap(np.asarray(b))) == np.asarray(b))
