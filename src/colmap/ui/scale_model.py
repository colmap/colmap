import sys
import pycolmap

reconstruction = pycolmap.Reconstruction()
reconstruction.read_text(sys.argv[1])
new_from_old = pycolmap.Sim3d()
new_from_old.scale = 2
new_from_old.translation = [100000, 2000000, -300000]
reconstruction.transform(new_from_old)
reconstruction.write_binary(sys.argv[1])
