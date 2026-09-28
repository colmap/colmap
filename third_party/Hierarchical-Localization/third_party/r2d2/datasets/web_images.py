# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import os, pdb
from tqdm import trange

from .dataset import Dataset


class RandomWebImages (Dataset):
    """ 1 million distractors from Oxford and Paris Revisited
        see http://ptak.felk.cvut.cz/revisitop/revisitop1m/
    """
    def __init__(self, start=0, end=1024, root="data/revisitop1m"):
        Dataset.__init__(self)
        self.root = root
        
        bar = None
        self.imgs  = []
        for i in range(start, end):
            try: 
                # read cached list
                img_list_path = os.path.join(self.root, "image_list_%d.txt"%i) 
                cached_imgs = [e.strip() for e in open(img_list_path)]
                assert cached_imgs, f"Cache '{img_list_path}' is empty!"
                self.imgs += cached_imgs

            except IOError:
                if bar is None: 
                    bar = trange(start, 4*end, desc='Caching')
                    bar.update(4*i)
                
                # create it
                imgs = []
                for d in range(i*4,(i+1)*4): # 4096 folders in total, on average 256 each
                    key = hex(d)[2:].zfill(3)
                    folder = os.path.join(self.root, key)
                    if not os.path.isdir(folder): continue
                    imgs += [f for f in os.listdir(folder) if verify_img(folder,f)]
                    bar.update(1)
                assert imgs, f"No images found in {folder}/"
                open(img_list_path,'w').write('\n'.join(imgs))
                self.imgs += imgs

        if bar: bar.update(bar.total - bar.n)
        self.nimg = len(self.imgs)

    def get_key(self, i):
        key = self.imgs[i]
        return os.path.join(key[:3], key)


def verify_img(folder, f):
    path = os.path.join(folder, f)
    if not f.endswith('.jpg'): return False
    try: 
        from PIL import Image
        Image.open(path).convert('RGB') # try to open it
        return True
    except: 
        return False


