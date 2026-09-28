import os
from .generic import ImageListLabels, ImageList

DB_ROOT = os.environ['DB_ROOT']

class Landmarks18_train(ImageListLabels):
    def __init__(self):
        ImageListLabels.__init__(self, os.path.join(DB_ROOT, 'landmarks18/lists/train.txt'),
                                 os.path.join(DB_ROOT, 'landmarks18/'))

class Landmarks18(ImageListLabels):
    def __init__(self):
        ImageListLabels.__init__(self, os.path.join(DB_ROOT, 'landmarks18/lists/train_all.txt'),
                                 os.path.join(DB_ROOT, 'landmarks18/'))

class Landmarks18_lite(ImageListLabels):
    def __init__(self):
        ImageListLabels.__init__(self, os.path.join(DB_ROOT, 'landmarks18/lists/train_lite.txt'),
                                 os.path.join(DB_ROOT, 'landmarks18/'))

class Landmarks18_mid(ImageListLabels):
    def __init__(self):
        ImageListLabels.__init__(self, os.path.join(DB_ROOT, 'landmarks18/lists/train_mid.txt'),
                                 os.path.join(DB_ROOT, 'landmarks18/'))

class Landmarks18_5K(ImageListLabels):
    def __init__(self):
        ImageListLabels.__init__(self, os.path.join(DB_ROOT, 'landmarks18/lists/train_5K.txt'),
                                 os.path.join(DB_ROOT, 'landmarks18/'))

class Landmarks18_val(ImageListLabels):
    def __init__(self):
        ImageListLabels.__init__(self, os.path.join(DB_ROOT, 'landmarks18/lists/val.txt'),
                                 os.path.join(DB_ROOT, 'landmarks18/'))

class Landmarks18_valdstr(ImageListLabels):
    def __init__(self):
        ImageListLabels.__init__(self, os.path.join(DB_ROOT, 'landmarks18/lists/val_distractors.txt'),
                                 os.path.join(DB_ROOT, 'landmarks18/'))

class Landmarks18_index(ImageList):
    def __init__(self):
        ImageList.__init__(self, os.path.join(DB_ROOT, 'landmarks18/lists/index.txt'),
                                 os.path.join(DB_ROOT, 'landmarks18/'))

class Landmarks18_new_index(ImageList):
    def __init__(self):
        ImageList.__init__(self, os.path.join(DB_ROOT, 'landmarks18/lists/index_new.txt'),
                                 os.path.join(DB_ROOT, 'landmarks18/'))

class Landmarks18_test(ImageList):
    def __init__(self):
        ImageList.__init__(self, os.path.join(DB_ROOT, 'landmarks18/lists/test.txt'),
                                 os.path.join(DB_ROOT, 'landmarks18/'))

class Landmarks18_pca(ImageList):
    def __init__(self):
        ImageList.__init__(self, os.path.join(DB_ROOT, 'landmarks18/lists/train_pca.txt'),
                                 os.path.join(DB_ROOT, 'landmarks18/'))

class Landmarks18_missing_index(ImageList):
    def __init__(self):
        ImageList.__init__(self, os.path.join(DB_ROOT, 'landmarks18/lists/missing_index.txt'),
                                 os.path.join(DB_ROOT, 'landmarks18/'))


