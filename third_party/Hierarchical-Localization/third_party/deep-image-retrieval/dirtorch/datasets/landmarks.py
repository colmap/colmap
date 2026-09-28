import os
from .generic import ImageListLabels

DB_ROOT = os.environ['DB_ROOT']

class Landmarks_clean(ImageListLabels):
    def __init__(self):
        ImageListLabels.__init__(self, os.path.join(DB_ROOT, 'landmarks/annotations/annotation_clean_train.txt'),
                                 os.path.join(DB_ROOT, 'landmarks/'))

class Landmarks_clean_val(ImageListLabels):
    def __init__(self):
        ImageListLabels.__init__(self, os.path.join(DB_ROOT, 'landmarks/annotations/annotation_clean_val.txt'),
                                 os.path.join(DB_ROOT, 'landmarks/'))

class Landmarks_lite(ImageListLabels):
    def __init__(self):
        ImageListLabels.__init__(self, os.path.join(DB_ROOT, 'landmarks/annotations/extra_landmark_images.txt'),
                                 os.path.join(DB_ROOT, 'landmarks/'))
