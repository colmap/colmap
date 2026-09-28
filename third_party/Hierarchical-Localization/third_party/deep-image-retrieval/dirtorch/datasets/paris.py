from .generic import ImageListRelevants
import os

DB_ROOT = os.environ['DB_ROOT']

class Paris6K(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'paris6k/gnd_paris6k.pkl'),
                                 root=os.path.join(DB_ROOT, 'paris6k'))

class RParis6K(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'paris6k/gnd_rparis6k.pkl'),
                                 root=os.path.join(DB_ROOT, 'paris6k'))
