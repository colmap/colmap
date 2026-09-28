import os
from .generic import ImageListRelevants

DB_ROOT = os.environ['DB_ROOT']

class Oxford5K(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'oxford5k/gnd_oxford5k.pkl'),
                                 root=os.path.join(DB_ROOT, 'oxford5k'))

class ROxford5K(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'oxford5k/gnd_roxford5k.pkl'),
                                 root=os.path.join(DB_ROOT, 'oxford5k'))
