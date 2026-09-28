try: from .oxford import *
except ImportError: pass
try: from .paris import *
except ImportError: pass
try: from .distractors import *
except ImportError: pass
try: from .landmarks import Landmarks_clean, Landmarks_clean_val, Landmarks_lite
except ImportError: pass
try: from .landmarks18 import *
except ImportError: pass

# create a dataset from a string
from .create import *
create = DatasetCreator(globals())

from .dataset import split, deploy, deploy_and_split
from .generic import *
