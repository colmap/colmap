from .dataset import split, deploy, deploy_and_split
from .generic import *


class DatasetCreator:
    ''' Create a dataset from a string.

    dataset_cmd (str):
        Command to execute.
        ex: "ImageList('path/to/list.txt')"

    Returns:
        instanciated dataset.
    '''
    def __init__(self, globs):
        for k, v in globs.items():
            globals()[k] = v
    
    def __call__(self, dataset_cmd ):
        if '(' not in dataset_cmd:
            dataset_cmd += "()"

        try:
            return eval(dataset_cmd)
        except NameError:
            import sys, inspect
            dbs = [name for name,obj in globals().items() if name[0]!='_' and name not in ('DatasetCreator','defaultdict') and inspect.isclass(obj)]
            print("Error: unknown dataset %s\nAvailable datasets: %s" % (dataset_cmd.replace('()',''), ', '.join(sorted(dbs))), file=sys.stderr) 
            sys.exit(1)

