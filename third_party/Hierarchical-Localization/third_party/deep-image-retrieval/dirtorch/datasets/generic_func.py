''' Generic functions for Dataset() class
'''
import pdb
import numpy as np
from collections import defaultdict


def find_and_list_classes(labels, cls_idx=None ):
    ''' Given a list of image labels, deduce the list of classes.

    Parameters:
    -----------
    labels : list
        per-image labels (can be str, int, ...)

    cls_idx : dict or None

    Returns:
    --------
        classes = [class0_name, class1_name, ...]
        cls_idx = {class_name : class_index}
    '''
    assert not isinstance(labels, set), 'labels must be ordered'
    if not cls_idx: cls_idx = {} # don't put it as default arg!!

    # map string labels to integers
    uniq_labels = set(labels)
    nlabels = len(uniq_labels)
    for label in cls_idx:
        assert label in uniq_labels, "error: missing forced label '%s'" % str(label)
        nlabels += (label not in uniq_labels) # one other label

    classes = {idx:cls for cls,idx in cls_idx.items()}
    remaining_labels = set(range(nlabels)) - set(cls_idx.values())
    for cls in labels:
        if cls in cls_idx: continue # already there
        cls_idx[cls] = i = remaining_labels.pop()
        classes[cls_idx[cls]] = cls

    assert min(classes.keys()) == 0 and len(classes) == max(classes.keys()) + 1 # no holes between integers
    classes = [classes[c] for c in range(len(classes))] # dict --> list

    return classes, cls_idx


def find_relevants(labels):
    """ For each class, find the set of images from the same class.

    Returns:
    --------
    c_relevant_idx = {class_name: [list of image indices]}
    """
    assert not isinstance(labels, set), 'labels must be ordered'

    # Get relevants images for each class
    c_relevant_idx = defaultdict(list)
    for i in range(len(labels)):
        c_relevant_idx[labels[i]].append(i)

    return c_relevant_idx


