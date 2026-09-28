import os
import json
import pdb
import numpy as np
import pickle
import os.path as osp
import json

from .dataset import Dataset
from .generic_func import *


class ImageList(Dataset):
    ''' Just a list of images (no labels, no query).

    Input:  text file, 1 image path per row
    '''
    def __init__(self, img_list_path, root='', imgs=None):
        self.root = root
        if imgs is not None:
            self.imgs = imgs
        else:
            self.imgs = [e.strip() for e in open(img_list_path)]

        self.nimg = len(self.imgs)
        self.nclass = 0
        self.nquery = 0

    def get_key(self, i):
        return self.imgs[i]


class LabelledDataset (Dataset):
    """ A dataset with per-image labels
        and some convenient functions.
    """
    def find_classes(self, *arg, **cls_idx):
        labels = arg[0] if arg else self.labels
        self.classes, self.cls_idx = find_and_list_classes(labels, cls_idx=cls_idx)
        self.nclass = len(self.classes)
        self.c_relevant_idx = find_relevants(self.labels)


class ImageListLabels(LabelledDataset):
    ''' Just a list of images with labels (no queries).

    Input:  text file, 1 image path and label per row (space-separated)
    '''
    def __init__(self, img_list_path, root=None):
        self.root = root
        if osp.splitext(img_list_path)[1] == '.txt':
            tmp = [e.strip() for e in open(img_list_path)]
            self.imgs = [e.split(' ')[0] for e in tmp]
            self.labels = [e.split(' ')[1] for e in tmp]
        elif osp.splitext(img_list_path)[1] == '.json':
            d = json.load(open(img_list_path))
            self.imgs = []
            self.labels = []
            for i, l in d.items():
                self.imgs.append(i)
                self.labels.append(l)
        self.find_classes()

        self.nimg = len(self.imgs)
        self.nquery = 0

    def get_key(self, i):
        return self.imgs[i]

    def get_label(self, i, toint=False):
        label = self.labels[i]
        if toint:
            label = self.cls_idx[label]
        return label

    def get_query_db(self):
        return self


class ImageListLabelsQ(ImageListLabels):
    ''' Two list of images with labels: one for the dataset and one for the queries.

    Input:  text file, 1 image path and label per row (space-separated)
    '''
    def __init__(self, img_list_path, query_list_path, root=None):
        self.root = root
        tmp = [e.strip() for e in open(img_list_path)]
        self.imgs = [e.split(' ')[0] for e in tmp]
        self.labels = [e.split(' ')[1] for e in tmp]
        tmp = [e.strip() for e in open(query_list_path)]
        self.qimgs = [e.split(' ')[0] for e in tmp]
        self.qlabels = [e.split(' ')[1] for e in tmp]
        self.find_classes()

        self.nimg = len(self.imgs)
        self.nquery = len(self.qimgs)

    def find_classes(self, *arg, **cls_idx):
        labels = arg[0] if arg else self.labels + self.qlabels
        self.classes, self.cls_idx = find_and_list_classes(labels, cls_idx=cls_idx)
        self.nclass = len(self.classes)
        self.c_relevant_idx = find_relevants(self.labels)

    def get_query_db(self):
        return ImagesAndLabels(self.qimgs, self.qlabels, self.cls_idx, root=self.root)


class ImagesAndLabels(ImageListLabels):
    ''' Just a list of images with labels.

    Input:  two arrays containing the text file, 1 image path and label per row (space-separated)
    '''
    def __init__(self, imgs, labels, cls_idx, root=None):
        self.root = root
        self.imgs = imgs
        self.labels = labels
        self.cls_idx = cls_idx
        self.nclass = len(self.cls_idx.keys())

        self.nimg = len(self.imgs)
        self.nquery = 0


class ImageListRelevants(Dataset):
    """ A dataset composed by a list of images, a list of indices used as queries,
        and for each query a list of relevant and junk indices (ie. Oxford-like GT format)

        Input: path to the pickle file
    """
    def __init__(self, gt_file, root=None, img_dir='jpg', ext='.jpg'):
        self.root = root
        self.img_dir = img_dir

        with open(gt_file, 'rb') as f:
            gt = pickle.load(f)
        self.imgs = [osp.splitext(e)[0] + (osp.splitext(e)[1] if osp.splitext(e)[1] else ext) for e in gt['imlist']]
        self.qimgs = [osp.splitext(e)[0] + (osp.splitext(e)[1] if osp.splitext(e)[1] else ext) for e in gt['qimlist']]
        self.qroi = [tuple(e['bbx']) for e in gt['gnd']]
        if 'ok' in gt['gnd'][0]:
            self.relevants = [e['ok'] for e in gt['gnd']]
        else:
            self.relevants = None
            self.easy = [e['easy'] for e in gt['gnd']]
            self.hard = [e['hard'] for e in gt['gnd']]
        self.junk = [e['junk'] for e in gt['gnd']]

        self.nimg = len(self.imgs)
        self.nquery = len(self.qimgs)

    def get_relevants(self, qimg_idx, mode='classic'):
        if mode == 'classic':
            rel = self.relevants[qimg_idx]
        elif mode == 'easy':
            rel = self.easy[qimg_idx]
        elif mode == 'medium':
            rel = self.easy[qimg_idx] + self.hard[qimg_idx]
        elif mode == 'hard':
            rel = self.hard[qimg_idx]
        return rel

    def get_junk(self, qimg_idx, mode='classic'):
        if mode == 'classic':
            junk = self.junk[qimg_idx]
        elif mode == 'easy':
            junk = self.junk[qimg_idx] + self.hard[qimg_idx]
        elif mode == 'medium':
            junk = self.junk[qimg_idx]
        elif mode == 'hard':
            junk = self.junk[qimg_idx] + self.easy[qimg_idx]
        return junk

    def get_query_filename(self, qimg_idx, root=None):
        return os.path.join(root or self.root, self.img_dir, self.get_query_key(qimg_idx))

    def get_query_roi(self, qimg_idx):
        return self.qroi[qimg_idx]

    def get_key(self, i):
        return self.imgs[i]

    def get_query_key(self, i):
        return self.qimgs[i]

    def get_query_db(self):
        return ImageListROIs(self.root, self.img_dir, self.qimgs, self.qroi)

    def get_query_groundtruth(self, query_idx, what='AP', mode='classic'):
        # negatives
        res = -np.ones(self.nimg, dtype=np.int8)
        # positive
        res[self.get_relevants(query_idx, mode)] = 1
        # junk
        res[self.get_junk(query_idx, mode)] = 0
        return res

    def eval_query_AP(self, query_idx, scores):
        """ Evaluates AP for a given query.
        """
        from ..utils.evaluation import compute_average_precision
        if self.relevants:
            gt = self.get_query_groundtruth(query_idx, 'AP')  # labels in {-1, 0, 1}
            assert gt.shape == scores.shape, "scores should have shape %s" % str(gt.shape)
            assert -1 <= gt.min() and gt.max() <= 1, "bad ground-truth labels"
            keep = (gt != 0)  # remove null labels

            gt, scores = gt[keep], scores[keep]
            gt_sorted = gt[np.argsort(scores)[::-1]]
            positive_rank = np.where(gt_sorted == 1)[0]
            return compute_average_precision(positive_rank)
        else:
            d = {}
            for mode in ('easy', 'medium', 'hard'):
                gt = self.get_query_groundtruth(query_idx, 'AP', mode)  # labels in {-1, 0, 1}
                assert gt.shape == scores.shape, "scores should have shape %s" % str(gt.shape)
                assert -1 <= gt.min() and gt.max() <= 1, "bad ground-truth labels"
                keep = (gt != 0)  # remove null labels
                if sum(gt[keep] > 0) == 0:  # exclude queries with no relevants from the evaluation
                    d[mode] = -1
                else:
                    gt2, scores2 = gt[keep], scores[keep]
                    gt_sorted = gt2[np.argsort(scores2)[::-1]]
                    positive_rank = np.where(gt_sorted == 1)[0]
                    d[mode] = compute_average_precision(positive_rank)
            return d


class ImageListROIs(Dataset):
    def __init__(self, root, img_dir, imgs, rois):
        self.root = root
        self.img_dir = img_dir
        self.imgs = imgs
        self.rois = rois

        self.nimg = len(self.imgs)
        self.nclass = 0
        self.nquery = 0

    def get_key(self, i):
        return self.imgs[i]

    def get_roi(self, i):
        return self.rois[i]

    def get_image(self, img_idx, resize=None):
        from PIL import Image
        img = Image.open(self.get_filename(img_idx)).convert('RGB')
        img = img.crop(self.rois[img_idx])
        if resize:
            img = img.resize(resize, Image.ANTIALIAS if np.prod(resize) < np.prod(img.size) else Image.BICUBIC)
        return img


def not_none(label):
    return label is not None


class ImageClusters(LabelledDataset):
    ''' Just a list of images with labels (no query).

    Input:  JSON, dict of {img_path:class, ...}
    '''
    def __init__(self, json_path, root=None, filter=not_none):
        self.root = root
        self.imgs = []
        self.labels = []
        if isinstance(json_path, dict):
            data = json_path
        else:
            data = json.load(open(json_path))
            assert isinstance(data, dict), 'json content is not a dictionary'

        for img, cls in data.items():
            assert type(img) is str
            if not filter(cls):
                continue
            if type(cls) not in (str, int, type(None)):
                continue
            self.imgs.append(img)
            self.labels.append(cls)

        self.find_classes()
        self.nimg = len(self.imgs)
        self.nquery = 0

    def get_key(self, i):
        return self.imgs[i]

    def get_label(self, i, toint=False):
        label = self.labels[i]
        if toint:
            label = self.cls_idx[label]
        return label


class NullCluster(ImageClusters):
    ''' Select only images with null label
    '''
    def __init__(self, json_path, root=None):
        ImageClusters.__init__(self, json_path, root, lambda c: c is None)
