# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import os, pdb
import numpy as np
from PIL import Image

from .dataset import Dataset, CatDataset
from tools.transforms import instanciate_transformation
from tools.transforms_tools import persp_apply


class PairDataset (Dataset):
    """ A dataset that serves image pairs with ground-truth pixel correspondences.
    """
    def __init__(self):
        Dataset.__init__(self)
        self.npairs = 0

    def get_filename(self, img_idx, root=None):
        if is_pair(img_idx): # if img_idx is a pair of indices, we return a pair of filenames
            return tuple(Dataset.get_filename(self, i, root) for i in img_idx)
        return Dataset.get_filename(self, img_idx, root)

    def get_image(self, img_idx):
        if is_pair(img_idx): # if img_idx is a pair of indices, we return a pair of images
            return tuple(Dataset.get_image(self, i) for i in img_idx)
        return Dataset.get_image(self, img_idx)

    def get_corres_filename(self, pair_idx):
        raise NotImplementedError()

    def get_homography_filename(self, pair_idx):
        raise NotImplementedError()

    def get_flow_filename(self, pair_idx):
        raise NotImplementedError()

    def get_mask_filename(self, pair_idx):
        raise NotImplementedError()

    def get_pair(self, idx, output=()):
        """ returns (img1, img2, `metadata`)
        
        `metadata` is a dict() that can contain:
            flow: optical flow
            aflow: absolute flow
            corres: list of 2d-2d correspondences
            mask: boolean image of flow validity (in the first image)
            ...
        """
        raise NotImplementedError()

    def get_paired_images(self):
        fns = set()
        for i in range(self.npairs):
            a,b = self.image_pairs[i]
            fns.add(self.get_filename(a))
            fns.add(self.get_filename(b))
        return fns

    def __len__(self):
        return self.npairs # size should correspond to the number of pairs, not images
    
    def __repr__(self):
        res =  'Dataset: %s\n' % self.__class__.__name__
        res += '  %d images,' % self.nimg
        res += ' %d image pairs' % self.npairs
        res += '\n  root: %s...\n' % self.root
        return res

    @staticmethod
    def _flow2png(flow, path):
        flow = np.clip(np.around(16*flow), -2**15, 2**15-1)
        bytes = np.int16(flow).view(np.uint8)
        Image.fromarray(bytes).save(path)
        return flow / 16

    @staticmethod
    def _png2flow(path):
        try:
            flow = np.asarray(Image.open(path)).view(np.int16)
            return np.float32(flow) / 16
        except:
            raise IOError("Error loading flow for %s" % path)



class StillPairDataset (PairDataset):
    """ A dataset of 'still' image pairs.
        By overloading a normal image dataset, it appends the get_pair(i) function
        that serves trivial image pairs (img1, img2) where img1 == img2 == get_image(i).
    """
    def get_pair(self, pair_idx, output=()):
        if isinstance(output, str): output = output.split()
        img1, img2 = map(self.get_image, self.image_pairs[pair_idx])

        W,H = img1.size
        sx = img2.size[0] / float(W)
        sy = img2.size[1] / float(H)

        meta = {}
        if 'aflow' in output or 'flow' in output:
            mgrid = np.mgrid[0:H, 0:W][::-1].transpose(1,2,0).astype(np.float32)
            meta['aflow'] = mgrid * (sx,sy)
            meta['flow'] = meta['aflow'] - mgrid

        if 'mask' in output:
            meta['mask'] = np.ones((H,W), np.uint8)

        if 'homography' in output:
            meta['homography'] = np.diag(np.float32([sx, sy, 1]))

        return img1, img2, meta



class SyntheticPairDataset (PairDataset):
    """ A synthetic generator of image pairs.
        Given a normal image dataset, it constructs pairs using random homographies & noise.
    """
    def __init__(self, dataset, scale='', distort=''):
        self.attach_dataset(dataset)
        self.distort = instanciate_transformation(distort)
        self.scale = instanciate_transformation(scale)

    def attach_dataset(self, dataset):
        assert isinstance(dataset, Dataset) and not isinstance(dataset, PairDataset)
        self.dataset = dataset
        self.npairs = dataset.nimg
        self.get_image = dataset.get_image
        self.get_key = dataset.get_key
        self.get_filename = dataset.get_filename
        self.root = None
        
    def make_pair(self, img):
        return img, img

    def get_pair(self, i, output=('aflow')):
        """ Procedure:
        This function applies a series of random transformations to one original image 
        to form a synthetic image pairs with perfect ground-truth.
        """
        if isinstance(output, str): 
            output = output.split()
            
        original_img = self.dataset.get_image(i)
        
        scaled_image = self.scale(original_img)
        scaled_image, scaled_image2 = self.make_pair(scaled_image)
        scaled_and_distorted_image = self.distort(
            dict(img=scaled_image2, persp=(1,0,0,0,1,0,0,0)))
        W, H = scaled_image.size
        trf = scaled_and_distorted_image['persp']

        meta = dict()
        if 'aflow' in output or 'flow' in output:
            # compute optical flow
            xy = np.mgrid[0:H,0:W][::-1].reshape(2,H*W).T
            aflow = np.float32(persp_apply(trf, xy).reshape(H,W,2))
            meta['flow'] = aflow - xy.reshape(H,W,2)
            meta['aflow'] = aflow
        
        if 'homography' in output:
            meta['homography'] = np.float32(trf+(1,)).reshape(3,3)

        return scaled_image, scaled_and_distorted_image['img'], meta
    
    def __repr__(self):
        res =  'Dataset: %s\n' % self.__class__.__name__
        res += '  %d images and pairs' % self.npairs
        res += '\n  root: %s...' % self.dataset.root
        res += '\n  Scale: %s' % (repr(self.scale).replace('\n',''))
        res += '\n  Distort: %s' % (repr(self.distort).replace('\n',''))
        return res + '\n'



class TransformedPairs (PairDataset):
    """ Automatic data augmentation for pre-existing image pairs.
        Given an image pair dataset, it generates synthetically jittered pairs
        using random transformations (e.g. homographies & noise).
    """
    def __init__(self, dataset, trf=''):
        self.attach_dataset(dataset)
        self.trf = instanciate_transformation(trf)

    def attach_dataset(self, dataset):
        assert isinstance(dataset, PairDataset)
        self.dataset = dataset
        self.nimg = dataset.nimg
        self.npairs = dataset.npairs
        self.get_image = dataset.get_image
        self.get_key = dataset.get_key
        self.get_filename = dataset.get_filename
        self.root = None
        
    def get_pair(self, i, output=''):
        """ Procedure:
        This function applies a series of random transformations to one original image 
        to form a synthetic image pairs with perfect ground-truth.
        """
        img_a, img_b_, metadata = self.dataset.get_pair(i, output)

        img_b = self.trf({'img': img_b_, 'persp':(1,0,0,0,1,0,0,0)})
        trf = img_b['persp']

        if 'aflow' in metadata or 'flow' in metadata:
            aflow = metadata['aflow']
            aflow[:] = persp_apply(trf, aflow.reshape(-1,2)).reshape(aflow.shape)
            W, H = img_a.size
            flow = metadata['flow']
            mgrid = np.mgrid[0:H, 0:W][::-1].transpose(1,2,0).astype(np.float32)
            flow[:] = aflow - mgrid

        if 'corres' in metadata:
            corres = metadata['corres']
            corres[:,1] = persp_apply(trf, corres[:,1])
        
        if 'homography' in metadata:
            # p_b = homography * p_a
            trf_ = np.float32(trf+(1,)).reshape(3,3)
            metadata['homography'] = np.float32(trf_ @ metadata['homography'])

        return img_a, img_b['img'], metadata

    def __repr__(self):
        res =  'Transformed Pairs from %s\n' % type(self.dataset).__name__
        res += '  %d images and pairs' % self.npairs
        res += '\n  root: %s...' % self.dataset.root
        res += '\n  transform: %s' % (repr(self.trf).replace('\n',''))
        return res + '\n'



class CatPairDataset (CatDataset):
    ''' Concatenation of several pair datasets.
    '''
    def __init__(self, *datasets):
        CatDataset.__init__(self, *datasets)
        pair_offsets = [0]
        for db in datasets:
            pair_offsets.append(db.npairs)
        self.pair_offsets = np.cumsum(pair_offsets)
        self.npairs = self.pair_offsets[-1]

    def __len__(self):
        return self.npairs

    def __repr__(self):
        fmt_str = "CatPairDataset("
        for db in self.datasets:
            fmt_str += str(db).replace("\n"," ") + ', '
        return fmt_str[:-2] + ')'

    def pair_which(self, i):
        pos = np.searchsorted(self.pair_offsets, i, side='right')-1
        assert pos < self.npairs, 'Bad pair index %d >= %d' % (i, self.npairs)
        return pos, i - self.pair_offsets[pos]

    def pair_call(self, func, i, *args, **kwargs):
        b, j = self.pair_which(i)
        return getattr(self.datasets[b], func)(j, *args, **kwargs)

    def get_pair(self, i, output=()):
        b, i = self.pair_which(i)
        return self.datasets[b].get_pair(i, output)

    def get_flow_filename(self, pair_idx, *args, **kwargs):
        return self.pair_call('get_flow_filename', pair_idx, *args, **kwargs)

    def get_mask_filename(self, pair_idx, *args, **kwargs):
        return self.pair_call('get_mask_filename', pair_idx, *args, **kwargs)

    def get_corres_filename(self, pair_idx, *args, **kwargs):
        return self.pair_call('get_corres_filename', pair_idx, *args, **kwargs)



def is_pair(x):
    if isinstance(x, (tuple,list)) and len(x) == 2:
        return True
    if isinstance(x, np.ndarray) and x.ndim == 1 and x.shape[0] == 2:
        return True
    return False

