import pdb

from PIL import Image
import numpy as np
import random

import torch
import torch.utils.data as data


def get_loader( dataset, trf_chain, iscuda,
                preprocess = {}, # variables for preprocessing (input_size, mean, std, ...)
                output = ('img','label'),
                batch_size = None,
                threads = 1,
                shuffle = True,
                balanced = 0, use_all = False,
                totensor = True,
                **_useless_kw):
    ''' Get a data loader, given the dataset and some parameters.

    Parameters
    ----------
    dataset : Dataset().
        Class containing all images and labels.

    trf_chain : list
        list of transforms

    iscuda : bool

    output : tuple of str
        tells what to return. 'img', 'label', ... See PytorchLoader().

    preprocess : dict
        {input_size:..., mean=..., std:..., ...}

    batch_size : int

    threads : int

    shuffle : int

    balanced : float in [0,1]
        if balanced>0, then will pick dataset samples such that each class is equally represented.

    use_all : bool
        if True, will force to use all dataset samples at least once (even if balanced>0)

    Returns
    -------
        a pytorch loader.
    '''
    from . import transforms
    trf_chain = transforms.create(trf_chain, to_tensor=True, **preprocess)

    sampler = None
    if balanced:
        sampler = BalancedSampler(dataset, use_all=use_all, balanced=balanced)
        shuffle = False

    loader = PytorchLoader(dataset, transform=trf_chain, output=output)

    if threads == 1:
        return loader
    else:
        return data.DataLoader(
            loader,
            batch_size = batch_size,
            shuffle = shuffle,
            sampler = sampler,
            num_workers = threads,
            pin_memory = iscuda)




class PytorchLoader (data.Dataset):
    """A pytorch dataset-loader

     Args:
        dataset (object):  dataset inherited from dataset.Dataset()

        transform (deprecated, callable): pytorch transforms. Use img_and_target_transform instead.

        target_transform (deprecated, callable): applied on target. Use img_and_target_transform instead.

        img_and_target_transform (callable):
                applied on dict(img=, label=, bbox=, ...)
                and should return a similar dictionary.

     Attributes:
        dataset (object): subclass of dataset.Dataset()
    """

    def __init__(self, dataset, transform=None,
                       target_transform=None,
                       img_and_target_transform=None,
                       output=['img','label']):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.img_and_target_transform = img_and_target_transform
        self.output = output

    def __getitem__(self, index):
        img_filename = self.dataset.get_filename(index)

        img_and_label = dict(
            img_filename = img_filename,
            img_key   = self.dataset.get_key(index),
            img   = self.dataset.get_image(index),
            label = try_to_get(self.dataset.get_label, index, toint=True) )

        if self.img_and_target_transform:
            # label depends on image (bbox, polygons, etc)
            assert self.transform is None
            assert self.target_transform is None

            # add optional attributes
            if 'bbox' in self.output:
                bbox = try_to_get(self.dataset.get_bbox, index)
                if bbox: img_and_label['bbox'] = bbox

            if any(a.endswith('_map') for a in self.output):
                original_polygons = try_to_get(self.dataset.get_polygons, index, toint=True)
                if original_polygons is not None:
                    img_and_label['polygons'] = original_polygons

            img_and_label = self.img_and_target_transform(img_and_label)

            if original_polygons is not None:
                transformed_polygons = img_and_label['polygons']

            imsize = img_and_label['img'].size
            if not isinstance(imsize, tuple):
                imsize = imsize()[-2:][::-1] # returns h,w

            if 'label_map' in self.output:
                pixlabel = self.dataset.get_label_map(index, imsize, polygons=transformed_polygons)
                img_and_label['label_map'] = pixlabel.astype(int)

            # instance level attributes
            for out_key in self.output:
                for type in ['_instance_map', '_angle_map']:
                    if not out_key.endswith(type): continue
                    cls = out_key[:-len(type)]
                    get_func = getattr(self.dataset,'get'+type)
                    pixlabel = get_func(index, cls, imsize, polygons=transformed_polygons)
                    img_and_label[out_key] = pixlabel
        else:
            # just plain old transform, no influence on labels

            if self.transform is not None:
                img_and_label['img'] = self.transform(img_and_label['img'])

            if self.target_transform:
                img_and_label['label'] = self.target_transform(img_and_label['label'])

        for o in self.output:
            assert img_and_label.get(o) is not None, "Missing field %s for img %s" % (o,img_filename)
        return [img_and_label[o] for o in self.output]

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.dataset.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: %d\n' % len(self.dataset)
        fmt_str += '    Root Location: %s\n' % self.dataset.__dict__.get('root','(unknown)')
        if self.img_and_target_transform:
            tmp = '    Image_and_target transforms: '
            fmt_str += '{0}{1}\n'.format(tmp, repr(self.img_and_target_transform).replace('\n', '\n' + ' ' * len(tmp)))
        if self.transform:
            tmp = '    Image transforms: '
            fmt_str += '{0}{1}\n'.format(tmp, repr(self.transform).replace('\n', '\n' + ' ' * len(tmp)))
        if self.target_transform:
            tmp = '    Target transforms: '
            fmt_str += '{0}{1}\n'.format(tmp, repr(self.target_transform).replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



class BalancedSampler (data.sampler.Sampler):
    """ Data sampler that will provide an equal number of each class
    to the network.

    size:   float in [0,2]
        specify the size increase/decrease w.r.t to the original dataset.
        1 means that the over-classes (with more than median n_per_class images)
        will have less items, but conversely, under-classes will have more items.

    balanced:  float in [0,1]
        specify whether the balance constraint should be respected firmly or not.
        if ==1: balance is exactly respected; if ==0, same as dataset (no change).

    use_all:    bool
        if True, will use all images that a class have, even if it is higher than
        what the algorithm wanted to use.
    """

    def __init__(self, dataset, size=1.0, balanced=1.0, use_all=False):
        assert 0 <= size <= 2
        assert 0 <= balanced <= 1

        # enumerate class images
        self.cls_imgs = [[] for i in range(dataset.nclass)]
        for i in range(len(dataset)):
            label = dataset.get_label(i, toint=True)
            self.cls_imgs[label].append(i)

        # decide on the number of example per class
        self.npc = np.percentile([len(imgs) for imgs in self.cls_imgs], max(0,min(50*size,100)))

        self.balanced = balanced
        self.use_all = use_all

        self.nelem = int(0.5 + self.npc * dataset.nclass)   # initial estimate

    def __iter__(self):
        indices = []
        for i,imgs in enumerate(self.cls_imgs):
            np.random.shuffle(imgs)  # randomize

            # target size for this class
            # target = logarithmic mean
            b = self.balanced
            if len(imgs):
                target = 2**(b*np.log2(self.npc) + (1-b)*np.log2(len(imgs)))
                target = int(0.5 + target)
            else:
                target = 0
            if self.use_all:    # use all images
                target = max(target, len(imgs))

            # augment classes until target
            res = []
            while len(res) < target:
                res += imgs
            res = res[:target] # cut

            indices += res

        np.random.shuffle(indices)
        self.nelem = len(indices)
        return iter(indices)

    def __len__(self):
        return self.nelem




### Helper functions with get_loader() and DatasetLoader()

def load_one_img( loader ):
    ''' Helper to iterate on get_loader()

    loader: output of get_loader()
    '''
    iterator = iter(loader)
    batch = []
    while iterator:
        if not batch: # refill
            things = next(iterator)
            batch = list(zip(*[t.numpy() if torch.is_tensor(t) else t for t in things]))
        yield batch.pop(0)


def tensor2img(tensor, model):
    """ convert a numpy tensor to a PIL Image
        (undo the ToTensor() and Normalize() transforms)
    """
    mean = model.preprocess['mean']
    std = model.preprocess['std']
    if not isinstance(tensor, np.ndarray):
        if not isinstance(tensor, torch.Tensor):
            tensor = tensor.data
        tensor = tensor.squeeze().cpu().numpy()

    res = np.uint8(np.clip(255*((tensor.transpose(1,2,0) * std) + mean), 0, 255))

    from PIL import Image
    return Image.fromarray(res)


def test_loader_speed(loader_):
    ''' Test the speed of a data loader
    '''
    from tqdm import tqdm
    loader = load_one_img(loader_)
    for _ in tqdm(loader):
        pass
    pdb.set_trace()



def try_to_get(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except NotImplementedError:
        return None
