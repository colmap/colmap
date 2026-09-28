import os
import json
import pdb
import numpy as np
from collections import defaultdict


class Dataset(object):
    ''' Base class for a dataset. To be overloaded.

        Contains:
            - images                --> get_image(i) --> image
            - image labels          --> get_label(i)
            - list of image queries --> get_query(i) --> image
            - list of query ROIs    --> get_query_roi(i)

        Creation:
            Use dataset.create( "..." ) to instanciate one.
            db = dataset.create( "ImageList('path/to/list.txt')" )

        Attributes:
            root:       image directory root
            nimg:       number of images == len(self)
            nclass:     number of classes
    '''
    root = ''
    img_dir = ''
    nimg = 0
    nclass = 0
    ninstance = 0

    classes = []    # all class names (len == nclass)
    labels = []     # all image labels (len == nimg)
    c_relevant_idx = {} # images belonging to each class (c_relevant_idx[cl_name] = [idx list])

    def __len__(self):
        return self.nimg

    def get_filename(self, img_idx, root=None):
        return os.path.join(root or self.root, self.img_dir, self.get_key(img_idx))

    def get_key(self, img_idx):
        raise NotImplementedError()

    def key_to_index(self, key):
        if not hasattr(self, '_key_to_index'):
            self._key_to_index = {self.get_key(i):i for i in range(len(self))}
        return self._key_to_index[key]

    def get_image(self, img_idx, resize=None):
        from PIL import Image
        img = Image.open(self.get_filename(img_idx)).convert('RGB')
        if resize:
            img = img.resize(resize, Image.ANTIALIAS if np.prod(resize) < np.prod(img.size) else Image.BICUBIC)
        return img

    def get_image_size(self, img_idx):
        return self.imsize

    def get_label(self, img_idx, toint=False):
        raise NotImplementedError()

    def has_label(self):
        try: self.get_label(0); return True
        except NotImplementedError: return False

    def get_query_db(self):
        raise NotImplementedError()

    def get_query_groundtruth(self, query_idx, what='AP'):
        query_db = self.get_query_db()
        assert self.nclass == query_db.nclass
        if what == 'AP':
            res = -np.ones(self.nimg, dtype=np.int8) # all negatives
            res[self.c_relevant_idx[query_db.get_label(query_idx)]] = 1 # positives
            if query_db == self: res[query_idx] = 0 # query is junk
        elif what == 'label':
            res = query_db.get_label(query_idx)
        else:
            raise ValueError("Unknown ground-truth type: %s" % what)
        return res

    def eval_query_AP(self, query_idx, scores):
        """ Evaluates AP for a given query.
        """
        from ..utils.evaluation import compute_AP
        gt = self.get_query_groundtruth(query_idx, 'AP') # labels in {-1, 0, 1}
        assert gt.shape == scores.shape, "scores should have shape %s" % str(gt.shape)
        assert -1 <= gt.min() and gt.max() <= 1, "bad ground-truth labels"
        keep = (gt != 0)  # remove null labels
        if sum(gt[keep]>0) == 0: return -1 # exclude queries with no relevants form the evaluation
        return compute_AP(gt[keep]>0, scores[keep])

    def eval_query_top(self, query_idx, scores, k=(1,5,10,20,50,100)):
        """ Evaluates top-k for a given query.
        """
        if not self.labels: raise NotImplementedError()
        q_label = self.get_query_groundtruth(query_idx, 'label')
        correct = np.bool8([l==q_label for l in self.labels])
        correct = correct[(-scores).argsort()]
        return {k_:float(correct[:k_].any()) for k_ in k if k_<len(correct)}

    def original(self):
        return self # overload this when the dataset is derived from another one

    def __repr__(self):
        res =  'Dataset: %s\n' % self.__class__.__name__
        res += '  %d images' % len(self)
        if self.nclass: res += ", %d classes" % (self.nclass)
        if self.ninstance: res += ', %d instances' % (self.ninstance)
        try:
            res += ', %d queries' % (self.get_query_db().nimg)
        except NotImplementedError:
            pass
        res += '\n  root: %s...' % self.root
        return res





def split( dataset, train_prop, val_prop=0, method='balanced' ):
    ''' Split a dataset into several subset:
        train, val and test

        method = hash:
            Split are reliable, i.e. unaffected by adding/removing images.
            But some clusters might be uneven (not respecting props well)
        method = balanced:
            splits are balanced (they respect props well), but not
            stable to modifications of the dataset.

        Returns:
            (train, val, test)
            if val_prop==0: return (train, test)
    '''
    assert 0 <= train_prop <= 1
    assert 0 <= val_prop < 1
    assert train_prop + val_prop <= 1

    train = []
    val = []
    test = []

    # redefine hash(), because built-in is not session-consistent anymore
    import hashlib
    hash = lambda x: int(hashlib.md5(bytes(x,"ascii")).hexdigest(),16)

    if method == 'balanced':
        test_prop = 1 - train_prop - val_prop

        perclass = [[] for i in range(dataset.nclass)]
        for i in range(len(dataset)):
            label = dataset.get_label(i, toint=True)
            h = hash(dataset.get_key(i))
            perclass[label].append( (h,i) )

        for imgs in perclass:
            nn = len(imgs)
            imgs.sort() # randomize order consistently with hash
            if imgs:
                imgs = list(list(zip(*imgs))[1]) # discard hash
            if imgs and train_prop > 0:
                train.append( imgs.pop() ) # ensure at least 1 training sample
            for i in range(int(0.9999+val_prop*nn)):
                if imgs: val.append( imgs.pop() )
            for i in range(int(0.9999+test_prop*nn)):
                if imgs: test.append( imgs.pop() )
            if imgs: train += imgs

        train.sort()
        val.sort()
        test.sort()

    elif method == 'hash':
        val_prop2 = train_prop + val_prop
        for i in range(len(dataset)):
            fname = dataset.get_key(i)

            # compute file hash
            h = (hash(fname)%100)/100.0
            if h < train_prop:
                train.append( i )
            elif h < val_prop2:
                val.append( i )
            else:
                test.append( i )
    else:
        raise ValueError("bad split method "+method)

    train = SubDataset(dataset, train)
    val = SubDataset(dataset, val)
    test = SubDataset(dataset, test)

    if val_prop == 0:
        return train, test
    else:
        return train, val, test


class SubDataset(Dataset):
    ''' Contains a sub-part of another dataset.
    '''
    def __init__(self, dataset, indices):
        self.root = dataset.root
        self.img_dir = dataset.img_dir
        self.dataset = dataset
        self.indices = indices

        self.nimg = len(self.indices)
        self.nclass = self.dataset.nclass

    def get_key(self, i):
        return self.dataset.get_key(self.indices[i])

    def get_label(self, i, **kw):
        return self.dataset.get_label(self.indices[i],**kw)

    def get_bbox(self, i, **kw):
        if hasattr(self.dataset,'get_bbox'):
            return self.dataset.get_bbox(self.indices[i],**kw)
        else:
            raise NotImplementedError()

    def __repr__(self):
        res =  'SubDataset(%s)\n' % self.dataset.__class__.__name__
        res += '  %d/%d images, %d classes\n' % (len(self),len(self.dataset),self.nclass)
        res += '  root: %s...' % os.path.join(self.root,self.img_dir)
        return res

    def viz_distr(self):
        from matplotlib import pyplot as pl; pl.ion()
        count = [0]*self.nclass
        for i in range(self.nimg):
            count[ self.get_label(i,toint=True) ] += 1
        cid = list(range(self.nclass))
        pl.bar(cid, count)
        pdb.set_trace()


class CatDataset(Dataset):
    ''' Concatenation of several datasets.
    '''
    def __init__(self, *datasets):
        assert len(datasets) >= 1
        self.datasets = datasets

        db = datasets[0]
        self.root = os.path.normpath(os.path.join(db.root, db.img_dir)) + os.sep
        self.labels = self.imgs = None # cannot access it the normal way
        self.classes = db.classes
        self.nclass = db.nclass
        self.c_relevant_idx = defaultdict(list)

        offsets = [0]
        full_root = lambda db: os.path.normpath(os.path.join(db.root, db.img_dir))
        for db in datasets:
            assert db.nclass == self.nclass, 'All dataset must have the same number of classes'
            assert db.classes == self.classes, 'All datasets must have the same classes'

            # look for a common root
            self.root = os.path.commonprefix((self.root, full_root(db) + os.sep))
            assert self.root, 'no common root between datasets'
            self.root = self.root[:self.root.rfind(os.sep)] + os.sep

            offset = sum(offsets)
            for label, rel in db.c_relevant_idx.items():
                self.c_relevant_idx[label] += [i+offset for i in rel]
            offsets.append(db.nimg)

        self.roots = [full_root(db)[len(self.root):] for db in datasets]
        self.offsets = np.cumsum(offsets)
        self.nimg = self.offsets[-1]

    def which(self, i):
        pos = np.searchsorted(self.offsets, i, side='right')-1
        assert pos < self.nimg, 'Bad image index %d >= %d' % (i, self.nimg)
        return pos, i - self.offsets[pos]

    def get(self, i, attr):
        b, j = self.which(i)
        return getattr(self.datasets[b],attr)

    def __getattr__(self, name):
        # try getting it
        val = getattr(self.datasets[0], name)
        assert not callable(val), 'CatDataset: %s is not a shared attribute, use call()' % str(name)
        for db in self.datasets[1:]:
            assert np.all(val == getattr(db, name)), 'CatDataset: inconsistent shared attribute %s, use get()' % str(name)
        return val

    def call(self, i, func, *args, **kwargs):
        b, j = self.which(i)
        return getattr(self.datasets[b],attr)(j,*args, **kwargs)

    def get_key(self, i):
        b, i = self.which(i)
        key = self.datasets[b].get_key(i)
        return os.path.join(self.roots[b], key)

    def get_label(self, i, toint=False):
        b, i = self.which(i)
        return self.datasets[b].get_label(i,toint=toint)

    def get_bbox(self,i):
        b, i = self.which(i)
        return self.datasets[b].get_bbox(i)

    def get_polygons(self,i,**kw):
        b, i = self.which(i)
        return self.datasets[b].get_polygons(i,**kw)




def deploy( dataset, target_dir, transforms=None, redo=False, ext=None, **savekwargs):
    if not target_dir: return dataset
    from PIL import Image
    from fcntl import flock, LOCK_EX
    import tqdm

    if transforms is not None:
        # identify transform with a unique hash
        import hashlib
        def get_params(trf):
            if type(trf).__name__ == 'Compose':
                return [get_params(t) for t in trf.transforms]
            else:
                return {type(trf).__name__:vars(trf)}
        params = get_params(transforms)
        unique_key = json.dumps(params, sort_keys=True).encode('utf-8')
        h = hashlib.md5().hexdigest()
        target_dir = os.path.join(target_dir, h)
    print("Deploying in '%s'" % target_dir)

    try:
        imsizes_path = os.path.join(target_dir,'imsizes.json')
        imsize_file = open(imsizes_path,'r+')
        #print("opening %s in r+ mode"%imsize_file)
    except IOError:
        try: os.makedirs(os.path.split(imsizes_path)[0])
        except OSError: pass
        imsize_file = open(imsizes_path,'w+')
        #print("opening %s in w+ mode"%imsize_file)

    # block access to this file, only one process can continue
    from time import time as now
    t0 = now()
    flock(imsize_file, LOCK_EX)
    #print("exclusive access lock for %s after %ds"%(imsize_file,now()-t0))

    try:
        imsizes = json.load(imsize_file)
        imsizes = {img:tuple(size) for img,size in imsizes.items()}
    except:
        imsizes = {}

    def check_one_image(i):
        key = dataset.get_key(i)
        target = os.path.join(target_dir, key)
        if ext: target = os.path.splitext(target)[0]+'.'+ext

        updated = 0
        if redo or (not os.path.isfile(target)) or key not in imsizes:
            # load image and transform it
            img = Image.open(dataset.get_filename(i)).convert('RGB')
            imsizes[key] = img.size

            if transforms is not None:
                img = transforms(img)

            odir = os.path.split( target )[0]
            try: os.makedirs(odir)
            except FileExistsError: pass
            img.save( target, **savekwargs )

            updated = 1
            if (i % 100) == 0:
                imsize_file.seek(0) # goto begining
                json.dump(dict(imsizes), imsize_file)
                imsize_file.truncate()
                updated = 0

        return updated

    from nltools.gutils import job_utils
    for i in range(len(dataset)):
        updated = check_one_image(i) # first try without any threads
        if updated: break
    if i+1 < len(dataset):
        updated += sum(job_utils.parallel_threads(range(i+1,len(dataset)), check_one_image,
                desc='Deploying dataset', n_threads=0, front_num=0))

    if updated:
        imsize_file.seek(0) # goto begining
        json.dump(dict(imsizes), imsize_file)
        imsize_file.truncate()
        imsize_file.close() # now, other processes can access too

    return DeployedDataset(dataset, target_dir, imsizes, trfs=transforms, ext=ext)



class DeployedDataset(Dataset):
    '''Just a deployed dataset with a different root and image extension.
    '''
    def __init__(self, dataset, root, imsizes=None, trfs=None, ext=None):
        self.dataset = dataset
        if root[-1] != '/': root += '/'
        self.root = root
        self.ext = ext
        self.imsizes = imsizes or json.load(open(root+'imsizes.json'))
        self.trfs = trfs or (lambda x: x)
        assert isinstance(self.imsizes, dict)
        assert len(self.imsizes) >= dataset.nimg, pdb.set_trace()

        self.nimg = dataset.nimg
        self.nclass = dataset.nclass

        self.labels = dataset.labels
        self.c_relevant_idx = dataset.c_relevant_idx
        #self.c_non_relevant_idx = dataset.c_non_relevant_idx

        self.get_label = dataset.get_label
        self.classes = dataset.classes
        if '/query_db/' not in root:
            try:
                query_db = dataset.get_query_db()
                if query_db is not dataset:
                    self.query_db = deploy(query_db, os.path.join(root,'query_db'), transforms=trfs, ext=ext)
                    self.get_query_db = lambda: self.query_db
            except NotImplementedError:
                pass
        self.get_query_groundtruth = dataset.get_query_groundtruth
        if hasattr(dataset, 'eval_query_AP'):
            self.eval_query_AP = dataset.eval_query_AP

        if hasattr(dataset, 'true_pairs'):
            self.true_pairs = dataset.true_pairs
            self.get_false_pairs = dataset.get_false_pairs

    def __repr__(self):
        res =  self.dataset.__repr__()
        res += '  deployed at %s/...%s' % (self.root, self.ext or '')
        return res

    def __len__(self):
        return self.nimg

    def get_key(self, i):
        key = self.dataset.get_key(i)
        if self.ext:  key = os.path.splitext(key)[0]+'.'+self.ext
        return key

    def get_something(self, what, i, *args, **fmt):
        try:
            get_func = getattr(self.dataset, 'get_'+what)
        except AttributeError:
            raise NotImplementedError()
        imsize = self.imsizes[self.dataset.get_key(i)]
        sth = get_func(i,*args,**fmt)
        return self.trfs({'imsize':imsize, what:sth})[what]

    def get_bbox(self, i, **kw):
        return self.get_something('bbox', i, **kw)

    def get_polygons(self, i, *args, **kw):
        return self.get_something('polygons', i, *args, **kw)

    def get_label_map(self, i, *args, **kw):
        assert 'polygons' in kw, "you need to supply polygons because image has been transformed"
        return self.dataset.get_label_map(i, *args, **kw)
    def get_instance_map(self, i, *args, **kw):
        assert 'polygons' in kw, "you need to supply polygons because image has been transformed"
        return self.dataset.get_instance_map(i, *args, **kw)
    def get_angle_map(self, i, *args, **kw):
        assert 'polygons' in kw, "you need to supply polygons because image has been transformed"
        return self.dataset.get_angle_map(i, *args, **kw)

    def original(self):
        return self.dataset



def deploy_and_split( trainset, deploy_trf=None, deploy_dir='/dev/shm',
                      valset=None, split_val=0.0,
                      img_ext='jpg', img_quality=95,
                      **_useless ):
    ''' Deploy and split a dataset into train / val.
    if valset is not provided, then trainset is automatically split into train/val
    based on the split_val proportion.
    '''
    # first, deploy the training set
    traindb = deploy( trainset, deploy_dir, transforms=deploy_trf, ext=img_ext, quality=img_quality )

    if valset:
        # load a validation db
        valdb = deploy( valset, deploy_dir, transforms=deploy_trf, ext=img_ext, quality=img_quality )

    else:
        if split_val > 0:
            # automatic split in train/val
            traindb, valdb = split( traindb, train_prop=1-split_val )
        else:
            valdb = None

    print( "\n>> Training set:" ); print( traindb )
    print( "\n>> Validation set:" ); print( valdb )
    return traindb, valdb




class CropDataset(Dataset):
    """list_of_imgs_and_crops = [(img_key, (l, t, r, b)), ...]
    """
    def __init__(self, dataset, list_of_imgs_and_crops):
        self.dataset = dataset
        self.root = dataset.root
        self.img_dir = dataset.img_dir
        self.imgs, self.crops = zip(*list_of_imgs_and_crops)
        self.nimg = len(self.imgs)

    def get_image(self, img_idx):
        # even if the image have multiple signage polygon?
        org_img = dataset.get_image(self, img_idx)
        crop_signs = crop_image(org_img, self.crops[img_idx])

        return crop_signs[0] # temporary use one, but have to change for multiple signages

    def get_filename(self, img_idx):
        return self.dataset.get_filename(img_idx)

    def get_key(self, img_idx):
        return self.dataset.get_key(img_idx)

    def crop_image(self, img, polygons):
        import cv2
        crop_signs=[]
        if len(polygons)==0:
            pdb.set_trace()

        for Polycc in polygons:
            rgbimg = img.copy()
            rgbimg = np.array(rgbimg) # pil to cv2
            Poly_s = np.array(Polycc)

            ## rearrange
            if Poly_s[0, 1]<Poly_s[1, 1]:
                temp = Poly_s[1, :].copy()
                Poly_s[1, :]= Poly_s[0, :]
                Poly_s[0, :]=temp

            if Poly_s[2, 1]>Poly_s[3, 1]:
                temp = Poly_s[3, :].copy()
                Poly_s[3, :]= Poly_s[2, :]
                Poly_s[2, :]=temp

            cy_s = np.mean( Poly_s[:,0] )
            cx_s = np.mean( Poly_s[:,1] )
            w_s = np.abs( Poly_s[0][1]-Poly_s[1][1] )
            h_s = np.abs( Poly_s[0][0]-Poly_s[2][0] )
            Poly_d = np.array([(cy_s-h_s/2, cx_s+w_s/2), (cy_s-h_s/2, cx_s-w_s/2), (cy_s+h_s/2, cx_s-w_s/2), (cy_s+h_s/2, cx_s+w_s/2)]).astype(np.int)

            M, mask= cv2.findHomography(Poly_s, Poly_d)

            warpimg = Image.fromarray(cv2.warpPerspective(rgbimg, M, (645,800))) # from cv2 type rgbimg
            crop_sign = warpimg.crop([np.min(Poly_d[:,0]), np.min(Poly_d[:,1]), np.max(Poly_d[:,0]), np.max(Poly_d[:,1])])

            ### append
            crop_signs.append(crop_sign)

        return crop_signs














