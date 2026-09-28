import pdb
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as tvf
import random
from math import ceil

from . import transforms_tools as F


def create(cmd_line, to_tensor=False, **vars):
    ''' Create a sequence of transformations.

    cmd_line: (str)
        Comma-separated list of transformations.
        Ex: "Rotate(10), Scale(256)"

    to_tensor: (bool)
        Whether to add the "ToTensor(), Normalize(mean, std)"
        automatically to the end of the transformation string

    vars: (dict)
        dictionary of global variables.
    '''
    if to_tensor:
        if not cmd_line:
            cmd_line = "ToTensor(), Normalize(mean=mean, std=std)"
        elif to_tensor and 'ToTensor' not in cmd_line:
            cmd_line += ", ToTensor(), Normalize(mean=mean, std=std)"

    assert isinstance(cmd_line, str)

    cmd_line = "tvf.Compose([%s])" % cmd_line
    try:
        return eval(cmd_line, globals(), vars)
    except Exception as e:
        raise SyntaxError("Cannot interpret this transform list: %s\nReason: %s" % (cmd_line, e))


class Identity (object):
    """ Identity transform. It does nothing!
    """
    def __call__(self, inp):
        return inp


class Pad(object):
    """ Pads the shortest side of the image to a given size

    If size is shorter than the shortest image, then the image will be untouched
    """

    def __init__(self, size, color=(127,127,127)):
        self.size = size
        assert len(color) == 3
        if not all(isinstance(c,int) for c in color):
            color = tuple([int(255*c) for c in color])
        self.color = color

    def __call__(self, inp):
        img = F.grab_img(inp)
        w, h = img.size
        if w >= h:
            newh = max(h,self.size)
            neww = w
        else:
            newh = h
            neww = max(w,self.size)

        if (neww,newh) != img.size:
            img2 = Image.new('RGB', (neww,newh), self.color)
            img2.paste(img, ((neww-w)//2,(newh-h)//2) )
            img = img2

        return F.update_img_and_labels(inp, img, aff=(1,0,0,0,1,0))

class PadSquare (object):
    """ Pads the image to a square size

    The dimension of the output image will be equal to size x size

    If size is None, then the image will be padded to the largest dimension

    If size is smaller than the original image size, the image will be cropped
    """

    def __init__(self, size=None, color=(127,127,127)):
        self.size = size
        assert len(color) == 3
        if not all(isinstance(c,int) for c in color):
            color = tuple([int(255*c) for c in color])
        self.color = color

    def __call__(self, inp):
        img = F.grab_img(inp)
        w, h = img.size
        s = self.size or max(w, h)


        if (s,s) != img.size:
            img2 = Image.new('RGB', (s,s), self.color)
            img2.paste(img, ((s-w)//2,(s-h)//2) )
            img = img2

        return F.update_img_and_labels(inp, img, aff=(1,0,0,0,1,0))


class RandomBorder (object):
    """ Expands the image with a random size border
    """

    def __init__(self, min_size, max_size, color=(127,127,127)):
        assert isinstance(min_size, int) and min_size >= 0
        assert isinstance(max_size, int) and min_size <= max_size
        self.min_size = min_size
        self.max_size = max_size
        assert len(color) == 3
        if not all(isinstance(c,int) for c in color):
            color = tuple([int(255*c) for c in color])
        self.color = color

    def __call__(self, inp):
        img = F.grab_img(inp)

        bh = random.randint(self.min_size, self.max_size)
        bw = random.randint(self.min_size, self.max_size)

        img = ImageOps.expand(img, border=(bw,bh,bw,bh), fill=self.color)

        return F.update_img_and_labels(inp, img, aff=(1,0,0,0,1,0))


class Scale (object):
    """ Rescale the input PIL.Image to a given size.
        Same as torchvision.Scale

    The smallest dimension of the resulting image will be = size.

    if largest == True: same behaviour for the largest dimension.

    if not can_upscale: don't upscale
    if not can_downscale: don't downscale
    """
    def __init__(self, size, interpolation=Image.BILINEAR, largest=False, can_upscale=True, can_downscale=True):
        assert isinstance(size, (float,int)) or (len(size) == 2)
        self.size = size
        if isinstance(self.size, float):
            assert 0 < self.size <= 4, 'bad float self.size, cannot be outside of range ]0,4]'
        self.interpolation = interpolation
        self.largest = largest
        self.can_upscale = can_upscale
        self.can_downscale = can_downscale

    def get_params(self, imsize):
        w,h = imsize
        if isinstance(self.size, int):
            is_smaller = lambda a,b: (a>=b) if self.largest else (a<=b)
            if (is_smaller(w, h) and w == self.size) or (is_smaller(h, w) and h == self.size):
                ow, oh = w, h
            elif is_smaller(w, h):
                ow = self.size
                oh = int(0.5 + self.size * h / w)
            else:
                oh = self.size
                ow = int(0.5 + self.size * w / h)

        elif isinstance(self.size, float):
            ow, oh = int(0.5 + self.size*w), int(0.5 + self.size*h)

        else: # tuple of ints
            ow, oh = self.size
        return ow, oh

    def __call__(self, inp):
        img = F.grab_img(inp)
        w, h = img.size

        size2 = ow,oh = self.get_params(img.size)

        if size2 != img.size:
            a1, a2 = img.size, size2
            if (self.can_upscale and min(a1) < min(a2)) or (self.can_downscale and min(a1) > min(a2)):
                img = img.resize(size2, self.interpolation)

        return F.update_img_and_labels(inp, img, aff=(ow/w,0,0,0,oh/h,0))



class RandomScale (Scale):
    """Rescale the input PIL.Image to a random size.

    Args:
        min_size (int): min size of the smaller edge of the picture.
        max_size (int): max size of the smaller edge of the picture.

        ar (float or tuple):
            max change of aspect ratio (width/height).

        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, min_size, max_size, ar=1, can_upscale=False, can_downscale=True, interpolation=Image.BILINEAR, largest=False):
        Scale.__init__(self, 0, can_upscale=can_upscale, can_downscale=can_downscale, interpolation=interpolation, largest=largest)
        assert isinstance(min_size, int) and min_size >= 1
        assert isinstance(max_size, int) and min_size <= max_size
        self.min_size = min_size
        self.max_size = max_size
        if type(ar) in (float,int): ar = (min(1/ar,ar),max(1/ar,ar))
        assert 0.2 < ar[0] <= ar[1] < 5
        self.ar = ar
        self.largest = largest

    def get_params(self, imsize):
        w,h = imsize
        if self.can_upscale:
            max_size = self.max_size
        else:
            max_size = min(self.max_size,min(w,h))
        size = max(min(int(0.5 + F.rand_log_uniform(self.min_size,self.max_size)), self.max_size), self.min_size)
        ar = F.rand_log_uniform(*self.ar) # change of aspect ratio

        if not self.largest:
            if w < h : # image is taller
                ow = size
                oh = int(0.5 + size * h / w / ar)
                if oh < self.min_size:
                    ow,oh = int(0.5 + ow*float(self.min_size)/oh),self.min_size
            else: # image is wider
                oh = size
                ow = int(0.5 + size * w / h * ar)
                if ow < self.min_size:
                    ow,oh = self.min_size,int(0.5 + oh*float(self.min_size)/ow)
            assert ow >= self.min_size
            assert oh >= self.min_size
        else: # if self.largest
            if w > h: # image is wider
                ow = size
                oh = int(0.5 + size * h / w / ar)
            else: # image is taller
                oh = size
                ow = int(0.5 + size * w / h * ar)
            assert ow <= self.max_size
            assert oh <= self.max_size

        return ow, oh


class RandomCrop (object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        assert h >= th and w >= tw, "Image of %dx%d is too small for crop %dx%d" % (w,h,tw,th)

        y = np.random.randint(0, h - th) if h > th else 0
        x = np.random.randint(0, w - tw) if w > tw else 0
        return x, y, tw, th

    def __call__(self, inp):
        img = F.grab_img(inp)

        padl = padt = 0
        if self.padding > 0:
            if F.is_pil_image(img):
                img = ImageOps.expand(img, border=self.padding, fill=0)
            else:
                assert isinstance(img, F.DummyImg)
                img = img.expand(border=self.padding)
            if isinstance(self.padding, int):
                padl = padt = self.padding
            else:
                padl, padt = self.padding[0:2]

        i, j, tw, th = self.get_params(img, self.size)
        img = img.crop((i, j, i+tw, j+th))

        return F.update_img_and_labels(inp, img, aff=(1,0,padl-i,0,1,padt-j))



class CenterCrop (RandomCrop):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        y = int(0.5 +((h - th) / 2.))
        x = int(0.5 +((w - tw) / 2.))
        return x, y, tw, th



class CropToBbox(object):
    """ Crop the image according to the bounding box.

    margin (float):
        ensure a margin around the bbox equal to (margin * min(bbWidth,bbHeight))

    min_size (int):
        result cannot be smaller than this size
    """
    def __init__(self, margin=0.5, min_size=0):
        self.margin = margin
        self.min_size = min_size

    def __call__(self, inp):
        img = F.grab_img(inp)
        w, h = img.size

        assert min(w,h) >= self.min_size

        x0,y0,x1,y1 = inp['bbox']
        assert x0 < x1 and y0 < y1, pdb.set_trace()
        bbw, bbh = x1-x0, y1-y0
        margin = int(0.5 + self.margin * min(bbw, bbh))

        i = max(0, x0 - margin)
        j = max(0, y0 - margin)
        w = min(w, x1 + margin) - i
        h = min(h, y1 + margin) - j

        if w < self.min_size:
            i = max(0, i-(self.min_size-w)//2)
            w = self.min_size
        if h < self.min_size:
            j = max(0, j-(self.min_size-h)//2)
            h = self.min_size

        img = img.crop((i,j,i+w,j+h))

        return F.update_img_and_labels(inp, img, aff=(1,0,-i,0,1,-j))



class RandomRotation(object):
    """Rescale the input PIL.Image to a random size.

    Args:
        degrees (float):
            rotation angle.

        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, degrees, interpolation=Image.BILINEAR):
        self.degrees = degrees
        self.interpolation = interpolation

    def __call__(self, inp):
        img = F.grab_img(inp)
        w, h = img.size

        angle = np.random.uniform(-self.degrees, self.degrees)

        img = img.rotate(angle, resample=self.interpolation)
        w2, h2 = img.size

        aff = F.aff_translate(-w/2,-h/2)
        aff = F.aff_mul(aff, F.aff_rotate(-angle * np.pi/180))
        aff = F.aff_mul(aff, F.aff_translate(w2/2,h2/2))
        return F.update_img_and_labels(inp, img, aff=aff)


class RandomFlip (object):
    """Randomly flip the image.
    """
    def __call__(self, inp):
        img = F.grab_img(inp)
        w, h = img.size

        flip = np.random.rand() < 0.5
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        return F.update_img_and_labels(inp, img, aff=(-1,0,w-1,0,1,0))



class RandomTilting(object):
    """Apply a random tilting (left, right, up, down) to the input PIL.Image

    Args:
        maginitude (float):
            maximum magnitude of the random skew (value between 0 and 1)
        directions (string):
            tilting directions allowed (all, left, right, up, down)
            examples: "all", "left,right", "up-down-right"
    """

    def __init__(self, magnitude, directions='all'):
        self.magnitude = magnitude
        self.directions = directions.lower().replace(',',' ').replace('-',' ')

    def __call__(self, inp):
        img = F.grab_img(inp)
        w, h = img.size

        x1,y1,x2,y2 = 0,0,h,w
        original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]

        max_skew_amount = max(w, h)
        max_skew_amount = int(ceil(max_skew_amount * self.magnitude))
        skew_amount = random.randint(1, max_skew_amount)

        if self.directions == 'all':
            choices = [0,1,2,3]
        else:
            dirs = ['left', 'right', 'up', 'down']
            choices = []
            for d in self.directions.split():
                try:
                    choices.append(dirs.index(d))
                except:
                    raise ValueError('Tilting direction %s not recognized' % d)

        skew_direction = random.choice(choices)

        if skew_direction == 0:
            # Left Tilt
            new_plane = [(y1, x1 - skew_amount),  # Top Left
                         (y2, x1),                # Top Right
                         (y2, x2),                # Bottom Right
                         (y1, x2 + skew_amount)]  # Bottom Left
        elif skew_direction == 1:
            # Right Tilt
            new_plane = [(y1, x1),                # Top Left
                         (y2, x1 - skew_amount),  # Top Right
                         (y2, x2 + skew_amount),  # Bottom Right
                         (y1, x2)]                # Bottom Left
        elif skew_direction == 2:
            # Forward Tilt
            new_plane = [(y1 - skew_amount, x1),  # Top Left
                         (y2 + skew_amount, x1),  # Top Right
                         (y2, x2),                # Bottom Right
                         (y1, x2)]                # Bottom Left
        elif skew_direction == 3:
            # Backward Tilt
            new_plane = [(y1, x1),                # Top Left
                         (y2, x1),                # Top Right
                         (y2 + skew_amount, x2),  # Bottom Right
                         (y1 - skew_amount, x2)]  # Bottom Left

        # To calculate the coefficients required by PIL for the perspective skew,
        # see the following Stack Overflow discussion: https://goo.gl/sSgJdj
        matrix = []

        for p1, p2 in zip(new_plane, original_plane):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        A = np.matrix(matrix, dtype=np.float)
        B = np.array(original_plane).reshape(8)

        homography = np.dot(np.linalg.pinv(A), B)
        homography = tuple(np.array(homography).reshape(8))

        img =  img.transform(img.size, Image.PERSPECTIVE, homography, resample=Image.BICUBIC)

        homography = np.linalg.pinv(np.float32(homography+(1,)).reshape(3,3)).ravel()[:8]
        return F.update_img_and_labels(inp, img, persp=tuple(homography))



class StillTransform (object):
    """ Takes and return an image, without changing its shape or geometry.
    """
    def _transform(self, img):
        raise NotImplementedError()

    def __call__(self, inp):
        img = F.grab_img(inp)

        # transform the image (size should not change)
        img = self._transform(img)

        return F.update_img_and_labels(inp, img, aff=(1,0,0,0,1,0))



class ColorJitter (StillTransform):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
        Transform which randomly adjusts brightness, contrast and
        saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(tvf.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(tvf.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(tvf.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(tvf.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = tvf.Compose(transforms)

        return transform

    def _transform(self, img):
        transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        return transform(img)


class RandomErasing (StillTransform):
    """
    Class that performs Random Erasing, an augmentation technique described
    in `https://arxiv.org/abs/1708.04896 <https://arxiv.org/abs/1708.04896>`_
    by Zhong et al. To quote the authors, random erasing:

    "*... randomly selects a rectangle region in an image, and erases its
    pixels with random values.*"

    The size of the random rectangle is controlled using the
    :attr:`area` parameter. This area is random in its
    width and height.

    Args:
        area: The percentage area of the image to occlude.
    """
    def __init__(self, area):
        self.area = area

    def _transform(self, image):
        """
        Adds a random noise rectangle to a random area of the passed image,
        returning the original image with this rectangle superimposed.

        :param image: The image to add a random noise rectangle to.
        :type image: PIL.Image
        :return: The image with the superimposed random rectangle as type
         image PIL.Image
        """
        w, h = image.size

        w_occlusion_max = int(w * self.area)
        h_occlusion_max = int(h * self.area)

        w_occlusion_min = int(w * self.area/2)
        h_occlusion_min = int(h * self.area/2)

        if not (w_occlusion_min < w_occlusion_max and h_occlusion_min < h_occlusion_max):
            return image
        w_occlusion = np.random.randint(w_occlusion_min, w_occlusion_max)
        h_occlusion = np.random.randint(h_occlusion_min, h_occlusion_max)

        if len(image.getbands()) == 1:
            rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion) * 255))
        else:
            rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion, len(image.getbands())) * 255))

        assert w > w_occlusion and h > h_occlusion, pdb.set_trace()
        random_position_x = np.random.randint(0, w - w_occlusion)
        random_position_y = np.random.randint(0, h - h_occlusion)

        image = image.copy() # don't modify the original
        image.paste(rectangle, (random_position_x, random_position_y))

        return image


class ToTensor (StillTransform, tvf.ToTensor):
    def _transform(self, img):
        return tvf.ToTensor.__call__(self, img)

class Normalize (StillTransform, tvf.Normalize):
    def _transform(self, img):
        return tvf.Normalize.__call__(self, img)


class BBoxToPixelLabel (object):
    """ Convert a bbox into per-pixel label
    """
    def __init__(self, nclass, downsize, mode):
        self.nclass = nclass
        self.downsize = downsize
        self.mode = mode
        self.nbin = 5
        self.log_scale = 1.5
        self.ref_scale = 8.0

    def __call__(self, inp):
        assert isinstance(inp, dict)

        w, h = inp['img'].size
        ds = self.downsize
        assert w % ds == 0
        assert h % ds == 0

        x0,y0,x1,y1 = inp['bbox']
        inp['bbox'] = np.int64(inp['bbox'])

        ll = x0/ds
        rr = (x1-1)/ds
        tt = y0/ds
        bb = (y1-1)/ds
        l = max(0, int(ll))
        r = min(w//ds, 1+int(rr))
        t = max(0, int(tt))
        b = min(h//ds, 1+int(bb))
        inp['bbox_downscaled'] = np.array((l,t,r,b), dtype=np.int64)

        W, H = w//ds, h//ds
        res = np.zeros((H,W), dtype=np.int64)
        res[:] = self.nclass # last bin is null class
        res[t:b, l:r] = inp['label']
        inp['pix_label'] = res

        if self.mode == 'hough':
            # compute hough parameters
            topos = lambda left, pos, right: np.floor( self.nbin * (pos - left) / (right - left) )
            def tolog(size):
                size = max(size,1e-8) # make it positive
                return np.round( np.log(size / self.ref_scale) / np.log(self.log_scale) + (self.nbin-1)/2 )

            # for each pixel, find its x and y position
            yc,xc = np.mgrid[0:H, 0:W]
            res = -np.ones((4, H, W), dtype=np.int64)
            res[0] = topos(ll, xc, rr)
            res[1] = topos(tt, yc, bb)
            res[2] = tolog(rr - ll)
            res[3] = tolog(bb - tt)
            res = np.clip(res, 0, self.nbin-1)
            inp['pix_bbox_hough'] = res

        elif self.mode == 'regr':
            topos = lambda left, pos, right: (pos - left) / (right - left)
            def tolog(size):
                size = max(size,1) # make it positive
                return np.log(size / self.ref_scale) / np.log(self.log_scale)

            # for each pixel, find its x and y position
            yc,xc = np.float64(np.mgrid[0:H, 0:W]) + 0.5
            res = -np.ones((4, H, W), dtype=np.float32)
            res[0] = topos(ll, xc, rr)
            res[1] = topos(tt, yc, bb)
            res[2] = tolog(rr - ll)
            res[3] = tolog(bb - tt)
            inp['pix_bbox_regr'] = res

        else:
            raise NotImplementedError()

        return inp





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Script to try out and visualize transformations")
    parser.add_argument('--img', type=str, default='$HERE/test.png', help='input image')
    parser.add_argument('--trfs', type=str, required=True, help='sequence of transformations')

    parser.add_argument('--bbox', action='store_true', help='add a bounding box')
    parser.add_argument('--polygons', action='store_true', help='add a polygon')

    parser.add_argument('--input_size', type=int, default=224, help='optional param')
    parser.add_argument('--layout', type=int, nargs=2, default=(3,3), help='Number of rows and columns')

    args = parser.parse_args()

    import os
    args.img = args.img.replace('$HERE',os.path.dirname(__file__))
    img = Image.open(args.img)

    if args.bbox or args.polygons:
        img = dict(img=img)

    if args.bbox:
        w, h = img['img'].size
        img['bbox'] = (w/4,h/4,3*w/4,3*h/4)
    if args.polygons:
        w, h = img['img'].size
        img['polygons'] = [(1,[(w/4,h/4),(w/2,h/4),(w/4,h/2)])]

    trfs = create(args.trfs, input_size=args.input_size)

    from matplotlib import pyplot as pl
    pl.ion()
    pl.subplots_adjust(0,0,1,1)

    nr,nc = args.layout

    while True:
        for j in range(nr):
            for i in range(nc):
                pl.subplot(nr,nc,i+j*nc+1)
                if i==j==0:
                    img2 = img
                else:
                    img2 = trfs(img.copy())
                if isinstance(img2, dict):
                    if 'bbox' in img2:
                        l,t,r,b = img2['bbox']
                        x,y = [l,r,r,l,l], [t,t,b,b,t]
                        pl.plot(x,y,'--',lw=5)
                    if 'polygons' in img2:
                        for label, pts in img2['polygons']:
                            x,y = zip(*pts)
                            pl.plot(x,y,'-',lw=5)
                    img2 = img2['img']
                pl.imshow(img2)
                pl.xlabel("%d x %d" % img2.size)
                pl.xticks(())
                pl.yticks(())
        pdb.set_trace()
