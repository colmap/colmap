# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import pdb
import numpy as np
from PIL import Image, ImageOps, ImageEnhance


class DummyImg:
    ''' This class is a dummy image only defined by its size.
    '''
    def __init__(self, size):
        self.size = size
        
    def resize(self, size, *args, **kwargs):
        return DummyImg(size)
        
    def expand(self, border):
        w, h = self.size
        if isinstance(border, int):
            size = (w+2*border, h+2*border)
        else:
            l,t,r,b = border
            size = (w+l+r, h+t+b)
        return DummyImg(size)

    def crop(self, border):
        w, h = self.size
        l,t,r,b = border
        assert 0 <= l <= r <= w
        assert 0 <= t <= b <= h
        size = (r-l, b-t)
        return DummyImg(size)
    
    def rotate(self, angle):
        raise NotImplementedError

    def transform(self, size, *args, **kwargs):
        return DummyImg(size)


def grab_img( img_and_label ):
    ''' Called to extract the image from an img_and_label input
    (a dictionary). Also compatible with old-style PIL images.
    '''
    if isinstance(img_and_label, dict):
        # if input is a dictionary, then
        # it must contains the img or its size.
        try:
            return img_and_label['img']
        except KeyError:
            return DummyImg(img_and_label['imsize'])
            
    else:
        # or it must be the img directly
        return img_and_label


def update_img_and_labels(img_and_label, img, persp=None):
    ''' Called to update the img_and_label
    '''
    if isinstance(img_and_label, dict):
        img_and_label['img'] = img
        img_and_label['imsize'] = img.size

        if persp:
            if 'persp' not in img_and_label:
                img_and_label['persp'] = (1,0,0,0,1,0,0,0)
            img_and_label['persp'] = persp_mul(persp, img_and_label['persp'])
        
        return img_and_label
        
    else:
        # or it must be the img directly
        return img


def rand_log_uniform(a, b):
    return np.exp(np.random.uniform(np.log(a),np.log(b)))


def translate(tx, ty):
    return (1,0,tx,
            0,1,ty,
            0,0)

def rotate(angle):
    return (np.cos(angle),-np.sin(angle), 0,
            np.sin(angle), np.cos(angle), 0,
            0, 0)


def persp_mul(mat, mat2):
    ''' homography (perspective) multiplication.
    mat: 8-tuple (homography transform)
    mat2: 8-tuple (homography transform) or 2-tuple (point)
    '''
    assert isinstance(mat, tuple)
    assert isinstance(mat2, tuple)

    mat = np.float32(mat+(1,)).reshape(3,3)
    mat2 = np.array(mat2+(1,)).reshape(3,3)
    res = np.dot(mat, mat2)
    return tuple((res/res[2,2]).ravel()[:8])


def persp_apply(mat, pts):
    ''' homography (perspective) transformation.
    mat: 8-tuple (homography transform)
    pts: numpy array
    '''
    assert isinstance(mat, tuple)
    assert isinstance(pts, np.ndarray)
    assert pts.shape[-1] == 2
    mat = np.float32(mat+(1,)).reshape(3,3)

    if pts.ndim == 1:
        pt = np.dot(pts, mat[:,:2].T).ravel() + mat[:,2]
        pt /= pt[2] # homogeneous coordinates
        return tuple(pt[:2])
    else:
        pt = np.dot(pts, mat[:,:2].T) + mat[:,2]
        pt[:,:2] /= pt[:,2:3] # homogeneous coordinates
        return pt[:,:2]


def is_pil_image(img):
    return isinstance(img, Image.Image)


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
    img (PIL Image): PIL Image to be adjusted.
    brightness_factor (float):  How much to adjust the brightness. Can be
    any non negative number. 0 gives a black image, 1 gives the
    original image while 2 increases the brightness by a factor of 2.
    Returns:
    PIL Image: Brightness adjusted image.
    Copied from https://github.com/pytorch in torchvision/transforms/functional.py
    """
    if not is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.
    Args:
    img (PIL Image): PIL Image to be adjusted.
    contrast_factor (float): How much to adjust the contrast. Can be any
    non negative number. 0 gives a solid gray image, 1 gives the
    original image while 2 increases the contrast by a factor of 2.
    Returns:
    PIL Image: Contrast adjusted image.
    Copied from https://github.com/pytorch in torchvision/transforms/functional.py
    """
    if not is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.
    Args:
    img (PIL Image): PIL Image to be adjusted.
    saturation_factor (float):  How much to adjust the saturation. 0 will
    give a black and white image, 1 will give the original image while
    2 will enhance the saturation by a factor of 2.
    Returns:
    PIL Image: Saturation adjusted image.
    Copied from https://github.com/pytorch in torchvision/transforms/functional.py
    """
    if not is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See https://en.wikipedia.org/wiki/Hue for more details on Hue.
    Args:
    img (PIL Image): PIL Image to be adjusted.
    hue_factor (float):  How much to shift the hue channel. Should be in
    [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
    HSV space in positive and negative direction respectively.
    0 means no shift. Therefore, both -0.5 and 0.5 will give an image
    with complementary colors while 0 gives the original image.
    Returns:
    PIL Image: Hue adjusted image.
    Copied from https://github.com/pytorch in torchvision/transforms/functional.py
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
        h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img



