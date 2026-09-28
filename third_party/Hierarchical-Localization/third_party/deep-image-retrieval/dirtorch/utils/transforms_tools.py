import pdb
import numpy as np
from PIL import Image, ImageOps, ImageEnhance


def is_pil_image(img):
    return isinstance(img, Image.Image)

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
        assert 0 <= l <= r <= h
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


def update_img_and_labels(img_and_label, img, aff=None, persp=None):
    ''' Called to update the img_and_label
    '''
    if isinstance(img_and_label, dict):
        img_and_label['img'] = img
        
        if 'bbox' in img_and_label:
            l,t,r,b = img_and_label['bbox']
            corners = [(l,t),(l,b),(r,b),(r,t)]
            if aff:
                pts = [aff_mul(aff, pt) for pt in corners]
            elif persp:
                pts = [persp_mul(persp, pt) for pt in corners]
            else:
                raise NotImplementedError()
            x,y = map(list,zip(*pts))
            x.sort()
            y.sort()
            l, r = np.mean(x[:2]), np.mean(x[2:])
            t, b = np.mean(y[:2]), np.mean(y[2:])
            img_and_label['bbox'] = int_tuple(l,t,r,b)
        
        if 'polygons' in img_and_label:
            polygons = []
            for label,pts in img_and_label['polygons']:
                if aff:
                    pts = [int_tuple(*aff_mul(aff, pt)) for pt in pts]
                elif persp:
                    pts = [int_tuple(*persp_mul(persp, pt)) for pt in pts]
                else:
                    raise NotImplementedError()
                polygons.append((label, pts))
            img_and_label['polygons'] = polygons
        
        return img_and_label
        
    else:
        # or it must be the img directly
        return img


def rand_log_uniform(a, b):
    return np.exp(np.random.uniform(np.log(a),np.log(b)))


def int_tuple(*args):
    return tuple(map(int,args))

def aff_translate(tx, ty):
    return (1,0,tx,
            0,1,ty)

def aff_rotate(angle):
    return (np.cos(angle),-np.sin(angle), 0,
            np.sin(angle), np.cos(angle), 0)

def aff_mul(aff, aff2):
    ''' affine multiplication.
    aff: 6-tuple (affine transform)
    aff2: 6-tuple (affine transform) or 2-tuple (point)
    '''
    assert isinstance(aff, tuple)
    assert isinstance(aff2, tuple)
    aff = np.array(aff+(0,0,1)).reshape(3,3)

    if len(aff2) == 6:
        aff2 = np.array(aff2+(0,0,1)).reshape(3,3)
        return tuple(np.dot(aff2, aff)[:2].ravel())

    elif len(aff2) == 2:
        return tuple(np.dot(aff2+(1,), aff.T).ravel()[:2])

    else:
        raise ValueError("bad input %s" % str(aff2))

def persp_mul(mat, mat2):
    ''' homography (perspective) multiplication.
    mat: 8-tuple (homography transform)
    mat2: 8-tuple (homography transform) or 2-tuple (point)
    '''
    assert isinstance(mat, tuple)
    assert isinstance(mat2, tuple)
    mat = np.array(mat+(1,)).reshape(3,3)

    if len(mat2) == 8:
        mat2 = np.array(mat2+(1,)).reshape(3,3)
        return tuple(np.dot(mat2, mat).ravel()[:8])

    elif len(mat2) == 2:
        pt = np.dot(mat2+(1,), mat.T).ravel()
        pt /= pt[2] # homogeneous coordinates
        return tuple(pt[:2])

    else:
        raise ValueError("bad input %s" % str(aff2))



def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
    img (PIL Image): PIL Image to be adjusted.
    brightness_factor (float):  How much to adjust the brightness. Can be
    any non negative number. 0 gives a black image, 1 gives the
    original image while 2 increases the brightness by a factor of 2.
    Returns:
    PIL Image: Brightness adjusted image.
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



