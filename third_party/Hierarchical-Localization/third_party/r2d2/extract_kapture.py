# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use


from PIL import Image

from tools import common
from tools.dataloader import norm_RGB
from nets.patchnet import *

from extract import load_network, NonMaxSuppression, extract_multiscale

# Kapture is a pivot file format, based on text and binary files, used to describe SfM (Structure From Motion) and more generally sensor-acquired data
# it can be installed with
# pip install kapture
# for more information check out https://github.com/naver/kapture
import kapture
from kapture.io.records import get_image_fullpath
from kapture.io.csv import kapture_from_dir
from kapture.io.csv import get_csv_fullpath, keypoints_to_file, descriptors_to_file
from kapture.io.features import get_keypoints_fullpath, keypoints_check_dir, image_keypoints_to_file
from kapture.io.features import get_descriptors_fullpath, descriptors_check_dir, image_descriptors_to_file

def extract_kapture_keypoints(args):
    """
    Extract r2d2 keypoints and descritors to the kapture format directly 
    """
    print('extract_kapture_keypoints...')
    kdata = kapture_from_dir(args.kapture_root, matches_pairs_file_path=None, 
    skip_list= [kapture.GlobalFeatures,
                kapture.Matches,
                kapture.Points3d,
                kapture.Observations])

    assert kdata.records_camera is not None
    image_list = [filename for _, _, filename in kapture.flatten(kdata.records_camera)]
    if kdata.keypoints is not None and kdata.descriptors is not None:
        image_list = [name for name in image_list if name not in kdata.keypoints or name not in kdata.descriptors]

    if len(image_list) == 0:
        print('All features were already extracted')
        return
    else:
        print(f'Extracting r2d2 features for {len(image_list)} images')

    iscuda = common.torch_set_gpu(args.gpu)

    # load the network...
    net = load_network(args.model)
    if iscuda: net = net.cuda()

    # create the non-maxima detector
    detector = NonMaxSuppression(
        rel_thr = args.reliability_thr, 
        rep_thr = args.repeatability_thr)

    keypoints_dtype = None if kdata.keypoints is None else kdata.keypoints.dtype
    descriptors_dtype = None if kdata.descriptors is None else kdata.descriptors.dtype

    keypoints_dsize = None if kdata.keypoints is None else kdata.keypoints.dsize
    descriptors_dsize = None if kdata.descriptors is None else kdata.descriptors.dsize

    for image_name in image_list:
        img_path = get_image_fullpath(args.kapture_root, image_name)
        
        print(f"\nExtracting features for {img_path}")
        img = Image.open(img_path).convert('RGB')
        W, H = img.size
        img = norm_RGB(img)[None] 
        if iscuda: img = img.cuda()
        
        # extract keypoints/descriptors for a single image
        xys, desc, scores = extract_multiscale(net, img, detector,
            scale_f   = args.scale_f, 
            min_scale = args.min_scale, 
            max_scale = args.max_scale,
            min_size  = args.min_size, 
            max_size  = args.max_size, 
            verbose = True)

        xys = xys.cpu().numpy()
        desc = desc.cpu().numpy()
        scores = scores.cpu().numpy()
        idxs = scores.argsort()[-args.top_k or None:]
        
        xys = xys[idxs]
        desc = desc[idxs]
        if keypoints_dtype is None or descriptors_dtype is None:
            keypoints_dtype = xys.dtype
            descriptors_dtype = desc.dtype

            keypoints_dsize = xys.shape[1]
            descriptors_dsize = desc.shape[1]

            kdata.keypoints = kapture.Keypoints('r2d2', keypoints_dtype, keypoints_dsize)
            kdata.descriptors = kapture.Descriptors('r2d2', descriptors_dtype, descriptors_dsize)

            keypoints_config_absolute_path = get_csv_fullpath(kapture.Keypoints, args.kapture_root)
            descriptors_config_absolute_path = get_csv_fullpath(kapture.Descriptors, args.kapture_root)

            keypoints_to_file(keypoints_config_absolute_path, kdata.keypoints)
            descriptors_to_file(descriptors_config_absolute_path, kdata.descriptors)
        else:
            assert kdata.keypoints.type_name == 'r2d2'
            assert kdata.descriptors.type_name == 'r2d2'
            assert kdata.keypoints.dtype == xys.dtype
            assert kdata.descriptors.dtype == desc.dtype
            assert kdata.keypoints.dsize == xys.shape[1]
            assert kdata.descriptors.dsize == desc.shape[1]

        keypoints_fullpath = get_keypoints_fullpath(args.kapture_root, image_name)
        print(f"Saving {xys.shape[0]} keypoints to {keypoints_fullpath}")
        image_keypoints_to_file(keypoints_fullpath, xys)
        kdata.keypoints.add(image_name)

        
        descriptors_fullpath = get_descriptors_fullpath(args.kapture_root, image_name)
        print(f"Saving {desc.shape[0]} descriptors to {descriptors_fullpath}")
        image_descriptors_to_file(descriptors_fullpath, desc)
        kdata.descriptors.add(image_name)
    
    if not keypoints_check_dir(kdata.keypoints, args.kapture_root) or \
            not descriptors_check_dir(kdata.descriptors, args.kapture_root):
        print('local feature extraction ended successfully but not all files were saved')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Extract r2d2 local features for all images in a dataset stored in the kapture format")
    parser.add_argument("--model", type=str, required=True, help='model path')
    
    parser.add_argument("--kapture-root", type=str, required=True, help='path to kapture root directory')
    
    parser.add_argument("--top-k", type=int, default=5000, help='number of keypoints')

    parser.add_argument("--scale-f", type=float, default=2**0.25)
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)
    
    parser.add_argument("--reliability-thr", type=float, default=0.7)
    parser.add_argument("--repeatability-thr", type=float, default=0.7)

    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help='use -1 for CPU')
    args = parser.parse_args()

    extract_kapture_keypoints(args)
