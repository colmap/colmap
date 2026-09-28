import h5py

import numpy as np

from PIL import Image

import os

import torch
from torch.utils.data import Dataset

import time

from tqdm import tqdm

from lib.utils import preprocess_image


class MegaDepthDataset(Dataset):
    def __init__(
            self,
            scene_list_path='megadepth_utils/train_scenes.txt',
            scene_info_path='/local/dataset/megadepth/scene_info',
            base_path='/local/dataset/megadepth',
            train=True,
            preprocessing=None,
            min_overlap_ratio=.5,
            max_overlap_ratio=1,
            max_scale_ratio=np.inf,
            pairs_per_scene=100,
            image_size=256
    ):
        self.scenes = []
        with open(scene_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.scenes.append(line.strip('\n'))

        self.scene_info_path = scene_info_path
        self.base_path = base_path

        self.train = train

        self.preprocessing = preprocessing

        self.min_overlap_ratio = min_overlap_ratio
        self.max_overlap_ratio = max_overlap_ratio
        self.max_scale_ratio = max_scale_ratio

        self.pairs_per_scene = pairs_per_scene

        self.image_size = image_size

        self.dataset = []

    def build_dataset(self):
        self.dataset = []
        if not self.train:
            np_random_state = np.random.get_state()
            np.random.seed(42)
            print('Building the validation dataset...')
        else:
            print('Building a new training dataset...')
        for scene in tqdm(self.scenes, total=len(self.scenes)):
            scene_info_path = os.path.join(
                self.scene_info_path, '%s.npz' % scene
            )
            if not os.path.exists(scene_info_path):
                continue
            scene_info = np.load(scene_info_path, allow_pickle=True)
            overlap_matrix = scene_info['overlap_matrix']
            scale_ratio_matrix = scene_info['scale_ratio_matrix']

            valid =  np.logical_and(
                np.logical_and(
                    overlap_matrix >= self.min_overlap_ratio,
                    overlap_matrix <= self.max_overlap_ratio
                ),
                scale_ratio_matrix <= self.max_scale_ratio
            )
            
            pairs = np.vstack(np.where(valid))
            try:
                selected_ids = np.random.choice(
                    pairs.shape[1], self.pairs_per_scene
                )
            except:
                continue
            
            image_paths = scene_info['image_paths']
            depth_paths = scene_info['depth_paths']
            points3D_id_to_2D = scene_info['points3D_id_to_2D']
            points3D_id_to_ndepth = scene_info['points3D_id_to_ndepth']
            intrinsics = scene_info['intrinsics']
            poses = scene_info['poses']
            
            for pair_idx in selected_ids:
                idx1 = pairs[0, pair_idx]
                idx2 = pairs[1, pair_idx]
                matches = np.array(list(
                    points3D_id_to_2D[idx1].keys() &
                    points3D_id_to_2D[idx2].keys()
                ))

                # Scale filtering
                matches_nd1 = np.array([points3D_id_to_ndepth[idx1][match] for match in matches])
                matches_nd2 = np.array([points3D_id_to_ndepth[idx2][match] for match in matches])
                scale_ratio = np.maximum(matches_nd1 / matches_nd2, matches_nd2 / matches_nd1)
                matches = matches[np.where(scale_ratio <= self.max_scale_ratio)[0]]
                
                point3D_id = np.random.choice(matches)
                point2D1 = points3D_id_to_2D[idx1][point3D_id]
                point2D2 = points3D_id_to_2D[idx2][point3D_id]
                nd1 = points3D_id_to_ndepth[idx1][point3D_id]
                nd2 = points3D_id_to_ndepth[idx2][point3D_id]
                central_match = np.array([
                    point2D1[1], point2D1[0],
                    point2D2[1], point2D2[0]
                ])
                self.dataset.append({
                    'image_path1': image_paths[idx1],
                    'depth_path1': depth_paths[idx1],
                    'intrinsics1': intrinsics[idx1],
                    'pose1': poses[idx1],
                    'image_path2': image_paths[idx2],
                    'depth_path2': depth_paths[idx2],
                    'intrinsics2': intrinsics[idx2],
                    'pose2': poses[idx2],
                    'central_match': central_match,
                    'scale_ratio': max(nd1 / nd2, nd2 / nd1)
                })
        np.random.shuffle(self.dataset)
        if not self.train:
            np.random.set_state(np_random_state)

    def __len__(self):
        return len(self.dataset)

    def recover_pair(self, pair_metadata):
        depth_path1 = os.path.join(
            self.base_path, pair_metadata['depth_path1']
        )
        with h5py.File(depth_path1, 'r') as hdf5_file:
            depth1 = np.array(hdf5_file['/depth'])
        assert(np.min(depth1) >= 0)
        image_path1 = os.path.join(
            self.base_path, pair_metadata['image_path1']
        )
        image1 = Image.open(image_path1)
        if image1.mode != 'RGB':
            image1 = image1.convert('RGB')
        image1 = np.array(image1)
        assert(image1.shape[0] == depth1.shape[0] and image1.shape[1] == depth1.shape[1])
        intrinsics1 = pair_metadata['intrinsics1']
        pose1 = pair_metadata['pose1']

        depth_path2 = os.path.join(
            self.base_path, pair_metadata['depth_path2']
        )
        with h5py.File(depth_path2, 'r') as hdf5_file:
            depth2 = np.array(hdf5_file['/depth'])
        assert(np.min(depth2) >= 0)
        image_path2 = os.path.join(
            self.base_path, pair_metadata['image_path2']
        )
        image2 = Image.open(image_path2)
        if image2.mode != 'RGB':
            image2 = image2.convert('RGB')
        image2 = np.array(image2)
        assert(image2.shape[0] == depth2.shape[0] and image2.shape[1] == depth2.shape[1])
        intrinsics2 = pair_metadata['intrinsics2']
        pose2 = pair_metadata['pose2']

        central_match = pair_metadata['central_match']
        image1, bbox1, image2, bbox2 = self.crop(image1, image2, central_match)

        depth1 = depth1[
            bbox1[0] : bbox1[0] + self.image_size,
            bbox1[1] : bbox1[1] + self.image_size
        ]
        depth2 = depth2[
            bbox2[0] : bbox2[0] + self.image_size,
            bbox2[1] : bbox2[1] + self.image_size
        ]

        return (
            image1, depth1, intrinsics1, pose1, bbox1,
            image2, depth2, intrinsics2, pose2, bbox2
        )

    def crop(self, image1, image2, central_match):
        bbox1_i = max(int(central_match[0]) - self.image_size // 2, 0)
        if bbox1_i + self.image_size >= image1.shape[0]:
            bbox1_i = image1.shape[0] - self.image_size
        bbox1_j = max(int(central_match[1]) - self.image_size // 2, 0)
        if bbox1_j + self.image_size >= image1.shape[1]:
            bbox1_j = image1.shape[1] - self.image_size

        bbox2_i = max(int(central_match[2]) - self.image_size // 2, 0)
        if bbox2_i + self.image_size >= image2.shape[0]:
            bbox2_i = image2.shape[0] - self.image_size
        bbox2_j = max(int(central_match[3]) - self.image_size // 2, 0)
        if bbox2_j + self.image_size >= image2.shape[1]:
            bbox2_j = image2.shape[1] - self.image_size

        return (
            image1[
                bbox1_i : bbox1_i + self.image_size,
                bbox1_j : bbox1_j + self.image_size
            ],
            np.array([bbox1_i, bbox1_j]),
            image2[
                bbox2_i : bbox2_i + self.image_size,
                bbox2_j : bbox2_j + self.image_size
            ],
            np.array([bbox2_i, bbox2_j])
        )

    def __getitem__(self, idx):
        (
            image1, depth1, intrinsics1, pose1, bbox1,
            image2, depth2, intrinsics2, pose2, bbox2
        ) = self.recover_pair(self.dataset[idx])

        image1 = preprocess_image(image1, preprocessing=self.preprocessing)
        image2 = preprocess_image(image2, preprocessing=self.preprocessing)

        return {
            'image1': torch.from_numpy(image1.astype(np.float32)),
            'depth1': torch.from_numpy(depth1.astype(np.float32)),
            'intrinsics1': torch.from_numpy(intrinsics1.astype(np.float32)),
            'pose1': torch.from_numpy(pose1.astype(np.float32)),
            'bbox1': torch.from_numpy(bbox1.astype(np.float32)),
            'image2': torch.from_numpy(image2.astype(np.float32)),
            'depth2': torch.from_numpy(depth2.astype(np.float32)),
            'intrinsics2': torch.from_numpy(intrinsics2.astype(np.float32)),
            'pose2': torch.from_numpy(pose2.astype(np.float32)),
            'bbox2': torch.from_numpy(bbox2.astype(np.float32))
        }
