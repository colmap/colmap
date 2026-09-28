import sys
import os
import os.path as osp
import pdb

import json
import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from dirtorch.utils.convenient import mkdir
from dirtorch.utils import common
from dirtorch.utils.common import tonumpy, matmul, pool
from dirtorch.utils.pytorch_loader import get_loader

import dirtorch.test_dir as test
import dirtorch.nets as nets
import dirtorch.datasets as datasets
import dirtorch.datasets.downloader as dl

import pickle as pkl
import hashlib


def extract_features(db, net, trfs, pooling='mean', gemp=3, detailed=False, whiten=None,
                     threads=8, batch_size=16, output=None, dbg=()):
    """ Extract features from trained model (network) on a given dataset.
    """
    print("\n>> Extracting features...")
    try:
        query_db = db.get_query_db()
    except NotImplementedError:
        query_db = None

    # extract DB feats
    bdescs = []
    qdescs = []

    trfs_list = [trfs] if isinstance(trfs, str) else trfs

    for trfs in trfs_list:
        kw = dict(iscuda=net.iscuda, threads=threads, batch_size=batch_size, same_size='Pad' in trfs or 'Crop' in trfs)
        bdescs.append(test.extract_image_features(db, trfs, net, desc="DB", **kw))

        # extract query feats
        if query_db is not None:
            qdescs.append(bdescs[-1] if db is query_db
                          else test.extract_image_features(query_db, trfs, net, desc="query", **kw))

    # pool from multiple transforms (scales)
    bdescs = tonumpy(F.normalize(pool(bdescs, pooling, gemp), p=2, dim=1))
    if query_db is not None:
        qdescs = tonumpy(F.normalize(pool(qdescs, pooling, gemp), p=2, dim=1))

    if whiten is not None:
        bdescs = common.whiten_features(bdescs, net.pca, **whiten)
        if query_db is not None:
            qdescs = common.whiten_features(qdescs, net.pca, **whiten)

    mkdir(output, isfile=True)
    if query_db is db or query_db is None:
        np.save(output, bdescs)
    else:
        o = osp.splitext(output)
        np.save(o[0]+'.qdescs'+o[1], qdescs)
        np.save(o[0]+'.dbdescs'+o[1], bdescs)
    print('Features extracted.')


def load_model(path, iscuda):
    checkpoint = common.load_checkpoint(path, iscuda)
    net = nets.create_model(pretrained="", **checkpoint['model_options'])
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    if 'pca' in checkpoint:
        net.pca = checkpoint.get('pca')
    return net


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a model')

    parser.add_argument('--dataset', '-d', type=str, required=True, help='Command to load dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to weights')

    parser.add_argument('--trfs', type=str, required=False, default='', nargs='+', help='test transforms (can be several)')
    parser.add_argument('--pooling', type=str, default="gem", help='pooling scheme if several trf chains')
    parser.add_argument('--gemp', type=int, default=3, help='GeM pooling power')

    parser.add_argument('--out-json', type=str, default="", help='path to output json')
    parser.add_argument('--detailed', action='store_true', help='return detailed evaluation')
    parser.add_argument('--output', type=str, default="", help='path to output features')

    parser.add_argument('--threads', type=int, default=8, help='number of thread workers')
    parser.add_argument('--gpu', type=int, nargs='+', help='GPU ids')
    parser.add_argument('--dbg', default=(), nargs='*', help='debugging options')
    # post-processing
    parser.add_argument('--whiten', type=str, default=None, help='applies whitening')

    parser.add_argument('--whitenp', type=float, default=0.5, help='whitening power, default is 0.5 (i.e., the sqrt)')
    parser.add_argument('--whitenv', type=int, default=None, help='number of components, default is None (i.e. all components)')
    parser.add_argument('--whitenm', type=float, default=1.0, help='whitening multiplier, default is 1.0 (i.e. no multiplication)')

    args = parser.parse_args()
    args.iscuda = common.torch_set_gpu(args.gpu)

    dataset = datasets.create(args.dataset)
    print("Dataset:", dataset)

    net = load_model(args.checkpoint, args.iscuda)

    if args.whiten:
        net.pca = net.pca[args.whiten]
        args.whiten = {'whitenp': args.whitenp, 'whitenv': args.whitenv, 'whitenm': args.whitenm}
    else:
        net.pca = None
        args.whiten = None

    # Evaluate
    res = extract_features(dataset, net, args.trfs, pooling=args.pooling, gemp=args.gemp, detailed=args.detailed,
                           threads=args.threads, dbg=args.dbg, whiten=args.whiten, output=args.output)


