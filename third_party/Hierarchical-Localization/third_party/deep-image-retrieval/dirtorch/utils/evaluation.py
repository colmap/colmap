'''Evaluation metrics
'''
import pdb
import numpy as np
import torch


def accuracy_topk(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k

    output: torch.FloatTensoror np.array(float)
            shape = B * L [* H * W]
            L: number of possible labels

    target: torch.IntTensor or np.array(int)
            shape = B     [* H * W]
            ground-truth labels
    """
    if isinstance(output, np.ndarray):
        pred = (-output).argsort(axis=1)
        target = np.expand_dims(target, axis=1)
        correct = (pred == target)

        res = []
        for k in topk:
            correct_k = correct[:, :k].sum()
            res.append(correct_k / target.size)

    if isinstance(output, torch.Tensor):
        _, pred = output.topk(max(topk), 1, True, True)
        correct = pred.eq(target.unsqueeze(1))

        res = []
        for k in topk:
            correct_k = correct[:, :k].float().view(-1).sum(0)
            res.append(correct_k.mul_(1.0 / target.numel()))

    return res


def compute_AP(label, score):
    from sklearn.metrics import average_precision_score
    return average_precision_score(label, score)


def compute_average_precision(positive_ranks):
    """
    Extracted from: https://github.com/tensorflow/models/blob/master/research/delf/delf/python/detect_to_retrieve/dataset.py

    Computes average precision according to dataset convention.
    It assumes that `positive_ranks` contains the ranks for all expected positive
    index images to be retrieved. If `positive_ranks` is empty, returns
    `average_precision` = 0.
    Note that average precision computation here does NOT use the finite sum
    method (see
    https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision)
    which is common in information retrieval literature. Instead, the method
    implemented here integrates over the precision-recall curve by averaging two
    adjacent precision points, then multiplying by the recall step. This is the
    convention for the Revisited Oxford/Paris datasets.
    Args:
        positive_ranks: Sorted 1D NumPy integer array, zero-indexed.
    Returns:
        average_precision: Float.
    """
    average_precision = 0.0

    num_expected_positives = len(positive_ranks)
    if not num_expected_positives:
        return average_precision

    recall_step = 1.0 / num_expected_positives
    for i, rank in enumerate(positive_ranks):
        if not rank:
            left_precision = 1.0
        else:
            left_precision = i / rank

        right_precision = (i + 1) / (rank + 1)
        average_precision += (left_precision + right_precision) * recall_step / 2

    return average_precision


def compute_average_precision_quantized(labels, idx, step=0.01):
    recall_checkpoints = np.arange(0, 1, step)

    def mymax(x, default):
        return np.max(x) if len(x) else default

    Nrel = np.sum(labels)
    if Nrel == 0:
        return 0
    recall = np.cumsum(labels[idx])/float(Nrel)
    irange = np.arange(1, len(idx)+1)
    prec = np.cumsum(labels[idx]).astype(np.float32) / irange
    precs = np.array([mymax(prec[np.where(recall > v)], 0) for v in recall_checkpoints])
    return np.mean(precs)


def pixelwise_iou(output, target):
    """ For each image, for each label, compute the IoU between
    """
    assert False
