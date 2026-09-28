import os
import sys
import pdb
import shutil
from collections import OrderedDict
import numpy as np
import sklearn.decomposition

import torch
import torch.nn.functional as F

try:
    import torch
    import torch.nn as nn
except ImportError:
    pass


def typename(x):
    return type(x).__module__


def tonumpy(x):
    if typename(x) == torch.__name__:
        return x.cpu().numpy()
    else:
        return x


def matmul(A, B):
    if typename(A) == np.__name__:
        B = tonumpy(B)
        scores = np.dot(A, B.T)
    elif typename(B) == torch.__name__:
        scores = torch.matmul(A, B.t()).cpu().numpy()
    else:
        raise TypeError("matrices must be either numpy or torch type")
    return scores


def pool(x, pooling='mean', gemp=3):
    if len(x) == 1:
        return x[0]
    x = torch.stack(x, dim=0)
    if pooling == 'mean':
        return torch.mean(x, dim=0)
    elif pooling == 'gem':
        def sympow(x, p, eps=1e-6):
            s = torch.sign(x)
            return (x*s).clamp(min=eps).pow(p) * s
        x = sympow(x, gemp)
        x = torch.mean(x, dim=0)
        return sympow(x, 1/gemp)
    else:
        raise ValueError("Bad pooling mode: "+str(pooling))


def torch_set_gpu(gpus, seed=None, randomize=True):
    if type(gpus) is int:
            gpus = [gpus]

    assert gpus, 'error: empty gpu list, use --gpu N N ...'

    cuda = all(gpu >= 0 for gpu in gpus)

    if cuda:
        if any(gpu >= 1000 for gpu in gpus):
            visible_gpus = [int(gpu) for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(visible_gpus[gpu-1000]) for gpu in gpus])
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])
        assert cuda and torch.cuda.is_available(), "%s has GPUs %s unavailable" % (
            os.environ['HOSTNAME'], os.environ['CUDA_VISIBLE_DEVICES'])
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.fastest = True
        print('Launching on GPUs ' + os.environ['CUDA_VISIBLE_DEVICES'])
    else:
        print('Launching on >> CPU <<')

    torch_set_seed(seed, cuda, randomize=randomize)
    return cuda


def torch_set_seed(seed, cuda, randomize=True):
    if seed:
        # this makes it 3x SLOWER but deterministic
        torch.backends.cudnn.enabled = False

    if randomize and not seed:
        import time
        try:
            seed = int(np.uint32(hash(time.time())))
        except OverflowError:
            seed = int.from_bytes(os.urandom(4), byteorder='little', signed=False)
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed(seed)


def save_checkpoint(state, is_best, filename):
    try:
        dirs = os.path.split(filename)[0]
        if not os.path.isdir(dirs):
            os.makedirs(dirs)
        torch.save(state, filename)
        if is_best:
            filenamebest = filename+'.best'
            shutil.copyfile(filename, filenamebest)
            filename = filenamebest
        print("saving to "+filename)
    except:
        print("Error: Could not save checkpoint at %s, skipping" % filename)


def load_checkpoint(filename, iscuda=False):
    if not filename:
        return None
    assert os.path.isfile(filename), "=> no checkpoint found at '%s'" % filename
    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    print("=> loading checkpoint '%s'" % filename, end='')
    for key in ['epoch', 'iter', 'current_iter']:
        if key in checkpoint:
            print(" (%s %d)" % (key, checkpoint[key]), end='')
    print()

    new_dict = OrderedDict()
    for k, v in list(checkpoint['state_dict'].items()):
        if k.startswith('module.'):
            k = k[7:]
        new_dict[k] = v
    checkpoint['state_dict'] = new_dict

    if iscuda and 'optimizer' in checkpoint:
        try:
            for state in checkpoint['optimizer']['state'].values():
                for k, v in state.items():
                    if iscuda and torch.is_tensor(v):
                        state[k] = v.cuda()
        except RuntimeError as e:
            print("RuntimeError:", e, "(machine %s, GPU %s)" % (
                os.environ['HOSTNAME'], os.environ['CUDA_VISIBLE_DEVICES']),
                file=sys.stderr)
            sys.exit(1)

    return checkpoint


def switch_model_to_cuda(model, iscuda=True, checkpoint=None):
    if iscuda:
        if checkpoint:
            checkpoint['state_dict'] = {'module.' + k: v for k, v in checkpoint['state_dict'].items()}
        try:
            model = torch.nn.DataParallel(model)

            # copy attributes automatically
            for var in dir(model.module):
                if var.startswith('_'):
                    continue
                val = getattr(model.module, var)
                if isinstance(val, (bool, int, float, str, dict)) or \
                   (callable(val) and var.startswith('get_')):
                    setattr(model, var, val)

            model.cuda()
            model.isasync = True
        except RuntimeError as e:
            print("RuntimeError:", e, "(machine %s, GPU %s)" % (
                os.environ['HOSTNAME'], os.environ['CUDA_VISIBLE_DEVICES']),
                file=sys.stderr)
            sys.exit(1)

    model.iscuda = iscuda
    return model


def model_size(model):
    ''' Computes the number of parameters of the model
    '''
    size = 0
    for weights in model.state_dict().values():
        size += np.prod(weights.shape)
    return size


def freeze_batch_norm(model, freeze=True, only_running=False):
    model.freeze_bn = bool(freeze)
    if not freeze:
        return

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # Eval mode freezes the running mean and std
            m.eval()
            for param in m.named_parameters():
                if only_running:
                    # Weight and bias can be updated
                    param[1].requires_grad = True
                else:
                    # Freeze the weight and bias
                    param[1].requires_grad = False


def variables(inputs, iscuda, not_on_gpu=[]):
    ''' convert several Tensors to cuda.Variables
    Tensor whose index are in not_on_gpu stays on cpu.
    '''
    inputs_var = []

    for i, x in enumerate(inputs):
        if i not in not_on_gpu and not isinstance(x, (tuple, list)):
            if iscuda:
                x = x.cuda(non_blocking=True)
            x = torch.autograd.Variable(x)
        inputs_var.append(x)

    return inputs_var


def transform(pca, X, whitenp=0.5, whitenv=None, whitenm=1.0, use_sklearn=True):
    if use_sklearn:
        # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/base.py#L99
        if pca.mean_ is not None:
            X = X - pca.mean_
        X_transformed = np.dot(X, pca.components_[:whitenv].T)
        if pca.whiten:
            X_transformed /= whitenm * np.power(pca.explained_variance_[:whitenv], whitenp)
    else:
        X = X - pca['means']
        X_transformed = np.dot(X, pca['W'])
    return X_transformed


def whiten_features(X, pca, l2norm=True, whitenp=0.5, whitenv=None, whitenm=1.0, use_sklearn=True):
    res = transform(pca, X, whitenp=whitenp, whitenv=whitenv, whitenm=whitenm, use_sklearn=use_sklearn)
    if l2norm:
        res = res / np.expand_dims(np.linalg.norm(res, axis=1), axis=1)
    return res

