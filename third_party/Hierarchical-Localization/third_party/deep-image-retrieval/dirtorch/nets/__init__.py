''' List all architectures at the bottom of this file.

To list all available architectures, use:
    python -m nets
'''
import os
import pdb
import torch
from collections import OrderedDict

from .backbones.resnet import resnet101, resnet50, resnet18, resnet152
from .rmac_resnet import resnet18_rmac, resnet50_rmac, resnet101_rmac, resnet152_rmac
from .rmac_resnet_fpn import resnet18_fpn_rmac, resnet50_fpn_rmac, resnet101_fpn_rmac, resnet101_fpn0_rmac, resnet152_fpn_rmac

internal_funcs = set(globals().keys())


def list_archs():
    model_names = {name for name in globals()
                   if name.islower() and not name.startswith("__")
                   and name not in internal_funcs
                   and callable(globals()[name])}
    return model_names


def create_model(arch, pretrained='', delete_fc=False, *args, **kwargs):
    ''' Create an empty network for RMAC.

    arch : str
        name of the function to call

    kargs : list
        mandatory arguments

    kwargs : dict
        optional arguments
    '''
    # creating model
    if arch not in globals():
        raise NameError("unknown model architecture '%s'\nSelect one in %s" % (
                         arch, ','.join(list_archs())))
    model = globals()[arch](*args, **kwargs)

    model.preprocess = dict(
        mean=model.rgb_means,
        std=model.rgb_stds,
        input_size=max(model.input_size)
    )

    if os.path.isfile(pretrained or ''):
        class watcher:
            class AverageMeter:
                pass

            class Watch:
                pass
        import sys
        sys.modules['utils.watcher'] = watcher
        weights = torch.load(pretrained, map_location=lambda storage, loc: storage)['state_dict']
        load_pretrained_weights(model, weights, delete_fc=delete_fc)

    elif pretrained:
        assert hasattr(model, 'load_pretrained_weights'), 'Model %s must be initialized with a valid model file (not %s)' % (arch, pretrained)
        model.load_pretrained_weights(pretrained)

    return model


def load_pretrained_weights(net, state_dict, delete_fc=False):
    """ Load the pretrained weights (chop the last FC layer if needed)
        If layers are missing or of  wrong shape, will not load them.
    """

    new_dict = OrderedDict()
    for k, v in list(state_dict.items()):
        if k.startswith('module.'):
            k = k.replace('module.', '')
        new_dict[k] = v

    # Add missing weights from the network itself
    d = net.state_dict()
    for k, v in list(d.items()):
        if k not in new_dict:
            if not k.endswith('num_batches_tracked'):
                print("Loading weights for %s: Missing layer %s" % (type(net).__name__, k))
            new_dict[k] = v
        elif v.shape != new_dict[k].shape:
            print("Loading weights for %s: Bad shape for layer %s, skipping" % (type(net).__name__, k))
            new_dict[k] = v

    net.load_state_dict(new_dict)

    # Remove the FC layer if size doesn't match
    if delete_fc:
        fc = net.fc_name
        del new_dict[fc+'.weight']
        del new_dict[fc+'.bias']






























