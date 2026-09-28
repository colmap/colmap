from collections import OrderedDict


def load_pretrained_weights(net, state_dict):
    """ Load the pretrained weights.
        If layers are missing or of  wrong shape, will not load them.
    """
    new_dict = OrderedDict()
    for k,v in list(state_dict.items()):
        if k.startswith('module.'): k = k.replace('module.', '')
        new_dict[k]=v

    # Add missing weights from the network itself
    d = net.state_dict()
    for k,v in list(d.items()):
        if k not in new_dict:
            if not k.endswith('num_batches_tracked'):
                print("Loading weights for %s: Missing layer %s" % (type(net).__name__,k))
            new_dict[k] = v
        elif v.shape != new_dict[k].shape:
            print("Loading weights for %s: Bad shape for layer %s, skipping" % (type(net).__name__,k))
            new_dict[k] = v

    net.load_state_dict(new_dict)
