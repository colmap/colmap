import os

################################################
# file stuff

def mkdir(d):
    try: os.makedirs(d)
    except OSError: pass


def mkdir( fname, isfile='auto' ):
    ''' Make a directory given a file path
        If the path is already a directory, make sure it ends with '/' !
    '''
    if isfile == 'auto':
        isfile = bool(os.path.splitext(fname)[1])
    if isfile:
        directory = os.path.split(fname)[0]
    else:
        directory = fname
    if directory and not os.path.isdir( directory ):
        os.makedirs(directory)
_mkdir = mkdir


def touch(filename):
    ''' Touch is file. Create the file and directory if necessary.
    '''
    assert isinstance(filename, str), 'filename "%s" must be a string' % (str(filename))
    dirs = os.path.split(filename)[0]
    mkdir(dirs)
    open(filename,'r+' if os.path.isfile(filename) else 'w') # touch


def assert_outpath( path, ext='', mkdir=False ):
    """ Verify that the output file has correct format.
    """
    folder, fname = os.path.split(path)
    if ext:  assert os.path.splitext(fname)[1] == ext, 'Bad file extension, should be '+ext
    if mkdir: _mkdir(folder, isfile=False)
    assert os.path.isdir(folder), 'Destination folder not found '+folder
    assert not os.path.isfile(path), 'File already exists '+path



################################################
# Multiprocessing stuff
import multiprocessing as mp
import multiprocessing.dummy

class _BasePool (object):
  def __init__(self, nt=0):
    self.n = max(1,min(mp.cpu_count(), nt if nt>0 else nt+mp.cpu_count()))
  def starmap(self, func, args):
    self.map(lambda a: func(*a), args)

class ProcessPool (_BasePool):
  def __init__(self, nt=0):
    CorePool.__init__(self, nt)
    self.map = map if self.n==1 else mp.Pool(self.n).map

class ThreadPool (_BasePool):
  def __init__(self, nt=0):
    CorePool.__init__(self, nt)
    self.map = map if self.n==1 else mp.dummy.Pool(self.n).map


################################################
# List utils

def is_iterable(val, exclude={str}):
    if type(exclude) not in (tuple, dict, set):
      exclude = {exclude}
    try:
        if type(val) in exclude:    # iterable but no
            raise TypeError()
        plouf = iter(val)
        return True
    except TypeError:
        return False


def listify( val, exclude={str} ):
    # make it iterable
    return val if is_iterable(val,exclude=exclude) else (val,)


def unlistify( lis ):
    # if it contains just one element, returns it
    if len(lis) == 1:
        for e in lis: return e
    return lis


################################################
# file stuff

def sig_folder_ext(f):
    return (os.path.split(f)[0], os.path.splitext(f)[1])
def sig_folder(f):
    return os.path.split(f)[0]
def sig_ext(f):
    return os.path.splitext(f)[1]
def sig_3folder_ext(f):
    f = f.replace('//','/')
    f = f.replace('//','/')
    return tuple(f.split('/')[:3]) + (os.path.splitext(f)[1],)
def sig_all(f):
    return ()

def saferm(f, sig=sig_folder_ext ):
    if not os.path.isfile(f):
        return True
    if not hasattr(saferm,'signature'):
        saferm.signature = set() # init

    if sig(f) not in saferm.signature:
        # reset if the signature is different
        saferm.ask = True
        saferm.signature.add( sig(f) )

    if saferm.ask:
        print('confirm removal of %s ? (y/n/all)' %f, end=' ')
        ans = input()
        if ans not in ('y','all'): return False
        if ans == 'all': saferm.ask = False

    os.remove(f)
    return True


################################################
# measuring time

_tics = dict()
from collections import defaultdict
_tics_cum = defaultdict(float)

def tic(tag='tic'):
    from time import time as now
    _tics['__last__'] = tag
    _tics[tag] = now()

def toc(tag='', cum=False):
    from time import time as now
    t = now()
    tag = tag or _tics['__last__']
    t -= _tics[tag]
    if cum:
        nb, oldt = _tics_cum.get(tag,(0,0))
        nb += 1
        t += oldt
        _tics_cum[tag] = nb,t
        if cum=='avg': t/=nb
    print('%selpased time since %s = %gs' % ({False:'',True:'cumulated ','avg':'average '}[cum], tag,t))
    return t
































