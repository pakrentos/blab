import scipy.io as spio
import mat73  # pip install mat73
from .itertools import true_meshgrid
import numpy.ma as ma
import numpy as np
import itertools


def load_masked(name):
    temp = np.load(name)
    return ma.array(temp, mask=np.isnan(temp))


def _loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    ks = '__header__', '__version__', '__globals__'
    for k in ks:
        data.pop(k)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
        elif isinstance(dict[key], np.ndarray):
            for ind, obj in enumerate(dict[key]):
                dict[key][ind] = _todict(obj)
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def loadmat(fname):
    "Loads any MAT file"
    try:
        return mat73.loadmat(fname)
    except TypeError:
        return _loadmat(fname)


def local_mins(sig):
    sig_len = sig.shape[-1]
    local_min_points_mask = np.diff((np.diff(sig) < 0).astype(int)) < 0
    local_min_points_inds = np.r_[2:sig_len][local_min_points_mask]
    local_min_points_values = sig[local_min_points_inds]
    return local_min_points_inds, local_min_points_values


def local_maxes(sig):
    sig_len = sig.shape[-1]
    local_min_points_mask = np.diff((np.diff(sig) < 0).astype(int)) > 0
    local_min_points_inds = np.r_[2:sig_len][local_min_points_mask]
    local_min_points_values = sig[local_min_points_inds]
    return local_min_points_inds, local_min_points_values


def create_shape(obj, hard=False):
    if hasattr(obj, '__len__'):
        if len(obj) == 0:
            return [0]
        if not hard:
            if not hasattr(obj[0], '__len__'):
                return [len(obj)]
        lens = [create_shape(i) for i in obj]
        pure_lens = [i for i in lens if i is not None]
        depths = []
        if len(pure_lens) != 0:
            depths = np.array(list(itertools.zip_longest(*pure_lens, fillvalue=0))).max(axis=-1).tolist()
        obj_len = len(obj)
        temp = [len(obj)] + depths
        return [i for i in temp if i != 0]


def marray_from_lists(obj):
    shape = np.array(create_shape(obj))
    #     print(shape)
    ndims = len(shape)
    if ndims == 1:
        return ma.array(obj)
    else:
        return _marray_from_lists(obj, shape, 0)


def _marray_from_lists(obj, shape, depth):
    ndims = len(shape)
    if depth == ndims - 1:
        if hasattr(obj, '__len__'):
            pass  # для учтения кейса
        else:
            if obj is not None:
                obj = [obj]
            else:
                temp = ma.empty(shape[depth])
                temp.mask = True
                return temp
        ntrail = shape[depth] - len(obj)
        trail = ma.empty(ntrail)
        trail.mask = True
        temp = ma.array(obj)
        return ma.concatenate((obj, trail))
    else:
        if not hasattr(obj, '__len__'):
            temp = ma.empty(shape[depth:])
            if obj is not None:
                mask = np.ones(shape[depth:]).flatten()
                temp = ma.empty(shape[depth:]).flatten()
                mask[0] = 0
                temp[0] = obj
                temp.mask = mask
                temp = temp.reshape(shape[depth:])
            else:
                temp.mask = True
            return temp
        else:
            trail_shape = shape[depth:].copy()
            trail_shape[0] -= len(obj)
            #             print(trail_shape, shape)
            trail = ma.empty(trail_shape)
            trail.mask = True
            temp = ma.stack([_marray_from_lists(i, shape, depth + 1) for i in obj])
            temp = ma.concatenate((temp, trail))
            return temp