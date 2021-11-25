import numpy as np

def true_meshgrid(*arr):
    return np.swapaxes(np.array(np.meshgrid(*arr)), 1, 2).reshape(len(arr), -1).T