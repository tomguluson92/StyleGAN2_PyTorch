# -*- coding: utf-8 -*-


"""
    Miscellaneous utility classes and functions For StyleGAN2 Network.
"""
import torch
import numpy as np


# TWO = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5,
#        64: 6, 128: 7, 256: 8, 512: 9, 1024: 10}

TWO = [pow(2, _) for _ in range(11)]

def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


def _approximate_size(feature_size):
    """
        return most approximate 2**(x).
    :param feature_size (int): feature height (feature weight == feature height)
    :return:
    """

    tmp = map(lambda x: abs(x - int(feature_size)), TWO)
    tmp = list(tmp)

    idxs = tmp.index(min(tmp))
    return pow(2, idxs)


if __name__ == "__main__":
    # k = _setup_kernel([1, 3, 3, 1])
    # print(k)
    # print(k[::-1, ::-1])

    _approximate_size(0)