# -*- coding: utf-8 -*-


"""
    Miscellaneous utility classes and functions For StyleGAN2 Network.
"""
import torch
import numpy as np


def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


if __name__ == "__main__":
    k = _setup_kernel([1, 3, 3, 1])
    print(k)
    print(k[::-1, ::-1])