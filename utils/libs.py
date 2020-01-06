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


# function to calculate the Exponential moving averages for the Generator weights
# This function updates the exponential average weights based on the current training
def update_average(model_tgt, model_src, beta):
    """
    update the model_target using exponential moving averages
    :param model_tgt: target model
    :param model_src: source model
    :param beta: value of decay beta
    :return: None (updates the target model)
    """

    # utility function for toggling the gradient requirements of the models
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    # turn off gradient calculation
    toggle_grad(model_tgt, False)
    toggle_grad(model_src, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    # turn back on the gradient calculation
    toggle_grad(model_tgt, True)
    toggle_grad(model_src, True)


class ShrinkFun(torch.autograd.Function):
    # Define grad for shrinked [-1, 1].

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input_x = input.clone()
        input_x = input_x / torch.max(torch.abs(input_x))
        return input_x

    @staticmethod
    def backward(ctx, grad_output):
        # function
        grad_input = grad_output.clone()
        return grad_input


def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'Conv2d':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


if __name__ == "__main__":
    # k = _setup_kernel([1, 3, 3, 1])
    # print(k)
    # print(k[::-1, ::-1])

    _approximate_size(0)
