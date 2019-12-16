# coding: UTF-8
"""
    @author: samuel ko
"""

from torch.autograd import Variable
from torch.autograd import grad
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np


def D_logistic_r1(real_img, D, gamma=10.0):
    # gradient penalty
    reals = Variable(real_img, requires_grad=True).to(real_img.device)
    real_logit = D(reals)

    real_grads = grad(torch.sum(real_logit), reals)[0]
    gradient_penalty = torch.sum(torch.mul(real_grads, real_grads), dim=[1, 2, 3])
    return gradient_penalty * (gamma * 0.5)



# ==============================================================================
# R1 and R2 regularizers from the paper
# "Which Training Methods for GANs do actually Converge?", Mescheder et al. 2018
# ==============================================================================

# def D_logistic_r1(fake_img, real_img, D, gamma=10.0):
#     real_img = Variable(real_img, requires_grad=True).to(real_img.device)
#     fake_img = Variable(fake_img, requires_grad=True).to(fake_img.device)
#
#     real_score = D(real_img)
#     fake_score = D(fake_img)
#
#     loss = F.softplus(fake_score)
#     loss = loss + F.softplus(-real_score)
#
#     # GradientPenalty
#     # One of the differentiated Tensors does not require grad?
#     # https://discuss.pytorch.org/t/one-of-the-differentiated-tensors-does-not-require-grad/54694
#     real_grads = grad(torch.sum(real_score), real_img)[0]
#     gradient_penalty = torch.sum(torch.mul(real_grads, real_grads), dim=[1, 2, 3])
#     reg = gradient_penalty * (gamma * 0.5)
#
#     # fixme: only support non-lazy mode
#     return loss + reg


def D_logistic_r2(fake_img, real_img, D, gamma=10.0):
    real_img = Variable(real_img, requires_grad=True).to(real_img.device)
    fake_img = Variable(fake_img, requires_grad=True).to(fake_img.device)

    real_score = D(real_img)
    fake_score = D(fake_img)

    loss = F.softplus(fake_score)
    loss = loss + F.softplus(-real_score)

    # GradientPenalty
    # One of the differentiated Tensors does not require grad?
    # https://discuss.pytorch.org/t/one-of-the-differentiated-tensors-does-not-require-grad/54694
    fake_grads = grad(torch.sum(fake_score), fake_img)[0]
    gradient_penalty = torch.sum(torch.square(fake_grads), dim=[1, 2, 3])
    reg = gradient_penalty * (gamma * 0.5)

    # fixme: only support non-lazy mode
    return loss + reg


# ==============================================================================
# Non-saturating logistic loss with path length regularizer from the paper
# "Analyzing and Improving the Image Quality of StyleGAN", Karras et al. 2019
# ==============================================================================


def G_logistic_ns_pathreg(x, D, opts, pl_decay=0.01, pl_weight=2.0):

    fake_images_out, fake_dlatents_out = x

    fake_images_out = Variable(fake_images_out, requires_grad=True).to(fake_images_out.device)
    fake_scores_out = D(fake_images_out)
    loss = F.softplus(-fake_scores_out)

    fake_dlatents_out = Variable(fake_dlatents_out, requires_grad=True).to(fake_dlatents_out.device)
    # Compute |J*y|.
    pl_noise = torch.randn(fake_images_out.shape) / np.sqrt(fake_images_out.shape[2] * fake_images_out.shape[3])
    pl_noise = pl_noise.to(fake_images_out.device)
    pl_grads = grad(torch.sum(fake_images_out * pl_noise), fake_dlatents_out, retain_graph=True)[0]
    pl_lengths = torch.sqrt(torch.sum(torch.sum(torch.mul(pl_grads, pl_grads), dim=2), dim=1))
    pl_mean = pl_decay * torch.sum(pl_lengths)

    # Calculate (|J*y|-a)^2.
    # Computes square of x element-wise
    # https://discuss.pytorch.org/t/computes-square-of-x-element-wise/9079
    pl_penalty = torch.mul(pl_lengths - pl_mean, pl_lengths - pl_mean)

    # Apply weight.
    # Note: The division in pl_noise decreases the weight by num_pixels, and the reduce_mean
    # in pl_lengths decreases it by num_affine_layers. The effective weight then becomes:
    #
    # gamma_pl = pl_weight / num_pixels / num_affine_layers
    # = 2 / (r^2) / (log2(r) * 2 - 2)
    # = 1 / (r^2 * (log2(r) - 1))
    # = ln(2) / (r^2 * (ln(r) - ln(2))
    #
    reg = pl_penalty * pl_weight

    # fixme: only support non-lazy mode
    return loss + reg
