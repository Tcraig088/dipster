import torch
import torch.nn as nn
import torchvision
import sys

import numpy as np
from PIL import Image
import PIL
import numpy as np

from math import pi


def np_to_torch(img_np, dev):
    '''Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    #if is numpy array
    if isinstance(img_np, np.ndarray):
        return torch.from_numpy(img_np).to(dev)
    else:
        return img_np

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    #if is torch tensor
    if isinstance(img_var, torch.Tensor):
        img_var = img_var.detach().cpu().numpy()
    return img_var
