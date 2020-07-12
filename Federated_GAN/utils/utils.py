import os, sys
from pathlib import Path
sys.path.append(os.getcwd())
import time
import functools
import argparse
import numpy as np

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import grad
from timeit import default_timer as timer
import torch.nn.init as init

from cxr_dataset import PatchDataset

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_uniform_(m.weight.data)
    
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

def generate_image(args, netG, noise=None):
    if noise is None:
        rand_label = np.random.randint(0, args.num_classes, args.bs)
        noise = gen_rand_noise_with_label(rand_label)
    with torch.no_grad():
        noisev = noise
    samples = netG(noisev)
    samples = samples.view(args.bs, 1, 128, 128)

    samples = samples * 0.5 + 0.5

    return samples