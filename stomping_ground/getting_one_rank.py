# Replica of code with same name on Maintainable
# Goal: Gets images from same class out of our dataset

import torch
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


from __future__ import absolute_import
import sys
import time
import logging
import datetime
from pathlib import Path
import numpy as np


# Return only images of certain class (eg. airplanes = class 0)
def get_same_index(target, label):
    label_indices = []

    for i in range(len(target)):
        if target[i] == label:
            label_indices.append(i)

    return label_indices

