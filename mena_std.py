"""
computing the mean and the standard deviation per channel of any datasets with PyTorch
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

Path_to_Dataset = " "
def mean_std(data):
    cnt = 0
    mean = torch.empty(3)
    std = torch.empty(3)
    # import pdb;
    # pdb.set_trace()
    for data, label in data:
        b, c, h, w = data.size()
        num_pixels = b * h * w
        _sum = torch.sum(data, dim=[0, 2, 3])
        square = torch.sum(data ** 2, dim=[0, 2, 3])
        mean = (cnt * mean + _sum) / (cnt + num_pixels)
        std = (cnt * std + square) / (cnt + num_pixels)
        std = torch.sqrt(std - mean ** 2)

        cnt += num_pixels
        
    return mean, std 

data_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder("Path_to_Dataset",
                                                                           transform=transforms.Compose
                                                                           ([transforms.ToTensor()])),
                                                                            batch_size=1, shuffle=False, num_workers=4)

mean, std = mean_std(data_loader)
print(mean, std)
# output --> tensor([X.XXX, X.XXX, X.XXX], [X.XXX, X.XXX, X.XXX])
