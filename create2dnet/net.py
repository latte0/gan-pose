import torch.utils.data as data
import numpy as np
import ref
import torch
from h5py import File
import cv2
import rochvision.models as models



class net2d(nn.Module):
    def __init__(self ):
        super(Vnect, self).__init__()
        resnet = models.resnet50(pretrained = True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-3])
        #self.maxpool = nn.MaxPool2d(kernel_size=7)
        #self.fc = nn.Linear(2048, 10)
	

    def forward(self, x):

