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
        self.deconv1 = nn.ConvTranspose2d(1024, 1024, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 1024, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(1024)
        self.deconv3 = nn.ConvTranspose2d(1024, 1024, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(1024)
        self.deconv4 = nn.ConvTranspose2d(1024, 1024, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(1024)
        self.deconv5 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(512)
        self.deconv6 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(256)
        self.deconv7 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.deconv7_bn = nn.BatchNorm2d(256)
        self.deconv8 = nn.ConvTranspose2d(256, 256, 4, 2, 1)



    def forward(self, input):
        x = resnet(input)

        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(x))), 0.5, training=True)
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        d8 = self.deconv8(F.relu(d7))
        o = F.tanh(d8)

        return o
