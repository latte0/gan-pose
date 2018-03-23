import torch.utils.data as data
import numpy as np
import ref
import torch
from h5py import File
import cv2

import torchvision.models as models

class Vnect(nn.Module):
    def __init__(self ):
        super(Vnect, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.resnet.parameters():
            param.requires_grad = False

        #256/8 = 32

        self.res5a_branch2a_new = nn.Conv2d(32 , 512, 1, stride=2, padding=1)
        self.res5a_branch2b_new = nn.Conv2d(512, 512, 3, stride=2, padding=1)

        #with out relu
        self.res5a_branch2c_new = nn.Conv2d(512, 1024, 1, stride=2, padding=1)
        self.res5a_branch1_new = nn.Conv2d(32, 1024, 4, stride=2, padding=1)


        #63 is  bone
        self.res5c_branch1a = nn.ConvTranspose2d(512, 63, 4, stride=2, padding=0)
        self.res5c_branch2a = nn.ConvTranspose2d(1024, 128, 4, stride=2, padding=0)


    def forward(self, x):

        ge0 = resnet(x)


        ge1 = F.leaky_relu( self.res5a_branch2a_new(ge0), negative_slope=0.2 )
        ge2 = F.leaky_relu( self.res5a_branch2b_new(ge1), negative_slope=0.2 )

        ge3 = self.res5a_branch2c_new(ge2)

        ge1_1 = self.res5a_branch1_new(ge0)

        ge4 = ge3 + ge1_1



        """

        # Residual block 5a
        self.res5a_branch2a_new = tc.layers.conv2d(self.res4f, kernel_size=1, num_outputs=512, scope='res5a_branch2a_new')
        self.res5a_branch2b_new = tc.layers.conv2d(self.res5a_branch2a_new, kernel_size=3, num_outputs=512, scope='res5a_branch2b_new')
        self.res5a_branch2c_new = tc.layers.conv2d(self.res5a_branch2b_new, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res5a_branch2c_new')
        self.res5a_branch1_new = tc.layers.conv2d(self.res4f, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res5a_branch1_new')
        self.res5a = tf.add(self.res5a_branch2c_new, self.res5a_branch1_new, name='res5a_add')
        self.res5a = tf.nn.relu(self.res5a, name='res5a')

        # Residual block 5b
        self.res5b_branch2a_new = tc.layers.conv2d(self.res5a, kernel_size=1, num_outputs=256, scope='res5b_branch2a_new')
        self.res5b_branch2b_new = tc.layers.conv2d(self.res5b_branch2a_new, kernel_size=3, num_outputs=128, scope='res5b_branch2b_new')
        self.res5b_branch2c_new = tc.layers.conv2d(self.res5b_branch2b_new, kernel_size=1, num_outputs=256, scope='res5b_branch2c_new')


        self.res5c_branch1a = tf.layers.conv2d_transpose(self.res5b_branch2c_new, kernel_size=4, filters=63, activation=None, strides=2, padding='same', use_bias=False, name='res5c_branch1a')
        self.res5c_branch2a = tf.layers.conv2d_transpose(self.res5b_branch2c_new, kernel_size=4, filters=128, activation=None, strides=2, padding='same', use_bias=False, name='res5c_branch2a')
        self.bn5c_branch2a = tc.layers.batch_norm(self.res5c_branch2a, scale=True, is_training=self.is_training, scope='bn5c_branch2a')
        self.bn5c_branch2a = tf.nn.relu(self.bn5c_branch2a)

        self.res5c_delta_x, self.res5c_delta_y, self.res5c_delta_z = tf.split(self.res5c_branch1a, num_or_size_splits=3, axis=3)
        self.res5c_branch1a_sqr = tf.multiply(self.res5c_branch1a, self.res5c_branch1a, name='res5c_branch1a_sqr')
        self.res5c_delta_x_sqr, self.res5c_delta_y_sqr, self.res5c_delta_z_sqr = tf.split(self.res5c_branch1a_sqr, num_or_size_splits=3, axis=3)
        self.res5c_bone_length_sqr = tf.add(tf.add(self.res5c_delta_x_sqr, self.res5c_delta_y_sqr), self.res5c_delta_z_sqr)
        self.res5c_bone_length = tf.sqrt(self.res5c_bone_length_sqr)

        self.res5c_branch2a_feat = tf.concat([self.bn5c_branch2a, self.res5c_delta_x, self.res5c_delta_y, self.res5c_delta_z, self.res5c_bone_length],
                                             axis=3, name='res5c_branch2a_feat')

        self.res5c_branch2b = tc.layers.conv2d(self.res5c_branch2a_feat, kernel_size=3, num_outputs=128, scope='res5c_branch2b')
        self.res5c_branch2c = tf.layers.conv2d(self.res5c_branch2b, kernel_size=1, filters=84, activation=None, use_bias=False, name='res5c_branch2c')
        self.heapmap, self.x_heatmap, self.y_heatmap, self.z_heatmap = tf.split(self.res5c_branch2c, num_or_size_splits=4, axis=3)
        """
