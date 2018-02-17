import torch.utils.data as data
import numpy as np
import ref
import torch
from h5py import File
import cv2

class Vnect(nn.Module):
    def __init__(self ):
        super(Vnect, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1)
        self.bnc1 = nn.BatchNorm2d(128)
        self.bnc2 = nn.BatchNorm2d(256)
        self.bnc3 = nn.BatchNorm2d(512)
        self.bnc4 = nn.BatchNorm2d(512)
        self.fc6 = nn.Linear(256*6*6, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, Nj*3)
        self.Nj = Nj


    def forward(self, x):
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
