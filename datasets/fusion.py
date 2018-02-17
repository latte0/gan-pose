import torch.utils.data as data
import numpy as np
import ref
import torch
import cv2
#from mpii import MPII
from datasets.h36m import H36M

class Fusion(data.Dataset):
  def __init__(self, opt, split):
    self.ratio3D = opt.ratio3D
    self.split = split
    self.dataset3D = H36M(opt, split)
    self.nImages3D = len(self.dataset3D)

    print( '#Images3D {}'.format( self.nImages3D ) )
  def __getitem__(self, index):
      return self.dataset3D[index]

  def __len__(self):
    return self.nImages3D
