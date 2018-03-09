import sys
import torch
from opts import opts
import ref
from utils.debugger import Debugger
from utils.eval import getPreds
import cv2
import numpy as np
from Network import AlexNet
from utils.img import Crop, DrawGaussian, Transform3D

def main():
  opt = opts().parse()
  #if opt.loadModel != 'none':
  model = AlexNet(ref.nJoints).cuda()

  model.load_state_dict(torch.load("save.model"))
  #model = torch.load("save.model").cuda()
  #else:
  #  model = torch.load('hgreg-3d.pth').cuda()
  img = cv2.imread("images/demo/s_01_act_02_subact_01_ca_01_000001.jpg")
  #input = torch.from_numpy(img.transpose(2, 0, 1)).float() / 256.
  c = np.ones(2) * ref.h36mImgSize / 2
  s = ref.h36mImgSize * 1.0
  img2=  Crop(img, c, s, 0, ref.inputRes) / 256. 

  input = torch.from_numpy(img2)

  input = input.contiguous().view(1, input.size(0), input.size(1), input.size(2))

  print(input.size())

  input_var = torch.autograd.Variable(input).float().cuda()
  output = model(input_var)
  print(output.size())
#  pred = getPreds((output[-2].data).cpu().numpy())[0] * 4
  reg = (output.data).cpu().numpy()#.reshape(pred.shape[0], 1)
  
  print(reg)

  """
  debugger = Debugger()
  debugger.addImg((input[0].numpy().transpose(1, 2, 0)*256).astype(np.uint8))
  debugger.addPoint2D(pred, (255, 0, 0))
  debugger.addPoint3D(np.concatenate([pred, (reg + 1) / 2. * 256], axis = 1))
  debugger.showImg(pause = True)
  debugger.show3D()
  """

if __name__ == '__main__':
  main()
