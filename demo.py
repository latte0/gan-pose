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
import os

def main():
  opt = opts().parse()
  #if opt.loadModel != 'none':
  model = AlexNet(ref.nJoints).cuda()

  model.load_state_dict(torch.load("save.model"))
  

  for (i,filename) in enumerate(os.listdir("./testimages/")):

    img = cv2.imread("./testimages/" + filename)
    c = np.ones(2) * ref.h36mImgSize / 2
    s = ref.h36mImgSize * 1.0
    img2=  Crop(img, c, s, 0, ref.inputRes) / 256. 

    input = torch.from_numpy(img2)

    input = input.contiguous().view(1, input.size(0), input.size(1), input.size(2))

    print(input.size())

    input_var = torch.autograd.Variable(input).float().cuda()
    output = model(input_var)
    print(output.size())
    reg = (output.data).cpu().numpy()#.reshape(pred.shape[0], 1)
  
    four = lambda t: t * 4.57
    fourfunc = np.vectorize(four)
    reg = fourfunc(reg)

    print(reg)

    debugger = Debugger()
    debugger.addImg((input[0].numpy().transpose(1, 2, 0)*256).astype(np.uint8))
    debugger.addPoint2D(reg, (255, 0, 0))
  
    #debugger.addPoint3D(np.concatenate([pred, (reg + 1) / 2. * 256], axis = 1))
  
    debugger.saveImg( path = "./result/" + filename )
    
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    with open("./result/" + filename[:-4] + ".out", 'w') as f:
      f.write(np.array2string(reg, separator=', '))
    


    """
    debugger.showImg(pause = True)
    debugger.show3D()
    """

if __name__ == '__main__':
  main()
