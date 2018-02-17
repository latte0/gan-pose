import torch
import numpy as np
from utils.utils import AverageMeter
from utils.eval import Accuracy, getPreds, MPJPE
from utils.debugger import Debugger
#from models.layers.FusionCriterion import FusionCriterion
import cv2
import ref
from progress.bar import Bar

def step(split, epoch, opt, dataLoader, model, criterion, optimizer = None):

  if split == 'train':
    model.train()
  else:
    model.eval()

  Loss, Acc, Mpjpe, Loss3D = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

  nIters = len(dataLoader)
  bar = Bar('==>', max=nIters)

  for i, (input, target3D, meta) in enumerate(dataLoader):
    input_var = torch.autograd.Variable(input).float().cuda()
    target3D_var = torch.autograd.Variable(target3D).float().cuda()

    output = model(intput_var)
#    reg = output[opt.nStack]

    optimizer.zero_grad()
    loss = mean_squared_error(output, target, self.use_visibility)
    loss.backward()
    optimizer.step()

    Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | Loss3D {loss3d.avg:.6f} | Acc {Acc.avg:.6f} | Mpjpe {Mpjpe.avg:.6f} ({Mpjpe.val:.6f})'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, Acc=Acc, split = split, Mpjpe=Mpjpe, loss3d = Loss3D)
    bar.next()
    bar.finish()

  return Loss.avg, Acc.avg, Mpjpe.avg, Loss3D.avg


def train(epoch, opt, train_loader, model, criterion, optimizer):
  return step('train', epoch, opt, train_loader, model, criterion, optimizer)

def val(epoch, opt, val_loader, model, criterion):
  return step('val', epoch, opt, val_loader, model, criterion)
