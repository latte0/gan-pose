import numpy as np
import cv2
import ref



oo = 128
def show3D(ax, points, c = (255, 0, 0)):
  points = points.reshape(ref.nJoints, 3)
  #print 'show3D', c, points
  x, y, z = np.zeros((3, ref.nJoints))
  for j in range(ref.nJoints):
    x[j] = points[j, 0]
    y[j] = - points[j, 1]
    z[j] = - points[j, 2]
  ax.scatter(z, x, y, c = c)
  for e in ref.edges:
    ax.plot(z[e], x[e], y[e], c = c)

def show2D(img, points, c):
  points = ((points.reshape(ref.nJoints, -1))).astype(np.int32)
  for j in range(ref.nJoints):
    cv2.circle(img, (points[j, 0], points[j, 1]), 3, c, -1)
  for e in ref.edges:
    cv2.line(img, (points[e[0], 0], points[e[0], 1]),
                  (points[e[1], 0], points[e[1], 1]), c, 2)
  return img

class Debugger(object):

  def __init__(self):
    self.imgs = {}

  def addImg(self, img, imgId = 0):
    self.imgs[imgId] = img.copy()

  def addPoint2D(self, point, c, imgId = 0):
    self.imgs[imgId] = show2D(self.imgs[imgId], point, c)

  def showImg(self, pause = False, imgId = 0):
    cv2.imshow('{}'.format(imgId), self.imgs[imgId])
    if pause:
      cv2.waitKey()

  def saveImg(self, path = 'debug/debug.png', imgId = 0):
    cv2.imwrite(path, self.imgs[imgId])

  def showAllImg(self, pause = False):
    for i, v in self.imgs.items():
      cv2.imshow('{}'.format(i), v)
    if pause:
      cv2.waitKey()
