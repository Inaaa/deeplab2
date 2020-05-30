
import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
import cv2

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.python.platform import gfile
import glob

import tensorflow as tf

import time


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, GRAPH_PB_PATH):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())


    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    #width, height = image.size
    #resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    #target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize((500,150), Image.ANTIALIAS)
    print('resized_image = {}'.format(resized_image.size))

    time5 =time.time()
    batch_seg_map = self.sess.run(
    self.OUTPUT_TENSOR_NAME, feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    time6 =time.time()
    print('time of session {}'.format(time6-time5))

    seg_map = batch_seg_map[0]
    #resized_seg_map = np.resize(seg_map, (600, 2000))

    im = Image.fromarray(np.uint8(seg_map))
    resized_seg_map = im.resize((2000,600), Image.ANTIALIAS)
    #print(resized_seg_map.size)
    #self.road(image,resized_seg_map)
    self.polygon(image, resized_seg_map)


    return resized_seg_map

  def polygon(self,image,seg_map):
    kernel = np.ones((5, 5), np.uint8)
    local_instance_mask = np.array(seg_map) == 0

    erosion2 = cv2.erode(local_instance_mask.astype('uint8'), kernel, iterations=1)
    dilation2 = cv2.dilate(erosion2, kernel, iterations = 1)
    boundry =(dilation2-erosion2)*255
    boundry2 = cv2.dilate(boundry, kernel, iterations=1)

    #Image.fromarray(boundry2.astype('uint8')).show()
    plt.figure()
    plt.imshow(image)
    #plt.imshow(erosion2,alpha=0.5)
    plt.imshow(erosion2,alpha=0.5)
    #plt.imsave("/mrtstorage/users/chli/real_data/image_thesis/")
    plt.show()







    #opening = cv2.morphologyEx(seg_map, cv2.MORPH_OPEN, kernel)
    #self.road(image,opening)


  def road(self,image, seg_map):

    # get the mask only for person
    seg_map = np.array(seg_map)
    mask = np.zeros([seg_map.shape[0], seg_map.shape[1]])

    print(seg_map.shape)

    # print('mask_size{}'.format(mask.shape))
    mask = mask.astype('uint8')
    for i in range(seg_map.shape[0]):
      for j in range(seg_map.shape[1]):
        if seg_map[i][j] == 0:
          mask[i][j] = 255
        else:
          mask[i][j] =0


    plt.figure()
    plt.imshow(image)
    plt.imshow(mask,alpha=0.7)
    plt.show()


    np.save("./opencv_image/1.npy", mask)


time1 = time.time()
os.environ["CUDA_VISIBLE_DEVICES"]="1"

MODEL = DeepLabModel("/mrtstorage/users/chli/cityscapes/exp/train_on_train_set/train_fine/model/frozen_inference_graph.pb") # deeplab finetune
#MODEL = DeepLabModel("/home/chli/cc_code2/deeplab/deeplabv3_cityscapes_train_2018_02_06/deeplabv3_cityscapes_train/frozen_inference_graph.pb") #deeplab origional


def run_visualization(url):
  """Inferences DeepLab model and visualizes result."""
  original_im = Image.open(url)
  original_im =original_im.crop((1000, 936, 3000, 1536))
  print('original_im ={}'.format(original_im.size))

  seg_map = MODEL.run(original_im)
  #print('time of run {}'.format(time4 -time3))

  return original_im, seg_map



path ="/mrtstorage/users/chli/real_data/image/" # real image path  # need crop
#path = "/mrtstorage/users/chli/cityscapes/leftImg8bit/test/berlin/"  # data of origin cityscapes dataset
#path = "/mrtstorage/users/students/chli/cityscapes_slope/leftImg8bit/test/"
#path2 ="/mrtstorage/users/chli/real_data/crop_image2/"
#path3 ="/mrtstorage/users/chli/real_data/gt_image2/"

files = os.listdir(path)
for index, file in enumerate(files):

  image_path = path+file
  if index >2:
    image,seg_map =run_visualization(image_path)
    print(type(seg_map))
    #image.save(path2+file)
    #seg_map.save(path3 + file)


    #np.savetxt("path2+file",seg_map)





  #print(type(image))

