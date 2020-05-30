#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
import time
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import glob 
#import cv2
from tensorflow.python.platform import gfile
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="3,4"


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, GRAPH_PB_PATH):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()
    graph_def = None
    
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

    """
    for i in range(5):
      t1 = time.time()

      batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME,feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(image)]})
      t2 = time.time()
      print(t2-t1)


    seg_map = batch_seg_map[0]

    return seg_map



MODEL = DeepLabModel("/mrtstorage/users/chli/cityscapes/exp/train_on_train_set/train2/model/frozen_inference_graph.pb")
print('model loaded successfully!')


# Loading Images for inference

#image_path = glob.glob('/mrtstorage/users/rehman/datasets/cityscapes/leftImg8bit/val/munster/*.png')
image_path =glob.glob("/mrtstorage/users/chli/real_data/image/*.png")


image1 = Image.open(image_path[0])  #11,87
#image = np.expand_dims(image1, 0)
#image1
image1 =image1.crop((1000, 900, 3000, 1500))
seg_map = MODEL.run(image1)






