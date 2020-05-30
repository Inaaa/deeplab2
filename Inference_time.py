#!/usr/bin/env python
# coding: utf-8


import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
import time
from matplotlib import image
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import glob 
from tensorflow.python.platform import gfile
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="2"


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 1001
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, GRAPH_PB_PATH):
    """Loads pretrained deeplab model."""
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

    Returns:

      seg_map: Segmentation map.
    """
    inference_time = []
    for i in range(11):
        time1 = time.time()
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [image]})
        time2 =time.time()
        print(time2-time1)
        inference_time.append(time2-time1)
        seg_map = batch_seg_map[0]
    return seg_map, inference_time



#MODEL = DeepLabModel("/home/chli/cc_code2/deeplab/deeplabv3_cityscapes_train_2018_02_06/deeplabv3_cityscapes_train/frozen_inference_graph.pb")

MODEL = DeepLabModel("/mrtstorage/users/chli/cityscapes/exp/train_on_train_set/train2/model/frozen_inference_graph.pb")

print('model loaded successfully!')


path ="/mrtstorage/users/chli/real_data/image/"

files = os.listdir(path)
for index, file in enumerate(files):
    #image = image.imread(path+file)
    image =Image.open(path+file)
    print(path+file)
    im=image.crop((1000,900,3000,1500))
    im=np.array(im)

    #image = np.expand_dims(img1, 3)

    seg_map, inference_time = MODEL.run(im)

    inference_time = np.mean(np.asarray(inference_time[1:]))

    print('Mean inference time : ' + str(inference_time))
    im = Image.fromarray(seg_map)
    im.show()
    im.show()



