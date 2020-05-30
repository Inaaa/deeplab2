
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

import tensorflow as tf

import time

os.environ["CUDA_VISIBLE_DEVICES"]="2"

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

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
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    print('resized_image = {}'.format(resized_image.size))

    time5 =time.time()
    batch_seg_map = self.sess.run(
      self.OUTPUT_TENSOR_NAME,
      feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    time6 =time.time()
    print('time of session {}'.format(time6-time5))

    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  print('!!!!!!!!!!!!',type(seg_map))
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.5)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  print('unique_labels = {}'.format(unique_labels))
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')

  plt.show()


time1 = time.time()

#LABEL_NAMES = np.asarray(['road', 'others'])

LABEL_NAMES = np.asarray([
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle', 'license plate'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

#checkpoint_path = "/home/chli/cc_code2/deeplab/deeplab_mnv3_large_cityscapes_trainfine_2019_11_15.tar.gz"

#checkpoint_path = "/home/chli/cc_code2/deeplab/deeplabv3_cityscapes_train_2018_02_06.tar.gz"
checkpoint_path = "/mrtstorage/users/chli/cityscapes/exp/train_on_train_set/train_fine/model.tar.xz"
#checkpoint_path = "/mrtstorage/users/chli/cityscapes_slope/exp/train_on_train_set/train_test.tar.xz"
#checkpoint_path = "/mrtstorage/users/chli/cityscapes_slope/exp/train_on_train_set/train3/model.tar.xz"
#checkpoint_path = "/mrtstorage/users/chli/cityscapes/exp/train_on_train_set/train5/model.tar.xz"

MODEL = DeepLabModel(checkpoint_path)


def run_visualization(url):
  """Inferences DeepLab model and visualizes result."""
  original_im = Image.open(url)
  original_im =original_im.crop((500, 936, 3500, 1536))
  print('original_im ={}'.format(original_im.size))
  #print(original_im.type)

  #print(original_im)

  resized_im, seg_map = MODEL.run(original_im)
  #print('time of run {}'.format(time4 -time3))

  vis_segmentation(resized_im, seg_map)
  return original_im, seg_map


#path = "/home/chli/cc_code2/deeplab/images/"
#path = "/mrtstorage/users/chli/cityscapes_slope/Image/"
#path ="/home/chli/cc_code2/deeplab/kitti_image/testing/image_2/"
#path2 ="/home/chli/cc_code2/deeplab/kitti_image/testing/"
path ="/mrtstorage/users/chli/real_data/image/"
#path = "/mrtstorage/users/students/chli/cityscapes_slope/leftImg8bit/test/"

#path2 ="/mrtstorage/users/chli/real_data/gt_image/"
files = os.listdir(path)
for index, file in enumerate(files):

  image_path = path+file

  image,seg_map =run_visualization(image_path)
  print(type(seg_map))

  #im = Image.fromarray(seg_map)
  #im.save(path2 + file)

  dim = image.size
  print('dim',dim)
  image =np.array(image)

  print('image_shape',image.shape)
  #seg_map = np.resize(seg_map,(dim[1],dim[0]))
  print(seg_map.shape)


  #resized_image = cv2.resize(seg_map, (image.shape[0],image.shape[1]),interpolation = cv2.INTER_AREA)
  #print('resized_image{}'.format(resized_image.shape))
  #seg_image = label_to_color_image(resized_image).astype(np.uint8)
  #plt.imshow(resized_image)
  #plt.title('segmentation map')


  #print(path2+file)
  #


  print(type(image))


