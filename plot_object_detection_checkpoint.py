#!/usr/bin/env python
# coding: utf-8
'''
OBJECT DETECTION From TF2 Checkpoint
====================================
Source: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_checkpoint.html
25 July 2021.

6 Aug 2021: With the following environment this code runs (also  displays images, running under just upgraded Spyder 5.1.1).'
11 Aug 2021

RUNTIME ~50 Sec.

Input: Model from: http://download.tensorflow.org/models/object_detection/tf2/
        Images to analyze (same source)
Output: 2 images, with objects (like people) detected, boxed w/% confidence.

ENVIRONMENT:
Conda Envronment:   MLFlowProtoBuf
Gpu  Support:       True
Cuda Support:       True
Tensor Flow:        2.5.0
Python version:      3.8.8.
The numpy version:   1.19.5.
The panda version:   1.2.4.
Tensorboard version  2.6.0.
Summary of the h5py configuration
---------------------------------

h5py    3.1.0
HDF5    1.12.0
Python  3.8.8 | packaged by conda-forge | (default, Feb 20 2021, 15:50:08) [MSC v.1916 64 bit (AMD64)]
sys.platform    win32
sys.maxsize     9223372036854775807
numpy   1.19.5
cython (built with) 0.29.21
numpy (built against) 1.17.5
HDF5 (built against) 1.12.0

Last 3 output:
    Loading model... Done! Took 0.6106808185577393 seconds
Running inference for C:\\Users\steve\.keras\datasets\image1.jpg... Done
Running inference for C:\\Users\steve\.keras\datasets\image2.jpg... Done
Note: when run for first time after starting Python, the images don't show. (!)
        Run a 'simpler' script that shows images, then run this one.
'''

# %%
# This demo will take you through the steps of running an "out-of-the-box" TensorFlow 2 compatible
# detection model on a collection of images. More specifically, in this example we will be using
# the `Checkpoint Format <https://www.tensorflow.org/guide/checkpoint>`__ to load the model.

# %%
# Download the test images
# ~~~~~~~~~~~~~~~~~~~~~~~~
# First we will download the images that we will use throughout this tutorial. The code snippet
# shown bellow will download the test images from the `TensorFlow Model Garden <https://github.com/tensorflow/models/tree/master/research/object_detection/test_images>`_
# and save them inside the ``data/images`` folder.
import sys 
import os
sys.path.append(os.path.abspath("/users/steve/documents/GitHub/Misc"))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)

import tensorPrepStarter as tps
import h5py; print (h5py.version.info)
h5py.__version__
import pathlib
import tensorflow as tf

# following 3 lines from: https://github.com/Leonardo-Blanger/detr_tensorflow/issues/3
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

def download_images():
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/'
    filenames = ['image1.jpg', 'image2.jpg']
    image_paths = []
    for filename in filenames:
        image_path = tf.keras.utils.get_file(fname=filename,
                                            origin=base_url + filename,
                                            untar=False)
        image_path = pathlib.Path(image_path)
        image_paths.append(str(image_path))
    return image_paths

IMAGE_PATHS = download_images()
# 'C:\\Users\\steve\\.keras\\datasets\\image1.jpg'

# %% Download the model
# ~~~~~~~~~~~~~~~~~~
# The code snippet shown below is used to download the pre-trained object detection model we shall
# use to perform inference. The particular detection algorithm we will use is the
# `CenterNet HourGlass104 1024x1024`. More models can be found in the `TensorFlow 2 Detection Model Zoo <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md>`_.
# To use a different model you will need the URL name of the specific model. This can be done as
# follows:
#
# 1. Right click on the `Model name` of the model you would like to use;
# 2. Click on `Copy link address` to copy the download link of the model;
# 3. Paste the link in a text editor of your choice. You should observe a link similar to ``download.tensorflow.org/models/object_detection/tf2/YYYYYYYY/XXXXXXXXX.tar.gz``;
# 4. Copy the ``XXXXXXXXX`` part of the link and use it to replace the value of the ``MODEL_NAME`` variable in the code shown below;
# 5. Copy the ``YYYYYYYY`` part of the link and use it to replace the value of the ``MODEL_DATE`` variable in the code shown below.
#
# For example, the download link for the model used below is: ``download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_1024x1024_coco17_tpu-32.tar.gz``

# Download and extract model
def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)

# SAME MODEL AS THE 'SAVED MODEL' CODE...
MODEL_DATE = '20200711'
MODEL_NAME = 'centernet_hg104_1024x1024_coco17_tpu-32'
PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)

# %%  Download the labels
# ~~~~~~~~~~~~~~~~~~~
# The coode snippet shown below is used to download the labels file (.pbtxt) which contains a list
# of strings used to add the correct label to each detection (e.g. person). Since the pre-trained
# model we will use has been trained on the COCO dataset, we will need to download the labels file
# corresponding to this dataset, named ``mscoco_label_map.pbtxt``. A full list of the labels files
# included in the TensorFlow Models Garden can be found `here <https://github.com/tensorflow/models/tree/master/research/object_detection/data>`__.

# Download labels file
def download_labels(filename):
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    label_dir = tf.keras.utils.get_file(fname=filename,
                                        origin=base_url + filename,
                                        untar=False)
    label_dir = pathlib.Path(label_dir)
    return str(label_dir)

LABEL_FILENAME = 'mscoco_label_map.pbtxt'
PATH_TO_LABELS = download_labels(LABEL_FILENAME)
# PATH to labels: 'C:\\Users\\steve\\.keras\\datasets\\mscoco_label_map.pbtxt'
# %% # Load the model
# ~~~~~~~~~~~~~~
# Next we load the downloaded model
import time
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"
PATH_TO_CKPT = PATH_TO_MODEL_DIR + "/checkpoint"

print('Loading model... ', end='')
start_time = time.time()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# %% # Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Label maps correspond index numbers to category names, so that when our convolution network
# predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility
# functions, but anything that returns a dictionary mapping integers to appropriate string labels
# would be fine.

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

# %% Putting everything together
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The code shown below loads an image, runs it through the detection model and visualizes the
# detection results, including the keypoints.
#
# Note that this will take a long time (several minutes) the first time you run this code due to
# tf.function's trace-compilation --- on subsequent runs (e.g. on new images), things will be
# faster.
#
# Here are some simple things to try out if you are curious:
#
# * Modify some of the input images and see if detection still works. Some simple things to try out here (just uncomment the relevant portions of code) include flipping the image horizontally, or converting to grayscale (note that we still expect the input image to have 3 channels).
# * Print out `detections['detection_boxes']` and try to match the box locations to the boxes in the image.  Notice that coordinates are given in normalized form (i.e., in the interval [0, 1]).
# * Set ``min_score_thresh`` to other values (between 0 and 1) to allow more detections in or to filter out more detections.
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))
condaenv = os.environ['CONDA_DEFAULT_ENV']
modelstart = time.strftime('%a %b %Y')

for image_path in IMAGE_PATHS:

    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

    plt.figure()
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'DL\\plot_object_detection_checkpoint.py  {condaenv}  TF 2.4.1')
#    plt.xlabel(f' on Wed 21 July 2021 Model Downloaded in: {elapsed_time:.2f} seconds  h5py: 3.2.1' \
    plt.xlabel(f' on {modelstart} Model Downloaded in: {elapsed_time:.2f} seconds  h5py: 3.2.1' \
               f'\nPretrained Model: {MODEL_NAME}')
#    plt.axis('off')
    plt.imshow(image_np_with_detections)
    plt.show()
    print('Done')
    plt.show()

# sphinx_gallery_thumbnail_number = 2
'''
https://strftime.org/
time.strftime('%x')
time.strftime('%a %b %Y')-7
%x = Preferred date representation              ... 08/06/21
%I = Hour as a decimal number (12-hour clock). 
%M = Minutes in decimal ranging from 00 to 59. 
%p = Either “AM” or “PM” according to the given time value, etc. 
%a = Abbreviated weekday name 
%^a = Abbreviated weekday name in capital letters
%A = Full weekday name 
%b = Abbreviated month name 

%^b = Abbreviated month name in capital letters
%B = Full month name March 
%c = Date and time representation 
%d = Day of the month (01-31) 
%H = Hour in 24h format (00-23) 
%I = Hour in 12h format (01-12) 
%j = Day of the year (001-366) 
%m = Month as a decimal number (01-12) 
%M = Minute (00-59)

'''
