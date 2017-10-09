#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append('../../python/')
import caffe

class FeatureExtractor():
  def __init__(self, weights_path, image_net_proto, device_id=-1):
    if device_id >= 0:
      caffe.set_mode_gpu()
      caffe.set_device(device_id)
    else:
      caffe.set_mode_cpu()
    # Setup image processing net.
    phase = caffe.TEST
    self.image_net = caffe.Net(image_net_proto, weights_path, phase)
    image_data_shape = self.image_net.blobs['data'].data.shape
    self.transformer = caffe.io.Transformer({'data': image_data_shape})
    channel_mean = np.zeros(image_data_shape[1:])
    channel_mean_values = [104, 117, 123]
    assert channel_mean.shape[0] == len(channel_mean_values)
    for channel_index, mean_val in enumerate(channel_mean_values):
      channel_mean[channel_index, ...] = mean_val
    self.transformer.set_mean('data', channel_mean)
    self.transformer.set_channel_swap('data', (2, 1, 0)) # BGR
    self.transformer.set_transpose('data', (2, 0, 1))

  def set_image_batch_size(self, batch_size):
    self.image_net.blobs['data'].reshape(batch_size,
        *self.image_net.blobs['data'].data.shape[1:])
  
  def preprocess_image(self, image, verbose=False):
    if type(image) in (str, unicode):
      image = plt.imread(image)
    crop_edge_ratio = (256. - 224.) / 256. / 2
    ch = int(image.shape[0] * crop_edge_ratio + 0.5)
    cw = int(image.shape[1] * crop_edge_ratio + 0.5)
    cropped_image = image[ch:-ch, cw:-cw]
    if len(cropped_image.shape) == 2:
      cropped_image = np.tile(cropped_image[:, :, np.newaxis], (1, 1, 3))
    preprocessed_image = self.transformer.preprocess('data', cropped_image)
    if verbose:
      print 'Preprocessed image has shape %s, range (%f, %f)' % \
          (preprocessed_image.shape,
           preprocessed_image.min(),
           preprocessed_image.max())
    return preprocessed_image

  def image_to_feature(self, image, output_name='fc7'):
    net = self.image_net
    if net.blobs['data'].data.shape[0] > 1:
      batch = np.zeros_like(net.blobs['data'].data)
      batch[0] = image
    else:
      batch = image.reshape(net.blobs['data'].data.shape)
    net.forward(data=batch)
    feature = net.blobs[output_name].data[0].copy()
    return feature

  def compute_features(self, image_list, output_name='fc7'):
    batch = np.zeros_like(self.image_net.blobs['data'].data)
    batch_shape = batch.shape
    batch_size = batch_shape[0]
    features_shape = (len(image_list), ) + \
        self.image_net.blobs[output_name].data.shape[1:]
    features = np.zeros(features_shape)
    for batch_start_index in range(0, len(image_list), batch_size):
      batch_list = image_list[batch_start_index:(batch_start_index + batch_size)]
      for batch_index, image_path in enumerate(batch_list):
        batch[batch_index:(batch_index + 1)] = self.preprocess_image(image_path)
      current_batch_size = min(batch_size, len(image_list) - batch_start_index)
      print 'Computing features for images %d-%d of %d' % \
          (batch_start_index, batch_start_index + current_batch_size - 1,
           len(image_list))
      self.image_net.forward(data=batch)
      features[batch_start_index:(batch_start_index + current_batch_size)] = \
          self.image_net.blobs[output_name].data[:current_batch_size]
    return features

def write_features_to_file(image_list, features, output_file):
  with open(output_file, 'w') as opfd:
    for i, image_path in enumerate(image_list):
      image_feature = features[i].tolist()
      text_features = ','.join(map(str, image_feature))
      opfd.write('%s,%s\n' % (image_list[i], text_features))

def compute_single_image_feature(feature_extractor, image_path, out_file):
  assert os.path.exists(image_path)
  preprocessed_image = feature_extractor.preprocess_image(image_path)
  feature = feature_extractor.image_to_feature(preprocessed_image)
  write_features_to_file([image_path], [feature], out_file)

def compute_image_list_features(feature_extractor, images_file_path, out_file):
  assert os.path.exists(images_file_path)
  with open(images_file_path, 'r') as infd:
    image_list = infd.read().splitlines()
  features = feature_extractor.compute_features(image_list)
  write_features_to_file(image_list, features, out_file)

def main():
  BASE_DIR = ''
  IMAGE_LIST_FILE = 'images_paths_list.txt' # List of images 1 image path per line.
  IMAGE_PATH = '../images/cat.jpg'
  OUTPUT_FILE = 'output_features.csv'
  BATCH_SIZE = 10

  # NOTE: Download these files from the Caffe Model Zoo.
  IMAGE_NET_FILE = '../../models/vgg/vgg_orig_16layer.deploy.prototxt'
  MODEL_FILE = BASE_DIR + 'Nets/vgg/VGG_ILSVRC_16_layers.caffemodel'
  DEVICE_ID = 0
  feature_extractor = FeatureExtractor(MODEL_FILE, IMAGE_NET_FILE, DEVICE_ID)
  feature_extractor.set_image_batch_size(BATCH_SIZE)

  # compute features for a list of images in a file
  compute_image_list_features(feature_extractor, IMAGE_LIST_FILE, OUTPUT_FILE)
  # compute features for a single image
  # feature_extractor.set_image_batch_size(1)
  # compute_single_image_feature(feature_extractor, IMAGE_PATH, OUTPUT_FILE)

if __name__=="__main__":
  main()
