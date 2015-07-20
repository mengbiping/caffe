#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import os
import pprint
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_iter_in_k", default="55")
args = parser.parse_args()

# Make sure that caffe is on the python path:
caffe_root = './'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
#MODEL_FILE = './models/bvlc_reference_caffenet/deploy.prototxt'
#PRETRAINED = './models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
# PRETRAINED = caffe_root + 'models/clothes_caffe/snapshots/caffenet_train_iter_%s000.caffemodel'%args.model_iter_in_k
# MODEL_FILE = caffe_root + 'models/clothes/deploy.prototxt'
PRETRAINED = caffe_root + 'models/clothes_googlenet/quick_snapshots/clothes_googlenet_iter_%s000.caffemodel'%args.model_iter_in_k
MODEL_FILE = caffe_root + 'models/clothes_googlenet/deploy.prototxt'
# original mean: 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
MEAN_FILE = caffe_root + 'data/clothes/clothes_mean.npy'
IMAGE_DIR = caffe_root + '../crawler/test_data/'
#IMAGE_FILES = [ IMAGE_PATH + 'v-test1.jpg', IMAGE_PATH + 'v-test2.jpg', IMAGE_PATH + 'v-test3.jpg',
#                IMAGE_PATH + 'round-test1.jpg', IMAGE_PATH + 'round-test2.jpg', IMAGE_PATH + 'round-test3.jpg']

pp = pprint.PrettyPrinter(indent=2)
caffe.set_mode_gpu()

def LoadSynsetDict(dict_file) :
  with open('dict_file') as f:
    lines = f.read().splitlines()
  return lines

def classify(net, image_file) :
  input_image = caffe.io.load_image(image_file)
  prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
  # print 'classifying image: ' + image_file
  # print 'prediction shape:', prediction[0].shape
  plt.plot(prediction[0])
  # print 'predicted class:', prediction[0].argmax()
  # pp.pprint(prediction[0])
  c = prediction[0].argmax()
  return (c, prediction[0][c])


net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(MEAN_FILE).mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

i = 0
wrong = 0
stats = {}
for image in os.listdir(IMAGE_DIR):
    i += 1
    (c, prob) = classify(net, IMAGE_DIR + image)
    if (c == 0 and image[:1] == "r") :
        wrong += 1
        print "Wrong: %s with prob: %f"%(image, prob)
    if (c == 1 and image[:1] == "v") :
        wrong += 1
        print "Wrong: %s with prob: %f"%(image, prob)
    if (prob < 0.7) :
        c = 2
    if (image[:1] not in stats) :
        stats[image[:1]] = [0, 0, 0]
    stats[image[:1]][c] += 1

print "Wrong images: %d"%wrong
print "Total images: %d"%i
print "Wrong images(high prob): %d"%(stats['r'][0] + stats['v'][1])
print "Total images(high prob): %d"%(i - stats['r'][2] - stats['v'][2])
pp.pprint(stats)
