#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import pprint
import argparse
import os
import re

os.chdir(os.getenv("HOME") + '/caffe')
import sys
sys.path.insert(0, 'python')
import caffe

parser = argparse.ArgumentParser()
parser.add_argument("--model_iter_in_k", default="25")
parser.add_argument("--data_name", default="clothes_neck")
parser.add_argument("--net_name", default="googlenet")
parser.add_argument("--snapshot", default="quick_snapshots")
parser.add_argument("--mode", default="gpu")
parser.add_argument("--min_confidence", default="0.7", type=float)
args = parser.parse_args()

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
model_name = "%s_%s" % (args.data_name, args.net_name)
model_file = "models/%s/%s/%s_iter_%s000.caffemodel" %(model_name, args.snapshot, model_name, args.model_iter_in_k)
deploy_file = "models/%s/deploy.prototxt" % model_name

mean_file = 'data/%s/%s_mean.npy' % (args.data_name, args.data_name)
image_dir = 'data/%s/test/' % args.data_name
#IMAGE_FILES = [ IMAGE_PATH + 'v-test1.jpg', IMAGE_PATH + 'v-test2.jpg', IMAGE_PATH + 'v-test3.jpg',
#                IMAGE_PATH + 'round-test1.jpg', IMAGE_PATH + 'round-test2.jpg', IMAGE_PATH + 'round-test3.jpg']

pp = pprint.PrettyPrinter(indent=2)
if args.mode == 'gpu' :
  caffe.set_mode_gpu()
else :
  caffe.set_mode_cpu()

def classify (net, image_file) :
  input_image = caffe.io.load_image(image_file)
  prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
  # print 'classifying image: ' + image_file
  # print 'prediction shape:', prediction[0].shape
  # plt.plot(prediction[0])
  # print 'predicted class:', prediction[0].argmax()
  # pp.pprint(prediction[0])
  c = prediction[0].argmax()
  return (c, prediction[0][c])


net = caffe.Classifier(deploy_file, model_file,
                       mean=np.load(mean_file).mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

total = 0
wrong = 0
total_high = 0
wrong_high = 0
stats = {}
ground_truth_extractor = re.compile("([0-9]+)-")
for image in os.listdir(image_dir):
  total += 1
  (c, prob) = classify(net, image_dir + image)
  matched = ground_truth_extractor.match(image)
  ground_truth = int(matched.group(1))
  if c != ground_truth :
    wrong += 1
    print "Wrong classification %d: %s with prob: %f"%(c, image, prob)
  # process low confidence.
  if prob < args.min_confidence :
    if c != ground_truth :
      wrong_high -= 1
    c = -1 
    total_high -= 1
  if ground_truth not in stats :
    stats[ground_truth] = {}
  if c not in stats[ground_truth]:
    stats[ground_truth][c] = 0
  stats[ground_truth][c] += 1

total_high += total
wrong_high += wrong

print "Wrong images: %d"%wrong
print "Total images: %d"%total
print "Precision: %f" % (1 -wrong * 1.0 / total)
print "Wrong images(high prob): %d"% wrong_high
print "Total images(high prob): %d"% total_high
print "Precision(high prob): %f" % (1 - wrong_high * 1.0 / total_high)
print "Drop rate(high prob): %f" % (1 - total_high * 1.0 / total)
pp.pprint(stats)
