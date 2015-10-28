#!/usr/bin/python
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pprint
import os
import re
import sys

sys.path.insert(0, os.path.join(os.getenv('HOME'), 'caffe', 'python'))
import caffe

parser = argparse.ArgumentParser()
parser.add_argument("--model_file", default="")
parser.add_argument("--deploy_file", default="")
parser.add_argument("--mean_file", default="")
parser.add_argument("--image_dir", default="")
parser.add_argument("--mode", default="gpu")
parser.add_argument("--min_confidence", default="0.7", type=float)
parser.add_argument("--output_predication", default="")
parser.add_argument("--label_file", default ="")
args = parser.parse_args()

model_file = args.model_file
deploy_file = args.deploy_file
mean_file = args.mean_file
image_dir = args.image_dir

pp = pprint.PrettyPrinter(indent=2)
print "Using %s mode" % args.mode
if args.mode == 'gpu' :
  caffe.set_mode_gpu()
else :
  caffe.set_mode_cpu()

def classify (net, image_file) :
  input_image = caffe.io.load_image(image_file)
  # predict takes any number of images, and formats them for the Caffe net
  # automatically
  prediction = net.predict([input_image])
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
ground_truth_extractor = re.compile("^([0-9]+)-")
output_predications = []
labels = []
if args.label_file:
  labels = open(args.label_file).readlines()
  labels = [label.strip() for label in labels]

for image in sorted(os.listdir(image_dir)) :
  total += 1
  (c, prob) = classify(net, os.path.join(image_dir, image))
  label = str(c)
  if c < len(labels):
      label = labels[c]
  print image, c, prob, label
  output_predications.append("{} {} {}".format(image, label, prob))
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
if args.output_predication:
  with open(args.output_predication, "w") as of:
    of.write('\n'.join(output_predications))
