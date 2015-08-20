#!/usr/bin/python
# Take a folder as input. The folder contains several folders in it, each
# containing certain number of images. We treat the folder names as the label
# for the containing images.
# The output will be:
# 1. two folders of images, train and val
# 2. two text files, train.txt and val.txt containing the list of image file
# names with their labels.
# 3. number_of_categories * args.test_count_in_each_type images in test folder.

import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import pprint
import argparse
import random
import imghdr
import math

parser = argparse.ArgumentParser()
parser.add_argument("data_dir")
parser.add_argument("--output_dir", default=".")
parser.add_argument("--traning_persentage", default="0.9", type=float)
parser.add_argument("--validating_persentage", default="0.1", type=float)
# Reserved for test.
parser.add_argument("--test_count_in_each_type", default="35", type=int)
args = parser.parse_args()

pp = pprint.PrettyPrinter(indent=2)

print args.traning_persentage
training_persentage = args.traning_persentage / (args.traning_persentage + args.validating_persentage)

categories = {}
for cat in os.listdir(args.data_dir) :
  categories[cat] = {}
  categories[cat]['all'] = os.listdir(args.data_dir + '/' + cat)
  categories[cat]['test'] = random.sample(categories[cat]['all'],
                                          args.test_count_in_each_type)
  rest = [item for item in categories[cat]['all'] if item not in
          categories[cat]['test']]
  # print rest
  training_count = int(math.floor(len(rest) * training_persentage))
  # print training_count
  categories[cat]['train'] = random.sample(rest, training_count)
  categories[cat]['val'] = [item for item in rest if item not in categories[cat]['train']]
  assert len(categories[cat]['all']) == len(categories[cat]['test']) + len(categories[cat]['train']) + len(categories[cat]['val']) 
  assert len(categories[cat]['test']) == args.test_count_in_each_type
  assert len(categories[cat]['train']) > 0
  assert len(categories[cat]['val']) > 0

try :
  os.mkdir(args.output_dir + '/train')
  os.mkdir(args.output_dir + '/val')
  os.mkdir(args.output_dir + '/test')
except OSError:
  print "Output dir already exists"

print args.output_dir + '/train.txt'
train_list_file = open(args.output_dir + '/train.txt', 'w')
val_list_file = open(args.output_dir + '/val.txt', 'w')
i = 0
for cat in categories:
  input_dir = args.data_dir + '/' + cat + '/'
  print "Category %s labeled as %d" % (cat, i) 
  print "Generating training data..."
  output_prefix = args.output_dir + '/train/' + cat + '-'
  for train_file in categories[cat]['train']:
    src = input_dir + train_file
    dest = output_prefix + train_file
    suffix = ''
    if not src.endswith('jpg') and not src.endswith('jpeg') and not src.endswith('png'):
      dest = dest + '.' + imghdr.what(src)
      suffix = '.' + imghdr.what(src)
    shutil.copyfile(src, dest)
    train_list_file.write("%s-%s%s %d\n" % (cat, train_file, suffix, i))

  print "Generating validation data..."
  output_prefix = args.output_dir + '/val/' + cat + '-'
  for val_file in categories[cat]['val']:
    src = input_dir + val_file
    dest = output_prefix + val_file
    suffix = ''
    if not src.endswith('jpg') and not src.endswith('jpeg') and not src.endswith('png'):
      dest = dest + "." + imghdr.what(src)
      suffix = '.' + imghdr.what(src)
    shutil.copyfile(src, dest)
    val_list_file.write("%s-%s%s %d\n" % (cat, val_file, suffix, i))

  print "Generating testing data..."
  output_prefix = '%s/test/%d-%s' % (args.output_dir, i, cat)
  j = 0
  for test_file in categories[cat]['test']:
    src = input_dir + test_file
    dest = '%s-%d.%s' % (output_prefix, j, imghdr.what(src))
    shutil.copyfile(src, dest)
    j += 1
  i += 1

#pp.pprint(stats)
