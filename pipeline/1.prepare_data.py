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
import random,cv2,imghdr,math
from background_remover import remove_background
from skin_detector import skin_detect

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="../../result_amazon")
parser.add_argument("--output_dir", default="result_data")
parser.add_argument("--test_count_in_each_type", default="70", type=int,
        help="The number instances reserved for testing for each category.")
parser.add_argument("--training_persentage", default="0.9", type=float,
        help="The persentage of data used for training.")
parser.add_argument("--validating_persentage", default="0.1", type=float,
        help="The persentage of data used for validation.")
parser.add_argument("--remove_background", default="True", type=bool,
        help="True to remove background from the images we process.")
parser.add_argument("--remove_skin", default="True", type=bool,
        help="True to remove skin from the images we process.")
args = parser.parse_args()

# Process image from src and write the result to dest.
def proc_and_copy_image (src, dest) :
  #print src
  if not args.remove_background and not args.remove_skin :
    shutil.copyfile(src, dest)
    return
  img = cv2.imread(src)
  fore = 255 * np.ones([img.shape[0],img.shape[1]])
  skinfore = 255 * np.ones([img.shape[0],img.shape[1]])
  if args.remove_background :
    fore = remove_background(src)
  if args.remove_skin :
    skinfore = skin_detect(img) 
  fore = cv2.bitwise_and(fore, fore, mask = skinfore) # the foreground mask 
  b = img[:,:,0]
  g = img[:,:,1]
  r = img[:,:,2]
  img_merge = cv2.merge((b,g,r,fore))
  dot_index = dest.rfind('.')
  dest = dest[:dot_index] + '.png'
  cv2.imwrite(dest, img_merge)

pp = pprint.PrettyPrinter(indent=2)

# print args.training_persentage
training_persentage = args.training_persentage / (args.training_persentage + args.validating_persentage)

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
    proc_and_copy_image(src, dest)
    suffix = '.png'
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
    proc_and_copy_image(src, dest)
    suffix = '.png'
    val_list_file.write("%s-%s%s %d\n" % (cat, val_file, suffix, i))

  print "Generating testing data..."
  output_prefix = '%s/test/%d-%s' % (args.output_dir, i, cat)
  j = 0
  for test_file in categories[cat]['test']:
    src = input_dir + test_file
    dest = '%s-%d.%s' % (output_prefix, j, imghdr.what(src))
    proc_and_copy_image(src, dest)
    j += 1
  i += 1

#pp.pprint(stats)
