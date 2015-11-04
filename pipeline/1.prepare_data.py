#!/usr/bin/python
# Take a folder as input. The folder contains several folders in it, each
# containing certain number of images. We treat the folder names as the label
# for the containing images.
# The output will be:
# 1. two folders of images, train and val
# 2. two text files, train.txt and val.txt containing the list of image file
# names with their labels.
# 3. number_of_categories * args.test_count_in_each_category images in test folder.

import argparse
import cv2
import imghdr
import math
import md5
import numpy as np
import os
import random
import shutil
import sys

_script_path = os.path.dirname(os.path.abspath( __file__ ))
sys.path.insert(0, os.path.join(_script_path, 'lib'))

from background_remover import remove_background
from skin_detector import skin_detect

args = None  # commandline flags
images_for_category = {}  # parsed from commandline max_images_for_category


def parse_args():
  """Parse commandline flags"""
  global args, images_for_category
  parser = argparse.ArgumentParser()
  parser.add_argument("data_dir")
  parser.add_argument("--output_dir", default=".")
  parser.add_argument("--test_count_in_each_category", default="70", type=int,
          help="The number instances reserved for testing for each category.")
  parser.add_argument("--training_percentage", default="0.9", type=float,
          help="The persentage of data used for training.")
  parser.add_argument("--validating_percentage", default="0.1", type=float,
          help="The persentage of data used for validation.")
  parser.add_argument("--remove_background", default=False, action="store_true",
          help="True to remove background from the images we process.")
  parser.add_argument("--remove_skin", default=False, action="store_true",
          help="True to remove skin from the images we process.")
  parser.add_argument("--max_images_for_train_per_category", default="10000",
          type=int, help="Maximum images used for train per category.")
  parser.add_argument("--max_images_for_category", default="", type=str,
          help=("Specify number of images for category, the format " +
            "is category1:count1,category2:count2"))

  args = parser.parse_args()
  if args.max_images_for_category:
    items = args.max_images_for_category.split(',')
    for item in items:
      cat_count = item.split(':')
      images_for_category[cat_count[0]] = int(cat_count[1])
    print('Set image count for category: {}'.format(images_for_category))


def proc_and_copy_image (src, dest) :
  """Process image from src and write the result to dest."""
  if not args.remove_background and not args.remove_skin :
    shutil.copyfile(src, dest)
    return dest
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
  return dest


def load_images_recursively(input_dir):
  """Load images from input_dir and sub directories.

  Returns:
    files list relative to input dir
  """
  files = []
  for dirname, subdir, filenames in os.walk(input_dir):
    print len(filenames), 'files in directory', dirname
    striped_dirname = dirname.lstrip(input_dir + '/')
    for filename in filenames:
      # Skip non image file.
      if not imghdr.what(os.path.join(dirname, filename)):
        continue
      files.append(os.path.join(striped_dirname, filename))
  return files


def load_images_for_category(cat, training_percentage):
  """Load images filenames for category."""
  print 'Loading category', cat, '...'
  category = {}
  category['all'] = load_images_recursively(
      os.path.join(args.data_dir, cat))
  print 'Loaded', len(category['all']), 'files for', cat
  category['test'] = random.sample(category['all'],
                                          args.test_count_in_each_category)
  rest = [item for item in category['all'] if item not in
          category['test']]
  # print rest
  training_count = int(math.floor(len(rest) * training_percentage))
  max_trains = images_for_category.get(cat,
      args.max_images_for_train_per_category)
  if max_trains > 0 and training_count > max_trains:
    training_count = max_trains
  # print training_count
  category['train'] = random.sample(rest, training_count)
  val_count = int(training_count * args.validating_percentage /
      args.training_percentage)
  val_set = [item for item in rest if item not in category['train']]
  if len(val_set) > val_count:
    category['val'] = random.sample(val_set, val_count)
  else:
    category['val'] = val_set

  assert len(category['test']) == args.test_count_in_each_category
  assert len(category['train']) > 0
  assert len(category['val']) > 0
  return category


def load_images():
  """Load images filenames from input directory.
  
  Returns:
    A map[cat][usage] contains all image files.
  """
  # print args.training_percentage
  training_percentage = args.training_percentage / (args.training_percentage +
      args.validating_percentage)

  categories = {}
  for cat in os.listdir(args.data_dir) :
    if not os.path.isdir(os.path.join(args.data_dir, cat)):
      continue
    categories[cat] = load_images_for_category(cat, training_percentage)
  return categories


def generate_image_list(images, input_dir, output_prefix, label):
  """Generate image list for category and image type.

  Args:
    images: images list
    input_dir: input directory for image file
    output_prefix: prefix of output file path
    label: the index label of this images set

  Returns:
    The list of image files.
  """
  image_list = []
  output_dir = args.output_dir
  if not output_dir.endswith('/'):
    output_dir += '/'
  for filename in images:
    src = os.path.join(input_dir, filename)
    # the filename could contains directory structure
    filename = filename.replace('/', '-')
    dest = output_prefix + filename
    if (not src.endswith('jpg') and not src.endswith('jpeg') and
        not src.endswith('png')):
      dest = dest + '.' + imghdr.what(src)
    dest = md5filename(dest)
    dest = proc_and_copy_image(src, dest)
    image_list.append("%s %d" % (os.path.basename(dest), label))
  return image_list


def prepare_output_directories():
  """Create and clear the directories in output_dir."""
  subdirectories = ['train', 'val', 'test']
  for subdirectory in subdirectories:
    subdirectory = os.path.join(args.output_dir, subdirectory)
    if os.path.isdir(subdirectory):
      shutil.rmtree(subdirectory)
    os.makedirs(subdirectory)


def md5filename(dest):
  """Replace the filename with the md5 of filename."""
  dirname = os.path.dirname(dest)
  basename = os.path.basename(dest)
  names = os.path.splitext(basename)
  basename = md5.new(names[0]).hexdigest() + names[1]
  return os.path.join(dirname, basename)


def generate_images_for_train_and_test(categories):
  """Copy the images to output directories and output text file."""
  label = 0
  train_file_list = []
  val_file_list = []
  for cat in categories:
    input_dir = os.path.join(args.data_dir, cat)
    print "Category %s labeled as %d" % (cat, label) 
    print "Generating training data..."
    output_prefix = os.path.join(args.output_dir, 'train', cat + '-')
    l = generate_image_list(categories[cat]['train'], input_dir, output_prefix,
        label)
    train_file_list.extend(l)

    print "Generating validation data..."
    output_prefix = os.path.join(args.output_dir, 'val', cat + '-')
    l = generate_image_list(categories[cat]['val'], input_dir, output_prefix,
        label)
    val_file_list.extend(l)

    print "Generating testing data..."
    output_prefix = os.path.join(args.output_dir, 'test',
        '%d-%s-' % (label, cat))
    j = 0
    for test_file in categories[cat]['test']:
      src = os.path.join(input_dir, test_file)
      dest = '%s-%05d.%s' % (output_prefix, j, imghdr.what(src))
      proc_and_copy_image(src, dest)
      j += 1

    label += 1

  print 'Writing train.txt with files', len(train_file_list)
  with open(os.path.join(args.output_dir, 'train.txt'), 'w') as ofile:
    ofile.write('\n'.join(train_file_list))

  print 'Writing val.txt with files', len(val_file_list)
  with open(os.path.join(args.output_dir, 'val.txt'), 'w') as ofile:
    ofile.write('\n'.join(val_file_list))

if __name__ == '__main__':
    parse_args()
    categories = load_images()
    prepare_output_directories()
    generate_images_for_train_and_test(categories)
