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

from background_remover import remove_background
from skin_detector import skin_detect

args = None  # commandline flags


def parse_args():
  """Parse commandline flags"""
  global args
  parser = argparse.ArgumentParser()
  parser.add_argument("data_dir")
  parser.add_argument("--output_dir", default=".")
  parser.add_argument("--test_count_in_each_category", default="70", type=int,
          help="The number instances reserved for testing for each category.")
  parser.add_argument("--training_percentage", default="0.9", type=float,
          help="The persentage of data used for training.")
  parser.add_argument("--validating_percentage", default="0.1", type=float,
          help="The persentage of data used for validation.")
  parser.add_argument("--remove_background", default="True", type=bool,
          help="True to remove background from the images we process.")
  parser.add_argument("--remove_skin", default="True", type=bool,
          help="True to remove skin from the images we process.")
  parser.add_argument("--max_images_for_train_per_category", default="10000",
          type=int, help="Maximum images used for train per category.")
  args = parser.parse_args()


def proc_and_copy_image (src, dest) :
  """Process image from src and write the result to dest."""
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
  print src, dest
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
    print 'Loading category', cat, '...'
    categories[cat] = {}
    categories[cat]['all'] = load_images_recursively(
        os.path.join(args.data_dir, cat))
    print 'Loaded', len(categories[cat]['all']), 'files for', cat
    categories[cat]['test'] = random.sample(categories[cat]['all'],
                                            args.test_count_in_each_category)
    rest = [item for item in categories[cat]['all'] if item not in
            categories[cat]['test']]
    # print rest
    training_count = int(math.floor(len(rest) * training_percentage))
    max_trains = args.max_images_for_train_per_category
    if max_trains > 0 and training_count > max_trains:
      training_count = max_trains
    # print training_count
    categories[cat]['train'] = random.sample(rest, training_count)
    val_count = int(training_count * args.validating_percentage /
        args.training_percentage)
    val_set = [item for item in rest if item not in categories[cat]['train']]
    if len(val_set) > val_count:
      categories[cat]['val'] = random.sample(val_set, val_count)
    else:
      categories[cat]['val'] = val_set

    assert len(categories[cat]['test']) == args.test_count_in_each_category
    assert len(categories[cat]['train']) > 0
    assert len(categories[cat]['val']) > 0
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
    os.mkdir(subdirectory)


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
