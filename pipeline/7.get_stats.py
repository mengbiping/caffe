#!/usr/bin/python

import os
import shutil
import sys

def split(line):
  items = line.split(' ')
  assert len(items) == 3, "Bad Line: {}".format(line)
  filename = os.path.basename(items[0])
  is_digital_label = True
  try:
    label = int(items[1])
  except:
    is_digital_label = False

  if is_digital_label:
    dash_idx = filename.find('-')
    assert dash_idx > 0, "No dash after dot: {}".format(line)
    category = filename[:dash_idx]
  else:
    dot_idx = filename.find('.')
    assert dot_idx > 0, "No dot: {}".format(line)
    dot_idx += 1
    dash_idx = filename.find('-', dot_idx)
    assert dash_idx > 0, "No dash after dot: {}".format(line)
    category = filename[dot_idx:dash_idx]
  return (category, items[1], float(items[2]), items[0])


class CategoryStats:
  def __init__(self, category):
    self.category = category
    self.total = 0
    self.good = 0
    self.total_high = 0
    self.good_high = 0
    self.wrong_high_category = {}
    self.wrong_low_category = {}

  def update(self, classified_category, high_confidence):
    self.total += 1
    if high_confidence:
      self.total_high += 1
    if self.category == classified_category:
      self.good += 1
      if high_confidence:
        self.good_high += 1
    else:
      if classified_category not in self.wrong_low_category:
        self.wrong_low_category[classified_category] = 0
      if classified_category not in self.wrong_high_category:
        self.wrong_high_category[classified_category] = 0
      if high_confidence:
        self.wrong_high_category[classified_category] += 1
      else:
        self.wrong_low_category[classified_category] += 1

  def output(self):
    print('----------------- Category {}----------------------'.format(self.category))
    p = 1.0 * self.good / self.total
    print('T: {} G: {} P: {:0.3f}'.format(self.total, self.good, p))
    if self.total_high > 0:
      p = 1.0 * self.good_high / self.total_high
    else:
      p = 0.0
    print('High T: {} G: {} P:{:0.3f}'.format(self.total_high, self.good_high, p))
    p = 1 - 1.0 * self.total_high / self.total
    print('Dropped: {:0.3f}'.format(p))
    for key, value in self.wrong_high_category.iteritems():
      print('  {:10s} {:4d} {:4d}'.format(key, value, self.wrong_low_category[key]))


class Stats:
  def __init__(self):
    self.total = 0
    self.good = 0
    # total image classified with high confidence
    self.total_high = 0
    # total image classified as good category with high confidence
    self.good_high = 0
    # stats for each category
    self.categories = {}

  def update(self, category, classified_category, high_confidence):
    """Updates stats.

    Args:
      high_confidence: True if it's high confident result.
    """
    self.total += 1
    if high_confidence:
      self.total_high += 1
    if category == classified_category:
      self.good += 1
      if high_confidence:
        self.good_high += 1
    if category not in self.categories:
      self.categories[category] = CategoryStats(category)
    self.categories[category].update(classified_category, high_confidence)

  def output(self):
    p = 1.0 * self.good / self.total
    print('T: {} G: {} P: {:0.3f}'.format(self.total, self.good, p))
    if self.total_high > 0:
      p = 1.0 * self.good_high / self.total_high
    else:
      p = 0.0
    print('High T: {} G: {} P:{:0.3f}'.format(self.total_high, self.good_high, p))
    p = 1 - 1.0 * self.total_high / self.total
    print('Dropped: {:0.3f}'.format(p))
    for _, stat in self.categories.iteritems():
      stat.output()


if __name__ == '__main__':
  if len(sys.argv) < 3:
      print 'Usage: %s predication_file min_confidence [eval_dir] [data_dir]'
      sys.exit(-1)

  eval_dir = None
  data_dir = None
  if len(sys.argv) == 5:
    eval_dir = sys.argv[3]
    data_dir = sys.argv[4]

  stats = Stats()
  min_confidence = float(sys.argv[2])
  with open(sys.argv[1]) as ifile:
    for line in ifile:
      line = line.strip()
      if not line:
        continue
      category, classified_category, confidence, filename = split(line)
      is_high = not (confidence < min_confidence)
      stats.update(category, classified_category, is_high)
      if eval_dir and category != classified_category:
        items = os.path.splitext(filename)
        new_filename = '{}-{}{}'.format(items[0], classified_category, items[1])
        new_filename = os.path.join(eval_dir, new_filename)
        shutil.copy(os.path.join(data_dir, filename), new_filename)
  stats.output()

