import Image,cv2,os,time,shutil
import numpy as np
import argparse
import pickle
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from BackRemove_Mask_Simple import remove_background
from SkinDetect import skin_detect
from Build_Table import build_metadata
from Build_Table import find_ref_color

class ColorClassifier:
  def __init__(self, metadata_file_path, ref_color_table_file_path,
          decay_colors, decay_percentage, max_skin_percentage) :
    metadata = None
    with open(metadata_file_path, 'r') as m_file:
      metadata = pickle.load(m_file)
    m_file.close()
    self.initialize(metadata, decay_colors, decay_percentage, max_skin_percentage)

    self.rgb_to_color = None
    with open(ref_color_table_file_path, 'r') as r_file:
      self.rgb_to_color = pickle.load(r_file)
    r_file.close()

  def __init__(self, color_ref_dir, decay_colors, decay_percentage, max_skin_percentage) :
    metadata = build_metadata(color_ref_dir)
    self.initialize(metadata, decay_colors, decay_percentage, max_skin_percentage)
    self.rgb_to_color = np.full((256,256,256), -1)

  def initialize(self, metadata, decay_colors, decay_percentage, max_skin_percentage) :
    self.color_ref_rgb = metadata['color_ref_rgb'] # array of [r,g,b] for the reference colors.
    self.color_ref_lab = metadata['color_ref_lab'] # array of lab object for the reference colors.
    self.color_ref_name = metadata['color_ref_name'] # array of color names.
    self.color_num = metadata['color_num'] # The total number of colors.
    self.decay_color_index = set() # The index for colors to be decayed, e.g., white and silver.
    decay_color_name = decay_colors.split(",")
    for i in range(len(self.color_ref_name)) :
      if self.color_ref_name[i] in decay_color_name :
        print "Decaying color: %s" % self.color_ref_name[i]
        self.decay_color_index.add(i)
    self.decay_percentage = decay_percentage
    self.max_skin_percentage = max_skin_percentage

  def detect(self, image_file_path) :
    img = cv2.imread(image_file_path)
    start_time = time.time()
    # generate the foreground
    fore = remove_background(image_file_path) # background removal
    fore1 = fore
    skinfore = skin_detect(image_file_path) # skin removal
    fore = cv2.bitwise_and(fore, fore, mask = skinfore) # the foreground mask
    # In case of clothes in skin color.
    if 1 - sum(map(sum, fore))/1.0/sum(map(sum, fore1)) > self.max_skin_percentage:
      print "Too much skin"
    fore = fore1
  
    # Find the nearest reference color for each pixel and count
    color_count = [0] * self.color_num
    for i in range(len(fore)):
      for j in range(len(fore[0])):
        if fore[i][j] != 255:
          continue
        if self.rgb_to_color[img[i][j][2], img[i][j][1], img[i][j][0]] < 0 :
          self.rgb_to_color[img[i][j][2], img[i][j][1], img[i][j][0]] = find_ref_color(self.color_ref_lab, img[i][j][2], img[i][j][1], img[i][j][0])
  
        color_index = int(self.rgb_to_color[img[i][j][2], img[i][j][1], img[i][j][0]])
        color_count[color_index] += 1

    # Decay colors.
    for decay_color in self.decay_color_index :
      color_count[decay_color] *= self.decay_percentage
  
    max_color_count = max(color_count)
    return (color_count.index(max_color_count), fore)

  def color_index_to_name(self, index) :
    return self.color_ref_name[int(index)]

def main() :
  parser = argparse.ArgumentParser()
  parser.add_argument("--metadata", default="")
  parser.add_argument("--rgb_to_color", default="")
  parser.add_argument("--colors_to_decay", default="White,Silver,Black")
  parser.add_argument("--decay_percentage", default="0.7", type=float)
  parser.add_argument("--max_skin_percentage", default="0.6", type=float)
  parser.add_argument("--color_ref_dir", default="/home/mengbiping/Documents/color-detection/color-0818/color_32")
  parser.add_argument("--image_dir", default="bad_cases")
  parser.add_argument("--result_dir", default="result_32_2/")
  args = parser.parse_args()
  classifier = None
  if len(args.metadata) > 0 and len(args.rgb_to_color) > 0 :
    classifier = ColorClassifier(args.metadata, args.rgb_to_color,
            args.colors_to_decay, args.decay_percentage,
            args.max_skin_percentage)
  else :
    assert len(args.color_ref_dir) > 0
    classifier = ColorClassifier(args.color_ref_dir, args.colors_to_decay,
            args.decay_percentage, args.max_skin_percentage)
  count = 0
  for name in sorted(os.listdir(args.image_dir)) :
    if not name.endswith('.jpg'):# search for all the jpg formate images for processing
      continue
    print "Processing image No. %d ..." % count
    count = count + 1
    filename = os.path.join(args.image_dir, name)
    start_time = time.time()
    (result_color_index, foreground_mask) = classifier.detect(filename)
    result_color_name = classifier.color_index_to_name(result_color_index)

    # output the original image and that with background and skin removed showed in alpha channel
    img = Image.open(filename)
    filename1 =  args.result_dir + result_color_name + '/' + str(count)+'.jpg'
    img.save(filename1)

    img = cv2.imread(filename,-1)
    filename1 = args.result_dir + result_color_name + '/' + str(count)+'.png'
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    img_merge = cv2.merge((b, g, r, foreground_mask))
    cv2.imwrite(filename1, img_merge)

    print("---image %d: %s, %s seconds ---" % (count, result_color_name, (time.time() - start_time)))

if __name__ == "__main__":
  main()
