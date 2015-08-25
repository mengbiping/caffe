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

# Prepare metadata.
metadata = None
if len(args.metadata) > 0 :
  with open(args.metadata, 'r') as m_file:
      metadata = pickle.load(m_file)
  m_file.close()
else :
  metadata = build_metadata(args.color_ref_dir)
color_ref_rgb = metadata['color_ref_rgb'] # array of [r,g,b] for the reference colors.
color_ref_lab = metadata['color_ref_lab'] # array of lab object for the reference colors.
color_ref_name = metadata['color_ref_name'] # array of color names.
color_num = metadata['color_num'] # The total number of colors.

# Prepare decay colors.
decay_color_index = set() # The index for colors to be decayed, e.g., white and silver.
decay_color_name = args.colors_to_decay.split(",")
for i in range(len(color_ref_name)) :
  if color_ref_name[i] in decay_color_name :
    print "Decaying color: %s" % color_ref_name[i]
    decay_color_index.add(i)

# Prepare the mapping of rgb -> ref_color.
rgb_to_color = None
if len(args.rgb_to_color) > 0 :
  with open(args.rgb_to_color, 'r') as r_file:
      rgb_to_color = pickle.load(r_file)
  r_file.close()
else :
  rgb_to_color = np.full((256,256,256), -1)

# Prepare resulting dir.
if not os.path.exists(args.result_dir)
  os.mkdir(args.result_dir)
for name in color_ref_name :
  path = os.path.join(args.result_dir, name)
  if not os.path.exists(path)
    os.mkdir(path)

# Process images
count = 0
for name in sorted(os.listdir(args.image_dir)) :
  if not name.endswith('.jpg'):# search for all the jpg formate images for processing
    continue
  filename = os.path.join(args.image_dir, name)
  img = cv2.imread(filename)
  print "Processing image No. %d ..." % count
  count = count + 1
  start_time = time.time()
  # generate the foreground
  fore = remove_background(filename) # background removal
  fore1 = fore
  skinfore = skin_detect(filename) # skin removal
  fore = cv2.bitwise_and(fore, fore, mask = skinfore) # the foreground mask
  # In case of clothes in skin color.
  if 1 - sum(map(sum, fore))/1.0/sum(map(sum, fore1)) > args.max_skin_percentage:
    print "Too much skin"
  fore = fore1

  # find the nearest reference color for each pixel and count
  color_count = [0] * color_num
  for i in range(len(fore)):
    for j in range(len(fore[0])):
      if fore[i][j] != 255:
        continue
      if rgb_to_color[img[i][j][2], img[i][j][1], img[i][j][0]] < 0 :
        rgb_to_color[img[i][j][2], img[i][j][1], img[i][j][0]] = find_ref_color(img[i][j][2], img[i][j][1], img[i][j][0], color_ref_lab)

      color_index = int(rgb_to_color[img[i][j][2], img[i][j][1], img[i][j][0]])
      color_count[color_index] += 1
      for decay_color in decay_color_index :
        color_count[decay_color] *= args.decay_percentage # for the case of blank area

  max_color_count = max(color_count)
  result_color_index = color_count.index(max_color_count)
  #print result_color_index,count_ref,result_color_2,count_ref_2

  # output the original image and that with background and skin removed showed in alpha channel
  img = Image.open(filename)
  filename1 =  args.result_dir + color_ref_name[result_color_index] + '/' + str(count)+'.jpg'
  img.save(filename1)

  img = cv2.imread(filename,-1)
  filename1 = args.result_dir + color_ref_name[result_color_index] + '/' + str(count)+'.png'
  b = img[:,:,0]
  g = img[:,:,1]
  r = img[:,:,2]
  img_merge = cv2.merge((b,g,r,fore))
  cv2.imwrite(filename1,img_merge)

  print("---image %d: %s, %s seconds ---" % (count, color_ref_name[result_color_index], (time.time() - start_time)))
