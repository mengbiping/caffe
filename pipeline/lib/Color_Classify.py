import Image,cv2,os,time,shutil
import numpy as np
import argparse
import pickle
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from BackRemove_Mask_Simple import remove_background
from SkinDetect import skin_detect

parser = argparse.ArgumentParser()
parser.add_argument("--metadata", default="metadata.obj")
parser.add_argument("--rgb_to_color", default="rgb_to_color_table.obj")
parser.add_argument("--colors_to_decay", default="White,Silver,Black")
parser.add_argument("--decay_percentage", default="0.7", type=float)
args = parser.parse_args()

# calculate the reference common color
color_dir_path = '/home/mengbiping/Documents/color-detection/color-0818/color_32' # the folder of color pictures
img_dir_path = '/home/mengbiping/caffe/data/clothes_neck/train' # the folder of images to process
result_dir_path = 'result_32/' # the folder to save the resulted images

# recored the average rgb value for the color pictures as reference
color_ref_rgb = [] # array of [r,g,b] for the reference colors.
color_ref_lab = [] # array of lab object for the reference colors.
color_ref_name = [] # array of color names.
color_num = 0 # The total number of colors.

with open(args.metadata, 'r') as m_file:
    metadata = pickle.load(args.metadata)
    color_ref_rgb = metadata['color_ref_rgb']
    color_ref_lab = metadata['color_ref_lab']
    color_ref_name = metadata['color_ref_name']
    color_num = metadata['color_num']
m_file.close()

decay_color_index = set() # The index for colors to be decayed, e.g., white and silver.
decay_color_name = args.colores_to_decay.split(",")
for i in range(len(color_ref_name)) :
    if color_ref_name[i] in decay_color_name :
        print "Decaying color: %s" . color_ref_name[i]
        decay_color_index.add(i)

rgb_to_color = None
with open(args.rgb_to_color, 'r') as r_file:
    rgb_to_color = pickle.load(args.rgb_to_color)
r_file.close()

# process images
count = 0
for root, dirs, files in os.walk(img_dir_path):
    for name in files:
        if name.endswith('.jpg'):# search for all the jpg formate images for processing
	    filename = os.path.join(root,name)
	    img = cv2.imread(filename)
            print "Processing image No. %d ..." % count
	    count = count + 1
	    start_time = time.time()
	    # generate the foreground
            fore = remove_background(filename) # background removal
            fore1 = fore
	    skinfore = skin_detect(filename) # skin removal
            fore = cv2.bitwise_and(fore, fore, mask = skinfore) # the foreground mask 
	    if 1 - sum(map(sum, fore))/1.0/sum(map(sum, fore1)) > 0.58: # for the case where clothes is regared as skin
                print "too much skin"
		fore = fore1

	    # find the nearest reference color for each pixel and count
            color_count = [0] * color_num
	    for i in range(len(fore)):
	        for j in range(len(fore[0])):
		    if fore[i][j] == 255:
                        color_index = rgb_to_color[img[i][j][2], img[i][j][1], img[i][j][0]]
	                color_count[color_index] += 1
            
            for decay_color in decay_color_index
                color_count[decay_color] *= args.decay_percentage # for the case of blank area
	    
	    result_color_index = 0
            count_ref = 0
	    result_color_2 = ''
            count_ref_2 = 0
            for l in range(len(color_count)):
	        if color_count[l] > count_ref:
		    result_color_2 = result_color_index
            	    count_ref_2 = count_ref
		    count_ref = color_count[l]
	            result_color_index = l
	    #print result_color_index,count_ref,result_color_2,count_ref_2
	    
	    # output the original image and that with background and skin removed showed in alpha channel
            img = Image.open(filename)
	    filename1 =  result_dir_path + colore_ref_name[result_color_index] + '/' + str(count)+'.jpg'
            img.save(filename1)

	    img = cv2.imread(filename,-1)
	    filename1 = result_dir_path + colore_ref_name[result_color_index] + '/' + str(count)+'.png'
	    b = img[:,:,0]
	    g = img[:,:,1]
	    r = img[:,:,2]
            img_merge = cv2.merge((b,g,r,fore))
            cv2.imwrite(filename1,img_merge)

	    print("---image %d: %s, %s seconds ---" % (count, colore_ref_name[result_color_index], (time.time() - start_time)))
