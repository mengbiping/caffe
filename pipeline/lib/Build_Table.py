import Image,cv2,os,time,shutil
import numpy as np
import pickle
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from BackRemove_Mask_Simple import remove_background
from SkinDetect import skin_detect

color_dir_path = '/home/mengbiping/Documents/color-detection/color-0818/color_32' # the folder of color pictures

# recored the average rgb value for the color pictures as reference
color_ref_rgb = [] # array of [r,g,b] for the reference colors.
color_ref_lab = [] # array of lab object for the reference colors.
color_ref_name = [] # array of color names.
color_num = 0 # The total number of colors.
white_index = 0 # The index for white color.
for root, dirs, files in os.walk(color_dir_path) :
    for name in files:
        if name.endswith('.png') or name.endswith('.jpg') :
            filename = os.path.join(root, name)
            img = cv2.imread(filename)
            name_length = len(name)
            color_name = name[:name_length-4]
            rgb = tuple(img.mean(axis=0))[0]
            color_ref_rgb.append(rgb)
            rgb_obj = sRGBColor(rgb[2]/255.0, rgb[1]/255.0, rgb[0]/255.0)
            color_ref_lab.append(convert_color(rgb_obj, LabColor))
            color_ref_name.append(color_name)
            if color_name == 'White' :
                white_index = color_num
            color_num += 1

print "Building the color ref matrix."
rgb_to_color = np.zeros((256,256,256))
for i in range(256) :
    for j in range(256) :
        for k in range(256) :
            delta = 10000
            color_index = 0
            comparison_rgb = sRGBColor(i/255.0, j/255.0, k/255.0)
            comparison_lab = convert_color(comparison_rgb, LabColor)
            for l in range (len(color_ref_lab)) :
                delta_e = delta_e_cie2000(comparison_lab, color_ref_lab[l])
                if delta_e < delta:
                    delta = delta_e
                    color_index = l
            rgb_to_color[i,j,k] = color_index
    print "%d finished" % i

rgb_table_filehandler = open(b"rgb_to_color_table.obj","wb")
pickle.dump(rgb_to_color, rgb_table_filehandler)

metadat_filehandler = open(b"metadata.obj","wb")
metadata = {}
metadata['color_ref_rgb'] = color_ref_rgb
metadata['color_ref_lab'] = color_ref_lab
metadata['color_ref_name'] = color_ref_name
metadata['color_num'] = color_num
metadata['white_index'] = white_index
pickle.dump(metadata, metadat_filehandler)
