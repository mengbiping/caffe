import Image,cv2,os,time,shutil
import numpy as np
import argparse
import pickle
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_diff import delta_e_cie1976
from BackRemove_Mask_Simple import remove_background
from SkinDetect import skin_detect
from multiprocessing import Pool
from functools import partial

def build_metadata(color_ref_dir):
    metadata = {}
    color_ref_rgb = [] # array of [r,g,b] for the reference colors.
    color_ref_lab = [] # array of lab object for the reference colors.
    color_ref_name = [] # array of color names.
    color_num = 0 # The total number of colors.
    for name in sorted(os.listdir(color_ref_dir)) :
        if name.endswith('.png') or name.endswith('.jpg') :
            filename = os.path.join(color_ref_dir, name)
            img = cv2.imread(filename)
            name_length = len(name)
            color_name = name[:name_length-4]
            rgb = tuple(img.mean(axis=0))[0]
            color_ref_rgb.append(rgb)
            rgb_obj = sRGBColor(rgb[2]/255.0, rgb[1]/255.0, rgb[0]/255.0)
            color_ref_lab.append(convert_color(rgb_obj, LabColor))
            color_ref_name.append(color_name)
            color_num += 1
    metadata['color_ref_rgb'] = color_ref_rgb
    metadata['color_ref_lab'] = color_ref_lab
    metadata['color_ref_name'] = color_ref_name
    metadata['color_num'] = color_num
    return metadata

def find_ref_color(color_ref_lab, r, g, b) :
    comparison_rgb = sRGBColor(r/255.0, g/255.0, b/255.0)
    comparison_lab = convert_color(comparison_rgb, LabColor)
    result = map(partial(delta_e_cie2000, comparison_lab), color_ref_lab)
    delta = min(result)
    color_index = result.index(delta)
    return color_index

def find_all_blue(color_ref_lab, r, g) :
    return map(partial(find_ref_color, color_ref_lab, r, g), range(256))

def build_color_table(color_ref_lab) :
    rgb_to_color = np.zeros((256,256,256))
    for i in range(256) :
        pool = Pool(processes=8)
        rgb_to_color[i] = pool.map(partial(find_all_blue, color_ref_lab, i), range(256))
        # for j in range(256) :
        #   for k in range(256) :
        #        rgb_to_color[i,j,k] = find_ref_color(i, j, k, color_ref_lab)
        pool.close()
        pool.join()
        print "%d finished" % i
    return rgb_to_color

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="metadata.obj")
    parser.add_argument("--rgb_to_color", default="rgb_to_color_table.obj")
    parser.add_argument("--color_ref_dir", default="/home/mengbiping/Documents/color-detection/color-0818/color_32")
    args = parser.parse_args()

    metadat_filehandler = open(b"metadata.obj","wb")
    metadata = build_metadata(args.color_ref_dir)
    pickle.dump(metadata, metadat_filehandler)

    print "Building the color ref matrix."
    rgb_table_filehandler = open(b"rgb_to_color_table.obj","wb")
    rgb_to_color = build_color_table(metadata['color_ref_lab'])
    pickle.dump(rgb_to_color, rgb_table_filehandler)

if __name__ == "__main__":
    main()
