import Image,cv2,os,time,shutil
import numpy as np
import argparse
import pickle
import ast
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_diff import delta_e_cie1976
from BackRemove_Mask_Simple import remove_background
from SkinDetect import skin_detect
#from multiprocessing import Pool
#import pathos.multiprocessing as mp
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

class ColorTableBuilder(object) :
    # twiddler is a dict that stores the mapping from color index to its distance penalty.
    def __init__(self, color_ref_lab, twiddler = {}) :
        # the table of rgb color to 
        self.rgb_to_color = np.full((256,256,256), -1)
        self._color_ref_lab = color_ref_lab
        self._twiddler = twiddler
        # color_index is the set of color index that will be rebuilt.
        # if empty computes all color_indexes.
        self._color_index = set()

    def load_color_table(self, rgb_to_color) :
        self.rgb_to_color = rgb_to_color

    def find_color(self, r, g, b) :
        # skip the computing if the current result color is not in _color_index.
        if len(self._color_index) > 0 and self.rgb_to_color[r,g,b] not in self._color_index and self.rgb_to_color[r,g,b] >= 0 :
            return self.rgb_to_color[r,g,b]
        comparison_rgb = sRGBColor(r/255.0, g/255.0, b/255.0)
        comparison_lab = convert_color(comparison_rgb, LabColor)
        distance = map(partial(delta_e_cie2000, comparison_lab), self._color_ref_lab)
        for i in range(len(distance)) :
            if self._twiddler.has_key(i) :
                distance[i] = self._twiddler[i] * distance[i]
        delta = min(distance)
        color_index = distance.index(delta)
        return color_index

    def reset_color(self, r, g, b) :
        self.rgb_to_color[r,g,b] = find_color(r, g, b)

    def _find_all_blue(self, r, g) :
        return map(partial(self.find_color, r, g), range(256))

    def build_color_table(self) :
        for i in range(256) :
            # pool = Pool(processes=8)
            # pool = mp.ProcessingPool(8)
            self.rgb_to_color[i] = map(partial(self._find_all_blue, i), range(256))
            # pool.close()
            # pool.join()
            print "%d finished" % i

    def set_color_index_to_compute(self, color_index) :
        self._color_index = color_index

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--color_ref_dir", default="/home/mengbiping/Documents/color-detection/color-0818/color_32")
    parser.add_argument("--input_metadata", default="metadata.obj")
    parser.add_argument("--output_metadata", default="")
    parser.add_argument("--input_rgb_to_color", default="rgb_to_color_table.obj")
    parser.add_argument("--output_rgb_to_color", default="new_rgb_to_color_table.obj")
    parser.add_argument("--colors_to_build", default="Silver,White,Ivory")
    parser.add_argument("--color_distance_penalty",
            default="{'Silver':1.4,'White':1.4,'Ivory':1.4}")
    args = parser.parse_args()

    metadata = None
    # Build metadata table
    if len(args.input_metadata) > 0 :
        with open(args.input_metadata, 'rb') as m_file:
            metadata = pickle.load(m_file)
        m_file.close()
    else :
        metadata = build_metadata(args.color_ref_dir)

    # Save metadata table
    if len(args.output_metadata) > 0 :
        with open(args.output_metadata, "wb") as m_file :
            pickle.dump(metadata, m_file)
        m_file.close()

    # Load color distance penalty.
    color_distance_penalty = ast.literal_eval(args.color_distance_penalty)
    color_index_distance_penalty = {}
    for i in range(metadata['color_num']) :
        color_name = metadata['color_ref_name'][i]
        if color_name in color_distance_penalty :
            penalty = color_distance_penalty[color_name]
            print "Color penalty: %s %f" % (color_name, penalty)
            color_index_distance_penalty[i] = penalty
    
    # Try to load existing color table.
    color_table_builder = ColorTableBuilder(metadata["color_ref_lab"],
            color_index_distance_penalty)
    if len(args.input_rgb_to_color) > 0 :
        with open(args.input_rgb_to_color, "rb") as r_file :
            rgb_to_color = pickle.load(r_file)
            color_table_builder.load_color_table(rgb_to_color)
        r_file.close()

    # Load color indexes to rebuild.
    color_index = set()
    colors_to_build = args.colors_to_build.split(",")
    for i in range(metadata['color_num']) :
        if metadata['color_ref_name'][i] in colors_to_build :
            print "Rebuilding color: %s" % metadata['color_ref_name'][i]
            color_index.add(i)
    color_table_builder.set_color_index_to_compute(color_index)

    print "Building the color ref matrix."
    color_table_builder.build_color_table()

    # Output color table.
    with open(args.output_rgb_to_color, "wb") as r_file :
        pickle.dump(color_table_builder.rgb_to_color, r_file)
    r_file.close()

if __name__ == "__main__":
    main()
