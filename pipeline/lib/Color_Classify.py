import Image,cv2,os,time,shutil
import numpy as np
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from BackRemove_Mask_Simple import remove_background
from SkinDetect import skin_detect

# calculate the reference common color
color_dir_path = '/home/mengbiping/Documents/color-detection/color-0818/color_32' # the folder of color pictures
img_dir_path = '/home/mengbiping/caffe/data/clothes_neck/train' # the folder of images to process
result_dir_path = 'result_32/' # the folder to save the resulted images

# recored the average rgb value for the color pictures as reference
color_ref_rgb = [] # array of [r,g,b] for the reference colors.
color_ref_lab = [] # array of lab object for the reference colors.
color_ref_name = [] # array of color names.
color_num = 0 # The total number of colors.
white_index = 0 # The index for white color.
for root, dirs, files in os.walk(color_dir_path):
    for name in files:
        if name.endswith('.png') or name.endswith('.jpg'):
	    filename = os.path.join(root,name)
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
	    if os.path.exists(result_dir_path + color_name):
		shutil.rmtree(result_dir_path + color_name)
            os.mkdir(result_dir_path + color_name)	    
            color_num += 1

print "Building the color ref matrix."
rgb_to_color = np.zeros((256,256,256))
for i in range(256) :
    print "%d finished" % i
    for j in range(256) :
        for k in range(256) :
            delta = 10000
            color_index = 0
            comparison_rgb = sRGBColor(k/255.0, j/255.0, i/255.0)
            comparison_lab = convert_color(comparison_rgb, LabColor)
            for l in range (len(color_ref_lab)) :
                delta_e = delta_e_cie2000(comparison_lab, color_ref_lab[l])
                if delta_e < delta:
                    delta = delta_e
                    color_index = l
            rgb_to_color[i,j,k] = color_index


# process images
count = 0
for root, dirs, files in os.walk(img_dir_path):
    for name in files:
        if name.endswith('.jpg'):# search for all the jpg formate images for processing
	    filename = os.path.join(root,name)
	    img = cv2.imread(filename)
	    count = count + 1
            print count
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
                        color_index = rgb_to_color[img[i][j][0], img[i][j][1], img[i][j][2]]
	                color_count[color_index] += 1
            
            color_count[white_index] *= 0.8  # for the case of blank area
	    
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
