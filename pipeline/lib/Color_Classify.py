import Image,cv2,os,time,shutil
import numpy as np
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from BackRemove_Mask_Simple import remove_background
from SkinDetect import skin_detect

# calculate the reference common color
color_dir_path = 'color_32' # the folder of color pictures
img_dir_path = 'test_2_files' # the folder of images to process
result_dir_path = 'result_32/' # the folder to save the resulted images

# recored the average rgb value for the color pictures as reference
color_ref = {}
color_count_ = {}
color_rgb_name = {}
for root, dirs, files in os.walk(color_dir_path):
    for name in files:
        if name.endswith('.png') or name.endswith('.jpg'):
	    filename = os.path.join(root,name)
	    img = cv2.imread(filename)
    	    count = len(name)
	    color_name = name[:count-4]
	    color_ref[color_name] = tuple(img.mean(axis=0))[0]
            color_count_[color_name] = 0
	    if os.path.exists(result_dir_path + color_name):
		shutil.rmtree(result_dir_path + color_name)
            os.mkdir(result_dir_path + color_name)	    

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
            for (color_name_ref,col_rgb) in color_count_.items():
                color_count_[color_name_ref] = 0
	    for i in range(len(fore)):
	        for j in range(len(fore[0])):
		    if fore[i][j] == 255:
			key_num = (int(img[i][j][2]) * 256 + int(img[i][j][1])) * 256 + int(img[i][j][0])
			if color_rgb_name.has_key(key_num):
                            color_name = color_rgb_name[key_num]
                        else:
			    delta = 10**4
			    color_name = ''
			    color_rgb = sRGBColor(int(img[i][j][2])/255.0, int(img[i][j][1])/255.0, int(img[i][j][0])/255.0)
			    color_lab = convert_color(color_rgb,LabColor)
			    for (color_name_ref,col_rgb) in color_ref.items():
			        color_ref_rgb = sRGBColor(col_rgb[2]/255.0, col_rgb[1]/255.0, col_rgb[0]/255.0)
			        color_ref_lab = convert_color(color_ref_rgb,LabColor)
			        delta_e = delta_e_cie2000(color_lab, color_ref_lab)
			        if delta_e < delta:
			    	    delta = delta_e
			       	    color_name = color_name_ref
                            color_rgb_name[key_num] = color_name
	                color_count_[color_name] += 1
            
            color_count_['White'] *= 0.8  # for the case of blank area
	    
	    result_color = ''
            count_ref = 0
	    result_color_2 = ''
            count_ref_2 = 0
            for (color_name_ref,col_count) in color_count_.items():
	        if col_count > count_ref:
		    result_color_2 = result_color
            	    count_ref_2 = count_ref
		    count_ref = col_count
	            result_color = color_name_ref
	    #print result_color,count_ref,result_color_2,count_ref_2
	    
	    # output the original image and that with background and skin removed showed in alpha channel
            img = Image.open(filename)
	    filename1 =  result_dir_path + result_color + '/' + str(count)+'.jpg'
            img.save(filename1)

	    img = cv2.imread(filename,-1)
	    filename1 = result_dir_path + result_color + '/' + str(count)+'.png'
	    b = img[:,:,0]
	    g = img[:,:,1]
	    r = img[:,:,2]
            img_merge = cv2.merge((b,g,r,fore))
            cv2.imwrite(filename1,img_merge)

	    print("---image %d: %s, %s seconds ---" % (count, result_color, (time.time() - start_time)))





