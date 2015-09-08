import Image,cv2,os
from background_remover import remove_background
from skin_detector import skin_detect

# calculate the reference common color
img_dir_path = 'test_files' # the folder of images to process
result_dir_path = 'result_32/' # the folder to save the resulted images
    
# process images
count = 0
for root, dirs, files in os.walk(img_dir_path):
    for name in files:
        if name.endswith('.jpg'):# search for all the jpg formate images for processing
	    filename = os.path.join(root,name)
	    img = cv2.imread(filename)
	    count = count + 1
            print count
	    # generate the foreground
            fore = remove_background(filename) # background removal
            fore1 = fore
	    skinfore = skin_detect(filename) # skin removal
            fore = cv2.bitwise_and(fore, fore, mask = skinfore) # the foreground mask 
	    if 1 - sum(map(sum, fore))/1.0/sum(map(sum, fore1)) > 0.58: # for the case where clothes is regared as skin
                print "too much skin"
		fore = fore1
	    
	    # output the original image and that with background and skin removed showed in alpha channel
            img = Image.open(filename)
	    filename1 =  result_dir_path + str(count)+'.jpg'
            img.save(filename1)

	    img = cv2.imread(filename,-1)
	    filename1 = result_dir_path + str(count)+'.png'
	    b = img[:,:,0]
	    g = img[:,:,1]
	    r = img[:,:,2]
            img_merge = cv2.merge((b,g,r,fore))
            cv2.imwrite(filename1,img_merge)
