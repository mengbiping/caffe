import Image,cv2,os,ImageFont,ImageDraw,time,scipy,scipy.misc,shutil,urllib,urllib2

img_dir_path = '../images.csv' # the path of image url file
count = 0
f1 = open(img_dir_path, "r")  
while True:  
    line = f1.readline()  
    count = count + 1 
    if count < 166854:
        print count
        continue
    if count >= 180000:
        break
    if line: 
        dot_num = line.find(',')
        color_name = line[:dot_num]
        dou_r_num = line.rfind('"')
        pic_url = line[dot_num+2:dou_r_num]
        dot_r_num = line.rfind('.')
        pic_format = line[dot_r_num:dou_r_num] 
        name = 'crawler/' + str(count) + pic_format # the path to save images
        print count, pic_url, name
        if pic_format == '' or len(pic_url) < 10:
            continue
        try:
            response = urllib.urlopen(pic_url).read()
	    f = open(name,'wb')  
            f.write(response)  
            f.close()
        except:
            continue
	finally:
            pass
    else:  
       break  
f1.close()  

