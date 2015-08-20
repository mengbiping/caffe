import pgmagick as pg
import Image,cv2
import numpy as np

def trans_mask_sobel(img):
    """ Generate a transparency mask for a given image """

    image = pg.Image(img)

    # Find object
    image.negate()
    image.edge()
    image.blur(1)
    image.threshold(24)
    image.adaptiveThreshold(5, 5, 5)

    # Fill background
    image.fillColor('magenta')
    w, h = image.size().width(), image.size().height()
    image.floodFillColor('0x0', 'magenta')
    image.floodFillColor('0x0+%s+0' % (w-1), 'magenta')
    image.floodFillColor('0x0+0+%s' % (h-1), 'magenta')
    image.floodFillColor('0x0+%s+%s' % (w-1, h-1), 'magenta')

    image.transparent('magenta')
    return image

def alpha_composite(image, mask):
    """ Composite two images together by overriding one opacity channel """

    compos = pg.Image(mask)
    compos.composite(
        image,
        image.size(),
        pg.CompositeOperator.CopyOpacityCompositeOp
    )
    return compos

def remove_background(filename):
    """ Remove the background of the image in 'filename' """
    img = pg.Image(filename)
    image = cv2.imread(filename)
    transmask = trans_mask_sobel(img)
    img = alpha_composite(transmask,img)
    flag = 1
    # if the foreground is not segmented well, we will not remove the background 
    data = np.zeros([img.rows(),img.columns()])
    count = 0
    xmin = img.rows()
    xmax = 0
    ymin = img.columns()
    ymax = 0
    for i in range(img.rows()):
        for j in range(img.columns()):
	    data[i,j] = 255 - img.pixelColor(j,i).alphaQuantum()
            count += data[i,j]
	    if data[i,j] > 0:
		if i < xmin:
 		    xmin = i
		if i > xmax:
 		    xmax = i
		if j < ymin:
 		    ymin = j
		if j > ymax:
 		    ymax = j
    diff = (xmax-xmin)/float(img.rows())*(ymax-ymin)/float(img.columns())
    ratio = 1 - count/255/img.rows()/img.columns()# the ration of background
    if ratio > 0.85 and diff >= 0.5:
	data = 255 - np.zeros([img.rows(),img.columns()])
	flag = 0
    data_u8 = data.astype('uint8')
    return data_u8, flag



