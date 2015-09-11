import pgmagick as pg
# import Image,cv2
import numpy as np

def trans_mask_sobel(img, thres):
    """ Generate a transparency mask for a given image """
    image = pg.Image(img)

    # Find object
    image.negate()
    image.edge()
    image.blur(1)
    image.threshold(thres)
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
    img = pg.Image(filename.encode('utf-8'))
    transmask = trans_mask_sobel(img, 15)
    img = alpha_composite(transmask,img)
    flag = 1
    data = np.zeros([img.rows(),img.columns()])
    count = 0
    for i in range(img.rows()):
        for j in range(img.columns()):
	    data[i,j] = 255 - img.pixelColor(j,i).alphaQuantum()
            count += data[i,j] 
    #print 1 - count/255.0/img.rows()/img.columns()
    if 1 - count/255.0/img.rows()/img.columns() > 0.79:
        #print "too much background" 
        transmask = trans_mask_sobel(img, 5)
	img = alpha_composite(transmask,img)
	data = np.zeros([img.rows(),img.columns()])
	for i in range(img.rows()):
	    for j in range(img.columns()):
	        data[i,j] = 255 - img.pixelColor(j,i).alphaQuantum()
    data_u8 = data.astype('uint8')
    return data_u8
