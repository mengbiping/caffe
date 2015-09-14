import pgmagick as pg
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

def sobel_background_remove(img, thres):
    """ Returns the mask of img with background-removed and the
    persentage of background part. """
    transmask = trans_mask_sobel(img, thres)
    img = alpha_composite(transmask,img)
    data = np.zeros([img.rows(),img.columns()])
    count = 0
    for i in range(img.rows()):
        for j in range(img.columns()):
            data[i,j] = 255 - img.pixelColor(j,i).alphaQuantum()
            count += data[i,j]
    return (data, 1 - count/255.0/img.rows()/img.columns())

def remove_background(filename, starting_thres=15,
        thres_decay_multiplier=3.0,
        max_background_persentage=0.79):
    """ Remove the background of the image in 'filename', returns a
    foreground mask."""
    img = pg.Image(filename.encode('utf-8'))
    thres = starting_thres
    background_persentage = 1.1
    while thres > 1 and background_persentage > max_background_persentage:
        (data, background_persentage) = sobel_background_remove(img, thres)
        thres /= float(thres_decay_multiplier)

    if background_persentage > max_background_persentage:
        # Too much background. Treat as no-background at all.
        data = np.full([img.rows(),img.columns()], 255)
    return data.astype('uint8')
