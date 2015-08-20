from pyimagesearch import imutils
import numpy as np
import cv2,Image,os
def skin_detect(imagepath):
	# define the upper and lower boundaries of the HSV pixel
	# intensities to be considered 'skin'
	lower = np.array([0, 48, 80], dtype = "uint8")
	upper = np.array([20, 255, 255], dtype = "uint8")
	
	img = cv2.imread(imagepath)
	# convert it to the HSV color space,
	# and determine the HSV pixel intensities that fall into
	# the speicifed upper and lower boundaries
	frame = img

	frame = imutils.resize(img, width = 400)
	
	converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	skinMask = cv2.inRange(converted, lower, upper)

	# apply a series of erosions and dilations to the mask
	# using an elliptical kernel
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
	skinMask = cv2.erode(skinMask, kernel, iterations = 2)
	skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

	# blur the mask to help remove noise, then apply the
	# mask to the frame
	skinMask = 255 - cv2.GaussianBlur(skinMask, (3, 3), 0)
	skinMask = imutils.resize(skinMask, height=img.shape[0], width = img.shape[1])
	skinMask=cv2.resize(skinMask,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_CUBIC)
	return skinMask


