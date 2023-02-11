import numpy as np
import cv2 as cv

img = cv.imread('telea.jpg')
mask = cv.imread('1frame_mask.jpg',0)
telea = cv.inpaint(img,mask,3,cv.INPAINT_TELEA)
ns = cv.inpaint(img,mask,3,cv.INPAINT_NS)
cv.imwrite('telea2.jpg',telea)
#cv.imwrite('ns.jpg',ns)
cv.waitKey(0)
cv.destroyAllWindows()