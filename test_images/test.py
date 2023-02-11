import cv2
import numpy as np

# Read images : src image will be cloned into dst
dst = cv2.imread("background.jpg")
obj= cv2.imread("foreground.jpg")

# Create an all white mask
#mask = 255 * np.ones(obj.shape, obj.dtype)
mask = 255 * np.ones_like(obj)
print(mask)

# The location of the center of the src in the dst
width, height, channels = dst.shape
center = (int(height/2), int(width/2))

# Seamlessly clone src into dst and put the results in output
#normal_clone = cv2.seamlessClone(obj, dst, mask, center, cv2.NORMAL_CLONE)
mixed_clone = cv2.seamlessClone(obj, dst, mask, center, cv2.MIXED_CLONE)

# Write results
#cv2.imwrite("a_output.jpg", normal_clone)
cv2.imwrite("b_output.jpg", mixed_clone)