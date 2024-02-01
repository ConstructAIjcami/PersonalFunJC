import cv2
import numpy as np
image_path ="data/images/brickwall4.jpg"
img = cv2.imread(image_path)
# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# define range of gray color in HSV
lower_gray = np.array([0,0,0])
upper_gray = np.array([255,10,255])
# Threshold the HSV image to get only gray colors
mask = cv2.inRange(hsv, lower_gray, upper_gray)
# Bitwise-AND mask and original image
res = cv2.bitwise_and(img,img, mask= mask)
img = [img.shape[0], img.shape[1]]

lower_gray_locations = []
upper_gray_locations = []

for x_pixel in range(img[0]):
    for y_pixel in range(img[1]):
        sample = res[x_pixel, y_pixel]
        if (lower_gray + np.array([10, 10, 10]) > np.array(sample)).all():
          lower_gray_locations.append( (x_pixel, y_pixel) ) 
          print("Dry mortar")
        if np.logical_and((upper_gray - np.array([10, 10, 10]) > np.array(sample)).all(), (np.array(sample) < upper_gray + np.array([10, 10, 10])).all()):
          print("Wet mortar")
          upper_gray_locations.append( (x_pixel, y_pixel) )
cv2.imwrite("output.png",res)