#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:23:59 2022

@author: jcami
"""
import cv2
import argparse
import math
#from graphics import *
# Importing Image and ImageDraw from PIL
#from PIL import Image, ImageDraw

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
wd=0

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
    global refPt, cropping, wd
    print ("In Cropping")
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
	# check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
        refPt.append((x, y))
        cropping = False
		# draw a rectangle around the region of interest
        if wd == 0:
            cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
            cv2.imshow("image", image)
        elif wd == 1:
            cv2.rectangle(roi, refPt[0], refPt[1], (255, 255, 0), 2)
            cv2.imshow("ROI",roi)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
# load the image, clone it, and setup the mouse callback function
print(args)
print(args["image"])
image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()
	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break
# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
    print (refPt)
    TDx=refPt[1][0]-refPt[0][0]
    TDy=refPt[1][1]-refPt[0][1]
    TDarea=TDx*TDy
    if refPt[0][1]>refPt[1][1] or refPt[0][0]>refPt[1][0]:
        print("Incorrect selection")
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        exit()
    
    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]] 
    cv2.imshow("ROI", roi) 
    cv2.waitKey(0)
    ref_toDo=refPt
    print(ref_toDo)
    wd=1
    cv2.namedWindow("ROI")
    print("mousecallback1")
    cv2.setMouseCallback("ROI", click_and_crop)
    print("mousecallback2")
    cv2.waitKey(0)
    while True:
        cv2.imshow("ROI",roi)
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            croi = roi.copy()
    	# if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
            
    if len(refPt) == 2:
        cv2.rectangle(clone,ref_toDo[0],ref_toDo[1], (0,255,0),2)
        pt1=[(refPt[0][0]+ref_toDo[0][0],refPt[0][1]+ref_toDo[0][1])]
        pt1.append((refPt[1][0]+ref_toDo[0][0],refPt[1][1]+ref_toDo[0][1]))  
        print(ref_toDo[0],ref_toDo[1])
        print(pt1)
        print(pt1[0],pt1[1])
        WDx=pt1[1][0]-pt1[0][0]
        WDy=pt1[1][1]-pt1[0][1]
        WDarea=WDx*WDy
        cv2.rectangle(clone,pt1[0],pt1[1],(255, 0, 0),2)
        Wcomplete=WDarea/TDarea
        text = str(round(Wcomplete*100)) + '% Complete'
        org = (pt1[0][0],pt1[0][1])
        fontFace=cv2.FORMATTER_FMT_MATLAB
        fontScale=8
        color=(10,155,90)
        cv2.putText(clone, text, org, fontFace, fontScale, color, 10, 2)
        print (Wcomplete*100)
        cv2.imshow("Work Done",clone)
    cv2.waitKey(0)





# close all open windows
cv2.destroyAllWindows()


"""
win = GraphWin("Draw a Triangle")
message = Text(Point(5, 0.5), "Click on three points") 
message.draw(win)

# Get and draw three vertices of triangle p1 = win.getMouse()

p1.draw(win)

p2.draw(win)

p3.draw(win)



  
# Opening the image to be used
img = Image.open('img_path.png')
  
# Creating a Draw object
draw = ImageDraw.Draw(img)
  
# Drawing a green rectangle
# in the middle of the image
draw.rectangle(xy = (50, 50, 150, 150),
               fill = (0, 127, 0),
               outline = (255, 255, 255),
               width = 5)
  
# Method to display the modified image
img.show()

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
"""