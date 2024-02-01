#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:22:36 2022

@author: jcami
"""

import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt



while True:



    image = cv2.imread("HVACPlan.jpg")
    cv2.imshow("orig",image)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image2=cv2.inRange(gray,0,115)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    cv2.imshow("image2",image2)
    ret, img_masked = cv2.threshold(blurred,115,255,cv2.THRESH_BINARY)
    minLineLength = 100
    edge2 = cv2.Canny(img_masked,50,150,apertureSize = 3)
    line2 = cv2.HoughLinesP(image=edge2,rho=1,theta=np.pi/180, threshold=300,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)
    xes=[]
    yes=[]
    for index in range(len(line2)):
        xes.append(line2[index][0][0])
        xes.append(line2[index][0][2])
        yes.append(line2[index][0][1])
        yes.append(line2[index][0][3])

    box1=(min(xes),min(yes))
    box2=(max(xes),max(yes))
    color=(255,255,255)
    #cv2.rectangle(image2, box1, box2, color, 2)
    cv2.rectangle(image, box1,box2, color)
    cv2.imshow("orig",image)
    print("got here")

    cv2.imshow("mask",img_masked)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("d"):
        break